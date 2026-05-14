from dataclasses import dataclass
from numbers import Real
import os
from typing import Any

import numpy as np
import torch
import wandb
from clearml import Task


@dataclass
class LoggedMedia:
    local_path: str
    max_history: int = -1


def to_builtin_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_builtin_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin_config(v) for v in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class WandbWriter:
    def __init__(self, run):
        self.run = run
        self.run.define_metric("*", step_metric="train/step")

    def log(self, writer_dict: dict[str, Any], step: int | None = None) -> None:
        if step is None:
            step = writer_dict.get("train/step")
        if torch.is_tensor(step):
            step = step.item()
        if isinstance(step, np.generic):
            step = step.item()
        payload = {}
        for key, value in writer_dict.items():
            if key == "train/step":
                continue
            if isinstance(value, LoggedMedia):
                format = os.path.splitext(value.local_path)[1].lstrip(".") or None
                payload[key] = wandb.Video(value.local_path, format=format)
            else:
                payload[key] = value
        if step is not None:
            payload["train/step"] = step
        if payload:
            self.run.log(payload)

    def finish(self, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)


class ClearMLWriter:
    def __init__(self, task: Task):
        self.task = task
        self.logger = task.get_logger()
        self.logger.set_default_debug_sample_history(-1)

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        if "/" in key:
            return key.split("/", 1)
        return "metrics", key

    @staticmethod
    def _to_scalar(value: Any) -> float | int | None:
        if isinstance(value, LoggedMedia):
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, Real):
            return value
        return None

    def log(self, writer_dict: dict[str, Any], step: int | None = None) -> None:
        if step is None:
            step = writer_dict.get("train/step")
        if torch.is_tensor(step):
            step = step.item()
        if isinstance(step, np.generic):
            step = step.item()
        iteration = int(step) if step is not None else 0
        for key, value in writer_dict.items():
            if key == "train/step":
                continue
            title, series = self._split_key(key)
            if isinstance(value, LoggedMedia):
                self.logger.report_media(
                    title=title,
                    series=series,
                    iteration=iteration,
                    local_path=value.local_path,
                    max_history=value.max_history,
                )
                continue
            scalar = self._to_scalar(value)
            if scalar is not None:
                self.logger.report_scalar(
                    title=title,
                    series=series,
                    value=scalar,
                    iteration=iteration,
                )

    def finish(self, exit_code: int = 0) -> None:
        self.task.flush(wait_for_uploads=True)
        self.task.close()


def init_writer(persistence_config, output_dir: str, cfg):
    backend = persistence_config.tracker_backend.lower()
    if backend == "wandb":
        run = wandb.init(
            project=persistence_config.proj_name,
            name=persistence_config.exp_name,
            dir=output_dir,
            config=to_builtin_config(cfg),
            id=persistence_config.exp_name,
            resume="auto",
        )
        return WandbWriter(run)
    if backend == "clearml":
        task = Task.init(
            project_name=persistence_config.proj_name,
            task_name=persistence_config.exp_name,
            reuse_last_task_id=True,
            continue_last_task=False,
            output_uri=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
        )
        task.connect(to_builtin_config(cfg), name="config")
        return ClearMLWriter(task)
    raise ValueError(f"Unsupported tracker backend: {persistence_config.tracker_backend}")
