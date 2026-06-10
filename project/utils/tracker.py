from dataclasses import dataclass
import json
from numbers import Real
import os
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


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


class TensorBoardWriter:
    def __init__(self, log_dir: str, cfg):
        self.writer = SummaryWriter(log_dir=log_dir)
        config_text = json.dumps(
            to_builtin_config(cfg),
            indent=2,
            ensure_ascii=False,
        )
        self.writer.add_text("config", f"```json\n{config_text}\n```", 0)

    @staticmethod
    def _to_step(step: Any) -> int | None:
        if torch.is_tensor(step):
            step = step.item()
        if isinstance(step, np.generic):
            step = step.item()
        if step is None:
            return None
        return int(step)

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

    @staticmethod
    def _to_histogram(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return value.detach().float().cpu()
        if isinstance(value, np.ndarray):
            if value.size == 0 or not np.issubdtype(value.dtype, np.number):
                return None
            return torch.from_numpy(value).float()
        return None

    def log(self, writer_dict: dict[str, Any], step: int | None = None) -> None:
        if step is None:
            step = writer_dict.get("train/step")
        global_step = self._to_step(step)
        for key, value in writer_dict.items():
            if key == "train/step":
                continue
            if isinstance(value, LoggedMedia):
                self._log_media(key, value, global_step)
                continue
            scalar = self._to_scalar(value)
            if scalar is not None:
                self.writer.add_scalar(key, scalar, global_step)
                continue
            histogram = self._to_histogram(value)
            if histogram is not None:
                self.writer.add_histogram(key, histogram, global_step)
                continue
            if isinstance(value, str):
                self.writer.add_text(key, value, global_step)
        self.writer.flush()

    def _log_media(self, key: str, value: LoggedMedia, step: int | None) -> None:
        suffix = os.path.splitext(value.local_path)[1].lower()
        try:
            if suffix in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}:
                from torchvision.io import read_image

                image = read_image(value.local_path)
                self.writer.add_image(key, image, step)
                return
            if suffix in {".avi", ".gif", ".mkv", ".mov", ".mp4", ".webm"}:
                from torchvision.io import read_video

                video, _, info = read_video(
                    value.local_path,
                    pts_unit="sec",
                    output_format="TCHW",
                )
                if video.numel() == 0:
                    self.writer.add_text(f"{key}/local_path", value.local_path, step)
                    return
                frames_per_second = int(round(info.get("video_fps", 4)))
                self.writer.add_video(
                    key,
                    video.unsqueeze(0),
                    step,
                    fps=max(frames_per_second, 1),
                )
                return
        except Exception as error:
            self.writer.add_text(f"{key}/local_path", value.local_path, step)
            self.writer.add_text(f"{key}/media_error", str(error), step)
            return
        self.writer.add_text(f"{key}/local_path", value.local_path, step)

    def finish(self, exit_code: int = 0) -> None:
        self.writer.flush()
        self.writer.close()


def init_writer(persistence_config, output_dir: str, cfg):
    backend = persistence_config.tracker_backend.lower()
    if backend not in {"tensorboard", "tb"}:
        raise ValueError(f"Only TensorBoard tracker backend is supported: {persistence_config.tracker_backend}")
    log_dir = os.path.join(output_dir, "tensorboard")
    return TensorBoardWriter(log_dir=log_dir, cfg=cfg)
