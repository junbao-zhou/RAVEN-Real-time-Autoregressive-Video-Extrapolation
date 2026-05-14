"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from __future__ import annotations

import gc
import itertools
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from project.diffusion.samplers import SAMPLER_REGISTRY, BaseSampler
from project.diffusion.schedules import SCHEDULE_REGISTRY, BaseSchedule
from project.diffusion.timesteps import TIMESTEP_REGISTRY
from project.engines import BaseEngine
from project.utils import comm, fs
from project.utils.config import CfgNode
from project.utils.dataclass import Dataclass
from project.utils.file_io import maybe_download
from project.utils.misc import TEXT_SUFFIXES, save_video
from project.utils.random import RandomState

logger = logging.getLogger()

ASYNC_FUTS = defaultdict(lambda: None)

if TYPE_CHECKING:
    from project.meta_models import BaseMetaModel


@dataclass
class VBenchT2VInferConfig(Dataclass):
    meta_cfg: CfgNode = field(default_factory=CfgNode)  # flexibility for meta model specific configs
    sampling_timestep: CfgNode = field(default=None)
    sampler: CfgNode = field(default=None)
    num_prompts: int = field(default=-1)  # total number of samples to generate, -1 for all
    num_samples_per_prompt: int = field(default=1)
    positive_prompt: str = field(default=None)
    negative_prompt: str = field(default=None)
    num_frames: int = field(default=81)
    height: int = field(default=480)
    width: int = field(default=832)
    batch_size: int = field(default=4)
    seed: int = field(default=0)
    shift_seed: bool = field(default=False)
    nrow: int = field(default=4)  # number of video samples per row when saving video grid
    fps: int = field(default=16)
    crf: int = field(default=18)


@dataclass
class VBenchT2VEngineConfig(Dataclass):
    schedule: CfgNode = field(default=None)
    inference: CfgNode = field(default=None)
    resume: bool = field(default=False)  # whether to resume from existing samples in output dir, only for run not generate


class VBenchT2V(BaseEngine):
    backbone: nn.Module
    schedule: BaseSchedule

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        self.engine_config = VBenchT2VEngineConfig(**self.config["_config"])
        self.backbone = self.models["backbone"]

        self.configure_diffusion()

    def configure_diffusion(self):
        # schedule
        schedule_cls = SCHEDULE_REGISTRY.get(self.engine_config.schedule["_class_name"])
        self.schedule = schedule_cls(**self.engine_config.schedule["_config"])

    @staticmethod
    def save_results_async(
        fut: torch.Future,
        videos: Union[torch.Tensor, List[torch.Tensor]],
        local_path: str,
        remote_dir: str = None,
        audio_path: str = None,
        **kwargs
    ):
        if fut is not None:
            fut.wait()

        if isinstance(videos, list):
            assert videos[0].ndim == 4, f"each video should be a list of tensors with shape [C, T, H, W], got {videos[0].shape}"
            videos = torch.stack(videos, dim=0)
        videos = videos.cpu().float()
        assert videos.ndim == 5, f"videos should be a tensor with shape [B, C, T, H, W], got {videos.shape}"

        def _fn(fut: torch.Future, videos, local_path, remote_dir, audio_path, **kwargs):
            save_video(videos, local_path, audio_path=audio_path, **kwargs)
            if os.path.exists(local_path) and remote_dir is not None and os.path.abspath(remote_dir) != os.path.abspath(os.path.dirname(local_path)):
                fs.copy(local_path, remote_dir)
            fut.set_result((local_path, remote_dir))

        def _cb(fut: torch.Future):
            local_path, remote_dir = fut.value()
            logger.info(f"Saving video at local path {local_path} and remote dir {remote_dir} async done")

        fut = torch.futures.Future()
        fut.add_done_callback(_cb)
        worker = threading.Thread(target=_fn, args=(fut, videos, local_path, remote_dir, audio_path), kwargs=kwargs)
        worker.start()
        return fut

    @staticmethod
    @torch.no_grad()
    def execute(
        meta_model: BaseMetaModel,
        model: nn.Module,
        schedule: BaseSchedule,
        infer_config: VBenchT2VInferConfig,
        local_dir: str,
        remote_dir: str = None,
        save_fn: Callable[[int], str] = lambda idx: f"{idx:04d}.mp4",  # convert index to filename for saving
        resume: bool = False
    ):  # main inference loop yielding results, called by both run and generate
        timestep_cls = TIMESTEP_REGISTRY.get(infer_config.sampling_timestep["_class_name"])
        sampling_timesteps = timestep_cls(**infer_config.sampling_timestep["_config"])
        sampler_cls = SAMPLER_REGISTRY.get(infer_config.sampler["_class_name"])
        sampler = sampler_cls(schedule=schedule, **infer_config.sampler["_config"])

        with open("assets/vbench_all_dimension.txt", "r") as f:
            vbench_shorts = [line.strip() for line in f.readlines()]

        positive_prompt_path = maybe_download(infer_config.positive_prompt)
        indices, positive_prompts, filenames = list(), list(), list()

        with open(positive_prompt_path, "r") as f:
            for index, (line, short) in enumerate(zip(f.readlines(), vbench_shorts)):
                prompt = line.strip()
                repeat = 25 if index < 75 else 5
                base_index = index * 25 if index < 75 else 25 * 75 + (index - 75) * 5
                for i in range(repeat):
                    filename = f"{short}-{i}.mp4"
                    local_path = os.path.join(local_dir, filename)
                    if os.path.exists(local_path):
                        logger.info(f"Found existing sample at {local_path}, skip generation")
                        continue
                    positive_prompts.append(prompt)
                    indices.append(base_index + i)
                    filenames.append(filename)

        if infer_config.num_prompts >= 0:
            positive_prompts = positive_prompts[:infer_config.num_prompts]
            indices = indices[:infer_config.num_prompts]
            filenames = filenames[:infer_config.num_prompts]

        if infer_config.negative_prompt is None:
            negative_prompt = ""
        elif fs.exists(infer_config.negative_prompt):
            negative_prompt = maybe_download(infer_config.negative_prompt)
            if negative_prompt.endswith(TEXT_SUFFIXES):
                with open(negative_prompt, "r") as f:
                    negative_prompt = f.read().strip()
        else:
            negative_prompt = infer_config.negative_prompt

        all_items = list(zip(indices, positive_prompts, filenames))
        num_prompts = len(all_items)
        if num_prompts == 0:
            logger.info("No VBench prompts need generation after filtering existing samples.")
            yield 0
            return

        sp_size = 1
        sp_rank = 0

        # padding and make sure ranks have the same number of prompts to generate
        group_idx = comm.get_rank()
        num_groups = comm.get_world_size()
        portion_size = (num_prompts - 1) // num_groups + 1
        start = portion_size * group_idx
        end = min(portion_size * (group_idx + 1), num_prompts)
        items = all_items[start:end]
        duplicate = [False for _ in range(len(items))]
        if max(0, end - start) < portion_size:
            items += [all_items[-1]] * (portion_size - max(0, end - start))
            duplicate += [True] * (portion_size - max(0, end - start))

        # start inference
        num_batches = (portion_size - 1) // infer_config.batch_size + 1
        logger.info(f"Start inference on {num_prompts} prompts, from index {start} to {end}, "
                    f"each sp_group has {portion_size} prompts after padding, "
                    f"total {num_batches} batches with batch size {infer_config.batch_size}.")
        yield num_prompts  # indicating initialization done, yield num_prompts for logging purpose

        for batch_index in range(num_batches):
            start_index = batch_index * infer_config.batch_size
            end_index = min((batch_index + 1) * infer_config.batch_size, portion_size)
            batch_items = items[start_index:end_index]
            bsz = len(batch_items)

            indices, positive_prompts, batch_filenames = zip(*batch_items)
            seeds = [(infer_config.seed + index) if infer_config.shift_seed else infer_config.seed for index in indices]
            rngs = [RandomState(seed=seed) for seed in seeds]
            negative_prompts = [negative_prompt] * bsz

            batch = dict(
                prompts=positive_prompts,
                neg_prompts=negative_prompts,
                infer_config=infer_config
            )

            pos_inputs, neg_inputs = meta_model.prepare_inference_inputs(
                batch=batch,
                infer_config=infer_config,
                rngs=rngs
            )
            videos = meta_model.infer(
                model=model,
                pos_inputs=pos_inputs,
                neg_inputs=neg_inputs,
                rng=rngs,
                sampling_timesteps=sampling_timesteps,
                schedule=schedule,
                sampler=sampler
            )  # iterable (List of Batched) [C, T, H, W] pixel tensors in [-1, 1]
            yield videos

            # save per prompt with audio
            for i in range(bsz):
                if duplicate[i] or sp_rank != 0:  # do not save duplicated prompts
                    continue
                videos_to_save = videos[i*infer_config.num_samples_per_prompt:(i+1)*infer_config.num_samples_per_prompt]

                ASYNC_FUTS[f"video_idx{i}"] = VBenchT2V.save_results_async(
                    fut=ASYNC_FUTS[f"video_idx{i}"],
                    videos=videos_to_save,
                    local_path=os.path.join(local_dir, batch_filenames[i]),
                    remote_dir=remote_dir,
                    fps=infer_config.fps,
                    nrow=infer_config.nrow,
                    normalize=True,
                    value_range=(-1, 1),
                    crf=infer_config.crf
                )
            comm.barrier()

    @torch.no_grad()
    def run(self):  # called by engine run
        local_dir = os.path.join(self.output_dir, "videos")
        os.makedirs(local_dir, exist_ok=True)
        remote_dir = os.path.join(self.save_dir, "videos")
        fs.mkdir(remote_dir)

        infer_config = VBenchT2VInferConfig(**self.engine_config.inference)
        generator = self.execute(
            meta_model=self,
            model=self.backbone,
            schedule=self.schedule,
            infer_config=infer_config,
            local_dir=local_dir,
            remote_dir=remote_dir,
            resume=self.engine_config.resume
        )

        num_prompts = next(generator)  # for logging purpose
        num_groups = comm.get_world_size()
        portion_size = (num_prompts - 1) // num_groups + 1
        num_batches = (portion_size - 1) // infer_config.batch_size + 1

        self.mfu_start.record()
        start_time = time.perf_counter()
        for batch_index, _ in enumerate(generator):
            self.mfu_end.record()
            mfu_dict = self.get_mfu(show=batch_index < 1)
            batch_time = time.perf_counter() - start_time
            self.average_meter.update({
                "batch_time": batch_time,  # in seconds
                **mfu_dict
            })

            if (batch_index + 1) % self.persistence_config.log_interval == 0 or (batch_index + 1) == num_batches:
                self.average_meter.sync()
                sstring = f"[Batches {(batch_index + 1):04d}/{num_batches:04d}]"
                for k, v in self.average_meter.items():
                    sstring += f" | {k}: {v:.4f}"

                # memory info
                memory_free, memory_total = torch.cuda.mem_get_info()
                memory_used = (memory_total - memory_free) / (1024 ** 3)  # in GB
                num_alloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]
                sstring += f" | memory_used: {memory_used:.2f} GB"
                sstring += f" | num_alloc_retries: {num_alloc_retries}"

                logger.info(sstring)
                self.average_meter.reset()
                gc.collect()

            start_time = time.perf_counter()
            self.mfu_start.record()

        for fut in ASYNC_FUTS.values():
            if fut is not None:
                fut.wait()
        logger.info(f"All videos saved at local dir {local_dir} and remote dir {remote_dir}.")
