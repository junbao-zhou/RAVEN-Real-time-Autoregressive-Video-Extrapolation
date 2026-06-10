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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

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
from project.utils.tracker import LoggedMedia
from project.utils.dataclass import Dataclass
from project.utils.file_io import maybe_download
from project.utils.misc import TEXT_SUFFIXES, save_video
from project.utils.random import RandomState

logger = logging.getLogger()

ASYNC_FUTS = defaultdict(lambda: None)

if TYPE_CHECKING:
    from project.meta_models import BaseMetaModel


@dataclass
class GenerateT2VInferConfig(Dataclass):
    meta_cfg: CfgNode = field(default_factory=CfgNode)  # flexibility for meta model specific configs
    sampling_timestep: CfgNode = field(default=None)
    sampler: CfgNode = field(default=None)
    num_prompts: int = field(default=-1)  # total number of samples to generate, -1 for all
    log_prompts: int = field(default=-1)
    num_samples_per_prompt: int = field(default=1)
    positive_prompt: str = field(default=None)
    negative_prompt: str = field(default=None)
    num_frames: int = field(default=81)
    height: int = field(default=480)
    width: int = field(default=832)
    batch_size: int = field(default=4)
    seed: int = field(default=0)
    shift_seed: bool = field(default=False)
    save_on_every: bool = field(default=False)
    nrow: int = field(default=4)  # number of video samples per row when saving video grid
    fps: int = field(default=16)
    crf: int = field(default=18)
    reward_model_name: str = field(default="reward_model")


@dataclass
class GenerateT2VEngineConfig(Dataclass):
    schedule: CfgNode = field(default=None)
    inference: CfgNode = field(default=None)
    resume: bool = field(default=False)  # whether to resume from existing samples in output dir, only for run not generate


class GenerateT2V(BaseEngine):
    backbone: nn.Module
    schedule: BaseSchedule

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        self.engine_config = GenerateT2VEngineConfig(**self.config["_config"])
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
        infer_config: GenerateT2VInferConfig,
        local_dir: str,
        remote_dir: str = None,
        save_fn: Callable[[int, str], str] = lambda idx, prompt: f"{idx:04d}-{prompt[:50].replace(' ', '_')}.mp4",  # convert index and prompt to filename for saving
        resume: bool = False,
        yield_inputs: bool = False,
    ):  # main inference loop yielding results, called by both run and generate
        timestep_cls = TIMESTEP_REGISTRY.get(infer_config.sampling_timestep["_class_name"])
        sampling_timesteps = timestep_cls(**infer_config.sampling_timestep["_config"])
        sampler_cls = SAMPLER_REGISTRY.get(infer_config.sampler["_class_name"])
        sampler = sampler_cls(schedule=schedule, **infer_config.sampler["_config"])

        positive_prompt = maybe_download(infer_config.positive_prompt)
        negative_prompt = maybe_download(infer_config.negative_prompt)

        indices, positive_prompts, negative_prompts = list(), list(), list()

        # indices & positive prompts
        if positive_prompt.endswith(TEXT_SUFFIXES):  # prompt list file
            with open(positive_prompt, "r") as f:
                for idx, line in enumerate(f.readlines()):
                    prompt = line.strip()
                    positive_prompts.append(prompt)
                    indices.append(idx)
        else:  # single prompt
            positive_prompts.append(positive_prompt)
            indices.append(0)

        # negative prompts
        if negative_prompt.endswith(TEXT_SUFFIXES):  # prompt list file
            with open(negative_prompt, "r") as f:
                for line in f.readlines():
                    prompt = line.strip()
                    negative_prompts.append(prompt)
        else:
            negative_prompts = [negative_prompt] * len(indices)

        assert len(indices) == len(positive_prompts) == len(negative_prompts), \
            "lengths of indices, positive_prompts, negative_prompts should be the same"
        if infer_config.num_prompts > 0 and infer_config.num_prompts < len(indices):
            if infer_config.num_prompts == 1:
                selected_positions = [len(indices) // 2]
            else:
                selected_positions = [
                    round(i * (len(indices) - 1) / (infer_config.num_prompts - 1))
                    for i in range(infer_config.num_prompts)
                ]
            indices = [indices[i] for i in selected_positions]
            positive_prompts = [positive_prompts[i] for i in selected_positions]
            negative_prompts = [negative_prompts[i] for i in selected_positions]

        # skip existing samples
        all_items = list()
        for idx, positive_prompt, negative_prompt in zip(indices, positive_prompts, negative_prompts):
            if resume:
                if infer_config.save_on_every:
                    filename = save_fn(idx, positive_prompt)
                    local_paths = [
                        os.path.join(
                            local_dir,
                            f"idx_{sample_idx:04d}",
                            filename
                        )
                        for sample_idx in range(infer_config.num_samples_per_prompt)
                    ]
                    if all(os.path.exists(local_path) for local_path in local_paths):
                        logger.info(f"Skip existing sample {idx} in save_on_every folders")
                        continue
                elif os.path.exists(os.path.join(local_dir, save_fn(idx, positive_prompt))):
                    logger.info(f"Skip existing sample {idx} at {os.path.join(local_dir, save_fn(idx, positive_prompt))}")
                    continue
            all_items.append((idx, positive_prompt, negative_prompt))

        sp_size = 1
        sp_rank = 0

        # padding and make sure ranks have the same number of prompts to generate
        group_idx = comm.get_rank()
        num_groups = comm.get_world_size()
        num_prompts = len(all_items)
        if num_prompts == 0:
            logger.info("No prompts need generation after filtering existing samples.")
            yield 0
            return
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

            indices, positive_prompts, negative_prompts = zip(*batch_items)
            seeds = [(infer_config.seed + index) if infer_config.shift_seed else infer_config.seed for index in indices]
            rngs = [RandomState(seed=seed) for seed in seeds]

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
            batch_videos = []
            sample_count = int(pos_inputs.batch_size.item())
            for sample_start in range(0, sample_count, infer_config.batch_size):
                sample_end = min(sample_start + infer_config.batch_size, sample_count)
                sample_mask = torch.zeros(sample_count, dtype=torch.bool, device=pos_inputs.batch_size.device)
                sample_mask[sample_start:sample_end] = True
                batch_videos.append(meta_model.infer(
                    model=model,
                    pos_inputs=meta_model.mask_inputs(pos_inputs, sample_mask),
                    neg_inputs=meta_model.mask_inputs(neg_inputs, sample_mask),
                    rng=rngs[sample_start:sample_end],
                    sampling_timesteps=sampling_timesteps,
                    schedule=schedule,
                    sampler=sampler
                ))
            videos = meta_model.merge_tensors(batch_videos)
            yield dict(videos=videos, pos_inputs=pos_inputs) if yield_inputs else videos

            # save per prompt with audio
            for i in range(bsz):
                if duplicate[i] or sp_rank != 0:  # do not save duplicated prompts
                    continue
                videos_to_save = videos[i*infer_config.num_samples_per_prompt:(i+1)*infer_config.num_samples_per_prompt]

                if infer_config.save_on_every:
                    filename = save_fn(indices[i], positive_prompts[i])
                    for sample_idx in range(infer_config.num_samples_per_prompt):
                        sample_local_dir = os.path.join(local_dir, f"idx_{sample_idx:04d}")
                        sample_remote_dir = os.path.join(remote_dir, f"idx_{sample_idx:04d}") if remote_dir is not None else None
                        os.makedirs(sample_local_dir, exist_ok=True)
                        if sample_remote_dir is not None:
                            fs.mkdir(sample_remote_dir, distributed=True, sync=False)
                        ASYNC_FUTS[f"video_idx{i}_sample{sample_idx}"] = GenerateT2V.save_results_async(
                            fut=ASYNC_FUTS[f"video_idx{i}_sample{sample_idx}"],
                            videos=videos_to_save[sample_idx:sample_idx + 1],
                            local_path=os.path.join(sample_local_dir, filename),
                            remote_dir=sample_remote_dir,
                            fps=infer_config.fps,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1),
                            crf=infer_config.crf
                        )
                else:
                    ASYNC_FUTS[f"video_idx{i}"] = GenerateT2V.save_results_async(
                        fut=ASYNC_FUTS[f"video_idx{i}"],
                        videos=videos_to_save,
                        local_path=os.path.join(local_dir, save_fn(indices[i], positive_prompts[i])),
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
        local_dir = os.path.join(self.output_dir, "t2v_results")
        os.makedirs(local_dir, exist_ok=True)
        remote_dir = os.path.join(self.save_dir, "t2v_results")
        fs.mkdir(remote_dir)

        infer_config = GenerateT2VInferConfig(**self.engine_config.inference)
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
                sstring = f"[Generate {(batch_index + 1):04d}/{num_batches:04d}]"
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

    @staticmethod
    @torch.no_grad()
    def validate(
        meta_model: BaseMetaModel,
        model: nn.Module,
        schedule: BaseSchedule,
        infer_cfg: CfgNode,
        output_dir: str,
        save_dir: str,
    ) -> Dict[str, Any]:  # called during training for visualization
        local_dir = os.path.join(output_dir, "t2v_results")
        os.makedirs(local_dir, exist_ok=True)
        remote_dir = os.path.join(save_dir, "t2v_results")
        fs.mkdir(remote_dir)

        infer_config = GenerateT2VInferConfig(**infer_cfg)
        reward_model = getattr(meta_model, infer_config.reward_model_name, None)
        if reward_model is None and hasattr(meta_model, "models"):
            reward_model = meta_model.models.get(infer_config.reward_model_name, None)
        generator = GenerateT2V.execute(
            meta_model=meta_model,
            model=model,
            schedule=schedule,
            infer_config=infer_config,
            local_dir=local_dir,
            remote_dir=remote_dir,
            yield_inputs=reward_model is not None
        )
        num_prompts = next(generator)  # for logging purpose
        num_log_prompts = num_prompts if infer_config.log_prompts < 0 else min(infer_config.log_prompts, num_prompts)
        num_groups = comm.get_world_size()
        group_idx = comm.get_rank()
        portion_size = (num_prompts - 1) // num_groups + 1

        log_dict = {}
        video_list = []
        reward_list = defaultdict(list)
        sp_rank = 0
        for batch_index, output in enumerate(generator):
            if reward_model is not None:
                videos = output["videos"]
                pos_inputs = output["pos_inputs"]
                reward_output = meta_model.reward_fn(reward_model, pos_inputs, videos)
                if torch.is_tensor(reward_output):
                    reward_output = {"reward": reward_output}
                else:
                    assert isinstance(reward_output, dict), \
                        f"reward_fn should return a tensor or dict, got {type(reward_output)}"
                    reward_output = {
                        key: value
                        for key, value in reward_output.items()
                        if torch.is_tensor(value)
                    }
            else:
                videos = output

            if sp_rank == 0:
                bsz = len(videos) // infer_config.num_samples_per_prompt
                for i in range(bsz):
                    prompt_position = group_idx * portion_size + batch_index * infer_config.batch_size + i
                    if prompt_position >= num_log_prompts or prompt_position >= num_prompts:
                        continue
                    vid = videos[i*infer_config.num_samples_per_prompt:(i+1)*infer_config.num_samples_per_prompt]
                    video_list.append(torch.stack(vid[:1]).cpu().float())
                if reward_model is not None:
                    for key, value in reward_output.items():
                        reward_list[key].append(value.detach().float().cpu())

        if reward_model is not None and sp_rank == 0:
            local_rewards = {
                key: torch.cat(values, dim=0)
                for key, values in reward_list.items()
                if len(values) > 0
            }
            all_rewards = comm.gather_object(local_rewards, dst=0)
            if comm.get_rank() == 0:
                num_samples = num_prompts * infer_config.num_samples_per_prompt
                reward_keys = sorted({key for reward_dict in all_rewards for key in reward_dict.keys()})
                for key in reward_keys:
                    values = [reward_dict[key] for reward_dict in all_rewards if key in reward_dict]
                    if len(values) == 0:
                        continue
                    rewards = torch.cat(values, dim=0)[:num_samples]
                    log_dict[f"rewards/{key}"] = rewards.mean()

        if sp_rank == 0:
            videos = None
            if len(video_list) > 0:
                max_height = max([v.shape[-2] for v in video_list])
                max_width = max([v.shape[-1] for v in video_list])
                for i in range(len(video_list)):
                    v = video_list[i]
                    _, _, _, h, w = v.shape
                    padding_left, padding_right = (max_width - w) // 2, max_width - w - (max_width - w) // 2
                    padding_top, padding_bottom = (max_height - h) // 2, max_height - h - (max_height - h) // 2
                    video_list[i] = F.pad(v, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=-1.0)
                videos = torch.cat(video_list, dim=0)

                cols = infer_config.nrow
                rows = (num_log_prompts - 1) // cols + 1
                MAX_PIXELS = 1920 * 1080 // (cols * rows)
                total_pixels = videos.shape[-2] * videos.shape[-1]
                if total_pixels > MAX_PIXELS:
                    scale_factor = (MAX_PIXELS / total_pixels) ** 0.5
                    new_height = int(videos.shape[-2] * scale_factor)
                    new_width = int(videos.shape[-1] * scale_factor)
                    new_height = new_height - (new_height % 2)
                    new_width = new_width - (new_width % 2)
                    bsz = videos.shape[0]
                    videos = rearrange(videos, "b c t h w -> (b t) c h w")
                    videos = F.interpolate(videos, size=(new_height, new_width), mode="bilinear", align_corners=False)
                    videos = rearrange(videos, "(b t) c h w -> b c t h w", b=bsz)

            all_videos = comm.gather_object(videos, dst=0)
            if comm.get_rank() == 0:
                all_videos = [videos for videos in all_videos if videos is not None]
                all_videos = list(itertools.chain(*all_videos))
                all_videos = all_videos[:num_log_prompts]
                if len(all_videos) > 0:
                    all_videos = torch.stack(all_videos, dim=0)

                    ASYNC_FUTS["videos_all"] = GenerateT2V.save_results_async(
                        fut=ASYNC_FUTS["videos_all"],
                        videos=all_videos,
                        local_path=os.path.join(output_dir, "t2v_results.mp4"),
                        remote_dir=save_dir,
                        fps=infer_config.fps,
                        nrow=infer_config.nrow,
                        normalize=True,
                        value_range=(-1, 1),
                        crf=infer_config.crf
                    )

        comm.barrier()
        if num_log_prompts > 0 and ASYNC_FUTS["videos_all"] is not None:
            ASYNC_FUTS["videos_all"].wait()
            if comm.get_rank() == 0:
                log_dict["media"] = LoggedMedia(os.path.join(output_dir, "t2v_results.mp4"))
        return log_dict
