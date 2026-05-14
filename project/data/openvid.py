"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import decord
import torch
import torch.nn.functional as F

from project.data.parquet import ParquetDataset
from project.data.utils import prepare_flex_attention_mask
from project.utils.dataclass import Dataclass


@dataclass
class CausalOpenVidDatasetConfig(Dataclass):
    # parquet settings
    paths: List[str] = field(default_factory=list)
    weights: Optional[List[float]] = field(default=None)
    seed: int = field(default=42)
    verbose: bool = field(default=False)
    columns: Optional[List[str]] = field(default=None)
    total_seqlen: Optional[int] = field(default=None)
    max_seqlen: Optional[int] = field(default=None)
    max_seqlen_per_sample: Optional[int] = field(default=None)
    max_retries: int = field(default=5)
    # data settings
    native_fps: int = field(default=48)
    native_height: int = field(default=720)
    native_width: int = field(default=1280)
    fps: int = field(default=16)
    height: int = field(default=480)
    width: int = field(default=832)
    num_frames: int = field(default=81)
    extended_prompt_dir: Optional[str] = field(default=None)
    # causal settings
    chunk_size: Union[int, List[int]] = field(default=3)
    independent_first_chunk: Union[int, List[int]] = field(default=4)
    sink: Union[int, List[int]] = field(default=0)
    window_size: Union[int, List[int], None] = field(default=None)


class CausalOpenVidDataset(ParquetDataset):
    def __init__(
        self,
        collate_fn,
        state_dict,
        # arguments from meta model config
        separated_first_frame: bool = True,
        patch_size: List[int] = [1, 2, 2],
        vae_stride: List[int] = [4, 8, 8],
        # dataset config
        **kwargs
    ):
        self.separated_first_frame = separated_first_frame
        self.patch_size = patch_size
        self.vae_stride = vae_stride

        self.config = CausalOpenVidDatasetConfig(**kwargs)
        self.extended_prompt_dir = None
        self.extended_prompts = {}
        if self.config.extended_prompt_dir is not None:
            self.extended_prompt_dir = Path(self.config.extended_prompt_dir)
            if not self.extended_prompt_dir.exists():
                raise ValueError(f"extended_prompt_dir not found: {self.extended_prompt_dir}")
        super().__init__(
            paths=self.config.paths,
            seed=self.config.seed,
            verbose=self.config.verbose,
            columns=self.config.columns,
            total_seqlen=self.config.total_seqlen,
            max_seqlen=self.config.max_seqlen,
            max_seqlen_per_sample=self.config.max_seqlen_per_sample,
            max_retries=self.config.max_retries,
            weights=self.config.weights,
            collate_fn=collate_fn,
            state_dict=state_dict
        )

        # process causal packing settings
        def to_list(x):
            return x if isinstance(x, list) else [x]

        self.chunk_size = to_list(self.config.chunk_size)
        self.independent_first_chunk = to_list(self.config.independent_first_chunk)
        self.sink = to_list(self.config.sink)
        self.window_size = to_list(self.config.window_size)
        assert len(self.chunk_size) == len(self.independent_first_chunk) == len(self.sink) == len(self.window_size), \
            "chunk_size, independent_first_chunk, sink and window_size should have the same length."
        self.independent_first_chunk = [x if x is not None else cs for x, cs in zip(self.independent_first_chunk, self.chunk_size)]

        assert self.config.native_fps % self.config.fps == 0, f"fps {self.config.fps} must divide native fps {self.config.native_fps}"
        self.native_aspect_ratio = self.config.native_width / self.config.native_height
        self.aspect_ratio = self.config.width / self.config.height
        if self.aspect_ratio > self.native_aspect_ratio:  # crop height
            self.crop_height = int(self.config.native_width / self.aspect_ratio)
            self.crop_width = self.config.native_width
        else:  # crop width
            self.crop_height = self.config.native_height
            self.crop_width = int(self.config.native_height * self.aspect_ratio)

    def process_data(self, data: dict, rng: random.Random) -> dict:
        vr = decord.VideoReader(io.BytesIO(data["raw_video"]))
        info = torch.load(io.BytesIO(data["info"]), map_location="cpu", weights_only=False)
        prompt = info["caption"]
        if self.extended_prompt_dir is not None:
            source_path = data.get("source_path")
            source_row_index = data.get("source_row_index")
            if source_path is not None and source_row_index is not None:
                prompt_path = self.extended_prompt_dir / f"{Path(source_path).stem}.txt"
                prompts = self.extended_prompts.get(prompt_path)
                if prompts is None and prompt_path.exists():
                    with prompt_path.open("r", encoding="utf-8") as f:
                        prompts = [line.rstrip("\n") for line in f]
                    self.extended_prompts[prompt_path] = prompts
                if prompts is not None and 0 <= source_row_index < len(prompts):
                    extended_prompt = prompts[source_row_index].strip()
                    if extended_prompt:
                        prompt = extended_prompt

        random_index = rng.randint(0, len(self.chunk_size) - 1)
        chunk_size = self.chunk_size[random_index]
        independent_first_chunk = self.independent_first_chunk[random_index]
        sink = self.sink[random_index]
        window_size = self.window_size[random_index]

        total_frames = len(vr)
        stride = self.config.native_fps // self.config.fps
        num_frames_after_downsample = (total_frames - 1) // stride + 1
        if num_frames_after_downsample < self.config.num_frames:
            return None
        max_start = num_frames_after_downsample - self.config.num_frames
        random_start = rng.randint(0, max_start)
        frame_indices = [
            random_start * stride + j * stride
            for j in range(self.config.num_frames)
        ]
        frames_array = vr.get_batch(frame_indices).asnumpy()  # [T, H, W, C]

        # center crop + resize
        h_start = (self.config.native_height - self.crop_height) // 2
        w_start = (self.config.native_width - self.crop_width) // 2
        frames_array = frames_array[:, h_start:h_start + self.crop_height, w_start:w_start + self.crop_width]
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
        if frames_tensor.shape[2] != self.config.height or frames_tensor.shape[3] != self.config.width:
            frames_tensor = F.interpolate(
                frames_tensor,
                size=(self.config.height, self.config.width),
                mode="bilinear",
                align_corners=False
            )
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]

        # compute seqlen
        separated_first_frame = self.separated_first_frame
        lat_t = (frames_tensor.shape[1] - separated_first_frame) // self.vae_stride[0] + separated_first_frame
        lat_h = frames_tensor.shape[2] // self.vae_stride[1]
        lat_w = frames_tensor.shape[3] // self.vae_stride[2]
        patch_t, patch_h, patch_w = self.patch_size
        seqlen_per_frame = (lat_h // patch_h) * (lat_w // patch_w)
        if separated_first_frame:
            # The first latent frame is processed individually, and the remaining latent frames are grouped by patch_t.
            remaining_latent_frames = lat_t - 1
            temporal_tokens = 1 + remaining_latent_frames // patch_t
        else:
            temporal_tokens = lat_t // patch_t
        seqlen = temporal_tokens * seqlen_per_frame

        # 5) causal packing
        seqlen_per_chunk = chunk_size * seqlen_per_frame
        seqlen_first_chunk = seqlen_per_frame * independent_first_chunk
        num_chunks = 1 + (lat_t - independent_first_chunk) // chunk_size

        position_id, latent_index, latent_seqlen = list(), list(), list()
        noisy_latent_relative_index, noisy_latent_seqlen = list(), list()
        split_lens, attn_modes, frame_shifts = list(), list(), list()
        sample_len = 0
        curr, curr_noisy, curr_rope, curr_frame = 0, 0, 0, 0

        for j in range(num_chunks):
            if j == 0 and independent_first_chunk:  # first chunk
                # noisy latent
                position_id.extend([curr_rope] * seqlen_first_chunk)
                latent_index.extend(list(range(curr, curr + seqlen_first_chunk)))
                noisy_latent_relative_index.extend(list(range(curr_noisy, curr_noisy + seqlen_first_chunk)))
                curr += seqlen_first_chunk
                curr_noisy += seqlen_first_chunk
                latent_seqlen.append(seqlen_first_chunk)
                noisy_latent_seqlen.append(seqlen_first_chunk)
                split_lens.append(seqlen_first_chunk)
                attn_modes.append("noise")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_first_chunk

                # clean latent
                position_id.extend([curr_rope] * seqlen_first_chunk)
                latent_index.extend(list(range(curr, curr + seqlen_first_chunk)))
                curr += seqlen_first_chunk
                curr_noisy += seqlen_first_chunk
                latent_seqlen.append(seqlen_first_chunk)
                split_lens.append(seqlen_first_chunk)
                attn_modes.append("full")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_first_chunk

                curr_rope += 1
                curr_frame += independent_first_chunk

            elif j == num_chunks - 1:  # last chunk: noisy latent only
                # noisy latent
                position_id.extend([curr_rope] * seqlen_per_chunk)
                latent_index.extend(list(range(curr, curr + seqlen_per_chunk)))
                noisy_latent_relative_index.extend(list(range(curr_noisy, curr_noisy + seqlen_per_chunk)))
                curr += seqlen_per_chunk
                curr_noisy += seqlen_per_chunk
                latent_seqlen.append(seqlen_per_chunk)
                noisy_latent_seqlen.append(seqlen_per_chunk)
                split_lens.append(seqlen_per_chunk)
                attn_modes.append("noise")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_per_chunk

                curr_rope += 1
                curr_frame += chunk_size

            else:  # middle chunks: noisy latent + clean latent
                # noisy latent
                position_id.extend([curr_rope] * seqlen_per_chunk)
                latent_index.extend(list(range(curr, curr + seqlen_per_chunk)))
                noisy_latent_relative_index.extend(list(range(curr_noisy, curr_noisy + seqlen_per_chunk)))
                curr += seqlen_per_chunk
                curr_noisy += seqlen_per_chunk
                latent_seqlen.append(seqlen_per_chunk)
                noisy_latent_seqlen.append(seqlen_per_chunk)
                split_lens.append(seqlen_per_chunk)
                attn_modes.append("noise")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_per_chunk

                # clean latent
                position_id.extend([curr_rope] * seqlen_per_chunk)
                latent_index.extend(list(range(curr, curr + seqlen_per_chunk)))
                curr += seqlen_per_chunk
                curr_noisy += seqlen_per_chunk
                latent_seqlen.append(seqlen_per_chunk)
                split_lens.append(seqlen_per_chunk)
                attn_modes.append("full")
                frame_shifts.append(curr_frame)
                sample_len += seqlen_per_chunk

                curr_rope += 1
                curr_frame += chunk_size

        position_ids = torch.tensor(position_id, dtype=torch.int32)
        latent_indexes = torch.tensor(latent_index, dtype=torch.int32)
        latent_seqlens = torch.tensor(latent_seqlen, dtype=torch.int32)
        noisy_latent_relative_indexes = torch.tensor(noisy_latent_relative_index, dtype=torch.int32)
        noisy_latent_seqlens = torch.tensor(noisy_latent_seqlen, dtype=torch.int32)
        q_ranges, k_ranges, attn_type_map, attn_workloads = prepare_flex_attention_mask(
            split_lens, attn_modes, sink=sink, window_size=window_size
        )

        return dict(
            # general
            videos=frames_tensor,
            prompts=prompt,
            seqlens=seqlen,
            # packed
            position_ids=position_ids,
            latent_indexes=latent_indexes,
            latent_seqlens=latent_seqlens,
            noisy_latent_relative_indexes=noisy_latent_relative_indexes,
            noisy_latent_seqlens=noisy_latent_seqlens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            attn_workloads=attn_workloads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            sample_lens=sample_len,
            frame_shifts=frame_shifts,
            # causal
            chunk_sizes=chunk_size,
            independent_first_chunks=independent_first_chunk,
            sinks=sink,
            window_sizes=window_size
        )


class CausalOpenVidDataset_DF(CausalOpenVidDataset):
    def process_data(self, data: dict, rng: random.Random) -> dict:
        sample = super().process_data(data, rng)
        if sample is None:
            return None

        frames_tensor = sample["videos"]
        chunk_size = sample["chunk_sizes"]
        independent_first_chunk = sample["independent_first_chunks"]
        sink = sample["sinks"]
        window_size = sample["window_sizes"]

        separated_first_frame = self.separated_first_frame
        lat_t = (frames_tensor.shape[1] - separated_first_frame) // self.vae_stride[0] + separated_first_frame
        lat_h = frames_tensor.shape[2] // self.vae_stride[1]
        lat_w = frames_tensor.shape[3] // self.vae_stride[2]
        patch_t, patch_h, patch_w = self.patch_size
        seqlen_per_frame = (lat_h // patch_h) * (lat_w // patch_w)
        if separated_first_frame:
            remaining_latent_frames = lat_t - 1
            temporal_tokens = 1 + remaining_latent_frames // patch_t
        else:
            temporal_tokens = lat_t // patch_t
        seqlen = temporal_tokens * seqlen_per_frame

        seqlen_per_chunk = chunk_size * seqlen_per_frame
        seqlen_first_chunk = seqlen_per_frame * independent_first_chunk
        num_chunks = 1 + (lat_t - independent_first_chunk) // chunk_size

        position_id, latent_index, latent_seqlen = list(), list(), list()
        noisy_latent_relative_index, noisy_latent_seqlen = list(), list()
        split_lens, attn_modes, frame_shifts = list(), list(), list()
        sample_len = 0
        curr, curr_noisy, curr_rope, curr_frame = 0, 0, 0, 0

        for j in range(num_chunks):
            chunk_len = seqlen_first_chunk if j == 0 and independent_first_chunk else seqlen_per_chunk
            chunk_frames = independent_first_chunk if j == 0 and independent_first_chunk else chunk_size

            position_id.extend([curr_rope] * chunk_len)
            latent_index.extend(list(range(curr, curr + chunk_len)))
            noisy_latent_relative_index.extend(list(range(curr_noisy, curr_noisy + chunk_len)))
            curr += chunk_len
            curr_noisy += chunk_len
            latent_seqlen.append(chunk_len)
            noisy_latent_seqlen.append(chunk_len)
            split_lens.append(chunk_len)
            attn_modes.append("full")
            frame_shifts.append(curr_frame)
            sample_len += chunk_len

            curr_rope += 1
            curr_frame += chunk_frames

        position_ids = torch.tensor(position_id, dtype=torch.int32)
        latent_indexes = torch.tensor(latent_index, dtype=torch.int32)
        latent_seqlens = torch.tensor(latent_seqlen, dtype=torch.int32)
        noisy_latent_relative_indexes = torch.tensor(noisy_latent_relative_index, dtype=torch.int32)
        noisy_latent_seqlens = torch.tensor(noisy_latent_seqlen, dtype=torch.int32)
        q_ranges, k_ranges, attn_type_map, attn_workloads = prepare_flex_attention_mask(
            split_lens, attn_modes, sink=sink, window_size=window_size
        )

        sample.update(dict(
            seqlens=seqlen,
            position_ids=position_ids,
            latent_indexes=latent_indexes,
            latent_seqlens=latent_seqlens,
            noisy_latent_relative_indexes=noisy_latent_relative_indexes,
            noisy_latent_seqlens=noisy_latent_seqlens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            attn_workloads=attn_workloads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            sample_lens=sample_len,
            frame_shifts=frame_shifts,
        ))
        return sample
