"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch.utils.data import IterableDataset

from project.data import utils
from project.data.utils import prepare_flex_attention_mask
from project.utils import comm, fs
from project.utils.dataclass import Dataclass
from project.utils.file_io import maybe_download
from project.utils.random import yield_seed

logger = logging.getLogger()


@dataclass
class CausalTextVideoDatasetConfig(Dataclass):
    paths: List[str] = field(default_factory=list)
    seed: int = field(default=42)
    sync_group_size: int = field(default=1)
    verbose: bool = field(default=False)
    total_seqlen: Optional[int] = field(default=None)
    max_seqlen: Optional[int] = field(default=None)
    max_seqlen_per_sample: Optional[int] = field(default=None)
    max_retries: int = field(default=5)
    height: int = field(default=480)
    width: int = field(default=832)
    num_frames: int = field(default=81)
    chunk_size: Union[int, List[int]] = field(default=3)
    independent_first_chunk: Union[int, List[int]] = field(default=4)
    sink: Union[int, List[int]] = field(default=0)
    window_size: Union[int, List[int], None] = field(default=None)


class CausalTextVideoDataset(IterableDataset):
    def __init__(
        self,
        collate_fn,
        state_dict,
        separated_first_frame: bool = True,
        patch_size: List[int] = [1, 2, 2],
        vae_stride: List[int] = [4, 8, 8],
        **kwargs
    ):
        self.separated_first_frame = separated_first_frame
        self.patch_size = patch_size
        self.vae_stride = vae_stride

        self.config = CausalTextVideoDatasetConfig(**kwargs)
        self.collate_fn = utils.get_collate_fn(collate_fn)
        self.max_seqlen = self.config.max_seqlen if self.config.total_seqlen is None else self.config.total_seqlen // comm.get_world_size()
        self.max_seqlen_per_sample = self.config.max_seqlen_per_sample
        self.max_retries = self.config.max_retries
        self.state_dict = state_dict if state_dict is not None else {}
        self.rank = comm.get_rank()
        self.sync_group_size = max(1, self.config.sync_group_size)
        self.offset = self.config.seed
        self.worker_prompts = self.prompts = None

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

        self.prompts = self._load_prompts(self.config.paths)
        assert len(self.prompts) > 0, "No valid prompts found in the provided text files."

    def _load_prompts(self, paths: List[str]) -> List[str]:
        text_files = []
        for path in paths:
            if not fs.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")
            if fs.isdir(path):
                text_files.extend(
                    sorted(f for f in fs.listdir(path, recursive=True) if f.endswith(".txt"))
                )
                continue
            if path.endswith(".txt"):
                text_files.append(path)
                continue
            raise ValueError(f"Only .txt files are supported, got: {path}")

        prompts = []
        seen = set()
        for file_path in sorted(set(text_files)):
            local_path = maybe_download(file_path)
            with open(local_path, "r", encoding="utf8") as f:
                for line in f:
                    prompt = line.strip()
                    if not prompt or prompt in seen:
                        continue
                    seen.add(prompt)
                    prompts.append(prompt)
        return prompts

    def load(self):
        real_worker_id = utils.get_worker_id()
        num_workers = utils.get_num_workers()

        last_worker_id = self.state_dict.get("last_worker_id", -1)
        worker_id = (real_worker_id + last_worker_id + 1) % num_workers

        seed_group_rank = self.rank // self.sync_group_size
        offset = self.offset + seed_group_rank * num_workers + worker_id
        state_dict_offsets = self.state_dict.get(str(worker_id), {})
        self.offset = state_dict_offsets.get("offset", offset)

        logger.info(
            f"Dataset {self.__class__.__name__} Rank {self.rank} Worker {real_worker_id}"
            f" shift to resume worker {worker_id}, seed_group_rank {seed_group_rank}, offset {self.offset}"
        )
        worker_prompts = list(self.prompts)
        random.Random(self.config.seed).shuffle(worker_prompts)
        self.worker_prompts = worker_prompts
        return worker_id

    def _should_skip(self, data, seqlen, cur_seqlen):
        if data is None:
            return True, "process_data returned None"
        if self.max_seqlen_per_sample is not None and seqlen > self.max_seqlen_per_sample:
            return True, f"seqlen {seqlen} exceeds max_seqlen_per_sample {self.max_seqlen_per_sample}"
        if cur_seqlen + seqlen > self.max_seqlen:
            return True, f"seqlen {seqlen} would exceed max_seqlen budget"
        return False, ""

    def process_data(self, prompt: str, rng: random.Random) -> dict:
        random_index = rng.randint(0, len(self.chunk_size) - 1)
        chunk_size = self.chunk_size[random_index]
        independent_first_chunk = self.independent_first_chunk[random_index]
        sink = self.sink[random_index]
        window_size = self.window_size[random_index]

        frames_tensor = torch.randn(3, self.config.num_frames, self.config.height, self.config.width)

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

    def __iter__(self):
        worker_id = self.load()
        avg_seqlen = 0.0
        cnt = 0

        while True:
            rng = random.Random(self.offset)
            ret = []
            cur_seqlen = 0
            num_retries = 0

            while len(ret) == 0 or cur_seqlen + avg_seqlen <= self.max_seqlen:
                prompt = self.worker_prompts[rng.randrange(len(self.worker_prompts))]
                data = self.process_data(prompt, rng)

                seqlen = data["seqlens"] if data is not None else 0
                skip, reason = self._should_skip(data, seqlen, cur_seqlen)

                if skip:
                    if data is not None and cur_seqlen + seqlen > self.max_seqlen:
                        num_retries += 1
                        if num_retries >= self.max_retries:
                            if self.config.verbose:
                                logger.warning(f"break after {num_retries} retries, avg_seqlen={avg_seqlen:.1f}")
                            break
                    if self.config.verbose:
                        logger.warning(f"{self.__class__.__name__}: {reason}")
                    continue

                avg_seqlen = avg_seqlen * cnt / (cnt + 1) + seqlen / (cnt + 1)
                cnt += 1
                cur_seqlen += seqlen
                num_retries = 0
                ret.append(data)

            self.offset = yield_seed(self.offset)
            result = self.collate_fn(ret)
            result["worker_id"] = worker_id
            result["offsets"] = dict(offset=self.offset)
            yield result
