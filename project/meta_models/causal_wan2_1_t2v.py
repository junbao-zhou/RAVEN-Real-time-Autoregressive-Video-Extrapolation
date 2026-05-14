"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import itertools
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp import FSDPModule
from torch.nn.attention.flex_attention import create_block_mask
from tqdm import tqdm

from project.data.utils import create_sparse_mask
from project.diffusion.samplers import BaseSampler
from project.engines.generation.generate_t2v import GenerateT2VInferConfig
from project.diffusion.schedules import BaseSchedule
from project.diffusion.timesteps import BaseSamplingTimesteps
from project.meta_models import BaseForwardInput, BaseMetaModel
from project.meta_models.wan2_1_t2v import Wan2_1_T2V, Wan2_1_T2VForwardInput
from project.models.wan2_1 import CausalWanModel
from project.models.wan2_1.causal_model import NaiveCache
from project.utils import comm
from project.utils.config import CfgNode
from project.utils.dataclass import Dataclass
from project.utils.mfu import disable_flops_accumulate
from project.utils.misc import bin_losses, deepcopy_with_tensor, to_device, unwrap_model, log_n_times
from project.utils.random import RandomState

torch._dynamo.config.cache_size_limit = int(os.environ.get("TORCH_DYNAMO_CACHE_SIZE_LIMIT", 2048))
torch._dynamo.config.accumulated_cache_size_limit = int(os.environ.get("TORCH_DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT", 16384))
FLEX_ATTN_BLOCK_SIZE = 128

create_block_mask = torch.compile(create_block_mask)

logger = logging.getLogger()


@dataclass
class CausalWan2_1_T2VForwardInput(Wan2_1_T2VForwardInput):
    # common packed inputs prepared in training dataloader and infer func respectively
    packed_position_ids: torch.IntTensor = field(default=None)
    packed_latent_indexes: torch.IntTensor = field(default=None)
    packed_latent_seqlens: torch.IntTensor = field(default=None)
    packed_noisy_latent_relative_indexes: torch.IntTensor = field(default=None)
    packed_noisy_latent_seqlens: torch.IntTensor = field(default=None)
    sample_lens: List[int] = field(default=None)
    frame_shifts: List[int] = field(default=None)
    # universal inputs prepared in training dataloader and prepare_inference_inputs func respectively
    chunk_sizes: List[int] = field(default=None)
    independent_first_chunks: List[int] = field(default=None)
    sinks: List[int] = field(default=None)
    window_sizes: List[Optional[int]] = field(default=None)
    # training only prepared in training dataloader
    split_lens: List[int] = field(default=None)
    attn_modes: List[str] = field(default=None)
    q_ranges: torch.IntTensor = field(default=None)
    k_ranges: torch.IntTensor = field(default=None)
    attn_type_map: torch.IntTensor = field(default=None)
    attn_workloads: List[int] = field(default=None)
    # inference only initialized within infer func and updated in each sampling step
    past_key_values_self_attn: Optional[NaiveCache] = field(default=None)
    update_past_key_values_self_attn: bool = field(default=False)
    past_key_values_cross_attn: Optional[NaiveCache] = field(default=None)
    update_past_key_values_cross_attn: bool = field(default=False)
    past_key_values_cross_attn_img: Optional[NaiveCache] = field(default=None)
    update_past_key_values_cross_attn_img: bool = field(default=False)


@dataclass
class CausalWan2_1_T2VMetaInferConfig(Dataclass):
    guidance_scale: float = field(default=None)
    chunk_size: int = field(default=31)
    independent_first_chunk: int = field(default=None)
    sink: int = field(default=0)
    window_size: Optional[int] = field(default=None)


@dataclass
class CausalWan2_1_T2VMetaModelConfig(Dataclass):
    patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    vae_stride: List[int] = field(default_factory=lambda: [4, 8, 8])
    z_dim: int = field(default=16)
    dummy_latents: bool = field(default=False)
    guidance_min: Optional[float] = field(default=None)
    guidance_max: Optional[float] = field(default=None)
    default_neg_prompt: str = field(default="slow motion, time dilation, 高帧率，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    first_chunk_aux_loss: Optional[Dict[str, float]] = field(default=None)
    chunk_wise_weighting: Optional[Dict[str, Union[float, str]]] = field(default=None)
    chunk_wise_balancing: Optional[Dict[str, bool]] = field(default=None)
    # performance tuning
    compile_vae_encoder: bool = field(default=False)


class CausalWan2_1_T2V(Wan2_1_T2V):
    def __init__(self, cfg: CfgNode):
        BaseMetaModel.__init__(self, cfg)
        self.meta_model_config = CausalWan2_1_T2VMetaModelConfig(**cfg["meta_model"]["_config"])

    def setup_dataloader(self):
        return dict(
            separated_first_frame=True,
            patch_size=self.meta_model_config.patch_size,
            vae_stride=self.meta_model_config.vae_stride,
        )

    @torch.no_grad()
    @disable_flops_accumulate()
    def setup_meta_model(self):
        super().setup_meta_model()
        # default negative prompt
        self.default_neg_context = self.encode_prompts([self.meta_model_config.default_neg_prompt])[0]
        # compile
        if self.meta_model_config.compile_vae_encoder:
            self.vae.encoder = torch.compile(self.vae.encoder, dynamic=False, mode="max-autotune-no-cudagraphs")

    def _unpack_packed_samples(
        self,
        inputs: CausalWan2_1_T2VForwardInput,
    ) -> List[Dict[str, Union[List[int], List[str], Tensor, int, Optional[int], Optional[Tensor]]]]:
        assert inputs.split_lens is not None and inputs.attn_modes is not None and inputs.sample_lens is not None, \
            "Training packed fields must exist before unpacking"
        assert inputs.past_key_values_self_attn is None and inputs.past_key_values_cross_attn is None and inputs.past_key_values_cross_attn_img is None, \
            "Unpacking cached inference inputs is not supported"

        batch_size = int(inputs.batch_size.item())
        device = inputs.seqlens.device

        sample_lens_cumsum = torch.cumsum(
            torch.tensor([0] + inputs.sample_lens, dtype=torch.int32, device=device),
            dim=0
        )
        sample_token_indexes = [
            torch.arange(sample_lens_cumsum[i], sample_lens_cumsum[i + 1], dtype=torch.int32, device=device)
            for i in range(batch_size)
        ]

        sample_latent_masks = [torch.isin(inputs.packed_latent_indexes, sample_token_indexes[i]) for i in range(batch_size)]
        latent_lens = [int(mask.sum().item()) for mask in sample_latent_masks]
        latent_lens_cumsum = torch.cumsum(
            torch.tensor([0] + latent_lens, dtype=torch.int32, device=device),
            dim=0
        )

        packed_latent_block_starts = torch.cumsum(inputs.packed_latent_seqlens, dim=0) - inputs.packed_latent_seqlens
        packed_latent_block_sample_indexes = inputs.packed_latent_indexes[packed_latent_block_starts.long()]
        sample_latent_block_masks = [torch.isin(packed_latent_block_sample_indexes, sample_token_indexes[i]) for i in range(batch_size)]
        latent_block_counts = [int(mask.sum().item()) for mask in sample_latent_block_masks]

        packed_noisy_block_starts = torch.cumsum(inputs.packed_noisy_latent_seqlens, dim=0) - inputs.packed_noisy_latent_seqlens
        packed_noisy_block_latent_indexes = inputs.packed_noisy_latent_relative_indexes[packed_noisy_block_starts.long()]
        sample_latent_indexes_per_sample = [
            inputs.packed_latent_indexes[sample_latent_masks[i]] - sample_lens_cumsum[i]
            for i in range(batch_size)
        ]
        noisy_block_counts = []
        for i in range(batch_size):
            sample_noisy_block_mask = torch.isin(
                packed_noisy_block_latent_indexes,
                torch.arange(latent_lens_cumsum[i], latent_lens_cumsum[i + 1], dtype=torch.int32, device=device)
            )
            noisy_block_counts.append(int(sample_noisy_block_mask.sum().item()))

        latent_block_cumsum = torch.cumsum(torch.tensor([0] + latent_block_counts, dtype=torch.int32, device=device), dim=0)
        noisy_block_cumsum = torch.cumsum(torch.tensor([0] + noisy_block_counts, dtype=torch.int32, device=device), dim=0)

        sample_q_range_masks = [
            torch.isin(inputs.q_ranges[:, 0], sample_token_indexes[i]) if inputs.q_ranges is not None else
            torch.zeros((0,), dtype=torch.bool, device=device)
            for i in range(batch_size)
        ]

        unpacked_samples = []
        for i in range(batch_size):
            sample_position_ids = inputs.packed_position_ids[sample_lens_cumsum[i]:sample_lens_cumsum[i + 1]] if inputs.packed_position_ids is not None else \
                torch.zeros((0,), dtype=torch.int32, device=device)

            sample_latent_indexes = sample_latent_indexes_per_sample[i]
            sample_latent_seqlens = inputs.packed_latent_seqlens[latent_block_cumsum[i]:latent_block_cumsum[i + 1]]

            sample_noisy_relative_mask = torch.isin(
                inputs.packed_noisy_latent_relative_indexes,
                torch.arange(latent_lens_cumsum[i], latent_lens_cumsum[i + 1], dtype=torch.int32, device=device)
            )
            sample_noisy_relative_indexes = inputs.packed_noisy_latent_relative_indexes[sample_noisy_relative_mask] - latent_lens_cumsum[i]
            sample_noisy_latent_seqlens = inputs.packed_noisy_latent_seqlens[noisy_block_cumsum[i]:noisy_block_cumsum[i + 1]]

            sample_q_ranges = inputs.q_ranges[sample_q_range_masks[i]] - sample_lens_cumsum[i]
            sample_k_ranges = inputs.k_ranges[sample_q_range_masks[i]] - sample_lens_cumsum[i]
            sample_attn_type_map = inputs.attn_type_map[sample_q_range_masks[i]]
            sample_attn_workload = inputs.attn_workloads[i]

            assert sample_latent_indexes.numel() == int(sample_latent_seqlens.sum().item()), \
                f"Sample {i} latent indexes/seqlens mismatch: {sample_latent_indexes.numel()} vs {int(sample_latent_seqlens.sum().item())}"
            assert sample_noisy_relative_indexes.numel() == int(sample_noisy_latent_seqlens.sum().item()), \
                f"Sample {i} noisy indexes/seqlens mismatch: {sample_noisy_relative_indexes.numel()} vs {int(sample_noisy_latent_seqlens.sum().item())}"

            unpacked_samples.append(dict(
                position_ids=sample_position_ids,
                latent_indexes=sample_latent_indexes,
                latent_seqlens=sample_latent_seqlens,
                noisy_latent_relative_indexes=sample_noisy_relative_indexes,
                noisy_latent_seqlens=sample_noisy_latent_seqlens,
                q_ranges=sample_q_ranges,
                k_ranges=sample_k_ranges,
                attn_type_map=sample_attn_type_map,
                attn_workload=sample_attn_workload,
                split_lens=inputs.split_lens[i],
                attn_modes=inputs.attn_modes[i],
                sample_lens=inputs.sample_lens[i],
                frame_shifts=inputs.frame_shifts[i],
                chunk_sizes=inputs.chunk_sizes[i],
                independent_first_chunks=inputs.independent_first_chunks[i],
                sinks=inputs.sinks[i],
                window_sizes=inputs.window_sizes[i],
            ))

        return unpacked_samples

    def _repack_from_samples(
        self,
        inputs: CausalWan2_1_T2VForwardInput,
        unpacked_samples: List[Dict[str, Union[List[int], List[str], Tensor, int, Optional[int], Optional[Tensor]]]]
    ) -> CausalWan2_1_T2VForwardInput:
        device = inputs.seqlens.device

        sample_lens = [sample["sample_lens"] for sample in unpacked_samples]
        sample_lens_cumsum = torch.cumsum(torch.tensor([0] + sample_lens, dtype=torch.int32, device=device), dim=0)[:-1]

        packed_position_ids = torch.cat([sample["position_ids"] for sample in unpacked_samples], dim=0) if len(unpacked_samples) > 0 else \
            torch.zeros((0,), dtype=torch.int32, device=device)
        packed_latent_indexes = torch.cat([
            sample["latent_indexes"] + sample_lens_cumsum[i] for i, sample in enumerate(unpacked_samples)
        ], dim=0) if len(unpacked_samples) > 0 else torch.zeros((0,), dtype=torch.int32, device=device)
        packed_latent_seqlens = torch.cat([sample["latent_seqlens"] for sample in unpacked_samples], dim=0) if len(unpacked_samples) > 0 else \
            torch.zeros((0,), dtype=torch.int32, device=device)
        packed_noisy_latent_seqlens = torch.cat([sample["noisy_latent_seqlens"] for sample in unpacked_samples], dim=0) if len(unpacked_samples) > 0 else \
            torch.zeros((0,), dtype=torch.int32, device=device)

        q_ranges = []
        k_ranges = []
        attn_type_map = []
        attn_workloads = []
        for i, sample in enumerate(unpacked_samples):
            q_ranges.append(sample["q_ranges"] + sample_lens_cumsum[i])
            k_ranges.append(sample["k_ranges"] + sample_lens_cumsum[i])
            attn_type_map.append(sample["attn_type_map"])
            attn_workloads.append(sample["attn_workload"])
        q_ranges = torch.cat(q_ranges, dim=0) if len(q_ranges) > 0 else torch.zeros((0, 2), dtype=torch.int32, device=device)
        k_ranges = torch.cat(k_ranges, dim=0) if len(k_ranges) > 0 else torch.zeros((0, 2), dtype=torch.int32, device=device)
        attn_type_map = torch.cat(attn_type_map, dim=0) if len(attn_type_map) > 0 else torch.zeros((0,), dtype=torch.int32, device=device)

        latent_offsets = [0]
        for sample in unpacked_samples:
            latent_offsets.append(latent_offsets[-1] + sample["latent_indexes"].numel())
        packed_noisy_latent_relative_indexes = torch.cat([
            sample["noisy_latent_relative_indexes"] + latent_offsets[i] for i, sample in enumerate(unpacked_samples)
        ], dim=0) if len(unpacked_samples) > 0 else torch.zeros((0,), dtype=torch.int32, device=device)

        assert packed_latent_indexes.numel() == int(packed_latent_seqlens.sum().item()), \
            f"Mismatched packed latent indexes and seqlens: {packed_latent_indexes.numel()} vs {int(packed_latent_seqlens.sum().item())}"
        assert packed_noisy_latent_relative_indexes.numel() == int(packed_noisy_latent_seqlens.sum().item()), \
            f"Mismatched packed noisy relative indexes and seqlens: {packed_noisy_latent_relative_indexes.numel()} vs {int(packed_noisy_latent_seqlens.sum().item())}"

        inputs.update(dict(
            packed_position_ids=packed_position_ids,
            packed_latent_indexes=packed_latent_indexes,
            packed_latent_seqlens=packed_latent_seqlens,
            packed_noisy_latent_relative_indexes=packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens=packed_noisy_latent_seqlens,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            attn_workloads=attn_workloads,
        ))
        return inputs

    def merge_inputs(
        self,
        *inputs: CausalWan2_1_T2VForwardInput,
        detach: bool = True
    ) -> CausalWan2_1_T2VForwardInput:
        if len(inputs) == 0:
            raise AssertionError("At least one input is required")
        if inputs[0].split_lens is None:
            return super().merge_inputs(*inputs, detach=detach)

        merged_inputs = super().merge_inputs(*inputs, detach=False)
        unpacked_samples = []
        for input_i in inputs:
            unpacked_samples.extend(self._unpack_packed_samples(input_i))
        merged_inputs = self._repack_from_samples(merged_inputs, unpacked_samples)
        if detach:
            merged_inputs = deepcopy_with_tensor(merged_inputs)
        return merged_inputs

    def mask_inputs(
        self,
        inputs: CausalWan2_1_T2VForwardInput,
        mask: Union[Tensor, List[bool]],
        detach: bool = True
    ) -> CausalWan2_1_T2VForwardInput:
        if inputs.split_lens is None:
            return super().mask_inputs(inputs, mask, detach=detach)

        batch_size = int(inputs.batch_size.item())
        if isinstance(mask, Tensor):
            assert mask.ndim == 1, f"Mask should be 1D, got shape {tuple(mask.shape)}"
            mask = mask.to(device=inputs.batch_size.device, dtype=torch.bool)
        else:
            mask = torch.tensor(mask, device=inputs.batch_size.device, dtype=torch.bool)
        assert mask.numel() == batch_size, f"Mask length {mask.numel()} does not match batch size {batch_size}"

        masked_inputs = super().mask_inputs(inputs, mask, detach=False)
        unpacked_samples = self._unpack_packed_samples(inputs)
        unpacked_samples = [sample for sample, keep in zip(unpacked_samples, mask.tolist()) if keep]
        masked_inputs = self._repack_from_samples(masked_inputs, unpacked_samples)
        if detach:
            masked_inputs = deepcopy_with_tensor(masked_inputs)
        return masked_inputs

    def prepare_training_inputs(self, batch: dict, rng: RandomState) -> Dict[str, BaseForwardInput]:
        # move all tensors to device
        batch = to_device(batch, self.device, non_blocking=True)

        # encode videos
        videos: List[Tensor] = batch["videos"]
        videos = [video * 2.0 - 1.0 for video in videos]  # List of Tensor in [C, T, H, W], ranging [0, 1] to [-1, 1]
        if not self.meta_model_config.dummy_latents:
            latents = self.encode_latents(videos)
        else:
            latents = [
                torch.empty(
                    self.meta_model_config.z_dim,
                    (video.shape[1] - 1) // self.meta_model_config.vae_stride[0] + 1,
                    video.shape[2] // self.meta_model_config.vae_stride[1],
                    video.shape[3] // self.meta_model_config.vae_stride[2],
                    device=self.device
                ).normal_(generator=rng.torch_cuda_generator) for video in videos
            ]

        # encode prompts
        if batch.get("prompts") is not None:
            context = self.encode_prompts(batch["prompts"])
        elif batch.get("prompt_embs") is not None:
            context = batch["prompt_embs"]
        else:
            raise ValueError("Either prompts or prompt_embs should be provided in the batch")

        # sample scale
        guidance = None
        if self.meta_model_config.guidance_min is not None and self.meta_model_config.guidance_max is not None:
            guidance_min, guidance_max = self.meta_model_config.guidance_min, self.meta_model_config.guidance_max
            guidance = torch.rand(len(videos), device=self.device, generator=rng.torch_cuda_generator) * (guidance_max - guidance_min) + guidance_min

        # pack data
        sample_lens = batch["sample_lens"]
        sample_lens_cumsum = torch.cumsum(torch.tensor([0] + sample_lens), dim=0)[:-1]
        packed_position_ids = torch.cat(batch["position_ids"], dim=0)
        packed_latent_indexes = torch.cat([latent_indexes + sample_lens_cumsum[i] for i, latent_indexes in enumerate(batch["latent_indexes"])], dim=0)
        packed_latent_seqlens = torch.cat(batch["latent_seqlens"], dim=0)
        packed_noisy_latent_seqlens = torch.cat(batch["noisy_latent_seqlens"], dim=0)
        q_ranges = torch.cat([q_range + sample_lens_cumsum[i] for i, q_range in enumerate(batch["q_ranges"])])
        k_ranges = torch.cat([k_range + sample_lens_cumsum[i] for i, k_range in enumerate(batch["k_ranges"])])
        attn_type_map = torch.cat(batch["attn_type_map"])

        curr, curr_noisy = 0, 0
        noisy_latent_relative_indexes = list()
        for i in range(len(sample_lens)):
            sample_len = sample_lens[i]
            noisy_latent_relative_index = batch["noisy_latent_relative_indexes"][i]
            sample_indexes = torch.tensor(list(range(curr, curr + sample_len)), dtype=torch.int32, device=self.device)
            sample_latent_indexes = packed_latent_indexes[torch.isin(packed_latent_indexes, sample_indexes)]
            noisy_latent_relative_indexes.append(curr_noisy + noisy_latent_relative_index)
            curr += sample_len
            curr_noisy += len(sample_latent_indexes)
        packed_noisy_latent_relative_indexes = torch.cat(noisy_latent_relative_indexes, dim=0)

        # construct pos inputs
        max_seq_len = max(batch["seqlens"])
        seqlens = torch.tensor(batch["seqlens"], dtype=torch.int32, device=self.device)
        batch_size = torch.tensor(len(videos), dtype=torch.int32, device=self.device)
        pos_inputs = CausalWan2_1_T2VForwardInput(
            latents=latents,
            batch_size=batch_size,
            seqlens=seqlens,
            context=context,
            max_seq_len=max_seq_len,
            guidance=guidance,
            prompts=batch.get("prompts", None),
            # packed
            packed_position_ids=packed_position_ids,
            packed_latent_indexes=packed_latent_indexes,
            packed_latent_seqlens=packed_latent_seqlens,
            packed_noisy_latent_relative_indexes=packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens=packed_noisy_latent_seqlens,
            sample_lens=sample_lens,
            frame_shifts=batch["frame_shifts"],
            # training only
            split_lens=batch["split_lens"],
            attn_modes=batch["attn_modes"],
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            attn_workloads=batch["attn_workloads"],
            # universal
            chunk_sizes=batch["chunk_sizes"],
            independent_first_chunks=batch["independent_first_chunks"],
            sinks=batch["sinks"],
            window_sizes=batch["window_sizes"],
        )

        # construct neg inputs
        if batch.get("neg_prompts") is None:
            batch["neg_prompt_embs"] = [deepcopy_with_tensor(self.default_neg_context) for _ in range(batch_size)]
        neg_inputs = self.prepare_negative_inputs(batch, pos_inputs)
        return pos_inputs, neg_inputs

    def prepare_inference_inputs(
        self,
        batch: dict,
        infer_config: GenerateT2VInferConfig,
        rngs: List[RandomState]
    ) -> Tuple[CausalWan2_1_T2VForwardInput, CausalWan2_1_T2VForwardInput, Tensor]:
        # specialized meta infer config
        meta_infer_config = CausalWan2_1_T2VMetaInferConfig(**infer_config.meta_cfg)

        # latent init
        bsz = len(batch["prompts"])
        seqlens = []
        noises = []

        for i in range(bsz):
            h, w = infer_config.height, infer_config.width
            lat_t = (infer_config.num_frames - 1) // self.meta_model_config.vae_stride[0] + 1
            lat_h = round(
                h // self.meta_model_config.vae_stride[1] //
                self.meta_model_config.patch_size[1] * self.meta_model_config.patch_size[1]
            )
            lat_w = round(
                w // self.meta_model_config.vae_stride[2] //
                self.meta_model_config.patch_size[2] * self.meta_model_config.patch_size[2]
            )
            h = lat_h * self.meta_model_config.vae_stride[1]
            w = lat_w * self.meta_model_config.vae_stride[2]

            seqlen = lat_t * lat_h * lat_w // math.prod(self.meta_model_config.patch_size)
            seqlens.extend([seqlen] * infer_config.num_samples_per_prompt)

            noise = torch.empty(
                infer_config.num_samples_per_prompt, self.meta_model_config.z_dim, lat_t, lat_h, lat_w,
                dtype=torch.float32, device=self.device
            ).normal_(generator=rngs[i].torch_cuda_generator)
            noises.extend([noise[j] for j in range(infer_config.num_samples_per_prompt)])

        max_seq_len = max(seqlens)
        batch_size = torch.tensor(bsz, device=self.device) * infer_config.num_samples_per_prompt
        seqlens = torch.tensor(seqlens, device=self.device)

        # text encoding
        context = self.encode_prompts(batch["prompts"])
        context = [u.clone() for u in context for _ in range(infer_config.num_samples_per_prompt)]

        # causal
        chunk_sizes = [meta_infer_config.chunk_size] * batch_size
        independent_first_chunks = [meta_infer_config.independent_first_chunk] * batch_size
        sinks = [meta_infer_config.sink] * batch_size
        window_sizes = [meta_infer_config.window_size] * batch_size

        # construct forward input
        guidance = torch.tensor([meta_infer_config.guidance_scale] * (bsz * infer_config.num_samples_per_prompt), device=self.device)
        pos_inputs = CausalWan2_1_T2VForwardInput(
            noises=noises,
            batch_size=batch_size,
            seqlens=seqlens,
            context=context,
            prompts=[p for p in batch["prompts"] for _ in range(infer_config.num_samples_per_prompt)],
            max_seq_len=max_seq_len,
            guidance=guidance,
            # universal
            chunk_sizes=chunk_sizes,
            independent_first_chunks=independent_first_chunks,
            sinks=sinks,
            window_sizes=window_sizes
        )

        batch["neg_prompts"] = [c for c in batch["neg_prompts"] for _ in range(infer_config.num_samples_per_prompt)]
        neg_inputs = self.prepare_negative_inputs(batch, pos_inputs)
        rngs[:] = [rng.fork(sample_index) for rng in rngs for sample_index in range(infer_config.num_samples_per_prompt)]
        return pos_inputs, neg_inputs

    def infer(
        self,
        model: CausalWanModel,
        rng: Union[RandomState, List[RandomState]],
        pos_inputs: CausalWan2_1_T2VForwardInput,
        neg_inputs: CausalWan2_1_T2VForwardInput,
        sampling_timesteps: BaseSamplingTimesteps,
        schedule: BaseSchedule,
        sampler: BaseSampler,
        return_trajectory: bool = False,
        latent_x0s: Optional[Union[List[Tensor], Tensor]] = None,
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[List[Tensor]], List[List[Tensor]]]]:
        if not isinstance(unwrap_model(model), CausalWanModel):
            return super().infer(model, rng, pos_inputs, neg_inputs, sampling_timesteps, schedule, sampler, return_trajectory)

        bsz = pos_inputs.batch_size
        noises = pos_inputs.noises
        chunk_sizes, independent_first_chunks, sinks, window_sizes = pos_inputs.chunk_sizes, pos_inputs.independent_first_chunks, pos_inputs.sinks, pos_inputs.window_sizes
        if isinstance(latent_x0s, Tensor):
            latent_x0s = [latent_x0s[i] for i in range(latent_x0s.size(0))]
        if latent_x0s is not None:
            assert len(latent_x0s) == bsz, f"latent_x0s length {len(latent_x0s)} does not match batch size {bsz}"

        seqlens_per_frame = [noise.size(2) * noise.size(3) // (model.patch_size[1] * model.patch_size[2]) for noise in noises]
        seqlens_per_chunk = [seqlen_per_frame * chunk_size for seqlen_per_frame, chunk_size in zip(seqlens_per_frame, chunk_sizes)]
        independent_first_chunks = [independent_first_chunks[i] if independent_first_chunks[i] is not None else chunk_sizes[i] for i in range(bsz)]
        num_chunks = [(noises[i].size(1) - independent_first_chunks[i]) // chunk_sizes[i] + 1 for i in range(bsz)]
        max_num_chunks = torch.tensor(max(num_chunks), dtype=torch.int32, device=self.device)
        comm.all_reduce(max_num_chunks, op=dist.ReduceOp.MAX)

        # prepare first chunk
        noisy_latents = [noise[:, :first_chunk_shift, :, :] for noise, first_chunk_shift in zip(noises, independent_first_chunks)]
        frame_shifts = [[0] * bsz]

        # common
        sample_lens, curr, curr_noisy = list(), 0, 0
        position_ids = list()
        latent_indexes, latent_seqlens = list(), list()
        noisy_latent_relative_indexes, noisy_latent_seqlens = list(), list()
        for i, (seqlen_per_frame, first_chunk_shift) in enumerate(zip(seqlens_per_frame, independent_first_chunks)):
            latent_seqlen = seqlen_per_frame * first_chunk_shift
            sample_len = latent_seqlen

            sample_lens.append(sample_len)
            position_ids.extend([0] * latent_seqlen)
            latent_indexes.extend(list(range(curr, curr + latent_seqlen)))
            latent_seqlens.append(latent_seqlen)
            noisy_latent_relative_indexes.extend(list(range(curr_noisy, curr_noisy + latent_seqlen)))
            noisy_latent_seqlens.append(latent_seqlen)

            curr += latent_seqlen
            curr_noisy += latent_seqlen

        packed_position_ids = torch.tensor(position_ids, dtype=torch.int32, device=self.device)
        packed_latent_indexes = torch.tensor(latent_indexes, dtype=torch.int32, device=self.device)
        packed_latent_seqlens = torch.tensor(latent_seqlens, dtype=torch.int32, device=self.device)
        packed_noisy_latent_relative_indexes = torch.tensor(noisy_latent_relative_indexes, dtype=torch.int32, device=self.device)
        packed_noisy_latent_seqlens = torch.tensor(noisy_latent_seqlens, dtype=torch.int32, device=self.device)

        # inference only kvcache
        self_attn_cache = NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes)
        cross_attn_cache = NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes)
        cross_attn_cache_img = NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes)
        neg_self_attn_cache = NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes)
        neg_cross_attn_cache = NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes)
        neg_cross_attn_cache_img = NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes)

        # update pos_inputs and neg_inputs with the initialized common packed inputs and kvcaches
        pos_inputs.update(dict(
            packed_position_ids=packed_position_ids,
            packed_latent_indexes=packed_latent_indexes,
            packed_latent_seqlens=packed_latent_seqlens,
            packed_noisy_latent_relative_indexes=packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens=packed_noisy_latent_seqlens,
            sample_lens=sample_lens,
            frame_shifts=frame_shifts,
            past_key_values_self_attn=self_attn_cache,
            past_key_values_cross_attn=cross_attn_cache,
            past_key_values_cross_attn_img=cross_attn_cache_img
        ))
        neg_inputs.update(dict(
            packed_position_ids=packed_position_ids,
            packed_latent_indexes=packed_latent_indexes,
            packed_latent_seqlens=packed_latent_seqlens,
            packed_noisy_latent_relative_indexes=packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens=packed_noisy_latent_seqlens,
            sample_lens=sample_lens,
            frame_shifts=frame_shifts,
            past_key_values_self_attn=neg_self_attn_cache,
            past_key_values_cross_attn=neg_cross_attn_cache,
            past_key_values_cross_attn_img=neg_cross_attn_cache_img
        ))

        # infer first chunk
        latents = []
        if return_trajectory:
            trajectory_xt = [[] for _ in range(max_num_chunks)]
            trajectory_pred = [[] for _ in range(max_num_chunks)]

        for t in tqdm(sampling_timesteps.timesteps, disable=return_trajectory or comm.get_local_rank() != 0):
            if isinstance(model, FSDPModule):
                model.unshard()  # trigger 1st all-gather earlier

            if return_trajectory:
                trajectory_xt[0].append(noisy_latents)
            t = t.expand((bsz,))
            s = sampling_timesteps.get_next_timesteps(t)

            pos_inputs = self.set_timesteps(pos_inputs, t)
            pos_inputs = self.set_noisy_latents(pos_inputs, noisy_latents)
            neg_inputs = self.set_timesteps(neg_inputs, t)
            neg_inputs = self.set_noisy_latents(neg_inputs, noisy_latents)
            pred = self.pred_cfg(model, pos_inputs, neg_inputs)

            noisy_latents = self.step_to(
                sampler=sampler, inputs=pos_inputs, pred=pred, s=s, rng=rng
            )
            if return_trajectory:
                trajectory_pred[0].append(pred)  # list of [C, T, H, W]

        # cache first chunk
        latents.append(noisy_latents)
        if isinstance(model, FSDPModule):
            model.unshard()  # trigger 1st all-gather earlier
        cache_latents = noisy_latents if latent_x0s is None else [
            latent_x0s[i][:, :independent_first_chunks[i], :, :]
            for i in range(bsz)
        ]
        pos_inputs = self.set_noisy_latents(pos_inputs, cache_latents)
        neg_inputs = self.set_noisy_latents(neg_inputs, cache_latents)
        pos_inputs, neg_inputs = self.cache(model, pos_inputs, neg_inputs)

        # common
        sample_lens, curr, curr_noisy = list(), 0, 0
        position_ids = list()
        latent_indexes, latent_seqlens = list(), list()
        noisy_latent_relative_indexes, noisy_latent_seqlens = list(), list()
        for i, latent_seqlen in enumerate(seqlens_per_chunk):
            sample_lens.append(latent_seqlen)
            position_ids.extend([0] * latent_seqlen)
            latent_indexes.extend(list(range(curr, curr + latent_seqlen)))
            latent_seqlens.append(latent_seqlen)
            noisy_latent_relative_indexes.extend(list(range(curr_noisy, curr_noisy + latent_seqlen)))
            noisy_latent_seqlens.append(latent_seqlen)

            curr += latent_seqlen
            curr_noisy += latent_seqlen

        packed_position_ids = torch.tensor(position_ids, dtype=torch.int32, device=self.device)
        packed_latent_indexes = torch.tensor(latent_indexes, dtype=torch.int32, device=self.device)
        packed_latent_seqlens = torch.tensor(latent_seqlens, dtype=torch.int32, device=self.device)
        packed_noisy_latent_relative_indexes = torch.tensor(noisy_latent_relative_indexes, dtype=torch.int32, device=self.device)
        packed_noisy_latent_seqlens = torch.tensor(noisy_latent_seqlens, dtype=torch.int32, device=self.device)

        # update pos_inputs and neg_inputs with the initialized common packed inputs and kvcaches
        pos_inputs.update(dict(
            packed_position_ids=packed_position_ids,
            packed_latent_indexes=packed_latent_indexes,
            packed_latent_seqlens=packed_latent_seqlens,
            packed_noisy_latent_relative_indexes=packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens=packed_noisy_latent_seqlens,
            sample_lens=sample_lens,
        ))
        neg_inputs.update(dict(
            packed_position_ids=packed_position_ids,
            packed_latent_indexes=packed_latent_indexes,
            packed_latent_seqlens=packed_latent_seqlens,
            packed_noisy_latent_relative_indexes=packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens=packed_noisy_latent_seqlens,
            sample_lens=sample_lens,
        ))

        for i in range(max_num_chunks - 1):
            # prepare rest chunk
            noisy_latents = [
                noise[:, min(i * chunk_size + first_chunk_shift, noise.size(1) - chunk_size) :
                      min(i * chunk_size + first_chunk_shift, noise.size(1) - chunk_size) + chunk_size, :, :]
                for noise, chunk_size, first_chunk_shift in zip(noises, chunk_sizes, independent_first_chunks)
            ]

            frame_shifts = [[first_chunk_shift + i * chunk_size] for chunk_size, first_chunk_shift in zip(chunk_sizes, independent_first_chunks)]
            pos_inputs.update(dict(frame_shifts=frame_shifts))
            neg_inputs.update(dict(frame_shifts=frame_shifts))

            # infer rest chunk
            for t in tqdm(sampling_timesteps.timesteps, disable=return_trajectory or comm.get_local_rank() != 0):
                if isinstance(model, FSDPModule):
                    model.unshard()  # trigger 1st all-gather earlier

                if return_trajectory:
                    trajectory_xt[i+1].append(noisy_latents)
                t = t.expand((bsz,))
                s = sampling_timesteps.get_next_timesteps(t)

                pos_inputs = self.set_timesteps(pos_inputs, t)
                neg_inputs = self.set_timesteps(neg_inputs, t)
                pos_inputs = self.set_noisy_latents(pos_inputs, noisy_latents=noisy_latents)
                neg_inputs = self.set_noisy_latents(neg_inputs, noisy_latents=noisy_latents)
                pred = self.pred_cfg(model, pos_inputs, neg_inputs)

                noisy_latents = self.step_to(
                    sampler=sampler, inputs=pos_inputs, pred=pred, s=s, rng=rng
                )
                if return_trajectory:
                    trajectory_pred[i+1].append(pred)  # list of [C, T, H, W]

            # cache rest chunk
            latents.append(noisy_latents)
            if i != max_num_chunks - 2:  # no need to cache the last chunk
                if isinstance(model, FSDPModule):
                    model.unshard()  # trigger 1st all-gather earlier
                if latent_x0s is None:
                    cache_latents = noisy_latents
                else:
                    cache_latents = [
                        latent_x0s[j][:, min(i * chunk_sizes[j] + independent_first_chunks[j], latent_x0s[j].size(1) - chunk_sizes[j]) :
                                           min(i * chunk_sizes[j] + independent_first_chunks[j], latent_x0s[j].size(1) - chunk_sizes[j]) + chunk_sizes[j], :, :]
                        for j in range(bsz)
                    ]
                pos_inputs = self.set_noisy_latents(pos_inputs, noisy_latents=cache_latents)
                neg_inputs = self.set_noisy_latents(neg_inputs, noisy_latents=cache_latents)
                pos_inputs, neg_inputs = self.cache(model, pos_inputs, neg_inputs)

        # aggregate latents
        latents = [torch.cat([latents[i][j] for i in range(num_chunks[j])], dim=1) for j in range(bsz)]

        if return_trajectory:  # during training
            trajectory_xt = [
                [torch.cat([trajectory_xt[i][k][j] for i in range(num_chunks[j])], dim=1) for j in range(bsz)]
                for k in range(len(trajectory_xt[0]))  # i-th chunk, j-th sample, k-th timestep
            ]
            trajectory_pred = [
                [torch.cat([trajectory_pred[i][k][j] for i in range(num_chunks[j])], dim=1) for j in range(bsz)]
                for k in range(len(trajectory_pred[0]))  # i-th chunk, j-th sample, k-th timestep
            ]
            return latents, trajectory_xt, trajectory_pred

        videos = self.decode_latents(latents)
        return videos

    def cache(
        self,
        model: CausalWanModel,
        inputs: CausalWan2_1_T2VForwardInput,
        neg_inputs: CausalWan2_1_T2VForwardInput = None,
    ) -> Union[CausalWan2_1_T2VForwardInput, Tuple[CausalWan2_1_T2VForwardInput, CausalWan2_1_T2VForwardInput]]:
        if neg_inputs is not None and inputs.guidance is not None and (inputs.guidance > 1.0).any():
            pos_inputs = self.cache(model, inputs)
            neg_inputs = self.cache(model, neg_inputs)
            return pos_inputs, neg_inputs

        # if not cfg_pred, no need to update anything inside neg_inputs
        ori_packed_noisy_latent_relative_indexes = inputs.packed_noisy_latent_relative_indexes
        packed_noisy_latent_relative_indexes = torch.tensor([], dtype=torch.int32, device=self.device)
        inputs.update(dict(
            packed_noisy_latent_relative_indexes=packed_noisy_latent_relative_indexes,
            update_past_key_values_self_attn=True,
            update_past_key_values_cross_attn=True,
            update_past_key_values_cross_attn_img=True
        ))
        _ = self.pred(model, inputs, neg_inputs)
        inputs.update(dict(
            packed_noisy_latent_relative_indexes=ori_packed_noisy_latent_relative_indexes,
            update_past_key_values_self_attn=False,
            update_past_key_values_cross_attn=False,
            update_past_key_values_cross_attn_img=False,
            # do not update kvcache for cross attn anymore in the following caching process
            context=[],
        ))
        if neg_inputs is not None:
            return inputs, neg_inputs
        else:
            return inputs

    def pred(
        self,
        model: CausalWanModel,
        inputs: CausalWan2_1_T2VForwardInput,
        neg_inputs: CausalWan2_1_T2VForwardInput = None,
    ) -> List[Tensor]:  # forward pass
        if not isinstance(unwrap_model(model), CausalWanModel):
            return super().pred(model, inputs, neg_inputs)

        # check if inference
        use_flex = inputs.past_key_values_self_attn is None

        # check if use pytorch flex attention
        from project.models.wan2_1.attention import FLEX_FLASH_ATTN_AVAILABLE
        if not FLEX_FLASH_ATTN_AVAILABLE and use_flex:  # block mask needed
            seqlen = sum(inputs.sample_lens)
            seqlen_pad = (seqlen + FLEX_ATTN_BLOCK_SIZE - 1) // FLEX_ATTN_BLOCK_SIZE * FLEX_ATTN_BLOCK_SIZE
            pad_len = seqlen_pad - seqlen
            if pad_len > 0:
                inputs.sample_lens = deepcopy_with_tensor(inputs.sample_lens) + [pad_len]
                inputs.split_lens = deepcopy_with_tensor(inputs.split_lens) + [[pad_len]]
                inputs.attn_modes = deepcopy_with_tensor(inputs.attn_modes) + [['causal']]
            split_lens = list(itertools.chain(*inputs.split_lens))
            attn_modes = list(itertools.chain(*inputs.attn_modes))
            sparse_mask = create_sparse_mask(inputs.sample_lens, split_lens, attn_modes, self.device,
                                             sink=inputs.sinks, window_size=inputs.window_sizes)
            seqlen = sum(inputs.sample_lens)
            assert seqlen % FLEX_ATTN_BLOCK_SIZE == 0, f"seqlen {seqlen} not divisible by block size {FLEX_ATTN_BLOCK_SIZE}"
            attention_mask = create_block_mask(
                sparse_mask, B=1, H=model.num_heads, Q_LEN=seqlen, KV_LEN=seqlen,
                device=self.device, BLOCK_SIZE=FLEX_ATTN_BLOCK_SIZE
            )
        else:
            attention_mask = None

        if use_flex:  # training with xts/timestep/guidance in sample-level, not chunk-level
            # split xts/timestep/guidance if training
            xs, ts = list(), list()
            curr, concat_indexes = 0, list()

            for i, xt in enumerate(inputs.xts):
                first_chunk_shift = inputs.independent_first_chunks[i] if inputs.independent_first_chunks[i] else inputs.chunk_sizes[i]

                cond_i = inputs.latents[i]
                concat_index = list()

                # first noisy chunk
                xs.append(xt[:, :first_chunk_shift, ...])
                ts.append(inputs.timesteps[i:i+1])
                concat_index.append(curr)
                curr += 1

                # first clean chunk
                cond = cond_i[:, :first_chunk_shift, ...]
                xs.append(cond)
                curr += 1

                t_rest = xt.size(1) - first_chunk_shift
                assert t_rest % inputs.chunk_sizes[i] == 0, f"t_rest {t_rest} not divisible by chunk size {inputs.chunk_sizes[i]}"
                num_rest_chunks = t_rest // inputs.chunk_sizes[i]

                for j in range(num_rest_chunks):
                    start_idx = j * inputs.chunk_sizes[i] + first_chunk_shift
                    end_idx = start_idx + inputs.chunk_sizes[i]
                    xs.append(xt[:, start_idx:end_idx, ...])  # noisy chunk
                    concat_index.append(curr)
                    curr += 1
                    if j != num_rest_chunks - 1:
                        cond = cond_i[:, start_idx:end_idx, ...]
                        xs.append(cond)  # clean chunk
                        curr += 1

                ts.extend([inputs.timesteps[i:i+1]] * num_rest_chunks)
                concat_indexes.append(concat_index)

            ts = torch.cat(ts, dim=0)
        else:
            xs = inputs.xts
            ts = inputs.timesteps
            concat_indexes = None

        out = model(
            x           = [x.to(dtype=model.weight_dtype) for x in xs],
            t           = ts,
            context     = [c.to(dtype=model.weight_dtype) for c in inputs.context],
            # common packed
            packed_position_ids                     = inputs.packed_position_ids,
            packed_latent_indexes                   = inputs.packed_latent_indexes,
            packed_latent_seqlens                   = inputs.packed_latent_seqlens,
            packed_noisy_latent_relative_indexes    = inputs.packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens             = inputs.packed_noisy_latent_seqlens,
            sample_lens                             = inputs.sample_lens,
            frame_shifts                            = list(itertools.chain(*inputs.frame_shifts)),
            # training only
            attention_mask  = attention_mask,
            q_ranges        = inputs.q_ranges,
            k_ranges        = inputs.k_ranges,
            attn_type_map   = inputs.attn_type_map,
            attn_workloads  = inputs.attn_workloads,
            # inference only
            past_key_values_self_attn               = inputs.past_key_values_self_attn,
            update_past_key_values_self_attn        = inputs.update_past_key_values_self_attn,
            past_key_values_cross_attn              = inputs.past_key_values_cross_attn,
            update_past_key_values_cross_attn       = inputs.update_past_key_values_cross_attn,
            past_key_values_cross_attn_img          = inputs.past_key_values_cross_attn_img,
            update_past_key_values_cross_attn_img   = inputs.update_past_key_values_cross_attn_img
        )

        return [
            torch.cat([out[concat_index[i]] for i in range(len(concat_index))], dim=1)  # [C, T, H, W]
            for concat_index in concat_indexes
        ] if concat_indexes is not None else out

    def pred_cfg(
        self,
        model: CausalWanModel,
        pos_inputs: CausalWan2_1_T2VForwardInput,
        neg_inputs: CausalWan2_1_T2VForwardInput,
    ) -> List[Tensor]:
        if not isinstance(unwrap_model(model), CausalWanModel):
            return super().pred_cfg(model, pos_inputs, neg_inputs)
        if neg_inputs is None or pos_inputs.guidance is None or not (pos_inputs.guidance > 1.0).any():
            return self.pred(model, pos_inputs, neg_inputs)

        bsz = pos_inputs.batch_size
        pred_pos = self.pred(model, pos_inputs)
        pred_neg = self.pred(model, neg_inputs)

        guidance = pos_inputs.guidance
        pred_cfg = [pred_neg[i] + guidance[i] * (pred_pos[i] - pred_neg[i]) for i in range(bsz)]
        return pred_cfg

    def loss_fn(
        self,
        inputs: CausalWan2_1_T2VForwardInput,
        pred: Union[List[Tensor], Tensor],
        target: Optional[Union[List[Tensor], Tensor]] = None,
        key_prefix: str = "losses"
    ) -> Tensor:
        loss = super().loss_fn(inputs, pred, target, key_prefix)

        apply_chunk_wise_balancing = (
            self.meta_model_config.chunk_wise_balancing is not None and
            self.meta_model_config.chunk_wise_balancing.get(key_prefix, False)
        )
        apply_chunk_wise_weighting = (
            self.meta_model_config.chunk_wise_weighting is not None and
            key_prefix in self.meta_model_config.chunk_wise_weighting
        )

        if self.meta_model_config.chunk_wise_balancing is not None and key_prefix not in self.meta_model_config.chunk_wise_balancing:
            log_n_times(f"WARNING: Chunk-wise balancing config provided but no balancing specified for {key_prefix}.", id=key_prefix)
        if self.meta_model_config.chunk_wise_weighting is not None and key_prefix not in self.meta_model_config.chunk_wise_weighting:
            log_n_times(f"WARNING: Chunk-wise weighting config provided but no weighting specified for {key_prefix}.", id=key_prefix)

        if apply_chunk_wise_balancing or apply_chunk_wise_weighting:
            weighting_cfg = self.meta_model_config.chunk_wise_weighting[key_prefix] if apply_chunk_wise_weighting else None
            weighted_losses = []
            weighted_seqlens = []

            for i, pred_i in enumerate(pred):
                first_chunk_size = inputs.independent_first_chunks[i] if inputs.independent_first_chunks[i] is not None else inputs.chunk_sizes[i]
                chunk_sizes = [first_chunk_size]
                t_rest = pred_i.size(1) - first_chunk_size
                assert t_rest % inputs.chunk_sizes[i] == 0, f"t_rest {t_rest} not divisible by chunk size {inputs.chunk_sizes[i]}"
                chunk_sizes.extend([inputs.chunk_sizes[i]] * (t_rest // inputs.chunk_sizes[i]))

                chunk_preds = []
                chunk_targets = []
                start_idx = 0
                for chunk_size in chunk_sizes:
                    end_idx = start_idx + chunk_size
                    chunk_preds.append(pred_i[:, start_idx:end_idx, ...])
                    if target is not None:
                        chunk_targets.append(target[i][:, start_idx:end_idx, ...])
                    start_idx = end_idx

                chunk_seqlens = torch.tensor([math.prod(chunk_pred.shape) for chunk_pred in chunk_preds], device=self.device)
                if apply_chunk_wise_weighting:
                    participation = torch.flip(torch.cumsum(torch.flip(chunk_seqlens, dims=[0]), dim=0), dims=[0]).float()
                    participation = participation / participation[0].clamp_min(1.0)

                    if isinstance(weighting_cfg, str):
                        parts = weighting_cfg.lower().split("_")
                        if parts[0] in ["ln", "logitnormal"]:
                            if len(parts) != 3:
                                raise ValueError(f"LogitNormal chunk-wise weighting should be ln_loc_scale, got {weighting_cfg}")
                            loc, scale = float(parts[1]), float(parts[2])
                            if scale <= 0:
                                raise ValueError(f"LogitNormal chunk-wise weighting scale must be positive, got {scale}")
                            upper = participation
                            lower = torch.cat([participation[1:], participation.new_zeros(1)])
                            upper_cdf = 0.5 * (1 + torch.erf((torch.logit(upper.clamp(1e-6, 1 - 1e-6)) - loc) / (scale * math.sqrt(2.0))))
                            lower_cdf = 0.5 * (1 + torch.erf((torch.logit(lower.clamp(1e-6, 1 - 1e-6)) - loc) / (scale * math.sqrt(2.0))))
                            upper_cdf = torch.where(upper >= 1, torch.ones_like(upper_cdf), upper_cdf)
                            lower_cdf = torch.where(lower <= 0, torch.zeros_like(lower_cdf), lower_cdf)
                            raw_weights = (upper_cdf - lower_cdf) / (upper - lower).clamp_min(1e-6)
                        elif parts[0] == "mode":
                            if len(parts) == 2:
                                scale = float(parts[1])
                            elif len(parts) == 3:
                                scale = float(f"{parts[1]}.{parts[2]}")
                            else:
                                raise ValueError(f"Mode chunk-wise weighting should be mode_scale or mode_0_1, got {weighting_cfg}")
                            upper = participation
                            lower = torch.cat([participation[1:], participation.new_zeros(1)])
                            upper_lo, upper_hi = torch.zeros_like(upper), torch.ones_like(upper)
                            lower_lo, lower_hi = torch.zeros_like(lower), torch.ones_like(lower)
                            for _ in range(32):
                                upper_mid = (upper_lo + upper_hi) / 2
                                lower_mid = (lower_lo + lower_hi) / 2
                                upper_value = 1 - upper_mid - scale * (torch.cos(torch.pi / 2 * upper_mid) ** 2 - 1 + upper_mid)
                                lower_value = 1 - lower_mid - scale * (torch.cos(torch.pi / 2 * lower_mid) ** 2 - 1 + lower_mid)
                                upper_lo = torch.where(upper_value > upper, upper_mid, upper_lo)
                                upper_hi = torch.where(upper_value > upper, upper_hi, upper_mid)
                                lower_lo = torch.where(lower_value > lower, lower_mid, lower_lo)
                                lower_hi = torch.where(lower_value > lower, lower_hi, lower_mid)
                            upper_inverse = (upper_lo + upper_hi) / 2
                            lower_inverse = (lower_lo + lower_hi) / 2
                            raw_weights = (lower_inverse - upper_inverse) / (upper - lower).clamp_min(1e-6)
                        else:
                            raise ValueError(f"Unsupported chunk-wise weighting strategy: {weighting_cfg}")
                        if (raw_weights < 0).any():
                            raise ValueError(f"Chunk-wise weighting strategy {weighting_cfg} produced negative weights")
                    elif weighting_cfg > 0:
                        raw_weights = weighting_cfg * participation / (1 + (weighting_cfg - 1) * participation)
                    elif weighting_cfg < 0:
                        reverse_participation = participation[-1] / participation.clamp_min(1e-6)
                        shift = -weighting_cfg
                        raw_weights = shift * reverse_participation / (1 + (shift - 1) * reverse_participation)
                    else:
                        raw_weights = torch.ones_like(participation)
                else:
                    raw_weights = torch.ones_like(chunk_seqlens, dtype=torch.float32)

                if apply_chunk_wise_balancing:
                    raw_weights = raw_weights / chunk_seqlens.float().clamp_min(1.0)
                raw_weighted_seqlens = (raw_weights * chunk_seqlens).sum()
                if raw_weighted_seqlens <= 0:
                    raise ValueError("Chunk-wise weighting produced non-positive total weight")
                norm = chunk_seqlens.sum() / raw_weighted_seqlens
                weights = raw_weights * norm

                if target is not None:
                    chunk_losses = torch.stack([
                        F.mse_loss(chunk_pred.float(), chunk_target.float().detach(), reduction="none").sum()
                        for chunk_pred, chunk_target in zip(chunk_preds, chunk_targets)
                    ])
                else:
                    chunk_losses = torch.stack([chunk_pred.sum() for chunk_pred in chunk_preds])

                weighted_losses.append((weights * chunk_losses).sum())
                weighted_seqlens.append((weights * chunk_seqlens).sum())

            weighted_losses = torch.stack(weighted_losses)
            weighted_seqlens = torch.stack(weighted_seqlens)
            total_weighted_seqlens = weighted_seqlens.sum()
            handle = comm.all_reduce(total_weighted_seqlens, op=dist.ReduceOp.SUM, async_op=True)

            if handle is not None:
                handle.wait()
            loss = (weighted_losses / total_weighted_seqlens * comm.get_world_size()).sum()

        if self.meta_model_config.first_chunk_aux_loss is not None and key_prefix in self.meta_model_config.first_chunk_aux_loss:
            aux_weight = self.meta_model_config.first_chunk_aux_loss[key_prefix]
            first_chunk_sizes = [
                independent_first_chunk if independent_first_chunk is not None else chunk_size
                for chunk_size, independent_first_chunk in zip(inputs.chunk_sizes, inputs.independent_first_chunks)
            ]
            first_chunk_pred = [pred_i[:, :first_chunk_size, ...] for pred_i, first_chunk_size in zip(pred, first_chunk_sizes)]
            seqlens = torch.tensor([math.prod(pred_i.shape) for pred_i in first_chunk_pred], device=self.device)
            total_seqlens = seqlens.sum()
            handle = comm.all_reduce(total_seqlens, op=dist.ReduceOp.SUM, async_op=True)

            if target is not None:
                first_chunk_target = [target_i[:, :first_chunk_size, ...] for target_i, first_chunk_size in zip(target, first_chunk_sizes)]
                losses = [F.mse_loss(pred_i.float(), target_i.float().detach(), reduction="none")
                          for pred_i, target_i in zip(first_chunk_pred, first_chunk_target)]
                losses = torch.stack([loss.sum() for loss in losses])
            else:
                losses = torch.stack([loss.sum() for loss in first_chunk_pred])

            binned_loss_dict, binned_cnt_dict = bin_losses(
                losses=losses, seqlens=seqlens, values=inputs.timesteps,
                bin_size=100, min_value=0, max_value=1000,
                key_prefix=f"{key_prefix}_first_chunk_in_timesteps"
            )
            self.average_meter.update(binned_loss_dict, binned_cnt_dict)

            if inputs.guidance is not None:
                binned_loss_dict, binned_cnt_dict = bin_losses(
                    losses=losses, seqlens=seqlens, values=inputs.guidance,
                    bin_size=1.0, min_value=1.0, max_value=10.0,
                    key_prefix=f"{key_prefix}_first_chunk_in_guidance"
                )
                self.average_meter.update(binned_loss_dict, binned_cnt_dict)

            if handle is not None:
                handle.wait()
            aux_loss = (losses / total_seqlens * comm.get_world_size()).sum()
            loss = loss + aux_weight * aux_loss

        return loss

    def reward_fn(
        self,
        reward_model: nn.Module,
        inputs: Wan2_1_T2VForwardInput,
        tensors: Union[List[Tensor], Tensor],
        **kwargs,
    ) -> Union[Dict[str, Tensor], Tensor]:
        samples = list(tensors) if isinstance(tensors, Tensor) else tensors
        assert len(samples) > 0, "reward_fn got empty tensors"

        channels = samples[0].shape[0]
        if channels == self.meta_model_config.z_dim:
            video_tensors = self.decode_latents(samples)
        elif channels == 3:
            video_tensors = samples
        else:
            raise ValueError(f"Unsupported tensor shape for reward_fn: {tuple(samples[0].shape)}")

        source_fps = 16.0
        return super().reward_fn(
            reward_model,
            inputs,
            video_tensors,
            source_fps=source_fps,
            **kwargs,
        )
