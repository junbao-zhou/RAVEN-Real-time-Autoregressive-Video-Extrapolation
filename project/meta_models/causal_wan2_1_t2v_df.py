import itertools
from typing import List

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask

from project.data.utils import create_sparse_mask
from project.meta_models.causal_wan2_1_t2v import (
    CausalWan2_1_T2V,
    CausalWan2_1_T2VForwardInput,
    FLEX_ATTN_BLOCK_SIZE,
)
from project.models.wan2_1 import CausalWanModel
from project.utils.misc import deepcopy_with_tensor, unwrap_model


class CausalWan2_1_T2V_DF(CausalWan2_1_T2V):
    def _chunk_values(self, values, inputs):
        chunks, indexes = list(), list()
        curr = 0
        for i, value in enumerate(values):
            first_chunk_shift = inputs.independent_first_chunks[i] if inputs.independent_first_chunks[i] else inputs.chunk_sizes[i]
            sample_indexes = list()

            chunks.append(value[:, :first_chunk_shift, ...])
            sample_indexes.append(curr)
            curr += 1

            t_rest = value.size(1) - first_chunk_shift
            assert t_rest % inputs.chunk_sizes[i] == 0, f"t_rest {t_rest} not divisible by chunk size {inputs.chunk_sizes[i]}"
            num_rest_chunks = t_rest // inputs.chunk_sizes[i]

            for j in range(num_rest_chunks):
                start_idx = j * inputs.chunk_sizes[i] + first_chunk_shift
                end_idx = start_idx + inputs.chunk_sizes[i]
                chunks.append(value[:, start_idx:end_idx, ...])
                sample_indexes.append(curr)
                curr += 1

            indexes.append(sample_indexes)
        return chunks, indexes

    def _sample_chunk_timesteps(self, inputs, concat_indexes):
        sampling_timesteps = getattr(self, "sampling_timesteps", None)
        if sampling_timesteps is None or sampling_timesteps.timesteps is None:
            raise RuntimeError("CausalWan2_1_T2V_DF requires sampling_timesteps to sample per-chunk diffusion timesteps")

        if getattr(self, "score_timesteps", None) is None:
            max_indexes = sampling_timesteps.index(inputs.timesteps)
            if (max_indexes < 0).any():
                raise RuntimeError(f"Cannot locate generator timesteps {inputs.timesteps.tolist()} in sampling_timesteps")
        else:
            max_indexes = torch.full(
                (len(concat_indexes),),
                sampling_timesteps.num_sampling_steps - 1,
                dtype=torch.int32,
                device=inputs.timesteps.device,
            )

        chunk_timesteps = list()
        for i, concat_index in enumerate(concat_indexes):
            max_index = int(max_indexes[i].item())
            random_index = torch.randint(0, max_index + 1, (len(concat_index),), device=inputs.timesteps.device)
            if sampling_timesteps.timesteps.ndim == 1:
                chunk_timesteps.append(sampling_timesteps.timesteps[random_index].to(device=inputs.timesteps.device))
            else:
                chunk_timesteps.append(sampling_timesteps.timesteps[i, random_index].to(device=inputs.timesteps.device))
        return chunk_timesteps

    def pred(
        self,
        model: CausalWanModel,
        inputs: CausalWan2_1_T2VForwardInput,
        neg_inputs: CausalWan2_1_T2VForwardInput = None,
    ) -> List[Tensor]:
        if not isinstance(unwrap_model(model), CausalWanModel) or inputs.past_key_values_self_attn is not None:
            return super().pred(model, inputs, neg_inputs)

        from project.models.wan2_1.attention import FLEX_FLASH_ATTN_AVAILABLE

        if not FLEX_FLASH_ATTN_AVAILABLE:
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

        latent_chunks, concat_indexes = self._chunk_values(inputs.latents, inputs)
        noise_chunks, _ = self._chunk_values(inputs.noises, inputs)
        chunk_timesteps = self._sample_chunk_timesteps(inputs, concat_indexes)
        ts = torch.cat(chunk_timesteps, dim=0)
        xs = [
            self.schedule.forward(latent, noise, ts[i:i+1])
            for i, (latent, noise) in enumerate(zip(latent_chunks, noise_chunks))
        ]
        inputs.update(dict(
            xts=[
                torch.cat([xs[i] for i in concat_index], dim=1)
                for concat_index in concat_indexes
            ],
            df_chunk_timesteps=chunk_timesteps,
        ))

        out = model(
            x           = [x.to(dtype=model.weight_dtype) for x in xs],
            t           = ts,
            context     = [c.to(dtype=model.weight_dtype) for c in inputs.context],
            packed_position_ids                     = inputs.packed_position_ids,
            packed_latent_indexes                   = inputs.packed_latent_indexes,
            packed_latent_seqlens                   = inputs.packed_latent_seqlens,
            packed_noisy_latent_relative_indexes    = inputs.packed_noisy_latent_relative_indexes,
            packed_noisy_latent_seqlens             = inputs.packed_noisy_latent_seqlens,
            sample_lens                             = inputs.sample_lens,
            frame_shifts                            = list(itertools.chain(*inputs.frame_shifts)),
            attention_mask  = attention_mask,
            q_ranges        = inputs.q_ranges,
            k_ranges        = inputs.k_ranges,
            attn_type_map   = inputs.attn_type_map,
            attn_workloads  = inputs.attn_workloads,
            past_key_values_self_attn               = inputs.past_key_values_self_attn,
            update_past_key_values_self_attn        = inputs.update_past_key_values_self_attn,
            past_key_values_cross_attn              = inputs.past_key_values_cross_attn,
            update_past_key_values_cross_attn       = inputs.update_past_key_values_cross_attn,
            past_key_values_cross_attn_img          = inputs.past_key_values_cross_attn_img,
            update_past_key_values_cross_attn_img   = inputs.update_past_key_values_cross_attn_img
        )

        return [
            torch.cat([out[concat_index[i]] for i in range(len(concat_index))], dim=1)
            for concat_index in concat_indexes
        ]

    def get_endpoint(self, schedule, inputs, pred):
        chunk_timesteps = getattr(inputs, "df_chunk_timesteps", None)
        if chunk_timesteps is None:
            return super().get_endpoint(schedule, inputs, pred)

        pred_chunks, concat_indexes = self._chunk_values(pred, inputs)
        xt_chunks, _ = self._chunk_values(inputs.xts, inputs)
        ts = torch.cat(chunk_timesteps, dim=0)
        x0_chunks = [
            schedule.convert_from_pred(pred_chunks[i], xt_chunks[i], ts[i:i+1])[0]
            for i in range(len(pred_chunks))
        ]
        return [
            torch.cat([x0_chunks[i] for i in concat_index], dim=1)
            for concat_index in concat_indexes
        ]

    def step_to(self, sampler, inputs, pred, s, rng):
        chunk_timesteps = getattr(inputs, "df_chunk_timesteps", None)
        if chunk_timesteps is None:
            return super().step_to(sampler, inputs, pred, s, rng)

        pred_chunks, concat_indexes = self._chunk_values(pred, inputs)
        xt_chunks, _ = self._chunk_values(inputs.xts, inputs)
        ts = torch.cat(chunk_timesteps, dim=0)
        x_s_chunks = list()
        for sample_index, concat_index in enumerate(concat_indexes):
            rng_i = rng[sample_index] if isinstance(rng, list) else rng
            for i in concat_index:
                x_s_chunks.append(sampler.step_to(pred_chunks[i], xt_chunks[i], ts[i:i+1], s[sample_index:sample_index+1], rng_i))
        return [
            torch.cat([x_s_chunks[i] for i in concat_index], dim=1)
            for concat_index in concat_indexes
        ]
