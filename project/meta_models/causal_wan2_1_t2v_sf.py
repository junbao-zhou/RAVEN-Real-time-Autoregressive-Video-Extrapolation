from typing import List

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.fsdp import FSDPModule

from project.meta_models.causal_wan2_1_t2v import CausalWan2_1_T2V, CausalWan2_1_T2VForwardInput
from project.models.wan2_1 import CausalWanModel
from project.models.wan2_1.causal_model import NaiveCache
from project.utils import comm
from project.utils.misc import unwrap_model


class CausalWan2_1_T2V_SF(CausalWan2_1_T2V):
    def _snapshot_cache(self, cache: NaiveCache) -> NaiveCache:
        snapshot = NaiveCache(
            cache.num_layers,
            cache.batch_size,
            sink=list(cache.sink) if isinstance(cache.sink, list) else cache.sink,
            window_size=list(cache.window_size) if isinstance(cache.window_size, list) else cache.window_size,
        )
        snapshot.key_cache = dict(cache.key_cache)
        snapshot.value_cache = dict(cache.value_cache)
        snapshot.kvlens = list(cache.kvlens)
        snapshot.curr_rope = cache.curr_rope.clone()
        snapshot.chunk_lens = [list(chunk_lens) for chunk_lens in cache.chunk_lens]
        return snapshot

    def _snapshot_inputs(self, inputs: CausalWan2_1_T2VForwardInput) -> CausalWan2_1_T2VForwardInput:
        snapshot = CausalWan2_1_T2VForwardInput(**dict(inputs))
        snapshot.update(dict(
            past_key_values_self_attn=self._snapshot_cache(inputs.past_key_values_self_attn),
            past_key_values_cross_attn=self._snapshot_cache(inputs.past_key_values_cross_attn),
            past_key_values_cross_attn_img=self._snapshot_cache(inputs.past_key_values_cross_attn_img),
        ))
        return snapshot

    def pred(
        self,
        model: CausalWanModel,
        inputs: CausalWan2_1_T2VForwardInput,
        neg_inputs: CausalWan2_1_T2VForwardInput = None,
    ) -> List[Tensor]:
        if (
            not isinstance(unwrap_model(model), CausalWanModel)
            or inputs.past_key_values_self_attn is not None
            or inputs.chunk_sizes is None
            or inputs.independent_first_chunks is None
        ):
            return super().pred(model, inputs, neg_inputs)

        bsz = int(inputs.batch_size.item()) if isinstance(inputs.batch_size, Tensor) else int(inputs.batch_size)
        xts = inputs.xts
        if isinstance(xts, Tensor):
            xts = [xts[i] for i in range(xts.size(0))]

        chunk_sizes = inputs.chunk_sizes
        independent_first_chunks = [
            inputs.independent_first_chunks[i] if inputs.independent_first_chunks[i] is not None else chunk_sizes[i]
            for i in range(bsz)
        ]
        sinks = inputs.sinks
        window_sizes = inputs.window_sizes
        seqlens_per_frame = [
            xt.size(2) * xt.size(3) // (model.patch_size[1] * model.patch_size[2])
            for xt in xts
        ]
        seqlens_per_chunk = [
            seqlen_per_frame * chunk_size
            for seqlen_per_frame, chunk_size in zip(seqlens_per_frame, chunk_sizes)
        ]
        num_chunks = [
            (xts[i].size(1) - independent_first_chunks[i]) // chunk_sizes[i] + 1
            for i in range(bsz)
        ]
        max_num_chunks = torch.tensor(max(num_chunks), dtype=torch.int32, device=self.device)
        comm.all_reduce(max_num_chunks, op=dist.ReduceOp.MAX)
        max_num_chunks = int(max_num_chunks.item())

        sf_inputs = CausalWan2_1_T2VForwardInput(**dict(inputs))
        sf_inputs.update(dict(
            past_key_values_self_attn=NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes),
            past_key_values_cross_attn=NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes),
            past_key_values_cross_attn_img=NaiveCache(model.num_layers, bsz, sink=sinks, window_size=window_sizes),
            update_past_key_values_self_attn=False,
            update_past_key_values_cross_attn=False,
            update_past_key_values_cross_attn_img=False,
        ))
        outputs = [[] for _ in range(bsz)]

        sample_lens, curr, curr_noisy = list(), 0, 0
        position_ids, latent_indexes, latent_seqlens = list(), list(), list()
        noisy_latent_relative_indexes, noisy_latent_seqlens = list(), list()
        for i, (seqlen_per_frame, first_chunk_shift) in enumerate(zip(seqlens_per_frame, independent_first_chunks)):
            latent_seqlen = seqlen_per_frame * first_chunk_shift
            sample_lens.append(latent_seqlen)
            position_ids.extend([0] * latent_seqlen)
            latent_indexes.extend(list(range(curr, curr + latent_seqlen)))
            latent_seqlens.append(latent_seqlen)
            noisy_latent_relative_indexes.extend(list(range(curr_noisy, curr_noisy + latent_seqlen)))
            noisy_latent_seqlens.append(latent_seqlen)
            curr += latent_seqlen
            curr_noisy += latent_seqlen

        sf_inputs.update(dict(
            packed_position_ids=torch.tensor(position_ids, dtype=torch.int32, device=self.device),
            packed_latent_indexes=torch.tensor(latent_indexes, dtype=torch.int32, device=self.device),
            packed_latent_seqlens=torch.tensor(latent_seqlens, dtype=torch.int32, device=self.device),
            packed_noisy_latent_relative_indexes=torch.tensor(noisy_latent_relative_indexes, dtype=torch.int32, device=self.device),
            packed_noisy_latent_seqlens=torch.tensor(noisy_latent_seqlens, dtype=torch.int32, device=self.device),
            sample_lens=sample_lens,
            frame_shifts=[[0] * bsz],
        ))

        noisy_latents = [
            xt[:, :first_chunk_shift, :, :]
            for xt, first_chunk_shift in zip(xts, independent_first_chunks)
        ]
        sf_inputs = self.set_noisy_latents(sf_inputs, noisy_latents)
        sf_inputs = self.set_timesteps(sf_inputs, inputs.timesteps)
        if isinstance(model, FSDPModule):
            model.unshard()
        pred_inputs = self._snapshot_inputs(sf_inputs)
        pred = super().pred(model, pred_inputs)
        for i in range(bsz):
            outputs[i].append(pred[i])

        if max_num_chunks > 1:
            cache_latents = [x0.detach() for x0 in self.get_endpoint(self.schedule, pred_inputs, pred)]
            sf_inputs = self.set_noisy_latents(sf_inputs, cache_latents)
            with torch.no_grad():
                if isinstance(model, FSDPModule):
                    model.unshard()
                self.cache(model, sf_inputs)

        sample_lens, curr, curr_noisy = list(), 0, 0
        position_ids, latent_indexes, latent_seqlens = list(), list(), list()
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

        sf_inputs.update(dict(
            packed_position_ids=torch.tensor(position_ids, dtype=torch.int32, device=self.device),
            packed_latent_indexes=torch.tensor(latent_indexes, dtype=torch.int32, device=self.device),
            packed_latent_seqlens=torch.tensor(latent_seqlens, dtype=torch.int32, device=self.device),
            packed_noisy_latent_relative_indexes=torch.tensor(noisy_latent_relative_indexes, dtype=torch.int32, device=self.device),
            packed_noisy_latent_seqlens=torch.tensor(noisy_latent_seqlens, dtype=torch.int32, device=self.device),
            sample_lens=sample_lens,
        ))

        for chunk_idx in range(max_num_chunks - 1):
            noisy_latents = [
                xt[:, min(chunk_idx * chunk_size + first_chunk_shift, xt.size(1) - chunk_size):
                   min(chunk_idx * chunk_size + first_chunk_shift, xt.size(1) - chunk_size) + chunk_size, :, :]
                for xt, chunk_size, first_chunk_shift in zip(xts, chunk_sizes, independent_first_chunks)
            ]
            sf_inputs.update(dict(
                frame_shifts=[
                    [first_chunk_shift + chunk_idx * chunk_size]
                    for chunk_size, first_chunk_shift in zip(chunk_sizes, independent_first_chunks)
                ]
            ))
            sf_inputs = self.set_noisy_latents(sf_inputs, noisy_latents)
            sf_inputs = self.set_timesteps(sf_inputs, inputs.timesteps)
            if isinstance(model, FSDPModule):
                model.unshard()
            pred_inputs = self._snapshot_inputs(sf_inputs)
            pred = super().pred(model, pred_inputs)
            for i in range(bsz):
                outputs[i].append(pred[i])

            if chunk_idx != max_num_chunks - 2:
                cache_latents = [x0.detach() for x0 in self.get_endpoint(self.schedule, pred_inputs, pred)]
                sf_inputs = self.set_noisy_latents(sf_inputs, cache_latents)
                with torch.no_grad():
                    if isinstance(model, FSDPModule):
                        model.unshard()
                    self.cache(model, sf_inputs)

        return [
            torch.cat(outputs[i][:num_chunks[i]], dim=1)
            for i in range(bsz)
        ]
