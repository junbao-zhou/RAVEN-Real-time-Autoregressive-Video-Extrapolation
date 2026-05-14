"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp import FSDPModule
from tqdm import tqdm
from project.diffusion.samplers import BaseSampler
from project.diffusion.schedules import BaseSchedule
from project.diffusion.timesteps import BaseSamplingTimesteps

from project.engines.generation.generate_t2v import GenerateT2VInferConfig
from project.meta_models import BaseForwardInput, BaseMetaModel
from project.models.wan2_1 import HuggingfaceTokenizer, T5Encoder, WanModel, WanVAE
from project.utils import comm
from project.utils.config import CfgNode
from project.utils.dataclass import Dataclass
from project.utils.misc import bin_losses, deepcopy_with_tensor
from project.utils.random import RandomState

logger = logging.getLogger()


@dataclass
class Wan2_1_T2VForwardInput(BaseForwardInput):
    context: List[Tensor] = field(default=None)
    max_seq_len: int = field(default=None)
    guidance: Tensor = field(default=None)


@dataclass
class Wan2_1_T2VMetaInferConfig(Dataclass):
    guidance_scale: float = field(default=None)


@dataclass
class Wan2_1_T2VMetaModelConfig(Dataclass):
    patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    vae_stride: List[int] = field(default_factory=lambda: [4, 8, 8])
    z_dim: int = field(default=16)
    guidance_min: Optional[float] = field(default=None)
    guidance_max: Optional[float] = field(default=None)


class Wan2_1_T2V(BaseMetaModel):
    vae: WanVAE
    tokenizer: HuggingfaceTokenizer
    text_encoder: T5Encoder

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        self.meta_model_config = Wan2_1_T2VMetaModelConfig(**cfg["meta_model"]["_config"])

    def setup_meta_model(self):
        self.vae = self.models["vae"]
        self.tokenizer = self.models["tokenizer"]
        self.text_encoder = self.models["text_encoder"]

    def encode_prompts(self, texts: List[str] = None, token_ids: List[List[int]] = None) -> List[torch.Tensor]:
        if token_ids is not None:
            seq_lens = [len(ids) for ids in token_ids]
            max_len = max(seq_lens) if len(seq_lens) > 0 else 0
            if max_len == 0:
                return [torch.tensor([], dtype=torch.float32, device=self.device)]
            padded = [ids + [0] * (max_len - len(ids)) for ids in token_ids]
            ids = torch.tensor(padded, dtype=torch.long).to(self.device)
            mask = torch.zeros(len(token_ids), max_len, dtype=torch.long).to(self.device)
            for i, l in enumerate(seq_lens):
                mask[i, :l] = 1
        else:
            ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
            ids = ids.to(self.device)
            mask = mask.to(self.device)
            seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

    def encode_latents(self, xs: List[Tensor]) -> List[Tensor]:
        return [
            self.vae.encode(x.unsqueeze(0).to(self.vae.weight_dtype)).float().squeeze(0)
            for x in xs
        ]

    def decode_latents(self, zs: List[Tensor]) -> List[Tensor]:
        return [
            self.vae.decode(u.unsqueeze(0).to(self.vae.weight_dtype)).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]

    def merge_inputs(self, *inputs, detach: bool = False):
        merged_inputs = super().merge_inputs(*inputs, detach=detach)
        merged_inputs.update(dict(
            max_seq_len=max(input_i.max_seq_len for input_i in inputs)
        ))
        return merged_inputs

    def mask_inputs(self, inputs, mask, detach: bool = True):
        masked_inputs = super().mask_inputs(inputs, mask, detach=detach)
        max_seq_len = int(masked_inputs.seqlens.max().item()) if int(masked_inputs.batch_size.item()) > 0 else 0
        masked_inputs.update(dict(max_seq_len=max_seq_len))
        return masked_inputs

    def pred_cfg(
        self,
        model: WanModel,
        pos_inputs,
        neg_inputs,
    ) -> List[Tensor]:
        if neg_inputs is None or pos_inputs.guidance is None or not (pos_inputs.guidance > 1.0).any() or model.guidance_embeds is not None:
            return self.pred(model, pos_inputs, neg_inputs)

        bsz = pos_inputs.batch_size
        merged_inputs = self.merge_inputs(pos_inputs, neg_inputs, detach=False)
        pred_all = self.pred(model, merged_inputs)
        pred_pos, pred_neg = pred_all[:bsz], pred_all[bsz:]

        guidance = pos_inputs.guidance
        pred_cfg = [pred_neg[i] + guidance[i] * (pred_pos[i] - pred_neg[i]) for i in range(bsz)]
        return pred_cfg

    def loss_fn(
        self,
        inputs,
        pred: Union[List[Tensor], Tensor],
        target: Optional[Union[List[Tensor], Tensor]] = None,
        key_prefix: str = "losses",
    ) -> Tensor:
        seqlens = torch.tensor([math.prod(pred[i].shape) for i in range(len(pred))], device=self.device)
        total_seqlens = seqlens.sum()
        handle = comm.all_reduce(total_seqlens, op=dist.ReduceOp.SUM, async_op=True)

        if target is not None:
            losses = [F.mse_loss(pred[i].float(), target[i].float().detach(), reduction="none")
                      for i in range(len(pred))]
            losses = torch.stack([loss.sum() for loss in losses])
        else:
            losses = torch.stack([loss.sum() for loss in pred])

        binned_loss_dict, binned_cnt_dict = bin_losses(
            losses=losses, seqlens=seqlens, values=inputs.timesteps,
            bin_size=100, min_value=0, max_value=1000,
            key_prefix=f"{key_prefix}_in_timesteps"
        )
        self.average_meter.update(binned_loss_dict, binned_cnt_dict)

        if inputs.guidance is not None:
            binned_loss_dict, binned_cnt_dict = bin_losses(
                losses=losses, seqlens=seqlens, values=inputs.guidance,
                bin_size=1.0, min_value=1.0, max_value=10.0,
                key_prefix=f"{key_prefix}_in_guidance"
            )
            self.average_meter.update(binned_loss_dict, binned_cnt_dict)

        if handle is not None:
            handle.wait()
        loss = (losses / total_seqlens * comm.get_world_size()).sum()
        return loss

    def prepare_inference_inputs(
        self,
        batch: dict,
        infer_config: GenerateT2VInferConfig,
        rngs: List[RandomState]
    ) -> Tuple[Wan2_1_T2VForwardInput, Wan2_1_T2VForwardInput, Tensor]:
        # specialized meta infer config
        meta_infer_config = Wan2_1_T2VMetaInferConfig(**infer_config.meta_cfg)

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

        # construct forward input
        guidance = torch.tensor([meta_infer_config.guidance_scale] * (bsz * infer_config.num_samples_per_prompt), device=self.device)
        pos_inputs = Wan2_1_T2VForwardInput(
            noises=noises,
            batch_size=batch_size,
            seqlens=seqlens,
            context=context,
            max_seq_len=max_seq_len,
            guidance=guidance
        )

        batch["neg_prompts"] = [c for c in batch["neg_prompts"] for _ in range(infer_config.num_samples_per_prompt)]
        neg_inputs = self.prepare_negative_inputs(batch, pos_inputs)
        rngs[:] = [rng.fork(sample_index) for rng in rngs for sample_index in range(infer_config.num_samples_per_prompt)]
        return pos_inputs, neg_inputs

    def prepare_negative_inputs(
        self,
        batch: dict,
        pos_inputs: Wan2_1_T2VForwardInput,
    ) -> Wan2_1_T2VForwardInput:
        neg_inputs = deepcopy_with_tensor(pos_inputs)
        if batch.get("neg_prompts") is not None:
            neg_context = self.encode_prompts(batch["neg_prompts"])
        elif batch.get("neg_prompt_embs") is not None:
            neg_context = batch["neg_prompt_embs"]
        else:
            raise ValueError("Either neg_prompts or neg_prompt_embs should be provided in the batch")

        neg_inputs.context = neg_context
        return neg_inputs

    def pred(
        self,
        model: WanModel,
        inputs: Wan2_1_T2VForwardInput,
        neg_inputs: Wan2_1_T2VForwardInput = None,
    ) -> List[Tensor]:  # forward pass
        return model(
            x               = [x.to(dtype=model.weight_dtype) for x in inputs.xts],
            t               = inputs.timesteps,
            context         = [c.to(dtype=model.weight_dtype) for c in inputs.context],
            seq_len         = inputs.max_seq_len,
            guidance        = inputs.guidance
        )

    def drop_condition(
        self,
        pos_inputs: Wan2_1_T2VForwardInput,
        neg_inputs: Wan2_1_T2VForwardInput,
        uncond_mask: Tensor,
        rng: RandomState,
    ) -> Wan2_1_T2VForwardInput:
        for i, uncond in enumerate(uncond_mask):
            if uncond:
                pos_inputs.context[i] = deepcopy_with_tensor(neg_inputs.context[i])
        return pos_inputs

    def infer(
        self,
        model: WanModel,
        rng: Union[RandomState, List[RandomState]],
        pos_inputs: Wan2_1_T2VForwardInput,
        neg_inputs: Wan2_1_T2VForwardInput,
        sampling_timesteps: BaseSamplingTimesteps,
        schedule: BaseSchedule,
        sampler: BaseSampler,
        return_trajectory: bool = False,
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[List[Tensor]], List[List[Tensor]]]]:
        bsz = pos_inputs.batch_size
        noisy_latents = pos_inputs.noises

        if return_trajectory:
            trajectory_xt = []
            trajectory_pred = []

        for t in tqdm(sampling_timesteps.timesteps, disable=return_trajectory or comm.get_local_rank() != 0):
            if isinstance(model, FSDPModule):
                model.unshard()  # trigger 1st all-gather earlier

            if return_trajectory:
                trajectory_xt.append(noisy_latents)
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
                trajectory_pred.append(pred)  # list of [C, T, H, W]

        latent_x0s = noisy_latents
        if return_trajectory:  # during training
            return latent_x0s, trajectory_xt, trajectory_pred

        videos = self.decode_latents(latent_x0s)
        return videos
