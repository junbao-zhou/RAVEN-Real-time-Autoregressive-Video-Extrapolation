"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FSDPModule

from project.diffusion.samplers import SAMPLER_REGISTRY, BaseSampler
from project.diffusion.schedules import SCHEDULE_REGISTRY, BaseSchedule
from project.diffusion.timesteps import TIMESTEP_REGISTRY, BaseTrainingTimesteps, BaseSamplingTimesteps
from project.engines.diffusion_finetuning import DiffusionFinetuning
from project.engines.base_engine import BaseEngine
from project.meta_models import BaseForwardInput
from project.utils import comm, running
from project.utils.config import CfgNode
from project.utils.random import RandomState
from project.utils.dataclass import Dataclass
from project.utils.misc import deepcopy_with_tensor

logger = logging.getLogger()


@dataclass
class DistributionMatchingDistillationConfig(Dataclass):
    # diffusion
    training_timestep: CfgNode = field(default=None)
    fake_training_timestep: CfgNode = field(default=None)
    sampling_timestep: CfgNode = field(default=None)
    score_timestep: CfgNode = field(default=None)
    schedule: CfgNode = field(default=None)
    sampler: CfgNode = field(default=None)
    fake_schedule: CfgNode = field(default=None)
    fake_sampler: CfgNode = field(default=None)
    tea_schedule: CfgNode = field(default=None)
    tea_sampler: CfgNode = field(default=None)
    # general
    validation: List[CfgNode] = field(default_factory=list)
    save_before_train: bool = field(default=False)
    val_before_train: bool = field(default=True)
    val_backbone: bool = field(default=True)
    val_ema_model: bool = field(default=True)
    fake_step_models: List[str] = field(default_factory=lambda: ["fake_model"])
    gen_step_models: List[str] = field(default_factory=lambda: ["backbone"])
    # training
    training_steps: int = field(default=1000000)
    ga_steps: int = field(default=1)  # gradient accumulation steps
    slices_per_step: int = field(default=1)
    prepare_before_sync: bool = field(default=True)  # whether to prepare inputs before sync for unified parallel
    prepare_after_sync: bool = field(default=False)  # whether to prepare inputs after sync for unified parallel
    uncond_fake_train_prob: float = field(default=0.0)  # per-sample Bernoulli prob of dropping cond in fake training step (0.0 = never, 1.0 = always)
    uncond_fake_eval: bool = field(default=False)  # whether to drop cond for fake_model in gen_training_step (DMD score / eval path)
    fake_loss_type: str = field(default="v_lerp")  # x_0, x_T, v_cos, v_lerp, x_t
    dmd_loss_type: str = field(default="dmd")  # ["dmd", "sim", "sid"]
    norm_clip_min: float = field(default=1e-5)  # for dmd and sid
    phuber_c: float = field(default=0.001)  # for sim only
    alpha: float = field(default=1.0)  # for sid only, alpha=1.0 equals to dmd w/ fake and real gradient bwd
    fake_grad_enabled: bool = field(default=False)
    real_grad_enabled: bool = field(default=False)
    ttur_gen: int = field(default=1)
    gen_use_trajectory: bool = field(default=False)  # whether to use trajectory prediction for gen training step
    gen_use_real_data: bool = field(default=False)  # whether to use true x0 for gen training step, only valid when gen_use_trajectory is False
    gen_random_noise: bool = field(default=False)
    ttur_fake: int = field(default=1)
    fake_use_trajectory: bool = field(default=False)  # whether to use trajectory prediction for fake training step
    fake_use_prev: bool = field(default=False)
    fake_random_noise: bool = field(default=False)
    fake_reset_iter: int = field(default=0)


class DistributionMatchingDistillation(DiffusionFinetuning):
    backbone: nn.Module
    fake_model: nn.Module
    tea_model: nn.Module

    training_timesteps: BaseTrainingTimesteps
    fake_training_timesteps: BaseTrainingTimesteps
    sampling_timesteps: BaseSamplingTimesteps
    score_timesteps: BaseTrainingTimesteps
    schedule: BaseSchedule
    sampler: BaseSampler
    fake_schedule: BaseSchedule
    fake_sampler: BaseSampler
    tea_schedule: BaseSchedule
    tea_sampler: BaseSampler

    def __init__(self, cfg: CfgNode):
        BaseEngine.__init__(self, cfg)
        self.setup_writer(cfg)

        self.engine_config = DistributionMatchingDistillationConfig(**self.config["_config"])
        assert self.engine_config.slices_per_step >= 1, \
            f"slices_per_step must be >= 1, got {self.engine_config.slices_per_step}"
        assert self.engine_config.ga_steps == 1, \
            "DistributionMatchingDistillation does not support ga_steps > 1, use slices_per_step instead"
        self.backbone = self.models["backbone"]
        self.fake_model = self.models["fake_model"]
        self.tea_model = self.models["tea_model"]
        self.configure_diffusion()

        self.sync_inputs = lambda inputs: [inputs]

    def configure_diffusion(self):
        # backbone training timestep
        timestep_cls = TIMESTEP_REGISTRY.get(self.engine_config.training_timestep["_class_name"])
        self.training_timesteps = timestep_cls(**self.engine_config.training_timestep["_config"])
        # fake training timestep
        fake_timestep_cls = TIMESTEP_REGISTRY.get(self.engine_config.fake_training_timestep["_class_name"])
        self.fake_training_timesteps = fake_timestep_cls(**self.engine_config.fake_training_timestep["_config"])
        # sampling timestep
        sampling_timestep_cls = TIMESTEP_REGISTRY.get(self.engine_config.sampling_timestep["_class_name"])
        self.sampling_timesteps = sampling_timestep_cls(**self.engine_config.sampling_timestep["_config"])
        # backbone schedule
        schedule_cls = SCHEDULE_REGISTRY.get(self.engine_config.schedule["_class_name"])
        self.schedule = schedule_cls(**self.engine_config.schedule["_config"])
        # backbone sampler
        sampler_cls = SAMPLER_REGISTRY.get(self.engine_config.sampler["_class_name"])
        self.sampler = sampler_cls(schedule=self.schedule, **self.engine_config.sampler["_config"])
        # fake schedule
        fake_schedule_cls = SCHEDULE_REGISTRY.get(self.engine_config.fake_schedule["_class_name"])
        self.fake_schedule = fake_schedule_cls(**self.engine_config.fake_schedule["_config"])
        # fake sampler
        fake_sampler_cls = SAMPLER_REGISTRY.get(self.engine_config.fake_sampler["_class_name"])
        self.fake_sampler = fake_sampler_cls(schedule=self.fake_schedule, **self.engine_config.fake_sampler["_config"])
        # tea schedule
        tea_schedule_cls = SCHEDULE_REGISTRY.get(self.engine_config.tea_schedule["_class_name"])
        self.tea_schedule = tea_schedule_cls(**self.engine_config.tea_schedule["_config"])
        # tea sampler
        tea_sampler_cls = SAMPLER_REGISTRY.get(self.engine_config.tea_sampler["_class_name"])
        self.tea_sampler = tea_sampler_cls(schedule=self.tea_schedule, **self.engine_config.tea_sampler["_config"])

        # [optional] score timesteps to decouple from training timesteps
        if self.engine_config.score_timestep is not None:
            score_timestep_cls = TIMESTEP_REGISTRY.get(self.engine_config.score_timestep["_class_name"])
            self.score_timesteps = score_timestep_cls(**self.engine_config.score_timestep["_config"])
        else:
            self.score_timesteps = None

    def training_loop(
        self,
        inputs: Tuple[BaseForwardInput, BaseForwardInput],
        rng: RandomState
    ) -> Dict[str, float]:
        # log data stats
        pos_inputs, neg_inputs = inputs
        running.get_running_accumulator().put_scalar(f"data/num_total_tokens", sum(pos_inputs.seqlens))
        running.get_running_accumulator().put_scalar(f"data/num_samples", pos_inputs.batch_size)
        log_dict = dict()
        batch_size = int(pos_inputs.batch_size.item()) if isinstance(pos_inputs.batch_size, torch.Tensor) else int(pos_inputs.batch_size)
        slice_indices = torch.arange(batch_size, device=self.device).chunk(min(self.engine_config.slices_per_step, batch_size))
        slice_rollouts = []

        for slice_index in slice_indices:
            slice_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            slice_mask[slice_index] = True
            slice_pos_inputs = self.mask_inputs(pos_inputs, slice_mask)
            slice_neg_inputs = self.mask_inputs(neg_inputs, slice_mask)
            bwd_pos_inputs = deepcopy_with_tensor(slice_pos_inputs)
            bwd_neg_inputs = deepcopy_with_tensor(slice_neg_inputs)

            self.sampling_timesteps.set_timesteps(seqlen=bwd_pos_inputs.seqlens, device=self.device)
            sampling_noises = self.sample_noises(bwd_pos_inputs, rng)
            bwd_pos_inputs = self.set_noises(bwd_pos_inputs, sampling_noises)
            bwd_neg_inputs = self.set_noises(bwd_neg_inputs, sampling_noises)
            with torch.no_grad():
                latent_x0s, trajectory_xt, trajectory_pred = self.infer(
                    model=self.backbone,
                    rng=rng,
                    pos_inputs=bwd_pos_inputs,
                    neg_inputs=bwd_neg_inputs,
                    sampling_timesteps=self.sampling_timesteps,
                    schedule=self.schedule,
                    sampler=self.sampler,
                    return_trajectory=True,
                )
            slice_rollouts.append((
                slice_pos_inputs,
                slice_neg_inputs,
                (latent_x0s, trajectory_xt, trajectory_pred),
            ))

        # ttur_gen > 1 means more generator steps, skip fake step
        if (self.iter // self.engine_config.ga_steps + 1) % self.engine_config.ttur_gen == 0:
            self.unfreeze(self.fake_model)
            if isinstance(self.fake_model, FSDP) and (self.iter + 1) % self.engine_config.ga_steps != 0:
                ctx = self.fake_model.no_sync()
            else:
                ctx = nullcontext()
            if isinstance(self.fake_model, FSDPModule):
                self.fake_model.set_requires_gradient_sync(requires_gradient_sync=(self.iter + 1) % self.engine_config.ga_steps == 0)

            with ctx:
                fake_slice_losses = []
                for slice_pos_inputs, slice_neg_inputs, trajectory in slice_rollouts:
                    self.trajectory = trajectory
                    running.set_training_phase(running.TrainingPhase.IN_FORWARD)
                    fake_loss = self.fake_training_step(
                        deepcopy_with_tensor(slice_pos_inputs),
                        deepcopy_with_tensor(slice_neg_inputs),
                        rng,
                    )
                    fake_slice_losses.append((fake_loss.detach(), int(slice_pos_inputs.batch_size.item())))
                    running.set_training_phase(running.TrainingPhase.IN_BACKWARD)
                    (fake_loss * (int(slice_pos_inputs.batch_size.item()) / (batch_size * self.engine_config.ga_steps))).backward()
                total_fake_loss_weight = sum(loss_weight for _, loss_weight in fake_slice_losses)
                log_dict["train/fake_loss"] = (
                    torch.stack([loss * loss_weight for loss, loss_weight in fake_slice_losses]).sum() / total_fake_loss_weight
                ).item()
            running.set_training_phase(running.TrainingPhase.IN_OPTIMIZATION)
            fake_norm_dict = self.optimize(self.engine_config.fake_step_models)
            log_dict.update(fake_norm_dict)

        # ttur_fake > 1 means more fake steps, skip gen step
        if (self.iter // self.engine_config.ga_steps + 1) % self.engine_config.ttur_fake == 0:
            self.freeze(self.fake_model)
            if isinstance(self.backbone, FSDP) and (self.iter + 1) % self.engine_config.ga_steps != 0:
                ctx = self.backbone.no_sync()
            else:
                ctx = nullcontext()
            if isinstance(self.backbone, FSDPModule):
                self.backbone.set_requires_gradient_sync(requires_gradient_sync=(self.iter + 1) % self.engine_config.ga_steps == 0)

            with ctx:
                gen_slice_losses = []
                for slice_pos_inputs, slice_neg_inputs, trajectory in slice_rollouts:
                    self.trajectory = trajectory
                    running.set_training_phase(running.TrainingPhase.IN_FORWARD)
                    gen_loss = self.gen_training_step(
                        deepcopy_with_tensor(slice_pos_inputs),
                        deepcopy_with_tensor(slice_neg_inputs),
                        rng,
                    )
                    gen_slice_losses.append((gen_loss.detach(), int(slice_pos_inputs.batch_size.item())))
                    running.set_training_phase(running.TrainingPhase.IN_BACKWARD)
                    (gen_loss * (int(slice_pos_inputs.batch_size.item()) / (batch_size * self.engine_config.ga_steps))).backward()
                total_gen_loss_weight = sum(loss_weight for _, loss_weight in gen_slice_losses)
                log_dict["train/gen_loss"] = (
                    torch.stack([loss * loss_weight for loss, loss_weight in gen_slice_losses]).sum() / total_gen_loss_weight
                ).item()
            running.set_training_phase(running.TrainingPhase.IN_OPTIMIZATION)
            gen_norm_dict = self.optimize(self.engine_config.gen_step_models)
            log_dict.update(gen_norm_dict)

        # fake reset iter
        if self.engine_config.fake_reset_iter > 0 and (self.iter // self.engine_config.ga_steps + 1) % self.engine_config.fake_reset_iter == 0:
            self.unfreeze(self.fake_model)
            with torch.no_grad():
                for src, tgt in zip(self.tea_model.parameters(), self.fake_model.parameters()):
                    if not tgt.requires_grad:
                        continue
                    tgt.detach().copy_(src)

        return log_dict

    def fake_training_step(
        self,
        pos_inputs: BaseForwardInput,
        neg_inputs: BaseForwardInput,
        rng: RandomState
    ) -> torch.Tensor:
        batch_size = pos_inputs.batch_size
        fake_inputs = pos_inputs
        latent_x0s, trajectory_xt, trajectory_pred = self.trajectory

        # trigger 1st all-gather earlier
        if isinstance(self.fake_model, FSDPModule):
            self.fake_model.unshard()

        fake_timesteps = self.sample_timesteps(fake_inputs, self.fake_training_timesteps, rng)
        if self.engine_config.fake_use_trajectory:
            self.sampling_timesteps.set_timesteps(seqlen=fake_inputs.seqlens, device=self.device)
            sampling_timesteps = self.sampling_timesteps.timesteps
            if sampling_timesteps.ndim == 1:
                sampling_timesteps = sampling_timesteps[None, :]
            mask = sampling_timesteps > fake_timesteps[:, None]
            random_index = mask.sum(dim=1).clamp(min=1) - 1
            start_timesteps = self.sampling_timesteps.timesteps[random_index] if self.sampling_timesteps.timesteps.ndim == 1 else \
                self.sampling_timesteps.timesteps[torch.arange(batch_size, device=self.device), random_index]
            fake_inputs = self.set_timesteps(fake_inputs, start_timesteps)
            fake_inputs = self.set_noisy_latents(
                inputs=fake_inputs,
                noisy_latents=self.concat([trajectory_xt[random_index[i]][i] for i in range(batch_size)])
            )
            fake_noisy_latents = self.step_to(
                sampler=self.sampler,
                inputs=fake_inputs,
                pred=self.concat([trajectory_pred[random_index[i]][i] for i in range(batch_size)]),
                s=fake_timesteps,
                rng=rng
            )
            fake_inputs = self.set_latents(fake_inputs, latent_x0s)
            fake_inputs = self.set_noises(fake_inputs, trajectory_xt[0])
            fake_inputs = self.set_timesteps(fake_inputs, fake_timesteps)
            fake_inputs = self.set_noisy_latents(fake_inputs, fake_noisy_latents)
        elif self.engine_config.fake_use_prev:
            self.sampling_timesteps.set_timesteps(seqlen=fake_inputs.seqlens, device=self.device)
            sampling_timesteps = self.sampling_timesteps.timesteps
            if sampling_timesteps.ndim == 1:
                sampling_timesteps = sampling_timesteps[None, :]
            mask = sampling_timesteps > fake_timesteps[:, None]
            random_index = mask.sum(dim=1).clamp(min=1) - 1
            start_timesteps = self.sampling_timesteps.timesteps[random_index] if self.sampling_timesteps.timesteps.ndim == 1 else \
                self.sampling_timesteps.timesteps[torch.arange(batch_size, device=self.device), random_index]
            next_timesteps = self.sampling_timesteps.get_next_timesteps(start_timesteps)
            fake_inputs = self.set_timesteps(fake_inputs, start_timesteps)
            fake_inputs = self.set_noisy_latents(
                inputs=fake_inputs,
                noisy_latents=self.concat([trajectory_xt[random_index[i]][i] for i in range(batch_size)])
            )
            fake_noisy_latents_prev = self.step_to(
                sampler=self.sampler,
                inputs=fake_inputs,
                pred=self.concat([trajectory_pred[random_index[i]][i] for i in range(batch_size)]),
                s=next_timesteps,
                rng=rng
            )

            fake_inputs = self.set_timesteps(fake_inputs, next_timesteps)
            fake_inputs = self.set_noisy_latents(fake_inputs, fake_noisy_latents_prev)
            fake_noises = self.sample_noises(fake_inputs, rng)
            fake_inputs = self.set_noises(fake_inputs, fake_noises)
            fake_noisy_latents = self.add_partial_noise(self.schedule, fake_inputs, t=fake_timesteps)
            fake_inputs = self.set_latents(fake_inputs, latent_x0s)
            fake_inputs = self.set_timesteps(fake_inputs, fake_timesteps)
            fake_inputs = self.set_noisy_latents(fake_inputs, fake_noisy_latents)
        elif self.engine_config.fake_random_noise:
            fake_inputs = self.set_latents(fake_inputs, latent_x0s)
            fake_noises = self.sample_noises(fake_inputs, rng)
            fake_inputs = self.set_noises(fake_inputs, fake_noises)
            fake_inputs = self.set_timesteps(fake_inputs, fake_timesteps)
            fake_noisy_latents = self.add_noises(self.fake_schedule, fake_inputs)
            fake_inputs = self.set_noisy_latents(fake_inputs, fake_noisy_latents)
        else:
            fake_inputs = self.set_latents(fake_inputs, latent_x0s)
            fake_inputs = self.set_noises(fake_inputs, trajectory_xt[0])
            fake_inputs = self.set_timesteps(fake_inputs, fake_timesteps)
            fake_noisy_latents = self.add_noises(self.fake_schedule, fake_inputs)
            fake_inputs = self.set_noisy_latents(fake_inputs, fake_noisy_latents)

        if self.engine_config.uncond_fake_train_prob > 0.0:
            uncond_mask = torch.rand(batch_size, device=self.device, generator=rng.torch_cuda_generator) < self.engine_config.uncond_fake_train_prob
            fake_inputs = self.drop_condition(fake_inputs, neg_inputs, uncond_mask, rng)
        fake_pred = self.pred(self.fake_model, fake_inputs, neg_inputs)

        # convert and calculate loss
        if self.engine_config.fake_loss_type == "x_t":
            assert self.engine_config.fake_use_prev, "fake_use_prev must be True when fake_loss_type is x_t"
            pred = self.step_to(
                sampler=self.fake_sampler,
                inputs=fake_inputs,
                pred=fake_pred,
                s=next_timesteps,
                rng=rng
            )
            target = fake_noisy_latents_prev
        else:
            pred = self.convert_pred(
                schedule=self.fake_schedule,
                inputs=fake_inputs,
                pred=fake_pred,
                loss_type=self.engine_config.fake_loss_type,
            )
            target = self.convert_target(
                schedule=self.fake_schedule,
                inputs=fake_inputs,
                loss_type=self.engine_config.fake_loss_type,
            )

        loss = self.loss_fn(fake_inputs, pred, target, key_prefix="fake_losses")
        return loss

    def gen_training_step(
        self,
        pos_inputs: BaseForwardInput,
        neg_inputs: BaseForwardInput,
        rng: RandomState
    )-> torch.Tensor:
        batch_size = pos_inputs.batch_size
        gen_inputs = pos_inputs
        fake_inputs = deepcopy_with_tensor(pos_inputs)
        real_pos_inputs, real_neg_inputs = deepcopy_with_tensor(pos_inputs), deepcopy_with_tensor(neg_inputs)
        latent_x0s, trajectory_xt, trajectory_pred = self.trajectory

        # gen pred
        if self.score_timesteps is not None:
            self.sampling_timesteps.set_timesteps(seqlen=gen_inputs.seqlens, device=self.device)
            gen_timesteps = self.sample_timesteps(gen_inputs, self.sampling_timesteps, rng)
            random_index = self.sampling_timesteps.index(gen_timesteps)
            score_timesteps = self.sample_timesteps(fake_inputs, self.score_timesteps, rng)
        else:
            score_timesteps = self.sample_timesteps(gen_inputs, self.training_timesteps, rng)
            self.sampling_timesteps.set_timesteps(seqlen=gen_inputs.seqlens, device=self.device)
            sampling_timesteps = self.sampling_timesteps.timesteps
            if sampling_timesteps.ndim == 1:
                sampling_timesteps = sampling_timesteps[None, :]
            mask = sampling_timesteps > score_timesteps[:, None]
            random_index = mask.sum(dim=1).clamp(min=1) - 1
            gen_timesteps = self.sampling_timesteps.timesteps[random_index] if self.sampling_timesteps.timesteps.ndim == 1 else \
                self.sampling_timesteps.timesteps[torch.arange(batch_size, device=self.device), random_index]

        if self.engine_config.gen_use_real_data:
            latent_x0s = gen_inputs.latents
        gen_inputs = self.set_latents(gen_inputs, latent_x0s)
        gen_inputs = self.set_timesteps(gen_inputs, gen_timesteps)

        if self.engine_config.gen_use_trajectory:
            gen_inputs = self.set_noisy_latents(
                inputs=gen_inputs,
                noisy_latents=self.concat([trajectory_xt[random_index[i]][i] for i in range(batch_size)])
            )
        elif self.engine_config.gen_use_real_data or self.engine_config.gen_random_noise:
            gen_noises = self.sample_noises(gen_inputs, rng)
            gen_inputs = self.set_noises(gen_inputs, gen_noises)
            noisy_latents = self.add_noises(self.schedule, gen_inputs)
            gen_inputs = self.set_noisy_latents(gen_inputs, noisy_latents)
        else:
            gen_inputs = self.set_noises(gen_inputs, trajectory_xt[0])
            noisy_latents = self.add_noises(self.schedule, gen_inputs)
            gen_inputs = self.set_noisy_latents(gen_inputs, noisy_latents)

        gen_pred = self.pred(self.backbone, gen_inputs, neg_inputs)
        score_x0s = self.get_endpoint(self.schedule, gen_inputs, gen_pred)
        score_xts = self.step_to(self.sampler, gen_inputs, gen_pred, score_timesteps, rng)

        # score pred
        fake_ctx = torch.no_grad() if not self.engine_config.fake_grad_enabled else nullcontext()
        real_ctx = torch.no_grad() if not self.engine_config.real_grad_enabled else nullcontext()

        if isinstance(self.fake_model, FSDPModule):
            self.fake_model.unshard()  # trigger 1st all-gather earlier

        fake_inputs = self.set_latents(fake_inputs, latent_x0s)
        fake_inputs = self.set_noisy_latents(fake_inputs, score_xts)
        fake_inputs = self.set_timesteps(fake_inputs, score_timesteps)
        if self.engine_config.uncond_fake_eval:
            uncond_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            fake_inputs = self.drop_condition(fake_inputs, neg_inputs, uncond_mask, rng)

        with fake_ctx:
            fake_pred = self.pred(self.fake_model, fake_inputs, neg_inputs)
            fake_pred_x0s = self.get_endpoint(self.fake_schedule, fake_inputs, fake_pred)

        if isinstance(self.tea_model, FSDPModule):
            self.tea_model.unshard()  # trigger 1st all-gather earlier

        real_pos_inputs = self.set_latents(real_pos_inputs, latent_x0s)
        real_neg_inputs = self.set_latents(real_neg_inputs, latent_x0s)
        real_pos_inputs = self.set_noisy_latents(real_pos_inputs, score_xts)
        real_neg_inputs = self.set_noisy_latents(real_neg_inputs, score_xts)
        real_pos_inputs = self.set_timesteps(real_pos_inputs, score_timesteps)
        real_neg_inputs = self.set_timesteps(real_neg_inputs, score_timesteps)

        with real_ctx:
            real_pred = self.pred_cfg(self.tea_model, real_pos_inputs, real_neg_inputs)
            real_pred_x0s = self.get_endpoint(self.tea_schedule, real_pos_inputs, real_pred)

        # score distill loss
        losses = []
        for x0, fake, real in zip(score_x0s, fake_pred_x0s, real_pred_x0s):
            x0, fake, real = x0.double(), fake.double(), real.double()

            if self.engine_config.dmd_loss_type == "dmd":
                norm = torch.abs(x0 - real).mean()
                if self.engine_config.norm_clip_min is not None:
                    norm.clamp_min_(self.engine_config.norm_clip_min)
                loss = (real - fake) * (fake - x0) / norm.detach()

            elif self.engine_config.dmd_loss_type == "sim":
                diff = real - fake
                norm = ((diff ** 2).sum() + self.engine_config.phuber_c ** 2).sqrt()
                loss = (real - fake) * (fake - x0) / norm

            elif self.engine_config.dmd_loss_type == "sid":
                norm = torch.abs(x0 - real).mean()
                if self.engine_config.norm_clip_min is not None:
                    norm.clamp_min_(self.engine_config.norm_clip_min)
                loss = (real - fake) * ((real - x0) - self.engine_config.alpha * (real - fake)) / norm.detach()

            running.get_running_average_meter().put_scalar("running/dmd_loss/norm", norm.item())
            losses.append(loss)

        loss = self.loss_fn(gen_inputs, losses, key_prefix="dmd_losses")
        return loss
