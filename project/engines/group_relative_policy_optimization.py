"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import hashlib
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from project.diffusion.samplers import SAMPLER_REGISTRY, BaseSampler
from project.diffusion.schedules import SCHEDULE_REGISTRY, BaseSchedule
from project.diffusion.timesteps import TIMESTEP_REGISTRY, BaseSamplingTimesteps
from project.engines.base_engine import BaseEngine
from project.engines.diffusion_finetuning import DiffusionFinetuning
from project.meta_models import BaseForwardInput
from project.utils import comm
from project.utils import running
from project.utils.config import CfgNode
from project.utils.dataclass import Dataclass
from project.utils.misc import deepcopy_with_tensor
from project.utils.random import RandomState

class PerPromptStatTracker:
    def __init__(self, global_std: bool = False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()
        self._pending = []

    def _prompt_to_id(self, prompt: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest(),
            byteorder="big",
            signed=True,
        )

    def _update(self, prompt_ids, rewards):
        rewards = rewards.detach().cpu().to(dtype=torch.float32)
        unique_prompts = list(dict.fromkeys(prompt_ids))
        advantages = torch.empty_like(rewards, dtype=torch.float32)

        for prompt in unique_prompts:
            prompt_indices = [idx for idx, value in enumerate(prompt_ids) if value == prompt]
            prompt_rewards = rewards[prompt_indices]
            self.stats.setdefault(prompt, []).append(prompt_rewards)
            self.history_prompts.add(prompt)

        global_std = rewards.std(unbiased=False).clamp_min(1e-4) if self.global_std else None
        for prompt in unique_prompts:
            prompt_indices = [idx for idx, value in enumerate(prompt_ids) if value == prompt]
            prompt_rewards = rewards[prompt_indices]
            history = torch.cat(self.stats[prompt], dim=0)
            mean = history.mean()
            std = global_std if global_std is not None else history.std(unbiased=False).clamp_min(1e-4)
            advantages[prompt_indices] = (prompt_rewards - mean) / (std + 1e-5)
        return advantages

    def start_update(self, prompts, rewards):
        repeated_prompts = []
        for _ in range(int(rewards.shape[0])):
            repeated_prompts.extend(prompts)
        prompt_ids = torch.tensor(
            [self._prompt_to_id(prompt) for prompt in repeated_prompts],
            dtype=torch.int64,
            device=rewards.device,
        )
        flat_rewards = rewards.reshape(-1)
        world_size = comm.get_world_size()
        gathered_ids = [torch.empty_like(prompt_ids) for _ in range(world_size)]
        gathered_rewards = [torch.empty_like(flat_rewards) for _ in range(world_size)]
        id_handle = comm.all_gather(prompt_ids, gathered_ids, async_op=True)
        reward_handle = comm.all_gather(flat_rewards, gathered_rewards, async_op=True)
        self._pending.append((prompt_ids, gathered_ids, gathered_rewards, id_handle, reward_handle, rewards.shape, rewards.device))

    def finish_updates(self):
        local_rank = comm.get_rank()
        local_prompt_ids = []
        local_shapes = []
        local_devices = []
        synced_prompt_ids = []
        synced_rewards = []

        for prompt_ids, gathered_ids, gathered_rewards, id_handle, reward_handle, reward_shape, reward_device in self._pending:
            if id_handle is not None:
                id_handle.wait()
            if reward_handle is not None:
                reward_handle.wait()

            local_prompt_ids.append(prompt_ids)
            local_shapes.append(reward_shape)
            local_devices.append(reward_device)
            synced_prompt_ids.append(torch.cat(gathered_ids, dim=0))
            synced_rewards.append(torch.cat(gathered_rewards, dim=0))

        all_synced_prompt_ids = torch.cat(synced_prompt_ids, dim=0).cpu().tolist()
        all_synced_rewards = torch.cat(synced_rewards, dim=0)
        local_all_prompt_ids = torch.cat(local_prompt_ids, dim=0).cpu().tolist()
        global_prompt_counts = Counter(all_synced_prompt_ids)
        local_prompt_counts = Counter(local_all_prompt_ids)
        if local_prompt_counts:
            local_group_sizes = torch.tensor(
                [local_prompt_counts[prompt] for prompt in local_prompt_counts],
                dtype=torch.float32,
            )
            effective_group_sizes = torch.tensor(
                [global_prompt_counts[prompt] for prompt in local_prompt_counts],
                dtype=torch.float32,
            )
            effective_sync_group_sizes = effective_group_sizes / local_group_sizes.clamp_min(1.0)
            running.get_running_average_meter().put_scalar(
                "running/reward_local_group_size",
                local_group_sizes.mean(),
            )
            running.get_running_average_meter().put_scalar(
                "running/reward_effective_group_size",
                effective_group_sizes.mean(),
            )
            running.get_running_average_meter().put_scalar(
                "running/reward_effective_sync_group_size",
                effective_sync_group_sizes.mean(),
            )
        all_synced_advantages = self._update(all_synced_prompt_ids, all_synced_rewards)

        local_advantages = []
        global_offset = 0
        for prompt_ids, reward_shape, reward_device in zip(local_prompt_ids, local_shapes, local_devices):
            local_offset = global_offset + local_rank * prompt_ids.numel()
            local_advantages.append(
                all_synced_advantages[local_offset:local_offset + prompt_ids.numel()].to(
                    device=reward_device, dtype=torch.float32
                ).reshape(reward_shape)
            )
            global_offset += comm.get_world_size() * prompt_ids.numel()

        local_advantages = torch.cat(local_advantages, dim=0)
        self._pending.clear()
        self.clear()
        return local_advantages

    def clear(self):
        self.stats = {}


def select_reward_scores(tensor_rewards, reward_field, reward_aggregation, groups_per_infer, batch_size):
    if reward_field is None:
        return next(iter(tensor_rewards.values()))
    if isinstance(reward_field, str):
        assert reward_field in tensor_rewards, \
            f"reward_fn missing requested reward field: {reward_field}"
        return tensor_rewards[reward_field]

    if isinstance(reward_field, dict):
        reward_items = [(field, float(weight)) for field, weight in reward_field.items()]
    else:
        reward_items = [(field, 1.0) for field in reward_field]
    assert len(reward_items) > 0, "reward_field must not be empty"
    missing_fields = [field for field, _ in reward_items if field not in tensor_rewards]
    assert len(missing_fields) == 0, \
        f"reward_fn missing requested reward fields: {missing_fields}"
    normalizer = sum(abs(weight) for _, weight in reward_items)
    assert normalizer > 0.0, "reward_field weights must not all be zero"

    if reward_aggregation == "sum_then_normalize":
        return torch.stack([
            tensor_rewards[field].to(dtype=torch.float32) * weight
            for field, weight in reward_items
        ], dim=0).sum(dim=0) / normalizer
    if reward_aggregation == "normalize_then_sum":
        rewards = torch.stack([
            tensor_rewards[field].to(dtype=torch.float32)
            for field, _ in reward_items
        ], dim=0).reshape(len(reward_items), groups_per_infer, batch_size)
        weights = torch.tensor(
            [weight for _, weight in reward_items],
            device=rewards.device,
            dtype=rewards.dtype,
        ).view(-1, 1, 1)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
        return (((rewards - mean) / (std + 1e-5)) * weights).sum(dim=0).reshape(-1) / normalizer
    raise ValueError(f"Unsupported reward_aggregation: {reward_aggregation}")


@dataclass
class GroupRelativePolicyOptimizationConfig(Dataclass):
    # diffusion
    sampling_timestep: CfgNode = field(default=None)
    schedule: CfgNode = field(default=None)
    sampler: CfgNode = field(default=None)
    # general
    validation: List[CfgNode] = field(default_factory=list)
    save_before_train: bool = field(default=False)
    val_before_train: bool = field(default=True)
    val_backbone: bool = field(default=True)
    val_ema_model: bool = field(default=True)
    step_models: List[str] = field(default_factory=lambda: ["backbone"])
    # training
    training_steps: int = field(default=1000000)
    ga_steps: int = field(default=1)
    prepare_before_sync: bool = field(default=True)
    prepare_after_sync: bool = field(default=False)
    # grpo
    group_size: int = field(default=4)
    groups_per_infer: int = field(default=2)
    slices_per_step: int = field(default=1)
    optim_steps: int = field(default=1)
    reward_field: Union[str, List[str], Dict[str, float]] = field(default="MQ")
    reward_aggregation: str = field(default="sum_then_normalize")
    adv_clip_max: float = field(default=5.0)
    reverse_advantage: bool = field(default=False)
    reverse_policy_advantage: bool = field(default=False)
    beta: float = field(default=0.0)
    per_prompt_stat_tracking: bool = field(default=True)
    global_std: bool = field(default=False)
    skip_timesteps: int = field(default=1)
    random_policy_timestep: bool = field(default=False)
    policy_loss_scaling: float = field(default=1.0)
    policy_timestep_reweight: bool = field(default=False)
    policy_timestep_weight_eps: float = field(default=1e-6)


class GroupRelativePolicyOptimization(DiffusionFinetuning):
    backbone: nn.Module
    reward_model: nn.Module
    ref_model: Optional[nn.Module]

    sampling_timesteps: BaseSamplingTimesteps
    schedule: BaseSchedule
    sampler: BaseSampler

    def __init__(self, cfg: CfgNode):
        BaseEngine.__init__(self, cfg)
        self.setup_writer(cfg)

        self.engine_config = GroupRelativePolicyOptimizationConfig(**self.config["_config"])
        assert self.engine_config.group_size > 1, \
            f"group_size must be > 1 for GroupRelativePolicyOptimization, got {self.engine_config.group_size}"
        assert self.engine_config.groups_per_infer >= 1, \
            f"groups_per_infer must be >= 1, got {self.engine_config.groups_per_infer}"
        assert self.engine_config.group_size % self.engine_config.groups_per_infer == 0, \
            f"group_size {self.engine_config.group_size} must be divisible by groups_per_infer {self.engine_config.groups_per_infer}"
        assert self.engine_config.slices_per_step >= 1, \
            f"slices_per_step must be >= 1, got {self.engine_config.slices_per_step}"
        assert self.engine_config.optim_steps >= 1, \
            f"optim_steps must be >= 1, got {self.engine_config.optim_steps}"
        assert not (self.engine_config.optim_steps > 1 and self.engine_config.ga_steps > 1), \
            f"optim_steps ({self.engine_config.optim_steps}) and ga_steps ({self.engine_config.ga_steps}) cannot both be > 1"
        assert self.engine_config.adv_clip_max > 0.0, \
            f"adv_clip_max must be > 0, got {self.engine_config.adv_clip_max}"
        if self.engine_config.reverse_advantage:
            self.engine_config.reverse_policy_advantage = True
        assert self.engine_config.skip_timesteps >= 0, \
            f"skip_timesteps must be >= 0, got {self.engine_config.skip_timesteps}"
        assert self.engine_config.policy_timestep_weight_eps > 0.0, \
            f"policy_timestep_weight_eps must be > 0, got {self.engine_config.policy_timestep_weight_eps}"

        self.backbone = self.models["backbone"]
        self.reward_model = self.models["reward_model"]

        self.ref_model = self.models.get("ref_model")
        if self.engine_config.beta > 0.0:
            assert self.ref_model is not None, \
                "beta > 0 requires a reference model named 'ref_model' in cfg.models"

        self.configure_diffusion()
        self.per_prompt_stat_tracker = PerPromptStatTracker(global_std=self.engine_config.global_std) \
            if self.engine_config.per_prompt_stat_tracking else None

        self.sync_inputs = lambda inputs: [inputs]

    def configure_diffusion(self):
        timestep_cls = TIMESTEP_REGISTRY.get(self.engine_config.sampling_timestep["_class_name"])
        self.sampling_timesteps = timestep_cls(**self.engine_config.sampling_timestep["_config"])

        schedule_cls = SCHEDULE_REGISTRY.get(self.engine_config.schedule["_class_name"])
        self.schedule = schedule_cls(**self.engine_config.schedule["_config"])

        sampler_cls = SAMPLER_REGISTRY.get(self.engine_config.sampler["_class_name"])
        self.sampler = sampler_cls(schedule=self.schedule, **self.engine_config.sampler["_config"])

    def _build_group_rollouts(
        self,
        pos_inputs: BaseForwardInput,
        neg_inputs: BaseForwardInput,
        rng: RandomState
    ):
        groups_per_infer = self.engine_config.groups_per_infer
        num_infers = self.engine_config.group_size // groups_per_infer

        infer_rollouts = []

        self.sampling_timesteps.set_timesteps(seqlen=pos_inputs.seqlens, device=self.device)
        assert self.sampling_timesteps.timesteps.ndim == 1, \
            "GroupRelativePolicyOptimization currently expects 1D sampling timesteps"

        batch_size = int(pos_inputs.batch_size.item()) if isinstance(pos_inputs.batch_size, torch.Tensor) else int(pos_inputs.batch_size)
        for _ in range(num_infers):
            infer_pos_input_groups = [pos_inputs for _ in range(groups_per_infer)]
            infer_neg_input_groups = [neg_inputs for _ in range(groups_per_infer)]
            infer_pos_inputs = self.merge_inputs(*infer_pos_input_groups)
            infer_neg_inputs = self.merge_inputs(*infer_neg_input_groups)

            rollout_pos_inputs = deepcopy_with_tensor(infer_pos_inputs)
            rollout_neg_inputs = deepcopy_with_tensor(infer_neg_inputs)

            self.sampling_timesteps.set_timesteps(seqlen=rollout_pos_inputs.seqlens, device=self.device)
            sampling_noises = self.sample_noises(rollout_pos_inputs, rng)
            rollout_pos_inputs = self.set_noises(rollout_pos_inputs, sampling_noises)
            rollout_neg_inputs = self.set_noises(rollout_neg_inputs, sampling_noises)

            slice_batch_size = int(rollout_pos_inputs.batch_size.item()) \
                if isinstance(rollout_pos_inputs.batch_size, torch.Tensor) else int(rollout_pos_inputs.batch_size)
            rollout_t = [t.expand(slice_batch_size) for t in self.sampling_timesteps.timesteps]
            rollout_s = [self.sampling_timesteps.get_next_timesteps(t) for t in rollout_t]

            with torch.no_grad():
                latent_x0s, trajectory_xt, _ = self.infer(
                    model=self.backbone,
                    rng=rng,
                    pos_inputs=rollout_pos_inputs,
                    neg_inputs=rollout_neg_inputs,
                    sampling_timesteps=self.sampling_timesteps,
                    schedule=self.schedule,
                    sampler=self.sampler,
                    return_trajectory=True,
                )
                reward_output = self.reward_fn(
                    self.reward_model,
                    infer_pos_inputs,
                    latent_x0s,
                )
                if isinstance(reward_output, torch.Tensor):
                    reward_scores = reward_output
                else:
                    assert isinstance(reward_output, dict), \
                        f"reward_fn should return a tensor or dict, got {type(reward_output)}"
                    tensor_rewards = {
                        key: value
                        for key, value in reward_output.items()
                        if isinstance(value, torch.Tensor)
                    }
                    assert len(tensor_rewards) > 0, "reward_fn returned no tensor rewards"
                    for key, value in tensor_rewards.items():
                        if value.numel() == 0:
                            continue
                        running.get_running_average_meter().put_scalar(
                            f"rewards/{key}",
                            value.detach().to(dtype=torch.float32).mean(),
                            cnt=value.numel(),
                        )
                    reward_scores = select_reward_scores(
                        tensor_rewards,
                        self.engine_config.reward_field,
                        self.engine_config.reward_aggregation,
                        groups_per_infer,
                        batch_size,
                    )
                assert reward_scores.ndim == 1, \
                    f"selected reward should be a 1D tensor, got shape {tuple(reward_scores.shape)}"
                infer_rewards = reward_scores.to(device=self.device, dtype=torch.float32).reshape(groups_per_infer, batch_size)
                if self.per_prompt_stat_tracker is not None:
                    self.per_prompt_stat_tracker.start_update(pos_inputs.prompts, infer_rewards)

            infer_rollouts.append((
                infer_pos_inputs,
                infer_neg_inputs,
                (latent_x0s, trajectory_xt, rollout_t, rollout_s),
                infer_rewards,
            ))

        return infer_rollouts

    def training_loop(
        self,
        inputs: Tuple[BaseForwardInput, BaseForwardInput],
        rng: RandomState
    ) -> Dict[str, float]:
        pos_inputs, neg_inputs = inputs
        running.get_running_accumulator().update({
            "data/num_total_tokens": sum(pos_inputs.seqlens),
            "data/num_samples": pos_inputs.batch_size,
        })
        log_dict = dict()

        self.backbone.train(False)
        infer_rollouts = self._build_group_rollouts(pos_inputs, neg_inputs, rng)
        self.backbone.train(True)
        rewards = torch.cat([infer_rollout[3] for infer_rollout in infer_rollouts], dim=0)
        reward_std = rewards.std(dim=0, keepdim=True, unbiased=False)
        reward_max = rewards.max(dim=0).values
        reward_min = rewards.min(dim=0).values
        if self.per_prompt_stat_tracker is None:
            reward_mean = rewards.mean(dim=0, keepdim=True)
            advantages = (rewards - reward_mean) / (reward_std + 1e-5)
        else:
            advantages = self.per_prompt_stat_tracker.finish_updates()
        assert advantages.shape[0] == self.engine_config.group_size, \
            f"advantages first dim {advantages.shape[0]} must equal group_size {self.engine_config.group_size}"

        running.get_running_average_meter().update({
            "running/reward_mean": rewards.mean(),
            "running/reward_std": reward_std.mean(),
            "running/reward_group_max_mean": reward_max.mean(),
            "running/reward_group_min_mean": reward_min.mean(),
            "running/reward_gap": (reward_max - reward_min).mean(),
            "running/reward_advantage_abs": advantages.abs().mean(),
        })

        group_rollouts = []
        for infer_pos_inputs, infer_neg_inputs, infer_trajectory, _ in infer_rollouts:
            latent_x0s, trajectory_xt, trajectory_t, trajectory_s = infer_trajectory
            infer_batch_size = int(infer_pos_inputs.batch_size.item()) \
                if isinstance(infer_pos_inputs.batch_size, torch.Tensor) else int(infer_pos_inputs.batch_size)
            batch_size = infer_batch_size // self.engine_config.groups_per_infer
            for group_idx in range(self.engine_config.groups_per_infer):
                group_start = group_idx * batch_size
                group_end = group_start + batch_size
                group_mask = torch.zeros(infer_batch_size, device=self.device, dtype=torch.bool)
                group_mask[group_start:group_end] = True
                group_rollouts.append((
                    self.mask_inputs(infer_pos_inputs, group_mask),
                    self.mask_inputs(infer_neg_inputs, group_mask),
                    deepcopy_with_tensor((
                        self.mask_tensors(latent_x0s, group_mask),
                        [self.mask_tensors(xt, group_mask) for xt in trajectory_xt],
                        [self.mask_tensors(t, group_mask) for t in trajectory_t],
                        [self.mask_tensors(s, group_mask) for s in trajectory_s],
                    )),
                ))

        assert len(group_rollouts) == self.engine_config.group_size, \
            f"group_rollouts length {len(group_rollouts)} must equal group_size {self.engine_config.group_size}"
        num_policy_steps = len(group_rollouts[0][2][2]) - self.engine_config.skip_timesteps
        assert num_policy_steps > 0, \
            f"skip_timesteps {self.engine_config.skip_timesteps} must be smaller than the total number of rollout steps {len(group_rollouts[0][2][2])}"
        generator = rng.torch_cuda_generator if advantages.is_cuda else rng.torch_generator
        if self.engine_config.random_policy_timestep:
            random_step_indices = torch.randint(
                num_policy_steps,
                (len(group_rollouts),),
                device=advantages.device,
                generator=generator,
            )
            policy_slice_indices = [
                (group_idx, step_idx)
                for group_idx, step_idx in enumerate(random_step_indices.cpu().tolist())
            ]
        else:
            policy_slice_indices = [
                (group_idx, step_idx)
                for group_idx in range(len(group_rollouts))
                for step_idx in range(num_policy_steps)
            ]
        assert len(policy_slice_indices) % self.engine_config.optim_steps == 0, \
            f"total policy slices {len(policy_slice_indices)} must be divisible by optim_steps {self.engine_config.optim_steps}"
        if len(policy_slice_indices) > 1:
            permutation = torch.randperm(len(policy_slice_indices), device=advantages.device, generator=generator)
            policy_slice_indices = [policy_slice_indices[int(idx)] for idx in permutation.cpu().tolist()]

        if isinstance(self.backbone, FSDP) and (self.iter + 1) % self.engine_config.ga_steps != 0:
            ctx = self.backbone.no_sync()
        else:
            ctx = nullcontext()
        if isinstance(self.backbone, FSDPModule):
            self.backbone.set_requires_gradient_sync(requires_gradient_sync=(self.iter + 1) % self.engine_config.ga_steps == 0)

        with ctx:
            policy_slice_losses = []
            averaged_norm_dict = defaultdict(list)
            policy_slices_per_optim_step = len(policy_slice_indices) // self.engine_config.optim_steps
            for optim_step_idx in range(self.engine_config.optim_steps):
                chunk_start = optim_step_idx * policy_slices_per_optim_step
                chunk_end = chunk_start + policy_slices_per_optim_step
                optim_policy_slice_indices = policy_slice_indices[chunk_start:chunk_end]
                for slice_start in range(0, len(optim_policy_slice_indices), self.engine_config.slices_per_step):
                    slice_end = min(slice_start + self.engine_config.slices_per_step, len(optim_policy_slice_indices))
                    step_policy_slice_indices = optim_policy_slice_indices[slice_start:slice_end]
                    running.set_training_phase(running.TrainingPhase.IN_FORWARD)
                    step_loss = self.policy_training_step(
                        group_rollouts, advantages, step_policy_slice_indices
                    )
                    policy_slice_losses.append((step_loss.detach(), len(step_policy_slice_indices)))

                    running.set_training_phase(running.TrainingPhase.IN_BACKWARD)
                    (step_loss * (len(step_policy_slice_indices) / (len(optim_policy_slice_indices) * self.engine_config.ga_steps))).backward()

                running.set_training_phase(running.TrainingPhase.IN_OPTIMIZATION)
                norm_dict = self.optimize(self.engine_config.step_models)
                for key, value in norm_dict.items():
                    averaged_norm_dict[key].append(value)

            total_loss_weight = sum(loss_weight for _, loss_weight in policy_slice_losses)
            log_dict["train/loss"] = (
                torch.stack([loss * loss_weight for loss, loss_weight in policy_slice_losses]).sum() / total_loss_weight
            ).item()
            log_dict.update({
                key: sum(values) / len(values)
                for key, values in averaged_norm_dict.items()
            })
        return log_dict

    def policy_training_step(
        self,
        group_rollouts: List[Tuple],
        advantages: torch.Tensor,
        policy_slice_indices: List[Tuple[int, int]],
    ) -> torch.Tensor:
        pos_step_inputs = []
        neg_step_inputs = []
        next_xts = []
        step_ss = []
        step_advantages = []
        step_indices = []
        num_policy_steps = len(group_rollouts[0][2][2]) - self.engine_config.skip_timesteps

        for group_idx, step_idx in policy_slice_indices:
            group_pos_inputs, group_neg_inputs, group_trajectory = group_rollouts[group_idx]
            latent_x0s, trajectory_xt, trajectory_t, trajectory_s = group_trajectory
            pos_inputs_i = deepcopy_with_tensor(group_pos_inputs)
            neg_inputs_i = deepcopy_with_tensor(group_neg_inputs)
            pos_inputs_i = self.set_latents(pos_inputs_i, latent_x0s)
            neg_inputs_i = self.set_latents(neg_inputs_i, latent_x0s)
            pos_inputs_i = self.set_timesteps(pos_inputs_i, trajectory_t[step_idx])
            pos_inputs_i = self.set_noisy_latents(pos_inputs_i, trajectory_xt[step_idx])
            neg_inputs_i = self.set_timesteps(neg_inputs_i, trajectory_t[step_idx])
            neg_inputs_i = self.set_noisy_latents(neg_inputs_i, trajectory_xt[step_idx])
            pos_step_inputs.append(pos_inputs_i)
            neg_step_inputs.append(neg_inputs_i)
            next_xts.append(trajectory_xt[step_idx + 1] if step_idx + 1 < len(trajectory_xt) else latent_x0s)
            step_ss.append(trajectory_s[step_idx])
            group_advantages = advantages[group_idx].reshape(-1)
            step_advantages.append(group_advantages)
            step_indices.extend([step_idx] * int(group_advantages.numel()))

        pos_step_inputs = self.merge_inputs(*pos_step_inputs)
        neg_step_inputs = self.merge_inputs(*neg_step_inputs)
        next_xt = self.merge_tensors(next_xts)
        s = torch.cat(step_ss, dim=0)
        slice_advantages = torch.cat(step_advantages, dim=0)
        policy_advantages = -slice_advantages if self.engine_config.reverse_policy_advantage else slice_advantages
        slice_step_indices = step_indices

        if self.engine_config.beta > 0.0:
            ref_pos_step_inputs = deepcopy_with_tensor(pos_step_inputs)
            ref_neg_step_inputs = deepcopy_with_tensor(neg_step_inputs)

        if isinstance(self.backbone, FSDPModule):
            self.backbone.unshard()
        pred = self.pred(self.backbone, pos_step_inputs, neg_step_inputs)
        mean, std = self.transition_kernel(self.sampler, pred, pos_step_inputs.xts, pos_step_inputs.timesteps, s)
        if self.engine_config.policy_timestep_reweight:
            if isinstance(std, list):
                slice_timestep_weights = torch.stack([
                    std_i.detach().to(device=pos_step_inputs.timesteps.device, dtype=torch.float32).mean()
                    for std_i in std
                ])
            else:
                slice_timestep_weights = std.detach().to(device=pos_step_inputs.timesteps.device, dtype=torch.float32)
                slice_timestep_weights = slice_timestep_weights.reshape(slice_timestep_weights.shape[0], -1).mean(dim=1)
            dt = (
                (pos_step_inputs.timesteps.to(dtype=torch.float32) - s.to(dtype=torch.float32)).abs()
                / self.sampler._sigma_denominator()
            ).clamp_min(self.engine_config.policy_timestep_weight_eps)
            slice_timestep_weights = slice_timestep_weights / dt
        else:
            slice_timestep_weights = torch.ones_like(
                pos_step_inputs.timesteps,
                device=pos_step_inputs.timesteps.device,
                dtype=torch.float32,
            )

        clipped_advantages = torch.clamp(
            policy_advantages,
            -self.engine_config.adv_clip_max,
            self.engine_config.adv_clip_max,
        )
        pred_x_0 = self.get_endpoint(self.schedule, pos_step_inputs, pred)
        score_grad_coeff = self.sampler.transition_score_grad_coeff(pred, pos_step_inputs.xts, pos_step_inputs.timesteps, s)
        policy_losses = []
        policy_grad_norm_values = []
        for step_idx, advantage, timestep_weight, x0, coeff, next_xt_i, mean_i, std_i in zip(
            slice_step_indices,
            clipped_advantages,
            slice_timestep_weights,
            pred_x_0,
            score_grad_coeff,
            next_xt,
            mean,
            std,
        ):
            x0 = x0.float()
            std_i = std_i.clamp_min(1e-6)
            score_grad = 0.5 * coeff.detach() * (-advantage).detach() * (
                (next_xt_i - mean_i).detach() / std_i.detach().pow(2)
            )
            policy_grad = (
                2.0 * score_grad.detach()
                * float(step_idx < num_policy_steps)
                * timestep_weight.detach().to(dtype=score_grad.dtype)
                * self.engine_config.policy_loss_scaling
            )
            target = x0.detach() - score_grad.detach()
            policy_losses.append(
                F.mse_loss(x0, target.float(), reduction="none").to(dtype=x0.dtype)
                * float(step_idx < num_policy_steps)
                * timestep_weight.detach().to(dtype=x0.dtype)
            )
            policy_grad_norm_values.append(policy_grad.reshape(-1).norm().to(dtype=torch.float32))
        policy_loss = self.loss_fn(pos_step_inputs, policy_losses, key_prefix="policy_losses")
        total_loss = policy_loss * self.engine_config.policy_loss_scaling

        running.get_running_average_meter().put_scalar(
            "running/policy_loss",
            policy_loss.detach(),
            cnt=pos_step_inputs.batch_size,
        )
        running.get_running_average_meter().put_scalar(
            "running/policy_timestep_weight",
            slice_timestep_weights.detach().mean(),
            cnt=pos_step_inputs.batch_size,
        )
        running.get_running_average_meter().put_scalar(
            "running/policy_loss/x0_grad_norm",
            torch.stack(policy_grad_norm_values).mean(),
            cnt=pos_step_inputs.batch_size,
        )

        if self.engine_config.beta > 0.0:
            with torch.no_grad():
                if isinstance(self.ref_model, FSDPModule):
                    self.ref_model.unshard()
                ref_pred = self.pred_cfg(self.ref_model, ref_pos_step_inputs, ref_neg_step_inputs)
                ref_mean, _ = self.transition_kernel(self.sampler, ref_pred, pos_step_inputs.xts, pos_step_inputs.timesteps, s)
            kl_losses = []
            for x0, coeff, mean_i, ref_mean_i, std_i in zip(
                pred_x_0,
                score_grad_coeff,
                mean,
                ref_mean,
                std,
            ):
                x0 = x0.float()
                kl_score_grad = 0.5 * coeff.detach() * (
                    (mean_i - ref_mean_i).detach() / std_i.detach().pow(2)
                )
                kl_target = x0.detach() - kl_score_grad.detach()
                kl_losses.append(
                    F.mse_loss(x0, kl_target.float(), reduction="none").to(dtype=x0.dtype)
                )
            kl_loss = self.loss_fn(pos_step_inputs, kl_losses, key_prefix="kl_losses")
            total_loss = total_loss + self.engine_config.beta * kl_loss
            running.get_running_average_meter().put_scalar(
                "running/kl_loss",
                kl_loss.detach(),
                cnt=pos_step_inputs.batch_size,
            )
        return total_loss
