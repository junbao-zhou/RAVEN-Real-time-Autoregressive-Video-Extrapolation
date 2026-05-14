"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from project.diffusion.samplers import BaseSampler
from project.diffusion.schedules import BaseSchedule
from project.diffusion.timesteps import BaseTimesteps
from project.models.reward_models.base_reward_model import BaseRewardModel
from project.utils import comm
from project.utils.config import CfgNode
from project.utils.dataclass import Dataclass
from project.utils.misc import deepcopy_with_tensor
from project.utils.random import RandomState, local_seed, yield_seed
from project.utils.running import AverageMeter


@dataclass
class BaseForwardInput(Dataclass):
    xts: Union[List[Tensor], Tensor] = field(default_factory=list)
    timesteps: Tensor = field(default=None)
    latents: Union[List[Tensor], Tensor] = field(default_factory=list)
    noises: Union[List[Tensor], Tensor] = field(default_factory=list)
    batch_size: Tensor = field(default=None)
    seqlens: Tensor = field(default=None)
    prompts: List[str] = field(default=None)  # for reward function only, not used in forward pass


class BaseMetaModel(ABC):
    models: Dict[str, nn.Module]

    ###########################################################################
    # Setup
    ###########################################################################
    @abstractmethod
    def __init__(self, cfg: CfgNode):
        self.device = comm.get_device()
        self.average_meter = AverageMeter()

    @abstractmethod
    def setup_meta_model(self):
        """
        Set up meta model by loading necessary sub-models and building necessary components.
        Called at the end of engine init.
        """
        pass

    ###########################################################################
    # Training (Only meta models to be trained implement these)
    ###########################################################################
    def setup_dataloader(self):
        """
        Get dataloader configs specified by meta model.
        Only used during training.
        """
        kwargs = CfgNode()
        return kwargs

    def prepare_before_sync(
        self,
        batch: dict,
        rng: RandomState
    ) -> Tuple[BaseForwardInput, BaseForwardInput]:
        """ Prepare inputs for training (called before sync_inputs) """
        return self.prepare_training_inputs(batch, rng)

    def prepare_after_sync(
        self,
        batch: Union[dict, Tuple[BaseForwardInput, BaseForwardInput]],
        rng: RandomState
    ) -> Tuple[BaseForwardInput, BaseForwardInput]:
        """ Prepare inputs for training (called after sync_inputs) """
        return self.prepare_training_inputs(batch, rng)

    def prepare_training_inputs(
        self,
        batch: dict,
        rng: RandomState
    ) -> Tuple[BaseForwardInput, BaseForwardInput]:
        """ Prepare inputs for training (called before sync by default) """
        pass

    def drop_condition(
        self,
        pos_inputs: BaseForwardInput,
        neg_inputs: BaseForwardInput,
        uncond_mask: Tensor,
        rng: RandomState,
    ) -> BaseForwardInput:
        pass

    def loss_fn(
        self,
        inputs: BaseForwardInput,
        pred: Union[List[Tensor], Tensor],
        target: Optional[Union[List[Tensor], Tensor]] = None,
        key_prefix: str = "losses"
    ) -> torch.Tensor:
        pass

    ###########################################################################
    # Inference (Must be implemented for any meta model)
    ###########################################################################
    @abstractmethod
    def prepare_inference_inputs(
        self,
        batch: dict,
        infer_config: Dataclass,
        rngs: List[RandomState]
    ) -> Tuple[BaseForwardInput, BaseForwardInput]:
        """ Prepare both positive and negative inputs for inference """
        pass

    @abstractmethod
    def prepare_negative_inputs(
        self,
        batch: dict,
        pos_inputs: BaseForwardInput,
    ) -> BaseForwardInput:
        """ Prepare negative inputs for training or inference """
        pass

    @abstractmethod
    def infer(
        self,
        model: nn.Module,
        rng: Union[RandomState, List[RandomState]],
        pos_inputs: BaseForwardInput,
        neg_inputs: BaseForwardInput,
        return_trajectory: bool = False,  # should always set to True during training
        **kwargs  # utilities like diffusion timesteps, schedule and sampler
    ) -> Union[Union[List[Tensor], Tensor], Tuple[Union[List[Tensor], Tensor], List[Union[List[Tensor], Tensor]], List[Union[List[Tensor], Tensor]]]]:
        """ Inference on batched inputs """
        pass

    @abstractmethod
    def pred(
        self,
        model: nn.Module,
        inputs: BaseForwardInput,
        neg_inputs: BaseForwardInput = None,
    ) -> Union[List[Tensor], Tensor]:  # forward pass
        pass

    @abstractmethod
    def pred_cfg(
        self,
        model: nn.Module,
        pos_inputs: BaseForwardInput,
        neg_inputs: BaseForwardInput,
    ) -> Union[List[Tensor], Tensor]:  # forward pass
        pass

    ###########################################################################
    # General utils with default behavior, can be overridden if necessary
    ###########################################################################
    def concat(self, xs: List[Tensor]) -> Union[List[Tensor], Tensor]:
        return xs

    def merge_tensors(
        self,
        values: List[Union[List[Tensor], Tensor]],
    ) -> Union[List[Tensor], Tensor]:
        first = values[0]
        if isinstance(first, Tensor):
            return torch.cat(values, dim=0)
        if isinstance(first, list):
            return self.concat(sum(values, []))
        raise TypeError(f"Unsupported values type: {type(first)}")

    def mask_tensors(
        self,
        values: Union[List[Tensor], Tensor],
        mask: Union[Tensor, List[bool]],
    ) -> Union[List[Tensor], Tensor]:
        if isinstance(mask, Tensor):
            assert mask.ndim == 1, f"Mask should be 1D, got shape {tuple(mask.shape)}"
            mask = mask.to(dtype=torch.bool)
        else:
            mask = torch.tensor(mask, dtype=torch.bool)

        if isinstance(values, Tensor):
            return values[mask.to(device=values.device)]
        if isinstance(values, list):
            mask_list = mask.tolist()
            return [value for value, keep in zip(values, mask_list) if keep]
        raise TypeError(f"Unsupported values type: {type(values)}")

    def merge_inputs(
        self,
        *inputs: BaseForwardInput,
        detach: bool = True
    ) -> BaseForwardInput:
        assert len(inputs) > 0, "At least one input is required"
        typing = type(inputs[0])
        for input_i in inputs[1:]:
            assert isinstance(input_i, typing), f"Input types do not match: {typing} vs {type(input_i)}"

        merged = {}
        for f in fields(inputs[0]):
            if f.name == "__defaults__":
                continue
            values = [getattr(input_i, f.name) for input_i in inputs]
            if all(isinstance(v, Tensor) and v.ndim > 0 for v in values):
                merged[f.name] = torch.cat(values, dim=0)
            elif all(isinstance(v, list) for v in values):
                merged[f.name] = sum(values, [])

        merged_inputs = type(inputs[0])(
            **merged,
            batch_size=sum(input_i.batch_size for input_i in inputs)
        )
        if detach:
            merged_inputs = deepcopy_with_tensor(merged_inputs)
        return merged_inputs

    def mask_inputs(
        self,
        inputs: BaseForwardInput,
        mask: Union[Tensor, List[bool]],
        detach: bool = True
    ) -> BaseForwardInput:
        batch_size = int(inputs.batch_size.item())
        if isinstance(mask, Tensor):
            assert mask.ndim == 1, f"Mask should be 1D, got shape {tuple(mask.shape)}"
            mask = mask.to(device=inputs.batch_size.device, dtype=torch.bool)
        else:
            mask = torch.tensor(mask, device=inputs.batch_size.device, dtype=torch.bool)

        assert mask.numel() == batch_size, \
            f"Mask length {mask.numel()} does not match batch size {batch_size}"

        kept = int(mask.sum().item())
        mask_list = mask.tolist()
        masked = {}

        for f in fields(inputs):
            if f.name == "__defaults__":
                continue

            value = getattr(inputs, f.name)
            if isinstance(value, Tensor) and value.ndim > 0 and value.size(0) == batch_size:
                masked[f.name] = value[mask.to(device=value.device)]
            elif isinstance(value, list) and len(value) == batch_size:
                masked[f.name] = [v for v, keep in zip(value, mask_list) if keep]

        masked_inputs = type(inputs)(
            **masked,
            batch_size=inputs.batch_size.new_tensor(kept)
        )
        if detach:
            masked_inputs = deepcopy_with_tensor(masked_inputs)
        return masked_inputs

    ###########################################################################
    # Diffusion utils with default behavior, can be overridden if necessary
    ###########################################################################
    def set_latents(
        self,
        inputs: BaseForwardInput,
        latents: Union[List[Tensor], Tensor],
    ) -> BaseForwardInput:
        inputs.latents = latents
        return inputs

    def set_timesteps(
        self,
        inputs: BaseForwardInput,
        timesteps: Tensor,
    ) -> BaseForwardInput:
        inputs.timesteps = timesteps
        return inputs

    def set_noises(
        self,
        inputs: BaseForwardInput,
        noises: Union[List[Tensor], Tensor],
    ) -> BaseForwardInput:
        inputs.noises = noises
        return inputs

    def set_noisy_latents(
        self,
        inputs: BaseForwardInput,
        noisy_latents: Union[List[Tensor], Tensor],
    ) -> BaseForwardInput:
        inputs.xts = noisy_latents
        return inputs

    def sample_timesteps(
        self,
        inputs: BaseForwardInput,
        timesteps: BaseTimesteps,
        rng: RandomState,
        **kwargs
    ) -> Tensor:
        with local_seed(rng.seed):
            ts = timesteps.sample(
                size=(inputs.batch_size,),
                seqlens=inputs.seqlens,
                device=comm.get_device(),
                **kwargs
            )
        rng.seed = yield_seed(rng.seed)
        return ts

    def sample_noises(
        self,
        inputs: BaseForwardInput,
        rng: RandomState
    ) -> Union[List[Tensor], Tensor]:
        noises = [
            torch.empty_like(latent).normal_(generator=rng.torch_cuda_generator)
            for latent in inputs.latents
        ]
        return self.concat(noises)

    def add_noises(
        self,
        schedule: BaseSchedule,
        inputs: BaseForwardInput,
    ) -> Union[List[Tensor], Tensor]:
        xts = schedule.forward(
            x_0=inputs.latents,
            x_T=inputs.noises,
            t=inputs.timesteps
        )
        return xts

    def add_partial_noise(
        self,
        schedule: BaseSchedule,
        inputs: BaseForwardInput,
        t: Tensor
    ) -> Union[List[Tensor], Tensor]:
        return schedule.forward_from_prev(
            x_prev=inputs.xts,
            noise=inputs.noises,
            t=t,
            s=inputs.timesteps
        )

    def step_to(
        self,
        sampler: BaseSampler,
        inputs: BaseForwardInput,
        pred: Union[List[Tensor], Tensor],
        s: Tensor,
        rng: Union[RandomState, List[RandomState]]
    ) -> Union[List[Tensor], Tensor]:
        return sampler.step_to(
            pred=pred,
            x_t=inputs.xts,
            t=inputs.timesteps,
            s=s,
            rng=rng,
            seqlens=inputs.seqlens
        )

    def transition_kernel(
        self,
        sampler: BaseSampler,
        pred: Union[List[Tensor], Tensor],
        x_t: Union[List[Tensor], Tensor],
        t: Tensor,
        s: Tensor,
    ) -> Tuple[Union[List[Tensor], Tensor], Union[List[Tensor], Tensor]]:
        return sampler.transition_kernel(
            pred=pred,
            x_t=x_t,
            t=t,
            s=s,
        )

    def get_endpoint(
        self,
        schedule: BaseSchedule,
        inputs: BaseForwardInput,
        pred: Union[List[Tensor], Tensor],
    ) -> Union[List[Tensor], Tensor]:
        pred_x_0, _ = schedule.convert_from_pred(pred, inputs.xts, inputs.timesteps)
        return pred_x_0

    def convert_pred(
        self,
        schedule: BaseSchedule,
        inputs: BaseForwardInput,
        pred: Union[List[Tensor], Tensor],
        loss_type: str,
    ) -> Union[List[Tensor], Tensor]:
        pred_x_0, pred_x_T = schedule.convert_from_pred(
            pred=pred,
            x_t=inputs.xts,
            t=inputs.timesteps
        )
        return schedule.convert_to_pred(
            x_0=pred_x_0,
            x_T=pred_x_T,
            t=inputs.timesteps,
            pred_type=loss_type
        )

    def convert_target(
        self,
        schedule: BaseSchedule,
        inputs: BaseForwardInput,
        loss_type: str
    ) -> Union[List[Tensor], Tensor]:
        return schedule.convert_to_pred(
            x_0=inputs.latents,
            x_T=inputs.noises,
            t=inputs.timesteps,
            pred_type=loss_type
        )

    ###########################################################################
    # RLHF utils (Must be implemented for meta models to be trained with RLHF)
    ###########################################################################
    def gaussian_log_prob(
        self,
        sample: Union[List[Tensor], Tensor],
        mean: Union[List[Tensor], Tensor],
        std: Union[List[Tensor], Tensor],
        min_std: float = 1e-6,
    ) -> Tensor:
        log_probs = []
        for sample_i, mean_i, std_i in zip(sample, mean, std):
            std_i = std_i.to(device=sample_i.device, dtype=sample_i.dtype).clamp_min(min_std)
            normalized = (sample_i - mean_i) / std_i
            log_probs.append(-0.5 * (
                normalized.float().pow(2)
                + 2 * torch.log(std_i).float()
                + math.log(2 * math.pi)
            ).mean())
        return torch.stack(log_probs)

    def reward_fn(
        self,
        reward_model: nn.Module,
        inputs: BaseForwardInput,
        tensors: Union[List[Tensor], Tensor],
        **kwargs,
    ) -> Union[Dict[str, Tensor], Tensor]:
        if type(reward_model) is BaseRewardModel:
            reward_outputs = []
            for model_name, model in BaseRewardModel.collect_sub_reward_models(self.models):
                reward_outputs.append((model_name, model(tensors, inputs.prompts, **kwargs)))
            return BaseRewardModel.merge_reward_outputs(reward_outputs)
        return reward_model(tensors, inputs.prompts, **kwargs)
