"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import bisect
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from project.utils.registry import Registry

LR_SCHEDULER_REGISTRY = Registry("LR_SCHEDULER")


class BaseLRScheduler(ABC):
    """ Base class for learning rate scheduler. """

    @abstractmethod
    def step(self, t):
        """ Update lr at each step. """
        pass


@LR_SCHEDULER_REGISTRY.register()
class WarmupLRScheduler(BaseLRScheduler):
    """ Simple lr scheduler with only warmup.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_t: Optional[int] = None,
        warmup_init_lr: float = 0.0,
        **kwargs
    ):
        self.optimizer = optimizer
        self.warmup_t = warmup_t
        self.warmup_init_lr = warmup_init_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, t):
        if self.warmup_t is not None and t < self.warmup_t:
            warmup_factor = (t + 1) / self.warmup_t
            for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = self.warmup_init_lr + warmup_factor * (lr - self.warmup_init_lr)
        else:
            for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = lr


@LR_SCHEDULER_REGISTRY.register()
class MultiStepLRScheduler(WarmupLRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_t: Optional[int] = None,
        warmup_init_lr: float = 0.0,
        **kwargs
    ):
        super().__init__(optimizer, warmup_t, warmup_init_lr)
        self.milestones = milestones
        self.gamma = gamma

    def step(self, t):
        if self.warmup_t is not None and t < self.warmup_t:
            super().step(t)
        else:
            exp = bisect.bisect_right(self.milestones, t)
            for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = lr * self.gamma ** exp


@LR_SCHEDULER_REGISTRY.register()
class ExponentialLRScheduler(WarmupLRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.1,
        min_lr: float = 0.0,
        warmup_t: Optional[int] = None,
        warmup_init_lr: float = 0.0,
        **kwargs
    ):
        super().__init__(optimizer, warmup_t, warmup_init_lr)
        self.gamma = gamma
        self.min_lr = min_lr

    def step(self, t):
        if self.warmup_t is not None and t < self.warmup_t:
            super().step(t)
        else:
            for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = max(self.min_lr, lr * self.gamma ** (t - self.warmup_t))
