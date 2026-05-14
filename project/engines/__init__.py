"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.utils.registry import Registry

ENGINE_REGISTRY = Registry("ENGINE")

from .base_engine import BaseEngine
from .generation import *
from .evaluation import *

from .diffusion_finetuning import DiffusionFinetuning
from .distribution_matching_distillation import DistributionMatchingDistillation
from .group_relative_policy_optimization import GroupRelativePolicyOptimization

ENGINE_REGISTRY.register(DiffusionFinetuning)
ENGINE_REGISTRY.register(DistributionMatchingDistillation)
ENGINE_REGISTRY.register(GroupRelativePolicyOptimization)
