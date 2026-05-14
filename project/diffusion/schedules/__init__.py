"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.utils.registry import Registry

from .base_schedule import BaseSchedule
from .cos import CosineSchedule
from .lerp import LinearInterpolationSchedule
from .vp import DiscreteVariancePreservingSchedule

SCHEDULE_REGISTRY = Registry("SCHEDULE")

SCHEDULE_REGISTRY.register(CosineSchedule)
SCHEDULE_REGISTRY.register(LinearInterpolationSchedule)
SCHEDULE_REGISTRY.register(DiscreteVariancePreservingSchedule)
