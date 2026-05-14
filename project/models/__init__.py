"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

LORA_CUSTOM_MODULE_MAPPING_SRC = Registry("LORA_CUSTOM_MODULE_MAPPING_SRC")
LORA_CUSTOM_MODULE_MAPPING_TGT = Registry("LORA_CUSTOM_MODULE_MAPPING_TGT")

from . import wan2_1
from .reward_models import (
    BaseRewardModel,
    UnifiedRewardQwenPointScoreRewardModel,
    VBenchAestheticRewardModel,
    VBenchBackgroundConsistencyRewardModel,
    VBenchImagingRewardModel,
    VBenchMotionSmoothnessRewardModel,
    VideoAlignRewardModel,
    VBenchRAFTRewardModel,
    VBenchSubjectConsistencyRewardModel,
)

MODEL_REGISTRY.register(BaseRewardModel)
MODEL_REGISTRY.register(UnifiedRewardQwenPointScoreRewardModel)
MODEL_REGISTRY.register(VBenchAestheticRewardModel)
MODEL_REGISTRY.register(VBenchBackgroundConsistencyRewardModel)
MODEL_REGISTRY.register(VBenchImagingRewardModel)
MODEL_REGISTRY.register(VBenchMotionSmoothnessRewardModel)
MODEL_REGISTRY.register(VideoAlignRewardModel)
MODEL_REGISTRY.register(VBenchRAFTRewardModel)
MODEL_REGISTRY.register(VBenchSubjectConsistencyRewardModel)
