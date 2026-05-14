"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.utils.registry import Registry

META_MODEL_REGISTRY = Registry("META_MODEL")

from .base_meta_model import BaseForwardInput, BaseMetaModel
from .wan2_1_t2v import Wan2_1_T2V
from .causal_wan2_1_t2v import CausalWan2_1_T2V
from .causal_wan2_1_t2v_df import CausalWan2_1_T2V_DF
from .causal_wan2_1_t2v_sf import CausalWan2_1_T2V_SF

META_MODEL_REGISTRY.register(Wan2_1_T2V)
META_MODEL_REGISTRY.register(CausalWan2_1_T2V)
META_MODEL_REGISTRY.register(CausalWan2_1_T2V_DF)
META_MODEL_REGISTRY.register(CausalWan2_1_T2V_SF)
