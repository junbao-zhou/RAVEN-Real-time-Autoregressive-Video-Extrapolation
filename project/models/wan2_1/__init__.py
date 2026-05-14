"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.models import MODEL_REGISTRY

from .t5 import T5Encoder
from .tokenizers import HuggingfaceTokenizer
from .vae import WanVAE
from .clip import VisionTransformer
from .model import WanModel
from .packed_model import PackedWanModel
from .causal_model import CausalWanModel

MODEL_REGISTRY.register(T5Encoder)
MODEL_REGISTRY.register(HuggingfaceTokenizer)
MODEL_REGISTRY.register(WanVAE)
MODEL_REGISTRY.register(VisionTransformer)
MODEL_REGISTRY.register(WanModel)
MODEL_REGISTRY.register(PackedWanModel)
MODEL_REGISTRY.register(CausalWanModel)
