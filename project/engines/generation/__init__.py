"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.engines import ENGINE_REGISTRY

from .generate_t2v import GenerateT2V

ENGINE_REGISTRY.register(GenerateT2V)

__all__ = [
    "GenerateT2V",
]
