"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.engines import ENGINE_REGISTRY

from .vbench_t2v import VBenchT2V

ENGINE_REGISTRY.register(VBenchT2V)

__all__ = [
    "VBenchT2V",
]
