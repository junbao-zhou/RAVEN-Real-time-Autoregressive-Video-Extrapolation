"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.utils.registry import Registry

from .base_sampler import BaseSampler
from .ddim import DDIMSampler
from .consistency import ConsistencySampler
from .euler_maruyama import EulerMaruyamaSampler
from .tcd import TCDSampler

SAMPLER_REGISTRY = Registry("SAMPLER")

SAMPLER_REGISTRY.register(DDIMSampler)
SAMPLER_REGISTRY.register(ConsistencySampler)
SAMPLER_REGISTRY.register(EulerMaruyamaSampler)
SAMPLER_REGISTRY.register(TCDSampler)
