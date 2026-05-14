"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.diffusion.timesteps import TIMESTEP_REGISTRY

from .base_sampling_timestep import BaseSamplingTimesteps
from .trailing import TrailingSamplingTimesteps

TIMESTEP_REGISTRY.register(TrailingSamplingTimesteps)
