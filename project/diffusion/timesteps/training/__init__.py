"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.diffusion.timesteps import TIMESTEP_REGISTRY

from .base_training_timestep import BaseTrainingTimesteps
from .logitnormal import LogitNormalTrainingTimesteps
from .mode import ModeTrainingTimesteps
from .uniform import UniformTrainingTimesteps

TIMESTEP_REGISTRY.register(LogitNormalTrainingTimesteps)
TIMESTEP_REGISTRY.register(UniformTrainingTimesteps)
TIMESTEP_REGISTRY.register(ModeTrainingTimesteps)
