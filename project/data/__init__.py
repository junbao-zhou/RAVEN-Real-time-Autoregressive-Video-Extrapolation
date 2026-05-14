"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from project.utils.registry import Registry

# datasets
from .openvid import CausalOpenVidDataset, CausalOpenVidDataset_DF
from .text import CausalTextVideoDataset

DATASET_REGISTRY = Registry("DATASET")

DATASET_REGISTRY.register(CausalOpenVidDataset)
DATASET_REGISTRY.register(CausalOpenVidDataset_DF)
DATASET_REGISTRY.register(CausalTextVideoDataset)

# dataloaders
from .dataloader import BaseDataloader

DATALOADER_REGISTRY = Registry("DATALOADER")

DATALOADER_REGISTRY.register(BaseDataloader)
