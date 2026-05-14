"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer
from torch.distributed.fsdp import CPUOffload, CPUOffloadPolicy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, MixedPrecisionPolicy, ShardingStrategy, fully_shard

from project.engines.base_engine import FSDPConfig
from project.utils import comm
from project.utils.misc import to_torch_dtype
from project.utils.registry import Registry

DEVICE_MESH = None

FSDP_WRAP_POLICY_REGISTRY = Registry("FSDP_WRAP_POLICY")
FSDP_IGNORE_POLICY_REGISTRY = Registry("FSDP_IGNORE_POLICY")

logger = logging.getLogger()


def set_device_mesh(device_mesh):
    global DEVICE_MESH
    DEVICE_MESH = device_mesh


def get_device_mesh():
    return DEVICE_MESH


def override_mixed_precision(
    model: nn.Module,
    module_classes_to_ignore: Union[List, Tuple] = (),
    dtype: torch.dtype = torch.float32,
):
    if module_classes_to_ignore:
        for module in model.modules():
            if isinstance(module, module_classes_to_ignore):
                module.to(dtype)


def setup_fsdp(model: nn.Module, fsdp_config: FSDPConfig, model_name: str):
    if fsdp_config.version == 1:
        return setup_fsdp1(model, fsdp_config, model_name)
    elif fsdp_config.version == 2:
        return setup_fsdp2(model, fsdp_config, model_name)


def setup_fsdp2(model: nn.Module, fsdp_config: FSDPConfig, model_name: str):
    training = model.training
    params = sum([p.numel() for p in model.parameters()]) / 1e6
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad and training]) / 1e6
    logger.info(f"before FSDP2, {model_name} with total params: {params:.1f}M, "
                f"trainable params: {trainable_params:.1f}M")

    module_classes_to_wrap = FSDP_WRAP_POLICY_REGISTRY.get(fsdp_config.auto_wrap_policy) if fsdp_config.auto_wrap_policy is not None else ()
    _module_classes_to_wrap = module_classes_to_wrap[0] if module_classes_to_wrap and isinstance(module_classes_to_wrap[0], tuple) else module_classes_to_wrap
    _module_names_to_wrap = module_classes_to_wrap[1] if module_classes_to_wrap and isinstance(module_classes_to_wrap[0], tuple) else ()
    if fsdp_config.sharding_strategy == "FULL_SHARD":
        reshard_after_forward = True
    elif fsdp_config.sharding_strategy == "SHARD_GRAD_OP":
        reshard_after_forward = False
    elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
        reshard_after_forward = True
    elif fsdp_config.sharding_strategy == "_HYBRID_SHARD_ZERO2":
        reshard_after_forward = False
    else:
        raise ValueError(f"Unknown sharding strategy: {fsdp_config.sharding_strategy}")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=to_torch_dtype(fsdp_config.weight_dtype),
        reduce_dtype=to_torch_dtype(fsdp_config.reduce_dtype),
    )
    offload_policy = CPUOffloadPolicy() if fsdp_config.cpu_offload else None
    fsdp_kwargs = dict(
        mesh=DEVICE_MESH,
        reshard_after_forward=reshard_after_forward,
        shard_placement_fn=None,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )

    def _should_wrap(module: nn.Module) -> bool:
        if isinstance(module, _module_classes_to_wrap):
            return True
        layer_name = getattr(module, "layer_name", None)
        if layer_name and any(layer_name.endswith(n) for n in _module_names_to_wrap):
            return True
        return False

    def _apply_fsdp_recursive(module: nn.Module) -> None:
        for child in module.children():
            _apply_fsdp_recursive(child)
        if _should_wrap(module):
            fully_shard(module, **fsdp_kwargs)

    for child in model.children():
        _apply_fsdp_recursive(child)
    fully_shard(model, **fsdp_kwargs)

    logger.info(f"wrapped FSDP2 for {model_name}: {model}")
    params = sum([p.to_local().numel() for p in model.parameters()]) / 1e6
    trainable_params = sum([p.to_local().numel() for p in model.parameters() if p.requires_grad and training]) / 1e6
    logger.info(f"after FSDP2, {model_name} with total params: {params:.1f}M, "
                f"trainable params: {trainable_params:.1f}M")

    model.train(training)
    return model


def setup_fsdp1(model: nn.Module, fsdp_config: FSDPConfig, model_name: str):
    device_id = comm.get_local_rank()
    training = model.training

    params = sum([p.numel() for p in model.parameters()]) / 1e6
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad and training]) / 1e6
    logger.info(f"before FSDP1, {model_name} with total params: {params:.1f}M, "
                f"trainable params: {trainable_params:.1f}M")

    def _auto_wrap_policy(module, recurse, module_classes_to_wrap, **kwargs):
        if recurse:
            return True
        if module.layer_name.endswith(".base_layer") and isinstance(module, nn.Linear):
            return True
        if module.layer_name.endswith(".modules_to_save.default"):
            return True
        _module_classes_to_wrap = module_classes_to_wrap[0] if module_classes_to_wrap and isinstance(module_classes_to_wrap[0], tuple) else module_classes_to_wrap
        _module_names_to_wrap = module_classes_to_wrap[1] if module_classes_to_wrap and isinstance(module_classes_to_wrap[0], tuple) else ()
        return isinstance(module, _module_classes_to_wrap) or isinstance(module, LoraLayer) or \
            any([module.layer_name.endswith(name) for name in _module_names_to_wrap])

    module_classes_to_wrap = FSDP_WRAP_POLICY_REGISTRY.get(fsdp_config.auto_wrap_policy) if fsdp_config.auto_wrap_policy is not None else ()
    auto_wrap_policy = partial(_auto_wrap_policy, module_classes_to_wrap=module_classes_to_wrap)
    module_classes_to_ignore = FSDP_IGNORE_POLICY_REGISTRY.get(fsdp_config.auto_ignore_policy)
    override_mixed_precision(model, module_classes_to_ignore)

    model = FSDP(
        module=model,
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=to_torch_dtype(fsdp_config.weight_dtype),
            reduce_dtype=to_torch_dtype(fsdp_config.reduce_dtype),
            buffer_dtype=to_torch_dtype(fsdp_config.buffer_dtype),
            _module_classes_to_ignore=module_classes_to_ignore
        ),
        device_id=device_id,
        device_mesh=DEVICE_MESH,
        sync_module_states=fsdp_config.sync_module_states,
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        use_orig_params=fsdp_config.use_orig_params,
        forward_prefetch=fsdp_config.forward_prefetch,
        limit_all_gathers=fsdp_config.limit_all_gathers,
    )

    logger.info(f"wrapped FSDP1 for {model_name}: {model}")
    params = sum([p.numel() for p in model.parameters()]) / 1e6
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad and training]) / 1e6
    logger.info(f"after FSDP1, {model_name} with total params: {params:.1f}M, "
                f"trainable params: {trainable_params:.1f}M")

    model.train(training)
    return model


# fsdp ignore policy
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm

FSDP_IGNORE_POLICY_REGISTRY.register("default", (_BatchNorm,))
FSDP_IGNORE_POLICY_REGISTRY.register("all_norms", (GroupNorm, _BatchNorm, LayerNorm))
FSDP_IGNORE_POLICY_REGISTRY.register("all_norms_wo_gn", (_BatchNorm, LayerNorm))


# fsdp wrap policy
from project.models.wan2_1.t5 import T5SelfAttention
from project.models.wan2_1.model import WanAttentionBlock
from project.models.wan2_1.clip import AttentionBlock
from project.models.wan2_1.packed_model import PackedWanAttentionBlock
from project.models.wan2_1.causal_model import CausalWanAttentionBlock
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionBlock, Qwen2VLDecoderLayer

FSDP_WRAP_POLICY_REGISTRY.register("wan2_1_t5_wrap_policy", (T5SelfAttention,))
FSDP_WRAP_POLICY_REGISTRY.register("wan2_1_dit_wrap_policy", (WanAttentionBlock, PackedWanAttentionBlock, CausalWanAttentionBlock))
FSDP_WRAP_POLICY_REGISTRY.register("wan2_1_clip_wrap_policy", (AttentionBlock,))
FSDP_WRAP_POLICY_REGISTRY.register("qwen2_vl_wrap_policy", (Qwen2VLVisionBlock, Qwen2VLDecoderLayer))
FSDP_WRAP_POLICY_REGISTRY.register("qwen2_5_vl_wrap_policy", (Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer))
