"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import builtins
import functools
import gc
import hashlib
import itertools
import logging
import os
import json
import pickle
import random
import re
import sys
import threading
import time
import uuid
from abc import ABC
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from accelerate import init_empty_weights
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
                                                     set_optimizer_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import _set_pg_timeout
from torch.distributed.fsdp import FSDPModule, FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType

from project.data import DATALOADER_REGISTRY, BaseDataloader
from project.models import LORA_CUSTOM_MODULE_MAPPING_SRC, LORA_CUSTOM_MODULE_MAPPING_TGT, MODEL_REGISTRY
from project.utils import comm, fs
from project.utils.activation_offload import enable_activation_offloading
from project.utils.config import CfgNode
from project.utils.dataclass import Dataclass
from project.utils.file_io import maybe_download, maybe_upload
from project.utils.lr_scheduler import LR_SCHEDULER_REGISTRY, BaseLRScheduler
from project.utils.mfu import enable_flops_accumulate, get_mfu, register_flops_hook
from project.utils.misc import to_torch_dtype
from project.utils.running import get_running_accumulator
from project.utils.tracker import init_writer

logger = logging.getLogger()

if TYPE_CHECKING:
    # During the static analysis phase (when writing code), let the IDE think that DefaultEngine inherits from BaseMetaModel
    # This way, self.base_func() will be highlighted
    from project.meta_models import BaseMetaModel
    _Base = BaseMetaModel
else:
    # At runtime, the base class is ABC, and the real inheritance relationship is dynamically modified in __new__.
    _Base = ABC


@dataclass
class FSDPConfig(Dataclass):
    enabled: bool = field(default=False)
    version: int = field(default=2)
    sharding_strategy: str = field(default="HYBRID_SHARD")
    auto_wrap_policy: str = field(default=None)
    auto_ignore_policy: str = field(default="default")
    weight_dtype: str = field(default="bfloat16")  # set by model config
    reduce_dtype: str = field(default="bfloat16")
    buffer_dtype: str = field(default="float32")
    sync_module_states: bool = field(default=True)
    cpu_offload: bool = field(default=False)
    use_orig_params: bool = field(default=False)
    forward_prefetch: bool = field(default=False)
    limit_all_gathers: bool = field(default=True)


@dataclass
class AutoCastConfig(Dataclass):
    enabled: bool = field(default=False)
    dtype: str = field(default="bfloat16")
    cache_enabled: bool = field(default=True)


@dataclass
class LRSchedulerConfig(Dataclass):
    enabled: bool = field(default=False)
    _class_name: str = field(default=None)
    _config: CfgNode = field(default=None)


@dataclass
class OptimizerConfig(Dataclass):
    _class_name: str = field(default=None)
    _config: CfgNode = field(default=None)
    param_groups: List[str] = field(default=None)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class EMAConfig(Dataclass):
    enabled: bool = field(default=False)
    decay: float = field(default=0.9995)
    weight: str = field(default=None)


@dataclass
class LoRAConfig(Dataclass):
    enabled: bool = field(default=False)
    r: int = field(default=256)
    lora_alpha: int = field(default=256)
    weight: str = field(default=None)
    target_modules: List[str] = field(default_factory=list)
    custom_module_mapping: Dict[str, str] = field(default_factory=dict)
    modules_to_save: List[str] = field(default_factory=list)
    rank_pattern: Dict[str, int] = field(default_factory=dict)
    alpha_pattern: Dict[str, int] = field(default_factory=dict)
    merge: bool = field(default=False)
    save_peft: bool = field(default=False)
    save_merged: bool = field(default=False)


@dataclass
class ModelConfig(Dataclass):
    _class_name: str = field(default=None)
    _config: CfgNode = field(default_factory=CfgNode)
    pretrained_model_name_or_path: str = field(default=None)
    weight: str = field(default=None)
    weight_dtype: str = field(default="bfloat16")
    meta_init: bool = field(default=True)
    training_state: bool = field(default=False)
    requires_grad: bool = field(default=None)
    mfu_enabled: bool = field(default=False)
    grad_enabled: bool = field(default=False)
    autocast: AutoCastConfig = field(default_factory=AutoCastConfig)
    fsdp: Dict[str, Any] = field(default_factory=dict)  # did not specify FSDPConfig here because we use MetaModelConfig.fsdp as default, not FSDPConfig's
    optimizer: List[OptimizerConfig] = field(default_factory=list)
    ema: EMAConfig = field(default_factory=EMAConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    gradient_checkpointing: bool = field(default=False)
    gc_start_idx: int = field(default=0)
    gc_step: int = field(default=1)
    activation_offloading: bool = field(default=False)
    wrapped_func: List[str] = field(default_factory=lambda: ["forward"])
    trace_backward: bool = field(default=False)
    clip_grad_value: Optional[float] = field(default=None)
    clip_grad_norm: float = field(default=1000.)


@dataclass
class EnvironmentConfig(Dataclass):
    benchmark: bool = field(default=False)
    deterministic: bool = field(default=False)
    hybrid_gpu_num: int = field(default=-1)  # num_gpus per group in fsdp hybrid shard, -1 indicates gpus per node
    fsdp_default: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class PersistenceConfig(Dataclass):
    seed: int = field(default=1019)
    output_dir: str = field(default=None)  # local dir
    save_dir: str = field(default=None)  # remote dir
    proj_name: str = field(default=None)
    exp_name: str = field(default=None)
    tracker_backend: str = field(default="wandb")
    save_interval: int = field(default=1000)
    save_start_iter: int = field(default=0)
    val_interval: int = field(default=1000)
    val_start_iter: int = field(default=0)
    log_interval: int = field(default=10)
    save_dcp: bool = field(default=False)
    resume_dir: str = field(default=None)  # optional, useful when you want to resume from another experiment
    resume_iter: int = field(default=None)  # optional, useful when you want to resume from a specific iteration
    git_commit: str = field(default=None)  # auto filled by main.py
    git_clean: bool = field(default=None)  # auto filled by main.py
    max_to_keep: int = field(default=-1)  # maximum number of checkpoints to keep

@dataclass
class DefaultEngineConfig(Dataclass):
    _class_name: str = field(default=None)
    _config: CfgNode = field(default=None)
    environment: CfgNode = field(default_factory=CfgNode)
    persistence: CfgNode = field(default_factory=CfgNode)
    dataloader: CfgNode = field(default=None)
    meta_model: CfgNode = field(default=None)
    models: Dict[str, ModelConfig] = field(default_factory=dict)


def convert_dcp_to_torch_save(weight):
    if os.path.isdir(weight):
        if comm.get_rank() == 0 or (comm.get_local_rank() == 0 and not fs.is_mnt_path(weight)):
            target_path = os.path.join(weight, "all_model_states.pt")
            if not os.path.exists(target_path):
                logger.info(f"found dcp dir {weight}, try converting to torch save")
                tmp_name = f"_all_model_states_{uuid.uuid4().hex[:8]}.pt"
                tmp_path = os.path.join(weight, tmp_name)
                dcp_to_torch_save(weight, tmp_path)

                if not os.path.exists(target_path):
                    try:
                        os.rename(tmp_path, target_path)
                    except Exception:
                        pass

                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        elif fs.is_mnt_path(weight):
            while not os.path.exists(os.path.join(weight, "all_model_states.pt")):
                time.sleep(1)
        comm.barrier()
        weight = os.path.join(weight, "all_model_states.pt")
    return weight


def dcp_save(state_dict, first_row_group, tmp_dir, tgt_dir):
    comm.barrier(first_row_group)
    if dist.get_rank(first_row_group) >= 0:
        dcp.save(state_dict, storage_writer=dcp.FileSystemWriter(tmp_dir), process_group=first_row_group)

    if fs.is_mnt_path(tmp_dir):
        if comm.get_rank(first_row_group) == 0:
            fs.copy(tmp_dir, tgt_dir)
    else:
        if os.path.exists(tmp_dir) and comm.get_local_rank() == 0:
            fs.copy(tmp_dir, tgt_dir)

    del state_dict
    torch.cuda.empty_cache()
    comm.barrier(first_row_group)


class BaseEngine(_Base):
    models: Dict[str, nn.Module]
    optimizers: Dict[str, torch.optim.Optimizer]
    lr_schedulers: Dict[str, BaseLRScheduler]
    dataloader: BaseDataloader

    def __new__(cls, cfg: CfgNode):
        # this __new__ function will initialize meta model class before engine class
        # such that engine can access, inherit or override meta model function conveniently
        from project.meta_models import META_MODEL_REGISTRY
        assert "_class_name" in cfg["meta_model"], "You must specify a meta model for engine via _class_name."
        meta_model_class = META_MODEL_REGISTRY.get(cfg["meta_model"]["_class_name"])
        return super().__new__(type("MixedEngine", (cls, meta_model_class), {}))

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        self.config = DefaultEngineConfig(**cfg)

        self.setup_environment()
        self.configure_persistence()
        self.build_dataloader()
        self.build_models()
        self.build_optimizers()
        self.setup_meta_model()

    def get_mfu(self, show: bool = False):
        torch.cuda.synchronize()
        iter_time = self.mfu_start.elapsed_time(self.mfu_end) / 1e3
        return get_mfu(iter_time, self.models_with_flops_state, show=show)

    def setup_environment(self):
        self.env_config = EnvironmentConfig(**self.config.environment)

        torch.cuda.set_device(self.device)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # set the correct cuda visible devices (using pci order)
        torch.cuda.empty_cache()  # clear cache before training
        torch.backends.cudnn.benchmark = self.env_config.benchmark
        if self.env_config.deterministic:
            torch.use_deterministic_algorithms(self.env_config.deterministic)
            # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # distributed
        dist.init_process_group(
            backend="nccl" if not self.env_config.fsdp_default.cpu_offload else "cuda:nccl,cpu:gloo",
            rank=comm.get_rank(),
            world_size=comm.get_world_size(),
            timeout=timedelta(minutes=60),
            # device_id=comm.get_local_rank()  # NOTE: bound device id cause gloo new group hang
        )

        if "HYBRID_SHARD" in self.env_config.fsdp_default.sharding_strategy:
            hybrid_gpu_num = self.env_config.hybrid_gpu_num
            if hybrid_gpu_num == -1:
                hybrid_gpu_num = comm.get_local_world_size()  # hybrid shard
            hybrid_gpu_num = min(comm.get_local_world_size(), hybrid_gpu_num)
            assert comm.get_world_size() % hybrid_gpu_num == 0, f"world_size {comm.get_world_size()} must be divisible by hybrid_gpu_num {hybrid_gpu_num}"

            from project.distributed.fsdp import set_device_mesh
            device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(comm.get_world_size() // hybrid_gpu_num, hybrid_gpu_num),
                mesh_dim_names=("dp", "fsdp")
            )
            _set_pg_timeout(timedelta(minutes=30), device_mesh.get_group(mesh_dim=0))
            _set_pg_timeout(timedelta(minutes=30), device_mesh.get_group(mesh_dim=1))
            set_device_mesh(device_mesh)

        # mfu
        self.models_with_flops_state = []
        self.mfu_start = torch.cuda.Event(enable_timing=True)
        self.mfu_end = torch.cuda.Event(enable_timing=True)

    def configure_persistence(self):
        self.persistence_config = PersistenceConfig(**self.config.persistence)
        if self.persistence_config.proj_name is None:
            self.persistence_config.proj_name = f"{self.config.meta_model['_class_name']}_{self.config._class_name}"

        self.output_dir = os.path.join(self.persistence_config.output_dir, self.persistence_config.proj_name, self.persistence_config.exp_name)
        fs.mkdir(self.output_dir)

        self.save_dir = self.output_dir
        if self.persistence_config.save_dir is not None:
            self.save_dir = os.path.join(self.persistence_config.save_dir, self.persistence_config.proj_name, self.persistence_config.exp_name)
            fs.mkdir(self.save_dir)

        # this is necessary in case several packages call logging first and break the configuration here
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        fmt = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            datefmt=datefmt,
            filename=f"{self.output_dir}/log_rank{comm.get_rank()}.txt",
            filemode="a"
        )

        if comm.get_local_rank() == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(fmt, datefmt))
            logger.addHandler(console_handler)

        # suppress redundant print from others
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

        logger.info("Global config:\n{}".format(self.config.dump()))
        logger.info(f"Git commit: {self.persistence_config.git_commit}, clean: {self.persistence_config.git_clean}")
        if comm.get_rank() == 0:
            self.config.save(f"{self.output_dir}/config.jsonc")

        states_dir = os.path.join(self.persistence_config.resume_dir or self.save_dir, "states")
        if not fs.exists(os.path.join(states_dir, "latest")):
            self.resume = None
            self.iter = 0
            random.seed(self.persistence_config.seed)
            np.random.seed(self.persistence_config.seed)
            torch.manual_seed(self.persistence_config.seed)
            torch.cuda.manual_seed(self.persistence_config.seed)
            return

        if self.persistence_config.resume_iter is None:
            latest = maybe_download(os.path.join(states_dir, "latest"), override=True)
            obj = pickle.loads(open(latest, "rb").read())
            step = obj["latest_iter"]
            get_running_accumulator().load_state_dict(obj["accumulator"])
        else:
            step = int(self.persistence_config.resume_iter)

        # state directories
        states_dir = os.path.join(states_dir, f"{step:07d}")
        models_dir = os.path.join(states_dir, "models")
        optimizers_dir = os.path.join(states_dir, "optimizers")
        dataloaders_dir = os.path.join(states_dir, "dataloaders")
        rng_states_dir = os.path.join(states_dir, "rng_states")

        # restore rng states right now
        rng_state_path = os.path.join(rng_states_dir, f"rank_{comm.get_rank()}.pkl")
        if fs.exists(rng_state_path):
            rng_state_path = maybe_download(rng_state_path)
            rng_state = pickle.loads(open(rng_state_path, "rb").read())
            random.setstate(rng_state["random_state"])
            np.random.set_state(rng_state["np_state"])
            torch.set_rng_state(rng_state["torch_state"])
            torch.cuda.set_rng_state(rng_state["torch_cuda_state"])

        def get_state(dir):
            paths = fs.listdir(dir)
            names = [os.path.splitext(os.path.basename(path))[0] for path in paths]
            return {name: path for name, path in zip(names, paths)}

        # resume dict
        self.resume = dict(
            models=get_state(models_dir),
            optimizers=get_state(optimizers_dir),
            dataloaders=dataloaders_dir
        )
        self.iter = step

    def build_dataloader(self):
        if self.config.dataloader is None:
            return

        ckpt_path = None
        if self.resume is not None:
            ckpt_path = self.resume["dataloaders"]
            logger.info(f"resume dataloader from {ckpt_path}")

        dataloader_cls = DATALOADER_REGISTRY.get(self.config.dataloader["_class_name"])
        meta_cfg = self.setup_dataloader()
        self.dataloader = dataloader_cls(
            meta_cfg=meta_cfg,
            dataset_cfg=self.config.dataloader["dataset_cfg"],
            ckpt_path=ckpt_path,
            **self.config.dataloader["_config"]
        )

    def build_models(self):
        def context_wrapper(*args, mfu_enabled: bool, grad_enabled: bool, autocast_config: AutoCastConfig):
            def wrapper(orig_func):
                if hasattr(orig_func, "__self__"):
                    func = orig_func.__func__
                else:
                    func = orig_func

                @functools.wraps(func)
                def wrapped(*args, **kwargs):
                    with ExitStack() as stack:
                        if mfu_enabled:
                            stack.enter_context(enable_flops_accumulate())
                        if grad_enabled is False:
                            stack.enter_context(torch.no_grad())
                        if autocast_config.enabled:
                            dtype = to_torch_dtype(autocast_config.dtype)
                            cache_enabled = autocast_config.cache_enabled
                            stack.enter_context(torch.autocast("cuda", dtype=dtype, cache_enabled=cache_enabled))
                        return func(*args, **kwargs)

                if hasattr(orig_func, "__self__"):
                    self = orig_func.__self__
                    return wrapped.__get__(self, self.__class__)
                else:
                    return wrapped

            if args and callable(args[0]):
                return wrapper(args[0])
            else:
                return wrapper

        def annotate_layer_name(model: nn.Module, model_name: str):
            for name, module in model.named_modules():
                assert not hasattr(module, "layer_name"), f"layer_name ({module.layer_name}) already exists " \
                                                          f"in {model_name}.{name} from {model_name}."
                module.layer_name = f"{model_name}.{name}" if name != "" else model_name

        def track_progress(module, args, output):  # for bwd debug usage
            def get_shapes(data):
                if data is None:
                    return "None"
                if isinstance(data, torch.Tensor):
                    return str(tuple(data.shape))
                if isinstance(data, (tuple, list)):
                    return f"({', '.join(get_shapes(item) for item in data)})"
                if isinstance(data, dict):
                    return "{" + ", ".join(f"{k}: {get_shapes(v)}" for k, v in data.items()) + "}"
                if is_dataclass(data):
                    return (
                        f"{type(data).__name__}("
                        + ", ".join(f"{f.name}={get_shapes(getattr(data, f.name))}" for f in dataclass_fields(data))
                        + ")"
                    )
                return str(type(data).__name__)

            def find_first_grad_tensor(data):
                if isinstance(data, torch.Tensor):
                    return data if data.requires_grad else None
                if isinstance(data, (tuple, list)):
                    for item in data:
                        tensor = find_first_grad_tensor(item)
                        if tensor is not None:
                            return tensor
                    return None
                if isinstance(data, dict):
                    for item in data.values():
                        tensor = find_first_grad_tensor(item)
                        if tensor is not None:
                            return tensor
                    return None
                if is_dataclass(data):
                    for f in dataclass_fields(data):
                        tensor = find_first_grad_tensor(getattr(data, f.name))
                        if tensor is not None:
                            return tensor
                return None

            def check_finite(grad_data):
                if not (torch.is_floating_point(grad_data) or torch.is_complex(grad_data)):
                    return
                finite_mask = torch.isfinite(grad_data)
                if finite_mask.all():
                    return
                nan_count = torch.isnan(grad_data).sum().item()
                inf_count = torch.isinf(grad_data).sum().item()
                raise FloatingPointError(
                    f"Non-finite gradient detected in {module.layer_name}: "
                    f"shape={tuple(grad_data.shape)}, dtype={grad_data.dtype}, "
                    f"nan_count={nan_count}, inf_count={inf_count}"
                )

            grad_tensor = find_first_grad_tensor(output)
            if grad_tensor is None:
                return
            args_shapes = get_shapes(args)
            output_shapes = get_shapes(output)

            def _hook(grad):
                check_finite(grad)
                logger.info(
                    f"✓ Completed: {module.layer_name}, args={args_shapes}, output={output_shapes}, grad={tuple(grad.shape)}"
                )
                return grad

            grad_tensor.register_hook(_hook)

        def setup_model(model: nn.Module, model_name: str, weight: str, lora_config: LoRAConfig) -> nn.Module:
            # load pretrained
            if weight is not None:
                weight = maybe_download(weight)
                logger.info(f"Loading {model_name} from {weight}")
                if weight.endswith(".safetensors.index.json"):
                    with open(weight, "r", encoding="utf-8") as f:
                        index = json.load(f)
                    loaded_keys = index["weight_map"].keys()
                    shard_files = list(dict.fromkeys(index["weight_map"].values()))
                    for shard_file in shard_files:
                        state_dict = safe_load(os.path.join(os.path.dirname(weight), shard_file), device="cpu")
                        model.load_state_dict(state_dict, strict=False, assign=True)
                        del state_dict
                        gc.collect()
                    model_keys = model.state_dict().keys()
                    msg = torch.nn.modules.module._IncompatibleKeys(
                        [key for key in model_keys if key not in loaded_keys],
                        [key for key in loaded_keys if key not in model_keys],
                    )
                else:
                    weight = convert_dcp_to_torch_save(weight)
                    if weight.endswith(".safetensor") or weight.endswith(".safetensors"):
                        state_dict = safe_load(weight, device="cpu")
                    elif weight.endswith(".pkl"):
                        state_dict = torch.load(weight, map_location="cpu")
                    else:
                        state_dict = torch.load(weight, map_location="cpu", mmap=True)

                    msg = model.load_state_dict(state_dict, strict=False, assign=True)

                logger.info(f"Loaded {model_name} from {weight} with missing keys: {msg.missing_keys}")
                logger.info(f"Loaded {model_name} from {weight} with unexpected keys: {msg.unexpected_keys}")

            # wrap lora
            if lora_config.enabled:
                peft_config = LoraConfig(
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    target_modules=lora_config.target_modules,
                    modules_to_save=lora_config.modules_to_save,
                    rank_pattern=lora_config.rank_pattern,
                    alpha_pattern=lora_config.alpha_pattern
                )

                if len(lora_config.custom_module_mapping) > 0:
                    custom_module_mapping_dict = {}
                    for src, tgt in lora_config.custom_module_mapping:
                        src_module = LORA_CUSTOM_MODULE_MAPPING_SRC.get(src)
                        tgt_module = LORA_CUSTOM_MODULE_MAPPING_TGT.get(tgt)
                        custom_module_mapping_dict[src_module] = tgt_module
                    peft_config._register_custom_module(custom_module_mapping_dict)

                model = get_peft_model(model, peft_config)

                if lora_config.weight is not None:
                    lora_weight = maybe_download(lora_config.weight)
                    lora_weight = convert_dcp_to_torch_save(lora_weight)
                    logger.info(f"Loading lora for {model_name} from {lora_config.weight}")
                    if lora_weight.endswith(".safetensor") or lora_weight.endswith(".safetensors"):
                        state_dict = safe_load(lora_weight, device="cpu")
                        msg = set_peft_model_state_dict(model, state_dict)
                        logger.info(f"Loading lora for {model_name} from {lora_weight} with unexpected keys: {msg.unexpected_keys}")
                    else:
                        state_dict = torch.load(lora_weight, map_location="cpu", mmap=True)
                        msg = model.load_state_dict(state_dict, strict=False, assign=True)
                        logger.info(f"Loading lora for {model_name} from {lora_weight} with unexpected keys: {msg.unexpected_keys}")
                        logger.info(f"Loading lora for {model_name} from {lora_weight} with missing keys: {msg.missing_keys}")

                if lora_config.save_peft and comm.get_rank() == 0:
                    peft_save_path = os.path.join(self.output_dir, f"{model_name}_peft.safetensors")
                    logger.info(f"saving peft lora for {model_name} to {peft_save_path}")
                    peft_state_dict = get_peft_model_state_dict(model, model.state_dict())
                    save_file(peft_state_dict, peft_save_path)
                comm.barrier()

                if lora_config.merge:
                    logger.info(f"merging lora for {model_name}")
                    model = model.merge_and_unload()
                    model.requires_grad_(True)

                    if lora_config.save_merged and comm.get_rank() == 0:
                        merged_save_path = os.path.join(self.output_dir, f"{model_name}_merged.pt")
                        logger.info(f"saving merged lora for {model_name} to {merged_save_path}")
                        merged_state_dict = model.state_dict()
                        torch.save(merged_state_dict, merged_save_path)
                    comm.barrier()

            return model

        def get_attribute(x, attr_str, default=None):
            """
            Get the attribute value of the given object.

            Parameters:
            x (object): The input Python object.
            attr_str (str): The attribute path, in the format of "a.b.c".

            Returns:
            Any type: The value of the corresponding attribute.
            """
            res, attrs = x, attr_str.split(".")
            while attrs:
                res = getattr(res, attrs.pop(0), None)
                if res is None:
                    return default
            return res

        self.models = defaultdict(lambda: None)
        for model_name, model_config in self.config.models.items():
            # build model
            model_cls = MODEL_REGISTRY.get(model_config._class_name)
            if model_config.pretrained_model_name_or_path is None:
                assert model_config._config is not None, f"model config must be provided to initialize {model_name} from scratch"
                init_context = init_empty_weights if model_config.meta_init else nullcontext
                with init_context():
                    model = model_cls(**model_config._config)
            else:
                assert hasattr(model_cls, "from_pretrained"), f"{model_name} class {model_cls} does not contain from_pretrained func"
                pretrained_model_name_or_path = maybe_download(model_config.pretrained_model_name_or_path)
                model = model_cls.from_pretrained(pretrained_model_name_or_path, **model_config._config)

            if isinstance(model, nn.Module):
                # setup model
                model = setup_model(model, model_name, model_config.weight, model_config.lora)

                # build ema model
                if model_config.ema.enabled:
                    assert model_config._config is not None, f"model config must be provided to setup ema model for {model_name}"
                    init_context = init_empty_weights if model_config.meta_init else nullcontext
                    with init_context():
                        ema_model: nn.Module = model_cls(**model_config._config)

                    # setup ema model
                    ema_weight = model_config.ema.weight if model_config.ema.weight is not None else model_config.weight
                    ema_model = setup_model(ema_model, f"{model_name}_ema", ema_weight, model_config.lora)

                # resume
                if model_config.optimizer and self.resume is not None:
                    resume_path = self.resume["models"][model_name]
                    ckpt_path = maybe_download(resume_path)
                    use_dcp = os.path.isdir(ckpt_path)
                    ckpt_path = convert_dcp_to_torch_save(ckpt_path)

                    state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)
                    if isinstance(model, PeftModel) and not use_dcp:  # lora
                        msg = set_peft_model_state_dict(model, state_dict)
                        assert len(msg.missing_keys) == 0, f"missing keys found when resuming lora for {model_name} from {resume_path}: {msg.missing_keys}"
                    else:
                        msg = model.load_state_dict(state_dict, strict=False, assign=True)
                        logger.info(f"Resuming {model_name} from {resume_path} with missing keys: {msg.missing_keys}")
                        logger.info(f"Resuming {model_name} from {resume_path} with unexpected keys: {msg.unexpected_keys}")

                    if model_config.ema.enabled:
                        resume_path = self.resume["models"][model_name + "_ema"]
                        ckpt_path = maybe_download(resume_path)
                        use_dcp = os.path.isdir(ckpt_path)
                        ckpt_path = convert_dcp_to_torch_save(ckpt_path)

                        ema_state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)
                        if isinstance(ema_model, PeftModel) and not use_dcp:  # lora
                            msg = set_peft_model_state_dict(ema_model, ema_state_dict)
                            logger.info(f"Resuming lora for {model_name}_ema from {resume_path} with unexpected keys: {msg.unexpected_keys}")
                        else:
                            msg = ema_model.load_state_dict(ema_state_dict, strict=False, assign=True)
                            logger.info(f"Resuming {model_name}_ema from {resume_path} with missing keys: {msg.missing_keys}")
                            logger.info(f"Resuming {model_name}_ema from {resume_path} with unexpected keys: {msg.unexpected_keys}")

                # check meta
                metas = [n for n, b in itertools.chain(model.named_parameters(), model.named_buffers()) if b.is_meta]
                assert not metas, f"{model_name} got meta tensor: {metas}"

                # wrap func and setup training
                model.train(model_config.training_state)
                if model_config.requires_grad is not None:
                    model.requires_grad_(model_config.requires_grad)
                annotate_layer_name(model, model_name)

                if model_config.ema.enabled:
                    ema_model.train(False).requires_grad_(False)
                    annotate_layer_name(ema_model, f"{model_name}_ema")

                for wrap_key in model_config.wrapped_func:
                    logger.info(f"Wrapping {model_name}.{wrap_key} with grad_enabled={model_config.grad_enabled}, "
                                f"autocast_enabled={model_config.autocast.enabled}")
                    setattr(model, wrap_key, context_wrapper(
                        mfu_enabled=model_config.mfu_enabled,
                        grad_enabled=model_config.grad_enabled,
                        autocast_config=model_config.autocast
                    )(get_attribute(model, wrap_key)))

                    if model_config.ema.enabled:
                        logger.info(f"Wrapping {model_name}_ema.{wrap_key} with grad_enabled=False, "
                                    f"autocast_enabled={model_config.autocast.enabled}")
                        setattr(ema_model, wrap_key, context_wrapper(
                            mfu_enabled=model_config.mfu_enabled,
                            grad_enabled=False,
                            autocast_config=model_config.autocast
                        )(get_attribute(ema_model, wrap_key)))

                # gradient checkpointing
                if not hasattr(model, "set_gradient_checkpointing"):
                    logger.info(f"found {model_name} does not implement set_gradient_checkpointing func, "
                                f"skip setting gradient_checkpointing={model_config.gradient_checkpointing}")
                elif not model_config.activation_offloading:
                    logger.info(f"Setting gradient_checkpointing={model_config.gradient_checkpointing} for {model_name}")
                    model.set_gradient_checkpointing(model_config.gradient_checkpointing, gc_start_idx=model_config.gc_start_idx, gc_step=model_config.gc_step)
                    if model_config.ema.enabled:
                        logger.info(f"Setting gradient_checkpointing={model_config.gradient_checkpointing} for {model_name}_ema")
                        ema_model.set_gradient_checkpointing(model_config.gradient_checkpointing, gc_start_idx=model_config.gc_start_idx, gc_step=model_config.gc_step)

                # move to gpu
                setattr(model, "weight_dtype", to_torch_dtype(model_config.weight_dtype))
                fsdp_config = self.env_config.fsdp_default.copy()
                fsdp_config.update({"weight_dtype": model_config.weight_dtype})
                fsdp_config.update(model_config.fsdp)

                if fsdp_config.enabled:
                    from project.distributed.fsdp import setup_fsdp
                    model = setup_fsdp(model, fsdp_config, model_name)
                    if model_config.activation_offloading:
                        logger.info(f"Enabling activation offloading for {model_name} with gradient_checkpointing={model_config.gradient_checkpointing}")
                        enable_activation_offloading(model, enable_ckpt=model_config.gradient_checkpointing)
                else:
                    model.to(device=self.device, dtype=to_torch_dtype(model_config.weight_dtype))

                if model_config.ema.enabled:
                    setattr(ema_model, "weight_dtype", to_torch_dtype(model_config.weight_dtype))
                    if fsdp_config.enabled:
                        from project.distributed.fsdp import setup_fsdp
                        ema_model = setup_fsdp(ema_model, fsdp_config, f"{model_name}_ema")
                        if model_config.activation_offloading:
                            logger.info(f"Enabling activation offloading for {model_name}_ema with gradient_checkpointing={model_config.gradient_checkpointing}")
                            enable_activation_offloading(ema_model, enable_ckpt=model_config.gradient_checkpointing)
                    else:
                        ema_model.to(device=self.device, dtype=to_torch_dtype(model_config.weight_dtype))

                # trace backward if needed to debug bwd
                if model_config.trace_backward:
                    logger.info(f"Tracking backward for {model_name}")
                    for module in model.modules():
                        module.register_forward_hook(track_progress)
                    if model_config.ema.enabled:
                        for module in ema_model.modules():
                            module.register_forward_hook(track_progress)

                # record model with mfu
                if model_config.mfu_enabled:
                    register_flops_hook(model, model_name)
                    self.models_with_flops_state.append(model)
                    if model_config.ema.enabled:
                        register_flops_hook(ema_model, f"{model_name}_ema")
                        self.models_with_flops_state.append(ema_model)

            self.models[model_name] = model
            logger.info(f"Building {model_name} done")
            if model_config.ema.enabled:
                self.models[f"{model_name}_ema"] = ema_model
                logger.info(f"Building {model_name}_ema done")

    def build_optimizers(self):
        self.optimizers = dict()
        self.lr_schedulers = dict()

        for model_name, model_config in self.config.models.items():
            if not model_config.optimizer:
                continue

            model = self.models[model_name]
            for idx, optim_config in enumerate(model_config.optimizer):
                logger.info(f"Building optimizer for {model_name} in group {idx}")
                optimizer_cls = getattr(torch.optim, optim_config._class_name)
                if optim_config.param_groups is not None:
                    params = [p for n, p in model.named_parameters() if p.requires_grad and any([pg in n for pg in optim_config.param_groups])]
                else:
                    params = [p for p in model.parameters() if p.requires_grad]

                self.optimizers[f"{model_name}_{idx}"] = optimizer_cls(params, **optim_config._config)

                if optim_config.lr_scheduler.enabled:
                    logger.info(f"Building lr_scheduler for {model_name} in group {idx}")
                    lr_scheduler_cls = LR_SCHEDULER_REGISTRY.get(optim_config.lr_scheduler._class_name)
                    self.lr_schedulers[f"{model_name}_{idx}"] = lr_scheduler_cls(
                        self.optimizers[f"{model_name}_{idx}"], **optim_config.lr_scheduler._config)
                    logger.info(f"Building lr_scheduler for {model_name} in group {idx} done")

                logger.info(f"Building optimizer for {model_name} in group {idx} done")

            if self.resume is not None:
                assert isinstance(model, FSDP) or isinstance(model, FSDPModule), f"{model_name} is not FSDP, not supported for training & resume"

                if isinstance(model, FSDP):
                    for idx in range(len(model_config.optimizer)):
                        logger.info(f"Resuming optimizer for {model_name} from {self.resume['optimizers'][f'{model_name}_{idx}']}")
                        ckpt_path = convert_dcp_to_torch_save(maybe_download(self.resume["optimizers"][f"{model_name}_{idx}"]))
                        optim_state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)

                        with FSDP.state_dict_type(
                            model,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
                            FullOptimStateDictConfig(rank0_only=False, offload_to_cpu=True),
                        ):
                            optim_state_dict = FSDP.optim_state_dict_to_load(model, self.optimizers[f"{model_name}_{idx}"], optim_state_dict)
                            self.optimizers[f"{model_name}_{idx}"].load_state_dict(optim_state_dict)
                            logger.info(f"Resuming optimizer for {model_name} from {self.resume['optimizers'][f'{model_name}_{idx}']} done")

                elif isinstance(model, FSDPModule):
                    for idx in range(len(model_config.optimizer)):
                        logger.info(f"Resuming optimizer for {model_name} from {self.resume['optimizers'][f'{model_name}_{idx}']}")
                        optim_state_dict = get_optimizer_state_dict(
                            model=model,
                            optimizers=self.optimizers[f"{model_name}_{idx}"],
                            options=StateDictOptions(full_state_dict=False, cpu_offload=True)  # it seems that dcp.load could directly assign weights to optim_state_dict in gpu w/o cpu offload
                        )
                        dcp.load(
                            optim_state_dict,
                            storage_reader=dcp.FileSystemReader(maybe_download(self.resume["optimizers"][f"{model_name}_{idx}"]))
                        )
                        set_optimizer_state_dict(
                            model=model,
                            optimizers=self.optimizers[f"{model_name}_{idx}"],
                            optim_state_dict=optim_state_dict,
                            options=StateDictOptions(strict=True)
                        )
                        logger.info(f"Resuming optimizer for {model_name} from {self.resume['optimizers'][f'{model_name}_{idx}']} done")

    def save(self, save_before_train: bool = False):
        def _cleanup_old_checkpoints(save_dir: str, max_to_keep: int, sync: bool = True):
            if max_to_keep == -1:
                return
            if sync:
                comm.barrier()
            states_dir = os.path.join(save_dir, "states")
            all_entries = fs.listdir(states_dir)
            ckpt_dirs = sorted(
                [e for e in all_entries if re.match(r'^\d{7}$', os.path.basename(e))],
                key=lambda x: int(os.path.basename(x))
            )

            if len(ckpt_dirs) > max_to_keep:
                dirs_to_delete = ckpt_dirs[:len(ckpt_dirs) - max_to_keep]
                for d in dirs_to_delete:
                    logger.info(f"Removing old checkpoint: {d}")
                    fs.remove(d, sync=sync)
                    logger.info(f"Removed old checkpoint: {d}")

        def _save(fut: torch.Future, state_dicts: List[Tuple[Dict, str, str]], save_dir: str, obj: dict):
            if fut is not None:
                fut.wait()

            def _fn(fut: torch.Future, state_dicts: List[Tuple[Dict, str, str]], save_dir: str, obj: dict):
                for (state_dict, filename, directory) in state_dicts:
                    maybe_upload(state_dict, filename, directory)
                    del state_dict
                fut.set_result((save_dir, obj))

            def _cb(fut):
                save_dir, obj = fut.value()
                logger.info(f"Saving state at iter {obj['latest_iter']} async done")
                if not save_before_train:
                    maybe_upload(obj, "latest", save_dir)
                    logger.info(f"Updated {save_dir}/latest to {obj['latest_iter']}")
                    _cleanup_old_checkpoints(
                        self.save_dir,
                        self.persistence_config.max_to_keep,
                        sync=False
                    )

            fut = torch.futures.Future()
            fut.add_done_callback(_cb)
            worker = threading.Thread(target=_fn, args=(fut, state_dicts, save_dir, obj))
            worker.start()
            return fut

        if (
            (self.iter + 1) >= self.persistence_config.save_start_iter
            and (self.iter + 1) % self.persistence_config.save_interval == 0
        ) or save_before_train:
            logger.info(f"Saving state at iter {self.iter + 1}...")
            torch.cuda.empty_cache()

            states_dir = os.path.join(self.save_dir, f"states/{(self.iter + 1):07d}")
            models_dir = os.path.join(states_dir, "models")
            optimizers_dir = os.path.join(states_dir, "optimizers")
            dataloaders_dir = os.path.join(states_dir, "dataloaders")
            rng_states_dir = os.path.join(states_dir, "rng_states")
            fs.mkdir(models_dir)
            fs.mkdir(optimizers_dir)
            fs.mkdir(dataloaders_dir)
            fs.mkdir(rng_states_dir)

            if self.env_config.fsdp_default.version == 2:  # fsdp2
                logger.info(f"Saving FSDP2 in dcp, all state dict except dataloader will be checkpointed in shard...")
                from project.distributed.fsdp import get_device_mesh
                first_row_ranks = get_device_mesh().mesh[0, :].tolist()
                first_row_group = dist.new_group(ranks=first_row_ranks, backend="gloo")

                model_name_saved = set()
                for optim_name in self.optimizers.keys():
                    logger.info(f"Processing {optim_name} in optimizer keys...")
                    model_name = re.sub(r'_\d+$', '', optim_name)
                    model = self.models[model_name]
                    optimizer = self.optimizers[optim_name]

                    if model_name not in model_name_saved:
                        logger.info(f"Saving MODEL {model_name}...")
                        state_dict = get_model_state_dict(model=model, options=StateDictOptions(full_state_dict=False, cpu_offload=True))

                        sha256 = hashlib.sha256(f"models/{model_name}".encode("utf-8")).hexdigest()[:12]
                        local_dir = os.environ.get("PROJECT_TMP_DIR", "/tmp/project")
                        tmp_dir = os.path.join(local_dir, sha256, "models", model_name)
                        logger.info(f"Cleaning tmp dir before saving MODEL {model_name}: {tmp_dir}")
                        fs.remove(tmp_dir, sync=True, group=first_row_group)

                        logger.info(f"Saving MODEL {model_name} to {tmp_dir}...")
                        if dist.get_rank(first_row_group) >= 0:
                            dcp_save(state_dict, first_row_group, tmp_dir, os.path.join(models_dir, model_name))
                        comm.barrier()
                        logger.info(f"Saving MODEL {model_name} done")
                        model_name_saved.add(model_name)

                    optim_state_dict = get_optimizer_state_dict(
                        model=model,
                        optimizers=optimizer,
                        options=StateDictOptions(full_state_dict=False, cpu_offload=True)
                    )

                    sha256 = hashlib.sha256(f"optimizers/{optim_name}".encode("utf-8")).hexdigest()[:12]
                    local_dir = os.environ.get("PROJECT_TMP_DIR", "/tmp/project")
                    tmp_dir = os.path.join(local_dir, sha256, "optimizers", optim_name)
                    logger.info(f"Cleaning tmp dir before saving OPTIMIZER {optim_name}: {tmp_dir}")
                    fs.remove(tmp_dir, sync=True, group=first_row_group)

                    logger.info(f"Saving OPTIMIZER {optim_name} to {tmp_dir}...")
                    if dist.get_rank(first_row_group) >= 0:
                        dcp_save(optim_state_dict, first_row_group, tmp_dir, os.path.join(optimizers_dir, optim_name))
                    comm.barrier()
                    logger.info(f"Saving OPTIMIZER {optim_name} done")

                    if f"{model_name}_ema" in self.models and f"{model_name}_ema" not in model_name_saved:
                        logger.info(f"Saving MODEL {model_name}_ema...")
                        ema_model = self.models[f"{model_name}_ema"]

                        ema_state_dict = get_model_state_dict(
                            model=ema_model,
                            options=StateDictOptions(full_state_dict=False, cpu_offload=True)
                        )

                        sha256 = hashlib.sha256(f"models/{model_name}_ema".encode("utf-8")).hexdigest()[:12]
                        local_dir = os.environ.get("PROJECT_TMP_DIR", "/tmp/project")
                        tmp_dir = os.path.join(local_dir, sha256, "models", f"{model_name}_ema")
                        logger.info(f"Cleaning tmp dir before saving MODEL {model_name}_ema: {tmp_dir}")
                        fs.remove(tmp_dir, sync=True, group=first_row_group)

                        logger.info(f"Saving model {model_name}_ema to {tmp_dir}...")
                        if dist.get_rank(first_row_group) >= 0:
                            dcp_save(ema_state_dict, first_row_group, tmp_dir, os.path.join(models_dir, f"{model_name}_ema"))
                        comm.barrier()
                        logger.info(f"Saving MODEL {model_name}_ema done")
                        model_name_saved.add(f"{model_name}_ema")

                # save dataloaders
                logger.info(f"Saving dataloaders...")
                self.dataloader.save(dataloaders_dir)
                logger.info(f"Saving dataloaders done")

                # save rng states
                logger.info(f"Saving rng states...")
                rng_state_dict = {
                    "random_state": random.getstate(),
                    "np_state": np.random.get_state(),
                    "torch_state": torch.get_rng_state(),
                    "torch_cuda_state": torch.cuda.get_rng_state(),
                }
                maybe_upload(rng_state_dict, f"rank_{comm.get_rank()}.pkl", rng_states_dir)
                comm.barrier()
                logger.info(f"Saving rng states done")

                if comm.get_rank() == 0 and not save_before_train:  # only update latest when not save_before_train
                    flag_dir = os.path.join(self.save_dir, "states")
                    obj = {
                        "latest_iter": (self.iter + 1),
                        "accumulator": get_running_accumulator().state_dict(),
                    }
                    maybe_upload(obj, "latest", flag_dir)
                    logger.info(f"Updated latest to {(self.iter + 1)}")

                _cleanup_old_checkpoints(self.save_dir, self.persistence_config.max_to_keep)

            elif self.persistence_config.save_dcp:
                logger.info(f"Saving dcp, all state dict except dataloader will be checkpointed in shard...")
                from project.distributed.fsdp import get_device_mesh
                first_row_ranks = get_device_mesh().mesh[0, :].tolist()
                first_row_group = dist.new_group(ranks=first_row_ranks, backend="gloo")

                # save trainable models and optimizers
                model_name_saved = set()
                for optim_name in self.optimizers.keys():
                    logger.info(f"Processing {optim_name} in optimizer keys...")
                    model_name = re.sub(r'_\d+$', '', optim_name)
                    model = self.models[model_name]
                    optimizer = self.optimizers[optim_name]

                    if dist.get_rank(first_row_group) >= 0:
                        with FSDP.state_dict_type(
                            model,
                            StateDictType.SHARDED_STATE_DICT,
                            ShardedStateDictConfig(offload_to_cpu=True),
                            ShardedOptimStateDictConfig(offload_to_cpu=True)
                        ):
                            if model_name not in model_name_saved:
                                logger.info(f"Saving MODEL {model_name}...")
                                state_dict = model.state_dict()
                                sha256 = hashlib.sha256(f"models/{model_name}".encode("utf-8")).hexdigest()[:12]
                                local_dir = os.environ.get("PROJECT_TMP_DIR", "/tmp/project")
                                tmp_dir = os.path.join(local_dir, sha256, "models", model_name)
                                logger.info(f"Cleaning tmp dir before saving MODEL {model_name}: {tmp_dir}")
                                fs.remove(tmp_dir, sync=True, group=first_row_group)

                                logger.info(f"Saving MODEL {model_name} to {tmp_dir}...")
                                dcp_save(state_dict, first_row_group, tmp_dir, os.path.join(models_dir, model_name))
                                logger.info(f"Saving MODEL {model_name} done")
                                model_name_saved.add(model_name)

                            optim_state_dict = FSDP.optim_state_dict(model, optimizer, group=first_row_group)
                            sha256 = hashlib.sha256(f"optimizers/{optim_name}".encode("utf-8")).hexdigest()[:12]
                            local_dir = os.environ.get("PROJECT_TMP_DIR", "/tmp/project")
                            tmp_dir = os.path.join(local_dir, sha256, "optimizers", optim_name)
                            logger.info(f"Cleaning tmp dir before saving OPTIMIZER {optim_name}: {tmp_dir}")
                            fs.remove(tmp_dir, sync=True, group=first_row_group)

                            logger.info(f"Saving OPTIMIZER {optim_name} to {tmp_dir}...")
                            dcp_save(optim_state_dict, first_row_group, tmp_dir, os.path.join(optimizers_dir, optim_name))
                            logger.info(f"Saving OPTIMIZER {optim_name} done")

                    comm.barrier()

                    if f"{model_name}_ema" in self.models and f"{model_name}_ema" not in model_name_saved:
                        logger.info(f"Saving MODEL {model_name}_ema...")
                        ema_model = self.models[f"{model_name}_ema"]

                        if dist.get_rank(first_row_group) >= 0:
                            with FSDP.state_dict_type(
                                ema_model,
                                StateDictType.SHARDED_STATE_DICT,
                                ShardedStateDictConfig(offload_to_cpu=True),
                                ShardedOptimStateDictConfig(offload_to_cpu=True)
                            ):
                                ema_state_dict = ema_model.state_dict()
                                sha256 = hashlib.sha256(f"models/{model_name}_ema".encode("utf-8")).hexdigest()[:12]
                                local_dir = os.environ.get("PROJECT_TMP_DIR", "/tmp/project")
                                tmp_dir = os.path.join(local_dir, sha256, "models", f"{model_name}_ema")
                                logger.info(f"Cleaning tmp dir before saving MODEL {model_name}_ema: {tmp_dir}")
                                fs.remove(tmp_dir, sync=True, group=first_row_group)

                                logger.info(f"Saving model {model_name}_ema to {tmp_dir}...")
                                dcp_save(ema_state_dict, first_row_group, tmp_dir, os.path.join(models_dir, f"{model_name}_ema"))

                        comm.barrier()
                        logger.info(f"Saving MODEL {model_name}_ema done")
                        model_name_saved.add(f"{model_name}_ema")

                # save dataloaders
                logger.info(f"Saving dataloaders...")
                self.dataloader.save(dataloaders_dir)
                logger.info(f"Saving dataloaders done")

                # save rng states
                logger.info(f"Saving rng states...")
                rng_state_dict = {
                    "random_state": random.getstate(),
                    "np_state": np.random.get_state(),
                    "torch_state": torch.get_rng_state(),
                    "torch_cuda_state": torch.cuda.get_rng_state(),
                }
                maybe_upload(rng_state_dict, f"rank_{comm.get_rank()}.pkl", rng_states_dir)
                comm.barrier()
                logger.info(f"Saving rng states done")

                if comm.get_rank() == 0 and not save_before_train:  # only update latest when not save_before_train
                    flag_dir = os.path.join(self.save_dir, "states")
                    obj = {
                        "latest_iter": (self.iter + 1),
                        "accumulator": get_running_accumulator().state_dict(),
                    }
                    maybe_upload(obj, "latest", flag_dir)
                    logger.info(f"Updated latest to {(self.iter + 1)}")

                _cleanup_old_checkpoints(self.save_dir, self.persistence_config.max_to_keep)

            else:
                logger.info(f"Saving async, all state dict except dataloader will be offload to cpu at rank 0 first...")
                state_dicts = []

                # save trainable models and optimziers
                model_name_saved = set()
                for optim_name in self.optimizers.keys():
                    model_name = re.sub(r'_\d+$', '', optim_name)
                    model = self.models[model_name]
                    optimizer = self.optimizers[optim_name]

                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    ):
                        if model_name not in model_name_saved:
                            logger.info(f"Saving MODEL {model_name}...")
                            state_dict = model.state_dict()
                            if comm.get_rank() == 0:
                                if hasattr(model, "peft_config"):
                                    state_dict = get_peft_model_state_dict(model, state_dict=state_dict)
                                state_dicts.append((state_dict, f"{model_name}.pth", models_dir))
                            logger.info(f"Saving MODEL {model_name} done")
                            model_name_saved.add(model_name)

                        optim_state_dict = FSDP.optim_state_dict(model, optimizer)
                        if comm.get_rank() == 0:
                            state_dicts.append((optim_state_dict, f"{optim_name}.pth", optimizers_dir))
                        logger.info(f"Saving OPTIMIZER {optim_name} done")

                    comm.barrier()

                    if f"{model_name}_ema" in self.models and f"{model_name}_ema" not in model_name_saved:
                        logger.info(f"Saving MODEL {model_name}_ema...")
                        ema_model = self.models[f"{model_name}_ema"]

                        with FSDP.state_dict_type(
                            ema_model,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        ):
                            ema_state_dict = ema_model.state_dict()

                        if comm.get_rank() == 0:
                            if isinstance(ema_model, PeftModel):
                                ema_state_dict = get_peft_model_state_dict(ema_model, state_dict=ema_state_dict)
                            state_dicts.append((ema_state_dict, f"{model_name}_ema.pth", models_dir))

                        comm.barrier()
                        logger.info(f"Saving MODEL {model_name}_ema done")
                        model_name_saved.add(f"{model_name}_ema")

                # save dataloaders
                logger.info(f"Saving dataloaders...")
                self.dataloader.save(dataloaders_dir)
                logger.info(f"Saving dataloaders done")

                # save rng states
                logger.info(f"Saving rng states...")
                rng_state_dict = {
                    "random_state": random.getstate(),
                    "np_state": np.random.get_state(),
                    "torch_state": torch.get_rng_state(),
                    "torch_cuda_state": torch.cuda.get_rng_state(),
                }
                maybe_upload(rng_state_dict, f"rank_{comm.get_rank()}.pkl", rng_states_dir)
                comm.barrier()
                logger.info(f"Saving rng states done")

                # save async call
                if comm.get_rank() == 0:
                    flag_dir = os.path.join(self.save_dir, "states")
                    obj = {
                        "latest_iter": (self.iter + 1),
                        "accumulator": get_running_accumulator().state_dict(),
                    }
                    self.fut = _save(getattr(self, "fut", None), state_dicts, flag_dir, obj)

            comm.barrier()
            logger.info(f"Saving state at iter {self.iter + 1} done")
            torch.cuda.empty_cache()
            gc.collect()

    ############################################################
    # training utils
    ############################################################
    def setup_writer(self, cfg):
        self.writer = None
        if comm.get_rank() == 0 and os.environ.get("D", 0) == 0:
            self.writer = init_writer(self.persistence_config, self.output_dir, cfg)

    def freeze(self, model: nn.Module):
        if getattr(model, "is_first_freeze_call", True):
            trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
            setattr(model, "trainable_param_names", trainable_param_names)
            setattr(model, "is_first_freeze_call", False)
        model.requires_grad_(False)

    def unfreeze(self, model: nn.Module):
        trainable_param_names = getattr(model, "trainable_param_names", None)
        if trainable_param_names is not None:
            for name, param in model.named_parameters():
                if name in trainable_param_names:
                    param.requires_grad_(True)
