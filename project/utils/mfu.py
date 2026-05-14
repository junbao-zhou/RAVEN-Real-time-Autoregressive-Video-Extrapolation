import functools
import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Union

import torch
from tabulate import tabulate
from torch import nn
from torch.compiler import is_dynamo_compiling as is_torchdynamo_compiling
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.modules.conv import _ConvNd

_ENABLE_LOG_FLOP = False  # model-level
_DISABLE_LOG_FLOP = False  # context-level


def set_enable_log_flop(enable: bool):
    global _ENABLE_LOG_FLOP
    _ENABLE_LOG_FLOP = enable


def set_disable_log_flop(disable: bool):
    global _DISABLE_LOG_FLOP
    _DISABLE_LOG_FLOP = disable


def is_log_flop_enabled():
    return _ENABLE_LOG_FLOP and not _DISABLE_LOG_FLOP


class enable_flops_accumulate:
    def __enter__(self) -> None:
        set_enable_log_flop(True)

    def __exit__(self, exc_type=None, exc_value=None, traceback=None) -> None:
        set_enable_log_flop(False)


class disable_flops_accumulate:
    def __init__(self, func=None):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return functools.partial(self.__call__, obj)

    def __enter__(self) -> None:
        set_disable_log_flop(True)

    def __exit__(self, exc_type=None, exc_value=None, traceback=None) -> None:
        set_disable_log_flop(False)

    def __call__(self, *args, **kwargs):
        if self.func is None:
            self.func = args[0]
            return self

        with self:
            return self.func(*args, **kwargs)


class CustomFlops(ABC):
    """
    For custom functions,
    1. run the func within nn.Module to support register_forward_hook
    2. inherit `CustomFlops` and implement the `tflops` method for flops calculation.
    """

    @abstractmethod
    def tflops(self, args, kwargs, output) -> float:
        pass


def conv_tflops_func(module, args, kwargs, output):
    return (2 * math.prod(module.kernel_size) * module.in_channels * (output.numel() / 1e6)) / 1e6


def linear_tflops_func(module, args, kwargs, output):
    return (2 * module.in_features * (output.numel() / 1e6)) / 1e6


basic_flops_func = {
    _ConvNd: conv_tflops_func,
    nn.Linear: linear_tflops_func,
}


class FlopsAccumulator:
    """
    Accumulate total FLOPs(T) each iteration.
    This is intended for mfu logging purposes.
    """

    def __init__(self, name: str, disable=False):
        self.accumulator: Dict[str, float] = {}
        self.name = name
        self.disable = disable
        self.logger = logging.getLogger("MFU::accumulator")

    def reset(self):
        self.accumulator.clear()

    def __call__(self, key: str, tflops: Union[float, torch.Tensor], is_training: bool):
        if self.disable:
            return
        mfu_factor = 3 if is_training else 1
        key = f"{'train' if is_training else 'eval'}_{key}_x{mfu_factor}"
        if key not in self.accumulator:
            self.accumulator[key] = 0.0
        self.accumulator[key] += tflops * mfu_factor

    def total(self):
        res = sum(self.accumulator.values())
        if torch.is_tensor(res):
            res = res.item()
        return res

    def show(self):
        items = list(self.accumulator.items())
        tflops = sum(v for _, v in items)
        items = [(k, f"{v:.6f}") for k, v in items]
        items.append(("", ""))
        items.append(("total", f"{tflops:.6f}"))

        table = tabulate(
            items,
            headers=[self.name, "TFLOPs"],
            tablefmt="heavy_outline",
            numalign="left",
            stralign="left",
        )
        if tflops > 0:
            self.logger.info(f"\n{table}")


def get_flops_accumulator_hook(
    parent_module_name: str,
    flops_accumulator: FlopsAccumulator,
    flops_func: Callable,
):
    def _hook(module, args, kwargs, output):
        # if not is_torchdynamo_compiling() and is_log_flop_enabled():
        if is_log_flop_enabled():  # already support fwd hooks for compiled model
            flops_accumulator(
                f"{parent_module_name}_{module.__class__.__name__}",
                (
                    flops_func(args, kwargs, output)
                    if isinstance(module, CustomFlops)
                    else flops_func(module, args, kwargs, output)
                ),
                torch.is_grad_enabled()
            )

    return _hook


class FlopsState:
    """
    A wrapper for enable online mfu tracker.
    """

    def __init__(self, module: nn.Module, name: str):
        super().__init__()
        self._module: nn.Module = module
        self.flops_accumulator = FlopsAccumulator(name)
        self.handlers = []
        self._register_flops_accumulator_hook()

    def summary_and_reset(self, show: bool = False):
        if show:
            self.flops_accumulator.show()
        total_tflops = self.flops_accumulator.total()
        self.flops_accumulator.reset()
        return total_tflops

    def _register_flops_accumulator_hook(self):
        def _register_hooks(parent_name: str, module: nn.Module):
            for sub_module in module.children():
                # Custom hooks have higher privilege.
                if isinstance(sub_module, CustomFlops):
                    assert isinstance(sub_module, nn.Module)
                    self.handlers.append(
                        sub_module.register_forward_hook(
                            get_flops_accumulator_hook(
                                parent_name, self.flops_accumulator, sub_module.tflops
                            ),
                            with_kwargs=True,
                        )
                    )
                    continue

                # Built-in hooks.
                is_registered = False
                for base_m, flops_func in basic_flops_func.items():
                    if (isinstance(base_m, str) and sub_module.__class__.__name__ == base_m) or (
                        isinstance(base_m, type) and isinstance(sub_module, base_m)
                    ):
                        self.handlers.append(
                            sub_module.register_forward_hook(
                                get_flops_accumulator_hook(
                                    parent_name, self.flops_accumulator, flops_func
                                ),
                                with_kwargs=True,
                            )
                        )
                        is_registered = True
                        break

                # Recursive.
                if not is_registered:
                    _register_hooks(
                        parent_name=(
                            sub_module.__class__.__name__
                            if not isinstance(
                                sub_module, (nn.ModuleList, nn.ModuleDict, nn.Sequential, FSDP)
                            )
                            else parent_name
                        ),
                        module=sub_module,
                    )

        _register_hooks(f"{self._module.__class__.__name__}", self._module)

    def __del__(self):
        for hdl in self.handlers:
            hdl.remove()


def get_device_infos():
    arch = torch.cuda.get_device_capability()
    if arch[0] == 8 and arch[1] == 0:  # A100/A800
        peak_tflops = 312
    elif arch[0] == 9 and arch[1] == 0:
        if torch.cuda.get_device_name().endswith("H20"):  # H20
            peak_tflops = 148
        else:  # H200/H100/H800
            peak_tflops = 989
    elif arch[0] == 8 and arch[1] == 9:  # L20
        peak_tflops = 119.5
    else:
        raise ValueError(f"unknown default tflops of device capability {arch[0]}.{arch[1]}")

    return peak_tflops


def get_mfu(iter_time, modules: List[nn.Module], show: bool = False):
    ideal_TFLOPS = get_device_infos()
    achieve_TFLOPs = 0

    for module in modules:
        assert hasattr(module, "flops_state")
        tflops = module.flops_state.summary_and_reset(show)
        achieve_TFLOPs += tflops

    mfu = achieve_TFLOPs / iter_time / ideal_TFLOPS

    return {f"mfu({ideal_TFLOPS})": mfu}


def register_flops_hook(module: nn.Module, module_name: str):
    assert not hasattr(module, "flops_state")
    module.flops_state = FlopsState(module, name=module_name)
    return module
