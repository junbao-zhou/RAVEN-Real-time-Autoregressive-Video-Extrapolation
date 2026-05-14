"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import copy
import io
import inspect
import logging
import os
import subprocess
import threading
from typing import Callable, Iterable, Union

import torch
import torch.nn as nn
import torchvision
from peft import PeftModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.checkpoint import checkpoint

from project.utils import comm

TEXT_SUFFIXES = ('.txt', '.prompt')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')
VIDEO_SUFFIXES = ('.mp4', '.avi', '.mov')
AUDIO_SUFFIXES = ('.wav', '.mp3', '.flac')

_LOG_N_TIMES_CALLS = dict()
_LOG_ONCE_LOCK = threading.Lock()

logger = logging.getLogger()


def gradient_checkpointing(module: Union[Callable, nn.Module], *args, use_reentrant, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(
            module,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )
    else:
        return module(*args, **kwargs)


def maybe_checkpoint(module, *args, enabled=True, gc_step=1, gc_start_idx=0, use_reentrant=False, **kwargs):
    if isinstance(module, Iterable):
        def create_custom_forward_sequential(modules, start, end):
            def custom_forward(*args, **kwargs):
                for idx in range(start, end):
                    args_ = modules[idx](*args, **kwargs)
                    if not isinstance(args_, tuple):
                        args_ = (args_,)
                    assert len(args_) == len(args), "All arguments must be returned from each module in the sequential checkpointing."
                    args = args_
                return args
            return custom_forward

        # if module.training is False, we should still enable gradient checkpointing if it is not within torch.no_grad
        if enabled and torch.is_grad_enabled():
            if gc_start_idx > 0:
                args = gradient_checkpointing(
                    create_custom_forward_sequential(module, 0, gc_start_idx),
                    *args,
                    use_reentrant=use_reentrant,
                    enabled=False,
                    **kwargs
                )

            num_chunks = (len(module) - gc_start_idx - 1) // gc_step + 1
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * gc_step + gc_start_idx
                end_idx = min((chunk_idx + 1) * gc_step + gc_start_idx, len(module))
                args = gradient_checkpointing(
                    create_custom_forward_sequential(module, start_idx, end_idx),
                    *args,
                    use_reentrant=use_reentrant,
                    enabled=enabled,
                    **kwargs
                )
            if len(args) == 1:
                args = args[0]
            return args
        else:
            for sub_module in module:
                args_ = sub_module(*args, **kwargs)
                if not isinstance(args_, tuple):
                    args_ = (args_,)
                assert len(args_) == len(args), "All arguments must be returned from each module in the sequential checkpointing."
                args = args_
            if len(args) == 1:
                args = args[0]
            return args
    else:
        def create_custom_forward(module):
            def custom_forward(*args, **kwargs):
                return module(*args, **kwargs)
            return custom_forward

        # if module.training is False, we should still enable gradient checkpointing if it is not within torch.no_grad
        if enabled and torch.is_grad_enabled():
            return gradient_checkpointing(create_custom_forward(module), *args, use_reentrant=use_reentrant, enabled=enabled, **kwargs)
        else:
            return module(*args, **kwargs)


def deepcopy_with_tensor(obj):
    if isinstance(obj, dict):
        typing = type(obj)
        return typing(**{k: deepcopy_with_tensor(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [deepcopy_with_tensor(v) for v in obj]
    elif torch.is_tensor(obj):
        return obj.detach().clone()
    else:
        return copy.deepcopy(obj)


def to_torch_dtype(dtype: str):
    if dtype in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16", "torch.float16"):
        return torch.float16
    if dtype in ("fp32", "float32", "torch.float32"):
        return torch.float32
    if dtype in ("fp64", "float64", "torch.float64"):
        return torch.float64
    if dtype in ("double", "torch.double"):
        return torch.double

    raise ValueError(f"Unrecognized dtype {dtype}")


def save_video(
    tensor: torch.Tensor,
    save_path,
    audio_path=None,
    audio_tensor: torch.Tensor = None,
    audio_sample_rate: int = 16000,
    fps=30,
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
    crf=18
):
    if audio_path is not None and audio_tensor is not None:
        raise ValueError("Only one of audio_path and audio_tensor can be provided.")

    # preprocess
    tensor = tensor.clamp(min=value_range[0], max=value_range[1])
    tensor = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=nrow, normalize=normalize, value_range=value_range)
        for u in tensor.unbind(2)  # T x [B, C, H, W] -> T x [C, H, W]
    ], dim=1).permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
    tensor = (tensor * 255).type(torch.uint8).cpu().numpy()
    # tensor: [T, H, W, C]
    T, H, W, C = tensor.shape
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', 'pipe:0',
    ]
    audio_bytes = None
    audio_read_fd = None
    audio_write_fd = None
    pass_fds = ()
    if audio_tensor is not None:
        audio_tensor = audio_tensor.detach().cpu().float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim != 2:
            raise ValueError(
                f"audio_tensor should have shape [T] or [C, T], got {tuple(audio_tensor.shape)}"
            )

        if audio_tensor.shape[0] > audio_tensor.shape[1] and audio_tensor.shape[1] <= 8:
            audio_tensor = audio_tensor.transpose(0, 1)

        audio_buffer = io.BytesIO()
        from torchcodec.encoders import AudioEncoder  # torchcodec requires LD_LIBRARY_PATH set
        AudioEncoder(audio_tensor, sample_rate=audio_sample_rate).to_file_like(audio_buffer, format="wav")
        audio_bytes = audio_buffer.getvalue()
        audio_read_fd, audio_write_fd = os.pipe()
        pass_fds = (audio_read_fd,)
        cmd += ['-f', 'wav', '-i', f'pipe:{audio_read_fd}', '-map', '0:v', '-map', '1:a', '-shortest']
    elif audio_path:
        cmd += ['-i', audio_path, '-map', '0:v', '-map', '1:a', '-shortest']
    cmd += [
        '-vcodec', 'libx264',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-acodec', 'aac',
        save_path
    ]
    audio_writer = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=pass_fds,
        )
        if audio_read_fd is not None:
            os.close(audio_read_fd)
            audio_read_fd = None
        if audio_write_fd is not None:
            def _write_audio():
                try:
                    with os.fdopen(audio_write_fd, "wb") as f:
                        f.write(audio_bytes)
                except BrokenPipeError:
                    logger.warning(f"FFmpeg audio pipe closed early when saving {save_path}.")

            audio_writer = threading.Thread(target=_write_audio)
            audio_writer.start()
            audio_write_fd = None
        _, stderr = proc.communicate(input=tensor.tobytes())
        if audio_writer is not None:
            audio_writer.join()
        if proc.returncode != 0:
            logger.warning(f"FFmpeg failed when saving {save_path}:\n{stderr.decode()}")
    finally:
        if audio_read_fd is not None:
            os.close(audio_read_fd)
        if audio_write_fd is not None:
            os.close(audio_write_fd)


def bin_losses(
    losses: torch.Tensor,
    seqlens: torch.Tensor,
    values: torch.Tensor,
    bin_size: Union[int, float] = 100,
    min_value: Union[int, float] = 0,
    max_value: Union[int, float] = 1000,
    key_prefix: str = "values",
):
    nbins = round((max_value - min_value) / bin_size)
    bin_size = (max_value - min_value) / nbins  # recompute to avoid rounding drift

    # validate all values are within range
    if not (values.min() >= min_value and values.max() <= max_value):
        raise ValueError(f"Values out of range [{min_value}, {max_value}]")

    # accumulate losses and seqlens into bins
    bin_indices = ((values - min_value) / bin_size).long().clamp(0, nbins - 1)
    binned_losses = torch.zeros((nbins,), dtype=losses.dtype, device=losses.device)
    binned_seqlens = torch.zeros((nbins,), dtype=seqlens.dtype, device=seqlens.device)
    binned_losses.scatter_add_(0, bin_indices, losses)
    binned_seqlens.scatter_add_(0, bin_indices, seqlens)

    # build output dicts: avg loss and token count per bin
    binned_loss_dict = {}
    binned_cnt_dict = {}
    for bin_index in range(nbins):
        if binned_seqlens[bin_index] <= 0:
            continue
        bin_start = min_value + bin_index * bin_size
        bin_end = bin_start + bin_size
        if isinstance(bin_size, float):
            bin_key = f"{key_prefix}/{bin_start:.3f}-{bin_end:.3f}"
        else:
            bin_key = f"{key_prefix}/{int(bin_start)}-{int(bin_end)}"
        binned_loss_dict[bin_key] = (binned_losses[bin_index] / binned_seqlens[bin_index]).item()
        binned_cnt_dict[bin_key] = binned_seqlens[bin_index].item()

    return binned_loss_dict, binned_cnt_dict


def to_device(x, device=None, non_blocking=True):
    if device is None:
        device = comm.get_device()
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    elif isinstance(x, dict):
        for k in x.keys():
            x[k] = to_device(x[k], device)
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = to_device(x[i], device)
    return x


def unwrap_model(model) -> nn.Module:
    if isinstance(model, FSDP):
        model = model.module
    if isinstance(model, PeftModel):
        model = model.base_model.model
    return model


def merge_log_cnt_dicts(log_dicts, cnt_dicts):
    log_dict = dict()
    cnt_dict = dict()
    all_log_keys = set().union(*[d.keys() for d in log_dicts])
    all_cnt_keys = set().union(*[d.keys() for d in cnt_dicts])
    for k in all_cnt_keys:
        total_cnt = sum(d[k] for d in cnt_dicts if k in d)
        cnt_dict[k] = total_cnt
    for k in all_log_keys:
        weighted_sum = 0.0
        total_weight = 0.0
        for i, log_d in enumerate(log_dicts):
            if k not in log_d:
                continue
            cnt_d = cnt_dicts[i]
            if k in cnt_d:
                weight = cnt_d[k]
            else:
                weight = 1.0
            weighted_sum += log_d[k] * weight
            total_weight += weight

        if total_weight > 0:
            log_dict[k] = weighted_sum / total_weight
        else:
            log_dict[k] = 0.0
    return log_dict, cnt_dict


def no_tensor_dtype_cast(x):
    """
    This disables any dtype casting for a pytorch tensor
    """

    if x is not None:
        assert type(x) is torch.Tensor
        assert x.to != no_tensor_dtype_cast

        org_dtype = x.dtype

        def to(self, *args, **kwargs):
            x = torch.Tensor.to(self, *args, **kwargs)
            if x is self:
                return self
            else:
                x = torch.Tensor.to(x, dtype=org_dtype)
                x = torch.Tensor.to(self, x)
                return no_tensor_dtype_cast(x)

        x.to = to.__get__(x)

        return x


def log_n_times(message: str, n: int = 1, id: str = None):
    if n <= 0:
        return

    frame = inspect.currentframe()
    try:
        caller = frame.f_back
        if caller is None:
            logger.info(message)
            return

        callsite = (caller.f_code.co_filename, caller.f_lineno, id)
        with _LOG_ONCE_LOCK:
            count = _LOG_N_TIMES_CALLS.get(callsite, 0)
            if count >= n:
                return
            _LOG_N_TIMES_CALLS[callsite] = count + 1
    finally:
        del frame

    logger.info(message)


def log_once(message: str, identifier: str = None):
    log_n_times(message, n=1, identifier=identifier)
