"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import os
import warnings
from typing import List

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from project.utils.mfu import CustomFlops

torch._dynamo.config.cache_size_limit = int(os.environ.get("TORCH_DYNAMO_CACHE_SIZE_LIMIT", 2048))
torch._dynamo.config.accumulated_cache_size_limit = int(os.environ.get("TORCH_DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT", 16384))

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_hopper
    FLASH_ATTN_3_AVAILABLE = bool(int(os.environ.get("FLASH_ATTN_3_AVAILABLE", "1")))
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_2_AVAILABLE = bool(int(os.environ.get("FLASH_ATTN_2_AVAILABLE", "1")))
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from magi_attention.api import flex_flash_attn_func
    FLEX_FLASH_ATTN_AVAILABLE = bool(int(os.environ.get("FLEX_FLASH_ATTN_AVAILABLE", "1")))
except ModuleNotFoundError:
    FLEX_FLASH_ATTN_AVAILABLE = False

flex_attention = torch.compile(flex_attention)


class _FlexAttention(nn.Module, CustomFlops):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def tflops(self, args, kwargs, output) -> float:
        attn_workloads = sum(kwargs["attn_workloads"]) / 1e12
        if FLEX_FLASH_ATTN_AVAILABLE:
            _, h, d = output.shape
        else:
            _, h, _, d = output.shape
        return h * (4 * d * attn_workloads)

    def forward(self, *args, attn_workloads, **kwargs):
        if FLEX_FLASH_ATTN_AVAILABLE:
            return flex_flash_attn_func(*args, **kwargs)[0]
        else:
            return flex_attention(*args, **kwargs)


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class FlexAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flex_attn_impl = _FlexAttention()

    def forward(
        self,
        packed_query_states: torch.Tensor,
        packed_key_states: torch.Tensor,
        packed_value_states: torch.Tensor,
        attention_mask: BlockMask,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor,
        attn_workloads: List[int],
        sample_lens: torch.IntTensor=None,
        dtype=torch.bfloat16,
    ):
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        q_dtype = packed_query_states.dtype

        packed_query_states = half(packed_query_states)
        packed_key_states = half(packed_key_states)
        packed_value_states = half(packed_value_states)

        if FLEX_FLASH_ATTN_AVAILABLE:
            packed_attn_output = self._flex_attn_impl(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                auto_range_merge=True,
                attn_workloads=attn_workloads,
            )
        else:
            assert sample_lens is not None, "sample_lens must be provided for padding when flex flash attention is not available."
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = pad_sequence(packed_query_states.permute(1, 0, 2), pad_size)
            packed_key_states = pad_sequence(packed_key_states.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)

            packed_attn_output = self._flex_attn_impl(
                packed_query_states.unsqueeze(0),  # [1, num_head, L, head_dim]
                packed_key_states.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                block_mask=attention_mask,
                attn_workloads=attn_workloads,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]
            packed_attn_output = packed_attn_output.transpose(0, 1)  # [L, num_head, head_dim]

        return packed_attn_output.to(q_dtype)


class _FlashAttentionVarlen(nn.Module, CustomFlops):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        causal = kwargs.get("causal", False)
        _, h, d = output.shape
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if causal:
            min_s = torch.min(seqlens_q, seqlens_k)
            square_part = ((1 + min_s) * min_s) / 2
            full_part = torch.relu(seqlens_q - seqlens_k) * min_s
            valid = (full_part / 1e12 + square_part / 1e12).sum()
        else:
            valid = ((seqlens_q / 1e6) * (seqlens_k / 1e6)).sum()
        return h * (4 * d * valid)

    def forward(self, *args, **kwargs):
        version = kwargs.pop("version", None)
        # apply attention
        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            # Note: dropout_p, window_size are not supported in FA3 now.
            return flash_attn_varlen_func_hopper(*args, **kwargs)
        else:
            assert FLASH_ATTN_2_AVAILABLE
            return flash_attn_varlen_func(*args, **kwargs)


class FlashAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flash_attn_impl = _FlashAttentionVarlen()

    def forward(
        self,
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        """
        q:              [B, Lq, Nq, C1] or packed [L, Nq, C1].
        k:              [B, Lk, Nk, C1] or packed [L, Nk, C1].
        v:              [B, Lk, Nk, C2] or packed [L, Nk, C2]. Nq must be divisible by Nk.
        q_lens:         [B].
        k_lens:         [B].
        dropout_p:      float. Dropout probability.
        softmax_scale:  float. The scaling of QK^T before applying softmax.
        causal:         bool. Whether to apply causal attention mask.
        window_size:    (left right). If not (-1, -1), apply sliding window local attention.
        deterministic:  bool. If True, slightly slower and uses more memory.
        dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
        """
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        assert q.device.type == 'cuda' and q.size(-1) <= 256

        # params
        # b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
        out_dtype = q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        # preprocess query
        q_is_packed = q.ndim == 3
        if q_is_packed:  # packed
            assert q_lens is not None, "q_lens must be provided when q is packed."
            b = len(q_lens)
            lq = max(q_lens).item()
            q = half(q)
        else:
            b, lq = q.size(0), q.size(1)
            if q_lens is None:
                q = half(q.flatten(0, 1))
                q_lens = torch.tensor(
                    [lq] * b, dtype=torch.int32).to(
                        device=q.device, non_blocking=True)
            else:
                q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
        # q_lens = torch.tensor(q_lens, dtype=torch.int32, device=q.device)
        # out_dtype = q.dtype

        # preprocess key, value
        kv_is_packed = k.ndim == 3
        if kv_is_packed:
            assert k_lens is not None, "k_lens must be provided when k is packed."
            lk = max(k_lens).item()
            k = half(k)
            v = half(v)
        else:
            lk = k.size(1)
            if k_lens is None:
                k = half(k.flatten(0, 1))
                v = half(v.flatten(0, 1))
                k_lens = torch.tensor(
                    [lk] * b, dtype=torch.int32).to(
                        device=k.device, non_blocking=True)
            else:
                k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
                v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if q_scale is not None:
            q = q * q_scale

        if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
            warnings.warn(
                'Flash attention 3 is not available, use flash attention 2 instead.'
            )

        # apply attention
        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            # Note: dropout_p, window_size are not supported in FA3 now.
            # x = flash_attn_interface.flash_attn_varlen_func(
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)
            x = self._flash_attn_impl(version=3, **kwargs)
        else:
            assert FLASH_ATTN_2_AVAILABLE
            # x = flash_attn.flash_attn_varlen_func(
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic)
            x = self._flash_attn_impl(version=2, **kwargs)

        # output
        if q_is_packed:
            return x.type(out_dtype)
        else:
            return x.unflatten(0, (b, lq)).type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_varlen_func_hopper(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
