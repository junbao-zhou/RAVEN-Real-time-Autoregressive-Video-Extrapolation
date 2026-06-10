"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import itertools
from typing import List, Optional

import torch
import torch.cuda.amp as amp
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask

from project.utils import comm
from project.utils.misc import maybe_checkpoint
from project.utils.running import TrainingPhase, get_running_average_meter, get_training_phase

from . import model as wan
from .attention import FlexAttention


class NaiveCache:
    def __init__(self, num_layers, batch_size=None, sink=0, window_size=None):
        self.batch_size = batch_size
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}
        if batch_size is not None:
            self.kvlens = [0] * batch_size
            self.curr_rope = torch.tensor([0] * batch_size, device=comm.get_device())

        self.sink = sink
        self.window_size = window_size
        # Unified history of chunk lengths for all steps
        # List[List[int]], outer: time step (chunk), inner: batch
        self.chunk_lens = []
        # self.kvlens_sink = []    # List[List[int]], inner batch, outer chunk
        # self.kvlens_window = []  # List[List[int]], inner batch, outer chunk

    def update_kvlens(self, new_sample_lens):
        assert self.batch_size is not None, "Batch size must be specified to update kvlens."

        # 1. Record the length of the incoming chunk for history
        self.chunk_lens.append(new_sample_lens)
        current_step = len(self.chunk_lens)

        # 2. Update kvlens per sample
        for b in range(self.batch_size):
            s = self.sink[b]
            w = self.window_size[b]
            new_len = new_sample_lens[b]

            # Calculate pop length if window is full
            # Condition: We have more chunks than sink + window
            pop_len = 0
            if w is not None and current_step > s + w:
                # The chunk to pop is the one that falls out of the window.
                # Index in history: current_step - w - 1 (the new one) - 1 (to get the one before window starts? No)
                # Logic:
                # Chunks: [0, 1, 2, 3], s=1, w=2. Keep 0 (sink), 2, 3 (window). Pop 1.
                # current_step=4. Pop index = 4 - 2 - 1 = 1. Correct.
                pop_idx = current_step - w - 1
                pop_len = self.chunk_lens[pop_idx][b]

            self.kvlens[b] = self.kvlens[b] + new_len - pop_len

    def update_kvcache(self, layer_idx, new_keys, new_values, new_kvlens):
        # Note: At this point, update_kvlens has already been called, and self.chunk_lens contains the length of the current step.
        # self.kvlens has also been updated (the pop_len has been subtracted).
        current_step = len(self.chunk_lens)

        chunks_k = []
        chunks_v = []
        input_offset = 0
        for b in range(self.batch_size):
            s = self.sink[b]
            w = self.window_size[b]

            # 1. Calculate Sink Length (s_len)
            # Sum lengths of the first 's' chunks for this batch index
            # Optimization: If s=0, s_len=0.
            s_len = 0
            if s > 0:
                # Only accumulate the actual existing chunks to prevent initial step < s
                limit = min(current_step, s)
                for i in range(limit):
                    s_len += self.chunk_lens[i][b]
            # 2. Calculate Pop Length (p_len)
            # This is the length of the chunk that is being evicted in this step
            p_len = 0
            if w is not None and current_step > s + w:
                pop_idx = current_step - w - 1
                p_len = self.chunk_lens[pop_idx][b]

            # 3. Calculate total length of the sample in the INPUT tensor (new_keys)
            # Input contains: [Retained Old Cache] + [Popped Part] + [New Token]
            # self.kvlens[b] is the target length (Retained + New).
            # So input length = self.kvlens[b] + p_len
            current_input_len = self.kvlens[b] + p_len

            # 4. Slicing Logic
            if p_len == 0:
                # No eviction, keep everything for this sample
                chunks_k.append(new_keys[input_offset : input_offset + current_input_len])
                chunks_v.append(new_values[input_offset : input_offset + current_input_len])
            else:
                # Eviction happens: Keep Sink + Skip Pop + Keep Window(including new)
                # Part 1: Sink
                if s_len > 0:
                    chunks_k.append(new_keys[input_offset : input_offset + s_len])
                    chunks_v.append(new_values[input_offset : input_offset + s_len])

                # Part 2: Window (Skip the popped part)
                # Start after sink + pop_len
                window_start = input_offset + s_len + p_len
                window_end = input_offset + current_input_len

                if window_end > window_start:
                    chunks_k.append(new_keys[window_start : window_end])
                    chunks_v.append(new_values[window_start : window_end])
            # Move offset
            input_offset += current_input_len
        # 5. Concatenate and Update
        self.key_cache[layer_idx] = torch.cat(chunks_k, dim=0)
        self.value_cache[layer_idx] = torch.cat(chunks_v, dim=0)

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_len(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0

    def seq_lens(self, idx):
        if self.key_cache[idx] is not None:
            return self.key_cache[idx].shape[0]
        else:
            return 0

    @staticmethod
    def merge(caches):
        """ Merge a list of NaiveCache into a single NaiveCache by concatenating along batch dimension. """
        assert len(caches) > 0
        num_layers = caches[0].num_layers
        assert all([cache.num_layers == num_layers for cache in caches]), "All caches must have the same number of layers."
        total_batch_size = sum([len(cache.kvlens) for cache in caches])
        merged_cache = NaiveCache(num_layers, total_batch_size)
        for layer_idx in range(num_layers):
            merged_keys = torch.cat([cache.key_cache[layer_idx] for cache in caches], dim=0)
            merged_values = torch.cat([cache.value_cache[layer_idx] for cache in caches], dim=0)
            merged_cache.key_cache[layer_idx] = merged_keys
            merged_cache.value_cache[layer_idx] = merged_values
        merged_cache.kvlens = list(itertools.chain.from_iterable([cache.kvlens for cache in caches]))
        merged_cache.curr_rope = torch.cat([cache.curr_rope for cache in caches], dim=0)

        # Merge chunk_lens
        # chunk_lens is List[List[int]] (Time, Batch).
        # We need to concatenate the inner lists along the batch dimension for each time step.
        # Assuming all caches have the same number of steps (chunks).
        if len(caches) > 0 and hasattr(caches[0], 'chunk_lens'):
            num_steps = len(caches[0].chunk_lens)
            merged_cache.chunk_lens = []
            for i in range(num_steps):
                # Combine the batch lists for step i
                step_lens = list(itertools.chain.from_iterable([c.chunk_lens[i] for c in caches]))
                merged_cache.chunk_lens.append(step_lens)

        # Merge sink and window_size lists
        merged_cache.sink = list(itertools.chain.from_iterable([c.sink for c in caches]))
        merged_cache.window_size = list(itertools.chain.from_iterable([c.window_size for c in caches]))

        return merged_cache


@amp.autocast(device_type="cuda", enabled=False)
def apply_latent_pos_embed(xs, grid_sizes, freqs, frame_shifts=None, packed=False):
    # n, c = x.size(2), x.size(3) // 2
    c = freqs.size(1)
    if frame_shifts is None:
        frame_shifts = [0] * len(grid_sizes)

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    curr = 0
    for i, ((f, h, w), fs) in enumerate(zip(grid_sizes, frame_shifts)):
        seq_len = f * h * w
        if packed:  # xs: [S, D]
            x = xs[curr:curr+seq_len]  # [S, D]
            curr += seq_len
        else:  # xs: list of [1, S, D]
            x = xs[i]  # [1, S, D]

        # precompute multipliers
        x_i = torch.view_as_complex(x.contiguous().to(torch.float64).reshape(
            seq_len, -1, c, 2))  # [seqlen, n_head, head_dim // 2]
        freqs_i = torch.cat([
            freqs[0][fs:fs+f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)  # [seqlen, 1, head_dim // 2]

        # apply rotary embedding
        if packed:
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)  # [seqlen, n_head, head_dim]
        else:
            x_i = torch.view_as_real(x_i * freqs_i).flatten(1)  # [S, D]
        # x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)

    if packed:
        assert curr == xs.size(0), f"Expected curr ({curr}) to equal xs.size(0) ({xs.size(0)})"
        return torch.cat(output).float()
    else:
        return [u.float() for u in output]


class CausalWanSelfAttention(wan.WanSelfAttention):
    def __init__(
        self,
        *args,
        layer_idx,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx
        self.flex_attention = FlexAttention()

    def forward(
        self,
        packed_sequence,
        grid_sizes,
        freqs,
        frame_shifts,
        attention_mask,
        q_ranges,
        k_ranges,
        attn_type_map,
        attn_workloads,
        sample_lens,
        past_key_values: Optional[NaiveCache] = None,
        update_past_key_values: bool = False,
        key_value_lens: torch.IntTensor = None,
        packed_query_indexes: Optional[torch.IntTensor] = None,
        packed_past_key_value_indexes: Optional[torch.IntTensor] = None,
    ):
        s, n, d = packed_sequence.size(0), self.num_heads, self.head_dim

        packed_query_states = self.norm_q(self.q(packed_sequence)).view(s, n, d)
        packed_key_states = self.norm_k(self.k(packed_sequence)).view(s, n, d)
        packed_value_states = self.v(packed_sequence).view(s, n, d)

        packed_query_states = apply_latent_pos_embed(packed_query_states, grid_sizes, freqs, frame_shifts, packed=True)
        packed_key_states = apply_latent_pos_embed(packed_key_states, grid_sizes, freqs, frame_shifts, packed=True)

        if past_key_values is not None:  # use flash attn
            if past_key_values.seq_lens(self.layer_idx) > 0:  # merge required
                past_key_states = past_key_values.key_cache[self.layer_idx]
                past_value_states = past_key_values.value_cache[self.layer_idx]
                seqlens = len(packed_query_indexes) + len(packed_past_key_value_indexes)
                merged_key_states = past_key_states.new_zeros(size=[seqlens, n, d])
                merged_value_states = past_value_states.new_zeros(size=[seqlens, n, d])
                merged_key_states[packed_query_indexes] = packed_key_states
                merged_key_states[packed_past_key_value_indexes] = past_key_states
                merged_value_states[packed_query_indexes] = packed_value_states
                merged_value_states[packed_past_key_value_indexes] = past_value_states
                packed_key_states, packed_value_states = merged_key_states, merged_value_states

            if update_past_key_values:
                past_key_values.update_kvcache(self.layer_idx, packed_key_states, packed_value_states, sample_lens[:len(key_value_lens)])

            packed_attn_output = self.flash_attention(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                q_lens=sample_lens[:len(key_value_lens)],
                k_lens=key_value_lens
            )

        else:  # use flex attn
            packed_attn_output = self.flex_attention(
                packed_query_states,
                packed_key_states,
                packed_value_states,
                attention_mask,
                q_ranges,
                k_ranges,
                attn_type_map,
                attn_workloads,
                sample_lens
            )

        return self.o(packed_attn_output.flatten(1))


class CausalWanT2VCrossAttention(wan.WanSelfAttention):
    def __init__(self, *args, layer_idx, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx

    def forward(self, x, context, sample_lens,
                past_key_values: Optional[NaiveCache] = None,
                update_past_key_values: bool = False,
                key_value_lens: torch.IntTensor = None,
                packed_new_key_value_indexes: Optional[torch.IntTensor] = None,
                packed_past_key_value_indexes: Optional[torch.IntTensor] = None,
                ):
        b, s, n, d = len(key_value_lens), x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(-1, n, d)

        if context is not None:
            k = self.norm_k(self.k(context)).view(-1, n, d)
            v = self.v(context).view(-1, n, d)

        # kv cache cross attn
        if past_key_values is not None and past_key_values.seq_lens(self.layer_idx) > 0:
            if context is not None:  # merge required
                past_key_states = past_key_values.key_cache[self.layer_idx]
                past_value_states = past_key_values.value_cache[self.layer_idx]
                seqlens = len(packed_new_key_value_indexes) + len(packed_past_key_value_indexes)
                merged_key_states = past_key_states.new_zeros(size=[seqlens, n, d])
                merged_value_states = past_value_states.new_zeros(size=[seqlens, n, d])
                merged_key_states[packed_new_key_value_indexes] = k
                merged_key_states[packed_past_key_value_indexes] = past_key_states
                merged_value_states[packed_new_key_value_indexes] = v
                merged_value_states[packed_past_key_value_indexes] = past_value_states
                k, v = merged_key_states, merged_value_states
            else:
                k = past_key_values.key_cache[self.layer_idx]
                v = past_key_values.value_cache[self.layer_idx]

        if update_past_key_values and context is not None:
            assert all(sink == 0 for sink in past_key_values.sink) and \
                all(window_size is None for window_size in past_key_values.window_size), \
                "Cross-attention cache only supports full cache update."
            past_key_values.key_cache[self.layer_idx] = k
            past_key_values.value_cache[self.layer_idx] = v

        # compute attention
        q_lens, key_value_lens = key_value_lens
        x = self.flash_attention(q, k, v, q_lens=q_lens, k_lens=key_value_lens)

        return self.o(x.flatten(1))


class CausalWanI2VCrossAttention(wan.WanI2VCrossAttention):

    def __init__(self, *args, layer_idx, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx

    def forward(self, x, context, sample_lens,
                past_key_values: Optional[List[NaiveCache]] = None,
                update_past_key_values: List[bool] = False,
                key_value_lens: List[torch.IntTensor] = None,
                packed_new_key_value_indexes: Optional[List[torch.IntTensor]] = None,
                packed_past_key_value_indexes: Optional[List[torch.IntTensor]] = None,
                ):
        context, context_img = context
        past_key_values, past_key_values_img = past_key_values
        update_past_key_values, update_past_key_values_img = update_past_key_values
        key_value_lens, key_value_lens_img = key_value_lens
        packed_new_key_value_indexes, packed_new_key_value_indexes_img = packed_new_key_value_indexes
        packed_past_key_value_indexes, packed_past_key_value_indexes_img = packed_past_key_value_indexes
        b, s, n, d = len(key_value_lens), x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(-1, n, d)

        # txt context
        if context is not None:
            k = self.norm_k(self.k(context)).view(-1, n, d)
            v = self.v(context).view(-1, n, d)

        if past_key_values is not None and past_key_values.seq_lens(self.layer_idx) > 0:
            if context is not None:  # merge required
                past_key_states = past_key_values.key_cache[self.layer_idx]
                past_value_states = past_key_values.value_cache[self.layer_idx]
                seqlens = len(packed_new_key_value_indexes) + len(packed_past_key_value_indexes)
                merged_key_states = past_key_states.new_zeros(size=[seqlens, n, d])
                merged_value_states = past_value_states.new_zeros(size=[seqlens, n, d])
                merged_key_states[packed_new_key_value_indexes] = k
                merged_key_states[packed_past_key_value_indexes] = past_key_states
                merged_value_states[packed_new_key_value_indexes] = v
                merged_value_states[packed_past_key_value_indexes] = past_value_states
                k, v = merged_key_states, merged_value_states
            else:
                k = past_key_values.key_cache[self.layer_idx]
                v = past_key_values.value_cache[self.layer_idx]

        if update_past_key_values and context is not None:
            assert all(sink == 0 for sink in past_key_values.sink) and \
                all(window_size is None for window_size in past_key_values.window_size), \
                "Cross-attention cache only supports full cache update."
            past_key_values.key_cache[self.layer_idx] = k
            past_key_values.value_cache[self.layer_idx] = v

        # img context
        if context_img is not None:
            k_img = self.norm_k_img(self.k_img(context_img)).view(-1, n, d)
            v_img = self.v_img(context_img).view(-1, n, d)

        if past_key_values_img is not None and past_key_values_img.seq_lens(self.layer_idx) > 0:
            if context_img is not None:  # merge required
                past_key_states_img = past_key_values_img.key_cache[self.layer_idx]
                past_value_states_img = past_key_values_img.value_cache[self.layer_idx]
                seqlens_img = len(packed_new_key_value_indexes_img) + len(packed_past_key_value_indexes_img)
                merged_key_states_img = past_key_states_img.new_zeros(size=[seqlens_img, n, d])
                merged_value_states_img = past_value_states_img.new_zeros(size=[seqlens_img, n, d])
                merged_key_states_img[packed_new_key_value_indexes_img] = k_img
                merged_key_states_img[packed_past_key_value_indexes_img] = past_key_states_img
                merged_value_states_img[packed_new_key_value_indexes_img] = v_img
                merged_value_states_img[packed_past_key_value_indexes_img] = past_value_states_img
                k_img, v_img = merged_key_states_img, merged_value_states_img
            else:
                k_img = past_key_values_img.key_cache[self.layer_idx]
                v_img = past_key_values_img.value_cache[self.layer_idx]

        if update_past_key_values_img and context_img is not None:
            assert all(sink == 0 for sink in past_key_values_img.sink) and \
                all(window_size is None for window_size in past_key_values_img.window_size), \
                "Cross-attention cache only supports full cache update."
            past_key_values_img.key_cache[self.layer_idx] = k_img
            past_key_values_img.value_cache[self.layer_idx] = v_img

        # compute attention
        img_q_lens, key_value_lens_img = key_value_lens_img
        q_lens, key_value_lens = key_value_lens
        img_x = self.flash_attention(q, k_img, v_img, q_lens=img_q_lens, k_lens=key_value_lens_img)
        x = self.flash_attention(q, k, v, q_lens=q_lens, k_lens=key_value_lens)

        return self.o((x + img_x).flatten(1))


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': CausalWanT2VCrossAttention,
    'i2v_cross_attn': CausalWanI2VCrossAttention,
}


class CausalWanAttentionBlock(nn.Module):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 layer_idx=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.layer_idx = layer_idx

        # layers
        self.norm1 = wan.WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, window_size, qk_norm, eps,
                                                layer_idx=layer_idx)
        self.norm3 = wan.WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps, layer_idx=layer_idx)
        self.norm2 = wan.WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        attention_mask,
        q_ranges,
        k_ranges,
        attn_type_map,
        attn_workloads,
        sample_lens,
        # seq_lens,
        grid_sizes,
        freqs,
        frame_shifts,
        context,
        # context_lens,
        # self attn
        past_key_values_self_attn=None,
        update_past_key_values_self_attn=False,
        key_value_lens_self_attn=None,
        packed_query_indexes_self_attn=None,
        packed_past_key_value_indexes_self_attn=None,
        # cross attn
        past_key_values_cross_attn=None,
        update_past_key_values_cross_attn=False,
        key_value_lens_cross_attn=None,
        packed_new_key_value_indexes_cross_attn=None,
        packed_past_key_value_indexes_cross_attn=None,
        # complex indexes
        packed_latent_indexes=None,
    ):
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).unbind(dim=1)

        # self-attention
        norm1_x = self.norm1(x).float()
        modulated_norm1_x = norm1_x * (1 + e[1]) + e[0]

        y = self.self_attn(
            modulated_norm1_x, grid_sizes, freqs, frame_shifts,
            attention_mask, q_ranges, k_ranges, attn_type_map, attn_workloads, sample_lens,
            past_key_values_self_attn, update_past_key_values_self_attn, key_value_lens_self_attn,
            packed_query_indexes_self_attn, packed_past_key_value_indexes_self_attn)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        _x = x[packed_latent_indexes]
        _x = _x + self.cross_attn(self.norm3(_x), context, sample_lens,
            past_key_values_cross_attn, update_past_key_values_cross_attn, key_value_lens_cross_attn,
            packed_new_key_value_indexes_cross_attn, packed_past_key_value_indexes_cross_attn)
        x = _x

        norm2_x = self.norm2(x).float()
        modulated_norm2_x = norm2_x * (1 + e[4]) + e[3]
        y = self.ffn(modulated_norm2_x)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]

        return x


class CausalWanHead(wan.Head):
    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [L, C]
            e(Tensor): Shape [L, C]

            modulation: Shape [1, 2, C]
        """
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).unbind(dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class CausalWanModel(wan.WanModel):
    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 embed_checkpoint_enabled=False,
                 block_checkpoint_enabled=True,
                 block_checkpoint_step=1,
                 block_checkpoint_start_idx=0,
                 ):
        nn.Module.__init__(self)

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.embed_checkpoint_enabled = embed_checkpoint_enabled
        self.block_checkpoint_enabled = block_checkpoint_enabled
        self.block_checkpoint_step = block_checkpoint_step
        self.block_checkpoint_start_idx = block_checkpoint_start_idx

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        # cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        attn_type = {
            't2v':'t2v_cross_attn',
            'i2v':'i2v_cross_attn',
        }
        cross_attn_type = attn_type[model_type]
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    window_size, qk_norm, cross_attn_norm, eps, layer_idx=i)
            for i in range(num_layers)
        ])

        # head
        self.head = CausalWanHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            wan.rope_params(1024, d - 4 * (d // 6)),
            wan.rope_params(1024, 2 * (d // 6)),
            wan.rope_params(1024, 2 * (d // 6))
        ], dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = wan.MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

        # update trainable_param_names in the first freeze call
        self.is_first_freeze_call = True
        self.trainable_param_names = []

        self.gradient_checkpointing = False

    def load_state_dict(self, state_dict, *args, **kwargs):
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
        normalized_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k.replace("model.", "", 1)
            normalized_state_dict[k] = v
        state_dict = normalized_state_dict
        msg = super().load_state_dict(state_dict, *args, **kwargs)
        return msg

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def preprocess_kvcache_cross_attn(
        self,
        context: Optional[torch.Tensor],
        key_value_lens_cross_attn: Optional[torch.IntTensor],
        past_key_values_cross_attn: Optional[NaiveCache],
        update_past_key_values_cross_attn: bool,
    ):
        if past_key_values_cross_attn is not None:  # inference
            if past_key_values_cross_attn.seq_len > 0:
                past_key_value_len_cross_attn = sum(past_key_values_cross_attn.kvlens)
                past_key_value_lens_cross_attn_tensor = torch.tensor(past_key_values_cross_attn.kvlens, dtype=torch.int32).to(self.device, non_blocking=True)

                if key_value_lens_cross_attn is not None:  # merge required
                    key_value_lens_cumsum = torch.cumsum(past_key_value_lens_cross_attn_tensor, dim=0)
                    key_value_lens_cumsum_repeat = torch.repeat_interleave(key_value_lens_cumsum, key_value_lens_cross_attn, dim=0)
                    packed_new_key_value_indexes_cross_attn = torch.arange(context.size(0), device=self.device) + key_value_lens_cumsum_repeat

                    key_value_lens_cross_attn_cumsum = torch.cumsum(key_value_lens_cross_attn, dim=0)
                    key_value_lens_cross_attn_cumsum = torch.cat([torch.tensor([0], device=self.device), key_value_lens_cross_attn_cumsum[:-1]], dim=0)
                    key_value_lens_cross_attn_cumsum_repeat = torch.repeat_interleave(key_value_lens_cross_attn_cumsum, past_key_value_lens_cross_attn_tensor, dim=0)
                    packed_past_key_value_indexes_cross_attn = torch.arange(past_key_value_len_cross_attn, device=self.device) + key_value_lens_cross_attn_cumsum_repeat

                    # update merged key_value_lens_cross_attn
                    key_value_lens_cross_attn = key_value_lens_cross_attn + past_key_value_lens_cross_attn_tensor

                else:  # no new context, use only past
                    packed_new_key_value_indexes_cross_attn = None
                    packed_past_key_value_indexes_cross_attn = None
                    key_value_lens_cross_attn = past_key_value_lens_cross_attn_tensor

            else:  # no history, use flash-attn directly
                packed_new_key_value_indexes_cross_attn = None
                packed_past_key_value_indexes_cross_attn = None

            if update_past_key_values_cross_attn:
                # assert past_key_values_cross_attn.sink == 0 and past_key_values_cross_attn.window_size is None, \
                assert all(sink == 0 for sink in past_key_values_cross_attn.sink) and \
                    all(window_size is None for window_size in past_key_values_cross_attn.window_size), \
                    f"Only non-windowed cross-attention with sink=0 is supported for kv cache update."
                past_key_values_cross_attn.kvlens = key_value_lens_cross_attn.tolist()

        else:  # training
            packed_new_key_value_indexes_cross_attn = None
            packed_past_key_value_indexes_cross_attn = None

        return key_value_lens_cross_attn, packed_new_key_value_indexes_cross_attn, packed_past_key_value_indexes_cross_attn

    def forward(
        self,
        x,
        t,
        context,
        # seq_len,
        packed_position_ids: torch.IntTensor,
        packed_latent_indexes: torch.IntTensor,
        packed_latent_seqlens: torch.IntTensor,
        packed_noisy_latent_relative_indexes: torch.IntTensor,
        packed_noisy_latent_seqlens: torch.IntTensor,
        sample_lens: List[int],
        frame_shifts: List[int],
        attention_mask: Optional[BlockMask] = None,
        q_ranges: Optional[torch.IntTensor] = None,
        k_ranges: Optional[torch.IntTensor] = None,
        attn_type_map: Optional[torch.IntTensor] = None,
        attn_workloads: Optional[List[int]] = None,
        past_key_values_self_attn: Optional[NaiveCache] = None,
        update_past_key_values_self_attn: bool = False,
        past_key_values_cross_attn: Optional[NaiveCache] = None,
        update_past_key_values_cross_attn: bool = False,
        past_key_values_cross_attn_img: Optional[NaiveCache] = None,
        update_past_key_values_cross_attn_img: bool = False,
        clip_fea=None,
        y=None,
    ):
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.int32) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]  # [1, C, T, H, W] -> [1, S, D]
        # seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # assert seq_lens.max() <= seq_len
        # x = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
        #               dim=1) for u in x
        # ])

        packed_latent_tokens = torch.cat(x, dim=1)[0]  # [S, D]

        packed_sequence = packed_latent_tokens.new_zeros(size=(len(packed_position_ids), self.dim))
        packed_sequence[packed_latent_indexes] = packed_latent_tokens

        sample_lens = torch.tensor(sample_lens, dtype=torch.int32, device=device)
        bsz = len(context) if context is not None and len(context) > 0 else len(x)

        # preprocess kv cache self attn
        if past_key_values_self_attn is not None:  # inference
            if past_key_values_self_attn.seq_len > 0:  # merge required
                query_len, device = packed_sequence.shape[0], packed_sequence.device
                query_lens_tensor = sample_lens[:bsz]
                past_key_value_len = sum(past_key_values_self_attn.kvlens)
                past_key_value_lens_tensor = torch.tensor(past_key_values_self_attn.kvlens, dtype=torch.int32, device=device)

                key_value_lens_cumsum = torch.cumsum(past_key_value_lens_tensor, dim=0)
                key_value_lens_cumsum_repeat = torch.repeat_interleave(key_value_lens_cumsum, query_lens_tensor, dim=0)
                packed_query_indexes_self_attn = torch.arange(query_len, device=device) + key_value_lens_cumsum_repeat

                query_lens_cumsum = torch.cumsum(query_lens_tensor, dim=0)
                query_lens_cumsum = torch.cat([torch.tensor([0], device=device), query_lens_cumsum[:-1]], dim=0)
                query_lens_cumsum_repeat = torch.repeat_interleave(query_lens_cumsum, past_key_value_lens_tensor, dim=0)
                packed_past_key_value_indexes_self_attn = torch.arange(past_key_value_len, device=device) + query_lens_cumsum_repeat

                # update merged key_value_lens_self_attn
                key_value_lens_self_attn = torch.cat([
                    sample_lens[i:i+1] + past_key_values_self_attn.kvlens[i] for i in range(len(sample_lens))
                ])

                past_position_ids = past_key_values_self_attn.curr_rope  # [bsz,], i.e. curr_rope
                past_position_ids = torch.repeat_interleave(past_position_ids, query_lens_tensor, dim=0)
                packed_position_ids = packed_position_ids + past_position_ids

            else:  # use flash-attn but w/o history
                packed_query_indexes_self_attn = None
                packed_past_key_value_indexes_self_attn = None
                key_value_lens_self_attn = sample_lens[:bsz]

            if update_past_key_values_self_attn:  # in-place update
                # past_key_values_self_attn.kvlens = key_value_lens_self_attn.tolist()
                past_key_values_self_attn.update_kvlens(sample_lens[:bsz].tolist())
                position_ids = packed_position_ids.split(sample_lens[:bsz].tolist())
                curr_rope = [position_ids[i][-1] + 1 for i in range(bsz)]
                past_key_values_self_attn.curr_rope = torch.tensor(curr_rope).to(device, non_blocking=True)
        else:  # training
            packed_query_indexes_self_attn = None
            packed_past_key_value_indexes_self_attn = None
            key_value_lens_self_attn = None

        def time_emb(t, packed_latent_indexes, packed_noisy_latent_relative_indexes, packed_noisy_latent_seqlens):
            packed_timesteps = t.new_zeros(size=(len(packed_latent_indexes),))
            if len(packed_noisy_latent_relative_indexes) > 0:
                packed_timesteps[packed_noisy_latent_relative_indexes] = torch.repeat_interleave(
                    t, packed_noisy_latent_seqlens, dim=0)
            with amp.autocast(dtype=torch.float32):
                e = self.time_embedding(
                    wan.sinusoidal_embedding_1d(self.freq_dim, packed_timesteps).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            return e, e0

        e, e0 = maybe_checkpoint(
            time_emb,
            t, packed_latent_indexes, packed_noisy_latent_relative_indexes, packed_noisy_latent_seqlens,
            enabled=self.gradient_checkpointing and self.embed_checkpoint_enabled,
        )
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        if get_training_phase() == TrainingPhase.IN_FORWARD:
            get_running_average_meter().put_scalar("running/time_emb/std", e0.std().item())

        # context
        if context is not None and len(context) > 0:
            key_value_lens_cross_attn = torch.tensor(
                [self.text_len for _ in context], dtype=torch.int32, device=device)
            context = self.text_embedding(
                torch.cat([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
        else:
            context = None
            key_value_lens_cross_attn = None

        key_value_lens_cross_attn, packed_new_key_value_indexes_cross_attn, packed_past_key_value_indexes_cross_attn = self.preprocess_kvcache_cross_attn(
            context, key_value_lens_cross_attn, past_key_values_cross_attn, update_past_key_values_cross_attn)
        q_lens = []  # only latent tokens should attend to text/img/audio in cross-attn
        sample_lens_cumsum = torch.cat([torch.tensor([0], device=device), torch.cumsum(sample_lens, dim=0)], dim=0)
        for i in range(bsz):
            sample_idxs = torch.arange(sample_lens_cumsum[i], sample_lens_cumsum[i+1], device=device)
            sample_latent_indexes = packed_latent_indexes[torch.isin(packed_latent_indexes, sample_idxs)]
            q_lens.append(len(sample_latent_indexes))
        q_lens = torch.tensor(q_lens, dtype=torch.int32, device=device)
        key_value_lens_cross_attn = (q_lens, key_value_lens_cross_attn)

        if self.model_type=="i2v":  # img context for i2v
            if clip_fea is not None and len(clip_fea) > 0:
                context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
                key_value_lens_cross_attn_img = torch.tensor(
                    [context_clip.size(1) for _ in context_clip], dtype=torch.int32, device=device)
                context_clip = context_clip.view(-1, context_clip.size(2))
            else:
                context_clip = None
                key_value_lens_cross_attn_img = None

            key_value_lens_cross_attn_img, packed_new_key_value_indexes_cross_attn_img, packed_past_key_value_indexes_cross_attn_img = self.preprocess_kvcache_cross_attn(
                context_clip, key_value_lens_cross_attn_img, past_key_values_cross_attn_img, update_past_key_values_cross_attn_img)
            key_value_lens_cross_attn_img = (q_lens, key_value_lens_cross_attn_img)

            context = [context, context_clip]
            key_value_lens_cross_attn = [key_value_lens_cross_attn, key_value_lens_cross_attn_img]
            packed_new_key_value_indexes_cross_attn = [packed_new_key_value_indexes_cross_attn, packed_new_key_value_indexes_cross_attn_img]
            packed_past_key_value_indexes_cross_attn = [packed_past_key_value_indexes_cross_attn, packed_past_key_value_indexes_cross_attn_img]
            past_key_values_cross_attn = [past_key_values_cross_attn, past_key_values_cross_attn_img]
            update_past_key_values_cross_attn = [update_past_key_values_cross_attn, update_past_key_values_cross_attn_img]

        # arguments
        kwargs = dict(
            e=e0,
            attention_mask=attention_mask,  # flex attn
            q_ranges=q_ranges,              # magi attn
            k_ranges=k_ranges,              # magi attn
            attn_type_map=attn_type_map,    # magi attn
            attn_workloads=attn_workloads,
            sample_lens=sample_lens,
            # seq_lens=seq_lens,
            grid_sizes=grid_sizes.tolist(),
            freqs=self.freqs,
            frame_shifts=frame_shifts,
            context=context,
            # context_lens=context_lens,
            # self attn
            past_key_values_self_attn=past_key_values_self_attn,
            update_past_key_values_self_attn=update_past_key_values_self_attn,
            key_value_lens_self_attn=key_value_lens_self_attn,
            packed_query_indexes_self_attn=packed_query_indexes_self_attn,
            packed_past_key_value_indexes_self_attn=packed_past_key_value_indexes_self_attn,
            # cross attn
            past_key_values_cross_attn=past_key_values_cross_attn,
            update_past_key_values_cross_attn=update_past_key_values_cross_attn,
            key_value_lens_cross_attn=key_value_lens_cross_attn,
            packed_new_key_value_indexes_cross_attn=packed_new_key_value_indexes_cross_attn,
            packed_past_key_value_indexes_cross_attn=packed_past_key_value_indexes_cross_attn,
            # complex indexes
            packed_latent_indexes=packed_latent_indexes,
        )

        packed_sequence = maybe_checkpoint(
            self.blocks,
            packed_sequence,
            enabled=self.gradient_checkpointing and self.block_checkpoint_enabled,
            gc_step=self.block_checkpoint_step,
            gc_start_idx=self.block_checkpoint_start_idx,
            **kwargs
        )

        # head
        x = self.head(packed_sequence[packed_latent_indexes], e)

        # unpatchify
        xs = x.split(packed_latent_seqlens.tolist(), dim=0)
        x = self.unpatchify(xs, grid_sizes)
        return [u.float() for u in x]
