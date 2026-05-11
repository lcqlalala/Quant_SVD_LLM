# coding:utf8
"""
SVD-aware projection layers for Meta-Llama-3/3.1 attention with GQA.

This file intentionally follows the two-factor layout used by
utils/mixed_precision.py:

    W ~= U V

where each compressed projection is represented by a pair named
``*_u_proj`` and ``*_v_proj``.  The mixed-precision pipeline discovers these
pairs and later installs ``*_mp_proj`` modules.  Therefore this module must not
use a separate ``*_s_proj`` parameter in the forward path.

The important LLaMA-3/3.1 difference from LLaMA-1/2 is GQA: K/V projections
output ``num_key_value_heads * head_dim`` rather than ``hidden_size``.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig

__all__ = [
    "SVD_Llama3Attention",
    "SVD_Llama3MLP",
]


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(hidden_states.dtype)


class LlamaRotaryEmbedding(nn.Module):
    """RoPE cache compatible with the older SVD-LLaMA wrapper API."""

    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        self._set_cos_sin_cache(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, *, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids is None:
        position_ids = torch.arange(q.shape[-2], device=q.device).unsqueeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand K/V heads to match Q heads for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _low_rank(in_features: int, out_features: int, ratio: float) -> int:
    rank = int(out_features * in_features * ratio / (out_features + in_features))
    return max(rank, 1)


class SVD_Llama3MLP(nn.Module):
    def __init__(self, *, hidden_size: int, intermediate_size: int, hidden_act: str, ratio: float):
        super().__init__()
        rank_gate = _low_rank(hidden_size, intermediate_size, ratio)
        rank_up = _low_rank(hidden_size, intermediate_size, ratio)
        rank_down = _low_rank(intermediate_size, hidden_size, ratio)

        self.gate_u_proj = nn.Linear(rank_gate, intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(hidden_size, rank_gate, bias=False)

        self.down_u_proj = nn.Linear(rank_down, hidden_size, bias=False)
        self.down_v_proj = nn.Linear(intermediate_size, rank_down, bias=False)

        self.up_u_proj = nn.Linear(rank_up, intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(hidden_size, rank_up, bias=False)

        self.gate_mp_proj = None
        self.down_mp_proj = None
        self.up_mp_proj = None
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.up_mp_proj is None:
            up = self.up_u_proj(self.up_v_proj(x))
        else:
            up = self.up_mp_proj(x)

        if self.gate_mp_proj is None:
            gate = self.gate_u_proj(self.gate_v_proj(x))
        else:
            gate = self.gate_mp_proj(x)

        hidden = self.act_fn(gate) * up
        if self.down_mp_proj is None:
            return self.down_u_proj(self.down_v_proj(hidden))
        return self.down_mp_proj(hidden)


class SVD_Llama3Attention(nn.Module):
    def __init__(self, config: LlamaConfig, *, ratio: float = 1.0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.num_kv_groups = self.num_key_value_groups
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.is_causal = True

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim "
                f"({self.hidden_size} != {self.num_heads} * {self.head_dim})"
            )
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"({self.num_heads} vs {self.num_key_value_heads})"
            )

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim
        o_in = self.num_heads * self.head_dim

        q_rank = _low_rank(self.hidden_size, q_out, ratio)
        k_rank = _low_rank(self.hidden_size, kv_out, ratio)
        v_rank = _low_rank(self.hidden_size, kv_out, ratio)
        o_rank = _low_rank(o_in, self.hidden_size, ratio)

        self.q_u_proj = nn.Linear(q_rank, q_out, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, q_rank, bias=False)

        self.k_u_proj = nn.Linear(k_rank, kv_out, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, k_rank, bias=False)

        self.v_u_proj = nn.Linear(v_rank, kv_out, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, v_rank, bias=False)

        self.o_u_proj = nn.Linear(o_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(o_in, o_rank, bias=False)

        self.q_mp_proj = None
        self.k_mp_proj = None
        self.v_mp_proj = None
        self.o_mp_proj = None

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _check_last_dim(self, name: str, tensor: torch.Tensor, expected: int):
        got = int(tensor.shape[-1])
        if got != expected:
            raise RuntimeError(
                f"{name} output dim mismatch before view: got {got}, expected {expected}. "
                "For LLaMA-3/3.1 GQA, k/v output dim must be "
                "num_key_value_heads * head_dim. If this triggers after MP "
                "quantization, ensure the checkpoint was generated with this "
                "two-factor svd_llama3_1.py and not the old *_s_proj version."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim

        if self.q_mp_proj is None:
            query_states = self.q_u_proj(self.q_v_proj(hidden_states))
        else:
            query_states = self.q_mp_proj(hidden_states)
        self._check_last_dim("q_proj", query_states, q_out)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.k_mp_proj is None:
            key_states = self.k_u_proj(self.k_v_proj(hidden_states))
        else:
            key_states = self.k_mp_proj(hidden_states)
        self._check_last_dim("k_proj", key_states, kv_out)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.v_mp_proj is None:
            value_states = self.v_u_proj(self.v_v_proj(hidden_states))
        else:
            value_states = self.v_mp_proj(hidden_states)
        self._check_last_dim("v_proj", value_states, kv_out)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            min_val = torch.finfo(attn_weights.dtype).min
            attn_weights = torch.max(attn_weights, torch.tensor(min_val, device=attn_weights.device))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        if self.o_mp_proj is None:
            attn_output = self.o_u_proj(self.o_v_proj(attn_output))
        else:
            attn_output = self.o_mp_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value
