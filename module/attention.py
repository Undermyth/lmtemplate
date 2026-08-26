from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from fla.modules import ShortConvolution
from jaxtyping import Float, Int
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input

from .cache_utils import Cache

@dataclass
class FlashAttentionConfig:
    _attn_implementation: str = 'flash_attention_2'

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    x: Float[torch.Tensor, 'b h t d'],
    cos: Float[torch.Tensor, 'b t rd'],
    sin: Float[torch.Tensor, 'b t rd'],
    position_ids=None,
    unsqueeze_dim: int = 1
) -> torch.Tensor:
    """
    Applies rotary positional embedding to the input tensor.

    Args:
        cos (Float[torch.Tensor, 'b t rd']): Cosine values for rotation.
        sin (Float[torch.Tensor, 'b t rd']): Sine values for rotation.
        position_ids (Optional): Position IDs for the embeddings (default: None).
        unsqueeze_dim (int): Dimension to unsqueeze for broadcasting (default: 1).

    Returns:
        torch.Tensor: The input tensor with rotary positional embedding applied.
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # [B, 1, T, d_rot]
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    dim = x.shape[-1]
    if rotary_dim != dim:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = x_rot * cos + rotate_half(x_rot) * sin
        x = torch.cat((x_rot, x_pass), dim=-1)
    else:
        x = (x * cos) + (rotate_half(x) * sin)
    return x

class Attention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int = 128,
        n_heads: int = 8,
        rotary_emb_dim: int = 8,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rotary_emb_dim = rotary_emb_dim

        # ensure rotary_emb_dim does not exceed head_dim and is even
        assert self.rotary_emb_dim <= self.head_dim, "rotary_emb_dim must be less than or equal to head_dim"
        assert self.rotary_emb_dim % 2 == 0, "rotary_emb_dim must be even"
        assert self.head_dim % 2 == 0, "head_dim must be even"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=self.dim,
            kernel_size=4,
            activation="silu",
        )

        self.k_conv1d = ShortConvolution(
            hidden_size=self.dim,
            kernel_size=4,
            activation="silu",
        )

        self.v_conv1d = ShortConvolution(
            hidden_size=self.dim,
            kernel_size=4,
            activation="silu"
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)

        self.is_causal = True   # used by flash attention
        self.config = FlashAttentionConfig()    # used by flash attention

    def forward(
        self,
        hidden_states: Float[torch.Tensor, 'b t d'],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        batch_size, q_len, _ = hidden_states.shape

        if use_cache:
            assert past_key_values is not None

        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            past_state = past_key_values[self.layer_idx]
            q_conv_state, k_conv_state, v_conv_state = past_state['conv_state']
            key_state, value_state = past_state['key_cache'], past_state['value_cache']
        else:
            q_conv_state, k_conv_state, v_conv_state = None, None, None
            key_state, value_state = None, None

        cu_seqlens = None
        if attention_mask is not None:      # only effective in inference
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        q = self.q_proj(hidden_states)
        q, q_conv_state = self.q_conv1d(q, cache=q_conv_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k = self.k_proj(hidden_states)
        k, k_conv_state = self.k_conv1d(k, cache=k_conv_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v = self.v_proj(hidden_states)
        v, v_conv_state = self.v_conv1d(v, cache=v_conv_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)

        if attention_mask is not None:
            q, k, v = map(lambda x: pad_input(x.squeeze(0), indices, batch_size, q_len), (q, k, v))

        q, k, v = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), (q, k, v))
        q = self.q_norm(q).transpose(1, 2).contiguous()
        k = self.k_norm(k).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        cos, sin = position_embeddings
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Save current-step k/v before concatenating with past cache
        current_k = k
        current_v = v

        if key_state is not None and value_state is not None:
            k = torch.cat([key_state, k], dim=-2)
            v = torch.cat([value_state, v], dim=-2)

        attention_interface = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        attn_output, _ = attention_interface(       # q, k, v in [B, H, T, d], but attn_output in [B, T, H, d]
            self,
            q, k, v,
            attention_mask
        )
        o_attn = attn_output.contiguous().view(-1, q_len, self.dim)
        o_attn = self.o_proj(o_attn)

        if use_cache:
            conv_state = None
            if q_conv_state is not None and k_conv_state is not None and v_conv_state is not None:
                conv_state = (q_conv_state, k_conv_state, v_conv_state)
            past_key_values.update(
                key_state=current_k,
                value_state=current_v,
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        return o_attn, None, past_key_values    # keep compatible for fla; reserved position for output_attention
