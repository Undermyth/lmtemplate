import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional, Union
from jaxtyping import Float

from fla.modules import ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.layers.utils import get_unpad_data, index_first_axis
from fla.ops.deltaformer import deltaformer_attn
from fla.layers import DeltaFormerAttention

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .cache_utils import Cache

class Heaviside(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        # input, = ctx.saved_tensors
        # s = torch.sigmoid(input)
        # return 2 * s * (1 - s)

heaviside = Heaviside.apply

class STE(torch.autograd.Function):
    '''
    straight through estimator
    '''

    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ste = STE.apply

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
        x = torch.cat((-x_rot, x_pass), dim=-1)
    else:
        x = (x * cos) + (rotate_half(x) * sin)
    return x

class Switcher(nn.Module):
    def __init__(self, dim: int = 128, n_heads: int = 4, rotary_emb_dim: int = 8, **kwargs):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rotary_emb_dim = rotary_emb_dim

        # 确保 rotary_emb_dim 不超过 head_dim，并且是偶数
        assert self.rotary_emb_dim <= self.head_dim, "rotary_emb_dim must be less than or equal to head_dim"
        assert self.rotary_emb_dim % 2 == 0, "rotary_emb_dim must be even"
        assert self.head_dim % 2 == 0, "head_dim must be even"

        # 线性变换层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)

        self.b_proj = nn.Linear(dim, n_heads, bias=False)

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

        self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-5)

        self.s_proj = nn.Linear(self.dim, self.n_heads, bias=False)

        self.threshold = nn.Parameter(torch.ones(1, 1, self.n_heads, 1).cuda() * 0.5)

        self.mode = 'random'

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, _ = self.q_conv1d(q, output_final_state=False)
        k, _ = self.k_conv1d(k, output_final_state=False)
        v, _ = self.v_conv1d(v, output_final_state=False)

        beta = self.b_proj(hidden_states).sigmoid()

        q, k, v = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), (q, k, v))

        # [VARI] should qk norm also be applied on softmax attention?
        o, _ = chunk_delta_rule(
            q=q, k=k, v=v, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # q, k = self.rope(q, k)

        o_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o_attn = o_attn.transpose(1, 2).contiguous()
        o_attn = self.o_norm(o_attn)

        o = self.o_norm(o)      # [VARI] should norm also be shared between attentions?

        if self.mode == 'random':
            s = torch.ones(o_attn.shape[0], o_attn.shape[1], o_attn.shape[2], 1, device=o.device) * 0.3
            s = torch.bernoulli(s).detach()
        else:
            o_reshaped = rearrange(o, "b t h d -> b t (h d)")
            s = self.s_proj(o_reshaped).sigmoid().unsqueeze(-1)    # [VARI] maybe before norm?
            s = heaviside(s - self.threshold)

        if self.mode == 'quadratic':
            o = o_attn
        elif self.mode == 'hybrid' or self.mode == 'random':
            o = (1 - s) * o + s * o_attn  # [VARI] maybe the inverse?
        else:
            assert self.mode == 'linear'
        
        g = self.g_proj(hidden_states)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        o = o * F.sigmoid(g)

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        return o, s

class CrossSwitcher(nn.Module):
    def __init__(self, layer_idx: int, dim: int = 128, n_heads: int = 4, **kwargs):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.layer_idx = layer_idx

        assert self.head_dim % 2 == 0, "head_dim must be even"

        # 线性变换层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)

        self.b_proj = nn.Linear(dim, n_heads, bias=False)

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

        self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-5)

        self.s_proj = nn.Linear(self.dim, self.n_heads, bias=False)

        self.threshold = nn.Parameter(torch.ones(1, 1, self.n_heads, 1).cuda() * 0.5)

        self.mode = 'random'
        # self.mode = 'softmax'
        self.s_proj.weight.requires_grad_(False)
        self.threshold.requires_grad_(False)

        self.is_causal = True   # used for huggingface flash attention option

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        force_attn: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,                # do not support attention_mask by default. use cu_seqlens instead
        past_key_values: Optional[Cache] = None, 
        use_cache: bool = False, 
        *args, **kwargs
    ):
        '''
        past_key_values are used by linear attention. Only used for decoding during inference.
        k_cache, v_cache are from encoder. Used both for training and inference.
        '''
        q_len = hidden_states.shape[1]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if use_cache and len(past_key_values) > self.layer_idx:
            assert past_key_values is not None
            past_state = past_key_values[self.layer_idx]
            q_conv_state, k_conv_state, v_conv_state = past_state['conv_state']
            recurrent_state = past_state['recurrent_state']
        else:
            q_conv_state = None
            k_conv_state = None
            v_conv_state = None
            recurrent_state = None

        q, q_conv_state = self.q_conv1d(q, cache=q_conv_state, cu_seqlens=cu_seqlens, output_final_state=use_cache)
        k, k_conv_state = self.k_conv1d(k, cache=k_conv_state, cu_seqlens=cu_seqlens, output_final_state=use_cache)
        v, v_conv_state = self.v_conv1d(v, cache=v_conv_state, cu_seqlens=cu_seqlens, output_final_state=use_cache)

        q, k, v = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), (q, k, v))
        beta = self.b_proj(hidden_states).sigmoid()

        # # [VARI] should qk norm also be applied on softmax attention?
        # gamma = torch.relu(self.a_proj(hidden_states).float() + self.dt_bias)
        # gamma = 2 / (torch.exp(gamma) + torch.exp(-gamma))
        # gamma = gamma.log()
        if self.training and self.mode == 'random':
            random_gate = torch.rand(1).item()
            random_use_attn = (random_gate > 0.7) or force_attn
        else:
            random_use_attn = False
            
        if not random_use_attn and self.mode != 'softmax':    # we will need to skip the computation of linear part iff under random selection or softmax mode 
            if q_len > 64:
                o, recurrent_state = chunk_delta_rule(
                    q=q, k=k, v=v, beta=beta, initial_state=recurrent_state, cu_seqlens=cu_seqlens, output_final_state=use_cache, use_qk_l2norm_in_kernel=True
                )
            else:
                o, recurrent_state = fused_recurrent_delta_rule(
                    q=q, k=k, v=v, beta=beta, initial_state=recurrent_state, cu_seqlens=cu_seqlens, output_final_state=use_cache, use_qk_l2norm_in_kernel=True
                )

            if use_cache:
                past_key_values.update(
                    recurrent_state,
                    (q_conv_state, k_conv_state, v_conv_state), 
                    self.layer_idx, 
                    q_len
                )

        # ------------------------------------------------------------------------------
        # hybrid attention into output
        # ------------------------------------------------------------------------------
        s = None
        if self.training or q_len == 1 or self.mode == 'softmax':     # in inference, we do not allow any attention in prefilling, only switching in decoding
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.mode == 'hybrid':
                o = self.o_norm(o)
                o_reshaped = rearrange(o, "b t h d -> b t (h d)")
                s = self.s_proj(o_reshaped).sigmoid().unsqueeze(-1)    # [VARI] maybe before norm?
                s = heaviside(s - self.threshold)
            if (self.mode == 'random' and random_use_attn) or self.mode == 'hybrid' or self.mode == 'softmax':
                q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
                # k = k / torch.linalg.norm(k, dim=-1, keepdim=True)
                cos, sin = position_embeddings
                q = apply_rotary_pos_emb(q, cos, sin)   # k will be embedded in the kv cache
                # k = apply_rotary_pos_emb(k, cos, sin)

                q = q.transpose(1, 2).contiguous().bfloat16()
                # k = k.transpose(1, 2).contiguous().bfloat16()
                # v = v.transpose(1, 2).contiguous().bfloat16()
                k_cache = k_cache.transpose(1, 2).contiguous().bfloat16()
                v_cache = v_cache.transpose(1, 2).contiguous().bfloat16()
                o_attn = deltaformer_attn(
                    q, k_cache, v_cache,
                    beta,
                    attention_mask=None,
                    cu_seqlens=cu_seqlens
                )
                # attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
                # o_attn, attn_weights = attention_interface(
                #     self,                        # used to determine dtype in huggingface wrapper
                #     q, k_cache, v_cache,
                #     attention_mask=None,
                #     cu_seq_lens_q=cu_seqlens,
                #     cu_seq_lens_k=cu_seqlens
                # )
                # o_attn = o_attn.transpose(1, 2).contiguous()
                o_attn = self.o_norm(o_attn)
                if self.mode == 'random' or self.mode == 'softmax':
                    o = o_attn
                else:
                    o = self.o_norm(o)
                    o = (1 - s) * o + s * o_attn
            else:
                o = self.o_norm(o)

        else:    # inference, prefilling
            o = self.o_norm(o)

            # o = self.o_norm(o)      # [VARI] should norm also be shared between attentions?

            # if self.mode == 'random':
            #     s = torch.ones(o_attn.shape[0], o_attn.shape[1], o_attn.shape[2], 1, device=o.device) * 0.3
            #     s = torch.bernoulli(s).detach()
            # else:
            #     o_reshaped = rearrange(o, "b t h d -> b t (h d)")
            #     s = self.s_proj(o_reshaped).sigmoid().unsqueeze(-1)    # [VARI] maybe before norm?
            #     s = heaviside(s - self.threshold)

            # if self.mode == 'quadratic':
            #     o = o_attn
            # elif self.mode == 'hybrid' or self.mode == 'random':
            #     o = (1 - s) * o + s * o_attn  # [VARI] maybe the inverse?
            # else:
            #     assert self.mode == 'linear'

        if self.training:
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            # o = o + 0 * (beta.unsqueeze(-1) + gamma.unsqueeze(-1) + k + v)    # prevent unused parameter warning in DDP
            o = o + 0 * (beta.unsqueeze(-1) + k + v)    # prevent unused parameter warning in DDP
        
        g = self.g_proj(hidden_states)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        o = o * F.sigmoid(g)

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        return o, s

class DeltaFormerBlockWrapper(nn.Module):
    def __init__(self, layer_idx, dim, n_heads):
        super().__init__()
        self.block = DeltaFormerAttention(hidden_size=dim, num_heads=n_heads, qk_norm=True, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        self.dim = dim
        self.n_heads = n_heads

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        # k_cache: torch.Tensor, 
        # v_cache: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        force_attn: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,                # do not support attention_mask by default. use cu_seqlens instead
        past_key_values: Optional[Cache] = None, 
        use_cache: bool = False, 
        *args, **kwargs
    ):
        o, _, _ = self.block(
            hidden_states=hidden_states,
        )
        return o, None
        
