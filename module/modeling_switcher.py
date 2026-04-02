import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from fla.layers.utils import get_unpad_data, index_first_axis
from fla.models.delta_net.configuration_delta_net import DeltaNetConfig
from fla.models.delta_net.modeling_delta_net import DeltaNetBlock
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.mlp import GatedMLP
from jaxtyping import Float, Int

from .cache_utils import Cache
from .switcher import CrossSwitcher, apply_rotary_pos_emb


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layers: int = 24
    dim: int = 1024
    rotary_dim: int = 128
    n_heads: int = 8
    rotary_base: int = 10000

@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[dict] = None
    loss: Optional[torch.Tensor] = None

class RotaryEmbedding(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.rotary_base = config.rotary_base

        assert self.rotary_dim % 2 == 0

        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)


    def forward(
        self, 
        hidden_state: Float[torch.Tensor, 'b t d'],     # RoPE actually does not use hidden states. Only to obtain the device
        position_ids: Int[torch.Tensor, 'b t']
    ) -> Tuple[
        Float[torch.Tensor, 'b t rd'], 
        Float[torch.Tensor, 'b t rd']
    ]:
        self.inv_freq = self.inv_freq.to(position_ids)
        freqs = torch.einsum('b t, d->b t d', position_ids, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos.to(hidden_state), sin.to(hidden_state)

class SwitcherLayer(nn.Module):

    def __init__(self, layer_idx: int, config: ModelConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.input_norm = nn.RMSNorm(self.dim)
        self.post_norm = nn.RMSNorm(self.dim)

        self.attn = CrossSwitcher(self.layer_idx, self.dim, self.n_heads)
        self.mlp = GatedMLP(hidden_size=config.dim, hidden_ratio=config.expand_ratio)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        force_attn: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        layer_cache: Optional[Cache] = None, 
        use_cache: bool = False,
    ):
        x = self.input_norm(hidden_states)
        x, s = self.attn(
            x, 
            k_cache=k_cache, 
            v_cache=v_cache,
            position_embeddings=position_embeddings,
            force_attn=force_attn,
            cu_seqlens=cu_seqlens,
            past_key_values=layer_cache, 
            use_cache=use_cache
        )
        hidden_states = hidden_states + x

        x = self.post_norm(hidden_states)
        x = self.mlp(x)
        hidden_states = hidden_states + x

        return hidden_states, layer_cache
        
class LanguageModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_layers % 2 == 0
        self.config = config
        self.n_layers = config.n_layers

        delta_config = DeltaNetConfig(
            hidden_size=config.dim,
            expand_v=1,
            num_heads=config.n_heads,
            head_dim=config.dim // config.n_heads,
            hidden_ratio=config.expand_ratio
        )

        self.rotary_emb = RotaryEmbedding(config)

        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            SwitcherLayer(i, config) if i % config.hybrid_freq == 0 else DeltaNetBlock(config=delta_config, layer_idx=i) 
            for i in range(self.n_dec_layers)
        ])
        self.norm = nn.RMSNorm(config.dim)

    def forward(
        self, 
        input_ids: Int[torch.Tensor, 'b t'], 
        position_ids: Optional[Int[torch.Tensor, 'b t']] = None,      # decided by generate(). when decoding, will contain only decoded part
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,    # decided by generate(). when decoding, will contain both prefill and generated part
        use_cache: bool = False, 
        past_key_values: Optional[Cache] = None
    ):

        layer_cache = None
        if use_cache and past_key_values is None:
            layer_cache = Cache()
            past_key_values = {
                'k_cache': None,
                'v_cache': None,
                'layer_cache': layer_cache,    # recurrent cache, for rnn state [B, H, d, d] * L and conv state [B, Hd, conv_d] * L 
            }

        hidden_states = self.emb(input_ids)
        
        cu_seqlens = None
        if attention_mask is not None:
            input_length = hidden_states.shape[1]
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -input_length:])    # padding is on the left during inference. Disabled in training.
            # hidden_states: [B, T, d] -> [1, (total_length), d]
            hidden_states = index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'), indices).unsqueeze(0)
            
        if position_ids is None:
            # past_seen_tokens = layer_cache.get_sequence_length()   # [NOTE] not really past tokens. used only for training to decide position ids
            if attention_mask is not None:
                position_ids = torch.cumsum(attention_mask, dim=1).long() - 1
                position_ids = index_first_axis(position_ids.flatten().unsqueeze(-1), indices).squeeze(-1).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)    # maximum length in the batch
        position_ids = position_ids.to(hidden_states.device)

        attention_mask = None    # only use cu_seqlens instead of attention_mask for varlen inference

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states, kv_cache = layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_embeddings=position_embeddings,
                past_key_values=kv_cache, 
                use_cache=use_cache, 
                cu_seqlens=cu_seqlens
            )
            
        hidden_states = self.norm(hidden_states)
        # return ModelOutput(last_hidden_state=hidden_states, past_key_values=past_key_values)
        return hidden_states, past_key_values

class ModelForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = LanguageModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.criteria = FusedLinearCrossEntropyLoss()

    def forward(
        self, 
        input_ids: Int[torch.Tensor, 'b t'], 
        position_ids: Optional[Int[torch.Tensor, 'b t']] = None,      # decided by generate(). when decoding, will contain only decoded part
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,    # decided by generate(). when decoding, will contain both prefill and generated part
        label: Optional[Int[torch.Tensor, 'b t']] = None,             # already shifted labels
        use_cache: bool = False, 
        past_key_values: Optional[Cache] = None
    ):
        outputs = self.model(
            input_ids, 
            position_ids=position_ids,
            attention_mask=attention_mask, 
            use_cache=use_cache, 
            past_key_values=past_key_values
        )
        hidden_states = outputs[0]
        past_key_values = outputs[1]
        
        logits = None
        if not self.training:
            logits = self.lm_head(hidden_states)
            
        loss = None
        if label is not None:
            # label = torch.cat((label[..., 1:], torch.full_like(label[:, :1], self.criterion.ignore_index)), dim=1)
            loss = self.criteria(hidden_states, label, self.lm_head.weight, self.lm_head.bias)
            
        return ModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            loss=loss
        )