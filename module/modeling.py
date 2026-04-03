from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.mlp import GatedMLP
from jaxtyping import Float, Int

from .attention import Attention as AttentionBlock
from .cache_utils import Cache


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layers: int = 16
    dim: int = 1024
    rotary_dim: int = 128
    n_heads: int = 8
    expand_ratio: int = 4
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

class Layer(nn.Module):

    def __init__(self, layer_idx: int, config: ModelConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.input_norm = nn.RMSNorm(self.dim)
        self.post_norm = nn.RMSNorm(self.dim)

        self.attn = AttentionBlock(self.layer_idx, self.dim, self.n_heads)
        self.mlp = GatedMLP(hidden_size=config.dim, hidden_ratio=config.expand_ratio)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        x = self.input_norm(hidden_states)
        x, past_key_values = self.attn(
            x, 
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values, 
            use_cache=use_cache
        )
        hidden_states = hidden_states + x

        x = self.post_norm(hidden_states)
        x = self.mlp(x)
        hidden_states = hidden_states + x

        return hidden_states, past_key_values
        
class LanguageModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_layers % 2 == 0
        self.config = config
        self.n_layers = config.n_layers

        self.rotary_emb = RotaryEmbedding(config)
        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            Layer(i, config) for i in range(self.n_layers)
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

        if use_cache and past_key_values is None:
            past_key_values = Cache()

        hidden_states = self.emb(input_ids)
       
        if position_ids is None:
            # past_seen_tokens = layer_cache.get_sequence_length()   # [NOTE] not really past tokens. used only for training to decide position ids
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)    # maximum length in the batch
        position_ids = position_ids.to(hidden_states.device)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states, past_key_values = layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_embeddings=position_embeddings,
                past_key_values=past_key_values, 
                use_cache=use_cache, 
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