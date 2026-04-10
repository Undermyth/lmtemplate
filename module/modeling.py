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
        attn_output = self.attn(
            x, 
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values, 
            use_cache=use_cache
        )

        aux_loss = None
        if len(attn_output) == 3:
            x, _, past_key_values = attn_output
        else:
            x, _, past_key_values, aux_loss = attn_output

        hidden_states = hidden_states + x

        x = self.post_norm(hidden_states)
        x = self.mlp(x)
        hidden_states = hidden_states + x

        return hidden_states, past_key_values, aux_loss
        
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
        total_aux_loss = None
        for layer in self.layers:
            hidden_states, past_key_values, aux_loss = layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_embeddings=position_embeddings,
                past_key_values=past_key_values, 
                use_cache=use_cache, 
            )
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss if total_aux_loss is not None else aux_loss
            
        hidden_states = self.norm(hidden_states)
        # return ModelOutput(last_hidden_state=hidden_states, past_key_values=past_key_values)
        return hidden_states, past_key_values, total_aux_loss

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
        aux_loss = outputs[2]
        
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
            loss=loss,
            aux_loss=aux_loss
        )
    
    def generate(
        self,
        input_ids: Int[torch.Tensor, 'b t'],
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        use_cache: bool = True,
        past_key_values: Optional[Cache] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        eos_token_id: int = 2,
        **kwargs
    ):
        # Determine stopping condition
        if max_new_tokens is None and max_length is None:
            max_new_tokens = 100

        if max_new_tokens is not None:
            target_length = input_ids.shape[1] + max_new_tokens
        else:
            target_length = max_length
            max_new_tokens = max_length - input_ids.shape[1]

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Prefill phase
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        hidden_states = outputs[0]
        past_key_values = outputs[1]

        # Get logits for next token prediction
        logits = self.lm_head(hidden_states)
        next_token_logits = logits[:, -1, :]

        # Initialize generated sequence
        generated = input_ids

        # Decode loop
        while past_key_values.get_sequence_length() < target_length:
            # Apply sampling logic
            if do_sample:
                # Temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Top-k
                if top_k > 0:
                    kth_vals = torch.kthvalue(next_token_logits, logits.size(-1) - top_k + 1, dim=-1).values
                    next_token_logits = torch.where(next_token_logits < kth_vals.unsqueeze(-1), torch.full_like(next_token_logits, float('-inf')), next_token_logits)

                # Top-p (nucleus)
                if top_p < 1.0:
                    sorted_logits, indices = torch.sort(next_token_logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumsum = torch.cumsum(probs, dim=-1)
                    mask = cumsum > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
                    next_token_logits = torch.gather(sorted_logits, dim=-1, index=indices)

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check EOS
            if (next_token == eos_token_id).all():
                break

            # Prepare for next iteration
            next_token_id = next_token
            next_position_ids = torch.tensor([past_key_values.get_sequence_length()]).unsqueeze(0).expand(batch_size, -1).to(device)

            # Extend attention mask (all ones since no padding in generation)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)], dim=-1)

            # Forward pass with cached states
            outputs = self.model(
                input_ids=next_token_id,
                position_ids=next_position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            hidden_states = outputs[0]
            past_key_values = outputs[1]

            # Get logits
            logits = self.lm_head(hidden_states)
            next_token_logits = logits[:, -1, :]

        return generated