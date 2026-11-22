from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.mlp import GatedMLP
from fla.layers import GatedAntiHebbAttention
from fla.models.cache_utils import Cache
from jaxtyping import Float, Int
from torch.utils.checkpoint import checkpoint

from . import decoding
from .attention import Attention as AttentionBlock

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layers: int = 24
    dim: int = 1024
    rotary_dim: int = 128
    n_heads: int = 8
    expand_ratio: int = 4
    rotary_base: int = 10000
    key_expand: int = 2

@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[dict] = None
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None

class RotaryEmbedding(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.rotary_base = config.rotary_base

        assert self.rotary_dim % 2 == 0

        # [NOTE] inv_freq is NOT stored as a buffer: nn.Module.to(dtype) would cast it
        # to bfloat16 (only ~8 mantissa bits), and upcasting back to fp32 in forward
        # cannot recover the lost precision -> phase error grows with position
        # (measured ~126 rad at position 16384). Recompute it in fp32 every call instead.


    def forward(
        self, 
        hidden_state: Float[torch.Tensor, 'b t d'],     # RoPE actually does not use hidden states. Only to obtain the device
        position_ids: Int[torch.Tensor, 'b t']
    ) -> Tuple[
        Float[torch.Tensor, 'b t rd'], 
        Float[torch.Tensor, 'b t rd']
    ]:
        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, self.rotary_dim, 2, device=hidden_state.device, dtype=torch.float32) / self.rotary_dim))
        freqs = torch.einsum('b t, d->b t d', position_ids.to(torch.float32), inv_freq)
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
        self.rotary_dim = config.rotary_dim

        self.input_norm = nn.RMSNorm(self.dim)
        self.post_norm = nn.RMSNorm(self.dim)

        self.attn = GatedAntiHebbAttention(
            hidden_size=self.dim,
            head_dim=self.head_dim,
            num_heads=self.n_heads,
            layer_idx=layer_idx
        )
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

        hidden_states = hidden_states + x   # mark attn

        x = self.post_norm(hidden_states)
        x = self.mlp(x)
        hidden_states = hidden_states + x   # mark mlp

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
        past_key_values: Optional[Cache] = None,
        use_gradient_checkpoint: bool = False
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
            if use_gradient_checkpoint:
                hidden_states, past_key_values, aux_loss = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask=attention_mask, 
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values, 
                    use_cache=use_cache, 
                    use_reentrant=False
                )
            else:
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
        # lazily captured CUDA-graph decode engine, see module/decoding.py
        self._decode_engine = None
        self._decode_engine_signature = None

    def tie_weights(self):
        return None

    def get_input_embeddings(self):
        return self.model.emb
    
    def get_output_embeddings(self):
        return self.lm_head

    def get_output_norm(self):
        return self.model.norm

    def get_decoder(self):
        return self.model.layers

    def forward(
        self, 
        input_ids: Int[torch.Tensor, 'b t'], 
        position_ids: Optional[Int[torch.Tensor, 'b t']] = None,      # decided by generate(). when decoding, will contain only decoded part
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,    # decided by generate(). when decoding, will contain both prefill and generated part
        label: Optional[Int[torch.Tensor, 'b t']] = None,             # already shifted labels
        use_cache: bool = False, 
        past_key_values: Optional[Cache] = None,
        use_gradient_checkpoint: bool = False
    ):
        outputs = self.model(
            input_ids, 
            position_ids=position_ids,
            attention_mask=attention_mask, 
            use_cache=use_cache, 
            past_key_values=past_key_values,
            use_gradient_checkpoint=use_gradient_checkpoint
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
    
    def train(self, mode: bool = True):
        # captured decode graphs hold pointers to the weight storage; drop the
        # engine so a later generate() re-captures against the current weights
        self._decode_engine = None
        self._decode_engine_signature = None
        return super().train(mode)

    def _get_decode_engine(self, live_cache, batch_size: int, capacity: int, device: torch.device):
        """Return a captured decode engine for this state signature.

        Rebuilt only when the signature (batch size, capacity, device, layer
        state shapes) changes, so repeated generate() calls replay the same
        graph after re-seeding.
        """
        signature = decoding.decode_signature(live_cache, batch_size, capacity, device)
        if self._decode_engine is not None and self._decode_engine_signature == signature:
            return self._decode_engine
        engine = decoding.DecodeEngine(self, live_cache, batch_size, capacity, device)
        engine.signature = signature
        self._decode_engine = engine
        self._decode_engine_signature = signature
        return engine

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
        stopping_criteria=None,
        use_cuda_graph: Optional[bool] = None,
        eos_check_every: int = 1,
        **kwargs
    ):
        # The full loop (prefill, eager decode, CUDA-graph decode) lives in
        # module/decoding.py so every branch shares one implementation; this
        # wrapper only keeps the public API stable.
        return decoding.generate(
            self,
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            past_key_values=past_key_values,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cuda_graph=use_cuda_graph,
            eos_check_every=eos_check_every,
        )
