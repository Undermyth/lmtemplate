import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.mlp import GatedMLP
from jaxtyping import Float, Int
from torch.utils.checkpoint import checkpoint

from .attention import Attention as AttentionBlock
from fla.layers.kda import KimiDeltaAttention
from .cache_utils import Cache


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
        # follow the device only: `.to(position_ids)` would also cast the
        # frequencies to int64 and truncate them to zero.
        inv_freq = self.inv_freq.to(device=hidden_state.device, dtype=torch.float32)
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

        # self.attn = AttentionBlock(layer_idx=self.layer_idx, dim=self.dim, n_heads=self.n_heads, rotary_emb_dim=self.rotary_dim)
        self.attn = KimiDeltaAttention(
            hidden_size=self.dim,
            expand_v=1,
            head_dim=self.head_dim,
            num_heads=self.n_heads,
            layer_idx=self.layer_idx
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
        self._graph_decoder: Optional[dict] = None

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

    # ------------------------------------------------------------------ #
    #                          generation                                #
    # ------------------------------------------------------------------ #
    #
    # A decode step (1 layer x 24 layers of KDA) is a chain of ~360 tiny
    # kernels whose GPU time is ~2 ms but whose host time is ~24 ms, i.e. the
    # loop was launch-bound.  It is now replayed from a CUDA graph, which is
    # possible because every state tensor has a prompt-independent shape.
    # `LM_CUDA_GRAPH=0` disables the graph path.

    _MIN_GRAPH_TOKENS = 4
    _GRAPH_WARMUP_STEPS = 2

    @staticmethod
    def _cuda_graph_default() -> bool:
        flag = os.environ.get("LM_CUDA_GRAPH", "1").strip().lower()
        return flag not in {"0", "off", "false", "no", "none"}

    @staticmethod
    def _sample_next_token(
        next_token_logits: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if not do_sample:
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)

        logits = next_token_logits
        if temperature is not None and temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and 0 < top_k < logits.size(-1):
            kth_vals = torch.kthvalue(logits, logits.size(-1) - top_k + 1, dim=-1).values
            logits = torch.where(
                logits < kth_vals.unsqueeze(-1),
                torch.full_like(logits, float("-inf")),
                logits,
            )
        if top_p is not None and top_p < 1.0:
            sorted_logits, indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.gather(sorted_logits, dim=-1, index=indices)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _state_signature(cache: Cache, batch_size: int, device: torch.device):
        signature = [batch_size, str(device)]
        for i in range(len(cache)):
            layer = cache[i]
            recurrent = layer["recurrent_state"]
            signature.append((tuple(recurrent.shape), recurrent.dtype, str(recurrent.device)))
            signature.append(tuple((tuple(c.shape), c.dtype) for c in layer["conv_state"]))
        return tuple(signature)

    @staticmethod
    def _copy_state(src: Cache, dst: Cache) -> None:
        for i in range(len(src)):
            src_layer, dst_layer = src[i], dst[i]
            dst_layer["recurrent_state"].copy_(src_layer["recurrent_state"])
            for dst_conv, src_conv in zip(dst_layer["conv_state"], src_layer["conv_state"]):
                dst_conv.copy_(src_conv)

    def _get_graph_decoder(self, template_cache: Cache, batch_size: int, device: torch.device) -> dict:
        """Capture (once per state signature) a CUDA graph for one q_len=1 step."""
        signature = self._state_signature(template_cache, batch_size, device)
        decoder = self._graph_decoder
        if decoder is not None and decoder["signature"] == signature:
            return decoder

        static_cache = Cache()
        static_state = []
        for i in range(len(template_cache)):
            layer = template_cache[i]
            recurrent = torch.zeros_like(layer["recurrent_state"])
            conv = tuple(torch.zeros_like(c) for c in layer["conv_state"])
            static_cache.cache.append({"recurrent_state": recurrent, "conv_state": conv})
            static_state.append((recurrent, conv))

        static_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        static_pos = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        static_tok = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        def graph_step():
            outputs = self.model(
                input_ids=static_ids,
                position_ids=static_pos,
                attention_mask=None,
                use_cache=True,
                past_key_values=static_cache,
            )
            logits = self.lm_head(outputs[0])
            static_tok.copy_(torch.argmax(logits[:, -1, :], dim=-1, keepdim=True))
            # `fused_recurrent_kda` allocates a new recurrent state on every call
            # (it passes `inplace_final_state=False`), so fold it back into the
            # static buffers; otherwise a replay would keep reading the state
            # captured at capture time.
            for layer_idx, (recurrent, conv) in enumerate(static_state):
                layer = static_cache[layer_idx]
                if layer["recurrent_state"] is not recurrent:
                    recurrent.copy_(layer["recurrent_state"])
                    layer["recurrent_state"] = recurrent
                if layer["conv_state"] is not conv:
                    for dst_conv, src_conv in zip(conv, layer["conv_state"]):
                        if dst_conv is not src_conv:
                            dst_conv.copy_(src_conv)
                    layer["conv_state"] = conv

        # warm up outside the capture stream (lazy allocations + kernel JIT)
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(self._GRAPH_WARMUP_STEPS):
                graph_step()
        torch.cuda.current_stream(device).wait_stream(side_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_step()
        torch.cuda.synchronize(device)

        decoder = {
            "signature": signature,
            "graph": graph,
            "cache": static_cache,
            "ids": static_ids,
            "pos": static_pos,
            "tok": static_tok,
        }
        self._graph_decoder = decoder
        return decoder

    def _decode_greedy_cuda_graph(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache,
        next_token_logits: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        stopping_criteria=None,
        seq_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        decoder = self._get_graph_decoder(past_key_values, batch_size, device)

        static_cache = decoder["cache"]
        self._copy_state(past_key_values, static_cache)
        static_cache.set_sequence_length(past_key_values.get_sequence_length())

        static_ids, static_pos, static_tok = decoder["ids"], decoder["pos"], decoder["tok"]
        graph = decoder["graph"]

        length = past_key_values.get_sequence_length()
        # the first token comes from the prefill logits, every following one
        # needs one replay (feed token i-1 -> get token i)
        token = int(torch.argmax(next_token_logits, dim=-1, keepdim=True))
        tokens: list[int] = []
        while True:
            tokens.append(token)
            length += 1
            if eos_token_id is not None and token == eos_token_id:
                break
            if stopping_criteria is not None:
                seq_buffer[:, length - 1] = token
                if bool(stopping_criteria(seq_buffer[:, :length], None)):
                    break
            if len(tokens) >= max_new_tokens:
                break
            static_ids.fill_(token)
            static_pos.fill_(length - 1)
            graph.replay()
            token = int(static_tok)

        generated = torch.tensor([tokens], dtype=torch.long, device=device)
        return torch.cat([input_ids, generated], dim=-1)

    def _decode_eager(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        next_token_logits: torch.Tensor,
        past_key_values: Optional[Cache],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id: Optional[int],
        stopping_criteria=None,
        seq_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        incremental = past_key_values is not None

        if incremental:
            length = past_key_values.get_sequence_length()
        else:
            length = input_ids.shape[1]
        position_buffer = torch.empty(batch_size, 1, dtype=torch.long, device=device)

        # an all-ones mask is a no-op for the kernels but costs a nonzero +
        # `.item()` device sync per layer per step, so drop it while decoding
        decode_mask = None
        if attention_mask is not None and not bool(attention_mask.all()):
            decode_mask = attention_mask

        out_tokens: list[torch.Tensor] = []
        logits = next_token_logits
        for step in range(max_new_tokens):
            next_token = self._sample_next_token(logits, do_sample, temperature, top_k, top_p)
            out_tokens.append(next_token)
            length += 1

            stop = False
            if eos_token_id is not None:
                if batch_size == 1:
                    stop = int(next_token) == eos_token_id
                else:
                    stop = bool((next_token == eos_token_id).all())
            if not stop and stopping_criteria is not None:
                if seq_buffer is not None:
                    seq_buffer[:, length - 1] = next_token[:, 0]
                    stop = bool(stopping_criteria(seq_buffer[:, :length], None))
                else:
                    seq = torch.cat([input_ids] + out_tokens, dim=-1)
                    stop = bool(stopping_criteria(seq, None))
            if stop or step == max_new_tokens - 1:
                break

            if incremental:
                position_buffer.fill_(length - 1)
                outputs = self.model(
                    input_ids=next_token,
                    position_ids=position_buffer,
                    attention_mask=decode_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            else:
                seq = torch.cat([input_ids] + out_tokens, dim=-1)
                positions = torch.arange(seq.shape[1], device=device).unsqueeze(0)
                outputs = self.model(
                    input_ids=seq,
                    position_ids=positions,
                    attention_mask=decode_mask,
                    use_cache=False,
                    past_key_values=None,
                )
            hidden_states, past_key_values = outputs[0], outputs[1]
            logits = self.lm_head(hidden_states)[:, -1, :]

        return torch.cat([input_ids] + out_tokens, dim=-1)

    @torch.no_grad()
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
        use_cuda_graph: Optional[bool] = None,
        stopping_criteria=None,
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
        max_new_tokens = int(min(max_new_tokens, target_length - input_ids.shape[1]))
        if max_new_tokens <= 0:
            return input_ids

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Prefill phase
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        hidden_states, past_key_values = outputs[0], outputs[1]
        logits = self.lm_head(hidden_states)[:, -1, :]

        # stop sequences (lm-eval passes `until=[...]` through here)
        criteria = stopping_criteria
        if criteria is not None and hasattr(criteria, "__len__") and len(criteria) == 0:
            criteria = None

        seq_buffer = None
        if criteria is not None:
            seq_buffer = torch.empty(batch_size, seq_len + max_new_tokens, dtype=torch.long, device=device)
            seq_buffer[:, :seq_len] = input_ids

        if use_cuda_graph is None:
            use_cuda_graph = self._cuda_graph_default()

        graph_ready = (
            use_cuda_graph
            and input_ids.is_cuda
            and batch_size == 1
            and not do_sample
            and past_key_values is not None
            and max_new_tokens >= self._MIN_GRAPH_TOKENS
            and (attention_mask is None or bool(attention_mask.all()))
            and len(past_key_values) > 0
            and isinstance(past_key_values[0], dict)
            and "recurrent_state" in past_key_values[0]
            and past_key_values[0].get("conv_state") is not None
        )
        if graph_ready:
            try:
                return self._decode_greedy_cuda_graph(
                    input_ids, past_key_values, logits, max_new_tokens,
                    eos_token_id, criteria, seq_buffer,
                )
            except Exception as exc:      # pragma: no cover - defensive
                warnings.warn(f"CUDA-graph decoding disabled ({type(exc).__name__}: {exc}); "
                              f"falling back to the eager decode loop.", stacklevel=2)
                self._graph_decoder = None

        return self._decode_eager(
            input_ids, attention_mask, logits, past_key_values, max_new_tokens,
            do_sample, temperature, top_k, top_p, eos_token_id, criteria, seq_buffer,
        )
