"""
Fast generation for lmtemplate models.

This module owns the generation loop, the sampling logic and the CUDA-graph
decode engine, so every branch of the repo shares one implementation.
Model-family-specific behaviour lives in the layer forwards (see the static
decode branch in module/attention.py); the loop, the engine and the cache
container here are agnostic to whether the model is attention-based
(append-style KV cache) or recurrent (replace-style state).

Cache contract for the graph path
---------------------------------
`StaticDecodeCache` subclasses fla's `Cache` and keeps fla's per-layer state
dict layout (recurrent_state / attn_state / conv_state / ffn_state), but the
buffers are preallocated with fixed shapes and the update semantics differ
from the dynamic cache:

* `attn_state` holds a key/value buffer of shape [batch, capacity, h*d] and
  is APPEND-style: the per-step append happens inside
  `flash_attn_with_kvcache` (in-kernel, CUDA-graph safe), so `update()`
  ignores any incoming `attn_state`.
* `recurrent_state` / `conv_state` / `ffn_state` are REPLACE-style
  fixed-shape buffers: `update()` copies incoming tensors into the
  preallocated buffers (generic fold-back) so that kernels which return a
  freshly allocated state on every call stay graph-safe.

The decode graph is captured once per state signature (batch size, capacity,
device, layer state shapes) and replayed per token; `LM_CUDA_GRAPH=0` forces
the eager loop.
"""
import os
import warnings
from typing import Optional

import torch
from fla.models.cache_utils import FLALayer, Cache

__all__ = [
    "sample_token",
    "generate",
    "decode_signature",
    "DecodeEngine",
    "StaticDecodeCache",
    "cuda_graph_default",
]

# A decode graph costs ~100 ms to capture; below this many new tokens the
# capture overhead cannot pay off, so fall back to the eager loop.
_MIN_GRAPH_TOKENS = 4
_GRAPH_WARMUP_STEPS = 2
_STATE_KEYS = ("recurrent_state", "attn_state", "conv_state", "ffn_state")


def cuda_graph_default() -> bool:
    flag = os.environ.get("LM_CUDA_GRAPH", "1").strip().lower()
    return flag not in {"0", "off", "false", "no", "none"}


def normalize_stopping_criteria(criteria):
    """lm-eval passes `until=[]` through `stopping_criteria`; treat as absent."""
    if criteria is not None and hasattr(criteria, "__len__") and len(criteria) == 0:
        return None
    return criteria


def resolve_new_tokens(seq_len: int, max_length: Optional[int], max_new_tokens: Optional[int]) -> int:
    if max_new_tokens is None and max_length is None:
        max_new_tokens = 100
    if max_new_tokens is None:
        max_new_tokens = max_length - seq_len
    return int(max_new_tokens)


def sample_token(
    next_token_logits: torch.Tensor,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Return the next token [batch, 1]; runs entirely on-device."""
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


def _state_key(value):
    """Cheap metadata key of a (possibly nested) state value."""
    if isinstance(value, (tuple, list)):
        return tuple(_state_key(v) for v in value)
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), value.dtype)
    return type(value).__name__


def decode_signature(live_cache, batch_size: int, capacity: int, device: torch.device):
    """Identify one captured graph: only state metadata, no tensor copies."""
    layers = tuple(
        tuple(_state_key(layer.get(key)) for key in _STATE_KEYS)
        for layer in live_cache
    )
    return (batch_size, capacity, str(device), layers)


def _sync_static(static, new):
    """Copy `new`'s tensors into the preallocated `static` structure in place."""
    if static is None or new is None:
        return static
    if isinstance(static, (tuple, list)):
        if not isinstance(new, (tuple, list)):
            return static
        for s, n in zip(static, new):
            _sync_static(s, n)
        return static
    if isinstance(static, torch.Tensor) and isinstance(new, torch.Tensor) and new is not static:
        static.copy_(new)
    return static


class StaticDecodeCache(Cache):
    """fla `Cache` with preallocated fixed-shape buffers for graph decoding.

    Update semantics are described in the module docstring: attention KV
    buffers are appended in-kernel (incoming `attn_state` ignored), while
    recurrent/conv/ffn states are replaced via in-place copy into the
    preallocated buffers.
    """

    def __init__(self):
        super().__init__()
        self.capacity = 0
        self.cache_seqlens = None   # int32 [batch] device tensor, engine-owned
        self.static_states = []     # per layer: dict key -> preallocated buffers

    def update(
        self,
        recurrent_state=None,
        attn_state=None,
        conv_state=None,
        ffn_state=None,
        layer_idx: int = 0,
        offset: int = 1,
        cache_kwargs: Optional[dict] = None,
    ):
        # The KV buffers are maintained in place by the attention kernel; the
        # cat-based update of the base class must not run during graph decoding.
        attn_state = None
        state = super().update(
            recurrent_state=recurrent_state,
            attn_state=None,
            conv_state=conv_state,
            ffn_state=ffn_state,
            layer_idx=layer_idx,
            offset=offset,
            cache_kwargs=cache_kwargs,
        )
        self._fold_back(layer_idx)
        return state

    def _fold_back(self, layer_idx: int):
        static = self.static_states[layer_idx]
        state = self.layers[layer_idx].state
        for key, buf in static.items():
            if buf is None:
                continue
            state[key] = _sync_static(buf, state.get(key))


class DecodeEngine:
    """Owns the static decode cache and the captured CUDA graph of one step.

    Built from the post-prefill dynamic cache; captures a q_len=1 decode step
    whose body is fully device-driven (position ids and cache lengths
    self-advance inside the graph), so a replay needs only the new token id.
    Reused across `generate` calls with the same state signature; the caller
    must `seed()` it before each generation.
    """

    def __init__(self, model, live_cache, batch_size: int, capacity: int, device: torch.device):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.capacity = capacity
        self.graph = None
        self.signature = None

        cache = StaticDecodeCache()
        cache.capacity = capacity
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        cache.cache_seqlens = self.cache_seqlens

        self.static_states = []
        for layer_idx in range(len(live_cache)):
            src = live_cache[layer_idx]
            static = {}
            attn_state = src.get("attn_state")
            if attn_state is not None:
                flat_dim = attn_state[0].shape[-1]
                k_buf = torch.zeros(batch_size, capacity, flat_dim, dtype=attn_state[0].dtype, device=device)
                v_buf = torch.zeros_like(k_buf)
                static["attn_state"] = (k_buf, v_buf)
            for key in ("recurrent_state", "conv_state", "ffn_state"):
                value = src.get(key)
                if value is None:
                    static[key] = None
                elif isinstance(value, (tuple, list)):
                    static[key] = type(value)(torch.zeros_like(t) for t in value)
                else:
                    static[key] = torch.zeros_like(value)
            self.static_states.append(static)

        cache.static_states = self.static_states
        self.cache = cache

        vocab = model.lm_head.out_features
        self.ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.tok = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.logits = torch.zeros(batch_size, vocab, dtype=model.lm_head.weight.dtype, device=device)

        self.seed(live_cache)    # meaningful state for warmup
        self._capture()
        # warmup + capture advanced the state; the caller re-seeds for real use

    # ------------------------------------------------------------------ #
    # seeding                                                            #
    # ------------------------------------------------------------------ #

    def seed(self, live_cache):
        """Reset all static state from the post-prefill dynamic cache."""
        seq_len = live_cache.get_seq_length()

        layers = []
        for static in self.static_states:
            layer = FLALayer()
            # Bypass FLALayer.update(): _seen_tokens must count valid tokens
            # (seq_len), not the capacity-sized attn buffer.
            layer.state = {key: static.get(key) for key in _STATE_KEYS}
            layer._seen_tokens = seq_len
            layer.is_initialized = True
            layer.device = self.device
            layers.append(layer)
        self.cache.layers = layers
        self.cache._seen_tokens = seq_len

        self.cache_seqlens.fill_(seq_len)
        self.pos.fill_(seq_len)
        self.ids.zero_()

        for layer_idx, static in enumerate(self.static_states):
            src = live_cache[layer_idx]
            src_attn = src.get("attn_state")
            if src_attn is not None:
                k_buf, v_buf = static["attn_state"]
                k_buf[:, :seq_len].copy_(src_attn[0])
                v_buf[:, :seq_len].copy_(src_attn[1])
            for key in ("recurrent_state", "conv_state", "ffn_state"):
                if static[key] is not None:
                    _sync_static(static[key], src.get(key))

    # ------------------------------------------------------------------ #
    # graph capture                                                      #
    # ------------------------------------------------------------------ #

    def _step_body(self):
        """One q_len=1 decode step; every tensor read/written is static."""
        hidden_states, _, _ = self.model.model(
            input_ids=self.ids,
            position_ids=self.pos,
            attention_mask=None,
            use_cache=True,
            past_key_values=self.cache,
        )
        torch.mm(hidden_states[:, -1, :], self.model.lm_head.weight.t(), out=self.logits)
        self.tok.copy_(torch.argmax(self.logits, dim=-1, keepdim=True))
        # self-advance so replays need no host-side bookkeeping
        self.pos.add_(1)
        self.cache_seqlens.add_(1)

    def _capture(self):
        device = self.device

        # warm up outside the capture stream: lazy allocations, kernel JIT and
        # triton autotuning must not happen inside the capture
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(_GRAPH_WARMUP_STEPS):
                self._step_body()
        torch.cuda.current_stream(device).wait_stream(side_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step_body()
        torch.cuda.synchronize(device)
        self.graph = graph


def _emit(
    tok: torch.Tensor,
    out_tokens: torch.Tensor,
    seq_buffer: Optional[torch.Tensor],
    seq_len: int,
    step: int,
) -> None:
    """Write `tok` as the `step`-th generated token."""
    out_tokens[:, step] = tok.squeeze(-1)
    if seq_buffer is not None:
        seq_buffer[:, seq_len + step] = tok.squeeze(-1)


def _check_stop(
    tok: torch.Tensor,
    step: int,
    seq_buffer: Optional[torch.Tensor],
    seq_len: int,
    eos_token_id: Optional[int],
    eos_check_every: int,
    stopping_criteria,
) -> bool:
    """`step` tokens have been emitted. True = stop now."""
    if eos_token_id is not None and step % eos_check_every == 0:
        # break only when every row emitted EOS (rows past their own EOS keep
        # running until then, matching the historical behaviour)
        if bool((tok == eos_token_id).all()):
            return True
    if stopping_criteria is not None:
        if bool(stopping_criteria(seq_buffer[:, : seq_len + step], None)):
            return True
    return False


def _decode_with_graph(
    model,
    engine: DecodeEngine,
    input_ids: torch.Tensor,
    out_tokens: torch.Tensor,
    next_token_logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: Optional[int],
    eos_check_every: int,
    stopping_criteria,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    max_new_tokens = out_tokens.shape[1]

    seq_buffer = None
    if stopping_criteria is not None:
        seq_buffer = torch.empty(batch_size, seq_len + max_new_tokens, dtype=torch.long, device=input_ids.device)
        seq_buffer[:, :seq_len] = input_ids

    tok = sample_token(next_token_logits, do_sample, temperature, top_k, top_p)
    step = 0
    while step < max_new_tokens:
        _emit(tok, out_tokens, seq_buffer, seq_len, step)
        step += 1
        if _check_stop(tok, step, seq_buffer, seq_len,
                       eos_token_id, eos_check_every, stopping_criteria):
            break
        engine.ids.copy_(tok)
        engine.graph.replay()
        if do_sample:
            tok = sample_token(engine.logits, do_sample, temperature, top_k, top_p)
        else:
            tok = engine.tok

    return torch.cat([input_ids, out_tokens[:, :step]], dim=-1)


def _decode_eager(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values,
    out_tokens: torch.Tensor,
    next_token_logits: torch.Tensor,
    use_cache: bool,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: Optional[int],
    eos_check_every: int,
    stopping_criteria,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    max_new_tokens = out_tokens.shape[1]
    incremental = use_cache and past_key_values is not None

    # absolute position of the first decoded token
    base_position = past_key_values.get_seq_length() if incremental else seq_len

    position_buffer = torch.empty(batch_size, 1, dtype=torch.long, device=device)

    # an all-ones mask is a no-op for the kernels but costs a nonzero + a
    # device sync per layer per step, so drop it while decoding
    decode_mask = None
    if attention_mask is not None and not bool(attention_mask.all()):
        decode_mask = attention_mask

    seq_buffer = None
    if stopping_criteria is not None:
        seq_buffer = torch.empty(batch_size, seq_len + max_new_tokens, dtype=torch.long, device=device)
        seq_buffer[:, :seq_len] = input_ids

    logits = next_token_logits
    step = 0
    while step < max_new_tokens:
        tok = sample_token(logits, do_sample, temperature, top_k, top_p)
        _emit(tok, out_tokens, seq_buffer, seq_len, step)
        step += 1
        if _check_stop(tok, step, seq_buffer, seq_len,
                       eos_token_id, eos_check_every, stopping_criteria):
            break

        if incremental:
            position_buffer.fill_(base_position + step - 1)
            if decode_mask is not None:
                decode_mask = torch.cat(
                    [decode_mask, torch.ones(batch_size, 1, device=device, dtype=decode_mask.dtype)],
                    dim=-1,
                )
            outputs = model.model(
                input_ids=tok,
                position_ids=position_buffer,
                attention_mask=decode_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
        else:
            # no cache: recompute the whole sequence each step (kept as a
            # correct fallback, e.g. when the caller disables the cache)
            full = torch.cat([input_ids, out_tokens[:, :step]], dim=-1)
            positions = torch.arange(full.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
            # the mask only covers the prompt; generated tokens are all valid
            if decode_mask is not None:
                full_mask = torch.cat(
                    [decode_mask,
                     torch.ones(batch_size, step, device=device, dtype=decode_mask.dtype)],
                    dim=-1,
                )
            else:
                full_mask = None
            outputs = model.model(
                input_ids=full,
                position_ids=positions,
                attention_mask=full_mask,
                use_cache=False,
                past_key_values=None,
            )
        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)[:, -1, :]

    return torch.cat([input_ids, out_tokens[:, :step]], dim=-1)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    use_cache: bool = True,
    past_key_values=None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = False,
    eos_token_id: int = 2,
    stopping_criteria=None,
    use_cuda_graph: Optional[bool] = None,
    eos_check_every: int = 1,
):
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    max_new_tokens = resolve_new_tokens(seq_len, max_length, max_new_tokens)
    if max_new_tokens <= 0:
        return input_ids

    criteria = normalize_stopping_criteria(stopping_criteria)

    # Prefill phase (dynamic cache path; runs once, not worth graphing)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    outputs = model.model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=use_cache,
        past_key_values=past_key_values,
    )
    hidden_states, past_key_values = outputs[0], outputs[1]
    next_token_logits = model.lm_head(hidden_states)[:, -1, :]

    out_tokens = torch.empty(batch_size, max_new_tokens, dtype=torch.long, device=device)

    if use_cuda_graph is None:
        use_cuda_graph = cuda_graph_default()
    graph_ready = (
        use_cuda_graph
        and input_ids.is_cuda
        and use_cache
        and past_key_values is not None
        and len(past_key_values) > 0
        and max_new_tokens >= _MIN_GRAPH_TOKENS
        and (attention_mask is None or bool(attention_mask.all()))
    )
    if graph_ready:
        try:
            # capacity must cover the cached prompt(s) plus the new tokens
            capacity = past_key_values.get_seq_length() + max_new_tokens
            engine = model._get_decode_engine(
                past_key_values, batch_size, capacity, device
            )
            engine.seed(past_key_values)
            return _decode_with_graph(
                model, engine, input_ids, out_tokens, next_token_logits,
                do_sample, temperature, top_k, top_p, eos_token_id,
                eos_check_every, criteria,
            )
        except Exception as exc:    # pragma: no cover - defensive fallback
            warnings.warn(
                f"CUDA-graph decoding disabled ({type(exc).__name__}: {exc}); "
                f"falling back to the eager decode loop.",
                stacklevel=2,
            )
            model._decode_engine = None
            model._decode_engine_signature = None

    return _decode_eager(
        model, input_ids, attention_mask, past_key_values, out_tokens,
        next_token_logits, use_cache, do_sample, temperature, top_k, top_p,
        eos_token_id, eos_check_every, criteria,
    )
