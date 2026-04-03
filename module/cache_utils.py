from typing import Optional

import torch
from jaxtyping import Float


class Cache:
    def __init__(self):
        self._seen_tokens = 0
        self.cache = []

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, layer_idx):
        return self.cache[layer_idx]
    
    def get_sequence_length(self):
        return self._seen_tokens

    def update(
        self, 
        recurrent_state: Optional[Float[torch.Tensor, 'b h d D']] = None,    # for recurrent models 
        key_state: Optional[Float[torch.Tensor, 'b h t d']] = None,          # for transformer models
        value_state: Optional[Float[torch.Tensor, 'b h t d']] = None,
        conv_state = None,                                                   # common state
        layer_idx: int = -1,
        offset: int = 0,
    ):
        if recurrent_state is not None:
            if len(self) <= layer_idx:
                self.cache.append({
                    'recurrent_state': recurrent_state,
                })
            else:
                self.cache[layer_idx]['recurrent_state'] = recurrent_state
        
        elif key_state is not None:
            assert value_state is not None
            if len(self) <= layer_idx:
                self.cache.append({
                    'key_cache': key_state,
                    'value_cache': value_state
                })
            else:
                key_cache = self.cache[layer_idx]['key_cache']
                value_cache = self.cache[layer_idx]['value_cache']
                key_cache = torch.cat([key_cache, key_state], dim=-2)
                value_cache = torch.cat([value_cache, value_state], dim=-2)
                self.cache[layer_idx]['key_cache'] = key_cache
                self.cache[layer_idx]['value_cache'] = value_cache

        if conv_state is not None:
            self.cache[layer_idx]['conv_state'] = conv_state

        if layer_idx == 0:
            self._seen_tokens += offset
