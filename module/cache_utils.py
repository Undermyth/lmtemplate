
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

    def update(self, recurrent_state, conv_state, layer_idx, offset):
        if len(self) <= layer_idx:
            self.cache.append({
                'recurrent_state': recurrent_state,
                'conv_state': conv_state,
            })
        else:
            self.cache[layer_idx]['recurrent_state'] = recurrent_state
            self.cache[layer_idx]['conv_state'] = conv_state
        self._seen_tokens += offset
