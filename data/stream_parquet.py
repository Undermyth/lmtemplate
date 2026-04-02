import os
from collections import deque
from typing import Dict, Optional
import pyarrow.parquet as pq

import torch
from torch.utils.data import IterableDataset

def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def list_parquet_files(data_dir):
    """ Looks into a data dir and returns full paths to all parquet files. """
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


class StreamingParquet(IterableDataset):
    
    def __init__(
        self,
        parquet_dir: str,
        batch_size: int,
        seq_len: int,
        tokenizer,
        ddp_rank: int,
        world_size: int,
        device: str = 'cuda',
        tokenizer_batch_size: int = 128,    # just for tokenizer efficiency, will not influence training batch size
        tokenizer_threads: int = 8,
        split: str = 'train',
        state_dict: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        self.parquet_dir = parquet_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.device = device
        self.split = split
        self.tokenizer_batch_size = tokenizer_batch_size
        self.tokenizer_threads = tokenizer_threads
        self.pq_idx = state_dict['pq_idx'] if state_dict is not None else 0
        self.rg_idx = state_dict['rg_idx'] if state_dict is not None else None
        if state_dict is not None:
            print(f'[streaming parquet] start loader at pq_idx = {self.pq_idx}, rg_idx = {self.rg_idx}')

        self.ddp_rank = ddp_rank
        self.world_size = world_size
        self.inf_batch_document_iterator = self.document_batches()

    def load_state_dict(self, state_dict: Dict[str, int]):
        self.pq_idx = state_dict['pq_idx']
        self.rg_idx = state_dict['rg_idx']
        print(f'[streaming parquet] start loader at pq_idx = {self.pq_idx}, rg_idx = {self.rg_idx}')
                
    def document_batches(self):
        # ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        parquet_paths = list_parquet_files(self.parquet_dir)
        parquet_paths = parquet_paths[:-1] if self.split == "train" else parquet_paths[-1:]
        resume_pq_idx = self.pq_idx
        resume_rg_idx = self.rg_idx
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // self.world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * self.world_size + self.ddp_rank
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = self.ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), self.tokenizer_batch_size):
                        # print(pq_idx, rg_idx, i, len(batch))
                        yield batch[i:i+self.tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += self.world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
                

    def __iter__(self):
        # Now emit batches of tokens.
        needed_tokens = self.batch_size * self.seq_len + 1 # +1 is because we also need the target at the last token
        # get the tokenizer and the bos token
        # bos_token = self.tokenizer.get_bos_token_id()
        bos_token = self.tokenizer.bos_token
        # scratch buffer holds the tokens for one iteration
        token_buffer = deque() # we stream tokens on the right and pop from the left
        while True:
            # Accumulate enough tokens for one iteration before yielding.
            while len(token_buffer) < needed_tokens:
                doc_batch, (pq_idx, rg_idx) = next(self.inf_batch_document_iterator)
                # token_lists = self.tokenizer.encode(doc_batch, prepend=bos_token, num_threads=self.tokenizer_threads)
                token_lists = self.tokenizer(doc_batch, add_special_tokens=True).input_ids
                for tokens in token_lists:
                    token_buffer.extend(tokens)
            # Move tokens from the deque into the scratch buffer
            tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
            # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
            scratch = torch.tensor(tokens, dtype=torch.long) # in PyTorch, long=int64
            # Create the inputs/targets as 1D tensors
            inputs_cpu = scratch[:-1]
            targets_cpu = scratch[1:]
            # Reshape to 2D and move to GPU async
            inputs = inputs_cpu.view(self.batch_size, self.seq_len)
            targets = targets_cpu.view(self.batch_size, self.seq_len)
            state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
            # print(self.ddp_rank, state_dict)
            yield inputs, targets, state_dict

if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b', legacy=False)
    dataset = StreamingParquet(
        parquet_dir='/data2/csy/transformers/hub/datasets--skymizer--fineweb-edu-dedup-45B/parquets',
        batch_size=4,
        seq_len=2048,
        tokenizer=tokenizer
    )
    loader = DataLoader(dataset)
    iterator = enumerate(loader)
    x, y, state_dict = next(iterator)[1]
    print(x.shape, y.shape, state_dict)
