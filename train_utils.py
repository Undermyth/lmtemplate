import os
from typing import List, Optional

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# --resume-checkpoint sentinel: auto-resume from the newest checkpoint in the checkpoint dir.
# Falls back to training from scratch if the dir is empty or missing.
RESUME_LATEST = "latest"


def create_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-6,
    initial_lr: Optional[float] = None,
    last_epoch: int = -1
):
    """
    Create a combined 'linear warmup + cosine annealing' LR scheduler (no restarts).

    Args:
        optimizer (torch.optim.Optimizer): the optimizer instance
        warmup_epochs (int): number of warmup epochs
        total_epochs (int): total training epochs (cosine cycle length)
        eta_min (float): minimum LR for cosine annealing, default 1e-6
        initial_lr (float, optional): initial LR. If None, taken from the optimizer
        last_epoch (int): index of the last epoch, for resuming training, default -1

    Returns:
        torch.optim.lr_scheduler.ChainedScheduler: the scheduler chain applied in order
    """
    
    if initial_lr is None:
        initial_lr = optimizer.param_groups[0]['lr']
    
    # Phase 1: linear warmup (from 0.1 * initial_lr up to initial_lr)
    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.1,           # starts at 10% of the initial LR
        end_factor=1.0,             # ends at 100%
        total_iters=warmup_epochs,  # total warmup steps
        last_epoch=last_epoch       # supports resume
    )
    
    # Phase 2: cosine annealing (from the post-warmup LR down to eta_min)
    # note: cosine cycle length = total_epochs - warmup_epochs
    cosine_duration = max(1, total_epochs - warmup_epochs)  # at least 1, to avoid division by zero
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_duration,      # cosine cycle length
        eta_min=eta_min,
        last_epoch=last_epoch - warmup_epochs if last_epoch >= warmup_epochs else -1
    )
    
    # chain the two schedulers in order with SequentialLR
    # note: SequentialLR calls each scheduler.step() in turn
    scheduler = SequentialLR(
        optimizer=optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
        last_epoch=last_epoch
    )
    
    return scheduler

def get_checkpoint_steps(
    n_checkpoint_tokens: int,    # save a checkpoint every n billion tokens
    n_optimizers: int,           # number of optimizers used in the model
    n_grad_accum_steps: int,     # steps of gradient accumulation
    n_gpus: int,
    micro_batch_size: int,       # batch size per gpu
    seq_len: int                 # training sequence length
) -> int:

    '''
    The parameter `every_n_train_steps` in Lightning is counted as how many times the optimizer.step() is called.
    Given that multiple optimizers may be used, the parameter is calculated as

    every_n_train_steps / n_optimizers * n_grad_accum_steps * n_gpus * micro_batch_size * seq_len = n_tokens
    └────────────────────────────────┘
            real optimize steps
    └─────────────────────────────────────────────────────┘
                           batch steps
                                                              └─────────────────────────────────┘
                                                                        tokens per batch
    '''
    n_tokens = n_checkpoint_tokens * 1024 * 1024 * 1024
    tokens_per_batch = n_gpus * micro_batch_size * seq_len
    every_n_train_steps = int(n_tokens / tokens_per_batch / n_grad_accum_steps) * n_optimizers
    return every_n_train_steps

def get_optimizer_steps(
    n_optimize_tokens: int,     # token count within the planned scheduler epoch (in billion)
    n_grad_accum_steps: int,     # steps of gradient accumulation
    n_gpus: int,
    micro_batch_size: int,       # batch size per gpu
    seq_len: int                 # training sequence length
) -> int:

    '''
    optimizer_steps * n_grad_accum_steps * n_gpus * micro_batch_size * seq_len = n_tokens
    └──────────────────────────────────┘   └─────────────────────────────────┘
                 batch steps                         tokens per batch                                              
    '''
    n_tokens = n_optimize_tokens * 1024 * 1024 * 1024
    tokens_per_batch = n_gpus * micro_batch_size * seq_len
    optimizer_steps = n_tokens / tokens_per_batch / n_grad_accum_steps
    return int(optimizer_steps)

def is_deepspeed_checkpoint_dir(path: str) -> bool:
    """Whether ``path`` is a DeepSpeed checkpoint directory (a directory, not a single file).

    The layout matches Lightning's ``DeepSpeedStrategy`` / deepspeed's saving logic
    (``deepspeed_engine.save_checkpoint(dir, tag="checkpoint")``):

    ::

        <name>/
        ├── checkpoint/                  # tag subdirectory
        │   ├── mp_rank_00_model_states.pt
        │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
        │   └── ...
        ├── latest
        └── zero_to_fp32.py

    The rule matches Lightning's own ``_is_deepspeed_checkpoint``:
    a directory containing a ``checkpoint/`` subdirectory is a DeepSpeed checkpoint.
    """
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, "checkpoint"))


def is_checkpoint_entry(path: str) -> bool:
    """Whether ``path`` is a resumable checkpoint:

    - plain DDP checkpoint: a single ``.ckpt`` file;
    - DeepSpeed checkpoint: a directory (a set of sharded files rather than a single file).
    """
    if os.path.isfile(path):
        return path.endswith(".ckpt")
    return is_deepspeed_checkpoint_dir(path)


def _entry_mtime(path: str) -> float:
    """Most recent modification time of a checkpoint entry.

    For a directory (DeepSpeed checkpoint) take the max mtime of the dir itself and its direct
    children: shard files are written into the dir one by one, so the dir's own mtime may be
    older than the last written file.
    """
    if os.path.isdir(path):
        try:
            children = [os.path.getmtime(os.path.join(path, n)) for n in os.listdir(path)]
        except OSError:
            children = []
        return max([os.path.getmtime(path)] + children)
    return os.path.getmtime(path)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Scan ``checkpoint_dir`` and return the path of the newest checkpoint.

    Supports both plain ``.ckpt`` single files (DDP mode) and DeepSpeed checkpoint directories.
    Hidden entries and temp files (``.`` / ``tmp`` prefixes, e.g. an in-progress ``.tmp.ckpt`` or
    a temp dir not yet in place) are ignored to avoid picking an unwritten checkpoint.

    Returns ``None`` when the dir is missing, empty, or has no recognizable checkpoint.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    candidates: List[tuple] = []
    for name in sorted(os.listdir(checkpoint_dir)):
        if name.startswith(".") or name.startswith("tmp"):
            continue
        path = os.path.join(checkpoint_dir, name)
        if is_checkpoint_entry(path):
            candidates.append((_entry_mtime(path), path))

    if not candidates:
        return None
    # sort by mtime, newest first; tie-break by lexicographic name (e.g. synced copies).
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][1]


def _rank_zero_print(msg: str) -> None:
    """Before training starts (Trainer not yet initialized) lightning's rank_zero_only is
    unavailable, so gate on the ``RANK`` env var (set by torchrun) directly."""
    if os.environ.get("RANK", "0") == "0":
        print(msg, flush=True)


def resolve_resume_checkpoint(
    resume_checkpoint: Optional[str],
    checkpoint_dir: str,
) -> Optional[str]:
    """Resolve the actual value of the ``--resume-checkpoint`` argument.

    - ``None``                    -> ``None`` (train from scratch)
    - ``RESUME_LATEST`` ("latest")-> path of the newest checkpoint in ``checkpoint_dir``;
                                    ``None`` if the dir is empty/missing (train from scratch)
    - any other value             -> returned as-is (explicit path)
    """
    if resume_checkpoint is None:
        return None
    if resume_checkpoint != RESUME_LATEST:
        return resume_checkpoint

    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is None:
        _rank_zero_print(
            f"[resume] `{RESUME_LATEST}` specified but no usable checkpoint found in `{checkpoint_dir}`"
            " (dir missing or empty); training from scratch."
        )
    else:
        _rank_zero_print(f"[resume] `{RESUME_LATEST}` -> {latest}")
    return latest


def get_eval_steps(
    every_n_train_steps: int,    # return value of `get_checkpoint_steps`
    n_eval_per_checkpoint: int,  # run evaluation for n times per checkpoint epoch 
    n_optimizers: int,           # number of optimizers used in the model
    n_grad_accum_steps: int,     # steps of gradient accumulation
    n_gpus: int,
    micro_batch_size: int,       # batch size per gpu
    seq_len: int                 # training sequence length
) -> int:
    '''
    by specifying `n_eval_per_checkpoint`, we aim to run evaluation every n_token / n_eval_per_checkpoint, 
    where n_token refer to `n_checkpoint_tokens`.
    `val_check_interval` in Lightning is counted in batch, so we have

    val_check_interval * n_gpus * micro_batch_size * seq_len = n_tokens / n_eval_per_checkpoint
    └────────────────┘   └─────────────────────────────────┘
        batch steps                tokens per batch

    combined with calculation in `get_checkpoint_steps`, we have

    every_n_train_steps / n_optimizers * n_grad_accum_steps = val_check_interval * n_eval_per_checkpoint
    └─────────────────────────────────────────────────────┘
                          batch steps

    LHS is sure to be integer, so we only need to assert that batch_steps % n_eval_per_checkpoint == 0
    '''
    batch_steps = every_n_train_steps / n_optimizers * n_grad_accum_steps
    assert batch_steps % n_eval_per_checkpoint == 0, f'batch steps ({batch_steps}) can not be divided by n_eval_per_checkpoint ({n_eval_per_checkpoint})'
    return int(batch_steps / n_eval_per_checkpoint)