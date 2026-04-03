from typing import Optional

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-6,
    initial_lr: Optional[float] = None,
    last_epoch: int = -1
):
    """
    创建一个 '线性预热 + 余弦退火' 的组合学习率调度器（无重启）。
    
    参数:
        optimizer (torch.optim.Optimizer): 优化器实例
        warmup_epochs (int): 预热阶段的 epoch 数量
        total_epochs (int): 总训练轮数（余弦周期长度）
        eta_min (float): 余弦退火的最小学习率，默认 1e-6
        initial_lr (float, optional): 初始学习率。若为 None，则自动从 optimizer 获取
        last_epoch (int): 上一个 epoch 的索引，用于恢复训练，默认 -1
    
    返回:
        torch.optim.lr_scheduler.ChainedScheduler: 按顺序应用的调度器链
    """
    
    if initial_lr is None:
        initial_lr = optimizer.param_groups[0]['lr']
    
    # 第一阶段：线性预热（从 0.1 * initial_lr 线性增长到 initial_lr）
    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.1,           # 起始为 10% 的初始学习率
        end_factor=1.0,             # 结束为 100%
        total_iters=warmup_epochs,  # 预热总步数
        last_epoch=last_epoch       # 支持断点恢复
    )
    
    # 第二阶段：余弦退火（从 pre-warmup 结束后的 lr 开始，衰减到 eta_min）
    # 注意：余弦周期长度 = total_epochs - warmup_epochs
    cosine_duration = max(1, total_epochs - warmup_epochs)  # 至少为1，避免除零
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_duration,      # 余弦周期长度
        eta_min=eta_min,
        last_epoch=last_epoch - warmup_epochs if last_epoch >= warmup_epochs else -1
    )
    
    # 使用 ChainedScheduler 按顺序串联两个调度器
    # 注意：ChainedScheduler 会依次调用每个 scheduler.step()
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