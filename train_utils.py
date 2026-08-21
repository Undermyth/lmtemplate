import os
from typing import List, Optional

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# --resume-checkpoint 保留字：自动从 checkpoint 目录中寻找最新的检查点并续训。
# 目录为空或不存在时回退为从头训练。
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

def is_deepspeed_checkpoint_dir(path: str) -> bool:
    """判断 ``path`` 是否为 DeepSpeed 检查点目录（目录而非单文件）。

    DeepSpeed 检查点目录的布局与 Lightning 的 ``DeepSpeedStrategy`` / deepspeed
    的保存逻辑一致（``deepspeed_engine.save_checkpoint(dir, tag="checkpoint")``）：

    ::

        <name>/
        ├── checkpoint/                  # tag 子目录
        │   ├── mp_rank_00_model_states.pt
        │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
        │   └── ...
        ├── latest
        └── zero_to_fp32.py

    判定规则与 Lightning 自身的 ``_is_deepspeed_checkpoint`` 保持一致：
    是目录且包含 ``checkpoint/`` 子目录即视为 DeepSpeed 检查点。
    """
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, "checkpoint"))


def is_checkpoint_entry(path: str) -> bool:
    """``path`` 是否是一个可续训的检查点：

    - 朴素 DDP 检查点：单个 ``.ckpt`` 文件；
    - DeepSpeed 检查点：目录（其内容为一组分片文件而非单个文件）。
    """
    if os.path.isfile(path):
        return path.endswith(".ckpt")
    return is_deepspeed_checkpoint_dir(path)


def _entry_mtime(path: str) -> float:
    """检查点条目的最近修改时间。

    对目录（DeepSpeed 检查点）取目录本身与其直接子项 mtime 的最大值：
    分片文件是逐个写入目录的，目录自身的 mtime 不一定晚于最后写入的子文件。
    """
    if os.path.isdir(path):
        try:
            children = [os.path.getmtime(os.path.join(path, n)) for n in os.listdir(path)]
        except OSError:
            children = []
        return max([os.path.getmtime(path)] + children)
    return os.path.getmtime(path)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """扫描 ``checkpoint_dir``，返回其中最新的检查点路径。

    同时支持朴素 ``.ckpt`` 单文件（DDP 模式）与 DeepSpeed 检查点目录。
    隐藏条目与临时文件（``.`` / ``tmp`` 前缀，如写入中的 ``.tmp.ckpt`` 或
    尚未落位的临时目录）会被忽略，避免选中未写完的检查点。

    目录不存在、为空或不含任何可识别检查点时返回 ``None``。
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
    # 以 mtime 排序，最新者优先；mtime 相同（如同步复制）时按名称字典序兜底。
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][1]


def _rank_zero_print(msg: str) -> None:
    """训练启动前（Trainer 尚未初始化）无法使用 lightning 的 rank_zero_only，
    直接用 ``RANK`` 环境变量（torchrun 设置）做 rank-0 门控。"""
    if os.environ.get("RANK", "0") == "0":
        print(msg, flush=True)


def resolve_resume_checkpoint(
    resume_checkpoint: Optional[str],
    checkpoint_dir: str,
) -> Optional[str]:
    """解析 ``--resume-checkpoint`` 参数的实际值。

    - ``None``                    -> ``None``（从头训练）
    - ``RESUME_LATEST``（"latest"）-> ``checkpoint_dir`` 中最新检查点的路径；
                                    目录为空/不存在时返回 ``None``（从头训练）
    - 其他任意值                   -> 原样返回（显式指定路径）
    """
    if resume_checkpoint is None:
        return None
    if resume_checkpoint != RESUME_LATEST:
        return resume_checkpoint

    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is None:
        _rank_zero_print(
            f"[resume] `{RESUME_LATEST}` 已指定，但 `{checkpoint_dir}` 中没有可用检查点"
            "（目录不存在或为空），将从零开始训练。"
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