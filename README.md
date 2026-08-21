# Language Model Training Template

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=lightning" alt="Lightning">
</p>

A lightweight and modular training template for language model pre-training and evaluation.

---

## Project Structure

```
.
├── module/
│   ├── attention.py       # Example Attention mechanism with RoPE and flash attention
│   ├── cache_utils.py     # KV-cache utilities
│   └── modeling.py        # HuggingFace-style architecture (RotaryEmbedding, Layer, LanguageModel, ModelForCausalLM)
├── data/
│   └── stream_parquet.py  # Streaming dataset loader for parquet files
├── train.py               # Training script (Lightning-based)
├── test.py                # Evaluation script (lm_eval-based)
├── model.py               # LightningModule wrapper with optimizer configuration
├── nan_tracer.py          # NanTrap: NaN/Inf 临界点捕捉（首次 NaN 时 dump 现场并中止）
├── reproduce_nan.py       # 离线复现 NanTrap 保存的临界点，定位 NaN 产生的算子
└── train_utils.py         # Training utilities (schedulers, step calculations)
```

---

## Dependencies

- **PyTorch** >= 2.0
- **Lightning** >= 2.0
- **Transformers** >= 4.0
- **lm_eval** (for evaluation)
- **flash-attn** (for efficient attention)
- **fla** (fast linear attention)
- **einops**, **jaxtyping**, **rich**

---

## Training

```bash
python train.py \
    --tokenizer-name meta-llama/Llama-2-7b \
    --parquet-path /path/to/training/data.parquet \
    --seqlen 2048 \
    --micro-batch-size 6 \
    --ngpus 4 \
    --checkpoint-tokens 0.5 \
    --eval-per-checkpoint 2 \
    --optimizer muon \
    --lr 0.02 \
    --log-to-wandb \
    --wandb-project my-project \
    --wandb-runname my-run
```

### Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tokenizer-name` | Pretrained tokenizer name or path | `meta-llama/Llama-2-7b` |
| `--parquet-path` | Path to training parquet file | **required** |
| `--seqlen` | Training sequence length | `2048` |
| `--micro-batch-size` | Batch size per GPU | `6` |
| `--ngpus` | Number of GPUs for DDP | `4` |
| `--optimizer` | Optimizer type (`adam` or `muon`) | `muon` |
| `--lr` | Learning rate | `0.02` |
| `--log-to-wandb` | Enable W&B logging | `False` |
| `--eval-tasks` | Comma-separated eval tasks | `None` |

### Muon Optimizer

This template supports the **Muon** optimizer (Mish et al., 2024) for training. When using Muon:

- Embedding parameters receive 10x higher learning rate
- LM head parameters receive 0.1x learning rate
- 2 optimizers are used in parallel

### Checkpointing

Checkpoints are saved every `--checkpoint-tokens` billion tokens. Evaluation runs `--eval-per-checkpoint` times within each checkpoint interval.

#### Resume from a specific checkpoint

Pass the explicit checkpoint path to `--resume-checkpoint`:

```bash
python train.py ... \
    --resume-checkpoint /path/to/checkpoints/0-1000.ckpt
```

- **DDP 模式**的检查点是单个 `.ckpt` 文件，直接传文件路径；
- **DeepSpeed（`--parallel zero1/zero2`）模式**的检查点是一个目录（包含 `checkpoint/` 分片子目录），传目录路径。

#### Auto-resume to the latest checkpoint

用保留字 `latest` 让脚本自动扫描 `--checkpoint-dir`，找到其中**最新**的检查点并继续训练
（同时兼容 DDP 的单文件 `.ckpt` 与 DeepSpeed 的目录型检查点）：

```bash
python train.py ... \
    --checkpoint-dir ./checkpoints \
    --resume-checkpoint latest
```

- 目录为空或不存在时自动回退为**从头训练**（等价于不传 `--resume-checkpoint`），并打印提示；
- 按写入时间（mtime）判断新旧，目录型检查点取其内部最新分片文件的时间；
- 写入中的临时文件/目录（`.`、`tmp` 前缀）会被忽略，避免选中未写完的检查点；
- 解析出的实际路径会打印在启动日志中，并体现在配置表中。

---

## NaN 检测与临界点保存 (NanTrap)

训练中 loss / 梯度 / 权重出现 NaN 或 Inf 时，NanTrap 会在**第一次**出现非有限值的临界点
自动保存现场并中止训练，便于离线复现与定位。三个检测点（每步依次执行）：

1. **loss**  —— forward 之后、backward 之前检查 loss（含 aux_loss）
2. **grad**  —— backward 之后、grad clip / optimizer step 之前检查所有参数梯度
3. **weight**—— optimizer step 之后检查权重（兜底：捕捉"有限梯度被优化器污染成 NaN"的 bug）

DDP 下任一 rank 非有限会先跨 rank all-reduce，所有 rank 一起 dump 并中止，避免单 rank 提前退出卡死。
不传 `--nan-dump-dir` 时所有检测均为 no-op，训练零开销、零行为变化。

### 启用

```bash
python train.py ... \
    --nan-dump-dir ./nan_dump \        # 保存现场并中止
    --nan-dump-optimizer               # 可选：一并保存优化器状态（体积可能数 GB）
```

触发后现场保存在 `<nan_dump_dir>/rank{r}_step{s}_{ts}/` 下：

| 文件 | 内容 |
|------|------|
| `model.pt`   | 触发时（尚未被 step 污染）的模型权重 |
| `samples.pt` | 当前梯度累积窗口内的全部 micro-batch 样本 |
| `rng.pt`     | CPU / CUDA RNG 状态（dropout 等随机性复现用） |
| `meta.json`  | 触发类型、step、rank、grad_accum、micro_batch 位置、模型 config、git 版本等 |
| `optimizer.pt` | （可选）优化器状态 |

### 复现与定位

```bash
python reproduce_nan.py <dump_dir> [--anomaly] [--probe-samples 10]
```

脚本会原样重放窗口内样本的 forward / backward：

- forward 出现非有限 loss → 用 forward hooks 定位**第一个**出现 inf/NaN 的算子，并打印各层激活的 `max|.|` 轨迹；
- forward 有限但 backward 非有限 → 列出非有限梯度的参数，用 backward hooks 定位 NaN 产生层，`--anomaly` 可让 torch 直接报出 NaN 算子；
- `--probe-samples N` 用随机样本探测，区分"该样本触发"还是"权重已病态（任意样本都炸）"。

> 注意：若 dump 的权重本身已含 NaN（`meta.json` 中 `params_all_finite_at_dump=false`），说明属于
> optimizer 污染场景或检测点过晚，无法复现"首次" NaN。

---

## Evaluation

```bash
python test.py \
    --tokenizer-name meta-llama/Llama-2-7b \
    --checkpoint-path /path/to/checkpoint.pt \
    --tasks hellaswag,winogrande,piqa
```

### Key Evaluation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tokenizer-name` | Pretrained tokenizer name or path | `meta-llama/Llama-2-7b` |
| `--checkpoint-path` | Path to model checkpoint | **required** |
| `--tasks` | Comma-separated list of lm_eval tasks | `hellaswag,winogrande,piqa` |

---

## ModelConfig Defaults

```python
vocab_size: int = 32000
n_layers: int = 16
dim: int = 1024
n_heads: int = 8
expand_ratio: int = 4
rotary_dim: int = 128
rotary_base: int = 10000
```