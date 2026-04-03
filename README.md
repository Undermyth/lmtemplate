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