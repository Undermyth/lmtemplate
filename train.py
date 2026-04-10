import argparse
import os

import lightning as L
import lightning.pytorch.callbacks as cbs
import torch
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.table import Table
from torchinfo import summary
from transformers import AutoTokenizer

from model import LanguageModel, OptimizeConfig
from module.modeling import ModelConfig, ModelForCausalLM
from train_utils import get_checkpoint_steps, get_eval_steps

torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-2-7b")
parser.add_argument("--parquet-path", type=str, required=True)
parser.add_argument("--log-to-wandb", action="store_true")
parser.add_argument("--wandb-project", type=str, default=None)
parser.add_argument("--wandb-runname", type=str, default=None)
parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
parser.add_argument("--resume-checkpoint", type=str, default=None)
parser.add_argument("--seqlen", type=int, default=2048)
parser.add_argument("--ngpus", type=int, default=4)
parser.add_argument("--micro-batch-size", type=int, default=6)  # batch size per gpu
parser.add_argument("--grad-accum-steps", type=int, default=1)
parser.add_argument("--checkpoint-tokens", type=float, default=0.5)   # save a checkpoint every n billion tokens
parser.add_argument("--eval-per-checkpoint", type=int, default=2)   # evaluate n times in one checkpoint duration
parser.add_argument("--val-sanity", action="store_true")
parser.add_argument("--eval-tasks", type=str, default=None, help="Comma-separated list of eval tasks, e.g., arc_challenge,arc_easy,lambada_openai")

# optimize settings
parser.add_argument("--optimizer", type=str, default="muon", choices=["adam", "muon"])
parser.add_argument("--optimize-tokens", type=float, default=20)      # scheduler epoch for n billion tokens
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--min-lr-frac", type=float, default=0.001)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--grad-clip", type=float, default=None)
parser.add_argument("--warmup-ratio", type=float, default=0.005)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, legacy=False)

config = ModelConfig()
model = ModelForCausalLM(config).to(torch.bfloat16)

@rank_zero_only
def log_config_and_model():
    console = Console()
    table = Table(title="Language Model Training Configuration")
    table.add_column("Argument", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for arg in vars(args):
        val = getattr(args, arg)
        if arg == "eval_tasks" and val is not None:
            val = val.split(",")
        table.add_row(arg, str(val))

    console.print(table)
    console.print()
    summary(model)

@rank_zero_only
def wandb_log_code(wandb_run):
    wandb_run.log_code("./")

log_config_and_model()

optimize_config = OptimizeConfig(
    peak_lr=args.lr,
    warmup_ratio=args.warmup_ratio,
    min_lr_frac=args.min_lr_frac,
    weight_decay=args.weight_decay,
    grad_clip=args.grad_clip
)

model = LanguageModel(
    model=model,
    tokenizer=tokenizer,
    parquet_path=args.parquet_path,
    seq_len=args.seqlen,
    batch_size=args.micro_batch_size,
    grad_accum_steps=args.grad_accum_steps,
    n_gpus=args.ngpus,
    optimize_tokens=args.optimize_tokens,
    eval_tasks=args.eval_tasks.split(","),
    optimizer=args.optimizer,
    optimize_config=optimize_config
)

every_n_train_steps = get_checkpoint_steps(
    n_checkpoint_tokens=args.checkpoint_tokens,
    n_optimizers=2 if args.optimizer == "muon" else 1,
    n_grad_accum_steps=args.grad_accum_steps,
    n_gpus=args.ngpus,
    micro_batch_size=args.micro_batch_size,
    seq_len=args.seqlen
)

checkpoint_callback = cbs.ModelCheckpoint(
    dirpath=args.checkpoint_dir,
    # save_last='link',
    save_on_exception=False,
    every_n_train_steps=every_n_train_steps,
    save_top_k=-1,
)

if args.log_to_wandb:
    assert args.wandb_project is not None and args.wandb_runname is not None, "project name and run name not specified"
    logger = WandbLogger(project=args.wandb_project, name=args.wandb_runname)
    wandb_run = logger.experiment
    wandb_log_code(wandb_run)
else:
    logger = CSVLogger(save_dir='./logs/', flush_logs_every_n_steps=50)

prog_callback = cbs.TQDMProgressBar()

lr_monitor = cbs.LearningRateMonitor(logging_interval="step")

eval_steps = get_eval_steps(
    every_n_train_steps=every_n_train_steps,
    n_eval_per_checkpoint=args.eval_per_checkpoint,
    n_optimizers=2 if args.optimizer == "muon" else 1,
    n_grad_accum_steps=args.grad_accum_steps,
    n_gpus=args.ngpus,
    micro_batch_size=args.micro_batch_size,
    seq_len=args.seqlen
)

trainer = L.Trainer(
    max_epochs=1,
    accelerator="gpu",
    devices=args.ngpus,
    strategy="ddp",
    precision="bf16-mixed",
    callbacks=[checkpoint_callback, prog_callback, lr_monitor],
    logger=logger,
    log_every_n_steps=20, 
    val_check_interval=eval_steps,
    num_sanity_val_steps=-1 if args.val_sanity else 0,
    # accumulate_grad_batches=8,
    enable_model_summary=True,
)

trainer.fit(
    model=model,
    ckpt_path=args.resume_checkpoint
)
