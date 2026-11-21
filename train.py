import argparse
import datetime
import os
import sys
from dataclasses import asdict

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
from model import LanguageModel
from module.modeling import ModelForCausalLM, load_model_config
from nan_tracer import NanTrapTriggered
from optim_utils import load_optimizer_config
from train_utils import get_checkpoint_steps, get_eval_steps, resolve_resume_checkpoint

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
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--nan-dump-dir", type=str, default=None,
                    help="Directory to dump the NaN trigger point. When set, NanTrap captures the first "
                         "non-finite loss/gradient/weight, saves (model weights + sample window + RNG + metadata) "
                         "and aborts training so reproduce_nan.py can reproduce it.")
parser.add_argument("--nan-dump-optimizer", action="store_true",
                    help="Also dump optimizer state on NaN (can be several GB; usually not needed)")

# parallelism settings & infra optimization options
parser.add_argument("--parallel", type=str, default="ddp", choices=["ddp", "zero1", "zero2"])
parser.add_argument("--gradient-checkpoint", action="store_true")
parser.add_argument("--offload-optimizer", action="store_true")

# model scale spec: which size the network is built with (see model_0.4b.yaml /
# model_1.3b.yaml). Resolved relative to cwd, then to the repo root.
parser.add_argument("--model-config", type=str, default="model_0.4b.yaml",
                    help="YAML spec file for the model scale (layers, width, heads, ...)")
# optimize settings
parser.add_argument("--optimizer-config", type=str, default="optimizer.yaml",
                    help="YAML config file for optimizer hyperparams and param groups")
parser.add_argument("--optimizer", type=str, default="muon", choices=["adam", "muon"])
parser.add_argument("--optimize-tokens", type=float, default=40)      # scheduler epoch for n billion tokens
# The following args override corresponding values in the YAML config.
# Leave as None to use the YAML value.
parser.add_argument("--lr", type=float, default=None, help="Override peak_lr in YAML")
parser.add_argument("--min-lr-frac", type=float, default=None, help="Override min_lr_frac in YAML")
parser.add_argument("--weight-decay", type=float, default=None, help="Override weight_decay in YAML")
parser.add_argument("--grad-clip", type=float, default=None, help="Override grad_clip in YAML")
parser.add_argument("--warmup-ratio", type=float, default=None, help="Override warmup_ratio in YAML")
args = parser.parse_args()

# --resume-checkpoint latest: auto-resume from the newest checkpoint in --checkpoint-dir,
# falling back to training from scratch if the dir is empty or missing.
# The resolved path is shown in the config table below.
args.resume_checkpoint = resolve_resume_checkpoint(args.resume_checkpoint, args.checkpoint_dir)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, legacy=False)

config = load_model_config(args.model_config)
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

    model_table = Table(title="Model Configuration", title_style="bold cyan")
    model_table.add_column("Field", style="cyan", no_wrap=True)
    model_table.add_column("Value", style="magenta")
    for key, val in asdict(config).items():
        model_table.add_row(key, str(val))
    console.print(model_table)
    console.print()
    summary(model.model)

@rank_zero_only
def wandb_log_code(wandb_run):
    wandb_run.log_code("./")


def describe_group_match(spec) -> str:
    """Human-readable description of a param group's matching rule."""
    if spec.source is not None:
        return f'source: "{spec.source}"'
    lines = []
    for s in spec.sources:
        parts = [f"attr={s.attr!r}"]
        if s.dim:
            parts.append(f"dim={s.dim!r}")
        if s.name:
            parts.append(f"name~{s.name!r}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


@rank_zero_only
def log_optimizer_config(optimize_config, wandb_run=None):
    """Pretty-print the loaded optimizer config as rich tables and, when a
    wandb run is active, capture it in ``run.config`` for the dashboard."""
    console = Console()

    # --- Global hyperparameters ---
    table = Table(title="Optimizer Configuration (Global)", title_style="bold cyan")
    table.add_column("Hyperparameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    rows = [
        ("optimizer_type", args.optimizer),
        ("peak_lr", optimize_config.peak_lr),
        ("warmup_ratio", optimize_config.warmup_ratio),
        ("min_lr_frac", optimize_config.min_lr_frac),
        ("weight_decay", optimize_config.weight_decay),
        ("grad_clip", optimize_config.grad_clip),
    ]
    for key, val in rows:
        display = "None (disabled)" if val is None else str(val)
        table.add_row(key, display)
    console.print(table)
    console.print()

    # --- Parameter groups ---
    groups = Table(title="Optimizer Parameter Groups", title_style="bold cyan")
    groups.add_column("Group", style="cyan", no_wrap=True)
    groups.add_column("Match Rule", style="green")
    groups.add_column("lr_mult", justify="right")
    groups.add_column("Effective LR", justify="right")
    groups.add_column("weight_decay", justify="right")
    groups.add_column("betas")
    groups.add_column("use_muon")
    groups.add_column("muon_momentum", justify="right")

    for g in optimize_config.param_groups:
        eff_lr = optimize_config.peak_lr * g.lr_mult
        wd = g.weight_decay if g.weight_decay is not None else optimize_config.weight_decay
        if g.use_muon:
            betas = "—"
            momentum = f"{g.muon_momentum:g}"
            use_muon = "yes"
        else:
            betas = f"({g.betas[0]:g}, {g.betas[1]:g})"
            momentum = "—"
            use_muon = "no"
        groups.add_row(
            g.name,
            describe_group_match(g),
            f"{g.lr_mult:g}",
            f"{eff_lr:.3e}",
            str(wd),
            betas,
            use_muon,
            momentum,
        )
    console.print(groups)
    console.print()

    # --- wandb capture ---
    if wandb_run is not None:
        wandb_run.config.update(
            {"optimizer": {"type": args.optimizer, **optimize_config.to_dict()}},
            allow_val_change=True,
        )


optimize_config = load_optimizer_config(args.optimizer_config)

# CLI overrides for YAML values
if args.lr is not None:
    optimize_config.peak_lr = args.lr
if args.min_lr_frac is not None:
    optimize_config.min_lr_frac = args.min_lr_frac
if args.weight_decay is not None:
    optimize_config.weight_decay = args.weight_decay
if args.grad_clip is not None:
    optimize_config.grad_clip = args.grad_clip
if args.warmup_ratio is not None:
    optimize_config.warmup_ratio = args.warmup_ratio

if args.parallel != "ddp" and args.optimizer == "muon" and optimize_config.grad_clip is not None:
    raise RuntimeError("ZeRO with Muon does not support grad_clip currently")

if args.parallel != "ddp" and args.nan_dump_dir:
    print(
        "[warn] --nan-dump-dir with ZeRO: Lightning's deepspeed strategy silently skips "
        "NanTrap's gradient-checkpoint hooks (params/grads are partitioned across ranks); "
        "only the loss/weight traps are reliable, so a missing grad dump does not mean the "
        "gradients were NaN-free.",
        flush=True,
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
    optimizer=args.optimizer,
    optimize_config=optimize_config,
    use_gradient_checkpoint=args.gradient_checkpoint,
    use_parallel=True,
    use_advanced_parallel=(args.parallel != "ddp"),
    nan_dump_dir=args.nan_dump_dir,
    nan_dump_optimizer=args.nan_dump_optimizer,
    eval_tasks=args.eval_tasks.split(",") if args.eval_tasks else [],
)

if args.dry_run:
    log_config_and_model()
    log_optimizer_config(optimize_config)
    print(model)
    exit()

every_n_train_steps = get_checkpoint_steps(
    n_checkpoint_tokens=args.checkpoint_tokens,
    n_optimizers=1,
    n_grad_accum_steps=args.grad_accum_steps,
    n_gpus=args.ngpus,
    micro_batch_size=args.micro_batch_size,
    seq_len=args.seqlen
)

save_format = '{epoch}-{step}'
if args.wandb_runname is not None:
    save_format = args.wandb_runname + '-{step}'

checkpoint_callback = cbs.ModelCheckpoint(
    dirpath=args.checkpoint_dir,
    # save_last='link',
    save_on_exception=False,
    every_n_train_steps=every_n_train_steps,
    save_top_k=-1,
    filename=save_format
)

wandb_run = None
if args.log_to_wandb:
    assert args.wandb_project is not None and args.wandb_runname is not None, "project name and run name not specified"
    logger = WandbLogger(project=args.wandb_project, name=args.wandb_runname)
    wandb_run = logger.experiment
    wandb_log_code(wandb_run)
    # capture the model scale spec on the dashboard, alongside the optimizer config
    wandb_run.config.update(
        {"model": {**asdict(config), "config_file": args.model_config}},
        allow_val_change=True,
    )
else:
    logger = CSVLogger(save_dir='./logs/', flush_logs_every_n_steps=50)

log_config_and_model()
log_optimizer_config(optimize_config, wandb_run=wandb_run)

prog_callback = cbs.TQDMProgressBar()

lr_monitor = cbs.LearningRateMonitor(logging_interval="step")

eval_steps = get_eval_steps(
    every_n_train_steps=every_n_train_steps,
    n_eval_per_checkpoint=args.eval_per_checkpoint,
    n_optimizers=1,
    n_grad_accum_steps=args.grad_accum_steps,
    n_gpus=args.ngpus,
    micro_batch_size=args.micro_batch_size,
    seq_len=args.seqlen
) if args.eval_tasks else None

eval_trainer_kwargs = (
    {"val_check_interval": eval_steps}
    if eval_steps is not None else
    # no mid-training eval requested (--eval-tasks not given): disable the validation
    # loop entirely instead of failing on `None.split(",")` or running an empty eval
    {"limit_val_batches": 0}
)

def ensure_deepspeed_zero_muon(parallel: str):
    """Fail early with a readable message when ZeRO mode cannot get deepspeed's Muon support.

    The `deepspeed.runtime.zero.muon` module exists since deepspeed 0.20.0; older (or missing)
    installs make every ZeRO run crash deep inside optimizer construction.
    """
    try:
        import deepspeed  # noqa: F401
        from deepspeed.runtime.zero import muon  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"`--parallel {parallel}` requires deepspeed >= 0.20.0 "
            f"(module `deepspeed.runtime.zero.muon`); got {e.__class__.__name__}: {e}. "
            "Install/upgrade deepspeed (e.g. `uv pip install 'deepspeed>=0.20.0'`), "
            "or run with `--parallel ddp`."
        ) from e


if args.parallel != "ddp":
    # load deepspeed basic configuration
    import json

    ensure_deepspeed_zero_muon(args.parallel)

    from lightning.pytorch.strategies import DeepSpeedStrategy
    # the template lives next to train.py; fall back to cwd for older layouts
    deepspeed_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepspeed.json')
    if not os.path.isfile(deepspeed_json):
        deepspeed_json = 'deepspeed.json'
    with open(deepspeed_json, 'r') as f:
        deepspeed_config = json.load(f)

    zero_cfg = deepspeed_config.get("zero_optimization")
    if not isinstance(zero_cfg, dict):
        raise RuntimeError(f"deepspeed.json ({deepspeed_json}): missing 'zero_optimization' section")
    stage = zero_cfg.get("stage")
    if stage not in (1, 2):
        raise RuntimeError(
            f"deepspeed.json ({deepspeed_json}): zero_optimization.stage must be 1 or 2, got {stage!r}"
        )

    # update with cmd args
    expected_stage = {"zero1": 1, "zero2": 2}[args.parallel]
    if stage != expected_stage:
        print(f"[deepspeed] overriding zero_optimization.stage {stage} -> {expected_stage} "
              f"(from --parallel {args.parallel})", flush=True)
    zero_cfg["stage"] = expected_stage
    # keep the engine's bookkeeping batch size in sync with the real micro batch size
    deepspeed_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size

    if args.offload_optimizer:
        zero_cfg.setdefault("offload_optimizer", {"device": "cpu", "pin_memory": True})
    else:
        # optimizer offload is opt-in via --offload-optimizer; remove any stale entry from the json
        zero_cfg.pop("offload_optimizer", None)

    strategy = DeepSpeedStrategy(config=deepspeed_config, timeout=datetime.timedelta(hours=2))
else:
    from lightning.pytorch.strategies import DDPStrategy
    strategy = DDPStrategy(timeout=datetime.timedelta(hours=2))

trainer = L.Trainer(
    max_epochs=1,
    accelerator="gpu",
    devices=args.ngpus,
    strategy=strategy,
    precision="bf16-mixed",
    callbacks=[checkpoint_callback, prog_callback, lr_monitor],
    logger=logger,
    log_every_n_steps=20,
    **eval_trainer_kwargs,
    num_sanity_val_steps=-1 if args.val_sanity else 0,
    accumulate_grad_batches=args.grad_accum_steps,
    gradient_clip_val=optimize_config.grad_clip,
    enable_model_summary=True,
)

try:
    trainer.fit(
        model=model,
        ckpt_path=args.resume_checkpoint
    )
except NanTrapTriggered as e:
    print(f"\n[NanTrap] Training aborted due to NaN/Inf: {e}", flush=True)
    if args.nan_dump_dir:
        print(f"[NanTrap] Trigger-point snapshot saved to: {args.nan_dump_dir}  "
              f"(run `python reproduce_nan.py {args.nan_dump_dir}` to reproduce)", flush=True)
    sys.exit(1)
