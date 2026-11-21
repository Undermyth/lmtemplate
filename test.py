import argparse
import os
import subprocess
import sys
import tempfile

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoTokenizer

from module.modeling import ModelConfig, ModelForCausalLM, load_model_config
from train_utils import is_deepspeed_checkpoint_dir

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-2-7b")
parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint "
                    "(a Lightning .ckpt file or a DeepSpeed checkpoint directory)")
parser.add_argument("--model-config", type=str, default=None, help="YAML spec file for the model "
                    "scale (see model_0.4b.yaml / model_1.3b.yaml). Defaults to the model_config "
                    "recorded inside a Lightning .ckpt, falling back to model_0.4b.yaml. DeepSpeed "
                    "checkpoint directories carry no such metadata: pass --model-config explicitly "
                    "for anything other than the 0.4B scale.")
parser.add_argument("--tasks", type=str, default="hellaswag,winogrande,piqa", help="Comma-separated list of eval tasks")
parser.add_argument("--convert-output-dir", type=str, default=None,
                    help="Directory for the automatic zero_to_fp32.py conversion output (default: a temp dir under $TMPDIR, /tmp).")
args = parser.parse_args()


def _load_state_dict_file(path: str):
    """Load a checkpoint file into a bare state dict plus its top-level extras.

    Handles both Lightning checkpoint files (with a 'state_dict' key) and bare
    state-dict files such as the output of deepspeed's zero_to_fp32.py. Returns
    ``(state_dict, extras)``: *extras* holds the remaining top-level keys of a
    Lightning checkpoint (used to recover the stored ``model_config``) and is
    empty for bare state-dict files.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        extras = {k: v for k, v in ckpt.items() if k != "state_dict"}
        return ckpt["state_dict"], extras
    if not isinstance(ckpt, dict):
        raise TypeError(f"{path}: expected a checkpoint dict, got {type(ckpt).__name__}")
    return ckpt, {}


def _convert_deepspeed_checkpoint(checkpoint_dir: str, output_dir: str):
    """Run deepspeed's zero_to_fp32.py (shipped inside the checkpoint dir) on ``checkpoint_dir``."""
    script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(
            f"{checkpoint_dir} is a DeepSpeed checkpoint directory but contains no "
            f"zero_to_fp32.py; convert it manually first:\n"
            f"  python zero_to_fp32.py {checkpoint_dir} <output_dir>\n"
            f"then pass the converted directory to --checkpoint-path."
        )
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([sys.executable, script, checkpoint_dir, output_dir], check=True)


def load_model_weights(checkpoint_path: str):
    """Load LanguageModel weights and stored model config from a checkpoint.

    - single ``.ckpt`` file  -> Lightning checkpoint ('state_dict' key, keys like ``model.model.*``);
      the stored ``model_config`` (written by ``on_save_checkpoint``) is returned alongside.
    - DeepSpeed directory    -> prefer an already-converted artifact in the dir root, otherwise
                                run the dir's own ``zero_to_fp32.py`` into a temp dir;
      such directories carry no model_config metadata (``None`` is returned).
    The state dict is normalized to bare model weights (``model.*`` / ``lm_head.*`` prefix stripped).
    """
    model_config_meta = None
    if os.path.isfile(checkpoint_path):
        state_dict, extras = _load_state_dict_file(checkpoint_path)
        model_config_meta = extras.get("model_config")
    elif is_deepspeed_checkpoint_dir(checkpoint_path):
        # 1) an already-converted artifact in the dir root (zero_to_fp32.py output)
        converted = [
            name for name in sorted(os.listdir(checkpoint_path))
            if name.endswith((".bin", ".pt", ".safetensors"))
        ]
        if converted:
            state_dict = {}
            for name in converted:
                sd, _ = _load_state_dict_file(os.path.join(checkpoint_path, name))
                state_dict.update(sd)
        else:
            # 2) convert on the fly into a temp dir (output: pytorch_model.bin, possibly sharded).
            # The fp32 state dict is model-size, so honor $TMPDIR / --convert-output-dir to keep
            # it off a nearly-full root filesystem.
            output_dir = (
                os.path.join(args.convert_output_dir, "zero_to_fp32")
                if args.convert_output_dir else
                tempfile.mkdtemp(prefix="zero_to_fp32_")
            )
            _convert_deepspeed_checkpoint(checkpoint_path, output_dir)
            shards = [
                name for name in sorted(os.listdir(output_dir))
                if name.endswith((".bin", ".safetensors")) and not name.endswith(".index.json")
            ]
            if not shards:
                raise RuntimeError(f"conversion of {checkpoint_path} produced no weight files in {output_dir}")
            state_dict = {}
            for name in shards:
                sd, _ = _load_state_dict_file(os.path.join(output_dir, name))
                state_dict.update(sd)
    else:
        raise FileNotFoundError(
            f"{checkpoint_path}: not a .ckpt file and not a DeepSpeed checkpoint directory "
            "(a directory containing a checkpoint/ subdirectory)"
        )

    # LightningModule keys look like 'model.model.*' / 'model.lm_head.*'; strip one level of
    # 'model.' so the dict matches ModelForCausalLM exactly (strict load below)
    stripped = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    return stripped, model_config_meta


def resolve_model_config(explicit_path, ckpt_model_config) -> ModelConfig:
    """Pick the model-scale config: explicit --model-config > checkpoint metadata > 0.4b default."""
    if explicit_path is not None:
        print(f"[model-config] using explicit --model-config: {explicit_path}")
        return load_model_config(explicit_path)
    if ckpt_model_config is not None:
        print("[model-config] using the model_config recorded in the checkpoint")
        return ModelConfig.from_dict(ckpt_model_config)
    print(
        "[model-config] no --model-config given and the checkpoint carries no model_config; "
        "falling back to model_0.4b.yaml (a scale mismatch fails loudly at the strict load below)"
    )
    return load_model_config("model_0.4b.yaml")


state_dict, ckpt_model_config = load_model_weights(args.checkpoint_path)
config = resolve_model_config(args.model_config, ckpt_model_config)
model = ModelForCausalLM(config).to(torch.bfloat16)
model.load_state_dict(state_dict, strict=True)
print(f"loaded {len(state_dict)} tensors from {args.checkpoint_path}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, legacy=False, add_bos_token=True)
model = model.cuda()
model.device = torch.device('cuda')
model = HFLM(
    pretrained=model,
    tokenizer=tokenizer,
    dtype=torch.bfloat16,
    max_length=16384,
    backend='causal',
    add_bos_token=True
)
task_manager = TaskManager(
    metadata={
        "max_seq_lengths": [1024, 2048, 4096, 8192],
        "shuffle": True,
        "tokenizer": tokenizer,
        "enable_cache": True,
        "num_samples": 500,
    },
)
results = lm_eval.simple_evaluate(
    model=model,
    task_manager=task_manager,
    tasks=args.tasks.split(","),
    batch_size=1,
    apply_chat_template=False
)
print(results['results'])
