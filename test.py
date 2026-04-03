import argparse

import lm_eval  # type: ignore
import torch
from lm_eval.models.huggingface import HFLM  # type: ignore
from lm_eval.tasks import TaskManager  # type: ignore
from transformers import AutoTokenizer

from module.modeling import ModelConfig, ModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-2-7b")
parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--tasks", type=str, default="hellaswag,winogrande,piqa", help="Comma-separated list of eval tasks")
args = parser.parse_args()

config = ModelConfig()
model = ModelForCausalLM(config).to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, legacy=False, add_bos_token=True)
ckpt = torch.load(args.checkpoint_path)
ckpt = {k[6:]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
model.load_state_dict(ckpt)
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
