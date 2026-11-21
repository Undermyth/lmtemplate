'''
Test whether the basic dataflow works, including fwd, bwd and generation with cache
'''
import argparse

import torch
from transformers import AutoTokenizer

from module.modeling import ModelForCausalLM, load_model_config

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-2-7b")
parser.add_argument("--model-config", type=str, default="model_0.4b.yaml",
                    help="YAML spec file for the model scale (see model_0.4b.yaml / model_1.3b.yaml)")
args = parser.parse_args()

config = load_model_config(args.model_config)
model = ModelForCausalLM(config).to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, legacy=False)
model = model.cuda()
model.train()

# batched forward & backward
x = torch.randint(1, 10258, (4, 2048), dtype=torch.long).cuda()
label = torch.randint(1, 10258, (4, 2048), dtype=torch.long).cuda()
output = model(x, label=label)
output.loss.backward()

# inference
model.eval()
model.generate(x, max_new_tokens=64)


