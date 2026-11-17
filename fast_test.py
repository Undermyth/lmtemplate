'''
Test whether the basic dataflow works, including fwd, bwd and generation with cache
'''
import argparse

import torch
from transformers import AutoTokenizer

from module.modeling import ModelConfig, ModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-2-7b")
args = parser.parse_args()

config = ModelConfig()
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


