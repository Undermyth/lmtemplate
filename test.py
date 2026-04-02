import torch
# from torchinfo import summary
import fla
import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from fla.models import GatedDeltaNetForCausalLM
from model import LanguageModel
from module.modeling_switcher import SwitcherConfig, SwitcherModelForCausalLM

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b', legacy=False)
# model = AutoModelForCausalLM.from_pretrained('m-a-p/340M-20B-GatedDeltaNet-pure-baseline', torch_dtype=torch.bfloat16, local_files_only=True).cuda()
# tokenizer = AutoTokenizer.from_pretrained('m-a-p/340M-20B-GatedDeltaNet-pure-baseline', local_files_only=True)

config = SwitcherConfig()
model = SwitcherModelForCausalLM(config).to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b', legacy=False, add_bos_token=True)
ckpt = torch.load('checkpoints/archive/hoplm/hoplm-step=106470-19.5bt.ckpt')
ckpt = {k[6:]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
model.load_state_dict(ckpt)
model = model.cuda()
model.device = torch.device('cuda')
model = HFLM(
    # pretrained="m-a-p/340M-20B-GatedDeltaNet-pure-baseline",
    pretrained=model,
    # tokenizer="m-a-p/340M-20B-GatedDeltaNet-pure-baseline",
    tokenizer=tokenizer,
    dtype=torch.bfloat16,
    max_length=16384,
    backend='causal',
    add_bos_token=True
)
task_manager = TaskManager(
    metadata={
        "max_seq_lengths": [1024, 2048, 4096, 8192],
        # "tokenizer": "m-a-p/340M-20B-GatedDeltaNet-pure-baseline",
        "shuffle": True,
        "enable_cache": True,
        "num_samples": 500,
    },
)
results = lm_eval.simple_evaluate(
    model=model,
    task_manager=task_manager,
    tasks=['hellaswag', 'winogrande', 'piqa'],
    batch_size=1,
    apply_chat_template=False
)        
print(results['results']) 
