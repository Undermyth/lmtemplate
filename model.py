
from dataclasses import dataclass
from typing import List, Literal, Optional

import lightning as L
import lm_eval  # type: ignore
import torch
from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam as ZeroMuonWithAuxAdam
from lm_eval.models.huggingface import HFLM  # type: ignore
from lm_eval.tasks import TaskManager  # type: ignore
from rich.pretty import pprint
from torch.utils.data import DataLoader

from data.stream_parquet import StreamingParquet
from muon import MuonWithAuxAdam as DDPMuonWithAdam
from muon import SingleDeviceMuonWithAuxAdam
from train_utils import create_warmup_cosine_scheduler, get_optimizer_steps


@dataclass
class OptimizeConfig:
    peak_lr: float = 0.02
    warmup_ratio: float = 0.005
    min_lr_frac: float = 0.01
    weight_decay: float = 0.1
    grad_clip: Optional[float] = None

# recommended optimize config for AdamW:
# peak_lr = 3e-4
# weight_decay = 1e-3


class LanguageModel(L.LightningModule):
    def __init__(
        self, 
        model, 
        tokenizer, 
        parquet_path: str, 
        seq_len: int, 
        batch_size: int, 
        grad_accum_steps: int, 
        n_gpus: int,
        optimize_tokens: int,
        eval_tasks: List[str],
        optimizer: Literal["adam", "muon"] = "muon",
        optimize_config: OptimizeConfig = OptimizeConfig(),
        use_gradient_checkpoint: bool = False,
        use_parallel: bool = True,
        use_advanced_parallel: bool = False,    # whether parallel like FSDP/ZeRO is used
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.grad_accum = grad_accum_steps
        self.n_gpus = n_gpus
        self.optimize_tokens = optimize_tokens
        self.eval_tasks = eval_tasks
        self.optimizer = optimizer
        self.optimize_config = optimize_config
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.use_parallel = use_parallel
        self.use_advanced_parallel = use_advanced_parallel

        # resume training dataset
        self.pq_idx = 0
        self.rg_idx = None

        self.automatic_optimization = True
        self.stream_loss = 0

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            ddp_rank = self.global_rank
            world_size = self.trainer.world_size
            self.train_dataset = StreamingParquet(self.parquet_path, self.batch_size, self.seq_len, self.tokenizer, ddp_rank=ddp_rank, world_size=world_size, split='train')
            # self.test_dataset = StreamingParquet(self.parquet_path, self.batch_size, self.seq_len, self.tokenizer, ddp_rank=ddp_rank, world_size=world_size, split='test')
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1)    # distributed sampling and batching done within StreamingParquet

    def val_dataloader(self):
         # return DataLoader(self.test_dataset, batch_size=1)    # dummy
         return [1]

    def training_step(self, batch, batch_idx):
        x, y, state_dict = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        self.pq_idx = state_dict['pq_idx']
        self.rg_idx = state_dict['rg_idx']
        output = self.model(input_ids=x, label=y, use_gradient_checkpoint=self.use_gradient_checkpoint)
        loss = output.loss
        aux_loss = output.aux_loss
        if aux_loss is not None:
            loss = loss + 0.1 * aux_loss

        self.stream_loss += loss.item() / self.grad_accum
        if (batch_idx + 1) % self.grad_accum == 0:
            total_norm = torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None]).pow(2).sum().pow(0.5)
            self.log('train/grad_norm', total_norm, on_step=True, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss', self.stream_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            if aux_loss is not None:
                self.log('train/aux_loss', aux_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.stream_loss = 0
        
        self.log('train/pq_idx', self.pq_idx.item(), on_step=True, prog_bar=True, logger=False)
        self.log('train/rg_idx', self.rg_idx.item(), on_step=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, idx):
        pass

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            self.model.device = torch.device(f'cuda:{self.global_rank}')
            model = HFLM(
                pretrained=self.model,
                tokenizer=self.tokenizer,
                backend='causal',
                add_bos_token=True,
                max_length=16384,
            )
            task_manager = TaskManager(
                metadata={
                    "max_seq_lengths": [1024, 2048, 4096],
                    "tokenizer": self.tokenizer,
                    "shuffle": True,
                    "enable_cache": True,
                    "num_samples": 100,
                },
            )
            results = lm_eval.simple_evaluate(
                model=model,
                task_manager=task_manager,
                tasks=self.eval_tasks,
                batch_size=4,
                apply_chat_template=False
            )        
            pprint(results['results'], expand_all=True, indent_guides=True)
            for task_name, result in results['results'].items():
                if 'niah' in task_name:
                    for metric_name, metric in result.items():
                        if metric_name.split(',')[0].isdigit():
                            self.log(f'{task_name}/{metric_name.split(",")[0]}-acc', metric, logger=True)
                else:
                    if 'acc,none' in result.keys():
                        self.log(f'{task_name}/acc', result['acc,none'], logger=True)
                    if 'perplexity,none' in result.keys():
                        self.log(f'{task_name}/perplexity', result['perplexity,none'], logger=True)
        self.trainer.strategy.barrier()
                
        
    def on_save_checkpoint(self, checkpoint):
        pq_idx = torch.tensor([self.pq_idx], device=self.device)        
        rg_idx = torch.tensor([self.rg_idx], device=self.device)
        pq_idx = self.all_gather(pq_idx)
        rg_idx = self.all_gather(rg_idx)
        if self.global_rank == 0:
            checkpoint['dataset_state_dict'] = {'pq_idx': pq_idx, 'rg_idx': rg_idx}

    def on_load_checkpoint(self, checkpoint):
        pq_idx = checkpoint['dataset_state_dict']['pq_idx']
        rg_idx = checkpoint['dataset_state_dict']['rg_idx']
        self.pq_idx = pq_idx[self.global_rank].item()
        self.rg_idx = rg_idx[self.global_rank].item()
        self.print(f'resume to dataset at pq_idx = {self.pq_idx}, rg_idx = {self.rg_idx}')
        state_dict = {'pq_idx': pq_idx[self.global_rank].item(), 'rg_idx': rg_idx[self.global_rank].item()}
        self.train_dataset.load_state_dict(state_dict)
        # self.tune_optimizer(checkpoint)

    def tune_optimizer(self, state_dict):
        state_dict['lr_schedulers'][0]['_schedulers'][1]['T_max'] = 53893
        state_dict['lr_schedulers'][1]['_schedulers'][1]['T_max'] = 53893
    
    def build_muon_optimizers(self, distributed: bool = True, use_zero: bool = False):    # use deepspeed ZeRO Muon optimizer. Only support single scheduler
        peak_lr = self.optimize_config.peak_lr
        optimizer_steps = get_optimizer_steps(
            n_optimize_tokens=self.optimize_tokens,
            n_grad_accum_steps=self.grad_accum,
            n_gpus=self.n_gpus,
            micro_batch_size=self.batch_size,
            seq_len=self.seq_len
        )
        warmup_steps = self.optimize_config.warmup_ratio * optimizer_steps
        emb_params = list(self.model.get_input_embeddings().parameters())
        hidden_1d_params = [p for n, p in self.model.get_decoder().named_parameters() if p.dim() < 2 or p.dim() == 3]    # 3 for causal convolution parameters
        hidden_1d_params = hidden_1d_params + [p for n, p in self.model.get_output_norm().named_parameters()]
        hidden_2d_params = [p for n, p in self.model.get_decoder().named_parameters() if p.dim() == 2]
        for p in hidden_2d_params:
            p.use_muon = True
        head_params = list(self.model.get_output_embeddings().parameters())
        adam_groups = [
            dict(params=emb_params, lr=10 * peak_lr),
            dict(params=head_params, lr=0.1 * peak_lr),
            dict(params=hidden_1d_params, lr=peak_lr)
        ]
        adam_groups = [dict(**g, betas=(0.9, 0.95), use_muon=False, weight_decay=self.optimize_config.weight_decay) for g in adam_groups]
        muon_group = dict(params=hidden_2d_params, lr=peak_lr, momentum=0.95, use_muon=True, weight_decay=self.optimize_config.weight_decay)
        param_groups = [*adam_groups, muon_group]
        Muon = SingleDeviceMuonWithAuxAdam 
        if distributed:
            Muon = ZeroMuonWithAuxAdam if use_zero else DDPMuonWithAdam
        optimizer = Muon(param_groups)
        scheduler = create_warmup_cosine_scheduler(
            optimizer=optimizer, warmup_epochs=warmup_steps, total_epochs=optimizer_steps, eta_min=self.optimize_config.min_lr_frac * peak_lr
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def build_adam_optimizers(self):
        peak_lr = self.optimize_config.peak_lr
        optimizer_steps = get_optimizer_steps(
            n_optimize_tokens=self.optimize_tokens,
            n_grad_accum_steps=self.grad_accum,
            n_gpus=self.n_gpus,
            micro_batch_size=self.batch_size,
            seq_len=self.seq_len
        )
        warmup_steps = self.optimize_config.warmup_ratio * optimizer_steps
        emb_params = list(self.model.get_input_embeddings().parameters())
        hidden_1d_params = [p for n, p in self.model.get_decoder().named_parameters() if p.dim() < 2 or p.dim() == 3]    # 3 for causal convolution parameters
        hidden_1d_params = hidden_1d_params + [p for n, p in self.model.get_output_norm().named_parameters()]
        hidden_2d_params = [p for n, p in self.model.get_decoder().named_parameters() if p.dim() == 2]
        head_params = list(self.model.get_output_embeddings().parameters())
        groups = [
            dict(params=emb_params, lr=peak_lr, weight_decay=self.optimize_config.weight_decay),
            dict(params=head_params, lr=peak_lr, weight_decay=self.optimize_config.weight_decay),
            dict(params=hidden_2d_params, lr=peak_lr, weight_decay=self.optimize_config.weight_decay),
            dict(params=hidden_1d_params, lr=peak_lr, weight_decay=0.0)
        ]
        param_groups = [dict(**g, betas=(0.9, 0.95)) for g in groups]
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = create_warmup_cosine_scheduler(
            optimizer=optimizer, warmup_epochs=warmup_steps, total_epochs=optimizer_steps, eta_min=self.optimize_config.min_lr_frac * peak_lr
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        
    def configure_optimizers(self):
        if self.optimizer == "muon":
            return self.build_muon_optimizers(distributed=self.use_parallel, use_zero=self.use_advanced_parallel)
        elif self.optimizer == "adam":
            return self.build_adam_optimizers()
        
