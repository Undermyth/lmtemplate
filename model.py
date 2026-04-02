
from dataclasses import dataclass
from typing import Literal, Optional

import lightning as L
import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from torch.utils.data import DataLoader

from data.stream_parquet import StreamingParquet
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
        optimizer: Literal["adam", "muon"] = "muon",
        optimize_config: OptimizeConfig = OptimizeConfig(),
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
        self.optimizer = optimizer
        self.optimize_config = optimize_config
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # resume training dataset
        self.pq_idx = 0
        self.rg_idx = None

        self.automatic_optimization = False
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
        output = self.model(input_ids=x, label=y)
        loss = output.loss

        # manual optimization for multiple optimizers
        loss = loss / self.grad_accum
        self.stream_loss += loss.item()
        self.manual_backward(loss)
        if (batch_idx + 1) % self.grad_accum == 0:

            # get optimizers and schdulers
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
            schs = self.lr_schedulers()
            if not isinstance(schs, list):
                schs = [schs]

            # optimize and schedule    
            for opt in opts:
                if self.optimize_config.grad_clip is not None:
                    self.clip_gradients(opt, gradient_clip_val=self.optimize_config.grad_clip, gradient_clip_algorithm='norm')
                opt.step()
                opt.zero_grad()
            for sch in schs:
                sch.step()

            self.log('train/loss', self.stream_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.stream_loss = 0
        
        # self.log('train/loss', loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log('train/pq_idx', self.pq_idx.item(), on_step=True, prog_bar=True, logger=False)
        self.log('train/rg_idx', self.rg_idx.item(), on_step=True, prog_bar=True, logger=False)
        return loss
   

    def validation_step(self, batch, idx):
        pass

    def on_validation_epoch_end(self):
        self.model.device = torch.device(f'cuda:{self.global_rank}')
        model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            backend='causal',
            add_bos_token=True
        )
        task_manager = TaskManager(
            metadata={
                "max_seq_lengths": [1024, 2048, 4096, 8192],
                "tokenizer": self.tokenizer,
                "shuffle": True,
                "enable_cache": True,
                "num_samples": 500,
            },
        )
        results = lm_eval.simple_evaluate(
            model=model,
            task_manager=task_manager,
            tasks=['lambada_openai', 'arc_challenge'],
            batch_size=4,
            apply_chat_template=False
        )        
        self.print(results['results'])
        self.log('lambada_openai/perplexity', results['results']['lambada_openai']['perplexity,none'], logger=True, sync_dist=True)
        self.log('lambada_openai/acc', results['results']['lambada_openai']['acc,none'], logger=True, sync_dist=True)
        self.log('arc/easy', results['results']['arc_challenge']['acc,none'], logger=True, sync_dist=True)
        
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

    def build_muon_optimizers(self):
        peak_lr = self.optimize_config.peak_lr
        optimizer_steps = get_optimizer_steps(
            n_optimize_tokens=self.optimize_tokens,
            n_grad_accum_steps=self.grad_accum,
            n_gpus=self.n_gpus,
            micro_batch_size=self.batch_size,
            seq_len=self.seq_len
        )
        warmup_steps = self.optimize_config.warmup_ratio * optimizer_steps
        emb_params = list(self.model.model.emb.parameters())
        hidden_1d_params = [p for n, p in self.model.model.enc.named_parameters() if p.dim() < 2 or p.dim() == 3]    # 3 for causal convolution parameters
        hidden_2d_params = [p for n, p in self.model.model.enc.named_parameters() if p.dim() == 2]
        head_params = list(self.model.lm_head.parameters())
        adam_groups = [
            dict(params=emb_params, lr=10 * peak_lr),
            dict(params=head_params, lr=0.1 * peak_lr),
            dict(params=hidden_1d_params, lr=peak_lr)
        ]
        adam_opt = torch.optim.AdamW(adam_groups, betas=(0.8, 0.95), weight_decay=self.optimize_config.weight_decay)
        muon_opt = torch.optim.Muon(hidden_2d_params, lr=peak_lr, momentum=0.95, weight_decay=self.optimize_config.weight_decay)
        adam_scheduler = create_warmup_cosine_scheduler(
            optimizer=adam_opt, warmup_epochs=warmup_steps, total_epochs=optimizer_steps, eta_min=self.optimize_config.min_lr_frac * 0.1 * peak_lr
        )
        muon_scheduler = create_warmup_cosine_scheduler(
            optimizer=muon_opt, warmup_epochs=270, total_epochs=54163, eta_min=self.optimize_config.min_lr_frac * peak_lr
        )
        adam_scheduler_cfg = {
            'scheduler': adam_scheduler,
            'interval': "step",
            'frequency': 1
        }
        muon_scheduler_cfg = {
            'scheduler': muon_scheduler,
            'interval': "step",
            'frequency': 1
        }
        return [adam_opt, muon_opt], [adam_scheduler_cfg, muon_scheduler_cfg]

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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=self.optimize_config.weight_decay)
        scheduler = create_warmup_cosine_scheduler(
            optimizer=optimizer, warmup_epochs=warmup_steps, total_epochs=optimizer_steps, eta_min=self.optimize_config.min_lr_frac * peak_lr
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        
    def configure_optimizers(self):
        if self.optimizer == "muon":
            return self.build_muon_optimizers()
        elif self.optimizer == "adam":
            return self.build_adam_optimizers()
        
