"""NanTrap —— 捕捉训练中"第一次"出现 NaN/Inf 的临界点并固化现场。

核心思想
========
训练中 loss 与 grad norm 同时变 NaN 通常是下面两种情形之一：

  * 情形 (a): 某一步 k 的 forward 在权重 W_k 上、对某条样本 sample_k 计算出
              NaN loss（bf16 溢出 / 病态数值），于是 backward 产生 NaN 梯度。
  * 情形 (b): forward 的 loss 有限，但 backward 中某个算子（除法、CG 迭代等）
              产生 NaN 梯度。

无论哪种情形，**在 optimizer.step() 之前**，权重都还是有限的 W_k：
保存 (W_k, 触发样本) 之后离线重放，就能在"临界点"上复现首次 NaN，
再用 hooks / autograd anomaly 定位到具体算子。

本模块在三个检测点工作（每个检测点都先做跨 rank 的 all-reduce，任一 rank
非有限则所有 rank 一起 dump + 中止，避免 DDP 下单 rank 提前退出卡死）：

  1. training_step 中 forward 之后  -> 检查 loss（含 aux_loss）       (kind="loss")
  2. backward() 中 loss.backward() 之后 -> 检查所有参数梯度           (kind="grad")
  3. optimizer_step 中 step() 之后  -> 检查权重本身                   (kind="weight", 兜底)

其中 (3) 是 (a)/(b) 的盲区兜底：若优化器把有限梯度变成 NaN 权重（例如 Muon
的 Newton-Schulz 迭代发散这类 bug），此时保存的权重已被污染，(a)/(b) 的
"正常模型 + 触发样本"前提不成立，dump 的 meta 里 `params_all_finite_at_dump`
会标记为 False，复现脚本会给出明确警告。

用法
====
给 LightningModule 混入 NanTrapMixin，并把 nan_dump_dir 传给构造函数即可。
不传 nan_dump_dir 时所有检测均为 no-op，正常训练零开销零行为变化。
"""

import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch

__all__ = ["NanTrapTriggered", "NanTrapMixin", "params_finite", "grads_finite"]


class NanTrapTriggered(RuntimeError):
    """捕捉到首个 NaN/Inf 后抛出；train.py 捕获它并打印 dump 位置后退出。"""


def params_finite(module: torch.nn.Module) -> bool:
    return all(
        torch.isfinite(p).all()
        for p in module.parameters()
        if p.dtype.is_floating_point
    )


def grads_finite(module: torch.nn.Module) -> bool:
    return all(
        torch.isfinite(p.grad).all()
        for p in module.parameters()
        if p.grad is not None
    )


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.getcwd(),
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class NanTrapMixin:
    """混入 L.LightningModule。构造时设置 ``self._nan_dir = nan_dump_dir``（None 则禁用）。"""

    _nan_dir: Optional[str] = None
    _nan_dump_optimizer: bool = False
    _nan_window: List[Tuple[torch.Tensor, torch.Tensor]] = []
    _nan_last_state: Dict[str, Any] = {}

    # ------------------------------------------------------------- 工具
    def _nan_rank(self) -> int:
        try:
            return int(self.global_rank)
        except Exception:
            return int(getattr(self, "_global_rank", 0))

    def _nan_world_size(self) -> int:
        try:
            trainer = getattr(self, "trainer", None)
            return int(trainer.world_size) if trainer is not None else 1
        except Exception:
            return 1

    def _nan_any(self, local_bad: bool) -> bool:
        """任一 rank 非有限则返回 True。必须是所有 rank 同时调用的集体操作。"""
        if not self._nan_dir:
            return local_bad
        if (
            self._nan_world_size() > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            flag = torch.tensor(1 if local_bad else 0, dtype=torch.int, device=self.device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            return bool(flag.item())
        return local_bad

    # ------------------------------------------------------------- 检测
    def _nan_track_sample(self, x, y, batch_idx, state_dict):
        """把当前 micro-batch 记入滚动窗口（CPU 拷贝），供 dump 时保存整窗样本。"""
        if not self._nan_dir:
            return
        try:
            pq = int(state_dict.get("pq_idx", -1))
        except Exception:
            pq = -1
        try:
            rg = int(state_dict.get("rg_idx", -1))
        except Exception:
            rg = -1
        self._nan_last_state = {
            "batch_idx": int(batch_idx),
            "micro_idx": int(batch_idx) % max(1, int(self.grad_accum)),
            "pq_idx": pq,
            "rg_idx": rg,
        }
        self._nan_window.append((x.detach().cpu(), y.detach().cpu()))
        maxlen = max(1, int(getattr(self, "grad_accum", 1)))
        if len(self._nan_window) > maxlen:
            self._nan_window.pop(0)

    def _nan_trap(self, kind: str, loss=None, aux=None, extra: Optional[Dict[str, Any]] = None):
        """kind in {'loss','grad','weight'}：本地检查 -> 跨 rank 协调 -> dump -> 中止。"""
        if not self._nan_dir:
            return
        bad = False
        details: Dict[str, Any] = {}
        if loss is not None:
            if not bool(torch.isfinite(loss).all()):
                bad = True
                details["loss"] = float(loss.detach().float())
        if aux is not None:
            if not bool(torch.isfinite(aux).all()):
                bad = True
                details["aux"] = float(aux.detach().float())
        if kind == "grad":
            bad_names = [
                n for n, p in self.model.named_parameters()
                if p.grad is not None and not bool(torch.isfinite(p.grad).all())
            ]
            if bad_names:
                bad = True
                details["bad_grad_params"] = bad_names[:20]
                details["bad_grad_count"] = len(bad_names)
        elif kind == "weight":
            if not params_finite(self.model):
                bad = True
                details["note"] = (
                    "post-optimizer-step weights non-finite while the pre-step grad check "
                    "passed -> optimizer suspected (finite grads became NaN weights)"
                )
        if extra:
            details.update(extra)
        if not self._nan_any(bad):
            return
        self._nan_dump(kind, details)
        # 所有 rank 都完成 dump 后再一起中止，避免 IO 耗时差异导致 teardown 竞态
        if (
            self._nan_world_size() > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.barrier()
        raise NanTrapTriggered(
            f"[NanTrap] non-finite detected in '{kind}' (rank {self._nan_rank()}): {details}"
        )

    # ------------------------------------------------------------- dump
    def _nan_dump(self, kind: str, details: Dict[str, Any]):
        try:
            trainer = getattr(self, "trainer", None)
            step = int(trainer.global_step) if trainer is not None else -1
        except Exception:
            step = -1
        rank = self._nan_rank()
        ts = time.strftime("%Y%m%d-%H%M%S")
        dump_dir = Path(self._nan_dir) / f"rank{rank}_step{step}_{ts}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        # 1) 权重 W_k（CPU 拷贝）—— 触发 NaN 的 forward 所用的、尚未被 step 污染的权重
        sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        _atomic_save(sd, dump_dir / "model.pt")

        # 2) 当前梯度累积窗口内的全部 micro-batch 样本（触发样本通常是最后一条）
        torch.save(
            {i: {"x": x.cpu(), "y": y.cpu()} for i, (x, y) in enumerate(self._nan_window)},
            dump_dir / "samples.pt",
        )

        # 3) RNG 状态（dropout / 非确定性 kernel 复现用）
        rng: Dict[str, Any] = {"cpu": torch.get_rng_state()}
        try:
            rng["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
        torch.save(rng, dump_dir / "rng.pt")

        # 4) 元信息
        cfg = getattr(getattr(self, "model", None), "config", None)
        try:
            model_config = asdict(cfg) if cfg is not None else {}
        except TypeError:
            # 非 dataclass 的 config（dict / SimpleNamespace 等）直接原样记录
            model_config = dict(cfg) if isinstance(cfg, dict) else {}
        meta = {
            "kind": kind,
            "details": details,
            "rank": rank,
            "world_size": self._nan_world_size(),
            "global_step": step,
            "grad_accum": int(getattr(self, "grad_accum", 1)),
            "use_gradient_checkpoint": bool(getattr(self, "use_gradient_checkpoint", False)),
            "params_all_finite_at_dump": bool(params_finite(self.model)),
            "window_size": len(self._nan_window),
            "micro_batch": self._nan_last_state,
            "model_config": model_config,
            "torch": torch.__version__,
            "lightning": L.__version__,
            "git": _git_rev(),
            "time": time.ctime(),
        }
        (dump_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[NanTrap] critical state dumped to {dump_dir}", flush=True)

        # 5) 可选：优化器状态（复现 optimizer 污染场景用，体积可能数 GB）
        if getattr(self, "_nan_dump_optimizer", False):
            try:
                opt = self.optimizers()
                if isinstance(opt, (list, tuple)):
                    opt = opt[0]
                _atomic_save(opt.state_dict(), dump_dir / "optimizer.pt")
            except Exception as e:
                print(f"[NanTrap] failed to save optimizer state: {e}", flush=True)
