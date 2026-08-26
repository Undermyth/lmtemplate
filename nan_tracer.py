"""NanTrap -- capture the "first" NaN/Inf trigger point during training and freeze the scene.

Core idea
=========
When both loss and grad norm turn NaN during training, it is usually one of two cases:

  * Case (a): at step k, forward computes a NaN loss on weight W_k with some sample sample_k
              (bf16 overflow / ill-conditioned values), so backward produces NaN gradients.
  * Case (b): forward loss is finite, but some op in backward (division, CG iterations, etc.)
              produces NaN gradients.

In either case, **before optimizer.step()** the weights are still finite W_k:
saving (W_k, triggering sample) and replaying offline reproduces the first NaN at the
"trigger point", then hooks / autograd anomaly pinpoint the exact op.

This module works at three checkpoints (each does a cross-rank all-reduce first: if any rank
is non-finite, all ranks dump + abort together, avoiding a deadlock from one rank exiting early):

  1. after forward in training_step     -> check loss (incl. aux_loss)    (kind="loss")
  2. after loss.backward() in backward  -> check all param gradients      (kind="grad")
  3. after step() in optimizer_step     -> check weights themselves       (kind="weight", fallback)

(3) is the blind-spot fallback for (a)/(b): if the optimizer turns finite gradients into NaN
weights (e.g. a Muon Newton-Schulz iteration divergence bug), the saved weights are already
contaminated, the "healthy model + triggering sample" premise of (a)/(b) breaks, and
`params_all_finite_at_dump` in the dump meta is flagged False; the reproduction script warns clearly.

Usage
=====
Mix NanTrapMixin into the LightningModule and pass nan_dump_dir to the constructor.
Without nan_dump_dir all checks are no-ops: zero overhead, zero behavior change.
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
    """Raised once the first NaN/Inf is caught; train.py catches it, prints the dump location, and exits."""


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
    """Mixin for L.LightningModule. The constructor sets ``self._nan_dir = nan_dump_dir`` (None disables)."""

    _nan_dir: Optional[str] = None
    _nan_dump_optimizer: bool = False
    _nan_window: List[Tuple[torch.Tensor, torch.Tensor]] = []
    _nan_last_state: Dict[str, Any] = {}

    # ------------------------------------------------------------- helpers
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
        """Return True if any rank is non-finite. Must be a collective called by all ranks simultaneously."""
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

    # ------------------------------------------------------------- checks
    def _nan_track_sample(self, x, y, batch_idx, state_dict):
        """Record the current micro-batch into the rolling window (CPU copy) for saving full-window samples on dump."""
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
        """kind in {'loss','grad','weight'}: local check -> cross-rank coordination -> dump -> abort."""
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
        # abort all ranks only after every rank finishes dumping, to avoid teardown races from uneven IO time
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

        # 1) weights W_k (CPU copy) -- the weights used by the NaN-triggering forward, not yet polluted by step
        sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        _atomic_save(sd, dump_dir / "model.pt")

        # 2) all micro-batch samples in the current grad-accum window (the trigger sample is usually the last one)
        torch.save(
            {i: {"x": x.cpu(), "y": y.cpu()} for i, (x, y) in enumerate(self._nan_window)},
            dump_dir / "samples.pt",
        )

        # 3) RNG state (for reproducing dropout / non-deterministic kernels)
        rng: Dict[str, Any] = {"cpu": torch.get_rng_state()}
        try:
            rng["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
        torch.save(rng, dump_dir / "rng.pt")

        # 4) metadata
        cfg = getattr(getattr(self, "model", None), "config", None)
        try:
            model_config = asdict(cfg) if cfg is not None else {}
        except TypeError:
            # non-dataclass config (dict / SimpleNamespace etc.) is recorded as-is
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

        # 5) optional: optimizer state (for reproducing optimizer-corruption scenarios; can be several GB)
        if getattr(self, "_nan_dump_optimizer", False):
            try:
                opt = self.optimizers()
                if isinstance(opt, (list, tuple)):
                    opt = opt[0]
                _atomic_save(opt.state_dict(), dump_dir / "optimizer.pt")
            except Exception as e:
                print(f"[NanTrap] failed to save optimizer state: {e}", flush=True)
