#!/usr/bin/env python
"""reproduce_nan.py -- reproduce the "first NaN trigger point" saved by NanTrap.

Usage:
    python reproduce_nan.py <dump_dir> [--no-backward] [--anomaly] [--probe-samples N]

Procedure:
  1. Load the dump: W_k weights / window samples / RNG / metadata.
  2. Verify the "healthy model" premise: all weights must be finite (otherwise the
     weights were already corrupted at dump time -- an optimizer-corruption scenario
     or a too-late checkpoint -- and "first" cannot be reproduced).
  3. Replay forward on each sample exactly as in training (same call / same bf16 / same flags).
     - If loss/aux is non-finite -> use forward hooks to locate the first op producing inf/NaN
       (hooks are attached to every submodule, architecture-agnostic), and print the max|.| trace
       of each module's activations.
  4. If forward is fully finite -> replay forward+backward:
     - List the params with non-finite grads; use backward hooks to find the first layer with
       non-finite grads in backward execution order (= where NaN originates); --anomaly enables
       autograd anomaly detection so torch reports the offending op directly.
  5. Replay the full window with accumulated forward+backward (matching training grad-accum semantics).
  6. --probe-samples N: probe W_k with random samples to tell "this sample triggers it"
     from "the model is already sick (any sample blows up)".
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------- dump loading
def load_dump(dump_dir: Path):
    meta = json.loads((dump_dir / "meta.json").read_text())
    samples = torch.load(dump_dir / "samples.pt", map_location="cpu", weights_only=False)
    rng = torch.load(dump_dir / "rng.pt", map_location="cpu", weights_only=False)
    return meta, samples, rng


def build_model(meta, dump_dir: Path):
    from module.modeling import ModelConfig, ModelForCausalLM
    cfg = ModelConfig(**meta["model_config"])
    model = ModelForCausalLM(cfg).to(torch.bfloat16).cuda()
    sd = torch.load(dump_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.train()
    return model, cfg


def params_finite(model) -> bool:
    return all(
        torch.isfinite(p).all()
        for p in model.parameters()
        if p.dtype.is_floating_point
    )


def restore_rng(rng):
    try:
        torch.set_rng_state(rng["cpu"])
    except Exception as e:
        print(f"[warn] restore cpu rng failed: {e}")
    try:
        if "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as e:
        print(f"[warn] restore cuda rng failed: {e}")


def compute_loss(model, x, y, use_ckpt):
    out = model(input_ids=x, label=y, use_gradient_checkpoint=use_ckpt)
    loss = out.loss
    aux = out.aux_loss
    total = loss
    if aux is not None:
        total = total + 0.1 * aux
    return total, loss, aux


def fmt(t):
    try:
        v = float(t.detach())
    except Exception:
        try:
            v = float(t)
        except Exception:
            return str(t)
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "+inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.6g}"


# ---------------------------------------------------------------- locating utilities
def _register_all_forward_hooks(root, record):
    """Attach forward hooks to all submodules of root (incl. root itself); append
    (name, bad_count, max_abs) to record in forward execution order. Architecture-agnostic."""
    handles = []

    def mk(name):
        def hook(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) else None)
            if t is None or not t.is_floating_point():
                return
            with torch.no_grad():
                f = t.isfinite()
                record.append((name, int((~f).sum()), float(t.abs().max())))
        return hook

    def walk(prefix, mod):
        handles.append(mod.register_forward_hook(mk(prefix or type(mod).__name__)))
        for child_name, child in mod.named_children():
            walk(f"{prefix}.{child_name}" if prefix else child_name, child)

    walk("", root)
    return handles


def finite_report(model, x, y):
    """Grad-free forward with all-submodule hooks: returns (events, first_bad). events record
    (name, bad_count, max_abs) in forward execution order."""
    events = []
    handles = _register_all_forward_hooks(model.model, events)
    try:
        with torch.no_grad():
            model.model(x, use_gradient_checkpoint=False)
    finally:
        for h in handles:
            h.remove()
    first_bad = next(((n, c, m) for n, c, m in events if c > 0), None)
    return events, first_bad


def grad_report(model, x, y, anomaly=False):
    """forward+backward: returns (bad_param_names, bwd_events). bwd_events records the first
    layer with non-finite grads in backward execution order (= where NaN originates)."""
    model.zero_grad(set_to_none=True)
    if anomaly:
        torch.autograd.set_detect_anomaly(True)
    bwd_events = []
    handles = []

    def hk(mod, gin, gout, name):
        g = gin[0] if gin else None
        if g is not None and g.is_floating_point():
            n = int((~g.isfinite()).sum())
            if n:
                bwd_events.append((name, n))

    def walk(prefix, mod):
        handles.append(mod.register_full_backward_hook(
            lambda m, gi, go, _p=prefix: hk(m, gi, go, _p)))
        for child_name, child in mod.named_children():
            walk(f"{prefix}.{child_name}" if prefix else child_name, child)

    walk("", model.model)
    try:
        total, _, _ = compute_loss(model, x, y, use_ckpt=False)
        total.backward()
    except Exception as e:
        bwd_events.append(("EXCEPTION", str(e)))
    finally:
        for h in handles:
            h.remove()
        if anomaly:
            torch.autograd.set_detect_anomaly(False)
    bad = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not bool(torch.isfinite(p.grad).all())
    ]
    return bad, bwd_events


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--no-backward", action="store_true", help="replay forward only, skip backward")
    ap.add_argument("--anomaly", action="store_true",
                    help="enable torch autograd NaN anomaly detection during backward (slow, but pinpoints the NaN op)")
    ap.add_argument("--probe-samples", type=int, default=0,
                    help="probe W_k with N random samples to tell sample-triggered from a sick model")
    args = ap.parse_args()

    meta, samples, rng = load_dump(args.dump_dir)
    print("=" * 78)
    print(f"dump meta: kind={meta.get('kind')}  global_step={meta.get('global_step')}  "
          f"rank={meta.get('rank')}  window={meta.get('window_size')}  grad_accum={meta.get('grad_accum')}")
    print(f"  details: {json.dumps(meta.get('details', {}), default=str)}")
    print(f"  micro_batch: {meta.get('micro_batch')}  use_gradient_checkpoint={meta.get('use_gradient_checkpoint')}")

    model, cfg = build_model(meta, args.dump_dir)
    restore_rng(rng)

    ok_weights = params_finite(model)
    if ok_weights:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[ok] all weights finite ({n_params} params) -- 'healthy model' premise holds")
    else:
        print("[!!!] saved weights are not all finite -- weights were already NaN-contaminated at dump time.")
        print("      (usually an optimizer-corruption scenario, or the checkpoint was set too late.)")
        print("      this dump cannot reproduce the 'first NaN'; results below are for reference only.")

    loss_nan_micros = []
    grad_nan_micros = []
    first_bad_tag = None
    for k in sorted(samples.keys()):
        s = samples[k]
        x = s["x"].cuda()
        y = s["y"].cuda()
        print("-" * 78)
        print(f"micro-batch #{k}: x={tuple(x.shape)} y={tuple(y.shape)}")
        restore_rng(rng)
        total, loss, aux = compute_loss(model, x, y, use_ckpt=bool(meta.get("use_gradient_checkpoint", False)))
        tf = bool(torch.isfinite(total).all())
        print(f"  forward : total_loss={fmt(total)}  finite={tf}"
              + (f"  loss={fmt(loss)}  aux={fmt(aux)}" if aux is not None else ""))
        if not tf:
            loss_nan_micros.append(k)
            events, first_bad = finite_report(model, x, y)
            if first_bad is not None and first_bad_tag is None:
                first_bad_tag = first_bad[0]
            # print per-layer max|.| trace (backbone modules only, to avoid aux detail spam)
            for name, c, mx in events:
                parts = name.split(".")
                is_backbone = (
                    len(parts) <= 3
                    and (parts[0] == "emb" or parts[0].startswith("layer") or parts[0] in ("norm", "rotary_emb"))
                )
                if is_backbone or c > 0:
                    flag = "   <== NON-FINITE" if c > 0 else ""
                    print(f"    {name:36s} bad={c:<6d} max|.|={mx:.6g}{flag}")
            if first_bad is not None:
                print(f"  [locate] first op with inf/NaN: {first_bad[0]} (bad={first_bad[1]}, max|.|={first_bad[2]:.6g})")
            else:
                print("  [note] intermediate activations all finite but loss is non-finite -> likely fused CE (internal logits overflow) or aux_loss")
            continue
        if args.no_backward:
            continue
        restore_rng(rng)
        bad, bwd_events = grad_report(model, x, y, anomaly=args.anomaly)
        if bad:
            grad_nan_micros.append(k)
            print(f"  backward: {len(bad)} params with non-finite grads:")
            for n in bad[:15]:
                print(f"    grad NaN: {n}")
            if bwd_events:
                print(f"  [locate] first non-finite grad in backward execution order: {bwd_events[0]}")
        else:
            print("  backward: all grads finite")

    # full-window accumulated replay (matches training semantics)
    if len(samples) > 1 and not args.no_backward and ok_weights:
        print("-" * 78)
        print("full-window accumulated forward+backward (matches training grad accumulation):")
        model.zero_grad(set_to_none=True)
        for k in sorted(samples.keys()):
            s = samples[k]
            restore_rng(rng)
            total, _, _ = compute_loss(model, s["x"].cuda(), s["y"].cuda(), use_ckpt=False)
            total.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        print(f"  params with non-finite grads after accumulation: {len(bad)}" + (f"  first few: {bad[:8]}" if bad else ""))

    # random-sample probe
    if args.probe_samples > 0 and ok_weights:
        print("-" * 78)
        print(f"random-sample probe (N={args.probe_samples}) -- does W_k blow up on any sample?")
        n_bad = 0
        for i in range(args.probe_samples):
            x = torch.randint(0, cfg.vocab_size, (1, 64)).cuda()
            y = torch.randint(0, cfg.vocab_size, (1, 64)).cuda()
            total, _, _ = compute_loss(model, x, y, use_ckpt=False)
            fin = bool(torch.isfinite(total).all())
            n_bad += (not fin)
            print(f"  probe {i}: loss={fmt(total)}  finite={fin}")
        verdict = "weights sick (blows up on any sample)" if n_bad == args.probe_samples else "sample-triggered (only specific samples)"
        print(f"  -> {n_bad}/{args.probe_samples} random samples non-finite: {verdict}")

    print("=" * 78)
    if loss_nan_micros:
        print(f"[conclusion] reproduced: micro-batch #{loss_nan_micros[0]} forward first shows non-finite loss"
              + (f", earliest non-finite op: {first_bad_tag}" if first_bad_tag else ""))
    elif grad_nan_micros:
        print(f"[conclusion] reproduced: forward loss finite, but micro-batch #{grad_nan_micros[0]} "
              f"backward first shows non-finite grads")
    elif not ok_weights:
        print("[conclusion] not reproduced: weights were NaN-contaminated at dump time (optimizer-corruption "
              "scenario); 'first' cannot be reproduced on the saved samples.")
    else:
        print("[conclusion] not reproduced: all saved samples are finite in forward/backward. Possible causes:")
        print("       1) optimizer-corruption scenario (finite grads -> NaN weights); check the optimizer step separately;")
        print("       2) non-deterministic kernels (try a different GPU / rerun a few times);")
        print("       3) the trigger needs longer context / multi-window accumulation, outside the saved window.")


if __name__ == "__main__":
    main()
