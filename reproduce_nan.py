#!/usr/bin/env python
"""reproduce_nan.py —— 复现 NanTrap 保存的"首次 NaN 临界点"。

用法:
    python reproduce_nan.py <dump_dir> [--no-backward] [--anomaly] [--probe-samples N]

流程:
  1. 加载 dump: W_k 权重 / 窗口内样本 / RNG / 元信息。
  2. 校验"正常模型"前提: 权重必须全部有限（否则说明 dump 时权重已被污染，
     属于 optimizer 污染场景或检测点过晚，无法复现"首次"）。
  3. 逐条样本原样重放 forward（与训练完全相同的调用方式 / 相同 bf16 / 相同 flag）。
     - 若 loss/aux 非有限 -> 用 forward hooks 定位第一个出现 inf/NaN 的算子
       （hooks 挂在所有子模块上，与具体架构无关），并打印各模块激活的 max|.| 轨迹。
  4. 若 forward 全部有限 -> 重放 forward+backward:
     - 列出非有限梯度所属参数；用 backward hooks 找 backward 执行顺序中
       第一个非有限梯度的层（= NaN 产生的位置）；--anomaly 开启 autograd
       异常检测，让 torch 直接报出产生 NaN 的算子。
  5. 整窗口累积 forward+backward 复现（与训练梯度累积语义对齐）。
  6. --probe-samples N: 用随机样本探测 W_k，区分"该样本触发"还是
     "模型已病态（任意样本都炸）"。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------- dump 加载
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


# ---------------------------------------------------------------- 定位工具
def _register_all_forward_hooks(root, record):
    """在 root 的所有子模块（含 root 自身）上挂 forward hook，按 forward 执行
    顺序把 (name, bad_count, max_abs) 追加到 record。与具体架构无关。"""
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
    """无梯度 forward + 全子模块 hooks：返回 (events, first_bad)。events 按
    forward 执行顺序记录 (name, bad_count, max_abs)。"""
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
    """forward+backward：返回 (bad_param_names, bwd_events)。bwd_events 按
    backward 执行顺序记录第一个出现非有限梯度的层（= NaN 产生处）。"""
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


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--no-backward", action="store_true", help="只重放 forward，不做 backward")
    ap.add_argument("--anomaly", action="store_true",
                    help="backward 时开启 torch autograd NaN 异常检测（慢，但直接报出 NaN 算子）")
    ap.add_argument("--probe-samples", type=int, default=0,
                    help="用 N 条随机样本探测 W_k，判断是样本触发还是模型已病态")
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
        print(f"[ok] 权重全部有限（{n_params} 参数）—— '正常模型'前提成立")
    else:
        print("[!!!] 保存的权重并非全部有限 —— dump 时刻权重已被 NaN 污染。")
        print("      (通常意味着属于 optimizer 污染场景，或检测点设置过晚。)")
        print("      该 dump 无法复现'首次 NaN'，下面的结果仅供参考。")

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
            # 打印每层的 max|.| 轨迹（只看主干模块，避免 aux 细节刷屏）
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
                print(f"  [定位] 第一个出现 inf/NaN 的算子: {first_bad[0]} (bad={first_bad[1]}, max|.|={first_bad[2]:.6g})")
            else:
                print("  [注意] 中间激活均有限但 loss 非有限 -> 问题可能在 fused CE(内部 logits 溢出) 或 aux_loss")
            continue
        if args.no_backward:
            continue
        restore_rng(rng)
        bad, bwd_events = grad_report(model, x, y, anomaly=args.anomaly)
        if bad:
            grad_nan_micros.append(k)
            print(f"  backward: {len(bad)} 个参数梯度非有限:")
            for n in bad[:15]:
                print(f"    grad NaN: {n}")
            if bwd_events:
                print(f"  [定位] backward 执行顺序中第一个非有限梯度: {bwd_events[0]}")
        else:
            print(f"  backward: 梯度全部有限")

    # 整窗口累积复现（对齐训练语义）
    if len(samples) > 1 and not args.no_backward and ok_weights:
        print("-" * 78)
        print("整窗口累积 forward+backward（与训练梯度累积一致）:")
        model.zero_grad(set_to_none=True)
        for k in sorted(samples.keys()):
            s = samples[k]
            restore_rng(rng)
            total, _, _ = compute_loss(model, s["x"].cuda(), s["y"].cuda(), use_ckpt=False)
            total.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        print(f"  累积后非有限梯度参数数: {len(bad)}" + (f"  前几个: {bad[:8]}" if bad else ""))

    # 随机样本探测
    if args.probe_samples > 0 and ok_weights:
        print("-" * 78)
        print(f"随机样本探测 (N={args.probe_samples}) —— 判断 W_k 是否'任意样本都炸':")
        n_bad = 0
        for i in range(args.probe_samples):
            x = torch.randint(0, cfg.vocab_size, (1, 64)).cuda()
            y = torch.randint(0, cfg.vocab_size, (1, 64)).cuda()
            total, _, _ = compute_loss(model, x, y, use_ckpt=False)
            fin = bool(torch.isfinite(total).all())
            n_bad += (not fin)
            print(f"  probe {i}: loss={fmt(total)}  finite={fin}")
        verdict = "权重已病态(任意样本都炸)" if n_bad == args.probe_samples else "样本触发(仅特定样本炸)"
        print(f"  -> {n_bad}/{args.probe_samples} 条随机样本触发非有限: {verdict}")

    print("=" * 78)
    if loss_nan_micros:
        print(f"[结论] 复现成功：micro-batch #{loss_nan_micros[0]} 的 forward 首次出现非有限 loss"
              + (f"，最早的非有限算子: {first_bad_tag}" if first_bad_tag else ""))
    elif grad_nan_micros:
        print(f"[结论] 复现成功：forward loss 有限，但 micro-batch #{grad_nan_micros[0]} 的 "
              f"backward 首次出现非有限梯度")
    elif not ok_weights:
        print("[结论] 复现失败：dump 时权重已被 NaN 污染（optimizer 污染场景），"
              "无法在保存的样本上复现'首次'。")
    else:
        print("[结论] 复现失败：dump 中所有样本 forward/backward 均有限。可能原因：")
        print("       1) 属于 optimizer 污染场景（有限梯度 -> NaN 权重），需另查 optimizer step；")
        print("       2) 非确定性 kernel（换 GPU/重跑几次试试）；")
        print("       3) 触发条件依赖更长的上下文/多窗累积，不在本次保存的窗口内。")


if __name__ == "__main__":
    main()
