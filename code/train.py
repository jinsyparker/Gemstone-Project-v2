"""Generic two-stage transfer-learning trainer (one recipe for every model; D3 to D5 reuse it).

Stage 1: backbone frozen, train the head.            (config benchmark.stage1)
Stage 2: unfreeze the top fraction of the backbone, much smaller LR, cosine decay with warm-up,
         early stopping on validation macro-F1.        (config benchmark.stage2)
The checkpoint kept is the one with the best VALIDATION macro-F1. The test split is never loaded here.

Usage: python code/train.py --model ResNet50 [--resume] [--smoke]
       python code/train.py --model EfficientNetV2-S --label-smoothing 0.1 --tag pilot_ls0.1 \
              --s1-epochs 6 --s2-epochs 6 --out-dir models/_pilot/ls0.1
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import (build_model, count_params, model_dir, normalisation, param_groups,  # noqa: E402
                       set_stage1, set_stage2, set_train_mode)
from common import ROOT, load_config, set_seed  # noqa: E402
from data import CachedDataset, class_names, make_loader  # noqa: E402

log = logging.getLogger("train")


def get_device() -> torch.device:
    """MPS when available; GEM_DEVICE=cpu forces CPU (used by the tests, which must be deterministic)."""
    forced = os.environ.get("GEM_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Logits (N, K) and labels (N,) for a loader, in loader order."""
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def val_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    """Plain cross-entropy (no label smoothing), accuracy and macro-F1."""
    lt = torch.from_numpy(logits)
    yt = torch.from_numpy(y)
    loss = float(nn.functional.cross_entropy(lt, yt))
    pred = logits.argmax(1)
    return {"loss": loss, "acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro"))}


def lr_lambda(total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def train_epoch(model, loader, opt, sched, criterion, device) -> tuple[float, float]:
    set_train_mode(model)
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        tot += float(loss.detach()) * len(y)
        correct += int((out.argmax(1) == y).sum())
        n += len(y)
    return tot / n, correct / n


def train_model(cfg: dict, model: nn.Module, out_dir: Path, hist_path: Path, seed: int = 42,
                label_smoothing: float | None = None, limit: int | None = None, s1_epochs: int | None = None,
                s2_epochs: int | None = None, resume: bool = False, name: str = "model",
                variant: str = "original") -> dict:
    """Train `model` with the benchmark recipe; returns the history dict."""
    b = cfg["benchmark"]
    s1, s2 = b["stage1"], b["stage2"]
    n1 = s1["epochs"] if s1_epochs is None else s1_epochs
    n2 = s2["max_epochs"] if s2_epochs is None else s2_epochs
    ls = b["label_smoothing"] if label_smoothing is None else label_smoothing
    device = get_device()
    model.to(device)
    mean, std = normalisation(model)
    set_seed(seed)

    tr_ds = CachedDataset(cfg, "train", mean, std, train=True, limit=limit, variant=variant)
    va_ds = CachedDataset(cfg, "validation", mean, std, train=False, limit=limit, variant=variant)
    va_loader = make_loader(va_ds, b["batch_size"], False, seed)
    steps_per_epoch = math.ceil(len(tr_ds) / b["batch_size"])
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist = {"name": name, "seed": seed, "label_smoothing": ls, "stage1_epochs": n1, "stage2_max_epochs": n2,
            "params": count_params(model), "epochs": []}
    state = {"best_f1": -1.0, "best_epoch": -1, "s2_best": -1.0, "s2_bad": 0, "start": 0}
    opt = sched = None
    last = out_dir / "last.pt"

    def enter_stage(g: int):
        nonlocal opt, sched
        if g < n1:
            set_stage1(model)
            opt = torch.optim.AdamW(param_groups(model, s1["lr_head"], None, b["weight_decay"]["head"],
                                                 b["weight_decay"]["backbone"]))
            sched = None
        else:
            info = set_stage2(model, s2["unfreeze_fraction"])
            hist["stage2_unfreeze"] = info
            opt = torch.optim.AdamW(param_groups(model, s2["lr_head"], s2["lr_backbone"], b["weight_decay"]["head"],
                                                 b["weight_decay"]["backbone"]))
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda(n2 * steps_per_epoch, s2["warmup_epochs"] * steps_per_epoch))

    if resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        hist, state = ck["hist"], ck["state"]
        enter_stage(state["start"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        log.info("resumed %s at epoch %d", name, state["start"])

    cur_stage = (1 if state["start"] < n1 else 2) if opt is not None else None  # set when resumed
    for g in range(state["start"], n1 + n2):
        stage = 1 if g < n1 else 2
        if cur_stage != stage:  # first epoch, or the switch from stage 1 to stage 2
            enter_stage(g)
            cur_stage = stage
        torch.manual_seed(seed * 1000 + g)  # per-epoch seed: identical order with or without resume
        tr_loader = make_loader(tr_ds, b["batch_size"], True, seed * 1000 + g)
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, opt, sched, criterion, device)
        logits, y = predict(model, va_loader, device)
        m = val_metrics(logits, y)
        row = {"epoch": g + 1, "stage": stage, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": m["loss"], "val_acc": m["acc"], "val_macro_f1": m["macro_f1"],
               "lr": [gp["lr"] for gp in opt.param_groups], "seconds": round(time.time() - t0, 1)}
        hist["epochs"].append(row)
        log.info("%s ep%02d s%d train %.3f/%.3f val %.3f acc %.3f F1 %.4f (%.0fs)", name, g + 1, stage, tr_loss,
                 tr_acc, m["loss"], m["acc"], m["macro_f1"], row["seconds"])
        if m["macro_f1"] > state["best_f1"]:
            state.update(best_f1=m["macro_f1"], best_epoch=g + 1)
            torch.save({"state_dict": model.state_dict(), "epoch": g + 1, "val": m, "name": name,
                        "timm": cfg["benchmark"]["models"].get(name, {}).get("timm"), "seed": seed,
                        "classes": class_names(cfg)}, out_dir / "best.pt")
        stop = False
        if stage == 2:
            if m["macro_f1"] > state["s2_best"] + s2["min_delta"]:
                state.update(s2_best=m["macro_f1"], s2_bad=0)
            else:
                state["s2_bad"] += 1
                stop = state["s2_bad"] >= s2["patience"]
        state["start"] = g + 1
        hist.update(best_epoch=state["best_epoch"], best_val_macro_f1=state["best_f1"])
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None, "hist": hist, "state": state}, last)
        hist_path.write_text(json.dumps(hist, indent=1))
        if stop:
            hist["stopped_early_at"] = g + 1
            log.info("early stopping at epoch %d (best val macro-F1 %.4f at epoch %d)", g + 1, state["best_f1"],
                     state["best_epoch"])
            break
    hist["total_epochs"] = len(hist["epochs"])
    hist_path.write_text(json.dumps(hist, indent=1))
    return hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float)
    ap.add_argument("--s1-epochs", type=int)
    ap.add_argument("--s2-epochs", type=int)
    ap.add_argument("--tag", help="history/log name (default: model name)")
    ap.add_argument("--out-dir", help="checkpoint directory (default: models/<model dir>)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1+1 epochs, nothing written under models/")
    ap.add_argument("--variant", default="original", help="image version to train and validate on (background study)")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tag = a.tag or a.model
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "results" / "logs" / f"{tag}.log")])
    spec = {"timm": cfg["proposed"]["backbone"] + " + colour/texture branch", "dir": model_dir(cfg, a.model)} \
        if a.model == "Proposed" else cfg["benchmark"]["models"][a.model]
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "models" / spec["dir"]
    hist_path = ROOT / "results" / "baseline_history" / f"{tag}.json"
    if a.smoke:
        out_dir = ROOT / "results" / "_smoke" / spec["dir"]
        hist_path = ROOT / "results" / "_smoke" / f"{tag}.json"
    set_seed(a.seed)
    model = build_model(cfg, a.model, pretrained=True)
    log.info("%s | %s | %.1fM params | device %s | label_smoothing %s", a.model, spec["timm"],
             count_params(model) / 1e6, get_device(), a.label_smoothing)
    train_model(cfg, model, out_dir, hist_path, seed=a.seed, label_smoothing=a.label_smoothing,
                limit=64 if a.smoke else None, s1_epochs=1 if a.smoke else a.s1_epochs,
                s2_epochs=1 if a.smoke else a.s2_epochs, resume=a.resume, name=tag, variant=a.variant)


if __name__ == "__main__":
    main()
