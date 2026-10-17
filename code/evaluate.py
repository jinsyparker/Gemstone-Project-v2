"""Evaluation, benchmark table and figures.

  python code/evaluate.py --val              validation metrics of every trained model (free to run)
  python code/evaluate.py --final            test evaluation + results/benchmark.csv + figures

The test split is read ONLY here, only with --final, once per checkpoint: logits are saved to
results/test_predictions/<model>.npz and every access is appended to results/test_access_log.json.
Re-running --final rebuilds the table from the saved predictions without touching the test images.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import build_model, count_params, normalisation  # noqa: E402
from common import ROOT, load_config, set_seed  # noqa: E402
from data import CachedDataset, class_names, make_loader  # noqa: E402
from train import get_device, predict  # noqa: E402

log = logging.getLogger("evaluate")

# Reported by the user for the earlier paper: 11 classes, different split, train/validation only.
HISTORICAL = [
    ("ResNet50", 97.58, 95.44, 95.29, 95.28),
    ("EfficientNet50 (as reported)", 97.02, 97.18, 97.02, 97.02),
    ("MobileNetV2", 95.29, 95.44, 95.29, 95.28),
    ("VGG16", 97.65, 97.69, 97.65, 97.64),
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def full_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy, macro precision/recall/F1, weighted F1, plain cross-entropy (no label smoothing)."""
    pred = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "loss": float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y))),
    }


def load_trained(cfg: dict, name: str, device) -> tuple[nn.Module, Path]:
    ck_path = ROOT / "models" / cfg["benchmark"]["models"][name]["dir"] / "best.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = build_model(cfg, name, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck_path


def run_split(cfg: dict, model: nn.Module, split: str, device, allow_test: bool = False):
    mean, std = normalisation(model)
    ds = CachedDataset(cfg, split, mean, std, train=False, allow_test=allow_test)
    return predict(model, make_loader(ds, 64, False, 0), device)


@torch.no_grad()
def time_inference(model: nn.Module, device, size: int, batch: int, warmup: int = 20, iters: int = 200) -> float:
    """Median milliseconds per image for a forward pass at the given batch size."""
    model.eval()
    x = torch.randn(batch, 3, size, size, device=device)
    sync = torch.mps.synchronize if device.type == "mps" else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) / batch * 1000)


def access_log_path() -> Path:
    return ROOT / "results" / "test_access_log.json"


def log_test_access(entry: dict) -> None:
    p = access_log_path()
    rows = json.loads(p.read_text()) if p.exists() else []
    rows.append({**entry, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    p.write_text(json.dumps(rows, indent=1))


def val_row(cfg: dict, name: str, device) -> dict:
    model, ck_path = load_trained(cfg, name, device)
    logits, y = run_split(cfg, model, "validation", device)
    m = full_metrics(logits, y)
    hist = json.loads((ROOT / "results" / "baseline_history" / f"{name}.json").read_text())
    return {"model": name, "val_" + "accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_loss": m["loss"],
            "best_epoch": hist.get("best_epoch"), "epochs_run": len(hist["epochs"]), "params": count_params(model),
            "ckpt_sha256": sha256_file(ck_path)}


def background_control(cfg: dict) -> dict:
    """Random forest on backdrop colour, image size and brightness only (no stone pixels): a control."""
    from sklearn.ensemble import RandomForestClassifier

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocessing import read_qc

    q, _ = read_qc(cfg)
    feats = q.set_index("image_id")[["bg_r", "bg_g", "bg_b", "width", "height", "brightness"]]
    bd = ROOT / cfg["benchmark"]["dir"]
    tr, te = pd.read_csv(bd / "train.csv"), pd.read_csv(bd / "test.csv")
    rf = RandomForestClassifier(300, random_state=cfg["seed"], n_jobs=-1).fit(feats.loc[tr.image_id].values, tr.class_idx)
    proba = rf.predict_proba(feats.loc[te.image_id].values)
    full = np.zeros((len(te), len(class_names(cfg))))
    full[:, rf.classes_] = proba
    y = te.class_idx.to_numpy()
    pred = full.argmax(1)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "weighted_f1": float(f1_score(y, pred, average="weighted")),
            "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y, pred, average="macro")),
            "loss": float(log_loss(y, np.clip(full, 1e-6, 1), labels=list(range(full.shape[1]))))}


def build_table(cfg: dict, device, force: bool) -> pd.DataFrame:
    names = list(cfg["benchmark"]["models"])
    missing = [n for n in names if not (ROOT / "models" / cfg["benchmark"]["models"][n]["dir"] / "best.pt").exists()]
    if missing:
        raise SystemExit(f"cannot run --final: no trained checkpoint for {missing}")
    (ROOT / "results" / "test_predictions").mkdir(parents=True, exist_ok=True)
    size = cfg["benchmark"]["input_size"]
    rows = []
    for name in names:
        model, ck_path = load_trained(cfg, name, device)
        sha = sha256_file(ck_path)
        pred_path = ROOT / "results" / "test_predictions" / f"{name}.npz"
        if pred_path.exists() and str(np.load(pred_path)["sha"]) == sha and not force:
            z = np.load(pred_path)
            logits, y = z["logits"], z["y"]
            log.info("%s: test predictions reused (checkpoint unchanged), test set not read", name)
        else:
            logits, y = run_split(cfg, model, "test", device, allow_test=True)
            np.savez(pred_path, logits=logits, y=y, sha=sha)
            log_test_access({"model": name, "split": "test", "ckpt_sha256": sha, "purpose": "final benchmark"})
            log.info("%s: TEST evaluated and logged", name)
        m = full_metrics(logits, y)
        v = val_row(cfg, name, device)
        rows.append({
            "block": "controlled", "model": name, "params": count_params(model), "test_accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"], "precision": m["precision"],
            "recall": m["recall"], "test_loss": m["loss"],
            "inference_ms_per_image": round(time_inference(model, device, size, 1), 2),
            "inference_ms_per_image_batch32": round(time_inference(model, device, size, 32, iters=40), 2),
            "val_accuracy": v["val_accuracy"], "val_macro_f1": v["val_macro_f1"], "best_epoch": v["best_epoch"],
            "epochs_run": v["epochs_run"], "note": f"timm {cfg['benchmark']['models'][name]['timm']}"})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    ctrl_path = ROOT / "results" / "test_predictions" / "background_control.json"
    if ctrl_path.exists() and not force:
        c = json.loads(ctrl_path.read_text())
    else:
        c = background_control(cfg)
        ctrl_path.write_text(json.dumps(c))
        log_test_access({"model": "background_control_rf", "split": "test", "ckpt_sha256": None, "purpose": "control row"})
    rows.append({"block": "control", "model": "Background-only random forest", "params": None,
                 "test_accuracy": c["accuracy"], "macro_f1": c["macro_f1"], "weighted_f1": c["weighted_f1"],
                 "precision": c["precision"], "recall": c["recall"], "test_loss": c["loss"],
                 "note": "backdrop colour, image size, brightness only; no stone pixels"})
    for n, acc, pr, rc, f1 in HISTORICAL:
        rows.append({"block": "historical", "model": n, "test_accuracy": acc / 100, "precision": pr / 100,
                     "recall": rc / 100, "macro_f1": f1 / 100,
                     "note": "as reported; earlier paper, 11 classes, different split, train/validation only"})
    return pd.DataFrame(rows)


def figures(cfg: dict, table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ResNet50": "#2a78d6", "DenseNet121": "#eb6834", "MobileNetV2": "#1baf7a",
              "EfficientNetV2-S": "#eda100", "ConvNeXt-Tiny": "#e87ba4"}
    curve_colors = {**colors, "Proposed": "#4a3aa7"}   # the proposed model joins the curves once it is trained
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    s1 = cfg["benchmark"]["stage1"]["epochs"]
    for name in curve_colors:
        p = ROOT / "results" / "baseline_history" / f"{name}.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        ep = [r["epoch"] for r in h["epochs"]]
        lw = 2.6 if name == "Proposed" else 2
        ax[0].plot(ep, [r["val_macro_f1"] for r in h["epochs"]], color=curve_colors[name], lw=lw, label=name)
        be = h["best_epoch"]
        ax[0].plot([be], [h["epochs"][be - 1]["val_macro_f1"]], "o", color=curve_colors[name], ms=7, mec="white", mew=1.5)
        ax[1].plot(ep, [r["train_loss"] for r in h["epochs"]], color=curve_colors[name], lw=lw, label=name)
    for a, t in zip(ax, ("Validation macro-F1 (dot = best epoch)", "Training loss")):
        a.axvline(s1 + 0.5, color="#8a96a3", ls="--", lw=1)
        a.set_xlabel("epoch")
        a.set_title(t, loc="left", fontsize=11)
        a.grid(axis="y", color="#e3e8ee", lw=0.8)
        a.text(s1 + 0.7, a.get_ylim()[0], "stage 2: fine-tune", color="#5b6b7f", fontsize=8.5, va="bottom")
        a.text(s1 + 0.3, a.get_ylim()[0], "stage 1: head only", color="#5b6b7f", fontsize=8.5, va="bottom", ha="right")
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "training_curves.png", dpi=170)
    plt.close(fig)

    ctl = table[table.block == "controlled"].sort_values("macro_f1").reset_index(drop=True)
    ctrl = table[table.block == "control"]
    lo, hi = 0.94, 0.985
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.hlines(range(len(ctl)), lo, hi, color="#e3e8ee", lw=1, zorder=0)
    for y, r in ctl.iterrows():
        ax.plot([r.macro_f1], [y], "o", color=colors[r.model], ms=11, mec="white", mew=1.5, zorder=3)
        ax.text(r.macro_f1 + 0.0011, y, f"{r.macro_f1:.4f}", va="center", fontsize=10, fontweight="bold")
        ax.text(hi - 0.0005, y, f"{r.params / 1e6:.1f}M params", va="center", ha="right", fontsize=8.5, color="#5b6b7f")
    ax.set_yticks(range(len(ctl)))
    ax.set_yticklabels(ctl.model)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(ctl) - 0.4)
    ax.set_xlabel(f"test macro-F1   (axis starts at {lo}, not 0)")
    ax.set_title("Controlled benchmark: 10 classes, one recipe, single seed", loc="left", fontsize=11)
    note = "One test image is 0.13 points, so differences of a few tenths are within noise until the multi-seed runs (D9)."
    if len(ctrl):
        note = f"Background-only control: {float(ctrl.macro_f1.iloc[0]):.3f} (off this scale).\n" + note
    fig.text(0.01, 0.01, note, fontsize=8, color="#5b6b7f")
    ax.grid(axis="x", color="#e3e8ee", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ROOT / "figures" / "benchmark.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--val", action="store_true")
    g.add_argument("--final", action="store_true")
    g.add_argument("--figures", action="store_true", help="rebuild the figures from results/benchmark.csv only")
    ap.add_argument("--force", action="store_true", help="re-evaluate on the test split even if predictions are saved")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(cfg["seed"])
    device = get_device()
    if a.val:
        rows = []
        for n, spec in cfg["benchmark"]["models"].items():
            if (ROOT / "models" / spec["dir"] / "best.pt").exists():
                rows.append(val_row(cfg, n, device))
        df = pd.DataFrame(rows)
        (ROOT / "results").mkdir(exist_ok=True)
        df.to_csv(ROOT / "results" / "baseline_val.csv", index=False)
        print(df.to_string(index=False))
        return
    if a.figures:
        figures(cfg, pd.read_csv(ROOT / "results" / "benchmark.csv"))
        return
    table = build_table(cfg, device, a.force)
    table["params"] = table["params"].astype("Int64")  # whole numbers in the CSV
    (ROOT / "results").mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / "benchmark.csv", index=False, float_format="%.4f")
    figures(cfg, table)
    show = table[["block", "model", "params", "test_accuracy", "macro_f1", "weighted_f1", "precision", "recall",
                  "test_loss", "inference_ms_per_image"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
