"""Robustness study (Deliverable 6): how much does each model lose under controlled image perturbations?

Perturbations are applied to the cached 256 px uint8 crop (what training saw before the resize), then the unchanged
evaluation transform. Whole image, backdrop included. Everything is deterministic.

  python code/robustness.py sheet          contact sheet of one VALIDATION stone under every condition
  python code/robustness.py val            development run on the validation split (never touches test)
  python code/robustness.py final --test   the one test-set run; predictions are saved, access is logged
  python code/robustness.py figure         redraw the figure from the saved results

Pre-stated reading rules (fixed before any number was seen): a model is robust to a condition if its macro-F1 drop is
within 2 points; the proposed model is more fragile than a baseline only if the paired stone-level 95% interval of
(proposed drop - baseline drop) lies entirely below zero.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

log = logging.getLogger("robustness")
SEED = 42
MODELS = ["ResNet50", "DenseNet121", "MobileNetV2", "EfficientNetV2-S", "ConvNeXt-Tiny", "Proposed"]
FORMAL = ["ConvNeXt-Tiny", "EfficientNetV2-S"]  # the only baselines compared formally with the proposed model


# ----------------------------------------------------------------------------- perturbations
def _clip(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def brightness(x: np.ndarray, f: float) -> np.ndarray:
    return _clip(x.astype(np.float32) * f)


def contrast(x: np.ndarray, f: float) -> np.ndarray:
    """Blend with each image's mean grey level (PIL ImageEnhance.Contrast semantics)."""
    xf = x.astype(np.float32)
    grey = (0.299 * xf[..., 0] + 0.587 * xf[..., 1] + 0.114 * xf[..., 2]).mean(axis=(1, 2), keepdims=True)[..., None]
    return _clip(grey + f * (xf - grey))


def gaussian_noise(x: np.ndarray, sigma: float, seed: int = SEED) -> np.ndarray:
    """Additive noise, one generator per image (seed + index), so levels share one noise pattern scaled by sigma."""
    out = np.empty_like(x)
    for i in range(len(x)):
        n = np.random.default_rng(seed + i).standard_normal(x[i].shape).astype(np.float32)
        out[i] = _clip(x[i].astype(np.float32) + sigma * n)
    return out


def rotate(x: np.ndarray, deg: float) -> np.ndarray:
    h, w = x.shape[1:3]
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), deg, 1.0)
    return np.stack([cv2.warpAffine(im, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for im in x])


def hflip(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x[:, :, ::-1])


def blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([cv2.GaussianBlur(im, (0, 0), sigma) for im in x])


# condition label -> (family, level, core?, function returning a LIST of views of the batch)
def conditions() -> dict[str, dict]:
    c: dict[str, dict] = {}
    for f in (0.6, 0.8, 1.2, 1.4):
        c[f"Brightness {int(round((f - 1) * 100)):+d}%".replace("-", "−")] = dict(family="brightness", level=f, core=f in (0.8, 1.2), fn=lambda x, f=f: [brightness(x, f)])
    for f in (0.6, 0.8, 1.2, 1.4):
        c[f"Contrast {int(round((f - 1) * 100)):+d}%".replace("-", "−")] = dict(family="contrast", level=f, core=f in (0.8, 1.2), fn=lambda x, f=f: [contrast(x, f)])
    for s in (5, 10, 20):
        c[f"Noise σ={s}"] = dict(family="noise", level=s, core=s == 10, fn=lambda x, s=s: [gaussian_noise(x, s)])
    for d in (5, 10, 20):
        c[f"Rotation ±{d}°"] = dict(family="rotation", level=d, core=d == 10, fn=lambda x, d=d: [rotate(x, d), rotate(x, -d)])
    c["Horizontal flip"] = dict(family="flip", level=1, core=True, fn=lambda x: [hflip(x)])
    for s in (0.5, 1, 2):
        c[f"Blur σ={s:g}"] = dict(family="blur", level=s, core=s == 1, fn=lambda x, s=s: [blur(x, s)])
    return c


COND = conditions()
CORE = [k for k, v in COND.items() if v["core"]]
FAMILIES = ["brightness", "contrast", "noise", "rotation", "blur"]


# ---------------------------------------------------------------------------------- statistics
def _f1(y, p, k):
    from background_analysis import macro_f1

    return macro_f1(y, p, k)


def _stone_index(groups: np.ndarray) -> list[np.ndarray]:
    codes, uniq = pd.factorize(groups)
    return [np.flatnonzero(codes == g) for g in range(len(uniq))]


def paired_delta(y, p_clean, p_cond, groups, k, n=1000, seed=SEED) -> tuple[float, float, float]:
    """macro-F1(condition) - macro-F1(clean) and its paired 95% interval, resampling whole stones. All arrays share one
    length (clean predictions repeated once per view)."""
    idx = _stone_index(groups)
    rng = np.random.default_rng(seed)
    base = _f1(y, p_cond, k) - _f1(y, p_clean, k)
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.integers(0, len(idx), len(idx))])
        vals.append(_f1(y[ii], p_cond[ii], k) - _f1(y[ii], p_clean[ii], k))
    return float(base), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired_drop_difference(y, cleanA, condA, cleanB, condB, groups, k, n=1000, seed=SEED) -> tuple[float, float, float]:
    """(drop of A) - (drop of B) on the same resampled stones. A negative interval means A loses more than B."""
    idx = _stone_index(groups)
    rng = np.random.default_rng(seed)
    d = lambda ii: (_f1(y[ii], condA[ii], k) - _f1(y[ii], cleanA[ii], k)) - (_f1(y[ii], condB[ii], k) - _f1(y[ii], cleanB[ii], k))
    base = d(np.arange(len(y)))
    vals = [d(np.concatenate([idx[g] for g in rng.integers(0, len(idx), len(idx))])) for _ in range(n)]
    return float(base), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ---------------------------------------------------------------------------------- evaluation
def load_model(cfg: dict, name: str, device):
    import torch

    from baselines import build_model, model_dir

    ck = ROOT / "models" / model_dir(cfg, name) / "best.pt"
    m = build_model(cfg, name, pretrained=False)
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
    return m.to(device).eval(), ck


def predict_views(cfg: dict, model, x_views: list[np.ndarray], y: np.ndarray, device):
    """Logits for each view, concatenated view after view."""
    from baselines import normalisation
    from data import ArrayDataset, make_loader
    from train import predict

    mean, std = normalisation(model)
    out = []
    for xv in x_views:
        lg, yy = predict(model, make_loader(ArrayDataset(cfg, xv, y, mean, std), 64, False, 0), device)
        assert (yy == y).all()
        out.append(lg)
    return np.concatenate(out)


def run_split(cfg: dict, split: str, models: list[str], clean_override: dict | None = None, log_access: bool = False):
    """Predictions of every model for the clean images and all conditions. `clean_override[name]` supplies saved clean
    logits (test split, baselines) so they are not read again. Returns (preds, y, groups, k)."""
    from data import load_cache_array
    from evaluate import log_test_access, sha256_file
    from train import get_device

    device = get_device()
    meta = pd.read_csv(ROOT / cfg["benchmark"]["dir"] / f"{split}.csv")
    y, groups = meta["class_idx"].to_numpy(), meta["group_id"].to_numpy()
    k = int(meta["class_idx"].nunique())
    x = np.array(load_cache_array(cfg, split, allow_test=(split == "test")))
    loaded = {name: load_model(cfg, name, device) for name in models}
    preds: dict[str, dict[str, np.ndarray]] = {name: {} for name in models}
    for name, (model, ck) in loaded.items():
        if clean_override and name in clean_override:
            preds[name]["clean"] = clean_override[name].argmax(1)
        else:
            lg = predict_views(cfg, model, [x], y, device)
            preds[name]["clean"] = lg.argmax(1)
            if split == "test":
                np.savez(ROOT / "results" / "test_predictions" / f"{name}.npz", logits=lg, y=y, sha=sha256_file(ck))
                log_test_access({"model": name, "split": "test", "ckpt_sha256": sha256_file(ck),
                                 "purpose": "final evaluation, clean images (first test read of this model)"})
    for cname, c in COND.items():  # views are built once per condition and dropped before the next one
        views = c["fn"](x)
        for name, (model, ck) in loaded.items():
            preds[name][cname] = predict_views(cfg, model, views, y, device).argmax(1)
        log.info("%s done (%s)", cname, split)
        del views
    if split == "test" and log_access:
        for name, (model, ck) in loaded.items():
            log_test_access({"model": name, "split": "test", "ckpt_sha256": sha256_file(ck), "conditions": len(COND),
                             "purpose": "robustness: 18 perturbation conditions"})
    return preds, y, groups, k


def tables(preds: dict, y, groups, k, classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-form results for every model and condition, and the paired comparisons proposed vs the formal baselines."""
    rows = []
    for name, P in preds.items():
        clean = P["clean"]
        for cname, c in [("Original", None)] + list(COND.items()):
            p = clean if c is None else P[cname]
            nviews = len(p) // len(y)
            yy, gg, pc = np.tile(y, nviews), np.tile(groups, nviews), np.tile(clean, nviews)
            from background_analysis import stone_bootstrap

            lo, hi = stone_bootstrap(yy, p, gg, k)
            d, dlo, dhi = (0.0, 0.0, 0.0) if c is None else paired_delta(yy, pc, p, gg, k)
            r = dict(model=name, condition=cname, family="clean" if c is None else c["family"], level=None if c is None else c["level"],
                     core=bool(c is None or c["core"]), accuracy=float((p == yy).mean()), macro_f1=_f1(yy, p, k), f1_ci_lo=lo, f1_ci_hi=hi,
                     delta_f1=d, delta_ci_lo=dlo, delta_ci_hi=dhi, flip_rate=float((p != pc).mean()))
            for ci, cl in enumerate(classes):
                r[f"recall_{cl}"] = float((p[yy == ci] == ci).mean())
            rows.append(r)
    long = pd.DataFrame(rows)
    comp = []
    if "Proposed" in preds:
        for base in FORMAL:
            for cname, c in COND.items():
                nviews = len(preds["Proposed"][cname]) // len(y)
                yy, gg = np.tile(y, nviews), np.tile(groups, nviews)
                diff, lo, hi = paired_drop_difference(yy, np.tile(preds["Proposed"]["clean"], nviews), preds["Proposed"][cname],
                                                      np.tile(preds[base]["clean"], nviews), preds[base][cname], gg, k)
                comp.append(dict(baseline=base, condition=cname, core=c["core"], diff_in_drop=diff, ci_lo=lo, ci_hi=hi,
                                 proposed_more_fragile=bool(hi < 0), proposed_more_robust=bool(lo > 0)))
    return long, pd.DataFrame(comp)


# -------------------------------------------------------------------------------------- figures
def figure(cfg: dict, long: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ResNet50": "#2a78d6", "DenseNet121": "#eb6834", "MobileNetV2": "#1baf7a", "EfficientNetV2-S": "#eda100",
              "ConvNeXt-Tiny": "#e87ba4", "Proposed": "#4a3aa7"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    conds = [c for c in CORE]
    for i, cname in enumerate(conds):
        for j, m in enumerate(MODELS):
            q = long[(long.model == m) & (long.condition == cname)]
            if q.empty:
                continue
            q = q.iloc[0]
            y = i + (j - 2.5) * 0.12
            big = m == "Proposed"
            if big:
                ax.plot([100 * q.delta_ci_lo, 100 * q.delta_ci_hi], [y, y], color=colors[m], lw=2.2, alpha=.8)
            ax.plot([100 * q.delta_f1], [y], "o", color=colors[m], ms=9 if big else 6, mec="white", mew=1, label=m if i == 0 else None)
    ax.axvline(0, color="#8a96a3", lw=1)
    ax.axvspan(-2, 0, color="#dff2e8", alpha=.5, zorder=0)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.invert_yaxis()
    ax.set_xlabel("change in test macro-F1 (points); green band = within 2 points;\nwhiskers: proposed model, 95% paired interval")
    ax.set_title("Core conditions", loc="left", fontsize=11)
    ax.grid(axis="x", color="#e3e8ee")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    sub = gs[1].subgridspec(2, 3, wspace=0.35, hspace=0.55)
    for n, fam in enumerate(FAMILIES):
        a = fig.add_subplot(sub[n // 3, n % 3])
        for m in MODELS:
            q = long[(long.model == m) & (long.family == fam)].sort_values("level")
            clean = long[(long.model == m) & (long.condition == "Original")].iloc[0].macro_f1
            xs = list(q.level)
            ys = list(100 * q.macro_f1)
            if fam in ("brightness", "contrast"):
                xs.append(1.0); ys.append(100 * clean)
            elif fam in ("noise", "rotation", "blur"):
                xs.insert(0, 0); ys.insert(0, 100 * clean)
            o = np.argsort(xs)
            a.plot(np.array(xs)[o], np.array(ys)[o], color=colors[m], lw=2.6 if m == "Proposed" else 1.5, marker="o", ms=3.5)
        a.set_title(fam.capitalize(), loc="left", fontsize=10)
        a.grid(color="#e3e8ee")
        if n % 3 == 0:
            a.set_ylabel("macro-F1 %")
    fig.suptitle("Robustness of the proposed model against five baselines (test set, one run)", x=0.02, ha="left", fontsize=13)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def contact_sheet(cfg: dict, path: Path) -> None:
    """One VALIDATION stone under every condition."""
    from PIL import Image, ImageDraw

    from data import load_cache_array

    meta = pd.read_csv(ROOT / cfg["benchmark"]["dir"] / "validation.csv")
    x = np.array(load_cache_array(cfg, "validation"))
    picks = [int(meta.index[meta.label == c][3]) for c in ("Ruby", "Blue Zircon")]
    names = ["Original"] + list(COND)
    T, cols = 150, 5
    rows = math.ceil(len(names) / cols) * len(picks)
    sheet = Image.new("RGB", (cols * T, rows * (T + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for pi, i in enumerate(picks):
        one = x[i:i + 1]
        for ni, nm in enumerate(names):
            arr = one if nm == "Original" else COND[nm]["fn"](one)[0]
            im = Image.fromarray(arr[0]).resize((T, T))
            r, c = divmod(ni, cols)
            yy = (pi * math.ceil(len(names) / cols) + r) * (T + 16)
            sheet.paste(im, (c * T, yy + 16))
            d.text((c * T + 2, yy + 2), nm, fill="black")
    path.parent.mkdir(exist_ok=True)
    sheet.save(path, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sheet", "val", "final", "figure"])
    ap.add_argument("--test", action="store_true", help="required for `final`: confirms this reads the test set")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json

    classes = json.loads((ROOT / cfg["benchmark"]["dir"] / "classes.json").read_text())
    res = ROOT / "results"
    if a.stage == "sheet":
        contact_sheet(cfg, ROOT / "dataset" / "audit" / "robustness_check_validation.jpg")
        log.info("wrote the validation contact sheet")
    elif a.stage == "val":
        preds, y, groups, k = run_split(cfg, "validation", MODELS)
        long, comp = tables(preds, y, groups, k, classes)
        long.to_csv(res / "robustness_validation.csv", index=False)
        comp.to_csv(res / "robustness_comparisons_validation.csv", index=False)
        np.savez(res / "robustness_predictions_validation.npz", **{f"{m}|{c}": p for m, P in preds.items() for c, p in P.items()})
        print(long[long.core][["model", "condition", "accuracy", "macro_f1", "delta_f1", "flip_rate"]].round(4).to_string(index=False))
    elif a.stage == "final":
        if not a.test:
            raise SystemExit("`final` reads the test set. Re-run with --test to confirm.")
        pred_path = res / "robustness_predictions.npz"
        clean_override = {}
        for m in MODELS[:-1]:  # baselines: clean test predictions were saved at Deliverable 2, not read again
            clean_override[m] = np.load(res / "test_predictions" / f"{m}.npz")["logits"]
        if pred_path.exists():
            z = np.load(pred_path)
            preds: dict = {}
            for key in z.files:
                m, c = key.split("|")
                preds.setdefault(m, {})[c] = z[key]
            meta = pd.read_csv(ROOT / cfg["benchmark"]["dir"] / "test.csv")
            y, groups, k = meta["class_idx"].to_numpy(), meta["group_id"].to_numpy(), int(meta["class_idx"].nunique())
            log.info("rebuilding from saved predictions; the test set is not read")
        else:
            preds, y, groups, k = run_split(cfg, "test", MODELS, clean_override=clean_override, log_access=True)
            np.savez(pred_path, **{f"{m}|{c}": p for m, P in preds.items() for c, p in P.items()})
        long, comp = tables(preds, y, groups, k, classes)
        long.to_csv(res / "robustness_all_models.csv", index=False)
        comp.to_csv(res / "robustness_comparisons.csv", index=False)
        p = long[long.model == "Proposed"][["condition", "accuracy", "macro_f1", "f1_ci_lo", "f1_ci_hi", "delta_f1", "delta_ci_lo", "delta_ci_hi", "flip_rate"]]
        p.to_csv(res / "robustness.csv", index=False)
        figure(cfg, long, ROOT / "figures" / "robustness.png")
        print(p.round(4).to_string(index=False))
    elif a.stage == "figure":
        figure(cfg, pd.read_csv(res / "robustness_all_models.csv"), ROOT / "figures" / "robustness.png")


if __name__ == "__main__":
    main()
