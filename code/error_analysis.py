"""Class-wise error analysis from the saved per-checkpoint outputs: confusion matrices, per-class metrics, top confused pairs, examples.

  python code/error_analysis.py --split validation      development on validation outputs (examples drawn from validation images)
  python code/error_analysis.py --split test            the test outputs; the example figure displays test images (logged)

Models: Proposed, ConvNeXt-Tiny, EfficientNetV2-S averaged over seeds 42-44; ResNet50, DenseNet121, MobileNetV2 (seed 42).
Intervals resample stones (code/seedstats.py). Examples: a fixed-seed random draw of misclassified images of each of the top
confused pairs of the Proposed model, shown with ConvNeXt-Tiny's answer for the same image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedstats as ss  # noqa: E402
from common import ROOT, load_config  # noqa: E402
from final_tables import MAIN, Store  # noqa: E402

MODELS = list(MAIN) + ["ResNet50", "DenseNet121", "MobileNetV2"]
EX_SEED = 20261105  # fixed seed of the example draw


def confusion(y: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def prf(cm: np.ndarray) -> np.ndarray:
    """Rows: precision, recall, F1 per class."""
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return np.stack([prec, rec, f1])


def rids_of(st: Store, model: str) -> list[str]:
    return st.rids(model) if model in MAIN else [r for r in [f"{model}_s42"] if r in st.data]


def class_metrics(st: Store, model: str) -> pd.DataFrame:
    rids = rids_of(st, model)
    k = st.k
    point = np.mean([prf(confusion(st.y, st.pred(r, "clean"), k)) for r in rids], axis=0)
    per_seed = np.array([prf(confusion(st.y, st.pred(r, "clean"), k)) for r in rids])  # seeds x 3 x k
    boot = np.empty((len(st.draws), 3, k))
    for i, ii in enumerate(st.draws):
        boot[i] = np.mean([prf(confusion(st.y[ii], st.pred(r, "clean")[ii], k)) for r in rids], axis=0)
    rows = []
    for c in range(k):
        for j, name in enumerate(("precision", "recall", "f1")):
            lo, hi = ss.interval(boot[:, j, c])
            rows.append({"model": model, "class": st.classes[c], "metric": name, "value": point[j, c], "ci_lo": lo, "ci_hi": hi,
                         "sd_across_seeds": float(per_seed[:, j, c].std(ddof=1)) if len(rids) > 1 else float("nan")})
    return pd.DataFrame(rows)


def top_pairs(st: Store, model: str, n: int = 10) -> pd.DataFrame:
    rids = rids_of(st, model)
    k = st.k
    cms = [confusion(st.y, st.pred(r, "clean"), k) for r in rids]
    mean = np.mean(cms, axis=0)
    rows = []
    for i in range(k):
        for j in range(k):
            if i == j or mean[i, j] == 0:
                continue
            stones = np.mean([len(set(st.groups[(st.y == i) & (st.pred(r, "clean") == j)])) for r in rids])
            rows.append({"model": model, "true": st.classes[i], "predicted": st.classes[j], "images_mean_over_seeds": float(mean[i, j]),
                         "stones_mean_over_seeds": float(stones), "share_of_true_class": float(mean[i, j] / max(mean[i].sum(), 1))})
    return pd.DataFrame(rows).sort_values("images_mean_over_seeds", ascending=False).head(n)


def confusion_figure(st: Store, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5))
    for ax, m in zip(axes.ravel(), MODELS):
        rids = rids_of(st, m)
        if not rids:
            ax.axis("off")
            continue
        cm = np.mean([confusion(st.y, st.pred(r, "clean"), st.k) for r in rids], axis=0)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(st.k):
            for j in range(st.k):
                if cm[i, j] >= 0.5:
                    ax.text(j, i, f"{cm[i, j]:.0f}" if cm[i, j] == int(cm[i, j]) else f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if norm[i, j] > 0.55 else "#1a2430")
        ax.set_xticks(range(st.k))
        ax.set_yticks(range(st.k))
        ax.set_xticklabels(st.classes, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(st.classes, fontsize=8)
        ax.set_title(f"{m}  ({len(rids)} seed{'s' if len(rids) > 1 else ''}, mean counts)", fontsize=10, loc="left")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle(f"Confusion matrices ({st.split} set); colour = row-normalised, numbers = images", x=0.02, ha="left", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def example_figure(cfg: dict, st: Store, pairs: pd.DataFrame, path: Path, per_pair: int = 4, n_pairs: int = 3) -> None:
    """Fixed-seed random misclassified examples of the top confused pairs of the Proposed model (seed 42), with ConvNeXt-Tiny's answer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data import load_cache_array
    from evaluate import log_test_access

    split = st.split
    x = load_cache_array(cfg, split, allow_test=(split == "test"))
    if split == "test":
        log_test_access({"model": "Proposed_s42 + ConvNeXt-Tiny_s42", "split": "test", "purpose": "error-example figure: display of test images, "
                         "after all analyses were frozen; no analysis reads this"})
    pr, cv = st.pred("Proposed_s42", "clean"), st.pred("ConvNeXt-Tiny_s42", "clean")
    rng = np.random.default_rng(EX_SEED)
    top = pairs[pairs.model == "Proposed"].head(n_pairs)
    fig, axes = plt.subplots(len(top), per_pair, figsize=(3.1 * per_pair, 3.5 * len(top)), squeeze=False)
    rows = []
    for r, (_, p) in enumerate(top.iterrows()):
        i, j = st.classes.index(p["true"]), st.classes.index(p["predicted"])
        cand = np.flatnonzero((st.y == i) & (pr == j))
        stones = pd.Series(cand).groupby(st.groups[cand]).first().values  # one image per stone
        pick = rng.choice(stones, size=min(per_pair, len(stones)), replace=False)
        for c in range(per_pair):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(pick):
                idx = int(pick[c])
                ax.imshow(np.array(x[idx]))
                ax.set_title(f"true {st.classes[i]}\nProposed: {st.classes[pr[idx]]}\nConvNeXt-Tiny: {st.classes[cv[idx]]}", fontsize=8.5, loc="left")
                rows.append({"pair": f"{p['true']} -> {p['predicted']}", "index": idx, "group": st.groups[idx], "proposed": st.classes[pr[idx]],
                             "convnext": st.classes[cv[idx]]})
    fig.suptitle(f"Random misclassified examples ({split} set, fixed-seed draw, one image per stone)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)


def build(split: str, examples: bool = True) -> Path:
    cfg = load_config()
    st = Store(split)
    out = ROOT / "results" / ("error_analysis" if split == "test" else "error_analysis_validation")
    out.mkdir(parents=True, exist_ok=True)
    present = [m for m in MODELS if rids_of(st, m)]
    pd.concat([class_metrics(st, m) for m in present]).to_csv(out / "class_metrics.csv", index=False)
    pairs = pd.concat([top_pairs(st, m) for m in present])
    pairs.to_csv(out / "top_confused_pairs.csv", index=False)
    for m in present:
        cm = np.mean([confusion(st.y, st.pred(r, "clean"), st.k) for r in rids_of(st, m)], axis=0)
        pd.DataFrame(cm, index=st.classes, columns=st.classes).round(2).to_csv(out / f"confusion_{m}.csv")
    fig_dir = ROOT / "figures"
    confusion_figure(st, fig_dir / ("confusion_matrices.png" if split == "test" else "confusion_matrices_validation.png"))
    if examples and "Proposed_s42" in st.data and "ConvNeXt-Tiny_s42" in st.data:
        example_figure(cfg, st, pairs, fig_dir / ("error_examples.png" if split == "test" else "error_examples_validation.png"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args()
    print("wrote", build(a.split, not a.no_examples))
