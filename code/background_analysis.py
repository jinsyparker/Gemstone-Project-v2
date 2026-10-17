"""Background-dependence experiment (Deliverable 2b).

Do the frozen baselines rely on the studio backdrop? Stages:
  masks      gemstone masks for the cached 256 px crops           -> dataset/_cache/mask256/<split>.npy
  masksheet  overlay contact sheet of VALIDATION masks for the visual quality gate
  variants   original / fg_only / bg_only / swap_* image arrays    -> dataset/_cache/variants/<variant>_<split>.npy
  evaluate   frozen models on every variant (+ stone-level bootstrap intervals)

Test images are never viewed: their masks and variants are produced blind.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

log = logging.getLogger("background")
SPLITS = ("train", "validation", "test")


# --------------------------------------------------------------------------- masks
def _backdrop_model(img: np.ndarray, ring: int = 10) -> np.ndarray:
    """Quadratic surface per channel fitted to the outer ring of the crop: the smooth studio backdrop
    (lighting gradient included) extrapolated over the whole frame."""
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ringm = np.zeros((h, w), bool)
    ringm[:ring] = ringm[-ring:] = True
    ringm[:, :ring] = ringm[:, -ring:] = True
    x, y = (xx / w - 0.5), (yy / h - 0.5)
    basis = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1)
    A = basis[ringm]
    out = np.empty(img.shape, np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, img[..., c][ringm].astype(np.float32), rcond=None)
        out[..., c] = basis @ coef
    return out


def _largest_filled(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(m)


def _hull_repair(m: np.ndarray, max_growth: float = 1.25) -> np.ndarray:
    """Fill ragged notches of pale, translucent stones: replace the mask by its convex hull when the hull is
    at most `max_growth` times larger (mild notches only; real concavities such as hearts stay as they are)."""
    pts = cv2.findNonZero(m.astype(np.uint8))
    if pts is None:
        return m
    hull = cv2.convexHull(pts)
    filled = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool) if filled.sum() <= max_growth * max(1, m.sum()) else m


def compute_mask(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Gemstone mask for one 256x256x3 uint8 crop. GrabCut seeded from the distance to the fitted backdrop;
    falls back to a plain distance threshold. Returns (bool mask, info)."""
    h, w, _ = img.shape
    bg = _backdrop_model(img)
    dist = np.sqrt(((img.astype(np.float32) - bg) ** 2).sum(-1))
    dist = cv2.GaussianBlur(dist, (0, 0), 1.2)
    ring = np.zeros((h, w), bool)
    ring[:8] = ring[-8:] = True
    ring[:, :8] = ring[:, -8:] = True
    centre = np.zeros((h, w), bool)
    centre[h // 6: -h // 6, w // 6: -w // 6] = True
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[dist > 22] = cv2.GC_PR_FGD
    gc[(dist > 50) & centre] = cv2.GC_FGD
    gc[(dist < 9) | ring] = cv2.GC_BGD
    # Cast shadows are the backdrop's own colour, only darker; gemstones are coloured. Chroma shift small
    # and brightness lower than the fitted backdrop => definite background (overrides the foreground seeds).
    lum = img.astype(np.float32).mean(-1) + 1e-3
    bgl = bg.mean(-1) + 1e-3
    chroma_shift = np.linalg.norm(img / lum[..., None] - bg / bgl[..., None], axis=-1)
    ratio = cv2.GaussianBlur(lum / bgl, (0, 0), 1.0)
    shadow = (cv2.GaussianBlur(chroma_shift, (0, 0), 1.0) < 0.07) & (ratio < 0.96) & (ratio > 0.40)
    gc[shadow] = cv2.GC_BGD
    method = "grabcut"
    try:
        b, f = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(np.ascontiguousarray(img[..., ::-1]), gc, None, b, f, 5, cv2.GC_INIT_WITH_MASK)
        m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        m = _hull_repair(_largest_filled(m))
    except cv2.error:
        m = np.zeros((h, w), bool)
    area = m.mean()
    if not (0.03 <= area <= 0.70):  # implausible: fall back to the plain threshold
        method = "threshold"
        m = cv2.morphologyEx((dist > 28).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        m = _largest_filled(m)
        area = m.mean()
    touch = float((m[:3].any() or m[-3:].any() or m[:, :3].any() or m[:, -3:].any()))
    ok = bool(0.03 <= area <= 0.70 and not touch)
    return m, {"area": float(area), "method": method, "touches_border": bool(touch), "ok": ok}


def _mask_job(img: np.ndarray):
    m, info = compute_mask(img)
    return m.astype(np.uint8), info


def build_masks(cfg: dict, workers: int = 6) -> pd.DataFrame:
    size = cfg["benchmark"]["cache_size"]
    cdir = ROOT / cfg["audit"]["cache"] / f"img{size}"
    out = ROOT / cfg["audit"]["cache"] / f"mask{size}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        x = np.load(cdir / f"{split}.npy", mmap_mode="r")
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_mask_job, (np.array(a) for a in x), chunksize=16))
        np.save(out / f"{split}.npy", np.stack([r[0] for r in res]))
        df = pd.DataFrame([r[1] for r in res])
        df["split"] = split
        df["idx"] = range(len(df))
        rows.append(df)
        log.info("%s: %d masks, %d flagged, methods %s", split, len(df), int((~df.ok).sum()), df.method.value_counts().to_dict())
    info = pd.concat(rows, ignore_index=True)
    info.to_csv(out / "mask_info.csv", index=False)
    return info


def mask_sheet(cfg: dict, n_random: int = 100, n_worst: int = 20) -> Path:
    """Overlay contact sheet on VALIDATION images only: n_random random + n_worst by a simple badness score."""
    size = cfg["benchmark"]["cache_size"]
    x = np.load(ROOT / cfg["audit"]["cache"] / f"img{size}" / "validation.npy", mmap_mode="r")
    m = np.load(ROOT / cfg["audit"]["cache"] / f"mask{size}" / "validation.npy", mmap_mode="r")
    info = pd.read_csv(ROOT / cfg["audit"]["cache"] / f"mask{size}" / "mask_info.csv")
    info = info[info.split == "validation"].reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"])
    rand_idx = sorted(rng.choice(len(info), n_random, replace=False).tolist())
    bad = info.assign(score=(info.touches_border * 2 + (info.method == "threshold") + np.abs(info.area - 0.25))).sort_values("score", ascending=False)
    worst_idx = [i for i in bad.idx.tolist() if i not in set(rand_idx)][:n_worst]
    T, cols = 128, 12
    items = [("r", i) for i in rand_idx] + [("w", i) for i in worst_idx]
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * T, cols * T, 3), 255, np.uint8)
    for k, (kind, i) in enumerate(items):
        im = np.array(x[i])
        cnt, _ = cv2.findContours(np.array(m[i]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnt, -1, (255, 0, 0) if kind == "r" else (255, 0, 255), 2)
        t = cv2.resize(im, (T, T), interpolation=cv2.INTER_AREA)
        r, c = divmod(k, cols)
        sheet[r * T:(r + 1) * T, c * T:(c + 1) * T] = t
    out = ROOT / "dataset" / "audit" / "mask_check_validation.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


# ------------------------------------------------------------------------- variants
STUDIO_GREY = np.array([134, 134, 136], np.float32)  # the dominant backdrop of nine of the ten classes


def soft_alpha(mask: np.ndarray) -> np.ndarray:
    """Mask as a soft matte (about 1.5 px edge) so composites have no halo."""
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]


def composite(img: np.ndarray, alpha: np.ndarray, backdrop) -> np.ndarray:
    out = alpha * img.astype(np.float32) + (1 - alpha) * np.asarray(backdrop, np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def remove_stone(img: np.ndarray, mask: np.ndarray, grow: int = 6, reach: int = 45, tol: float = 9.0) -> np.ndarray:
    """Backdrop only. Removes the stone AND what it causes on the backdrop (cast shadow, coloured reflection):
    every pixel within `reach` px of the stone that differs from the fitted backdrop by more than `tol`.
    The hole (grown by `grow` px) is inpainted from the surrounding clean backdrop. Without this the
    stone's reflection survives and leaks its colour into the 'background-only' image."""
    bg = _backdrop_model(img)
    dist = cv2.GaussianBlur(np.sqrt(((img.astype(np.float32) - bg) ** 2).sum(-1)), (0, 0), 1.0)
    near = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
    ext = mask.astype(bool) | ((dist > tol) & near.astype(bool))
    ext = cv2.morphologyEx(ext.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    hole = cv2.dilate(ext, k) * 255
    return cv2.inpaint(np.ascontiguousarray(img), hole, 5, cv2.INPAINT_TELEA)


def _variant_job(args):
    img, mask = args
    a = soft_alpha(mask)
    return {"fg_only": composite(img, a, STUDIO_GREY), "bg_only": remove_stone(img, mask),
            "swap_black": composite(img, a, (0, 0, 0)), "swap_white": composite(img, a, (255, 255, 255))}


def build_variants(cfg: dict, workers: int = 6) -> dict:
    """fg_only / bg_only for every split; swap_* only for validation and test. Row-aligned with the CSVs."""
    size = cfg["benchmark"]["cache_size"]
    cdir = ROOT / cfg["audit"]["cache"]
    out = cdir / "variants"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        x = np.load(cdir / f"img{size}" / f"{split}.npy", mmap_mode="r")
        m = np.load(cdir / f"mask{size}" / f"{split}.npy", mmap_mode="r")
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_variant_job, ((np.array(x[i]), np.array(m[i])) for i in range(len(x))), chunksize=16))
        names = ["fg_only", "bg_only"] + ([] if split == "train" else ["swap_black", "swap_white"])
        for n in names:
            np.save(out / f"{n}_{split}.npy", np.stack([r[n] for r in res]))
        if split != "train":  # stone on the (inpainted) backdrop of a randomly chosen image of ANOTHER class
            y = pd.read_csv(ROOT / cfg["benchmark"]["dir"] / f"{split}.csv")["class_idx"].to_numpy()
            rng = np.random.default_rng(cfg["seed"])
            bg = np.stack([r["bg_only"] for r in res])
            swap = np.empty_like(bg)
            for i in range(len(x)):
                j = rng.choice(np.flatnonzero(y != y[i]))
                swap[i] = composite(np.array(x[i]), soft_alpha(np.array(m[i])), bg[j])
            np.save(out / f"swap_other_{split}.npy", swap)
            names.append("swap_other")
        counts[split] = names
        log.info("%s variants: %s", split, names)
    return counts


# ------------------------------------------------------------------------ evaluation
VARIANTS = ["original", "fg_only", "bg_only", "swap_black", "swap_white", "swap_other"]


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return float(f1.mean())


def stone_bootstrap(y, p, groups, k, n=1000, seed=42) -> tuple[float, float]:
    """95% interval of macro-F1 from resampling whole stones (group_id), not images."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    idx_by_group = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(idx_by_group), len(idx_by_group))
        ii = np.concatenate([idx_by_group[g] for g in pick])
        vals.append(macro_f1(y[ii], p[ii], k))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def summarise(model: str, split: str, variant: str, logits: np.ndarray, y, groups, k, orig_pred) -> dict:
    p = logits.argmax(1)
    lo, hi = stone_bootstrap(y, p, groups, k)
    row = {"model": model, "split": split, "variant": variant, "n": len(y), "accuracy": float((p == y).mean()),
           "macro_f1": macro_f1(y, p, k), "f1_ci_lo": lo, "f1_ci_hi": hi,
           "flip_rate_vs_original": float((p != orig_pred).mean()) if orig_pred is not None else 0.0}
    for c in range(k):
        row[f"recall_{c}"] = float((p[y == c] == c).mean()) if (y == c).any() else float("nan")
    return row


def evaluate_models(cfg: dict, split: str, models: dict, variants: list, purpose: str,
                    reuse_original: bool = False, preds: dict | None = None) -> pd.DataFrame:
    """Frozen checkpoints on each image variant. On the test split, `original` can be read from the saved
    predictions (no new test access for it); variants are evaluated once and every access is logged.
    `models` maps a display name (optionally 'Base@tag') to a checkpoint path."""
    import torch

    from baselines import build_model, normalisation
    from data import CachedDataset, make_loader
    from evaluate import log_test_access, sha256_file
    from train import get_device, predict

    device = get_device()
    meta = pd.read_csv(ROOT / cfg["benchmark"]["dir"] / f"{split}.csv")
    y, groups = meta["class_idx"].to_numpy(), meta["group_id"].to_numpy()
    k = len(meta["class_idx"].unique())
    rows = []
    for name, ck in models.items():
        base = name.split("@")[0]
        model = build_model(cfg, base, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
        model.to(device).eval()
        mean, std = normalisation(model)
        orig_pred = None
        for v in variants:
            if v == "original" and split == "test" and reuse_original:
                logits = np.load(ROOT / "results" / "test_predictions" / f"{base}.npz")["logits"]
            else:
                ds = CachedDataset(cfg, split, mean, std, train=False, allow_test=(split == "test"), variant=v)
                logits, yy = predict(model, make_loader(ds, 64, False, 0), device)
                assert (yy == y).all()
                if split == "test":
                    log_test_access({"model": name, "split": "test", "variant": v, "purpose": purpose,
                                     "ckpt_sha256": sha256_file(Path(ck))})
            if v == "original":
                orig_pred = logits.argmax(1)
            if preds is not None:
                preds[(name, v)] = logits.argmax(1)
            rows.append(summarise(name, split, v, logits, y, groups, k, orig_pred))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        log.info("%s done (%s)", name, split)
    df = pd.DataFrame(rows)
    ref = df[df.variant == "original"].set_index("model").macro_f1
    df["delta_f1_vs_original"] = df.macro_f1 - df.model.map(ref)
    return df


def baseline_checkpoints(cfg: dict) -> dict:
    return {n: ROOT / "models" / s["dir"] / "best.pt" for n, s in cfg["benchmark"]["models"].items()}


# ------------------------------------------------------------------------ final run
def flip_destinations(cfg: dict, split: str, preds: dict, classes: list[str]) -> pd.DataFrame:
    """For each model and altered variant: where do the changed predictions go? Concentration on a
    backdrop-linked class (Blue Sapphire = white backdrop) would indicate class-specific reliance;
    a diffuse spread indicates general sensitivity to unusual backdrops."""
    rows = []
    for (model, variant), p in preds.items():
        if variant == "original" or (model, "original") not in preds:
            continue
        o = preds[(model, "original")]
        ch = p != o
        if ch.sum() == 0:
            continue
        dest = np.bincount(p[ch], minlength=len(classes))
        top = int(dest.argmax())
        rows.append({"model": model, "split": split, "variant": variant, "n_changed": int(ch.sum()),
                     "top_destination": classes[top], "top_share": float(dest[top] / ch.sum()),
                     "to_Blue_Sapphire_share": float(dest[classes.index("Blue Sapphire")] / ch.sum())})
    return pd.DataFrame(rows)


def run_final(cfg: dict) -> None:
    """Arm A on test (frozen baselines, once, logged) + Arm B (MobileNetV2 retrained on bg_only / fg_only)."""
    import json

    classes = json.loads((ROOT / cfg["benchmark"]["dir"] / "classes.json").read_text())
    res = ROOT / "results"
    predsA: dict = {}
    A = evaluate_models(cfg, "test", baseline_checkpoints(cfg), VARIANTS, "background dependence (Arm A)",
                        reuse_original=True, preds=predsA)
    A["arm"] = "A frozen baselines"
    bdir = ROOT / "models" / "_background"
    armB = {"MobileNetV2@bg_only": bdir / "bg_only" / "best.pt", "MobileNetV2@fg_only": bdir / "fg_only" / "best.pt"}
    predsB: dict = {}
    Bv = evaluate_models(cfg, "validation", armB, ["original", "fg_only", "bg_only"], "development", preds={})
    Bt = evaluate_models(cfg, "test", armB, ["original", "fg_only", "bg_only"], "background dependence (Arm B)", preds=predsB)
    Bv["arm"] = Bt["arm"] = "B retrained MobileNetV2"
    out = pd.concat([A, Bt], ignore_index=True)
    out.to_csv(res / "background_dependence.csv", index=False)
    pd.concat([pd.read_csv(res / "background_dependence_validation.csv").assign(arm="A frozen baselines"), Bv]).to_csv(
        res / "background_dependence_validation.csv", index=False)
    pc = out.melt(id_vars=["model", "split", "variant", "arm"], value_vars=[f"recall_{i}" for i in range(len(classes))],
                  var_name="class_idx", value_name="recall")
    pc["class"] = pc.class_idx.str.replace("recall_", "").astype(int).map(dict(enumerate(classes)))
    pc.drop(columns="class_idx").to_csv(res / "background_dependence_perclass.csv", index=False)
    flip_destinations(cfg, "test", predsA, classes).to_csv(res / "background_flips.csv", index=False)
    log.info("wrote results/background_dependence*.csv")
    background_figure(cfg, out)


def background_figure(cfg: dict, df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ResNet50": "#2a78d6", "DenseNet121": "#eb6834", "MobileNetV2": "#1baf7a",
              "EfficientNetV2-S": "#eda100", "ConvNeXt-Tiny": "#e87ba4"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
    A = df[df.arm.str.startswith("A")]
    rows = ["original", "fg_only", "swap_other", "swap_white", "swap_black", "bg_only"]
    labels = {"original": "original", "fg_only": "stone only\n(grey backdrop)", "bg_only": "backdrop only\n(stone removed)",
              "swap_black": "stone on black", "swap_white": "stone on white", "swap_other": "stone on another\nclass's backdrop"}
    for r, v in enumerate(rows):
        for k, (m, c) in enumerate(colors.items()):
            q = A[(A.model == m) & (A.variant == v)].iloc[0]
            y = r + (k - 2) * 0.13
            ax.plot([q.f1_ci_lo, q.f1_ci_hi], [y, y], color=c, lw=1.4, alpha=0.7)
            ax.plot([q.macro_f1], [y], "o", color=c, ms=6.5, mec="white", mew=1, label=m if r == 0 else None)
    ax.axvline(0.10, color="#8a96a3", ls="--", lw=1)
    ax.text(0.108, -0.62, "chance 0.10", color="#5b6b7f", fontsize=8.5, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels[v] for v in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("test macro-F1 (whiskers: 95% interval, resampling stones)")
    ax.set_title("Frozen baselines on altered images", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.36))
    ax.grid(axis="x", color="#e3e8ee", lw=0.8)
    B = df[df.arm.str.startswith("B")]
    orig = A[(A.model == "MobileNetV2") & (A.variant == "original")].iloc[0]
    items = [("trained and tested on\noriginal images", orig.macro_f1, orig.f1_ci_lo, orig.f1_ci_hi, "#2a78d6"),
             ("trained and tested on\nstone only", *B[(B.model == "MobileNetV2@fg_only") & (B.variant == "fg_only")].iloc[0][["macro_f1", "f1_ci_lo", "f1_ci_hi"]], "#eb6834"),
             ("trained and tested on\nbackdrop only", *B[(B.model == "MobileNetV2@bg_only") & (B.variant == "bg_only")].iloc[0][["macro_f1", "f1_ci_lo", "f1_ci_hi"]], "#1baf7a")]
    for y, (lab, v, lo, hi, c) in enumerate(items):
        bx.plot([lo, hi], [y, y], color=c, lw=2, alpha=0.7)
        bx.plot([v], [y], "o", color=c, ms=9, mec="white", mew=1.2)
        if v > 0.7:  # label to the left of the dot so it never sits on the marker
            bx.text(v - 0.035, y, f"{v:.3f}", va="center", ha="right", fontsize=10, fontweight="bold")
        else:
            bx.text(hi + 0.02, y, f"{v:.3f}", va="center", ha="left", fontsize=10, fontweight="bold")
    bx.axvline(0.10, color="#8a96a3", ls="--", lw=1)
    bx.set_yticks(range(3))
    bx.set_yticklabels([i[0] for i in items])
    bx.invert_yaxis()
    bx.set_xlim(0, 1)
    bx.set_xlabel("test macro-F1")
    bx.set_title("MobileNetV2 retrained on each version", loc="left", fontsize=11)
    bx.grid(axis="x", color="#e3e8ee", lw=0.8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "background_dependence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["masks", "masksheet", "variants", "evaluate-val", "final", "figure"])
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.stage == "masks":
        build_masks(cfg)
    elif a.stage == "masksheet":
        log.info("wrote %s", mask_sheet(cfg))
    elif a.stage == "variants":
        build_variants(cfg)
    elif a.stage == "final":
        run_final(cfg)
    elif a.stage == "figure":
        background_figure(cfg, pd.read_csv(ROOT / "results" / "background_dependence.csv"))
    elif a.stage == "evaluate-val":
        df = evaluate_models(cfg, "validation", baseline_checkpoints(cfg), VARIANTS, "development")
        out = ROOT / "results" / "background_dependence_validation.csv"
        df.to_csv(out, index=False)
        print(df[["model", "variant", "accuracy", "macro_f1", "f1_ci_lo", "f1_ci_hi", "delta_f1_vs_original", "flip_rate_vs_original"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
