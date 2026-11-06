"""Statistics of the pre-registered analyses (results/PREREGISTRATION.md, section 4).

Everything is computed on stone-level bootstrap draws that are SHARED by every model, seed and condition, so any difference is
paired. A "seed-mean" statistic is the mean over seeds; its interval is the spread of that mean over the draws (test-set
sampling). The SD across seeds and each seed's own value are reported beside it.

A claim that two models differ needs (a) the 95% interval of the seed-mean difference to exclude zero and (b) the same sign in
every individual seed. Otherwise the result is "no clear difference".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_DRAWS = 1000
SEED = 42


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return float(f1.mean())


def stone_draws(groups: np.ndarray, n: int = N_DRAWS, seed: int = SEED) -> list[np.ndarray]:
    """Image-index arrays, one per draw, resampling whole stones with replacement."""
    codes, uniq = pd.factorize(groups)
    by = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([by[g] for g in rng.integers(0, len(by), len(by))]) for _ in range(n)]


def _tile(y: np.ndarray, ii: np.ndarray, views: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    idx = np.concatenate([ii + j * n for j in range(views)]) if views > 1 else ii
    return np.tile(y, views)[idx], idx


def f1_vector(y: np.ndarray, pred: np.ndarray, draws: list[np.ndarray], k: int) -> tuple[float, np.ndarray]:
    """Point macro-F1 and its value on every bootstrap draw. `pred` may hold several views of the images, one after another."""
    views = len(pred) // len(y)
    yy = np.tile(y, views)
    point = macro_f1(yy, pred, k)
    vec = np.empty(len(draws))
    for i, ii in enumerate(draws):
        ys, idx = _tile(y, ii, views)
        vec[i] = macro_f1(ys, pred[idx], k)
    return point, vec


def acc_vector(y: np.ndarray, pred: np.ndarray, draws: list[np.ndarray]) -> tuple[float, np.ndarray]:
    views = len(pred) // len(y)
    ok = (pred == np.tile(y, views)).astype(float)
    n = len(y)
    return float(ok.mean()), np.array([np.mean([ok[ii + j * n].mean() for j in range(views)]) for ii in draws])


def mean_vector(values: np.ndarray, draws: list[np.ndarray]) -> tuple[float, np.ndarray]:
    """Mean of a per-image quantity (for example a Grad-CAM score) and its bootstrap distribution."""
    ok = ~np.isnan(values)
    v = np.where(ok, values, 0.0)
    return float(v[ok].mean()), np.array([v[ii].sum() / max(ok[ii].sum(), 1) for ii in draws])


def interval(vec: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(vec, 0.025)), float(np.quantile(vec, 0.975))


def seed_mean(items: list[tuple[float, np.ndarray]]) -> tuple[float, np.ndarray, list[float], float]:
    """(mean point, mean vector, per-seed points, SD across seeds) of per-seed (point, vector) pairs."""
    pts = [p for p, _ in items]
    return float(np.mean(pts)), np.mean([v for _, v in items], axis=0), pts, float(np.std(pts, ddof=1)) if len(pts) > 1 else float("nan")


def difference(a: tuple[float, np.ndarray], b: tuple[float, np.ndarray]) -> tuple[float, np.ndarray]:
    return a[0] - b[0], a[1] - b[1]


def verdict(point: float, vec: np.ndarray, per_seed: list[float], higher: str = "higher", lower: str = "lower") -> dict:
    """The pre-registered two-part rule for a seed-mean difference (first minus second)."""
    lo, hi = interval(vec)
    excl = lo > 0 or hi < 0
    signs = {np.sign(d) for d in per_seed}
    consistent = len(signs) == 1 and 0 not in signs and np.sign(point) in signs
    label = (higher if point > 0 else lower) if (excl and consistent) else "no clear difference"
    return {"diff": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(excl), "same_sign_all_seeds": bool(consistent),
            "verdict": label}


def compare(a_runs: list[tuple[float, np.ndarray]], b_runs: list[tuple[float, np.ndarray]], **labels) -> dict:
    """Seed-mean difference of two models given their per-seed (point, vector) pairs (same seed order)."""
    am, bm = seed_mean(a_runs), seed_mean(b_runs)
    point, vec = am[0] - bm[0], am[1] - bm[1]
    per_seed = [pa - pb for pa, pb in zip(am[2], bm[2])]
    out = verdict(point, vec, per_seed, **labels)
    out["per_seed_diff"] = per_seed
    return out
