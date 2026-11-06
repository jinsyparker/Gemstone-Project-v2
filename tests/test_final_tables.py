"""Table builder on synthetic per-checkpoint outputs with planted effects: Proposed keeps its accuracy under noise,
EfficientNetV2-S loses 15 points, ConvNeXt-Tiny behaves like Proposed."""
import os
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import final_tables as ft  # noqa: E402
import robustness as rb  # noqa: E402
import runs  # noqa: E402

K, PER, STONES = 10, 3, 120


def noisy(y, err, seed):
    rng = np.random.default_rng(seed)
    p = y.copy()
    w = rng.random(len(y)) < err
    p[w] = (p[w] + 1 + rng.integers(0, K - 1, w.sum())) % K
    return p


def fake_outputs(rid, y, groups, base_err, noise_err, seed):
    rng = np.random.default_rng(seed)
    out = {"y": y, "groups": groups.astype(str), "sha": np.array("x")}
    for v in ("original", "fg_only", "bg_only", "swap_black", "swap_white", "swap_other"):
        err = {"original": base_err, "fg_only": base_err + 0.01, "bg_only": 0.7, "swap_white": base_err + 0.05}.get(v, base_err + 0.02)
        out[f"bg|{v}|logits"] = np.eye(K)[noisy(y, err, seed + sum(map(ord, v)))] * 5
    for c, cond in rb.COND.items():
        err = noise_err if c == "Noise σ=10" else base_err + 0.005
        views = 2 if cond["family"] == "rotation" else 1
        out[f"rob|{c}|pred"] = np.concatenate([noisy(y, err, seed + 7 * j + len(c)) for j in range(views)]).astype(np.int8)
    for v in ("original", "swap_white"):
        sh = rng.random(len(y))
        out.update({f"cam|{v}|share": sh, f"cam|{v}|area": np.full(len(y), 0.4), f"cam|{v}|enrichment": sh / 0.4,
                    f"cam|{v}|point20": rng.random(len(y)), f"cam|{v}|peak_on": (rng.random(len(y)) > 0.1).astype(float),
                    f"cam|{v}|valid": np.ones(len(y)), f"cam|{v}|pred": out["bg|original|logits"].argmax(1)})
    return out


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    d = tmp_path_factory.mktemp("final")
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(STONES), PER)
    y = np.repeat(np.arange(STONES) % K, PER)
    for rid in runs.eval_run_ids():
        base, seed = runs.parse(rid)
        noise_err = 0.30 if base == "EfficientNetV2-S" else 0.06
        z = fake_outputs(rid, y, groups, 0.05, noise_err, seed * 17 + len(base))
        np.savez(d / f"{rid}.npz", **z)
    out = ft.build("test", directory=d, out=d / "tables", n_draws=150)
    return out


def test_all_tables_written(tables):
    for n in ("clean", "clean_comparisons", "robustness_seeds", "robustness_comparisons", "background_seeds", "background_comparisons",
              "white_swap_destinations", "ablation", "ablation_comparisons", "gradcam"):
        assert (tables / f"{n}.csv").exists(), n


def test_planted_robustness_difference_is_found_and_null_is_not(tables):
    c = pd.read_csv(tables / "robustness_comparisons.csv")
    noise = c[(c.condition == "Noise σ=10")].set_index("baseline")
    assert noise.loc["EfficientNetV2-S", "verdict"] == "Proposed more robust"
    assert noise.loc["ConvNeXt-Tiny", "verdict"] == "no clear difference"
    other = c[(c.condition == "Brightness −20%") & (c.baseline == "EfficientNetV2-S")]
    assert other.verdict.iloc[0] == "no clear difference"


def test_within_two_points_flag_and_seed_means(tables):
    r = pd.read_csv(tables / "robustness_seeds.csv")
    n = r[r.condition == "Noise σ=10"].set_index("model")
    assert n.loc["Proposed", "within_2_points"] and not n.loc["EfficientNetV2-S", "within_2_points"]
    assert (r.groupby("model").size() == len(rb.COND)).all()


def test_clean_and_ablation_tables(tables):
    c = pd.read_csv(tables / "clean.csv")
    assert set(c.model) == set(ft.MAIN) and (c.n_seeds == 3).all()
    a = pd.read_csv(tables / "ablation.csv")
    assert len(a) == 6
    ac = pd.read_csv(tables / "ablation_comparisons.csv")
    assert {c.split(" (")[0] for c in ac.comparison} == {"E vs A-FT", "E vs D", "D vs C", "D vs A"}
    assert (ac[ac.measure == "clean macro-F1"].shape[0]) == 4


def test_background_and_gradcam_tables(tables):
    b = pd.read_csv(tables / "background_seeds.csv")
    assert set(b.statistic) >= {"stone_only_change", "backdrop_only_f1", "white_backdrop_change"}
    g = pd.read_csv(tables / "gradcam.csv")
    assert {"original", "swap_white"} == set(g.variant) and (g.enrichment > 0).all()
    assert {"ResNet50", "DenseNet121", "MobileNetV2"} <= set(g.model)
