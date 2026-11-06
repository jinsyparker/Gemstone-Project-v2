"""The seed-mean statistics and the pre-registered two-part claim rule, on synthetic predictions."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import seedstats as ss  # noqa: E402


def toy(n_groups=80, per=4, k=4, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = np.repeat(rng.integers(0, k, n_groups), per)
    return y, groups, k


def noisy(y, k, err, seed):
    rng = np.random.default_rng(seed)
    p = y.copy()
    wrong = rng.random(len(y)) < err
    p[wrong] = (p[wrong] + 1 + rng.integers(0, k - 1, wrong.sum())) % k
    return p


def test_draws_resample_whole_stones_and_are_reproducible():
    y, g, k = toy()
    d1, d2 = ss.stone_draws(g, 50), ss.stone_draws(g, 50)
    assert all(np.array_equal(a, b) for a, b in zip(d1, d2))
    for ii in d1[:5]:
        counts = np.bincount(g[ii], minlength=g.max() + 1)
        assert set(counts) <= {0, 4, 8, 12, 16, 20}  # every drawn stone contributes all of its 4 images, so counts are multiples of 4


def test_point_matches_plain_macro_f1_and_multi_view_tiles_correctly():
    y, g, k = toy()
    p = noisy(y, k, 0.2, 1)
    d = ss.stone_draws(g, 30)
    point, vec = ss.f1_vector(y, p, d, k)
    assert point == ss.macro_f1(y, p, k) and len(vec) == 30
    two = np.concatenate([p, p])
    p2, v2 = ss.f1_vector(y, two, d, k)
    assert p2 == point and np.allclose(v2, vec)


def test_identical_models_have_zero_difference_and_no_claim():
    y, g, k = toy()
    d = ss.stone_draws(g, 200)
    runs = [ss.f1_vector(y, noisy(y, k, 0.2, s), d, k) for s in (1, 2, 3)]
    out = ss.compare(runs, runs)
    assert out["diff"] == 0 and out["verdict"] == "no clear difference"


def test_a_real_and_consistent_difference_is_claimed():
    y, g, k = toy(n_groups=200)
    d = ss.stone_draws(g, 300)
    good = [ss.f1_vector(y, noisy(y, k, 0.05, s), d, k) for s in (1, 2, 3)]
    bad = [ss.f1_vector(y, noisy(y, k, 0.30, s + 10), d, k) for s in (1, 2, 3)]
    out = ss.compare(good, bad)
    assert out["verdict"] == "higher" and out["excludes_zero"] and out["same_sign_all_seeds"]
    assert ss.compare(bad, good)["verdict"] == "lower"


def test_sign_disagreement_blocks_a_claim_even_if_the_mean_interval_excludes_zero():
    vec = np.random.default_rng(0).normal(0.05, 0.005, 500)  # interval clearly above zero
    out = ss.verdict(0.05, vec, [0.09, 0.08, -0.02])
    assert out["excludes_zero"] and not out["same_sign_all_seeds"] and out["verdict"] == "no clear difference"
    assert ss.verdict(0.05, vec, [0.04, 0.05, 0.06])["verdict"] == "higher"


def test_wide_interval_is_not_a_claim():
    vec = np.random.default_rng(1).normal(0.01, 0.05, 500)
    assert ss.verdict(0.01, vec, [0.01, 0.02, 0.01])["verdict"] == "no clear difference"


def test_mean_vector_ignores_nan_and_tracks_the_mean():
    y, g, k = toy()
    d = ss.stone_draws(g, 100)
    v = np.random.default_rng(0).random(len(y))
    v[3] = np.nan
    point, vec = ss.mean_vector(v, d)
    assert abs(point - np.nanmean(v)) < 1e-9 and abs(vec.mean() - point) < 0.05


def test_seed_mean_returns_sd_across_seeds():
    items = [(0.9, np.full(5, 0.9)), (0.8, np.full(5, 0.8)), (0.7, np.full(5, 0.7))]
    m, vec, pts, sd = ss.seed_mean(items)
    assert abs(m - 0.8) < 1e-12 and np.allclose(vec, 0.8) and pts == [0.9, 0.8, 0.7] and abs(sd - 0.1) < 1e-12
