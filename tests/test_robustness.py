"""Checks of the robustness tooling (perturbations, paired statistics, test-set guard). CPU only, no model needed."""
import os
import subprocess
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import robustness as rb  # noqa: E402
from common import load_config  # noqa: E402
from data import HeldOutSplitError, load_cache_array  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    x = rng.integers(60, 190, (4, 64, 64, 3), dtype=np.uint8)
    x[:, 20:44, 20:44] = [200, 40, 60]  # a coloured block so rotation and blur have something to move
    return x


def test_registry_shape():
    assert len(rb.COND) == 18
    assert len(rb.CORE) == 8  # brightness x2, contrast x2, noise, rotation, flip, blur
    assert set(c["family"] for c in rb.COND.values()) == {"brightness", "contrast", "noise", "rotation", "flip", "blur"}


def test_all_conditions_keep_dtype_shape_range(batch):
    for name, c in rb.COND.items():
        views = c["fn"](batch)
        assert len(views) == (2 if c["family"] == "rotation" else 1), name
        for v in views:
            assert v.dtype == np.uint8 and v.shape == batch.shape, name


def test_neutral_levels_are_identity(batch):
    assert np.array_equal(rb.brightness(batch, 1.0), batch)
    assert np.array_equal(rb.contrast(batch, 1.0), batch)
    assert np.array_equal(rb.gaussian_noise(batch, 0.0), batch)
    assert np.array_equal(rb.rotate(batch, 0), batch)
    assert np.allclose(rb.blur(batch, 1e-3).astype(int), batch.astype(int), atol=1)


def test_brightness_scales_and_clips():
    x = np.full((1, 8, 8, 3), 100, np.uint8)
    assert (rb.brightness(x, 1.2) == 120).all() and (rb.brightness(x, 0.8) == 80).all()
    assert (rb.brightness(x, 3.0) == 255).all()


def test_contrast_keeps_mean_grey_and_scales_spread(batch):
    grey = lambda a: (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean(axis=(1, 2))
    lo = rb.contrast(batch, 0.8)
    assert np.allclose(grey(lo), grey(batch), atol=1.0)
    assert lo.astype(float).std() < batch.astype(float).std()
    assert rb.contrast(batch, 1.2).astype(float).std() > batch.astype(float).std()


def test_noise_has_requested_sigma_and_is_deterministic():
    x = np.full((6, 64, 64, 3), 128, np.uint8)
    a, b = rb.gaussian_noise(x, 10), rb.gaussian_noise(x, 10)
    assert np.array_equal(a, b)
    assert abs((a.astype(float) - 128).std() - 10) < 0.5
    # one noise pattern scaled by sigma, so a stronger level is the same pattern louder
    n10, n20 = a.astype(float) - 128, rb.gaussian_noise(x, 20).astype(float) - 128
    assert np.corrcoef(n10.ravel(), n20.ravel())[0, 1] > 0.98
    assert not np.array_equal(a[0], a[1])  # different images get different noise


def test_rotation_has_no_black_corners_and_moves_content():
    x = np.full((1, 64, 64, 3), 200, np.uint8)
    r = rb.rotate(x, 20)
    assert r.min() >= 199  # reflected borders, not zeros
    blk = np.zeros((1, 64, 64, 3), np.uint8)
    blk[:, 10:30, 10:30] = 255
    assert not np.array_equal(rb.rotate(blk, 10), blk)


def test_flip_matches_array_flip_and_is_an_involution(batch):
    f = rb.hflip(batch)
    assert np.array_equal(f, batch[:, :, ::-1])
    assert np.array_equal(rb.hflip(f), batch)
    assert f.flags["C_CONTIGUOUS"]


def test_blur_removes_high_frequency_energy(batch):
    hf = lambda a: np.abs(np.diff(a.astype(float), axis=2)).mean()
    assert hf(rb.blur(batch, 1.0)) < hf(batch)
    assert hf(rb.blur(batch, 2.0)) < hf(rb.blur(batch, 1.0))


def _toy(n_groups=60, per=5, k=3, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = np.repeat(rng.integers(0, k, n_groups), per)
    return y, groups, k


def test_paired_delta_zero_for_identical_predictions():
    y, g, k = _toy()
    assert rb.paired_delta(y, y.copy(), y.copy(), g, k) == (0.0, 0.0, 0.0)


def test_paired_delta_brackets_a_known_drop():
    y, g, k = _toy()
    cond = y.copy()
    wrong = np.arange(len(y)) % 4 == 0
    cond[wrong] = (cond[wrong] + 1) % k
    d, lo, hi = rb.paired_delta(y, y.copy(), cond, g, k)
    assert d < -0.1 and lo <= d <= hi and hi < 0


def test_drop_difference_sign_and_null():
    y, g, k = _toy()
    cond_bad = y.copy()
    w = np.arange(len(y)) % 3 == 0
    cond_bad[w] = (cond_bad[w] + 1) % k
    # A loses more than B -> negative, interval below zero
    d, lo, hi = rb.paired_drop_difference(y, y.copy(), cond_bad, y.copy(), y.copy(), g, k)
    assert d < 0 and hi < 0
    # identical behaviour -> exactly zero, never "more fragile"
    assert rb.paired_drop_difference(y, y.copy(), cond_bad, y.copy(), cond_bad, g, k) == (0.0, 0.0, 0.0)


def test_test_split_is_guarded_and_final_needs_the_flag():
    cfg = load_config()
    with pytest.raises(HeldOutSplitError):
        load_cache_array(cfg, "test")
    r = subprocess.run([sys.executable, str(ROOT / "code" / "robustness.py"), "final"], capture_output=True, text=True)
    assert r.returncode != 0 and "--test" in (r.stderr + r.stdout)
