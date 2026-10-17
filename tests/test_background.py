"""Synthetic checks of the background-dependence tooling (masks, variants, statistics). CPU only."""
import os
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import background_analysis as ba  # noqa: E402
from common import load_config  # noqa: E402
from data import CachedDataset, HeldOutSplitError  # noqa: E402


def make_scene(seed=0, hue=(200, 40, 60)):
    """256x256 studio-like scene: vignetted grey backdrop, a coloured stone, a grey cast shadow and a
    faint coloured reflection under it. Returns (image, true stone mask, clean backdrop)."""
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    r = np.sqrt((xx - 128) ** 2 + (yy - 120) ** 2) / 180
    backdrop = np.clip(150 - 30 * r, 0, 255)[..., None] * np.array([1.0, 1.0, 1.01], np.float32)
    img = backdrop.copy()
    stone = ((xx - 128) / 62) ** 2 + ((yy - 112) / 50) ** 2 < 1
    shadow = (((xx - 140) / 70) ** 2 + ((yy - 172) / 12) ** 2 < 1) & ~stone
    refl = (((xx - 128) / 50) ** 2 + ((yy - 168) / 7) ** 2 < 1) & ~stone
    img[shadow] *= 0.68                                         # grey, darker than the backdrop
    img[refl] = 0.8 * img[refl] + 0.2 * np.array(hue, np.float32)   # coloured reflection
    rng = np.random.default_rng(seed)
    tex = rng.normal(0, 18, (256, 256, 1)).astype(np.float32)
    img[stone] = (np.array(hue, np.float32) + tex)[stone]
    return np.clip(img, 0, 255).astype(np.uint8), stone, backdrop.astype(np.uint8)


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_mask_covers_the_stone_and_stays_close_to_it():
    """The stone is fully covered. The coloured reflection touching it is included by design (it carries the
    stone's colour), so extra area is allowed, but it must stay small and near the stone, and the far part of
    the grey cast shadow must be excluded."""
    img, stone, _ = make_scene()
    m, info = ba.compute_mask(img)
    assert info["ok"]
    assert (m & stone).sum() / stone.sum() > 0.98
    extra = m & ~stone
    assert extra.sum() < 0.30 * stone.sum()
    far = ~cv2.dilate(stone.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    assert (m & far).sum() == 0
    yy, xx = np.mgrid[0:256, 0:256]
    far_shadow = (((xx - 140) / 70) ** 2 + ((yy - 172) / 12) ** 2 < 1) & (xx > 192)   # the distant right end of the shadow
    assert (m & far_shadow).sum() == 0


def test_fg_only_keeps_stone_and_flattens_the_rest():
    img, stone, _ = make_scene()
    m, _ = ba.compute_mask(img)
    out = ba.composite(img, ba.soft_alpha(m), ba.STUDIO_GREY)
    inner = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    outer = ~cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert np.array_equal(out[inner], img[inner])
    assert (out[outer] == np.rint(ba.STUDIO_GREY).astype(np.uint8)).all()


def test_bg_only_removes_stone_shadow_and_coloured_reflection():
    img, stone, backdrop = make_scene()
    m, _ = ba.compute_mask(img)
    out = ba.remove_stone(img, m)
    area = cv2.dilate(stone.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert np.abs(out.astype(float) - backdrop.astype(float))[area].mean() < 10   # close to the true clean backdrop
    lum = out.astype(float).mean(-1, keepdims=True) + 1e-3
    tint = np.linalg.norm((out / lum).reshape(-1, 3)[area.ravel()].mean(0) - (backdrop / (backdrop.mean(-1, keepdims=True) + 1e-3)).reshape(-1, 3)[area.ravel()].mean(0))
    assert tint < 0.02                                                              # no stone colour left behind
    reflection_zone = out[160:176, 90:170].astype(float)
    assert reflection_zone[..., 0].mean() - reflection_zone[..., 1].mean() < 6      # red reflection tint gone


def test_swap_variants_keep_the_stone_and_replace_the_backdrop():
    img, _, _ = make_scene()
    m, _ = ba.compute_mask(img)
    a = ba.soft_alpha(m)
    inner = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    for colour in ((0, 0, 0), (255, 255, 255)):
        out = ba.composite(img, a, colour)
        assert np.array_equal(out[inner], img[inner])
        assert (out[~cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)] == colour[0]).all()


def test_macro_f1_matches_sklearn_and_bootstrap_brackets_it():
    rng = np.random.default_rng(1)
    k, n = 5, 400
    y = rng.integers(0, k, n)
    p = np.where(rng.random(n) < 0.8, y, rng.integers(0, k, n))
    assert ba.macro_f1(y, p, k) == pytest.approx(f1_score(y, p, average="macro"))
    groups = np.repeat(np.arange(n // 4), 4)
    lo, hi = ba.stone_bootstrap(y, p, groups, k, n=300)
    assert lo < ba.macro_f1(y, p, k) < hi
    lo2, hi2 = ba.stone_bootstrap(y[:80], p[:80], groups[:80], k, n=300)
    assert (hi2 - lo2) > (hi - lo)                       # fewer stones, wider interval


def test_variant_loading_and_test_guard():
    cfg = load_config()
    ds = CachedDataset(cfg, "validation", (0.5,) * 3, (0.5,) * 3, variant="fg_only", limit=20)
    x, y = ds[0]
    assert tuple(x.shape) == (3, 224, 224)
    with pytest.raises(HeldOutSplitError):
        CachedDataset(cfg, "test", (0.5,) * 3, (0.5,) * 3, variant="bg_only")
