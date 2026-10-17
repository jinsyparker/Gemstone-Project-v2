"""Tests for the dual-branch Colour-Texture Attention network (D3). CPU only, no pretrained downloads."""
import colorsys
import copy
import math
import os
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import baselines as bl  # noqa: E402
import proposed_model as pm  # noqa: E402
import train as tr  # noqa: E402
from common import load_config  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def build(cfg, **kw):
    torch.manual_seed(0)
    return pm.build_proposed(cfg, pretrained=False, **kw).eval()


def test_hsv_matches_colorsys():
    rng = np.random.default_rng(0)
    img = rng.random((1, 3, 16, 16)).astype(np.float32)
    img[0, :, 0, 0] = 0.4                                   # a pure grey pixel (hue undefined)
    h, s, v = pm.rgb_to_hsv(torch.from_numpy(img))
    for y in range(16):
        for x in range(16):
            hh, ss, vv = colorsys.rgb_to_hsv(*img[0, :, y, x])
            assert float(s[0, 0, y, x]) == pytest.approx(ss, abs=1e-4)
            assert float(v[0, 0, y, x]) == pytest.approx(vv, abs=1e-5)
            if ss > 1e-3:
                d = abs(float(h[0, 0, y, x]) - hh)
                assert min(d, 1 - d) < 1e-4


def test_sobel_and_laplacian_match_opencv_in_the_interior():
    rng = np.random.default_rng(1)
    g = rng.random((40, 40)).astype(np.float32)
    t = torch.from_numpy(g)[None, None]
    sob = pm.sobel_magnitude(t)[0, 0].numpy() * (4 * math.sqrt(2))
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    assert np.allclose(sob[2:-2, 2:-2], np.sqrt(gx ** 2 + gy ** 2)[2:-2, 2:-2], atol=1e-4)
    lap = pm.laplacian(t)[0, 0].numpy() * 4
    assert np.allclose(lap[2:-2, 2:-2], cv2.Laplacian(g, cv2.CV_32F, ksize=1)[2:-2, 2:-2], atol=1e-4)


def test_colour_texture_map_shape_and_range():
    x = torch.rand(3, 3, 64, 64)
    m = pm.colour_texture_map(x)
    assert m.shape == (3, 6, 64, 64)
    assert m[:, :4].min() >= -1 - 1e-5 and m[:, :4].max() <= 1 + 1e-5      # s*cos, s*sin, s, v
    assert m[:, 4].min() >= 0 and m[:, 4].max() <= 1 + 1e-5                  # Sobel magnitude
    assert m[:, 5].abs().max() <= 1 + 1e-5                                   # Laplacian
    grey = torch.full((1, 3, 32, 32), 0.5)
    assert pm.colour_texture_map(grey)[:, :2].abs().max() == 0               # grey has no hue signal


@pytest.mark.parametrize("flags,width", [
    (dict(use_rgb=True, use_ct=False, fusion="none"), 256),                  # ablation A
    (dict(use_rgb=False, use_ct=True, fusion="none"), 256),                  # ablation B
    (dict(use_rgb=True, use_ct=True, fusion="concat"), 512),                 # ablation C
    (dict(use_rgb=True, use_ct=True, fusion="gated"), 512),                  # ablation D and E
])
def test_every_ablation_configuration_runs(cfg, flags, width):
    m = build(cfg, **flags)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10) and torch.isfinite(out).all()
    assert m.fc.in_features == width
    assert (m.backbone is not None) == flags["use_rgb"] and (m.ct_net is not None) == flags["use_ct"]


def test_invalid_configurations_are_rejected(cfg):
    with pytest.raises(ValueError):
        pm.build_proposed(cfg, pretrained=False, use_rgb=False, use_ct=False)
    with pytest.raises(ValueError):
        pm.build_proposed(cfg, pretrained=False, use_rgb=True, use_ct=True, fusion="none")


def test_gates_are_probabilities_start_neutral_and_see_both_branches(cfg):
    m = build(cfg)
    m(torch.randn(2, 3, 224, 224))
    g_r, g_c = m.last_gates
    assert g_r.shape == g_c.shape == (2, 256, 7, 7)                           # model-level wiring
    assert 0.0 <= float(g_r.min()) and float(g_r.max()) <= 1.0
    assert m.gate_rgb.in_channels == m.gate_ct.in_channels == 512             # each gate reads [P_rgb ; P_ct]
    # gate mathematics on controlled inputs (an untrained backbone's raw features are not representative)
    torch.manual_seed(1)
    p_rgb, p_ct = torch.randn(2, 256, 7, 7), torch.randn(2, 256, 7, 7)
    with torch.no_grad():
        g = torch.sigmoid(m.gate_rgb(torch.cat([p_rgb, p_ct], 1)))
        g2 = torch.sigmoid(m.gate_rgb(torch.cat([p_rgb, torch.randn_like(p_ct)], 1)))
    assert 0 < float(g.min()) and float(g.max()) < 1
    assert abs(float(g.mean()) - 0.5) < 0.05                                   # neutral start: ~0.5, not saturated
    assert not torch.allclose(g, g2)                                           # the RGB gate depends on the CT branch


def test_parameter_partition_and_counts(cfg):
    m = build(cfg)
    new, old = m.new_layer_params(), m.pretrained_params()
    assert {id(p) for p in new}.isdisjoint({id(p) for p in old})
    assert len(new) + len(old) == len(list(m.parameters()))
    bb = {id(p) for p in m.backbone.parameters()}
    assert fc_in_new(m) and all(id(p) in bb for p in old)
    s = pm.summary(m)
    assert s["total"] == s["pretrained_backbone"] + s["new_layers"]
    assert 20e6 < s["pretrained_backbone"] < 21e6 and s["branch_B"] < 1.5e6    # Branch B stays lightweight


def fc_in_new(m):
    ids = {id(p) for p in m.new_layer_params()}
    return all(id(p) in ids for p in m.fc.parameters())


def test_stage_freezing_protocol(cfg):
    m = build(cfg)
    old, new = m.pretrained_params(), m.new_layer_params()
    bl.set_stage1(m)
    assert not any(p.requires_grad for p in old) and all(p.requires_grad for p in new)
    info = bl.set_stage2(m, 0.5)
    assert abs(sum(p.requires_grad for p in old) / len(old) - 0.5) < 0.02 and info["backbone_tensors"] == len(old)
    assert all(p.requires_grad for p in new)                                  # new layers keep training in stage 2
    assert not old[0].requires_grad and old[-1].requires_grad
    bl.set_stage1(m)
    bl.set_train_mode(m)
    assert not m.backbone.bn1.training and m.ct_net[0][1].training            # frozen EfficientNet BN eval, new BN train


def test_eval_forward_is_deterministic(cfg):
    m = build(cfg)
    x = torch.randn(2, 3, 224, 224)
    assert torch.equal(m(x), m(x))


def test_baseline_helpers_unchanged_for_plain_models(cfg):
    plain = bl.build_model(cfg, "MobileNetV2", pretrained=False)
    head, backbone = bl.head_and_backbone(plain)
    assert len(head) == 2 and {id(p) for p in head} == {id(p) for p in plain.get_classifier().parameters()}


def test_two_stage_training_smoke_with_the_shared_trainer(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["benchmark"]["batch_size"] = 8
    h = tr.train_model(c, build(c).train(), tmp_path / "ck", tmp_path / "h.json", seed=42, limit=20,
                       s1_epochs=1, s2_epochs=1, name="Proposed")
    assert [r["stage"] for r in h["epochs"]] == [1, 2] and (tmp_path / "ck" / "best.pt").exists()
    assert h["stage2_unfreeze"]["unfrozen_tensors"] > 0
