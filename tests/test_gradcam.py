"""Grad-CAM tooling on synthetic networks: the map lands where the evidence is, and the proposed-model replay is exact."""
import os
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import gradcam as gc  # noqa: E402
from common import load_config  # noqa: E402


class Toy(nn.Module):
    """Logit 1 grows with image brightness, logit 0 shrinks: a bright patch is the only evidence."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(32)  # 224 -> 7 x 7
        self.fc = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[-1.0], [1.0]]))

    def forward_features(self, x):
        return self.pool(x.mean(1, keepdim=True))

    def forward_head(self, f):
        return self.fc(f.mean((2, 3)))


def toy_batch(n=4):
    x = torch.zeros(n, 3, 224, 224)
    x[:, :, 64:160, 64:160] = 1.0
    mask = torch.zeros(n, 224, 224, dtype=torch.bool)
    mask[:, 64:160, 64:160] = True
    return x, mask


def test_cam_mass_lands_on_the_planted_patch():
    net = Toy().eval()
    x, mask = toy_batch()
    cam, logits = gc.grad_cam(*gc.split_model(net), x)
    assert (logits.argmax(1) == 1).all()
    st = gc.cam_stats(cam, mask)
    # the 7 x 7 map blurs a little past the patch edge, so the share is high but not 1; the patch is 18% of the image
    assert (st["share"] > 0.8).all() and (st["peak_on"] == 1).all() and (st["point20"] > 0.85).all()  # the top 20% of pixels is a little larger than the patch
    assert (st["enrichment"] > 3).all()


def test_cam_follows_the_patch_when_it_moves():
    net = Toy().eval()
    x = torch.zeros(1, 3, 224, 224)
    x[:, :, 0:64, 0:64] = 1.0
    cam, _ = gc.grad_cam(*gc.split_model(net), x)
    on_new = torch.zeros(1, 224, 224, dtype=torch.bool)
    on_new[:, 0:96, 0:96] = True
    off = torch.zeros(1, 224, 224, dtype=torch.bool)
    off[:, 128:, 128:] = True
    assert gc.cam_stats(cam, on_new)["share"][0] > 0.9
    assert gc.cam_stats(cam, off)["share"][0] < 0.05


def test_stats_on_known_maps():
    mask = torch.zeros(1, 224, 224, dtype=torch.bool)
    mask[:, :112, :] = True  # half the image
    uniform = torch.ones(1, 224, 224)
    st = gc.cam_stats(uniform, mask)
    assert st["share"][0] == pytest.approx(0.5) and st["enrichment"][0] == pytest.approx(1.0)
    only_mask = mask.float()
    st = gc.cam_stats(only_mask, mask)
    assert st["share"][0] == pytest.approx(1.0) and st["point20"][0] == pytest.approx(1.0) and st["peak_on"][0] == 1
    st = gc.cam_stats(torch.zeros(1, 224, 224), mask)
    assert st["valid"][0] == 0


def test_proposed_replay_reproduces_the_logits_exactly():
    from proposed_model import build_proposed

    cfg = load_config()
    torch.manual_seed(0)
    for over in ({}, dict(fusion="concat"), dict(use_ct=False, fusion="none"), dict(use_rgb=False, fusion="none")):
        m = build_proposed(cfg, pretrained=False, **over).eval()
        x = torch.randn(2, 3, 224, 224)
        features, head = gc.split_model(m)
        with torch.no_grad():
            assert torch.allclose(m(x), head(features(x)), atol=1e-5), over


def test_proposed_cam_has_the_right_shape():
    from proposed_model import build_proposed

    cfg = load_config()
    m = build_proposed(cfg, pretrained=False).eval()
    cam, logits = gc.grad_cam(*gc.split_model(m), torch.randn(2, 3, 224, 224))
    assert cam.shape == (2, 224, 224) and logits.shape == (2, 10)
    assert float(cam.min()) >= 0 and float(cam.max()) <= 1.0001


def test_mask_loader_matches_the_resize():
    cfg = load_config()
    path = Path(__file__).resolve().parent.parent / cfg["audit"]["cache"] / "mask256" / "validation.npy"
    if not path.exists():
        pytest.skip("mask cache not present")
    m = gc.load_masks(cfg, "validation", 224)
    assert m.shape[1:] == (224, 224) and m.dtype == bool and 0.05 < m.mean() < 0.95
