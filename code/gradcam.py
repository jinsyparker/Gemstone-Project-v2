"""Grad-CAM, quantified against the stone masks (supporting evidence only).

The class activation map of a model is taken on its last spatial feature map, before global pooling:
  timm baselines  model.forward_features / model.forward_head (ResNet50 layer4, DenseNet121 final features, MobileNetV2 and
                  EfficientNetV2-S last convolution, ConvNeXt-Tiny last stage)
  Proposed        the fused map (after the gates, before pooling), recomputed here from the model's own layers so that
                  proposed_model.py stays untouched; `tests/test_gradcam.py` checks the logits are identical.
For every image the map is scored against the stone mask (GrabCut, `dataset/_cache/mask256`):
  share       fraction of the map's mass that falls on the stone
  area        fraction of the image the stone covers (what a map spread evenly would score)
  enrichment  share / area  (1 = indifferent, above 1 = attends to the stone more than its size suggests)
  point20     fraction of the strongest 20% of map pixels that lie on the stone
  peak_on     the single strongest pixel lies on the stone
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402


def split_model(model: nn.Module):
    """(features(x) -> B,C,h,w ; head(features) -> logits) for any model used in this project."""
    from proposed_model import ColourTextureAttentionNet, colour_texture_map

    if isinstance(model, ColourTextureAttentionNet):
        m = model

        def features(x):
            p_rgb = p_ct = None
            if m.use_rgb:
                p_rgb = m.proj_rgb(m.backbone.forward_features(x))
            if m.use_ct:
                x01 = (x * m._std + m._mean).clamp(0, 1)
                ct = m.proj_ct(m.ct_net(colour_texture_map(x01)))
                p_ct = F.adaptive_avg_pool2d(ct, p_rgb.shape[-2:]) if p_rgb is not None else ct
            if m.fusion == "gated":
                z = torch.cat([p_rgb, p_ct], 1)
                g_r, g_c = torch.sigmoid(m.gate_rgb(z)), torch.sigmoid(m.gate_ct(z))
                return torch.cat([g_r * p_rgb, g_c * p_ct], 1)
            if m.fusion == "concat":
                return torch.cat([p_rgb, p_ct], 1)
            return p_rgb if p_rgb is not None else p_ct

        def head(f):
            return m.fc(m.drop(f.mean((2, 3))))

        return features, head
    return model.forward_features, model.forward_head


def grad_cam(features, head, x: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """CAM (B, H, W) at the input size, normalised to [0, 1] per image, and the logits. Target = predicted class by default."""
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)  # keeps the graph alive even if every parameter is frozen
        f = features(x)
        f.retain_grad()
        logits = head(f)
        t = logits.argmax(1) if target is None else target
        logits.gather(1, t[:, None]).sum().backward()
        alpha = f.grad.mean((2, 3), keepdim=True)
        cam = F.relu((alpha * f.detach()).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mx = cam.flatten(1).max(1).values.clamp_min(1e-12)
    return (cam / mx[:, None, None]).detach(), logits.detach()


def cam_stats(cam: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Scores of maps (B,H,W) against boolean stone masks (B,H,W)."""
    mask = mask.bool()
    total = cam.sum((1, 2))
    valid = total > 1e-9
    share = (cam * mask).sum((1, 2)) / total.clamp_min(1e-12)
    area = mask.float().mean((1, 2))
    thr = torch.quantile(cam.flatten(1), 0.8, dim=1)[:, None, None]
    sel = (cam >= thr) & (cam > 0)
    point20 = (sel & mask).sum((1, 2)) / sel.sum((1, 2)).clamp_min(1)
    peak = cam.flatten(1).argmax(1)
    peak_on = mask.flatten(1).gather(1, peak[:, None])[:, 0]
    return {"share": share, "area": area, "enrichment": share / area.clamp_min(1e-6), "point20": point20,
            "peak_on": peak_on.float(), "valid": valid.float()}


def load_masks(cfg: dict, split: str, size: int) -> np.ndarray:
    """Stone masks of a split resized like the images (256 px cache -> `size` px), boolean (N, size, size)."""
    m = np.load(ROOT / cfg["audit"]["cache"] / "mask256" / f"{split}.npy")
    t = F.interpolate(torch.from_numpy(m)[:, None].float(), size=(size, size), mode="area")[:, 0]
    return (t > 0.5).numpy()


def split_stats(cfg: dict, model: nn.Module, split: str, variants: tuple[str, ...], device, allow_test: bool = False,
                batch: int = 32, target_true: bool = False, limit: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Per-image CAM scores for each image variant of a split. The target is the model's predicted class unless
    `target_true` is set (then the true class)."""
    from baselines import normalisation
    from data import CachedDataset, make_loader

    mean, std = normalisation(model)
    size = cfg["benchmark"]["input_size"]
    masks_all = torch.from_numpy(load_masks(cfg, split, size))
    features, head = split_model(model)
    out: dict[str, dict[str, np.ndarray]] = {}
    model.eval()
    for v in variants:
        ds = CachedDataset(cfg, split, mean, std, train=False, allow_test=allow_test, variant=v, limit=limit)
        masks = masks_all[torch.from_numpy(ds.keep)]
        rows: dict[str, list] = {}
        preds, i = [], 0
        for x, y in make_loader(ds, batch, False, 0):
            n = len(y)
            cam, logits = grad_cam(features, head, x.to(device), (y.to(device) if target_true else None))
            st = cam_stats(cam.cpu(), masks[i:i + n])
            for k, val in st.items():
                rows.setdefault(k, []).append(val.numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            i += n
        out[v] = {k: np.concatenate(val) for k, val in rows.items()}
        out[v]["pred"] = np.concatenate(preds)
    return out
