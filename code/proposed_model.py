"""Dual-branch Colour-Texture Attention Network (Deliverable 3).

  RGB image --> Branch A: EfficientNetV2-S (pretrained)  --> 1x1 projection --> P_rgb  (256 x 7 x 7)
            \\-> HSV + Sobel/Laplacian map --> Branch B: lightweight CNN --> 1x1 projection --> P_ct
  P_rgb, P_ct --> gated attention fusion --> global average pool --> dropout --> linear --> 10 classes

Gated attention fusion: z = [P_rgb ; P_ct];  g_rgb = sigmoid(conv(z)), g_ct = sigmoid(conv(z));
fused = [g_rgb * P_rgb ; g_ct * P_ct]. Each gate sees BOTH modalities, so colour/texture can modulate the
RGB features and vice versa (cross gating). `fusion="concat"` skips the gates; `fusion="none"` is for a
single branch. The flags use_rgb / use_ct / fusion are the Deliverable 5 ablation switches.

Branch B input, 6 channels, computed on the fly from the augmented RGB tensor (no learning, no extra files):
  s*cos(2*pi*h), s*sin(2*pi*h)   hue as a point on a circle, weighted by saturation (grey pixels have no hue)
  s, v                           saturation and value
  |Sobel|(grey) / (4*sqrt2)      edge strength, scaled to [0, 1]
  Laplacian(grey) / 4            second-derivative texture, scaled to [-1, 1]

Usage: python code/proposed_model.py --summary | --diagram
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ------------------------------------------------------------------ colour / texture features
def rgb_to_hsv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: (B,3,H,W) in [0,1] -> h in [0,1), s, v each (B,1,H,W)."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    maxc, _ = x.max(1, keepdim=True)
    minc, _ = x.min(1, keepdim=True)
    delta = maxc - minc
    v = maxc
    s = delta / (maxc + 1e-8)
    safe = delta + 1e-8
    h = torch.where(maxc == r, ((g - b) / safe) % 6,
                    torch.where(maxc == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6.0
    h = torch.where(delta > 1e-8, h, torch.zeros_like(h))
    return h % 1.0, s, v


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def sobel_magnitude(gray: torch.Tensor) -> torch.Tensor:
    kx = _SOBEL_X.to(gray).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    p = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(p, kx), F.conv2d(p, ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-12) / (4.0 * math.sqrt(2.0))


def laplacian(gray: torch.Tensor) -> torch.Tensor:
    k = _LAPLACE.to(gray).view(1, 1, 3, 3)
    return F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), k) / 4.0


@torch.no_grad()
def colour_texture_map(x01: torch.Tensor) -> torch.Tensor:
    """x01: (B,3,H,W) RGB in [0,1] -> (B,6,H,W) colour/texture input of Branch B."""
    h, s, v = rgb_to_hsv(x01)
    ang = 2 * math.pi * h
    gray = (0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3])
    return torch.cat([s * torch.cos(ang), s * torch.sin(ang), s, v, sobel_magnitude(gray), laplacian(gray)], 1)


# ----------------------------------------------------------------------------- the network
def conv_bn_relu(cin: int, cout: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def project(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class ColourTextureAttentionNet(nn.Module):
    def __init__(self, num_classes: int = 10, backbone: str = "tf_efficientnetv2_s.in1k", pretrained: bool = True,
                 dim: int = 256, dropout: float = 0.3, use_rgb: bool = True, use_ct: bool = True,
                 fusion: str = "gated"):
        super().__init__()
        if not (use_rgb or use_ct):
            raise ValueError("at least one branch is needed")
        if fusion not in ("gated", "concat", "none"):
            raise ValueError(f"unknown fusion {fusion!r}")
        if use_rgb and use_ct and fusion == "none":
            raise ValueError("two branches need fusion='gated' or 'concat'")
        if not (use_rgb and use_ct):
            fusion = "none"
        self.use_rgb, self.use_ct, self.fusion, self.dim = use_rgb, use_ct, fusion, dim
        # the backbone's own normalisation, used to recover [0,1] RGB for Branch B
        self.pretrained_cfg = dict(timm.models.get_pretrained_cfg(backbone).to_dict())
        self.register_buffer("_mean", torch.tensor(self.pretrained_cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(self.pretrained_cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.backbone = self.proj_rgb = self.ct_net = self.proj_ct = self.gate_rgb = self.gate_ct = None
        if use_rgb:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
            self.proj_rgb = project(self.backbone.num_features, dim)
        if use_ct:
            self.ct_net = nn.Sequential(conv_bn_relu(6, 32), conv_bn_relu(32, 64), conv_bn_relu(64, 128),
                                        conv_bn_relu(128, 256), conv_bn_relu(256, 256))   # 224 px -> 7 x 7
            self.proj_ct = project(256, dim)
        if fusion == "gated":
            self.gate_rgb = nn.Conv2d(2 * dim, dim, 1)
            self.gate_ct = nn.Conv2d(2 * dim, dim, 1)
            for g in (self.gate_rgb, self.gate_ct):  # start neutral: every gate ~0.5, so no saturation at step 0
                nn.init.normal_(g.weight, std=0.01)
                nn.init.zeros_(g.bias)
        width = dim * (2 if (use_rgb and use_ct) else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width, num_classes)
        self.last_gates: tuple[torch.Tensor, torch.Tensor] | None = None

    # ---- interface used by the shared trainer (see baselines.head_and_backbone)
    def get_classifier(self) -> nn.Module:
        return self.fc

    def pretrained_params(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters()) if self.backbone is not None else []

    def new_layer_params(self) -> list[nn.Parameter]:
        ids = {id(p) for p in self.pretrained_params()}
        return [p for p in self.parameters() if id(p) not in ids]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        p_rgb = p_ct = None
        if self.use_rgb:
            p_rgb = self.proj_rgb(self.backbone.forward_features(x))
        if self.use_ct:
            x01 = (x * self._std + self._mean).clamp(0, 1)
            ct = self.proj_ct(self.ct_net(colour_texture_map(x01)))
            p_ct = F.adaptive_avg_pool2d(ct, p_rgb.shape[-2:]) if p_rgb is not None else ct
        if self.fusion == "gated":
            z = torch.cat([p_rgb, p_ct], 1)
            g_r, g_c = torch.sigmoid(self.gate_rgb(z)), torch.sigmoid(self.gate_ct(z))
            self.last_gates = (g_r.detach(), g_c.detach())
            feats = [g_r * p_rgb, g_c * p_ct]
        elif self.fusion == "concat":
            feats = [p_rgb, p_ct]
        else:
            feats = [p_rgb if p_rgb is not None else p_ct]
        fused = torch.cat(feats, 1)
        return self.fc(self.drop(fused.mean((2, 3))))


def build_proposed(cfg: dict, pretrained: bool = True, num_classes: int | None = None, **overrides) -> ColourTextureAttentionNet:
    """Proposed model from config.yaml `proposed:`; keyword overrides switch the ablation flags."""
    if num_classes is None:
        from data import class_names

        num_classes = len(class_names(cfg))
    p = cfg["proposed"]
    kw = dict(backbone=p["backbone"], dim=p["dim"], dropout=p["dropout"], use_rgb=p["use_rgb"], use_ct=p["use_ct"],
              fusion=p["fusion"])
    kw.update(overrides)
    return ColourTextureAttentionNet(num_classes=num_classes, pretrained=pretrained, **kw)


# ------------------------------------------------------------------------------ summary / figure
def summary(model: ColourTextureAttentionNet) -> dict:
    n = lambda ps: sum(p.numel() for p in ps)
    return {"total": n(model.parameters()), "pretrained_backbone": n(model.pretrained_params()),
            "new_layers": n(model.new_layer_params()),
            "branch_B": n(model.ct_net.parameters()) + n(model.proj_ct.parameters()) if model.ct_net is not None else 0}


def draw_architecture(path: Path, model: ColourTextureAttentionNet) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    s = summary(model)
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#16324f", fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#3e4e62", lw=1.3))

    box(1, 26, 14, 10, "RGB image\n224 x 224", "#eef3f8", bold=True)
    box(24, 42, 28, 12, "Branch A\nEfficientNetV2-S (ImageNet-1k)\n1280 x 7 x 7", "#dbe9fb")
    box(24, 8, 28, 12, "HSV + Sobel + Laplacian\n6 channels", "#fdf0d5")
    box(56, 8, 22, 12, "Branch B\n5 conv-BN-ReLU blocks\n256 x 7 x 7", "#fdf0d5")
    box(58, 44, 18, 8, "1x1 projection\nP_rgb  256 x 7 x 7", "#dbe9fb")
    box(82, 8, 18, 8, "1x1 projection\nP_ct  256 x 7 x 7", "#fdf0d5")
    box(80, 25, 28, 14, "Gated attention fusion\ng = sigmoid(conv[P_rgb ; P_ct])\nout = [g_rgb * P_rgb ; g_ct * P_ct]", "#e0f2e9", fs=8.6)
    box(112, 28, 12, 8, "Global avg\npool, dropout\n0.3, linear\n(10 classes)", "#eef3f8", fs=8.2)
    arrow(15, 33, 24, 48)
    arrow(15, 29, 24, 14)
    arrow(52, 48, 58, 48)
    arrow(52, 14, 56, 14)
    arrow(78, 14, 82, 12)
    arrow(76, 46, 90, 39)
    arrow(94, 16, 94, 25)
    arrow(108, 32, 112, 32)
    ax.text(62.5, 59, "Dual-branch Colour-Texture Attention Network", ha="center", fontsize=13, fontweight="bold")
    ax.text(62.5, 1.0, f"{s['total'] / 1e6:.1f} M parameters  |  Branch A pretrained {s['pretrained_backbone'] / 1e6:.1f} M  |  "
                       f"Branch B {s['branch_B'] / 1e6:.2f} M  |  new layers (trained from scratch) {s['new_layers'] / 1e6:.2f} M",
            ha="center", fontsize=9.5, color="#3e4e62")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import json

    from common import ROOT, load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--diagram", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    m = build_proposed(cfg, pretrained=False)
    if a.summary or not a.diagram:
        print(json.dumps(summary(m), indent=1))
    if a.diagram:
        draw_architecture(ROOT / "figures" / "architecture.png", m)
        print("wrote figures/architecture.png")
