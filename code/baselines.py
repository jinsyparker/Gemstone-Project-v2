"""The five baseline architectures (timm, ImageNet-1k weights) and the freezing helpers.

The freezing/param-group helpers work on any nn.Module that exposes `get_classifier()`, so the
proposed model in D3 to D5 reuses them unchanged.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import timm
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_model(cfg: dict, name: str, pretrained: bool = True, num_classes: int | None = None) -> nn.Module:
    """timm model with a fresh `num_classes`-way head (default: the benchmark's class count)."""
    if num_classes is None:
        from data import class_names

        num_classes = len(class_names(cfg))
    if name == "Proposed":
        from proposed_model import build_proposed

        return build_proposed(cfg, pretrained=pretrained, num_classes=num_classes)
    spec = cfg["benchmark"]["models"][name]
    return timm.create_model(spec["timm"], pretrained=pretrained, num_classes=num_classes)


def model_dir(cfg: dict, name: str) -> str:
    """Folder under models/ for a model name (baselines from benchmark.models, the proposed model from proposed.dir)."""
    return cfg["proposed"]["dir"] if name == "Proposed" else cfg["benchmark"]["models"][name]["dir"]


def normalisation(model: nn.Module) -> tuple[tuple, tuple]:
    """Each pretrained model's own mean/std (part of its weight definition)."""
    c = model.pretrained_cfg
    return tuple(c["mean"]), tuple(c["std"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def head_and_backbone(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Head = the classifier layer returned by get_classifier(); backbone = every other parameter,
    in registration order (early layers first)."""
    if hasattr(model, "new_layer_params"):  # dual-branch model: everything new is the "head", only the pretrained branch is frozen
        return model.new_layer_params(), model.pretrained_params()
    head = list(model.get_classifier().parameters())
    ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in ids]
    return head, backbone


def set_stage1(model: nn.Module) -> None:
    """Backbone frozen, head trainable."""
    head, backbone = head_and_backbone(model)
    for p in backbone:
        p.requires_grad = False
    for p in head:
        p.requires_grad = True


def set_stage2(model: nn.Module, fraction: float) -> dict:
    """Unfreeze the last `fraction` of backbone parameter tensors (one generic rule for every
    architecture); the rest stay frozen. Head stays trainable."""
    head, backbone = head_and_backbone(model)
    n_unfreeze = max(1, math.ceil(fraction * len(backbone)))
    for p in backbone:
        p.requires_grad = False
    for p in backbone[len(backbone) - n_unfreeze:]:
        p.requires_grad = True
    for p in head:
        p.requires_grad = True
    return {"backbone_tensors": len(backbone), "unfrozen_tensors": n_unfreeze,
            "unfrozen_params": sum(p.numel() for p in backbone[len(backbone) - n_unfreeze:]),
            "backbone_params": sum(p.numel() for p in backbone)}


def set_train_mode(model: nn.Module) -> None:
    """model.train(), except modules whose parameters are all frozen stay in eval mode, so frozen
    BatchNorm layers keep their pretrained running statistics."""
    model.train()
    for m in model.modules():
        params = list(m.parameters(recurse=False))
        if params and not any(p.requires_grad for p in params):
            m.eval()
        elif not params and isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def param_groups(model: nn.Module, lr_head: float, lr_backbone: float | None, wd_head: float,
                 wd_backbone: float) -> list[dict]:
    """AdamW groups over the currently trainable parameters."""
    head, backbone = head_and_backbone(model)
    groups = [{"params": [p for p in head if p.requires_grad], "lr": lr_head, "weight_decay": wd_head, "name": "head"}]
    bb = [p for p in backbone if p.requires_grad]
    if bb:
        groups.append({"params": bb, "lr": lr_backbone, "weight_decay": wd_backbone, "name": "backbone"})
    return groups
