"""Shared helpers: config loading, seeding, path resolution."""
from __future__ import annotations

import os
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for an entry of cfg['paths']."""
    return ROOT / cfg["paths"][key]


def class_by_slug(cfg: dict) -> dict:
    return {c["slug"]: c for c in cfg["classes"]}


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
