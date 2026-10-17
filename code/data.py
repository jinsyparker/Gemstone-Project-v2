"""Benchmark data layer: cached square-crop images, identical transforms for every model.

Images come from dataset/_cache/img256/<split>.npy (uint8, N x 256 x 256 x 3), row-aligned with
dataset/benchmark/<split>.csv. The test split can only be loaded with allow_test=True; only
evaluate.py passes it (and logs the access).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402


class HeldOutSplitError(RuntimeError):
    pass


def bench_dir(cfg: dict) -> Path:
    return ROOT / cfg["benchmark"]["dir"]


def class_names(cfg: dict) -> list[str]:
    import json

    return json.loads((bench_dir(cfg) / "classes.json").read_text())


class _Rot90:
    """Random multiple of 90 degrees (no interpolation artefacts); gems have no canonical orientation."""

    def __call__(self, x):
        return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))


def build_transforms(cfg: dict, train: bool, mean, std):
    b = cfg["benchmark"]
    size = b["input_size"]
    norm = v2.Normalize(mean=list(mean), std=list(std))
    if not train:
        return v2.Compose([v2.Resize((size, size), antialias=True), v2.ToDtype(torch.float32, scale=True), norm])
    a = b["aug"]
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=tuple(a["scale"]), ratio=tuple(a["ratio"]), antialias=True),
        v2.RandomHorizontalFlip(0.5),
        _Rot90(),
        v2.ToDtype(torch.float32, scale=True),
        # brightness/contrast only: hue and saturation are diagnostic for gemstones
        v2.ColorJitter(brightness=a["jitter"], contrast=a["jitter"]),
        norm,
    ])


class CachedDataset(Dataset):
    def __init__(self, cfg: dict, split: str, mean, std, train: bool = False, allow_test: bool = False,
                 limit: int | None = None, variant: str = "original"):
        if split == "test" and not allow_test:
            raise HeldOutSplitError("the test split may only be loaded by evaluate.py --final (allow_test=True)")
        size = cfg["benchmark"]["cache_size"]
        self.df = pd.read_csv(bench_dir(cfg) / f"{split}.csv")
        if variant == "original":
            src = ROOT / cfg["audit"]["cache"] / f"img{size}" / f"{split}.npy"
        else:  # fg_only / bg_only / swap_*: built by background_analysis.py variants
            src = ROOT / cfg["audit"]["cache"] / "variants" / f"{variant}_{split}.npy"
        self.x = np.load(src, mmap_mode="r")
        assert len(self.x) == len(self.df), "cache and CSV are out of sync: rebuild with preprocessing.py cache"
        self.y = self.df["class_idx"].to_numpy(dtype=np.int64)
        if limit:  # smoke tests: a class-balanced prefix
            keep = self.df.groupby("class_idx", group_keys=False).head(max(1, limit // self.df["class_idx"].nunique())).index
            self.keep = np.array(sorted(keep))
        else:
            self.keep = np.arange(len(self.df))
        self.tf = build_transforms(cfg, train, mean, std)

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, i: int):
        j = int(self.keep[i])
        img = torch.from_numpy(np.array(self.x[j])).permute(2, 0, 1)  # uint8 CHW
        return self.tf(img), int(self.y[j])


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=num_workers,
                      drop_last=False, persistent_workers=False)


def load_cache_array(cfg: dict, split: str, allow_test: bool = False, mmap: bool = True) -> np.ndarray:
    """The cached uint8 crops of a split, row-aligned with dataset/benchmark/<split>.csv. The test split is
    guarded exactly like CachedDataset: it needs allow_test=True (only evaluation code passes it)."""
    if split == "test" and not allow_test:
        raise HeldOutSplitError("the test split may only be loaded by evaluation code (allow_test=True)")
    size = cfg["benchmark"]["cache_size"]
    return np.load(ROOT / cfg["audit"]["cache"] / f"img{size}" / f"{split}.npy", mmap_mode="r" if mmap else None)


class ArrayDataset(Dataset):
    """In-memory uint8 images (N,H,W,3) and labels with the standard evaluation transform. Used by the robustness
    study, where the crops are perturbed before the unchanged transform (resize, normalise)."""

    def __init__(self, cfg: dict, x: np.ndarray, y: np.ndarray, mean, std):
        self.x, self.y = x, np.asarray(y, dtype=np.int64)
        self.tf = build_transforms(cfg, False, mean, std)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.tf(torch.from_numpy(np.array(self.x[i])).permute(2, 0, 1)), int(self.y[i])
