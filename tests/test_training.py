"""Logic and smoke tests for the benchmark pipeline (D2). Runs on CPU so it is deterministic and does
not compete with a training run for the GPU. Only train/validation images are ever read."""
import copy
import os
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import baselines as bl  # noqa: E402
import evaluate as ev  # noqa: E402
import train as tr  # noqa: E402
from common import ROOT, load_config  # noqa: E402
from data import CachedDataset, HeldOutSplitError  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def tiny_model(cfg):
    torch.manual_seed(0)
    return bl.build_model(cfg, "MobileNetV2", pretrained=False)


def test_benchmark_lists_match_frozen_split(cfg):
    bd = ROOT / cfg["benchmark"]["dir"]
    for name in ("train", "validation", "test"):
        d1 = pd.read_csv(ROOT / "dataset" / f"{name}.csv")
        b = pd.read_csv(bd / f"{name}.csv")
        assert "White Sapphire" not in set(b["label"])
        assert set(b["image_id"]) == set(d1[d1["label"] != "White Sapphire"]["image_id"])  # same split as D1
        assert sorted(b["class_idx"].unique()) == list(range(10))
    sizes = [len(pd.read_csv(bd / f"{n}.csv")) for n in ("train", "validation", "test")]
    assert sizes == [3518, 755, 751]
    allb = pd.concat([pd.read_csv(bd / f"{n}.csv") for n in ("train", "validation", "test")])
    assert (allb.groupby("group_id")["split"].nunique() == 1).all()  # still no stone across splits


def test_test_split_cannot_be_loaded_by_training_code(cfg):
    m = tiny_model(cfg)
    mean, std = bl.normalisation(m)
    with pytest.raises(HeldOutSplitError):
        CachedDataset(cfg, "test", mean, std)


def test_freezing_stages(cfg):
    m = tiny_model(cfg)
    head, backbone = bl.head_and_backbone(m)
    bl.set_stage1(m)
    assert not any(p.requires_grad for p in backbone) and all(p.requires_grad for p in head)
    before = [p.detach().clone() for p in backbone]
    opt = torch.optim.AdamW(bl.param_groups(m, 1e-3, None, 0.01, 0.05))
    bl.set_train_mode(m)
    loss = m(torch.randn(4, 3, 64, 64)).sum()
    loss.backward()
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, backbone))     # backbone untouched in stage 1
    assert not any(mod.training for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d))  # frozen BN in eval
    info = bl.set_stage2(m, 0.5)
    frac = sum(p.requires_grad for p in backbone) / len(backbone)
    assert abs(frac - 0.5) < 0.02 and info["unfrozen_tensors"] == sum(p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in head)
    assert [p.requires_grad for p in backbone][-1] and not [p.requires_grad for p in backbone][0]  # top half unfrozen


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 10)).astype(np.float32)
    y = rng.integers(0, 10, 200)
    m = ev.full_metrics(logits, y)
    assert m["macro_f1"] == pytest.approx(f1_score(y, logits.argmax(1), average="macro"))
    assert m["accuracy"] == pytest.approx((logits.argmax(1) == y).mean())
    assert m["loss"] == pytest.approx(float(torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y))))


def test_lr_schedule_warmup_then_cosine():
    f = tr.lr_lambda(total_steps=100, warmup_steps=10)
    assert f(0) < f(9) <= 1.0 and f(10) == pytest.approx(1.0)
    assert f(55) < f(10) and f(99) < 0.01


def run_tiny(cfg, tmp, **kw):
    m = tiny_model(cfg)
    return tr.train_model(cfg, m, tmp / "ck", tmp / "hist.json", seed=42, limit=40, name="tiny", **kw)


def test_two_stage_training_smoke_checkpoint_and_determinism(cfg, tmp_path):
    h1 = run_tiny(cfg, tmp_path / "a", s1_epochs=1, s2_epochs=1)
    assert [r["stage"] for r in h1["epochs"]] == [1, 2]
    assert (tmp_path / "a" / "ck" / "best.pt").exists() and (tmp_path / "a" / "hist.json").exists()
    h2 = run_tiny(cfg, tmp_path / "b", s1_epochs=1, s2_epochs=1)
    assert h1["epochs"][0]["train_loss"] == pytest.approx(h2["epochs"][0]["train_loss"], rel=1e-5)  # seeded
    assert h1["epochs"][1]["val_macro_f1"] == pytest.approx(h2["epochs"][1]["val_macro_f1"])


def test_resume_continues_where_it_stopped(cfg, tmp_path):
    full = run_tiny(cfg, tmp_path / "full", s1_epochs=1, s2_epochs=2)
    part = tmp_path / "part"
    run_tiny(cfg, part, s1_epochs=1, s2_epochs=1)             # stops after epoch 2 of 3
    resumed = tr.train_model(cfg, tiny_model(cfg), part / "ck", part / "hist.json", seed=42, limit=40,
                             s1_epochs=1, s2_epochs=2, resume=True, name="tiny")
    assert len(resumed["epochs"]) == 3
    # the stage-2 LR schedule was defined over 1 epoch in the first run, so only check structure + stage tags
    assert [r["stage"] for r in resumed["epochs"]] == [1, 2, 2] == [r["stage"] for r in full["epochs"]]


def test_early_stopping_in_stage_two(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["benchmark"]["stage2"]["patience"] = 1
    c["benchmark"]["stage2"]["min_delta"] = 10.0              # nothing can ever count as an improvement
    h = tr.train_model(c, tiny_model(c), tmp_path / "ck", tmp_path / "h.json", seed=42, limit=40,
                       s1_epochs=1, s2_epochs=5, name="tiny")
    assert h["stopped_early_at"] == 2 and len(h["epochs"]) == 2
