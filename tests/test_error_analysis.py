"""Error-analysis tables on synthetic outputs with a planted confusion."""
import os
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import error_analysis as ea  # noqa: E402
import final_tables as ft  # noqa: E402
import runs  # noqa: E402
from test_final_tables import K, PER, STONES, fake_outputs  # noqa: E402


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    d = tmp_path_factory.mktemp("ea")
    groups = np.repeat(np.arange(STONES), PER)
    y = np.repeat(np.arange(STONES) % K, PER)
    for rid in runs.eval_run_ids():
        base, seed = runs.parse(rid)
        z = fake_outputs(rid, y, groups, 0.04, 0.06, seed * 17 + len(base))
        if base == "Proposed":  # plant: half of class 0 is called class 3
            p = z["bg|original|logits"].argmax(1)
            idx = np.flatnonzero(y == 0)
            p[idx[: len(idx) // 2]] = 3
            z["bg|original|logits"] = np.eye(K)[p] * 5
        np.savez(d / f"{rid}.npz", **z)
    return ft.Store("test", directory=d, n_draws=60)


def test_confusion_and_prf_basics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    cm = ea.confusion(y, p, 2)
    assert cm.tolist() == [[1, 1], [0, 2]]
    m = ea.prf(cm)
    assert m[1].tolist() == [0.5, 1.0] and m[0].tolist() == [1.0, 2 / 3]


def test_planted_confusion_is_the_top_pair(store):
    pairs = ea.top_pairs(store, "Proposed")
    top = pairs.iloc[0]
    assert (top["true"], top["predicted"]) == (store.classes[0], store.classes[3])
    assert top.images_mean_over_seeds >= len(np.flatnonzero(store.y == 0)) // 2 - 1
    assert top.stones_mean_over_seeds > 0


def test_class_metrics_shape_and_recall_drop(store):
    m = ea.class_metrics(store, "Proposed")
    assert len(m) == K * 3 and set(m.metric) == {"precision", "recall", "f1"}
    rec0 = m[(m["class"] == store.classes[0]) & (m.metric == "recall")].iloc[0]
    assert rec0.value < 0.6 and rec0.ci_lo <= rec0.value <= rec0.ci_hi
    other = m[(m["class"] == store.classes[5]) & (m.metric == "recall")].iloc[0]
    assert other.value > 0.85


def test_extra_baselines_use_seed_42_only(store):
    assert ea.rids_of(store, "ResNet50") == ["ResNet50_s42"]
    assert len(ea.rids_of(store, "Proposed")) == 3
