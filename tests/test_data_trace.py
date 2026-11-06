"""The image trace must reconcile row by row with the dataset files, and the benchmark lists must equal the trace."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import data_trace as dt  # noqa: E402
from common import load_config  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    cfg = load_config()
    t = dt.load_tables(cfg)
    pc = dt.per_class(t, cfg)
    return cfg, t, pc, dt.trace(pc)


def test_roles_sum_to_listed_and_downloaded_to_kept(tables):
    _, t, pc, _ = tables
    assert (pc.hand_shots + pc.certificates + pc.other_files + (pc.downloaded)).eq(pc.listed).all()
    assert (pc.downloaded == pc.kept + pc.removed_promo + pc.removed_duplicate).all()
    assert len(t["scraped"]) == pc.listed.sum()


def test_split_counts_add_up_per_class(tables):
    _, _, pc, _ = tables
    assert (pc.train + pc.validation + pc.test == pc.kept).all()


def test_trace_steps_chain(tables):
    _, _, _, tr = tables
    n = dict(zip(tr.stage, tr.images))
    assert n["New scrape: gallery photos listed"] + n["Excluded by role (hand shots, certificates, other)"] == n["Loose-stone photos downloaded"]
    assert n["Loose-stone photos downloaded"] + n["Cleaning and deduplication"] == n["After cleaning (11 classes)"]
    assert n["After cleaning (11 classes)"] + n["White Sapphire removed"] == n["Benchmark (10 classes)"]


def test_benchmark_matches_derived_lists(tables):
    cfg, t, pc, tr = tables
    b = t["bench"]
    n = dict(zip(tr.stage, tr.images))
    assert n["Benchmark (10 classes)"] == len(b) == 5024
    bench = pc[pc.in_benchmark]
    for split in ("train", "validation", "test"):
        assert bench[split].sum() == (b.split == split).sum()
    assert dict(zip(tr.stage, tr.stones))["Benchmark (10 classes)"] == b.group_id.nunique()
    assert "White Sapphire" not in set(b.label)


def test_earlier_paper_table_sums_to_published_total():
    assert sum(dt.EARLIER_PAPER.values()) == 3182
    assert dt.EARLIER_SPLIT["train"] + dt.EARLIER_SPLIT["validation"] == 3182


def test_backdrop_stats_flag_white_sapphire():
    cfg = load_config()
    qc = ROOT / cfg["audit"]["cache"] / "qc.csv"
    if not qc.exists():
        pytest.skip("image cache not present (qc.csv is regenerable and not committed)")
    t = dt.load_tables(cfg)
    stats, summary = dt.backdrop_stats(t, cfg, qc)
    assert stats.loc["White Sapphire", "black_pct"] > 90
    assert stats.drop(index="White Sapphire").black_pct.max() < 5
    assert stats.loc["White Sapphire", "backdrop_only_recall_11"] > 0.9
    assert summary.loc["10 classes", "accuracy"] < summary.loc["11 classes", "accuracy"]
