"""FLOP counting on layers with known cost, and the latency summary."""
import os
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import efficiency as ef  # noqa: E402


def test_flop_counter_on_known_layers():
    conv = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))
    f = ef.count_flops(conv, size=8)
    assert abs(f * 1e9 - 2 * 3 * 8 * 9 * 8 * 8) < 1  # 2 x MACs = 27,648
    lin = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
    assert abs(ef.count_flops(lin, size=4) * 1e9 - 2 * 48 * 10) < 1


def test_latency_summary_is_ordered_and_positive():
    r = ef.latency(nn.Conv2d(3, 4, 3), "cpu", batch=2, warmup=2, iters=10, size=16)
    assert 0 < r["q25_ms"] <= r["median_ms"] <= r["q75_ms"]


def test_proposed_model_has_more_flops_than_its_rgb_branch_alone():
    import runs
    from common import load_config

    cfg = load_config()
    a = ef.count_flops(runs.build_run_model(cfg, "Abl-A_s42", pretrained=False))
    d = ef.count_flops(runs.build_run_model(cfg, "Abl-D_s42", pretrained=False))
    b = ef.count_flops(runs.build_run_model(cfg, "Abl-B_s42", pretrained=False))
    assert d > a > 1 and b < a


def test_machine_state_keys():
    s = ef.machine_state()
    assert {"busy_processes", "on_ac_power", "torch"} <= set(s)
