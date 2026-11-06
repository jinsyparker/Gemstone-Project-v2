"""Parameters, FLOPs and latency of every model under one documented protocol (docs/efficiency_protocol.md).

  python code/efficiency.py            measure everything (needs an idle, plugged-in machine); writes results/efficiency.csv
  python code/efficiency.py --check    only report whether the machine is idle

FLOPs: torch.utils.flop_counter.FlopCounterMode on one 224 x 224 image (2 x multiply-accumulates of convolutions and matrix
products; element-wise operations, pooling and the proposed model's Sobel/Laplacian/HSV arithmetic are not counted, which
understates the colour/texture branch slightly). Latency: median and inter-quartile range of per-image time over 200 timed
forward passes after 20 warm-up passes, fp32, inference mode, synthetic input.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

WARMUP, ITERS, SIZE, CPU_THREADS = 20, 200, 224, 4
# (display name, registry run id): six main models, then the ablation variants
MODELS = [("Proposed (E)", "Proposed_s42"), ("ConvNeXt-Tiny", "ConvNeXt-Tiny_s42"), ("EfficientNetV2-S", "EfficientNetV2-S_s42"),
          ("ResNet50", "ResNet50_s42"), ("DenseNet121", "DenseNet121_s42"), ("MobileNetV2", "MobileNetV2_s42"),
          ("A  RGB only", "Abl-A_s42"), ("B  colour/texture only", "Abl-B_s42"), ("C  both, concatenated", "Abl-C_s42"), ("D  both, gated", "Abl-D_s42")]


def count_flops(model: torch.nn.Module, size: int = SIZE) -> float:
    """Billions of FLOPs (2 x MACs) for one image."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.randn(1, 3, size, size)
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def latency(model: torch.nn.Module, device: str, batch: int, warmup: int = WARMUP, iters: int = ITERS, size: int = SIZE) -> dict:
    """Median and quartiles of milliseconds per image."""
    model = model.to(device).eval()
    x = torch.randn(batch, 3, size, size, device=device)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    ts = []
    with torch.inference_mode():
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            model(x)
            sync()
            if i >= warmup:
                ts.append((time.perf_counter() - t0) / batch * 1000)
    q = np.quantile(ts, [0.25, 0.5, 0.75])
    return {"median_ms": float(q[1]), "q25_ms": float(q[0]), "q75_ms": float(q[2])}


def machine_state() -> dict:
    busy = subprocess.run("ps -axo command | grep -E 'train.py|final_eval.py|run_queue.sh|robustness.py' | grep -v grep", shell=True,
                          capture_output=True, text=True).stdout.strip()
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout if platform.system() == "Darwin" else ""
    return {"busy_processes": busy.splitlines(), "on_ac_power": "AC Power" in batt, "platform": platform.platform(), "torch": torch.__version__,
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def measure(cfg: dict) -> pd.DataFrame:
    import runs

    torch.set_num_threads(CPU_THREADS)
    rows = []
    for name, rid in MODELS:
        m = runs.build_run_model(cfg, rid, pretrained=False).eval()
        total = sum(p.numel() for p in m.parameters())
        pre = sum(p.numel() for p in m.pretrained_params()) if hasattr(m, "pretrained_params") else total
        row = {"model": name, "params_M": total / 1e6, "pretrained_params_M": pre / 1e6, "new_params_M": (total - pre) / 1e6,
               "gflops_per_image": count_flops(m)}
        for dev, batch in (("mps", 1), ("mps", 32), ("cpu", 1)):
            if dev == "mps" and not torch.backends.mps.is_available():
                continue
            r = latency(m.to(dev), dev, batch)
            row.update({f"{dev}_b{batch}_{k}": v for k, v in r.items()})
        rows.append(row)
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k != "model"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="measure although the machine is busy (the result is then labelled)")
    a = ap.parse_args()
    state = machine_state()
    print(state)
    if a.check:
        sys.exit(0)
    if (state["busy_processes"] or not state["on_ac_power"]) and not a.force:
        raise SystemExit("machine is not idle and plugged in; stop training/evaluation first (or --force, which labels the result)")
    df = measure(load_config())
    df["machine_busy"] = bool(state["busy_processes"])
    df.to_csv(ROOT / "results" / "efficiency.csv", index=False)
    (ROOT / "results" / "efficiency_machine.json").write_text(__import__("json").dumps(state, indent=1))
    print("wrote results/efficiency.csv")
