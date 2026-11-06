"""Registry of every model that enters the final analyses.

Main comparison: Proposed, ConvNeXt-Tiny and EfficientNetV2-S, seeds 42-44.
Ablation (Proposed code with flag overrides), seeds 42-44:
  Abl-A    RGB branch only, pretrained backbone frozen (stage 1 only)
  Abl-B    colour/texture branch only (nothing pretrained, stage 1 only)
  Abl-C    both branches, concatenated (stage 1 only)
  Abl-D    both branches, gated fusion (stage 1 only)
  Abl-A-FT RGB branch only with the full two-stage recipe (the fully fine-tuned control)
  E        is the main-comparison Proposed model (gated fusion, two-stage recipe)

  python code/runs.py list [--todo]     run ids in training order
  python code/runs.py status            one line per run: trained or not
Seed-42 runs of the three main models already exist from Deliverables 2 to 4 and keep their original paths.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

SEEDS = (42, 43, 44)
MAIN = ("Proposed", "ConvNeXt-Tiny", "EfficientNetV2-S")
EXTRA = ("ResNet50", "DenseNet121", "MobileNetV2")  # seed 42 only: evaluated, not retrained
ABLATION = {
    "Abl-A": dict(overrides=dict(use_rgb=True, use_ct=False, fusion="none"), stage2=False, label="A: RGB only, frozen backbone"),
    "Abl-B": dict(overrides=dict(use_rgb=False, use_ct=True, fusion="none"), stage2=False, label="B: colour/texture only"),
    "Abl-C": dict(overrides=dict(use_rgb=True, use_ct=True, fusion="concat"), stage2=False, label="C: both branches, concatenated"),
    "Abl-D": dict(overrides=dict(use_rgb=True, use_ct=True, fusion="gated"), stage2=False, label="D: both branches, gated"),
    "Abl-A-FT": dict(overrides=dict(use_rgb=True, use_ct=False, fusion="none"), stage2=True, label="A-FT: RGB only, fully fine-tuned"),
}


def run_id(base: str, seed: int) -> str:
    return f"{base}_s{seed}"


def parse(rid: str) -> tuple[str, int]:
    base, _, s = rid.rpartition("_s")
    return base, int(s)


def spec(cfg: dict, rid: str) -> dict:
    """Everything needed to train, find and evaluate one run."""
    base, seed = parse(rid)
    from baselines import model_dir

    if base in MAIN or base in EXTRA:
        legacy = seed == 42
        d = dict(run_id=rid, base=base, group="main" if base in MAIN else "extra", model=base, overrides={}, seed=seed, s2_epochs=None, legacy=legacy,
                 label=base)
        if legacy:  # trained in Deliverables 2 to 4: original locations
            d.update(ckpt=ROOT / "models" / model_dir(cfg, base) / "best.pt", hist=ROOT / "results" / "baseline_history" / f"{base}.json",
                     log_tag=base)
    elif base in ABLATION:
        a = ABLATION[base]
        d = dict(run_id=rid, base=base, group="ablation", model="Proposed", overrides=dict(a["overrides"]), seed=seed,
                 s2_epochs=None if a["stage2"] else 0, legacy=False, label=a["label"])
    else:
        raise KeyError(rid)
    d.setdefault("ckpt", ROOT / "models" / "runs" / rid / "best.pt")
    d.setdefault("hist", ROOT / "results" / "run_history" / f"{rid}.json")
    d.setdefault("log_tag", rid)
    d["out_dir"] = d["ckpt"].parent
    d["final"] = ROOT / "results" / "final" / f"{rid}.npz"  # all test-set outputs of one checkpoint, written once
    return d


def all_run_ids() -> list[str]:
    """Training order: the main runs of seeds 43 and 44 first (they carry the headline claim), then the fine-tuned control,
    then the stage-1 ablation variants."""
    ids = [run_id(m, s) for s in SEEDS for m in MAIN]
    ids += [run_id("Abl-A-FT", s) for s in SEEDS]
    ids += [run_id(b, s) for s in SEEDS for b in ("Abl-D", "Abl-A", "Abl-C", "Abl-B")]
    return ids


def eval_run_ids() -> list[str]:
    """Every checkpoint that gets a test pass: the three main models x seeds, the three extra baselines (seed 42), the ablation."""
    ids = [run_id(m, s) for s in SEEDS for m in MAIN] + [run_id(m, 42) for m in EXTRA]
    return ids + [run_id(b, s) for b in ABLATION for s in SEEDS]


def is_trained(cfg: dict, rid: str) -> bool:
    s = spec(cfg, rid)
    if not s["ckpt"].exists() or not s["hist"].exists():
        return False
    return "total_epochs" in json.loads(s["hist"].read_text())


def build_run_model(cfg: dict, rid: str, pretrained: bool = True):
    from baselines import build_model

    s = spec(cfg, rid)
    return build_model(cfg, s["model"], pretrained=pretrained, overrides=s["overrides"])


def load_run_model(cfg: dict, rid: str, device):
    """Frozen checkpoint of a run, in eval mode on `device`."""
    import torch

    s = spec(cfg, rid)
    m = build_run_model(cfg, rid, pretrained=False)
    m.load_state_dict(torch.load(s["ckpt"], map_location="cpu", weights_only=False)["state_dict"])
    return m.to(device).eval(), s["ckpt"]


def main() -> None:
    cfg = load_config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    ids = all_run_ids()
    if cmd == "list":
        for r in ids:
            if "--todo" in sys.argv and is_trained(cfg, r):
                continue
            print(r)
    elif cmd == "status":
        for r in ids:
            print(f"{r:22s} {'trained' if is_trained(cfg, r) else 'todo'}")


if __name__ == "__main__":
    main()
