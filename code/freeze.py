"""Freeze manifests: SHA-256 of files that must not change, plus a hash of the recipe that produced them.

  baselines  (results/FREEZE.json)           the benchmark split, the five baseline checkpoints, their saved
                                             test predictions, the benchmark table and the benchmark recipe
  proposed   (results/FREEZE_proposed.json)  the proposed model's checkpoint, history, model code, recipe, and
                                             (once they exist) its saved test predictions and robustness results

  python code/freeze.py make [baselines|proposed]   write the manifest (default: baselines); checkpoints become read-only
  python code/freeze.py check                       recompute every existing manifest and compare (used by tests/test_freeze.py)

Re-making the proposed manifest after new results exist (for example after Deliverable 6) is a deliberate act: it
records a new date and the larger file list.
"""
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

FREEZE = ROOT / "results" / "FREEZE.json"          # kept for backwards compatibility with the tests
MANIFESTS = {"baselines": FREEZE, "proposed": ROOT / "results" / "FREEZE_proposed.json"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def baseline_files(cfg: dict) -> list[Path]:
    b = ROOT / cfg["benchmark"]["dir"]
    files = [b / f"{n}.csv" for n in ("train", "validation", "test")] + [b / "classes.json"]
    files += [ROOT / "models" / m["dir"] / "best.pt" for m in cfg["benchmark"]["models"].values()]
    files += sorted((ROOT / "results" / "test_predictions").glob("*"))
    files += [ROOT / "results" / "benchmark.csv", ROOT / "results" / "baseline_val.csv"]
    return files


def proposed_files(cfg: dict) -> list[Path]:
    files = [ROOT / "models" / cfg["proposed"]["dir"] / "best.pt", ROOT / "results" / "baseline_history" / "Proposed.json",
             ROOT / "code" / "proposed_model.py"]
    optional = [ROOT / "results" / "test_predictions" / "Proposed.npz", ROOT / "results" / "robustness.csv",
                ROOT / "results" / "robustness_all_models.csv", ROOT / "results" / "robustness_comparisons.csv"]
    return files + [p for p in optional if p.exists()]


def recipe_hash(cfg: dict, which: str = "baselines") -> str:
    """Hash of everything that defines the recipe and split (not of unrelated config sections)."""
    if which == "proposed":
        return _hash({"seed": cfg["seed"], "proposed": cfg["proposed"], "benchmark": cfg["benchmark"]})
    return _hash({"seed": cfg["seed"], "split_ratios": cfg["split_ratios"], "benchmark": cfg["benchmark"]})


def compute(cfg: dict, which: str = "baselines") -> dict:
    files = baseline_files(cfg) if which == "baselines" else proposed_files(cfg)
    return {"files": {str(p.relative_to(ROOT)): sha256(p) for p in files}, "recipe_sha256": recipe_hash(cfg, which)}


def make(which: str = "baselines") -> None:
    cfg = load_config()
    m = compute(cfg, which)
    m["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    m["note"] = ("D1/D2 split and the five baseline models are frozen. Do not retrain or re-split without a new manifest."
                 if which == "baselines" else
                 "The proposed model (seed 42) is frozen. A redesign is a separately labelled v2, never a silent retrain.")
    MANIFESTS[which].write_text(json.dumps(m, indent=1))
    ckpts = ([ROOT / "models" / s["dir"] / "best.pt" for s in cfg["benchmark"]["models"].values()] if which == "baselines"
             else [ROOT / "models" / cfg["proposed"]["dir"] / "best.pt"])
    for c in ckpts:  # checkpoints read-only
        os.chmod(c, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"froze {len(m['files'])} files + recipe hash -> {MANIFESTS[which]}")


def check_one(which: str) -> list[str]:
    cfg = load_config()
    old = json.loads(MANIFESTS[which].read_text())
    new = compute(cfg, which)
    bad = [f for f, h in old["files"].items() if new["files"].get(f) != h]
    if old["recipe_sha256"] != new["recipe_sha256"]:
        bad.append(f"{which} recipe (config.yaml)")
    return bad


def check() -> list[str]:
    """Problems across every manifest that exists (the baselines manifest is required)."""
    bad = check_one("baselines")
    for which, path in MANIFESTS.items():
        if which != "baselines" and path.exists():
            bad += [f"[{which}] {b}" for b in check_one(which)]
    return bad


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "make":
        make(sys.argv[2] if len(sys.argv) > 2 else "baselines")
    else:
        problems = check()
        print("FROZEN FILES UNCHANGED" if not problems else f"CHANGED: {problems}")
        sys.exit(1 if problems else 0)
