"""One test pass per checkpoint, producing every pre-registered output (results/PREREGISTRATION.md, section 3).

  python code/final_eval.py --run Proposed_s43 --split validation [--limit 48]   development on validation images
  python code/final_eval.py --run Proposed_s43 --split test --test               the single logged test pass
  python code/final_eval.py --all --split test --test                            every checkpoint not yet done

Outputs (one .npz per checkpoint): results/final/<run>.npz (test) or results/final_validation/<run>.npz:
  y, groups, sha                         labels, stone ids, checkpoint hash
  bg|<variant>|logits                    original, fg_only, bg_only, swap_black, swap_white, swap_other
  rob|<condition>|pred                   the 18 perturbation conditions (rotation: both signs concatenated)
  cam|<variant>|<stat>                   Grad-CAM scores against the stone masks, variants original and swap_white
The test split is only read when every run is trained and frozen (FREEZE_runs.json), the checkpoint is part of a manifest and
its hash matches, and `--test` is given. Each pass is logged once in results/test_access_log.json. An existing output file is
never overwritten: tables are rebuilt from it without reading the test images again.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

log = logging.getLogger("final_eval")
VARIANTS = ["original", "fg_only", "bg_only", "swap_black", "swap_white", "swap_other"]
CAM_VARIANTS = ("original", "swap_white")


def out_path(rid: str, split: str) -> Path:
    return ROOT / "results" / ("final" if split == "test" else "final_validation") / f"{rid}.npz"


def manifest_hashes() -> dict[str, str]:
    """Every file hash recorded in any freeze manifest."""
    h: dict[str, str] = {}
    for p in (ROOT / "results").glob("FREEZE*.json"):
        h.update(json.loads(p.read_text())["files"])
    return h


def assert_ready_for_test(cfg: dict, rid: str) -> None:
    """The pre-registered conditions for reading the test set."""
    import runs
    from evaluate import sha256_file

    h = manifest_hashes()
    todo = [r for r in runs.all_run_ids() if not runs.is_trained(cfg, r)]
    if todo:
        raise SystemExit(f"not every run is trained yet ({len(todo)} left, e.g. {todo[0]}); the test set stays closed")
    for r in runs.all_run_ids():
        ck = runs.spec(cfg, r)["ckpt"]
        if h.get(str(ck.relative_to(ROOT))) != sha256_file(ck):
            raise SystemExit(f"{r}: checkpoint is not frozen in a manifest (or its hash changed); run freeze.py make runs first")
    ck = runs.spec(cfg, rid)["ckpt"]
    if h.get(str(ck.relative_to(ROOT))) != sha256_file(ck):
        raise SystemExit(f"{rid}: checkpoint not frozen")


def evaluate_checkpoint(cfg: dict, rid: str, split: str, device, limit: int | None = None) -> dict[str, np.ndarray]:
    import gradcam
    import robustness as rb
    import runs
    from baselines import normalisation
    from data import CachedDataset, load_cache_array, make_loader
    from evaluate import sha256_file
    from train import predict

    allow = split == "test"
    meta = pd.read_csv(ROOT / cfg["benchmark"]["dir"] / f"{split}.csv")
    keep = CachedDataset(cfg, split, (0, 0, 0), (1, 1, 1), train=False, allow_test=allow, limit=limit).keep  # class-balanced prefix when limited
    meta = meta.iloc[keep]
    y, groups = meta["class_idx"].to_numpy(), meta["group_id"].to_numpy()
    model, ck = runs.load_run_model(cfg, rid, device)
    mean, std = normalisation(model)
    out: dict[str, np.ndarray] = {"y": y, "groups": groups.astype(str), "sha": np.array(sha256_file(ck))}
    for v in VARIANTS:
        ds = CachedDataset(cfg, split, mean, std, train=False, allow_test=allow, limit=limit, variant=v)
        lg, yy = predict(model, make_loader(ds, 64, False, 0), device)
        assert (yy == y).all()
        out[f"bg|{v}|logits"] = lg.astype(np.float32)
    x = np.array(load_cache_array(cfg, split, allow_test=allow))[keep]
    for cname, c in rb.COND.items():
        out[f"rob|{cname}|pred"] = rb.predict_views(cfg, model, c["fn"](x), y, device).argmax(1).astype(np.int8)
    st = gradcam.split_stats(cfg, model, split, CAM_VARIANTS, device, allow_test=allow, limit=limit)
    for v, d in st.items():
        for k, a in d.items():
            out[f"cam|{v}|{k}"] = a.astype(np.float32 if a.dtype.kind == "f" else np.int16)
    return out


def regression_check(cfg: dict, rid: str, out: dict) -> dict:
    """Seed-42 checkpoints were evaluated before: the new pass must reproduce the stored clean logits and robustness predictions."""
    import runs

    s = runs.spec(cfg, rid)
    if not s["legacy"]:
        return {}
    base = s["base"]
    res = {}
    old = np.load(ROOT / "results" / "test_predictions" / f"{base}.npz")["logits"]
    res["clean_max_abs_logit_diff"] = float(np.abs(old - out["bg|original|logits"]).max())
    res["clean_pred_identical"] = bool((old.argmax(1) == out["bg|original|logits"].argmax(1)).all())
    rp = ROOT / "results" / "robustness_predictions.npz"
    if rp.exists():
        z = np.load(rp)
        agree = [float((z[f"{base}|{c}"] == out[f"rob|{c}|pred"]).mean()) for c in __import__("robustness").COND
                 if f"{base}|{c}" in z.files]
        res["robustness_min_agreement"] = min(agree) if agree else None
    return res


def run_one(cfg: dict, rid: str, split: str, test_flag: bool, limit: int | None) -> None:
    from evaluate import log_test_access, sha256_file
    from train import get_device

    import runs

    path = out_path(rid, split)
    if path.exists() and not limit:
        log.info("%s: %s exists, not re-reading", rid, path.name)
        return
    if split == "test":
        if not test_flag:
            raise SystemExit("this reads the test set; re-run with --test to confirm")
        assert_ready_for_test(cfg, rid)
        if limit:
            raise SystemExit("--limit is for validation development only")
    device = get_device()
    out = evaluate_checkpoint(cfg, rid, split, device, limit)
    chk = regression_check(cfg, rid, out) if (split == "test" and not limit) else {}
    if limit:
        path = path.with_name(f"{rid}_limit{limit}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    if split == "test":
        log_test_access({"model": rid, "split": "test", "purpose": "final_eval: the pre-registered single pass (clean, 6 background "
                         "variants, 18 perturbations, Grad-CAM)", "ckpt_sha256": sha256_file(runs.spec(cfg, rid)["ckpt"]),
                         "regression_vs_earlier_outputs": chk})
        log.info("%s regression vs earlier outputs: %s", rid, chk)
    log.info("%s -> %s", rid, path.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--test", action="store_true", help="confirm that the test set is read")
    ap.add_argument("--limit", type=int, help="first N images only (validation development)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    import runs

    ids = runs.eval_run_ids() if a.all else [a.run]
    if not ids or ids == [None]:
        ap.error("give --run or --all")
    for rid in ids:
        run_one(cfg, rid, a.split, a.test, a.limit)


if __name__ == "__main__":
    main()
