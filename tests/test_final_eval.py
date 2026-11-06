"""The test set stays closed until every run is trained and frozen; these tests never read test images."""
import os
import sys
from pathlib import Path

os.environ["GEM_DEVICE"] = "cpu"

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import final_eval as fe  # noqa: E402
import runs  # noqa: E402
from common import load_config  # noqa: E402


def test_run_registry_is_consistent():
    cfg = load_config()
    ids = runs.all_run_ids()
    assert len(ids) == len(set(ids)) == 9 + 3 + 12
    ev = runs.eval_run_ids()
    assert len(ev) == len(set(ev)) == 9 + 3 + 15
    assert set(ids) <= set(ev)
    # seed-42 main models are the frozen Deliverable 2 to 4 checkpoints; everything else gets its own folder
    for r in ids:
        s = runs.spec(cfg, r)
        assert s["legacy"] == (r.endswith("_s42") and s["group"] == "main")
        if not s["legacy"]:
            assert "models/runs/" in str(s["ckpt"])


def test_ablation_variants_match_the_pre_registration():
    a = runs.ABLATION
    assert a["Abl-A"]["overrides"] == dict(use_rgb=True, use_ct=False, fusion="none") and not a["Abl-A"]["stage2"]
    assert a["Abl-B"]["overrides"] == dict(use_rgb=False, use_ct=True, fusion="none") and not a["Abl-B"]["stage2"]
    assert a["Abl-C"]["overrides"]["fusion"] == "concat" and not a["Abl-C"]["stage2"]
    assert a["Abl-D"]["overrides"]["fusion"] == "gated" and not a["Abl-D"]["stage2"]
    assert a["Abl-A-FT"]["overrides"] == a["Abl-A"]["overrides"] and a["Abl-A-FT"]["stage2"]
    cfg = load_config()
    assert runs.spec(cfg, "Abl-A_s43")["s2_epochs"] == 0 and runs.spec(cfg, "Abl-A-FT_s43")["s2_epochs"] is None


def test_variant_models_build_with_the_expected_branches():
    cfg = load_config()
    for rid, (rgb, ct, fusion) in {"Abl-A_s42": (True, False, "none"), "Abl-B_s42": (False, True, "none"),
                                   "Abl-C_s42": (True, True, "concat"), "Abl-D_s42": (True, True, "gated")}.items():
        m = runs.build_run_model(cfg, rid, pretrained=False)
        assert (m.use_rgb, m.use_ct, m.fusion) == (rgb, ct, fusion), rid


def test_test_pass_needs_the_explicit_flag(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(fe, "out_path", lambda rid, split: tmp_path / "missing.npz")  # a checkpoint that has no output yet
    with pytest.raises(SystemExit, match="--test"):
        fe.run_one(cfg, "Proposed_s42", "test", False, None)


def test_test_stays_closed_while_a_run_is_untrained(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(runs, "is_trained", lambda c, r: r != "Abl-B_s44")
    with pytest.raises(SystemExit, match="not every run is trained"):
        fe.assert_ready_for_test(cfg, "Proposed_s42")


def test_test_stays_closed_until_checkpoints_are_frozen(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(runs, "is_trained", lambda c, r: True)
    monkeypatch.setattr(fe, "manifest_hashes", lambda: {})
    with pytest.raises(SystemExit, match="not frozen"):
        fe.assert_ready_for_test(cfg, "Proposed_s42")


def test_limit_is_refused_on_test(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(fe, "out_path", lambda rid, split: tmp_path / "missing.npz")
    monkeypatch.setattr(fe, "assert_ready_for_test", lambda c, r: None)
    with pytest.raises(SystemExit, match="validation development only"):
        fe.run_one(cfg, "Proposed_s42", "test", True, 48)


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    cfg = load_config()
    p = tmp_path / "x.npz"
    p.write_bytes(b"keep")
    monkeypatch.setattr(fe, "out_path", lambda rid, split: p)
    fe.run_one(cfg, "Proposed_s42", "test", True, None)  # returns without reading anything
    assert p.read_bytes() == b"keep"
