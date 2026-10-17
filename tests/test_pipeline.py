"""Synthetic end-to-end checks of the audit / grouping / split logic.

Synthetic 'stones' share ONE silhouette (like same-cut gems on the same studio background) and differ
only in a seeded facet texture, so SIFT must separate *the same photo* from *a different stone of the
same cut*. Embeddings are faked (random unit vector per content id; noise for re-encoded copies): the
tests exercise pipeline logic, not DINOv2. Real thresholds are validated on the real data with the
contact sheets in dataset/audit/.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import preprocessing as pp  # noqa: E402
from common import load_config  # noqa: E402


def make_gem(seed: int, hue: tuple[int, int, int], size: int = 320) -> Image.Image:
    """Same silhouette for every seed; facet texture depends on the seed."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (134, 134, 136))
    d = ImageDraw.Draw(img)
    d.ellipse([60, 70, 260, 250], fill=hue)
    d.text((228, 8), "Yellow Light", fill=(235, 235, 235))  # identical overlay on every photo (real-data trap)
    d.text((228, 22), "GEMPUNDIT", fill=(235, 235, 235))
    for _ in range(90):  # seeded facet lines / blobs
        x0, y0 = rng.integers(70, 250), rng.integers(80, 240)
        x1, y1 = np.clip(x0 + rng.integers(-45, 45), 70, 250), np.clip(y0 + rng.integers(-45, 45), 80, 240)
        shade = int(rng.integers(-90, 90))
        col = tuple(int(np.clip(c + shade, 0, 255)) for c in hue)
        d.line([x0, y0, x1, y1], fill=col, width=int(rng.integers(2, 6)))
        r = int(rng.integers(2, 7))
        d.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=col)
    return img


@pytest.fixture()
def cfg():
    c = copy.deepcopy(load_config())
    c["classes"] = [k for k in c["classes"] if k["slug"] in ("ruby", "pink_sapphire")]
    c["audit"]["workers"] = 1
    c["audit"]["template_exemplars"] = ["promo_exemplar"]
    return c


@pytest.fixture()
def world(tmp_path, cfg):
    items = []  # (image_id, class, sku, content_id, image)
    for cls, base_hue, off in (("Ruby", (200, 30, 40), 0), ("Pink Sapphire", (230, 120, 160), 1000)):
        for s in range(40):
            for v in range(3):
                cid = off + s * 10 + v
                items.append((f"{cls[:2]}{s:03d}_{v}", cls, f"SKU{off + s:05d}", cid, make_gem(cid, base_hue)))
    ruby0 = items[0][4]
    items.append(("ruby_exact_copy", "Ruby", "SKU90001", 0, ruby0.copy()))                      # exact copy, other SKU
    items.append(("ruby_resaved", "Ruby", "SKU90002", 3, items[3][4].resize((280, 280))))       # resized + re-encoded
    items.append(("ruby_lookalike", "Ruby", "SKU90003", 77777, make_gem(77777, (200, 30, 40))))  # same cut/hue, other stone
    items.append(("cross_class_dup", "Pink Sapphire", "SKU90004", 0, ruby0.copy()))             # same image, other class
    # promo mock-up + a copy of it filed under a real stone's SKU (that stone's real photos must survive)
    promo = make_gem(5555, (30, 200, 30))
    items.append(("promo_exemplar", "Ruby", "SKU90010", 5555, promo))
    items.append(("promo_on_real_sku", "Ruby", "SKU00005", 5555, promo.resize((300, 300))))
    rows = []
    for iid, cls, sku, cid, img in items:
        p = tmp_path / f"{iid}.jpg"
        img.save(p, quality=92)
        rows.append({"image_id": iid, "filepath": str(p), "class": cls, "class_idx": 0, "sku": sku,
                     "image_url": "", "cid": cid})
    trunc = tmp_path / "ruby_trunc.jpg"
    items[5][4].save(trunc, quality=92)
    trunc.write_bytes(trunc.read_bytes()[:700])  # truncated JPEG
    rows.append({"image_id": "ruby_trunc", "filepath": str(trunc), "class": "Ruby", "class_idx": 0,
                 "sku": "SKU90005", "image_url": "", "cid": 5})
    zero = tmp_path / "ruby_zero.jpg"
    zero.write_bytes(b"")
    rows.append({"image_id": "ruby_zero", "filepath": str(zero), "class": "Ruby", "class_idx": 0,
                 "sku": "SKU90006", "image_url": "", "cid": 6})
    return pd.DataFrame(rows)


def fake_embeddings(df: pd.DataFrame, noisy_ids=("ruby_resaved", "promo_on_real_sku")) -> np.ndarray:
    out = []
    for r in df.itertuples():
        v = np.random.default_rng(int(r.cid) + 12345).normal(size=64)
        if r.image_id in noisy_ids:
            v = v + np.random.default_rng(1).normal(size=64) * 0.05
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


@pytest.fixture()
def pipeline(world, cfg):
    q, thumbs = pp.run_qc(world, cfg, workers=1)
    ok = (q["qc_flag"] == "").values
    qq, th = q[ok].reset_index(drop=True), thumbs[ok]
    emb = fake_embeddings(qq)
    keep = ~pp.flag_templates(qq, emb, cfg)
    qk, thk, ek = qq[keep].reset_index(drop=True), th[keep], emb[keep]
    # candidates the (fake) embedding would not propose but we want SIFT to judge:
    pairs = pp.find_pairs(qk, ek, thk, cfg, extra_pairs=[("ruby_resaved", "Ru001_0"), ("ruby_lookalike", "Ru001_0"),
                                                         ("ruby_lookalike", "Ru002_0")])
    path_of = {r.image_id: Path(r.filepath) for r in qk.itertuples()}
    pairs = pp.add_sift(pairs, path_of, cfg, workers=1)
    grouped = pp.assign_groups(q, pairs, cfg, template=pp.flag_templates(q, fake_embeddings(q), cfg))
    return cfg, q, pairs, grouped, qq, emb


def pairs_of(pairs, name):
    return pairs[(pairs.image_a == name) | (pairs.image_b == name)]


def test_corrupt_images_flagged(pipeline):
    _, q, *_ = pipeline
    flags = dict(zip(q["image_id"], q["qc_flag"]))
    assert flags["ruby_trunc"] == "corrupt"
    assert flags["ruby_zero"] == "corrupt"
    assert sum(f == "corrupt" for f in flags.values()) == 2  # nothing else flagged


def test_exact_copy_detected_across_skus(pipeline):
    cfg, _, pairs, *_ = pipeline
    row = pairs_of(pairs, "ruby_exact_copy")
    row = row[row.exact]
    assert len(row) >= 1 and pp.is_edge(row, cfg).all()


def test_resaved_copy_is_near_duplicate_via_sift(pipeline):
    cfg, _, pairs, *_ = pipeline
    m = pairs[((pairs.image_a == "ruby_resaved") & (pairs.image_b == "Ru001_0"))
              | ((pairs.image_b == "ruby_resaved") & (pairs.image_a == "Ru001_0"))]
    assert len(m) == 1
    assert int(m.sift_inliers.iloc[0]) >= cfg["audit"]["near_dup_sift_inliers"]
    assert pp.is_edge(m, cfg).all()


def test_same_cut_different_stone_not_merged(pipeline):
    """Same silhouette, same hue, same background, different texture: a candidate that must be rejected."""
    cfg, _, pairs, grouped, *_ = pipeline
    m = pairs_of(pairs, "ruby_lookalike")
    assert len(m) >= 2                                            # non-vacuous: SIFT really judged them
    assert (m.sift_inliers < cfg["audit"]["near_dup_sift_inliers"]).all()
    assert not pp.is_edge(m, cfg).any()
    g = grouped.set_index("image_id")
    assert g.loc["ruby_lookalike", "drop_reason"] == ""
    assert (g["group_id"] == g.loc["ruby_lookalike", "group_id"]).sum() == 1


def test_cross_class_duplicate_removed_from_both_classes(pipeline):
    _, _, _, grouped, *_ = pipeline
    g = grouped.set_index("image_id")
    assert g.loc["cross_class_dup", "drop_reason"] == "cross_class_component"
    assert g.loc["ruby_exact_copy", "drop_reason"] == "cross_class_component"
    assert g.loc["Ru000_0", "drop_reason"] == "cross_class_component"  # the original it duplicates


def test_template_images_dropped_but_real_photos_of_sku_kept(pipeline):
    _, _, _, grouped, *_ = pipeline
    g = grouped.set_index("image_id")
    assert g.loc["promo_exemplar", "drop_reason"] == "non_stone_template"
    assert g.loc["promo_on_real_sku", "drop_reason"] == "non_stone_template"
    assert all(g.loc[f"Ru005_{v}", "drop_reason"] == "" for v in range(3))  # same SKU, real photos survive


def test_stone_views_share_group_and_split(pipeline):
    cfg, _, _, grouped, *_ = pipeline
    split = pp.split_groups(grouped, cfg)
    kept = split[split["split"] != ""]
    assert (kept.groupby("sku")["split"].nunique() == 1).all()
    assert (kept.groupby("group_id")["split"].nunique() == 1).all()


def test_split_ratios_deterministic_and_seed_sensitive(pipeline):
    cfg, _, _, grouped, *_ = pipeline
    a = pp.split_groups(grouped, cfg)
    b = pp.split_groups(grouped, cfg)
    pd.testing.assert_frame_equal(a, b)
    tab = pp.distribution_table(a, cfg)
    assert pp.ratio_drift(tab, cfg)["ok"].all()
    assert tab.iloc[-1]["Total"] == (a["split"] != "").sum()
    cfg2 = copy.deepcopy(cfg)
    cfg2["seed"] = 7
    assert not (a["split"] == pp.split_groups(grouped, cfg2)["split"]).all()


def test_leakage_audit_passes_and_detects_injected_leak(pipeline):
    cfg, q, pairs, grouped, qq, emb = pipeline
    split = pp.split_groups(grouped, cfg)
    pos = {i: k for k, i in enumerate(qq["image_id"])}
    e = np.zeros((len(split), emb.shape[1]), dtype=np.float32)
    for r, iid in enumerate(split["image_id"]):
        if iid in pos:
            e[r] = emb[pos[iid]]
    assert pp.leakage_audit(split, e, pairs, cfg)["passed"]
    leaked = split.copy()
    victim = leaked[leaked["split"] == "train"].index[0]
    leaked.loc[victim, "split"] = "test"  # one image of a train stone moves to test
    res = pp.leakage_audit(leaked, e, pairs, cfg)
    assert not res["passed"]
    assert res["checks"]["group_id_in_multiple_splits"] >= 1
    assert res["checks"]["sku_in_multiple_splits"] >= 1
