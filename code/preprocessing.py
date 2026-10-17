"""Dataset audit + leak-free stratified split (Deliverable 1).

Pipeline (each stage caches its output; `all` runs them in order):
  qc      decode every scraped image, hash it, thumbnail it          -> dataset/_cache/qc.csv
  embed   L2-normalised deep embeddings (DINOv2-small)               -> dataset/_cache/embeddings.npy
  pairs   cross-SKU exact / near-duplicate candidate pairs           -> dataset/near_dup_pairs.csv
  sheets  contact sheets of pairs per similarity band (for choosing thresholds)
  split   groups -> cross-class / giant-component removal -> 70/15/15 split
          -> train/validation/test.csv, removed.csv, all_images.csv, split_config.json
  report  distribution table, audit_report.md, figures

Group = one physical stone. Start with SKU, then merge SKUs connected by exact/near-duplicate
images (union-find). Whole groups are assigned to a single split.

Usage: python code/preprocessing.py {qc|embed|pairs|sheets|split|report|all}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config, resolve, set_seed  # noqa: E402

log = logging.getLogger("preprocessing")
SPLITS = ["train", "validation", "test"]


# ============================================================================== QC
def _qc_one(args: tuple[str, str]) -> dict:
    """Decode one image fully; return hashes + stats, or the error. Top-level for pickling."""
    image_id, path = args
    out = {"image_id": image_id, "qc_ok": False, "qc_error": ""}
    try:
        size = os.path.getsize(path)
        out["file_bytes"] = size
        if size == 0:
            raise ValueError("zero-byte file")
        with open(path, "rb") as f:
            out["file_sha256"] = hashlib.sha256(f.read()).hexdigest()
        with Image.open(path) as im:
            fmt = im.format
            im.verify()  # structural check (does not catch truncation) ...
        with Image.open(path) as im:
            im.load()  # ... so decode fully as well (raises on truncated data)
            im = ImageOps.exif_transpose(im)
            rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        out.update(
            width=rgb.width, height=rgb.height, fmt=fmt,
            pixel_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
        )
        import imagehash

        out["phash"] = str(imagehash.phash(rgb))
        out["dhash"] = str(imagehash.dhash(rgb))
        out["whash"] = str(imagehash.whash(rgb))
        thumb = np.asarray(rgb.resize((16, 16), Image.BILINEAR), dtype=np.uint8)
        out["_thumb"] = thumb
        border = np.concatenate([arr[:8].reshape(-1, 3), arr[-8:].reshape(-1, 3),
                                 arr[:, :8].reshape(-1, 3), arr[:, -8:].reshape(-1, 3)])
        out["bg_r"], out["bg_g"], out["bg_b"] = (float(x) for x in border.mean(0))
        out["brightness"] = float(arr.mean())
        out["qc_ok"] = True
    except Exception as e:  # noqa: BLE001 - any decode failure is a QC failure
        out["qc_error"] = f"{type(e).__name__}: {e}"
    return out


def run_qc(df: pd.DataFrame, cfg: dict, workers: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Add QC columns to df (needs image_id, filepath); returns (df, thumbs[n,16,16,3])."""
    acfg = cfg["audit"]
    jobs = [(r.image_id, str(ROOT / r.filepath)) for r in df.itertuples()]
    workers = workers or acfg["workers"]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_qc_one, jobs, chunksize=16))
    else:
        res = [_qc_one(j) for j in jobs]
    thumbs = np.zeros((len(res), 16, 16, 3), dtype=np.uint8)
    for i, r in enumerate(res):
        t = r.pop("_thumb", None)
        if t is not None:
            thumbs[i] = t
    q = pd.DataFrame(res)
    out = df.merge(q, on="image_id", how="left", validate="1:1")
    # order must match thumbs: merge preserves left order
    assert (out["image_id"].values == np.array([r["image_id"] for r in res])).all()
    ok = out["qc_ok"].fillna(False).astype(bool)
    small = ok & (np.minimum(out["width"], out["height"]) < acfg["min_side"])
    aspect = ok & (np.maximum(out["width"] / out["height"], out["height"] / out["width"]) > acfg["max_aspect"])
    out["qc_flag"] = np.select(
        [~ok, small, aspect], ["corrupt", "too_small", "bad_aspect"], default="")
    return out, thumbs


# ======================================================================= embeddings
def compute_embeddings(paths: list[str], cfg: dict) -> tuple[np.ndarray, str]:
    """L2-normalised pooled features. Tries DINOv2-small, then fallbacks (all via timm)."""
    import timm
    import torch
    from torchvision import transforms as T

    a = cfg["audit"]
    set_seed(cfg["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, name = None, None
    for cand in [a["embed_model"], *a["embed_fallbacks"]]:
        try:
            kw = {"img_size": a["embed_size"]} if "vit" in cand else {}
            model = timm.create_model(cand, pretrained=True, num_classes=0, **kw).eval().to(device)
            name = cand
            break
        except Exception as e:  # noqa: BLE001
            log.warning("embedding model %s unavailable: %s", cand, e)
    if model is None:
        raise RuntimeError("no embedding model could be loaded")
    dc = timm.data.resolve_data_config({}, model=model)
    tf = T.Compose([T.Resize((a["embed_size"], a["embed_size"]), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(dc["mean"], dc["std"])])
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), a["embed_batch"]):
            batch = []
            for p in paths[i:i + a["embed_batch"]]:
                with Image.open(p) as im:
                    batch.append(tf(ImageOps.exif_transpose(im).convert("RGB")))
            f = model(torch.stack(batch).to(device)).float().cpu().numpy()
            feats.append(f)
            if (i // a["embed_batch"]) % 20 == 0:
                log.info("embedded %d/%d", min(i + a["embed_batch"], len(paths)), len(paths))
    e = np.concatenate(feats)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    return e.astype(np.float32), name


# ============================================================================ pairs
def _hex_to_u64(hexes: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def find_pairs(df: pd.DataFrame, emb: np.ndarray, thumbs: np.ndarray, cfg: dict,
               extra_pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Candidate cross-SKU pairs: exact duplicates, embedding cosine >= pair_floor_cos, or pHash
    Hamming <= pair_floor_hamming (plus any `extra_pairs` of image_ids, for tests). These are only
    *candidates*; `add_sift` verifies them. df rows must align with emb / thumbs."""
    a = cfg["audit"]
    n = len(df)
    sku = df["sku"].to_numpy(dtype=object)
    ph = _hex_to_u64(df["phash"])
    cand: dict[tuple[int, int], dict] = {}

    def add(i, j, **kw):
        if sku[i] == sku[j]:
            return
        key = (i, j) if i < j else (j, i)
        cand.setdefault(key, {}).update(kw)

    S = emb @ emb.T
    iu, ju = np.where(np.triu(S >= a["pair_floor_cos"], k=1))
    for i, j in zip(iu, ju):
        add(i, j)
    for i in range(n):  # pHash all-pairs (n is a few thousand: cheap)
        ham = _popcount64(ph[i + 1:] ^ ph[i])
        for off in np.where(ham <= a["pair_floor_hamming"])[0]:
            add(i, i + 1 + off)
    pos = {iid: k for k, iid in enumerate(df["image_id"])}
    for ia, ib in extra_pairs or []:
        add(pos[ia], pos[ib])
    for col in ("pixel_sha256", "file_sha256"):
        for _, idx in df.groupby(col).indices.items():
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    add(idx[x], idx[y], exact=True)

    th = thumbs.reshape(n, -1).astype(np.float32) / 255.0
    stone = thumbs[:, 5:11, 5:11, :].reshape(n, -1, 3).astype(np.float32).mean(1) / 255.0  # stone-region colour
    rows = []
    for (i, j), kw in sorted(cand.items()):
        ham = int(_popcount64(np.array([ph[i] ^ ph[j]], dtype=np.uint64))[0])
        rows.append({
            "image_a": df.image_id.iat[i], "image_b": df.image_id.iat[j],
            "sku_a": sku[i], "sku_b": sku[j],
            "class_a": df["class"].iat[i], "class_b": df["class"].iat[j],
            "cos": round(float(S[i, j]), 4), "hamming": ham,
            "color_l1": round(float(np.abs(th[i] - th[j]).mean()), 4),
            "stone_l1": round(float(np.abs(stone[i] - stone[j]).mean()), 4),
            "exact": bool(kw.get("exact", False)),
        })
    cols = ["image_a", "image_b", "sku_a", "sku_b", "class_a", "class_b", "cos", "hamming", "color_l1", "stone_l1", "exact"]
    return pd.DataFrame(rows, columns=cols)


def _sift_job(args: tuple[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    path, s = args
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    sc = s["max_side"] / max(h, w)
    if sc < 1:
        im = cv2.resize(im, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    mask = None
    if s.get("mask"):
        h2, w2 = im.shape
        x0, y0, x1, y1 = s["mask"]
        mask = np.zeros((h2, w2), np.uint8)
        mask[int(y0 * h2):int(y1 * h2), int(x0 * w2):int(x1 * w2)] = 255
    kp, des = cv2.SIFT_create(nfeatures=s["nfeatures"]).detectAndCompute(im, mask)
    if des is None or len(kp) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    pts = np.array([k.pt for k in kp], np.float32)
    return pts, np.clip(np.rint(des), 0, 255).astype(np.uint8)


def sift_inliers(f1: tuple, f2: tuple, s: dict) -> int:
    """# of RANSAC inliers (similarity transform) among ratio-test SIFT matches."""
    import cv2

    (p1, d1), (p2, d2) = f1, f2
    if len(d1) < 8 or len(d2) < 8:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < s["ratio"] * m[1].distance]
    if len(good) < 6:
        return 0
    src = p1[[g.queryIdx for g in good]]
    dst = p2[[g.trainIdx for g in good]]
    _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=s["ransac_px"], maxIters=2000, confidence=0.99)
    return 0 if inl is None else int(inl.sum())


def add_sift(pairs: pd.DataFrame, path_of: dict, cfg: dict, workers: int | None = None) -> pd.DataFrame:
    """Add `sift_inliers` to candidate pairs (exact pairs get a large sentinel). path_of: image_id -> path."""
    import cv2

    s = cfg["audit"]["sift"]
    cv2.setRNGSeed(cfg["seed"])
    if pairs.empty:
        return pairs.assign(sift_inliers=pd.Series(dtype=int))
    ids = sorted(set(pairs["image_a"]) | set(pairs["image_b"]))
    jobs = [(str(path_of[i]), s) for i in ids]
    workers = workers or cfg["audit"]["workers"]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            feats = list(ex.map(_sift_job, jobs, chunksize=16))
    else:
        feats = [_sift_job(j) for j in jobs]
    fx = dict(zip(ids, feats))
    pairs = pairs.copy()
    pairs["sift_inliers"] = [
        999 if r.exact else sift_inliers(fx[r.image_a], fx[r.image_b], s) for r in pairs.itertuples()]
    return pairs


def is_edge(p: pd.DataFrame, cfg: dict) -> pd.Series:
    """Same stone / sister stone / same photo (see config.yaml audit.near_dup_*)."""
    a = cfg["audit"]
    geo = (p["sift_inliers"] >= a["near_dup_sift_inliers"]) & (p["stone_l1"] <= a["near_dup_geo_stone_l1"])
    vis = (p["cos"] >= a["near_dup_cos"]) & (p["stone_l1"] <= a["near_dup_vis_stone_l1"])
    return p["exact"] | geo | vis


def flag_templates(df: pd.DataFrame, emb: np.ndarray, cfg: dict) -> np.ndarray:
    """Boolean mask: promo/mock-up images, i.e. cosine >= template_cos to a curated exemplar."""
    a = cfg["audit"]
    ex = [i for i in a.get("template_exemplars", []) if i in set(df["image_id"])]
    if not ex:
        return np.zeros(len(df), bool)
    pos = {i: k for k, i in enumerate(df["image_id"])}
    return (emb @ emb[[pos[e] for e in ex]].T).max(1) >= a["template_cos"]


def contact_sheets(pairs: pd.DataFrame, path_of: dict, seed: int, out_dir: Path, per_band: int = 24) -> list[Path]:
    """Side-by-side thumbnails of pairs sampled per SIFT-inlier band (for threshold review),
    plus a 'high cosine but no SIFT support' band showing why cosine alone is not used.
    path_of maps image_id -> absolute path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("pairs_*.jpg"):
        f.unlink()
    bands = [("inl_100plus", pairs["sift_inliers"] >= 100), ("inl_60_99", pairs["sift_inliers"].between(60, 99)),
             ("inl_40_59", pairs["sift_inliers"].between(40, 59)), ("inl_30_39", pairs["sift_inliers"].between(30, 39)),
             ("inl_20_29", pairs["sift_inliers"].between(20, 29)), ("inl_12_19", pairs["sift_inliers"].between(12, 19)),
             ("inl_lt12_cos97", (pairs["sift_inliers"] < 12) & (pairs["cos"] >= 0.97))]
    rng = random.Random(seed)
    made = []
    for name, mask in bands:
        sub = pairs[mask]
        if sub.empty:
            continue
        sub = sub.iloc[sorted(rng.sample(range(len(sub)), min(per_band, len(sub))))]
        T, cols = 200, 3
        rows_n = (len(sub) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (2 * T + 10), rows_n * (T + 24)), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(sub.itertuples()):
            x, y = (k % cols) * (2 * T + 10), (k // cols) * (T + 24)
            for m, iid in enumerate((r.image_a, r.image_b)):
                try:
                    with Image.open(path_of[iid]) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        im.thumbnail((T, T))
                        sheet.paste(im, (x + m * T, y + 14))
                except Exception:  # noqa: BLE001
                    pass
            d.text((x + 2, y), f"inl={r.sift_inliers} cos={r.cos:.3f} {r.class_a[:6]}/{r.class_b[:6]}", fill="black")
        p = out_dir / f"pairs_{name}.jpg"
        sheet.save(p, quality=85)
        made.append(p)
    return made


# ======================================================================== grouping
class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)  # deterministic root


def assign_groups(df: pd.DataFrame, pairs: pd.DataFrame, cfg: dict, template: np.ndarray | None = None) -> pd.DataFrame:
    """Add group_id / removal reasons. df has qc columns; returns df with columns
    `group_id` and `drop_reason` ('' = kept). Order of operations (first reason wins):
      1. qc failures (corrupt / too_small / bad_aspect)
      1b. promo/mock-up images (`template` mask, aligned with df) -> non_stone_template.
          Only the image is dropped; the SKU's real photographs stay.
      2. exact duplicate inside the same SKU (keep lowest image_id)
      3. connected components over SKUs linked by near-duplicate edges:
         a. component spans >1 class  -> cross_class_component   (label noise)
         b. component has > max_group_images images -> giant_component (needs review)
    """
    df = df.copy()
    df["_tmpl"] = template if template is not None else False
    df = df.sort_values("image_id").reset_index(drop=True)
    df["drop_reason"] = df["qc_flag"].fillna("")
    df.loc[(df["drop_reason"] == "") & df["_tmpl"].astype(bool), "drop_reason"] = "non_stone_template"
    df = df.drop(columns="_tmpl")
    live = df["drop_reason"] == ""
    dup = live & df.duplicated(subset=["sku", "pixel_sha256"], keep="first")
    df.loc[dup, "drop_reason"] = "exact_dup_same_sku"

    alive_ids = set(df.loc[df["drop_reason"] == "", "image_id"])
    uf = UnionFind()
    for s in df.loc[df["drop_reason"] == "", "sku"]:
        uf.find(s)
    if len(pairs):
        e = pairs[is_edge(pairs, cfg) & pairs["image_a"].isin(alive_ids) & pairs["image_b"].isin(alive_ids)]
        for r in e.itertuples():
            uf.union(r.sku_a, r.sku_b)
    kept = df["drop_reason"] == ""
    df["_root"] = ""
    df.loc[kept, "_root"] = df.loc[kept, "sku"].map(uf.find)
    comp = df[kept].groupby("_root")
    classes = comp["class"].nunique()
    sizes = comp.size()
    bad_cls = set(classes[classes > 1].index)
    giant = set(sizes[sizes > cfg["audit"]["max_group_images"]].index) - bad_cls
    df.loc[kept & df["_root"].isin(bad_cls), "drop_reason"] = "cross_class_component"
    df.loc[kept & df["_root"].isin(giant), "drop_reason"] = "giant_component"
    kept = df["drop_reason"] == ""
    roots = sorted(df.loc[kept, "_root"].unique())
    gid = {r: f"G{i:05d}" for i, r in enumerate(roots)}
    df["group_id"] = np.where(kept, df["_root"].map(gid), "")
    return df.drop(columns="_root")


# ============================================================================ split
def split_groups(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-class, group-atomic stratified split. Groups are processed largest-first (seeded
    shuffle breaks ties) and each goes to the split with the largest remaining deficit vs its
    image-count target. Deterministic for a given seed."""
    ratios = cfg["split_ratios"]
    df = df.copy()
    df["split"] = ""
    kept = df[df["drop_reason"] == ""]
    for klass in cfg["classes"]:
        k = kept[kept["class"] == klass["name"]]
        if k.empty:
            continue
        sizes = k.groupby("group_id").size()
        ids = sorted(sizes.index)
        random.Random(f"{cfg['seed']}-{klass['slug']}").shuffle(ids)
        ids.sort(key=lambda g: -sizes[g])  # stable: seeded order kept among equal sizes
        total = int(sizes.sum())
        target = {s: ratios[s] * total for s in SPLITS}
        got = {s: 0 for s in SPLITS}
        assign = {}
        for g in ids:
            s = max(SPLITS, key=lambda s_: (target[s_] - got[s_], -SPLITS.index(s_)))
            assign[g] = s
            got[s] += int(sizes[g])
        df.loc[k.index, "split"] = k["group_id"].map(assign)
    return df


def distribution_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    kept = df[df["split"] != ""]
    rows = []
    for klass in cfg["classes"]:
        k = kept[kept["class"] == klass["name"]]
        row = {"Class": klass["name"]}
        for s in SPLITS:
            row[s] = int((k["split"] == s).sum())
        row["Total"] = len(k)
        for s in SPLITS:
            row[f"{s}_%"] = round(100 * row[s] / row["Total"], 1) if row["Total"] else 0.0
        row["Groups"] = k["group_id"].nunique()
        rows.append(row)
    t = pd.DataFrame(rows)
    tot = {"Class": "Total", "Groups": int(t["Groups"].sum())}
    for s in SPLITS:
        tot[s] = int(t[s].sum())
    tot["Total"] = int(t["Total"].sum())
    for s in SPLITS:
        tot[f"{s}_%"] = round(100 * tot[s] / tot["Total"], 1) if tot["Total"] else 0.0
    return pd.concat([t, pd.DataFrame([tot])], ignore_index=True)


# ============================================================================ leakage
def leakage_audit(df: pd.DataFrame, emb: np.ndarray, pairs: pd.DataFrame, cfg: dict) -> dict:
    """Hard checks that no stone / file / near-duplicate straddles splits. `df` rows align with emb."""
    a = cfg["audit"]
    kept = df["split"] != ""
    res: dict = {"checks": {}}
    for col in ("group_id", "file_sha256", "pixel_sha256", "sku"):
        per = df[kept].groupby(col)["split"].nunique()
        res["checks"][f"{col}_in_multiple_splits"] = int((per > 1).sum())
    sp = df["split"].values
    idx = {s: np.where(sp == s)[0] for s in SPLITS}
    res["max_cos_to_train"] = {}
    for s in ("validation", "test"):
        if len(idx[s]) and len(idx["train"]):
            S = emb[idx[s]] @ emb[idx["train"]].T
            res["max_cos_to_train"][s] = {"max": round(float(S.max()), 4),
                                         "p99": round(float(np.quantile(S.max(1), 0.99)), 4),
                                         "median": round(float(np.median(S.max(1))), 4)}
    if len(idx["test"]) and len(idx["validation"]):
        res["max_cos_to_train"]["test_vs_validation"] = round(float((emb[idx["test"]] @ emb[idx["validation"]].T).max()), 4)
    # any near-duplicate edge crossing splits?
    split_of = dict(zip(df["image_id"], df["split"]))
    edges = pairs[is_edge(pairs, cfg)] if len(pairs) else pairs
    cross = [(r.image_a, r.image_b) for r in edges.itertuples()
             if split_of.get(r.image_a, "") and split_of.get(r.image_b, "")
             and split_of[r.image_a] != split_of[r.image_b]]
    res["checks"]["near_dup_edges_crossing_splits"] = len(cross)
    res["crossing_examples"] = cross[:10]
    res["passed"] = all(v == 0 for v in res["checks"].values())
    return res


def ratio_drift(tab: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per class/split |achieved% - target%| in percentage points; flags > tolerance."""
    rows = []
    for _, r in tab[tab["Class"] != "Total"].iterrows():
        for s in SPLITS:
            d = abs(r[f"{s}_%"] - 100 * cfg["split_ratios"][s])
            rows.append({"Class": r["Class"], "split": s, "drift_pts": round(d, 2),
                         "ok": d <= cfg["audit"]["split_tolerance_pts"]})
    return pd.DataFrame(rows)


# ============================================================================== I/O
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_manifest(cfg: dict) -> pd.DataFrame:
    """scraped_images.csv -> manifest of downloaded stone photos (image_id, filepath, class_idx...)."""
    ds = resolve(cfg, "dataset")
    s = pd.read_csv(ds / "scraped_images.csv", dtype=str, keep_default_na=False)
    idx = {c["name"]: i for i, c in enumerate(cfg["classes"])}
    s["class_idx"] = s["class"].map(idx)
    s["filepath"] = s["local_path"]
    s["image_id"] = s["slug"] + "/" + s["filename"].str.rsplit(".", n=1).str[0]
    return s


def cache_dir(cfg: dict) -> Path:
    d = ROOT / cfg["audit"]["cache"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_qc(cfg: dict) -> None:
    man = load_manifest(cfg)
    dl = man[(man["role"] == "stone") & (man["filepath"] != "")].sort_values("image_id").reset_index(drop=True)
    log.info("QC on %d downloaded stone images", len(dl))
    q, thumbs = run_qc(dl, cfg)
    q.to_csv(cache_dir(cfg) / "qc.csv", index=False)
    np.save(cache_dir(cfg) / "thumbs.npy", thumbs)
    log.info("QC flags: %s", dict(Counter(q["qc_flag"].replace("", "ok"))))


def read_qc(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    q = pd.read_csv(cache_dir(cfg) / "qc.csv", dtype={"qc_error": str, "qc_flag": str}, keep_default_na=False)
    return q, np.load(cache_dir(cfg) / "thumbs.npy")


def usable(q: pd.DataFrame) -> np.ndarray:
    return (q["qc_flag"] == "").values


def stage_embed(cfg: dict) -> None:
    q, _ = read_qc(cfg)
    ok = usable(q)
    paths = [str(ROOT / p) for p in q.loc[ok, "filepath"]]
    emb, name = compute_embeddings(paths, cfg)
    np.save(cache_dir(cfg) / "embeddings.npy", emb)
    q.loc[ok, ["image_id"]].to_csv(cache_dir(cfg) / "embeddings_index.csv", index=False)
    (cache_dir(cfg) / "embeddings_meta.json").write_text(json.dumps({"model": name, "n": len(emb)}))


def read_embeddings(cfg: dict, q: pd.DataFrame) -> np.ndarray:
    emb = np.load(cache_dir(cfg) / "embeddings.npy")
    idx = pd.read_csv(cache_dir(cfg) / "embeddings_index.csv")["image_id"].tolist()
    pos = {i: k for k, i in enumerate(idx)}
    out = np.zeros((len(q), emb.shape[1]), dtype=np.float32)
    for r, iid in enumerate(q["image_id"]):
        if iid in pos:
            out[r] = emb[pos[iid]]
    return out


def stage_pairs(cfg: dict) -> None:
    q, thumbs = read_qc(cfg)
    ok = usable(q)
    qq, th = q[ok].reset_index(drop=True), thumbs[ok]
    emb = read_embeddings(cfg, qq)
    tmpl = flag_templates(qq, emb, cfg)
    log.info("template/mock-up images flagged: %d %s", int(tmpl.sum()), qq.loc[tmpl, "image_id"].tolist())
    keep = ~tmpl  # mock-ups must not create candidate pairs either
    qq, th, emb = qq[keep].reset_index(drop=True), th[keep], emb[keep]
    pairs = find_pairs(qq, emb, th, cfg)
    log.info("%d candidate pairs (cos>=%.2f or pHash<=%d); verifying with SIFT+RANSAC ...", len(pairs),
             cfg["audit"]["pair_floor_cos"], cfg["audit"]["pair_floor_hamming"])
    path_of = {r.image_id: ROOT / r.filepath for r in qq.itertuples()}
    pairs = add_sift(pairs, path_of, cfg)
    pairs.to_csv(resolve(cfg, "dataset") / "near_dup_pairs.csv", index=False)
    e = pairs[is_edge(pairs, cfg)]
    log.info("recorded %d pairs: %d edges (%d exact, %d cross-class) -> these merge stones into groups",
             len(pairs), len(e), int(pairs["exact"].sum()), int((e["class_a"] != e["class_b"]).sum()))


def stage_sheets(cfg: dict) -> None:
    ds = resolve(cfg, "dataset")
    q, _ = read_qc(cfg)
    path_of = {r.image_id: ROOT / r.filepath for r in q.itertuples()}
    pairs = pd.read_csv(ds / "near_dup_pairs.csv")
    made = contact_sheets(pairs, path_of, cfg["seed"], ds / "audit")
    for p in made:
        log.info("wrote %s", p)


def stage_split(cfg: dict) -> None:
    set_seed(cfg["seed"])
    ds = resolve(cfg, "dataset")
    q, thumbs = read_qc(cfg)
    ok = usable(q)
    pairs = pd.read_csv(ds / "near_dup_pairs.csv") if (ds / "near_dup_pairs.csv").exists() else pd.DataFrame()
    emb_all = read_embeddings(cfg, q)
    grouped = assign_groups(q, pairs, cfg, template=flag_templates(q, emb_all, cfg))
    split = split_groups(grouped, cfg)
    # --- leakage audit on kept images (embedding rows aligned by image_id)
    emb_q = read_embeddings(cfg, split)
    leak = leakage_audit(split, emb_q, pairs, cfg)
    tab = distribution_table(split, cfg)
    drift = ratio_drift(tab, cfg)
    if not leak["passed"]:
        raise SystemExit(f"LEAKAGE CHECK FAILED: {leak['checks']}")

    cols = ["filepath", "class", "class_idx", "group_id", "sku", "image_id", "file_sha256", "width",
            "height", "image_url", "split"]
    kept = split[split["split"] != ""].copy()
    for fname in ("train.csv", "validation.csv", "test.csv"):  # test.csv is read-only after a run
        if (ds / fname).exists():
            os.chmod(ds / fname, 0o644)
    for s, fname in zip(SPLITS, ["train.csv", "validation.csv", "test.csv"]):
        sub = kept[kept["split"] == s].sort_values("image_id")[cols].rename(columns={"class": "label"})
        sub["class_idx"] = sub["class_idx"].astype(int)
        sub.to_csv(ds / fname, index=False)
    removed = split[split["split"] == ""][["image_id", "class", "sku", "filepath", "drop_reason", "qc_error"]]
    removed.sort_values("image_id").to_csv(ds / "removed.csv", index=False)
    alls = split.sort_values("image_id")[["image_id", "class", "sku", "filepath", "group_id", "split",
                                          "drop_reason", "file_sha256", "pixel_sha256", "width", "height"]]
    alls.to_csv(ds / "all_images.csv", index=False)
    tab.to_csv(ds / "distribution_table.csv", index=False)
    (ds / "distribution_table.md").write_text(tab.to_markdown(index=False) if hasattr(tab, "to_markdown") else tab.to_string(index=False))
    meta = json.loads((cache_dir(cfg) / "embeddings_meta.json").read_text())
    import sklearn
    import torch

    cfg_json = {
        "seed": cfg["seed"], "split_ratios": cfg["split_ratios"], "audit": cfg["audit"],
        "classes": [c["name"] for c in cfg["classes"]], "embedding_model": meta["model"],
        "n_images": {s: int((kept["split"] == s).sum()) for s in SPLITS},
        "n_groups": int(kept["group_id"].nunique()), "n_removed": int(len(removed)),
        "removed_by_reason": dict(Counter(removed["drop_reason"])),
        "ratio_drift_max_pts": float(drift["drift_pts"].max()), "ratio_drift_ok": bool(drift["ok"].all()),
        "leakage": leak,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "torch": torch.__version__, "sklearn": sklearn.__version__},
        "csv_sha256": {f: sha256_file(ds / f) for f in ("train.csv", "validation.csv", "test.csv")},
    }
    (ds / "split_config.json").write_text(json.dumps(cfg_json, indent=1))
    # freeze the test set
    for p in [ds / "test.csv", *[ROOT / f for f in kept.loc[kept["split"] == "test", "filepath"]]]:
        os.chmod(p, 0o444)
    log.info("split done: %s | removed %s | leakage passed=%s | max ratio drift %.2f pts",
             cfg_json["n_images"], cfg_json["removed_by_reason"], leak["passed"], cfg_json["ratio_drift_max_pts"])


def stage_report(cfg: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = resolve(cfg, "dataset")
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    tab = pd.read_csv(ds / "distribution_table.csv")
    body = tab[tab["Class"] != "Total"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(body))
    for s, c in zip(SPLITS, ["#4C78A8", "#F58518", "#54A24B"]):
        ax.bar(body["Class"], body[s], bottom=bottom, label=s, color=c)
        bottom += body[s].values
    ax.set_ylabel("images")
    ax.set_title("Images per class and split")
    ax.legend()
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "dataset_distribution.png", dpi=150)
    plt.close(fig)
    pairs = pd.read_csv(ds / "near_dup_pairs.csv")
    if len(pairs):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(np.minimum(pairs["sift_inliers"], 150), bins=50)
        ax.set_yscale("log")
        ax.axvline(cfg["audit"]["near_dup_sift_inliers"], color="r", ls="--", label="near_dup_sift_inliers (needs colour too)")
        ax.set_xlabel("SIFT+RANSAC inliers of candidate cross-SKU pair (capped at 150)")
        ax.set_ylabel("pairs")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "dedup_sift_hist.png", dpi=150)
        plt.close(fig)
    sc = json.loads((ds / "split_config.json").read_text())
    alls = pd.read_csv(ds / "all_images.csv", keep_default_na=False)
    kept = alls[alls["split"] != ""]
    q, _ = read_qc(cfg)
    q = q[q["image_id"].isin(kept["image_id"])]
    bias = q.groupby("class").agg(n=("image_id", "size"), width=("width", "median"), height=("height", "median"),
                                  bytes_kb=("file_bytes", lambda s: round(s.median() / 1024)),
                                  brightness=("brightness", "mean"), bg_r=("bg_r", "mean"),
                                  bg_g=("bg_g", "mean"), bg_b=("bg_b", "mean")).round(1)
    counts = kept.groupby("class").size()
    imb = counts.max() / counts.min()
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum() / np.log(len(p)))
    md = ["# Dataset audit report", "",
          f"- Seed: **{sc['seed']}**   Embedding model: `{sc['embedding_model']}`",
          f"- Images kept: {sc['n_images']}  (groups: {sc['n_groups']}; removed: {sc['n_removed']})",
          f"- Removed by reason: {sc['removed_by_reason']}",
          f"- Class imbalance ratio (max/min): **{imb:.2f}**, normalised entropy: **{ent:.3f}**",
          f"- Max split-ratio drift vs 70/15/15: {sc['ratio_drift_max_pts']:.2f} pts (tolerance {cfg['audit']['split_tolerance_pts']})",
          "", "## Leakage checks (all must be 0)", "", "```", json.dumps(sc["leakage"]["checks"], indent=1), "```",
          f"Max cosine similarity val/test -> train: `{sc['leakage']['max_cos_to_train']}`",
          "", "## Distribution", "", tab.to_markdown(index=False),
          "", "## Per-class image statistics (bias scan; kept images)", "", bias.to_markdown(),
          "", "## Thresholds", "", "```", json.dumps({k: v for k, v in cfg['audit'].items()}, indent=1), "```",
          "", f"Versions: {sc['versions']}", "", f"CSV SHA-256: {sc['csv_sha256']}"]
    (ds / "audit_report.md").write_text("\n".join(md))
    log.info("wrote audit_report.md, distribution figure")



# ===================================================================== benchmark (D2+)
def bench_classes(cfg: dict) -> list[str]:
    """Class names of the benchmark, in config order, minus the excluded ones."""
    ex = set(cfg["benchmark"]["exclude_classes"])
    return [c["name"] for c in cfg["classes"] if c["name"] not in ex]


def bench_dir(cfg: dict) -> Path:
    return ROOT / cfg["benchmark"]["dir"]


def make_benchmark_lists(cfg: dict) -> dict:
    """Filter the frozen D1 lists to the benchmark classes and re-index class_idx 0..K-1.
    Every remaining image keeps the split it had in D1. Only CSV rows are read here (no images)."""
    ds, out = resolve(cfg, "dataset"), bench_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    classes = bench_classes(cfg)
    idx = {c: i for i, c in enumerate(classes)}
    info = {"classes": classes, "excluded": cfg["benchmark"]["exclude_classes"], "rows": {}, "sha256": {}}
    for name in ("train", "validation", "test"):
        df = pd.read_csv(ds / f"{name}.csv")
        keep = df[df["label"].isin(idx)].copy()
        keep["class_idx"] = keep["label"].map(idx).astype(int)
        path = out / f"{name}.csv"
        if path.exists():
            os.chmod(path, 0o644)
        keep.to_csv(path, index=False)
        info["rows"][name] = len(keep)
        info["sha256"][name] = sha256_file(path)
    os.chmod(out / "test.csv", 0o444)
    (out / "classes.json").write_text(json.dumps(classes, indent=1))
    (out / "benchmark_split.json").write_text(json.dumps(info, indent=1))
    return info


def square_crop_box(w: int, h: int) -> tuple[int, int, int, int]:
    """Centre square of the photo (the middle h/w of the width for a landscape image)."""
    s = min(w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return x0, y0, x0 + s, y0 + s


def _cache_one(args: tuple[str, int]) -> np.ndarray:
    path, size = args
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = im.crop(square_crop_box(*im.size)).resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def build_cache(cfg: dict, workers: int | None = None) -> dict:
    """Square centre-crop every benchmark image to cache_size and store one uint8 array per split,
    row-aligned with dataset/benchmark/<split>.csv. Blind to labels and splits (pure pixel op)."""
    size = cfg["benchmark"]["cache_size"]
    cdir = cache_dir(cfg) / f"img{size}"
    cdir.mkdir(parents=True, exist_ok=True)
    workers = workers or cfg["audit"]["workers"]
    counts = {}
    for name in ("train", "validation", "test"):
        df = pd.read_csv(bench_dir(cfg) / f"{name}.csv")
        jobs = [(str(ROOT / p), size) for p in df["filepath"]]
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                arrs = list(ex.map(_cache_one, jobs, chunksize=16))
        else:
            arrs = [_cache_one(j) for j in jobs]
        np.save(cdir / f"{name}.npy", np.stack(arrs))
        counts[name] = len(arrs)
        log.info("cached %s: %d images -> %s", name, len(arrs), cdir / f"{name}.npy")
    return counts


def crop_check_sheet(cfg: dict, n: int = 100) -> Path:
    """Contact sheet of n random TRAIN images with the square crop box drawn on them."""
    df = pd.read_csv(bench_dir(cfg) / "train.csv")
    rows = df.iloc[sorted(random.Random(cfg["seed"]).sample(range(len(df)), n))]
    T, cols = 190, 10
    sheet = Image.new("RGB", (cols * T, ((n + cols - 1) // cols) * int(T / 1.6 + 2)), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows.itertuples()):
        with Image.open(ROOT / r.filepath) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            box = square_crop_box(*im.size)
            sc = T / im.width
            th = im.resize((T, int(im.height * sc)))
        dd = ImageDraw.Draw(th)
        dd.rectangle([box[0] * sc, 0, box[2] * sc - 1, th.height - 1], outline=(255, 0, 0), width=2)
        sheet.paste(th, ((k % cols) * T, (k // cols) * int(T / 1.6 + 2)))
    out = resolve(cfg, "dataset") / "audit" / "crop_check.jpg"
    out.parent.mkdir(exist_ok=True)
    sheet.save(out, quality=88)
    return out


def stage_benchmark(cfg: dict) -> None:
    info = make_benchmark_lists(cfg)
    log.info("benchmark lists: %s classes=%d (excluded %s)", info["rows"], len(info["classes"]), info["excluded"])


def stage_cache(cfg: dict) -> None:
    log.info("cache: %s", build_cache(cfg))


def stage_cropcheck(cfg: dict) -> None:
    log.info("wrote %s", crop_check_sheet(cfg))


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["qc", "embed", "pairs", "sheets", "split", "report", "all", "benchmark", "cache", "cropcheck"])
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    order = ["qc", "embed", "pairs", "sheets", "split", "report"] if args.stage == "all" else [args.stage]
    fn = {"qc": stage_qc, "embed": stage_embed, "pairs": stage_pairs, "sheets": stage_sheets,
          "split": stage_split, "report": stage_report, "benchmark": stage_benchmark,
          "cache": stage_cache, "cropcheck": stage_cropcheck}
    for s in order:
        log.info("=== %s ===", s)
        fn[s](cfg)


if __name__ == "__main__":
    main()
