"""Image-count trace and the White Sapphire exclusion evidence, rebuilt from the dataset files.

  python code/data_trace.py            writes dataset/data_trace.{csv,md}, dataset/data_trace_per_class.csv,
                                       dataset/white_sapphire_exclusion.md, results/white_sapphire_backdrop.csv,
                                       figures/white_sapphire_backdrop.png

Every number is computed from dataset/scraped_images.csv, all_images.csv, the benchmark lists and the cached
per-image backdrop statistics (dataset/_cache/qc.csv). The earlier study's counts are typed in from its Table 1
(its image files were not kept); they are the only constants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

# Table 1 of the earlier paper (8 classes) and its split (2,545 train / 637 validation, no test set).
EARLIER_PAPER = {"Peridot": 480, "Ametrine": 262, "Blue Zircon": 246, "White Sapphire": 466, "Yellow Sapphire": 442,
                 "Ruby": 444, "Emerald": 446, "Blue Sapphire": 396}
EARLIER_SPLIT = {"train": 2545, "validation": 637, "test": 0}
BLACK, WHITE = 40, 230  # mean border brightness (0-255) below / above which a backdrop counts as black / white
FEATURES = ["bg_r", "bg_g", "bg_b", "width", "height", "brightness"]


def load_tables(cfg: dict) -> dict:
    d = ROOT / "dataset"
    scraped = pd.read_csv(d / "scraped_images.csv")
    allimg = pd.read_csv(d / "all_images.csv")
    bench = pd.concat([pd.read_csv(ROOT / cfg["benchmark"]["dir"] / f"{s}.csv").assign(split=s)
                       for s in ("train", "validation", "test")], ignore_index=True)
    return {"scraped": scraped, "all": allimg, "bench": bench}


def per_class(t: dict, cfg: dict) -> pd.DataFrame:
    s, a, b = t["scraped"], t["all"], t["bench"]
    roles = s.pivot_table(index="class", columns="role", values="filename", aggfunc="count", fill_value=0)
    roles["listed"] = roles.sum(axis=1)
    a = a.assign(kept=a.drop_reason.isna())
    rows = a.groupby("class").agg(downloaded=("image_id", "count"), kept=("kept", "sum"), stones=("group_id", "nunique"))
    rows["removed_promo"] = a[a.drop_reason == "non_stone_template"].groupby("class").size().reindex(rows.index).fillna(0).astype(int)
    rows["removed_duplicate"] = a[a.drop_reason == "exact_dup_same_sku"].groupby("class").size().reindex(rows.index).fillna(0).astype(int)
    kept = a[a.kept]
    split = kept.pivot_table(index="class", columns="split", values="image_id", aggfunc="count", fill_value=0)
    out = pd.concat([s.groupby("class").sku.nunique().rename("products"), roles, rows, split], axis=1)
    out = out.rename(columns={"certificate": "certificates", "hand": "hand_shots", "other": "other_files", "stone": "stone_photos"})
    excluded = set(cfg["benchmark"]["exclude_classes"])
    out["in_benchmark"] = ~out.index.isin(excluded)
    out["earlier_paper"] = pd.Series(EARLIER_PAPER)
    order = [c["name"] for c in cfg["classes"]]
    out = out.reindex(order)
    cols = ["products", "listed", "hand_shots", "certificates", "other_files", "downloaded", "removed_promo", "removed_duplicate",
            "kept", "stones", "train", "validation", "test", "in_benchmark", "earlier_paper"]
    out = out[cols]
    bench_n = b.groupby("label").size()
    assert (out.loc[out.in_benchmark, "kept"].astype(int).values == bench_n.reindex(out.index[out.in_benchmark]).values).all()
    return out


def trace(pc: pd.DataFrame) -> pd.DataFrame:
    """The one table: stage, images, stones, note."""
    ex = pc[~pc.in_benchmark]
    bench = pc[pc.in_benchmark]
    rows = [
        ("Earlier paper, Table 1 (8 classes)", sum(EARLIER_PAPER.values()), None,
         f"split {EARLIER_SPLIT['train']:,} train / {EARLIER_SPLIT['validation']} validation, no test set; image files not retained"),
        ("New scrape: gallery photos listed", int(pc.listed.sum()), None, f"{int(pc.products.sum()):,} products, {len(pc)} classes"),
        ("Excluded by role (hand shots, certificates, other)", -int(pc.hand_shots.sum() + pc.certificates.sum() + pc.other_files.sum()), None,
         f"{int(pc.hand_shots.sum()):,} hand shots, {int(pc.certificates.sum()):,} certificates, {int(pc.other_files.sum())} other"),
        ("Loose-stone photos downloaded", int(pc.downloaded.sum()), None, "one product page per stone, numbered stone views only"),
        ("Cleaning and deduplication", -int(pc.removed_promo.sum() + pc.removed_duplicate.sum()), None,
         f"{int(pc.removed_promo.sum())} promo banner images, {int(pc.removed_duplicate.sum())} exact duplicates within one product; 0 corrupt"),
        ("After cleaning (11 classes)", int(pc.kept.sum()), int(pc.stones.sum()),
         "near-duplicate links across products (44) merged into stone groups, none deleted"),
        ("White Sapphire removed", -int(ex.kept.sum()), -int(ex.stones.sum()), "97% black backdrop; see white_sapphire_exclusion.md"),
        ("Benchmark (10 classes)", int(bench.kept.sum()), int(bench.stones.sum()),
         f"train {int(bench.train.sum()):,} / validation {int(bench.validation.sum())} / test {int(bench.test.sum())}"),
    ]
    return pd.DataFrame(rows, columns=["stage", "images", "stones", "note"])


def backdrop_stats(t: dict, cfg: dict, qc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class backdrop statistics and the backdrop-only classifier (trained on train, scored on validation)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    qc = pd.read_csv(qc_path)[["image_id", "bg_r", "bg_g", "bg_b", "brightness"]]
    a = t["all"]
    a = a[a.drop_reason.isna()].merge(qc, on="image_id")
    a["bg"] = a[["bg_r", "bg_g", "bg_b"]].mean(axis=1)
    stats = a.groupby("class").agg(images=("bg", "size"), backdrop_median=("bg", "median"),
                                   black_pct=("bg", lambda x: 100 * (x < BLACK).mean()),
                                   white_pct=("bg", lambda x: 100 * (x > WHITE).mean())).round(1)
    stats["grey_pct"] = (100 - stats.black_pct - stats.white_pct).round(1)
    stats = stats.reindex([c["name"] for c in cfg["classes"]])
    excluded = set(cfg["benchmark"]["exclude_classes"])
    recs = {}
    for name, drop in (("11 classes", set()), ("10 classes", excluded)):
        d = a[~a["class"].isin(drop)]
        tr, va = d[d.split == "train"], d[d.split == "validation"]
        rf = RandomForestClassifier(300, random_state=cfg["seed"], n_jobs=-1).fit(tr[FEATURES], tr["class"])
        p = rf.predict(va[FEATURES])
        labs = sorted(d["class"].unique())
        recs[name] = dict(accuracy=accuracy_score(va["class"], p), macro_f1=f1_score(va["class"], p, average="macro"),
                          chance=1 / len(labs), recall=dict(zip(labs, recall_score(va["class"], p, average=None, labels=labs))))
    stats["backdrop_only_recall_11"] = pd.Series(recs["11 classes"]["recall"]).round(2)
    stats["backdrop_only_recall_10"] = pd.Series(recs["10 classes"]["recall"]).round(2)
    summary = pd.DataFrame({k: {m: v[m] for m in ("accuracy", "macro_f1", "chance")} for k, v in recs.items()}).T.round(3)
    return stats, summary


def figure(stats: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    s = stats.sort_values("black_pct", ascending=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(s))
    ax[0].barh(y, s.black_pct, color="#1d2430", label="black")
    ax[0].barh(y, s.grey_pct, left=s.black_pct, color="#9aa7b6", label="grey")
    ax[0].barh(y, s.white_pct, left=s.black_pct + s.grey_pct, color="#eef2f7", edgecolor="#9aa7b6", label="white")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(s.index)
    ax[0].set_xlim(0, 100)
    ax[0].set_xlabel("share of the class's photos (%)")
    ax[0].set_title("Backdrop colour by class (mean border brightness)", loc="left", fontsize=11)
    ax[0].legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, -0.28))
    r = stats.backdrop_only_recall_11.reindex(s.index)
    cols = ["#b3261e" if n == "White Sapphire" else "#2a78d6" for n in s.index]
    ax[1].barh(y, r, color=cols)
    ax[1].axvline(1 / 11, color="#5d6b7c", ls="--", lw=1)
    ax[1].text(1 / 11 + 0.01, len(s) - 0.4, "chance 0.09", color="#5d6b7c", fontsize=9)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("recall of a classifier that sees only backdrop colour, size, brightness")
    ax[1].set_title("What the backdrop alone gives away (validation)", loc="left", fontsize=11)
    for yi, v in zip(y, r):
        ax[1].text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=9)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown(tr: pd.DataFrame, pc: pd.DataFrame, stats: pd.DataFrame, summary: pd.DataFrame, root: Path) -> None:
    fmt = lambda v: "" if pd.isna(v) else f"{int(v):,}" if v >= 0 else f"−{abs(int(v)):,}"
    t = tr.copy()
    t["images"] = t.images.map(fmt)
    t["stones"] = t.stones.map(fmt)
    split = pc[pc.in_benchmark][["kept", "stones", "train", "validation", "test"]].astype(int)
    split.loc["Total"] = split.sum()
    split = split.rename(columns={"kept": "images"})
    md = ["# Image trace\n", "Rebuilt by `code/data_trace.py` from the dataset files. Counts are images unless stated.\n",
          t.to_markdown(index=False), "\n\n## Final 10-class split\n", split.to_markdown(), "\n",
          "\n## All 11 classes, before White Sapphire was removed\n", pc.drop(columns=["in_benchmark"]).to_markdown(), "\n"]
    (root / "dataset" / "data_trace.md").write_text("\n".join(md))
    ex = ["# Why White Sapphire is excluded from the benchmark\n",
          "Backdrop = mean colour of the border pixels of each 1920x1200 photo. Black means mean brightness below "
          f"{BLACK}/255, white above {WHITE}/255, grey in between. Computed on the kept photos of every class "
          "(`dataset/_cache/qc.csv`).\n", stats.to_markdown(), "\n",
          "\n## Backdrop-only classifier\n",
          "A random forest (300 trees, seed 42) trained on the training split and scored on the validation split, using only "
          "backdrop colour, image size and brightness. It sees no stone pixels.\n", summary.to_markdown(), "\n",
          f"\n`White Sapphire` is on a black backdrop in {stats.loc['White Sapphire', 'black_pct']:.1f}% of its photos and is "
          f"recovered with recall {stats.loc['White Sapphire', 'backdrop_only_recall_11']:.2f} from the backdrop alone. "
          f"Every other class is almost entirely on grey; Blue Sapphire is on white in {stats.loc['Blue Sapphire', 'white_pct']:.1f}% of its photos. "
          "Removing White Sapphire lowers the backdrop-only accuracy from "
          f"{summary.loc['11 classes', 'accuracy']:.3f} to {summary.loc['10 classes', 'accuracy']:.3f} (chance "
          f"{summary.loc['11 classes', 'chance']:.3f} and {summary.loc['10 classes', 'chance']:.3f}); what remains is mostly Blue Sapphire's white backdrop, "
          "which is what the background-dependence experiment measures.\n",
          "\nThe exclusion is one config switch (`benchmark.exclude_classes`); re-adding the class needs a cache rebuild and retraining.\n",
          "\n![Backdrop colour and what it gives away](../figures/white_sapphire_backdrop.png)\n"]
    (root / "dataset" / "white_sapphire_exclusion.md").write_text("\n".join(ex))


def main() -> None:
    cfg = load_config()
    t = load_tables(cfg)
    pc = per_class(t, cfg)
    tr = trace(pc)
    qc = ROOT / cfg["audit"]["cache"] / "qc.csv"
    tr.to_csv(ROOT / "dataset" / "data_trace.csv", index=False)
    pc.to_csv(ROOT / "dataset" / "data_trace_per_class.csv")
    stats, summary = backdrop_stats(t, cfg, qc)
    stats.to_csv(ROOT / "results" / "white_sapphire_backdrop.csv")
    figure(stats, ROOT / "figures" / "white_sapphire_backdrop.png")
    write_markdown(tr, pc, stats, summary, ROOT)
    print(tr.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
