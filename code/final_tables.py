"""Tables of the pre-registered analyses, rebuilt from the per-checkpoint outputs of final_eval.py (no images are read).

  python code/final_tables.py --split test          results/final_tables/*.csv from results/final/*.npz
  python code/final_tables.py --split validation    the same from results/final_validation/*.npz (development)

Tables: clean.csv, robustness_seeds.csv, robustness_comparisons.csv, background_seeds.csv, background_comparisons.csv,
ablation.csv, ablation_comparisons.csv, gradcam.csv. Statistics and the claim rule: code/seedstats.py and
results/PREREGISTRATION.md (section 4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedstats as ss  # noqa: E402
from common import ROOT, load_config  # noqa: E402

MAIN = ("Proposed", "ConvNeXt-Tiny", "EfficientNetV2-S")
BASELINES = ("ConvNeXt-Tiny", "EfficientNetV2-S")
ABL = ("Abl-A", "Abl-B", "Abl-C", "Abl-D", "Abl-A-FT")
ABL_NAME = {"Abl-A": "A  RGB only, frozen", "Abl-B": "B  colour/texture only", "Abl-C": "C  both, concatenated", "Abl-D": "D  both, gated",
            "Abl-A-FT": "A-FT  RGB only, fine-tuned", "Proposed": "E  both, gated, fine-tuned"}
ROB_WORDS = dict(higher="Proposed more robust", lower="Proposed more fragile")


class Store:
    """The saved outputs of every available run, plus cached bootstrap vectors."""

    def __init__(self, split: str, seeds: tuple[int, ...] = (42, 43, 44), directory: Path | None = None,
                 n_draws: int = ss.N_DRAWS):
        import robustness as rb
        import runs

        self.rb, self.runs_mod, self.split = rb, runs, split
        d = directory or ROOT / "results" / ("final" if split == "test" else "final_validation")
        self.data = {rid: dict(np.load(d / f"{rid}.npz")) for rid in runs.eval_run_ids() if (d / f"{rid}.npz").exists()}
        if not self.data:
            raise SystemExit(f"no outputs in {d}")
        first = next(iter(self.data.values()))
        self.y, self.groups = first["y"], first["groups"]
        for rid, z in self.data.items():
            assert np.array_equal(z["y"], self.y), rid
        self.k = int(self.y.max()) + 1
        self.draws = ss.stone_draws(self.groups, n_draws)
        self._cache: dict = {}
        self.seeds = seeds
        self.classes = json.loads((ROOT / load_config()["benchmark"]["dir"] / "classes.json").read_text())

    def rids(self, base: str) -> list[str]:
        return [self.runs_mod.run_id(base, s) for s in self.seeds if self.runs_mod.run_id(base, s) in self.data]

    def pred(self, rid: str, key: str) -> np.ndarray:
        z = self.data[rid]
        if key == "clean":
            return z["bg|original|logits"].argmax(1)
        kind, _, name = key.partition(":")
        return z[f"bg|{name}|logits"].argmax(1) if kind == "bg" else z[f"rob|{name}|pred"].astype(int)

    def vec(self, rid: str, key: str) -> tuple[float, np.ndarray]:
        if (rid, key) not in self._cache:
            self._cache[(rid, key)] = ss.f1_vector(self.y, self.pred(rid, key), self.draws, self.k)
        return self._cache[(rid, key)]

    def level(self, base: str, key: str) -> list[tuple[float, np.ndarray]]:
        return [self.vec(r, key) for r in self.rids(base)]

    def change(self, base: str, key: str) -> list[tuple[float, np.ndarray]]:
        """Per-seed change of the metric against that seed's own clean result (paired draws)."""
        return [(self.vec(r, key)[0] - self.vec(r, "clean")[0], self.vec(r, key)[1] - self.vec(r, "clean")[1]) for r in self.rids(base)]


def _row_level(model: str, items: list) -> dict:
    m, vec, pts, sd = ss.seed_mean(items)
    lo, hi = ss.interval(vec)
    return {"model": model, "n_seeds": len(items), "per_seed": [round(p, 4) for p in pts], "mean": m, "sd_across_seeds": sd,
            "ci_lo": lo, "ci_hi": hi}


def clean_table(st: Store) -> pd.DataFrame:
    rows = []
    for base in MAIN:
        r = _row_level(base, st.level(base, "clean"))
        accs = [ss.acc_vector(st.y, st.pred(x, "clean"), st.draws) for x in st.rids(base)]
        r["accuracy_mean"] = float(np.mean([a for a, _ in accs]))
        rows.append(r)
    df = pd.DataFrame(rows)
    comp = []
    for b in BASELINES:
        if st.rids("Proposed") and st.rids(b):
            comp.append({"comparison": f"Proposed vs {b}", **ss.compare(st.level("Proposed", "clean"), st.level(b, "clean"))})
    st.clean_comp = pd.DataFrame(comp)
    return df


def robustness_tables(st: Store) -> tuple[pd.DataFrame, pd.DataFrame]:
    conds = list(st.rb.COND)
    rows, comps = [], []
    for base in MAIN:
        for c in conds:
            lvl = st.level(base, f"rob:{c}")
            chg = st.change(base, f"rob:{c}")
            if not lvl:
                continue
            m, vec, pts, sd = ss.seed_mean(chg)
            lo, hi = ss.interval(vec)
            rows.append({"model": base, "condition": c, "core": c in st.rb.CORE, "f1_per_seed": [round(p[0], 4) for p in lvl],
                         "f1_seed_mean": float(np.mean([p[0] for p in lvl])), "change_seed_mean": m, "change_sd_across_seeds": sd,
                         "change_ci_lo": lo, "change_ci_hi": hi, "within_2_points": bool(m >= -0.02)})
    for b in BASELINES:
        for c in conds:
            a, bb = st.change("Proposed", f"rob:{c}"), st.change(b, f"rob:{c}")
            if a and bb:
                v = ss.compare(a, bb, **ROB_WORDS)
                comps.append({"baseline": b, "condition": c, "core": c in st.rb.CORE, **v})
    return pd.DataFrame(rows), pd.DataFrame(comps)


def background_tables(st: Store) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = {"stone_only_change": ("bg:fg_only", "change"), "backdrop_only_f1": ("bg:bg_only", "level"),
             "white_backdrop_change": ("bg:swap_white", "change"), "black_backdrop_change": ("bg:swap_black", "change"),
             "other_backdrop_change": ("bg:swap_other", "change")}
    rows, comps = [], []
    for base in MAIN:
        for name, (key, kind) in stats.items():
            items = st.change(base, key) if kind == "change" else st.level(base, key)
            if items:
                rows.append({**_row_level(base, items), "statistic": name})
    for b in BASELINES:
        for name in ("stone_only_change", "backdrop_only_f1", "white_backdrop_change"):
            key, kind = stats[name]
            f = st.change if kind == "change" else st.level
            if f("Proposed", key) and f(b, key):
                words = dict(higher="Proposed higher", lower="Proposed lower")
                comps.append({"baseline": b, "statistic": name, **ss.compare(f("Proposed", key), f(b, key), **words)})
    # where the changed predictions go when the stone is put on white
    dest = []
    for rid in st.data:
        o, w = st.pred(rid, "clean"), st.pred(rid, "bg:swap_white")
        ch = o != w
        if ch.sum():
            d = np.bincount(w[ch], minlength=st.k)
            dest.append({"run": rid, "n_changed": int(ch.sum()), "top_destination": st.classes[int(d.argmax())],
                         "to_Blue_Sapphire_share": float(d[st.classes.index("Blue Sapphire")] / ch.sum())})
    st.dest = pd.DataFrame(dest)
    return pd.DataFrame(rows), pd.DataFrame(comps)


def ablation_tables(st: Store) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = list(ABL) + ["Proposed"]
    core = [f"rob:{c}" for c in st.rb.CORE]
    rows, comps = [], []
    for v in variants:
        if not st.rids(v):
            continue
        r = _row_level(ABL_NAME[v], st.level(v, "clean"))
        r["accuracy_mean"] = float(np.mean([ss.acc_vector(st.y, st.pred(x, "clean"), st.draws)[0] for x in st.rids(v)]))
        drops = [ss.seed_mean(st.change(v, k))[0] for k in core]
        r["worst_core_change"] = float(min(drops))
        r["core_within_2_points"] = int(sum(d >= -0.02 for d in drops))
        r["params_note"] = ""
        rows.append(r)
    pairs = [("Proposed", "Abl-A-FT", "E vs A-FT (colour/texture + gates, equal budget)"), ("Proposed", "Abl-D", "E vs D (fine-tuning)"),
             ("Abl-D", "Abl-C", "D vs C (gating vs concatenation)"), ("Abl-D", "Abl-A", "D vs A (adding colour/texture, frozen)")]
    for a, b, label in pairs:
        if not (st.rids(a) and st.rids(b)):
            continue
        comps.append({"comparison": label, "measure": "clean macro-F1", **ss.compare(st.level(a, "clean"), st.level(b, "clean"))})
        for c in st.rb.CORE:
            comps.append({"comparison": label, "measure": f"change under {c}",
                          **ss.compare(st.change(a, f"rob:{c}"), st.change(b, f"rob:{c}"), higher="first loses less", lower="first loses more")})
    return pd.DataFrame(rows), pd.DataFrame(comps)


def gradcam_table(st: Store) -> pd.DataFrame:
    rows = []
    for base in MAIN + ("ResNet50", "DenseNet121", "MobileNetV2"):
        rids = st.rids(base) if base in MAIN else [r for r in [f"{base}_s42"] if r in st.data]
        if not rids:
            continue
        for v in ("original", "swap_white"):
            per_stat = {}
            for stat in ("share", "area", "enrichment", "point20", "peak_on"):
                items = [ss.mean_vector(st.data[r][f"cam|{v}|{stat}"].astype(float), st.draws) for r in rids]
                m, vec, pts, sd = ss.seed_mean(items)
                per_stat[stat] = (m, ss.interval(vec), sd)
            row = {"model": base, "variant": v, "n_seeds": len(rids)}
            for stat, (m, (lo, hi), sd) in per_stat.items():
                row.update({stat: m, f"{stat}_ci_lo": lo, f"{stat}_ci_hi": hi, f"{stat}_sd_seeds": sd})
            if v == "swap_white":  # attention when the model's answer changed against when it did not
                ch = np.concatenate([(st.pred(r, "clean") != st.pred(r, "bg:swap_white")) for r in rids])
                sh = np.concatenate([st.data[r]["cam|swap_white|share"] for r in rids])
                row["share_when_changed"] = float(np.nanmean(sh[ch])) if ch.any() else float("nan")
                row["share_when_unchanged"] = float(np.nanmean(sh[~ch]))
                row["n_changed"] = int(ch.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def build(split: str, directory: Path | None = None, out: Path | None = None, n_draws: int = ss.N_DRAWS) -> Path:
    st = Store(split, directory=directory, n_draws=n_draws)
    out = out or ROOT / "results" / ("final_tables" if split == "test" else "final_tables_validation")
    out.mkdir(parents=True, exist_ok=True)
    clean = clean_table(st)
    rob, rob_c = robustness_tables(st)
    bg, bg_c = background_tables(st)
    abl, abl_c = ablation_tables(st)
    cam = gradcam_table(st)
    for name, df in (("clean", clean), ("clean_comparisons", st.clean_comp), ("robustness_seeds", rob), ("robustness_comparisons", rob_c),
                     ("background_seeds", bg), ("background_comparisons", bg_c), ("white_swap_destinations", st.dest),
                     ("ablation", abl), ("ablation_comparisons", abl_c), ("gradcam", cam)):
        df.to_csv(out / f"{name}.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    d = build(ap.parse_args().split)
    print("wrote", d)
