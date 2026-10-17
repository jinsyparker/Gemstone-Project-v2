"""Cross-check of the group-aware split against scikit-learn's StratifiedGroupKFold.

Builds a 70/15/15 split with StratifiedGroupKFold (seeded) from the same kept images and groups,
then compares per-class ratio drift and group integrity with the delivered split.
Writes dataset/split_crosscheck.json. Reads only train/validation/test.csv (labels and groups).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, resolve  # noqa: E402

cfg = load_config()
ds = resolve(cfg, "dataset")
df = pd.concat([pd.read_csv(ds / f"{n}.csv") for n in ("train", "validation", "test")], ignore_index=True)
target = cfg["split_ratios"]


def drift(frame: pd.DataFrame, col: str) -> float:
    """max over classes and splits of |achieved % - target %|, in percentage points."""
    share = pd.crosstab(frame["label"], frame[col], normalize="index") * 100
    return float(max(abs(share[s] - 100 * target[s]).max() for s in target))


# sklearn: one fold of 7 -> test (14.3%); one fold of 6 from the rest -> validation (~14.3% of total)
sgk = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=cfg["seed"])
rest, test = next(sgk.split(df, df["label"], df["group_id"]))
sgk2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=cfg["seed"])
tr_i, va_i = next(sgk2.split(df.iloc[rest], df.iloc[rest]["label"], df.iloc[rest]["group_id"]))
alt = np.full(len(df), "", dtype=object)
alt[test] = "test"
alt[rest[va_i]] = "validation"
alt[rest[tr_i]] = "train"
df["sk_split"] = alt

res = {
    "delivered_split": {
        "max_class_ratio_drift_pts": round(drift(df, "split"), 2),
        "groups_in_multiple_splits": int((df.groupby("group_id")["split"].nunique() > 1).sum()),
        "test_images": int((df["split"] == "test").sum()),
    },
    "sklearn_StratifiedGroupKFold": {
        "max_class_ratio_drift_pts": round(drift(df, "sk_split"), 2),
        "groups_in_multiple_splits": int((df.groupby("group_id")["sk_split"].nunique() > 1).sum()),
        "test_images": int((df["sk_split"] == "test").sum()),
        "overall_pct": {s: round(float((df["sk_split"] == s).mean() * 100), 1) for s in target},
    },
}
(ds / "split_crosscheck.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
