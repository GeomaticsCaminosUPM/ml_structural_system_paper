"""Base-city models: 95% paired block-bootstrap CI of the macro F1
difference between each city's best and second-ranked model (Table
tab:basecities). Both models are scored on the same resample of the test
set's spatial blocks (2,000 resamples), as in block_bootstrap_ci.py.

Output: best_vs_second_bootstrap.json plus a printout.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.metrics import f1_score

from results_analysis import RESULTS_DIR, GROUP_DIR, MODELS, load_class_names, load_sample_df

N_BOOT = 2000
CITIES = ["guatemala", "san_jose", "santo_domingo"]


def main() -> None:
    rng = np.random.default_rng(42)
    out = {}
    for city in CITIES:
        df = load_sample_df(city)
        y = df["y_true"].to_numpy()
        labels = sorted(load_class_names(city))
        f1 = lambda p, i: f1_score(y[i], p[i], average="macro", labels=labels, zero_division=0)
        everyone = np.arange(len(y))
        ranked = sorted(MODELS, key=lambda m: f1(df[f"pred_{m}"].to_numpy(), everyone), reverse=True)
        a, b = (df[f"pred_{m}"].to_numpy() for m in ranked[:2])
        blocks = gpd.read_file(GROUP_DIR / city / "splits" / "test_df.gpkg")["block_id"].to_numpy()
        _, inv = np.unique(blocks, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
        diffs = []
        for _ in range(N_BOOT):
            idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
            diffs.append(f1(a, idx) - f1(b, idx))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[city] = {"best": ranked[0], "second": ranked[1], "diff": float(f1(a, everyone) - f1(b, everyone)),
                     "ci95": [float(lo), float(hi)], "includes_zero": bool(lo <= 0 <= hi)}
        print(city, out[city])
    path = RESULTS_DIR / "best_vs_second_bootstrap.json"
    path.write_text(json.dumps(out, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
