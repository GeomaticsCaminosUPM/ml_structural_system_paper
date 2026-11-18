"""95% confidence intervals for the macro F1 in the paper's results tables,
from a percentile bootstrap that resamples the test set's spatial blocks
(the blocks of touching buildings that define the spatially grouped split)
rather than individual buildings, so the interval reflects the same spatial
dependence the split is designed around.

Classes averaged over match each table: the experiment's label set for the
base-city and pooled tables (every class is present in those test sets), and
the classes present in the target test set for the transfer table (as in
present_class_metrics.py).

Output: block_bootstrap_ci.json plus a printout.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.metrics import f1_score

from results_analysis import RESULTS_DIR, GROUP_DIR, MODELS, load_class_names, load_sample_df

N_BOOT = 2000
SEED = 42

BASE = ["guatemala", "san_jose", "santo_domingo"]
TRANSFER = ["loo_guatemala", "loo_san_jose", "loo_santo_domingo", "zona1_to_zona10",
            "mata_redonda_to_san_jose_sample", "ensanche_quisquella_to_santo_domingo_center"]
POOLED = ["all_cities"]


def block_ci(y_true: np.ndarray, y_pred: np.ndarray, blocks: np.ndarray, labels: list[int],
             rng: np.random.Generator) -> list[float]:
    _, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    scores = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        scores[b] = f1_score(y_true[idx], y_pred[idx], average="macro", labels=labels, zero_division=0)
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {}
    for exp in BASE + TRANSFER + POOLED:
        df = load_sample_df(exp)
        blocks = gpd.read_file(GROUP_DIR / exp / "splits" / "test_df.gpkg")["block_id"].to_numpy()
        assert len(blocks) == len(df), f"{exp}: test_df.gpkg and per-sample rows differ"
        y_true = df["y_true"].to_numpy()
        labels = sorted(np.unique(y_true)) if exp in TRANSFER else sorted(load_class_names(exp))
        out[exp] = {"n_test": len(df), "n_blocks": int(len(np.unique(blocks))), "models": {}}
        for m in MODELS:
            y_pred = df[f"pred_{m}"].to_numpy()
            f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
            ci = block_ci(y_true, y_pred, blocks, labels, rng)
            out[exp]["models"][m] = {"macro_f1": f1, "ci95": ci}
            print(f"{exp:28s} {m:22s} F1={f1:.3f} CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    path = RESULTS_DIR / "block_bootstrap_ci.json"
    path.write_text(json.dumps(out, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
