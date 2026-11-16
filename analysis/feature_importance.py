"""Consensus feature ranking per base city (Section 5.5, Figure 7): the mean
rank of each feature across the five models (SHAP for four of them,
permutation importance for Histogram-based gradient boosting), computed on
each city's test split.

Output: feature_importance.json (six highest-ranked features, for the
figure) and feature_importance_full.json (every feature's rank).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_structural_system import assess
from ml_structural_system.config import load_config

GROUP_DIR = Path(__file__).resolve().parent.parent / "experiments"
RESULTS_DIR = GROUP_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "HistGradientBoosting"]

out = {}
out_full = {}
for exp, city in [("guatemala", "Guatemala"), ("san_jose", "San Jose"), ("santo_domingo", "Santo Domingo")]:
    cfg = load_config(GROUP_DIR / exp / "config.yaml")
    df = pd.read_csv(GROUP_DIR / exp / "preprocessed_splits" / "test_preprocessed.csv")
    label = cfg.label
    X = df.drop(columns=[label])
    y_raw = df[label]
    name_to_code = {name: code for code, name in enumerate(cfg.class_names)}
    y = y_raw.astype(str).map(name_to_code) if not pd.api.types.is_integer_dtype(y_raw) else y_raw

    result = assess.run(
        cfg, X, y, models=MODELS, log_to_mlflow=False, explain=True, compute_shap=True,
        compute_bootstrap=False,
    )
    consensus = result.explainability.consensus_ranking.sort_values()
    top6 = consensus.head(6)
    print(f"=== {city} (5-model consensus) ===")
    print(top6)
    out[city] = {k: float(v) for k, v in top6.items()}
    # Full ranking (not just top 6) so a feature's exact rank/total is
    # citable directly, e.g. results.tex's "which features are safe to
    # drop" discussion in sec:exp-interpretability needs year's and
    # code_quality's rank, not just whether they clear the top 6.
    out_full.setdefault(city, {})
    n_total = len(consensus)
    out_full[city] = {
        "n_features": n_total,
        "rank": {name: i + 1 for i, name in enumerate(consensus.index)},
    }

with open(RESULTS_DIR / "feature_importance.json", "w") as f:
    json.dump(out, f, indent=2)
with open(RESULTS_DIR / "feature_importance_full.json", "w") as f:
    json.dump(out_full, f, indent=2)
print("written", RESULTS_DIR / "feature_importance.json", RESULTS_DIR / "feature_importance_full.json")
