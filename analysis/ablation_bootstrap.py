"""Attribute ablation (no roof material, code quality or construction era):
change in Cohen's kappa and macro F1 of each city's best base model
(Table tab:basecities) on the same test buildings, with a 95% paired
percentile block bootstrap (2,000 resamples of the test set's spatial
blocks; both models are scored on the same resample).

Needs report/per_sample_5models.csv in <city> and <city>_no_roof_code_year
(the Guatemala City one from experiments/guatemala_ablation.py).

Output: ablation_bootstrap.json plus a printout.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from results_analysis import RESULTS_DIR, GROUP_DIR, load_class_names, load_sample_df

N_BOOT = 2000
# Each city's best base model (highest test macro F1), from results_analysis.py.
_RANKING = json.loads((RESULTS_DIR / "results_analysis.json").read_text())["base_cities"]
BEST = {exp: _RANKING[city]["ranking"][0]["model"]
        for exp, city in [("guatemala", "Guatemala"), ("san_jose", "San Jose"), ("santo_domingo", "Santo Domingo")]}


def main() -> None:
    rng = np.random.default_rng(42)
    out = {}
    for city, model in BEST.items():
        base, abl = load_sample_df(city), load_sample_df(f"{city}_no_roof_code_year")
        y = base["y_true"].to_numpy()
        assert len(base) == len(abl) and (y == abl["y_true"].to_numpy()).all(), f"{city}: test sets differ"
        p0, p1 = base[f"pred_{model}"].to_numpy(), abl[f"pred_{model}"].to_numpy()
        labels = sorted(load_class_names(city))
        blocks = gpd.read_file(GROUP_DIR / city / "splits" / "test_df.gpkg")["block_id"].to_numpy()
        _, inv = np.unique(blocks, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]

        def diff(i):
            k = cohen_kappa_score(y[i], p1[i], labels=labels) - cohen_kappa_score(y[i], p0[i], labels=labels)
            f = (f1_score(y[i], p1[i], average="macro", labels=labels, zero_division=0)
                 - f1_score(y[i], p0[i], average="macro", labels=labels, zero_division=0))
            return k, f

        point = diff(np.arange(len(y)))
        boot = np.array([diff(np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]))
                         for _ in range(N_BOOT)])
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        out[city] = {
            "model": model, "n_test": len(y),
            "kappa_base": float(cohen_kappa_score(y, p0, labels=labels)),
            "kappa_ablation": float(cohen_kappa_score(y, p1, labels=labels)),
            "delta_kappa": float(point[0]), "delta_kappa_ci95": ci[:, 0].tolist(),
            "delta_f1": float(point[1]), "delta_f1_ci95": ci[:, 1].tolist(),
        }
        r = out[city]
        print(f"{city:14s} {model:22s} kappa {r['kappa_base']:.3f} -> {r['kappa_ablation']:.3f} "
              f"dk={r['delta_kappa']:+.3f} [{ci[0, 0]:+.3f}, {ci[1, 0]:+.3f}]  "
              f"dF1={r['delta_f1']:+.3f} [{ci[0, 1]:+.3f}, {ci[1, 1]:+.3f}]")
    path = RESULTS_DIR / "ablation_bootstrap.json"
    path.write_text(json.dumps(out, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
