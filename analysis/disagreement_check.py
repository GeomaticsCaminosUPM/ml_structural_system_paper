"""Does disagreement among the five models flag the buildings they get
wrong? (Section 5.3, last paragraph.)

For every test set, per building: the number of distinct labels predicted
by the five models (1 = unanimous) and the vote entropy. Against the error
of the majority vote, reports
  - error rate on unanimous vs non-unanimous buildings,
  - ROC AUC of vote entropy as an error detector,
  - share of all errors that fall on non-unanimous buildings, and how many
    buildings that is.

Output: disagreement_check.json plus a printout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from results_analysis import RESULTS_DIR, MODELS, load_sample_df

EXPERIMENTS = ["guatemala", "san_jose", "santo_domingo",
               "loo_guatemala", "loo_san_jose", "loo_santo_domingo",
               "zona1_to_zona10", "mata_redonda_to_san_jose_sample", "ensanche_quisquella_to_santo_domingo_center",
               "all_cities"]


def main() -> None:
    out = {}
    print(f"{'experiment':28s} {'n':>5s} {'err':>5s} {'%nonun':>7s} {'err|un':>7s} {'err|non':>8s} "
          f"{'AUC':>5s} {'%err found':>10s}")
    for exp in EXPERIMENTS:
        df = load_sample_df(exp)
        y = df["y_true"].to_numpy()
        preds = np.stack([df[f"pred_{m}"].to_numpy() for m in MODELS], axis=1)
        n_distinct = np.array([len(set(r)) for r in preds])
        counts = np.array([np.bincount(r, minlength=preds.max() + 1) for r in preds])
        p = counts / counts.sum(1, keepdims=True)
        entropy = -(np.where(p > 0, p * np.log(p), 0)).sum(1)
        vote = counts.argmax(1)
        err = vote != y
        non = n_distinct > 1
        r = {
            "n": int(len(y)), "error_rate": float(err.mean()), "share_nonunanimous": float(non.mean()),
            "error_rate_unanimous": float(err[~non].mean()) if (~non).any() else None,
            "error_rate_nonunanimous": float(err[non].mean()) if non.any() else None,
            "auc_entropy": float(roc_auc_score(err, entropy)) if 0 < err.sum() < len(err) else None,
            "share_errors_in_nonunanimous": float(err[non].sum() / err.sum()) if err.any() else None,
        }
        out[exp] = r
        f = lambda v: "  -  " if v is None else f"{v:.3f}"
        print(f"{exp:28s} {r['n']:5d} {r['error_rate']:.3f} {r['share_nonunanimous']:7.3f} "
              f"{f(r['error_rate_unanimous']):>7s} {f(r['error_rate_nonunanimous']):>8s} "
              f"{f(r['auc_entropy']):>5s} {f(r['share_errors_in_nonunanimous']):>10s}")
    path = RESULTS_DIR / "disagreement_check.json"
    path.write_text(json.dumps(out, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
