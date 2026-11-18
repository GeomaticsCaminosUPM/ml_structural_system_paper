"""Learning-curve numbers quoted in Section 5.2: XGBoost test macro F1 at the
smallest and largest training-set sizes of each city, using only points
backed by at least three valid repeats (as in Figure 5).

Output: learning_curve_stats.json plus a printout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MIN_REPEATS = 3


def main() -> None:
    df = pd.read_csv(ROOT / "experiments" / "learning_curve_summary.csv")
    df = df[(df["model"] == "XGBoost") & (df["count"] >= MIN_REPEATS)]
    out = {}
    for city, sub in df.groupby("city"):
        sub = sub.sort_values("target_size")
        first, last = sub.iloc[0], sub.iloc[-1]
        at_100 = sub.loc[sub["target_size"] == 100, "mean"]
        out[city] = {
            "points": {int(r.target_size): round(float(r["mean"]), 4) for _, r in sub.iterrows()},
            "smallest_size": int(first.target_size), "f1_smallest": float(first["mean"]),
            "largest_size": int(last.target_size), "f1_largest": float(last["mean"]),
            "gain": float(last["mean"] - first["mean"]),
            "f1_at_100": float(at_100.iloc[0]) if len(at_100) else None,
        }
        r = out[city]
        print(f"{city:14s} {r['smallest_size']:5d}: {r['f1_smallest']:.2f} -> {r['largest_size']:5d}: "
              f"{r['f1_largest']:.2f} (gain {r['gain']:+.2f})")
    path = RESULTS_DIR / "learning_curve_stats.json"
    path.write_text(json.dumps(out, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
