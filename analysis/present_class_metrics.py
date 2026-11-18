"""Transfer (Table tab:transfer) and pooled per-city (Table tab:pooled)
metrics.

Macro F1 is averaged over the classes with at least one building in each
test set; a prediction of a class absent from that test set still counts as
an error for the building's true class. Delta F1 compares each transfer
result with the best city-specific model (Table tab:basecities) on the
classes present in both test sets. Kappa and TSS use results_analysis.py's
definitions.

Output: present_class_metrics.json plus a printout.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.metrics import f1_score

from results_analysis import (
    RESULTS_DIR,
    GROUP_DIR,
    MODELS,
    load_class_names,
    load_sample_df,
    majority_baseline,
    model_metrics,
)

warnings.filterwarnings("ignore", category=UserWarning)

BASE = {"Guatemala": "guatemala", "San Jose": "san_jose", "Santo Domingo": "santo_domingo"}
TRANSFER = [
    ("loo_guatemala", "loo", "Guatemala"),
    ("loo_san_jose", "loo", "San Jose"),
    ("loo_santo_domingo", "loo", "Santo Domingo"),
    ("zona1_to_zona10", "district", "Guatemala"),
    ("mata_redonda_to_san_jose_sample", "district", "San Jose"),
    ("ensanche_quisquella_to_santo_domingo_center", "district", "Santo Domingo"),
]
POOLED_CITY = {"Ciudad de Guatemala": "Guatemala", "San Jose": "San Jose", "Santo Domingo": "Santo Domingo"}


def present_labels(y_true: np.ndarray) -> list[int]:
    return sorted(int(c) for c in np.unique(y_true))


def macro_f1(y_true, y_pred, labels) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def evaluate(df, full_labels: list[int]) -> dict:
    """Five models and the majority baseline: macro F1 over present classes, kappa and TSS."""
    y_true = df["y_true"].to_numpy()
    labels = present_labels(y_true)
    rows = []
    for m in MODELS:
        met = model_metrics(df, m, full_labels)
        met["macro_f1"] = macro_f1(y_true, df[f"pred_{m}"].to_numpy(), labels)
        met["model"] = m
        rows.append(met)
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    maj = majority_baseline(df, full_labels)
    majority = np.full_like(y_true, np.bincount(y_true).argmax())
    maj["macro_f1"] = macro_f1(y_true, majority, labels)
    return {"present_labels": labels, "ranking": rows, "majority_baseline": maj}


def main() -> None:
    report: dict = {"base_cities": {}, "transfer": {}, "pooled_per_city": {}}

    base_best = {}
    for city, exp in BASE.items():
        df = load_sample_df(exp)
        names = load_class_names(exp)
        res = evaluate(df, sorted(names))
        assert res["present_labels"] == sorted(names), f"{exp}: label set has a class absent from test"
        best = res["ranking"][0]
        base_best[city] = (df, names, best["model"])
        report["base_cities"][city] = {"best_model": best["model"], "macro_f1": best["macro_f1"]}

    print("=== Transfer (macro F1 over classes present in the test set) ===")
    for exp, kind, city in TRANSFER:
        df = load_sample_df(exp)
        names = load_class_names(exp)
        res = evaluate(df, sorted(names))
        best = res["ranking"][0]
        y_true = df["y_true"].to_numpy()

        # Delta F1 over the classes present in both this test set and the base city's.
        b_df, b_names, b_model = base_best[city]
        shared = sorted({names[c] for c in res["present_labels"]} & {b_names[c] for c in present_labels(b_df["y_true"].to_numpy())})
        t_codes = [c for c, n in names.items() if n in shared]
        b_codes = [c for c, n in b_names.items() if n in shared]
        f1_t_shared = macro_f1(y_true, df[f"pred_{best['model']}"].to_numpy(), t_codes)
        f1_b_shared = macro_f1(b_df["y_true"].to_numpy(), b_df[f"pred_{b_model}"].to_numpy(), b_codes)

        report["transfer"][exp] = {
            **res, "kind": kind, "city": city, "n_test": len(df),
            "present_class_names": [names[c] for c in res["present_labels"]],
            "shared_classes": shared,
            "delta_f1": {"transfer_f1_shared": f1_t_shared, "base_model": b_model,
                         "base_f1_shared": f1_b_shared, "delta": f1_t_shared - f1_b_shared},
        }
        maj = res["majority_baseline"]
        print(f"{exp:30s} n={len(df):5d} classes={report['transfer'][exp]['present_class_names']}")
        for r in res["ranking"]:
            print(f"   {r['model']:22s} f1={r['macro_f1']:.3f} "
                  f"kappa={r['cohen_kappa']:.3f} tss={r['tss_macro']:.3f}")
        print(f"   majority               f1={maj['macro_f1']:.3f}")
        print(f"   shared={shared}  "
              f"dF1={f1_t_shared:.3f}-{f1_b_shared:.3f}({b_model})={f1_t_shared - f1_b_shared:+.3f}")

    print("\n=== Pooled model, per-city macro F1 (classes present in that city's test rows) ===")
    df = load_sample_df("all_cities")
    city = gpd.read_file(GROUP_DIR / "all_cities" / "splits" / "test_df.gpkg")["city"].to_numpy()
    assert len(city) == len(df), "test_df.gpkg and per_sample_5models.csv row counts differ"
    for raw, label in POOLED_CITY.items():
        sub = df[city == raw]
        y_true = sub["y_true"].to_numpy()
        labels = present_labels(y_true)
        out = {m: {"macro_f1": macro_f1(y_true, sub[f"pred_{m}"].to_numpy(), labels)} for m in MODELS}
        report["pooled_per_city"][label] = {"n_test": len(sub), "present_labels": labels, "models": out}
        print(f"{label} (n={len(sub)})")
        for m, v in out.items():
            print(f"   {m:22s} f1={v['macro_f1']:.3f}")

    # Pooled vs city-specific models on the classes the two share (Section 5.4).
    print("\n=== Pooled model, per-city macro F1 on the classes shared with the city-specific model ===")
    report["pooled_shared_with_city"] = {}
    for raw, label in POOLED_CITY.items():
        sub = df[city == raw]
        y_true = sub["y_true"].to_numpy()
        pooled_names = load_class_names("all_cities")
        _, b_names, b_model = base_best[label]
        shared = sorted({pooled_names[c] for c in present_labels(y_true)} & set(b_names.values()))
        codes = [c for c, n in pooled_names.items() if n in shared]
        f1s = {m: macro_f1(y_true, sub[f"pred_{m}"].to_numpy(), codes) for m in MODELS}
        report["pooled_shared_with_city"][label] = {
            "shared_classes": shared, "models": f1s,
            "city_best_model": b_model, "city_best_f1": report["base_cities"][label]["macro_f1"],
        }
        print(f"{label}: shared={shared} pooled {min(f1s.values()):.2f}-{max(f1s.values()):.2f} "
              f"vs city-specific {b_model} {report['base_cities'][label]['macro_f1']:.2f}")

    out_path = RESULTS_DIR / "present_class_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
