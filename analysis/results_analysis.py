"""Base-city results (Section 5.1, Table tab:basecities) and the shared
helpers the other analysis scripts import.

Reads each sub-experiment's report/per_sample_5models.csv (written by
run_5model_assess.sh), which has one row per test-set building: every
feature column, one `pred_<Model>` column for each of the five models, and
`y_true`. Computes, per city: macro F1, accuracy, Cohen's kappa and TSS of
every model and of the majority-class baseline; the errors shared by all
five models; and Cohen's d of building height (in meters, from the test
split, with the reconciled height and with GlobalBuildingAtlas height
alone) between the two classes the paper compares.

Output: results_analysis.json plus a printout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
GROUP_DIR = ROOT / "experiments"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "HistGradientBoosting"]


def macro_tss(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    total = cm.sum()
    per_class = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        per_class.append(sens + spec - 1)
    return float(np.nanmean(per_class))


def model_metrics(df: pd.DataFrame, model: str, labels: list[int]) -> dict:
    y_true = df["y_true"].to_numpy()
    y_pred = df[f"pred_{model}"].to_numpy()
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "tss_macro": macro_tss(y_true, y_pred, labels),
    }


def rank_models(df: pd.DataFrame, labels: list[int]) -> list[dict]:
    rows = []
    for m in MODELS:
        if f"pred_{m}" not in df.columns:
            continue
        met = model_metrics(df, m, labels)
        met["model"] = m
        rows.append(met)
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    return rows


def majority_baseline(df: pd.DataFrame, labels: list[int]) -> dict:
    y_true = df["y_true"].to_numpy()
    majority = pd.Series(y_true).mode().iloc[0]
    y_pred = np.full_like(y_true, majority)
    return model_metrics(df.assign(**{"pred_maj": y_pred}), "maj", labels)


def unanimous_agreement(df: pd.DataFrame, class_names: dict[int, str]) -> dict:
    pred_cols = [f"pred_{m}" for m in MODELS if f"pred_{m}" in df.columns]
    preds = df[pred_cols].to_numpy()
    unanimous_mask = (preds == preds[:, [0]]).all(axis=1)
    unanimous_df = df[unanimous_mask]
    n_unanimous = len(unanimous_df)
    wrong_mask = unanimous_df[pred_cols[0]].to_numpy() != unanimous_df["y_true"].to_numpy()
    n_wrong = int(wrong_mask.sum())
    wrong_df = unanimous_df[wrong_mask]
    pair_counts: dict[str, int] = {}
    for true_c, pred_c in zip(wrong_df["y_true"], wrong_df[pred_cols[0]]):
        pair = f"{class_names.get(int(true_c), true_c)}->{class_names.get(int(pred_c), pred_c)}"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return {
        "n_test": len(df),
        "n_unanimous": n_unanimous,
        "pct_unanimous": n_unanimous / len(df) * 100,
        "n_unanimous_wrong": n_wrong,
        "pct_of_test_unanimous_wrong": n_wrong / len(df) * 100,
        "pct_of_unanimous_wrong": (n_wrong / n_unanimous * 100) if n_unanimous else float("nan"),
        "confusion_pairs": dict(sorted(pair_counts.items(), key=lambda kv: -kv[1])),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_std)


def height_cohens_d(exp: str, pair: tuple[str, str]) -> dict:
    """Cohen's d of height between two classes on the test split, in meters."""
    import geopandas as gpd
    import yaml
    with open(GROUP_DIR / "main.yaml") as f:
        replacements = yaml.safe_load(f)["preprocessing"]["label_replacements"]
    test = gpd.read_file(GROUP_DIR / exp / "splits" / "test_df.gpkg", ignore_geometry=True)
    label = test["survey_structural_system"].map(lambda v: replacements.get(v, v))
    out = {}
    for col in ("height", "height_gba"):
        h = test[col].astype(float).to_numpy()
        out[col] = abs(cohens_d(h[(label == pair[0]).to_numpy()], h[(label == pair[1]).to_numpy()]))
    return out


def load_class_names(exp: str) -> dict[int, str]:
    import yaml
    with open(GROUP_DIR / exp / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    enc = cfg.get("cost_matrix", {}).get("label_encoding")
    if enc:
        return {v: k for k, v in enc.items()}
    # extends main.yaml's own default, fall back to reading preprocessing.json
    import json as _json
    prep = _json.load(open(GROUP_DIR / exp / "preprocessed_splits" / "preprocessing.json"))
    return {v: k for k, v in prep["label_mapping"].items()}


def load_sample_df(exp: str) -> pd.DataFrame:
    path = GROUP_DIR / exp / "report" / "per_sample_5models.csv"
    return pd.read_csv(path)


def main() -> None:
    report: dict = {}

    # ---- base cities -----------------------------------------------------
    report["base_cities"] = {}
    for exp, city in [("guatemala", "Guatemala"), ("san_jose", "San Jose"), ("santo_domingo", "Santo Domingo")]:
        df = load_sample_df(exp)
        class_names = load_class_names(exp)
        labels = sorted(class_names)
        ranking = rank_models(df, labels)
        maj = majority_baseline(df, labels)
        unanimous = unanimous_agreement(df, class_names)

        entry = {
            "class_names": class_names,
            "n_test": len(df),
            "ranking": ranking,
            "majority_baseline": maj,
            "unanimous_agreement": unanimous,
        }
        report["base_cities"][city] = entry
        print(f"=== {city} ===")
        for r in ranking:
            print(f"  {r['model']:22s} f1={r['macro_f1']:.3f} acc={r['accuracy']:.3f} "
                  f"kappa={r['cohen_kappa']:.3f} tss={r['tss_macro']:.3f}")
        print(f"  majority: f1={maj['macro_f1']:.3f} acc={maj['accuracy']:.3f} kappa={maj['cohen_kappa']:.3f} tss={maj['tss_macro']:.3f}")
        print(f"  unanimous agreement: {unanimous}")
        print()

    # ---- height Cohen's d (meters) ----------------------------------
    report["height_cohens_d"] = {}
    for exp, city, pair in [
        ("guatemala", "Guatemala", ("CR", "M")),
        ("santo_domingo", "Santo Domingo", ("CR", "M")),
        ("san_jose", "San Jose", ("W", "M")),
    ]:
        d = height_cohens_d(exp, pair)
        report["height_cohens_d"][city] = {"pair": pair, **d}
        print(f"{city}: Cohen's d height {pair[0]} vs {pair[1]} = {d['height']:.2f} "
              f"(GlobalBuildingAtlas only: {d['height_gba']:.2f})")

    out_path = RESULTS_DIR / "results_analysis.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
