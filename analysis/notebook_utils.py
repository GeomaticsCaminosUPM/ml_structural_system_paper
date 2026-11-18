"""Helpers for paper_results.ipynb: data loading, folium maps, metric tables and plots.

Only needs geopandas, folium, matplotlib, scikit-learn and pyyaml, so it runs in
the default Google Colab environment. Training runs in the repository's own
environment through reproduce_paper.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from branca.element import Element
from sklearn.metrics import confusion_matrix, f1_score

from paper_style import CLASS_COLORS, CLASS_NAMES, CLASS_ORDER, PARTITION_COLORS
from results_analysis import load_class_names, majority_baseline, model_metrics

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"
MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "HistGradientBoosting"]
REPLACEMENTS = yaml.safe_load((EXPERIMENTS / "main.yaml").read_text())["preprocessing"]["label_replacements"]

SURVEYS = {  # city: [(survey name, file)], larger-area survey first
    "Guatemala City": [("Zona 10", "guatemala/zona10.gpkg"), ("Zona 1", "guatemala/zona1.gpkg")],
    "San José": [("citywide sample", "san_jose/san_jose_sample.gpkg"), ("Mata Redonda", "san_jose/mata_redonda.gpkg")],
    "Santo Domingo": [("citywide sample", "santo_domingo/santo_domingo_center.gpkg"),
                      ("Ensanche Quisqueya", "santo_domingo/ensanche_quisquella.gpkg")],
}
TOOLTIP = ["survey", "class", "height", "year", "code_quality", "roof_material", "relativePosition"]
ERROR_COLOR = "#C0392B"


def final_class(code: object) -> str | None:
    """Map a survey code to the paper's class (ADO, CR, M, S), or None."""
    if not isinstance(code, str):
        return None
    out = REPLACEMENTS.get(code, code)
    return None if out in (None, "None") else out


def load_labeled(city: str) -> gpd.GeoDataFrame:
    """Labeled buildings of both surveys of a city, in WGS 84, with a ``class`` column."""
    parts = []
    for survey, path in SURVEYS[city]:
        g = gpd.read_file(DATA / path)
        g["class"] = g["survey_structural_system"].map(final_class)
        g = g[g["class"].notna()].to_crs(4326)
        g["survey"] = survey
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def class_counts() -> pd.DataFrame:
    """Labeled buildings per city and class (Table 1)."""
    rows = {}
    for city in SURVEYS:
        counts = load_labeled(city)["class"].value_counts()
        rows[city] = {c: int(counts.get(c, 0)) for c in CLASS_ORDER}
    table = pd.DataFrame(rows).T
    table.loc["All three pooled"] = table.sum()
    table.insert(0, "N", table.sum(axis=1))
    return table.rename(columns=CLASS_NAMES)


def _base_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    minx, miny, maxx, maxy = gdf.total_bounds
    return folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=14, tiles="CartoDB positron")


def _fit(m: folium.Map, gdf: gpd.GeoDataFrame) -> None:
    minx, miny, maxx, maxy = gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])


def _legend(m: folium.Map, title: str, items: dict[str, str]) -> None:
    rows = "".join(f'<div><span style="background:{c};width:12px;height:12px;display:inline-block;'
                   f'margin-right:6px;border:1px solid #333"></span>{name}</div>' for name, c in items.items())
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #999;border-radius:4px;font:12px sans-serif"><b>{title}</b>{rows}</div>')
    m.get_root().html.add_child(Element(html))


def _layer(gdf: gpd.GeoDataFrame, name: str, color_of, fields: list[str], show: bool = True,
           line_color=None) -> folium.GeoJson:
    fields = [f for f in fields if f in gdf.columns]
    gdf = gdf[fields + ["geometry"]].copy()
    for f in fields:
        gdf[f] = gdf[f].map(lambda v: round(v, 2) if isinstance(v, float) else v).astype(str)
    return folium.GeoJson(
        gdf, name=name, show=show,
        style_function=lambda feat: {
            "fillColor": color_of(feat["properties"]), "fillOpacity": 0.85, "weight": 0.6,
            "color": line_color(feat["properties"]) if line_color else "#333333"},
        tooltip=folium.GeoJsonTooltip(fields=fields),
    )


def survey_map(city: str) -> folium.Map:
    """Interactive map of the labeled buildings of both surveys of a city, one layer per survey."""
    gdf = load_labeled(city)
    m = _base_map(gdf)
    for survey, _ in SURVEYS[city]:
        sub = gdf[gdf["survey"] == survey]
        _layer(sub, f"{survey} ({len(sub):,} labeled)", lambda p: CLASS_COLORS[p["class"]], TOOLTIP).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _legend(m, "Structural system", {CLASS_NAMES[k]: CLASS_COLORS[k] for k in CLASS_ORDER})
    _fit(m, gdf)
    return m


def feature_by_class(city: str, feature: str) -> plt.Figure:
    """Distribution of one feature per class in a city (box plot)."""
    gdf = load_labeled(city)
    classes = [c for c in CLASS_ORDER if (gdf["class"] == c).sum() >= 5]
    fig, ax = plt.subplots(figsize=(6, 3))
    data = [gdf.loc[gdf["class"] == c, feature].dropna() for c in classes]
    box = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False)
    for patch, c in zip(box["boxes"], classes):
        patch.set_facecolor(CLASS_COLORS[c])
    ax.set_yticks(range(1, len(classes) + 1), [f"{CLASS_NAMES[c]} (n={len(d)})" for c, d in zip(classes, data)])
    ax.set_xlabel(feature)
    ax.set_title(city, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def finished_experiments() -> list[str]:
    """Experiments with per-building test predictions (report/per_sample_5models.csv)."""
    names = yaml.safe_load((EXPERIMENTS / "main.yaml").read_text())["experiments"]
    return [n for n in names if (EXPERIMENTS / n / "report" / "per_sample_5models.csv").exists()]


def split_map(exp: str) -> folium.Map:
    """Interactive map of the spatially grouped train, validation and test partitions."""
    parts = []
    for name, file in [("train", "train_df"), ("validation", "val_df"), ("test", "test_df")]:
        g = gpd.read_file(EXPERIMENTS / exp / "splits" / f"{file}.gpkg").to_crs(4326)
        g["partition"] = name
        g["class"] = g["survey_structural_system"].map(final_class)
        parts.append(g)
    gdf = pd.concat(parts, ignore_index=True)
    m = _base_map(gdf)
    for name in ["train", "validation", "test"]:
        sub = gdf[gdf["partition"] == name]
        _layer(sub, f"{name} ({len(sub):,})", lambda p: PARTITION_COLORS[p["partition"]],
               ["partition", "class", "block_id", "dataset"]).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _legend(m, "Partition", PARTITION_COLORS)
    _fit(m, gdf)
    return m


def predictions(exp: str) -> gpd.GeoDataFrame:
    """Test buildings of an experiment with the true class and every model's prediction."""
    test = gpd.read_file(EXPERIMENTS / exp / "splits" / "test_df.gpkg").to_crs(4326)
    preds = pd.read_csv(EXPERIMENTS / exp / "report" / "per_sample_5models.csv")
    assert len(test) == len(preds), "test_df.gpkg and per_sample_5models.csv do not line up"
    names = load_class_names(exp)
    test["observed"] = preds["y_true"].map(names).to_numpy()
    for model in MODELS:
        test[model] = preds[f"pred_{model}"].map(names).to_numpy()
    test["models_agreeing_with_vote"] = (
        test[MODELS].apply(lambda r: r.value_counts().iloc[0], axis=1).astype(int))
    return test


def metrics(exp: str) -> pd.DataFrame:
    """Macro F1 (over the classes present in the test set), accuracy, kappa and TSS of every model."""
    df = pd.read_csv(EXPERIMENTS / exp / "report" / "per_sample_5models.csv")
    labels = sorted(load_class_names(exp))
    present = sorted(int(c) for c in np.unique(df["y_true"]))
    rows = {}
    for model in MODELS:
        r = model_metrics(df, model, labels)
        r["macro_f1"] = f1_score(df["y_true"], df[f"pred_{model}"], labels=present, average="macro", zero_division=0)
        rows[model] = r
    maj = majority_baseline(df, labels)
    y = df["y_true"].to_numpy()
    maj["macro_f1"] = f1_score(y, np.full_like(y, np.bincount(y).argmax()), labels=present, average="macro",
                               zero_division=0)
    rows["Majority baseline"] = maj
    out = pd.DataFrame(rows).T[["macro_f1", "accuracy", "cohen_kappa", "tss_macro"]]
    out.columns = ["Macro F1", "Accuracy", "Cohen's kappa", "TSS"]
    return out.sort_values("Macro F1", ascending=False).round(3)


def plot_confusion(exp: str, model: str) -> plt.Figure:
    """Row-normalized confusion matrix of one model, with building counts."""
    test = predictions(exp)
    classes = [c for c in CLASS_ORDER if c in set(test["observed"]) | set(test[model])]
    cm = confusion_matrix(test["observed"], test[model], labels=classes)
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 1.5, 1.1 * len(classes) + 1))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{pct[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 60 else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(f"{exp}: {model}", fontsize=10)
    fig.tight_layout()
    return fig


def prediction_map(exp: str, model: str) -> folium.Map:
    """Interactive map of the test buildings: observed class, prediction, errors and model agreement."""
    test = predictions(exp)
    test["correct"] = np.where(test["observed"] == test[model], "yes", "no")
    m = _base_map(test)
    fields = ["observed", model, "correct", "models_agreeing_with_vote", "height", "year", "roof_material",
              "relativePosition"]
    _layer(test, "Observed class", lambda p: CLASS_COLORS[p["observed"]], fields, show=False).add_to(m)
    _layer(test, f"Predicted by {model} (red outline: wrong)", lambda p: CLASS_COLORS[p[model]], fields,
           line_color=lambda p: ERROR_COLOR if p["correct"] == "no" else "#333333").add_to(m)
    agree = {5: "#1F4E79", 4: "#6F9CC3", 3: "#F0B27A", 2: ERROR_COLOR}
    _layer(test, "Agreement among the five models", lambda p: agree.get(int(p["models_agreeing_with_vote"]),
                                                                        ERROR_COLOR), fields, show=False).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _legend(m, "Structural system", {CLASS_NAMES[k]: CLASS_COLORS[k] for k in CLASS_ORDER})
    _fit(m, test)
    return m


# ---------------------------------------------------------------------------
# Paper tables (after the full run)
# ---------------------------------------------------------------------------

def _load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def paper_tables() -> dict[str, pd.DataFrame]:
    """Tables 2 to 4 as DataFrames, from the JSON files of the results step (empty if missing)."""
    res, boot, pcm = _load("results_analysis"), _load("block_bootstrap_ci"), _load("present_class_metrics")
    if not (res and boot and pcm):
        return {}
    base = {"Guatemala": "guatemala", "San Jose": "san_jose", "Santo Domingo": "santo_domingo"}
    rows = []
    for city, exp in base.items():
        for r in res["base_cities"][city]["ranking"]:
            ci = boot[exp]["models"][r["model"]]["ci95"]
            rows.append({"City": city, "Model": r["model"], "Macro F1": r["macro_f1"], "CI low": ci[0],
                         "CI high": ci[1], "Accuracy": r["accuracy"], "Kappa": r["cohen_kappa"], "TSS": r["tss_macro"]})
    table2 = pd.DataFrame(rows).round(2)
    rows = []
    for exp, t in pcm["transfer"].items():
        best = t["ranking"][0]
        ci = boot[exp]["models"][best["model"]]["ci95"]
        rows.append({"Experiment": exp, "Kind": t["kind"], "City": t["city"], "Best model": best["model"],
                     "Macro F1": best["macro_f1"], "CI low": ci[0], "CI high": ci[1],
                     "Delta F1": t["delta_f1"]["delta"], "Kappa": best["cohen_kappa"], "TSS": best["tss_macro"]})
    table3 = pd.DataFrame(rows).round(2)
    table4 = pd.DataFrame({city: {m: v["macro_f1"] for m, v in d["models"].items()}
                           for city, d in pcm["pooled_per_city"].items()})
    table4.insert(0, "All", {m: boot["all_cities"]["models"][m]["macro_f1"] for m in table4.index})
    return {"Table 2: base-city models": table2, "Table 3: transfer": table3,
            "Table 4: pooled model (macro F1 per city)": table4.round(2)}
