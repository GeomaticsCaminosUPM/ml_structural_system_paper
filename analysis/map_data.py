"""Data of the interactive map (docs/map/), all precomputed so the page only displays it.

For each city, every footprint of its two surveys (surveyed or not) gets:
- its attributes (height, era, code level, roof, footprint area, shape and
  neighbor indices),
- the prediction of each of the five city models (experiments/<city>), the
  models trained on that city's surveys.

Every building is predicted from the features stored in data/. The
footprints of unsurveyed buildings in data/ are altered and their height, era,
code level and roof are shuffled, and their shape and neighbor indices were
computed on the altered footprints (analysis/footprint_features.py), so their
predictions do not describe the real buildings.

Writes docs/map/data/buildings_<city>.pmtiles (vector tiles, layer "buildings")
and docs/map/data/experiments.json (per experiment, city and model: test
predictions, scores and confusion matrices, plus map extents).
Needs the experiments step of reproduce_paper.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from ml_structural_system.predict import predict
from results_analysis import MODELS, load_class_names, macro_tss
from results_figures import _round_row_to_100

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXPERIMENTS = ROOT / "experiments"
OUT = ROOT / "docs" / "map" / "data"
MAIN = yaml.safe_load((EXPERIMENTS / "main.yaml").read_text())["preprocessing"]
REPLACEMENTS = MAIN["label_replacements"]
MODEL_KEYS = {"LogisticRegression": "LR", "RandomForest": "RF", "XGBoost": "XGB", "CatBoost": "CB",
              "HistGradientBoosting": "HGB"}

CITIES = {  # map city key: (title, city column value, [(survey title, file)])
    "guatemala": ("Guatemala City", "Ciudad de Guatemala",
                  [("Zona 10", "guatemala/zona10.gpkg"), ("Zona 1", "guatemala/zona1.gpkg")]),
    "san_jose": ("San José", "San Jose",
                 [("Citywide sample", "san_jose/san_jose_sample.gpkg"), ("Mata Redonda", "san_jose/mata_redonda.gpkg")]),
    "santo_domingo": ("Santo Domingo", "Santo Domingo",
                      [("Citywide sample", "santo_domingo/santo_domingo_center.gpkg"),
                       ("Ensanche Quisqueya", "santo_domingo/ensanche_quisquella.gpkg")]),
}
EXPERIMENT_TITLES = {
    "guatemala": "City model: Guatemala City",
    "san_jose": "City model: San José",
    "santo_domingo": "City model: Santo Domingo",
    "all_cities": "Three-city model (combined data)",
    "loo_guatemala": "Leave-one-city-out: Guatemala City",
    "loo_san_jose": "Leave-one-city-out: San José",
    "loo_santo_domingo": "Leave-one-city-out: Santo Domingo",
    "zona1_to_zona10": "District transfer: Zona 1 to Zona 10",
    "mata_redonda_to_san_jose_sample": "District transfer: Mata Redonda to citywide sample",
    "ensanche_quisquella_to_santo_domingo_center": "District transfer: Ensanche Quisqueya to citywide sample",
}
ATTRIBUTES = {  # property name on the map: (column, decimals or None for text)
    "height": ("height", 1),
    "year": ("year", 0),
    "code": ("code_quality", None),
    "roof": ("roof_material", None),
    "position": ("relativePosition", None),
    "hull": ("convex_hull_irregularity", 2),
    "compact": ("EC8_compactness", 2),
    "eccentricity": ("EC8_eccentricityRatio", 2),
    "confinement": ("contact_confinementRatio", 2),
}


def final_class(code: object) -> str | None:
    """Map a survey code to the paper's class, or None."""
    if not isinstance(code, str):
        return None
    out = REPLACEMENTS.get(code, code)
    return None if out in (None, "None") else out


def city_buildings(city: str) -> gpd.GeoDataFrame:
    """Every footprint of both surveys of a city with its attributes and the city models' predictions."""
    model_cols = [f"p_{k}" for k in MODEL_KEYS.values()]
    parts = []
    for survey, path in CITIES[city][2]:
        g = gpd.read_file(DATA / path)
        g["k"] = g["dataset"] + ":" + g["id"].astype(str)
        g["survey"] = survey
        g["cls"] = g["survey_structural_system"].map(final_class)
        g["area"] = g.geometry.area.round(0)  # footprint size, m² (the files are in metric coordinates)
        pred = predict(EXPERIMENTS / city / "config.yaml", g.drop(columns=["survey_structural_system"]))
        for model, key in MODEL_KEYS.items():
            g[f"p_{key}"] = g["k"].map(pred.set_index("k")[f"pred_{model}"])
        for name, (col, dec) in ATTRIBUTES.items():
            g[name] = g[col].round(dec) if dec is not None else g[col]  # after predicting: rounds height in place
        cols = ["k", "survey", "cls", "area", *ATTRIBUTES, *model_cols, "geometry"]
        parts.append(g[cols].to_crs(4326))
        print(f"{CITIES[city][0]}, {survey}: {len(g):,} buildings, {int(g[model_cols[0]].notna().sum()):,} predicted",
              flush=True)
    return pd.concat(parts, ignore_index=True)


def scores(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict:
    """Scores and confusion matrix of one model on one set of test buildings."""
    present = sorted(set(y))
    codes = {c: i for i, c in enumerate(classes)}
    cm = confusion_matrix(y, p, labels=classes)
    return {
        "macro_f1": round(float(f1_score(y, p, labels=present, average="macro", zero_division=0)), 3),
        "accuracy": round(float(accuracy_score(y, p)), 3),
        "kappa": round(float(cohen_kappa_score(y, p)), 3),
        "tss": round(macro_tss(np.vectorize(codes.get)(y), np.vectorize(codes.get)(p),
                               [codes[c] for c in present]), 3),
        "confusion": cm.tolist(),
        # Row percentages rounded to sum to 100, exactly as in Figure 4.
        "percent": [_round_row_to_100(row / row.sum() * 100).tolist() if row.sum() else None for row in cm],
    }


def experiment(name: str) -> dict | None:
    """Per city: test-building keys, true classes, each model's predictions and scores."""
    report = EXPERIMENTS / name / "report" / "per_sample_5models.csv"
    if not report.exists():
        return None
    preds = pd.read_csv(report)
    test = gpd.read_file(EXPERIMENTS / name / "splits" / "test_df.gpkg", ignore_geometry=True)
    assert len(test) == len(preds), f"{name}: test split and predictions do not line up"
    names = load_class_names(name)
    classes = [names[c] for c in sorted(names)]
    keys = (test["dataset"] + ":" + test["id"].astype(str)).to_numpy()
    y = preds["y_true"].map(names).to_numpy()
    by_city = {}
    for city, (_, city_value, _) in CITIES.items():
        mask = (test["city"] == city_value).to_numpy()
        if not mask.any():
            continue
        model_preds = {m: preds.loc[mask, f"pred_{m}"].map(names).tolist() for m in MODELS}
        entry = {"n": int(mask.sum()), "keys": keys[mask].tolist(), "true": y[mask].tolist(), "pred": model_preds,
                 "scores": {m: scores(y[mask], np.array(model_preds[m]), classes) for m in MODELS}}
        entry["best"] = max(MODELS, key=lambda m: entry["scores"][m]["macro_f1"])
        by_city[city] = entry
    return {"title": EXPERIMENT_TITLES[name], "classes": classes, "cities": by_city}


def bounds(gdf: gpd.GeoDataFrame) -> list[float]:
    """Bounding box [west, south, east, north] in degrees."""
    return [round(float(v), 6) for v in gdf.total_bounds]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    extents = {}
    for city in CITIES:
        gdf = city_buildings(city)
        extents[city] = {"all": bounds(gdf), **{s: bounds(gdf[gdf["survey"] == s]) for s, _ in CITIES[city][2]}}
        path = OUT / f"buildings_{city}.pmtiles"
        path.unlink(missing_ok=True)
        # Zooms 11 to 14 at full precision (8192 units per tile, 0.3 m at zoom 14),
        # with no size limit, so no tile is simplified: the map overzooms zoom 14
        # beyond it, and the coarser tiles shown while loading keep the shapes.
        gdf.to_file(path, driver="PMTiles", layer="buildings",
                    dataset_options={"MINZOOM": 11, "MAXZOOM": 14, "EXTENT": 8192, "NAME": f"buildings_{city}",
                                     "MAX_SIZE": 20_000_000, "MAX_FEATURES": 10_000_000})
        print(f"written {path} ({len(gdf):,} buildings, {path.stat().st_size / 1e6:.1f} MB)", flush=True)
    experiments = {n: e for n in EXPERIMENT_TITLES if (e := experiment(n))}
    meta = {"cities": {k: v[0] for k, v in CITIES.items()},
            "surveys": {k: [s for s, _ in v[2]] for k, v in CITIES.items()},
            "extents": extents, "models": MODELS, "model_keys": MODEL_KEYS, "experiments": experiments}
    path = OUT / "experiments.json"
    path.write_text(json.dumps(meta, separators=(",", ":")))
    print(f"written {path} ({len(experiments)} experiments, {path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
