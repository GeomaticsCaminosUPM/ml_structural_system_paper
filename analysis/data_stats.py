"""Dataset statistics (Section 3): labeled buildings per city and class
(Table tab:class-imbalance) and buildings surveyed twice, i.e. labeled
footprints of the two surveys of the same city with an intersection over
union of at least 0.5 (Section 3.1).

Output: data_stats.json plus a printout.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CLASSES = ["ADO", "CR", "M", "S"]
IOU_MIN = 0.5
CITIES = {  # city: (single-district survey, larger-area survey)
    "Guatemala": ("guatemala/zona1.gpkg", "guatemala/zona10.gpkg"),
    "San Jose": ("san_jose/mata_redonda.gpkg", "san_jose/san_jose_sample.gpkg"),
    "Santo Domingo": ("santo_domingo/ensanche_quisquella.gpkg", "santo_domingo/santo_domingo_center.gpkg"),
}
REPLACEMENTS = yaml.safe_load((ROOT / "experiments" / "main.yaml").read_text())["preprocessing"]["label_replacements"]


def labeled(path: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(DATA / path)
    g["final_class"] = g["survey_structural_system"].map(
        lambda v: (None if REPLACEMENTS.get(v, v) in (None, "None") else REPLACEMENTS.get(v, v))
        if isinstance(v, str) else None)
    return g[g["final_class"].notna()]


def overlaps(small: gpd.GeoDataFrame, large: gpd.GeoDataFrame) -> pd.DataFrame:
    large = large.to_crs(small.crs)
    pairs = gpd.sjoin(small[["geometry", "final_class"]], large[["geometry", "final_class"]],
                      predicate="intersects")
    rows = []
    for i, row in pairs.iterrows():
        a, b = small.geometry.loc[i], large.geometry.loc[row["index_right"]]
        iou = a.intersection(b).area / a.union(b).area
        if iou >= IOU_MIN:
            rows.append({"same_class": row["final_class_left"] == row["final_class_right"]})
    return pd.DataFrame(rows, columns=["same_class"])


def main() -> None:
    out: dict = {"class_counts": {}, "overlap": {}}
    total = pd.Series(0, index=CLASSES)
    for city, (small_path, large_path) in CITIES.items():
        small, large = labeled(small_path), labeled(large_path)
        counts = pd.concat([small, large])["final_class"].value_counts().reindex(CLASSES, fill_value=0)
        total += counts
        out["class_counts"][city] = {k: int(v) for k, v in counts.items()}
        print(city, int(counts.sum()), counts.to_dict())
        if city != "Guatemala":
            ov = overlaps(small, large)
            out["overlap"][city] = {"n": len(ov), "same_class": int(ov["same_class"].sum()),
                                    "n_labeled_large": len(large), "pct_of_large": 100 * len(ov) / len(large)}
            print("  surveyed twice:", out["overlap"][city])
    out["class_counts"]["All"] = {k: int(v) for k, v in total.items()}
    print("All", int(total.sum()), total.to_dict())
    path = RESULTS_DIR / "data_stats.json"
    path.write_text(json.dumps(out, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
