"""Footprint indices of every building, computed with footprint_attributes.

The 18 plan-shape indices and the 5 neighbor descriptors are computed on the
real footprints (data_private/), where every neighbor of a surveyed building
is in place. The contact forces are proportional to the shared wall area, so
they use each building's height (missing heights take the median of their
file). Walls closer than CONTACT_BUFFER_M count as shared: neighboring
footprints that touch rarely have exactly collinear edges, and without this
tolerance most of such a shared wall can be missed.

Writes the indices of every building into data_private/ and those of the
surveyed buildings into data/ (matched by id). The unsurveyed footprints of
data/ are altered, so their indices are computed on the altered footprints
(with the surveyed buildings as neighbors) and written into data/ too, for the
map's predictions.

Usage:
    python analysis/footprint_features.py [--jobs N] [--only private|public]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import footprint_attributes
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRIVATE, PUBLIC = ROOT / "data_private", ROOT / "data"
LABEL = "survey_structural_system"
SHAPE = [
    "EC8_eccentricityRatio", "EC8_radiusRatio", "EC8_compactness",
    "ASCE7_setbackRatio", "ASCE7_holeRatio", "ASCE7_parallelityAngle",
    "GNDTII_beta1_mainShapeSlenderness", "GNDTII_beta2_setbackRatio",
    "GNDTII_beta4_eccentricityRatio", "GNDTII_beta6_setbackSlenderness",
    "CSCR2010_eccentricityRatio", "NTC23_setbackRatio", "NTC23_holeRatio",
    "polsby_popper", "convex_hull_irregularity", "inertia_circle_ratio",
    "slenderness_bbox", "slenderness_inertia",
]
CONTACT = ["contact_force", "contact_confinementRatio", "contact_angularAcc", "contact_angle"]
POSITION = "relativePosition"  # called blockPosition by footprint_attributes 1.1
FEATURES = [*SHAPE, *CONTACT, POSITION]
CONTACT_BUFFER_M = 0.10


def indices(g: gpd.GeoDataFrame, jobs: int) -> pd.DataFrame:
    """Footprint indices of every row of g."""
    work = g[["geometry"]].copy()
    work["_h"] = g["height"].fillna(g["height"].median())
    out = footprint_attributes.run(
        work, {"columns": [*SHAPE, "position"], "position": {"height_column": "_h", "buffer": CONTACT_BUFFER_M}}, n_jobs=jobs)
    out = out.rename(columns={"blockPosition": POSITION})
    return pd.DataFrame(out[FEATURES].to_numpy(), index=g.index, columns=FEATURES).astype(
        {c: float for c in [*SHAPE, *CONTACT]})


def write(g: gpd.GeoDataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.gpkg")
    tmp.unlink(missing_ok=True)
    g.to_file(tmp, driver="GPKG", layer=path.stem)
    tmp.replace(path)


def public_unsurveyed(jobs: int) -> None:
    """Indices of the unsurveyed (altered) footprints of data/, surveyed rows untouched."""
    for path in sorted(PUBLIC.glob("*/*.gpkg")):
        if path.parent.name in ("context", "map"):
            continue
        g = gpd.read_file(path)
        unsurveyed = g[LABEL].isna()
        new = indices(g, jobs)
        for col in FEATURES:
            g.loc[unsurveyed, col] = new.loc[unsurveyed, col]
        write(g, path)
        print(f"{path.relative_to(ROOT)}: indices of {int(unsurveyed.sum()):,} unsurveyed buildings", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=8, help="worker processes (-1: every core)")
    parser.add_argument("--only", choices=["private", "public"],
                        help="private: real footprints and the surveyed rows of data/; public: unsurveyed rows of data/")
    args = parser.parse_args()
    jobs = args.jobs
    if args.only != "private":
        public_unsurveyed(jobs)
    if args.only == "public":
        return
    if not PRIVATE.is_dir():
        raise SystemExit("data_private/ is needed: the unsurveyed footprints of data/ are not the real ones")
    for src in sorted(PRIVATE.glob("*/*.gpkg")):
        if src.parent.name.startswith("_"):
            continue
        g = gpd.read_file(src)
        new = indices(g, jobs)
        changed = (g.loc[g[LABEL].notna(), POSITION] != new.loc[g[LABEL].notna(), POSITION]).sum()
        g[FEATURES] = new
        write(g, src)

        dst = PUBLIC / src.relative_to(PRIVATE)
        pub = gpd.read_file(dst)
        surveyed = pub[LABEL].notna()
        by_id = g.set_index("id").loc[pub.loc[surveyed, "id"], FEATURES]
        assert len(by_id) == surveyed.sum(), f"{dst}: surveyed ids missing from {src}"
        for col in FEATURES:
            pub.loc[surveyed, col] = by_id[col].to_numpy()
        write(pub, dst)
        print(f"{src.relative_to(ROOT)}: {len(g):,} buildings, {int(g[LABEL].notna().sum()):,} surveyed, "
              f"position changed for {changed:,} surveyed", flush=True)


if __name__ == "__main__":
    main()
