"""Download Overture Maps building footprints around the surveys whose own
files do not cover the surroundings (Figure 1 background).

Overture merges OpenStreetMap, Microsoft and Google Open Buildings footprints.
Needs uv (runs the overturemaps command line with uvx).

Output: data/context/overture_buildings_<survey>.gpkg, with the geometry and
the source dataset of each footprint.
Data: Overture Maps Foundation, buildings theme (attribution per source:
(c) OpenStreetMap contributors, Microsoft, Google), Open Database License.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "context"
MARGIN = 0.006  # degrees around the survey (about 650 m)
SURVEYS = {"zona1": "guatemala/zona1.gpkg", "mata_redonda": "san_jose/mata_redonda.gpkg"}


def overture_buildings(bounds) -> gpd.GeoDataFrame:
    """Overture buildings intersecting bounds (lon/lat)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "buildings.parquet"
        subprocess.run(["uvx", "overturemaps", "download", f"--bbox={','.join(f'{v:.4f}' for v in bounds)}",
                        "-f", "geoparquet", "--type=building", "-o", str(path)], check=True)
        g = gpd.read_parquet(path)
    g["source"] = g["sources"].map(lambda s: s[0]["dataset"])
    return g[["source", "geometry"]].set_crs(4326, allow_override=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rel in SURVEYS.items():
        minx, miny, maxx, maxy = gpd.read_file(ROOT / "data" / rel).to_crs(4326).total_bounds
        g = overture_buildings((minx - MARGIN, miny - MARGIN, maxx + MARGIN, maxy + MARGIN))
        path = OUT / f"overture_buildings_{name}.gpkg"
        path.unlink(missing_ok=True)
        g.to_file(path, driver="GPKG", layer=path.stem)
        print(f"{path}: {len(g)} buildings")


if __name__ == "__main__":
    main()
