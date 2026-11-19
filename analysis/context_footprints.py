"""Download OpenStreetMap building footprints around the surveys whose own
files do not cover the surroundings (Figure 1 background).

Output: data/context/osm_buildings_<survey>.gpkg, geometry only.
Data (c) OpenStreetMap contributors, Open Database License.
"""
from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "context"
OSM_API = "https://api.openstreetmap.org/api/0.6/map"
MARGIN = 0.006  # degrees around the survey (about 650 m)
SURVEYS = {"zona1": "guatemala/zona1.gpkg", "mata_redonda": "san_jose/mata_redonda.gpkg"}


def osm_buildings(bounds) -> gpd.GeoDataFrame:
    """Closed ways tagged building inside bounds (lon/lat), from the OpenStreetMap API."""
    url = f"{OSM_API}?bbox={','.join(f'{v:.6f}' for v in bounds)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ml_structural_system_paper"})
    with urllib.request.urlopen(req, timeout=300) as r:
        root = ET.parse(r).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    polys = []
    for way in root.iter("way"):
        if not any(t.get("k") == "building" for t in way.iter("tag")):
            continue
        refs = [nd.get("ref") for nd in way.iter("nd")]
        if len(refs) >= 4 and refs[0] == refs[-1] and all(r in nodes for r in refs):
            poly = Polygon([nodes[r] for r in refs])
            if poly.is_valid:
                polys.append(poly)
    return gpd.GeoDataFrame(geometry=polys, crs=4326)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rel in SURVEYS.items():
        minx, miny, maxx, maxy = gpd.read_file(ROOT / "data" / rel).to_crs(4326).total_bounds
        g = osm_buildings((minx - MARGIN, miny - MARGIN, maxx + MARGIN, maxy + MARGIN))
        path = OUT / f"osm_buildings_{name}.gpkg"
        path.unlink(missing_ok=True)
        g.to_file(path, driver="GPKG", layer=path.stem)
        print(f"{path}: {len(g)} buildings")


if __name__ == "__main__":
    main()
