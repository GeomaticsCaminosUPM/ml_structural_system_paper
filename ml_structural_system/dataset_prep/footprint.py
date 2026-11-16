"""``footprint_attributes.run()`` wrapper."""

from __future__ import annotations

import geopandas as gpd


def compute_missing_columns(gdf: gpd.GeoDataFrame, missing_columns: list[str]) -> gpd.GeoDataFrame:
    """Compute ``missing_columns`` on ``gdf`` via ``footprint_attributes.run()``."""
    import footprint_attributes

    original_crs = gdf.crs
    working = gdf
    if original_crs is not None and original_crs.is_geographic:
        working = gdf.to_crs(gdf.estimate_utm_crs())

    result = footprint_attributes.run(working, config={"columns": missing_columns})

    if original_crs is not None and result.crs != original_crs:
        result = result.to_crs(original_crs)
    return result
