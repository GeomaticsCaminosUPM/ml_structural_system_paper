"""Loading/saving raw tabular data, and geometry/footprint-attribute prep."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ..exceptions import DatasetError
from . import code_quality as code_quality_mod
from . import footprint

logger = logging.getLogger(__name__)


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make a GeoDataFrame safe to feed into ``footprint_attributes.run()``."""
    gdf = gdf.copy()

    missing_mask = gdf.geometry.isna() | gdf.geometry.is_empty
    n_missing = int(missing_mask.sum())
    if n_missing:
        logger.warning("Dropped %d row(s) with missing/empty geometry.", n_missing)
    gdf = gdf[~missing_mask]

    multi_mask = gdf.geometry.geom_type.isin(["MultiPolygon", "GeometryCollection"])
    n_multi = int(multi_mask.sum())
    if n_multi:
        logger.info("Exploded %d multi-part geometry row(s) into separate polygons.", n_multi)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid:
        repaired = gdf.loc[invalid_mask, "geometry"].buffer(0)
        fixable = repaired.is_valid & ~repaired.is_empty & (repaired.geom_type == "Polygon")
        n_fixed = int(fixable.sum())
        if n_fixed:
            gdf.loc[repaired[fixable].index, "geometry"] = repaired[fixable]
            logger.info(
                "Repaired %d invalid geometry row(s) with buffer(0).", n_fixed,
            )
        if n_invalid - n_fixed:
            logger.warning(
                "%d invalid geometry row(s) could not be repaired with buffer(0) "
                "(still invalid, empty, or no longer a single Polygon afterwards).",
                n_invalid - n_fixed,
            )

    nonpoly_mask = ~gdf.geometry.is_valid | ~gdf.geometry.geom_type.isin(["Polygon"])
    n_nonpoly = int(nonpoly_mask.sum())
    if n_nonpoly:
        logger.warning(
            "Dropped %d row(s) with invalid or non-Polygon geometry, after "
            "multi-part explosion and buffer(0) repair attempts.",
            n_nonpoly,
        )
    gdf = gdf[~nonpoly_mask].reset_index(drop=True)

    return gdf


def _drop_rows_with_missing_features(
    gdf: gpd.GeoDataFrame | pd.DataFrame,
    feature_columns: list[str],
    source_label: str = "",
    label_col: str | None = None,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Drop rows with a missing value in any feature column."""
    if not feature_columns:
        return gdf

    prefix = f"[{source_label}] " if source_label else ""
    missing_mask = gdf[feature_columns].isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    if n_missing:
        per_col = {
            col: int(gdf.loc[missing_mask, col].isna().sum())
            for col in feature_columns
            if gdf.loc[missing_mask, col].isna().any()
        }
        labeled_note = ""
        if label_col is not None and label_col in gdf.columns:
            n_labeled = int((missing_mask & gdf[label_col].notna()).sum())
            labeled_note = (
                f" {n_labeled} of them have a '{label_col}' label and are "
                "lost as training/evaluation samples"
            )
        logger.warning(
            "%sDropped %d building(s) with a missing value in at least one "
            "feature column: %s.%s",
            prefix, n_missing, per_col, labeled_note,
        )
        gdf = gdf[~missing_mask].reset_index(drop=True)
    return gdf


HEIGHT_PER_FLOOR = 3.0  # meters, used to derive one of height/n_floors from the other


def _apply_height_floors_mapping(
    gdf: gpd.GeoDataFrame | pd.DataFrame,
    feature_columns: list[str],
    source_label: str = "",
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Fill a missing ``height`` value from ``n_floors``, or vice versa."""
    if "height" not in feature_columns or "n_floors" not in feature_columns:
        return gdf
    if "height" not in gdf.columns or "n_floors" not in gdf.columns:
        return gdf

    height_missing = gdf["height"].isna()
    floors_missing = gdf["n_floors"].isna()
    fill_height_mask = height_missing & ~floors_missing
    fill_floors_mask = floors_missing & ~height_missing

    if not fill_height_mask.any() and not fill_floors_mask.any():
        return gdf

    prefix = f"[{source_label}] " if source_label else ""
    gdf = gdf.copy()
    if fill_height_mask.any():
        n = int(fill_height_mask.sum())
        gdf.loc[fill_height_mask, "height"] = (
            gdf.loc[fill_height_mask, "n_floors"] * HEIGHT_PER_FLOOR
        )
        logger.warning(
            "%sDerived 'height' from 'n_floors' (x%.1f m/floor) for %d building(s) "
            "missing 'height'.",
            prefix, HEIGHT_PER_FLOOR, n,
        )
    if fill_floors_mask.any():
        n = int(fill_floors_mask.sum())
        gdf.loc[fill_floors_mask, "n_floors"] = (
            gdf.loc[fill_floors_mask, "height"] / HEIGHT_PER_FLOOR
        ).round()
        logger.warning(
            "%sDerived 'n_floors' from 'height' (/%.1f m/floor) for %d building(s) "
            "missing 'n_floors'.",
            prefix, HEIGHT_PER_FLOOR, n,
        )
    return gdf


def ensure_feature_columns(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    code_type: str | None = None,
    source_label: str = "",
    label_col: str | None = None,
    compute_missing_features: bool = True,
) -> gpd.GeoDataFrame:
    """Compute missing feature columns and drop rows with missing values."""
    if not feature_columns:
        return gdf

    missing = [col for col in feature_columns if col not in gdf.columns]
    if missing:
        if not compute_missing_features:
            raise DatasetError(
                f"{source_label + ': ' if source_label else ''}feature column(s) {missing} "
                "are missing and preprocessing.compute_missing_features is False, so "
                "footprint_attributes.run() was not called to compute them. This source was "
                "configured as already-complete data, either fix the source file, or set "
                "compute_missing_features: true to allow computing them."
            )
        gdf = footprint.compute_missing_columns(gdf, missing)

    gdf = _apply_height_floors_mapping(gdf, feature_columns, source_label=source_label)

    gdf = code_quality_mod.apply_code_quality_mapping(
        gdf, feature_columns, code_type, source_label=source_label
    )

    still_missing = [col for col in feature_columns if col not in gdf.columns]
    if still_missing:
        raise DatasetError(
            f"Feature column(s) {still_missing} are not present in the data and "
            "could not be computed by footprint_attributes.run()."
        )

    gdf = _drop_rows_with_missing_features(gdf, feature_columns, source_label, label_col)
    return gdf


def _load_one_tabular_file(path: str | Path) -> gpd.GeoDataFrame | pd.DataFrame:
    """Load a single ``.gpkg`` or ``.csv`` file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".gpkg":
        return gpd.read_file(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    raise DatasetError(f"Unsupported dataset file extension '{suffix}' ({p}).")


def load_tabular_source(
    source: str | Path | list[str | Path],
    feature_columns: list[str] | None = None,
    code_type: str | dict[str, str] | None = None,
    label_col: str | None = None,
    compute_missing_features: bool = True,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Load one or more dataset files and concatenate them into a single frame."""
    paths = [source] if isinstance(source, (str, Path)) else list(source)
    if not paths:
        raise DatasetError("load_tabular_source() received no paths.")

    frames = []
    target_crs = None
    for raw_path in paths:
        frame = _load_one_tabular_file(raw_path)
        if Path(raw_path).suffix.lower() == ".gpkg":
            frame = clean_geometries(frame)
            file_code_type = (
                code_type.get(str(raw_path)) if isinstance(code_type, dict) else code_type
            )
            frame = ensure_feature_columns(
                frame, feature_columns or [], code_type=file_code_type,
                source_label=str(raw_path), label_col=label_col,
                compute_missing_features=compute_missing_features,
            )
            if target_crs is None:
                target_crs = frame.crs
            elif frame.crs != target_crs:
                logger.info(
                    "Reprojecting %s from %s to %s to match the combined dataset's CRS.",
                    raw_path, frame.crs, target_crs,
                )
                frame = frame.to_crs(target_crs)
        # Tags every row with the source file's stem (e.g. "zona10"), always
        #, even for a single source, so downstream code can split/report
        # per-dataset without callers having to configure anything extra.
        # Skipped if a "dataset" column already exists: reloading an
        # already-split train_df.gpkg/test_df.gpkg (e.g. mlss preprocess
        # re-reading mlss split's output) must keep each row's *original*
        # source-zone tag, not get overwritten with the split file's own
        # stem ("train_df"/"test_df").
        frame = frame.copy()
        if "dataset" not in frame.columns:
            frame["dataset"] = Path(raw_path).stem
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if "geometry" in combined.columns:
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=target_crs)
    return combined


def save_tabular(data: gpd.GeoDataFrame | pd.DataFrame, path_without_ext: str | Path) -> Path:
    """Save data as ``.gpkg`` (if it has a geometry column) or ``.csv`` otherwise."""
    path_without_ext = Path(path_without_ext)
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    if "geometry" in data.columns:
        out_path = path_without_ext.with_suffix(".gpkg")
        gpd.GeoDataFrame(data).to_file(out_path, driver="GPKG")
    else:
        out_path = path_without_ext.with_suffix(".csv")
        pd.DataFrame(data).to_csv(out_path, index=False)
    return out_path


def find_tabular_file(directory: str | Path, stem: str) -> Path:
    """Find a dataset file by stem, preferring gpkg over csv."""
    directory = Path(directory)
    for ext in (".gpkg", ".csv"):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Neither {stem}.gpkg nor {stem}.csv found in {directory}.")
