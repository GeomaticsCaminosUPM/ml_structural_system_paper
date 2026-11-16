"""``year`` <-> ``code_quality`` mapping, per national seismic design code."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

CODE_YEAR_TABLES: dict[str, dict[int, str]] = {
    "costa_rica": {1974: "low_code", 1986: "medium_code", 2003: "high_code"},
    "guatemala": {1996: "low_code"},
    "dominican_republic": {1979: "low_code", 2011: "medium_code"},
}


def available_code_types() -> list[str]:
    """List the ``code_type`` keys with a configured year table."""
    return sorted(CODE_YEAR_TABLES)


def _thresholds(code_type: str) -> list[tuple[int, str]]:
    """Look up and sort the (year, label) thresholds for ``code_type``."""
    if code_type not in CODE_YEAR_TABLES:
        raise ValueError(f"Unknown code_type '{code_type}'. Known types: {available_code_types()}")
    return sorted(CODE_YEAR_TABLES[code_type].items())


def _year_to_code_quality(year: float, thresholds: list[tuple[int, str]]) -> str | None:
    """Map a single construction year to its ``code_quality`` label."""
    if pd.isna(year):
        return None
    quality = "pre_code"
    for threshold_year, label in thresholds:
        if year >= threshold_year:
            quality = label
    return quality


def year_series_to_code_quality(years: pd.Series, code_type: str) -> pd.Series:
    """Map each value of ``years`` to a ``code_quality`` label for ``code_type``."""
    thresholds = _thresholds(code_type)
    return years.map(lambda y: _year_to_code_quality(y, thresholds))


PRE_CODE_YEARS_BEFORE_FIRST = 20


def code_quality_to_year_series(qualities: pd.Series, code_type: str) -> pd.Series:
    """Map each ``code_quality`` label back to the first year of its era for ``code_type``.
    """
    thresholds = _thresholds(code_type)
    label_to_year = {label: year for year, label in thresholds}
    if thresholds:
        label_to_year.setdefault("pre_code", thresholds[0][0] - PRE_CODE_YEARS_BEFORE_FIRST)
    return qualities.map(label_to_year)


def apply_code_quality_mapping(
    df: pd.DataFrame,
    feature_columns: list[str],
    code_type: str | None,
    source_label: str = "",
) -> pd.DataFrame:
    """Fill in ``code_quality`` from ``year``, or ``year`` from ``code_quality``."""
    prefix = f"[{source_label}] " if source_label else ""
    needs_code_quality = "code_quality" in feature_columns and "code_quality" not in df.columns
    needs_year = "year" in feature_columns and "year" not in df.columns

    if needs_code_quality and "year" in df.columns:
        if code_type is None:
            logger.warning(
                "%s'code_quality' is a needed feature and a 'year' column is present, "
                "but no code_type is configured for this source, skipping the mapping.",
                prefix,
            )
            return df
        df = df.copy()
        df["code_quality"] = year_series_to_code_quality(df["year"], code_type)
        n_mapped = int(df["code_quality"].notna().sum())
        logger.info(
            "%sMapped 'year' -> 'code_quality' (code_type='%s') for %d building(s).",
            prefix, code_type, n_mapped,
        )
    elif needs_year and "code_quality" in df.columns:
        if code_type is None:
            logger.warning(
                "%s'year' is a needed feature and a 'code_quality' column is present, "
                "but no code_type is configured for this source, skipping the mapping.",
                prefix,
            )
            return df
        df = df.copy()
        df["year"] = code_quality_to_year_series(df["code_quality"], code_type)
        n_mapped = int(df["year"].notna().sum())
        logger.info(
            "%sMapped 'code_quality' -> 'year' (code_type='%s') for %d building(s).",
            prefix, code_type, n_mapped,
        )

    return df
