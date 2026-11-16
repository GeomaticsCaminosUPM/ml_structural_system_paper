"""Distribution stats and label/category validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..exceptions import DatasetError
from .config import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class LabelDistribution:
    """Class counts/proportions for one dataframe's label column."""

    counts: dict[str, int]
    proportions: dict[str, float]
    total: int

    @classmethod
    def from_series(cls, s: pd.Series) -> "LabelDistribution":
        """Build a ``LabelDistribution`` from a label column."""
        vc = s.value_counts()
        total = int(vc.sum())
        return cls(
            counts={str(k): int(v) for k, v in vc.items()},
            proportions=({str(k): float(v) / total for k, v in vc.items()} if total else {}),
            total=total,
        )

    def as_text(self, title: str = "Label distribution") -> str:
        """Render this distribution as a human-readable text bar chart."""
        lines = [title + ":"]
        for cls_name, count in self.counts.items():
            pct = self.proportions[cls_name] * 100
            bar = "#" * int(pct / 100 * 30)
            lines.append(f"  {cls_name:<12} {count:>6}  ({pct:5.1f}%)  {bar}")
        return "\n".join(lines)


@dataclass
class GroupDistributionCheck:
    """One (group column, split) distribution-similarity check result."""

    column: str
    split_name: str
    kind: Literal["numeric", "categorical"]
    statistic: float
    p_value: float | None
    ok: bool


def group_checks_text(checks: list[GroupDistributionCheck]) -> str:
    """Render group-distribution check results as human-readable text."""
    if not checks:
        return ""
    lines = ["Group column distribution checks:"]
    for c in checks:
        status = "OK" if c.ok else "WARNING: distribution mismatch"
        if c.kind == "numeric":
            lines.append(
                f"  [{c.column}] {c.split_name} vs overall (KS test): "
                f"stat={c.statistic:.4f}, p={c.p_value:.4f} -> {status}"
            )
        else:
            lines.append(
                f"  [{c.column}] {c.split_name} vs overall (value counts): "
                f"max category diff={c.statistic:.4f} -> {status}"
            )
    return "\n".join(lines)


def check_group_distributions(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: list[str],
    categorical_tolerance: float = 0.05,
    numeric_alpha: float = 0.05,
) -> list[GroupDistributionCheck]:
    """Verify each group column keeps a similar distribution across splits."""
    checks: list[GroupDistributionCheck] = []
    for col in group_cols:
        if col not in full_df.columns:
            logger.warning("Group column '%s' not found in data, skipping.", col)
            continue

        for split_name, split_df in [("train", train_df), ("test", test_df)]:
            if pd.api.types.is_numeric_dtype(full_df[col]):
                stat, p_value = ks_2samp(full_df[col].dropna(), split_df[col].dropna())
                checks.append(
                    GroupDistributionCheck(
                        column=col, split_name=split_name, kind="numeric",
                        statistic=float(stat), p_value=float(p_value),
                        ok=p_value > numeric_alpha,
                    )
                )
            else:
                overall_props = full_df[col].value_counts(normalize=True)
                split_props = (
                    split_df[col].value_counts(normalize=True)
                    .reindex(overall_props.index).fillna(0)
                )
                max_diff = float((overall_props - split_props).abs().max())
                checks.append(
                    GroupDistributionCheck(
                        column=col, split_name=split_name, kind="categorical",
                        statistic=max_diff, p_value=None,
                        ok=max_diff < categorical_tolerance,
                    )
                )
    return checks


def apply_label_replacements(
    df: pd.DataFrame,
    label_col: str,
    label_replacements: dict[str, str | None],
    *,
    drop: bool = True,
) -> pd.DataFrame:
    """Apply configured label replacements/exclusions to a dataframe."""
    if not label_replacements or label_col not in df.columns:
        return df

    drop_values = {
        k for k, v in label_replacements.items()
        if v is None or (isinstance(v, str) and v.strip().lower() == "none")
    }
    keep_replacements = {k: v for k, v in label_replacements.items() if k not in drop_values}

    df = df.copy()
    if keep_replacements:
        df[label_col] = df[label_col].replace(keep_replacements)

    if drop_values:
        mask = df[label_col].isin(drop_values)
        n_affected = int(mask.sum())
        if n_affected:
            logger.info(
                "%s %d building(s) with label value(s) %s mapped to None in label_replacements.",
                "Dropped" if drop else "Masked (kept row, cleared label for)",
                n_affected, sorted(drop_values),
            )
        if drop:
            df = df[~mask].reset_index(drop=True)
        else:
            df.loc[mask, label_col] = np.nan

    return df


def drop_rare_labels(
    df: pd.DataFrame,
    label_col: str,
    min_fraction: float,
    name: str = "",
) -> pd.DataFrame:
    """Drop rows whose label class makes up less than ``min_fraction`` of ``df``."""
    if label_col not in df.columns or min_fraction <= 0 or df.empty:
        return df

    counts = df[label_col].value_counts()  # NaN excluded by default
    total = int(counts.sum())
    if total == 0:
        return df

    proportions = counts / total
    rare_classes = proportions[proportions < min_fraction].index.tolist()
    if not rare_classes:
        return df

    mask = df[label_col].isin(rare_classes)
    n_affected = int(mask.sum())
    where = f" in {name}" if name else ""
    rare_pcts = ", ".join(f"{c}={proportions[c] * 100:.2f}%" for c in rare_classes)
    logger.warning(
        "Dropped %d building(s)%s with rare label class(es) below %.1f%% of the "
        "%d labeled rows%s: %s.",
        n_affected, where, min_fraction * 100, total, where, rare_pcts,
    )
    return df[~mask].reset_index(drop=True)


def drop_rare_labels_across(
    dfs: dict[str, pd.DataFrame],
    label_col: str,
    min_fraction: float,
) -> dict[str, pd.DataFrame]:
    """``drop_rare_labels``, but for data that's already split into separate dataframes
    (e.g.
    """
    non_empty = [d for d in dfs.values() if label_col in d.columns]
    if not non_empty or min_fraction <= 0:
        return dfs

    combined_counts = pd.concat([d[label_col] for d in non_empty]).value_counts()
    total = int(combined_counts.sum())
    if total == 0:
        return dfs

    proportions = combined_counts / total
    rare_classes = proportions[proportions < min_fraction].index.tolist()
    if not rare_classes:
        return dfs

    rare_pcts = ", ".join(f"{c}={proportions[c] * 100:.2f}%" for c in rare_classes)
    result: dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        if label_col not in df.columns:
            result[name] = df
            continue
        mask = df[label_col].isin(rare_classes)
        n_affected = int(mask.sum())
        if n_affected:
            logger.warning(
                "Dropped %d building(s) from %s with rare label class(es) below %.1f%% "
                "of the combined %d labeled rows across %s: %s.",
                n_affected, name, min_fraction * 100, total, "+".join(dfs), rare_pcts,
            )
        result[name] = df[~mask].reset_index(drop=True)
    return result


def enforce_min_class_size_per_split(
    splits: dict[str, pd.DataFrame | None],
    label_col: str,
    min_per_split: int,
    id_col: str | None,
    random_state: int,
) -> dict[str, pd.DataFrame | None]:
    """Guarantee >= ``min_per_split`` rows of every surviving class in every active split.
    """
    if min_per_split <= 0:
        return splits

    active = {name: df for name, df in splits.items() if df is not None}
    if not active:
        return splits
    n_active = len(active)
    min_total = min_per_split * n_active

    all_labels = set(pd.concat([df[label_col] for df in active.values()]).dropna().unique())
    counts_per_label = {
        label: sum(int((df[label_col] == label).sum()) for df in active.values())
        for label in all_labels
    }

    dropped = {label: total for label, total in counts_per_label.items() if total < min_total}
    if dropped:
        for label, total in dropped.items():
            logger.warning(
                "Dropping class %r entirely: only %d building(s) across %s combined "
                "(%s), need >= %d (%d per split x %d active splits) to guarantee every split.",
                label, total, "+".join(active), ", ".join(
                    f"{name}={int((df[label_col] == label).sum())}" for name, df in active.items()
                ), min_total, min_per_split, n_active,
            )
        for name in active:
            active[name] = active[name][~active[name][label_col].isin(dropped)].reset_index(drop=True)

    priority = [n for n in ("test", "val", "train") if n in active]
    for label in all_labels - set(dropped):
        counts = {name: int((df[label_col] == label).sum()) for name, df in active.items()}
        if all(c >= min_per_split for c in counts.values()):
            continue

        pooled_parts = []
        for name in active:
            part = active[name][active[name][label_col] == label]
            pooled_parts.append(part)
            active[name] = active[name][active[name][label_col] != label]

        # Splits pooled from multiple raw district sources aren't
        # guaranteed to share a CRS (observed: a blocked San Jose split
        # mixing 'WGS 84 / UTM zone 15N' and 'CR05 / CRTM05'), pd.concat
        # on GeoDataFrames with mismatched CRS raises ValueError outright
        # rather than silently reprojecting so every part
        # is reprojected to the first one's CRS first. A no-op for plain
        # (non-geo) DataFrames, which have no `.crs` attribute.
        target_crs = getattr(pooled_parts[0], "crs", None) if pooled_parts else None
        if target_crs is not None:
            pooled_parts = [
                part.to_crs(target_crs) if getattr(part, "crs", None) not in (None, target_crs) else part
                for part in pooled_parts
            ]
        pooled = pd.concat(pooled_parts)

        if id_col and id_col in pooled.columns:
            units = list(pooled[id_col].drop_duplicates().sample(frac=1, random_state=random_state))
        else:
            units = list(pooled.sample(frac=1, random_state=random_state).index)

        assigned: dict[str, list] = {name: [] for name in active}
        i = 0
        for name in priority:
            for _ in range(min_per_split):
                if i >= len(units):
                    break
                assigned[name].append(units[i])
                i += 1
        leftover_target = "train" if "train" in active else priority[0]
        while i < len(units):
            assigned[leftover_target].append(units[i])
            i += 1

        for name in active:
            keys = assigned[name]
            if not keys:
                continue
            sub = pooled[pooled[id_col].isin(keys)] if id_col and id_col in pooled.columns else pooled.loc[keys]
            # `sub` was pooled (and reprojected) above to `target_crs`,
            # which need not match `active[name]`'s own CRS (e.g. `pooled`
            # took its CRS from a different city's district than `name`'s)
            #, same mismatched-CRS pd.concat issue as `pooled` itself,
            # one assignment step later.
            dest_crs = getattr(active[name], "crs", None)
            if dest_crs is not None and getattr(sub, "crs", None) not in (None, dest_crs):
                sub = sub.to_crs(dest_crs)
            active[name] = pd.concat([active[name], sub])

    result: dict[str, pd.DataFrame | None] = dict(splits)
    for name, df in active.items():
        result[name] = df.reset_index(drop=True)
    return result


def drop_unlabeled(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Drop rows with no label value at all."""
    if label_col not in df.columns:
        return df

    mask = df[label_col].isna()
    n_affected = int(mask.sum())
    if n_affected:
        logger.info(
            "Dropped %d building(s) with no %s value (never surveyed / not yet "
            "labeled) before building the train/test split.",
            n_affected, label_col,
        )
    return df[~mask].reset_index(drop=True)


def validate_and_normalize(
    df: pd.DataFrame,
    cfg: PreprocessingConfig,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Validate label/category columns and apply ordinal category ordering."""
    if cfg.label not in df.columns:
        raise DatasetError(
            f"Label column '{cfg.label}' not found in data columns: {list(df.columns)}"
        )

    df = df.copy()

    if cfg.label_values:
        actual_labels = set(df[cfg.label].dropna().unique())
        invalid_labels = actual_labels - set(cfg.label_values)
        if invalid_labels:
            raise DatasetError(
                f"Label column '{cfg.label}' contains values not in the configured "
                f"label_values {cfg.label_values}: {sorted(invalid_labels)}"
            )

    for col, valid_values in cfg.categorical_feature_mappings.items():
        if col not in df.columns:
            continue
        actual_values = set(df[col].dropna().unique())
        invalid_values = actual_values - set(valid_values)
        if invalid_values:
            raise DatasetError(
                f"Column '{col}' contains values not in its configured "
                f"categorical_feature_mappings {valid_values}: {sorted(invalid_values)}"
            )

    for col, (lo, hi) in cfg.numeric_feature_ranges.items():
        if col not in df.columns:
            continue
        non_null_mask = df[col].notna()
        fractional_mask = non_null_mask & (df[col] % 1 != 0)
        n_rounded = int(fractional_mask.sum())
        if n_rounded:
            logger.info(
                "Column '%s': rounded %d non-integer value(s) to the nearest "
                "whole number (only integers are allowed).",
                col, n_rounded,
            )
            df.loc[fractional_mask, col] = df.loc[fractional_mask, col].round()

        values = df[col].dropna()
        out_of_range = values[(values < lo) | (values > hi)]
        if not out_of_range.empty:
            raise DatasetError(
                f"Column '{col}' must be between {lo} and {hi}, found "
                f"out-of-range value(s): {sorted(out_of_range.unique().tolist())[:10]}"
            )

    ordinal_feature_mappings = {k: list(v) for k, v in cfg.ordinal_feature_mappings.items()}
    for col, defined_order in ordinal_feature_mappings.items():
        if col not in df.columns:
            continue

        actual_values = set(df[col].dropna().unique())
        defined_set = set(defined_order)

        if actual_values != defined_set:
            missing_from_data = defined_set - actual_values
            if missing_from_data:
                logger.warning(
                    "Column '%s': categories defined but not in data: %s (trimming mapping).",
                    col, sorted(missing_from_data),
                )
                defined_order = [v for v in defined_order if v in actual_values]
                ordinal_feature_mappings[col] = defined_order

            missing_from_definition = actual_values - defined_set
            if missing_from_definition:
                raise DatasetError(
                    f"Column '{col}' contains categories not in its configured "
                    f"ordinal_feature_mappings: {sorted(missing_from_definition)}"
                )

        df[col] = pd.Categorical(df[col], categories=defined_order, ordered=True)

    return df, ordinal_feature_mappings
