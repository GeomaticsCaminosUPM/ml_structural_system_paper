"""Raw train/test splitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split

from ..exceptions import ConfigError
from .config import PreprocessingConfig
from .io import load_tabular_source, save_tabular
from .stats import (
    GroupDistributionCheck,
    LabelDistribution,
    apply_label_replacements,
    check_group_distributions,
    drop_rare_labels,
    drop_rare_labels_across,
    drop_unlabeled,
    enforce_min_class_size_per_split,
    group_checks_text,
    validate_and_normalize,
)

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    group_col: str | None,
    test_size: float,
    random_state: int,
    id_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stratified train-test split respecting group boundaries."""
    if id_col is not None and id_col in df.columns:
        representative = df.drop_duplicates(subset=[id_col], keep="first")
        rep_train, _rep_test, n_single = stratified_split(
            representative, label_col, group_col, test_size, random_state,
        )
        train_ids = set(rep_train[id_col])
        train_df = df[df[id_col].isin(train_ids)]
        test_df = df[~df[id_col].isin(train_ids)]
        return train_df, test_df, n_single

    df_copy = df.copy()
    strata_col = "_strata"
    df_copy[strata_col] = (
        df_copy[label_col].astype(str) + "_" + df_copy[group_col].astype(str)
        if group_col is not None
        else df_copy[label_col].astype(str)
    )

    strata_counts = df_copy[strata_col].value_counts()
    single_sample_strata = strata_counts[strata_counts < 2].index
    n_single = len(single_sample_strata)

    if n_single:
        logger.warning(
            "Found %d stratum/strata with only 1 sample; placing them in the training set.",
            n_single,
        )
        single_sample_df = df_copy[df_copy[strata_col].isin(single_sample_strata)]
        splittable_df = df_copy[~df_copy[strata_col].isin(single_sample_strata)]
    else:
        single_sample_df = pd.DataFrame()
        splittable_df = df_copy

    train_df, test_df = train_test_split(
        splittable_df,
        test_size=test_size,
        random_state=random_state,
        stratify=splittable_df[strata_col],
    )

    if not single_sample_df.empty:
        train_df = pd.concat([train_df, single_sample_df])

    return train_df.drop(columns=[strata_col]), test_df.drop(columns=[strata_col]), n_single


def stratified_split_by_dataset(
    df: pd.DataFrame,
    label_col: str,
    group_col: str | None,
    val_size: float | None,
    test_size: float,
    random_state: int,
    id_col: str | None = None,
    dataset_col: str = "dataset",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Three-way stratified train/val/test split, done independently per source dataset.
    """
    groups = (
        list(df.groupby(dataset_col, sort=False))
        if dataset_col in df.columns
        else [("_all", df)]
    )

    train_parts, val_parts, test_parts = [], [], []
    n_single_total = 0
    for _name, sub in groups:
        if test_size > 0:
            remainder, test_sub, n_single = stratified_split(
                sub, label_col, group_col, test_size, random_state, id_col,
            )
        else:
            remainder, test_sub, n_single = sub, sub.iloc[0:0], 0
        n_single_total += n_single

        if val_size:
            remainder_frac = 1 - test_size
            val_frac_of_remainder = val_size / remainder_frac if remainder_frac > 0 else 0
            train_sub, val_sub, n_single_v = stratified_split(
                remainder, label_col, group_col, val_frac_of_remainder, random_state, id_col,
            )
            n_single_total += n_single_v
        else:
            train_sub, val_sub = remainder, remainder.iloc[0:0]

        train_parts.append(train_sub)
        val_parts.append(val_sub)
        test_parts.append(test_sub)

    empty = df.iloc[0:0]
    train_df = pd.concat(train_parts) if train_parts else empty
    val_df = pd.concat(val_parts) if val_parts else empty
    test_df = pd.concat(test_parts) if test_parts else empty
    return train_df, val_df, test_df, n_single_total


def cap_rows_per_group(
    df: pd.DataFrame,
    label_col: str,
    group_col: str | None,
    cap: int,
    min_per_label: int,
    random_state: int,
) -> pd.DataFrame:
    """Downsample ``df`` to at most ``cap`` rows per ``group_col`` value."""
    groups = (
        list(df.groupby(group_col, sort=False))
        if group_col is not None and group_col in df.columns
        else [(None, df)]
    )

    parts = []
    for name, sub in groups:
        if len(sub) <= cap:
            parts.append(sub)
            continue

        guaranteed_parts, remainder_parts = [], []
        for _lbl, g in sub.groupby(label_col, sort=False):
            take = min(min_per_label, len(g))
            chosen = g.sample(n=take, random_state=random_state)
            guaranteed_parts.append(chosen)
            remainder_parts.append(g.drop(chosen.index))

        guaranteed = pd.concat(guaranteed_parts) if guaranteed_parts else sub.iloc[0:0]
        remainder = pd.concat(remainder_parts) if remainder_parts else sub.iloc[0:0]

        n_more = cap - len(guaranteed)
        if n_more > 0 and len(remainder) > 0:
            extra = remainder.sample(n=min(n_more, len(remainder)), random_state=random_state)
            group_result = pd.concat([guaranteed, extra])
        else:
            if n_more < 0:
                logger.warning(
                    "Group %r needs >= %d rows to guarantee %d per label; cap is %d. "
                    "Keeping all %d guaranteed rows (cap exceeded to preserve the "
                    "per-label minimum).",
                    name, len(guaranteed), min_per_label, cap, len(guaranteed),
                )
            group_result = guaranteed

        parts.append(group_result)

    return pd.concat(parts).reset_index(drop=True) if parts else df


def _apply_max_per_group(
    cfg: PreprocessingConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Apply ``cfg.max_per_group`` capping to train/val/test, if configured."""
    if not cfg.max_per_group:
        return train_df, val_df, test_df

    if "train" in cfg.max_per_group:
        train_df = cap_rows_per_group(
            train_df, cfg.label, cfg.group_col,
            cfg.max_per_group["train"], cfg.min_per_label, cfg.seed,
        )
    if val_df is not None and "val" in cfg.max_per_group:
        val_df = cap_rows_per_group(
            val_df, cfg.label, cfg.group_col,
            cfg.max_per_group["val"], cfg.min_per_label, cfg.seed,
        )
    if "test" in cfg.max_per_group:
        test_df = cap_rows_per_group(
            test_df, cfg.label, cfg.group_col,
            cfg.max_per_group["test"], cfg.min_per_label, cfg.seed,
        )
    return train_df, val_df, test_df


def _apply_min_class_size_per_split(
    cfg: PreprocessingConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    include_test: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Apply ``cfg.min_class_size_per_split`` enforcement to train/val/test, if configured.
    """
    if cfg.min_class_size_per_split <= 0:
        return train_df, val_df, test_df

    splits = {"train": train_df, "val": val_df}
    if include_test:
        splits["test"] = test_df
    result = enforce_min_class_size_per_split(
        splits, cfg.label, cfg.min_class_size_per_split, cfg.id_col, cfg.seed,
    )
    if include_test:
        return result["train"], result["val"], result["test"]

    # test_df was deliberately excluded from redistribution above (see
    # include_test's docstring), but a class dropped entirely from
    # train/val for lack of support must also be dropped from test: the
    # label encoder is fit on train_df's surviving classes only
    # (dataset_prep.encoding.preprocess_split), and raises the moment it
    # meets a class it never saw during training. This is a one-way drop of
    # those specific test rows, not a redistribution, no test data moves
    # into train/val, it just stops being scored on a class the model was
    # never given a chance to learn (same rationale as drop_rare_labels
    # elsewhere in this module).
    surviving_parts = [result["train"][cfg.label]]
    if result["val"] is not None:
        surviving_parts.append(result["val"][cfg.label])
    surviving = set(pd.concat(surviving_parts).dropna().unique())
    dropped = set(test_df[cfg.label].dropna().unique()) - surviving
    if dropped:
        drop_mask = test_df[cfg.label].isin(dropped)
        logger.warning(
            "Dropping %d test row(s) with class(es) %s: absent from train/val "
            "after min_class_size_per_split enforcement, so the fitted label "
            "encoder (train-only) has no encoding for them.",
            int(drop_mask.sum()), sorted(dropped),
        )
        test_df = test_df[~drop_mask].reset_index(drop=True)
    return result["train"], result["val"], test_df


@dataclass
class RawSplitResult:
    """Result of preparing raw train/(val/)test splits (before preprocessing)."""

    train_df: gpd.GeoDataFrame | pd.DataFrame
    test_df: gpd.GeoDataFrame | pd.DataFrame
    full_df: gpd.GeoDataFrame | pd.DataFrame
    mode: Literal["auto-split", "pre-split"]
    full_label_distribution: LabelDistribution
    train_label_distribution: LabelDistribution
    test_label_distribution: LabelDistribution
    group_distribution_checks: list[GroupDistributionCheck]
    n_single_sample_strata: int
    ordinal_feature_mappings: dict[str, list[str]]
    val_df: gpd.GeoDataFrame | pd.DataFrame | None = None
    val_label_distribution: LabelDistribution | None = None

    def summary_text(self) -> str:
        """Render a human-readable summary of the split."""
        parts = [
            f"Mode: {self.mode}",
            self.full_label_distribution.as_text("\nFull dataset label distribution"),
            self.train_label_distribution.as_text("\nTrain label distribution"),
        ]
        if self.val_df is not None and self.val_label_distribution is not None:
            parts.append(self.val_label_distribution.as_text("\nVal label distribution"))
        parts.append(self.test_label_distribution.as_text("\nTest label distribution"))
        if self.n_single_sample_strata:
            parts.append(
                f"\n{self.n_single_sample_strata} stratum/strata had only 1 sample "
                "(placed in train)."
            )
        checks_text = group_checks_text(self.group_distribution_checks)
        if checks_text:
            parts.append("\n" + checks_text)
        row_counts = f"\nTrain: {len(self.train_df):,} rows"
        if self.val_df is not None:
            row_counts += f"   Val: {len(self.val_df):,} rows"
        row_counts += f"   Test: {len(self.test_df):,} rows"
        parts.append(row_counts)
        return "\n".join(parts)


def prepare_raw_splits(
    cfg: PreprocessingConfig,
    test_size: float | None = None,
) -> RawSplitResult:
    """Load raw data per ``cfg`` and produce a train/test split."""
    val_df: pd.DataFrame | None = None

    if cfg.train_source is not None and cfg.test_source is not None:
        train_df = load_tabular_source(
            cfg.resolve_source(cfg.train_source), cfg.feature_columns, cfg.code_type,
            label_col=cfg.label, compute_missing_features=cfg.compute_missing_features,
        )
        test_df = load_tabular_source(
            cfg.resolve_source(cfg.test_source), cfg.feature_columns, cfg.code_type,
            label_col=cfg.label, compute_missing_features=cfg.compute_missing_features,
        )
        train_df = apply_label_replacements(train_df, cfg.label, cfg.label_replacements)
        test_df = apply_label_replacements(test_df, cfg.label, cfg.label_replacements)
        train_df = drop_unlabeled(train_df, cfg.label)
        test_df = drop_unlabeled(test_df, cfg.label)

        # Validated independently (each side may not carry every category),
        # but the train-derived mapping is the one that matters downstream:
        # preprocess_split() fits the encoder on train_df.
        train_df, ord_map = validate_and_normalize(train_df, cfg)
        test_df, _ = validate_and_normalize(test_df, cfg)

        # Judged from the combined train+test(+val, since val comes out of
        # train below) proportions, there's no single pool to filter
        # before an internal split here, see drop_rare_labels_across,
        # then dropped from both, so a class can never end up
        # rare-but-kept on one side and dropped from the other.
        filtered = drop_rare_labels_across(
            {"train": train_df, "test": test_df}, cfg.label, cfg.min_class_fraction,
        )
        train_df, test_df = filtered["train"], filtered["test"]

        n_single = 0
        if cfg.val_size:
            # Val is carved out of the train_source pool only, per dataset,
            # with no test slice taken from it here (test_size=0), the
            # held-out side of a leave-one-out experiment is entirely test.
            train_df, val_df, _empty_test, n_single = stratified_split_by_dataset(
                train_df, cfg.label, cfg.group_col, cfg.val_size, 0.0, cfg.seed, cfg.id_col,
            )

        train_df, val_df, test_df = _apply_max_per_group(cfg, train_df, val_df, test_df)
        # include_test=False: test_df here is an independently named,
        # held-out source (test_source), not part of the same pool as
        # train/val, see _apply_min_class_size_per_split's docstring for
        # why pooling it in would silently leak test rows into training.
        train_df, val_df, test_df = _apply_min_class_size_per_split(
            cfg, train_df, val_df, test_df, include_test=False,
        )

        # full_df is only used for label/group distribution stats below, never
        # written to disk, geometry is dropped before concatenating so a
        # train_source/test_source CRS mismatch (independently reprojected by
        # two separate load_tabular_source() calls, e.g. leave-one-out pooling
        # cities recorded in different national CRSs) can't break this.
        _parts = [train_df, val_df, test_df] if val_df is not None else [train_df, test_df]
        full_df = pd.concat(
            [pd.DataFrame(p).drop(columns=["geometry"], errors="ignore") for p in _parts],
            ignore_index=True,
        )
        mode: Literal["auto-split", "pre-split"] = "pre-split"
    elif cfg.source is not None:
        gdf = load_tabular_source(
            cfg.resolve_source(cfg.source), cfg.feature_columns, cfg.code_type,
            label_col=cfg.label, compute_missing_features=cfg.compute_missing_features,
        )
        gdf = apply_label_replacements(gdf, cfg.label, cfg.label_replacements)
        gdf = drop_unlabeled(gdf, cfg.label)
        gdf, ord_map = validate_and_normalize(gdf, cfg)

        # Checked on the full pool, *before* splitting, see
        # drop_rare_labels's docstring for why this must happen here and not
        # on train_df/test_df afterwards: it's what guarantees train, val,
        # and test always end up with the exact same surviving class set.
        gdf = drop_rare_labels(gdf, cfg.label, cfg.min_class_fraction)

        effective_test_size = test_size if test_size is not None else cfg.test_size
        if cfg.val_size:
            train_df, val_df, test_df, n_single = stratified_split_by_dataset(
                gdf, cfg.label, cfg.group_col, cfg.val_size, effective_test_size, cfg.seed, cfg.id_col,
            )
        else:
            train_df, test_df, n_single = stratified_split(
                gdf, cfg.label, cfg.group_col, effective_test_size, cfg.seed, cfg.id_col,
            )
        train_df, val_df, test_df = _apply_max_per_group(cfg, train_df, val_df, test_df)
        train_df, val_df, test_df = _apply_min_class_size_per_split(cfg, train_df, val_df, test_df)
        full_df = gdf
        mode = "auto-split"
    else:
        raise ConfigError(
            "preprocessing.data must set either 'source' (+ optional 'test_size') "
            "or both 'train_source' and 'test_source'."
        )

    group_checks = check_group_distributions(full_df, train_df, test_df, cfg.group)

    return RawSplitResult(
        train_df=train_df,
        test_df=test_df,
        full_df=full_df,
        mode=mode,
        full_label_distribution=LabelDistribution.from_series(full_df[cfg.label]),
        train_label_distribution=LabelDistribution.from_series(train_df[cfg.label]),
        test_label_distribution=LabelDistribution.from_series(test_df[cfg.label]),
        group_distribution_checks=group_checks,
        n_single_sample_strata=n_single,
        ordinal_feature_mappings=ord_map,
        val_df=val_df,
        val_label_distribution=(
            LabelDistribution.from_series(val_df[cfg.label]) if val_df is not None else None
        ),
    )


def save_raw_split(split: RawSplitResult, output_dir: str | Path) -> dict[str, Path]:
    """Write a raw split's train/(val/)test dataframes to disk."""
    output_dir = Path(output_dir)
    train_path = save_tabular(split.train_df, output_dir / "train_df")
    test_path = save_tabular(split.test_df, output_dir / "test_df")
    paths = {"train": train_path, "test": test_path}
    if split.val_df is not None:
        paths["val"] = save_tabular(split.val_df, output_dir / "val_df")
    return paths
