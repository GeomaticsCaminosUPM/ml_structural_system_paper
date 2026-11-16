"""Typed config for the raw-data pipeline: config.yaml's `preprocessing:` section."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config_merge import load_yaml_with_extends
from ..exceptions import ConfigError
from . import code_quality

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MIN_CLASS_FRACTION = 0.04


@dataclass
class PreprocessingConfig:
    """Typed view of ``config.yaml``'s ``preprocessing:`` section."""

    label: str
    seed: int = DEFAULT_SEED
    label_values: list[str] = field(default_factory=list)
    label_replacements: dict[str, str] = field(default_factory=dict)
    ignore_features: list[str] = field(default_factory=list)
    ordinal_feature_mappings: dict[str, list[str]] = field(default_factory=dict)
    categorical_feature_mappings: dict[str, list[str]] = field(default_factory=dict)
    # Numeric feature_columns validated to be a whole number within
    # [min, max] (inclusive); a value outside the range or with a fractional
    # part raises DatasetError, see stats.validate_and_normalize. Missing
    # (NaN) values are unaffected here (handled separately by
    # io._drop_rows_with_missing_features).
    numeric_feature_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    group: list[str] = field(default_factory=list)
    id_col: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    source: Any = None
    train_source: Any = None
    test_source: Any = None
    test_size: float = DEFAULT_TEST_SIZE
    # Fraction of rows set aside as a val split, in addition to test. None
    # (the default) keeps the historical train/test-only behavior. When set,
    # splitting always happens per source-dataset (see io.load_tabular_source's
    # "dataset" tag) so every dataset contributes proportionally to
    # train/val/test rather than risking a whole dataset landing on one side.
    val_size: float | None = None
    code_type: Any = None
    # A label class making up less than this fraction of the labeled data is
    # dropped and a warning logged, checked once, before the train/test
    # split (single-`source` mode: on the full pool; `train_source`/
    # `test_source` mode: on their combined proportions), so train and test
    # always agree on which classes survive. See
    # dataset_prep.stats.drop_rare_labels/drop_rare_labels_across. 0 (or a
    # negative value) disables the check entirely.
    min_class_fraction: float = DEFAULT_MIN_CLASS_FRACTION
    # Minimum rows of each surviving class required in *every* active split
    # (train/val/test, whichever are configured), enforced after the split
    # already exists, see dataset_prep.stats.enforce_min_class_size_per_split.
    # A class without enough rows across all active splits combined
    # (min_class_size_per_split * number of active splits) is dropped
    # entirely (warning logged), even if it survived min_class_fraction;
    # otherwise rows are redistributed between splits to satisfy the
    # minimum. 0 (the default) disables this check entirely.
    min_class_size_per_split: int = 0
    # Per-group (e.g. per-city) row cap applied to the split output, keyed by
    # split name ("train"/"val"/"test"); a split missing from this dict is
    # left uncapped. Used to build a small, fast dataset for hyperparameter
    # search (see dataset_prep.splitting.cap_rows_per_group). Empty (the
    # default) disables capping entirely.
    max_per_group: dict[str, int] = field(default_factory=dict)
    # Minimum rows of each label guaranteed within a capped group, when that
    # many exist to begin with (see cap_rows_per_group). Only relevant when
    # max_per_group is set.
    min_per_label: int = 5
    # Compute a missing geometry-derived feature column with footprint_attributes.
    # If False, a missing column raises instead.
    compute_missing_features: bool = True
    config_dir: Path = field(default_factory=Path.cwd)

    @property
    def group_col(self) -> str | None:
        """First configured group column, used for stratified splitting."""
        return self.group[0] if self.group else None

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path relative to this config's directory unless absolute."""
        p = Path(path)
        return p if p.is_absolute() else (self.config_dir / p).resolve()

    def resolve_source(self, source: Any) -> str | list[str]:
        """Resolve one or more source paths relative to this config's directory."""
        if isinstance(source, (list, tuple)):
            return [str(self.resolve(p)) for p in source]
        return str(self.resolve(source))


def _resolve_seed(seed: Any) -> int:
    """Coerce a config value into a valid seed."""
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    return DEFAULT_SEED


def load_preprocessing_config(config_path: str | Path | None = None) -> PreprocessingConfig:
    """Load and validate the ``preprocessing:`` section of ``config.yaml``."""
    config_path = Path(config_path) if config_path is not None else Path("config.yaml")
    config_path = config_path.resolve()
    raw = load_yaml_with_extends(config_path)

    if "preprocessing" not in raw:
        raise ConfigError(f"'{config_path}' is missing a top-level 'preprocessing' section.")
    section = raw["preprocessing"] or {}

    label = section.get("label")
    if not label:
        raise ConfigError(f"'{config_path}''s preprocessing section is missing required key 'label'.")

    data_cfg = section.get("data") or {}
    group_raw = section.get("group")
    if group_raw is None:
        group = []
    elif isinstance(group_raw, str):
        group = [group_raw] if group_raw else []
    else:
        group = list(group_raw)

    cfg = PreprocessingConfig(
        label=label,
        seed=_resolve_seed(section.get("seed")),
        label_values=list(section.get("label_values") or []),
        label_replacements=dict(section.get("label_replacements") or {}),
        ignore_features=list(section.get("ignore_features") or []),
        ordinal_feature_mappings={
            k: list(v) for k, v in (section.get("ordinal_feature_mappings") or {}).items()
        },
        categorical_feature_mappings={
            k: list(v) for k, v in (section.get("categorical_feature_mappings") or {}).items()
        },
        numeric_feature_ranges={
            k: (float(v[0]), float(v[1]))
            for k, v in (section.get("numeric_feature_ranges") or {}).items()
        },
        group=group,
        id_col=section.get("id_col") or None,
        feature_columns=list(section.get("feature_columns") or []),
        source=data_cfg.get("source"),
        train_source=data_cfg.get("train_source"),
        test_source=data_cfg.get("test_source"),
        test_size=float(data_cfg.get("test_size", DEFAULT_TEST_SIZE)),
        val_size=float(data_cfg["val_size"]) if data_cfg.get("val_size") is not None else None,
        code_type=data_cfg.get("code_type"),
        min_class_fraction=float(section.get("min_class_fraction", DEFAULT_MIN_CLASS_FRACTION)),
        min_class_size_per_split=int(section.get("min_class_size_per_split", 0)),
        max_per_group={k: int(v) for k, v in (data_cfg.get("max_per_group") or {}).items()},
        min_per_label=int(data_cfg.get("min_per_label", 5)),
        compute_missing_features=bool(section.get("compute_missing_features", True)),
        config_dir=config_path.parent,
    )
    _validate_and_resolve_code_type(cfg, config_path)
    return cfg


def _validate_and_resolve_code_type(cfg: PreprocessingConfig, config_path: Path) -> None:
    """Validate and resolve ``cfg.code_type`` in place."""
    known = code_quality.available_code_types()

    def check(value: Any) -> None:
        if value is not None and value not in known:
            raise ConfigError(
                f"'{config_path}''s preprocessing.data.code_type value '{value}' is not a "
                f"known code_type. Known types: {known}"
            )

    if isinstance(cfg.code_type, dict):
        for v in cfg.code_type.values():
            check(v)
        cfg.code_type = {str(cfg.resolve(k)): v for k, v in cfg.code_type.items()}
    else:
        check(cfg.code_type)
