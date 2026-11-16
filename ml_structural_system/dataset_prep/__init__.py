"""Raw-data preparation: split and preprocess labeled building datasets."""

from __future__ import annotations

from .code_quality import (
    available_code_types,
    apply_code_quality_mapping,
    code_quality_to_year_series,
    year_series_to_code_quality,
)
from .config import (
    DEFAULT_MIN_CLASS_FRACTION,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    PreprocessingConfig,
    load_preprocessing_config,
)
from .encoding import (
    FeatureTypes,
    PreprocessResult,
    build_preprocessor,
    get_feature_names,
    identify_feature_types,
    load_preprocessor,
    preprocess_split,
    save_preprocess_result,
)
from .io import (
    clean_geometries,
    ensure_feature_columns,
    find_tabular_file,
    load_tabular_source,
    save_tabular,
)
from .splitting import RawSplitResult, prepare_raw_splits, save_raw_split, stratified_split
from .stats import (
    GroupDistributionCheck,
    LabelDistribution,
    apply_label_replacements,
    drop_rare_labels,
    drop_rare_labels_across,
    drop_unlabeled,
    validate_and_normalize,
)

__all__ = [
    # code_quality
    "available_code_types",
    "apply_code_quality_mapping",
    "code_quality_to_year_series",
    "year_series_to_code_quality",
    # config
    "DEFAULT_SEED",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_MIN_CLASS_FRACTION",
    "PreprocessingConfig",
    "load_preprocessing_config",
    # io
    "clean_geometries",
    "ensure_feature_columns",
    "find_tabular_file",
    "load_tabular_source",
    "save_tabular",
    # stats
    "GroupDistributionCheck",
    "LabelDistribution",
    "apply_label_replacements",
    "drop_rare_labels",
    "drop_rare_labels_across",
    "drop_unlabeled",
    "validate_and_normalize",
    # splitting
    "RawSplitResult",
    "prepare_raw_splits",
    "save_raw_split",
    "stratified_split",
    # encoding
    "FeatureTypes",
    "PreprocessResult",
    "build_preprocessor",
    "get_feature_names",
    "identify_feature_types",
    "load_preprocessor",
    "preprocess_split",
    "save_preprocess_result",
]
