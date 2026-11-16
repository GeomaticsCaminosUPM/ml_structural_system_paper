"""Raw split and preprocessing of one experiment directory (used by the learning curve).
"""

from __future__ import annotations

from pathlib import Path

from . import dataset_prep as prep_module


def prepare_data(config_path: Path, exp_dir: Path) -> None:
    """Write ``exp_dir/splits`` and ``exp_dir/preprocessed_splits`` from the raw data.
    """
    splits_dir = exp_dir / "splits"
    preproc_dir = exp_dir / "preprocessed_splits"

    prep_cfg = prep_module.load_preprocessing_config(config_path)
    raw_split = prep_module.prepare_raw_splits(prep_cfg)
    prep_module.save_raw_split(raw_split, splits_dir)

    train_path = prep_module.find_tabular_file(splits_dir, "train_df")
    test_path = prep_module.find_tabular_file(splits_dir, "test_df")
    train_df = prep_module.load_tabular_source(
        str(train_path), prep_cfg.feature_columns, prep_cfg.code_type,
        compute_missing_features=prep_cfg.compute_missing_features,
    )
    test_df = prep_module.load_tabular_source(
        str(test_path), prep_cfg.feature_columns, prep_cfg.code_type,
        compute_missing_features=prep_cfg.compute_missing_features,
    )
    train_df = prep_module.apply_label_replacements(train_df, prep_cfg.label, prep_cfg.label_replacements)
    test_df = prep_module.apply_label_replacements(test_df, prep_cfg.label, prep_cfg.label_replacements)
    train_df, ord_map = prep_module.validate_and_normalize(train_df, prep_cfg)
    test_df, _ = prep_module.validate_and_normalize(test_df, prep_cfg)

    val_df = None
    if raw_split.val_df is not None:
        val_path = prep_module.find_tabular_file(splits_dir, "val_df")
        val_df = prep_module.load_tabular_source(
            str(val_path), prep_cfg.feature_columns, prep_cfg.code_type,
            compute_missing_features=prep_cfg.compute_missing_features,
        )
        val_df = prep_module.apply_label_replacements(val_df, prep_cfg.label, prep_cfg.label_replacements)
        val_df, _ = prep_module.validate_and_normalize(val_df, prep_cfg)

    result = prep_module.preprocess_split(train_df, test_df, prep_cfg, ord_map, val_df=val_df)
    prep_module.save_preprocess_result(result, preproc_dir)
