"""Guatemala City attribute ablation (no roof material, code quality or
construction era) on the same split as the base Guatemala City model
(Table tab:basecities).

The split depends on the feature columns, so a fresh split of the ablation
pool would not reproduce the base test set; guatemala/splits/ is copied
verbatim and only preprocessing, training, testing and the 5-model
assessment are run here.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ml_structural_system import dataset_prep as prep_module
from ml_structural_system import experiment_report
from ml_structural_system.config import load_config

GROUP_DIR = Path(__file__).resolve().parent.parent / "experiments"
BASE = GROUP_DIR / "guatemala"
EXP = GROUP_DIR / "guatemala_no_roof_code_year"
MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "HistGradientBoosting"]


def preprocess(config_path: Path, exp_dir: Path) -> None:
    prep_cfg = prep_module.load_preprocessing_config(config_path)
    splits = exp_dir / "splits"

    def load(name: str):
        df = prep_module.load_tabular_source(
            str(prep_module.find_tabular_file(splits, name)), prep_cfg.feature_columns, prep_cfg.code_type,
            compute_missing_features=prep_cfg.compute_missing_features,
        )
        return prep_module.apply_label_replacements(df, prep_cfg.label, prep_cfg.label_replacements)

    train_df, ord_map = prep_module.validate_and_normalize(load("train_df"), prep_cfg)
    test_df, _ = prep_module.validate_and_normalize(load("test_df"), prep_cfg)
    val_df, _ = prep_module.validate_and_normalize(load("val_df"), prep_cfg)
    result = prep_module.preprocess_split(train_df, test_df, prep_cfg, ord_map, val_df=val_df)
    prep_module.save_preprocess_result(result, exp_dir / "preprocessed_splits")


def main() -> None:
    if not (EXP / "splits").exists():
        shutil.copytree(BASE / "splits", EXP / "splits")
    config_path = EXP / "config.yaml"
    preprocess(config_path, EXP)
    experiment_report.run(load_config(config_path), EXP / "report")
    subprocess.run(
        ["mlss", "assess", "--config", "config.yaml", "--data", "preprocessed_splits/test_preprocessed.csv",
         "--models", *MODELS, "--output", "report/per_sample_5models.csv",
         "--gen-output", "report/generalizability_5models.csv", "--no-mlflow"],
        cwd=EXP, check=True,
    )


if __name__ == "__main__":
    main()
