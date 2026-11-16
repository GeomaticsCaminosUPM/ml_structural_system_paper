"""Fitting a preprocessor (encode/scale) on a train/test split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

from ..exceptions import DatasetError
from .config import PreprocessingConfig


@dataclass
class FeatureTypes:
    """Columns grouped by how they should be encoded."""

    numeric: list[str]
    ordinal: list[str]
    categorical: list[str]


def identify_feature_types(
    df: pd.DataFrame,
    cfg: PreprocessingConfig,
    ordinal_feature_mappings: dict[str, list[str]] | None = None,
) -> FeatureTypes:
    """Categorize columns into numeric/ordinal/categorical."""
    ordinal_mappings = (
        ordinal_feature_mappings if ordinal_feature_mappings is not None
        else cfg.ordinal_feature_mappings
    )
    exclude_cols = {cfg.label, *cfg.ignore_features}
    all_cols = [col for col in df.columns if col not in exclude_cols]

    ordinal = [col for col in ordinal_mappings if col in all_cols]
    numeric = [
        col for col in all_cols
        if col not in ordinal
        and df[col].dtype in ["int64", "int32", "float64", "float32", "bool", "boolean"]
    ]
    categorical = [
        col for col in all_cols
        if col not in ordinal and col not in numeric
        and df[col].dtype in ["object", "category"]
    ]
    return FeatureTypes(numeric=numeric, ordinal=ordinal, categorical=categorical)


def build_preprocessor(
    df: pd.DataFrame,
    cfg: PreprocessingConfig,
    ordinal_feature_mappings: dict[str, list[str]] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer | Pipeline:
    """Build a ``ColumnTransformer`` for numeric/ordinal/categorical features."""
    ordinal_mappings = (
        ordinal_feature_mappings if ordinal_feature_mappings is not None
        else cfg.ordinal_feature_mappings
    )
    feature_types = identify_feature_types(df, cfg, ordinal_mappings)
    transformers = []

    if feature_types.numeric:
        num_transformer = (
            PowerTransformer(method="yeo-johnson", standardize=False)
            if scale_numeric else "passthrough"
        )
        transformers.append(("num", num_transformer, feature_types.numeric))

    if feature_types.ordinal:
        ordinal_categories = [ordinal_mappings[col] for col in feature_types.ordinal]
        ord_transformer = OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        transformers.append(("ord", ord_transformer, feature_types.ordinal))

    if feature_types.categorical:
        cat_transformer = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        transformers.append(("cat", cat_transformer, feature_types.categorical))

    col_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    if scale_numeric:
        return Pipeline([("col_transform", col_transformer), ("final_scale", StandardScaler())])
    return col_transformer


def get_feature_names(preprocessor: Any, df: pd.DataFrame) -> list[str]:
    """List the feature names produced by a fitted preprocessor."""
    col_transform = (
        preprocessor.named_steps["col_transform"]
        if hasattr(preprocessor, "named_steps") and "col_transform" in preprocessor.named_steps
        else preprocessor
    )
    feature_names: list[str] = []
    for name, transformer, columns in col_transform.transformers_:
        if name in ("num", "ord"):
            feature_names.extend(columns)
        elif name == "cat":
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
    return feature_names


@dataclass
class PreprocessResult:
    """Result of fitting a preprocessor on a train split and applying it to test."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer | Pipeline
    label_encoder: LabelEncoder
    feature_names: list[str]
    feature_types: FeatureTypes
    label_mapping: dict[str, int]
    metadata: dict[str, Any]
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None

    @property
    def train_df(self) -> pd.DataFrame:
        """``X_train`` + label as a single DataFrame, ready to write to CSV."""
        out = pd.DataFrame(self.X_train, columns=self.feature_names)
        out[self.metadata["label"]] = self.y_train
        return out

    @property
    def test_df(self) -> pd.DataFrame:
        """``X_test`` + label as a single DataFrame, ready to write to CSV."""
        out = pd.DataFrame(self.X_test, columns=self.feature_names)
        out[self.metadata["label"]] = self.y_test
        return out

    @property
    def val_df(self) -> pd.DataFrame | None:
        """``X_val`` + label as a single DataFrame, or ``None`` if no val split was given.
        """
        if self.X_val is None:
            return None
        out = pd.DataFrame(self.X_val, columns=self.feature_names)
        out[self.metadata["label"]] = self.y_val
        return out

    def summary_text(self) -> str:
        """Render a human-readable summary of the preprocessing result."""
        ft = self.feature_types
        shapes = f"Train shape: {self.X_train.shape}"
        if self.X_val is not None:
            shapes += f"   Val shape: {self.X_val.shape}"
        shapes += f"   Test shape: {self.X_test.shape}"
        return "\n".join([
            f"Feature types: {len(ft.numeric)} numeric, {len(ft.ordinal)} ordinal, "
            f"{len(ft.categorical)} categorical",
            f"Encoded feature count: {len(self.feature_names)}",
            f"Label mapping: {self.label_mapping}",
            shapes,
        ])


def preprocess_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PreprocessingConfig,
    ordinal_feature_mappings: dict[str, list[str]] | None = None,
    scale_numeric: bool = True,
    val_df: pd.DataFrame | None = None,
) -> PreprocessResult:
    """Fit a preprocessor + label encoder on ``train_df``, apply both to
    ``test_df``/``val_df``.
    """
    ordinal_mappings = (
        ordinal_feature_mappings if ordinal_feature_mappings is not None
        else cfg.ordinal_feature_mappings
    )
    named_splits = [("train_df", train_df), ("test_df", test_df)]
    if val_df is not None:
        named_splits.append(("val_df", val_df))
    for name, df in named_splits:
        if cfg.label not in df.columns:
            raise DatasetError(f"Label column '{cfg.label}' not found in {name}.")

    train_df = train_df.copy()
    test_df = test_df.copy()
    val_df = val_df.copy() if val_df is not None else None
    if cfg.feature_columns:
        keep = [*cfg.feature_columns, cfg.label]
        for name, df in named_splits:
            missing = [c for c in cfg.feature_columns if c not in df.columns]
            if missing:
                raise DatasetError(
                    f"Configured feature_columns {missing} are missing from {name}. "
                    "footprint_attributes only auto-computes missing columns for .gpkg "
                    "sources, check that this data went through load_tabular_source()."
                )
        train_df = train_df[keep]
        test_df = test_df[keep]
        if val_df is not None:
            val_df = val_df[keep]

    drop_cols = [c for c in cfg.ignore_features if c in train_df.columns]
    train_features = train_df.drop(columns=drop_cols, errors="ignore")
    test_features = test_df.drop(columns=drop_cols, errors="ignore")
    val_features = val_df.drop(columns=drop_cols, errors="ignore") if val_df is not None else None

    preprocessor = build_preprocessor(train_features, cfg, ordinal_mappings, scale_numeric)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_features[cfg.label])
    label_mapping = {str(c): int(i) for i, c in enumerate(label_encoder.classes_)}

    X_train_df = train_features.drop(columns=[cfg.label])
    X_test_df = test_features.drop(columns=[cfg.label])
    y_train = label_encoder.transform(train_features[cfg.label])
    y_test = label_encoder.transform(test_features[cfg.label])

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    X_val = y_val = None
    if val_features is not None:
        X_val_df = val_features.drop(columns=[cfg.label])
        y_val = label_encoder.transform(val_features[cfg.label])
        X_val = preprocessor.transform(X_val_df)

    feature_names = get_feature_names(preprocessor, X_train_df)
    feature_types = identify_feature_types(train_features, cfg, ordinal_mappings)

    metadata = {
        "ignore_features": cfg.ignore_features,
        "ordinal_feature_mappings": ordinal_mappings,
        "categorical_feature_mappings": cfg.categorical_feature_mappings,
        "label": cfg.label,
        "label_values": cfg.label_values,
        "group": cfg.group,
        "label_replacements": cfg.label_replacements,
        "feature_columns": cfg.feature_columns,
        "code_type": cfg.code_type,
        "compute_missing_features": cfg.compute_missing_features,
        "feature_names": feature_names,
        # {class_name: code}, straight from the fitted LabelEncoder, the
        # one place the real class names and their integer codes are both
        # still known. Once train_preprocessed.csv is written, the label
        # column is just integers; without this saved here, that mapping is
        # unrecoverable (see ExperimentConfig.label_map).
        "label_mapping": label_mapping,
    }

    return PreprocessResult(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        preprocessor=preprocessor, label_encoder=label_encoder,
        feature_names=feature_names, feature_types=feature_types,
        label_mapping=label_mapping, metadata=metadata,
        X_val=X_val, y_val=y_val,
    )


def save_preprocess_result(result: PreprocessResult, output_dir: str | Path) -> dict[str, Path]:
    """Write a preprocessing result's artifacts to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_preprocessed.csv"
    test_path = output_dir / "test_preprocessed.csv"
    result.train_df.to_csv(train_path, index=False)
    result.test_df.to_csv(test_path, index=False)

    preprocessor_path = output_dir / "preprocessor.pkl"
    label_encoder_path = output_dir / "label_encoder.pkl"
    metadata_path = output_dir / "preprocessing_metadata.pkl"
    joblib.dump(result.preprocessor, preprocessor_path)
    joblib.dump(result.label_encoder, label_encoder_path)
    joblib.dump(result.metadata, metadata_path)

    json_path = output_dir / "preprocessing.json"
    with open(json_path, "w") as f:
        json.dump(result.metadata, f, indent=2)

    paths = {
        "train_csv": train_path,
        "test_csv": test_path,
        "preprocessor": preprocessor_path,
        "label_encoder": label_encoder_path,
        "metadata_pkl": metadata_path,
        "metadata_json": json_path,
    }
    if result.val_df is not None:
        val_path = output_dir / "val_preprocessed.csv"
        result.val_df.to_csv(val_path, index=False)
        paths["val_csv"] = val_path
    return paths


def load_preprocessor(
    preprocessed_dir: str | Path,
) -> tuple[Any, LabelEncoder, dict[str, Any] | None]:
    """Load a previously fitted preprocessor and label encoder from disk."""
    preprocessed_dir = Path(preprocessed_dir)
    pp_path = preprocessed_dir / "preprocessor.pkl"
    le_path = preprocessed_dir / "label_encoder.pkl"

    if not pp_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {pp_path}. Re-run 'mlss preprocess' to generate it."
        )
    if not le_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found at {le_path}. Re-run 'mlss preprocess' to generate it."
        )

    preprocessor = joblib.load(pp_path)
    label_encoder = joblib.load(le_path)

    meta_path = preprocessed_dir / "preprocessing_metadata.pkl"
    metadata = joblib.load(meta_path) if meta_path.exists() else None

    return preprocessor, label_encoder, metadata
