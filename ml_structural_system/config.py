"""Experiment configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config_merge import load_yaml_with_extends
from .exceptions import ConfigError


@dataclass
class CostMatrixConfig:
    """Asymmetric misclassification cost matrix derived from a fragility ranking."""

    fragility_rank: dict[str, int]   # original_class_name -> rank (0 = safest)
    pessimism_factor: float = 3.0
    label_encoding: dict[str, int] | None = None  # original_name -> integer_code
    label_replacements: dict[str, str | None] | None = None  # raw_name -> final_name (None = dropped)
    _resolved_fragility_rank: dict[str, int] | None = field(default=None, init=False, repr=False)

    def _resolve_fragility_rank(self) -> dict[str, int]:
        """Collapse ``fragility_rank`` through ``label_replacements`` into final-name
        ranks.
        """
        if self._resolved_fragility_rank is not None:
            return self._resolved_fragility_rank
        if not self.label_replacements:
            self._resolved_fragility_rank = dict(self.fragility_rank)
            return self._resolved_fragility_rank
        resolved: dict[str, int] = {}
        for raw_name, rank in self.fragility_rank.items():
            final_name = self.label_replacements.get(raw_name, raw_name)
            # Dropped by label_replacements (e.g. "I: None"), YAML parses an
            # unquoted "None" as the string "None", not null, so both forms
            # must be checked (matches dataset_prep.stats.apply_label_replacements).
            if final_name is None or (isinstance(final_name, str) and final_name.strip().lower() == "none"):
                continue
            if final_name not in resolved or rank > resolved[final_name]:
                resolved[final_name] = rank
        self._resolved_fragility_rank = resolved
        return resolved

    def _rank_for_code(self, class_names: list[str]) -> np.ndarray:
        """Return a fragility-rank array indexed by class code (position in class_names).
        """
        ranks = self._resolve_fragility_rank()
        if self.label_encoding:
            code_to_rank = {
                str(code): ranks.get(name, 0)
                for name, code in self.label_encoding.items()
            }
            return np.array([code_to_rank.get(str(name), 0) for name in class_names])
        return np.array([ranks.get(str(name), 0) for name in class_names])

    def build_matrix(self, class_names: list[str]) -> np.ndarray:
        """Build the misclassification cost matrix for a set of classes."""
        n = len(class_names)
        ranks = self._rank_for_code(class_names)
        C = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                delta = int(ranks[i]) - int(ranks[j])
                if delta > 0:
                    C[i, j] = delta * self.pessimism_factor
                else:
                    C[i, j] = abs(delta) * 1.0
        return C

    def class_weights(self, class_names: list[str]) -> dict[int, float]:
        """Compute per-class weights proportional to the total misclassification cost for
        each class.
        """
        C = self.build_matrix(class_names)
        row_sums = C.sum(axis=1)
        if row_sums.max() == 0:
            return {i: 1.0 for i in range(len(class_names))}
        min_w = float(row_sums[row_sums > 0].min())
        return {i: float(row_sums[i] / min_w) for i in range(len(class_names))}


@dataclass
class BootstrapConfig:
    """Bootstrap CI settings for per-model uncertainty estimation in quality assessment.
    """
    n_samples: int = 1000
    confidence: float = 0.95
    models: list[str] = field(default_factory=list)  # empty list = disabled for all models

    def enabled_for(self, model_name: str) -> bool:
        """Check whether bootstrap CI is enabled for a given model."""
        return model_name in self.models

    @property
    def any_enabled(self) -> bool:
        """Whether bootstrap CI is enabled for at least one model."""
        return len(self.models) > 0


@dataclass
class ExperimentConfig:
    # Paths (resolved relative to the config.yaml's own directory)
    config_dir: Path
    train_path: Path
    test_path: Path
    models_dir: Path

    # Data
    label: str
    random_state: int = 42
    n_splits: int = 5

    # Optional raw GPKGs (with geometry), used by the report's prediction
    # map. None unless set via dataset.train_gpkg / dataset.test_gpkg.
    train_gpkg: Path | None = None
    test_gpkg: Path | None = None

    # Optional explicit val split (preprocessed CSV + raw gpkg), mirroring
    # train/test above. None unless set via dataset.val_path/dataset.val_gpkg
    # in config.yaml, when present, threshold calibration uses this split
    # instead of carving an internal holdout out of train (see
    # models/base.py's run_final/_optimize_thresholds).
    val_path: Path | None = None
    val_gpkg: Path | None = None

    # MLflow
    mlflow_tracking_uri: str = ""
    experiment_cv: str = "multiclass_model_selection"
    experiment_final: str = "final_evaluation"

    # Optional redirect for run_final()'s hyperparameter lookup to a
    # *different* experiment's MLflow db, e.g. a small, fast "hp_search"
    # dataset (subsampled per group) whose CV results are reused across
    # several full-size experiments instead of each one re-running its own
    # grid search. None (the default) keeps the historical behavior: look up
    # the best run in this same experiment's own db/experiment_cv. Set via
    # mlflow.hyperparam_db_path (resolved like mlflow.db_path) and, if the
    # other experiment logs under a different name, mlflow.hyperparam_experiment_cv.
    hyperparam_tracking_uri: str | None = None
    hyperparam_experiment_cv: str | None = None

    # Models
    enabled_models: list[str] = field(default_factory=list)
    skip_completed: bool = True
    start_from_model: str | None = None

    # Assessment
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    # Cost-sensitive training (optional)
    cost_matrix: CostMatrixConfig | None = None

    # What threshold calibration optimizes. None keeps the historical
    # behavior: expected cost when cost_matrix is configured, macro F1
    # otherwise. Set to "f1_macro" to calibrate on macro F1 even when a
    # cost_matrix is present (the cost_sensitive CV strategy still runs).
    calibration_objective: Literal["expected_cost", "f1_macro"] | None = None

    # Group columns from preprocessing.group (e.g. city/district), used for
    # the optional per-group test breakdown when test_gpkg is configured.
    group_columns: list[str] = field(default_factory=list)

    # Cached at load time so every pipeline doesn't re-read the CSV to know
    # how many classes there are (XGBoost's num_class needs this up front),
    # and so train/test always share the exact same string->int mapping.
    _num_classes: int | None = field(default=None, init=False, repr=False)
    _class_names: list[str] | None = field(default=None, init=False, repr=False)
    _label_mapping: dict[str, int] | None = field(default=None, init=False, repr=False)
    _label_mapping_loaded: bool = field(default=False, init=False, repr=False)

    @property
    def num_classes(self) -> int | None:
        """Number of distinct classes in the training data."""
        self._ensure_classes_known()
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        """Class names in the fixed order used for label encoding (index == encoded value).
        """
        self._ensure_classes_known()
        return self._class_names

    def _ensure_classes_known(self) -> None:
        """Populate ``_class_names``/``_num_classes`` from the training CSV, once."""
        if self._class_names is not None:
            return
        df = pd.read_csv(self.train_path)
        self._class_names = sorted(df[self.label].astype(str).unique())
        self._num_classes = len(self._class_names)

    def inverse_label(self, encoded: int) -> str:
        """Map an encoded integer label back to its original class name."""
        return self.class_names[encoded] # type: ignore

    def _load_label_mapping(self) -> dict[str, int] | None:
        """Load ``{class_name: code}`` from preprocessing.json, if present."""
        if self._label_mapping_loaded:
            return self._label_mapping
        self._label_mapping_loaded = True
        json_path = self.train_path.parent / "preprocessing.json"
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
            mapping = meta.get("label_mapping")
            if mapping:
                self._label_mapping = {str(k): int(v) for k, v in mapping.items()}
        return self._label_mapping

    @property
    def label_map(self) -> dict[int, str]:
        """Build ``{code: "name (code)"}`` for every class."""
        label_mapping = self._load_label_mapping()
        if label_mapping:
            code_to_name = {v: k for k, v in label_mapping.items()}
        elif self.cost_matrix and self.cost_matrix.label_encoding:
            code_to_name = {v: k for k, v in self.cost_matrix.label_encoding.items()}
        elif self.class_names:
            code_to_name = dict(enumerate(self.class_names))
        else:
            return {}
        return {code: f"{name} ({code})" for code, name in code_to_name.items()}

    @property
    def display_names(self) -> list[str]:
        """Build "name (code)" strings for every class, in code order (index == code).
        """
        lm = self.label_map
        n = len(self.class_names) if self.class_names else 0
        return [lm.get(i, str(i)) for i in range(n)]

    def load_data(self, split: Literal["train", "test", "val"]) -> tuple[pd.DataFrame, pd.Series]:
        """Load a data split's features and labels, normalizing labels to integers."""
        if split == "val":
            if self.val_path is None:
                raise ConfigError("load_data('val') called but no dataset.val_path is configured.")
            path = self.val_path
        else:
            path = self.train_path if split == "train" else self.test_path
        df = pd.read_csv(path)
        if self.label not in df.columns:
            raise ConfigError(
                f"Label column '{self.label}' not found in {path}. "
                f"Available columns: {list(df.columns)}"
            )
        x = df.drop(columns=[self.label])
        y = df[self.label]

        if not pd.api.types.is_integer_dtype(y):
            name_to_code = {name: code for code, name in enumerate(self.class_names)} # type: ignore
            unknown = set(y.astype(str).unique()) - set(name_to_code)
            if unknown:
                raise ConfigError(
                    f"Label(s) {sorted(unknown)} found in {split} split but not in the "
                    "training set's class list. Every class in test must also appear in train."
                )
            y = y.astype(str).map(name_to_code)

        return x, y


def propagate_preprocessing_metadata(cfg: "ExperimentConfig", work_dir: Path) -> None:
    """Copy preprocessing.json into a specific model's work directory."""
    src = cfg.train_path.parent / "preprocessing.json"
    if not src.exists():
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / "preprocessing.json"
    if src.resolve() == dst.resolve():
        return
    dst.write_bytes(src.read_bytes())


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate an ``ExperimentConfig`` from a YAML file."""
    config_path = Path(config_path).resolve()
    raw = load_yaml_with_extends(config_path)
    config_dir = config_path.parent

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (config_dir / path).resolve()

    try:
        dataset = raw["dataset"]
        train_path = resolve(dataset["train_path"])
        test_path = resolve(dataset["test_path"])
        label = raw["label"]
    except KeyError as exc:
        raise ConfigError(f"Missing required config key: {exc}") from exc

    if not train_path.exists():
        raise ConfigError(f"dataset.train_path does not exist: {train_path}")
    if not test_path.exists():
        raise ConfigError(f"dataset.test_path does not exist: {test_path}")

    train_gpkg = resolve(dataset["train_gpkg"]) if dataset.get("train_gpkg") else None
    test_gpkg = resolve(dataset["test_gpkg"]) if dataset.get("test_gpkg") else None
    if train_gpkg is not None and not train_gpkg.exists():
        raise ConfigError(f"dataset.train_gpkg does not exist: {train_gpkg}")
    if test_gpkg is not None and not test_gpkg.exists():
        raise ConfigError(f"dataset.test_gpkg does not exist: {test_gpkg}")

    val_path = resolve(dataset["val_path"]) if dataset.get("val_path") else None
    if val_path is not None and not val_path.exists():
        raise ConfigError(f"dataset.val_path does not exist: {val_path}")
    val_gpkg = resolve(dataset["val_gpkg"]) if dataset.get("val_gpkg") else None
    if val_gpkg is not None and not val_gpkg.exists():
        raise ConfigError(f"dataset.val_gpkg does not exist: {val_gpkg}")

    models_cfg = raw.get("models", {})
    enabled_models = models_cfg.get("enabled", [])
    if not enabled_models:
        raise ConfigError("models.enabled must list at least one model name.")

    # Fail fast on a typo'd model name instead of discovering it mid-run.
    from .models.registry import MODEL_REGISTRY  # local import: avoids a cycle with models/base.py

    unknown = set(enabled_models) - set(MODEL_REGISTRY)
    if unknown:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ConfigError(
            f"Unknown model(s) in models.enabled: {sorted(unknown)}. Available: {available}."
        )

    n_splits = raw.get("n_splits", 5)
    if n_splits < 2:
        raise ConfigError(f"n_splits must be >= 2 (got {n_splits}).")

    models_dir = resolve(raw.get("models_dir", "trained_models"))
    mlflow_cfg = raw.get("mlflow", {})
    mlflow_db_path = resolve(mlflow_cfg.get("db_path", "mlflow.db"))

    hyperparam_tracking_uri: str | None = None
    if mlflow_cfg.get("hyperparam_tracking_uri"):
        hyperparam_tracking_uri = mlflow_cfg["hyperparam_tracking_uri"]
    elif mlflow_cfg.get("hyperparam_db_path"):
        hyperparam_tracking_uri = f"sqlite:///{resolve(mlflow_cfg['hyperparam_db_path'])}"

    assessment_cfg = raw.get("assessment", {})
    bootstrap_raw = assessment_cfg.get("bootstrap", {})
    bootstrap = BootstrapConfig(
        n_samples=int(bootstrap_raw.get("n_samples", 1000)),
        confidence=float(bootstrap_raw.get("confidence", 0.95)),
        models=list(bootstrap_raw.get("models") or []),
    )

    calibration_objective = raw.get("calibration_objective")
    if calibration_objective is not None:
        if calibration_objective not in ("expected_cost", "f1_macro"):
            raise ConfigError(
                f"calibration_objective must be 'expected_cost' or 'f1_macro' "
                f"(got {calibration_objective!r})."
            )
        if calibration_objective == "expected_cost" and not raw.get("cost_matrix"):
            raise ConfigError(
                "calibration_objective: expected_cost requires a cost_matrix section."
            )

    cost_matrix_raw = raw.get("cost_matrix")
    cost_matrix: CostMatrixConfig | None = None
    if cost_matrix_raw:
        enc_raw = cost_matrix_raw.get("label_encoding")
        label_replacements_raw = raw.get("preprocessing", {}).get("label_replacements")
        cost_matrix = CostMatrixConfig(
            fragility_rank={str(k): int(v) for k, v in cost_matrix_raw.get("fragility_rank", {}).items()},
            pessimism_factor=float(cost_matrix_raw.get("pessimism_factor", 3.0)),
            label_encoding={str(k): int(v) for k, v in enc_raw.items()} if enc_raw else None,
            label_replacements=dict(label_replacements_raw) if label_replacements_raw else None,
        )

    return ExperimentConfig(
        config_dir=config_dir,
        train_path=train_path,
        test_path=test_path,
        train_gpkg=train_gpkg,
        test_gpkg=test_gpkg,
        val_path=val_path,
        val_gpkg=val_gpkg,
        models_dir=models_dir,
        label=label,
        random_state=raw.get("random_state", 42),
        n_splits=n_splits,
        mlflow_tracking_uri=mlflow_cfg.get("tracking_uri", f"sqlite:///{mlflow_db_path}"),
        experiment_cv=mlflow_cfg.get("experiment_cv", "multiclass_model_selection"),
        experiment_final=mlflow_cfg.get("experiment_final", "final_evaluation"),
        hyperparam_tracking_uri=hyperparam_tracking_uri,
        hyperparam_experiment_cv=mlflow_cfg.get("hyperparam_experiment_cv"),
        enabled_models=enabled_models,
        skip_completed=models_cfg.get("skip_completed", True),
        start_from_model=models_cfg.get("start_from_model"),
        bootstrap=bootstrap,
        cost_matrix=cost_matrix,
        calibration_objective=calibration_objective,
        # "dataset" (tagged automatically by load_tabular_source with each
        # source file's stem) is always included alongside whatever's
        # configured, so a per-source-dataset breakdown is available for
        # every experiment without needing explicit config.
        group_columns=list(
            dict.fromkeys([*(raw.get("preprocessing", {}).get("group") or []), "dataset"])
        ),
    )
