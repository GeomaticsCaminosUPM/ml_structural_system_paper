"""Base pipeline shared by every model (LogisticRegression, RandomForest, XGBoost,
CatBoost, HistGradientBoosting).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from sklearn.base import BaseEstimator

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from tqdm.auto import tqdm

from ..config import ExperimentConfig
from ..mlflow_utils import cast_params, extract_params, get_best_run
from ..progress import bar_disabled

logger = logging.getLogger(__name__)


def _macro_true_skill_statistic(cm: np.ndarray) -> float:
    """Macro-averaged True Skill Statistic (Allouche et al. 2006) from a confusion matrix.
    """
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    tss_per_class = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        tss_per_class.append(sensitivity + specificity - 1.0)
    return float(np.mean(tss_per_class))


class BaseModelPipeline(ABC):
    """Shared CV/final/test pipeline for a single model type."""

    #: Short, stable identifier used in MLflow params and file names.
    model_type: str = "BaseModel"

    #: Hyperparameter grid swept during cross-validation.
    param_grid: dict[str, Iterable[Any]] = {}

    #: Resampling/weighting strategies compared during cross-validation.
    #: "resampler" is None | "smote" | "smote_tomek".
    #: smote_tomek oversamples minority classes and then removes borderline
    #: majority samples, which tends to produce cleaner decision boundaries
    #: than plain SMOTE alone.
    cv_strategies: dict[str, dict] = {
        "baseline":    {"resampler": None,          "use_class_weight": False},
        "class_weight":{"resampler": None,          "use_class_weight": True},
        "smote":       {"resampler": "smote",       "use_class_weight": False},
        "smote_tomek": {"resampler": "smote_tomek", "use_class_weight": False},
    }

    def __init__(self, cfg: ExperimentConfig, work_dir: Path) -> None:
        """Initialize the pipeline and its MLflow tracking URI."""
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        if cfg.cost_matrix is not None:
            self.cv_strategies = dict(self.cv_strategies)
            self.cv_strategies["cost_sensitive"] = {
                "resampler": None,
                "use_class_weight": False,
                "use_cost_weight": True,
            }

    # ------------------------------------------------------------------
    # Model-specific hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Return an unfitted estimator configured with ``params``."""
        raise NotImplementedError

    def build_final_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Return an unfitted estimator for the final refit (``run_final``)."""
        return self.build_model(params, class_weight=class_weight)

    def fit_params(self, x_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Return extra keyword arguments to pass to ``model.fit()`` during CV."""
        return {}

    def sample_weight_for(
        self, y_train: pd.Series, class_weight: dict | None = None
    ) -> np.ndarray | None:
        """Return per-sample weights to pass as ``sample_weight`` during ``model.fit()``.
        """
        return None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def class_weights(y: pd.Series) -> dict:
        """Compute frequency-based {class_code: weight} for ``y``."""
        counts = y.value_counts()
        return (counts.max() / counts).to_dict()

    def _model_final_path(self) -> Path:
        """Return the path the final refit model is saved to/loaded from."""
        return self.work_dir / f"model_final_{self.model_type}.pkl"

    def _thresholds_path(self) -> Path:
        """Return the path the calibrated per-class thresholds are saved to/loaded from.
        """
        return self.work_dir / f"thresholds_{self.model_type}.npy"

    def _optimize_thresholds(
        self, model: Any, x_cal: pd.DataFrame, y_cal: pd.Series
    ) -> np.ndarray:
        """Find per-class probability divisors that optimize the decision criterion."""
        proba = model.predict_proba(x_cal)
        y_arr = y_cal.values
        n_classes = proba.shape[1]

        cost_matrix = (
            self.cfg.cost_matrix.build_matrix(self.cfg.class_names)  # type: ignore[arg-type]
            if self.cfg.cost_matrix is not None
            and self.cfg.calibration_objective != "f1_macro"
            else None
        )

        def score(t: np.ndarray) -> float:
            adjusted = proba / t
            y_pred = adjusted.argmax(axis=1)
            if cost_matrix is not None:
                return float(cost_matrix[y_arr, y_pred].mean())
            return -float(f1_score(y_arr, y_pred, average="macro", zero_division=0))

        # Log-spaced grid: 0.05 (very eager to predict class) → 5.0 (very reluctant)
        grid = np.exp(np.linspace(np.log(0.05), np.log(5.0), 120))
        thresholds = np.ones(n_classes)

        for _ in range(2):  # two passes of coordinate descent
            for k in range(n_classes):
                best_t = thresholds[k]
                best_s = score(thresholds)
                for t in grid:
                    trial = thresholds.copy()
                    trial[k] = t
                    s = score(trial)
                    if s < best_s:
                        best_s = s
                        best_t = t
                thresholds[k] = best_t

        return thresholds

    def _apply_resampler(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        resampler: str | None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Oversample minority classes using the requested strategy."""
        if resampler is None:
            return x, y
        min_class_count = int(y.value_counts().min())
        if min_class_count < 2:
            raise ValueError(
                f"Cannot resample: the rarest class has only {min_class_count} sample(s). "
                "Reduce n_splits in config.yaml or increase the minimum class count in "
                "dataset preprocessing."
            )
        k_neighbors = max(1, min(5, min_class_count - 1))
        smote = SMOTE(random_state=self.cfg.random_state, k_neighbors=k_neighbors)
        if resampler == "smote":
            sampler = smote
        elif resampler == "smote_tomek":
            sampler = SMOTETomek(random_state=self.cfg.random_state, smote=smote)
        else:
            raise ValueError(f"Unknown resampler {resampler!r}. Choose 'smote' or 'smote_tomek'.")
        x_res, y_res = sampler.fit_resample(x, y)
        return pd.DataFrame(x_res, columns=x.columns), pd.Series(y_res, name=y.name)

    def _fit_fold(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        model: Any,
        resampler: str | None,
        use_class_weight: bool,
        use_cost_weight: bool = False,
    ) -> tuple[Any, float, float, np.ndarray]:
        """Resample, fit, and evaluate a single CV fold."""
        x_res, y_res = self._apply_resampler(x_train, y_train, resampler)

        fit_kwargs = self.fit_params(x_val, y_val)
        if use_cost_weight and self.cfg.cost_matrix is not None:
            cw = self.cfg.cost_matrix.class_weights(self.cfg.class_names)  # type: ignore[arg-type]
            weights = self.sample_weight_for(y_res, cw)
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
        elif use_class_weight:
            weights = self.sample_weight_for(y_res)
            if weights is not None:
                fit_kwargs["sample_weight"] = weights

        model.fit(x_res, y_res, **fit_kwargs)
        y_pred = model.predict(x_val)
        all_codes = list(range(len(self.cfg.class_names)))  # type: ignore[arg-type]
        f1 = f1_score(y_val, y_pred, average="macro", labels=all_codes)
        acc = accuracy_score(y_val, y_pred)
        f1_per_class = f1_score(y_val, y_pred, average=None, labels=all_codes, zero_division=0)
        return model, f1, acc, f1_per_class

    # ------------------------------------------------------------------
    # Stage 1: cross-validation
    # ------------------------------------------------------------------

    def run_cv(self) -> None:
        """Cross-validated hyperparameter search, fully logged to MLflow."""
        mlflow.set_experiment(self.cfg.experiment_cv)

        x, y = self.cfg.load_data(split="train")
        skf = StratifiedKFold(
            n_splits=self.cfg.n_splits, shuffle=True, random_state=self.cfg.random_state
        )

        grid = list(ParameterGrid(self.param_grid)) if self.param_grid else [{}]
        total_fits = len(self.cv_strategies) * len(grid) * self.cfg.n_splits
        # tqdm no-ops pbar.update() (and leaves pbar.n at 0) when disable=True
        #, exactly the case here whenever this runs unattended, so the
        # heartbeat log below tracks fit count itself rather than reading it
        # back from pbar.n.
        fits_done = 0

        with tqdm(total=total_fits, desc=f"{self.model_type} CV", unit="fit", disable=bar_disabled()) as pbar:
            for strategy_name, strategy in self.cv_strategies.items():
                for grid_params in grid:
                    run_name = f"{self.model_type}_{strategy_name}_{grid_params}"
                    with mlflow.start_run(run_name=run_name):
                        mlflow.log_param("model_type", self.model_type)
                        mlflow.log_param("strategy", strategy_name)
                        mlflow.log_params(grid_params)

                        f1_scores: list[float] = []
                        acc_scores: list[float] = []
                        f1_per_class_folds: list[np.ndarray] = []
                        best_iters: list[int] = []

                        use_cost_weight = strategy.get("use_cost_weight", False)

                        for fold, (tr_idx, val_idx) in enumerate(skf.split(x, y), start=1):
                            x_train, x_val = x.iloc[tr_idx], x.iloc[val_idx]
                            y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                            if use_cost_weight and self.cfg.cost_matrix is not None:
                                class_weight = self.cfg.cost_matrix.class_weights(
                                    self.cfg.class_names  # type: ignore[arg-type]
                                )
                            elif strategy["use_class_weight"]:
                                class_weight = self.class_weights(y_train)
                            else:
                                class_weight = None

                            model = self.build_model(grid_params, class_weight=class_weight)

                            pbar.set_postfix(
                                strategy=strategy_name,
                                fold=f"{fold}/{self.cfg.n_splits}",
                                status="fitting",
                            )
                            fitted_model, f1, acc, f1_per_class = self._fit_fold(
                                x_train,
                                y_train,
                                x_val,
                                y_val,
                                model,
                                resampler=strategy["resampler"],
                                use_class_weight=strategy["use_class_weight"],
                                use_cost_weight=use_cost_weight,
                            )

                            mlflow.log_metric(f"f1_macro_fold_{fold}", f1)
                            mlflow.log_metric(f"accuracy_fold_{fold}", acc)
                            for cls_idx, cls_f1 in enumerate(f1_per_class):
                                mlflow.log_metric(f"f1_class{cls_idx}_fold_{fold}", float(cls_f1))

                            if (
                                hasattr(fitted_model, "best_iteration_")
                                and fitted_model.best_iteration_ is not None
                            ):
                                best_iters.append(int(fitted_model.best_iteration_))

                            f1_scores.append(f1)
                            acc_scores.append(acc)
                            f1_per_class_folds.append(f1_per_class)

                            pbar.set_postfix(
                                strategy=strategy_name,
                                fold=f"{fold}/{self.cfg.n_splits}",
                                f1=f"{f1:.3f}",
                            )
                            pbar.update(1)
                            fits_done += 1

                        if best_iters:
                            mlflow.log_metric(
                                "mean_best_iteration",
                                int(round(np.mean(best_iters))) + 1,
                            )

                        mlflow.log_metric("mean_f1_macro", float(np.mean(f1_scores)))
                        mlflow.log_metric("std_f1_macro", float(np.std(f1_scores)))
                        mlflow.log_metric("mean_acc", float(np.mean(acc_scores)))
                        # tqdm's own CV bar (pbar above) is disabled outside a
                        # real terminal (see progress.bar_disabled), this is
                        # the periodic, single-line progress marker for
                        # unattended runs, showing which combo just finished
                        # and the grid search's overall position.
                        logger.info(
                            "  [%s] %s %s: mean_f1_macro=%.4f (%d/%d fits done)",
                            self.model_type, strategy_name, grid_params,
                            float(np.mean(f1_scores)), fits_done, total_fits,
                        )
                        if f1_per_class_folds:
                            mean_per_class = np.mean(f1_per_class_folds, axis=0)
                            for cls_idx, cls_mean in enumerate(mean_per_class):
                                mlflow.log_metric(f"mean_f1_class{cls_idx}", float(cls_mean))

    # ------------------------------------------------------------------
    # Stage 2: final refit on full training data
    # ------------------------------------------------------------------

    def run_final(self) -> Path:
        """Refit on the full training set using the best CV configuration."""
        mlflow.set_experiment(self.cfg.experiment_cv)

        x, y = self.cfg.load_data(split="train")

        # Hyperparameters normally come from this experiment's own CV run
        # (self.cfg.experiment_cv, in this experiment's own MLflow db). When
        # mlflow.hyperparam_db_path (config.yaml) is set, look them up in a
        # different experiment's db instead, e.g. a small, fast "hp_search"
        # dataset whose CV results are shared across several full-size
        # experiments, then switch the tracking URI back before any further
        # MLflow calls in this method, which must log to this experiment's
        # own db.
        if self.cfg.hyperparam_tracking_uri:
            mlflow.set_tracking_uri(self.cfg.hyperparam_tracking_uri)
            hp_experiment = self.cfg.hyperparam_experiment_cv or self.cfg.experiment_cv
            mlflow.set_experiment(hp_experiment)
            best_run = get_best_run(hp_experiment, self.model_type)
            mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
            mlflow.set_experiment(self.cfg.experiment_cv)
        else:
            best_run = get_best_run(self.cfg.experiment_cv, self.model_type)
        _raw = best_run.get("metrics.mean_best_iteration")
        self._final_n_estimators: int | None = int(_raw) if _raw is not None else None
        if isinstance(self.param_grid, dict):
            _allowed = set(self.param_grid.keys())
        else:
            _allowed = {k for d in self.param_grid for k in d}
        best_params = extract_params(best_run, allowed_keys=_allowed)
        best_params = cast_params(best_params)

        strategy_name = best_run.get("params.strategy", "baseline")
        strategy = self.cv_strategies.get(
            strategy_name, {"resampler": None, "use_class_weight": False}
        )
        resampler = strategy["resampler"]
        use_class_weight = strategy["use_class_weight"]
        use_cost_weight = strategy.get("use_cost_weight", False)
        logger.info(
            "Best CV strategy for %s: '%s' (resampler=%s, class_weight=%s)",
            self.model_type, strategy_name, resampler, use_class_weight,
        )

        # --- Threshold calibration: on the explicit val split when
        # configured (dataset.val_path in config.yaml, see
        # dataset_prep.splitting's per-dataset 3-way split), else on a
        # stratified 20% holdout of train (the historical behavior) ---
        if self.cfg.val_path is not None:
            x_fit, y_fit = x, y
            x_cal, y_cal = self.cfg.load_data(split="val")
        else:
            x_fit, x_cal, y_fit, y_cal = train_test_split(
                x, y, test_size=0.2, stratify=y, random_state=self.cfg.random_state
            )
        x_fit_res, y_fit_res = self._apply_resampler(x_fit, y_fit, resampler)

        if use_cost_weight and self.cfg.cost_matrix is not None:
            cw_cal = self.cfg.cost_matrix.class_weights(self.cfg.class_names)  # type: ignore[arg-type]
        elif use_class_weight:
            cw_cal = self.class_weights(y_fit_res)
        else:
            cw_cal = None

        cal_model = self.build_final_model(best_params, class_weight=cw_cal)
        cal_fit_kwargs: dict[str, Any] = {}
        if use_cost_weight and self.cfg.cost_matrix is not None:
            w = self.sample_weight_for(y_fit_res, cw_cal)
            if w is not None:
                cal_fit_kwargs["sample_weight"] = w
        elif use_class_weight:
            w = self.sample_weight_for(y_fit_res)
            if w is not None:
                cal_fit_kwargs["sample_weight"] = w
        cal_model.fit(x_fit_res, y_fit_res, **cal_fit_kwargs)

        if hasattr(cal_model, "predict_proba"):
            thresholds = self._optimize_thresholds(cal_model, x_cal, y_cal)
            np.save(self._thresholds_path(), thresholds)
            logger.info(
                "Calibrated thresholds for %s: %s",
                self.model_type, np.round(thresholds, 4),
            )

        # --- Final refit on ALL training data ---
        x_res, y_res = self._apply_resampler(x, y, resampler)

        if use_cost_weight and self.cfg.cost_matrix is not None:
            class_weight = self.cfg.cost_matrix.class_weights(self.cfg.class_names)  # type: ignore[arg-type]
        elif use_class_weight:
            class_weight = self.class_weights(y_res)
        else:
            class_weight = None

        model = self.build_final_model(best_params, class_weight=class_weight)

        fit_kwargs: dict[str, Any] = {}
        if use_cost_weight and self.cfg.cost_matrix is not None:
            weights = self.sample_weight_for(y_res, class_weight)
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
        elif use_class_weight:
            weights = self.sample_weight_for(y_res)
            if weights is not None:
                fit_kwargs["sample_weight"] = weights

        model.fit(x_res, y_res, **fit_kwargs)

        model_path = self._model_final_path()
        joblib.dump(model, model_path)

        with mlflow.start_run(
            run_name=f"final_model_{self.model_type}",
            tags={"model_type": self.model_type, "stage": "final"},
        ):
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("strategy", strategy_name)
            mlflow.log_param("resampler", resampler or "none")
            mlflow.log_param("use_class_weight", use_class_weight)
            mlflow.log_param("parent_cv_run", best_run["run_id"])
            if self._final_n_estimators is not None:
                mlflow.log_param("final_n_estimators", self._final_n_estimators)
            mlflow.set_tag("parent_cv_run", best_run["run_id"])
            mlflow.log_artifact(str(model_path))
            if self._thresholds_path().exists():
                mlflow.log_artifact(str(self._thresholds_path()))

        return model_path

    # ------------------------------------------------------------------
    # Stage 3: test (labeled test set, this IS your held-out evaluation)
    # ------------------------------------------------------------------

    def run_test(self) -> dict[str, float]:
        """Evaluate the final model on the labeled test split."""
        mlflow.set_experiment(self.cfg.experiment_final)

        model = joblib.load(self._model_final_path())
        x_test, y_test = self.cfg.load_data(split="test")

        thresholds_path = self._thresholds_path()
        if thresholds_path.exists() and hasattr(model, "predict_proba"):
            thresholds = np.load(thresholds_path)
            proba = model.predict_proba(x_test)
            y_pred = (proba / thresholds).argmax(axis=1)
            logger.info(
                "Applied calibrated thresholds for %s: %s",
                self.model_type, np.round(thresholds, 4),
            )
        else:
            y_pred = model.predict(x_test)

        # Every class code, even one missing from this particular y_test/y_pred,
        # so labels/target_names always line up (report.py always shows
        # "name (code)", never a bare int, see ExperimentConfig.display_names).
        all_codes = list(range(len(self.cfg.class_names)))  # type: ignore[arg-type]
        display_names = self.cfg.display_names

        report = classification_report(
            y_test, y_pred, labels=all_codes, target_names=display_names, zero_division=0,
        )
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=all_codes)
        # Cohen's kappa and TSS: chance-corrected agreement between this
        # model's predictions and ground truth (unlike Fleiss' kappa
        # elsewhere in this codebase, which measures agreement *among*
        # several models). Reported for comparability with the remote-sensing
        # building-classification literature, which typically reports overall
        # accuracy + kappa + TSS rather than macro F1.
        cohen_kappa = cohen_kappa_score(y_test, y_pred, labels=all_codes)
        tss_macro = _macro_true_skill_statistic(cm)

        # Per-class F1: index i corresponds to cfg.inverse_label(i).
        f1_per_class = f1_score(y_test, y_pred, average=None, labels=all_codes)
        per_class_metrics = {
            f"f1_class_{self.cfg.inverse_label(i)}": float(f)
            for i, f in enumerate(f1_per_class)
        }

        logger.info("Classification report for %s:\n%s", self.model_type, report)

        cm_path = self.work_dir / "confusion_matrix_test.csv"
        report_path = self.work_dir / "classification_report_test.txt"
        pd.DataFrame(cm, index=display_names, columns=display_names).to_csv(cm_path)
        report_path.write_text(report)

        group_path: Path | None = None
        group_breakdown = self._group_breakdown(y_test, y_pred)
        if group_breakdown is not None:
            group_path = self.work_dir / "group_f1_test.csv"
            group_breakdown.to_csv(group_path, index=False)
            logger.info(
                "Per-group test breakdown for %s:\n%s",
                self.model_type, group_breakdown.to_string(index=False),
            )

        with mlflow.start_run(
            run_name=f"test_{self.model_type}",
            tags={"model_type": self.model_type, "stage": "test"},
        ):
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_metric("test_macro_f1", macro_f1)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_cohen_kappa", cohen_kappa)
            mlflow.log_metric("test_tss_macro", tss_macro)
            for key, val in per_class_metrics.items():
                mlflow.log_metric(key, val)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_text(report, "classification_report_test.txt")
            if group_path is not None:
                mlflow.log_artifact(str(group_path))

        return {
            "macro_f1": macro_f1, "accuracy": accuracy,
            "cohen_kappa": cohen_kappa, "tss_macro": tss_macro,
            **per_class_metrics,
        }

    def _group_breakdown(
        self, y_test: pd.Series, y_pred: np.ndarray
    ) -> pd.DataFrame | None:
        """Macro F1/accuracy per group value (e.g. per city or district) on test."""
        if self.cfg.test_gpkg is None or not self.cfg.group_columns:
            return None
        try:
            import geopandas as gpd

            gdf = gpd.read_file(self.cfg.test_gpkg)
        except Exception:  # noqa: BLE001 - breakdown is auxiliary, never fail the test stage
            logger.warning(
                "Could not read %s; skipping per-group test breakdown.",
                self.cfg.test_gpkg, exc_info=True,
            )
            return None

        cols = [c for c in self.cfg.group_columns if c in gdf.columns]
        if not cols:
            return None
        if len(gdf) != len(y_test):
            logger.warning(
                "test_gpkg has %d rows but the test split has %d; "
                "skipping per-group test breakdown.",
                len(gdf), len(y_test),
            )
            return None

        df = pd.DataFrame({col: gdf[col].to_numpy() for col in cols})
        df["y_true"] = np.asarray(y_test)
        df["y_pred"] = y_pred
        rows = []
        for col in cols:
            for value, g in df.groupby(col, dropna=False):
                rows.append({
                    "group_column": col,
                    "group_value": value,
                    "n": len(g),
                    "macro_f1": f1_score(
                        g["y_true"], g["y_pred"], average="macro", zero_division=0
                    ),
                    "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Convenience: run every stage in order
    # ------------------------------------------------------------------
