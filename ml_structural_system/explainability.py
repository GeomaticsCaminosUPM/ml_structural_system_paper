"""Explainability module: feature importances and SHAP-based explanations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureImportanceResult:
    """Feature importances for one model."""
    model_name: str
    method: str              # "built_in", "coef_abs", "unsupported"
    importances: pd.Series   # index=feature_name, values sorted descending
    # mean |SHAP| per sample: shape (n_samples, n_features), or None if unavailable
    shap_values: np.ndarray | None = None


@dataclass
class ExplainabilityResult:
    """Aggregated explainability across all trained models."""
    per_model: list[FeatureImportanceResult]
    # Consensus ranking: mean rank position (1-based) of each feature across
    # models. Lower = feature consistently ranks high across models.
    consensus_ranking: pd.Series   # index=feature_name, values=mean rank (float)
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_artifacts(model_name: str, cfg: ExperimentConfig) -> dict[str, Any]:
    """Load trained model artifact(s) from disk."""
    work_dir = cfg.models_dir / model_name
    model_path = work_dir / f"model_final_{model_name}.pkl"

    if not model_path.exists():
        logger.warning("No trained model found at %s, skipping %s.", model_path, model_name)
        return {}

    artifact = joblib.load(model_path)

    return {"model": artifact}


def _is_gpu_trained_catboost(model: Any) -> bool:
    """Whether ``model`` is a CatBoost estimator whose ``task_type`` was ``"GPU"`` at fit
    time, see the crash this guards against in ``_compute_shap``'s own comment.
    """
    get_params = getattr(model, "get_params", None)
    if get_params is None:
        return False
    try:
        params = get_params()
    except Exception:  # noqa: BLE001, defensive, get_params() should not normally raise
        return False
    return str(params.get("task_type", "")).upper() == "GPU"


def _compute_catboost_native_shap(model: Any, X: pd.DataFrame, model_name: str) -> np.ndarray | None:
    """Mean |SHAP| values for a GPU-trained CatBoost model via CatBoost's own native SHAP
    implementation, bypassing the external ``shap`` package's TreeExplainer entirely
    (see the crash it avoids, documented in ``_compute_shap``'s call site).
    """
    try:
        from catboost import Pool  # noqa: PLC0415, only needed on this GPU-CatBoost path

        raw = np.asarray(model.get_feature_importance(Pool(X), type="ShapValues"))
        if raw.ndim == 3:
            # (n_samples, n_classes, n_features+1) -> drop base value, mean |.| across classes
            return np.abs(raw[:, :, :-1]).mean(axis=1)
        if raw.ndim == 2:
            # (n_samples, n_features+1) -> drop base value column
            return np.abs(raw[:, :-1])
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("CatBoost native SHAP computation failed for %s: %s", model_name, exc)
        return None


def _extract_importances(model: Any, feature_names: list[str]) -> tuple[pd.Series, str]:
    """Extract a feature-importance Series from a fitted estimator."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(
            model.feature_importances_,
            index=feature_names,
            name="importance",
        )
        return imp.sort_values(ascending=False), "built_in"

    if hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_))
        if coef.ndim == 2:
            coef = coef.mean(axis=0)
        imp = pd.Series(coef, index=feature_names, name="importance")
        return imp.sort_values(ascending=False), "coef_abs"

    return pd.Series(dtype=float, name="importance"), "unsupported"


def _compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> tuple[pd.Series, str]:
    """Model-agnostic importance via macro-F1 drop when each feature is shuffled."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(random_state)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    baseline_f1 = f1_score(
        y_arr, model.predict(X), average="macro", zero_division=0
    )

    scores = np.zeros((len(feature_names), n_repeats))
    for j in range(len(feature_names)):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm.iloc[:, j] = rng.permutation(X_perm.iloc[:, j].values)
            perm_f1 = f1_score(
                y_arr, model.predict(X_perm), average="macro", zero_division=0
            )
            scores[j, r] = baseline_f1 - perm_f1

    imp = pd.Series(scores.mean(axis=1), index=feature_names, name="importance")
    imp = imp.clip(lower=0).sort_values(ascending=False)
    return imp, "permutation"


def _shap_to_array(shap_output: Any) -> np.ndarray | None:
    """Normalise the various output shapes that shap returns."""
    # Unwrap Explanation objects
    if hasattr(shap_output, "values"):
        shap_output = shap_output.values

    if isinstance(shap_output, list):
        arrays = [np.asarray(v, dtype=float) for v in shap_output]
        if not arrays:
            return None
        arr = np.stack(arrays, axis=-1)   # → (n_samples, n_features, n_classes)
        return arr

    arr = np.asarray(shap_output, dtype=float)
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]      # treat as single class
    return arr                            # assume already 3-D


def _compute_shap(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
) -> np.ndarray | None:
    """Compute mean |SHAP| values across classes."""
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        logger.warning("shap not installed; skipping SHAP explanations. Run: uv add shap")
        return None

    try:
        if _is_gpu_trained_catboost(model):
            # shap.TreeExplainer can crash the process on a GPU-trained CatBoost
            # model, so use CatBoost's native SHAP values instead.
            return _compute_catboost_native_shap(model, X, model_name)
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(X)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X)
            raw = explainer.shap_values(X)
        else:
            return None

        arr = _shap_to_array(raw)
        if arr is None:
            return None
        # mean |SHAP| across classes → (n_samples, n_features)
        return np.abs(arr).mean(axis=-1)

    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP computation failed for %s: %s", model_name, exc)
        return None


def _build_consensus_ranking(results: list[FeatureImportanceResult]) -> pd.Series:
    """Compute a consensus feature ranking across all models."""
    rank_frames: list[pd.Series] = []

    for res in results:
        if res.shap_values is not None:
            # Rank by mean |SHAP| across samples
            mean_shap = pd.Series(
                res.shap_values.mean(axis=0),
                index=res.importances.index,
            )
            ranked = mean_shap.rank(ascending=False, method="min")
        elif len(res.importances) > 0:
            ranked = res.importances.rank(ascending=False, method="min")
        else:
            continue
        rank_frames.append(ranked.rename(res.model_name))

    if not rank_frames:
        return pd.Series(dtype=float, name="consensus_rank")

    df = pd.concat(rank_frames, axis=1)
    consensus = df.mean(axis=1).sort_values()
    consensus.name = "consensus_rank"
    return consensus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_explainability(
    cfg: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series | None = None,
    *,
    models: list[str] | None = None,
    compute_shap: bool = True,
) -> ExplainabilityResult:
    """Load each trained model and compute feature importances and SHAP values."""
    model_names = models if models is not None else cfg.enabled_models
    feature_names = list(X.columns)
    per_model: list[FeatureImportanceResult] = []

    for name in model_names:
        logger.info("Computing feature importances for %s …", name)
        artifacts = _load_artifacts(name, cfg)

        if not artifacts:
            continue

        model = artifacts["model"]
        imp, method = _extract_importances(model, feature_names)
        shap_vals = _compute_shap(model, X, name) if compute_shap else None

        if imp.empty and y is not None:
            logger.info(
                "  No built-in importances for %s, falling back to permutation importance.", name
            )
            try:
                imp, method = _compute_permutation_importance(
                    model, X, y, feature_names
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Permutation importance failed for %s: %s", name, exc)

        per_model.append(
            FeatureImportanceResult(
                model_name=name,
                method=method,
                importances=imp,
                shap_values=shap_vals,
            )
        )

    consensus = _build_consensus_ranking(per_model)

    return ExplainabilityResult(
        per_model=per_model,
        consensus_ranking=consensus,
        feature_names=feature_names,
    )
