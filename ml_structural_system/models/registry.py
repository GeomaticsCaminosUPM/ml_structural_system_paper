"""Central registry mapping a model name (as used in ``config.yaml``) to its pipeline
class.
"""

from __future__ import annotations

from ..exceptions import ModelNotRegisteredError
from .base import BaseModelPipeline
from .catboost_model import CatBoostPipeline
from .hist_gradient_boosting import HistGradientBoostingPipeline
from .logistic_regression import LogisticRegressionPipeline
from .random_forest import RandomForestPipeline
from .xgboost_model import XGBoostPipeline

MODEL_REGISTRY: dict[str, type[BaseModelPipeline]] = {
    "LogisticRegression": LogisticRegressionPipeline,
    "RandomForest": RandomForestPipeline,
    "XGBoost": XGBoostPipeline,
    "CatBoost": CatBoostPipeline,
    "HistGradientBoosting": HistGradientBoostingPipeline,
}


def get_pipeline_class(model_name: str) -> type[BaseModelPipeline]:
    """Look up the pipeline class registered under ``model_name``."""
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ModelNotRegisteredError(
            f"Unknown model '{model_name}'. Available models: {available}."
        ) from exc
