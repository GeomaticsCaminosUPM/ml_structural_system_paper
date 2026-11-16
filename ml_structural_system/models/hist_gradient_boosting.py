"""HistGradientBoosting pipeline. See ``models/base.py`` for shared CV/final/test logic.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier

from .base import BaseModelPipeline


class HistGradientBoostingPipeline(BaseModelPipeline):
    """sklearn's LightGBM-equivalent: histogram-based gradient boosting."""

    model_type = "HistGradientBoosting"

    # early_stopping=True in build_model() sets the number of iterations.
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6, None],
        "min_samples_leaf": [10, 30],
    }

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted HistGradientBoostingClassifier estimator."""
        return HistGradientBoostingClassifier(
            max_iter=1000,
            early_stopping=True,
            random_state=self.cfg.random_state,
            class_weight=class_weight,
            **params,
        )
