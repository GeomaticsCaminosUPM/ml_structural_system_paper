"""LogisticRegression pipeline. See ``models/base.py`` for shared CV/final/test logic.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from .base import BaseModelPipeline


class LogisticRegressionPipeline(BaseModelPipeline):
    model_type = "LogisticRegression"

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
    }

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted LogisticRegression estimator."""
        return LogisticRegression(
            max_iter=2000,
            random_state=self.cfg.random_state,
            class_weight=class_weight,
            l1_ratio=0,  # pure L2
            **params,
        )
