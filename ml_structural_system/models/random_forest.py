"""RandomForest pipeline. See ``models/base.py`` for shared CV/final/test logic."""

from __future__ import annotations

from typing import Any

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator

from .base import BaseModelPipeline


class RandomForestPipeline(BaseModelPipeline):
    model_type = "RandomForest"

    # BalancedRandomForest already undersamples per tree, so no SMOTE strategies.
    cv_strategies = {
        "baseline":     {"resampler": None, "use_class_weight": False},
        "class_weight": {"resampler": None, "use_class_weight": True},
        # cost_sensitive added dynamically by __init__ when cost_matrix is set
    }

    param_grid = {
        # n_estimators fixed at 200 during CV (500 for final refit via build_final_model)
        "max_depth": [6, 12, None],
        "min_samples_split": [2, 8],
        "max_features": ["sqrt", 0.5],  # sqrt≈6 features/split vs 50% of 35
    }

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted BalancedRandomForestClassifier for cross-validation."""
        return BalancedRandomForestClassifier(
            n_estimators=200,
            random_state=self.cfg.random_state,
            n_jobs=-1,
            class_weight=class_weight,
            **params,
        )

    def build_final_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build the final BalancedRandomForestClassifier for the full-data refit."""
        return BalancedRandomForestClassifier(
            n_estimators=500,
            random_state=self.cfg.random_state,
            n_jobs=-1,
            class_weight=class_weight,
            **params,
        )
