"""CatBoost pipeline. See ``models/base.py`` for shared CV/final/test logic."""

from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

from ..gpu import catboost_task_type
from ..progress import bar_disabled
from .base import BaseModelPipeline


class _TqdmCatBoostCallback:
    """tqdm progress bar for CatBoost iterations (passed via fit(callbacks=[...]))."""

    def __init__(self, n_iterations: int) -> None:
        """Initialize the callback."""
        self._n_iterations = n_iterations
        self._pbar: tqdm | None = None

    def after_iteration(self, info: Any) -> bool:
        """Advance the progress bar after each CatBoost iteration."""
        if self._pbar is None:
            self._pbar = tqdm(
                total=self._n_iterations,
                desc="  catboost",
                leave=False,
                unit="iter",
                dynamic_ncols=True,
                disable=bar_disabled(),
            )
        self._pbar.update(1)
        return True  # True = continue training

    def __del__(self) -> None:
        """Close the progress bar on garbage collection, if still open."""
        if self._pbar is not None:
            self._pbar.close()


class CatBoostPipeline(BaseModelPipeline):
    model_type = "CatBoost"

    param_grid = {
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [1, 10],
    }

    def _class_weights_list(self, class_weight: dict | None) -> list[float] | None:
        """Convert a ``{class_code: weight}`` dict (this codebase's shared convention, see
        ``BaseModelPipeline.class_weights``/``cost_matrix. class_weights``) into the
        ordered list CatBoost's ``class_weights`` constructor kwarg actually needs.
        """
        if class_weight is None:
            return None
        n_classes = len(self.cfg.class_names) if self.cfg.class_names else max(class_weight) + 1
        return [class_weight.get(i, 1.0) for i in range(n_classes)]

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted CatBoostClassifier estimator for cross-validation."""
        return CatBoostClassifier(
            iterations=1000,
            early_stopping_rounds=50,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_state=self.cfg.random_state,
            class_weights=self._class_weights_list(class_weight),
            verbose=False,
            task_type=catboost_task_type(),
            **params,
        )

    def fit_params(self, x_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Build extra keyword arguments passed to ``fit()`` during CV."""
        params: dict[str, Any] = {"eval_set": (x_val, y_val), "verbose": False}
        if catboost_task_type() != "GPU":
            params["callbacks"] = [_TqdmCatBoostCallback(1000)]
        return params

    def build_final_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build the final CatBoostClassifier for the full-data refit."""
        n = getattr(self, "_final_n_estimators", None) or 1000
        return CatBoostClassifier(
            iterations=n,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_state=self.cfg.random_state,
            class_weights=self._class_weights_list(class_weight),
            verbose=False,
            task_type=catboost_task_type(),
            **params,
        )
