"""XGBoost pipeline. See ``models/base.py`` for shared CV/final/test logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
from xgboost.callback import TrainingCallback

from ..gpu import XGBClassifierCPUPredict, xgboost_device
from ..progress import bar_disabled
from .base import BaseModelPipeline


class _TqdmXGBCallback(TrainingCallback):
    """tqdm progress bar for XGBoost rounds (auto-closes on early stopping)."""

    def __init__(self, n_estimators: int) -> None:
        """Initialize the callback."""
        super().__init__()
        self._n_estimators = n_estimators
        self._pbar: tqdm | None = None

    def before_training(self, model: Any) -> Any:
        """Open the progress bar before training starts."""
        self._pbar = tqdm(
            total=self._n_estimators,
            desc="  xgb",
            leave=False,
            unit="round",
            dynamic_ncols=True,
            disable=bar_disabled(),
        )
        return model

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        """Advance the progress bar after each boosting round."""
        if self._pbar is not None:
            self._pbar.update(1)
        return False  # False = do not stop early (XGBoost handles that itself)

    def after_training(self, model: Any) -> Any:
        """Close the progress bar once training completes."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        return model


class XGBoostPipeline(BaseModelPipeline):
    model_type = "XGBoost"

    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 5],  # higher = fewer splits on rare classes (regularization)
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    def _objective_kwargs(self) -> dict[str, Any]:
        """Choose the XGBoost objective/eval_metric kwargs for this dataset."""
        if (self.cfg.num_classes or 0) > 2:
            return {
                "objective": "multi:softprob",
                "num_class": self.cfg.num_classes,
                "eval_metric": "mlogloss",
            }
        return {"objective": "binary:logistic", "eval_metric": "logloss"}

    def build_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build an unfitted XGBClassifier estimator for cross-validation."""
        return XGBClassifierCPUPredict(
            **self._objective_kwargs(),
            n_estimators=1000,
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=self.cfg.random_state,
            device=xgboost_device(),
            callbacks=[_TqdmXGBCallback(1000)],
            **params,
        )

    def fit_params(self, x_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Build extra keyword arguments passed to ``fit()`` during CV."""
        return {"eval_set": [(x_val, y_val)], "verbose": False}

    def build_final_model(
        self, params: dict[str, Any], class_weight: dict | None = None
    ) -> BaseEstimator:
        """Build the final XGBClassifier for the full-data refit."""
        n = getattr(self, "_final_n_estimators", None) or 1000
        return XGBClassifierCPUPredict(
            **self._objective_kwargs(),
            n_estimators=n,
            n_jobs=-1,
            random_state=self.cfg.random_state,
            device=xgboost_device(),
            callbacks=[_TqdmXGBCallback(n)],
            **params,
        )

    def sample_weight_for(self, y_train: pd.Series, class_weight: dict | None = None) -> np.ndarray | None:
        """Compute per-sample weights from class weights for the "class_weight" strategy.
        """
        weights = class_weight if class_weight is not None else self.class_weights(y_train)
        return y_train.map(weights).to_numpy()
