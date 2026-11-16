"""Shared MLflow helpers."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd


def safe_cast_param(value: Any) -> Any:
    """Cast an MLflow param string back to int/float/None when possible, else leave as-is.
    """
    if not isinstance(value, str):
        return value
    if value == "None":
        return None
    try:
        if "." not in value and value.lstrip("-").isdigit():
            return int(value)
        return float(value)
    except ValueError:
        return value


def cast_params(params: dict[str, Any]) -> dict[str, Any]:
    """Apply :func:`safe_cast_param` to every value in a dict."""
    return {key: safe_cast_param(value) for key, value in params.items()}


def get_best_run(experiment_name: str, model_type: str, metric: str = "mean_f1_macro") -> pd.Series:
    """Return the best finished CV run for ``model_type`` in ``experiment_name``, ranked by
    ``metric`` (descending).
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"params.model_type = '{model_type}' and attributes.status = 'FINISHED'",
        output_format="pandas",
    )
    if len(runs) == 0:
        raise RuntimeError(
            f"No completed '{model_type}' runs found in experiment '{experiment_name}'. "
            "Check that run_cv() ran successfully for this model."
        )

    metric_col = next((c for c in runs.columns if c.endswith(metric)), None)
    if metric_col is None:
        raise RuntimeError(
            f"No '{metric}' metric found among runs for '{model_type}'. "
            "Check the metric name logged in run_cv()."
        )

    return runs.sort_values(metric_col, ascending=False).iloc[0]


def extract_params(run: pd.Series, allowed_keys: set[str]) -> dict[str, Any]:
    """Pull ``params.*`` columns off an MLflow run and keep only allowed ones."""
    all_params = {
        key.replace("params.", ""): value
        for key, value in run.items()
        if key.startswith("params.")
    }
    return {key: value for key, value in all_params.items() if key in allowed_keys and value is not None}
