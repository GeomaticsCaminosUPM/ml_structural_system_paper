"""Training orchestrator: runs the CV and final-fit stages of every enabled model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import ExperimentConfig, propagate_preprocessing_metadata
from .mlflow_utils import get_best_run
from .models import get_pipeline_class

logger = logging.getLogger(__name__)


@dataclass
class ModelRunResult:
    model_name: str
    status: str  # "completed" | "failed" | "skipped"
    error: str | None = None


def _model_work_dir(cfg: ExperimentConfig, model_name: str) -> Path:
    """Return the work directory a model's artifacts live in."""
    return cfg.models_dir / model_name


def _already_completed(work_dir: Path, model_name: str) -> bool:
    """Check whether a model's final refit has already been produced."""
    model_file = work_dir / f"model_final_{model_name}.pkl"
    return model_file.exists()


def run(cfg: ExperimentConfig, final_only: bool = False) -> list[ModelRunResult]:
    """Run cross-validation and final refit for every model listed in
    ``cfg.enabled_models``.
    """
    results: list[ModelRunResult] = []

    models_to_run = cfg.enabled_models
    if cfg.start_from_model and cfg.start_from_model in models_to_run:
        start_idx = models_to_run.index(cfg.start_from_model)
        models_to_run = models_to_run[start_idx:]
        logger.info("Starting from model: %s", cfg.start_from_model)

    for model_name in models_to_run:
        work_dir = _model_work_dir(cfg, model_name)
        propagate_preprocessing_metadata(cfg, work_dir)

        if not final_only and cfg.skip_completed and _already_completed(work_dir, model_name):
            logger.info("Skipping %s: already completed.", model_name)
            results.append(ModelRunResult(model_name, status="skipped"))
            continue

        logger.info("%s model: %s", "Refitting (final only)" if final_only else "Training", model_name)
        try:
            pipeline_cls = get_pipeline_class(model_name)
            pipeline = pipeline_cls(cfg, work_dir)
            if not final_only:
                pipeline.run_cv()
            pipeline.run_final()
            results.append(ModelRunResult(model_name, status="completed"))
            logger.info("Completed %s.", model_name)
        except Exception as exc:  # noqa: BLE001 - one model failing must not stop the rest
            logger.exception("Model %s failed.", model_name)
            results.append(ModelRunResult(model_name, status="failed", error=str(exc)))

    _log_strategy_summary(cfg, results)
    return results


def _log_strategy_summary(cfg: ExperimentConfig, results: list[ModelRunResult]) -> None:
    """Log which CV strategy won for each completed model."""
    completed = [r for r in results if r.status == "completed"]
    if not completed:
        return
    logger.info("=== CV Strategy Summary ===")
    for result in completed:
        try:
            best = get_best_run(cfg.experiment_cv, result.model_name)
            strategy = best.get("params.strategy", "?")
            cv_f1 = best.get("metrics.mean_f1_macro", float("nan"))
            logger.info(
                "  %-20s best strategy: %-15s CV macro_f1=%.4f",
                result.model_name, strategy, cv_f1,
            )
        except Exception:  # noqa: BLE001
            logger.info("  %-20s (could not retrieve best strategy from MLflow)", result.model_name)
