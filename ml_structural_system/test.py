"""Test orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import ExperimentConfig
from .models import get_pipeline_class

logger = logging.getLogger(__name__)


def _format_metrics(metrics: dict[str, float]) -> str:
    """Format a metrics dict as a one-line human-readable summary."""
    macro = metrics.get("macro_f1", float("nan"))
    acc = metrics.get("accuracy", float("nan"))
    summary = f"macro_f1={macro:.4f}  acc={acc:.4f}"
    per_class = sorted((k, v) for k, v in metrics.items() if k.startswith("f1_class_"))
    if per_class:
        class_str = "  ".join(f"[{k[9:]}]={v:.3f}" for k, v in per_class)
        summary += f"  {class_str}"
    return summary


@dataclass
class ModelTestResult:
    model_name: str
    status: str  # "completed" | "failed" | "skipped"
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def run(cfg: ExperimentConfig) -> list[ModelTestResult]:
    """Evaluate every model in ``cfg.enabled_models`` on the labeled test set."""
    results: list[ModelTestResult] = []

    for model_name in cfg.enabled_models:
        work_dir = cfg.models_dir / model_name
        model_path = work_dir / f"model_final_{model_name}.pkl"

        if not model_path.exists():
            logger.warning("Skipping %s: no final model found at %s.", model_name, model_path)
            results.append(ModelTestResult(model_name, status="skipped"))
            continue

        logger.info("Testing model: %s", model_name)
        try:
            pipeline_cls = get_pipeline_class(model_name)
            pipeline = pipeline_cls(cfg, work_dir)
            metrics = pipeline.run_test()
            results.append(ModelTestResult(model_name, status="completed", metrics=metrics))
            logger.info("Tested %s: %s", model_name, _format_metrics(metrics))
        except Exception as exc:  # noqa: BLE001 - one model failing must not stop the rest
            logger.exception("Testing %s failed.", model_name)
            results.append(ModelTestResult(model_name, status="failed", error=str(exc)))

    return results
