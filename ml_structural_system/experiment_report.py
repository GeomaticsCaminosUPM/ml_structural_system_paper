"""Train, test and assess every model of one experiment and write its report files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import assess, test, train
from .config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentReport:
    train_results: list
    test_results: list
    assess_result: "assess.AssessmentResult | None"
    output_dir: Path


def run(cfg: ExperimentConfig, output_dir: str | Path, *, final_only: bool = False) -> ExperimentReport:
    """Train, test, and assess every model in ``cfg``, writing report files to
    ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_results = train.run(cfg, final_only=final_only)
    test_results = test.run(cfg)

    metrics_rows = []
    for result in test_results:
        if result.status != "completed":
            continue
        row = {
            "model": result.model_name,
            "macro_f1": result.metrics.get("macro_f1"),
            "accuracy": result.metrics.get("accuracy"),
            "cohen_kappa": result.metrics.get("cohen_kappa"),
            "tss_macro": result.metrics.get("tss_macro"),
        }
        row.update({k: v for k, v in result.metrics.items() if k.startswith("f1_class_")})
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "test_metrics.csv", index=False)

    assess_result: assess.AssessmentResult | None = None
    gen_rows: list[dict] = []
    consensus: pd.Series | None = None
    try:
        x_test, y_test = cfg.load_data(split="test")
        assess_result = assess.run(cfg, x_test, y_test, explain=True, log_to_mlflow=True)

        gen_rows = [
            {
                "model": g.model_name,
                "cv_mean_f1": g.cv_mean_f1,
                "cv_std_f1": g.cv_std_f1,
                "test_f1": g.test_f1,
                "test_f1_ci_lower": g.test_f1_ci.lower if g.test_f1_ci else None,
                "test_f1_ci_upper": g.test_f1_ci.upper if g.test_f1_ci else None,
                "gap": g.generalizability_gap,
                "is_low_generalizability": g.is_low_generalizability,
            }
            for g in assess_result.generalizability
        ]
        pd.DataFrame(gen_rows).to_csv(output_dir / "generalizability.csv", index=False)

        if assess_result.explainability is not None:
            consensus = assess_result.explainability.consensus_ranking.rename("mean_rank")
            consensus.to_csv(output_dir / "feature_importance.csv")
    except Exception:  # noqa: BLE001
        logger.warning("Assessment / explainability step failed.", exc_info=True)

    _write_summary(output_dir / "summary.txt", cfg, train_results, test_results, assess_result)

    return ExperimentReport(
        train_results=train_results,
        test_results=test_results,
        assess_result=assess_result,
        output_dir=output_dir,
    )


def _write_summary(
    path: Path,
    cfg: ExperimentConfig,
    train_results: list,
    test_results: list,
    assess_result: "assess.AssessmentResult | None",
) -> None:
    """Write the human-readable ``summary.txt`` report for one experiment run."""
    lines = [f"Experiment config: {cfg.config_dir}", ""]

    failed = [r for r in train_results if r.status == "failed"]
    if failed:
        lines.append(f"{len(failed)} model(s) failed during training: {[r.model_name for r in failed]}")
        lines.append("")

    lines.append("=== Test Summary ===")
    for result in test_results:
        if result.status == "completed":
            macro = result.metrics.get("macro_f1", float("nan"))
            acc = result.metrics.get("accuracy", float("nan"))
            kappa = result.metrics.get("cohen_kappa", float("nan"))
            tss = result.metrics.get("tss_macro", float("nan"))
            lines.append(
                f"{result.model_name:<20} macro_f1={macro:.4f}  acc={acc:.4f}  "
                f"kappa={kappa:.4f}  tss={tss:.4f}"
            )
        else:
            lines.append(f"{result.model_name:<20} {result.status}")

    if assess_result is not None:
        ga = assess_result.global_agreement
        lines.append("")
        lines.append("=== Assessment Summary ===")
        kappa_ci = (
            f"  95% CI=[{ga.fleiss_kappa_ci.lower:.4f}, {ga.fleiss_kappa_ci.upper:.4f}]"
            if ga.fleiss_kappa_ci else ""
        )
        lines.append(
            f"Inter-model agreement  Fleiss' Kappa={ga.fleiss_kappa:.4f}{kappa_ci}  "
            f"contested={ga.pct_contested:.1f}%"
        )
        if assess_result.soft_ensemble.f1 is not None:
            lines.append(
                f"Soft ensemble          macro_f1={assess_result.soft_ensemble.f1:.4f}  "
                f"({len(assess_result.soft_ensemble.models_included)} models)"
            )
        if assess_result.explainability is not None:
            lines.append("")
            lines.append("=== Feature Importance (consensus across models) ===")
            for rank_pos, (feat, rank_val) in enumerate(
                assess_result.explainability.consensus_ranking.head(15).items(), start=1
            ):
                lines.append(f"  {rank_pos:2d}. {feat:<35}  mean rank={rank_val:.1f}")

    path.write_text("\n".join(lines) + "\n")
