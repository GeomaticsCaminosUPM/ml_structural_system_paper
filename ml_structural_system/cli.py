"""Command-line entry point (``mlss``): split, preprocess, train, test, report, assess."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import assess as assess_module
from . import dataset_prep as prep_module
from . import experiment_report as report_module
from . import test as test_module
from . import train as train_module
from .config import load_config
from .exceptions import ConfigError, DatasetError, MLStructuralSystemError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("config.yaml")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _train_cmd(args: argparse.Namespace) -> int:
    """Handle the ``mlss train`` subcommand."""
    cfg = load_config(args.config)
    results = train_module.run(cfg, final_only=getattr(args, "final_only", False))
    failed = [r for r in results if r.status == "failed"]
    for r in failed:
        logger.error("Training failed: %s, %s", r.model_name, r.error)
    return 1 if failed else 0


def _report_cmd(args: argparse.Namespace) -> int:
    """Handle the ``mlss report`` subcommand."""
    cfg = load_config(args.config)
    result = report_module.run(cfg, args.output_dir, final_only=args.final_only)
    failed = [r for r in result.train_results if r.status == "failed"]
    print(f"Report written to {result.output_dir}")
    return 1 if failed else 0


def _test_cmd(args: argparse.Namespace) -> int:
    """Handle the ``mlss test`` subcommand."""
    cfg = load_config(args.config)
    results = test_module.run(cfg)
    failed = [r for r in results if r.status == "failed"]
    for r in failed:
        logger.error("Test failed: %s, %s", r.model_name, r.error)
    return 1 if failed else 0


def _assess_cmd(args: argparse.Namespace) -> int:
    """Handle the ``mlss assess`` subcommand."""
    cfg = load_config(args.config)

    data_path = args.data if args.data is not None else cfg.test_path
    df = pd.read_csv(data_path)
    label = cfg.label

    if label in df.columns:
        X = df.drop(columns=[label])
        y_raw = df[label]
        # Encode string labels to integers using the same mapping as cfg.load_data()
        if not pd.api.types.is_integer_dtype(y_raw):
            name_to_code = {
                name: code for code, name in enumerate(cfg.class_names)  # type: ignore[arg-type]
            }
            unknown = set(y_raw.astype(str).unique()) - set(name_to_code)
            if unknown:
                logger.error(
                    "Label(s) %s found in --data but not in the training set. "
                    "Check that the label column uses the same class names.",
                    sorted(unknown),
                )
                return 1
            y: pd.Series | None = y_raw.astype(str).map(name_to_code)
        else:
            y = y_raw
        logger.info("Detected label column '%s', F1 metrics will be computed.", label)
    else:
        X = df
        y = None
        logger.info("No label column '%s', running unlabeled assessment.", label)

    result = assess_module.run(
        cfg,
        X,
        y,
        models=args.models or None,
        contested_threshold=args.contested_threshold,
        gap_threshold=args.gap_threshold,
        log_to_mlflow=not args.no_mlflow,
        explain=args.explain,
    )

    _print_summary(result)

    if args.output:
        result.sample_dataframe.to_csv(args.output, index=False)
        logger.info("Per-sample results written to %s", args.output)

    if args.gen_output:
        rows = [
            {
                "model_name": g.model_name,
                "cv_mean_f1": g.cv_mean_f1,
                "cv_std_f1": g.cv_std_f1,
                "test_f1": g.test_f1,
                "gap": g.generalizability_gap,
                "is_low_generalizability": g.is_low_generalizability,
            }
            for g in result.generalizability
        ]
        pd.DataFrame(rows).to_csv(args.gen_output, index=False)
        logger.info("Generalizability stats written to %s", args.gen_output)

    return 0


def _fmt_ci(ci: "assess_module.BootstrapCI | None") -> str:
    """Format a bootstrap confidence interval for display."""
    if ci is None:
        return ""
    pct = int(ci.confidence * 100)
    return f"[{ci.lower:.4f}, {ci.upper:.4f}] ({pct}% CI)"


def _print_summary(result: assess_module.AssessmentResult) -> None:
    """Print a human-readable summary of an assessment result to stdout."""
    g = result.global_agreement
    kappa_ci_s = f"  {_fmt_ci(g.fleiss_kappa_ci)}" if g.fleiss_kappa_ci else ""
    print("\n=== Agreement Summary ===")
    print(f"  Models evaluated    : {g.n_models}")
    print(f"  Samples             : {g.n_samples}")
    print(f"  Fleiss' Kappa       : {g.fleiss_kappa:.4f}{kappa_ci_s}")
    print(f"  Mean agreement      : {g.mean_agreement_ratio:.4f}")
    print(f"  Mean entropy (norm) : {g.mean_normalized_entropy:.4f}")
    print(f"  Contested samples   : {g.pct_contested:.1f}%")

    se = result.soft_ensemble
    if se.models_included:
        print("\n=== Soft Ensemble (weighted by CV F1) ===")
        if se.f1 is not None:
            f1_ci_s = f"  {_fmt_ci(se.f1_ci)}" if se.f1_ci else ""
            print(f"  F1 (macro)          : {se.f1:.4f}{f1_ci_s}")
        if se.model_weights:
            sorted_w = sorted(se.model_weights.items(), key=lambda x: x[1], reverse=True)
            weights_str = "  ".join(f"{name} {w:.3f}" for name, w in sorted_w)
            print(f"  Weights             : {weights_str}")
        else:
            print(f"  Models included     : {', '.join(se.models_included)}")
        if se.models_excluded:
            print(f"  Models excluded     : {', '.join(se.models_excluded)}")

    if result.generalizability:
        has_cv = any(g.cv_mean_f1 is not None for g in result.generalizability)
        has_test = any(g.test_f1 is not None for g in result.generalizability)
        has_ci = any(g.test_f1_ci is not None for g in result.generalizability)
        if has_cv or has_test:
            print("\n=== Generalizability ===")
            header = f"  {'Model':<25}"
            if has_cv:
                header += f"  {'CV F1':>7}  {'±':>6}"
            if has_test:
                header += f"  {'Test F1':>7}  {'Gap':>7}  Flag"
            if has_ci:
                header += f"  CI"
            print(header)
            for gs in result.generalizability:
                row = f"  {gs.model_name:<25}"
                if has_cv:
                    cv_s = f"{gs.cv_mean_f1:.4f}" if gs.cv_mean_f1 is not None else "      -"
                    std_s = f"{gs.cv_std_f1:.4f}" if gs.cv_std_f1 is not None else "     -"
                    row += f"  {cv_s:>7}  {std_s:>6}"
                if has_test:
                    test_s = f"{gs.test_f1:.4f}" if gs.test_f1 is not None else "      -"
                    gap_s = (
                        f"{gs.generalizability_gap:+.4f}"
                        if gs.generalizability_gap is not None
                        else "      -"
                    )
                    flag = "LOW" if gs.is_low_generalizability else "   "
                    row += f"  {test_s:>7}  {gap_s:>7}  {flag}"
                if has_ci:
                    row += f"  {_fmt_ci(gs.test_f1_ci)}" if gs.test_f1_ci else ""
                print(row)

    if result.cross_model_f1_std is not None:
        print(f"\n  Cross-model F1 std  : {result.cross_model_f1_std:.4f}")

    if result.explainability is not None:
        expl = result.explainability
        print("\n=== Feature Importance, Consensus (mean rank across models) ===")
        for i, (feat, rank_val) in enumerate(expl.consensus_ranking.head(15).items(), 1):
            print(f"  {i:>2}. {feat:<40} mean rank = {rank_val:.1f}")

        print("\n=== Feature Importance, Per model (top 5) ===")
        for m in expl.per_model:
            if m.importances.empty:
                print(f"  {m.model_name:<25}, (not available)")
                continue
            top = list(m.importances.items())[:5]
            top_str = "  ".join(f"{f} ({v:.3f})" for f, v in top)
            method_tag = f"[{m.method}]" if m.method != "built_in" else ""
            print(f"  {m.model_name:<25} {top_str}  {method_tag}")

    print()


# ---------------------------------------------------------------------------
# Dataset preparation: split / preprocess
#
# Both wrap ml_structural_system.dataset_prep.
# ---------------------------------------------------------------------------

def _split_cmd(args: argparse.Namespace) -> int:
    """Handle the ``mlss split`` subcommand."""
    cfg = prep_module.load_preprocessing_config(args.config)
    raw = prep_module.prepare_raw_splits(cfg, test_size=args.test_size)
    print(raw.summary_text())
    paths = prep_module.save_raw_split(raw, args.output_dir)
    written = f"{paths['train']}, {paths['test']}"
    if "val" in paths:
        written += f", {paths['val']}"
    print(f"\nRaw splits saved: {written}")
    return 0


def _preprocess_cmd(args: argparse.Namespace) -> int:
    """Handle the ``mlss preprocess`` subcommand."""
    cfg = prep_module.load_preprocessing_config(args.config)

    train_path = prep_module.find_tabular_file(args.input_dir, "train_df")
    test_path = prep_module.find_tabular_file(args.input_dir, "test_df")
    train_df = prep_module.load_tabular_source(
        str(train_path), cfg.feature_columns, cfg.code_type,
        compute_missing_features=cfg.compute_missing_features,
    )
    test_df = prep_module.load_tabular_source(
        str(test_path), cfg.feature_columns, cfg.code_type,
        compute_missing_features=cfg.compute_missing_features,
    )
    train_df = prep_module.apply_label_replacements(train_df, cfg.label, cfg.label_replacements)
    test_df = prep_module.apply_label_replacements(test_df, cfg.label, cfg.label_replacements)
    train_df, ord_map = prep_module.validate_and_normalize(train_df, cfg)
    test_df, _ = prep_module.validate_and_normalize(test_df, cfg)

    val_df = None
    try:
        val_path = prep_module.find_tabular_file(args.input_dir, "val_df")
    except FileNotFoundError:
        val_path = None
    if val_path is not None:
        val_df = prep_module.load_tabular_source(
            str(val_path), cfg.feature_columns, cfg.code_type,
            compute_missing_features=cfg.compute_missing_features,
        )
        val_df = prep_module.apply_label_replacements(val_df, cfg.label, cfg.label_replacements)
        val_df, _ = prep_module.validate_and_normalize(val_df, cfg)

    result = prep_module.preprocess_split(
        train_df, test_df, cfg, ord_map, scale_numeric=not args.no_scale, val_df=val_df,
    )
    print(result.summary_text())
    paths = prep_module.save_preprocess_result(result, args.output_dir)
    print(f"\nPreprocessed artifacts saved to {args.output_dir}/")
    logger.debug("Paths written: %s", paths)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the ``mlss`` top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="mlss",
        description="ml_structural_system: split, preprocess, train, test and assess tabular classifiers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_p = sub.add_parser("train", help="Cross-validate and refit all configured models.")
    train_p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="Path to config.yaml"
    )
    train_p.add_argument(
        "--final-only",
        action="store_true",
        help="Skip CV and redo only the final refit + threshold calibration, "
        "reusing the best CV runs already logged in MLflow.",
    )
    train_p.set_defaults(func=_train_cmd)

    # ---- test ----
    test_p = sub.add_parser("test", help="Evaluate final models on the labeled test split.")
    test_p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="Path to config.yaml"
    )
    test_p.set_defaults(func=_test_cmd)

    # ---- report ----
    report_p = sub.add_parser(
        "report",
        help="Train + test + assess all configured models and write summary.txt, "
             "test_metrics.csv, generalizability.csv, feature_importance.csv.",
    )
    report_p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="Path to config.yaml"
    )
    report_p.add_argument(
        "--output-dir", type=Path, default=Path("report"),
        help="Directory to write report artifacts to (default: ./report).",
    )
    report_p.add_argument(
        "--final-only", action="store_true",
        help="Skip CV and redo only the final refit + threshold calibration.",
    )
    report_p.set_defaults(func=_report_cmd)

    # ---- assess ----
    assess_p = sub.add_parser(
        "assess",
        help="Multi-model agreement and generalizability analysis.",
    )
    assess_p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="Path to config.yaml"
    )
    assess_p.add_argument(
        "--data", type=Path, default=None,
        help="CSV file with features (and optionally the label column). "
             "Defaults to dataset.test_path from config.yaml.",
    )
    assess_p.add_argument(
        "--output", type=Path, default=None,
        help="Write per-sample results to this CSV.",
    )
    assess_p.add_argument(
        "--gen-output", dest="gen_output", type=Path, default=None,
        help="Write per-model generalizability stats to this CSV.",
    )
    assess_p.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of model names to include (default: all enabled models).",
    )
    assess_p.add_argument(
        "--contested-threshold", dest="contested_threshold", type=float, default=0.5,
        help="Agreement ratio below which a sample is flagged as contested (default: 0.5).",
    )
    assess_p.add_argument(
        "--gap-threshold", dest="gap_threshold", type=float, default=0.05,
        help="F1 drop threshold to flag low generalizability (default: 0.05).",
    )
    assess_p.add_argument(
        "--no-mlflow", dest="no_mlflow", action="store_true",
        help="Do not log assessment results to MLflow.",
    )
    assess_p.add_argument(
        "--explain", action="store_true",
        help="Compute SHAP feature importances and consensus ranking across models.",
    )
    assess_p.set_defaults(func=_assess_cmd)

    #, split -----------------------------------------------------------------
    split_p = sub.add_parser(
        "split",
        help="Split raw labeled data into train/test per config.yaml's 'preprocessing' section.",
    )
    split_p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="Path to config.yaml"
    )
    split_p.add_argument(
        "--output-dir", type=Path, default=Path("splits"),
        help="Where to write the raw train_df/test_df files (default: splits).",
    )
    split_p.add_argument(
        "--test-size", type=float, default=None,
        help="Override preprocessing.data.test_size from config.yaml.",
    )
    split_p.set_defaults(func=_split_cmd)

    #, preprocess --------------------------------------------------------------
    preprocess_p = sub.add_parser(
        "preprocess",
        help="Fit a preprocessor on raw train/test splits and write preprocessed CSVs.",
    )
    preprocess_p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="Path to config.yaml"
    )
    preprocess_p.add_argument(
        "--input-dir", type=Path, default=Path("splits"),
        help="Directory with raw train_df/test_df (.gpkg or .csv) (default: splits).",
    )
    preprocess_p.add_argument(
        "--output-dir", type=Path, default=Path("preprocessed_splits"),
        help="Where to write preprocessed CSVs + preprocessor artifacts "
             "(default: preprocessed_splits).",
    )
    preprocess_p.add_argument(
        "--no-scale", action="store_true",
        help="Skip the final StandardScaler step (numeric features keep their raw scale).",
    )
    preprocess_p.set_defaults(func=_preprocess_cmd)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI arguments, dispatch to the matching subcommand, and handle top-level
    errors.
    """
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except DatasetError as exc:
        logger.error("Data error: %s", exc)
        return 1
    except MLStructuralSystemError as exc:
        logger.error("Error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 1
    except Exception:
        logger.exception("Unexpected error.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
