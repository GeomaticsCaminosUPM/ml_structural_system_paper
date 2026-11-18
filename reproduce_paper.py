"""Reproduce every table and figure of the paper, training every model from scratch.

    python reproduce_paper.py                                # everything (several hours)
    python reproduce_paper.py figures tables                 # only some steps
    python reproduce_paper.py experiments --experiments santo_domingo
    python reproduce_paper.py --resume                       # continue an interrupted run

Steps, in order:

    experiments     split the data, then train and test the five models in each experiment
    learning_curve  learning curve of XGBoost in each city (Section 5.2)
    results         every number in the tables and text -> results/*.json
    figures         Figures 1-7 -> figures/*.jpg
    tables          Tables 1-4 -> results/tables/*.tex
    map             data of the interactive map -> docs/map/data/

The experiments and learning_curve steps first delete what an earlier run left
(splits, trained models, reports), so models are always trained from scratch.
With --resume, existing splits and trained models are kept and reused instead.
XGBoost and CatBoost use the GPU when one is available.

Trained models, test splits, reports, results and figures are tracked by git
(see .gitignore). At the end of every run, outputs over 30 MB are listed in
.gitignore so they are never committed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
ANALYSIS = ROOT / "analysis"
MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "HistGradientBoosting"]
MLSS = str(Path(sys.executable).parent / "mlss")
ALL_EXPERIMENTS: list[str] = yaml.safe_load((EXPERIMENTS / "main.yaml").read_text())["experiments"]
# Reuses the split of guatemala/, so it is run by its own script after the others.
ABLATION = "guatemala_no_roof_code_year"
MAX_TRACKED_MB = 30
GITIGNORE_START = "# >>> Outputs over 30 MB, listed by reproduce_paper.py after each run (do not edit)"
GITIGNORE_END = "# <<<"
# Everything an experiment writes. Deleted before a from-scratch run.
OUTPUTS = ["splits", "preprocessed_splits", "trained_models", "report", "mlflow.db", "mlflow.db-journal",
           "mlruns", "catboost_info"]


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def python(script: str, cwd: Path) -> None:
    run([sys.executable, script], cwd)


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def experiments(names: list[str], resume: bool) -> None:
    if not resume:
        for name in names:
            for out in OUTPUTS:
                remove(EXPERIMENTS / name / out)
        remove(EXPERIMENTS / "catboost_info")
    for name in names:
        if name == ABLATION:
            continue
        print(f"=== {name}", flush=True)
        exp = EXPERIMENTS / name
        if not (exp / "preprocessed_splits" / "train_preprocessed.csv").exists():
            run([MLSS, "split", "--config", "config.yaml", "--output-dir", "splits"], exp)
            run([MLSS, "preprocess", "--config", "config.yaml", "--input-dir", "splits",
                 "--output-dir", "preprocessed_splits"], exp)
        run([MLSS, "report", "--config", "config.yaml", "--output-dir", "report"], exp)
        run([MLSS, "assess", "--config", "config.yaml", "--data", "preprocessed_splits/test_preprocessed.csv",
             "--models", *MODELS, "--output", "report/per_sample_5models.csv",
             "--gen-output", "report/generalizability_5models.csv", "--no-mlflow"], exp)
    if ABLATION in names:
        print(f"=== {ABLATION}", flush=True)
        if not (EXPERIMENTS / "guatemala" / "splits").exists():
            sys.exit(f"{ABLATION} reuses the split of guatemala: run the guatemala experiment too.")
        python("guatemala_ablation.py", EXPERIMENTS)


def learning_curve(resume: bool) -> None:
    if not resume:
        remove(EXPERIMENTS / "learning_curve")
        for csv in EXPERIMENTS.glob("learning_curve_*.csv"):
            csv.unlink()
    python("learning_curve.py", EXPERIMENTS)


def results() -> None:
    for script in ["data_stats.py",                # Table 1, survey overlaps (Section 3.1)
                   "results_analysis.py",          # Table 2, shared errors, Cohen's d (Section 5.1)
                   "best_vs_second_bootstrap.py",  # best vs second model per city (Section 5.1)
                   "learning_curve_stats.py",      # learning-curve numbers (Section 5.2)
                   "present_class_metrics.py",     # Tables 3 and 4, Delta F1 (Sections 5.3, 5.4)
                   "block_bootstrap_ci.py",        # 95% confidence intervals of Tables 2-4
                   "disagreement_check.py",        # model disagreement (Section 5.3)
                   "feature_importance.py",        # feature ranking (Section 5.5)
                   "ablation_bootstrap.py"]:       # attribute ablation (Section 5.5)
        python(script, ANALYSIS)


def figures() -> None:
    python("method_figures.py", ANALYSIS)        # Figures 1-3 (data only)
    if not (ROOT / "results" / "present_class_metrics.json").exists():
        print("Figures 4-7 skipped: run the experiments, learning_curve and results steps first.")
        return
    python("results_figures.py", ANALYSIS)       # Figures 4, 5 and 7
    python("transfer_loss_figure.py", ANALYSIS)  # Figure 6


def tables() -> None:
    python("data_stats.py", ANALYSIS)  # Table 1 only needs the data
    python("make_tables.py", ANALYSIS)


def ignore_large_outputs() -> None:
    """List every output over MAX_TRACKED_MB in .gitignore's managed block, so git never tracks it."""
    large = sorted(
        "/" + path.relative_to(ROOT).as_posix()
        for folder in (EXPERIMENTS, ROOT / "results", ROOT / "figures") if folder.exists()
        for path in folder.rglob("*")
        if path.is_file() and path.stat().st_size > MAX_TRACKED_MB * 1024**2
    )
    gitignore = ROOT / ".gitignore"
    text = gitignore.read_text()
    start, end = text.index(GITIGNORE_START), text.index(GITIGNORE_END, text.index(GITIGNORE_START))
    block = "\n".join([GITIGNORE_START, *large, ""])
    gitignore.write_text(text[:start] + block + text[end:])
    for path in large:
        print(f"Not tracked by git (over {MAX_TRACKED_MB} MB): {path}")


def map_data() -> None:
    python("map_data.py", ANALYSIS)  # buildings, predictions and scores shown by docs/map/


STEPS = ["experiments", "learning_curve", "results", "figures", "tables", "map"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", metavar="step",
                        help=f"steps to run, in order (default: all). Any of {', '.join(STEPS)}")
    parser.add_argument("--experiments", nargs="+", choices=ALL_EXPERIMENTS, default=ALL_EXPERIMENTS,
                        metavar="name", help="experiments to run in the experiments step (default: all 13)")
    parser.add_argument("--resume", action="store_true",
                        help="keep existing splits and trained models instead of starting from scratch")
    args = parser.parse_args()
    unknown = [s for s in args.steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s) {unknown}. Choose from {STEPS}")
    args.steps = args.steps or STEPS
    for step in [s for s in STEPS if s in args.steps]:
        if step == "experiments":
            experiments([e for e in ALL_EXPERIMENTS if e in args.experiments], args.resume)
        elif step == "learning_curve":
            learning_curve(args.resume)
        else:
            {"results": results, "figures": figures, "tables": tables, "map": map_data}[step]()
    ignore_large_outputs()


if __name__ == "__main__":
    main()
