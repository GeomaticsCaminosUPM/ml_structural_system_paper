#!/usr/bin/env bash
# Learning curve of XGBoost in each city (Section 5.2). Reuses the
# hyperparameters selected in run_experiments.sh, so run that first.
set -euo pipefail
cd "$(dirname "$0")/../experiments"
python learning_curve.py
