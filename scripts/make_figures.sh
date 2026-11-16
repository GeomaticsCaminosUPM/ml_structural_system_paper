#!/usr/bin/env bash
# Figures of the paper, written to figures/ as PDF, SVG and PNG.
# Figures 1-3 only need the data. Figures 4-7 need run_experiments.sh,
# run_learning_curve.sh and compute_results.sh.
set -euo pipefail
cd "$(dirname "$0")/../analysis"
python method_figures.py        # Figures 1-3
if [ -d ../results ]; then
  python results_figures.py     # Figures 4, 5 and 7
  python transfer_loss_figure.py  # Figure 6
else
  echo "results/ not found: skipping Figures 4-7 (run compute_results.sh first)"
fi
