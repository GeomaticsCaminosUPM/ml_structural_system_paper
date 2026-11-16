#!/usr/bin/env bash
# Everything, in order. Expect several hours on a recent CPU.
set -euo pipefail
cd "$(dirname "$0")"
./run_experiments.sh
./run_learning_curve.sh
./compute_results.sh
./make_figures.sh
