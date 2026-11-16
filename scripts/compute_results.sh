#!/usr/bin/env bash
# Numbers reported in the tables and text. Each script writes a JSON file
# to results/.
set -euo pipefail
cd "$(dirname "$0")/../analysis"
python results_analysis.py          # Table 2, shared errors, Cohen's d (Section 5.1)
python best_vs_second_bootstrap.py  # best vs second model per city (Section 5.1)
python present_class_metrics.py     # Tables 3 and 4, Delta F1 (Sections 5.3 and 5.4)
python block_bootstrap_ci.py        # 95% confidence intervals of Tables 2-4
python disagreement_check.py        # model disagreement (Section 5.3)
python feature_importance.py        # feature ranking (Section 5.5)
python ablation_bootstrap.py        # attribute ablation (Section 5.5)
