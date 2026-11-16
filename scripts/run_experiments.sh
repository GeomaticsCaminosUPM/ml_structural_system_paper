#!/usr/bin/env bash
# Train and evaluate the five models in every experiment, then write the
# per-building predictions that the analysis scripts read.
# Re-running is safe: models that are already trained are skipped.
set -euo pipefail
cd "$(dirname "$0")/../experiments"

EXPERIMENTS="guatemala san_jose santo_domingo all_cities
loo_guatemala loo_san_jose loo_santo_domingo
zona1_to_zona10 mata_redonda_to_esquivel ensanche_quisquella_to_naco
san_jose_no_roof_code_year santo_domingo_no_roof_code_year"
MODELS="LogisticRegression RandomForest XGBoost CatBoost HistGradientBoosting"

for exp in $EXPERIMENTS; do
  echo "=== $exp"
  (
    cd "$exp"
    if [ ! -f preprocessed_splits/train_preprocessed.csv ]; then
      mlss split --config config.yaml --output-dir splits
      mlss preprocess --config config.yaml --input-dir splits --output-dir preprocessed_splits
    fi
    mlss report --config config.yaml --output-dir report
    mlss assess --config config.yaml --data preprocessed_splits/test_preprocessed.csv \
      --models $MODELS --output report/per_sample_5models.csv \
      --gen-output report/generalizability_5models.csv --no-mlflow
  )
done

# The Guatemala City ablation reuses the split of guatemala/, so it runs last.
echo "=== guatemala_no_roof_code_year"
python guatemala_ablation.py
