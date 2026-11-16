# Spatial transferability of machine learning classifiers for building structural-system identification

This repository contains the data and code needed to reproduce the results of the paper of the same title. The models predict the structural system of individual buildings (adobe, reinforced concrete, masonry or wood) from the building footprint, its neighbors and a few low-cost attributes such as height and construction era. They are trained on field surveys from Guatemala City, San José (Costa Rica) and Santo Domingo (Dominican Republic) and evaluated within each city, across cities, across districts of the same city, and as a single pooled model.

![Workflow](docs/figures/fig3_workflow.png)

## Contents

```
data/            the six labeled surveys (GeoPackage); data/figures/ holds extra footprints for Figures 1 and 2
experiments/     one folder per experiment with its config.yaml; main.yaml holds the shared settings
ml_structural_system/   the training and evaluation library (installs the `mlss` command)
analysis/        scripts that compute the reported numbers and draw the figures
scripts/         the shell scripts you run, listed below
```

## Data

Each file in `data/` holds the building footprints of one survey area. Buildings that were surveyed carry their structural system in the column `survey_structural_system`; the rest are kept as context, since several features describe a building's neighbors. Every feature used by the models is already computed.

| City | File | Survey | Labeled buildings |
|---|---|---|---|
| Guatemala City | `guatemala/zona10.gpkg` | Zona 10 | 4,442 |
| Guatemala City | `guatemala/zona1.gpkg` | Zona 1 | 238 |
| San José | `san_jose/esquivel.gpkg` | citywide sample | 6,425 |
| San José | `san_jose/mata_redonda.gpkg` | Mata Redonda | 571 |
| Santo Domingo | `santo_domingo/naco.gpkg` | dispersed sample | 211 |
| Santo Domingo | `santo_domingo/ensanche_quisquella.gpkg` | Ensanche Quisqueya | 758 |

The survey codes are grouped into the four classes of the paper through `label_replacements` in `experiments/main.yaml`.

![Study area](docs/figures/fig1_study_area.png)

The gray footprints are surveyed-area buildings without a label. In the Mata Redonda and Ensanche Quisqueya panels they also include the other survey of the same city.

![Model input](docs/figures/fig2_model_input.png)

## Installation

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GeomaticsCaminosUPM/StructuralSystemMLPaperCode.git
cd StructuralSystemMLPaperCode
uv sync
source .venv/bin/activate
```

`pip install -e .` inside a Python 3.12 environment works too. Package versions are pinned to the ones used for the paper, because even minor releases of scikit-learn can shift the results slightly. A GPU is not required.

## Reproducing the paper

Run the scripts from the repository root, in this order:

| Script | What it does | Output |
|---|---|---|
| `scripts/run_experiments.sh` | Trains and tests the five models in each of the 13 experiments | `experiments/*/trained_models/`, `experiments/*/report/` |
| `scripts/run_learning_curve.sh` | Learning curve of XGBoost in each city (Section 5.2) | `experiments/learning_curve_summary.csv` |
| `scripts/compute_results.sh` | Every number in the tables and text | `results/*.json` |
| `scripts/make_figures.sh` | The seven figures of the paper | `figures/` (PDF, SVG, PNG) |

Or run everything at once:

```bash
./scripts/reproduce_all.sh
```

The first two steps take several hours on a laptop CPU. Both can be interrupted and started again: trained models are not retrained. Figures 1 to 3 only depend on the data, so `./scripts/make_figures.sh` draws them straight after installation.

The experiments are:

- `guatemala`, `san_jose`, `santo_domingo`: one model per city (Section 5.1).
- `loo_guatemala`, `loo_san_jose`, `loo_santo_domingo`: trained on two cities, tested on the third (Section 5.3).
- `zona1_to_zona10`, `mata_redonda_to_esquivel`, `ensanche_quisquella_to_naco`: trained on one district, tested on the larger survey of the same city (Section 5.3).
- `all_cities`: one model for the three cities (Section 5.4).
- `guatemala_no_roof_code_year`, `san_jose_no_roof_code_year`, `santo_domingo_no_roof_code_year`: city models without roof material, code level and construction era (Section 5.5).

## Results

Confusion matrices of the best model in each city:

![Confusion matrices](docs/figures/fig4_confusion_matrices.png)

Macro F1 as the number of training buildings grows:

![Learning curve](docs/figures/fig5_learning_curve.png)

Macro F1 lost when a model is used outside the area it was trained on:

![Transfer loss](docs/figures/fig6_transfer_loss.png)

Features that the five models rank highest in each city:

![Feature importance](docs/figures/fig7_feature_importance.png)

## Citation

If you use this code or the data, please cite the paper. The full reference will be added here once it is published.
