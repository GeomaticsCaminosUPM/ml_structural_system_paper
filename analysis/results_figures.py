"""Results figures: confusion matrices of each city's best model (Figure 4),
per-city learning curve (Figure 5) and consensus feature ranking
(Figure 7). Style comes from paper_style.py and figures are drawn at the
paper's real text width.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from paper_style import ACCENT, BLUES, INK, MUTED, SEQUENTIAL_CMAP, TEXT_WIDTH_IN, apply_style, save_figure

GROUP_DIR = Path(__file__).resolve().parent.parent / "experiments"
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

apply_style()

FEATURE_NAMES = {
    "height": "Height",
    "year": "Construction era",
    "code_quality": "Code level",
    "roof_material_metallic": "Roof: metallic",
    "roof_material_bright_concrete": "Roof: bright concrete",
    "roof_material_dark_shadow": "Roof: dark or shadow",
    "roof_material_asphalt": "Roof: asphalt",
    "EC8_eccentricityRatio": "Eccentricity ratio (Eurocode 8)",
    "EC8_radiusRatio": "Radius ratio (Eurocode 8)",
    "EC8_compactness": "Compactness (Eurocode 8)",
    "ASCE7_setbackRatio": "Setback ratio (ASCE 7)",
    "ASCE7_holeRatio": "Hole ratio (ASCE 7)",
    "ASCE7_parallelityAngle": "Parallelity angle (ASCE 7)",
    "GNDTII_beta1_mainShapeSlenderness": "Slenderness (GNDT-II)",
    "GNDTII_beta2_setbackRatio": "Setback ratio (GNDT-II)",
    "GNDTII_beta4_eccentricityRatio": "Eccentricity (GNDT-II)",
    "GNDTII_beta6_setbackSlenderness": "Setback slenderness (GNDT-II)",
    "CSCR2010_eccentricityRatio": "Eccentricity (CSCR-2010)",
    "NTC23_setbackRatio": "Setback ratio (NTC-23)",
    "NTC23_holeRatio": "Hole ratio (NTC-23)",
    "polsby_popper": "Polsby-Popper compactness",
    "convex_hull_irregularity": "Convex-hull irregularity",
    "inertia_circle_ratio": "Inertia-to-circle ratio",
    "slenderness_bbox": "Slenderness (bounding box)",
    "slenderness_inertia": "Slenderness (inertia)",
    "contact_force": "Contact force",
    "contact_confinementRatio": "Contact confinement",
    "contact_angularAcc": "Contact angular accel.",
    "contact_angle": "Contact angle",
    "relativePosition_corner": "Position: corner",
    "relativePosition_confined": "Position: confined",
    "relativePosition_isolated": "Position: isolated",
    "relativePosition_lateral": "Position: lateral",
    "relativePosition_torque": "Position: torque",
}


def save(fig, name: str) -> None:
    save_figure(fig, OUT, name)


# Each city's best model (highest test macro F1), from results_analysis.py.
BEST_MODEL = {city: entry["ranking"][0]["model"] for city, entry in json.loads(
    (Path(__file__).resolve().parent.parent / "results" / "results_analysis.json").read_text())["base_cities"].items()}
CITY_EXP = {"Guatemala": "guatemala", "San Jose": "san_jose", "Santo Domingo": "santo_domingo"}
CITY_TITLE = {"Guatemala": "Guatemala City", "San Jose": "San José", "Santo Domingo": "Santo Domingo"}


def _round_row_to_100(pct_row: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding: rounds each value to an integer while
    keeping the row sum exactly 100, instead of rounding each cell
    independently (which can drift to 99 or 101, e.g. three cells at
    23.3/3.3/73.3 all round down to 23+3+73=99)."""
    floors = np.floor(pct_row).astype(int)
    remainder = 100 - floors.sum()
    order = np.argsort(-(pct_row - floors))
    floors[order[:remainder]] += 1
    return floors


def fig_confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN * 1.17, 1.27))
    for ax, city in zip(axes, ["Guatemala", "San Jose", "Santo Domingo"]):
        exp = CITY_EXP[city]
        model = BEST_MODEL[city]
        df = pd.read_csv(GROUP_DIR / exp / "report" / "per_sample_5models.csv")
        with open(GROUP_DIR / exp / "config.yaml") as f:
            import yaml
            cfg = yaml.safe_load(f)
        enc = cfg.get("cost_matrix", {}).get("label_encoding")
        if not enc:
            prep = json.load(open(GROUP_DIR / exp / "preprocessed_splits" / "preprocessing.json"))
            enc = prep["label_mapping"]
        code_to_name = {v: k for k, v in enc.items()}
        labels = sorted(code_to_name)
        names = [code_to_name[c] for c in labels]

        y_true = df["y_true"].to_numpy()
        y_pred = df[f"pred_{model}"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        cm_pct_int = np.stack([_round_row_to_100(row) for row in cm_pct])

        ax.imshow(cm_pct, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=100, aspect="auto")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm_pct_int[i, j]}%\n({cm[i, j]})", ha="center", va="center",
                        fontsize=6.3, linespacing=1.15,
                        color=INK if cm_pct[i, j] < 60 else "white")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Predicted")
        if city == "Guatemala":
            ax.set_ylabel("Observed")
        ax.text(0, 1.13, CITY_TITLE[city], transform=ax.transAxes, fontsize=7.5, fontweight="bold")
        ax.text(0, 1.03, model.replace("HistGradientBoosting", "Hist. GB"), transform=ax.transAxes,
                fontsize=6.6, color=MUTED)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig4_confusion_matrices")


def fig_learning_curve():
    df = pd.read_csv(GROUP_DIR / "learning_curve_summary.csv")
    # learning_curve_summary.csv only aggregates repeats scored on the
    # repeat's full outer test class set (see experiments/learning_curve.py's
    # main(): a class with < 20 training buildings is dropped from both
    # training and test, leaving an easier, reduced-class test). A handful
    # of points end up backed by too few surviving repeats to trust as a
    # "5 repeated splits" mean (every city's target_size=50 loses all or
    # most repeats, san_jose's target_size=100 keeps 2 of 5). Drop any
    # point with fewer than 3 valid repeats, uniformly across cities,
    # rather than plotting it as if it were solid.
    df = df[df["count"] >= 3]
    # XGBoost is used uniformly across all three cities so the curves are
    # directly comparable; it is NOT each city's top-ranked model (Table
    # tab:basecities has Guatemala's best as HistGradientBoosting, which
    # has no learning-curve run in learning_curve_summary.csv to plot
    # instead, labeling this panel "best" would misstate that table).
    model = "XGBoost"
    titles = {"guatemala": "Guatemala City", "san_jose": "San José", "santo_domingo": "Santo Domingo"}
    # Explicit per-city ticks rather than an automatic log-decade locator:
    # Santo Domingo's whole range (50-785) sits inside a single decade, so a
    # decade-only locator (10^2, 10^3, ...) shows just one tick and makes
    # the axis look like it stops instead of spanning the panel; each
    # city's own list below always includes a tick near its max so the
    # data's real extent is legible.
    ticks_by_city = {
        "guatemala": [100, 300, 1000],
        "san_jose": [100, 300, 1000, 3000],
        "santo_domingo": [100, 300, 700],
    }

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN * 1.17, 1.45), sharey=True)
    for ax, city in zip(axes, ["guatemala", "san_jose", "santo_domingo"]):
        sub = df[(df.city == city) & (df.model == model)].sort_values("target_size")
        x = sub["target_size"].to_numpy()
        y = sub["mean"].to_numpy()
        s = sub["std"].fillna(0).to_numpy()
        ax.fill_between(x, y - s, y + s, color=BLUES[2], alpha=0.6, linewidth=0, zorder=2)
        ax.plot(x, y, color=ACCENT, linewidth=1.1, marker="o", markersize=2.2, zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ticks = ticks_by_city[city]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks])
        ax.set_xlim(80, x.max() * 1.3)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(titles[city], fontsize=7.5, fontweight="bold", loc="left")
        ax.set_xlabel("Training buildings")
        ax.grid(axis="y")
    axes[0].set_ylabel("Test macro F1")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig5_learning_curve")


def fig_feature_importance():
    with open(Path(__file__).resolve().parent.parent / "results" / "feature_importance.json") as f:
        fi = json.load(f)
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH_IN * 0.62, 2.7), sharex=True)
    for ax, city in zip(axes, ["Guatemala", "San Jose", "Santo Domingo"]):
        items = list(fi[city].items())[::-1]
        names = [FEATURE_NAMES.get(k, k) for k, _ in items]
        vals = [v for _, v in items]
        ax.barh(names, vals, color=[ACCENT if n == "Height" else BLUES[3] for n in names], height=0.7)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(CITY_TITLE[city], fontsize=7.5, fontweight="bold", loc="left")
        ax.grid(axis="x")
    axes[-1].set_xlabel("Mean rank across models (lower = more important)")
    fig.subplots_adjust(hspace=0.45)
    save(fig, "fig7_feature_importance")


if __name__ == "__main__":
    fig_confusion_matrices()
    fig_learning_curve()
    fig_feature_importance()
