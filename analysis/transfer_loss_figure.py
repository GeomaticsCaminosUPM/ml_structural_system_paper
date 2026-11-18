"""Figure 6: macro F1 lost when the city-specific model is replaced by a
transferred one (single-district survey -> larger survey of the same city,
or leave-one-city-out).

Every point is the best model of its scenario (as in Tables tab:basecities
and tab:transfer), scored on the classes shared by the two test sets
(reinforced concrete and masonry), so the gap is the Delta F1 of Table
tab:transfer. Error bars: 95% percentile block bootstrap (2,000 resamples of
the test set's spatial blocks), as in block_bootstrap_ci.py.

Output: figures/fig6_transfer_loss.jpg.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from results_analysis import RESULTS_DIR, GROUP_DIR, load_class_names, load_sample_df
from paper_style import ACCENT, BLUES, INK, MUTED, TEXT_WIDTH_IN, apply_style, save_figure

apply_style()
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 2000
SHARED = ["CR", "M"]

CITIES = [("Guatemala", "Guatemala City", "guatemala", "zona1_to_zona10", "loo_guatemala"),
          ("San Jose", "San José", "san_jose", "mata_redonda_to_san_jose_sample", "loo_san_jose"),
          ("Santo Domingo", "Santo Domingo", "santo_domingo", "ensanche_quisquella_to_santo_domingo_center", "loo_santo_domingo")]
SCENARIOS = [("City-specific model", ACCENT, "o", "full"),
             ("Single-district transfer", BLUES[3], "s", "full"),
             ("Leave-one-city-out", ACCENT, "D", "none")]


def shared_f1(exp: str, model: str, rng: np.random.Generator) -> tuple[float, list[float]]:
    df = load_sample_df(exp)
    names = load_class_names(exp)
    codes = [c for c, n in names.items() if n in SHARED]
    y_true, y_pred = df["y_true"].to_numpy(), df[f"pred_{model}"].to_numpy()
    f1 = lambda i: f1_score(y_true[i], y_pred[i], average="macro", labels=codes, zero_division=0)
    blocks = gpd.read_file(GROUP_DIR / exp / "splits" / "test_df.gpkg")["block_id"].to_numpy()
    _, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    boot = [f1(np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]))
            for _ in range(N_BOOT)]
    return float(f1(np.arange(len(y_true)))), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main() -> None:
    rng = np.random.default_rng(42)
    pcm = json.loads((RESULTS_DIR / "present_class_metrics.json").read_text())
    values = {}
    for key, _, base, district, loo in CITIES:
        models = [pcm["base_cities"][key]["best_model"],
                  pcm["transfer"][district]["ranking"][0]["model"],
                  pcm["transfer"][loo]["ranking"][0]["model"]]
        values[key] = [shared_f1(e, m, rng) for e, m in zip([base, district, loo], models)]
        print(key, [(m, round(v, 3), [round(c, 3) for c in ci]) for m, (v, ci) in zip(models, values[key])])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.62, 1.55))
    offsets = [0.22, 0.0, -0.22]
    for row, (key, title, *_) in enumerate(CITIES):
        y0 = len(CITIES) - 1 - row
        f1s = [v for v, _ in values[key]]
        ax.plot([f1s[0], f1s[0]], [y0 - 0.36, y0 + 0.36], color=ACCENT, linewidth=0.6, linestyle=(0, (2, 1.5)), zorder=1)
        for (label, color, marker, fill), off, (v, (lo, hi)) in zip(SCENARIOS, offsets, values[key]):
            if off != offsets[0]:
                ax.plot([v, f1s[0]], [y0 + off, y0 + off], color=BLUES[1], linewidth=3.2, solid_capstyle="butt", zorder=1)
                ax.text(1.005, y0 + off, f"{v - f1s[0]:+.2f}".replace("-", "\u2212"), transform=ax.get_yaxis_transform(),
                        va="center", ha="left", fontsize=6.5, color=INK)
            ax.errorbar(v, y0 + off, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=MUTED,
                        elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=2)
            ax.plot(v, y0 + off, marker=marker, markersize=4.2, color=color, markeredgecolor=color,
                    markeredgewidth=0.9, fillstyle=fill, markerfacecolor=color if fill == "full" else "white",
                    linestyle="none", zorder=3, label=label if row == 0 else None)
    ax.text(1.005, len(CITIES) - 0.5, "\u0394F1", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=6.5, color=MUTED)
    ax.set_yticks(range(len(CITIES)))
    ax.set_yticklabels([t for _, t, *_ in CITIES][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.55, len(CITIES) - 0.45)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Macro F1 (reinforced concrete, masonry)")
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    save_figure(fig, OUT, "fig6_transfer_loss")


if __name__ == "__main__":
    main()
