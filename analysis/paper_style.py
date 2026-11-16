"""Shared visual style for every figure in paper/: one font, one palette,
and figure sizes in the paper's real text width (elsarticle 1p, 11 pt:
\\textwidth = 384 pt), so the font sizes set here are the printed sizes.

Palette logic:
- One accent hue (blue) for anything sequential or "data flow": the
  confusion-matrix scale, the learning curve, feature-importance bars and
  the train/validation/test partitions.
- Structural classes use "material" colors, distinct from the blue ramp:
  brick red (adobe), concrete gray (reinforced concrete), sand (masonry) and
  dark green (wood), far enough apart in lightness to survive color blindness.
- Relative-position categories use distinct muted hues (a single blue ramp
  was too hard to tell apart on a map).
"""
from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

TEXT_WIDTH_IN = 384 / 72

INK = "#262626"
MUTED = "#6B6B6B"
GRID = "#E6E6E6"
LIGHT = "#EDEDED"

# Blue ramp, light to dark
BLUES = ["#F2F6FA", "#D6E4F0", "#A9C6DF", "#6F9CC3", "#3E6F9E", "#1F4E79"]
ACCENT = BLUES[5]
PARTITION_COLORS = {"train": BLUES[5], "validation": BLUES[3], "test": BLUES[1]}
POSITION_COLORS = {
    "isolated": "#E4E4E4",  # gray
    "lateral": "#A8CBE6",   # sky blue
    "corner": "#BCA9D8",    # lavender
    "confined": "#7DBBAE",  # teal
    "torque": "#DE95A3",    # rose
}

CLASS_COLORS = {"ADO": "#B24A2A", "CR": "#9AA5AE", "M": "#F0D78C", "W": "#2B6B60"}
CLASS_NAMES = {"ADO": "Adobe", "CR": "Reinforced concrete", "M": "Masonry", "W": "Wood"}
CLASS_ORDER = ["ADO", "CR", "M", "W"]

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("paper_blues", ["#FFFFFF"] + BLUES[1:])

FONT_SIZE = 7.5


def apply_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],  # TrueType: embeds cleanly as Type 42
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 0.5,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 0.5,
        "ytick.labelsize": FONT_SIZE - 0.5,
        "legend.fontsize": FONT_SIZE - 0.5,
        "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
