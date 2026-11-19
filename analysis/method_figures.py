"""Figures 1-3: study area (six surveys), model input for one Zona 1 block
of Guatemala City, and the workflow diagram.

Figure 2 is illustrative. Plan-shape values and remote-sensing attributes
are the dataset's own. Relative positions and contact forces are recomputed
with footprint_attributes using a 0.15 m contact buffer (the footprint pixel
size), whereas the dataset's features were computed with buffer=0, which
treats walls digitized a few centimeters apart as not touching. Unlabeled
buildings have no height in the shared data, so missing heights are filled
with the dataset median for that computation only.

Each figure is written to figures/ as a JPG.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from footprint_attributes import position
from footprint_attributes.position import contact_force_vectors
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
from shapely.geometry import box

from paper_style import (BLUES, CLASS_COLORS, CLASS_NAMES, CLASS_ORDER, INK, LIGHT, MUTED,
                         PARTITION_COLORS, POSITION_COLORS, TEXT_WIDTH_IN, apply_style, save_figure)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
# Real footprints when available (local only), public versions otherwise:
# the unlabeled buildings of data/ are altered (analysis/anonymize_data.py).
DATA = ROOT / "data_private" if (ROOT / "data_private").is_dir() else ROOT / "data"
ZONA1 = DATA / "guatemala" / "zona1.gpkg"
# Reinforced concrete, corner, L-shaped (convex-hull irregularity 0.25,
# inertia-to-circle ratio 0.48, slenderness 2.24): the convex hull, bounding
# rectangle and equal-area circle all differ visibly. Its block mixes classes.
FOCUS_ID = 54
CONTACT_BUFFER_M = 0.15
LABEL_REPLACEMENTS = {"S_light": "S", "S_frame": "CR", "MUR": "M", "MCF": "M", "MR": "M", "I": None}
UNLABELED = "#FFFFFF"

apply_style()


def save(fig, name: str) -> None:
    save_figure(fig, OUT, name)


# ---------------------------------------------------------------------------
# Figure: what the model sees
# ---------------------------------------------------------------------------

def load_block() -> tuple[gpd.GeoDataFrame, object]:
    gdf = gpd.read_file(ZONA1)
    gdf["final_class"] = gdf["survey_structural_system"].map(
        lambda v: LABEL_REPLACEMENTS.get(v, v) if isinstance(v, str) else None)
    gdf["_h"] = gdf["height"].fillna(gdf["height"].median())
    pos = position(gdf[["geometry", "_h"]].copy(), height_column="_h", buffer=CONTACT_BUFFER_M)
    gdf["position"] = pos["relativePosition"].to_numpy()
    gdf["confinement"] = pos["contact_confinementRatio"].to_numpy()
    _, resultants = contact_force_vectors(gdf[["geometry", "_h"]].copy(), buffer=CONTACT_BUFFER_M,
                                          height_column="_h")
    in_block = gdf["block_id"] == gdf.loc[gdf["id"] == FOCUS_ID, "block_id"].iloc[0]
    resultants = resultants[resultants["geom_id"].isin(list(gdf.index[in_block]))]
    return gdf[in_block], resultants


def frame(ax, geoms: gpd.GeoDataFrame, pad: float = 3.0, bottom_extra: float = 0.0) -> None:
    minx, miny, maxx, maxy = geoms.total_bounds
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad - bottom_extra, maxy + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def scale_bar(ax, length_m: float) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs, ys = x0 + 0.03 * (x1 - x0), y0 + 0.01 * (y1 - y0)
    ax.plot([xs, xs + length_m], [ys, ys], color=INK, lw=1.2, solid_capstyle="butt")
    ax.text(xs + length_m / 2, ys + 0.02 * (y1 - y0), f"{length_m:g} m", ha="center", va="bottom", fontsize=6.3)


def fig_what_the_model_sees() -> None:
    block, resultants = load_block()
    focus = block[block["id"] == FOCUS_ID].iloc[0]
    others = block[block["id"] != FOCUS_ID]

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 3.75))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.05, 0.95, 1.05, 0.95], wspace=0.04, hspace=0.18,
                          left=0.0, right=1.0, top=0.93, bottom=0.0)
    ax_a, leg_a, ax_b, leg_b = (fig.add_subplot(gs[0, i]) for i in range(4))
    ax_c, leg_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2:])
    for ax in (leg_a, leg_b, leg_c):
        ax.axis("off")
    legend_kw = dict(loc="center left", frameon=False, fontsize=6.6, handlelength=1.5, handleheight=1.0,
                     borderaxespad=0.0, labelspacing=0.55, borderpad=0.0)
    lw = 0.35

    # (a) survey label
    block.plot(ax=ax_a, color=[CLASS_COLORS[c] if c else UNLABELED for c in block["final_class"]],
               edgecolor=INK, linewidth=lw)
    gpd.GeoSeries([focus.geometry], crs=block.crs).plot(ax=ax_a, facecolor="none", edgecolor=INK, linewidth=1.4)
    frame(ax_a, block, bottom_extra=9)
    scale_bar(ax_a, 20)
    present = [c for c in CLASS_ORDER if c in set(block["final_class"].dropna())]
    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor=INK, linewidth=lw, label=CLASS_NAMES[c]) for c in present]
    handles += [Patch(facecolor=UNLABELED, edgecolor=INK, linewidth=lw, label="Unlabeled"),
                Patch(facecolor="none", edgecolor=INK, linewidth=1.4, label="Building in (c), (d)")]
    leg_a.legend(handles=handles, **legend_kw)

    # (b) relative position and resultant contact forces
    block.plot(ax=ax_b, color=[POSITION_COLORS[p] for p in block["position"]], edgecolor=INK, linewidth=lw)
    # Arrows start at the building centroid and show the net push the building
    # receives from the neighbors it abuts, i.e. pointing away from them (the
    # reverse of footprint_attributes' vectors, which point toward the
    # neighbors; the classifier uses only magnitudes and angles, so the sign
    # is a display choice). Length is proportional to the
    # net contact force normalized as in the position classifier:
    # |resultant| / (sqrt(footprint area) x height), whose lateral-contact
    # threshold is 0.166. Values under 0.5 would render as bare arrowheads
    # and are omitted (mostly confined buildings, whose contacts cancel).
    meters_per_unit = 11.0
    for gid, anchor, v in zip(resultants["geom_id"], resultants["anchor"], resultants["vector"]):
        b = block.loc[gid]
        norm = np.hypot(*v) / (np.sqrt(b.geometry.area) * b["_h"])
        if norm < 0.5:
            continue
        d = -np.asarray(v) / np.hypot(*v) * norm * meters_per_unit
        arr = FancyArrowPatch(anchor, (anchor[0] + d[0], anchor[1] + d[1]), arrowstyle="-|>",
                              mutation_scale=4, color=INK, lw=0.7, shrinkA=0, shrinkB=0)
        arr.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="white")])
        ax_b.add_patch(arr)
    frame(ax_b, block, bottom_extra=9)
    present_pos = [p for p in POSITION_COLORS if p in set(block["position"])]
    handles = [Patch(facecolor=POSITION_COLORS[p], edgecolor=INK, linewidth=lw, label=p.capitalize())
               for p in present_pos]
    handles.append(Line2D([0], [0], color=INK, lw=0.8, marker=">", markersize=3, label="Net push from\nneighbors"))
    leg_b.legend(handles=handles, **legend_kw)

    # (c) plan-shape and contact descriptors of the focus building
    geom = focus.geometry
    near = others[others.distance(geom) < CONTACT_BUFFER_M]
    view = gpd.GeoSeries([geom.minimum_rotated_rectangle.buffer(3)], crs=block.crs)
    others[others.intersects(view.iloc[0])].plot(ax=ax_c, color=LIGHT, edgecolor="#C4C4C4", linewidth=lw)
    gpd.GeoSeries([geom], crs=block.crs).plot(ax=ax_c, facecolor=to_rgba(CLASS_COLORS[focus["final_class"]], 0.3),
                                              edgecolor=INK, linewidth=0.6)  # light, so the lines on top read
    shared = geom.boundary.intersection(near.buffer(CONTACT_BUFFER_M * 2).union_all())
    if not shared.is_empty:
        gpd.GeoSeries([shared], crs=block.crs).plot(ax=ax_c, color=BLUES[5], linewidth=2.2)
    ax_c.plot(*geom.convex_hull.exterior.xy, color=INK, lw=0.9, ls=(0, (4, 2)))
    ax_c.plot(*geom.minimum_rotated_rectangle.exterior.xy, color=MUTED, lw=0.9, ls=(0, (1, 1.5)))
    c = geom.centroid
    ax_c.add_patch(Circle((c.x, c.y), np.sqrt(geom.area / np.pi), fill=False, ec="#8F2F18", lw=0.9))
    frame(ax_c, view, pad=0)
    handles = [
        Line2D([0], [0], color=INK, lw=0.9, ls=(0, (4, 2)),
               label=f"Convex hull\nirregularity {focus['convex_hull_irregularity']:.2f}"),
        Line2D([0], [0], color=MUTED, lw=0.9, ls=(0, (1, 1.5)),
               label=f"Bounding rectangle\nslenderness {focus['slenderness_bbox']:.2f}"),
        Line2D([0], [0], color="#8F2F18", lw=0.9,
               label=f"Equal-area circle\ninertia ratio {focus['inertia_circle_ratio']:.2f}"),
        Line2D([0], [0], color=BLUES[5], lw=2.2,
               label="Shared walls"),
    ]
    if focus["ASCE7_holeRatio"] > 0:
        handles.insert(3, Patch(facecolor="white", edgecolor=INK, linewidth=0.6,
                                label=f"Courtyard\nhole ratio {focus['ASCE7_holeRatio']:.2f}"))
    leg_c.legend(handles=handles, **legend_kw)

    # (d) remote-sensing attributes of the focus building, as a 2 x 2 card
    ax_d.axis("off")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.add_patch(FancyBboxPatch((0.06, 0.14), 0.92, 0.72, boxstyle="round,pad=0,rounding_size=0.03",
                                  fc="#F7F7F7", ec="#D0D0D0", lw=0.6))
    cells = [
        ("Height", f"{focus['height']:.1f} m", "GlobalBuildingAtlas,\ndrone or survey"),
        ("Construction era", f"{int(focus['year'])}", "World Settlement Footprint\nEvolution, GHS-BUILT-S"),
        ("Code-quality level", focus["code_quality"].replace("_", "-"), "From era and\nnational code"),
        ("Roof material", focus["roof_material"].replace("_", " "), "Sentinel-2 +\northophoto"),
    ]
    for k, (name, value, source) in enumerate(cells):
        x, y = 0.12 + 0.45 * (k % 2), 0.8 - 0.34 * (k // 2)
        ax_d.text(x, y, name, fontsize=6.3, color=MUTED, va="top")
        ax_d.text(x, y - 0.065, value, fontsize=7.8, fontweight="bold", va="top")
        ax_d.text(x, y - 0.15, source, fontsize=5.9, color=MUTED, style="italic", va="top", linespacing=1.2)

    fig.canvas.draw()
    for ax, letter, text in ((ax_a, "a", "Survey label"), (ax_b, "b", "Neighbors"),
                             (ax_c, "c", "Plan shape"), (ax_d, "d", "Remote-sensing attributes")):
        row = (ax_a, ax_b) if ax in (ax_a, ax_b) else (ax_c, ax_d)
        top = max(a.get_position().y1 for a in row)
        x0 = ax.get_subplotspec().get_position(fig).x0 + (0.06 * ax.get_position().width if ax is ax_d else 0)
        fig.text(x0, top + 0.012, f"({letter}) {text}", fontsize=7.8, fontweight="bold", ha="left", va="bottom")

    save(fig, "fig2_model_input")


# ---------------------------------------------------------------------------
# Figure: workflow
# ---------------------------------------------------------------------------
# Compact overview band in the style of Nature methods overviews (e.g. the
# AlphaFold 2 overview, Jumper et al. 2021, Fig. 1): grayscale boxes, one
# accent hue, thin arrows, short labels.

BOX_EC = "#8C8C8C"
BOX_FC = "#F5F5F5"
SPLIT_COLORS = [PARTITION_COLORS["train"], PARTITION_COLORS["validation"], PARTITION_COLORS["test"]]


class FitChecker:
    """Fails loudly if any registered text overflows its box or overlaps
    another registered artist (text or marker)."""

    def __init__(self) -> None:
        self.pairs: list = []
        self.obstacles: list = []

    def add(self, text, patch) -> None:
        self.pairs.append((text, patch))

    def avoid(self, artist) -> None:
        self.obstacles.append(artist)

    def check(self, fig) -> None:
        renderer = fig.canvas.get_renderer()
        bad = []
        extents = [(text, text.get_window_extent(renderer)) for text, _ in self.pairs]
        for text, patch in self.pairs:
            t = text.get_window_extent(renderer)
            b = patch.get_window_extent(renderer)
            if t.x0 < b.x0 + 1 or t.x1 > b.x1 - 1 or t.y0 < b.y0 + 1 or t.y1 > b.y1 - 1:
                bad.append("overflow: " + text.get_text().replace("\n", " / "))
        others = extents + [(a, a.get_window_extent(renderer)) for a in self.obstacles]
        for i, (text, t) in enumerate(extents):
            for other, o in others[i + 1:]:
                if t.overlaps(o):
                    name = other.get_text() if hasattr(other, "get_text") else "a marker"
                    bad.append(f"overlap: {text.get_text()!r} with {name!r}")
        if bad:
            raise ValueError("; ".join(bad))


def rounded(ax, x, y, w, h, fc=BOX_FC, ec=BOX_EC, lw=0.6):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec, lw=lw)
    ax.add_patch(patch)
    return patch


def flow_arrow(ax, start, end) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, color="#5A5A5A",
                                 lw=0.7, shrinkA=0, shrinkB=0))


def fig_workflow() -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 31 / 100))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 31)
    ax.axis("off")
    fit = FitChecker()

    def header(x, text):
        ax.text(x, 28.2, text, fontsize=7.5, fontweight="bold", va="bottom")

    def label_box(x, y, w, h, text, size=6.8):
        patch = rounded(ax, x, y, w, h)
        fit.add(ax.text(x + w / 2, y + h / 2, text, fontsize=size, ha="center", va="center"), patch)

    # Inputs
    header(0, "Inputs")
    label_box(0, 20.5, 16, 6, "Field surveys")
    label_box(0, 12.5, 16, 6, "Footprints")
    rs = rounded(ax, 0, 0.5, 16, 10)
    fit.add(ax.text(8, 7.6, "Remote sensing", fontsize=6.8, ha="center", va="center"), rs)
    fit.add(ax.text(8, 3.7, "GBA, WSF, GHSL,\nSentinel-2", fontsize=6, color=MUTED, ha="center", va="center",
                    linespacing=1.15), rs)

    # Building table
    header(20, "Building table")
    table = rounded(ax, 20, 0.5, 18, 26.5, fc="white")
    rows = [("label", True), ("ADO, CR, M, S", False), ("", False), ("27 attributes", True),
            ("plan shape", False), ("neighbors", False), ("height, era,", False), ("code level, roof", False)]
    y = 24.6
    for text, strong in rows:
        if text:
            fit.add(ax.text(21.6, y, text, fontsize=6.8 if strong else 6.3, va="center",
                            fontweight="bold" if strong else "normal", color=INK if strong else MUTED), table)
        y -= 2.85
    for yy in (23.5, 15.5, 5.5):
        flow_arrow(ax, (16, yy), (20, yy))

    # Spatial split: a toy map of blocks, about 2/3 train, the rest val. and test
    header(42, "Spatial split")
    split = rounded(ax, 42, 0.5, 14, 26.5, fc="white")
    pattern = [[0, 0, 1, 0], [0, 2, 0, 0], [1, 0, 0, 2], [0, 0, 2, 0]]
    for r, row in enumerate(pattern):
        for c, k in enumerate(row):
            fit.avoid(ax.add_patch(FancyBboxPatch((43.6 + c * 2.8, 22.6 - r * 2.8), 2.2, 2.2,
                                                  boxstyle="square,pad=0", fc=SPLIT_COLORS[k], ec="none")))
    fit.add(ax.text(49, 12.6, "touching blocks", fontsize=5.8, color=MUTED, ha="center", va="center"), split)
    for k, name in enumerate(("train", "val.", "test")):
        yk = 9.0 - k * 2.9
        fit.avoid(ax.add_patch(FancyBboxPatch((43.6, yk - 0.75), 1.5, 1.5, boxstyle="square,pad=0",
                                              fc=SPLIT_COLORS[k], ec="none")))
        fit.add(ax.text(46, yk, name, fontsize=6.3, va="center", color=MUTED), split)
    flow_arrow(ax, (38, 13.75), (42, 13.75))

    # Model pipeline
    header(60, "Model pipeline")
    ax.text(60, 25.9, "5 classifiers", fontsize=6, color=MUTED, va="center")
    steps = ["1  CV on train", "2  Tune on val.", "3  Score on test"]
    for k, text in enumerate(steps):
        y0 = 17 - k * 8.2
        patch = rounded(ax, 60, y0, 19, 6.6, fc="white")
        bar = ax.add_patch(FancyBboxPatch((60, y0), 1.2, 6.6, boxstyle="square,pad=0", fc=SPLIT_COLORS[k], ec="none"))
        bar.set_clip_path(patch)  # keep the accent bar inside the rounded corners
        fit.add(ax.text(62.3, y0 + 3.3, text, fontsize=6.8, va="center"), patch)
        if k < 2:
            flow_arrow(ax, (69.5, y0), (69.5, y0 - 1.6))
    flow_arrow(ax, (56, 13.75), (60, 13.75))

    # Experiments
    header(83, "Experiments")
    exps = rounded(ax, 83, 0.5, 17, 26.5, fc="white")
    items = ["Within city", "Cross city", "Cross district", "Combined cities", "", "Learning curves",
             "Feature ranking", "Ablation"]
    y = 24.6
    for text in items:
        if text:
            fit.add(ax.text(84.6, y, text, fontsize=6.6, va="center", color=INK if y > 13 else MUTED), exps)
        y -= 2.85
    flow_arrow(ax, (79, 13.75), (83, 13.75))

    fig.canvas.draw()
    fit.check(fig)
    save(fig, "fig3_workflow")


# ---------------------------------------------------------------------------
# Figure: study area (the six labeled surveys)
# ---------------------------------------------------------------------------
# All six panels share one scale (a 1 km wide window), so building
# footprints are drawn at true size and stay legible: labeled buildings in
# their class color over the survey's other footprints in light gray.
# Surveys larger than the window show the 1 km window with the most labeled
# buildings, plus a locator inset of the whole survey with the window
# outlined. Minority classes are drawn last so masonry does not hide them.
# Labeled buildings come from the datasets used in the paper, so n matches Table
# tab:class-imbalance.

SURVEYS = [  # top row: larger-area surveys; bottom row: single-district surveys
    ("guatemala/zona10.gpkg", "Guatemala City", "Zona 10"),
    ("san_jose/san_jose_sample.gpkg", "San José", "citywide sample"),
    ("santo_domingo/santo_domingo_center.gpkg", "Santo Domingo", "citywide sample"),
    ("guatemala/zona1.gpkg", "Guatemala City", "Zona 1"),
    ("san_jose/mata_redonda.gpkg", "San José", "Mata Redonda"),
    ("santo_domingo/ensanche_quisquella.gpkg",
     "Santo Domingo", "Ensanche Quisqueya"),
]
# Gray background of the panels whose survey file does not cover the
# surroundings: the other survey of the same city, then OpenStreetMap
# footprints (analysis/context_footprints.py) where neither survey reaches.
OSM = ROOT / "data" / "context"
CONTEXT_SOURCES = {
    "Zona 1": [OSM / "osm_buildings_zona1.gpkg"],
    "Mata Redonda": [DATA / "san_jose" / "san_jose_sample.gpkg", OSM / "osm_buildings_mata_redonda.gpkg"],
    "Ensanche Quisqueya": [DATA / "santo_domingo" / "santo_domingo_center.gpkg"],
}
MAX_CONTEXT_OVERLAP = 0.2  # share of a context footprint already drawn by the survey
DRAW_ORDER = ["M", "CR", "S", "ADO"]  # majority first, minorities on top
WINDOW_W = 1000.0  # meters, every panel
PANEL_ASPECT = 1.2  # width / height
CONTEXT_GRAY = "#DADADA"


def best_window(points, half_w: float, half_h: float, step: float = 50.0) -> tuple[float, float]:
    """Center of the window holding the most points (ties: more central)."""
    xs, ys = points[:, 0], points[:, 1]
    best, best_c = -1, (xs.mean(), ys.mean())
    for cx_ in np.arange(xs.min(), xs.max() + step, step):
        in_x = np.abs(xs - cx_) <= half_w
        if in_x.sum() <= best:
            continue
        for cy_ in np.arange(ys.min(), ys.max() + step, step):
            n = int((in_x & (np.abs(ys - cy_) <= half_h)).sum())
            if n > best:
                best, best_c = n, (cx_, cy_)
    return best_c


def context_footprints(survey: str, g: gpd.GeoDataFrame, win: tuple[float, float, float, float]) -> gpd.GeoSeries:
    """Footprints of CONTEXT_SOURCES in the panel, without those already drawn."""
    if survey not in CONTEXT_SOURCES:
        return gpd.GeoSeries(crs=g.crs)
    window = gpd.GeoSeries([box(win[0], win[2], win[1], win[3])], crs=g.crs)
    drawn = g.cx[win[0]:win[1], win[2]:win[3]].geometry
    kept = []
    for source in CONTEXT_SOURCES[survey]:
        if not source.exists():
            continue
        other = gpd.read_file(source, bbox=window).to_crs(g.crs).geometry
        if other.empty:
            continue
        covered = drawn.union_all() if len(drawn) else None
        overlap = other.intersection(covered).area / other.area if covered is not None else 0
        other = other[overlap <= MAX_CONTEXT_OVERLAP]
        kept.append(other)
        drawn = gpd.GeoSeries(list(drawn) + list(other), crs=g.crs)
    return gpd.GeoSeries(pd.concat(kept), crs=g.crs) if kept else gpd.GeoSeries(crs=g.crs)


def fig_study_area() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_IN, 3.75))
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.08, wspace=0.05, hspace=0.2)
    half_w, half_h = WINDOW_W / 2, WINDOW_W / 2 / PANEL_ASPECT
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
    for ax, (rel, city, survey) in zip(axes.flat, SURVEYS):
        g = gpd.read_file(DATA / rel)
        g = g.to_crs(g.estimate_utm_crs())  # meters, so the window and scale bar are true distances
        g["final_class"] = g["survey_structural_system"].map(
            lambda v: LABEL_REPLACEMENTS.get(v, v) if isinstance(v, str) else None)
        lab = g[g["final_class"].notna()]
        cent = np.column_stack([lab.geometry.centroid.x, lab.geometry.centroid.y])
        minx, miny, maxx, maxy = lab.total_bounds
        fits = (maxx - minx) <= WINDOW_W * 0.95 and (maxy - miny) <= 2 * half_h * 0.95
        cx_, cy_ = ((minx + maxx) / 2, (miny + maxy) / 2) if fits else best_window(cent, half_w, half_h)
        win = (cx_ - half_w, cx_ + half_w, cy_ - half_h, cy_ + half_h)

        ctx = g.cx[win[0]:win[1], win[2]:win[3]]
        unlabeled = ctx[ctx["final_class"].isna()]
        unlabeled.plot(ax=ax, color=CONTEXT_GRAY, linewidth=0, zorder=1, rasterized=True)
        extra = context_footprints(survey, g, win)
        if len(extra):
            extra.plot(ax=ax, color=CONTEXT_GRAY, linewidth=0, zorder=1, rasterized=True)
        for cls in DRAW_ORDER:
            sub = ctx[ctx["final_class"] == cls]
            if len(sub):
                sub.plot(ax=ax, color=CLASS_COLORS[cls], edgecolor=INK, linewidth=0.05,
                         zorder=2)
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_color("#BDBDBD")

        # scale bar (identical in every panel) and labeled-building count
        xs, ys = win[0] + 0.05 * WINDOW_W, win[2] + 0.06 * 2 * half_h
        ax.plot([xs, xs + 250], [ys, ys], color=INK, lw=1.0, solid_capstyle="butt", zorder=4, path_effects=halo)
        ax.text(xs + 125, ys + 0.03 * 2 * half_h, "250 m", ha="center", va="bottom", fontsize=6, zorder=4,
                path_effects=halo)
        ax.text(0.97, 0.05, f"n = {len(lab):,}", transform=ax.transAxes, fontsize=6.2, ha="right", va="bottom",
                zorder=4, path_effects=halo)
        ax.set_title(f"{city}: {survey}", fontsize=6.8, fontweight="bold", loc="left", pad=3)

        if not fits:  # locator: whole survey, window outlined
            ins = ax.inset_axes([0.66, 0.6, 0.32, 0.37])
            pad = 0.08 * max(maxx - minx, maxy - miny)
            reach = max(maxx - minx, maxy - miny) / 2 + 2 * pad  # beyond what the inset shows
            ins_win = ((minx + maxx) / 2 - reach, (minx + maxx) / 2 + reach,
                       (miny + maxy) / 2 - reach, (miny + maxy) / 2 + reach)
            for footprints in (g[g["final_class"].isna()].geometry, context_footprints(survey, g, ins_win)):
                if len(footprints):
                    footprints.plot(ax=ins, color=CONTEXT_GRAY, edgecolor=CONTEXT_GRAY, linewidth=0.1,
                                    rasterized=True)
            ins.scatter(cent[:, 0], cent[:, 1], s=0.15, color=MUTED, linewidths=0, zorder=2, rasterized=True)
            ins.add_patch(plt.Rectangle((win[0], win[2]), WINDOW_W, 2 * half_h, fill=False, ec="#C0392B", lw=0.8))
            ins.set_xlim(minx - pad, maxx + pad)
            ins.set_ylim(miny - pad, maxy + pad)
            ins.set_aspect("equal", adjustable="datalim")
            ins.set_xticks([])
            ins.set_yticks([])
            ins.set_facecolor("white")
            for spine in ins.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_color("#9A9A9A")

    handles = [Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=CLASS_COLORS[c],
                      markeredgecolor=INK, markeredgewidth=0.3, markersize=5.5, label=CLASS_NAMES[c])
               for c in CLASS_ORDER]
    handles.append(Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=CONTEXT_GRAY,
                          markeredgecolor="none", markersize=5.5, label="Unlabeled footprint"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig1_study_area")


if __name__ == "__main__":
    fig_study_area()
    fig_what_the_model_sees()
    fig_workflow()
