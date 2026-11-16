"""Per-city learning curve (Section 5.2, Figure 5; XGBoost only).

Refits the already-tuned model on growing spatially blocked subsets of each
city's training pool (sizes from 50 buildings upward) and scores it on a fixed
held-out set, repeated 5 times with different block assignments. The winning
CV configuration of each city's main experiment is reused (no new search).
Subsets that miss a class present in the test set are skipped and logged.
"""

from __future__ import annotations

import functools
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_structural_system import test as test_module
from ml_structural_system import train as train_module
from ml_structural_system.config import load_config
from ml_structural_system.dataset_prep import load_preprocessing_config, prepare_raw_splits
from ml_structural_system.prepare import prepare_data

GROUP_DIR = Path(__file__).resolve().parent
DATA_DIR = GROUP_DIR.parent / "data"
OUT_DIR = GROUP_DIR / "learning_curve"

N_REPEATS = 5
BASE_SEED = 300
TEST_SIZE = 0.2
MIN_SIZE = 50
GROWTH = 1.5

CITIES = {
    "guatemala": [
        DATA_DIR / "guatemala" / "zona10.gpkg",
        DATA_DIR / "guatemala" / "zona1.gpkg",
    ],
    "san_jose": [
        DATA_DIR / "san_jose" / "esquivel.gpkg",
        DATA_DIR / "san_jose" / "mata_redonda.gpkg",
    ],
    "santo_domingo": [
        DATA_DIR / "santo_domingo" / "ensanche_quisquella.gpkg",
        DATA_DIR / "santo_domingo" / "naco.gpkg",
    ],
}


def size_grid(n_pool: int) -> list[int]:
    """Adaptive absolute-count grid: fine (50-100-150) at first, then ~1.5x
    geometric growth (rounded to the nearest 10), capped at n_pool."""
    if n_pool <= MIN_SIZE:
        return [n_pool]
    sizes = [50, 100, 150]
    cur = 150.0
    while True:
        cur *= GROWTH
        rounded = int(round(cur / 10.0) * 10)
        if rounded >= n_pool:
            break
        sizes.append(rounded)
    sizes = [s for s in sizes if s < n_pool]
    sizes.append(n_pool)
    return sorted(set(sizes))


def class_interleaved_block_order(train_df: pd.DataFrame, label_col: str, seed: int) -> list[str]:
    """Order block_ids so a small prefix is more likely to cover every class:
    each class's own blocks shuffled independently, then round-robin across
    classes (in a shuffled class order) so the first pass touches every
    class once before any class's second block."""
    rng = np.random.RandomState(seed)
    block_label = train_df.groupby("block_id")[label_col].agg(lambda s: s.value_counts().idxmax())
    blocks_by_class: dict = {}
    for block_id, label in block_label.items():
        blocks_by_class.setdefault(label, []).append(block_id)
    for label in blocks_by_class:
        rng.shuffle(blocks_by_class[label])
    class_order = list(blocks_by_class.keys())
    rng.shuffle(class_order)

    order: list[str] = []
    idx = {c: 0 for c in class_order}
    progress = True
    while progress:
        progress = False
        for c in class_order:
            i = idx[c]
            blocks = blocks_by_class[c]
            if i < len(blocks):
                order.append(blocks[i])
                idx[c] = i + 1
                progress = True
    return order


def subset_for_size(train_df: pd.DataFrame, order: list[str], n_target: int) -> pd.DataFrame:
    block_sizes = train_df.groupby("block_id").size().to_dict()
    chosen: list[str] = []
    cum = 0
    for b in order:
        if cum >= n_target:
            break
        chosen.append(b)
        cum += block_sizes[b]
    return train_df[train_df["block_id"].isin(chosen)].reset_index(drop=True)


def write_outer_config(city: str, k: int) -> Path:
    repeat_dir = OUT_DIR / city / f"repeat{k}" / "outer"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "extends": str((GROUP_DIR / city / "config.yaml").resolve()),
        "preprocessing": {
            "seed": BASE_SEED + k,
            "id_col": "block_id",
            "data": {
                "source": [str(p.resolve()) for p in CITIES[city]],
                "test_size": TEST_SIZE,
                "val_size": 0,
            },
        },
    }
    config_path = repeat_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    return config_path


def write_point_config(city: str, k: int, n_target: int, train_path: Path, test_path: Path) -> Path:
    point_dir = OUT_DIR / city / f"repeat{k}" / f"size{n_target}"
    point_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "extends": str((GROUP_DIR / city / "config.yaml").resolve()),
        "dataset": {
            # No val split at this size (val_size: 0 below writes no
            # val_preprocessed.csv), so main.yaml's dataset.val_path/val_gpkg
            # must be cleared too, load_config() hard-fails if a configured
            # val_path doesn't exist on disk, even though val_path is
            # otherwise optional.
            "val_path": None,
            "val_gpkg": None,
        },
        "preprocessing": {
            "seed": BASE_SEED + k,
            "data": {
                "train_source": [str(train_path.resolve())],
                "test_source": [str(test_path.resolve())],
                "val_size": 0,
            },
        },
        "mlflow": {
            # The city's own mlflow.db holds its CV results, so
            # get_best_run() finds the frozen CV winner per model.
            "db_path": str((GROUP_DIR / city / "mlflow.db").resolve()),
            "experiment_final": f"final_evaluation_lc_r{k}_n{n_target}",
        },
        "models": {
            "enabled": [
                "XGBoost",
            ],
        },
    }
    config_path = point_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    return config_path


def run_repeat(city: str, k: int) -> list[dict]:
    print(f"=== {city} repeat {k}: outer space-aware split (test_size={TEST_SIZE}) ===", flush=True)
    outer_config_path = write_outer_config(city, k)
    prep_cfg = load_preprocessing_config(outer_config_path)
    raw_split = prepare_raw_splits(prep_cfg)
    train_df, test_df = raw_split.train_df, raw_split.test_df
    print(f"  outer split: {len(train_df)} train rows, {len(test_df)} test rows", flush=True)

    outer_dir = outer_config_path.parent
    train_full_path = outer_dir / "train_full.gpkg"
    test_full_path = outer_dir / "test_full.gpkg"
    gpd.GeoDataFrame(train_df).to_file(train_full_path, driver="GPKG")
    gpd.GeoDataFrame(test_df).to_file(test_full_path, driver="GPKG")

    order = class_interleaved_block_order(train_df, prep_cfg.label, seed=BASE_SEED + k)
    rows: list[dict] = []

    for n_target in size_grid(len(train_df)):
        subset_df = subset_for_size(train_df, order, n_target)
        point_dir = OUT_DIR / city / f"repeat{k}" / f"size{n_target}"
        point_dir.mkdir(parents=True, exist_ok=True)
        subset_path = point_dir / "train_subset.gpkg"
        gpd.GeoDataFrame(subset_df).to_file(subset_path, driver="GPKG")

        config_path = write_point_config(city, k, n_target, subset_path, test_full_path)
        print(f"  size={n_target} (actual n_train={len(subset_df)}) ...", flush=True)
        try:
            prepare_data(config_path, point_dir)
            cfg = load_config(config_path)
            train_module.run(cfg, final_only=True)
            results = test_module.run(cfg)
        except Exception as exc:  # noqa: BLE001 - one bad point must not stop the sweep
            print(f"    FAILED: {exc}", flush=True)
            traceback.print_exc()
            continue

        for r in results:
            if r.status != "completed":
                continue
            rows.append({
                "city": city,
                "repeat": k,
                "target_size": n_target,
                "actual_n_train": len(subset_df),
                "model": r.model_name,
                "macro_f1": r.metrics.get("macro_f1"),
                "accuracy": r.metrics.get("accuracy"),
                "cohen_kappa": r.metrics.get("cohen_kappa"),
                "tss_macro": r.metrics.get("tss_macro"),
            })
            print(f"    {r.model_name}: macro_f1={r.metrics.get('macro_f1'):.4f}", flush=True)

    return rows


def run_city(city: str) -> None:
    out_path = GROUP_DIR / f"learning_curve_{city}.csv"
    # Resume support: each point here is a full refit+calibration+test (not
    # just a skip_completed model check), so a from-scratch re-run of this
    # sweep costs real time (thousands of points across cities/repeats/
    # A city whose CSV already covers every repeat is skipped, so a re-run resumes.
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if not existing.empty and set(range(N_REPEATS)) <= set(existing["repeat"].unique()):
            print(f"  {out_path} already covers all {N_REPEATS} repeats, skipping (resume).", flush=True)
            return
    all_rows: list[dict] = []
    for k in range(N_REPEATS):
        all_rows.extend(run_repeat(city, k))
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"=== {city} done, written to {out_path} ===", flush=True)


@functools.cache
def full_test_classes(city: str, k: int, n_target: int) -> bool:
    """Whether this point's test set kept every class of the repeat's outer test set."""
    outer_dir = OUT_DIR / city / f"repeat{k}" / "outer"
    label = load_preprocessing_config(outer_dir / "config.yaml").label
    outer = gpd.read_file(outer_dir / "test_full.gpkg")
    point = gpd.read_file(OUT_DIR / city / f"repeat{k}" / f"size{n_target}" / "splits" / "test_df.gpkg")
    return set(point[label].dropna()) == set(outer[label].dropna())


def main() -> None:
    cities = [sys.argv[1]] if len(sys.argv) > 1 else list(CITIES)
    for city in cities:
        run_city(city)

    if len(sys.argv) <= 1:
        dfs = [pd.read_csv(GROUP_DIR / f"learning_curve_{c}.csv") for c in CITIES]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(GROUP_DIR / "learning_curve_results.csv", index=False)
        # At small n, a class with fewer than min_class_size_per_split (20)
        # buildings in the training subset is dropped from training, and
        # therefore also from that point's test set (dataset_prep.splitting
        # drops test rows of classes the label encoder never saw). Such a
        # point is scored on an easier, reduced-class task than the rest of
        # the curve, at the extreme a single-class test, giving a trivial
        # macro_f1=1.0 (every target_size=50 repeat, and san_jose's
        # target_size=100 repeat 1). Only repeats scored on the repeat's
        # full outer test class set are aggregated.
        kept = df[df.apply(lambda r: full_test_classes(r["city"], r["repeat"], r["target_size"]), axis=1)]
        summary = kept.groupby(["city", "model", "target_size"])["macro_f1"].agg(["mean", "std", "count"])
        summary.to_csv(GROUP_DIR / "learning_curve_summary.csv")
        print("\n=== Summary written to learning_curve_summary.csv ===")


if __name__ == "__main__":
    main()
