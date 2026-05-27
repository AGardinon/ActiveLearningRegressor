#!/usr/bin/env python3
"""
Build RMSE trajectory data from per-cycle model predictions across all lab replicates.

For each replicate (experiment × R1/R2/R3) and each cycle that has a prediction .npy,
this script computes:

  rmse_dense  — RMSE vs dense GT (GP predictions over all 1000 points).
                Comparable to benchmark rmse_vs_gt_pool.

  rmse_sparse — RMSE vs sparse GT (raw averaged measurements at the ~495 covered points).
                Model-free; uses only actual experimental data.

  y_best      — cumulative maximum Mean_Diameter measured up to and including this cycle.
  n_points    — cumulative number of validated training points up to this cycle.

Validated points per cycle are read from the *_validated_points.csv file inside each
cycle_N/ folder.  This is also the source for the benchmark-format train_points_data.csv,
which carries actual coordinates and target values alongside cycle/repetition labels.

Output long-form CSV: lab_experiments/analysis/lab_trajectory.csv
  columns: experiment, replicate, cycle, n_points, rmse_dense, rmse_sparse, y_best

Benchmark-format outputs (one subdirectory per experiment under analysis/):
  <experiment>/benchmark_data.csv
  <experiment>/train_points_data.csv

Usage:
    python scripts/lab_analysis/build_trajectory.py
"""

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path

COORD_COLS = [
    "Conc_AP_Lys_100_2_(mM)",
    "Conc_AP_Asp_100_2_(mM)",
    "Conc_NaCl_(mM)",
]
TARGET_COL = "Mean_Diameter"
EXPERIMENTS = ["Andrea", "Andrea2", "Andrea3", "Andrea4"]
REPLICATES = ["R1", "R2", "R3"]
PREFIX = "RoboLab124_Lys100_Asp100_NaCl"


def get_prediction_npy(cycle_dir: Path, exp_id: str, cycle: int) -> np.ndarray | None:
    ml_dir = cycle_dir / f"cycle_{cycle}" / "machine_learning"
    if not ml_dir.exists():
        return None
    candidates = list(ml_dir.glob(f"{exp_id}_cycle_{cycle}_output_points_prediction.npy"))
    if not candidates:
        return None
    return np.load(candidates[0])


def load_validated_points(cycle_path: Path) -> pd.DataFrame:
    """Load the validated points CSV from a cycle's machine_learning subfolder."""
    ml_dir = cycle_path / "machine_learning"
    if not ml_dir.exists():
        return pd.DataFrame()
    candidates = list(ml_dir.glob("*validated_points*.csv"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_csv(candidates[0])


def process_replicate(
    lab_dir: Path,
    experiment: str,
    replicate: str,
    gt_sparse_values: np.ndarray,
    gt_sparse_indices: np.ndarray,
    gt_dense: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Process one replicate and return (trajectory_rows, point_rows).

    trajectory_rows — one dict per cycle, feeds lab_trajectory.csv.
    point_rows      — one dict per validated point, feeds train_points_data.csv.
    """
    exp_id = f"{PREFIX}_{experiment}_{replicate}"
    rep_dir = lab_dir / exp_id
    cycle_dir = rep_dir / "cycles"

    cycle_folders = sorted(
        [d for d in cycle_dir.iterdir() if d.is_dir() and re.match(r"cycle_\d+$", d.name)],
        key=lambda d: int(d.name.split("_")[1]),
    )

    traj_rows = []
    point_rows = []
    cumulative_best = -np.inf
    cumulative_n_points = 0

    for cycle_path in cycle_folders:
        cycle = int(cycle_path.name.split("_")[1])

        # Always collect validated points — even cycles without a prediction npy
        # (e.g. cycle_0 init) must contribute their points to train_points_data.csv
        # and update the cumulative counters.
        validated = load_validated_points(cycle_path)
        if not validated.empty and TARGET_COL in validated.columns:
            measured = validated[TARGET_COL].dropna()
            cycle_max = measured.max() if len(measured) > 0 else np.nan
            if not np.isnan(cycle_max):
                cumulative_best = max(cumulative_best, cycle_max)
            cumulative_n_points += len(measured)

            for _, row in validated.iterrows():
                if pd.notna(row.get(TARGET_COL)):
                    point_rows.append({
                        **{col: row[col] for col in COORD_COLS if col in row.index},
                        TARGET_COL: float(row[TARGET_COL]),
                        "cycle": cycle,
                        "experiment": experiment,
                        "replicate": replicate,
                    })

        # RMSE uses cycle_{N+1}'s prediction, which is the model trained on cycle_N's data.
        # This means cycle_0 gets cycle_1's prediction (first model fit), and the last
        # cycle is dropped (no cycle_{N+1} prediction exists).
        pred = get_prediction_npy(cycle_dir, exp_id, cycle + 1)
        if pred is None:
            continue

        y_best = cumulative_best if cumulative_best > -np.inf else np.nan
        n_points = cumulative_n_points

        rmse_dense = float(np.sqrt(np.mean((pred - gt_dense) ** 2)))
        pred_at_sparse = pred[gt_sparse_indices]
        rmse_sparse = float(np.sqrt(np.mean((pred_at_sparse - gt_sparse_values) ** 2)))

        traj_rows.append({
            "experiment": experiment,
            "replicate": replicate,
            "cycle": cycle,
            "n_points": n_points,
            "rmse_dense": rmse_dense,
            "rmse_sparse": rmse_sparse,
            "y_best": y_best,
        })

    return traj_rows, point_rows


def export_benchmark_format(
    traj_df: pd.DataFrame,
    points_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write benchmark-compatible CSVs from the lab trajectory and validated-points data.

    Produces one subdirectory per experiment under output_dir, each containing:
      - benchmark_data.csv    : one row per (repetition, cycle).
      - train_points_data.csv : one row per validated point with coordinates, target,
                                cycle, repetition, and acquisition_source placeholder.

    Replicate labels are mapped to integers: R1->1, R2->2, R3->3.
    cycle=0 (initialization) is included in both files so that cycle=0 anchors
    x=0 on the sample axis and serves as the normalization baseline in
    interpolate_simple_metric.
    """
    replicate_map = {r: i + 1 for i, r in enumerate(REPLICATES)}

    for exp in traj_df["experiment"].unique():
        exp_dir = output_dir / exp
        exp_dir.mkdir(parents=True, exist_ok=True)

        # benchmark_data.csv
        exp_traj = traj_df[traj_df["experiment"] == exp].copy()
        exp_traj["repetition"] = exp_traj["replicate"].map(replicate_map)
        benchmark_df = (
            exp_traj[["repetition", "cycle", "y_best", "rmse_dense", "rmse_sparse"]]
            .sort_values(["repetition", "cycle"])
        )
        benchmark_df.to_csv(exp_dir / "benchmark_data.csv", index=False)

        # train_points_data.csv
        exp_points = points_df[points_df["experiment"] == exp].copy()
        exp_points["repetition"] = exp_points["replicate"].map(replicate_map)
        exp_points["acquisition_source"] = "lab_protocol"
        train_cols = COORD_COLS + ["cycle", "repetition", "acquisition_source", TARGET_COL]
        train_points_df = (
            exp_points[train_cols]
            .sort_values(["repetition", "cycle"])
            .reset_index(drop=True)
        )
        train_points_df.to_csv(exp_dir / "train_points_data.csv", index=False)

        print(f"  Benchmark CSVs written -> {exp_dir}")


def main(lab_dir: Path, analysis_dir: Path, output_path: Path) -> None:
    gt = pd.read_csv(analysis_dir / "ground_truth.csv")
    gt_dense = np.load(analysis_dir / "gt_predictions.npy")

    gt_sparse_indices = gt["row_idx"].values.astype(int)
    gt_sparse_values = gt["Mean_Diameter_mean"].values

    all_traj_rows = []
    all_point_rows = []
    for exp in EXPERIMENTS:
        for rep in REPLICATES:
            print(f"Processing {exp}_{rep}...")
            traj_rows, point_rows = process_replicate(
                lab_dir, exp, rep, gt_sparse_values, gt_sparse_indices, gt_dense
            )
            all_traj_rows.extend(traj_rows)
            all_point_rows.extend(point_rows)
            print(f"  {len(traj_rows)} cycles with predictions, {len(point_rows)} validated points")

    traj = pd.DataFrame(all_traj_rows)
    traj.to_csv(output_path, index=False)
    print(f"\nTrajectory saved -> {output_path}")
    print(f"  Total rows: {len(traj)}")
    print(traj.groupby("experiment")[["cycle", "n_points", "rmse_dense"]].agg(["min", "max"]).to_string())

    points = pd.DataFrame(all_point_rows)
    print("\nExporting benchmark-format CSVs...")
    export_benchmark_format(traj, points, output_path.parent)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    default_lab = repo_root / "lab_experiments"
    default_analysis = default_lab / "analysis"

    parser = argparse.ArgumentParser(description="Build RMSE trajectory from per-cycle lab predictions")
    parser.add_argument("--lab-dir", type=Path, default=default_lab)
    parser.add_argument("--analysis-dir", type=Path, default=default_analysis)
    parser.add_argument("--output", type=Path, default=default_analysis / "lab_trajectory.csv")
    args = parser.parse_args()

    main(args.lab_dir, args.analysis_dir, args.output)
