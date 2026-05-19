#!/usr/bin/env python3
"""
Build ground truth by pooling measured data from all lab experiment replicates.

Each replicate's base CSV contains the full 1000-point design space with measured
Mean_Diameter values filled in where experiments were conducted.  This script:
  1. Loads all 9 base CSVs (3 experiments x 3 replicates)
  2. Pools every non-null (coordinate, Mean_Diameter) pair
  3. Averages duplicate coordinates and records measurement statistics
  4. Records the row index of each GT point in the shared design space

Outputs (in --output-dir):
  ground_truth.csv   — GT points with mean/std/count and row index
  design_space.csv   — the 1000-point design space (coordinate → row index reference)

Usage:
    python scripts/lab_analysis/build_ground_truth.py
    python scripts/lab_analysis/build_ground_truth.py --lab-dir path/to/lab_experiments
"""

import argparse
import pandas as pd
from pathlib import Path

COORD_COLS = [
    "Conc_AP_Lys_100_2_(mM)",
    "Conc_AP_Asp_100_2_(mM)",
    "Conc_NaCl_(mM)",
]
TARGET_COL = "Mean_Diameter"
EXPERIMENTS = ["Andrea", "Andrea2", "Andrea3"]
REPLICATES = ["R1", "R2", "R3"]
PREFIX = "RoboLab124_Lys100_Asp100_NaCl"


def load_replicate_base_csv(lab_dir: Path, experiment: str, replicate: str) -> pd.DataFrame:
    exp_id = f"{PREFIX}_{experiment}_{replicate}"
    path = lab_dir / exp_id / "dataset" / f"{exp_id}.csv"
    print(f"Loading\n - \t{path}...")
    df = pd.read_csv(path)
    return df[df[TARGET_COL].notna()][COORD_COLS + [TARGET_COL]].copy()


def load_design_space(lab_dir: Path) -> pd.DataFrame:
    """Return the 1000-point design space from Andrea_R1 (all replicates are identical)."""
    ref_id = f"{PREFIX}_Andrea_R1"
    path = lab_dir / ref_id / "dataset" / f"{ref_id}.csv"
    df = pd.read_csv(path)
    return df[COORD_COLS].reset_index(drop=True)


def build_ground_truth(lab_dir: Path) -> pd.DataFrame:
    frames = []
    for exp in EXPERIMENTS:
        for rep in REPLICATES:
            df = load_replicate_base_csv(lab_dir, exp, rep)
            df["source"] = f"{exp}_{rep}"
            frames.append(df)
            print(f"  {exp}_{rep}: {len(df)} measured points")

    all_data = pd.concat(frames, ignore_index=True)

    gt = (
        all_data.groupby(COORD_COLS)[TARGET_COL]
        .agg(
            Mean_Diameter_mean="mean",
            Mean_Diameter_std="std",
            n_measurements="count",
        )
        .reset_index()
    )
    gt["Mean_Diameter_std"] = gt["Mean_Diameter_std"].fillna(0.0)
    return gt


def add_row_indices(gt: pd.DataFrame, design_space: pd.DataFrame) -> pd.DataFrame:
    """Add the row index in the shared design space for each GT point."""
    ds_indexed = design_space.copy()
    ds_indexed["row_idx"] = ds_indexed.index

    merged = gt.merge(ds_indexed, on=COORD_COLS, how="left")
    n_missing = merged["row_idx"].isna().sum()
    if n_missing > 0:
        raise ValueError(f"{n_missing} GT coordinates not found in design space")
    merged["row_idx"] = merged["row_idx"].astype(int)
    return merged


def main(lab_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading design space...")
    design_space = load_design_space(lab_dir)
    ds_path = output_dir / "design_space.csv"
    design_space.to_csv(ds_path, index=True, index_label="row_idx")
    print(f"  Saved {len(design_space)}-point design space -> {ds_path}")

    print("\nBuilding ground truth...")
    gt = build_ground_truth(lab_dir)
    gt = add_row_indices(gt, design_space)

    gt_path = output_dir / "ground_truth.csv"
    gt.to_csv(gt_path, index=False)

    print("\nGround truth summary:")
    print(f"  Unique coordinates: {len(gt)} / {len(design_space)}")
    print(f"  Total measurements: {int(gt['n_measurements'].sum())}")
    print(f"  Points measured 1x: {(gt['n_measurements'] == 1).sum()}")
    print(f"  Points measured 2x: {(gt['n_measurements'] == 2).sum()}")
    print(f"  Points measured 3x+: {(gt['n_measurements'] >= 3).sum()}")
    print(f"  Mean_Diameter range: {gt['Mean_Diameter_mean'].min():.3f} - {gt['Mean_Diameter_mean'].max():.3f}")
    print(f"\nSaved -> {gt_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Build pooled ground truth from all lab experiment replicates")
    parser.add_argument("--lab-dir", type=Path, default=repo_root / "lab_experiments")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "lab_experiments" / "analysis")
    args = parser.parse_args()

    main(args.lab_dir, args.output_dir)
