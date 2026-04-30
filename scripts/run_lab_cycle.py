#!/usr/bin/env python3
"""
Real-lab single-cycle Active Learning script.

Run once per lab cycle::

    python scripts/run_lab_cycle.py -c path/to/config.yaml

Cycle 0 (first run)
    Creates the experiment directory, saves POOL / EVIDENCE / CANDIDATES CSVs,
    fits and saves the scaler.  If ``initial_evidence_file`` is set in the
    config, the first AL cycle is run immediately; otherwise initial sampling
    (FPS / voronoi / random) is performed and the result written to
    ``cycle_0/output_sampled.csv`` for the user to measure.

Cycle N > 0
    Reads ``cycle_{N-1}/validated.csv`` (filled in by the user after
    measurement), validates it, updates EVIDENCE and CANDIDATES, then runs
    the AL cycle.

In-silico mode
    If ``ground_truth_file`` is set, after writing ``output_sampled.csv`` the
    script auto-fills ``validated.csv`` from the ground truth — no manual
    measurement needed.  This turns the single-cycle script into a simple
    N-cycle simulation.
"""

import yaml
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from activereg.format import LAB_AL_REPO, DATASETS_REPO
from activereg.utils import create_strict_folder, save_to_json
from activereg.sampling import sample_landscape
from activereg.experiment import (
    setup_data_pool,
    setup_multi_property_ml_model,
    remove_evidence_from_gt,
    run_single_al_cycle,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    """Load a CSV from an absolute path, relative path, or DATASETS_REPO fallback."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return pd.read_csv(p)
    return pd.read_csv(DATASETS_REPO / path)


def _make_scaler(scaler_type: str):
    if scaler_type == 'StandardScaler':
        return StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        return MinMaxScaler()
    else:
        raise ValueError(f"Unsupported data_scaler: {scaler_type!r}. Choose StandardScaler or MinMaxScaler.")


def _resolve_output_dir(config: dict) -> Path:
    raw = config.get('output_dir')
    return Path(raw) if raw else LAB_AL_REPO


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_validated_csv(
    validated_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    search_vars: list[str],
    target_vars: list[str],
    path: Path,
) -> None:
    """Raise ValueError with a human-readable message if validated.csv is malformed."""
    missing_search = [v for v in search_vars if v not in validated_df.columns]
    if missing_search:
        raise ValueError(
            f"{path}: missing feature columns: {missing_search}\n"
            "  Fix: ensure all search_space_variables columns are present."
        )
    missing_target = [v for v in target_vars if v not in validated_df.columns]
    if missing_target:
        raise ValueError(
            f"{path}: missing target columns: {missing_target}\n"
            "  Fix: add a column for each target_variable with measured values."
        )
    nan_cols = [v for v in target_vars if validated_df[v].isna().any()]
    if nan_cols:
        raise ValueError(
            f"{path}: NaN values in target columns {nan_cols}\n"
            "  Fix: fill in all measured values before re-running."
        )
    if len(validated_df) != len(sampled_df):
        raise ValueError(
            f"{path}: row count mismatch — validated.csv has {len(validated_df)} rows "
            f"but output_sampled.csv has {len(sampled_df)} rows.\n"
            "  Fix: do not add or remove rows from validated.csv."
        )
    if not (
        validated_df[search_vars]
        .reset_index(drop=True)
        .equals(sampled_df[search_vars].reset_index(drop=True))
    ):
        raise ValueError(
            f"{path}: feature values do not match output_sampled.csv.\n"
            "  Fix: do not modify the feature columns in validated.csv."
        )


# ── Initialization (cycle 0) ──────────────────────────────────────────────────

def _initialize_experiment(
    experiment_dir: Path,
    pool_df: pd.DataFrame,
    evidence_df: pd.DataFrame | None,
    search_vars: list[str],
    target_vars: list[str],
    scaler_type: str,
):
    """Create folder structure, save dataset CSVs, fit+save scaler.

    Args:
        experiment_dir: Root of this experiment (created here).
        pool_df: Full search space, features only.
        evidence_df: Optional initial evidence (features + targets).
        search_vars: Feature column names.
        target_vars: Target column names.
        scaler_type: 'StandardScaler' or 'MinMaxScaler'.

    Returns:
        Fitted scaler instance.
    """
    dataset_dir = experiment_dir / 'dataset'
    create_strict_folder(str(dataset_dir))

    pool_features = pool_df[search_vars]
    pool_features.to_csv(dataset_dir / 'POOL.csv', index=False)

    if evidence_df is not None and len(evidence_df) > 0:
        evidence_df.to_csv(dataset_dir / 'EVIDENCE.csv', index=False)
        candidates_df = remove_evidence_from_gt(pool_features, evidence_df, search_vars)
    else:
        pd.DataFrame(columns=search_vars + target_vars).to_csv(
            dataset_dir / 'EVIDENCE.csv', index=False
        )
        candidates_df = pool_features.copy()

    candidates_df.to_csv(dataset_dir / 'CANDIDATES.csv', index=False)

    scaler = _make_scaler(scaler_type)
    _, scaler = setup_data_pool(df=pool_features, search_var=search_vars, scaler=scaler)
    joblib.dump(scaler, dataset_dir / 'scaler.joblib')

    return scaler


# ── State update (cycle N > 0) ────────────────────────────────────────────────

def _state_update(
    experiment_dir: Path,
    prev_cycle: int,
    search_vars: list[str],
    target_vars: list[str],
) -> None:
    """Read and validate validated.csv from the previous cycle; update EVIDENCE + CANDIDATES."""
    prev_cycle_dir = experiment_dir / f'cycle_{prev_cycle}'
    validated_path = prev_cycle_dir / 'validated.csv'
    sampled_path   = prev_cycle_dir / 'output_sampled.csv'

    if not validated_path.exists():
        raise FileNotFoundError(
            f"{validated_path} not found.\n"
            "  Fix: fill in measured target values, save as validated.csv, then re-run."
        )

    validated_df = pd.read_csv(validated_path)
    sampled_df   = pd.read_csv(sampled_path)

    _validate_validated_csv(validated_df, sampled_df, search_vars, target_vars, validated_path)

    dataset_dir   = experiment_dir / 'dataset'
    evidence_df   = pd.read_csv(dataset_dir / 'EVIDENCE.csv')
    candidates_df = pd.read_csv(dataset_dir / 'CANDIDATES.csv')

    # Append new measurements to evidence
    evidence_df = pd.concat([evidence_df, validated_df], ignore_index=True)
    evidence_df.to_csv(dataset_dir / 'EVIDENCE.csv', index=False)

    # Remove newly measured points from candidates
    measured_set  = set(validated_df[search_vars].apply(tuple, axis=1))
    candidates_df = candidates_df[
        ~candidates_df[search_vars].apply(tuple, axis=1).isin(measured_set)
    ]
    candidates_df.to_csv(dataset_dir / 'CANDIDATES.csv', index=False)

    print(f'State updated: evidence={len(evidence_df)} pts, candidates={len(candidates_df)} pts')


# ── In-silico auto-validation ─────────────────────────────────────────────────

def _auto_validate(
    gt_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    search_vars: list[str],
    target_vars: list[str],
    output_path: Path,
) -> None:
    """Look up target values from gt_df and write validated.csv."""
    validated_df = pd.merge(sampled_df[search_vars], gt_df, on=search_vars, how='left')
    nan_targets  = [v for v in target_vars if validated_df[v].isna().any()]
    if nan_targets:
        raise ValueError(
            f"Auto-validation failed: could not match all sampled points in the "
            f"ground truth for columns {nan_targets}.\n"
            "  Check that search_space_variables match between pool and ground_truth_file."
        )
    validated_df.to_csv(output_path, index=False)


# ── Cycle output saving ───────────────────────────────────────────────────────

def _save_cycle_outputs(
    cycle_dir: Path,
    result: dict,
    pool_df: pd.DataFrame,
    acqui_params: list[dict],
    search_vars: list[str],
    target_vars: list[str],
    cycle: int,
) -> None:
    """Write per-cycle output files from run_single_al_cycle result dict."""
    # Points to measure (features + NaN targets)
    result['next_batch_df'].to_csv(cycle_dir / 'output_sampled.csv', index=False)

    # Model predictions over full pool
    pred_records = {col: pool_df[col].values for col in search_vars}
    y_pred = result['y_pred']
    y_unc  = result['y_unc']
    if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
        pred_flat = y_pred.ravel()
        unc_flat  = y_unc.ravel()
        pred_records['y_pred']        = pred_flat
        pred_records['y_uncertainty'] = unc_flat
    else:
        for i, t in enumerate(target_vars):
            pred_records[f'y_pred_{t}']        = y_pred[:, i]
            pred_records[f'y_uncertainty_{t}'] = y_unc[:, i]
    pd.DataFrame(pred_records).to_csv(cycle_dir / 'predictions.csv', index=False)

    # Acquisition landscapes over candidates
    candidates_df = result['candidates_df']
    land_records  = {col: candidates_df[col].values for col in search_vars}
    landscapes    = result['landscapes']
    for i in range(landscapes.shape[0]):
        mode = acqui_params[i].get('acquisition_mode', str(i)) if i < len(acqui_params) else str(i)
        land_records[f'landscape_{mode}'] = landscapes[i]
    pd.DataFrame(land_records).to_csv(cycle_dir / 'landscapes.csv', index=False)

    # Model snapshot
    joblib.dump(result['trained_model'], cycle_dir / 'model_snapshot.pkl')

    # Structured log
    log = {
        'cycle':       cycle,
        'y_best':      result['y_best'],
        'n_sampled':   len(result['sampled_idx']),
        'sampled_idx': [int(i) for i in result['sampled_idx']],
        'model_repr':  repr(result['trained_model']),
    }
    save_to_json(log, cycle_dir / 'log.json', timestamp=False)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run one AL cycle for a real-lab or in-silico experiment.'
    )
    parser.add_argument('-c', '--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # ── Config extraction ──────────────────────────────────────────────────

    EXP_NAME      = config['experiment_name']
    EXP_NOTES     = config.get('experiment_notes', '')
    SEARCH_VAR    = config['search_space_variables']
    TARGET_VAR    = config['target_variables']
    SCALER_TYPE   = config.get('data_scaler', 'StandardScaler')
    ACQUI_PARAMS  = config.get('acquisition_parameters', [])
    INIT_BATCH    = config.get('init_batch_size', 8)
    INIT_SAMPLING = config.get('init_sampling', 'fps')
    GT_FILE       = config.get('ground_truth_file')
    POOL_FILE     = config.get('pool_file')

    BATCH_SEL_METHOD = config.get('batch_selection_method', 'highest_landscape')
    BATCH_SEL_PARAMS = config.get('batch_selection_params') or {}

    pen_cfg      = config.get('penalization_params')
    PENALIZATION = (float(pen_cfg['radius']), float(pen_cfg['strength'])) if pen_cfg else None

    output_dir = _resolve_output_dir(config)
    EXP_DIR    = output_dir / EXP_NAME

    # ── Pool loading ───────────────────────────────────────────────────────

    if POOL_FILE is None and GT_FILE is None:
        raise ValueError("Config must specify 'pool_file' (real-lab) or 'ground_truth_file' (in-silico).")

    pool_raw_df = _load_csv(POOL_FILE) if POOL_FILE else _load_csv(GT_FILE)
    pool_df     = pool_raw_df[SEARCH_VAR]   # features only

    # ── Ground truth for in-silico auto-validation ─────────────────────────

    gt_df = _load_csv(GT_FILE) if GT_FILE else None

    # ── Cycle detection ────────────────────────────────────────────────────

    cycle = len(list(EXP_DIR.glob('cycle_*'))) if EXP_DIR.exists() else 0

    print(
        f'\n# ----------------------------------------------------------------\n'
        f'#   Experiment : {EXP_NAME}\n'
        f'#   Cycle      : {cycle}\n'
        f'# ----------------------------------------------------------------\n'
        f'Notes: {EXP_NOTES}\n'
    )

    # ── Cycle 0: initialization ────────────────────────────────────────────

    if cycle == 0:
        initial_evidence_file = config.get('initial_evidence_file')
        evidence_df = _load_csv(initial_evidence_file) if initial_evidence_file else None

        scaler = _initialize_experiment(
            experiment_dir=EXP_DIR,
            pool_df=pool_df,
            evidence_df=evidence_df,
            search_vars=SEARCH_VAR,
            target_vars=TARGET_VAR,
            scaler_type=SCALER_TYPE,
        )
        print(f'Experiment initialized: {EXP_DIR}')

        cycle_dir = EXP_DIR / 'cycle_0'
        create_strict_folder(str(cycle_dir))

        if evidence_df is not None and len(evidence_df) > 0:
            # Run first AL cycle using the provided initial evidence
            ml_model = setup_multi_property_ml_model(config, TARGET_VAR)
            result = run_single_al_cycle(
                pool_df=pool_df,
                evidence_df=evidence_df,
                scaler=scaler,
                ml_model=ml_model,
                search_vars=SEARCH_VAR,
                target_vars=TARGET_VAR,
                acquisition_params=ACQUI_PARAMS,
                batch_selection_method=BATCH_SEL_METHOD,
                batch_selection_params=BATCH_SEL_PARAMS,
                penalization_params=PENALIZATION,
            )
            _save_cycle_outputs(
                cycle_dir=cycle_dir,
                result=result,
                pool_df=pool_df,
                acqui_params=ACQUI_PARAMS,
                search_vars=SEARCH_VAR,
                target_vars=TARGET_VAR,
                cycle=0,
            )
            sampled_df = result['next_batch_df']

        else:
            # No initial evidence: run initial random/fps/voronoi sampling
            candidates_df = pd.read_csv(EXP_DIR / 'dataset' / 'CANDIDATES.csv')
            X_candidates  = scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
            sampled_sub_idx = sample_landscape(
                X_landscape=X_candidates,
                n_points=INIT_BATCH,
                sampling_mode=INIT_SAMPLING,
            )
            sampled_df = candidates_df.iloc[sampled_sub_idx][SEARCH_VAR].copy()
            for t in TARGET_VAR:
                sampled_df[t] = np.nan
            sampled_df.to_csv(cycle_dir / 'output_sampled.csv', index=False)

            log = {
                'cycle':               0,
                'init_sampling_mode':  INIT_SAMPLING,
                'n_points':            INIT_BATCH,
                'sampled_idx':         [int(i) for i in sampled_sub_idx],
                'candidates_df_shape': list(candidates_df.shape),
            }
            save_to_json(log, cycle_dir / 'log.json', timestamp=False)

    # ── Cycle N > 0: state update then AL cycle ────────────────────────────

    else:
        dataset_dir = EXP_DIR / 'dataset'
        scaler      = joblib.load(dataset_dir / 'scaler.joblib')

        _state_update(
            experiment_dir=EXP_DIR,
            prev_cycle=cycle - 1,
            search_vars=SEARCH_VAR,
            target_vars=TARGET_VAR,
        )

        evidence_df  = pd.read_csv(dataset_dir / 'EVIDENCE.csv')
        pool_stored  = pd.read_csv(dataset_dir / 'POOL.csv')

        ml_model = setup_multi_property_ml_model(config, TARGET_VAR)
        result = run_single_al_cycle(
            pool_df=pool_stored,
            evidence_df=evidence_df,
            scaler=scaler,
            ml_model=ml_model,
            search_vars=SEARCH_VAR,
            target_vars=TARGET_VAR,
            acquisition_params=ACQUI_PARAMS,
            batch_selection_method=BATCH_SEL_METHOD,
            batch_selection_params=BATCH_SEL_PARAMS,
            penalization_params=PENALIZATION,
        )

        cycle_dir = EXP_DIR / f'cycle_{cycle}'
        create_strict_folder(str(cycle_dir))

        _save_cycle_outputs(
            cycle_dir=cycle_dir,
            result=result,
            pool_df=pool_stored,
            acqui_params=ACQUI_PARAMS,
            search_vars=SEARCH_VAR,
            target_vars=TARGET_VAR,
            cycle=cycle,
        )
        sampled_df = result['next_batch_df']

    # ── In-silico auto-validation ──────────────────────────────────────────

    if gt_df is not None:
        validated_path = cycle_dir / 'validated.csv'
        _auto_validate(
            gt_df=gt_df,
            sampled_df=sampled_df,
            search_vars=SEARCH_VAR,
            target_vars=TARGET_VAR,
            output_path=validated_path,
        )
        print(f'[in-silico] validated.csv written: {validated_path}')

    # ── Summary ────────────────────────────────────────────────────────────

    print(f'\nOutput: {cycle_dir / "output_sampled.csv"}')
    if gt_df is None:
        print(f'Next : fill in {cycle_dir / "validated.csv"} then re-run.')
