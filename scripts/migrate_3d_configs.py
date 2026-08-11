#!

"""Migrate legacy benchmark run configs to the current ``config/`` directory schema.

Historical benchmark runs stored their configuration as a single flat
``config.yaml`` inside the run folder.  The current ``benchmark_gtlandscape.py``
expects three separate documents under ``<run>/config/``:

    config/benchmark_config.yaml            (-bc)
    config/model_config.yaml                (-mc)
    config/acquisition_mode_settings.yaml   (-acqmodes)

No ``target_function_config.yaml`` is produced: ground-truth-landscape runs read
their targets from ``ground_truth_file``, and the script takes no ``-tfc``.

Two legacy populations exist and are detected automatically:

``v1-flat``
    Oldest layout.  ``cycle_sampling`` at top level, ``kernel_recipe`` at top
    level, no ``SEED`` / ``batch_selection`` / ``adaptive_refinement``.

``v2-dir``
    Later layout.  Already carries ``SEED``, ``batch_selection`` and
    ``adaptive_refinement``, but still nests ``kernel_recipe`` inside
    ``model_parameters`` where the current code expects ``kernel``.

Both populations break against current ``main`` for two shared reasons:
``model_parameters.use_gridsearch`` is splatted into the ``GPR`` constructor,
and ``create_gpr_instance`` asserts on ``kernel`` rather than ``kernel_recipe``.

The original file is never deleted — it is renamed to ``config.original.yaml``
alongside the new directory so the provenance record survives verbatim.

Usage:
    python scripts/migrate_3d_configs.py benchmarks/3D              # dry run
    python scripts/migrate_3d_configs.py benchmarks/3D --apply
    python scripts/migrate_3d_configs.py benchmarks/3D --verify-only
"""

import sys
import yaml
import argparse
from pathlib import Path

# --------------------------------------------------------------------------------
# SCHEMA DEFINITION

ORIGINAL_BACKUP_NAME = 'config.original.yaml'

# Keys copied verbatim from the legacy config into benchmark_config.yaml.
BENCHMARK_KEYS = (
    'acquisition_protocol',
    'data_scaler',
    'scaler_params',
    'experiment_name',
    'experiment_notes',
    'ground_truth_file',
    'init_batch_size',
    'init_sampling',
    'n_cycles',
    'search_space_variables',
    'target_variables',
    'landscape_penalization',
)

# Keys the current pipeline expects but the v1-flat layout never wrote.
BENCHMARK_DEFAULTS = {
    'SEED': None,
    'adaptive_refinement': None,
    'experiment_evidence': None,
}

# Read by nothing in activereg/ or scripts/; dropped rather than carried over.
DROPPED_KEYS = ('ground_truth_parameters',)

# Default cadence for the grid-search block. Irrelevant while
# perform_grid_search is False, but the key is expected to exist.
GRID_SEARCH_EVERY_N_POINTS = 20


# --------------------------------------------------------------------------------
# TRANSLATION


def detect_schema(config: dict) -> str:
    """Identify which legacy layout a config belongs to.

    Args:
        config (dict): Parsed legacy ``config.yaml``.

    Returns:
        str: ``'v1-flat'`` or ``'v2-dir'``.
    """
    return 'v1-flat' if 'cycle_sampling' in config else 'v2-dir'


def resolve_percentile(config: dict) -> float:
    """Derive the global batch-selection percentile from per-entry values.

    The v1-flat layout stored ``percentile`` on each acquisition entry; the
    current schema stores a single value under
    ``batch_selection.method_params``.  Only the modes a protocol actually uses
    participate — entries that are defined but never scheduled (commonly
    ``maximum_predicted_value`` with ``percentile: Max``) are ignored.

    Args:
        config (dict): Parsed legacy ``config.yaml``.

    Returns:
        float: The single percentile shared by all scheduled modes.

    Raises:
        ValueError: If the scheduled modes disagree, or none declare a
            percentile. Both cases need a human decision rather than a guess.
    """
    protocol = config.get('acquisition_protocol') or {}
    scheduled = {
        mode
        for stage in protocol.values()
        for mode in (stage or {}).get('acquisition_modes', []) or []
    }
    percentiles = {
        entry.get('percentile')
        for entry in config.get('acquisition_parameters') or []
        if entry.get('acquisition_mode') in scheduled and 'percentile' in entry
    }
    if len(percentiles) != 1:
        raise ValueError(
            f"cannot derive a single batch-selection percentile: scheduled modes "
            f"{sorted(scheduled)} declare {percentiles or '{}'}. Resolve manually."
        )
    return percentiles.pop()


def translate(config: dict) -> dict[str, dict]:
    """Convert a legacy config into the three current-schema documents.

    Pure function — performs no I/O and does not mutate ``config``.

    Args:
        config (dict): Parsed legacy ``config.yaml``.

    Returns:
        dict[str, dict]: Mapping of output filename to document, containing
            ``benchmark_config.yaml``, ``model_config.yaml`` and
            ``acquisition_mode_settings.yaml``.

    Raises:
        ValueError: If a required key is absent or the percentile is ambiguous.
    """
    schema = detect_schema(config)

    # -- benchmark_config.yaml ---------------------------------------------
    benchmark = {key: config[key] for key in BENCHMARK_KEYS if key in config}
    for key, default in BENCHMARK_DEFAULTS.items():
        benchmark[key] = config.get(key, default)

    if schema == 'v1-flat':
        # cycle_sampling named only the spatial sampler; the enclosing strategy
        # was implicitly highest_landscape, and the percentile lived per-entry.
        benchmark['batch_selection'] = {
            'method': 'highest_landscape',
            'method_params': {
                'percentile': resolve_percentile(config),
                'sampling_method': config['cycle_sampling'],
            },
        }
    else:
        if 'batch_selection' not in config:
            raise ValueError("v2-dir config is missing 'batch_selection'")
        benchmark['batch_selection'] = config['batch_selection']

    # -- model_config.yaml -------------------------------------------------
    model_parameters = dict(config.get('model_parameters') or {})

    # use_gridsearch is not a GPR constructor argument; it is splatted into the
    # model and raises TypeError. Lift it into the grid_search block instead.
    use_gridsearch = bool(model_parameters.pop('use_gridsearch', False))

    # The kernel recipe lived at top level (v1-flat) or under model_parameters
    # (v2-dir); current code asserts on model_parameters['kernel'].
    kernel = model_parameters.pop('kernel_recipe', None) or config.get('kernel_recipe')
    if kernel is None:
        raise ValueError("no kernel recipe found in 'kernel_recipe' (top level or model_parameters)")
    model_parameters['kernel'] = kernel

    model = {
        'ml_model': config['ml_model'],
        'model_parameters': model_parameters,
        'grid_search': {
            'perform_grid_search': use_gridsearch,
            'every_n_points': GRID_SEARCH_EVERY_N_POINTS,
        },
    }

    # -- acquisition_mode_settings.yaml ------------------------------------
    # Per-entry 'percentile' keys are retained: AcquisitionFunction reads its
    # kwargs with .get(), so they are inert, and they preserve the only record
    # of how unscheduled modes were configured.
    acquisition = {'acquisition_parameters': config.get('acquisition_parameters') or []}

    return {
        'benchmark_config.yaml': benchmark,
        'model_config.yaml': model,
        'acquisition_mode_settings.yaml': acquisition,
    }


# --------------------------------------------------------------------------------
# VERIFICATION


def verify(config_dir: Path) -> None:
    """Assert a migrated run loads through the real pipeline entry points.

    Exercises the two failure modes the migration exists to fix: the ``kernel``
    assertion in ``create_gpr_instance`` and the ``**model_parameters`` splat
    into the model constructor. No AL cycles are run.

    Args:
        config_dir (Path): A run's ``config/`` directory.

    Raises:
        AssertionError: If a required document or key is missing.
        Exception: Whatever the pipeline raises on a malformed config.
    """
    from activereg.experiment import setup_experiment_variables, setup_ml_model

    def load(name):
        path = config_dir / name
        assert path.is_file(), f"missing {name}"
        return yaml.safe_load(path.read_text())

    benchmark = load('benchmark_config.yaml')
    model = load('model_config.yaml')
    acquisition = load('acquisition_mode_settings.yaml')

    setup_experiment_variables(benchmark)
    setup_ml_model(model['ml_model'], model['model_parameters'])

    assert benchmark.get('batch_selection'), "batch_selection missing or empty"
    assert acquisition.get('acquisition_parameters'), "acquisition_parameters missing or empty"


# --------------------------------------------------------------------------------
# DRIVER


def migrate_run(legacy_path: Path, apply: bool) -> str:
    """Migrate a single run folder.

    Args:
        legacy_path (Path): Path to the run's flat ``config.yaml``.
        apply (bool): Write to disk when True; otherwise translate only.

    Returns:
        str: One of ``'skipped'``, ``'ok'`` or ``'would-migrate'``.
    """
    run_dir = legacy_path.parent
    config_dir = run_dir / 'config'

    if config_dir.is_dir():
        return 'skipped'

    config = yaml.safe_load(legacy_path.read_text())
    documents = translate(config)

    if not apply:
        return 'would-migrate'

    config_dir.mkdir()
    for name, document in documents.items():
        (config_dir / name).write_text(
            yaml.safe_dump(document, sort_keys=True, default_flow_style=False)
        )
    legacy_path.rename(run_dir / ORIGINAL_BACKUP_NAME)
    return 'ok'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path,
                        help="Directory to search recursively for legacy config.yaml files")
    parser.add_argument("--apply", action='store_true',
                        help="Write the migration; without it the run is a dry run")
    parser.add_argument("--verify-only", action='store_true',
                        help="Skip migration and verify already-migrated runs")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory")
        return 2

    failures = []

    if not args.verify_only:
        legacy = sorted(args.root.rglob('config.yaml'))
        legacy = [p for p in legacy if p.parent.name != 'config']
        counts = {}
        for path in legacy:
            run = path.parent.relative_to(args.root)
            try:
                schema = detect_schema(yaml.safe_load(path.read_text()))
                status = migrate_run(path, args.apply)
            except Exception as exc:
                failures.append((run, f"{type(exc).__name__}: {exc}"))
                status, schema = 'FAILED', '?'
            counts[status] = counts.get(status, 0) + 1
            print(f"  {status:<14} {schema:<8} {run}")
        print(f"\n{len(legacy)} legacy configs: " +
              ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        if not args.apply:
            print("dry run — nothing written. Re-run with --apply.")

    if args.apply or args.verify_only:
        print("\nverifying migrated runs through the pipeline:")
        migrated = sorted(args.root.rglob('config/benchmark_config.yaml'))
        for path in migrated:
            run = path.parent.parent.relative_to(args.root)
            try:
                verify(path.parent)
                print(f"  PASS  {run}")
            except Exception as exc:
                failures.append((run, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {run}  {type(exc).__name__}: {exc}")
        print(f"\n{len(migrated)} verified, {len(failures)} failure(s)")

    if failures:
        print("\nfailures:")
        for run, message in failures:
            print(f"  {run}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
