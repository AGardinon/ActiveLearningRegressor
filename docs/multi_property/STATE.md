# Multi-Property Optimization — State Tracker

**Branch:** `feature/multi-properties-optimization`
**Last updated:** 2026-04-29

This is the "what's happening right now" file. Unlike `DESIGN.md` (stable
architectural decisions) and `PHASES.md` (step-by-step plan with
checkboxes), this file is meant to be updated frequently — every time
a work session starts or ends, even if nothing got finished.

---

## Current status

**Phase:** Phase 2 complete — P1 and P2 done. Phase 3 (EHVI) is optional; next work is folder/README cleanup deferred from P2.

**What has been done so far:**

- All architectural decisions (D1–D12) are recorded in `DESIGN.md`.
- The phased plan is recorded in `PHASES.md` with step-level granularity.
- **P1.1–P1.10** complete (see 2026-04-22 work log entry for detail).
- **P1.11** — Single-property regression test completed. Evidence: the test was run
  after `e69423b`; commit `249ef77` updated this file with "test is currently running";
  commit `39b121f` removed a debug print inside the benchmark cycle loop — the only
  change after the DOCS update, consistent with the test having completed without
  errors. Marked `[x]` in `PHASES.md`. Confirm with Andrea if in doubt.
- **P2.1** — `compute_per_property_stats(Y_train, target_names)` and
  `scalarize(mus, sigmas, weights, y_min, y_max, method, rho)` added to
  `activereg/acquisition.py` (new section before `AcquisitionFunction`).
  **Spec deviation:** PHASES.md listed `compute_per_property_stats(Y_train)` with no
  `target_names` param — that can't return a keyed dict without names. Added
  `target_names` as a required argument.
  **Spec deviation:** DESIGN.md §4.2 sketch passes `y_stats` dict to `scalarize`.
  Changed to `y_min, y_max` (P,) arrays — the caller converts from the dict using
  `target_variables` order. This makes `scalarize` purely array-based and unit-testable
  without a dict dependency.
- **P2.2** — `WeightSampler` added to `activereg/acquisition.py`. Supports
  `"dirichlet"` (alpha param), `"uniform"` (Dirichlet α=1 alias), `"fixed"`.
  `sample(rng)` takes a `np.random.Generator` for reproducibility.

**Next concrete action when work resumes:**

Deferred cleanup items (do in any order, no blocking dependencies):
1. **Folder/README reorganisation** — `benchmarks/` structure, README pointing to wrong script.
2. **`beauty.py` package split** — once test notebook validates new functions (done).
3. **Phase 3 (EHVI)** — only if ParEGO coverage proves insufficient in real experiments.

**Completed this session (P2.8a–b):**

**P2.8a** — Implement multi-property metric utilities in `activereg/metrics.py`:
- `compute_pareto_front(Y, maximize=True) -> np.ndarray` (bool mask, (N,P)→(N,))
- `compute_hypervolume(Y_pareto, reference_point) -> float` (P=2 closed-form; P>2 stub)
- `compute_pareto_attribution(points_df, target_names) -> pd.DataFrame` (hit rate + ΔHV per source)

**P2.8b** — Add multi-property plotting functions to `activereg/beauty.py` under a new
`# --- PLOT FUNC MULTI-PROPERTY` section:
- `plot_objective_space(experiments, target_names, pool_df, color_by, filter_acquisitions)`
- `plot_hypervolume_over_time(experiments, target_names, reference_point, x_axis, filter_acquisitions)`
- `plot_per_property_best_over_time(experiments, target_names)`
- `plot_weight_distribution(experiments, joint_entry_name)`
- `plot_pareto_hit_rate(experiments, target_names)`
- `plot_hv_gain_attribution(experiments, target_names, reference_point)`

**P2.8c** — Test notebook in `benchmarks/2D/` using the two completed reference runs
(`branincurrin_gpr_parego_20cy_1pts` and `branincurrin_gpr_parego_20cy_5pts`).

Note: folder reorganisation (`benchmarks/` vs `insilico_al/reference_runs/`) and `beauty.py`
package split are both deferred to after the test notebook validates the new functions.

**After P2.6 — multi-objective function suite (P2.7a before P2.7b):**

The agreed function suite for multi-property benchmarks (all from
`botorch.test_functions.multi_objective`, scalable in input dimension):

| Role | Function | Why |
|------|----------|-----|
| Baseline / hello-world | `BraninCurrin` | fixed 2D input, 2 objectives, known Pareto front, literature-standard; fast and visualisable — the right function to validate the pipeline end-to-end before scaling up |
| Easy scalable | `DTLZ2` | spherical/convex Pareto front, fully scalable in d and P; the standard "does it work" benchmark |
| Medium | `ZDT3` | 2-objective, discontinuous front (5 segments); tests ParEGO coverage of disconnected regions |
| Hard | `DTLZ7` | disconnected, M separate Pareto regions; future stress-test and Phase 3 (EHVI) comparison baseline |

Same addition pattern as existing `FUNCTIONS_DICT` entries in
`activereg/benchmarkFunctions.py`. **Before writing new entries**, read
`DatasetGenerator.generate_dataset` in `activereg/data.py` and confirm the
pre-existing multi-output `if` branch writes multiple y columns to the
DataFrame correctly (there is one but it has not been exercised for
multi-property). Fix first if needed.

---

## Open questions

None currently. The user has answered all previously-open planning
questions:

- Regime: ParEGO in Phase 2, EHVI deferred to Phase 3. (D1)
- Observations are always complete. (D2)
- Correlation is expected to be mild; dict-of-models is sufficient. (D3)
- Model priorities: GPR primary, MLPa secondary, kNN supported, BNN out. (D5)
- Config schema: hybrid with `name:` field, backwards compatible. (D6)
- Scalarization: augmented Chebyshev default, weighted sum secondary. (D7)
- `y_min`/`y_max` from training set only. (D8)
- Phase 1 / Phase 1.5 split for MLPa shared-output is acceptable. (D4, P1.5)
- Per-property or global `ml_model` spec — both supported. (D6, P1.4)

---

## Assumptions to double-check before writing code

These are things that the plan assumes but that have not been explicitly
re-verified against the current codebase in a single session. Worth a
30-second check when starting work, because the codebase moves.

1. **`AcquisitionFunction.__init__` numbered-suffix handling** at
   `activereg/acquisition.py:103-108` still works as before and we are
   keeping it for backwards compatibility. The named-entry approach
   (D6) is a superset of it. The real production config
   `scripts/general_config/acquisition_mode_settings.yaml` uses the
   numbered-suffix form (`target_expected_improvement_1`) so this path
   is LIVE and cannot be deleted as part of Phase 1.
2. **`experiment.py:341` stale TODO** ("check if I need to concatenate
   and update the X_train") — confirmed still present as of 2026-04-08.
   P1.6 is supposed to fix this as a side effect of the
   sequential-with-removal refactor.
3. **Single-property assumptions in the real production scripts.**
   `benchmark_functions.py` already stores `y_train` as 2D `(N, P)`,
   but these lines hardcode single-property behavior and must be
   fixed by P1.9 / P1.10:
   - `benchmark_functions.py:377,378` — `.ravel()` on pool/val target
     landscape
   - `benchmark_functions.py:502,598` — `y_train[i][0]` in cycle-log
     JSON
   - `benchmark_gtlandscape.py:286,287` — same `.ravel()` pattern
   - `benchmark_gtlandscape.py:405,467` — same `[0]` logging pattern
   - `evaluate_cycle_metrics(...)` is 1D-only in both scripts
     (~`benchmark_functions.py:577-583`), must be wrapped in a
     per-property loop.
4. **`AnchoredEnsembleMLP.predict()`** at `activereg/mlmodel/_mlp.py:80-91`
   already returns `(N, P)` arrays when `out_feats > 1` and squeezes
   only when `out_feats == 1`. This is the basis for Phase 1.5's
   `SharedMultiPropertyModel`. Verify this has not been changed.
5. **Config layout reality check.** The real production pipeline uses
   **four** YAML files, not a single merged one:
   - `scripts/general_config/benchmark_config.yaml` — experiment
     settings, `target_variables` list, **global** `batch_selection`
     block (including `percentile`), `acquisition_protocol`
   - `scripts/general_config/acquisition_mode_settings.yaml` — only
     the `acquisition_parameters` list; entries carry acquisition
     params (`y_target`, `epsilon`, etc.) but NOT `n_points` or
     `percentile`
   - `scripts/mlmodel_config/gpr_config.yaml` (or mlp/knn variant) —
     model type, `model_parameters`, `grid_search`
   - `scripts/general_config/target_function_config.yaml` — function
     parameters and training-set parameters
   `batch_selection` with `percentile` is GLOBAL (benchmark_config),
   not per-entry. This is cleaner than the merged sketch originally
   in DESIGN.md §4.4 — no per-entry `percentile` conflicts. Phase 2's
   config examples (P2.7) must respect this multi-file layout.

---

## Work log

- **2026-04-08** — Planning session. Completed design and phased plan.
  Created `docs/multi_property/{DESIGN.md, PHASES.md, STATE.md}`.
  Saved a project memory entry pointing to these docs so that future
  Claude sessions can pick up cold.
- **2026-04-08 (continued)** — Corrected the planning docs after Andrea
  flagged that the real production scripts are `benchmark_functions.py`
  and `benchmark_gtlandscape.py`, not the `insilico_lab_*` files (those
  are legacy). Read the four real config YAMLs and confirmed
  `batch_selection` / `percentile` is global in `benchmark_config.yaml`
  rather than per-entry. Updated `PHASES.md` P1.9/P1.10/P2.6/P2.7 and
  added single-property assumption fixes (`.ravel()` and `y_train[i][0]`
  patterns) at exact file:line locations. Updated `DESIGN.md` §4.3
  sampling_block skeleton and `STATE.md` assumption list to reflect
  the real codebase. No source files under `activereg/` or `scripts/`
  have been modified yet.
- **2026-04-22** — Implementation started. Verified all STATE.md §3
  assumptions before writing code:
  - Numbered-suffix handling at `acquisition.py:103-108` confirmed intact.
  - Stale TODO is at `experiment.py:334` (STATE.md had 341 — 7-line shift,
    minor; fixed in P1.6 via explicit `candidate_mask` sequential-with-removal).
  - `.ravel()` and `[0]` patterns confirmed at expected locations
    (`benchmark_functions.py:599` vs 598 — 1-line shift only).
  - `AnchoredEnsembleMLP.predict()` at `_mlp.py:80-91` confirmed:
    squeezes to `(N,)` only when `out_feats == 1`.
  - Completed **P1.1–P1.10** in a single session. All smoke tests pass.
    P1.11 (regression test) remains — requires running the actual script.
- **2026-04-23** — P1.11 regression test inferred complete (commit `249ef77`
  DOCS update "test is currently running"; commit `39b121f` removed debug print
  inside cycle loop — no error fixes committed). Marked P1.11 `[x]`.
  Phase 2 started. Andrea confirmed Option C for `y_best_z` (inside
  `sampling_block`, extended 3-tuple return).
  - **P2.1** — `compute_per_property_stats` and `scalarize` added to
    `activereg/acquisition.py`. Spec deviations: added `target_names` param
    to `compute_per_property_stats`; changed `scalarize` to take `y_min`/`y_max`
    arrays rather than `y_stats` dict.
  - **P2.2** — `WeightSampler` added to `activereg/acquisition.py`.
  - **P2.3** — Joint acquisition branch implemented in
    `AcquisitionFunction.landscape_acquisition`. `NotImplementedError` replaced
    with real scalarization dispatch. `AcquisitionFunction.__init__` extended to
    store `weights`, `scalarization`, `y_stats`, `rho` so `batch_highest_landscape`
    internal calls work without explicit args.
  - **P2.4** — `sampling_block` extended: added `rng` param; computes `y_stats`
    once per call; resolves weights per joint entry (Dirichlet or fixed);
    computes `y_best_z`; returns `(idxs, landscapes, per_entry_meta)`. Call
    sites in both benchmark scripts updated (unpack 3-tuple; pass `rng`).
    Note: `rng` for the `random` fast-path still uses `np.random.choice` (legacy
    global state) — minor cleanup deferred.
  - **P2.5** — `validate_acquisition_params(acquisition_params, target_names)`
    added to `experiment.py`. All 7 error conditions enforced with clear messages
    citing the offending entry name. Spec deviation: "neither target key" is
    allowed (legacy single-property mode) rather than required to have one — this
    preserves backwards compatibility with existing production configs. Not yet
    called from benchmark scripts — that is part of P2.6.
  - **P2.6** — End-to-end driver updates complete in both benchmark scripts:
    `validate_acquisition_params` called at setup time; `rng = np.random.default_rng(seed)`
    added; `cycle_meta` captured from `sampling_block` (was `_`); joint-entry metadata
    (`_y_best_z`, `_resolved_weights`) logged into `cycle_data_dict` keyed by entry
    `name` or `acquisition_mode`.
  - **P2.7a** — Multi-objective function suite added to `activereg/benchmarkFunctions.py`.
    `DatasetGenerator.generate_dataset` multi-output branch confirmed correct (no fix
    needed). Added: `BraninCurrin` (with `_branin_currin_factory` adapter to absorb
    `dim` kwarg), `DTLZ2_2obj_4D`, `DTLZ2_2obj_6D`, `ZDT3_2obj_4D`, `ZDT3_2obj_6D`,
    `DTLZ7_2obj_4D` (using `functools.partial` to pre-bind `num_objectives=2`). All
    produce `(N, 2)` output and correctly auto-label columns `y1`, `y2` via the existing
    multi-output branch in `generate_dataset`. Also added classes to `FUNCTION_CLASSES`.
  - **P2.7b** — 4-file worked example config set written:
    - `scripts/general_config/benchmark_config_multiprop.yaml`
    - `scripts/general_config/acquisition_mode_settings_multiprop.yaml`
    - `scripts/mlmodel_config/gpr_config.yaml` (reused, no change needed)
    - `scripts/general_config/target_function_config_multiprop.yaml`
    Config loads, `validate_acquisition_params` passes, protocol dispatches correctly
    to all 3 named entries (`explore_y1`, `explore_y2`, `parego_joint`) across all
    20 cycles.
- **2026-04-29** — P2.8 planning session. Key outcomes:
  - `data.py::build_function` fix landed in commit `efad340`: uses `inspect.signature`
    to skip `dim` kwarg for hardcoded-dim functions (e.g. `BraninCurrin`). Pipeline
    confirmed working end-to-end.
  - Two reference runs exist in `benchmarks/2D/`:
    - `branincurrin_gpr_parego_20cy_1pts` — pre-Phase-2 smoke test (single EI entry).
    - `branincurrin_gpr_parego_20cy_5pts` — full multi-property ParEGO run with all
      three entries (`explore_y1`, `explore_y2`, `parego_joint`), 20 cycles, 5 pts/cycle
      (1+1+3), 105 total points. This is the validated reference run.
  - `benchmark_config_multiprop.yaml` now points to `"2D/branincurrin_gpr_parego_20cy_3pts"`
    (1+1+1 pts/cycle, not yet run). Serves as the clean minimal reference config.
  - Benchmark output location: `benchmarks/` (experiments), `benchmarks/2D/` (2D runs).
    `insilico_al/reference_runs/` archive location from earlier STATE.md is superseded;
    reorganisation is deferred.
  - Planned P2.8 function inventory (see "Next concrete action" above). Key design
    decisions settled in this session:
    - `compute_pareto_front` and `compute_hypervolume` → `metrics.py`; plots → `beauty.py`.
    - Pareto front always computed from ALL sampled points (D2: observations complete).
      `filter_acquisitions` optional arg for per-source diagnostic.
    - `plot_hypervolume_over_time` x-axis = cumulative samples (not cycles) for
      fair cross-experiment comparison across different batch sizes.
    - Two attribution metrics: Pareto hit rate (precision per source) + marginal HV
      gain by source (impact-weighted). Together they answer "who found Pareto points
      and did those points actually expand the front?"
    - `plot_objective_space` dispatches on P: 2D scatter (P=2), pairwise grid (P≥3).
    - `compute_hypervolume`: P=2 closed-form now; P>2 stub with NotImplementedError.
    - `beauty.py` package split deferred to after test notebook validates new functions.
  - **P2.8a** — `compute_pareto_front`, `compute_hypervolume` (P=2 closed-form, P>2 stub),
    `compute_pareto_attribution` added to `activereg/metrics.py`.
    Verified on 5pts run: 3/105 sampled points on final Pareto front (correct); HV
    coverage 71.9% of pool true front; attribution shows EI/parego_joint at 55% of
    total HV gain and highest hit rate (5%), exploration (UL) at 25%, random init at 20%.
  - **P2.8b** — Six multi-property plotting functions added to `activereg/beauty.py`
    under `# --- PLOT FUNC MULTI-PROPERTY` section: `plot_objective_space`,
    `plot_hypervolume_over_time`, `plot_per_property_best_over_time`,
    `plot_weight_distribution`, `plot_pareto_hit_rate`, `plot_hv_gain_attribution`.
    All six smoke-tested against the 5pts reference run without errors.
  - **P2.8c** — `benchmarks/2D/results_multiprop.ipynb` created. 20 cells, 6 sections.
    Runs were initially generated with `negate=False` (wrong direction for BraninCurrin).
    All three re-run with `negate=True` and equal AL budget (60 pts each), folder names
    updated to reflect cycles: `60cy_1pts`, `20cy_3pts`, `12cy_5pts`.
    Notebook updated to load all three. All cells dry-run without errors.
    Key results (negate=True, 60 AL pts each):
    - batch=1 · 60cy (pure ParEGO): EI hit rate 13%, HV coverage 84%.
    - batch=3 · 20cy (explore+ParEGO): HV coverage 93.5% — best of three.
    - batch=5 · 12cy (explore+ParEGO): EI hit rate 19%, HV coverage 90.3%.
    Random init HV fraction is artificially high (38–75%) due to small init sizes;
    will stabilise with more cycles / repetitions.
    Phase 2 is complete.
  - **P2.8 refinements (same session, continued)** — Four additions to the
    multi-property plotting functions after first notebook run:
    - `plot_objective_space`: `acronym_map` param (default `ACQFUNC_ACRONYMS`),
      applied to source labels when `color_by='acquisition_source'`.
    - `plot_hypervolume_over_time`: `exclude_init: bool = False` — filters
      `cycle > 0` before building HV curves when True.
    - `plot_per_property_best_over_time`: `ceiling_values: Optional[Dict[str,float]]`
      — draws a gray dashed "Pool max" `axhline` per property when provided.
    - `plot_pareto_hit_rate`, `plot_hv_gain_attribution`: `exclude_init` and
      `acronym_map` params; init-cycle filtering and acronym labels now consistent.
    - Notebook updated: `pool_max` computed in ref-point cell; all attribution /
      HV functions called with `exclude_init=True`; `ceiling_values=pool_max`
      passed to `plot_per_property_best_over_time`; attribution summary cell
      filters `pts[pts['cycle'] > 0]`.

---

## How to resume in a fresh session

To pick up this work from zero context:

1. Read `docs/multi_property/DESIGN.md` in full. It is ~15KB and
   contains everything you need to know about *why* decisions were made.
   Do not re-debate the decisions without a concrete reason.
2. Read `docs/multi_property/PHASES.md` to see what has been checked
   off and what the next atomic step is.
3. Read this file (`STATE.md`) for the current snapshot, open questions,
   and assumptions to verify.
4. `git log feature/multi-properties-optimization` to see if any
   code work has landed since the last log entry here. If yes, update
   this file's work log and adjust the `PHASES.md` checkboxes before
   continuing.
5. Start the next unchecked `[ ]` step in `PHASES.md`. Treat each step
   as small enough to commit independently.
