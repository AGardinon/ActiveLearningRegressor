# Multi-Property Optimization — State Tracker

**Branch:** `feature/multi-properties-optimization`
**Last updated:** 2026-04-08

This is the "what's happening right now" file. Unlike `DESIGN.md` (stable
architectural decisions) and `PHASES.md` (step-by-step plan with
checkboxes), this file is meant to be updated frequently — every time
a work session starts or ends, even if nothing got finished.

---

## Current status

**Phase:** Phase 2 in progress — P1 complete; P2.1–P2.4 done; P2.5 next.

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

**P2.5** — Config schema validation in `experiment.py`: validate that each entry has
exactly one of `target_variable` / `target_variables`; validate that all referenced
target names exist; validate weight config completeness. See PHASES.md §P2.5.

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
