# Multi-Property Optimization — Phased Plan

**Branch:** `feature/multi-properties-optimization`
**Last updated:** 2026-04-08
**Status:** No phase started yet.

This document is the step-by-step implementation plan. It is meant to be
checked and updated as work progresses. For architecture and decision
rationale, see `DESIGN.md`. For current progress tracking, see `STATE.md`.

**Checkbox convention:** `[ ]` = not started, `[~]` = in progress,
`[x]` = done, `[!]` = blocked.

---

## Phase 1 — Infrastructure (multi-property wrapper + sampling_block refactor)

**Goal:** Introduce the `MultiPropertyMLModel` interface, refactor
`sampling_block` and the acquisition dispatch so that the pipeline
internally always operates on a multi-property wrapper, and verify that
existing single-property configs still run end-to-end unchanged.

**No scalarization, no ParEGO, no new acquisition modes in this phase.**
Phase 1 is purely plumbing. It should be reviewable as a refactoring PR
that enables future work.

### P1.1 — Create `activereg/mlmodel/_multi_property.py` `[x]`

- Define `MultiPropertyMLModel` as a `Protocol` (or ABC) with:
  - `target_names: list[str]`
  - `train(X, Y) -> None` where `Y.shape == (N, P)`
  - `predict(X) -> (y_hat, mu, sigma)` all of shape `(N, P)`
  - `predict_property(X, name) -> (y_hat, mu, sigma)` all of shape `(N,)`
  - `__repr__`
- Define `IndependentMultiPropertyModel` implementing the protocol:
  - Constructor takes `models: dict[str, MLModel]`
  - `train(X, Y)` slices `Y[:, i]` and calls `models[name].train(X, y_i)`
    for each target
  - `predict(X)` stacks per-property outputs into `(N, P)` arrays
  - `predict_property(X, name)` is a direct delegate
- Define `wrap_single_property(model: MLModel, target_name: str)
  -> IndependentMultiPropertyModel`.
  This is a one-liner that constructs an `IndependentMultiPropertyModel`
  with a single target. Used by single-property driver scripts to keep
  `sampling_block` uniform.

### P1.2 — Update `activereg/mlmodel/__init__.py` `[x]`

Re-export `MultiPropertyMLModel`, `IndependentMultiPropertyModel`,
`wrap_single_property`. Do not touch existing re-exports.

### P1.3 — Refactor `setup_experiment_variables` `[x]`

File: `activereg/experiment.py`.

- Parse `target_variables` as the authoritative list of target names.
  Already exists in the current code as a config key; it just needs
  to be used downstream.
- No functional change to the return tuple for now; document that
  `target_vars` is now guaranteed to be a non-empty list.

### P1.4 — Add `setup_multi_property_ml_model` factory `[x]`

File: `activereg/experiment.py`.

- New function: `setup_multi_property_ml_model(config, target_names)
  -> MultiPropertyMLModel`.
- Two paths:
  - **Global spec path**: `config["ml_model"]` is a string, and
    `config["model_parameters"]` applies to all targets. Build P
    independent instances of the same model type, wrap in
    `IndependentMultiPropertyModel`.
  - **Per-property spec path**: `config["ml_model"]` is a dict keyed
    by target name. For each target, read that target's model type
    and parameters, instantiate, collect into a dict, wrap.
- Dispatch to the existing `create_gpr_instance`,
  `create_knn_instance`, `create_anchored_ensemble_mlp` helpers for
  each per-property instance. Do not touch the factory functions.
- For single-property experiments, the global-spec path produces a
  `IndependentMultiPropertyModel` with a single entry — behaves
  identically to the old single-property path upstream, modulo the
  wrapper.

### P1.5 — Update `AcquisitionFunction.landscape_acquisition` `[x]`

File: `activereg/acquisition.py`.

- Change signature to accept a `MultiPropertyMLModel` and optional
  `target_variable` / `target_variables` / `weights` / `scalarization`
  parameters. See `DESIGN.md` §4.2.
- Phase 1 only wires the **per-property** branch (via
  `predict_property`). The **joint** branch (`target_variables` +
  scalarize) raises `NotImplementedError` with a clear message
  pointing to Phase 2. This is fine: Phase 1 only needs to keep
  single-property experiments working, and those go through the
  per-property branch with `target_variable = target_names[0]` under
  the wrapper.
- Existing single-property formulas stay untouched.

### P1.6 — Refactor `sampling_block` `[x]`

File: `activereg/experiment.py`.

- Change signature to accept `Y_train: (N, P)` and
  `ml_model: MultiPropertyMLModel`.
- Internally: extract `y_train_for_best` as the per-target y_best
  dictionary (per-property for per-property entries; joint handled
  in Phase 2).
- Implement sequential-with-removal via a `candidate_mask` that shrinks
  across entries. Carefully handle index remapping between the active
  sub-pool and the full pool when selecting and logging landscapes.
- Pad per-entry landscape outputs to full candidate length (so the
  stacked `landscape_list` shape stays `(n_entries, M)`).
- Fix the stale TODO at the former `experiment.py:341` as part of
  this refactor (the candidate-removal logic is now explicit).
- Preserve the `random` and `maximum_predicted_value` fast-paths but
  route them through the same candidate_mask mechanism.

### P1.7 — Update `AcquisitionParametersGenerator` for named entries `[x]`

File: `activereg/experiment.py`.

- Add an `_entry_identifier(entry)` helper returning
  `entry.get("name") or entry["acquisition_mode"]`.
- Change the protocol lookup (`_protocol_params_for_cycle`) to match
  stage entries by identifier instead of by `acquisition_mode` string.
  Existing single-property configs (no `name:` field) still match by
  `acquisition_mode` thanks to the fallback.
- Keep the legacy numbered-suffix handling in
  `AcquisitionFunction.__init__` (`acquisition.py:103-108`) for now.
  It still works and removing it is a separate cleanup.

### P1.8 — Decide `landscape_sanity_check` policy `[ ]`

File: `activereg/acquisition.py`.

- Keep the strict 1D guard. Multi-property scalarization happens
  **before** this check; the check guarantees that anything reaching
  the batch selector is 1D.
- Add a docstring note explaining the guarantee.

### P1.9 — Update `scripts/benchmark_functions.py` `[ ]`

This is the **main production benchmark script**. It takes 4 configs via
CLI (`-bc` benchmark, `-mc` model, `-acqmodes` acquisition,
`-tfc` target_function) and runs adaptive-refinement experiments.

- Construct the multi-property wrapper once via
  `setup_multi_property_ml_model(model_config, TARGET_VAR)` at the
  start of each repetition; training happens in-place across cycles.
- `y_train` is already 2D `(N, P)` in this script via
  `train_df[TARGET_VAR].to_numpy()` when `TARGET_VAR` is a list.
  Rename the local to `Y_train` for clarity and confirm it is 2D
  (add an explicit `.reshape(-1, len(TARGET_VAR))` guard).
- Fix the single-property `.ravel()` assumptions:
  - `benchmark_functions.py:377`:
    `pool_target_landscape = pool_df[TARGET_VAR].to_numpy().ravel()`
  - `benchmark_functions.py:378`:
    `val_target_landscape = val_df[TARGET_VAR].to_numpy().ravel()`
  These must become per-property 2D arrays. Downstream metric/logging
  code that consumes them needs to loop over properties.
- Fix the single-property cycle-log assumptions:
  - `benchmark_functions.py:502`: `"y_value": y_train[i][0]`
  - `benchmark_functions.py:598`: `"y_value": y_candidates[sampled_indexes][i][0]`
  Both hardcode `[0]` (first target). Replace with a per-property dict
  `{name: y_train[i][j] for j, name in enumerate(TARGET_VAR)}` or
  similar.
- `evaluate_cycle_metrics(...)` (~line 577–583) is 1D-only. Wrap it in a
  per-property loop and store the results keyed by target name.
- Pass `Y_train` and the `MultiPropertyMLModel` into `sampling_block`.
  The existing call already passes `batch_selection_method` and
  `batch_selection_params` explicitly — no change to that contract.
- No change to the dataframe I/O, the validation block, or the
  adaptive refinement logic.

### P1.10 — Update `scripts/benchmark_gtlandscape.py` `[ ]`

Variant of `benchmark_functions.py` that reads a CSV ground truth
instead of calling a known benchmark function. Takes 3 configs
(`-bc`, `-mc`, `-acqmodes`).

- Same wrapper construction and `Y_train` reshape as P1.9.
- Fix the same single-property assumptions:
  - `benchmark_gtlandscape.py:286`: `.ravel()` on pool target
  - `benchmark_gtlandscape.py:287`: `.ravel()` on val target
  - `benchmark_gtlandscape.py:405`: `"y_value": y_train[i][0]`
  - `benchmark_gtlandscape.py:467`: `"y_value": y_candidates[...][i][0]`
- Wrap `evaluate_cycle_metrics` in a per-property loop.

**Out of scope (legacy, do not touch):** `scripts/insilico_lab_*.py`
and `scripts/run_labAL_cycle.py`. These predate the current production
pipeline and will not be migrated in this branch.

### P1.11 — Regression test: single-property configs still run `[ ]`

- Pick an existing single-property Ackley 6D config
  (`ak6d_*` in `scripts/`) and run it end-to-end via
  `benchmark_functions.py` after Phase 1.
- Compare the sampled indices (or at least the first couple of
  cycles) against a pre-Phase-1 run to confirm no behavioral drift
  for single-property cases.
- If there is any drift, it must be root-caused and fixed, not
  shrugged off. The Phase 1 refactor is explicitly supposed to be
  behavior-preserving for single-property experiments.

### Phase 1 validation criteria

- An existing single-property Ackley 6D config runs end-to-end via
  `benchmark_functions.py` and produces the same sampled points as
  the pre-refactor code.
- At least one multi-property config set (can be a mock with two
  properties derived from the same Ackley surface at different
  scales/shifts) loads, runs the per-property branch, and produces
  sensible cycle outputs **using only per-property acquisition entries**
  (no joint entries yet — those come in Phase 2).
- `IndependentMultiPropertyModel` is callable with any mix of GPR,
  kNN, and single-output MLPa.

---

## Phase 1.5 — Shared-output MLPa backend (deferred, optional)

**Goal:** Add `SharedMultiPropertyModel` to eliminate the P-fold
training cost when using `AnchoredEnsembleMLP` with multiple properties.

**Trigger:** Implement this only when MLPa training time in
`IndependentMultiPropertyModel` mode becomes an observable pain point,
or when the user explicitly wants to run MLPa-primary experiments.

### P1.5.1 — Implement `SharedMultiPropertyModel` `[ ]`

File: `activereg/mlmodel/_multi_property.py`.

- Constructor takes a single `MLModel` instance configured with
  `out_feats = P`.
- `train(X, Y)` passes `Y` directly (no slicing).
- `predict(X)` calls the underlying model and returns the `(N, P)`
  outputs directly (the model already returns this shape for
  `out_feats > 1`).
- `predict_property(X, name)` predicts once and slices the column
  corresponding to `name`.
- `target_names` list determines the column order at construction time.

### P1.5.2 — Extend `setup_multi_property_ml_model` `[ ]`

- Detect when the model type is `AnchoredEnsembleMLP` **and** the
  config is a global spec **and** `len(target_names) > 1`.
- In that case, construct a single `AnchoredEnsembleMLP` with
  `out_feats = len(target_names)` and wrap in
  `SharedMultiPropertyModel`.
- All other cases continue to use `IndependentMultiPropertyModel`.
- Per-property specs always use `IndependentMultiPropertyModel`,
  even if all properties happen to request MLPa — mixing at the
  wrapper level is the simpler rule.

### P1.5.3 — Verify parity of interface `[ ]`

- Run the same Phase 1 regression tests but switch the MLPa config
  between dict-of-MLPa and shared-output MLPa; confirm that the
  downstream `sampling_block`, acquisition, and batch selection
  behave identically modulo the underlying model choice.

---

## Phase 2 — ParEGO scalarization (the real multi-property AL)

**Goal:** Implement augmented Chebyshev scalarization with per-cycle
random weight sampling, plus a generic "scalarized" acquisition path
that reuses the existing single-property formulas.

### P2.1 — Add scalarization utilities `[ ]`

File: `activereg/acquisition.py` (or new `activereg/_scalarization.py`
if it gets too big).

- `compute_per_property_stats(Y_train) -> dict[str, (y_min, y_max)]`
- `scalarize(mus, sigmas, weights, method="augmented_chebyshev",
              y_stats, rho=0.05) -> (mu_z, sigma_z)` for 1D mu, sigma
  arrays.
- Support `method in ("augmented_chebyshev", "weighted_sum")`.
- Augmented Chebyshev uses the subgradient approach described in
  `DESIGN.md` §D9 for `sigma_z`.

### P2.2 — Add `WeightSampler` `[ ]`

File: `activereg/acquisition.py`.

- Small class supporting:
  - `"dirichlet"` with `alpha` parameter (default uniform Dirichlet)
  - `"uniform"` (uniform on the simplex — Dirichlet with alpha=1)
  - `"fixed"` wrapping a user-supplied weight vector
- Method: `sample(rng) -> np.ndarray`.
- Seeded via a `np.random.Generator` so runs are reproducible.

### P2.3 — Wire joint acquisition entries through the dispatcher `[ ]`

File: `activereg/acquisition.py`.

- Remove the `NotImplementedError` from Phase 1 in the joint branch of
  `landscape_acquisition`.
- Implement the joint path as described in `DESIGN.md` §4.2:
  extract `(mus, sigmas)` for the requested `target_variables`,
  apply `scalarize`, feed the resulting `(mu_z, sigma_z)` into the
  existing single-property formula.
- The single-property formulas stay untouched.

### P2.4 — Per-cycle weight resolution in `sampling_block` `[ ]`

File: `activereg/experiment.py`.

- For each acquisition entry with a `weight_sampler`, sample fresh
  weights at the start of the cycle and attach them to the entry
  (e.g., as `entry["_resolved_weights"]`).
- Compute `y_stats` from `Y_train` once per cycle.
- Compute `y_best_z` per joint entry using the same weights — this
  is the reference scalar for EI-style formulas on scalarized targets.
- For per-property entries, `y_best` stays per-property (as before).
- Log the sampled weights into the cycle log JSON so experiments
  are reproducible and postmortem-able.

### P2.5 — Config schema validation `[ ]`

File: `activereg/experiment.py` (or a new `_config.py`).

- Validate that every acquisition entry has exactly one of
  `target_variable` or `target_variables`, never both.
- Validate that all referenced target names exist in
  `target_variables`.
- Validate that joint entries have either `weight_sampler` or
  `weights`, never both.
- Validate that `len(weights) == len(target_variables)` when fixed.
- Clear error messages that cite the offending entry `name`.

### P2.6 — End-to-end driver updates `[ ]`

- `scripts/benchmark_functions.py`: extend the per-cycle loop to
  handle multi-property logging (per-target `y_best`, per-entry
  resolved weights, per-entry landscape labeling).
- `scripts/benchmark_gtlandscape.py`: same.

### P2.7 — Worked example configs `[ ]`

Add a fully worked **multi-file** config set (matching the real
`benchmark_functions.py` contract — 4 YAML files):

- `scripts/general_config/benchmark_config_multiprop.yaml` with
  `target_variables: [y1, y2]`, `batch_selection` block (global
  percentile), and an `acquisition_protocol` referencing
  `name:`-identified entries.
- `scripts/general_config/acquisition_mode_settings_multiprop.yaml`
  with at least one per-property uncertainty entry (e.g.,
  `name: explore_y1`, `target_variable: y1`) and one joint ParEGO
  entry (e.g., `name: parego_joint`, `target_variables: [y1, y2]`,
  `weight_sampler: dirichlet`, `scalarization: augmented_chebyshev`).
- `scripts/mlmodel_config/gpr_config.yaml` — unchanged (global spec
  path, applied to both properties).
- `scripts/general_config/target_function_config_multiprop.yaml`
  using a 2D benchmark (e.g., two Ackley surfaces with different
  centers, or Ackley + Rastrigin as the two properties).

### P2.8 — Reference run `[ ]`

- Run the worked-example config for ~15-20 cycles.
- Produce a scatter of the Pareto front on the benchmark to
  visually verify ParEGO is sweeping trade-offs (not collapsing to
  one corner).
- Archive the run and the plot under
  `insilico_al/reference_runs/multi_property_parego/` for future
  regression comparison.

### Phase 2 validation criteria

- Augmented Chebyshev scalarization is the default path for joint
  entries and reproduces ParEGO-like behavior on the worked example.
- Weight sampler produces reproducible weights given a seed.
- The Pareto front scatter from the reference run shows spread
  along the front (not clustering at a single weighted-sum optimum).
- Single-property configs still run unchanged (continued Phase 1
  regression).

---

## Phase 3 — (Optional) Single-shot EHVI with `highest_landscape`

**Goal:** Add a true Pareto-aware acquisition mode for cases where
ParEGO coverage is not enough. **Only pursued if Phase 2 reveals a
concrete need.**

**Compatibility caveat:** The only variant compatible with the
`highest_landscape` batch strategy is **single-shot EHVI** — compute
EHVI once over the candidate pool, take the top percentile, diversify
geometrically. Sequential greedy EHVI is incompatible with
`highest_landscape` and is out of scope. See `DESIGN.md` §D1, §D10.

### P3.1 — Pareto front utility `[ ]`

Small `pareto_front(Y_train: (N, P)) -> np.ndarray` returning the
indices (or mask) of non-dominated points.

### P3.2 — 2D closed-form EHVI `[ ]`

Implement Emmerich 2006 closed-form EHVI for P=2. Self-contained,
no new dependencies.

### P3.3 — EHVI acquisition mode `[ ]`

Add `"expected_hypervolume_improvement"` mode to `AcquisitionFunction`.
Produces a 1D landscape of EHVI values per candidate. Plugs directly
into `batch_highest_landscape`.

### P3.4 — (Optional) BoTorch backend for P >= 3 `[ ]`

If 3+ properties are actually needed, bring in BoTorch for its EHVI
implementation. This is a non-trivial dependency addition and should
be weighed seriously. Ignore unless demanded by experimental need.

### Phase 3 validation criteria

- On a benchmark with a known Pareto front, single-shot EHVI
  achieves hypervolume convergence comparable to ParEGO for 2
  properties.
- No regressions in single-property or ParEGO paths.

---

## Phase 4 — Joint qEHVI (on hold, unlikely)

**Status:** Not planned. Would require a batch-level acquisition API
where `batch_acquire` is itself the optimization target, not a wrapper
around a 1D landscape. Only pursue if Phase 3 is demonstrably insufficient,
which is unlikely for the kinds of experiments in the user's pipeline.

---

## Cross-cutting work (any phase)

These are not tied to a specific phase and should be picked up
opportunistically when touching the relevant code.

- `[ ]` Remove or consolidate the legacy numbered-suffix acquisition
  naming (`target_expected_improvement_800`) once configs migrate to
  the `name:` field.
- `[ ]` Add a small unit test suite for
  `IndependentMultiPropertyModel` covering shape contracts.
- `[ ]` Add unit tests for `scalarize` (augmented Chebyshev and
  weighted sum) with known small inputs.
- `[ ]` Add a test that `sampling_block` sequential-with-removal
  never selects duplicate indices across entries.
