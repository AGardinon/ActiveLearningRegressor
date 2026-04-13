# Multi-Property Optimization — Design

**Branch:** `feature/multi-properties-optimization`
**Author:** Andrea Gardin (with Claude Code planning assistance)
**Last updated:** 2026-04-08
**Status:** Planning complete. No code changes yet.

This document captures the architectural decisions and rationale for adding
multi-property optimization to the `activereg` batch pool-based active
learning framework. It is intentionally stable: once a decision is recorded
here, it should not be re-litigated without being crossed out with a reason.

For the step-by-step implementation plan, see `PHASES.md`.
For current progress and next actions, see `STATE.md`.

---

## 1. Purpose

Enable active learning campaigns where several target properties are
optimized jointly. Example use cases from the lab:

- viscosity and diameter of a coacervate
- uptake of a cargo and size of the particle

The scientific goal is to explore **trade-offs** between properties, not to
collapse them to a single fixed-weight scalar. Properties are expected to
be mildly related (physically/chemically) but not tightly coupled —
strongly correlated properties are actively avoided because they make
multi-objective optimization redundant.

---

## 2. Scope

### In scope

- Pool-based AL (candidate set is a finite dataframe)
- Dict of independent single-output models, one per target property
- ParEGO-style scalarized Bayesian optimization with random weight sampling
  and augmented Chebyshev scalarization
- Backwards compatibility with existing single-property configs and
  experiment scripts
- `highest_landscape` batch selection (the only batch strategy in practical
  use in current experiments)

### Out of scope (for now)

- Synthesis-based AL (continuous-space optimization of the acquisition).
  This is a separate axis, orthogonal to multi-property. Pool-based
  multi-objective BO is standard in the literature.
- Multi-task GP coregionalization (ICM/LCM kernels).
- True joint qEHVI — a batch-level acquisition that jointly optimizes the
  full batch. Would require a batch-level acquisition API, not in plan.
- Partial observations (some properties measured, others missing for the
  same point). Assumption D2 below forbids this.
- `BayesianNN` as a multi-property backend. Excluded for now because of
  pre-existing training issues unrelated to multi-property work.

---

## 3. Design decisions

### D1. Optimization regime: ParEGO-first, EHVI deferred

**Decision:** Phase 2 implements ParEGO (random-weight augmented Chebyshev
scalarization per cycle). Phase 3 (single-shot EHVI compatible with
`highest_landscape`) is optional and only pursued if ParEGO's Pareto
coverage proves insufficient in practice.

**Rationale:** For 2–3 properties, ParEGO is competitive with EHVI in
hypervolume convergence and requires **zero** changes to the batch
selection machinery. It also avoids dependency on BoTorch for the initial
implementation. Sequential greedy EHVI is structurally incompatible with
`highest_landscape` (it wants to update a hallucinated Pareto front
between picks, while `highest_landscape` computes the landscape once and
diversifies geometrically), so real EHVI in Phase 3 would be the
"single-shot" variant — good enough for typical cases, a known
approximation for pathological Pareto topologies.

### D2. Observations are complete

**Decision:** Every sampled point has measurements for all target properties.
`Y_train` is always a dense `(N, P)` numpy array. No masking logic anywhere.

**Rationale:** Matches the user's actual lab and benchmark workflows.
Partial observations is one of the largest sources of API ugliness in
multi-property AL libraries; being able to exclude it keeps the wrapper
interface simple.

### D3. Property correlation is assumed mild

**Decision:** Dict of independent single-output models is sufficient.
No multi-task GP coregionalization.

**Rationale:** Strongly correlated properties make multi-objective
optimization redundant and are avoided by design. Even if modest
correlations exist, the extra machinery (ICM kernel, coregionalization
matrix hyperparameters, GPyTorch dependency) is not worth it for the
expected AL data scales. `sklearn`'s `GaussianProcessRegressor` does not
do true multi-output in any meaningful sense anyway, so "independent
models" and "sklearn multi-output GPR" are implementation-equivalent.

### D4. Two model backends behind one interface

**Decision:** Define a `MultiPropertyMLModel` protocol with two
implementations:

1. **`IndependentMultiPropertyModel`** — `dict[target_name, MLModel]`,
   one single-output model per property. **Implemented in Phase 1.**
2. **`SharedMultiPropertyModel`** — a single model instance with
   `out_feats = P`, slices outputs by property. **Deferred to Phase 1.5.**
   Used for `AnchoredEnsembleMLP` in shared-output mode to avoid the
   P-fold training cost of running independent anchored ensembles.

The public interface of `MultiPropertyMLModel` is designed from day one
to accommodate both backends without changes to callers.

**Rationale:** `sklearn` GPR has no benefit from shared representation;
dict-of-models is as good as it gets. `AnchoredEnsembleMLP`, on the
other hand, already accepts `out_feats > 1` and produces `(N, P)`
outputs — running one ensemble with P outputs is P× cheaper than
running P ensembles with 1 output, and it keeps the shared-hidden-layer
inductive bias. Both backends are worth supporting, but Phase 1 starts
with the universally-applicable one so the infrastructure stabilizes
before the MLPa-specific optimization lands.

### D5. Model family priorities

| Family | Status |
|---|---|
| `GPR` | **Primary.** BO baseline. Dict-of-models only. |
| `AnchoredEnsembleMLP` | **Secondary.** Dict-of-models in Phase 1; shared-output in Phase 1.5. Will ultimately be competitive with GPR for some campaigns. |
| `kNNRegressorAL` | Supported in dict-of-models at zero cost. Proof-of-concept only. |
| `BayesianNN` | Out of scope for now (unresolved training issues). |

### D6. Config schema — backwards compatible

**Decision:** Extend the existing YAML schema without breaking existing
single-property configs. The two main additions are:

1. **Per-property `ml_model` spec** (optional). If `ml_model` is a dict
   keyed by target name, each property can use a different model type and
   different parameters (including mixed model families). If `ml_model` is
   a single spec (current form), it applies to all properties uniformly.

2. **Named acquisition entries.** Each entry in `acquisition_parameters`
   may optionally have a `name` field. If omitted, the name defaults to
   the `acquisition_mode` string (matches existing behavior). The
   `acquisition_protocol` stages reference entries by `name` (or by
   `acquisition_mode` if `name` is absent, preserving backwards
   compatibility). This replaces the current "numbered suffix" workaround
   (`target_expected_improvement_800` / `_1200`) with an explicit and
   cleaner disambiguation.

Per-property acquisition entries carry `target_variable` (singular).
Joint acquisition entries carry `target_variables` (plural) and optionally
`weight_sampler` (ParEGO-style) or `weights` (fixed). A dispatcher in the
sampling loop routes each entry appropriately.

### D7. Scalarization: augmented Chebyshev by default

**Decision:** Augmented Chebyshev is the default scalarization method for
joint acquisition entries. Weighted sum is available as a secondary
`scalarization: "weighted_sum"` option.

**Formula (augmented Chebyshev):**

```
mu_tilde_i(x) = (mu_i(x) - y_min_i) / (y_max_i - y_min_i + eps)
z(x) = max_i [ w_i * mu_tilde_i(x) ] + rho * sum_i [ w_i * mu_tilde_i(x) ]
```

with `rho = 0.05` default, `eps = 1e-12`.

**Rationale:** Plain weighted sums fail to cover non-convex Pareto fronts;
augmented Chebyshev is the formulation that gives ParEGO its Pareto-front
coverage property. Making the cleaner option the default matters because
most users will not question the default.

### D8. Per-property normalization from training set

**Decision:** `y_min_i` / `y_max_i` come from the current training set at
each cycle. They update naturally as the AL campaign grows. No fixed
pre-declared ranges are required in config.

**Rationale:** Standard ParEGO. Avoids demanding prior knowledge of
target ranges. Works equally well for benchmark and lab experiments.

### D9. Scalarized uncertainty under independence

**Decision:** For dict-of-independent-models and scalarization
`z = f(mu_1, ..., mu_P)`, the scalar uncertainty is:

```
sigma_z(x) = sqrt( sum_i (d_f/d_mu_i)^2 * sigma_i^2(x) )
```

For weighted sum, `d_f/d_mu_i = w_i`. For augmented Chebyshev, the
gradient is subdifferential; we use the gradient of the active argmax
term plus the `rho`-weighted linear term as a smooth surrogate.

**Rationale:** Delta method, sound under independence. For correlated
models (not our case), the covariance matrix would be needed.

### D10. Batch selection stays `highest_landscape`

**Decision:** All scalarized acquisition entries return a 1D landscape
(as today). `batch_highest_landscape` is **not** modified. The other
batch strategies (`constant_liar`, `kriging_believer`, `local_penalization`)
remain available in the codebase but are not exercised by multi-property
experiments in the current scope.

**Rationale:** Interpretability. The user explicitly values the
"compute landscape once, geometrically diversify" philosophy and has
committed to `highest_landscape` as the production batch strategy.

### D11. Sequential-with-removal across entries

**Decision:** When a cycle contains multiple acquisition entries
(e.g., per-property exploration plus a joint ParEGO entry), entries run
sequentially. Indices selected by entry k are removed from the candidate
pool before entry k+1 runs. Order of entries in the config is the order
of application; order-dependence is documented behavior.

**Rationale:** Minimal extension that prevents duplicate selection without
breaking the `highest_landscape` philosophy. Also fixes the existing
stale `TODO` at `activereg/experiment.py:341`.

### D12. Single unified `sampling_block`

**Decision:** Refactor the existing `sampling_block` in `experiment.py`
rather than introduce a parallel multi-property version. The refactored
block always operates on a `MultiPropertyMLModel`. Single-property
experiments use a `wrap_single_property(model, target_name)` adapter
at the top of driver scripts, so there is only one code path internally.

**Rationale:** Two parallel control flows would drift. The current
`sampling_block` is already at the point where a refactor would be
worthwhile on its own (nested fast-paths, stale TODO), so the
multi-property refactor piggybacks on a cleanup that is overdue.

---

## 4. Architecture sketch

### 4.1 Model interface

```
MultiPropertyMLModel  (Protocol, new in activereg/mlmodel/_multi_property.py)
    target_names: list[str]
    train(X: (N, d), Y: (N, P)) -> None
    predict(X: (N, d)) -> (y_hat: (N, P), mu: (N, P), sigma: (N, P))
    predict_property(X, name) -> (y_hat: (N,), mu: (N,), sigma: (N,))

    |
    +-- IndependentMultiPropertyModel        [Phase 1]
    |       _models: dict[str, MLModel]      # one MLModel per target
    |
    +-- SharedMultiPropertyModel             [Phase 1.5 — deferred]
            _model: MLModel                  # single instance, out_feats = P

wrap_single_property(model: MLModel, target_name: str) -> MultiPropertyMLModel
    # Thin adapter: single-property code uses the same interface as
    # multi-property, keeping the sampling_block signature uniform.
```

### 4.2 Acquisition dispatch

```
AcquisitionFunction.landscape_acquisition(
    X_candidates,
    ml_model: MultiPropertyMLModel,
    target_variable: str | None = None,
    target_variables: list[str] | None = None,
    weights: np.ndarray | None = None,
    scalarization: str = "augmented_chebyshev",
    y_stats: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray  # 1D

  if target_variable is not None:
      # Per-property branch: unchanged single-property formula.
      _, mu, sigma = ml_model.predict_property(X_candidates, target_variable)
      return existing_single_property_formula(mu, sigma, self.y_best, ...)

  if target_variables is not None:
      # Joint branch: scalarize then apply single-property formula.
      _, mus, sigmas = ml_model.predict(X_candidates)   # (N, P)
      # Slice to the requested subset of properties
      mus, sigmas = select_targets(mus, sigmas, target_variables)
      mu_z, sigma_z = scalarize(mus, sigmas, weights, method=scalarization,
                                y_stats=y_stats)
      return existing_single_property_formula(mu_z, sigma_z, self.y_best_z, ...)
```

### 4.3 sampling_block skeleton

The current production signature (from `scripts/benchmark_functions.py`
and `scripts/benchmark_gtlandscape.py`) already takes `batch_selection_method`
and `batch_selection_params` as explicit arguments. Phase 1 keeps that
contract and adds multi-property support by swapping `y_train` for
`Y_train: (N, P)` and `ml_model` for a `MultiPropertyMLModel`.

```
sampling_block(
    X_candidates: (M, d),
    X_train: (N, d),
    Y_train: (N, P),                          # was y_train, now 2D
    ml_model: MultiPropertyMLModel,           # was MLModel
    acquisition_params: list[dict],           # named 'acquisition_params' in real signature
    batch_selection_method: str,              # already explicit in current code
    batch_selection_params: dict,             # already explicit in current code
    penalization_params: tuple[float, float] | None = (0.25, 1.0),
    rng: np.random.Generator | None = None,
) -> (list[int], np.ndarray)

    candidate_mask = np.ones(M, dtype=bool)
    sampled_new_idx = []
    landscape_list = []
    # Per-cycle weight sampling happens outside sampling_block,
    # weights are already embedded in each entry when this is called.

    y_stats = compute_per_property_stats(Y_train)    # D8

    for entry in acquisition_entries:
        X_sub = X_candidates[candidate_mask]
        idx_map = np.flatnonzero(candidate_mask)

        if entry["acquisition_mode"] == "random":
            <random fast-path over X_sub>
        elif entry["acquisition_mode"] == "maximum_predicted_value":
            <MPV fast-path over X_sub>
        else:
            acqui_func = AcquisitionFunction(...per entry...)
            landscape = acqui_func.landscape_acquisition(
                X_sub, ml_model,
                target_variable=entry.get("target_variable"),
                target_variables=entry.get("target_variables"),
                weights=entry.get("_resolved_weights"),  # ParEGO-sampled per cycle
                scalarization=entry.get("scalarization", "augmented_chebyshev"),
                y_stats=y_stats,
            )
            landscape = landscape_sanity_check(landscape)
            landscape_list.append(pad_to_full(landscape, idx_map, M))

            if penalization_params:
                landscape = penalize_landscape_fast(landscape, X_sub,
                                                    X_train_with_already_selected)

            batch = BatchSelectionStrategy(...).batch_acquire(
                X_candidates=X_sub, model=ml_model, acquisition_function=acqui_func,
                batch_size=entry["n_points"],
                X_train=..., y_train=...,
            )
            original_idx = idx_map[batch]
            sampled_new_idx.extend(original_idx)
            candidate_mask[original_idx] = False    # D11 sequential-with-removal

    return sampled_new_idx, np.vstack(landscape_list)
```

### 4.4 Config schema sketch

The real production pipeline driven by `scripts/benchmark_functions.py`
(and its CSV-ground-truth variant `scripts/benchmark_gtlandscape.py`)
uses **four separate YAML files**, not a single merged config:

| File | Role |
|---|---|
| `scripts/general_config/benchmark_config.yaml` | Experiment settings: `SEED`, `experiment_name`, `search_space_variables`, `target_variables` (**authoritative target list**), `n_cycles`, `init_batch_size`, `landscape_penalization`, **global** `batch_selection` block (including `method` and `percentile`), `acquisition_protocol`, `adaptive_refinement`. |
| `scripts/general_config/acquisition_mode_settings.yaml` | Only the `acquisition_parameters` list. Entries carry acquisition-specific params (`y_target`, `epsilon`, `percentage`, ...) but NOT `n_points` or `percentile`. Phase 2 adds `target_variables`, `weight_sampler`, `weights`, `scalarization` here. |
| `scripts/mlmodel_config/<family>_config.yaml` | Model type, `model_parameters`, `grid_search` settings. The per-property vs. global distinction (D6) lives here: either `ml_model: "GPR"` applies to all targets, or `ml_model: { y1: "GPR", y2: "AnchoredEnsembleMLP" }` splits by target. |
| `scripts/general_config/target_function_config.yaml` | Benchmark function / dataset generation. (`benchmark_gtlandscape.py` does not use this one.) |

**Important consequence:** `batch_selection.percentile` is **global**,
not per-entry, so there is no risk of per-entry percentile conflicts
across mixed per-property / joint entries in a single cycle. This is
simpler than an early sketch that embedded `percentile` inside each
entry and should be preserved.

A fully worked multi-property four-file example is part of P2.7 in
`PHASES.md`.

---

## 5. What stays unchanged

These modules are **not** touched by Phases 1, 1.5, or 2:

- `activereg/mlmodel/_gpr.py`
- `activereg/mlmodel/_knn.py`
- `activereg/mlmodel/_mlp.py` (except consumption of `out_feats > 1` in
  Phase 1.5, which is already supported by the existing code)
- `activereg/mlmodel/_bnn.py`
- `activereg/mlmodel/_kernels.py`
- `activereg/mlmodel/_base.py`
- `activereg/sampling.py`
- `activereg/cycle.py` (the `lab_al_cycle` / `active_learning_cycle_insilico`
  functions may get a light pass-through update, but their internals
  are stable)
- `activereg/hyperparams.py`
- `activereg/data.py`, `metrics.py`, `beauty.py`, `format.py`, `utils.py`
- The single-property acquisition formulas in `acquisition.py`:
  `upper_confidence_bound`, `expected_improvement`,
  `target_expected_improvement`, `percentage_target_expected_improvement`,
  `exploration_mutual_info`, `maximum_predicted_value`,
  `uncertainty_landscape`
- All `BatchSelectionStrategy` branches including `batch_highest_landscape`

Concentrating change in `experiment.py` + a new `_multi_property.py`
submodule + a modest `acquisition.py` extension keeps the blast radius
small and the review surface manageable.

---

## 6. References

- Knowles, J. (2006). *ParEGO: A hybrid algorithm with on-line landscape
  approximation for expensive multiobjective optimization problems.*
  IEEE TEC 10(1), 50–66. — The ParEGO algorithm, augmented Chebyshev
  scalarization with random weights.
- Emmerich, M. (2006). *Single- and Multi-objective Evolutionary
  Optimization Assisted by Gaussian Random Field Metamodels.* — Original
  2D closed-form EHVI (relevant for Phase 3 if ever pursued).
- Yang, K. et al. (2019). *Multi-Objective Bayesian Global Optimization
  using expected hypervolume improvement gradient.* — EHVI for higher
  dimensions (relevant for Phase 3+ only).
