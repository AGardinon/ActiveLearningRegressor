# Lab Scripts — Implementation State

Branch: `refactor/simulation-scripts`
Design reference: `docs/lab_scripts/DESIGN.md`

---

## Status legend

- `[ ]` not started
- `[~]` in progress
- `[x]` done

---

## Phase 1 — Core library additions (`activereg/experiment.py`)

- [x] Add `prepare_training_data()` (migrated + adapted from `cycle.py`)
- [x] Add `run_single_al_cycle()` (new core abstraction)
- [x] Add `LAB_AL_REPO` to `activereg/format.py`

## Phase 2 — Deletions

- [x] Delete `activereg/cycle.py`
- [x] Delete `run_labAL_cycle.py` (root)

## Phase 3 — New script: `scripts/run_lab_cycle.py`

- [x] Experiment initialization (cycle 0): create folder structure, save POOL/CANDIDATES/EVIDENCE CSVs, fit + save scaler
- [x] Cycle detection (count `cycle_N/` folders)
- [x] State update (read + validate `validated.csv`, update EVIDENCE + CANDIDATES)
- [x] `validated.csv` validation checks (columns, no NaN, shape, feature match)
- [x] Call `run_single_al_cycle()` and save outputs
- [x] In-silico mode (auto-validate when `ground_truth_file` is set)
- [x] Write example config `scripts/lab_cycle_config_template.yaml`

## Phase 4 — Fix `scripts/insilico_lab_al_simulation.py`

- [x] Fix `setup_ml_model(config)` → `setup_multi_property_ml_model(config, target_vars)`
- [x] Fix `sampling_block` call (new signature: `Y_train` 2-D, `batch_selection_method`, `batch_selection_params`)
- [x] Update `insilico_lab_al_simulation_config.yaml` (add `batch_selection_method`/`batch_selection_params`)
- [x] Remove dead `CYCLE_SAMPLING` parameter from config + script

## Phase 5 — README

- [x] Remove `cycle.py` from Package Structure
- [x] Update Quick Start example (replace `lab_al_cycle` with `run_single_al_cycle`)
- [x] Add Lab Use section (single-cycle workflow + insilico simulation)
- [x] Update Configuration section

---

## Notes

_Add session notes here as work progresses._
