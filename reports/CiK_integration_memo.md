# CiK Integration Memo

## Goal
- Port the STIC principle from numeric exogenous forecasting to a minimal CiK text-context prototype without changing the core semantics:
  - history-only branch
  - text-conditioned context-aware branch
  - trust gate

## Current CiK Interface Readout
- Repo: `/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark`
- Core task interface lives in [base.py](/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark/base.py).
- Every task exposes:
  - `past_time`: history dataframe
  - `future_time`: forecast horizon dataframe
  - `constraints`, `background`, `scenario`: text context fields
- `UnivariateCRPSTask` evaluates the last column of `future_time`, so STIC can keep the same “target is last channel” convention.
- Evaluation entrypoint is [evaluation.py](/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark/evaluation.py), where a `method_callable(task_instance, n_samples)` is the only hard requirement.

## Best Prototype Task Candidates

### Candidate 1. ElectricityIncreaseInPredictionTask
- File: [electricity_tasks.py](/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark/tasks/electricity_tasks.py)
- Why it fits:
  - real time-series windows
  - short, explicit scenario text
  - clear future effect that should sometimes matter and sometimes be retrieved correctly
  - distractor variants already exist, which naturally match the STIC “harmful context” story
- Recommended initial variants:
  - `ElectricityIncreaseInPredictionTask`
  - `ElectricityIncreaseInPredictionWithDistractorText`

### Candidate 2. OraclePredUnivariateConstraintsTask family
- File: [predictable_constraints_real_data.py](/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark/tasks/predictable_constraints_real_data.py)
- Why it fits:
  - real-data windows with textual future constraints
  - context usefulness is explicit and measurable
  - good second-stage prototype once the electricity task path works
- Caveat:
  - this family is closer to constraint following than numeric exogenous covariates, so it is slightly less direct than electricity for the first prototype

## Minimal STIC Mapping for CiK

### History-only branch
- Input:
  - `past_time.iloc[:, -1]`
- Output:
  - univariate forecast over the CiK horizon
- Closest current implementation:
  - current STIC `pred_h` path

### Text-conditioned context-aware branch
- Input:
  - target history
  - text context from `scenario` plus optional `background` / `constraints`
- Minimal prototype rule:
  - start with a single fused text string
  - produce one conditioning vector
  - inject it into the context-aware branch only
- Initial goal:
  - prove the text branch changes `pred_c` in a controlled way before optimizing accuracy

### Trust gate
- Keep the same selective-trust formula:
  - `pred = pred_h + g * (pred_c - pred_h)`
- Initial CiK objective should mirror current numeric STIC:
  - soft utility target
  - no pair-rank in the first text prototype

## Minimal Code-Touch List

### Read-only dependencies from CiK
- [base.py](/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark/base.py)
- [evaluation.py](/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark/evaluation.py)
- [electricity_tasks.py](/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark/tasks/electricity_tasks.py)
- [predictable_constraints_real_data.py](/Volumes/dkhan/hj_song/context-is-key-forecasting/cik_benchmark/tasks/predictable_constraints_real_data.py)

### Likely new files in TSLib workspace
- `scripts/cik/run_stic_cik.py`
  - loads CiK tasks and runs a `method_callable` wrapper
- `utils/cik_adapter.py`
  - converts `task_instance` to tensors and merged text context
- `models/STIC_CiK.py` or `models/STIC_text_adapter.py`
  - minimal text-conditioned branch wrapper built on top of current STIC semantics

### Files to avoid touching first
- current ETTh1 reference code paths
- numeric branch defaults in `models/STIC.py`
- long-term forecasting training loop unless reuse becomes clearly cheaper than a thin CiK adapter

## Minimal Prototype Plan
1. Implement a CiK adapter that extracts `past_time`, `future_time`, and merged text context.
2. Start with `ElectricityIncreaseInPredictionTask` only.
3. Build a history-only baseline using only `past_time.iloc[:, -1]`.
4. Add a text-conditioned branch that projects the merged text into a small conditioning vector.
5. Reuse the STIC gate rule to combine history-only and text-conditioned forecasts.
6. Validate first on:
   - forward shape
   - one task / one seed smoke test
   - distractor vs non-distractor qualitative gap

## Success Criterion for the Next Round
- One CiK task runs end-to-end with:
  - history-only forecast
  - text-conditioned forecast
  - trust gate output
- and the text-conditioned branch can be shown to change `pred_c` on relevant scenarios without introducing leakage.
