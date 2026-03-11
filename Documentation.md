# Documentation.md

## Project Snapshot
STIC studies selective trust in context for forecasting settings where context can help but can also mislead.

Fixed framing:
STIC is useful when context is essential but not always trustworthy.

## Evidence Roles
- ETTh1: frozen mechanism reference
- Exchange: main numeric evidence
- CiK: text transfer evidence

## Active Gate Set
- `g0`: default gate
- `g1b`: separation gate
- `g1b-topclip`: mitigation gate
- `lite-0.75`: compromise gate

No additional active gate variants should be introduced in the current narrative.

## Numeric Narrative
- ETTh1 is no longer an active search space.
- Exchange is the main numeric section because it shows stronger selective-trust signal than ETTh1.
- `g0` remains the accuracy-default reference.
- `g1b` remains the mechanism-strong separation reference.
- `g1b-topclip` and `lite-0.75` remain bounded mitigation/compromise references.
- S3 / Hybrid S3 are archived as negative-but-informative simplification results.

## CiK Prototype Scope
The current CiK path is intentionally thin:
- HF direct-load
- thin adapter
- history-only branch
- text-conditioned context-aware branch
- simple g0-style trust gate
- final prediction: `pred_h + gate * (pred_c - pred_h)`

This is not a benchmark-solve claim.
It is a transfer test of the selective-trust principle under text context.

## CiK Evaluation Status
Current runner:
- `scripts/cik/run_stic_cik.py`

Supported evaluation modes:
- `--eval-mode debug`
- `--eval-mode official`
- `--eval-mode both`

Official evaluation path:
- uses the benchmark's `threshold_weighted_crps` implementation through a compatibility loader
- evaluates `pred_h`, `pred_c`, and `pred_final` separately
- keeps debug metrics secondary

Additional runner note:
- `--limit` is a global row cap by default
- `--limit-per-task` applies the cap independently to each selected task

## Server-Validated CiK Results
Execution environment:
- server: `dkhan@100.70.96.36`
- python: `/home/dkhan/anaconda3/bin/python3`
- CiK dataset split used: `test`

Artifact-freeze outputs:
- `reports/cik_suite_freeze_rows_20260311.csv`
- `reports/cik_suite_freeze_20260311.csv`
- `reports/cik_suite_freeze_20260311.json`
- `reports/CiK_official_eval_summary_20260311.md`

### Clean Relevant Text
Task:
- `ElectricityIncreaseInPredictionTask`

Batch sanity:
- `pred_h / pred_c / gate / pred_final = (2, 24, 1)` on server debug runs

Official summary:
- `official_h_metric = 0.0866`
- `official_c_metric = 0.0075`
- `official_final_metric = 0.0219`

Debug summary:
- `mse_h = 4828618.6591`
- `mse_c = 40520.9670`
- `mse_final = 318499.4241`
- `mae_h = 511.0727`
- `mae_c = 106.6161`
- `mae_final = 182.7227`
- `gate_mean = 0.1586`
- `gate_roi_mean = 0.8178`
- `gate_non_roi_mean = 0.1051`

Interpretation:
- clean relevant text strongly helps the context-aware branch
- final prediction is more conservative than `pred_c`
- this is acceptable and not a failure mode

### Constraint Family Wiring
Task:
- `OraclePredUnivariateConstraintsTask`

Official summary:
- `official_h_metric = 0.0866`
- `official_c_metric = 0.0398`
- `official_final_metric = 0.0569`

Debug summary:
- `mse_h = 2.0413`
- `mse_c = 1.6641`
- `mse_final = 1.7774`
- `mae_h = 0.6037`
- `mae_c = 0.5437`
- `mae_final = 0.5693`
- `gate_mean = 0.1372`

Interpretation:
- constraint-aware context wiring works
- this is secondary evidence, not the main CiK robustness result

### Oracle Harmful-Variant Search
Benchmark-provided Oracle/constraint-family task names found in the shipped task registry:
- `OraclePredUnivariateConstraintsTask`
- `BoundedPredConstraintsBasedOnPredQuantilesTask`

Current conclusion:
- benchmark-provided harmful Oracle variant not found
- no Oracle-family distractor, misleading, split-context, or conflicting-constraint task was found in the benchmark task names
- `BoundedPredConstraintsBasedOnPredQuantilesTask` is the nearest extra constraint-family task, but it is not harmful-text evidence

Secondary bounded-task check:
- official summary on `BoundedPredConstraintsBasedOnPredQuantilesTask`:
  - `official_h_metric = 5.1895`
  - `official_c_metric = 0.1980`
  - `official_final_metric = 1.4650`
- interpretation:
  - the current constraint parser transfers to another benchmark-provided constraint task
  - this supports constraint-family coverage, but not harmful-text robustness

### Harmful-Context Electricity Variants
Selected tasks:
- `ElectricityIncreaseInPredictionWithDistractorText`
- `ElectricityIncreaseInPredictionWithDistractorWithDates`
- `ElectricityIncreaseInPredictionWithSplitContext`

Selection rationale:
- all are actual benchmark task names
- all preserve the same electricity event semantics while changing context reliability
- `WithSplitContext` is the strongest harmful variant because it contains an explicit contradictory magnitude correction

#### DistractorText
Official summary:
- `official_h_metric = 0.0865`
- `official_c_metric = 0.0075`
- `official_final_metric = 0.0257`

Interpretation:
- distractor is weak for the current parser
- `pred_c` still dominates
- final remains conservative but does not add robustness value here

#### DistractorWithDates
Official summary:
- `official_h_metric = 0.0866`
- `official_c_metric = 0.0289`
- `official_final_metric = 0.0321`

Debug summary:
- `gate_mean = 0.1493`
- `gate_roi_mean = 0.7782`
- `gate_non_roi_mean = 0.0987`

Interpretation:
- the variant is partially harmful but still not enough to make `pred_final` beat `pred_c` on average

#### SplitContext
Official summary:
- `official_h_metric = 0.0866`
- `official_c_metric = 0.1008`
- `official_final_metric = 0.0649`

Debug summary:
- `mse_h = 4828618.6591`
- `mse_c = 4164467.9805`
- `mse_final = 1410704.9791`
- `mae_h = 511.0727`
- `mae_c = 524.1412`
- `mae_final = 344.3502`
- `gate_mean = 0.1481`
- `gate_roi_mean = 0.8184`
- `gate_non_roi_mean = 0.0944`

Interpretation:
- this is the current key CiK robustness result
- misleading text makes `pred_c` worse than history-only
- gated final prediction improves over both `pred_h` and `pred_c`
- this is direct evidence for selective trust under harmful text

## Current Recommendation
CiK should remain a main text-evidence candidate for now, because `WithSplitContext` shows the right robustness pattern under the official metric.

That recommendation is conditional:
- keep CiK in the main narrative only if the harmful-context pattern remains stable when rerun and summarized cleanly
- otherwise demote it to appendix-level transfer evidence

## Known Risks
- CiK still relies on a rule-based text parser rather than a learned text model
- the HF dataset exposes only the `test` split, so this is evaluation-only evidence
- not every distractor variant is equally harmful for the current parser
- `pred_final` is intentionally conservative, so clean-text averages alone can understate its value

## Transparency Note
- the official CiK metric path is executed through a Python 3.8 compatibility loader
- this is required because the upstream metric package uses newer type-annotation syntax on import
- the metric logic itself still comes from the official benchmark code rather than a custom rewrite

## Next Step
1. Keep the CiK claim centered on harmful-text degradation control, not benchmark solving.
2. Decide whether a second harmful text family exists beyond electricity; if not, keep the limitation explicit.
3. Package numeric and CiK sections with the fixed role split above.
