# CiK Official Eval Artifact Freeze (2026-03-11)

## Fixed Thesis
STIC is useful when context is essential but not always trustworthy.

CiK is used here as text transfer evidence for selective trust under unreliable context.
This is not a benchmark-solve or SOTA claim.

## Reproducible 5-Task Suite
Task suite:
- `ElectricityIncreaseInPredictionTask`
- `ElectricityIncreaseInPredictionWithDistractorText`
- `ElectricityIncreaseInPredictionWithDistractorWithDates`
- `ElectricityIncreaseInPredictionWithSplitContext`
- `OraclePredUnivariateConstraintsTask`

Execution details:
- split: `test`
- seed: `7`
- `num_sample_paths = 32`
- `per_task_limit = 5`
- eval mode: `both`
- runner: `scripts/cik/run_stic_cik.py`

Server command:
```bash
/home/dkhan/anaconda3/bin/python3 scripts/cik/run_stic_cik.py \
  --task-names \
    ElectricityIncreaseInPredictionTask \
    ElectricityIncreaseInPredictionWithDistractorText \
    ElectricityIncreaseInPredictionWithDistractorWithDates \
    ElectricityIncreaseInPredictionWithSplitContext \
    OraclePredUnivariateConstraintsTask \
  --limit 5 \
  --limit-per-task \
  --batch-size 4 \
  --device cpu \
  --eval-mode both \
  --num-sample-paths 32 \
  --seed 7 \
  --summary-csv reports/cik_suite_freeze_20260311.csv \
  --summary-json reports/cik_suite_freeze_20260311.json \
  --output-csv reports/cik_suite_freeze_rows_20260311.csv
```

Saved artifacts:
- row CSV: `reports/cik_suite_freeze_rows_20260311.csv`
- summary CSV: `reports/cik_suite_freeze_20260311.csv`
- summary JSON: `reports/cik_suite_freeze_20260311.json`

## Official Metric Table
Lower is better.

| Task | pred_h | pred_c | pred_final |
|---|---:|---:|---:|
| ElectricityIncreaseInPredictionTask | 0.0866 | 0.0075 | 0.0219 |
| ElectricityIncreaseInPredictionWithDistractorText | 0.0865 | 0.0075 | 0.0257 |
| ElectricityIncreaseInPredictionWithDistractorWithDates | 0.0865 | 0.0289 | 0.0321 |
| ElectricityIncreaseInPredictionWithSplitContext | 0.0865 | 0.1008 | 0.0649 |
| OraclePredUnivariateConstraintsTask | 0.0866 | 0.0398 | 0.0569 |

## Main Takeaways
- Clean relevant electricity context strongly helps `pred_c`, while `pred_final` remains a conservative interpolation.
- Weak distractor variants do not yet justify the gate on average, but they also do not break the selective-trust story.
- Strong harmful text in `ElectricityIncreaseInPredictionWithSplitContext` flips the ranking: `pred_c` becomes worse than `pred_h`, while `pred_final` beats both. This is the current key degradation-control result.

## Oracle Harmful-Variant Search
Constraint-family task names found in the benchmark task registry and HF dataset:
- `OraclePredUnivariateConstraintsTask`
- `BoundedPredConstraintsBasedOnPredQuantilesTask`

Search result:
- benchmark-provided harmful Oracle variant not found
- no Oracle-family distractor, misleading, split-context, or conflicting-constraint task was found in the shipped task names

Nearest extra constraint-family task:
- `BoundedPredConstraintsBasedOnPredQuantilesTask`
- official metric: `pred_h = 5.1895`, `pred_c = 0.1980`, `pred_final = 1.4650`
- interpretation: useful as a secondary constraint-family check, but not harmful-text evidence

## Recommendation
Keep CiK as a main text-evidence candidate.

Reason:
- the fixed 5-task suite is now reproducible
- `WithSplitContext` harmful-text robustness is stable under rerun
- Oracle harmful evidence is absent, but its absence is now transparently documented

## Remaining Risks
- CiK still relies on a rule-based text parser rather than a learned text model
- harmful evidence is currently strongest in the electricity family, not the Oracle family
- clean-text averages alone can understate the value of a conservative gated final forecast

## Transparency Note
The official CiK metric path is preserved through a Python 3.8 compatibility loader.
This loader exists because the upstream metric package uses newer type-annotation syntax that fails on the server Python version.
The metric logic itself still comes from the official benchmark implementation rather than a custom reimplementation.
