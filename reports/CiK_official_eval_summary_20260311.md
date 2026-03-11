# CiK Official Eval Summary (2026-03-11)

## Framing
CiK is used here as text transfer evidence for STIC, not as a benchmark-solve claim.

Fixed thesis:
STIC is useful when context is essential but not always trustworthy.

## Runner
- script: `scripts/cik/run_stic_cik.py`
- eval mode: `both`
- official metric: `threshold_weighted_crps.metric`
- sample paths: `32`
- dataset split: `test`

## Server Command
```bash
/home/dkhan/anaconda3/bin/python3 scripts/cik/run_stic_cik.py \
  --task-names \
    ElectricityIncreaseInPredictionTask \
    ElectricityIncreaseInPredictionWithDistractorText \
    ElectricityIncreaseInPredictionWithDistractorWithDates \
    ElectricityIncreaseInPredictionWithSplitContext \
  --limit 5 \
  --limit-per-task \
  --batch-size 4 \
  --device cpu \
  --eval-mode both \
  --num-sample-paths 32 \
  --summary-csv reports/cik_official_eval_20260311.csv \
  --summary-json reports/cik_official_eval_20260311.json \
  --output-csv reports/cik_official_eval_rows_20260311.csv
```

## Official Summary
| Task | pred_h | pred_c | pred_final |
|---|---:|---:|---:|
| ElectricityIncreaseInPredictionTask | 0.0866 | 0.0075 | 0.0219 |
| ElectricityIncreaseInPredictionWithDistractorText | 0.0865 | 0.0075 | 0.0257 |
| ElectricityIncreaseInPredictionWithDistractorWithDates | 0.0865 | 0.0289 | 0.0321 |
| ElectricityIncreaseInPredictionWithSplitContext | 0.0865 | 0.1008 | 0.0649 |

Lower is better.

## Main Takeaways
- Clean relevant text: `pred_c` is strongest and `pred_final` is a conservative interpolation.
- Weak distractor text: gate does not add much beyond `pred_c`.
- Strong harmful text (`WithSplitContext`): `pred_c` becomes worse than history-only, while `pred_final` improves over both branches.

## Decision
Current recommendation is to keep CiK as a main text-evidence candidate.
The decisive evidence is `ElectricityIncreaseInPredictionWithSplitContext`, where the gated final forecast reduces harmful-text degradation under the official metric.
