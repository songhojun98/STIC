# Plan.md

## Project
STIC: Selective Trust in Context for time-series forecasting under unreliable context.

## Fixed Thesis
STIC is useful when context is essential but not always trustworthy.

## Narrative Freeze
- ETTh1 = frozen mechanism reference
- Exchange = main numeric evidence
- CiK = text transfer evidence
- Active gate space is frozen to `g0 / g1b / g1b-topclip / lite-0.75`
- No new gate family, backbone, or heavy text module in the current round

## Gate Aliases
- `g0` = default gate
- `g1b` = separation gate
- `g1b-topclip` = mitigation gate
- `lite-0.75` = compromise gate

## Current Round Objectives
1. Freeze the numeric narrative in documentation and naming.
2. Wire the CiK thin runner to the official benchmark evaluation path.
3. Test whether selective trust improves robustness under harmful or misleading text.
4. Produce outputs that support a paper-level main-vs-appendix decision for CiK.

## Workstreams

### Workstream 1. Numeric Narrative Freeze
Goal:
- Keep numeric evidence focused on Exchange.
- Keep ETTh1 only as a fixed mechanism reference.

Deliverables:
- documentation with fixed role split
- explicit gate aliases
- explicit statement that active gate space is frozen

Success criteria:
- no further gate expansion in docs
- ETTh1 / Exchange / CiK roles are unambiguous

### Workstream 2. CiK Official Evaluation Wiring
Goal:
- Preserve the thin runner while using the official CiK evaluation path.

Implementation scope:
- `scripts/cik/run_stic_cik.py`
- `utils/cik_stic.py`
- `utils/cik_adapter.py`

Requirements:
- support `--eval-mode debug|official|both`
- evaluate `pred_h`, `pred_c`, and `pred_final` separately
- keep debug metrics as secondary outputs
- prefer benchmark-provided metric code over reimplementation

Success criteria:
- at least one CiK task runs with official evaluation
- `pred_h / pred_c / pred_final` official scores are directly comparable

### Workstream 3. CiK Harmful-Context Robustness
Goal:
- Test whether gating reduces degradation under misleading text.

Primary tasks:
- `ElectricityIncreaseInPredictionTask`
- `ElectricityIncreaseInPredictionWithDistractorText`
- `ElectricityIncreaseInPredictionWithDistractorWithDates`
- `ElectricityIncreaseInPredictionWithSplitContext`
- `OraclePredUnivariateConstraintsTask` as secondary task-family evidence

Evaluation principle:
- clean relevant text can favor `pred_c`
- harmful-context robustness is judged by whether `pred_final` degrades less than `pred_c`

Success criteria:
- at least one harmful-context task shows robustness gain for `pred_final` over `pred_c`
- gate behavior is interpretable on ROI or constrained regions

## Current Status
- Numeric STIC core is already in place and should not be refactored in this round.
- CiK thin prototype already runs history-only, text-conditioned, and gated forecasts.
- Official CiK evaluation wiring is now the highest-priority implementation task.
- Harmful-context evidence, not clean-text average performance, will determine CiK section strength.

## Decision Rule For This Round
- If harmful-context `pred_final` reduces degradation relative to `pred_c`, keep CiK as a main text-evidence candidate.
- Otherwise, keep CiK as appendix-level transfer evidence and center the paper on numeric evidence.

## Files In Scope
- `AGENTS.md`
- `Plan.md`
- `Documentation.md`
- `scripts/cik/run_stic_cik.py`
- `utils/cik_stic.py`
- `utils/cik_adapter.py`
- optional short memos under `reports/`

## Files Out Of Scope
- `models/STIC.py`
- `run.py`
- `exp_long_term_forecasting.py`
- Exchange / ETTh1 active runtime paths
