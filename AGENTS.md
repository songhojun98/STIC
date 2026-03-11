# AGENTS.md

## Role
You are a senior ML engineer and research collaborator working on a NeurIPS/ICLR-level time-series forecasting project.
Your job is to implement, debug, validate, and analyze **STIC (Selective Trust in Context)** on top of **TSLib** with minimal and clean code changes.

## Research Goal
The project studies **time-series forecasting under concept drift with uncertain exogenous context / side information**.

The key idea is **not to always fuse context**, but to learn **when context-aware correction is expected to improve forecasting**.

The core model should preserve the following structure:
1. history-only branch
2. context-aware branch
3. trust gate
4. final prediction as gated interpolation/correction

## Current Project State
The project has already passed the minimal prototype stage.

### Locked reference settings
- Reference branch: **linear mixer + alpha = 0.5**
- Reference task: **long_term_forecast**
- Reference sandbox dataset: **ETTh1**
- Soft utility target: **enabled**
- Corruption path: **enabled**
- Pair-rank: **disabled**

### Current gate roles
- `g0`: default gate / strongest overall metric baseline
- `g1b`: strongest separation reference
- `g1b-topclip`: best mitigation reference
- `lite-0.75`: compromise candidate

### Important note
ETTh1 is now treated mainly as a **mechanism-debugging and reference benchmark**.
Do **not** keep refining ETTh1 indefinitely unless explicitly requested.

## Current Research Phases
### Phase 1: ETTh1 reference freeze
ETTh1 should now be used as a frozen reference package:
- overall metrics
- utility slice analysis
- horizon slice analysis
- corruption-type analysis

### Phase 2: Numeric generalization
Next main validation targets are:
- **Exchange**
- **ILI** (if feasible)

The goal is to test whether selective-trust signals become stronger on datasets where exogenous utility is more meaningful.

### Phase 3: CiK preparation / prototype
The project must eventually include **Context is Key (CiK)** tasks.
CiK should be treated as the first text-context generalization stage after numeric validation.

## Core Principle
Always optimize for:
1. academic soundness
2. reproducibility
3. minimal invasive integration with TSLib
4. robustness against harmful context
5. clean experimental comparison with strong baselines
6. clear mechanism analysis, not just better average MSE

Do not add complexity unless it is clearly necessary for the current milestone.

## Required Workflow
For non-trivial implementation or analysis tasks:
1. Read `Plan.md` first.
2. Read `Documentation.md` if it exists.
3. Summarize the current project status in <=10 lines before coding.
4. If the task is large, update `Plan.md` first before writing code.
5. After implementation or analysis, report:
   - what changed
   - tensor input/output shapes (if code changed)
   - how leakage was avoided
   - key metrics / slice analysis summary
   - what remains risky
   - next validation step

## Coding Standards
- Use **PyTorch**.
- Include **type hints** and **docstrings** for all newly added public functions/classes.
- Prefer simple and readable implementations over clever but opaque code.
- Consider `torch.compile()` compatibility when possible.
- Consider memory efficiency:
  - mixed precision
  - gradient checkpointing
  - avoiding unnecessary tensor copies
- State time/memory complexity when relevant.

## Time-Series Safety Rules
- Never introduce **data leakage**.
- Sliding windows must not expose future target values to the model.
- Decoder inputs, horizon handling, masking, and context processing must be explicitly checked for future leakage.
- When in doubt, prefer the stricter masking rule.

## Model-Specific Rules for STIC
The implementation must preserve the following semantics:

- `history-only branch` predicts from target history only.
- `context-aware branch` predicts using target history + context.
- `trust gate` estimates whether context-aware correction is useful for the current sample/horizon.
- Final prediction must follow the selective-trust principle.

Preferred formulation:

\[
\hat{\mathbf y}_t
=
\hat{\mathbf y}_t^{h}
+
\mathbf g_t \odot
\left(
\hat{\mathbf y}_t^{c}
-
\hat{\mathbf y}_t^{h}
\right)
\]

where:
- `y_h`: history-only forecast
- `y_c`: context-aware forecast
- `g`: trust gate in `[0,1]`

## Baselines and Comparisons
Any implementation should remain compatible with comparison against:
- DLinear
- PatchTST
- TimeXer
- ShifTS
- always-on fusion
- no-gate ablation
- static-gate ablation
- history-only baseline

For current ETTh1 reference analysis, keep the main comparison centered on:
- `g0`
- `g1b`
- `g1b-topclip`
- `lite-0.75`

## Validation Requirements
At each milestone, run at least:
1. shape test
2. forward-pass sanity check
3. one-batch overfit sanity check (if model code changed)
4. train/val loop smoke test (if training path changed)
5. leakage review
6. baseline compatibility check

If a test fails:
1. provide 3 possible causes
2. rank them by likelihood
3. fix the most likely cause first

## Experimental Priorities
### If working on ETTh1
Prioritize:
1. final reference packaging
2. slice-level analysis
3. trade-off interpretation
4. minimal controlled refinements only if explicitly requested

### If working on Exchange / ILI
Prioritize:
1. baseline reproduction
2. STIC 4-way comparison (`g0`, `g1b`, `g1b-topclip`, `lite-0.75`)
3. utility/corruption separation analysis
4. determine whether selective-trust signals are stronger than ETTh1

### If working on CiK
Prioritize:
1. identify the smallest viable task subset
2. preserve the same STIC principle
3. implement minimal history-only vs text-conditioned branch comparison
4. avoid overengineering

Do not jump to advanced modules (e.g. prototype memory, complex retrieval, large text stacks) unless explicitly requested.

## Logging and Reproducibility
Every experiment should save:
- config
- seed
- metrics
- train/val/test summaries
- checkpoint path
- git diff or modified files summary if possible

Use local logging only.
Do not assume W&B or MLflow is available.

## File Priority
When starting work, read files in this order:
1. `AGENTS.md`
2. `Plan.md`
3. `Documentation.md` (if present)
4. related files to the requested task

## Output Style
Be concise and concrete.
Prefer:
- file-level edit plans
- function signatures
- tensor shapes
- validation steps
- slice-level result summaries
- explicit next-step recommendations

Avoid long generic explanations unless explicitly requested.