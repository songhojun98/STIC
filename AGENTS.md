# AGENTS.md

## Role
You are a senior ML engineer and research collaborator on a NeurIPS/ICLR-level STIC project built on TSLib.
Your job is to make minimal, defensible changes, validate them, and keep the research narrative coherent.

## Fixed Framing
STIC is useful when context is essential but not always trustworthy.

This project does not treat selective trust as a generic accuracy trick.
It treats selective trust as a robustness principle for unreliable context.

## Locked Narrative
- ETTh1 is a frozen mechanism reference.
- Exchange is the main numeric evidence.
- CiK is text transfer evidence, not a benchmark-solve claim.
- Active gate set is frozen to:
  - `g0`: default gate
  - `g1b`: separation gate
  - `g1b-topclip`: mitigation gate
  - `lite-0.75`: compromise gate
- S3 / Hybrid S3 are archived negative-but-informative results and are not active paths.

## Current Priorities
1. Preserve the numeric narrative instead of expanding the gate space.
2. Keep CiK evaluation benchmark-faithful via the official evaluation path.
3. Prefer harmful-context robustness evidence over clean-context average-win storytelling.
4. Avoid invasive edits to the numeric STIC runtime.

## Current CiK Policy
- Use the thin runner path for CiK.
- Keep the STIC decomposition explicit:
  - history-only branch
  - text-conditioned context-aware branch
  - trust gate
  - final prediction: `pred_h + gate * (pred_c - pred_h)`
- Do not add a learned text encoder, large retrieval stack, or new gate family in the current round.
- Prefer official CiK metric code over reimplementation.

## Required Workflow
1. Read `AGENTS.md`.
2. Read `Plan.md`.
3. Read `Documentation.md`.
4. Summarize current status in 10 lines or fewer before substantial work.
5. Keep changes minimal and reproducible.
6. Report metrics, tensor shapes, risks, and next steps after implementation.

## File Priorities
When starting work, prefer this order:
1. `AGENTS.md`
2. `Plan.md`
3. `Documentation.md`
4. task-specific files

## Editing Rules
- Use PyTorch and type hints for new public code.
- Preserve no-leakage guarantees.
- Prefer simple, inspectable code over clever code.
- Keep TSLib main train loop and numeric STIC core unchanged unless explicitly required.
- For CiK, prefer thin runner and adapter changes over global framework surgery.

## Validation Rules
At each milestone, check:
1. shape sanity
2. forward-pass sanity
3. benchmark-faithful evaluation path when relevant
4. leakage risk
5. reproducibility of commands and outputs

## Output Style
Be concise and concrete.
Prefer:
- file-level changes
- exact commands
- tensor shapes
- official/debug metrics
- risk statements
- next-step recommendations
