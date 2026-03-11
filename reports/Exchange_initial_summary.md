# Exchange Initial Summary

## Scope
- Dataset/task: `exchange_rate.csv + long_term_forecast`
- Fixed branch: `linear mixer + alpha=0.5`
- Fixed training path: soft utility target on, corruption path on, pair-rank off
- Screening gate set:
  - `g0`
  - `g1b`
  - `g1b-topclip`
  - `lite-0.75`

## Dataset Wiring
- Root path: `./dataset/exchange_rate/`
- Data path: `exchange_rate.csv`
- Loader mode: `custom`
- Features/target: `MS + OT`
- Shapes: `enc_in=8`, `dec_in=8`, `c_out=8`
- Horizon setup: `seq_len=96`, `label_len=48`, `pred_len=96`

## Baseline
- DLinear 1-seed (`2021`): `mse=0.12607`, `mae=0.28877`
- DLinear 3-seed:
  - `mse=0.12461±0.00412`
  - `mae=0.28792±0.00535`

## STIC 1-Seed Screening

| Gate | MSE | MAE | clean corr | corrupt corr | gap | win-rate |
| --- | --- | --- | --- | --- | --- | --- |
| g0 | `0.13046` | `0.28922` | `-0.0432` | `0.0487` | `0.0028` | `0.4987` |
| g1b | `0.13048` | `0.29087` | `0.1348` | `0.0406` | `0.0031` | `0.6500` |
| g1b-topclip | `0.13890` | `0.30041` | `0.1329` | `0.0537` | `0.0011` | `0.5784` |
| lite-0.75 | `0.13421` | `0.29523` | `0.1357` | `0.0440` | `0.0020` | `0.6261` |

## Screening Decision
- Most informative 2-gate comparison for 3-seed:
  - `g0` as default reference
  - `g1b` as strongest separation candidate
- Reason:
  - `g1b` shows the sharpest utility-aligned gate signal already at 1 seed
  - `g1b-topclip` and `lite-0.75` remain informative but are weaker than `g1b` on the core selective-trust signal

## STIC 3-Seed Follow-up

| Gate | MSE | MAE | clean corr | gap | win-rate |
| --- | --- | --- | --- | --- | --- |
| g0 | `0.12666±0.00821` | `0.28746±0.01620` | `-0.0342±0.0111` | `0.0017±0.0010` | `0.5728±0.0921` |
| g1b | `0.13231±0.00929` | `0.29636±0.01651` | `0.0662±0.0511` | `0.0022±0.0011` | `0.6358±0.0541` |

## Slice-Level Readout
- Utility slices:
  - `g1b` beats `g0` on paired win-rate in all three utility terciles:
    - bottom `0.7067` vs `0.6880`
    - middle `0.6892` vs `0.5483`
    - top `0.6776` vs `0.5692`
  - `g1b` also keeps positive utility-slice gap in all three terciles, while `g0` turns negative in middle/top.
- Horizon slices:
  - `g1b` keeps positive gate gap in all 4 horizon buckets (`~0.0046-0.0047`).
  - `g0` stays near zero across buckets.
- Corruption types:
  - `shuffle`: `g1b` utility drop `0.1449` vs `g0 0.1076`
  - `swap`: both remain weak, but `g1b` is slightly less negative on gap
  - `dropout`: `g1b` win-rate `0.9817` vs `g0 0.7640`

## Current Interpretation
- Exchange produces a stronger selective-trust signal than ETTh1.
- The gain is clearest in:
  - higher paired win-rate
  - positive slice-level gap persistence
  - stronger horizon-consistent utility drop under corruption
- The trade-off is that `g1b` still loses average MSE/MAE to both DLinear and `g0`.

## Next Decision
- Keep `g0` as the deployment/default metric reference.
- Keep `g1b` as the main numeric selective-trust mechanism reference.
- Use ILI next only if we want to test whether this stronger separation pattern persists outside Exchange.
