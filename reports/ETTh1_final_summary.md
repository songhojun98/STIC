# ETTh1 Final Summary

## Scope
- Dataset/task: `ETTh1 + long_term_forecast`
- Fixed branch: `linear mixer + alpha=0.5`
- Fixed training path: soft utility target on, corruption path on, pair-rank off
- Frozen gate set:
  - `g0`
  - `g1b`
  - `g1b-topclip`
  - `lite-0.75`

## Gate Roles
- `g0`: default gate and strongest average-metric STIC reference
- `g1b`: strongest separation reference
- `g1b-topclip`: best failure-mitigation reference
- `lite-0.75`: compromise candidate between separation retention and mitigation

## Overall Summary

| Model | 3-seed MSE | 3-seed MAE | clean corr | gap | win-rate | Role |
| --- | --- | --- | --- | --- | --- | --- |
| DLinear baseline | `0.07667±0.00056` | `0.20653±0.00086` | `-` | `-` | `-` | baseline |
| g0 | `0.06729±0.00482` | `0.19415±0.00468` | `0.0864` | `0.0002` | `0.4978` | default STIC |
| g1b | `0.07205±0.00890` | `0.20084±0.01164` | `0.0830` | `0.0010` | `0.5768` | strongest separation |
| g1b-topclip | `0.07107±0.00979` | `0.19941±0.01282` | `0.0696` | `0.0004` | `0.5656` | best mitigation |
| lite-0.75 | `0.07163±0.00950` | `0.20023±0.01246` | `0.0768` | `0.0007` | `0.5729` | compromise |

## Utility Slice Summary
- Slice definition: clean sample utility terciles defined from `g1b` per seed.

| Model | Bottom MSE | Middle MSE | Top MSE | Bottom gap | Middle gap | Top gap | Bottom win | Middle win | Top win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g0 | `0.1283±0.0269` | `0.0346±0.0006` | `0.0389±0.0121` | `0.0012±0.0013` | `0.0006±0.0011` | `0.0006±0.0007` | `0.5703±0.0871` | `0.5197±0.1237` | `0.5249±0.0759` |
| g1b | `0.1381±0.0336` | `0.0389±0.0048` | `0.0390±0.0113` | `0.0023±0.0006` | `0.0019±0.0019` | `0.0023±0.0012` | `0.6127±0.0835` | `0.5837±0.1488` | `0.6210±0.0981` |
| g1b-topclip | `0.1375±0.0357` | `0.0379±0.0053` | `0.0377±0.0113` | `0.0015±0.0003` | `0.0013±0.0011` | `0.0016±0.0007` | `0.6201±0.0746` | `0.5951±0.1435` | `0.6257±0.0948` |
| lite-0.75 | `0.1380±0.0350` | `0.0384±0.0052` | `0.0384±0.0114` | `0.0021±0.0005` | `0.0017±0.0017` | `0.0021±0.0011` | `0.6162±0.0783` | `0.5872±0.1464` | `0.6234±0.0956` |

## Horizon Slice Summary

| Model | h1-24 MSE / gap | h25-48 MSE / gap | h49-72 MSE / gap | h73-96 MSE / gap |
| --- | --- | --- | --- | --- |
| g0 | `0.0398±0.0016 / 0.0009±0.0011` | `0.0617±0.0037 / 0.0008±0.0010` | `0.0770±0.0050 / 0.0008±0.0010` | `0.0906±0.0092 / 0.0008±0.0010` |
| g1b | `0.0422±0.0043 / 0.0022±0.0012` | `0.0675±0.0067 / 0.0022±0.0012` | `0.0826±0.0118 / 0.0022±0.0012` | `0.0959±0.0134 / 0.0022±0.0012` |
| g1b-topclip | `0.0420±0.0051 / 0.0015±0.0007` | `0.0663±0.0078 / 0.0015±0.0007` | `0.0819±0.0126 / 0.0015±0.0007` | `0.0941±0.0141 / 0.0015±0.0007` |
| lite-0.75 | `0.0421±0.0048 / 0.0020±0.0011` | `0.0670±0.0074 / 0.0020±0.0011` | `0.0823±0.0124 / 0.0020±0.0011` | `0.0951±0.0139 / 0.0020±0.0011` |

## Corruption-Type Summary

| Model | Shuffle gap / win | Swap gap / win | Dropout gap / win |
| --- | --- | --- | --- |
| g0 | `0.0019±0.0020 / 0.5451±0.0259` | `-0.0000±0.0000 / 0.4786±0.0024` | `0.0006±0.0012 / 0.5904±0.2752` |
| g1b | `0.0045±0.0013 / 0.6161±0.0072` | `-0.0004±0.0005 / 0.4469±0.0526` | `0.0025±0.0023 / 0.7528±0.3030` |
| g1b-topclip | `0.0030±0.0008 / 0.6235±0.0167` | `-0.0003±0.0003 / 0.4480±0.0497` | `0.0016±0.0014 / 0.7675±0.2816` |
| lite-0.75 | `0.0040±0.0013 / 0.6202±0.0095` | `-0.0004±0.0004 / 0.4456±0.0529` | `0.0022±0.0020 / 0.7592±0.2934` |

## Trade-off Conclusion
- `g0` remains the best default because it keeps the best average STIC metric and avoids overcommitting to separation.
- `g1b` is the cleanest mechanism reference because it consistently maximizes gap and win-rate.
- `g1b-topclip` is the best mitigation reference because it reduces the top-utility error while keeping useful separation.
- `lite-0.75` is the best compromise candidate, but it does not beat `g1b-topclip` on mitigation or `g1b` on separation.
