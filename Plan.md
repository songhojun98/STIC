# Plan.md

## Project
STIC: Selective Trust in Context for Drift-Aware Time-Series Forecasting

## Current Objective
ETTh1 frozen reference와 Exchange numeric section을 기준으로 active STIC gate space를 `g0 / g1b / g1b-topclip / lite-0.75`로 고정하고, ILI 및 CiK 준비 단계로 넘어간다.

## Current Status
- STIC 최소 통합, branch repair, corruption training, gate-input screening까지 ETTh1 sandbox에서 완료했다.
- ETTh1 기준 고정 결론은 다음과 같다: `g0` default, `g1b` strongest separation reference, `g1b-topclip` best mitigation reference, `g1b-topclip-lite-0.75` compromise candidate.
- branch는 `linear mixer + alpha=0.5`로 고정한다.
- 현재 task는 `long_term_forecast`, 현재 backbone family는 `DLinear` 기반 STIC로 제한한다.
- Exchange에서는 ETTh1보다 stronger selective-trust signal이 이미 관찰됐지만, `g1b`는 separation 이득과 average metric 손실 사이 trade-off가 남아 있다.
- pure S3와 Hybrid S3는 모두 negative simplification result로 정리했고, active runtime path에서는 제거한다.
- 다음 active comparison set은 `g0`, `g1b`, `g1b-topclip`, `g1b-topclip-lite-0.75`만 유지한다.

## Fixed Research Constraints
- Base framework: TSLib + PyTorch
- Minimal invasive changes
- No data leakage
- No W&B / MLflow
- Must support strong baseline comparison
- Must preserve STIC structure:
  - history-only branch
  - context-aware branch
  - trust gate

## Current Milestone
Milestone 13 — CiK thin prototype implementation for text-conditioned selective-trust

## Files To Modify
- [ ] Plan.md
- [ ] Documentation.md
- [ ] utils/cik_stic.py
- [ ] scripts/cik/run_stic_cik.py
- [ ] utils/cik_adapter.py

## Phase Roadmap

### Archived Simplification Track
**Scope**
- S3-Gate와 Hybrid S3는 negative simplification track으로 종료했다.

**Result**
- 둘 다 Exchange 기준 `clean_corr < 0` 또는 큰 metric penalty를 보여 active STIC gate reference로 승격하지 않았다.

**Decision**
- runtime code path에서는 제거하고, 결과는 문서상 historical note로만 유지한다.

### Phase 1. ETTh1 Reference Freeze
**Goal**
- ETTh1를 이후 모든 비교의 frozen reference package로 정리한다.

**Deliverables**
- `ETTh1 final summary table`
- `ETTh1 trade-off summary`

**Validation**
- overall metric, utility slice, horizon slice, corruption type 요약이 하나의 패키지로 정리됨
- 각 gate(`g0`, `g1b`, `g1b-topclip`, `g1b-topclip-lite-0.75`)의 역할이 한 줄로 명시됨

**Success Criteria**
- 이후 Exchange/ILI/CiK 단계에서 ETTh1 코드를 더 수정하지 않고 참조용 기준선으로 사용할 수 있음

### Phase 2. Numeric Generalization
**Goal**
- Exchange 우선, 가능하면 ILI까지 확장해 selective-trust 신호가 ETTh1 밖에서도 재현되는지 확인한다.

**Fixed Settings**
- branch: `linear mixer + alpha=0.5`
- soft utility target: `on`
- corruption path: `on`
- pair-rank: `off`
- candidate gates: `g0`, `g1b`, `g1b-topclip`, `g1b-topclip-lite-0.75`

**Deliverables**
- dataset wiring note
- 1-seed screening 결과
- informative top-2 gate의 3-seed 결과
- utility/horizon/corruption slice summary

**Validation**
- dataset load, tensor shape, baseline reproduction
- STIC 4-way 1-seed screening
- 가장 informative한 gate 2개에 대해서만 3-seed 재현

**Success Criteria**
- Exchange 또는 ILI 중 최소 1개에서 ETTh1보다 더 강한 utility separation 또는 gate signal을 관찰
- `g1b`와 `g1b-topclip`의 trade-off가 numeric generalization 환경에서도 해석 가능하게 재현됨

### Phase 3. CiK Minimal Prototype Planning
**Goal**
- STIC의 selective-trust 원리를 text-conditioned context task로 옮길 수 있는 최소 prototype 설계를 만든다.

**Deliverables**
- `CiK integration memo`
- candidate task 1~2개
- 필요한 코드 변경 파일 목록
- minimal prototype implementation plan

**Validation**
- CiK repo/data interface 확인
- STIC의 `history-only / text-conditioned context-aware / trust gate` 구조로 옮길 최소 경로 정의

**Success Criteria**
- 다음 라운드에서 바로 CiK prototype 구현을 시작할 수 있을 정도로 task와 파일 범위가 구체화됨

## Step-by-Step Plan

### Step 1. Baseline Reproduction
**Goal**
- `ETTh1 + DLinear + long_term_forecast + features=MS + target=OT` 조합이 정상 작동하는지 확인

**Validation**
- train/val/test 1회 실행
- metric 저장 확인
- input/output tensor shape 기록
- checkpoint와 results 파일 생성 확인

**Success Criteria**
- baseline이 crash 없이 돌아감
- 결과 파일 저장됨
- 이후 STIC 비교 기준으로 사용할 수 있음

**Observed Result**
- 서버 `dkhan_tailscale`에서 1 epoch 실행 성공
- train/val/test loss: `0.1926981 / 0.1086500 / 0.0771806`
- final test metric: `mse=0.07649919`, `mae=0.20616125`
- 저장 확인: `checkpoint.pth`, `metrics.npy`, `pred.npy`, `true.npy`
- 실제 shape: `batch_x [32, 96, 7]`, `batch_y [32, 144, 7]`, `dec_inp [32, 144, 7]`, raw output `[32, 96, 7]`, MS target slice `[32, 96, 1]`

---

### Step 2. Minimal STIC Forward
**Goal**
- `history-only branch + context-aware branch + trust gate`를 포함하는 최소 STIC forward 구현

**Validation**
- dummy tensor forward
- 실제 batch forward
- output dict keys 확인

**Expected Output**
- `pred`
- `pred_h`
- `pred_c`
- `gate`

**Success Criteria**
- forward pass 성공
- shape mismatch 없음
- selective-trust 식 `pred_h + g * (pred_c - pred_h)`가 코드상 명시됨

**Observed Result**
- `models/STIC.py` 추가 완료
- dummy batch output keys: `pred`, `pred_h`, `pred_c`, `gate`
- dummy shape: 모든 출력이 `[4, 96, 1]`
- 실제 ETTh1 batch shape: `pred/pred_h/pred_c/gate = [32, 96, 1]`
- 실제 gate 범위: 약 `[0.478, 0.501]`

---

### Step 3. Training Loop Integration
**Goal**
- 기존 `exp_long_term_forecasting.py`에서 STIC output dict를 처리하고 STIC 전용 loss를 결합

**Validation**
- one-batch backward 성공
- loss finite
- optimizer step 성공

**Success Criteria**
- NaN 없음
- backward 가능

**Observed Result**
- `exp/exp_long_term_forecasting.py`가 tensor output과 STIC dict output을 모두 처리하도록 수정됐다.
- STIC loss는 `pred_loss + 0.1 * aux_loss + 0.1 * gate_loss` 기본값으로 연결했다.
- 서버 `dkhan_tailscale`에서 `--model STIC --train_epochs 1` smoke run 성공
- smoke metric: `mse=0.06885652`, `mae=0.19501153`

---

### Step 4. Sanity Checks
**Goal**
- STIC가 최소한 학습 가능한지 확인

**Validation**
- one-batch overfit
- small-split smoke test
- gate 값 분포 확인

**Success Criteria**
- one-batch loss 유의미하게 감소
- 학습 루프 안정적

**Observed Result**
- one-batch overfit 40 step에서 total loss `0.29299 -> 0.24910` 감소
- 마지막 pred loss는 `0.16344`
- 1 epoch smoke run에서 checkpoint/test metric 저장까지 완료

---

### Step 5. Ablation-Compatible Refactor
**Goal**
- `no-gate / static-gate / history-only / always-on` 설정 지원

**Validation**
- flag별 forward pass
- 동일 train loop 재사용 가능 여부

**Success Criteria**
- 별도 코드 복제 없이 ablation 가능

**Observed Result**
- `run.py`에 `stic_mode`, `stic_static_gate_value`, `stic_aux_weight`, `stic_gate_weight`, `stic_target_index`를 추가했다.
- `models/STIC.py`가 `dynamic`, `static`, `history_only`, `always_on`, `no_gate`를 지원한다.
- `no_gate`는 `always_on` alias로 동작함을 동일 가중치 기준으로 확인했다.
- real-batch forward 확인:
  - `dynamic`: gate trainable `True`, gate mean `0.4499`
  - `static(0.3)`: gate trainable `False`, gate mean `0.3`
  - `history_only`: gate `0`, aux loss disabled
  - `always_on`: gate `1`, aux loss enabled
- 1-epoch smoke metrics:
  - `dynamic`: `mse=0.06885652`, `mae=0.19501153`
  - `static(0.3)`: `mse=0.06796160`, `mae=0.19431528`
  - `history_only`: `mse=0.06920320`, `mae=0.19654149`
  - `always_on`: `mse=0.07005470`, `mae=0.19771333`

---

### Step 6. Corruption Training
**Goal**
- harmful context robustness를 위한 context corruption 추가

**Validation**
- context swap / shuffle / dropout이 train loop에 적용됨
- corrupt batch에서 gate가 낮아지는지 확인

**Success Criteria**
- corruption loss 정상 작동
- clean training은 유지됨

**Implementation Scope**
- corruption 대상은 context channel만으로 제한하고 target history channel은 건드리지 않는다.
- 최소 corruption 모드는 `shuffle`, `swap`, `dropout`, `mixed`를 지원한다.
- corruption은 train loop에서만 적용한다.
- corrupt batch에는 추가 gate regularization으로 낮은 gate를 유도한다.
- clean/corrupt gate mean/std와 `pred_c < pred_h` 조건별 gate 평균을 기록한다.

**Observed Result**
- `run.py`에 `stic_context_corruption_mode`, `stic_context_corruption_prob`, `stic_context_dropout_p`, `stic_context_corruption_gate_weight`, `stic_corrupt_context_aux_weight`를 추가했다.
- `exp/exp_long_term_forecasting.py`가 train loop에서 context-only corruption을 적용하고 clean/corrupt gate diagnostics를 집계하도록 수정됐다.
- corrupt batch에서는 `pred_c` auxiliary loss를 선택적으로 약화할 수 있고, 현재 smoke run은 `stic_corrupt_context_aux_weight=0.0`으로 수행했다.
- 서버 `dkhan_tailscale` smoke run (`mixed`, `prob=0.5`, `gate_weight=0.3`) 결과:
  - final test metric: `mse=0.06718096`, `mae=0.19326566`
  - gate stats: `clean_gate_mean=0.3931`, `corrupt_gate_mean=0.3931`
  - 해석: corruption 경로는 정상 동작하지만 gate는 아직 clean/corrupt를 분리하지 못한다.
- 서버 추가 smoke run (`mixed`, `prob=0.5`, `corruption_gate_weight=1.0`) 결과:
  - final test metric: `mse=0.06465479`, `mae=0.19007719`
  - gate stats: `clean_gate_mean=0.3857`, `corrupt_gate_mean=0.3858`
  - 해석: stronger corruption regularization은 전체 gate level과 test error를 개선했지만, gate collapse 문제를 해결하진 못했다.

---

### Step 6.5 Gate Utility Checkpoint
**Goal**
- gate가 상수 보간 계수가 아니라 context utility 추정기로 학습되는지 먼저 확인

**Validation**
- hard BCE 대신 soft utility target 적용
- 같은 원본 윈도우에서 clean/corrupt gate를 직접 비교
- utility-gate alignment와 paired gate gap을 로그로 확인

**Success Criteria**
- `gate_utility_corr`가 0보다 커짐
- `paired_gate_gap = E[g_clean] - E[g_corrupt]`가 양수
- gate variance와 horizon profile이 완전히 상수 형태에서 벗어남

**Implementation Scope**
- gate target은 `u = err_h - err_c` 기반으로 생성한다.
- `stic_gate_target_mode=soft`일 때 `sigmoid(u / tau)`를 사용한다.
- corruption이 적용된 batch는 clean/corrupt 두 forward를 모두 계산하고 ranking loss를 선택적으로 추가한다.
- 진단 로그에는 `gate_utility_corr`, `paired_gate_gap`, `paired_gate_win_rate`를 포함한다.

**Observed Result**
- 서버 `dkhan_tailscale`에서 `soft target only` smoke run (`tau=0.02`, `mixed corruption`, `prob=0.5`)을 실행했다.
  - final test metric: `mse=0.06453028`, `mae=0.18993369`
  - gate diagnostics:
    - `clean_gate_utility_corr=0.0635`
    - `corrupt_gate_utility_corr=0.0740`
    - `paired_gate_gap=0.0000`
    - `paired_gate_win_rate=0.0000`
- 같은 설정에 `pair_rank_weight=0.1`, `pair_rank_margin=0.05`를 추가한 smoke run도 실행했다.
  - final test metric: `mse=0.06453028`, `mae=0.18993369`
  - gate diagnostics는 soft-only run과 사실상 동일했다.
- 해석:
  - soft utility target은 gate와 utility 사이 약한 양의 상관을 만들었다.
  - 하지만 clean/corrupt gate separation은 아직 형성되지 않았고, 현재 pair rank loss만으로는 gate collapse를 깨지 못했다.
  - 추가 probe 결과, 같은 batch에서 context corruption(`shuffle`, `swap`, `dropout`)을 가해도 `pred_c(clean)`와 `pred_c(corrupt)` 차이가 0이었다.
  - 원인: 현재 `models/STIC.py`의 `context_branch`는 DLinear-style channel-wise branch이고, 최종 `pred_c`는 target channel slice만 사용한다. 따라서 non-target context channel corruption이 target output에 전혀 전파되지 않는다.

---

### Step 6.6 Context-Aware Branch Repair
**Goal**
- context-aware branch가 실제로 non-target exogenous context를 target prediction에 반영하도록 최소 수정

**Validation**
- history-only branch는 그대로 유지
- clean/corrupt same-batch probe에서 `pred_c`와 utility가 실제로 변하는지 확인
- 기존 gate/soft target/corruption 루프와 호환되는지 확인

**Success Criteria**
- `pred_c_abs_delta > 0`
- `utility_drop > 0`

**Implementation Scope**
- `models/STIC.py`의 context-aware branch만 교체한다.
- full multivariate input을 target-specific mixed scalar sequence로 압축한 뒤 DLinear temporal head에 넣는다.
- `pred_h`, gate, corruption path는 기존 구조를 유지한다.

**Observed Result**
- `models/STIC.py`에 `target-specific channel mixer + temporal head` 구조를 추가했다.
- history-only branch는 그대로 두고, context-aware branch만 `full x -> mixed target sequence -> DLinear head`로 교체했다.
- 서버 `dkhan_tailscale` 1-epoch smoke run 결과:
  - final test metric: `mse=0.07869560`, `mae=0.20921060`
  - 해석: 성능은 아직 baseline보다 약하지만, 이번 라운드의 목적은 branch responsiveness 검증이라 metric 최적화는 보류한다.
- 서버 clean/corrupt probe 결과:
  - `shuffle`: `pred_c_abs_delta=0.083488`, `utility_drop=0.021300`
  - `swap`: `pred_c_abs_delta=0.013489`, `utility_drop=0.000772`
  - `dropout`: `pred_c_abs_delta=0.024959`, `utility_drop=0.009128`
  - `pred_h_abs_delta=0` 유지
- 해석:
  - 새 context-aware branch는 처음으로 non-target context corruption에 반응했다.
  - `pred_c_abs_delta > 0`, `utility_drop > 0`가 모두 성립해 구조적 blocker는 해소됐다.
  - 다만 `gate_abs_delta`는 아직 매우 작고 전체 예측 성능도 낮아, 다음 단계는 gate가 아니라 branch 품질 안정화다.

**Branch Sweep Result**
- 서버 `dkhan_tailscale`에서 `linear alpha = {0.05, 0.1, 0.2, 0.5}`와 `mlp alpha = 0.1`을 1 epoch씩 비교했다.
- 요약:
  - `linear alpha=0.05`: `mse=0.06517141`, `mae=0.19062598`
  - `linear alpha=0.1`: `mse=0.06434670`, `mae=0.18978760`
  - `linear alpha=0.2`: `mse=0.06285711`, `mae=0.18849078`
  - `linear alpha=0.5`: `mse=0.06055605`, `mae=0.18780553`
  - `mlp alpha=0.1`: `mse=0.06777434`, `mae=0.19367655`
- `linear alpha=0.5`가 현재 best valid context-responsive branch였다.
- probe 기준:
  - `shuffle`: `pred_c_abs_delta=0.073596`, `utility_drop=0.007964`
  - `swap`: `pred_c_abs_delta=0.034087`, `utility_drop=0.001689`
  - `dropout`: `pred_c_abs_delta=0.018505`, `utility_drop=-0.000271`
- 해석:
  - `linear`가 `mlp`보다 더 단순하면서도 metric과 context responsiveness를 동시에 개선했다.
  - 현재는 `linear alpha=0.5`를 기본 branch로 채택하는 것이 가장 합리적이다.
- `dropout` utility는 아직 미세하게 불안정하므로, 다음 단계는 branch를 더 복잡하게 만드는 것이 아니라 `linear branch` 기준으로 안정화하는 것이다.

---

### Step 6.7 Context Branch Stabilization
**Goal**
- `linear mixer` branch를 기준으로 seed variance와 alpha 민감도를 확인하고 gate 재진입 가능 상태인지 판단

**Validation**
- `linear alpha=0.5` 3-seed 재현
- `alpha=0.3/0.5/0.7` 짧은 재확인
- metric과 probe를 함께 비교

**Success Criteria**
- baseline보다 좋은 성능이 seed를 바꿔도 유지됨
- `shuffle/swap`에서 `utility_drop > 0`가 안정적으로 유지됨
- `dropout` utility sign이 지나치게 흔들리지 않음

**Observed Result**
- 서버 `dkhan_tailscale`에서 `linear alpha=0.5` 3-seed와 `alpha=0.7` 3-seed 비교를 수행했다.
- 결과:
  - `alpha=0.5, seed=2021`: `mse=0.06055605`, `mae=0.18780553`
  - `alpha=0.5, seed=2022`: `mse=0.06972746`, `mae=0.19569139`
  - `alpha=0.5, seed=2023`: `mse=0.07158417`, `mae=0.19895145`
  - `alpha=0.7, seed=2021`: `mse=0.06109189`, `mae=0.18831255`
  - `alpha=0.7, seed=2022`: `mse=0.07264452`, `mae=0.19934960`
  - `alpha=0.7, seed=2023`: `mse=0.07951774`, `mae=0.21008612`
- `alpha=0.5` 3-seed 요약:
  - `mse_mean=0.06728923`, `mse_std=0.00482104`
  - `mae_mean=0.19414946`, `mae_std=0.00467911`
- `alpha=0.7` 3-seed 요약:
  - `mse_mean=0.07108472`, `mse_std=0.00760275`
  - `mae_mean=0.19924942`, `mae_std=0.00888931`
- probe 해석:
  - `alpha=0.5` 평균 corruption utility는 `shuffle=0.019512`, `swap=0.002531`, `dropout=-0.000653`였다.
  - `alpha=0.7` 평균 corruption utility는 `shuffle=0.044095`, `swap=0.003775`, `dropout=-0.001321`였다.
  - `alpha=0.7`는 shuffle 민감도와 평균 gate correlation이 더 컸지만, paired gate win rate는 `0.4565`로 `alpha=0.5`의 `0.4978`보다 낮았다.
- 결론:
  - `linear mixer` 방향 자체는 유지한다.
  - `alpha=0.5`를 STIC 기본 context branch로 확정한다.
  - 이유는 `alpha=0.7`보다 평균 metric과 seed 안정성이 더 좋고, gate 재진입용 기본 branch로 더 균형적이기 때문이다.
  - `alpha=0.7`는 corruption sensitivity 참고 variant로만 유지한다.

---

### Step 8. Gate Re-Entry Diagnostics
**Goal**
- stabilized `linear alpha=0.5` branch 위에서 gate가 실제 context utility를 더 잘 반영하는지 다시 측정

**Validation**
- soft utility target 유지
- pair rank는 끈 상태로 clean/corrupt gate diagnostics 재집계
- seed 평균 기준으로 gate-utility correlation, clean/corrupt gate gap, paired gate win rate, horizon profile 비교

**Success Criteria**
- gate-utility correlation이 branch repair 전보다 개선됨
- clean/corrupt gate gap이 양수로 유지되거나 paired gate win rate가 0.5를 넘음
- horizon-wise gate profile이 거의 상수인 상태에서 벗어남

**Current Readout**
- `linear alpha=0.5` branch를 고정한 뒤 G1 계열 gate input screening을 다시 수행했다.
  - `g0`: `mse=0.06055605`, `mae=0.18780553`, `clean_corr=0.1169`, `gap=0.0003`, `win=0.5383`
  - `g1a`: `mse=0.06450973`, `mae=0.19199406`, `clean_corr=0.0385`, `gap=-0.0005`, `win=0.4312`
  - `g1b`: `mse=0.06349987`, `mae=0.19030987`, `clean_corr=0.1362`, `gap=0.0011`, `win=0.5623`
  - `g1c`: `mse=0.06758192`, `mae=0.19487345`, `clean_corr=0.2502`, `gap=0.0009`, `win=0.5543`
- 1-seed 기준으로는 `g1b`가 우선순위(`gap -> win -> clean_corr -> metric`)에 가장 잘 맞았다.
- 동일 코드 기준 3-seed 비교는 다음과 같다.
  - `g0`: `mse_mean=0.06728923`, `mse_std=0.00482104`, `mae_mean=0.19414946`, `mae_std=0.00467911`, `clean_corr_mean=0.0864`, `gap_mean=0.0002`, `win_mean=0.4978`
  - `g1b`: `mse_mean=0.07205066`, `mse_std=0.00889805`, `mae_mean=0.20083574`, `mae_std=0.01163550`, `clean_corr_mean=0.0830`, `gap_mean=0.0010`, `win_mean=0.5768`
- 해석:
  - `g1a`는 mean-only summary가 너무 약해서 separation과 metric 모두 나빠졌다.
  - `g1c`는 correlation은 가장 크게 올렸지만 metric 손실이 크고, `g1b`보다 `gap`과 `win`이 약했다.
  - `g1b`는 `gap`과 `win`을 동시에 가장 잘 올린 현재 최선의 redesign이다.
  - 다만 `clean_corr_mean`은 G0를 안정적으로 넘지 못했고 metric도 더 나빠져, 아직 기본 gate input을 교체할 단계는 아니다.

---

### Step 8.2 Compact G1 Screening
**Goal**
- `linear alpha=0.5` branch를 고정한 상태에서 compact한 G1 summary가 separation을 유지하면서 metric 손실을 줄일 수 있는지 확인

**Validation**
- branch, soft utility target, corruption path, pair-rank off 조건을 그대로 유지
- `g0`, `g1b`, `g1-lite`, `g1-diff`, `g1-norm`를 동일한 1-seed 기준으로 비교
- best compact candidate만 3-seed로 확장

**Success Criteria**
- `paired_gate_gap`이 `g0`보다 커짐
- `paired_gate_win_rate`가 `0.5` 이상 유지되거나 `g0`보다 개선됨
- `clean_gate_utility_corr`가 유지 또는 상승하면서 metric 손실이 `g1b`보다 줄어듦

**Implementation Scope**
- `models/STIC.py`에서 gate input summary만 바꾼다.
- `run.py`에는 compact gate mode만 추가한다.
- `exp_long_term_forecasting.py`와 branch 구조, alpha, corruption, loss objective는 건드리지 않는다.

**Open Question**
- `g1b`의 metric 손실은 mean+std summary 자체의 tradeoff인지, history summary까지 같이 넣는 설계의 noise인지 아직 분리되지 않았다.

**Observed Result**
- `g0`, `g1b`, `g1-lite`, `g1-diff`, `g1-norm`를 1-seed로 비교했다.
  - `g0`: `mse=0.06055605`, `mae=0.18780553`, `clean_corr=0.1169`, `gap=0.0003`, `win=0.5383`
  - `g1b`: `mse=0.06349987`, `mae=0.19030987`, `clean_corr=0.1362`, `gap=0.0011`, `win=0.5623`
  - `g1-lite`: `mse=0.06356514`, `mae=0.19012345`, `clean_corr=0.2085`, `gap=0.0005`, `win=0.5384`
  - `g1-diff`: `mse=0.06158479`, `mae=0.18729742`, `clean_corr=0.0652`, `gap=-0.0004`, `win=0.4250`
  - `g1-norm`: `mse=0.06293690`, `mae=0.18883191`, `clean_corr=-0.1221`, `gap=0.0040`, `win=0.5409`
- 우선순위(`gap -> win -> clean_corr -> metric`) 기준으로는 `g1b`가 여전히 best였다.
- 해석:
  - `g1-lite`는 가장 balanced하다. `clean_corr`가 가장 높고 metric 손실도 작지만, `gap` 이득이 `g1b`보다 약하다.
  - `g1-diff`는 signed difference만으로는 gate signal이 너무 약해 separation과 win rate가 모두 무너졌다.
  - `g1-norm`은 `gap`은 가장 크게 키웠지만 `clean_corr`가 음수로 뒤집혀 utility alignment 관점에서는 실패다.
  - 이번 라운드에서는 새 compact mode가 `g1b`를 명확히 이기지 못했다. 따라서 3-seed 확장 후보는 계속 `g1b`다.
  - 추가로 `g1-lite` 3-seed를 확인한 결과:
    - `mse_mean=0.07580999`, `mse_std=0.00951202`
    - `mae_mean=0.20569150`, `mae_std=0.01227263`
    - `clean_corr_mean=0.1247`, `corrupt_corr_mean=0.1188`
    - `gap_mean=0.0005`, `win_mean=0.5402`
  - 해석:
    - `g1-lite`는 `clean_corr_mean`은 `g0`/`g1b`보다 높지만, `gap_mean`과 `win_mean`은 `g1b`보다 작다.
    - metric 평균도 `g1b`보다 더 나빠, compact track 기준으로는 “balanced but not best”로 보는 것이 맞다.

---

### Step 8.3 Slice Analysis
**Goal**
- `g0 / g1-lite / g1b`의 separation gain과 metric tradeoff가 utility slice, horizon slice, corruption type 중 어디에서 갈리는지 설명

**Validation**
- seed 2021 checkpoint 기준으로 clean + deterministic corruption 재평가
- utility slice는 `g0` clean sample utility tercile(`bottom/middle/top`)로 고정
- horizon slice는 `1-24 / 25-48 / 49-72 / 73-96`
- corruption type은 `shuffle / swap / dropout`

**Success Criteria**
- `g1b` separation 이득이 발생하는 slice를 설명할 수 있음
- `g1-lite`의 metric/separation tradeoff를 slice별로 설명할 수 있음
- 다음 단계가 `g1b refinement`인지 `g0 유지 + 보조 분석`인지 결정 가능

**Observed Result**
- utility slice:
  - `g1b` 이득은 `bottom/middle` slice에 집중된다. `bottom`에서 `gap=0.0014`, `middle`에서 `gap=0.0013`으로 `g0`보다 크다.
  - `top` slice에서는 `g1b`와 `g1-lite` 모두 gap이 무너지며(`g1b=-0.0004`, `g1-lite=0.0004`) metric도 크게 악화된다.
  - `g1-lite`는 `middle` slice에서 `mse=0.0302`, `win=0.5337`로 `g1b`보다 더 balanced했다.
- horizon slice:
  - `g1-lite`와 `g1b` 모두 전 horizon에서 `g0`보다 positive gate gap을 만든다.
  - `g1-lite`는 전 bucket에서 `gate_gap≈0.0010`, `g1b`는 `≈0.0007`이다.
  - 다만 `g1b`는 `h1_24`에서 더 낮은 MSE를 보이고, 이후 구간은 `g1-lite`와 비슷하거나 약간 불리하다.
- corruption type:
  - `shuffle`에서 `g1b` separation 이득이 가장 크다: `gap=0.0037`, `win=0.6249`, `metric_drop=0.0016`.
  - `dropout`에서는 `g1-lite`가 더 낫다: `gap≈0.0000`, `win=0.4590` vs `g1b`의 `gap=-0.0007`, `win=0.3243`.
  - `swap`에서는 두 모델 모두 separation이 약하지만 `g1-lite`가 덜 나쁘다.
- 결론:
  - `g1b`는 aggressive separation candidate이고, 이득은 주로 `shuffle + low/mid utility` 구간에서 나온다.
  - `g1-lite`는 balanced candidate로, `middle utility`와 `dropout/swap`에서 tradeoff가 더 낫다.
  - 현재 단계의 결정은 `g0` 기본 유지 + `g1b refinement`가 가장 합리적이다.

---

### Step 8.4 G1b Refinement Screening
**Goal**
- `g1b`의 strong separation(`low/mid utility`, `shuffle`)은 유지하면서 `top utility`와 `dropout/swap` failure를 완화하는 refinement를 찾는다.

**Validation**
- branch, alpha, soft utility target, corruption path, pair-rank off를 그대로 유지한다.
- `g0`, `g1b`, `g1b-meanheavy(beta_std=0.5)`, `g1b-diff-lite`, `g1b-topclip(gamma_hidden=0.5)`를 1-seed로 비교한다.
- 1-seed best 후보만 3-seed로 확장한다.

**Success Criteria**
- `top utility` slice의 MSE 악화가 `g1b`보다 줄어듦
- `dropout/swap`에서 negative gap 또는 low win-rate가 완화됨
- `low/mid utility + shuffle`에서의 `g1b` separation 이점은 최대한 유지됨
- 전체 metric이 `g1b`보다 개선되거나 최소 비슷함

**Implementation Scope**
- `models/STIC.py`와 `run.py`에 refinement gate input만 추가한다.
- `scripts/analyze_stic_gate_slices.py`로 utility/horizon/corruption 분석을 재사용한다.
- `exp_long_term_forecasting.py`, branch 구조, alpha, corruption 종류, objective는 수정하지 않는다.

**Open Question**
- `g1b`의 failure는 std summary가 과도해서 생기는지, diff redundancy 때문인지, hidden summary scale이 너무 큰지 아직 분리되지 않았다.

**Observed Result**
- `models/STIC.py`와 `run.py`에 `g1b-meanheavy`, `g1b-diff-lite`, `g1b-topclip`를 추가하고, `stic_gate_std_scale`, `stic_gate_hidden_scale` CLI를 연결했다.
- 서버 `dkhan_tailscale` 1-seed screening 결과:
  - `g0`: `mse=0.06055605`, `mae=0.18780553`, `gap=0.0003`, `win=0.5383`
  - `g1b`: `mse=0.06349987`, `mae=0.19030987`, `gap=0.0011`, `win=0.5623`
  - `g1b-meanheavy(beta_std=0.5)`: `mse=0.06348336`, `mae=0.19036694`, `gap=0.0008`, `win=0.5294`
  - `g1b-diff-lite`: `mse=0.06158479`, `mae=0.18729742`, `gap=-0.0004`, `win=0.4250`
  - `g1b-topclip(gamma_hidden=0.5)`: `mse=0.06087272`, `mae=0.18708013`, `gap=0.0005`, `win=0.5153`
- slice analysis 기준:
  - `g1b-topclip`는 `top utility` MSE를 `0.1247 -> 0.1179`로 줄였고, `dropout` failure도 `gap=-0.0007 -> -0.0003`, `win=0.3243 -> 0.3698`로 완화했다.
  - 동시에 `shuffle`에서는 `gap=0.0030`, `win=0.6391`로 `g1b`의 separation 이점을 대부분 유지했다.
  - `g1b-diff-lite`는 `top utility`와 `swap`은 좋아졌지만 `low/mid + shuffle` separation이 무너져 탈락했다.
  - `g1b-meanheavy`는 `swap`은 일부 개선했지만 `dropout`과 `top utility`에서 충분한 완화가 없었다.
- 따라서 1-seed best refinement는 `g1b-topclip`으로 결정했고, 추가로 3-seed를 수행했다.
- `g1b-topclip` 3-seed 결과:
  - `mse_mean=0.07107104±0.00979103`
  - `mae_mean=0.19940976±0.01281509`
  - `clean_corr_mean=0.0696`
  - `gap_mean=0.0004`
  - `win_mean=0.5656`
- 해석:
  - `g1b-topclip`는 `g1b` 대비 metric 손실을 줄이고 failure slice를 완화하는 방향으로는 성공했다.
  - 하지만 3-seed 평균 separation(`gap`, `win`)은 여전히 `g1b`보다 약하고 seed variance도 남아 있어, 아직 기본 gate input을 교체할 단계는 아니다.
- 현재 결정은 `g0` default 유지, `g1b` strongest-separation reference 유지, `g1b-topclip`을 가장 유망한 refinement 후보로 보류하는 것이다.

---

### Step 8.5 G1b-Topclip 3-Seed Slice Validation
**Goal**
- `g1b-topclip`가 seed 2021 단발성이 아니라 3-seed 평균에서도 `g1b`의 failure pattern을 실제로 완화하는지 검증한다.

**Validation**
- `g1b`와 `g1b-topclip`만 비교한다.
- seed는 `2021 / 2022 / 2023`으로 고정한다.
- utility slice는 각 seed에서 `g1b` clean utility tercile(`bottom/middle/top`)을 기준으로 정의한다.
- utility / horizon / corruption type별 평균과 표준편차를 기록한다.

**Success Criteria**
- `top utility` slice에서 `g1b-topclip`의 MSE가 평균적으로 더 낮다.
- `dropout` 또는 `swap`에서 failure metric(gap, win-rate, utility_drop)이 `g1b`보다 완화된다.
- `shuffle` separation 이득을 크게 잃지 않는다.
- 전체 metric이 `g1b`와 비슷하거나 더 좋다.

**Observed Result**
- `scripts/analyze_stic_gate_slices_multiseed.py`를 추가해 `g1b` reference slice 기준 3-seed aggregate를 계산했다.
- utility slice 평균:
  - `top` MSE는 `g1b 0.0390±0.0113 -> g1b-topclip 0.0377±0.0113`로 개선됐다.
  - `middle`도 `0.0389±0.0048 -> 0.0379±0.0053`, `bottom`도 `0.1381±0.0336 -> 0.1375±0.0357`로 소폭 개선됐다.
  - 반면 `paired_gate_gap`은 모든 slice에서 `g1b-topclip`가 더 작았다.
- horizon slice 평균:
  - `g1b-topclip`는 전 horizon bucket에서 MSE/MAE가 소폭 더 낮고, `utility_drop`도 0에 더 가깝다.
  - 다만 `gate_gap`은 `g1b 0.0022±0.0012` 대비 `0.0015±0.0007`로 줄었다.
- corruption type 평균:
  - `shuffle`: `gap 0.0045±0.0013 -> 0.0030±0.0008`로 약해졌지만 `win 0.6161±0.0072 -> 0.6235±0.0167`은 유지/소폭 개선됐다.
  - `swap`: `gap -0.0004±0.0005 -> -0.0003±0.0003`, `win 0.4469±0.0526 -> 0.4480±0.0497`로 아주 약한 개선만 있었다.
  - `dropout`: `gap 0.0025±0.0023 -> 0.0016±0.0014`로 줄었지만 `win 0.7528±0.3030 -> 0.7675±0.2816`, `utility_drop -0.0050±0.0067 -> -0.0046±0.0058`로 완화 신호는 유지됐다.
- 전체 metric 비교:
  - `g1b` 3-seed: `mse=0.07205066±0.00889805`, `mae=0.20083574±0.01163550`
  - `g1b-topclip` 3-seed: `mse=0.07107104±0.00979103`, `mae=0.19940976±0.01281509`
- 해석:
  - `g1b-topclip`는 seed 평균에서도 `top utility` MSE와 overall metric을 줄였다.
  - `swap/dropout` failure는 약하게 완화됐지만, `shuffle`에서는 gap을 일부 희생했다.
  - 따라서 현재 판단은 `g1b-topclip` 승격이 아니라, `g1b` strongest-separation reference 유지 + `g1b-topclip` refinement candidate 유지가 가장 타당하다.

---

### Step 8.6 Weaker Hidden-Summary Refinement Search
**Goal**
- `g1b-topclip`의 `top/dropout` 완화는 유지하면서 `shuffle` gap 손실을 덜 내는 더 약한 hidden-summary refinement를 찾는다.

**Validation**
- branch, alpha, loss, corruption, gate objective는 그대로 유지한다.
- 1-seed screening 후보는 `g1b`, `g1b-topclip(0.5)`, `g1b-topclip-lite(0.75/0.875)`, `g1b-sumreg-rms`, `g1b-sumreg-clip`이다.
- best new candidate만 `2021 / 2022 / 2023` 3-seed로 확장한다.

**Success Criteria**
- `top utility` MSE는 `g1b`보다 낮고, 가능하면 `g1b-topclip`와 비슷하다.
- `dropout/swap` failure는 `g1b`보다 완화된다.
- `shuffle` gap은 `g1b-topclip`보다 덜 줄어든다.
- overall metric은 `g1b-topclip`과 비슷하거나 더 좋다.

**Observed Result**
- `models/STIC.py`와 `run.py`에 `g1b-topclip-lite`, `g1b-sumreg-rms`, `g1b-sumreg-clip`를 추가하고, hidden scale / summary regularization CLI를 연결했다.
- 1-seed screening 결과:
  - `g1b`: `mse=0.06349987`, `mae=0.19030987`
  - `g1b-topclip`: `mse=0.06087272`, `mae=0.18708013`
  - `g1b-topclip-lite-0.75`: `mse=0.06213221`, `mae=0.18860379`
  - `g1b-topclip-lite-0.875`: `mse=0.06283351`, `mae=0.18947007`
  - `g1b-sumreg-rms`: `mse=0.06262263`, `mae=0.18933190`
  - `g1b-sumreg-clip`: `mse=0.05996817`, `mae=0.18659467`
- 1-seed slice analysis 기준:
  - `g1b-topclip-lite-0.75`가 `top` MSE `0.0534`로 new candidates 중 가장 `g1b-topclip(0.0528)`에 가까웠고, `shuffle` gap은 `0.0037`로 `g1b` 수준까지 회복했다.
  - `g1b-sumreg-clip`은 overall metric과 `dropout` 완화는 가장 좋았지만 `top` MSE `0.0545`로 `g1b(0.0537)`보다도 나빠져 탈락했다.
  - 따라서 best new candidate는 `g1b-topclip-lite-0.75`로 선택했다.
- `g1b-topclip-lite-0.75` 3-seed 결과:
  - `mse=0.07163194±0.00950216`
  - `mae=0.20022704±0.01246099`
  - `clean_corr=0.0768±0.0412`
  - `gap=0.0007±0.0004`
  - `win=0.5729±0.0292`
- 3-seed slice analysis 기준:
  - `top` MSE는 `g1b 0.0390±0.0113`, `g1b-topclip 0.0377±0.0113`, `topclip-lite-0.75 0.0384±0.0114`였다.
  - `shuffle` gap은 `g1b 0.0045±0.0013`, `g1b-topclip 0.0030±0.0008`, `topclip-lite-0.75 0.0040±0.0013`으로 topclip 대비 대부분 회복했다.
  - `dropout` gap/win은 `topclip-lite-0.75`가 `g1b`보다는 나았지만 `g1b-topclip`보다는 약간 약했다.
- 해석:
  - `g1b-topclip-lite-0.75`는 `shuffle` gap 회복에는 성공했다.
  - 하지만 `top utility` 완화와 overall metric에서는 `g1b-topclip`를 명확히 넘지 못했고, `dropout/swap` 완화도 topclip 대비 우위가 크지 않았다.
  - 현재 결론은 `g1b` strongest-separation reference 유지, `g1b-topclip` best mitigation reference 유지, `g1b-topclip-lite-0.75`는 best compromise candidate로 보류하는 것이다.

---

### Step 9. Evaluation Utilities
**Goal**
- clean / corrupt / drift-slice / gate calibration 평가 코드 작성

**Validation**
- clean MSE/MAE 계산
- corruption별 성능 저하율 계산
- gate summary 출력

**Success Criteria**
- 표 형태 결과를 만들 수 있음

---

### Step 10. ETTh1 Reference Freeze
**Goal**
- ETTh1 결과를 frozen reference package로 잠그고 이후 numeric/text generalization의 비교 기준으로 사용한다.

**Deliverables**
- `reports/ETTh1_final_summary.md`
- `reports/CiK_integration_memo.md`와 분리된 ETTh1 전용 trade-off package

**Validation**
- overall metric
- utility slice
- horizon slice
- corruption type
- gate role summary

**Observed Result**
- `reports/ETTh1_final_summary.md`를 작성했다.
- frozen gate set은 `g0 / g1b / g1b-topclip / lite-0.75`로 확정했다.
- 요약:
  - `g0`: best average STIC metric
  - `g1b`: strongest separation reference
  - `g1b-topclip`: best mitigation reference
  - `lite-0.75`: compromise candidate
- ETTh1 3-seed aggregate slice 결과를 문서에 고정했고, 이후 ETTh1 코드는 더 수정하지 않는 방향으로 잠갔다.

---

### Step 11. Exchange Numeric Generalization
**Goal**
- ETTh1에서 정리한 selective-trust 메시지가 Exchange에서도 더 강하게 드러나는지 확인한다.

**Validation**
- dataset wiring 확인
- DLinear baseline reproduction
- STIC 4-way 1-seed screening
- informative top-2 gate 3-seed follow-up
- utility / horizon / corruption slice analysis

**Observed Result**
- Exchange dataset wiring을 `custom + MS + target=OT + seq_len=96 + pred_len=96 + enc_in=8`로 고정했다.
- DLinear baseline:
  - 1-seed `mse=0.12607`, `mae=0.28877`
  - 3-seed `mse=0.12461±0.00412`, `mae=0.28792±0.00535`
- STIC 1-seed screening:
  - `g0`: `mse=0.13046`, `mae=0.28922`, `clean_corr=-0.0432`, `gap=0.0028`, `win=0.4987`
  - `g1b`: `mse=0.13048`, `mae=0.29087`, `clean_corr=0.1348`, `gap=0.0031`, `win=0.6500`
  - `g1b-topclip`: `mse=0.13890`, `mae=0.30041`, `clean_corr=0.1329`, `gap=0.0011`, `win=0.5784`
  - `lite-0.75`: `mse=0.13421`, `mae=0.29523`, `clean_corr=0.1357`, `gap=0.0020`, `win=0.6261`
- informative top-2 gate는 `g0`와 `g1b`로 선정했다.
- `g0` 3-seed:
  - `mse=0.12666±0.00821`
  - `mae=0.28746±0.01620`
  - `clean_corr=-0.0342±0.0111`
  - `gap=0.0017±0.0010`
  - `win=0.5728±0.0921`
- `g1b` 3-seed:
  - `mse=0.13231±0.00929`
  - `mae=0.29636±0.01651`
  - `clean_corr=0.0662±0.0511`
  - `gap=0.0022±0.0011`
  - `win=0.6358±0.0541`
- slice 해석:
  - `g1b`는 utility tercile 전 구간에서 `g0`보다 높은 paired win-rate를 보였다.
  - horizon bucket 전 구간에서 `g1b` gate gap은 양수였고, `g0`는 거의 0에 머물렀다.
  - `shuffle` 및 `dropout`에서 `g1b`의 selective-trust signal이 더 강했다.
- 결론:
  - Exchange는 ETTh1보다 stronger selective-trust signal을 보여주는 첫 numeric generalization dataset이다.
  - 다만 average metric 기준 default는 여전히 `g0`, mechanism reference는 `g1b`다.

---

### Step 12. CiK Minimal Prototype Planning
**Goal**
- STIC를 text-conditioned context branch로 옮기기 위한 최소 prototype 설계를 만든다.

**Deliverables**
- `reports/CiK_integration_memo.md`

**Validation**
- task interface 확인
- candidate task 1~2개 선정
- 최소 코드 변경 파일 목록 정리

**Observed Result**
- CiK interface를 확인했고, `BaseTask` / `UnivariateCRPSTask` 기준으로 `past_time`, `future_time`, `background`, `scenario`, `constraints`를 활용할 수 있음을 정리했다.
- first prototype candidate는 `ElectricityIncreaseInPredictionTask` 계열로 선정했다.
- second candidate는 `OraclePredUnivariateConstraintsTask` family로 정리했다.
- 최소 prototype plan은 `history-only branch + text-conditioned context-aware branch + trust gate`를 유지하면서 thin adapter로 CiK evaluation callable에 연결하는 방향으로 고정했다.

---

### Step 12.5. CiK Thin Prototype Implementation
**Goal**
- `ElectricityIncreaseInPredictionTask`와 `OraclePredUnivariateConstraintsTask`에서 history-only, text-conditioned context-aware, gated final prediction을 thin runner로 end-to-end 실행한다.

**Deliverables**
- `scripts/cik/run_stic_cik.py`
- `utils/cik_stic.py`

**Validation**
- server-side `inspect_cik_task.py` shape sanity
- server-side `run_stic_cik.py` inference/evaluation
- debug output:
  - batch history/future shape
  - `pred_h / pred_c / gate / pred` shape
  - sample text preview
  - parsed effect summary
  - ROI or constraint summary

**Success Criteria**
- Electricity task에서 `pred_h != pred_c`가 실제로 발생한다.
- gate가 trivial constant가 아니고 `pred = pred_h + g * (pred_c - pred_h)`를 만든다.
- Oracle constraints task도 같은 STIC skeleton으로 wiring된다.

**Observed Result**
- `utils/cik_stic.py`를 rule-based text-conditioned branch + simple g0-style prediction gate로 정리했다.
- history-only branch는 seasonal replay + clipped tail trend baseline으로 고정했다.
- Electricity parser는 heat-wave timestamp, duration, multiplier/percent increase를 읽어 horizon-local multiplicative adjustment를 적용한다.
- Oracle parser는 textual lower/upper bound를 읽어 clipping-based context-aware branch를 만든다.
- `scripts/cik/run_stic_cik.py`는 `--task-name --split --limit --batch-size --device --max-samples` CLI와 dataloader 기반 batch debug/eval path를 지원한다.
- server-side smoke result (`split=test`, available rows=5):
  - Electricity: `mse_h=4828618.6591`, `mse_c=40520.9670`, `mse=318499.4241`, `mae_h=511.0727`, `mae_c=106.6161`, `mae=182.7227`, `roi_mae=1033.3815`, `gate_mean=0.1586`
  - Oracle: `mse_h=2.0413`, `mse_c=1.6641`, `mse=1.7774`, `mae_h=0.6037`, `mae_c=0.5437`, `mae=0.5693`, `gate_mean=0.1372`
- Electricity batch debug shape: `pred_h/pred_c/gate/pred = (4, 24, 1)`
- Oracle batch debug shape: `pred_h/pred_c/gate/pred = (4, 24, 1)`
- CiK HF dataset currently exposes only `test` split, so this round remains a thin prototype validation rather than a larger-sample benchmark.

---

### Step 13. S3-Gate Simplification
**Goal**
- `g1b`의 full hidden-summary gate를 더 단순한 bucket-wise sufficient-statistic shrinkage gate(S3-Gate)로 치환할 수 있는지 검증한다.

**Deliverables**
- ETTh1 sanity summary for `s3-soft`, `s3-oracle`
- Exchange 1-seed screening for `g0`, `g1b`, `s3-soft`, `s3-oracle`
- best S3 variant 3-seed summary against Exchange `g0`, `g1b`

**Validation**
- bucket gate shape test: `gate_bucket_values [B, 4, 1]`, expanded `gate [B, 96, 1]`
- ETTh1 smoke run without metric collapse
- Exchange 1-seed screening with utility/horizon/corruption slices
- best S3 variant 3-seed mean/std comparison

**Observed Result**
- `models/STIC.py`에 S3-Gate를 추가했다. S3 input은 bucket별 `m_bucket`과 전역 `d_mu`, `d_sigma`만 사용하고, monotone shrinkage gate를 `sigmoid(alpha + w_m*m + w_mu*d_mu - w_sigma*d_sigma)`로 구현했다.
- `exp_long_term_forecasting.py`는 `s3-oracle`에서만 bucket-wise oracle shrinkage teacher를 계산하도록 최소 수정했다. oracle teacher는 training target으로만 쓰고 inference input에는 미래 target을 사용하지 않는다.
- ETTh1 sanity:
  - `s3-soft`: `mse=0.06489874`, `mae=0.19240686`
  - `s3-oracle`: `mse=0.06490658`, `mae=0.19241823`
  - 두 모드 모두 shape/forward/backward/train loop는 정상이고 metric 붕괴는 없었다.
- Exchange 1-seed:
  - `g0`: `mse=0.13045767`, `mae=0.28921866`, `clean_corr=-0.0432`, `gap=0.0028`, `win=0.4987`
  - `g1b`: `mse=0.13047999`, `mae=0.29086992`, `clean_corr=0.1348`, `gap=0.0031`, `win=0.6500`
  - `s3-soft`: `mse=0.17040557`, `mae=0.33048159`, `clean_corr=-0.0026`, `gap=0.0041`, `win=0.6195`
  - `s3-oracle`: `mse=0.17035396`, `mae=0.33043465`, `clean_corr=-0.0028`, `gap=0.0041`, `win=0.6192`
- best S3 variant는 `s3-soft`로 두었다. `s3-oracle`은 teacher만 달랐고 실질 metric과 separation이 거의 동일했다.
- Exchange `s3-soft` 3-seed:
  - `mse=0.13920003±0.02216821`
  - `mae=0.30366290±0.02047961`
  - `clean_corr=-0.1232±0.1130`
  - `gap=0.0038±0.0005`
  - `win=0.6213±0.0060`
- 결론:
  - S3는 `g1b`보다 단순한 gate input으로 동작하지만, current Exchange setting에서는 gap만 키우고 utility alignment와 average metric을 같이 잃는다.
  - 따라서 현재 reference 체계는 유지한다: `g0` default, `g1b` mechanism reference. S3는 negative-but-informative simplification result로 기록한다.

---

### Step 14. Hybrid S3 Evaluation
**Goal**
- `g0`의 directional bucket summary와 S3의 저차원 sufficient statistics를 결합한 Hybrid S3(`hs3`, `hs3-lite`)가 `g1b`보다 더 나은 accuracy-separation trade-off를 내는지 검증한다.

**Deliverables**
- ETTh1 sanity summary for `g0`, `g1b`, `hs3-soft`, `hs3-oracle`, `hs3-lite-soft`
- Exchange 1-seed screening summary
- Exchange best Hybrid S3 3-seed summary
- Exchange 3-seed slice aggregate (`g0`, `g1b`, best hs3)

**Validation**
- dummy/forward shape
- one-batch overfit
- ETTh1 1-epoch sanity
- Exchange 1-seed screening + utility/horizon/corruption slice analysis
- Exchange 3-seed mean/std summary and multiseed slice aggregation

**Observed Result**
- `models/STIC.py`에 `hs3`, `hs3-lite`를 추가했다. `hs3` input은 bucket별 `[y_h_bar, y_c_bar, delta_bar, m_bucket, a_bucket, d_sigma]`, `hs3-lite`는 마지막 `d_sigma`를 제거한 `[y_h_bar, y_c_bar, delta_bar, m_bucket, a_bucket]`를 사용한다.
- `hs3`/`hs3-lite`는 bucket-wise scalar gate를 예측한 뒤 `[B, 96, 1]` horizon gate로 확장한다. local dummy shape는 `hs3 gate_stats=[B,4,6]`, `hs3-lite gate_stats=[B,4,5]`, `gate_bucket_values=[B,4,1]`였다.
- one-batch overfit (`hs3-oracle`, ETTh1 train batch 20 step):
  - `start_loss=0.23758`
  - `end_loss=0.21126`
  - `pred_shape=(32,96,1)`, `bucket_shape=(32,4,1)`
- ETTh1 sanity (`seed=2021`):
  - `g0`: `mse=0.06055605`, `mae=0.18780553`
  - `g1b`: `mse=0.06349987`, `mae=0.19030987`
  - `hs3-soft`: `mse=0.07288060`, `mae=0.20353019`, `clean_corr=-0.1343`, `gap=-0.0045`, `win=0.5674`
  - `hs3-oracle`: `mse=0.07283033`, `mae=0.20343855`, `clean_corr=-0.1319`, `gap=-0.0046`, `win=0.5668`
  - `hs3-lite-soft`: `mse=0.07305948`, `mae=0.20372717`, `clean_corr=-0.1338`, `gap=-0.0060`, `win=0.5574`
- Exchange 1-seed (`seed=2021`):
  - `g0`: `mse=0.13045767`, `mae=0.28921866`, `clean_corr=-0.0432`, `gap=0.0028`, `win=0.4987`
  - `g1b`: `mse=0.13047999`, `mae=0.29086992`, `clean_corr=0.1348`, `gap=0.0031`, `win=0.6500`
  - `hs3-soft`: `mse=0.20816092`, `mae=0.36535507`, `clean_corr=-0.0499`, `gap=-0.0031`, `win=0.5256`
  - `hs3-oracle`: `mse=0.20784783`, `mae=0.36508825`, `clean_corr=-0.0501`, `gap=-0.0035`, `win=0.5222`
  - `hs3-lite-soft`: `mse=0.20819175`, `mae=0.36534145`, `clean_corr=-0.0487`, `gap=-0.0051`, `win=0.5196`
- best Hybrid S3 candidate는 `hs3-oracle`로 정했다. 이유는 `hs3-soft`와 동급 separation에서 metric이 가장 낮고, `hs3-lite`보다 sign instability가 덜했기 때문이다.
- Exchange `hs3-oracle` 3-seed:
  - `mse=0.17156475±0.02745911`
  - `mae=0.33928200±0.02461116`
  - `clean_corr=-0.0585±0.0466`
  - `corrupt_corr=-0.0964±0.0197`
  - `gap=0.0001±0.0034`
  - `win=0.5548±0.0321`
- Exchange 3-seed slice aggregate (`reference_label=g1b`)에서도 `hs3-oracle`은 `g1b` 대비 전 utility slice에서 metric이 더 나빴고, `middle` utility slice `clean_corr=-0.0570±0.0574`, `paired_gap=-0.0048±0.0017`로 utility-aligned separation이 회복되지 않았다.
- corruption aggregate에서도 `hs3-oracle`은 `shuffle metric_drop=0.0541±0.0115`, `paired_gap=-0.0408±0.0078`, `win=0.2713±0.0460`로 `g0`/`g1b`보다 훨씬 나빴다. `dropout`에서는 `paired_gap=0.0446±0.0085`, `win=0.9582±0.0459`로 강했지만 이는 metric-drop trade-off를 정당화할 수준은 아니었다.
- 결론:
  - Hybrid S3는 `g0`의 directional signal을 되살렸지만, current Exchange setting에서는 `clean_corr`와 average metric을 회복하지 못했다.
  - 따라서 current gate references는 유지한다: `g0` default, `g1b` strongest separation, `g1b-topclip` mitigation, `lite-0.75` compromise. Hybrid S3는 second negative-but-informative simplification result로 잠근다.

## Tensor Shape Notes
- input `x`: `[B, L, D]`
- target history `x_tar`: `[B, L, 1]`
- context `x_ctx`: `[B, L, D-1]`
- baseline DLinear output before MS slice: `[B, H, D]`
- final prediction `pred`: `[B, H, 1]`
- history-only prediction `pred_h`: `[B, H, 1]`
- context-aware prediction `pred_c`: `[B, H, 1]`
- gate `g`: `[B, H, 1]`

## Leakage Checklist
- [x] sliding window가 미래 target 포함하지 않음
- [x] decoder input이 미래 ground truth를 사용하지 않음
- [x] context가 미래 시점 정보 포함하지 않음
- [x] train/val/test split이 시간순 유지
- [x] normalization 통계가 split 규칙 위반하지 않음

## Experiment Configuration
- Dataset: `ETTh1`
- Backbone: `DLinear`
- seq_len: `96`
- pred_len: `96`
- features: `MS`
- target: `OT`
- batch_size: `32`
- learning_rate: `1e-4`
- seed: `2021`
- precision: `fp32` first, `amp` later
- compile on/off: `off` for first milestone

## Risks
1. STIC loss 기본값과 static gate 값이 초기 휴리스틱이라 dataset별 재튜닝 가능성이 높다.
2. `features=MS`에서 target이 마지막 채널이라는 가정을 STIC 내부에서 명시해야 하며, dataset별 열 순서 차이가 있으면 일반화가 깨질 수 있다.
3. optional dependency가 data loader top-level import에 묶여 있어 task와 무관한 패키지 누락도 실행을 막을 수 있다.

## Decisions Made
- 첫 구현은 `DLinear` backbone만 지원한다.
- 첫 task는 `long_term_forecast`만 지원한다.
- 첫 구현은 data loader를 수정하지 않고 model 내부에서 `target/context`를 분리하는 방향으로 간다.
- 첫 기준 설정은 `features=MS`, `target=OT`, target channel last를 전제로 한다.
- STIC 전용 실험 파일을 새로 만들기보다 기존 `exp_long_term_forecasting.py`를 재사용하는 방향을 우선 검토한다.
- baseline 재현은 서버 `dkhan_tailscale`에서 수행한다.
- STIC gate는 per-horizon `sigmoid(MLP([pred_h, pred_c, pred_c - pred_h]))`로 시작한다.
- STIC 기본 학습 손실은 `pred`, `pred_h/pred_c` auxiliary, gate BCE supervision 조합으로 시작한다.
- ablation은 단일 `models/STIC.py` 내부 모드 스위치로 처리하고 별도 모델 파일은 만들지 않는다.

## Next Immediate Action
1. Exchange numeric section은 `g0 default vs g1b mechanism` 구도로 잠그고, S3/Hybrid S3를 negative simplification baselines로 정리한다.
2. ILI를 같은 fixed-setting 비교군(`g0`, `g1b`, `g1b-topclip`, `lite-0.75`)으로 확장할지 결정한다.
3. CiK electricity task용 thin adapter prototype 파일 뼈대를 다음 라운드에서 시작한다.
