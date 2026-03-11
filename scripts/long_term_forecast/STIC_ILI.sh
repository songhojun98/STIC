#!/usr/bin/env bash

set -euo pipefail

MODEL_KIND="${1:-STIC}"
GATE_VARIANT="${2:-g0}"
SEED="${3:-2021}"
GPU="${4:-0}"
EPOCHS="${5:-1}"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

COMMON_ARGS=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/illness/
  --data_path national_illness.csv
  --data custom
  --features MS
  --target OT
  --seq_len 36
  --label_len 18
  --pred_len 24
  --enc_in 7
  --dec_in 7
  --c_out 7
  --batch_size 32
  --learning_rate 1e-4
  --train_epochs "${EPOCHS}"
  --itr 1
  --num_workers 2
  --seed "${SEED}"
)

if [[ "${MODEL_KIND}" == "DLinear" ]]; then
  python -u run.py \
    "${COMMON_ARGS[@]}" \
    --model DLinear \
    --model_id "dlinear_ili_base_s${SEED}" \
    --des ili_baseline
  exit 0
fi

GATE_MODE="${GATE_VARIANT}"
EXTRA_ARGS=()

case "${GATE_VARIANT}" in
  g0|g1b)
    ;;
  g1b-topclip)
    EXTRA_ARGS+=(--stic_gate_hidden_scale 0.5)
    ;;
  lite-0.75)
    GATE_MODE="g1b-topclip-lite"
    EXTRA_ARGS+=(--stic_gate_hidden_scale 0.75)
    ;;
  *)
    echo "Unsupported gate variant: ${GATE_VARIANT}" >&2
    exit 1
    ;;
esac

python -u run.py \
  "${COMMON_ARGS[@]}" \
  --model STIC \
  --model_id "stic_ili_${GATE_VARIANT}_s${SEED}" \
  --des ili_screen \
  --stic_mode dynamic \
  --stic_gate_target_mode soft \
  --stic_gate_soft_tau 0.02 \
  --stic_gate_input_mode "${GATE_MODE}" \
  --stic_gate_hidden_feat_dim 8 \
  --stic_context_mixer_type linear \
  --stic_context_residual_scale 0.5 \
  --stic_context_corruption_mode mixed \
  --stic_context_corruption_prob 0.5 \
  --stic_context_dropout_p 0.3 \
  --stic_context_corruption_gate_weight 1.0 \
  --stic_corrupt_context_aux_weight 0.0 \
  --stic_pair_rank_weight 0.0 \
  "${EXTRA_ARGS[@]}"
