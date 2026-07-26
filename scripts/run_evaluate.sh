#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_evaluate.sh [env_config] [checkpoint] [save_root] [extra_args...]
#
# Defaults:
#   env_config - config/ht_dthink_r2r.yaml
#   checkpoint - model/DThinkVLN-7B-SFT-S3/checkpoint-3000
#   save_root  - runs/dthink_eval

ENV_CFG="${1:-config/ht_dthink_r2r.yaml}"
CKPT="${2:-runs/DThinkVLN-P-7B-GRPO-r2xr-selected-6.0.0/checkpoint-50}"
SAVE_ROOT="${3:-runs/DThinkVLN-P-7B-GRPO-6.0.0-50-r2r-pass1}"
EXTRA_ARGS=("${@:4}")

# Force single GPU and single-threaded math libs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}

echo "[evaluate] env_config : ${ENV_CFG}"
echo "[evaluate] checkpoint : ${CKPT}"
echo "[evaluate] save_root  : ${SAVE_ROOT}"
echo "[evaluate] extra args : ${EXTRA_ARGS[*]-<none>}"

torchrun --nproc_per_node "7" --master_port 29555 -m src.eval.evaluate_multi \
  --env_config "${ENV_CFG}" \
  --checkpoint "${CKPT}" \
  --save_root "${SAVE_ROOT}" \
  --verbose \
  --retry_times "1"\
  --noise_var "0.0"\
  --temperature "0.4"\
  "${EXTRA_ARGS[@]}"



