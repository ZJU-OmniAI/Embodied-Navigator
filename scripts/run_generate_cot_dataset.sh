#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_generate_cot_dataset.sh [cfg] [output_dir] [split] [extra_args...]
#
# Defaults:
#   cfg         - config/ht_dthink_base.yaml
#   output_dir  - runs/dthink_sft_data/data_buffer
#   split       - sample

CFG="${1:-config/ht_dthink_base.yaml}"
OUTPUT_DIR="${2:-runs/dthink_sft_data/data_buffer}"
SPLIT="${3:-sample}"
EXTRA_ARGS=("${@:4}")
NPROC=${NPROC:-1}

echo "[generate_cot_dataset] cfg: ${CFG}"
echo "[generate_cot_dataset] output_dir: ${OUTPUT_DIR}"
echo "[generate_cot_dataset] split: ${SPLIT}"
echo "[generate_cot_dataset] extra args: ${EXTRA_ARGS[*]-<none>}"
echo "[generate_cot_dataset] nproc_per_node: ${NPROC}"

torchrun --nproc_per_node "${NPROC}" -m src.dataset.utils.generate_cot_dataset \
  --cfg "${CFG}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  --overwrite \
  --save-png \
  --record-sensor rgb_front
