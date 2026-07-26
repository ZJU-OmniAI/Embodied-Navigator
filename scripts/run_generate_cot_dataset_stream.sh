#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_generate_cot_dataset_stream.sh [cfg] [output_dir] [split] [extra_args...]
#
# Defaults:
#   cfg         - config/ht_dthink_base.yaml
#   output_dir  - runs/dthink_sft_data/data_buffer
#   split       - sample
#
# Behavior:
#   - Runs with 16 worker processes.
#   - Limits ready (non-underscore) episode dirs under output_root to 100 (streaming).

CFG="${1:-config/ht_dthink_base.yaml}"
OUTPUT_DIR="${2:-runs/dthink_sft_data/data_buffer}"
SPLIT="${3:-train}"
NPROC=${NPROC:-7}

echo "[generate_cot_dataset_stream] cfg: ${CFG}"
echo "[generate_cot_dataset_stream] output_dir: ${OUTPUT_DIR}"
echo "[generate_cot_dataset_stream] split: ${SPLIT}"
echo "[generate_cot_dataset_stream] nproc_per_node: ${NPROC}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7

torchrun --nproc_per_node "${NPROC}" -m src.dataset.utils.generate_cot_dataset \
  --cfg "${CFG}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite \
  --stream-max-ready 100

# torchrun --nproc_per_node "${NPROC}" -m src.dataset.utils.generate_cot_dataset \
#   --cfg "${CFG}" \
#   --split "${SPLIT}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --overwrite \
#   --save-png \
#   --record-sensor rgb_front \
#   --stream-max-ready 10

# torchrun --nproc_per_node "${NPROC}" -m src.dataset.utils.generate_cot_dataset \
#   --cfg "${CFG}" \
#   --split "${SPLIT}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --overwrite \
#   --save-png \
#   --record-sensor rgb_front \
#   --stream-max-ready 100

