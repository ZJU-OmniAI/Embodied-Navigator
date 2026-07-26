#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_generate_cot_dataset_stream_pixel.sh [cfg] [output_dir] [split] [pixel_sensor] [min_retry] [max_retry] [extra_args...]
#
# Defaults:
#   cfg           - config/ht_dthink_base.yaml
#   output_dir    - runs/dthink_sft_data/pixel_buffer
#   split         - train
#   pixel_sensor  - auto (project on all recorded rgb_* views)
#   min_retry     - 3 (至少尝试次数)
#   max_retry     - 6 (最多尝试次数；若成功轨迹不足2条会持续到上限)
#
# Notes:
#   - Uses torchrun; set NPROC env to control worker count (default 4).
#   - Saves PNGs and records rgb_front video by default for inspection.

CFG="${1:-config/ht_dthink_base.yaml}"
OUTPUT_DIR="${2:-data/dthink_sft_data/val_unseen_buffer}"
SPLIT="${3:-val_unseen}"
PIXEL_SENSOR="${4:-auto}"
MIN_RETRY="${5:-${MIN_RETRY:-4}}"
MAX_RETRY="${6:-${MAX_RETRY:-32}}"
EXTRA_ARGS=("${@:7}")
NPROC=${NPROC:-4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

echo "[generate_cot_dataset_pixel] cfg: ${CFG}"
echo "[generate_cot_dataset_pixel] output_dir: ${OUTPUT_DIR}"
echo "[generate_cot_dataset_pixel] split: ${SPLIT}"
echo "[generate_cot_dataset_pixel] pixel_sensor: ${PIXEL_SENSOR}"
echo "[generate_cot_dataset_pixel] min_retry: ${MIN_RETRY}"
echo "[generate_cot_dataset_pixel] max_retry: ${MAX_RETRY}"
echo "[generate_cot_dataset_pixel] extra args: ${EXTRA_ARGS[*]-<none>}"
echo "[generate_cot_dataset_pixel] nproc_per_node: ${NPROC}"

rm -rf DThinkVLN/data/dthink_sft_data/train_vlnce_buffer/_*

torchrun --nproc_per_node "${NPROC}" -m src.dataset.utils.generate_cot_dataset_pixel \
  --cfg "${CFG}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --pixel-sensor "${PIXEL_SENSOR}" \
  --min-retry "${MIN_RETRY}" \
  --max-retry "${MAX_RETRY}" \
  --stream-max-ready 0 \
  "${EXTRA_ARGS[@]}"

# torchrun --nproc_per_node "${NPROC}" -m src.dataset.utils.generate_cot_dataset_pixel \
#   --cfg "${CFG}" \
#   --split "${SPLIT}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --pixel-sensor "${PIXEL_SENSOR}" \
#   --min-retry "${MIN_RETRY}" \
#   --max-retry "${MAX_RETRY}" \
#   --overwrite \
#   --save-png \
#   --record-sensor rgb_front \
#   --stream-max-ready 100 \
#   "${EXTRA_ARGS[@]}"
