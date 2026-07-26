#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_generate_cot_dataset_pixel.sh [cfg] [output_dir] [split] [pixel_sensor] [extra_args...]
#
# Defaults:
#   cfg           - config/ht_dthink_base.yaml
#   output_dir    - runs/dthink_sft_data/pixel_buffer
#   split         - sample
#   pixel_sensor  - auto (project on all recorded rgb_* views)
#
# Notes:
#   - Uses torchrun; set NPROC env to control worker count (default 1).
#   - Saves PNGs and records rgb_front video by default for inspection.

CFG="${1:-config/ht_dthink_base.yaml}"
OUTPUT_DIR="${2:-runs/dthink_sft_data/pixel_buffer}"
SPLIT="${3:-sample}"
PIXEL_SENSOR="${4:-auto}"
EXTRA_ARGS=("${@:5}")
NPROC=${NPROC:-1}

echo "[generate_cot_dataset_pixel] cfg: ${CFG}"
echo "[generate_cot_dataset_pixel] output_dir: ${OUTPUT_DIR}"
echo "[generate_cot_dataset_pixel] split: ${SPLIT}"
echo "[generate_cot_dataset_pixel] pixel_sensor: ${PIXEL_SENSOR}"
echo "[generate_cot_dataset_pixel] extra args: ${EXTRA_ARGS[*]-<none>}"
echo "[generate_cot_dataset_pixel] nproc_per_node: ${NPROC}"

torchrun --nproc_per_node "${NPROC}" -m src.dataset.utils.generate_cot_dataset_pixel \
  --cfg "${CFG}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --pixel-sensor "${PIXEL_SENSOR}" \
  --overwrite \
  --save-png \
  --record-sensor rgb_front \
  "${EXTRA_ARGS[@]}"

