#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_generate_cot_dataset_pixel_from_eval.sh [input_dir] [output_dir] [min_success] [max_episodes] [extra_args...]
#
# Defaults:
#   input_dir   - runs/DThinkVLN-P-7B-SFT-S1-V114_1-600-EVAL
#   output_dir  - runs/dthink_sft_data/data_buffer_sample
#   min_success - 0.5
#   max_episodes- unset (no cap)
#
# Notes:
#   - Converts eval rollouts (actions.jsonl/result.json) into pixel episodes
#     compatible with DThinkEpisodeDatasetPixelStream.
#   - PNGs are always saved; pass --save-npz via extra_args if faster loading is needed.
#   - Use --overwrite in extra_args to refresh existing episode_* folders.

INPUT_DIR1="${1:-DThinkVLN/data/dthink_sft_data/real}"
INPUT_DIR2="${1:-runs/DThinkVLN-P-7B-SFT-S1-V114_1-3600-TRAIN-pass6}"
OUTPUT_DIR="${2:-runs/dthink_sft_data/data_buffer}"
MIN_SUCCESS="${3:-0.5}"
MAX_EPISODES="${4:-}"
EXTRA_ARGS=("${@:5}")

rm -rf DThinkVLN/data/dthink_sft_data/data_buffer/*

echo "[cot_from_eval] input_dir   : ${INPUT_DIR1}"
echo "[cot_from_eval] input_dir   : ${INPUT_DIR2}"
echo "[cot_from_eval] output_dir  : ${OUTPUT_DIR}"
echo "[cot_from_eval] min_success : ${MIN_SUCCESS}"
echo "[cot_from_eval] max_episodes: ${MAX_EPISODES:-<none>}"
echo "[cot_from_eval] extra args  : ${EXTRA_ARGS[*]-<none>}"

python -m src.dataset.utils.generate_cot_dataset_pixel_from_eval \
    --input-dir "${INPUT_DIR1}" \
    --output-dir "${OUTPUT_DIR}" \
    --min-success "${MIN_SUCCESS}" \
    --stream-max-ready 200

# python -m src.dataset.utils.generate_cot_dataset_pixel_from_eval \
#     --input-dir "${INPUT_DIR2}" \
#     --output-dir "${OUTPUT_DIR}" \
#     --min-success "${MIN_SUCCESS}" \
#     --stream-max-ready 100 