#!/usr/bin/env bash
set -euo pipefail

INPUT="data/dthink_sft_data/processed_objnav_data_with_cots_sum_4_test.json"
OUTPUT="${INPUT%.json}_polish.json"

python src/dataset/utils/polish_cot.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --max-workers 24
