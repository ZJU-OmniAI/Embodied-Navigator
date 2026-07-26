#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_format2habitat.sh [input_json] [output_json] [output_gz]
#
# Defaults:
#   input_json  - data/dthink_sft_data/sample_polished.json
#   output_json - <input basename>_habitat.json
#   output_gz   - <output_json>.gz

INPUT_JSON="${1:-data/dthink_sft_data/processed_vlnce_unseen_data_long_fix.json}"
OUTPUT_JSON="${2:-${INPUT_JSON%.json}_habitat.json}"
OUTPUT_GZ="${3:-${OUTPUT_JSON}.gz}"

echo "[format2habitat] input: ${INPUT_JSON}"
echo "[format2habitat] output json: ${OUTPUT_JSON}"
echo "[format2habitat] output gz: ${OUTPUT_GZ}"

python src/dataset/utils/format2habitat.py \
  --input "${INPUT_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --output_gz "${OUTPUT_GZ}"
