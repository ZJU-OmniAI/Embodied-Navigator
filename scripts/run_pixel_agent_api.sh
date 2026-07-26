#!/usr/bin/env bash
set -euo pipefail

# Example:
#   ./scripts/run_pixel_agent_api.sh 0.0.0.0 8000

export CUDA_VISIBLE_DEVICES=4
export DTHINK_MODEL_PATH="runs/DThinkVLN-P-7B-GRPO-V2.0/checkpoint-300"
export DTHINK_MAX_NEW_TOKENS=512
export DTHINK_TEMPERATURE=0.5
export DTHINK_TOP_P=0.2

HOST="${1:-0.0.0.0}"
PORT="${2:-11451}"

python3 -m uvicorn src.server.pixel_agent_api:app --host "${HOST}" --port "${PORT}"

