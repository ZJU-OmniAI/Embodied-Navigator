#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=8
export TORCH_INTRAOP_THREADS=8
export TORCH_INTEROP_THREADS=8
export TOKENIZERS_PARALLELISM=false
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export WANDB_PROJECT=DThinkVLN

CONFIG_PATH="${1:-config/grpo_dthink_7b.yaml}"
MASTER_PORT="${MASTER_PORT:-13314}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    -m src.train.grpo_train \
    --config "${CONFIG_PATH}" \
    "${@:2}" \
    2> >(grep -vF 'PluginManager::Manager: duplicate static plugin' >&2)