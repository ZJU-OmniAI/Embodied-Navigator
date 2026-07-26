#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  -m src.train.train_grpo_countdown