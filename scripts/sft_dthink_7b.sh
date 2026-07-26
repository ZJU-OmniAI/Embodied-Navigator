export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

export TORCH_NUM_THREADS=8
export TORCH_INTRAOP_THREADS=8
export TORCH_INTEROP_THREADS=8

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="13313" \
    -m src.train.sft_train \
    --config config/sft_dthink_7b_pixel.yaml \
