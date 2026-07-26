<div align="center">

# TAMP-Nav

**Point, Think, Memorize, and Align for Efficient Embodied Navigation**

Anonymous research artifact for double-blind review

[Sample data](data/) · [Configurations](config/) · [Training and evaluation scripts](scripts/)

</div>

> [!IMPORTANT]
> This repository is intentionally anonymous. Author names, affiliations, acknowledgements, personal project links, and a project citation are omitted during double-blind review. The manuscript and paper source are distributed separately and intentionally excluded from this repository.

## Overview

TAMP-Nav is a vision-language navigation framework designed to improve spatial alignment, reasoning efficiency, and long-horizon memory. A Qwen2.5-VL-7B-based policy observes four RGB views and predicts a view together with a 2D pixel waypoint. The corresponding depth observation projects that waypoint into 3D, where a low-level Habitat/SLAM controller executes the motion.

The framework contains four complementary components:

- **Point — Pixel-to-3D actions.** The VLM operates in its native 2D visual space while geometric projection and low-level control handle physical execution.
- **Think — Selective reasoning.** Chain-of-thought is triggered at difficult decision points instead of at every navigation step.
- **Memorize — Anchor-Trajectory Memory.** Critical observations are retained as visual anchors, while routine trajectory segments are compressed into lightweight Space-Time Indicator (STI) tokens.
- **Align — Two-Level GRPO.** Step-level process rewards and trajectory-level outcome rewards jointly align reasoning, action selection, and navigation success.

The anonymous manuscript reports the following results on validation-unseen splits:

| Benchmark | NE ↓ | OS ↑ | SR ↑ | SPL ↑ | nDTW ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2R-CE | 3.85 | 74.5 | 66.2 | 58.8 | — |
| RxR-CE | 4.32 | — | 65.7 | 56.9 | 72.4 |

These values are transcribed from the paper; they were not recomputed from this reduced review artifact. The full evaluation datasets and trained checkpoints are not included.

## Artifact contents

| Path | Description |
| --- | --- |
| `src/agent/` | Pixel-action agents, prompt construction, action parsing, and navigation memory |
| `src/env/` | Habitat environment wrapper, VLN task extensions, metrics, and pixel-to-3D geometry |
| `src/model/` | Customized Qwen2.5-VL model and tokenizer implementation; model weights are not included |
| `src/train/` | Supervised fine-tuning and Two-Level GRPO training code |
| `src/eval/` | Single- and multi-process evaluation with per-episode artifacts and aggregate metrics |
| `src/dataset/` | Dataset readers, trajectory generation, CoT processing, and reward-log visualization |
| `src/server/` | FastAPI policy service and an optional ROS 2 client for robot deployment |
| `config/` | SFT, GRPO, DeepSpeed, R2R-CE, and RxR-CE configurations |
| `scripts/` | Reference launchers for data preparation, training, evaluation, and serving |
| `data/` | A 1,000-episode review subset with Habitat-style metadata and CoT/waypoint annotations |

## Environment

Run all commands from the repository root. A pinned environment file is not part of this anonymous artifact, so install mutually compatible versions of PyTorch, CUDA, Habitat-Sim, and Habitat-Lab for the target machine.

1. Create an isolated environment:

   ```bash
   conda create -n tamp-nav python=3.10 -y
   conda activate tamp-nav
   ```

2. Install a CUDA-compatible build of [PyTorch](https://pytorch.org/get-started/locally/) and `torchvision`.

3. Install matched versions of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab). The configuration files use Habitat's Hydra-based configuration API.

4. Install the remaining core dependencies. Transformers `4.57.3` is the version recorded in the bundled model configuration.

   ```bash
   python -m pip install \
     "transformers==4.57.3" trl accelerate datasets deepspeed peft \
     liger-kernel wandb numpy numpy-quaternion scipy pillow imageio \
     opencv-python tqdm pyyaml requests packaging regex

   python -m pip install flash-attn --no-build-isolation
   ```

5. Install optional dependencies only for the corresponding utilities:

   ```bash
   # CoT post-processing and the HTTP policy service
   python -m pip install openai fastapi "uvicorn[standard]"
   ```

   ROS 2 deployment additionally requires `rclpy`, `sensor_msgs`, `std_msgs`, and `tf2_ros` from a compatible ROS 2 installation. The optional depth service requires Depth Anything 3 and its model weights.

The reference experiments used 8 NVIDIA A800 GPUs with 80 GB of memory per GPU. The paper reports approximately 160 GPU-hours for SFT and 600 GPU-hours for GRPO. Other hardware configurations require corresponding changes to GPU counts, batch sizes, gradient accumulation, and DeepSpeed settings.

## Data preparation

### Included review subset

The repository includes two aligned 1,000-episode JSON files:

- `data/train_92_split.json`: Habitat-style episodes, instructions, goals, poses, and reference trajectories.
- `data/train_92_cot_split.json`: selective CoT, pixel projections, sensor poses, and compressed-trajectory metadata.

Quick integrity check:

```bash
jq '.episodes | length' data/train_92_split.json
jq 'length' data/train_92_cot_split.json
```

Both commands should print `1000`. Observation images, Matterport3D scenes, the complete MultiNav-CoT dataset, and the full GQA-derived auxiliary set are not redistributed in this review package. As noted in [`data/README.md`](data/README.md), the complete dataset is planned for release after paper acceptance.

### Habitat assets

Obtain Matterport3D scenes and R2R-CE/RxR-CE episodes under their original terms. The [official VLN-CE repository](https://github.com/jacobkrantz/VLN-CE) documents both datasets and their expected scene layout. The checked-in configurations expect paths equivalent to:

```text
data/share/habitat/
├── scene_datasets/
└── datasets/
    ├── r2r/{split}/{split}.json.gz
    └── rxr/{split}/{split}_guide_en.json.gz
```

Update `scenes_dir` and `data_path` in the selected `config/ht_dthink_*.yaml` file if assets are stored elsewhere.

### Optional waypoint-data generation

To generate pixel-waypoint episodes from a prepared Habitat split:

```bash
NPROC=1 bash scripts/run_generate_cot_dataset_pixel.sh \
  config/ht_dthink_base.yaml \
  runs/cot_pixel_sample \
  sample \
  auto
```

The wrapper enables overwrite mode. Use a new output directory, or inspect the script before pointing it at existing data.

## Configure a run

The checked-in YAML files preserve run-specific local paths. Before launching a job, review and replace at least the following fields:

- `model_name_or_path` and `output_dir` in the SFT/GRPO configuration;
- dataset paths in `config/sft_dthink_7b_pixel.yaml`;
- `scenes_dir`, `data_path`, and `split` in the selected Habitat configuration;
- GPU IDs, process counts, and ports in the shell launcher;
- Weights & Biases settings, or disable `report_to` for offline runs.

The upstream initialization model is [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). The local model implementation adds navigation-specific position encoding and action behavior; use a compatible prepared checkpoint when a configuration expects those additions.

## Training

### Stage 1: supervised fine-tuning

After configuring the SFT initialization checkpoint and dataset paths:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 \
  -m src.train.sft_train \
  --config config/sft_dthink_7b_pixel.yaml \
  --model_name_or_path /path/to/initial-checkpoint \
  --output_dir runs/tamp_nav_sft
```

The paper's SFT setting uses bf16, one epoch, a learning rate of `5e-6`, a per-device batch size of `1`, gradient accumulation of `32`, a 4,096-token context, gradient checkpointing, and Flash Attention 2.

### Stage 2: Two-Level GRPO

Point the GRPO configuration to the SFT checkpoint and a prepared Habitat training split, then run:

```bash
bash scripts/grpo_dthink_7b.sh \
  config/grpo_dthink_7b.yaml \
  --model_name_or_path /path/to/sft-checkpoint \
  --output_dir runs/tamp_nav_grpo
```

The reference launcher uses eight processes. The supplied GRPO configuration records a trajectory group size of `8`, four step-level candidate actions, temperature `0.7`, top-p `0.9`, learning rate `2e-6`, 4,096 prompt tokens, 512 completion tokens, and no KL penalty.

## Evaluation

Use a single process for a smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
  -m src.eval.evaluate_multi \
  --env_config config/ht_dthink_r2r.yaml \
  --checkpoint /path/to/checkpoint \
  --save_root runs/eval_r2r \
  --max_episodes 10 \
  --temperature 0.4 \
  --top_p 0.6
```

Remove `--max_episodes 10` for the complete split. Use `config/ht_dthink_rxr.yaml` for RxR-CE. Each episode directory contains `actions.jsonl` and `result.json`; the evaluation root contains aggregate metrics in `result.json`.

[`scripts/run_evaluate.sh`](scripts/run_evaluate.sh) is the multi-GPU reference launcher. It is configured for seven workers and contains local default checkpoint/output paths, so pass explicit arguments or adapt it to the available hardware:

```bash
bash scripts/run_evaluate.sh \
  config/ht_dthink_r2r.yaml \
  /path/to/checkpoint \
  runs/eval_r2r
```

## Reward dashboard

GRPO writes per-rank JSONL logs under the run's `reward_logs` directory. Inspect them with the dependency-free local dashboard backend:

```bash
python src/dataset/visual/app.py \
  --host 127.0.0.1 \
  --port 18091 \
  --log-dir runs/tamp_nav_grpo/reward_logs
```

Open `http://127.0.0.1:18091` in a browser. See [`src/dataset/visual/README.md`](src/dataset/visual/README.md) for the log schema and chart extension points.

## HTTP policy service

The FastAPI service exposes `/health`, `/reset`, `/step`, `/cam_info`, and optional depth endpoints. To serve a trained pixel-action policy without loading Depth Anything 3:

```bash
DTHINK_MODEL_PATH=/path/to/checkpoint \
DTHINK_PRELOAD_DA3=0 \
CUDA_VISIBLE_DEVICES=0 \
python -m uvicorn src.server.pixel_agent_api:app \
  --host 0.0.0.0 \
  --port 11451
```

Check `http://127.0.0.1:11451/health` and use `http://127.0.0.1:11451/docs` for the generated request/response schema. Do not expose this unauthenticated research service directly to an untrusted network.

## Reproducibility boundaries

- Reported benchmark values require the full licensed Habitat assets, the complete training data, and trained checkpoints; these are not all present in the reduced review artifact.
- Configuration files and shell scripts contain run-specific local paths and fixed GPU allocations. Review them before execution.
- No exact dependency lockfile or container image is included. Record the final package versions and exact YAML used for every reproduced result.
- Data-generation wrappers may use `--overwrite`; choose fresh output paths unless replacement is intentional.

## Citation and license

Citation metadata is withheld to preserve double-blind anonymity. No project-specific software or data license is included in this review artifact; treat the material as review-only unless and until an explicit license accompanies the public release. Upstream models, datasets, simulators, and third-party code remain subject to their respective licenses and terms of use.
