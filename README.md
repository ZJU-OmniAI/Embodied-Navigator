<div align="center">

# TAMP-Nav

### Point, Think, Memorize, and Align for Efficient Embodied Navigation

Anonymous research artifact for double-blind review

[Architecture](#architecture) | [Results](#main-results) | [Analysis](#what-the-model-learns) | [Examples](#qualitative-examples)

</div>

## Overview

TAMP-Nav is a unified vision-language navigation framework that aligns high-level visual reasoning with low-level physical execution. Instead of asking a vision-language model (VLM) to regress 3D coordinates or emit long sequences of atomic actions, TAMP-Nav lets the model act as a visual pointer: it selects a camera view and a 2D pixel waypoint, which is projected into 3D and executed by a low-level SLAM controller.

The framework couples this vision-centric action space with selective reasoning, long-horizon memory, and hierarchical reinforcement learning. The resulting RGB-based navigator reasons at difficult decision points, preserves the topology of long trajectories, and learns from both immediate physical feedback and complete-task outcomes.

> **Core contribution:** TAMP-Nav unifies spatial alignment, adaptive reasoning, compact trajectory memory, and dense policy optimization in one embodied navigation framework.

## Architecture

<p align="center">
  <img src="docs/img/architecture.png" alt="TAMP-Nav architecture" width="100%">
</p>

<p align="center"><em>
TAMP-Nav combines multi-view RGB observations, language instructions, Anchor-Trajectory Memory, selective reasoning, visual waypoint prediction, and 2D-to-3D execution.
</em></p>

At each navigation step, the policy combines four egocentric RGB views with a compressed history. It decides whether the scene warrants explicit reasoning, predicts a view and pixel waypoint, and delegates metric execution to the controller. Depth is used only after the VLM prediction for geometric projection; it is not an input to the VLM.

| Component | Mechanism | Effect |
| --- | --- | --- |
| **Point** | For the first time, use pixel pointing as a bridge from VLM pre-training to VLN, then project the selected pixel into 3D | Smoothly transfers pretrained visual-grounding ability into VLN while leaving geometry and motion control to deterministic modules |
| **Think** | Trigger Chain-of-Thought only at critical topological nodes | Concentrates computation on crossroads, doorways, and target-relevant decisions |
| **Memorize** | Store critical states as visual-reasoning anchors and routine motion as Space-Time Indicators | Preserves long-horizon topology without retaining every visual frame |
| **Align** | Optimize the policy through reinforcement learning with Two-Level GRPO, combining local action and global trajectory advantages | Assigns credit from immediate decisions to final navigation success |

### Point: visual actions, metric execution

TAMP-Nav is the first to formulate pixel pointing as a natural bridge for smoothly transferring the VLM's visual-grounding ability acquired during pre-training to VLN. The VLM observes four views covering 360 degrees, chooses the most relevant view, and predicts a pixel coordinate. Projecting that pixel through the aligned depth map produces a local 3D waypoint for the SLAM controller. This separation lets the learned policy focus on visual-semantic grounding rather than learning metric geometry implicitly.

### Think and Memorize: adaptive cognition over long horizons

TAMP-Nav treats reasoning events as memory anchors. A critical node retains its visual evidence, spatial-temporal state, and reasoning summary. Between anchors, redundant images are discarded and the traversed path is represented by lightweight Space-Time Indicators encoding position, orientation, and time. The resulting alternating anchor-trajectory sequence preserves both semantic landmarks and geometric connectivity.

### Align: Two-Level GRPO

<p align="center">
  <img src="docs/img/two_level_grpo.png" alt="Two-Level GRPO with step-level and trajectory-level rollouts" width="100%">
</p>

<p align="center"><em>
Two-Level GRPO is a reinforcement-learning objective that superimposes local action advantages and global trajectory advantages across multi-branch rollouts.
</em></p>

At every decision point, the policy explores several candidate visual actions and receives local feedback. Complete rollouts receive global feedback for success, path efficiency, and reasoning density. Combining both levels reduces the credit-assignment gap between a final outcome and the intermediate decisions that produced it.

| Optimization level | Signals | What it teaches |
| --- | --- | --- |
| **Local step** | Target approach, collision avoidance, stop correctness, reasoning value, output validity | Which action and reasoning choice is useful at the current state |
| **Global trajectory** | Task success, SPL, reasoning density | Whether the complete plan is successful, efficient, and cognitively economical |
| **Combined advantage** | Global advantage + local advantage | How local decisions contribute to long-term navigation quality |

## Main Results

All values below are reported in the anonymous manuscript on validation-unseen splits. Higher is better for OS, SR, SPL, and nDTW; lower is better for NE.

### R2R-CE Val-Unseen

| Method | NE (lower) | OS | SR | SPL |
| --- | ---: | ---: | ---: | ---: |
| StreamVLN | 4.98 | 64.2 | 56.9 | 51.9 |
| NavFoM | 4.61 | 72.1 | 61.7 | 55.3 |
| DualVLN | 4.05 | 70.7 | 64.3 | 58.5 |
| TAMP-Nav (SFT only) | 4.88 | 62.0 | 55.7 | 50.3 |
| **TAMP-Nav** | **3.85** | **74.5** | **66.2** | **58.8** |

### RxR-CE Val-Unseen

| Method | NE (lower) | SR | SPL | nDTW |
| --- | ---: | ---: | ---: | ---: |
| StreamVLN | 6.22 | 52.9 | 46.0 | 61.9 |
| NavFoM | 4.74 | 64.4 | 56.2 | 65.8 |
| DualVLN | 4.58 | 61.4 | 51.8 | 70.0 |
| TAMP-Nav (SFT only) | 6.10 | 52.4 | 46.2 | 62.1 |
| **TAMP-Nav** | **4.32** | **65.7** | **56.9** | **72.4** |

The full framework improves R2R-CE success from **55.7% to 66.2%** over its SFT-only initialization, a gain of 10.5 percentage points from reinforcement-learning alignment.

## What The Model Learns

### Reasoning on demand

<p align="center">
  <img src="docs/img/reasoning_heatmap.png" alt="Spatial density of reasoning triggers before and after reinforcement learning" width="100%">
</p>

<p align="center"><em>
Spatial density of reasoning triggers: SFT initialization (left) and RL-aligned TAMP-Nav (right).
</em></p>

| Trigger strategy | CoT ratio | R2R-CE SR |
| --- | ---: | ---: |
| Dense CoT | 100.0% | 66.8% |
| Fixed interval (1/3) | 36.2% | 60.1% |
| **Adaptive trigger (TAMP-Nav)** | **26.3%** | **66.2%** |

Adaptive triggering nearly matches dense reasoning while invoking CoT on roughly one quarter of the steps. Spatial analysis further shows that reasoning assigned to straight corridors falls from **38% after SFT to 11% after RL alignment**, while triggers concentrate around intersections, doorways, and target-relevant regions.

### Long-horizon memory

The long-horizon subset contains 5,927 trajectories whose expert paths exceed 12.5 meters.

| Memory or navigation variant | SR |
| --- | ---: |
| StreamVLN | 30.9 |
| DualVLN | 41.9 |
| TAMP-Nav with uniform sampling | 40.5 |
| TAMP-Nav with full history | 42.4 |
| TAMP-Nav without Space-Time Indicators | 45.6 |
| **TAMP-Nav with Anchor-Trajectory Memory** | **49.8** |

Explicit anchors preserve high-value semantic evidence, while Space-Time Indicators retain the geometry of compressed path segments. Removing the indicators reduces long-horizon SR by 4.2 percentage points.

### Efficiency, robustness, and transfer

| Evaluation | TAMP-Nav | Comparison or reference |
| --- | ---: | --- |
| Training scale | **90k trajectories / 700k interactions** | SFT cold start followed by Two-Level GRPO |
| Average policy interactions per trajectory | **9** | Approximately 30 for StreamVLN and DualVLN |
| Average inference time per task on one A800 | **16.58 s** | 37.47 s for StreamVLN; 41.46 s for DualVLN |
| SR with 0.2 multiplicative depth noise | **63.4%** | 66.2% without noise |
| Zero-shot real-world SR over 100 trials | **60.0%** | 49.0% for StreamVLN; 53.0% for DualVLN |

The Two-Level GRPO analysis reports a final success reward of **0.59** with trajectory rewards alone, **0.64** after adding local step advantages, and **0.68** with the full annealed guided-sampling strategy.

## Qualitative Examples

### Simulation

<p align="center">
  <img src="docs/img/simulation_trajectory.png" alt="TAMP-Nav simulation trajectory with selected visual waypoints and sparse reasoning" width="92%">
</p>

<p align="center"><em>
A simulated trajectory illustrating multi-view pixel actions and sparse reasoning at task-relevant decision points.
</em></p>

### Real world

<p align="center">
  <img src="docs/img/real_world_trajectory.png" alt="TAMP-Nav real-world navigation trajectory" width="68%">
  <img src="docs/img/real_world_success.png" alt="Real-world success-rate comparison" width="29%">
</p>

<p align="center"><em>
Zero-shot navigation in an unmapped real-world environment and success rates over 100 trials.
</em></p>

### Real-world deployment videos

Six representative zero-shot trials on the Unitree Go2 quadruped, drawn from the
100-episode real-world evaluation. Each video shows the synchronized multi-view RGB
observations, the selected pixel waypoints with reasoning traces, and the executed
trajectory. Success examples are listed first; the failure case is listed last.
The same videos play inline on the [project homepage](docs/index.html).

| # | Video | Scene | Notes |
| --- | --- | --- | --- |
| 1 | [cross-scenario.mp4](docs/img/cross-scenario.mp4) | Cross-scenario | Long-horizon episode with an indoor-to-outdoor transition |
| 2 | [hall.mp4](docs/img/hall.mp4) | Indoor hall | Long-corridor navigation with sparse reasoning |
| 3 | [meeting-room.mp4](docs/img/meeting-room.mp4) | Meeting room | Meeting-room navigation |
| 4 | [outdoors.mp4](docs/img/outdoors.mp4) | Outdoors | Outdoor scene navigation |
| 5 | [playground.mp4](docs/img/playground.mp4) | Laboratory test area | Indoor laboratory test-area navigation |
| 6 | [outdoors-failed.mp4](docs/img/outdoors-failed.mp4) | Outdoors (**failure case**) | The policy selects an occluded waypoint, oscillates between two waypoints, and requires human intervention |

## Repository Structure

| Path | Research role |
| --- | --- |
| `src/agent/` | Navigation policy, selective reasoning, action parsing, and memory |
| `src/model/` | Navigation-adapted Qwen2.5-VL implementation |
| `src/train/` | SFT, Two-Level GRPO, and reward definitions |
| `src/env/` | Continuous navigation environment and geometry utilities |
| `src/eval/` | Evaluation harness and metrics (NE, OS, SR, SPL, nDTW) |
| `src/dataset/` | MultiNav-CoT processing pipeline and dataset loaders |
| `src/server/` | Real-robot FastAPI service and ROS2 client |
| `config/` | Experiment configurations (SFT, GRPO, eval) |
| `scripts/` | Training and evaluation shell scripts |
| `data/` | Anonymized review subset (full dataset post-acceptance) |
| `docs/` | Project homepage (figures and deployment videos) |
| `paper/` | LaTeX manuscript source (anonymized) |

## Getting Started

### Prerequisites

```bash
# Core dependencies
Python 3.10+
PyTorch 2.0+
Habitat-Lab 0.3.0
Transformers 4.40+

# See requirements.txt for complete list
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TAMP-Nav

# Install dependencies
pip install -r requirements.txt

# Download anonymized data subset
# (90k MultiNav-CoT subset provided for review)
```

### Quick Start

```bash
# Supervised fine-tuning
bash scripts/sft_train.sh config/sft_dthink_7b.yaml

# Two-Level GRPO alignment
bash scripts/grpo_train.sh config/grpo_dthink_7b.yaml

# Evaluation
bash scripts/evaluate.sh config/eval_r2r_ce.yaml
```

See `CLAUDE.md` for detailed setup instructions and configuration options.

## Key Components

### Training Pipeline

1. **Supervised Fine-Tuning (SFT)**
   - Base model: Qwen2.5-VL-7B
   - Dataset: MultiNav-CoT (90k trajectories with Gemini 2.5 Flash–generated CoT)
   - Output: Policy initialization for RL

2. **Two-Level GRPO Alignment**
   - Local step rewards: target approach, collision avoidance, stop correctness, reasoning value
   - Global trajectory rewards: success, SPL, reasoning density
   - Annealed guided sampling with β schedule

3. **Evaluation Harness**
   - Benchmarks: R2R-CE, RxR-CE val-unseen splits
   - Metrics: NE, OS, SR, SPL, nDTW
   - Multi-GPU support via torchrun

### Real-Robot Deployment

The `src/server/` directory contains interfaces for physical robot deployment:

- FastAPI service for `/reset` and `/step` endpoints
- ROS2 client for Unitree Go2 quadruped
- Integration with D435i depth camera and Hesai LiDAR
- FastLIO odometry and FAR Planner obstacle avoidance

Zero-shot real-world deployment requires no robot-specific fine-tuning.

## Project Homepage

Visit `docs/index.html` for the full project page with:
- Interactive figure viewer
- Detailed method description
- Complete results tables
- Qualitative examples and deployment videos

To preview locally:
```bash
cd docs
python3 -m http.server 8000
# Open http://localhost:8000
```

Page conventions for maintainers are documented in `docs/README.md`.

## Citation

Citation metadata is anonymized during review and will be replaced on acceptance.

```bibtex
@inproceedings{anonymous2026tampnav,
  title     = {TAMP-Nav: Point, Think, Memorize, and Align
               for Efficient Embodied Navigation},
  author    = {Anonymous Authors},
  booktitle = {Under double-blind review},
  year      = {2026},
  note      = {Author and venue metadata withheld during review}
}
```

## Artifact Scope

This repository is an anonymous review artifact. Author identities, affiliations, acknowledgements, personal project links, and citation metadata are intentionally withheld.

**Included in this artifact:**
- Navigation policy implementation (`src/agent/`)
- Navigation-adapted VLM (`src/model/`)
- SFT and Two-Level GRPO training code (`src/train/`)
- Continuous-navigation environment and evaluation (`src/env/`, `src/eval/`)
- MultiNav-CoT processing pipeline with anonymized review subset (`src/dataset/`, `data/`)
- Real-robot serving interfaces (`src/server/`)
- Experiment configurations and scripts (`config/`, `scripts/`)

**Distributed separately (post-acceptance):**
- Full licensed Habitat-Matterport3D simulation assets
- Complete 90k-trajectory MultiNav-CoT dataset
- Pre-trained model checkpoints
- Complete recordings of all 100 real-world trials (six representative videos are included in `docs/img/`)
- Full manuscript and LaTeX source

**Key dependencies:**
- The method is trained on MultiNav-CoT, which uses Gemini 2.5 Flash–generated Chain-of-Thought annotations
- The Two-Level GRPO rewards use simulator signals (geodesic distance, oracle success, SPL) that require adaptation for real-world training
- The full system uses depth for pixel-to-3D projection and odometry for Space-Time Indicator encoding, though the VLM observes RGB only

## License

License information will be provided upon acceptance. This artifact is provided for review purposes only.
