import os
import random
from pathlib import Path
from typing import Any, Dict, List
import torch
import wandb
from torch.utils.data import Dataset
from trl import ModelConfig, TrlParser, get_peft_config

from ..model.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from ..utils import env_registry
from .trainer import GRPOConfig, GRPOTrainer
from .grpo_reward import (
    set_reward_log_dir,
    success_reward,
    trajectory_similarity_reward,
    reasoning_density_reward,
    format_step_reward,
    collision_reward,
    target_approach_reward,
    reasoning_reward,
    stop_reward,
    weighted_target_approach_step_select,
    random_step_select
)

def print_trainable_params(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

class EpisodeIdDataset(Dataset):
    def __init__(self, episode_ids: List[str]) -> None:
        self.episode_ids = [str(ep_id) for ep_id in episode_ids]

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"episode_id": self.episode_ids[idx]}


def _extract_episode_ids(env: Any) -> List[str]:
    if hasattr(env, "get_episode_list"):
        episode_ids = env.get_episode_list()
    elif hasattr(env, "env") and hasattr(env.env, "_dataset") and hasattr(env.env._dataset, "episodes"):
        episode_ids = [getattr(ep, "episode_id", None) for ep in env.env._dataset.episodes]
    else:
        raise RuntimeError("Failed to fetch episode ids from env.")

    episode_ids = [str(ep_id) for ep_id in episode_ids if ep_id is not None]
    if not episode_ids:
        raise ValueError("No episodes found in env dataset.")
    return episode_ids


def build_episode_dataset_from_env(
    env_type: str,
    env_config: str,
    gpu_device_id: int = 0,
) -> EpisodeIdDataset:
    from ..env import DThinkEnv  # noqa: F401, ensures env registration side effects

    probe_env = env_registry.create(env_type, env_config, gpu_device_id=gpu_device_id)
    try:
        episode_ids = _extract_episode_ids(probe_env)
    finally:
        close_fn = getattr(probe_env, "close", None)
        if callable(close_fn):
            close_fn()
    return EpisodeIdDataset(episode_ids)


def main():
    parser = TrlParser((GRPOConfig, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if "REWARD_DEBUG_LOG_DIR" in os.environ:
        set_reward_log_dir(Path(os.environ["REWARD_DEBUG_LOG_DIR"]))
    else:
        set_reward_log_dir(Path(training_args.output_dir) / "reward_logs")

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    rank = int(os.environ.get("RANK", "0"))
    seed = int(training_args.seed) + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    processed_dataset = build_episode_dataset_from_env(
        training_args.env_type,
        training_args.env_config,
        gpu_device_id=int(os.environ.get("LOCAL_RANK", "0")),
    )
    print(f"Built episode-id dataset from env: {len(processed_dataset)} episodes")

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    print_trainable_params(model)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        processing_class=processor,
        reward_funcs=[
            success_reward,
            trajectory_similarity_reward,
            reasoning_density_reward,
        ],
        step_reward_funcs=[
            collision_reward,
            target_approach_reward,
            reasoning_reward,
            stop_reward,
            format_step_reward,
        ],
        step_select_func=weighted_target_approach_step_select,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()
