import liger_kernel.transformers
from .. import dataset as pointvln_dataset
import numpy as np
import torch
import requests
import random
import json
import os
from ..model.qwen2_5_vl import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLForConditionalGeneration,
    apply_liger_kernel_to_my_qwen2_5_vl
)
import liger_kernel
liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl = apply_liger_kernel_to_my_qwen2_5_vl
from typing import List, Dict, Any
import wandb
from datasets import Dataset, DatasetDict
from collections.abc import Mapping, Sequence, Set
from ..utils.qwen_vl_utils import process_vision_info
from accelerate import Accelerator
import datasets
from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from .trainer.sft_trainer import SFTTrainer
from .trainer.sft_config import SFTConfig

def print_trainable_params(model):
    trainable, total = 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print("TRAIN:", name)
    print(f"\nTrainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser(
        (SFTConfig, ModelConfig, pointvln_dataset.DatasetArguments))
    training_args, model_config, dataset_args = parser.parse_args_and_config()

    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    processed_dataset = pointvln_dataset.load_and_merge_datasets(
        dataset_args.datasets, mode=dataset_args.dataset_merge_type)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map()
    )

    if training_args.use_liger_kernel:
        apply_liger_kernel_to_my_qwen2_5_vl()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path, **model_kwargs)
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # for p in model.model.visual.parameters():
    #     p.requires_grad = False
    
    # for p in model.parameters():
    #     p.requires_grad = False

    # for p in model.model.language_model.embed_tokens.parameters():
    #     p.requires_grad = True

    # for p in model.model.pos_encoder.parameters():
    #     p.requires_grad = True

    # for p in model.model.actnav_head.parameters():
    #     p.requires_grad = True

    print_trainable_params(model)

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="PointVLN")

    # Initialize trainer
    cache_use_liger_kernel, training_args.use_liger_kernel = training_args.use_liger_kernel, False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=pointvln_dataset.create_qwen_collate_fn(processor),
        peft_config=get_peft_config(model_config),
    )
    trainer.args.use_liger_kernel, training_args.use_liger_kernel = cache_use_liger_kernel, cache_use_liger_kernel
    if training_args.use_liger_kernel:
        apply_liger_kernel_to_my_qwen2_5_vl(model=model)
    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
