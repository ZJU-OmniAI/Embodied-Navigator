
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Union
import torch
from transformers import Qwen2VLProcessor
from ..utils.qwen_vl_utils import (
    process_act_info,
    process_pos_info,
    process_vision_info,
)
import time


def drop_none(obj):
    if isinstance(obj, Mapping):
        return obj.__class__(
            (k, drop_none(v))
            for k, v in obj.items()
            if v is not None
        )
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        cleaned = [drop_none(item) for item in obj if item is not None]
        return obj.__class__(cleaned) if isinstance(obj, tuple) else cleaned
    elif isinstance(obj, Set) and not isinstance(obj, (str, bytes, bytearray)):
        return obj.__class__(drop_none(item) for item in obj if item is not None)
    return obj


def _normalize_media_batch(batch_media):
    if batch_media is None:
        return None
    norm = []
    for m in batch_media:
        if m is None:
            norm.append([])
        elif isinstance(m, (list, tuple)):
            norm.append(list(m))
        else:
            norm.append([m])
    return norm


def create_qwen_collate_fn(processor: Qwen2VLProcessor, last_trun=True):
    im_start_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')

    assistant_prompt_chunk = "assistant\n"
    assistant_prompt_ids = processor.tokenizer(
        assistant_prompt_chunk, add_special_tokens=False
    ).input_ids
    assistant_prompt_len = len(assistant_prompt_ids)

    def tok(s: str) -> int:
        tid = processor.tokenizer.convert_tokens_to_ids(s)
        return -1 if tid is None else tid

    ignore_token_ids = {
        tok("<image>"),
        tok("<video>"),
        tok("<|im_start|>"),
        tok("<|endofchunk|>"),
    }
    ignore_token_ids = {t for t in ignore_token_ids if t >= 0}

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        examples = [drop_none(x['messages']) for x in examples]
        texts = [processor.apply_chat_template(x, tokenize=False)
                 for x in examples]
        # import pdb; pdb.set_trace()
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            examples, return_video_kwargs=True
        )
        pos_inputs = process_pos_info(examples)
        act_nav_labels = process_act_info(examples)

        image_inputs = _normalize_media_batch(image_inputs)
        video_inputs = _normalize_media_batch(video_inputs)
        video_kwargs = video_kwargs or [{} for _ in range(len(examples))]

        proc_kwargs = dict(text=texts, return_tensors="pt", padding=True)
        if image_inputs is not None and any(len(v) > 0 for v in image_inputs):
            proc_kwargs["images"] = image_inputs
        if video_inputs is not None and any(len(v) > 0 for v in video_inputs):
            proc_kwargs["videos"] = video_inputs
            if video_kwargs is not None:
                proc_kwargs["videos_kwargs"] = video_kwargs
        if pos_inputs is not None:
            proc_kwargs["pos"] = pos_inputs

        inputs = processor(**proc_kwargs)

        labels = inputs["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = -100
        labels[labels == pad_id] = -100
        for t in ignore_token_ids:
            labels[labels == t] = -100
        for i, messages in enumerate(examples):
            ids_i = inputs["input_ids"][i]

            start_idx = (ids_i == im_start_id).nonzero(as_tuple=True)[0]
            end_idx = (ids_i == im_end_id).nonzero(as_tuple=True)[0]
            num_turns = min(len(start_idx), len(end_idx))
            roles = [messages[t].get('role', 'user') for t in range(num_turns)]
            for t in range(num_turns):
                s = start_idx[t].item()
                e = end_idx[t].item()
                role = roles[t]

                if role in ('user', 'system'):
                    labels[i, s:e+1] = -100
                elif role == 'assistant':
                    if not last_trun or 'assistant' not in roles[t+1:]:
                        prompt_end = s + 1 + assistant_prompt_len
                        labels[i, s:prompt_end] = -100
                    else:
                        labels[i, s:e+1] = -100
                else:
                    labels[i, s:e+1] = -100
        inputs["labels"] = labels
        if act_nav_labels is not None:
            inputs["act_nav_labels"] = act_nav_labels
        return inputs

    return collate_fn
