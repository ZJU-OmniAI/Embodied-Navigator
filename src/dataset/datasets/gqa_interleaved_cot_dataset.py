from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset

from ...utils import dataset_registry


@dataset_registry.register("gqa_interleaved_cot")
class GQAInterleavedCOTDataset(Dataset):
    """
    Map-style dataset for data/share/GQA_Interleaved_COT.

    Each sample yields a chat-style message list:
      - system: optional helper prompt
      - user: image + question
      - assistant: two content chunks
          * think: concatenated CoT from reasoning_steps
          * text : final answer (generated_answer fallback to answer)
    """

    SPLIT_FILES: Dict[str, str] = {
        "train_easy": "gqa_train_easy_data_final.json",
        "train_easy_pos": "gqa_train_easy_data_positives_final.json",
        "train_easy_neg": "gqa_train_easy_data_negatives_final.json",
        "train_med": "gqa_train_med_data_final.json",
        "train_med_fixed": "gqa_train_med_data_fixed_final.json",
        "train_med_pos": "gqa_train_med_data_positives_final.json",
        "train_med_neg": "gqa_train_med_data_negatives_final.json",
        "train_hard": "gqa_train_hard_data_final.json",
        "train_hard_pos": "gqa_train_hard_data_positives_final.json",
        "train_hard_neg": "gqa_train_hard_data_negatives_final.json",
    }
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant."
    )

    def __init__(
        self,
        base_path: Union[str, Path],
        *,
        split: str = "train_easy",
        max_pixels: Optional[int] = 256*256,
        system_prompt: Optional[str] = None,
        use_generated_answer: bool = True,
    ) -> None:
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.base_path}")
        
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_pixels = max_pixels
        self.use_generated_answer = bool(use_generated_answer)

        json_path = self._resolve_split_path(split)
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = json.load(f)

        self.image_dir = self.base_path / "all_images"
        self._missing_images: List[tuple[Any, str]] = []
        self.samples = self._validate_and_filter_samples(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = self._resolve_image(sample.get("images", ""))
        question = sample.get("question", "")
        answer = self._pick_answer(sample)
        cot = self._build_cot(sample.get("reasoning_steps") or [])

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        **({"max_pixels": self.max_pixels} if self.max_pixels is not None else {}),
                    },
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "think", "think": cot},
                    {"type": "text", "text": answer},
                ],
            },
        ]

        return {
            "messages": messages,
            "question_id": sample.get("id"),
            "image_path": image_path,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_split_path(self, split: str) -> Path:
        if (self.base_path / split).is_file():
            return self.base_path / split
        fname = self.SPLIT_FILES.get(split)
        if fname is None:
            available = ", ".join(sorted(self.SPLIT_FILES))
            raise ValueError(f"Unknown split '{split}'. Available: {available}")
        path = self.base_path / fname
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        return path

    def _validate_and_filter_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove samples whose image file is missing. Missing ones are recorded
        and printed once for visibility.
        """
        kept: List[Dict[str, Any]] = []
        for sample in samples:
            img_field = sample.get("images", "")
            name = Path(img_field).name
            cand = self.image_dir / name
            fallback = self.base_path / img_field
            if cand.exists() or fallback.exists():
                kept.append(sample)
            else:
                self._missing_images.append((sample.get("id"), img_field))

        if self._missing_images:
            print(
                f"[GQAInterleavedCOTDataset] dropped {len(self._missing_images)} samples with missing images"
            )
        return kept

    def _resolve_image(self, img_field: str) -> str:
        """
        Convert placeholder path like 'images/123.jpg' into real path
        under base_path/all_images.
        """
        name = Path(img_field).name
        cand = self.image_dir / name
        if cand.exists():
            return str(cand)
        fallback = self.base_path / img_field
        if fallback.exists():
            return str(fallback)
        raise FileNotFoundError(f"Image not found for entry: {img_field}")

    @staticmethod
    def _build_cot(steps: List[Dict[str, Any]]) -> str:
        if not steps:
            return ""
        lines = []
        for i, step in enumerate(steps):
            thought = step.get("thought") or ""
            # Remove any leading "Step <num>:" prefixes in the provided text.
            thought = re.sub(r"^\\s*Step\\s*\\d+\\s*:\\s*", "", thought, flags=re.IGNORECASE)
            lines.append(thought)
        return "\n".join(lines)

    def _pick_answer(self, sample: Dict[str, Any]) -> str:
        if self.use_generated_answer:
            gen = sample.get("generated_answer")
            if gen:
                return gen
        return sample.get("answer", "")