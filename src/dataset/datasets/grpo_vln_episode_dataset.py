from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from torch.utils.data import Dataset

from ...utils import dataset_registry


def _load_episode_file(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes") or []
    return episodes


@dataset_registry.register("grpo_vln_episode")
class GRPOVLNEpisodeDataset(Dataset):
    """
    轻量版 GRPO VLN 数据集读取器。
    读取 Habitat 标准 episode json（参考 data/dthink_sft_data/r2xr_cot/sample/sample.json），
    返回包含 episode_id / instruction / scene_id 等字段的样本，供 GRPOTrainer rollout 使用。
    """

    def __init__(self, base_path: Union[str, Path], *, split: str = "train") -> None:
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.base_path}")

        json_files: List[Path]
        if self.base_path.is_dir():
            json_files = sorted(self.base_path.rglob("*.json"))
        else:
            json_files = [self.base_path]

        self.episodes: List[Dict[str, Any]] = []
        for jp in json_files:
            self.episodes.extend(_load_episode_file(jp))

        if len(self.episodes) == 0:
            raise ValueError(f"No episodes found in {self.base_path}")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = self.episodes[idx]
        return {
            "episode_id": ep.get("episode_id"),
            "trajectory_id": ep.get("trajectory_id"),
            "scene_id": ep.get("scene_id"),
            "instruction": ep.get("instruction", ""),
        }
