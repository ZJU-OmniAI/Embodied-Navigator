from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from ...utils import dataset_registry, prompt_registry
from .dthink_episode_dataset import DThinkEpisodeDataset

def calc_yaw(step: Dict[str, Any]):
    agent_pose = step.get("agent_pose")
    if not agent_pose:
        return None
    euler = agent_pose.get("euler")
    if not euler:
        return None
    traj = step.get("future_traj_spline")
    if not traj:
        return None

    yaw0 = float(euler[1])
    traj = np.asarray(traj, dtype=np.float64)

    if len(traj) == 0:
        return []
    if len(traj) == 1:
        x0, z0 = float(traj[0, 0]), float(traj[0, 2])
        return [[x0, z0, yaw0]]

    xy = traj[:, [0, 2]]
    dxy = xy[1:] - xy[:-1]
    yaw_seg = np.arctan2(-dxy[:, 0], -dxy[:, 1])
    yaw_pts = np.unwrap(np.concatenate([[yaw0], yaw_seg]))
    out = np.concatenate([xy, yaw_pts[:, None]], axis=1)
    return out.tolist()


@dataset_registry.register("dthink_episode_stream")
class DThinkEpisodeDatasetStream(Dataset):
    """
    Map-style Dataset：
    - __len__ 固定为 length（默认 10000000）
    - __getitem__ 忽略 idx，每次返回“下一个样本”
    - 通过原子 rename 抢占 episode_*，读取后放入本地 cache
    - episode 的所有 prefix 消费完后删除该 episode 目录
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        *,
        split: str = "empty",
        prompt_name: str = "dthink_cot_prompt",
        max_pixels: Optional[int] = 256*256,
        length: int = 10000000,
        cache_episodes: int = 8,
        seed: int = 0,
        claim_retries: int = 64,
    ) -> None:
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Episode root not found: {self.base_path}")

        self.prompt_fn = prompt_registry.get(prompt_name)
        self.max_pixels = max_pixels

        self._length = int(length)
        self.cache_episodes = int(cache_episodes)
        self.seed = int(seed)
        self.claim_retries = int(claim_retries)

        # 每个 worker 内部状态（Dataset 会被 worker 进程各自拷贝）
        self._rng: Optional[np.random.Generator] = None
        self._queue: List[Tuple[Dict[str, Any], int]] = []  # (episode, prefix_len)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # idx ignored
        if self._rng is None:
            wi = get_worker_info()
            wid = 0 if wi is None else wi.id
            self._rng = np.random.default_rng(self.seed + 9973 * wid)

        if not self._queue:
            self._refill_queue(self.cache_episodes)
            if not self._queue:
                raise RuntimeError("No more episodes available under base_path (stream exhausted).")

        ep, prefix_len = self._queue.pop(0)

        steps = ep["steps"]
        # import pdb; pdb.set_trace()
        prompt_steps = self._build_prompt_steps(ep, prefix_len)
        messages = self.prompt_fn.build_message(
            steps=prompt_steps,
            instruction=ep["instruction"],
            maxpixel=self.max_pixels,
        )
        step_ids = [s.get("step_id", i) for i, s in enumerate(steps[:prefix_len])]

        # episode 最后一个 prefix 消费完 -> 删除目录
        if prefix_len >= ep["_num_steps"]:
            self._cleanup_episode(ep)

        return {
            "messages": messages,
            "episode_id": ep["episode_id"],
            "step_ids": step_ids,
        }

    # ------------------------------------------------------------------
    # cache / claiming
    # ------------------------------------------------------------------
    def _refill_queue(self, n_episodes: int) -> None:
        assert self._rng is not None
        for _ in range(max(0, n_episodes)):
            ep_dir = self._claim_one_episode_dir()
            if ep_dir is None:
                return
            ep = self._load_episode_dir(ep_dir)
            if ep is None:
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue
            # 把该 episode 的所有 prefix 都塞进 queue
            for prefix_len in range(1, ep["_num_steps"] + 1):
                self._queue.append((ep, prefix_len))

    def _claim_one_episode_dir(self) -> Optional[Path]:
        assert self._rng is not None
        for _ in range(self.claim_retries):
            candidates = [
                p for p in self.base_path.iterdir()
                if p.is_dir() and p.name.startswith("episode_")
            ]
            if not candidates:
                time.sleep(5)
                continue

            src = candidates[int(self._rng.integers(0, len(candidates)))]
            dst_name = f"_load_cache_episode_{src.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            dst = src.with_name(dst_name)
            try:
                src.rename(dst)  # 原子 rename（同一文件系统）
                return dst
            except OSError:
                time.sleep(5)
                continue
        return None

    def _load_episode_dir(self, ep_dir: Path) -> Optional[Dict[str, Any]]:
        ep_json = ep_dir / "episode.json"
        if not ep_json.exists():
            return None
        try:
            with open(ep_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None

        steps = payload.get("steps") or []
        if not steps:
            return None

        images_dir = ep_dir / "images"
        npz_path = ep_dir / "images.npz"
        return {
            "episode_id": payload.get("episode_id") or ep_dir.name,
            "instruction": payload.get("instruct", ""),
            "steps": steps,
            "_num_steps": len(steps),
            "dir": ep_dir,
            "images_dir": images_dir,
            "has_png": images_dir.is_dir(),
            "npz_path": npz_path if npz_path.exists() else None,
            "npz_cache": None,
        }

    def _cleanup_episode(self, ep: Dict[str, Any]) -> None:
        npz = ep.get("npz_cache")
        try:
            if npz is not None and hasattr(npz, "close"):
                npz.close()
        except Exception:
            pass
        shutil.rmtree(ep["dir"], ignore_errors=True)

    # ------------------------------------------------------------------
    # prompt building (shared logic)
    # ------------------------------------------------------------------
    def _build_prompt_steps(self, episode: Dict[str, Any], prefix_len: int) -> List[Dict[str, Any]]:
        return DThinkEpisodeDataset._build_prompt_steps(
            episode,
            prefix_len,
            resolve_image_handle=self._resolve_image_handle,
        )

    def _resolve_image_handle(self, episode: Dict[str, Any], key: str) -> Image.Image:
        # 为了能“消费完就删目录”，这里强制把 PNG 也读成 PIL（避免返回路径）
        if episode["has_png"]:
            png_path = episode["images_dir"] / f"{key}.png"
            if png_path.exists():
                with Image.open(png_path) as im:
                    return im.copy()

        npz = self._load_npz_cache(episode)
        if key not in npz:
            raise KeyError(f"Image key '{key}' missing in {episode['episode_id']}")
        arr = np.array(npz[key])
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _load_npz_cache(episode: Dict[str, Any]) -> Any:
        if episode.get("npz_cache") is None:
            npz_path = episode.get("npz_path")
            if npz_path is None:
                raise FileNotFoundError(
                    f"No PNG directory or images.npz available for episode {episode['episode_id']}"
                )
            episode["npz_cache"] = np.load(npz_path)
        return episode["npz_cache"]
