from __future__ import annotations

import json
import os
import shutil
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from ...utils import dataset_registry, prompt_registry


def _extract_sensor_pose(entry: Dict[str, Any]) -> List[float]:
    if not entry:
        return [0.0, 0.0, 0.0]
    pos = entry.get("pos") or entry.get("position") or [0.0, 0.0, 0.0]
    euler = entry.get("euler") or entry.get("euler_rotation") or [0.0, 0.0, 0.0]
    x = float(pos[0]) if len(pos) >= 1 else 0.0
    y = float(pos[2]) if len(pos) >= 3 else float(pos[-1]) if pos else 0.0
    theta = float(euler[1]) if len(euler) >= 2 else 0.0
    return [x, y, theta]


def _build_pixel_action(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    point = step.get("point") or {}
    proj = point.get("chosen_projection") or {}
    u, v = proj.get("u"), proj.get("v")
    if u is None or v is None:
        return None
    choice = point.get("chosen_sensor") or proj.get("sensor") or ""
    return {"choice": choice, "pixel": [float(u), float(v)]}


@dataset_registry.register("dthink_episode_pixel_stream")
class DThinkEpisodeDatasetPixelStream(Dataset):
    """流式消费 pixel 版 episode，action 为 {"choice":..., "pixel":[u,v]}"""

    def __init__(
        self,
        base_path: Union[str, Path],
        *,
        split: str = "empty",
        prompt_name: str = "dthink_cot_prompt_pixel",
        max_pixels: Optional[int] = 10 * 10 * 28 * 28,
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

        self._rng: Optional[np.random.Generator] = None
        self._queue: List[Tuple[Dict[str, Any], int]] = []

    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.__getitem_inner__(idx)
        # while item["prefix_len"] < 4 and random.random() > 0.5:
        #     item = self.__getitem_inner__(idx)
        return item

    def __getitem_inner__(self, idx: int) -> Dict[str, Any]:
        if self._rng is None:
            wi = get_worker_info()
            wid = 0 if wi is None else wi.id
            self._rng = np.random.default_rng(self.seed + 9973 * wid)

        if not self._queue:
            self._refill_queue(self.cache_episodes)
            if not self._queue:
                raise RuntimeError("No more episodes available under base_path (stream exhausted).")

        ep, prefix_len = self._queue[0]
        steps = ep["steps"]

        prompt_steps = self._build_prompt_steps(ep, prefix_len)
        
        messages = self.prompt_fn.build_message(
            steps=prompt_steps,
            instruction=ep.get("instruction", ""),
            maxpixel=self.max_pixels,
        )
        
        is_reduce_hall = random.random() > 0.8 and len(prompt_steps[-1]["cot"]) > 10
        if is_reduce_hall:
            messages = [
                messages[0],
                messages[-2],
                {
                    "role": "assistant",
                    "content": [
                        messages[-1]["content"][0]
                    ],
                },               
                {
                    "role": "assistant",
                    "content": [
                        messages[-1]["content"][1]
                    ],
                },
            ]
        
        
        step_ids = [s.get("step_id", i) for i, s in enumerate(steps[:prefix_len])]

        # action = prompt_steps[-1]['action']
        # if action is not None and action['choice'] in ["rgb_front"]:
        #     if random.random() > 0.4:
        #         self._queue.pop(0)
        if is_reduce_hall:
            pass
        # elif 3 < prefix_len < ep["_num_steps"] - 1:
        #     if random.random() > 0.6:
        #         self._queue.pop(0)
        #         if prefix_len >= ep["_num_steps"]:
        #             self._cleanup_episode(ep)
        # elif 0 < prefix_len < ep["_num_steps"] - 1 and len(prompt_steps[-1]["cot"]) > 10:
        #     if random.random() > 0.5:
        #         self._queue.pop(0)
        #         if prefix_len >= ep["_num_steps"]:
        #             self._cleanup_episode(ep)
        else:
            self._queue.pop(0)
            if prefix_len >= ep["_num_steps"]:
                self._cleanup_episode(ep)
        
        # self._queue.pop(0)
        # if prefix_len >= ep["_num_steps"]:
        #     self._cleanup_episode(ep)

        return {
            "messages": messages,
            "episode_id": ep["episode_id"],
            "step_ids": step_ids,
            "prefix_len": prefix_len,
        }

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
            for prefix_len in range(1, ep["_num_steps"] + 1):
                self._queue.append((ep, prefix_len))

    def _claim_one_episode_dir(self) -> Optional[Path]:
        assert self._rng is not None
        for _ in range(self.claim_retries):
            candidates = [p for p in self.base_path.iterdir() if p.is_dir() and p.name.startswith("episode_")]
            if not candidates:
                time.sleep(5)
                continue
            src = candidates[int(self._rng.integers(0, len(candidates)))]
            dst_name = f"_load_cache_episode_{src.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            dst = src.with_name(dst_name)
            try:
                src.rename(dst)
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
    # prompt helpers
    # ------------------------------------------------------------------
    def _build_prompt_steps(self, episode: Dict[str, Any], prefix_len: int) -> List[Dict[str, Any]]:
        prompt_steps: List[Dict[str, Any]] = []
        if prefix_len <= 0:
            return prompt_steps

        history_steps = episode["steps"][: max(prefix_len - 1, 0)]
        current_step = episode["steps"][prefix_len - 1]

        keypoint_indices = [idx for idx, step in enumerate(history_steps) if len(step.get("cot") or "") > 5]

        if keypoint_indices:
            first_kp = keypoint_indices[0]
            prefix_traj = history_steps[:first_kp]
            if prefix_traj:
                prompt_steps.extend(self._build_trajectory_entries(prefix_traj))

        for i, kp_idx in enumerate(keypoint_indices):
            prompt_steps.append(self._build_keypoint_entry(episode, history_steps[kp_idx]))
            next_kp_idx = keypoint_indices[i + 1] if i + 1 < len(keypoint_indices) else None
            if next_kp_idx is not None:
                segment = history_steps[kp_idx + 1: next_kp_idx]
                if segment:
                    prompt_steps.extend(self._build_trajectory_entries(segment))

        recent_start = keypoint_indices[-1] + 1 if keypoint_indices else 0
        recent_steps = history_steps[recent_start:]
        if recent_steps:
            prompt_steps.extend(self._build_recent_entries(episode, recent_steps))

        # current obs
        prompt_steps.append(
            {
                "type": "currentobs",
                "images": self._format_image_entries(episode, current_step),
                "sensors_pos": self._format_sensor_entries(current_step),
            }
        )

        act_json = _build_pixel_action(current_step)
        prompt_steps.append({"type": "currentact", "cot": current_step.get("cot").replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "") or "", "action": act_json})

        return prompt_steps

    # formatting helpers (reuse from non-stream but simplified)
    def _build_keypoint_entry(self, episode: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "keypoint",
            "cot": step.get("cot", ""),
            "images": self._format_image_entries(episode, step),
            "sensors_pos": self._format_sensor_entries(step),
        }

    def _build_trajectory_entries(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for step in steps:
            pose = step.get("agent_pose")
            if not pose:
                continue
            entries.append({"type": "trajectory", "agent_pos": [_extract_sensor_pose(pose)]})
        return entries

    def _build_recent_entries(self, episode: Dict[str, Any], steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for step in steps:
            sensor_name, image_entry = self._select_single_view_image(episode, step)
            sensor_pose = self._format_single_sensor_entry(sensor_name, step)
            recent_step = {"type": "recentstep"}
            if image_entry:
                recent_step["image"] = image_entry
            if sensor_pose:
                recent_step["sensor_pos"] = sensor_pose
            entries.append(recent_step)
        return entries

    def _format_image_entries(self, episode: Dict[str, Any], step: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        image_map = step.get("images") or {}
        for sensor_name in sorted(image_map.keys()):
            key = image_map[sensor_name]
            entries.append({"name": sensor_name, "image": self._resolve_image_handle(episode, key)})
        return entries

    def _format_sensor_entries(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        sensor_map = step.get("sensor_poses") or {}
        for sensor_name in sorted(sensor_map.keys()):
            pose = sensor_map[sensor_name] or {}
            entries.append({"name": sensor_name, "pos": _extract_sensor_pose(pose)})
        return entries

    def _format_single_sensor_entry(self, sensor_name: str, step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sensor_map = step.get("sensor_poses") or {}
        if sensor_name in sensor_map:
            data = sensor_map[sensor_name]
            return {"name": sensor_name, "pos": _extract_sensor_pose(data)}
        return None
    
    def _select_single_view_image(self, episode: Dict[str, Any], step: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        image_map = step.get("images") or {}
        if not image_map:
            return "", {}
        point = step.get("point") or {}
        proj = point.get("chosen_projection") or {}
        chosen = proj.get("sensor") or "rgb_front"
        key = image_map[chosen]
        return chosen, {"name": chosen, "image": self._resolve_image_handle(episode, key)}

    # image loading
    def _resolve_image_handle(self, episode: Dict[str, Any], key: str) -> Image.Image:
        if episode["has_png"]:
            png_path = episode["images_dir"] / f"{key}.png"
            if png_path.exists():
                with Image.open(png_path) as im:
                    return im.copy()

        npz = self._load_npz_cache(episode)
        if key not in npz:
            raise KeyError(f"Image key {key} missing in {episode.get('episode_id', '-1')}")
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
                    f"No PNG directory or images.npz available for episode {episode.get('episode_id', '-1')}"
                )
            episode["npz_cache"] = np.load(npz_path)
        return episode["npz_cache"]

