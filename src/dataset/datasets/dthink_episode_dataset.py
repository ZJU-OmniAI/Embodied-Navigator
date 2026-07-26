from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ...utils import dataset_registry, prompt_registry


def _extract_sensor_pose(entry: Dict[str, Any]) -> List[float]:
    """
    Convert Habitat-style pose dict into [x, y, theta] tuple used by prompt builder.
    """
    if not entry:
        return [0.0, 0.0, 0.0]
    pos = entry.get("pos") or entry.get("position") or [0.0, 0.0, 0.0]
    euler = entry.get("euler") or entry.get(
        "euler_rotation") or [0.0, 0.0, 0.0]
    x = float(pos[0]) if len(pos) >= 1 else 0.0
    # Use world Z as planar Y to stay consistent with navigation projection.
    y = float(pos[2]) if len(pos) >= 3 else float(pos[-1]) if pos else 0.0
    theta = float(euler[1]) if len(euler) >= 2 else 0.0
    return [x, y, theta]


def calc_yaw(step):
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

def world_traj_to_body(traj_xzyaw):
    traj = np.asarray(traj_xzyaw, dtype=np.float64)
    if traj.size == 0:
        return []
    if traj.shape[0] == 1:
        return [[0.0, 0.0, 0.0]]

    x0, z0, yaw0 = traj[0]
    d = traj[:, :2] - np.array([x0, z0], dtype=np.float64)  # 平移到起点

    c, s = np.cos(yaw0), np.sin(yaw0)
    # world -> body: 逆时针旋转 yaw0（因为 yaw 正方向定义为顺时针）
    xb =  c * d[:, 0] - s * d[:, 1]
    zb =  s * d[:, 0] + c * d[:, 1]

    yaw = np.unwrap(traj[:, 2])
    yb = yaw - yaw[0]  # 起点机体系 yaw=0

    out = np.stack([xb, zb, yb], axis=1)
    return out.tolist()

@dataset_registry.register("dthink_episode")
class DThinkEpisodeDataset(Dataset):
    """
    Iterable dataset that reads generated D-Think episodes and produces chat messages
    via the DThink prompt builder. For an episode with N steps this dataset yields
    N samples, each corresponding to the prefixes [0], [0,1], ..., [0, ..., N-1].
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        *,
        split: str = "empty",
        prompt_name: str = "dthink_cot_prompt",
        max_pixels: Optional[int] = 256*256,
    ) -> None:
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Episode root not found: {self.base_path}")

        self.prompt_fn = prompt_registry.get(prompt_name)
        self.max_pixels = max_pixels

        self._episodes: List[Dict[str, Any]] = []
        self._index: List[Tuple[int, int]] = []
        self._build_episode_index()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_idx, prefix_len = self._index[idx]
        episode = self._episodes[episode_idx]

        prompt_steps = self._build_prompt_steps(episode, prefix_len)
        messages = self.prompt_fn.build_message(
            steps=prompt_steps,
            instruction=episode["instruction"],
            maxpixel=self.max_pixels,
        )
        step_ids = [step.get("step_id", i)
                    for i, step in enumerate(episode["steps"][:prefix_len])]

        return {
            "messages": messages,
            "episode_id": episode["episode_id"],
            "step_ids": step_ids,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_episode_index(self) -> None:
        episode_dirs = sorted(
            p for p in self.base_path.iterdir() if p.is_dir() and p.name.startswith("episode_")
        )
        for ep_dir in episode_dirs:
            ep_json = ep_dir / "episode.json"
            if not ep_json.exists():
                continue
            with open(ep_json, "r", encoding="utf-8") as f:
                payload = json.load(f)

            steps = payload.get("steps") or []
            if len(steps) == 0:
                continue

            episode_entry = {
                "episode_id": payload.get("episode_id") or ep_dir.name,
                "instruction": payload.get("instruct", ""),
                "steps": steps,
                "dir": ep_dir,
                "images_dir": ep_dir / "images",
                "has_png": (ep_dir / "images").is_dir(),
                "npz_path": ep_dir / "images.npz" if (ep_dir / "images.npz").exists() else None,
                "npz_cache": None,
            }
            self._episodes.append(episode_entry)
            episode_idx = len(self._episodes) - 1
            for prefix_len in range(1, len(steps) + 1):
                self._index.append((episode_idx, prefix_len))

    @staticmethod
    def _build_prompt_steps(
        episode: Dict[str, Any],
        prefix_len: int,
        *,
        resolve_image_handle: Optional[Callable[[Dict[str, Any], str], Union[str, Image.Image]]] = None,
    ) -> List[Dict[str, Any]]:
        prompt_steps: List[Dict[str, Any]] = []
        if prefix_len <= 0:
            return prompt_steps

        builder = DThinkEpisodeDataset
        image_resolver = resolve_image_handle or builder._resolve_image_handle

        history_steps = episode["steps"][: max(prefix_len - 1, 0)]
        current_step = episode["steps"][prefix_len - 1]

        keypoint_indices = [
            idx for idx, step in enumerate(history_steps) if len(step.get("cot")) > 5
        ]

        if keypoint_indices:
            first_kp = keypoint_indices[0]
            prefix_traj = history_steps[:first_kp]
            if prefix_traj:
                prompt_steps.extend(builder._build_trajectory_entries(prefix_traj))
        else:
            first_kp = None

        for i, kp_idx in enumerate(keypoint_indices):
            prompt_steps.append(
                builder._build_keypoint_entry(episode, history_steps[kp_idx], image_resolver)
            )
            next_kp_idx = keypoint_indices[i + 1] if i + \
                1 < len(keypoint_indices) else None
            if next_kp_idx is not None:
                segment = history_steps[kp_idx + 1: next_kp_idx]
                if segment:
                    prompt_steps.extend(builder._build_trajectory_entries(segment))

        if keypoint_indices:
            recent_start = keypoint_indices[-1] + 1
        else:
            recent_start = 0
        recent_steps = history_steps[recent_start:]
        if recent_steps:
            prompt_steps.extend(builder._build_recent_entries(episode, recent_steps, image_resolver))

        prompt_steps.append(
            {
                "type": "currentobs",
                "images": builder._format_image_entries(episode, current_step, image_resolver),
                "sensors_pos": builder._format_sensor_entries(current_step),
            }
        )
        
        if current_step.get("future_traj_spline", False):
            prompt_steps.append({"type": "currentact", "cot": current_step.get(
                "cot") or "", "traj": world_traj_to_body(calc_yaw(current_step))})

        return prompt_steps

    @staticmethod
    def _build_keypoint_entry(
        episode: Dict[str, Any],
        step: Dict[str, Any],
        resolve_image_handle: Callable[[Dict[str, Any], str], Union[str, Image.Image]],
    ) -> Dict[str, Any]:
        return {
            "type": "keypoint",
            "cot": step.get("cot", ""),
            "images": DThinkEpisodeDataset._format_image_entries(episode, step, resolve_image_handle),
            "sensors_pos": DThinkEpisodeDataset._format_sensor_entries(step),
        }

    @staticmethod
    def _build_trajectory_entries(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for step in steps:
            pose = step.get("agent_pose")
            if not pose:
                continue
            entries.append({"type": "trajectory", "agent_pos": [
                           _extract_sensor_pose(pose)]})
        return entries

    @staticmethod
    def _build_recent_entries(
        episode: Dict[str, Any],
        steps: List[Dict[str, Any]],
        resolve_image_handle: Callable[[Dict[str, Any], str], Union[str, Image.Image]],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for step in steps:
            sensor_name, image_entry = DThinkEpisodeDataset._select_single_view_image(
                episode, step, resolve_image_handle
            )
            sensor_pose = DThinkEpisodeDataset._format_single_sensor_entry(sensor_name, step)
            recent_step = {"type": "recentstep"}
            if image_entry:
                recent_step["image"] = image_entry
            if sensor_pose:
                recent_step["sensor_pos"] = sensor_pose
            entries.append(recent_step)
        return entries

    @staticmethod
    def _format_image_entries(
        episode: Dict[str, Any],
        step: Dict[str, Any],
        resolve_image_handle: Callable[[Dict[str, Any], str], Union[str, Image.Image]],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        image_map = step.get("images") or {}
        for sensor_name in sorted(image_map.keys()):
            key = image_map[sensor_name]
            entries.append(
                {
                    "name": sensor_name,
                    "image": resolve_image_handle(episode, key),
                }
            )
        return entries

    @staticmethod
    def _format_sensor_entries(step: Dict[str, Any]) -> List[Dict[str, Any]]:
        sensor_entries: List[Dict[str, Any]] = []
        sensor_map = step.get("sensor_poses") or {}
        for name in sorted(sensor_map.keys()):
            pose = sensor_map[name] or {}
            sensor_entries.append(
                {"name": name, "pos": _extract_sensor_pose(pose)})
        return sensor_entries

    @staticmethod
    def _select_single_view_image(
        episode: Dict[str, Any],
        step: Dict[str, Any],
        resolve_image_handle: Callable[[Dict[str, Any], str], Union[str, Image.Image]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        image_map = step.get("images") or {}
        priority = ("rgb_front", "rgb_left", "rgb_right", "rgb_back")
        selected_name = None
        for name in priority:
            if name in image_map:
                selected_name = name
                break
        if selected_name is None and image_map:
            selected_name = sorted(image_map.keys())[0]
        if selected_name is None:
            return None, None
        key = image_map[selected_name]
        return selected_name, {"name": selected_name, "image": resolve_image_handle(episode, key)}

    @staticmethod
    def _format_single_sensor_entry(sensor_name: Optional[str], step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sensor_map = step.get("sensor_poses") or {}
        pose = None
        if sensor_name and sensor_name in sensor_map:
            pose = sensor_map[sensor_name]
        elif sensor_map:
            first_name = sorted(sensor_map.keys())[0]
            pose = sensor_map.get(first_name)
            sensor_name = sensor_name or first_name
        if pose is None:
            return None
        return {"name": sensor_name, "pos": _extract_sensor_pose(pose)}

    @staticmethod
    def _resolve_image_handle(episode: Dict[str, Any], key: str) -> Union[str, Image.Image]:
        if episode["has_png"]:
            png_path = episode["images_dir"] / f"{key}.png"
            if png_path.exists():
                return str(png_path)
        npz = DThinkEpisodeDataset._load_npz_cache(episode)
        if key not in npz:
            raise KeyError(
                f"Image key '{key}' missing in {episode['episode_id']}")
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
