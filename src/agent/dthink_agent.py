from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:  # torch is optional at runtime
    import torch
except ImportError:  # pragma: no cover - fallback when torch is unavailable
    torch = None

from ..dataset.collate_fn import drop_none
from ..utils import agent_registry, prompt_registry
from ..utils.qwen_vl_utils import process_pos_info, process_vision_info


@dataclass
class AgentState:
    """Lightweight container passed between env and agent."""

    instruction: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    step_id: int = 0
    episode_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@agent_registry.register("dthink_agent")
class DThinkAgent:
    """
    Minimal online agent that keeps no direct handle to the environment.
    Interaction happens through `reset` and `step` by passing observations
    and the rolling AgentState.
    """

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        *,
        prompt_name: str = "dthink_cot_prompt",
        max_pixels: int = 8192,
        device: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        do_sample: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        if not prompt_registry.has(prompt_name):
            raise KeyError(f"Prompt '{prompt_name}' not registered")
        self.prompt = prompt_registry.get(prompt_name)
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        if device is None and torch is not None and torch.cuda.is_available():
            device = "cuda"
        self.device = device or "cpu"
        if self.model is not None and hasattr(self.model, "to"):
            try:
                self.model.to(self.device)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        instruction: str,
        first_obs: Optional[Dict[str, Any]] = None,
        *,
        sensor_poses: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """
        Start a new episode. No environment handle is stored; caller feeds
        observations through subsequent `step` calls.
        """
        state = AgentState(
            instruction=instruction or "",
            episode_id=episode_id,
            meta=meta or {},
        )
        if first_obs is not None:
            state.meta["last_obs"] = first_obs
        if sensor_poses is not None:
            state.meta["last_sensor_poses"] = sensor_poses
        return state

    def step(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        *,
        sensor_poses: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, AgentState, Dict[str, Any]]:
        """
        Build a prompt from history + current observation, run the model, and
        return (action, new_state, raw_model_output).
        """
        prompt_steps = self._build_prompt_steps(state, obs, sensor_poses)
        messages = self.prompt.build_message(
            steps=prompt_steps,
            instruction=state.instruction,
            maxpixel=self.max_pixels,
        )

        model_out = self._run_model(messages)
        action = model_out.get("action") or "STOP"
        cot = model_out.get("cot") or model_out.get("text") or ""
        traj = model_out.get("traj")

        state.history.append(
            self._history_entry(obs, sensor_poses, action, cot, traj)
        )
        state.step_id += 1
        state.meta["last_obs"] = obs
        if sensor_poses is not None:
            state.meta["last_sensor_poses"] = sensor_poses
        state.meta["last_messages"] = messages

        return action, state, model_out

    # ------------------------------------------------------------------ #
    # Prompt construction                                                #
    # ------------------------------------------------------------------ #
    def _build_prompt_steps(
        self,
        state: AgentState,
        obs: Dict[str, Any],
        sensor_poses: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        history_steps = state.history

        keypoint_indices = [
            idx for idx, s in enumerate(history_steps)
            if len((s.get("cot") or "")) > 5
        ]

        if keypoint_indices:
            first_kp = keypoint_indices[0]
            prefix_traj = history_steps[:first_kp]
            if prefix_traj:
                steps.extend(self._build_trajectory_entries(prefix_traj))

        for i, kp_idx in enumerate(keypoint_indices):
            steps.append(self._build_keypoint_entry(history_steps[kp_idx]))
            next_kp_idx = keypoint_indices[i + 1] if i + 1 < len(keypoint_indices) else None
            if next_kp_idx is not None:
                seg = history_steps[kp_idx + 1: next_kp_idx]
                if seg:
                    steps.extend(self._build_trajectory_entries(seg))

        recent_start = (keypoint_indices[-1] + 1) if keypoint_indices else 0
        recent_steps = history_steps[recent_start:]
        if recent_steps:
            steps.extend(self._build_recent_entries(recent_steps))

        steps.append(
            {
                "type": "currentobs",
                "images": self._format_image_entries(obs),
                "sensors_pos": self._format_sensor_entries(sensor_poses, obs),
            }
        )
        return steps

    def _action_query_text(self) -> str:
        return (
            f"{self.prompt.THINK_START}"
            f"Reason briefly then output the next navigation action."
            f"{self.prompt.THINK_END}"
            f"{self.prompt.ACTION_SUFFIX}"
        )

    # ------------------------------------------------------------------ #
    # Model execution                                                    #
    # ------------------------------------------------------------------ #
    def _run_model(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.model is None or self.processor is None:
            return {"text": "", "action": "STOP"}

        inputs, prompt_len = self._build_model_inputs(messages)
        if not inputs:
            return {"text": "", "action": "STOP"}

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )

        # Strip the prompt portion before decoding
        if isinstance(generated, tuple):
            generated = generated[0]
        if hasattr(generated, "tolist") and "input_ids" in inputs:
            prompt_tokens = prompt_len
            decoded = self.processor.batch_decode(
                generated[:, prompt_tokens:], skip_special_tokens=True
            )
            text_out = decoded[0] if decoded else ""
        else:
            text_out = ""

        return {
            "text": text_out,
            "action": self._parse_action(text_out),
            "messages": messages,
        }

    def _build_model_inputs(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], int]:
        clean_messages = drop_none(messages)
        text = self.processor.apply_chat_template(
            clean_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [clean_messages], return_video_kwargs=True
        )
        pos_inputs = process_pos_info([clean_messages])

        proc_kwargs: Dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
        }
        if image_inputs and any(len(v) > 0 for v in image_inputs):
            proc_kwargs["images"] = image_inputs
        if video_inputs and any(len(v) > 0 for v in video_inputs):
            proc_kwargs["videos"] = video_inputs
            if video_kwargs is not None:
                proc_kwargs["videos_kwargs"] = video_kwargs
        if pos_inputs is not None:
            proc_kwargs["pos"] = pos_inputs

        inputs = self.processor(**proc_kwargs)
        if self.device and torch is not None:
            inputs = {k: self._move_to_device(v) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        return inputs, prompt_len

    # ------------------------------------------------------------------ #
    # Utilities                                                          #
    # ------------------------------------------------------------------ #
    def _history_entry(
        self,
        obs: Dict[str, Any],
        sensor_poses: Optional[Dict[str, Any]],
        action: str,
        cot: str,
        traj: Optional[Any],
    ) -> Dict[str, Any]:
        image_entries = self._format_image_entries(obs)
        sensor_entries = self._format_sensor_entries(sensor_poses, obs)
        return {
            "images": image_entries,
            "sensor_poses": sensor_entries,
            "action": action,
            "cot": cot,
            "traj": traj,
            "agent_pose": self._extract_agent_pose(obs),
        }

    def _format_image_entries(self, obs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if obs is None:
            return []
        entries: List[Dict[str, Any]] = []
        image_map = None
        if isinstance(obs, dict):
            # Prefer explicit image map
            if "images" in obs and isinstance(obs["images"], dict):
                image_map = obs["images"]
            else:
                image_map = obs
        if not image_map:
            return []
        for name, value in image_map.items():
            img = self._maybe_to_pil(value)
            if img is None:
                continue
            entries.append({"name": name, "image": img})
        return entries

    def _maybe_to_pil(self, value: Any) -> Optional[Image.Image]:
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, np.ndarray):
            arr = value
            if arr.ndim == 2:  # grayscale / depth not supported here
                return None
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                arr = arr.astype(np.uint8)
                return Image.fromarray(arr)
        return None

    def _format_sensor_entries(
        self, sensor_poses: Optional[Dict[str, Any]], obs: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        source = sensor_poses or (obs.get("sensor_poses") if isinstance(obs, dict) else None)
        if not source:
            return []
        entries: List[Dict[str, Any]] = []
        for name, pose in source.items():
            pos = self._extract_sensor_pose(pose)
            if pos is None:
                continue
            entries.append({"name": name, "pos": pos})
        return entries

    def _extract_sensor_pose(self, entry: Any) -> Optional[List[float]]:
        if entry is None:
            return None
        if isinstance(entry, dict):
            pos = entry.get("pos") or entry.get("position")
            euler = entry.get("euler") or entry.get("euler_rotation")
            if pos is not None and euler is not None and len(euler) >= 2:
                x = float(pos[0]) if len(pos) >= 1 else 0.0
                y = float(pos[2]) if len(pos) >= 3 else float(pos[-1]) if pos else 0.0
                theta = float(euler[1])
                return [x, y, theta]
            if all(k in entry for k in ("x", "y", "theta")):
                return [float(entry["x"]), float(entry["y"]), float(entry["theta"])]
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            return [float(entry[0]), float(entry[1]), float(entry[2])]
        return None

    def _extract_agent_pose(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(obs, dict):
            return None
        pose = obs.get("agent_pose") or obs.get("agent_state")
        if pose is None:
            return None
        pos = None
        euler = None
        if isinstance(pose, dict):
            pos = pose.get("pos") or pose.get("position")
            euler = pose.get("euler") or pose.get("euler_rotation")
        if pos is None or euler is None or len(pos) < 3 or len(euler) < 2:
            return None
        return {"pos": pos, "euler": euler}

    def _pick_primary_image(
        self, entries: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not entries:
            return None
        priority = ("rgb_front", "rgb_left", "rgb_right", "rgb_back")
        for p in priority:
            for e in entries:
                if e.get("name") == p:
                    return e
        return entries[0]

    def _build_keypoint_entry(self, step: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "keypoint",
            "cot": step.get("cot", ""),
            "images": step.get("images") or [],
            "sensors_pos": step.get("sensor_poses") or [],
        }

    def _build_trajectory_entries(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for step in steps:
            pose = step.get("agent_pose")
            pos = self._extract_sensor_pose(pose) if pose else None
            if pos is None:
                continue
            out.append({"type": "trajectory", "agent_pos": [pos]})
        return out

    def _build_recent_entries(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for step in steps:
            images = step.get("images") or []
            sensor_poses = step.get("sensor_poses") or []
            primary_image = self._pick_primary_image(images)
            primary_pose = None
            if sensor_poses:
                # pick matching pose if possible
                if primary_image:
                    match_name = primary_image.get("name")
                    for sp in sensor_poses:
                        if sp.get("name") == match_name:
                            primary_pose = sp
                            break
                primary_pose = primary_pose or sensor_poses[0]

            item: Dict[str, Any] = {"type": "recentstep"}
            if primary_image:
                item["image"] = primary_image
            if primary_pose:
                item["sensor_pos"] = primary_pose
            out.append(item)
        return out

    def _parse_action(self, text: str) -> str:
        if not text:
            return "STOP"
        upper = text.upper()
        for name in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            if name in upper:
                return name
        if "LEFT" in upper:
            return "TURN_LEFT"
        if "RIGHT" in upper:
            return "TURN_RIGHT"
        if "FORWARD" in upper or "ADVANCE" in upper:
            return "MOVE_FORWARD"
        return "STOP"

    def _move_to_device(self, obj: Any) -> Any:
        if torch is None or self.device is None:
            return obj
        if hasattr(obj, "to"):
            try:
                return obj.to(self.device)
            except Exception:
                return obj
        if isinstance(obj, (list, tuple)):
            return obj.__class__(self._move_to_device(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        return obj
