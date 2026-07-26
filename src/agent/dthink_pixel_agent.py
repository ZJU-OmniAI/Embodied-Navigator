from __future__ import annotations

import json
import re
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
from .dthink_agent import AgentState


def _maybe_to_pil(value: Any) -> Optional[Image.Image]:
    """Convert ndarray/PIL to PIL.Image; skip unsupported shapes."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 2:  # grayscale / depth not supported here
            return None
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr.astype(np.uint8)
            return Image.fromarray(arr)
        if arr.ndim == 3 and arr.shape[-1] in (1):
            arr = (arr*255).astype(np.uint8)
            return Image.fromarray(arr)
    return None


@agent_registry.register("dthink_pixel_agent")
class DThinkPixelAgent:
    """
    Pixel-action agent compatible with DThinkEnv.
    - Builds multi-view prompts via DThinkCotPromptPixel.
    - Runs Qwen2.5-VL to predict a pixel (u, v) on a chosen sensor view.
    - Parses model text -> structured action dict and keeps step history.
    """

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        *,
        model_path: Optional[str] = None,
        prompt_name: str = "dthink_cot_prompt_pixel",
        max_pixels: int = 10 * 10 * 28 * 28,
        device: Optional[str] = None,
        device_map: Optional[str] = "auto",
        torch_dtype: Any = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> None:
        if not prompt_registry.has(prompt_name):
            raise KeyError(f"Prompt '{prompt_name}' not registered")

        if model is None or processor is None:
            if model_path is None:
                raise ValueError(
                    "model_path is required when model or processor is None")
            from ..model.qwen2_5_vl import (  # lazy import to keep import cost low
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLProcessor,
            )

            processor = processor or Qwen2_5_VLProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            model = model or Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=self._resolve_dtype(torch_dtype),
                trust_remote_code=True,
            )

        self.model = model
        self.processor = processor
        self.prompt = prompt_registry.get(prompt_name)
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
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
    ) -> Tuple[Dict[str, Any], AgentState, Dict[str, Any]]:
        prompt_steps = self._build_prompt_steps(state, obs, sensor_poses)
        messages = self.prompt.build_message(
            steps=prompt_steps,
            instruction=state.instruction,
            maxpixel=self.max_pixels,
        )
        model_out = self._run_model_twice(messages)
        text_out = model_out.get("text") or ""
        action = self._parse_action(text_out, obs)
        model_out["action"] = action

        state.history.append(
            self._history_entry(obs, sensor_poses, action,
                                text_out, model_out.get("traj"))
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
            next_kp_idx = keypoint_indices[i + 1] if i + \
                1 < len(keypoint_indices) else None
            if next_kp_idx is not None:
                # steps[-1]['cot'] = ""
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
            if pose is None:
                continue
            out.append({"type": "trajectory", "agent_pos": pose})
        return out

    def _build_recent_entries(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for step in steps:
            images = step.get("images") or []
            sensor_poses = step.get("sensor_poses") or []
            primary_image = self._pick_primary_image(images)
            primary_pose = None
            if sensor_poses:
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

    # ------------------------------------------------------------------ #
    # Model execution                                                    #
    # ------------------------------------------------------------------ #
    def _run_model(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.model is None or self.processor is None:
            return {"text": "", "action": {"type": "CMD", "action": "STOP"}}

        inputs, prompt_len = self._build_model_inputs(messages)
        if not inputs:
            return {"text": "", "action": {"type": "CMD", "action": "STOP"}}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
        }
        if self.do_sample and self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p

        generated = self.model.generate(**inputs, **gen_kwargs)

        if isinstance(generated, tuple):
            generated = generated[0]
        if hasattr(generated, "tolist") and "input_ids" in inputs:
            prompt_tokens = prompt_len
            decoded = self.processor.batch_decode(
                generated[:, prompt_tokens:], skip_special_tokens=False
            )
            text_out = decoded[0] if decoded else ""
        else:
            text_out = ""

        return {
            "text": text_out,
            "action": self._parse_action(text_out),
            "messages": messages,
        }

    def _run_model_twice(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_out = self._run_model(messages)
        return raw_out
        think = raw_out['text']

        start_tok, end_tok = "<|think_start|>", "<|think_end|>"
        start, end = think.find(start_tok), think.rfind(end_tok)
        think = think[start + len(start_tok): end].replace(start_tok,
                                                           "").replace(end_tok, "").strip()

        if len(think) < 5:
            return raw_out

        messages_current = [
            messages[0],
            messages[1],
            messages[-1],
            {
                "role": "assistant",
                "content": [
                    {"type": "think", "think": think},
                    {"type": "text", "text": ""}
                ],
            }
        ]
        self.do_sample, self.top_p, cache_s, cache_p = True, 1e-5, self.do_sample, self.top_p
        cur_out = self._run_model(messages_current)
        self.do_sample, self.top_p = cache_s, cache_p

        return {
            "text": f"{start_tok}{think}{end_tok}{cur_out['text']}",
            "action": cur_out['action'],
            "messages": messages,
        }

    def _build_model_inputs(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], int]:
        clean_messages = drop_none(messages)
        is_last_assistant = messages[-1]['role'] == "assistant"
        if is_last_assistant:
            text = self.processor.apply_chat_template(
                clean_messages,
                tokenize=False,
                continue_final_message=True,
            )
        else:
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
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
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
    # Parsing helpers                                                    #
    # ------------------------------------------------------------------ #
    def _parse_action(
        self,
        text: str,
        obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parse model text into a MOVE_TO pixel command.
        Expected format: {"choice": <sensor>, "pixel": [u, v]}
        """
        if not text:
            return {"type": "CMD", "action": "STOP", "choice": ""}

        cleaned = text.strip()
        cleaned = cleaned.strip("`").strip()

        payload = self._extract_json_object(cleaned)

        choice = ""
        pixel: Optional[List[float]] = None
        if isinstance(payload, dict):
            choice = str(payload.get("choice") or payload.get(
                "sensor") or payload.get("view") or "")
            pixel = self._normalize_pixel(payload.get(
                "pixel") or payload.get("point") or payload.get("action"))
        if pixel is None:
            pixel = self._fallback_pixel(cleaned)
        if not choice:
            choice = self._default_sensor_name(obs)

        if pixel is not None:
            return {
                "type": "MOVE_TO",
                "choice": choice,
                "sensor": choice,
                "pixel": pixel,
                "action": pixel,
            }

        if "STOP" in cleaned.upper():
            return {"type": "CMD", "action": "STOP", "choice": choice}
        return {"type": "CMD", "action": "STOP", "choice": choice}

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Grab first JSON object from text if present."""
        try:
            return json.loads(text)
        except Exception:
            pass
        for match in re.finditer(r"\{[^\{\}]*\}", text):
            try:
                return json.loads(match.group(0))
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_pixel(val: Any) -> Optional[List[float]]:
        if val is None:
            return None
        if isinstance(val, dict) and "pixel" in val:
            val = val["pixel"]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                return [float(val[0]), float(val[1])]
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _fallback_pixel(text: str) -> Optional[List[float]]:
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        if len(nums) >= 2:
            try:
                return [float(nums[0]), float(nums[1])]
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _default_sensor_name(obs: Optional[Dict[str, Any]]) -> str:
        if not isinstance(obs, dict):
            return ""
        if "images" in obs and isinstance(obs["images"], dict) and obs["images"]:
            for key in ("rgb_front", "rgb_left", "rgb_right", "rgb_back"):
                if key in obs["images"]:
                    return key
            return next(iter(obs["images"].keys()), "")
        for key in ("rgb_front", "rgb_left", "rgb_right", "rgb_back"):
            if key in obs:
                return key
        for k in obs.keys():
            if isinstance(obs.get(k), (list, tuple)):
                continue
            return k
        return ""

    # ------------------------------------------------------------------ #
    # Utilities                                                          #
    # ------------------------------------------------------------------ #
    def _history_entry(
        self,
        obs: Dict[str, Any],
        sensor_poses: Optional[Dict[str, Any]],
        action: Dict[str, Any],
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
            "agent_pose": [se for se in sensor_entries if "front" in se['name']][0]['pos'],
        }

    def _format_image_entries(self, obs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if obs is None:
            return []
        entries: List[Dict[str, Any]] = []
        image_map = None
        if isinstance(obs, dict):
            if "images" in obs and isinstance(obs["images"], dict):
                image_map = obs["images"]
            else:
                image_map = obs
        if not image_map:
            return []
        for name, value in image_map.items():
            if "rgb_" not in name:
                continue
            img = _maybe_to_pil(value)
            if img is None:
                continue
            entries.append({"name": name, "image": img})
        return entries

    def _format_sensor_entries(
        self, sensor_poses: Optional[Dict[str, Any]], obs: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        source = sensor_poses or (
            obs.get("sensor_poses") if isinstance(obs, dict) else None)
        if not source:
            return []
        entries: List[Dict[str, Any]] = []
        for name, pose in source.items():
            if "rgb_" not in name:
                continue
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
                y = float(pos[2]) if len(pos) >= 3 else float(
                    pos[-1]) if pos else 0.0
                theta = float(euler[1])
                return [x, y, theta]
            if all(k in entry for k in ("x", "y", "theta")):
                return [float(entry["x"]), float(entry["y"]), float(entry["theta"])]
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            return [float(entry[0]), float(entry[1]), float(entry[2])]
        return None

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

    @staticmethod
    def _resolve_dtype(dtype: Any) -> Any:
        if torch is None:
            return dtype
        if isinstance(dtype, str):
            mapping = {
                "auto": "auto",
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            return mapping.get(dtype, dtype)
        return dtype
