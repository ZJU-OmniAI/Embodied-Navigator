from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
from ..utils import prompt_registry


@prompt_registry.register("dthink_cot_prompt_pixel")
class DThinkCotPromptPixel:
    """
    Pixel-action 版 prompt：action 输出为 {"choice": ..., "pixel": [u, v]}。
    其它结构与 dthink_cot_prompt 相同。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are D-Think, a vision-and-language navigation agent. "
        "Use history (keypoints/trajectories/recent obs) plus the current observation to decide the next step. "
        "Think step-by-step only when you need it; if the answer is obvious, keep reasoning concise. "
        "Action output must be a JSON object: {\"choice\": <sensor_name>, \"pixel\": [u, v]} where (u,v) is the 2D target point in that sensor image."
    )

    @staticmethod
    def build_message(
        steps: Sequence[Dict[str, Any]],
        instruction: str,
        maxpixel: Optional[int] = None,
        system_prompt: Optional[str] = None,
        max_keypoint_num: Optional[int] = 5
    ) -> List[Dict[str, Any]]:
        if not isinstance(steps, Sequence):
            raise TypeError("steps must be a sequence of dictionaries")

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [DThinkCotPromptPixel._text_chunk(system_prompt or DThinkCotPromptPixel.DEFAULT_SYSTEM_PROMPT)],
            },
        ]
        
        if len(instruction) > 0:
           messages.append({"role": "user", "content": [DThinkCotPromptPixel._text_chunk(instruction or "")]}) 
        
        idx = 0
        if max_keypoint_num is not None:
            stype_list = [step.get("type") for step in steps]
            keypoint_list = [i for i, stype in enumerate(stype_list) if stype == "keypoint"]
            if len(keypoint_list) > max_keypoint_num: idx = keypoint_list[-max_keypoint_num]
        
        while idx < len(steps):
            step = steps[idx] or {}
            stype = step.get("type")

            if stype == "keypoint":
                messages.extend(DThinkCotPromptPixel._build_keypoint_block(step, maxpixel))
                idx += 1
            elif stype == "trajectory":
                traj_chunk: List[Any] = []
                while idx < len(steps) and (steps[idx] or {}).get("type") == "trajectory":
                    traj_chunk.extend((steps[idx] or {}).get("agent_pos") or [])
                    idx += 1
                if traj_chunk:
                    messages.append(DThinkCotPromptPixel._build_trajectory_block(traj_chunk))
            elif stype == "recentstep":
                recent_chunk: List[Dict[str, Any]] = []
                while idx < len(steps) and (steps[idx] or {}).get("type") == "recentstep":
                    recent_chunk.append(steps[idx] or {})
                    idx += 1
                if recent_chunk:
                    messages.append(DThinkCotPromptPixel._build_recent_block(recent_chunk, maxpixel))
            elif stype == "currentobs":
                messages.append(DThinkCotPromptPixel._build_current_obs_block(step, maxpixel))
                idx += 1
            elif stype == "currentact":
                messages.append(DThinkCotPromptPixel._build_current_action(step))
                idx += 1
            else:
                idx += 1

        return messages

    # ------------------------------------------------------------------
    @staticmethod
    def _build_keypoint_block(step: Dict[str, Any], maxpixel: Optional[int]) -> List[Dict[str, Any]]:
        system_message = {
            "role": "system",
            "content": DThinkCotPromptPixel._multi_view_content(
                title="keypoint_obs:\n",
                views=step.get("images"),
                sensor_pos=step.get("sensors_pos"),
                maxpixel=maxpixel,
            ),
        }
        if len(step.get("cot", "")) < 5:
            return [system_message]
        assistant_message = {
            "role": "assistant",
            "content": [DThinkCotPromptPixel._think_chunk(step.get("cot", ""))],
        }
        return [system_message, assistant_message]

    @staticmethod
    def _build_trajectory_block(agent_positions: Sequence[Any]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [DThinkCotPromptPixel._text_chunk("trajectory:\n")]
        for entry in agent_positions:
            pos_payload = DThinkCotPromptPixel._pos_chunk(entry)
            if pos_payload:
                content.append(pos_payload)
        return {"role": "system", "content": content}

    @staticmethod
    def _build_recent_block(steps: Sequence[Dict[str, Any]], maxpixel: Optional[int]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps, start=1):
            content.append(DThinkCotPromptPixel._text_chunk(f"recent_obs {idx}:\n"))
            views = step.get("image") or step.get("images")
            sensor_pos = step.get("sensor_pos") or step.get("sensors_pos")
            content.extend(
                DThinkCotPromptPixel._multi_view_content(
                    title=None, views=views, sensor_pos=sensor_pos, maxpixel=maxpixel, pooling=2
                )
            )
        return {"role": "system", "content": content}

    @staticmethod
    def _build_current_obs_block(step: Dict[str, Any], maxpixel: Optional[int]) -> Dict[str, Any]:
        return {
            "role": "system",
            "content": DThinkCotPromptPixel._multi_view_content(
                title="current_obs:\n",
                views=step.get("images"),
                sensor_pos=step.get("sensors_pos"),
                maxpixel=maxpixel,
            ),
        }

    @staticmethod
    def _build_current_action(step: Dict[str, Any]) -> Dict[str, Any]:
        cot = step.get("cot") or ""
        action = step.get("action") or {}
        return {
            "role": "assistant",
            "content": [
                DThinkCotPromptPixel._think_chunk(cot),
                DThinkCotPromptPixel._act_pixel_chunk(action),
            ],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _multi_view_content(
        title: Optional[str],
        views: Any,
        sensor_pos: Any,
        maxpixel: Optional[int],
        pooling: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if title:
            content.append(DThinkCotPromptPixel._text_chunk(title))

        view_entries = DThinkCotPromptPixel._collect_named_views(views)
        pos_mapping = DThinkCotPromptPixel._normalize_pos_mapping(sensor_pos)

        for idx, (name, payload) in enumerate(view_entries, start=1):
            label = f"image_{name or idx}"
            content.append(DThinkCotPromptPixel._text_chunk(label))
            pose = DThinkCotPromptPixel._pos_chunk(
                pos_mapping.get(name) or pos_mapping.get(str(idx)) or pos_mapping.get(idx)
            )
            if pose:
                content.append(pose)
            image_chunk = DThinkCotPromptPixel._image_chunk(payload, maxpixel=maxpixel, pooling=pooling)
            if image_chunk:
                content.append(image_chunk)
            content.append(DThinkCotPromptPixel._text_chunk("\n"))
        return content

    @staticmethod
    def _think_chunk(cot: str) -> Dict[str, str]:
        return {"type": "think", "think": cot or ""}

    @staticmethod
    def _act_pixel_chunk(action: Dict[str, Any]) -> Dict[str, Any]:
        if not action:
            return {"type": "text", "text":""}
        choice = action.get("choice", "")
        pixel = action.get("pixel") or []
        return {"type": "text", "text": json.dumps({"choice": choice, "pixel": [round(p) for p in pixel]})}

    @staticmethod
    def _text_chunk(text: str) -> Dict[str, str]:
        return {"type": "text", "text": text or ""}

    @staticmethod
    def _image_chunk(image_entry: Any, maxpixel: Optional[int], pooling: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if image_entry is None:
            return None
        payload = dict(image_entry) if isinstance(image_entry, dict) else {"image": image_entry}
        payload.pop("name", None)
        payload.setdefault("type", "image")
        if maxpixel is not None and "max_pixels" not in payload:
            payload["max_pixels"] = maxpixel
        if pooling is not None and "pooling" not in payload:
            payload["pooling"] = pooling
        return payload

    @staticmethod
    def _pos_chunk(pos_entry: Any) -> Optional[Dict[str, Any]]:
        coords = DThinkCotPromptPixel._normalize_pose_entry(pos_entry)
        if coords is None:
            return None
        return {"type": "pos", "pos": coords}

    @staticmethod
    def _normalize_pose_entry(entry: Any) -> Optional[List[float]]:
        if entry is None:
            return None
        if isinstance(entry, dict):
            if all(k in entry for k in ("x", "y", "theta")):
                return [float(entry["x"]), float(entry["y"]), float(entry["theta"])]
            for key in ("pos", "pos_values", "value"):
                if key in entry:
                    return DThinkCotPromptPixel._normalize_pose_entry(entry[key])
        elif isinstance(entry, (list, tuple)):
            if len(entry) == 3:
                try:
                    return [float(entry[0]), float(entry[1]), float(entry[2])]
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _collect_named_views(views: Any) -> List[Tuple[str, Any]]:
        if not views:
            return []
        if isinstance(views, dict):
            if "name" in views and any(k in views for k in ("image", "image_url", "video")):
                name = views.get("name") or "view"
                payload = dict(views)
                payload.pop("name", None)
                return [(name, payload)]
            return [(str(name), data) for name, data in views.items()]
        if isinstance(views, (list, tuple)):
            normalized: List[Tuple[str, Any]] = []
            for idx, item in enumerate(views, start=1):
                if isinstance(item, dict) and "name" in item:
                    entry = dict(item)
                    name = str(entry.pop("name"))
                    normalized.append((name, entry))
                else:
                    normalized.append((str(idx), item))
            return normalized
        return [("1", views)]

    @staticmethod
    def _normalize_pos_mapping(pos_field: Any) -> Dict[str, Any]:
        if not pos_field:
            return {}
        if isinstance(pos_field, dict):
            if "name" in pos_field and any(k in pos_field for k in ("x", "y", "theta", "pos", "pos_values")):
                name = str(pos_field["name"])
                entry = dict(pos_field)
                entry.pop("name", None)
                return {name: entry}
            return {str(name): value for name, value in pos_field.items()}
        if isinstance(pos_field, (list, tuple)):
            mapping: Dict[str, Any] = {}
            for idx, entry in enumerate(pos_field, start=1):
                if isinstance(entry, dict) and "name" in entry:
                    tmp = dict(entry)
                    name = str(tmp.pop("name"))
                    mapping[name] = tmp
                else:
                    mapping[str(idx)] = entry
            return mapping
        return {"1": pos_field}
