from __future__ import annotations

import base64
import io
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from PIL import Image

VIEW_ORDER = ["front", "left", "back", "right"]
_SUMMARY_INDEX_RE = re.compile(r"\b(step|image|img|frame)\s*[-#:]?\s*\d+\b", re.IGNORECASE)


@dataclass
class COTStepSpec:
    step_idx: int
    view_idx: Optional[int] = None
    u: Optional[float] = None
    v: Optional[float] = None
    gt_action: Optional[str] = None
def _encode_image_pil(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_json_data(text: str) -> Any:
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object or array found.")
    return json.loads(match.group(0))


def _response_text(resp: Any) -> str:
    message = resp.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str):
        return reasoning_content
    return ""


def _check_analysis_quality(text: str) -> Tuple[bool, str]:
    if "Analyze the current observations" not in text:
        return False, "Incorrect format"

    forbidden = [
        "red circle",
        "red point",
        "marked spot",
        "pixel",
        "coordinate",
        "proportion",
        "that location",
        "that spot",
        "the target position",
    ]
    lowered = text.lower()
    for token in forbidden:
        if token in lowered:
            return False, f"Forbidden content: {token}"

    if "```" in text or "**" in text:
        return False, "Markdown detected"

    if len(text) < 100:
        return False, "Analysis too short"

    if "I am an AI" in text or "cannot" in text or "sorry" in text:
        return False, "Refusal detected"

    return True, "OK"


def _check_infer_quality(text: str, gt_action: str) -> Tuple[bool, str]:
    if '"Infer the next step":' not in text:
        return False, "Missing section"

    forbidden = [
        "red circle",
        "red point",
        "marked spot",
        "pixel",
        "coordinate",
        "proportion",
        "that location",
        "that spot",
        "the target position",
    ]
    lowered = text.lower()
    for token in forbidden:
        if token in lowered:
            return False, f"Forbidden content: {token}"

    if gt_action == "stop":
        if "further movement is needed" not in lowered and "arriv" not in lowered and "complete" not in lowered:
            return False, "Stop but no arrival declared"
    else:
        if "further movement is needed" in lowered:
            return False, "Incorrect arrival claim"

    if "```" in text or "**" in text:
        return False, "Markdown detected"

    if len(text) < 100:
        return False, "Infer too short"

    if "I am an AI" in text or "cannot" in text or "sorry" in text:
        return False, "Refusal detected"

    return True, "OK"


def _check_summary_quality(text: str) -> Tuple[bool, str]:
    if text is None or not str(text).strip():
        return False, "Empty summary"
    if _SUMMARY_INDEX_RE.search(text):
        return False, "Contains step/image index"
    return True, "OK"


def _normalize_summary_item(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if isinstance(item.get("summary"), str):
            return item["summary"]
        if isinstance(item.get("text"), str):
            return item["text"]
    return None


def _parse_step_specs(step_specs: Sequence[Union[COTStepSpec, Mapping[str, Any], Sequence[Any]]]) -> List[COTStepSpec]:
    parsed: List[COTStepSpec] = []
    for raw in step_specs:
        if isinstance(raw, COTStepSpec):
            parsed.append(raw)
            continue

        if isinstance(raw, Mapping):
            step_idx = int(raw.get("step", raw.get("step_idx")))
            view_idx = raw.get("view", raw.get("view_idx"))
            u = raw.get("u")
            v = raw.get("v")
            gt_action = raw.get("gt_action")
            parsed.append(
                COTStepSpec(
                    step_idx=step_idx,
                    view_idx=int(view_idx) if view_idx is not None else None,
                    u=float(u) if u is not None else None,
                    v=float(v) if v is not None else None,
                    gt_action=str(gt_action) if gt_action is not None else None,
                )
            )
            continue

        if isinstance(raw, Sequence) and len(raw) >= 4:
            step_idx = int(raw[0])
            view_idx = raw[1]
            u = raw[2]
            v = raw[3]
            gt_action = raw[4] if len(raw) >= 5 else None
            parsed.append(
                COTStepSpec(
                    step_idx=step_idx,
                    view_idx=int(view_idx) if view_idx is not None else None,
                    u=float(u) if u is not None else None,
                    v=float(v) if v is not None else None,
                    gt_action=str(gt_action) if gt_action is not None else None,
                )
            )
            continue

        raise ValueError(f"Unsupported step spec format: {raw!r}")

    return parsed


class COTGenerator:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_retries: int = 5,
        temperature: float = 0.05,
        top_p: float = 0.8,
        enable_thinking: Optional[bool] = True,
    ) -> None:
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking

    def _chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        flag = self.enable_thinking if enable_thinking is None else enable_thinking
        if flag is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(flag)}}

        return self.client.chat.completions.create(**kwargs)

    def generate_summary(
        self,
        *,
        task: str,
        all_images: Sequence[Image.Image],
        indices: Sequence[int],
        extra_views_by_step: Optional[Mapping[int, int]] = None,
        step_to_image_index: Optional[Mapping[int, int]] = None,
    ) -> List[str]:
        if len(indices) <= 1:
            return []

        extra_views_by_step = extra_views_by_step or {}
        encoded_images = [_encode_image_pil(img) for img in all_images]
        target_keys = [f"0-{k}" for k in indices[1:]]
        segment_list_text = ", ".join(target_keys)
        final_index = indices[-1]

        extra_items: List[str] = []
        for step_idx, view_idx in sorted(extra_views_by_step.items()):
            if view_idx is None or not (0 <= view_idx < len(VIEW_ORDER)):
                continue
            direction = VIEW_ORDER[view_idx]
            if step_to_image_index is not None:
                if step_idx not in step_to_image_index:
                    raise KeyError(f"step_to_image_index missing step {step_idx}")
                extra_img_idx = step_to_image_index[step_idx] + 1
                extra_items.append(
                    f"image {extra_img_idx}: extra image is {direction} view (turn {direction} in place)"
                )
            else:
                extra_items.append(
                    f"step {step_idx}: extra image is {direction} view (turn {direction} in place)"
                )
        extra_view_text = ", ".join(extra_items) if extra_items else "None"

        system_message = f"""
You are an expert at summarizing navigation trajectories.

You will receive:
- A navigation instruction (task)
- A sequence of egocentric images (front view per step, with optional extra target views)
- A list of target image indices k defining summary ranges [0 -> k]
- Some steps include an extra target-view image in addition to the front view

Your job for EACH k:
- Summarize the trajectory from image 0 -> k in ONE continuous paragraph.
- Use BOTH the task (to identify sub-goals) AND the images (to describe what happened).
- Explicitly state which part of the task has been completed so far.
- At the final index k={final_index}, explicitly state that the entire task has been completed.
- For all non-final ranges, do NOT say you have arrived or completed the task.
- Do NOT mention any parts of the task that are not yet completed within 0 -> k; no future steps, no pending parts.
- DO NOT invent extra segments. Only summarize exactly the ranges given.
- Never mention the index/number of any specific image or any step.

OUTPUT FORMAT (very important):
Return ONLY a JSON object where each key is a range string from the list below (e.g., "0-5").
Each value must be the summary string for that range.
Do NOT output anything outside the JSON object.
""".strip()

        user_prompt = f"""
Task:
{task}

Summary ranges (use EXACTLY these):
{segment_list_text}

Image indexing:
- Indices refer to the attached image order (0-based).
- By default, each step contributes one front-view image.
- For the listed extra-view images, an additional image is inserted immediately after that step's front view.
- The extra image is a turn-in-place view (no movement): left/back/right as specified.
- Only the listed image indices are extra views; all other images are front views.
- Do NOT describe the extra view as a separate action; treat it as another view at the same step.
- Extra views (extra image index and direction): {extra_view_text}

For EACH range [0 -> k]:
- Write ONE paragraph in first person.
- Describe the trajectory and which part of the task is completed by that point.
- For the final k={final_index}, explicitly state the task is fully completed.
- For k != {final_index}, do NOT claim you finish the task.
- Do NOT mention any uncompleted or future parts beyond 0 -> k.

Output format:
- Return ONLY a JSON object.
- Keys must be the range strings (e.g., "0-5").
- Length must equal the number of ranges in {segment_list_text}.

Images follow below.
""".strip()

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        user_content = messages[1]["content"]
        for enc in encoded_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{enc}"},
            })

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._chat(messages=messages, enable_thinking=True)
                text = _response_text(resp).strip()
                data = _extract_json_data(text)

                results: List[Any] = []
                if isinstance(data, dict):
                    for key in target_keys:
                        val = data.get(key)
                        if val is None:
                            val = data.get(f"[{key}]")
                        if val is None:
                            val = data.get(key.replace("-", "->"))
                        if val is not None:
                            results.append(val)
                elif isinstance(data, list):
                    results = data

                if len(results) != len(indices) - 1:
                    raise ValueError(
                        f"Summary mismatch: expected {len(indices) - 1}, got {len(results)}"
                    )

                normalized: List[str] = []
                for idx, item in enumerate(results):
                    summary_text = _normalize_summary_item(item)
                    if summary_text is None:
                        raise ValueError(f"Summary {idx} is not a string")
                    ok, reason = _check_summary_quality(summary_text)
                    if not ok:
                        raise ValueError(f"Summary {idx} invalid: {reason}")
                    normalized.append(summary_text)
                return normalized
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) + random.random())

        raise RuntimeError(f"Failed to generate summary: {last_error}")

    def generate_analysis(
        self,
        *,
        task: str,
        current_observations: Sequence[Image.Image],
        summary: str,
    ) -> str:
        if len(current_observations) != 4:
            raise ValueError("current_observations must contain exactly 4 images.")

        encoded_current = [_encode_image_pil(img) for img in current_observations]

        system_message = """
You are a controlled reasoning module for generating the 'Analyze the current observations' section.

- Describe the indoor scene layout, including open paths, obstacles, landmarks, doorways, and exact floor locations relative to visible objects.
- The attached views are: front, left, back, right (in that order). Refer to views by direction only.
- Focus on details relevant to navigation.
- Only describe objects/structures that are visible in the current views; if unsure, omit the detail.
- Do NOT mention targets, goals, history, or next steps.
- Output FORMAT: "\"Analyze the current observations\": [your detailed description]"
- No extra text, no markdown.
- Never set content to null. Never use tool_calls.
""".strip()

        user_prompt = f"""
Overall Task: {task}
History Summary (for context): {summary}

Current Observations (input details):
The attached images are the current views in order: front, left, back, right.
Analyze visible elements across these views for a cohesive scene understanding.
You don’t need to describe all the images; only mention information that is helpful for the upcoming navigation.

Generate ONLY the "Analyze the current observations" section as specified.
Output FORMAT: "\"Analyze the current observations\": [your detailed description]"
""".strip()

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        user_content = messages[1]["content"]
        for enc in encoded_current:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}})

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._chat(messages=messages, enable_thinking=True)
                text = _response_text(resp).strip()
                ok, reason = _check_analysis_quality(text)
                if ok:
                    return text
                raise ValueError(reason)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) + random.random())

        raise RuntimeError(f"Failed to generate analysis: {last_error}")

    def generate_infer(
        self,
        *,
        task: str,
        summary: str,
        analysis: str,
        target_view_idx: int,
        target_guidance: str,
        gt_action: str,
        current_observations: Sequence[Image.Image],
    ) -> str:
        if len(current_observations) != 4:
            raise ValueError("current_observations must contain exactly 4 images.")

        if gt_action == "stop":
            view_idx = 0
            encoded_current: List[str] = []
        else:
            if not (0 <= target_view_idx < len(VIEW_ORDER)):
                raise ValueError(f"target_view_idx out of range: {target_view_idx}")
            view_idx = target_view_idx
            encoded_current = [_encode_image_pil(current_observations[view_idx])]

        view_name = VIEW_ORDER[view_idx]

        system_message = """
You are a controlled reasoning module for generating the 'Infer the next step' section.

=== IRON RULE ABOUT ARRIVAL ===
IF gt_action == "stop":
- Infer: A paragraph declaring arrival, e.g., "Based on the task and current position, I have arrived at the goal destination. No further movement is needed."

ELSE:
- Infer: Reason step-by-step how the next move advances the task (without leaking target coordinates), ending with one complete action sentence specifying a physical movement and visible destination, e.g., "Therefore, walk forward to the area in front of the gray sofa."
- If you cannot determine a clear reason, you may omit the reason and directly provide the next action sentence.
- Any stated reason MUST be grounded only in the task, history summary, and current observations; do NOT use Target Guidance as a reason.
- If the instruction for this step includes a left or right turn, explicitly use that as one reason for the chosen view/region (e.g., "turn left, so choose the left view" or "turn left, so choose a left-leaning area in the front view").
- The final action sentence MUST explicitly name the target view (front/left/back/right) AND an object-defined region within that view (e.g., "between the two sofas", "through the open door"). It MUST include the exact target-view phrase shown in the user prompt (e.g., "front view"), and MUST NOT mention any other view labels as a viewpoint.
- The reasoning and action MUST align with the Target Guidance's view and region; do NOT cite or leak the guidance or coordinates, and do not introduce contradictions.
- Only describe objects/structures that are visible in the attached target view; if unsure, omit the detail.
- NO rotations or view changes like "Turn left." Actions must involve forward movement to a described location.
- FORBIDDEN: Any mention of arrival, stop, or completion unless gt_action == "stop".
- FORBIDDEN phrases: red circle, point, marked spot, pixel, coordinate, proportion, that location, the spot.

Output FORMAT (single section):
"\"Infer the next step\": [full reasoning paragraph ending with action]"

No extra text, no markdown.
Never set content to null. Never use tool_calls.
""".strip()

        user_prompt = f"""
Overall Task: {task}

History Summary: {summary}

Current Analysis: {analysis}

Current Observations:
The attached image is the target view: {view_name} view.

Target Guidance (view + coordinates, do not mention or leak): {view_name} view: {target_guidance}
Use this guidance to locate the precise region in the attached view, but do NOT mention coordinates or the guidance itself.
In your final action sentence, include the exact phrase "{view_name} view".
Do not mention that the target view was given to you in advance; instead, treat it as an inference result and state it at the end.
All reasoning and the final action MUST correspond to this target, but you can Not use the target guidance as a reason.
If the instruction indicates a left/right turn for this step, include that as a reason for selecting the view or region.


gt_action (controls arrival): {gt_action}
Follow the IRON RULE strictly based on this.

Generate ONLY the infer section as specified in the system instructions.
""".strip()

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        user_content = messages[1]["content"]
        for enc in encoded_current:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}})

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._chat(messages=messages, enable_thinking=True)
                text = _response_text(resp).strip()
                ok, reason = _check_infer_quality(text, gt_action=gt_action)
                if ok:
                    return text
                raise ValueError(reason)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) + random.random())

        raise RuntimeError(f"Failed to generate infer: {last_error}")

    def auto_generate_cot(
        self,
        *,
        task: str,
        summary: str,
        current_observations: Sequence[Image.Image],
        target_view_idx: int,
        target_guidance: str,
        gt_action: str,
    ) -> str:
        analysis = self.generate_analysis(
            task=task,
            current_observations=current_observations,
            summary=summary,
        )
        infer = self.generate_infer(
            task=task,
            summary=summary,
            analysis=analysis,
            target_view_idx=target_view_idx,
            target_guidance=target_guidance,
            gt_action=gt_action,
            current_observations=current_observations,
        )
        return f"summarize the history: {summary}\n{analysis}\n{infer}"

    def polish_cot(self, cot: str) -> str:
        system_prompt = """You are an expert at rewriting chain-of-thought reasoning.

Given a three-stage chain-of-thought in English, rewrite it into a single, coherent paragraph of English reasoning.

Requirements:
- Keep the original three-step logic (recall the task and the history of actions -> describe the current observations -> infer the next step), but do NOT include any stage titles.
- Preserve the necessary spatial information (such as front, back, left, right, object locations, and whether different paths are traversable).
- In the final action description, explicitly preserve which view the action occurs in (front/left/back/right).
- When describing the environment, strictly avoid mentioning specific image numbers (such as Image 0, Image 1, etc.). Instead, please directly use the corresponding viewing directions (such as 'Front', 'Left', 'Back', 'Right') to refer to the observed views.
- Output only the optimized English reasoning as ONE single paragraph, without any additional explanations.
- Remove repeated or redundant expressions
- Never miss any important information.
"""
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cot},
                    ],
                    temperature=0.04,
                    top_p=0.8,
                    enable_thinking=False,
                )
                text = _response_text(resp).strip()
                if not text:
                    raise ValueError("Empty polish result")
                return text
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) + random.random())

        raise RuntimeError(f"Failed to polish COT: {last_error}")


def generate_episode_cots(
    *,
    generator: COTGenerator,
    task: str,
    all_images: Sequence[Sequence[Image.Image]],
    cot_step_specs: Sequence[Union[COTStepSpec, Mapping[str, Any], Sequence[Any]]],
    image_size: int = 280,
    initial_summary: str = "This is the starting position. No actions have been taken yet.",
    polish: bool = True,
) -> Dict[str, Any]:
    specs = _parse_step_specs(cot_step_specs)
    if not specs:
        raise ValueError("cot_step_specs is empty.")

    total_steps = len(all_images)
    for spec in specs:
        if not (0 <= spec.step_idx < total_steps):
            raise ValueError(f"step_idx out of range: {spec.step_idx}")

    extra_views_by_step: Dict[int, int] = {}
    for spec in specs:
        if spec.view_idx in (1, 2, 3):
            extra_views_by_step[spec.step_idx] = spec.view_idx

    all_history_images: List[Image.Image] = []
    step_to_image_index: Dict[int, int] = {}
    image_index = -1
    for step_idx, step_views in enumerate(all_images):
        if len(step_views) != 4:
            raise ValueError(f"all_images[{step_idx}] must contain 4 views.")
        all_history_images.append(step_views[0])
        image_index += 1
        step_to_image_index[step_idx] = image_index
        extra_view = extra_views_by_step.get(step_idx)
        if extra_view in (1, 2, 3):
            all_history_images.append(step_views[extra_view])
            image_index += 1

    thinking_steps = [spec.step_idx for spec in specs]
    thinking_image_indices = [step_to_image_index[idx] for idx in thinking_steps]

    summary_parts = [initial_summary]
    summary_parts.extend(
        generator.generate_summary(
            task=task,
            all_images=all_history_images,
            indices=thinking_image_indices,
            extra_views_by_step=extra_views_by_step,
            step_to_image_index=step_to_image_index,
        )
    )

    raw_cots: List[str] = []
    for i, spec in enumerate(specs):
        step_idx = spec.step_idx
        current_observations = all_images[step_idx]

        gt_action = spec.gt_action
        if gt_action is None:
            gt_action = "stop" if i == len(specs) - 1 else "did not stop"

        if gt_action == "stop":
            target_view_idx = 0
            target_guidance = "arrived at the goal"
        else:
            if spec.view_idx is None or spec.u is None or spec.v is None:
                raise ValueError(
                    f"Missing view/u/v for non-stop step spec at index {i}: {spec}"
                )
            if not (0 <= spec.view_idx < len(VIEW_ORDER)):
                raise ValueError(f"Invalid view_idx: {spec.view_idx}")
            target_view_idx = spec.view_idx
            target_guidance = f"proportion ({spec.u / image_size:.4f}, {spec.v / image_size:.4f})"

        cot = generator.auto_generate_cot(
            task=task,
            summary=summary_parts[i],
            current_observations=current_observations,
            target_view_idx=target_view_idx,
            target_guidance=target_guidance,
            gt_action=gt_action,
        )
        raw_cots.append(cot)

    final_cots = [generator.polish_cot(cot) for cot in raw_cots] if polish else raw_cots

    step_results = []
    for spec, raw_cot, cot in zip(specs, raw_cots, final_cots):
        step_results.append(
            {
                "step_index": spec.step_idx,
                "raw_cot": raw_cot,
                "cot": cot,
            }
        )

    return {
        "summaries": summary_parts,
        "step_results": step_results,
    }
