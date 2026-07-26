import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# ----------------------------- #
# Reward scales (default = 1.0)
# ----------------------------- #
SUCCESS_REWARD_SCALE = float(os.environ.get("SUCCESS_REWARD_SCALE", "3.0"))
TRAJECTORY_SIMILARITY_REWARD_SCALE = float(os.environ.get("TRAJECTORY_SIMILARITY_REWARD_SCALE", "1.0"))
REASONING_DENSITY_REWARD_SCALE = float(os.environ.get("REASONING_DENSITY_REWARD_SCALE", "0.0"))

FORMAT_REWARD_SCALE = float(os.environ.get("FORMAT_REWARD_SCALE", "0.2"))
COLLISION_REWARD_SCALE = float(os.environ.get("COLLISION_REWARD_SCALE", "0.2"))
TARGET_APPROACH_REWARD_SCALE = float(os.environ.get("TARGET_APPROACH_REWARD_SCALE", "0.5"))
REASONING_REWARD_SCALE = float(os.environ.get("REASONING_REWARD_SCALE", "0.5"))
STOP_REWARD_SCALE = float(os.environ.get("STOP_REWARD_SCALE", "1.2"))

_THINK_RE = re.compile(r"<\|think_start\|>(.*?)<\|think_end\|>", flags=re.DOTALL)
_FORMAT_ONLY_THINK_RE = re.compile(r"^<\|think_start\|>.*?<\|think_end\|>\s*(?:<\|im_end\|>)?\s*$", flags=re.DOTALL)
_FORMAT_THINK_JSON_RE = re.compile(r"^<\|think_start\|>.*?<\|think_end\|>\s*(\{.*\})\s*(?:<\|im_end\|>)?\s*$", flags=re.DOTALL)
_REASONING_DENSITY_X1 = 0.4
_REASONING_DENSITY_X2 = 0.6

REWARD_DEBUG_LOG_ENABLED = str(os.environ.get("REWARD_DEBUG_LOG", "1")).lower() not in {"0", "false", "off", "no"}
REWARD_DEBUG_LOG_DIR = Path(os.environ.get("REWARD_DEBUG_LOG_DIR", "debug/reward_logs"))
REWARD_DEBUG_MAX_STR = int(os.environ.get("REWARD_DEBUG_MAX_STR", "800"))
REWARD_DEBUG_MAX_ITEMS = int(os.environ.get("REWARD_DEBUG_MAX_ITEMS", "20"))


def set_reward_log_dir(path: Path) -> None:
    global REWARD_DEBUG_LOG_DIR
    REWARD_DEBUG_LOG_DIR = path


def _json_safe(obj: Any, depth: int = 0):
    if depth > 4:
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        if len(obj) <= REWARD_DEBUG_MAX_STR:
            return obj
        return obj[:REWARD_DEBUG_MAX_STR] + "...<truncated>"
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _json_safe(obj.item(), depth + 1)
        shape = list(obj.shape)
        flat = obj.reshape(-1)
        limit = min(len(flat), REWARD_DEBUG_MAX_ITEMS)
        return {
            "_type": "ndarray",
            "shape": shape,
            "dtype": str(obj.dtype),
            "sample": [_json_safe(x.item(), depth + 1) for x in flat[:limit]],
        }
    if torch.is_tensor(obj):
        shape = list(obj.shape)
        if obj.ndim == 0:
            return _json_safe(obj.item(), depth + 1)
        flat = obj.detach().reshape(-1)
        limit = min(flat.numel(), REWARD_DEBUG_MAX_ITEMS)
        return {
            "_type": "tensor",
            "shape": shape,
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            "sample": [_json_safe(x.item(), depth + 1) for x in flat[:limit]],
        }
    if isinstance(obj, (list, tuple)):
        limit = min(len(obj), REWARD_DEBUG_MAX_ITEMS)
        out = [_json_safe(obj[i], depth + 1) for i in range(limit)]
        if len(obj) > limit:
            out.append(f"...<{len(obj) - limit} more>")
        return out
    if isinstance(obj, dict):
        out = {}
        count = 0
        for key, val in obj.items():
            if count >= REWARD_DEBUG_MAX_ITEMS:
                out["..."] = f"<{len(obj) - REWARD_DEBUG_MAX_ITEMS} more keys>"
                break
            out[str(key)] = _json_safe(val, depth + 1)
            count += 1
        return out
    return repr(obj)


def _debug_log(name: str, payload: Dict[str, Any]) -> None:
    if not REWARD_DEBUG_LOG_ENABLED:
        return
    rank = int(os.environ.get("RANK", "0"))
    path = REWARD_DEBUG_LOG_DIR / f"{name}_rank{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "rank": rank,
        "pid": os.getpid(),
        "payload": _json_safe(payload),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _extract_think_spans(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return [span.strip() for span in _THINK_RE.findall(text)]


def _has_reasoning(text: str, min_len: int = 5) -> bool:
    for span in _extract_think_spans(text):
        if len(span) > min_len:
            return True
    return False


def _completion_reasoning_ratio(completion_text: str) -> float:
    steps = [line for line in completion_text.split("\n") if line.strip()]
    if not steps:
        return 0.0
    if len(steps) <= 2:
        return 0.0
    steps = steps[1:-1]
    reasoning_steps = sum(1 for step in steps if _has_reasoning(step))
    return float(reasoning_steps) / float(len(steps))


def _piecewise_reasoning_density_reward(x: float) -> float:
    if x <= 0.0:
        return 1.0
    if x <= _REASONING_DENSITY_X1:
        # (0,1.0) -> (0.4,0.6)
        return 1.0 - x
    if x <= _REASONING_DENSITY_X2:
        # (0.4,0.6) -> (0.6,0.0)
        return 0.6 - 3.0 * (x - _REASONING_DENSITY_X1)
    return 0.0


def _is_correct_trajectory(metrics: Any) -> bool:
    if not isinstance(metrics, dict):
        return False
    success = _safe_float(metrics.get("success", 0.0), 0.0) > 0.0
    oracle_success = _safe_float(metrics.get("oracle_success", 0.0), 0.0) > 0.0
    return success


def _sigmoid_signed(x: float) -> float:
    # 2 * sigmoid(x) - 1, maps R -> (-1, 1)
    x = max(min(x, 60.0), -60.0)
    return (2.0 / (1.0 + math.exp(-x))) - 1.0


def _pearson_corr(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    a_std = float(np.std(a_arr))
    b_std = float(np.std(b_arr))
    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0
    c = float(np.corrcoef(a_arr, b_arr)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return max(min(c, 1.0), -1.0)


def _is_stop_action(action: Dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    action_type = str(action.get("type", "")).upper()
    action_cmd = str(action.get("action", "")).upper()
    return action_type == "CMD" and action_cmd == "STOP"


def _is_stop_near_oracle_goal(action: Dict[str, Any], env: Any, threshold: float = 2.5) -> bool:
    if not _is_stop_action(action):
        return False
    metrics = getattr(env, "metrics", None)
    if not isinstance(metrics, dict):
        return False
    one = _safe_float(metrics.get("oracle_navigation_error", float("inf")), float("inf"))
    return math.isfinite(one) and one < threshold


def _action_to_world_point(action: Dict[str, Any], env: Any) -> np.ndarray:
    if not isinstance(action, dict):
        return None
    if action.get("type") != "MOVE_TO":
        return None
    pixel = action.get("pixel") or action.get("action")
    if not isinstance(pixel, (list, tuple)) or len(pixel) < 2:
        return None

    sensor_name = str(action.get("sensor") or action.get("choice") or "depth")
    u = min(max(int(round(float(pixel[0]))), 0), 279)
    v = min(max(int(round(float(pixel[1]))), 0), 279)
    if not hasattr(env, "sg"):
        return None
    try:
        world = env.sg.pixel_to_world(u, v, sensor_name=sensor_name, convention="z")
    except Exception:
        fallback_sensor = sensor_name.replace("rgb", "depth")
        try:
            world = env.sg.pixel_to_world(u, v, sensor_name=fallback_sensor, convention="z")
        except Exception:
            return None
    try:
        return np.asarray(world, dtype=np.float32).reshape(-1)[:3]
    except Exception:
        return None


def _expert_distance(env: Any, point_3d: np.ndarray = None) -> float:
    try:
        info = env.expert_dist(point_3d) if point_3d is not None else env.expert_dist()
        dist = _safe_float(info.get("expert_distance", float("inf")), float("inf"))
        if not math.isfinite(dist):
            return float("inf")
        return dist
    except Exception:
        return float("inf")


def _target_approach_delta(action: Dict[str, Any], env: Any) -> float:
    current_dist = _expert_distance(env, None)
    if not math.isfinite(current_dist):
        return 0.0
    target_point = _action_to_world_point(action, env)
    target_dist = _expert_distance(env, target_point) if target_point is not None else current_dist
    if not math.isfinite(target_dist):
        target_dist = current_dist
    return current_dist - target_dist


def _target_approach_score(action: Dict[str, Any], env: Any) -> float:
    return _sigmoid_signed(_target_approach_delta(action, env))


def success_reward(
    prompts: List[str],
    final_metrics: List[Dict[str, Any]] = None,
    **kwargs,
) -> List[float]:
    metrics_list = final_metrics or [{} for _ in prompts]
    rewards: List[float] = []
    details: List[Dict[str, Any]] = []
    for metrics in metrics_list:
        metrics = metrics or {}
        success = _safe_float(metrics.get("success", 0.0), 0.0) > 0.0
        oracle_success = _safe_float(metrics.get("oracle_success", 0.0), 0.0) > 0.0
        if success:
            score = 1.0
        elif oracle_success:
            score = 0.5
        else:
            score = 0.0
        rewards.append(score * SUCCESS_REWARD_SCALE)
        details.append(
            {
                "success": success,
                "oracle_success": oracle_success,
                "base_score": score,
                "scaled_score": rewards[-1],
                "oracle_navigation_error": _safe_float(metrics.get("oracle_navigation_error", float("inf")), float("inf")),
            }
        )
    _debug_log(
        "success_reward",
        {
            "num_samples": len(prompts),
            "scale": SUCCESS_REWARD_SCALE,
            "details": details,
        },
    )
    return rewards


def trajectory_similarity_reward(
    prompts: List[str],
    final_metrics: List[Dict[str, Any]] = None,
    **kwargs,
) -> List[float]:
    metrics_list = final_metrics or [{} for _ in prompts]
    rewards: List[float] = []
    details: List[Dict[str, Any]] = []
    for metrics in metrics_list:
        metrics = metrics or {}
        spl = _safe_float(metrics.get("spl", 0.0), 0.0)
        scaled = spl * TRAJECTORY_SIMILARITY_REWARD_SCALE
        rewards.append(scaled)
        details.append({"spl": spl, "scaled_score": scaled})
    _debug_log(
        "trajectory_similarity_reward",
        {
            "num_samples": len(prompts),
            "scale": TRAJECTORY_SIMILARITY_REWARD_SCALE,
            "details": details,
        },
    )
    return rewards


def reasoning_density_reward(
    prompts: List[str],
    completions: List[str],
    final_metrics: List[Dict[str, Any]] = None,
    **kwargs,
) -> List[float]:
    metrics_list = final_metrics or [{} for _ in prompts]
    rewards: List[float] = []
    details: List[Dict[str, Any]] = []
    for idx, completion_text in enumerate(completions):
        metrics = metrics_list[idx] if idx < len(metrics_list) else {}
        ratio = _completion_reasoning_ratio(completion_text or "")
        piecewise_score = _piecewise_reasoning_density_reward(ratio)
        score = piecewise_score
        trajectory_success = _is_correct_trajectory(metrics)
        if not trajectory_success:
            score = min(score, 1.0 - _REASONING_DENSITY_X1)
        scaled = score * REASONING_DENSITY_REWARD_SCALE
        rewards.append(scaled)
        details.append(
            {
                "reasoning_ratio": ratio,
                "piecewise_score": piecewise_score,
                "final_score": score,
                "scaled_score": scaled,
                "trajectory_success": trajectory_success,
            }
        )
    _debug_log(
        "reasoning_density_reward",
        {
            "num_samples": len(prompts),
            "scale": REASONING_DENSITY_REWARD_SCALE,
            "x1": _REASONING_DENSITY_X1,
            "x2": _REASONING_DENSITY_X2,
            "details": details,
        },
    )
    return rewards


def _is_valid_format_choice(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value in {"rgb_front", "rgb_back", "rgb_left", "rgb_right"}


def _is_valid_format_pixel(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except Exception:
        return False
    return True


def _extract_action_pixel(action: Any) -> Any:
    if not isinstance(action, dict):
        return None
    pixel = action.get("pixel")
    if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
        return pixel
    fallback = action.get("action")
    if isinstance(fallback, (list, tuple)) and len(fallback) == 2:
        return fallback
    return None


def _format_reward_single(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {"valid": False, "reason": "not_string"}
    if _FORMAT_ONLY_THINK_RE.match(text):
        return {"valid": True, "mode": "think_only"}
    match = _FORMAT_THINK_JSON_RE.match(text)
    if not match:
        return {"valid": False, "reason": "pattern_mismatch"}
    payload = match.group(1)
    try:
        obj = json.loads(payload)
    except Exception:
        return {"valid": False, "reason": "json_parse_failed"}
    if not isinstance(obj, dict):
        return {"valid": False, "reason": "json_not_object"}
    if set(obj.keys()) != {"choice", "pixel"}:
        return {"valid": False, "reason": "json_keys_invalid", "keys": list(obj.keys())}
    if not _is_valid_format_choice(obj.get("choice")):
        return {"valid": False, "reason": "choice_invalid", "choice": obj.get("choice")}
    if not _is_valid_format_pixel(obj.get("pixel")):
        return {"valid": False, "reason": "pixel_invalid", "pixel": obj.get("pixel")}
    return {"valid": True, "mode": "think_json", "choice": obj.get("choice"), "pixel": obj.get("pixel")}


def format_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    rewards: List[float] = []
    details: List[Dict[str, Any]] = []
    for completion_text in completions:
        result = _format_reward_single(completion_text or "")
        score = 1.0 if result.get("valid") else 0.0
        scaled = score * FORMAT_REWARD_SCALE
        rewards.append(scaled)
        details.append({**result, "base_score": score, "scaled_score": scaled})
    _debug_log(
        "format_reward",
        {
            "num_samples": len(prompts),
            "scale": FORMAT_REWARD_SCALE,
            "details": details,
        },
    )
    return rewards


def format_step_reward(
    action: Dict[str, Any],
    state: Any,
    model_out: Dict[str, Any],
    env: Any,
    **kwargs,
) -> float:
    step_id = kwargs.get("step_id", None)
    candidate_idx = kwargs.get("candidate_idx", None)
    text = ""
    if isinstance(model_out, dict):
        text = str(model_out.get("text", ""))
    result = _format_reward_single(text)
    score = 1.0 if result.get("valid") else 0.0

    pixel_penalty = 1.0
    action_pixel = _extract_action_pixel(action)
    action_pixel_float = None
    pixel_out_of_bound = False
    if action_pixel is not None:
        try:
            u = float(action_pixel[0])
            v = float(action_pixel[1])
            action_pixel_float = [u, v]
            pixel_out_of_bound = (u < 0.0) or (u > 279.0) or (v < 0.0) or (v > 279.0)
            if pixel_out_of_bound:
                pixel_penalty = 0.5
        except Exception:
            pass
    final_score = score * pixel_penalty
    scaled = final_score * FORMAT_REWARD_SCALE
    
    _debug_log(
        "format_step_reward",
        {
            "step_id": step_id,
            "candidate_idx": candidate_idx,
            "action": action,
            "valid": result.get("valid"),
            "mode": result.get("mode"),
            "reason": result.get("reason"),
            "base_score": score,
            "scaled_score": scaled,
            "action_pixel": action_pixel_float,
            "pixel_out_of_bound_279": pixel_out_of_bound,
            "pixel_penalty_factor": pixel_penalty,
            "final_score": final_score,
        },
    )
    return scaled


def collision_reward(
    action: Dict[str, Any],
    state: Any,
    model_out: Dict[str, Any],
    env: Any,
    **kwargs,
) -> float:
    step_id = kwargs.get("step_id", None)
    candidate_idx = kwargs.get("candidate_idx", None)
    stop_override = _is_stop_near_oracle_goal(action, env)
    if stop_override:
        result = 1.0 * COLLISION_REWARD_SCALE
        _debug_log(
            "collision_reward",
            {
                "step_id": step_id,
                "candidate_idx": candidate_idx,
                "action": action,
                "stop_override": True,
                "oracle_navigation_error": _safe_float(getattr(env, "metrics", {}).get("oracle_navigation_error", float("inf")), float("inf")),
                "scaled_score": result,
            },
        )
        return result

    point = _action_to_world_point(action, env)
    if point is None:
        point = getattr(env, "position", None)
    try:
        distance = _safe_float(env.obstacle_distance(point), 0.0)
    except Exception:
        distance = 0.0
    score = max(0.0, min(distance, 1.0))
    result = score * COLLISION_REWARD_SCALE
    _debug_log(
        "collision_reward",
        {
            "step_id": step_id,
            "candidate_idx": candidate_idx,
            "action": action,
            "stop_override": False,
            "world_point": point,
            "obstacle_distance": distance,
            "clamped_score": score,
            "scaled_score": result,
        },
    )
    return result


def target_approach_reward(
    action: Dict[str, Any],
    state: Any,
    model_out: Dict[str, Any],
    env: Any,
    **kwargs,
) -> float:
    step_id = kwargs.get("step_id", None)
    candidate_idx = kwargs.get("candidate_idx", None)
    stop_override = _is_stop_near_oracle_goal(action, env)
    if stop_override:
        result = 1.0 * TARGET_APPROACH_REWARD_SCALE
        _debug_log(
            "target_approach_reward",
            {
                "step_id": step_id,
                "candidate_idx": candidate_idx,
                "action": action,
                "stop_override": True,
                "oracle_navigation_error": _safe_float(getattr(env, "metrics", {}).get("oracle_navigation_error", float("inf")), float("inf")),
                "scaled_score": result,
            },
        )
        return result
    delta = _target_approach_delta(action, env)
    delta = delta if delta <= 3.0 else max(7 - delta, 0.0)
    score = _sigmoid_signed(delta)
    result = score * TARGET_APPROACH_REWARD_SCALE
    _debug_log(
        "target_approach_reward",
        {
            "step_id": step_id,
            "candidate_idx": candidate_idx,
            "action": action,
            "stop_override": False,
            "delta": delta,
            "sigmoid_score": score,
            "scaled_score": result,
        },
    )
    return result


def reasoning_reward(
    action: Dict[str, Any],
    state: Any,
    model_out: Dict[str, Any],
    env: Any,
    candidates: List[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    step_id = kwargs.get("step_id", None)
    candidate_idx = kwargs.get("candidate_idx", None)
    stop_override = _is_stop_near_oracle_goal(action, env)
    if stop_override:
        result = 1.0 * REASONING_REWARD_SCALE
        _debug_log(
            "reasoning_reward",
            {
                "step_id": step_id,
                "candidate_idx": candidate_idx,
                "action": action,
                "stop_override": True,
                "oracle_navigation_error": _safe_float(getattr(env, "metrics", {}).get("oracle_navigation_error", float("inf")), float("inf")),
                "scaled_score": result,
            },
        )
        return result

    candidate_list = candidates or [{"action": action, "model_out": model_out}]
    cot_flags: List[float] = []
    approach_scores: List[float] = []
    tot_a, tot_b = 0, 0
    for cand in candidate_list:
        cand_text = str((cand.get("model_out") or {}).get("text", ""))
        cot_flags.append(1.0 if _has_reasoning(cand_text) else 0.0)
        s = _target_approach_score(cand.get("action"), env)
        approach_scores.append(s)
        if _has_reasoning(cand_text): tot_a += s
        else: tot_b += s
    tot_a /= len(candidate_list)
    tot_b /= len(candidate_list)
    c = _pearson_corr(cot_flags, approach_scores)

    cur_text = str((model_out or {}).get("text", ""))
    cur_has_cot = _has_reasoning(cur_text)
    cur_approach = _target_approach_score(action, env)

    if cur_approach > 0:
        score = max(0.0, c) if cur_has_cot else max(0.0, -c)
    else:
        score = min(0.0, c) if cur_has_cot else min(0.0, -c)
        
    result = score * REASONING_REWARD_SCALE
    _debug_log(
        "reasoning_reward",
        {
            "step_id": step_id,
            "candidate_idx": candidate_idx,
            "action": action,
            "stop_override": False,
            "candidate_count": len(candidate_list),
            "cot_flags": cot_flags,
            "approach_scores": approach_scores,
            "corr": c,
            "current_has_cot": cur_has_cot,
            "current_approach": cur_approach,
            "base_score": score,
            "scaled_score": result,
        },
    )
    return result


def stop_reward(
    action: Dict[str, Any],
    state: Any,
    model_out: Dict[str, Any],
    env: Any,
    **kwargs,
) -> float:
    step_id = kwargs.get("step_id", None)
    candidate_idx = kwargs.get("candidate_idx", None)
    is_stop = _is_stop_action(action)
    current_dist = _expert_distance(env, None)
    if not math.isfinite(current_dist):
        _debug_log(
            "stop_reward",
            {
                "step_id": step_id,
                "candidate_idx": candidate_idx,
                "action": action,
                "is_stop": is_stop,
                "current_expert_distance": current_dist,
                "note": "non-finite current distance",
                "scaled_score": 0.0,
            },
        )
        return 0.0

    target_point = _action_to_world_point(action, env)
    target_dist = _expert_distance(env, target_point) if target_point is not None else current_dist

    score = 0.0
    if is_stop and current_dist < 1.5:
        score = 1.0
    elif is_stop and current_dist > 3.0:
        score = -1.0
    elif current_dist < 2.5:
        if is_stop or (math.isfinite(target_dist) and target_dist < current_dist):
            score = 0.5
        elif math.isfinite(target_dist) and target_dist > current_dist:
            score = -0.5
    result = score * STOP_REWARD_SCALE
    _debug_log(
        "stop_reward",
        {
            "step_id": step_id,
            "candidate_idx": candidate_idx,
            "action": action,
            "is_stop": is_stop,
            "current_expert_distance": current_dist,
            "target_expert_distance": target_dist,
            "base_score": score,
            "scaled_score": result,
        },
    )
    return result


def weighted_target_approach_step_select(
    reward_dict: Dict[str, List[float]],
    advantages: List[float],
    candidates: List[Dict[str, Any]],
    **kwargs,
) -> int:
    step_id = kwargs.get("step_id", None)
    if not candidates:
        return 0
    n = len(candidates)
    target_scores = reward_dict.get("target_approach_reward")
    if not isinstance(target_scores, list) or len(target_scores) != n:
        picked = random.randrange(n)
        _debug_log(
            "weighted_target_approach_step_select",
            {
                "step_id": step_id,
                "candidate_count": n,
                "reason": "missing_or_invalid_target_approach_reward",
                "reward_dict_keys": list(reward_dict.keys()),
                "picked_index": picked,
            },
        )
        return picked

    ranked = []
    for idx, val in enumerate(target_scores):
        try:
            score = float(val)
        except Exception:
            score = -float("inf")
        ranked.append((score, idx))
    ranked.sort(key=lambda x: x[0], reverse=True)

    best_idx = ranked[0][1]
    second_idx = ranked[1][1] if n > 1 else best_idx
    worst_idx = ranked[-1][1]

    p = random.random()
    if p < 0.45:
        picked = best_idx
        policy = "best"
    elif p < 0.55:
        picked = second_idx
        policy = "second"
    elif p < 0.70:
        picked = worst_idx
        policy = "worst"
    else:
        picked = random.randrange(n)
        policy = "random"

    _debug_log(
        "weighted_target_approach_step_select",
        {
            "step_id": step_id,
            "candidate_count": n,
            "reward_dict": reward_dict,
            "advantages": advantages,
            "target_scores": target_scores,
            "ranked": ranked,
            "best_idx": best_idx,
            "second_idx": second_idx,
            "worst_idx": worst_idx,
            "rand": p,
            "policy": policy,
            "picked_index": picked,
        },
    )
    return picked


def random_step_select(
    reward_dict,
    advantages,
    candidates,
    **kwargs,
):
    step_id = kwargs.get("step_id", None)
    if not candidates:
        return 0

    picked = random.randrange(len(candidates))
    _debug_log(
        "random_step_select",
        {
            "step_id": step_id,
            "candidate_count": len(candidates),
            "policy": "random",
            "picked_index": picked,
        },
    )
    return picked
