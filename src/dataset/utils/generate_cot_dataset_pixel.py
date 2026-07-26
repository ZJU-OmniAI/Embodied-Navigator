import argparse
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

# NOTE: keep SensorGeometry import for type hints / static helpers; runtime env.sg is used for projection
from ...env.habitat_extensions.rgb2pos import SensorGeometry


GEN_CACHE_PREFIX = "_gen_cache"


class EpisodeFilteredError(RuntimeError):
    """Raised when an episode should be discarded due to quality filters."""


# -----------------------------
# --- Distributed helpers -----
# -----------------------------


def get_rank_info() -> tuple[int, int]:
    """Return (world_size, rank) using torchrun environment variables if present."""
    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        world_size = 1
    try:
        rank = max(0, int(os.environ.get("RANK", "0")))
    except ValueError:
        rank = 0
    if rank >= world_size:
        rank = world_size - 1
    return world_size, rank


def detect_resume_start(
    output_root: Path,
    world_size: int,
    rank: int,
    total_episodes: int,
) -> Optional[int]:
    """
    Infer resume start index for this rank by inspecting existing episode folders.
    - Collect episode ids from dirs named like episode_<id>.
    - Sort and split into contiguous segments where neighbor diff < total/world_size/8.
    - Assert the number of segments equals world_size and return the index of the
      segment tail within ordered_ids for this rank.
    """
    existing_ids = [
        int(p.name.split("_")[1])
        for p in output_root.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and len(p.name.split("_")) > 1
    ]

    if not existing_ids:
        return None

    existing_ids.sort()
    threshold = total_episodes / world_size / 4.0

    segments: List[int] = []
    cur = existing_ids[0]
    for nxt in existing_ids:
        if nxt - cur > threshold:
            segments.append(cur)
        cur = nxt
    segments.append(cur)

    assert (
        len(segments) == world_size
    ), f"Resume segments {len(segments)} != world_size {world_size}"

    return segments[rank]


# -----------------------------
# --------- Utils -------------
# -----------------------------


def to_quat_wxyz(q: Any) -> List[float]:
    """Convert Habitat quaternion-like objects to [w, x, y, z] list."""
    if hasattr(q, "w"):
        return [float(q.w), float(q.x), float(q.y), float(q.z)]
    arr = np.asarray(q, dtype=np.float32).reshape(4)
    return arr.tolist()


def json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _sanitize_nan(obj: Any):
    """Recursively replace NaN/Inf with None so json dumps to null."""
    import math as _m

    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _sanitize_nan(v) for v in obj ]
    if isinstance(obj, np.ndarray):
        return _sanitize_nan(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if not _m.isfinite(val):
            return None
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def ready_dir_count(output_root: Path) -> int:
    return sum(
        1 for p in output_root.iterdir() if p.is_dir() and not p.name.startswith("_")
    )


def wait_for_stream_slot(output_root: Path, max_ready: int, sleep_s: float) -> None:
    """
    When streaming is enabled (max_ready > 0), block until the number of
    non-underscore dirs under output_root drops below max_ready.
    """
    if max_ready <= 0:
        return
    while True:
        ready = ready_dir_count(output_root)
        if ready < max_ready:
            break
        time.sleep(sleep_s)


def clear_output_dir(output_root: Path) -> None:
    """Remove all files and directories inside output_root."""
    for child in output_root.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"[WARN] Failed to remove {child}: {exc}")


def cumulative_dist(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def compress_trajectory(points: np.ndarray, eps: float) -> Tuple[np.ndarray, List[int]]:
    """Compress adjacent points that are closer than eps; return kept points and original indices."""
    if len(points) == 0:
        return points, []
    kept = [points[0]]
    kept_idx = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - kept[-1]) >= eps:
            kept.append(points[i])
            kept_idx.append(i)
    if kept_idx[-1] != len(points) - 1:
        kept.append(points[-1])
        kept_idx.append(len(points) - 1)
    return np.stack(kept), kept_idx


def align_reference_to_traj(
    traj: np.ndarray, reference: Sequence[Sequence[float]]
) -> List[int]:
    """
    Greedy monotonic alignment from reference_path points to trajectory indices.
    Ensures indices are non-decreasing.
    """
    if len(reference) == 0 or len(traj) == 0:
        return []
    traj = np.asarray(traj, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    align_idx: List[int] = []
    start = 0
    for ref_pt in ref:
        best = start
        best_dist = float("inf")
        for i in range(start, len(traj)):
            d = float(np.linalg.norm(traj[i] - ref_pt))
            if d < best_dist + 0.05:
                best_dist = d
                best = i
            if best_dist < 0.05 and d > best_dist + 0.2:
                break
        align_idx.append(best)
        start = best
    align_idx[0] = 0
    align_idx[-1] = len(traj) - 1
    return align_idx


# -----------------------------
# ----- Data structures -------
# -----------------------------


@dataclass
class RolloutRecord:
    position: np.ndarray
    rotation: np.ndarray
    rotation_euler: np.ndarray
    sensor_states: Dict[str, Dict[str, np.ndarray]]
    obs: Dict[str, np.ndarray]
    depth: Dict[str, np.ndarray]


def capture_state(env, obs: Dict[str, Any]) -> RolloutRecord:
    """Capture agent & sensor poses plus rgb observations."""
    agent_pos = np.asarray(env.position, dtype=np.float32)
    agent_rot = np.asarray(env.rotation, dtype=np.float32)
    agent_rote = np.asarray(env.rotation_euler, dtype=np.float32)

    sensor_states: Dict[str, Dict[str, np.ndarray]] = {}
    pose_dict = env.get_sensor_pose()  # use DThinkEnv utility
    for name, ss in pose_dict.items():
        sensor_states[name] = {
            "pos": np.asarray(ss["position"], dtype=np.float32),
            "quat": np.asarray(ss["rotation"], dtype=np.float32),
            "euler": np.asarray(ss["euler_rotation"], dtype=np.float32),
        }

    rgb_obs: Dict[str, np.ndarray] = {}
    for k, v in obs.items():
        if not k.startswith("rgb_"):
            continue
        rgb_obs[k] = np.asarray(v)
        
    depth: Dict[str, np.ndarray] = {}
    for k, v in obs.items():
        if not k.startswith("depth_"):
            continue
        depth[k] = np.asarray(v)

    return RolloutRecord(agent_pos, agent_rot, agent_rote, sensor_states, rgb_obs, depth)


def _black_pixel_ratio(image: np.ndarray) -> float:
    """Return the fraction of pixels that are pure black (all channels == 0)."""
    arr = np.asarray(image)
    if arr.size == 0:
        return 0.0
    if arr.ndim >= 3:
        black_mask = np.all(arr <= 2, axis=-1)
    else:
        black_mask = arr == 0
    return float(np.mean(black_mask))


def detect_blackout_observation(
    records: Sequence[RolloutRecord],
    threshold: float,
) -> Optional[Tuple[int, str, float]]:
    """
    Scan all rgb observations and return (record_index, sensor_name, ratio)
    if any frame exceeds the given black pixel ratio threshold.
    """
    if threshold is None or threshold >= 1.0:
        return None
    if threshold <= 0.0:
        return None
    for ridx, rec in enumerate(records):
        for sensor_name, img in rec.obs.items():
            ratio = _black_pixel_ratio(img)
            if ratio > threshold:
                return ridx, sensor_name, ratio
    return None


def replay_actions(env, actions: Sequence[Any]) -> List[RolloutRecord]:
    """
    Reset to current episode start and replay given actions, capturing state/obs after each.
    Actions sequence should align with move_to_end outputs (starts with START).
    """
    obs = env.reset(next=False)
    records = [capture_state(env, obs)]
    for act in actions[1:]:
        obs = env.step(act)
        records.append(capture_state(env, obs))
    return records


def episode_dirs(output_root: Path, episode_id: Any) -> Tuple[Path, Path]:
    ep_dir = output_root / f"episode_{episode_id}"
    cache_dir = ep_dir.parent / f"{GEN_CACHE_PREFIX}{ep_dir.name}"
    return ep_dir, cache_dir


# -----------------------------
# ----- Projection helpers ----
# -----------------------------


def _pick_sensor_name(sensor_states: Dict[str, Dict[str, np.ndarray]], desired: Optional[str]) -> str:
    """
    Resolve sensor name for projection using recorded sensor states.
    Falls back to the first available sensor when desired is missing.
    """
    if desired and desired in sensor_states:
        return desired
    if desired and desired.endswith("_sensor") and desired[:-7] in sensor_states:
        return desired[:-7]
    for name in sensor_states:
        if desired and name.startswith(desired):
            return name
    if sensor_states:
        return list(sensor_states.keys())[0]
    raise KeyError("No sensor_states recorded to resolve sensor name")


def _resolve_project_sensors(
    rec: RolloutRecord, pixel_sensor_arg: Optional[str]
) -> List[str]:
    """
    Decide which sensors to use for projection.
    - auto (default) → prioritize four-view rgb_* sensors if present; otherwise all obs keys.
    - comma-separated list → explicit sensors.
    - single name → just that one (resolved with _pick_sensor_name).
    """
    if pixel_sensor_arg is None or pixel_sensor_arg == "" or pixel_sensor_arg.lower() == "auto":
        four_view = [
            n for n in ["rgb_front", "rgb_right", "rgb_back", "rgb_left"] if n in rec.obs
        ]
        if four_view:
            return four_view
        rgb_keys = [k for k in rec.obs.keys()]
        if rgb_keys:
            return rgb_keys
        return list(rec.sensor_states.keys())

    if "," in pixel_sensor_arg:
        return [p.strip() for p in pixel_sensor_arg.split(",") if p.strip()]

    chosen = _pick_sensor_name(rec.sensor_states, pixel_sensor_arg)
    return [chosen]


def world_point_to_pixel(
    point_world: np.ndarray,
    rec: RolloutRecord,
    env,
    sensor_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Project a world point to pixel for one or more sensors using recorded poses.
    Returns {sensor_name: projection_result}.
    Uses env.sg (runtime SensorGeometry instance) to access intrinsics.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for desired in sensor_names:
        try:
            chosen = _pick_sensor_name(rec.sensor_states, desired)
            ss = rec.sensor_states[chosen]

            Kinfo = env.sg.get_K_from_cfg(chosen)
            T_ws = SensorGeometry.build_T_world_X(ss["pos"], ss["quat"])
            W, H = int(Kinfo["width"]), int(Kinfo["height"])

            res = SensorGeometry.world_to_pixel_static(
                point_world,
                Kinfo["K"],
                T_ws,
                image_size=(W, H),
                clip_outside=True,
                return_depth=True,
            )
            res.update({"sensor": chosen, "width": W, "height": H})
            # Attach observed depth (if matching depth sensor exists and depth map recorded)
            depth_name = chosen.replace("rgb", "depth", 1)
            if depth_name != chosen and res.get("visible", False):
                depth_key = None
                for cand in (depth_name, f"{depth_name}_sensor"):
                    if cand in rec.depth:
                        depth_key = cand
                        break
                if depth_key:
                    depth_img = rec.depth[depth_key]
                    if depth_img.size > 0:
                        H_d, W_d = depth_img.shape[:2]
                        u_pix = int(np.clip(round(res["u"]), 0, max(W_d - 1, 0)))
                        v_pix = int(np.clip(round(res["v"]), 0, max(H_d - 1, 0)))
                        if depth_img.ndim == 3 and depth_img.shape[2] == 1:
                            d_raw = float(depth_img[v_pix, u_pix, 0])
                        else:
                            d_raw = float(depth_img[v_pix, u_pix])
                        try:
                            scfg = SensorGeometry._get_depth_cfg(
                                env.sg._cfg, depth_name, getattr(env.sg, "_agent_name", "main_agent")
                            )
                            d_m = SensorGeometry._depth_raw_to_meters(d_raw, scfg)
                        except Exception:
                            d_m = float(d_raw)
                        res.update({"obs_depth": d_m})
            # Determine blockage by comparing observed depth and geometric ray depth
            obs_d = res.get("obs_depth", None)
            ray_d = res.get("z_depth", None)
            is_blocked = False
            if (
                obs_d is not None
                and ray_d is not None
                and np.isfinite(obs_d)
                and np.isfinite(ray_d)
            ):
                is_blocked = abs(float(obs_d) - float(ray_d)) > 0.3
            res.update({"is_blocked": bool(is_blocked)})
            out[chosen] = res
        except Exception as exc:  # pragma: no cover - best effort per sensor
            out[desired] = {"error": str(exc)}
    return out


def _score_projection(proj: Dict[str, Any], margin: int = 10) -> float:
    """Return banded center score (3=best,0=invalid) similar to PointVLN."""
    if not proj or not proj.get("visible", False):
        return 0.0
    W, H = int(proj.get("width", 0)), int(proj.get("height", 0))
    u, v = float(proj.get("u", np.nan)), float(proj.get("v", np.nan))
    if not (np.isfinite(u) and np.isfinite(v)):
        return 0.0
    if not (margin < u < W - margin and margin < v < H - margin):
        return 0.0

    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    half_w = W / 2.0
    best_r = abs(u - cx) / max(half_w, 1e-6)
    band = int(np.floor(min(max(best_r, 0.0), 1.0) * 3.0))
    band = min(band, 2)
    score = 3 - band
    return float(score)


def _visible_segment_for_sensor(
    compressed: np.ndarray,
    start_idx: int,
    candidate_idx: Iterable[int],
    rec: RolloutRecord,
    env,
    sensor: str,
    margin: int = 10,
    max_keep: int = 8,
) -> Tuple[float, List[int], Dict[int, Dict[str, Any]]]:
    """
    Collect a contiguous list of visible waypoints for one sensor starting after start_idx.
    Returns (score, step_list, projection_cache_per_idx).
    Score follows PointVLN rule based on closest-to-center point in the segment.
    """
    step_list: List[int] = []
    proj_cache: Dict[int, Dict[str, Any]] = {}

    failed_cnt = 0
    for idx in candidate_idx:
        if idx <= start_idx:
            continue
        if len(step_list) >= max_keep:
            break
        pw = compressed[idx]
        proj = world_point_to_pixel(pw, rec, env, [sensor]).get(sensor, {})
        if proj.get("visible", False) and (not proj.get("is_blocked", False)) and _score_projection(proj, margin) > 0:
            step_list.append(idx)
            proj_cache[idx] = proj
        else:
            if step_list:
                failed_cnt += 1
                if failed_cnt > 1:
                    break  # keep contiguous segment like PointVLN

    if len(step_list) < 3:
        return 0.0, [], proj_cache

    # compute band score using closest-to-center u distance
    best_r = float("inf")
    W, H = int(proj_cache[step_list[0]]["width"]), int(
        proj_cache[step_list[0]]["height"])
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    half_w = W / 2.0
    for idx in step_list:
        u = float(proj_cache[idx]["u"])
        best_r = min(best_r, abs(u - cx) / max(half_w, 1e-6))

    band = int(np.floor(min(max(best_r, 0.0), 1.0) * 3.0))
    band = min(band, 2)
    score = 3 - band
    return float(score), step_list, proj_cache


def _choose_target_with_randomness(
    compressed: np.ndarray,
    start_idx: int,
    anchor_indices: List[int],
    rec: RolloutRecord,
    env,
    sensors: List[str],
    min_step_m: float,
    max_dist_m: float = 3.0,
) -> Optional[Tuple[int, str, Dict[str, Any], Dict[int, Dict[str, Any]], List[int]]]:
    """
    单次尝试：仅在当前点到下一个 CoT 锚点之间（且距离不超过 max_dist_m）的轨迹里选点。
    - 锚点若在距离内且可视，必选，即使可视段不足 3 个点。
    - 若锚点超出距离或不可视，则在距离内的其他可视段中择优；无可视则返回 None。
    """
    next_anchor = None
    for a in anchor_indices:
        if a > start_idx:
            next_anchor = a
            break

    # 限定候选索引：在下一个锚点之前（含锚点）且距离不超过 max_dist_m
    candidate_idx: List[int] = []
    for idx in range(start_idx + 1, len(compressed)):
        if next_anchor is not None and idx > next_anchor:
            break
        dist = float(np.linalg.norm(compressed[idx] - compressed[start_idx]))
        if dist <= max_dist_m + 1e-6:
            candidate_idx.append(idx)

    sensors_shuffled = sensors[:]
    np.random.shuffle(sensors_shuffled)

    # 1) 如果锚点在距离内且可视，则直接选锚点（无需 3 连续）
    if next_anchor is not None and next_anchor in candidate_idx:
        for s in sensors_shuffled:
            proj = world_point_to_pixel(compressed[next_anchor], rec, env, [s]).get(s, {})
            if proj.get("visible", False) and (not proj.get("is_blocked", False)) and _score_projection(proj) > 0:
                proj_cache = {next_anchor: proj}
                return next_anchor, s, proj, proj_cache, [next_anchor]

    # 2) 否则在距离内的可视段里择优
    sensor_segments = []
    for s in sensors_shuffled:
        score, step_list, proj_cache = _visible_segment_for_sensor(
            compressed, start_idx, candidate_idx, rec, env, s
        )
        if score > 0 and step_list:
            sensor_segments.append((score, s, step_list, proj_cache))

    if not sensor_segments:
        return None

    sensor_segments.sort(key=lambda x: x[0], reverse=True)
    score, chosen_sensor, step_list, proj_cache = sensor_segments[0]

    n = len(step_list)
    start = n // 3
    end = 2 * n // 3
    first_third = step_list[:start]
    middle_third = step_list[start:end]
    last_third = step_list[end:]

    candidates = first_third + middle_third * 3 + last_third * 2
    if not candidates:
        target_idx = step_list[-1]
    else:
        target_idx = int(np.random.choice(candidates))

    dist = float(np.linalg.norm(
        compressed[target_idx] - compressed[start_idx]))
    short_penalty = max(0.0, (min_step_m - dist) / max(min_step_m, 1e-6))
    total_score = score - 0.5 * short_penalty

    chosen_proj = proj_cache.get(target_idx, {})

    return target_idx, chosen_sensor, chosen_proj, proj_cache, step_list


# -----------------------------
# ----- Episode pipeline ------
# -----------------------------


def build_cot_map(
    episode: Dict[str, Any], traj: np.ndarray
) -> Tuple[Dict[int, str], List[int]]:
    reference_path = episode.get("reference_path", []) or []
    reasoning = episode.get("reasoning", []) or []
    ref_to_traj = align_reference_to_traj(traj, reference_path)

    cot_map: Dict[int, str] = {}
    for ref_idx, traj_idx in enumerate(ref_to_traj):
        text = ""
        if ref_idx < len(reasoning) and reasoning[ref_idx]:
            step_info = reasoning[ref_idx]
            text = (
                step_info.get("polish_cot")
                or step_info.get("cot")
                or step_info.get("action")
                or ""
            )
            text = text.strip()
        if text and traj_idx is not None:
            cot_map.setdefault(traj_idx, text)
    return cot_map, ref_to_traj


def _pad_to_mb16(img: np.ndarray, mb: int = 16) -> np.ndarray:
    h, w = img.shape[:2]
    nh = (h + mb - 1) // mb * mb
    nw = (w + mb - 1) // mb * mb
    if nh == h and nw == w:
        return img
    pad_h = nh - h
    pad_w = nw - w
    if img.ndim == 2:
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_videos(records, record_sensor, record_video_path, fps=4):
    frames = []
    for rec in records:
        img = rec.obs[record_sensor]
        frames.append(_pad_to_mb16(img))
    imageio.mimsave(record_video_path, frames, fps=fps, codec="libx264")


def process_episode(
    env,
    episode_id: Any,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_episode.json"
    episode_json = output_dir / "episode.json"
    record_video_path = output_dir / "record.mp4"
    images_npz_path = output_dir / "images.npz"
    images_png_dir = output_dir / "images"

    # 1) Save raw episode metadata for the current environment state
    raw_episode = env.episode_dict
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_nan(raw_episode), f, ensure_ascii=False,
                  indent=2, allow_nan=False, default=json_default)

    # 2) Rollout reference path and compress trajectory
    full_traj, actions = env.move_to_end(reset=True)
    full_traj = np.asarray(full_traj, dtype=np.float32)
    actions = list(actions)

    # 3) Replay to capture obs/poses
    records = replay_actions(env, actions)

    blackout = detect_blackout_observation(records, args.max_black_ratio)
    if blackout:
        rec_idx, sensor_name, ratio = blackout
        raise EpisodeFilteredError(
            f"black frame detected (record #{rec_idx}, sensor {sensor_name} ratio={ratio:.2%} > {args.max_black_ratio:.2%})"
        )

    if args.record_sensor is not None:
        save_videos(records, args.record_sensor, record_video_path)
    positions = np.stack([r.position for r in records], axis=0)

    compressed, kept_idx = compress_trajectory(positions, args.compress_eps)

    # 4) Map CoT anchors and keep their indices
    cot_map, ref_to_traj = build_cot_map(raw_episode, compressed)
    anchor_indices = sorted(cot_map.keys())

    # 5) 外层 retry：至少 min_retry 次；若成功轨迹不足 2 条则继续到 max_retry 或够 2 条
    if args.save_png:
        images_png_dir.mkdir(parents=True, exist_ok=True)

    attempts: List[Tuple[float, List[Dict[str, Any]], Dict[str, np.ndarray]]] = []

    min_retry = max(1, int(args.min_retry))
    max_retry = max(min_retry, int(args.max_retry))

    attempt_cnt = 0
    while attempt_cnt < max_retry and (attempt_cnt < min_retry or len(attempts) < 2):
        attempt_cnt += 1
        images_npz: Dict[str, np.ndarray] = {}
        steps: List[Dict[str, Any]] = []

        current_idx = 0
        step_id = 0
        success_path = True

        while current_idx < len(compressed) - 1:
            orig_idx = kept_idx[current_idx]
            rec = records[orig_idx]

            sensors_to_use = _resolve_project_sensors(rec, args.pixel_sensor)
            choice = _choose_target_with_randomness(
                compressed,
                current_idx,
                anchor_indices,
                rec,
                env,
                sensors_to_use,
                min_step_m=args.min_step_m,
                max_dist_m=5.0,
            )

            if choice is None:
                success_path = False
                break

            target_idx, chosen_sensor, chosen_proj, proj_cache, step_list = choice

            target_world = compressed[target_idx]
            all_proj = world_point_to_pixel(target_world, rec, env, sensors_to_use)
            dist_to_target = float(np.linalg.norm(
                target_world - compressed[current_idx]))

            image_keys: Dict[str, str] = {}
            for name, img in rec.obs.items():
                key = f"step{step_id:03d}_{name}"
                image_keys[name] = key
                images_npz[key] = img
                if args.save_png:
                    try:
                        imageio.imwrite(images_png_dir / f"{key}.png", img)
                    except Exception as e:  # pragma: no cover - optional convenience
                        print("=" * 50)
                        print(f"[WARN] Failed to save PNG for {key}: {e}")
                        traceback.print_exc()
                        print("=" * 50)

            sensor_poses = {
                name: {
                    "pos": data["pos"].tolist(),
                    "quat": data["quat"].tolist(),
                    "euler": data["euler"].tolist(),
                }
                for name, data in rec.sensor_states.items()
                if name.startswith("rgb_") or name.startswith("depth") or name.startswith("rgb")
            }
            agent_pose = {
                "pos": rec.position.tolist(),
                "quat": rec.rotation.tolist(),
                "euler": rec.rotation_euler.tolist(),
            }

            steps.append(
                {
                    "step_id": step_id,
                    # cot 应对应当前所处点（起点），而非目标点
                    "cot": cot_map.get(current_idx, ""),
                    "point": {
                        "world": target_world.tolist(),
                        "target_index": int(target_idx),
                        "chosen_sensor": chosen_sensor,
                        "projections": all_proj,
                        "chosen_projection": chosen_proj,
                        "distance_from_start": dist_to_target,
                        "segment_indices": step_list,
                        "image_key": image_keys.get(chosen_sensor, ""),
                    },
                    "images": image_keys,
                    "sensor_poses": sensor_poses,
                    "agent_pose": agent_pose,
                }
            )

            step_id += 1
            if target_idx <= current_idx:
                success_path = False
                break
            current_idx = target_idx

            if current_idx >= len(compressed) - 1:
                break

        if not success_path:
            continue

        cumdist = cumulative_dist(compressed)
        remaining = float(cumdist[-1] - cumdist[current_idx])
        attempts.append((remaining, steps, images_npz))

    if not attempts:
        raise EpisodeFilteredError("no valid path after retries")

    attempts.sort(key=lambda x: x[0])  # 剩余距离越小越好
    if len(attempts) >= 2:
        top_candidates = attempts[:2]
        chosen_remaining, steps, images_npz = top_candidates[np.random.randint(2)]
    else:
        chosen_remaining, steps, images_npz = attempts[0]

    # 额外追加终止 step（到达最终点/停止），cot 取终点锚点（若有）
    final_idx = len(compressed) - 1
    final_rec = records[kept_idx[-1]]
    final_image_keys: Dict[str, str] = {}
    for name, img in final_rec.obs.items():
        key = f"step{len(steps):03d}_{name}"
        final_image_keys[name] = key
        images_npz[key] = img
        if args.save_png:
            try:
                imageio.imwrite(images_png_dir / f"{key}.png", img)
            except Exception as e:
                print("=" * 50)
                print(f"[WARN] Failed to save PNG for {key}: {e}")
                traceback.print_exc()
                print("=" * 50)

    final_sensor_poses = {
        name: {
            "pos": data["pos"].tolist(),
            "quat": data["quat"].tolist(),
            "euler": data["euler"].tolist(),
        }
        for name, data in final_rec.sensor_states.items()
        if name.startswith("rgb_") or name.startswith("depth") or name.startswith("rgb")
    }
    final_agent_pose = {
        "pos": final_rec.position.tolist(),
        "quat": final_rec.rotation.tolist(),
        "euler": final_rec.rotation_euler.tolist(),
    }

    steps.append(
        {
            "step_id": len(steps),
            "cot": cot_map.get(final_idx, ""),
            "point": {
                "world": compressed[final_idx].tolist(),
                "target_index": int(final_idx),
                "chosen_sensor": "",
                "projections": {},
                "chosen_projection": {},
                "distance_from_start": 0.0,
                "segment_indices": [],
                "image_key": "",
            },
            "images": final_image_keys,
            "sensor_poses": final_sensor_poses,
            "agent_pose": final_agent_pose,
            "stop": True,
        }
    )

    # 6) Episode payload with debug info
    instruction = ""
    if "instruction" in env.episode_dict:
        ins_obj = env.episode_dict["instruction"]
        instruction = getattr(ins_obj, "instruction_text", "") or str(ins_obj)

    payload = {
        "episode_id": raw_episode.get("episode_id"),
        "instruct": instruction,
        "steps": steps,
        "debug": {
            "compressed_traj": compressed.tolist(),
            "kept_indices_from_rollout": kept_idx,
            "reference_to_traj": ref_to_traj,
            "anchor_indices": anchor_indices,
        },
    }

    with open(episode_json, "w", encoding="utf-8") as f:
        json.dump(_sanitize_nan(payload), f, ensure_ascii=False,
                  indent=2, allow_nan=False, default=json_default)
    np.savez(images_npz_path, **images_npz)


# -----------------------------
# --------- CLI ---------------
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pixel-point CoT dataset (next visible 2D nav point + images) from DThinkEnv episodes."
    )
    parser.add_argument("--cfg", type=str,
                        default="config/ht_dthink_base.yaml")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cot_pixel_rollouts"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--pixel-sensor",
        type=str,
        default="auto",
        help="Sensors used for projection: auto (four rgb views), single name, or comma-separated list.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive) when slicing the dataset episode list.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive) when slicing the dataset episode list.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also dump rgb images to <episode>/images/*.png for manual inspection.",
    )
    parser.add_argument(
        "--max-black-ratio",
        type=float,
        default=0.9,
        help=(
            "Skip an episode if any rgb observation contains a fraction of pure-black pixels "
            "greater than this value. Set to >=1 to disable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode folders.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing episode_* directories instead of clearing output_dir.",
    )
    parser.add_argument(
        "--record-sensor",
        type=str,
        default=None,
        help="Sensor name to record video frames from (e.g. rgb_front).",
    )
    parser.add_argument("--target-step-m", type=float, default=3.5)
    parser.add_argument("--max-step-m", type=float, default=5.0)
    parser.add_argument("--min-step-m", type=float, default=2.0)
    parser.add_argument("--hard-min-step-m", type=float, default=1.0)
    parser.add_argument("--compress-eps", type=float, default=1e-2)
    parser.add_argument(
        "--retry",
        type=int,
        default=None,
        help="[兼容] 若提供则同时设置 min-retry 和 max-retry 为该值。",
    )
    parser.add_argument("--min-retry", type=int, default=3,
                        help="每个 episode 至少尝试生成完整轨迹的次数。")
    parser.add_argument("--max-retry", type=int, default=6,
                        help="最多尝试次数；不足 min-retry 或成功轨迹少于2条时会继续尝试，直到该上限。")
    
    parser.add_argument(
        "--stream-max-ready",
        type=int,
        default=0,
        help=(
            "If > 0, block each worker until the number of non-underscore "
            "directories in the output root drops below this value (streaming mode)."
        ),
    )
    parser.add_argument(
        "--stream-sleep-sec",
        type=float,
        default=5.0,
        help="Sleep duration between stream capacity checks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.retry is not None:
        args.min_retry = max(args.min_retry, args.retry)
        args.max_retry = max(args.max_retry, args.retry)
    if args.max_retry < args.min_retry:
        args.max_retry = args.min_retry
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    # if not args.resume:
    #     clear_output_dir(output_root)
    world_size, rank = get_rank_info()
    if world_size > 1:
        print(f"[Dist] rank {rank}/{world_size}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip() != "":
        visible = visible.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[rank % len(visible)]

    from ...env import DThinkEnv
    
    for i in range(1):

        env = DThinkEnv(args.cfg, split=args.split)

        try:
            all_ids = [str(eid) for eid in env.get_episode_list()]
            total_ordered = len(all_ids)
            print(f"[Generate] total episodes to process: {total_ordered}")

            if total_ordered == 0:
                print("[Generate] No episodes available in dataset.")
                return

            end = args.end if args.end is not None else total_ordered
            end = min(end, total_ordered)
            start = max(0, args.start)

            per_rank = math.ceil((end - start) / world_size)
            start_idx = start + rank * per_rank
            end_idx = min(end, start + (rank + 1) * per_rank)
            if world_size > 1:
                print(
                    f"[Dist] Rank {rank} handling episodes [{start_idx}:{end_idx})")

            if args.resume:
                resume_idx = detect_resume_start(
                    output_root, world_size, rank, total_ordered)
                if resume_idx is not None:
                    start_idx = resume_idx
                    if start_idx >= end_idx:
                        print(
                            f"[Resume] Rank {rank}: no episodes left after resume.")
                        return
                    print(
                        f"[Resume] Rank {rank} skipping to global index {start_idx}.")

            env.slice_episodes(start_idx, end_idx)

            for _ in tqdm(range(end_idx - start_idx), desc="Episodes"):
                wait_for_stream_slot(
                    output_root, args.stream_max_ready, args.stream_sleep_sec)
                try:
                    env.reset(next=True)
                except Exception as exc:
                    print("=" * 50)
                    print(f"[ERROR] Failed to reset environment: {exc}")
                    print("=" * 50)
                    continue

                current_id = str(env.episode_dict.get("episode_id"))
                ep_dir, cache_dir = episode_dirs(output_root, current_id)
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

                if ep_dir.exists() and args.overwrite:
                    shutil.rmtree(ep_dir)

                if ep_dir.exists() and not args.overwrite:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    print(f"[Skip] {current_id}: exists (use --overwrite to redo)")
                    continue

                try:
                    process_episode(env, current_id, cache_dir, args)
                    os.replace(cache_dir, ep_dir)
                except EpisodeFilteredError as exc:
                    print(f"[Skip] {current_id}: {exc}")
                    shutil.rmtree(cache_dir, ignore_errors=True)
                except Exception as exc:
                    print("=" * 50)
                    print(f"[ERROR] Failed on episode {current_id}: {exc}")
                    traceback.print_exc()
                    print("=" * 50)
                    shutil.rmtree(cache_dir, ignore_errors=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
