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
from scipy import interpolate
from tqdm import tqdm


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
    
    existing_ids.sort()
    threshold = total_episodes / world_size / 4.0

    segments: List[List[int]] = []
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
        except Exception as exc:
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
            if d < best_dist:
                best_dist = d
                best = i
            # Early break once distance starts to climb after a good match
            if best_dist < 0.05 and d > best_dist + 0.2:
                break
        align_idx.append(best)
        start = best
    return align_idx


def pick_index_by_distance(
    cumdist: np.ndarray, target_dist: float, lo: int, hi: int
) -> int:
    """Pick index in [lo, hi] whose cumulative distance is closest to target_dist."""
    sub = cumdist[lo : hi + 1]
    idx = int(np.argmin(np.abs(sub - target_dist)))
    return lo + idx


def segment_trajectory(
    traj: np.ndarray,
    forced_boundaries: Iterable[int],
    target_len: float,
    max_len: float,
    min_len: float,
    hard_min: float,
) -> List[int]:
    """
    Split compressed trajectory into step start indices.
    - forced_boundaries: indices that must be step starts (e.g., reference points with CoT).
    Returns a sorted list of start indices; the last element is the final point index.
    """
    n = len(traj)
    assert n >= 1, "Empty trajectory"
    cumdist = cumulative_dist(traj)

    forced = sorted({i for i in forced_boundaries if 0 <= i < n})
    if 0 not in forced:
        forced = [0] + forced

    starts = [0]

    def segment_interval(s_idx: int, e_idx: int):
        cur = s_idx
        while cur < e_idx:
            remaining = cumdist[e_idx] - cumdist[cur]
            if remaining <= max_len:
                # Accept final hop (even if shorter than min when unavoidable)
                if remaining < hard_min and cur != s_idx:
                    pass
                starts.append(e_idx)
                break

            if remaining < max_len + min_len:
                target = cumdist[e_idx] - min_len
            else:
                target = min(cumdist[cur] + target_len, cumdist[cur] + max_len)
                if cumdist[e_idx] - target < min_len:
                    target = cumdist[e_idx] - min_len

            nxt = pick_index_by_distance(cumdist, target, cur + 1, e_idx)

            # Enforce min/hard_min if possible
            seg_len = cumdist[nxt] - cumdist[cur]
            if seg_len < hard_min and nxt < e_idx:
                # Move forward to the first index that satisfies hard_min, if any
                candidates = np.where(
                    cumdist[cur + 1 : e_idx + 1] - cumdist[cur] >= hard_min
                )[0]
                if len(candidates) > 0:
                    nxt = cur + 1 + int(candidates[0])

            starts.append(nxt)
            cur = nxt

    # Walk through forced boundaries
    forced_tail = forced[1:]
    prev = 0
    for fb in forced_tail:
        segment_interval(prev, fb)
        prev = fb

    # Final stretch to the end
    if prev != n - 1:
        segment_interval(prev, n - 1)
    elif starts[-1] != n - 1:
        starts.append(n - 1)

    # Deduplicate in case something slipped
    dedup = [starts[0]]
    for idx in starts[1:]:
        if idx != dedup[-1]:
            dedup.append(idx)
    return dedup


def future_traj_window(
    traj: np.ndarray,
    cumdist: np.ndarray,
    start_idx: int,
    anchor_indices: Iterable[int],
    max_forward: float,
    anchor_extra: float,
) -> List[List[float]]:
    """
    Slice future trajectory (including start point) up to max_forward meters.
    If the next anchor (reference point with CoT) lies within max_forward + anchor_extra,
    include it even if that slightly exceeds the limit; if the anchor is before max_forward,
    allow up to +anchor_extra beyond it.
    """
    anchors = sorted(i for i in anchor_indices if i > start_idx)
    start_d = cumdist[start_idx]
    limit = start_d + max_forward
    if anchors:
        nxt = anchors[0]
        dist_to_anchor = cumdist[nxt] - start_d
        if dist_to_anchor <= max_forward:
            limit = min(start_d + max_forward, start_d + dist_to_anchor + anchor_extra)
        elif dist_to_anchor <= max_forward + anchor_extra:
            limit = start_d + dist_to_anchor

    pts: List[List[float]] = []
    for i in range(start_idx, len(traj)):
        pts.append(traj[i].tolist())
        if cumdist[i] >= limit:
            break
    if not pts:
        pts.append(traj[start_idx].tolist())
    return pts


def smooth_and_sample(
    points: Sequence[Sequence[float]], target_spacing: float = 0.5, min_points: int = 10
) -> List[List[float]]:
    """Spline smooth + uniform arc-length sampling; falls back to linear when needed."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        return [pts[0].tolist()] * max(1, min_points)

    chord_cum = cumulative_dist(pts)
    chord_total = chord_cum[-1]
    if chord_total < 1e-6:
        return [pts[0].tolist()] * max(1, min_points)

    try:
        # Parameterize by chord length to reduce distortions
        base_t = chord_cum / (chord_total + 1e-8)
        k = min(3, len(pts) - 1)
        tck, _ = interpolate.splprep(pts.T, u=base_t, s=0.0, k=k)
        dense_u = np.linspace(0.0, 1.0, 200)
        dense = np.stack(interpolate.splev(dense_u, tck), axis=1)
    except Exception:
        dense = pts

    dense_cum = cumulative_dist(dense)
    total_len = dense_cum[-1]
    if total_len < 1e-6:
        return [dense[0].tolist()] * max(1, min_points)

    # Determine sampling distances
    spacing = target_spacing
    min_required = max(min_points, 2)
    spacing = min(spacing, total_len / max(min_required - 1, 1))
    count = max(int(math.floor(total_len / spacing)) + 1, min_required)
    dists = np.linspace(0.0, total_len, count)

    sampled: List[List[float]] = []
    for d in dists:
        interp_pt = [float(np.interp(d, dense_cum, dense[:, dim])) for dim in range(3)]
        sampled.append(interp_pt)
    return sampled


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

    return RolloutRecord(agent_pos, agent_rot, agent_rote, sensor_states, rgb_obs)


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
    Actions sequence should align with move_to_end outputs (starts with 'START').
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

def save_videos(records, record_sensor, record_video_path, fps=4):
    frames = []
    for rec in records:
        img = rec.obs[record_sensor]  
        frames.append(img)
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
        json.dump(raw_episode, f, ensure_ascii=False, indent=2, default=json_default)

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
            f"black frame detected (record #{rec_idx}, sensor '{sensor_name}' ratio={ratio:.2%} > {args.max_black_ratio:.2%})"
        )

    if args.record_sensor is not None:
        save_videos(records, args.record_sensor, record_video_path)
    positions = np.stack([r.position for r in records], axis=0)

    compressed, kept_idx = compress_trajectory(positions, args.compress_eps)
    cumdist = cumulative_dist(compressed)

    # 4) Map reference-path CoT to trajectory indices
    cot_map, ref_to_traj = build_cot_map(raw_episode, compressed)
    forced_boundaries = set(cot_map.keys()) | {0}

    # 5) Step segmentation
    step_starts = segment_trajectory(
        compressed,
        forced_boundaries,
        target_len=args.target_step_m,
        max_len=args.max_step_m,
        min_len=args.min_step_m,
        hard_min=args.hard_min_step_m,
    )

    # 6) Build steps + collect images
    if args.save_png:
        images_png_dir.mkdir(parents=True, exist_ok=True)
    images_npz: Dict[str, np.ndarray] = {}
    steps: List[Dict[str, Any]] = []
    anchor_indices = sorted(cot_map.keys())

    step_indices = list(step_starts[:-1])
    if step_starts and step_starts[-1] in cot_map:
        if not step_indices or step_indices[-1] != step_starts[-1]:
            step_indices.append(step_starts[-1])

    for sid, start_idx in enumerate(step_indices):
        orig_idx = kept_idx[start_idx]
        rec = records[orig_idx]

        # Images
        image_keys: Dict[str, str] = {}
        for name, img in rec.obs.items():
            key = f"step{sid:03d}_{name}"
            image_keys[name] = key
            images_npz[key] = img
            if args.save_png:
                try:
                    imageio.imwrite(images_png_dir / f"{key}.png", img)
                except Exception as e:  # pragma: no cover - optional convenience
                    print("="*50)
                    print(f"[WARN] Failed to save PNG for {key}: {e}")
                    traceback.print_exc()
                    print("="*50)

        # Sensor poses (only rgb_ sensors)
        sensor_poses = {
            name: {"pos": data["pos"].tolist(), "quat": data["quat"].tolist(), "euler": data["euler"].tolist()}
            for name, data in rec.sensor_states.items()
            if name.startswith("rgb_")
        }
        agent_pose = {"pos": rec.position.tolist(), "quat": rec.rotation.tolist(), "euler": rec.rotation_euler.tolist()}

        future_raw = future_traj_window(
            compressed,
            cumdist,
            start_idx,
            anchor_indices,
            max_forward=args.future_max_m,
            anchor_extra=args.future_anchor_extra_m,
        )
        future_spline = smooth_and_sample(
            future_raw, target_spacing=0.5, min_points=9
        )

        steps.append(
            {
                "step_id": sid,
                "cot": cot_map.get(start_idx, ""),
                "images": image_keys,
                "sensor_poses": sensor_poses,
                "agent_pose": agent_pose,
                "future_traj_raw": future_raw,
                "future_traj_spline": future_spline,
            }
        )

    # 7) Episode payload with debug info
    instruction = ""
    if "instruction" in env.episode_dict:
        ins_obj = env.episode_dict["instruction"]
        instruction = getattr(ins_obj, "instruction_text", "") or str(ins_obj)

    payload = {
        "instruct": instruction,
        "steps": steps,
        "debug": {
            "compressed_traj": compressed.tolist(),
            "kept_indices_from_rollout": kept_idx,
            "reference_to_traj": ref_to_traj,
            "step_start_indices": step_starts,
        },
    }

    with open(episode_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)
    np.savez(images_npz_path, **images_npz)


# -----------------------------
# --------- CLI ---------------
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT-aware trajectory dataset from DThinkEnv episodes."
    )
    parser.add_argument("--cfg", type=str, default="config/ht_dthink_base.yaml")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cot_rollouts"),
        help="Root output directory.",
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
        default=0.75,
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
        help="Sensor name to record video frames from (e.g. rgb, rgb_front).",
    )
    parser.add_argument("--target-step-m", type=float, default=2.5)
    parser.add_argument("--max-step-m", type=float, default=3.0)
    parser.add_argument("--min-step-m", type=float, default=1.5)
    parser.add_argument("--hard-min-step-m", type=float, default=0.5)
    parser.add_argument("--compress-eps", type=float, default=1e-2)
    parser.add_argument("--future-max-m", type=float, default=3.0)
    parser.add_argument("--future-anchor-extra-m", type=float, default=0.5)
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
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        clear_output_dir(output_root)
    world_size, rank = get_rank_info()
    if world_size > 1:
        print(f"[Dist] rank {rank}/{world_size}")
        
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip() != "":
        visible = visible.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[rank % len(visible)]
        
    from ...env import DThinkEnv
    env = DThinkEnv(args.cfg, split=args.split)

    try:
        # Determine episode list
        all_ids = [str(eid) for eid in env.get_episode_list()]
        total_ordered = len(all_ids)
        print(f"[Generate] total episodes to process: {total_ordered}")

        if total_ordered == 0:
            print("[Generate] No episodes available in dataset.")
            return

        end = args.end if args.end is not None else total_ordered
        end = min(end, total_ordered)
        start = max(0, args.start)

        # Split episodes evenly across torchrun ranks
        per_rank = math.ceil((end - start) / world_size)
        start_idx = start + rank * per_rank
        end_idx = min(end, start + (rank + 1) * per_rank)
        if world_size > 1:
            print(f"[Dist] Rank {rank} handling episodes [{start_idx}:{end_idx})")

        if args.resume:
            resume_idx = detect_resume_start(output_root, world_size, rank, total_ordered)
            if resume_idx is not None:
                start_idx = resume_idx
                if start_idx >= end_idx:
                    print(f"[Resume] Rank {rank}: no episodes left after resume.")
                    return
                print(f"[Resume] Rank {rank} skipping to global index {start_idx}.")

        env.slice_episodes(start_idx, end_idx)

        for _ in tqdm(range(end_idx - start_idx), desc="Episodes"):
            wait_for_stream_slot(output_root, args.stream_max_ready, args.stream_sleep_sec)
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
