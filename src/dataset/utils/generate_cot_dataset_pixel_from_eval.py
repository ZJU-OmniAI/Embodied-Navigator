"""
Convert successful evaluation rollouts into training-ready pixel episode folders.

The script reads evaluation outputs produced by ``src/eval/evaluate_multi.py``
inside a run directory (e.g. ``runs/DThinkVLN-P-7B-SFT-S1-V114_1-600-EVAL``),
filters episodes whose ``result.json`` has ``success`` above a threshold, and
rewrites them into the same directory structure expected by
``DThinkEpisodeDatasetPixelStream`` / ``generate_cot_dataset_pixel.py``:

output_root/
  └── episode_<origid>_<retry>/
        ├── episode.json
        └── images/
              step000_rgb_front.png
              ...

Only PNGs are written by default; ``--save-npz`` additionally emits
``images.npz`` for faster loading.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

GEN_CACHE_PREFIX = "_gen_cache"


# -----------------------------
# ----- JSON helpers ----------
# -----------------------------

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
        return [_sanitize_nan(v) for v in obj]
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


# -----------------------------
# --------- Utils -------------
# -----------------------------

def ready_dir_count(output_root: Path) -> int:
    """Count ready episode directories (exclude underscore-prefixed in-progress dirs)."""
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
        if ready_dir_count(output_root) < max_ready:
            return
        time.sleep(sleep_s)


def episode_dirs(output_root: Path, episode_dir_name: str) -> Tuple[Path, Path]:
    """
    Return (final_episode_dir, cache_episode_dir) for atomic streaming output.

    cache dir name is prefixed with GEN_CACHE_PREFIX so it starts with '_' and won't
    be claimed by DThinkEpisodeDatasetPixelStream while still being written.
    """
    ep_dir = output_root / episode_dir_name
    cache_dir = output_root / f"{GEN_CACHE_PREFIX}{episode_dir_name}"
    return ep_dir, cache_dir


def _parse_sensor_from_image(path: str) -> Tuple[str, str]:
    """
    Given an eval image path like '.../3_rgb_front.png', return
    (sensor_name, key_suffix). key_suffix is used to build the output key.
    """
    stem = Path(path).stem  # e.g. "3_rgb_front"
    if "_" in stem:
        parts = stem.split("_", 1)
        step_prefix, sensor = parts[0], parts[1]
    else:
        step_prefix, sensor = stem, stem
    return sensor, step_prefix


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_actions(jsonl_path: Path) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                actions.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - best effort
                print(f"[WARN] skip malformed line in {jsonl_path}: {exc}")
                continue
    return actions


def _is_success(result_path: Path, threshold: float) -> bool:
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        return float(res.get("success", 0.0)) >= threshold
    except Exception:
        return False


def _pick_agent_pose(sensor_poses: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer the aggregated 'rgb' pose; otherwise first available."""
    if "rgb" in sensor_poses:
        return sensor_poses["rgb"]
    if "depth" in sensor_poses:
        return sensor_poses["depth"]
    if sensor_poses:
        # deterministic pick: first key sorted
        key = sorted(sensor_poses.keys())[0]
        return sensor_poses[key]
    return {}


def _build_projection(
    step: Dict[str, Any],
    chosen_sensor: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return (chosen_projection, all_projections) sanitized for json.
    Falls back to model/env action pixels when goal projection is missing.
    """
    projections = step.get("goal_pixel") or {}
    chosen_proj = projections.get(chosen_sensor) if chosen_sensor else None

    def _valid_uv(proj: Dict[str, Any]) -> bool:
        if not isinstance(proj, dict):
            return False
        u, v = proj.get("u"), proj.get("v")
        return u is not None and v is not None and math.isfinite(float(u)) and math.isfinite(float(v))

    if not _valid_uv(chosen_proj):
        pixel = None
        model_act = step.get("model_action") or {}
        if isinstance(model_act, dict):
            pixel = model_act.get("pixel") or model_act.get("action")
        env_act = step.get("env_action") or {}
        if pixel is None and isinstance(env_act, dict):
            pixel = env_act.get("pixel") or env_act.get("action")
        if isinstance(pixel, (list, tuple)) and len(pixel) >= 2:
            chosen_proj = {
                "sensor": chosen_sensor,
                "u": float(pixel[0]),
                "v": float(pixel[1]),
                "visible": True,
            }
        else:
            chosen_proj = {}
    return _sanitize_nan(chosen_proj), _sanitize_nan(projections)


def _copy_images(
    step_idx: int,
    image_paths: List[str],
    out_dir: Path,
    save_npz: bool,
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Copy images into out_dir and build {sensor: key} mapping plus optional npz cache.
    """
    images_map: Dict[str, str] = {}
    npz_cache: Dict[str, np.ndarray] = {}

    for p in image_paths:
        sensor, _ = _parse_sensor_from_image(p)
        key = f"step{step_idx:03d}_{sensor}"
        dst = out_dir / f"{key}.png"
        try:
            shutil.copy2(p, dst)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[WARN] failed to copy image {p} -> {dst}: {exc}")
            continue
        images_map[sensor] = key
        if save_npz:
            try:
                npz_cache[key] = imageio.imread(dst)
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] failed to read {dst} for npz: {exc}")
    return images_map, npz_cache


def _build_step(
    step_idx: int,
    step: Dict[str, Any],
    images_dir: Path,
    save_npz: bool,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    def _extract_cot(text: str) -> str:
        """
        Pull content between <|think_start|> and <|think_end|>.
        If markers are absent, return the raw text.
        """
        if not isinstance(text, str):
            return ""
        start_tok = "<|think_start|>"
        end_tok = "<|think_end|>"
        start = text.find(start_tok)
        end = text.rfind(end_tok)
        if start != -1 and end != -1 and end > start:
            return text[start + len(start_tok): end].replace("<|think_start|>", "").replace("<|think_end|>", "").strip()
        return text.replace("<|think_start|>", "").replace("<|think_end|>", "").strip()
    images_map, npz_cache = _copy_images(step_idx, step.get("images") or [], images_dir, save_npz)
    sensor_poses = step.get("sensor_poses") or {}
    agent_pose = _pick_agent_pose(sensor_poses)

    chosen_sensor = ""
    model_act = step.get("model_action") or {}
    if isinstance(model_act, dict):
        chosen_sensor = model_act.get("choice") or model_act.get("sensor") or ""
    if not chosen_sensor:
        env_act = step.get("env_action") or {}
        if isinstance(env_act, dict):
            sensor_env = env_act.get("sensor") or ""
            if sensor_env.startswith("depth"):
                chosen_sensor = sensor_env.replace("depth", "rgb", 1)
            else:
                chosen_sensor = sensor_env

    chosen_proj, projections = _build_projection(step, chosen_sensor)

    metrics = step.get("metrics") or {}
    dist = float(metrics.get("distance_to_goal", 0.0)) if isinstance(metrics, dict) else 0.0

    out_step: Dict[str, Any] = {
        "step_id": step_idx,
        "cot": _extract_cot(step.get("model_text", "")),
        "point": {
            "world": [],
            "target_index": step_idx,
            "chosen_sensor": chosen_sensor,
            "projections": projections,
            "chosen_projection": chosen_proj,
            "distance_from_start": dist,
            "segment_indices": [],
            "image_key": images_map.get(chosen_sensor, ""),
        },
        "images": images_map,
        "sensor_poses": sensor_poses,
        "agent_pose": agent_pose,
    }

    env_act = step.get("env_action") or {}
    if isinstance(env_act, dict) and str(env_act.get("action", "")).upper() == "STOP":
        out_step["stop"] = True

    return out_step, npz_cache


def process_episode(
    ep_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> Optional[str]:
    result_json = ep_dir / "result.json"
    actions_jsonl = ep_dir / "actions.jsonl"
    if not actions_jsonl.exists():
        return None
    if result_json.exists() and not _is_success(result_json, args.min_success):
        return None

    actions = _load_actions(actions_jsonl)
    if not actions:
        return None

    wait_for_stream_slot(output_root, args.stream_max_ready, args.stream_sleep_sec)

    out_name = f"episode_{ep_dir.name}_3"
    out_ep_dir, cache_dir = episode_dirs(output_root, out_name)

    # Ensure a clean in-progress cache dir; never write directly into the ready dir
    # to avoid consumers claiming partially-written episodes.
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    if out_ep_dir.exists():
        if not args.overwrite:
            return f"skip {ep_dir.name}: exists (use --overwrite)"
        shutil.rmtree(out_ep_dir, ignore_errors=True)

    try:
        images_dir = cache_dir / "images"
        _ensure_dir(images_dir)

        steps: List[Dict[str, Any]] = []
        npz_cache_total: Dict[str, np.ndarray] = {}
        for idx, step in enumerate(actions):
            built, npz_part = _build_step(idx, step, images_dir, args.save_npz)
            steps.append(built)
            npz_cache_total.update(npz_part)

        # Ensure last step marked stop if the logged action was STOP
        if steps:
            last_env = actions[-1].get("env_action") or {}
            if isinstance(last_env, dict) and str(last_env.get("action", "")).upper() == "STOP":
                steps[-1]["stop"] = True

        payload = {
            "episode_id": ep_dir.name,
            "instruct": actions[0].get("instruction", "") if actions else "",
            "steps": steps,
            "debug": {
                "source_dir": str(ep_dir),
                "success": True,
            },
        }

        with open(cache_dir / "episode.json", "w", encoding="utf-8") as f:
            json.dump(
                _sanitize_nan(payload),
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=json_default,
            )

        if args.save_npz and npz_cache_total:
            try:
                np.savez(cache_dir / "images.npz", **npz_cache_total)
            except Exception as exc:  # pragma: no cover - optional
                print(f"[WARN] failed to save images.npz for {ep_dir.name}: {exc}")

        # Atomic promote: cache dir -> ready dir (claimed by dataset stream)
        os.replace(cache_dir, out_ep_dir)
    except Exception:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise

    return f"ok {ep_dir.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert successful eval rollouts into pixel episode folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs/DThinkVLN-P-7B-SFT-S1-V114_1-600-EVAL"),
        help="Eval run directory containing episode folders and result.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cot_pixel_from_eval"),
        help="Output root for generated episode_* folders.",
    )
    parser.add_argument(
        "--min-success",
        type=float,
        default=0.5,
        help="Keep episodes whose result.json success >= this threshold.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode_* folders in output-dir.",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also store images.npz besides PNGs for faster training load.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on number of successful episodes to convert.",
    )
    parser.add_argument(
        "--stream-max-ready",
        type=int,
        default=0,
        help=(
            "If > 0, block until the number of non-underscore directories in output-dir "
            "drops below this value (streaming mode)."
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
    input_dir: Path = args.input_dir
    output_root: Path = args.output_dir

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    _ensure_dir(output_root)

    subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
    processed = 0

    for ep_dir in tqdm(sorted(subdirs), desc="Episodes"):
        msg = process_episode(ep_dir, output_root, args)
        if msg is None:
            continue
        print(msg)
        processed += 1
        if args.max_episodes is not None and processed >= args.max_episodes:
            break

    print(f"[Done] converted {processed} episodes into {output_root}")


if __name__ == "__main__":
    main()
