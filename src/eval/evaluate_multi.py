#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation entrypoint for DThinkVLN pixel-action agents.

参考 cache/PointVLN/src/eval/evaluate_multi.py 的输出结构，并兼容
src/agent/dthink_pixel_agent.py 的输入/输出签名：
- agent.reset(instruction, first_obs, sensor_poses, episode_id) -> AgentState
- agent.step(obs, state, sensor_poses) -> (action_dict, state, model_out)

功能：
- 运行一个或多个 episode，保存逐步动作日志 actions.jsonl、最终 metrics、
  以及拼接的可视化长图。
- 支持多视角传感器（默认 rgb_front / rgb_left / rgb_right / rgb_back）。
"""
from __future__ import annotations
import os
import math

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

world_size, rank = get_rank_info()
if world_size > 1:
    print(f"[Dist] rank {rank}/{world_size}")
    
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if visible.strip() != "":
    visible = visible.split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[rank % len(visible)]

import argparse
import json
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import tqdm
import yaml

from ..env import SensorGeometry
from ..utils import agent_registry, env_registry
from ..agent import *

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(p: str, obj: Dict[str, Any]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(p: str, obj: Dict[str, Any]) -> None:
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def round_floats(x: Any, ndigits: int = 4) -> Any:
    """递归地将对象中的 float 四舍五入到 ndigits 位。"""
    if isinstance(x, float):
        return round(x, ndigits)
    if isinstance(x, dict):
        return {k: round_floats(v, ndigits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [round_floats(v, ndigits) for v in x]
        return type(x)(t) if not isinstance(x, tuple) else tuple(t)
    return x


def to_serializable(x: Any) -> Any:
    """轻量转 python 原生类型，便于 json.dump。"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]
    return x


def safe_to_uint8(img: np.ndarray) -> np.ndarray:
    """将任意 HxWx{3,4} 的浮点/整型图像变为 uint8 BGR，用于 cv2.imwrite。"""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[..., :3][:, :, ::-1].copy()
    return img


def _apply_depth_noise_to_obs(obs: Optional[Dict[str, Any]], noise_var: float) -> None:
    """
    对 obs 中 key 含 'depth' 的数值数组逐元素乘高斯噪声 N(1, noise_var)。
    原地修改 obs。
    """
    if noise_var < 0:
        raise ValueError(f"`noise_var` must be non-negative, got {noise_var}.")
    if not isinstance(obs, dict) or noise_var == 0:
        return

    std = float(math.sqrt(noise_var))

    def _inject(container: Dict[str, Any]) -> None:
        for k, v in list(container.items()):
            if not (isinstance(k, str) and "depth" in k):
                continue
            try:
                arr = np.asarray(v)
            except Exception:
                continue
            if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
                continue
            noisy = arr.astype(np.float32, copy=False) * np.random.normal(
                loc=1.0,
                scale=std,
                size=arr.shape,
            ).astype(np.float32)
            if np.issubdtype(arr.dtype, np.floating):
                noisy = noisy.astype(arr.dtype, copy=False)
            container[k] = noisy

    _inject(obs)
    if isinstance(obs.get("images"), dict):
        _inject(obs["images"])


def _draw_wrapped_text(
    img: np.ndarray,
    txt: str,
    org: tuple,
    width: int = 38,
    color=(0, 0, 255),
    line_height: int = 26,
    font_scale: float = 0.75,
    thickness: int = 2,
) -> None:
    wrapped = "\n".join(textwrap.fill(line, width=width)
                        for line in txt.splitlines())
    x, y = org
    for line in wrapped.split("\n"):
        cv2.putText(
            img, line, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
        )
        y += line_height


def _resize_to_same_height(imgs: List[np.ndarray], min_w: int = 640) -> List[np.ndarray]:
    h_min = min(im.shape[0] for im in imgs)
    outs = []
    for im in imgs:
        if im.shape[0] != h_min:
            s = h_min / im.shape[0]
            new_w = int(round(im.shape[1] * s))
            outs.append(cv2.resize(im, (new_w, h_min), interpolation=cv2.INTER_AREA))
        else:
            outs.append(im)

    outs2 = []
    for im in outs:
        h, w = im.shape[:2]
        if w < min_w:
            s = min_w / w
            new_h = int(round(h * s))
            outs2.append(cv2.resize(im, (min_w, new_h), interpolation=cv2.INTER_LINEAR))
        else:
            outs2.append(im)

    h_min2 = min(im.shape[0] for im in outs2)
    outs3 = []
    for im in outs2:
        if im.shape[0] != h_min2:
            s = h_min2 / im.shape[0]
            new_w = int(round(im.shape[1] * s))
            outs3.append(cv2.resize(im, (new_w, h_min2), interpolation=cv2.INTER_AREA))
        else:
            outs3.append(im)

    return outs3


def _clamp_point(u: float, v: float, w: int, h: int) -> tuple:
    uu = int(max(0, min(w - 1, round(u))))
    vv = int(max(0, min(h - 1, round(v))))
    return uu, vv


def _normalize_action_for_env(
    action: Any,
    default_sensor: Optional[str],
) -> Dict[str, Any]:
    """
    统一 agent 返回的动作，以便 DThinkEnv.point_vln_step 直接消费。
    - MOVE_TO：确保含 action=[u,v]，sensor 使用 depth_* 名称
    - CMD    ：保持原 action 字段，默认 STOP
    """
    if not isinstance(action, dict):
        return {"type": "CMD", "action": "STOP"}

    out = dict(action)
    out["type"] = out.get("type") or out.get("action_type") or "CMD"

    if out["type"] == "MOVE_TO":
        pixel = out.get("pixel") or out.get("action")
        if not (isinstance(pixel, (list, tuple)) and len(pixel) >= 2):
            return {"type": "CMD", "action": "STOP"}
        out["action"] = [float(pixel[0]), float(pixel[1])]

        sensor = out.get("sensor") or out.get("choice") or default_sensor or ""
        if sensor.startswith("rgb"):
            sensor = sensor.replace("rgb", "depth", 1)
        out["sensor"] = sensor or default_sensor or "depth_front"
    else:
        out["action"] = out.get("action") or "STOP"

    return out


def _get_instruction_from_env(env) -> str:
    try:
        ep = getattr(env, "episode_dict", {}) or {}
        ins_node = ep.get("instruction", None)
        if ins_node is None:
            return ""
        if isinstance(ins_node, dict):
            return (
                ins_node.get("instruction_text")
                or ins_node.get("text")
                or ins_node.get("instruction")
                or ""
            )
        return getattr(ins_node, "instruction_text", "") or ""
    except Exception:
        return ""


def _goal_pixels(env, select_sensors: Sequence[str]) -> Dict[str, Any]:
    """将 episode goal 投影到各视角像素坐标；缺失字段时返回空 dict。"""
    try:
        goal = env.episode_dict["goals"][0].position
    except Exception:
        return {}

    if not hasattr(env, "sg") or not isinstance(env.sg, SensorGeometry):
        return {}

    pixels = {}
    for sensor in select_sensors:
        depth_name = sensor.replace("rgb", "depth")
        try:
            pixels[sensor] = env.sg.world_to_pixel(
                goal,
                sensor_name=depth_name,
                clip_outside=True,
                return_concealed=True,
                return_depth=True,
                convention="z",
            )
        except Exception:
            continue
    return pixels


# --------------------------------------------------------------------------- #
# Visualization                                                               #
# --------------------------------------------------------------------------- #
def _load_images(paths: List[str]) -> List[np.ndarray]:
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            im = np.full((720, 1280, 3), 200, dtype=np.uint8)
            cv2.putText(
                im,
                f"Load fail: {os.path.basename(p)}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        imgs.append(im)
    return imgs


def create_multi_view_image(
    image_paths: List[str],
    select_sensors: Sequence[str],
    metric: Dict[str, Any],
    model_text: str,
    action: Dict[str, Any],
    goal_pixel: Optional[Dict[str, Dict[str, Any]]],
    instruction: str,
) -> np.ndarray:
    """按步生成一张横向拼接图，左->右依次对应 select_sensors。"""

    # 先load（不resize）
    imgs = _load_images(image_paths)

    # 画 goal 圆点（蓝/紫）与预测像素（红）
    sensor_to_img = {s: img for s, img in zip(select_sensors, imgs)}

    def _draw_goal(img: np.ndarray, entry: Dict[str, Any], color) -> None:
        if not entry or not entry.get("visible", False):
            return
        h, w = img.shape[:2]
        u, v = _clamp_point(entry.get("u", 0), entry.get("v", 0), w, h)
        cv2.circle(img, (u, v), 10, color, 2, cv2.LINE_AA)

    if goal_pixel:
        for sensor, entry in goal_pixel.items():
            if sensor in sensor_to_img:
                _draw_goal(sensor_to_img[sensor], entry, (255, 0, 0))

    if isinstance(action, dict) and action.get("type") == "MOVE_TO":
        pix = action.get("pixel") or action.get("action")
        sensor = action.get("sensor") or action.get("choice")
        if pix is not None and sensor in sensor_to_img:
            h, w = sensor_to_img[sensor].shape[:2]
            u, v = _clamp_point(pix[0], pix[1], w, h)
            cv2.circle(sensor_to_img[sensor], (u, v), 4,
                       (0, 0, 255), -1, cv2.LINE_AA)

    # 再resize到同高
    imgs = _resize_to_same_height(imgs)

    # 文字区：metrics / model_text / action / instruction
    metrics_txt = yaml.safe_dump(
        round_floats(metric or {}), allow_unicode=True, sort_keys=False
    ).strip()
    action_txt = yaml.safe_dump(
        to_serializable(action or {}), allow_unicode=True, sort_keys=False
    ).strip()
    instruction_txt = instruction or ""

    if imgs:
        _draw_wrapped_text(imgs[0], metrics_txt, (20, 40), color=(0, 0, 255))
    if len(imgs) >= 2:
        _draw_wrapped_text(imgs[1], model_text or "", (20, 40),
                           color=(255, 0, 0))
    if len(imgs) >= 3:
        _draw_wrapped_text(imgs[2], action_txt, (20, 40),
                           color=(0, 128, 0))
    if len(imgs) >= 4 and instruction_txt:
        _draw_wrapped_text(imgs[3], instruction_txt, (20, 40),
                           color=(128, 0, 128))

    return cv2.hconcat(imgs)

def vstack_images(
    episode_data: List[dict],
    out_path: str,
    select_sensors: Sequence[str],
) -> None:
    """将若干张 multi-view 图按时间顺序竖向拼接为长图。"""
    if not episode_data:
        return
    panels = []
    for data in episode_data:
        panels.append(
            create_multi_view_image(
                data["images"],
                select_sensors,
                data.get("metrics", {}),
                data.get("model_text", ""),
                data.get("model_action", {}),
                data.get("goal_pixel", {}),
                data.get("instruction", ""),
            )
        )
    result = np.concatenate(panels, axis=0)
    cv2.imwrite(out_path, result)


# --------------------------------------------------------------------------- #
# Core evaluation                                                             #
# --------------------------------------------------------------------------- #
def evaluate(
    env,
    agent,
    save_root: str = "./eval_out",
    instruction: Optional[str] = None,
    max_steps: Optional[int] = None,
    episode_prefix: str = "episode",
    save_long_image: bool = False,
    max_steps_per_move: int = 16,
    goal_radius: float = 0.45,
    verbose: bool = True,
    select_sensors: Sequence[str] = ("rgb_front", "rgb_left", "rgb_right", "rgb_back"),
    noise_var: float = 0.01,
) -> Dict[str, Any]:
    """
    执行单个 episode：
    - 循环直到 env.episode_over 或达到 max_steps
    - 保存逐步可视化与动作日志
    返回：该 episode 的最终 metrics
    """
    try:
        episode_id = str(env.episode_dict["episode_id"])
    except Exception as e:
        raise RuntimeError("无法从 env.episode_dict['episode_id'] 获取 episode_id") from e

    # 确保唯一输出目录（与 PointVLN evaluate 行为一致）
    retry_idx = 1
    ep_dir = os.path.join(save_root, f"{episode_id}_{retry_idx}")
    while os.path.exists(ep_dir):
        retry_idx += 1
        ep_dir = os.path.join(save_root, f"{episode_id}_{retry_idx}")
    img_dir = os.path.join(ep_dir, "images")
    ensure_dir(img_dir)

    actions_jsonl = os.path.join(ep_dir, "actions.jsonl")
    result_json = os.path.join(ep_dir, "result.json")
    long_png = os.path.join(ep_dir, f"{episode_prefix}_{episode_id}.png")

    # 初始观测
    obs = getattr(env, "observations", None)
    if obs is None:
        if hasattr(env, "reset"):
            obs = env.reset(False)
        else:
            raise RuntimeError("环境未提供 observations，也没有 reset() 方法以产生首帧。")
    _apply_depth_noise_to_obs(obs, noise_var)

    instruction = instruction if instruction is not None else _get_instruction_from_env(env)
    instruction = instruction or ""

    # Agent 初始化（传入第一帧 & 位姿，便于构造 prompt）
    sensor_poses = to_serializable(getattr(env, "get_sensor_pose", lambda: {})())
    state = agent.reset(
        instruction=instruction,
        first_obs=obs,
        sensor_poses=sensor_poses,
        episode_id=episode_id,
    )

    # 清空 actions.jsonl
    if os.path.exists(actions_jsonl):
        os.remove(actions_jsonl)
    open(actions_jsonl, "a").close()

    episode_data: List[Dict[str, Any]] = []
    step_idx = 0

    default_sensor = select_sensors[0] if select_sensors else "rgb_front"
    default_sensor = default_sensor.replace("rgb", "depth")

    while True:
        # 保存多视角图像
        saved_imgs: List[str] = []
        for sensor in select_sensors:
            if sensor not in obs:
                continue
            img_path = os.path.join(img_dir, f"{step_idx}_{sensor}.png")
            cv2.imwrite(img_path, safe_to_uint8(np.asarray(obs[sensor])))
            saved_imgs.append(img_path)

        goal_pixel = _goal_pixels(env, select_sensors)

        # Agent 推理
        action_raw, state, model_out = agent.step(
            obs,
            state,
            sensor_poses=sensor_poses,
        )
        model_text = model_out.get("text", "") if isinstance(model_out, dict) else ""

        env_action = _normalize_action_for_env(action_raw, default_sensor)
        if hasattr(env, "point_vln_step"):
            try:
                env.point_vln_step(env_action, sg=getattr(env, "sg", None),
                               max_steps=max_steps_per_move, goal_radius=goal_radius)
            except:
                pass
        elif hasattr(env, "step"):
            env.step(env_action)
        else:
            raise RuntimeError("环境既没有 point_vln_step(action) 也没有 step(action)")

        metrics = round_floats(to_serializable(getattr(env, "metrics", {})))

        step_record = {
            "step": step_idx,
            "instruction": instruction,
            "images": saved_imgs,
            "model_text": model_text,
            "model_action": to_serializable(action_raw),
            "env_action": env_action,
            "metrics": metrics,
            "goal_pixel": to_serializable(goal_pixel),
            "sensor_poses": sensor_poses,
        }
        append_jsonl(actions_jsonl, step_record)
        episode_data.append(step_record)

        step_idx += 1
        if getattr(env, "episode_over", False):
            break
        if max_steps is not None and step_idx >= max_steps:
            if verbose:
                print(f"[episode {episode_id}] 达到 max_steps={max_steps}，强制结束。")
            break

        obs = getattr(env, "observations", None) or obs
        sensor_poses = to_serializable(getattr(env, "get_sensor_pose", lambda: {})())
        _apply_depth_noise_to_obs(obs, noise_var)

    final_metrics = round_floats(to_serializable(getattr(env, "metrics", {})))
    save_json(result_json, final_metrics)
    if save_long_image:
        try:
            vstack_images(episode_data, long_png, select_sensors)
        except Exception as e:
            if verbose:
                print(f"[episode {episode_id}] 生成长图失败：{e}")

    if verbose:
        print(f"[episode {episode_id}] 输出目录：{ep_dir}")
    return final_metrics


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def cmp_metrics(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    """简单比较：优先 success / 距离，其次 spl。"""
    if a is None:
        return False
    if b is None:
        return True
    for key, larger_is_better in [
        ("success", True),
        ("oracle_success", True),
        ("spl", True),
        ("distance_to_goal", False),
    ]:
        if key in a and key in b and a[key] != b[key]:
            return a[key] > b[key] if larger_is_better else a[key] < b[key]
    return True


def _add_metrics(accum: Dict[str, float], metrics: Dict[str, Any]) -> None:
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                accum[f"{k}/{sub_k}"] += float(sub_v)
        else:
            accum[k] += float(v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DThinkVLN Evaluate")
    p.add_argument("--env_config", type=str,
                   default="config/ht_dthink_r2r.yaml", help="环境配置文件")
    p.add_argument("--checkpoint", type=str,
                   default="model/DThinkVLN-7B-SFT-S3/checkpoint-3000",
                   help="模型 checkpoint 路径")
    p.add_argument("--save_root", type=str,
                   default="runs/dthink_eval", help="评估结果保存根目录")
    p.add_argument("--retry_times", type=int, default=4, help="每个 episode 重试次数")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="最多评估多少个 episode，0 表示全部")
    p.add_argument("--max_steps_per_move", type=int, default=64,
                   help="point_vln_step 内部允许的最大 step 数")
    p.add_argument("--goal_radius", type=float, default=0.45,
                   help="目标半径阈值（米）")
    p.add_argument("--episode_prefix", type=str,
                   default="episode", help="长图文件前缀")
    p.add_argument("--save_long_image", action="store_true",
                   help="是否生成长图（默认关闭）")
    p.add_argument("--max_steps", type=int, default=24,
                   help="可选全局步数上限（防卡死）")
    p.add_argument("--verbose", action="store_true",
                   help="打印调试信息")
    p.add_argument("--instruction", type=str, default=None,
                   help="覆盖默认的指令文本")

    p.add_argument("--env_type", type=str,
                   default="dthink_base", help="env_registry 名称")
    p.add_argument("--agent_type", type=str,
                   default="dthink_pixel_agent", help="agent_registry 名称")
    p.add_argument("--select_sensors", type=str, nargs="+",
                   default=["rgb_front", "rgb_left", "rgb_right", "rgb_back"],
                   help="要保存/可视化的传感器名字列表（应与 obs 键一致）")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="生成时的最大 token 数（透传给 agent）")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="采样温度（透传给 agent）")
    p.add_argument("--top_p", type=float, default=0.6,
                   help="采样 top-p（透传给 agent）")
    p.add_argument("--noise_var", type=float, default=0.00,
                   help="对 obs 中 depth 图逐像素乘 N(1, noise_var) 噪声；设为 0 可关闭")
    return p.parse_args()



def main() -> None:
    args = parse_args()

    agent = agent_registry.create(
        args.agent_type,
        model_path=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    env = env_registry.create(
        args.env_type,
        args.env_config,
    )
        
    all_ids = [str(eid) for eid in env.get_episode_list()]
    total_ordered = len(all_ids)
    print(f"[Generate] total episodes to process: {total_ordered}")

    if total_ordered == 0:
        print("[Generate] No episodes available in dataset.")
        return

    end = args.max_episodes if args.max_episodes is not None else total_ordered
    end = min(end, total_ordered)
    start = 0

    per_rank = math.ceil((end - start) / world_size)
    start_idx = start + rank * per_rank
    end_idx = min(end, start + (rank + 1) * per_rank)
    if world_size > 1:
        print(
            f"[Dist] Rank {rank} handling episodes [{start_idx}:{end_idx})")
    env.slice_episodes(start_idx, end_idx)

    ensure_dir(args.save_root)

    agg_metrics: Dict[str, float] = defaultdict(float)
    count = 0

    for ep_idx in tqdm.tqdm(range(len(env.get_episode_list()))):
        env.reset()
        best = None
        for _ in range(args.retry_times):
            env.reset(False)
            metrics = evaluate(
                env=env,
                agent=agent,
                save_root=args.save_root,
                instruction=args.instruction,
                max_steps=args.max_steps,
                episode_prefix=args.episode_prefix,
                save_long_image=args.save_long_image,
                max_steps_per_move=args.max_steps_per_move,
                goal_radius=args.goal_radius,
                verbose=args.verbose,
                select_sensors=args.select_sensors,
                noise_var=args.noise_var,
            )
            if cmp_metrics(metrics, best):
                best = metrics

        if best is None:
            continue

        count += 1
        _add_metrics(agg_metrics, best)
        avg = {k: v / count for k, v in agg_metrics.items()}
        tqdm.tqdm.write(f"Episode {count}: {avg}")

    result_json = os.path.join(args.save_root, "result.json")
    save_json(result_json, {"avg_metrics": {k: v / max(count, 1)
                                            for k, v in agg_metrics.items()}})


if __name__ == "__main__":
    main()
