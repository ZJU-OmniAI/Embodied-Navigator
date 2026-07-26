from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

try:
    from .log_loader import LogBundle
    from .transforms import (
        categorical_window_ratio,
        downsample_xy,
        histogram,
        mean,
        rolling_mean,
        safe_float,
        std,
    )
except ImportError:
    from log_loader import LogBundle
    from transforms import (
        categorical_window_ratio,
        downsample_xy,
        histogram,
        mean,
        rolling_mean,
        safe_float,
        std,
    )


POLICY_ORDER = ["best", "second", "worst", "random"]
STEP_REWARD_KEYS = [
    "collision_reward",
    "target_approach_reward",
    "reasoning_reward",
    "stop_reward",
    "format_step_reward",
]


def _finite(value: Optional[float]) -> Optional[float]:
    return safe_float(value)


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float_list(value: Any) -> List[Optional[float]]:
    if not isinstance(value, list):
        return []
    return [safe_float(item) for item in value]


def _sum_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [value for value in values if value is not None]
    if not nums:
        return None
    return float(sum(nums))


def _selected_value(values: Sequence[Optional[float]], idx: int) -> Optional[float]:
    if idx < 0 or idx >= len(values):
        return None
    return values[idx]


@dataclass
class DashboardContext:
    bundle: LogBundle
    ranks: Optional[Set[int]]

    def records(self, log_type: str):
        return self.bundle.records(log_type, self.ranks)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_selected_rows(ctx: DashboardContext) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in ctx.records("weighted_target_approach_step_select"):
        payload = record.payload
        picked_index = _as_int(payload.get("picked_index"), -1)
        policy = str(payload.get("policy") or "unknown")
        step_id = _as_int(payload.get("step_id"), -1)
        candidate_count = _as_int(payload.get("candidate_count"), 0)

        reward_dict_raw = payload.get("reward_dict") if isinstance(payload.get("reward_dict"), dict) else {}
        reward_dict: Dict[str, List[Optional[float]]] = {
            key: _as_float_list(value) for key, value in reward_dict_raw.items()
        }

        selected_rewards: Dict[str, Optional[float]] = {}
        for key in STEP_REWARD_KEYS:
            selected_rewards[key] = _selected_value(reward_dict.get(key, []), picked_index)

        advantages = _as_float_list(payload.get("advantages"))
        target_scores = _as_float_list(payload.get("target_scores"))
        selected_advantage = _selected_value(advantages, picked_index)
        selected_target = _selected_value(target_scores, picked_index)
        best_target = max([v for v in target_scores if v is not None], default=None)

        selected_total = _sum_optional(list(selected_rewards.values()))
        row = {
            "rank": record.rank,
            "ts": record.ts_unix,
            "step_id": step_id,
            "policy": policy,
            "candidate_count": candidate_count,
            "picked_index": picked_index,
            "selected_rewards": selected_rewards,
            "selected_total_reward": selected_total,
            "selected_advantage": selected_advantage,
            "target_scores": target_scores,
            "target_std": std(target_scores),
            "advantage_std": std(advantages),
            "selected_target": selected_target,
            "target_gap_to_best": None if (best_target is None or selected_target is None) else float(best_target - selected_target),
        }
        rows.append(row)

    rows.sort(key=lambda item: ((item.get("ts") or 10**12), item.get("rank", 0)))
    for idx, row in enumerate(rows, start=1):
        row["interaction_idx"] = idx
    return rows


def extract_success_samples(ctx: DashboardContext) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for record in ctx.records("success_reward"):
        details = record.payload.get("details")
        if not isinstance(details, list):
            continue
        for item in details:
            if not isinstance(item, dict):
                continue
            success = 1.0 if bool(item.get("success")) else 0.0
            samples.append(
                {
                    "rank": record.rank,
                    "ts": record.ts_unix,
                    "success": success,
                }
            )
    return samples


def extract_spl_samples(ctx: DashboardContext) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for record in ctx.records("trajectory_similarity_reward"):
        details = record.payload.get("details")
        if not isinstance(details, list):
            continue
        for item in details:
            if not isinstance(item, dict):
                continue
            spl = _finite(item.get("spl"))
            if spl is None:
                continue
            samples.append({"rank": record.rank, "ts": record.ts_unix, "spl": spl})
    return samples


def extract_reasoning_ratio_samples(ctx: DashboardContext) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for record in ctx.records("reasoning_density_reward"):
        details = record.payload.get("details")
        if not isinstance(details, list):
            continue
        for item in details:
            if not isinstance(item, dict):
                continue
            ratio = _finite(item.get("reasoning_ratio"))
            if ratio is None:
                continue
            samples.append({"rank": record.rank, "ts": record.ts_unix, "reasoning_ratio": ratio})
    return samples


def extract_stop_rate_by_step(ctx: DashboardContext) -> Dict[int, float]:
    counter_total: Dict[int, int] = defaultdict(int)
    counter_stop: Dict[int, int] = defaultdict(int)
    for record in ctx.records("stop_reward"):
        step_id = _as_int(record.payload.get("step_id"), -1)
        if step_id < 0:
            continue
        is_stop = bool(record.payload.get("is_stop"))
        counter_total[step_id] += 1
        if is_stop:
            counter_stop[step_id] += 1

    out: Dict[int, float] = {}
    for step_id in sorted(counter_total.keys()):
        total = counter_total[step_id]
        stop_count = counter_stop.get(step_id, 0)
        out[step_id] = 100.0 * float(stop_count) / float(total) if total else 0.0
    return out


def extract_action_choice_counts(ctx: DashboardContext) -> Counter:
    counts: Counter = Counter()
    for record in ctx.records("rollout_step"):
        action = record.payload.get("action") if isinstance(record.payload.get("action"), dict) else {}
        action_type = str(action.get("type") or "UNKNOWN").upper()
        if action_type == "CMD" and str(action.get("action") or "").upper() == "STOP":
            key = "STOP"
        elif action.get("choice"):
            key = str(action.get("choice"))
        else:
            key = action_type
        counts[key] += 1
    return counts


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def build_selected_reward_chart(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = [row["interaction_idx"] for row in rows]
    series = []
    color_map = {
        "collision_reward": "#00a3a3",
        "target_approach_reward": "#ff7f11",
        "reasoning_reward": "#f45b69",
        "stop_reward": "#2f4f4f",
        "format_step_reward": "#6b8e23",
        "selected_total_reward": "#2b2d42",
        "selected_advantage": "#ff4d6d",
    }

    for key in STEP_REWARD_KEYS:
        points = [[xx, _finite(row["selected_rewards"].get(key))] for xx, row in zip(x, rows)]
        series.append(
            {
                "name": key,
                "points": downsample_xy(points, max_points=1600),
                "smoothable": True,
                "color": color_map.get(key),
            }
        )

    total_points = [[xx, _finite(row.get("selected_total_reward"))] for xx, row in zip(x, rows)]
    adv_points = [[xx, _finite(row.get("selected_advantage"))] for xx, row in zip(x, rows)]
    series.append(
        {
            "name": "selected_total_reward",
            "points": downsample_xy(total_points, max_points=1600),
            "smoothable": True,
            "color": color_map.get("selected_total_reward"),
            "line_width": 2.2,
        }
    )
    series.append(
        {
            "name": "selected_advantage",
            "points": downsample_xy(adv_points, max_points=1600),
            "smoothable": True,
            "color": color_map.get("selected_advantage"),
            "line_width": 1.8,
            "line_style": "dashed",
        }
    )

    return {
        "id": "selected_reward_timeline",
        "title": "Selected Reward Timeline",
        "description": "Per interaction-step selected candidate reward decomposition and advantage.",
        "kind": "line",
        "x_label": "Interaction Step",
        "y_label": "Reward",
        "options": {
            "area": False,
            "stack": False,
            "baseline_zero": True,
        },
        "series": series,
    }


def build_policy_mix_chart(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    policies = [str(row.get("policy") or "unknown") for row in rows]
    if not policies:
        ratios = {name: [] for name in POLICY_ORDER}
        x_points: List[int] = []
    else:
        window = max(20, len(policies) // 120)
        ratios = categorical_window_ratio(policies, POLICY_ORDER, window)
        x_points = [min((idx + 1) * window, len(policies)) for idx in range(len(next(iter(ratios.values()), [])))]

    series = []
    color_map = {
        "best": "#2a9d8f",
        "second": "#8ab17d",
        "worst": "#e76f51",
        "random": "#457b9d",
    }
    for name in POLICY_ORDER:
        points = [[x, _finite(y)] for x, y in zip(x_points, ratios.get(name, []))]
        series.append(
            {
                "name": name,
                "points": points,
                "smoothable": False,
                "color": color_map.get(name),
            }
        )

    return {
        "id": "policy_mix_timeline",
        "title": "Step Select Policy Mix",
        "description": "Windowed percentage of step-select policy (best/second/worst/random).",
        "kind": "line",
        "x_label": "Interaction Step",
        "y_label": "Policy Ratio (%)",
        "options": {
            "area": True,
            "stack": True,
            "percent": True,
            "y_min": 0,
            "y_max": 100,
        },
        "series": series,
    }


def build_traj_quality_chart(
    success_samples: List[Dict[str, Any]],
    spl_samples: List[Dict[str, Any]],
    reasoning_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    success_vals = [item["success"] for item in success_samples]
    spl_vals = [item["spl"] for item in spl_samples]
    reasoning_vals = [item["reasoning_ratio"] for item in reasoning_samples]

    window = 64
    success_roll = rolling_mean(success_vals, window=window)
    spl_roll = rolling_mean(spl_vals, window=window)
    reasoning_roll = rolling_mean(reasoning_vals, window=window)

    max_len = max(len(success_roll), len(spl_roll), len(reasoning_roll), 0)

    def _build_points(values: Sequence[Optional[float]]) -> List[List[Optional[float]]]:
        points = [[idx + 1, _finite(values[idx] if idx < len(values) else None)] for idx in range(max_len)]
        return downsample_xy(points, max_points=1400)

    return {
        "id": "trajectory_quality",
        "title": "Trajectory Quality (Rolling)",
        "description": f"Rolling mean (window={window}) over trajectory-level metrics.",
        "kind": "line",
        "x_label": "Trajectory Sample",
        "y_label": "Score",
        "options": {
            "area": False,
            "stack": False,
            "y_min": 0,
            "y_max": 1,
        },
        "series": [
            {
                "name": "success_rate_roll",
                "points": _build_points(success_roll),
                "smoothable": False,
                "color": "#2a9d8f",
                "line_width": 2.2,
            },
            {
                "name": "spl_roll",
                "points": _build_points(spl_roll),
                "smoothable": False,
                "color": "#f4a261",
                "line_width": 2.0,
            },
            {
                "name": "reasoning_ratio_roll",
                "points": _build_points(reasoning_roll),
                "smoothable": False,
                "color": "#577590",
                "line_width": 1.8,
            },
        ],
    }


def build_candidate_spread_chart(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = [row["interaction_idx"] for row in rows]
    target_std_points = [[xx, _finite(row.get("target_std"))] for xx, row in zip(x, rows)]
    adv_std_points = [[xx, _finite(row.get("advantage_std"))] for xx, row in zip(x, rows)]
    gap_points = [[xx, _finite(row.get("target_gap_to_best"))] for xx, row in zip(x, rows)]

    return {
        "id": "candidate_spread",
        "title": "Candidate Spread",
        "description": "Inter-candidate score dispersion and selected-vs-best target gap.",
        "kind": "line",
        "x_label": "Interaction Step",
        "y_label": "Spread",
        "options": {
            "area": False,
            "stack": False,
            "baseline_zero": True,
        },
        "series": [
            {
                "name": "target_score_std",
                "points": downsample_xy(target_std_points, max_points=1500),
                "smoothable": True,
                "color": "#003049",
            },
            {
                "name": "advantage_std",
                "points": downsample_xy(adv_std_points, max_points=1500),
                "smoothable": True,
                "color": "#d62828",
            },
            {
                "name": "selected_gap_to_best",
                "points": downsample_xy(gap_points, max_points=1500),
                "smoothable": True,
                "color": "#f77f00",
            },
        ],
    }


def build_step_depth_chart(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    step_ids = [row.get("step_id") for row in rows if isinstance(row.get("step_id"), int) and row.get("step_id") >= 0]
    xs, ys = histogram(step_ids)
    points = [[x, y] for x, y in zip(xs, ys)]
    return {
        "id": "selected_step_depth",
        "title": "Selected Step Depth Distribution",
        "description": "How often selected interaction happens at each rollout step_id.",
        "kind": "bar",
        "x_label": "Step ID",
        "y_label": "Count",
        "options": {},
        "series": [
            {
                "name": "selected_count",
                "points": points,
                "color": "#264653",
            }
        ],
    }


def build_stop_rate_chart(stop_rate_by_step: Dict[int, float]) -> Dict[str, Any]:
    points = [[step_id, _finite(rate)] for step_id, rate in sorted(stop_rate_by_step.items())]
    return {
        "id": "stop_rate_by_depth",
        "title": "STOP Rate By Step Depth",
        "description": "Candidate-level STOP action ratio for each step_id (from stop_reward).",
        "kind": "line",
        "x_label": "Step ID",
        "y_label": "STOP Rate (%)",
        "options": {
            "area": False,
            "stack": False,
            "y_min": 0,
            "y_max": 100,
        },
        "series": [
            {
                "name": "stop_rate",
                "points": points,
                "smoothable": False,
                "color": "#e76f51",
                "line_width": 2.0,
            }
        ],
    }


def build_rank_overview_chart(
    rows: List[Dict[str, Any]],
    success_samples: List[Dict[str, Any]],
    spl_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    per_rank_selected: Dict[int, List[Optional[float]]] = defaultdict(list)
    for row in rows:
        rank = _as_int(row.get("rank"), -1)
        if rank < 0:
            continue
        per_rank_selected[rank].append(_finite(row.get("selected_total_reward")))

    per_rank_success: Dict[int, List[Optional[float]]] = defaultdict(list)
    for item in success_samples:
        rank = _as_int(item.get("rank"), -1)
        if rank < 0:
            continue
        per_rank_success[rank].append(_finite(item.get("success")))

    per_rank_spl: Dict[int, List[Optional[float]]] = defaultdict(list)
    for item in spl_samples:
        rank = _as_int(item.get("rank"), -1)
        if rank < 0:
            continue
        per_rank_spl[rank].append(_finite(item.get("spl")))

    ranks = sorted(set(per_rank_selected) | set(per_rank_success) | set(per_rank_spl))

    selected_points = [[rank, mean(per_rank_selected.get(rank, []))] for rank in ranks]
    success_points = [[rank, mean(per_rank_success.get(rank, []))] for rank in ranks]
    spl_points = [[rank, mean(per_rank_spl.get(rank, []))] for rank in ranks]

    return {
        "id": "rank_overview",
        "title": "Rank Overview",
        "description": "Per-rank mean selected reward, success rate, and SPL.",
        "kind": "bar",
        "x_label": "Rank",
        "y_label": "Mean Value",
        "options": {
            "grouped": True,
        },
        "series": [
            {
                "name": "mean_selected_reward",
                "points": selected_points,
                "color": "#1d3557",
            },
            {
                "name": "success_rate",
                "points": success_points,
                "color": "#2a9d8f",
            },
            {
                "name": "mean_spl",
                "points": spl_points,
                "color": "#f4a261",
            },
        ],
    }


def build_action_mix_chart(action_counts: Counter) -> Dict[str, Any]:
    top_items = action_counts.most_common(12)
    points = [[name, count] for name, count in top_items]
    return {
        "id": "action_mix",
        "title": "Action Choice Mix",
        "description": "Top action choices from rollout candidates.",
        "kind": "bar",
        "x_label": "Action",
        "y_label": "Count",
        "options": {
            "x_category": True,
        },
        "series": [
            {
                "name": "count",
                "points": points,
                "color": "#457b9d",
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Summary + Dashboard
# --------------------------------------------------------------------------- #
def build_cards(
    ctx: DashboardContext,
    rows: List[Dict[str, Any]],
    success_samples: List[Dict[str, Any]],
    spl_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rollout_count = len(ctx.records("rollout_step"))
    selected_count = len(rows)
    success_rate = mean([sample.get("success") for sample in success_samples])
    mean_spl = mean([sample.get("spl") for sample in spl_samples])
    mean_selected_reward = mean([row.get("selected_total_reward") for row in rows])

    format_records = ctx.records("format_step_reward")
    format_valid = mean([1.0 if bool(record.payload.get("valid")) else 0.0 for record in format_records])

    stop_records = ctx.records("stop_reward")
    stop_rate = mean([1.0 if bool(record.payload.get("is_stop")) else 0.0 for record in stop_records])

    cards = [
        {
            "id": "candidates",
            "label": "Candidate Rollouts",
            "value": rollout_count,
            "format": "int",
        },
        {
            "id": "interactions",
            "label": "Selected Interactions",
            "value": selected_count,
            "format": "int",
        },
        {
            "id": "success_rate",
            "label": "Success Rate",
            "value": success_rate,
            "format": "percent",
        },
        {
            "id": "mean_spl",
            "label": "Mean SPL",
            "value": mean_spl,
            "format": "float3",
        },
        {
            "id": "selected_reward",
            "label": "Mean Selected Reward",
            "value": mean_selected_reward,
            "format": "float3",
        },
        {
            "id": "format_valid",
            "label": "Format Valid Rate",
            "value": format_valid,
            "format": "percent",
        },
        {
            "id": "stop_rate",
            "label": "STOP Action Rate",
            "value": stop_rate,
            "format": "percent",
        },
    ]
    return cards


def build_dashboard(bundle: LogBundle, ranks: Optional[Set[int]] = None) -> Dict[str, Any]:
    ctx = DashboardContext(bundle=bundle, ranks=ranks)

    selected_rows = extract_selected_rows(ctx)
    success_samples = extract_success_samples(ctx)
    spl_samples = extract_spl_samples(ctx)
    reasoning_samples = extract_reasoning_ratio_samples(ctx)
    stop_rate_by_step = extract_stop_rate_by_step(ctx)
    action_choice_counts = extract_action_choice_counts(ctx)

    charts = [
        build_selected_reward_chart(selected_rows),
        build_policy_mix_chart(selected_rows),
        build_traj_quality_chart(success_samples, spl_samples, reasoning_samples),
        build_candidate_spread_chart(selected_rows),
        build_step_depth_chart(selected_rows),
        build_stop_rate_chart(stop_rate_by_step),
        build_rank_overview_chart(selected_rows, success_samples, spl_samples),
        build_action_mix_chart(action_choice_counts),
    ]

    filtered_counts = {
        log_type: len(bundle.records(log_type, ranks))
        for log_type in bundle.available_types
    }

    return {
        "meta": {
            "log_dir": str(bundle.log_dir),
            "ranks": bundle.available_ranks if ranks is None else sorted(ranks),
            "record_counts": filtered_counts,
        },
        "cards": build_cards(ctx, selected_rows, success_samples, spl_samples),
        "charts": charts,
    }


# --------------------------------------------------------------------------- #
# Extension notes:
# Add a chart by creating a `build_xxx_chart(...)` function and append it in
# `build_dashboard(...)->charts` list.
# --------------------------------------------------------------------------- #
