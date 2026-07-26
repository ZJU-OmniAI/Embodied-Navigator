from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if is_finite_number(v)]
    if not nums:
        return None
    return sum(nums) / float(len(nums))


def std(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if is_finite_number(v)]
    if len(nums) < 2:
        return 0.0 if nums else None
    mu = sum(nums) / float(len(nums))
    var = sum((x - mu) ** 2 for x in nums) / float(len(nums))
    return math.sqrt(var)


def rolling_mean(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return [safe_float(v) for v in values]
    out: List[Optional[float]] = []
    buf: List[float] = []
    for value in values:
        num = safe_float(value)
        if num is not None:
            buf.append(num)
        if len(buf) > window:
            buf = buf[-window:]
        if not buf:
            out.append(None)
        else:
            out.append(sum(buf) / float(len(buf)))
    return out


def rolling_std(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return [0.0 if safe_float(v) is not None else None for v in values]
    out: List[Optional[float]] = []
    buf: List[float] = []
    for value in values:
        num = safe_float(value)
        if num is not None:
            buf.append(num)
        if len(buf) > window:
            buf = buf[-window:]
        out.append(std(buf))
    return out


def downsample_xy(points: Sequence[Sequence[Optional[float]]], max_points: int = 1200) -> List[List[Optional[float]]]:
    """
    Downsample XY points by bucket-mean on y and right-edge x.
    Keeps shape stable for large logs and browser rendering.
    """
    points = [list(p) for p in points if len(p) >= 2]
    if len(points) <= max_points:
        return points

    bucket = len(points) / float(max_points)
    out: List[List[Optional[float]]] = []
    for i in range(max_points):
        start = int(i * bucket)
        end = int((i + 1) * bucket)
        if end <= start:
            end = min(start + 1, len(points))
        chunk = points[start:end]
        if not chunk:
            continue

        x = chunk[-1][0]
        ys = [safe_float(item[1]) for item in chunk]
        y = mean(ys)
        out.append([x, y])
    return out


def categorical_window_ratio(
    labels: Sequence[str],
    categories: Sequence[str],
    window: int,
) -> Dict[str, List[float]]:
    if window <= 0:
        window = 1

    counters: Dict[str, List[float]] = {category: [] for category in categories}
    for start in range(0, len(labels), window):
        chunk = labels[start : start + window]
        count = Counter(chunk)
        denom = float(len(chunk)) if chunk else 1.0
        for category in categories:
            counters[category].append(100.0 * float(count.get(category, 0)) / denom)
    return counters


def histogram(values: Iterable[int]) -> Tuple[List[int], List[int]]:
    counter = defaultdict(int)
    for value in values:
        try:
            key = int(value)
        except Exception:
            continue
        counter[key] += 1
    keys = sorted(counter.keys())
    return keys, [counter[key] for key in keys]
