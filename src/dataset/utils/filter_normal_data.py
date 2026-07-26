import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


"""
Filter trajectories in the original DThink polished dataset without changing the format.

Rules (mirrors logic used in format2habitat.py):
- Drop a trajectory if its reference_path (constructed from step.pos) contains
  duplicate points when rounded to 1e-4 precision.
- Keep only trajectories that contain at least one step with a non-trivial
  polish_cot (len > 10), as checked by _has_polish_cot.

Input/Output format is preserved: { scene_id: [trajectory, ...], ... }.
No field renaming or structure conversion is performed.
"""


DEFAULT_INPUT = Path("data/dthink_sft_data/polish_data_with_cots_sum_4_guide1.json")
DEFAULT_OUTPUT = Path("data/dthink_sft_data/polish_data_with_cots_sum_4_guide1_filtered.json")


def _has_polish_cot(steps: List[Dict[str, Any]]) -> bool:
    """Return True if any step contains a non-empty polish_cot (>10 chars)."""
    for step in steps:
        if "polish_cot" in step and len(str(step["polish_cot"])) > 10:
            return True
    return False


def _has_duplicate_reference_path(reference_path: List[List[float]]) -> bool:
    """Detect duplicate coordinates in the reference path at 1e-4 precision.

    Same logic as src/dataset/format2habitat.py to ensure consistent filtering.
    """
    seen = set()
    for pos in reference_path:
        rounded = tuple(round(float(coord), 4) for coord in pos)
        if rounded in seen:
            return True
        seen.add(rounded)
    return False


def filter_dataset(input_path: Path, output_path: Path) -> Dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw: Dict[str, List[Dict[str, Any]]] = json.load(f)

    total_traj = 0
    kept_traj = 0
    skipped_duplicate_path = 0
    skipped_no_polish = 0

    filtered: Dict[str, List[Dict[str, Any]]] = {}

    for scene_id, trajectories in raw.items():
        out_list: List[Dict[str, Any]] = []
        for traj_idx, traj in enumerate(trajectories):
            total_traj += 1
            steps: List[Dict[str, Any]] = traj.get("step", [])
            if not steps:
                # Empty steps are considered invalid
                skipped_no_polish += 1
                continue

            reference_path = [step.get("pos") for step in steps if "pos" in step]
            if not reference_path or _has_duplicate_reference_path(reference_path):
                # Duplicate or malformed path -> drop
                skipped_duplicate_path += 1
                continue

            # Require at least one polish_cot
            if not _has_polish_cot(steps):
                skipped_no_polish += 1
                continue

            out_list.append(traj)
            kept_traj += 1

        if out_list:
            filtered[scene_id] = out_list

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(
        f"Trajectories: {total_traj} original, {kept_traj} kept, "
        f"{skipped_no_polish} skipped (no polish_cot), "
        f"{skipped_duplicate_path} skipped (duplicate reference_path)"
    )
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter polished dataset without format conversion.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = filter_dataset(args.input, args.output)
    kept = sum(len(v) for v in result.values())
    print(f"Filtered dataset saved to {args.output} with {kept} trajectories kept.")
