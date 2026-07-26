import argparse
import copy
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List
import tqdm

"""
Convert polished DThink trajectories into the Habitat dataset episode format.

Input:  data/dthink_sft_data/sample_polished.json
Output: data/dthink_sft_data/sample_polished_habitat.json (+ .gz)
"""

INPUT_PATH = Path("data/dthink_sft_data/sample_polished.json")
OUTPUT_JSON = Path("data/dthink_sft_data/sample_polished_habitat.json")
OUTPUT_GZ = OUTPUT_JSON.with_suffix(OUTPUT_JSON.suffix + ".gz")

EPISODE_TEMPLATE = {
    "episode_id": -1,
    "trajectory_id": -1,
    "scene_id": "",
    "start_position": [],
    "start_rotation": [],
    "goals": [
        {
            "position": [],
            "radius": 1.5,
        }
    ],
    "instruction": {"instruction_text": ""},
    "reference_path": [],
    "reasoning": [],
}


def _has_reasoning(step: Dict[str, Any]) -> bool:
    flag = step.get("is_cot", "")
    if isinstance(flag, str):
        return flag.strip().lower() == "yes"
    return bool(flag)


def _has_polish_cot(steps: List[Dict[str, Any]]) -> bool:
    for step in steps:
        if "polish_cot" in step and len(str(step["polish_cot"])) > 10:
            return True
    return False


def _has_duplicate_reference_path(reference_path: List[List[float]]) -> bool:
    seen = set()
    for pos in reference_path:
        rounded = tuple(round(float(coord), 4) for coord in pos)
        if rounded in seen:
            return True
        seen.add(rounded)
    return False


def convert(input_path: Path, output_json: Path, output_gz: Path) -> Dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    res: Dict[str, Any] = {"episodes": [], "instruction_vocab": {"word_list": []}}
    episode_id = 1
    total_traj = 0
    kept_traj = 0
    skipped_no_polish = 0
    skipped_duplicate_path = 0

    for scene_id, trajectories in tqdm.tqdm(raw.items()):
        for traj_idx, traj in enumerate(trajectories):
            total_traj += 1
            steps: List[Dict[str, Any]] = traj.get("step", [])
            if not steps:
                continue
            if not _has_polish_cot(steps):
                print(
                    f"Skip trajectory {traj_idx} in scene {scene_id}: missing valid polish_cot"
                )
                skipped_no_polish += 1
                continue
            
            episode = copy.deepcopy(EPISODE_TEMPLATE)
            episode["episode_id"] = episode_id
            episode["trajectory_id"] = traj_idx
            episode["scene_id"] = scene_id
            episode["start_position"] = steps[0]["pos"]
            episode["start_rotation"] = steps[0]["rot"]
            episode["goals"][0]["position"] = steps[-1]["pos"]
            episode["instruction"]["instruction_text"] = traj.get("ins", "")
            episode["reference_path"] = [step["pos"] for step in steps]
            if _has_duplicate_reference_path(episode["reference_path"]):
                print(
                    f"Skip trajectory {traj_idx} in scene {scene_id}: duplicate points in reference_path at 1e-4 precision"
                )
                skipped_duplicate_path += 1
                continue

            reasoning: List[Any] = []
            for step in steps:
                if _has_reasoning(step):
                    reasoning.append(
                        {
                            "cot": step.get("cot", ""),
                            "polish_cot": step.get("polish_cot", ""),
                            "action": step.get("action", ""),
                        }
                    )
                else:
                    reasoning.append(None)
            episode["reasoning"] = reasoning

            res["episodes"].append(episode)
            episode_id += 1
            kept_traj += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    with gzip.open(output_gz, "wt", encoding="utf-8") as gz:
        json.dump(res, gz, ensure_ascii=False)

    print(
        f"Trajectories: {total_traj} original, {kept_traj} kept, "
        f"{skipped_no_polish} skipped (no polish_cot), "
        f"{skipped_duplicate_path} skipped (duplicate reference_path)"
    )
    return res


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert polished trajectories to Habitat episodes."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Input JSON path")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=OUTPUT_JSON,
        help="Where to save the Habitat-style JSON",
    )
    parser.add_argument(
        "--output_gz",
        type=Path,
        default=OUTPUT_GZ,
        help="Where to save the gzip copy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    converted = convert(args.input, args.output_json, args.output_gz)
    print(f"Converted {len(converted['episodes'])} episodes to {args.output_json}")
