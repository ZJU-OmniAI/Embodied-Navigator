import argparse
import json
import os
import random
from typing import List
import concurrent.futures
from tqdm import tqdm

from openai import OpenAI

YUNWU_MODEL_NAME = "gemini-2.5-flash"

OPTIMIZE_SYSTEM_PROMPT = """You are an expert at rewriting chain-of-thought reasoning.

Given a three-stage chain-of-thought in English, rewrite it into a single, coherent paragraph of English reasoning.

Requirements:
- Keep the original three-step logic (recall the task and the history of actions → describe the current observations → infer the next step), but do NOT include any stage titles.
- Preserve the necessary spatial information (such as front, back, left, right, object locations, and whether different paths are traversable).
- Remove repeated, redundant, or overly verbose expressions so that the reasoning becomes more concise and logically accurate.
- Output only the optimized English reasoning as ONE single paragraph, without any additional explanations.
"""

def call_yunwu_gemini_flash(
    cot: str,
    clients: List[OpenAI],
    model_name: str,
    retries: int = 4,
) -> str:
    last_err = None
    for _ in range(retries):
        try:
            client = random.choice(clients)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": OPTIMIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": cot},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
    raise last_err


def optimize_chain(cot: str, clients: List[OpenAI], model_name: str) -> str:
    """
    对单条三段式思维链进行优化。
    """
    return call_yunwu_gemini_flash(cot, clients=clients, model_name=model_name)


def build_clients(base_url: str, api_keys: List[str]) -> List[OpenAI]:
    return [
        OpenAI(
            api_key=key,
            base_url=base_url,
        )
        for key in api_keys
    ]

def polish_dataset(
    input_path: str,
    output_path: str,
    clients: List[OpenAI],
    model_name: str,
    max_workers: int = 8,
    save_every: int = 1000,
) -> None:
    with open(input_path, "r") as f:
        data = json.load(f)

    total_steps = 0
    for episodes in data.values():
        for ep in episodes:
            total_steps += len(ep.get("step", []))

    processed = 0
    pending: dict[concurrent.futures.Future, dict] = {}
    queue_limit = max_workers * 4

    def save_snapshot():
        with open(output_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def handle_future(fut):
        nonlocal processed
        step = pending.pop(fut)
        try:
            step["polish_cot"] = fut.result()
        except Exception as e:
            step["polish_cot"] = f"[ERROR: {e}]\n{step.get('cot', '')}"
        processed += 1
        progress.update(1)
        if processed % save_every == 0:
            save_snapshot()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=total_steps, desc="Polishing steps") as progress:
            for episodes in data.values():
                for ep in episodes:
                    for step in ep.get("step", []):
                        if not step.get("cot") or step.get("is_cot") != "Yes":
                            step["polish_cot"] = ""
                            processed += 1
                            progress.update(1)
                            if processed % save_every == 0:
                                save_snapshot()
                            continue

                        future = executor.submit(
                            optimize_chain,
                            step["cot"],
                            clients,
                            model_name,
                        )
                        pending[future] = step
                        if len(pending) >= queue_limit:
                            done_future = next(
                                concurrent.futures.as_completed(pending)
                            )
                            handle_future(done_future)

            for future in concurrent.futures.as_completed(pending):
                handle_future(future)

    save_snapshot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polish CoT fields in dataset json.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample.json",
        help="Path to input json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample.json",
        help="Path to save updated json (overwritten on each save).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Thread count for polishing.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50000,
        help="Save after this many processed steps.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="LLM API base URL, e.g. https://yunwu.ai/v1",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        required=True,
        help="API key. Repeat this argument to pass multiple keys.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=YUNWU_MODEL_NAME,
        help="Model name for chat completion.",
    )
    args = parser.parse_args()
    clients = build_clients(base_url=args.base_url, api_keys=args.api_key)

    polish_dataset(
        input_path=args.input,
        output_path=args.output,
        clients=clients,
        model_name=args.model_name,
        max_workers=args.max_workers,
        save_every=args.save_every,
    )
