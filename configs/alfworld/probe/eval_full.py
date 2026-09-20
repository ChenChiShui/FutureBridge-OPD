#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALFWorld full evaluation on test tasks.

Usage:
  python eval_full.py \
      --tcod_root /path/to/this/repo \
      --model_path /path/to/checkpoint/actor/huggingface \
      --test_data  /path/to/test_unseen.jsonl \
      --output     /path/to/results.json \
      --tp 4 --max_env_steps 30 --temperature 0.4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcod_root", required=True,
                        help="Root of this repository (contains trinity/)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max_env_steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def run_one_task(
    model,
    game_file: str,
    max_env_steps: int,
    temperature: float,
) -> dict:
    from trinity.common.workflows.envs.TCOD.alfworld.utils import (
        _create_alfworld_env,
        format_observation,
        _extract_task,
        _format_history,
    )

    try:
        env = _create_alfworld_env(game_file)
    except Exception as e:
        return {
            "game_file": game_file,
            "success": False,
            "steps": 0,
            "error": str(e),
        }

    try:
        obs, info = env.reset()
        task_desc = _extract_task(obs)
    except Exception as e:
        env.close()
        return {
            "game_file": game_file,
            "success": False,
            "steps": 0,
            "error": str(e),
        }

    history: List[str] = []
    memory: List[dict] = []
    success = False
    steps = 0

    MAX_MEMORY_TURNS = 10

    for step in range(max_env_steps):
        admissible = info.get("admissible_commands", [])

        if admissible and isinstance(admissible[0], list):
            admissible = admissible[0]

        user_content = build_messages(
            format_observation(obs),
            admissible,
            history,
            task_desc,
            step,
        )[0]["content"]

        recent_memory = memory[-(MAX_MEMORY_TURNS * 2):]

        messages = recent_memory + [
            {"role": "user", "content": user_content}
        ]

        action, raw_response = model.generate_action(
            messages,
            temperature=temperature,
            enable_thinking=False,
        )

        if not action or action not in admissible:
            action = admissible[0] if admissible else "look"
            raw_response = f"<action>{action}</action>"

        memory.append({
            "role": "user",
            "content": user_content,
        })

        memory.append({
            "role": "assistant",
            "content": raw_response,
        })

        history.append(
            _format_history(
                format_observation(obs),
                step + 1,
                action,
            )
        )

        obs, reward, done, info = env.step(action)
        steps += 1

        if done:
            success = True
            break

    env.close()

    return {
        "game_file": game_file,
        "success": success,
        "steps": steps,
    }


def main():
    args = parse_args()

    sys.path.insert(0, args.tcod_root)
    sys.path.insert(0, str(Path(__file__).parent))

    from probe_model import ProbeModel, build_messages

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}"
        )

    tasks = []

    with open(args.test_data) as f:
        for line in f:
            record = json.loads(line.strip())

            if record.get("game_file"):
                tasks.append(record["game_file"])

    model = ProbeModel(
        str(model_path),
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=12288,
        max_gen_tokens=512,
        enforce_eager=True,
    )

    results = []
    n_success = 0
    start_time = time.time()

    for i, game_file in enumerate(tasks):
        result = run_one_task(
            model,
            game_file,
            args.max_env_steps,
            args.temperature,
        )

        results.append(result)
        n_success += int(result["success"])

        success_rate = n_success / (i + 1)

        print(
            f"[{i + 1}/{len(tasks)}] "
            f"success={result['success']} "
            f"steps={result['steps']} "
            f"| SR={success_rate:.3f} "
            f"| {time.time() - start_time:.0f}s"
        )

    model.unload()

    success_rate = (
        n_success / len(tasks)
        if tasks
        else 0.0
    )

    output = {
        "evaluator_script": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "parsed_args": vars(args).copy(),
        "model_path": str(model_path),
        "test_data": args.test_data,
        "n_tasks": len(tasks),
        "n_success": n_success,
        "success_rate": success_rate,
        "temperature": args.temperature,
        "max_env_steps": args.max_env_steps,
        "generation_config": {
            "temperature": args.temperature,
            "enable_thinking": False,
            "max_gen_tokens": 512,
            "max_model_len": 12288,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"Success rate: {success_rate:.3f} "
        f"({n_success}/{len(tasks)})"
    )

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
