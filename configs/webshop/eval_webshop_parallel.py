#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebShop parallel evaluation: n_workers × tp=1.

Compared to eval_webshop.py (tp=8, serial):
  - Each worker handles a subset of sessions
  - Workers run on separate GPUs (CUDA_VISIBLE_DEVICES)
  - ~8× faster for 100 sessions
  - ~8-12 min total (vs 25-35 min serial)

Usage:
  python eval_webshop_parallel.py \
      --model_path /path/to/hf \
      --webshop_path /path/to/webshop \   # or set WEBSHOP_PATH
      --output results.json \
      --n_workers 8 \
      --n_sessions 100 \
      --session_start 4096 \
      --temperature 0.0
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from multiprocessing import Process

WEBSHOP_DIR = Path(__file__).parent
TCOD_DIR = WEBSHOP_DIR.parent.parent.parent
sys.path.insert(0, str(TCOD_DIR))


def _setup_runtime(webshop_path: str, java_home: str):
    """Configure WEBSHOP_PATH / JAVA_HOME before creating the env."""
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = f"{java_home}/bin:{os.environ.get('PATH', '')}"
    if webshop_path:
        sys.path.insert(0, webshop_path)


def worker_fn(worker_id: int, gpu_id: int, model_path: str,
              session_ids: list, temperature: float, max_env_steps: int,
              result_path: str, webshop_path: str):
    """Worker: runs on one GPU, evaluates assigned sessions."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Local imports after setting CUDA
    from vllm import LLM, SamplingParams
    from web_agent_site.envs import WebAgentTextEnv

    # Import eval utilities from eval_webshop
    sys.path.insert(0, str(WEBSHOP_DIR))
    from eval_webshop import run_one_session, _create_webshop_env_full

    from trinity.common.workflows.envs.TCOD.webshop.utils import (
        WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS, HISTORY_LENGTH,
        parse_action, validate_action, format_observation,
        _format_available_actions, _format_history,
    )

    print(f"[worker {worker_id}] GPU={gpu_id}, {len(session_ids)} sessions, loading model...")

    class LocalModel:
        def __init__(self):
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                dtype="bfloat16",
                enforce_eager=True,
                gpu_memory_utilization=0.85,
                trust_remote_code=True,
            )
            self.tokenizer = self.llm.get_tokenizer()

        def chat(self, messages, temperature=0.0, max_tokens=512):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outputs = self.llm.generate([prompt], params)
            return outputs[0].outputs[0].text

    model = LocalModel()

    print(f"[worker {worker_id}] loading WebShop env (1.18M products, ~3-5 min)...")
    env = _create_webshop_env_full(webshop_path)
    print(f"[worker {worker_id}] env ready. Starting evaluation...")

    results = []
    for i, sid in enumerate(session_ids):
        result = run_one_session(env, model, sid, max_env_steps, temperature)
        results.append(result)
        if (i + 1) % 5 == 0:
            avg_r = sum(r["reward"] for r in results) / len(results)
            print(f"[worker {worker_id}] {i+1}/{len(session_ids)}  avg_reward={avg_r:.3f}")

    env.close()

    with open(result_path, "w") as f:
        json.dump(results, f)
    print(f"[worker {worker_id}] done. Saved {len(results)} results to {result_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--webshop_path", default=os.environ.get("WEBSHOP_PATH", ""),
                        help="Path to your cloned WebShop repository (contains web_agent_site/ "
                             "and data/items_shuffle.json). Can also be set via the WEBSHOP_PATH env var.")
    parser.add_argument("--java_home", default=os.environ.get("JAVA_HOME", ""),
                        help="Path to a Java 11+ home. Defaults to the JAVA_HOME env var.")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--n_sessions", type=int, default=100)
    parser.add_argument("--session_start", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_env_steps", type=int, default=15)
    args = parser.parse_args()

    if not args.webshop_path:
        sys.exit("ERROR: --webshop_path (or the WEBSHOP_PATH env var) is required. "
                 "Set it to your cloned WebShop repository, which contains web_agent_site/ "
                 "and data/items_shuffle.json.")
    _setup_runtime(args.webshop_path, args.java_home)

    session_ids = list(range(args.session_start, args.session_start + args.n_sessions))
    n_workers = min(args.n_workers, len(session_ids))

    # Split sessions across workers
    chunks = [[] for _ in range(n_workers)]
    for i, sid in enumerate(session_ids):
        chunks[i % n_workers].append(sid)

    print(f"Parallel WebShop eval: {len(session_ids)} sessions, {n_workers} workers")
    for i, chunk in enumerate(chunks):
        print(f"  worker {i} (GPU {i}): {len(chunk)} sessions {chunk[0]}-{chunk[-1]}")

    # Temp files for worker results
    tmp_dir = tempfile.mkdtemp(prefix="webshop_eval_")
    result_paths = [os.path.join(tmp_dir, f"worker_{i}.json") for i in range(n_workers)]

    # Launch workers
    import time
    t0 = time.time()
    procs = []
    for i in range(n_workers):
        p = Process(
            target=worker_fn,
            args=(i, i, args.model_path, chunks[i], args.temperature,
                  args.max_env_steps, result_paths[i], args.webshop_path),
            daemon=True,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    elapsed = time.time() - t0
    print(f"\nAll workers done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Merge results
    all_results = []
    for path in result_paths:
        if os.path.exists(path):
            with open(path) as f:
                all_results.extend(json.load(f))
        else:
            print(f"WARNING: {path} missing (worker failed?)")

    if not all_results:
        print("ERROR: no results collected!")
        sys.exit(1)

    success_rate = sum(r["success"] for r in all_results) / len(all_results)
    avg_reward   = sum(r["reward"]  for r in all_results) / len(all_results)

    output = {
        "model_path": args.model_path,
        "n_sessions": len(all_results),
        "session_range": f"{session_ids[0]}-{session_ids[-1]}",
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "elapsed_seconds": elapsed,
        "n_workers": n_workers,
        "results": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults: SR={success_rate:.3f}  avg_reward={avg_reward:.3f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
