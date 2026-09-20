#!/usr/bin/env python3
"""
并行版本的 eval_full.py
将 134 个任务分成 N 份，每份用 1 个 GPU（tp=1）独立跑，最后合并结果

Usage:
  python eval_full_parallel.py \
      --model_path /path/to/huggingface \
      --test_data test_unseen.jsonl \
      --output eval_full_unseen.json \
      --n_workers 8 \        # 并行进程数（= GPU 数量）
      --temperature 0.0
"""

import argparse, json, os, sys, time, math, subprocess, tempfile
from pathlib import Path

PROBE_DIR = Path(__file__).parent
EVAL_SCRIPT = str(PROBE_DIR / "eval_full.py")

def split_tasks(tasks, n):
    """将任务列表均分成 n 份"""
    size = math.ceil(len(tasks) / n)
    return [tasks[i*size:(i+1)*size] for i in range(n) if tasks[i*size:(i+1)*size]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcod_root", required=True,
                        help="Root of this repository (contains trinity/); passed through to eval_full.py")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_workers", type=int, default=8,
                        help="并行进程数（每个进程使用 1 个 GPU，tp=1）")
    parser.add_argument("--max_env_steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    # 读取任务列表
    tasks = []
    with open(args.test_data) as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get("game_file"):
                tasks.append(d["game_file"])
    n = min(args.n_workers, len(tasks))
    chunks = split_tasks(tasks, n)
    print(f"[eval_parallel] {len(tasks)} tasks → {n} workers × ~{len(chunks[0])} tasks | tp=1 each")

    # 为每个 worker 创建临时任务文件和输出文件
    tmp_dir = tempfile.mkdtemp()
    procs = []
    out_files = []

    for i, chunk in enumerate(chunks):
        # 写分片任务文件
        chunk_data = args.test_data  # 保留原文件格式
        chunk_jsonl = os.path.join(tmp_dir, f"chunk_{i}.jsonl")
        with open(chunk_jsonl, 'w') as f:
            with open(args.test_data) as src:
                for line in src:
                    d = json.loads(line.strip())
                    if d.get("game_file") in chunk:
                        f.write(line)

        out_file = os.path.join(tmp_dir, f"result_{i}.json")
        out_files.append(out_file)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        cmd = [
            sys.executable, EVAL_SCRIPT,
            "--tcod_root", args.tcod_root,
            "--model_path", args.model_path,
            "--test_data", chunk_jsonl,
            "--output", out_file,
            "--tp", "1",
            "--max_env_steps", str(args.max_env_steps),
            "--temperature", str(args.temperature),
            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        ]
        print(f"[eval_parallel] GPU {i}: {len(chunk)} tasks")
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    # 等待所有进程完成
    t0 = time.time()
    for i, p in enumerate(procs):
        p.wait()
        if p.returncode != 0:
            print(f"[eval_parallel] WARNING: worker {i} exited with code {p.returncode}")

    print(f"[eval_parallel] All workers done in {time.time()-t0:.0f}s")

    # 合并结果
    all_results = []
    for out_file in out_files:
        if os.path.exists(out_file):
            with open(out_file) as f:
                d = json.load(f)
                all_results.extend(d.get("results", []))
        else:
            print(f"[eval_parallel] WARNING: {out_file} not found")

    n_success = sum(1 for r in all_results if r.get("success", False))
    sr = n_success / len(all_results) if all_results else 0.0

    summary = {
        "evaluator_script": str(Path(EVAL_SCRIPT).resolve()),
        "argv": sys.argv,
        "model_path": args.model_path,
        "test_data": args.test_data,
        "n_tasks": len(all_results),
        "n_success": n_success,
        "success_rate": sr,
        "temperature": args.temperature,
        "max_env_steps": args.max_env_steps,
        "n_workers": n,
        "results": all_results,
    }
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[eval_parallel] SR={sr:.1%}  ({n_success}/{len(all_results)})  → {args.output}")

if __name__ == "__main__":
    main()
