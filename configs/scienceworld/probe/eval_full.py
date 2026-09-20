#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScienceWorld full evaluation after training.

Usage:
  python eval_full.py \
      --model_path /path/to/checkpoint/actor/huggingface \
      --test_data  /path/to/sciworld_data/test.jsonl \
      --output     /path/to/results.json \
      --tp 1 --max_env_steps 30 --temperature 0.0

Score metric: avg normalized score (0-1), success = score >= 1.0
"""

import argparse, json, os, sys, time
from pathlib import Path

PROBE_DIR = Path(__file__).parent
TCOD_DIR  = PROBE_DIR.parent.parent.parent.parent
sys.path.insert(0, str(TCOD_DIR))


def _setup_java(java_home: str):
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = f"{java_home}/bin:{os.environ.get('PATH', '')}"


from vllm import LLM, SamplingParams
from trinity.common.workflows.envs.TCOD.scienceworld.utils import (
    _create_scienceworld_env,
    _reset_scienceworld_env,
    _get_compact_action_info,
    _format_history,
    format_observation,
    parse_action,
    HISTORY_LENGTH,
    SCIWORLD_TEMPLATE,
    SCIWORLD_TEMPLATE_NO_HIS,
    SCIWORLD_SYSTEM_PROMPT,
)


class SciProbeModel:
    def __init__(self, model_path, tp=1, gpu_util=0.85, max_model_len=12288):
        print(f"[SciProbeModel] Loading {model_path} tp={tp}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_model_len,
            enforce_eager=True,
            dtype="bfloat16",
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("[SciProbeModel] Loaded.")

    def generate(self, messages, temperature=0.0, max_tokens=512):
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                stop=["</action>"])
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        outs = self.llm.generate([prompt], sampling_params=params)
        raw = outs[0].outputs[0].text
        if "<action>" in raw and "</action>" not in raw:
            raw += "</action>"
        return parse_action(raw), raw


def run_episode(model, task_config, max_env_steps=30, temperature=0.0):
    try:
        env = _create_scienceworld_env(task_config, max_env_steps=max_env_steps)
    except Exception as e:
        return {"task": task_config, "score": 0.0, "success": False,
                "steps": 0, "error": str(e)}

    try:
        observation, info, task_description = _reset_scienceworld_env(env)
        best_score = info.get("score", 0.0)  # raw ScienceWorld score, 0-100 scale
        history = []
        system_msg = [{"role": "system", "content": SCIWORLD_SYSTEM_PROMPT}]

        for step in range(max_env_steps):
            fmt_obs = format_observation(observation)
            action_templates, objects = _get_compact_action_info(env)
            reformatted_actions = ", ".join(f"'{s}'" for s in action_templates if s != "help")
            reformatted_objects = ", ".join(f"'{s}'" for s in objects)

            if len(history) < HISTORY_LENGTH:
                user_content = SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=task_description,
                    current_observation=fmt_obs,
                    action_templates=reformatted_actions,
                    objects=reformatted_objects,
                )
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = SCIWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=step,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=step + 1,
                    current_observation=fmt_obs,
                    action_templates=reformatted_actions,
                    objects=reformatted_objects,
                )

            messages = system_msg + [{"role": "user", "content": user_content}]
            action, _ = model.generate(messages, temperature=temperature)
            history.append(_format_history(fmt_obs, step + 1, action))

            observation, reward, done, info = env.step(action)
            best_score = max(best_score, info.get("score", best_score + reward))
            if done:
                break

        env.close()
        # Normalize raw ScienceWorld score (0-100) to (0-1), matching the
        # training workflow convention (see TCOD/scienceworld/utils.py: best_score / 100.0)
        norm_score = best_score / 100.0
        return {
            "task_name": task_config.get("task_name", ""),
            "var_num": task_config.get("var_num", 0),
            "score": norm_score,
            "success": norm_score >= 1.0,
            "steps": step + 1,
        }
    except Exception as e:
        try: env.close()
        except: pass
        return {"task": task_config, "score": 0.0, "success": False,
                "steps": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data",  required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--java_home", default=os.environ.get("JAVA_HOME", ""),
                        help="Path to a Java 11+ home (ScienceWorld runs on the JVM). "
                             "Defaults to the JAVA_HOME env var.")
    parser.add_argument("--tp",         type=int,   default=1)
    parser.add_argument("--max_env_steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--n_tasks",    type=int,   default=None,
                        help="Limit number of test tasks (None=all)")
    args = parser.parse_args()
    cli_args = vars(args).copy()

    _setup_java(args.java_home)

    tasks = []
    with open(args.test_data) as f:
        for line in f:
            d = json.loads(line.strip())
            task_config = json.loads(d["task_desc"])
            tasks.append(task_config)
    if args.n_tasks:
        tasks = tasks[:args.n_tasks]
    print(f"Evaluating {len(tasks)} tasks from {args.test_data}")
    print("[sci_eval_full] Parsed args:")
    print(json.dumps(cli_args, indent=2, ensure_ascii=False))
    print(f"[sci_eval_full] argv: {' '.join(sys.argv)}")

    model_runtime_config = {
        "tensor_parallel_size": args.tp,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": 12288,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "disable_log_stats": True,
        "enable_thinking": False,
        "max_gen_tokens": 512,
    }
    model = SciProbeModel(args.model_path, tp=model_runtime_config["tensor_parallel_size"],
                          gpu_util=model_runtime_config["gpu_memory_utilization"],
                          max_model_len=model_runtime_config["max_model_len"])

    results = []
    t0 = time.time()
    for i, tc in enumerate(tasks):
        r = run_episode(model, tc, args.max_env_steps, args.temperature)
        results.append(r)
        elapsed = time.time() - t0
        avg_score = sum(x["score"] for x in results) / len(results)
        sr = sum(x["success"] for x in results) / len(results)
        print(f"  [{i+1}/{len(tasks)}] {tc.get('task_name','?')} v{tc.get('var_num',0)}: "
              f"score={r['score']:.3f} | avg_score={avg_score:.3f} SR={sr*100:.1f}% "
              f"t={elapsed:.0f}s")

    n = len(results)
    avg_score = sum(x["score"] for x in results) / n  # normalized 0-1
    sr = sum(x["success"] for x in results) / n
    print(f"\n=== ScienceWorld Eval Results ===")
    print(f"  n_tasks={n}, avg_score={avg_score:.3f} ({avg_score*100:.2f}%), SR={sr*100:.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
                   "evaluator_script": str(Path(__file__).resolve()),
                   "argv": sys.argv,
                   "parsed_args": cli_args,
                   "generation_config": {
                       "temperature": args.temperature,
                       "enable_thinking": False,
                       "max_gen_tokens": model_runtime_config["max_gen_tokens"],
                   },
                   "model_runtime_config": model_runtime_config,
                   "n_tasks": n,
                   "avg_score": avg_score,
                   "avg_score_pct": avg_score * 100.0,
                   "success_rate": sr,
                   "tasks": results}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
