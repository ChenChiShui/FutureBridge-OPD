#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebShop full evaluation on test sessions.

Usage:
  python eval_webshop.py \
      --model_path /path/to/checkpoint/actor/huggingface \
      --webshop_path /path/to/webshop \   # your cloned WebShop repo (or set WEBSHOP_PATH)
      --n_sessions 100 \
      --session_start 4096 \
      --output /path/to/results.json \
      --tp 8 --temperature 0.0 --max_env_steps 15

Uses the full 1.18M WebShop dataset (items_shuffle.json).
The env is created once and reused across all sessions to avoid repeated loading.
"""

import argparse
import json
import os
import sys
from pathlib import Path

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


from trinity.common.workflows.envs.TCOD.webshop.utils import (
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    parse_action,
    validate_action,
    format_observation,
    _format_available_actions,
    _format_history,
)


def _create_webshop_env_full(webshop_path: str):
    """Create WebShop env with full 1.18M product dataset."""
    full_file_path = os.path.join(webshop_path, "data", "items_shuffle.json")
    full_attr_path = os.path.join(webshop_path, "data", "items_ins_v2.json")
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
    return gym.make(
        "WebAgentTextEnv-v0",
        observation_mode="text_rich",
        num_products=None,
        human_goals=True,
        file_path=full_file_path,
        attr_path=full_attr_path,
    )


def run_one_session(env, model, session_id: int, max_env_steps: int, temperature: float, max_tokens: int = 512) -> dict:
    """Run one WebShop session and return result. Reuses an existing env."""
    try:
        env.reset(session=session_id)
    except Exception as e:
        return {"session_id": session_id, "reward": 0.0, "success": False,
                "steps": 0, "error": str(e)}

    observation = env.observation
    history = []
    memory = []
    final_reward = 0.0
    done = False

    try:
        from trinity.common.workflows.envs.TCOD.webshop.utils import _extract_task_description
        task_description = _extract_task_description(observation)
    except Exception:
        task_description = "Find a product."

    for turn in range(max_env_steps):
        available_actions = env.get_available_actions()
        formatted_obs = format_observation(observation)
        formatted_actions = _format_available_actions(available_actions)

        if len(history) < HISTORY_LENGTH:
            user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=task_description,
                current_observation=formatted_obs,
                available_actions=formatted_actions,
            )
        else:
            action_history_str = "\n".join(history[-HISTORY_LENGTH:])
            user_content = WEBSHOP_TEMPLATE.format(
                task_description=task_description,
                step_count=turn,
                history_length=min(HISTORY_LENGTH, len(history)),
                action_history=action_history_str,
                current_step=turn + 1,
                current_observation=formatted_obs,
                available_actions=formatted_actions,
            )

        memory = memory + [{"role": "user", "content": user_content}]
        try:
            response_text = model.chat(memory, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            return {"session_id": session_id, "reward": final_reward, "success": False,
                    "steps": turn + 1, "error": f"model_error: {e}"}
        memory.append({"role": "assistant", "content": response_text})

        action = parse_action(response_text)
        action_valid, error_msg = validate_action(action, available_actions)
        history.append(_format_history(formatted_obs, turn + 1, action))

        if action_valid:
            observation, reward, done, _ = env.step(action)
            final_reward = float(reward)
        else:
            observation = error_msg
            done = False

        if done:
            break

    return {
        "session_id": session_id,
        "reward": final_reward,
        "success": done and final_reward > 0.5,
        "steps": turn + 1,
    }


class VllmProbeModel:
    """Simple vllm-based model for evaluation."""
    def __init__(self, model_path: str, tp: int, enable_thinking: bool = False):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            enforce_eager=True,
            dtype="bfloat16",
            max_model_len=20480,
            enable_prefix_caching=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.enable_thinking = enable_thinking

    def chat(self, messages, temperature=0.0, max_tokens=512):
        from vllm import SamplingParams
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate([prompt], params)
        text = outputs[0].outputs[0].text
        # 若 thinking 模式，去掉 <think>...</think> 块，只保留实际 action
        if self.enable_thinking and "</think>" in text:
            text = text.split("</think>")[-1].lstrip()
        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--webshop_path", default=os.environ.get("WEBSHOP_PATH", ""),
                        help="Path to your cloned WebShop repository (contains web_agent_site/ "
                             "and data/items_shuffle.json). Can also be set via the WEBSHOP_PATH env var.")
    parser.add_argument("--java_home", default=os.environ.get("JAVA_HOME", ""),
                        help="Path to a Java 11+ home. Defaults to the JAVA_HOME env var.")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_env_steps", type=int, default=15)
    parser.add_argument("--session_start", type=int, default=4096,
                        help="Start session ID (test set starts at 4096)")
    parser.add_argument("--n_sessions", type=int, default=100)
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable thinking tokens (for Qwen3 thinking models)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per action step (default 512)")
    args = parser.parse_args()
    cli_args = vars(args).copy()

    if not args.webshop_path:
        sys.exit("ERROR: --webshop_path (or the WEBSHOP_PATH env var) is required. "
                 "Set it to your cloned WebShop repository, which contains web_agent_site/ "
                 "and data/items_shuffle.json.")
    _setup_runtime(args.webshop_path, args.java_home)

    print(f"Loading model from {args.model_path}...")
    print(f"  enable_thinking={args.enable_thinking}, max_tokens={args.max_tokens}")
    print("[eval_webshop] Parsed args:")
    print(json.dumps(cli_args, indent=2, ensure_ascii=False))
    print(f"[eval_webshop] argv: {' '.join(sys.argv)}")
    model = VllmProbeModel(args.model_path, args.tp, enable_thinking=args.enable_thinking)

    print("Creating WebShop env (loading full 1.18M product dataset, ~2-5 min)...")
    env = _create_webshop_env_full(args.webshop_path)
    print("WebShop env ready.")

    results = []
    session_ids = list(range(args.session_start, args.session_start + args.n_sessions))
    print(f"Evaluating {len(session_ids)} sessions (ID {session_ids[0]}-{session_ids[-1]})...")

    for i, sid in enumerate(session_ids):
        result = run_one_session(env, model, sid, args.max_env_steps, args.temperature, args.max_tokens)
        results.append(result)
        if (i + 1) % 10 == 0:
            done_so_far = sum(r["success"] for r in results)
            avg_r = sum(r["reward"] for r in results) / len(results)
            print(f"  [{i+1}/{len(session_ids)}] SR={done_so_far/(i+1):.3f} avg_reward={avg_r:.3f}")

    success_rate = sum(r["success"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)

    output = {
        "evaluator_script": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "parsed_args": cli_args,
        "generation_config": {
            "temperature": args.temperature,
            "enable_thinking": args.enable_thinking,
            "max_tokens": args.max_tokens,
        },
        "model_runtime_config": {
            "tensor_parallel_size": args.tp,
            "dtype": "bfloat16",
            "max_model_len": 20480,
            "enable_prefix_caching": False,
            "enforce_eager": True,
        },
        "model_path": args.model_path,
        "n_sessions": len(session_ids),
        "session_range": f"{session_ids[0]}-{session_ids[-1]}",
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    env.close()
    print(f"\nResults: SR={success_rate:.3f} avg_reward={avg_reward:.3f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
