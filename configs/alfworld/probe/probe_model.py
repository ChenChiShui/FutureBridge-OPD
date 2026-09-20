# -*- coding: utf-8 -*-
"""
Model wrapper for the probe experiment.
Uses vLLM LLM (synchronous) for generation and logprob scoring.
"""

import gc
import os
import json
from typing import List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams

from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
    HISTORY_LENGTH,
    parse_action,
    format_observation,
)


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_action_prompt(
    observation: str,
    admissible_commands: List[str],
    history: List[str],
    task_description: str,
    step_idx: int,
) -> str:
    """Build the user message for step step_idx."""
    reformatted = "\n ".join(
        f"'{s}'" for s in admissible_commands if s != "help"
    )
    format_obs = format_observation(observation)

    if len(history) < HISTORY_LENGTH:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=format_obs,
            admissible_actions=reformatted,
        )
    else:
        action_history_str = "\n".join(history[-HISTORY_LENGTH:])
        return ALFWORLD_TEMPLATE.format(
            task_description=task_description,
            step_count=step_idx,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history=action_history_str,
            current_step=step_idx + 1,
            current_observation=format_obs,
            admissible_actions=reformatted,
        )


def build_messages(
    observation: str,
    admissible_commands: List[str],
    history: List[str],
    task_description: str,
    step_idx: int,
) -> List[dict]:
    """Return OpenAI-style messages list for vLLM chat template."""
    user_content = format_action_prompt(
        observation, admissible_commands, history, task_description, step_idx
    )
    return [{"role": "user", "content": user_content}]


# ── vLLM model wrapper ────────────────────────────────────────────────────────

class ProbeModel:
    """Wraps a vLLM LLM instance for action generation and logprob scoring."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 12288,
        enforce_eager: bool = True,
        dtype: str = "bfloat16",
        visible_devices: Optional[str] = None,
        max_gen_tokens: int = 1024,   # allow long think phases (32B needs >512)
    ):
        print(
            f"[ProbeModel] Loading {model_path} (TP={tensor_parallel_size}"
            f", visible_devices={visible_devices or os.environ.get('CUDA_VISIBLE_DEVICES', '<default>')})"
        )
        self.model_path = model_path
        self.visible_devices = visible_devices
        self.max_model_len = max_model_len

        old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if visible_devices is not None and visible_devices != "":
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                dtype=dtype,
                trust_remote_code=True,
                disable_log_stats=True,
            )
        finally:
            if visible_devices is not None and visible_devices != "":
                if old_visible_devices is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices
        self.max_gen_tokens = max_gen_tokens
        self.tokenizer = self.llm.get_tokenizer()
        self.chat_template = getattr(self.tokenizer, "chat_template", None) or ""
        self.use_plain_chat_template = (
            isinstance(self.chat_template, str)
            and ("<tool_call>" in self.chat_template or "For each function call" in self.chat_template)
        )
        self._score_truncation_warned = False
        self._generate_truncation_warned = False
        if self.use_plain_chat_template:
            print("[ProbeModel] Detected tool-call chat_template; using plain chat rendering for action tasks.")
        print(f"[ProbeModel] Loaded: {model_path}")

    def _render_plain_chat_prompt(
        self,
        messages: List[dict],
        *,
        enable_thinking: bool = False,
    ) -> str:
        """Render a minimal Qwen-style chat prompt without any tool-call logic.

        Recent tokenizer chat templates bundled with some Qwen checkpoints include
        tool-calling branches and can trigger pathological generations such as
        repeated ``<tool_call>`` tags in offline ALFWorld probing. For our action
        tasks we only need a plain multi-turn chat transcript.
        """
        chunks: List[str] = []
        for msg in messages:
            role = (msg or {}).get("role", "user") or "user"
            content = (msg or {}).get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            if role == "tool":
                role = "user"
                content = f"<tool_response>\n{content}\n</tool_response>"
            elif role not in {"system", "user", "assistant"}:
                role = "user"

            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        chunks.append("<|im_start|>assistant\n")
        if not enable_thinking:
            chunks.append("<think>\n\n</think>\n\n")
        return "".join(chunks)

    def _apply_chat_template_token_ids(
        self,
        messages: List[dict],
        *,
        enable_thinking: bool = False,
    ) -> List[int]:
        if self.use_plain_chat_template:
            prompt = self._render_plain_chat_prompt(
                messages,
                enable_thinking=enable_thinking,
            )
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

    def _truncate_generation_prompt_ids(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
    ) -> List[int]:
        if self.max_model_len is None or self.max_model_len <= 0:
            return prompt_ids

        reserve = max(1, int(max_new_tokens))
        max_prompt_tokens = self.max_model_len - reserve
        if max_prompt_tokens <= 0:
            return prompt_ids[-1:]
        if len(prompt_ids) <= max_prompt_tokens:
            return prompt_ids

        if not self._generate_truncation_warned:
            print(
                f"[ProbeModel] generate_action truncating prompt from {len(prompt_ids)} "
                f"to {max_prompt_tokens} tokens to respect max_model_len={self.max_model_len}."
            )
            self._generate_truncation_warned = True
        return prompt_ids[-max_prompt_tokens:]

    def unload(self):
        """Free GPU memory + Ray workers before loading another model."""
        # 1. shutdown vllm engine first (kills EngineCore subprocess)
        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        # 2. Shutdown Ray workers spawned by vLLM for TP>1
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                print("[ProbeModel] Ray shut down.")
        except Exception as e:
            print(f"[ProbeModel] Ray shutdown warning: {e}")
        # 3. Force-kill any lingering vllm worker subprocesses.
        #    These are spawned by vllm.v1.executor.multiproc_executor and are NOT
        #    cleaned up by ray.shutdown(); they hold CUDA contexts, blocking the
        #    next ProbeModel from torch.cuda.init() in new worker processes.
        #    Without this, Phase 2 reload fails with:
        #      RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.
        try:
            import subprocess, time, os
            # Kill child processes that match vllm worker patterns (avoid killing self)
            my_pid = os.getpid()
            subprocess.run(
                f"ps -o pid,ppid,cmd -u $USER | "
                f"grep -E 'VllmWorker|EngineCore|multiproc_executor|VLLM_WORKER' | "
                f"grep -v grep | awk '$2 != {my_pid} && $1 != {my_pid} {{print $1}}' | "
                f"xargs -r kill -9",
                shell=True, timeout=10,
            )
            time.sleep(5)  # give CUDA driver time to release handles
            print("[ProbeModel] Killed lingering vllm worker subprocesses.")
            # Additional: try to reset cuda context by checking nvidia-smi state
            try:
                subprocess.run(["nvidia-smi"], timeout=10, capture_output=True)
            except Exception:
                pass
        except Exception as e:
            print(f"[ProbeModel] pkill warning: {e}")
        print(f"[ProbeModel] Unloaded: {self.model_path}")

    # ── Action generation ─────────────────────────────────────────────────────

    def generate_action(
        self,
        messages: List[dict],
        temperature: float = 0.0,
        max_tokens: int = -1,
        enable_thinking: bool = False,  # match training: config.enable_thinking=False
    ) -> Tuple[str, str]:
        """
        Generate an action given messages.
        Returns: (parsed_action, raw_response_text)

        Uses tokenizer.apply_chat_template() directly so we can pass
        enable_thinking=False, matching the TCOD training default.
        llm.chat() in vLLM 0.8.x doesn't expose chat_template_kwargs,
        so we pre-tokenize and call llm.generate() instead.
        """
        n_tokens = self.max_gen_tokens if max_tokens == -1 else max_tokens
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=n_tokens,
            stop=["</action>"],
        )

        prompt_ids = self._apply_chat_template_token_ids(
            messages,
            enable_thinking=enable_thinking,
        )
        prompt_ids = self._truncate_generation_prompt_ids(prompt_ids, n_tokens)

        outputs = self.llm.generate(prompt_token_ids=[prompt_ids], sampling_params=sampling_params)
        raw = outputs[0].outputs[0].text

        # Add closing tag that stop token consumed
        if "<action>" in raw and "</action>" not in raw:
            raw = raw + "</action>"

        action = parse_action(raw)
        return action, raw

    def generate_actions_batch(
        self,
        messages_list: List[List[dict]],
        temperature: float = 0.0,
        max_tokens: int = -1,
    ) -> List[Tuple[str, str]]:
        """Batch version of generate_action."""
        n_tokens = self.max_gen_tokens if max_tokens == -1 else max_tokens
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=n_tokens,
            stop=["</action>"],
        )
        prompt_token_ids = [
            self._truncate_generation_prompt_ids(
                self._apply_chat_template_token_ids(messages),
                n_tokens,
            )
            for messages in messages_list
        ]
        outputs = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        results = []
        for out in outputs:
            raw = out.outputs[0].text
            if "<action>" in raw and "</action>" not in raw:
                raw = raw + "</action>"
            action = parse_action(raw)
            results.append((action, raw))
        return results

    # ── Logprob scoring ───────────────────────────────────────────────────────

    def score_action_logprob(
        self,
        messages: List[dict],
        target_action: str,
        max_context_tokens: Optional[int] = None,
    ) -> float:
        """
        Compute log P(target_action | messages).

        Strategy:
          1. Apply chat template to get the prompt token IDs
          2. Tokenize the target response (minimal: <action>action</action>)
          3. Concatenate prompt + response token IDs
          4. Call llm.generate() with prompt_logprobs=5 on the FULL sequence
             (treating full sequence as prompt, generate 1 dummy token)
          5. Sum the logprobs of response token positions

        Using prompt_token_ids directly avoids tokenization round-trip issues.

        Note: vLLM prompt_logprobs scales memory roughly with prompt length.
        For long ALFWorld trajectories, scoring the full prompt can OOM even on
        80GB GPUs. `max_context_tokens` keeps only the tail of the prompt when
        scoring, which is a deliberate approximation for stability.
        """
        if not target_action:
            return float("-inf")

        target_response = f"<action>{target_action}</action>"

        # Tokenize via chat template application
        prompt_ids: List[int] = self._apply_chat_template_token_ids(
            messages,
            enable_thinking=False,
        )

        # Tokenize the response (no BOS/EOS)
        response_ids: List[int] = self.tokenizer.encode(
            target_response, add_special_tokens=False
        )

        if max_context_tokens is not None and max_context_tokens > 0:
            max_prompt_tokens = max_context_tokens - len(response_ids)
            if max_prompt_tokens <= 0:
                return float("-inf")
            if len(prompt_ids) > max_prompt_tokens:
                if not self._score_truncation_warned:
                    print(
                        f"[ProbeModel] score_action_logprob truncating prompt "
                        f"from {len(prompt_ids)} to {max_prompt_tokens} tokens "
                        f"for memory safety.")
                    self._score_truncation_warned = True
                prompt_ids = prompt_ids[-max_prompt_tokens:]

        # Build full token sequence: prompt + response
        full_ids = list(prompt_ids) + list(response_ids)

        # We only need the selected token logprob for each response token.
        # prompt_logprobs=1 keeps the returned structure minimal; the actual
        # response token is always included by vLLM even if it is not top-1.
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0.0,
        )

        # Pass token IDs directly to avoid decode/encode round-trip
        outputs = self.llm.generate(
            prompt_token_ids=[full_ids],
            sampling_params=sampling_params,
        )

        prompt_logprobs = outputs[0].prompt_logprobs  # List[Optional[Dict[int, Logprob]]]
        if prompt_logprobs is None:
            return float("-inf")

        # Sum logprobs for the response token positions
        total_logprob = 0.0
        response_start = len(prompt_ids)

        for i, tok_id in enumerate(response_ids):
            pos = response_start + i
            if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
                total_logprob += -20.0  # penalize out-of-range
                continue

            lp_dict = prompt_logprobs[pos]
            if tok_id in lp_dict:
                total_logprob += lp_dict[tok_id].logprob
            else:
                # Token not in the returned top-k; assign low logprob
                total_logprob += -20.0

        return total_logprob
