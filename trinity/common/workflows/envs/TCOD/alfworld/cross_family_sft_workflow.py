# -*- coding: utf-8 -*-
"""Cross-family action-level SFT workflow for ALFWorld.

Student (Llama) rolls out in ALFWorld. Teacher (Qwen/GiGPO) generates
response on same states. Extract canonical action from teacher response.
Train with explicit SFT CE loss on the target response tokens.

Mechanism:
  - Student (Llama) rollout, recording messages per turn
  - Teacher (Qwen/GiGPO) generates response on same state
  - Extract canonical action from teacher response via parse_action()
  - Build target_response = "<action>{teacher_action}</action>"
  - convert_messages_to_experience_async(prompt + target_response)
      → Llama tokenizer tokenizes the full sequence
      → Llama model computes logprobs on target tokens
      → action_mask masks response tokens (target span)
  - Trainer uses policy_loss_fn=sft:
      loss = -mean(logprob * action_mask)  (= CE / NLL on target tokens)

This avoids cross-tokenizer KL entirely: teacher's tokenizer is never
involved in loss computation. Only the teacher's TEXT OUTPUT (action)
is used as a hard label, tokenized by the student's own tokenizer.

Note: teacher_logprobs is set to zeros only to satisfy the advantage_fn
(multi_turn_opd) which requires the field to exist. The SFT loss
completely ignores advantages — it only uses logprob and action_mask.
"""

from dataclasses import asdict
from typing import List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
    HISTORY_LENGTH,
    parse_action,
    format_observation,
    _extract_task,
    _format_history,
    _create_alfworld_env,
)


@WORKFLOWS.register_module("cross_family_sft_alfworld_workflow")
@WORKFLOWS.register_module("cf_opd_hard_alfworld_workflow")
class CrossFamilyActionSFTWorkflow(Workflow):
    """Cross-family action-level SFT workflow for ALFWorld.

    Student (Llama) explores ALFWorld. Teacher (Qwen/GiGPO) provides
    canonical action labels. Student is trained with SFT CE loss on
    the teacher's action span using its own tokenizer.

    Required yaml config:
      algorithm.policy_loss_fn: sft
      algorithm.advantage_fn: multi_turn_opd  (needed for schema, but ignored by SFT loss)
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.reset(task)

        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "Cross-family SFT requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.is_eval = task.is_eval

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
        game_file_path = self.task_desc
        env = _create_alfworld_env(game_file_path)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_experiences: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            admissible_commands = info.get("admissible_commands", [])
            if admissible_commands and isinstance(admissible_commands[0], list):
                admissible_commands = admissible_commands[0]
            reformatted_admissible = "\n ".join(
                f"'{s}'" for s in admissible_commands if s != "help"
            )

            if len(history) < HISTORY_LENGTH:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )
            else:
                action_history_str = "\n".join(
                    history[-HISTORY_LENGTH:]
                    if len(history) >= HISTORY_LENGTH
                    else history
                )
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )

            memory = memory + [{"role": "user", "content": user_content}]

            MAX_MEMORY_TURNS = 10
            recent_memory = memory[-(MAX_MEMORY_TURNS * 2):]

            # Step 1: Student (Llama) generates response and executes action
            responses = await self.model.chat_async(recent_memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            student_action = parse_action(response_text)

            with open("exp/cross_family_debug.log", "a") as f:
                f.write(f"turn {r}: student_action={student_action!r} resp_len={len(response_text)}\n")

            memory.append({"role": "assistant", "content": response_text})
            history.append(_format_history(format_obs, r + 1, student_action))

            # Step 2: Teacher (Qwen/GiGPO) generates response on same state.
            # Skip teacher during eval — eval is pure student rollout.
            teacher_action = ""
            if not self.is_eval:
                teacher_kwargs = {"temperature": 0.4, "max_tokens": 512, "n": 1}
                try:
                    teacher_responses = await self.teacher_model.chat_async(
                        recent_memory, **teacher_kwargs
                    )
                    teacher_response = teacher_responses[0]
                    teacher_raw = teacher_response.response_text or ""
                    teacher_action = parse_action(teacher_raw)
                    with open("exp/cross_family_debug.log", "a") as f:
                        f.write(f"turn {r}: teacher_raw={teacher_raw[:300]!r} action={teacher_action!r}\n")
                except Exception as e:
                    with open("exp/cross_family_debug.log", "a") as f:
                        f.write(f"turn {r}: teacher FAILED: {e}\n")
                    teacher_action = ""

            # In eval mode or teacher failed: use student's own response as
            # the Experience for metrics collection.
            # - Eval mode: experiences are not sent to train buffer (workflow_runner
            #   returns [] for eval tasks), so no contamination.
            # - Teacher failed (training mode): do NOT append to turn_experiences,
            #   so failed-teacher turns are filtered from the train batch.
            if self.is_eval:
                if response.logprobs is not None:
                    response.teacher_logprobs = torch.zeros_like(response.logprobs)
                response.reward = 0.0
                response.eid.run = getattr(self, "run_id_base", 0)
                response.eid.step = len(turn_experiences)
                if response.metrics is None:
                    response.metrics = {}
                response.info["student_action"] = student_action
                turn_experiences.append(response)

                observation, reward, done, info = env.step(student_action)
                if done:
                    self._env_done = True
                    self._env_rounds = r + 1
                    self._final_reward = 1.0
                    break
                continue

            # Training mode + teacher failed: skip this turn (no training signal)
            if not teacher_action:
                observation, reward, done, info = env.step(student_action)
                if done:
                    self._env_done = True
                    self._env_rounds = r + 1
                    self._final_reward = 1.0
                    break
                continue

            # Step 3: Build target response with teacher's canonical action.
            # Only the action span is the training target (no <think>).
            target_response = f"<action>{teacher_action}</action>"
            # Use full recent_memory as context + target as final assistant msg.
            target_messages = recent_memory + [
                {"role": "assistant", "content": target_response}
            ]

            # Tokenize with student (Llama) tokenizer + compute student logprobs
            try:
                target_exp = await self.model.convert_messages_to_experience_async(
                    messages=target_messages,
                    temperature=self.temperature,
                )
            except Exception as e:
                with open("exp/cross_family_debug.log", "a") as f:
                    f.write(f"turn {r}: convert_messages FAILED: {e}\n")
                observation, reward, done, info = env.step(student_action)
                if done:
                    self._env_done = True
                    self._env_rounds = r + 1
                    self._final_reward = 1.0
                    break
                continue

            # Fix action_mask: default tokenizer marks ALL assistant turns
            # (including historical student responses) as 1. We only want
            # the LAST assistant turn (teacher target action) to be 1.
            # Zero out historical assistant tokens by comparing with
            # prompt-only tokenization (without the final target assistant).
            prompt_only_messages = recent_memory  # without target assistant
            try:
                prompt_only_exp = await self.model.convert_messages_to_experience_async(
                    messages=prompt_only_messages,
                    temperature=self.temperature,
                )
                # prompt_only_exp.tokens length = full sequence without target
                # target_exp.tokens length = full sequence with target
                # Target tokens start at len(prompt_only_exp.tokens) in the
                # full sequence. In the response portion (after prompt_length),
                # this corresponds to offset:
                #   target_start = len(prompt_only_exp.tokens) - target_exp.prompt_length
                target_start = len(prompt_only_exp.tokens) - target_exp.prompt_length
                if target_start > 0 and target_exp.action_mask is not None:
                    # Zero out historical assistant tokens
                    target_exp.action_mask[:target_start] = 0
            except Exception:
                # If prompt-only tokenization fails, keep default mask
                # (loss on historical tokens is small, not fatal)
                pass

            # Skip if prompt was truncated
            if target_exp.truncate_status == "prompt_truncated":
                with open("exp/cross_family_debug.log", "a") as f:
                    f.write(f"turn {r}: prompt_truncated, target_tokens={len(target_exp.tokens)} prompt_len={target_exp.prompt_length}\n")
                observation, reward, done, info = env.step(student_action)
                if done:
                    self._env_done = True
                    self._env_rounds = r + 1
                    self._final_reward = 1.0
                    break
                continue

            # Skip if logprobs not computed
            if target_exp.logprobs is None:
                observation, reward, done, info = env.step(student_action)
                if done:
                    self._env_done = True
                    self._env_rounds = r + 1
                    self._final_reward = 1.0
                    break
                continue

            # Step 4: Set teacher_logprobs = zeros to satisfy advantage_fn schema.
            # SFT loss (policy_loss_fn=sft) ignores advantages entirely.
            # It computes: loss = -mean(logprob * action_mask) = CE / NLL.
            target_exp.teacher_logprobs = torch.zeros_like(target_exp.logprobs)

            # Set metadata
            target_exp.reward = 0.0  # Set after episode finishes
            target_exp.eid.run = getattr(self, "run_id_base", 0)
            target_exp.eid.step = len(turn_experiences)

            if target_exp.metrics is None:
                target_exp.metrics = {}
            target_exp.info["teacher_action"] = teacher_action
            target_exp.info["student_action"] = student_action
            target_exp.metrics["action_match"] = (
                1.0 if teacher_action.strip() == student_action.strip() else 0.0
            )

            turn_experiences.append(target_exp)

            # Execute STUDENT's action in env (not teacher's)
            observation, reward, done, info = env.step(student_action)

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0

        # If no valid teacher actions were obtained (all teacher calls failed),
        # return a dummy experience with the student's first response to avoid
        # the "empty experience" assertion in workflow_runner.
        print(f"[CrossFamilySFT] Episode done: turns={len(turn_experiences)} env_done={self._env_done} rounds={self._env_rounds}")
        with open("exp/cross_family_debug.log", "a") as f:
            f.write(f"=== Episode done: turns={len(turn_experiences)} env_done={self._env_done} rounds={self._env_rounds} ===\n")
        if not turn_experiences and not self.is_eval:
            dummy = responses[0] if 'responses' in dir() else None
            if dummy is not None:
                if dummy.logprobs is not None:
                    dummy.teacher_logprobs = torch.zeros_like(dummy.logprobs)
                else:
                    dummy.teacher_logprobs = torch.zeros(1, dtype=torch.float32)
                dummy.reward = self._final_reward
                dummy.eid.run = getattr(self, "run_id_base", 0)
                dummy.eid.step = 0
                if dummy.metrics is None:
                    dummy.metrics = {}
                dummy.metrics["env_rounds"] = self._env_rounds
                dummy.metrics["env_done"] = 1.0 if self._env_done else 0.0
                dummy.metrics["action_match_rate"] = 0.0
                dummy.metrics["all_teacher_failed"] = 1.0
                turn_experiences.append(dummy)

        # Set final reward for all turn experiences
        for exp in turn_experiences:
            exp.reward = self._final_reward

        # Trajectory-level metrics
        if turn_experiences:
            last_exp = turn_experiences[-1]
            if last_exp.metrics is None:
                last_exp.metrics = {}
            last_exp.metrics["env_rounds"] = self._env_rounds
            last_exp.metrics["env_done"] = 1.0 if self._env_done else 0.0
            match_rates = [
                e.metrics.get("action_match", 0) for e in turn_experiences
            ]
            last_exp.metrics["action_match_rate"] = sum(match_rates) / max(
                len(match_rates), 1
            )

        return turn_experiences
