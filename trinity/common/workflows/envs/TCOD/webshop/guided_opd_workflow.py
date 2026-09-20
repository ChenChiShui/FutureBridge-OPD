# -*- coding: utf-8 -*-
"""Guided-OPD workflow for WebShop.

Per-turn independent Bernoulli sampling: teacher with prob β_s, student with prob 1-β_s.
Teacher turns → expert_mask=True → SFT loss.
Student turns → expert_mask=False → OPD loss.
"""

import math
import random
from dataclasses import asdict
from typing import List, Optional

import torch

from trinity.common.experience import CustomField, Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

from trinity.common.workflows.envs.TCOD.webshop.utils import (
    HISTORY_LENGTH,
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    _create_webshop_env,
    _extract_task_description,
    _format_available_actions,
    _format_history,
    format_observation,
    parse_action,
    validate_action,
)


def guided_teacher_prob(
    step: int,
    total_steps: int = 250,
    curriculum_ratio: float = 0.8,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
) -> float:
    """Cosine decay schedule for teacher probability."""
    decay_steps = max(1, int(total_steps * curriculum_ratio))
    progress = min(step / decay_steps, 1.0)
    cosine_progress = (1.0 - math.cos(math.pi * progress)) / 2.0
    beta = beta_start + (beta_end - beta_start) * cosine_progress
    return float(beta)


@WORKFLOWS.register_module("guided_opd_webshop_workflow")
class GuidedOPDWebshopWorkflow(Workflow):
    """Guided-OPD workflow for WebShop.

    Required yaml config:
      algorithm.policy_loss_fn: mix
      algorithm.advantage_fn: multi_turn_opd
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = True

    _GUIDED_CUSTOM_FIELDS = [
        CustomField("is_expert", "expert_mask", torch.bool),
    ]

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
        ), "Guided-OPD requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 15)
        self.env = _create_webshop_env()

        # Guided-OPD curriculum parameters
        self.beta_start = task.workflow_args.get("beta_start", 1.0)
        self.beta_end = task.workflow_args.get("beta_end", 0.0)
        self.curriculum_ratio = task.workflow_args.get("curriculum_ratio", 0.8)
        self.total_steps = task.workflow_args.get("total_steps", 250)

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval
        self.repeat_times = task.repeat_times or 1

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
        raw_id = int(self.task_desc)
        n_goals = len(self.env.server.goals)
        session_id = raw_id % n_goals if n_goals > 0 else raw_id
        all_turn_responses: List[Experience] = []

        for rollout_idx in range(self.repeat_times):
            rollout_responses = await self._run_episode(
                session_id=session_id,
                run_id=self.run_id_base + rollout_idx,
            )
            all_turn_responses.extend(rollout_responses)

        return all_turn_responses

    async def _run_episode(self, session_id: int, run_id: int) -> List[Experience]:
        self.env.reset(session=session_id)
        observation = self.env.observation
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

        task_description = _extract_task_description(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_experiences: List[Experience] = []
        teacher_turn_flags: List[bool] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        # Compute beta from training step (batch_id), not run_id_base (repeat index)
        if self.is_eval:
            beta = 0.0
        else:
            global_step = 0
            if hasattr(self.task, 'batch_id'):
                batch_id = self.task.batch_id
                if isinstance(batch_id, int):
                    global_step = batch_id
                elif isinstance(batch_id, str):
                    import re as _re
                    m = _re.match(r'^(\d+)', batch_id)
                    if m:
                        global_step = int(m.group(1))
            beta = guided_teacher_prob(
                step=global_step,
                total_steps=self.total_steps,
                curriculum_ratio=self.curriculum_ratio,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )

        n_teacher_turns = 0
        n_student_turns = 0

        for r in range(self.max_env_steps):
            available_actions = self.env.get_available_actions()
            formatted_observation = format_observation(observation)
            formatted_actions = _format_available_actions(available_actions)

            if len(history) < HISTORY_LENGTH:
                user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_description,
                    current_observation=formatted_observation,
                    available_actions=formatted_actions,
                )
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = WEBSHOP_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=formatted_observation,
                    available_actions=formatted_actions,
                )

            memory = memory + [{"role": "user", "content": user_content}]

            # Per-turn independent Bernoulli sampling
            use_teacher = (not self.is_eval) and (random.random() < beta)

            if use_teacher:
                # Teacher generates response
                teacher_kwargs = {"temperature": 1.0, "max_tokens": 512, "n": 1}
                teacher_responses = await self.teacher_model.chat_async(
                    memory, **teacher_kwargs
                )
                teacher_response = teacher_responses[0]
                response_text = teacher_response.response_text or ""

                # Build experience with student tokenizer
                target_messages = memory + [
                    {"role": "assistant", "content": response_text}
                ]
                try:
                    target_exp = await self.model.convert_messages_to_experience_async(
                        messages=target_messages,
                        temperature=self.temperature,
                    )
                except Exception:
                    action = parse_action(response_text)
                    action_valid, error_msg = validate_action(action, available_actions)
                    history.append(_format_history(formatted_observation, r + 1, action))
                    memory.append({"role": "assistant", "content": response_text})
                    if action_valid:
                        observation, reward, done, _ = self.env.step(action)
                    else:
                        observation = error_msg
                        done = False
                    if done:
                        self._env_done = True
                        self._env_rounds = r + 1
                        self._final_reward = float(reward) if action_valid else 0.0
                        break
                    continue

                # Fix action_mask: only last assistant turn should be 1
                try:
                    prompt_only_exp = await self.model.convert_messages_to_experience_async(
                        messages=memory,
                        temperature=self.temperature,
                    )
                    target_start = len(prompt_only_exp.tokens) - target_exp.prompt_length
                    if target_start > 0 and target_exp.action_mask is not None:
                        target_exp.action_mask[:target_start] = 0
                except Exception:
                    pass

                # Skip if prompt truncated
                if target_exp.truncate_status == "prompt_truncated":
                    action = parse_action(response_text)
                    action_valid, error_msg = validate_action(action, available_actions)
                    history.append(_format_history(formatted_observation, r + 1, action))
                    memory.append({"role": "assistant", "content": response_text})
                    if action_valid:
                        observation, reward, done, _ = self.env.step(action)
                    else:
                        observation = error_msg
                        done = False
                    if done:
                        self._env_done = True
                        self._env_rounds = r + 1
                        self._final_reward = float(reward) if action_valid else 0.0
                        break
                    continue

                # Set teacher_logprobs = zeros (SFT loss ignores advantages)
                if target_exp.logprobs is not None:
                    target_exp.teacher_logprobs = torch.zeros_like(target_exp.logprobs)
                else:
                    target_exp.teacher_logprobs = torch.zeros(1, dtype=torch.float32)

                # Mark as expert
                target_exp.info["is_expert"] = True
                target_exp.custom_fields = list(self._GUIDED_CUSTOM_FIELDS)

                target_exp.eid.run = run_id
                target_exp.eid.step = len(turn_experiences)
                if target_exp.metrics is None:
                    target_exp.metrics = {}
                target_exp.metrics["is_teacher_turn"] = 1.0

                turn_experiences.append(target_exp)
                teacher_turn_flags.append(True)
                n_teacher_turns += 1

                # Execute teacher's action
                action = parse_action(response_text)
            else:
                # Student generates response
                responses = await self.model.chat_async(memory, **kwargs)
                response = responses[0]
                response_text = response.response_text or ""

                if response.logprobs is None:
                    raise RuntimeError(
                        "GuidedOPD requires student model to return logprobs."
                    )

                # Mark as non-expert
                response.info["is_expert"] = False
                response.custom_fields = list(self._GUIDED_CUSTOM_FIELDS)

                response.eid.run = run_id
                response.eid.step = len(turn_experiences)
                if response.metrics is None:
                    response.metrics = {}
                response.metrics["is_teacher_turn"] = 0.0

                turn_experiences.append(response)
                teacher_turn_flags.append(False)
                n_student_turns += 1

                action = parse_action(response_text)

            # Shared environment history
            memory.append({"role": "assistant", "content": response_text})
            action_valid, error_msg = validate_action(action, available_actions)
            history.append(_format_history(formatted_observation, r + 1, action))

            if action_valid:
                observation, reward, done, _ = self.env.step(action)
            else:
                observation = error_msg
                reward = 0.0
                done = False

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = float(reward)
                break
        else:
            self._env_rounds = self.max_env_steps

        # ── Phase 2: Compute teacher logprobs for student turns (OPD) ──
        per_turn_kl_sums: List[float] = []
        for i, exp in enumerate(turn_experiences):
            if teacher_turn_flags[i]:
                # Teacher turn: teacher_logprobs already set to zeros
                per_turn_kl_sums.append(0.0)
                continue

            # Student turn: compute teacher logprobs (same tokenizer, direct)
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=exp.tokens.tolist(),
                temperature=self.temperature,
            )

            resp_start = exp.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = exp.logprobs

            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher={len(teacher_resp_logprobs)}, "
                f"student={len(student_resp_logprobs)}"
            )

            exp.teacher_logprobs = teacher_resp_logprobs

            if exp.metrics is None:
                exp.metrics = {}
            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        # Set reward and trajectory-level metrics
        for exp in turn_experiences:
            exp.reward = self._final_reward

        if turn_experiences:
            last_exp = turn_experiences[-1]
            if last_exp.metrics is None:
                last_exp.metrics = {}
            last_exp.metrics["env_rounds"] = self._env_rounds
            last_exp.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_exp.metrics["kl_divergence"] = sum(per_turn_kl_sums)
            last_exp.metrics["session_id"] = float(session_id)
            last_exp.metrics["guided_beta"] = beta
            last_exp.metrics["teacher_turns"] = float(n_teacher_turns)
            last_exp.metrics["student_turns"] = float(n_student_turns)
            last_exp.metrics["teacher_turn_ratio"] = (
                n_teacher_turns / max(n_teacher_turns + n_student_turns, 1)
            )

        return turn_experiences
