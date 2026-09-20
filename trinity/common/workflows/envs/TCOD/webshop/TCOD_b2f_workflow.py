# -*- coding: utf-8 -*-
"""TCOD Backward-to-Forward workflow for WebShop."""

import re
from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.common.workflows.envs.TCOD.webshop.utils import (
    HISTORY_LENGTH,
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    _create_webshop_env,
    _create_webshop_env_with_checkpoint,
    _extract_task_description,
    _format_available_actions,
    _format_history,
    format_observation,
    parse_action,
    validate_action,
)


@WORKFLOWS.register_module("TCOD_b2f_webshop_workflow")
class TCOD_b2f_webshop_workflow(Workflow):
    """TCOD Backward-to-Forward workflow for WebShop task."""

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
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 15)
        self.is_eval = task.is_eval

        self.checkpoint_strategy = task.workflow_args.get("checkpoint_strategy", "linear")
        self.checkpoint_steps = task.workflow_args.get("checkpoint_steps", 5)
        self.total_steps = task.workflow_args.get("total_steps", 200)

        self._current_training_step = 0
        self._total_training_steps = 0

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

    def set_training_progress(self, current_step: int, total_steps: int):
        self._current_training_step = current_step
        self._total_training_steps = total_steps

    def _linear_checkpoint_step(
        self, predefined_actions: Optional[List[str]] = None
    ) -> Optional[int]:
        if predefined_actions is None or len(predefined_actions) == 0:
            return 0
        current_step = self._current_training_step
        max_expert_actions = len(predefined_actions) - 1
        reduction = current_step // self.checkpoint_steps
        checkpoint_step = max(0, min(max_expert_actions - reduction, max_expert_actions))
        return checkpoint_step

    def compute_reward(self, response: Experience) -> float:
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
        if self.is_eval:
            env = _create_webshop_env()
            try:
                return await self._run_episode(env, int(self.task_desc))
            finally:
                env.close()

        current_step = 0
        if hasattr(self.task, "batch_id"):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                match = re.match(r"^(\d+)", batch_id)
                if match:
                    current_step = int(match.group(1))

        self.set_training_progress(current_step, self.total_steps)

        session_id = int(self.task_desc)
        predefined_actions = self.raw_task.get("actions", None)

        if self.checkpoint_strategy == "linear":
            effective_checkpoint_step = self._linear_checkpoint_step(predefined_actions)
        else:
            effective_checkpoint_step = 0

        if (
            effective_checkpoint_step is not None
            and effective_checkpoint_step > 0
            and predefined_actions is not None
        ):
            (
                env,
                observation,
                history,
                task_description,
                start_step,
                replay_done,
                replay_reward,
            ) = _create_webshop_env_with_checkpoint(
                session_id, predefined_actions, effective_checkpoint_step
            )
            try:
                if replay_done:
                    self._env_done = True
                    self._env_rounds = start_step
                    self._final_reward = replay_reward
                    return []
                return await self._run_episode_from_checkpoint(
                    env,
                    session_id,
                    observation,
                    history,
                    task_description,
                    start_step,
                )
            finally:
                env.close()

        env = _create_webshop_env()
        try:
            return await self._run_episode(env, session_id)
        finally:
            env.close()

    async def _run_episode(self, env, session_id: int) -> List[Experience]:
        env.reset(session=session_id)
        observation = env.observation
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

        task_description = _extract_task_description(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            available_actions = env.get_available_actions()
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

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "TCOD_b2f_webshop_workflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            action_valid, error_msg = validate_action(action, available_actions)
            history.append(_format_history(formatted_observation, r + 1, action))

            if action_valid:
                observation, reward, done, _ = env.step(action)
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

        per_turn_kl_sums: List[float] = []
        for i, response in enumerate(turn_responses):
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )

            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs

            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )

            response.teacher_logprobs = teacher_resp_logprobs

            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        trajectory_kl_divergence = sum(per_turn_kl_sums)

        if self.is_eval:
            total_expert_actions = 0
        else:
            total_expert_actions = len(self.raw_task.get("actions", []))

        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["student_env_rounds"] = self._env_rounds
            last_response.metrics["teacher_env_rounds"] = 0
            last_response.metrics["expected_teacher_env_rounds"] = total_expert_actions
            last_response.metrics["if_teacher"] = 0
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
            last_response.metrics["session_id"] = float(session_id)

        return turn_responses

    async def _run_episode_from_checkpoint(
        self,
        env,
        session_id: int,
        observation,
        history: List[str],
        task_description: str,
        start_step: int,
    ) -> List[Experience]:
        self._env_done = False
        self._env_rounds = start_step
        self._final_reward = 0.0

        memory = self.format_messages()
        turn_responses: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(start_step, self.max_env_steps):
            available_actions = env.get_available_actions()
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

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "TCOD_b2f_webshop_workflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            action_valid, error_msg = validate_action(action, available_actions)
            history.append(_format_history(formatted_observation, r + 1, action))

            if action_valid:
                observation, reward, done, _ = env.step(action)
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

        per_turn_kl_sums: List[float] = []
        for i, response in enumerate(turn_responses):
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )

            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs

            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )

            response.teacher_logprobs = teacher_resp_logprobs

            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = start_step + i

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        trajectory_kl_divergence = sum(per_turn_kl_sums)
        total_expert_actions = len(self.raw_task.get("actions", []))
        expert_actions_remaining = max(total_expert_actions - start_step, 0)

        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["student_env_rounds"] = self._env_rounds - start_step
            last_response.metrics["teacher_env_rounds"] = start_step
            last_response.metrics["if_teacher"] = 1 if start_step > 0 else 0
            last_response.metrics["expected_teacher_env_rounds"] = expert_actions_remaining
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
            last_response.metrics["session_id"] = float(session_id)

        return turn_responses
