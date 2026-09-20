# -*- coding: utf-8 -*-
"""TCOD Forward-to-Backward workflow for ScienceWorld."""

import re
from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.common.workflows.envs.TCOD.scienceworld.utils import (
    HISTORY_LENGTH,
    SCIWORLD_TEMPLATE,
    SCIWORLD_TEMPLATE_NO_HIS,
    _create_scienceworld_env,
    _format_history,
    _get_compact_action_info,
    _reset_scienceworld_env,
    format_observation,
    parse_action,
)


@WORKFLOWS.register_module("TCOD_f2b_scienceworld_workflow")
class TCOD_f2b_scienceworld_workflow(Workflow):
    """TCOD Forward-to-Backward workflow for ScienceWorld."""

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
        self.is_eval = task.is_eval

        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
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

    def _compute_distill_window(self) -> int:
        return 1 + (self._current_training_step // self.checkpoint_steps)

    def compute_reward(self, response: Experience) -> float:
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
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

        env = _create_scienceworld_env(
            self.task_desc,
            max_env_steps=self.max_env_steps,
            generate_gold_path=False,
        )
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _run_episode(self, env) -> List[Experience]:
        observation, info, task_description = _reset_scienceworld_env(env)
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []
        best_score = info.get("score", 0)

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        distill_window = self._compute_distill_window()
        effective_steps = self.max_env_steps if self.is_eval else min(
            distill_window, self.max_env_steps
        )

        for r in range(effective_steps):
            format_obs = format_observation(observation)
            action_templates, objects = _get_compact_action_info(env)
            reformatted_actions = ", ".join(
                f"'{s}'" for s in action_templates if s != "help"
            )
            reformatted_objects = ", ".join(f"'{s}'" for s in objects)

            if len(history) < HISTORY_LENGTH:
                user_content = SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=task_description,
                    current_observation=format_obs,
                    action_templates=reformatted_actions,
                    objects=reformatted_objects,
                )
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = SCIWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=format_obs,
                    action_templates=reformatted_actions,
                    objects=reformatted_objects,
                )

            memory = memory + [{"role": "user", "content": user_content}]

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "TCOD_f2b_scienceworld_workflow requires student model to return "
                    "logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            best_score = max(best_score, info.get("score", best_score + reward))

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                break
        else:
            self._env_rounds = effective_steps

        self._final_reward = best_score / 100.0

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
            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i

        trajectory_kl_divergence = sum(per_turn_kl_sums)
        total_expert_actions = 0 if self.is_eval else len(
            self.raw_task.get("actions", [])
        )

        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            expected_teacher_env_rounds = max(
                total_expert_actions - self._env_rounds,
                0,
            )
            last_response.metrics["student_env_rounds"] = self._env_rounds
            last_response.metrics["teacher_env_rounds"] = 0
            last_response.metrics["expected_teacher_env_rounds"] = (
                expected_teacher_env_rounds
            )
            last_response.metrics["if_teacher"] = (
                0
                if self.is_eval
                else 1 if expected_teacher_env_rounds > 0 and not self._env_done else 0
            )
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence

        return turn_responses
