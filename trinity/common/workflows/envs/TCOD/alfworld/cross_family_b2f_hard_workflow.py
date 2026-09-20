# -*- coding: utf-8 -*-
"""CF-B2F-Hard: Cross-family B2F with action-level hard distillation.

Teacher (Qwen) executes prefix actions for the first k_s turns (linear decay),
Student (Llama) takes over afterwards. ALL turns use Teacher action as SFT target.
Student tokenizer tokenizes everything. No cross-tokenizer KL.
"""

import re
from dataclasses import asdict
from typing import List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    _create_alfworld_env,
    _create_alfworld_env_with_checkpoint,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)
from trinity.common.workflows.envs.TCOD.alfworld.cross_family_utils import (
    build_user_content,
    canonicalize_action,
    make_cross_family_sft_experience,
)


@WORKFLOWS.register_module("cf_b2f_hard_alfworld_workflow")
class CFB2FHardWorkflow(Workflow):
    """Cross-family B2F-Hard: Teacher prefix + action SFT for Llama student."""

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
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset(task)

        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "CF-B2F-Hard requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.is_eval = task.is_eval

        self.checkpoint_strategy = task.workflow_args.get("checkpoint_strategy", "linear")
        self.checkpoint_steps = task.workflow_args.get("checkpoint_steps", 5)
        self.total_steps = task.workflow_args.get("total_steps", 250)

        self._current_training_step = 0

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

    def _linear_checkpoint_step(self, predefined_actions):
        if not predefined_actions:
            return 0
        max_expert_actions = len(predefined_actions) - 1
        reduction = self._current_training_step // self.checkpoint_steps
        return max(0, min(max_expert_actions - reduction, max_expert_actions))

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
        if self.is_eval:
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env, prefix_length=0)
            finally:
                env.close()

        # Extract training step
        current_step = 0
        if hasattr(self.task, 'batch_id'):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                match = re.match(r'^(\d+)', batch_id)
                if match:
                    current_step = int(match.group(1))
        self.set_training_progress(current_step, self.total_steps)

        game_file_path = self.task_desc
        predefined_actions = self.raw_task.get("actions", None)

        prefix_length = 0
        if self.checkpoint_strategy == "linear":
            prefix_length = self._linear_checkpoint_step(predefined_actions)

        if prefix_length > 0 and predefined_actions:
            env, observation, info, history, task_description, start_step, replay_done = (
                _create_alfworld_env_with_checkpoint(
                    game_file_path, predefined_actions, prefix_length
                )
            )
            try:
                if replay_done:
                    self._env_done = True
                    self._env_rounds = start_step
                    self._final_reward = 1.0
                    return []
                return await self._run_episode_from_checkpoint(
                    env, observation, info, history, task_description, start_step,
                    prefix_length, predefined_actions
                )
            finally:
                env.close()
        else:
            env = _create_alfworld_env(game_file_path)
            try:
                return await self._run_episode(env, prefix_length=0)
            finally:
                env.close()

    async def _run_episode(self, env, prefix_length=0) -> List[Experience]:
        """Run episode from start (no checkpoint)."""
        observation, info = env.reset()
        return await self._rollout(
            env, observation, info, [], _extract_task(observation), 0, prefix_length, None
        )

    async def _run_episode_from_checkpoint(
        self, env, observation, info, history, task_description, start_step,
        prefix_length, predefined_actions
    ) -> List[Experience]:
        """Run episode from a checkpoint (teacher has executed first prefix_length actions)."""
        return await self._rollout(
            env, observation, info, history, task_description, start_step,
            prefix_length, predefined_actions
        )

    async def _rollout(
        self, env, observation, info, history, task_description, start_step,
        prefix_length, predefined_actions
    ) -> List[Experience]:
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

        memory = self.format_messages()
        turn_experiences: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}

        for r in range(start_step, self.max_env_steps):
            user_content = build_user_content(
                task_description, history, observation, info.get("admissible_commands", []), r
            )
            memory = memory + [{"role": "user", "content": user_content}]
            MAX_MEMORY_TURNS = 10
            recent_memory = memory[-(MAX_MEMORY_TURNS * 2):]

            # Determine who acts: teacher for first prefix_length turns, student after
            use_teacher_actor = (r - start_step) < prefix_length

            if use_teacher_actor and predefined_actions:
                # Teacher executes predefined action
                action = predefined_actions[r]
                response_text = f"<action>{action}</action>"
            else:
                # Student generates
                responses = await self.model.chat_async(recent_memory, **kwargs)
                response = responses[0]
                response_text = response.response_text or ""
                action = parse_action(response_text) or ""

            memory.append({"role": "assistant", "content": response_text})

            # Teacher generates target action on same state
            teacher_kwargs = {"temperature": 0.4, "max_tokens": 512, "n": 1}
            try:
                teacher_responses = await self.teacher_model.chat_async(
                    recent_memory, **teacher_kwargs
                )
                teacher_action = canonicalize_action(
                    teacher_responses[0].response_text or ""
                )
            except Exception:
                teacher_action = None

            # Build SFT experience if teacher provided a valid action
            if teacher_action:
                exp = await make_cross_family_sft_experience(
                    self.model, recent_memory, teacher_action, self.temperature
                )
                if exp is not None:
                    exp.info["teacher_action"] = teacher_action
                    exp.info["student_action"] = action
                    exp.metrics["action_match"] = (
                        1.0 if teacher_action.strip() == action.strip() else 0.0
                    )
                    exp.reward = 0.0
                    exp.eid.run = getattr(self, "run_id_base", 0)
                    exp.eid.step = len(turn_experiences)
                    turn_experiences.append(exp)

            # Execute action in environment
            obs_format = format_observation(observation)
            history.append(_format_history(obs_format, r + 1, action))
            observation, reward, done, info = env.step(action)

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0

        # Set reward for all experiences
        for exp in turn_experiences:
            exp.reward = self._final_reward

        if turn_experiences:
            last = turn_experiences[-1]
            if last.metrics is None:
                last.metrics = {}
            last.metrics["env_rounds"] = float(self._env_rounds)
            last.metrics["env_done"] = 1.0 if self._env_done else 0.0

        return turn_experiences
