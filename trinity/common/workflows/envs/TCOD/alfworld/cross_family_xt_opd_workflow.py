# -*- coding: utf-8 -*-
"""Cross-Tokenizer OPD workflow for AlfWorld (Qwen teacher → Llama student).

Student (Llama) rolls out in ALFWorld. For each turn, both student and teacher
score the SAME response text using their own tokenizers. Per-byte normalized
logprob gap becomes the turn-level advantage. PPO surrogate loss trains student.

No token alignment needed — both models independently score the same text.
"""

import re
from dataclasses import asdict
from typing import List, Optional

import torch

from trinity.common.experience import CustomField, Experience
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
)
from trinity.common.workflows.envs.TCOD.alfworld.cross_tokenizer_opd_utils import (
    compute_student_score_from_logprobs,
    score_response_per_byte,
    compute_turn_level_advantages,
)


# Custom fields passed from experience.info to batch tensors
_XT_OPD_CUSTOM_FIELDS = [
    CustomField("is_expert", "expert_mask", torch.bool),
    CustomField("turn_advantage", "turn_advantage", torch.float32),
]


@WORKFLOWS.register_module("cf_xt_opd_alfworld_workflow")
class CFXTOPDAlfworldWorkflow(Workflow):
    """Cross-tokenizer OPD: student rollout + per-byte turn-level advantage.

    Required yaml config:
      algorithm.policy_loss_fn: mix (PPO for student, SFT for bridge)
      algorithm.advantage_fn: cross_tokenizer_opd
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
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self._reset_task(task)

        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "CF-XT-OPD requires teacher model."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.is_eval = task.is_eval
        self.total_steps = task.workflow_args.get("total_steps", 250)
        self.checkpoint_steps = task.workflow_args.get("checkpoint_steps", 5)
        self._current_training_step = 0

    def _reset_task(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

    def reset(self, task: Task):
        self._reset_task(task)

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def set_training_progress(self, current_step, total_steps):
        self._current_training_step = current_step

    def _linear_checkpoint_step(self, predefined_actions):
        if not predefined_actions:
            return 0
        max_expert = len(predefined_actions) - 1
        reduction = self._current_training_step // self.checkpoint_steps
        return max(0, min(max_expert - reduction, max_expert))

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
        if self.is_eval:
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env, 0, None)
            finally:
                env.close()

        # Extract training step
        current_step = 0
        if hasattr(self.task, 'batch_id'):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                m = re.match(r'^(\d+)', batch_id)
                if m:
                    current_step = int(m.group(1))
        self.set_training_progress(current_step, self.total_steps)

        predefined_actions = self.raw_task.get("actions", None)
        k_star = self._linear_checkpoint_step(predefined_actions)

        if k_star > 0 and predefined_actions:
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                env = _create_alfworld_env(self.task_desc)
                try:
                    obs, info = env.reset()
                    return await self._run_episode(env, 0, None)
                finally:
                    env.close()
            env, obs, info, history, task_desc, start_step, done = result
            if done:
                env.close()
                return []
            try:
                return await self._run_episode(env, start_step, predefined_actions)
            finally:
                env.close()
        else:
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                return await self._run_episode(env, 0, None)
            finally:
                env.close()

    async def _run_episode(self, env, start_step, predefined_actions) -> List[Experience]:
        """Student rollout + cross-tokenizer turn-level OPD scoring."""
        if start_step == 0:
            observation, info = env.reset()
            task_description = _extract_task(observation)
            history = []
        else:
            observation = env.observation if hasattr(env, 'observation') else None
            info = {}
            task_description = _extract_task(observation) if observation else ""

        return await self._rollout_with_scoring(
            env, observation, info, history, task_description, start_step
        )

    async def _rollout_with_scoring(
        self, env, observation, info, history, task_description, start_step
    ) -> List[Experience]:
        """Student rollout + cross-tokenizer per-byte scoring."""
        self._env_done = False
        self._env_rounds = start_step
        self._final_reward = 0.0

        memory = self.format_messages()
        turn_experiences: List[Experience] = []
        turn_messages: List[List[dict]] = []
        turn_response_texts: List[str] = []
        # Cache for subclass (CF-XT-FTB) to access
        self._turn_messages_cache = turn_messages

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0
        for r in range(start_step, self.max_env_steps):
            user_content = build_user_content(
                task_description, history, observation,
                info.get("admissible_commands", []), r
            )
            memory = memory + [{"role": "user", "content": user_content}]

            MAX_MEMORY_TURNS = 10
            recent_memory = memory[-(MAX_MEMORY_TURNS * 2):]
            turn_messages.append(list(recent_memory))

            # Student generates with logprobs
            responses = await self.model.chat_async(recent_memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""

            if response.logprobs is None:
                raise RuntimeError("CF-XT-OPD requires student logprobs.")

            turn_response_texts.append(response_text)
            memory.append({"role": "assistant", "content": response_text})

            # Execute student action
            action = parse_action(response_text) or ""
            obs_fmt = format_observation(observation)
            history = history + [_format_history(obs_fmt, r + 1, action)]
            observation, _, done, info = env.step(action)

            # Store experience
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = r - start_step
            if response.metrics is None:
                response.metrics = {}
            turn_experiences.append(response)

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0

        # ── Compute cross-tokenizer turn-level gaps ──
        gaps = []
        for i, exp in enumerate(turn_experiences):
            response_text = turn_response_texts[i]

            # Student score: use existing logprobs
            s_score = compute_student_score_from_logprobs(exp.logprobs, response_text)

            # Teacher score: re-tokenize with teacher tokenizer
            t_score = await score_response_per_byte(
                self.teacher_model, turn_messages[i], response_text, self.temperature
            )

            if s_score is not None and t_score is not None:
                gap = t_score - s_score
            else:
                gap = 0.0
            gaps.append(gap)

            # Set teacher_logprobs = zeros (CrossTokenizerOpdAdvantage doesn't use them)
            exp.teacher_logprobs = torch.zeros_like(exp.logprobs)

            # Mark as non-expert (student turn → PPO loss)
            exp.info["is_expert"] = False
            exp.info["turn_advantage"] = gap  # Pre-normalization; advantage_fn normalizes
            exp.custom_fields = list(_XT_OPD_CUSTOM_FIELDS)

            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["xt_gap"] = float(gap)
            exp.metrics["s_score"] = float(s_score) if s_score else 0.0
            exp.metrics["t_score"] = float(t_score) if t_score else 0.0

        # Normalize: subtract batch mean, clamp
        normalized_advs = compute_turn_level_advantages(gaps, clamp_val=5.0)
        for i, exp in enumerate(turn_experiences):
            exp.info["turn_advantage"] = normalized_advs[i]
            exp.metrics["turn_advantage"] = float(normalized_advs[i])

        # Set reward
        for exp in turn_experiences:
            exp.reward = self._final_reward

        if turn_experiences:
            last = turn_experiences[-1]
            if last.metrics is None:
                last.metrics = {}
            last.metrics["env_rounds"] = float(self._env_rounds)
            last.metrics["env_done"] = 1.0 if self._env_done else 0.0

        return turn_experiences
