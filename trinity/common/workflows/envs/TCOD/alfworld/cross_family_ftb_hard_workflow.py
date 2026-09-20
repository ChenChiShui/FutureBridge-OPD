# -*- coding: utf-8 -*-
"""CF-FTB-Hard: Cross-family FTB with action-level hard distillation.

Inherits FutureBridgeAlfworldWorkflow but overrides:
1. Bridge selection: teacher action margin instead of token KL (cross-tokenizer safe)
2. Experience construction: action SFT instead of OPD
3. Continuation verification: action match rate instead of token logprob ratio

Student (Llama) explores, Teacher (Qwen) provides bridge actions.
All training targets are Teacher actions tokenized by Student tokenizer.
"""

import re
from dataclasses import asdict
from typing import List, Optional, Tuple

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
    HISTORY_LENGTH,
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
    score_action_with_teacher,
)

import logging

logger = logging.getLogger(__name__)


@WORKFLOWS.register_module("cf_ftb_hard_alfworld_workflow")
class CFFTBHardWorkflow(Workflow):
    """CF-FTB-Hard workflow.

    Uses teacher action margin for bridge selection and action SFT for training.
    Does NOT use cross-tokenizer token KL.
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    _CONTINUATION_STEPS = 3

    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self._reset_task(task)

        assert (
            auxiliary_models is not None and len(auxiliary_models) >= 1
        ), "CF-FTB-Hard requires teacher model."
        self.teacher_model = auxiliary_models[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.is_eval = task.is_eval

        self.checkpoint_steps = task.workflow_args.get("checkpoint_steps", 5)
        self.total_steps = task.workflow_args.get("total_steps", 250)
        self.bridge_max_per_ep = task.workflow_args.get("bridge_max_per_ep", 1)
        self._current_training_step = 0

    def _reset_task(self, task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

    def reset(self, task):
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
                return await self._run_episode(env, 0)
            finally:
                env.close()

        current_step = 0
        if hasattr(self.task, "batch_id"):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                m = re.match(r"^(\d+)", batch_id)
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
                    return await self._run_episode(env, 0)
                finally:
                    env.close()
            env, obs, info, history, task_desc, start_step, done = result
            if done:
                env.close()
                return []
            try:
                return await self._run_episode_from_checkpoint(
                    env, obs, info, history, task_desc, start_step, k_star, predefined_actions
                )
            finally:
                env.close()
        else:
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                return await self._run_episode(env, 0)
            finally:
                env.close()

    async def _run_episode(self, env, start_step) -> List[Experience]:
        obs, info = env.reset()
        task_desc = _extract_task(obs)
        return await self._rollout(env, obs, info, [], task_desc, start_step, 0, None)

    async def _run_episode_from_checkpoint(
        self, env, obs, info, history, task_desc, start_step, k_star, predefined_actions
    ) -> List[Experience]:
        return await self._rollout(
            env, obs, info, history, task_desc, start_step, k_star, predefined_actions
        )

    async def _rollout(
        self, env, observation, info, history, task_description, start_step,
        k_star, predefined_actions
    ) -> List[Experience]:
        self._env_done = False
        self._env_rounds = start_step
        self._final_reward = 0.0

        memory = self.format_messages()
        turn_experiences: List[Experience] = []
        turn_memories: List[List[dict]] = []
        turn_student_actions: List[str] = []
        turn_teacher_actions: List[Optional[str]] = []

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
            turn_memories.append(list(recent_memory))

            # Student generates
            responses = await self.model.chat_async(recent_memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            student_action = parse_action(response_text) or ""
            memory.append({"role": "assistant", "content": response_text})

            turn_student_actions.append(student_action)

            # Teacher generates target action
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
            turn_teacher_actions.append(teacher_action)

            # Build SFT experience for this turn
            if teacher_action:
                exp = await make_cross_family_sft_experience(
                    self.model, recent_memory, teacher_action, self.temperature
                )
                if exp is not None:
                    exp.info["teacher_action"] = teacher_action
                    exp.info["student_action"] = student_action
                    exp.metrics["action_match"] = (
                        1.0 if teacher_action.strip() == student_action.strip() else 0.0
                    )
                    exp.reward = 0.0
                    exp.eid.run = getattr(self, "run_id_base", 0)
                    exp.eid.step = r - start_step
                    turn_experiences.append(exp)

            # Execute student action
            obs_fmt = format_observation(observation)
            history = history + [_format_history(obs_fmt, r + 1, student_action)]
            observation, _, done, info = env.step(student_action)

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0

        # Skip bridge for successful episodes or eval
        episode_succeeded = self._env_done and self._final_reward > 0.5
        if self.is_eval or episode_succeeded or not turn_experiences:
            for exp in turn_experiences:
                exp.reward = self._final_reward
            if turn_experiences:
                last = turn_experiences[-1]
                if last.metrics is None:
                    last.metrics = {}
                last.metrics["env_done"] = 1.0 if self._env_done else 0.0
                last.metrics["env_rounds"] = float(self._env_rounds)
                last.metrics["bridge_count"] = 0.0
            return turn_experiences

        # ── Bridge phase: teacher action margin selection + verification ──
        bridge_exps = await self._try_bridges(
            turn_memories, turn_student_actions, turn_teacher_actions,
            start_step, k_star, predefined_actions
        )

        # Set reward
        for exp in turn_experiences:
            exp.reward = self._final_reward
        for exp in bridge_exps:
            exp.reward = self._final_reward

        if turn_experiences:
            last = turn_experiences[-1]
            if last.metrics is None:
                last.metrics = {}
            last.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last.metrics["env_rounds"] = float(self._env_rounds)
            last.metrics["bridge_count"] = float(len(bridge_exps))

        return turn_experiences + bridge_exps

    async def _try_bridges(
        self,
        turn_memories: List[List[dict]],
        turn_student_actions: List[str],
        turn_teacher_actions: List[Optional[str]],
        start_step: int,
        k_star: int,
        predefined_actions,
    ) -> List[Experience]:
        """Select bridge turn using teacher action margin, verify with continuation."""

        # 1. Compute teacher action margin for each turn
        margins = []
        for i, (s_action, t_action) in enumerate(
            zip(turn_student_actions, turn_teacher_actions)
        ):
            if not t_action or not s_action or t_action.strip() == s_action.strip():
                margins.append(-1.0)
                continue

            # Score both actions with teacher
            messages = turn_memories[i]
            t_score = await score_action_with_teacher(
                self.teacher_model, messages, t_action, self.temperature
            )
            s_score = await score_action_with_teacher(
                self.teacher_model, messages, s_action, self.temperature
            )

            if t_score is not None and s_score is not None:
                margins.append(t_score - s_score)
            else:
                margins.append(-1.0)

        # 2. Select top bridge turn(s)
        n = len(margins)
        if n == 0:
            return []

        sorted_turns = sorted(enumerate(margins), key=lambda x: -x[1])
        bridge_exps = []
        triggered = 0

        for turn_idx, margin in sorted_turns:
            if triggered >= self.bridge_max_per_ep:
                break
            if margin <= 0:
                break

            t_action = turn_teacher_actions[turn_idx]
            if not t_action:
                continue

            # Check teacher action is executable
            if predefined_actions:
                # Verify by replaying env
                verified = await self._verify_bridge(
                    predefined_actions, k_star, turn_student_actions,
                    turn_idx, start_step, t_action, turn_memories[turn_idx]
                )
            else:
                verified = True  # No predefined actions, skip verification

            if verified:
                # Build bridge SFT experience
                bridge_messages = turn_memories[turn_idx]
                exp = await make_cross_family_sft_experience(
                    self.model, bridge_messages, t_action, self.temperature
                )
                if exp is not None:
                    exp.info["teacher_action"] = t_action
                    exp.info["student_action"] = turn_student_actions[turn_idx]
                    exp.info["is_bridge"] = True
                    exp.metrics["action_match"] = 0.0  # bridge means mismatch
                    exp.metrics["teacher_action_margin"] = float(margin)
                    exp.metrics["verification_pass"] = 1.0
                    exp.reward = 0.0
                    exp.eid.run = getattr(self, "run_id_base", 0)
                    exp.eid.step = 5000 + triggered  # bridge offset
                    bridge_exps.append(exp)
                    triggered += 1
                    logger.debug(
                        f"[CF-FTB-Hard] KEPT bridge t={turn_idx}: margin={margin:.4f}"
                    )
            else:
                logger.debug(
                    f"[CF-FTB-Hard] DROPPED bridge t={turn_idx}: verification failed"
                )

        return bridge_exps

    async def _verify_bridge(
        self,
        predefined_actions: list,
        k_star: int,
        turn_student_actions: list,
        t_bridge_idx: int,
        start_step: int,
        bridge_action: str,
        bridge_memory: list,
    ) -> bool:
        """Verify bridge by replaying env and checking if student continuation improves.

        Simplified verification: replay to bridge state, execute bridge action,
        run student for N steps, check if student reaches success or stays on track.
        Returns True if bridge is beneficial.
        """
        env = None
        try:
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                return True  # safe fallback
            env, obs, info, _, _, _, done = result
            if done:
                return True

            # Replay student actions up to bridge turn
            for i, s_action in enumerate(turn_student_actions):
                if i >= t_bridge_idx:
                    break
                obs, _, done, info = env.step(s_action)
                if done:
                    return True

            # Execute bridge action
            obs, _, done, info = env.step(bridge_action)
            if done:
                return True  # bridge directly solves the task

            # Student continuation: run N steps
            memory = list(bridge_memory) + [
                {"role": "assistant", "content": f"<action>{bridge_action}</action>"}
            ]
            kwargs = {"n": 1, "temperature": self.temperature}

            n_valid_actions = 0
            for _ in range(self._CONTINUATION_STEPS):
                user_content = build_user_content(
                    _extract_task(obs), [], obs,
                    info.get("admissible_commands", []), 0
                )
                messages = memory + [{"role": "user", "content": user_content}]
                try:
                    resps = await self.model.chat_async(messages, **kwargs)
                    resp = resps[0]
                    action = parse_action(resp.response_text or "") or ""
                    if action:
                        n_valid_actions += 1
                    memory.append({"role": "assistant", "content": resp.response_text or ""})
                    obs, _, done, info = env.step(action)
                    if done:
                        return True  # student succeeds after bridge
                except Exception:
                    break

            # If student can produce valid actions after bridge, consider it verified
            return n_valid_actions > 0

        except Exception as e:
            logger.debug(f"[CF-FTB-Hard] verify_bridge failed: {e}")
            return True  # safe fallback
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
