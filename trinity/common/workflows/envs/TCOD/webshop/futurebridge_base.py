# -*- coding: utf-8 -*-
"""
FutureBridge-OPD base workflow for WebShop.

Design (mirrors ALFWorld _FutureBridgeB2FAlfworldBase):
  1. B2F curriculum: the teacher generates the first k steps online (no
     pre-stored expert actions required for WebShop).
  2. KL Bridge: find the Student turn with the highest token-average
     teacher-student discrepancy and insert a teacher demonstration there.

Bridge experience creation (same pattern as Bridge_TCOD_kl_alfworld_workflow):
  - teacher.chat_async(memory_at_t_bridge) -> teacher tokens
  - student.logprobs_async(teacher_tokens)  -> student logprobs on teacher tokens
  - teacher.logprobs_async(teacher_tokens) -> teacher logprobs on teacher tokens
  - exp.logprobs         = student logprobs  (OPD advantage = teacher_lp - student_lp)
  - exp.teacher_logprobs = teacher logprobs
  - exp.reward           = 0.0              (placeholder; not used by the OPD loss)
  - exp.eid.step         = 5000 + bridge_turn  (marks as bridge)

Key difference from ALFWorld:
  - WebShop reward is continuous [0, 1] (based on attribute matching).
  - "Failed" = final_reward < bridge_reward_threshold (default 0.5).
  - Max 15 env steps (vs 30 in ALFWorld).
"""

import copy
import logging
from dataclasses import asdict
from typing import List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.webshop.TCOD_b2f_workflow import (
    TCOD_b2f_webshop_workflow,
)
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

logger = logging.getLogger(__name__)

BRIDGE_STEP_OFFSET = 5000  # eid.step offset distinguishing bridge from normal turns


@WORKFLOWS.register_module("LegacyLiveTeacherB2FWebShopWorkflow")
class LegacyLiveTeacherB2FWebShopWorkflow(TCOD_b2f_webshop_workflow):
    """
    Pure B2F for WebShop: teacher generates first k steps online, no bridge.
    Baseline for comparing against FTB.

    Extends TCOD_b2f_webshop_workflow with live teacher rollout
    (no pre-stored actions needed).
    """

    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        assert self.auxiliary_model_wrappers, "B2F requires a teacher model."
        self.teacher_model = self.auxiliary_model_wrappers[0]
        self.temperature = task.workflow_args.get("temperature", 1.0)
        self._current_training_step = 0
        self._total_training_steps = task.workflow_args.get("total_steps", 250)

    def set_training_progress(self, current_step: int, total_steps: int):
        self._current_training_step = current_step
        self._total_training_steps = total_steps

    def _live_checkpoint_step(self) -> int:
        max_k = self.max_env_steps - 1
        reduction = self._current_training_step // max(1, self.checkpoint_steps)
        return max(0, max_k - reduction)

    async def run_async(self) -> List[Experience]:
        import re as _re
        if self.is_eval:
            env = _create_webshop_env()
            try:
                return await self._run_episode(env, int(self.task_desc))
            finally:
                env.close()

        current_step = 0
        if hasattr(self.task, "batch_id"):
            bid = self.task.batch_id
            if isinstance(bid, int):
                current_step = bid
            elif isinstance(bid, str):
                m = _re.match(r"^(\d+)", bid)
                if m:
                    current_step = int(m.group(1))
        self.set_training_progress(current_step, self._total_training_steps)

        session_id = int(self.task_desc)
        if self.checkpoint_strategy == "linear":
            k = self._live_checkpoint_step()
        else:
            k = 0

        env = _create_webshop_env()
        try:
            if k > 0:
                teacher_result = await self._run_teacher_phase(env, session_id, k)
                if teacher_result is None:
                    # Teacher completed early; fall back to full student episode
                    env.reset(session=session_id)
                    obs = env.observation
                    task_desc = _extract_task_description(obs)
                    return await self._run_student_episode(env, session_id, obs, [], task_desc, 0)
                obs, history, task_desc, start_step = teacher_result
                return await self._run_student_episode(env, session_id, obs, history, task_desc, start_step)
            else:
                env.reset(session=session_id)
                obs = env.observation
                task_desc = _extract_task_description(obs)
                return await self._run_student_episode(env, session_id, obs, [], task_desc, 0)
        finally:
            env.close()

    async def _run_teacher_phase(self, env, session_id: int, k: int):
        """Teacher generates k steps online."""
        env.reset(session=session_id)
        obs = env.observation
        task_desc = _extract_task_description(obs)
        history: List[str] = []
        memory: List[dict] = []
        kwargs_t = {"n": 1, "temperature": self.temperature, "logprobs": 0}

        for step in range(k):
            available_actions = env.get_available_actions()
            formatted_obs = format_observation(obs)
            formatted_actions = _format_available_actions(available_actions)
            if len(history) < HISTORY_LENGTH:
                user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_desc, current_observation=formatted_obs,
                    available_actions=formatted_actions)
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = WEBSHOP_TEMPLATE.format(
                    task_description=task_desc, step_count=step,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str, current_step=step + 1,
                    current_observation=formatted_obs, available_actions=formatted_actions)
            memory = memory + [{"role": "user", "content": user_content}]
            try:
                t_resps = await self.teacher_model.chat_async(memory, **kwargs_t)
                t_resp = t_resps[0]
            except Exception as e:
                logger.warning(f"[B2F] Teacher generation failed: {e}")
                return None
            response_text = t_resp.response_text or ""
            memory.append({"role": "assistant", "content": response_text})
            action = parse_action(response_text)
            action_valid, error_msg = validate_action(action, available_actions)
            history.append(_format_history(formatted_obs, step + 1, action))
            if action_valid:
                obs, reward, done, _ = env.step(action)
                if done:
                    return None
            else:
                obs = error_msg
        return obs, history, task_desc, k

    async def _run_student_episode(self, env, session_id, obs, history, task_desc, start_step):
        """Student runs from start_step with OPD scoring. No bridge."""
        self._env_done = False
        self._env_rounds = start_step
        self._final_reward = 0.0
        turn_responses: List[Experience] = []
        memory: List[dict] = []
        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(start_step, self.max_env_steps):
            available_actions = env.get_available_actions()
            formatted_obs = format_observation(obs)
            formatted_actions = _format_available_actions(available_actions)
            if len(history) < HISTORY_LENGTH:
                user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_desc, current_observation=formatted_obs,
                    available_actions=formatted_actions)
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = WEBSHOP_TEMPLATE.format(
                    task_description=task_desc, step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str, current_step=r + 1,
                    current_observation=formatted_obs, available_actions=formatted_actions)
            memory = memory + [{"role": "user", "content": user_content}]
            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})
            turn_responses.append(response)
            action = parse_action(response_text)
            action_valid, error_msg = validate_action(action, available_actions)
            history.append(_format_history(formatted_obs, r + 1, action))
            if action_valid:
                obs, reward, done, _ = env.step(action)
            else:
                obs = error_msg; reward = 0.0; done = False
            if done:
                self._env_done = True; self._env_rounds = r + 1; self._final_reward = float(reward); break
        else:
            self._env_rounds = self.max_env_steps

        # OPD scoring
        per_turn_kl: List[float] = []
        for i, response in enumerate(turn_responses):
            tl_full = await self.teacher_model.logprobs_async(tokens=response.tokens.tolist(), temperature=self.temperature)
            rs = response.prompt_length - 1
            teacher_lp = tl_full[rs:]; student_lp = response.logprobs
            response.teacher_logprobs = teacher_lp
            response.reward = float(self._final_reward)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = start_step + i
            per_turn_kl.append((student_lp - teacher_lp).sum().item())

        if turn_responses:
            last = turn_responses[-1]
            if last.metrics is None: last.metrics = {}
            last.metrics.update({
                "student_env_rounds": self._env_rounds - start_step,
                "teacher_env_rounds": start_step,
                "if_teacher": 1 if start_step > 0 else 0,
                "env_rounds": self._env_rounds,
                "env_done": 1.0 if self._env_done else 0.0,
                "final_reward": self._final_reward,
                "kl_divergence": sum(per_turn_kl),
                "session_id": float(session_id),
                "bridge_verified": 0, "is_bridge": 0,
            })
        return turn_responses

    async def _run_episode(self, env, session_id: int) -> List[Experience]:
        """Eval: full student episode."""
        env.reset(session=session_id)
        obs = env.observation
        task_desc = _extract_task_description(obs)
        return await self._run_student_episode(env, session_id, obs, [], task_desc, 0)


@WORKFLOWS.register_module("_FutureBridgeWebShopBase")
class _FutureBridgeWebShopBase(TCOD_b2f_webshop_workflow):
    """
    FutureBridge-OPD base for WebShop:
      - B2F linear curriculum with live teacher rollout (the teacher
        generates the first k steps online);
      - KL Bridge generation from the Student suffix.
    """

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

        wargs = task.workflow_args or {}
        self.bridge_reward_threshold = wargs.get("bridge_reward_threshold", 0.5)
        self.bridge_kl_lambda = wargs.get("bridge_kl_lambda", 0.5)
        self.bridge_max_per_ep = wargs.get("bridge_max_per_ep", 1)
        # Online B2F prefix actions of the current episode, used by
        # subclasses to rebuild the environment for future validation.
        self._teacher_actions: List[str] = []
        self._episode_session_id: int = 0

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run_async(self) -> List[Experience]:
        if self.is_eval:
            env = _create_webshop_env()
            try:
                # _run_full_student_episode returns (student_exps, bridge_exps); flatten for eval
                student_exps, _ = await self._run_full_student_episode(env, int(self.task_desc))
                return student_exps
            finally:
                env.close()

        # Determine current training step from batch_id
        import re as _re
        current_step = 0
        if hasattr(self.task, "batch_id"):
            bid = self.task.batch_id
            if isinstance(bid, int):
                current_step = bid
            elif isinstance(bid, str):
                m = _re.match(r"^(\d+)", bid)
                if m:
                    current_step = int(m.group(1))
        self.set_training_progress(current_step, self.total_steps)

        session_id = int(self.task_desc)
        self._episode_session_id = session_id

        if self.checkpoint_strategy == "linear":
            k = self._live_checkpoint_step()
        else:
            k = 0

        env = _create_webshop_env()
        try:
            if k > 0:
                teacher_result = await self._run_teacher_phase(env, session_id, k)
                if teacher_result is None:
                    # Teacher completed the task early; fall back to a full
                    # student episode.
                    self._teacher_actions = []
                    student_exps, bridge_exps = await self._run_full_student_episode(
                        env, session_id
                    )
                    return student_exps + bridge_exps
                obs, history, task_desc, start_step = teacher_result
                student_exps, bridge_exps = await self._run_student_phase(
                    env, session_id, obs, history, task_desc, start_step
                )
                return student_exps + bridge_exps
            self._teacher_actions = []
            student_exps, bridge_exps = await self._run_full_student_episode(
                env, session_id
            )
            return student_exps + bridge_exps
        finally:
            env.close()

    # ── Teacher phase (B2F): teacher generates the first k steps online ───────

    def _live_checkpoint_step(self) -> int:
        """
        Number of leading steps the teacher generates online in the current
        training step. Decreases by one every `checkpoint_steps` training
        steps, handing control to the student progressively earlier.
        """
        max_k = self.max_env_steps - 1
        reduction = self._current_training_step // max(1, self.checkpoint_steps)
        return max(0, max_k - reduction)

    async def _run_teacher_phase(
        self, env, session_id: int, k: int
    ) -> Optional[Tuple]:
        """
        Teacher runs the first k steps from the start of the episode.
        Returns (last_obs, history, task_desc, start_step) for the student
        handoff, or None when the teacher completes the task or fails.
        """
        self._teacher_actions = []
        env.reset(session=session_id)
        obs = env.observation
        task_desc = _extract_task_description(obs)
        history: List[str] = []
        memory: List[dict] = []
        kwargs_t = {"n": 1, "temperature": self.temperature, "logprobs": 0}

        for step in range(k):
            available_actions = env.get_available_actions()
            formatted_obs = format_observation(obs)
            formatted_actions = _format_available_actions(available_actions)

            if len(history) < HISTORY_LENGTH:
                user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_desc,
                    current_observation=formatted_obs,
                    available_actions=formatted_actions,
                )
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = WEBSHOP_TEMPLATE.format(
                    task_description=task_desc,
                    step_count=step,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=step + 1,
                    current_observation=formatted_obs,
                    available_actions=formatted_actions,
                )

            memory = memory + [{"role": "user", "content": user_content}]
            try:
                t_resps = await self.teacher_model.chat_async(memory, **kwargs_t)
                t_resp = t_resps[0]
            except Exception as e:
                logger.warning(f"[FutureBridge] Teacher generation failed at step {step}: {e}")
                return None

            response_text = t_resp.response_text or ""
            memory.append({"role": "assistant", "content": response_text})
            action = parse_action(response_text)
            action_valid, error_msg = validate_action(action, available_actions)
            history.append(_format_history(formatted_obs, step + 1, action))
            self._teacher_actions.append(action if action_valid else "")

            if action_valid:
                obs, reward, done, _ = env.step(action)
                if done:
                    return None
            else:
                obs = error_msg

        return obs, history, task_desc, k

    # ── Student phase ──────────────────────────────────────────────────────────

    async def _run_student_phase(
        self,
        env,
        session_id: int,
        observation,
        history: List[str],
        task_description: str,
        start_step: int,
    ) -> Tuple[List[Experience], List[Experience]]:
        """
        Student runs from start_step to max_env_steps.
        Returns (student_experiences, bridge_experiences).
        """
        self._env_done = False
        self._env_rounds = start_step
        self._final_reward = 0.0

        memory: List[dict] = []
        turn_responses: List[Experience] = []
        memory_snapshots: List[List[dict]] = []  # memory before each student turn

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        obs = observation
        for r in range(start_step, self.max_env_steps):
            available_actions = env.get_available_actions()
            formatted_obs = format_observation(obs)
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
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=formatted_obs,
                    available_actions=formatted_actions,
                )

            memory = memory + [{"role": "user", "content": user_content}]
            memory_snapshots.append(list(memory))  # snapshot before model call

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "_FutureBridgeWebShopBase requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            action_valid, error_msg = validate_action(action, available_actions)
            history.append(_format_history(formatted_obs, r + 1, action))

            if action_valid:
                obs, reward, done, _ = env.step(action)
            else:
                obs = error_msg
                reward = 0.0
                done = False

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = float(reward)
                break
        else:
            self._env_rounds = self.max_env_steps

        # ── Compute teacher logprobs and per-turn KL ──────────────────────────
        per_turn_kl: List[float] = []
        for i, response in enumerate(turn_responses):
            teacher_lp_full = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )
            rs = response.prompt_length - 1
            teacher_lp = teacher_lp_full[rs:]
            student_lp = response.logprobs

            assert len(teacher_lp) == len(student_lp), (
                f"Length mismatch: teacher={len(teacher_lp)}, student={len(student_lp)}"
            )

            response.teacher_logprobs = teacher_lp
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = start_step + i

            # Token-average teacher-student discrepancy (paper Eq. 3): sum
            # normalized by the number of student-generated tokens, so turn
            # selection is not biased toward longer responses.
            kl = ((student_lp - teacher_lp).sum() / max(1, len(student_lp))).item()
            per_turn_kl.append(kl)

        trajectory_kl = sum(per_turn_kl)

        # ── Set summary metrics on last response ──────────────────────────────
        if turn_responses:
            last = turn_responses[-1]
            if last.metrics is None:
                last.metrics = {}
            n_student = self._env_rounds - start_step
            last.metrics["student_env_rounds"] = n_student
            last.metrics["teacher_env_rounds"] = start_step
            last.metrics["if_teacher"] = 1 if start_step > 0 else 0
            last.metrics["expected_teacher_env_rounds"] = start_step
            last.metrics["env_rounds"] = self._env_rounds
            last.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last.metrics["final_reward"] = self._final_reward  # WebShop attribute-matching score [0,1]
            last.metrics["kl_divergence"] = trajectory_kl
            last.metrics["session_id"] = float(session_id)
            last.metrics["bridge_verified"] = 0
            last.metrics["is_bridge"] = 0

        # ── Try KL bridge on failed episodes (training only) ──────────────────
        bridge_exps: List[Experience] = []
        if not self.is_eval:
            bridge_exps = await self._try_kl_bridge(
                start_step, memory_snapshots, per_turn_kl, turn_responses
            )

        return turn_responses, bridge_exps

    async def _run_full_student_episode(
        self, env, session_id: int
    ) -> Tuple[List[Experience], List[Experience]]:
        """Full student episode (no teacher B2F prefix). Used in eval and when k=0."""
        env.reset(session=session_id)
        obs = env.observation
        task_desc = _extract_task_description(obs)

        return await self._run_student_phase(
            env, session_id, obs, [], task_desc, start_step=0
        )

    # ── KL Bridge ─────────────────────────────────────────────────────────────

    async def _try_kl_bridge(
        self,
        start_step: int,
        memory_snapshots: List[List[dict]],
        per_turn_kl: List[float],
        turn_responses: List[Experience],
    ) -> List[Experience]:
        """
        Try to add KL bridge experiences to a failed episode.
        Bridge is triggered when final_reward < bridge_reward_threshold.
        """
        if self._final_reward >= self.bridge_reward_threshold:
            return []  # Episode succeeded; no bridge needed
        if not per_turn_kl:
            return []

        # Find turn(s) with highest KL (up to bridge_max_per_ep)
        indexed_kl = sorted(enumerate(per_turn_kl), key=lambda x: x[1], reverse=True)
        bridge_exps: List[Experience] = []

        for rank, (turn_idx, trigger_kl) in enumerate(indexed_kl):
            if len(bridge_exps) >= self.bridge_max_per_ep:
                break
            if trigger_kl <= 0:
                continue

            bridge_weight = 1.0  # can be extended to rank-based weighting
            bridge_idx = start_step + turn_idx  # global episode step index

            exps = await self._generate_kl_bridge(
                memory_at_turn=memory_snapshots[turn_idx],
                trigger_kl=trigger_kl,
                bridge_idx=bridge_idx,
                bridge_weight=bridge_weight,
            )
            bridge_exps.extend(exps)

        return bridge_exps

    async def _generate_kl_bridge(
        self,
        memory_at_turn: List[dict],
        trigger_kl: float,
        bridge_idx: int,
        bridge_weight: float = 1.0,
    ) -> List[Experience]:
        """
        Teacher generates its optimal response at the high-KL turn.
        Student learns to imitate via OPD advantage (teacher_lp - student_lp).

        No env interaction needed: only the conversation context (memory_at_turn)
        is required to generate and score the bridge response.
        """
        kwargs_t = {
            **asdict(self.task.rollout_args),
            "n": 1,
            "logprobs": 0,
            "temperature": self.temperature,
        }

        try:
            bridge_resps = await self.teacher_model.chat_async(
                memory_at_turn, **kwargs_t
            )
            bridge_resp = bridge_resps[0]

            full_tokens = bridge_resp.tokens.tolist()
            rs = bridge_resp.prompt_length - 1

            # Score teacher's tokens with both models
            student_lp_full = await self.model.logprobs_async(
                tokens=full_tokens, temperature=self.temperature
            )
            teacher_lp_full = await self.teacher_model.logprobs_async(
                tokens=full_tokens, temperature=self.temperature
            )
        except Exception as e:
            logger.warning(f"[FutureBridge] Bridge generation failed at step {bridge_idx}: {e}")
            return []

        student_lp = student_lp_full[rs:]
        teacher_lp = teacher_lp_full[rs:]

        if len(student_lp) != len(teacher_lp) or len(student_lp) == 0:
            return []

        exp = copy.copy(bridge_resp)
        exp.logprobs = student_lp          # student scores teacher's tokens
        exp.teacher_logprobs = teacher_lp  # teacher scores its own tokens
        exp.reward = 0.0                   # placeholder; not used by the OPD loss
        exp.eid.run = getattr(self, "run_id_base", 0)
        exp.eid.step = BRIDGE_STEP_OFFSET + bridge_idx

        if exp.metrics is None:
            exp.metrics = {}
        exp.metrics["bridge_verified"] = 1
        exp.metrics["bridge_lambda"] = self.bridge_kl_lambda * bridge_weight
        exp.metrics["trigger_kl"] = trigger_kl
        exp.metrics["bridge_weight"] = bridge_weight
        exp.metrics["env_done"] = 1.0
        exp.metrics["is_bridge"] = 1
        exp.metrics["if_teacher"] = 0

        return [exp]
