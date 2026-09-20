# -*- coding: utf-8 -*-
"""
FutureBridge-OPD (FTB) workflow implementations for ALFWorld.

Main method:
  FutureBridgeAlfworldWorkflow  -- FTB (Full): candidate localization + teacher bridge + future validation

Ablations (used in Table 4):
  FutureBridgeNoBridgeExecutionAlfworldWorkflow -- FTB w/o Bridge Exec.: future gate without env bridge execution
  FutureBridgeNoFutureValidationAlfworldWorkflow -- FTB w/o Future Validation: bridge without future gate

Base classes (required by inheritance):
  _TeacherReliableAnchorAlfworldBase         -- base: B2F + bridge-KL
  _FutureBridgeB2FAlfworldBase     -- B2F anchor variant
  _FutureBridgeGateAlfworldBase  -- shared gate base (NoBridgeExecution parent)
"""

import logging
import math
from typing import List, Optional, Tuple

from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import Task
from trinity.common.workflows.envs.TCOD.alfworld.Bridge_TCOD_kl_workflow import (
    Bridge_TCOD_kl_alfworld_workflow,
)
from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    parse_action,
    format_observation,
    _format_history,
    _create_alfworld_env,
    _create_alfworld_env_with_checkpoint,
    _extract_task,
)

logger = logging.getLogger(__name__)

_BUDGET_BUCKETS = [2, 4, 8, 16, 30]


def _bucketize_M(M: int) -> int:
    for b in _BUDGET_BUCKETS:
        if M <= b:
            return b
    return _BUDGET_BUCKETS[-1]


class _TeacherReliableAnchorAlfworldBase(Bridge_TCOD_kl_alfworld_workflow):
    """
    FutureBridge-OPD: FLEX anchor selection + Bridge-KL + binary reliability gate.

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)

    # (implementation detail)
      # (implementation detail)
    """

    # anchor types that are considered teacher-reliable → bridge ON
    _RELIABLE_TYPES = {"hard", "b2f"}

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        # Per-worker FLEX executability cache: (task_desc, k, M_bucket) → bool
        self._flex_cache: dict = {}
        # Reliability flag set per episode in run_async, read in _try_kl_bridges
        self._current_anchor_reliable: bool = True
        self._current_anchor_type: str = "b2f"

    # ── FLEX: Teacher executability check ─────────────────────────────────────
    # (Copied from FLEX_TCOD_workflow; kept here to avoid diamond inheritance)

    async def _teacher_executable_within_M(
        self,
        game_file: str,
        expert_actions: List[str],
        k: int,
        M: int,
    ) -> bool:
        # (implementation detail)
        result = _create_alfworld_env_with_checkpoint(game_file, expert_actions, k)
        if result is None:
            logger.warning(f"[FutureBridge] env creation returned None: game={game_file}, k={k}")
            return False

        env, obs, info, history, task_desc, _, done = result
        if done:
            env.close()
            return True

        memory = self.format_messages()
        kwargs_teacher = {"n": 1, "temperature": self.temperature, "logprobs": 0}

        try:
            for turn in range(M):
                admissible = info.get("admissible_commands", [])
                if admissible and isinstance(admissible[0], list):
                    admissible = admissible[0]
                admissible = [s for s in admissible if s != "help"]
                reformatted = "\n ".join(f"'{s}'" for s in admissible)

                if len(history) < HISTORY_LENGTH:
                    user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=format_observation(obs),
                        admissible_actions=reformatted,
                    )
                else:
                    user_content = ALFWORLD_TEMPLATE.format(
                        task_description=task_desc,
                        step_count=k + turn,
                        history_length=min(HISTORY_LENGTH, len(history)),
                        action_history="\n".join(history[-HISTORY_LENGTH:]),
                        current_step=k + turn + 1,
                        current_observation=format_observation(obs),
                        admissible_actions=reformatted,
                    )

                messages = memory + [{"role": "user", "content": user_content}]
                try:
                    resps = await self.teacher_model.chat_async(messages, **kwargs_teacher)
                    resp_text = resps[0].response_text or ""
                except Exception as e:
                    logger.warning(f"[FutureBridge] teacher chat_async failed: k={k} turn={turn}: {e}")
                    env.close()
                    return False

                action = parse_action(resp_text)
                if not action:
                    for line in resp_text.split('\n'):
                        stripped = line.strip()
                        if stripped.lower().startswith('action:'):
                            action = stripped[len('action:'):].strip()
                            break

                memory = messages + [{"role": "assistant", "content": resp_text}]
                history = history + [_format_history(
                    format_observation(obs), k + turn + 1, action
                )]

                obs, reward, done, info = env.step(action)
                if done:
                    env.close()
                    return reward > 0.0

        except Exception as e:
            logger.warning(f"[FutureBridge] unexpected error in executability check: k={k} M={M}: {e}")
            env.close()
            return False

        env.close()
        return False

    async def _flex_select_anchor(
        self,
        expert_actions: List[str],
        M_u: int,
    ) -> Tuple[int, str]:
        # (implementation detail)
        T = len(expert_actions)
        M_bucket = _bucketize_M(M_u)

        k_hard = max(0, T - 2 * M_u)
        k_b2f  = max(0, T - M_u)
        k_easy = max(0, T - max(1, math.ceil(M_u / 2)))

        candidate_map = {}
        for k, t in [(k_hard, "hard"), (k_b2f, "b2f"), (k_easy, "easy")]:
            if k not in candidate_map:
                candidate_map[k] = t
        candidates_sorted = sorted(candidate_map.keys())

        for k in candidates_sorted:
            anchor_type = candidate_map[k]
            cache_key = (self.task_desc, k, M_bucket)

            if cache_key in self._flex_cache:
                if self._flex_cache[cache_key]:
                    return k, anchor_type
                continue

            success = await self._teacher_executable_within_M(
                game_file=self.task_desc,
                expert_actions=expert_actions,
                k=k,
                M=M_bucket,
            )
            self._flex_cache[cache_key] = success

            if success:
                return k, anchor_type

        return k_b2f, "fallback_b2f"

    # ── Override run_async: FLEX anchor + Bridge-KL episode ───────────────────

    async def run_async(self):
        import re as _re

        if self.is_eval:
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env)
            finally:
                env.close()

        # Extract training step
        current_step = 0
        if hasattr(self.task, 'batch_id'):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                m = _re.match(r'^(\d+)', batch_id)
                if m:
                    current_step = int(m.group(1))

        self.set_training_progress(current_step, self.total_steps)

        predefined_actions = self.raw_task.get("actions", None)

        # No expert trajectory → pure student episode (no bridge possible)
        if not predefined_actions:
            self._current_anchor_reliable = False
            self._current_anchor_type = "fallback_b2f"
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                task_desc = _extract_task(obs)
                return await self._run_episode_from_checkpoint(
                    env, obs, info, [], task_desc, 0
                )
            finally:
                env.close()

        # ── FLEX anchor selection ─────────────────────────────────────────────
        k_b2f_ref = self._linear_checkpoint_step(predefined_actions)
        M_u = max(1, len(predefined_actions) - k_b2f_ref) if k_b2f_ref is not None else 1
        k_star, anchor_type = await self._flex_select_anchor(predefined_actions, M_u)

        # Binary reliability gate
        self._current_anchor_type = anchor_type
        self._current_anchor_reliable = anchor_type in self._RELIABLE_TYPES

        logger.debug(
            f"[FutureBridge] k_b2f={k_b2f_ref} M_u={M_u} k*={k_star} "
            f"type={anchor_type} reliable={self._current_anchor_reliable}"
        )

        # ── Student takeover from k_star ──────────────────────────────────────
        if k_star > 0:
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                # Fallback: pure student from start, no bridge
                self._current_anchor_reliable = False
                env = _create_alfworld_env(self.task_desc)
                try:
                    obs, info = env.reset()
                    task_desc = _extract_task(obs)
                    return await self._run_episode_from_checkpoint(
                        env, obs, info, [], task_desc, 0
                    )
                finally:
                    env.close()

            env, obs, info, history, task_desc, _, done = result
            if done:
                env.close()
                return []
            try:
                return await self._run_episode_from_checkpoint(
                    env, obs, info, history, task_desc, k_star
                )
            finally:
                env.close()
        else:
            # k_star == 0: no teacher prefix, student from start
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                task_desc = _extract_task(obs)
                return await self._run_episode_from_checkpoint(
                    env, obs, info, [], task_desc, 0
                )
            finally:
                env.close()

    # ── Binary reliability gate on bridge ─────────────────────────────────────

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        """
        Reliability gate: only bridge if anchor is teacher-reliable (hard or b2f).
        easy / fallback → return [] (no bridge, degrade to B2F-style OPD).
        """
        if not self._current_anchor_reliable:
            logger.debug(
                f"[FutureBridge] bridge suppressed: anchor_type={self._current_anchor_type}"
            )
            return []

        # Teacher-reliable anchor: full Bridge-KL bridge logic
        return await super()._try_kl_bridges(turn_responses, turn_memories)

    # ── Extra metrics ──────────────────────────────────────────────────────────

    async def _b2f_episode_with_memories(self, env, observation, info, history, task_description, start_step):
        """Inject FutureBridge anchor metrics into the last experience."""
        normal_exps, turn_memories = await super()._b2f_episode_with_memories(
            env, observation, info, history, task_description, start_step
        )
        if normal_exps:
            last = normal_exps[-1]
            if last.metrics is None:
                last.metrics = {}
            _type_code = {"hard": 0.0, "b2f": 1.0, "easy": 2.0, "fallback_b2f": 3.0}
            last.metrics.update({
                "rfb_anchor_type":     _type_code.get(self._current_anchor_type, -1.0),
                "rfb_is_reliable":     float(self._current_anchor_reliable),
                "rfb_is_hard":         float(self._current_anchor_type == "hard"),
                "rfb_is_b2f":          float(self._current_anchor_type == "b2f"),
                "rfb_is_easy":         float(self._current_anchor_type == "easy"),
                "rfb_is_fallback":     float(self._current_anchor_type == "fallback_b2f"),
                "rfb_cache_size":      float(len(self._flex_cache)),
            })
        return normal_exps, turn_memories

    # (implementation detail)

    async def _run_episode_from_checkpoint(
        self, env, observation, info, history, task_description, start_step
    ):
        """
        Override to inject bridge_token_ratio metric.
        bridge_token_ratio = bridge_tokens / (normal_tokens + bridge_tokens)
        # (implementation detail)
        """
        normal_exps, turn_memories = await self._b2f_episode_with_memories(
            env, observation, info, history, task_description, start_step
        )

        if self.is_eval or not normal_exps:
            return normal_exps

        episode_succeeded = self._env_done and self._final_reward > 0.5
        if episode_succeeded:
            # (implementation detail)
            if normal_exps:
                last = normal_exps[-1]
                if last.metrics is None:
                    last.metrics = {}
                last.metrics["bridge_token_ratio"] = 0.0
                last.metrics["bridge_token_count"] = 0.0
                last.metrics["normal_token_count"] = float(
                    sum(len(e.tokens) for e in normal_exps if hasattr(e, 'tokens') and e.tokens is not None)
                )
            return normal_exps

        bridge_exps = await self._try_kl_bridges(normal_exps, turn_memories)

        # (implementation detail)
        import torch
        def _count_tokens(exps):
            total = 0
            for e in exps:
                if hasattr(e, 'tokens') and e.tokens is not None:
                    t = e.tokens
                    total += t.numel() if isinstance(t, torch.Tensor) else len(t)
            return float(total)

        n_normal = _count_tokens(normal_exps)
        n_bridge = _count_tokens(bridge_exps)
        ratio = n_bridge / (n_normal + n_bridge) if (n_normal + n_bridge) > 0 else 0.0

        if normal_exps:
            last = normal_exps[-1]
            if last.metrics is None:
                last.metrics = {}
            last.metrics["bridge_token_ratio"] = ratio
            last.metrics["bridge_token_count"] = n_bridge
            last.metrics["normal_token_count"] = n_normal

        return normal_exps + bridge_exps


class _FutureBridgeB2FAlfworldBase(_TeacherReliableAnchorAlfworldBase):
    """
    FutureBridge: B2F anchor + teacher executability gate

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)
      # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)
      # (implementation detail)
    """

    async def run_async(self):
        import re as _re

        if self.is_eval:
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env)
            finally:
                env.close()

        current_step = 0
        if hasattr(self.task, 'batch_id'):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                m = _re.match(r'^(\d+)', batch_id)
                if m:
                    current_step = int(m.group(1))

        self.set_training_progress(current_step, self.total_steps)
        predefined_actions = self.raw_task.get("actions", None)

        if not predefined_actions:
            self._current_anchor_reliable = False
            self._current_anchor_type = "no_traj"
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                task_desc = _extract_task(obs)
                return await self._run_episode_from_checkpoint(
                    env, obs, info, [], task_desc, 0
                )
            finally:
                env.close()

        # (implementation detail)
        k_star = self._linear_checkpoint_step(predefined_actions)
        M_u = max(1, len(predefined_actions) - k_star)
        M_bucket = _bucketize_M(M_u)

        # (implementation detail)
        cache_key = (self.task_desc, k_star, M_bucket)
        if cache_key in self._flex_cache:
            reliable = self._flex_cache[cache_key]
        else:
            reliable = await self._teacher_executable_within_M(
                game_file=self.task_desc,
                expert_actions=predefined_actions,
                k=k_star,
                M=M_bucket,
            )
            self._flex_cache[cache_key] = reliable

        self._current_anchor_reliable = reliable
        self._current_anchor_type = "b2f_verified" if reliable else "b2f_unverified"

        logger.debug(
            f"[FutureBridgeB2F] k={k_star} M_u={M_u} reliable={reliable}"
        )

        # (implementation detail)
        if k_star > 0:
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                self._current_anchor_reliable = False
                env = _create_alfworld_env(self.task_desc)
                try:
                    obs, info = env.reset()
                    task_desc = _extract_task(obs)
                    return await self._run_episode_from_checkpoint(
                        env, obs, info, [], task_desc, 0
                    )
                finally:
                    env.close()

            env, obs, info, history, task_desc, _, done = result
            if done:
                env.close()
                return []
            try:
                return await self._run_episode_from_checkpoint(
                    env, obs, info, history, task_desc, k_star
                )
            finally:
                env.close()
        else:
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                task_desc = _extract_task(obs)
                return await self._run_episode_from_checkpoint(
                    env, obs, info, [], task_desc, 0
                )
            finally:
                env.close()

class _FutureBridgeGateAlfworldBase(_FutureBridgeB2FAlfworldBase):
    """
    # (implementation detail)

    # (implementation detail)
    # (implementation detail)
    # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      advantage = mean(teacher_logp - student_logp) on bridge tokens
        # (implementation detail)
        # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)
      # (implementation detail)
                 # (implementation detail)
      # (implementation detail)

    # (implementation detail)
               # (implementation detail)
               # (implementation detail)
    """

    async def run_async(self):
        # (implementation detail)
        import re as _re

        if self.is_eval:
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env)
            finally:
                env.close()

        current_step = 0
        if hasattr(self.task, 'batch_id'):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                m = _re.match(r'^(\d+)', batch_id)
                if m:
                    current_step = int(m.group(1))

        self.set_training_progress(current_step, self.total_steps)
        predefined_actions = self.raw_task.get("actions", None)

        if not predefined_actions:
            self._current_anchor_reliable = False
            self._current_anchor_type = "no_traj"
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                task_desc = _extract_task(obs)
                return await self._run_episode_from_checkpoint(
                    env, obs, info, [], task_desc, 0
                )
            finally:
                env.close()

        k_star = self._linear_checkpoint_step(predefined_actions)

        # (implementation detail)
        self._current_anchor_reliable = True
        self._current_anchor_type = "b2f_v3"

        logger.debug(f"[FutureBridgeGate] k={k_star} gate=post_hoc_advantage")

        if k_star > 0:
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                self._current_anchor_reliable = False
                env = _create_alfworld_env(self.task_desc)
                try:
                    obs, info = env.reset()
                    task_desc = _extract_task(obs)
                    return await self._run_episode_from_checkpoint(
                        env, obs, info, [], task_desc, 0
                    )
                finally:
                    env.close()

            env, obs, info, history, task_desc, _, done = result
            if done:
                env.close()
                return []
            try:
                return await self._run_episode_from_checkpoint(
                    env, obs, info, history, task_desc, k_star
                )
            finally:
                env.close()
        else:
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                task_desc = _extract_task(obs)
                return await self._run_episode_from_checkpoint(
                    env, obs, info, [], task_desc, 0
                )
            finally:
                env.close()

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        """
        # (implementation detail)

        # (implementation detail)
          # (implementation detail)
            # (implementation detail)
            # (implementation detail)
          # (implementation detail)

        # (implementation detail)
          # (implementation detail)
          # (implementation detail)
          # (implementation detail)

        # (implementation detail)
        """
        import torch, re as _re

        # (implementation detail)
        turn_adv = []
        for resp in turn_responses:
            if resp.teacher_logprobs is not None and resp.logprobs is not None:
                tl = resp.teacher_logprobs if isinstance(resp.teacher_logprobs, torch.Tensor) \
                     else torch.tensor(resp.teacher_logprobs)
                sl = resp.logprobs if isinstance(resp.logprobs, torch.Tensor) \
                     else torch.tensor(resp.logprobs)
                turn_adv.append((tl.float() - sl.float()).mean().item())
            else:
                turn_adv.append(0.0)

        n_turns = len(turn_adv)
        episode_mean_adv = sum(turn_adv) / n_turns if n_turns > 0 else 0.0

        # (implementation detail)
        def _has_valid_action(resp):
            text = resp.response_text or ""
            return bool(_re.search(r"<action>.*?</action>", text, _re.DOTALL))

        turn_kls = []
        for resp in turn_responses:
            if not _has_valid_action(resp):
                turn_kls.append(0.0)
            elif resp.teacher_logprobs is not None and resp.logprobs is not None:
                kl = ((resp.logprobs - resp.teacher_logprobs).sum() / max(1, len(resp.logprobs))).item()
                turn_kls.append(kl)
            else:
                turn_kls.append(0.0)

        k = max(1, int(n_turns * self.bridge_kl_top_ratio))
        sorted_kl_vals = sorted(turn_kls, reverse=True)
        kl_threshold = sorted_kl_vals[k - 1] if sorted_kl_vals else 0.0
        bridge_turn_map = {}   # bridge_idx → turn_idx
        triggered = 0
        for turn_idx, kl_val in sorted(enumerate(turn_kls), key=lambda x: -x[1]):
            if triggered >= self.bridge_kl_max_per_ep:
                break
            if kl_val < kl_threshold:
                break
            bridge_turn_map[triggered] = turn_idx
            triggered += 1

        # (implementation detail)
        candidates = await super(
            _FutureBridgeB2FAlfworldBase, self
        )._try_kl_bridges(turn_responses, turn_memories)

        if not candidates:
            return []

        # 4. Future-based gate
        kept = []
        for exp in candidates:
            # (implementation detail)
            turn_idx = bridge_turn_map.get(b_idx)

            if turn_idx is None:
                kept.append(exp)
                continue

            # (implementation detail)
            future_advs = turn_adv[turn_idx + 1:]
            if not future_advs:
                # (implementation detail)
                kept.append(exp)
                continue

            future_mean_adv = sum(future_advs) / len(future_advs)

            # (implementation detail)
            if future_mean_adv < episode_mean_adv:
                kept.append(exp)
                logger.debug(
                    f"[FutureBridgeGate] KEPT turn={turn_idx}: "
                    f"future={future_mean_adv:.3f} < episode={episode_mean_adv:.3f}"
                )
            else:
                logger.debug(
                    f"[FutureBridgeGate] DROPPED turn={turn_idx}: "
                    f"future={future_mean_adv:.3f} >= episode={episode_mean_adv:.3f}"
                )

        return kept


class FutureBridgeNoFutureValidationAlfworldWorkflow(_FutureBridgeB2FAlfworldBase):
    """
    # (implementation detail)

    # (implementation detail)
    # (implementation detail)
    # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      bridge_adv = mean(teacher_logprobs - logprobs)
        # (implementation detail)
        # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)
      # (implementation detail)
      # (implementation detail)

    # (implementation detail)
    """

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        import torch

        # (implementation detail)
        candidates = await super()._try_kl_bridges(turn_responses, turn_memories)
        if not candidates:
            return []

        # Gate: bridge token advantage > 0
        kept = []
        for exp in candidates:
            if exp.teacher_logprobs is None or exp.logprobs is None:
                kept.append(exp)
                continue

            tl = exp.teacher_logprobs if isinstance(exp.teacher_logprobs, torch.Tensor) \
                 else torch.tensor(exp.teacher_logprobs)
            sl = exp.logprobs if isinstance(exp.logprobs, torch.Tensor) \
                 else torch.tensor(exp.logprobs)

            bridge_adv = (tl.float() - sl.float()).mean().item()

            if bridge_adv > 0:
                kept.append(exp)
                logger.debug(f"[FutureBridgeNoFutureVal] KEPT bridge_adv={bridge_adv:.3f}")
            else:
                logger.debug(f"[FutureBridgeNoFutureVal] DROPPED bridge_adv={bridge_adv:.3f} (student converged)")

        return kept


class FutureBridgeNoBridgeExecutionAlfworldWorkflow(_FutureBridgeGateAlfworldBase):
    """
    # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)
      # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)
    """

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        import torch, re as _re

        # (implementation detail)
        turn_pos_ratio = []
        for resp in turn_responses:
            if resp.teacher_logprobs is not None and resp.logprobs is not None:
                tl = resp.teacher_logprobs if isinstance(resp.teacher_logprobs, torch.Tensor) \
                     else torch.tensor(resp.teacher_logprobs)
                sl = resp.logprobs if isinstance(resp.logprobs, torch.Tensor) \
                     else torch.tensor(resp.logprobs)
                adv = tl.float() - sl.float()
                pos = (adv > 0).sum().item()
                tot = len(adv)
                turn_pos_ratio.append(pos / tot if tot > 0 else 0.0)
            else:
                turn_pos_ratio.append(0.0)

        n_turns = len(turn_pos_ratio)
        episode_pos_ratio = sum(turn_pos_ratio) / n_turns if n_turns > 0 else 0.1

        # (implementation detail)
        def _has_valid_action(resp):
            text = resp.response_text or ""
            return bool(_re.search(r"<action>.*?</action>", text, _re.DOTALL))

        turn_kls = []
        for resp in turn_responses:
            if not _has_valid_action(resp):
                turn_kls.append(0.0)
            elif resp.teacher_logprobs is not None and resp.logprobs is not None:
                kl = ((resp.logprobs - resp.teacher_logprobs).sum() / max(1, len(resp.logprobs))).item()
                turn_kls.append(kl)
            else:
                turn_kls.append(0.0)

        k = max(1, int(n_turns * self.bridge_kl_top_ratio))
        sorted_kl_vals = sorted(turn_kls, reverse=True)
        kl_threshold = sorted_kl_vals[k - 1] if sorted_kl_vals else 0.0
        bridge_turn_map = {}
        triggered = 0
        for turn_idx, kl_val in sorted(enumerate(turn_kls), key=lambda x: -x[1]):
            if triggered >= self.bridge_kl_max_per_ep:
                break
            if kl_val < kl_threshold:
                break
            bridge_turn_map[triggered] = turn_idx
            triggered += 1

        # (implementation detail)
        candidates = await super(
            _FutureBridgeB2FAlfworldBase, self
        )._try_kl_bridges(turn_responses, turn_memories)

        if not candidates:
            return []

        # 4. Future pos_ratio gate
        kept = []
        for exp in candidates:
            b_idx = exp.eid.step - 5000
            turn_idx = bridge_turn_map.get(b_idx)

            if turn_idx is None:
                kept.append(exp)
                continue

            future_ratios = turn_pos_ratio[turn_idx + 1:]
            if not future_ratios:
                kept.append(exp)
                continue

            future_pos_ratio = sum(future_ratios) / len(future_ratios)

            # (implementation detail)
            if future_pos_ratio < episode_pos_ratio:
                kept.append(exp)
                logger.debug(
                    f"[FutureBridgeNoBridgeExec] KEPT turn={turn_idx}: "
                    f"future_pos={future_pos_ratio:.3f} < episode={episode_pos_ratio:.3f}"
                )
            else:
                logger.debug(
                    f"[FutureBridgeNoBridgeExec] DROPPED turn={turn_idx}: "
                    f"future_pos={future_pos_ratio:.3f} >= episode={episode_pos_ratio:.3f}"
                )

        return kept


class FutureBridgeAlfworldWorkflow(_FutureBridgeB2FAlfworldBase):
    """
    # (implementation detail)

    # (implementation detail)
      # (implementation detail)
      # (implementation detail)
      # (implementation detail)
      # (implementation detail)
         # (implementation detail)
      # (implementation detail)

    # (implementation detail)
    # (implementation detail)
    """

    _CONTINUATION_STEPS = 3

    async def run_async(self):
        import re as _re

        if self.is_eval:
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env)
            finally:
                env.close()

        current_step = 0
        if hasattr(self.task, 'batch_id'):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                m = _re.match(r'^(\d+)', batch_id)
                if m:
                    current_step = int(m.group(1))

        self.set_training_progress(current_step, self.total_steps)
        predefined_actions = self.raw_task.get("actions", None)

        if not predefined_actions:
            self._current_anchor_reliable = False
            self._current_anchor_type = "no_traj"
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                task_desc = _extract_task(obs)
                return await self._run_episode_from_checkpoint(env, obs, info, [], task_desc, 0)
            finally:
                env.close()

        k_star = self._linear_checkpoint_step(predefined_actions)
        self._current_anchor_reliable = True
        self._current_anchor_type = "b2f_v4"
        # (implementation detail)

        logger.debug(f"[FutureBridge] k={k_star} gate=student_continuation")

        if k_star > 0:
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                self._current_anchor_reliable = False
                env = _create_alfworld_env(self.task_desc)
                try:
                    obs, info = env.reset()
                    task_desc = _extract_task(obs)
                    return await self._run_episode_from_checkpoint(env, obs, info, [], task_desc, 0)
                finally:
                    env.close()
            env, obs, info, history, task_desc, _, done = result
            if done:
                env.close()
                return []
            try:
                return await self._run_episode_from_checkpoint(env, obs, info, history, task_desc, k_star)
            finally:
                env.close()
        else:
            env = _create_alfworld_env(self.task_desc)
            try:
                obs, info = env.reset()
                task_desc = _extract_task(obs)
                return await self._run_episode_from_checkpoint(env, obs, info, [], task_desc, 0)
            finally:
                env.close()

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        import torch, re as _re

        def _has_valid_action(resp):
            text = resp.response_text or ""
            return bool(_re.search(r"<action>.*?</action>", text, _re.DOTALL))

        # (implementation detail)
        turn_pos_counts = []
        for resp in turn_responses:
            if resp.teacher_logprobs is not None and resp.logprobs is not None:
                tl = resp.teacher_logprobs if isinstance(resp.teacher_logprobs, torch.Tensor) \
                     else torch.tensor(resp.teacher_logprobs)
                sl = resp.logprobs if isinstance(resp.logprobs, torch.Tensor) \
                     else torch.tensor(resp.logprobs)
                adv = tl.float() - sl.float()
                turn_pos_counts.append(((adv > 0).sum().item(), len(adv)))
            else:
                turn_pos_counts.append((0, 0))

        # (implementation detail)
        turn_kls = []
        for resp in turn_responses:
            if not _has_valid_action(resp):
                turn_kls.append(0.0)
            elif resp.teacher_logprobs is not None and resp.logprobs is not None:
                kl = ((resp.logprobs - resp.teacher_logprobs).sum() / max(1, len(resp.logprobs))).item()
                turn_kls.append(kl)
            else:
                turn_kls.append(0.0)

        n = len(turn_kls)
        if n == 0:
            return []

        k = max(1, int(n * self.bridge_kl_top_ratio))
        sorted_kl_vals = sorted(turn_kls, reverse=True)
        kl_threshold = sorted_kl_vals[k - 1]
        sorted_turns = sorted(enumerate(turn_kls), key=lambda x: -x[1])
        kl_max, kl_min = sorted_kl_vals[0], sorted_kl_vals[-1]
        kl_range = max(kl_max - kl_min, 1e-6)

        predefined_actions = self.raw_task.get("actions", None)
        k_star = getattr(self, '_k_star_v4', 0)

        bridge_exps = []
        triggered = 0

        for turn_idx, kl_val in sorted_turns:
            if triggered >= self.bridge_kl_max_per_ep:
                break
            if kl_val < kl_threshold:
                break

            t_bridge = turn_responses[turn_idx].eid.step
            memory_at_turn = turn_memories[turn_idx]
            weight = (kl_val - kl_min) / kl_range

            # (implementation detail)
            cands = await self._generate_kl_bridge(
                memory_at_turn=memory_at_turn,
                trigger_kl=kl_val,
                bridge_idx=triggered,
                bridge_weight=weight,
            )
            if not cands:
                continue

            bridge_exp = cands[0]
            bridge_action = parse_action(bridge_exp.response_text or "")

            if not bridge_action or not predefined_actions:
                # (implementation detail)
                bridge_exps.extend(cands)
                triggered += 1
                continue

            # (implementation detail)
            bridge_memory = memory_at_turn + [
                {"role": "assistant", "content": bridge_exp.response_text or ""}
            ]
            pos_ratio_after = await self._student_continuation_ratio(
                predefined_actions=predefined_actions,
                k_star=k_star,
                turn_responses=turn_responses,
                t_bridge=t_bridge,
                bridge_action=bridge_action,
                bridge_memory=bridge_memory,
            )

            if pos_ratio_after is None:
                # (implementation detail)
                bridge_exps.extend(cands)
                triggered += 1
                continue

            # (implementation detail)
            base_counts = turn_pos_counts[turn_idx + 1 : turn_idx + 1 + self._CONTINUATION_STEPS]
            if not base_counts:
                continue
            base_pos = sum(p for p, _ in base_counts)
            base_tot = sum(t for _, t in base_counts)
            if base_tot == 0:
                continue
            base_ratio = base_pos / base_tot

            if pos_ratio_after > base_ratio:
                bridge_exps.extend(cands)
                triggered += 1
                logger.debug(
                    f"[FutureBridge] KEPT t_bridge={t_bridge}: "
                    f"after={pos_ratio_after:.3f} > base={base_ratio:.3f}"
                )
            else:
                logger.debug(
                    f"[FutureBridge] DROPPED t_bridge={t_bridge}: "
                    f"after={pos_ratio_after:.3f} <= base={base_ratio:.3f}"
                )

        return bridge_exps

    async def _student_continuation_ratio(
        self,
        predefined_actions: list,
        k_star: int,
        turn_responses: list,
        t_bridge: int,
        bridge_action: str,
        bridge_memory: list,
    ) -> Optional[float]:
        """
        # (implementation detail)
        # (implementation detail)
        # (implementation detail)
        """
        import torch
        env = None
        try:
            # (implementation detail)
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                return None
            env, obs, info, _, _, _, done = result
            if done:
                return None

            # (implementation detail)
            for resp in turn_responses:
                if resp.eid.step >= t_bridge:
                    break
                action = parse_action(resp.response_text or "")
                obs, _, done, info = env.step(action)
                if done:
                    return None

            # (implementation detail)
            obs, _, done, info = env.step(bridge_action)
            if done:
                # (implementation detail)
                return 1.0

            # (implementation detail)
            memory = list(bridge_memory)
            kwargs = {"n": 1, "temperature": self.temperature, "logprobs": 0}
            pos_count, total_count = 0, 0

            for _ in range(self._CONTINUATION_STEPS):
                admissible = info.get("admissible_commands", [])
                if admissible and isinstance(admissible[0], list):
                    admissible = admissible[0]
                admissible = [s for s in admissible if s != "help"]
                reformatted = "\n ".join(f"'{s}'" for s in admissible)

                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_observation(obs),
                    admissible_actions=reformatted,
                )
                messages = memory + [{"role": "user", "content": user_content}]

                # (implementation detail)
                resps = await self.model.chat_async(messages, **kwargs)
                student_resp = resps[0]
                student_text = student_resp.response_text or ""

                # (implementation detail)
                full_tokens = student_resp.tokens.tolist()
                resp_start = student_resp.prompt_length - 1

                sl_full = await self.model.logprobs_async(
                    tokens=full_tokens, temperature=self.temperature
                )
                tl_full = await self.teacher_model.logprobs_async(
                    tokens=full_tokens, temperature=self.temperature
                )
                sl = sl_full[resp_start:]
                tl = tl_full[resp_start:]

                adv = tl.float() - sl.float()
                pos_count += (adv > 0).sum().item()
                total_count += len(adv)

                memory.append({"role": "user", "content": user_content})
                memory.append({"role": "assistant", "content": student_text})

                action = parse_action(student_text)
                obs, _, done, info = env.step(action)
                if done:
                    break

            return pos_count / total_count if total_count > 0 else 0.5

        except Exception as e:
            logger.debug(f"[FutureBridge] env check failed: {e}")
            return None
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass