"""FutureBridge-OPD workflows and component ablations for ALFWorld.

The main workflow extends the B2F curriculum with maximum-disagreement
localization, a single Teacher bridge, paired frozen-Student continuations,
and the strict future-validation gate described in the paper.
"""

import logging
import math
from typing import List, Optional, Tuple

from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task
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


def _segment_pos_ratio(responses) -> Optional[float]:
    """Token-pooled teacher-preference ratio for an observed segment."""
    import torch

    positive = 0
    total = 0
    for response in responses:
        if response.teacher_logprobs is None or response.logprobs is None:
            return None
        teacher_lp = (
            response.teacher_logprobs
            if isinstance(response.teacher_logprobs, torch.Tensor)
            else torch.tensor(response.teacher_logprobs)
        )
        student_lp = (
            response.logprobs
            if isinstance(response.logprobs, torch.Tensor)
            else torch.tensor(response.logprobs)
        )
        advantage = teacher_lp.float() - student_lp.float()
        positive += int((advantage > 0).sum())
        total += len(advantage)
    return positive / total if total > 0 else None


def _bucketize_M(M: int) -> int:
    for b in _BUDGET_BUCKETS:
        if M <= b:
            return b
    return _BUDGET_BUCKETS[-1]


class _TeacherReliableAnchorAlfworldBase(Bridge_TCOD_kl_alfworld_workflow):
    """
    FutureBridge-OPD: Teacher-reliable anchor selection + shared bridge + binary reliability gate.

    This support class chooses an anchor through a Teacher executability check
    and enables bridging only for reliable anchor types. The bridge generation
    and scoring logic is inherited from the shared bridge implementation.
    """

    _RELIABLE_TYPES = {"hard", "b2f"}

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self._anchor_cache: dict = {}
        self._current_anchor_reliable: bool = True
        self._current_anchor_type: str = "b2f"
        wargs = task.workflow_args or {}
        self.bridge_failed_episodes_only = bool(
            wargs.get("bridge_failed_episodes_only", False)
        )
        self.bridge_require_full_continuation = bool(
            wargs.get("bridge_require_full_continuation", True)
        )


    async def _teacher_executable_within_M(
        self,
        game_file: str,
        expert_actions: List[str],
        k: int,
        M: int,
    ) -> bool:
        """Test whether the Teacher can finish within M turns from s_k."""
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

    async def _select_teacher_reliable_anchor(
        self,
        expert_actions: List[str],
        M_u: int,
    ) -> Tuple[int, str]:
        """Select the hardest Teacher-executable anchor from a fixed set."""
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

            if cache_key in self._anchor_cache:
                if self._anchor_cache[cache_key]:
                    return k, anchor_type
                continue

            success = await self._teacher_executable_within_M(
                game_file=self.task_desc,
                expert_actions=expert_actions,
                k=k,
                M=M_bucket,
            )
            self._anchor_cache[cache_key] = success

            if success:
                return k, anchor_type

        return k_b2f, "fallback_b2f"


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

        k_b2f_ref = self._linear_checkpoint_step(predefined_actions)
        M_u = max(1, len(predefined_actions) - k_b2f_ref) if k_b2f_ref is not None else 1
        k_star, anchor_type = await self._select_teacher_reliable_anchor(predefined_actions, M_u)

        self._current_anchor_type = anchor_type
        self._current_anchor_reliable = anchor_type in self._RELIABLE_TYPES

        logger.debug(
            f"[FutureBridge] k_b2f={k_b2f_ref} M_u={M_u} k*={k_star} "
            f"type={anchor_type} reliable={self._current_anchor_reliable}"
        )

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
        Reliability gate: only bridge if anchor is teacher-reliable (hard or b2f).
        easy / fallback → return [] (no bridge, degrade to B2F-style OPD).
        """
        if not self._current_anchor_reliable:
            logger.debug(
                f"[FutureBridge] bridge suppressed: anchor_type={self._current_anchor_type}"
            )
            return []

        return await super()._try_kl_bridges(turn_responses, turn_memories)


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
                "anchor_type":     _type_code.get(self._current_anchor_type, -1.0),
                "anchor_is_reliable": float(self._current_anchor_reliable),
                "anchor_is_hard": float(self._current_anchor_type == "hard"),
                "anchor_is_b2f": float(self._current_anchor_type == "b2f"),
                "anchor_is_easy": float(self._current_anchor_type == "easy"),
                "anchor_is_fallback": float(
                    self._current_anchor_type == "fallback_b2f"
                ),
                "anchor_cache_size": float(len(self._anchor_cache)),
            })
        return normal_exps, turn_memories


    async def _run_episode_from_checkpoint(
        self, env, observation, info, history, task_description, start_step
    ):
        """
        Override to inject bridge_token_ratio metric.
        bridge_token_ratio = bridge_tokens / (normal_tokens + bridge_tokens)
        This supports reporting the fraction of bridge supervision.
        """
        normal_exps, turn_memories = await self._b2f_episode_with_memories(
            env, observation, info, history, task_description, start_step
        )

        if self.is_eval or not normal_exps:
            return normal_exps

        episode_succeeded = self._env_done and self._final_reward > 0.5
        if self.bridge_failed_episodes_only and episode_succeeded:
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
    Shared B2F bridge implementation used by main and the bridge-token-gate ablation.
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

        k_star = self._linear_checkpoint_step(predefined_actions)
        M_u = max(1, len(predefined_actions) - k_star)
        M_bucket = _bucketize_M(M_u)

        cache_key = (self.task_desc, k_star, M_bucket)
        if cache_key in self._anchor_cache:
            reliable = self._anchor_cache[cache_key]
        else:
            reliable = await self._teacher_executable_within_M(
                game_file=self.task_desc,
                expert_actions=predefined_actions,
                k=k_star,
                M=M_bucket,
            )
            self._anchor_cache[cache_key] = reliable

        self._current_anchor_reliable = reliable
        self._current_anchor_type = "b2f_verified" if reliable else "b2f_unverified"

        logger.debug(
            f"[FutureBridge] k={k_star} M_u={M_u} reliable={reliable}"
        )

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


class BridgeTokenGateAblationAlfworldWorkflow(_FutureBridgeB2FAlfworldBase):
    """
    bridge-token-gate ablation: retain a bridge when its mean token advantage is positive.
    """

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        import torch

        candidates = await super()._try_kl_bridges(turn_responses, turn_memories)
        if not candidates:
            return []

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
                logger.debug(f"[BridgeTokenGate] KEPT bridge_adv={bridge_adv:.3f}")
            else:
                logger.debug(f"[BridgeTokenGate] DROPPED bridge_adv={bridge_adv:.3f} (student converged)")

        return kept


class FutureBridgeAlfworldWorkflow(_FutureBridgeB2FAlfworldBase):
    """
    Main main method: compare paired frozen-Student continuations from the
    original action and Teacher bridge at the same restored state.
    """

    _CONTINUATION_STEPS = 3

    def __init__(self, *, task: Task, model: ModelWrapper, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.bridge_position_top_k = max(
            1, int(task.workflow_args.get("bridge_position_top_k", 1))
        )

    def _select_bridge_candidates(self, turn_responses):
        """Return (turn_index, token-average disagreement) candidates."""
        import re as _re

        scored_turns = []
        for turn_idx, resp in enumerate(turn_responses[:-1]):
            text = resp.response_text or ""
            if not _re.search(r"<action>.*?</action>", text, _re.DOTALL):
                continue
            if resp.teacher_logprobs is None or resp.logprobs is None:
                continue
            disagreement = (
                resp.logprobs - resp.teacher_logprobs
            ).float().mean().item()
            scored_turns.append((turn_idx, disagreement))
        scored_turns.sort(key=lambda item: item[1], reverse=True)
        return scored_turns[:self.bridge_position_top_k]

    async def run_async(self):
        import re as _re

        if self.is_eval:
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env)
            finally:
                env.close()

        self._CONTINUATION_STEPS = int(self.task.workflow_args.get("continuation_steps", 3))

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
        self._current_anchor_type = "b2f"
        self._bridge_checkpoint_step = k_star

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
        if len(turn_responses) < 2:
            return []

        candidate_turns = self._select_bridge_candidates(turn_responses)
        if not candidate_turns:
            return []

        predefined_actions = self.raw_task.get("actions", None)
        if not predefined_actions:
            return []
        k_star = getattr(self, '_bridge_checkpoint_step', 0)

        bridge_exps = []
        for bridge_idx, (turn_idx, disagreement) in enumerate(candidate_turns):
            if bridge_idx >= self.bridge_max_per_ep:
                break

            t_bridge = turn_responses[turn_idx].eid.step
            memory_at_turn = turn_memories[turn_idx]

            cands = await self._generate_kl_bridge(
                memory_at_turn=memory_at_turn,
                trigger_kl=disagreement,
                bridge_idx=bridge_idx,
                bridge_weight=1.0,
            )
            if not cands:
                return []

            bridge_exp = cands[0]
            bridge_action = parse_action(bridge_exp.response_text or "")
            base_exp = turn_responses[turn_idx]
            base_action = parse_action(base_exp.response_text or "")
            if not bridge_action or not base_action:
                return []

            base_memory = memory_at_turn + [
                {"role": "assistant", "content": base_exp.response_text or ""}
            ]
            bridge_memory = memory_at_turn + [
                {"role": "assistant", "content": bridge_exp.response_text or ""}
            ]

            base_ratio = await self._student_continuation_ratio(
                predefined_actions=predefined_actions,
                k_star=k_star,
                turn_responses=turn_responses,
                t_bridge=t_bridge,
                branch_action=base_action,
                branch_memory=base_memory,
            )
            if base_ratio is None:
                return []

            bridge_ratio = await self._student_continuation_ratio(
                predefined_actions=predefined_actions,
                k_star=k_star,
                turn_responses=turn_responses,
                t_bridge=t_bridge,
                branch_action=bridge_action,
                branch_memory=bridge_memory,
            )
            if bridge_ratio is None:
                return []

            if bridge_ratio > base_ratio:
                bridge_exps.extend(cands)
                logger.debug(
                    f"[FutureBridge] KEPT t_bridge={t_bridge}: "
                    f"bridge={bridge_ratio:.3f} > base={base_ratio:.3f}"
                )
            else:
                logger.debug(
                    f"[FutureBridge] DROPPED t_bridge={t_bridge}: "
                    f"bridge={bridge_ratio:.3f} <= base={base_ratio:.3f}"
                )

        return bridge_exps

    async def _student_continuation_ratio(
        self,
        predefined_actions: list,
        k_star: int,
        turn_responses: list,
        t_bridge: int,
        branch_action: str,
        branch_memory: list,
    ) -> Optional[float]:
        """
        Restore the selected student state, execute one fixed branch action,
        and roll out the frozen student for H turns. Returns the token-level
        teacher-preference ratio, or None when the branch cannot be validated.
        """
        import torch
        env = None
        try:
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                return None
            env, obs, info, history, task_desc, actual_step, done = result
            if done:
                return None

            for resp in turn_responses:
                if resp.eid.step >= t_bridge:
                    break
                action = parse_action(resp.response_text or "")
                if not action:
                    return None
                admissible = info.get("admissible_commands", [])
                if admissible and isinstance(admissible[0], list):
                    admissible = admissible[0]
                if action not in admissible:
                    return None
                formatted_obs = format_observation(obs)
                history.append(_format_history(formatted_obs, actual_step + 1, action))
                obs, _, done, info = env.step(action)
                actual_step += 1
                if done:
                    return None

            admissible = info.get("admissible_commands", [])
            if admissible and isinstance(admissible[0], list):
                admissible = admissible[0]
            if branch_action not in admissible:
                return None
            formatted_obs = format_observation(obs)
            history.append(
                _format_history(formatted_obs, actual_step + 1, branch_action)
            )
            obs, _, done, info = env.step(branch_action)
            actual_step += 1
            if done:
                return (
                    None
                    if self.bridge_require_full_continuation
                    else 1.0
                )

            memory = list(branch_memory)
            kwargs = {"n": 1, "temperature": self.temperature, "logprobs": 0}
            pos_count, total_count = 0, 0

            for continuation_idx in range(self._CONTINUATION_STEPS):
                admissible = info.get("admissible_commands", [])
                if admissible and isinstance(admissible[0], list):
                    admissible = admissible[0]
                admissible = [s for s in admissible if s != "help"]
                reformatted = "\n ".join(f"'{s}'" for s in admissible)

                formatted_obs = format_observation(obs)
                if len(history) < HISTORY_LENGTH:
                    user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=formatted_obs,
                        admissible_actions=reformatted,
                    )
                else:
                    user_content = ALFWORLD_TEMPLATE.format(
                        task_description=task_desc,
                        step_count=actual_step,
                        history_length=min(HISTORY_LENGTH, len(history)),
                        action_history="\n".join(history[-HISTORY_LENGTH:]),
                        current_step=actual_step + 1,
                        current_observation=formatted_obs,
                        admissible_actions=reformatted,
                    )
                messages = memory + [{"role": "user", "content": user_content}]

                resps = await self.model.chat_async(messages, **kwargs)
                student_resp = resps[0]
                student_text = student_resp.response_text or ""

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
                if not action or action not in admissible:
                    return None
                history.append(
                    _format_history(formatted_obs, actual_step + 1, action)
                )
                obs, _, done, info = env.step(action)
                actual_step += 1
                if done:
                    if (
                        self.bridge_require_full_continuation
                        and continuation_idx + 1 < self._CONTINUATION_STEPS
                    ):
                        return None
                    break

            return pos_count / total_count if total_count > 0 else None

        except Exception as e:
            logger.debug(f"[FutureBridge] env check failed: {e}")
            return None
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


@WORKFLOWS.register_module("FutureBridgeNoFutureValidationAlfworldWorkflow")
class FutureBridgeNoFutureValidationAlfworldWorkflow(FutureBridgeAlfworldWorkflow):
    """
    Ablation without future-validation filtering.

    The bridge and continuation are still executed, but a valid continuation
    is retained without comparing its teacher-preferred-token ratio against
    the original trajectory.
    """

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        candidate_turns = self._select_bridge_candidates(turn_responses)
        if not candidate_turns:
            return []

        predefined_actions = self.raw_task.get("actions", None)
        if not predefined_actions:
            return []
        k_star = getattr(self, '_bridge_checkpoint_step', 0)

        bridge_exps = []
        for bridge_idx, (turn_idx, disagreement) in enumerate(candidate_turns):
            if bridge_idx >= self.bridge_max_per_ep:
                break

            t_bridge = turn_responses[turn_idx].eid.step
            memory_at_turn = turn_memories[turn_idx]

            cands = await self._generate_kl_bridge(
                memory_at_turn=memory_at_turn,
                trigger_kl=disagreement,
                bridge_idx=bridge_idx,
                bridge_weight=1.0,
            )
            if not cands:
                return []

            bridge_exp = cands[0]
            bridge_action = parse_action(bridge_exp.response_text or "")
            if not bridge_action:
                return []

            bridge_memory = memory_at_turn + [
                {"role": "assistant", "content": bridge_exp.response_text or ""}
            ]
            pos_ratio_after = await self._student_continuation_ratio(
                predefined_actions=predefined_actions,
                k_star=k_star,
                turn_responses=turn_responses,
                t_bridge=t_bridge,
                branch_action=bridge_action,
                branch_memory=bridge_memory,
            )

            if pos_ratio_after is None:
                return []

            bridge_exps.extend(cands)
            logger.debug(
                f"[FutureBridge-no-future-validation] KEPT t_bridge={t_bridge}: "
                f"env_verified (pos_ratio_after={pos_ratio_after:.3f}, no gate)"
            )

        return bridge_exps


@WORKFLOWS.register_module("FutureBridgeNoBridgeExecutionAlfworldWorkflow")
class FutureBridgeNoBridgeExecutionAlfworldWorkflow(
    FutureBridgeAlfworldWorkflow
):
    """Select a teacher action without executing it in the environment."""

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        candidates = self._select_bridge_candidates(turn_responses)
        if not candidates:
            return []

        bridge_exps = []
        for bridge_idx, (turn_idx, disagreement) in enumerate(candidates):
            if bridge_idx >= self.bridge_max_per_ep:
                break
            cands = await self._generate_kl_bridge(
                memory_at_turn=turn_memories[turn_idx],
                trigger_kl=disagreement,
                bridge_idx=bridge_idx,
                bridge_weight=1.0,
            )
            if not cands or not parse_action(cands[0].response_text or ""):
                return []

            before = _segment_pos_ratio(turn_responses[:turn_idx + 1])
            after = _segment_pos_ratio(turn_responses[turn_idx + 1:])
            if before is None or after is None:
                return []
            if after < before:
                bridge_exps.extend(cands)

        return bridge_exps
