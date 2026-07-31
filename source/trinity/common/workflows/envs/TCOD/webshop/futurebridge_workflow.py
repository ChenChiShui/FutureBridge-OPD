"""
FutureBridge-OPD and its component ablations for WebShop.

The main workflow rebuilds the environment at the selected bridge state and
rolls out paired original-action and teacher-bridge continuations. It retains
the bridge only when the bridge branch has a higher teacher-preferred-token
ratio than the original-action branch.
"""

import logging
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.webshop.futurebridge_base import (
    _FutureBridgeWebShopBase,
    BRIDGE_STEP_OFFSET,
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


def _segment_pos_ratio(responses: List[Experience]) -> Optional[float]:
    """Token-pooled teacher-preference ratio for an observed segment."""
    positive = 0
    total = 0
    for response in responses:
        if response.teacher_logprobs is None or response.logprobs is None:
            return None
        advantage = (
            response.teacher_logprobs.float() - response.logprobs.float()
        )
        positive += int((advantage > 0).sum())
        total += len(advantage)
    return positive / total if total > 0 else None



@WORKFLOWS.register_module("FutureBridgeWebShopWorkflow")
class FutureBridgeWebShopWorkflow(_FutureBridgeWebShopBase):
    """
    WebShop FutureBridge-OPD: env-based student continuation gate (mirrors ALFWorld FutureBridge-OPD).

    After bridge generation:
      1. Rebuild env (replay reference B2F + Student pre-bridge actions)
      2. Execute the original and bridge actions in separate restored branches
      3. Roll out the same frozen student for H=3 turns in each branch
      4. Keep the bridge iff rho(bridge continuation) > rho(base continuation)
    """

    _CONTINUATION_STEPS = 3

    def __init__(self, *, task: Task, model: ModelWrapper,
                 auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self._CONTINUATION_STEPS = int(task.workflow_args.get("continuation_steps", 3))
        self.bridge_position_top_k = max(
            1, int(task.workflow_args.get("bridge_position_top_k", 1))
        )

    def _select_bridge_candidates(self, per_turn_disagreement):
        """Select non-final turns by token-average sampled disagreement."""
        indexed = list(enumerate(per_turn_disagreement[:-1]))
        indexed.sort(key=lambda item: item[1], reverse=True)
        return indexed[:self.bridge_position_top_k]

    async def _try_kl_bridge(
        self,
        start_step: int,
        memory_snapshots: List[List[dict]],
        per_turn_kl: List[float],
        turn_responses: List[Experience],
    ) -> List[Experience]:
        if (
            self.bridge_failed_episodes_only
            and self._final_reward >= self.bridge_reward_threshold
        ):
            return []
        if len(per_turn_kl) < 2:
            return []

        candidates = self._select_bridge_candidates(per_turn_kl)
        bridge_exps: List[Experience] = []

        for bridge_rank, (turn_idx, trigger_disagreement) in enumerate(candidates):
            if bridge_rank >= self.bridge_max_per_ep:
                break

            bridge_idx = start_step + turn_idx

            cands = await self._generate_kl_bridge(
                memory_at_turn=memory_snapshots[turn_idx],
                trigger_kl=trigger_disagreement,
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

            student_pre_bridge = [
                parse_action(r.response_text or "") for r in turn_responses[:turn_idx]
            ]

            base_ratio = await self._student_continuation_pos_ratio(
                session_id=self._episode_session_id,
                prefix_actions=self._reference_prefix_actions,
                student_pre_bridge=student_pre_bridge,
                branch_action=base_action,
                branch_memory=memory_snapshots[turn_idx] + [
                    {"role": "assistant", "content": base_exp.response_text or ""}
                ],
            )
            if base_ratio is None:
                return []

            bridge_ratio = await self._student_continuation_pos_ratio(
                session_id=self._episode_session_id,
                prefix_actions=self._reference_prefix_actions,
                student_pre_bridge=student_pre_bridge,
                branch_action=bridge_action,
                branch_memory=memory_snapshots[turn_idx] + [
                    {"role": "assistant", "content": bridge_exp.response_text or ""}
                ],
            )
            if bridge_ratio is None:
                return []

            if bridge_ratio > base_ratio:
                bridge_exps.extend(cands)
                logger.debug(
                    f"[FutureBridge] KEPT bridge turn={turn_idx}: "
                    f"bridge={bridge_ratio:.3f} > base={base_ratio:.3f}"
                )
            else:
                logger.debug(
                    f"[FutureBridge] DROPPED bridge turn={turn_idx}: "
                    f"bridge={bridge_ratio:.3f} <= base={base_ratio:.3f}"
                )

        return bridge_exps

    async def _student_continuation_pos_ratio(
        self,
        session_id: int,
        prefix_actions: List[str],
        student_pre_bridge: List[str],
        branch_action: str,
        branch_memory: List[dict],
    ) -> Optional[float]:
        """
        Rebuild env to bridge state, execute bridge, student continues N steps.
        Returns teacher-approval pos_ratio of student's continuation.
        """
        env = None
        try:
            env = _create_webshop_env()
            env.reset(session=session_id)
            obs = env.observation
            task_desc = _extract_task_description(obs)
            history: List[str] = []
            actual_step = 0

            for action in prefix_actions:
                if not action:
                    return None
                action_valid, _ = validate_action(
                    action, env.get_available_actions()
                )
                if not action_valid:
                    return None
                formatted_obs = format_observation(obs)
                history.append(
                    _format_history(formatted_obs, actual_step + 1, action)
                )
                obs, _, done, _ = env.step(action)
                actual_step += 1
                if done:
                    return None

            for action in student_pre_bridge:
                if action:
                    action_valid, _ = validate_action(
                        action, env.get_available_actions()
                    )
                    if not action_valid:
                        return None
                    formatted_obs = format_observation(obs)
                    history.append(
                        _format_history(formatted_obs, actual_step + 1, action)
                    )
                    obs, _, done, _ = env.step(action)
                    actual_step += 1
                    if done:
                        return None
                else:
                    return None

            action_valid, _ = validate_action(
                branch_action, env.get_available_actions()
            )
            if not action_valid:
                return None
            formatted_obs = format_observation(obs)
            history.append(
                _format_history(formatted_obs, actual_step + 1, branch_action)
            )
            obs, _, done, _ = env.step(branch_action)
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
                    user_content = WEBSHOP_TEMPLATE.format(
                        task_description=task_desc,
                        step_count=actual_step,
                        history_length=min(HISTORY_LENGTH, len(history)),
                        action_history="\n".join(history[-HISTORY_LENGTH:]),
                        current_step=actual_step + 1,
                        current_observation=formatted_obs,
                        available_actions=formatted_actions,
                    )
                msgs = memory + [{"role": "user", "content": user_content}]
                student_resps = await self.model.chat_async(msgs, **kwargs)
                student_resp = student_resps[0]
                student_text = student_resp.response_text or ""

                full_tokens = student_resp.tokens.tolist()
                rs = student_resp.prompt_length - 1
                sl_full = await self.model.logprobs_async(
                    tokens=full_tokens, temperature=self.temperature
                )
                tl_full = await self.teacher_model.logprobs_async(
                    tokens=full_tokens, temperature=self.temperature
                )
                sl = sl_full[rs:]
                tl = tl_full[rs:]
                adv = tl.float() - sl.float()
                pos_count += int((adv > 0).sum())
                total_count += len(adv)

                memory.append({"role": "user", "content": user_content})
                memory.append({"role": "assistant", "content": student_text})

                action = parse_action(student_text)
                action_valid, _ = validate_action(action, available_actions)
                if not action_valid:
                    return None
                history.append(
                    _format_history(formatted_obs, actual_step + 1, action)
                )
                obs, _, done, _ = env.step(action)
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
            logger.debug(f"[FutureBridge] env continuation failed: {e}")
            return None
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


@WORKFLOWS.register_module("FutureBridgeNoFutureValidationWebShopWorkflow")
class FutureBridgeNoFutureValidationWebShopWorkflow(FutureBridgeWebShopWorkflow):
    """
    Ablation without future-validation filtering.

    The bridge and continuation are still executed, but a valid continuation
    is retained without comparing its teacher-preferred-token ratio against
    the original trajectory.
    """

    async def _try_kl_bridge(
        self,
        start_step: int,
        memory_snapshots: List[List[dict]],
        per_turn_kl: List[float],
        turn_responses: List[Experience],
    ) -> List[Experience]:
        """Override: skip pos_ratio gate, keep all env-verified bridges."""
        if (
            self.bridge_failed_episodes_only
            and self._final_reward >= self.bridge_reward_threshold
        ):
            return []
        if len(per_turn_kl) < 2:
            return []

        candidates = self._select_bridge_candidates(per_turn_kl)
        bridge_exps: List[Experience] = []

        for bridge_rank, (turn_idx, trigger_disagreement) in enumerate(candidates):
            if bridge_rank >= self.bridge_max_per_ep:
                break

            bridge_idx = start_step + turn_idx

            cands = await self._generate_kl_bridge(
                memory_at_turn=memory_snapshots[turn_idx],
                trigger_kl=trigger_disagreement,
                bridge_idx=bridge_idx,
                bridge_weight=1.0,
            )
            if not cands:
                return []

            bridge_exp = cands[0]
            bridge_action = parse_action(bridge_exp.response_text or "")

            if not bridge_action:
                return []

            student_pre_bridge = [
                parse_action(r.response_text or "") for r in turn_responses[:turn_idx]
            ]

            pos_after = await self._student_continuation_pos_ratio(
                session_id=self._episode_session_id,
                prefix_actions=self._reference_prefix_actions,
                student_pre_bridge=student_pre_bridge,
                branch_action=bridge_action,
                branch_memory=memory_snapshots[turn_idx] + [
                    {"role": "assistant", "content": bridge_exp.response_text or ""}
                ],
            )

            if pos_after is not None:
                bridge_exps.extend(cands)
                logger.debug(
                    f"[FutureBridge-no-future-validation] KEPT bridge turn={turn_idx}: "
                    f"env_verified (pos_after={pos_after:.3f}, no gate)"
                )
            else:
                logger.debug(
                    f"[FutureBridge-no-future-validation] DROPPED bridge turn={turn_idx}: env verification failed"
                )

        return bridge_exps


@WORKFLOWS.register_module("FutureBridgeNoBridgeExecutionWebShopWorkflow")
class FutureBridgeNoBridgeExecutionWebShopWorkflow(FutureBridgeWebShopWorkflow):
    """Select a teacher action without executing it in the environment."""

    async def _try_kl_bridge(
        self,
        start_step: int,
        memory_snapshots: List[List[dict]],
        per_turn_kl: List[float],
        turn_responses: List[Experience],
    ) -> List[Experience]:
        if (
            self.bridge_failed_episodes_only
            and self._final_reward >= self.bridge_reward_threshold
        ):
            return []

        candidates = self._select_bridge_candidates(per_turn_kl)
        if not candidates:
            return []

        bridge_exps = []
        for bridge_rank, (turn_idx, disagreement) in enumerate(candidates):
            if bridge_rank >= self.bridge_max_per_ep:
                break
            cands = await self._generate_kl_bridge(
                memory_at_turn=memory_snapshots[turn_idx],
                trigger_kl=disagreement,
                bridge_idx=start_step + turn_idx,
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
