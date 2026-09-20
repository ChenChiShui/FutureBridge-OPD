# -*- coding: utf-8 -*-
"""
FutureBridge-OPD workflows for WebShop — mirroring the ALFWorld implementations.

FutureBridgeNoBridgeExecutionWebShopWorkflow (FTB w/o Bridge Exec.):
  Same KL bridge generation, but the gate compares future vs episode
  pos_ratio without executing the bridge in the environment.

FutureBridgeWebShopWorkflow (FTB, full):
  Same KL bridge generation, but verifies causally:
  rebuild env to the bridge state, student continues N=3 steps,
  keep the bridge only if pos_ratio improves after the teacher's correction.
"""

import copy
import logging
from dataclasses import asdict
from typing import List, Optional, Tuple

import torch

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


# ── Helper: compute pos_ratio for a response ─────────────────────────────────

def _pos_ratio(response: Experience) -> float:
    """fraction of tokens where teacher_logp > student_logp (by 0.01 threshold)."""
    if response.teacher_logprobs is None or response.logprobs is None:
        return 0.0
    tl = response.teacher_logprobs if isinstance(response.teacher_logprobs, torch.Tensor) \
         else torch.tensor(response.teacher_logprobs)
    sl = response.logprobs if isinstance(response.logprobs, torch.Tensor) \
         else torch.tensor(response.logprobs)
    adv = tl.float() - sl.float()
    tot = len(adv)
    return float((adv > 0).sum()) / tot if tot > 0 else 0.0


# ── FTB w/o Bridge Execution ───────────────────────────────────────────────────

@WORKFLOWS.register_module("FutureBridgeNoBridgeExecutionWebShopWorkflow")
class FutureBridgeNoBridgeExecutionWebShopWorkflow(_FutureBridgeWebShopBase):
    """
    WebShop FTB w/o Bridge Execution: pos_ratio future gate (mirrors ALFWorld).

    Gate: after bridge generation, keep only when
      future_pos_ratio(turns after t_bridge) < episode_pos_ratio
      → student genuinely struggles after that turn → bridge has value
    """

    async def _try_kl_bridge(
        self,
        start_step: int,
        memory_snapshots: List[List[dict]],
        per_turn_kl: List[float],
        turn_responses: List[Experience],
    ) -> List[Experience]:
        if self._final_reward >= self.bridge_reward_threshold:
            return []
        if not per_turn_kl:
            return []

        # 1. Per-turn pos_ratio (needs teacher_logprobs already set in _run_student_phase)
        turn_pos_ratio = [_pos_ratio(r) for r in turn_responses]
        n = len(turn_pos_ratio)
        episode_pos_ratio = sum(turn_pos_ratio) / n if n > 0 else 0.1

        # 2. Find highest-KL turn
        indexed_kl = sorted(enumerate(per_turn_kl), key=lambda x: x[1], reverse=True)
        bridge_exps: List[Experience] = []

        for rank, (turn_idx, trigger_kl) in enumerate(indexed_kl):
            if len(bridge_exps) >= self.bridge_max_per_ep:
                break
            if trigger_kl <= 0:
                continue

            # 3. Pos-ratio gate: student must struggle in future turns
            future_ratios = turn_pos_ratio[turn_idx + 1:]
            if future_ratios:
                future_pos_ratio = sum(future_ratios) / len(future_ratios)
                if future_pos_ratio >= episode_pos_ratio:
                    logger.debug(
                        f"[FutureBridgeNoBridgeExec] DROPPED turn={turn_idx}: "
                        f"future_pos={future_pos_ratio:.3f} >= episode={episode_pos_ratio:.3f}"
                    )
                    continue

            bridge_idx = start_step + turn_idx
            exps = await self._generate_kl_bridge(
                memory_at_turn=memory_snapshots[turn_idx],
                trigger_kl=trigger_kl,
                bridge_idx=bridge_idx,
                bridge_weight=1.0,
            )
            bridge_exps.extend(exps)

        return bridge_exps


# ── FTB (full): Student Continuation Gate ──────────────────────────────────────

@WORKFLOWS.register_module("FutureBridgeWebShopWorkflow")
class FutureBridgeWebShopWorkflow(_FutureBridgeWebShopBase):
    """
    WebShop FTB (full): env-based student continuation gate (mirrors ALFWorld).

    After bridge generation:
      1. Rebuild env (reset session, replay teacher B2F + student pre-bridge actions)
      2. Execute bridge action → new env state
      3. Student continues N=3 steps
      4. Keep bridge if pos_ratio improves after bridge vs episode baseline
    """

    _CONTINUATION_STEPS = 3

    def __init__(self, *, task: Task, model: ModelWrapper,
                 auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        # _teacher_actions (replayed reference prefix) and _episode_session_id
        # are maintained by the base class.
        self._episode_session_id: int = 0

    async def _try_kl_bridge(
        self,
        start_step: int,
        memory_snapshots: List[List[dict]],
        per_turn_kl: List[float],
        turn_responses: List[Experience],
    ) -> List[Experience]:
        if self._final_reward >= self.bridge_reward_threshold:
            return []
        if not per_turn_kl:
            return []

        # Per-turn teacher-preferred token counts, used for the base continuation
        def _pos_counts(r: Experience):
            if r.teacher_logprobs is None or r.logprobs is None:
                return (0, 0)
            tl = r.teacher_logprobs if isinstance(r.teacher_logprobs, torch.Tensor) \
                else torch.tensor(r.teacher_logprobs)
            sl = r.logprobs if isinstance(r.logprobs, torch.Tensor) \
                else torch.tensor(r.logprobs)
            adv = tl.float() - sl.float()
            return (int((adv > 0).sum()), len(adv))

        turn_pos_counts = [_pos_counts(r) for r in turn_responses]

        indexed_kl = sorted(enumerate(per_turn_kl), key=lambda x: x[1], reverse=True)
        bridge_exps: List[Experience] = []

        for rank, (turn_idx, trigger_kl) in enumerate(indexed_kl):
            if len(bridge_exps) >= self.bridge_max_per_ep:
                break
            if trigger_kl <= 0:
                continue

            bridge_idx = start_step + turn_idx

            # Generate bridge candidate
            cands = await self._generate_kl_bridge(
                memory_at_turn=memory_snapshots[turn_idx],
                trigger_kl=trigger_kl,
                bridge_idx=bridge_idx,
                bridge_weight=1.0,
            )
            if not cands:
                continue

            bridge_exp = cands[0]
            bridge_action = parse_action(bridge_exp.response_text or "")

            if not bridge_action:
                bridge_exps.extend(cands)  # fallback: keep
                continue

            # Student pre-bridge actions: parse from turn_responses
            student_pre_bridge = [
                parse_action(r.response_text or "") for r in turn_responses[:turn_idx]
            ]

            # Base continuation: the H student turns that followed the original
            # response in the B2F rollout (paper Eq. 5/6).
            base_counts = turn_pos_counts[turn_idx + 1 : turn_idx + 1 + self._CONTINUATION_STEPS]
            if not base_counts:
                continue
            base_pos = sum(p for p, _ in base_counts)
            base_tot = sum(t for _, t in base_counts)
            if base_tot == 0:
                continue
            base_ratio = base_pos / base_tot

            # Env-based gate: rebuild env and check student continuation
            pos_after = await self._student_continuation_pos_ratio(
                session_id=self._episode_session_id,
                teacher_actions=self._teacher_actions,
                student_pre_bridge=student_pre_bridge,
                bridge_action=bridge_action,
                bridge_memory=memory_snapshots[turn_idx] + [
                    {"role": "assistant", "content": bridge_exp.response_text or ""}
                ],
                episode_pos_ratio=base_ratio,
            )

            if pos_after is None or pos_after > base_ratio:
                bridge_exps.extend(cands)
                logger.debug(
                    f"[FutureBridge] KEPT bridge turn={turn_idx}: "
                    f"pos_after={pos_after:.3f} > base={base_ratio:.3f}"
                )
            else:
                logger.debug(
                    f"[FutureBridge] DROPPED bridge turn={turn_idx}: "
                    f"pos_after={pos_after:.3f} <= base={base_ratio:.3f}"
                )

        return bridge_exps


# ── FTB w/o Future Validation ──────────────────────────────────────────────────

@WORKFLOWS.register_module("FutureBridgeNoFutureValidationWebShopWorkflow")
class FutureBridgeNoFutureValidationWebShopWorkflow(FutureBridgeWebShopWorkflow):
    """
    Ablation without future validation (paper Table 3): bridges are generated
    and executed, but retained without comparing the induced student
    continuation against the original trajectory. A local bridge-token
    advantage gate is used instead, mirroring the ALFWorld ablation.
    """

    async def _try_kl_bridge(
        self,
        start_step: int,
        memory_snapshots: List[List[dict]],
        per_turn_kl: List[float],
        turn_responses: List[Experience],
    ) -> List[Experience]:
        if self._final_reward >= self.bridge_reward_threshold:
            return []
        if not per_turn_kl:
            return []

        indexed_kl = sorted(enumerate(per_turn_kl), key=lambda x: x[1], reverse=True)
        bridge_exps: List[Experience] = []

        for turn_idx, trigger_kl in indexed_kl:
            if len(bridge_exps) >= self.bridge_max_per_ep:
                break
            if trigger_kl <= 0:
                continue

            cands = await self._generate_kl_bridge(
                memory_at_turn=memory_snapshots[turn_idx],
                trigger_kl=trigger_kl,
                bridge_idx=start_step + turn_idx,
                bridge_weight=1.0,
            )
            if not cands:
                continue

            for exp in cands:
                if exp.teacher_logprobs is None or exp.logprobs is None:
                    bridge_exps.append(exp)
                    continue
                tl = exp.teacher_logprobs if isinstance(exp.teacher_logprobs, torch.Tensor) \
                    else torch.tensor(exp.teacher_logprobs)
                sl = exp.logprobs if isinstance(exp.logprobs, torch.Tensor) \
                    else torch.tensor(exp.logprobs)
                bridge_adv = (tl.float() - sl.float()).mean().item()
                if bridge_adv > 0:
                    bridge_exps.append(exp)
                    logger.debug(
                        f"[FutureBridgeNoFutureVal] KEPT bridge_adv={bridge_adv:.3f}"
                    )
                else:
                    logger.debug(
                        f"[FutureBridgeNoFutureVal] DROPPED bridge_adv={bridge_adv:.3f}"
                    )

        return bridge_exps

    async def _student_continuation_pos_ratio(
        self,
        session_id: int,
        teacher_actions: List[str],
        student_pre_bridge: List[str],
        bridge_action: str,
        bridge_memory: List[dict],
        episode_pos_ratio: float,
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

            # Replay teacher B2F actions
            for action in teacher_actions:
                if action:
                    obs, _, done, _ = env.step(action)
                    if done:
                        return None

            # Replay student pre-bridge actions
            for action in student_pre_bridge:
                if action:
                    obs, _, done, _ = env.step(action)
                    if done:
                        return None

            # Execute bridge action
            obs, _, done, _ = env.step(bridge_action)
            if done:
                return 1.0  # bridge completed task → always useful

            # Student continues N steps, track pos_ratio
            memory = list(bridge_memory)
            kwargs = {"n": 1, "temperature": self.temperature, "logprobs": 0}
            pos_count, total_count = 0, 0

            for _ in range(self._CONTINUATION_STEPS):
                available_actions = env.get_available_actions()
                formatted_obs = format_observation(obs)
                formatted_actions = _format_available_actions(available_actions)
                user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_desc,
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
                if action:
                    obs, _, done, _ = env.step(action)
                    if done:
                        break

            return pos_count / total_count if total_count > 0 else 0.5

        except Exception as e:
            logger.debug(f"[FutureBridge] env continuation failed: {e}")
            return None
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
