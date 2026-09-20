# -*- coding: utf-8 -*-
"""
FutureBridge-OPD workflows for ScienceWorld: FTB (full) and FTB w/o Bridge Exec..

FTB (FutureBridgeScienceWorldWorkflow):
  Extends TCOD_b2f_scienceworld_workflow with a KL-guided bridge mechanism:
  1. Run B2F episode (student from checkpoint, teacher logprobs computed).
  2. Find the turn with highest token-average teacher-student discrepancy.
  3. At that turn, restore env to that state using the gold trajectory.
  4. Teacher generates one bridge action from that state.
  5. Future validation: student rolls out a short continuation from the
     bridged state; the bridge is kept only when that continuation has a
     higher teacher-preferred token ratio than the original student suffix.
  6. Return bridge Experiences (OPD loss) alongside the original episode.

This mirrors the core logic of _FutureBridgeB2FAlfworldBase adapted to ScienceWorld.
"""

from typing import List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.scienceworld.TCOD_b2f_workflow import (
    TCOD_b2f_scienceworld_workflow,
)
from trinity.common.workflows.envs.TCOD.scienceworld.utils import (
    HISTORY_LENGTH,
    SCIWORLD_TEMPLATE,
    SCIWORLD_TEMPLATE_NO_HIS,
    _create_scienceworld_env_with_checkpoint,
    _format_history,
    _get_compact_action_info,
    format_observation,
    parse_action,
)


@WORKFLOWS.register_module("FutureBridgeScienceWorldWorkflow")
class FutureBridgeScienceWorldWorkflow(TCOD_b2f_scienceworld_workflow):
    """
    FutureBridge-OPD (FTB) for ScienceWorld.

    Adds KL-guided bridge with future validation on top of B2F:
    - After the student episode, identify the turn with max token-average
      teacher-student discrepancy.
    - At that turn, use the gold trajectory to restore the environment,
      the teacher generates a bridge action, and the same frozen student
      rolls out a short continuation from the bridged state.
    - Future validation gate: the bridge is retained only when the induced
      continuation has a higher teacher-preferred token ratio than the
      original student suffix (paper Eq. 5/6).
    - bridge_kl_lambda controls the bridge loss weight.
    - bridge_attempt_prob controls fraction of episodes attempting bridge.
    """

    _CONTINUATION_STEPS = 3

    def __init__(self, *, task: Task, model: ModelWrapper, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.bridge_kl_lambda = task.workflow_args.get("bridge_kl_lambda", 0.5)
        self.bridge_attempt_prob = float(task.workflow_args.get("bridge_attempt_prob", 1.0))
        self.bridge_max_steps = int(task.workflow_args.get("bridge_max_steps", 4))
        self.bridge_kl_max_per_ep = int(task.workflow_args.get("bridge_kl_max_per_ep", 1))

    async def _finalize_turn_responses(
        self, turn_responses: List[Experience], *, start_step: int
    ) -> List[Experience]:
        """Override to also generate bridge experiences after computing KL."""
        # 1. Run parent's finalize (computes teacher logprobs + metrics)
        normal_exps = await super()._finalize_turn_responses(
            turn_responses, start_step=start_step
        )

        # 2. Attempt bridge if training (not eval) and expert actions available
        if (
            self.is_eval
            or not self._expert_actions
            or not normal_exps
        ):
            return normal_exps

        import random
        if random.random() >= self.bridge_attempt_prob:
            return normal_exps

        bridge_exps = await self._try_kl_bridge(normal_exps, start_step)
        return normal_exps + bridge_exps

    async def _try_kl_bridge(
        self,
        turn_responses: List[Experience],
        start_step: int,
    ) -> List[Experience]:
        """
        Find max-KL turn, restore env to that state via gold trajectory,
        have teacher generate bridge action, student continues.
        Returns bridge Experiences or [].
        """
        if not turn_responses:
            return []

        # Find turn with max KL divergence
        kl_vals = []
        for resp in turn_responses:
            if resp.logprobs is not None and resp.teacher_logprobs is not None:
                kl = ((resp.logprobs - resp.teacher_logprobs).sum() / max(1, len(resp.logprobs))).item()
                kl_vals.append(kl)
            else:
                kl_vals.append(0.0)

        if not kl_vals:
            return []

        # Select highest KL turn (student deviated most)
        max_kl_idx = max(range(len(kl_vals)), key=lambda i: kl_vals[i])
        bridge_turn = start_step + max_kl_idx  # absolute step in gold trajectory

        # We need gold actions up to bridge_turn to restore env
        if bridge_turn >= len(self._expert_actions):
            return []

        cands = await self._generate_one_bridge(bridge_turn)
        if not cands:
            return []

        # Future validation gate (paper Eq. 5/6): restore the environment to
        # the bridge state, execute the bridge, and roll out the same frozen
        # student for H turns. Keep the bridge only when the induced
        # continuation has a higher teacher-preferred token ratio than the
        # original student suffix.
        bridge_exp = cands[0]
        bridge_action = parse_action(bridge_exp.response_text or "")
        if not bridge_action:
            return cands

        pos_after = await self._bridge_continuation_pos_ratio(
            bridge_turn=bridge_turn,
            bridge_action=bridge_action,
            bridge_memory=self.format_messages()
            + [{"role": "assistant", "content": bridge_exp.response_text or ""}],
        )
        if pos_after is None:
            return cands

        # Base continuation: the H student turns that followed the original
        # response in the B2F rollout (paper Eq. 5/6).
        base_turns = turn_responses[max_kl_idx + 1 : max_kl_idx + 1 + self._CONTINUATION_STEPS]
        if not base_turns:
            return []
        base_ratio = self._episode_pos_ratio(base_turns)

        if pos_after > base_ratio:
            logger.debug(
                f"[FutureBridge] KEPT bridge_turn={bridge_turn}: "
                f"after={pos_after:.3f} > base={base_ratio:.3f}"
            )
            return cands
        logger.debug(
            f"[FutureBridge] DROPPED bridge_turn={bridge_turn}: "
            f"after={pos_after:.3f} <= base={base_ratio:.3f}"
        )
        return []

    @staticmethod
    def _episode_pos_ratio(turn_responses: List[Experience]) -> float:
        """Teacher-preferred token ratio over the student suffix (paper Eq. 5)."""
        import torch as _torch

        pos, tot = 0, 0
        for resp in turn_responses:
            if resp.teacher_logprobs is None or resp.logprobs is None:
                continue
            tl = resp.teacher_logprobs if isinstance(resp.teacher_logprobs, _torch.Tensor) \
                else _torch.tensor(resp.teacher_logprobs)
            sl = resp.logprobs if isinstance(resp.logprobs, _torch.Tensor) \
                else _torch.tensor(resp.logprobs)
            adv = tl.float() - sl.float()
            pos += (adv > 0).sum().item()
            tot += len(adv)
        return pos / tot if tot > 0 else 0.5

    async def _bridge_continuation_pos_ratio(
        self, bridge_turn: int, bridge_action: str, bridge_memory: List[dict]
    ) -> "Optional[float]":
        """
        Rebuild the environment at bridge_turn, execute the bridge action,
        and roll out the frozen student for _CONTINUATION_STEPS turns.
        Returns the teacher-preferred token ratio of that continuation.
        """
        if bridge_turn >= len(self._expert_actions):
            return None
        try:
            result = _create_scienceworld_env_with_checkpoint(
                self.task_desc,
                self._expert_actions,
                bridge_turn,
                max_env_steps=self.max_env_steps,
            )
        except Exception:
            return None

        env, observation, info, history, task_description, start_step, done, _ = result
        if done:
            env.close()
            return None

        try:
            observation, reward, done, info = env.step(bridge_action)
            if done:
                return None

            kwargs = {**self.rollout_args, "n": 1}
            if kwargs.get("logprobs") is None:
                kwargs["logprobs"] = 0

            memory = list(bridge_memory)
            pos, tot = 0, 0
            for r in range(start_step + 1, min(start_step + 1 + self._CONTINUATION_STEPS, self.max_env_steps)):
                format_obs = format_observation(observation)
                action_templates, objects = _get_compact_action_info(env)
                ref_actions = ", ".join(f"'{s}'" for s in action_templates if s != "help")
                ref_objects = ", ".join(f"'{s}'" for s in objects)

                if len(history) < HISTORY_LENGTH:
                    user_content = SCIWORLD_TEMPLATE_NO_HIS.format(
                        task_description=task_description,
                        current_observation=format_obs,
                        action_templates=ref_actions,
                        objects=ref_objects,
                    )
                else:
                    action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                    user_content = SCIWORLD_TEMPLATE.format(
                        task_description=task_description,
                        current_observation=format_obs,
                        step_count=r,
                        history_length=min(HISTORY_LENGTH, len(history)),
                        action_history=action_history_str,
                        current_step=r + 1,
                        action_templates=ref_actions,
                        objects=ref_objects,
                    )

                memory = memory + [{"role": "user", "content": user_content}]
                try:
                    resps = await self.model.chat_async(memory, **kwargs)
                    resp = resps[0]
                except Exception:
                    break
                memory.append({"role": "assistant", "content": resp.response_text or ""})

                if resp.logprobs is None:
                    break
                try:
                    teacher_lp = await self.teacher_model.logprobs_async(
                        tokens=resp.tokens.tolist(),
                        temperature=self.temperature,
                    )
                except Exception:
                    break
                rs = resp.prompt_length - 1
                tl = teacher_lp[rs:]
                if len(tl) != len(resp.logprobs):
                    break
                import torch as _torch
                sl = resp.logprobs if isinstance(resp.logprobs, _torch.Tensor) \
                    else _torch.tensor(resp.logprobs)
                adv = _torch.tensor(tl).float() - sl.float()
                pos += (adv > 0).sum().item()
                tot += len(adv)

                action = parse_action(resp.response_text or "")
                history.append(_format_history(format_obs, r + 1, action))
                observation, reward, done, info = env.step(action)
                if done:
                    break

            return pos / tot if tot > 0 else None
        finally:
            env.close()

    async def _generate_one_bridge(self, bridge_turn: int) -> List[Experience]:
        """
        Restore env to bridge_turn using gold trajectory,
        teacher generates action, student continues for bridge_max_steps,
        compute OPD loss on student continuation.
        """
        if bridge_turn >= len(self._expert_actions):
            return []

        # Restore env to bridge_turn state using gold actions
        try:
            result = _create_scienceworld_env_with_checkpoint(
                self.task_desc,
                self._expert_actions,
                bridge_turn,
                max_env_steps=self.max_env_steps,
            )
        except Exception:
            return []

        env, observation, info, history, task_description, actual_step, done, checkpoint_reward = result
        if done:
            env.close()
            return []

        try:
            return await self._run_bridge_from_state(
                env, observation, info, history, task_description, actual_step
            )
        finally:
            env.close()

    async def _run_bridge_from_state(
        self,
        env,
        observation: str,
        info: dict,
        history: List[str],
        task_description: str,
        start_step: int,
    ) -> List[Experience]:
        """
        Teacher generates one action, then student continues for bridge_max_steps.
        Compute OPD loss (teacher_logprobs) on student continuation.
        """
        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        # Build prompt for teacher bridge action
        format_obs = format_observation(observation)
        action_templates, objects = _get_compact_action_info(env)
        ref_actions = ", ".join(f"'{s}'" for s in action_templates if s != "help")
        ref_objects = ", ".join(f"'{s}'" for s in objects)

        memory = self.format_messages()
        if len(history) < HISTORY_LENGTH:
            user_content = SCIWORLD_TEMPLATE_NO_HIS.format(
                task_description=task_description,
                current_observation=format_obs,
                action_templates=ref_actions,
                objects=ref_objects,
            )
        else:
            action_history_str = "\n".join(history[-HISTORY_LENGTH:])
            user_content = SCIWORLD_TEMPLATE.format(
                task_description=task_description,
                step_count=start_step,
                history_length=min(HISTORY_LENGTH, len(history)),
                action_history=action_history_str,
                current_step=start_step + 1,
                current_observation=format_obs,
                action_templates=ref_actions,
                objects=ref_objects,
            )

        memory = memory + [{"role": "user", "content": user_content}]

        # Teacher generates bridge action (greedy, no logprobs needed)
        # enable_thinking=False is set at model config level in yaml
        teacher_kwargs = {"n": 1, "logprobs": 0, "temperature": self.temperature}
        try:
            bridge_resps = await self.teacher_model.chat_async(memory, **teacher_kwargs)
            bridge_resp = bridge_resps[0]
        except Exception:
            return []

        bridge_text = bridge_resp.response_text or ""
        bridge_action = parse_action(bridge_text)
        if not bridge_action:
            return []

        # Step env with bridge action
        memory.append({"role": "assistant", "content": bridge_text})
        history = history + [_format_history(format_obs, start_step + 1, bridge_action)]
        observation, reward, done, info = env.step(bridge_action)
        start_step += 1
        if done:
            return []

        # Student continues for bridge_max_steps with OPD loss
        bridge_exps: List[Experience] = []
        best_score = info.get("score", 0)

        for r in range(start_step, min(start_step + self.bridge_max_steps, self.max_env_steps)):
            format_obs = format_observation(observation)
            action_templates, objects = _get_compact_action_info(env)
            ref_actions = ", ".join(f"'{s}'" for s in action_templates if s != "help")
            ref_objects = ", ".join(f"'{s}'" for s in objects)

            if len(history) < HISTORY_LENGTH:
                user_content = SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=task_description,
                    current_observation=format_obs,
                    action_templates=ref_actions,
                    objects=ref_objects,
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
                    action_templates=ref_actions,
                    objects=ref_objects,
                )

            memory = memory + [{"role": "user", "content": user_content}]
            try:
                resps = await self.model.chat_async(memory, **kwargs)
                resp = resps[0]
            except Exception:
                break

            resp_text = resp.response_text or ""
            memory.append({"role": "assistant", "content": resp_text})

            if resp.logprobs is None:
                break

            # Compute teacher logprobs for this student response (OPD loss)
            try:
                teacher_lp = await self.teacher_model.logprobs_async(
                    tokens=resp.tokens.tolist(),
                    temperature=self.temperature,
                )
                rs = resp.prompt_length - 1
                teacher_resp_lp = teacher_lp[rs:]
                if len(teacher_resp_lp) != len(resp.logprobs):
                    break

                resp.teacher_logprobs = teacher_resp_lp
                resp.reward = self.bridge_kl_lambda * best_score / 100.0
                resp.eid.run = getattr(self, "run_id_base", 0)
                resp.eid.step = r
                if resp.metrics is None:
                    resp.metrics = {}
                resp.metrics["bridge_verified"] = 1
                resp.metrics["env_done"] = 0.0
                resp.metrics["kl_divergence"] = (resp.logprobs - teacher_resp_lp).sum().item()
                bridge_exps.append(resp)
            except Exception:
                break

            action = parse_action(resp_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            best_score = max(best_score, info.get("score", best_score + reward))
            if done:
                if bridge_exps:
                    bridge_exps[-1].reward = self.bridge_kl_lambda
                    if bridge_exps[-1].metrics:
                        bridge_exps[-1].metrics["env_done"] = 1.0
                break

        return bridge_exps


@WORKFLOWS.register_module("FutureBridgeNoBridgeExecutionScienceWorldWorkflow")
class FutureBridgeNoBridgeExecutionScienceWorldWorkflow(FutureBridgeScienceWorldWorkflow):
    """
    FutureBridge-OPD w/o Bridge Execution (FTB-Gate) for ScienceWorld.

    Adds a pos_ratio gate on top of FTB:
    - pos_ratio per turn = fraction of tokens where teacher_logp > student_logp.
    - episode_pos_ratio = mean over all turns.
    - Bridge turn selected from top-k% KL turns.
    - Gate: only attempt bridge if the future segment (turns after bridge)
      has LOWER pos_ratio than episode mean, meaning student is struggling more
      going forward (bridge is more valuable).

    This mirrors FutureBridgeNoBridgeExecutionAlfworldWorkflow adapted for ScienceWorld.
    """

    def __init__(self, *, task: Task, model: ModelWrapper, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.bridge_kl_top_ratio = float(task.workflow_args.get("bridge_kl_top_ratio", 0.3))

    async def _try_kl_bridge(
        self,
        turn_responses: List[Experience],
        start_step: int,
    ) -> List[Experience]:
        """
        Gate without bridge execution: select turn by token-average KL, keep only when the future pos_ratio is below the episode average.
        """
        if not turn_responses:
            return []

        # 1. Compute pos_ratio and KL per turn
        turn_pos_ratio = []
        turn_kls = []

        for resp in turn_responses:
            # pos_ratio: fraction of tokens where teacher_logp > student_logp
            if resp.teacher_logprobs is not None and resp.logprobs is not None:
                tl = resp.teacher_logprobs if isinstance(resp.teacher_logprobs, torch.Tensor) \
                     else torch.tensor(resp.teacher_logprobs)
                sl = resp.logprobs if isinstance(resp.logprobs, torch.Tensor) \
                     else torch.tensor(resp.logprobs)
                adv = tl.float() - sl.float()
                pos = (adv > 0).sum().item()
                tot = len(adv)
                turn_pos_ratio.append(pos / tot if tot > 0 else 0.0)
                kl = ((sl - tl).sum() / max(1, len(sl))).item()
                turn_kls.append(kl)
            else:
                turn_pos_ratio.append(0.0)
                turn_kls.append(0.0)

        n_turns = len(turn_pos_ratio)
        if n_turns == 0:
            return []
        episode_pos_ratio = sum(turn_pos_ratio) / n_turns

        # 2. Select bridge turn: top-k% KL, gated by future pos_ratio
        k = max(1, int(n_turns * self.bridge_kl_top_ratio))
        kl_threshold = sorted(turn_kls, reverse=True)[k - 1] if turn_kls else 0.0

        bridge_turn_idx = None
        for turn_idx, kl_val in sorted(enumerate(turn_kls), key=lambda x: -x[1]):
            if kl_val < kl_threshold:
                break
            # Gate: future pos_ratio < episode mean → student struggling → bridge valuable
            future_ratios = turn_pos_ratio[turn_idx + 1:]
            if not future_ratios:
                # Last turn, no future → skip (no value in bridging)
                continue
            future_pos_ratio = sum(future_ratios) / len(future_ratios)
            if future_pos_ratio < episode_pos_ratio:
                bridge_turn_idx = turn_idx
                break

        if bridge_turn_idx is None:
            # Gate blocked all candidates → no bridge
            return []

        # 3. Convert turn_idx to absolute gold step
        bridge_turn = start_step + bridge_turn_idx
        if bridge_turn >= len(self._expert_actions):
            return []

        return await self._generate_one_bridge(bridge_turn)


@WORKFLOWS.register_module("FutureBridgeNoFutureValidationScienceWorldWorkflow")
class FutureBridgeNoFutureValidationScienceWorldWorkflow(FutureBridgeScienceWorldWorkflow):
    """
    Ablation without future validation (paper Table 3): bridges are generated
    and executed, but retained without comparing the induced student
    continuation against the original trajectory.
    """
    pass
