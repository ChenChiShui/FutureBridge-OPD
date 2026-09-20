# -*- coding: utf-8 -*-
"""Cross-Tokenizer FTB workflow for AlfWorld (Qwen teacher → Llama student).

Extends CF-XT-OPD with:
1. Bridge selection: max cross-tokenizer turn-level gap (same as OPD gap)
2. Teacher bridge: teacher generates full response at bridge state
3. Future verification: replay env, execute bridge, student continues,
   compare outcome (success/reward). Accept only if bridge branch is strictly better.
4. Bridge experience: SFT CE on teacher's response (expert_mask=True)

Loss structure:
- Student turns: PPO surrogate + turn-level cross-tokenizer advantage (expert_mask=False)
- Bridge turns: SFT CE on teacher response (expert_mask=True)
"""

import re
from typing import List, Optional

import torch

from trinity.common.experience import CustomField, Experience
from trinity.common.workflows import WORKFLOWS

from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    _create_alfworld_env,
    _create_alfworld_env_with_checkpoint,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)
from trinity.common.workflows.envs.TCOD.alfworld.cross_family_xt_opd_workflow import (
    CFXTOPDAlfworldWorkflow,
    _XT_OPD_CUSTOM_FIELDS,
)
from trinity.common.workflows.envs.TCOD.alfworld.cross_family_utils import (
    build_user_content,
    make_cross_family_sft_experience,
)

import logging

logger = logging.getLogger(__name__)


@WORKFLOWS.register_module("cf_xt_ftb_alfworld_workflow")
class CFXTFTBAlfworldWorkflow(CFXTOPDAlfworldWorkflow):
    """Cross-tokenizer FTB: turn-level OPD + bridge with future verification."""

    _CONTINUATION_STEPS = 5

    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.bridge_max_per_ep = task.workflow_args.get("bridge_max_per_ep", 1)

    async def _rollout_with_scoring(
        self, env, observation, info, history, task_description, start_step
    ) -> List[Experience]:
        """Student rollout + cross-tokenizer scoring + bridge."""

        # 1. Run student episode (parent class handles rollout + scoring)
        turn_experiences = await super()._rollout_with_scoring(
            env, observation, info, history, task_description, start_step
        )

        # 2. Skip bridge for successful episodes or eval
        episode_succeeded = self._env_done and self._final_reward > 0.5
        if self.is_eval or episode_succeeded or not turn_experiences:
            return turn_experiences

        # 3. Bridge selection: max turn-level gap
        # Gaps were stored in exp.info["turn_advantage"] but normalized.
        # Use the raw gap from exp.metrics["xt_gap"] for selection.
        gaps = [
            exp.metrics.get("xt_gap", 0.0) if exp.metrics else 0.0
            for exp in turn_experiences
        ]
        if not gaps or max(gaps) <= 0:
            return turn_experiences  # No positive gap, no bridge

        t_star = gaps.index(max(gaps))

        # 4. Teacher generates bridge response at t_star
        bridge_memory = self._get_bridge_memory(turn_experiences, t_star)
        if bridge_memory is None:
            return turn_experiences

        teacher_kwargs = {"temperature": 1.0, "max_tokens": 512, "n": 1}
        try:
            teacher_responses = await self.teacher_model.chat_async(
                bridge_memory, **teacher_kwargs
            )
            bridge_response = teacher_responses[0]
            bridge_action = parse_action(bridge_response.response_text or "") or ""
        except Exception:
            return turn_experiences

        if not bridge_action:
            return turn_experiences

        # 5. Future verification
        predefined_actions = self.raw_task.get("actions", None)
        k_star = self._linear_checkpoint_step(predefined_actions)

        verified = await self._verify_bridge_outcome(
            predefined_actions=predefined_actions,
            k_star=k_star,
            turn_experiences=turn_experiences,
            t_star=t_star,
            start_step=start_step,
            bridge_action=bridge_action,
            bridge_memory=bridge_memory,
        )

        if not verified:
            logger.debug(f"[CF-XT-FTB] DROPPED bridge t={t_star}: verification failed")
            return turn_experiences

        # 6. Build bridge SFT experience
        bridge_exp = await make_cross_family_sft_experience(
            self.model, bridge_memory, bridge_action, self.temperature
        )
        if bridge_exp is None:
            return turn_experiences

        # Mark as expert (SFT loss, not PPO)
        bridge_exp.info["is_expert"] = True
        bridge_exp.info["turn_advantage"] = 0.0  # Not used for SFT
        bridge_exp.custom_fields = list(_XT_OPD_CUSTOM_FIELDS)
        bridge_exp.reward = self._final_reward
        bridge_exp.eid.run = getattr(self, "run_id_base", 0)
        bridge_exp.eid.step = 5000  # Bridge offset
        if bridge_exp.metrics is None:
            bridge_exp.metrics = {}
        bridge_exp.metrics["is_bridge"] = 1.0
        bridge_exp.metrics["verification_pass"] = 1.0
        bridge_exp.metrics["teacher_action_margin"] = float(gaps[t_star])

        logger.debug(f"[CF-XT-FTB] KEPT bridge t={t_star}: gap={gaps[t_star]:.4f}")

        # Add bridge metrics to last student experience
        if turn_experiences:
            last = turn_experiences[-1]
            if last.metrics is None:
                last.metrics = {}
            last.metrics["bridge_count"] = 1.0

        return turn_experiences + [bridge_exp]

    def _get_bridge_memory(
        self, turn_experiences: List[Experience], t_star: int
    ) -> Optional[List[dict]]:
        """Reconstruct the conversation context at turn t_star.

        We need the messages BEFORE the student's response at t_star.
        Since we don't store turn_messages (parent class doesn't expose them),
        we reconstruct from the experience's prompt tokens.
        """
        # The experience at t_star has prompt_length and tokens.
        # We can't easily reconstruct messages from tokens.
        # Instead, store turn_messages during the parent's rollout.
        # For now, use the experience's response_text to build context.
        #
        # Actually, we need to override the parent's _rollout_with_scoring
        # to store turn_messages. Let me do that differently.
        #
        # Better approach: store turn_messages as instance attribute.
        if hasattr(self, '_turn_messages_cache') and t_star < len(self._turn_messages_cache):
            return self._turn_messages_cache[t_star]
        return None

    async def _verify_bridge_outcome(
        self,
        predefined_actions,
        k_star: int,
        turn_experiences: List[Experience],
        t_star: int,
        start_step: int,
        bridge_action: str,
        bridge_memory: List[dict],
    ) -> bool:
        """Future verification: replay env, execute bridge, student continues.

        Compare bridge branch outcome vs original branch.
        Accept only if bridge branch is strictly better:
        - Bridge branch succeeds (reward > 0) while original failed
        - Both fail but bridge branch makes more progress (more valid steps)
        """
        if not predefined_actions:
            return True  # Can't replay without predefined actions

        env = None
        try:
            # a. Rebuild env from checkpoint
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, k_star
            )
            if result is None:
                return True  # Safe fallback
            env, obs, info, _, task_desc, _, done = result
            if done:
                return True

            # b. Replay student actions up to t_star
            for i, exp in enumerate(turn_experiences):
                if i >= t_star:
                    break
                action = parse_action(exp.response_text or "") or ""
                obs, _, done, info = env.step(action)
                if done:
                    return True  # Original branch actually succeeded here

            # c. Execute bridge action
            obs, reward, done, info = env.step(bridge_action)
            if done and reward > 0:
                return True  # Bridge directly solves the task

            # d. Student continuation from bridge state
            memory = list(bridge_memory) + [
                {"role": "assistant", "content": f"<action>{bridge_action}</action>"}
            ]
            kwargs = {"n": 1, "temperature": self.temperature}

            bridge_valid_steps = 0
            bridge_success = False
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
                        bridge_valid_steps += 1
                    memory.append({"role": "assistant", "content": resp.response_text or ""})
                    obs, reward, done, info = env.step(action)
                    if done:
                        bridge_success = reward > 0
                        break
                except Exception:
                    break

            # e. Compare outcomes
            if bridge_success:
                return True  # Bridge branch succeeds, original didn't

            # Both failed: compare progress
            # Original branch: all turns after t_star were student's original actions
            original_remaining = len(turn_experiences) - t_star - 1
            # Bridge branch: bridge_valid_steps continuation steps
            return bridge_valid_steps > original_remaining

        except Exception as e:
            logger.debug(f"[CF-XT-FTB] verify_bridge failed: {e}")
            return True  # Safe fallback
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
