"""
Bridge-TCOD KL: Future-Compatible Bridge via KL Divergence

Core insight:
  Teacher-student KL divergence on the student's action IS the future
  incompatibility signal — no environment replay needed.

  High KL(student || teacher) on action a_t
    ← teacher strongly disagrees with student's choice
    ← current state has low "future compatibility"
    ← bridge needed: teacher shows optimal action from same context

Advantages over env-based bridge:
  - Zero extra environment calls (no S/BS condition replay)
  - No dependency on expert trajectories or benchmark-specific replay
  - No rejection sampling — every high-KL turn produces a bridge signal
  - Works for any environment that supports OPD training

Training signal:
  L = L_OPD (normal)
    + bridge_kl_lambda * L_bridge (on teacher's optimal action tokens,
                                    weighted by how much KL exceeded threshold)

Extra workflow_args:
  bridge_kl_top_ratio:   bridge top-K% turns by KL divergence     (default: 0.3)
  bridge_kl_lambda:     bridge loss weight               (default: 0.5)
  bridge_max_per_ep: max bridge turns per episode     (default: 1)
"""

import copy
from dataclasses import asdict
from typing import List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.alfworld.TCOD_b2f_workflow import (
    TCOD_b2f_alfworld_workflow,
)
from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    parse_action,
    format_observation,
    _format_history,
    _create_alfworld_env,
    _extract_task,
)


@WORKFLOWS.register_module("Bridge_TCOD_kl_alfworld_workflow")
class Bridge_TCOD_kl_alfworld_workflow(TCOD_b2f_alfworld_workflow):
    """
    KL-Triggered Bridge: purely model-based future compatibility detection.

    At each student turn, action_kl = sum(student_lp - teacher_lp).
    Bridge top bridge_kl_top_ratio fraction of turns by KL (student overconfident):
      → teacher generates its optimal response from the same context
      → student learns to imitate teacher's optimal action (bridge experience)

    No environment replay. No suffix checking. No expert trajectories needed
    beyond what TCOD B2F already uses for the curriculum.
    """

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        wargs = task.workflow_args
        self.bridge_kl_top_ratio  = float(wargs.get("bridge_kl_top_ratio",  0.3))
        self.bridge_kl_lambda     = float(wargs.get("bridge_kl_lambda",     0.5))
        self.bridge_max_per_ep = int(wargs.get("bridge_max_per_ep",   1))


    async def run_async(self):
        """
        Override to always call _run_episode_from_checkpoint (our KL bridge version)
        even when checkpoint_step=0 (teacher fully withdrawn).

        Parent TCOD_b2f falls through to _run_episode when checkpoint_step=0,
        which bypasses our KL bridge. We route ALL training episodes through
        _run_episode_from_checkpoint so KL bridge fires throughout training.
        """
        import re as _re
        from trinity.common.workflows.envs.TCOD.alfworld.utils import _create_alfworld_env_with_checkpoint

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
                if m: current_step = int(m.group(1))

        self.set_training_progress(current_step, self.total_steps)

        predefined_actions = self.raw_task.get("actions", None)
        effective_checkpoint_step = self._linear_checkpoint_step(predefined_actions)

        if effective_checkpoint_step is not None and effective_checkpoint_step > 0 and predefined_actions:
            result = _create_alfworld_env_with_checkpoint(
                self.task_desc, predefined_actions, effective_checkpoint_step
            )
            if result is None:
                env = _create_alfworld_env(self.task_desc)
                try:
                    return await self._run_episode(env)
                finally:
                    env.close()
            env, obs, info, history, task_desc, _, done = result
            if done:
                env.close()
                return []
            try:
                return await self._run_episode_from_checkpoint(
                    env, obs, info, history, task_desc, effective_checkpoint_step
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


    async def _run_episode_from_checkpoint(
        self,
        env,
        observation: str,
        info: dict,
        history: List[str],
        task_description: str,
        start_step: int,
    ) -> List[Experience]:
        """
        1. Run B2F student episode (parent logic), capturing memory at each step.
        2. After teacher logprobs are computed, check per-turn KL.
        3. For high-KL turns: generate teacher's optimal response as bridge.
        """
        normal_exps, turn_memories = await self._b2f_episode_with_memories(
            env, observation, info, history, task_description, start_step
        )

        if self.is_eval or not normal_exps:
            return normal_exps

        episode_succeeded = self._env_done and self._final_reward > 0.5
        if episode_succeeded:
            return normal_exps

        bridge_exps = await self._try_kl_bridges(normal_exps, turn_memories)
        return normal_exps + bridge_exps


    async def _b2f_episode_with_memories(
        self,
        env,
        observation: str,
        info: dict,
        history: List[str],
        task_description: str,
        start_step: int,
    ) -> Tuple[List[Experience], List[List[dict]]]:
        """
        Standard B2F student episode + OPD teacher logprobs.
        Returns (turn_responses, turn_memories) where turn_memories[i] is the
        conversation context BEFORE the student's response at step i
        (suitable for asking teacher: "what would you do here?").
        """
        expert_actions = self.raw_task.get("actions", [])
        original_suffix = (
            expert_actions[start_step + 1:] if start_step + 1 < len(expert_actions) else []
        )

        self._env_done   = False
        self._env_rounds = start_step
        self._final_reward = 0.0

        memory = self.format_messages()
        turn_responses: List[Experience] = []
        turn_memories:  List[List[dict]] = []

        kwargs = {**asdict(self.task.rollout_args), "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        handoff_captured = False

        for r in range(start_step, self.max_env_steps):
            admissible = info.get("admissible_commands", [])
            if admissible and isinstance(admissible[0], list):
                admissible = admissible[0]
            reformatted = "\n ".join(f"'{s}'" for s in admissible if s != "help")

            if len(history) < HISTORY_LENGTH:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_observation(observation),
                    admissible_actions=reformatted,
                )
            else:
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history="\n".join(history[-HISTORY_LENGTH:]),
                    current_step=r + 1,
                    current_observation=format_observation(observation),
                    admissible_actions=reformatted,
                )

            memory = memory + [{"role": "user", "content": user_content}]

            turn_memories.append(list(memory))

            responses = await self.model.chat_async(memory, **kwargs)
            response  = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "Bridge-TCOD-KL requires logprobs. Set rollout_args.logprobs (e.g. 0)."
                )
            turn_responses.append(response)

            student_action = parse_action(response_text)

            if not handoff_captured and len(original_suffix) > 0:
                handoff_captured = True
                self._kl_dev_info = {
                    "start_step": start_step,
                    "original_suffix": original_suffix,
                }

            history = history + [_format_history(
                format_observation(observation), r + 1, student_action
            )]
            observation, _, done, info = env.step(student_action)

            if done:
                self._env_done   = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds   = self.max_env_steps
            self._final_reward = 0.0

        per_turn_kl = []
        expert_actions = self.raw_task.get("actions", [])
        total_expert = len(expert_actions)
        expert_remaining = total_expert - start_step

        for i, resp in enumerate(turn_responses):
            teacher_lp = await self.teacher_model.logprobs_async(
                tokens=resp.tokens.tolist(),
                temperature=self.temperature,
            )
            rs = resp.prompt_length - 1
            resp.teacher_logprobs = teacher_lp[rs:]
            if resp.metrics is None:
                resp.metrics = {}
            resp.reward   = self.compute_reward(resp)
            resp.eid.run  = getattr(self, "run_id_base", 0)
            resp.eid.step = start_step + i

            kl = (resp.logprobs - resp.teacher_logprobs).sum().item()
            per_turn_kl.append(kl)

        total_kl = sum(per_turn_kl)
        if turn_responses:
            last = turn_responses[-1]
            if last.metrics is None:
                last.metrics = {}
            last.metrics["student_env_rounds"]          = self._env_rounds - start_step
            last.metrics["teacher_env_rounds"]          = start_step
            last.metrics["if_teacher"]                  = 1 if start_step > 0 else 0
            last.metrics["expected_teacher_env_rounds"] = expert_remaining
            last.metrics["env_rounds"]                  = self._env_rounds
            last.metrics["env_done"]                    = 1.0 if self._env_done else 0.0
            last.metrics["kl_divergence"]               = total_kl
            last.metrics["bridge_verified"]             = 0

        return turn_responses, turn_memories


    async def _try_kl_bridges(
        self,
        turn_responses: List[Experience],
        turn_memories: List[List[dict]],
    ) -> List[Experience]:
        """
        Percentile-based bridge trigger: bridge the top bridge_kl_top_ratio
        fraction of turns by KL divergence, adaptive to this episode's distribution.

        Avoids hardcoded thresholds that break as KL evolves across training stages.
        e.g., top_ratio=0.3 → bridge the worst 30% of turns per episode.

        Bridge weight = relative KL rank within episode, scaled to [0, 1].
        """
        import re as _re
        def _has_valid_action(resp):
            text = resp.response_text or ""
            return bool(_re.search(r"<action>.*?</action>", text, _re.DOTALL))

        turn_kls = []
        for resp in turn_responses:
            if not _has_valid_action(resp):
                turn_kls.append(0.0)
            elif resp.teacher_logprobs is not None and resp.logprobs is not None:
                kl = (resp.logprobs - resp.teacher_logprobs).sum().item()
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
        kl_max = sorted_kl_vals[0]
        kl_min = sorted_kl_vals[-1]
        kl_range = max(kl_max - kl_min, 1e-6)

        bridge_exps = []
        triggered   = 0

        for turn_idx, kl_val in sorted_turns:
            if triggered >= self.bridge_max_per_ep:
                break
            if kl_val < kl_threshold:
                break

            memory_at_turn = turn_memories[turn_idx]
            weight = (kl_val - kl_min) / kl_range
            exps = await self._generate_kl_bridge(
                memory_at_turn=memory_at_turn,
                trigger_kl=kl_val,
                bridge_idx=triggered,
                bridge_weight=weight,
            )
            bridge_exps.extend(exps)
            triggered += 1

        return bridge_exps

    async def _generate_kl_bridge(
        self,
        memory_at_turn: List[dict],
        trigger_kl: float,
        bridge_idx: int,
        bridge_weight: float = 1.0,
    ) -> List[Experience]:
        """
        Generate teacher's optimal response to the same context as the
        high-KL student turn.  Student then learns to imitate teacher.

        bridge_weight: normalized rank within the episode [0, 1].
                       Computed by caller based on percentile, not hardcoded threshold.
        """
        kwargs_teacher = {
            **asdict(self.task.rollout_args),
            "n": 1, "logprobs": 0,
            "temperature": self.temperature,
        }

        try:
            bridge_resps = await self.teacher_model.chat_async(
                memory_at_turn, **kwargs_teacher
            )
            bridge_resp = bridge_resps[0]

            full_tokens = bridge_resp.tokens.tolist()
            resp_start  = bridge_resp.prompt_length - 1

            student_lp_full = await self.model.logprobs_async(
                tokens=full_tokens, temperature=self.temperature
            )
            teacher_lp_full = await self.teacher_model.logprobs_async(
                tokens=full_tokens, temperature=self.temperature
            )
        except Exception:
            return []

        student_lp = student_lp_full[resp_start:]
        teacher_lp = teacher_lp_full[resp_start:]

        exp = copy.copy(bridge_resp)
        exp.logprobs         = student_lp
        exp.teacher_logprobs = teacher_lp
        exp.reward           = 0.0
        exp.eid.run          = getattr(self, "run_id_base", 0)
        exp.eid.step         = 5000 + bridge_idx

        if exp.metrics is None:
            exp.metrics = {}
        exp.metrics["bridge_verified"] = 1
        exp.metrics["bridge_lambda"]   = self.bridge_kl_lambda * bridge_weight
        exp.metrics["trigger_kl"]      = trigger_kl
        exp.metrics["bridge_weight"]   = bridge_weight
        exp.metrics["is_bridge"]       = 1

        return [exp]
