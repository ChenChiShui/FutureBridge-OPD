# -*- coding: utf-8 -*-
"""Bridge Probe workflow for ALFWorld — extends TCOD-B2F.

Probe experiment: insert a legal-but-wrong action at the teacher prefix
boundary, then compare:
  A  clean B2F prefix       → student continuation          (baseline)
  B  corrupted prefix       → student continuation          (can student recover?)
  C  corrupted prefix       → teacher bridge (1-N steps) → student continuation
  D  corrupted prefix       → teacher runs to end          (upper bound)

Key metrics logged per episode:
  env_done           — did student/teacher solve the task?
  probe_condition    — which of A/B/C/D this episode used
  corrupt_action     — the wrong action that was injected (B/C/D only)
  corrupt_step       — teacher prefix length at corruption point
  bridge_length      — how many teacher bridge steps were used (C only)

Usage: set `probe_condition` and `bridge_length` in workflow_args of the config.

NOTE: This file does NOT modify any existing workflow code.
"""

import random
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
    _create_alfworld_env,
    _create_alfworld_env_with_checkpoint,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_corrupt_step(actions: List[str], ratio_min: float, ratio_max: float) -> int:
    """Pick a corruption insertion point in [ratio_min, ratio_max) of trajectory."""
    n = len(actions)
    lo = max(1, int(n * ratio_min))
    hi = max(lo + 1, int(n * ratio_max))
    hi = min(hi, n)          # can't corrupt beyond last step
    return random.randint(lo, hi - 1) if lo < hi else lo


def _get_wrong_action(
    info: dict,
    true_action: str,
    corrupt_types: List[str],
) -> Optional[str]:
    """Return a legal-but-wrong action of the requested type, or None.

    Only uses admissible_commands from the environment; never returns
    invalid/help actions.

    Args:
        info: env info dict containing `admissible_commands`
        true_action: the action the teacher would have taken (to avoid)
        corrupt_types: list of perturbation types to try in order
          - "wrong_navigation": different "go to X" action
          - "wrong_pickup":     different "take X" action

    Returns:
        A wrong-but-legal action string, or None if no candidate found.
    """
    admissible = info.get("admissible_commands", [])
    if admissible and isinstance(admissible[0], list):
        admissible = admissible[0]

    # Filter out the true action and 'help'
    candidates = [a for a in admissible if a != true_action and a != "help"]

    for ctype in corrupt_types:
        if ctype == "wrong_navigation" and true_action.startswith("go to"):
            pool = [a for a in candidates if a.startswith("go to")]
            if pool:
                return random.choice(pool)

        elif ctype == "wrong_pickup" and true_action.startswith("take "):
            pool = [a for a in candidates if a.startswith("take ")]
            if pool:
                return random.choice(pool)

    # Fallback: any safe admissible action that is navigation or pickup
    safe = [a for a in candidates
            if a.startswith("go to") or a.startswith("take ")]
    if safe:
        return random.choice(safe)

    return None


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@WORKFLOWS.register_module("bridge_probe_b2f_workflow")
class BridgeProbeB2FWorkflow(TCOD_b2f_alfworld_workflow):
    """Bridge Probe workflow — B2F with optional corruption and teacher bridge.

    New workflow_args (all optional):
        probe_condition   str   "A" / "B" / "C" / "D"  (default "B")
        bridge_length     int   teacher bridge steps for condition C (default 1)
        corrupt_ratio_min float lower bound for corruption point (default 0.2)
        corrupt_ratio_max float upper bound for corruption point (default 0.6)
        corrupt_types     list  perturbation types to attempt, in priority order
                                default: ["wrong_navigation", "wrong_pickup"]
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
        self.probe_condition   = wargs.get("probe_condition",   "B")
        self.bridge_length     = int(wargs.get("bridge_length",     1))
        self.corrupt_ratio_min = float(wargs.get("corrupt_ratio_min", 0.2))
        self.corrupt_ratio_max = float(wargs.get("corrupt_ratio_max", 0.6))
        self.corrupt_types     = wargs.get(
            "corrupt_types", ["wrong_navigation", "wrong_pickup"]
        )

        # Probe-specific bookkeeping (reset per episode)
        self._probe_corrupt_action: str = ""
        self._probe_corrupt_step:   int = 0
        self._probe_bridge_steps:   int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_async(self) -> List[Experience]:
        """Dispatch to the right probe condition."""
        # Condition A = plain B2F (parent handles everything)
        if self.probe_condition == "A":
            return await super().run_async()

        # B / C / D need expert actions; fall back gracefully if absent
        actions = self.raw_task.get("actions", None)
        if not actions or len(actions) < 3:
            # No expert trajectory — run plain eval episode
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env)
            finally:
                env.close()

        return await self._run_probe(actions)

    # ------------------------------------------------------------------
    # Core probe logic
    # ------------------------------------------------------------------

    async def _run_probe(self, actions: List[str]) -> List[Experience]:
        """Execute the probe for conditions B / C / D."""

        # 1. Pick corruption point
        corrupt_step = _pick_corrupt_step(
            actions, self.corrupt_ratio_min, self.corrupt_ratio_max
        )
        self._probe_corrupt_step = corrupt_step

        # 2. Replay clean teacher prefix up to corrupt_step
        (
            env,
            observation,
            info,
            history,
            task_description,
            prefix_len,
            replay_done,
        ) = _create_alfworld_env_with_checkpoint(
            self.task_desc, actions, corrupt_step
        )

        try:
            # If prefix already solved the task, nothing to probe
            if replay_done:
                self._env_done   = True
                self._env_rounds = prefix_len
                self._final_reward = 1.0
                return []

            # 3. Inject wrong action
            true_action   = actions[corrupt_step]
            wrong_action  = _get_wrong_action(info, true_action, self.corrupt_types)

            if wrong_action is None:
                # Cannot corrupt this step — fall back to condition A from here
                return await self._run_episode_from_checkpoint(
                    env, observation, info, history, task_description, prefix_len
                )

            self._probe_corrupt_action = wrong_action

            # Execute the wrong action in the environment
            format_obs_pre_corrupt = format_observation(observation)
            history = list(history) + [
                _format_history(format_obs_pre_corrupt, prefix_len + 1, wrong_action)
            ]
            observation, _reward, done, info = env.step(wrong_action)
            current_step = prefix_len + 1

            if done:
                # Wrong action accidentally finished the task (edge case)
                self._env_done    = True
                self._env_rounds  = current_step
                self._final_reward = 1.0
                return []

            # 4. Dispatch by condition
            if self.probe_condition == "B":
                experiences = await self._run_episode_from_checkpoint(
                    env, observation, info, history, task_description, current_step
                )

            elif self.probe_condition == "C":
                (
                    observation,
                    info,
                    history,
                    current_step,
                    bridge_done,
                ) = await self._run_teacher_bridge(
                    env, observation, info, history, task_description,
                    current_step, self.bridge_length
                )
                self._probe_bridge_steps = (
                    current_step - prefix_len - 1  # steps after wrong action
                )
                if bridge_done:
                    # Teacher bridge already solved it
                    self._env_done    = True
                    self._env_rounds  = current_step
                    self._final_reward = 1.0
                    # Return a minimal experience so metrics are logged
                    experiences = await self._run_one_student_step_for_metrics(
                        env, observation, info, history, task_description, current_step
                    )
                else:
                    experiences = await self._run_episode_from_checkpoint(
                        env, observation, info, history, task_description, current_step
                    )

            elif self.probe_condition == "D":
                # Teacher runs to completion — no student contribution
                (
                    observation,
                    info,
                    history,
                    current_step,
                    teacher_done,
                ) = await self._run_teacher_bridge(
                    env, observation, info, history, task_description,
                    current_step, self.max_env_steps  # up to max
                )
                self._env_done    = teacher_done
                self._env_rounds  = current_step
                self._final_reward = 1.0 if teacher_done else 0.0
                # Single student step to carry metrics
                experiences = await self._run_one_student_step_for_metrics(
                    env, observation, info, history, task_description, current_step
                )

            else:
                raise ValueError(f"Unknown probe_condition: {self.probe_condition!r}")

            # 5. Append probe-specific metrics to the last experience
            self._attach_probe_metrics(experiences)
            return experiences

        finally:
            env.close()

    # ------------------------------------------------------------------
    # Teacher bridge runner
    # ------------------------------------------------------------------

    async def _run_teacher_bridge(
        self,
        env,
        observation: str,
        info: dict,
        history: List[str],
        task_description: str,
        start_step: int,
        n_steps: int,
    ) -> Tuple[str, dict, List[str], int, bool]:
        """Let teacher generate up to n_steps actions from the current state.

        Returns:
            (observation, info, history, current_step, done)
        """
        kwargs = {**asdict(self.task.rollout_args), "n": 1, "logprobs": 0}
        current_obs  = observation
        current_info = info
        current_hist = list(history)
        done         = False
        steps_taken  = 0   # count how many steps actually executed

        for _ in range(n_steps):
            r = start_step + steps_taken
            format_obs = format_observation(current_obs)
            admissible = current_info.get("admissible_commands", [])
            if admissible and isinstance(admissible[0], list):
                admissible = admissible[0]
            reformatted = "\n ".join(f"'{s}'" for s in admissible if s != "help")

            if len(current_hist) < HISTORY_LENGTH:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_obs,
                    admissible_actions=reformatted,
                )
            else:
                action_history_str = "\n".join(current_hist[-HISTORY_LENGTH:])
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(current_hist)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=format_obs,
                    admissible_actions=reformatted,
                )

            messages = [{"role": "user", "content": user_content}]
            responses = await self.teacher_model.chat_async(messages, **kwargs)
            response  = responses[0]
            action    = parse_action(response.response_text or "")

            current_hist = current_hist + [_format_history(format_obs, r + 1, action)]
            current_obs, _reward, done, current_info = env.step(action)
            steps_taken += 1

            if done:
                break

        # final_step: environment step index after bridge
        final_step = start_step + steps_taken
        return current_obs, current_info, current_hist, final_step, done

    # ------------------------------------------------------------------
    # Single student step (to carry probe metrics when teacher finishes)
    # ------------------------------------------------------------------

    async def _run_one_student_step_for_metrics(
        self,
        env,
        observation: str,
        info: dict,
        history: List[str],
        task_description: str,
        current_step: int,
    ) -> List[Experience]:
        """Run one student step purely to get a valid Experience object.

        Used when teacher already finished (condition C bridge done, or D)
        so that probe metrics are attached to a real Experience and logged.
        The game is over, so this step has no environmental impact.
        """
        kwargs = {**asdict(self.task.rollout_args), "n": 1, "logprobs": 0}
        format_obs = format_observation(observation)
        admissible = info.get("admissible_commands", [])
        if admissible and isinstance(admissible[0], list):
            admissible = admissible[0]
        reformatted = "\n ".join(f"'{s}'" for s in admissible if s != "help")

        if len(history) < HISTORY_LENGTH:
            user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=format_obs,
                admissible_actions=reformatted,
            )
        else:
            user_content = ALFWORLD_TEMPLATE.format(
                task_description=task_description,
                step_count=current_step,
                history_length=min(HISTORY_LENGTH, len(history)),
                action_history="\n".join(history[-HISTORY_LENGTH:]),
                current_step=current_step + 1,
                current_observation=format_obs,
                admissible_actions=reformatted,
            )

        messages = [{"role": "user", "content": user_content}]
        responses = await self.model.chat_async(messages, **kwargs)
        response  = responses[0]

        # Compute teacher logprobs on this student response (required by pipeline)
        teacher_logprobs = await self.teacher_model.logprobs_async(
            tokens=response.tokens.tolist(),
            temperature=self.temperature,
        )
        resp_start = response.prompt_length - 1
        response.teacher_logprobs = teacher_logprobs[resp_start:]

        response.reward = self._final_reward
        response.eid.run  = getattr(self, "run_id_base", 0)
        response.eid.step = current_step

        if response.metrics is None:
            response.metrics = {}
        # Reflect the already-computed outcome
        response.metrics["env_done"]     = 1.0 if self._env_done else 0.0
        response.metrics["env_rounds"]   = self._env_rounds
        response.metrics["if_teacher"]   = (
            1 if self._probe_corrupt_step > 0 else 0
        )
        return [response]

    # ------------------------------------------------------------------
    # Attach probe-specific metrics to last experience
    # ------------------------------------------------------------------

    def _attach_probe_metrics(self, experiences: List[Experience]) -> None:
        """Append bridge-probe metadata to the last experience's metrics dict."""
        if not experiences:
            return
        last = experiences[-1]
        if last.metrics is None:
            last.metrics = {}
        last.metrics["probe_condition"]  = ord(self.probe_condition)   # A=65 B=66 C=67 D=68
        last.metrics["corrupt_step"]     = self._probe_corrupt_step
        last.metrics["bridge_steps_used"] = self._probe_bridge_steps
