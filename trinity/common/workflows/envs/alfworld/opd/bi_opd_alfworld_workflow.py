# -*- coding: utf-8 -*-
"""Bidirectional On-Policy Distillation (Bi-OPD) workflow for AlfWorld.

This workflow implements a gradual distillation strategy where:
- Student executes actions independently (no teacher action intervention)
- Teacher gradually distills more previous steps as training progresses:
  * Initially: distills only the first step
  * After decay_step: distills first 2 steps
  * After 2*decay_step: distills first 3 steps
  * Eventually: distills all previous steps

The distillation happens on the prefix (previous actions), not on the current action.
"""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow

from trinity.common.workflows.envs.alfworld.opd.prompts import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
)
from trinity.common.workflows.envs.alfworld.alfworld_workflow import (
    parse_action,
)

HISTORY_LENGTH = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def format_observation(observation: str):
    return observation.strip()

def _extract_task(text_obs: str) -> str:
    """Extract the task description from the text observation."""
    task_start = text_obs.find("Your task is to: ")
    if task_start != -1:
        return text_obs[task_start + len("Your task is to: ") :].strip()
    raise ValueError("Task description not found in text observation.")


def _format_history(observation: str, step_num: int, act: str) -> str:
    """Format observation and action for action_history."""
    return MEMORY_FORMAT.format(step_num=step_num, obs=observation.strip(), act=act)

def _create_alfworld_env(game_file_path: str):
    """Create AlfWorld textworld environment."""
    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredExpert,
            AlfredExpertType,
        )

        expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
        request_infos = textworld.EnvInfos(
            description=True, inventory=True, admissible_commands=True
        )
        env_id = textworld.gym.register_game(
            game_file_path, request_infos, wrappers=[AlfredDemangler(), expert]
        )
        return textworld.gym.make(env_id)
    except Exception as e:
        raise ImportError(
            f"Error creating AlfWorld env: {e}. "
            "Ensure alfworld is installed: "
        ) from e


class BiOpdAlfworldWorkflow(Workflow):
    """Bidirectional On-Policy Distillation workflow for AlfWorld.

    Key differences from standard OPD:
    - Student executes all actions independently (no teacher intervention)
    - Teacher gradually distills more previous steps as training progresses
    - Distillation is applied to the prefix (history), not the current action

    The decay_step parameter controls how quickly the distillation window grows:
    - Training step 0 to decay_step-1: distill first 1 step
    - Training step decay_step to 2*decay_step-1: distill first 2 steps
    - Training step 2*decay_step to 3*decay_step-1: distill first 3 steps
    - And so on...

    Use advantage_fn: multi_turn_opd (MultiTurnOpdAdvantage) for this workflow.
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.reset(task)
        self.is_eval = task.is_eval

        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "Bi-OPD requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)

        # Decay step: controls how quickly distillation window grows
        # e.g., decay_step=5 means every 5 training steps, distill one more previous step
        self.decay_step = task.workflow_args.get("decay_step", 5)

        # Track current training step for dynamic distillation window
        self._current_training_step = 0
        self._total_training_steps = 0

    def reset(self, task: Task):
        """Reset the workflow with a new task."""
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def set_training_progress(self, current_step: int, total_steps: int):
        """Set current training progress for dynamic distillation window.

        Args:
            current_step: Current training step (e.g., current epoch or iteration)
            total_steps: Total training steps (e.g., total epochs)
        """
        self._current_training_step = current_step
        self._total_training_steps = total_steps

    def _compute_distill_window(self) -> int:
        """Compute how many previous steps to distill based on current training progress.

        Returns:
            Number of previous steps to distill (starting from step 1)
        """
        current_step = self._current_training_step
        # Calculate distillation window: starts at 1, increases by 1 every decay_step
        distill_window = 1 + (current_step // self.decay_step)
        return distill_window

    def compute_reward(self, response: Experience) -> float:
        """Return episode-level reward (same for all turns in the trajectory)."""
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        """Format initial messages for the episode."""
        return []

    async def run_async(self) -> List[Experience]:
        # Extract current training step from task batch_id
        current_step = 0
        if hasattr(self.task, 'batch_id'):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                import re
                match = re.match(r'^(\d+)', batch_id)
                if match:
                    current_step = int(match.group(1))

        total_steps = self.task.workflow_args.get('total_steps', 200)
        self.set_training_progress(current_step, total_steps)

        game_file_path = self.task_desc
        env = _create_alfworld_env(game_file_path)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        # Compute distillation window for this episode
        distill_window = self._compute_distill_window()

        # Eval should run normal student rollout up to max_env_steps without distillation.
        if self.is_eval:
            effective_steps = self.max_env_steps
        else:
            # Student only executes distill_window steps (the steps teacher will distill)
            # But cap it at max_env_steps to avoid exceeding environment limits
            effective_steps = min(distill_window, self.max_env_steps)

        for r in range(effective_steps):
            format_obs = format_observation(observation)
            admissible_commands = info.get("admissible_commands", [])
            if admissible_commands and isinstance(admissible_commands[0], list):
                admissible_commands = admissible_commands[0]
            reformatted_admissible = "\n ".join(
                f"'{s}'" for s in admissible_commands if s != "help"
            )

            if len(history) < HISTORY_LENGTH:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )
            else:
                action_history_str = "\n".join(
                    history[-HISTORY_LENGTH:]
                    if len(history) >= HISTORY_LENGTH
                    else history
                )
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )

            memory = memory + [{"role": "user", "content": user_content}]

            # Student samples this turn independently
            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "BiOpdAlfworldWorkflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)

            # Track if environment is done during distillation window
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = reward
                break
        else:
            # Completed effective_steps without finishing
            self._env_rounds = effective_steps
            self._final_reward = 0.0

        # Teacher distillation: distill all executed steps
        per_turn_kl_sums: List[float] = []

        for i, response in enumerate(turn_responses):
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )

            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs

            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )

            response.teacher_logprobs = teacher_resp_logprobs
            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

            if response.metrics is None:
                response.metrics = {}
            compute_reward_fn = getattr(self, "compute_reward", None)
            response.reward = (
                compute_reward_fn(response)
                if callable(compute_reward_fn)
                else getattr(self, "_final_reward", 0.0)
            )
            response.eid.run = self.run_id_base
            response.eid.step = i

        # Trajectory-level metrics
        trajectory_kl_divergence = sum(per_turn_kl_sums)

        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
            last_response.metrics["distill_window"] = distill_window
            last_response.metrics["effective_steps"] = effective_steps
            last_response.metrics["total_steps"] = len(turn_responses)

        return turn_responses
