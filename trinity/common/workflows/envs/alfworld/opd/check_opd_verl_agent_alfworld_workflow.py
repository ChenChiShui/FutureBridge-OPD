# -*- coding: utf-8 -*-
"""On-Policy Distillation (OPD) workflow for AlfWorld.

Reference: OnPolicyDistillWorkflow logic; adapted for multi-turn AlfWorld.

Algorithm:
1. Student pre-samples trajectory (runs episode turn-by-turn with logprobs)
2. Split by turns: each turn has fixed prefix [system, obs_1, resp_1, ..., obs_t]
3. Teacher computes logprobs on same (prefix + response) per turn
4. Store teacher_logprobs in experience; advantage_fn uses teacher_logprobs - student_logprobs
5. Return one Experience per turn (like OPD returning one per sample)
"""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow

# from trinity.common.workflows.envs.alfworld.opd.prompts import (
#     ALFWORLD_TEMPLATE_NO_HIS,
#     ALFWORLD_TEMPLATE,
# )
from trinity.common.workflows.envs.alfworld.opd.prompts3 import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
)
from trinity.common.workflows.envs.alfworld.alfworld_workflow import (
    # format_observation,
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


def _create_alfworld_env_with_checkpoint(game_file_path: str, actions: List[str], checkpoint_step: int):
    """Create AlfWorld environment and execute actions up to checkpoint_step.

    Args:
        game_file_path: Path to the game file
        actions: List of predefined expert actions
        checkpoint_step: Number of actions to execute before letting model take over

    Returns:
        tuple: (env, observation, info, history, task_description, current_step)
    """
    env = _create_alfworld_env(game_file_path)
    observation, info = env.reset()

    task_description = _extract_task(observation)
    history: List[str] = []

    # Execute predefined actions up to checkpoint_step
    for step in range(min(checkpoint_step, len(actions))):
        action = actions[step]
        format_obs = format_observation(observation)
        history.append(_format_history(format_obs, step + 1, action))
        observation, reward, done, info = env.step(action)

        if done:
            # If task completes before checkpoint, return the final state
            break

    return env, observation, info, history, task_description, len(history)


class OnPolicyDistillVerlAgentAlfworldWorkflow(Workflow):
    """On-policy distillation workflow for AlfWorld.

    Computes and stores teacher_logprobs in each turn's experience.
    The advantage_fn in trainer will compute:
        advantages = teacher_logprobs - student_logprobs

    Use advantage_fn: multi_turn_opd (MultiTurnOpdAdvantage) for this workflow,
    since it returns List[Experience] (one per turn), not a single response.

    Logic aligned with OnPolicyDistillWorkflow:
    - Student samples (with logprobs); teacher computes logprobs on same sequences.
    - Per-turn split: prefix fixed per turn, one Experience per turn.
    - compute_reward() can be overridden by subclasses (default: episode final reward).
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

        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.checkpoint_step = task.workflow_args.get("checkpoint_step", None)
        self.is_eval = task.is_eval

        # Naive mode: if True, always use _run_episode without any checkpoint (baseline)
        self.naive = task.workflow_args.get("naive", False)

        # Checkpoint strategy configuration
        self.checkpoint_strategy = task.workflow_args.get("checkpoint_strategy", None)
        if self.checkpoint_strategy == "linear_decay":
            # checkpoint_decay_steps: number of training steps before reducing expert actions by 1
            # e.g., 5 means every 5 training steps, expert actions decrease by 1
            # Training step 0-4: expert_actions = len(actions) - 1
            # Training step 5-9: expert_actions = len(actions) - 2
            # Training step 10-14: expert_actions = len(actions) - 3
            # ... until expert_actions = 0
            self.checkpoint_decay_steps = task.workflow_args.get(
                "checkpoint_decay_steps", 5
            )

        # Track current training step for dynamic checkpoint
        self._current_training_step = 0
        self._total_training_steps = 0

    def reset(self, task: Task):
        """Reset the workflow with a new task.

        Unlike BaseSimpleWorkflow, this does NOT require reward_fn.
        """
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval
        self.naive = task.workflow_args.get("naive", False)

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def set_training_progress(self, current_step: int, total_steps: int):
        """Set current training progress for dynamic checkpoint strategy.

        Args:
            current_step: Current training step (e.g., current epoch or iteration)
            total_steps: Total training steps (e.g., total epochs)
        """
        self._current_training_step = current_step
        self._total_training_steps = total_steps

    def _compute_checkpoint_step(self, predefined_actions: Optional[List[str]] = None) -> Optional[int]:
        """Compute checkpoint step based on strategy.

        Args:
            predefined_actions: List of predefined actions (to get max possible checkpoint)

        Returns:
            Checkpoint step to use, or None if no checkpoint should be used
        """
        # If no strategy is set, use the fixed checkpoint_step
        if self.checkpoint_strategy is None:
            return 0

        elif self.checkpoint_strategy == "linear_decay":
            if predefined_actions is None or len(predefined_actions) == 0:
                return 0

            current_step = self._current_training_step

            # Calculate how many expert actions to execute
            # Start: len(actions) - 1, decrease by 1 every checkpoint_decay_steps
            max_expert_actions = len(predefined_actions) - 1
            reduction = current_step // self.checkpoint_decay_steps
            checkpoint_step = max_expert_actions - reduction

            # Clamp to valid range [0, len(actions)-1]
            # When checkpoint_step <= 0, return 0 (no expert actions, model starts from beginning)
            checkpoint_step = max(0, min(checkpoint_step, len(predefined_actions) - 1))

            return checkpoint_step
        else:
            return 0

    def compute_reward(self, response: Experience) -> float:
        """Return episode-level reward (same for all turns in the trajectory).

        Set in _run_episode: env reward when done, 0.0 when max steps exhausted.
        """
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        """Format initial messages for the episode.

        Uses ALFWORLD_TEMPLATE_NO_HIS / ALFWORLD_TEMPLATE from prompts.py.
        No system prompt; each user message is self-contained.
        """
        return []

    async def run_async(self) -> List[Experience]:
        # If naive mode is enabled, always use normal _run_episode (baseline, no checkpoint)
        # If this is an eval task, always use normal _run_episode (no checkpoint)
        if self.naive or self.is_eval:
            # Set _used_checkpoint_step to 0 for metrics tracking
            self._used_checkpoint_step = 0
            env = _create_alfworld_env(self.task_desc)
            try:
                return await self._run_episode(env)
            finally:
                env.close()

        # Extract current training step from task batch_id
        current_step = 0
        if hasattr(self.task, 'batch_id'):
            batch_id = self.task.batch_id
            if isinstance(batch_id, int):
                current_step = batch_id
            elif isinstance(batch_id, str):
                # Extract the first integer from the string (e.g., "5/eval_task" -> 5)
                import re
                match = re.match(r'^(\d+)', batch_id)
                if match:
                    current_step = int(match.group(1))

        # Get total_steps from workflow_args if provided, otherwise use a reasonable default
        # Note: You can add "total_steps: 200" to workflow_args in your config file
        # to explicitly set this value
        total_steps = self.task.workflow_args.get('total_steps', 200)

        # Update training progress for dynamic checkpoint strategy
        # Always update to get the latest training step from batch_id
        self.set_training_progress(current_step, total_steps)

        game_file_path = self.task_desc
        predefined_actions = self.raw_task.get("actions", None)

        # Compute checkpoint step dynamically based on strategy
        effective_checkpoint_step = self._compute_checkpoint_step(predefined_actions)

        # Store for metrics
        self._used_checkpoint_step = effective_checkpoint_step if effective_checkpoint_step is not None else 0

        # Check if we should use checkpoint mode
        # Only use checkpoint if effective_checkpoint_step > 0 and predefined_actions exist
        if effective_checkpoint_step is not None and effective_checkpoint_step > 0 and predefined_actions is not None:
            env, observation, info, history, task_description, current_step = (
                _create_alfworld_env_with_checkpoint(
                    game_file_path, predefined_actions, effective_checkpoint_step
                )
            )
            try:
                return await self._run_episode_from_checkpoint(
                    env, observation, info, history, task_description, current_step
                )
            finally:
                env.close()
        else:
            # Original mode: start from beginning (no checkpoint or checkpoint_step == 0)
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

        for r in range(self.max_env_steps):
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

            # Step 1: Student samples this turn (same pattern as OnPolicyDistillWorkflow)
            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "OnPolicyDistillAlfworldWorkflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = reward
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0  # failure: exhausted max steps

        # Step 2 & 3: Teacher logprobs and fill experience (mirror OnPolicyDistillWorkflow.run_async)
        # response.tokens is the full sequence for this turn: [prefix | response], where
        # prefix = system + obs_1 + resp_1 + ... + obs_t (same input as student had).
        per_turn_kl_sums: List[float] = []
        for i, response in enumerate(turn_responses):
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),  # full input = prefix + student's response
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

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        # Trajectory-level metrics (computed once for the whole trajectory)
        trajectory_kl_divergence = sum(per_turn_kl_sums)

        # Calculate detailed checkpoint metrics
        # In naive/eval mode, we don't use expert actions, so set them to 0
        if self.naive or self.is_eval:
            total_expert_actions = 0
            expert_actions_executed = 0
            expert_actions_remaining = 0
            model_executed_steps = self._env_rounds
        else:
            total_expert_actions = len(self.raw_task.get("actions", []))
            expert_actions_executed = self._used_checkpoint_step  # Use the computed checkpoint step
            expert_actions_remaining = total_expert_actions - expert_actions_executed
            model_executed_steps = self._env_rounds - expert_actions_executed

        # Log trajectory-level metrics once per task to avoid duplicated eval aggregation.
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
            # New detailed metrics (more intuitive)
            last_response.metrics["total_expert_actions"] = total_expert_actions
            last_response.metrics["expert_actions_executed"] = expert_actions_executed
            last_response.metrics["expert_actions_remaining"] = expert_actions_remaining
            last_response.metrics["model_executed_steps"] = model_executed_steps

        return turn_responses

    async def _run_episode_from_checkpoint(
        self, env, observation, info, history: List[str], task_description: str, start_step: int
    ) -> List[Experience]:
        """Run episode from a checkpoint with predefined history.

        Args:
            env: AlfWorld environment already at checkpoint state
            observation: Current observation at checkpoint
            info: Current info dict at checkpoint
            history: List of formatted history strings from predefined actions
            task_description: Extracted task description
            start_step: Step number to start from (number of predefined actions executed)

        Returns:
            List of Experience objects for turns after checkpoint
        """
        self._env_done = False
        self._env_rounds = start_step

        memory = self.format_messages()
        turn_responses: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(start_step, self.max_env_steps):
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

            # Student samples this turn
            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "OnPolicyDistillAlfworldWorkflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = reward
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0  # failure: exhausted max steps

        # Teacher logprobs and fill experience
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

            if response.metrics is None:
                response.metrics = {}
            compute_reward_fn = getattr(self, "compute_reward", None)
            response.reward = (
                compute_reward_fn(response)
                if callable(compute_reward_fn)
                else getattr(self, "_final_reward", 0.0)
            )
            response.eid.run = self.run_id_base
            response.eid.step = start_step + i

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        # Trajectory-level metrics
        trajectory_kl_divergence = sum(per_turn_kl_sums)

        # Calculate detailed checkpoint metrics
        total_expert_actions = len(self.raw_task.get("actions", []))
        expert_actions_executed = start_step  # Number of expert actions executed before model takes over
        expert_actions_remaining = total_expert_actions - expert_actions_executed  # Expert actions not executed
        model_executed_steps = self._env_rounds - start_step  # Steps executed by model

        # Log trajectory-level metrics once per task to avoid duplicated eval aggregation.
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence

            # New detailed metrics (more intuitive)
            last_response.metrics["total_expert_actions"] = total_expert_actions
            last_response.metrics["expert_actions_executed"] = expert_actions_executed
            last_response.metrics["expert_actions_remaining"] = expert_actions_remaining
            last_response.metrics["model_executed_steps"] = model_executed_steps

        return turn_responses
