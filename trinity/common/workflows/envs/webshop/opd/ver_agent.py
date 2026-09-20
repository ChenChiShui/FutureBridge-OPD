# -*- coding: utf-8 -*-
"""On-Policy Distillation (OPD) workflow for WebShop.

Algorithm:
1. Student interacts with WebShop step-by-step and samples actions with logprobs.
2. Each turn keeps the exact prefix used by the student.
3. Teacher computes logprobs on the same full sequence (prefix + student response).
4. Store `teacher_logprobs` in each turn's experience.
5. Return one `Experience` per turn for multi-turn OPD training.
"""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow

from trinity.common.workflows.envs.webshop.prompts import (
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
)
from trinity.common.workflows.envs.webshop.webshop_workflow import (
    parse_action,
    validate_action,
)

SPARSE_REWARD = False
HISTORY_LENGTH = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def format_observation(observation: str) -> str:
    return observation.strip()


def _compact_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_task_description(observation: str) -> str:
    """Extract the instruction text from a WebShop observation."""
    lines = [line.strip() for line in observation.splitlines()]
    try:
        instruction_idx = lines.index("Instruction:")
    except ValueError:
        return _compact_text(observation)

    task_lines = []
    for line in lines[instruction_idx + 1 :]:
        if not line:
            continue
        if line.startswith("[button]"):
            break
        task_lines.append(line)

    return " ".join(task_lines).strip() or _compact_text(observation)


def _format_history(observation: str, step_num: int, action: str) -> str:
    return MEMORY_FORMAT.format(
        step_num=step_num,
        obs=_compact_text(observation),
        act=action,
    )


def _format_available_actions(available_actions: dict) -> str:
    formatted_actions = []
    if available_actions.get("has_search_bar", False):
        formatted_actions.append("search[<query>]")
    for clickable in available_actions.get("clickables", []):
        formatted_actions.append(f"click[{clickable}]")
    return "\n".join(formatted_actions)


def _create_webshop_env():
    try:
        import gym  # type: ignore[import-not-found]
        from web_agent_site.envs import WebAgentTextEnv  # type: ignore[import-not-found]  # noqa: F401
    except Exception as e:
        error_message = (
            f"Error importing WebAgentTextEnv {str(e)}. "
            "Please make sure you have installed the web_agent_site package, "
            "following the instructions in "
        )
        raise ImportError(error_message) from e

    return gym.make(
        "WebAgentTextEnv-v0",
        observation_mode="text_rich",
        num_products=None,
        human_goals=True,
    )


class OnPolicyDistillVerlAgentWebshopWorkflow(Workflow):
    """On-policy distillation workflow for WebShop."""

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
        self.max_env_steps = task.workflow_args.get("max_env_steps", 15)
        self.env = _create_webshop_env()

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval
        self.repeat_times = task.repeat_times or 1

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def compute_reward(self, response: Experience) -> float:
        return getattr(self, "_final_reward", -0.1)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
        session_id = int(self.task_desc)
        all_turn_responses: List[Experience] = []

        for rollout_idx in range(self.repeat_times):
            rollout_responses = await self._run_episode(
                session_id=session_id,
                run_id=self.run_id_base + rollout_idx,
            )
            all_turn_responses.extend(rollout_responses)

        return all_turn_responses

    async def _run_episode(self, session_id: int, run_id: int) -> List[Experience]:
        self.env.reset(session=session_id)
        observation = self.env.observation
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = -0.1

        task_description = _extract_task_description(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []
        done = False

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            available_actions = self.env.get_available_actions()
            formatted_observation = format_observation(observation)
            formatted_actions = _format_available_actions(available_actions)

            if len(history) < HISTORY_LENGTH:
                user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_description,
                    current_observation=formatted_observation,
                    available_actions=formatted_actions,
                )
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = WEBSHOP_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=formatted_observation,
                    available_actions=formatted_actions,
                )

            memory = memory + [{"role": "user", "content": user_content}]

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "OnPolicyDistillVerlAgentWebshopWorkflow requires student model "
                    "to return logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            action_valid, error_msg = validate_action(action, available_actions)

            history.append(_format_history(formatted_observation, r + 1, action))

            if action_valid:
                observation, reward, done, _ = self.env.step(action)
            else:
                observation = error_msg
                reward = 0.0
                done = False

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = reward
                break
        else:
            self._env_rounds = self.max_env_steps

        if SPARSE_REWARD:
            if self._final_reward >= 0.99:
                self._final_reward = 1
            elif self._final_reward >= 0:
                self._final_reward = 0
            else:
                self._final_reward = -0.1

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
            response.reward = self.compute_reward(response)
            response.eid.run = run_id
            response.eid.step = i

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        trajectory_kl_divergence = sum(per_turn_kl_sums)
        for response in turn_responses:
            response.metrics["env_rounds"] = self._env_rounds
            response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            response.metrics["kl_divergence"] = trajectory_kl_divergence
            response.metrics["session_id"] = float(session_id)

        return turn_responses
