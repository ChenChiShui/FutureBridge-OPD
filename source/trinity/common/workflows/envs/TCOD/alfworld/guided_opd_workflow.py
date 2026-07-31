"""Guided-OPD workflow for AlfWorld.

Paper: Guided On-Policy Distillation
- Each turn, independently sample role: teacher (prob β_s) or student (prob 1-β_s)
- β_s follows cosine decay from β_start=1.0 to β_end=0.0 over first ρ*total_steps
- Teacher turn: SFT CE loss on teacher-generated tokens (expert_mask=True)
- Student turn: OPD reverse-KL loss (expert_mask=False)
- Shared environment trajectory: whoever acts, the response is appended to history

Uses MIXPolicyLossFn with expert_mask to route teacher turns → SFT, student turns → PPO+OPD.
"""

import math
import random
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from trinity.common.experience import CustomField, Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
    HISTORY_LENGTH,
    parse_action,
    format_observation,
    _extract_task,
    _format_history,
    _create_alfworld_env,
)


def guided_teacher_prob(
    step: int,
    total_steps: int = 250,
    curriculum_ratio: float = 0.8,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
) -> float:
    """Cosine decay schedule for teacher probability."""
    decay_steps = max(1, int(total_steps * curriculum_ratio))
    progress = min(step / decay_steps, 1.0)
    cosine_progress = (1.0 - math.cos(math.pi * progress)) / 2.0
    beta = beta_start + (beta_end - beta_start) * cosine_progress
    return float(beta)


@WORKFLOWS.register_module("guided_opd_alfworld_workflow")
class GuidedOPDAlfworldWorkflow(Workflow):
    """Guided-OPD workflow for AlfWorld.

    Per-turn independent Bernoulli sampling: teacher with prob β_s, student with prob 1-β_s.
    Teacher turns → expert_mask=True → SFT loss.
    Student turns → expert_mask=False → OPD loss (PPO surrogate with OPD advantage).

    Required yaml config:
      algorithm.policy_loss_fn: mix
      algorithm.advantage_fn: multi_turn_opd
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    _GUIDED_CUSTOM_FIELDS = [
        CustomField("is_expert", "expert_mask", torch.bool),
    ]

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
        ), "Guided-OPD requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.is_eval = task.is_eval

        self.beta_start = task.workflow_args.get("beta_start", 1.0)
        self.beta_end = task.workflow_args.get("beta_end", 0.0)
        self.curriculum_ratio = task.workflow_args.get("curriculum_ratio", 0.8)
        self.total_steps = task.workflow_args.get("total_steps", 250)

        self._s2t_map: Optional[Dict[int, int]] = None
        self._student_tokenizer = None
        self._teacher_tokenizer = None

    @property
    def s2t_map(self) -> Dict[int, int]:
        if self._s2t_map is None:
            from trinity.common.workflows.envs.TCOD.alfworld.cross_tokenizer_opd import (
                build_exact_token_map,
            )
            self._s2t_map = build_exact_token_map(self.student_tokenizer, self.teacher_tokenizer)
        return self._s2t_map

    @property
    def student_tokenizer(self):
        if self._student_tokenizer is None:
            from transformers import AutoTokenizer
            model_path = self.model.config.model_path if hasattr(self.model, 'config') else None
            self._student_tokenizer = AutoTokenizer.from_pretrained(
                model_path or "Qwen/Qwen3-1.7B", trust_remote_code=True
            )
        return self._student_tokenizer

    @property
    def teacher_tokenizer(self):
        if self._teacher_tokenizer is None:
            from transformers import AutoTokenizer
            model_path = self.teacher_model.config.model_path if hasattr(self.teacher_model, 'config') else None
            self._teacher_tokenizer = AutoTokenizer.from_pretrained(
                model_path or "meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True
            )
        return self._teacher_tokenizer

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
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
        self._final_reward = 0.0

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_experiences: List[Experience] = []
        turn_messages: List[List[dict]] = []
        teacher_turn_flags: List[bool] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        if self.is_eval:
            beta = 0.0
        else:
            global_step = 0
            bid = getattr(self.task, 'batch_id', None)
            if hasattr(self.task, 'batch_id'):
                batch_id = self.task.batch_id
                if isinstance(batch_id, int):
                    global_step = batch_id
                elif isinstance(batch_id, str):
                    import re as _re
                    m = _re.match(r'^(\d+)', batch_id)
                    if m:
                        global_step = int(m.group(1))
            beta = guided_teacher_prob(
                step=global_step,
                total_steps=self.total_steps,
                curriculum_ratio=self.curriculum_ratio,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )

        n_teacher_turns = 0
        n_student_turns = 0

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

            MAX_MEMORY_TURNS = 10
            recent_memory = memory[-(MAX_MEMORY_TURNS * 2):]

            use_teacher = (not self.is_eval) and (random.random() < beta)

            if use_teacher:
                teacher_kwargs = {"temperature": 1.0, "max_tokens": 512, "n": 1}
                teacher_responses = await self.teacher_model.chat_async(
                    recent_memory, **teacher_kwargs
                )
                teacher_response = teacher_responses[0]
                response_text = teacher_response.response_text or ""

                target_messages = recent_memory + [
                    {"role": "assistant", "content": response_text}
                ]
                try:
                    target_exp = await self.model.convert_messages_to_experience_async(
                        messages=target_messages,
                        temperature=self.temperature,
                    )
                except Exception:
                    action = parse_action(response_text)
                    history.append(_format_history(format_obs, r + 1, action))
                    memory.append({"role": "assistant", "content": response_text})
                    observation, reward, done, info = env.step(action)
                    if done:
                        self._env_done = True
                        self._env_rounds = r + 1
                        self._final_reward = 1.0
                        break
                    continue

                prompt_only_messages = recent_memory
                try:
                    prompt_only_exp = await self.model.convert_messages_to_experience_async(
                        messages=prompt_only_messages,
                        temperature=self.temperature,
                    )
                    target_start = len(prompt_only_exp.tokens) - target_exp.prompt_length
                    if target_start > 0 and target_exp.action_mask is not None:
                        target_exp.action_mask[:target_start] = 0
                except Exception:
                    pass

                if target_exp.truncate_status == "prompt_truncated":
                    action = parse_action(response_text)
                    history.append(_format_history(format_obs, r + 1, action))
                    memory.append({"role": "assistant", "content": response_text})
                    observation, reward, done, info = env.step(action)
                    if done:
                        self._env_done = True
                        self._env_rounds = r + 1
                        self._final_reward = 1.0
                        break
                    continue

                if target_exp.logprobs is not None:
                    target_exp.teacher_logprobs = torch.zeros_like(target_exp.logprobs)
                else:
                    target_exp.teacher_logprobs = torch.zeros(1, dtype=torch.float32)

                target_exp.info["is_expert"] = True
                target_exp.custom_fields = list(self._GUIDED_CUSTOM_FIELDS)

                target_exp.eid.run = getattr(self, "run_id_base", 0)
                target_exp.eid.step = len(turn_experiences)
                if target_exp.metrics is None:
                    target_exp.metrics = {}
                target_exp.metrics["is_teacher_turn"] = 1.0

                turn_messages.append(list(recent_memory))
                turn_experiences.append(target_exp)
                teacher_turn_flags.append(True)
                n_teacher_turns += 1

                action = parse_action(response_text)
            else:
                responses = await self.model.chat_async(recent_memory, **kwargs)
                response = responses[0]
                response_text = response.response_text or ""

                if response.logprobs is None:
                    raise RuntimeError(
                        "GuidedOPD requires student model to return logprobs. "
                        "Set rollout_args.logprobs in task config."
                    )

                response.info["is_expert"] = False
                response.custom_fields = list(self._GUIDED_CUSTOM_FIELDS)

                response.eid.run = getattr(self, "run_id_base", 0)
                response.eid.step = len(turn_experiences)
                if response.metrics is None:
                    response.metrics = {}
                response.metrics["is_teacher_turn"] = 0.0

                turn_messages.append(list(recent_memory))
                turn_experiences.append(response)
                teacher_turn_flags.append(False)
                n_student_turns += 1

                action = parse_action(response_text)

            memory.append({"role": "assistant", "content": response_text})
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0

        for i, exp in enumerate(turn_experiences):
            if teacher_turn_flags[i]:
                continue

            messages_i = turn_messages[i]

            if not messages_i:
                if exp.logprobs is not None:
                    exp.teacher_logprobs = torch.zeros_like(exp.logprobs)
                else:
                    exp.teacher_logprobs = torch.zeros(1, dtype=torch.float32)
                if exp.metrics is None:
                    exp.metrics = {}
                exp.metrics["opd_aligned"] = 0
                exp.metrics["opd_total"] = 0
                exp.metrics["opd_ratio"] = 0.0
                continue

            teacher_exp = await self.teacher_model.convert_messages_to_experience_async(
                messages=messages_i,
                temperature=self.temperature,
            )
            teacher_tokens = teacher_exp.tokens.tolist()

            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=teacher_tokens,
                temperature=self.temperature,
            )

            from trinity.common.workflows.envs.TCOD.alfworld.cross_tokenizer_opd import (
                compute_shared_support_opd,
            )
            student_tokens = exp.tokens.tolist()
            student_prompt_length = exp.prompt_length
            student_resp_logprobs = exp.logprobs

            teacher_resp_logprobs, n_aligned, n_total = compute_shared_support_opd(
                student_tokens=student_tokens,
                student_prompt_length=student_prompt_length,
                student_logprobs=student_resp_logprobs,
                teacher_tokens=teacher_tokens,
                teacher_prompt_length=teacher_exp.prompt_length,
                teacher_logprobs=teacher_logprobs,
                student_tok=self.student_tokenizer,
                teacher_tok=self.teacher_tokenizer,
                s2t_map=self.s2t_map,
            )

            exp.teacher_logprobs = teacher_resp_logprobs
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["opd_aligned"] = n_aligned
            exp.metrics["opd_total"] = n_total
            exp.metrics["opd_ratio"] = n_aligned / max(n_total, 1)

        for exp in turn_experiences:
            exp.reward = self._final_reward

        if turn_experiences:
            last_exp = turn_experiences[-1]
            if last_exp.metrics is None:
                last_exp.metrics = {}
            last_exp.metrics["env_rounds"] = self._env_rounds
            last_exp.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_exp.metrics["guided_beta"] = beta
            last_exp.metrics["teacher_turns"] = float(n_teacher_turns)
            last_exp.metrics["student_turns"] = float(n_student_turns)
            last_exp.metrics["teacher_turn_ratio"] = (
                n_teacher_turns / max(n_teacher_turns + n_student_turns, 1)
            )

        return turn_experiences
