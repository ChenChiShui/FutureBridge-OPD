# -*- coding: utf-8 -*-
"""SDPO (Self-Distilled Policy Optimization) workflows.

This workflow implements a practical on-policy self-distillation loop:
1. Student samples responses on the original prompt.
2. Optionally compute reward for each response.
3. Build reprompted teacher context with feedback + successful examples.
4. Re-score the same student response tokens under the reprompted context.
5. Store `teacher_logprobs` in each experience for trainer-side advantage/loss.
"""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.reward_fn import RewardFn
from trinity.common.workflows.workflow import Task, Workflow


class SDPOWorkflow(Workflow):
    """General SDPO workflow for single-turn tasks."""

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
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset(task)

        # SDPO default: self as teacher; user may also pass one teacher model explicitly.
        self.teacher_model = (
            self.auxiliary_model_wrappers[0]
            if self.auxiliary_model_wrappers
            else self.model
        )

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.success_reward_threshold = task.workflow_args.get(
            "success_reward_threshold", 1.0
        )
        self.max_success_examples = int(task.workflow_args.get("max_success_examples", 3))
        self.dont_reprompt_on_self_success = bool(
            task.workflow_args.get("dont_reprompt_on_self_success", True)
        )
        self.feedback_key = task.workflow_args.get("feedback_key", "feedback")
        self.reprompt_template = task.workflow_args.get(
            "reprompt_template",
            "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.",
        )
        self.solution_template = task.workflow_args.get(
            "solution_template",
            "\n\nCorrect solution:\n\n{successful_previous_attempt}",
        )
        self.feedback_template = task.workflow_args.get(
            "feedback_template",
            "\n\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}",
        )

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.reply_prefix = task.format_args.reply_prefix
        self.raw_task = task.raw_task or {}
        self.task_desc = task.task_desc
        self.truth = task.truth

        reward_fn = task.reward_fn
        self.reward_fn: Optional[RewardFn] = None
        if isinstance(reward_fn, type) and issubclass(reward_fn, RewardFn):
            self.reward_fn = reward_fn(**task.reward_fn_args)

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self) -> List[dict]:
        messages = []
        messages.append({"role": "user", "content": self.task_desc})
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        return messages

    def compute_reward(self, response: Experience) -> float:
        """Compute reward for one sampled response.

        Default behavior:
        - If no reward_fn is configured, return 0.0
        - If reward_fn returns dict metrics, sum numeric values as scalar reward
        """
        if self.reward_fn is None or response.response_text is None:
            return 0.0

        reward_out = self.reward_fn(response=response.response_text, truth=self.truth)  # type: ignore[misc]
        if response.metrics is None:
            response.metrics = {}

        if isinstance(reward_out, dict):
            numeric_metrics = {
                k: float(v)
                for k, v in reward_out.items()
                if isinstance(v, (float, int))
            }
            response.metrics.update(numeric_metrics)
            return sum(numeric_metrics.values())

        if isinstance(reward_out, (float, int)):
            return float(reward_out)
        return 0.0

    def _build_teacher_messages(
        self,
        current_response: Experience,
        successful_examples: List[Experience],
    ) -> List[dict]:
        """Build reprompted teacher context for SDPO scoring."""
        student_prompt = self.task_desc or ""
        feedback_raw = self.raw_task.get(self.feedback_key)
        feedback_block = ""
        if feedback_raw:
            feedback_block = self.feedback_template.format(feedback_raw=str(feedback_raw))

        sampled_successes = successful_examples[: self.max_success_examples]
        if self.dont_reprompt_on_self_success and current_response in sampled_successes:
            sampled_successes = [exp for exp in sampled_successes if exp is not current_response]

        successful_attempts = [
            exp.response_text for exp in sampled_successes if exp.response_text
        ]
        solution_block = ""
        if successful_attempts:
            solution_block = self.solution_template.format(
                successful_previous_attempt="\n\n".join(successful_attempts)
            )

        reprompted_user_content = self.reprompt_template.format(
            prompt=student_prompt,
            solution=solution_block,
            feedback=feedback_block,
        )

        return [{"role": "user", "content": reprompted_user_content}]

    def _reprompt_used_success_count(
        self,
        current_response: Experience,
        successful_examples: List[Experience],
    ) -> int:
        sampled_successes = successful_examples[: self.max_success_examples]
        if self.dont_reprompt_on_self_success and current_response in sampled_successes:
            sampled_successes = [exp for exp in sampled_successes if exp is not current_response]
        return sum(1 for exp in sampled_successes if exp.response_text)

    async def run_async(self) -> List[Experience]:
        messages = self.format_messages()
        responses = await self.model.chat_async(messages, **self.rollout_args)
        if not responses:
            return responses

        for i, response in enumerate(responses):
            if response.logprobs is None:
                raise RuntimeError(
                    "SDPOWorkflow requires student logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = i + self.run_id_base

        successful_examples = [
            exp for exp in responses if (exp.reward or 0.0) >= self.success_reward_threshold
        ]

        for response in responses:
            is_current_success = (response.reward or 0.0) >= self.success_reward_threshold
            if is_current_success:
                teacher_messages = self._build_teacher_messages(
                    current_response=response,
                    successful_examples=successful_examples,
                )
                used_success_count = self._reprompt_used_success_count(
                    response, successful_examples
                )
            else:
                # For incorrect student responses, skip SDPO reprompt and score on original prompt.
                teacher_messages = self.format_messages()
                used_success_count = 0
            teacher_prompt_exp = await self.teacher_model.convert_messages_to_experience_async(
                teacher_messages,
                temperature=self.temperature,
            )
            teacher_prompt_tokens = teacher_prompt_exp.tokens.tolist()
            student_response_tokens = response.tokens[response.prompt_length :].tolist()
            full_teacher_tokens = teacher_prompt_tokens + student_response_tokens

            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=full_teacher_tokens,
                temperature=self.temperature,
            )
            teacher_resp_logprobs = teacher_logprobs[len(teacher_prompt_tokens) - 1 :]

            if len(teacher_resp_logprobs) != len(response.logprobs):
                raise RuntimeError(
                    f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                    f"student_logprobs={len(response.logprobs)}. "
                    f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
                )

            response.teacher_logprobs = teacher_resp_logprobs
            response.metrics["kl_divergence"] = (
                response.logprobs - teacher_resp_logprobs
            ).sum().item()
            # Split KL by whether teacher reprompt contains successful attempts.
            has_success_attempt = used_success_count > 0
            response.metrics["sdpo_teacher_reprompt_applied"] = (
                1.0 if is_current_success else 0.0
            )
            response.metrics["sdpo_reprompt_has_success_attempt"] = (
                1.0 if has_success_attempt else 0.0
            )
            if has_success_attempt:
                response.metrics["kl_divergence_with_success_attempt"] = response.metrics[
                    "kl_divergence"
                ]
            else:
                response.metrics["kl_divergence_without_success_attempt"] = response.metrics[
                    "kl_divergence"
                ]
            response.metrics["sdpo_success_pool_size"] = float(len(successful_examples))

        return responses



class SDPOSCIENQAWorkflow(SDPOWorkflow):
    """SDPO workflow for SciKnowEval dataset (MCQ format).

    This workflow:
    - Uses MCQ format with <reasoning> and <answer> tags
    - Parses answer from <answer> tags and compares with ground truth
    - Computes batch accuracy and identifies successful attempts
    - Tracks category-wise accuracy (biology, chemistry, material, physics)
    """

    def format_messages(self) -> List[dict]:
        # Use system prompt from raw_task if available, otherwise use default
        system_prompt = self.raw_task.get("system", "")
        if not system_prompt:
            system_prompt = """Given a question and four options, please select the right answer. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else. Do not restate the answer text. For example, if the answer is "A", just output:
<answer>
A
</answer>"""

        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.task_desc})
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        return messages

    def compute_reward(self, response: Experience) -> float:
        """Parse MCQ answer and compute accuracy."""
        if not response.response_text or not self.truth:
            return 0.0

        import re
        answer_pattern = r'<answer>\s*([A-D])\s*</answer>'
        match = re.search(answer_pattern, response.response_text, re.IGNORECASE)

        parsed_answer = match.group(1).upper() if match else None
        ground_truth = str(self.truth).strip().upper()

        is_correct = parsed_answer == ground_truth if parsed_answer else False

        if response.metrics is None:
            response.metrics = {}
        response.metrics["accuracy"] = 1.0 if is_correct else 0.0
        response.metrics["parsed_answer"] = parsed_answer
        response.metrics["ground_truth"] = ground_truth

        # Track category-wise accuracy using 'domain' field from dataset
        category = self.raw_task.get("domain", self.raw_task.get("dataset", "unknown"))
        response.metrics[f"accuracy_{category}"] = 1.0 if is_correct else 0.0

        return 1.0 if is_correct else 0.0

    async def run_async(self) -> List[Experience]:
        """Run workflow with eval mode support (no reprompting during eval)."""
        messages = self.format_messages()
        responses = await self.model.chat_async(messages, **self.rollout_args)
        if not responses:
            return responses

        # Check if this is eval mode (no teacher reprompting needed)
        is_eval = self.task.rollout_args.temperature < 0.5

        for i, response in enumerate(responses):
            if response.logprobs is None:
                raise RuntimeError(
                    "SDPOWorkflow requires student logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = i + self.run_id_base

        # Skip teacher reprompting during eval
        if is_eval:
            return responses

        successful_examples = [
            exp for exp in responses if (exp.reward or 0.0) >= self.success_reward_threshold
        ]

        for response in responses:
            is_current_success = (response.reward or 0.0) >= self.success_reward_threshold
            if is_current_success:
                teacher_messages = self._build_teacher_messages(
                    current_response=response,
                    successful_examples=successful_examples,
                )
                used_success_count = self._reprompt_used_success_count(
                    response, successful_examples
                )
            else:
                # For incorrect student responses, skip SDPO reprompt and score on original prompt.
                teacher_messages = self.format_messages()
                used_success_count = 0
            teacher_prompt_exp = await self.teacher_model.convert_messages_to_experience_async(
                teacher_messages,
                temperature=self.temperature,
            )
            teacher_prompt_tokens = teacher_prompt_exp.tokens.tolist()
            student_response_tokens = response.tokens[response.prompt_length :].tolist()
            full_teacher_tokens = teacher_prompt_tokens + student_response_tokens

            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=full_teacher_tokens,
                temperature=self.temperature,
            )
            teacher_resp_logprobs = teacher_logprobs[len(teacher_prompt_tokens) - 1 :]

            if len(teacher_resp_logprobs) != len(response.logprobs):
                raise RuntimeError(
                    f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                    f"student_logprobs={len(response.logprobs)}. "
                    f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
                )

            response.teacher_logprobs = teacher_resp_logprobs
            response.metrics["kl_divergence"] = (
                response.logprobs - teacher_resp_logprobs
            ).sum().item()
            # Split KL by whether teacher reprompt contains successful attempts.
            has_success_attempt = used_success_count > 0
            response.metrics["sdpo_teacher_reprompt_applied"] = (
                1.0 if is_current_success else 0.0
            )
            response.metrics["sdpo_reprompt_has_success_attempt"] = (
                1.0 if has_success_attempt else 0.0
            )
            if has_success_attempt:
                response.metrics["kl_divergence_with_success_attempt"] = response.metrics[
                    "kl_divergence"
                ]
            else:
                response.metrics["kl_divergence_without_success_attempt"] = response.metrics[
                    "kl_divergence"
                ]
            response.metrics["sdpo_success_pool_size"] = float(len(successful_examples))

        return responses
