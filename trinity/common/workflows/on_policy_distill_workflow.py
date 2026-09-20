# -*- coding: utf-8 -*-
"""On-Policy Distillation Workflow.

Reference: Tinker library's on-policy distillation implementation.

Algorithm:
1. Student samples trajectories (with logprobs)
2. Teacher computes logprobs on same trajectories
3. Store teacher_logprobs in experience.info["teacher_logprobs"]
4. Trainer's advantage_fn computes: advantages = teacher_logprobs - student_logprobs
5. Train with importance_sampling loss
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import re
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.qwen25_eval import verify_math_answer
from trinity.common.workflows.workflow import Task, Workflow


class OnPolicyDistillWorkflow(Workflow):
    """On-policy distillation workflow.

    Computes and stores teacher_logprobs in experience.info.
    The advantage_fn in trainer will compute:
        advantages = teacher_logprobs - student_logprobs

    Note: This workflow does NOT use reward_fn because:
    - Advantage is computed from teacher-student logprobs difference
    - No external reward signal is needed
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
            self.auxiliary_model_wrappers is not None and len(self.auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)

    def reset(self, task: Task):
        """Reset the workflow with a new task.

        Unlike BaseSimpleWorkflow, this does NOT require reward_fn.
        """
        self.task = task
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        """Format messages for the instruct model.

        Default format: system_prompt (optional) + task_desc + reply_prefix (optional)
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.task_desc})
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        return messages

    def compute_reward(self, response: Experience) -> float:
        """Compute reward for a response.

        In base class, returns 0.0 as advantage is computed from teacher-student logprobs.
        Subclasses can override this to compute actual rewards.
        """
        return 0.0

    async def run_async(self) -> List[Experience]:
        messages = self.format_messages()

        # Step 1: Student samples trajectories
        responses = await self.model.chat_async(messages, **self.rollout_args)

        for i, response in enumerate(responses):
            # Step 2: Teacher computes logprobs
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )

            # Extract response portion
            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs

            # Verify lengths match (they should be equal for the same token sequence)
            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )

            # Step 3: Store teacher_logprobs for advantage_fn
            response.teacher_logprobs = teacher_resp_logprobs

            # Initialize metrics
            if response.metrics is None:
                response.metrics = {}

            # Compute reward (subclasses can override compute_reward)
            response.reward = self.compute_reward(response)

            response.eid.run = i + self.run_id_base

            # KL divergence for monitoring
            # KL = sum(student_logprob - teacher_logprob) over all response tokens
            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            # kl_mean = (student_resp_logprobs - teacher_resp_logprobs).mean().item()
            response.metrics["kl_divergence"] = kl_sum  # Total KL (for backward compatibility)
            # response.metrics["kl_divergence_per_token"] = kl_mean  # Average KL per token (more interpretable)

        return responses




class OnPolicyDistillMathWorkflow(OnPolicyDistillWorkflow):
    """On-policy distillation workflow with Qwen2.5-Math style format.

    This workflow:
    - Uses Qwen2.5-Math style prompt format (same as math_eval_workflow)
    - Computes accuracy using verify_math_answer as reward
    - Suitable for math reasoning tasks like GSM8K, MATH, etc.
    """

    def format_messages(self):
        """Format messages using Qwen2.5-Math style.

        System prompt: "You are a helpful assistant."
        User prompt: "{question}\nPlease reason step by step, and put your final answer within \\boxed{}."
        """
        system_prompt = "You are a helpful assistant."
        user_prompt = f"{self.task_desc}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def compute_reward(self, response: Experience) -> float:
        """Compute accuracy as reward using Qwen2.5-Math evaluation.

        Returns 1.0 if answer is correct, 0.0 otherwise.
        """
        if response.response_text and self.truth:
            accuracy, _ = verify_math_answer(
                response_text=response.response_text, ground_truth=self.truth
            )
            # Store accuracy in metrics
            if response.metrics is None:
                response.metrics = {}
            response.metrics["accuracy"] = accuracy
            return float(accuracy)
        return 0.0


def verify_answer(response_text: str, ground_truth: str) -> Tuple[float, Dict[str, Any]]:
    """Extract answer from <answer>...</answer> tags and verify correctness.
    
    Args:
        response_text: Model response text containing <answer>...</answer> tags
        ground_truth: Ground truth answer (A, B, C, or D)
    
    Returns:
        Tuple of (accuracy: float, details: dict)
        - accuracy: 1.0 if correct, 0.0 otherwise
        - details: Dictionary with parsed_prediction, ground_truth, is_correct
    """
    try:
        # Extract answer from <answer>...</answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if not matches:
            # Try alternative patterns if <answer> tag not found
            # Look for answer after reasoning tag
            reasoning_pattern = r'</reasoning>(.*?)(?:<answer>|$)'
            after_reasoning = re.search(reasoning_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if after_reasoning:
                # Try to extract single letter (A, B, C, D) from the text
                letter_match = re.search(r'\b([ABCD])\b', after_reasoning.group(1), re.IGNORECASE)
                if letter_match:
                    parsed_prediction = letter_match.group(1).upper()
                else:
                    parsed_prediction = None
            else:
                # Last resort: look for A/B/C/D anywhere in the response
                letter_match = re.search(r'\b([ABCD])\b', response_text, re.IGNORECASE)
                parsed_prediction = letter_match.group(1).upper() if letter_match else None
        else:
            # Extract letter from answer tag content
            answer_content = matches[-1].strip()  # Use last match if multiple
            # Extract single letter (A, B, C, D)
            letter_match = re.search(r'\b([ABCD])\b', answer_content, re.IGNORECASE)
            parsed_prediction = letter_match.group(1).upper() if letter_match else None
        
        # Normalize ground truth
        ground_truth_normalized = str(ground_truth).strip().upper()
        
        # Check correctness
        is_correct = False
        if parsed_prediction and ground_truth_normalized:
            is_correct = parsed_prediction == ground_truth_normalized
        
        accuracy = 1.0 if is_correct else 0.0
        
        eval_details = {
            "parsed_prediction": parsed_prediction,
            "ground_truth": ground_truth_normalized,
            "is_correct": is_correct,
        }
        
        return accuracy, eval_details
    except Exception as e:
        # If any parsing error occurs, treat as incorrect
        return 0.0, {
            "parsed_prediction": None,
            "ground_truth": str(ground_truth),
            "is_correct": False,
            "error": str(e),
        }


class OnPolicyDistillSCIQAWorkflow(OnPolicyDistillWorkflow):
    """On-policy distillation workflow with SC-IQa style format.

    This workflow:
    - Uses SC-IQa style prompt format with <reasoning> and <answer> tags
    - Computes accuracy by extracting answer from <answer> tags and comparing with ground truth
    - Suitable for SC-IQa tasks (multiple choice questions with A/B/C/D answers)
    """

    def format_messages(self): 
        system_prompt = """
        Given a question and four options, please select the right answer. Respond in the following format: 
        <reasoning> ... </reasoning> 
        <answer> ... </answer> 
        For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else. 
        Do not restate the answer text. For example, if the answer is "A", just output: <answer> A </answer> 
        """
        user_prompt = f"{self.task_desc}\nPlease reason step by step and select the right answer."
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def compute_reward(self, response: Experience) -> float:
        """Compute accuracy as reward by verifying answer correctness.
        
        Extracts answer from <answer>...</answer> tags in the response
        and compares with ground truth (self.truth).
        
        Returns:
            1.0 if answer is correct, 0.0 otherwise
        """
        if response.response_text and self.truth:
            accuracy, eval_details = verify_answer(
                response_text=response.response_text,
                ground_truth=self.truth
            )
            # Store accuracy and details in metrics
            if response.metrics is None:
                response.metrics = {}
            response.metrics["accuracy"] = accuracy
            response.metrics.update(eval_details)
            return float(accuracy)
        return 0.0