# -*- coding: utf-8 -*-
"""Shared utilities for cross-family (Qwen teacher → Llama student) workflows.

All methods use action-level hard distillation: Teacher action as SFT target,
tokenized by Student's own tokenizer. No cross-tokenizer KL is needed.
"""

from typing import List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
    HISTORY_LENGTH,
    parse_action,
    format_observation,
    _format_history,
)


def canonicalize_action(raw_response: str) -> Optional[str]:
    """Extract canonical action from model response."""
    return parse_action(raw_response) or None


def wrap_action_target(action: str) -> str:
    """Wrap canonical action into SFT target format."""
    return f"<action>{action}</action>"


def build_user_content(
    task_description: str,
    history: List[str],
    observation: str,
    admissible_commands: List[str],
    step: int,
) -> str:
    """Build user message content from environment state.

    Shared by both Llama and Qwen — the text is model-agnostic.
    Each model's tokenizer applies its own chat template.
    """
    format_obs = format_observation(observation)
    if admissible_commands and isinstance(admissible_commands[0], list):
        admissible_commands = admissible_commands[0]
    reformatted = "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")

    if len(history) < HISTORY_LENGTH:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=format_obs,
            admissible_actions=reformatted,
        )
    else:
        action_history_str = "\n".join(
            history[-HISTORY_LENGTH:] if len(history) >= HISTORY_LENGTH else history
        )
        return ALFWORLD_TEMPLATE.format(
            task_description=task_description,
            step_count=step,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history=action_history_str,
            current_step=step + 1,
            current_observation=format_obs,
            admissible_actions=reformatted,
        )


async def make_cross_family_sft_experience(
    student_model,
    messages: List[dict],
    target_action: str,
    temperature: float = 1.0,
) -> Optional[Experience]:
    """Build an SFT target experience using the student (Llama) tokenizer.

    1. Append target action as assistant message
    2. Tokenize full conversation with student tokenizer
    3. Fix action_mask: only the LAST assistant turn (target) should have mask=1
    4. Set teacher_logprobs = zeros (SFT loss ignores advantages)

    Returns None if tokenization fails or prompt is truncated.
    """
    target_response = wrap_action_target(target_action)
    target_messages = messages + [{"role": "assistant", "content": target_response}]

    try:
        target_exp = await student_model.convert_messages_to_experience_async(
            messages=target_messages,
            temperature=temperature,
        )
    except Exception:
        return None

    if target_exp.truncate_status == "prompt_truncated":
        return None
    if target_exp.logprobs is None:
        return None

    # Fix action_mask: zero out historical assistant tokens
    try:
        prompt_only_exp = await student_model.convert_messages_to_experience_async(
            messages=messages,
            temperature=temperature,
        )
        target_start = len(prompt_only_exp.tokens) - target_exp.prompt_length
        if target_start > 0 and target_exp.action_mask is not None:
            target_exp.action_mask[:target_start] = 0
    except Exception:
        pass

    # SFT loss ignores advantages; set teacher_logprobs to satisfy schema
    target_exp.teacher_logprobs = torch.zeros_like(target_exp.logprobs)

    return target_exp


async def score_action_with_teacher(
    teacher_model,
    messages: List[dict],
    action_text: str,
    temperature: float = 1.0,
) -> Optional[float]:
    """Compute length-normalized teacher logprob on an action.

    Used for teacher action margin in CF-FTB-Hard bridge selection.
    Teacher tokenizes (messages + action) and computes logprob on response tokens.
    """
    action_messages = messages + [{"role": "assistant", "content": action_text}]
    try:
        exp = await teacher_model.convert_messages_to_experience_async(
            messages=action_messages,
            temperature=temperature,
        )
        if exp.logprobs is None:
            return None
        # Length-normalized mean logprob on response tokens
        resp_lp = exp.logprobs
        if hasattr(resp_lp, 'mean'):
            return float(resp_lp.mean())
        return None
    except Exception:
        return None
