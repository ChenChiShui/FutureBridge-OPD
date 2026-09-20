# -*- coding: utf-8 -*-
"""Cross-tokenizer OPD utilities.

Provides turn-level per-byte logprob scoring for cross-family distillation
(Qwen teacher → Llama student). No token alignment needed — both tokenizers
score the same response text independently, normalized by UTF-8 byte length.
"""

from typing import List, Optional

import torch


async def score_response_per_byte(
    model_wrapper,
    messages: List[dict],
    response_text: str,
    temperature: float = 1.0,
) -> Optional[float]:
    """Score a response text using the model's own tokenizer.

    The model re-tokenizes (messages + response) with its own tokenizer,
    computes logprobs on response tokens, and returns logprob_sum / byte_length.

    This is model-agnostic: the same response_text can be scored by both
    Qwen and Llama models, each using their own tokenizer.

    Args:
        model_wrapper: ModelWrapper (student or teacher)
        messages: Conversation messages BEFORE the response (model-agnostic)
        response_text: The response text to score
        temperature: Temperature for logprob computation

    Returns:
        Per-byte logprob score (float), or None if scoring fails.
    """
    byte_len = len(response_text.encode("utf-8"))
    if byte_len == 0:
        return None

    full_messages = messages + [{"role": "assistant", "content": response_text}]

    try:
        exp = await model_wrapper.convert_messages_to_experience_async(
            messages=full_messages,
            temperature=temperature,
        )
    except Exception:
        return None

    if exp.logprobs is None or len(exp.logprobs) == 0:
        return None

    logprob_sum = float(exp.logprobs.sum().item())
    return logprob_sum / byte_len


def compute_student_score_from_logprobs(
    logprobs: torch.Tensor,
    response_text: str,
) -> Optional[float]:
    """Compute student per-byte score from existing logprobs.

    Student already has logprobs from chat_async, no need to re-tokenize.

    Args:
        logprobs: Student response logprobs tensor [resp_len]
        response_text: The response text (for byte length)

    Returns:
        Per-byte logprob score (float), or None if invalid.
    """
    byte_len = len(response_text.encode("utf-8"))
    if byte_len == 0 or logprobs is None or len(logprobs) == 0:
        return None

    return float(logprobs.sum().item()) / byte_len


def compute_turn_level_advantages(
    gaps: List[float],
    clamp_val: float = 5.0,
) -> List[float]:
    """Batch-level normalization: subtract mean, clamp.

    Args:
        gaps: Per-turn cross-tokenizer gaps (d_t = s_T - s_S)
        clamp_val: Maximum absolute advantage value

    Returns:
        Normalized advantages (mean-subtracted, clamped)
    """
    if not gaps:
        return []
    mean_gap = sum(gaps) / len(gaps)
    return [
        max(-clamp_val, min(clamp_val, g - mean_gap))
        for g in gaps
    ]
