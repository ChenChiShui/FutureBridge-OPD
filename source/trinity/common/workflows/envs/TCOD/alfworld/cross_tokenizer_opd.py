"""Cross-tokenizer shared-support OPD utilities."""

from typing import Dict, List, Optional, Tuple

import torch


def build_exact_token_map(student_tok, teacher_tok) -> Dict[int, int]:
    """Map Student token IDs to byte-identical one-token Teacher IDs."""
    student_to_teacher = {}
    student_vocab_size = getattr(student_tok, "vocab_size", len(student_tok))
    for student_id in range(student_vocab_size):
        try:
            student_text = student_tok.decode(
                [student_id], skip_special_tokens=False
            )
        except Exception:
            continue
        if not student_text:
            continue
        teacher_ids = teacher_tok.encode(
            student_text, add_special_tokens=False
        )
        if len(teacher_ids) != 1:
            continue
        teacher_id = teacher_ids[0]
        teacher_text = teacher_tok.decode(
            [teacher_id], skip_special_tokens=False
        )
        if teacher_text.encode("utf-8") == student_text.encode("utf-8"):
            student_to_teacher[student_id] = teacher_id
    return student_to_teacher


def compute_byte_spans(
    token_ids: List[int], tokenizer
) -> List[Tuple[int, int]]:
    """Compute half-open UTF-8 byte spans for a token sequence."""
    spans = []
    byte_offset = 0
    for token_id in token_ids:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        byte_length = len(text.encode("utf-8"))
        spans.append((byte_offset, byte_offset + byte_length))
        byte_offset += byte_length
    return spans


def align_tokens_by_bytes(
    student_token_ids: List[int],
    teacher_token_ids: List[int],
    student_tok,
    teacher_tok,
    student_to_teacher: Dict[int, int],
) -> List[Optional[int]]:
    """Return the Teacher position for each exact one-to-one Student token."""
    student_spans = compute_byte_spans(student_token_ids, student_tok)
    teacher_spans = compute_byte_spans(teacher_token_ids, teacher_tok)
    teacher_position_by_span = {
        span: position for position, span in enumerate(teacher_spans)
    }

    alignment = []
    for student_position, span in enumerate(student_spans):
        student_id = student_token_ids[student_position]
        if student_id not in student_to_teacher:
            alignment.append(None)
            continue
        teacher_position = teacher_position_by_span.get(span)
        if teacher_position is None:
            alignment.append(None)
            continue
        teacher_id = teacher_token_ids[teacher_position]
        alignment.append(
            teacher_position
            if teacher_id == student_to_teacher[student_id]
            else None
        )
    return alignment


def compute_shared_support_opd(
    student_tokens: List[int],
    student_prompt_length: int,
    student_logprobs: torch.Tensor,
    teacher_tokens: List[int],
    teacher_prompt_length: int,
    teacher_logprobs: torch.Tensor,
    student_tok,
    teacher_tok,
    s2t_map: Dict[int, int],
) -> Tuple[torch.Tensor, float, float]:
    """Score byte-aligned one-token support and mask all other positions."""
    student_response_tokens = student_tokens[student_prompt_length:]
    teacher_response_tokens = teacher_tokens[teacher_prompt_length:]
    alignment = align_tokens_by_bytes(
        student_response_tokens,
        teacher_response_tokens,
        student_tok,
        teacher_tok,
        s2t_map,
    )

    teacher_response_logprobs = student_logprobs.clone()
    aligned = 0
    for student_position, teacher_position in enumerate(alignment):
        if teacher_position is None:
            continue
        teacher_logprob_position = (
            teacher_prompt_length - 1 + teacher_position
        )
        if teacher_logprob_position < len(teacher_logprobs):
            teacher_response_logprobs[student_position] = teacher_logprobs[
                teacher_logprob_position
            ]
            aligned += 1

    return teacher_response_logprobs, aligned, len(student_response_tokens)
