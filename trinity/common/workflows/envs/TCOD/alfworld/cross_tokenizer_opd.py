"""
Cross-tokenizer shared-support OPD utilities.
Enables Llama teacher → Qwen student by computing OPD advantages
only on byte-aligned one-to-one token positions.
"""
import torch
from typing import Dict, List, Tuple, Optional


def build_exact_token_map(
    student_tok, teacher_tok
) -> Dict[int, int]:
    """Build one-to-one token id mapping from student vocab to teacher vocab.

    A student token q_id maps to teacher token l_id only when:
    1. decode(q_id) → text → teacher.encode(text) produces exactly 1 token
    2. teacher.decode(l_id) → text has identical UTF-8 bytes

    Returns:
        Dict mapping student_token_id → teacher_token_id
    """
    s2t = {}
    s_vocab_size = getattr(student_tok, "vocab_size", len(student_tok))
    for s_id in range(s_vocab_size):
        try:
            s_str = student_tok.decode([s_id], skip_special_tokens=False)
        except Exception:
            continue
        if not s_str:
            continue
        t_ids = teacher_tok.encode(s_str, add_special_tokens=False)
        if len(t_ids) != 1:
            continue
        t_id = t_ids[0]
        t_str = teacher_tok.decode([t_id], skip_special_tokens=False)
        if t_str.encode("utf-8") == s_str.encode("utf-8"):
            s2t[s_id] = t_id
    return s2t


def compute_byte_spans(token_ids: List[int], tokenizer) -> List[Tuple[int, int]]:
    """Compute byte span [start, end) for each token in the sequence.

    Returns list of (byte_start, byte_end) tuples.
    """
    spans = []
    byte_offset = 0
    for tid in token_ids:
        # Decode single token to get its byte length
        text = tokenizer.decode([tid], skip_special_tokens=False)
        byte_len = len(text.encode("utf-8"))
        spans.append((byte_offset, byte_offset + byte_len))
        byte_offset += byte_len
    return spans


def align_tokens_by_bytes(
    student_token_ids: List[int],
    teacher_token_ids: List[int],
    student_tok,
    teacher_tok,
    s2t_map: Dict[int, int],
) -> List[Optional[int]]:
    """Find byte-aligned positions between student and teacher token sequences.

    For each student token position, returns:
    - The corresponding teacher token position index (if byte-aligned one-to-one match exists)
    - None (if no match)

    Args:
        student_token_ids: Student response token ids
        teacher_token_ids: Teacher response token ids (same text, different tokenizer)
        s2t_map: Precomputed student→teacher token id map
        student_tok: Student tokenizer
        teacher_tok: Teacher tokenizer

    Returns:
        List of Optional[int], length = len(student_token_ids)
    """
    s_spans = compute_byte_spans(student_token_ids, student_tok)
    t_spans = compute_byte_spans(teacher_token_ids, teacher_tok)

    # Build a dict: byte_span → teacher_position_index
    # Only for tokens that are in the s2t_map
    t_span_to_pos = {}
    for t_idx, (t_start, t_end) in enumerate(t_spans):
        t_id = teacher_token_ids[t_idx]
        t_span_to_pos[(t_start, t_end)] = t_idx

    alignment = []
    for s_idx, (s_start, s_end) in enumerate(s_spans):
        s_id = student_token_ids[s_idx]
        if s_id not in s2t_map:
            alignment.append(None)
            continue
        # Check if there's a teacher token with the exact same byte span
        if (s_start, s_end) in t_span_to_pos:
            t_idx = t_span_to_pos[(s_start, s_end)]
            t_id = teacher_token_ids[t_idx]
            # Verify the mapping is correct
            if t_id == s2t_map[s_id]:
                alignment.append(t_idx)
            else:
                alignment.append(None)
        else:
            alignment.append(None)

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
    """Compute shared-support OPD teacher logprobs.

    For each student response token, if it has a byte-aligned one-to-one match
    in the teacher's tokenization, use the teacher's logprob. Otherwise mask
    (set teacher_logprob = student_logprob so advantage = 0).

    Args:
        student_tokens: Full student token sequence [prompt | response]
        student_prompt_length: Number of prompt tokens in student sequence
        student_logprobs: Student response logprobs, shape [resp_len]
        teacher_tokens: Full teacher token sequence [prompt | response]
        teacher_prompt_length: Number of prompt tokens in teacher sequence
        teacher_logprobs: Teacher logprobs for full sequence, shape [seq_len - 1]
        student_tok: Student tokenizer
        teacher_tok: Teacher tokenizer
        s2t_map: Precomputed student→teacher token id map

    Returns:
        (teacher_resp_logprobs, n_aligned, n_total)
        - teacher_resp_logprobs: shape [resp_len], masked positions have student_logprob
        - n_aligned: Number of byte-aligned positions
        - n_total: Total response tokens
    """
    # Extract response token ids
    s_resp_tokens = student_tokens[student_prompt_length:]
    t_resp_tokens = teacher_tokens[teacher_prompt_length:]

    # Align by byte spans
    alignment = align_tokens_by_bytes(
        s_resp_tokens, t_resp_tokens, student_tok, teacher_tok, s2t_map
    )

    # Build teacher response logprobs
    n_resp = len(s_resp_tokens)
    teacher_resp_logprobs = student_logprobs.clone()  # Default: mask (advantage=0)

    n_aligned = 0
    for s_idx, t_idx in enumerate(alignment):
        if t_idx is not None:
            # Teacher logprob position: prompt_length - 1 + t_idx
            # (because logprobs[i] is the logprob of token[i+1] given token[:i+1])
            t_logprob_pos = teacher_prompt_length - 1 + t_idx
            if t_logprob_pos < len(teacher_logprobs):
                teacher_resp_logprobs[s_idx] = teacher_logprobs[t_logprob_pos]
                n_aligned += 1

    return teacher_resp_logprobs, n_aligned, n_resp
