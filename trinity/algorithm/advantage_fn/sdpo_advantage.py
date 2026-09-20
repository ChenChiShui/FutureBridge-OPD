# -*- coding: utf-8 -*-
"""SDPO advantage computation.

Supports three alpha-controlled distillation modes:
- alpha = 0.0: Forward KL (configured as in SDPO policy snippet)
- alpha = 1.0: Reverse KL
- 0 < alpha < 1: Generalized Jensen-Shannon Divergence
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn


def _to_prob_and_log_prob(log_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.exp(log_probs)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
    norm_log_probs = torch.log(probs + 1e-12)
    return probs, norm_log_probs


def _prepare_topk_distribution(topk_log_probs: torch.Tensor, add_tail: bool) -> torch.Tensor:
    probs = torch.exp(topk_log_probs)
    if add_tail:
        tail_prob = 1.0 - probs.sum(dim=-1, keepdim=True)
        tail_prob = torch.clamp(tail_prob, min=1e-12)
        probs = torch.cat([probs, tail_prob], dim=-1)
        return torch.log(probs + 1e-12)

    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
    return torch.log(probs + 1e-12)


def _build_two_class_log_distribution(token_log_probs: torch.Tensor) -> torch.Tensor:
    """Build [chosen_token, tail] log distribution from token logprob."""
    chosen_prob = torch.exp(token_log_probs)
    chosen_prob = torch.clamp(chosen_prob, min=1e-12, max=1.0 - 1e-12)
    tail_prob = torch.clamp(1.0 - chosen_prob, min=1e-12, max=1.0)
    probs = torch.stack([chosen_prob, tail_prob], dim=-1)
    return torch.log(probs + 1e-12)


class SDPOAdvantage(AdvantageFn):
    """Advantage function for SDPO.

    Args:
        kl_coef: Scaling coefficient for the computed per-token advantages.
        alpha: Distillation mode controller.
            - 0.0: forward KL branch
            - 1.0: reverse KL branch
            - (0, 1): generalized JSD branch
        full_logit_distillation: If True, use full distribution KL/JSD.
        distillation_use_topk: If True (with full_logit_distillation), use top-k distribution.
        distillation_add_tail: Whether to add tail probability in top-k mode.
    """

    def __init__(
        self,
        kl_coef: float = 1.0,
        alpha: float = 0.5,
        full_logit_distillation: bool = False,
        distillation_use_topk: bool = False,
        distillation_add_tail: bool = True,
    ) -> None:
        self.kl_coef = kl_coef
        self.alpha = float(alpha)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"Invalid alpha: {self.alpha}. Expected alpha in [0, 1].")
        self.full_logit_distillation = full_logit_distillation
        self.distillation_use_topk = distillation_use_topk
        self.distillation_add_tail = distillation_add_tail

    def _select_student_teacher_distill_log_probs(
        self,
        old_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        exps: DataProto,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.full_logit_distillation:
            # Fallback for token-only pipelines: approximate each token as a 2-class distribution.
            return (
                _build_two_class_log_distribution(old_log_probs),
                _build_two_class_log_distribution(teacher_log_probs),
            )

        if self.distillation_use_topk:
            student_topk_log_probs = exps.batch.get("student_topk_log_probs")
            teacher_topk_log_probs = exps.batch.get("teacher_topk_log_probs")
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError(
                    "`distillation_use_topk=True` requires student_topk_log_probs and teacher_topk_log_probs in batch."
                )
            return (
                _prepare_topk_distribution(student_topk_log_probs, self.distillation_add_tail),
                _prepare_topk_distribution(teacher_topk_log_probs, self.distillation_add_tail),
            )

        student_all_log_probs = exps.batch.get("student_all_log_probs")
        teacher_all_log_probs = exps.batch.get("teacher_all_log_probs")
        if student_all_log_probs is None or teacher_all_log_probs is None:
            raise ValueError(
                "`full_logit_distillation=True` requires student_all_log_probs and teacher_all_log_probs in batch."
            )
        _, student_distill_log_probs = _to_prob_and_log_prob(student_all_log_probs)
        _, teacher_distill_log_probs = _to_prob_and_log_prob(teacher_all_log_probs)
        return student_distill_log_probs, teacher_distill_log_probs

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        old_log_probs = exps.batch["old_log_probs"]
        teacher_log_probs = exps.batch["teacher_logprobs"]
        response_mask = exps.batch["response_mask"]
        alpha = self.alpha

        student_distill_log_probs, teacher_distill_log_probs = (
            self._select_student_teacher_distill_log_probs(old_log_probs, teacher_log_probs, exps)
        )

        if alpha == 0.0:
            # Forward KL branch (follows requested SDPO branch definition)
            kl_loss = F.kl_div(
                student_distill_log_probs,
                teacher_distill_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            kl_mode = "forward_kl"
        elif alpha == 1.0:
            # Reverse KL branch
            kl_loss = F.kl_div(
                teacher_distill_log_probs,
                student_distill_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            kl_mode = "reverse_kl"
        else:
            alpha_t = torch.tensor(alpha, dtype=student_distill_log_probs.dtype, device=student_distill_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_distill_log_probs + torch.log1p(-alpha_t),
                        teacher_distill_log_probs + torch.log(alpha_t),
                    ]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(
                mixture_log_probs,
                teacher_distill_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            kl_student = F.kl_div(
                mixture_log_probs,
                student_distill_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)
            kl_mode = "generalized_jsd"

        teacher_valid_mask = exps.batch.get("teacher_logprobs_valid_mask")
        if teacher_valid_mask is not None:
            teacher_valid_mask = teacher_valid_mask.to(
                dtype=response_mask.dtype, device=response_mask.device
            )
            effective_mask = response_mask * teacher_valid_mask
        else:
            effective_mask = response_mask

        # Convert divergence loss to maximize-style signal for policy gradient.
        advantages = -self.kl_coef * kl_loss
        advantages = advantages * effective_mask

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = advantages.clone()

        kl_sum = (kl_loss * effective_mask).sum(dim=-1)
        metrics = {
            "sdpo/alpha": alpha,
            "sdpo/kl_mode": kl_mode,
            "sdpo/kl_mean": kl_sum.mean().item(),
            "sdpo/kl_std": kl_sum.std().item() if kl_sum.numel() > 1 else 0.0,
            "sdpo/advantages_mean": advantages.sum(dim=-1).mean().item(),
        }
        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "kl_coef": 1.0,
            "alpha": 0.5,
            "full_logit_distillation": False,
            "distillation_use_topk": False,
            "distillation_add_tail": True,
        }
