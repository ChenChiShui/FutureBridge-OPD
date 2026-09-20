# -*- coding: utf-8 -*-
"""SDPO policy loss implementation."""

from typing import Any, Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


def _to_prob_and_log_prob(log_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.exp(log_probs)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
    norm_log_probs = torch.log(probs + 1e-12)
    return probs, norm_log_probs


def _prepare_topk_distribution(
    topk_log_probs: torch.Tensor,
    add_tail: bool,
) -> torch.Tensor:
    probs = torch.exp(topk_log_probs)
    if add_tail:
        tail_prob = 1.0 - probs.sum(dim=-1, keepdim=True)
        tail_prob = torch.clamp(tail_prob, min=1e-12)
        probs = torch.cat([probs, tail_prob], dim=-1)
        return torch.log(probs + 1e-12)

    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
    return torch.log(probs + 1e-12)


def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute SDPO loss from config-described branches."""
    cfg = self_distillation_config or {}
    alpha = float(cfg.get("alpha", 0.5))
    full_logit_distillation = bool(cfg.get("full_logit_distillation", True))
    use_topk = bool(cfg.get("distillation_use_topk", False))
    add_tail = bool(cfg.get("distillation_add_tail", True))
    is_clip = cfg.get("is_clip", None)

    response_mask = response_mask.to(dtype=student_log_probs.dtype)

    if self_distillation_mask is not None:
        seq_mask = self_distillation_mask.to(dtype=student_log_probs.dtype).unsqueeze(-1)
        loss_mask = response_mask * seq_mask
    else:
        loss_mask = response_mask

    if full_logit_distillation:
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError(
                    "`distillation_use_topk=True` requires student_topk_log_probs and teacher_topk_log_probs."
                )
            student_distill_log_probs = _prepare_topk_distribution(student_topk_log_probs, add_tail)
            teacher_distill_log_probs = _prepare_topk_distribution(teacher_topk_log_probs, add_tail)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError(
                    "`full_logit_distillation=True` requires student_all_log_probs and teacher_all_log_probs."
                )
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        student_probs, student_distill_log_probs = _to_prob_and_log_prob(student_distill_log_probs)
        teacher_probs, teacher_distill_log_probs = _to_prob_and_log_prob(teacher_distill_log_probs)

        if alpha <= 0.0:
            # KL(Student || Teacher)
            per_token_loss = (
                student_probs * (student_distill_log_probs - teacher_distill_log_probs)
            ).sum(dim=-1)
        elif alpha >= 1.0:
            # KL(Teacher || Student)
            per_token_loss = (
                teacher_probs * (teacher_distill_log_probs - student_distill_log_probs)
            ).sum(dim=-1)
        else:
            # Generalized JSD:
            # M = (1-alpha) * Student + alpha * Teacher
            mixture_probs = (1.0 - alpha) * student_probs + alpha * teacher_probs
            mixture_log_probs = torch.log(mixture_probs + 1e-12)
            kl_teacher = (mixture_probs * (mixture_log_probs - teacher_distill_log_probs)).sum(dim=-1)
            kl_student = (mixture_probs * (mixture_log_probs - student_distill_log_probs)).sum(dim=-1)
            per_token_loss = (1.0 - alpha) * kl_student + alpha * kl_teacher
    else:
        # Token-level distillation
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = -(log_ratio.detach()) * student_log_probs

    if is_clip is not None and old_log_probs is not None:
        ratio = torch.exp(student_log_probs - old_log_probs)
        ratio = torch.clamp(ratio, min=0.0, max=float(is_clip))
        per_token_loss = per_token_loss * ratio

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    if loss_mask.sum() <= 0:
        loss = student_log_probs.new_tensor(0.0)
        metrics = {
            "self_distill_loss": 0.0,
            "empty_target_batch": 1.0,
        }
        return loss, metrics

    loss = aggregate_loss(per_token_loss, loss_mask, loss_agg_mode=loss_agg_mode)
    metrics = {
        "self_distill_loss": loss.detach().item(),
        "empty_target_batch": 0.0,
        "self_distill_alpha": float(alpha),
        "sdpo_alpha": float(alpha),
        "self_distill_valid_tokens": float(loss_mask.sum().detach().item()),
        "self_distill_per_token_mean": masked_mean(per_token_loss, loss_mask).detach().item(),
    }
    return loss, metrics


class SDPOPolicyLossFn(PolicyLossFn):
    """SDPO policy loss.

    Preferred path: consume trainer-computed ``advantages`` (from SDPOAdvantage)
    and optimize with an IS-style objective.

    Backward-compatible path: if ``advantages`` is absent, falls back to the
    legacy inline self-distillation loss.
    """

    def __init__(
        self,
        backend: str = "verl",
        alpha: float = 0.5,
        full_logit_distillation: bool = True,
        distillation_use_topk: bool = False,
        distillation_add_tail: bool = True,
        is_clip: Optional[float] = None,
        loss_agg_mode: str = "token-mean",
    ) -> None:
        super().__init__(backend=backend)
        self.loss_agg_mode = loss_agg_mode
        self.self_distillation_config = {
            "alpha": alpha,
            "full_logit_distillation": full_logit_distillation,
            "distillation_use_topk": distillation_use_topk,
            "distillation_add_tail": distillation_add_tail,
            "is_clip": is_clip,
        }

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: Optional[torch.Tensor] = None,
        teacher_logprobs: Optional[torch.Tensor] = None,
        old_logprob: Optional[torch.Tensor] = None,
        student_all_log_probs: Optional[torch.Tensor] = None,
        teacher_all_log_probs: Optional[torch.Tensor] = None,
        student_topk_log_probs: Optional[torch.Tensor] = None,
        teacher_topk_log_probs: Optional[torch.Tensor] = None,
        self_distillation_mask: Optional[torch.Tensor] = None,
        rollout_is_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        # Preferred path: policy-gradient update over trainer-side SDPO advantages.
        if advantages is not None and old_logprob is not None:
            log_ratio = torch.clamp(logprob - old_logprob, min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            is_clip = self.self_distillation_config.get("is_clip", None)
            if is_clip is not None:
                ratio = torch.clamp(ratio, min=0.0, max=float(is_clip))
            if rollout_is_weights is not None:
                ratio = ratio * rollout_is_weights

            per_token_loss = -advantages * ratio
            loss = aggregate_loss(per_token_loss, action_mask, loss_agg_mode=self.loss_agg_mode)
            metrics = {
                "sdpo_loss": loss.detach().item(),
                "self_distill_loss": loss.detach().item(),
                "sdpo_alpha": float(self.self_distillation_config.get("alpha", 0.5)),
                "self_distill_alpha": float(self.self_distillation_config.get("alpha", 0.5)),
                "ratio/mean": masked_mean(ratio, action_mask).detach().item(),
                "approx_kl": masked_mean(-log_ratio, action_mask).detach().item(),
            }
            return loss, metrics

        # Legacy path: SDPO loss computed directly inside policy loss.
        if teacher_logprobs is None:
            raise ValueError(
                "SDPOPolicyLossFn requires either (advantages + old_logprob) "
                "or teacher_logprobs for legacy inline distillation."
            )

        loss, metrics = compute_self_distillation_loss(
            student_log_probs=logprob,
            teacher_log_probs=teacher_logprobs,
            response_mask=action_mask,
            self_distillation_config=self.self_distillation_config,
            old_log_probs=old_logprob,
            student_all_log_probs=student_all_log_probs,
            teacher_all_log_probs=teacher_all_log_probs,
            student_topk_log_probs=student_topk_log_probs,
            teacher_topk_log_probs=teacher_topk_log_probs,
            self_distillation_mask=self_distillation_mask,
            loss_agg_mode=self.loss_agg_mode,
            rollout_is_weights=rollout_is_weights,
        )
        metrics["sdpo_loss"] = metrics["self_distill_loss"]
        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "alpha": 0.5,
            "full_logit_distillation": True,
            "distillation_use_topk": False,
            "distillation_add_tail": True,
            "is_clip": None,
            "loss_agg_mode": "token-mean",
        }
