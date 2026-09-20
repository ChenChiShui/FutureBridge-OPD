# -*- coding: utf-8 -*-
"""On-Policy Distillation advantage computation.

Reference: Tinker library's on-policy distillation.

advantages = -(student_logprobs - teacher_logprobs)
           = teacher_logprobs - student_logprobs

For multi-turn workflows (OPD_alfworld_workflow, OPD_scienceworld_workflow, etc.):
- Workflow returns List[Experience] (one per turn), not a single response
- Each turn is one row in the batch; use MultiTurnOpdAdvantage
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn


def _compute_opd_advantage(
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_valid_mask: torch.Tensor,
    kl_coef: float,
) -> Tuple[torch.Tensor, Dict]:
    """Core OPD advantage computation with optional teacher_valid_mask.

    When teacher_valid_mask is provided (hint workflow), use it to mask positions
    where teacher_logprobs was padded (e.g. due to length mismatch).
    """
    advantages = kl_coef * (teacher_log_probs - old_log_probs)

    # apply mask
    effective_mask = response_mask & teacher_valid_mask
    advantages = advantages * effective_mask

    # metrics
    kl_per_token = old_log_probs - teacher_log_probs
    kl_sum = (kl_per_token * effective_mask).sum(dim=-1)
    metrics = {
        "kl/mean": kl_sum.mean().item(),
        "kl/std": kl_sum.std().item() if kl_sum.numel() > 1 else 0.0,
        "advantages/mean": advantages.sum(dim=-1).mean().item(),
    }
    return advantages, metrics


class OnPolicyDistillAdvantage(AdvantageFn):
    """Advantage function for on-policy distillation.

    Computes: advantages = kl_coef * (teacher_logprobs - student_logprobs)

    The teacher_logprobs should be stored in Experience.teacher_logprobs
    by the workflow during exploration.
    """

    def __init__(self, kl_coef: float = 1.0) -> None:
        self.kl_coef = kl_coef

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Compute advantages from teacher and student logprobs.

        Args:
            exps: DataProto containing:
                - old_log_probs: student's sampling logprobs [batch, seq]
                - teacher_logprobs: teacher's logprobs [batch, seq]
                - response_mask: mask for response tokens [batch, seq]

        Returns:
            exps: DataProto with advantages and returns added
            metrics: Dict with kl and advantage statistics
        """
        old_log_probs = exps.batch["old_log_probs"]
        teacher_log_probs = exps.batch["teacher_logprobs"]
        response_mask = exps.batch["response_mask"]

        # Standard OPD: teacher_valid_mask = response_mask (no length mismatch)
        teacher_valid_mask = response_mask

        advantages, metrics = _compute_opd_advantage(
            old_log_probs,
            teacher_log_probs,
            response_mask,
            teacher_valid_mask,
            self.kl_coef,
        )

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = advantages.clone()

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {"kl_coef": 1.0}


def _parse_run_ids(unique_ids) -> List[str]:
    """Parse unique_ids (format: batch/task/run/step/suffix) to get run_id (batch/task/run)."""
    run_ids = []
    for uid in unique_ids:
        parts = str(uid).split("/")
        if len(parts) >= 4:
            run_ids.append("/".join(parts[:3]))  # batch/task/run
        else:
            run_ids.append(str(uid))
    return run_ids


class MultiTurnOpdAdvantage(AdvantageFn):
    """Advantage function for multi-turn on-policy distillation.

    Used with OPD_alfworld_workflow, OPD_scienceworld_workflow, OPD_webshop_workflow
    and TCOD variants where:
    - Workflow returns List[Experience] (turn_responses), one Experience per turn
    - Each turn is one row in the batch: batch dim = num_turns across all episodes
    - Same per-token formula: advantages = kl_coef * (teacher_logprobs - student_logprobs)

    Unlike single-turn OPD where batch = [num_samples], here batch = [num_turns].
    Adds trajectory-level metrics (kl/trajectory_mean) by grouping turns by run_id.
    """

    def __init__(self, kl_coef: float = 1.0) -> None:
        self.kl_coef = kl_coef

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Compute advantages for multi-turn OPD (list of turn responses).

        Args:
            exps: DataProto containing:
                - old_log_probs: student's sampling logprobs [num_turns, seq]
                - teacher_logprobs: teacher's logprobs [num_turns, seq]
                - response_mask: mask for response tokens [num_turns, seq]
                - unique_ids: (optional) for trajectory grouping, format batch/task/run/step/suffix

        Returns:
            exps: DataProto with advantages and returns added
            metrics: Dict with kl, advantage, and trajectory-level statistics
        """
        old_log_probs = exps.batch["old_log_probs"]
        teacher_log_probs = exps.batch["teacher_logprobs"]
        response_mask = exps.batch["response_mask"]

        teacher_valid_mask = exps.batch.get("teacher_logprobs_valid_mask", None)
        if teacher_valid_mask is not None:
            teacher_valid_mask = teacher_valid_mask.to(
                dtype=response_mask.dtype, device=response_mask.device
            )
        else:
            teacher_valid_mask = response_mask

        advantages, metrics = _compute_opd_advantage(
            old_log_probs,
            teacher_log_probs,
            response_mask,
            teacher_valid_mask,
            self.kl_coef,
        )

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = advantages.clone()

        # Trajectory-level metrics: group by run_id and sum KL per trajectory
        unique_ids = exps.batch.get("unique_ids", None)
        if unique_ids is not None:
            run_ids = _parse_run_ids(unique_ids)
            kl_per_token = old_log_probs - teacher_log_probs
            effective_mask = response_mask & teacher_valid_mask
            kl_per_turn = (kl_per_token * effective_mask).sum(dim=-1)

            traj_kl_sums: Dict[str, float] = defaultdict(float)
            for run_id, kl in zip(run_ids, kl_per_turn.tolist()):
                traj_kl_sums[run_id] += kl

            if traj_kl_sums:
                traj_kl_list = list(traj_kl_sums.values())
                n = len(traj_kl_list)
                mean_kl = sum(traj_kl_list) / n
                metrics["kl/trajectory_mean"] = mean_kl
                metrics["kl/trajectory_std"] = (
                    (sum((x - mean_kl) ** 2 for x in traj_kl_list) / n) ** 0.5
                    if n > 1
                    else 0.0
                )

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {"kl_coef": 1.0}


