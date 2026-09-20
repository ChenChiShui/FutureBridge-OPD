# -*- coding: utf-8 -*-
"""Cross-tokenizer turn-level OPD advantage function.

Reads pre-computed turn-level advantages from the batch (passed via CustomField
from the workflow) and broadcasts them to all response tokens.

Workflow sets exp.info["turn_advantage"] = gap (float) for each experience.
The CustomField mechanism passes this to batch["turn_advantage"] as a [batch] tensor.
This advantage function broadcasts it to [batch, seq_len] using response_mask.

The advantage is detached (comes from exp.info, not part of the computation graph),
so only the PPO ratio provides gradient flow.
"""

from typing import Any, Dict, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn


class CrossTokenizerOpdAdvantage(AdvantageFn):
    """Turn-level cross-tokenizer OPD advantage.

    Experiences must have:
    - exp.info["turn_advantage"]: float (precomputed per-byte gap, already normalized)
    - exp.teacher_logprobs: zeros (placeholder, not used)
    - CustomField("turn_advantage", "turn_advantage", torch.float32) configured

    The advantage is broadcast from turn-level [batch] to per-token [batch, seq]
    using response_mask.
    """

    def __init__(self, kl_coef: float = 1.0) -> None:
        self.kl_coef = kl_coef

    def __call__(self, exps: Any, **kwargs) -> Tuple[Any, Dict]:
        # Read pre-computed turn-level advantages
        # Shape: [batch_size]
        turn_advs = exps.batch.get("turn_advantage", None)
        if turn_advs is None:
            # Fallback: no turn_advantage, use zeros
            response_mask = exps.batch["response_mask"]
            advantages = torch.zeros_like(response_mask, dtype=torch.float32)
            exps.batch["advantages"] = advantages
            exps.batch["returns"] = advantages.clone()
            return exps, {"xt_opd/no_turn_advantage": 1.0}

        response_mask = exps.batch["response_mask"]  # [batch, seq]

        # Broadcast: [batch] → [batch, 1] * [batch, seq] → [batch, seq]
        advantages = turn_advs.unsqueeze(-1).float() * response_mask.float() * self.kl_coef

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = advantages.clone()

        # Metrics
        valid_mask = response_mask.bool()
        n_valid = valid_mask.sum().item()
        n_batch = turn_advs.shape[0]
        adv_mean = turn_advs.float().mean().item()
        adv_std = turn_advs.float().std().item() if n_batch > 1 else 0.0
        n_positive = (turn_advs > 0).sum().item()

        metrics = {
            "xt_opd/turn_adv_mean": adv_mean,
            "xt_opd/turn_adv_std": adv_std,
            "xt_opd/turn_adv_positive_ratio": n_positive / max(n_batch, 1),
            "xt_opd/n_turns": float(n_batch),
            "xt_opd/n_valid_tokens": float(n_valid),
        }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {"kl_coef": 1.0}
