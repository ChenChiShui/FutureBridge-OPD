"""
Random-turn localization ablation for WebShop FutureBridge-OPD.

Same as ALFWorld version: replace KL-based turn selection with random selection.
All other settings (reliability gate, bridge generation, OPD loss) are identical.
"""

import random

from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.webshop.futurebridge_workflow import (
    FutureBridgeWebShopWorkflow,
)


@WORKFLOWS.register_module("FutureBridgeRandomTurnWebShopWorkflow")
class FutureBridgeRandomTurnWebShopWorkflow(FutureBridgeWebShopWorkflow):
    """Random Bridge for WebShop: randomly select turn for bridge instead of KL-based."""

    def _select_bridge_candidates(self, per_turn_disagreement):
        """Randomize only localization; inherit FutureBridge execution and paired gate."""
        valid_indices = list(range(max(0, len(per_turn_disagreement) - 1)))
        if not valid_indices:
            return []

        n_candidates = min(self.bridge_position_top_k, len(valid_indices))
        episode_key = (
            f"{self.task_desc}:{getattr(self.task, 'batch_id', '')}:"
            f"{getattr(self, 'run_id_base', 0)}"
        )
        rng = random.Random(episode_key)
        return [(idx, 0.0) for idx in rng.sample(valid_indices, n_candidates)]
