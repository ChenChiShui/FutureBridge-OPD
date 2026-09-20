# -*- coding: utf-8 -*-
"""
Random Turn Ablation for WebShop FutureBridge-OPD.

Same as ALFWorld version: replace KL-based turn selection with random selection.
All other settings (reliability gate, bridge generation, OPD loss) are identical.
"""

import random
from typing import List, Tuple

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.webshop.futurebridge_base import (
    _FutureBridgeWebShopBase,
)


@WORKFLOWS.register_module("FutureBridgeRandomTurnWebShopWorkflow")
class FutureBridgeRandomTurnWebShopWorkflow(_FutureBridgeWebShopBase):
    """Random Bridge for WebShop: randomly select turn for bridge instead of KL-based."""

    async def _try_kl_bridge(
        self,
        start_step: int,
        memory_snapshots: List[List[dict]],
        per_turn_kl: List[float],
        turn_responses: List[Experience],
    ) -> List[Experience]:
        """Randomly select turn for bridge, keep everything else identical."""
        if self._final_reward >= self.bridge_reward_threshold:
            return []
        if not per_turn_kl:
            return []

        # Randomly select turn(s) instead of KL-based selection
        valid_indices = list(range(len(per_turn_kl)))
        if not valid_indices:
            return []

        n_bridge = min(self.bridge_max_per_ep, len(valid_indices))
        selected = random.sample(valid_indices, n_bridge)

        bridge_exps: List[Experience] = []
        for idx, turn_idx in enumerate(selected):
            bridge_idx = start_step + turn_idx
            exps = await self._generate_kl_bridge(
                memory_at_turn=memory_snapshots[turn_idx],
                trigger_kl=0.0,
                bridge_idx=bridge_idx,
                bridge_weight=1.0,
            )
            bridge_exps.extend(exps)

        return bridge_exps
