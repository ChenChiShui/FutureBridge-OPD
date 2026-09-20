# -*- coding: utf-8 -*-
"""
Random Turn Ablation for ScienceWorld FutureBridge-OPD.

Same as WebShop/ALFWorld version: replace KL-based turn selection with random selection.
All other settings (reliability gate, bridge generation, OPD loss) are identical.
"""

import random
from typing import List

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.scienceworld.futurebridge_workflow import (
    FutureBridgeScienceWorldWorkflow,
)


@WORKFLOWS.register_module("FutureBridgeRandomTurnScienceWorldWorkflow")
class FutureBridgeRandomTurnScienceWorldWorkflow(FutureBridgeScienceWorldWorkflow):
    """Random Bridge for ScienceWorld: randomly select turn for bridge instead of KL-based."""

    async def _try_kl_bridge(
        self,
        turn_responses: List[Experience],
        start_step: int,
    ) -> List[Experience]:
        """Randomly select turn(s) for bridge, keep everything else identical to FTB."""
        if not turn_responses:
            return []

        # Compute per-turn KL only to identify valid candidates (have logprobs)
        valid_indices: List[int] = []
        for i, resp in enumerate(turn_responses):
            if resp.logprobs is not None and resp.teacher_logprobs is not None:
                valid_indices.append(i)

        if not valid_indices:
            return []

        # Randomly select turn(s) instead of KL-based top-1
        n_bridge = min(self.bridge_kl_max_per_ep, len(valid_indices))
        selected = random.sample(valid_indices, n_bridge)

        bridge_exps: List[Experience] = []
        for turn_idx in selected:
            bridge_turn = start_step + turn_idx  # absolute step in gold trajectory
            if bridge_turn >= len(self._expert_actions):
                continue
            exps = await self._generate_one_bridge(bridge_turn)
            bridge_exps.extend(exps)

        return bridge_exps
