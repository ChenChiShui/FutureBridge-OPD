# -*- coding: utf-8 -*-
"""
Random Turn Ablation for FutureBridge-OPD.

# (implementation detail)
# (implementation detail)

# (implementation detail)
  # (implementation detail)
  # (implementation detail)
  # (implementation detail)
"""

import random
from typing import List

from trinity.common.workflows.envs.TCOD.alfworld.futurebridge_workflow import (
    _TeacherReliableAnchorAlfworldBase,
)


class FutureBridgeRandomTurnAlfworldWorkflow(_TeacherReliableAnchorAlfworldBase):
    """
    # (implementation detail)
    # (implementation detail)

    # (implementation detail)
    # (implementation detail)
    """

    async def _try_kl_bridges(self, turn_responses, turn_memories):
        # (implementation detail)
        # (implementation detail)
        if not self._current_anchor_reliable:
            return []

        import re as _re

        def _has_valid_action(resp):
            text = resp.response_text or ""
            return bool(_re.search(r"<action>.*?</action>", text, _re.DOTALL))

        # (implementation detail)
        valid_indices = [
            i for i, resp in enumerate(turn_responses)
            if _has_valid_action(resp)
        ]
        if not valid_indices:
            return []

        n_bridge = min(self.bridge_kl_max_per_ep, len(valid_indices))
        selected = random.sample(valid_indices, n_bridge)

        bridge_exps = []
        for idx, turn_idx in enumerate(selected):
            memory_at_turn = turn_memories[turn_idx]
            # (implementation detail)
            exps = await self._generate_kl_bridge(
                memory_at_turn=memory_at_turn,
                # (implementation detail)
                bridge_idx=idx,
                # (implementation detail)
            )
            bridge_exps.extend(exps)

        return bridge_exps
