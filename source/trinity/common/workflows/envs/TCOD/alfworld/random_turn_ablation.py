"""Random-turn localization ablation for ALFWorld FutureBridge-OPD."""

import random

from trinity.common.workflows.envs.TCOD.alfworld.futurebridge_workflow import (
    FutureBridgeAlfworldWorkflow,
)


class FutureBridgeRandomTurnAlfworldWorkflow(FutureBridgeAlfworldWorkflow):
    """Randomize only the bridge position and retain the full paired gate."""

    def _select_bridge_candidates(self, turn_responses):
        """Randomize only localization; inherit execution and the paired gate."""
        import re as _re

        def _has_valid_action(resp):
            text = resp.response_text or ""
            return bool(_re.search(r"<action>.*?</action>", text, _re.DOTALL))

        valid_indices = [
            i for i, resp in enumerate(turn_responses[:-1])
            if _has_valid_action(resp)
        ]
        if not valid_indices:
            return []

        n_candidates = min(self.bridge_position_top_k, len(valid_indices))
        task_identity = "/".join(
            str(self.task_desc).replace("\\", "/").split("/")[-3:]
        )
        episode_key = (
            f"{task_identity}:{getattr(self.task, 'batch_id', '')}:"
            f"{getattr(self, 'run_id_base', 0)}"
        )
        rng = random.Random(episode_key)
        return [(idx, 0.0) for idx in rng.sample(valid_indices, n_candidates)]
