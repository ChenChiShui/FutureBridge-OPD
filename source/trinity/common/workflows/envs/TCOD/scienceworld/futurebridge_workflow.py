"""FutureBridge-OPD workflows for ScienceWorld."""

import copy
import json
import logging
import random
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.scienceworld.TCOD_b2f_workflow import (
    TCOD_b2f_scienceworld_workflow,
)
from trinity.common.workflows.envs.TCOD.scienceworld.utils import (
    HISTORY_LENGTH,
    SCIWORLD_TEMPLATE,
    SCIWORLD_TEMPLATE_NO_HIS,
    _create_scienceworld_env_with_checkpoint,
    _format_history,
    _get_admissible_commands,
    _get_compact_action_info,
    format_observation,
    parse_action,
)

logger = logging.getLogger(__name__)

BRIDGE_STEP_OFFSET = 5000


def _is_valid_action(env, info, action: str) -> bool:
    """Check an action against the exact commands available in this state."""
    valid = _get_admissible_commands(info)
    if not valid:
        try:
            valid = env.get_valid_action_object_combinations()
        except AttributeError:
            valid = env.getValidActionObjectCombinations()
    return bool(action) and action in valid


def _segment_pos_ratio(responses: List[Experience]) -> Optional[float]:
    """Token-pooled teacher-preference ratio for an observed segment."""
    positive = 0
    total = 0
    for response in responses:
        if response.teacher_logprobs is None or response.logprobs is None:
            return None
        advantage = (
            response.teacher_logprobs.float() - response.logprobs.float()
        )
        positive += int((advantage > 0).sum())
        total += len(advantage)
    return positive / total if total > 0 else None


@WORKFLOWS.register_module("FutureBridgeScienceWorldWorkflow")
class FutureBridgeScienceWorldWorkflow(TCOD_b2f_scienceworld_workflow):
    """ScienceWorld implementation of the paper's main method."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models=None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.continuation_steps = int(
            task.workflow_args.get("continuation_steps", 3)
        )
        self.bridge_position_top_k = max(
            1, int(task.workflow_args.get("bridge_position_top_k", 1))
        )
        self.bridge_max_per_ep = max(
            1, int(task.workflow_args.get("bridge_max_per_ep", 1))
        )
        self.bridge_reward_threshold = float(
            task.workflow_args.get("bridge_reward_threshold", 0.5)
        )
        self.bridge_failed_episodes_only = bool(
            task.workflow_args.get("bridge_failed_episodes_only", False)
        )
        self.bridge_require_full_continuation = bool(
            task.workflow_args.get("bridge_require_full_continuation", True)
        )

    def _select_bridge_candidates(self, turn_responses):
        """Return non-final turns ranked by token-average disagreement."""
        candidates = []
        for turn_idx, response in enumerate(turn_responses[:-1]):
            if response.logprobs is None or response.teacher_logprobs is None:
                continue
            if not parse_action(response.response_text or ""):
                continue
            disagreement = (
                response.logprobs - response.teacher_logprobs
            ).float().mean().item()
            candidates.append((turn_idx, disagreement))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:self.bridge_position_top_k]

    async def _finalize_turn_responses(
        self,
        turn_responses: List[Experience],
        *,
        start_step: int,
        memory_snapshots: Optional[List[List[dict]]] = None,
    ) -> List[Experience]:
        normal_exps = await super()._finalize_turn_responses(
            turn_responses,
            start_step=start_step,
            memory_snapshots=memory_snapshots,
        )
        if (
            self.is_eval
            or not self._expert_actions
            or len(normal_exps) < 2
            or not memory_snapshots
        ):
            return normal_exps
        if (
            self.bridge_failed_episodes_only
            and self._final_reward >= self.bridge_reward_threshold
        ):
            return normal_exps

        bridge_exps = await self._try_bridge(
            normal_exps,
            memory_snapshots,
            start_step,
        )
        return normal_exps + bridge_exps

    async def _try_bridge(
        self,
        turn_responses: List[Experience],
        memory_snapshots: List[List[dict]],
        start_step: int,
    ) -> List[Experience]:
        candidates = self._select_bridge_candidates(turn_responses)
        if not candidates:
            return []

        accepted = []
        for bridge_rank, (turn_idx, disagreement) in enumerate(candidates):
            if bridge_rank >= self.bridge_max_per_ep:
                break

            bridge_exp = await self._generate_bridge(
                memory_snapshots[turn_idx],
                disagreement,
                start_step + turn_idx,
            )
            if bridge_exp is None:
                return []

            base_exp = turn_responses[turn_idx]
            base_action = parse_action(base_exp.response_text or "")
            bridge_action = parse_action(bridge_exp.response_text or "")
            if not base_action or not bridge_action:
                return []

            is_accepted = await self._validate_bridge(
                turn_responses=turn_responses,
                memory_at_turn=memory_snapshots[turn_idx],
                start_step=start_step,
                turn_idx=turn_idx,
                base_exp=base_exp,
                base_action=base_action,
                bridge_exp=bridge_exp,
                bridge_action=bridge_action,
            )
            if is_accepted is None:
                return []
            if is_accepted:
                accepted.append(bridge_exp)

        return accepted

    async def _validate_bridge(
        self,
        *,
        turn_responses,
        memory_at_turn,
        start_step,
        turn_idx,
        base_exp,
        base_action,
        bridge_exp,
        bridge_action,
    ) -> Optional[bool]:
        """Strict paired gate; failure of either branch rejects the bridge."""
        base_ratio = await self._continuation_ratio(
            turn_responses=turn_responses,
            start_step=start_step,
            turn_idx=turn_idx,
            branch_action=base_action,
            branch_memory=memory_at_turn + [
                {"role": "assistant", "content": base_exp.response_text or ""}
            ],
        )
        if base_ratio is None:
            return None

        bridge_ratio = await self._continuation_ratio(
            turn_responses=turn_responses,
            start_step=start_step,
            turn_idx=turn_idx,
            branch_action=bridge_action,
            branch_memory=memory_at_turn + [
                {"role": "assistant", "content": bridge_exp.response_text or ""}
            ],
        )
        if bridge_ratio is None:
            return None

        logger.debug(
            "[scienceworld-futurebridge] turn=%s bridge=%.3f base=%.3f",
            turn_idx,
            bridge_ratio,
            base_ratio,
        )
        return bridge_ratio > base_ratio

    async def _generate_bridge(
        self,
        memory_at_turn: List[dict],
        disagreement: float,
        bridge_idx: int,
    ) -> Optional[Experience]:
        kwargs = {
            **self.rollout_args,
            "n": 1,
            "logprobs": 0,
            "temperature": self.temperature,
        }
        try:
            bridge_resp = (await self.teacher_model.chat_async(
                memory_at_turn, **kwargs
            ))[0]
            full_tokens = bridge_resp.tokens.tolist()
            response_start = bridge_resp.prompt_length - 1
            student_full = await self.model.logprobs_async(
                tokens=full_tokens,
                temperature=self.temperature,
            )
            teacher_full = await self.teacher_model.logprobs_async(
                tokens=full_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            logger.debug("[scienceworld-futurebridge] bridge generation failed: %s", exc)
            return None

        student_lp = student_full[response_start:]
        teacher_lp = teacher_full[response_start:]
        if len(student_lp) == 0 or len(student_lp) != len(teacher_lp):
            return None

        experience = copy.copy(bridge_resp)
        experience.logprobs = student_lp
        experience.teacher_logprobs = teacher_lp
        experience.reward = 0.0
        experience.eid.run = getattr(self, "run_id_base", 0)
        experience.eid.step = BRIDGE_STEP_OFFSET + bridge_idx
        if experience.metrics is None:
            experience.metrics = {}
        experience.metrics.update(
            {
                "bridge_verified": 1,
                "trigger_kl": disagreement,
                "is_bridge": 1,
            }
        )
        return experience

    async def _continuation_ratio(
        self,
        *,
        turn_responses: List[Experience],
        start_step: int,
        turn_idx: int,
        branch_action: str,
        branch_memory: List[dict],
    ) -> Optional[float]:
        env = None
        try:
            (
                env,
                observation,
                info,
                history,
                task_description,
                actual_step,
                done,
                _,
            ) = _create_scienceworld_env_with_checkpoint(
                self.task_desc,
                self._expert_actions,
                start_step,
                max_env_steps=self.max_env_steps,
            )
            if done:
                return None

            for response in turn_responses[:turn_idx]:
                action = parse_action(response.response_text or "")
                if not _is_valid_action(env, info, action):
                    return None
                formatted_obs = format_observation(observation)
                history.append(_format_history(formatted_obs, actual_step + 1, action))
                observation, _, done, info = env.step(action)
                actual_step += 1
                if done:
                    return None

            if not _is_valid_action(env, info, branch_action):
                return None
            formatted_obs = format_observation(observation)
            history.append(
                _format_history(formatted_obs, actual_step + 1, branch_action)
            )
            observation, _, done, info = env.step(branch_action)
            actual_step += 1
            if done:
                return (
                    None
                    if self.bridge_require_full_continuation
                    else 1.0
                )

            memory = list(branch_memory)
            kwargs = {
                **self.rollout_args,
                "n": 1,
                "logprobs": 0,
                "temperature": self.temperature,
            }
            positive = 0
            total = 0

            for continuation_idx in range(self.continuation_steps):
                formatted_obs = format_observation(observation)
                action_templates, objects = _get_compact_action_info(env)
                action_text = ", ".join(
                    f"'{action}'"
                    for action in action_templates
                    if action != "help"
                )
                object_text = ", ".join(f"'{obj}'" for obj in objects)

                if len(history) < HISTORY_LENGTH:
                    user_content = SCIWORLD_TEMPLATE_NO_HIS.format(
                        task_description=task_description,
                        current_observation=formatted_obs,
                        action_templates=action_text,
                        objects=object_text,
                    )
                else:
                    user_content = SCIWORLD_TEMPLATE.format(
                        task_description=task_description,
                        step_count=actual_step,
                        history_length=min(HISTORY_LENGTH, len(history)),
                        action_history="\n".join(history[-HISTORY_LENGTH:]),
                        current_step=actual_step + 1,
                        current_observation=formatted_obs,
                        action_templates=action_text,
                        objects=object_text,
                    )

                messages = memory + [{"role": "user", "content": user_content}]
                response = (await self.model.chat_async(messages, **kwargs))[0]
                if response.logprobs is None:
                    return None
                full_tokens = response.tokens.tolist()
                response_start = response.prompt_length - 1
                teacher_full = await self.teacher_model.logprobs_async(
                    tokens=full_tokens,
                    temperature=self.temperature,
                )
                teacher_lp = teacher_full[response_start:]
                if len(teacher_lp) != len(response.logprobs):
                    return None

                advantage = teacher_lp.float() - response.logprobs.float()
                positive += int((advantage > 0).sum())
                total += len(advantage)

                response_text = response.response_text or ""
                action = parse_action(response_text)
                if not _is_valid_action(env, info, action):
                    return None
                memory.extend(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": response_text},
                    ]
                )
                history.append(
                    _format_history(formatted_obs, actual_step + 1, action)
                )
                observation, _, done, info = env.step(action)
                actual_step += 1
                if done:
                    if (
                        self.bridge_require_full_continuation
                        and continuation_idx + 1 < self.continuation_steps
                    ):
                        return None
                    break

            return positive / total if total > 0 else None
        except Exception as exc:
            logger.debug("[scienceworld-futurebridge] continuation failed: %s", exc)
            return None
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


@WORKFLOWS.register_module(
    "FutureBridgeNoFutureValidationScienceWorldWorkflow"
)
class FutureBridgeNoFutureValidationScienceWorldWorkflow(
    FutureBridgeScienceWorldWorkflow
):
    """Execute the bridge continuation but omit the paired-ratio filter."""

    async def _validate_bridge(self, **kwargs) -> Optional[bool]:
        ratio = await self._continuation_ratio(
            turn_responses=kwargs["turn_responses"],
            start_step=kwargs["start_step"],
            turn_idx=kwargs["turn_idx"],
            branch_action=kwargs["bridge_action"],
            branch_memory=kwargs["memory_at_turn"] + [
                {
                    "role": "assistant",
                    "content": kwargs["bridge_exp"].response_text or "",
                }
            ],
        )
        return None if ratio is None else True


@WORKFLOWS.register_module("FutureBridgeRandomTurnScienceWorldWorkflow")
class FutureBridgeRandomTurnScienceWorldWorkflow(
    FutureBridgeScienceWorldWorkflow
):
    """Randomize only localization and retain the full paired gate."""

    def _select_bridge_candidates(self, turn_responses):
        valid = [
            idx
            for idx, response in enumerate(turn_responses[:-1])
            if parse_action(response.response_text or "")
        ]
        if not valid:
            return []
        count = min(self.bridge_position_top_k, len(valid))
        try:
            task_config = (
                self.task_desc
                if isinstance(self.task_desc, dict)
                else json.loads(self.task_desc)
            )
            task_identity = (
                f"{task_config['task_name']}:{task_config['var_num']}"
            )
        except (KeyError, TypeError, ValueError):
            task_identity = str(self.task_desc)
        episode_key = (
            f"{task_identity}:{getattr(self.task, 'batch_id', '')}:"
            f"{getattr(self, 'run_id_base', 0)}"
        )
        rng = random.Random(episode_key)
        return [(idx, 0.0) for idx in rng.sample(valid, count)]


@WORKFLOWS.register_module("FutureBridgeNoBridgeExecutionScienceWorldWorkflow")
class FutureBridgeNoBridgeExecutionScienceWorldWorkflow(
    FutureBridgeScienceWorldWorkflow
):
    """Select a teacher action without executing it in the environment."""

    async def _validate_bridge(self, **kwargs) -> Optional[bool]:
        turn_idx = kwargs["turn_idx"]
        turn_responses = kwargs["turn_responses"]
        before = _segment_pos_ratio(turn_responses[:turn_idx + 1])
        after = _segment_pos_ratio(turn_responses[turn_idx + 1:])
        if before is None or after is None:
            return None
        return after < before
