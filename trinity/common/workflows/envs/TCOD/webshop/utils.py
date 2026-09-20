# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Optional


# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

HISTORY_LENGTH = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def parse_action(response: str) -> str:
    try:
        if not response:
            return ""
        if "<action>" in response:
            return response.split("<action>", 1)[1].split("</action>", 1)[0].strip()

        normalized = response.strip()
        if "<tool_call>" in normalized:
            normalized = normalized.split("<tool_call>", 1)[1]
        normalized = normalized.replace("</tool_call>", " ").strip()

        for line in normalized.splitlines():
            line = line.strip(" `\"'")
            if not line:
                continue
            if re.match(r"^(click\[[^\]]+\]|search\[[^\]]+\])$", line):
                return line

        m = re.search(r"(click\[[^\]]+\]|search\[[^\]]+\])", normalized)
        if m:
            return m.group(1).strip()

        raise ValueError("no action-like span found")
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""


def validate_action(action: str, available_actions: dict):
    pattern = re.compile(r"(.+)\[(.+)\]")
    match = re.match(pattern, action)
    if match is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = match.groups()

    if action_arg is not None:
        action_arg = action_arg.lower()

    if (
        action_name == "search"
        and action_arg is not None
        and action_arg != ""
        and available_actions.get("has_search_bar", False)
    ):
        return True, ""
    if action_name == "click" and action_arg in available_actions.get("clickables", []):
        return True, ""

    if action_name == "search":
        if action_arg == "" or action_arg is None:
            return (
                False,
                "Invalid action, please type in the query you want to search in the square brackets here.",
            )
        return (
            False,
            "Can not perform search action without search bar. Please click the Back to Search button first.",
        )

    if action_name == "click" and action_arg not in available_actions.get("clickables", []):
        return (
            False,
            "Incorrect action format, make sure you have the correct button name that is "
            f"within the current page. The buttons you can click now are: {available_actions.get('clickables', [])}",
        )

    return (
        False,
        "Invalid action. You should wrap your action with the <action> </action> tag and follow the action format.",
    )


def format_observation(observation: str) -> str:
    return observation.strip()


def _compact_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_task_description(observation: str) -> str:
    """Extract the instruction text from a WebShop observation."""
    lines = [line.strip() for line in observation.splitlines()]
    try:
        instruction_idx = lines.index("Instruction:")
    except ValueError:
        return _compact_text(observation)

    task_lines = []
    for line in lines[instruction_idx + 1 :]:
        if not line:
            continue
        if line.startswith("[button]"):
            break
        task_lines.append(line)

    return " ".join(task_lines).strip() or _compact_text(observation)


def _format_history(observation: str, step_num: int, action: str) -> str:
    return MEMORY_FORMAT.format(
        step_num=step_num,
        obs=_compact_text(observation),
        act=action,
    )


def _format_available_actions(available_actions: dict) -> str:
    formatted_actions = []
    if available_actions.get("has_search_bar", False):
        formatted_actions.append("search[<query>]")
    for clickable in available_actions.get("clickables", []):
        formatted_actions.append(f"click[{clickable}]")
    return "\n".join(formatted_actions)


def _create_webshop_env():
    try:
        import gym  # type: ignore[import-not-found]
        from web_agent_site.envs import WebAgentTextEnv  # type: ignore[import-not-found]  # noqa: F401
    except Exception as e:
        raise ImportError(
            f"Error importing WebAgentTextEnv {e}. "
            "Please make sure you have installed the web_agent_site package, "
            "following the instructions in https://github.com/princeton-nlp/WebShop"
        ) from e

    # human_goals=False: load from product database (supports task_ids 0-4095)
    # num_products=None: use full product index (the only option that covers all sessions)
    return gym.make(
        "WebAgentTextEnv-v0",
        observation_mode="text_rich",
        human_goals=False,
    )


def _create_webshop_env_with_checkpoint(
    session_id: int,
    actions: List[str],
    checkpoint_step: int,
):
    """Create WebShop environment and execute actions up to checkpoint_step."""
    env = _create_webshop_env()
    env.reset(session=session_id)
    observation = env.observation
    history: List[str] = []
    task_description = _extract_task_description(observation)
    done = False
    reward = -0.1

    for step in range(min(checkpoint_step, len(actions))):
        action = actions[step]
        formatted_observation = format_observation(observation)
        history.append(_format_history(formatted_observation, step + 1, action))
        observation, reward, done, _ = env.step(action)
        if done:
            break

    return env, observation, history, task_description, len(history), done, reward
