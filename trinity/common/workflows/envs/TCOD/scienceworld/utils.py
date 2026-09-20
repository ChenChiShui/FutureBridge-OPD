import json
import re
from typing import Any, Dict, List, Tuple

# --------------------- ScienceWorld --------------------- #
SCIWORLD_SYSTEM_PROMPT = """
You are an expert agent operating in the ScienceWorld text environment.

At each step, you must first reason step-by-step within <think> </think> tags,
then output exactly one environment action within <action> </action> tags.
Do not talk to the user. Solve the task by interacting with the environment.
"""

SCIWORLD_TEMPLATE_NO_HIS = """
Your ScienceWorld task is: {task_description}
Your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.
"""

SCIWORLD_TEMPLATE = """
Your ScienceWorld task is: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.
"""


def _extract_action_like_span(text: str) -> str:
    if not text:
        return ""
    normalized = text.strip()
    if not normalized:
        return ""

    if "<tool_call>" in normalized:
        normalized = normalized.split("<tool_call>", 1)[1]
    normalized = normalized.replace("</tool_call>", " ").strip()

    lines = [ln.strip(" `\"'") for ln in normalized.splitlines() if ln.strip()]
    action_prefixes = (
        "go to ", "open ", "close ", "take ", "put ", "move ",
        "clean ", "heat ", "cool ", "use ", "look", "inventory",
        "examine ", "focus on ", "activate ", "deactivate ",
        "pour ", "mix ", "read ", "teleport ",
    )
    for line in lines:
        lower = line.lower()
        # Some malformed tool-call style outputs can drop the leading motion
        # verb and only keep the destination span, e.g. "<tool_call> to lab".
        # Treat these as navigation-style actions rather than discarding them.
        if lower.startswith("to ") and len(line.split()) <= 16:
            return "go to " + line[3:].strip()
        if lower.startswith(action_prefixes) and len(line.split()) <= 16:
            return line

    lower_text = normalized.lower()
    simple_action_patterns = [
        r'\blet\'s\s+look\s+around\b',
        r'\blook\s+around\b',
        r'["\'](look|inventory)["\']\s+action',
        r'\b(?:use|take|try)\s+the\s+["\']?(look|inventory)["\']?\s+action\b',
        r'\blet\'s\s+(look|inventory)\b',
        r'\b(look|inventory)\b[.!?"\']*$',
    ]
    for pattern in simple_action_patterns:
        m = re.search(pattern, lower_text)
        if m:
            if m.lastindex:
                return m.group(1)
            return "look around"

    regexes = [
        r"\b(go to [a-z0-9_\- ]+)\b",
        r"\b(open [a-z0-9_\- ]+)\b",
        r"\b(close [a-z0-9_\- ]+)\b",
        r"\b(take [a-z0-9_\- ]+? from [a-z0-9_\- ]+)\b",
        r"\b(put [a-z0-9_\- ]+? (?:in|on|to) [a-z0-9_\- ]+)\b",
        r"\b(move [a-z0-9_\- ]+? to [a-z0-9_\- ]+)\b",
        r"\b(clean [a-z0-9_\- ]+? with [a-z0-9_\- ]+)\b",
        r"\b(heat [a-z0-9_\- ]+? with [a-z0-9_\- ]+)\b",
        r"\b(cool [a-z0-9_\- ]+? with [a-z0-9_\- ]+)\b",
        r"\b(use [a-z0-9_\- ]+? on [a-z0-9_\- ]+)\b",
        r"\b(activate [a-z0-9_\- ]+)\b",
        r"\b(deactivate [a-z0-9_\- ]+)\b",
        r"\b(read [a-z0-9_\- ]+)\b",
        r"\b(mix [a-z0-9_\- ]+)\b",
        r"\b(pour [a-z0-9_\- ]+? (?:into|in|on) [a-z0-9_\- ]+)\b",
    ]
    candidates = []
    for pattern in regexes:
        for m in re.finditer(pattern, lower_text):
            candidates.append((m.start(), m.group(1).strip(" .,!?:;`\"'")))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    return ""


def parse_action(response: str) -> str:
    try:
        if not response:
            return ""
        if "<action>" in response:
            tail = response.split("<action>", 1)[1]
            action = tail.split("</action>", 1)[0].strip()
            if action:
                return action
        action = _extract_action_like_span(response)
        if action:
            return action
        raise ValueError("no action-like span found")
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""


def format_observation(observation: str) -> str:
    return observation.strip()


HISTORY_LENGTH = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def _extract_task(task_description: str) -> str:
    return task_description.strip()


def _format_history(observation: str, step_num: int, act: str) -> str:
    return MEMORY_FORMAT.format(step_num=step_num, obs=observation.strip(), act=act)


def _parse_task_config(task_desc: Any) -> Dict[str, Any]:
    if isinstance(task_desc, dict):
        return task_desc
    if isinstance(task_desc, str):
        return json.loads(task_desc)
    raise TypeError(f"Unsupported ScienceWorld task description type: {type(task_desc)}")


def _get_admissible_commands(info: Dict[str, Any]) -> List[str]:
    commands = info.get("valid", [])
    if isinstance(commands, str):
        return [commands] if commands else []
    if commands and isinstance(commands[0], list):
        commands = commands[0]
    return [cmd for cmd in commands if cmd]


def _get_compact_action_info(env) -> Tuple[List[str], List[str]]:
    """Get compact action representation: templates + objects separately.

    ScienceWorld's get_valid_action_object_combinations() returns ~1162 full
    combinations per step, which easily exceeds token limits. Instead, we use
    get_possible_actions() (~23 templates) and get_possible_objects() (~20 objects)
    which together are ~50 items — about 25x smaller.
    """
    try:
        actions = env.get_possible_actions()
    except AttributeError:
        actions = env.getPossibleActions()

    try:
        objects = env.get_possible_objects()
    except AttributeError:
        objects = env.getPossibleObjects()

    if isinstance(actions, str):
        actions = [actions] if actions else []
    if isinstance(objects, str):
        objects = [objects] if objects else []

    return actions, objects


def _create_scienceworld_env(
    task_desc: Any,
    *,
    max_env_steps: int = 30,
    generate_gold_path: bool = False,
):
    task_config = _parse_task_config(task_desc)

    try:
        from scienceworld import ScienceWorldEnv
    except Exception as e:
        raise ImportError(
            f"Error importing ScienceWorldEnv: {e}. "
            "Please make sure the scienceworld package is installed successfully: "
            "https://github.com/allenai/ScienceWorld"
        ) from e

    task_name = task_config["task_name"]
    var_num = task_config["var_num"]
    jar_path = task_config.get("jar_path", "")
    simplification_str = task_config.get("simplification_str", "")
    env_step_limit = max(task_config.get("env_step_limit", 100), max_env_steps + 5)

    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    env.load(task_name, var_num, simplification_str, generateGoldPath=generate_gold_path)
    return env


def _reset_scienceworld_env(env) -> Tuple[str, Dict[str, Any], str]:
    observation, info = env.reset()
    task_description = _extract_task(env.get_task_description())
    return observation, info, task_description


def _create_scienceworld_env_with_checkpoint(
    task_desc: Any,
    actions: List[str],
    checkpoint_step: int,
    *,
    max_env_steps: int = 30,
):
    env = _create_scienceworld_env(
        task_desc,
        max_env_steps=max_env_steps,
        generate_gold_path=True,
    )
    observation, info, task_description = _reset_scienceworld_env(env)

    history: List[str] = []
    best_score = info.get("score", 0)

    done = False
    for step in range(min(checkpoint_step, len(actions))):
        action = actions[step]
        format_obs = format_observation(observation)
        history.append(_format_history(format_obs, step + 1, action))
        observation, reward, done, info = env.step(action)
        best_score = max(best_score, info.get("score", best_score + reward))
        if done:
            break

    return (
        env,
        observation,
        info,
        history,
        task_description,
        len(history),
        done,
        best_score / 100.0,
    )
