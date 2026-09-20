import re
from typing import List

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

def _extract_action_like_span(text: str) -> str:
    """Best-effort fallback for non-canonical model outputs.

    The ideal format is <action>...</action>, but in practice the teacher can
    occasionally emit <tool_call> blocks or plain-text sentences such as
    "Let's go to cabinet 4." during offline probing. We only recover short,
    action-like spans and otherwise return an empty string.
    """
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
        "pour ", "mix ", "read ",
    )

    for line in lines:
        lower = line.lower()
        # Some malformed tool-call outputs drop the leading verb and only keep
        # the destination span, e.g. "<tool_call> to sinkbasin 1". In
        # ALFWorld this should be interpreted as a navigation action.
        if lower.startswith("to ") and len(line.split()) <= 12:
            return "go to " + line[3:].strip()
        if lower.startswith(action_prefixes) and len(line.split()) <= 12:
            return line

    lower_text = normalized.lower()
    simple_action_patterns = [
        r'["\'](look|inventory)["\']\s+action',
        r'\b(?:use|take|try)\s+the\s+["\']?(look|inventory)["\']?\s+action\b',
        r'\blet\'s\s+(look|inventory)\b',
        r'\blet\'s\s+look\s+around\b',
        r'\blook\s+around\b',
        r'\b(look|inventory)\b[.!?"\']*$',
    ]
    for pattern in simple_action_patterns:
        m = re.search(pattern, lower_text)
        if m:
            if m.lastindex:
                return m.group(1)
            return "look around"

    explicit_nav = []
    for pattern in [
        r"\b(?:proceed|head|walk|move)\s+to[^.]{0,80}?\b((?:cabinet|drawer|countertop|sinkbasin|fridge|microwave|diningtable|sidetable|dresser|sofa|coffeetable|bed|toilet|garbagecan|shelf)\s+\d+)\b",
    ]:
        for m in re.finditer(pattern, lower_text):
            explicit_nav.append((m.start(), "go to " + m.group(1).strip(" .,!?:;`\"'")))
    if explicit_nav:
        explicit_nav.sort(key=lambda x: x[0])
        return explicit_nav[-1][1]

    nav_aliases = []
    for pattern in [
        r"\b(?:proceed|head|walk)\s+to\s+([a-z0-9_\- ]+)\b",
    ]:
        for m in re.finditer(pattern, lower_text):
            nav_aliases.append((m.start(), "go to " + m.group(1).strip(" .,!?:;`\"'")))
    if nav_aliases:
        nav_aliases.sort(key=lambda x: x[0])
        return nav_aliases[-1][1]

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
    ]
    candidates = []
    for pattern in regexes:
        for m in re.finditer(pattern, lower_text):
            candidates.append((m.start(), m.group(1).strip(" .,!?:;`\"'")))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    return ""


def parse_action(response):
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

def format_observation(observation: str):
    return observation.strip()


HISTORY_LENGTH = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def _extract_task(text_obs: str) -> str:
    """Extract the task description from the text observation."""
    task_start = text_obs.find("Your task is to: ")
    if task_start != -1:
        return text_obs[task_start + len("Your task is to: ") :].strip()
    raise ValueError("Task description not found in text observation.")


def _format_history(observation: str, step_num: int, act: str) -> str:
    """Format observation and action for action_history."""
    return MEMORY_FORMAT.format(step_num=step_num, obs=observation.strip(), act=act)


def _create_alfworld_env(game_file_path: str):
    """Create AlfWorld textworld environment."""
    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredExpert,
            AlfredExpertType,
        )

        expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
        request_infos = textworld.EnvInfos(
            description=True, inventory=True, admissible_commands=True
        )
        env_id = textworld.gym.register_game(
            game_file_path, request_infos, wrappers=[AlfredDemangler(), expert]
        )
        return textworld.gym.make(env_id)
    except Exception as e:
        raise ImportError(
            f"Error creating AlfWorld env: {e}. "
            "Ensure alfworld is installed: https://github.com/alfworld/alfworld"
        ) from e


def _create_alfworld_env_with_checkpoint(
    game_file_path: str, actions: List[str], checkpoint_step: int
):
    """Create AlfWorld environment and execute actions up to checkpoint_step.

    Args:
        game_file_path: Path to the game file
        actions: List of predefined expert actions
        checkpoint_step: Number of actions to execute before letting model take over

    Returns:
        tuple: (
            env,
            observation,
            info,
            history,
            task_description,
            current_step,
            done,
        )
    """
    env = _create_alfworld_env(game_file_path)
    observation, info = env.reset()

    task_description = _extract_task(observation)
    history: List[str] = []

    # Execute predefined actions up to checkpoint_step
    done = False
    for step in range(min(checkpoint_step, len(actions))):
        action = actions[step]
        format_obs = format_observation(observation)
        history.append(_format_history(format_obs, step + 1, action))
        observation, reward, done, info = env.step(action)

        if done:
            # If task completes before checkpoint, return the final state
            break

    return env, observation, info, history, task_description, len(history), done
