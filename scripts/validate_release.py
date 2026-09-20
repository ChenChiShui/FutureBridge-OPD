#!/usr/bin/env python3
"""Static release checks that do not require the benchmark environments."""

import ast
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "trinity"
JUNK_SUFFIXES = {
    ".csv",
    ".db",
    ".jsonl",
    ".log",
    ".npy",
    ".npz",
    ".pdf",
    ".pyc",
    ".sqlite",
    ".tsv",
}
TEXT_SUFFIXES = {"", ".md", ".py", ".sh", ".tex", ".txt", ".yaml", ".yml"}


def main() -> None:
    for relative_path in (
        "fig/motivation.png",
        "fig/pipeline.png",
        "requirements_freeze.txt",
    ):
        assert (ROOT / relative_path).is_file(), f"missing release file: {relative_path}"

    requirements = (ROOT / "requirements_freeze.txt").read_text(encoding="utf-8")
    for dependency in (
        "verl==",
        "ray==",
        "transformers==",
        "datasets==",
        "vllm==0.8.5.post1",
        "flash_attn",
    ):
        assert dependency in requirements, f"missing dependency: {dependency}"

    classes = {}
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        classes[path] = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }

    names = set()
    buffers = set()
    runnable = 0
    for path in sorted((ROOT / "configs").glob("*/*.yaml")):
        if path.parent.name == "rl_teacher":
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        runnable += 1

        name = data["name"]
        assert name not in names, f"duplicate experiment name: {name}"
        names.add(name)

        explorer = data["buffer"]["explorer_input"]
        module, class_name = explorer["default_workflow_type"].rsplit(".", 1)
        module_path = SOURCE_ROOT / Path(*module.split(".")).with_suffix(".py")
        assert class_name in classes.get(module_path, set()), (
            f"{path}: missing workflow {module}.{class_name}"
        )

        buffer_name = data["buffer"]["trainer_input"]["experience_buffer"]["name"]
        assert buffer_name not in buffers, f"duplicate buffer: {buffer_name}"
        buffers.add(buffer_name)

        assert data["model"]["max_prompt_tokens"] == 10240
        assert data["model"]["max_response_tokens"] == 512
        assert len(explorer["eval_tasksets"]) == 1, f"{path}: expected one eval taskset"
        eval_taskset = explorer["eval_tasksets"][0]
        assert eval_taskset["task_selector"]["selector_type"] == "sequential", (
            f"{path}: evaluation must use sequential task selection"
        )
        expected_eval_steps = {
            "alfworld": 134,
            "webshop": 100,
            "scienceworld": 1308,
        }[path.parent.name]
        assert eval_taskset["total_steps"] == expected_eval_steps, (
            f"{path}: expected {expected_eval_steps} evaluation tasks"
        )
        eval_args = eval_taskset["rollout_args"]
        assert eval_args["temperature"] == 0.4
        assert eval_args["max_tokens"] == 512
        assert "enable_thinking" not in eval_args
        assert data["explorer"]["eval_on_startup"] is False

        if path.name != "bridge_token_gate_ablation.yaml":
            workflow_args = explorer["taskset"]["workflow_args"]
            assert workflow_args["temperature"] == 1.0
            assert workflow_args["bridge_position_top_k"] == 1
            assert workflow_args["continuation_steps"] == 3
            assert workflow_args["bridge_failed_episodes_only"] is False
            assert workflow_args["bridge_require_full_continuation"] is True
            if path.parent.name == "webshop":
                assert workflow_args["b2f_prefix_source"] == "reference"
        else:
            workflow_args = explorer["taskset"]["workflow_args"]
            assert workflow_args["bridge_failed_episodes_only"] is False

    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in SOURCE_ROOT.rglob("*.py")
    )
    assert not re.search(r"""temperature["']?\s*[:=]\s*0\.0""", source_text)
    assert "GuidedOPD-DEBUG" not in source_text
    assert "% n_goals" not in source_text
    webshop_ftb = (
        SOURCE_ROOT
        / "common/workflows/envs/TCOD/webshop/futurebridge_workflow.py"
    ).read_text(encoding="utf-8")
    assert "_run_teacher_phase" not in webshop_ftb
    assert "prefix_actions=self._reference_prefix_actions" in webshop_ftb

    teacher_root = ROOT / "configs" / "rl_teacher"
    teacher_shared = yaml.safe_load(
        (teacher_root / "shared.yaml").read_text(encoding="utf-8")
    )
    teacher_webshop = yaml.safe_load(
        (teacher_root / "webshop.yaml").read_text(encoding="utf-8")
    )
    teacher_alfworld = yaml.safe_load(
        (teacher_root / "alfworld.yaml").read_text(encoding="utf-8")
    )
    assert teacher_shared["training"] == {"temperature": 1.0, "sampling": True}
    assert teacher_shared["validation"] == {"temperature": 0.4, "sampling": True}
    assert teacher_shared["algorithm"] == "GiGPO"
    assert teacher_shared["base_model"] == "Qwen3-8B"
    assert teacher_shared["rollout_group_size"] == 8
    assert teacher_webshop["train_batch_size"] == 16
    assert teacher_webshop["validation_batch_size"] == 32
    assert teacher_webshop["max_env_steps"] == 15
    assert teacher_webshop["logprob_micro_batch_size_per_gpu"] == 8
    assert teacher_webshop["tensor_parallel_size"] == 2
    assert teacher_alfworld["train_batch_size"] == 32
    assert teacher_alfworld["max_env_steps"] == 50
    assert teacher_alfworld["logprob_micro_batch_size_per_gpu"] == 8
    assert teacher_alfworld["tensor_parallel_size"] == 2

    teacher_launcher = teacher_root / "train_webshop.sh"
    assert teacher_launcher.is_file()
    launcher_text = teacher_launcher.read_text(encoding="utf-8")
    for placeholder in (
        "GIGPO_DIR",
        "MODEL_PATH",
        "TRAIN_DATA",
        "VAL_DATA",
        "OUTPUT_DIR",
    ):
        assert placeholder in launcher_text
    for setting in (
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "trainer.total_epochs=100",
    ):
        assert setting in launcher_text
    assert not re.search(r"/(?:Users|home|mnt|root)/", launcher_text)

    latex_table = (teacher_root / "rl_teacher_settings.tex").read_text(
        encoding="utf-8"
    )
    assert "Log-prob micro-batch size per GPU\n& 8\n& 8" in latex_table
    assert "Tensor parallel size\n& 2\n& 2" in latex_table

    junk = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in JUNK_SUFFIXES or path.name == ".DS_Store")
    ]
    assert not junk, f"release contains generated/result files: {junk}"

    if (ROOT / "ANONYMITY.md").exists():
        absolute_roots = [
            "/" + "Users" + "/",
            "/" + "home" + "/",
            "/" + "mnt" + "/",
            "/" + "root" + "/",
        ]
        private = re.compile(
            "|".join(re.escape(value) for value in absolute_roots)
            + r"|https?://|"
            r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
            re.IGNORECASE,
        )
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
                assert not private.search(text), f"non-anonymous text in {path}"

    print(f"validated {runnable} runnable configurations")


if __name__ == "__main__":
    main()
