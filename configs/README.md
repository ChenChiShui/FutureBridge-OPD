# Configuration guide

## Main experiments

| Benchmark | Configuration | Student | Teacher |
| --- | --- | --- | --- |
| ALFWorld | `alfworld/ftb.yaml` | Qwen3-1.7B | Qwen3-32B |
| WebShop | `webshop/ftb.yaml` | Qwen3-1.7B | Qwen3-32B |
| ScienceWorld | `scienceworld/ftb.yaml` | Qwen3-1.7B | Qwen3-32B |
| ALFWorld | `alfworld/ftb_qwen3_32b_to_4b.yaml` | Qwen3-4B | Qwen3-32B |
| WebShop | `webshop/ftb_qwen3_32b_to_4b.yaml` | Qwen3-4B | Qwen3-32B |
| ScienceWorld | `scienceworld/ftb_qwen3_32b_to_4b.yaml` | Qwen3-4B | Qwen3-32B |
| ALFWorld, RL teacher | `alfworld/ftb_rl_teacher.yaml` | Qwen3-4B | Qwen3-8B RL |
| WebShop, RL teacher | `webshop/ftb_rl_teacher.yaml` | Qwen3-4B | Qwen3-8B RL |
| ScienceWorld, RL teacher | `scienceworld/ftb_rl_teacher.yaml` | Qwen3-4B | Qwen3-8B RL |

## Ablations

| Ablation | Configuration |
| --- | --- |
| Random bridge turn, ALFWorld | `alfworld/random_turn_ablation.yaml` |
| Random bridge turn, WebShop | `webshop/random_turn_ablation.yaml` |
| Random bridge turn, ScienceWorld | `scienceworld/random_turn_ablation.yaml` |
| No bridge execution, ALFWorld | `alfworld/no_bridge_execution_ablation.yaml` |
| No bridge execution, WebShop | `webshop/no_bridge_execution_ablation.yaml` |
| No bridge execution, ScienceWorld | `scienceworld/no_bridge_execution_ablation.yaml` |
| No future-validation filter, ALFWorld | `alfworld/no_future_validation_ablation.yaml` |
| No future-validation filter, WebShop | `webshop/no_future_validation_ablation.yaml` |
| No future-validation filter, ScienceWorld | `scienceworld/no_future_validation_ablation.yaml` |
| Bridge-token gate | `alfworld/bridge_token_gate_ablation.yaml` |

The bridge-token gate is an additional ablation, not the main method.
The paper's RL-Teacher and component-ablation tables report ALFWorld and
WebShop. The corresponding ScienceWorld YAML files are provided as additional
runnable extensions and are not presented as paper-result configurations.

`bridge_position_top_k` controls how many disagreement-ranked positions may be
attempted. Every paper configuration sets it to one. The selected score is the
token-average sampled log-probability ratio and the final Student turn is
excluded. `bridge_max_per_ep` separately limits how many bridges may be
retained.

The paper configurations also make two validation semantics explicit:

- `bridge_failed_episodes_only: false` disables reward/success filtering of
  bridge candidates.
- `bridge_require_full_continuation: true` rejects terminal shortcuts and
  continuations with fewer than `continuation_steps` Student turns.

Setting either option to its legacy value changes the auxiliary-training
distribution and is not a paper configuration.

## Evaluation sets

- ALFWorld uses all 134 tasks in the unseen split and at most 30 environment
  interactions per episode.
- WebShop uses the 100 held-out sessions 4096 through 4195 and at most 15
  interactions per episode.
- ScienceWorld uses all 1,308 entries in the disjoint task-type test split and
  at most 30 interactions per episode.

The large `eval_interval` disables expensive evaluation during optimization;
run the configured evaluation task set once after training.

## RL teachers

The exact appendix hyperparameters for the external GiGPO teacher-training
stage are recorded under `configs/rl_teacher/`. A sanitized, portable
transcription of the recovered WebShop entry point is included there; the
ALFWorld file is a framework-neutral parameter record. The copy-ready LaTeX
table is `configs/rl_teacher/rl_teacher_settings.tex`.

## Baseline workflow mappings

Use the same paper settings and replace `default_workflow_type` with the
appropriate dotted class path:

| Method | Suffix under `trinity.common.workflows.envs.TCOD.<benchmark>` |
| --- | --- |
| OPD | `OPD_workflow.OnPolicyDistillVerlAgent<Benchmark>Workflow` |
| TCOD-F2B | `TCOD_f2b_workflow.TCOD_f2b_<benchmark>_workflow` |
| TCOD-B2F | `TCOD_b2f_workflow.TCOD_b2f_<benchmark>_workflow` |
| Guided-OPD | `guided_opd_workflow.GuidedOPD<Benchmark>Workflow` |

For WebShop, both FTB and the mapped TCOD-B2F baseline use
`raw_task["actions"]` as the successful reference prefix. Every WebShop FTB
configuration fixes `b2f_prefix_source: reference`; no paper configuration
uses the legacy live-Teacher-prefix workflow.

For TCOD classes, the benchmark token is lowercase `alfworld`, `webshop`, or
`scienceworld`. For OPD and Guided-OPD classes, use `Alfworld`, `Webshop`, or
`Scienceworld`. Baseline-specific workflow arguments from the TCOD
configuration should be preserved.
