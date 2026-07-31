# Reproduction notes

This compact release is a TCOD workflow overlay rather than a benchmark or
model distribution. External benchmark and model assets remain normal runtime
prerequisites.

The released YAML files cover FTB, its reported component ablations, and the
reported model/Teacher settings. OPD, TCOD-F2B, TCOD-B2F, and Guided-OPD
workflow implementations are included for integration and inspection. The
external RL-Teacher stage is represented by the manuscript hyperparameters
and a portable WebShop launcher under `configs/rl_teacher/`.

The evaluation task sets and decoding settings are included in each experiment
configuration. In-training evaluation is disabled so that evaluation can be
run once on the exported Student checkpoint after optimization.

WebShop FTB and TCOD-B2F share the same input contract: every training record
must provide an `actions` list containing a pre-collected successful reference
trajectory. The released workflow raises a clear error instead of silently
falling back to a pure Student rollout when this field is missing.

Paper configurations disable reward-based bridge filtering and require all
three continuation turns. Compatibility switches for the legacy behaviors are
documented but disabled.

Training logs, result tables, exported CSV files, precomputed result figures,
checkpoints, and private infrastructure artifacts are outside the release
scope. The conceptual motivation and pipeline figures used by the README are
included under `fig/`.
