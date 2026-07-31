# Release manifest

This release contains:

- 23 Python source files selected from the server archive.
- 19 runnable FTB YAML configurations covering the three benchmarks, both
  reported Student sizes, the RL-teacher setting, and all reported ablations.
- Three framework-neutral YAML records, a portable WebShop launcher, and a
  copy-ready LaTeX table for the external RL-teacher stage.
- Portable analysis scripts for training dynamics, teacher-preference by turn,
  trigger disagreement, and retained-bridge behavior.
- Two conceptual figures used to introduce the paper and method.
- Optional log and database inspection tools under `extras/`.
- Documentation, recovered runtime provenance, an overlay installer,
  runtime requirements, release-boundary notes, a license, and ignore rules.

The source boundary includes the three environment adapters, OPD, TCOD-F2B,
TCOD-B2F, Guided-OPD, the bridge dependencies required by FTB, the main
workflows, the reported random-turn/no-execution/no-validation ablations, and
the additional bridge-token gate ablation.

Excluded categories include logs, model weights, checkpoints, result caches,
bytecode, hidden operating-system files, shell history, temporary probes,
duplicate experiment snapshots, nested archives, secrets, exported numerical
tables, and precomputed result figures. The raw data from the introduction's
motivating intervention experiment is also outside the release boundary.

Provenance: implementation files were selected from the supplied server
archive and packaged as a paper-aligned reference implementation. Release
cleanup removed private paths, secrets, internal labels, generated artifacts,
and non-portable configuration values. The named compatibility switches in the
paper YAML files explicitly select reward-independent bridge generation and
complete paired continuations; legacy reward-gated and partial-continuation
behaviors are opt-in and are not paper configurations.
