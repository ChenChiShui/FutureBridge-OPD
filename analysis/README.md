# Analysis code

This directory contains the portable analysis path extracted from the archived
server workspace. It includes code only: no exported tables, precomputed
curves, result CSV files, figures, logs, or paper-reported values are bundled.
All outputs are generated from databases or logs supplied by the user.

The motivating intervention experiment from the introduction is deliberately
outside this release.

## Paper analysis coverage

- `plot_training_dynamics.py` recreates the completion, disagreement,
  interaction-round, and advantage panels used for the training-dynamics
  figure.
- `plot_teacher_preference_by_turn.py` computes the teacher-preferred token
  ratio from serialized student experiences and aggregates it by student turn.
- `plot_trigger_disagreement.py` plots disagreement at selected trigger positions over
  training.
- `analyze_bridge_behavior.py` reconstructs candidate and retained bridges
  from the explorer databases, then analyzes trigger position, gate
  acceptance, and changes to student actions.
- `monitor.py` contains the shared Trinity monitor-log parser.

The bridge-behavior script is the cleaned server analysis, not a synthetic
result generator. Its embedded account paths and fixed run names were replaced
by command-line arguments. Because the explorer database stores serialized
framework objects, run it from an environment where the corresponding TCOD and
Trinity modules are importable.

## Examples

Training dynamics:

```bash
python analysis/plot_training_dynamics.py \
  --run ALFWorld=OPD=RUNS/alfworld/opd \
  --run ALFWorld=B2F=RUNS/alfworld/b2f \
  --run ALFWorld=F2B=RUNS/alfworld/f2b \
  --run ALFWorld=FTB=RUNS/alfworld/ftb
```

Teacher-preferred token ratio:

```bash
python analysis/plot_teacher_preference_by_turn.py \
  --run ALFWorld=OPD=RUNS/alfworld/opd/buffer/explorer_output.db \
  --run ALFWorld=FTB=RUNS/alfworld/ftb/buffer/explorer_output.db
```

Bridge characteristics:

```bash
python analysis/analyze_bridge_behavior.py \
  --alfworld-db RUNS/alfworld/ftb/buffer/explorer_output.db \
  --webshop-db RUNS/webshop/ftb/buffer/explorer_output.db
```

Trigger disagreement:

```bash
python analysis/plot_trigger_disagreement.py \
  --run "FTB=RUNS/alfworld/ftb" \
  --run "FTB w/o Future Validation=RUNS/alfworld/no_validation"
```

Generated outputs go under `analysis_outputs/` by default and are ignored by
version control.
