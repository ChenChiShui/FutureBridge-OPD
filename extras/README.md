# Optional diagnostics

These tools are informative but are not part of the main paper-reproduction
chain:

- `compare_logged_runs.py` aligns arbitrary monitor metrics by optimization
  step and produces an observational comparison for a caller-selected window.
- `inspect_experience_db.py` inventories available serialized fields without
  dumping prompts, responses, task identifiers, or result records.

No derived data are included. The tools write new artifacts only when run.
Step-aligned comparisons are descriptive and should not be presented as a
replacement for a controlled ablation.

Example:

```bash
python extras/compare_logged_runs.py \
  --run FTB=RUNS/alfworld/ftb \
  --run no_validation=RUNS/alfworld/no_validation \
  --metric rollout/bridge_verified/mean \
  --metric rollout/trigger_kl/mean
```
