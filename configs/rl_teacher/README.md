# RL-teacher configuration

Teacher training uses the external
[verl-agent/GiGPO codebase](https://github.com/langfengQ/verl-agent) with the
settings recorded in `shared.yaml` plus the benchmark override. The recovered
workspace was based on revision `796ed310287fa605c9292a0fce07a86d79fde05e`
and included environment-integration changes, so prepare that dependency
separately rather than treating the revision alone as the complete runtime.

`train_webshop.sh` is a portable transcription of the recovered final WebShop
GiGPO-8B entry point. It requires caller-supplied paths:

```bash
GIGPO_DIR=../gigpo \
MODEL_PATH=../models/Qwen3-8B \
TRAIN_DATA=../data/rl_teacher/train.parquet \
VAL_DATA=../data/rl_teacher/validation.parquet \
OUTPUT_DIR=../outputs/rl_teacher/webshop \
./configs/rl_teacher/train_webshop.sh
```

`alfworld.yaml` remains a framework-neutral parameter record because the
corresponding final executable entry point was not present in the recovered
bundle. No result values, model outputs, or machine-specific paths are
included.

`rl_teacher_settings.tex` is the copy-ready appendix table. The YAML records,
LaTeX table, and WebShop launcher agree on log-prob micro-batch size 8 and
tensor parallel size 2. The launcher uses `trainer.total_epochs=100`; epochs
and optimization steps are not treated as interchangeable.
