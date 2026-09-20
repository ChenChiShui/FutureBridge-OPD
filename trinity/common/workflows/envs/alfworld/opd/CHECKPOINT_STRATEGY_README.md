# Checkpoint Strategy for AlfWorld OPD Workflow

This document explains how to use the checkpoint strategy feature in the `check_opd_verl_agent_alfworld_workflow`.

## Overview

The checkpoint strategy allows the model to start from different points in the expert trajectory during training. This implements a curriculum learning approach where the model gradually learns to handle the full task autonomously.

## Configuration

### Fixed Checkpoint

Use a fixed number of expert actions before letting the model take over:

```yaml
workflow_args:
  checkpoint_step: 3  # Always execute first 3 expert actions
```

### Linear Decay Strategy (Recommended)

Gradually reduce the number of expert actions as training progresses:

```yaml
workflow_args:
  checkpoint_strategy: linear_decay
  checkpoint_decay_steps: 0.5  # Complete decay in first 50% of training steps
```

**Key Points:**
- **Start point (fixed)**: `len(actions) - 1` — Model only executes the last action
- **End point (fixed)**: `0` — Model executes the entire trajectory from beginning
- **Only one parameter**: `checkpoint_decay_steps` controls when decay completes

**Note**: The parameter name `checkpoint_decay_epochs` is also supported for backward compatibility, but `checkpoint_decay_steps` is recommended when using `total_steps` in your buffer configuration.

## How It Works

### Linear Decay Strategy

The strategy automatically:
1. **Starts** with `checkpoint_step = len(actions) - 1` (easiest: model only does last step)
2. **Decays linearly** to `checkpoint_step = 0` (hardest: model does everything)
3. **Completes decay** at `total_epochs × checkpoint_decay_epochs`
4. **Maintains** `checkpoint_step = 0` for remaining training

### Formula

```python
decay_completion_step = total_steps × checkpoint_decay_steps

if current_step < decay_completion_step:
    progress = current_step / decay_completion_step
    checkpoint_step = (len(actions) - 1) × (1 - progress)
else:
    checkpoint_step = 0  # Full autonomy
```

### Examples

#### Example 1: `checkpoint_decay_steps = 0.5`
With `total_steps = 200` and `num_actions = 7`:

```
Step 0:   checkpoint=6  → Model executes step 7 only (1 step)
Step 20:  checkpoint=5  → Model executes steps 6-7 (2 steps)
Step 40:  checkpoint=4  → Model executes steps 5-7 (3 steps)
Step 60:  checkpoint=2  → Model executes steps 3-7 (5 steps)
Step 80:  checkpoint=1  → Model executes steps 2-7 (6 steps)
Step 100: checkpoint=0  → Model executes steps 1-7 (all 7 steps) ← Decay complete
Step 120-200: checkpoint=0 → Full autonomy continues
```

**Meaning**: Decay completes at step 100 (50% of 200 steps), then full autonomy for remaining 50%.

#### Example 2: `checkpoint_decay_steps = 0.7`
With `total_steps = 200` and `num_actions = 10`:

```
Step 0:   checkpoint=9  → Model executes step 10 only
Step 20:  checkpoint=8  → Model executes steps 9-10
...
Step 120: checkpoint=1  → Model executes steps 2-10
Step 140: checkpoint=0  → Model executes all steps ← Decay complete
Step 160-200: checkpoint=0 → Full autonomy continues
```

**Meaning**: Decay completes at step 140 (70% of 200 steps), then full autonomy for remaining 30%.

#### Example 3: `checkpoint_decay_steps = 1.0`
With `total_steps = 200` and `num_actions = 5`:

```
Step 0:   checkpoint=4  → Model executes step 5 only
Step 40:  checkpoint=3  → Model executes steps 4-5
Step 80:  checkpoint=2  → Model executes steps 3-5
Step 120: checkpoint=2  → Model executes steps 3-5
Step 160: checkpoint=1  → Model executes steps 2-5
Step 200: checkpoint=0  → Model executes all steps ← Decay complete
```

**Meaning**: Decay completes at step 200 (100% of 200 steps), using the full training duration.

## Choosing `checkpoint_decay_steps`

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.3` | Fast decay (30% of training) | Model learns quickly, want early full autonomy |
| `0.5` | Moderate decay (50% of training) | **Recommended default** — balanced approach |
| `0.7` | Slow decay (70% of training) | Difficult tasks, need more gradual curriculum |
| `1.0` | Full training decay | Use entire training for curriculum |

**Rule of thumb**: Start with `0.5`. If model struggles, increase to `0.7` or `1.0`.

## Monitoring

The workflow records detailed metrics for each experience to help you understand the curriculum progress:

### Checkpoint Metrics

| Metric | Description | Example |
|--------|-------------|---------|
| `total_expert_actions` | Total number of actions in the expert trajectory | 7 |
| `expert_actions_executed` | Number of expert actions executed before model takes over | 5 |
| `expert_actions_remaining` | Number of expert actions NOT executed (model had to figure out) | 2 |
| `model_executed_steps` | Number of steps the model executed by itself | 3 |

### Legacy Metrics (for backward compatibility)

| Metric | Description |
|--------|-------------|
| `checkpoint_step` | Same as `expert_actions_executed` |
| `used_checkpoint_step` | Same as `expert_actions_executed` |

### Understanding the Metrics

**Example 1**: With checkpoint_step=5 and total_expert_actions=7
```python
total_expert_actions = 7        # Expert trajectory has 7 actions
expert_actions_executed = 5     # First 5 actions executed automatically
expert_actions_remaining = 2    # Last 2 actions not provided to model
model_executed_steps = 3        # Model executed 3 steps (may differ from remaining due to errors/retries)
```

**Example 2**: No checkpoint (full autonomy)
```python
total_expert_actions = 7        # Expert trajectory has 7 actions
expert_actions_executed = 0     # No expert guidance
expert_actions_remaining = 7    # Model must figure out all 7 actions
model_executed_steps = 8        # Model took 8 steps (more than expert due to exploration)
```

### Monitoring in WandB

You can track curriculum progress by plotting:
- `expert_actions_executed`: Should decrease over training (from ~total_expert_actions to 0)
- `expert_actions_remaining`: Should increase over training (from 0 to ~total_expert_actions)
- `model_executed_steps`: Shows how many steps model needs (may increase as it gets less guidance)

Example WandB query:
```python
# Track curriculum decay
wandb.log({
    "curriculum/expert_guidance": expert_actions_executed,
    "curriculum/model_autonomy": expert_actions_remaining,
    "curriculum/model_steps": model_executed_steps,
})
```

## Complete Configuration Example

```yaml
buffer:
  total_steps: 200  # Or use total_epochs: 10
  batch_size: 16
  explorer_input:
    taskset:
      path: 'train_expert.jsonl'
      split: train
      format:
        prompt_key: 'game_file'
      workflow_args:
        max_env_steps: 30
        checkpoint_strategy: linear_decay
        checkpoint_decay_steps: 0.5  # Use checkpoint_decay_epochs if using total_epochs
    default_workflow_type: 'check_opd_verl_agent_alfworld_workflow'
```

This configuration:
- Trains for 200 steps total (or 10 epochs if using `total_epochs`)
- Starts with model executing only the last action
- Linearly decays to full autonomy over first 100 steps (50%)
- Continues with full autonomy for remaining 100 steps

## Implementation Notes

1. **Automatic start/end**: No need to specify start and end steps — they're automatically determined from the expert trajectory length
2. **Per-trajectory adaptation**: Each trajectory may have different lengths, so the starting checkpoint adapts automatically
3. **Clamping**: Checkpoint step is always clamped to `[0, len(actions)-1]`
4. **Backward compatible**: Without `checkpoint_strategy`, uses fixed `checkpoint_step` if provided

## Comparison with Fixed Checkpoint

| Approach | Configuration | Behavior |
|----------|--------------|----------|
| **Fixed** | `checkpoint_step: 3` | Always execute first 3 actions, never changes |
| **Linear Decay** | `checkpoint_decay_epochs: 0.5` | Start from last action, gradually increase model responsibility |

**Recommendation**: Use linear decay for curriculum learning. Use fixed checkpoint only for debugging or specific experimental needs.
