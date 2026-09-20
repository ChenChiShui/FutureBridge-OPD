# ALFWorld Data

This directory contains the ALFWorld task data in `.jsonl` format, along with a utility script to fix hard-coded `game_file` paths so they point to your local ALFWorld installation.

## Files

| File | Description |
|------|-------------|
| `train.jsonl` | Training set |
| `train_expert.jsonl` | Expert demonstration training set |
| `train_hard.jsonl` | Hard training set |
| `test.jsonl` | Test set (seen environments) |
| `test_unseen.jsonl` | Test set (unseen environments) |
| `fix_game_paths.py` | Script to replace hard-coded `game_file` paths |

---

## fix_game_paths.py

Each `.jsonl` entry contains a `game_file` field with an absolute path, e.g.:

```json
{"game_file": "/nas/wjq/alfworld/json_2.1.1/train/.../game.tw-pddl", "target": ""}
```

Use `fix_game_paths.py` to replace the original prefix with the path to your local ALFWorld data root.

### Usage

#### 1. Interactive mode (prompts for the new path)

```bash
python fix_game_paths.py
```

Example session:

```
Original prefix in game_file: '/nas/wjq/alfworld/'
Enter the new root path to replace it with.
  Example: <alfworld-data>  or  ./alfworld_data
New root path: /your/local/alfworld
```

#### 2. Pass the new root directly

```bash
python fix_game_paths.py --new-root /your/local/alfworld
```

#### 3. Dry-run (preview without modifying any files)

```bash
python fix_game_paths.py --new-root /your/local/alfworld --dry-run
```

Example output:

```
Original prefix : '/nas/wjq/alfworld/'
Replacement     : '/your/local/alfworld/'
Mode            : DRY RUN (no files will be modified)

  test.jsonl                140 line(s) updated  [DRY RUN]
  test_unseen.jsonl         134 line(s) updated  [DRY RUN]
  train.jsonl               3553 line(s) updated  [DRY RUN]
  train_expert.jsonl        3553 line(s) updated  [DRY RUN]
  train_hard.jsonl          121 line(s) updated  [DRY RUN]

Done. Total lines updated: 7501
Re-run without --dry-run to apply the changes.
```

#### 4. Apply changes (with automatic backup)

```bash
python fix_game_paths.py --new-root /your/local/alfworld
```

By default, each original `.jsonl` file is backed up as `*.jsonl.bak` before being overwritten.

#### 5. Apply changes without backup

```bash
python fix_game_paths.py --new-root /your/local/alfworld --no-backup
```

### Options

| Option | Description |
|--------|-------------|
| `--new-root PATH` | New ALFWorld data root to replace the original prefix |
| `--dry-run` | Preview changes without writing any files |
| `--no-backup` | Skip creating `*.jsonl.bak` backup files |

### Recommended workflow

```bash
# Step 1: preview
python fix_game_paths.py --new-root /your/local/alfworld --dry-run

# Step 2: apply (backups created automatically)
python fix_game_paths.py --new-root /your/local/alfworld

# Step 3 (optional): remove backups after verifying
rm *.jsonl.bak
```
