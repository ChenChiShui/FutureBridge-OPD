"""
fix_game_paths.py
-----------------
Replace the hard-coded absolute path prefix in all *.jsonl files inside this
directory so that `game_file` values point to your local ALFWorld data root.

Usage
-----
    # Interactive (will prompt you for the new root):
    python fix_game_paths.py

    # Non-interactive (pass the new root directly):
    python fix_game_paths.py --new-root <alfworld-data>

    # Dry-run (preview without writing):
    python fix_game_paths.py --new-root <alfworld-data> --dry-run

The script keeps a backup of each original file alongside it (*.jsonl.bak)
unless you pass --no-backup.
"""

import argparse
import json
import os
import shutil
import sys

# ── The original hard-coded prefix embedded in every game_file path ──────────
ORIGINAL_PREFIX = "/nas/wjq/alfworld/"

# Directory that contains this script (i.e. alfworld_data/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def collect_jsonl_files(directory: str) -> list[str]:
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".jsonl")
    )


def fix_file(filepath: str, new_prefix: str, dry_run: bool, backup: bool) -> int:
    """
    Replace ORIGINAL_PREFIX with *new_prefix* in every `game_file` value of
    *filepath*.  Returns the number of lines that were changed.
    """
    changed = 0
    new_lines: list[str] = []

    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                new_lines.append(line)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                new_lines.append(line)
                continue

            if "game_file" in obj and obj["game_file"].startswith(ORIGINAL_PREFIX):
                obj["game_file"] = new_prefix + obj["game_file"][len(ORIGINAL_PREFIX):]
                changed += 1

            new_lines.append(json.dumps(obj, ensure_ascii=False))

    if changed and not dry_run:
        if backup:
            shutil.copy2(filepath, filepath + ".bak")
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(new_lines) + "\n")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace hard-coded ALFWorld game_file paths in *.jsonl files."
    )
    parser.add_argument(
        "--new-root",
        metavar="PATH",
        help="Your local ALFWorld data root (replaces the original prefix).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing any files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create *.jsonl.bak backup files.",
    )
    args = parser.parse_args()

    # ── Resolve the new root ──────────────────────────────────────────────────
    new_root: str
    if args.new_root:
        new_root = args.new_root
    else:
        print(f"Original prefix in game_file: {ORIGINAL_PREFIX!r}")
        print("Enter the new root path to replace it with.")
        print("  Example: <alfworld-data>  or  ./alfworld_data")
        new_root = input("New root path: ").strip()
        if not new_root:
            print("No path provided. Aborting.")
            sys.exit(1)

    # Normalise: ensure the prefix ends with exactly one slash so the
    # relative tail (e.g. "json_2.1.1/...") is joined correctly.
    new_prefix = new_root.rstrip("/") + "/"

    print(f"\nOriginal prefix : {ORIGINAL_PREFIX!r}")
    print(f"Replacement     : {new_prefix!r}")
    if args.dry_run:
        print("Mode            : DRY RUN (no files will be modified)\n")
    else:
        backup = not args.no_backup
        print(f"Backup          : {'yes (*.jsonl.bak)' if backup else 'no'}\n")

    # ── Process each .jsonl file ──────────────────────────────────────────────
    files = collect_jsonl_files(SCRIPT_DIR)
    if not files:
        print("No *.jsonl files found in", SCRIPT_DIR)
        sys.exit(0)

    total_changed = 0
    for filepath in files:
        n = fix_file(filepath, new_prefix, dry_run=args.dry_run, backup=not args.no_backup)
        status = f"{n} line(s) updated" if n else "no changes"
        marker = "  [DRY RUN]" if args.dry_run else ""
        print(f"  {os.path.basename(filepath):<25} {status}{marker}")
        total_changed += n

    print(f"\nDone. Total lines updated: {total_changed}")
    if args.dry_run and total_changed:
        print("Re-run without --dry-run to apply the changes.")


if __name__ == "__main__":
    main()
