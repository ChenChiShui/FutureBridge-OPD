#!/usr/bin/env python3
"""Merge GSM8K main and socratic subsets into a single dataset.

The main and socratic subsets have the same questions in the same order.
This script merges them so each sample has: question, answer (from main), socratic_answer (from socratic).

Usage:
    python scripts/merge_gsm8k_main_socratic.py \
        --main_dir /nas/wjq/openai/gsm8k/main \
        --socratic_dir /nas/wjq/openai/gsm8k/socratic \
        --output_dir /nas/wjq/openai/gsm8k/merged
"""

import argparse
from pathlib import Path

import pandas as pd


def merge_split(main_path: Path, socratic_path: Path, output_path: Path) -> None:
    """Merge main and socratic parquet files for a single split."""
    main_df = pd.read_parquet(main_path)
    socratic_df = pd.read_parquet(socratic_path)

    assert len(main_df) == len(socratic_df), (
        f"Length mismatch: main={len(main_df)}, socratic={len(socratic_df)}"
    )
    assert "question" in main_df.columns and "answer" in main_df.columns
    assert "question" in socratic_df.columns and "answer" in socratic_df.columns

    merged = main_df[["question", "answer"]].copy()
    merged["socratic_answer"] = socratic_df["answer"].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f"Saved {len(merged)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge GSM8K main and socratic subsets")
    parser.add_argument(
        "--main_dir",
        default="/nas/wjq/openai/gsm8k/main",
        help="Path to main subset directory (contains train-*.parquet, test-*.parquet)",
    )
    parser.add_argument(
        "--socratic_dir",
        default="/nas/wjq/openai/gsm8k/socratic",
        help="Path to socratic subset directory",
    )
    parser.add_argument(
        "--output_dir",
        default="/nas/wjq/openai/gsm8k/merged",
        help="Output directory for merged parquet files",
    )
    args = parser.parse_args()

    main_dir = Path(args.main_dir)
    socratic_dir = Path(args.socratic_dir)
    output_dir = Path(args.output_dir)

    for split in ["train", "test"]:
        main_files = list(main_dir.glob(f"{split}-*.parquet"))
        socratic_files = list(socratic_dir.glob(f"{split}-*.parquet"))
        if not main_files or not socratic_files:
            print(f"Skip {split}: missing parquet files")
            continue
        merge_split(
            main_path=main_files[0],
            socratic_path=socratic_files[0],
            output_path=output_dir / f"{split}.parquet",
        )


if __name__ == "__main__":
    main()
