"""Compute teacher-preferred token ratios from explorer databases.

For each non-bridge student response, this script measures the fraction of
realized tokens whose teacher log-probability is greater than the student's.
It aggregates those per-response ratios by student turn and plots the mean
with a bootstrap interval. No result data are bundled with this script.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monitor import parse_labeled_path


STYLES = {
    "OPD": dict(color="#7E8791", ls="--", lw=2.1),
    "TCOD-B2F": dict(color="#26758A", ls="-", lw=2.2),
    "FTB w/o Bridge Exec.": dict(color="#D08A1C", ls="--", lw=2.2),
    "FTB w/o Future Validation": dict(color="#8E76B3", ls="-.", lw=2.2),
    "FTB": dict(color="#FF4D4F", ls="-", lw=2.8),
}


def is_bridge(experience) -> bool:
    metrics = getattr(experience, "metrics", {}) or {}
    return bool(metrics.get("is_bridge", False))


def preferred_ratio(experience):
    teacher = getattr(experience, "teacher_logprobs", None)
    student = getattr(experience, "logprobs", None)
    if teacher is None or student is None:
        return None
    teacher = np.asarray(teacher, dtype=float)
    student = np.asarray(student, dtype=float)
    length = min(teacher.size, student.size)
    if not length:
        return None
    valid = np.isfinite(teacher[:length]) & np.isfinite(student[:length])
    if not valid.any():
        return None
    return float(np.mean(teacher[:length][valid] > student[:length][valid]))


def load_database(path: Path):
    by_turn = defaultdict(list)
    connection = sqlite3.connect(str(path))
    cursor = connection.cursor()
    cursor.execute("SELECT experience_bytes FROM pipeline_input ORDER BY id")
    for (blob,) in cursor:
        try:
            experience = pickle.loads(blob)
        except Exception:
            continue
        if is_bridge(experience):
            continue
        ratio = preferred_ratio(experience)
        eid = getattr(experience, "eid", None)
        step = getattr(eid, "step", None)
        if ratio is None or step is None:
            continue
        by_turn[int(step) + 1].append(ratio)
    connection.close()
    return by_turn


def summarize(values, generator, bootstrap_samples):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2 or bootstrap_samples < 1:
        return mean, mean, mean
    samples = generator.choice(values, size=(bootstrap_samples, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="BENCHMARK=METHOD=DB",
    )
    parser.add_argument("--max-turn", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/teacher_preference"),
    )
    args = parser.parse_args()

    grouped = defaultdict(dict)
    generator = np.random.default_rng()
    rows = []
    for spec in args.run:
        try:
            benchmark, method, db_text = parse_labeled_path(spec, 3)
        except ValueError as error:
            parser.error(str(error))
        db_path = Path(db_text)
        if not db_path.is_file():
            parser.error(f"database does not exist: {db_path}")
        grouped[benchmark][method] = load_database(db_path)

    for benchmark, methods in grouped.items():
        for method, by_turn in methods.items():
            for turn in sorted(by_turn):
                if turn > args.max_turn:
                    continue
                mean, low, high = summarize(
                    by_turn[turn], generator, args.bootstrap_samples
                )
                rows.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "turn": turn,
                        "mean": mean,
                        "low": low,
                        "high": high,
                        "count": len(by_turn[turn]),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "teacher_preference_by_turn.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ())
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    benchmarks = list(grouped)
    fig, axes = plt.subplots(
        1, len(benchmarks), figsize=(6.2 * len(benchmarks), 4.8), squeeze=False
    )
    for column, benchmark in enumerate(benchmarks):
        ax = axes[0, column]
        for method in grouped[benchmark]:
            subset = [row for row in rows if row["benchmark"] == benchmark and row["method"] == method]
            turns = np.asarray([row["turn"] for row in subset])
            means = np.asarray([row["mean"] for row in subset])
            low = np.asarray([row["low"] for row in subset])
            high = np.asarray([row["high"] for row in subset])
            style = STYLES.get(method, dict(lw=2.0))
            ax.fill_between(turns, low, high, color=style.get("color"), alpha=0.1)
            ax.plot(turns, means, label=method, **style)
        ax.set_title(benchmark)
        ax.set_xlabel("Student turn index")
        ax.set_ylabel("Teacher-preferred token ratio")
        ax.grid(alpha=0.3)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    figure_path = args.output_dir / "teacher_preference_by_turn.png"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(table_path)
    print(figure_path)


if __name__ == "__main__":
    main()
