"""Generic, observational comparison of metrics in existing run logs.

This tool is intentionally outside the main reproduction path. It aligns runs
by logged optimization step and emits raw aligned values plus simple window
means. It does not turn an observational comparison into a controlled
ablation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from monitor import parse_labeled_path, parse_monitor_log  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--log-name", default="explorer.log")
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/log_comparison"),
    )
    args = parser.parse_args()

    records = {}
    for spec in args.run:
        try:
            label, path_text = parse_labeled_path(spec, 2)
        except ValueError as error:
            parser.error(str(error))
        log_path = Path(path_text) / "log" / args.log_name
        if not log_path.is_file():
            parser.error(f"missing log: {log_path}")
        records[label] = parse_monitor_log(log_path)

    common_steps = set.intersection(*(set(item) for item in records.values()))
    if args.min_step is not None:
        common_steps = {step for step in common_steps if step >= args.min_step}
    if args.max_step is not None:
        common_steps = {step for step in common_steps if step <= args.max_step}

    aligned = []
    for step in sorted(common_steps):
        row = {"step": step}
        complete = True
        for label, run in records.items():
            for metric in args.metric:
                value = run[step].get(metric)
                if value is None:
                    complete = False
                    break
                row[f"{label}:{metric}"] = value
            if not complete:
                break
        if complete:
            aligned.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "aligned_metrics.csv"
    fields = list(aligned[0]) if aligned else ["step"]
    with aligned_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aligned)

    summary_path = args.output_dir / "window_means.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "metric", "mean", "steps"])
        writer.writeheader()
        for label in records:
            for metric in args.metric:
                key = f"{label}:{metric}"
                values = [float(row[key]) for row in aligned]
                writer.writerow(
                    {
                        "run": label,
                        "metric": metric,
                        "mean": mean(values) if values else "",
                        "steps": len(values),
                    }
                )
    print(aligned_path)
    print(summary_path)


if __name__ == "__main__":
    main()
