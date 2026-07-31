"""Recreate the training-dynamics panels directly from archived run logs.

This is a portable extraction of the plotting path used for the paper.  Run
directories are supplied explicitly instead of being embedded in the file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from monitor import metric_series, parse_labeled_path, parse_monitor_log, smooth


STYLES = {
    "OPD": dict(color="#7E8791", ls="--", lw=2.2),
    "B2F": dict(color="#26758A", ls="-", lw=2.2),
    "F2B": dict(color="#C97918", ls="-.", lw=2.2),
    "FTB": dict(color="#FF4D4F", ls="-", lw=2.8),
}
PANELS = (
    ("rollout/env_done/mean", "Completion rate", "a. Completion Rate"),
    ("kl_per_round", "KL per round", "b. Teacher–Student KL per Round"),
    ("rollout/env_rounds/mean", "Mean number of rounds", "c. Number of Rounds"),
    ("critic/advantages/mean", "Mean advantage", "d. Mean Advantage"),
)


def load_run(run_dir: Path):
    explorer = parse_monitor_log(run_dir / "log" / "explorer.log")
    trainer = parse_monitor_log(run_dir / "log" / "trainer.log")
    series = {}
    for key, _, _ in PANELS:
        if key == "critic/advantages/mean":
            series[key] = metric_series(trainer, key)
        elif key == "kl_per_round":
            steps_kl, kl = metric_series(explorer, "rollout/kl_divergence/mean")
            steps_rounds, rounds = metric_series(explorer, "rollout/env_rounds/mean")
            by_step = dict(zip(steps_rounds.tolist(), rounds.tolist()))
            keep_steps, values = [], []
            for step, value in zip(steps_kl, kl):
                denominator = by_step.get(int(step))
                if denominator is not None:
                    keep_steps.append(step)
                    values.append(value / max(denominator, np.finfo(float).eps))
            series[key] = np.asarray(keep_steps), np.asarray(values)
        else:
            series[key] = metric_series(explorer, key)
    return series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="BENCHMARK=METHOD=RUN_DIR",
        help="repeat once for every curve",
    )
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/training_dynamics.png"),
    )
    args = parser.parse_args()

    runs = defaultdict(dict)
    for spec in args.run:
        try:
            benchmark, method, path = parse_labeled_path(spec, 3)
        except ValueError as error:
            parser.error(str(error))
        run_dir = Path(path)
        if not (run_dir / "log" / "explorer.log").is_file():
            parser.error(f"missing explorer log under: {run_dir}")
        runs[benchmark][method] = load_run(run_dir)

    benchmarks = list(runs)
    fig, axes = plt.subplots(
        len(benchmarks),
        len(PANELS),
        figsize=(16, 4.2 * len(benchmarks)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, benchmark in enumerate(benchmarks):
        for method, data in runs[benchmark].items():
            style = STYLES.get(method, dict(lw=2.0))
            for column, (key, ylabel, title) in enumerate(PANELS):
                ax = axes[row, column]
                steps, values = data[key]
                if len(steps):
                    ax.plot(steps, values, alpha=0.12, lw=0.7, color=style.get("color"))
                    ax.plot(
                        steps,
                        smooth(values, args.smooth_window),
                        label=method,
                        **style,
                    )
                ax.set_title(title)
                ax.set_xlabel("Training step")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
        axes[row, 0].text(
            0.0,
            1.16,
            benchmark,
            transform=axes[row, 0].transAxes,
            fontweight="semibold",
        )

    methods = list(dict.fromkeys(method for benchmark in benchmarks for method in runs[benchmark]))
    handles = [
        Line2D([0], [0], label=method, **STYLES.get(method, dict(lw=2.0)))
        for method in methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
