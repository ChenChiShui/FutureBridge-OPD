"""Plot trigger-point disagreement from caller-supplied explorer logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monitor import metric_series, parse_labeled_path, parse_monitor_log, smooth


STYLES = {
    "FTB w/o Bridge Exec.": dict(color="#8E76B3", ls="-", lw=2.2),
    "FTB w/o Future Validation": dict(color="#C97918", ls="--", lw=2.2),
    "FTB": dict(color="#FF4D4F", ls="-", lw=2.8),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="METHOD=RUN_DIR")
    parser.add_argument("--metric", default="rollout/trigger_kl/mean")
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/trigger_disagreement.png"),
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    for spec in args.run:
        try:
            method, path_text = parse_labeled_path(spec, 2)
        except ValueError as error:
            parser.error(str(error))
        log_path = Path(path_text) / "log" / "explorer.log"
        if not log_path.is_file():
            parser.error(f"missing explorer log: {log_path}")
        steps, values = metric_series(parse_monitor_log(log_path), args.metric)
        style = STYLES.get(method, dict(lw=2.0))
        ax.plot(steps, smooth(values, args.smooth_window), label=method, **style)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Trigger-point sampled disagreement")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
