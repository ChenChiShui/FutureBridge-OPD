"""Shared parsers for Trinity monitor-style logs.

The archived runs write lines containing ``Step N: {metric_dict}``.  This
module intentionally contains no run names, paths, or precomputed results.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


STEP_PATTERN = re.compile(r"Step\s+(\d+):\s+(\{.*\})")


def _literal_dict(text: str) -> Dict[str, Any]:
    text = re.sub(r"<wandb[^>]*>", "None", text)
    text = re.sub(r"tensor\(([^()]+)\)", r"\1", text)
    value = ast.literal_eval(text)
    return value if isinstance(value, dict) else {}


def parse_monitor_log(path: Path) -> Dict[int, Dict[str, Any]]:
    """Return one merged metric dictionary per optimization step."""
    records: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = STEP_PATTERN.search(line)
            if not match:
                continue
            try:
                metrics = _literal_dict(match.group(2))
            except (SyntaxError, ValueError):
                continue
            records.setdefault(int(match.group(1)), {}).update(metrics)
    return records


def metric_series(
    records: Dict[int, Dict[str, Any]], key: str
) -> Tuple[np.ndarray, np.ndarray]:
    steps = []
    values = []
    for step in sorted(records):
        value = records[step].get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
            steps.append(step)
        except (TypeError, ValueError):
            continue
    return np.asarray(steps), np.asarray(values)


def smooth(values: Iterable[float], window: int) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - left - 1
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_labeled_path(spec: str, fields: int) -> Tuple[str, ...]:
    """Parse LABEL=PATH or BENCHMARK=METHOD=PATH without splitting PATH."""
    parts = spec.split("=", maxsplit=fields - 1)
    if len(parts) != fields or any(not part for part in parts):
        raise ValueError(f"invalid run specification: {spec!r}")
    return tuple(parts)
