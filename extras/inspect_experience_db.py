"""Inspect which portable fields are available in an explorer database.

The report contains field names and aggregate record counts only. It does not
dump prompts, responses, task identifiers, or serialized experiences.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from collections import Counter
from pathlib import Path


def public_fields(value):
    try:
        return sorted(key for key in vars(value) if not key.startswith("_"))
    except TypeError:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/database_inventory.json"),
    )
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error(f"database does not exist: {args.database}")

    connection = sqlite3.connect(str(args.database))
    cursor = connection.cursor()
    query = "SELECT experience_bytes FROM pipeline_input ORDER BY id"
    parameters = ()
    if args.limit is not None:
        query += " LIMIT ?"
        parameters = (args.limit,)

    report = {
        "records_read": 0,
        "unpickle_failures": 0,
        "experience_fields": Counter(),
        "metric_fields": Counter(),
        "eid_fields": Counter(),
    }
    for (blob,) in cursor.execute(query, parameters):
        report["records_read"] += 1
        try:
            experience = pickle.loads(blob)
        except Exception:
            report["unpickle_failures"] += 1
            continue
        report["experience_fields"].update(public_fields(experience))
        metrics = getattr(experience, "metrics", {}) or {}
        if isinstance(metrics, dict):
            report["metric_fields"].update(str(key) for key in metrics)
        report["eid_fields"].update(public_fields(getattr(experience, "eid", None)))
    connection.close()

    serializable = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in report.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
