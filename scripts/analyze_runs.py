#!/usr/bin/env python
"""Aggregate success rates across SummaryLog JSONL files.

Reads every JSONL file matching --pattern under a directory (as written by
scripts/run_episode.py / run_vla_only.py via vlm_vla.core.SummaryLog),
dedups by (mode, task_suite, task_id, episode) keeping the latest timestamp
per group, and reports success rates at three levels of granularity.

Example:
    python scripts/analyze_runs.py outputs/summaries
    python scripts/analyze_runs.py outputs/summaries --json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("analyze_runs")


def _load_records(directory: Path, pattern: str) -> list[dict]:
    records = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        with path.open() as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logging.warning("Skipping unparseable line %s:%d: %s", path, line_num, exc)
    return records


def _dedup(records: list[dict]) -> list[dict]:
    """Keep only the latest-timestamp record per (mode, task_suite, task_id, episode)."""
    latest: dict[tuple, dict] = {}
    for record in records:
        try:
            key = (record["mode"], record["task_suite"], record["task_id"], record["episode"])
        except KeyError as exc:
            logging.warning("Skipping record missing required field %s: %s", exc, record)
            continue
        existing = latest.get(key)
        if existing is None or record.get("timestamp", "") > existing.get("timestamp", ""):
            latest[key] = record
    return list(latest.values())


def _rate(successes: int, count: int) -> float:
    return successes / count if count else 0.0


def _aggregate(records: list[dict]):
    by_mode = defaultdict(lambda: [0, 0])  # mode -> [successes, count]
    by_mode_suite = defaultdict(lambda: [0, 0])  # (mode, task_suite) -> [successes, count]
    by_mode_suite_task = defaultdict(lambda: [0, 0])  # (mode, task_suite, task_id) -> [successes, count]

    for record in records:
        mode = record["mode"]
        task_suite = record["task_suite"]
        task_id = record["task_id"]
        success = bool(record.get("success"))

        for bucket, key in (
            (by_mode, mode),
            (by_mode_suite, (mode, task_suite)),
            (by_mode_suite_task, (mode, task_suite, task_id)),
        ):
            bucket[key][1] += 1
            if success:
                bucket[key][0] += 1

    return by_mode, by_mode_suite, by_mode_suite_task


def _to_summary_dict(bucket: dict) -> dict:
    return {
        key: {"successes": successes, "count": count, "success_rate": _rate(successes, count)}
        for key, (successes, count) in bucket.items()
    }


def _print_report(by_mode: dict, by_mode_suite: dict, by_mode_suite_task: dict) -> None:
    print("=== Overall (by mode) ===")
    for mode in sorted(by_mode):
        successes, count = by_mode[mode]
        print(f"{mode:<16} : {successes}/{count} = {_rate(successes, count) * 100:.1f}%")

    print()
    print("=== By mode x task_suite ===")
    for mode, task_suite in sorted(by_mode_suite):
        successes, count = by_mode_suite[(mode, task_suite)]
        print(f"{mode:<12} {task_suite:<15}: {successes}/{count} = {_rate(successes, count) * 100:.1f}%")

    print()
    print("=== By mode x task_suite x task_id (detail) ===")
    for mode, task_suite, task_id in sorted(by_mode_suite_task, key=lambda k: (k[0], k[1], k[2])):
        successes, count = by_mode_suite_task[(mode, task_suite, task_id)]
        print(
            f"{mode:<12} {task_suite:<15} task {task_id:<3}: "
            f"{successes}/{count} = {_rate(successes, count) * 100:.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory to search for summary JSONL files")
    parser.add_argument(
        "--pattern", default="**/*.jsonl", help="Glob pattern for summary files (default: **/*.jsonl)"
    )
    parser.add_argument("--json", action="store_true", help="Print aggregates as JSON instead of a text report")
    args = parser.parse_args()

    directory = Path(args.directory)
    records = _load_records(directory, args.pattern)
    deduped = _dedup(records)

    print(f"Read {len(records)} raw record(s); {len(deduped)} remain after dedup.")

    by_mode, by_mode_suite, by_mode_suite_task = _aggregate(deduped)

    if args.json:
        output = {
            "by_mode": _to_summary_dict(by_mode),
            "by_mode_task_suite": {
                f"{mode}|{task_suite}": v for (mode, task_suite), v in _to_summary_dict(by_mode_suite).items()
            },
            "by_mode_task_suite_task_id": {
                f"{mode}|{task_suite}|{task_id}": v
                for (mode, task_suite, task_id), v in _to_summary_dict(by_mode_suite_task).items()
            },
        }
        print(json.dumps(output, indent=2))
    else:
        _print_report(by_mode, by_mode_suite, by_mode_suite_task)


if __name__ == "__main__":
    main()
