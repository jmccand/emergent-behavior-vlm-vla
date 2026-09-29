"""Lightweight JSONL episode-result logging, independent of --record.

Every run_episode.py / run_vla_only.py invocation appends one line per
completed episode here, so `scripts/analyze_runs.py` can aggregate success
rates across runs without needing the heavier --record video/step logs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class SummaryLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a")

    def write(self, **fields) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
