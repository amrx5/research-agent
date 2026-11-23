"""JSONL trace: one line per event, one file per question. Traces are the evidence
for tool-selection evaluation and for the M3 'executed trace' requirement."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, trace_dir: Path | None, question_id: str = "q"):
        self.events: list[dict[str, Any]] = []
        self._t0 = time.monotonic()
        self.path: Path | None = None
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in question_id)[:40]
            self.path = trace_dir / f"{stamp}_{safe_id}.jsonl"

    def log(self, event: str, **data: Any) -> None:
        rec = {"t": round(time.monotonic() - self._t0, 3), "event": event, **data}
        self.events.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")

    def tools_called(self) -> list[str]:
        return [e["tool"] for e in self.events if e["event"] == "tool_result"]
