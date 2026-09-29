"""Run logger + event bus.

Emits the single canonical, immutable, totally-ordered JSONL event log for a run. Every event
carries a monotonic wall timestamp (ns) and a normalized media timestamp (ms). The log is
sufficient to recompute every non-model score post hoc (CI: "score recomputation test").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apex_voice.config import Condition
from apex_voice.schemas.run_event import Actor, EventType, RunEvent


class RunLogger:
    def __init__(
        self, run_id: str, task_version: str, condition: Condition, out_path: str | Path | None = None
    ) -> None:
        self.run_id = run_id
        self.task_version = task_version
        self.condition = condition
        self.events: list[RunEvent] = []
        self._seq = 0
        self._fh = None
        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("w")

    def emit(
        self,
        type: EventType,
        actor: Actor,
        wall_time_ns: int,
        media_time_ms: int,
        payload: dict[str, Any] | None = None,
        causal_parents: list[int] | None = None,
    ) -> RunEvent:
        ev = RunEvent(
            seq=self._seq,
            type=type,
            actor=actor,
            wall_time_ns=wall_time_ns,
            media_time_ms=media_time_ms,
            payload=payload or {},
            causal_parents=causal_parents or [],
            run_id=self.run_id,
            task_version=self.task_version,
            condition=self.condition,
        )
        self.events.append(ev)
        self._seq += 1
        if self._fh is not None:
            self._fh.write(ev.to_jsonl() + "\n")
            self._fh.flush()
        return ev

    def count(self, type: EventType) -> int:
        return sum(1 for e in self.events if e.type == type)

    def by_type(self, type: EventType) -> list[RunEvent]:
        return [e for e in self.events if e.type == type]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @staticmethod
    def load_jsonl(path: str | Path) -> list[RunEvent]:
        out: list[RunEvent] = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                out.append(RunEvent.model_validate_json(line))
        return out


__all__ = ["RunLogger"]
