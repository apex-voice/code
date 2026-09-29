"""Loading campaign output directories (see :mod:`apex_voice.campaign` for the layout)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    task_id: str
    rep: int
    pts: int
    components: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    duplex: dict[str, Any] = field(default_factory=dict)
    run_dir: Path | None = None  # directory holding events.jsonl / workspace / audio

    def final_workspace(self) -> dict[str, Any] | None:
        if self.run_dir is None:
            return None
        p = self.run_dir / "workspace" / "final_workspace.json"
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def events(self) -> Iterator[dict[str, Any]]:
        if self.run_dir is None:
            return
        p = self.run_dir / "events.jsonl"
        if not p.exists():
            return
        with p.open() as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _resolve_run_dir(rep_dir: Path, run_id: str | None) -> Path | None:
    if run_id and (rep_dir / run_id / "workspace" / "final_workspace.json").exists():
        return rep_dir / run_id
    # Fallback: the most recent completed sub-run (campaigns written before run_id was recorded).
    cands = [w.parent.parent for w in rep_dir.rglob("workspace/final_workspace.json")]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def load_campaign(campaign_dir: str | Path) -> dict[str, dict[int, RunRecord]]:
    """Return ``{task_id: {rep: RunRecord}}`` for every persisted run in a campaign directory."""
    root = Path(campaign_dir)
    out: dict[str, dict[int, RunRecord]] = {}
    for rj in sorted(root.glob("*/rep*/result.json")):
        try:
            r = json.loads(rj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tid, rep = r.get("task_id"), r.get("rep")
        if tid is None or rep is None:
            continue
        out.setdefault(tid, {})[int(rep)] = RunRecord(
            task_id=tid,
            rep=int(rep),
            pts=int(bool(r.get("pts"))),
            components=r.get("components") or {},
            latency=r.get("latency") or {},
            duplex=r.get("duplex") or {},
            run_dir=_resolve_run_dir(rj.parent, r.get("run_id")),
        )
    return out


__all__ = ["RunRecord", "load_campaign"]
