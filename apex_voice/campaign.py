"""Resumable benchmark campaigns: run one realtime model over a task set for N repetitions.

Output layout (one directory per model campaign)::

    <out>/
      campaign.json                         # campaign manifest (model, settings, versions)
      <task_id>/rep<k>/result.json          # per-run score summary (the unit of resumption)
      <task_id>/rep<k>/<run_id>/events.jsonl
      <task_id>/rep<k>/<run_id>/workspace/  # final workspace + version history
      <task_id>/rep<k>/<run_id>/audio/      # stereo.wav (L=user, R=agent), user.wav, agent.wav

Runs are ordered coverage-first (every task's rep0 before any rep1), so an interrupted campaign still
covers every task. A ``(task, rep)`` cell whose ``result.json`` exists is skipped on re-invocation.

Transient infrastructure failures (dropped WebSocket, gateway 5xx, keep-alive timeouts) are retried
with backoff. If a cell still fails transiently it is *not* persisted, so the next invocation re-runs
it instead of counting an infrastructure fault against the model. Non-transient errors are persisted
as failed runs (PTS = 0).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex_voice import BENCHMARK_VERSION, __version__
from apex_voice.config import Condition, RunConfig

MAX_ATTEMPTS = 3
DEFAULT_MAX_TURNS = 44
DEFAULT_REPEATS = 3
_TRANSIENT_MARKERS = (
    "connectionclosed",
    "keepalive",
    "no close frame",
    "ping timeout",
    "timeouterror",
    "timed out",
    "1011",
    "1006",
    "econnreset",
    "connection reset",
    "server disconnected",
    "connection closed",
    "temporarily",
    "handshake",
    "502",
    "503",
    "504",
)


def is_transient(message: str) -> bool:
    msg = message.lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


@dataclass
class CampaignSettings:
    model: str
    out_dir: Path
    repeats: int = DEFAULT_REPEATS
    condition: Condition = Condition.C2
    max_turns: int = DEFAULT_MAX_TURNS
    delay_s: float = 0.0
    user_audio: str = "auto"  # see apex_voice.user_sim.prerendered.compiler_for_task
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)


class CampaignLock:
    """Exclusive lock on a campaign output directory (stale locks from dead processes are reclaimed)."""

    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / ".campaign.lock"

    def __enter__(self) -> CampaignLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                self.path.unlink(missing_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"another campaign is already writing to {self.path.parent}") from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)


def _write_manifest(s: CampaignSettings, n_tasks: int) -> None:
    from apex_voice.adapters.registry import get_model
    from apex_voice.scoring.judge import JUDGE_VERSION

    spec = get_model(s.model)
    manifest = {
        "model": s.model,
        "display_name": spec.display_name,
        "condition": s.condition.value,
        "repeats": s.repeats,
        "max_turns": s.max_turns,
        "user_audio": s.user_audio,
        "n_tasks": n_tasks,
        "apex_voice_version": __version__,
        "benchmark_version": BENCHMARK_VERSION,
        "judge_version": JUDGE_VERSION,
    }
    path = s.out_dir / "campaign.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    manifest["created_at"] = prev.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(manifest, indent=2) + "\n")


async def run_cell(
    task_dir: Path, rep: int, s: CampaignSettings, judge: Any, log: Callable[[str], None] = print
) -> dict[str, Any] | None:
    """Run (or load) one ``(task, rep)`` cell. Returns its result row, or ``None`` if left empty."""
    from apex_voice.adapters.registry import create_adapter
    from apex_voice.harness.realtime_runner import run_realtime_task
    from apex_voice.tasks import load_task
    from apex_voice.user_sim.prerendered import compiler_for_task

    lt = load_task(task_dir)
    rep_dir = s.out_dir / lt.spec.id / f"rep{rep}"
    result_path = rep_dir / "result.json"
    if result_path.exists():
        try:
            return json.loads(result_path.read_text())
        except json.JSONDecodeError:
            pass

    cfg = RunConfig(task_id=lt.spec.id, task_version=lt.spec.version, model_id=s.model, condition=s.condition)
    row: dict[str, Any] | None = None
    for attempt in range(MAX_ATTEMPTS):
        adapter = create_adapter(s.model, **s.adapter_kwargs)
        t0 = time.monotonic()
        try:
            res = await run_realtime_task(
                lt,
                adapter,
                cfg,
                run_dir=rep_dir,
                max_turns=s.max_turns,
                judge=judge,
                compiler=compiler_for_task(task_dir, s.user_audio),
            )
            row = {
                "task_id": lt.spec.id,
                "rep": rep,
                "model": s.model,
                "run_id": res.run_id,
                "condition": s.condition.value,
                "pts": res.pts.pts,
                "components": res.pts.to_dict().get("components", {}),
                "turns": res.turns,
                "outcome": res.outcome,
                "duplex": res.duplex,
                "latency": res.latency,
                "wall_s": round(time.monotonic() - t0, 1),
                "error": None,
            }
            resolved = getattr(adapter, "resolved_model", None) or getattr(adapter, "model", None)
            if resolved:
                row["provider_model"] = resolved
            break
        except Exception as e:  # noqa: BLE001 - one task must not abort the campaign
            msg = f"{type(e).__name__}: {e}"
            transient = is_transient(msg)
            row = {
                "task_id": lt.spec.id,
                "rep": rep,
                "model": s.model,
                "pts": 0,
                "error": msg,
                "transient": transient,
                "attempt": attempt,
                "wall_s": round(time.monotonic() - t0, 1),
            }
            if transient and attempt < MAX_ATTEMPTS - 1:
                log(
                    f"  {lt.spec.id} rep{rep}: transient error ({msg[:80]}); "
                    f"retry {attempt + 1}/{MAX_ATTEMPTS - 1}"
                )
                await asyncio.sleep(5 * (attempt + 1))
                continue
            break
        finally:
            try:
                await adapter.close()
            except Exception:  # noqa: BLE001
                pass

    if row is None or (row.get("error") and row.get("transient")):
        log(f"  {lt.spec.id} rep{rep}: unrecovered transient failure; left empty for the next run")
        return None
    rep_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(row, indent=2, default=str) + "\n")
    comps = " ".join(f"{k}={int(bool(v))}" for k, v in (row.get("components") or {}).items())
    log(
        f"  {row['task_id']} rep{rep}: PTS={row['pts']} {comps} turns={row.get('turns', '-')} "
        f"{row['wall_s']}s" + (f"  ERROR {row['error']}" if row.get("error") else "")
    )
    return row


async def run_campaign(
    task_dirs: list[Path], s: CampaignSettings, judge: Any = None, log: Callable[[str], None] = print
) -> list[dict[str, Any]]:
    """Run ``s.repeats`` repetitions of every task, coverage-first. Returns all completed rows."""
    if judge is None:
        from apex_voice.scoring.judge import get_default_judge

        judge = get_default_judge()
    s.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with CampaignLock(s.out_dir):
        _write_manifest(s, len(task_dirs))
        log(
            f"campaign: {len(task_dirs)} tasks x {s.repeats} reps, model={s.model}, "
            f"condition={s.condition.value} -> {s.out_dir}"
        )
        ran_any = False
        for rep in range(s.repeats):
            for d in task_dirs:
                pending = not (s.out_dir / d.name / f"rep{rep}" / "result.json").exists()
                if pending and ran_any and s.delay_s:
                    await asyncio.sleep(s.delay_s)  # provider quota pacing
                ran_any = ran_any or pending
                row = await run_cell(d, rep, s, judge, log=log)
                if row is not None:
                    rows.append(row)
    return rows


__all__ = [
    "CampaignSettings",
    "CampaignLock",
    "run_cell",
    "run_campaign",
    "is_transient",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_REPEATS",
]
