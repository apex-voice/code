"""Duplex metrics.

Computed from the duplex-event records emitted by the voice runner. Each record carries the expected
floor action, whether the agent yielded, the interruption stop latency (ISL), whether task-critical
overlapped content was retained, and criticality.

- FDA  Floor Decision Accuracy: fraction of events where the agent's floor action matched expectation.
- ISL  Interruption Stop Latency: median/p90/p95 over yield-expected events + miss rate.
- OCCR Overlapped Critical Content Retention: retained / task-critical overlap events.
- IRA  Interruption Recovery Accuracy: correction/revocation/intent events with correct downstream
       handling (yielded + retained here; full state-repair check is folded in at grading).
- FYR  False Yield Rate: non-yield-expected events where the agent wrongly terminated.
"""

from __future__ import annotations

from statistics import median
from typing import Any


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1)))))
    return float(sorted_vals[k])


_CORRECTIONISH = {
    "MID_SPEECH_CORRECTION",
    "DELAYED_CORRECTION",
    "CANCELLATION_REVOCATION",
    "INTENT_SWITCH",
    "CONCURRENT_CRITICAL_INFO",
}


def duplex_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n_events": 0, "fda": None, "isl_ms": {}, "occr": None, "ira": None, "fyr": None}

    yield_expected = [r for r in records if r.get("expected_floor_action") == "YIELD"]
    continue_expected = [r for r in records if r.get("expected_floor_action") == "CONTINUE"]

    # FDA
    correct = 0
    for r in records:
        exp = r.get("expected_floor_action")
        yielded = bool(r.get("agent_yielded"))
        if (exp == "YIELD" and yielded) or (exp == "CONTINUE" and not yielded):
            correct += 1
    fda = correct / len(records)

    # ISL over yield-expected events that actually yielded
    isls = sorted(int(r["isl_ms"]) for r in yield_expected if r.get("agent_yielded") and "isl_ms" in r)
    misses = sum(1 for r in yield_expected if not r.get("agent_yielded"))
    isl = {
        "median": median(isls) if isls else None,
        "p90": _pct(isls, 0.90) if isls else None,
        "p95": _pct(isls, 0.95) if isls else None,
        "miss_rate": (misses / len(yield_expected)) if yield_expected else None,
        "n": len(isls),
    }

    # OCCR / IRA over task-critical correction-like events
    critical = [r for r in records if r.get("task_critical") and r.get("type") in _CORRECTIONISH]
    occr = (sum(1 for r in critical if r.get("retained_overlap")) / len(critical)) if critical else None
    ira = (
        (sum(1 for r in critical if r.get("agent_yielded") and r.get("retained_overlap")) / len(critical))
        if critical
        else None
    )

    # FYR: continue-expected events (backchannel/distractor) where the agent wrongly yielded
    fyr = (
        (sum(1 for r in continue_expected if r.get("agent_yielded")) / len(continue_expected))
        if continue_expected
        else None
    )

    return {
        "n_events": len(records),
        "fda": round(fda, 4),
        "isl_ms": isl,
        "occr": round(occr, 4) if occr is not None else None,
        "ira": round(ira, 4) if ira is not None else None,
        "fyr": round(fyr, 4) if fyr is not None else None,
    }


def repair_cost(events) -> dict[str, Any]:
    """Repair Cost: extra work after the last task-critical correction/revocation event.

    Reported as components (turns / tool-calls / media-time), never a single opaque scalar. Computed
    from the event log: everything after the last such DUPLEX_EVENT is the recovery work.
    """
    from apex_voice.schemas.run_event import EventType

    crit_times = [
        e.media_time_ms
        for e in events
        if e.type == EventType.DUPLEX_EVENT and e.payload.get("type") in _CORRECTIONISH
    ]
    if not crit_times:
        return {"applicable": False}
    t0 = max(crit_times)
    end = next((e.media_time_ms for e in reversed(events) if e.type == EventType.RUN_END), t0)
    turns = sum(1 for e in events if e.type == EventType.AGENT_AUDIO_START and e.media_time_ms >= t0)
    tool_calls = sum(1 for e in events if e.type == EventType.TOOL_CALL and e.media_time_ms >= t0)
    artifact_writes = sum(
        1
        for e in events
        if e.type == EventType.TOOL_RESULT
        and e.media_time_ms >= t0
        and isinstance(e.payload.get("result"), dict)
        and e.payload["result"].get("updated")
    )
    return {
        "applicable": True,
        "repair_turns": turns,
        "repair_tool_calls": tool_calls,
        "repair_artifact_writes": artifact_writes,
        "repair_time_ms": max(0, end - t0),
    }


__all__ = ["duplex_metrics", "repair_cost"]
