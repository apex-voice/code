"""Latency / efficiency metrics.

Computed from the canonical event log + audio timeline stats. Media-time quantities (RSL, TTR) are
harness-comparable across models; true wall latency is measured separately by realtime adapters.
Also reports action-timing diagnostics (tool/retrieval counts, user speaking time, silence ratio).
"""

from __future__ import annotations

from statistics import median
from typing import Any

from apex_voice.harness.audio import TimelineStats
from apex_voice.schemas.run_event import EventType, RunEvent


def latency_metrics(events: list[RunEvent], rsl_samples: list[int], tstats: TimelineStats) -> dict[str, Any]:
    first_agent = next((e.media_time_ms for e in events if e.type == EventType.AGENT_AUDIO_START), None)
    first_user = next((e.media_time_ms for e in events if e.type == EventType.USER_AUDIO_START), None)
    run_end = next((e.media_time_ms for e in reversed(events) if e.type == EventType.RUN_END), None)

    tool_calls = sum(1 for e in events if e.type == EventType.TOOL_CALL)
    retrieval_calls = sum(
        1
        for e in events
        if e.type == EventType.TOOL_CALL and e.payload.get("tool", "").startswith(("kb_search", "search"))
    )
    user_turns = sum(1 for e in events if e.type == EventType.USER_AUDIO_START)

    ttr = (run_end - first_user) if (run_end is not None and first_user is not None) else None
    return {
        "response_start_latency_ms": {
            "median": median(rsl_samples) if rsl_samples else None,
            "n": len(rsl_samples),
        },
        "first_agent_audio_ms": first_agent,
        "time_to_resolution_ms": ttr,
        "user_speaking_ms": round(tstats.user_speech_ms, 1),
        "agent_speaking_ms": round(tstats.agent_speech_ms, 1),
        "silence_ratio": tstats.silence_ratio,
        "overlap_ratio": tstats.overlap_ratio,
        "tool_calls": tool_calls,
        "retrieval_calls": retrieval_calls,
        "user_turns": user_turns,
    }


__all__ = ["latency_metrics"]
