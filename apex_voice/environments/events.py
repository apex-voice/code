"""External / follow-through event runtime (FOLLOW_THROUGH).

A FOLLOW_THROUGH event is a *seeded environment change* delivered through the world/event bus
(not invented by a user LLM). It applies world mutations (e.g. an appointment slot disappearing)
while the user's original goal/constraints remain active. Timing is relative
to observed agent behavior so it is stable across model latency.
"""

from __future__ import annotations

from dataclasses import dataclass

from apex_voice.schemas.flow import EventSpec


@dataclass
class ScheduledEvent:
    spec: EventSpec
    fired: bool = False
    fired_media_time_ms: int | None = None


class EventEngine:
    """Tracks event triggers and fires world/fact updates at controlled media times."""

    def __init__(self, events: list[EventSpec]) -> None:
        self._events: list[ScheduledEvent] = [ScheduledEvent(e) for e in events]

    @property
    def pending(self) -> list[ScheduledEvent]:
        return [e for e in self._events if not e.fired]

    def all_events(self) -> list[ScheduledEvent]:
        return list(self._events)

    def mark_fired(self, event_id: str, media_time_ms: int) -> ScheduledEvent | None:
        for se in self._events:
            if se.spec.id == event_id and not se.fired:
                se.fired = True
                se.fired_media_time_ms = media_time_ms
                return se
        return None

    def get(self, event_id: str) -> ScheduledEvent | None:
        for se in self._events:
            if se.spec.id == event_id:
                return se
        return None


__all__ = ["EventEngine", "ScheduledEvent"]
