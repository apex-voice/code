"""Run event-log schema.

Every run produces one canonical JSONL event log that must be sufficient to recompute every
score post hoc. Events carry both a monotonic wall timestamp (ns) and a normalized media
timestamp (ms) so latency and media-time semantics stay separable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from apex_voice.config import Condition, RealizationTrack


class EventType(str, Enum):
    """Minimum event classes."""

    RUN_START = "RUN_START"
    USER_STATE_TRANSITION = "USER_STATE_TRANSITION"
    USER_PLAN = "USER_PLAN"
    USER_TEXT = "USER_TEXT"
    USER_AUDIO_START = "USER_AUDIO_START"
    USER_AUDIO_CHUNK = "USER_AUDIO_CHUNK"
    USER_AUDIO_END = "USER_AUDIO_END"
    AGENT_AUDIO_START = "AGENT_AUDIO_START"
    AGENT_AUDIO_CHUNK = "AGENT_AUDIO_CHUNK"
    AGENT_AUDIO_END = "AGENT_AUDIO_END"
    AGENT_TEXT = "AGENT_TEXT"
    AGENT_TRANSCRIPT_PARTIAL = "AGENT_TRANSCRIPT_PARTIAL"
    AGENT_ACT = "AGENT_ACT"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    ARTIFACT_MUTATION = "ARTIFACT_MUTATION"
    WORLD_MUTATION = "WORLD_MUTATION"
    DUPLEX_EVENT = "DUPLEX_EVENT"
    POLICY_EVENT = "POLICY_EVENT"
    APPROVAL_EVENT = "APPROVAL_EVENT"
    RUN_END = "RUN_END"
    INFRA_FAILURE = "INFRA_FAILURE"
    SIMULATOR_FALLBACK = "SIMULATOR_FALLBACK"


class Actor(str, Enum):
    USER = "user"
    AGENT = "agent"
    ENVIRONMENT = "environment"
    HARNESS = "harness"


class RunEvent(BaseModel):
    """One immutable event in the canonical log."""

    model_config = ConfigDict(extra="forbid")

    seq: int  # total-order index within the run
    type: EventType
    actor: Actor
    wall_time_ns: int
    media_time_ms: int
    payload: dict[str, Any] = Field(default_factory=dict)
    causal_parents: list[int] = Field(default_factory=list)  # seq ids
    # Redundant context stamps for standalone auditability of a single line.
    run_id: str = ""
    task_version: str = ""
    condition: Condition = Condition.C0

    def to_jsonl(self) -> str:
        return self.model_dump_json()


class RunManifest(BaseModel):
    """Top-level manifest for a run directory."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    task_version: str
    model_id: str
    condition: Condition
    realization_track: RealizationTrack
    simulator_seed: int
    channel_seed: int
    event_seed: int
    started_at: str
    ended_at: str | None = None
    outcome: str | None = None  # SUCCESS | FAILURE | INFRA_FAILURE | SIMULATOR_FALLBACK
    fallback_events: int = 0
    infra_failures: int = 0
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["EventType", "Actor", "RunEvent", "RunManifest"]
