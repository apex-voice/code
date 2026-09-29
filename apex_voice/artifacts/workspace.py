"""Workspace and artifact runtime.

The workspace holds typed artifacts whose every field write is recorded as an immutable
:class:`ArtifactMutation` carrying media time and causal event ids. This
event-sourced history is what lets the grader detect stale-after-correction values (REVISE),
unsupported claims, and premature commits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from apex_voice.schemas.artifact import ArtifactSchema, LifecycleState


@dataclass
class ArtifactMutation:
    """One immutable field write."""

    seq: int
    artifact_id: str
    path: str
    old_value: Any
    new_value: Any
    wall_time_ns: int
    media_time_ms: int
    causal_events: list[str] = field(default_factory=list)


@dataclass
class Artifact:
    """A typed work product with lifecycle and field values."""

    artifact_id: str
    artifact_type: str
    schema: ArtifactSchema | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    lifecycle: LifecycleState = LifecycleState.EMPTY
    # Fields marked stale by a superseding correction awaiting downstream repair (REVISE).
    stale_fields: set[str] = field(default_factory=set)

    def snapshot(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "lifecycle": self.lifecycle.value,
            "fields": copy.deepcopy(self.fields),
            "stale_fields": sorted(self.stale_fields),
        }


class Workspace:
    """Typed, event-sourced professional workspace."""

    def __init__(self) -> None:
        self.artifacts: dict[str, Artifact] = {}
        self.history: list[ArtifactMutation] = []
        self._seq = 0

    # ---- lifecycle -----------------------------------------------------------------
    def create(
        self,
        artifact_id: str,
        artifact_type: str,
        schema: ArtifactSchema | None = None,
        initial_fields: dict[str, Any] | None = None,
        lifecycle: LifecycleState = LifecycleState.EMPTY,
    ) -> Artifact:
        art = Artifact(artifact_id, artifact_type, schema, dict(initial_fields or {}), lifecycle)
        self.artifacts[artifact_id] = art
        return art

    def get(self, artifact_id: str) -> Artifact | None:
        return self.artifacts.get(artifact_id)

    def set_lifecycle(self, artifact_id: str, state: LifecycleState) -> None:
        art = self.artifacts[artifact_id]
        art.lifecycle = state

    # ---- mutation ------------------------------------------------------------------
    def write(
        self,
        artifact_id: str,
        path: str,
        value: Any,
        media_time_ms: int = 0,
        wall_time_ns: int = 0,
        causal_events: list[str] | None = None,
    ) -> ArtifactMutation:
        art = self.artifacts[artifact_id]
        old = art.fields.get(path)
        art.fields[path] = copy.deepcopy(value)
        # A fresh write clears staleness on the directly-written field.
        art.stale_fields.discard(path)
        # Propagate staleness to declared dependents (REVISE dependency graph).
        if art.schema is not None:
            fs = art.schema.field(path)
            if fs is not None:
                for dep in fs.dependents:
                    art.stale_fields.add(dep)
        mut = ArtifactMutation(
            self._seq,
            artifact_id,
            path,
            copy.deepcopy(old),
            copy.deepcopy(value),
            wall_time_ns,
            media_time_ms,
            list(causal_events or []),
        )
        self.history.append(mut)
        self._seq += 1
        return mut

    def mark_stale(self, artifact_id: str, path: str) -> None:
        self.artifacts[artifact_id].stale_fields.add(path)

    # ---- serialization -------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {aid: art.snapshot() for aid, art in self.artifacts.items()}

    def history_records(self) -> list[dict[str, Any]]:
        return [
            {
                "seq": m.seq,
                "artifact_id": m.artifact_id,
                "path": m.path,
                "old": m.old_value,
                "new": m.new_value,
                "media_time_ms": m.media_time_ms,
                "wall_time_ns": m.wall_time_ns,
                "causal_events": m.causal_events,
            }
            for m in self.history
        ]

    def mutations_for(self, artifact_id: str, path: str) -> list[ArtifactMutation]:
        return [m for m in self.history if m.artifact_id == artifact_id and m.path == path]


__all__ = ["Workspace", "Artifact", "ArtifactMutation"]
