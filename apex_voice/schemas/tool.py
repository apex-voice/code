"""Tool schema.

Every tool declares typed args/results, permissions, latency, side effects, idempotency,
reversibility, and required authority/evidence. Consequential ``commit`` tools route through
the CommitGuard; the schema records that intent declaratively.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolClass(str, Enum):
    """Tool classes."""

    READ_ONLY = "read_only"
    RETRIEVAL = "retrieval"
    STATE_CHANGING = "state_changing"
    REVERSIBLE_DRAFT = "reversible_draft"
    IRREVERSIBLE_COMMIT = "irreversible_commit"
    USER_MEDIATED = "user_mediated"
    DISCOVERABLE = "discoverable"
    CROSS_APPLICATION = "cross_application"


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str  # string | integer | number | boolean | object | array
    required: bool = True
    description: str = ""
    enum: list[Any] | None = None


class LatencyProfile(BaseModel):
    """Injected tool latency. Measured in media milliseconds."""

    model_config = ConfigDict(extra="forbid")

    base_ms: int = 0
    jitter_ms: int = 0  # +/- uniform jitter, seeded
    stress_multiplier: float = 1.0  # 1x / 4x ablation knob


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    tool_class: ToolClass = ToolClass.READ_ONLY
    description: str = ""
    params: list[ParamSpec] = Field(default_factory=list)
    returns: str = "object"
    permissions: list[str] = Field(default_factory=list)
    latency: LatencyProfile = Field(default_factory=LatencyProfile)
    idempotent: bool = True
    reversible: bool = True
    # For commit tools: the action_type + target arg the CommitGuard scopes authorization to.
    commit_action_type: str | None = None
    commit_target_param: str | None = None
    required_authority: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    emits_events: list[str] = Field(default_factory=list)
    # Discoverable tools are hidden from the agent's declared toolset until found via knowledge.
    discoverable: bool = False
    # Runtime binding: name of a built-in handler (see environments.tools.BUILTIN_HANDLERS) or a
    # custom handler registered programmatically. ``effect`` parameterizes the built-in.
    handler: str | None = None
    effect: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_commit(self) -> bool:
        return self.tool_class == ToolClass.IRREVERSIBLE_COMMIT or self.commit_action_type is not None


class ToolSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[ToolSpec] = Field(default_factory=list)

    def by_name(self, name: str) -> ToolSpec | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None


__all__ = ["ToolClass", "ParamSpec", "LatencyProfile", "ToolSpec", "ToolSet"]
