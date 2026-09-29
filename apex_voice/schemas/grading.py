"""Grading specification schemas.

Grading is deterministic-first. A task declares required/forbidden predicates, precedence
constraints, terminal-state constraints, artifact expectations, and critical gates. Headline
Production Task Score (PTS) is ``GS x PC x RA x WA``; any critical gate failure
forces PTS=0.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from apex_voice.schemas.artifact import FieldGrader


class PredicateSpec(BaseModel):
    """A named predicate evaluated against the terminal workspace/world/event-log.

    ``kind`` selects the evaluator; ``args`` parameterizes it. Deterministic evaluators are
    resolved by ``apex_voice.scoring.predicates`` (state, artifact-field, tool-called,
    approval-valid, no-commit-after-revocation, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class PrecedenceConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before: str  # predicate/event id that must precede
    after: str


class CriticalGate(BaseModel):
    """A named critical failure gate. Failure forces PTS=0."""

    model_config = ConfigDict(extra="forbid")

    id: str
    predicate: PredicateSpec
    # If True the predicate must hold; if False it must NOT hold (a forbidden condition).
    must_hold: bool = True


class ArtifactFieldExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    expected: Any = None
    grader: FieldGrader = FieldGrader.EXACT
    required: bool = False
    tolerance: float | None = None


class ClaimAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    proposition: str
    source_fact: str | None = None


class ArtifactExpectation(BaseModel):
    """Per-artifact grading expectations."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_type: str
    lifecycle_min: str | None = None  # e.g. "COMMITTED" or "READY_FOR_REVIEW"
    fields: list[ArtifactFieldExpectation] = Field(default_factory=list)
    required_claims: list[ClaimAtom] = Field(default_factory=list)
    forbidden_claims: list[ClaimAtom] = Field(default_factory=list)
    required_caveats: list[ClaimAtom] = Field(default_factory=list)
    # For REVISE: fields that must reflect the corrected (non-stale) value.
    must_not_be_stale: list[str] = Field(default_factory=list)


class GoldSpec(BaseModel):
    """The complete grading contract (``grading/`` directory in the task package)."""

    model_config = ConfigDict(extra="forbid")

    # GS: terminal world/application state (PTS component).
    terminal_state: list[PredicateSpec] = Field(default_factory=list)
    # RA: required actions / information acquisition.
    required_actions: list[PredicateSpec] = Field(default_factory=list)
    required_evidence: list[PredicateSpec] = Field(default_factory=list)
    forbidden: list[PredicateSpec] = Field(default_factory=list)
    precedence: list[PrecedenceConstraint] = Field(default_factory=list)
    # PC: critical policy gates.
    critical_gates: list[CriticalGate] = Field(default_factory=list)
    # WA: required artifacts.
    artifact_expectations: list[ArtifactExpectation] = Field(default_factory=list)
    # Alternate acceptable terminal states.
    alternate_terminal_states: list[list[PredicateSpec]] = Field(default_factory=list)


__all__ = [
    "PredicateSpec",
    "PrecedenceConstraint",
    "CriticalGate",
    "ArtifactFieldExpectation",
    "ClaimAtom",
    "ArtifactExpectation",
    "GoldSpec",
]
