"""Artifact schema library (entity and document types).

A small registry of reusable, typed artifact schemas with field-level graders. Tasks may use
these directly or define bespoke schemas in ``workspace/artifact_schemas/``. Each schema declares
its lifecycle so the workspace/commit-guard know whether a COMMITTED state is reachable.
"""

from __future__ import annotations

from apex_voice.schemas.artifact import ArtifactSchema, FieldGrader, FieldSpec, LifecycleState

_DRAFT_REVIEW = [LifecycleState.DRAFT, LifecycleState.READY_FOR_REVIEW]
_FULL = [
    LifecycleState.EMPTY,
    LifecycleState.PARTIAL,
    LifecycleState.DRAFT,
    LifecycleState.READY_FOR_REVIEW,
    LifecycleState.APPROVED,
    LifecycleState.COMMITTED,
]


def structured_form(fields: list[FieldSpec], committed: bool = True) -> ArtifactSchema:
    return ArtifactSchema(
        artifact_type="STRUCTURED_FORM",
        fields=fields,
        lifecycle=_FULL if committed else _DRAFT_REVIEW,
    )


def evidence_matrix(competencies: list[str]) -> ArtifactSchema:
    """An interview / audit evidence record keyed by competency."""
    fields: list[FieldSpec] = [FieldSpec(name="subject_id", grader=FieldGrader.EXACT, required=True)]
    for comp in competencies:
        fields.append(FieldSpec(name=f"{comp}.evidence", type="array", grader=FieldGrader.SET_F1))
        fields.append(
            FieldSpec(
                name=f"{comp}.status",
                type="string",
                grader=FieldGrader.ENUM,
                enum=["covered", "partial", "not_covered"],
            )
        )
    fields.append(FieldSpec(name="notes", type="string", grader=FieldGrader.CLAIM_ATOMS))
    return ArtifactSchema(artifact_type="EVIDENCE_MATRIX", fields=fields, lifecycle=_DRAFT_REVIEW)


# ---- prebuilt common schemas ----------------------------------------------------------


def _expense_report() -> ArtifactSchema:
    return structured_form(
        [
            FieldSpec(name="employee_id", grader=FieldGrader.EXACT, required=True),
            FieldSpec(name="trip_purpose", grader=FieldGrader.CASEFOLD_EXACT, required=True),
            FieldSpec(name="cost_center", grader=FieldGrader.EXACT, required=True),
            FieldSpec(
                name="total_amount",
                type="number",
                grader=FieldGrader.NUMERIC_TOLERANCE,
                tolerance=0.01,
                required=True,
                dependents=["approval_status"],
            ),
            FieldSpec(
                name="line_items", type="array", grader=FieldGrader.SET_F1, dependents=["total_amount"]
            ),
            FieldSpec(
                name="approval_status", grader=FieldGrader.ENUM, enum=["draft", "submitted"], required=True
            ),
        ],
        committed=True,
    )


def _support_ticket() -> ArtifactSchema:
    return ArtifactSchema(
        artifact_type="TICKET",
        fields=[
            FieldSpec(name="customer_id", grader=FieldGrader.EXACT, required=True),
            FieldSpec(name="issue", grader=FieldGrader.CLAIM_ATOMS, required=True),
            FieldSpec(name="resolution", grader=FieldGrader.CLAIM_ATOMS),
            FieldSpec(
                name="status", grader=FieldGrader.ENUM, enum=["open", "resolved", "escalated"], required=True
            ),
        ],
        lifecycle=_FULL,
    )


def _schedule() -> ArtifactSchema:
    return ArtifactSchema(
        artifact_type="SCHEDULE",
        fields=[
            FieldSpec(name="attendee_id", grader=FieldGrader.EXACT, required=True),
            FieldSpec(name="slot_start", grader=FieldGrader.NORMALIZED_DATE, required=True),
            FieldSpec(name="slot_end", grader=FieldGrader.NORMALIZED_DATE),
            FieldSpec(name="status", grader=FieldGrader.ENUM, enum=["held", "booked"], required=True),
        ],
        lifecycle=_FULL,
    )


_REGISTRY = {
    "EXPENSE_REPORT": _expense_report,
    "SUPPORT_TICKET": _support_ticket,
    "SCHEDULE": _schedule,
}


def get_schema(name: str) -> ArtifactSchema:
    """Return a prebuilt schema by registry name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown artifact schema '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def registry_names() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["structured_form", "evidence_matrix", "get_schema", "registry_names"]
