"""Artifact grader.

Grades a terminal workspace against per-artifact :class:`ArtifactExpectation`s, producing the
artifact metric family: Artifact Field Accuracy (AFA), Artifact Completeness (AC), Stale Fact Rate
(SFR), and (when claim atoms are supplied) hooks for Unsupported Artifact Claim Rate (UACR). Only
deterministic field graders contribute to AFA here; claim-atom/free-text grading is delegated to
the judge layer and combined upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apex_voice.artifacts.field_graders import grade_field_2tier
from apex_voice.artifacts.workspace import Workspace
from apex_voice.schemas.artifact import FieldGrader, LifecycleState
from apex_voice.schemas.grading import ArtifactExpectation


@dataclass
class ArtifactFieldResult:
    artifact_id: str
    field: str
    score: float
    present: bool
    required: bool
    stale: bool
    deterministic: bool


@dataclass
class ArtifactScore:
    afa: float = 1.0  # field accuracy over deterministically-graded required+present fields
    completeness: float = 1.0  # required present / required
    stale_fact_rate: float = 0.0  # stale required fields / corrected fields
    lifecycle_ok: bool = True
    field_results: list[ArtifactFieldResult] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    claim_atom_fields: list[str] = field(default_factory=list)  # need judge grading

    def to_dict(self) -> dict[str, Any]:
        return {
            "afa": round(self.afa, 4),
            "completeness": round(self.completeness, 4),
            "stale_fact_rate": round(self.stale_fact_rate, 4),
            "lifecycle_ok": self.lifecycle_ok,
            "missing_required": self.missing_required,
            "claim_atom_fields": self.claim_atom_fields,
            "fields": [
                {
                    "artifact": r.artifact_id,
                    "field": r.field,
                    "score": round(r.score, 4),
                    "present": r.present,
                    "required": r.required,
                    "stale": r.stale,
                }
                for r in self.field_results
            ],
        }


_LIFECYCLE_ORDER = {
    s: i
    for i, s in enumerate(
        [
            LifecycleState.EMPTY,
            LifecycleState.PARTIAL,
            LifecycleState.DRAFT,
            LifecycleState.READY_FOR_REVIEW,
            LifecycleState.APPROVED,
            LifecycleState.COMMITTED,
        ]
    )
}


def grade_artifact(ws: Workspace, exp: ArtifactExpectation, judge: Any = None) -> ArtifactScore:
    art = ws.get(exp.artifact_id)
    score = ArtifactScore()
    if art is None:
        score.afa = 0.0
        score.completeness = 0.0
        score.lifecycle_ok = exp.lifecycle_min is None
        score.missing_required = [f.field for f in exp.fields if f.required]
        return score

    # lifecycle
    if exp.lifecycle_min is not None:
        need = _LIFECYCLE_ORDER.get(LifecycleState(exp.lifecycle_min), 0)
        have = _LIFECYCLE_ORDER.get(art.lifecycle, 0)
        score.lifecycle_ok = have >= need

    det_scores: list[float] = []
    required_total = 0
    required_present = 0
    corrected_fields = 0
    stale_required = 0

    for fexp in exp.fields:
        present = fexp.field in art.fields
        stale = fexp.field in art.stale_fields
        if fexp.required:
            required_total += 1
            if present:
                required_present += 1
            else:
                score.missing_required.append(fexp.field)
        # staleness accounting over must_not_be_stale
        if fexp.field in exp.must_not_be_stale:
            corrected_fields += 1
            if stale:
                stale_required += 1

        if fexp.grader == FieldGrader.CLAIM_ATOMS:
            score.claim_atom_fields.append(fexp.field)
            deterministic = False
            fscore = 0.0
        else:
            deterministic = fexp.grader != FieldGrader.SEMANTIC
            if present and fexp.expected is not None:
                fscore = grade_field_2tier(
                    fexp.grader,
                    art.fields[fexp.field],
                    fexp.expected,
                    fexp.tolerance,
                    field=fexp.field,
                    judge=judge,
                    context=f"{exp.artifact_id} ({exp.artifact_type})",
                    artifact=dict(art.fields),
                )
            elif present and fexp.expected is None:
                fscore = 1.0  # presence-only requirement satisfied
            else:
                fscore = 0.0
            # A stale value never counts as correct even if it textually matches an old expectation.
            if stale:
                fscore = 0.0
            if fexp.required or present:
                det_scores.append(fscore)

        score.field_results.append(
            ArtifactFieldResult(
                exp.artifact_id, fexp.field, fscore, present, fexp.required, stale, deterministic
            )
        )

    score.afa = sum(det_scores) / len(det_scores) if det_scores else 1.0
    score.completeness = required_present / required_total if required_total else 1.0
    score.stale_fact_rate = stale_required / corrected_fields if corrected_fields else 0.0
    return score


def grade_all(
    ws: Workspace, expectations: list[ArtifactExpectation], judge: Any = None
) -> dict[str, ArtifactScore]:
    return {exp.artifact_id: grade_artifact(ws, exp, judge=judge) for exp in expectations}


__all__ = ["grade_artifact", "grade_all", "ArtifactScore", "ArtifactFieldResult"]
