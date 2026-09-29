"""Whole-artifact re-grading of a run's final workspace (used by the duplex and slice analyses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apex_voice.artifacts.field_graders import grade_field_2tier
from apex_voice.schemas.artifact import FieldGrader
from apex_voice.tasks.loader import LoadedTask

LIFECYCLE_ORDER = ["EMPTY", "PARTIAL", "DRAFT", "READY_FOR_REVIEW", "APPROVED", "COMMITTED"]


@dataclass
class ArtifactGrade:
    fields_passed: int = 0
    fields_total: int = 0
    wa_ok: bool = True  # workspace-artifact gate
    reasons: list[str] = field(default_factory=list)
    missed_required: set[str] = field(default_factory=set)

    @property
    def afa(self) -> float | None:
        """Artifact Field Accuracy: fraction of graded fields that pass."""
        return self.fields_passed / self.fields_total if self.fields_total else None


def grade_artifact(lt: LoadedTask, final_workspace: dict[str, Any], judge: Any) -> ArtifactGrade:
    """Grade every expected artifact field; the WA gate fails on any wrong or stale required field,
    a missing artifact, or a lifecycle below the expected minimum."""
    g = ArtifactGrade()
    for ae in lt.gold.artifact_expectations:
        ad = final_workspace.get(ae.artifact_id, {})
        if not ad:
            g.wa_ok = False
            g.reasons.append("artifact-missing")
            continue
        fields = ad.get("fields", {})
        stale = set(ad.get("stale_fields", []))
        lcmin = getattr(ae, "lifecycle_min", None)
        got = str(ad.get("lifecycle", "")).upper()
        if lcmin and (
            got not in LIFECYCLE_ORDER
            or LIFECYCLE_ORDER.index(got) < LIFECYCLE_ORDER.index(str(lcmin).upper())
        ):
            g.wa_ok = False
            g.reasons.append(f"lifecycle<{lcmin}")
        for fe in ae.fields:
            pred = fields.get(fe.field, "")
            grader = fe.grader.value if hasattr(fe.grader, "value") else str(fe.grader)
            try:
                ok = (
                    grade_field_2tier(
                        FieldGrader(grader),
                        pred,
                        fe.expected,
                        fe.tolerance,
                        field=fe.field,
                        judge=judge,
                        context=ae.artifact_type,
                        artifact=dict(fields),
                    )
                    >= 1.0
                )
            except Exception:  # noqa: BLE001 - an ungradable value counts as wrong
                ok = False
            g.fields_total += 1
            g.fields_passed += int(ok)
            if fe.required and (not ok or fe.field in stale):
                g.wa_ok = False
                g.reasons.append(f"required-field:{fe.field}")
                g.missed_required.add(fe.field)
    return g


__all__ = ["ArtifactGrade", "grade_artifact", "LIFECYCLE_ORDER"]
