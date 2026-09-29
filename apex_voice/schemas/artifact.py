"""Artifact, latent-world, provenance, and approval schemas.

The latent world is the *source of truth*: artifacts are rendered from it so cross-artifact
consistency is guaranteed. Each artifact carries a source manifest
(origin/license/provenance) that gates release.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from apex_voice.schemas.enums import ArtifactOrigin


class LifecycleState(str, Enum):
    """Artifact lifecycle."""

    EMPTY = "EMPTY"
    PARTIAL = "PARTIAL"
    DRAFT = "DRAFT"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    APPROVED = "APPROVED"
    COMMITTED = "COMMITTED"


class FieldGrader(str, Enum):
    """Deterministic field graders."""

    EXACT = "exact"
    CASEFOLD_EXACT = "casefold_exact"
    NORMALIZED_PHONE = "normalized_phone"
    NORMALIZED_DATE = "normalized_date"
    NORMALIZED_ADDRESS = "normalized_address"
    ENUM = "enum"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    SET_EXACT = "set_exact"
    SET_F1 = "set_f1"
    SET_PRECISION = "set_precision"
    SET_RECALL = "set_recall"
    ORDERED_LIST = "ordered_list"
    INTERVAL_OVERLAP = "interval_overlap"
    SEMANTIC = "semantic"  # free-text: casefold fast-path, else strict LLM judge
    CLAIM_ATOMS = "claim_atoms"  # free-text claim-atom grading


class FieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = "string"
    grader: FieldGrader = FieldGrader.EXACT
    required: bool = False
    enum: list[Any] | None = None
    tolerance: float | None = None  # for numeric_tolerance
    # For dependency propagation (REVISE): fields that go stale when this field changes.
    dependents: list[str] = Field(default_factory=list)


class ArtifactSchema(BaseModel):
    """A typed artifact schema with field-level grading."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: str  # maps to ArtifactClass + a concrete entity/document type
    fields: list[FieldSpec] = Field(default_factory=list)
    lifecycle: list[LifecycleState] = Field(
        default_factory=lambda: [LifecycleState.DRAFT, LifecycleState.READY_FOR_REVIEW]
    )

    def field(self, name: str) -> FieldSpec | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upstream_name: str | None = None
    upstream_asset_id: str | None = None
    license: str | None = None
    transformation: str | None = None


class RendererRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0"


class ArtifactSourceManifest(BaseModel):
    """Mandatory per-artifact provenance record.

    Adapted assets cannot pass release QA with missing upstream provenance/license fields.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    origin: ArtifactOrigin = ArtifactOrigin.APEX_NATIVE
    latent_world_id: str | None = None
    renderer: RendererRef | None = None
    seed: int | None = None
    source_provenance: SourceProvenance = Field(default_factory=SourceProvenance)
    checksum: str | None = None
    # Planned/aspirational origin per the dataset plan when it differs from the actual (native)
    # rendering; the adapted/template targets are a post-licensing coverage gap (120-plan rule 9).
    planned_origin: ArtifactOrigin | None = None

    def provenance_errors(self) -> list[str]:
        """Return blocking errors for the release provenance gate."""
        errs: list[str] = []
        if self.origin == ArtifactOrigin.ADAPTED_OPEN_ASSET:
            p = self.source_provenance
            for name, val in (
                ("upstream_name", p.upstream_name),
                ("upstream_asset_id", p.upstream_asset_id),
                ("license", p.license),
                ("transformation", p.transformation),
            ):
                if not val:
                    errs.append(f"ADAPTED_OPEN_ASSET '{self.artifact_id}' missing provenance.{name}")
        if self.origin == ArtifactOrigin.APEX_NATIVE and self.latent_world_id is None:
            errs.append(f"APEX_NATIVE '{self.artifact_id}' must reference a latent_world_id")
        return errs


class LatentWorld(BaseModel):
    """Structured synthetic source-of-truth state."""

    model_config = ConfigDict(extra="forbid")

    id: str
    seed: int = 0
    entities: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    policies: dict[str, Any] = Field(default_factory=dict)
    # Intentional contradictions authored as task challenges (kept explicit, never accidental).
    authored_inconsistencies: list[str] = Field(default_factory=list)


class ApprovalToken(BaseModel):
    """Action-scoped, revocable authorization."""

    model_config = ConfigDict(frozen=True)

    token_id: str
    action_type: str
    target_id: str
    granted_media_time_ms: int
    source_event_id: str
    revoked_media_time_ms: int | None = None

    def is_valid_for(self, action_type: str, target_id: str, at_media_ms: int) -> bool:
        if self.action_type != action_type or self.target_id != target_id:
            return False
        if self.granted_media_time_ms > at_media_ms:
            return False
        if self.revoked_media_time_ms is not None and self.revoked_media_time_ms <= at_media_ms:
            return False
        return True


__all__ = [
    "LifecycleState",
    "FieldGrader",
    "FieldSpec",
    "ArtifactSchema",
    "SourceProvenance",
    "RendererRef",
    "ArtifactSourceManifest",
    "LatentWorld",
    "ApprovalToken",
]
