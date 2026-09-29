"""Task card / taxonomy schema.

This mirrors the required ``taxonomy.yaml`` contract exactly so the taxonomy validator
and the dataset coverage reports can be generated directly from task metadata rather
than hand-labeled for the paper.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apex_voice.config import Condition, RealizationTrack
from apex_voice.schemas.enums import (
    ECONOMIC_FUNCTIONS,
    INDUSTRY_SETTINGS,
    ArtifactClass,
    ArtifactOrigin,
    AutonomyLevel,
    DelegationPattern,
    DuplexPhenomenon,
    GradingMode,
    KnowledgeBurden,
    RiskTier,
    TemporalDynamics,
    ToolBurden,
    UserProfile,
    WorkArchetype,
)


class ProfessionalWork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_archetype: WorkArchetype
    secondary_archetypes: list[WorkArchetype] = Field(default_factory=list)
    economic_function: str
    industry_setting: str

    @field_validator("economic_function")
    @classmethod
    def _known_function(cls, v: str) -> str:
        # Warn-not-block: unknown values are allowed but flagged by the coverage validator.
        return v

    @field_validator("industry_setting")
    @classmethod
    def _known_setting(cls, v: str) -> str:
        return v

    def normalization_warnings(self) -> list[str]:
        warns: list[str] = []
        if self.economic_function not in ECONOMIC_FUNCTIONS:
            warns.append(f"economic_function '{self.economic_function}' not in normalized vocabulary")
        if self.industry_setting not in INDUSTRY_SETTINGS:
            warns.append(f"industry_setting '{self.industry_setting}' not in normalized vocabulary")
        return warns


class InteractionAxes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delegation_patterns: list[DelegationPattern] = Field(default_factory=list)
    user_profile: UserProfile = UserProfile.COOPERATIVE_STANDARD
    temporal_dynamics: TemporalDynamics = TemporalDynamics.D0_STATIC
    duplex_events: list[DuplexPhenomenon] = Field(default_factory=list)


class WorkspaceAxes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_artifact: ArtifactClass | None = None
    secondary_artifacts: list[ArtifactClass] = Field(default_factory=list)
    artifact_origin: ArtifactOrigin = ArtifactOrigin.APEX_NATIVE
    autonomy_level: AutonomyLevel = AutonomyLevel.A0_PREPARE_ONLY
    artifact_optional: bool = False


class KnowledgeAxes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    burden: KnowledgeBurden = KnowledgeBurden.K0_NONE
    tool_burden: ToolBurden = ToolBurden.T0_NONE


class GradingAxes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modes: list[GradingMode] = Field(default_factory=lambda: [GradingMode.G_STATE])
    critical_gates: list[str] = Field(default_factory=list)


class DifficultyDimensions(BaseModel):
    """L13 — multidimensional difficulty. All optional; never collapsed to a scalar."""

    model_config = ConfigDict(extra="forbid")

    required_fact_count: int = 0
    dependency_depth: int = 0
    number_of_valid_paths: int = 1
    ambiguity_count: int = 0
    correction_count: int = 0
    artifact_field_count: int = 0
    knowledge_doc_count: int = 0
    required_doc_count: int = 0
    tool_call_reference_count: int = 0
    consequential_commit_count: int = 0
    duplex_event_count: int = 0
    environment_change_count: int = 0
    conversation_target_minutes: float = 0.0


class PaperTags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flagship: bool = False
    matched_core: bool = False
    human_audit: bool = False


class TaxonomySpec(BaseModel):
    """The complete task card (``taxonomy.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    professional_work: ProfessionalWork
    interaction: InteractionAxes = Field(default_factory=InteractionAxes)
    workspace: WorkspaceAxes = Field(default_factory=WorkspaceAxes)
    knowledge: KnowledgeAxes = Field(default_factory=KnowledgeAxes)
    grading: GradingAxes = Field(default_factory=GradingAxes)
    conditions: list[Condition] = Field(default_factory=lambda: [Condition.C0])
    risk_tier: RiskTier = RiskTier.R0_ROUTINE
    difficulty: DifficultyDimensions = Field(default_factory=DifficultyDimensions)
    user_realization_track: RealizationTrack = RealizationTrack.FROZEN_SYNTHETIC
    paper_tags: PaperTags = Field(default_factory=PaperTags)

    def normalization_warnings(self) -> list[str]:
        return self.professional_work.normalization_warnings()


__all__ = [
    "ProfessionalWork",
    "InteractionAxes",
    "WorkspaceAxes",
    "KnowledgeAxes",
    "GradingAxes",
    "DifficultyDimensions",
    "PaperTags",
    "TaxonomySpec",
]
