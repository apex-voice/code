"""Top-level task specification (``task.yaml``).

A task is a *versioned executable professional workflow* T = (S0, U, K, A, P, E, W, G, R).
:class:`TaskSpec` is the manifest; it references the sibling files that hold each component
(user program, event program, tools, artifact schemas, gold/grading). :class:`LoadedTask`
(in ``apex_voice.tasks``) is the fully-resolved, in-memory bundle.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from apex_voice.config import Condition
from apex_voice.schemas.enums import AutonomyLevel, RiskTier, WorkArchetype


class AutonomySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: AutonomyLevel = AutonomyLevel.A0_PREPARE_ONLY
    approval_required_for: list[str] = Field(default_factory=list)  # action_types


class ScoringSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: str = "professional_task_success"
    critical_gates: list[str] = Field(default_factory=list)  # gate ids from GoldSpec


class TaskSpec(BaseModel):
    """The ``task.yaml`` manifest."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str = "1.0.0"
    title: str
    archetype: WorkArchetype
    profession: str = ""
    industry: str = ""
    risk_tier: RiskTier = RiskTier.R0_ROUTINE
    voice_native: bool = True
    duplex_native: bool = False
    time_budget_s: int = 720
    conditions: list[Condition] = Field(default_factory=lambda: [Condition.C0])

    # Relative paths within the task package.
    taxonomy: str = "taxonomy.yaml"
    initial_state: str = "initial_state.json"
    user_program: str = "user/flow.yaml"
    user_state: str = "user/user_state.yaml"
    reveal_graph: str = "user/reveal_graph.yaml"
    event_program: str = "user/events.yaml"
    persona: str = "user/persona.yaml"
    realization_bank: str = "user/realization_bank.jsonl"
    audio_manifest: str = "user/audio_manifest.json"
    knowledge_dir: str = "knowledge/"
    tools: str = "tools/tool_spec.yaml"
    latent_world: str = "workspace/latent_world.json"
    initial_workspace: str = "workspace/initial_workspace.json"
    artifact_schemas_dir: str = "workspace/artifact_schemas/"
    gold_dir: str = "grading/"

    # Declared load levels so empty knowledge/tool dirs are legal.
    knowledge_load: str = "supplied"  # none | supplied | search
    tool_load: str = "light"  # none | light | moderate | discoverable | multi_app

    autonomy: AutonomySpec = Field(default_factory=AutonomySpec)
    scoring: ScoringSpec = Field(default_factory=ScoringSpec)


__all__ = ["AutonomySpec", "ScoringSpec", "TaskSpec"]
