"""Taxonomy validator and dataset coverage report.

Validates a task card and aggregates a coverage table across a task set. A dataset freeze is
blocked if any required taxonomy field is missing or a headline slice falls below a frozen minimum
(enforced by the release packager against these counts).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from apex_voice.schemas.taxonomy import TaxonomySpec


@dataclass
class LintResult:
    task_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def lint_task(tax: TaxonomySpec) -> LintResult:
    res = LintResult(task_id=tax.task_id)
    # Required: a primary artifact unless explicitly optional.
    if tax.workspace.primary_artifact is None and not tax.workspace.artifact_optional:
        res.errors.append("no primary_artifact and artifact_optional is false")
    # Every C2/C3 task should carry >=1 duplex event — warn if not.
    from apex_voice.config import Condition

    if (
        Condition.C2 in tax.conditions or Condition.C3 in tax.conditions
    ) and not tax.interaction.duplex_events:
        res.warnings.append("C2/C3 task declares no duplex_events")
    # Grading should include at least one deterministic mode.
    from apex_voice.schemas.enums import GradingMode

    if all(m == GradingMode.G_JUDGE for m in tax.grading.modes):
        res.errors.append("grading relies solely on G_JUDGE (need a deterministic gate)")
    res.warnings.extend(tax.normalization_warnings())
    return res


@dataclass
class CoverageReport:
    n_tasks: int = 0
    archetypes: Counter = field(default_factory=Counter)
    industries: Counter = field(default_factory=Counter)
    functions: Counter = field(default_factory=Counter)
    delegation_patterns: Counter = field(default_factory=Counter)
    artifact_classes: Counter = field(default_factory=Counter)
    artifact_origins: Counter = field(default_factory=Counter)
    autonomy_levels: Counter = field(default_factory=Counter)
    knowledge_burden: Counter = field(default_factory=Counter)
    tool_burden: Counter = field(default_factory=Counter)
    duplex_phenomena: Counter = field(default_factory=Counter)
    temporal_dynamics: Counter = field(default_factory=Counter)
    user_profiles: Counter = field(default_factory=Counter)
    grading_modes: Counter = field(default_factory=Counter)
    risk_tiers: Counter = field(default_factory=Counter)
    realization_tracks: Counter = field(default_factory=Counter)
    conditions: Counter = field(default_factory=Counter)
    matched_core: int = 0
    human_audit: int = 0
    flagship: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "archetypes": dict(self.archetypes),
            "industries": dict(self.industries),
            "economic_functions": dict(self.functions),
            "delegation_patterns": dict(self.delegation_patterns),
            "artifact_classes": dict(self.artifact_classes),
            "artifact_origins": dict(self.artifact_origins),
            "autonomy_levels": dict(self.autonomy_levels),
            "knowledge_burden": dict(self.knowledge_burden),
            "tool_burden": dict(self.tool_burden),
            "duplex_phenomena": dict(self.duplex_phenomena),
            "temporal_dynamics": dict(self.temporal_dynamics),
            "user_profiles": dict(self.user_profiles),
            "grading_modes": dict(self.grading_modes),
            "risk_tiers": dict(self.risk_tiers),
            "realization_tracks": dict(self.realization_tracks),
            "conditions": dict(self.conditions),
            "matched_core": self.matched_core,
            "human_audit": self.human_audit,
            "flagship": self.flagship,
        }


def taxonomy_coverage(cards: list[TaxonomySpec]) -> CoverageReport:
    rep = CoverageReport(n_tasks=len(cards))
    for c in cards:
        rep.archetypes[c.professional_work.primary_archetype.value] += 1
        rep.industries[c.professional_work.industry_setting] += 1
        rep.functions[c.professional_work.economic_function] += 1
        for p in c.interaction.delegation_patterns:
            rep.delegation_patterns[p.value] += 1
        if c.workspace.primary_artifact:
            rep.artifact_classes[c.workspace.primary_artifact.value] += 1
        rep.artifact_origins[c.workspace.artifact_origin.value] += 1
        rep.autonomy_levels[c.workspace.autonomy_level.value] += 1
        rep.knowledge_burden[c.knowledge.burden.value] += 1
        rep.tool_burden[c.knowledge.tool_burden.value] += 1
        for d in c.interaction.duplex_events:
            rep.duplex_phenomena[d.value] += 1
        rep.temporal_dynamics[c.interaction.temporal_dynamics.value] += 1
        rep.user_profiles[c.interaction.user_profile.value] += 1
        for g in c.grading.modes:
            rep.grading_modes[g.value] += 1
        rep.risk_tiers[c.risk_tier.value] += 1
        rep.realization_tracks[c.user_realization_track.value] += 1
        for cond in c.conditions:
            rep.conditions[cond.value] += 1
        rep.matched_core += int(c.paper_tags.matched_core)
        rep.human_audit += int(c.paper_tags.human_audit)
        rep.flagship += int(c.paper_tags.flagship)
    return rep


__all__ = ["lint_task", "LintResult", "taxonomy_coverage", "CoverageReport"]
