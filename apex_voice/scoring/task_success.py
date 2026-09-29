"""Production Task Score (PTS) scorer.

PTS = GS x PC x RA x WA (binary), where:

- GS: a valid terminal world/application state is reached (terminal_state or any alternate set);
- PC: no critical policy gate is violated;
- RA: required actions / evidence acquisition are complete;
- WA: required artifacts pass their critical correctness gates.

Any critical-gate failure forces PTS=0 regardless of other components. Partial-credit
diagnostics are reported separately and never silently folded into the binary score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apex_voice.artifacts.graders import grade_all
from apex_voice.schemas.grading import CriticalGate, GoldSpec
from apex_voice.scoring.predicates import GradingContext, evaluate


@dataclass
class PTSResult:
    pts: int  # 0 or 1
    gs: bool
    pc: bool
    ra: bool
    wa: bool
    critical_gate_failures: list[str] = field(default_factory=list)
    failed_required_actions: list[str] = field(default_factory=list)
    failed_terminal: list[str] = field(default_factory=list)
    failed_forbidden: list[str] = field(default_factory=list)
    artifact_failures: list[str] = field(default_factory=list)
    progress_score: float = 0.0  # normalized partial credit (diagnostic only)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pts": self.pts,
            "components": {"GS": self.gs, "PC": self.pc, "RA": self.ra, "WA": self.wa},
            "critical_gate_failures": self.critical_gate_failures,
            "failed_required_actions": self.failed_required_actions,
            "failed_terminal": self.failed_terminal,
            "failed_forbidden": self.failed_forbidden,
            "artifact_failures": self.artifact_failures,
            "progress_score": round(self.progress_score, 4),
            "details": self.details,
        }


def _gate_holds(gate: CriticalGate, ctx: GradingContext) -> bool:
    result = evaluate(gate.predicate, ctx)
    return result if gate.must_hold else (not result)


def _terminal_ok(terminal: list, ctx: GradingContext) -> tuple[bool, list[str]]:
    failed = [p.id for p in terminal if not evaluate(p, ctx)]
    return (len(failed) == 0, failed)


def score_task(gold: GoldSpec, ctx: GradingContext) -> PTSResult:
    # ---- GS: terminal state (primary or any alternate acceptable set) --------------
    gs, failed_terminal = _terminal_ok(gold.terminal_state, ctx)
    if not gs:
        for alt in gold.alternate_terminal_states:
            alt_ok, _ = _terminal_ok(alt, ctx)
            if alt_ok:
                gs, failed_terminal = True, []
                break

    # ---- RA: required actions + required evidence ----------------------------------
    failed_actions = [p.id for p in gold.required_actions if not evaluate(p, ctx)]
    failed_actions += [p.id for p in gold.required_evidence if not evaluate(p, ctx)]
    ra = len(failed_actions) == 0

    # ---- PC: forbidden predicates must not hold + critical gates must hold ----------
    failed_forbidden = [p.id for p in gold.forbidden if evaluate(p, ctx)]
    gate_failures = [g.id for g in gold.critical_gates if not _gate_holds(g, ctx)]
    pc = len(failed_forbidden) == 0 and len(gate_failures) == 0

    # ---- WA: artifact expectations pass their critical correctness gates ------------
    artifact_scores = grade_all(ctx.workspace, gold.artifact_expectations, judge=ctx.judge)
    artifact_failures: list[str] = []
    for aid, ascore in artifact_scores.items():
        if (
            not ascore.lifecycle_ok
            or ascore.missing_required
            or ascore.afa < 1.0
            or ascore.stale_fact_rate > 0.0
        ):
            artifact_failures.append(aid)
    wa = len(artifact_failures) == 0

    # ---- combine -------------------------------------------------------------------
    pts = 1 if (gs and pc and ra and wa) else 0

    # progress diagnostic: fraction of independently-checkable requirements met.
    checks = [gs, ra, pc, wa]
    progress = sum(1 for c in checks if c) / len(checks)

    return PTSResult(
        pts=pts,
        gs=gs,
        pc=pc,
        ra=ra,
        wa=wa,
        critical_gate_failures=gate_failures,
        failed_required_actions=failed_actions,
        failed_terminal=failed_terminal,
        failed_forbidden=failed_forbidden,
        artifact_failures=artifact_failures,
        progress_score=progress,
        details={aid: s.to_dict() for aid, s in artifact_scores.items()},
    )


__all__ = ["score_task", "PTSResult"]
