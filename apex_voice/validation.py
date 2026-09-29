"""Task-package validation.

A task package is valid when it (1) loads and passes schema validation, (2) passes the taxonomy
lint, (3) is solved by its reference oracle policy (PTS = 1) without leaking hidden user state, and
(4) is *not* solved by an agent that does nothing. Checks (3) and (4) run the deterministic C0 text
runner offline; no network access or API keys are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from apex_voice.adapters.text import NoOpAgent, ScriptedAgent
from apex_voice.config import Condition, RunConfig


@dataclass
class ValidationReport:
    task_id: str
    loaded: bool = False
    lint_ok: bool = False
    oracle_pass: bool = False
    noop_fail: bool = False
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.loaded and self.lint_ok and self.oracle_pass and self.noop_fail

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "loaded": self.loaded,
            "lint_ok": self.lint_ok,
            "oracle_pass": self.oracle_pass,
            "noop_fail": self.noop_fail,
            "problems": self.problems,
            "warnings": self.warnings,
        }


def validate_task(task_dir: str | Path, run_policies: bool = True) -> ValidationReport:
    """Validate one task package. Never raises for task-level problems; see ``report.problems``."""
    from apex_voice.harness.runner import run_text_task
    from apex_voice.tasks import lint_task, load_task

    task_dir = Path(task_dir)
    rep = ValidationReport(task_id=task_dir.name)
    try:
        lt = load_task(task_dir)
    except Exception as e:  # noqa: BLE001
        rep.problems.append(f"load: {type(e).__name__}: {e}")
        return rep
    rep.task_id, rep.loaded = lt.spec.id, True

    lint = lint_task(lt.taxonomy)
    rep.lint_ok = lint.ok
    rep.problems += [f"taxonomy: {e}" for e in lint.errors]
    rep.warnings += [f"taxonomy: {w}" for w in lint.warnings]
    if not run_policies:
        rep.oracle_pass = rep.noop_fail = True
        return rep

    cfg = RunConfig(task_id=lt.spec.id, task_version=lt.spec.version, condition=Condition.C0)
    if not lt.reference_turns:
        rep.problems.append("oracle: task has no reference/policy.yaml")
    else:
        oracle = run_text_task(lt, ScriptedAgent("oracle", lt.reference_turns), cfg)
        rep.oracle_pass = oracle.success and not oracle.leakage_violations
        if not oracle.success:
            rep.problems.append(f"oracle: did not pass ({oracle.outcome})")
        if oracle.leakage_violations:
            rep.problems.append(f"oracle: leakage {oracle.leakage_violations}")

    noop = run_text_task(load_task(task_dir), NoOpAgent(), cfg)
    rep.noop_fail = not noop.success
    if noop.success:
        rep.problems.append("no-op: an inactive agent passes the task")
    return rep


__all__ = ["ValidationReport", "validate_task"]
