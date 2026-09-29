"""Cross-slice analysis over the task taxonomy, averaged over repetitions.

For every task, *success* is the mean over repetitions of the full PTS conjunction (GS, PC, RA from
the stored run components; WA re-graded from the final workspace) and AFA pools field passes across
repetitions. A slice cell reports ``success% (AFA%)`` for the tasks carrying that label.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import yaml

from apex_voice.analysis.artifact import grade_artifact
from apex_voice.analysis.runs import RunRecord
from apex_voice.tasks.loader import LoadedTask

# (axis key, title, fixed label order or None -> sort by group size)
AXES: list[tuple[str, str, list[str] | None]] = [
    ("archetype", "Work archetype", None),
    ("industry", "Industry / setting", None),
    ("approve", "Delegation: APPROVE-gating", None),
    ("artifact", "Primary artifact class", None),
    (
        "autonomy",
        "Autonomy / commit level",
        ["A0_PREPARE_ONLY", "A1_DRAFT_CONFIRM", "A2_LOW_RISK_EXECUTE", "A3_APPROVAL_GATED_COMMIT"],
    ),
    ("kburden", "Knowledge burden", ["K0_NONE", "K1_SUPPLIED", "K2_SEARCH_SMALL", "K3_MULTI_DOC_POLICY"]),
    ("tburden", "Tool burden", ["T1_LIGHT", "T2_MODERATE"]),
    ("uprofile", "User behaviour profile", None),
    ("difficulty", "Difficulty: required fact count", ["9 fields", "10 fields", "11-13 fields"]),
    (
        "risk",
        "Risk / safety tier",
        ["R0_ROUTINE", "R1_SENSITIVE_DATA_SIM", "R2_CONSEQUENTIAL_ACTION", "R3_SPECIAL_REVIEW"],
    ),
]


def _fact_band(n: int | None) -> str | None:
    if n is None:
        return None
    if n <= 9:
        return "9 fields"
    if n == 10:
        return "10 fields"
    return "11-13 fields"


def task_labels(lt: LoadedTask) -> dict[str, str | None]:
    t = yaml.safe_load((lt.root / "taxonomy.yaml").read_text()) or {}
    pw, it = t.get("professional_work", {}), t.get("interaction", {})
    ws, kn = t.get("workspace", {}), t.get("knowledge", {})
    return {
        "archetype": pw.get("primary_archetype"),
        "industry": pw.get("industry_setting"),
        "artifact": ws.get("primary_artifact"),
        "autonomy": ws.get("autonomy_level"),
        "kburden": kn.get("burden"),
        "tburden": kn.get("tool_burden"),
        "uprofile": it.get("user_profile"),
        "risk": t.get("risk_tier"),
        "approve": "APPROVE-gated"
        if "APPROVE" in (it.get("delegation_patterns") or [])
        else "not APPROVE-gated",
        "difficulty": _fact_band((t.get("difficulty") or {}).get("required_fact_count")),
    }


@dataclass
class TaskSliceScore:
    success: float
    afa_pass: int
    afa_total: int
    n_reps: int


def per_task_scores(
    runs: dict[str, dict[int, RunRecord]], tasks: dict[str, LoadedTask], judge: Any, reps: int = 3
) -> dict[str, TaskSliceScore]:
    out = {}
    for tid, lt in tasks.items():
        succ, ap, at = [], 0, 0
        for rep in range(reps):
            rec = runs.get(tid, {}).get(rep)
            fw = rec.final_workspace() if rec else None
            if fw is None:
                continue
            g = grade_artifact(lt, fw, judge)
            gates = all(bool(rec.components.get(k)) for k in ("GS", "PC", "RA")) and g.wa_ok
            succ.append(1.0 if gates else 0.0)
            ap += g.fields_passed
            at += g.fields_total
        if succ:
            out[tid] = TaskSliceScore(sum(succ) / len(succ), ap, at, len(succ))
    return out


def slice_rows(
    axis: str,
    order: list[str] | None,
    labels: dict[str, dict[str, str | None]],
    scores: dict[str, dict[str, TaskSliceScore]],
) -> list[dict[str, Any]]:
    """One row per label: ``{"label", "n", <model>: {"success", "afa"}}``."""
    groups: dict[Any, list[str]] = collections.defaultdict(list)
    for tid, lab in labels.items():
        groups[lab[axis]].append(tid)
    keys = order or sorted(groups, key=lambda k: (-len(groups[k]), str(k)))
    rows = []
    for lab in keys:
        tids = groups.get(lab, []) if lab is not None else []
        if not tids:
            continue
        row: dict[str, Any] = {"label": lab, "n": len(tids)}
        for model, per in scores.items():
            present = [t for t in tids if t in per]
            at = sum(per[t].afa_total for t in present)
            row[model] = {
                "success": sum(per[t].success for t in present) / len(present) if present else None,
                "afa": sum(per[t].afa_pass for t in present) / at if at else None,
            }
        rows.append(row)
    return rows


__all__ = ["AXES", "TaskSliceScore", "per_task_scores", "slice_rows", "task_labels"]
