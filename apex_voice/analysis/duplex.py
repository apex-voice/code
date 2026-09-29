"""Full-duplex analysis: floor control paired with correction uptake.

Floor control (all runs):
- **overlap** -- mean share of session time in which user and agent speak simultaneously (%).
- **yield** -- share of barge-in style user events (corrections, cancellations, interruptions,
  intent switches) at which the agent stopped speaking (%).
- **stop latency p50/p95** -- time from barge-in onset to agent silence, over genuine barge-ins
  (agent yielded and was actually speaking, ``isl_ms > 10``).

Correction uptake (tasks with mid-speech corrections of required fields):
- **AFA corrected / other** -- Artifact Field Accuracy on user-corrected required fields vs. all
  other required fields, and their **gap**.
- **attrib** -- share of WA-gate failures in which at least one corrected field is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from apex_voice.analysis.artifact import grade_artifact
from apex_voice.analysis.metrics import percentile
from apex_voice.analysis.runs import RunRecord
from apex_voice.tasks.loader import LoadedTask

YIELD_TYPES = frozenset(
    {
        "MID_SPEECH_CORRECTION",
        "CANCELLATION_REVOCATION",
        "USER_BARGE_IN",
        "URGENT_INTERRUPTION",
        "INTENT_SWITCH",
    }
)


def corrected_fields(lt: LoadedTask) -> set[str]:
    """Required gold fields that the user corrects mid-speech somewhere in the task."""
    p = lt.root / "user" / "events.yaml"
    if not p.exists():
        return set()
    d = yaml.safe_load(p.read_text()) or {}
    repaired = {
        str(r).split(".")[-1]
        for e in d.get("events", [])
        if "CORRECTION" in str(e.get("type", ""))
        for r in (e.get("expected_repairs") or [])
    }
    return repaired & required_fields(lt)


def required_fields(lt: LoadedTask) -> set[str]:
    return {
        fe.field for ae in lt.gold.artifact_expectations for fe in ae.fields if getattr(fe, "required", True)
    }


@dataclass
class DuplexStats:
    overlap_pct: float
    yield_pct: float
    stop_p50_ms: float
    stop_p95_ms: float
    n_barge_ins: int
    afa_corrected: float
    afa_other: float
    gap: float
    attrib_pct: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def duplex_stats(
    runs: dict[str, dict[int, RunRecord]], tasks: dict[str, LoadedTask], judge: Any, reps: int = 3
) -> DuplexStats:
    overlap, stop = [], []
    n_events = n_yield = 0
    for by_rep in runs.values():
        for rec in by_rep.values():
            if rec.duplex.get("overlap_ratio") is not None:
                overlap.append(rec.duplex["overlap_ratio"] * 100)
            for e in rec.events():
                if e.get("type") != "DUPLEX_EVENT":
                    continue
                p = e.get("payload") or {}
                if "agent_yielded" not in p or p.get("type") not in YIELD_TYPES:
                    continue
                n_events += 1
                if p.get("agent_yielded"):
                    n_yield += 1
                    isl = p.get("isl_ms")
                    if isl is not None and isl > 10:
                        stop.append(isl)

    cf = ct = nf = nt = wa_fail = wa_fail_corr = 0
    for tid, lt in tasks.items():
        corr = corrected_fields(lt)
        if not corr or tid not in runs:
            continue
        req = required_fields(lt)
        for rep in range(reps):
            rec = runs[tid].get(rep)
            fw = rec.final_workspace() if rec else None
            if fw is None:
                continue
            g = grade_artifact(lt, fw, judge)
            for fld in req:
                bad = fld in g.missed_required
                if fld in corr:
                    ct += 1
                    cf += bad
                else:
                    nt += 1
                    nf += bad
            if not g.wa_ok:
                wa_fail += 1
                wa_fail_corr += bool(corr & g.missed_required)
    afa_c = 100 * (1 - cf / ct) if ct else 0.0
    afa_o = 100 * (1 - nf / nt) if nt else 0.0
    return DuplexStats(
        overlap_pct=sum(overlap) / len(overlap) if overlap else 0.0,
        yield_pct=100 * n_yield / n_events if n_events else 0.0,
        stop_p50_ms=percentile(stop, 0.5) or 0.0,
        stop_p95_ms=percentile(stop, 0.95) or 0.0,
        n_barge_ins=len(stop),
        afa_corrected=afa_c,
        afa_other=afa_o,
        gap=afa_o - afa_c,
        attrib_pct=100 * wa_fail_corr / wa_fail if wa_fail else 0.0,
    )


__all__ = ["DuplexStats", "duplex_stats", "corrected_fields", "required_fields", "YIELD_TYPES"]
