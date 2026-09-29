"""Deterministic field-level comparison functions.

Each function returns a float in [0, 1]. These are the deterministic backbone of artifact grading
Free-text ``claim_atoms`` grading is handled separately (needs claim mapping)
and is *not* implemented here as a deterministic score.
"""

from __future__ import annotations

import re
from typing import Any

from apex_voice.schemas.artifact import FieldGrader


def _norm_str(v: Any) -> str:
    return str(v).strip()


def _digits(v: Any) -> str:
    return re.sub(r"\D", "", str(v))


_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        1,
    )
}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})


def _norm_date(v: Any) -> str:
    """Normalize a date to a canonical YYYY-MM-DD key. Handles ISO, US slash, and month-name forms
    ('February 1, 2026', 'Feb 1 2026', '1 February 2026') so an agent's natural date phrasing that
    denotes the SAME calendar date grades as correct. Best-effort, dependency-free."""
    s = str(v).strip().lower()
    # Month-name first (before any 't'-stripping that would mangle 'october' etc.).
    m = re.search(r"([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)  # "february 1, 2026"
    if m and m.group(1) in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9}),?\s+(\d{4})", s)  # "1 february 2026"
    if m and m.group(2) in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)  # ISO
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)  # US M/D/YYYY
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    # Fallback: collapse separators (keeps prior behavior for times / odd formats).
    s = re.sub(r"[^0-9:apm ]", "-", s.replace("t", " "))
    return re.sub(r"-+", "-", s).strip("- ")


def _to_number(v: Any) -> float | None:
    """Extract a numeric value from prose/currency: '$480' -> 480, '4,000' -> 4000, 'about 20' -> 20."""
    m = re.search(r"-?\d[\d,]*\.?\d*", str(v))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _as_set(v: Any) -> set[str]:
    if isinstance(v, (list, tuple, set)):
        return {_norm_str(x).casefold() for x in v}
    return {_norm_str(v).casefold()}


def exact(pred: Any, gold: Any) -> float:
    return 1.0 if _norm_str(pred) == _norm_str(gold) else 0.0


def casefold_exact(pred: Any, gold: Any) -> float:
    return 1.0 if _norm_str(pred).casefold() == _norm_str(gold).casefold() else 0.0


def normalized_phone(pred: Any, gold: Any) -> float:
    return 1.0 if _digits(pred) == _digits(gold) else 0.0


def normalized_date(pred: Any, gold: Any) -> float:
    return 1.0 if _norm_date(pred) == _norm_date(gold) else 0.0


def normalized_address(pred: Any, gold: Any) -> float:
    a = re.sub(r"[^a-z0-9]+", " ", str(pred).lower()).strip()
    b = re.sub(r"[^a-z0-9]+", " ", str(gold).lower()).strip()
    return 1.0 if a == b else 0.0


def enum_match(pred: Any, gold: Any) -> float:
    return exact(pred, gold)


def numeric_tolerance(pred: Any, gold: Any, tol: float = 0.0) -> float:
    p, g = _to_number(pred), _to_number(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if abs(p - g) <= tol else 0.0


def set_exact(pred: Any, gold: Any) -> float:
    return 1.0 if _as_set(pred) == _as_set(gold) else 0.0


def set_prf(pred: Any, gold: Any) -> tuple[float, float, float]:
    p, g = _as_set(pred), _as_set(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def set_f1(pred: Any, gold: Any) -> float:
    return set_prf(pred, gold)[2]


def set_precision(pred: Any, gold: Any) -> float:
    return set_prf(pred, gold)[0]


def set_recall(pred: Any, gold: Any) -> float:
    return set_prf(pred, gold)[1]


def ordered_list(pred: Any, gold: Any) -> float:
    p = list(pred) if isinstance(pred, (list, tuple)) else [pred]
    g = list(gold) if isinstance(gold, (list, tuple)) else [gold]
    return 1.0 if [_norm_str(x) for x in p] == [_norm_str(x) for x in g] else 0.0


def interval_overlap(pred: Any, gold: Any) -> float:
    """Jaccard overlap of two [start, end] numeric intervals."""
    try:
        ps, pe = float(pred[0]), float(pred[1])
        gs, ge = float(gold[0]), float(gold[1])
    except (TypeError, ValueError, IndexError):
        return 0.0
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def grade_field(grader: FieldGrader, pred: Any, gold: Any, tolerance: float | None = None) -> float:
    """Dispatch to the appropriate deterministic grader. Returns [0,1].

    ``CLAIM_ATOMS`` is not deterministically scorable here and returns 0.0; callers route free-text
    fields to the claim-atom grader instead.
    """
    if grader == FieldGrader.NUMERIC_TOLERANCE:
        return numeric_tolerance(pred, gold, tolerance or 0.0)
    fn = {
        FieldGrader.EXACT: exact,
        FieldGrader.CASEFOLD_EXACT: casefold_exact,
        FieldGrader.NORMALIZED_PHONE: normalized_phone,
        FieldGrader.NORMALIZED_DATE: normalized_date,
        FieldGrader.NORMALIZED_ADDRESS: normalized_address,
        FieldGrader.ENUM: enum_match,
        FieldGrader.SET_EXACT: set_exact,
        FieldGrader.SET_F1: set_f1,
        FieldGrader.SET_PRECISION: set_precision,
        FieldGrader.SET_RECALL: set_recall,
        FieldGrader.ORDERED_LIST: ordered_list,
        FieldGrader.INTERVAL_OVERLAP: interval_overlap,
    }.get(grader)
    if grader == FieldGrader.SEMANTIC:
        # Deterministic default (no judge available): casefold fast-path. Two-tier escalation to the
        # LLM judge happens in grade_field_2tier when a judge is supplied.
        return casefold_exact(pred, gold)
    if fn is None:  # CLAIM_ATOMS or unknown
        return 0.0
    return fn(pred, gold)


def _judge_ctx(context: str, artifact: dict | None) -> str:
    if artifact:
        items = ", ".join(f"{k}={v}" for k, v in list(artifact.items())[:40])
        return f"{context} | other fields in the same artifact: {items}"
    return context


# Graders whose mismatches are eligible for identifier-normalization + judge escalation (values a
# real agent may render with equivalent-but-different formatting). Numbers/dates/sets stay strict.
_ID_LIKE = {FieldGrader.EXACT, FieldGrader.CASEFOLD_EXACT, FieldGrader.ENUM}


def grade_field_2tier(
    grader: FieldGrader,
    pred: Any,
    gold: Any,
    tolerance: float | None,
    field: str = "",
    judge: Any = None,
    context: str = "",
    artifact: dict | None = None,
) -> float:
    """Two-tier grade. For SEMANTIC or identifier-like graders (exact/casefold/enum): a trivial
    exact/casefold match short-circuits (no judge call); anything else escalates to the LLM
    ``judge(field, pred, gold, context) -> bool`` with the WHOLE ARTIFACT as context, so equivalent
    identifiers ('CC4419'=='CC-4419') and semantic equivalents aren't wrongly failed. No deterministic
    normalization heuristics (single inspectable judge path). Numbers/dates/phones/sets stay strictly
    deterministic. Judge=None (offline/CI) => deterministic base grader only."""
    if grader == FieldGrader.SEMANTIC or grader in _ID_LIKE:
        # Trivial exact/casefold match short-circuits (identical value -> no judge call). Everything
        # else -> the LLM judge (single, inspectable path; no deterministic normalization heuristics,
        # which can misfire in hard-to-diagnose ways). Judge=None (offline/CI) -> deterministic result.
        base = FieldGrader.CASEFOLD_EXACT if grader == FieldGrader.SEMANTIC else grader
        if grade_field(base, pred, gold, tolerance) >= 1.0:
            return 1.0
        if not str(pred).strip():
            return 0.0
        if judge is not None:
            return 1.0 if judge(field, pred, gold, _judge_ctx(context, artifact)) else 0.0
        return 0.0
    if grader == FieldGrader.NUMERIC_TOLERANCE:
        # Deterministic tolerance check first (exact number within tolerance short-circuits, no judge
        # call). On a miss, escalate to the judge with the numeric target + tolerance in context so a
        # spoken/spelled-out/hedged number ('around eight years' == 8) is accepted while a genuinely
        # DIFFERENT number is still failed. Judge=None (offline/CI) -> deterministic result only.
        if grade_field(grader, pred, gold, tolerance) >= 1.0:
            return 1.0
        if not str(pred).strip():
            return 0.0
        if judge is not None:
            tol = tolerance if tolerance is not None else 0
            nctx = (
                f"This is a NUMERIC field: the reference number is {gold!r}; an answer is correct "
                f"only if it denotes that same number (values within +/-{tol} are equivalent). "
                f"Spoken numbers may be spelled out or hedged (e.g. 'around eight years' == 8, "
                f"'half a million' == 500000). A DIFFERENT number must FAIL. " + (context or "")
            )
            return 1.0 if judge(field, pred, gold, _judge_ctx(nctx.strip(), artifact)) else 0.0
        return 0.0
    return grade_field(grader, pred, gold, tolerance)


__all__ = [
    "grade_field",
    "grade_field_2tier",
    "set_prf",
    "exact",
    "casefold_exact",
    "numeric_tolerance",
    "set_f1",
    "normalized_date",
]
