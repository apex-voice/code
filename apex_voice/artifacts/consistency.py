"""Cross-artifact consistency constraints.

A task can fail even when each artifact is individually plausible if the workspace is internally
inconsistent (e.g. ``crm.next_meeting_time == calendar.followup.start_time``). Constraints are
authored as simple relational expressions over ``artifact_id.field`` paths and evaluated against
the terminal workspace snapshot.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from apex_voice.artifacts.workspace import Workspace

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}
_EXPR = re.compile(r"^\s*(.+?)\s*(==|!=|<=|>=|<|>)\s*(.+?)\s*$")


def _resolve(token: str, ws: Workspace) -> Any:
    """Resolve a token to either an artifact field value or a literal."""
    token = token.strip()
    # Literal string/number?
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        return token[1:-1]
    try:
        return float(token) if "." in token else int(token)
    except ValueError:
        pass
    # artifact_id.field.path
    parts = token.split(".", 1)
    if len(parts) == 2:
        art = ws.get(parts[0])
        if art is not None:
            return art.fields.get(parts[1])
    return None


@dataclass
class ConsistencyResult:
    constraint: str
    satisfied: bool
    lhs: Any = None
    rhs: Any = None


def check_constraints(constraints: list[str], ws: Workspace) -> list[ConsistencyResult]:
    results: list[ConsistencyResult] = []
    for c in constraints:
        m = _EXPR.match(c)
        if not m:
            results.append(ConsistencyResult(c, False))
            continue
        lhs_tok, op, rhs_tok = m.group(1), m.group(2), m.group(3)
        lhs, rhs = _resolve(lhs_tok, ws), _resolve(rhs_tok, ws)
        try:
            ok = _OPS[op](lhs, rhs)
        except TypeError:
            ok = False
        results.append(ConsistencyResult(c, bool(ok), lhs, rhs))
    return results


def all_satisfied(constraints: list[str], ws: Workspace) -> bool:
    return all(r.satisfied for r in check_constraints(constraints, ws))


__all__ = ["check_constraints", "all_satisfied", "ConsistencyResult"]
