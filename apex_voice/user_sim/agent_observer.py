"""Agent-act observer.

Maps agent behavior to a small semantic act ontology used by flow triggers. Observation order:
structured provider transcript / tool events first, then deterministic pattern parsing of agent
text. In text mode (C0) this uses tool events plus configurable regex patterns (the realtime runner
uses :mod:`apex_voice.user_sim.llm_observer`); every decision is logged for audit and the observer
never exposes its labels to the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from apex_voice.environments.tools import ToolCallResult


@dataclass
class ObservedAct:
    """A single observed agent act, e.g. act='ASK', field='mailing_address' -> 'ASK:mailing_address'."""

    act: str
    field: str | None = None
    value: Any = None
    raw: str = ""
    confidence: float = 1.0
    source: str = "pattern"  # pattern | tool | transcript | classifier

    def key(self) -> str:
        return f"{self.act}:{self.field}" if self.field else self.act


@dataclass
class ActPattern:
    """A regex → act mapping authored per task. Named group ``field`` fills the act field."""

    act: str
    pattern: str
    field: str | None = None  # static field, else uses named group 'field'
    _re: re.Pattern | None = None

    def compiled(self) -> re.Pattern:
        if self._re is None:
            self._re = re.compile(self.pattern, re.IGNORECASE)
        return self._re


class AgentActObserver:
    def __init__(
        self, patterns: list[ActPattern] | None = None, tool_act_map: dict[str, str] | None = None
    ) -> None:
        self.patterns = patterns or []
        # Map a tool name to an act, e.g. {"submit_form": "COMMIT", "request_approval": "REQUEST_APPROVAL"}.
        self.tool_act_map = tool_act_map or {}
        self.log: list[dict[str, Any]] = []

    def observe_text(self, text: str) -> list[ObservedAct]:
        acts: list[ObservedAct] = []
        for pat in self.patterns:
            for m in pat.compiled().finditer(text):
                fld = pat.field
                if fld is None and "field" in m.groupdict():
                    fld = m.group("field")
                value = m.groupdict().get("value")
                acts.append(ObservedAct(pat.act, fld, value, raw=m.group(0), source="pattern"))
        self.log.append({"kind": "text", "text": text, "acts": [a.key() for a in acts]})
        return acts

    def observe_tool(self, call: ToolCallResult) -> list[ObservedAct]:
        act_name = self.tool_act_map.get(call.tool)
        acts: list[ObservedAct] = []
        if act_name:
            # Field derives from a scoped action_type/target if present.
            field_val = None
            if isinstance(call.result, dict):
                field_val = (
                    call.result.get("action_type")
                    or call.result.get("target_id")
                    or call.result.get("artifact_id")
                )
            acts.append(ObservedAct(act_name, field_val, raw=call.tool, source="tool"))
        self.log.append({"kind": "tool", "tool": call.tool, "acts": [a.key() for a in acts]})
        return acts


__all__ = ["AgentActObserver", "ObservedAct", "ActPattern"]
