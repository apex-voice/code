"""User hidden-state manager.

Wraps :class:`UserState` with reveal-eligibility tracking and correction (supersession) handling.
The invariant is non-oracularity: :meth:`may_reveal` returns True only when the fact's reveal rule
is satisfied by observed agent behavior/events, so a fact can never surface before its trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apex_voice.schemas.user import UserFact, UserState


@dataclass
class RevealContext:
    """What the flow engine has observed, used to evaluate reveal rules."""

    observed_act_keys: set[str] = field(default_factory=set)  # cumulative "ASK:field" keys
    fired_events: set[str] = field(default_factory=set)
    entered_states: set[str] = field(default_factory=set)


class UserStateManager:
    def __init__(self, user_state: UserState) -> None:
        self.state = user_state
        self._facts: dict[str, UserFact] = {f.id: f for f in user_state.facts}
        # Track which facts have been revealed and which have been superseded.
        self.revealed: set[str] = set()
        self.superseded: set[str] = set()
        for f in user_state.facts:
            if f.supersedes:
                self.superseded.add(f.supersedes)

    def fact(self, fact_id: str) -> UserFact | None:
        return self._facts.get(fact_id)

    def value(self, fact_id: str) -> Any:
        f = self._facts.get(fact_id)
        return f.value if f else None

    def may_reveal(self, fact_id: str, ctx: RevealContext) -> bool:
        f = self._facts.get(fact_id)
        if f is None:
            return False
        rule = f.reveal
        if rule.may_volunteer:
            return True
        if any(a in ctx.observed_act_keys for a in rule.agent_acts):
            return True
        if any(e in ctx.fired_events for e in rule.events):
            return True
        if any(s in ctx.entered_states for s in rule.on_state):
            return True
        return False

    def mark_revealed(self, fact_id: str) -> None:
        self.revealed.add(fact_id)

    def current_value_for(self, fact_id: str) -> Any:
        """Return the value of the latest (non-superseded) version of a fact chain."""
        # If this fact id has been superseded by a newer version, follow to the newest.
        for f in self.state.facts:
            if f.supersedes == fact_id:
                return self.current_value_for(f.id)
        return self.value(fact_id)


__all__ = ["UserStateManager", "RevealContext"]
