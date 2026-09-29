"""Reference text-control agents used to validate tasks.

- :class:`ScriptedAgent` -- follows a fixed ordered list of turns. Used as the *oracle* (a correct
  reference policy) and for adversarial QA (over-eager, prohibited-question, premature-commit
  agents) by scripting the appropriate turn sequence.
- :class:`NoOpAgent` -- always silent, never acts. The benchmark must ensure inaction cannot pass
  (a validated task must fail under the no-op agent).
"""

from __future__ import annotations

from typing import Any

from apex_voice.adapters.base import AgentObservationView, AgentTurn, ToolCall


class ScriptedAgent:
    """Emits a predetermined turn each time it is asked to act (deterministic reference policy)."""

    def __init__(self, model_id: str, turns: list[AgentTurn]) -> None:
        self.model_id = model_id
        self._turns = list(turns)
        self._i = 0

    def reset(self, system_prompt: str, tools: list[dict[str, Any]]) -> None:
        self._i = 0

    def act(self, obs: AgentObservationView) -> AgentTurn:
        if self._i >= len(self._turns):
            return AgentTurn(end_call=True)
        turn = self._turns[self._i]
        self._i += 1
        return turn


class NoOpAgent:
    """Never speaks or acts; used to prove inaction cannot pass a task."""

    model_id = "noop"

    def reset(self, system_prompt: str, tools: list[dict[str, Any]]) -> None:
        return None

    def act(self, obs: AgentObservationView) -> AgentTurn:
        return AgentTurn(text="", end_call=True)


def turn(
    text: str = "", tools: list[tuple[str, dict[str, Any]]] | None = None, end: bool = False
) -> AgentTurn:
    """Convenience constructor for scripted turns."""
    return AgentTurn(
        text=text,
        tool_calls=[ToolCall(n, a) for n, a in (tools or [])],
        end_call=end,
    )


__all__ = ["ScriptedAgent", "NoOpAgent", "turn"]
