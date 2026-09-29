"""Adapter interfaces.

Two contracts:

- :class:`TextAgent` -- the synchronous, turn-based interface used by the C0 deterministic runtime.
  An agent observes the latest user message, tool results, available tools, and a workspace view,
  and returns an :class:`AgentTurn` (text + tool calls + optional end-of-call).
- :class:`AgentAdapter` -- the async streaming contract (audio deltas, tool calls, metadata) for
  realtime speech-to-speech providers, driven by :mod:`apex_voice.harness.realtime_runner`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    """One agent turn in text mode."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    end_call: bool = False


@dataclass
class AgentObservationView:
    """What a text agent sees each turn (never includes hidden user/gold state)."""

    user_text: str | None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    available_tools: list[dict[str, Any]] = field(default_factory=list)
    workspace_view: dict[str, Any] = field(default_factory=dict)
    knowledge_hint: str | None = None
    turn_index: int = 0


@runtime_checkable
class TextAgent(Protocol):
    """Synchronous turn-based agent (C0)."""

    model_id: str

    def reset(self, system_prompt: str, tools: list[dict[str, Any]]) -> None: ...

    def act(self, obs: AgentObservationView) -> AgentTurn: ...


class AgentAdapter(Protocol):
    """Async realtime speech-to-speech adapter contract used by the duplex runner.

    The runner drives turn-taking explicitly (server-side VAD is disabled so every model faces the
    same deterministic user): it streams a user utterance with :meth:`send_audio`, then calls
    :meth:`commit_audio` and :meth:`create_response`. Barge-ins call :meth:`cancel_response` while the
    model is speaking.

    :meth:`events` yields *normalized* event dicts, each with a ``type`` key:

    - ``response.audio.delta`` with ``audio``: raw 24 kHz mono PCM16 bytes
    - ``response.audio_transcript.delta`` with ``text``: what the model is saying
    - ``response.text.delta`` with ``text``: an auxiliary text channel (used only if no transcript)
    - ``response.output_item.done`` with ``tool_call``: ``{"name", "arguments" (JSON str), "call_id"}``
    - ``response.done``: end of the model's turn
    - ``error`` with ``error``: a provider error (logged; a disconnect ends the iterator)

    Any other event types are ignored by the runner.
    """

    model_id: str

    async def start_session(self, config: dict[str, Any]) -> None:
        """Open the connection. ``config`` holds ``instructions``, ``tools`` (function schemas),
        ``voice``, ``modalities``, ``turn_detection`` (always ``None``) and
        ``max_response_output_tokens``."""
        ...

    async def send_audio(self, pcm_chunk: bytes, timestamp_ms: int = 0) -> None:
        """Append 24 kHz mono PCM16 user audio to the input buffer."""
        ...

    async def commit_audio(self) -> None:
        """Mark the end of the user's utterance."""
        ...

    async def create_response(self) -> None:
        """Ask the model to respond to the committed input (and/or to pending tool results)."""
        ...

    async def cancel_response(self) -> None:
        """Stop the in-progress response (user barge-in)."""
        ...

    async def send_text(self, text: str, timestamp_ms: int = 0) -> None: ...

    async def send_tool_result(self, result: dict[str, Any], timestamp_ms: int = 0) -> None:
        """Return a tool result ``{"call_id", "output"}`` and let the model continue."""
        ...

    def events(self) -> AsyncIterator[dict[str, Any]]: ...

    async def close(self) -> None: ...


__all__ = ["ToolCall", "AgentTurn", "AgentObservationView", "TextAgent", "AgentAdapter"]
