"""Model adapters.

``base.py`` declares the synchronous text-agent interface and the asynchronous realtime
speech-to-speech adapter contract. ``text.py`` provides the reference oracle and no-op text agents
used to validate tasks. The remaining modules implement the realtime adapters for the benchmarked
models; see :mod:`apex_voice.adapters.registry`.
"""

from apex_voice.adapters.base import AgentAdapter, AgentTurn, TextAgent, ToolCall
from apex_voice.adapters.text import NoOpAgent, ScriptedAgent

__all__ = ["AgentAdapter", "AgentTurn", "TextAgent", "ToolCall", "NoOpAgent", "ScriptedAgent"]
