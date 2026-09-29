"""OpenAI **GPT-Live** adapter — model ``gpt-live-1``.

GPT-Live is NOT a self-contained speech-to-speech tool-caller like the realtime models. It is a
voice-orchestration layer that **delegates** task cognition to a backend model. We use *Responses
delegation*: OpenAI runs a backend text model (``gpt-4o`` by default; configurable via
``APEX_GPTLIVE_BACKEND``) that performs the actual function-calling, while
``gpt-live-1`` handles the spoken conversation. Reported results therefore reflect the composite
``gpt-live-1`` voice layer + a chosen backend brain (default ``gpt-4o``).

Protocol (discovered empirically; see the Live docs at developers.openai.com/api/docs/guides/live):
- Endpoint ``wss://api.openai.com/v1/live/sessions`` — NO query params (model goes in ``session.start``).
- Auth: bearer token from ``OPENAI_API_KEY``; endpoint overridable via ``OPENAI_LIVE_URL``.
- Handshake: send ``session.start`` {session:{model, audio, delegation}} -> server ``session.started``.
- Audio in:  ``session.input_audio.append`` {audio: base64 pcm16 @ 24 kHz}. Response trigger:
  ``response.create``. No explicit commit needed (create_response processes buffered audio).
- Voice out: ``session.output_audio.delta`` {delta: base64 pcm16 @ 24 kHz} and
  ``session.output_transcript.delta`` {delta: text}.
- Delegation/backend stream is forwarded wrapped as ``response.event`` {event:{<Responses-API event>}}.
  Function calls surface as inner ``response.output_item.done`` (item.type=="function_call": name,
  arguments, call_id). Inner ``response.completed`` terminates a (sub)response — even for a tool-call
  response, it fires without waiting for the tool result, so it maps cleanly to ``response.done``.
- Tool result: ``response.item.create`` {item:{type:function_call_output, call_id, output}} then
  ``response.create`` (continuation). ``websockets`` is imported lazily so offline CI is unaffected.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from apex_voice.config import Condition
from apex_voice.credentials import env_first, require_env

_ENDPOINT = "wss://api.openai.com/v1/live/sessions"
_SR = 24000
_DEFAULT_BACKEND = "gpt-4o"
_OPENAI_VOICES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"}
)


def _eid() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


class GptLive1Adapter:
    """OpenAI ``gpt-live-1`` (GPT-Live) over the Live WebSocket, with Responses delegation to a
    backend model that runs our function tools."""

    model_id = "gpt_live1"
    supported_conditions = [Condition.C1, Condition.C2, Condition.C3]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-live-1",
        backend: str | None = None,
        voice: str = "marin",
        input_sample_rate: int = _SR,
        output_sample_rate: int = _SR,
        endpoint: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.backend = backend or env_first("APEX_GPTLIVE_BACKEND", default=_DEFAULT_BACKEND)
        self.voice = voice
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.endpoint = endpoint or env_first("OPENAI_LIVE_URL", default=_ENDPOINT)
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._recv_q: asyncio.Queue = asyncio.Queue()
        self.session_id: str | None = None
        # Count of function calls in the current response still awaiting their output. Responses
        # delegation rejects response.create while ANY output is pending
        # ("function_call_outputs_required"), so we submit every output first, then create ONE
        # continuation — the runner calls send_tool_result once per collected tool call.
        self._pending_tools = 0

    def _resolve_voice(self, voice: str) -> str:
        return voice if voice in _OPENAI_VOICES else self.voice

    @staticmethod
    def _as_tool(t: dict[str, Any]) -> dict[str, Any]:
        """Backend Responses tools are FLAT function schemas {type,name,description,parameters}."""
        f = t["function"] if (t.get("type") == "function" and "function" in t) else t
        return {
            "type": "function",
            "name": f.get("name"),
            "description": f.get("description", ""),
            "parameters": f.get("parameters", {"type": "object", "properties": {}}),
        }

    # ---- session lifecycle ---------------------------------------------------------------
    async def start_session(self, config: dict[str, Any]) -> None:
        if not self.api_key:
            self.api_key = require_env("OPENAI_API_KEY", purpose="GPT-Live")
        try:
            from websockets.asyncio.client import connect
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pip install websockets to use the GPT-Live adapter") from e

        self._ws = await connect(
            self.endpoint,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
            open_timeout=30,
        )
        self._reader = asyncio.create_task(self._read_loop())

        # Pass the runner's standard instructions through UNMODIFIED — every model in the roster gets
        # the identical prompt, so gpt-live-1 is evaluated on the same footing (no model-specific
        # coaching / tuning, which would compromise benchmark comparability).
        instructions = config.get("instructions", "You are a helpful professional voice assistant.")
        tools = [self._as_tool(t) for t in (config.get("tools") or [])]
        responses: dict[str, Any] = {
            "model": self.backend,
            "instructions": instructions,
            "tools": tools,
            # Backend calls tools only with tool_choice set (the gpt-realtime lesson).
            "tool_choice": config.get("tool_choice", "auto"),
        }
        if config.get("max_response_output_tokens") is not None:
            responses["max_output_tokens"] = config["max_response_output_tokens"]
        session = {
            "model": self.model,
            "instructions": "Speak naturally and briefly; let the backend handle the task.",
            "audio": {
                "format": {"type": "audio/pcm", "rate": _SR},
                "output": {"voice": self._resolve_voice(config.get("voice", self.voice))},
            },
            "delegation": {"type": "responses", "responses": responses},
        }
        await self._send({"type": "session.start", "session": session})
        # Block until session.started (or an error) so a bad config surfaces at startup.
        for _ in range(50):
            await asyncio.sleep(0.05)
            if self.session_id is not None:
                return
        # session.started sets session_id in the read loop; if not set, keep going (best-effort).

    # ---- client -> server ----------------------------------------------------------------
    async def send_audio(self, pcm_chunk: bytes, timestamp_ms: int = 0) -> None:
        await self._send(
            {"type": "session.input_audio.append", "audio": base64.b64encode(pcm_chunk).decode()}
        )

    async def commit_audio(self) -> None:
        # GPT-Live has no explicit buffer commit; response.create processes appended audio.
        return

    async def create_response(self) -> None:
        # The matched-control runner drives turn-taking deterministically (turn_detection=None): it
        # calls create_response exactly once per user turn, and once per tool continuation via
        # send_tool_result after all outputs are in — so each call maps to one server response and
        # there is no in-flight response to collide with. We deliberately do NOT coalesce/defer here:
        # an autonomous deferred response.create desyncs the runner's per-turn lifecycle and silently
        # drops every turn after the first (the model stops responding). If a function-call output is
        # still pending we skip (send_tool_result issues the continuation once the batch is complete).
        if self._pending_tools > 0:
            return
        await self._send({"type": "response.create"})

    async def cancel_response(self) -> None:
        # GPT-Live exposes no response.cancel; barge-in cancel is a no-op. Audio may continue briefly;
        # this only affects duplex/ISL metrics, not artifact grading.
        return

    async def send_text(self, text: str, timestamp_ms: int = 0) -> None:
        # Not used by the matched-control runner (corrections arrive as audio); no-op for safety.
        return

    async def send_tool_result(self, result: dict[str, Any], timestamp_ms: int = 0) -> None:
        call_id = result.get("call_id") or result.get("id") or ""
        output = result.get("output", result)
        await self._send(
            {
                "type": "response.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output if isinstance(output, str) else json.dumps(output),
                },
            }
        )
        # Submit ALL pending outputs before creating the continuation (avoid the
        # function_call_outputs_required rejection when the model made parallel calls).
        if self._pending_tools > 0:
            self._pending_tools -= 1
        if self._pending_tools == 0:
            await self.create_response()

    async def _send(self, msg: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("session not started")
        await self._ws.send(json.dumps(msg))

    # ---- server -> client ----------------------------------------------------------------
    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:  # type: ignore[union-attr]
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                for ev in self._normalize(msg):
                    await self._recv_q.put(ev)
        except Exception as e:  # pragma: no cover - connection drop
            await self._recv_q.put({"type": "error", "error": str(e)})
        finally:
            await self._recv_q.put(None)

    def _normalize(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate a raw GPT-Live event into zero or more runner-normalized events."""
        t = msg.get("type", "")
        if t == "session.started":
            self.session_id = (msg.get("session") or {}).get("id") or "live"
            return []
        if t == "session.output_audio.delta":
            b64 = msg.get("delta") or msg.get("audio")
            return (
                [{"type": "response.audio.delta", "audio": base64.b64decode(b64), "raw": msg}] if b64 else []
            )
        if t == "session.output_transcript.delta":
            return [{"type": "response.audio_transcript.delta", "text": msg.get("delta", ""), "raw": msg}]
        if t == "session.delegation.created":
            return []
        if t == "error":
            return [{"type": "error", "error": msg.get("error"), "raw": msg}]
        if t == "response.event":
            return self._normalize_inner(msg.get("event", {}) or {}, msg)
        return []  # session.input_transcript.delta, session.usage.updated, etc. — not needed

    def _normalize_inner(self, ev: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
        it = ev.get("type", "")
        if it == "response.output_item.done":
            item = ev.get("item", {})
            if item.get("type") == "function_call":
                self._pending_tools += 1
                return [
                    {
                        "type": "response.output_item.done",
                        "raw": raw,
                        "tool_call": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments"),
                            "call_id": item.get("call_id") or item.get("id"),
                        },
                    }
                ]
            return []
        if it in ("response.completed", "response.failed", "response.incomplete"):
            return [{"type": "response.done", "raw": raw}]
        return []

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._recv_q.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        self._pending_tools = 0
        try:
            if self._ws is not None:
                await self._send({"type": "session.close"})
        except Exception:  # noqa: BLE001
            pass
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


__all__ = ["GptLive1Adapter"]
