"""Step-Audio3 Realtime adapter.

Implements the async :class:`~apex_voice.adapters.base.AgentAdapter` streaming contract against
StepFun's realtime chat API — an OpenAI-realtime-compatible WebSocket S2S protocol.

Protocol (confirmed against https://platform.stepfun.ai/docs/en/api-reference/realtime/chat):
- URL:  ``wss://api.stepfun.ai/v1/realtime?model=stepaudio-3-realtime-preview``
        (also ``stepaudio-2.5-realtime``)
- Auth: bearer token from ``STEPFUN_API_KEY``; endpoint overridable via ``STEPFUN_REALTIME_URL``.
- Handshake: server emits ``session.created`` on connect; client sends ``session.update`` with
  modalities/instructions/voice/{input,output}_audio_format=pcm16/turn_detection/tools.
- Input audio: ``input_audio_buffer.append`` (base64 PCM16) + ``input_audio_buffer.commit`` to close a
  user turn (server_vad also auto-commits). Text: ``conversation.item.create`` input_text.
- Output: ``response.audio.delta`` (base64 PCM16), ``response.audio_transcript.delta/.done``,
  ``response.text.delta/.done``, ``response.output_item.done`` (carries function_call items),
  ``response.done``. VAD: ``input_audio_buffer.speech_started/stopped``. Interrupt: ``response.cancel``.
- Tools: OpenAI-style ``function`` defs in ``session.tools``; function_call output items are returned
  via ``conversation.item.create`` (function_call_output) followed by ``response.create``.

``websockets`` is imported lazily so offline CI is unaffected.
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

_DEFAULT_URL = "wss://api.stepfun.ai/v1/realtime"
_DEFAULT_MODEL = "stepaudio-3-realtime-preview"
# StepFun docs do not pin the PCM sample rate; realtime PCM16 S2S is 24 kHz mono (OpenAI-realtime
# convention). Overridable via constructor if a probe shows otherwise.
_INPUT_SR = 24000
_OUTPUT_SR = 24000


def _eid() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


class Step3RealtimeAdapter:
    """Async realtime S2S adapter for Step-Audio3 (OpenAI-realtime-compatible WebSocket)."""

    model_id = "step_audio3_realtime"
    supported_conditions = [Condition.C1, Condition.C2, Condition.C3]

    def __init__(
        self,
        ws_url: str | None = None,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        input_sample_rate: int = _INPUT_SR,
        output_sample_rate: int = _OUTPUT_SR,
        voice: str = "soft-spoken-gentleman",
    ) -> None:
        self.ws_url = ws_url or env_first("STEPFUN_REALTIME_URL", default=_DEFAULT_URL)
        self.model = model
        self.api_key = api_key
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.voice = voice
        self._ws = None
        self._recv_q: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self.session_id: str | None = None
        # OpenAI-realtime allows only ONE active response at a time; a second response.create while
        # one is in progress is rejected ("conversation_already_has_active_response"). Track the
        # active-response lifecycle so create_response() coalesces instead of colliding.
        self._response_active = False
        self._pending_create = False

    def _default_api_key(self) -> str:
        return require_env("STEPFUN_API_KEY", purpose="Step-Audio3 realtime")

    # ---- session lifecycle ---------------------------------------------------------------
    async def start_session(self, config: dict[str, Any]) -> None:
        """Connect, wait for session.created, then push session.update. ``config`` may carry
        ``instructions`` (system prompt), ``tools`` (OpenAI function schemas), ``voice``,
        ``turn_detection`` and ``modalities``."""
        if not self.api_key:
            self.api_key = self._default_api_key()
        try:
            from websockets.asyncio.client import connect
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pip install websockets to use the realtime adapter") from e

        url = f"{self.ws_url}?model={self.model}"
        self._ws = await connect(
            url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
            open_timeout=30,
        )
        self._reader = asyncio.create_task(self._read_loop())

        session: dict[str, Any] = {
            "modalities": config.get("modalities", ["text", "audio"]),
            "instructions": config.get("instructions", "You are a helpful professional voice assistant."),
            "voice": self._resolve_voice(config.get("voice", self.voice)),
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": config.get(
                "turn_detection", {"type": "server_vad", "prefix_padding_ms": 300, "silence_duration_ms": 500}
            ),
        }
        if config.get("max_response_output_tokens") is not None:
            session["max_response_output_tokens"] = config["max_response_output_tokens"]
        tools = config.get("tools")
        if tools:
            session["tools"] = [self._as_tool(t) for t in tools]
            # Some OpenAI-realtime servers (gpt-realtime) default to NOT calling tools unless
            # tool_choice is set; subclasses set self.tool_choice="auto". Left unset for StepFun
            # (Step-Audio3), which already calls tools by default.
            tc = config.get("tool_choice") or getattr(self, "tool_choice", None)
            if tc:
                session["tool_choice"] = tc
        session = self._sanitize_session(session)
        await self._send({"event_id": _eid(), "type": "session.update", "session": session})

    def _sanitize_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Last-chance hook to drop/rename session fields a specific server rejects. Base is a
        passthrough; a subclass whose endpoint refuses a field overrides this. Critical because an
        OpenAI-realtime server rejects the ENTIRE session.update on one unknown field, silently
        dropping tools/tool_choice -> 0 tool calls, empty artifact."""
        return session

    def _resolve_voice(self, voice: str) -> str:
        """Map an incoming voice id to one this server accepts. Base is a passthrough (StepFun
        takes free-form voice names); subclasses with a fixed voice allowlist (gpt-realtime)
        override to clamp — an unsupported voice makes the server reject the whole session.update,
        which silently drops tools/tool_choice and yields 0 tool calls."""
        return voice

    @staticmethod
    def _as_tool(t: dict[str, Any]) -> dict[str, Any]:
        """Accept either an already-wrapped {type:function, function:{...}} or a bare
        {name, description, parameters} schema and normalize to the StepFun/OpenAI shape."""
        if t.get("type") == "function" and "function" in t:
            return t
        return {
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    # ---- client -> server ----------------------------------------------------------------
    async def send_audio(self, pcm_chunk: bytes, timestamp_ms: int = 0) -> None:
        await self._send(
            {
                "event_id": _eid(),
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_chunk).decode(),
            }
        )

    async def commit_audio(self) -> None:
        """Close the current user audio turn (needed when not relying on server_vad)."""
        await self._send({"event_id": _eid(), "type": "input_audio_buffer.commit"})

    async def create_response(self) -> None:
        """Request a model turn. If a response is already active, defer (coalesce) rather than send
        a colliding response.create — the deferred request fires when the active response completes."""
        if self._response_active:
            self._pending_create = True
            return
        self._response_active = True
        await self._send({"event_id": _eid(), "type": "response.create"})

    async def cancel_response(self) -> None:
        """Barge-in: cancel the in-progress model response."""
        self._response_active = False
        self._pending_create = False
        await self._send({"event_id": _eid(), "type": "response.cancel"})

    async def send_text(self, text: str, timestamp_ms: int = 0) -> None:
        await self._send(
            {
                "event_id": _eid(),
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def send_tool_result(self, result: dict[str, Any], timestamp_ms: int = 0) -> None:
        """Return a function_call_output item then request a new response (OpenAI-realtime flow)."""
        call_id = result.get("call_id") or result.get("id") or ""
        output = result.get("output", result)
        await self._send(
            {
                "event_id": _eid(),
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output if isinstance(output, str) else json.dumps(output),
                },
            }
        )
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
                mtype = msg.get("type")
                if mtype == "session.created":
                    self.session_id = (msg.get("session") or {}).get("id")
                elif mtype in ("response.created",):
                    self._response_active = True
                elif mtype in ("response.done", "response.cancelled"):
                    self._response_active = False
                    if self._pending_create:
                        self._pending_create = False
                        self._response_active = True
                        await self._send({"event_id": _eid(), "type": "response.create"})
                elif mtype == "error":
                    # A rejected response.create (e.g. empty buffer) would otherwise leave the
                    # optimistic active flag stuck; clear it so future turns can create responses.
                    self._response_active = False
                    self._pending_create = False
                await self._recv_q.put(self._normalize(msg))
        except Exception as e:  # pragma: no cover - connection drop
            await self._recv_q.put({"type": "error", "error": str(e)})
        finally:
            await self._recv_q.put(None)

    def _normalize(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw server event; decode audio deltas to raw PCM16 bytes.

        Returns a dict always carrying ``type`` and ``raw`` (the original), plus decoded fields:
        ``audio`` (bytes) for audio.delta, ``text`` for transcript/text deltas & dones, and
        ``tool_call`` ({name, arguments, call_id}) when an output item is a function_call.
        """
        t = msg.get("type", "")
        out: dict[str, Any] = {"type": t, "raw": msg}
        if t == "response.audio.delta":
            b64 = msg.get("delta") or msg.get("audio")  # StepFun/OpenAI carry base64 in "delta"
            if b64:
                out["audio"] = base64.b64decode(b64)
        elif t in ("response.audio_transcript.delta", "response.text.delta"):
            out["text"] = msg.get("delta", "")
        elif t in ("response.audio_transcript.done", "response.text.done"):
            out["text"] = msg.get("transcript") or msg.get("text", "")
        elif t == "conversation.item.input_audio_transcription.completed":
            out["input_transcript"] = msg.get("transcript", "")
        elif t == "response.output_item.done":
            # Canonical source of a completed function call (arguments fully assembled). The matching
            # .added / function_call_arguments.delta events stream partial args; ignore for dispatch.
            item = msg.get("item", {})
            if item.get("type") == "function_call":
                out["tool_call"] = {
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "call_id": item.get("call_id") or item.get("id"),
                }
        elif t == "error":
            out["error"] = msg.get("error")
        return out

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._recv_q.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        self._response_active = False
        self._pending_create = False
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


__all__ = ["Step3RealtimeAdapter"]
