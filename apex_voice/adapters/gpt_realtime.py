"""OpenAI GPT-realtime adapter (``gpt-realtime-2.1``).

GPT-realtime speaks the OpenAI realtime WebSocket protocol, which :class:`Step3RealtimeAdapter`
already implements, so this is a thin subclass that changes the endpoint, model, auth, voice
allowlist, and the handful of protocol differences documented on each override below.

Configuration (environment):

- ``OPENAI_API_KEY``        -- bearer token.
- ``OPENAI_REALTIME_URL``   -- WebSocket endpoint (default ``wss://api.openai.com/v1/realtime``).
  Any OpenAI-compatible realtime gateway may be used instead.

``websockets`` is imported lazily so offline CI is unaffected.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from apex_voice.adapters.realtime_step3 import Step3RealtimeAdapter, _eid
from apex_voice.config import Condition
from apex_voice.credentials import env_first, require_env

_DEFAULT_URL = "wss://api.openai.com/v1/realtime"
_DEFAULT_MODEL = "gpt-realtime-2.1"


class GptRealtimeAdapter(Step3RealtimeAdapter):
    """OpenAI ``gpt-realtime-2.1`` over the OpenAI realtime WebSocket protocol."""

    model_id = "gpt_realtime"
    supported_conditions = [Condition.C1, Condition.C2, Condition.C3]
    # gpt-realtime only calls tools when tool_choice is set (verified via raw WS dump: with
    # tool_choice="auto" it emits response.function_call_arguments.* + a function_call output item;
    # without it, it just talks and never fills the artifact -> 0% AFA).
    tool_choice = "auto"
    # gpt-realtime only accepts this fixed allowlist; an unknown voice (e.g. the Step-Audio3
    # default "soft-spoken-gentleman" the runner passes) makes the server reject the entire
    # session.update, silently dropping tools+tool_choice -> 0 tool calls, empty artifact.
    _OPENAI_VOICES = frozenset(
        {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"}
    )

    def __init__(
        self,
        ws_url: str | None = None,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        voice: str = "marin",
        **kw,
    ) -> None:
        super().__init__(
            ws_url=ws_url or env_first("OPENAI_REALTIME_URL", default=_DEFAULT_URL),
            api_key=api_key,
            model=model,
            voice=voice,
            **kw,
        )
        # Parallel function calls awaiting their outputs. gpt-realtime HANGS if response.create is
        # issued while a call is still unanswered, so on a multi-call turn we submit ALL outputs
        # before the single continuation (mirrors the Grok GA path).
        self._pending_tools = 0

    def _default_api_key(self) -> str:
        return require_env("OPENAI_API_KEY", purpose="GPT-realtime")

    def _resolve_voice(self, voice: str) -> str:
        return voice if voice in self._OPENAI_VOICES else self.voice

    async def create_response(self) -> None:
        # Defer the continuation until every parallel function-call output is in (else the model
        # stalls on the unanswered calls). User-turn creates always run (_pending_tools == 0).
        if self._pending_tools > 0:
            return
        await super().create_response()

    async def send_tool_result(self, result: dict[str, Any], timestamp_ms: int = 0) -> None:
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
        if self._pending_tools > 0:
            self._pending_tools -= 1
        if self._pending_tools == 0:  # all outputs in -> one continuation
            await self.create_response()

    def _normalize(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Map OpenAI-realtime **GA** event names onto the pre-GA names the base adapter handles.

        GA renamed the audio, transcript, and tool events; without this remap every audio and
        transcript event would be dropped and the agent would appear silent. Mirrors the Grok
        adapter."""
        t = msg.get("type", "")
        if t == "response.output_audio.delta":
            out: dict[str, Any] = {"type": "response.audio.delta", "raw": msg}
            b64 = msg.get("delta") or msg.get("audio")
            if b64:
                out["audio"] = base64.b64decode(b64)
            return out
        if t == "response.output_audio_transcript.delta":
            return {"type": "response.audio_transcript.delta", "raw": msg, "text": msg.get("delta", "")}
        if t == "response.output_audio_transcript.done":
            return {
                "type": "response.audio_transcript.done",
                "raw": msg,
                "text": msg.get("transcript") or msg.get("text", ""),
            }
        if t == "response.function_call_arguments.done":
            # Canonical GA tool-call source (name + call_id + fully-assembled arguments).
            self._pending_tools += 1  # awaiting this call's output before the continuation
            return {
                "type": "response.output_item.done",
                "raw": msg,
                "tool_call": {
                    "name": msg.get("name"),
                    "arguments": msg.get("arguments"),
                    "call_id": msg.get("call_id") or msg.get("item_id"),
                },
            }
        if t == "response.output_item.done":
            # Native function_call item also arrives under GA — neutralize so the call isn't
            # dispatched twice (already emitted from function_call_arguments.done above).
            return {"type": t, "raw": msg}
        if t == "error":
            self._pending_tools = 0  # a rejected turn must not wedge future create_response
        return super()._normalize(msg)

    def _sanitize_session(self, session: dict) -> dict:
        """Translate the base (pre-GA) session payload into the OpenAI-realtime **GA** shape.

        The GA endpoint requires ``session.type`` and nests audio settings under ``audio``; a
        pre-GA payload is rejected outright, which fails the whole ``session.update``. GA allows a
        single output modality, and ``"audio"`` still streams transcripts. The session-level token
        cap the runner sets is intentionally dropped (the benchmark runs did not apply it here)."""
        ga: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": session.get("instructions", ""),
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": session.get("turn_detection"),
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": session.get("voice", self.voice),
                },
            },
        }
        for key in ("tools", "tool_choice"):
            if key in session:
                ga[key] = session[key]
        return ga

    @staticmethod
    def _as_tool(t: dict) -> dict:
        """OpenAI realtime wants FLAT function tools ({type,name,description,parameters}) — not the
        chat-style {type:function, function:{...}} that Step-Audio3 accepts."""
        f = t["function"] if (t.get("type") == "function" and "function" in t) else t
        return {
            "type": "function",
            "name": f.get("name"),
            "description": f.get("description", ""),
            "parameters": f.get("parameters", {"type": "object", "properties": {}}),
        }


__all__ = ["GptRealtimeAdapter"]
