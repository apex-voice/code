"""xAI Grok Voice adapter (``grok-voice-think-fast-2.0``).

Grok Voice Think 2.0 is served over xAI's realtime WebSocket speech-to-speech API. The protocol is
OpenAI-realtime-compatible, the same family
:class:`~apex_voice.adapters.realtime_step3.Step3RealtimeAdapter` implements, so this is a thin
subclass that adjusts the endpoint, model, auth, voice allowlist, and the GA event-name variants.

Configuration (environment):

- ``XAI_API_KEY``        -- bearer token.
- ``XAI_REALTIME_URL``   -- WebSocket endpoint (default ``wss://api.x.ai/v1/realtime``).

GA event-name deltas vs the Step-Audio3 base (remapped in :meth:`_normalize` to the normalized names
the runner consumes):
- ``response.output_audio.delta``            -> ``response.audio.delta`` (base64 pcm16 in ``delta``)
- ``response.output_audio_transcript.delta`` -> ``response.audio_transcript.delta``
- ``response.output_audio_transcript.done``  -> ``response.audio_transcript.done``
- ``response.function_call_arguments.done``  -> ``response.output_item.done`` (tool_call assembled;
  carries name+call_id+arguments). Grok ALSO emits a native ``response.output_item.done`` function_call
  item, so we NEUTRALIZE that path (emit no tool_call from it) to avoid dispatching each call twice.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from apex_voice.adapters.realtime_step3 import Step3RealtimeAdapter, _eid
from apex_voice.config import Condition
from apex_voice.credentials import env_first, require_env

_DEFAULT_URL = "wss://api.x.ai/v1/realtime"
_DEFAULT_MODEL = "grok-voice-think-fast-2.0"
_DEFAULT_VOICE = "xai_ara"  # known-good Grok voice (server default); clamp unknowns to this.


class GrokVoiceAdapter(Step3RealtimeAdapter):
    """xAI ``grok-voice-think-fast-2.0`` over the xAI realtime WS (OpenAI-realtime GA protocol)."""

    model_id = "grok"
    supported_conditions = [Condition.C1, Condition.C2, Condition.C3]
    # Like gpt-realtime, only call tools when tool_choice is set (verified: session.update with
    # tool_choice="auto" is accepted and the model emits function_call events).
    tool_choice = "auto"

    def __init__(
        self,
        ws_url: str | None = None,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        voice: str = _DEFAULT_VOICE,
        **kw,
    ) -> None:
        super().__init__(
            ws_url=ws_url or env_first("XAI_REALTIME_URL", default=_DEFAULT_URL),
            api_key=api_key,
            model=model,
            voice=voice,
            **kw,
        )
        # Count of tool calls emitted in the current response awaiting their outputs. Grok (like
        # gpt-realtime) HANGS if response.create is issued while a function call still has no output,
        # so on a parallel-call turn we must submit ALL outputs before the single continuation.
        self._pending_tools = 0

    def _default_api_key(self) -> str:
        return require_env("XAI_API_KEY", purpose="Grok Voice")

    async def create_response(self) -> None:
        # Defer the continuation until every parallel function-call output has been submitted
        # (otherwise Grok stalls waiting on the unanswered calls). send_tool_result fires the single
        # create_response once the batch drains; user-turn creates always run (_pending_tools == 0).
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

    def _resolve_voice(self, voice: str) -> str:
        """xAI voices are ``xai_*`` (default ``xai_ara``). An unknown voice — e.g. the runner's
        default ``soft-spoken-gentleman`` — makes the server reject the whole session.update, silently
        dropping tools/tool_choice (0 tool calls). Clamp anything non-``xai_`` to the default."""
        return voice if isinstance(voice, str) and voice.startswith("xai_") else self.voice

    @staticmethod
    def _as_tool(t: dict[str, Any]) -> dict[str, Any]:
        """xAI realtime wants FLAT function tools ({type,name,description,parameters}) — verified
        accepted — not the chat-style {type:function, function:{...}} the Step-Audio3 base emits."""
        f = t["function"] if (t.get("type") == "function" and "function" in t) else t
        return {
            "type": "function",
            "name": f.get("name"),
            "description": f.get("description", ""),
            "parameters": f.get("parameters", {"type": "object", "properties": {}}),
        }

    def _normalize(self, msg: dict[str, Any]) -> dict[str, Any]:
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
            # Canonical tool-call source for Grok (name + call_id + fully-assembled arguments).
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
            # Native function_call item also arrives — neutralize (no tool_call) so the call isn't
            # dispatched twice; we already emitted it from function_call_arguments.done above.
            return {"type": t, "raw": msg}
        if t == "error":
            # A rejected turn must not leave tool-call bookkeeping stuck > 0 (would wedge every future
            # create_response). Clear it alongside the base's response-active reset.
            self._pending_tools = 0
        return super()._normalize(msg)


__all__ = ["GrokVoiceAdapter"]
