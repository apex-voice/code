"""Gemini Live realtime adapter.

Implements the async :class:`~apex_voice.adapters.base.AgentAdapter` contract against Google's Gemini
Live API (bidirectional WebSocket S2S via the ``google-genai`` SDK), and — crucially — emits the SAME
normalized event dicts that :mod:`apex_voice.harness.realtime_runner` already consumes, so the async
duplex runner drives Gemini with no changes (just a different adapter).

Confirmed specifics:
- SDK: ``from google import genai``; ``client.aio.live.connect(model, config)`` (async context mgr,
  entered manually for the session lifecycle). API key from ``GEMINI_API_KEY`` (or
  ``GOOGLE_API_KEY``); model from ``GEMINI_LIVE_MODEL`` (default ``gemini-3.8-live``).
- Audio: input PCM16 mono **16 kHz** (we resample the runner's 24 kHz frames down); output PCM16 mono
  **24 kHz** (emitted as-is onto the runner's R channel).
- Turn control: automatic VAD DISABLED for deterministic matched conditions — we bracket each user
  turn with activity_start/activity_end (commit_audio -> activity_end; Gemini then auto-responds, so
  create_response is a no-op). Barge-in (cancel_response) opens a new activity, which interrupts.
- Transcription: output_audio_transcription -> response.audio_transcript.delta; input_audio_
  transcription -> conversation.item.input_audio_transcription.completed.
- Tools: OpenAI-style {name,description,parameters} -> Gemini FunctionDeclaration; a tool call arrives
  as response.output_item.done with tool_call={name,arguments(JSON str),call_id}; results returned via
  session.send_tool_response.

``google.genai`` is imported lazily so offline CI/import is unaffected.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from apex_voice.config import Condition
from apex_voice.credentials import env_first, require_env
from apex_voice.harness.audio import resample

_INPUT_SR = 16000  # Gemini Live input
_OUTPUT_SR = 24000  # Gemini Live output
_DEFAULT_MODEL = "gemini-3.8-live"  # the model evaluated in the paper
# Fallback order used only when the model is explicitly set to "auto".
_MODEL_CANDIDATES = [
    "gemini-3.8-live",
    "gemini-live-2.5-flash-preview",
    "gemini-2.5-flash-preview-native-audio-dialog",
    "gemini-2.0-flash-live-001",
    "gemini-2.0-flash-exp",
]
_WORKING_MODEL: str | None = None  # module cache of the first model that opened this session
_MAX_1008_RETRIES = 4  # 1008 = quota/rate/policy -> back off + retry same model
_1008_BACKOFF_S = 8.0


def _to_gemini_schema(s: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a JSON-schema dict into the subset Gemini accepts (drops additionalProperties;
    uppercases type; recurses into properties/items)."""
    if not isinstance(s, dict):
        return {"type": "STRING"}
    out: dict[str, Any] = {}
    t = str(s.get("type", "string")).upper()
    out["type"] = t
    if "description" in s:
        out["description"] = s["description"]
    if t == "OBJECT":
        props = s.get("properties") or {}
        out["properties"] = {k: _to_gemini_schema(v) for k, v in props.items()}
        req = [r for r in s.get("required", []) if r in props]
        if req:
            out["required"] = req
    if t == "ARRAY" and "items" in s:
        out["items"] = _to_gemini_schema(s["items"])
    return out


class GeminiLiveAdapter:
    """Async realtime S2S adapter for Gemini Live, event-compatible with the Step-Audio3 adapter."""

    model_id = "gemini_live"
    supported_conditions = [Condition.C1, Condition.C2, Condition.C3]

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        input_sample_rate: int = _INPUT_SR,
        output_sample_rate: int = _OUTPUT_SR,
    ) -> None:
        self.api_key = api_key or env_first("GEMINI_API_KEY", "GOOGLE_API_KEY")
        model = model or env_first("GEMINI_LIVE_MODEL", default=_DEFAULT_MODEL)
        self.model = None if model == "auto" else model  # None -> first of _MODEL_CANDIDATES that opens
        self.resolved_model: str | None = None
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self._client = None
        self._cm = None
        self._session = None
        self._reader: asyncio.Task | None = None
        self._recv_q: asyncio.Queue = asyncio.Queue()
        self._activity_open = False
        self._call_names: dict[str, str] = {}

    # ---- session lifecycle ---------------------------------------------------------------
    def _client_ok(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
        return self._client

    def _build_config(self, config: dict[str, Any]):
        from google.genai import types

        decls = []
        for t in config.get("tools") or []:
            decls.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=_to_gemini_schema(t.get("parameters", {"type": "object"})),
                )
            )
        cfg: dict[str, Any] = {
            "response_modalities": ["AUDIO"],  # Live allows one modality; transcript covers text
            "system_instruction": config.get("instructions", "You are a professional voice assistant."),
            "output_audio_transcription": {},
            "input_audio_transcription": {},
            "realtime_input_config": {"automatic_activity_detection": {"disabled": True}},
        }
        if decls:
            cfg["tools"] = [types.Tool(function_declarations=decls)]
        return types.LiveConnectConfig(**cfg)

    async def start_session(self, config: dict[str, Any]) -> None:
        if not self.api_key:
            self.api_key = require_env("GEMINI_API_KEY", "GOOGLE_API_KEY", purpose="Gemini Live")
        client = self._client_ok()
        cfg = self._build_config(config)
        # Prefer an already-proven model (module cache) so we don't waste attempts on 404 candidates.
        global _WORKING_MODEL
        if self.model:
            candidates = [self.model]
        elif _WORKING_MODEL:
            candidates = [_WORKING_MODEL] + [m for m in _MODEL_CANDIDATES if m != _WORKING_MODEL]
        else:
            candidates = list(_MODEL_CANDIDATES)
        last_err: Exception | None = None
        for m in candidates:
            # A 1008 is quota/rate/policy (transient-ish), NOT "model missing": back off and retry the
            # SAME model a few times before giving up; only fall through to the next candidate on 404.
            for attempt in range(_MAX_1008_RETRIES + 1):
                try:
                    self._cm = client.aio.live.connect(model=m, config=cfg)
                    self._session = await self._cm.__aenter__()
                    self.resolved_model = m
                    _WORKING_MODEL = m
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    self._cm = None
                    msg = str(e)
                    if "1008" in msg and attempt < _MAX_1008_RETRIES:
                        await asyncio.sleep(_1008_BACKOFF_S * (2**attempt))
                        continue
                    break  # not-found / non-retryable -> try next candidate
            if self._session is not None:
                break
        if self._session is None:
            raise RuntimeError(f"Could not open a Gemini Live session (tried {candidates}): {last_err}")
        self._reader = asyncio.create_task(self._read_loop())

    # ---- client -> server ----------------------------------------------------------------
    async def _ensure_activity(self) -> None:
        from google.genai import types

        if not self._activity_open:
            await self._session.send_realtime_input(activity_start=types.ActivityStart())
            self._activity_open = True

    async def send_audio(self, pcm_chunk: bytes, timestamp_ms: int = 0) -> None:
        from google.genai import types

        pcm16 = np.frombuffer(pcm_chunk, dtype=np.int16)
        if self.output_sample_rate != self.input_sample_rate and pcm16.size:
            f = pcm16.astype(np.float32) / 32768.0
            f = resample(f, self.output_sample_rate, self.input_sample_rate)
            pcm16 = np.clip(f * 32768.0, -32768, 32767).astype(np.int16)
        await self._ensure_activity()
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm16.tobytes(), mime_type=f"audio/pcm;rate={self.input_sample_rate}")
        )

    async def commit_audio(self) -> None:
        """End the user turn; Gemini then auto-generates a response."""
        from google.genai import types

        if self._activity_open:
            await self._session.send_realtime_input(activity_end=types.ActivityEnd())
            self._activity_open = False

    async def create_response(self) -> None:
        return  # no-op: Gemini responds automatically after activity_end

    async def cancel_response(self) -> None:
        """Barge-in: opening a new user activity interrupts the in-progress model response."""
        await self._ensure_activity()

    async def send_text(self, text: str, timestamp_ms: int = 0) -> None:
        await self._session.send_realtime_input(text=text)

    async def send_tool_result(self, result: dict[str, Any], timestamp_ms: int = 0) -> None:
        from google.genai import types

        call_id = result.get("call_id") or ""
        output = result.get("output", result)
        if not isinstance(output, dict):
            output = {"result": output}
        fr = types.FunctionResponse(
            id=call_id or None, name=self._call_names.get(call_id, ""), response=output
        )
        await self._session.send_tool_response(function_responses=[fr])

    # ---- server -> client ----------------------------------------------------------------
    async def _read_loop(self) -> None:
        # `session.receive()` yields messages for ONE turn then the generator ends; re-invoke it for
        # each subsequent turn until the session is closed (the reader task is cancelled in close()).
        try:
            while self._session is not None:
                got = False
                async for msg in self._session.receive():
                    got = True
                    for ev in self._normalize(msg):
                        await self._recv_q.put(ev)
                if not got:
                    await asyncio.sleep(0.05)  # nothing pending; yield before re-arming receive()
        except asyncio.CancelledError:  # normal shutdown
            raise
        except Exception as e:  # pragma: no cover - connection drop
            await self._recv_q.put({"type": "error", "error": str(e)})
            await self._recv_q.put(None)

    def _normalize(self, msg: Any) -> list[dict[str, Any]]:
        evs: list[dict[str, Any]] = []
        data = getattr(msg, "data", None)
        if data:  # inline audio chunk (PCM16 @ 24k)
            evs.append({"type": "response.audio.delta", "audio": data})
        sc = getattr(msg, "server_content", None)
        if sc is not None:
            ot = getattr(sc, "output_transcription", None)
            if ot is not None and getattr(ot, "text", None):
                evs.append({"type": "response.audio_transcript.delta", "text": ot.text})
            it = getattr(sc, "input_transcription", None)
            if it is not None and getattr(it, "text", None):
                evs.append(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "input_transcript": it.text,
                    }
                )
            if getattr(sc, "turn_complete", False):
                evs.append({"type": "response.audio_transcript.done", "text": ""})
                evs.append({"type": "response.done"})
        tc = getattr(msg, "tool_call", None)
        if tc is not None and getattr(tc, "function_calls", None):
            for fc in tc.function_calls:
                cid = getattr(fc, "id", None) or fc.name
                self._call_names[cid] = fc.name
                args = getattr(fc, "args", None) or {}
                evs.append(
                    {
                        "type": "response.output_item.done",
                        "tool_call": {"name": fc.name, "arguments": json.dumps(dict(args)), "call_id": cid},
                    }
                )
        return evs

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._recv_q.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._cm = None
        self._session = None


__all__ = ["GeminiLiveAdapter"]
