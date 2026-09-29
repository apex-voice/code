"""Offline tests for the GPT-realtime adapter's GA session payload and event normalization."""

from __future__ import annotations

import base64

from apex_voice.adapters.gpt_realtime import GptRealtimeAdapter


def _adapter() -> GptRealtimeAdapter:
    return GptRealtimeAdapter(ws_url="ws://unused", api_key="test")


def test_session_is_translated_to_ga_shape():
    tools = [GptRealtimeAdapter._as_tool({"name": "save", "parameters": {"type": "object"}})]
    legacy = {
        "modalities": ["text", "audio"],
        "instructions": "be brief",
        "voice": "marin",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
        "max_response_output_tokens": 1024,
        "tools": tools,
        "tool_choice": "auto",
    }
    ga = _adapter()._sanitize_session(legacy)
    assert ga["type"] == "realtime"
    assert ga["output_modalities"] == ["audio"]
    assert ga["instructions"] == "be brief"
    assert ga["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert ga["audio"]["input"]["turn_detection"] == legacy["turn_detection"]
    assert ga["audio"]["output"]["voice"] == "marin"
    assert ga["tools"] == tools and ga["tool_choice"] == "auto"
    for pre_ga in (
        "modalities",
        "voice",
        "input_audio_format",
        "output_audio_format",
        "turn_detection",
        "max_response_output_tokens",
    ):
        assert pre_ga not in ga


def test_session_without_tools_omits_tool_keys():
    ga = _adapter()._sanitize_session({"instructions": "x", "turn_detection": None})
    assert "tools" not in ga and "tool_choice" not in ga


def test_ga_audio_and_tool_events_are_normalized():
    a = _adapter()
    pcm = b"\x01\x02\x03\x04"
    ev = a._normalize({"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
    assert ev["type"] == "response.audio.delta" and ev["audio"] == pcm
    ev = a._normalize(
        {"type": "response.function_call_arguments.done", "name": "save", "arguments": "{}", "call_id": "c1"}
    )
    assert ev["tool_call"] == {"name": "save", "arguments": "{}", "call_id": "c1"}
    assert a._pending_tools == 1
