"""Offline unit tests for the xAI Grok Voice adapter's event translation — no network.

Grok speaks the OpenAI-realtime GA protocol; the adapter remaps its ``output_audio*`` event names and
sources tool calls from ``response.function_call_arguments.done`` (neutralizing the duplicate native
``response.output_item.done`` function_call) to the runner's normalized single-dict contract.
"""

import base64

from apex_voice.adapters.grok_voice import GrokVoiceAdapter


def _a():
    return GrokVoiceAdapter(api_key="test-key")


def test_output_audio_delta_decodes_to_pcm_bytes():
    a = _a()
    pcm = b"\x01\x02\x03\x04"
    out = a._normalize({"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
    assert out["type"] == "response.audio.delta"
    assert out["audio"] == pcm


def test_output_transcript_delta_and_done_map_to_transcript():
    a = _a()
    d = a._normalize({"type": "response.output_audio_transcript.delta", "delta": "Hel"})
    assert d["type"] == "response.audio_transcript.delta" and d["text"] == "Hel"
    done = a._normalize({"type": "response.output_audio_transcript.done", "transcript": "Hello"})
    assert done["type"] == "response.audio_transcript.done" and done["text"] == "Hello"


def test_function_call_arguments_done_becomes_tool_call():
    a = _a()
    out = a._normalize(
        {
            "type": "response.function_call_arguments.done",
            "name": "update_enr_1",
            "arguments": '{"fields":{"a":"b"}}',
            "call_id": "call-1",
        }
    )
    assert out["type"] == "response.output_item.done"
    assert out["tool_call"] == {
        "name": "update_enr_1",
        "arguments": '{"fields":{"a":"b"}}',
        "call_id": "call-1",
    }


def test_native_output_item_done_is_neutralized_to_avoid_double_dispatch():
    a = _a()
    out = a._normalize(
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "name": "update_enr_1", "arguments": "{}", "call_id": "call-1"},
        }
    )
    assert out["type"] == "response.output_item.done"
    assert "tool_call" not in out  # sourced from function_call_arguments.done instead


def test_as_tool_flattens_wrapped_and_bare():
    bare = {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}
    for t in (bare, {"type": "function", "function": bare}):
        flat = GrokVoiceAdapter._as_tool(t)
        assert flat["type"] == "function" and flat["name"] == "f" and "function" not in flat


def test_voice_clamped_to_xai_default():
    a = GrokVoiceAdapter(api_key="k")
    assert a._resolve_voice("soft-spoken-gentleman") == "xai_ara"  # unknown -> default
    assert a._resolve_voice("xai_rex") == "xai_rex"  # xai_ passes through


def test_base_events_still_pass_through():
    a = _a()
    # non-Grok-specific events fall through to the Step3 base normalizer unchanged.
    assert a._normalize({"type": "response.done"})["type"] == "response.done"


def test_parallel_tool_outputs_batched_before_one_continuation():
    """Regression: the smoke hung because the base fired response.create after only the first of
    several parallel tool outputs, leaving the rest unanswered (Grok stalls). All outputs must be
    submitted before exactly one continuation create_response."""
    import asyncio

    a = _a()
    sent = []

    async def fake_send(msg):
        sent.append(msg["type"])

    a._send = fake_send

    # simulate two parallel tool calls arriving in one response
    a._normalize(
        {"type": "response.function_call_arguments.done", "name": "f", "arguments": "{}", "call_id": "c1"}
    )
    a._normalize(
        {"type": "response.function_call_arguments.done", "name": "g", "arguments": "{}", "call_id": "c2"}
    )
    assert a._pending_tools == 2

    async def go():
        await a.send_tool_result({"call_id": "c1", "output": {"ok": True}})
        await a.send_tool_result({"call_id": "c2", "output": {"ok": True}})

    asyncio.run(go())
    # two outputs submitted, then exactly one continuation create
    assert sent == ["conversation.item.create", "conversation.item.create", "response.create"]
    assert a._pending_tools == 0


def test_user_turn_create_response_fires_immediately():
    import asyncio

    a = _a()
    sent = []

    async def fake_send(msg):
        sent.append(msg["type"])

    a._send = fake_send
    asyncio.run(a.create_response())  # no pending tools -> fires
    assert sent == ["response.create"]


def test_error_resets_pending_tools():
    a = _a()
    a._pending_tools = 3
    a._normalize({"type": "error", "error": {"message": "boom"}})
    assert a._pending_tools == 0
