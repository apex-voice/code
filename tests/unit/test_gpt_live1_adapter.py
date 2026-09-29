"""Offline unit tests for the GPT-Live (`gpt-live-1`) adapter's event translation — no network.

Validates that the Live protocol's events map to the runner's normalized event contract:
audio deltas -> raw PCM16 bytes, transcript deltas -> text, wrapped backend function_call ->
tool_call, wrapped response.completed -> response.done, and the parallel-tool-call batching that
avoids the Live endpoint's ``function_call_outputs_required`` rejection.
"""

import asyncio
import base64

from apex_voice.adapters.gpt_live1 import GptLive1Adapter


def _a():
    return GptLive1Adapter(api_key="test-key")


def test_output_audio_delta_decodes_to_pcm_bytes():
    a = _a()
    pcm = b"\x01\x02\x03\x04"
    out = a._normalize({"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
    assert out == [{"type": "response.audio.delta", "audio": pcm, "raw": out[0]["raw"]}]
    assert out[0]["audio"] == pcm


def test_output_transcript_delta_maps_to_transcript_text():
    a = _a()
    out = a._normalize({"type": "session.output_transcript.delta", "delta": "Hello"})
    assert out[0]["type"] == "response.audio_transcript.delta"
    assert out[0]["text"] == "Hello"


def test_wrapped_function_call_becomes_tool_call_and_counts_pending():
    a = _a()
    raw = {
        "type": "response.event",
        "event": {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "update_field",
                "arguments": '{"field":"x","value":"y"}',
                "call_id": "call_1",
            },
        },
    }
    out = a._normalize(raw)
    assert out[0]["type"] == "response.output_item.done"
    assert out[0]["tool_call"] == {
        "name": "update_field",
        "arguments": '{"field":"x","value":"y"}',
        "call_id": "call_1",
    }
    assert a._pending_tools == 1  # awaiting its output


def test_response_completed_maps_to_done():
    a = _a()
    out = a._normalize({"type": "response.event", "event": {"type": "response.completed"}})
    assert out == [{"type": "response.done", "raw": out[0]["raw"]}]


def test_non_function_output_item_is_ignored():
    a = _a()
    out = a._normalize(
        {
            "type": "response.event",
            "event": {"type": "response.output_item.done", "item": {"type": "message"}},
        }
    )
    assert out == []


def test_input_transcript_and_usage_are_dropped():
    a = _a()
    assert a._normalize({"type": "session.input_transcript.delta", "delta": " Hi"}) == []
    assert a._normalize({"type": "session.usage.updated", "usage": {"seconds": 1}}) == []


def test_as_tool_flattens_wrapped_and_bare_schemas():
    bare = {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}
    wrapped = {"type": "function", "function": bare}
    for t in (bare, wrapped):
        flat = GptLive1Adapter._as_tool(t)
        assert flat["type"] == "function" and flat["name"] == "f"
        assert "function" not in flat  # flattened for the Responses backend


def test_voice_clamped_to_openai_allowlist():
    a = GptLive1Adapter(api_key="k", voice="marin")
    assert a._resolve_voice("soft-spoken-gentleman") == "marin"  # unknown -> default
    assert a._resolve_voice("cedar") == "cedar"  # allowed passes through


def test_create_response_fires_every_turn_no_coalesce_deadlock():
    """Regression: each user turn must emit its own response.create. The earlier autonomous
    coalescing left a stuck 'active' flag that dropped every turn after the first (model silent)."""
    a = GptLive1Adapter(api_key="k")
    sent = []

    async def fake_send(msg):
        sent.append(msg["type"])

    a._send = fake_send

    async def go():
        for _ in range(4):  # four consecutive user turns
            await a.create_response()

    asyncio.run(go())
    assert sent == ["response.create"] * 4  # one per turn, none dropped


def test_tool_result_batches_then_creates_one_continuation():
    """Parallel tool outputs are all submitted before a single continuation create_response, so the
    Live endpoint never sees response.create with a function-call output still pending."""
    a = GptLive1Adapter(api_key="k")
    a._pending_tools = 2  # two function calls emitted in the current response
    sent = []

    async def fake_send(msg):
        sent.append(msg["type"])

    a._send = fake_send

    async def go():
        await a.send_tool_result({"call_id": "c1", "output": {"ok": True}})
        await a.send_tool_result({"call_id": "c2", "output": {"ok": True}})

    asyncio.run(go())
    # two outputs submitted, then exactly one continuation create
    assert sent == ["response.item.create", "response.item.create", "response.create"]
    assert a._pending_tools == 0
