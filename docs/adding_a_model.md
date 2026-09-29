# Adding a model

APEX-Voice evaluates any realtime speech-to-speech model that can (1) accept streamed user audio,
(2) speak back, and (3) call native function tools. Adding a model takes two steps.

## 1. Implement the adapter

Implement the async contract in
[`apex_voice/adapters/base.py`](../apex_voice/adapters/base.py) (`AgentAdapter`):

```python
class MyAdapter:
    model_id = "my-model"

    async def start_session(self, config): ...     # instructions, tools, voice, modalities
    async def send_audio(self, pcm_chunk, timestamp_ms=0): ...   # 24 kHz mono PCM16
    async def commit_audio(self): ...              # end of user utterance
    async def create_response(self): ...           # model should respond now
    async def cancel_response(self): ...           # user barge-in
    async def send_text(self, text, timestamp_ms=0): ...
    async def send_tool_result(self, result, timestamp_ms=0): ...  # {"call_id", "output"}
    def events(self): ...                          # async iterator of normalized events
    async def close(self): ...
```

`events()` must yield these normalized event dicts:

| `type` | extra keys |
| --- | --- |
| `response.audio.delta` | `audio`: raw 24 kHz PCM16 bytes |
| `response.audio_transcript.delta` | `text`: transcript of the model's speech |
| `response.output_item.done` | `tool_call`: `{"name", "arguments" (JSON string), "call_id"}` |
| `response.done` | – (end of the model's turn) |
| `error` | `error` |

The runner disables server-side voice-activity detection and drives turn-taking explicitly, so
every model hears the same user at the same time.

For providers that speak the OpenAI Realtime protocol, subclass
[`Step3RealtimeAdapter`](../apex_voice/adapters/realtime_step3.py) as `gpt_realtime.py` and
`grok_voice.py` do. You usually only need to override the endpoint, the key lookup
(`_default_api_key`), and small protocol differences. For other SDKs, see `gemini_live.py`
(Google GenAI Live) or `gpt_live1.py` (a voice front-end with delegated reasoning).

Read credentials from environment variables through `apex_voice.credentials.require_env`. Never
read them from files.

## 2. Register it

Add a `ModelSpec` to [`apex_voice/adapters/registry.py`](../apex_voice/adapters/registry.py):

```python
ModelSpec("my-model", "My-Model-1.0", "MyCo",
          "apex_voice.adapters.my_model:MyAdapter", ("MYCO_API_KEY",), "realtime"),
```

Then smoke-test it on one task and run the full campaign:

```bash
apex-voice run --model my-model --only apexv1_001 --repeats 1 --out runs/smoke
apex-voice run --model my-model --out runs/my-model
apex-voice results --runs-root runs --out results/
```

Please add a unit test that exercises your adapter's event normalization offline. See
`tests/unit/test_grok_voice_adapter.py` for an example.
