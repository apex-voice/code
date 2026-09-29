"""Registry of the realtime speech-to-speech models evaluated in APEX-Voice.

Each entry maps a stable command-line key to its adapter class, the credentials it needs, and the
display name used in result tables. To benchmark a new model, implement the
:class:`~apex_voice.adapters.base.AgentAdapter` contract and add an entry here (see
``docs/adding_a_model.md``).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str  # command-line identifier, e.g. ``gpt-realtime``
    display_name: str  # name used in result tables
    provider: str
    adapter: str  # ``module:ClassName``
    required_env: tuple[str, ...]  # at least one of these must be set
    extra: str  # pip extra that installs the client dependency
    notes: str = ""

    def create(self, **kwargs: Any) -> Any:
        module, cls = self.adapter.split(":")
        return getattr(importlib.import_module(module), cls)(**kwargs)


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec(
            "gpt-realtime",
            "GPT-realtime-2.1",
            "OpenAI",
            "apex_voice.adapters.gpt_realtime:GptRealtimeAdapter",
            ("OPENAI_API_KEY",),
            "realtime",
            "Native speech-to-speech tool caller (OpenAI realtime protocol).",
        ),
        ModelSpec(
            "grok-voice",
            "Grok-Voice-Think-2.0",
            "xAI",
            "apex_voice.adapters.grok_voice:GrokVoiceAdapter",
            ("XAI_API_KEY",),
            "realtime",
            "Native speech-to-speech tool caller (OpenAI-realtime-compatible protocol).",
        ),
        ModelSpec(
            "gemini-live",
            "Gemini-3.8-Live",
            "Google",
            "apex_voice.adapters.gemini_live:GeminiLiveAdapter",
            ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            "gemini",
            "Gemini Live API via google-genai.",
        ),
        ModelSpec(
            "gpt-live-1",
            "GPT-live-1",
            "OpenAI",
            "apex_voice.adapters.gpt_live1:GptLive1Adapter",
            ("OPENAI_API_KEY",),
            "realtime",
            "Voice front-end with Responses delegation to a backend text model (gpt-4o).",
        ),
        ModelSpec(
            "step-audio3",
            "Step-Audio3",
            "StepFun",
            "apex_voice.adapters.realtime_step3:Step3RealtimeAdapter",
            ("STEPFUN_API_KEY",),
            "realtime",
            "StepFun realtime API (OpenAI-realtime-compatible protocol).",
        ),
    ]
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model '{key}'; choose from: {', '.join(MODELS)}") from None


def create_adapter(key: str, **kwargs: Any) -> Any:
    """Instantiate a fresh adapter for ``key`` (one adapter per session)."""
    return get_model(key).create(**kwargs)


__all__ = ["ModelSpec", "MODELS", "get_model", "create_adapter"]
