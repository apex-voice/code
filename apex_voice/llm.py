"""OpenAI-compatible chat client used by the semantic judge and the act observer.

Resolution order (first non-empty wins):

- API key:  ``APEX_VOICE_LLM_API_KEY``, then ``OPENAI_API_KEY``.
- Base URL: ``APEX_VOICE_LLM_BASE_URL``, then ``OPENAI_BASE_URL``, then the OpenAI default.

The dedicated ``APEX_VOICE_LLM_*`` variables let the judge/observer use a different account or an
OpenAI-compatible gateway from the one used by the GPT realtime models under test.
"""

from __future__ import annotations

from typing import Any

from apex_voice.credentials import env_first, require_env


def openai_chat_client(timeout: float = 45.0) -> Any:
    """Return an ``openai.OpenAI`` client configured from the environment."""
    from openai import OpenAI

    key = require_env("APEX_VOICE_LLM_API_KEY", "OPENAI_API_KEY", purpose="Judge / act observer")
    base = env_first("APEX_VOICE_LLM_BASE_URL", "OPENAI_BASE_URL")
    return OpenAI(api_key=key, base_url=base, timeout=timeout)


__all__ = ["openai_chat_client"]
