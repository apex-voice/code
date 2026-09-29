import pytest

from apex_voice.adapters.registry import MODELS, create_adapter, get_model
from apex_voice.credentials import MissingCredentialError, env_first, require_env


def test_registry_has_the_five_benchmarked_models():
    assert set(MODELS) == {"gpt-realtime", "grok-voice", "gemini-live", "gpt-live-1", "step-audio3"}
    for spec in MODELS.values():
        assert spec.required_env and ":" in spec.adapter


def test_unknown_model_lists_choices():
    with pytest.raises(KeyError, match="gpt-realtime"):
        get_model("nope")


@pytest.mark.parametrize("key", ["gpt-realtime", "grok-voice", "gpt-live-1", "step-audio3"])
def test_adapters_construct_without_network_or_keys(key, monkeypatch):
    for spec in MODELS.values():
        for name in spec.required_env:
            monkeypatch.delenv(name, raising=False)
    adapter = create_adapter(key)
    assert hasattr(adapter, "start_session") and hasattr(adapter, "events")


def test_env_first_and_require_env(monkeypatch):
    monkeypatch.delenv("APEX_T_A", raising=False)
    monkeypatch.setenv("APEX_T_B", " value ")
    assert env_first("APEX_T_A", "APEX_T_B") == "value"
    assert env_first("APEX_T_A", default="d") == "d"
    with pytest.raises(MissingCredentialError, match="APEX_T_A"):
        require_env("APEX_T_A", purpose="test")


def test_llm_client_requires_key(monkeypatch):
    from apex_voice.llm import openai_chat_client

    monkeypatch.delenv("APEX_VOICE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError):
        openai_chat_client()


def test_realtime_adapter_reads_key_from_env(monkeypatch):
    from apex_voice.adapters.realtime_step3 import Step3RealtimeAdapter

    monkeypatch.setenv("STEPFUN_API_KEY", "sk-test")
    assert Step3RealtimeAdapter()._default_api_key() == "sk-test"
    monkeypatch.delenv("STEPFUN_API_KEY")
    with pytest.raises(MissingCredentialError):
        Step3RealtimeAdapter()._default_api_key()
