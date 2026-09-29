import json

import pytest

import apex_voice.campaign as campaign
from apex_voice.adapters import registry
from apex_voice.campaign import CampaignLock, CampaignSettings, is_transient, run_campaign
from tests.fake_adapter import SilentAdapter
from tests.paths import TOY_FORM


@pytest.fixture
def silent_model(monkeypatch):
    spec = registry.ModelSpec(
        "silent", "Silent", "test", "tests.fake_adapter:SilentAdapter", ("PATH",), "none"
    )
    monkeypatch.setitem(registry.MODELS, "silent", spec)
    monkeypatch.setattr("apex_voice.user_sim.kokoro_compiler.kokoro_available", lambda: False)
    monkeypatch.setattr(campaign.asyncio, "sleep", _no_sleep_for_retries)
    SilentAdapter._failures = 0
    SilentAdapter.fail_times = 0
    return spec


_real_sleep = campaign.asyncio.sleep


async def _no_sleep_for_retries(s, *a, **k):
    await _real_sleep(min(s, 0.01))


def _judge(*_a, **_k):
    return False


async def test_campaign_writes_resumable_results(tmp_path, silent_model):
    s = CampaignSettings(model="silent", out_dir=tmp_path / "run", repeats=2, max_turns=1)
    rows = await run_campaign([TOY_FORM], s, judge=_judge, log=lambda _m: None)
    assert [(r["task_id"], r["rep"], r["pts"]) for r in rows] == [
        ("toy_form_001", 0, 0),
        ("toy_form_001", 1, 0),
    ]
    result = json.loads((s.out_dir / "toy_form_001" / "rep0" / "result.json").read_text())
    assert result["model"] == "silent" and result["condition"] == "C2" and result["error"] is None
    assert set(result["components"]) >= {"GS", "PC", "RA", "WA"}
    assert (s.out_dir / "toy_form_001" / "rep0" / result["run_id"] / "events.jsonl").exists()
    manifest = json.loads((s.out_dir / "campaign.json").read_text())
    assert manifest["model"] == "silent" and manifest["repeats"] == 2
    assert not (s.out_dir / ".campaign.lock").exists()

    # Re-running is a no-op that returns the persisted rows.
    SilentAdapter.fail_times = 99
    again = await run_campaign([TOY_FORM], s, judge=_judge, log=lambda _m: None)
    assert [r["run_id"] for r in again] == [r["run_id"] for r in rows]


async def test_transient_failures_are_retried_then_left_empty(tmp_path, silent_model):
    SilentAdapter.fail_times = 1
    s = CampaignSettings(model="silent", out_dir=tmp_path / "a", repeats=1, max_turns=1)
    rows = await run_campaign([TOY_FORM], s, judge=_judge, log=lambda _m: None)
    assert len(rows) == 1 and rows[0]["error"] is None

    SilentAdapter._failures = 0
    SilentAdapter.fail_times = 99
    s = CampaignSettings(model="silent", out_dir=tmp_path / "b", repeats=1, max_turns=1)
    assert await run_campaign([TOY_FORM], s, judge=_judge, log=lambda _m: None) == []
    assert not (s.out_dir / "toy_form_001" / "rep0" / "result.json").exists()


def test_is_transient():
    assert is_transient("ConnectionClosedError: no close frame received")
    assert is_transient("HTTP 503 Service Unavailable")
    assert not is_transient("ValueError: bad tool schema")


def test_lock_is_exclusive_and_reclaims_stale(tmp_path):
    with CampaignLock(tmp_path):
        with pytest.raises(RuntimeError, match="another campaign"):
            CampaignLock(tmp_path).__enter__()
    (tmp_path / ".campaign.lock").write_text("999999999")  # dead pid
    with CampaignLock(tmp_path):
        pass
