import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_judge_cache(tmp_path, monkeypatch):
    """Never read or write the user's real judge cache from tests."""
    monkeypatch.setenv("APEX_VOICE_JUDGE_CACHE", str(tmp_path / "judge_cache.json"))


def dataset_root() -> Path | None:
    env = os.environ.get("APEX_VOICE_DATA")
    return Path(env) if env else None
