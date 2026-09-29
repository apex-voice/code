"""Validates the released dataset. Runs only when APEX_VOICE_DATA points at a local copy."""

import json
import os
from pathlib import Path

import pytest

from apex_voice.data import list_task_dirs
from apex_voice.validation import validate_task

DATA = os.environ.get("APEX_VOICE_DATA")
pytestmark = [
    pytest.mark.dataset,
    pytest.mark.skipif(not DATA, reason="set APEX_VOICE_DATA to the dataset root"),
]


def test_dataset_index_matches_task_packages():
    root = Path(DATA)
    rows = [json.loads(line) for line in (root / "data" / "tasks.jsonl").read_text().splitlines()]
    dirs = list_task_dirs(root)
    assert len(rows) == len(dirs) == 120
    assert [r["task_id"] for r in rows] == [d.name for d in dirs]


def test_every_task_validates():
    bad = [r.task_id for r in map(validate_task, list_task_dirs(Path(DATA))) if not r.ok]
    assert not bad


def test_user_audio_covers_every_spoken_text():
    import hashlib

    import soundfile as sf

    from apex_voice.harness.realtime_runner import _CLOSINGS, _NUDGE
    from apex_voice.tasks import load_task
    from apex_voice.user_sim.prerendered import SAMPLE_RATE, audio_index_path

    problems = []
    for d in list_task_dirs(Path(DATA)):
        idx = audio_index_path(d)
        rows = [json.loads(line) for line in idx.read_text().splitlines()]
        texts = {r["text"] for r in rows}
        bank = load_task(d).realization_bank
        spoken = {v.text for pid in bank.plan_ids() for v in bank.get(pid).variants} | {*_CLOSINGS, _NUDGE}
        if spoken - texts:
            problems.append(f"{d.name}: {len(spoken - texts)} texts without a clip")
        for r in rows:
            q, sr = sf.read(str(idx.parent / r["file"]), dtype="int16")
            if (
                sr != SAMPLE_RATE
                or len(q) != r["num_samples"]
                or hashlib.sha256(q.tobytes()).hexdigest() != r["pcm16_sha256"]
            ):
                problems.append(f"{d.name}/{r['file']}: does not match its index row")
    assert not problems, problems[:10]
