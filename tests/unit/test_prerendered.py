import json

import numpy as np
import pytest
import soundfile as sf

from apex_voice.harness.realtime_runner import _f32_to_pcm16
from apex_voice.user_sim.prerendered import (
    PrerenderedSpeechCompiler,
    compiler_for_task,
    f32_to_pcm16,
    has_prerendered_audio,
    pcm16_to_f32,
)


def test_pcm16_decode_reencodes_exactly():
    q = np.concatenate([np.arange(-32767, 32768, dtype=np.int16), [0, 1, -1, 32767, -32767]]).astype(np.int16)
    assert np.array_equal(f32_to_pcm16(pcm16_to_f32(q)), q)


def test_conversion_matches_runner():
    x = np.random.default_rng(0).uniform(-1.2, 1.2, 50_000).astype(np.float32)
    assert f32_to_pcm16(x).tobytes() == _f32_to_pcm16(x)


@pytest.fixture
def task_dir(tmp_path):
    audio = tmp_path / "user" / "audio"
    audio.mkdir(parents=True)
    q = f32_to_pcm16(np.sin(np.linspace(0, 200, 2400)).astype(np.float32) * 0.5)
    sf.write(str(audio / "open_v1.flac"), q, 24000, format="FLAC", subtype="PCM_16")
    row = {"file": "open_v1.flac", "text": "Hello there.", "voice": "af_heart", "word_timestamps": []}
    (audio / "index.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path, q


def test_replay_streams_the_stored_pcm16(task_dir):
    root, q = task_dir
    cs = PrerenderedSpeechCompiler(root).compile("Hello there.", {"voice_id": "voice_01"})
    assert cs.sample_rate == 24000 and cs.duration_ms == 100
    assert _f32_to_pcm16(cs.waveform) == q.tobytes()


def test_missing_text_needs_a_fallback(task_dir):
    root, _ = task_dir
    with pytest.raises(KeyError):
        PrerenderedSpeechCompiler(root).compile("Not in the index.")

    class Fallback:
        sample_rate = 24000

        def compile(self, text, *args):
            return text

    assert (
        PrerenderedSpeechCompiler(root, fallback=Fallback()).compile("Not in the index.")
        == "Not in the index."
    )


def test_compiler_selection(task_dir, tmp_path_factory):
    root, _ = task_dir
    bare = tmp_path_factory.mktemp("bare")
    assert has_prerendered_audio(root) and not has_prerendered_audio(bare)
    assert isinstance(compiler_for_task(root, "auto"), PrerenderedSpeechCompiler)
    assert compiler_for_task(root, "kokoro") is None
    assert compiler_for_task(bare, "auto") is None
    with pytest.raises(FileNotFoundError):
        compiler_for_task(bare, "prerendered")
    with pytest.raises(ValueError):
        compiler_for_task(root, "nope")
