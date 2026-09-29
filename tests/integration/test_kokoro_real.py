"""Real Kokoro speech-compiler validation.

Skipped automatically where kokoro is not installed (keeps offline CI green); on a TTS-enabled env
(``pip install "apex-voice[tts]"``) it asserts the real compiler renders audible, well-timed audio.
"""

import numpy as np
import pytest

from apex_voice.harness.audio import peak_dbfs, rms_dbfs
from apex_voice.user_sim.kokoro_compiler import KokoroSpeechCompiler, kokoro_available

pytestmark = pytest.mark.skipif(not kokoro_available(), reason="kokoro not installed")


def test_kokoro_renders_audible_audio():
    c = KokoroSpeechCompiler(sample_rate=16000)
    cs = c.compile("I have been using Python for about four years.", {"voice_id": "voice_03"})
    assert cs.sample_rate == 16000
    assert cs.duration_ms > 800
    assert -35.0 < rms_dbfs(cs.waveform) < -5.0  # audible, not clipping
    assert peak_dbfs(cs.waveform) <= 0.0
    assert len(cs.word_timestamps) >= 6
    # timings are monotonic and within the clip
    ends = [w["end_ms"] for w in cs.word_timestamps]
    assert ends == sorted(ends)
    assert ends[-1] <= cs.duration_ms + 50


def test_kokoro_voice_selection_changes_audio():
    c = KokoroSpeechCompiler(sample_rate=16000)
    a = c.compile("hello there, how are you", {"voice_id": "voice_03"})
    b = c.compile("hello there, how are you", {"voice_id": "voice_07"})
    assert a.duration_ms > 0 and b.duration_ms > 0
    assert not np.array_equal(a.waveform, b.waveform)


def test_kokoro_deterministic():
    c = KokoroSpeechCompiler(sample_rate=16000)
    a = c.compile("about four years professionally", {"voice_id": "voice_03"})
    b = c.compile("about four years professionally", {"voice_id": "voice_03"})
    assert a.checksum() == b.checksum()
