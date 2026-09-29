import numpy as np

from apex_voice.user_sim.kokoro_compiler import KokoroSpeechCompiler, get_default_compiler, kokoro_available
from apex_voice.user_sim.speech_compiler import DummySpeechCompiler


def test_dummy_compiler_deterministic():
    c = DummySpeechCompiler()
    a = c.compile("about four years", {"voice_id": "voice_03"})
    b = c.compile("about four years", {"voice_id": "voice_03"})
    assert a.checksum() == b.checksum()
    assert a.duration_ms > 0 and a.sample_rate == 16000
    assert len(a.word_timestamps) == 3


def test_dummy_compiler_voice_affects_audio():
    c = DummySpeechCompiler()
    a = c.compile("hello there", {"voice_id": "voice_03"})
    b = c.compile("hello there", {"voice_id": "voice_07"})
    assert not np.array_equal(a.waveform, b.waveform)


def test_get_default_compiler_falls_back_to_dummy_when_no_kokoro():
    c = get_default_compiler(prefer_real=True, sample_rate=16000)
    if kokoro_available():
        assert isinstance(c, KokoroSpeechCompiler)
    else:
        assert isinstance(c, DummySpeechCompiler)


def test_kokoro_compiler_constructs_without_kokoro():
    # Construction must not import kokoro (lazy); only .compile() would.
    c = KokoroSpeechCompiler(sample_rate=16000)
    assert c.name == "kokoro"
    assert c._voice_for({"voice_id": "voice_07"}) == "am_michael"


def test_kokoro_compiler_pins_paper_weights_and_cpu():
    # The published runs used this Kokoro-82M snapshot on CPU; either changing alters the user audio.
    from apex_voice.user_sim.kokoro_compiler import KOKORO_REVISION

    c = KokoroSpeechCompiler()
    assert c.revision == KOKORO_REVISION == "f3ff3571791e39611d31c381e3a41a3af07b4987"
    assert c.device == "cpu"
