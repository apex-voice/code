"""Offline Kokoro speech compiler for the simulated user.

Renders each validated user text variant into an utterance-level clip with word timings using
:class:`~apex_voice.user_sim.kokoro_tts.KokoroSynth` (Kokoro-82M, fully offline, content-seeded, so
the same text and voice yield byte-identical audio). It implements the same
:class:`~apex_voice.user_sim.speech_compiler.SpeechCompiler` interface as the dummy compiler, so the
runners are agnostic to which is used. Kokoro is imported lazily; if it is not installed
:func:`get_default_compiler` falls back to :class:`DummySpeechCompiler`.

Persona -> voice: ``persona["voice_id"]`` is mapped to a Kokoro voice name via ``voice_map``
(falling back to the default voice).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apex_voice.harness.audio import DEFAULT_SR, resample
from apex_voice.user_sim.speech_compiler import CompiledSpeech

KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
# Hugging Face snapshot of Kokoro-82M used for every published APEX-Voice run. kokoro==0.9.4
# always fetches from `main`, so KokoroSynth downloads these pinned files and passes local paths.
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"

# APEX persona voice_id -> Kokoro voice name. Extend as personas are added.
DEFAULT_VOICE_MAP: dict[str, str] = {
    "voice_01": "af_heart",
    "voice_03": "af_bella",
    "voice_07": "am_michael",
    "default": "af_heart",
}


def kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401

        return True
    except Exception:
        return False


class KokoroSpeechCompiler:
    """Utterance-level Kokoro-82M speech compiler."""

    name = "kokoro"
    version = "1.0"

    def __init__(
        self,
        sample_rate: int = DEFAULT_SR,
        kokoro_sample_rate: int = 24000,
        speed: float = 1.0,
        # CPU by default: the published runs synthesized on CPU, and CUDA kernels change the samples.
        device: str = "cpu",
        base_seed: int = 0,
        voice_map: dict[str, str] | None = None,
        lang_code: str = "a",
        repo_id: str = KOKORO_REPO_ID,
        revision: str | None = KOKORO_REVISION,
    ) -> None:
        self.sample_rate = sample_rate
        self.kokoro_sr = kokoro_sample_rate
        self.speed = speed
        self.device = device
        self.base_seed = base_seed
        self.voice_map = dict(voice_map or DEFAULT_VOICE_MAP)
        self.lang_code = lang_code
        self.repo_id = repo_id
        self.revision = revision
        self._synth = None  # lazily constructed

    def _voice_for(self, persona: dict[str, Any] | None) -> str:
        voice_id = (persona or {}).get("voice_id", "default")
        return self.voice_map.get(voice_id, self.voice_map.get("default", "af_heart"))

    def _ensure_synth(self, voice: str) -> Any:
        if self._synth is not None:
            return self._synth
        from apex_voice.user_sim.kokoro_tts import KokoroSynth

        cfg = {
            "engine": "kokoro",
            "repo_id": self.repo_id,
            "revision": self.revision,
            "lang_code": self.lang_code,
            "sample_rate": self.kokoro_sr,
            "device": self.device,
            "speed": self.speed,
            # KokoroSynth requires a voices map keyed by speaker; we always synth as speaker "U".
            "voices": {"U": voice},
            "trim_silence": True,
            "trim_threshold_db": -45.0,
            "trim_margin_ms": 30,
        }
        self._synth = KokoroSynth(cfg, base_seed=self.base_seed)
        return self._synth

    def compile(
        self,
        text: str,
        persona: dict[str, Any] | None = None,
        pronunciation_profile: dict[str, Any] | None = None,
        acoustic_profile: dict[str, Any] | None = None,
    ) -> CompiledSpeech:
        voice = self._voice_for(persona)
        synth = self._ensure_synth(voice)
        # Rebind the voice for this utterance (single reused pipeline, per-call voice).
        synth._voices["U"] = voice  # noqa: SLF001
        ta = synth.synth(text, "U")
        audio = ta.audio.astype(np.float32)
        if ta.sample_rate != self.sample_rate:
            audio = resample(audio, ta.sample_rate, self.sample_rate)
        dur_ms = int(round(1000 * len(audio) / self.sample_rate))
        wts = [
            {"word": w.text, "start_ms": int(round(w.start_s * 1000)), "end_ms": int(round(w.end_s * 1000))}
            for w in ta.words
        ]
        cs = CompiledSpeech(
            waveform=audio,
            duration_ms=dur_ms,
            sample_rate=self.sample_rate,
            word_timestamps=wts,
            metadata={
                "compiler": self.name,
                "version": self.version,
                "voice": voice,
                "kokoro_revision": self.revision,
                "kokoro_sr": self.kokoro_sr,
                "speed": self.speed,
            },
        )
        return cs


def get_default_compiler(prefer_real: bool = True, **kwargs: Any):
    """Return the real Kokoro compiler if available and preferred, else the deterministic dummy."""
    from apex_voice.user_sim.speech_compiler import DummySpeechCompiler

    if prefer_real and kokoro_available():
        return KokoroSpeechCompiler(**kwargs)
    return DummySpeechCompiler(sample_rate=kwargs.get("sample_rate", DEFAULT_SR))


__all__ = [
    "DEFAULT_VOICE_MAP",
    "KOKORO_REPO_ID",
    "KOKORO_REVISION",
    "KokoroSpeechCompiler",
    "get_default_compiler",
    "kokoro_available",
]
