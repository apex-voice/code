"""Offline speech-compiler interface + dummy compiler.

The canonical V1 path pre-renders every validated text variant to an utterance-level audio asset.
The real compiler is :class:`apex_voice.user_sim.kokoro_compiler.KokoroSpeechCompiler` (Kokoro TTS,
offline, deterministic). For offline construction and CI we ship :class:`DummySpeechCompiler`, which
produces a deterministic silent/tone waveform so the asset-bank plumbing is exercised without any
TTS dependency. Both satisfy the same interface:

    compile(text, persona, pronunciation_profile, acoustic_profile)
        -> waveform, duration_ms, sample_rate, optional_word_timestamps, metadata
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from apex_voice.determinism import stable_int


@dataclass
class CompiledSpeech:
    waveform: np.ndarray  # float32 mono PCM in [-1, 1]
    duration_ms: int
    sample_rate: int
    word_timestamps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def checksum(self) -> str:
        return hashlib.blake2b(self.waveform.tobytes(), digest_size=16).hexdigest()


class SpeechCompiler(Protocol):
    name: str
    version: str

    def compile(
        self,
        text: str,
        persona: dict[str, Any] | None = None,
        pronunciation_profile: dict[str, Any] | None = None,
        acoustic_profile: dict[str, Any] | None = None,
    ) -> CompiledSpeech: ...


class DummySpeechCompiler:
    """Deterministic offline stand-in: word count -> duration; a faint seeded tone as waveform.

    Same text + persona always yields byte-identical audio (mirrors the Kokoro compiler's
    determinism contract), so the audio bank checksums are stable in CI.
    """

    name = "dummy_speech_compiler"
    version = "1.0"

    def __init__(self, sample_rate: int = 16000, ms_per_word: int = 320) -> None:
        self.sample_rate = sample_rate
        self.ms_per_word = ms_per_word

    def compile(
        self,
        text: str,
        persona: dict[str, Any] | None = None,
        pronunciation_profile: dict[str, Any] | None = None,
        acoustic_profile: dict[str, Any] | None = None,
    ) -> CompiledSpeech:
        words = text.split()
        n_words = max(1, len(words))
        duration_ms = n_words * self.ms_per_word
        n_samples = int(self.sample_rate * duration_ms / 1000)
        # Seeded faint tone so waveform is non-trivial but deterministic.
        seed = stable_int(text, (persona or {}).get("voice_id", "default"))
        freq = 110 + (seed % 220)
        t = np.arange(n_samples, dtype=np.float32) / self.sample_rate
        wave = (0.01 * np.sin(2 * math.pi * freq * t)).astype(np.float32)
        # Per-word timestamps evenly spaced (dummy alignment).
        wts = [
            {"word": w, "start_ms": i * self.ms_per_word, "end_ms": (i + 1) * self.ms_per_word}
            for i, w in enumerate(words)
        ]
        return CompiledSpeech(
            waveform=wave,
            duration_ms=duration_ms,
            sample_rate=self.sample_rate,
            word_timestamps=wts,
            metadata={
                "compiler": self.name,
                "version": self.version,
                "voice_id": (persona or {}).get("voice_id", "default"),
            },
        )


__all__ = ["CompiledSpeech", "SpeechCompiler", "DummySpeechCompiler"]
