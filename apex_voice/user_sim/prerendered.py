"""Replay the dataset's pre-rendered simulated-user clips instead of synthesizing them live.

Every task package on the Hub ships ``user/audio/``: one lossless FLAC per realization-bank variant
(plus the runner's fixed sign-off and nudge lines) rendered with the pinned Kokoro-82M setup, and an
``index.jsonl`` mapping each utterance text to its clip. The FLACs hold exactly the PCM16 samples the
realtime runner streams to the model, so replaying them reproduces the published runs' user audio
bit-for-bit on any machine, without torch or Kokoro (live synthesis is only bit-exact under the same
torch build and thread configuration).

Clips are rendered by ``scripts/render_user_audio.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from apex_voice.user_sim.speech_compiler import CompiledSpeech

AUDIO_DIR = "user/audio"
INDEX_FILE = "index.jsonl"
SAMPLE_RATE = 24000  # the realtime runner's streaming rate
USER_AUDIO_MODES = ("auto", "prerendered", "kokoro")


def audio_index_path(task_dir: str | Path) -> Path:
    return Path(task_dir) / AUDIO_DIR / INDEX_FILE


def has_prerendered_audio(task_dir: str | Path) -> bool:
    return audio_index_path(task_dir).is_file()


def f32_to_pcm16(x: np.ndarray) -> np.ndarray:
    """The runner's float -> PCM16 conversion (clip, scale by 32767, truncate toward zero)."""
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def pcm16_to_f32(q: np.ndarray) -> np.ndarray:
    """Inverse of :func:`f32_to_pcm16`: decode to the middle of each truncation bucket, so that
    re-encoding with :func:`f32_to_pcm16` returns exactly ``q``."""
    q = np.asarray(q, dtype=np.int16).astype(np.float64)
    return ((q + 0.5 * np.sign(q)) / 32767.0).astype(np.float32)


class PrerenderedSpeechCompiler:
    """Speech compiler that looks utterances up in a task's pre-rendered clip index.

    Texts missing from the index are delegated to ``fallback`` (e.g. a live Kokoro compiler at
    :data:`SAMPLE_RATE`); without a fallback they raise ``KeyError``.
    """

    name = "prerendered"
    version = "1.0"

    def __init__(self, task_dir: str | Path, fallback: Any | None = None) -> None:
        self.index_path = audio_index_path(task_dir)
        self.audio_dir = self.index_path.parent
        self.sample_rate = SAMPLE_RATE
        self.fallback = fallback
        if fallback is not None and getattr(fallback, "sample_rate", SAMPLE_RATE) != SAMPLE_RATE:
            raise ValueError(f"fallback compiler must render at {SAMPLE_RATE} Hz")
        self._rows: dict[str, dict[str, Any]] = {}
        with self.index_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    self._rows.setdefault(row["text"], row)

    def __contains__(self, text: str) -> bool:
        return text in self._rows

    def compile(
        self,
        text: str,
        persona: dict[str, Any] | None = None,
        pronunciation_profile: dict[str, Any] | None = None,
        acoustic_profile: dict[str, Any] | None = None,
    ) -> CompiledSpeech:
        row = self._rows.get(text)
        if row is None:
            if self.fallback is None:
                raise KeyError(
                    f"no pre-rendered clip for {text[:60]!r} in {self.index_path}; "
                    "install Kokoro (pip install 'apex-voice[tts]') to synthesize it live"
                )
            return self.fallback.compile(text, persona, pronunciation_profile, acoustic_profile)

        import soundfile as sf

        q, sr = sf.read(str(self.audio_dir / row["file"]), dtype="int16")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row['file']}: expected {SAMPLE_RATE} Hz, got {sr}")
        wav = pcm16_to_f32(q)
        return CompiledSpeech(
            waveform=wav,
            duration_ms=int(round(1000 * len(wav) / SAMPLE_RATE)),
            sample_rate=SAMPLE_RATE,
            word_timestamps=list(row.get("word_timestamps", [])),
            metadata={
                "compiler": self.name,
                "version": self.version,
                "voice": row.get("voice"),
                "file": row["file"],
            },
        )


def compiler_for_task(task_dir: str | Path, mode: str = "auto") -> Any | None:
    """Pick the simulated-user speech source for one task.

    ``auto`` replays the shipped clips when the task has them (falling back to live Kokoro for any
    text not in the index), ``prerendered`` requires them, and ``kokoro`` always synthesizes live.
    Returns ``None`` when the runner should use its default live compiler.
    """
    if mode not in USER_AUDIO_MODES:
        raise ValueError(f"user audio mode must be one of {USER_AUDIO_MODES}, got {mode!r}")
    if mode == "kokoro":
        return None
    if not has_prerendered_audio(task_dir):
        if mode == "prerendered":
            raise FileNotFoundError(f"no pre-rendered user audio at {audio_index_path(task_dir)}")
        return None
    from apex_voice.user_sim.kokoro_compiler import KokoroSpeechCompiler, kokoro_available

    fallback = KokoroSpeechCompiler(sample_rate=SAMPLE_RATE) if kokoro_available() else None
    return PrerenderedSpeechCompiler(task_dir, fallback=fallback)


__all__ = [
    "AUDIO_DIR",
    "INDEX_FILE",
    "SAMPLE_RATE",
    "USER_AUDIO_MODES",
    "PrerenderedSpeechCompiler",
    "audio_index_path",
    "compiler_for_task",
    "f32_to_pcm16",
    "has_prerendered_audio",
    "pcm16_to_f32",
]
