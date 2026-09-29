"""Per-turn speech synthesis with Kokoro-82M, including silence trimming and word timings.

Synthesis is content-seeded, so the same text and voice always yield byte-identical audio.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Word:
    """A synthesized word with its start/end time (seconds) inside the clip."""

    text: str
    start_s: float
    end_s: float


_WORDISH = re.compile(r"[A-Za-z0-9]")


@dataclass
class TurnAudio:
    audio: np.ndarray  # float32, mono, at `sample_rate`
    sample_rate: int
    words: list[Word]  # timings relative to the *trimmed* clip

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _trim_bounds(
    audio: np.ndarray, sr: int, threshold_db: float, margin_ms: int, frame_ms: int = 10
) -> tuple[int, int]:
    """Return (start, end) sample indices of the voiced region, padded by `margin_ms`.

    Kokoro emits ~0.4s of leading and ~0.9s of trailing silence per call. Left in place,
    that silence would be added to every turn boundary and the scheduled timings would
    describe something the audio does not do.
    """
    hop = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, len(audio) // hop)
    frames = audio[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) + 1e-12
    db = 20.0 * np.log10(rms)
    voiced = np.flatnonzero(db > threshold_db)
    if voiced.size == 0:
        return 0, len(audio)  # all-silent clip: leave it for QC to flag
    margin = int(sr * margin_ms / 1000)
    start = max(0, voiced[0] * hop - margin)
    end = min(len(audio), (voiced[-1] + 1) * hop + margin)
    return start, end


class KokoroSynth:
    """Thin wrapper over `kokoro.KPipeline`, one instance reused for all turns and voices."""

    def __init__(self, cfg: dict, base_seed: int = 0):
        from kokoro import KModel, KPipeline

        self.cfg = cfg
        self.sample_rate = cfg["sample_rate"]
        self.device = _resolve_device(cfg["device"])
        self.speed = cfg["speed"]
        self.repo_id = cfg["repo_id"]
        self.revision = cfg.get("revision")
        if self.revision:
            model = KModel(
                repo_id=self.repo_id,
                config=self._download("config.json"),
                model=self._download(KModel.MODEL_NAMES[self.repo_id]),
            )
            self._pipe = KPipeline(
                lang_code=cfg["lang_code"], repo_id=self.repo_id, model=model.to(self.device).eval()
            )
        else:
            self._pipe = KPipeline(lang_code=cfg["lang_code"], repo_id=self.repo_id, device=self.device)
        # Safety belt: the model already ships in eval mode, but assert it rather than
        # assume it, since active dropout would silently randomise every turn.
        if getattr(self._pipe, "model", None) is not None:
            self._pipe.model.eval()
        self._voices = dict(cfg["voices"])
        self.base_seed = int(base_seed)

    def _download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=self.repo_id, filename=filename, revision=self.revision)

    def _voice_source(self, voice: str) -> str:
        """What to hand KPipeline: a pinned local .pt path, or the bare name (fetched from main)."""
        if not self.revision or voice.endswith(".pt"):
            return voice
        return self._download(f"voices/{voice}.pt")

    def voice_for(self, speaker: str) -> str:
        return self._voices[speaker]

    def _seed_for(self, text: str, voice: str) -> int:
        """Content-derived seed: the same (text, voice, speed) always renders identically.

        Kokoro draws from the *global* torch RNG during inference, so without this the same
        sentence synthesizes differently on every call (measured max sample delta 0.15 on a
        0.47 peak -- audibly different prosody, not float noise). Deriving the seed from the
        content rather than a counter also makes a turn reproducible independently of how
        many turns were rendered before it.
        """
        key = f"{self.base_seed}|{voice}|{self.speed}|{text}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**31)

    @torch.inference_mode()
    def synth(self, text: str, speaker: str) -> TurnAudio:
        voice = self.voice_for(speaker)
        seed = self._seed_for(text, voice)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        chunks: list[np.ndarray] = []
        words: list[Word] = []
        offset = 0.0
        # The seed keys on the voice *name*; only the pipeline sees the resolved file path.
        for res in self._pipe(text, voice=self._voice_source(voice), speed=self.speed):
            if res.audio is None:
                continue
            a = res.audio.detach().cpu().numpy().astype(np.float32)
            # Kokoro's predicted token durations give us word timings for free.
            for tok in res.tokens or []:
                if tok.start_ts is None or tok.end_ts is None:
                    continue
                if not _WORDISH.search(tok.text or ""):
                    continue  # skip bare punctuation tokens
                words.append(Word(tok.text, offset + float(tok.start_ts), offset + float(tok.end_ts)))
            chunks.append(a)
            offset += len(a) / self.sample_rate

        if not chunks:
            raise RuntimeError(f"Kokoro returned no audio for speaker {speaker}: {text[:60]!r}")
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        if self.cfg["trim_silence"]:
            s, e = _trim_bounds(
                audio, self.sample_rate, self.cfg["trim_threshold_db"], self.cfg["trim_margin_ms"]
            )
            audio = audio[s:e]
            shift = s / self.sample_rate
            dur = len(audio) / self.sample_rate
            rebased = []
            for w in words:
                ws, we = w.start_s - shift, w.end_s - shift
                if we <= 0 or ws >= dur:
                    continue  # word fell entirely inside trimmed silence
                rebased.append(Word(w.text, max(0.0, ws), min(dur, we)))
            words = rebased

        return TurnAudio(audio=audio, sample_rate=self.sample_rate, words=words)
