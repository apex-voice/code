"""Audio core: PCM representation, stereo assembly, and interval/overlap math.

Canonical internal representation: float32 mono PCM in [-1, 1] at a fixed sample rate
(16 kHz by default). Conversations are archived as stereo (left = user, right = agent) plus isolated
per-speaker channels. The interval utilities (merge/intersect/total) back the overlap accounting
and the duplex metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SR = 16000


# ---- basic PCM ops --------------------------------------------------------------------


def silence(duration_ms: int, sr: int = DEFAULT_SR) -> np.ndarray:
    return np.zeros(int(round(sr * duration_ms / 1000)), dtype=np.float32)


def duration_ms(audio: np.ndarray, sr: int = DEFAULT_SR) -> int:
    return int(round(1000 * len(audio) / sr))


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample (dependency-free). Adequate for harness routing; providers get native rates."""
    if src_sr == dst_sr or len(audio) == 0:
        return audio.astype(np.float32)
    n_dst = int(round(len(audio) * dst_sr / src_sr))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.float32)
    x_src = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_dst, x_src, audio).astype(np.float32)


def rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
    return 20.0 * np.log10(rms)


def peak_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -np.inf
    peak = float(np.max(np.abs(audio)))
    return 20.0 * np.log10(peak + 1e-12)


def fade_out(audio: np.ndarray, fade_ms: int, sr: int = DEFAULT_SR) -> np.ndarray:
    """Return a copy with a linear fade-out — used when an interrupted clip is cut."""
    a = audio.copy()
    fade = min(len(a), int(sr * fade_ms / 1000))
    if fade > 1:
        a[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return a


def cut_to_ms(audio: np.ndarray, cut_ms: int, sr: int = DEFAULT_SR, fade_ms: int = 20) -> np.ndarray:
    """Hard-cut an utterance at ``cut_ms`` with a fade-out (models a genuine barge-in stop)."""
    n = max(0, min(len(audio), int(round(cut_ms * sr / 1000))))
    return fade_out(audio[:n], fade_ms, sr)


# ---- stereo assembly ------------------------------------------------------------------


@dataclass
class PlacedClip:
    """A mono clip placed on the timeline at a media-time offset, on a named channel."""

    channel: str  # "user" | "agent"
    start_ms: int
    audio: np.ndarray


_CHANNEL_INDEX = {"user": 0, "agent": 1}  # left = user, right = agent


def assemble_stereo(clips: list[PlacedClip], sr: int = DEFAULT_SR, tail_ms: int = 200) -> np.ndarray:
    """Mix placed clips into an (n, 2) float32 buffer. Channels never bleed (clean C2)."""
    if not clips:
        return np.zeros((int(sr * tail_ms / 1000), 2), dtype=np.float32)
    end_ms = max(c.start_ms + duration_ms(c.audio, sr) for c in clips) + tail_ms
    n = int(round(sr * end_ms / 1000))
    buf = np.zeros((n, 2), dtype=np.float32)
    for c in clips:
        ch = _CHANNEL_INDEX[c.channel]
        start = int(round(c.start_ms * sr / 1000))
        end = min(n, start + len(c.audio))
        if end > start:
            buf[start:end, ch] += c.audio[: end - start]
    np.clip(buf, -1.0, 1.0, out=buf)
    return buf


def isolated_channel(
    clips: list[PlacedClip], channel: str, sr: int = DEFAULT_SR, tail_ms: int = 200
) -> np.ndarray:
    stereo = assemble_stereo(clips, sr, tail_ms)
    return stereo[:, _CHANNEL_INDEX[channel]].copy()


def write_wav(path: str | Path, audio: np.ndarray, sr: int = DEFAULT_SR) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


# ---- interval / overlap math --------------------------

Interval = tuple[float, float]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def total_span(intervals: list[Interval]) -> float:
    return sum(e - s for s, e in intervals)


def intersect_total(a: list[Interval], b: list[Interval]) -> float:
    a, b = merge_intervals(a), merge_intervals(b)
    total, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


@dataclass
class TimelineStats:
    duration_ms: int
    user_speech_ms: float
    agent_speech_ms: float
    overlap_ms: float
    overlap_ratio: float
    silence_ratio: float


def timeline_stats(clips: list[PlacedClip], sr: int = DEFAULT_SR, tail_ms: int = 200) -> TimelineStats:
    if not clips:
        return TimelineStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dur = max(c.start_ms + duration_ms(c.audio, sr) for c in clips) + tail_ms
    u = [(c.start_ms, c.start_ms + duration_ms(c.audio, sr)) for c in clips if c.channel == "user"]
    a = [(c.start_ms, c.start_ms + duration_ms(c.audio, sr)) for c in clips if c.channel == "agent"]
    overlap = intersect_total(u, a)
    either = total_span(merge_intervals(u + a))
    return TimelineStats(
        duration_ms=dur,
        user_speech_ms=total_span(merge_intervals(u)),
        agent_speech_ms=total_span(merge_intervals(a)),
        overlap_ms=overlap,
        overlap_ratio=round(overlap / dur, 4) if dur else 0.0,
        silence_ratio=round(max(0.0, dur - either) / dur, 4) if dur else 0.0,
    )


__all__ = [
    "DEFAULT_SR",
    "silence",
    "duration_ms",
    "resample",
    "rms_dbfs",
    "peak_dbfs",
    "fade_out",
    "cut_to_ms",
    "PlacedClip",
    "assemble_stereo",
    "isolated_channel",
    "write_wav",
    "merge_intervals",
    "total_span",
    "intersect_total",
    "TimelineStats",
    "timeline_stats",
]
