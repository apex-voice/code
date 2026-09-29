"""Dual-clock harness.

Two clocks keep simulation time and real latency separable:

- the **media clock** timestamps audio events, floor transfers, scripted scenario events, and
  tool-result availability in media time (ms). It advances deterministically so a slow model
  cannot shift authored media-time events (CI: "slow model cannot shift authored media events").
- the **wall clock** measures actual model/retrieval/tool/network latency in monotonic ns.

In C0 text mode the media clock advances by authored per-turn increments; in voice modes it will
advance with streamed audio.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class DualClock:
    media_time_ms: int = 0
    _wall_origin_ns: int = 0

    def __post_init__(self) -> None:
        self._wall_origin_ns = time.monotonic_ns()

    # ---- media clock (deterministic) ------------------------------------------------
    def advance_media(self, ms: int) -> int:
        self.media_time_ms += max(0, int(ms))
        return self.media_time_ms

    def set_media(self, ms: int) -> None:
        self.media_time_ms = max(self.media_time_ms, int(ms))

    # ---- wall clock (real latency) --------------------------------------------------
    def wall_ns(self) -> int:
        return time.monotonic_ns()

    def wall_elapsed_ns(self) -> int:
        return time.monotonic_ns() - self._wall_origin_ns

    def wall_elapsed_s(self) -> float:
        return self.wall_elapsed_ns() / 1e9


__all__ = ["DualClock"]
