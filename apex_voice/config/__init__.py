"""Typed configuration, seed handling, and version stamps.

Everything the harness needs to reproduce a non-model computation is captured in a
:class:`RunConfig` (a single scored execution) or a :class:`CampaignConfig` (a model ×
task-set × condition × seed matrix). Configs are pydantic models so they serialize to a
stable JSON snapshot stored in the run manifest.
"""

from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from apex_voice import BENCHMARK_VERSION, __version__


class Condition(str, Enum):
    """Matched evaluation conditions."""

    C0 = "C0"  # text control
    C1 = "C1"  # turn-based voice
    C2 = "C2"  # full-duplex clean
    C3 = "C3"  # full-duplex realistic


class RealizationTrack(str, Enum):
    """User-realization tracks."""

    FROZEN_SYNTHETIC = "FROZEN_SYNTHETIC"
    LIVE_GENERATIVE = "LIVE_GENERATIVE"
    HUMAN_AUDIT = "HUMAN_AUDIT"


class VersionStamp(BaseModel):
    """Immutable provenance stamp embedded in every run manifest."""

    model_config = ConfigDict(frozen=True)

    apex_voice_version: str = __version__
    benchmark_version: str = BENCHMARK_VERSION
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class RunConfig(BaseModel):
    """Configuration for a single scored run of one task under one condition."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_version: str
    model_id: str = "oracle"
    condition: Condition = Condition.C0
    realization_track: RealizationTrack = RealizationTrack.FROZEN_SYNTHETIC
    # Seeds. ``simulator_seed`` drives user-flow branches + asset selection; ``channel_seed``
    # drives C3 acoustic perturbations; ``event_seed`` drives duplex-timing jitter.
    simulator_seed: int = 1
    channel_seed: int = 1
    event_seed: int = 1
    # Wall-clock budget for the run; media-time budget comes from the task's ``time_budget_s``.
    max_wall_seconds: float = 600.0
    # Free-form provider/adapter options resolved at integration time.
    adapter_options: dict[str, Any] = Field(default_factory=dict)
    version: VersionStamp = Field(default_factory=VersionStamp)

    @property
    def scenario_id(self) -> str:
        """Stable identity of the (task, condition, track) scenario for asset selection."""
        return f"{self.task_id}@{self.task_version}:{self.condition.value}:{self.realization_track.value}"


class CampaignConfig(BaseModel):
    """A model campaign over a task set."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    model_ids: list[str]
    task_set: str
    conditions: list[Condition] = Field(default_factory=lambda: [Condition.C3])
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    realization_track: RealizationTrack = RealizationTrack.FROZEN_SYNTHETIC
    version: VersionStamp = Field(default_factory=VersionStamp)


__all__ = [
    "Condition",
    "RealizationTrack",
    "VersionStamp",
    "RunConfig",
    "CampaignConfig",
]
