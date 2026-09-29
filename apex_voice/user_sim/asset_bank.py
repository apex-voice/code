"""Frozen realization / audio asset bank and deterministic selector.

The realization bank stores, per user_plan_id, 2-5 validated text variants with stable ids and
compile-time provenance. The audio bank maps each variant to a pre-rendered utterance clip. At
runtime the selector maps ``(scenario_id, simulator_seed, user_plan_id)`` deterministically to one
variant: the same semantic state + seed always yields the same surface realization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex_voice.determinism import stable_choice


@dataclass
class TextVariant:
    variant_id: str
    text: str
    expressed_fact_ids: list[str] = field(default_factory=list)
    compiler: dict[str, Any] = field(default_factory=dict)  # model/version/prompt-hash/seed
    checksum: str | None = None


@dataclass
class PlanVariants:
    user_plan_id: str
    speech_act: str
    variants: list[TextVariant] = field(default_factory=list)


class RealizationBank:
    """In-memory + JSONL-backed store of validated text variants (``realization_bank.jsonl``)."""

    def __init__(self) -> None:
        self._by_plan: dict[str, PlanVariants] = {}

    def add(self, pv: PlanVariants) -> None:
        self._by_plan[pv.user_plan_id] = pv

    def get(self, user_plan_id: str) -> PlanVariants | None:
        return self._by_plan.get(user_plan_id)

    def plan_ids(self) -> list[str]:
        return list(self._by_plan)

    def has_coverage(self, user_plan_id: str) -> bool:
        pv = self._by_plan.get(user_plan_id)
        return pv is not None and len(pv.variants) >= 1

    # ---- persistence ----------------------------------------------------------------
    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for pv in self._by_plan.values():
                fh.write(
                    json.dumps(
                        {
                            "user_plan_id": pv.user_plan_id,
                            "speech_act": pv.speech_act,
                            "variants": [
                                {
                                    "variant_id": v.variant_id,
                                    "text": v.text,
                                    "expressed_fact_ids": v.expressed_fact_ids,
                                    "compiler": v.compiler,
                                    "checksum": v.checksum,
                                }
                                for v in pv.variants
                            ],
                        }
                    )
                    + "\n"
                )

    @classmethod
    def load_jsonl(cls, path: str | Path) -> RealizationBank:
        bank = cls()
        p = Path(path)
        if not p.exists():
            return bank
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pv = PlanVariants(
                    d["user_plan_id"],
                    d.get("speech_act", ""),
                    [
                        TextVariant(
                            v["variant_id"],
                            v["text"],
                            v.get("expressed_fact_ids", []),
                            v.get("compiler", {}),
                            v.get("checksum"),
                        )
                        for v in d["variants"]
                    ],
                )
                bank.add(pv)
        return bank


@dataclass
class AudioAsset:
    variant_id: str
    path: str
    duration_ms: int
    sample_rate: int
    checksum: str
    persona_id: str | None = None
    word_timestamps: list[Any] = field(default_factory=list)


class AudioBank:
    """Maps variant_id -> pre-rendered utterance clip (``audio_manifest.json``)."""

    def __init__(self) -> None:
        self._by_variant: dict[str, AudioAsset] = {}

    def add(self, asset: AudioAsset) -> None:
        self._by_variant[asset.variant_id] = asset

    def get(self, variant_id: str) -> AudioAsset | None:
        return self._by_variant.get(variant_id)

    def save_manifest(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump({vid: a.__dict__ for vid, a in self._by_variant.items()}, fh, indent=2)

    @classmethod
    def load_manifest(cls, path: str | Path) -> AudioBank:
        bank = cls()
        p = Path(path)
        if not p.exists():
            return bank
        data = json.loads(p.read_text())
        for a in data.values():
            bank.add(AudioAsset(**a))
        return bank


class AssetSelector:
    """Deterministic runtime selection of a frozen realization."""

    def __init__(self, scenario_id: str, simulator_seed: int) -> None:
        self.scenario_id = scenario_id
        self.simulator_seed = simulator_seed

    def select_variant(self, bank: RealizationBank, user_plan_id: str) -> TextVariant | None:
        pv = bank.get(user_plan_id)
        if pv is None or not pv.variants:
            return None
        return stable_choice(pv.variants, self.scenario_id, self.simulator_seed, user_plan_id)


__all__ = [
    "TextVariant",
    "PlanVariants",
    "RealizationBank",
    "AudioAsset",
    "AudioBank",
    "AssetSelector",
]
