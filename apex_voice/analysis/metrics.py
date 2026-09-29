"""Reliability, gate, latency, and efficiency statistics computed from stored run results.

Definitions (per model, over the ``n`` tasks that have all ``k`` repetitions scored):

- **pass@1** -- mean over repetitions of the number of tasks passed (expected single-run passes).
- **pass@k** -- tasks passed in at least one of the ``k`` repetitions.
- **Reliable@k** -- tasks passed in *all* ``k`` repetitions (Pass^k).

Gate pass-rates, latency, and tool-use statistics pool every persisted run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from apex_voice.analysis.runs import RunRecord

GATES = ("GS", "PC", "RA", "WA")


def percentile(values: list[float | None], p: float) -> float | None:
    """Linear-interpolated percentile (``p`` in [0, 1]); ``None`` when empty."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    i = (len(v) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def mean(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def oracle_tool_counts(task_dirs: list[Path]) -> dict[str, int]:
    """Tool calls in each task's reference (oracle) trajectory, the per-task efficiency baseline."""
    counts = {}
    for d in task_dirs:
        pol = d / "reference" / "policy.yaml"
        if pol.exists():
            data = yaml.safe_load(pol.read_text()) or {}
            counts[d.name] = sum(len(t.get("tool_calls") or []) for t in data.get("turns", []))
    return counts


@dataclass
class ModelStats:
    reps: int
    n: int = 0
    runs: int = 0
    rep_pass: dict[int, int] = field(default_factory=dict)
    k_of_n: dict[str, int] = field(default_factory=dict)  # task -> passes out of `reps`
    distribution: dict[int, int] = field(default_factory=dict)
    pass1_mean: float = 0.0
    pass_at_k: int = 0
    reliable_at_k: int = 0
    gate_total: dict[str, int] = field(default_factory=dict)
    gate_pass: dict[str, int] = field(default_factory=dict)
    rsl_p50: float | None = None
    rsl_p95: float | None = None
    ttr_p50: float | None = None
    ttr_p95: float | None = None
    tool_calls_mean: float | None = None
    tool_ratio_mean: float | None = None
    tool_ratio_p50: float | None = None
    tool_ratio_by_task: dict[str, float] = field(default_factory=dict)

    def gate_rate(self, gate: str) -> float | None:
        tot = self.gate_total.get(gate, 0)
        return self.gate_pass.get(gate, 0) / tot if tot else None

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["gate_rates"] = {g: self.gate_rate(g) for g in GATES}
        return d


def model_stats(
    runs: dict[str, dict[int, RunRecord]], reps: int, oracle_tools: dict[str, int] | None = None
) -> ModelStats:
    oracle_tools = oracle_tools or {}
    s = ModelStats(reps=reps)
    full = sorted(t for t, v in runs.items() if all(k in v for k in range(reps)))
    s.n = len(full)
    s.rep_pass = {k: sum(runs[t][k].pts for t in full) for k in range(reps)}
    s.k_of_n = {t: sum(runs[t][k].pts for k in range(reps)) for t in full}
    s.distribution = {k: sum(1 for t in full if s.k_of_n[t] == k) for k in range(reps + 1)}
    s.pass1_mean = sum(s.rep_pass.values()) / reps if s.n else 0.0
    s.pass_at_k = sum(1 for t in full if s.k_of_n[t] >= 1)
    s.reliable_at_k = sum(1 for t in full if s.k_of_n[t] >= reps)

    s.gate_total = {g: 0 for g in GATES}
    s.gate_pass = {g: 0 for g in GATES}
    ttr, rsl, tool_calls = [], [], []
    calls_by_task: dict[str, list[float]] = {}
    for tid, by_rep in runs.items():
        for rec in by_rep.values():
            s.runs += 1
            for g in GATES:
                if g in rec.components:
                    s.gate_total[g] += 1
                    s.gate_pass[g] += int(bool(rec.components[g]))
            lat = rec.latency
            if lat.get("time_to_resolution_ms") is not None:
                ttr.append(lat["time_to_resolution_ms"] / 1000.0)
            r = (lat.get("response_start_latency_ms") or {}).get("median")
            if r is not None:
                rsl.append(r)
            if lat.get("tool_calls") is not None:
                tool_calls.append(lat["tool_calls"])
                calls_by_task.setdefault(tid, []).append(lat["tool_calls"])
    s.rsl_p50, s.rsl_p95 = percentile(rsl, 0.5), percentile(rsl, 0.95)
    s.ttr_p50, s.ttr_p95 = percentile(ttr, 0.5), percentile(ttr, 0.95)
    s.tool_calls_mean = mean(tool_calls)
    s.tool_ratio_by_task = {
        t: (sum(c) / len(c)) / oracle_tools[t] for t, c in calls_by_task.items() if oracle_tools.get(t)
    }
    ratios = list(s.tool_ratio_by_task.values())
    s.tool_ratio_mean, s.tool_ratio_p50 = mean(ratios), percentile(ratios, 0.5)
    return s


__all__ = ["GATES", "ModelStats", "model_stats", "oracle_tool_counts", "percentile", "mean"]
