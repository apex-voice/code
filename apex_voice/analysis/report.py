"""Build and write the full APEX-Voice results report from one or more campaign directories.

Outputs (in ``out_dir``):

- ``results.md``          -- human-readable tables (leaderboard, reliability, gates, latency and
  efficiency, duplex floor control + correction uptake, taxonomy slices)
- ``results.json``        -- every number in ``results.md``, machine-readable
- ``per_task.csv``        -- per-task passes out of k for each model (plus oracle tool calls)
- ``tooluse_by_task.csv`` -- per-task tool calls relative to the oracle trajectory
- ``slices.csv``          -- long-format taxonomy slice table
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apex_voice import __version__
from apex_voice.analysis.duplex import DuplexStats, duplex_stats
from apex_voice.analysis.metrics import GATES, ModelStats, model_stats, oracle_tool_counts
from apex_voice.analysis.runs import load_campaign
from apex_voice.analysis.slices import AXES, per_task_scores, slice_rows, task_labels


def _display_name(key: str) -> str:
    from apex_voice.adapters.registry import MODELS

    spec = MODELS.get(key)
    return spec.display_name if spec else key


@dataclass
class Report:
    models: list[str]
    display: dict[str, str]
    reps: int
    n_tasks: int
    stats: dict[str, ModelStats]
    oracle_tools: dict[str, int]
    duplex: dict[str, DuplexStats] = field(default_factory=dict)
    slices: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))

    @property
    def ranked(self) -> list[str]:
        """Models ordered by pass@k, then Reliable@k, then pass@1."""
        return sorted(
            self.models,
            key=lambda m: (-self.stats[m].pass_at_k, -self.stats[m].reliable_at_k, -self.stats[m].pass1_mean),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "apex_voice_version": __version__,
            "generated_at": self.generated_at,
            "reps": self.reps,
            "n_tasks": self.n_tasks,
            "models": {
                m: {
                    "display_name": self.display[m],
                    **self.stats[m].to_dict(),
                    **({"duplex": self.duplex[m].to_dict()} if m in self.duplex else {}),
                }
                for m in self.models
            },
            "slices": self.slices,
        }


def build_report(
    campaigns: dict[str, str | Path],
    tasks_dir: str | Path,
    *,
    reps: int = 3,
    judge: Any = None,
    artifact_analyses: bool = True,
    log: Callable[[str], None] = print,
) -> Report:
    """Aggregate ``{model_key: campaign_dir}`` against the task set in ``tasks_dir``.

    ``artifact_analyses`` enables the duplex correction-uptake and slice analyses, which re-grade
    every final workspace field (the semantic tier uses the LLM judge and its on-disk cache).
    """
    from apex_voice.data import list_task_dirs
    from apex_voice.tasks import load_task

    task_dirs = list_task_dirs(tasks_dir)
    oracle = oracle_tool_counts(task_dirs)
    runs = {m: load_campaign(d) for m, d in campaigns.items()}
    stats = {m: model_stats(r, reps, oracle) for m, r in runs.items()}
    rep = Report(
        models=list(campaigns),
        display={m: _display_name(m) for m in campaigns},
        reps=reps,
        n_tasks=len(task_dirs),
        stats=stats,
        oracle_tools=oracle,
    )
    if not artifact_analyses:
        return rep

    if judge is None:
        from apex_voice.scoring.judge import get_default_judge

        judge = get_default_judge()
    tasks = {d.name: load_task(d) for d in task_dirs}
    labels = {t: task_labels(lt) for t, lt in tasks.items()}
    scores = {}
    for m in rep.models:
        log(f"  grading artifacts: {rep.display[m]}")
        rep.duplex[m] = duplex_stats(runs[m], tasks, judge, reps)
        scores[m] = per_task_scores(runs[m], tasks, judge, reps)
    rep.slices = {axis: slice_rows(axis, order, labels, scores) for axis, _, order in AXES}
    return rep


def _f(x: float | None, fmt: str = "{:.1f}", none: str = "–") -> str:
    return fmt.format(x) if x is not None else none


def _pct(a: float, b: float) -> str:
    return f"{100 * a / b:.1f}" if b else "–"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def render_markdown(r: Report) -> str:
    k = r.reps
    ms = r.ranked
    L = [
        "# APEX-Voice results",
        "",
        f"Generated {r.generated_at} by apex-voice {__version__}. {r.n_tasks} tasks, condition C2 "
        f"(full-duplex), {k} repetitions per task. Counts are out of the tasks with all {k} "
        "repetitions scored (`n`).",
        "",
        "## Leaderboard",
        "",
    ]
    L += _table(
        [
            "Model",
            "n",
            f"pass@1 (mean of {k})",
            "pass@1 %",
            f"pass@{k}",
            f"pass@{k} %",
            f"Reliable@{k}",
            f"Reliable@{k} %",
        ],
        [
            [
                r.display[m],
                str(s.n),
                f"{s.pass1_mean:.1f}",
                _pct(s.pass1_mean, s.n),
                str(s.pass_at_k),
                _pct(s.pass_at_k, s.n),
                str(s.reliable_at_k),
                _pct(s.reliable_at_k, s.n),
            ]
            for m in ms
            for s in [r.stats[m]]
        ],
    )

    L += [f"## Per-repetition pass@1 and k-of-{k} distribution", ""]
    L += _table(
        ["Model", *[f"rep{i}" for i in range(k)], *[f"{i}/{k}" for i in range(k + 1)]],
        [
            [
                r.display[m],
                *[str(s.rep_pass.get(i, 0)) for i in range(k)],
                *[str(s.distribution.get(i, 0)) for i in range(k + 1)],
            ]
            for m in ms
            for s in [r.stats[m]]
        ],
    )

    L += [
        "## PTS gate pass rates (%, all runs)",
        "",
        "GS = goal satisfied, PC = process compliance, RA = required actions, WA = workspace artifact. PTS = GS ∧ PC ∧ RA ∧ WA.",
        "",
    ]
    L += _table(
        ["Model", "runs", *GATES],
        [
            [
                r.display[m],
                str(s.runs),
                *[_f(None if s.gate_rate(g) is None else 100 * s.gate_rate(g)) for g in GATES],
            ]
            for m in ms
            for s in [r.stats[m]]
        ],
    )

    L += [
        "## Latency and tool-use efficiency",
        "",
        "RSL = response-start latency (median per run, ms); TTR = time to resolution (s); "
        "tool ratio = mean tool calls per task relative to the oracle trajectory.",
        "",
    ]
    L += _table(
        [
            "Model",
            "RSL p50",
            "RSL p95",
            "TTR p50",
            "TTR p95",
            "tool calls",
            "tool ratio (mean)",
            "tool ratio (p50)",
        ],
        [
            [
                r.display[m],
                _f(s.rsl_p50, "{:.0f}"),
                _f(s.rsl_p95, "{:.0f}"),
                _f(s.ttr_p50),
                _f(s.ttr_p95),
                _f(s.tool_calls_mean),
                _f(s.tool_ratio_mean, "{:.2f}"),
                _f(s.tool_ratio_p50, "{:.2f}"),
            ]
            for m in ms
            for s in [r.stats[m]]
        ],
    )

    if r.duplex:
        L += [
            "## Full-duplex: floor control and correction uptake",
            "",
            "Overlap = simultaneous-speech share of the session; yield = barge-ins at which the "
            "agent stopped speaking; stop latency over genuine barge-ins. AFA = Artifact Field "
            "Accuracy on user-corrected vs. other required fields; attrib = share of WA failures "
            "with a wrong corrected field.",
            "",
        ]
        L += _table(
            [
                "Model",
                "overlap %",
                "yield %",
                "stop p50 (ms)",
                "stop p95 (ms)",
                "AFA corrected %",
                "AFA other %",
                "gap (pp)",
                "attrib %",
            ],
            [
                [
                    r.display[m],
                    f"{d.overlap_pct:.1f}",
                    f"{d.yield_pct:.1f}",
                    f"{d.stop_p50_ms:.0f}",
                    f"{d.stop_p95_ms:.0f}",
                    f"{d.afa_corrected:.1f}",
                    f"{d.afa_other:.1f}",
                    f"{d.gap:+.1f}",
                    f"{d.attrib_pct:.1f}",
                ]
                for m in ms
                for d in [r.duplex[m]]
            ],
        )

    if r.slices:
        L += ["## Taxonomy slices", "", f"Each cell is `success% (AFA%)` averaged over {k} repetitions.", ""]
        for axis, title, _ in AXES:
            L += [f"### {title}", ""]
            rows = []
            for row in r.slices.get(axis, []):
                cells = []
                for m in ms:
                    c = row.get(m) or {}
                    s = "–" if c.get("success") is None else f"{100 * c['success']:.0f}%"
                    a = "–" if c.get("afa") is None else f"{100 * c['afa']:.0f}%"
                    cells.append(f"{s} ({a})")
                rows.append([str(row["label"]), str(row["n"]), *cells])
            L += _table(["Label", "n", *[r.display[m] for m in ms]], rows)
    return "\n".join(L).rstrip() + "\n"


def write_report(r: Report, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    p = out / "results.md"
    p.write_text(render_markdown(r))
    written.append(p)

    p = out / "results.json"
    p.write_text(json.dumps(r.to_dict(), indent=2, default=str) + "\n")
    written.append(p)

    ms = r.ranked
    tids = sorted(set(r.oracle_tools) | {t for m in ms for t in r.stats[m].k_of_n})
    p = out / "per_task.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_id", "oracle_tool_calls", *[f"{m}_passes_of_{r.reps}" for m in ms]])
        for t in tids:
            w.writerow([t, r.oracle_tools.get(t, ""), *[r.stats[m].k_of_n.get(t, "") for m in ms]])
    written.append(p)

    p = out / "tooluse_by_task.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_id", "oracle_tool_calls", *[f"{m}_tool_ratio" for m in ms]])
        for t in tids:
            w.writerow(
                [
                    t,
                    r.oracle_tools.get(t, ""),
                    *[_f(r.stats[m].tool_ratio_by_task.get(t), "{:.3f}", "") for m in ms],
                ]
            )
    written.append(p)

    if r.slices:
        p = out / "slices.csv"
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["axis", "label", "n", "model", "success", "afa"])
            for axis, rows in r.slices.items():
                for row in rows:
                    for m in ms:
                        c = row.get(m) or {}
                        w.writerow(
                            [
                                axis,
                                row["label"],
                                row["n"],
                                m,
                                _f(c.get("success"), "{:.4f}", ""),
                                _f(c.get("afa"), "{:.4f}", ""),
                            ]
                        )
        written.append(p)
    return written


__all__ = ["Report", "build_report", "render_markdown", "write_report"]
