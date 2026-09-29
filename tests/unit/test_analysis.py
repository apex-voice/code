import json
import shutil
from pathlib import Path

import pytest

from apex_voice.analysis import build_report, load_campaign, write_report
from apex_voice.analysis.artifact import grade_artifact
from apex_voice.analysis.metrics import model_stats, percentile
from apex_voice.tasks import load_task
from tests.paths import TOY_FORM, TOY_INTERVIEW


def test_percentile_linear_interpolation():
    assert percentile([], 0.5) is None
    assert percentile([5], 0.95) == 5
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)


def _write_run(root: Path, task: str, rep: int, pts: int, *, workspace=None, tool_calls=12):
    rep_dir = root / task / f"rep{rep}"
    run_id = f"run{rep}"
    (rep_dir / run_id / "workspace").mkdir(parents=True)
    comps = {"GS": True, "PC": True, "RA": True, "WA": bool(pts)}
    (rep_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": task,
                "rep": rep,
                "run_id": run_id,
                "pts": pts,
                "components": comps,
                "latency": {
                    "time_to_resolution_ms": 100_000 + rep * 1000,
                    "tool_calls": tool_calls,
                    "response_start_latency_ms": {"median": 500 + rep},
                },
                "duplex": {"overlap_ratio": 0.02},
            }
        )
    )
    (rep_dir / run_id / "workspace" / "final_workspace.json").write_text(json.dumps(workspace or {}))
    (rep_dir / run_id / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "DUPLEX_EVENT",
                "payload": {"type": "MID_SPEECH_CORRECTION", "agent_yielded": True, "isl_ms": 40},
            }
        )
        + "\n"
    )


def test_reliability_metrics(tmp_path):
    # task A passes 3/3, task B 1/3, task C 0/3, task D is incomplete (excluded from n).
    for rep in range(3):
        _write_run(tmp_path, "A", rep, 1)
        _write_run(tmp_path, "B", rep, int(rep == 1))
        _write_run(tmp_path, "C", rep, 0)
    _write_run(tmp_path, "D", 0, 1)
    runs = load_campaign(tmp_path)
    s = model_stats(runs, reps=3, oracle_tools={"A": 12, "B": 6})
    assert s.n == 3 and s.runs == 10
    assert s.pass_at_k == 2 and s.reliable_at_k == 1
    assert s.pass1_mean == pytest.approx(4 / 3)
    assert s.distribution == {0: 1, 1: 1, 2: 0, 3: 1}
    assert s.gate_rate("GS") == 1.0 and s.gate_rate("WA") == pytest.approx(5 / 10)
    assert s.tool_ratio_by_task == {"A": 1.0, "B": 2.0}


def _oracle_workspace(lt):
    ws = {}
    for ae in lt.gold.artifact_expectations:
        ws[ae.artifact_id] = {
            "lifecycle": "COMMITTED",
            "stale_fields": [],
            "fields": {fe.field: fe.expected for fe in ae.fields},
        }
    return ws


def test_grade_artifact_gold_passes_and_missing_fails():
    lt = load_task(TOY_FORM)
    good = grade_artifact(lt, _oracle_workspace(lt), judge=None)
    assert good.wa_ok and good.afa == 1.0
    bad = grade_artifact(lt, {}, judge=None)
    assert not bad.wa_ok and "artifact-missing" in bad.reasons


def test_build_and_write_report(tmp_path):
    data = tmp_path / "data"
    for t in (TOY_FORM, TOY_INTERVIEW):
        shutil.copytree(t, data / "tasks" / t.name)
    camp = tmp_path / "runs" / "gpt-realtime"
    for t in (TOY_FORM, TOY_INTERVIEW):
        ws = _oracle_workspace(load_task(t))
        for rep in range(3):
            _write_run(camp, t.name, rep, int(t is TOY_FORM), workspace=ws)
    rep = build_report({"gpt-realtime": camp}, data, judge=lambda *a, **k: False, log=lambda _m: None)
    s = rep.stats["gpt-realtime"]
    assert (s.n, s.pass_at_k, s.reliable_at_k) == (2, 1, 1)
    assert rep.duplex["gpt-realtime"].yield_pct == 100.0
    assert rep.slices["archetype"]
    paths = {p.name for p in write_report(rep, tmp_path / "out")}
    assert paths == {"results.md", "results.json", "per_task.csv", "tooluse_by_task.csv", "slices.csv"}
    md = (tmp_path / "out" / "results.md").read_text()
    assert "GPT-realtime-2.1" in md and "## Leaderboard" in md
