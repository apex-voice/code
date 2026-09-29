from apex_voice.adapters.text import NoOpAgent, ScriptedAgent, turn
from apex_voice.config import Condition, RunConfig
from apex_voice.harness.runner import run_text_task
from apex_voice.tasks import load_task
from tests.paths import TOY_FORM


def _cfg():
    return RunConfig(task_id="toy_form_001", task_version="1.0.0", condition=Condition.C0, simulator_seed=1)


def _run(agent):
    lt = load_task(TOY_FORM)
    return run_text_task(lt, agent, _cfg())


def test_oracle_succeeds():
    lt = load_task(TOY_FORM)
    res = run_text_task(lt, ScriptedAgent("oracle", lt.reference_turns), _cfg())
    assert res.success
    assert res.fallback_events == 0
    assert res.leakage_violations == []


def test_noop_fails():
    assert not _run(NoOpAgent()).success


def test_revise_stale_fails():
    """Agent ignores the spoken correction, leaving a stale/wrong total -> WA fails."""
    stale = ScriptedAgent(
        "stale",
        [
            turn("What is your employee id?"),
            turn("What was the trip purpose?", [("update_expense", {"fields": {"employee_id": "E1042"}})]),
            turn("What cost center?", [("update_expense", {"fields": {"trip_purpose": "client onsite"}})]),
            turn("What was the total amount?", [("update_expense", {"fields": {"cost_center": "CC-330"}})]),
            turn(
                "Recorded. Confirming the total is 240 dollars.",
                [("update_expense", {"fields": {"total_amount": 240.0}})],
            ),
            # ignores the correction; requests approval anyway
            turn("Okay, shall I submit?", [("request_approval", {"target_id": "expense_1"})]),
            turn(
                "Submitting.",
                [("submit_expense", {"target_id": "expense_1", "artifact_id": "expense_1"})],
                end=True,
            ),
        ],
    )
    res = _run(stale)
    assert not res.success
    assert "expense_1" in res.pts.artifact_failures


def test_premature_commit_fails_critical_gate():
    """Submitting before approval trips the no_unauthorized_submit critical gate."""
    premature = ScriptedAgent(
        "premature",
        [
            turn(
                "Filling and submitting.",
                [
                    (
                        "update_expense",
                        {
                            "fields": {
                                "employee_id": "E1042",
                                "trip_purpose": "client onsite",
                                "cost_center": "CC-330",
                                "total_amount": 200.0,
                            }
                        },
                    ),
                    ("submit_expense", {"target_id": "expense_1", "artifact_id": "expense_1"}),
                ],
                end=True,
            ),
        ],
    )
    res = _run(premature)
    assert not res.success
    assert "no_unauthorized_submit" in res.pts.critical_gate_failures
    assert res.pts.pc is False


def test_event_log_written(tmp_path):
    lt = load_task(TOY_FORM)
    res = run_text_task(lt, ScriptedAgent("oracle", lt.reference_turns), _cfg(), run_dir=tmp_path)
    assert res.event_log_path and (tmp_path / res.run_id / "events.jsonl").exists()
