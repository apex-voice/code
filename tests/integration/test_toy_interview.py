from apex_voice.adapters.text import NoOpAgent, ScriptedAgent, turn
from apex_voice.config import Condition, RunConfig
from apex_voice.harness.runner import run_text_task
from apex_voice.tasks import load_task
from tests.paths import TOY_INTERVIEW


def _cfg():
    return RunConfig(task_id="toy_interview_001", task_version="1.0.0", condition=Condition.C0)


def test_oracle_succeeds():
    lt = load_task(TOY_INTERVIEW)
    res = run_text_task(lt, ScriptedAgent("oracle", lt.reference_turns), _cfg())
    assert res.success
    assert res.leakage_violations == []


def test_noop_fails():
    lt = load_task(TOY_INTERVIEW)
    assert not run_text_task(lt, NoOpAgent(), _cfg()).success


def test_fabricated_evidence_fails():
    """Recording an unsupported evidence item lowers set-F1 -> critical gate + WA fail."""
    lt = load_task(TOY_INTERVIEW)
    fabricator = ScriptedAgent(
        "fab",
        [
            turn("Tell me about your technical background."),
            turn(
                "Now your leadership experience.",
                [
                    (
                        "update_interview",
                        {
                            "fields": {
                                "technical_depth.evidence": [
                                    "Built a payments service",
                                    "Invented blockchain",
                                ],
                                "technical_depth.status": "covered",
                            }
                        },
                    ),
                ],
            ),
            turn(
                "Thanks, that's all.",
                [
                    (
                        "update_interview",
                        {
                            "fields": {
                                "leadership.evidence": ["Led a team of six engineers"],
                                "leadership.status": "covered",
                            }
                        },
                    ),
                    ("set_ready", {}),
                ],
                end=True,
            ),
        ],
    )
    res = run_text_task(lt, fabricator, _cfg())
    assert not res.success
    assert "no_fabricated_evidence" in res.pts.critical_gate_failures
