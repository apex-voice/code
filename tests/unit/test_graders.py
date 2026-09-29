from apex_voice.artifacts.field_graders import grade_field, set_f1
from apex_voice.artifacts.graders import grade_artifact
from apex_voice.artifacts.workspace import Workspace
from apex_voice.schemas.artifact import FieldGrader, LifecycleState
from apex_voice.schemas.grading import ArtifactExpectation, ArtifactFieldExpectation


def boom(*_args):
    raise AssertionError("the judge must not be called on an exact / fast-path match")


def test_field_graders():
    assert grade_field(FieldGrader.EXACT, "A", "A") == 1.0
    assert grade_field(FieldGrader.CASEFOLD_EXACT, "abc", "ABC") == 1.0
    assert grade_field(FieldGrader.NUMERIC_TOLERANCE, 200.001, 200.0, 0.01) == 1.0
    assert grade_field(FieldGrader.NUMERIC_TOLERANCE, 205, 200.0, 0.01) == 0.0
    # currency/prose in numeric fields must not break parsing
    assert grade_field(FieldGrader.NUMERIC_TOLERANCE, "$480", 480.0, 0.01) == 1.0
    assert grade_field(FieldGrader.NUMERIC_TOLERANCE, "4,000", 4000.0, 0.5) == 1.0
    assert grade_field(FieldGrader.NUMERIC_TOLERANCE, "about 20", 20.0, 0.5) == 1.0
    assert grade_field(FieldGrader.NUMERIC_TOLERANCE, "$999", 480.0, 0.01) == 0.0
    # month-name / US dates denote the same calendar date as ISO gold
    assert grade_field(FieldGrader.NORMALIZED_DATE, "February 1, 2026", "2026-02-01") == 1.0
    assert grade_field(FieldGrader.NORMALIZED_DATE, "Feb 1 2026", "2026-02-01") == 1.0
    assert grade_field(FieldGrader.NORMALIZED_DATE, "2/1/2026", "2026-02-01") == 1.0
    assert grade_field(FieldGrader.NORMALIZED_DATE, "March 3, 2026", "2026-02-01") == 0.0
    assert set_f1(["a", "b"], ["b", "a"]) == 1.0
    assert 0 < set_f1(["a", "x"], ["a", "b"]) < 1


def test_identifier_grading_is_judge_only():
    from apex_voice.artifacts.field_graders import grade_field_2tier

    # Exact match short-circuits without any judge call.
    assert grade_field_2tier(FieldGrader.EXACT, "CC-4419", "CC-4419", None, field="id", judge=boom) == 1.0
    # Any non-identical id -> the judge decides (no deterministic normalization shortcut).
    assert (
        grade_field_2tier(FieldGrader.EXACT, "CC4419", "CC-4419", None, field="id", judge=lambda *a: True)
        == 1.0
    )
    assert (
        grade_field_2tier(FieldGrader.EXACT, "CC4419", "CC5500", None, field="id", judge=lambda *a: False)
        == 0.0
    )
    # Offline/no judge: falls back to the deterministic base grader (strict).
    assert grade_field_2tier(FieldGrader.EXACT, "CC4419", "CC-4419", None, field="id", judge=None) == 0.0


def test_semantic_grader_two_tier():
    from apex_voice.artifacts.field_graders import grade_field_2tier

    # Offline (no judge): SEMANTIC behaves like casefold_exact -> deterministic.
    assert grade_field(FieldGrader.SEMANTIC, "Rollback Deploy", "rollback deploy") == 1.0
    assert grade_field(FieldGrader.SEMANTIC, "rollback of the deploy", "rollback deploy") == 0.0
    assert (
        grade_field_2tier(
            FieldGrader.SEMANTIC,
            "rollback of the deploy",
            "rollback deploy",
            None,
            field="mitigation",
            judge=None,
        )
        == 0.0
    )
    # Casefold fast-path passes without ever calling the judge.
    assert (
        grade_field_2tier(
            FieldGrader.SEMANTIC, "Rollback Deploy", "rollback deploy", None, field="mitigation", judge=boom
        )
        == 1.0
    )
    # On a miss, a supplied judge decides (stub returns True/False).
    assert (
        grade_field_2tier(
            FieldGrader.SEMANTIC,
            "rollback of the deploy",
            "rollback deploy",
            None,
            field="mitigation",
            judge=lambda *a: True,
        )
        == 1.0
    )
    assert (
        grade_field_2tier(
            FieldGrader.SEMANTIC,
            "different thing",
            "rollback deploy",
            None,
            field="mitigation",
            judge=lambda *a: False,
        )
        == 0.0
    )


def test_artifact_grader_afa_completeness_and_stale():
    ws = Workspace()
    ws.create("f1", "FORM", lifecycle=LifecycleState.COMMITTED)
    ws.write("f1", "a", "yes")
    ws.write("f1", "b", "no")
    ws.mark_stale("f1", "b")
    exp = ArtifactExpectation(
        artifact_id="f1",
        artifact_type="FORM",
        lifecycle_min="COMMITTED",
        must_not_be_stale=["b"],
        fields=[
            ArtifactFieldExpectation(field="a", expected="yes", grader=FieldGrader.EXACT, required=True),
            ArtifactFieldExpectation(field="b", expected="no", grader=FieldGrader.EXACT, required=True),
        ],
    )
    score = grade_artifact(ws, exp)
    assert score.completeness == 1.0
    assert score.lifecycle_ok
    # 'b' is stale so it scores 0 even though value matches -> AFA < 1
    assert score.afa < 1.0
    assert score.stale_fact_rate == 1.0
