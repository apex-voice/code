from apex_voice.artifacts.workspace import Workspace
from apex_voice.environments.state_store import StateStore
from apex_voice.schemas.artifact import ArtifactSchema, FieldGrader, FieldSpec, LifecycleState


def test_state_reset_determinism():
    st = StateStore({"a": {"b": 1}})
    c0 = st.snapshot_checksum()
    st.set("a.b", 2, media_time_ms=10)
    st.set("c", [1, 2], media_time_ms=20)
    assert st.snapshot_checksum() != c0
    st.reset()
    assert st.snapshot_checksum() == c0
    assert st.history == []


def test_state_history_records():
    st = StateStore({"x": 0})
    st.set("x", 5, media_time_ms=100, cause="tool")
    rec = st.history_records()[0]
    assert rec["old"] == 0 and rec["new"] == 5 and rec["media_time_ms"] == 100


def test_workspace_versioned_mutations_and_stale_propagation():
    schema = ArtifactSchema(
        artifact_type="FORM",
        fields=[
            FieldSpec(name="total", grader=FieldGrader.NUMERIC_TOLERANCE, dependents=["approval"]),
            FieldSpec(name="approval", grader=FieldGrader.ENUM),
        ],
    )
    ws = Workspace()
    ws.create("f1", "FORM", schema, lifecycle=LifecycleState.EMPTY)
    ws.write("f1", "total", 240, media_time_ms=10)
    ws.write("f1", "approval", "submitted", media_time_ms=20)
    # correcting 'total' marks its dependent 'approval' stale
    ws.write("f1", "total", 200, media_time_ms=30)
    art = ws.get("f1")
    assert "approval" in art.stale_fields
    assert art.fields["total"] == 200
    # writing approval again clears its staleness
    ws.write("f1", "approval", "submitted", media_time_ms=40)
    assert "approval" not in art.stale_fields
    # full mutation history is retained
    assert len(ws.mutations_for("f1", "total")) == 2


def test_workspace_external_stale_mark():
    ws = Workspace()
    ws.create("a", "FORM")
    ws.write("a", "x", 1)
    ws.mark_stale("a", "x")
    assert "x" in ws.get("a").stale_fields
