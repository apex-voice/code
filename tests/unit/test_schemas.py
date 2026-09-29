import pytest
from pydantic import ValidationError

from apex_voice import schemas as s
from apex_voice.schemas.flow import FlowSpec


def test_taskspec_roundtrip():
    t = s.TaskSpec(id="t1", title="T", archetype=s.WorkArchetype.INTERVIEW)
    assert s.TaskSpec.model_validate_json(t.model_dump_json()) == t


def test_taxonomy_roundtrip():
    from apex_voice.schemas.taxonomy import ProfessionalWork, TaxonomySpec

    tx = TaxonomySpec(
        task_id="t1",
        professional_work=ProfessionalWork(
            primary_archetype=s.WorkArchetype.FORM_FILL,
            economic_function="finance_operations",
            industry_setting="horizontal_enterprise",
        ),
    )
    assert TaxonomySpec.model_validate_json(tx.model_dump_json()) == tx


def test_invalid_enum_rejected():
    with pytest.raises(ValidationError):
        s.TaskSpec(id="t", title="T", archetype="NOT_A_REAL_ARCHETYPE")


def test_flow_rejects_dangling_transition():
    with pytest.raises(ValueError):
        FlowSpec.model_validate(
            {
                "start": "A",
                "states": {"A": {"transitions": [{"when": {}, "next": "MISSING"}]}},
            }
        )


def test_flow_rejects_missing_start():
    with pytest.raises(ValueError):
        FlowSpec.model_validate({"start": "X", "states": {"A": {}}})
