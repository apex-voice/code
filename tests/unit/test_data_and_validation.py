import shutil

import pytest

from apex_voice.data import list_task_dirs, resolve_tasks_dir
from apex_voice.validation import validate_task
from tests.paths import TOY_FORM, TOY_INTERVIEW


def test_resolve_accepts_dataset_root_task_dir_and_single_task(tmp_path, monkeypatch):
    monkeypatch.delenv("APEX_VOICE_DATA", raising=False)
    root = tmp_path / "APEX-Voice"
    shutil.copytree(TOY_FORM, root / "tasks" / TOY_FORM.name)
    shutil.copytree(TOY_INTERVIEW, root / "tasks" / TOY_INTERVIEW.name)
    assert resolve_tasks_dir(root) == root / "tasks"
    assert resolve_tasks_dir(root / "tasks") == root / "tasks"
    assert resolve_tasks_dir(TOY_FORM) == TOY_FORM
    assert [d.name for d in list_task_dirs(root)] == ["toy_form_001", "toy_interview_001"]
    assert [d.name for d in list_task_dirs(root, {"toy_form_001"})] == ["toy_form_001"]
    monkeypatch.setenv("APEX_VOICE_DATA", str(root))
    assert resolve_tasks_dir(None) == root / "tasks"


def test_resolve_without_data_raises(monkeypatch):
    monkeypatch.delenv("APEX_VOICE_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="APEX_VOICE_DATA"):
        resolve_tasks_dir(None)


@pytest.mark.parametrize("task", [TOY_FORM, TOY_INTERVIEW])
def test_toy_tasks_validate(task):
    r = validate_task(task)
    assert r.loaded and r.oracle_pass and r.noop_fail, r.problems


def test_broken_task_reports_problem(tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(TOY_FORM, bad)
    (bad / "grading" / "gold.yaml").write_text("not: [valid")
    r = validate_task(bad)
    assert not r.ok and r.problems
