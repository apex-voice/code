from typer.testing import CliRunner

from apex_voice import __version__
from apex_voice.cli import app, load_dotenv
from tests.paths import TOY_FORM

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and __version__ in r.output


def test_models_lists_all_keys():
    r = runner.invoke(app, ["models"])
    assert r.exit_code == 0
    for key in ("gpt-realtime", "grok-voice", "gemini-live", "gpt-live-1", "step-audio3"):
        assert key in r.output


def test_validate_single_task():
    r = runner.invoke(app, ["validate", "--data", str(TOY_FORM)])
    assert r.exit_code == 0, r.output
    assert "1/1 task packages valid" in r.output


def test_run_requires_credentials(monkeypatch):
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    r = runner.invoke(app, ["run", "--model", "step-audio3", "--data", str(TOY_FORM)])
    assert r.exit_code == 2 and "STEPFUN_API_KEY" in r.output


def test_run_rejects_unknown_model():
    r = runner.invoke(app, ["run", "--model", "nope", "--data", str(TOY_FORM)])
    assert r.exit_code == 2


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nAPEX_T_X=from_file\nexport APEX_T_Y='quoted'\nAPEX_T_Z=\n")
    monkeypatch.setenv("APEX_T_X", "from_shell")
    monkeypatch.delenv("APEX_T_Y", raising=False)
    monkeypatch.delenv("APEX_T_Z", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["APEX_T_X"] == "from_shell"
    assert os.environ["APEX_T_Y"] == "quoted"
    assert "APEX_T_Z" not in os.environ
