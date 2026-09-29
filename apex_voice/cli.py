"""``apex-voice`` command-line interface.

Commands:

- ``models``    list the benchmarked realtime models and the credentials each needs
- ``download``  fetch the task dataset from the Hugging Face Hub
- ``validate``  validate task packages offline (schema, taxonomy lint, oracle solves, no-op fails)
- ``run``       run a resumable benchmark campaign for one model
- ``results``   aggregate campaign directories into result tables
- ``version``   print package and benchmark versions
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from apex_voice import BENCHMARK_VERSION, __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="APEX-Voice: professional work completion benchmark for full-duplex voice agents.",
)
console = Console()

DATA_HELP = (
    "Dataset root, directory of task packages, or a single task package. "
    "Defaults to $APEX_VOICE_DATA, then the Hugging Face cache."
)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Populate ``os.environ`` from a ``KEY=VALUE`` file without overriding existing variables."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


@app.callback()
def _main(
    env_file: Path = typer.Option(
        Path(".env"), "--env-file", help="Load environment variables from this file if present."
    ),
) -> None:
    load_dotenv(env_file)


def _tasks(data: str | None, only: list[str] | None, download: bool = True) -> list[Path]:
    from apex_voice.data import list_task_dirs, resolve_tasks_dir

    try:
        root = resolve_tasks_dir(data, download=download)
    except (FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from None
    dirs = list_task_dirs(root, set(only) if only else None)
    if not dirs:
        console.print(f"[red]no task packages selected under {root}[/red]")
        raise typer.Exit(2)
    return dirs


def _require_llm_key() -> None:
    from apex_voice.credentials import env_first

    if not env_first("APEX_VOICE_LLM_API_KEY", "OPENAI_API_KEY"):
        console.print(
            "[red]The semantic field judge needs an OpenAI-compatible key: set "
            "APEX_VOICE_LLM_API_KEY (or OPENAI_API_KEY). See .env.example.[/red]"
        )
        raise typer.Exit(2)


@app.command()
def version() -> None:
    """Print package and benchmark versions."""
    console.print(f"apex-voice {__version__} (benchmark {BENCHMARK_VERSION})")


@app.command()
def models() -> None:
    """List the benchmarked realtime models."""
    from apex_voice.adapters.registry import MODELS
    from apex_voice.credentials import env_first

    t = Table("key", "model", "provider", "credentials", "extra", "ready")
    for m in MODELS.values():
        ready = "[green]yes[/green]" if env_first(*m.required_env) else "[yellow]no key[/yellow]"
        t.add_row(m.key, m.display_name, m.provider, " | ".join(m.required_env), m.extra, ready)
    console.print(t)


@app.command()
def download(
    local_dir: Path | None = typer.Option(None, help="Copy the snapshot here."),
    revision: str | None = typer.Option(None, help="Dataset revision (branch/tag/sha)."),
) -> None:
    """Download the APEX-Voice dataset from the Hugging Face Hub."""
    from apex_voice.data import download_dataset, hf_dataset_id

    path = download_dataset(revision=revision, local_dir=local_dir)
    console.print(f"{hf_dataset_id()} -> {path}")
    console.print(f"Tip: export APEX_VOICE_DATA={path}")


@app.command()
def validate(
    data: str | None = typer.Option(None, "--data", "-d", help=DATA_HELP),
    only: list[str] | None = typer.Option(None, "--only", help="Task id(s) to validate."),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON report per task."),
    skip_policies: bool = typer.Option(False, help="Skip the oracle / no-op runs."),
) -> None:
    """Validate task packages offline (no API keys needed)."""
    from apex_voice.validation import validate_task

    reports = [validate_task(d, run_policies=not skip_policies) for d in _tasks(data, only)]
    if as_json:
        for r in reports:
            print(json.dumps(r.to_dict()))
    else:
        for r in reports:
            if not r.ok:
                console.print(f"[red]FAIL[/red] {r.task_id}: " + "; ".join(r.problems))
            for w in r.warnings:
                console.print(f"[yellow]warn[/yellow] {r.task_id}: {w}")
    bad = sum(not r.ok for r in reports)
    console.print(f"{len(reports) - bad}/{len(reports)} task packages valid")
    raise typer.Exit(1 if bad else 0)


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model key (see `apex-voice models`)."),
    data: str | None = typer.Option(None, "--data", "-d", help=DATA_HELP),
    out: Path | None = typer.Option(None, "--out", "-o", help="Campaign directory [default: runs/<model>]."),
    repeats: int = typer.Option(3, help="Repetitions per task."),
    max_turns: int = typer.Option(44, help="Maximum user turns per session."),
    only: list[str] | None = typer.Option(None, "--only", help="Restrict to these task id(s)."),
    limit: int | None = typer.Option(None, help="Run only the first N tasks."),
    delay: float = typer.Option(0.0, help="Seconds to pause between sessions (rate limits)."),
    user_audio: str = typer.Option(
        "auto",
        help="Simulated-user speech: 'prerendered' replays the dataset's clips (bit-exact paper audio), "
        "'kokoro' synthesizes live, 'auto' prefers the clips when the task ships them.",
    ),
    allow_dummy_tts: bool = typer.Option(
        False, help="Use a synthetic tone voice if Kokoro is not installed (debugging only)."
    ),
) -> None:
    """Run a resumable benchmark campaign (condition C2, full-duplex) for one model."""
    from apex_voice.adapters.registry import get_model
    from apex_voice.campaign import CampaignSettings, run_campaign
    from apex_voice.credentials import env_first
    from apex_voice.user_sim.kokoro_compiler import kokoro_available
    from apex_voice.user_sim.prerendered import USER_AUDIO_MODES, has_prerendered_audio

    try:
        spec = get_model(model)
    except KeyError as e:
        console.print(f"[red]{e.args[0]}[/red]")
        raise typer.Exit(2) from None
    if not env_first(*spec.required_env):
        console.print(f"[red]{spec.display_name} needs {' or '.join(spec.required_env)}.[/red]")
        raise typer.Exit(2)
    _require_llm_key()
    if user_audio not in USER_AUDIO_MODES:
        console.print(f"[red]--user-audio must be one of {', '.join(USER_AUDIO_MODES)}.[/red]")
        raise typer.Exit(2)

    dirs = _tasks(data, only)[: limit or None]
    missing = [d.name for d in dirs if not has_prerendered_audio(d)]
    if user_audio == "prerendered" and missing:
        console.print(f"[red]No pre-rendered user audio for {len(missing)} task(s), e.g. {missing[0]}.[/red]")
        raise typer.Exit(2)
    needs_tts = user_audio == "kokoro" or (user_audio == "auto" and missing)
    if needs_tts and not kokoro_available() and not allow_dummy_tts:
        console.print(
            "[red]The simulated user speaks with Kokoro TTS: pip install 'apex-voice[tts]', use a dataset "
            "copy with pre-rendered user audio, or pass --allow-dummy-tts for a non-scoring smoke test.[/red]"
        )
        raise typer.Exit(2)
    settings = CampaignSettings(
        model=model,
        out_dir=out or Path("runs") / model,
        repeats=repeats,
        max_turns=max_turns,
        delay_s=delay,
        user_audio=user_audio,
    )
    rows = asyncio.run(run_campaign(dirs, settings, log=console.print))
    n_pass = sum(int(bool(r.get("pts"))) for r in rows)
    n_err = sum(bool(r.get("error")) for r in rows)
    console.print(
        f"[bold]{spec.display_name}[/bold]: {len(rows)}/{len(dirs) * repeats} runs "
        f"complete, {n_pass} passed, {n_err} errored -> {settings.out_dir}"
    )


def _parse_runs(run: list[str], runs_root: Path | None) -> dict[str, Path]:
    from apex_voice.adapters.registry import MODELS

    campaigns: dict[str, Path] = {}
    if runs_root:
        for key in MODELS:
            if (runs_root / key).is_dir():
                campaigns[key] = runs_root / key
    for item in run:
        key, sep, path = item.partition("=")
        if not sep:
            console.print(f"[red]--run expects MODEL=DIR, got '{item}'[/red]")
            raise typer.Exit(2)
        campaigns[key] = Path(path)
    if not campaigns:
        console.print("[red]no campaigns given: use --runs-root DIR or --run MODEL=DIR[/red]")
        raise typer.Exit(2)
    return campaigns


@app.command()
def results(
    run: list[str] = typer.Option([], "--run", "-r", help="MODEL=DIR campaign (repeatable)."),
    runs_root: Path | None = typer.Option(
        None, help="Directory containing one campaign sub-directory per model key."
    ),
    data: str | None = typer.Option(None, "--data", "-d", help=DATA_HELP),
    out: Path = typer.Option(Path("results"), "--out", "-o", help="Output directory."),
    repeats: int = typer.Option(3, help="Repetitions per task used for pass@k / Reliable@k."),
    fast: bool = typer.Option(
        False, help="Skip the artifact re-grading analyses (duplex correction uptake and taxonomy slices)."
    ),
) -> None:
    """Aggregate campaign directories into results.md / results.json / CSV tables."""
    from apex_voice.analysis import build_report, write_report
    from apex_voice.data import resolve_tasks_dir

    campaigns = _parse_runs(run, runs_root)
    try:
        tasks_dir = resolve_tasks_dir(data, download=True)
    except (FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from None
    judge = None
    if not fast:
        from apex_voice.scoring.judge import get_default_judge

        judge = get_default_judge()
    report = build_report(
        campaigns, tasks_dir, reps=repeats, judge=judge, artifact_analyses=not fast, log=console.print
    )
    for p in write_report(report, out):
        console.print(f"wrote {p}")
    if judge is not None and judge.errors:
        console.print(
            f"[yellow]warning: {judge.errors} semantic-judge calls failed and were scored "
            "as FAIL; set APEX_VOICE_LLM_API_KEY (or OPENAI_API_KEY) and re-run.[/yellow]"
        )
    t = Table("model", "n", "pass@1", f"pass@{repeats}", f"Reliable@{repeats}")
    for m in report.ranked:
        s = report.stats[m]
        t.add_row(report.display[m], str(s.n), f"{s.pass1_mean:.1f}", str(s.pass_at_k), str(s.reliable_at_k))
    console.print(t)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
