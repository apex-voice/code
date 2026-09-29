# Contributing to APEX-Voice

Thanks for your interest in improving APEX-Voice! Bug reports, new model adapters, and analysis
improvements are all welcome.

## Development setup

```bash
git clone https://github.com/apex-voice/code.git apex-voice
cd apex-voice
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,realtime,gemini]"
pre-commit install   # optional
```

## Before opening a pull request

```bash
make lint     # ruff check
make test     # offline unit + integration tests
```

If your change touches task loading, scoring or validation, also run the dataset check:

```bash
APEX_VOICE_DATA=/path/to/APEX-Voice make test-data
```

## Guidelines

- **Scoring changes are benchmark changes.** Anything that can change a PTS/AFA verdict on an
  existing run must bump `BENCHMARK_VERSION` (or `JUDGE_VERSION` for the judge prompt) and be
  called out in `CHANGELOG.md`.
- **No credentials in code or files.** Read keys from the environment via
  `apex_voice.credentials`.
- **Tests run offline.** Mock network clients. Tests that need Kokoro or the dataset must skip
  cleanly when those are absent.
- Keep pull requests focused, and add tests for new behaviour.
- New models: follow [docs/adding_a_model.md](docs/adding_a_model.md).

## Reporting issues

Please include the `apex-voice version` output, the command you ran, and (for run issues) the
`result.json` and the tail of `events.jsonl` for the affected session. Remove any credentials
before posting.

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
