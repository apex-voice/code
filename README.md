# APEX-Voice

**Measuring professional work completion by full-duplex voice agents.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-APEX--Voice-yellow.svg)](https://huggingface.co/datasets/puneetUMD/APEX-Voice)

Speech-to-speech voice agents are being deployed to do real professional work: enrolling
benefits, filing claims, screening candidates, negotiating orders. Existing voice benchmarks
measure transcription, latency, or turn-level dialogue quality. They do not measure whether the
**work product** is correct. APEX-Voice grades the professional artifact a full-duplex voice
agent produces while talking to a simulated user in real time, rather than grading the
conversation.

- **120 professional tasks** covering 10 work archetypes, 99 professions and 11 artifact classes.
  Tasks include mid-speech corrections, knowledge retrieval, approval-gated commits, and
  11–17 native tool calls per reference solution.
- **Production Task Score (PTS).** A task passes only if all four binary gates hold: goal
  satisfied (GS), process compliance (PC), required actions (RA), and a whole-artifact
  correctness gate (WA). **Artifact Field Accuracy (AFA)** gives partial credit per field.
- **A frozen synthetic user.** The user streams deterministic, pre-compiled speech with real
  duplex timing (overlap, back-channels, barge-ins), so every model faces the same user.
- **Reliability.** Every task is run 3 times, and we report pass@1, pass@3 and Reliable@3
  (the task passes in all 3 runs).

## Getting started

Everything below is copy-pasteable. Steps 1–4 are fully offline and need **no API keys**, so
you can verify your setup before spending a cent.

### Requirements

| | |
| --- | --- |
| Python | 3.11 or newer |
| Disk | ~150 MB for the package, ~2 GB for the dataset |
| OS | Linux, macOS, or Windows (WSL recommended) |
| Keys | only for step 5, and only for the models you actually want to run |

### Step 1 — Install the package

```bash
git clone https://github.com/apex-voice/code.git apex-voice
cd apex-voice

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[bench]"
```

`[bench]` (~150 MB) is everything needed to run and score the benchmark. Confirm it worked:

```bash
apex-voice version
apex-voice models                  # lists the 5 models and which keys you're missing
```

<details>
<summary><b>What the other extras do (and why the default is small)</b></summary>

| extra | installs | needed for |
| --- | --- | --- |
| `realtime` | `websockets` | GPT-realtime, GPT-live-1, Grok Voice, Step-Audio3 |
| `gemini` | `google-genai` | Gemini Live |
| `hf` | `huggingface_hub` | `apex-voice download` |
| **`bench`** | `realtime` + `gemini` + `hf` | **recommended — running and scoring the benchmark** |
| `tts` | `torch`, `kokoro`, `misaki[en]` | *only* to synthesize the simulated user's voice yourself |
| `dev` | `pytest`, `ruff`, `build` | development |
| `all` | `bench` + `tts` | everything |

The simulated user's speech ships **pre-rendered** inside the dataset and `apex-voice run`
replays it, so no TTS stack is required and every machine sends each model identical audio.
That is why `[tts]` is opt-in: it pulls PyTorch, which is ~3.5 GB from PyPI's default CUDA
wheels. If you do want to re-synthesize the speech, use the paper's CPU build (~200 MB):

```bash
pip install -e ".[all]" -c constraints/paper.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
</details>

### Step 2 — Download the dataset

The 120 task packages live on the Hugging Face Hub at
[`puneetUMD/APEX-Voice`](https://huggingface.co/datasets/puneetUMD/APEX-Voice).

```bash
apex-voice download --local-dir data/APEX-Voice
export APEX_VOICE_DATA=data/APEX-Voice     # Windows: set APEX_VOICE_DATA=data\APEX-Voice
```

The dataset is public, so no Hugging Face token is needed. If the download is gated on your
network, or you were given the data some other way, just point at any local copy instead —
`APEX_VOICE_DATA` (or `--data PATH`) accepts a dataset root containing `tasks/`, a directory of
task packages, or a single task package:

```bash
export APEX_VOICE_DATA=/path/to/APEX-Voice
```

<details>
<summary><b>Alternatives and troubleshooting</b></summary>

```bash
# Equivalent download via the huggingface-cli
huggingface-cli download puneetUMD/APEX-Voice --repo-type dataset --local-dir data/APEX-Voice

# Pin a specific revision (branch, tag or commit sha)
apex-voice download --local-dir data/APEX-Voice --revision main

# Omit --local-dir to keep the snapshot in the shared Hugging Face cache (~/.cache/huggingface)
apex-voice download
```

- **Point at a different Hub repo.** Set `APEX_VOICE_HF_REPO=org/name` to override the dataset id.
- **`dataset not found`.** `APEX_VOICE_DATA` isn't set and nothing is cached. Re-run step 2, or
  pass `--data PATH` explicitly to any command.
- **Slow or interrupted download.** Re-run the same command; it resumes and skips finished files.

After a successful download, `data/APEX-Voice/tasks/` should contain 120 directories named
`apexv1_001` … `apexv1_120`, each with a `task.yaml`.
</details>

### Step 3 — Verify everything offline

```bash
apex-voice validate
```

This runs each task package against its reference policy and against a do-nothing agent: a task
is only valid if the reference solves it and the no-op agent does not. Expect
`120/120 task packages valid`. **No API keys are used.** If this passes, your install and your
data are both good.

### Step 4 — Configure credentials

```bash
cp .env.example .env
```

Then edit `.env` and fill in only the keys for what you want to run. The CLI reads `.env` from
the working directory; real environment variables take precedence.

| you want to | set |
| --- | --- |
| Score any run (semantic judge + act observer, `gpt-4o-mini`) | `APEX_VOICE_LLM_API_KEY` — falls back to `OPENAI_API_KEY` |
| Run GPT-realtime / GPT-live-1 | `OPENAI_API_KEY` |
| Run Grok Voice | `XAI_API_KEY` |
| Run Gemini Live | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| Run Step-Audio3 | `STEPFUN_API_KEY` |

Any OpenAI-compatible endpoint works for the judge via `APEX_VOICE_LLM_BASE_URL`. Judge verdicts
are cached on disk at `APEX_VOICE_JUDGE_CACHE` (default `~/.cache/apex_voice/judge_cache.json`),
so re-scoring is cheap. `apex-voice models` shows which keys are still missing, and
[.env.example](.env.example) documents every variable including endpoint overrides.

### Step 5 — Run a model

Start with a **single-task smoke test** before launching a full campaign:

```bash
apex-voice run --model gpt-realtime --only apexv1_001 --repeats 1 --out runs/smoke
```

If that produces a `result.json`, run the real thing — 120 tasks × 3 repetitions:

```bash
apex-voice run --model gpt-realtime --out runs/gpt-realtime
```

`apex-voice run` is **resumable**: interrupt it and re-run the same command, and finished
`(task, repetition)` cells are skipped. Transient provider failures, such as dropped WebSockets
or gateway 5xx errors, are retried and never counted against the model.

Useful flags: `--limit N` (first N tasks), `--repeats N`, `--delay SECONDS` (ease rate limits),
`--only TASK_ID` (repeatable).

### Step 6 — Build the result tables

```bash
apex-voice results --runs-root runs --out results/
```

This writes `results.md`, `results.json` and CSV tables covering PTS, the four gates, AFA,
pass@1 / pass@3 / Reliable@3, latency, tool-use efficiency, duplex floor control, correction
uptake and taxonomy slices. Add `--fast` to skip the artifact re-grading analyses, or name
campaigns explicitly with `--run gpt-realtime=runs/gpt-realtime`.

### What a run produces

```
runs/gpt-realtime/apexv1_001/rep0/
  events.jsonl      full timestamped event log (audio, tool calls, user acts, barge-ins)
  workspace/        the final artifact the agent produced, plus every intermediate version
  session.wav       stereo audio — left channel = user, right channel = agent
  result.json       PTS, the four gates, AFA, latency and duplex metrics for this session
```

### Common problems

| symptom | fix |
| --- | --- |
| `dataset not found` | Run step 2, or pass `--data /path/to/APEX-Voice` |
| `<model> needs OPENAI_API_KEY` | Add the key to `.env` (step 4), then check `apex-voice models` |
| `No pre-rendered user audio for N task(s)` | Your dataset copy is incomplete — re-run step 2 |
| `The simulated user speaks with Kokoro TTS` | Same cause; or install `".[tts]"` to synthesize live |
| `N semantic-judge calls failed` | Set `APEX_VOICE_LLM_API_KEY` (or `OPENAI_API_KEY`) and re-run `results` |
| `command not found: apex-voice` | Activate the venv: `source .venv/bin/activate` |

## Models

| key | model | provider | credentials |
| --- | --- | --- | --- |
| `gpt-realtime` | GPT-realtime-2.1 | OpenAI | `OPENAI_API_KEY` |
| `grok-voice` | Grok-Voice-Think-2.0 | xAI | `XAI_API_KEY` |
| `gemini-live` | Gemini-3.8-Live | Google | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| `gpt-live-1` | GPT-live-1 | OpenAI | `OPENAI_API_KEY` |
| `step-audio3` | Step-Audio3 | StepFun | `STEPFUN_API_KEY` |

The semantic field judge and the dialogue-act observer call `gpt-4o-mini` through any
OpenAI-compatible endpoint (see step 4). To benchmark a new model, see
[docs/adding_a_model.md](docs/adding_a_model.md).

## How a run works

```
 task package ──► user simulator ──(Kokoro speech, 24 kHz)──► realtime model ◄── system prompt + tools
   (hidden state,    ▲  flow engine, barge-ins,                 │
    gold, policy)    │  corrections, approvals                  │ audio + native tool calls
                     └──────── act observer ◄───────────────────┤
                                                                ▼
                        instrumented environment (state store, knowledge base, workspace,
                        commit/approval guard) ──► PTS = GS ∧ PC ∧ RA ∧ WA, AFA, duplex + latency
```

Each session writes `events.jsonl`, which is the full timestamped event log. It also writes the
final and versioned workspace, and stereo audio (left channel = user, right channel = agent).
See [docs/metrics.md](docs/metrics.md) for every metric definition and
[docs/task_format.md](docs/task_format.md) for the task package format.

## Repository layout

```
apex_voice/
  adapters/      realtime model adapters + registry; reference text agents (oracle / no-op)
  analysis/      campaign aggregation: reliability, gates, latency, duplex, slices, reports
  artifacts/     versioned workspace and field / artifact graders
  environments/  state store, tool runtime, commit guard, events, knowledge base
  harness/       full-duplex realtime runner, text runner, clocks, audio, event logger
  scoring/       PTS, predicates, semantic field judge, duplex and latency metrics
  schemas/       pydantic models for every task file and run event
  tasks/         task loader and taxonomy lint
  user_sim/      hidden user state, flow engine, act observer, Kokoro speech compiler
  campaign.py    resumable multi-repetition campaigns
  cli.py         the `apex-voice` command
scripts/         dataset export for the Hugging Face Hub
tests/           unit + integration tests (toy task fixtures included)
docs/            metrics, reproduction, task format, adding a model
```

## Development

```bash
pip install -e ".[dev,realtime,gemini]"
make test         # pytest (offline; Kokoro tests are skipped if it is not installed)
make lint         # ruff
APEX_VOICE_DATA=/path/to/APEX-Voice make test-data   # also validate all 120 dataset tasks
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

| doc | covers |
| --- | --- |
| [docs/reproducing.md](docs/reproducing.md) | reproducing every paper table, with or without API keys |
| [docs/metrics.md](docs/metrics.md) | PTS, the four gates, AFA, latency and duplex definitions |
| [docs/task_format.md](docs/task_format.md) | the task package format |
| [docs/adding_a_model.md](docs/adding_a_model.md) | benchmarking a new model |

## Citation

```bibtex
@misc{mathur2026apexvoice,
  title  = {{APEX-Voice}: Can Voice Agents Complete Professional Workflows Through Full-Duplex Interaction},
  author = {Mathur, Puneet},
  year   = {2026},
  url    = {https://github.com/apex-voice/code}
}
```

## License

Code and data are released under the [Apache License 2.0](LICENSE). All people, companies,
identifiers and records in the tasks are synthetic.
