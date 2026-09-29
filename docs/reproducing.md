# Reproducing the paper results

There are two levels of reproduction:

- **Tables from the published runs** (offline, a few minutes, no API keys). This is section 0.
- **New campaigns** against the hosted models (sections 1–3). Hosted models change and sample
  stochastically, so a new campaign gives comparable numbers, not identical ones.

## 0. Regenerate the paper tables from the published runs

[`puneetUMD/APEX-Voice-Runs`](https://huggingface.co/datasets/puneetUMD/APEX-Voice-Runs)
holds the 1,800 scored sessions behind the paper (event logs, final workspaces, `result.json`)
and the judge-verdict cache. It doesn't include audio.

```bash
pip install -e ".[bench]"
apex-voice download --local-dir data/APEX-Voice
huggingface-cli download puneetUMD/APEX-Voice-Runs --repo-type dataset --local-dir data/APEX-Voice-Runs
export APEX_VOICE_DATA=data/APEX-Voice
export APEX_VOICE_JUDGE_CACHE=data/APEX-Voice-Runs/judge_cache.json
apex-voice results --runs-root data/APEX-Voice-Runs/runs --out results/
```

The output matches `reference_results/` in that dataset except for the `generated_at`
timestamp. It also matches the paper, apart from the one rounding difference described at the
end of this page.

## 1. Setup

```bash
pip install -e ".[bench]" -c constraints/paper.txt
apex-voice download --local-dir data/APEX-Voice
export APEX_VOICE_DATA=data/APEX-Voice
apex-voice validate                   # expect: 120/120 task packages valid
cp .env.example .env                  # add the provider keys + judge key
```

`[bench]` is enough to reproduce every number: the simulated user's speech is replayed from
the dataset. Swap in `".[all]" --extra-index-url https://download.pytorch.org/whl/cpu` only if
you want to re-synthesize that speech (see below).

**Simulated-user audio.** Each task package ships its user speech pre-rendered
(`user/audio/`: one FLAC per utterance variant plus an index). `apex-voice run` replays these
clips by default (`--user-audio auto`), so every machine streams the model exactly the same
16-bit PCM, and Kokoro/torch are needed only for texts missing from the index.

The clips are the canonical simulated-user audio for this benchmark. They were rendered with
the paper environment (`constraints/paper.txt`: CPU PyTorch 2.7.0, kokoro 0.9.4, misaki 0.9.4)
and the Kokoro-82M revision pinned in `apex_voice/user_sim/kokoro_compiler.py`, which is the
same pinned setup the published campaigns used to synthesize the user's speech live. Each
`user/audio/meta.json` records the exact renderer versions. Because the clips fix the audio
byte-for-byte, replaying them is *more* reproducible than the live synthesis used for the
published runs, whose waveforms depended on the host's CPU thread configuration.

To synthesize live instead (`--user-audio kokoro`), note that Kokoro-82M is content-seeded and
pinned to the paper's Hugging Face revision, but its output is bit-exact only under the same
PyTorch build *and* thread configuration. Other PyTorch versions, thread counts or a CUDA
device give the same durations and near-identical waveforms, but not identical samples — which
is exactly why the pre-rendered clips exist. `scripts/render_user_audio.py` regenerates them;
render single-threaded (`OMP_NUM_THREADS=1`) to get host-independent output.

## 2. Run the five campaigns

Each campaign is 120 tasks × 3 repetitions in condition C2. A session lasts about 3–6 minutes of
real time, because user speech is streamed in real time.

```bash
apex-voice run --model gpt-realtime --out runs/gpt-realtime
apex-voice run --model grok-voice   --out runs/grok-voice
apex-voice run --model gemini-live  --out runs/gemini-live
apex-voice run --model gpt-live-1   --out runs/gpt-live-1
apex-voice run --model step-audio3  --out runs/step-audio3
```

Useful options:

- `--only apexv1_001 --only apexv1_002` restricts the run to specific tasks.
- `--limit N` runs only the first N tasks.
- `--repeats`, `--max-turns` (default 44).
- `--delay SECONDS` pauses between sessions for tight provider rate limits.

Campaigns are **resumable** and run **coverage-first**: every task's first repetition runs
before any second repetition. Re-running the same command continues where it stopped.
Several models can run concurrently in separate processes. Each output directory is protected
by a lock file.

Output layout:

```
runs/<model>/
  campaign.json                          # model, settings, package/benchmark/judge versions
  <task_id>/rep<k>/result.json           # PTS, gate components, latency, duplex, errors
  <task_id>/rep<k>/<run_id>/events.jsonl # timestamped event log
  <task_id>/rep<k>/<run_id>/workspace/   # final workspace + version history
  <task_id>/rep<k>/<run_id>/audio/       # stereo.wav (L=user, R=agent), user.wav, agent.wav
```

## 3. Aggregate

```bash
apex-voice results --runs-root runs --out results/
# or explicitly:
apex-voice results --run gpt-realtime=runs/gpt-realtime --run grok-voice=runs/grok-voice --out results/
```

This writes `results.md`, `results.json`, `per_task.csv`, `tooluse_by_task.csv` and `slices.csv`.
The duplex correction-uptake and slice analyses re-grade every final workspace, so they need the
judge key unless every verdict is already cached. `--fast` skips them and reports only the
metrics stored in `result.json`.

## Expected numbers and variance

The paper's campaigns produced the leaderboard in the [README](../README.md). Realtime models are
hosted services that change over time and sample stochastically, so a new campaign will not
match per-task outcomes exactly. With 3 repetitions over 120 tasks, expect pass@1 to vary by a
few tasks between campaigns. Reliable@3 is the most stable ranking signal.

Things that affect comparability:

- **Model versions.** Adapters pin the evaluated model ids (`gpt-realtime-2.1`,
  `grok-voice-think-fast-2.0`, `gemini-3.8-live`, `gpt-live-1` with a `gpt-4o` backend,
  `stepaudio-3-realtime-preview`). Override them only if you intend to evaluate a different model.
- **Judge.** Use `gpt-4o-mini` (the default) so field verdicts are comparable. The judge prompt is
  versioned (`JUDGE_VERSION`) and recorded in `campaign.json`.
- **User voice.** Scored runs use the dataset's pre-rendered clips (or live Kokoro with the
  `[tts]` extra). `--allow-dummy-tts` exists only for
  plumbing tests; its synthetic tones are not intelligible speech.

Note on the duplex stop-latency statistics: they are computed only from the event log of each
repetition's *scored* session. Sessions abandoned after a transient infrastructure failure are
excluded. The analysis scripts behind the paper tables pooled those abandoned sessions, which
changes at most the rounding of the reported stop-latency percentiles (e.g., 47 → 46 ms).
