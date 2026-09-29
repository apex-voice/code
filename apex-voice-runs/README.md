---
license: apache-2.0
language:
  - en
pretty_name: APEX-Voice Paper Runs
size_categories:
  - 1K<n<10K
tags:
  - benchmark
  - voice-agents
  - full-duplex
  - speech-to-speech
  - tool-use
  - evaluation
  - reproducibility
---

# APEX-Voice: paper runs

The complete, scored benchmark runs behind the APEX-Voice paper: **5 realtime voice models × 120
tasks × 3 repetitions = 1,800 sessions**, condition C2 (full-duplex), plus the LLM-judge verdict
cache used to grade them. With this dataset and the
[APEX-Voice code](https://github.com/puneetmathur/apex-voice) you can regenerate every table in
the paper **offline, with no API keys**.

- Tasks: [`puneetUMD/APEX-Voice`](https://huggingface.co/datasets/puneetUMD/APEX-Voice)
- Code: <https://github.com/puneetmathur/apex-voice>
- License: Apache 2.0. All people, companies, identifiers and records are synthetic.

## Reproduce the paper tables

```bash
pip install -e ".[all]"                       # in a clone of the code repository
apex-voice download --local-dir data/APEX-Voice
huggingface-cli download puneetUMD/APEX-Voice-Runs --repo-type dataset --local-dir data/APEX-Voice-Runs

export APEX_VOICE_DATA=data/APEX-Voice
export APEX_VOICE_JUDGE_CACHE=data/APEX-Voice-Runs/judge_cache.json
apex-voice results --runs-root data/APEX-Voice-Runs/runs --out results/
diff -r results/ data/APEX-Voice-Runs/reference_results/   # only the generated_at timestamp differs
```

Every judge verdict is in the cache, so no request reaches a judge API.

## Contents

```
runs/<model>/<task_id>/rep<k>/
  result.json                       # scored outcome: PTS, gate components, latency, duplex metrics
  <run_id>/events.jsonl             # timestamped event log (user plans, agent transcripts, tool calls, barge-ins)
  <run_id>/delivery.json            # agent speech-delivery QC (repetition, ASR fidelity)
  <run_id>/workspace/final_workspace.json   # the artifact(s) the agent produced
  <run_id>/workspace/versions.jsonl         # every field write, with media time and causal events
judge_cache.json                    # 10,119 cached field verdicts (judge prompt version artifact-aware-3)
reference_results/                  # `apex-voice results` output for these runs
```

Each `rep<k>` directory contains exactly one session, the scored one. Sessions abandoned after a
transient infrastructure failure and then retried are not included.

| Directory | Model | Provider |
| :--- | :--- | :--- |
| `gpt-realtime` | GPT-realtime-2.1 (`gpt-realtime-2.1`) | OpenAI |
| `grok-voice` | Grok-Voice-Think-2.0 (`grok-voice-think-fast-2.0`) | xAI |
| `gemini-live` | Gemini-3.8-Live (`gemini-3.8-live`) | Google |
| `gpt-live-1` | GPT-live-1 (`gpt-live-1`, `gpt-4o` backend) | OpenAI |
| `step-audio3` | Step-Audio3 (`stepaudio-3-realtime-preview`) | StepFun |

Campaigns ran in September 2026. The judge was `gpt-4o-mini`.

### Audio

Session audio (about 55 MB per session, roughly 100 GB in total) is not included.

- **User audio** is published pre-rendered in the task dataset
  ([`puneetUMD/APEX-Voice`](https://huggingface.co/datasets/puneetUMD/APEX-Voice), `user/audio/` in
  each task). Those clips match the user channel of these rep0 and rep1 sessions sample for sample.
  The rep2 sessions were synthesized with a different CPU thread configuration: same text, voice
  and length, with inaudible float-level differences.
- **Agent audio** came from hosted models and cannot be regenerated. The agent transcripts are in
  `events.jsonl`.

## Known issues

- **Infrastructure errors in 6 scored sessions.** These sessions logged a provider-side failure
  (`INFRA_FAILURE`) but were scored normally, all as failures. In `gpt-live-1/apexv1_088/rep2`
  the account ran out of API credits 204 s into a 238 s session. In five Step-Audio3 sessions the
  provider's speech engine returned an internal error (`apexv1_027/rep2`, `apexv1_035/rep2`,
  `apexv1_069/rep2`, `apexv1_096/rep2`, `apexv1_101/rep1`). The other repetitions of each of
  these tasks also failed, so **Reliable@3 and pass@3 are unaffected**. pass@1 is understated by
  at most 1/360 runs (GPT-live-1) and 5/360 runs (Step-Audio3). The frequent
  `response_cancel_not_active` / "no ongoing response to cancel" events are harmless: they are
  cancel requests sent on a barge-in after the response had already finished.
- **Response-start latency is clamped at 0 ms.** RSL is measured from the end of the user's
  turn. GPT-live-1 usually starts speaking before the user finishes (34.8% overlap), so its RSL
  percentiles are 0–1 ms. Use the overlap and barge-in metrics to compare its turn-taking.
- **Stop-latency rounding.** The paper's scripts pooled abandoned sessions into the barge-in
  stop-latency percentiles. `apex-voice results` uses only scored sessions, which changes
  Gemini-3.8-Live's stop p95 from 47 ms to 46 ms. Every other number matches the paper.

## Citation

```bibtex
@misc{mathur2026apexvoice,
  title  = {{APEX-Voice}: Measuring Professional Work Completion by Full-Duplex Voice Agents},
  author = {Mathur, Puneet},
  year   = {2026},
  url    = {https://github.com/puneetmathur/apex-voice}
}
```
