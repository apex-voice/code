# Metrics

Every session is scored from the environment's final state, the versioned workspace, and the
timestamped event log. None of the scores depend on how the conversation "sounded".

## Production Task Score (PTS)

A binary task score, `PTS = GS ∧ PC ∧ RA ∧ WA`:

| gate | name | passes when |
| --- | --- | --- |
| **GS** | goal satisfied | the task's terminal world/application state is reached (or an accepted alternate) |
| **PC** | process compliance | no forbidden action and no critical policy gate is violated, e.g. committing without the user's approval, or leaving a field stale after a correction |
| **RA** | required actions | all required actions and evidence gathering happened (e.g., the relevant knowledge-base documents were retrieved) |
| **WA** | workspace artifact | every expected artifact exists, reached its minimum lifecycle state (e.g., `COMMITTED`), and **every required field is correct and not stale** |

A single wrong required field fails the whole deliverable. That is intentional: a benefits form
with the wrong dependent count is not "92% done".

### Field grading (two tiers)

Every gold field declares a grader: `exact`, `casefold_exact`, `numeric_tolerance`,
`normalized_date`, `set_f1`, or `semantic`. Deterministic graders run first. If the
deterministic check fails, the **semantic field judge** decides whether the agent's value
denotes the same fact as the gold value. The judge is `gpt-4o-mini` at temperature 0, with a
strict prompt that forgives dictation artifacts such as separators, case and prefixes on
identifiers, but fails on any different value, identifier, amount or decision. Judge verdicts
are cached, keyed by prompt version, field, prediction and gold value, so repeated grading is
deterministic.

## Artifact Field Accuracy (AFA)

Partial credit: the fraction of all gold artifact fields (required and optional) that pass the
two-tier grader. AFA is diagnostic. It is never folded into PTS.

## Reliability over repetitions

Each task is run `k = 3` times. Only tasks with all `k` repetitions scored count toward `n`.

| metric | definition |
| --- | --- |
| **pass@1 (mean of k)** | total passes across the k repetitions ÷ k (expected single-run passes) |
| **pass@k** | tasks passed in at least one repetition |
| **Reliable@k** | tasks passed in *all* k repetitions (Pass^k) |
| **k-of-3 distribution** | number of tasks passed exactly 0, 1, 2, 3 times |

## Latency and efficiency

| metric | definition |
| --- | --- |
| **RSL** | response-start latency: time from the end of a user utterance to the agent's first audio. Summarized per run by its median, then p50/p95 over runs. |
| **TTR** | time to resolution: media time until the workflow is complete (s), p50/p95 over runs |
| **tool calls** | native tool calls per run |
| **tool ratio** | per task: mean tool calls across repetitions ÷ tool calls in the reference (oracle) solution. Values below 1 usually mean skipped work, and values above 1 mean redundant calls. |

## Full-duplex behaviour

The simulated user fires scripted duplex events at trigger points: mid-speech corrections,
cancellations, barge-ins, interruptions, intent switches and back-channels. Each event declares
whether the agent is expected to **yield** the floor or **continue** speaking.

Floor control:

| metric | definition |
| --- | --- |
| **overlap** | share of session time during which user and agent speak at the same time |
| **yield** | share of barge-in style events (corrections, cancellations, barge-ins, urgent interruptions, intent switches) at which the agent stopped speaking |
| **stop latency** | time from barge-in onset to agent silence (ms), over genuine barge-ins (the agent yielded and was speaking, `isl_ms > 10`), p50/p95 |

Correction uptake (tasks whose user corrects a required field mid-speech):

| metric | definition |
| --- | --- |
| **AFA corrected** | field accuracy on the required fields the user corrected |
| **AFA other** | field accuracy on all other required fields |
| **gap** | AFA other − AFA corrected (percentage points) |
| **attrib** | share of WA failures in which at least one corrected field is wrong |

Per-run duplex diagnostics are stored in each `result.json` under `duplex`. They are floor-decision
accuracy (FDA), interruption stop latency (ISL), overlap-correction capture (OCCR) and
interruption recovery accuracy (IRA).

## Taxonomy slices

`apex-voice results` also reports `success% (AFA%)` averaged over repetitions for each label of
the task taxonomy: work archetype, industry, APPROVE-gating, artifact class, autonomy level,
knowledge burden, tool burden, user behaviour profile, required-fact count, and risk tier.
