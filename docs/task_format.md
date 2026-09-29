# Task package format

Each of the 120 tasks is a self-contained directory, `tasks/apexv1_NNN/`. Every file is
validated against the pydantic schemas in `apex_voice/schemas/`, and `apex-voice validate`
checks that each package loads, passes the taxonomy lint, is solved by its reference policy, and
is **not** solved by an agent that does nothing.

| file | contents |
| --- | --- |
| `task.yaml` | id, version, title, profession, industry, archetype, risk tier, time budget, autonomy level and the actions that require user approval |
| `taxonomy.yaml` | labels on every taxonomy axis: professional work, interaction, workspace, knowledge, temporal dynamics, risk, difficulty |
| `initial_state.json` | initial application/world state for the environment |
| `workspace/initial_workspace.json` | artifacts present when the session starts (usually empty drafts) |
| `workspace/latent_world.json` | the latent world the artifacts are rendered from (entities, facts, policies) |
| `tools/tool_spec.yaml` | the native function tools exposed to the model: name, class (read / reversible draft / commit), parameters, handler |
| `knowledge/manifest.json` | knowledge-base documents, with gold documents and distractors marked |
| `grading/gold.yaml` | the hidden grader: terminal state predicates, required actions, forbidden actions, critical gates, and per-field artifact expectations (expected value, grader, tolerance, required flag, lifecycle minimum) |
| `reference/policy.yaml` | the oracle trajectory (agent utterances + tool calls) that achieves PTS = 1 |
| `user/user_state.yaml` | the simulated user's hidden state: goals, facts known to the user (revealed only when asked), delegation behaviour, approval behaviour |
| `user/flow.yaml` | the user's state machine: transitions from observed agent dialogue acts to user plans |
| `user/observer.yaml` | patterns that map agent speech and tool calls to dialogue acts (e.g., `ASK:employee_id`) |
| `user/events.yaml` | scripted duplex events (mid-speech corrections, back-channels): trigger, delivery, timing window, expected floor action, fields to repair |
| `user/persona.yaml` | voice selection for the user's speech |
| `user/realization_bank.jsonl` | the frozen surface realizations: one line per user plan, with text variants and the facts each variant expresses |
| `assets/inputs/artifact_source_manifest.yaml` | provenance of every artifact (all native to APEX-Voice; renderer, seed) |

## Task index

`data/tasks.jsonl` in the dataset is a flat, one-row-per-task index. It holds identifiers and
taxonomy labels and is intended for browsing and filtering on the Hugging Face Hub. The task
packages remain the source of truth for running the benchmark.

## Conditions

Task files list the capability conditions C0–C3 from the benchmark design. C0 is the
deterministic text runtime used for validation. **C2** is full-duplex speech-to-speech with
native tool calling, the condition in which all reported results were obtained, and the one
`apex-voice run` executes.
