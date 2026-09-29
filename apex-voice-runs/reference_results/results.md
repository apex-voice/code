# APEX-Voice results

Generated 2026-09-29 09:13 UTC by apex-voice 1.0.0. 120 tasks, condition C2 (full-duplex), 3 repetitions per task. Counts are out of the tasks with all 3 repetitions scored (`n`).

## Leaderboard

| Model | n | pass@1 (mean of 3) | pass@1 % | pass@3 | pass@3 % | Reliable@3 | Reliable@3 % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Grok-Voice-Think-2.0 | 120 | 27.7 | 23.1 | 50 | 41.7 | 6 | 5.0 |
| GPT-realtime-2.1 | 120 | 28.3 | 23.6 | 44 | 36.7 | 13 | 10.8 |
| Gemini-3.8-Live | 120 | 16.3 | 13.6 | 34 | 28.3 | 2 | 1.7 |
| GPT-live-1 | 120 | 10.7 | 8.9 | 19 | 15.8 | 3 | 2.5 |
| Step-Audio3 | 120 | 3.0 | 2.5 | 6 | 5.0 | 1 | 0.8 |

## Per-repetition pass@1 and k-of-3 distribution

| Model | rep0 | rep1 | rep2 | 0/3 | 1/3 | 2/3 | 3/3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Grok-Voice-Think-2.0 | 24 | 26 | 33 | 70 | 23 | 21 | 6 |
| GPT-realtime-2.1 | 28 | 27 | 30 | 76 | 16 | 15 | 13 |
| Gemini-3.8-Live | 19 | 14 | 16 | 86 | 21 | 11 | 2 |
| GPT-live-1 | 13 | 7 | 12 | 101 | 9 | 7 | 3 |
| Step-Audio3 | 3 | 3 | 3 | 114 | 4 | 1 | 1 |

## PTS gate pass rates (%, all runs)

GS = goal satisfied, PC = process compliance, RA = required actions, WA = workspace artifact. PTS = GS ∧ PC ∧ RA ∧ WA.

| Model | runs | GS | PC | RA | WA |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Grok-Voice-Think-2.0 | 360 | 99.4 | 85.3 | 85.8 | 27.8 |
| GPT-realtime-2.1 | 360 | 81.1 | 89.2 | 61.4 | 35.3 |
| Gemini-3.8-Live | 360 | 84.4 | 74.2 | 49.7 | 24.4 |
| GPT-live-1 | 360 | 88.6 | 91.9 | 65.8 | 11.9 |
| Step-Audio3 | 360 | 34.4 | 56.4 | 56.9 | 4.2 |

## Latency and tool-use efficiency

RSL = response-start latency (median per run, ms); TTR = time to resolution (s); tool ratio = mean tool calls per task relative to the oracle trajectory.

| Model | RSL p50 | RSL p95 | TTR p50 | TTR p95 | tool calls | tool ratio (mean) | tool ratio (p50) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Grok-Voice-Think-2.0 | 598 | 727 | 206.3 | 315.3 | 19.6 | 1.51 | 1.40 |
| GPT-realtime-2.1 | 1242 | 2301 | 222.3 | 332.3 | 12.7 | 0.98 | 0.98 |
| Gemini-3.8-Live | 607 | 851 | 223.2 | 329.3 | 14.2 | 1.09 | 1.08 |
| GPT-live-1 | 0 | 1 | 206.6 | 330.1 | 12.1 | 0.94 | 0.90 |
| Step-Audio3 | 1107 | 1446 | 233.0 | 370.4 | 4.9 | 0.38 | 0.36 |

## Full-duplex: floor control and correction uptake

Overlap = simultaneous-speech share of the session; yield = barge-ins at which the agent stopped speaking; stop latency over genuine barge-ins. AFA = Artifact Field Accuracy on user-corrected vs. other required fields; attrib = share of WA failures with a wrong corrected field.

| Model | overlap % | yield % | stop p50 (ms) | stop p95 (ms) | AFA corrected % | AFA other % | gap (pp) | attrib % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Grok-Voice-Think-2.0 | 1.3 | 100.0 | 28 | 64 | 72.7 | 92.6 | +20.0 | 71.0 |
| GPT-realtime-2.1 | 2.5 | 99.8 | 156 | 294 | 74.0 | 95.1 | +21.1 | 73.8 |
| Gemini-3.8-Live | 1.8 | 100.0 | 20 | 46 | 64.3 | 88.2 | +23.8 | 76.9 |
| GPT-live-1 | 34.8 | 95.6 | 912 | 1424 | 49.4 | 78.6 | +29.2 | 81.5 |
| Step-Audio3 | 4.6 | 100.0 | 59 | 105 | 38.4 | 75.2 | +36.8 | 89.3 |

## Taxonomy slices

Each cell is `success% (AFA%)` averaged over 3 repetitions.

### Work archetype

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| COORDINATE | 20 | 22% (94%) | 20% (92%) | 12% (87%) | 23% (84%) | 3% (76%) |
| NEGOTIATE | 20 | 15% (86%) | 10% (88%) | 3% (79%) | 3% (75%) | 0% (63%) |
| ADVISE | 10 | 27% (83%) | 43% (96%) | 13% (90%) | 17% (79%) | 3% (74%) |
| DISCOVERY | 10 | 23% (91%) | 23% (96%) | 13% (86%) | 7% (65%) | 0% (88%) |
| FACILITATE | 10 | 43% (91%) | 47% (95%) | 27% (87%) | 23% (69%) | 7% (82%) |
| FORM_FILL | 10 | 13% (88%) | 7% (86%) | 0% (78%) | 3% (61%) | 0% (74%) |
| INSPECT | 10 | 30% (87%) | 13% (90%) | 3% (84%) | 0% (75%) | 0% (71%) |
| INTAKE | 10 | 20% (87%) | 27% (91%) | 20% (88%) | 3% (71%) | 0% (55%) |
| INTERVIEW | 10 | 27% (89%) | 47% (95%) | 27% (86%) | 7% (64%) | 10% (69%) |
| TROUBLESHOOT | 10 | 13% (87%) | 17% (88%) | 13% (83%) | 3% (59%) | 3% (78%) |

### Industry / setting

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| software_saas | 43 | 26% (90%) | 27% (94%) | 14% (86%) | 12% (72%) | 2% (76%) |
| horizontal_enterprise | 30 | 22% (90%) | 23% (91%) | 12% (85%) | 16% (78%) | 1% (75%) |
| manufacturing_field_ops | 25 | 20% (88%) | 9% (88%) | 7% (83%) | 1% (66%) | 1% (66%) |
| professional_services | 10 | 20% (87%) | 30% (89%) | 10% (79%) | 3% (63%) | 10% (76%) |
| workplace_hr | 8 | 21% (86%) | 33% (91%) | 17% (84%) | 12% (73%) | 4% (70%) |
| healthcare | 3 | 22% (87%) | 56% (92%) | 33% (82%) | 0% (75%) | 0% (61%) |
| insurance | 1 | 0% (88%) | 0% (100%) | 0% (100%) | 0% (85%) | 0% (24%) |

### Delegation: APPROVE-gating

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| not APPROVE-gated | 91 | 25% (89%) | 26% (92%) | 15% (86%) | 10% (71%) | 3% (75%) |
| APPROVE-gated | 29 | 14% (88%) | 15% (88%) | 2% (80%) | 9% (72%) | 0% (64%) |

### Primary artifact class

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEGOTIATION_RECORD | 20 | 15% (86%) | 10% (88%) | 3% (79%) | 3% (75%) | 0% (63%) |
| CASE_RECORD | 10 | 20% (87%) | 27% (91%) | 20% (88%) | 3% (71%) | 0% (55%) |
| CRM_RECORD | 10 | 23% (91%) | 23% (96%) | 13% (86%) | 7% (65%) | 0% (88%) |
| EVIDENCE_MATRIX | 10 | 27% (89%) | 47% (95%) | 27% (86%) | 7% (64%) | 10% (69%) |
| MEMO_REPORT | 10 | 27% (83%) | 43% (96%) | 13% (90%) | 17% (79%) | 3% (74%) |
| PLAN_CHECKLIST | 10 | 43% (91%) | 47% (95%) | 27% (87%) | 23% (69%) | 7% (82%) |
| SCHEDULE | 10 | 27% (94%) | 23% (93%) | 13% (86%) | 27% (84%) | 7% (80%) |
| STRUCTURED_FORM | 10 | 13% (88%) | 7% (86%) | 0% (78%) | 3% (61%) | 0% (74%) |
| TICKET | 10 | 13% (87%) | 17% (88%) | 13% (83%) | 3% (59%) | 3% (78%) |
| TIMELINE | 10 | 17% (94%) | 17% (92%) | 10% (88%) | 20% (84%) | 0% (72%) |
| WORK_ORDER | 10 | 30% (87%) | 13% (90%) | 3% (84%) | 0% (75%) | 0% (71%) |

### Autonomy / commit level

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0_PREPARE_ONLY | 51 | 29% (89%) | 34% (94%) | 20% (87%) | 9% (70%) | 5% (75%) |
| A1_DRAFT_CONFIRM | 29 | 18% (89%) | 14% (91%) | 8% (86%) | 14% (74%) | 2% (77%) |
| A2_LOW_RISK_EXECUTE | 16 | 17% (89%) | 21% (89%) | 10% (86%) | 6% (73%) | 0% (67%) |
| A3_APPROVAL_GATED_COMMIT | 24 | 17% (87%) | 15% (88%) | 1% (78%) | 8% (70%) | 0% (65%) |

### Knowledge burden

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| K0_NONE | 44 | 31% (90%) | 39% (93%) | 26% (86%) | 15% (71%) | 5% (77%) |
| K1_SUPPLIED | 6 | 6% (85%) | 6% (88%) | 0% (76%) | 0% (67%) | 0% (73%) |
| K2_SEARCH_SMALL | 51 | 19% (88%) | 12% (90%) | 3% (84%) | 6% (72%) | 1% (68%) |
| K3_MULTI_DOC_POLICY | 19 | 18% (86%) | 25% (93%) | 11% (85%) | 11% (75%) | 2% (72%) |

### Tool burden

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| T1_LIGHT | 44 | 31% (90%) | 39% (93%) | 26% (86%) | 15% (71%) | 5% (77%) |
| T2_MODERATE | 76 | 18% (88%) | 15% (90%) | 4% (83%) | 7% (72%) | 1% (69%) |

### User behaviour profile

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| COOPERATIVE_STANDARD | 24 | 33% (92%) | 28% (93%) | 12% (84%) | 10% (72%) | 1% (71%) |
| CORRECTION_PRONE | 24 | 18% (88%) | 19% (89%) | 11% (86%) | 8% (72%) | 0% (76%) |
| AMBIGUOUS_UNDERSPECIFIED | 12 | 17% (87%) | 31% (93%) | 14% (84%) | 14% (75%) | 0% (69%) |
| DISTRACTED_TIME_PRESSURED | 12 | 11% (85%) | 8% (90%) | 19% (90%) | 8% (71%) | 3% (77%) |
| DOMAIN_EXPERT | 12 | 28% (87%) | 17% (91%) | 11% (82%) | 11% (61%) | 3% (74%) |
| LOW_TECH_EXPERTISE | 12 | 31% (89%) | 31% (95%) | 11% (84%) | 14% (77%) | 11% (59%) |
| NOVICE_UNCERTAIN | 12 | 19% (92%) | 36% (93%) | 14% (86%) | 14% (76%) | 0% (75%) |
| VERBOSE_NARRATIVE | 12 | 17% (86%) | 19% (88%) | 6% (80%) | 0% (68%) | 6% (74%) |

### Difficulty: required fact count

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 fields | 24 | 26% (88%) | 31% (93%) | 12% (85%) | 14% (73%) | 4% (68%) |
| 10 fields | 74 | 22% (89%) | 17% (91%) | 9% (84%) | 10% (74%) | 1% (76%) |
| 11-13 fields | 22 | 21% (89%) | 38% (92%) | 21% (85%) | 3% (64%) | 5% (66%) |

### Risk / safety tier

| Label | n | Grok-Voice-Think-2.0 | GPT-realtime-2.1 | Gemini-3.8-Live | GPT-live-1 | Step-Audio3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0_ROUTINE | 63 | 25% (90%) | 21% (92%) | 12% (86%) | 13% (73%) | 3% (78%) |
| R1_SENSITIVE_DATA_SIM | 25 | 24% (88%) | 35% (93%) | 20% (87%) | 5% (67%) | 5% (68%) |
| R2_CONSEQUENTIAL_ACTION | 22 | 15% (86%) | 15% (88%) | 0% (78%) | 5% (69%) | 0% (65%) |
| R3_SPECIAL_REVIEW | 10 | 20% (88%) | 33% (92%) | 20% (85%) | 10% (79%) | 0% (64%) |
