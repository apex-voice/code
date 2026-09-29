# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026

Initial public release.

- 120-task APEX-Voice dataset (benchmark `v1.0`), distributed on the Hugging Face Hub.
- Full-duplex realtime runner with a frozen synthetic user (Kokoro speech), an instrumented tool
  environment, and a commit/approval guard.
- Production Task Score (PTS), Artifact Field Accuracy (AFA), duplex and latency metrics, and the
  semantic field judge (`artifact-aware-3`).
- Adapters for GPT-realtime-2.1, Grok-Voice-Think-2.0, Gemini-3.8-Live, GPT-live-1 and Step-Audio3.
- `apex-voice` CLI: `download`, `validate`, `run` (resumable campaigns), `results`, `models`.
- Kokoro-82M pinned to the Hugging Face revision used for the paper (synthesis on CPU by
  default), plus `constraints/paper.txt` pinning the paper environment.
- Pre-rendered simulated-user audio in every task package (`user/audio/`), replayed by
  `apex-voice run --user-audio auto|prerendered` so all machines stream identical PCM;
  `scripts/render_user_audio.py` regenerates it.
- Companion dataset `APEX-Voice-Runs`: the paper's 1,800 scored sessions and the judge cache,
  so the paper tables can be regenerated offline.
