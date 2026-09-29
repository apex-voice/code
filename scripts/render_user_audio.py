#!/usr/bin/env python3
"""Pre-render the simulated user's speech for every task with the pinned Kokoro-82M setup.

For each task package this writes ``user/audio/``:

    <plan>_<variant>.flac     one clip per realization-bank variant (24 kHz mono, lossless PCM16)
    _closing_<i>.flac         the runner's fixed sign-off lines, in the task's voice
    _nudge.flac               the runner's fixed "go on?" nudge, in the task's voice
    index.jsonl               text -> clip (+ voice, sample count, PCM16 sha256, word timings)
    meta.json                 renderer versions (Kokoro revision, torch, kokoro, misaki)

The FLACs hold exactly the PCM16 the realtime runner streams to the model, so
``apex-voice run --user-audio prerendered`` replays the published runs' user audio bit-for-bit.
Live Kokoro output is only bit-exact under the same torch build *and* thread configuration, so render
with the paper environment (``pip install -c constraints/paper.txt``) and pin the thread count with
``OMP_NUM_THREADS=1``, which makes the output independent of the host's core count:

    OMP_NUM_THREADS=1 python scripts/render_user_audio.py --data path/to/dataset --workers 48

Each ``user/audio/meta.json`` records the renderer versions actually used.

Run this *after* ``scripts/export_hf_dataset.py`` (which rewrites ``tasks/``). When ``--data`` is a
dataset root, ``SHA256SUMS`` is regenerated to cover the clips.

Usage:
    python scripts/render_user_audio.py --data path/to/dataset [--workers 6] [--only apexv1_001]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _clip_name(variant_id: str) -> str:
    return _SAFE.sub("_", variant_id).strip("_") + ".flac"


def _utterances(lt) -> list[dict]:
    """Every text the realtime runner can speak for this task, with a stable clip file name."""
    from apex_voice.harness.realtime_runner import _CLOSINGS, _NUDGE

    out: list[dict] = []
    for pid in lt.realization_bank.plan_ids():
        pv = lt.realization_bank.get(pid)
        for v in pv.variants:
            out.append(
                {
                    "file": _clip_name(v.variant_id),
                    "kind": "variant",
                    "user_plan_id": pid,
                    "variant_id": v.variant_id,
                    "text": v.text,
                }
            )
    for i, t in enumerate(_CLOSINGS):
        out.append({"file": f"_closing_{i}.flac", "kind": "closing", "text": t})
    out.append({"file": "_nudge.flac", "kind": "nudge", "text": _NUDGE})
    names = [u["file"] for u in out]
    if len(set(names)) != len(names):
        raise RuntimeError(f"{lt.spec.id}: clip file names collide")
    return out


def _versions() -> dict:
    from importlib.metadata import version

    from apex_voice.user_sim.kokoro_compiler import KOKORO_REPO_ID, KOKORO_REVISION

    return {
        "engine": "kokoro",
        "repo_id": KOKORO_REPO_ID,
        "revision": KOKORO_REVISION,
        "sample_rate": 24000,
        "encoding": "FLAC PCM_16 (runner PCM16: clip(x) * 32767 truncated)",
        **{pkg: version(pkg) for pkg in ("kokoro", "misaki", "torch", "numpy")},
    }


def render_task(task_dir: str, force: bool = False) -> tuple[str, int, float]:
    import soundfile as sf

    from apex_voice.tasks import load_task
    from apex_voice.user_sim.kokoro_compiler import KokoroSpeechCompiler
    from apex_voice.user_sim.prerendered import AUDIO_DIR, INDEX_FILE, SAMPLE_RATE, f32_to_pcm16

    lt = load_task(task_dir)
    out = Path(task_dir) / AUDIO_DIR
    if (out / INDEX_FILE).exists() and not force:
        return lt.spec.id, 0, 0.0
    out.mkdir(parents=True, exist_ok=True)
    comp = KokoroSpeechCompiler(sample_rate=SAMPLE_RATE)
    rows, total_s = [], 0.0
    for u in _utterances(lt):
        cs = comp.compile(u["text"], lt.persona or {}, None, None)
        q = f32_to_pcm16(cs.waveform.astype("float32"))
        sf.write(str(out / u["file"]), q, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        rows.append(
            {
                **u,
                "voice": cs.metadata["voice"],
                "sample_rate": SAMPLE_RATE,
                "num_samples": int(len(q)),
                "pcm16_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
                "word_timestamps": cs.word_timestamps,
            }
        )
        total_s += len(q) / SAMPLE_RATE
    (out / "meta.json").write_text(json.dumps(_versions(), indent=2) + "\n")
    with (out / INDEX_FILE).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return lt.spec.id, len(rows), total_s


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--data", required=True, type=Path, help="dataset root (with tasks/) or a tasks directory"
    )
    ap.add_argument("--only", nargs="*", help="restrict to these task ids")
    ap.add_argument("--workers", type=int, default=4, help="parallel processes (one task each)")
    ap.add_argument("--force", action="store_true", help="re-render tasks that already have clips")
    a = ap.parse_args()

    tasks_root = a.data / "tasks" if (a.data / "tasks").is_dir() else a.data
    dirs = sorted(d for d in tasks_root.iterdir() if (d / "task.yaml").exists())
    if a.only:
        dirs = [d for d in dirs if d.name in set(a.only)]
    if not dirs:
        sys.exit(f"no task packages found under {tasks_root}")

    n_clips, hours = 0, 0.0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(render_task, str(d), a.force) for d in dirs]
        for i, f in enumerate(as_completed(futs), 1):
            tid, n, secs = f.result()
            n_clips, hours = n_clips + n, hours + secs / 3600
            print(f"[{i}/{len(dirs)}] {tid}: {n} clips, {secs / 60:.1f} min", flush=True)
    print(f"rendered {n_clips} clips ({hours:.2f} h) for {len(dirs)} tasks")

    if tasks_root != a.data:
        files = sorted(p for p in [*tasks_root.rglob("*"), *(a.data / "data").rglob("*")] if p.is_file())
        (a.data / "SHA256SUMS").write_text(
            "".join(f"{_sha256(p)}  {p.relative_to(a.data).as_posix()}\n" for p in files)
        )
        print(f"wrote SHA256SUMS ({len(files)} files)")


if __name__ == "__main__":
    main()
