#!/usr/bin/env python3
"""Export a directory of APEX-Voice task packages into the Hugging Face dataset layout.

Output layout::

    <out>/
      tasks/<task_id>/...        # the task packages, copied verbatim (minus excluded files)
      data/tasks.jsonl           # one row of flat, viewer-friendly metadata per task
      SHA256SUMS                 # checksums of every file under tasks/ and data/

The dataset card (``README.md``) and ``LICENSE`` are maintained by hand in the dataset repo and are
not touched by this script. Every exported task is validated (loads, lints, oracle passes, no-op
fails) before anything is written.

Usage:
    python scripts/export_hf_dataset.py --src path/to/tasks --out path/to/dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

# Files present in authoring checkouts that are not part of the released task format.
EXCLUDE = {"qc_report.json"}


def _yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) or {}


def index_row(task_dir: Path) -> dict:
    from apex_voice.tasks import load_task

    lt = load_task(task_dir)
    tax = _yaml(task_dir / "taxonomy.yaml")
    pw, it = tax.get("professional_work", {}), tax.get("interaction", {})
    ws, kn = tax.get("workspace", {}), tax.get("knowledge", {})
    events = _yaml(task_dir / "user" / "events.yaml").get("events", [])
    policy = _yaml(task_dir / "reference" / "policy.yaml")
    required = [
        f"{ae.artifact_id}.{fe.field}"
        for ae in lt.gold.artifact_expectations
        for fe in ae.fields
        if fe.required
    ]
    n_plans = sum(
        1 for line in (task_dir / "user" / "realization_bank.jsonl").read_text().splitlines() if line.strip()
    )
    return {
        "task_id": lt.spec.id,
        "version": lt.spec.version,
        "title": lt.spec.title,
        "profession": lt.spec.profession,
        "industry": pw.get("industry_setting"),
        "economic_function": pw.get("economic_function"),
        "archetype": pw.get("primary_archetype"),
        "delegation_patterns": list(it.get("delegation_patterns") or []),
        "user_profile": it.get("user_profile"),
        "temporal_dynamics": it.get("temporal_dynamics"),
        "primary_artifact": ws.get("primary_artifact"),
        "autonomy_level": ws.get("autonomy_level"),
        "approval_required_for": list(lt.spec.autonomy.approval_required_for),
        "knowledge_burden": kn.get("burden"),
        "tool_burden": kn.get("tool_burden"),
        "risk_tier": tax.get("risk_tier"),
        "required_fact_count": (tax.get("difficulty") or {}).get("required_fact_count"),
        "time_budget_s": lt.spec.time_budget_s,
        "tools": [t.name for t in lt.toolset.tools],
        "duplex_event_types": [str(e.get("type")) for e in events],
        "required_fields": required,
        "oracle_tool_calls": sum(len(t.get("tool_calls") or []) for t in policy.get("turns", [])),
        "num_user_plans": n_plans,
        "path": f"tasks/{task_dir.name}",
    }


def validate(task_dir: Path) -> None:
    from apex_voice.validation import validate_task

    rep = validate_task(task_dir)
    if not rep.ok:
        raise SystemExit(f"validation failed for {task_dir.name}: {rep.problems}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="directory containing task packages")
    ap.add_argument("--out", required=True, type=Path, help="dataset root to (re)write")
    ap.add_argument("--skip-validation", action="store_true")
    a = ap.parse_args()

    task_dirs = sorted(d for d in a.src.iterdir() if (d / "task.yaml").exists())
    if not task_dirs:
        sys.exit(f"no task packages found under {a.src}")
    if not a.skip_validation:
        for d in task_dirs:
            validate(d)
        print(f"validated {len(task_dirs)} tasks")

    tasks_out, data_out = a.out / "tasks", a.out / "data"
    shutil.rmtree(tasks_out, ignore_errors=True)
    shutil.rmtree(data_out, ignore_errors=True)
    data_out.mkdir(parents=True)
    for d in task_dirs:
        shutil.copytree(d, tasks_out / d.name, ignore=shutil.ignore_patterns(*EXCLUDE))

    rows = [index_row(tasks_out / d.name) for d in task_dirs]
    with (data_out / "tasks.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    files = sorted(p for p in [*tasks_out.rglob("*"), *data_out.rglob("*")] if p.is_file())
    (a.out / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(a.out).as_posix()}\n" for p in files)
    )
    print(f"exported {len(rows)} tasks, {len(files)} files -> {a.out}")


if __name__ == "__main__":
    main()
