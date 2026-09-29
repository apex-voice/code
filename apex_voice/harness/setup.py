"""Shared run-setup helpers used by both the C0 text runner and the C1/C2 voice runner."""

from __future__ import annotations

import json
from typing import Any

from apex_voice.artifacts.workspace import Workspace
from apex_voice.environments.knowledge import Document, KnowledgeBase
from apex_voice.environments.tools import ToolRuntime
from apex_voice.schemas.artifact import ArtifactSchema, LifecycleState
from apex_voice.tasks.loader import LoadedTask
from apex_voice.user_sim.compiler import RealizationCompiler


def build_workspace(lt: LoadedTask) -> Workspace:
    ws = Workspace()
    for aid, adef in lt.initial_workspace.get("artifacts", {}).items():
        schema = ArtifactSchema.model_validate(adef["schema"]) if adef.get("schema") else None
        ws.create(
            artifact_id=aid,
            artifact_type=adef.get("artifact_type", "GENERIC"),
            schema=schema,
            initial_fields=adef.get("fields", {}),
            lifecycle=LifecycleState(adef.get("lifecycle", "EMPTY")),
        )
    return ws


def build_knowledge(lt: LoadedTask) -> KnowledgeBase:
    kb = KnowledgeBase()
    manifest = lt.root / lt.spec.knowledge_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        for d in data.get("documents", []):
            text = d.get("text", "")
            if "path" in d:
                fp = lt.root / lt.spec.knowledge_dir / d["path"]
                if fp.exists():
                    text = fp.read_text()
            kb.add(
                Document(
                    d["doc_id"],
                    d.get("title", d["doc_id"]),
                    text,
                    gold=d.get("gold", False),
                    distractor=d.get("distractor", False),
                )
            )
    return kb


def agent_tool_schemas(rt: ToolRuntime) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "params": [{"name": p.name, "type": p.type, "required": p.required} for p in t.params],
        }
        for t in rt.available_tools()
    ]


def ensure_realization_bank(lt: LoadedTask) -> None:
    if lt.realization_bank.plan_ids():
        return
    compiler = RealizationCompiler()
    inputs = compiler.enumerate_plan_inputs(lt.flow, lt.events, lt.user_state)
    bank, _ = compiler.compile_all(inputs)
    lt.realization_bank = bank


def write_workspace_outputs(run_root, workspace) -> None:
    """Persist final workspace + versioned mutation log for a run."""
    import json
    from pathlib import Path

    d = Path(run_root) / "workspace"
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_workspace.json").write_text(json.dumps(workspace.snapshot(), indent=2, default=str))
    with (d / "versions.jsonl").open("w") as fh:
        for rec in workspace.history_records():
            fh.write(json.dumps(rec, default=str) + "\n")


__all__ = [
    "build_workspace",
    "build_knowledge",
    "agent_tool_schemas",
    "ensure_realization_bank",
    "write_workspace_outputs",
]
