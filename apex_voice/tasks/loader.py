"""Task package loader / validator.

Loads a task directory into a fully-resolved :class:`LoadedTask` and
validates cross-file references and invariants. Invalid packages fail with actionable errors (CI:
"invalid task fixtures fail with actionable errors").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from apex_voice.adapters.base import AgentTurn, ToolCall
from apex_voice.schemas.artifact import LatentWorld
from apex_voice.schemas.flow import EventProgram, FlowSpec
from apex_voice.schemas.grading import GoldSpec
from apex_voice.schemas.task import TaskSpec
from apex_voice.schemas.taxonomy import TaxonomySpec
from apex_voice.schemas.tool import ToolSet
from apex_voice.schemas.user import UserState
from apex_voice.user_sim.agent_observer import ActPattern
from apex_voice.user_sim.asset_bank import RealizationBank


class TaskLoadError(Exception):
    pass


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) if path.exists() else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


@dataclass
class ObserverConfig:
    patterns: list[ActPattern] = field(default_factory=list)
    tool_act_map: dict[str, str] = field(default_factory=dict)


@dataclass
class LoadedTask:
    root: Path
    spec: TaskSpec
    taxonomy: TaxonomySpec
    initial_state: dict[str, Any]
    user_state: UserState
    flow: FlowSpec
    events: EventProgram
    persona: dict[str, Any]
    toolset: ToolSet
    latent_world: LatentWorld | None
    initial_workspace: dict[str, Any]
    gold: GoldSpec
    observer: ObserverConfig
    realization_bank: RealizationBank
    reference_turns: list[AgentTurn] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        return self.spec.id


def _parse_reference_turns(data: Any) -> list[AgentTurn]:
    turns: list[AgentTurn] = []
    for t in (data or {}).get("turns", []):
        turns.append(
            AgentTurn(
                text=t.get("text", ""),
                tool_calls=[ToolCall(c["name"], c.get("args", {})) for c in t.get("tool_calls", [])],
                end_call=t.get("end_call", False),
            )
        )
    return turns


def load_task(task_dir: str | Path) -> LoadedTask:
    root = Path(task_dir)
    if not root.is_dir():
        raise TaskLoadError(f"task directory not found: {root}")

    spec_data = _read_yaml(root / "task.yaml")
    if spec_data is None:
        raise TaskLoadError(f"missing task.yaml in {root}")
    try:
        spec = TaskSpec.model_validate(spec_data)
    except Exception as e:  # noqa: BLE001
        raise TaskLoadError(f"invalid task.yaml: {e}") from e

    def req(rel: str, kind: str) -> Path:
        p = root / rel
        return p

    # taxonomy
    tax_data = _read_yaml(root / spec.taxonomy)
    if tax_data is None:
        raise TaskLoadError(f"missing taxonomy file {spec.taxonomy}")
    taxonomy = TaxonomySpec.model_validate(tax_data)

    initial_state = _read_json(root / spec.initial_state) or {}
    user_state = UserState.model_validate(_read_yaml(root / spec.user_state) or {})
    flow = FlowSpec.model_validate(_read_yaml(root / spec.user_program) or {"states": {}})
    events = EventProgram.model_validate(_read_yaml(root / spec.event_program) or {"events": []})
    persona = _read_yaml(root / spec.persona) or {}

    tool_data = _read_yaml(root / spec.tools)
    toolset = ToolSet.model_validate(tool_data or {"tools": []})

    lw_data = _read_json(root / spec.latent_world)
    latent_world = LatentWorld.model_validate(lw_data) if lw_data else None
    initial_workspace = _read_json(root / spec.initial_workspace) or {}

    gold_data = _read_yaml(root / spec.gold_dir / "gold.yaml")
    if gold_data is None:
        raise TaskLoadError(f"missing grading/gold.yaml in {root}")
    gold = GoldSpec.model_validate(gold_data)

    obs_data = _read_yaml(root / "user" / "observer.yaml") or {}
    observer = ObserverConfig(
        patterns=[
            ActPattern(act=p["act"], pattern=p["pattern"], field=p.get("field"))
            for p in obs_data.get("patterns", [])
        ],
        tool_act_map=obs_data.get("tool_act_map", {}),
    )

    bank = RealizationBank.load_jsonl(root / spec.realization_bank)
    reference_turns = _parse_reference_turns(_read_yaml(root / "reference" / "policy.yaml"))

    lt = LoadedTask(
        root=root,
        spec=spec,
        taxonomy=taxonomy,
        initial_state=initial_state,
        user_state=user_state,
        flow=flow,
        events=events,
        persona=persona,
        toolset=toolset,
        latent_world=latent_world,
        initial_workspace=initial_workspace,
        gold=gold,
        observer=observer,
        realization_bank=bank,
        reference_turns=reference_turns,
    )
    _validate(lt)
    return lt


def _validate(lt: LoadedTask) -> None:
    errors: list[str] = []
    if lt.taxonomy.task_id != lt.spec.id:
        errors.append(f"taxonomy.task_id '{lt.taxonomy.task_id}' != task.id '{lt.spec.id}'")
    # Tool handler bindings exist.
    from apex_voice.environments.tools import BUILTIN_HANDLERS

    for t in lt.toolset.tools:
        if t.handler and t.handler not in BUILTIN_HANDLERS:
            errors.append(f"tool '{t.name}' binds unknown built-in handler '{t.handler}'")
    # Gold predicates reference artifacts that will exist.
    ws_ids = set(lt.initial_workspace.get("artifacts", {}).keys())
    for exp in lt.gold.artifact_expectations:
        if exp.artifact_id not in ws_ids:
            errors.append(f"gold artifact_expectation references unknown artifact '{exp.artifact_id}'")
    # Declared load levels vs contents.
    if lt.spec.tool_load == "none" and lt.toolset.tools:
        errors.append("tool_load=none but tools are defined")
    if errors:
        raise TaskLoadError(f"task '{lt.spec.id}' failed validation:\n  - " + "\n  - ".join(errors))


__all__ = ["LoadedTask", "load_task", "TaskLoadError", "ObserverConfig"]
