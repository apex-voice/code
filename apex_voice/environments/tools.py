"""Tool runtime.

Executes typed tool calls against the workspace/state/knowledge, injecting media-time latency and
routing consequential commits through the :class:`CommitGuard`. Tools bind to a *built-in handler*
by name (declarative, YAML-authorable) or to a custom Python callable registered at runtime.

Built-in handlers cover the operations the benchmark tasks need:

- ``kb_search``          -- retrieval over the task knowledge base.
- ``get_entity``         -- read-only latent/world entity lookup.
- ``update_artifact``    -- reversible draft write of one or more artifact fields.
- ``set_lifecycle``      -- advance an artifact's lifecycle state.
- ``request_approval``   -- surface a scoped pending action (observed by the flow engine).
- ``commit``             -- irreversible commit; authorized by the CommitGuard at media time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from apex_voice.artifacts.workspace import Workspace
from apex_voice.determinism import seeded_rng
from apex_voice.environments.commit_guard import AuthDecision, CommitGuard
from apex_voice.environments.knowledge import KnowledgeBase
from apex_voice.environments.state_store import StateStore
from apex_voice.schemas.artifact import LifecycleState
from apex_voice.schemas.tool import ToolSet, ToolSpec


@dataclass
class ToolCallResult:
    ok: bool
    tool: str
    args: dict[str, Any]
    result: Any
    latency_ms: int
    error: str | None = None
    auth_decision: str | None = None


# A handler receives the runtime, the tool spec, the call args, and media time.
Handler = Callable[["ToolRuntime", ToolSpec, dict[str, Any], int], Any]


class ToolError(Exception):
    pass


class ToolRuntime:
    def __init__(
        self,
        toolset: ToolSet,
        workspace: Workspace,
        state: StateStore,
        commit_guard: CommitGuard,
        knowledge: KnowledgeBase | None = None,
        seed: int = 1,
        latency_stress: float = 1.0,
    ) -> None:
        self.toolset = toolset
        self.workspace = workspace
        self.state = state
        self.commit_guard = commit_guard
        self.knowledge = knowledge or KnowledgeBase()
        self.seed = seed
        self.latency_stress = latency_stress
        self.call_log: list[ToolCallResult] = []
        self._custom: dict[str, Handler] = {}
        self._call_index = 0

    def register(self, name: str, handler: Handler) -> None:
        self._custom[name] = handler

    def available_tools(self, include_discoverable: bool = False) -> list[ToolSpec]:
        return [t for t in self.toolset.tools if include_discoverable or not t.discoverable]

    def _latency(self, spec: ToolSpec) -> int:
        rng = seeded_rng(self.seed, spec.name, self._call_index)
        jitter = rng.randint(-spec.latency.jitter_ms, spec.latency.jitter_ms) if spec.latency.jitter_ms else 0
        base = spec.latency.base_ms + jitter
        return max(0, int(base * spec.latency.stress_multiplier * self.latency_stress))

    def call(self, name: str, args: dict[str, Any], media_time_ms: int = 0) -> ToolCallResult:
        spec = self.toolset.by_name(name)
        if spec is None:
            res = ToolCallResult(False, name, args, None, 0, error=f"unknown tool '{name}'")
            self.call_log.append(res)
            return res

        latency = self._latency(spec)
        handler = self._custom.get(name) or BUILTIN_HANDLERS.get(spec.handler or "")
        if handler is None:
            res = ToolCallResult(
                False, name, args, None, latency, error=f"no handler bound for tool '{name}'"
            )
            self.call_log.append(res)
            self._call_index += 1
            return res
        try:
            out = handler(self, spec, args, media_time_ms)
            auth = out.pop("_auth_decision", None) if isinstance(out, dict) else None
            ok = True if auth is None else (auth == AuthDecision.AUTHORIZED.value)
            res = ToolCallResult(ok, name, args, out, latency, auth_decision=auth)
        except ToolError as e:
            res = ToolCallResult(False, name, args, None, latency, error=str(e))
        self.call_log.append(res)
        self._call_index += 1
        return res


# ---- built-in handlers ----------------------------------------------------------------


def _h_kb_search(rt: ToolRuntime, spec: ToolSpec, args: dict[str, Any], t: int) -> Any:
    return {"hits": rt.knowledge.search(args.get("query", ""), int(args.get("top_k", 3)))}


def _h_get_entity(rt: ToolRuntime, spec: ToolSpec, args: dict[str, Any], t: int) -> Any:
    key = spec.effect.get("state_path") or args.get("path") or args.get("entity_id", "")
    return {"entity": rt.state.get(key)}


def _h_update_artifact(rt: ToolRuntime, spec: ToolSpec, args: dict[str, Any], t: int) -> Any:
    artifact_id = args.get("artifact_id") or spec.effect.get("artifact_id")
    if not artifact_id or rt.workspace.get(artifact_id) is None:
        raise ToolError(f"update_artifact: unknown artifact '{artifact_id}'")
    fields = args.get("fields", {})
    if not isinstance(fields, dict):
        raise ToolError("update_artifact: 'fields' must be an object")
    causal = args.get("causal_events", [])
    written = []
    for path, value in fields.items():
        rt.workspace.write(artifact_id, path, value, media_time_ms=t, causal_events=causal)
        written.append(path)
    # A draft write moves an EMPTY artifact to DRAFT.
    art = rt.workspace.get(artifact_id)
    if art is not None and art.lifecycle in (LifecycleState.EMPTY, LifecycleState.PARTIAL):
        art.lifecycle = LifecycleState.DRAFT
    return {"updated": written, "artifact_id": artifact_id}


def _h_set_lifecycle(rt: ToolRuntime, spec: ToolSpec, args: dict[str, Any], t: int) -> Any:
    artifact_id = args.get("artifact_id") or spec.effect.get("artifact_id")
    state = args.get("state") or spec.effect.get("state")
    if rt.workspace.get(artifact_id) is None:
        raise ToolError(f"set_lifecycle: unknown artifact '{artifact_id}'")
    rt.workspace.set_lifecycle(artifact_id, LifecycleState(state))
    return {"artifact_id": artifact_id, "lifecycle": state}


def _h_request_approval(rt: ToolRuntime, spec: ToolSpec, args: dict[str, Any], t: int) -> Any:
    action_type = args.get("action_type") or spec.commit_action_type or spec.effect.get("action_type")
    target_id = args.get("target_id") or args.get(spec.commit_target_param or "target_id")
    # The approval itself is granted by the user flow engine (an APPROVE plan). This handler only
    # records the request so the observer/flow can react; it never self-authorizes.
    return {"requested_action": action_type, "target_id": target_id, "status": "awaiting_approval"}


def _h_commit(rt: ToolRuntime, spec: ToolSpec, args: dict[str, Any], t: int) -> Any:
    action_type = spec.commit_action_type or args.get("action_type") or spec.name
    target_param = spec.commit_target_param or "target_id"
    target_id = args.get(target_param) or spec.effect.get("target_id") or ""
    decision = rt.commit_guard.authorize(action_type, target_id, t)
    out: dict[str, Any] = {
        "action_type": action_type,
        "target_id": target_id,
        "_auth_decision": decision.decision.value,
    }
    if decision.authorized:
        # Apply the committing effect: advance artifact lifecycle + set any world/field effects.
        artifact_id = args.get("artifact_id") or spec.effect.get("artifact_id")
        if artifact_id and rt.workspace.get(artifact_id) is not None:
            rt.workspace.set_lifecycle(artifact_id, LifecycleState.COMMITTED)
            for path, value in spec.effect.get("set_fields", {}).items():
                rt.workspace.write(artifact_id, path, value, media_time_ms=t)
        for path, value in spec.effect.get("world_updates", {}).items():
            rt.state.set(path, value, media_time_ms=t, cause=f"commit:{spec.name}")
        out["committed"] = True
    else:
        out["committed"] = False
    return out


BUILTIN_HANDLERS: dict[str, Handler] = {
    "kb_search": _h_kb_search,
    "get_entity": _h_get_entity,
    "update_artifact": _h_update_artifact,
    "set_lifecycle": _h_set_lifecycle,
    "request_approval": _h_request_approval,
    "commit": _h_commit,
}


__all__ = ["ToolRuntime", "ToolCallResult", "ToolError", "Handler", "BUILTIN_HANDLERS"]
