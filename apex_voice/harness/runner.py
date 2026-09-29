"""Deterministic C0 text runner.

Orchestrates a full text-mode run: the user flow engine and the agent alternate beats; tool calls
mutate the workspace/state through the tool runtime and commit guard; the agent-act observer feeds
the next flow step; the event engine applies seeded corrections (REVISE) and environment changes
(FOLLOW_THROUGH) at controlled media times; everything is written to the canonical event log so all
non-model scores are recomputable. It is used to validate tasks with the oracle and no-op agents;
the full-duplex benchmark path is :mod:`apex_voice.harness.realtime_runner`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apex_voice.adapters.base import AgentObservationView, AgentTurn, TextAgent
from apex_voice.artifacts.workspace import Workspace
from apex_voice.config import Condition, RunConfig
from apex_voice.environments.commit_guard import CommitGuard
from apex_voice.environments.events import EventEngine
from apex_voice.environments.knowledge import Document, KnowledgeBase
from apex_voice.environments.state_store import StateStore
from apex_voice.environments.tools import ToolRuntime
from apex_voice.harness.clocks import DualClock
from apex_voice.harness.logger import RunLogger
from apex_voice.schemas.artifact import ApprovalToken, ArtifactSchema, LifecycleState
from apex_voice.schemas.run_event import Actor, EventType
from apex_voice.schemas.user import SpeechAct, UserFact
from apex_voice.scoring.predicates import GradingContext
from apex_voice.scoring.task_success import PTSResult, score_task
from apex_voice.tasks.loader import LoadedTask
from apex_voice.user_sim.agent_observer import AgentActObserver
from apex_voice.user_sim.asset_bank import AssetSelector
from apex_voice.user_sim.compiler import RealizationCompiler
from apex_voice.user_sim.flow_engine import Observation, UserFlowEngine
from apex_voice.user_sim.state import UserStateManager

# Nominal media-time increments for C0 (turn-based) so latency stays separable from media time.
_USER_UTTERANCE_MS = 2000
_AGENT_UTTERANCE_MS = 2000
_MAX_TURNS = 60


@dataclass
class RunResult:
    run_id: str
    task_id: str
    condition: Condition
    pts: PTSResult
    outcome: str
    turns: int
    fallback_events: int
    leakage_violations: list[str] = field(default_factory=list)
    event_log_path: str | None = None

    @property
    def success(self) -> bool:
        return self.pts.pts == 1


def _build_workspace(lt: LoadedTask) -> Workspace:
    ws = Workspace()
    for aid, adef in lt.initial_workspace.get("artifacts", {}).items():
        schema = None
        sd = adef.get("schema")
        if sd:
            schema = ArtifactSchema.model_validate(sd)
        ws.create(
            artifact_id=aid,
            artifact_type=adef.get("artifact_type", "GENERIC"),
            schema=schema,
            initial_fields=adef.get("fields", {}),
            lifecycle=LifecycleState(adef.get("lifecycle", "EMPTY")),
        )
    return ws


def _build_knowledge(lt: LoadedTask) -> KnowledgeBase:
    kb = KnowledgeBase()
    manifest = lt.root / lt.spec.knowledge_dir / "manifest.json"
    if manifest.exists():
        import json

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


def _agent_tool_schemas(rt: ToolRuntime, lt: LoadedTask | None = None) -> list[dict[str, Any]]:
    """Tool schemas for the agent. When ``lt`` is given, enrich the free-form draft-write ``fields``
    object with the artifact's EXACT field keys (same as the realtime runner) so the model records
    under the names the grader expects rather than guessing (e.g. ``category`` not ``vendor_category``).
    """
    fmap: dict[str, list[tuple[str, str]]] = {}
    tool_aid: dict[str, str] = {}
    if lt is not None:
        from apex_voice.harness.realtime_runner import _artifact_field_map

        fmap = _artifact_field_map(lt)
        for t in lt.toolset.tools:
            aid = (t.effect or {}).get("artifact_id")
            if aid:
                tool_aid[t.name] = aid
    out = []
    for t in rt.available_tools():
        aid = tool_aid.get(t.name)
        keys = [nm for nm, _ in fmap.get(aid, [])] if aid else []
        params = []
        for p in t.params:
            entry: dict[str, Any] = {"name": p.name, "type": p.type, "required": p.required}
            if p.name == "fields" and keys:
                entry["field_keys"] = keys
            params.append(entry)
        desc = t.description
        if keys and any(p.name == "fields" for p in t.params):
            desc = f"{desc} Valid field keys for {aid}: " + ", ".join(keys) + "."
        out.append({"name": t.name, "description": desc, "params": params})
    return out


def ensure_realization_bank(lt: LoadedTask) -> None:
    """Compile a frozen text bank on the fly if the task ships none (dev/CI convenience)."""
    if lt.realization_bank.plan_ids():
        return
    compiler = RealizationCompiler()
    inputs = compiler.enumerate_plan_inputs(lt.flow, lt.events, lt.user_state)
    bank, _ = compiler.compile_all(inputs)
    lt.realization_bank = bank


def run_text_task(
    lt: LoadedTask,
    agent: TextAgent,
    config: RunConfig,
    run_dir: str | Path | None = None,
    observer: AgentActObserver | None = None,
    recover: bool = False,
) -> RunResult:
    """Execute one C0 run of ``lt`` with ``agent`` and grade it.

    ``observer`` overrides the default deterministic act observer (e.g. an :class:`LLMActObserver`
    semantic classifier for real free-form models). ``recover`` enables orchestrator hardening
    (re-state / answer-next / wind-down when the flow yields no plan) so a free-form model that asks
    out of order is not starved into a stall; leave it off for the deterministic oracle path.
    """
    run_id = uuid.uuid4().hex[:12]
    log_path = None
    if run_dir is not None:
        log_path = Path(run_dir) / run_id / "events.jsonl"
    logger = RunLogger(run_id, lt.spec.version, config.condition, log_path)
    clock = DualClock()

    # ---- environment ----------------------------------------------------------------
    state = StateStore(lt.initial_state)
    workspace = _build_workspace(lt)
    guard = CommitGuard(lt.spec.autonomy.approval_required_for)
    kb = _build_knowledge(lt)
    tools = ToolRuntime(
        lt.toolset, workspace, state, guard, kb, seed=config.simulator_seed, latency_stress=1.0
    )

    # ---- user sim -------------------------------------------------------------------
    usm = UserStateManager(lt.user_state)
    flow = UserFlowEngine(lt.flow, usm, config.scenario_id, config.simulator_seed)
    observer = observer or AgentActObserver(lt.observer.patterns, lt.observer.tool_act_map)
    ensure_realization_bank(lt)
    selector = AssetSelector(config.scenario_id, config.simulator_seed)
    events = EventEngine(lt.events.events)

    # Orchestrator hardening (mirrors realtime_runner): when the agent asks in an order the flow does
    # not expect, flow.step yields nothing -> without recovery the user goes mute and the model loops.
    # field2plan maps each askable field -> its ANSWER plan so we can re-state / answer-next / wind down
    # instead of dead-airing. This makes C0 grading fair vs C2 (same recovery both runners).
    field2plan: dict[str, tuple[str, list[str]]] = {}
    for _st in flow.flow.states.values():
        for _tr in _st.transitions:
            ak = _tr.when.agent_act
            if ak and ak.startswith("ASK:") and _tr.do is not None and _tr.do.act == SpeechAct.ANSWER:
                field2plan[ak[4:]] = (_tr.do.plan_id or f"ans_{ak[4:]}", list(_tr.do.facts))
    restate_counts: dict[str, int] = {}
    nudge_budget = 3
    finalize_budget = 2  # prompts to let the model record-all + set_ready before the call closes
    closed = False

    logger.emit(
        EventType.RUN_START,
        Actor.HARNESS,
        clock.wall_ns(),
        clock.media_time_ms,
        {"task_id": lt.spec.id, "model_id": agent.model_id, "condition": config.condition.value},
    )

    agent.reset(system_prompt=lt.spec.title, tools=_agent_tool_schemas(tools, lt))

    obs = Observation(media_time_ms=clock.media_time_ms)
    pending_action: tuple[str, str] | None = None  # (action_type, target_id) awaiting approval
    fallback_events = 0
    last_tool_results: list[dict[str, Any]] = []
    turn_index = 0

    while turn_index < _MAX_TURNS:
        turn_index += 1

        # ----- USER BEAT -------------------------------------------------------------
        # Drain all user plans reachable from the current observation (a real agent may ask several
        # fields at once / out of order → the permissive collection flow answers each).
        plans = []
        while len(plans) < 24:
            cur = flow.current
            p = flow.step(obs)
            if p is None:
                break
            plans.append(p)
            if flow.current != cur:  # state changed (linear flow) -> one plan per beat
                break
        user_texts: list[str] = []
        for plan in plans:
            variant = selector.select_variant(lt.realization_bank, plan.id)
            if variant is None:
                fallback_events += 1
                logger.emit(
                    EventType.SIMULATOR_FALLBACK,
                    Actor.HARNESS,
                    clock.wall_ns(),
                    clock.media_time_ms,
                    {"user_plan_id": plan.id},
                )
                txt = f"[FALLBACK:{plan.id}]"
            else:
                txt = variant.text
            user_texts.append(txt)
            logger.emit(
                EventType.USER_PLAN,
                Actor.USER,
                clock.wall_ns(),
                clock.media_time_ms,
                {"plan_id": plan.id, "speech_act": plan.speech_act.value, "facts": plan.fact_ids},
            )
            logger.emit(EventType.USER_TEXT, Actor.USER, clock.wall_ns(), clock.media_time_ms, {"text": txt})
            if plan.speech_act == SpeechAct.APPROVE and pending_action is not None:
                at, tid = pending_action
                guard.grant(
                    ApprovalToken(
                        token_id=f"tok_{run_id}_{turn_index}",
                        action_type=at,
                        target_id=tid,
                        granted_media_time_ms=clock.media_time_ms,
                        source_event_id=plan.id,
                    )
                )
                logger.emit(
                    EventType.APPROVAL_EVENT,
                    Actor.USER,
                    clock.wall_ns(),
                    clock.media_time_ms,
                    {"granted": at, "target_id": tid},
                )
            elif plan.speech_act in (SpeechAct.REVOKE, SpeechAct.DENY) and pending_action is not None:
                at, tid = pending_action
                guard.revoke(at, tid, clock.media_time_ms)
                logger.emit(
                    EventType.APPROVAL_EVENT,
                    Actor.USER,
                    clock.wall_ns(),
                    clock.media_time_ms,
                    {"revoked": at, "target_id": tid},
                )
            clock.advance_media(_USER_UTTERANCE_MS)

        # ----- USER-BEAT RECOVERY (no plan fired) ------------------------------------
        if not plans and (flow.done or not recover):
            if flow.done:
                break
        elif not plans:  # recover=True and flow not done
            asked = [k[4:] for k in obs.new_act_keys if k.startswith("ASK:")]
            targets = [f for f in asked if f in field2plan and restate_counts.get(f, 0) < 2]
            if not targets:
                targets = [
                    f for f in field2plan if f not in flow.user.revealed and restate_counts.get(f, 0) < 2
                ][:1]
            recovered = None
            for f in targets:
                pid, facts = field2plan[f]
                var = selector.select_variant(lt.realization_bank, pid)
                if var is not None:
                    recovered = (f, pid, facts, var.text)
                    break
            if recovered is not None:
                f, pid, facts, txt = recovered
                restate_counts[f] = restate_counts.get(f, 0) + 1
                for fid in facts:
                    flow.user.mark_revealed(fid)
                user_texts.append(txt)
                logger.emit(
                    EventType.USER_PLAN,
                    Actor.USER,
                    clock.wall_ns(),
                    clock.media_time_ms,
                    {"plan_id": pid, "speech_act": "ANSWER", "facts": facts, "recovered": True},
                )
                logger.emit(
                    EventType.USER_TEXT,
                    Actor.USER,
                    clock.wall_ns(),
                    clock.media_time_ms,
                    {"text": txt, "recovered": True},
                )
                clock.advance_media(_USER_UTTERANCE_MS)
            elif not [f for f in field2plan if f not in flow.user.revealed]:
                # All facts given. Give the model turns to finish RECORDING + FINALIZE before ending:
                # first a couple of finalize prompts (consumes nudge_budget), then a clean sign-off.
                if finalize_budget > 0 and not closed:
                    finalize_budget -= 1
                    txt = (
                        "That's all my information. Could you make sure everything is recorded and "
                        "mark it ready for review?"
                    )
                    user_texts.append(txt)
                    logger.emit(
                        EventType.USER_TEXT,
                        Actor.USER,
                        clock.wall_ns(),
                        clock.media_time_ms,
                        {"text": txt, "finalize_nudge": True},
                    )
                    clock.advance_media(_USER_UTTERANCE_MS)
                elif closed:
                    break  # already signed off -> end the call
                else:
                    closed = True
                    txt = "Great, thanks for your help!"
                    user_texts.append(txt)
                    logger.emit(
                        EventType.USER_TEXT,
                        Actor.USER,
                        clock.wall_ns(),
                        clock.media_time_ms,
                        {"text": txt, "closing": True},
                    )
                    clock.advance_media(_USER_UTTERANCE_MS)
            else:
                if nudge_budget <= 0:
                    break  # genuinely stuck: fields pending but none recoverable
                nudge_budget -= 1
                txt = "Sorry — could you go on?"
                user_texts.append(txt)
                logger.emit(
                    EventType.USER_TEXT,
                    Actor.USER,
                    clock.wall_ns(),
                    clock.media_time_ms,
                    {"text": txt, "nudge": True},
                )
                clock.advance_media(_USER_UTTERANCE_MS)

        user_text: str | None = " ".join(user_texts) if user_texts else None

        # ----- AGENT BEAT ------------------------------------------------------------
        view = AgentObservationView(
            user_text=user_text,
            tool_results=last_tool_results,
            available_tools=_agent_tool_schemas(tools, lt),
            turn_index=turn_index,
        )
        agent_turn: AgentTurn = agent.act(view)
        wall = clock.wall_ns()

        new_act_keys: list[str] = []
        tool_calls_seen: list[str] = []
        approval_requests: list[str] = []
        commit_attempts: list[str] = []
        last_tool_results = []

        if agent_turn.text:
            logger.emit(
                EventType.AGENT_TEXT, Actor.AGENT, wall, clock.media_time_ms, {"text": agent_turn.text}
            )
            for a in observer.observe_text(agent_turn.text):
                new_act_keys.append(a.key())
                logger.emit(
                    EventType.AGENT_ACT,
                    Actor.AGENT,
                    wall,
                    clock.media_time_ms,
                    {"act": a.key(), "source": a.source},
                )

        muts_before = len(workspace.history)
        for tc in agent_turn.tool_calls:
            logger.emit(
                EventType.TOOL_CALL,
                Actor.AGENT,
                wall,
                clock.media_time_ms,
                {"tool": tc.name, "args": tc.args},
            )
            result = tools.call(tc.name, tc.args, clock.media_time_ms)
            tool_calls_seen.append(tc.name)
            logger.emit(
                EventType.TOOL_RESULT,
                Actor.ENVIRONMENT,
                clock.wall_ns(),
                clock.media_time_ms,
                {
                    "tool": tc.name,
                    "ok": result.ok,
                    "result": result.result,
                    "auth_decision": result.auth_decision,
                },
            )
            last_tool_results.append({"tool": tc.name, "ok": result.ok, "result": result.result})
            # Log artifact mutations that just occurred.
            spec = lt.toolset.by_name(tc.name)
            if spec and spec.is_commit and isinstance(result.result, dict):
                commit_attempts.append(result.result.get("action_type", tc.name))
            # Track pending approval request.
            if isinstance(result.result, dict) and result.result.get("status") == "awaiting_approval":
                pending_action = (result.result.get("requested_action"), result.result.get("target_id"))
                approval_requests.append(result.result.get("requested_action") or tc.name)
            for a in observer.observe_tool(result):
                new_act_keys.append(a.key())
                logger.emit(
                    EventType.AGENT_ACT,
                    Actor.AGENT,
                    clock.wall_ns(),
                    clock.media_time_ms,
                    {"act": a.key(), "source": a.source},
                )

        clock.advance_media(_AGENT_UTTERANCE_MS)
        art_muts = [f"{m.artifact_id}.{m.path}" for m in workspace.history[muts_before:]]

        # ----- EVENT ENGINE (REVISE corrections, FOLLOW_THROUGH changes) -------------
        fired_now: list[str] = []
        for se in events.pending:
            trig = se.spec.trigger
            matched = True
            if trig.agent_act is not None and trig.agent_act not in new_act_keys:
                matched = False
            if trig.tool_call is not None and trig.tool_call not in tool_calls_seen:
                matched = False
            if trig.artifact_mutation is not None and trig.artifact_mutation not in art_muts:
                matched = False
            if trig.commit_attempt is not None and trig.commit_attempt not in commit_attempts:
                matched = False
            if trig.media_time_ms_gte is not None and clock.media_time_ms < trig.media_time_ms_gte:
                matched = False
            if not matched:
                continue
            events.mark_fired(se.spec.id, clock.media_time_ms)
            fired_now.append(se.spec.id)
            logger.emit(
                EventType.DUPLEX_EVENT,
                Actor.USER,
                clock.wall_ns(),
                clock.media_time_ms,
                {"event_id": se.spec.id, "type": se.spec.type},
            )
            # Apply REVISE fact corrections: add a superseding fact + mark dependents stale.
            for fid, new_val in se.spec.fact_updates.items():
                new_fact = UserFact(id=f"{fid}__v_evt_{se.spec.id}", value=new_val, supersedes=fid)
                usm.state.facts.append(new_fact)
                usm._facts[new_fact.id] = new_fact  # noqa: SLF001
                usm.superseded.add(fid)
            for repair_path in se.spec.expected_repairs:
                if "." in repair_path:
                    aid, fld = repair_path.split(".", 1)
                    if workspace.get(aid) is not None:
                        workspace.mark_stale(aid, fld)
            for path, val in se.spec.world_updates.items():
                state.set(path, val, media_time_ms=clock.media_time_ms, cause=f"event:{se.spec.id}")
            # Revocation event revokes matching tokens at user speech onset.
            if se.spec.forbidden_tool_commits and pending_action is not None:
                at, tid = pending_action
                guard.revoke(at, tid, clock.media_time_ms)

        # ----- build next observation ------------------------------------------------
        rep_counts: dict[str, int] = {}
        for k in new_act_keys:
            rep_counts[k] = sum(1 for x in observer.log if k in x.get("acts", []))
        obs = Observation(
            media_time_ms=clock.media_time_ms,
            new_act_keys=new_act_keys,
            tool_calls=tool_calls_seen,
            commit_attempts=commit_attempts,
            approval_requests=approval_requests,
            fired_events=fired_now,
            artifact_mutations=art_muts,
            repetition_counts=rep_counts,
        )

        if agent_turn.end_call:
            break

    # ---- grade ----------------------------------------------------------------------
    ctx = GradingContext(
        workspace=workspace,
        state=state,
        tool_log=tools.call_log,
        events=logger.events,
        knowledge=kb,
        commit_guard=guard,
        observed_act_keys=set(flow._cum_acts),  # noqa: SLF001
    )
    pts = score_task(lt.gold, ctx)

    outcome = "SUCCESS" if pts.pts == 1 else "FAILURE"
    if fallback_events:
        outcome = "SIMULATOR_FALLBACK"
    logger.emit(
        EventType.RUN_END,
        Actor.HARNESS,
        clock.wall_ns(),
        clock.media_time_ms,
        {"outcome": outcome, "pts": pts.pts},
    )
    logger.close()
    if log_path is not None:
        from apex_voice.harness.setup import write_workspace_outputs

        write_workspace_outputs(Path(log_path).parent, workspace)

    return RunResult(
        run_id=run_id,
        task_id=lt.spec.id,
        condition=config.condition,
        pts=pts,
        outcome=outcome,
        turns=turn_index,
        fallback_events=fallback_events,
        leakage_violations=list(flow.leakage_violations),
        event_log_path=str(log_path) if log_path else None,
    )


__all__ = ["run_text_task", "RunResult", "ensure_realization_bank"]
