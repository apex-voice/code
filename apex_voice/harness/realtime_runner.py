"""Async full-duplex runner for live realtime speech-to-speech models.

Drives a genuine full-duplex model (e.g. Step-Audio3 Realtime) against an APEX task: the frozen
user turns are synthesized (Kokoro) and streamed to the model in real time; the model's audio/text/
tool-calls stream back and are placed on the R channel at their true arrival time (so overlaps are
real); barge-in corrections are injected *while the model is still speaking* and the interrupt-stop-
latency (ISL) is measured from ``response.cancel`` to the model's last audio delta. Tool calls are
dispatched through the same ToolRuntime + CommitGuard as the text/voice runners, so PTS is graded
identically. Everything is logged to the canonical event log; stereo (L=user, R=agent) + isolated
channels are written for review.

This is the path that exercises a model's duplex behavior. Adapters and TTS are imported lazily,
so offline CI is unaffected.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from apex_voice.config import Condition, RunConfig
from apex_voice.environments.commit_guard import CommitGuard
from apex_voice.environments.events import EventEngine
from apex_voice.environments.tools import ToolRuntime
from apex_voice.harness.audio import PlacedClip, assemble_stereo, isolated_channel, resample, write_wav
from apex_voice.harness.logger import RunLogger
from apex_voice.harness.setup import build_knowledge, build_workspace, write_workspace_outputs
from apex_voice.schemas.artifact import ApprovalToken
from apex_voice.schemas.run_event import Actor, EventType
from apex_voice.schemas.user import FloorAction, SpeechAct, UserFact
from apex_voice.scoring.predicates import GradingContext
from apex_voice.scoring.task_success import PTSResult, score_task
from apex_voice.tasks.loader import LoadedTask
from apex_voice.user_sim.asset_bank import AssetSelector
from apex_voice.user_sim.flow_engine import Observation, UserFlowEngine
from apex_voice.user_sim.llm_observer import LLMActObserver, candidate_acts_for
from apex_voice.user_sim.state import UserStateManager

_SR = 24000  # realtime S2S pcm16 rate (Step-Audio3)
_CHUNK_MS = 40  # user audio streamed in 40 ms real-time frames
# Natural sign-offs for a smooth end-of-call once the task is complete (non-graded; one is chosen by seed).
_CLOSINGS = [
    "No, that's everything — thanks so much for your help!",
    "That's all I needed. Thanks a lot, appreciate it!",
    "Nope, that covers it — thank you, have a good one!",
    "That's everything on my end. Thanks for sorting that out!",
]
_NUDGE = "Sorry — could you go on?"  # bounded nudge when the user is stuck mid-call
_BARGE_AFTER_MS = 700  # let the model speak this long before injecting a barge-in correction
_MAX_TURNS = 40
_DRAIN_TIMEOUT = 30.0  # per-event wait before giving up on a response
_MAX_SUBRESPONSES = 8  # tool<->speak cycles within one agent beat

_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "text": "string",
    "number": "number",
    "float": "number",
    "int": "integer",
    "integer": "integer",
    "bool": "boolean",
    "boolean": "boolean",
    "object": "object",
    "dict": "object",
    "array": "array",
}

_INSTRUCTIONS = (
    "You are a professional voice assistant completing a work task with the caller over the phone. "
    "Speak like a busy professional on a call: warm but BRIEF.\n"
    "CRITICAL RULES:\n"
    "1. Each spoken turn is AT MOST ONE short sentence — either a single question, or a five-word "
    "acknowledgement then the next question ('Got it. What's your date of birth?'). Hard cap ~15 "
    "words. NEVER narrate your reasoning, the caller's answer, your plan, or your tool use. Forbidden "
    "openings include 'Let me...', 'I should...', 'I'll update/record/note...', 'The caller said...', "
    "'So that means...', 'First I need to...', 'Now I will...'. Do NOT read back the value you just "
    "recorded. Call the tool SILENTLY and simply ask the next question. If you catch yourself "
    "explaining what you are doing, stop and just ask the next question instead.\n"
    "2. Record EVERY piece of information the caller gives you by immediately calling the matching "
    "update tool (e.g. update_<artifact>) with that field — do this as soon as you hear each answer, "
    "before asking the next question.\n"
    "3. Accept the caller's answers as given. Identifiers may be names, codes, or numbers — do not "
    "insist on a particular format.\n"
    "4. Ask for the information you still need, ONE item at a time, and keep going until you have "
    "gathered everything the task requires. Do not end the call early.\n"
    "5. If a policy lookup is relevant, call the knowledge/search tool.\n"
    "6. If the caller corrects something they said earlier, call the update tool again to fix the "
    "affected field(s).\n"
    "7. For any action that needs approval, first summarize it and ask the caller to confirm; only "
    "after they say yes, call the tool to perform it. Do not ask for information you already have.\n"
    "8. When you have recorded everything the task requires, finalize the record before wrapping up: "
    "call the tool that marks it ready for review (e.g. set_ready / mark_ready / finalize) if one is "
    "available. Do this after the last field is recorded and before you say goodbye."
)


# ---- pcm helpers ----------------------------------------------------------------------


def _f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _pcm16_to_f32(b: bytes) -> np.ndarray:
    if not b:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def _artifact_field_map(lt: LoadedTask) -> dict[str, list[tuple[str, str]]]:
    """artifact_id -> [(field_name, json_type)] from the workspace schema (drop lifecycle 'status')."""
    out: dict[str, list[tuple[str, str]]] = {}
    for aid, adef in lt.initial_workspace.get("artifacts", {}).items():
        sc = adef.get("schema") or {}
        fields = sc.get("fields")
        names: list[tuple[str, str]] = []
        if isinstance(fields, list):
            for f in fields:
                nm = f.get("name")
                if nm and nm != "status":
                    names.append((nm, _TYPE_MAP.get(str(f.get("type", "string")).lower(), "string")))
        out[aid] = names
    return out


def _tool_schema(rt: ToolRuntime, lt: LoadedTask) -> list[dict[str, Any]]:
    fmap = _artifact_field_map(lt)
    tool_specs = {t.name: t for t in lt.toolset.tools}
    out = []
    for t in rt.available_tools():
        props, required = {}, []
        spec = tool_specs.get(t.name)
        aid = (spec.effect or {}).get("artifact_id") if spec else None
        for p in t.params:
            jt = _TYPE_MAP.get(str(p.type).lower(), "string")
            # Enrich the free-form draft-write "fields" object with the artifact's real field keys so
            # the model records under the exact names the grader expects (not guessed names).
            if p.name == "fields" and jt == "object" and aid and fmap.get(aid):
                fprops = {nm: {"type": ft} for nm, ft in fmap[aid]}
                props[p.name] = {
                    "type": "object",
                    "properties": fprops,
                    "additionalProperties": False,
                    "description": "Record one or more of these fields using these EXACT "
                    "key names: " + ", ".join(nm for nm, _ in fmap[aid]),
                }
            else:
                props[p.name] = {"type": jt}
            if p.required:
                required.append(p.name)
        desc = t.description
        if aid and fmap.get(aid) and any(p.name == "fields" for p in t.params):
            desc = f"{desc} Valid field keys for {aid}: " + ", ".join(nm for nm, _ in fmap[aid]) + "."
        out.append(
            {
                "name": t.name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            }
        )
    return out


@dataclass
class RealtimeRunResult:
    run_id: str
    task_id: str
    condition: Condition
    pts: PTSResult
    outcome: str
    turns: int
    transcript: list[dict[str, str]] = field(default_factory=list)
    duplex: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    audio_dir: str | None = None
    event_log_path: str | None = None
    fallback_events: int = 0

    @property
    def success(self) -> bool:
        return self.pts.pts == 1


@dataclass
class _AgentResponse:
    audio: np.ndarray
    transcript: str
    tool_calls: list[dict[str, Any]]
    start_ms: int | None
    end_ms: int | None
    barged: bool = False
    isl_ms: int | None = None


class _RealtimeSession:
    """Owns the timing clock, event iterator, and audio placement for one run."""

    def __init__(self, adapter, logger: RunLogger, sr: int = _SR) -> None:
        self.adapter = adapter
        self.logger = logger
        self.sr = sr
        self.clips: list[PlacedClip] = []
        self._t0 = time.monotonic()
        self._evgen = None
        self.rsl_samples: list[int] = []
        self.duplex_events: list[dict[str, Any]] = []

    def now_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def start_events(self) -> None:
        self._evgen = self.adapter.events().__aiter__()

    async def _next(self) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._evgen.__anext__(), _DRAIN_TIMEOUT)
        except (TimeoutError, StopAsyncIteration):
            return None

    async def stream_user(self, audio_f32: np.ndarray) -> tuple[int, int]:
        """Stream a user clip to the model in real time; place it on L; return (onset_ms, dur_ms)."""
        onset = self.now_ms()
        pcm = _f32_to_pcm16(audio_f32)
        bytes_per_frame = int(self.sr * _CHUNK_MS / 1000) * 2
        for i in range(0, len(pcm), bytes_per_frame):
            await self.adapter.send_audio(pcm[i : i + bytes_per_frame])
            await asyncio.sleep(_CHUNK_MS / 1000)
        dur = int(round(1000 * len(audio_f32) / self.sr))
        self.clips.append(PlacedClip("user", onset, audio_f32.astype(np.float32)))
        return onset, dur

    async def collect_response(self, barge: dict[str, Any] | None = None) -> _AgentResponse:
        """Read events until response.done, accumulating audio/transcript/tool_calls.

        If ``barge`` is given ({'audio': f32, 'plan': UserPlan}), inject it once the model has spoken
        for ~_BARGE_AFTER_MS: stream the user audio, cancel the model response, and measure ISL from
        the cancel to the last audio delta.
        """
        parts: list[np.ndarray] = []
        spoken: list[str] = []  # response.audio_transcript.* — what the model actually says
        text_aux: list[str] = []  # response.text.* — separate text channel (often reasoning); ignored
        tool_calls: list[dict[str, Any]] = []
        start_ms: int | None = None
        acc_ms = 0.0
        cancel_wall: float | None = None
        last_audio_wall: float | None = None
        barge_task: asyncio.Task | None = None
        barged = False
        isl_ms: int | None = None

        while True:
            ev = await self._next()
            if ev is None:
                break
            t = ev.get("type", "")
            if t == "response.audio.delta" and ev.get("audio"):
                chunk = _pcm16_to_f32(ev["audio"])
                if start_ms is None:
                    start_ms = self.now_ms()
                parts.append(chunk)
                acc_ms += 1000.0 * len(chunk) / self.sr
                last_audio_wall = time.monotonic()
                # Fire the barge-in once the model has held the floor long enough.
                if barge is not None and not barged and acc_ms >= _BARGE_AFTER_MS:
                    barged = True
                    barge_task = asyncio.create_task(self.stream_user(barge["audio"]))
                    await self.adapter.cancel_response()
                    cancel_wall = time.monotonic()
            elif t == "response.audio_transcript.delta":
                if ev.get("text"):
                    spoken.append(ev["text"])
            elif t == "response.text.delta":
                if ev.get("text"):
                    text_aux.append(ev["text"])
            elif t in ("response.audio_transcript.done", "response.text.done"):
                pass  # deltas already captured; .done repeats the whole string
            elif t == "response.output_item.done" and ev.get("tool_call"):
                tool_calls.append(ev["tool_call"])
            elif t == "error":
                # Log but do NOT truncate the response: a spurious server error (e.g. a coalesced
                # duplicate response.create) must not cut off a legitimate in-progress turn — which
                # would drop a closing tool call such as set_ready. Real disconnects still terminate
                # the loop via the None sentinel from _next().
                self.logger.emit(
                    EventType.INFRA_FAILURE,
                    Actor.ENVIRONMENT,
                    0,
                    self.now_ms(),
                    {"error": str(ev.get("error"))},
                )
            elif t == "response.done":
                break

        if barge_task is not None:
            try:
                await barge_task
            except Exception:  # noqa: BLE001
                pass
            if cancel_wall is not None and last_audio_wall is not None:
                isl_ms = max(0, int(1000 * (last_audio_wall - cancel_wall)))

        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if start_ms is not None and audio.size:
            end_ms = start_ms + int(round(1000 * len(audio) / self.sr))
            self.clips.append(PlacedClip("agent", start_ms, audio))
        else:
            end_ms = None
        text = "".join(spoken).strip() or "".join(text_aux).strip()
        return _AgentResponse(
            audio=audio,
            transcript=text,
            tool_calls=tool_calls,
            start_ms=start_ms,
            end_ms=end_ms,
            barged=barged,
            isl_ms=isl_ms,
        )


async def run_realtime_task(
    lt: LoadedTask,
    adapter,
    config: RunConfig,
    run_dir: str | Path | None = None,
    compiler=None,
    max_turns: int = _MAX_TURNS,
    judge=None,
) -> RealtimeRunResult:
    from apex_voice.harness.runner import ensure_realization_bank  # local: avoids cycle at import
    from apex_voice.user_sim.kokoro_compiler import get_default_compiler

    condition = config.condition
    compiler = compiler or get_default_compiler(prefer_real=True, sample_rate=_SR)
    csr = getattr(compiler, "sample_rate", _SR)

    run_id = uuid.uuid4().hex[:12]
    run_root = Path(run_dir) / run_id if run_dir else None
    log_path = (run_root / "events.jsonl") if run_root else None
    logger = RunLogger(run_id, lt.spec.version, condition, log_path)

    # environment + user sim (identical wiring to the text/voice runners)
    from apex_voice.environments.state_store import StateStore

    state = StateStore(lt.initial_state)
    workspace = build_workspace(lt)
    guard = CommitGuard(lt.spec.autonomy.approval_required_for)
    kb = build_knowledge(lt)
    tools = ToolRuntime(lt.toolset, workspace, state, guard, kb, seed=config.simulator_seed)
    usm = UserStateManager(lt.user_state)
    flow = UserFlowEngine(lt.flow, usm, config.scenario_id, config.simulator_seed)
    ensure_realization_bank(lt)
    selector = AssetSelector(config.scenario_id, config.simulator_seed)
    events = EventEngine(lt.events.events)
    observer = LLMActObserver(
        lt.observer.patterns,
        lt.observer.tool_act_map,
        candidate_acts_for(lt.observer.patterns, lt.observer.tool_act_map, lt.flow),
    )
    persona = lt.persona or {}

    def _synth(text: str) -> np.ndarray:
        wav = compiler.compile(text, persona, None, None).waveform.astype(np.float32)
        return resample(wav, csr, _SR) if csr != _SR else wav

    logger.emit(
        EventType.RUN_START,
        Actor.HARNESS,
        0,
        0,
        {
            "task_id": lt.spec.id,
            "model_id": getattr(adapter, "model_id", "realtime"),
            "condition": condition.value,
            "user_audio": getattr(compiler, "name", "unknown"),
        },
    )

    await adapter.start_session(
        {
            "instructions": f"{_INSTRUCTIONS}\n\nTask: {lt.spec.title}.",
            "tools": _tool_schema(tools, lt),
            "turn_detection": None,  # deterministic matched control: we commit + create_response
            "voice": persona.get("realtime_voice", "soft-spoken-gentleman"),
            "modalities": ["text", "audio"],  # audio-only is rejected; we treat audio_transcript as speech
            "max_response_output_tokens": 600,  # bound each turn; prevents runaway repetition
        }
    )
    sess = _RealtimeSession(adapter, logger, sr=_SR)
    sess.start_events()
    # Drain the initial session.created/updated events without blocking forever.
    await asyncio.sleep(0.5)

    pending_action: tuple[str, str] | None = None
    last_tool_results: list[dict[str, Any]] = []
    transcript_log: list[dict[str, str]] = []
    fallback_events = 0
    obs = Observation(media_time_ms=0)
    turn_index = 0
    nudge_budget = 3  # neutral continuers when the model pauses without asking a mappable question
    closed = False  # emit exactly one warm sign-off, then end the call smoothly
    barge_next: dict[str, Any] | None = None  # a correction to inject into the next model response

    # Orchestrator hardening: map field -> (answer plan_id, facts) from the collection flow, so when
    # the flow yields nothing (model RE-asks an already-answered field, or the observer misses its
    # phrasing) we can still respond productively — re-state the answer / answer the next needed field
    # — instead of going mute and stalling the conversation.
    field2plan: dict[str, tuple[str, list[str]]] = {}
    for _st in flow.flow.states.values():
        for _tr in _st.transitions:
            ak = _tr.when.agent_act
            if ak and ak.startswith("ASK:") and _tr.do is not None and _tr.do.act == SpeechAct.ANSWER:
                field2plan[ak[4:]] = (_tr.do.plan_id or f"ans_{ak[4:]}", list(_tr.do.facts))
    restate_counts: dict[str, int] = {}

    # Flow transitions whose user plan barges in (corrections/revocations), keyed by the external
    # event that arms them — lets us detect mid-beat when a barge should fire and inject it into the
    # model's *follow-up* speech (which send_tool_result triggers) rather than a clean later turn.
    barge_event_ids: set[str] = set()
    for _st in lt.flow.states.values():
        for _tr in _st.transitions:
            if (
                _tr.do is not None
                and _tr.when.external_event
                and _tr.do.floor_action in (FloorAction.BARGE_IN, FloorAction.URGENT_OVERRIDE)
            ):
                barge_event_ids.add(_tr.when.external_event)

    def apply_events(new_act_keys, tool_calls_seen, art_muts, commit_attempts) -> list[str]:
        """Fire matching REVISE/FOLLOW_THROUGH events (idempotent via mark_fired). Mirrors the text
        runner; returns the ids fired this call."""
        nonlocal pending_action
        fired: list[str] = []
        for se in events.pending:
            trig = se.spec.trigger
            if trig.agent_act is not None and trig.agent_act not in new_act_keys:
                continue
            if trig.tool_call is not None and trig.tool_call not in tool_calls_seen:
                continue
            if trig.artifact_mutation is not None and trig.artifact_mutation not in art_muts:
                continue
            if trig.commit_attempt is not None and trig.commit_attempt not in commit_attempts:
                continue
            events.mark_fired(se.spec.id, sess.now_ms())
            fired.append(se.spec.id)
            logger.emit(
                EventType.DUPLEX_EVENT,
                Actor.USER,
                0,
                sess.now_ms(),
                {"event_id": se.spec.id, "type": se.spec.type},
            )
            for fid, new_val in se.spec.fact_updates.items():
                nf = UserFact(id=f"{fid}__v_evt_{se.spec.id}", value=new_val, supersedes=fid)
                usm.state.facts.append(nf)
                usm._facts[nf.id] = nf  # noqa: SLF001
                usm.superseded.add(fid)
            for rp in se.spec.expected_repairs:
                if "." in rp:
                    aid, fld = rp.split(".", 1)
                    if workspace.get(aid) is not None:
                        workspace.mark_stale(aid, fld)
            for path, val in se.spec.world_updates.items():
                state.set(path, val, media_time_ms=sess.now_ms(), cause=f"event:{se.spec.id}")
            if se.spec.forbidden_tool_commits and pending_action is not None:
                guard.revoke(pending_action[0], pending_action[1], sess.now_ms())
        return fired

    while turn_index < max_turns:
        turn_index += 1

        # ---------------- USER BEAT ----------------
        drained = []
        while len(drained) < 24:
            cur = flow.current
            p = flow.step(obs)
            if p is None:
                break
            drained.append(p)
            if flow.current != cur:
                break

        if not drained and flow.done:
            break
        if not drained:
            # Recovery (orchestrator hardening): the flow produced no plan. Rather than go mute, respond
            # productively so the conversation never stalls: (1) if the model just asked about a field
            # (even one already answered — models repeat), RE-STATE that field's answer; (2) else answer
            # the next still-needed field to keep making progress; (3) only if neither applies, a bounded
            # neutral continuer. Re-statements are capped per field to avoid ping-ponging with a stuck model.
            asked = [k[4:] for k in obs.new_act_keys if k.startswith("ASK:")]
            targets = [f for f in asked if f in field2plan and restate_counts.get(f, 0) < 2]
            if not targets:
                targets = [
                    f
                    for (f, (pid, _)) in field2plan.items()
                    if f not in flow.user.revealed and restate_counts.get(f, 0) < 2
                ][:1]
            recovered = None
            for f in targets:
                pid, facts = field2plan[f]
                var = selector.select_variant(lt.realization_bank, pid)
                if var is not None:
                    recovered = (f, pid, facts, var.text)
                    break
            if recovered is not None:
                f, pid, facts, text = recovered
                restate_counts[f] = restate_counts.get(f, 0) + 1
                for fid in facts:
                    flow.user.mark_revealed(fid)  # keep reveal state consistent
                audio = _synth(text)
                onset, _ = await sess.stream_user(audio)
                transcript_log.append({"speaker": "user", "plan": f"{pid}(recover)", "text": text})
                logger.emit(
                    EventType.USER_PLAN,
                    Actor.USER,
                    0,
                    onset,
                    {"plan_id": pid, "speech_act": "ANSWER", "facts": facts, "recovered": True},
                )
                logger.emit(
                    EventType.USER_AUDIO_START,
                    Actor.USER,
                    0,
                    onset,
                    {"text": text, "duration_ms": int(round(1000 * len(audio) / _SR)), "recovered": True},
                )
                await adapter.commit_audio()
                await adapter.create_response()
            else:
                pending = [f for f in field2plan if f not in flow.user.revealed]
                if not pending:
                    # Task complete from the user's side (all fields given/approved). Give ONE natural
                    # sign-off, let the model acknowledge, then end the call — a smooth wind-down instead
                    # of repeatedly nudging the model to "finish up".
                    if closed:
                        break
                    closed = True
                    txt = _CLOSINGS[(config.event_seed or 0) % len(_CLOSINGS)]
                    audio = _synth(txt)
                    onset, _ = await sess.stream_user(audio)
                    transcript_log.append({"speaker": "user", "plan": "(close)", "text": txt})
                    logger.emit(
                        EventType.USER_PLAN,
                        Actor.USER,
                        0,
                        onset,
                        {"plan_id": "close", "speech_act": "END", "facts": []},
                    )
                    logger.emit(
                        EventType.USER_AUDIO_START,
                        Actor.USER,
                        0,
                        onset,
                        {"text": txt, "duration_ms": int(round(1000 * len(audio) / _SR)), "closing": True},
                    )
                    await adapter.commit_audio()
                    await adapter.create_response()
                else:
                    # Genuinely stuck mid-call (fields still pending but none recoverable): bounded nudge.
                    if nudge_budget <= 0:
                        break
                    nudge_budget -= 1
                    filler = _NUDGE
                    audio = _synth(filler)
                    onset, _ = await sess.stream_user(audio)
                    transcript_log.append({"speaker": "user", "plan": "(nudge)", "text": filler})
                    logger.emit(
                        EventType.USER_AUDIO_START,
                        Actor.USER,
                        0,
                        onset,
                        {"text": filler, "duration_ms": int(round(1000 * len(audio) / _SR)), "nudge": True},
                    )
                    await adapter.commit_audio()
                    await adapter.create_response()
        else:
            for plan in drained:
                variant = selector.select_variant(lt.realization_bank, plan.id)
                if variant is None:
                    fallback_events += 1
                    logger.emit(
                        EventType.SIMULATOR_FALLBACK,
                        Actor.HARNESS,
                        0,
                        sess.now_ms(),
                        {"user_plan_id": plan.id},
                    )
                    continue
                text = variant.text
                transcript_log.append({"speaker": "user", "plan": plan.id, "text": text})
                audio = _synth(text)
                is_barge = plan.floor_action in (FloorAction.BARGE_IN, FloorAction.URGENT_OVERRIDE)
                # Stream every user turn (reliable fact delivery). Real overlap still arises because the
                # model streams audio concurrently and often talks past our turn boundary; a correction
                # is flagged so the duplex metrics record it as an interrupt.
                onset, dur = await sess.stream_user(audio)
                if is_barge:
                    sess.duplex_events.append(
                        {
                            "event_id": plan.event_id,
                            "type": "USER_CORRECTION",
                            "expected_floor_action": "YIELD",
                            "onset_ms": onset,
                        }
                    )
                    logger.emit(
                        EventType.DUPLEX_EVENT,
                        Actor.USER,
                        0,
                        onset,
                        {"event_id": plan.event_id, "type": "USER_CORRECTION", "onset_ms": onset},
                    )
                logger.emit(
                    EventType.USER_PLAN,
                    Actor.USER,
                    0,
                    onset,
                    {
                        "plan_id": plan.id,
                        "speech_act": plan.speech_act.value,
                        "facts": plan.fact_ids,
                        "floor_action": plan.floor_action.value,
                    },
                )
                logger.emit(
                    EventType.USER_AUDIO_START,
                    Actor.USER,
                    0,
                    onset,
                    {"text": text, "duration_ms": int(round(1000 * len(audio) / _SR))},
                )

                # approval / revocation effective at onset
                if plan.speech_act == SpeechAct.APPROVE and pending_action is not None:
                    at, tid = pending_action
                    guard.grant(
                        ApprovalToken(
                            token_id=f"tok_{run_id}_{turn_index}",
                            action_type=at,
                            target_id=tid,
                            granted_media_time_ms=onset,
                            source_event_id=plan.id,
                        )
                    )
                    logger.emit(
                        EventType.APPROVAL_EVENT, Actor.USER, 0, onset, {"granted": at, "target_id": tid}
                    )
                elif plan.speech_act in (SpeechAct.REVOKE, SpeechAct.DENY) and pending_action is not None:
                    at, tid = pending_action
                    guard.revoke(at, tid, onset)
                    logger.emit(
                        EventType.APPROVAL_EVENT, Actor.USER, 0, onset, {"revoked": at, "target_id": tid}
                    )

        # Close the user turn so the model responds (the nudge path already did this; a deferred
        # barge is committed mid-agent-response instead).
        if drained and barge_next is None:
            await adapter.commit_audio()
            await adapter.create_response()

        # ---------------- AGENT BEAT ----------------
        new_act_keys: list[str] = []
        tool_calls_seen: list[str] = []
        approval_requests: list[str] = []
        commit_attempts: list[str] = []
        last_tool_results = []
        beat_transcript: list[str] = []
        beat_fired: list[str] = []
        muts_before = len(workspace.history)

        for _sub in range(_MAX_SUBRESPONSES):
            commit_wall = time.monotonic()
            resp = await sess.collect_response(barge=barge_next)
            if barge_next is not None and resp.barged:
                p = barge_next["plan"]
                yielded = resp.isl_ms is not None and resp.isl_ms <= 1500
                logger.emit(
                    EventType.DUPLEX_EVENT,
                    Actor.USER,
                    0,
                    resp.start_ms or sess.now_ms(),
                    {
                        "event_id": p.event_id,
                        "type": "MID_SPEECH_CORRECTION",
                        "isl_ms": resp.isl_ms,
                        "agent_yielded": yielded,
                        "onset_ms": resp.start_ms,
                    },
                )
                logger.emit(
                    EventType.USER_AUDIO_START,
                    Actor.USER,
                    0,
                    sess.now_ms(),
                    {
                        "text": next(
                            (t["text"] for t in reversed(transcript_log) if t.get("plan") == p.id), ""
                        ),
                        "duration_ms": int(round(1000 * len(barge_next["audio"]) / _SR)),
                        "barge": True,
                    },
                )
                sess.duplex_events.append(
                    {
                        "event_id": p.event_id,
                        "type": "MID_SPEECH_CORRECTION",
                        "isl_ms": resp.isl_ms if resp.isl_ms is not None else 0,
                        "agent_yielded": yielded,
                        "expected_floor_action": "YIELD",
                    }
                )
                # a barged REVOKE/DENY takes effect at interrupt onset
                if p.speech_act in (SpeechAct.REVOKE, SpeechAct.DENY) and pending_action is not None:
                    guard.revoke(pending_action[0], pending_action[1], sess.now_ms())
                    logger.emit(
                        EventType.APPROVAL_EVENT,
                        Actor.USER,
                        0,
                        sess.now_ms(),
                        {"revoked": pending_action[0], "target_id": pending_action[1]},
                    )
                barge_next = None
                # the model was cut off; let it react to the correction
                await adapter.commit_audio()
                await adapter.create_response()

            if resp.start_ms is not None:
                sess.rsl_samples.append(max(0, resp.start_ms - int((commit_wall - sess._t0) * 1000)))
                logger.emit(
                    EventType.AGENT_AUDIO_START,
                    Actor.AGENT,
                    0,
                    resp.start_ms,
                    {
                        "text": resp.transcript,
                        "duration_ms": (resp.end_ms - resp.start_ms) if resp.end_ms else 0,
                    },
                )
                if resp.end_ms:
                    logger.emit(EventType.AGENT_AUDIO_END, Actor.AGENT, 0, resp.end_ms, {})
            if resp.transcript:
                beat_transcript.append(resp.transcript)

            # dispatch tool calls (same as text runner)
            for tc in resp.tool_calls:
                import json as _json

                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = _json.loads(args) if args else {}
                    except Exception:  # noqa: BLE001
                        args = {}
                args = args or {}
                logger.emit(
                    EventType.TOOL_CALL, Actor.AGENT, 0, sess.now_ms(), {"tool": tc.get("name"), "args": args}
                )
                result = tools.call(tc.get("name"), args, sess.now_ms())
                tool_calls_seen.append(tc.get("name"))
                logger.emit(
                    EventType.TOOL_RESULT,
                    Actor.ENVIRONMENT,
                    0,
                    sess.now_ms(),
                    {
                        "tool": tc.get("name"),
                        "ok": result.ok,
                        "result": result.result,
                        "auth_decision": result.auth_decision,
                    },
                )
                last_tool_results.append({"tool": tc.get("name"), "ok": result.ok, "result": result.result})
                spec = lt.toolset.by_name(tc.get("name"))
                if spec and spec.is_commit and isinstance(result.result, dict):
                    commit_attempts.append(result.result.get("action_type", tc.get("name")))
                if isinstance(result.result, dict) and result.result.get("status") == "awaiting_approval":
                    pending_action = (result.result.get("requested_action"), result.result.get("target_id"))
                    approval_requests.append(result.result.get("requested_action") or tc.get("name"))
                for a in observer.observe_tool(result):
                    new_act_keys.append(a.key())
                await adapter.send_tool_result({"call_id": tc.get("call_id"), "output": result.result})

            # After recording, a REVISE correction may arm (the caller catches the value the model
            # just wrote). Fire the event engine now; if it arms a barge-in correction, pop that user
            # plan and inject it into the model's follow-up speech (send_tool_result already asked the
            # model to continue) so the interrupt is genuinely mid-speech and ISL is measurable.
            if resp.tool_calls:
                art_now = [f"{m.artifact_id}.{m.path}" for m in workspace.history[muts_before:]]
                fired = apply_events(new_act_keys, tool_calls_seen, art_now, commit_attempts)
                beat_fired.extend(fired)
                if barge_next is None and set(fired) & barge_event_ids:
                    bplan = flow.step(Observation(media_time_ms=sess.now_ms(), fired_events=fired))
                    if bplan is not None and bplan.floor_action in (
                        FloorAction.BARGE_IN,
                        FloorAction.URGENT_OVERRIDE,
                    ):
                        bvar = selector.select_variant(lt.realization_bank, bplan.id)
                        if bvar is not None:
                            barge_next = {"audio": _synth(bvar.text), "plan": bplan}
                            transcript_log.append(
                                {"speaker": "user", "plan": bplan.id, "text": bvar.text, "barge": True}
                            )
                            logger.emit(
                                EventType.USER_PLAN,
                                Actor.USER,
                                0,
                                sess.now_ms(),
                                {
                                    "plan_id": bplan.id,
                                    "speech_act": bplan.speech_act.value,
                                    "facts": bplan.fact_ids,
                                    "floor_action": bplan.floor_action.value,
                                },
                            )
                continue  # model will produce its next response (question / confirmation)
            break

        full_transcript = " ".join(beat_transcript).strip()
        if full_transcript:
            transcript_log.append({"speaker": "agent", "text": full_transcript})
            logger.emit(EventType.AGENT_TEXT, Actor.AGENT, 0, sess.now_ms(), {"text": full_transcript})
            for a in observer.observe_text(full_transcript):
                new_act_keys.append(a.key())
                logger.emit(
                    EventType.AGENT_ACT, Actor.AGENT, 0, sess.now_ms(), {"act": a.key(), "source": a.source}
                )

        art_muts = [f"{m.artifact_id}.{m.path}" for m in workspace.history[muts_before:]]

        # Fire any events not already caught mid-beat (agent_act / commit-triggered REVISE etc.).
        fired_now = beat_fired + apply_events(new_act_keys, tool_calls_seen, art_muts, commit_attempts)

        rep_counts: dict[str, int] = {}
        for k in new_act_keys:
            rep_counts[k] = sum(1 for x in observer.log if k in x.get("acts", []))
        obs = Observation(
            media_time_ms=sess.now_ms(),
            new_act_keys=new_act_keys,
            tool_calls=tool_calls_seen,
            commit_attempts=commit_attempts,
            approval_requests=approval_requests,
            fired_events=fired_now,
            artifact_mutations=art_muts,
            repetition_counts=rep_counts,
        )

    # ---- grade (identical to the other runners) ----
    ctx = GradingContext(
        workspace=workspace,
        state=state,
        tool_log=tools.call_log,
        events=logger.events,
        knowledge=kb,
        commit_guard=guard,
        observed_act_keys=set(flow._cum_acts),
        judge=judge,
    )  # noqa: SLF001
    pts = score_task(lt.gold, ctx)
    outcome = "SUCCESS" if pts.pts == 1 else "FAILURE"
    if fallback_events:
        outcome = "SIMULATOR_FALLBACK"
    logger.emit(EventType.RUN_END, Actor.HARNESS, 0, sess.now_ms(), {"outcome": outcome, "pts": pts.pts})

    # ---- metrics + audio ----
    from apex_voice.harness.audio import timeline_stats
    from apex_voice.scoring.duplex import duplex_metrics
    from apex_voice.scoring.latency import latency_metrics

    tstats = timeline_stats(sess.clips, _SR)
    dmet = duplex_metrics(sess.duplex_events)
    dmet["overlap_ms"] = tstats.overlap_ms
    dmet["overlap_ratio"] = tstats.overlap_ratio
    lmet = latency_metrics(logger.events, sess.rsl_samples, tstats)

    audio_dir = None
    if run_root is not None:
        write_workspace_outputs(run_root, workspace)
        audio_dir = str(run_root / "audio")
        write_wav(run_root / "audio" / "stereo.wav", assemble_stereo(sess.clips, _SR), _SR)
        write_wav(run_root / "audio" / "user.wav", isolated_channel(sess.clips, "user", _SR), _SR)
        write_wav(run_root / "audio" / "agent.wav", isolated_channel(sess.clips, "agent", _SR), _SR)
    logger.close()
    await adapter.close()

    return RealtimeRunResult(
        run_id=run_id,
        task_id=lt.spec.id,
        condition=condition,
        pts=pts,
        outcome=outcome,
        turns=turn_index,
        transcript=transcript_log,
        duplex=dmet,
        latency=lmet,
        audio_dir=audio_dir,
        event_log_path=str(log_path) if log_path else None,
        fallback_events=fallback_events,
    )


__all__ = ["run_realtime_task", "RealtimeRunResult"]
