"""User flow engine.

A deterministic state machine over observed agent behavior. On each step it: (1) evaluates the
current state's transitions against the observation surface; (2) on the first matching transition,
emits the associated :class:`UserPlan` and advances state; (3) enforces reveal-graph eligibility so
a plan can never express a fact before its reveal trigger fires.

Determinism: given the same seed and the same observation sequence, the engine emits the same
plans (CI gate: "fixed seed + same observations yields same semantic plan").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from apex_voice.schemas.flow import Action, FlowSpec, Trigger
from apex_voice.schemas.user import UserPlan
from apex_voice.user_sim.state import RevealContext, UserStateManager


@dataclass
class Observation:
    """The surface the flow engine reads each step."""

    media_time_ms: int = 0
    new_act_keys: list[str] = field(default_factory=list)  # acts observed since last step
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    artifact_mutations: list[str] = field(default_factory=list)  # "artifact.field"
    commit_attempts: list[str] = field(default_factory=list)  # action_types
    approval_requests: list[str] = field(default_factory=list)
    fired_events: list[str] = field(default_factory=list)
    agent_speech_continuous_ms: int = 0
    repetition_counts: dict[str, int] = field(default_factory=dict)  # act_key -> count
    world_predicates: set[str] = field(default_factory=set)  # names of currently-true predicates


class FlowError(Exception):
    pass


class UserFlowEngine:
    def __init__(
        self,
        flow: FlowSpec,
        user_state: UserStateManager,
        scenario_id: str,
        seed: int,
        reveal_enforcement: bool = True,
    ) -> None:
        self.flow = flow
        self.user = user_state
        self.scenario_id = scenario_id
        self.seed = seed
        self.reveal_enforcement = reveal_enforcement
        self.current = flow.start
        self._reveal = RevealContext(entered_states={flow.start})
        self._entered_emitted: set[str] = set()
        self.plan_log: list[UserPlan] = []
        self.leakage_violations: list[str] = []
        self._cum_acts: set[str] = set()
        # One-shot triggers already answered. An external event fires once and never un-fires; an
        # approval request is answered once, but re-arms if a *fresh* request appears later. Without
        # this, a self-loop transition (e.g. APPROVE/BACKCHANNEL) whose trigger persists across the
        # runner's drain would re-fire every drain iteration (up to the cap).
        self._consumed_triggers: set[str] = set()

    @staticmethod
    def _oneshot_token(trig: Trigger) -> str | None:
        if trig.external_event is not None:
            return f"evt:{trig.external_event}"
        if trig.approval_request is not None:
            return f"appr:{trig.approval_request}"
        return None

    @property
    def done(self) -> bool:
        return self.flow.states[self.current].terminal

    def _trigger_matches(self, trig: Trigger, obs: Observation) -> bool:
        if trig.agent_act is not None and trig.agent_act not in obs.new_act_keys:
            return False
        if trig.tool_call is not None and trig.tool_call not in obs.tool_calls:
            return False
        if trig.tool_result is not None and trig.tool_result not in obs.tool_results:
            return False
        if trig.artifact_mutation is not None and trig.artifact_mutation not in obs.artifact_mutations:
            return False
        if trig.commit_attempt is not None and trig.commit_attempt not in obs.commit_attempts:
            return False
        if trig.approval_request is not None and trig.approval_request not in obs.approval_requests:
            return False
        if trig.external_event is not None and trig.external_event not in obs.fired_events:
            return False
        if trig.world_predicate is not None and trig.world_predicate not in obs.world_predicates:
            return False
        if trig.media_time_ms_gte is not None and obs.media_time_ms < trig.media_time_ms_gte:
            return False
        if (
            trig.agent_speech_continuous_ms_gte is not None
            and obs.agent_speech_continuous_ms < trig.agent_speech_continuous_ms_gte
        ):
            return False
        if trig.repetition_count_gte is not None:
            if max(obs.repetition_counts.values(), default=0) < trig.repetition_count_gte:
                return False
        if trig.fact_unrevealed is not None and trig.fact_unrevealed in self.user.revealed:
            return False
        return True

    def _build_plan(self, action: Action, state_id: str) -> UserPlan:
        plan_id = action.plan_id or f"{state_id}:{action.act.value}"
        # Reveal enforcement: strip/deny facts that are not yet eligible.
        allowed_facts: list[str] = []
        for fid in action.facts:
            if not self.reveal_enforcement or self.user.may_reveal(fid, self._reveal):
                allowed_facts.append(fid)
            else:
                self.leakage_violations.append(f"{plan_id} attempted to reveal ineligible fact '{fid}'")
        for fid in allowed_facts:
            self.user.mark_revealed(fid)
        return UserPlan(
            id=plan_id,
            speech_act=action.act,
            fact_ids=allowed_facts,
            floor_action=action.floor_action,
            urgency=action.urgency,
            event_id=action.event_id,
            style=action.style,
        )

    def step(self, obs: Observation) -> UserPlan | None:
        """Advance the flow one step, returning a UserPlan to deliver (or None)."""
        # Update cumulative reveal context.
        self._cum_acts.update(obs.new_act_keys)
        self._reveal.observed_act_keys = set(self._cum_acts)
        self._reveal.fired_events.update(obs.fired_events)

        # Re-arm approval one-shots once their request is no longer pending (a genuinely new request
        # later can be answered again). Event one-shots are permanent — an event never un-fires.
        pending = {f"appr:{a}" for a in obs.approval_requests}
        self._consumed_triggers = {
            t for t in self._consumed_triggers if not t.startswith("appr:") or t in pending
        }

        # Emit on_enter for the current state exactly once upon entry.
        st = self.flow.states[self.current]
        if self.current not in self._entered_emitted and st.on_enter is not None:
            self._entered_emitted.add(self.current)
            plan = self._build_plan(st.on_enter, self.current)
            self.plan_log.append(plan)
            return plan
        self._entered_emitted.add(self.current)

        # Evaluate transitions in authored order; first match wins (deterministic).
        for tr in st.transitions:
            if self._trigger_matches(tr.when, obs):
                tok = self._oneshot_token(tr.when)
                if tok is not None and tok in self._consumed_triggers:
                    continue  # this event/approval was already answered — don't re-fire on drain
                if tok is not None:
                    self._consumed_triggers.add(tok)
                self.current = tr.next
                self._reveal.entered_states.add(self.current)
                new_st = self.flow.states[self.current]
                # Prefer the transition's own action; else the new state's on_enter.
                action = tr.do
                if action is not None:
                    plan = self._build_plan(action, tr.next)
                    self.plan_log.append(plan)
                    return plan
                if new_st.on_enter is not None:
                    self._entered_emitted.add(self.current)
                    plan = self._build_plan(new_st.on_enter, self.current)
                    self.plan_log.append(plan)
                    return plan
                return None
        return None


__all__ = ["UserFlowEngine", "Observation", "FlowError"]
