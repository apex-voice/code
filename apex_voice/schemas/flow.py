"""User-flow and event-program schemas.

The flow graph is a deterministic state machine over *observed agent behavior*. Each state has
``on_enter`` actions and guarded transitions. The event program layers semantic duplex events
and seeded follow-through events on top, with timing defined relative to observed agent behavior
 rather than absolute wall time.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from apex_voice.schemas.user import FloorAction, SpeechAct


class Trigger(BaseModel):
    """A guard on a flow transition or event.

    A trigger fires when all *populated* fields match the current observation. Unset fields are
    wildcards. Fields map onto the observation surface the flow engine maintains.
    """

    model_config = ConfigDict(extra="forbid")

    agent_act: str | None = None  # e.g. "ASK:mailing_address" or "CONFIRM:mailing_address"
    tool_call: str | None = None  # tool name observed being called
    tool_result: str | None = None
    artifact_mutation: str | None = None  # "artifact_id.field"
    commit_attempt: str | None = None  # action_type of a commit attempt
    approval_request: str | None = None
    world_predicate: str | None = None  # named predicate resolved against world state
    external_event: str | None = None  # event id from the event program
    media_time_ms_gte: int | None = None
    agent_speech_continuous_ms_gte: int | None = None
    repetition_count_gte: int | None = None
    # Robustness guard for permissive collection flows: transition fires only if this fact has NOT
    # yet been revealed (so an agent may ask fields in any order / bundle them without re-answering).
    fact_unrevealed: str | None = None


class Action(BaseModel):
    """The user action produced when a transition/state fires. Compiles into a UserPlan."""

    model_config = ConfigDict(extra="forbid")

    act: SpeechAct
    facts: list[str] = Field(default_factory=list)
    floor_action: FloorAction = FloorAction.AFTER_FLOOR_RELEASE
    urgency: str = "normal"
    event_id: str | None = None
    # Stable id used for asset selection; defaults derived from (state, act) if omitted.
    plan_id: str | None = None
    style: dict[str, Any] = Field(default_factory=dict)


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: Trigger
    do: Action | None = None
    next: str  # target state id


class FlowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    on_enter: Action | None = None
    transitions: list[Transition] = Field(default_factory=list)
    terminal: bool = False


class FlowSpec(BaseModel):
    """The user behavior program (``flow.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    start: str = "START"
    states: dict[str, FlowState]

    def model_post_init(self, __context: Any) -> None:
        if self.start not in self.states:
            raise ValueError(f"flow start state '{self.start}' not defined")
        for sid, st in self.states.items():
            for tr in st.transitions:
                if tr.next not in self.states:
                    raise ValueError(f"transition from '{sid}' targets unknown state '{tr.next}'")


class EventSpec(BaseModel):
    """A semantic duplex or follow-through event."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str  # DuplexPhenomenon value or a FOLLOW_THROUGH env-change type
    criticality: str = "task_critical"  # task_critical | nuisance
    trigger: Trigger = Field(default_factory=Trigger)
    # Timing window relative to observed agent speech (media ms). [earliest, latest].
    start_window_ms: tuple[int, int] | None = None
    delivery: FloorAction = FloorAction.AFTER_FLOOR_RELEASE
    expected_floor_action: str | None = None  # YIELD | CONTINUE | ...
    max_stop_latency_ms: int | None = None
    fact_updates: dict[str, Any] = Field(default_factory=dict)  # fact_id -> new value
    expected_repairs: list[str] = Field(default_factory=list)  # artifact/state paths
    # For FOLLOW_THROUGH: world mutations this event applies (state_store keys).
    world_updates: dict[str, Any] = Field(default_factory=dict)
    forbidden_tool_commits: list[str] = Field(default_factory=list)


class EventProgram(BaseModel):
    """The event program (``events.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    events: list[EventSpec] = Field(default_factory=list)


__all__ = [
    "Trigger",
    "Action",
    "Transition",
    "FlowState",
    "FlowSpec",
    "EventSpec",
    "EventProgram",
]
