"""User-simulator schemas: hidden state, facts, reveal rules, and the structured UserPlan.

The invariant that gives the benchmark its
validity is *non-oracularity*: the user engine holds hidden facts and future reveals, but the
surface realizer (and therefore the agent) only ever sees what the reveal graph permits at the
current step.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from apex_voice.schemas.enums import DelegationPattern, UserProfile


class SpeechAct(str, Enum):
    """User-side speech acts a flow rule may emit."""

    OPENING_REQUEST = "OPENING_REQUEST"
    ANSWER = "ANSWER"
    ASK = "ASK"
    CONFIRM = "CONFIRM"
    CORRECT = "CORRECT"
    APPROVE = "APPROVE"
    REVOKE = "REVOKE"
    DENY = "DENY"
    CLARIFY = "CLARIFY"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    BACKCHANNEL = "BACKCHANNEL"
    INTENT_SWITCH = "INTENT_SWITCH"
    END = "END"


class FloorAction(str, Enum):
    """How the user takes/holds the conversational floor when delivering a plan."""

    AFTER_FLOOR_RELEASE = "AFTER_FLOOR_RELEASE"
    BARGE_IN = "BARGE_IN"
    BACKCHANNEL = "BACKCHANNEL"
    OVERLAP_WITHOUT_FLOOR_CLAIM = "OVERLAP_WITHOUT_FLOOR_CLAIM"
    URGENT_OVERRIDE = "URGENT_OVERRIDE"
    RESUME = "RESUME"


class FactVisibility(str, Enum):
    PRIVATE_USER = "private_user"
    PUBLIC = "public"


class RevealRule(BaseModel):
    """When a fact may be revealed.

    A fact becomes eligible when ANY of ``agent_acts`` is observed, ANY of ``events`` fires, or
    ``on_state`` is entered. ``may_volunteer`` allows unsolicited disclosure.
    """

    model_config = ConfigDict(extra="forbid")

    may_volunteer: bool = False
    agent_acts: list[str] = Field(default_factory=list)  # e.g. "ASK:mailing_address"
    events: list[str] = Field(default_factory=list)  # event IDs from the event program
    on_state: list[str] = Field(default_factory=list)  # flow states


class UserFact(BaseModel):
    """A single user-held fact with reveal policy and correction versioning."""

    model_config = ConfigDict(extra="forbid")

    id: str
    value: Any
    source: str = "task_fixture"
    visibility: FactVisibility = FactVisibility.PRIVATE_USER
    truth_status: bool = True
    criticality: str = "required"  # required | optional | distractor
    supersedes: str | None = None  # id of the fact version this correction replaces
    reveal: RevealRule = Field(default_factory=RevealRule)


class ApprovalStateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pending_actions: list[str] = Field(default_factory=list)
    active_tokens: list[str] = Field(default_factory=list)
    revoked_tokens: list[str] = Field(default_factory=list)


class DelegationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patterns: list[DelegationPattern] = Field(default_factory=list)
    user_outcome_request: str | None = None
    # Guard against procedure leakage in DELEGATE tasks: the user engine
    # must never hold a step-by-step procedure the surface model could leak.
    procedural_steps_revealed: bool = False


class UserState(BaseModel):
    """The complete hidden user state.

    Only the user-side engine ever reads this. It must never contain gold grader labels.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    user_id: str
    role: str = "user"
    profile: UserProfile = UserProfile.COOPERATIVE_STANDARD
    persona: dict[str, Any] = Field(default_factory=dict)
    facts: list[UserFact] = Field(default_factory=list)
    goals_primary: str | None = None
    goals_secondary: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    delegation: DelegationInfo = Field(default_factory=DelegationInfo)
    approval_state: ApprovalStateSchema = Field(default_factory=ApprovalStateSchema)

    def fact(self, fact_id: str) -> UserFact | None:
        for f in self.facts:
            if f.id == fact_id:
                return f
        return None


class UserPlan(BaseModel):
    """A structured, wording-free plan emitted by the flow engine.

    Crucially contains *no surface text*: the deterministic asset selector maps
    ``(scenario_id, seed, id)`` to a prevalidated realization at runtime.
    """

    model_config = ConfigDict(extra="forbid")

    id: str  # stable user_plan_id used for asset selection
    speech_act: SpeechAct
    fact_ids: list[str] = Field(default_factory=list)
    floor_action: FloorAction = FloorAction.AFTER_FLOOR_RELEASE
    urgency: str = "normal"  # normal | high
    expected_agent_floor_action: str | None = None  # e.g. "YIELD"
    event_id: str | None = None
    style: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "SpeechAct",
    "FloorAction",
    "FactVisibility",
    "RevealRule",
    "UserFact",
    "ApprovalStateSchema",
    "DelegationInfo",
    "UserState",
    "UserPlan",
]
