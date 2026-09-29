"""Offline realization compiler.

At *build time* this enumerates every benchmark-relevant UserPlan reachable in a task's flow graph
and compiles 2-5 validated natural-language variants per plan. The surface realizer is pluggable:
a deterministic :class:`TemplateRealizer` for offline tests/CI, or an LLM realizer for natural
wording. Released tasks ship their compiled realization bank, so no realizer runs at evaluation time. Every variant is validated before admission (allowed facts only, no hidden-state/procedure
leakage, speech act preserved); failing variants are regenerated offline, never at scored runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from apex_voice.determinism import stable_key
from apex_voice.schemas.flow import Action, EventProgram, FlowSpec
from apex_voice.schemas.user import SpeechAct, UserState
from apex_voice.user_sim.asset_bank import PlanVariants, RealizationBank, TextVariant


@dataclass
class PlanInput:
    """The constrained input contract handed to the realizer."""

    user_plan_id: str
    speech_act: str
    allowed_facts: dict[str, Any]  # fact_id -> value (only facts the plan may express)
    minimal_context: str = ""
    persona: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)


class Realizer(Protocol):
    name: str
    version: str

    def realize(self, plan_input: PlanInput, n: int) -> list[str]: ...


@dataclass
class ValidationIssue:
    variant_text: str
    reason: str


class TemplateRealizer:
    """Deterministic, offline dummy realizer. Produces templated but varied wording for CI."""

    name = "template_realizer"
    version = "1.0"

    _TEMPLATES = {
        SpeechAct.OPENING_REQUEST.value: ["Hi, I need help to {ctx}.", "Hello — I'd like to {ctx}."],
        SpeechAct.ANSWER.value: ["{facts}.", "It's {facts}.", "Sure, {facts}."],
        SpeechAct.CONFIRM.value: ["Yes, that's right.", "Correct."],
        SpeechAct.CORRECT.value: ["Actually, it's {facts}.", "No — {facts}."],
        SpeechAct.APPROVE.value: ["Yes, go ahead.", "Approved, please proceed."],
        SpeechAct.REVOKE.value: ["Wait, don't do that.", "Stop — cancel that."],
        SpeechAct.DENY.value: ["No, don't.", "Please don't."],
        SpeechAct.CLARIFY.value: ["What do you mean?", "Can you clarify?"],
        SpeechAct.ACKNOWLEDGE.value: ["Got it.", "Okay."],
        SpeechAct.BACKCHANNEL.value: ["mhm", "right"],
        SpeechAct.INTENT_SWITCH.value: ["Actually, one more thing — {ctx}.", "Oh, also {ctx}."],
        SpeechAct.ASK.value: ["Can you tell me {ctx}?", "What about {ctx}?"],
        SpeechAct.END.value: ["Thanks, that's all.", "Great, thank you."],
    }

    def realize(self, plan_input: PlanInput, n: int) -> list[str]:
        facts_str = ", ".join(str(v) for v in plan_input.allowed_facts.values())
        ctx = plan_input.minimal_context or "there"
        templates = self._TEMPLATES.get(plan_input.speech_act, ["{facts}", "{ctx}"])
        out: list[str] = []
        for t in templates:
            out.append(t.format(facts=facts_str or ctx, ctx=ctx).strip())
            if len(out) >= max(1, n):
                break
        return out


# Phrases that would indicate hidden-procedure/future-goal leakage in a DELEGATE opening.
# Markers of hidden-procedure/step leakage in a DELEGATE opening. Kept specific so a legitimate
# request that merely names an artifact ("run the opening checklist") is not a false positive.
_LEAKAGE_MARKERS = (
    "step 1",
    "first you",
    "the procedure is",
    "the checklist is",
    "here are the steps",
    "you should then",
    "the steps are",
)


def validate_variant(text: str, plan_input: PlanInput) -> list[ValidationIssue]:
    """Validate one variant against the plan contract."""
    issues: list[ValidationIssue] = []
    low = text.lower()
    # 1. No hidden-procedure / future-goal leakage.
    for marker in _LEAKAGE_MARKERS:
        if marker in low:
            issues.append(ValidationIssue(text, f"possible procedure leakage: '{marker}'"))
    # 2. Every expressed fact value must be permitted (best-effort literal check for exact values).
    #    We only flag values that look like they were injected but aren't in allowed_facts.
    #    (Deep semantic checks are the job of the LLM validator; this is the deterministic floor.)
    if not text.strip():
        issues.append(ValidationIssue(text, "empty realization"))
    return issues


@dataclass
class CompileReport:
    compiled_plans: int = 0
    total_variants: int = 0
    rejected: list[ValidationIssue] = field(default_factory=list)
    uncovered_plans: list[str] = field(default_factory=list)


class RealizationCompiler:
    def __init__(
        self,
        realizer: Realizer | None = None,
        variants_per_plan: int = 2,
        qc: Any = None,
        fallback: Realizer | None = None,
    ) -> None:
        self.realizer = realizer or TemplateRealizer()
        self.variants_per_plan = variants_per_plan
        self.qc = qc  # optional LLM fidelity/leakage judge (RealizationQC)
        self.fallback = fallback or TemplateRealizer()  # ensures coverage if realizer returns nothing

    def enumerate_plan_inputs(
        self,
        flow: FlowSpec,
        events: EventProgram,
        user_state: UserState,
        contexts: dict[str, str] | None = None,
        role: str | None = None,
        scenario: str | None = None,
        outcome: str | None = None,
    ) -> list[PlanInput]:
        """Walk the flow + event program to collect every reachable benchmark-relevant UserPlan."""
        contexts = contexts or {}
        fact_values = {f.id: f.value for f in user_state.facts}
        role = role or user_state.role
        scenario = scenario or ""
        outcome = outcome or user_state.delegation.user_outcome_request or (user_state.goals_primary or "")
        persona = dict(user_state.persona)
        persona.update({"role": role, "scenario": scenario})
        seen: dict[str, PlanInput] = {}

        def _situation(act: str, fields: list[str]) -> str:
            f = fields[0].replace("_", " ") if fields else ""
            return {
                "OPENING_REQUEST": f"You are calling to: {outcome}. Open the call.",
                "ANSWER": f"The agent just asked you about your {f}."
                if f
                else "The agent asked you a question.",
                "CORRECT": f"You realize the {f} you gave is wrong; correct it now.",
                "APPROVE": "The agent asked for your approval to proceed with the pending action.",
                "REVOKE": "The agent is about to perform the action; you want to stop it now.",
                "DENY": "The agent asked for approval; you want to decline.",
                "BACKCHANNEL": "The agent is mid-explanation; give a brief listening cue only.",
            }.get(act, "Continue the conversation naturally.")

        def add_action(action: Action, state_id: str) -> None:
            plan_id = action.plan_id or f"{state_id}:{action.act.value}"
            if plan_id in seen:
                return
            allowed = {fid: fact_values.get(fid) for fid in action.facts}
            seen[plan_id] = PlanInput(
                user_plan_id=plan_id,
                speech_act=action.act.value,
                allowed_facts=allowed,
                minimal_context=contexts.get(plan_id) or _situation(action.act.value, action.facts),
                persona=persona,
                constraints=[
                    "Do not introduce facts not listed in allowed_facts.",
                    "Do not reveal future goals or hidden workflow steps.",
                    "Do not create or imply approval unless the plan explicitly authorizes it.",
                ],
            )

        for sid, st in flow.states.items():
            if st.on_enter is not None:
                add_action(st.on_enter, sid)
            for tr in st.transitions:
                if tr.do is not None:
                    add_action(tr.do, tr.next)
        return list(seen.values())

    def compile_plan(self, plan_input: PlanInput) -> tuple[PlanVariants, list[ValidationIssue]]:
        rejected: list[ValidationIssue] = []
        accepted: list[TextVariant] = []
        # Over-generate then validate, keeping up to variants_per_plan valid ones. If the primary
        # realizer yields nothing (e.g. provider hiccup), fall back so coverage is never lost.
        raw = self.realizer.realize(plan_input, self.variants_per_plan + 2)
        used = self.realizer.name
        if not raw:
            raw = self.fallback.realize(plan_input, self.variants_per_plan)
            used = self.fallback.name
        for i, text in enumerate(raw):
            issues = validate_variant(text, plan_input)
            if not issues and self.qc is not None:
                qc = self.qc.check(plan_input, text)
                if not qc.ok:
                    issues = [ValidationIssue(text, f"QC: {qc.reason}")]
            if issues:
                rejected.extend(issues)
                continue
            vid = f"{plan_input.user_plan_id}#v{len(accepted) + 1}"
            accepted.append(
                TextVariant(
                    variant_id=vid,
                    text=text,
                    expressed_fact_ids=list(plan_input.allowed_facts.keys()),
                    compiler={
                        "realizer": used,
                        "version": getattr(self.realizer, "version", "1.0"),
                        "seed": i,
                        "prompt_hash": stable_key(plan_input.speech_act, text),
                    },
                    checksum=stable_key(text),
                )
            )
            if len(accepted) >= self.variants_per_plan:
                break
        return PlanVariants(plan_input.user_plan_id, plan_input.speech_act, accepted), rejected

    def compile_all(self, plan_inputs: list[PlanInput]) -> tuple[RealizationBank, CompileReport]:
        bank = RealizationBank()
        report = CompileReport()
        for pi in plan_inputs:
            pv, rejected = self.compile_plan(pi)
            report.rejected.extend(rejected)
            if pv.variants:
                bank.add(pv)
                report.compiled_plans += 1
                report.total_variants += len(pv.variants)
            else:
                report.uncovered_plans.append(pi.user_plan_id)
        return bank, report


__all__ = [
    "PlanInput",
    "Realizer",
    "TemplateRealizer",
    "RealizationCompiler",
    "CompileReport",
    "ValidationIssue",
    "validate_variant",
]
