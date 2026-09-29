from apex_voice.tasks import load_task
from apex_voice.user_sim.asset_bank import AssetSelector, PlanVariants, RealizationBank, TextVariant
from apex_voice.user_sim.compiler import (
    PlanInput,
    RealizationCompiler,
    TemplateRealizer,
    validate_variant,
)
from apex_voice.user_sim.flow_engine import Observation, UserFlowEngine
from apex_voice.user_sim.state import UserStateManager
from tests.paths import TOY_FORM


def _fresh_flow(seed=1):
    lt = load_task(TOY_FORM)
    usm = UserStateManager(lt.user_state)
    return lt, UserFlowEngine(lt.flow, usm, "toy_form_001@1.0.0:C0:FROZEN_SYNTHETIC", seed)


def test_flow_deterministic_under_seed():
    _, f1 = _fresh_flow()
    _, f2 = _fresh_flow()
    obs_seq = [
        Observation(new_act_keys=[]),
        Observation(new_act_keys=["ASK:employee_id"]),
        Observation(new_act_keys=["ASK:trip_purpose"]),
    ]
    p1 = [f1.step(o) for o in obs_seq]
    p2 = [f2.step(o) for o in obs_seq]
    assert [p.id if p else None for p in p1] == [p.id if p else None for p in p2]
    assert p1[0].id == "open"


def test_flow_no_reveal_before_trigger():
    """A fact whose reveal trigger has not fired must not be expressed (non-oracularity)."""
    lt = load_task(TOY_FORM)
    usm = UserStateManager(lt.user_state)
    f = UserFlowEngine(lt.flow, usm, "sc", 1)
    # Opening emitted first (no facts).
    opening = f.step(Observation())
    assert opening.speech_act.value == "OPENING_REQUEST"
    # employee_id only becomes eligible after ASK:employee_id is observed.
    plan = f.step(Observation(new_act_keys=["ASK:employee_id"]))
    assert "employee_id" in plan.fact_ids
    assert not f.leakage_violations


def test_flow_blocks_ineligible_fact():
    """If a flow action tries to reveal a fact before its trigger, it's stripped + logged."""
    from apex_voice.schemas.flow import Action, FlowSpec, FlowState, Transition, Trigger
    from apex_voice.schemas.user import RevealRule, SpeechAct, UserFact, UserState

    us = UserState(
        user_id="u", facts=[UserFact(id="secret", value=42, reveal=RevealRule(agent_acts=["ASK:secret"]))]
    )
    flow = FlowSpec(
        start="A",
        states={
            "A": FlowState(
                transitions=[
                    Transition(
                        when=Trigger(agent_act="OPEN"),
                        do=Action(act=SpeechAct.ANSWER, facts=["secret"], plan_id="leak"),
                        next="B",
                    )
                ]
            ),
            "B": FlowState(terminal=True),
        },
    )
    f = UserFlowEngine(flow, UserStateManager(us), "sc", 1)
    plan = f.step(Observation(new_act_keys=["OPEN"]))  # trigger fires but 'secret' not eligible
    assert plan.fact_ids == []  # stripped
    assert f.leakage_violations


def test_oneshot_trigger_fires_once_under_drain():
    """A self-loop APPROVE/BACKCHANNEL on a persistent one-shot trigger must fire exactly once even
    when the same observation is re-presented (the runner drains self-loops). Regression: prior to
    the consumed-trigger guard, APPROVE re-fired every drain iteration (~24x)."""
    from apex_voice.schemas.flow import Action, FlowSpec, FlowState, Transition, Trigger
    from apex_voice.schemas.user import SpeechAct, UserState

    flow = FlowSpec(
        start="S",
        states={
            "S": FlowState(
                transitions=[
                    Transition(
                        when=Trigger(external_event="evt_bc"),
                        do=Action(act=SpeechAct.BACKCHANNEL, plan_id="bc"),
                        next="S",
                    ),
                    Transition(
                        when=Trigger(approval_request="submit"),
                        do=Action(act=SpeechAct.APPROVE, plan_id="approve"),
                        next="S",
                    ),
                ]
            ),
        },
    )
    f = UserFlowEngine(flow, UserStateManager(UserState(user_id="u", facts=[])), "sc", 1)
    obs = Observation(fired_events=["evt_bc"], approval_requests=["submit"])
    got = [p for _ in range(24) if (p := f.step(obs)) is not None]
    plan_ids = [p.id for p in got]
    assert plan_ids == ["bc", "approve"]  # each once, then None — no drain repetition

    # A genuinely fresh approval request later re-arms APPROVE (event stays one-shot).
    assert f.step(Observation(approval_requests=[])) is None
    reobs = Observation(approval_requests=["submit"])
    assert (p := f.step(reobs)) is not None and p.id == "approve"
    assert f.step(reobs) is None


def test_compiler_rejects_procedure_leakage():
    pi = PlanInput(user_plan_id="p", speech_act="OPENING_REQUEST", allowed_facts={}, minimal_context="ctx")
    issues = validate_variant("First you should fill step 1 of the checklist", pi)
    assert issues


def test_compiler_coverage_and_selection_determinism():
    lt = load_task(TOY_FORM)
    compiler = RealizationCompiler(TemplateRealizer(), variants_per_plan=2)
    inputs = compiler.enumerate_plan_inputs(lt.flow, lt.events, lt.user_state)
    bank, report = compiler.compile_all(inputs)
    assert report.uncovered_plans == []
    assert report.compiled_plans >= len(inputs)
    sel = AssetSelector("toy_form_001@1.0.0:C0:FROZEN_SYNTHETIC", 1)
    v1 = sel.select_variant(bank, "ans_emp")
    v2 = sel.select_variant(bank, "ans_emp")
    assert v1 is not None and v1.variant_id == v2.variant_id


def test_bank_jsonl_roundtrip(tmp_path):
    bank = RealizationBank()
    bank.add(PlanVariants("p1", "ANSWER", [TextVariant("p1#v1", "hello", ["f"])]))
    path = tmp_path / "bank.jsonl"
    bank.save_jsonl(path)
    loaded = RealizationBank.load_jsonl(path)
    assert loaded.get("p1").variants[0].text == "hello"
