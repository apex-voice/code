from apex_voice.scoring.duplex import duplex_metrics


def _yield_evt(yielded=True, isl=400, retained=True, typ="MID_SPEECH_CORRECTION"):
    return {
        "expected_floor_action": "YIELD",
        "agent_yielded": yielded,
        "isl_ms": isl,
        "retained_overlap": retained,
        "task_critical": True,
        "type": typ,
    }


def _continue_evt(yielded=False):
    return {
        "expected_floor_action": "CONTINUE",
        "agent_yielded": yielded,
        "task_critical": False,
        "type": "BACKCHANNEL",
    }


def test_perfect_duplex():
    m = duplex_metrics([_yield_evt(), _continue_evt(False)])
    assert m["fda"] == 1.0
    assert m["isl_ms"]["median"] == 400
    assert m["isl_ms"]["miss_rate"] == 0.0
    assert m["occr"] == 1.0
    assert m["ira"] == 1.0
    assert m["fyr"] == 0.0


def test_missed_yield_and_false_yield():
    m = duplex_metrics([_yield_evt(yielded=False), _continue_evt(yielded=True)])
    assert m["fda"] == 0.0
    assert m["isl_ms"]["miss_rate"] == 1.0
    assert m["occr"] == 1.0  # retained flag still set in this synthetic record
    assert m["ira"] == 0.0  # not yielded -> recovery fails
    assert m["fyr"] == 1.0


def test_non_retaining_agent_occr_zero():
    m = duplex_metrics([_yield_evt(retained=False)])
    assert m["occr"] == 0.0
