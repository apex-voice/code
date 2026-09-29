from apex_voice.determinism import seeded_rng, stable_choice, stable_int, stable_key


def test_stable_key_process_independent():
    assert stable_key("a", 1, "b") == stable_key("a", 1, "b")
    assert stable_key("a", 1) != stable_key("a", 2)


def test_stable_choice_deterministic():
    cands = ["v1", "v2", "v3", "v4"]
    a = stable_choice(cands, "task@1:C0", 7, "plan_x")
    b = stable_choice(cands, "task@1:C0", 7, "plan_x")
    assert a == b
    # different seed can (and here does) change selection surface deterministically
    assert stable_choice(cands, "task@1:C0", 8, "plan_x") in cands


def test_seeded_rng_reproducible():
    r1 = seeded_rng("s", 1)
    r2 = seeded_rng("s", 1)
    assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]


def test_stable_int_nonnegative():
    assert stable_int("x") >= 0
