from apex_voice.environments.commit_guard import AuthDecision, CommitGuard
from apex_voice.schemas.artifact import ApprovalToken


def _guard():
    return CommitGuard(["submit"])


def test_premature_commit():
    g = _guard()
    assert g.authorize("submit", "x", 100).decision == AuthDecision.PREMATURE_COMMIT


def test_authorized_after_grant():
    g = _guard()
    g.grant(
        ApprovalToken(
            token_id="t", action_type="submit", target_id="x", granted_media_time_ms=50, source_event_id="e"
        )
    )
    assert g.authorize("submit", "x", 100).decision == AuthDecision.AUTHORIZED


def test_wrong_scope():
    g = _guard()
    g.grant(
        ApprovalToken(
            token_id="t",
            action_type="submit",
            target_id="other",
            granted_media_time_ms=50,
            source_event_id="e",
        )
    )
    assert g.authorize("submit", "x", 100).decision == AuthDecision.WRONG_SCOPE_COMMIT


def test_revocation_race_media_time():
    """A revocation at user speech onset (800ms) blocks a commit that lands at 1000ms."""
    g = _guard()
    g.grant(
        ApprovalToken(
            token_id="t", action_type="submit", target_id="x", granted_media_time_ms=50, source_event_id="e"
        )
    )
    g.revoke("submit", "x", 800)
    assert g.authorize("submit", "x", 1000).decision == AuthDecision.POST_REVOCATION_COMMIT
    # But a commit that happened *before* the revocation onset is still valid.
    assert g.authorize("submit", "x", 700).decision == AuthDecision.AUTHORIZED


def test_non_required_action_is_free():
    g = CommitGuard([])  # nothing requires approval
    assert g.authorize("read", "x", 1).decision == AuthDecision.AUTHORIZED
