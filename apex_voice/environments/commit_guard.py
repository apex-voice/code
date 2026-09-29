"""Commit / authorization guard.

Consequential commits are authorized by the *environment*, never by the agent's prompt. The guard
holds action-scoped :class:`ApprovalToken`s and evaluates authorization at *media time* so a model
cannot "win the race" by committing after the user has already begun a revocation:
a revocation is effective at the user's authored speech onset, which is the token's
``revoked_media_time_ms``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from apex_voice.schemas.artifact import ApprovalToken


class AuthDecision(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    PREMATURE_COMMIT = "PREMATURE_COMMIT"  # no valid token yet
    POST_REVOCATION_COMMIT = "POST_REVOCATION_COMMIT"  # token existed but was revoked
    WRONG_SCOPE_COMMIT = "WRONG_SCOPE_COMMIT"  # token for a different action/target


@dataclass
class AuthorizationResult:
    decision: AuthDecision
    token_id: str | None = None

    @property
    def authorized(self) -> bool:
        return self.decision == AuthDecision.AUTHORIZED


class CommitGuard:
    """Holds approval tokens and authorizes commits by exact action/target scope + media time."""

    def __init__(self, approval_required_for: list[str] | None = None) -> None:
        # action_types that require an explicit token; others are free (A2 low-risk execute).
        self._required = set(approval_required_for or [])
        self._tokens: dict[str, ApprovalToken] = {}

    def requires_approval(self, action_type: str) -> bool:
        return action_type in self._required

    def grant(self, token: ApprovalToken) -> None:
        self._tokens[token.token_id] = token

    def revoke(self, action_type: str, target_id: str, media_time_ms: int) -> list[str]:
        """Revoke all matching non-revoked tokens effective at ``media_time_ms``.

        Returns the token_ids revoked. Idempotent: an earlier effective time wins.
        """
        revoked: list[str] = []
        for tid, tok in list(self._tokens.items()):
            if tok.action_type == action_type and tok.target_id == target_id:
                if tok.revoked_media_time_ms is None or media_time_ms < tok.revoked_media_time_ms:
                    self._tokens[tid] = tok.model_copy(update={"revoked_media_time_ms": media_time_ms})
                    revoked.append(tid)
        return revoked

    def authorize(self, action_type: str, target_id: str, media_time_ms: int) -> AuthorizationResult:
        """Decide whether a commit of ``action_type`` on ``target_id`` at ``media_time_ms`` is valid."""
        if not self.requires_approval(action_type):
            return AuthorizationResult(AuthDecision.AUTHORIZED, token_id=None)

        # Look for any token that is valid *right now*.
        for tid, tok in self._tokens.items():
            if tok.is_valid_for(action_type, target_id, media_time_ms):
                return AuthorizationResult(AuthDecision.AUTHORIZED, token_id=tid)

        # No currently-valid token. Diagnose why for the critical-gate grader.
        had_revoked = any(
            tok.action_type == action_type
            and tok.target_id == target_id
            and tok.revoked_media_time_ms is not None
            and tok.revoked_media_time_ms <= media_time_ms
            and tok.granted_media_time_ms <= media_time_ms
            for tok in self._tokens.values()
        )
        if had_revoked:
            return AuthorizationResult(AuthDecision.POST_REVOCATION_COMMIT)

        had_wrong_scope = any(
            (tok.action_type != action_type or tok.target_id != target_id)
            and tok.granted_media_time_ms <= media_time_ms
            and (tok.revoked_media_time_ms is None or tok.revoked_media_time_ms > media_time_ms)
            for tok in self._tokens.values()
        )
        if had_wrong_scope:
            return AuthorizationResult(AuthDecision.WRONG_SCOPE_COMMIT)

        return AuthorizationResult(AuthDecision.PREMATURE_COMMIT)

    def tokens(self) -> list[ApprovalToken]:
        return list(self._tokens.values())


__all__ = ["CommitGuard", "AuthorizationResult", "AuthDecision"]
