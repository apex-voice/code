"""Credential and endpoint resolution from environment variables.

APEX-Voice never reads credentials from files. Every provider key and endpoint is resolved from
the process environment (optionally populated from a ``.env`` file by the CLI). See
``.env.example`` at the repository root for the full list of variables.
"""

from __future__ import annotations

import os


class MissingCredentialError(RuntimeError):
    """Raised when a required API key is not present in the environment."""


def env_first(*names: str, default: str | None = None) -> str | None:
    """Return the value of the first non-empty environment variable in ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return default


def require_env(*names: str, purpose: str) -> str:
    """Like :func:`env_first` but raise a helpful error if none of ``names`` is set."""
    value = env_first(*names)
    if not value:
        joined = " or ".join(names)
        raise MissingCredentialError(f"{purpose}: set {joined} in your environment (see .env.example).")
    return value


__all__ = ["MissingCredentialError", "env_first", "require_env"]
