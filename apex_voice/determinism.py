"""Deterministic primitives shared across the harness.

Every reproducibility guarantee in APEX-Voice ultimately rests on two things:

1. a *stable* hash that does not depend on Python's per-process ``PYTHONHASHSEED``; and
2. a *seeded* RNG whose stream is a pure function of an explicit seed tuple.

These are used by the user-asset selector, seeded event branches,
and parametric task generation. Never use the builtin ``hash()`` or the global
``random`` module for benchmark-critical decisions.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def stable_key(*parts: object) -> str:
    """Return a process-independent hex digest for an ordered tuple of parts.

    Parts are stringified and joined with a delimiter that cannot appear in the
    canonical string form we use for IDs/seeds, then hashed with BLAKE2b.
    """
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


def stable_int(*parts: object) -> int:
    """A deterministic non-negative 64-bit integer derived from ``parts``."""
    return int(stable_key(*parts)[:16], 16)


def stable_choice(candidates: Sequence[T], *parts: object) -> T:
    """Deterministically choose one element of ``candidates`` from a seed tuple.

    This is the canonical asset-selector primitive: the same
    ``(scenario_id, simulator_seed, user_plan_id)`` always maps to the same variant.
    """
    if not candidates:
        raise ValueError("stable_choice requires a non-empty candidate sequence")
    return candidates[stable_int(*parts) % len(candidates)]


def seeded_rng(*parts: object) -> random.Random:
    """A ``random.Random`` whose stream is a pure function of the seed tuple.

    Used for seeded stochastic branches and parametric fixture generation. The
    stream is stable across processes and platforms because it is seeded from
    :func:`stable_int` rather than system entropy.
    """
    return random.Random(stable_int(*parts))
