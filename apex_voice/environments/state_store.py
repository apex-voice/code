"""Event-sourced environment state store.

The world state is a plain nested dict, but every mutation is recorded in an append-only log so
reset and replay produce identical snapshots. This is the environment analogue of the artifact
workspace history; the two together make a run fully recomputable.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class WorldMutation:
    seq: int
    path: str
    old_value: Any
    new_value: Any
    media_time_ms: int
    cause: str | None = None


def _get_path(root: dict[str, Any], path: str) -> Any:
    node: Any = root
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_path(root: dict[str, Any], path: str, value: Any) -> Any:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    old = node.get(parts[-1])
    node[parts[-1]] = value
    return old


class StateStore:
    """Seeded, event-sourced world state with deterministic reset."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._initial: dict[str, Any] = copy.deepcopy(initial or {})
        self._state: dict[str, Any] = copy.deepcopy(self._initial)
        self.history: list[WorldMutation] = []
        self._seq = 0

    def reset(self) -> None:
        """Deterministically reset to the initial fixture."""
        self._state = copy.deepcopy(self._initial)
        self.history = []
        self._seq = 0

    def get(self, path: str, default: Any = None) -> Any:
        val = _get_path(self._state, path)
        return default if val is None else val

    def set(self, path: str, value: Any, media_time_ms: int = 0, cause: str | None = None) -> None:
        old = _set_path(self._state, path, copy.deepcopy(value))
        self.history.append(
            WorldMutation(self._seq, path, copy.deepcopy(old), copy.deepcopy(value), media_time_ms, cause)
        )
        self._seq += 1

    def apply_updates(
        self, updates: dict[str, Any], media_time_ms: int = 0, cause: str | None = None
    ) -> None:
        for path, value in updates.items():
            self.set(path, value, media_time_ms, cause)

    def snapshot(self) -> dict[str, Any]:
        """Deep, order-stable copy of current state for grading/replay."""
        return copy.deepcopy(self._state)

    def snapshot_checksum(self) -> str:
        import hashlib

        blob = json.dumps(self.snapshot(), sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()

    def history_records(self) -> list[dict[str, Any]]:
        return [
            {
                "seq": m.seq,
                "path": m.path,
                "old": m.old_value,
                "new": m.new_value,
                "media_time_ms": m.media_time_ms,
                "cause": m.cause,
            }
            for m in self.history
        ]


__all__ = ["StateStore", "WorldMutation"]
