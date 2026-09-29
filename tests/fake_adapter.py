"""A minimal in-process realtime adapter used to exercise the campaign/runner plumbing offline."""

from __future__ import annotations

import asyncio
from typing import Any


class SilentAdapter:
    """Accepts audio and immediately ends every response without speaking or calling tools."""

    model_id = "silent"
    fail_times = 0  # raise a transient error on the first N sessions (class-level counter)
    _failures = 0

    def __init__(self, **_: Any) -> None:
        self._q: asyncio.Queue = asyncio.Queue()

    async def start_session(self, config: dict[str, Any]) -> None:
        assert config["tools"] and config["turn_detection"] is None
        if SilentAdapter._failures < SilentAdapter.fail_times:
            SilentAdapter._failures += 1
            raise ConnectionError("server disconnected (1006)")

    async def send_audio(self, pcm_chunk: bytes, timestamp_ms: int = 0) -> None:
        pass

    async def commit_audio(self) -> None:
        pass

    async def create_response(self) -> None:
        await self._q.put({"type": "response.done"})

    async def cancel_response(self) -> None:
        pass

    async def send_text(self, text: str, timestamp_ms: int = 0) -> None:
        pass

    async def send_tool_result(self, result: dict[str, Any], timestamp_ms: int = 0) -> None:
        await self._q.put({"type": "response.done"})

    async def events(self):
        while True:
            yield await self._q.get()

    async def close(self) -> None:
        pass
