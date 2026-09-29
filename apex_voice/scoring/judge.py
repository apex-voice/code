"""Semantic field judge for free-text artifact grading.

The deterministic graders are the backbone; free-text prose fields (``FieldGrader.SEMANTIC``) get a
casefold fast-path and, on miss, a STRICT LLM judge. "Strict" per the benchmark decision: the agent's
value passes only if it captures ALL essential content of the reference AND introduces no contradicting
or unsupported specific facts — while ignoring case/punctuation/abbreviation/filler/equivalent phrasing.

Determinism/reproducibility: temperature 0 + an on-disk cache keyed by (version, field, pred, gold).
Offline-safe: if no judge is configured on the GradingContext, SEMANTIC falls back to casefold_exact,
so CI stays deterministic and network-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

JUDGE_VERSION = "artifact-aware-3"


def default_cache_path() -> Path:
    """Judge cache location: ``$APEX_VOICE_JUDGE_CACHE`` or ``~/.cache/apex_voice/judge_cache.json``."""
    env = os.environ.get("APEX_VOICE_JUDGE_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "apex_voice" / "judge_cache.json"


_PROMPT = (
    "You grade ONE field of a professional work artifact a voice agent produced (from a SPOKEN "
    "conversation), against a reference (gold) value. Decide whether the agent's value denotes the SAME "
    "thing as the reference, the way a reasonable professional reviewer would. Reward substance; forgive "
    "surface form (values were dictated aloud, so separators/case/formatting differ).\n"
    "PASS if they refer to the same fact/decision/answer/entity — even if TERSER, omitting secondary "
    "detail, adding consistent detail, different wording, abbreviations/expansions ('VP Eng'=='VP of "
    "Engineering', 'acct'=='act'=='account'), approximations of the same number ('~20'=='about 20'=="
    "'20'), affirmative/status phrasing ('Yes'=='filed', 'done'=='completed'), or the same action "
    "(gold 'cleared cache and retried' vs 'cleared browser cache and cookies').\n"
    "IDENTIFIERS especially: ignore case, separators (- _ space .), leading zeros, and omitted/added "
    "type-prefixes — these PASS: 'CC4419'=='CC-4419', 'act88'=='acct_88', 'INV 771'=='INV-771', "
    "'ref 3391'=='REF-3391', 'HVAC-007'=='HVAC-7', '2201'=='JOB-2201', 'OPS19'=='OPS-19', "
    "'acct-5521'=='acct_5521'.\n"
    "FAIL only if the agent's value: (1) is a DIFFERENT specific value/number/name/date/amount/decision "
    "or a DIFFERENT identifier (e.g. 'CC4419' vs 'CC5500', 'INV-771' vs 'INV-772', gold 'refund to card' "
    "vs 'store credit', gold 'prorated add-on' vs 'duplicate charge'); or (2) misses the reference's core "
    "meaning entirely; or (3) is empty.\n"
    "Use the other-fields context only to disambiguate; grade THIS field. When plausibly equivalent, PASS.\n\n"
    "Field: {field}\nContext: {ctx}\n"
    "Reference (gold): {gold!r}\nAgent wrote: {pred!r}\n\n"
    'Return strict JSON: {{"pass": true|false, "why": "<short>"}}'
)


class SemanticFieldJudge:
    """Callable ``(field, pred, gold, context) -> bool``. Strict, cached, temperature 0."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        cache_path: Path | None | str = "default",
        version: str = JUDGE_VERSION,
    ) -> None:
        """``cache_path``: ``"default"`` -> :func:`default_cache_path`; ``None`` disables caching."""
        if cache_path == "default":
            cache_path = default_cache_path()
        elif cache_path is not None:
            cache_path = Path(cache_path)
        self.model_id = model_id
        self.version = version
        self.cache_path = cache_path
        self._client = None
        self.errors = 0  # judge calls that failed (counted as conservative FAILs, not cached)
        self._cache: dict[str, bool] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except Exception:  # noqa: BLE001
                self._cache = {}

    def _key(self, field: str, pred: Any, gold: Any) -> str:
        return hashlib.md5(f"{self.version}|{field}|{pred}|{gold}".encode()).hexdigest()

    def _client_ok(self):
        if self._client is None:
            from apex_voice.llm import openai_chat_client

            self._client = openai_chat_client(timeout=45.0)
        return self._client

    def __call__(self, field: str, pred: Any, gold: Any, context: str = "") -> bool:
        if not str(pred).strip():
            return False
        k = self._key(field, pred, gold)
        if k in self._cache:
            return self._cache[k]
        prompt = _PROMPT.format(field=field, ctx=context, gold=gold, pred=pred)
        try:
            r = self._client_ok().chat.completions.create(
                model=self.model_id,
                temperature=0.0,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            m = re.search(r"\{.*\}", r.choices[0].message.content or "{}", re.S)
            ok = bool(json.loads(m.group(0)).get("pass")) if m else False
        except Exception:  # noqa: BLE001 - never let judge infra fail a run; conservative FAIL
            self.errors += 1
            return False
        self._cache[k] = ok
        self._flush()
        return ok

    def _flush(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self._cache))
            os.replace(tmp, self.cache_path)  # atomic: concurrent campaigns never see a torn file
        except Exception:  # noqa: BLE001
            pass


def get_default_judge(**kw: Any) -> SemanticFieldJudge:
    return SemanticFieldJudge(**kw)


__all__ = ["SemanticFieldJudge", "get_default_judge", "default_cache_path", "JUDGE_VERSION"]
