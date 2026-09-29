"""Frozen semantic act-observer.

The deterministic regex/tool observer is the first line, but a real model's free-form phrasing often
cannot be robustly parsed. This LLM-backed classifier maps an agent utterance to zero or more acts
from the task's *fixed* act ontology (the act keys the flow can react to). It never sees hidden user
state or gold — only the agent's words and the closed list of candidate acts — so it cannot leak.
It is deliberately used only as a fallback/augmentation of the deterministic parser.
"""

from __future__ import annotations

import json
import re

from apex_voice.user_sim.agent_observer import ActPattern, AgentActObserver, ObservedAct


class LLMActObserver(AgentActObserver):
    """Augments the deterministic observer with an LLM classifier over a closed act ontology."""

    def __init__(
        self,
        patterns: list[ActPattern],
        tool_act_map: dict[str, str],
        candidate_acts: list[str],
        model_id: str = "gpt-4o-mini",
    ) -> None:
        super().__init__(patterns, tool_act_map)
        self.candidate_acts = candidate_acts
        self.model_id = model_id
        self._client = None

    def _ensure(self):
        if self._client is None:
            from apex_voice.llm import openai_chat_client

            self._client = openai_chat_client(timeout=30.0)
        return self._client

    def observe_text(self, text: str) -> list[ObservedAct]:
        # deterministic first
        acts = super().observe_text(text)
        found = {a.key() for a in acts}
        remaining = [c for c in self.candidate_acts if c not in found]
        if not text.strip() or not remaining:
            return acts
        try:
            client = self._ensure()
            prompt = (
                "You label a professional voice agent's utterance with the semantic acts it performs, "
                "chosen ONLY from this closed list (return a JSON list of the exact strings that apply, "
                "or [] if none):\n"
                + json.dumps(remaining)
                + "\n\nAgent utterance:\n"
                + text[:600]
                + "\n\nReturn only a JSON array of matching act strings."
            )
            resp = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content or "[]"
            m = re.search(r"\[.*\]", raw, re.S)
            labels = json.loads(m.group(0)) if m else []
        except Exception:  # noqa: BLE001 — fall back to deterministic result on any error
            labels = []
        for lab in labels:
            if lab in remaining:
                fld = lab.split(":", 1)[1] if ":" in lab else None
                acts.append(ObservedAct(lab.split(":", 1)[0], fld, raw=text[:60], source="classifier"))
        self.log.append({"kind": "llm_classify", "labels": labels})
        return acts


def candidate_acts_for(observer_patterns: list[ActPattern], tool_act_map: dict[str, str], flow) -> list[str]:
    """Collect the closed set of act keys the flow can react to (from patterns + flow triggers)."""
    acts: set[str] = set()
    for p in observer_patterns:
        acts.add(f"{p.act}:{p.field}" if p.field else p.act)
    for st in flow.states.values():
        for tr in st.transitions:
            if tr.when.agent_act:
                acts.add(tr.when.agent_act)
    return sorted(acts)


__all__ = ["LLMActObserver", "candidate_acts_for"]
