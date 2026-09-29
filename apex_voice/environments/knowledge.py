"""Knowledge / retrieval layer.

Provides a canonical ``KB_SEARCH(query, top_k)`` interface over a small local document set, with
gold-document tracking and distractor support. Retrieval quality is deliberately simple (token
overlap) — the benchmark measures *utilization* of retrieved knowledge, not retrieval architecture.
Every search is logged so gold-document recall can be scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    gold: bool = False  # part of the minimal gold evidence set
    distractor: bool = False


@dataclass
class KnowledgeBase:
    """A per-task document corpus with a deterministic search interface."""

    documents: dict[str, Document] = field(default_factory=dict)
    search_log: list[dict[str, Any]] = field(default_factory=list)

    def add(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Deterministic token-overlap search. Ties broken by doc_id for reproducibility."""
        q = _tokens(query)
        scored: list[tuple[float, str]] = []
        for doc in self.documents.values():
            overlap = len(q & _tokens(doc.title + " " + doc.text))
            if overlap:
                scored.append((overlap, doc.doc_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        hits = [
            {"doc_id": did, "title": self.documents[did].title, "score": sc, "text": self.documents[did].text}
            for sc, did in scored[:top_k]
        ]
        self.search_log.append({"query": query, "top_k": top_k, "hits": [h["doc_id"] for h in hits]})
        return hits

    def get(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    @property
    def gold_doc_ids(self) -> set[str]:
        return {d.doc_id for d in self.documents.values() if d.gold}

    def retrieved_gold_recall(self) -> float:
        """Fraction of gold docs that appeared in at least one search result."""
        gold = self.gold_doc_ids
        if not gold:
            return 1.0
        seen: set[str] = set()
        for entry in self.search_log:
            seen.update(entry["hits"])
        return len(gold & seen) / len(gold)


__all__ = ["Document", "KnowledgeBase"]
