"""Scoring: predicates, Production Task Score (PTS), the semantic field judge, duplex and
latency metrics.

Grading is deterministic-first: PTS is computed from environment, artifact, and policy predicates;
the LLM judge is consulted only for free-text fields that fail the deterministic comparison.
"""
