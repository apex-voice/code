"""Deterministic user simulator.

Runtime pipeline: agent output -> act observer -> user flow engine -> user plan -> asset selector
-> speech synthesis. The benchmark logic runs live, while all benchmark-critical user language is
frozen in each task's offline-compiled realization bank.
"""
