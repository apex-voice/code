"""Aggregate campaign runs into the APEX-Voice result tables.

Entry point: :func:`apex_voice.analysis.report.build_report` (CLI: ``apex-voice results``).
"""

from apex_voice.analysis.report import Report, build_report, write_report
from apex_voice.analysis.runs import RunRecord, load_campaign

__all__ = ["Report", "build_report", "write_report", "RunRecord", "load_campaign"]
