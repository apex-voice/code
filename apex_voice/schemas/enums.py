"""Controlled vocabularies for the 17 taxonomy layers.

These enums are the single source of truth for every categorical value used in task cards,
coverage reports, and paper tables. Free-text drift is a validation failure: the taxonomy
validator rejects any value outside these vocabularies.
"""

from __future__ import annotations

from enum import Enum


class WorkArchetype(str, Enum):
    """L1 — professional work archetype."""

    FORM_FILL = "FORM_FILL"
    INTAKE = "INTAKE"
    INTERVIEW = "INTERVIEW"
    DISCOVERY = "DISCOVERY"
    TROUBLESHOOT = "TROUBLESHOOT"
    NEGOTIATE = "NEGOTIATE"
    COORDINATE = "COORDINATE"
    FACILITATE = "FACILITATE"
    ADVISE = "ADVISE"
    INSPECT = "INSPECT"
    DOCUMENT = "DOCUMENT"
    PLAN = "PLAN"
    AUDIT = "AUDIT"
    COACH = "COACH"
    OPERATE = "OPERATE"
    HANDOFF = "HANDOFF"


class DelegationPattern(str, Enum):
    """L4 — delegation patterns."""

    DELEGATE = "DELEGATE"
    COMPLETE = "COMPLETE"
    REVISE = "REVISE"
    FOLLOW_THROUGH = "FOLLOW_THROUGH"
    APPROVE = "APPROVE"


class ArtifactClass(str, Enum):
    """L5 — artifact / work-product classes."""

    STRUCTURED_FORM = "STRUCTURED_FORM"
    CASE_RECORD = "CASE_RECORD"
    EVIDENCE_MATRIX = "EVIDENCE_MATRIX"
    CRM_RECORD = "CRM_RECORD"
    TICKET = "TICKET"
    WORK_ORDER = "WORK_ORDER"
    PLAN_CHECKLIST = "PLAN_CHECKLIST"
    SCHEDULE = "SCHEDULE"
    NEGOTIATION_RECORD = "NEGOTIATION_RECORD"
    TIMELINE = "TIMELINE"
    MEMO_REPORT = "MEMO_REPORT"
    MESSAGE_EMAIL = "MESSAGE_EMAIL"
    MULTI_ARTIFACT_BUNDLE = "MULTI_ARTIFACT_BUNDLE"


class AutonomyLevel(str, Enum):
    """L6 — autonomy / commit level."""

    A0_PREPARE_ONLY = "A0_PREPARE_ONLY"
    A1_DRAFT_CONFIRM = "A1_DRAFT_CONFIRM"
    A2_LOW_RISK_EXECUTE = "A2_LOW_RISK_EXECUTE"
    A3_APPROVAL_GATED_COMMIT = "A3_APPROVAL_GATED_COMMIT"
    A4_SPECIAL_REVIEW = "A4_SPECIAL_REVIEW"


class KnowledgeBurden(str, Enum):
    """L7 — knowledge burden."""

    K0_NONE = "K0_NONE"
    K1_SUPPLIED = "K1_SUPPLIED"
    K2_SEARCH_SMALL = "K2_SEARCH_SMALL"
    K3_MULTI_DOC_POLICY = "K3_MULTI_DOC_POLICY"
    K4_DISCOVER_CAPABILITY = "K4_DISCOVER_CAPABILITY"


class ToolBurden(str, Enum):
    """L8 — tool burden."""

    T0_NONE = "T0_NONE"
    T1_LIGHT = "T1_LIGHT"
    T2_MODERATE = "T2_MODERATE"
    T3_DISCOVERABLE = "T3_DISCOVERABLE"
    T4_MULTI_APP = "T4_MULTI_APP"


class DuplexPhenomenon(str, Enum):
    """L9 — duplex phenomenon vocabulary."""

    CLEAN_HANDOFF = "CLEAN_HANDOFF"
    BACKCHANNEL = "BACKCHANNEL"
    USER_BARGE_IN = "USER_BARGE_IN"
    AGENT_BARGE_IN = "AGENT_BARGE_IN"
    MID_SPEECH_CORRECTION = "MID_SPEECH_CORRECTION"
    CANCELLATION_REVOCATION = "CANCELLATION_REVOCATION"
    INTENT_SWITCH = "INTENT_SWITCH"
    CLARIFICATION = "CLARIFICATION"
    CONCURRENT_CRITICAL_INFO = "CONCURRENT_CRITICAL_INFO"
    DISTRACTOR_SPEECH = "DISTRACTOR_SPEECH"
    FALSE_YIELD_TRAP = "FALSE_YIELD_TRAP"
    URGENT_INTERRUPTION = "URGENT_INTERRUPTION"
    LONG_NARRATIVE = "LONG_NARRATIVE"
    DELAYED_CORRECTION = "DELAYED_CORRECTION"
    SUSPEND_RESUME = "SUSPEND_RESUME"
    MULTI_PARTY_OVERLAP = "MULTI_PARTY_OVERLAP"


class TemporalDynamics(str, Enum):
    """L10 — temporal dynamics."""

    D0_STATIC = "D0_STATIC"
    D1_MIDCALL_CHANGE = "D1_MIDCALL_CHANGE"
    D2_DELAYED_EVENT = "D2_DELAYED_EVENT"
    D3_SUSPEND_RESUME = "D3_SUSPEND_RESUME"
    D4_LONG_HORIZON = "D4_LONG_HORIZON"


class UserProfile(str, Enum):
    """L11 — user behavior profiles."""

    COOPERATIVE_STANDARD = "COOPERATIVE_STANDARD"
    NOVICE_UNCERTAIN = "NOVICE_UNCERTAIN"
    VERBOSE_NARRATIVE = "VERBOSE_NARRATIVE"
    AMBIGUOUS_UNDERSPECIFIED = "AMBIGUOUS_UNDERSPECIFIED"
    CORRECTION_PRONE = "CORRECTION_PRONE"
    DISTRACTED_TIME_PRESSURED = "DISTRACTED_TIME_PRESSURED"
    LOW_TECH_EXPERTISE = "LOW_TECH_EXPERTISE"
    DOMAIN_EXPERT = "DOMAIN_EXPERT"
    EMOTIONALLY_FRUSTRATED = "EMOTIONALLY_FRUSTRATED"
    MULTI_PARTY = "MULTI_PARTY"


class GradingMode(str, Enum):
    """L12 — grading modes."""

    G_STATE = "G_STATE"
    G_SCHEMA = "G_SCHEMA"
    G_EVIDENCE = "G_EVIDENCE"
    G_SEQUENCE = "G_SEQUENCE"
    G_POLICY = "G_POLICY"
    G_JUDGE = "G_JUDGE"
    G_HYBRID = "G_HYBRID"


class RiskTier(str, Enum):
    """L15 — risk / safety tier."""

    R0_ROUTINE = "R0_ROUTINE"
    R1_SENSITIVE_DATA_SIM = "R1_SENSITIVE_DATA_SIM"
    R2_CONSEQUENTIAL_ACTION = "R2_CONSEQUENTIAL_ACTION"
    R3_SPECIAL_REVIEW = "R3_SPECIAL_REVIEW"


class ArtifactOrigin(str, Enum):
    """L17 — artifact origin / sourcing."""

    APEX_NATIVE = "APEX_NATIVE"
    ADAPTED_OPEN_ASSET = "ADAPTED_OPEN_ASSET"
    PUBLIC_TEMPLATE_SYNTHETIC_CONTENT = "PUBLIC_TEMPLATE_SYNTHETIC_CONTENT"


# L2 economic function and L3 industry setting are extensible-but-normalized string vocabularies
# . We keep them as frozensets so the validator can warn on drift without
# blocking the addition of a genuinely new normalized value.
ECONOMIC_FUNCTIONS: frozenset[str] = frozenset(
    {
        "recruiting",
        "hr_operations",
        "sales",
        "customer_success",
        "technical_support",
        "it_helpdesk",
        "claims_operations",
        "insurance_operations",
        "healthcare_operations",
        "legal_operations",
        "finance_operations",
        "procurement",
        "project_management",
        "operations",
        "field_service",
        "facilities",
        "research",
        "consulting",
        "compliance",
        "executive_assistance",
    }
)

INDUSTRY_SETTINGS: frozenset[str] = frozenset(
    {
        "horizontal_enterprise",
        "software_saas",
        "financial_services",
        "insurance",
        "healthcare_ops",
        "professional_services",
        "retail_ecommerce",
        "travel_hospitality",
        "logistics",
        "manufacturing_field_ops",
        "public_service_non_emergency",
        "workplace_hr",
    }
)


__all__ = [
    "WorkArchetype",
    "DelegationPattern",
    "ArtifactClass",
    "AutonomyLevel",
    "KnowledgeBurden",
    "ToolBurden",
    "DuplexPhenomenon",
    "TemporalDynamics",
    "UserProfile",
    "GradingMode",
    "RiskTier",
    "ArtifactOrigin",
    "ECONOMIC_FUNCTIONS",
    "INDUSTRY_SETTINGS",
]
