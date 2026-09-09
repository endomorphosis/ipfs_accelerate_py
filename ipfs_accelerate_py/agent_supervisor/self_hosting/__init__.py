"""Bounded, evidence-only self-hosting experiment surface (PCCE-045)."""

from .experiment import ExperimentPlan, SelfHostingTask
from .harness import (
    EVIDENCE_SCHEMA,
    AttemptEvidence,
    SelfHostingQualificationHarness,
    TypedFailure,
    is_evidence_envelope,
)

__all__ = (
    "AttemptEvidence",
    "EVIDENCE_SCHEMA",
    "ExperimentPlan",
    "SelfHostingQualificationHarness",
    "SelfHostingTask",
    "TypedFailure",
    "is_evidence_envelope",
)
