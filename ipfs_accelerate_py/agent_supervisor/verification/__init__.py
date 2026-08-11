"""Incremental-verification contracts.

This package boundary is side-effect free.  Final root-package lazy exports
are intentionally deferred to the release fan-in task.
"""

from __future__ import annotations

from .contracts import (
    PROOF_OBLIGATION_NOT_APPLICABLE_CID,
    TERMINAL_STATUS_PRECEDENCE,
    CacheReuseDecision,
    CacheReuseDisposition,
    CounterexampleReceipt,
    DiagnosticValueState,
    DirectExecutionObservation,
    ModelRoute,
    ModelRouteDecision,
    ProofReceipt,
    StaticAnalysisReceipt,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationBoundsError,
    VerificationBundle,
    VerificationCommitment,
    VerificationContractError,
    VerificationIdentityCompiler,
    VerificationIdentityError,
    VerificationPlan,
    VerificationReceipt,
    VerificationReceiptKey,
    VerificationReceiptKind,
    VerificationSummary,
    aggregate_terminal_status,
    build_verification_commitment,
)

__all__ = [
    "PROOF_OBLIGATION_NOT_APPLICABLE_CID",
    "TERMINAL_STATUS_PRECEDENCE",
    "CacheReuseDecision",
    "CacheReuseDisposition",
    "CounterexampleReceipt",
    "DiagnosticValueState",
    "DirectExecutionObservation",
    "ModelRoute",
    "ModelRouteDecision",
    "ProofReceipt",
    "StaticAnalysisReceipt",
    "TerminalStatus",
    "TestReceipt",
    "TypeCheckReceipt",
    "VerificationBoundsError",
    "VerificationBundle",
    "VerificationCommitment",
    "VerificationContractError",
    "VerificationIdentityCompiler",
    "VerificationIdentityError",
    "VerificationPlan",
    "VerificationReceipt",
    "VerificationReceiptKey",
    "VerificationReceiptKind",
    "VerificationSummary",
    "aggregate_terminal_status",
    "build_verification_commitment",
]
