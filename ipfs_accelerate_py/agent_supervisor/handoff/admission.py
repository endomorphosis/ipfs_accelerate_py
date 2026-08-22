"""Handoff trust, identity, privacy and retention admission (EAAEF-015).

Raw session, normalized stream, objective, context, repository and patch
identities stay distinct.  Only locally reverified or independently admitted
receipts may satisfy completion gates.  Imported history is provenance, never
authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .contracts import (
    AdmissionVerdict,
    HandoffAdmissionReceipt,
    HandoffBounds,
    HandoffContractError,
    HandoffIdentityError,
    HandoffTrustError,
    TrustClass,
)


ADMISSION_POLICY_ID: Final[str] = "eaaef-handoff-admission@1"


class HandoffAdmissionError(HandoffTrustError):
    """Handoff material is not eligible to satisfy a completion gate."""


def _distinct(pairs: Sequence[tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for name, identity in pairs:
        identity = str(identity or "").strip()
        if not identity:
            continue
        previous = seen.get(identity)
        if previous is not None and previous != name:
            raise HandoffIdentityError(
                f"{name} identity must be distinct from {previous}"
            )
        seen[identity] = name


def admit_handoff(
    *,
    request_id: str,
    session_id: str,
    raw_export_id: str,
    normalized_stream_id: str,
    trust_class: TrustClass | str,
    verdict: AdmissionVerdict | str = AdmissionVerdict.PREVIEW_ONLY,
    reason_code: str = "preview",
    objective_id: str = "",
    context_id: str = "",
    repository_id: str = "",
    patch_ids: Sequence[str] = (),
    imported_success_claim: bool = False,
    worker_self_approved: bool = False,
    truncated: bool = False,
    bounds: HandoffBounds | None = None,
    created_at_ms: int = 0,
    policy_id: str = ADMISSION_POLICY_ID,
) -> HandoffAdmissionReceipt:
    """Build a HandoffAdmissionReceipt@1.  Completion is fail-closed."""

    if truncated:
        raise HandoffAdmissionError("truncated exports cannot be admitted")
    if worker_self_approved:
        raise HandoffAdmissionError("workers cannot self-approve handoff admission")
    if imported_success_claim:
        raise HandoffAdmissionError("imported success claims cannot satisfy completion")
    trust = (
        trust_class
        if isinstance(trust_class, TrustClass)
        else TrustClass(str(trust_class))
    )
    outcome = (
        verdict
        if isinstance(verdict, AdmissionVerdict)
        else AdmissionVerdict(str(verdict))
    )
    _distinct(
        (
            ("request", request_id),
            ("session", session_id),
            ("raw_export", raw_export_id),
            ("normalized_stream", normalized_stream_id),
            ("objective", objective_id),
            ("context", context_id),
            ("repository", repository_id),
        )
    )
    for patch_id in patch_ids:
        if patch_id in {
            request_id,
            session_id,
            raw_export_id,
            normalized_stream_id,
            objective_id,
            context_id,
            repository_id,
        }:
            raise HandoffIdentityError("patch identity must be distinct from session artifacts")
    if outcome is AdmissionVerdict.ADMITTED and trust.imported:
        raise HandoffAdmissionError(
            "imported material cannot be admitted for completion without local reverify"
        )
    receipt = HandoffAdmissionReceipt(
        request_id=request_id,
        session_id=session_id,
        verdict=outcome,
        trust_class=trust,
        raw_export_id=raw_export_id,
        normalized_stream_id=normalized_stream_id,
        reason_code=reason_code,
        policy_id=policy_id,
        objective_id=objective_id,
        context_id=context_id,
        repository_id=repository_id,
        patch_ids=tuple(patch_ids),
        bounds=bounds or HandoffBounds(),
        created_at_ms=created_at_ms,
        completion_eligible=False,
    )
    if receipt.completion_eligible and not trust.may_satisfy_completion:
        raise HandoffAdmissionError("completion eligibility requires reverified trust")
    return receipt


def require_completion_eligible(receipt: HandoffAdmissionReceipt) -> HandoffAdmissionReceipt:
    """Raise unless this exact receipt may satisfy a completion gate."""

    if not isinstance(receipt, HandoffAdmissionReceipt):
        raise HandoffContractError("completion gate requires a HandoffAdmissionReceipt")
    if not receipt.completion_eligible:
        raise HandoffAdmissionError(
            "only locally reverified or independently admitted receipts may satisfy completion"
        )
    if receipt.verdict is not AdmissionVerdict.ADMITTED:
        raise HandoffAdmissionError("preview or quarantined receipts cannot complete")
    return receipt
