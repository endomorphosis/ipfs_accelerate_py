"""EAAEF-015: distinct identities; only reverified receipts complete."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.handoff.admission import (
    admit_handoff,
    require_completion_eligible,
    HandoffAdmissionError,
)
from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
    AdmissionVerdict,
    HandoffIdentityError,
    TrustClass,
)


SHA = ["sha256:" + (ch * 64) for ch in "abcdef123"]


def _kwargs(**overrides):
    payload = {
        "request_id": SHA[0],
        "session_id": SHA[1],
        "raw_export_id": SHA[2],
        "normalized_stream_id": SHA[3],
        "trust_class": TrustClass.LOCALLY_REVERIFIED,
        "verdict": AdmissionVerdict.ADMITTED,
        "reason_code": "reverified",
        "objective_id": SHA[4],
        "context_id": SHA[5],
        "repository_id": SHA[6],
        "patch_ids": (SHA[7],),
        "created_at_ms": 1_700_000_000_000,
    }
    payload.update(overrides)
    return payload


def test_reverified_receipt_is_completion_eligible() -> None:
    receipt = admit_handoff(**_kwargs())
    assert receipt.completion_eligible is True
    assert receipt.verdict is AdmissionVerdict.ADMITTED
    require_completion_eligible(receipt)
    ids = {
        receipt.request_id,
        receipt.session_id,
        receipt.raw_export_id,
        receipt.normalized_stream_id,
        receipt.objective_id,
        receipt.context_id,
        receipt.repository_id,
        *receipt.patch_ids,
    }
    assert len(ids) == 8


def test_imported_exportable_cannot_complete() -> None:
    receipt = admit_handoff(
        **_kwargs(
            trust_class=TrustClass.IMPORTED_EXPORTABLE,
            verdict=AdmissionVerdict.PREVIEW_ONLY,
            reason_code="imported_preview",
        )
    )
    assert receipt.completion_eligible is False
    with pytest.raises(HandoffAdmissionError, match="reverified"):
        require_completion_eligible(receipt)


def test_imported_success_and_self_approval_fail() -> None:
    with pytest.raises(HandoffAdmissionError, match="imported success"):
        admit_handoff(**_kwargs(imported_success_claim=True))
    with pytest.raises(HandoffAdmissionError, match="self-approve"):
        admit_handoff(**_kwargs(worker_self_approved=True))


def test_truncated_and_duplicate_identities_fail() -> None:
    with pytest.raises(HandoffAdmissionError, match="truncated"):
        admit_handoff(**_kwargs(truncated=True))
    with pytest.raises(HandoffIdentityError):
        admit_handoff(**_kwargs(session_id=SHA[0]))


def test_independently_admitted_may_complete() -> None:
    receipt = admit_handoff(
        **_kwargs(
            trust_class=TrustClass.INDEPENDENTLY_ADMITTED,
            reason_code="independent_review",
        )
    )
    assert receipt.completion_eligible is True
