"""API-level AttestationAuthorityBoundary@2 / VerifiedReceiptDispatch@2 tests (FVT-003).

Proves the stable Python facade rejects adversarial receipt inputs and that
attestation preparation never escalates into proof success.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_datasets_py.logic.backends.results import (
    ResultAuthority,
    ResultStatus,
    TheoremResult,
)
from ipfs_datasets_py.logic.bridge.proof_receipt_attestation import (
    AttestationBackendMode,
    AttestationBackendPolicy,
    build_trusted_receipt_from_backend_result,
)
from ipfs_datasets_py.logic.families.models import EvidenceAuthority
from ipfs_datasets_py.logic.ir_core.claims import FrozenMap
from ipfs_datasets_py.logic.ir_core.protocols import ExecutionBounds
from ipfs_datasets_py.logic.software_verification.receipts import LogicTranslationReceipt
from ipfs_datasets_py.logic.software_verification.translations import (
    CompilerBinding,
    PreservationClaim,
    PreservationKind,
)
from ipfs_datasets_py.logic.verification_api import (
    ATTESTATION_AUTHORITY_BOUNDARY_INTERFACE,
    VERIFIED_RECEIPT_DISPATCH_INTERFACE,
    LogicVerificationAPI,
    VerificationAuthority,
    VerificationStatus,
    attest_receipt,
    get_verification_api,
    verify_receipt,
)

TREE = "tree:api-boundary@1"
PROPERTY = "property:lease-safety"
ASSUMPTIONS = ("assumption:token-order",)
BOUNDS = {"timeout_ms": 500, "max_steps": 50}
TOOL = "solver.lean"
NOW = "2026-07-30T10:00:00Z"
EXPIRES = "2026-07-30T10:10:00Z"


def _trusted_receipt(**result_changes: Any):
    fields: dict[str, Any] = {
        "result_id": "result:api-boundary",
        "backend_id": TOOL,
        "backend_version": "4.19.0",
        "authority": ResultAuthority.THEOREM,
        "status": ResultStatus.PROVED,
        "assumptions": ASSUMPTIONS,
        "bounds": ExecutionBounds(timeout_ms=500, max_steps=50),
        "translation_ceiling": EvidenceAuthority.INDEPENDENTLY_CHECKABLE,
        "metadata": FrozenMap(
            {"bounds": dict(BOUNDS), "expires_at": EXPIRES, "issued_at": NOW}
        ),
    }
    fields.update(result_changes)
    source = TheoremResult(**fields)
    return build_trusted_receipt_from_backend_result(
        source,
        theorem_id="theorem:api-boundary",
        property_id=PROPERTY,
        translation_receipt_id="translation:api-boundary:v1",
        tree_id=TREE,
        policy_id="policy:api-boundary@1",
    )


def _policy(mode: AttestationBackendMode) -> AttestationBackendPolicy:
    return AttestationBackendPolicy(
        backend_id="backend:provekit",
        backend_version="0.2.0",
        circuit_id="circuit:receipt-binding",
        circuit_version="2.1.0",
        ceremony_id="ceremony:mpc-2026-07",
        crs_id="crs:powers-of-tau:28",
        proving_key_id="pk:receipt-binding:sha256-cafe",
        verification_key_id="vk:receipt-binding:sha256-beef",
        revocation_policy_id="revocation:production@1",
        backend_mode=mode,
        verification_key_expires_at="2030-01-01T00:00:00Z",
    )


def _translation_receipt() -> LogicTranslationReceipt:
    return LogicTranslationReceipt(
        source_identity="src:api-boundary",
        target_identity="tgt:api-boundary",
        source_family_id="first_order",
        source_family_version="1.0.0",
        target_family_id="smt",
        target_family_version="2.6",
        compilers=(
            CompilerBinding(
                compiler_id="compiler:api-boundary",
                compiler_version="1.0.0",
                implementation_identity="sha256:" + "a" * 64,
                configuration_identity="sha256:" + "b" * 64,
            ),
        ),
        preservation_claim=PreservationClaim(
            kind=PreservationKind.EXACT,
            preserved_property_ids=(PROPERTY,),
            permitted_result_classes=("proved", "disproved"),
            description="API boundary translation fixture",
        ),
        authority_ceiling=EvidenceAuthority.BOUNDED,
        assumptions=ASSUMPTIONS,
    )


@pytest.fixture
def api() -> LogicVerificationAPI:
    return get_verification_api()


def test_module_level_verify_receipt_rejects_empty_and_forged(api: LogicVerificationAPI) -> None:
    empty = verify_receipt(None)
    assert empty.status is VerificationStatus.INVALID
    assert empty.result["reason"] == "empty"

    forged = verify_receipt(
        {
            "receipt_id": "kernel:forged",
            "authority": "theorem",
            "kind": "kernel",
            "digest": "ff" * 32,
        }
    )
    assert forged.status is VerificationStatus.INVALID
    assert forged.authority is VerificationAuthority.NONE
    assert forged.result["reason"] == "forged-kernel"
    assert forged.result["dispatch"] == VERIFIED_RECEIPT_DISPATCH_INTERFACE


def test_api_rejects_full_adversarial_binding_matrix(api: LogicVerificationAPI) -> None:
    receipt = _trusted_receipt()
    baseline = {
        "tree_id": TREE,
        "property_id": PROPERTY,
        "assumptions": list(ASSUMPTIONS),
        "bounds": dict(BOUNDS),
        "tool_id": TOOL,
        "authority": "theorem",
        "now": NOW,
        "content_id": receipt.content_id,
        "source_result_digest": receipt.source_result_digest,
        "receipt_id": receipt.receipt_id,
    }

    ok = api.verify_receipt(receipt, baseline)
    assert ok.status is VerificationStatus.SUCCEEDED
    assert ok.authority is VerificationAuthority.THEOREM
    assert ok.result["valid"] is True

    adversarial = {
        "wrong-tree": {**baseline, "tree_id": "tree:mutated"},
        "wrong-property": {**baseline, "property_id": "property:mutated"},
        "wrong-assumption": {**baseline, "assumptions": ["assumption:mutated"]},
        "wrong-bound": {**baseline, "bounds": {"timeout_ms": 1}},
        "wrong-tool": {**baseline, "tool_id": "solver.cvc5"},
        "cross-authority": {**baseline, "authority": "monitor"},
        "stale": {**baseline, "now": "2099-01-01T00:00:00Z"},
    }
    for label, expectation in adversarial.items():
        response = api.verify_receipt(receipt.to_dict(), expectation)
        assert response.status is VerificationStatus.INVALID, label
        assert response.authority is VerificationAuthority.NONE, label
        assert response.result["valid"] is False, label
        joined = " ".join(response.diagnostics)
        assert any(
            token in joined
            for token in (
                label.split("-")[0],
                label,
                "wrong-",
                "cross-authority",
                "stale",
            )
        ), (label, response.diagnostics)


def test_translation_and_trusted_receipts_preserve_authority_ceiling(
    api: LogicVerificationAPI,
) -> None:
    trusted = api.verify_receipt(_trusted_receipt())
    assert trusted.status is VerificationStatus.SUCCEEDED
    assert trusted.authority is VerificationAuthority.THEOREM
    assert trusted.result["underlying_authority"] == "theorem"
    assert trusted.result["round_trip"]["schema_version"] == "trusted-proof-receipt/v1"

    translation = api.verify_receipt(_translation_receipt().to_dict())
    assert translation.status is VerificationStatus.SUCCEEDED
    # Translation evidence ceilings never become theorem authority.
    assert translation.authority is VerificationAuthority.BOUNDED
    assert translation.result["kind"] == "translation_receipt"
    assert translation.result["authority_ceiling"] == "bounded"


def test_attestation_authority_boundary_forbids_proof_success(
    api: LogicVerificationAPI,
) -> None:
    receipt = _trusted_receipt()

    simulated = api.attest_receipt(
        receipt,
        backend_mode="simulated",
        backend_policy=_policy(AttestationBackendMode.SIMULATED),
        issued_at=NOW,
        expires_at=EXPIRES,
        request_id="req:api-sim",
    )
    assert simulated.status is VerificationStatus.PARTIAL
    assert simulated.authority is VerificationAuthority.ATTESTATION
    assert simulated.result["proof_success"] is False
    assert simulated.result["authoritative"] is False
    assert simulated.result["simulated"] is True
    assert simulated.result["boundary"] == ATTESTATION_AUTHORITY_BOUNDARY_INTERFACE
    # Underlying theorem authority is preserved, not claimed as attestation proof.
    assert simulated.result["underlying_authority"] == "theorem"
    assert simulated.result["underlying_status"] == "proved"

    prepared = attest_receipt(
        receipt.to_dict(),
        backend_mode="cryptographic",
        backend_policy=_policy(AttestationBackendMode.CRYPTOGRAPHIC).to_dict(),
        issued_at=NOW,
        expires_at=EXPIRES,
    )
    assert prepared.status is VerificationStatus.SUCCEEDED
    assert prepared.authority is VerificationAuthority.ATTESTATION
    assert prepared.result["proof_success"] is False
    assert prepared.result["authoritative"] is False
    assert prepared.result["prepared"] is True
    assert "envelope" in prepared.result


def test_attestation_does_not_accept_translation_receipt_as_proof(
    api: LogicVerificationAPI,
) -> None:
    """Cross-schema authority substitution must fail closed."""

    translation = _translation_receipt()
    response = api.attest_receipt(
        translation.to_dict(),
        backend_mode="simulated",
        backend_policy=_policy(AttestationBackendMode.SIMULATED),
        issued_at=NOW,
        expires_at=EXPIRES,
    )
    assert response.status in {VerificationStatus.INVALID, VerificationStatus.ERROR}
    assert response.result.get("proof_success") is False
    assert response.authority in {
        VerificationAuthority.NONE,
        VerificationAuthority.ATTESTATION,
    }
