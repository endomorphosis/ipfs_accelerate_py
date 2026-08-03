"""PTR-154: receipt deferred envelope and controller context reconstruction."""

from __future__ import annotations

import json

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.receipt import (
    DEFERRED_ISSUANCE_ENVELOPE_INTERFACE,
    MAX_DEFERRED_RETAINED_PUBLIC_BYTES,
    REQUIRED_DEFERRED_CONTEXT_PINS,
    DeferredIssuanceEnvelope,
    public_deferred_mapping,
    reconstruct_controller_context_from_receipt_public,
    reconstruct_deferred_request_from_public,
)


def _receipt() -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid="cid:execution",
        locator_cid="cid:locator",
        nonce="nonce:receipt-ctx",
        policy_cid="cid:policy",
    )


def _complete_public(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "interface": DEFERRED_ISSUANCE_ENVELOPE_INTERFACE,
        "receipt_cid": "cid:receipt",
        "execution_key_cid": "cid:execution",
        "candidate_context_cid": "cid:candidate",
        "policy_cid": "cid:policy",
        "statement_cid": "cid:statement",
        "circuit_cid": "cid:circuit",
        "verifying_key_cid": "cid:vk",
        "issuer_id": "issuer:test",
        "epoch": "epoch:1",
        "backend_id": "groth16",
        "proof_system_id": "groth16",
        "locator_cid": "cid:locator",
    }
    payload.update(overrides)
    return payload


def test_required_deferred_pins_match_v2_set() -> None:
    assert REQUIRED_DEFERRED_CONTEXT_PINS == (
        "receipt_cid",
        "execution_key_cid",
        "candidate_context_cid",
        "policy_cid",
        "statement_cid",
        "circuit_cid",
        "verifying_key_cid",
        "issuer_id",
        "epoch",
        "backend_id",
    )


def test_public_deferred_mapping_strips_private_and_keeps_pins() -> None:
    public = public_deferred_mapping(
        {
            **_complete_public(),
            "witness": "SECRET",
            "private_key": "nope",
            "extra_scalar": "allowed-if-known",  # unknown non-public field dropped
        }
    )
    assert public is not None
    assert "witness" not in public
    assert "private_key" not in public
    assert public["backend_id"] == "groth16"
    assert public["interface"] == DEFERRED_ISSUANCE_ENVELOPE_INTERFACE


def test_public_deferred_mapping_rejects_oversized_retained_hex() -> None:
    oversized = ("ab" * (MAX_DEFERRED_RETAINED_PUBLIC_BYTES + 1))
    assert (
        public_deferred_mapping(
            {
                "receipt_cid": "cid:receipt",
                "retained_receipt_bytes_hex": oversized,
            }
        )
        is None
    )


def test_public_deferred_mapping_rejects_oversized_required_pin() -> None:
    assert (
        public_deferred_mapping(
            {
                "receipt_cid": "x" * 300,
                "execution_key_cid": "cid:execution",
            }
        )
        is None
    )


def test_public_deferred_mapping_rejects_malformed_hex() -> None:
    assert (
        public_deferred_mapping(
            {
                "receipt_cid": "cid:receipt",
                "retained_candidate_context_bytes_hex": "zz",
            }
        )
        is None
    )
    assert (
        public_deferred_mapping(
            {
                "receipt_cid": "cid:receipt",
                "retained_receipt_bytes_hex": "abc",
            }
        )
        is None
    )


def test_deferred_envelope_from_mapping_and_completeness() -> None:
    envelope = DeferredIssuanceEnvelope.from_mapping(_complete_public())
    assert envelope is not None
    assert envelope.is_complete is True
    assert envelope.missing_required_pins() == ()
    partial = DeferredIssuanceEnvelope.from_mapping(
        {"receipt_cid": "cid:receipt", "execution_key_cid": "cid:execution"}
    )
    assert partial is not None
    assert partial.is_complete is False
    assert "backend_id" in partial.missing_required_pins()


def test_from_admitted_receipt_ignores_certificate_fill_in() -> None:
    receipt = _receipt()
    retained = receipt.canonical_bytes()
    certificate = {
        "statement_cid": "cid:from-cert",
        "circuit_cid": "cid:from-cert",
        "verifying_key_cid": "cid:from-cert",
        "issuer_id": "issuer:from-cert",
        "epoch": "epoch:from-cert",
        "backend_id": "groth16",
    }
    envelope = DeferredIssuanceEnvelope.from_admitted_receipt(
        receipt,
        retained_receipt_bytes=retained,
        certificate=certificate,
    )
    assert envelope is not None
    assert envelope.receipt_cid == receipt.receipt_id
    assert envelope.retained_receipt_bytes_hex == retained.hex()
    # Certificate must not fill issuance pins.
    assert envelope.statement_cid == ""
    assert envelope.circuit_cid == ""
    assert envelope.backend_id == ""
    assert envelope.is_complete is False


def test_from_admitted_receipt_accepts_controller_owned_extras() -> None:
    receipt = _receipt()
    envelope = DeferredIssuanceEnvelope.from_admitted_receipt(
        receipt,
        candidate_context_cid="cid:candidate",
        backend_id="groth16",
        extras={
            "statement_cid": "cid:statement",
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:vk",
            "issuer_id": "issuer:test",
            "epoch": "epoch:1",
            "proof_system_id": "groth16",
        },
    )
    assert envelope is not None
    assert envelope.is_complete is True
    assert envelope.backend_id == "groth16"
    assert envelope.statement_cid == "cid:statement"


def test_from_admitted_receipt_rejects_oversized_retained_bytes() -> None:
    receipt = _receipt()
    oversized = b"z" * (MAX_DEFERRED_RETAINED_PUBLIC_BYTES + 8)
    assert (
        DeferredIssuanceEnvelope.from_admitted_receipt(
            receipt,
            retained_receipt_bytes=oversized,
        )
        is None
    )


def test_reconstruct_deferred_request_rehashes_and_ignores_certificate() -> None:
    retained = json.dumps(
        {"k": "v", "pad": "q" * 40},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    certificate = {
        "statement_cid": "cid:from-cert",
        "backend_id": "groth16",
        "circuit_cid": "cid:from-cert",
    }
    public = reconstruct_deferred_request_from_public(
        {
            "receipt_cid": "cid:receipt",
            "execution_key_cid": "cid:execution",
            "candidate_context_cid": "cid:candidate",
            "policy_cid": "cid:policy",
        },
        retained_receipt_bytes=retained,
        certificate=certificate,
    )
    assert public is not None
    assert public["retained_receipt_bytes_hex"] == retained.hex()
    assert public.get("statement_cid") in ("", None) or "statement_cid" not in public
    assert public.get("backend_id") in ("", None) or "backend_id" not in public


def test_reconstruct_deferred_request_rejects_oversized_retained() -> None:
    oversized = b"x" * (MAX_DEFERRED_RETAINED_PUBLIC_BYTES + 1)
    assert (
        reconstruct_deferred_request_from_public(
            {"receipt_cid": "cid:receipt"},
            retained_candidate_context_bytes=oversized,
        )
        is None
    )


def test_reconstruct_controller_context_from_receipt_public() -> None:
    retained = b'{"label":"ctx"}'
    context, reason = reconstruct_controller_context_from_receipt_public(
        _complete_public(retained_receipt_bytes_hex=retained.hex()),
    )
    assert reason == ""
    assert context is not None
    assert context.is_complete is True
    assert context.receipt_cid == "cid:receipt"
    assert context.backend_id == "groth16"
    assert context.retained_receipt_bytes == retained
    assert context.may_authorize_skip is False
