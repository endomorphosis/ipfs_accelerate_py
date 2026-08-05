"""PTR-154: bounded controller-owned candidate context through publication.

The controller reconstructs exact receipt/execution/candidate/policy/statement/
circuit/key/issuer/epoch/backend pins from controller-owned or CID-rehashed
public bytes.  Certificate fields never fill missing pins; incomplete/malformed/
oversized/stale context yields receipt-only DEFERRED with no candidate write.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    TestExecutionKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
    CandidateExecutionContext,
)
from ipfs_accelerate_py.testing.proof_reuse.candidate_publication import (
    CONTROLLER_OWNED_V2_VERIFICATION_CONTEXT_INTERFACE,
    MAX_CONTROLLER_V2_RETAINED_BYTES,
    REQUIRED_CONTROLLER_V2_PIN_FIELDS,
    CandidatePublicationEnvelope,
    ControllerOwnedV2VerificationContext,
    admit_controller_owned_v2_context,
    reconstruct_controller_owned_v2_context,
    rehash_controller_owned_public_bytes,
)


def _complete_pins(**overrides: Any) -> dict[str, Any]:
    pins = {
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
    pins.update(overrides)
    return pins


def _canonical_blob(label: str, padding: int = 32) -> bytes:
    return json.dumps(
        {"label": label, "padding": "p" * padding},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _complete_context(**overrides: Any) -> ControllerOwnedV2VerificationContext:
    retained_receipt = _canonical_blob("receipt")
    retained_candidate = _canonical_blob("candidate", padding=64)
    retained_key = _canonical_blob("execution_key")
    payload = _complete_pins(
        retained_receipt_bytes_hex=retained_receipt.hex(),
        retained_candidate_context_bytes_hex=retained_candidate.hex(),
        retained_execution_key_bytes_hex=retained_key.hex(),
        source="test",
    )
    payload.update(overrides)
    context = ControllerOwnedV2VerificationContext.from_mapping(payload)
    assert context is not None
    return context


def test_controller_owned_context_interface_and_no_publication_authority() -> None:
    context = _complete_context()
    assert context.interface == CONTROLLER_OWNED_V2_VERIFICATION_CONTEXT_INTERFACE
    assert context.may_authorize_skip is False
    assert context.may_publish_candidate is False
    assert context.is_complete is True
    assert context.missing_required_pins() == ()
    public = context.to_public_mapping()
    assert public["may_authorize_skip"] is False
    assert public["may_publish_candidate"] is False
    assert public["is_complete"] is True
    for name in REQUIRED_CONTROLLER_V2_PIN_FIELDS:
        assert public[name]
    assert "witness" not in json.dumps(public).lower()


def test_required_pins_are_exactly_the_v2_set() -> None:
    assert REQUIRED_CONTROLLER_V2_PIN_FIELDS == (
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


def test_incomplete_context_reports_missing_pins() -> None:
    context = ControllerOwnedV2VerificationContext.from_mapping(
        {
            "receipt_cid": "cid:receipt",
            "execution_key_cid": "cid:execution",
            "source": "partial",
        },
        rehash=False,
    )
    assert context is not None
    assert context.is_complete is False
    missing = set(context.missing_required_pins())
    assert "candidate_context_cid" in missing
    assert "backend_id" in missing
    assert "receipt_cid" not in missing


def test_certificate_cannot_fill_missing_expected_pins() -> None:
    partial = {
        "receipt_cid": "cid:receipt",
        "execution_key_cid": "cid:execution",
        "candidate_context_cid": "cid:candidate",
        "policy_cid": "cid:policy",
        # statement/circuit/key/issuer/epoch/backend intentionally absent
    }
    certificate = {
        "statement_cid": "cid:from-cert",
        "circuit_cid": "cid:from-cert-circuit",
        "verifying_key_cid": "cid:from-cert-vk",
        "issuer_id": "issuer:from-cert",
        "epoch": "epoch:from-cert",
        "backend_id": "groth16",
        "receipt_cid": "cid:receipt",
    }
    context, reason = reconstruct_controller_owned_v2_context(
        partial,
        certificate=certificate,
        require_complete=True,
    )
    assert context is None
    assert reason.startswith("controller_context_incomplete")
    # Without require_complete the partial context is still incomplete.
    context, reason = reconstruct_controller_owned_v2_context(
        partial,
        certificate=certificate,
        require_complete=False,
    )
    assert context is not None
    assert reason == ""
    assert context.statement_cid == ""
    assert context.circuit_cid == ""
    assert context.backend_id == ""
    assert context.is_complete is False


def test_oversized_retained_bytes_are_rejected_not_truncated() -> None:
    oversized = b"x" * (MAX_CONTROLLER_V2_RETAINED_BYTES + 1)
    context = ControllerOwnedV2VerificationContext.from_mapping(
        {
            **_complete_pins(),
            "retained_receipt_bytes_hex": oversized.hex(),
        }
    )
    assert context is None
    context, reason = reconstruct_controller_owned_v2_context(
        _complete_pins(),
        retained_receipt_bytes=oversized,
    )
    assert context is None
    assert "oversized" in reason


def test_malformed_retained_hex_is_rejected() -> None:
    assert (
        ControllerOwnedV2VerificationContext.from_mapping(
            {
                **_complete_pins(),
                "retained_receipt_bytes_hex": "not-hex!!",
            }
        )
        is None
    )
    assert (
        ControllerOwnedV2VerificationContext.from_mapping(
            {
                **_complete_pins(),
                "retained_candidate_context_bytes_hex": "abc",  # odd length
            }
        )
        is None
    )


def test_rehash_before_use_records_byte_cids() -> None:
    receipt_bytes = _canonical_blob("receipt-rehash")
    context = _complete_context(
        retained_receipt_bytes_hex=receipt_bytes.hex(),
    )
    assert context.receipt_bytes_cid
    assert context.receipt_bytes_cid == rehash_controller_owned_public_bytes(
        receipt_bytes
    )
    assert context.candidate_context_bytes_cid
    assert context.execution_key_bytes_cid


def test_stale_or_substituted_pin_rejected_by_admit() -> None:
    context = _complete_context()
    admitted, reason = admit_controller_owned_v2_context(
        context,
        expected_pins={"candidate_context_cid": "cid:other-candidate"},
    )
    assert admitted is None
    assert "pin_mismatch" in reason


def test_round_trip_public_mapping_preserves_pins_and_bytes() -> None:
    original = _complete_context()
    wire = original.to_public_mapping()
    restored = ControllerOwnedV2VerificationContext.from_mapping(wire)
    assert restored is not None
    for name in REQUIRED_CONTROLLER_V2_PIN_FIELDS:
        assert restored.pin_value(name) == original.pin_value(name)
    assert restored.retained_receipt_bytes == original.retained_receipt_bytes
    assert (
        restored.retained_candidate_context_bytes
        == original.retained_candidate_context_bytes
    )
    deferred = restored.to_deferred_public_mapping()
    assert deferred["receipt_cid"] == original.receipt_cid
    assert deferred["backend_id"] == original.backend_id


def test_candidate_publication_envelope_projects_retained_components() -> None:
    receipt = TestPassReceipt(
        execution_key_cid="cid:execution",
        locator_cid="cid:locator",
        nonce="nonce:ctx",
        policy_cid="cid:policy",
    )
    key = TestExecutionKey(
        locator_cid="cid:locator",
        test_ast_cid="cid:ast",
        static_trace_root_cid="cid:static",
        runtime_trace_root_cid="cid:runtime",
        repository_forest_cid="cid:forest",
        environment_cid="cid:env",
        policy_cid="cid:policy",
        dependency_lock_cid="cid:deps",
        installed_distributions_cid="cid:dists",
        platform_cid="cid:platform",
        hardware_capability_cid="cid:hw",
    )
    # Prefer real key identity when the contract compiles it.
    try:
        execution_key_cid = key.execution_key_id
    except Exception:
        execution_key_cid = "cid:execution"
    descriptor = CandidateExecutionContext(
        locator_cid="cid:locator",
        execution_key_cid=execution_key_cid,
        pass_receipt_cid=receipt.receipt_id,
        repository_forest_cid="cid:forest",
        test_ast_cid="cid:ast",
        static_trace_root_cid="cid:static",
        runtime_trace_root_cid="cid:runtime",
        environment_cid="cid:env",
        policy_cid="cid:policy",
        dependency_lock_cid="cid:deps",
        installed_distributions_cid="cid:dists",
        platform_cid="cid:platform",
        capability_root_cid="cid:hw",
        component_cids={
            "execution_key": execution_key_cid,
            "pass_receipt": receipt.receipt_id,
            "policy": "cid:policy",
            "runtime_trace": "cid:runtime",
            "static_trace": "cid:static",
            "repository_forest": "cid:forest",
            "environment": "cid:env",
        },
        retained_at_ms=1,
    )
    components = {
        "execution_key": _canonical_blob("execution_key"),
        "pass_receipt": receipt.canonical_bytes(),
        "policy": _canonical_blob("policy"),
        "runtime_trace": _canonical_blob("runtime"),
        "static_trace": _canonical_blob("static"),
        "repository_forest": _canonical_blob("forest"),
        "environment": _canonical_blob("environment"),
    }
    envelope = CandidatePublicationEnvelope(
        descriptor=descriptor,
        component_bytes=components,
        component_cids=dict(descriptor.component_cids),
        execution_key=key,
        receipt=receipt,
        runtime_trace=SimpleNamespace(complete=True, cid="cid:runtime"),
        retained_descriptor_bytes=descriptor.canonical_bytes(),
        authoritative=True,
    )
    assert envelope.retained_component("pass_receipt") == components["pass_receipt"]
    assert envelope.required_components_present() is True
    context = envelope.controller_owned_v2_pins(
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:vk",
        issuer_id="issuer:test",
        epoch="epoch:1",
        backend_id="groth16",
    )
    assert context.receipt_cid == receipt.receipt_id
    assert context.policy_cid == "cid:policy"
    assert context.retained_receipt_bytes == components["pass_receipt"]
    assert context.retained_candidate_context_bytes == descriptor.canonical_bytes()
    assert context.statement_cid == "cid:statement"
    assert context.backend_id == "groth16"
    # Without issuance pins the projected context is incomplete.
    incomplete = envelope.controller_owned_v2_pins()
    assert incomplete.is_complete is False
    assert "backend_id" in incomplete.missing_required_pins()


def test_from_candidate_publication_admits_complete_pins() -> None:
    receipt = TestPassReceipt(
        execution_key_cid="cid:execution",
        locator_cid="cid:locator",
        nonce="nonce:admit",
        policy_cid="cid:policy",
    )
    key = TestExecutionKey(
        locator_cid="cid:locator",
        test_ast_cid="cid:ast",
        static_trace_root_cid="cid:static",
        runtime_trace_root_cid="cid:runtime",
        repository_forest_cid="cid:forest",
        environment_cid="cid:env",
        policy_cid="cid:policy",
        dependency_lock_cid="cid:deps",
        installed_distributions_cid="cid:dists",
        platform_cid="cid:platform",
        hardware_capability_cid="cid:hw",
    )
    execution_key_cid = key.execution_key_id
    descriptor = CandidateExecutionContext(
        locator_cid="cid:locator",
        execution_key_cid=execution_key_cid,
        pass_receipt_cid=receipt.receipt_id,
        repository_forest_cid="cid:forest",
        test_ast_cid="cid:ast",
        static_trace_root_cid="cid:static",
        runtime_trace_root_cid="cid:runtime",
        environment_cid="cid:env",
        policy_cid="cid:policy",
        dependency_lock_cid="cid:deps",
        installed_distributions_cid="cid:dists",
        platform_cid="cid:platform",
        capability_root_cid="cid:hw",
        component_cids={"pass_receipt": receipt.receipt_id, "policy": "cid:policy"},
        retained_at_ms=1,
    )
    envelope = CandidatePublicationEnvelope(
        descriptor=descriptor,
        component_bytes={
            "pass_receipt": receipt.canonical_bytes(),
            "execution_key": _canonical_blob("key"),
            "policy": _canonical_blob("policy"),
            "runtime_trace": _canonical_blob("rt"),
            "static_trace": _canonical_blob("st"),
            "repository_forest": _canonical_blob("rf"),
            "environment": _canonical_blob("env"),
        },
        component_cids=dict(descriptor.component_cids),
        execution_key=key,
        receipt=receipt,
        runtime_trace=SimpleNamespace(complete=True, cid="cid:runtime"),
        retained_descriptor_bytes=descriptor.canonical_bytes(),
        authoritative=True,
    )
    admitted = ControllerOwnedV2VerificationContext.from_candidate_publication(
        envelope,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:vk",
        issuer_id="issuer:test",
        epoch="epoch:1",
        backend_id="groth16",
    )
    assert admitted is not None
    assert admitted.is_complete is True
    assert admitted.may_publish_candidate is False


def test_serial_and_direct_reconstruction_share_contract() -> None:
    """Serial/direct-node path uses the same reconstruction helper as xdist."""

    pins = _complete_pins()
    retained = _canonical_blob("serial-receipt")
    context_a, reason_a = reconstruct_controller_owned_v2_context(
        pins,
        retained_receipt_bytes=retained,
    )
    from ipfs_accelerate_py.testing.proof_reuse.receipt import (
        reconstruct_controller_context_from_receipt_public,
    )

    context_b, reason_b = reconstruct_controller_context_from_receipt_public(
        pins,
        retained_receipt_bytes=retained,
    )
    assert reason_a == reason_b == ""
    assert context_a is not None and context_b is not None
    assert context_a.to_public_mapping() == context_b.to_public_mapping()


def test_empty_source_and_unsupported_types_fail_closed() -> None:
    assert reconstruct_controller_owned_v2_context(None)[0] is None
    assert reconstruct_controller_owned_v2_context(object())[0] is None
    assert ControllerOwnedV2VerificationContext.from_mapping([]) is None
