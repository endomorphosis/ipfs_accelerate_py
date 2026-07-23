from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.artifact_store import (
    PROOF_ATTESTATION_KIND,
    query_proof_attestations,
    raw_ipfs_cid,
    read_proof_attestation_artifact,
    write_proof_attestation_artifact,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_cache import (
    CacheLookupStatus,
    CacheRejectionReason,
    FormalVerificationCache,
    build_proof_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof_attestation import (
    AttestationBackendPolicy,
    AttestationValidationError,
    PersistedAttestationRecord,
    PrivateAttestationWitness,
    REQUIRED_BACKEND_TEST_CASES,
    build_persisted_attestation_record,
    evaluate_backend_health,
    execute_cryptographic_attestation,
    prepare_receipt_attestation,
    public_artifact_contains,
    reproduce_attestation_verification,
)

NOW = "2026-07-23T12:00:00Z"
EXPIRES = "2026-07-23T12:05:00Z"


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=1_000,
        memory_bytes=1_000_000,
        max_processes=1,
        network_allowed=False,
    )


def _receipt() -> ProofReceipt:
    obligation_id = "obligation:attestation-store"
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel-proof",
        subject_id=obligation_id,
        verifier_id="kernel:lean@4.19",
        independent=True,
    )
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id="plan:attestation-store",
        attempt_id="attempt:kernel",
        repository_id="repo:test",
        repository_tree_id="tree:attestation-store",
        ast_scope_ids=("scope:attestation-store",),
        premise_ids=("premise:public",),
        translator_id="translator:test@1",
        solver_id="solver:test@1",
        kernel_id="kernel:lean@4.19",
        toolchain_id="toolchain:test@1",
        theorem_registry_id="registry:test@1",
        policy_id="policy:formal@1",
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        kernel_receipt_id="kernel-receipt:lean:immutable",
    )


def _key():
    receipt = _receipt()
    return build_proof_cache_key(
        obligation=receipt.obligation_id,
        premises=receipt.premise_ids,
        translator=receipt.translator_id,
        solver=receipt.solver_id,
        kernel=receipt.kernel_id,
        toolchain=receipt.toolchain_id,
        theorem_registry=receipt.theorem_registry_id,
        policy=receipt.policy_id,
        resource_budget=receipt.resource_budget.to_dict(),
        candidate_tree=receipt.repository_tree_id,
    )


def _policy() -> AttestationBackendPolicy:
    return AttestationBackendPolicy(
        backend_id="backend:provekit",
        backend_version="0.2.0",
        circuit_id="circuit:receipt-binding",
        circuit_version="2.1.0",
        public_input_schema_id="schema:receipt-public-inputs",
        public_input_schema_version="1.1.0",
        verification_key_id="vk:receipt-binding:sha256-beef",
        verification_key_version="ceremony-2026-07",
        verification_key_expires_at="2030-01-01T00:00:00Z",
    )


def _verification(*, secret: str = "private-witness-REF-266"):
    policy = _policy()
    health = evaluate_backend_health(
        policy,
        configured=True,
        available=True,
        outcomes={case: True for case in REQUIRED_BACKEND_TEST_CASES},
        evaluated_at=NOW,
    )
    request = prepare_receipt_attestation(
        _receipt(),
        backend_policy=policy,
        witness=PrivateAttestationWitness({"private_premise": secret}),
    )
    return execute_cryptographic_attestation(
        request,
        backend_health=health,
        prover=lambda _request: {
            "proof_artifact_id": "artifact:zkp:public",
            "proof_digest": "sha256:public-proof-digest",
        },
        verifier=lambda _envelope: True,
        prover_id="prover:provekit@0.2.0",
        verifier_id="verifier:provekit@0.2.0",
    )


def _record() -> PersistedAttestationRecord:
    return build_persisted_attestation_record(
        _receipt(),
        _verification(),
        created_at=NOW,
        expires_at=EXPIRES,
    )


def test_record_binds_receipt_public_inputs_backend_key_policy_and_expiry() -> None:
    receipt = _receipt()
    record = _record()
    payload = record.to_public_artifact()
    statement = record.envelope.statement

    assert record.proof_receipt_id == receipt.receipt_id
    assert record.kernel_receipt_id == receipt.kernel_receipt_id
    assert statement.receipt_id == receipt.receipt_id
    assert payload["public_input_digest"] == statement.public_input_digest
    assert payload["backend_policy_id"] == _policy().policy_id
    assert payload["formal_policy_id"] == receipt.policy_id
    assert payload["backend_id"] == _policy().backend_id
    assert payload["circuit_id"] == _policy().circuit_id
    assert payload["verification_key_id"] == _policy().verification_key_id
    assert payload["expires_at"] == EXPIRES
    assert PersistedAttestationRecord.from_dict(payload) == record
    assert record.effective_assurance_at(NOW) is AssuranceLevel.ATTESTED
    assert (
        record.effective_assurance_at(EXPIRES)
        is AssuranceLevel.KERNEL_VERIFIED
    )


def test_record_requires_an_immutable_kernel_receipt_reference() -> None:
    receipt = ProofReceipt(**{**_receipt().__dict__, "kernel_receipt_id": ""})
    policy = _policy()
    health = evaluate_backend_health(
        policy,
        configured=True,
        available=True,
        outcomes={case: True for case in REQUIRED_BACKEND_TEST_CASES},
        evaluated_at=NOW,
    )
    request = prepare_receipt_attestation(
        receipt,
        backend_policy=policy,
        witness=PrivateAttestationWitness({"private_premise": "secret"}),
    )
    verification = execute_cryptographic_attestation(
        request,
        backend_health=health,
        prover=lambda _request: {
            "proof_artifact_id": "artifact:zkp:public",
            "proof_digest": "sha256:public-proof-digest",
        },
        verifier=lambda _envelope: True,
        prover_id="prover:provekit@0.2.0",
        verifier_id="verifier:provekit@0.2.0",
    )
    with pytest.raises(AttestationValidationError, match="kernel receipt identity"):
        build_persisted_attestation_record(
            receipt,
            verification,
            created_at=NOW,
            expires_at=EXPIRES,
        )


@pytest.mark.parametrize(
    "field",
    (
        "proof_receipt_id",
        "public_input_digest",
        "backend_policy_id",
        "formal_policy_id",
        "backend_id",
        "backend_version",
        "circuit_id",
        "circuit_version",
        "public_input_schema_id",
        "public_input_schema_version",
        "verification_key_id",
        "verification_key_version",
        "envelope_id",
        "verification_id",
        "expires_at",
    ),
)
def test_record_rejects_every_forged_bound_identity(field: str) -> None:
    payload = _record().to_public_artifact()
    payload[field] = (
        "2026-07-23T12:04:00Z" if field == "expires_at" else f"forged:{field}"
    )

    with pytest.raises(AttestationValidationError, match="binding|identity"):
        PersistedAttestationRecord.from_dict(payload)


def test_public_reverification_is_reproducible_and_fails_closed() -> None:
    record = _record()
    seen = []

    reproduced = reproduce_attestation_verification(
        record.to_public_artifact(),
        verifier=lambda envelope: seen.append(envelope.envelope_id) or True,
        checked_at=NOW,
        receipt=_receipt(),
        backend_policy=_policy(),
    )
    rejected = reproduce_attestation_verification(
        record,
        verifier=lambda _envelope: False,
        checked_at=NOW,
    )

    assert reproduced.authoritative
    assert seen == [record.envelope_id]
    assert not rejected.authoritative
    with pytest.raises(AttestationValidationError, match="expected proof receipt"):
        reproduce_attestation_verification(
            record,
            verifier=lambda _envelope: True,
            checked_at=NOW,
            receipt=ProofReceipt(
                **{
                    **_receipt().__dict__,
                    "attempt_id": "attempt:different",
                }
            ),
        )
    with pytest.raises(AttestationValidationError, match="not current"):
        reproduce_attestation_verification(
            record,
            verifier=lambda _envelope: True,
            checked_at=EXPIRES,
        )


def test_cache_sidecar_upgrades_then_downgrades_without_changing_receipt(
    tmp_path: Path,
) -> None:
    now = [_epoch(NOW)]
    cache = FormalVerificationCache(
        tmp_path,
        clock=lambda: now[0],
        attestation_verifier=lambda _envelope: True,
    )
    receipt = _receipt()
    assert cache.put(_key(), receipt, ttl_seconds=3_600)

    kernel_only = cache.lookup(_key())
    assert kernel_only.status is CacheLookupStatus.HIT
    assert kernel_only.receipt == receipt
    assert (
        kernel_only.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    )
    assert (
        cache.lookup(_key(), required_assurance=AssuranceLevel.ATTESTED).status
        is CacheLookupStatus.REJECTED
    )

    stored = cache.put_attestation(
        _key(),
        _verification(),
        expires_at=EXPIRES,
        ipfs_cid="bafk-attestation-public",
    )
    assert stored.stored
    attested = cache.lookup(
        _key(), required_assurance=AssuranceLevel.ATTESTED
    )
    assert attested.status is CacheLookupStatus.HIT
    assert attested.receipt == receipt
    assert attested.receipt.receipt_id == receipt.receipt_id
    assert attested.authoritative_assurance is AssuranceLevel.ATTESTED
    assert cache.lookup_attestation(_key()).entry.ipfs_cid == "bafk-attestation-public"
    assert cache.get_attestation(receipt.receipt_id).record_id == stored.entry.record.record_id
    replay_rejected = cache.lookup_attestation(
        _key(),
        verifier=lambda _envelope: False,
        checked_at=NOW,
    )
    assert replay_rejected.status is CacheLookupStatus.REJECTED
    assert (
        CacheRejectionReason.ATTESTATION_VERIFICATION_FAILED.value
        in replay_rejected.reason_codes
    )

    now[0] = _epoch(EXPIRES)
    expired = cache.lookup_attestation(_key())
    assert expired.status is CacheLookupStatus.REJECTED
    assert expired.receipt == receipt
    assert expired.kernel_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert expired.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert (
        CacheRejectionReason.ATTESTATION_EXPIRED.value
        in expired.reason_codes
    )
    kernel_still_valid = cache.lookup(_key())
    assert kernel_still_valid.status is CacheLookupStatus.HIT
    assert kernel_still_valid.receipt == receipt
    assert (
        kernel_still_valid.authoritative_assurance
        is AssuranceLevel.KERNEL_VERIFIED
    )


def test_cache_requires_replay_and_replacement_removes_old_sidecar(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path, clock=lambda: _epoch(NOW))
    assert cache.put(_key(), _receipt(), ttl_seconds=3_600)
    assert cache.put_attestation(_key(), _record())

    unreplayed = cache.lookup(
        _key(), required_assurance=AssuranceLevel.ATTESTED
    )
    assert unreplayed.status is CacheLookupStatus.REJECTED
    assert (
        CacheRejectionReason.ATTESTATION_REPLAY_REQUIRED.value
        in unreplayed.reason_codes
    )
    assert unreplayed.kernel_receipt == _receipt()

    replayed = cache.lookup(
        _key(),
        required_assurance=AssuranceLevel.ATTESTED,
        attestation_verifier=lambda _envelope: True,
    )
    assert replayed.status is CacheLookupStatus.HIT
    assert replayed.authoritative_assurance is AssuranceLevel.ATTESTED

    replacement = ProofReceipt(
        **{**_receipt().__dict__, "attempt_id": "attempt:replacement"}
    )
    assert cache.put(_key(), replacement, ttl_seconds=3_600)
    missing = cache.lookup_attestation(_key())
    assert missing.status is CacheLookupStatus.MISS
    assert missing.receipt == replacement
    assert cache.lookup(_key()).receipt == replacement


def test_attestation_delete_and_tamper_leave_kernel_cache_intact(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path, clock=lambda: _epoch(NOW))
    assert cache.put(_key(), _receipt(), ttl_seconds=3_600)
    assert cache.put_attestation(_key(), _record())
    assert cache.delete_attestation(_key())
    assert cache.lookup_attestation(_key()).status is CacheLookupStatus.MISS
    assert cache.lookup(_key()).receipt == _receipt()

    assert cache.put_attestation(_key(), _record())
    connection = sqlite3.connect(cache.db_path)
    try:
        connection.execute(
            "UPDATE proof_attestation_entries SET backend_id='backend:forged'"
        )
        connection.commit()
    finally:
        connection.close()
    poisoned = cache.lookup_attestation(_key())
    assert poisoned.status is CacheLookupStatus.REJECTED
    assert (
        CacheRejectionReason.ATTESTATION_POISONED.value
        in poisoned.reason_codes
    )
    assert cache.lookup(_key()).receipt == _receipt()


def test_queryable_artifact_and_optional_ipfs_contain_only_public_data(
    tmp_path: Path,
) -> None:
    secret = "unique-private-witness-REF-266"
    record = build_persisted_attestation_record(
        _receipt(),
        _verification(secret=secret),
        created_at=NOW,
        expires_at=EXPIRES,
    )
    published = {}

    def publish(payload: bytes) -> str:
        published["payload"] = payload
        return "bafk-proof-attestation"

    path = tmp_path / "proof_attestations.json"
    rendered = write_proof_attestation_artifact(
        path,
        record,
        ipfs_publisher=publish,
    )

    assert rendered["query_store"]["artifact_kind"] == PROOF_ATTESTATION_KIND
    assert rendered["authoritative"] is False
    assert rendered["attestations"][0]["ipfs_cid"] == "bafk-proof-attestation"
    assert secret.encode() not in published["payload"]
    assert not public_artifact_contains(record, secret)
    loaded = read_proof_attestation_artifact(
        path,
        verifier=lambda _envelope: True,
        checked_at=NOW,
    )
    assert loaded["attestations"][0]["record_id"] == record.record_id
    assert loaded["authoritative"] is False
    assert loaded["attested_assurance_available"] is True
    rows = query_proof_attestations(
        path,
        columns=(
            "proof_receipt_id",
            "kernel_receipt_id",
            "backend_id",
            "circuit_id",
            "verification_key_id",
            "expires_at",
            "ipfs_cid",
        ),
    )["rows"]
    assert rows == [
        {
            "proof_receipt_id": _receipt().receipt_id,
            "kernel_receipt_id": _receipt().kernel_receipt_id,
            "backend_id": _policy().backend_id,
            "circuit_id": _policy().circuit_id,
            "verification_key_id": _policy().verification_key_id,
            "expires_at": EXPIRES,
            "ipfs_cid": "bafk-proof-attestation",
        }
    ]


def test_ipfs_roundtrip_validates_content_and_publication_failure_is_optional(
    tmp_path: Path,
) -> None:
    record = _record()
    blocks = {}

    class Backend:
        def block_put(self, payload: bytes, *, codec: str) -> str:
            assert codec == "raw"
            cid = raw_ipfs_cid(payload)
            blocks[cid] = payload
            return cid

        def block_get(self, cid: str) -> bytes:
            return blocks[cid]

    path = tmp_path / "published.json"
    written = write_proof_attestation_artifact(
        path, record, ipfs_backend=Backend()
    )
    cid = written["attestations"][0]["ipfs_cid"]
    loaded = read_proof_attestation_artifact(
        cid,
        ipfs_backend=Backend(),
        verifier=lambda _envelope: True,
        checked_at=NOW,
    )
    assert loaded["attestations"][0]["record_id"] == record.record_id

    def unavailable(_payload: bytes):
        raise RuntimeError("backend unavailable")

    fallback = write_proof_attestation_artifact(
        tmp_path / "fallback.json",
        record,
        ipfs_publisher=unavailable,
    )
    assert fallback["ipfs_record_count"] == 0
    assert fallback["attestations"][0]["ipfs_publication_error"]
    assert read_proof_attestation_artifact(
        tmp_path / "fallback.json"
    )["attestations"][0]["record_id"] == record.record_id

    inspected = read_proof_attestation_artifact(path, checked_at=EXPIRES)
    assert inspected["authoritative"] is False
    assert inspected["attested_assurance_available"] is False
    assert inspected["attestations"][0]["attestation_current"] is False
    assert (
        inspected["attestations"][0]["effective_assurance"]
        == AssuranceLevel.KERNEL_VERIFIED.value
    )

    tampered = json.loads(blocks[cid])
    tampered["backend_id"] = "backend:forged"
    blocks[cid] = json.dumps(tampered).encode()
    with pytest.raises(ValueError, match="requested CID"):
        read_proof_attestation_artifact(
            cid,
            ipfs_backend=Backend(),
        )
