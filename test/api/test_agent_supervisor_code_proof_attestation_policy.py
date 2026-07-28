"""CBP-200: attestation and real ZK policy guards.

Acceptance coverage:

1. Threat model names prover, verifier, protected witness, disclosure risk,
   trust boundary, replay/freshness, and why signed/kernel receipts are
   insufficient.
2. Reviewed use-case decision is mandatory before Groth16/Plonk/other selection;
   core CBP terminal not_applicable does not block core CBP.
3. Simulated ZKP/attestation cannot satisfy AssuranceLevel.ATTESTED.
4. Public inputs bind property, repository/tree, obligation, toolchain, policy,
   and kernel-receipt digests.
5. Private witnesses rejected from public receipts and attestation cache entries.
6. Real attestations re-verify and fail closed on drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof_attestation import (
    AttestationBackendMode,
    AttestationBackendPolicy,
    AttestationGate,
    AttestationValidationError,
    AttestationVerification,
    AttestationVerificationVerdict,
    CBP_REQUIRED_PUBLIC_BINDING_KEYS,
    CORE_CBP_ZK_USE_CASE_DECISION,
    CORE_CBP_ZK_USE_CASE_ID,
    PersistedAttestationRecord,
    PrivateAttestationWitness,
    REQUIRED_BACKEND_TEST_CASES,
    ReceiptAttestationEnvelope,
    WitnessDisclosureError,
    ZkBackendFamily,
    ZkUseCaseDecisionRecord,
    ZkUseCaseDisposition,
    build_cbp_public_bindings,
    build_persisted_attestation_record,
    build_receipt_attestation_statement,
    core_cbp_zk_use_case_decision,
    create_attestation_envelope,
    evaluate_backend_health,
    execute_cryptographic_attestation,
    prepare_receipt_attestation,
    public_attestation_cache_entry,
    record_attestation_verification,
    reject_private_witness_from_public_payload,
    reproduce_attestation_verification,
    require_zk_backend_selection_authorized,
    simulated_attestation_cannot_satisfy_attested,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
THREAT_MODEL = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "agent_supervisor_codebase_proof_zk_threat_model.md"
)
ZK_POLICY = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "agent_supervisor_codebase_proof_zk_policy.md"
)

NOW = "2026-07-28T12:00:00Z"
EXPIRES = "2026-07-28T13:00:00Z"
PROPERTY_ID = "property:lease-fencing@cbp"


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=30_000,
        memory_bytes=512 * 1024 * 1024,
        max_processes=2,
        network_allowed=False,
    )


def _receipt(*, kernel_receipt_id: str = "kernel-receipt:lean:immutable") -> ProofReceipt:
    obligation_id = "obligation:cbp-200-attestation"
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel-proof",
        subject_id=obligation_id,
        verifier_id="kernel:lean@4.19",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
    )
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id="plan:cbp-200",
        attempt_id="attempt:kernel",
        repository_id="repo:cbp-codebase-proof",
        repository_tree_id="git-tree:cbp-200",
        ast_scope_ids=("scope:attestation-policy",),
        premise_ids=("premise:public-fence",),
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id="kernel:lean@4.19",
        toolchain_id="toolchain:nix-lock-sha256",
        theorem_registry_id="registry:reviewed-v3",
        policy_id="policy:cbp-attestation@1",
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        freshness=EvidenceFreshness.CURRENT,
        kernel_receipt_id=kernel_receipt_id,
        metadata={"property_id": PROPERTY_ID},
    )


def _policy(
    *,
    mode: AttestationBackendMode = AttestationBackendMode.CRYPTOGRAPHIC,
) -> AttestationBackendPolicy:
    return AttestationBackendPolicy(
        backend_id=(
            "backend:simulated"
            if mode is AttestationBackendMode.SIMULATED
            else "backend:provekit"
        ),
        backend_version="0.2.0",
        circuit_id="circuit:receipt-binding",
        circuit_version="2.1.0",
        public_input_schema_id="schema:receipt-public-inputs",
        public_input_schema_version="1.1.0",
        verification_key_id="vk:receipt-binding:sha256-beef",
        verification_key_version="ceremony-2026-07",
        backend_mode=mode,
        verification_key_expires_at="2030-01-01T00:00:00Z",
    )


def _health(policy: AttestationBackendPolicy | None = None):
    return evaluate_backend_health(
        policy or _policy(),
        configured=True,
        available=True,
        outcomes={case: True for case in REQUIRED_BACKEND_TEST_CASES},
        evaluated_at=NOW,
    )


def _crypto_verification(
    *,
    secret: str = "private-premise-never-public",
    property_id: str = PROPERTY_ID,
) -> AttestationVerification:
    policy = _policy()
    request = prepare_receipt_attestation(
        _receipt(),
        backend_policy=policy,
        property_id=property_id,
        require_cbp_bindings=True,
        witness=PrivateAttestationWitness({"private_premise": secret}),
    )
    return execute_cryptographic_attestation(
        request,
        backend_health=_health(policy),
        prover=lambda _request: {
            "proof_artifact_id": "artifact:zkp:public",
            "proof_digest": "sha256:public-proof-digest",
        },
        verifier=lambda _envelope: True,
        prover_id="prover:provekit@0.2.0",
        verifier_id="verifier:provekit@0.2.0",
    )


def test_threat_model_names_required_roles_and_insufficiency() -> None:
    assert THREAT_MODEL.is_file(), f"missing threat model: {THREAT_MODEL}"
    text = THREAT_MODEL.read_text(encoding="utf-8").lower()
    for term in (
        "prover",
        "verifier",
        "protected witness",
        "disclosure risk",
        "trust boundary",
        "replay",
        "freshness",
        "signed",
        "kernel",
        "insufficient",
    ):
        assert term in text, f"threat model missing term: {term}"


def test_policy_records_not_applicable_without_blocking_core_cbp() -> None:
    assert ZK_POLICY.is_file(), f"missing policy doc: {ZK_POLICY}"
    text = ZK_POLICY.read_text(encoding="utf-8").lower()
    assert "not_applicable" in text or "not applicable" in text
    assert "blocks_core_cbp" in text or "does **not** block core cbp" in text
    assert "groth16" in text and "plonk" in text
    assert "reviewed" in text

    decision = core_cbp_zk_use_case_decision()
    assert decision is CORE_CBP_ZK_USE_CASE_DECISION
    assert decision.use_case_id == CORE_CBP_ZK_USE_CASE_ID
    assert decision.disposition is ZkUseCaseDisposition.NOT_APPLICABLE
    assert decision.blocks_core_cbp is False
    assert decision.is_terminal
    assert not decision.authorizes_backend_selection
    assert decision.qualifying_private_witness is False
    assert decision.qualifying_cross_trust_boundary is False

    # Round-trip the reviewed decision as a public artifact.
    assert (
        ZkUseCaseDecisionRecord.from_dict(decision.to_public_artifact()) == decision
    )


def test_backend_selection_requires_approved_use_case_decision() -> None:
    decision = core_cbp_zk_use_case_decision()
    with pytest.raises(AttestationValidationError, match="not applicable"):
        require_zk_backend_selection_authorized(
            decision, backend_family=ZkBackendFamily.GROTH16
        )
    with pytest.raises(AttestationValidationError, match="not applicable"):
        require_zk_backend_selection_authorized(decision, backend_family="plonk")
    with pytest.raises(AttestationValidationError, match="simulated"):
        require_zk_backend_selection_authorized(
            decision, backend_family="simulated"
        )

    with pytest.raises(AttestationValidationError, match="cannot block core CBP"):
        ZkUseCaseDecisionRecord(
            use_case_id="bad-not-applicable",
            disposition=ZkUseCaseDisposition.NOT_APPLICABLE,
            blocks_core_cbp=True,
            reviewed_by="reviewer",
            reviewed_at=NOW,
            rationale="invalid combination",
            protected_witness_summary="n/a",
            trust_boundary_summary="n/a",
            disclosure_risk_summary="n/a",
            replay_freshness_summary="n/a",
            why_signed_receipts_insufficient="n/a",
            qualifying_private_witness=False,
            qualifying_cross_trust_boundary=False,
        )

    approved = ZkUseCaseDecisionRecord(
        use_case_id="future-cross-org-attestation",
        disposition=ZkUseCaseDisposition.APPROVED,
        blocks_core_cbp=False,
        reviewed_by="security-review",
        reviewed_at=NOW,
        rationale="Third party must verify without learning private premises.",
        protected_witness_summary="private premises and kernel transcript",
        trust_boundary_summary="operator prover vs external verifier",
        disclosure_risk_summary="witness must not enter public receipts",
        replay_freshness_summary="re-verify; expire with verification key",
        why_signed_receipts_insufficient=(
            "signatures disclose the statement and do not hide the witness"
        ),
        qualifying_private_witness=True,
        qualifying_cross_trust_boundary=True,
        approved_backend_families=("plonk", "groth16"),
    )
    assert require_zk_backend_selection_authorized(
        approved, backend_family="plonk"
    ) is approved
    with pytest.raises(AttestationValidationError, match="not authorized"):
        require_zk_backend_selection_authorized(
            approved, backend_family="halo2"
        )


def test_simulated_attestation_cannot_satisfy_attested() -> None:
    statement = build_receipt_attestation_statement(
        _receipt(),
        circuit_id="circuit:sim",
        backend_id="backend:simulated-demo",
        verification_key_id="vk:sim",
    )
    request = prepare_receipt_attestation(
        _receipt(),
        circuit_id="circuit:sim",
        backend_id="backend:simulated-demo",
        verification_key_id="vk:sim",
        witness=PrivateAttestationWitness({"private_premise": "sim-secret"}),
    )
    envelope = create_attestation_envelope(
        request,
        backend_mode=AttestationBackendMode.SIMULATED,
        proof_artifact_id="artifact:sim",
        proof_digest="sha256:sim",
        prover_id="prover:sim",
    )
    assert envelope.simulated
    assert not envelope.authoritative
    verification = record_attestation_verification(
        envelope,
        verified=True,
        verifier_id="verifier:sim",
        independent=True,
    )
    assert verification.verified
    assert not verification.authoritative
    assert verification.authoritative_assurance is AssuranceLevel.UNVERIFIED
    assert verification.authoritative_assurance is not AssuranceLevel.ATTESTED
    assert not verification.satisfies_gate(AttestationGate.PRODUCTION)
    assert not verification.satisfies_gate(AttestationGate.COMPLETION)
    assert simulated_attestation_cannot_satisfy_attested(verification)

    # Simulated evidence on a receipt cannot mint ATTESTED either.
    sim_evidence = ProofEvidence(
        kind=EvidenceKind.CRYPTOGRAPHIC_ATTESTATION,
        authority=EvidenceAuthority.ATTESTATION_VERIFIER,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:sim-zk",
        subject_id="kernel-receipt:lean:immutable",
        verifier_id="simulated-zkp",
        independent=True,
        simulated=True,
    )
    receipt = ProofReceipt(
        **{
            **_receipt().__dict__,
            "evidence": (
                *_receipt().evidence,
                sim_evidence,
            ),
            "provider_claimed_assurance": AssuranceLevel.ATTESTED,
        }
    )
    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert receipt.authoritative_assurance is not AssuranceLevel.ATTESTED
    assert not receipt.satisfies(AssuranceLevel.ATTESTED)
    del statement  # statement build used only to assert eligibility above


def test_public_inputs_bind_property_repo_tree_obligation_toolchain_policy_kernel() -> None:
    receipt = _receipt()
    bindings = build_cbp_public_bindings(
        receipt,
        property_id=PROPERTY_ID,
        circuit_id="circuit:receipt-binding",
        backend_id="backend:provekit",
        verification_key_id="vk:receipt-binding:sha256-beef",
        backend_policy=_policy(),
    )
    for key in CBP_REQUIRED_PUBLIC_BINDING_KEYS:
        assert key in bindings and bindings[key], f"missing binding: {key}"

    statement = build_receipt_attestation_statement(
        receipt,
        backend_policy=_policy(),
        property_id=PROPERTY_ID,
        require_cbp_bindings=True,
    )
    assert statement.has_cbp_public_bindings
    public = statement.public_inputs
    assert public["property_id"] == PROPERTY_ID
    assert public["repository_id"] == receipt.repository_id
    assert public["repository_tree_id"] == receipt.repository_tree_id
    assert public["obligation_id"] == receipt.obligation_id
    assert public["toolchain_id"] == receipt.toolchain_id
    assert public["policy_id"] == receipt.policy_id
    assert public["kernel_receipt_id"] == receipt.kernel_receipt_id
    assert public["kernel_receipt_digest"]
    assert public["property_digest"]
    assert statement.require_cbp_public_bindings() == statement.cbp_public_bindings

    with pytest.raises(AttestationValidationError, match="CBP attestation requires"):
        build_receipt_attestation_statement(
            _receipt(kernel_receipt_id=""),
            backend_policy=_policy(),
            property_id=PROPERTY_ID,
            require_cbp_bindings=True,
        )


def test_private_witnesses_rejected_from_public_receipts_and_cache_entries() -> None:
    secret = "top-secret-witness-value"
    witness = PrivateAttestationWitness({"private_premise": secret})
    request = prepare_receipt_attestation(
        _receipt(),
        backend_policy=_policy(),
        property_id=PROPERTY_ID,
        require_cbp_bindings=True,
        witness=witness,
    )

    with pytest.raises(WitnessDisclosureError):
        reject_private_witness_from_public_payload(witness)
    with pytest.raises(WitnessDisclosureError):
        reject_private_witness_from_public_payload(request)
    with pytest.raises(WitnessDisclosureError):
        public_attestation_cache_entry(witness)
    with pytest.raises(WitnessDisclosureError):
        public_attestation_cache_entry(request)

    with pytest.raises(WitnessDisclosureError, match="private witness"):
        reject_private_witness_from_public_payload(
            {"receipt_id": "r1", "private_witness": secret}
        )
    with pytest.raises(WitnessDisclosureError):
        public_attestation_cache_entry(
            {"nested": {"hidden_witness": "leak"}, "ok": True}
        )

    # Redaction marker alone is safe; live secrets are not.
    reject_private_witness_from_public_payload({"private_witness_redacted": True})
    public_view = request.to_public_artifact()
    assert public_view["private_witness_redacted"] is True
    assert secret not in str(public_view)
    reject_private_witness_from_public_payload(public_view)

    verification = _crypto_verification(secret=secret)
    record = build_persisted_attestation_record(
        _receipt(),
        verification,
        created_at=NOW,
        expires_at=EXPIRES,
    )
    cache_entry = public_attestation_cache_entry(record)
    assert secret not in str(cache_entry)
    reject_private_witness_from_public_payload(cache_entry)
    assert "private_premise" not in str(cache_entry)


def test_real_attestation_reverify_and_fail_closed_on_drift() -> None:
    receipt = _receipt()
    verification = _crypto_verification()
    assert verification.authoritative
    assert verification.authoritative_assurance is AssuranceLevel.ATTESTED
    assert not verification.simulated

    record = build_persisted_attestation_record(
        receipt,
        verification,
        created_at=NOW,
        expires_at=EXPIRES,
    )
    assert record.effective_assurance_at(NOW) is AssuranceLevel.ATTESTED
    assert record.envelope.statement.has_cbp_public_bindings

    # Independent re-verification from persisted public contracts succeeds.
    replayed = reproduce_attestation_verification(
        record,
        verifier=lambda envelope: envelope.proof_digest == "sha256:public-proof-digest",
        checked_at=NOW,
        receipt=receipt,
        backend_policy=_policy(),
    )
    assert replayed.verdict is AttestationVerificationVerdict.VERIFIED
    assert replayed.authoritative
    assert replayed.authoritative_assurance is AssuranceLevel.ATTESTED

    # Drift: verifier rejects → non-authoritative.
    rejected = reproduce_attestation_verification(
        record,
        verifier=lambda _envelope: False,
        checked_at=NOW,
    )
    assert rejected.verdict is AttestationVerificationVerdict.REJECTED
    assert not rejected.authoritative
    assert rejected.authoritative_assurance is AssuranceLevel.UNVERIFIED

    # Expiry / freshness drift fails closed before crypto.
    with pytest.raises(AttestationValidationError, match="not current"):
        reproduce_attestation_verification(
            record,
            verifier=lambda _envelope: True,
            checked_at=EXPIRES,
        )
    assert (
        record.effective_assurance_at(EXPIRES) is AssuranceLevel.KERNEL_VERIFIED
    )

    # Binding drift: wrong receipt id fails closed.
    other = _receipt()
    # Force a different receipt identity via a distinct attempt.
    other = ProofReceipt(
        **{
            **receipt.__dict__,
            "attempt_id": "attempt:different",
        }
    )
    with pytest.raises(AttestationValidationError, match="does not match"):
        reproduce_attestation_verification(
            record,
            verifier=lambda _envelope: True,
            checked_at=NOW,
            receipt=other,
        )


def test_persisted_record_rejects_non_authoritative_and_simulated_paths() -> None:
    receipt = _receipt()
    sim_request = prepare_receipt_attestation(
        receipt,
        circuit_id="circuit:sim",
        backend_id="backend:educational",
        verification_key_id="vk:sim",
        witness=PrivateAttestationWitness({"private_premise": "x"}),
    )
    sim_envelope = create_attestation_envelope(
        sim_request,
        backend_mode=AttestationBackendMode.SIMULATED,
        proof_artifact_id="artifact:sim",
        proof_digest="sha256:sim",
    )
    sim_verification = record_attestation_verification(
        sim_envelope,
        verified=True,
        verifier_id="verifier:sim",
    )
    with pytest.raises(AttestationValidationError, match="authoritative"):
        build_persisted_attestation_record(
            receipt,
            sim_verification,
            created_at=NOW,
            expires_at=EXPIRES,
        )

    # Cryptographic but non-CBP managed path without production health still
    # cannot be persisted without an authoritative managed envelope.
    bare = record_attestation_verification(
        create_attestation_envelope(
            prepare_receipt_attestation(
                receipt,
                circuit_id="circuit:bare",
                backend_id="backend:bare",
                verification_key_id="vk:bare",
                witness=PrivateAttestationWitness({"private_premise": "y"}),
            ),
            backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
            proof_artifact_id="artifact:bare",
            proof_digest="sha256:bare",
        ),
        verified=True,
        verifier_id="verifier:bare",
    )
    # Legacy unmanaged crypto envelopes may claim production_eligible but still
    # lack managed backend policy required for persistence.
    with pytest.raises(AttestationValidationError):
        build_persisted_attestation_record(
            receipt,
            bare,
            created_at=NOW,
            expires_at=EXPIRES,
        )


def test_threat_model_and_policy_docs_cross_link_and_name_cache_boundary() -> None:
    threat = THREAT_MODEL.read_text(encoding="utf-8")
    policy = ZK_POLICY.read_text(encoding="utf-8")
    assert "agent_supervisor_codebase_proof_zk_policy" in threat
    assert "agent_supervisor_codebase_proof_zk_threat_model" in policy
    for text in (threat, policy):
        lowered = text.lower()
        assert "formal_verification_cache" in lowered
        assert "attested" in lowered
        assert "simulated" in lowered
    assert "private witness" in threat.lower()
    assert CORE_CBP_ZK_USE_CASE_ID in policy


def test_cbp_public_bindings_appear_in_authoritative_attestation_evidence() -> None:
    verification = _crypto_verification()
    evidence = verification.to_evidence()
    assert evidence.kind is EvidenceKind.CRYPTOGRAPHIC_ATTESTATION
    assert evidence.simulated is False
    assert evidence.metadata["obligation_id"]
    assert evidence.metadata["policy_id"]
    assert evidence.metadata["repository_tree_id"]
    statement = verification.envelope.statement
    for key in (
        "property_id",
        "repository_id",
        "toolchain_id",
        "kernel_receipt_id",
        "kernel_receipt_digest",
    ):
        assert key in statement.public_inputs


def test_module_exports_policy_surface_for_supervisor_integration() -> None:
    from ipfs_accelerate_py.agent_supervisor import proof_attestation as mod

    for name in (
        "ZkUseCaseDecisionRecord",
        "require_zk_backend_selection_authorized",
        "core_cbp_zk_use_case_decision",
        "build_cbp_public_bindings",
        "reject_private_witness_from_public_payload",
        "public_attestation_cache_entry",
        "PersistedAttestationRecord",
        "reproduce_attestation_verification",
    ):
        assert hasattr(mod, name), f"missing export: {name}"
    assert mod.CORE_CBP_ZK_USE_CASE_DECISION.blocks_core_cbp is False
