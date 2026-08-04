"""PDR-060: reasoning-run lineage and gated optional ZKP attestation."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.program_analysis_zkp import (
    PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
    PrivateProgramAnalysisWitness,
    ProgramZkpBackendMode,
    ProgramZkpVerdict,
    build_production_ready_capability_fixture,
    build_program_zkp_public_inputs,
    create_program_zkp_shadow_envelope,
    prepare_program_analysis_zkp,
    record_program_zkp_verification,
    verify_program_zkp_independently,
)
from ipfs_accelerate_py.agent_supervisor.program_analysis_zkp import (
    PUBLIC_INPUT_CODEC_ID as PROGRAM_ZKP_PUBLIC_INPUT_CODEC_ID,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.proof.planner_doctor_attestation import (
    ATTESTATION_DOES_NOT_PROVE,
    ATTESTATION_SCOPE_STATEMENT,
    DEFAULT_LINEAGE_CIRCUIT_ID,
    LINEAGE_EVIDENCE_TYPES,
    PLANNER_DOCTOR_ATTESTATION_INTERFACE,
    PLANNER_DOCTOR_ZKP_THREAT_MODEL_ID,
    PLANNER_DOCTOR_ZKP_USE_CASE_ID,
    PUBLIC_COMMITMENT_KEYS,
    REASONING_RUN_MANIFEST_INTERFACE,
    AttestationBackendError,
    AttestationClaimPromotionError,
    LineageEvidenceType,
    LineageOrderError,
    LineagePreimageError,
    LineageReplayError,
    LineageRootError,
    LineageSlot,
    PlannerDoctorAttestation,
    PlannerDoctorAttestationStatus,
    PlannerDoctorBackendMode,
    PlannerDoctorPublicInputs,
    PlannerDoctorZkpPredicate,
    PrivatePlannerDoctorWitness,
    ReasoningRunManifest,
    WitnessDisclosureError,
    attestation_does_not_prove,
    attestation_independent_of_semantic_authority,
    build_public_inputs_from_manifest,
    build_reasoning_run_manifest,
    create_failed_attestation,
    create_planner_doctor_attestation,
    create_simulated_attestation,
    create_unavailable_attestation,
    encode_public_input_vector,
    leaf_digest_for_slot,
    merkle_root_from_leaves,
    planner_doctor_program_zkp_result_commitment,
    prepare_planner_doctor_attestation,
    public_input_vector_digest,
    public_planner_doctor_artifact,
    reject_collapsed_evidence_types,
    reject_illegal_semantic_claim,
    reject_private_witness_from_public_payload,
    require_run_replay,
    seal_cryptographic_attested,
    simulated_attestation_cannot_satisfy_attested,
    typed_receipt_cid,
    verify_lineage_merkle_root,
    verify_lineage_preimages,
    verify_planner_doctor_attestation,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_attestation import (
    AttestationGate,
    AttestationVerificationVerdict,
)

RUN_ID = "run:planner-doctor-demo-001"
TREE_ID = "tree:repo-forest@deadbeef"
POLICY_ID = "policy:planner-doctor-authority@1"

THREAT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "agent_supervisor_planner_doctor_zkp_threat_model.md"
)


def _bodies(run_id: str = RUN_ID) -> dict[str, dict[str, str]]:
    return {
        kind: {"receipt": "%s-body" % kind, "payload": "v1"}
        for kind in LINEAGE_EVIDENCE_TYPES
    }


def _preimages(run_id: str = RUN_ID) -> dict[str, dict[str, str]]:
    return _bodies(run_id)


def _cids(run_id: str = RUN_ID) -> dict[str, str]:
    bodies = _bodies(run_id)
    return {
        kind: typed_receipt_cid(
            evidence_type=kind, run_id=run_id, body=bodies[kind]
        )
        for kind in LINEAGE_EVIDENCE_TYPES
    }


def _manifest(run_id: str = RUN_ID) -> ReasoningRunManifest:
    cids = _cids(run_id)
    return build_reasoning_run_manifest(
        run_id=run_id,
        repository_tree_id=TREE_ID,
        policy_id=POLICY_ID,
        planner_cid=cids["planner"],
        doctor_cid=cids["doctor"],
        cache_cid=cids["cache"],
        plan_cid=cids["plan"],
        permit_cid=cids["permit"],
        mutation_cid=cids["mutation"],
        fixed_point_cid=cids["fixed_point"],
        benchmark_cid=cids["benchmark"],
        promotion_cid=cids["promotion"],
        spans={
            "planner": "span:plan",
            "doctor": "span:doctor",
            "benchmark": "span:bench",
        },
        signature="sig:demo",
        signer_id="did:web:operator.example",
    )


def _public_inputs(
    manifest: ReasoningRunManifest | None = None,
) -> PlannerDoctorPublicInputs:
    man = manifest or _manifest()
    return build_public_inputs_from_manifest(
        man,
        proving_key_id="pk:planner-doctor-lineage@1",
        verifying_key_id="vk:planner-doctor-lineage@1",
        ceremony_id="ceremony:planner-doctor-lineage@1",
        predicate=PlannerDoctorZkpPredicate.PRIVATE_WITNESS_POSSESSION,
    )


def _witness(secret: str = "opening-secret-never-publish") -> PrivatePlannerDoctorWitness:
    return PrivatePlannerDoctorWitness(
        {
            "commitment_opening": secret,
            "membership_path": ["leaf-a", "leaf-b"],
            "private_premise": {"holdout": True},
        }
    )


def _production_public_inputs() -> PlannerDoctorPublicInputs:
    """Use the currently capability-qualified fixed-computation circuit."""

    return build_public_inputs_from_manifest(
        _manifest(),
        circuit_id=PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
        circuit_version="1",
        proving_key_id="pk:program-contract-trace@1:sha256-pk-fixture",
        verifying_key_id="vk:program-contract-trace@1:sha256-vk-fixture",
        ceremony_id="ceremony:program-contract-trace@1",
        predicate=PlannerDoctorZkpPredicate.FIXED_BOUNDED_COMPUTATION,
    )


def _program_zkp_inputs_for(envelope: PlannerDoctorAttestation):
    inputs = envelope.public_inputs
    return build_program_zkp_public_inputs(
        forest_commitment=inputs.repository_tree_id,
        inventory_commitment=inputs.manifest_id,
        contract_commitment=inputs.policy_id,
        call_slice_commitment=inputs.lineage_merkle_root,
        assumptions_commitment=inputs.public_input_digest,
        analyzer_version="analyzer:planner-doctor-lineage@1",
        resolver_version="resolver:planner-doctor-lineage@1",
        translator_version="translator:planner-doctor-zkp-bridge@1",
        prover_version="prover:program-contract-trace@1",
        result_commitment=planner_doctor_program_zkp_result_commitment(
            inputs,
            proof_artifact_id=envelope.proof_artifact_id,
            proof_digest=envelope.proof_digest,
            prover_id=envelope.prover_id,
        ),
        circuit_id=inputs.circuit_id,
        proving_key_id=inputs.proving_key_id,
        verifying_key_id=inputs.verifying_key_id,
        ceremony_id=inputs.ceremony_id,
        public_input_codec_id=PROGRAM_ZKP_PUBLIC_INPUT_CODEC_ID,
        public_input_codec_version=inputs.public_input_codec_version,
    )


def _program_receipt_for(
    envelope: PlannerDoctorAttestation,
    *,
    backend_mode: ProgramZkpBackendMode = ProgramZkpBackendMode.CRYPTOGRAPHIC,
):
    request = prepare_program_analysis_zkp(
        _program_zkp_inputs_for(envelope),
        witness=PrivateProgramAnalysisWitness(
            {"commitment_opening": "program-zkp-test-opening"}
        ),
        backend_mode=backend_mode,
    )
    program_envelope = create_program_zkp_shadow_envelope(
        request,
        proof_artifact_id=envelope.proof_artifact_id,
        proof_digest=envelope.proof_digest,
        prover_id="prover:program-zkp-bridge@1",
        backend_mode=backend_mode,
    )
    if backend_mode is not ProgramZkpBackendMode.CRYPTOGRAPHIC:
        return record_program_zkp_verification(
            program_envelope,
            verdict=ProgramZkpVerdict.VERIFIED,
            verifier_id="verifier:program-zkp-shadow@1",
            independent_verifier=True,
        )
    return verify_program_zkp_independently(
        program_envelope,
        capability=build_production_ready_capability_fixture(),
        verifier_id="verifier:program-zkp-independent@1",
        proof_bytes=b"production-proof-bytes-v1",
        verifying_key_material=b"vk-fixture-material-v1",
        cryptographic_verify=lambda proof, key, fields: bool(
            proof and key and fields
        ),
    )


# ---------------------------------------------------------------------------
# Manifest: distinct evidence types and CIDs
# ---------------------------------------------------------------------------


def test_manifest_links_all_lineage_cids_without_collapsing_types() -> None:
    manifest = _manifest()
    assert manifest.interface == REASONING_RUN_MANIFEST_INTERFACE
    assert tuple(slot.evidence_type.value for slot in manifest.slots) == (
        LINEAGE_EVIDENCE_TYPES
    )
    types = [slot.evidence_type.value for slot in manifest.slots]
    assert len(types) == len(set(types)) == len(LINEAGE_EVIDENCE_TYPES)
    cids = [slot.receipt_cid for slot in manifest.slots]
    assert len(cids) == len(set(cids))
    for kind in LineageEvidenceType:
        assert manifest.cid_for(kind)
    reject_collapsed_evidence_types(manifest)
    public = manifest.to_public_artifact()
    assert public["manifest_id"] == manifest.manifest_id
    assert public["evidence_types"] == list(LINEAGE_EVIDENCE_TYPES)
    assert ReasoningRunManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_rejects_missing_or_extra_evidence_types() -> None:
    cids = _cids()
    slots = [
        LineageSlot(evidence_type=kind, receipt_cid=cids[kind.value])
        for kind in LineageEvidenceType
        if kind is not LineageEvidenceType.PROMOTION
    ]
    with pytest.raises(LineageOrderError, match="missing"):
        ReasoningRunManifest(
            run_id=RUN_ID,
            repository_tree_id=TREE_ID,
            policy_id=POLICY_ID,
            slots=tuple(slots),
        )


def test_manifest_rejects_collapsed_duplicate_cids_across_types() -> None:
    cids = _cids()
    shared = cids["planner"]
    with pytest.raises(LineageOrderError, match="shared by evidence types"):
        reject_collapsed_evidence_types(
            ReasoningRunManifest(
                run_id=RUN_ID,
                repository_tree_id=TREE_ID,
                policy_id=POLICY_ID,
                slots=tuple(
                    LineageSlot(
                        evidence_type=kind,
                        receipt_cid=(
                            shared
                            if kind is LineageEvidenceType.DOCTOR
                            else cids[kind.value]
                        ),
                    )
                    for kind in LineageEvidenceType
                ),
            )
        )


def test_wrong_preimage_fails() -> None:
    manifest = _manifest()
    preimages = _preimages()
    verify_lineage_preimages(manifest, preimages)
    bad = dict(preimages)
    bad["doctor"] = {"receipt": "tampered-doctor"}
    with pytest.raises(LineagePreimageError, match="doctor"):
        verify_lineage_preimages(manifest, bad)


def test_wrong_order_changes_merkle_root() -> None:
    manifest = _manifest()
    leaves = list(manifest.leaf_digests())
    # Swap planner and doctor leaves — order is part of the root.
    leaves[0], leaves[1] = leaves[1], leaves[0]
    reordered_root = merkle_root_from_leaves(leaves)
    assert reordered_root != manifest.lineage_merkle_root
    with pytest.raises(LineageRootError):
        verify_lineage_merkle_root(manifest, expected_root=reordered_root)


def test_wrong_root_fails() -> None:
    manifest = _manifest()
    with pytest.raises(LineageRootError, match="supplied lineage root"):
        verify_lineage_merkle_root(
            manifest, expected_root="b" + ("a" * 58)
        )


def test_forged_manifest_root_rejected_at_construction() -> None:
    cids = _cids()
    with pytest.raises(LineageRootError, match="lineage_merkle_root"):
        ReasoningRunManifest(
            run_id=RUN_ID,
            repository_tree_id=TREE_ID,
            policy_id=POLICY_ID,
            slots=tuple(
                LineageSlot(evidence_type=kind, receipt_cid=cids[kind.value])
                for kind in LineageEvidenceType
            ),
            lineage_merkle_root="b" + ("f" * 58),
        )


def test_cross_run_replay_fails() -> None:
    manifest = _manifest(RUN_ID)
    other = _manifest("run:other-run-999")
    assert manifest.lineage_merkle_root != other.lineage_merkle_root
    with pytest.raises(LineageReplayError, match="run_id"):
        require_run_replay(
            manifest,
            run_id="run:other-run-999",
            repository_tree_id=TREE_ID,
            policy_id=POLICY_ID,
            lineage_merkle_root=manifest.lineage_merkle_root,
            preimages=_preimages(RUN_ID),
        )
    with pytest.raises(LineageReplayError, match="repository_tree_id"):
        require_run_replay(
            manifest,
            run_id=RUN_ID,
            repository_tree_id="tree:drifted",
            policy_id=POLICY_ID,
            lineage_merkle_root=manifest.lineage_merkle_root,
        )
    # Happy path
    require_run_replay(
        manifest,
        run_id=RUN_ID,
        repository_tree_id=TREE_ID,
        policy_id=POLICY_ID,
        lineage_merkle_root=manifest.lineage_merkle_root,
        preimages=_preimages(RUN_ID),
    )


def test_leaf_digest_binds_run_and_type() -> None:
    cid = _cids()["planner"]
    a = leaf_digest_for_slot("planner", cid, run_id=RUN_ID)
    b = leaf_digest_for_slot("doctor", cid, run_id=RUN_ID)
    c = leaf_digest_for_slot("planner", cid, run_id="run:other")
    assert a != b
    assert a != c


# ---------------------------------------------------------------------------
# Threat model document
# ---------------------------------------------------------------------------


def test_threat_model_names_exact_claim_and_signature_insufficiency() -> None:
    text = THREAT_MODEL_PATH.read_text(encoding="utf-8")
    assert "private witness" in text.lower() or "private-witness" in text.lower()
    assert "receipt lineage" in text.lower() or "receipt_lineage" in text
    assert "insufficient" in text.lower()
    assert "signature" in text.lower()
    assert "Merkle" in text or "merkle" in text
    assert "inventory completeness" in text.lower() or "inventory_completeness" in text
    assert "translator soundness" in text.lower() or "translator_soundness" in text
    assert "ATTESTED" in text or "attested" in text
    assert "simulated" in text.lower()
    assert PLANNER_DOCTOR_ZKP_USE_CASE_ID in text or "pdr-060" in text
    assert "PlannerDoctorAttestation" in text or "PlannerDoctorAttestation@1" in text
    assert "ReasoningRunManifest" in text


# ---------------------------------------------------------------------------
# Optional ZKP: public inputs, witness, independent verification
# ---------------------------------------------------------------------------


def test_public_inputs_bind_fixed_circuit_and_lineage() -> None:
    manifest = _manifest()
    inputs = _public_inputs(manifest)
    assert inputs.run_id == manifest.run_id
    assert inputs.manifest_id == manifest.manifest_id
    assert inputs.lineage_merkle_root == manifest.lineage_merkle_root
    assert inputs.circuit_id == DEFAULT_LINEAGE_CIRCUIT_ID
    assert inputs.use_case_id == PLANNER_DOCTOR_ZKP_USE_CASE_ID
    assert inputs.threat_model_id == PLANNER_DOCTOR_ZKP_THREAT_MODEL_ID
    assert tuple(inputs.public_inputs.keys()) == PUBLIC_COMMITMENT_KEYS
    vector = encode_public_input_vector(inputs.public_inputs)
    assert len(vector) == len(PUBLIC_COMMITMENT_KEYS)
    assert public_input_vector_digest(inputs.public_inputs) == inputs.public_input_digest
    assert PlannerDoctorPublicInputs.from_dict(inputs.to_dict()) == inputs


def test_every_public_commitment_changes_identity() -> None:
    baseline = _public_inputs()
    for key in PUBLIC_COMMITMENT_KEYS:
        if key in {
            "public_input_codec_id",
            "public_input_codec_version",
            "use_case_id",
            "threat_model_id",
        }:
            continue
        changed = baseline.with_overrides(**{key: "tampered:%s" % key})
        assert changed.public_input_digest != baseline.public_input_digest


def test_private_witness_never_serializes_or_leaks() -> None:
    secret = "opening-secret-never-publish"
    witness = _witness(secret)
    assert secret not in repr(witness)
    with pytest.raises(WitnessDisclosureError):
        witness.to_dict()
    with pytest.raises(WitnessDisclosureError):
        pickle.dumps(witness)
    with pytest.raises(WitnessDisclosureError):
        copy.copy(witness)

    request = prepare_planner_doctor_attestation(
        _public_inputs(),
        witness=witness,
        backend_mode=PlannerDoctorBackendMode.SHADOW,
    )
    public = public_planner_doctor_artifact(request)
    assert public["private_witness_redacted"] is True
    assert secret not in str(public)
    reject_private_witness_from_public_payload(public)
    with pytest.raises(WitnessDisclosureError):
        reject_private_witness_from_public_payload(
            {"private_witness": secret, "ok": True}
        )


def test_simulated_backend_never_emits_production_attested() -> None:
    inputs = _public_inputs()
    envelope = create_simulated_attestation(inputs)
    assert envelope.interface == PLANNER_DOCTOR_ATTESTATION_INTERFACE
    assert envelope.simulated is True
    assert envelope.status is PlannerDoctorAttestationStatus.SIMULATED
    assert envelope.production_eligible is False
    verification = verify_planner_doctor_attestation(
        envelope,
        verifier_id="verifier:planner-doctor@1",
        expected_public_input_digest=inputs.public_input_digest,
        expected_lineage_merkle_root=inputs.lineage_merkle_root,
        expected_run_id=inputs.run_id,
        accept_simulated=True,
    )
    assert verification.verdict is AttestationVerificationVerdict.VERIFIED
    assert verification.authoritative is False
    assert verification.authoritative_assurance is not AssuranceLevel.ATTESTED
    assert verification.authoritative_assurance is AssuranceLevel.CANDIDATE
    assert verification.satisfies_production_gate() is False
    assert verification.satisfies_completion_gate() is False
    assert simulated_attestation_cannot_satisfy_attested(verification) is True
    with pytest.raises(AttestationBackendError, match="cryptographic"):
        seal_cryptographic_attested(envelope)


def test_unavailable_and_failed_remain_non_attested() -> None:
    inputs = _public_inputs()
    unavailable = create_unavailable_attestation(inputs)
    failed = create_failed_attestation(inputs)
    for envelope, code in (
        (unavailable, "backend_unavailable"),
        (failed, "backend_failed"),
    ):
        result = verify_planner_doctor_attestation(
            envelope,
            verifier_id="verifier:planner-doctor@1",
            expected_public_input_digest=inputs.public_input_digest,
            expected_lineage_merkle_root=inputs.lineage_merkle_root,
            expected_run_id=inputs.run_id,
        )
        assert result.verdict is AttestationVerificationVerdict.ERROR
        assert result.diagnostic_code == code
        assert result.authoritative_assurance is AssuranceLevel.UNVERIFIED
        assert result.authoritative is False
        assert result.satisfies_gate(AttestationGate.PRODUCTION) is False
        assert simulated_attestation_cannot_satisfy_attested(result) is True or (
            envelope.status is PlannerDoctorAttestationStatus.FAILED
            and result.authoritative_assurance is not AssuranceLevel.ATTESTED
        )


def test_shadow_create_stays_candidate() -> None:
    request = prepare_planner_doctor_attestation(
        _public_inputs(),
        witness=_witness(),
        backend_mode=PlannerDoctorBackendMode.SHADOW,
    )
    envelope = create_planner_doctor_attestation(
        request,
        proof_artifact_id="artifact:zk-shadow",
        proof_digest="sha256:" + ("11" * 32),
        prover_id="prover:shadow",
    )
    assert envelope.status is PlannerDoctorAttestationStatus.CANDIDATE
    assert envelope.production_eligible is False
    result = verify_planner_doctor_attestation(
        envelope,
        verifier_id="verifier:planner-doctor@1",
        expected_public_input_digest=envelope.public_inputs.public_input_digest,
        expected_lineage_merkle_root=envelope.public_inputs.lineage_merkle_root,
        expected_run_id=envelope.public_inputs.run_id,
    )
    assert result.authoritative_assurance is AssuranceLevel.CANDIDATE
    assert result.authoritative is False


def test_cryptographic_attested_requires_production_seal_and_independent_verify() -> None:
    inputs = _production_public_inputs()
    request = prepare_planner_doctor_attestation(
        inputs,
        witness=_witness(),
        backend_mode=PlannerDoctorBackendMode.CRYPTOGRAPHIC,
        production_eligible=True,
    )
    generated = create_planner_doctor_attestation(
        request,
        proof_artifact_id="artifact:zk-crypto",
        proof_digest="sha256:" + ("22" * 32),
        prover_id="prover:crypto",
    )
    assert generated.status is PlannerDoctorAttestationStatus.GENERATED
    candidate = verify_planner_doctor_attestation(
        generated,
        verifier_id="verifier:planner-doctor@1",
        expected_public_input_digest=inputs.public_input_digest,
        expected_lineage_merkle_root=inputs.lineage_merkle_root,
        expected_run_id=inputs.run_id,
    )
    assert candidate.authoritative is False
    assert candidate.authoritative_assurance is AssuranceLevel.CANDIDATE

    # Prover-controlled flags alone cannot promote the envelope.
    with pytest.raises(
        AttestationBackendError, match="ProgramZkpVerificationReceipt"
    ):
        seal_cryptographic_attested(generated)

    program_receipt = _program_receipt_for(generated)
    assert program_receipt.authoritative is True
    sealed = seal_cryptographic_attested(
        generated, verification_receipt=program_receipt
    )
    assert sealed.status is PlannerDoctorAttestationStatus.ATTESTED
    assert sealed.program_zkp_verification_receipt == program_receipt
    assert PlannerDoctorAttestation.from_dict(sealed.to_dict()) == sealed
    verified = verify_planner_doctor_attestation(
        sealed,
        verifier_id="verifier:planner-doctor@1",
        expected_public_input_digest=inputs.public_input_digest,
        expected_lineage_merkle_root=inputs.lineage_merkle_root,
        expected_run_id=inputs.run_id,
    )
    assert verified.verdict is AttestationVerificationVerdict.VERIFIED
    assert verified.authoritative is True
    assert verified.authoritative_assurance is AssuranceLevel.ATTESTED
    assert verified.satisfies_production_gate() is True
    verified.require_replay(
        public_input_digest=inputs.public_input_digest,
        lineage_merkle_root=inputs.lineage_merkle_root,
        run_id=inputs.run_id,
        verifying_key_id=inputs.verifying_key_id,
        circuit_id=inputs.circuit_id,
    )


def test_verification_replay_rejects_wrong_run_root_or_digest() -> None:
    inputs = _production_public_inputs()
    generated = create_planner_doctor_attestation(
        prepare_planner_doctor_attestation(
            inputs,
            witness=_witness(),
            backend_mode=PlannerDoctorBackendMode.CRYPTOGRAPHIC,
            production_eligible=True,
        ),
        proof_artifact_id="artifact:zk-crypto",
        proof_digest="sha256:" + ("33" * 32),
        prover_id="prover:crypto",
        status=PlannerDoctorAttestationStatus.GENERATED,
    )
    sealed = seal_cryptographic_attested(
        generated,
        verification_receipt=_program_receipt_for(generated),
    )
    verified = verify_planner_doctor_attestation(
        sealed,
        verifier_id="verifier:planner-doctor@1",
        expected_public_input_digest=inputs.public_input_digest,
        expected_lineage_merkle_root=inputs.lineage_merkle_root,
        expected_run_id=inputs.run_id,
    )
    with pytest.raises(LineageReplayError, match="run_id"):
        verified.require_replay(
            public_input_digest=inputs.public_input_digest,
            lineage_merkle_root=inputs.lineage_merkle_root,
            run_id="run:forged",
        )
    with pytest.raises(LineageReplayError, match="lineage_merkle_root"):
        verified.require_replay(
            public_input_digest=inputs.public_input_digest,
            lineage_merkle_root="b" + ("0" * 58),
            run_id=inputs.run_id,
        )
    with pytest.raises(LineageReplayError, match="public_input_digest"):
        verified.require_replay(
            public_input_digest="b" + ("1" * 58),
            lineage_merkle_root=inputs.lineage_merkle_root,
            run_id=inputs.run_id,
        )
    # Independent verifier rejects mismatched pins without raising — typed reject.
    rejected = verify_planner_doctor_attestation(
        sealed,
        verifier_id="verifier:planner-doctor@1",
        expected_public_input_digest=inputs.public_input_digest,
        expected_lineage_merkle_root=inputs.lineage_merkle_root,
        expected_run_id="run:other",
    )
    assert rejected.verdict is AttestationVerificationVerdict.REJECTED
    assert rejected.authoritative_assurance is not AssuranceLevel.ATTESTED


def test_cannot_construct_attested_status_on_simulated_backend() -> None:
    with pytest.raises(AttestationBackendError):
        PlannerDoctorAttestation(
            public_inputs=_public_inputs(),
            proof_artifact_id="artifact:bad",
            proof_digest="sha256:" + ("44" * 32),
            prover_id="prover:bad",
            backend_mode=PlannerDoctorBackendMode.SIMULATED,
            production_eligible=False,
            status=PlannerDoctorAttestationStatus.ATTESTED,
        )


def test_cannot_construct_flag_only_cryptographic_attested_envelope() -> None:
    with pytest.raises(
        AttestationBackendError, match="ProgramZkpVerificationReceipt"
    ):
        PlannerDoctorAttestation(
            public_inputs=_production_public_inputs(),
            proof_artifact_id="artifact:flag-only",
            proof_digest="sha256:" + ("45" * 32),
            prover_id="prover:self-asserted",
            backend_mode=PlannerDoctorBackendMode.CRYPTOGRAPHIC,
            production_eligible=True,
            status=PlannerDoctorAttestationStatus.ATTESTED,
        )


def test_simulated_program_receipt_cannot_seal_crypto_envelope() -> None:
    inputs = _production_public_inputs()
    generated = create_planner_doctor_attestation(
        prepare_planner_doctor_attestation(
            inputs,
            witness=_witness(),
            backend_mode=PlannerDoctorBackendMode.CRYPTOGRAPHIC,
            production_eligible=True,
        ),
        proof_artifact_id="artifact:outer-crypto",
        proof_digest="sha256:" + ("46" * 32),
        prover_id="prover:outer-crypto",
    )
    simulated_receipt = _program_receipt_for(
        generated, backend_mode=ProgramZkpBackendMode.SIMULATED
    )
    assert simulated_receipt.authoritative is False
    with pytest.raises(AttestationBackendError, match="not authoritative"):
        seal_cryptographic_attested(
            generated, verification_receipt=simulated_receipt
        )


def test_program_receipt_cannot_replay_over_different_proof_artifact() -> None:
    inputs = _production_public_inputs()
    original = create_planner_doctor_attestation(
        prepare_planner_doctor_attestation(
            inputs,
            witness=_witness(),
            backend_mode=PlannerDoctorBackendMode.CRYPTOGRAPHIC,
            production_eligible=True,
        ),
        proof_artifact_id="artifact:original-proof",
        proof_digest="sha256:" + ("47" * 32),
        prover_id="prover:original",
    )
    receipt = _program_receipt_for(original)
    substituted = create_planner_doctor_attestation(
        prepare_planner_doctor_attestation(
            inputs,
            witness=_witness(),
            backend_mode=PlannerDoctorBackendMode.CRYPTOGRAPHIC,
            production_eligible=True,
        ),
        proof_artifact_id="artifact:substituted-proof",
        proof_digest=original.proof_digest,
        prover_id=original.prover_id,
    )
    with pytest.raises(AttestationBackendError, match="result_commitment"):
        seal_cryptographic_attested(
            substituted, verification_receipt=receipt
        )


def test_attestation_never_claims_semantics_inventory_or_translator() -> None:
    for claim in (
        "semantic_correctness",
        "inventory_completeness",
        "translator_soundness",
    ):
        assert attestation_does_not_prove(claim) is True
        with pytest.raises(AttestationClaimPromotionError, match=claim):
            reject_illegal_semantic_claim(claim)
    assert "inventory completeness" in ATTESTATION_SCOPE_STATEMENT.lower() or (
        "inventory_completeness" in ATTESTATION_DOES_NOT_PROVE
    )
    inputs = _public_inputs()
    envelope = create_simulated_attestation(inputs)
    verification = verify_planner_doctor_attestation(
        envelope,
        verifier_id="verifier:planner-doctor@1",
        expected_public_input_digest=inputs.public_input_digest,
        expected_lineage_merkle_root=inputs.lineage_merkle_root,
        expected_run_id=inputs.run_id,
        accept_simulated=True,
    )
    assert attestation_independent_of_semantic_authority(verification) is True
    public = verification.to_public_artifact()
    assert set(ATTESTATION_DOES_NOT_PROVE).issubset(set(public["does_not_prove"]))
    assert secret_free(public)


def secret_free(payload: object) -> bool:
    text = str(payload)
    return "opening-secret-never-publish" not in text


def test_envelope_round_trip_and_public_artifact() -> None:
    inputs = _public_inputs()
    envelope = create_simulated_attestation(inputs)
    restored = PlannerDoctorAttestation.from_dict(envelope.to_dict())
    assert restored == envelope
    public = envelope.to_public_artifact()
    assert public["attestation_id"] == envelope.attestation_id
    assert public["private_witness_redacted"] is True
    reject_private_witness_from_public_payload(public)


def test_create_rejects_forcing_attested_on_shadow_request() -> None:
    request = prepare_planner_doctor_attestation(
        _public_inputs(),
        witness=_witness(),
        backend_mode=PlannerDoctorBackendMode.SHADOW,
    )
    with pytest.raises(AttestationBackendError, match="ATTESTED"):
        create_planner_doctor_attestation(
            request,
            proof_artifact_id="artifact:x",
            proof_digest="sha256:" + ("55" * 32),
            prover_id="prover:x",
            status=PlannerDoctorAttestationStatus.ATTESTED,
        )
