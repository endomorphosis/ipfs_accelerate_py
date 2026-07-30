"""SCA-G082 / SCAEV082REALZK: real datasets ZK verified-receipt backend tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    CapabilityHealth,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
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
from ipfs_accelerate_py.agent_supervisor.proof.ipfs_datasets_zk_attestation import (
    APPROVED_VERIFIED_RECEIPT_PREDICATES,
    DEFAULT_VERIFIED_RECEIPT_PREDICATE,
    IPFS_DATASETS_ZK_ATTESTATION_INTERFACE,
    SCAEV082REALZK,
    DatasetsZkAttestationResult,
    DatasetsZkBackendSelection,
    DatasetsZkSetupIdentity,
    DatasetsZkStatus,
    IpfsDatasetsZkAttestation,
    build_datasets_zk_setup_identity,
    datasets_zkp_registry_available,
    probe_datasets_zk_backend,
    public_datasets_zk_artifact,
    run_datasets_zk_backend_self_tests,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_attestation import (
    AttestationBackendMode,
    AttestationGate,
    AttestationTrust,
    AttestationValidationError,
    BackendTestCase,
    DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_DECISION,
    DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_ID,
    PrivateAttestationWitness,
    REQUIRED_BACKEND_TEST_CASES,
    WitnessDisclosureError,
    ZkBackendFamily,
    ZkUseCaseDecisionRecord,
    ZkUseCaseDisposition,
    datasets_verified_receipt_zk_use_case_decision,
    public_artifact_contains,
    require_zk_backend_selection_authorized,
    simulated_attestation_cannot_satisfy_attested,
)


NOW = "2026-07-29T12:00:00Z"
SECRET = "witness-secret-SCAEV082-opening"


def _receipt(*, proved: bool = True) -> ProofReceipt:
    obligation_id = "obligation:datasets-zk-receipt"
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel-datasets-zk",
        subject_id=obligation_id,
        verifier_id="kernel:lean@4.19",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
    )
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id="plan:datasets-zk",
        attempt_id="attempt:datasets-zk",
        repository_id="repo:datasets-zk",
        repository_tree_id="git-tree:datasets-zk-abc",
        ast_scope_ids=("scope:datasets-zk",),
        premise_ids=("premise:datasets-zk",),
        translator_id="translator:datasets-zk@1",
        solver_id="solver:datasets-zk@1",
        kernel_id="kernel:lean@4.19",
        toolchain_id="toolchain:datasets-zk@1",
        policy_id="policy:datasets-zk@1",
        resource_budget=ResourceBudget(
            wall_time_ms=1_000,
            memory_bytes=1_000_000,
            max_processes=1,
            network_allowed=False,
        ),
        verdict=ProofVerdict.PROVED if proved else ProofVerdict.DISPROVED,
        evidence=(evidence,) if proved else (),
        kernel_receipt_id="kernel-receipt:datasets-zk",
    )


def _passing_cases() -> dict[BackendTestCase, bool]:
    return {case: True for case in REQUIRED_BACKEND_TEST_CASES}


def _setup(
    tmp_path: Path,
    *,
    family: str = "provekit",
    mode: AttestationBackendMode = AttestationBackendMode.CRYPTOGRAPHIC,
) -> DatasetsZkSetupIdentity:
    exe = tmp_path / ("provekit-cli" if family == "provekit" else "groth16")
    exe.write_bytes(b"#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    if family == "provekit":
        (artifacts / "circuit.pkp").write_bytes(b"pk")
        (artifacts / "circuit.pkv").write_bytes(b"vk")
    else:
        (artifacts / "proving_key.bin").write_bytes(b"pk")
        (artifacts / "verifying_key.bin").write_bytes(b"vk")
    return build_datasets_zk_setup_identity(
        backend_family=family,
        backend_mode=mode,
        executable_path=str(exe),
        artifacts_path=str(artifacts),
        backend_version="0.2.0",
        circuit_version="1.0.0",
        public_input_schema_version="1.0.0",
        verification_key_version="ceremony-2026-07",
        verification_key_expires_at="2030-01-01T00:00:00Z",
    )


def _selection(
    setup: DatasetsZkSetupIdentity,
    *,
    available: bool = True,
) -> DatasetsZkBackendSelection:
    return DatasetsZkBackendSelection(
        backend_family=setup.backend_family,
        decision=datasets_verified_receipt_zk_use_case_decision(),
        setup=setup,
        selected_at=NOW,
        available=available,
        reason="test selection",
    )


def _adapter(
    tmp_path: Path,
    *,
    accept: bool = True,
    family: str = "provekit",
) -> tuple[IpfsDatasetsZkAttestation, DatasetsZkBackendSelection]:
    setup = _setup(tmp_path, family=family)
    selection = _selection(setup)

    def prover(request):
        digest = "sha256:" + ("a" * 64)
        return {
            "proof_artifact_id": "artifact:datasets-zk-proof",
            "proof_digest": digest,
            "statement_id": request.statement.statement_id,
        }

    def verifier(envelope):
        return bool(accept and envelope.proof_digest.startswith("sha256:"))

    adapter = IpfsDatasetsZkAttestation(
        prover=prover,
        verifier=verifier,
        preferred_families=(family,),
        setup_overrides={
            family: {
                "executable_path": setup.executable_path,
                "artifacts_path": setup.artifacts_path,
                "backend_version": setup.backend_version,
                "circuit_version": setup.circuit_version,
                "public_input_schema_version": setup.public_input_schema_version,
                "verification_key_version": setup.verification_key_version,
                "verification_key_expires_at": setup.verification_key_expires_at,
            }
        },
        secret_probes=[SECRET],
    )
    return adapter, selection


def test_evidence_term_and_interface_are_stable() -> None:
    assert SCAEV082REALZK == "SCAEV082REALZK"
    assert IPFS_DATASETS_ZK_ATTESTATION_INTERFACE == "IpfsDatasetsZkAttestation@1"
    assert DEFAULT_VERIFIED_RECEIPT_PREDICATE in APPROVED_VERIFIED_RECEIPT_PREDICATES
    decision = datasets_verified_receipt_zk_use_case_decision()
    assert decision is DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_DECISION
    assert decision.use_case_id == DATASETS_VERIFIED_RECEIPT_ZK_USE_CASE_ID
    assert decision.disposition is ZkUseCaseDisposition.APPROVED
    assert decision.authorizes_backend_family(ZkBackendFamily.PROVEKIT)
    assert decision.authorizes_backend_family(ZkBackendFamily.GROTH16)
    assert decision.is_terminal
    public = decision.to_public_artifact()
    assert public["decision_id"] == decision.decision_id
    assert ZkUseCaseDecisionRecord.from_dict(public) == decision


def test_require_zk_backend_selection_authorized_for_datasets_use_case() -> None:
    decision = datasets_verified_receipt_zk_use_case_decision()
    require_zk_backend_selection_authorized(decision, backend_family="provekit")
    require_zk_backend_selection_authorized(decision, backend_family="groth16")
    with pytest.raises(AttestationValidationError, match="not authorized"):
        require_zk_backend_selection_authorized(decision, backend_family="plonk")
    with pytest.raises(AttestationValidationError, match="simulated"):
        require_zk_backend_selection_authorized(
            decision, backend_family="simulated"
        )


def test_setup_identity_binds_executable_and_artifacts(tmp_path: Path) -> None:
    setup = _setup(tmp_path, family="provekit")
    assert setup.configured
    assert setup.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC
    assert setup.executable_digest.startswith("sha256:")
    assert setup.artifacts_digest
    policy = setup.to_backend_policy()
    assert policy.backend_id == "backend:datasets:provekit"
    assert policy.circuit_id == setup.circuit_id
    assert policy.verification_key_id == setup.verification_key_id
    artifact = setup.to_public_artifact()
    assert artifact["setup_id"] == setup.setup_id
    assert artifact["evidence_id"] == SCAEV082REALZK
    assert DatasetsZkSetupIdentity.from_dict(artifact) == setup


def test_probe_unavailable_without_binaries_is_typed_and_non_authoritative() -> None:
    selection = probe_datasets_zk_backend(
        preferred_families=(ZkBackendFamily.PROVEKIT, ZkBackendFamily.GROTH16),
        selected_at=NOW,
    )
    assert not selection.available
    assert selection.authorizes_production is False
    public = selection.to_public_artifact()
    assert public["available"] is False
    assert "backend" in selection.reason.lower() or "missing" in selection.reason.lower()


def test_self_tests_gate_production_eligibility(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    failing = run_datasets_zk_backend_self_tests(
        setup,
        available=True,
        evaluated_at=NOW,
        cases={case: (lambda: False) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    assert failing.status is CapabilityHealth.DEGRADED
    assert not failing.production_eligible
    with pytest.raises(Exception):
        failing.require_production_eligible()

    passing = run_datasets_zk_backend_self_tests(
        setup,
        available=True,
        evaluated_at=NOW,
        cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    assert passing.status is CapabilityHealth.VERIFIED
    assert passing.production_eligible
    assert set(passing.results_by_case) == set(REQUIRED_BACKEND_TEST_CASES)
    for case, result in passing.results_by_case.items():
        assert result.passed
        assert result.result_id
        assert result.case is case
    passing.require_production_eligible()


def test_cryptographic_attestation_is_authoritative(tmp_path: Path) -> None:
    adapter, selection = _adapter(tmp_path, accept=True)
    health = adapter.evaluate_backend_health(
        selection,
        evaluated_at=NOW,
        cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        backend_health=health,
        evaluated_at=NOW,
        self_test_cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )

    assert result.status is DatasetsZkStatus.ATTESTED
    assert result.authoritative
    assert result.verified
    assert result.trust is AttestationTrust.AUTHORITATIVE
    assert result.satisfies_production_gate()
    assert result.satisfies_completion_gate()
    assert result.satisfies_gate(AttestationGate.PRODUCTION)
    assert result.statement is not None
    assert result.statement.tree_id == "git-tree:datasets-zk-abc"
    assert result.statement_id == result.statement.statement_id
    assert result.verification is not None
    assert result.verification.verification_id == result.verification_id
    evidence = result.to_evidence()
    assert evidence.simulated is False
    public = result.to_public_artifact()
    assert public["result_id"] == result.result_id
    assert public["status"] == DatasetsZkStatus.ATTESTED.value
    assert not public_artifact_contains(public, SECRET)
    assert DatasetsZkAttestationResult.from_dict(public).result_id == result.result_id


def test_verifier_rejection_is_non_authoritative(tmp_path: Path) -> None:
    adapter, selection = _adapter(tmp_path, accept=False)
    health = adapter.evaluate_backend_health(
        selection,
        evaluated_at=NOW,
        cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        backend_health=health,
        evaluated_at=NOW,
    )
    assert result.status is DatasetsZkStatus.REJECTED
    assert not result.authoritative
    assert not result.satisfies_production_gate()
    assert not result.satisfies_completion_gate()


def test_unverified_receipt_cannot_attest(tmp_path: Path) -> None:
    adapter, selection = _adapter(tmp_path)
    health = adapter.evaluate_backend_health(
        selection,
        evaluated_at=NOW,
        cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    result = adapter.attest_verified_receipt(
        _receipt(proved=False),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        backend_health=health,
        evaluated_at=NOW,
    )
    assert result.status is DatasetsZkStatus.REJECTED
    assert result.diagnostic_code == "receipt_not_kernel_verified"
    assert not result.authoritative


def test_unsupported_predicate_is_not_applicable(tmp_path: Path) -> None:
    adapter, _selection = _adapter(tmp_path)
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        predicate="source_code_correctness",
        evaluated_at=NOW,
    )
    assert result.status is DatasetsZkStatus.NOT_APPLICABLE
    assert result.diagnostic_code == "predicate_not_in_closed_catalog"


def test_unavailable_backend_emits_typed_status(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    selection = _selection(setup, available=False)
    adapter = IpfsDatasetsZkAttestation()
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        evaluated_at=NOW,
    )
    assert result.status is DatasetsZkStatus.UNAVAILABLE
    assert result.diagnostic_code == "backend_unavailable"
    assert not result.authoritative
    assert not result.satisfies_gate(AttestationGate.COMPLETION)


def test_simulated_backend_never_satisfies_attested(tmp_path: Path) -> None:
    setup = build_datasets_zk_setup_identity(
        backend_family="simulated",
        backend_mode=AttestationBackendMode.SIMULATED,
        verification_key_id="vk:simulated",
    )
    selection = DatasetsZkBackendSelection(
        backend_family="simulated",
        decision=datasets_verified_receipt_zk_use_case_decision(),
        setup=setup,
        selected_at=NOW,
        available=True,
        reason="simulated path",
    )
    adapter = IpfsDatasetsZkAttestation()
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        evaluated_at=NOW,
    )
    assert result.status is DatasetsZkStatus.SIMULATED
    assert result.simulated
    assert not result.authoritative
    assert not result.satisfies_production_gate()
    assert not result.satisfies_completion_gate()


def test_failed_self_tests_block_real_zk_claims(tmp_path: Path) -> None:
    adapter, selection = _adapter(tmp_path)
    health = adapter.evaluate_backend_health(
        selection,
        evaluated_at=NOW,
        cases={case: (lambda: False) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    assert not health.production_eligible
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        backend_health=health,
        evaluated_at=NOW,
    )
    assert result.status is DatasetsZkStatus.DEGRADED
    assert result.diagnostic_code == "backend_not_production_eligible"
    assert not result.authoritative


def test_witness_cannot_enter_public_artifacts(tmp_path: Path) -> None:
    adapter, selection = _adapter(tmp_path)
    health = adapter.evaluate_backend_health(
        selection,
        evaluated_at=NOW,
        cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        backend_health=health,
        evaluated_at=NOW,
    )
    public = public_datasets_zk_artifact(result)
    assert not public_artifact_contains(public, SECRET)
    with pytest.raises(WitnessDisclosureError):
        public_datasets_zk_artifact(
            PrivateAttestationWitness({"receipt_opening": SECRET})
        )


def test_prover_failure_does_not_fall_back_to_simulation(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    selection = _selection(setup)

    def broken_prover(_request):
        raise RuntimeError("native prover exploded")

    adapter = IpfsDatasetsZkAttestation(
        prover=broken_prover,
        verifier=lambda _envelope: True,
    )
    health = run_datasets_zk_backend_self_tests(
        setup,
        available=True,
        evaluated_at=NOW,
        cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        backend_health=health,
        evaluated_at=NOW,
    )
    assert result.status is DatasetsZkStatus.ERROR
    assert result.diagnostic_code == "cryptographic_backend_failure"
    assert result.backend_mode is AttestationBackendMode.CRYPTOGRAPHIC
    assert not result.authoritative


def test_groth16_family_path(tmp_path: Path) -> None:
    adapter, selection = _adapter(tmp_path, family="groth16")
    assert selection.backend_family == "groth16"
    health = adapter.evaluate_backend_health(
        selection,
        evaluated_at=NOW,
        cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        backend_health=health,
        evaluated_at=NOW,
    )
    assert result.status is DatasetsZkStatus.ATTESTED
    assert result.backend_family == "groth16"


def test_registry_probe_is_boolean() -> None:
    assert isinstance(datasets_zkp_registry_available(), bool)


def test_result_to_cache_record_is_public(tmp_path: Path) -> None:
    adapter, selection = _adapter(tmp_path)
    health = adapter.evaluate_backend_health(
        selection,
        evaluated_at=NOW,
        cases={case: (lambda: True) for case in REQUIRED_BACKEND_TEST_CASES},
    )
    result = adapter.attest_verified_receipt(
        _receipt(),
        witness=PrivateAttestationWitness({"receipt_opening": SECRET}),
        selection=selection,
        backend_health=health,
        evaluated_at=NOW,
    )
    cached = result.to_cache_record()
    assert cached["result_id"] == result.result_id
    assert not public_artifact_contains(cached, SECRET)
    # simulated_attestation helper remains coherent for non-simulated success
    assert result.verification is not None
    assert not result.verification.simulated
    assert result.verification.authoritative_assurance is AssuranceLevel.ATTESTED
    assert simulated_attestation_cannot_satisfy_attested(result.verification)
