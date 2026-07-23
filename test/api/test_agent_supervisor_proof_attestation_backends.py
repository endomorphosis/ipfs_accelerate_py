from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_capabilities import (
    CapabilityHealth,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    EvidenceAuthority,
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
    AttestationValidationError,
    AttestationVerification,
    BackendHealthReport,
    BackendTestCase,
    BackendTestResult,
    BackendTestVerdict,
    CryptographicBackendFailure,
    PrivateAttestationWitness,
    REQUIRED_BACKEND_TEST_CASES,
    ReceiptAttestationStatement,
    evaluate_backend_health,
    execute_cryptographic_attestation,
    prepare_receipt_attestation,
    public_artifact_contains,
    witness_no_leak_test_result,
)

NOW = "2026-07-22T12:00:00Z"


def _policy(
    *,
    mode: AttestationBackendMode = AttestationBackendMode.CRYPTOGRAPHIC,
    expires_at: str = "2030-01-01T00:00:00Z",
) -> AttestationBackendPolicy:
    return AttestationBackendPolicy(
        backend_id=(
            "backend:simulated"
            if mode is AttestationBackendMode.SIMULATED
            else "backend:provekit"
        ),
        backend_version="0.1.7",
        circuit_id="circuit:receipt-binding",
        circuit_version="1.3.0",
        public_input_schema_id="schema:receipt-public-inputs",
        public_input_schema_version="1.1.0",
        verification_key_id="vk:receipt-binding:sha256-beef",
        verification_key_version="ceremony-2026-07",
        backend_mode=mode,
        verification_key_expires_at=expires_at,
    )


def _outcomes(verdict: bool = True) -> dict[BackendTestCase, bool]:
    return {case: verdict for case in REQUIRED_BACKEND_TEST_CASES}


def _health(
    policy: AttestationBackendPolicy | None = None,
    *,
    outcomes: dict[BackendTestCase, bool | BackendTestVerdict] | None = None,
    configured: bool = True,
    available: bool = True,
) -> BackendHealthReport:
    return evaluate_backend_health(
        policy or _policy(),
        configured=configured,
        available=available,
        outcomes=_outcomes() if outcomes is None else outcomes,
        evaluated_at=NOW,
    )


def _receipt() -> ProofReceipt:
    obligation_id = "obligation:backend-gate"
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
        plan_id="plan:backend-gate",
        attempt_id="attempt:kernel",
        repository_id="repo:test",
        repository_tree_id="git-tree:123",
        ast_scope_ids=("scope:test",),
        premise_ids=("premise:test",),
        translator_id="translator:test@1",
        solver_id="solver:test@1",
        kernel_id="kernel:lean@4.19",
        toolchain_id="toolchain:test",
        policy_id="policy:formal-production@1",
        resource_budget=ResourceBudget(
            wall_time_ms=1_000,
            memory_bytes=1_000_000,
            max_processes=1,
            network_allowed=False,
        ),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
    )


def _request(policy: AttestationBackendPolicy | None = None):
    checked = policy or _policy()
    return prepare_receipt_attestation(
        _receipt(),
        backend_policy=checked,
        witness=PrivateAttestationWitness(
            {"private_premise": "witness-secret-REF-265"}
        ),
    )


def test_health_distinguishes_all_backend_readiness_states() -> None:
    assert _health(_policy(mode=AttestationBackendMode.SIMULATED)).status is (
        CapabilityHealth.SIMULATED
    )
    assert _health(configured=False, available=False).status is (
        CapabilityHealth.UNAVAILABLE
    )
    assert _health(outcomes={}, available=False).status is CapabilityHealth.CONFIGURED
    assert _health(outcomes={}).status is CapabilityHealth.AVAILABLE
    failed = _outcomes()
    failed[BackendTestCase.NEGATIVE] = False
    assert _health(outcomes=failed).status is CapabilityHealth.DEGRADED
    assert _health().status is CapabilityHealth.VERIFIED
    assert _health().production_eligible


def test_policy_and_statement_pin_every_backend_version() -> None:
    policy = _policy()
    statement = _request(policy).statement

    assert statement.matches_backend_policy(policy)
    assert statement.public_inputs["backend_policy_id"] == policy.policy_id
    assert statement.public_inputs["backend_version"] == policy.backend_version
    assert statement.public_inputs["circuit_version"] == policy.circuit_version
    assert (
        statement.public_inputs["public_input_schema_version"]
        == policy.public_input_schema_version
    )
    assert (
        statement.public_inputs["verification_key_version"]
        == policy.verification_key_version
    )
    assert ReceiptAttestationStatement.from_dict(statement.to_dict()) == statement

    for field_name in (
        "backend_version",
        "circuit_version",
        "public_input_schema_version",
        "verification_key_version",
    ):
        changed = replace(policy, **{field_name: "different-version"})
        assert changed.policy_id != policy.policy_id
        assert not statement.matches_backend_policy(changed)


@pytest.mark.parametrize("case", REQUIRED_BACKEND_TEST_CASES)
def test_every_required_case_gates_production_eligibility(
    case: BackendTestCase,
) -> None:
    missing = _outcomes()
    missing.pop(case)
    assert _health(outcomes=missing).status is CapabilityHealth.AVAILABLE
    assert not _health(outcomes=missing).production_eligible

    failed = _outcomes()
    failed[case] = False
    assert _health(outcomes=failed).status is CapabilityHealth.DEGRADED
    with pytest.raises(CryptographicBackendFailure):
        _health(outcomes=failed).require_production_eligible()


def test_stale_key_and_forged_health_identity_fail_closed() -> None:
    current = _health()
    assert BackendHealthReport.from_dict(current.to_public_artifact()) == current

    stale = _health(_policy(expires_at="2026-01-01T00:00:00Z"))
    assert stale.status is CapabilityHealth.DEGRADED
    assert "stale" in stale.reason

    payload = _health().to_public_artifact()
    payload["production_eligible"] = False
    with pytest.raises(AttestationValidationError, match="production_eligible"):
        BackendHealthReport.from_dict(payload)

    with pytest.raises(AttestationValidationError, match="must also be configured"):
        _health(configured=False, available=True)

    policy = _policy()
    future_results = tuple(
        BackendTestResult(
            case=case,
            verdict=BackendTestVerdict.PASSED,
            backend_policy_id=policy.policy_id,
            observed_at="2031-01-01T00:00:00Z",
        )
        for case in REQUIRED_BACKEND_TEST_CASES
    )
    future = BackendHealthReport(
        policy=policy,
        configured=True,
        available=True,
        test_results=future_results,
        evaluated_at=NOW,
    )
    assert future.status is CapabilityHealth.DEGRADED
    assert not future.production_eligible


def test_witness_no_leak_evidence_contains_neither_probe_nor_field_name() -> None:
    policy = _policy()
    secret = "witness-secret-REF-265"
    result = witness_no_leak_test_result(
        policy,
        artifacts=[{"proof": "public", "private_witness_redacted": True}],
        secret_probes=[secret],
        observed_at=NOW,
    )

    assert result.passed
    assert not public_artifact_contains(result, secret)
    assert not public_artifact_contains(result, "private_premise")


def test_cryptographic_failures_never_fall_back_to_simulation() -> None:
    policy = _policy()
    request = _request(policy)
    calls = {"cryptographic": 0, "simulated": 0}

    def failed_prover(_request):
        calls["cryptographic"] += 1
        raise RuntimeError("cryptographic failure")

    def simulated_fallback(_request):
        calls["simulated"] += 1
        return {"proof_artifact_id": "artifact:fake", "proof_digest": "sha256:fake"}

    with pytest.raises(CryptographicBackendFailure, match="generation failed"):
        execute_cryptographic_attestation(
            request,
            backend_health=_health(policy),
            prover=failed_prover,
            verifier=lambda _envelope: True,
            prover_id="prover:provekit",
            verifier_id="verifier:provekit",
        )

    # There is deliberately no fallback callback in the execution API.
    assert calls == {"cryptographic": 1, "simulated": 0}
    assert callable(simulated_fallback)


def test_golden_execution_is_authoritative_but_rejection_is_not() -> None:
    policy = _policy()
    kwargs = {
        "backend_health": _health(policy),
        "prover": lambda _request: {
            "proof_artifact_id": "artifact:proof",
            "proof_digest": "sha256:proof",
        },
        "prover_id": "prover:provekit@0.1.7",
        "verifier_id": "verifier:provekit@0.1.7",
    }
    accepted = execute_cryptographic_attestation(
        _request(policy),
        verifier=lambda _envelope: True,
        **kwargs,
    )
    rejected = execute_cryptographic_attestation(
        _request(policy),
        verifier=lambda _envelope: False,
        **kwargs,
    )

    assert accepted.authoritative and accepted.satisfies_production_gate()
    assert AttestationVerification.from_dict(accepted.to_dict()) == accepted
    assert not rejected.authoritative
    assert not rejected.satisfies_production_gate()
