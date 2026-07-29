"""SCA-081 capability-checked MCP proof receipt attestation tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import (
    LOGIC_IR_PROFILE,
    identify_strict_artifact,
)
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
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_attestation import (
    AttestationBackendMode,
    AttestationBackendSetup,
    AttestationCapabilityReport,
    AttestationIdentityPin,
    AttestationPredicateKind,
    AttestationStatus,
    AttestationVerification,
    McpAttestationError,
    PrivateAttestationWitness,
    ProofAttestation,
    ProofAttestationPolicy,
    REQUIRED_CAPABILITY_FIXTURES,
    ReplayGuard,
    WitnessDisclosureError,
    ZkpAttestationAdapter,
    build_attestation_public_inputs,
    public_attestation_artifact,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_proof_cache import (
    IdentityBinding,
    ProofCacheKey,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_prover import (
    ContractProofRoute,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_attestation import (
    ZkUseCaseDisposition,
)


EVALUATED = "2026-07-29T12:00:00Z"
ISSUED = "2026-07-29T12:01:00Z"
CHECKED = "2026-07-29T12:02:00Z"
EXPIRES = "2026-07-29T12:05:00Z"


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=10_000,
        cpu_time_ms=8_000,
        memory_bytes=64 * 1024 * 1024,
        max_processes=2,
        max_premises=4,
        network_allowed=False,
    )


def _binding(
    name: str,
    logical_id: str | None = None,
    *,
    revision: int = 1,
) -> IdentityBinding:
    return IdentityBinding.from_identity(
        identify_strict_artifact(
            {"artifact": name, "revision": revision}
        ),
        logical_id=logical_id or f"{name}-1",
    )


def _cache_key(**changes: object) -> ProofCacheKey:
    values: dict[str, object] = {
        "snapshot": _binding("snapshot", "tree-1"),
        "scope": (_binding("scope", "scope-1"),),
        "property_catalog": _binding("property", "property-root-1"),
        "obligation": _binding("obligation", "obligation-1"),
        "premises": (_binding("premise", "premise-1"),),
        "assumptions": (),
        "provider": _binding("provider", "provider-1"),
        "translator": _binding("translator", "translator-1"),
        "solver": _binding("solver", "solver-1"),
        "kernel": _binding("kernel", "kernel-1"),
        "toolchain": _binding("toolchain", "toolchain-1"),
        "theorem_registry": _binding("registry", "registry-1"),
        "policy": _binding("proof-policy", "proof-policy-1"),
        "capability_report": _binding(
            "proof-capability", "proof-capability-1"
        ),
        "resource_budget": _budget(),
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
        "route": ContractProofRoute.KERNEL,
    }
    values.update(changes)
    return ProofCacheKey(**values)


def _receipt(**changes: object) -> ProofReceipt:
    obligation_id = "obligation-1"
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="kernel-artifact-1",
        subject_id=obligation_id,
        verifier_id="kernel-1",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
    )
    values: dict[str, object] = {
        "obligation_id": obligation_id,
        "plan_id": "plan-1",
        "attempt_id": "attempt-1",
        "repository_id": "repository-1",
        "repository_tree_id": "tree-1",
        "ast_scope_ids": ("scope-1",),
        "premise_ids": ("premise-1",),
        "translator_id": "translator-1",
        "solver_id": "solver-1",
        "kernel_id": "kernel-1",
        "toolchain_id": "toolchain-1",
        "theorem_registry_id": "registry-1",
        "policy_id": "proof-policy-1",
        "resource_budget": _budget(),
        "verdict": ProofVerdict.PROVED,
        "evidence": (evidence,),
        "freshness": EvidenceFreshness.CURRENT,
        "kernel_receipt_id": "kernel-receipt-1",
    }
    values.update(changes)
    return ProofReceipt(**values)


def _pin(name: str, *, revision: int = 1) -> AttestationIdentityPin:
    return AttestationIdentityPin.from_binding(
        _binding(name, f"{name}-{revision}", revision=revision)
    )


def _setup(
    mode: AttestationBackendMode = AttestationBackendMode.CRYPTOGRAPHIC,
    *,
    revision: int = 1,
) -> AttestationBackendSetup:
    return AttestationBackendSetup(
        backend_family="provekit",
        backend_mode=mode,
        backend_policy=_pin("backend-policy", revision=revision),
        backend_implementation=_pin(
            "backend-implementation", revision=revision
        ),
        setup_manifest=_pin("setup-manifest", revision=revision),
        circuit=_pin("circuit", revision=revision),
        public_input_schema=_pin("public-input-schema", revision=revision),
        proving_key=_pin("proving-key", revision=revision),
        verification_key=_pin("verification-key", revision=revision),
        backend_version=f"1.{revision}.0",
        circuit_version=f"2.{revision}.0",
        setup_version=f"ceremony-{revision}",
        key_epoch=f"epoch-{revision}",
        verification_key_expires_at="2030-01-01T00:00:00Z",
    )


def _capability(
    setup: AttestationBackendSetup | None = None,
    *,
    health: CapabilityHealth | None = None,
    failed_fixture: str = "",
) -> AttestationCapabilityReport:
    selected = setup or _setup()
    results = {
        fixture.value: fixture.value != failed_fixture
        for fixture in REQUIRED_CAPABILITY_FIXTURES
    }
    resolved_health = health or (
        CapabilityHealth.SIMULATED
        if selected.backend_mode is AttestationBackendMode.SIMULATED
        else (
            CapabilityHealth.DEGRADED
            if failed_fixture
            else CapabilityHealth.VERIFIED
        )
    )
    return AttestationCapabilityReport(
        setup=selected,
        health=resolved_health,
        configured=True,
        available=True,
        fixture_results=results,
        evaluated_at=EVALUATED,
        expires_at="2026-07-29T13:00:00Z",
    )


def _policy(
    *,
    disposition: ZkUseCaseDisposition = ZkUseCaseDisposition.APPROVED,
    revision: int = 1,
) -> ProofAttestationPolicy:
    approved = disposition is ZkUseCaseDisposition.APPROVED
    return ProofAttestationPolicy(
        use_case_id="external-receipt-membership",
        disposition=disposition,
        predicate_kind=AttestationPredicateKind.RECEIPT_MEMBERSHIP,
        use_case_decision=_pin("use-case-decision", revision=revision),
        predicate_manifest=_pin("predicate-manifest", revision=revision),
        verifier_domain="external-auditor.example/v1",
        reviewed_by="sca-security-review",
        reviewed_at="2026-07-29T00:00:00Z",
        expires_at="2030-01-01T00:00:00Z",
        qualifying_private_witness=approved,
        qualifying_cross_trust_boundary=approved,
        authorized_backend_families=("provekit",) if approved else (),
        required_base_assurance=AssuranceLevel.KERNEL_VERIFIED,
        max_proof_age_seconds=600,
        result_set_root_required=True,
    )


def _inputs(
    *,
    policy: ProofAttestationPolicy | None = None,
    capability: AttestationCapabilityReport | None = None,
    result_root: IdentityBinding | None = None,
):
    selected_policy = policy or _policy()
    selected_capability = capability or _capability()
    return build_attestation_public_inputs(
        _receipt(),
        _cache_key(),
        policy=selected_policy,
        capability_report=selected_capability,
        challenge="nonce:4c642fb4-76ef-46bf-bfb8-9320de23e855",
        issued_at=ISSUED,
        expires_at=EXPIRES,
        revocation_epoch="revocation-epoch-7",
        result_set_root=result_root
        or _binding("result-set-root", "result-set-root-1"),
    )


def _generated():
    policy = _policy()
    capability = _capability()
    inputs = _inputs(policy=policy, capability=capability)
    witness = PrivateAttestationWitness(
        {
            "private_leaf": b"receipt-leaf-never-public",
            "membership_path": b"private-path-never-public",
        }
    )
    adapter = ZkpAttestationAdapter(
        prover=lambda statement, private: (
            b"proof-v1:" + hashlib.sha256(statement).digest()
            if private["private_leaf"]
            else b""
        ),
        verifier=lambda proof, statement: proof
        == b"proof-v1:" + hashlib.sha256(statement).digest(),
    )
    attestation = adapter.attest(
        inputs,
        policy=policy,
        capability_report=capability,
        witness=witness,
    )
    return adapter, policy, capability, inputs, witness, attestation


def test_public_inputs_bind_every_cid_and_identity_profile() -> None:
    inputs = _inputs()
    public = inputs.to_public_artifact()
    required = {
        "receipt",
        "cache_key",
        "property",
        "snapshot",
        "attestation_policy",
        "backend_policy",
        "backend_implementation",
        "setup_manifest",
        "result_set_root",
        "capability_report",
    }
    for name in required:
        assert public[name]["cid"].startswith("b")
        assert public[name]["identity_profile_id"]
    assert set(inputs.identity_profile_ids) >= required
    assert inputs.statement_id.startswith("b")
    assert inputs.public_input_digest.startswith("sha256:")
    assert inputs.to_public_artifact() == json.loads(
        json.dumps(inputs.to_public_artifact())
    )


def test_receipt_and_cache_must_be_current_kernel_verified_and_exact() -> None:
    policy = _policy()
    capability = _capability()
    with pytest.raises(McpAttestationError) as stale:
        build_attestation_public_inputs(
            _receipt(freshness=EvidenceFreshness.STALE),
            _cache_key(),
            policy=policy,
            capability_report=capability,
            challenge="nonce-1",
            issued_at=ISSUED,
            expires_at=EXPIRES,
            revocation_epoch="epoch-1",
            result_set_root=_binding("root"),
        )
    assert stale.value.reason_code == "stale_cache_entry"

    with pytest.raises(McpAttestationError) as wrong_tree:
        build_attestation_public_inputs(
            _receipt(repository_tree_id="tree-other"),
            _cache_key(),
            policy=policy,
            capability_report=capability,
            challenge="nonce-1",
            issued_at=ISSUED,
            expires_at=EXPIRES,
            revocation_epoch="epoch-1",
            result_set_root=_binding("root"),
        )
    assert wrong_tree.value.reason_code == "wrong_repository_tree"


def test_witness_is_zeroized_and_never_serialized() -> None:
    _, _, _, _, witness, attestation = _generated()
    assert witness.zeroized
    assert "receipt-leaf-never-public" not in repr(witness)
    artifact = public_attestation_artifact(attestation)
    encoded = json.dumps(artifact, sort_keys=True)
    assert "receipt-leaf-never-public" not in encoded
    assert "membership_path" not in encoded
    assert artifact["private_witness_redacted"] is True
    with pytest.raises(WitnessDisclosureError):
        witness.to_dict()
    with pytest.raises(WitnessDisclosureError):
        public_attestation_artifact(
            {"nested": {"private_witness": "must-not-persist"}}
        )


def test_real_capability_checked_proof_attests_independently() -> None:
    adapter, policy, capability, inputs, _, attestation = _generated()
    assert attestation.status is AttestationStatus.GENERATED
    assert not attestation.authoritative
    assert not attestation.provider_verified

    verified = adapter.verify(
        attestation,
        expected_public_inputs=inputs,
        policy=policy,
        current_capability_report=capability,
        checked_at=CHECKED,
    )
    assert verified.status is AttestationStatus.ATTESTED
    assert verified.authoritative
    assert verified.assurance is AssuranceLevel.ATTESTED
    assert verified.independent

    # Authority is reproduced, never restored from serialized flags.
    with pytest.raises(McpAttestationError) as injected:
        AttestationVerification.from_dict(verified.to_dict())
    assert injected.value.reason_code == "authority_injection"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("receipt", _pin("forged-receipt")),
        ("property", _pin("other-property")),
        ("snapshot", _pin("other-snapshot")),
        ("result_set_root", _pin("other-result-root")),
        ("backend_policy", _pin("other-backend-policy")),
        ("setup_manifest", _pin("other-setup")),
    ],
)
def test_forged_root_and_backend_setup_substitution_fail_closed(
    field: str,
    replacement: AttestationIdentityPin,
) -> None:
    adapter, policy, capability, inputs, _, attestation = _generated()
    changed = replace(inputs, **{field: replacement})
    result = adapter.verify(
        attestation,
        expected_public_inputs=changed,
        policy=policy,
        current_capability_report=capability,
        checked_at=CHECKED,
    )
    assert result.status is AttestationStatus.REJECTED
    assert result.diagnostic_code == "public_input_mismatch"


def test_changed_policy_and_cross_profile_identity_fail_closed() -> None:
    adapter, policy, capability, inputs, _, attestation = _generated()
    changed_policy = _policy(revision=2)
    policy_result = adapter.verify(
        attestation,
        expected_public_inputs=inputs,
        policy=changed_policy,
        current_capability_report=capability,
        checked_at=CHECKED,
    )
    assert policy_result.status is AttestationStatus.REJECTED
    assert policy_result.diagnostic_code == "policy_mismatch"

    cross_profile = replace(
        inputs.receipt,
        identity_profile_id=LOGIC_IR_PROFILE,
    )
    changed_inputs = replace(inputs, receipt=cross_profile)
    profile_result = adapter.verify(
        attestation,
        expected_public_inputs=changed_inputs,
        policy=policy,
        current_capability_report=capability,
        checked_at=CHECKED,
    )
    assert profile_result.status is AttestationStatus.REJECTED
    assert profile_result.diagnostic_code == "public_input_mismatch"


def test_setup_drift_stops_before_prover_dispatch() -> None:
    policy = _policy()
    capability = _capability()
    inputs = replace(
        _inputs(policy=policy, capability=capability),
        setup_manifest=_pin("forged-setup"),
    )
    calls = 0

    def provider(_statement, _private):
        nonlocal calls
        calls += 1
        return b"must-not-run"

    witness = PrivateAttestationWitness({"private_leaf": b"secret"})
    result = ZkpAttestationAdapter(prover=provider).attest(
        inputs,
        policy=policy,
        capability_report=capability,
        witness=witness,
    )
    assert result.status is AttestationStatus.DEGRADED
    assert result.diagnostic_code == "backend_mismatch"
    assert calls == 0
    assert witness.zeroized


def test_replay_and_expiry_fail_closed() -> None:
    adapter, policy, capability, inputs, _, attestation = _generated()
    guard = ReplayGuard()
    first = adapter.verify(
        attestation,
        expected_public_inputs=inputs,
        policy=policy,
        current_capability_report=capability,
        checked_at=CHECKED,
        replay_guard=guard,
    )
    second = adapter.verify(
        attestation,
        expected_public_inputs=inputs,
        policy=policy,
        current_capability_report=capability,
        checked_at=CHECKED,
        replay_guard=guard,
    )
    expired = adapter.verify(
        attestation,
        expected_public_inputs=inputs,
        policy=policy,
        current_capability_report=capability,
        checked_at=EXPIRES,
    )
    assert first.status is AttestationStatus.ATTESTED
    assert second.status is AttestationStatus.REJECTED
    assert second.diagnostic_code == "replay_detected"
    assert expired.status is AttestationStatus.REJECTED
    assert expired.diagnostic_code == "replay_or_expired"


def test_capability_drift_and_malformed_proof_fail_closed() -> None:
    adapter, policy, capability, inputs, _, attestation = _generated()
    drifted = _capability(
        capability.setup,
        failed_fixture="malformed_proof",
    )
    drift_result = adapter.verify(
        attestation,
        expected_public_inputs=inputs,
        policy=policy,
        current_capability_report=drifted,
        checked_at=CHECKED,
    )
    assert drift_result.status is AttestationStatus.REJECTED
    assert drift_result.diagnostic_code == "capability_drift"

    malformed = ProofAttestation(
        public_inputs=inputs,
        status=AttestationStatus.GENERATED,
        backend_mode=AttestationBackendMode.CRYPTOGRAPHIC,
        capability_report_id=capability.capability_id,
        proof=b"x",
    )
    malformed_result = adapter.verify(
        malformed,
        expected_public_inputs=inputs,
        policy=policy,
        current_capability_report=capability,
        checked_at=CHECKED,
    )
    assert malformed_result.status is AttestationStatus.REJECTED
    assert malformed_result.diagnostic_code == "malformed_proof"

    public = attestation.to_public_artifact()
    public["proof_b64"] = "%%not-base64%%"
    with pytest.raises(McpAttestationError) as bad_encoding:
        ProofAttestation.from_dict(public)
    assert bad_encoding.value.reason_code == "malformed_proof"


def test_simulation_and_provider_verified_claim_never_promote() -> None:
    setup = _setup(AttestationBackendMode.SIMULATED)
    capability = _capability(setup, health=CapabilityHealth.SIMULATED)
    policy = _policy()
    inputs = _inputs(policy=policy, capability=capability)
    witness = PrivateAttestationWitness({"private_leaf": b"secret"})
    adapter = ZkpAttestationAdapter(
        prover=lambda _statement, _private: {
            "proof_bytes": b"simulated-proof",
            "verified": True,
        },
        verifier=lambda _proof, _statement: True,
    )
    attestation = adapter.attest(
        inputs,
        policy=policy,
        capability_report=capability,
        witness=witness,
    )
    assert attestation.status is AttestationStatus.SIMULATED
    assert attestation.provider_verified
    verified = adapter.verify(
        attestation,
        expected_public_inputs=inputs,
        policy=policy,
        current_capability_report=capability,
        checked_at=CHECKED,
    )
    assert verified.status is AttestationStatus.SIMULATED
    assert not verified.authoritative
    assert verified.assurance is AssuranceLevel.UNVERIFIED

    with pytest.raises(McpAttestationError) as promoted:
        replace(attestation, status=AttestationStatus.GENERATED)
    assert promoted.value.reason_code == "simulation_promotion"


def test_not_applicable_is_terminal_without_backend_dispatch() -> None:
    policy = _policy(disposition=ZkUseCaseDisposition.NOT_APPLICABLE)
    capability = _capability()
    inputs = _inputs(policy=policy, capability=capability)
    calls = 0

    def provider(_statement, _private):
        nonlocal calls
        calls += 1
        return b"must-not-run"

    witness = PrivateAttestationWitness({"private_leaf": b"secret"})
    result = ZkpAttestationAdapter(prover=provider).attest(
        inputs,
        policy=policy,
        capability_report=capability,
        witness=witness,
    )
    assert result.status is AttestationStatus.NOT_APPLICABLE
    assert calls == 0
    assert witness.zeroized
    assert not result.proof


def test_serialized_forgery_and_authority_injection_are_rejected() -> None:
    _, _, _, _, _, attestation = _generated()
    public = attestation.to_public_artifact()
    public["authoritative"] = True
    with pytest.raises(McpAttestationError) as authority:
        ProofAttestation.from_dict(public)
    assert authority.value.reason_code == "authority_injection"

    forged = attestation.to_public_artifact()
    forged["public_inputs"]["result_set_root"]["cid"] = _pin(
        "forged-root"
    ).cid
    with pytest.raises(McpAttestationError) as root:
        ProofAttestation.from_dict(forged)
    assert root.value.reason_code == "forged_root"

    unknown = attestation.to_public_artifact()
    unknown["provider_authority_extension"] = True
    with pytest.raises(McpAttestationError) as open_schema:
        ProofAttestation.from_dict(unknown)
    assert open_schema.value.reason_code == "invalid_schema"
