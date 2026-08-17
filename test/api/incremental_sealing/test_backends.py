"""IPS-029: probe backend capabilities; admit recursion only when demonstrated."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.backends import (
    BACKEND_CAPABILITY_EVIDENCE,
    DEFAULT_RECURSION_PROBE_MAX_STEPS,
    KNOWN_BACKEND_IDS,
    RECURSION_PROBE_EVIDENCE,
    TEST_ONLY_CIRCUIT_ID,
    TEST_ONLY_RECURSION_MATERIAL_ID,
    TRUST_BASELINE_BACKEND_DECISIONS,
    AggregationDisposition,
    BackendAvailabilityStatus,
    BackendCapabilityError,
    BackendCapabilityRegistry,
    CapabilityReasonCode,
    HermeticTestOnlyRecursiveBackend,
    ProofBackendCapability,
    RecursionProbeArtifact,
    RecursionProbeMaterial,
    RecursionProbeResult,
    RecursionProbeVerdict,
    closed_aggregation_dispositions,
    closed_capability_reason_codes,
    closed_known_backend_ids,
    probe_backend_capability,
    run_bounded_recursion_probe,
)


class _FailingRecursiveBackend:
    """Adapter that proves a child but fails recursive verification."""

    def prove_child(self, material: RecursionProbeMaterial) -> RecursionProbeArtifact:
        return HermeticTestOnlyRecursiveBackend().prove_child(material)

    def verify_child(
        self, artifact: RecursionProbeArtifact, material: RecursionProbeMaterial
    ) -> bool:
        return HermeticTestOnlyRecursiveBackend().verify_child(artifact, material)

    def prove_recursive(
        self,
        child: RecursionProbeArtifact,
        material: RecursionProbeMaterial,
    ) -> RecursionProbeArtifact:
        # Intentionally omit child_root binding.
        return RecursionProbeArtifact(
            kind="recursive",
            proof_bytes=b"\x00" * 32,
            public_input_digest=material.public_input_digest(),
            circuit_id=material.circuit_id,
            test_only=True,
            child_root=None,
        )

    def verify_recursive(
        self, artifact: RecursionProbeArtifact, material: RecursionProbeMaterial
    ) -> bool:
        return False


class _ExplodingRecursiveBackend:
    def prove_child(self, material: RecursionProbeMaterial) -> RecursionProbeArtifact:
        raise RuntimeError("backend exploded during prove_child")

    def verify_child(self, artifact, material) -> bool:
        return False

    def prove_recursive(self, child, material) -> RecursionProbeArtifact:
        raise RuntimeError("unreachable")

    def verify_recursive(self, artifact, material) -> bool:
        return False


def test_evidence_subsets_and_closed_vocabularies() -> None:
    assert BACKEND_CAPABILITY_EVIDENCE == "ips/backend-capability-matrix@1"
    assert RECURSION_PROBE_EVIDENCE == "ips/recursion-probe@1"
    assert closed_known_backend_ids() == KNOWN_BACKEND_IDS
    reasons = closed_capability_reason_codes()
    for required in (
        "unknown_backend",
        "backend_unavailable",
        "simulated_production_forbidden",
        "recursion_not_demonstrated",
        "recursion_probe_failed",
        "recursion_probe_passed",
    ):
        assert required in reasons
    dispositions = closed_aggregation_dispositions()
    assert "manifest_aggregation" in dispositions
    assert "recursive_verification" in dispositions
    assert TRUST_BASELINE_BACKEND_DECISIONS["unknown"] == "rejected"
    assert TRUST_BASELINE_BACKEND_DECISIONS["existing_recursive_backend"] == "unsupported"


def test_unknown_backend_fails_typed() -> None:
    cap = probe_backend_capability("exotic-recursive-snark")
    assert cap.status is BackendAvailabilityStatus.UNKNOWN
    assert cap.unknown is True
    assert cap.available is False
    assert cap.recursive_verification is False
    assert cap.can_prove is False
    assert cap.can_verify is False
    assert cap.can_sign is False
    assert cap.can_direct_computation is False
    assert cap.can_aggregate is False
    assert cap.production_seal_allowed is False
    assert cap.reason_code == CapabilityReasonCode.UNKNOWN_BACKEND.value
    assert cap.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION
    assert cap.recursion_probe.passed is False
    assert cap.recursion_probe.attempted is False
    assert "unknown" in cap.message.casefold()


def test_unavailable_provekit_fails_typed() -> None:
    cap = probe_backend_capability(
        "provekit",
        which=lambda _name: None,
        availability_overrides={"provekit": False},
    )
    assert cap.status is BackendAvailabilityStatus.UNAVAILABLE
    assert cap.unavailable is True
    assert cap.recursive_verification is False
    assert cap.reason_code == CapabilityReasonCode.OPTIONAL_CAPABILITY_UNAVAILABLE.value
    assert cap.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION
    assert "unavailable" in cap.message.casefold()


def test_unavailable_groth16_fails_typed() -> None:
    cap = probe_backend_capability(
        "groth16",
        availability_overrides={"groth16": False},
    )
    assert cap.status is BackendAvailabilityStatus.UNAVAILABLE
    assert cap.recursive_verification is False
    assert cap.reason_code == CapabilityReasonCode.BACKEND_UNAVAILABLE.value
    assert cap.can_prove is False


def test_simulated_forbids_production_and_recursion() -> None:
    cap = probe_backend_capability(
        "simulated",
        recursive_backend=HermeticTestOnlyRecursiveBackend(),
    )
    assert cap.status is BackendAvailabilityStatus.SIMULATED_ONLY
    assert cap.production_seal_allowed is False
    assert cap.recursive_verification is False
    assert cap.reason_code == CapabilityReasonCode.SIMULATED_PRODUCTION_FORBIDDEN.value
    assert cap.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION
    # Even with a recursive adapter present, simulated never admits recursion.
    assert cap.recursion_probe.passed is False


def test_default_recursion_is_explicitly_false_for_known_backends() -> None:
    for backend_id in ("integrity", "signed_receipt", "merkle_manifest", "groth16"):
        cap = probe_backend_capability(
            backend_id,
            availability_overrides={"groth16": True},
            which=lambda _n: None,
        )
        assert cap.recursive_verification is False, backend_id
        assert (
            cap.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION
        ), backend_id
        assert cap.recursion_probe.passed is False, backend_id


def test_integrity_and_signature_and_manifest_capabilities() -> None:
    integrity = probe_backend_capability("integrity")
    assert integrity.available is True
    assert integrity.can_verify is True
    assert integrity.can_prove is False
    assert integrity.can_sign is False
    assert integrity.supports_timeout is True
    assert integrity.supports_cancellation is True
    assert integrity.supports_resource_limits is True

    signed = probe_backend_capability("signed_receipt")
    assert signed.can_sign is True
    assert signed.can_verify is True
    assert signed.recursive_verification is False

    manifest = probe_backend_capability("merkle_manifest")
    assert manifest.can_aggregate is True
    assert manifest.recursive_verification is False
    assert manifest.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION


def test_recursion_admitted_only_after_bounded_prove_and_verify() -> None:
    adapter = HermeticTestOnlyRecursiveBackend()
    cap = probe_backend_capability(
        "groth16",
        recursive_backend=adapter,
        availability_overrides={"groth16": True},
    )
    assert cap.available is True
    assert cap.recursive_verification is True
    assert cap.aggregation_disposition is AggregationDisposition.RECURSIVE_VERIFICATION
    assert cap.recursion_probe.passed is True
    assert cap.recursion_probe.attempted is True
    assert cap.recursion_probe.prove_ok is True
    assert cap.recursion_probe.verify_ok is True
    assert cap.recursion_probe.recursive_prove_ok is True
    assert cap.recursion_probe.recursive_verify_ok is True
    assert cap.recursion_probe.bounded is True
    assert cap.recursion_probe.test_only_material is True
    assert cap.recursion_probe.material_id == TEST_ONLY_RECURSION_MATERIAL_ID
    assert cap.recursion_probe.steps_executed == 4
    assert cap.recursion_probe.steps_executed <= DEFAULT_RECURSION_PROBE_MAX_STEPS
    assert cap.reason_code == CapabilityReasonCode.RECURSION_PROBE_PASSED.value
    # Witness / proving-key secrecy on the probe record.
    probe_payload = cap.recursion_probe.to_canonical()
    assert probe_payload["witness_exported"] is False
    assert probe_payload["proving_key_exported"] is False
    assert "witness" not in probe_payload


def test_failed_recursion_probe_keeps_recursion_false() -> None:
    cap = probe_backend_capability(
        "groth16",
        recursive_backend=_FailingRecursiveBackend(),
        availability_overrides={"groth16": True},
    )
    assert cap.recursive_verification is False
    assert cap.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION
    assert cap.recursion_probe.attempted is True
    assert cap.recursion_probe.passed is False
    assert cap.recursion_probe.verdict is RecursionProbeVerdict.FAILED
    assert cap.recursion_probe.reason_code == CapabilityReasonCode.RECURSION_PROBE_FAILED.value


def test_exploding_recursion_probe_is_typed_error() -> None:
    cap = probe_backend_capability(
        "groth16",
        recursive_backend=_ExplodingRecursiveBackend(),
        availability_overrides={"groth16": True},
    )
    assert cap.recursive_verification is False
    assert cap.recursion_probe.verdict is RecursionProbeVerdict.ERROR
    assert cap.recursion_probe.reason_code == CapabilityReasonCode.RECURSION_PROBE_ERROR.value
    assert "exploded" in cap.recursion_probe.message


def test_recursion_probe_timeout_is_typed() -> None:
    material = RecursionProbeMaterial(timeout_seconds=0.0000001)
    # Force immediate timeout by using a clock that jumps past the bound.
    ticks = {"n": 0}

    def advancing_clock() -> float:
        ticks["n"] += 1
        # First call is start; subsequent calls exceed timeout.
        return 0.0 if ticks["n"] == 1 else 10.0

    result = run_bounded_recursion_probe(
        HermeticTestOnlyRecursiveBackend(),
        material,
        monotonic=advancing_clock,
    )
    assert result.passed is False
    assert result.verdict is RecursionProbeVerdict.TIMEOUT
    assert result.reason_code == CapabilityReasonCode.RECURSION_PROBE_TIMEOUT.value


def test_test_only_material_rejects_production_designation() -> None:
    with pytest.raises(BackendCapabilityError, match="test_only"):
        RecursionProbeMaterial(test_only=False)


def test_hermetic_probe_round_trip_directly() -> None:
    material = RecursionProbeMaterial()
    result = run_bounded_recursion_probe(
        HermeticTestOnlyRecursiveBackend(),
        material,
    )
    assert result.passed is True
    assert result.steps_executed <= material.max_steps
    assert result.child_root is not None
    assert result.aggregate_root is not None
    assert result.material_id == TEST_ONLY_RECURSION_MATERIAL_ID
    canonical = material.to_canonical()
    assert canonical["test_only"] is True
    assert canonical["witness_exported"] is False
    assert canonical["circuit_id"] == TEST_ONLY_CIRCUIT_ID


def test_registry_matrix_defaults_to_manifest_aggregation() -> None:
    registry = BackendCapabilityRegistry(
        backend_ids=("integrity", "signed_receipt", "merkle_manifest", "simulated", "provekit"),
        which=lambda _n: None,
        availability_overrides={"provekit": False},
    )
    matrix = registry.matrix()
    assert matrix.any_recursive_verification is False
    assert matrix.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION
    payload = matrix.to_canonical()
    assert payload["evidence_subset"] == BACKEND_CAPABILITY_EVIDENCE
    assert payload["any_recursive_verification"] is False
    assert payload["aggregation_disposition"] == "manifest_aggregation"
    assert payload["unsupported_recursion_fallback"] == "manifest_aggregation"
    assert "exotic" not in payload["probed_backend_ids"]

    # Unknown backend via require is typed error.
    unknown = registry.probe("not-a-real-backend")
    assert unknown.unknown is True
    with pytest.raises(BackendCapabilityError, match="unknown backend"):
        registry.require("not-a-real-backend")

    unavailable = registry.probe("provekit")
    assert unavailable.unavailable is True
    with pytest.raises(BackendCapabilityError, match="unavailable backend"):
        registry.require("provekit")


def test_registry_admits_recursion_only_for_probed_backend() -> None:
    adapter = HermeticTestOnlyRecursiveBackend()
    registry = BackendCapabilityRegistry(
        backend_ids=("groth16", "merkle_manifest"),
        recursive_backends={"groth16": adapter},
        availability_overrides={"groth16": True},
    )
    assert registry.recursive_verification_admitted("groth16") is True
    assert registry.recursive_verification_admitted("merkle_manifest") is False
    matrix = registry.matrix()
    assert matrix.any_recursive_verification is True
    assert matrix.aggregation_disposition is AggregationDisposition.RECURSIVE_VERIFICATION
    groth = registry.probe("groth16")
    assert groth.recursive_verification is True
    assert groth.supports_resource_limits is True
    assert groth.supports_timeout is True
    assert groth.supports_cancellation is True


def test_registry_cache_and_refresh() -> None:
    registry = BackendCapabilityRegistry(
        backend_ids=("integrity",),
    )
    first = registry.probe("integrity")
    second = registry.probe("integrity")
    assert first is second
    refreshed = registry.probe("integrity", refresh=True)
    assert refreshed is not first
    assert refreshed.to_canonical() == first.to_canonical()


def test_capability_canonical_json_is_stable() -> None:
    cap = probe_backend_capability("integrity")
    text = cap.to_canonical_json()
    assert text == cap.to_canonical_json()
    assert '"recursive_verification":false' in text
    assert BACKEND_CAPABILITY_EVIDENCE in text or "ips/backend-capability" in text


def test_proof_backend_capability_rejects_inconsistent_recursion() -> None:
    probe = RecursionProbeResult(
        schema="x",
        verdict=RecursionProbeVerdict.NOT_ATTEMPTED,
        attempted=False,
        passed=False,
        prove_ok=False,
        verify_ok=False,
        recursive_prove_ok=False,
        recursive_verify_ok=False,
        bounded=True,
        test_only_material=True,
        material_id=TEST_ONLY_RECURSION_MATERIAL_ID,
        steps_executed=0,
        duration_ms=0,
        reason_code="recursion_not_demonstrated",
        message="no",
    )
    with pytest.raises(BackendCapabilityError, match="passed recursion probe"):
        ProofBackendCapability(
            schema="s",
            backend_id="groth16",
            status=BackendAvailabilityStatus.AVAILABLE,
            can_prove=True,
            can_verify=True,
            can_sign=False,
            can_direct_computation=True,
            can_aggregate=True,
            recursive_verification=True,
            supports_resource_limits=True,
            supports_timeout=True,
            supports_cancellation=True,
            aggregation_disposition=AggregationDisposition.RECURSIVE_VERIFICATION,
            production_seal_allowed=True,
            reason_code="x",
            message="x",
            recursion_probe=probe,
            trust_baseline_decision="x",
        )


def test_allow_recursion_probe_false_never_attempts() -> None:
    cap = probe_backend_capability(
        "groth16",
        recursive_backend=HermeticTestOnlyRecursiveBackend(),
        availability_overrides={"groth16": True},
        allow_recursion_probe=False,
    )
    assert cap.recursive_verification is False
    assert cap.recursion_probe.attempted is False
    assert cap.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION


def test_provekit_available_without_recursion_stays_false() -> None:
    cap = probe_backend_capability(
        "provekit",
        which=lambda name: f"/usr/bin/{name}" if name == "provekit-cli" else None,
        availability_overrides={"provekit": True},
    )
    assert cap.available is True
    assert cap.can_prove is True
    assert cap.can_verify is True
    assert cap.recursive_verification is False
    assert cap.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION


def test_provekit_with_successful_probe_admits_recursion() -> None:
    cap = probe_backend_capability(
        "provekit",
        which=lambda name: f"/usr/bin/{name}",
        availability_overrides={"provekit": True},
        recursive_backend=HermeticTestOnlyRecursiveBackend(),
    )
    assert cap.recursive_verification is True
    assert cap.aggregation_disposition is AggregationDisposition.RECURSIVE_VERIFICATION


def test_empty_backend_id_raises() -> None:
    with pytest.raises(BackendCapabilityError, match="backend_id"):
        probe_backend_capability("  ")


def test_no_installation_side_effects_on_import_and_probe() -> None:
    # Probe must not require network or install; unavailable remains typed.
    cap = probe_backend_capability("provekit", which=lambda _n: None)
    assert cap.status in {
        BackendAvailabilityStatus.UNAVAILABLE,
        BackendAvailabilityStatus.AVAILABLE,
    }
    if cap.unavailable:
        assert cap.recursive_verification is False
