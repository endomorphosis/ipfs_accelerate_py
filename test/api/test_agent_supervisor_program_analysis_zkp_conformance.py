"""Production ZK capability, ceremony, and verifier conformance (VFS-024).

Probes executable/architecture, backend, circuit version, setup artifacts,
ceremony, proving/verifying keys, public-input codec, proof schema, independent
verifier, bounds, and cancellation. Simulated defaults, knowledge-graph
fail-open fallback, placeholder field encoding, v1 nonzero-only circuits,
incompatible TDFOL-only circuits, unversioned or missing artifacts, and stale
capabilities must fail closed for authority.
"""

from __future__ import annotations

import hashlib
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.program_analysis_zkp import (
    BN254_SCALAR_FIELD_MODULUS,
    FIELD_ENCODING_BN254_SHA256,
    OBJECTIVE_GOAL_G081_ID,
    OBJECTIVE_TASK_G081_ID,
    PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
    PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION,
    PROGRAM_ZKP_CAPABILITY_CONFORMANCE_CLAIM_SCHEMA,
    PROGRAM_ZKP_EVIDENCE_CAPABILITY_CONFORMANCE,
    PROGRAM_ZKP_G081_EVIDENCE_TERMS,
    PROGRAM_ZKP_PROOF_SCHEMA_ID,
    PUBLIC_INPUT_CODEC_ID,
    PUBLIC_INPUT_CODEC_VERSION,
    REQUIRED_CAPABILITY_DIMENSIONS,
    ProgramZkpAuthorityDenialReason,
    ProgramZkpAuthorityError,
    ProgramZkpBackendMode,
    ProgramZkpCapabilityConformanceReport,
    ProgramZkpCapabilityDimension,
    ProgramZkpCapabilityError,
    ProgramZkpCircuitFamily,
    ProgramZkpClaimPromotionError,
    ProgramZkpFieldEncodingKind,
    ProgramZkpRolloutMode,
    ProgramZkpTamperError,
    ProgramZkpTrust,
    ProgramZkpVerdict,
    all_program_zkp_evidence_terms,
    build_production_ready_capability_fixture,
    build_program_zkp_public_inputs,
    capability_conformance_evidence_terms,
    classify_circuit_family,
    commitment_identity,
    create_program_zkp_shadow_envelope,
    encode_public_input_field_vector,
    field_element_from_text,
    grants_production_authority,
    invalidate_authority_on_capability_loss,
    prepare_program_analysis_zkp,
    probe_program_analysis_zkp_capability,
    proof_bytes_are_simulated,
    prove_zk_capability_conformance,
    record_production_program_zkp_verification,
    record_program_zkp_verification,
    reject_illegal_zk_claim_promotion,
    require_production_authority,
    rollout_mode_for_capability,
    shadow_only_rollout,
    verify_program_zkp_independently,
    PrivateProgramAnalysisWitness,
)
from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import ClaimLevel


def _public_inputs(**overrides: str):
    base = {
        "forest_commitment": commitment_identity("forest", {"root": "repo:alpha"}),
        "inventory_commitment": commitment_identity(
            "inventory", {"files": ["a.py", "b.py"]}
        ),
        "contract_commitment": commitment_identity(
            "contract", {"symbol": "pkg.api.call"}
        ),
        "call_slice_commitment": commitment_identity(
            "call_slice", {"path": ["main", "pkg.api.call"]}
        ),
        "assumptions_commitment": commitment_identity(
            "assumptions", {"items": ["finite_bounds", "hermetic_fixture"]}
        ),
        "analyzer_version": "analyzer:program-graph@1.0.0",
        "resolver_version": "resolver:call@2.1.0",
        "translator_version": "translator:contract-ir@1.3.0",
        "prover_version": "prover:program-contract-trace@0.1.0",
        "result_commitment": commitment_identity(
            "result", {"status": "contract_check_ok", "finite": True}
        ),
        "circuit_id": PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
        "proving_key_id": "pk:program-contract-trace@1:sha256-pk-fixture",
        "verifying_key_id": "vk:program-contract-trace@1:sha256-vk-fixture",
        "ceremony_id": "ceremony:program-contract-trace@1",
        "public_input_codec_id": PUBLIC_INPUT_CODEC_ID,
        "public_input_codec_version": PUBLIC_INPUT_CODEC_VERSION,
    }
    base.update(overrides)
    return build_program_zkp_public_inputs(**base)


def _witness() -> PrivateProgramAnalysisWitness:
    return PrivateProgramAnalysisWitness(
        {
            "source_text": "def call():\n    return 1\n",
            "commitment_opening": "opening-secret",
        }
    )


def _shadow_envelope(**public_overrides: str):
    request = prepare_program_analysis_zkp(
        _public_inputs(**public_overrides),
        witness=_witness(),
        backend_mode=ProgramZkpBackendMode.SHADOW,
    )
    return create_program_zkp_shadow_envelope(
        request,
        proof_artifact_id="artifact:zk-proof-shadow",
        proof_digest="sha256:" + ("ab" * 32),
        prover_id="prover:shadow-worker",
    )


def _crypto_envelope(**public_overrides: str):
    request = prepare_program_analysis_zkp(
        _public_inputs(**public_overrides),
        witness=_witness(),
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
    )
    return create_program_zkp_shadow_envelope(
        request,
        proof_artifact_id="artifact:zk-proof-crypto",
        proof_digest="sha256:" + ("cd" * 32),
        prover_id="prover:crypto-worker",
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
    )


def _honest_crypto_verify(
    proof: bytes, key: bytes, fields: tuple[int, ...]
) -> bool:
    del fields
    return bool(proof) and bool(key) and not proof_bytes_are_simulated(proof)


# ---------------------------------------------------------------------------
# Capability probe dimensions
# ---------------------------------------------------------------------------


def test_default_probe_is_shadow_only_and_non_authoritative() -> None:
    report = probe_program_analysis_zkp_capability()
    assert report.production_eligible is False
    assert report.shadow_only is True
    assert report.rollout_mode is ProgramZkpRolloutMode.SHADOW
    assert report.authoritative_allowed is False
    assert (
        report.to_dict()["evidence"] == PROGRAM_ZKP_EVIDENCE_CAPABILITY_CONFORMANCE
    )
    dims = {check.dimension for check in report.checks}
    assert dims == set(REQUIRED_CAPABILITY_DIMENSIONS)
    assert ProgramZkpAuthorityDenialReason.SHADOW_ONLY_ROLLOUT.value in (
        report.denial_reasons
    )


def test_probe_covers_every_required_dimension() -> None:
    report = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        backend_id="backend:provekit@1",
        circuit_id=PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
        proving_key_id="pk:program-contract-trace@1",
        verifying_key_id="vk:program-contract-trace@1",
        ceremony_id="ceremony:program-contract-trace@1",
        independent_verifier_available=True,
        verifier_id="verifier:x@1",
    )
    ordered = [check.dimension for check in report.checks]
    assert ordered == list(REQUIRED_CAPABILITY_DIMENSIONS)
    for dimension in REQUIRED_CAPABILITY_DIMENSIONS:
        assert dimension in report.checks_by_dimension


def test_production_fixture_is_eligible_and_round_trips() -> None:
    report = build_production_ready_capability_fixture()
    assert report.production_eligible is True
    assert report.rollout_mode is ProgramZkpRolloutMode.ENFORCEMENT
    assert shadow_only_rollout(report) is False
    assert rollout_mode_for_capability(report) is ProgramZkpRolloutMode.ENFORCEMENT
    assert all(check.production_eligible for check in report.checks)
    payload = report.to_public_artifact()
    assert payload["capability_epoch"] == report.capability_epoch
    assert payload["evidence"] == PROGRAM_ZKP_EVIDENCE_CAPABILITY_CONFORMANCE
    restored = ProgramZkpCapabilityConformanceReport.from_dict(payload)
    assert restored == report
    assert restored.capability_epoch == report.capability_epoch


# ---------------------------------------------------------------------------
# Fail-closed authority cases
# ---------------------------------------------------------------------------


def test_simulated_defaults_fail_closed() -> None:
    report = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.SIMULATED,
        backend_id="backend:simulated-v0.1",
    )
    assert report.production_eligible is False
    assert (
        ProgramZkpAuthorityDenialReason.SIMULATED_DEFAULT.value
        in report.denial_reasons
    )
    backend = report.checks_by_dimension[ProgramZkpCapabilityDimension.BACKEND]
    assert backend.production_eligible is False
    with pytest.raises(ProgramZkpAuthorityError, match="production ZK authority denied"):
        report.require_production_eligible()


def test_knowledge_graph_fail_open_fails_closed() -> None:
    report = build_production_ready_capability_fixture()
    # Reconstruct with fail-open flag forced.
    tainted = ProgramZkpCapabilityConformanceReport(
        checks=report.checks,
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        field_encoding=ProgramZkpFieldEncodingKind.BN254_SHA256,
        circuit_family=ProgramZkpCircuitFamily.PROGRAM_CONTRACT_TRACE,
        circuit_id=PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
        circuit_version=PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION,
        knowledge_graph_fail_open=True,
        stale=False,
        architecture=report.architecture,
        cancellation_supported=True,
    )
    assert tainted.production_eligible is False
    assert (
        ProgramZkpAuthorityDenialReason.KNOWLEDGE_GRAPH_FAIL_OPEN.value
        in tainted.denial_reasons
    )


def test_placeholder_field_encoding_fails_closed() -> None:
    report = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        field_encoding=ProgramZkpFieldEncodingKind.PLACEHOLDER,
        circuit_id=PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
        proving_key_id="pk:x@1",
        verifying_key_id="vk:x@1",
        ceremony_id="ceremony:x@1",
        ceremony_production_eligible=True,
        independent_verifier_available=True,
    )
    assert report.production_eligible is False
    assert (
        ProgramZkpAuthorityDenialReason.PLACEHOLDER_FIELD_ENCODING.value
        in report.denial_reasons
    )
    with pytest.raises(ProgramZkpAuthorityError, match="placeholder"):
        encode_public_input_field_vector(
            _public_inputs(),
            field_encoding=ProgramZkpFieldEncodingKind.PLACEHOLDER,
        )


def test_v1_nonzero_only_circuit_fails_closed() -> None:
    report = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        circuit_id="circuit:nonzero-only@1",
        circuit_family=ProgramZkpCircuitFamily.NONZERO_ONLY_V1,
        field_encoding=ProgramZkpFieldEncodingKind.NONZERO_ONLY_V1,
        proving_key_id="pk:nonzero@1",
        verifying_key_id="vk:nonzero@1",
        ceremony_id="ceremony:nonzero@1",
        ceremony_production_eligible=True,
        independent_verifier_available=True,
    )
    assert report.production_eligible is False
    assert (
        ProgramZkpAuthorityDenialReason.V1_NONZERO_ONLY_CIRCUIT.value
        in report.denial_reasons
    )
    assert classify_circuit_family("circuit:foo-nonzero-only@2") is (
        ProgramZkpCircuitFamily.NONZERO_ONLY_V1
    )


def test_incompatible_tdfol_only_circuit_fails_closed() -> None:
    report = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        circuit_id="circuit:tdfol-theorem@1",
        circuit_family=ProgramZkpCircuitFamily.TDFOL_ONLY,
        proving_key_id="pk:tdfol@1",
        verifying_key_id="vk:tdfol@1",
        ceremony_id="ceremony:tdfol@1",
        ceremony_production_eligible=True,
        independent_verifier_available=True,
    )
    assert report.production_eligible is False
    assert (
        ProgramZkpAuthorityDenialReason.INCOMPATIBLE_TDFOL_ONLY_CIRCUIT.value
        in report.denial_reasons
    )
    assert classify_circuit_family("circuit:tdfol-only@9") is (
        ProgramZkpCircuitFamily.TDFOL_ONLY
    )


def test_unversioned_and_missing_artifacts_fail_closed() -> None:
    report = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        circuit_id="circuit:program-contract-trace",  # unversioned
        circuit_version=PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION,
        proving_key_id="pk-without-version",
        verifying_key_id="",
        ceremony_id="",
        setup_dir="/nonexistent/setup/dir",
        independent_verifier_available=False,
    )
    assert report.production_eligible is False
    reasons = set(report.denial_reasons)
    assert ProgramZkpAuthorityDenialReason.UNVERSIONED_ARTIFACT.value in reasons
    assert ProgramZkpAuthorityDenialReason.MISSING_ARTIFACT.value in reasons
    assert (
        ProgramZkpAuthorityDenialReason.INDEPENDENT_VERIFIER_ABSENT.value in reasons
    )


def test_stale_capability_fails_closed() -> None:
    report = build_production_ready_capability_fixture()
    stale = ProgramZkpCapabilityConformanceReport(
        checks=report.checks,
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        field_encoding=ProgramZkpFieldEncodingKind.BN254_SHA256,
        circuit_family=ProgramZkpCircuitFamily.PROGRAM_CONTRACT_TRACE,
        circuit_id=PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
        circuit_version=PROGRAM_CONTRACT_TRACE_CIRCUIT_VERSION,
        knowledge_graph_fail_open=False,
        stale=True,
        architecture=report.architecture,
        cancellation_supported=True,
    )
    assert stale.production_eligible is False
    assert (
        ProgramZkpAuthorityDenialReason.STALE_CAPABILITY.value in stale.denial_reasons
    )
    assert stale.shadow_only is True


# ---------------------------------------------------------------------------
# Field encoding and proof schema
# ---------------------------------------------------------------------------


def test_bn254_field_encoding_is_deterministic() -> None:
    value = "circuit:program-contract-trace@1"
    a = field_element_from_text(value)
    b = field_element_from_text(value)
    assert a == b
    assert 0 <= a < BN254_SCALAR_FIELD_MODULUS
    assert FIELD_ENCODING_BN254_SHA256 == ProgramZkpFieldEncodingKind.BN254_SHA256.value
    fields = encode_public_input_field_vector(_public_inputs())
    assert len(fields) == 16
    assert all(0 < item < BN254_SCALAR_FIELD_MODULUS for item in fields)


def test_simulated_proof_layout_is_detected() -> None:
    padding = b"\x00" * (160 - 8)
    sim = b"SIMZKP\x00\x01" + padding
    assert len(sim) == 160
    assert proof_bytes_are_simulated(sim) is True
    assert proof_bytes_are_simulated(b"real-proof-bytes-not-sim") is False


# ---------------------------------------------------------------------------
# Deterministic receipt identity, independent replay, corruption rejection
# ---------------------------------------------------------------------------


def test_deterministic_proof_receipt_identity() -> None:
    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    proof = b"production-proof-bytes-v1"
    key = b"vk-fixture-material-v1"

    first = verify_program_zkp_independently(
        envelope,
        capability=capability,
        verifier_id="verifier:program-analysis-zkp-independent@1",
        proof_bytes=proof,
        verifying_key_material=key,
        cryptographic_verify=_honest_crypto_verify,
    )
    second = verify_program_zkp_independently(
        envelope,
        capability=capability,
        verifier_id="verifier:program-analysis-zkp-independent@1",
        proof_bytes=proof,
        verifying_key_material=key,
        cryptographic_verify=_honest_crypto_verify,
    )
    assert first.receipt_id == second.receipt_id
    assert first.content_id == second.content_id
    assert first.authoritative is True
    assert first.trust is ProgramZkpTrust.AUTHORITATIVE
    assert first.capability_epoch == capability.capability_epoch
    assert first.independent_verifier is True
    assert first.proof_schema_id == PROGRAM_ZKP_PROOF_SCHEMA_ID


def test_independent_replay_against_exact_pins() -> None:
    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    receipt = verify_program_zkp_independently(
        envelope,
        capability=capability,
        verifier_id="verifier:independent@1",
        proof_bytes=b"ok-proof",
        verifying_key_material=b"vk-fixture-material-v1",
        cryptographic_verify=_honest_crypto_verify,
    )
    pins = envelope.statement.public_inputs
    receipt.require_replay(
        public_inputs=pins,
        verifying_key_id=pins.verifying_key_id,
        circuit_id=pins.circuit_id,
        ceremony_id=pins.ceremony_id,
        public_input_codec_version=PUBLIC_INPUT_CODEC_VERSION,
        capability_epoch=capability.capability_epoch,
    )


def test_corrupted_proof_is_rejected() -> None:
    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    with pytest.raises(ProgramZkpAuthorityError, match="simulated proof"):
        verify_program_zkp_independently(
            envelope,
            capability=capability,
            verifier_id="verifier:independent@1",
            proof_bytes=b"SIMZKP\x00\x01" + b"\x00" * 152,
            verifying_key_material=b"vk-fixture-material-v1",
            cryptographic_verify=_honest_crypto_verify,
        )
    with pytest.raises(ProgramZkpTamperError, match="empty"):
        verify_program_zkp_independently(
            envelope,
            capability=capability,
            verifier_id="verifier:independent@1",
            proof_bytes=b"",
            verifying_key_material=b"vk-fixture-material-v1",
            cryptographic_verify=_honest_crypto_verify,
        )


def test_corrupted_key_is_rejected() -> None:
    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    expected = "sha256:" + hashlib.sha256(b"vk-fixture-material-v1").hexdigest()
    with pytest.raises(ProgramZkpTamperError, match="verifying key"):
        verify_program_zkp_independently(
            envelope,
            capability=capability,
            verifier_id="verifier:independent@1",
            proof_bytes=b"ok-proof",
            verifying_key_material=b"tampered-key-material",
            expected_verifying_key_digest=expected,
            cryptographic_verify=_honest_crypto_verify,
        )


def test_corrupted_public_input_is_rejected() -> None:
    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    drifted = _public_inputs(
        forest_commitment=commitment_identity("forest", {"root": "other"})
    )
    with pytest.raises(ProgramZkpTamperError, match="public inputs"):
        verify_program_zkp_independently(
            envelope,
            capability=capability,
            verifier_id="verifier:independent@1",
            proof_bytes=b"ok-proof",
            verifying_key_material=b"vk-fixture-material-v1",
            public_inputs=drifted,
            cryptographic_verify=_honest_crypto_verify,
        )


def test_cryptographic_without_callback_fails_closed() -> None:
    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    receipt = verify_program_zkp_independently(
        envelope,
        capability=capability,
        verifier_id="verifier:independent@1",
        proof_bytes=b"ok-proof",
        verifying_key_material=b"vk-fixture-material-v1",
        cryptographic_verify=None,
    )
    assert receipt.verdict is ProgramZkpVerdict.REJECTED
    assert receipt.authoritative is False


def test_production_record_requires_eligible_capability_and_callback() -> None:
    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    receipt = record_production_program_zkp_verification(
        envelope,
        capability=capability,
        verifier_id="verifier:independent@1",
        proof_bytes=b"ok-proof",
        verifying_key_material=b"vk-fixture-material-v1",
        cryptographic_verify=_honest_crypto_verify,
    )
    assert receipt.authoritative is True
    assert grants_production_authority(receipt, capability) is True
    require_production_authority(receipt, capability)

    shadow_cap = probe_program_analysis_zkp_capability()
    with pytest.raises(ProgramZkpAuthorityError):
        record_production_program_zkp_verification(
            envelope,
            capability=shadow_cap,
            verifier_id="verifier:independent@1",
            proof_bytes=b"ok-proof",
            verifying_key_material=b"vk-fixture-material-v1",
            cryptographic_verify=_honest_crypto_verify,
        )


# ---------------------------------------------------------------------------
# Capability loss invalidation and shadow-only rollout
# ---------------------------------------------------------------------------


def test_capability_loss_invalidates_prior_authority() -> None:
    previous = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    receipt = verify_program_zkp_independently(
        envelope,
        capability=previous,
        verifier_id="verifier:independent@1",
        proof_bytes=b"ok-proof",
        verifying_key_material=b"vk-fixture-material-v1",
        cryptographic_verify=_honest_crypto_verify,
    )
    assert receipt.authoritative is True

    # New probe with different architecture → different capability epoch.
    current = build_production_ready_capability_fixture(
        architecture="fixture-linux-aarch64"
    )
    assert current.capability_epoch != previous.capability_epoch
    assert grants_production_authority(receipt, current) is False
    with pytest.raises(ProgramZkpAuthorityError, match="authority denied"):
        require_production_authority(receipt, current)

    invalidated = invalidate_authority_on_capability_loss(
        receipt,
        previous_capability=previous,
        current_capability=current,
    )
    assert invalidated.authoritative is False
    assert invalidated.capability_production_eligible is False
    assert invalidated.capability_epoch == current.capability_epoch
    assert invalidated.verdict is receipt.verdict


def test_capability_loss_when_probe_degrades() -> None:
    previous = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    receipt = verify_program_zkp_independently(
        envelope,
        capability=previous,
        verifier_id="verifier:independent@1",
        proof_bytes=b"ok-proof",
        verifying_key_material=b"vk-fixture-material-v1",
        cryptographic_verify=_honest_crypto_verify,
    )
    degraded = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.SIMULATED
    )
    assert degraded.production_eligible is False
    invalidated = invalidate_authority_on_capability_loss(
        receipt,
        previous_capability=previous,
        current_capability=degraded,
    )
    assert invalidated.authoritative is False
    with pytest.raises(ProgramZkpCapabilityError, match="capability loss"):
        receipt.require_capability_epoch(degraded.capability_epoch)


def test_shadow_only_rollout_never_authoritative() -> None:
    envelope = _shadow_envelope()
    capability = probe_program_analysis_zkp_capability()
    assert capability.shadow_only is True
    receipt = record_program_zkp_verification(
        envelope,
        verdict=ProgramZkpVerdict.VERIFIED,
        verifier_id="verifier:shadow@1",
        independent_verifier=True,
    )
    assert receipt.verified is True
    assert receipt.authoritative is False
    assert grants_production_authority(receipt, capability) is False

    # Independent structural verification of shadow still non-authoritative.
    shadow_receipt = verify_program_zkp_independently(
        envelope,
        capability=capability,
        verifier_id="verifier:shadow@1",
        proof_bytes=b"shadow-proof",
        verifying_key_material=b"vk-shadow",
    )
    assert shadow_receipt.verdict is ProgramZkpVerdict.VERIFIED
    assert shadow_receipt.authoritative is False
    assert shadow_receipt.capability_production_eligible is False


def test_cryptographic_without_capability_binding_is_non_authoritative() -> None:
    envelope = _crypto_envelope()
    receipt = record_program_zkp_verification(
        envelope,
        verdict=ProgramZkpVerdict.VERIFIED,
        verifier_id="verifier:x@1",
        independent_verifier=True,
        capability_production_eligible=False,
        capability_epoch="",
    )
    assert receipt.authoritative is False


# ---------------------------------------------------------------------------
# Cancellation and bounds
# ---------------------------------------------------------------------------


def test_cancellation_fails_closed() -> None:
    cancel = threading.Event()
    cancel.set()
    report = probe_program_analysis_zkp_capability(cancellation_event=cancel)
    assert report.production_eligible is False
    assert ProgramZkpAuthorityDenialReason.CANCELLED.value in report.denial_reasons
    cancellation = report.checks_by_dimension[ProgramZkpCapabilityDimension.CANCELLATION]
    assert cancellation.production_eligible is False

    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    with pytest.raises(ProgramZkpCapabilityError, match="cancelled"):
        verify_program_zkp_independently(
            envelope,
            capability=capability,
            verifier_id="verifier:independent@1",
            proof_bytes=b"ok-proof",
            verifying_key_material=b"vk-fixture-material-v1",
            cryptographic_verify=_honest_crypto_verify,
            cancellation_event=cancel,
        )


def test_bounds_probe_rejects_overflow() -> None:
    report = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        max_trace_steps=2,
        observed_trace_steps=99,
        circuit_id=PROGRAM_CONTRACT_TRACE_CIRCUIT_ID,
        proving_key_id="pk:x@1",
        verifying_key_id="vk:x@1",
        ceremony_id="ceremony:x@1",
        ceremony_production_eligible=True,
        independent_verifier_available=True,
    )
    bounds = report.checks_by_dimension[ProgramZkpCapabilityDimension.BOUNDS]
    assert bounds.production_eligible is False
    assert (
        ProgramZkpAuthorityDenialReason.BOUNDS_EXCEEDED.value in bounds.denial_reasons
    )


# ---------------------------------------------------------------------------
# No semantic claim promotion
# ---------------------------------------------------------------------------


def test_authoritative_receipt_does_not_promote_semantic_claims() -> None:
    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    receipt = verify_program_zkp_independently(
        envelope,
        capability=capability,
        verifier_id="verifier:independent@1",
        proof_bytes=b"ok-proof",
        verifying_key_material=b"vk-fixture-material-v1",
        cryptographic_verify=_honest_crypto_verify,
    )
    assert receipt.authoritative is True
    assert receipt.claim_level is ClaimLevel.ZK_TRACE_ATTESTED
    payload = receipt.to_public_artifact()
    assert payload["semantic_proof"] is False
    assert payload["claim_level"] == ClaimLevel.ZK_TRACE_ATTESTED.value
    for target in (
        ClaimLevel.MODEL_PROVED,
        ClaimLevel.RUNTIME_WITNESSED,
        ClaimLevel.OBSERVED_SYNTAX,
    ):
        with pytest.raises(ProgramZkpClaimPromotionError):
            reject_illegal_zk_claim_promotion(ClaimLevel.ZK_TRACE_ATTESTED, target)


def test_forged_authoritative_flag_on_shadow_receipt_is_rejected() -> None:
    envelope = _shadow_envelope()
    receipt = record_program_zkp_verification(
        envelope,
        verdict=ProgramZkpVerdict.VERIFIED,
        verifier_id="verifier:shadow@1",
    )
    forged = receipt.to_public_artifact()
    forged["authoritative"] = True
    with pytest.raises(ProgramZkpTamperError, match="cannot assert authority"):
        type(receipt).from_dict(forged)


# ---------------------------------------------------------------------------
# VFS-G081: capability conformance evidence + proof-system non-substitution
# ---------------------------------------------------------------------------


def test_vfs_g081_capability_conformance_evidence_discoverable() -> None:
    """Cover vfs/zk-capability-conformance@1 for objective goal VFS-G081.

    Exact-text discovery anchors keep the supervisor backlog aligned with the
    objective heap.  Capability reports and prove-claims publish the domain
    evidence term; simulated or placeholder paths never acquire authority.
    """

    assert PROGRAM_ZKP_EVIDENCE_CAPABILITY_CONFORMANCE == (
        "vfs/zk-capability-conformance@1"
    )
    assert PROGRAM_ZKP_G081_EVIDENCE_TERMS == ("vfs/zk-capability-conformance@1",)
    assert capability_conformance_evidence_terms() == (
        "vfs/zk-capability-conformance@1",
    )
    assert OBJECTIVE_GOAL_G081_ID == "VFS-G081"
    assert OBJECTIVE_TASK_G081_ID == "VFS-076"
    assert "vfs/zk-capability-conformance@1" in all_program_zkp_evidence_terms()

    shadow = probe_program_analysis_zkp_capability()
    production = build_production_ready_capability_fixture()
    for report in (shadow, production):
        assert report.to_dict()["evidence"] == "vfs/zk-capability-conformance@1"
        claim = prove_zk_capability_conformance(report)
        assert claim["schema"] == PROGRAM_ZKP_CAPABILITY_CONFORMANCE_CLAIM_SCHEMA
        assert claim["evidence"] == "vfs/zk-capability-conformance@1"
        assert claim["requirement_id"] == "vfs/zk-capability-conformance@1"
        assert claim["goal_id"] == "VFS-G081"
        assert claim["task_id"] == "VFS-076"
        assert claim["authoritative"] is False
        assert claim["completion_authoritative"] is False
        assert claim["satisfied"] is True
        assert claim["acceptance_dimensions"]["simulated_backends_blocked"] is True
        assert claim["acceptance_dimensions"]["placeholder_encodings_blocked"] is True
        assert claim["acceptance_dimensions"]["no_proof_system_substitution"] is True
        # Round-trip via public artifact retains the evidence identity.
        public = report.to_public_artifact()
        assert public["evidence"] == PROGRAM_ZKP_EVIDENCE_CAPABILITY_CONFORMANCE
        restored = ProgramZkpCapabilityConformanceReport.from_dict(public)
        assert restored.capability_epoch == report.capability_epoch

    assert shadow.production_eligible is False
    assert production.production_eligible is True
    shadow_claim = prove_zk_capability_conformance(shadow)
    production_claim = prove_zk_capability_conformance(production)
    assert shadow_claim["shadow_only"] is True
    assert production_claim["production_eligible"] is True
    assert production_claim["rollout_mode"] == ProgramZkpRolloutMode.ENFORCEMENT.value


def test_proof_system_substitution_fails_closed() -> None:
    """Do not silently substitute one proof system for another (VFS-G081).

    A production-eligible capability for program_contract_trace must not
    authorize an envelope pinned to a different circuit family/id, and a
    capability that admits an incompatible family cannot mint authority.
    """

    capability = build_production_ready_capability_fixture()
    # Envelope claims a TDFOL circuit while the capability admits program_contract_trace.
    foreign = _crypto_envelope(circuit_id="circuit:tdfol-theorem@1")
    assert foreign.statement.public_inputs.circuit_id == "circuit:tdfol-theorem@1"
    with pytest.raises(ProgramZkpAuthorityError, match="proof system substitution"):
        verify_program_zkp_independently(
            foreign,
            capability=capability,
            verifier_id="verifier:independent@1",
            proof_bytes=b"ok-proof",
            verifying_key_material=b"vk-fixture-material-v1",
            cryptographic_verify=_honest_crypto_verify,
        )

    # Capability itself for an incompatible family is never production-eligible.
    tdfol_cap = probe_program_analysis_zkp_capability(
        backend_mode=ProgramZkpBackendMode.CRYPTOGRAPHIC,
        circuit_id="circuit:tdfol-theorem@1",
        circuit_family=ProgramZkpCircuitFamily.TDFOL_ONLY,
        proving_key_id="pk:tdfol@1",
        verifying_key_id="vk:tdfol@1",
        ceremony_id="ceremony:tdfol@1",
        ceremony_production_eligible=True,
        independent_verifier_available=True,
    )
    assert tdfol_cap.production_eligible is False
    assert (
        ProgramZkpAuthorityDenialReason.INCOMPATIBLE_TDFOL_ONLY_CIRCUIT.value
        in tdfol_cap.denial_reasons
    )
    # Even when the envelope matches the (ineligible) capability circuit, no
    # authoritative receipt may be minted — and verification fails closed on
    # family substitution against the only production-admitted system.
    matching_tdfol = _crypto_envelope(circuit_id="circuit:tdfol-theorem@1")
    with pytest.raises(ProgramZkpAuthorityError, match="proof system substitution"):
        verify_program_zkp_independently(
            matching_tdfol,
            capability=tdfol_cap,
            verifier_id="verifier:independent@1",
            proof_bytes=b"ok-proof",
            verifying_key_material=b"vk-fixture-material-v1",
            cryptographic_verify=_honest_crypto_verify,
        )

    # Honest production path still works under exact circuit binding.
    honest = _crypto_envelope()
    receipt = verify_program_zkp_independently(
        honest,
        capability=capability,
        verifier_id="verifier:independent@1",
        proof_bytes=b"ok-proof",
        verifying_key_material=b"vk-fixture-material-v1",
        cryptographic_verify=_honest_crypto_verify,
    )
    assert receipt.authoritative is True
    assert receipt.circuit_id == PROGRAM_CONTRACT_TRACE_CIRCUIT_ID
    assert grants_production_authority(receipt, capability) is True


def test_simulated_proof_under_cryptographic_path_fails_closed() -> None:
    """Simulated SIMZKP layouts never acquire authority on a crypto path."""

    capability = build_production_ready_capability_fixture()
    envelope = _crypto_envelope()
    sim_proof = b"SIMZKP\x00\x01" + (b"\x00" * 152)
    assert proof_bytes_are_simulated(sim_proof) is True
    with pytest.raises(ProgramZkpAuthorityError, match="simulated proof"):
        verify_program_zkp_independently(
            envelope,
            capability=capability,
            verifier_id="verifier:independent@1",
            proof_bytes=sim_proof,
            verifying_key_material=b"vk-fixture-material-v1",
            cryptographic_verify=lambda p, k, f: True,
        )
    with pytest.raises(ProgramZkpAuthorityError):
        record_production_program_zkp_verification(
            envelope,
            capability=capability,
            verifier_id="verifier:independent@1",
            proof_bytes=sim_proof,
            verifying_key_material=b"vk-fixture-material-v1",
            cryptographic_verify=lambda p, k, f: True,
        )


def test_forged_capability_conformance_identity_is_rejected() -> None:
    """Adversarial forgery of capability report content_id fails closed."""

    report = build_production_ready_capability_fixture()
    payload = report.to_public_artifact()
    assert payload["evidence"] == "vfs/zk-capability-conformance@1"
    payload["content_id"] = "baguqeer-forged-capability-epoch"
    with pytest.raises(ProgramZkpTamperError, match="forged capability"):
        ProgramZkpCapabilityConformanceReport.from_dict(payload)
    # Flipping production_eligible without recomputing identity also fails.
    payload = report.to_public_artifact()
    payload["production_eligible"] = not report.production_eligible
    restored = ProgramZkpCapabilityConformanceReport.from_dict(
        {k: v for k, v in payload.items() if k not in {"content_id", "capability_epoch"}}
    )
    # Production eligibility is derived, not caller-asserted.
    assert restored.production_eligible == report.production_eligible
    claim = prove_zk_capability_conformance(restored)
    assert claim["evidence"] == "vfs/zk-capability-conformance@1"
    assert claim["satisfied"] is True
