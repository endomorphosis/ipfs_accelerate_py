"""SCA-621 / SCA-G180 ProveKit setup tests for SCAEV180PROOFREADY.

Proves typed unavailable status without CLI/artifacts, content-addressed
executable/setup/circuit/verifier identities, mandatory positive/negative
self-tests, simulated non-attestation, and attestation only for an approved
kernel-verified receipt predicate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.provekit_setup import (
    APPROVED_VERIFIED_RECEIPT_PREDICATE,
    REQUIRED_PROVEKIT_SELF_TESTS,
    SCAEV180PROOFREADY,
    SCAEV180PROOFREADY_COVERAGE,
    SCAEV180PROOFREADY_EVIDENCE,
    ProveKitSelfTestCase,
    ProveKitSetupConfig,
    ProveKitSetupStatus,
    build_provekit_setup_report,
    evaluate_provekit_attestation_eligibility,
    probe_provekit_setup,
)


def _artifact_dir(tmp_path: Path) -> Path:
    root = tmp_path / "provekit-artifacts"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"circuit": "receipt-binding", "version": "1.0.0"}),
        encoding="utf-8",
    )
    (root / "circuit.pkp").write_bytes(b"prover-key-bytes")
    (root / "circuit.pkv").write_bytes(b"verifier-key-bytes")
    return root


def _executable(tmp_path: Path, name: str = "provekit-cli") -> Path:
    path = tmp_path / name
    path.write_bytes(b"#!/bin/sh\n# fake provekit\n")
    path.chmod(0o755)
    return path


def _passing_self_tests():
    return {case: (lambda _case: True) for case in REQUIRED_PROVEKIT_SELF_TESTS}


def test_scaev180proofready_shared_with_solver_readiness() -> None:
    assert SCAEV180PROOFREADY == "SCAEV180PROOFREADY"
    assert SCAEV180PROOFREADY_EVIDENCE == SCAEV180PROOFREADY
    assert "provekit-real-zk-gated-by-setup-identities-and-self-tests" in (
        SCAEV180PROOFREADY_COVERAGE
    )


def test_absent_cli_or_artifacts_emits_typed_unavailable() -> None:
    receipt = probe_provekit_setup(
        ProveKitSetupConfig(),
        which=lambda _name: None,
        environ={},
    )
    payload = receipt.to_dict()

    assert receipt.status is ProveKitSetupStatus.UNAVAILABLE
    assert receipt.configured is False
    assert receipt.available is False
    assert receipt.production_eligible is False
    assert receipt.reason_code == "provekit_unconfigured"
    assert receipt.proof_success is False
    assert SCAEV180PROOFREADY in payload["evidence"]["requirement_ids"]
    assert payload["evidence"]["coverage"] == list(SCAEV180PROOFREADY_COVERAGE)


def test_identities_are_content_addressed_when_configured(tmp_path: Path) -> None:
    exe = _executable(tmp_path)
    artifacts = _artifact_dir(tmp_path)
    receipt = probe_provekit_setup(
        ProveKitSetupConfig(
            executable_path=str(exe),
            artifacts_path=str(artifacts),
            backend_version="0.1.7",
        ),
        which=lambda _name: None,
        environ={},
    )

    assert receipt.configured is True
    assert receipt.available is True
    assert receipt.executable.present is True
    assert receipt.executable.digest.startswith("provekit-executable:sha256:")
    assert receipt.setup.present is True
    assert receipt.setup.digest.startswith("provekit-setup:sha256:")
    assert receipt.circuit.present is True
    assert receipt.prover.present is True
    assert receipt.verifier.present is True
    assert receipt.prover.digest.startswith("provekit-prover:sha256:")
    assert receipt.verifier.digest.startswith("provekit-verifier:sha256:")
    assert receipt.policy_material["backend_id"] == "backend:provekit"
    assert receipt.policy_material["executable_digest"] == receipt.executable.digest
    # Without self-tests, available but not production eligible.
    assert receipt.status is ProveKitSetupStatus.AVAILABLE
    assert receipt.production_eligible is False
    assert receipt.reason_code == "provekit_self_tests_required"


def test_mandatory_self_tests_gate_production_eligibility(tmp_path: Path) -> None:
    exe = _executable(tmp_path)
    artifacts = _artifact_dir(tmp_path)
    config = ProveKitSetupConfig(
        executable_path=str(exe),
        artifacts_path=str(artifacts),
        backend_version="0.1.7",
    )

    failed = probe_provekit_setup(
        config,
        which=lambda _name: None,
        environ={},
        self_tests={
            ProveKitSelfTestCase.POSITIVE: lambda _c: True,
            ProveKitSelfTestCase.NEGATIVE: lambda _c: False,
            ProveKitSelfTestCase.MALFORMED_PROOF: lambda _c: True,
            ProveKitSelfTestCase.WITNESS_NO_LEAK: lambda _c: True,
        },
    )
    assert failed.status is ProveKitSetupStatus.DEGRADED
    assert failed.production_eligible is False
    assert failed.reason_code == "provekit_self_test_failed"
    assert any(
        item.case is ProveKitSelfTestCase.NEGATIVE and not item.passed
        for item in failed.self_tests
    )

    verified = probe_provekit_setup(
        config,
        which=lambda _name: None,
        environ={},
        self_tests=_passing_self_tests(),
    )
    assert verified.status is ProveKitSetupStatus.VERIFIED
    assert verified.production_eligible is True
    assert verified.reason_code == "provekit_production_eligible"
    assert all(item.passed for item in verified.self_tests)
    assert set(item.case for item in verified.self_tests) == set(
        REQUIRED_PROVEKIT_SELF_TESTS
    )


def test_incomplete_surfaces_stay_non_production(tmp_path: Path) -> None:
    exe = _executable(tmp_path)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text("{}", encoding="utf-8")
    # Missing prover/verifier keys.
    receipt = probe_provekit_setup(
        ProveKitSetupConfig(
            executable_path=str(exe),
            artifacts_path=str(incomplete),
        ),
        which=lambda _name: None,
        environ={},
        self_tests=_passing_self_tests(),
    )
    assert receipt.production_eligible is False
    assert receipt.available is False
    assert "verifier" in receipt.reason or "prover" in receipt.reason


def test_simulated_provekit_is_never_attested_or_production_eligible() -> None:
    receipt = probe_provekit_setup(
        ProveKitSetupConfig(simulated=True),
        which=lambda _name: None,
        environ={},
        self_tests=_passing_self_tests(),
    )
    assert receipt.status is ProveKitSetupStatus.SIMULATED
    assert receipt.production_eligible is False
    assert receipt.simulated is True

    eligibility = evaluate_provekit_attestation_eligibility(
        receipt,
        predicate_id=APPROVED_VERIFIED_RECEIPT_PREDICATE,
        kernel_verified=True,
        kernel_receipt_id="kernel-receipt:1",
    )
    assert eligibility.eligible is False
    assert eligibility.attested is False
    assert eligibility.reason_code == "simulated_non_attested"


def test_attestation_requires_approved_kernel_verified_receipt_predicate(
    tmp_path: Path,
) -> None:
    exe = _executable(tmp_path)
    artifacts = _artifact_dir(tmp_path)
    setup = probe_provekit_setup(
        ProveKitSetupConfig(
            executable_path=str(exe),
            artifacts_path=str(artifacts),
            backend_version="0.1.7",
        ),
        which=lambda _name: None,
        environ={},
        self_tests=_passing_self_tests(),
    )
    assert setup.production_eligible is True

    unapproved = evaluate_provekit_attestation_eligibility(
        setup,
        predicate_id="arbitrary-function-correctness",
        kernel_verified=True,
        kernel_receipt_id="kernel-receipt:1",
    )
    assert unapproved.eligible is False
    assert unapproved.reason_code == "predicate_not_approved"

    not_kernel = evaluate_provekit_attestation_eligibility(
        setup,
        predicate_id=APPROVED_VERIFIED_RECEIPT_PREDICATE,
        kernel_verified=False,
        kernel_receipt_id="kernel-receipt:1",
    )
    assert not_kernel.eligible is False
    assert not_kernel.reason_code == "kernel_verification_required"

    missing_receipt = evaluate_provekit_attestation_eligibility(
        setup,
        predicate_id=APPROVED_VERIFIED_RECEIPT_PREDICATE,
        kernel_verified=True,
        kernel_receipt_id="",
    )
    assert missing_receipt.eligible is False
    assert missing_receipt.reason_code == "kernel_receipt_missing"

    eligible = evaluate_provekit_attestation_eligibility(
        setup,
        predicate_id=APPROVED_VERIFIED_RECEIPT_PREDICATE,
        kernel_verified=True,
        kernel_receipt_id="kernel-receipt:sha256:deadbeef",
    )
    payload = eligible.to_dict()
    assert eligible.eligible is True
    assert eligible.attested is False
    assert eligible.reason_code == "eligible_verified_receipt_predicate"
    assert SCAEV180PROOFREADY in payload["evidence"]["requirement_ids"]
    assert payload["approved_predicate"] == APPROVED_VERIFIED_RECEIPT_PREDICATE


def test_env_overrides_discover_executable_and_artifacts(tmp_path: Path) -> None:
    exe = _executable(tmp_path, name="custom-pk")
    artifacts = _artifact_dir(tmp_path)
    receipt = probe_provekit_setup(
        ProveKitSetupConfig(),
        which=lambda _name: None,
        environ={
            "PROVEKIT_CLI": str(exe),
            "PROVEKIT_ARTIFACTS_DIR": str(artifacts),
        },
        self_tests=_passing_self_tests(),
    )
    assert receipt.production_eligible is True
    assert receipt.executable.path == str(exe)


def test_which_discovery_for_provekit_cli(tmp_path: Path) -> None:
    exe = _executable(tmp_path)
    artifacts = _artifact_dir(tmp_path)
    receipt = probe_provekit_setup(
        ProveKitSetupConfig(artifacts_path=str(artifacts)),
        which=lambda name: str(exe) if name == "provekit-cli" else None,
        environ={},
        self_tests=_passing_self_tests(),
    )
    assert receipt.executable.present is True
    assert receipt.production_eligible is True


def test_setup_report_envelope_and_json_stability(tmp_path: Path) -> None:
    exe = _executable(tmp_path)
    artifacts = _artifact_dir(tmp_path)
    receipt = probe_provekit_setup(
        ProveKitSetupConfig(
            executable_path=str(exe),
            artifacts_path=str(artifacts),
        ),
        which=lambda _name: None,
        environ={},
        self_tests=_passing_self_tests(),
    )
    report = build_provekit_setup_report(receipt)
    assert report["production_eligible"] is True
    assert report["setup_identity"] == receipt.setup_identity
    assert SCAEV180PROOFREADY in report["evidence"]["requirement_ids"]
    assert json.loads(json.dumps(report)) == report


def test_self_test_exception_is_fail_closed(tmp_path: Path) -> None:
    exe = _executable(tmp_path)
    artifacts = _artifact_dir(tmp_path)

    def boom(_case: ProveKitSelfTestCase) -> bool:
        raise RuntimeError("canary exploded")

    receipt = probe_provekit_setup(
        ProveKitSetupConfig(
            executable_path=str(exe),
            artifacts_path=str(artifacts),
        ),
        which=lambda _name: None,
        environ={},
        self_tests={case: boom for case in REQUIRED_PROVEKIT_SELF_TESTS},
    )
    assert receipt.production_eligible is False
    assert receipt.status is ProveKitSetupStatus.DEGRADED
    assert all(not item.passed for item in receipt.self_tests)


def test_non_eligible_setup_blocks_attestation(tmp_path: Path) -> None:
    exe = _executable(tmp_path)
    artifacts = _artifact_dir(tmp_path)
    setup = probe_provekit_setup(
        ProveKitSetupConfig(
            executable_path=str(exe),
            artifacts_path=str(artifacts),
        ),
        which=lambda _name: None,
        environ={},
        # self-tests omitted -> not production eligible
    )
    eligibility = evaluate_provekit_attestation_eligibility(
        setup,
        predicate_id=APPROVED_VERIFIED_RECEIPT_PREDICATE,
        kernel_verified=True,
        kernel_receipt_id="kernel-receipt:1",
    )
    assert eligibility.eligible is False
    assert eligibility.reason_code == "setup_not_production_eligible"
