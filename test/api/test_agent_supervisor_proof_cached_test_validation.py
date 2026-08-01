from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_test_certificate_provider import (
    TestCertificateVerificationResult,
    TestCertificateVerificationStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    ProofBackendMode,
    ReuseAction,
    ReuseDecision,
    ReuseReasonCode,
    TestExecutionKey,
    TestPassReceipt,
    TestProofCertificate,
)
from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    AuthorityMode,
    RepositoryAuthority,
    RepositoryDescriptor,
    build_repository_descriptor,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_cached_test_validation import (
    ProofCachedTestValidation,
    ProofCachedTestValidationError,
    ProofCachedTestValidationReason,
    ProofCachedTestValidationReceipt,
    ProofCachedTestValidationResult,
    ValidationEvidence,
    validation_command_identity,
)

COMMAND = "python3 -m pytest test_sample.py -q"
POLICY_CID = "baguqeera-policy-current"
FOREST_CID = "baguqeera-repository-forest"
EPOCH = "epoch-2026-08"
NOW = 1_786_000_000.0


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "proof-validation@example.invalid")
    _git(tmp_path, "config", "user.name", "Proof Validation")
    (tmp_path / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n", encoding="utf-8"
    )
    _git(tmp_path, "add", "test_sample.py")
    _git(tmp_path, "commit", "-qm", "test state")
    return tmp_path


def _descriptor(root: Path) -> RepositoryDescriptor:
    return build_repository_descriptor(
        root,
        alias="validation-root",
        logical_name="validation-root",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )


@dataclass(frozen=True)
class Artifacts:
    key: TestExecutionKey
    receipt: TestPassReceipt
    certificate: TestProofCertificate
    decision: ReuseDecision


def _artifacts(
    descriptor: RepositoryDescriptor,
    *,
    command: str = COMMAND,
    epoch: str = EPOCH,
    policy_cid: str = POLICY_CID,
    backend_mode: ProofBackendMode = ProofBackendMode.CRYPTOGRAPHIC,
    authority: CertificateAuthority = CertificateAuthority.AUTHORITATIVE,
) -> Artifacts:
    key = TestExecutionKey(
        locator_cid="baguqeera-test-locator",
        repository_forest_cid=FOREST_CID,
        git_commit_id=descriptor.commit,
        git_tree_id=descriptor.tree,
        gitlink_state_cid=descriptor.portable_closure.gitlink_closure_cid,
        dirty_overlay_cid=descriptor.dirty_overlay_digest,
        command_semantics_cid=validation_command_identity(command),
        policy_cid=policy_cid,
    )
    receipt = TestPassReceipt(
        execution_key_cid=key.execution_key_id,
        locator_cid=key.locator_cid,
        dependency_forest_cid=key.repository_forest_cid,
        policy_cid=policy_cid,
        runner_identity="pytest",
        trust_domain="ci",
        issuer_key_id="key-2026-08",
    )
    certificate = TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=key.execution_key_id,
        statement_cid="baguqeera-proof-statement",
        circuit_cid="baguqeera-circuit",
        verifying_key_cid="baguqeera-verifying-key",
        proof_system_id="groth16",
        proof_artifact_cid="baguqeera-proof-artifact",
        backend_mode=backend_mode,
        authority=authority,
        issuer_id="proof-issuer",
        policy_cid=policy_cid,
        epoch=epoch,
    )
    decision = ReuseDecision(
        action=ReuseAction.SKIP,
        reason_code=ReuseReasonCode.PROOF_CACHE_HIT,
        certificate_cid=certificate.certificate_id,
        receipt_cid=receipt.receipt_id,
        authority=CertificateAuthority.AUTHORITATIVE,
    )
    return Artifacts(key, receipt, certificate, decision)


class RecordingVerifier:
    verifier_id = "recording-verifier@1"

    def __init__(
        self,
        *,
        status: TestCertificateVerificationStatus = (TestCertificateVerificationStatus.VERIFIED),
    ) -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def verify_retained_bytes(
        self,
        certificate_bytes: bytes,
        receipt_bytes: bytes,
        requirements: dict[str, Any],
        **kwargs: Any,
    ) -> TestCertificateVerificationResult:
        certificate = TestProofCertificate.from_json(certificate_bytes.decode("utf-8"))
        receipt = TestPassReceipt.from_json(receipt_bytes.decode("utf-8"))
        self.calls.append(
            {
                "certificate_bytes": certificate_bytes,
                "receipt_bytes": receipt_bytes,
                "requirements": requirements,
                "kwargs": kwargs,
            }
        )
        if self.status is TestCertificateVerificationStatus.VERIFIED:
            return TestCertificateVerificationResult(
                status=self.status,
                reason_code=ReuseReasonCode.PROOF_CACHE_HIT,
                authority=CertificateAuthority.AUTHORITATIVE,
                certificate_cid=certificate.certificate_id,
                receipt_cid=receipt.receipt_id,
                backend_id=self.verifier_id,
            )
        return TestCertificateVerificationResult(
            status=self.status,
            reason_code=(
                ReuseReasonCode.VERIFIER_UNAVAILABLE
                if self.status is TestCertificateVerificationStatus.UNAVAILABLE
                else ReuseReasonCode.TRUST_POLICY_REJECTED
            ),
            authority=CertificateAuthority.NON_ATTESTED,
            certificate_cid=certificate.certificate_id,
            receipt_cid=receipt.receipt_id,
            backend_id=self.verifier_id,
        )


def _validator(
    repository: Path,
    verifier: RecordingVerifier,
    **kwargs: Any,
) -> ProofCachedTestValidation:
    return ProofCachedTestValidation(
        verifier=verifier,
        verifier_id=verifier.verifier_id,
        repository_root=repository,
        freshness_seconds=30,
        clock=lambda: NOW,
        **kwargs,
    )


def _validate(
    validator: ProofCachedTestValidation,
    artifacts: Artifacts,
    **overrides: Any,
) -> ProofCachedTestValidationReceipt:
    values: dict[str, Any] = {
        "task_id": "PTR-060",
        "goal_id": "PTR-060",
        "goal_revision": "baguqeera-objective-revision",
        "validation_command": COMMAND,
        "decision": artifacts.decision,
        "execution_key": artifacts.key,
        "pass_receipt": artifacts.receipt,
        "certificate": artifacts.certificate,
        "policy_cid": POLICY_CID,
        "current_epoch": EPOCH,
        "proof_bytes": b"exact-proof-bytes",
    }
    values.update(overrides)
    return validator.validate(**values)


def test_authoritative_receipt_binds_current_state_and_exact_artifacts(
    repository: Path,
) -> None:
    descriptor = _descriptor(repository)
    artifacts = _artifacts(descriptor)
    verifier = RecordingVerifier()

    receipt = _validate(_validator(repository, verifier), artifacts)

    assert isinstance(receipt, ValidationEvidence)
    assert receipt.verifier_result is ProofCachedTestValidationResult.VERIFIED
    assert receipt.reason_codes == ("proof_reverified",)
    assert receipt.passed is True
    assert receipt.is_completion_evidence(
        now_ms=int(NOW * 1_000),
        task_id="PTR-060",
        goal_id="PTR-060",
        goal_revision="baguqeera-objective-revision",
        validation_command=COMMAND,
        repository_state_cid=descriptor.descriptor_cid,
    )
    assert receipt.validation_command_cid == validation_command_identity(COMMAND)
    assert receipt.repository_id == descriptor.repository_id
    assert receipt.repository_state_cid == descriptor.descriptor_cid
    assert receipt.repository_forest_cid == FOREST_CID
    assert receipt.git_commit_id == descriptor.commit
    assert receipt.git_tree_id == descriptor.tree
    assert receipt.gitlink_state_cid == descriptor.portable_closure.gitlink_closure_cid
    assert receipt.gitlink_closure_complete is True
    assert receipt.dirty == descriptor.dirty
    assert receipt.dirty_overlay_cid == descriptor.dirty_overlay_digest
    assert receipt.decision_cid == artifacts.decision.decision_id
    assert receipt.execution_key_cid == artifacts.key.execution_key_id
    assert receipt.test_receipt_cid == artifacts.receipt.receipt_id
    assert receipt.certificate_cid == artifacts.certificate.certificate_id
    assert receipt.policy_cid == POLICY_CID
    assert receipt.statement_cid == artifacts.certificate.statement_cid
    assert receipt.circuit_cid == artifacts.certificate.circuit_cid
    assert receipt.verifying_key_cid == artifacts.certificate.verifying_key_cid
    assert receipt.proof_system_id == artifacts.certificate.proof_system_id
    assert receipt.certificate_epoch == EPOCH
    assert receipt.certificate_authority is CertificateAuthority.AUTHORITATIVE
    assert receipt.verifier_authority is CertificateAuthority.AUTHORITATIVE
    assert receipt.fresh_until_ms - receipt.verified_at_ms == 30_000

    call = verifier.calls[0]
    assert call["certificate_bytes"] == artifacts.certificate.canonical_bytes()
    assert call["receipt_bytes"] == artifacts.receipt.canonical_bytes()
    assert call["kwargs"] == {"proof_bytes": b"exact-proof-bytes"}
    assert call["requirements"] == {
        "task_id": "PTR-060",
        "goal_id": "PTR-060",
        "goal_revision": "baguqeera-objective-revision",
        "validation_command_cid": validation_command_identity(COMMAND),
        "execution_key_cid": artifacts.key.execution_key_id,
        "receipt_cid": artifacts.receipt.receipt_id,
        "certificate_cid": artifacts.certificate.certificate_id,
        "repository_id": descriptor.repository_id,
        "repository_state_cid": descriptor.descriptor_cid,
        "repository_forest_cid": FOREST_CID,
        "git_commit_id": descriptor.commit,
        "git_tree_id": descriptor.tree,
        "gitlink_state_cid": descriptor.portable_closure.gitlink_closure_cid,
        "dirty": descriptor.dirty,
        "dirty_overlay_cid": descriptor.dirty_overlay_digest,
        "policy_cid": POLICY_CID,
        "statement_cid": artifacts.certificate.statement_cid,
        "circuit_cid": artifacts.certificate.circuit_cid,
        "verifying_key_cid": artifacts.certificate.verifying_key_cid,
        "proof_system_id": artifacts.certificate.proof_system_id,
        "allowed_epochs": [EPOCH],
    }


def test_receipt_round_trip_is_content_addressed_and_projects_completion(
    repository: Path,
) -> None:
    descriptor = _descriptor(repository)
    artifacts = _artifacts(descriptor)
    receipt = _validate(_validator(repository, RecordingVerifier()), artifacts)

    decoded = ProofCachedTestValidationReceipt.from_dict(receipt.to_record())
    assert decoded == receipt
    assert decoded.validation_receipt_cid == receipt.validation_receipt_cid
    completion = receipt.to_completion_evidence(
        acceptance_criterion="proof-backed test validation passes",
        now_ms=int(NOW * 1_000),
    )
    assert completion.validation_passed is True
    assert completion.validation_receipt["content_id"] == receipt.content_id
    assert completion.provenance_cid == receipt.validation_receipt_cid

    tampered = receipt.to_record()
    tampered["task_id"] = "PTR-tampered"
    with pytest.raises(ProofCachedTestValidationError, match="content identity"):
        ProofCachedTestValidationReceipt.from_dict(tampered)


def test_stale_receipt_is_not_completion_evidence(repository: Path) -> None:
    artifacts = _artifacts(_descriptor(repository))
    receipt = _validate(_validator(repository, RecordingVerifier()), artifacts)

    stale_at = receipt.fresh_until_ms + 1
    assert receipt.passed is True
    assert receipt.is_fresh(now_ms=stale_at) is False
    assert receipt.is_completion_evidence(now_ms=stale_at) is False
    assert (
        receipt.to_completion_evidence(
            acceptance_criterion="tests pass", now_ms=stale_at
        ).validation_passed
        is False
    )


def test_current_repository_change_rejects_historical_certificate(
    repository: Path,
) -> None:
    artifacts = _artifacts(_descriptor(repository))
    (repository / "new-untracked-file.txt").write_text(
        "changes the dirty overlay\n", encoding="utf-8"
    )
    verifier = RecordingVerifier()

    receipt = _validate(_validator(repository, verifier), artifacts)

    assert receipt.passed is False
    assert receipt.reason_codes == ("repository_state_mismatch",)
    assert receipt.dirty is True
    assert verifier.calls == []


def test_incomplete_recursive_gitlink_observation_rejects_skip(
    repository: Path,
) -> None:
    descriptor = _descriptor(repository)
    artifacts = _artifacts(descriptor)
    incomplete = replace(
        descriptor,
        portable_closure=replace(descriptor.portable_closure, gitlink_closure_complete=False),
    )
    verifier = RecordingVerifier()
    validator = _validator(
        repository,
        verifier,
        repository_observer=lambda: incomplete,
    )

    receipt = _validate(validator, artifacts)

    assert receipt.reason_codes == ("recursive_gitlinks_incomplete",)
    assert receipt.gitlink_closure_complete is False
    assert receipt.passed is False
    assert verifier.calls == []


def test_simulated_certificate_is_never_completion_authority(
    repository: Path,
) -> None:
    descriptor = _descriptor(repository)
    artifacts = _artifacts(
        descriptor,
        backend_mode=ProofBackendMode.SIMULATED,
        authority=CertificateAuthority.NON_ATTESTED,
    )
    verifier = RecordingVerifier()

    receipt = _validate(_validator(repository, verifier), artifacts)

    assert receipt.reason_codes == ("certificate_non_attested",)
    assert receipt.certificate_authority is CertificateAuthority.NON_ATTESTED
    assert receipt.passed is False
    assert verifier.calls == []


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"decision": "SKIPPED: proof cache hit"}, "plain_skip_not_evidence"),
        ({"task_id": ""}, "task_goal_missing"),
        ({"goal_id": ""}, "task_goal_missing"),
        ({"validation_command": ""}, "validation_command_missing"),
        ({"policy_cid": ""}, "policy_missing"),
        ({"current_epoch": ""}, "epoch_missing"),
        (
            {"validation_command": "python3 -m pytest other_test.py -q"},
            "validation_command_mismatch",
        ),
    ],
)
def test_untyped_or_unbound_claims_are_non_completion_evidence(
    repository: Path,
    overrides: dict[str, Any],
    reason: str,
) -> None:
    artifacts = _artifacts(_descriptor(repository))
    verifier = RecordingVerifier()

    receipt = _validate(_validator(repository, verifier), artifacts, **overrides)

    assert receipt.reason_codes == (reason,)
    assert receipt.passed is False
    assert receipt.is_completion_evidence(now_ms=int(NOW * 1_000)) is False
    assert verifier.calls == []


@pytest.mark.parametrize(
    ("status", "result", "reason"),
    [
        (
            TestCertificateVerificationStatus.REJECTED,
            ProofCachedTestValidationResult.REJECTED,
            ProofCachedTestValidationReason.VERIFIER_REJECTED.value,
        ),
        (
            TestCertificateVerificationStatus.UNAVAILABLE,
            ProofCachedTestValidationResult.UNAVAILABLE,
            ProofCachedTestValidationReason.VERIFIER_UNAVAILABLE.value,
        ),
    ],
)
def test_only_authoritative_verifier_success_can_complete(
    repository: Path,
    status: TestCertificateVerificationStatus,
    result: ProofCachedTestValidationResult,
    reason: str,
) -> None:
    artifacts = _artifacts(_descriptor(repository))
    verifier = RecordingVerifier(status=status)

    receipt = _validate(_validator(repository, verifier), artifacts)

    assert receipt.verifier_result is result
    assert receipt.verifier_authority is CertificateAuthority.NON_ATTESTED
    assert receipt.reason_codes[0] == reason
    assert receipt.passed is False
    assert receipt.is_completion_evidence(now_ms=int(NOW * 1_000)) is False
