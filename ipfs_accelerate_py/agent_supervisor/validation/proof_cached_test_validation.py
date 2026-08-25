"""Supervisor authority for proof-backed cached-test validation.

``ProofCachedTestValidation`` is the trust-boundary adapter between a pytest
proof-cache ``SKIP`` decision and agent-supervisor completion evidence.  A
historical certificate or a line of skip text is not validation.  This module
observes the live repository, re-verifies the retained canonical certificate
and pass receipt, and emits a short-lived, content-addressed validation
receipt.

The receipt is deliberately useful on failure: every path returns a typed
artifact with bounded reason codes.  Only a fresh receipt whose
``is_completion_evidence`` check succeeds may satisfy supervisor validation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Final

from ..integrations.ipfs_datasets_test_certificate_provider import (
    TestCertificateVerificationResult,
    TestCertificateVerificationStatus,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)
from ..proof.test_execution_contracts import (
    CertificateAuthority,
    ProofBackendMode,
    ReuseAction,
    ReuseDecision,
    TestExecutionKey,
    TestPassReceipt,
    TestProofCertificate,
)
from ..analysis.repository_forest import (
    AuthorityMode,
    RepositoryAuthority,
    RepositoryDescriptor,
    build_repository_descriptor,
)

PROOF_CACHED_TEST_VALIDATION_VERSION: Final = 1
PROOF_CACHED_TEST_VALIDATION_INTERFACE: Final = "ProofCachedTestValidation@1"
PROOF_CACHED_TEST_VALIDATION_RECEIPT_INTERFACE: Final = "ProofCachedTestValidationReceipt@1"
PROOF_CACHED_TEST_VALIDATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-cached-test-validation-receipt@1"
)
VALIDATION_COMMAND_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/validation-command@1"
)

DEFAULT_RECEIPT_FRESHNESS_SECONDS: Final = 300.0
MAX_RECEIPT_FRESHNESS_SECONDS: Final = 3_600.0
MAX_REASON_CODES: Final = 32
MAX_TEXT_CHARS: Final = 4_096


class ProofCachedTestValidationResult(StrEnum):
    """Closed verifier outcomes recorded by a validation receipt."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    NOT_ATTEMPTED = "not_attempted"


class ProofCachedTestValidationReason(StrEnum):
    """Bounded reasons which cannot be confused with completion authority."""

    PROOF_REVERIFIED = "proof_reverified"
    TASK_GOAL_MISSING = "task_goal_missing"
    VALIDATION_COMMAND_MISSING = "validation_command_missing"
    PLAIN_SKIP_NOT_EVIDENCE = "plain_skip_not_evidence"
    NOT_PROOF_BACKED_SKIP = "not_proof_backed_skip"
    MALFORMED_ARTIFACT = "malformed_artifact"
    ARTIFACT_CID_MISMATCH = "artifact_cid_mismatch"
    EXECUTION_KEY_MISMATCH = "execution_key_mismatch"
    RECEIPT_NOT_PASSING = "receipt_not_passing"
    VALIDATION_COMMAND_MISMATCH = "validation_command_mismatch"
    REPOSITORY_OBSERVATION_FAILED = "repository_observation_failed"
    REPOSITORY_STATE_MISMATCH = "repository_state_mismatch"
    RECURSIVE_GITLINKS_INCOMPLETE = "recursive_gitlinks_incomplete"
    POLICY_MISSING = "policy_missing"
    POLICY_MISMATCH = "policy_mismatch"
    EPOCH_MISSING = "epoch_missing"
    CERTIFICATE_STALE = "certificate_stale"
    CERTIFICATE_NON_ATTESTED = "certificate_non_attested"
    VERIFIER_REJECTED = "verifier_rejected"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    VERIFIER_NON_AUTHORITATIVE = "verifier_non_authoritative"
    INTERNAL_ERROR = "internal_error"


class ProofCachedTestValidationError(ValueError):
    """Raised for programmer misuse of the validator or receipt contract."""


def validation_command_identity(command: str) -> str:
    """Return the canonical identity of the exact declared shell command."""

    if not isinstance(command, str):
        raise TypeError("validation command must be a string")
    normalized = command.strip()
    if not normalized:
        raise ValueError("validation command must not be empty")
    # ``{"command": command}`` is the existing supervisor command-identity
    # projection used by protocol-verification receipts.
    return content_identity({"command": normalized})


def _text(value: Any, *, field_name: str, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ProofCachedTestValidationError(f"{field_name} must be a string")
    normalized = value.strip()
    if required and not normalized:
        raise ProofCachedTestValidationError(f"{field_name} is required")
    if len(normalized) > MAX_TEXT_CHARS:
        raise ProofCachedTestValidationError(f"{field_name} is too long")
    return normalized


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


def _clock_milliseconds(clock: Callable[[], Any]) -> int:
    value = clock()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return int(value.timestamp() * 1_000)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("clock must return seconds since epoch or datetime")
    return int(float(value) * 1_000)


def _bounded_reasons(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    if values is None:
        values = ()
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ProofCachedTestValidationError("reason_codes must be a sequence") from exc
    reasons: list[str] = []
    for raw in raw_values:
        value = _text(_enum_value(raw), field_name="reason_code", required=True)
        if value not in reasons:
            reasons.append(value)
    if len(reasons) > MAX_REASON_CODES:
        raise ProofCachedTestValidationError("too many reason codes")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class ProofCachedTestValidationReceipt(CanonicalContract):
    """Fresh supervisor validation authority over one proof-backed skip."""

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = PROOF_CACHED_TEST_VALIDATION_RECEIPT_SCHEMA

    task_id: str
    goal_id: str
    validation_command: str
    validation_command_cid: str
    repository_id: str
    repository_state_cid: str
    repository_forest_cid: str
    git_commit_id: str
    git_tree_id: str
    gitlink_state_cid: str
    gitlink_closure_complete: bool
    dirty: bool
    dirty_overlay_cid: str
    decision_cid: str
    execution_key_cid: str
    test_receipt_cid: str
    certificate_cid: str
    policy_cid: str
    statement_cid: str
    circuit_cid: str
    verifying_key_cid: str
    proof_system_id: str
    certificate_epoch: str
    certificate_authority: CertificateAuthority | str
    verifier_id: str
    verifier_result: ProofCachedTestValidationResult | str
    verifier_authority: CertificateAuthority | str
    verified_at_ms: int
    fresh_until_ms: int
    goal_revision: str = ""
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        required_text = (
            "task_id",
            "goal_id",
            "validation_command",
            "validation_command_cid",
        )
        optional_text = (
            "repository_id",
            "repository_state_cid",
            "repository_forest_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "dirty_overlay_cid",
            "decision_cid",
            "execution_key_cid",
            "test_receipt_cid",
            "certificate_cid",
            "policy_cid",
            "statement_cid",
            "circuit_cid",
            "verifying_key_cid",
            "proof_system_id",
            "certificate_epoch",
            "verifier_id",
            "goal_revision",
        )
        for name in required_text:
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        for name in optional_text:
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name),
            )
        if not isinstance(self.gitlink_closure_complete, bool):
            raise ProofCachedTestValidationError("gitlink_closure_complete must be boolean")
        if not isinstance(self.dirty, bool):
            raise ProofCachedTestValidationError("dirty must be boolean")

        try:
            certificate_authority = (
                self.certificate_authority
                if isinstance(self.certificate_authority, CertificateAuthority)
                else CertificateAuthority(str(self.certificate_authority))
            )
            verifier_authority = (
                self.verifier_authority
                if isinstance(self.verifier_authority, CertificateAuthority)
                else CertificateAuthority(str(self.verifier_authority))
            )
            verifier_result = (
                self.verifier_result
                if isinstance(self.verifier_result, ProofCachedTestValidationResult)
                else ProofCachedTestValidationResult(str(self.verifier_result))
            )
        except ValueError as exc:
            raise ProofCachedTestValidationError(
                "receipt carries an unsupported authority or verifier result"
            ) from exc
        object.__setattr__(self, "certificate_authority", certificate_authority)
        object.__setattr__(self, "verifier_authority", verifier_authority)
        object.__setattr__(self, "verifier_result", verifier_result)

        for name in ("verified_at_ms", "fresh_until_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProofCachedTestValidationError(f"{name} must be a non-negative integer")
        if self.fresh_until_ms < self.verified_at_ms:
            raise ProofCachedTestValidationError("fresh_until_ms must not precede verified_at_ms")
        object.__setattr__(self, "reason_codes", _bounded_reasons(self.reason_codes))

        if self.validation_command_cid != validation_command_identity(self.validation_command):
            raise ProofCachedTestValidationError(
                "validation_command_cid does not match validation_command"
            )
        if self.verifier_result is ProofCachedTestValidationResult.VERIFIED:
            if (
                self.certificate_authority is not CertificateAuthority.AUTHORITATIVE
                or self.verifier_authority is not CertificateAuthority.AUTHORITATIVE
            ):
                raise ProofCachedTestValidationError(
                    "verified receipts require authoritative certificate and verifier authority"
                )
            if self.reason_codes != (ProofCachedTestValidationReason.PROOF_REVERIFIED.value,):
                raise ProofCachedTestValidationError(
                    "verified receipt requires only proof_reverified"
                )
            for name in optional_text:
                if name == "goal_revision":
                    continue
                if not getattr(self, name):
                    raise ProofCachedTestValidationError(f"verified receipt is missing {name}")
            if not self.gitlink_closure_complete:
                raise ProofCachedTestValidationError(
                    "verified receipt requires a complete recursive Gitlink closure"
                )

    @property
    def interface(self) -> str:
        return PROOF_CACHED_TEST_VALIDATION_RECEIPT_INTERFACE

    @property
    def validation_receipt_cid(self) -> str:
        return self.content_id

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def status(self) -> str:
        return "passed" if self.authoritative else "failed"

    @property
    def passed(self) -> bool:
        """Whether the immutable verifier verdict is authoritative.

        Consumers must additionally call :meth:`is_completion_evidence` to
        enforce the receipt's time and live-context freshness.
        """

        return self.authoritative

    @property
    def authoritative(self) -> bool:
        return (
            self.verifier_result is ProofCachedTestValidationResult.VERIFIED
            and self.certificate_authority is CertificateAuthority.AUTHORITATIVE
            and self.verifier_authority is CertificateAuthority.AUTHORITATIVE
            and self.reason_codes == (ProofCachedTestValidationReason.PROOF_REVERIFIED.value,)
            and self.gitlink_closure_complete
        )

    def is_fresh(self, *, now_ms: int | None = None) -> bool:
        """Return whether the receipt is inside its closed freshness window."""

        current = int(time.time() * 1_000) if now_ms is None else int(now_ms)
        return self.verified_at_ms <= current <= self.fresh_until_ms

    def is_completion_evidence(
        self,
        *,
        now_ms: int | None = None,
        task_id: str = "",
        goal_id: str = "",
        goal_revision: str = "",
        validation_command: str = "",
        repository_state_cid: str = "",
    ) -> bool:
        """Fail closed unless this is the exact fresh authoritative receipt."""

        if not self.authoritative or not self.is_fresh(now_ms=now_ms):
            return False
        comparisons = (
            (task_id, self.task_id),
            (goal_id, self.goal_id),
            (goal_revision, self.goal_revision),
            (repository_state_cid, self.repository_state_cid),
        )
        if any(expected and expected != actual for expected, actual in comparisons):
            return False
        if validation_command:
            try:
                if validation_command_identity(validation_command) != self.validation_command_cid:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_CACHED_TEST_VALIDATION_VERSION,
            "interface": PROOF_CACHED_TEST_VALIDATION_RECEIPT_INTERFACE,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "goal_revision": self.goal_revision,
            "validation_command": self.validation_command,
            "validation_command_cid": self.validation_command_cid,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "git_commit_id": self.git_commit_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "gitlink_closure_complete": self.gitlink_closure_complete,
            "dirty": self.dirty,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "decision_cid": self.decision_cid,
            "execution_key_cid": self.execution_key_cid,
            "test_receipt_cid": self.test_receipt_cid,
            "certificate_cid": self.certificate_cid,
            "policy_cid": self.policy_cid,
            "statement_cid": self.statement_cid,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "proof_system_id": self.proof_system_id,
            "certificate_epoch": self.certificate_epoch,
            "certificate_authority": self.certificate_authority,
            "verifier_id": self.verifier_id,
            "verifier_result": self.verifier_result,
            "verifier_authority": self.verifier_authority,
            "verified_at_ms": self.verified_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "attempted": True,
            "passed": self.passed,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofCachedTestValidationReceipt:
        if not isinstance(payload, Mapping):
            raise ProofCachedTestValidationError("validation receipt must be a mapping")
        allowed = {
            "schema",
            "contract_version",
            "interface",
            "task_id",
            "goal_id",
            "goal_revision",
            "validation_command",
            "validation_command_cid",
            "repository_id",
            "repository_state_cid",
            "repository_forest_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "gitlink_closure_complete",
            "dirty",
            "dirty_overlay_cid",
            "decision_cid",
            "execution_key_cid",
            "test_receipt_cid",
            "certificate_cid",
            "policy_cid",
            "statement_cid",
            "circuit_cid",
            "verifying_key_cid",
            "proof_system_id",
            "certificate_epoch",
            "certificate_authority",
            "verifier_id",
            "verifier_result",
            "verifier_authority",
            "verified_at_ms",
            "fresh_until_ms",
            "attempted",
            "passed",
            "status",
            "reason_codes",
            "content_id",
            "validation_receipt_cid",
            "receipt_id",
        }
        if set(payload).difference(allowed):
            raise ProofCachedTestValidationError("validation receipt contains unsupported fields")
        if payload.get("schema") != cls.SCHEMA:
            raise ProofCachedTestValidationError("unsupported validation receipt schema")
        if payload.get("interface") != (PROOF_CACHED_TEST_VALIDATION_RECEIPT_INTERFACE):
            raise ProofCachedTestValidationError("unsupported validation receipt interface")
        if payload.get("contract_version") != (PROOF_CACHED_TEST_VALIDATION_VERSION):
            raise ProofCachedTestValidationError("unsupported validation receipt version")
        result = cls(
            task_id=payload.get("task_id", ""),
            goal_id=payload.get("goal_id", ""),
            goal_revision=payload.get("goal_revision", ""),
            validation_command=payload.get("validation_command", ""),
            validation_command_cid=payload.get("validation_command_cid", ""),
            repository_id=payload.get("repository_id", ""),
            repository_state_cid=payload.get("repository_state_cid", ""),
            repository_forest_cid=payload.get("repository_forest_cid", ""),
            git_commit_id=payload.get("git_commit_id", ""),
            git_tree_id=payload.get("git_tree_id", ""),
            gitlink_state_cid=payload.get("gitlink_state_cid", ""),
            gitlink_closure_complete=payload.get("gitlink_closure_complete", False),
            dirty=payload.get("dirty", False),
            dirty_overlay_cid=payload.get("dirty_overlay_cid", ""),
            decision_cid=payload.get("decision_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            test_receipt_cid=payload.get("test_receipt_cid", ""),
            certificate_cid=payload.get("certificate_cid", ""),
            policy_cid=payload.get("policy_cid", ""),
            statement_cid=payload.get("statement_cid", ""),
            circuit_cid=payload.get("circuit_cid", ""),
            verifying_key_cid=payload.get("verifying_key_cid", ""),
            proof_system_id=payload.get("proof_system_id", ""),
            certificate_epoch=payload.get("certificate_epoch", ""),
            certificate_authority=payload.get(
                "certificate_authority", CertificateAuthority.UNKNOWN
            ),
            verifier_id=payload.get("verifier_id", ""),
            verifier_result=payload.get(
                "verifier_result",
                ProofCachedTestValidationResult.NOT_ATTEMPTED,
            ),
            verifier_authority=payload.get("verifier_authority", CertificateAuthority.UNKNOWN),
            verified_at_ms=payload.get("verified_at_ms", 0),
            fresh_until_ms=payload.get("fresh_until_ms", 0),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        claimed = (
            payload.get("validation_receipt_cid")
            or payload.get("receipt_id")
            or payload.get("content_id")
        )
        if claimed and claimed != result.validation_receipt_cid:
            raise ProofCachedTestValidationError(
                "validation receipt content identity does not match payload"
            )
        if "attempted" in payload and payload["attempted"] is not True:
            raise ProofCachedTestValidationError(
                "validation receipt attempted flag is inconsistent"
            )
        if "passed" in payload:
            if not isinstance(payload["passed"], bool):
                raise ProofCachedTestValidationError(
                    "validation receipt passed flag must be boolean"
                )
            if payload["passed"] != result.passed:
                raise ProofCachedTestValidationError(
                    "validation receipt passed flag is inconsistent"
                )
        if "status" in payload and payload["status"] != result.status:
            raise ProofCachedTestValidationError("validation receipt status is inconsistent")
        return result

    def to_completion_evidence(
        self,
        *,
        acceptance_criterion: str,
        objective_revision: str = "",
        analyzer_version: str = PROOF_CACHED_TEST_VALIDATION_INTERFACE,
        configuration_revision: str = "",
        now_ms: int | None = None,
    ) -> Any:
        """Project this receipt into the supervisor ``CompletionEvidence`` API."""

        from ..objectives.goal_completion import CompletionEvidence

        current = self.is_completion_evidence(now_ms=now_ms)
        observed = datetime.fromtimestamp(self.verified_at_ms / 1_000, tz=UTC)
        fresh_until = datetime.fromtimestamp(self.fresh_until_ms / 1_000, tz=UTC)
        return CompletionEvidence(
            acceptance_criterion=acceptance_criterion,
            producing_task_or_scan=self.task_id,
            producer_id=self.task_id,
            producer_kind="task",
            producer_channel="proof_cached_test_validation",
            channel_proof_revision=self.interface,
            validation_receipt=self.to_record(),
            validation_passed=current,
            repository_id=self.repository_id,
            repository_tree=self.git_tree_id,
            tree_id=self.git_tree_id,
            objective_revision=objective_revision or self.goal_revision,
            analyzer_version=analyzer_version,
            configuration_revision=configuration_revision or self.policy_cid,
            observed_at=observed,
            fresh_until=fresh_until,
            freshness={"fresh": current, "status": "fresh" if current else "stale"},
            provenance_cid=self.validation_receipt_cid,
            metadata={
                "evidence_source_policy": {
                    "satisfies": current,
                    "reason_codes": [] if current else ["receipt_not_current"],
                },
                "proof_cached_test_validation_receipt_cid": (self.validation_receipt_cid),
                "certificate_cid": self.certificate_cid,
                "execution_key_cid": self.execution_key_cid,
            },
        )


# The supervisor's validation-evidence interface is the typed receipt itself;
# this public spelling avoids introducing a second, weaker evidence wrapper.
ValidationEvidence = ProofCachedTestValidationReceipt


@dataclass(frozen=True, slots=True)
class _RepositoryObservation:
    descriptor: RepositoryDescriptor | None
    reason: str = ""


class ProofCachedTestValidation:
    """Re-verify proof-backed pytest skips for supervisor completion."""

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = PROOF_CACHED_TEST_VALIDATION_INTERFACE

    def __init__(
        self,
        certificate_provider: Any | None = None,
        repository_root: str | Path | None = None,
        *,
        verifier: Any | None = None,
        verifier_id: str = "",
        repository_alias: str = "validation-root",
        freshness_seconds: float = DEFAULT_RECEIPT_FRESHNESS_SECONDS,
        clock: Callable[[], Any] = time.time,
        repository_observer: Callable[[], RepositoryDescriptor] | None = None,
    ) -> None:
        if certificate_provider is not None and verifier is not None:
            raise TypeError("supply certificate_provider or verifier, not both")
        selected_verifier = verifier if verifier is not None else certificate_provider
        if selected_verifier is None:
            raise TypeError("a certificate verifier is required")
        if isinstance(freshness_seconds, bool) or not isinstance(freshness_seconds, (int, float)):
            raise TypeError("freshness_seconds must be numeric")
        freshness = float(freshness_seconds)
        if not 0 < freshness <= MAX_RECEIPT_FRESHNESS_SECONDS:
            raise ValueError(
                "freshness_seconds must be positive and no greater than "
                f"{MAX_RECEIPT_FRESHNESS_SECONDS:g}"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        if repository_observer is not None and not callable(repository_observer):
            raise TypeError("repository_observer must be callable")

        self._verifier = selected_verifier
        self._repository_root = Path(repository_root or Path.cwd())
        self._repository_alias = _text(
            repository_alias,
            field_name="repository_alias",
            required=True,
        )
        self._freshness_ms = int(freshness * 1_000)
        self._clock = clock
        self._repository_observer = repository_observer
        inferred_id = (
            verifier_id
            or getattr(selected_verifier, "verifier_id", "")
            or getattr(selected_verifier, "backend_id", "")
            or (f"{type(selected_verifier).__module__}.{type(selected_verifier).__qualname__}")
        )
        self._verifier_id = _text(str(inferred_id), field_name="verifier_id", required=True)

    def _observe_repository(self) -> _RepositoryObservation:
        try:
            descriptor = (
                self._repository_observer()
                if self._repository_observer is not None
                else build_repository_descriptor(
                    self._repository_root,
                    alias=self._repository_alias,
                    logical_name=self._repository_alias,
                    authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
                )
            )
            if not isinstance(descriptor, RepositoryDescriptor):
                raise TypeError("repository_observer must return RepositoryDescriptor")
            return _RepositoryObservation(descriptor=descriptor)
        except Exception as exc:
            return _RepositoryObservation(
                descriptor=None,
                reason=type(exc).__name__,
            )

    @staticmethod
    def _coerce(value: Any, expected_type: type[Any]) -> Any:
        if isinstance(value, expected_type):
            return value
        if isinstance(value, Mapping):
            return expected_type.from_dict(value)
        raise TypeError(f"expected {expected_type.__name__} or canonical mapping")

    @staticmethod
    def _goal_identity(goal_id: str, goal: Mapping[str, Any] | Any | None) -> tuple[str, str]:
        if goal is None:
            return str(goal_id or "").strip(), ""
        if isinstance(goal, Mapping):
            mapped_id = goal.get("goal_id") or goal.get("objective_id") or goal.get("task_id") or ""
            revision = (
                goal.get("goal_revision")
                or goal.get("objective_revision")
                or goal.get("revision")
                or ""
            )
        else:
            mapped_id = (
                getattr(goal, "goal_id", "")
                or getattr(goal, "objective_id", "")
                or getattr(goal, "task_id", "")
            )
            revision = (
                getattr(goal, "goal_revision", "")
                or getattr(goal, "objective_revision", "")
                or getattr(goal, "revision", "")
            )
        explicit = str(goal_id or "").strip()
        derived = str(mapped_id or "").strip()
        if explicit and derived and explicit != derived:
            raise ValueError("goal_id does not match supplied goal")
        return explicit or derived, str(revision or "").strip()

    def _emit(
        self,
        *,
        task_id: str,
        goal_id: str,
        goal_revision: str,
        command: str,
        command_cid: str,
        observed_at_ms: int,
        observation: _RepositoryObservation,
        decision: ReuseDecision | None = None,
        execution_key: TestExecutionKey | None = None,
        pass_receipt: TestPassReceipt | None = None,
        certificate: TestProofCertificate | None = None,
        result: ProofCachedTestValidationResult = (ProofCachedTestValidationResult.NOT_ATTEMPTED),
        verifier_authority: CertificateAuthority = (CertificateAuthority.UNKNOWN),
        reason_codes: tuple[str, ...] = (),
    ) -> ProofCachedTestValidationReceipt:
        descriptor = observation.descriptor
        closure = descriptor.portable_closure if descriptor is not None else None
        return ProofCachedTestValidationReceipt(
            task_id=task_id or "unknown-task",
            goal_id=goal_id or "unknown-goal",
            goal_revision=goal_revision,
            validation_command=command or "<missing-validation-command>",
            validation_command_cid=command_cid,
            repository_id=descriptor.repository_id if descriptor else "",
            repository_state_cid=descriptor.descriptor_cid if descriptor else "",
            repository_forest_cid=(execution_key.repository_forest_cid if execution_key else ""),
            git_commit_id=descriptor.commit if descriptor else "",
            git_tree_id=descriptor.tree if descriptor else "",
            gitlink_state_cid=(closure.gitlink_closure_cid if closure is not None else ""),
            gitlink_closure_complete=(
                closure.gitlink_closure_complete if closure is not None else False
            ),
            dirty=descriptor.dirty if descriptor else False,
            dirty_overlay_cid=(descriptor.dirty_overlay_digest if descriptor else ""),
            decision_cid=decision.decision_id if decision else "",
            execution_key_cid=(execution_key.execution_key_id if execution_key else ""),
            test_receipt_cid=(pass_receipt.receipt_id if pass_receipt else ""),
            certificate_cid=(certificate.certificate_id if certificate else ""),
            policy_cid=(
                certificate.policy_cid
                if certificate is not None
                else execution_key.policy_cid
                if execution_key is not None
                else ""
            ),
            statement_cid=(certificate.statement_cid if certificate else ""),
            circuit_cid=certificate.circuit_cid if certificate else "",
            verifying_key_cid=(certificate.verifying_key_cid if certificate else ""),
            proof_system_id=(certificate.proof_system_id if certificate else ""),
            certificate_epoch=certificate.epoch if certificate else "",
            certificate_authority=(
                certificate.authority if certificate is not None else CertificateAuthority.UNKNOWN
            ),
            verifier_id=self._verifier_id,
            verifier_result=result,
            verifier_authority=verifier_authority,
            verified_at_ms=observed_at_ms,
            fresh_until_ms=observed_at_ms + self._freshness_ms,
            reason_codes=reason_codes,
        )

    def validate(
        self,
        *,
        task_id: str,
        validation_command: str,
        decision: ReuseDecision | Mapping[str, Any] | Any,
        execution_key: TestExecutionKey | Mapping[str, Any] | Any,
        certificate: TestProofCertificate | Mapping[str, Any] | Any,
        pass_receipt: TestPassReceipt | Mapping[str, Any] | Any | None = None,
        receipt: TestPassReceipt | Mapping[str, Any] | Any | None = None,
        goal_id: str = "",
        goal: Mapping[str, Any] | Any | None = None,
        goal_revision: str = "",
        policy_cid: str = "",
        current_epoch: str = "",
        proof: Any | None = None,
        proof_bytes: bytes | None = None,
        binding: Any | None = None,
    ) -> ProofCachedTestValidationReceipt:
        """Re-verify an exact proof-backed skip under current supervisor state."""

        observed_at_ms = _clock_milliseconds(self._clock)
        observation = self._observe_repository()

        normalized_task = str(task_id or "").strip()
        try:
            normalized_goal, derived_revision = self._goal_identity(goal_id, goal)
        except (TypeError, ValueError):
            normalized_goal, derived_revision = "", ""
        normalized_revision = str(goal_revision or derived_revision or "").strip()
        normalized_command = (
            validation_command.strip() if isinstance(validation_command, str) else ""
        )
        command_for_receipt = normalized_command or "<missing-validation-command>"
        command_cid = validation_command_identity(command_for_receipt)

        base = {
            "task_id": normalized_task,
            "goal_id": normalized_goal,
            "goal_revision": normalized_revision,
            "command": command_for_receipt,
            "command_cid": command_cid,
            "observed_at_ms": observed_at_ms,
            "observation": observation,
        }
        if not normalized_task or not normalized_goal:
            return self._emit(
                **base,
                reason_codes=(ProofCachedTestValidationReason.TASK_GOAL_MISSING.value,),
            )
        if not normalized_command:
            return self._emit(
                **base,
                reason_codes=(ProofCachedTestValidationReason.VALIDATION_COMMAND_MISSING.value,),
            )
        if isinstance(decision, str):
            return self._emit(
                **base,
                reason_codes=(ProofCachedTestValidationReason.PLAIN_SKIP_NOT_EVIDENCE.value,),
            )

        selected_receipt = pass_receipt if pass_receipt is not None else receipt
        if pass_receipt is not None and receipt is not None:
            return self._emit(
                **base,
                reason_codes=(ProofCachedTestValidationReason.MALFORMED_ARTIFACT.value,),
            )
        try:
            decision_obj = self._coerce(decision, ReuseDecision)
            key_obj = self._coerce(execution_key, TestExecutionKey)
            receipt_obj = self._coerce(selected_receipt, TestPassReceipt)
            certificate_obj = self._coerce(certificate, TestProofCertificate)
        except Exception:
            return self._emit(
                **base,
                reason_codes=(ProofCachedTestValidationReason.MALFORMED_ARTIFACT.value,),
            )
        artifacts = {
            "decision": decision_obj,
            "execution_key": key_obj,
            "pass_receipt": receipt_obj,
            "certificate": certificate_obj,
        }

        if (
            decision_obj.action is not ReuseAction.SKIP
            or decision_obj.authority is not CertificateAuthority.AUTHORITATIVE
        ):
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.NOT_PROOF_BACKED_SKIP.value,),
            )
        if (
            decision_obj.certificate_cid != certificate_obj.certificate_id
            or decision_obj.receipt_cid != receipt_obj.receipt_id
            or certificate_obj.receipt_cid != receipt_obj.receipt_id
        ):
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.ARTIFACT_CID_MISMATCH.value,),
            )
        if (
            receipt_obj.execution_key_cid != key_obj.execution_key_id
            or certificate_obj.execution_key_cid != key_obj.execution_key_id
            or receipt_obj.locator_cid != key_obj.locator_cid
            or receipt_obj.dependency_forest_cid != key_obj.repository_forest_cid
        ):
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.EXECUTION_KEY_MISMATCH.value,),
            )
        if not receipt_obj.admitted or not receipt_obj.all_phases_pass:
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.RECEIPT_NOT_PASSING.value,),
            )
        if key_obj.command_semantics_cid != command_cid:
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.VALIDATION_COMMAND_MISMATCH.value,),
            )
        if observation.descriptor is None:
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.REPOSITORY_OBSERVATION_FAILED.value,),
            )
        descriptor = observation.descriptor
        closure = descriptor.portable_closure
        if not closure.gitlink_closure_complete:
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.RECURSIVE_GITLINKS_INCOMPLETE.value,),
            )
        if (
            key_obj.git_commit_id != descriptor.commit
            or key_obj.git_tree_id != descriptor.tree
            or key_obj.gitlink_state_cid != closure.gitlink_closure_cid
            or key_obj.dirty_overlay_cid != descriptor.dirty_overlay_digest
        ):
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.REPOSITORY_STATE_MISMATCH.value,),
            )

        current_policy = str(policy_cid or "").strip()
        if not current_policy:
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.POLICY_MISSING.value,),
            )
        if (
            key_obj.policy_cid != current_policy
            or receipt_obj.policy_cid != current_policy
            or certificate_obj.policy_cid != current_policy
        ):
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.POLICY_MISMATCH.value,),
            )
        epoch = str(current_epoch or "").strip()
        if not epoch:
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.EPOCH_MISSING.value,),
            )
        if certificate_obj.epoch != epoch:
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.CERTIFICATE_STALE.value,),
            )
        if (
            certificate_obj.backend_mode is not ProofBackendMode.CRYPTOGRAPHIC
            or not certificate_obj.can_authorize_skip
        ):
            return self._emit(
                **base,
                **artifacts,
                reason_codes=(ProofCachedTestValidationReason.CERTIFICATE_NON_ATTESTED.value,),
            )

        requirements = {
            "task_id": normalized_task,
            "goal_id": normalized_goal,
            "goal_revision": normalized_revision,
            "validation_command_cid": command_cid,
            "execution_key_cid": key_obj.execution_key_id,
            "receipt_cid": receipt_obj.receipt_id,
            "certificate_cid": certificate_obj.certificate_id,
            "repository_id": descriptor.repository_id,
            "repository_state_cid": descriptor.descriptor_cid,
            "repository_forest_cid": key_obj.repository_forest_cid,
            "git_commit_id": descriptor.commit,
            "git_tree_id": descriptor.tree,
            "gitlink_state_cid": closure.gitlink_closure_cid,
            "dirty": descriptor.dirty,
            "dirty_overlay_cid": descriptor.dirty_overlay_digest,
            "policy_cid": current_policy,
            "statement_cid": certificate_obj.statement_cid,
            "circuit_cid": certificate_obj.circuit_cid,
            "verifying_key_cid": certificate_obj.verifying_key_cid,
            "proof_system_id": certificate_obj.proof_system_id,
            "allowed_epochs": [epoch],
        }
        verifier_kwargs: dict[str, Any] = {}
        if proof is not None:
            verifier_kwargs["proof"] = proof
        if proof_bytes is not None:
            verifier_kwargs["proof_bytes"] = proof_bytes
        if binding is not None:
            verifier_kwargs["binding"] = binding
        try:
            verify_retained = self._verifier.verify_retained_bytes
            verification = verify_retained(
                certificate_obj.canonical_bytes(),
                receipt_obj.canonical_bytes(),
                requirements,
                **verifier_kwargs,
            )
        except Exception:
            return self._emit(
                **base,
                **artifacts,
                result=ProofCachedTestValidationResult.UNAVAILABLE,
                verifier_authority=CertificateAuthority.NON_ATTESTED,
                reason_codes=(ProofCachedTestValidationReason.VERIFIER_UNAVAILABLE.value,),
            )
        if not isinstance(verification, TestCertificateVerificationResult):
            return self._emit(
                **base,
                **artifacts,
                result=ProofCachedTestValidationResult.REJECTED,
                verifier_authority=CertificateAuthority.NON_ATTESTED,
                reason_codes=(ProofCachedTestValidationReason.VERIFIER_NON_AUTHORITATIVE.value,),
            )
        if (
            verification.certificate_cid
            and verification.certificate_cid != certificate_obj.certificate_id
        ) or (verification.receipt_cid and verification.receipt_cid != receipt_obj.receipt_id):
            return self._emit(
                **base,
                **artifacts,
                result=ProofCachedTestValidationResult.REJECTED,
                verifier_authority=verification.authority,
                reason_codes=(ProofCachedTestValidationReason.ARTIFACT_CID_MISMATCH.value,),
            )
        if not verification.can_authorize_skip:
            unavailable = verification.status is TestCertificateVerificationStatus.UNAVAILABLE
            return self._emit(
                **base,
                **artifacts,
                result=(
                    ProofCachedTestValidationResult.UNAVAILABLE
                    if unavailable
                    else ProofCachedTestValidationResult.REJECTED
                ),
                verifier_authority=verification.authority,
                reason_codes=(
                    (
                        ProofCachedTestValidationReason.VERIFIER_UNAVAILABLE
                        if unavailable
                        else ProofCachedTestValidationReason.VERIFIER_REJECTED
                    ).value,
                    _enum_value(verification.reason_code),
                ),
            )
        return self._emit(
            **base,
            **artifacts,
            result=ProofCachedTestValidationResult.VERIFIED,
            verifier_authority=verification.authority,
            reason_codes=(ProofCachedTestValidationReason.PROOF_REVERIFIED.value,),
        )

    # Explicit compatibility spellings used by supervisor call sites.
    reverify = validate
    validate_skip = validate

    def __call__(self, **kwargs: Any) -> ProofCachedTestValidationReceipt:
        return self.validate(**kwargs)


__all__ = [
    "DEFAULT_RECEIPT_FRESHNESS_SECONDS",
    "MAX_RECEIPT_FRESHNESS_SECONDS",
    "PROOF_CACHED_TEST_VALIDATION_INTERFACE",
    "PROOF_CACHED_TEST_VALIDATION_RECEIPT_INTERFACE",
    "PROOF_CACHED_TEST_VALIDATION_RECEIPT_SCHEMA",
    "PROOF_CACHED_TEST_VALIDATION_VERSION",
    "ProofCachedTestValidation",
    "ProofCachedTestValidationError",
    "ProofCachedTestValidationReason",
    "ProofCachedTestValidationReceipt",
    "ProofCachedTestValidationResult",
    "ValidationEvidence",
    "validation_command_identity",
]
