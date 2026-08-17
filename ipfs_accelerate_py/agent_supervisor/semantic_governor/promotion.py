"""Authorized compare-and-swap promotion and rollback (SCG-034).

``promote_compression_policy`` and ``rollback_compression_policy`` revalidate
every publication gate, then publish exactly one policy version through the
kit expected-generation CAS repositories.

Normative fail-closed invariants (head must not mutate on any of these):

* Stale candidate (base policy CID/generation does not match the live head).
* Absent or unavailable release qualification.
* Absent promotion authorization.
* Reduced high-risk assurance (protected-threshold weakening at publication).
* Mismatched evaluation (wrong candidate, baseline, partition, or non-pass).
* CAS conflict / ABA / concurrent writer.
* Self-promotion (candidate, evaluation, proposal, policy, qualification, or
  seal cannot authorize itself).

Conflict policy: separate trusted authorization and expected-version CAS;
model output, candidate, evaluation, or seal cannot authorize itself.
Rollback is another authorized CAS, not history deletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Mapping, Optional, Protocol, Sequence
import re
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration_contracts import (
    EvidencePartition,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    CompressionPolicy,
    CompressionPolicyCandidate,
    CompressionPolicyPromotionReceipt,
    EvaluationVerdict,
    PolicyContractError,
    ProtectedThresholds,
    RuleEvaluationReport,
    assert_protected_threshold_change_authorized,
    protected_threshold_reductions,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.sealing import (
    QualificationPath,
    ReleaseQualification,
    SealingError,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_AUTHORIZED_PROMOTION_EVIDENCE: Final[str] = "scg/authorized-promotion@1"
PROMOTE_COMPRESSION_POLICY_INTERFACE: Final[str] = "promote_compression_policy@1"
ROLLBACK_COMPRESSION_POLICY_INTERFACE: Final[str] = (
    "rollback_compression_policy@1"
)
POLICY_PROMOTION_RESULT_INTERFACE: Final[str] = "PolicyPromotionResult@1"
POLICY_PROMOTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "policy-promotion-result@1"
)
POLICY_ROLLBACK_RESULT_INTERFACE: Final[str] = "PolicyRollbackResult@1"
POLICY_ROLLBACK_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "policy-rollback-result@1"
)

GENERATOR_ID: Final[str] = "policy_promoter"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "promotion.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_BLOCKING_REASONS: Final[int] = 256
MAX_METADATA_KEYS: Final[int] = 64
MAX_DIAGNOSTIC: Final[int] = 512

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$"
)
_OPERATION_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9](?:[a-z0-9._:-]{0,126}[a-z0-9])?"
)

# Closed reason codes for promotion / rollback fail-closed paths.
REASON_STALE_CANDIDATE: Final[str] = "stale_candidate"
REASON_ABSENT_QUALIFICATION: Final[str] = "absent_release_qualification"
REASON_UNAVAILABLE_QUALIFICATION: Final[str] = "unavailable_release_qualification"
REASON_ABSENT_AUTHORIZATION: Final[str] = "absent_authorization"
REASON_SELF_PROMOTION: Final[str] = "self_promotion_forbidden"
REASON_HIGH_RISK_REDUCTION: Final[str] = "high_risk_assurance_reduced"
REASON_MISMATCHED_EVALUATION: Final[str] = "mismatched_evaluation"
REASON_EVALUATION_NOT_PASS: Final[str] = "evaluation_verdict_not_pass"
REASON_CAS_CONFLICT: Final[str] = "cas_conflict"
REASON_CAS_UNAVAILABLE: Final[str] = "cas_unavailable"
REASON_CAS_CORRUPT: Final[str] = "cas_corrupt"
REASON_SCHEMA_INTEGRITY: Final[str] = "schema_or_integrity_failure"
REASON_POLICY_HEAD_MISMATCH: Final[str] = "policy_head_expectation_mismatch"
REASON_PROMOTED_POLICY_MISMATCH: Final[str] = "promoted_policy_cid_mismatch"
REASON_MISSING_REPOSITORY: Final[str] = "missing_policy_repository"
REASON_ROLLBACK_TARGET_INVALID: Final[str] = "invalid_rollback_target"
REASON_QUALIFICATION_MISMATCH: Final[str] = "qualification_identity_mismatch"
REASON_PROTECTED_REDUCTION_UNAUTHORIZED: Final[str] = (
    "protected_threshold_reduction_unauthorized"
)


class PromotionError(SemanticGovernorBaseError):
    """Raised when promotion/rollback inputs are malformed or unsafe to admit."""


class PromotionStatus(str, Enum):
    """Closed promotion publication outcomes."""

    PROMOTED = "promoted"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    CORRUPT = "corrupt"
    UNCHANGED = "unchanged"


class RollbackStatus(str, Enum):
    """Closed rollback publication outcomes."""

    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    CORRUPT = "corrupt"
    UNCHANGED = "unchanged"


# ---------------------------------------------------------------------------
# Repository protocols (structural; kit Durable* types satisfy these)
# ---------------------------------------------------------------------------


class _PolicyHeadLike(Protocol):
    policy_cid: Optional[str]
    generation: int
    transition_cid: Optional[str]
    namespace: str


class _PolicyCASResultLike(Protocol):
    status: Any
    before: _PolicyHeadLike
    after: _PolicyHeadLike
    transition_cid: Optional[str]
    reason_code: str
    local_durable: bool
    replicated: bool
    operation_id: str


class PolicyRepository(Protocol):
    """Minimal policy CAS surface used by promotion/rollback."""

    def current_policy(self, workspace: str) -> _PolicyHeadLike: ...

    def compare_and_swap_policy(
        self,
        workspace: str,
        *,
        expected_generation: int,
        expected_policy_cid: Optional[str],
        new_policy_cid: str,
        operation_id: str,
    ) -> _PolicyCASResultLike: ...


class RollbackPolicyRepository(PolicyRepository, Protocol):
    """Policy repository that also supports authorized rollback CAS."""

    def rollback_policy(
        self,
        workspace: str,
        *,
        expected_generation: int,
        expected_policy_cid: Optional[str],
        target_policy_cid: str,
        operation_id: str,
    ) -> _PolicyCASResultLike: ...


class _PromotionHeadLike(Protocol):
    promotion_cid: Optional[str]
    generation: int
    transition_cid: Optional[str]
    namespace: str


class _PromotionCASResultLike(Protocol):
    status: Any
    before: _PromotionHeadLike
    after: _PromotionHeadLike
    transition_cid: Optional[str]
    reason_code: str
    local_durable: bool
    replicated: bool
    operation_id: str
    candidate_cid: str
    authorization_cid: str


class PromotionStateRepository(Protocol):
    """Optional promotion-head CAS surface (receipt publication)."""

    def current_promotion(self, workspace: str) -> _PromotionHeadLike: ...

    def compare_and_swap_promotion(
        self,
        workspace: str,
        *,
        expected_generation: int,
        expected_promotion_cid: Optional[str],
        new_promotion_cid: str,
        operation_id: str,
        candidate_cid: str,
        authorization_cid: str,
    ) -> _PromotionCASResultLike: ...


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise PromotionError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise PromotionError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise PromotionError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise PromotionError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise PromotionError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _version(value: Any, name: str) -> str:
    text = _text(value, name)
    if _VERSION_RE.fullmatch(text) is None:
        raise PromotionError(f"{name} must be a normalized version token")
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PromotionError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise PromotionError(f"{name} must be a nonnegative integer")
    return value


def _operation_id(value: Any, name: str = "operation_id") -> str:
    text = _text(value, name)
    if _OPERATION_ID_RE.fullmatch(text) is None or len(text) > 128:
        raise PromotionError(
            f"{name} must be a normalized operation-id of length 1–128"
        )
    return text


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise PromotionError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise PromotionError(f"{name} exceeds maximum key count")
    # Fail closed on private / model-authority markers before durable emit.
    reject_private_and_model_authority(dict(value), path=name)
    return MappingProxyType(dict(value))


def _status_value(status: Any) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    if type(status) is str:
        return status
    # Duck-typed GovernorStoreStatus-like.
    value = getattr(status, "value", None)
    if type(value) is str:
        return value
    raise PromotionError("CAS status must be a string or enum")


def _normalize_candidate(
    value: CompressionPolicyCandidate | Mapping[str, Any],
) -> CompressionPolicyCandidate:
    if isinstance(value, CompressionPolicyCandidate):
        return value
    if isinstance(value, Mapping):
        try:
            return CompressionPolicyCandidate.from_dict(value)
        except (PolicyContractError, SemanticGovernorBaseError, TypeError, ValueError) as exc:
            raise PromotionError(
                f"invalid CompressionPolicyCandidate: {exc}"
            ) from exc
    raise PromotionError("candidate must be CompressionPolicyCandidate or mapping")


def _normalize_evaluation(
    value: RuleEvaluationReport | Mapping[str, Any],
) -> RuleEvaluationReport | dict[str, Any]:
    """Return a RuleEvaluationReport or a closed field dict for mapping inputs.

    Mapping form (as used by sealing tests) is admitted so callers can pass
    compact evaluation projections without reconstructing full contracts.
    """

    if isinstance(value, RuleEvaluationReport):
        return value
    if isinstance(value, Mapping):
        # Prefer full contract when schema is present.
        if "schema" in value and "interface_id" in value and "header" in value:
            try:
                return RuleEvaluationReport.from_dict(value)
            except (PolicyContractError, SemanticGovernorBaseError, TypeError, ValueError) as exc:
                raise PromotionError(
                    f"invalid RuleEvaluationReport: {exc}"
                ) from exc
        required = (
            "report_cid",
            "candidate_cid",
            "verdict",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise PromotionError(
                f"evaluation_report mapping missing fields: {missing}"
            )
        report_cid = _cid(value["report_cid"], "evaluation_report.report_cid")
        candidate_cid = _cid(
            value["candidate_cid"], "evaluation_report.candidate_cid"
        )
        baseline = _optional_cid(
            value.get("baseline_policy_cid"),
            "evaluation_report.baseline_policy_cid",
        )
        held_out = _optional_cid(
            value.get("held_out_benchmark_cid"),
            "evaluation_report.held_out_benchmark_cid",
        )
        verdict_raw = value["verdict"]
        if isinstance(verdict_raw, EvaluationVerdict):
            verdict = verdict_raw.value
        else:
            verdict = _token(verdict_raw, "evaluation_report.verdict")
        high_risk = value.get("high_risk_assurance_reduced", False)
        high_risk = _bool(high_risk, "evaluation_report.high_risk_assurance_reduced")
        partition = value.get("partition", EvidencePartition.HELD_OUT.value)
        if isinstance(partition, EvidencePartition):
            partition = partition.value
        else:
            partition = _token(partition, "evaluation_report.partition")
        blocking = tuple(value.get("blocking_reasons") or ())
        return {
            "report_cid": report_cid,
            "candidate_cid": candidate_cid,
            "baseline_policy_cid": baseline,
            "held_out_benchmark_cid": held_out,
            "verdict": verdict,
            "high_risk_assurance_reduced": high_risk,
            "partition": partition,
            "blocking_reasons": blocking,
            "declared_thresholds_applied": bool(
                value.get("declared_thresholds_applied", True)
            ),
        }
    raise PromotionError(
        "evaluation_report must be RuleEvaluationReport or mapping"
    )


def _evaluation_fields(
    evaluation: RuleEvaluationReport | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(evaluation, RuleEvaluationReport):
        verdict = evaluation.verdict
        if isinstance(verdict, EvaluationVerdict):
            verdict = verdict.value
        elif not isinstance(verdict, str):
            verdict = str(getattr(verdict, "value", verdict))
        partition = evaluation.partition
        if isinstance(partition, EvidencePartition):
            partition = partition.value
        elif not isinstance(partition, str):
            partition = str(getattr(partition, "value", partition))
        return {
            "report_cid": evaluation.report_cid,
            "candidate_cid": evaluation.candidate_cid,
            "baseline_policy_cid": evaluation.baseline_policy_cid,
            "held_out_benchmark_cid": evaluation.held_out_benchmark_cid,
            "verdict": verdict,
            "high_risk_assurance_reduced": evaluation.high_risk_assurance_reduced,
            "partition": partition,
            "blocking_reasons": tuple(evaluation.blocking_reasons),
            "declared_thresholds_applied": evaluation.declared_thresholds_applied,
        }
    # Already normalized mapping form.
    return dict(evaluation)


def _normalize_qualification(
    value: ReleaseQualification | Mapping[str, Any] | None,
) -> ReleaseQualification | None:
    if value is None:
        return None
    if isinstance(value, ReleaseQualification):
        return value
    if isinstance(value, Mapping):
        try:
            return ReleaseQualification.from_dict(value)
        except (SealingError, TypeError, ValueError, KeyError) as exc:
            raise PromotionError(
                f"invalid ReleaseQualification: {exc}"
            ) from exc
    raise PromotionError(
        "release_qualification must be ReleaseQualification or mapping"
    )


def _extract_authorization_cid(
    authorization: str | Mapping[str, Any] | None,
) -> str | None:
    if authorization is None:
        return None
    if type(authorization) is str:
        if not authorization:
            return None
        return _cid(authorization, "authorization")
    if isinstance(authorization, Mapping):
        for key in (
            "authorization_cid",
            "auth_cid",
            "cid",
            "promotion_authorization_cid",
        ):
            if key in authorization and authorization[key] is not None:
                return _cid(authorization[key], f"authorization.{key}")
        raise PromotionError(
            "authorization mapping must include authorization_cid"
        )
    raise PromotionError("authorization must be a CID string or mapping")


def _normalize_policy(
    value: CompressionPolicy | Mapping[str, Any] | None,
) -> CompressionPolicy | None:
    if value is None:
        return None
    if isinstance(value, CompressionPolicy):
        return value
    if isinstance(value, Mapping):
        try:
            return CompressionPolicy.from_dict(value)
        except (PolicyContractError, SemanticGovernorBaseError, TypeError, ValueError) as exc:
            raise PromotionError(f"invalid CompressionPolicy: {exc}") from exc
    raise PromotionError("promoted_policy must be CompressionPolicy or mapping")


def _normalize_thresholds(
    value: ProtectedThresholds | Mapping[str, Any],
) -> ProtectedThresholds:
    if isinstance(value, ProtectedThresholds):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value:
                return ProtectedThresholds.from_dict(value)
            return ProtectedThresholds(
                min_critical_omission_detection_bp=value[
                    "min_critical_omission_detection_bp"
                ],
                max_critical_omission_accepted=value[
                    "max_critical_omission_accepted"
                ],
                min_median_context_reduction_bp=value[
                    "min_median_context_reduction_bp"
                ],
                max_accepted_regression_bp=value["max_accepted_regression_bp"],
                min_shadow_sample_rate_bp=value["min_shadow_sample_rate_bp"],
                require_full_suite_fallback=value["require_full_suite_fallback"],
                allow_heuristic_as_exact=value["allow_heuristic_as_exact"],
                allow_assurance_reduction=value["allow_assurance_reduction"],
            )
        except (PolicyContractError, KeyError, TypeError, ValueError) as exc:
            raise PromotionError(
                f"invalid ProtectedThresholds: {exc}"
            ) from exc
    raise PromotionError("thresholds must be ProtectedThresholds or mapping")


def _cas_result_projection(result: _PolicyCASResultLike | None) -> dict[str, Any] | None:
    if result is None:
        return None
    status = _status_value(result.status)
    before = result.before
    after = result.after
    return {
        "status": status,
        "reason_code": str(result.reason_code),
        "operation_id": str(result.operation_id),
        "transition_cid": result.transition_cid,
        "local_durable": bool(result.local_durable),
        "replicated": bool(result.replicated),
        "before": {
            "policy_cid": before.policy_cid,
            "generation": int(before.generation),
            "transition_cid": before.transition_cid,
            "namespace": str(before.namespace),
        },
        "after": {
            "policy_cid": after.policy_cid,
            "generation": int(after.generation),
            "transition_cid": after.transition_cid,
            "namespace": str(after.namespace),
        },
    }


def _promotion_cas_projection(
    result: _PromotionCASResultLike | None,
) -> dict[str, Any] | None:
    if result is None:
        return None
    status = _status_value(result.status)
    before = result.before
    after = result.after
    return {
        "status": status,
        "reason_code": str(result.reason_code),
        "operation_id": str(result.operation_id),
        "transition_cid": result.transition_cid,
        "local_durable": bool(result.local_durable),
        "replicated": bool(result.replicated),
        "candidate_cid": result.candidate_cid,
        "authorization_cid": result.authorization_cid,
        "before": {
            "promotion_cid": before.promotion_cid,
            "generation": int(before.generation),
            "transition_cid": before.transition_cid,
            "namespace": str(before.namespace),
        },
        "after": {
            "promotion_cid": after.promotion_cid,
            "generation": int(after.generation),
            "transition_cid": after.transition_cid,
            "namespace": str(after.namespace),
        },
    }


def _stable_receipt_id(*parts: str) -> str:
    digest = cid_for_structured(
        {
            "interface_id": PROMOTE_COMPRESSION_POLICY_INTERFACE,
            "parts": list(parts),
        }
    )
    suffix = re.sub(r"[^a-z0-9]", "", digest.lower())[-24:] or "0"
    return f"promo_{suffix}"


def _stable_rollback_receipt_id(*parts: str) -> str:
    digest = cid_for_structured(
        {
            "interface_id": ROLLBACK_COMPRESSION_POLICY_INTERFACE,
            "parts": list(parts),
        }
    )
    suffix = re.sub(r"[^a-z0-9]", "", digest.lower())[-24:] or "0"
    return f"rollback_{suffix}"


def _build_receipt_header(
    *,
    candidate_cid: str,
    evaluation_report_cid: str,
    authorization_cid: str,
    previous_policy_cid: str,
    promoted_policy_cid: str,
    repository_state_cid: str | None,
    terminal_status: GovernorTerminalStatus,
) -> GovernorArtifactHeader:
    repo = repository_state_cid or previous_policy_cid
    return GovernorArtifactHeader(
        artifact_kind="compression_policy_promotion_receipt",
        repository_state_cid=repo,
        context_pack_cid=candidate_cid,
        verification_bundle_cid=evaluation_report_cid,
        generator=GeneratorIdentity(
            generator_id=GENERATOR_ID,
            generator_version=GENERATOR_VERSION,
            interface_id=PROMOTE_COMPRESSION_POLICY_INTERFACE,
        ),
        provenance=ArtifactProvenance(
            producer_id=PRODUCER_ID,
            producer_version=PRODUCER_VERSION,
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=tuple(
                sorted(
                    {
                        candidate_cid,
                        evaluation_report_cid,
                        authorization_cid,
                        previous_policy_cid,
                        promoted_policy_cid,
                    }
                )
            ),
            tool_ids=(TOOL_ID,),
            policy_cid=previous_policy_cid,
            notes=None,
        ),
        terminal_status=terminal_status,
        assumptions=(
            GovernorAssumption(
                assumption_id="expected_version_cas",
                kind=AssumptionKind.VERIFICATION,
                statement=(
                    "Promotion publishes one policy version only when "
                    "expected generation and policy CID match the live head"
                ),
                supporting_cids=(previous_policy_cid,),
            ),
            GovernorAssumption(
                assumption_id="no_self_promotion",
                kind=AssumptionKind.OTHER,
                statement=(
                    "Authorization is a distinct external CID; candidate, "
                    "evaluation, proposal, and policy cannot self-authorize"
                ),
                supporting_cids=(authorization_cid,),
            ),
        ),
        metadata={"evidence": SCG_AUTHORIZED_PROMOTION_EVIDENCE},
    )


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyPromotionResult:
    """Closed outcome of an authorized promotion attempt.

    ``head_mutated`` is true only when the policy CAS status is ``updated``.
    Rejected, conflict, unavailable, and corrupt outcomes never claim mutation.
    """

    status: str
    head_mutated: bool
    blocking_reasons: tuple[str, ...]
    workspace: str
    operation_id: str
    candidate_cid: str | None
    evaluation_report_cid: str | None
    authorization_cid: str | None
    qualification_cid: str | None
    expected_generation: int | None
    expected_policy_cid: str | None
    promoted_policy_cid: str | None
    receipt: CompressionPolicyPromotionReceipt | None
    policy_cas: Mapping[str, Any] | None
    promotion_cas: Mapping[str, Any] | None
    diagnostic: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "evidence",
            "status",
            "head_mutated",
            "blocking_reasons",
            "workspace",
            "operation_id",
            "candidate_cid",
            "evaluation_report_cid",
            "authorization_cid",
            "qualification_cid",
            "expected_generation",
            "expected_policy_cid",
            "promoted_policy_cid",
            "receipt",
            "policy_cas",
            "promotion_cas",
            "diagnostic",
            "metadata",
            "result_cid",
        }
    )

    def __post_init__(self) -> None:
        status = _token(self.status, "status")
        if status not in {item.value for item in PromotionStatus}:
            raise PromotionError(f"unknown promotion status: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "head_mutated", _bool(self.head_mutated, "head_mutated")
        )
        reasons = tuple(
            _token(item, f"blocking_reasons[{i}]")
            for i, item in enumerate(self.blocking_reasons or ())
        )
        if len(reasons) > MAX_BLOCKING_REASONS:
            reasons = reasons[:MAX_BLOCKING_REASONS]
        object.__setattr__(self, "blocking_reasons", reasons)
        object.__setattr__(self, "workspace", _token(self.workspace, "workspace"))
        object.__setattr__(
            self, "operation_id", _operation_id(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self, "candidate_cid", _optional_cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self,
            "evaluation_report_cid",
            _optional_cid(self.evaluation_report_cid, "evaluation_report_cid"),
        )
        object.__setattr__(
            self,
            "authorization_cid",
            _optional_cid(self.authorization_cid, "authorization_cid"),
        )
        object.__setattr__(
            self,
            "qualification_cid",
            _optional_cid(self.qualification_cid, "qualification_cid"),
        )
        if self.expected_generation is not None:
            object.__setattr__(
                self,
                "expected_generation",
                _nonneg_int(self.expected_generation, "expected_generation"),
            )
        object.__setattr__(
            self,
            "expected_policy_cid",
            _optional_cid(self.expected_policy_cid, "expected_policy_cid"),
        )
        object.__setattr__(
            self,
            "promoted_policy_cid",
            _optional_cid(self.promoted_policy_cid, "promoted_policy_cid"),
        )
        if self.receipt is not None and not isinstance(
            self.receipt, CompressionPolicyPromotionReceipt
        ):
            raise PromotionError("receipt must be CompressionPolicyPromotionReceipt")
        if self.policy_cas is not None and not isinstance(self.policy_cas, Mapping):
            raise PromotionError("policy_cas must be a mapping or None")
        if self.promotion_cas is not None and not isinstance(
            self.promotion_cas, Mapping
        ):
            raise PromotionError("promotion_cas must be a mapping or None")
        object.__setattr__(
            self,
            "policy_cas",
            MappingProxyType(dict(self.policy_cas)) if self.policy_cas else None,
        )
        object.__setattr__(
            self,
            "promotion_cas",
            MappingProxyType(dict(self.promotion_cas))
            if self.promotion_cas
            else None,
        )
        object.__setattr__(
            self, "diagnostic", _optional_text(self.diagnostic, "diagnostic")
        )
        if self.diagnostic is not None and len(self.diagnostic) > MAX_DIAGNOSTIC:
            object.__setattr__(self, "diagnostic", self.diagnostic[:MAX_DIAGNOSTIC])
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        # Consistency: only promoted+updated may claim mutation.
        if self.head_mutated and status != PromotionStatus.PROMOTED.value:
            raise PromotionError(
                "head_mutated requires status=promoted"
            )
        if status == PromotionStatus.PROMOTED.value and not self.head_mutated:
            raise PromotionError(
                "status=promoted requires head_mutated=True"
            )
        if self.head_mutated and self.blocking_reasons:
            raise PromotionError(
                "head_mutated cannot be true when blocking_reasons is nonempty"
            )
        if self.head_mutated and self.receipt is None:
            raise PromotionError("promoted results require a receipt")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": POLICY_PROMOTION_RESULT_SCHEMA,
            "interface_id": POLICY_PROMOTION_RESULT_INTERFACE,
            "evidence": SCG_AUTHORIZED_PROMOTION_EVIDENCE,
            "status": self.status,
            "head_mutated": self.head_mutated,
            "blocking_reasons": list(self.blocking_reasons),
            "workspace": self.workspace,
            "operation_id": self.operation_id,
            "candidate_cid": self.candidate_cid,
            "evaluation_report_cid": self.evaluation_report_cid,
            "authorization_cid": self.authorization_cid,
            "qualification_cid": self.qualification_cid,
            "expected_generation": self.expected_generation,
            "expected_policy_cid": self.expected_policy_cid,
            "promoted_policy_cid": self.promoted_policy_cid,
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "policy_cas": None if self.policy_cas is None else dict(self.policy_cas),
            "promotion_cas": (
                None if self.promotion_cas is None else dict(self.promotion_cas)
            ),
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_cid"] = self.result_cid
        return payload


@dataclass(frozen=True, slots=True)
class PolicyRollbackResult:
    """Closed outcome of an authorized rollback attempt."""

    status: str
    head_mutated: bool
    blocking_reasons: tuple[str, ...]
    workspace: str
    operation_id: str
    authorization_cid: str | None
    expected_generation: int | None
    expected_policy_cid: str | None
    target_policy_cid: str | None
    receipt: CompressionPolicyPromotionReceipt | None
    policy_cas: Mapping[str, Any] | None
    diagnostic: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = _token(self.status, "status")
        if status not in {item.value for item in RollbackStatus}:
            raise PromotionError(f"unknown rollback status: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "head_mutated", _bool(self.head_mutated, "head_mutated")
        )
        reasons = tuple(
            _token(item, f"blocking_reasons[{i}]")
            for i, item in enumerate(self.blocking_reasons or ())
        )
        if len(reasons) > MAX_BLOCKING_REASONS:
            reasons = reasons[:MAX_BLOCKING_REASONS]
        object.__setattr__(self, "blocking_reasons", reasons)
        object.__setattr__(self, "workspace", _token(self.workspace, "workspace"))
        object.__setattr__(
            self, "operation_id", _operation_id(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self,
            "authorization_cid",
            _optional_cid(self.authorization_cid, "authorization_cid"),
        )
        if self.expected_generation is not None:
            object.__setattr__(
                self,
                "expected_generation",
                _nonneg_int(self.expected_generation, "expected_generation"),
            )
        object.__setattr__(
            self,
            "expected_policy_cid",
            _optional_cid(self.expected_policy_cid, "expected_policy_cid"),
        )
        object.__setattr__(
            self,
            "target_policy_cid",
            _optional_cid(self.target_policy_cid, "target_policy_cid"),
        )
        if self.receipt is not None and not isinstance(
            self.receipt, CompressionPolicyPromotionReceipt
        ):
            raise PromotionError("receipt must be CompressionPolicyPromotionReceipt")
        object.__setattr__(
            self,
            "policy_cas",
            MappingProxyType(dict(self.policy_cas)) if self.policy_cas else None,
        )
        object.__setattr__(
            self, "diagnostic", _optional_text(self.diagnostic, "diagnostic")
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if self.head_mutated and status != RollbackStatus.ROLLED_BACK.value:
            raise PromotionError("head_mutated requires status=rolled_back")
        if status == RollbackStatus.ROLLED_BACK.value and not self.head_mutated:
            raise PromotionError(
                "status=rolled_back requires head_mutated=True"
            )
        if self.head_mutated and self.blocking_reasons:
            raise PromotionError(
                "head_mutated cannot be true when blocking_reasons is nonempty"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": POLICY_ROLLBACK_RESULT_SCHEMA,
            "interface_id": POLICY_ROLLBACK_RESULT_INTERFACE,
            "evidence": SCG_AUTHORIZED_PROMOTION_EVIDENCE,
            "status": self.status,
            "head_mutated": self.head_mutated,
            "blocking_reasons": list(self.blocking_reasons),
            "workspace": self.workspace,
            "operation_id": self.operation_id,
            "authorization_cid": self.authorization_cid,
            "expected_generation": self.expected_generation,
            "expected_policy_cid": self.expected_policy_cid,
            "target_policy_cid": self.target_policy_cid,
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
            "policy_cas": None if self.policy_cas is None else dict(self.policy_cas),
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_cid"] = self.result_cid
        return payload


# ---------------------------------------------------------------------------
# Gate evaluation (pure; no repository mutation)
# ---------------------------------------------------------------------------


def _collect_self_promotion_forbidden_cids(
    *,
    candidate: CompressionPolicyCandidate,
    evaluation: Mapping[str, Any],
    qualification: ReleaseQualification | None,
    promoted_policy_cid: str,
) -> set[str]:
    forbidden: set[str] = {
        candidate.candidate_cid,
        candidate.proposal_cid,
        candidate.base_policy_cid,
        candidate.proposed_policy_cid,
        promoted_policy_cid,
        evaluation["report_cid"],
    }
    if evaluation.get("held_out_benchmark_cid"):
        forbidden.add(evaluation["held_out_benchmark_cid"])
    if qualification is not None:
        forbidden.add(qualification.qualification_cid)
        if qualification.incremental_seal_cid:
            forbidden.add(qualification.incremental_seal_cid)
        if qualification.verification_bundle_cid:
            forbidden.add(qualification.verification_bundle_cid)
        # Seal / qualification artifacts cannot self-authorize promotion.
    return forbidden


def evaluate_promotion_gates(
    candidate: CompressionPolicyCandidate | Mapping[str, Any],
    evaluation_report: RuleEvaluationReport | Mapping[str, Any],
    authorization: str | Mapping[str, Any] | None,
    *,
    release_qualification: ReleaseQualification | Mapping[str, Any] | None,
    current_policy_cid: str | None,
    current_generation: int,
    expected_generation: int | None = None,
    expected_policy_cid: str | None = None,
    promoted_policy: CompressionPolicy | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Revalidate every publication gate without mutating any head.

    Returns a stable, unique tuple of blocking reason codes (empty when clear).
    """

    cand = _normalize_candidate(candidate)
    evaluation = _evaluation_fields(_normalize_evaluation(evaluation_report))
    auth_cid = _extract_authorization_cid(authorization)
    qual = _normalize_qualification(release_qualification)
    policy = _normalize_policy(promoted_policy)

    reasons: list[str] = []

    # --- evaluation integrity / match ---
    if evaluation["candidate_cid"] != cand.candidate_cid:
        reasons.append(REASON_MISMATCHED_EVALUATION)
    baseline = evaluation.get("baseline_policy_cid")
    if baseline is not None and baseline != cand.base_policy_cid:
        reasons.append(REASON_MISMATCHED_EVALUATION)
    partition = evaluation.get("partition") or EvidencePartition.HELD_OUT.value
    if partition != EvidencePartition.HELD_OUT.value:
        reasons.append(REASON_MISMATCHED_EVALUATION)
    verdict = evaluation["verdict"]
    if verdict != EvaluationVerdict.PASS.value:
        reasons.append(REASON_EVALUATION_NOT_PASS)
        if evaluation.get("blocking_reasons"):
            # Keep evaluation mismatch as primary; non-pass is already recorded.
            pass

    # --- high-risk assurance ---
    if evaluation.get("high_risk_assurance_reduced"):
        reasons.append(REASON_HIGH_RISK_REDUCTION)
    try:
        proposed = _normalize_thresholds(cand.proposed_protected_thresholds)
        baseline_thr = _normalize_thresholds(cand.baseline_protected_thresholds)
        reductions = protected_threshold_reductions(baseline_thr, proposed)
    except PromotionError:
        reasons.append(REASON_SCHEMA_INTEGRITY)
        reductions = ()
    if reductions:
        # Publication re-check: reductions require distinct external auth that
        # is not the candidate/proposal/policies/evaluation. Absence or self
        # identity blocks the head.
        try:
            assert_protected_threshold_change_authorized(
                baseline_thr,
                proposed,
                cand.external_authorization_cid,
                forbidden_self_cids={
                    cand.candidate_cid,
                    cand.proposal_cid,
                    cand.base_policy_cid,
                    cand.proposed_policy_cid,
                    evaluation["report_cid"],
                },
            )
        except PolicyContractError:
            reasons.append(REASON_PROTECTED_REDUCTION_UNAUTHORIZED)
        # Even authorized threshold reductions reduce high-risk assurance and
        # cannot pass evaluation; re-block at publication for defense in depth.
        if REASON_HIGH_RISK_REDUCTION not in reasons:
            reasons.append(REASON_HIGH_RISK_REDUCTION)

    # --- authorization ---
    if auth_cid is None:
        reasons.append(REASON_ABSENT_AUTHORIZATION)
    else:
        promoted_cid = cand.proposed_policy_cid
        if policy is not None:
            if policy.policy_cid != cand.proposed_policy_cid:
                reasons.append(REASON_PROMOTED_POLICY_MISMATCH)
            promoted_cid = policy.policy_cid
        forbidden = _collect_self_promotion_forbidden_cids(
            candidate=cand,
            evaluation=evaluation,
            qualification=qual,
            promoted_policy_cid=promoted_cid,
        )
        if auth_cid in forbidden:
            reasons.append(REASON_SELF_PROMOTION)

    # --- release qualification ---
    if qual is None:
        reasons.append(REASON_ABSENT_QUALIFICATION)
    else:
        if not qual.promotion_allowed:
            reasons.append(REASON_UNAVAILABLE_QUALIFICATION)
        if qual.path == QualificationPath.BLOCKED.value:
            if REASON_UNAVAILABLE_QUALIFICATION not in reasons:
                reasons.append(REASON_UNAVAILABLE_QUALIFICATION)
        if qual.candidate_cid != cand.candidate_cid:
            reasons.append(REASON_QUALIFICATION_MISMATCH)
        if qual.evaluation_report_cid != evaluation["report_cid"]:
            reasons.append(REASON_QUALIFICATION_MISMATCH)
        if (
            qual.baseline_policy_cid is not None
            and qual.baseline_policy_cid != cand.base_policy_cid
        ):
            reasons.append(REASON_QUALIFICATION_MISMATCH)

    # --- stale candidate / head expectation ---
    current_generation = _nonneg_int(current_generation, "current_generation")
    if current_policy_cid is not None:
        current_policy_cid = _cid(current_policy_cid, "current_policy_cid")

    if cand.base_policy_cid != current_policy_cid:
        # Includes generation-zero empty head (None) vs candidate base CID.
        reasons.append(REASON_STALE_CANDIDATE)

    exp_gen = (
        current_generation
        if expected_generation is None
        else _nonneg_int(expected_generation, "expected_generation")
    )
    exp_cid = (
        current_policy_cid
        if expected_policy_cid is None
        else _optional_cid(expected_policy_cid, "expected_policy_cid")
    )
    if exp_gen != current_generation or exp_cid != current_policy_cid:
        reasons.append(REASON_POLICY_HEAD_MISMATCH)

    # Stable unique order.
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def promote_compression_policy(
    candidate: CompressionPolicyCandidate | Mapping[str, Any],
    evaluation_report: RuleEvaluationReport | Mapping[str, Any],
    authorization: str | Mapping[str, Any] | None,
    *,
    release_qualification: ReleaseQualification | Mapping[str, Any] | None,
    policy_repository: PolicyRepository,
    workspace: str = "default",
    operation_id: str,
    expected_generation: int | None = None,
    expected_policy_cid: str | None = None,
    promoted_policy: CompressionPolicy | Mapping[str, Any] | None = None,
    promoted_policy_version: str | None = None,
    promotion_repository: PromotionStateRepository | None = None,
    repository_state_cid: str | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PolicyPromotionResult:
    """Authorize and CAS-publish a compression-policy successor.

    Revalidates evaluation, release qualification, high-risk assurance, and
    authorization at publication time, then performs one expected-version
    policy CAS. Stale, unauthorized, mismatched, or conflicting attempts leave
    the live head unchanged (``head_mutated=False``).
    """

    workspace = _token(workspace, "workspace")
    operation_id = _operation_id(operation_id)
    meta = _mapping(metadata, "metadata")
    notes = _optional_text(notes, "notes")

    if policy_repository is None:
        return PolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_MISSING_REPOSITORY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=None,
            evaluation_report_cid=None,
            authorization_cid=None,
            qualification_cid=None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            promoted_policy_cid=None,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic="policy_repository is required",
            metadata=meta,
        )

    try:
        cand = _normalize_candidate(candidate)
        evaluation_norm = _normalize_evaluation(evaluation_report)
        evaluation = _evaluation_fields(evaluation_norm)
        auth_cid = _extract_authorization_cid(authorization)
        qual = _normalize_qualification(release_qualification)
        policy = _normalize_policy(promoted_policy)
        if promoted_policy_version is not None:
            promoted_policy_version = _version(
                promoted_policy_version, "promoted_policy_version"
            )
        if policy is not None and promoted_policy_version is None:
            promoted_policy_version = policy.policy_version
        if promoted_policy_version is None:
            # Deterministic successor version label when not supplied.
            promoted_policy_version = f"{cand.base_policy_version}+promoted"
            # Version tokens disallow '+'? Check _VERSION_RE: [A-Za-z0-9._+-] — plus ok.
        current = policy_repository.current_policy(workspace)
        current_cid = current.policy_cid
        current_gen = int(current.generation)
    except PromotionError as exc:
        return PolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=None,
            evaluation_report_cid=None,
            authorization_cid=_extract_authorization_cid(authorization)
            if authorization is not None
            else None,
            qualification_cid=None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            promoted_policy_cid=None,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic=str(exc)[:MAX_DIAGNOSTIC],
            metadata=meta,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=None,
            evaluation_report_cid=None,
            authorization_cid=None,
            qualification_cid=None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            promoted_policy_cid=None,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic=str(exc)[:MAX_DIAGNOSTIC],
            metadata=meta,
        )

    exp_gen = current_gen if expected_generation is None else expected_generation
    exp_cid = current_cid if expected_policy_cid is None else expected_policy_cid

    # Already at the proposed successor: idempotent no-op (no second mutation).
    # Re-check authorization/self-promotion so an unauthorized party cannot claim
    # an unchanged success for an already-live head they did not authorize.
    if current_cid is not None and current_cid == cand.proposed_policy_cid:
        early_block: list[str] = []
        if auth_cid is None:
            early_block.append(REASON_ABSENT_AUTHORIZATION)
        else:
            forbidden = _collect_self_promotion_forbidden_cids(
                candidate=cand,
                evaluation=evaluation,
                qualification=qual,
                promoted_policy_cid=cand.proposed_policy_cid,
            )
            if auth_cid in forbidden:
                early_block.append(REASON_SELF_PROMOTION)
        if not early_block:
            return PolicyPromotionResult(
                status=PromotionStatus.UNCHANGED.value,
                head_mutated=False,
                blocking_reasons=(),
                workspace=workspace,
                operation_id=operation_id,
                candidate_cid=cand.candidate_cid,
                evaluation_report_cid=evaluation["report_cid"],
                authorization_cid=auth_cid,
                qualification_cid=(
                    None if qual is None else qual.qualification_cid
                ),
                expected_generation=exp_gen,
                expected_policy_cid=exp_cid,
                promoted_policy_cid=cand.proposed_policy_cid,
                receipt=None,
                policy_cas=None,
                promotion_cas=None,
                diagnostic="policy head already at proposed_policy_cid",
                metadata=meta,
            )

    blocking = evaluate_promotion_gates(
        cand,
        evaluation,
        auth_cid,
        release_qualification=qual,
        current_policy_cid=current_cid,
        current_generation=current_gen,
        expected_generation=exp_gen,
        expected_policy_cid=exp_cid,
        promoted_policy=policy,
    )

    qualification_cid = None if qual is None else qual.qualification_cid
    if blocking:
        return PolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=blocking,
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            qualification_cid=qualification_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic="promotion gates failed; policy head not mutated",
            metadata=meta,
        )

    # Gates clear — auth_cid and qual are non-None by construction.
    assert auth_cid is not None
    assert qual is not None

    # Build the promotion receipt *before* CAS so the receipt identity can be
    # published as the promotion head. Receipt construction fails closed on
    # self-authorization (defense in depth beyond evaluate_promotion_gates).
    try:
        receipt = CompressionPolicyPromotionReceipt(
            header=_build_receipt_header(
                candidate_cid=cand.candidate_cid,
                evaluation_report_cid=evaluation["report_cid"],
                authorization_cid=auth_cid,
                previous_policy_cid=cand.base_policy_cid,
                promoted_policy_cid=cand.proposed_policy_cid,
                repository_state_cid=repository_state_cid,
                terminal_status=GovernorTerminalStatus.COMPLETE,
            ),
            receipt_id=_stable_receipt_id(
                cand.candidate_cid,
                evaluation["report_cid"],
                auth_cid,
                operation_id,
            ),
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            proposal_cid=cand.proposal_cid,
            previous_policy_cid=cand.base_policy_cid,
            previous_policy_version=cand.base_policy_version,
            promoted_policy_cid=cand.proposed_policy_cid,
            promoted_policy_version=promoted_policy_version,
            rollback_policy_cid=cand.base_policy_cid,
            cas_expected_version=cand.base_policy_version,
            notes=notes,
            metadata={
                "evidence": SCG_AUTHORIZED_PROMOTION_EVIDENCE,
                "qualification_cid": qual.qualification_cid,
                "qualification_path": qual.path,
                "operation_id": operation_id,
                "expected_generation": exp_gen,
            },
        )
    except (PolicyContractError, SemanticGovernorBaseError) as exc:
        return PolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SELF_PROMOTION, REASON_SCHEMA_INTEGRITY),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            qualification_cid=qualification_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic=str(exc)[:MAX_DIAGNOSTIC],
            metadata=meta,
        )

    # Atomic policy CAS — only mutation path.
    try:
        cas = policy_repository.compare_and_swap_policy(
            workspace,
            expected_generation=int(exp_gen),
            expected_policy_cid=exp_cid,
            new_policy_cid=cand.proposed_policy_cid,
            operation_id=operation_id,
        )
    except Exception as exc:
        # Admission / integrity errors: head not mutated by failed call.
        return PolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            qualification_cid=qualification_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic=str(exc)[:MAX_DIAGNOSTIC],
            metadata=meta,
        )

    cas_status = _status_value(cas.status)
    cas_proj = _cas_result_projection(cas)

    if cas_status == "conflict":
        return PolicyPromotionResult(
            status=PromotionStatus.CONFLICT.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_CONFLICT,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            qualification_cid=qualification_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=f"policy CAS conflict: {cas.reason_code}",
            metadata=meta,
        )
    if cas_status == "unavailable":
        return PolicyPromotionResult(
            status=PromotionStatus.UNAVAILABLE.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_UNAVAILABLE,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            qualification_cid=qualification_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=f"policy CAS unavailable: {cas.reason_code}",
            metadata=meta,
        )
    if cas_status == "corrupt":
        return PolicyPromotionResult(
            status=PromotionStatus.CORRUPT.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_CORRUPT,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            qualification_cid=qualification_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=f"policy CAS corrupt: {cas.reason_code}",
            metadata=meta,
        )
    if cas_status == "unchanged":
        # Idempotent replay of a prior successful operation_id — head already
        # at the desired successor; report unchanged without double-counting.
        return PolicyPromotionResult(
            status=PromotionStatus.UNCHANGED.value,
            head_mutated=False,
            blocking_reasons=(),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            qualification_cid=qualification_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            receipt=receipt,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=f"policy CAS unchanged: {cas.reason_code}",
            metadata=meta,
        )
    if cas_status != "updated":
        return PolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation["report_cid"],
            authorization_cid=auth_cid,
            qualification_cid=qualification_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=f"unexpected policy CAS status: {cas_status}",
            metadata=meta,
        )

    # Optional promotion-head CAS (receipt). Failure does not roll back the
    # policy head — policy CAS already committed — but is recorded.
    promotion_cas_proj: dict[str, Any] | None = None
    if promotion_repository is not None:
        try:
            promo_head = promotion_repository.current_promotion(workspace)
            promo_cas = promotion_repository.compare_and_swap_promotion(
                workspace,
                expected_generation=int(promo_head.generation),
                expected_promotion_cid=promo_head.promotion_cid,
                new_promotion_cid=receipt.receipt_cid,
                operation_id=f"{operation_id}:promotion-head",
                candidate_cid=cand.candidate_cid,
                authorization_cid=auth_cid,
            )
            promotion_cas_proj = _promotion_cas_projection(promo_cas)
        except Exception as exc:
            promotion_cas_proj = {
                "status": "unavailable",
                "reason_code": "promotion_head_cas_failed",
                "diagnostic": str(exc)[:MAX_DIAGNOSTIC],
            }

    return PolicyPromotionResult(
        status=PromotionStatus.PROMOTED.value,
        head_mutated=True,
        blocking_reasons=(),
        workspace=workspace,
        operation_id=operation_id,
        candidate_cid=cand.candidate_cid,
        evaluation_report_cid=evaluation["report_cid"],
        authorization_cid=auth_cid,
        qualification_cid=qualification_cid,
        expected_generation=exp_gen,
        expected_policy_cid=exp_cid,
        promoted_policy_cid=cand.proposed_policy_cid,
        receipt=receipt,
        policy_cas=cas_proj,
        promotion_cas=promotion_cas_proj,
        diagnostic=None,
        metadata=meta,
    )


def rollback_compression_policy(
    authorization: str | Mapping[str, Any] | None,
    *,
    target_policy_cid: str,
    policy_repository: PolicyRepository,
    workspace: str = "default",
    operation_id: str,
    expected_generation: int | None = None,
    expected_policy_cid: str | None = None,
    current_policy_version: str = "current",
    target_policy_version: str = "rollback",
    candidate_cid: str | None = None,
    evaluation_report_cid: str | None = None,
    proposal_cid: str | None = None,
    repository_state_cid: str | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PolicyRollbackResult:
    """Authorize and CAS-publish a rollback to a prior policy CID.

    Rollback is a forward expected-version CAS (generation advances; history
    is retained). Self-authorization and absent authorization leave the head
    unchanged.
    """

    workspace = _token(workspace, "workspace")
    operation_id = _operation_id(operation_id)
    meta = _mapping(metadata, "metadata")
    notes = _optional_text(notes, "notes")
    target_policy_cid = _cid(target_policy_cid, "target_policy_cid")
    current_policy_version = _version(
        current_policy_version, "current_policy_version"
    )
    target_policy_version = _version(
        target_policy_version, "target_policy_version"
    )

    if policy_repository is None:
        return PolicyRollbackResult(
            status=RollbackStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_MISSING_REPOSITORY,),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=None,
            diagnostic="policy_repository is required",
            metadata=meta,
        )

    try:
        auth_cid = _extract_authorization_cid(authorization)
        current = policy_repository.current_policy(workspace)
        current_cid = current.policy_cid
        current_gen = int(current.generation)
    except PromotionError as exc:
        return PolicyRollbackResult(
            status=RollbackStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=None,
            diagnostic=str(exc)[:MAX_DIAGNOSTIC],
            metadata=meta,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PolicyRollbackResult(
            status=RollbackStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=None,
            diagnostic=str(exc)[:MAX_DIAGNOSTIC],
            metadata=meta,
        )

    blocking: list[str] = []
    if auth_cid is None:
        blocking.append(REASON_ABSENT_AUTHORIZATION)

    exp_gen = current_gen if expected_generation is None else expected_generation
    exp_cid = current_cid if expected_policy_cid is None else expected_policy_cid
    if exp_gen != current_gen or exp_cid != current_cid:
        blocking.append(REASON_POLICY_HEAD_MISMATCH)

    if current_cid is None:
        blocking.append(REASON_ROLLBACK_TARGET_INVALID)
    elif target_policy_cid == current_cid:
        blocking.append(REASON_ROLLBACK_TARGET_INVALID)

    # Self-promotion: authorization cannot be the current or target policy.
    if auth_cid is not None:
        forbidden = {target_policy_cid}
        if current_cid is not None:
            forbidden.add(current_cid)
        if candidate_cid is not None:
            forbidden.add(_cid(candidate_cid, "candidate_cid"))
        if evaluation_report_cid is not None:
            forbidden.add(_cid(evaluation_report_cid, "evaluation_report_cid"))
        if proposal_cid is not None:
            forbidden.add(_cid(proposal_cid, "proposal_cid"))
        if auth_cid in forbidden:
            blocking.append(REASON_SELF_PROMOTION)

    if blocking:
        seen: set[str] = set()
        ordered = []
        for reason in blocking:
            if reason not in seen:
                seen.add(reason)
                ordered.append(reason)
        return PolicyRollbackResult(
            status=RollbackStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=tuple(ordered),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=auth_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=None,
            diagnostic="rollback gates failed; policy head not mutated",
            metadata=meta,
        )

    assert auth_cid is not None
    assert current_cid is not None

    # Use rollback_policy when available (enforces prior-history target).
    try:
        rollback_fn = getattr(policy_repository, "rollback_policy", None)
        if callable(rollback_fn):
            cas = rollback_fn(
                workspace,
                expected_generation=int(exp_gen),
                expected_policy_cid=exp_cid,
                target_policy_cid=target_policy_cid,
                operation_id=operation_id,
            )
        else:
            cas = policy_repository.compare_and_swap_policy(
                workspace,
                expected_generation=int(exp_gen),
                expected_policy_cid=exp_cid,
                new_policy_cid=target_policy_cid,
                operation_id=operation_id,
            )
    except Exception as exc:
        return PolicyRollbackResult(
            status=RollbackStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY, REASON_ROLLBACK_TARGET_INVALID),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=auth_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=None,
            diagnostic=str(exc)[:MAX_DIAGNOSTIC],
            metadata=meta,
        )

    cas_status = _status_value(cas.status)
    cas_proj = _cas_result_projection(cas)

    if cas_status == "conflict":
        return PolicyRollbackResult(
            status=RollbackStatus.CONFLICT.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_CONFLICT,),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=auth_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            diagnostic=f"rollback CAS conflict: {cas.reason_code}",
            metadata=meta,
        )
    if cas_status == "unavailable":
        return PolicyRollbackResult(
            status=RollbackStatus.UNAVAILABLE.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_UNAVAILABLE,),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=auth_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            diagnostic=f"rollback CAS unavailable: {cas.reason_code}",
            metadata=meta,
        )
    if cas_status == "corrupt":
        return PolicyRollbackResult(
            status=RollbackStatus.CORRUPT.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_CORRUPT,),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=auth_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            diagnostic=f"rollback CAS corrupt: {cas.reason_code}",
            metadata=meta,
        )
    if cas_status == "unchanged":
        return PolicyRollbackResult(
            status=RollbackStatus.UNCHANGED.value,
            head_mutated=False,
            blocking_reasons=(),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=auth_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            diagnostic=f"rollback CAS unchanged: {cas.reason_code}",
            metadata=meta,
        )
    if cas_status != "updated":
        return PolicyRollbackResult(
            status=RollbackStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=auth_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            diagnostic=f"unexpected rollback CAS status: {cas_status}",
            metadata=meta,
        )

    # Neutral rollback receipt: reuses promotion receipt schema with distinct
    # synthetic candidate/evaluation identities derived from the operation so
    # authorization remains external and non-self.
    synth_candidate = candidate_cid or cid_for_structured(
        {
            "kind": "rollback_candidate_ref",
            "operation_id": operation_id,
            "from": current_cid,
            "to": target_policy_cid,
        }
    )
    synth_eval = evaluation_report_cid or cid_for_structured(
        {
            "kind": "rollback_evaluation_ref",
            "operation_id": operation_id,
            "target_policy_cid": target_policy_cid,
        }
    )
    synth_proposal = proposal_cid or cid_for_structured(
        {
            "kind": "rollback_proposal_ref",
            "operation_id": operation_id,
        }
    )

    try:
        receipt = CompressionPolicyPromotionReceipt(
            header=_build_receipt_header(
                candidate_cid=synth_candidate,
                evaluation_report_cid=synth_eval,
                authorization_cid=auth_cid,
                previous_policy_cid=current_cid,
                promoted_policy_cid=target_policy_cid,
                repository_state_cid=repository_state_cid,
                terminal_status=GovernorTerminalStatus.COMPLETE,
            ),
            receipt_id=_stable_rollback_receipt_id(
                current_cid, target_policy_cid, auth_cid, operation_id
            ),
            candidate_cid=synth_candidate,
            evaluation_report_cid=synth_eval,
            authorization_cid=auth_cid,
            proposal_cid=synth_proposal,
            previous_policy_cid=current_cid,
            previous_policy_version=current_policy_version,
            promoted_policy_cid=target_policy_cid,
            promoted_policy_version=target_policy_version,
            rollback_policy_cid=target_policy_cid,
            cas_expected_version=current_policy_version,
            notes=notes,
            metadata={
                "evidence": SCG_AUTHORIZED_PROMOTION_EVIDENCE,
                "kind": "rollback",
                "operation_id": operation_id,
                "expected_generation": exp_gen,
            },
        )
    except (PolicyContractError, SemanticGovernorBaseError) as exc:
        # Policy head already rolled back; report success with diagnostic that
        # receipt construction failed (still head_mutated).
        return PolicyRollbackResult(
            status=RollbackStatus.ROLLED_BACK.value,
            head_mutated=True,
            blocking_reasons=(),
            workspace=workspace,
            operation_id=operation_id,
            authorization_cid=auth_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            target_policy_cid=target_policy_cid,
            receipt=None,
            policy_cas=cas_proj,
            diagnostic=(
                f"rollback committed but receipt failed: {exc}"
            )[:MAX_DIAGNOSTIC],
            metadata=meta,
        )

    return PolicyRollbackResult(
        status=RollbackStatus.ROLLED_BACK.value,
        head_mutated=True,
        blocking_reasons=(),
        workspace=workspace,
        operation_id=operation_id,
        authorization_cid=auth_cid,
        expected_generation=exp_gen,
        expected_policy_cid=exp_cid,
        target_policy_cid=target_policy_cid,
        receipt=receipt,
        policy_cas=cas_proj,
        diagnostic=None,
        metadata=meta,
    )


__all__ = [
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "POLICY_PROMOTION_RESULT_INTERFACE",
    "POLICY_PROMOTION_RESULT_SCHEMA",
    "POLICY_ROLLBACK_RESULT_INTERFACE",
    "POLICY_ROLLBACK_RESULT_SCHEMA",
    "PROMOTE_COMPRESSION_POLICY_INTERFACE",
    "ROLLBACK_COMPRESSION_POLICY_INTERFACE",
    "PolicyPromotionResult",
    "PolicyRepository",
    "PolicyRollbackResult",
    "PromotionError",
    "PromotionStateRepository",
    "PromotionStatus",
    "REASON_ABSENT_AUTHORIZATION",
    "REASON_ABSENT_QUALIFICATION",
    "REASON_CAS_CONFLICT",
    "REASON_CAS_CORRUPT",
    "REASON_CAS_UNAVAILABLE",
    "REASON_EVALUATION_NOT_PASS",
    "REASON_HIGH_RISK_REDUCTION",
    "REASON_MISMATCHED_EVALUATION",
    "REASON_POLICY_HEAD_MISMATCH",
    "REASON_PROTECTED_REDUCTION_UNAUTHORIZED",
    "REASON_PROMOTED_POLICY_MISMATCH",
    "REASON_QUALIFICATION_MISMATCH",
    "REASON_ROLLBACK_TARGET_INVALID",
    "REASON_SCHEMA_INTEGRITY",
    "REASON_SELF_PROMOTION",
    "REASON_STALE_CANDIDATE",
    "REASON_UNAVAILABLE_QUALIFICATION",
    "ROLLBACK_COMPRESSION_POLICY_INTERFACE",
    "RollbackStatus",
    "SCG_AUTHORIZED_PROMOTION_EVIDENCE",
    "evaluate_promotion_gates",
    "promote_compression_policy",
    "rollback_compression_policy",
]
