"""Orchestrate authorized assurance-policy promotion, CAS, and new seal (AAE-047).

Interface surface:

* ``promote_assurance_policy@1`` — revalidate held-out qualification, external
  authorization, campaign/promotion receipt signature bindings, released
  incremental seal evidence, and expected-old policy CAS before publishing a
  successor policy head.

Authority rules (normative):

* Canonical identity comes only from ``software_contracts.content``.
* Candidates cannot self-promote: authorization CIDs must be distinct from
  candidate, evaluation, campaign, seal, plan, and promoted-policy identities.
* Held-out pass, no regression/vacuity, declared cost and coverage, verified
  signer/key/audience/action bindings on campaign and promotion receipts,
  expected-old CAS, and a released incremental seal are mandatory for
  ``promoted`` outcomes.
* Production policy never changes during a pure fixture campaign; CAS mutates
  only the caller's disposable coordination store via
  ``AssurancePolicyRepository@1``.
* Signature verification status is recorded and gated by the existing receipt
  authority; this module performs no host key operations.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Optional, Protocol

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceBaseError,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.held_out import (
    QualificationDisposition,
    RemediationQualificationResult,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.receipt_contracts import (
    ASSURANCE_POLICY_PROMOTION_RECEIPT_INTERFACE,
    AssuranceCampaignReceipt,
    AssurancePolicyPromotionReceipt,
    HeldOutResult,
    ReceiptAction,
    ReceiptContractError,
    ReceiptSignatureBinding,
    SealAvailabilityStatus,
    SealScopeItem,
    SignatureVerificationStatus,
    require_verified_signature_before_persistence,
    verify_campaign_receipt_identity,
    verify_promotion_receipt_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    EvaluationVerdict,
    GapRemediationPlan,
    RemediationEvaluationReport,
    RemediationPlanStatus,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

PROMOTE_ASSURANCE_POLICY_INTERFACE: Final[str] = "promote_assurance_policy@1"
AAE_PROMOTION_EVIDENCE: Final[str] = "aae/promotion@1"

POLICY_PROMOTION_RESULT_INTERFACE: Final[str] = "AssurancePolicyPromotionResult@1"
POLICY_PROMOTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "policy-promotion-result@1"
)

GENERATOR_ID: Final[str] = "assurance_policy_promoter"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "adversarial_assurance"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "promotion.v1"
ADAPTER_ID: Final[str] = "aae-047-promotion@1"

DEFAULT_AUDIENCE: Final[str] = "adversarial_assurance.store"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_BLOCKING_REASONS: Final[int] = 256
MAX_METADATA_KEYS: Final[int] = 64
MAX_DIAGNOSTIC: Final[int] = 512
MAX_COST_BP: Final[int] = 100_000

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$"
)
_OPERATION_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9](?:[a-z0-9._:-]{0,126}[a-z0-9])?$"
)

# Closed promotion seal scope (plan §10 / §14 + AAE-012 receipt contract).
REQUIRED_PROMOTION_SEAL_SCOPE: Final[tuple[str, ...]] = (
    SealScopeItem.FINAL_POLICY_REVISION.value,
    SealScopeItem.EVALUATION_TO_PROMOTION_BINDING.value,
    SealScopeItem.HELD_OUT_EVALUATIONS.value,
    SealScopeItem.STATUS_POLICY_SATISFACTION.value,
    SealScopeItem.PROMOTION_RECEIPT.value,
)

# Closed reason codes for fail-closed promotion paths.
REASON_ABSENT_AUTHORIZATION: Final[str] = "absent_authorization"
REASON_ABSENT_CANDIDATE: Final[str] = "absent_canonical_candidate"
REASON_ABSENT_CAMPAIGN_RECEIPT: Final[str] = "absent_campaign_receipt"
REASON_ABSENT_EVALUATION: Final[str] = "absent_evaluation"
REASON_ABSENT_SEAL: Final[str] = "absent_released_incremental_seal"
REASON_CAS_CONFLICT: Final[str] = "cas_conflict"
REASON_CAS_CORRUPT: Final[str] = "cas_corrupt"
REASON_CAS_UNAVAILABLE: Final[str] = "cas_unavailable"
REASON_COST_NOT_DECLARED: Final[str] = "cost_not_declared"
REASON_COST_EXCEEDED: Final[str] = "cost_exceeded"
REASON_COVERAGE_NOT_DECLARED: Final[str] = "coverage_not_declared"
REASON_EVALUATION_NOT_PASS: Final[str] = "evaluation_verdict_not_pass"
REASON_HELD_OUT_NOT_PASS: Final[str] = "held_out_not_pass"
REASON_INVALID_SIGNATURE_BINDINGS: Final[str] = "invalid_signature_bindings"
REASON_MISSING_REPOSITORY: Final[str] = "missing_policy_repository"
REASON_POLICY_HEAD_MISMATCH: Final[str] = "policy_head_expectation_mismatch"
REASON_PROMOTED_POLICY_MISMATCH: Final[str] = "promoted_policy_cid_mismatch"
REASON_REGRESSION_DETECTED: Final[str] = "regression_detected"
REASON_SCHEMA_INTEGRITY: Final[str] = "schema_or_integrity_failure"
REASON_SEAL_UNAVAILABLE: Final[str] = "released_seal_unavailable"
REASON_SELF_PROMOTION: Final[str] = "self_promotion_forbidden"
REASON_STALE_CANDIDATE: Final[str] = "stale_candidate"
REASON_UNVERIFIED_CAMPAIGN_RECEIPT: Final[str] = "unverified_campaign_receipt"
REASON_UNVERIFIED_PROMOTION_RECEIPT: Final[str] = "unverified_promotion_receipt"
REASON_VACUITY_DETECTED: Final[str] = "vacuity_detected"
REASON_MISSING_PROMOTED_POLICY: Final[str] = "missing_promoted_policy_cid"


class PromotionError(AssuranceBaseError):
    """Raised when promotion inputs are malformed or unsafe to admit."""


class PromotionStatus(str, Enum):
    """Closed promotion publication outcomes."""

    PROMOTED = "promoted"
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


class _PromotionCASResultLike(Protocol):
    status: Any
    before: Any
    after: Any
    transition_cid: Optional[str]
    reason_code: str
    local_durable: bool
    replicated: bool
    operation_id: str
    candidate_cid: str
    evaluation_cid: str
    authorization_cid: str
    expected_old_policy_generation: int
    expected_old_policy_cid: Optional[str]


class PolicyRepository(Protocol):
    """Minimal policy CAS surface used by assurance promotion."""

    def current_policy(self, workspace: str) -> _PolicyHeadLike: ...

    def promote_policy(
        self,
        workspace: str,
        *,
        expected_generation: int,
        expected_policy_cid: Optional[str],
        new_policy_cid: str,
        operation_id: str,
        candidate_cid: str,
        evaluation_cid: str,
        authorization_cid: str,
    ) -> _PromotionCASResultLike: ...


class PromotionStateRepository(Protocol):
    """Optional promotion-head CAS surface (receipt publication)."""

    def current_promotion(self, workspace: str) -> Any: ...

    def compare_and_swap_promotion(
        self,
        workspace: str,
        *,
        expected_generation: int,
        expected_promotion_cid: Optional[str],
        new_promotion_cid: str,
        operation_id: str,
        candidate_cid: str,
        evaluation_cid: str,
        authorization_cid: str,
        expected_old_policy_generation: int,
        expected_old_policy_cid: Optional[str],
    ) -> _PromotionCASResultLike: ...


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    empty: bool = False,
    maximum: int = MAX_TEXT_CHARS,
) -> str:
    if type(value) is not str or (not empty and not value):
        raise PromotionError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise PromotionError(f"{name} must be trimmed NFC text")
    if len(value) > maximum or any(not char.isprintable() for char in value):
        raise PromotionError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, empty=True)


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
        raise PromotionError(
            f"{name} must be a version token matching {_VERSION_RE.pattern}"
        )
    return text


def _operation_id(value: Any, name: str = "operation_id") -> str:
    text = _text(value, name, maximum=128)
    if _OPERATION_ID_RE.fullmatch(text) is None:
        raise PromotionError(f"{name} must be a durable operation id token")
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PromotionError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise PromotionError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise PromotionError(f"{name} exceeds maximum {maximum}")
    return value


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise PromotionError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise PromotionError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise PromotionError(f"{name} exceeds maximum key count")
    return MappingProxyType(dict(value))


def _status_value(status: Any) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    if type(status) is str:
        return status
    raise PromotionError("CAS status must be a string or enum")


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Normalized input records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalizedCandidate:
    """Canonical promotion candidate identity and policy pins."""

    candidate_cid: str
    proposed_policy_cid: str
    base_policy_cid: str
    base_policy_version: str
    proposed_policy_version: str
    plan_cid: str | None = None
    cost_delta_basis_points: int | None = None
    coverage_declared: bool = False
    coverage_partitions: tuple[str, ...] = ()
    draft_status: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NormalizedEvaluation:
    """Held-out evaluation / qualification projection used by promotion gates."""

    evaluation_report_cid: str
    verdict: str
    held_out_result: str
    held_out_killed: bool
    regression_detected: bool
    vacuity_detected: bool
    cost_delta_basis_points: int | None
    cost_declared: bool
    coverage_declared: bool
    coverage_partitions: tuple[str, ...]
    qualification_cid: str | None = None
    disposition: str | None = None
    qualified: bool = False
    max_cost_delta_bp: int = MAX_COST_BP
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


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


def _normalize_candidate(
    remediation: Any,
    *,
    promoted_policy_cid: str | None = None,
    promoted_policy_version: str | None = None,
    base_policy_cid: str | None = None,
    base_policy_version: str | None = None,
) -> NormalizedCandidate:
    """Admit plan/proposal/mapping remediation as a canonical candidate."""

    if remediation is None:
        raise PromotionError("remediation/candidate is required")

    if isinstance(remediation, GapRemediationPlan):
        plan = remediation
        candidate_cid = plan.plan_cid
        plan_cid = plan.plan_cid
        meta = dict(plan.metadata)
        draft_status = plan.plan_status
        proposed = (
            promoted_policy_cid
            or meta.get("proposed_policy_cid")
            or meta.get("promoted_policy_cid")
        )
        base = (
            base_policy_cid
            or meta.get("base_policy_cid")
            or meta.get("previous_policy_cid")
            or plan.header.provenance.policy_cid
        )
        base_ver = (
            base_policy_version
            or meta.get("base_policy_version")
            or meta.get("previous_policy_version")
            or plan.header.versions.campaign_policy_version
        )
        prop_ver = (
            promoted_policy_version
            or meta.get("proposed_policy_version")
            or meta.get("promoted_policy_version")
        )
        cost = meta.get("cost_delta_basis_points")
        coverage_declared = bool(
            meta.get("coverage_declared")
            or meta.get("coverage_impact_declared")
            or plan.candidate_test_cids
            or plan.candidate_proof_cids
            or plan.candidate_policy_cids
            or plan.candidate_analyzer_cids
        )
        partitions = tuple(
            str(item)
            for item in (meta.get("coverage_partitions") or ())
            if isinstance(item, str) and item
        )
        if proposed is None:
            raise PromotionError(
                "GapRemediationPlan requires promoted_policy_cid "
                "(argument or metadata)"
            )
        if base is None:
            raise PromotionError(
                "GapRemediationPlan requires base_policy_cid "
                "(argument, metadata, or provenance.policy_cid)"
            )
        if prop_ver is None:
            prop_ver = f"{base_ver}+promoted" if base_ver else "promoted.1"
        return NormalizedCandidate(
            candidate_cid=_cid(candidate_cid, "candidate_cid"),
            proposed_policy_cid=_cid(proposed, "proposed_policy_cid"),
            base_policy_cid=_cid(base, "base_policy_cid"),
            base_policy_version=_version(base_ver, "base_policy_version"),
            proposed_policy_version=_version(
                prop_ver, "proposed_policy_version"
            ),
            plan_cid=_cid(plan_cid, "plan_cid"),
            cost_delta_basis_points=(
                None
                if cost is None
                else _nonneg_int(int(cost), "cost_delta_basis_points", maximum=MAX_COST_BP)
            ),
            coverage_declared=coverage_declared,
            coverage_partitions=partitions,
            draft_status=str(draft_status) if draft_status is not None else None,
            metadata=MappingProxyType(meta),
        )

    if isinstance(remediation, Mapping):
        data = dict(remediation)
        candidate_cid = (
            data.get("candidate_cid")
            or data.get("plan_cid")
            or data.get("remediation_cid")
        )
        if candidate_cid is None:
            raise PromotionError(
                "remediation mapping must include candidate_cid or plan_cid"
            )
        proposed = (
            promoted_policy_cid
            or data.get("proposed_policy_cid")
            or data.get("promoted_policy_cid")
            or data.get("new_policy_cid")
        )
        base = (
            base_policy_cid
            or data.get("base_policy_cid")
            or data.get("previous_policy_cid")
            or data.get("expected_old_policy_cid")
        )
        base_ver = (
            base_policy_version
            or data.get("base_policy_version")
            or data.get("previous_policy_version")
            or data.get("expected_old_policy_version")
        )
        prop_ver = (
            promoted_policy_version
            or data.get("proposed_policy_version")
            or data.get("promoted_policy_version")
        )
        if proposed is None:
            raise PromotionError(
                "remediation mapping must include proposed_policy_cid"
            )
        if base is None or base_ver is None:
            raise PromotionError(
                "remediation mapping must include base_policy_cid and "
                "base_policy_version"
            )
        if prop_ver is None:
            prop_ver = f"{base_ver}+promoted"
        cost = data.get("cost_delta_basis_points", data.get("declared_cost_delta_bp"))
        coverage_declared = bool(
            data.get("coverage_declared")
            or data.get("coverage_impact_declared")
            or data.get("coverage_partitions")
            or data.get("declared_coverage")
        )
        partitions = tuple(
            str(item)
            for item in (data.get("coverage_partitions") or ())
            if isinstance(item, str) and item
        )
        plan_cid = data.get("plan_cid")
        return NormalizedCandidate(
            candidate_cid=_cid(candidate_cid, "candidate_cid"),
            proposed_policy_cid=_cid(proposed, "proposed_policy_cid"),
            base_policy_cid=_cid(base, "base_policy_cid"),
            base_policy_version=_version(base_ver, "base_policy_version"),
            proposed_policy_version=_version(
                prop_ver, "proposed_policy_version"
            ),
            plan_cid=_optional_cid(plan_cid, "plan_cid"),
            cost_delta_basis_points=(
                None
                if cost is None
                else _nonneg_int(int(cost), "cost_delta_basis_points", maximum=MAX_COST_BP)
            ),
            coverage_declared=coverage_declared,
            coverage_partitions=partitions,
            draft_status=(
                str(data["draft_status"])
                if data.get("draft_status") is not None
                else None
            ),
            metadata=_mapping(data.get("metadata"), "metadata"),
        )

    # Duck-typed objects exposing candidate_cid / proposed_policy_cid.
    candidate_cid = getattr(remediation, "candidate_cid", None) or getattr(
        remediation, "plan_cid", None
    )
    proposed = promoted_policy_cid or getattr(
        remediation, "proposed_policy_cid", None
    ) or getattr(remediation, "promoted_policy_cid", None)
    base = base_policy_cid or getattr(
        remediation, "base_policy_cid", None
    ) or getattr(remediation, "previous_policy_cid", None)
    base_ver = base_policy_version or getattr(
        remediation, "base_policy_version", None
    ) or getattr(remediation, "previous_policy_version", None)
    prop_ver = promoted_policy_version or getattr(
        remediation, "proposed_policy_version", None
    ) or getattr(remediation, "promoted_policy_version", None)
    if candidate_cid is None or proposed is None or base is None or base_ver is None:
        raise PromotionError(
            "remediation must be GapRemediationPlan, mapping, or object with "
            "candidate/policy pins"
        )
    if prop_ver is None:
        prop_ver = f"{base_ver}+promoted"
    return NormalizedCandidate(
        candidate_cid=_cid(candidate_cid, "candidate_cid"),
        proposed_policy_cid=_cid(proposed, "proposed_policy_cid"),
        base_policy_cid=_cid(base, "base_policy_cid"),
        base_policy_version=_version(base_ver, "base_policy_version"),
        proposed_policy_version=_version(prop_ver, "proposed_policy_version"),
        plan_cid=_optional_cid(getattr(remediation, "plan_cid", None), "plan_cid"),
        cost_delta_basis_points=None,
        coverage_declared=False,
        coverage_partitions=(),
        draft_status=None,
        metadata=MappingProxyType({}),
    )


def _held_out_from_flags(
    *,
    held_out_killed: bool,
    verdict: str,
    disposition: str | None,
    qualified: bool,
    held_out_result: str | None,
) -> str:
    if held_out_result is not None:
        try:
            return HeldOutResult(held_out_result).value
        except ValueError as exc:
            raise PromotionError(
                f"held_out_result has unsupported value {held_out_result!r}"
            ) from exc
    if (
        held_out_killed
        and (
            verdict == EvaluationVerdict.QUALIFIED.value
            or qualified
            or disposition == QualificationDisposition.QUALIFIED.value
        )
    ):
        return HeldOutResult.PASSED.value
    if verdict in {
        EvaluationVerdict.REJECTED.value,
        EvaluationVerdict.REGRESSION.value,
        EvaluationVerdict.OVERFIT.value,
        EvaluationVerdict.SAFETY_WEAKENED.value,
        EvaluationVerdict.COST_EXCEEDED.value,
        EvaluationVerdict.OVERCONSTRAINT.value,
    }:
        return HeldOutResult.FAILED.value
    return HeldOutResult.INCONCLUSIVE.value


def _normalize_evaluation(evaluation: Any) -> NormalizedEvaluation:
    if evaluation is None:
        raise PromotionError("evaluation is required")

    if isinstance(evaluation, RemediationEvaluationReport):
        report = evaluation
        verdict = str(report.verdict)
        held_out_killed = bool(report.held_out_killed)
        regression = bool(report.regression_detected)
        meta = dict(report.metadata)
        vacuity = bool(
            meta.get("vacuity_detected")
            or meta.get("new_vacuity_detected")
            or meta.get("vacuity_introduced")
        )
        cost = report.cost_delta_basis_points
        partitions = tuple(
            str(item.partition)
            if hasattr(item, "partition")
            else str(item.get("partition"))
            for item in report.partition_evidence
        )
        return NormalizedEvaluation(
            evaluation_report_cid=report.report_cid,
            verdict=verdict,
            held_out_result=_held_out_from_flags(
                held_out_killed=held_out_killed,
                verdict=verdict,
                disposition=None,
                qualified=verdict == EvaluationVerdict.QUALIFIED.value,
                held_out_result=meta.get("held_out_result"),
            ),
            held_out_killed=held_out_killed,
            regression_detected=regression,
            vacuity_detected=vacuity,
            cost_delta_basis_points=int(cost),
            cost_declared=True,
            coverage_declared=bool(partitions),
            coverage_partitions=partitions,
            qualification_cid=None,
            disposition=None,
            qualified=verdict == EvaluationVerdict.QUALIFIED.value,
            metadata=MappingProxyType(meta),
        )

    if isinstance(evaluation, RemediationQualificationResult):
        qual = evaluation
        verdict = str(qual.verdict)
        disposition = str(qual.disposition)
        meta = dict(qual.metadata)
        vacuity = bool(
            meta.get("vacuity_detected")
            or meta.get("new_vacuity_detected")
            or meta.get("vacuity_introduced")
        )
        return NormalizedEvaluation(
            evaluation_report_cid=qual.evaluation_report_cid,
            verdict=verdict,
            held_out_result=_held_out_from_flags(
                held_out_killed=bool(qual.held_out_killed),
                verdict=verdict,
                disposition=disposition,
                qualified=disposition == QualificationDisposition.QUALIFIED.value,
                held_out_result=meta.get("held_out_result"),
            ),
            held_out_killed=bool(qual.held_out_killed),
            regression_detected=bool(qual.regression_detected),
            vacuity_detected=vacuity,
            cost_delta_basis_points=int(qual.cost_delta_basis_points),
            cost_declared=True,
            coverage_declared=bool(qual.required_partitions_present),
            coverage_partitions=tuple(qual.missing_partitions)  # empty when ok
            if not qual.required_partitions_present
            else ("required_partitions",),
            qualification_cid=qual.result_cid,
            disposition=disposition,
            qualified=disposition == QualificationDisposition.QUALIFIED.value,
            metadata=MappingProxyType(meta),
        )

    # RemediationEvaluationRun or duck-typed / mapping form.
    if not isinstance(evaluation, Mapping):
        # Attribute access for run objects.
        report = getattr(evaluation, "evaluation_report", None)
        report_cid = getattr(evaluation, "evaluation_report_cid", None)
        if report is not None and isinstance(report, RemediationEvaluationReport):
            base = _normalize_evaluation(report)
            qual = getattr(evaluation, "qualification", None)
            qual_cid = getattr(evaluation, "qualification_cid", None)
            partitions = tuple(
                str(item)
                for item in (getattr(evaluation, "partitions_covered", ()) or ())
                if isinstance(item, str) and item
            )
            qualified = bool(getattr(evaluation, "qualified", base.qualified))
            disposition = getattr(evaluation, "disposition", base.disposition)
            meta = dict(base.metadata)
            meta.update(dict(getattr(evaluation, "metadata", {}) or {}))
            vacuity = bool(
                meta.get("vacuity_detected")
                or meta.get("new_vacuity_detected")
                or getattr(evaluation, "vacuity_detected", False)
            )
            cost = base.cost_delta_basis_points
            if cost is None and "cost_delta_basis_points" in meta:
                cost = int(meta["cost_delta_basis_points"])
            return NormalizedEvaluation(
                evaluation_report_cid=base.evaluation_report_cid
                if report_cid is None
                else _cid(report_cid, "evaluation_report_cid"),
                verdict=str(getattr(evaluation, "verdict", base.verdict)),
                held_out_result=_held_out_from_flags(
                    held_out_killed=base.held_out_killed,
                    verdict=str(getattr(evaluation, "verdict", base.verdict)),
                    disposition=str(disposition) if disposition else None,
                    qualified=qualified,
                    held_out_result=meta.get("held_out_result"),
                ),
                held_out_killed=base.held_out_killed,
                regression_detected=base.regression_detected,
                vacuity_detected=vacuity,
                cost_delta_basis_points=cost,
                cost_declared=base.cost_declared or cost is not None,
                coverage_declared=base.coverage_declared or bool(partitions),
                coverage_partitions=partitions or base.coverage_partitions,
                qualification_cid=(
                    None
                    if qual_cid is None and qual is None
                    else _cid(
                        qual_cid
                        if qual_cid is not None
                        else getattr(qual, "result_cid"),
                        "qualification_cid",
                    )
                ),
                disposition=str(disposition) if disposition else None,
                qualified=qualified,
                metadata=MappingProxyType(meta),
            )
        # Fall through to attribute mapping projection.
        evaluation = {
            "evaluation_report_cid": report_cid
            or getattr(evaluation, "report_cid", None),
            "verdict": getattr(evaluation, "verdict", None),
            "held_out_killed": getattr(evaluation, "held_out_killed", None),
            "held_out_result": getattr(evaluation, "held_out_result", None),
            "regression_detected": getattr(
                evaluation, "regression_detected", False
            ),
            "vacuity_detected": getattr(evaluation, "vacuity_detected", False),
            "cost_delta_basis_points": getattr(
                evaluation, "cost_delta_basis_points", None
            ),
            "coverage_declared": getattr(evaluation, "coverage_declared", False),
            "coverage_partitions": getattr(
                evaluation, "partitions_covered", ()
            )
            or getattr(evaluation, "coverage_partitions", ()),
            "qualification_cid": getattr(evaluation, "qualification_cid", None),
            "disposition": getattr(evaluation, "disposition", None),
            "qualified": getattr(evaluation, "qualified", False),
            "metadata": getattr(evaluation, "metadata", {}),
        }

    if isinstance(evaluation, Mapping):
        data = dict(evaluation)
        report_cid = (
            data.get("evaluation_report_cid")
            or data.get("report_cid")
            or data.get("evaluation_cid")
        )
        if report_cid is None:
            raise PromotionError(
                "evaluation mapping must include evaluation_report_cid"
            )
        verdict = data.get("verdict") or EvaluationVerdict.REJECTED.value
        try:
            verdict = EvaluationVerdict(verdict).value
        except ValueError:
            verdict = _token(str(verdict), "verdict")
        held_out_killed = bool(data.get("held_out_killed", False))
        if "held_out_killed" not in data and data.get("held_out_result") == (
            HeldOutResult.PASSED.value
        ):
            held_out_killed = True
        regression = bool(data.get("regression_detected", False))
        vacuity = bool(
            data.get("vacuity_detected")
            or data.get("new_vacuity_detected")
            or data.get("vacuity_introduced")
            or False
        )
        cost_present = (
            "cost_delta_basis_points" in data
            or "declared_cost_delta_bp" in data
            or "cost_declared" in data
        )
        cost_raw = data.get(
            "cost_delta_basis_points", data.get("declared_cost_delta_bp")
        )
        cost = (
            None
            if cost_raw is None and not data.get("cost_declared")
            else _nonneg_int(
                0 if cost_raw is None else int(cost_raw),
                "cost_delta_basis_points",
                maximum=MAX_COST_BP,
            )
        )
        partitions = tuple(
            str(item)
            for item in (
                data.get("coverage_partitions")
                or data.get("partitions_covered")
                or ()
            )
            if isinstance(item, str) and item
        )
        coverage_declared = bool(
            data.get("coverage_declared")
            or data.get("coverage_impact_declared")
            or data.get("required_partitions_present")
            or partitions
        )
        disposition = data.get("disposition")
        qualified = bool(
            data.get("qualified")
            or disposition == QualificationDisposition.QUALIFIED.value
            or verdict == EvaluationVerdict.QUALIFIED.value
        )
        meta = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
        return NormalizedEvaluation(
            evaluation_report_cid=_cid(report_cid, "evaluation_report_cid"),
            verdict=verdict,
            held_out_result=_held_out_from_flags(
                held_out_killed=held_out_killed,
                verdict=verdict,
                disposition=str(disposition) if disposition else None,
                qualified=qualified,
                held_out_result=(
                    str(data["held_out_result"])
                    if data.get("held_out_result") is not None
                    else None
                ),
            ),
            held_out_killed=held_out_killed,
            regression_detected=regression,
            vacuity_detected=vacuity,
            cost_delta_basis_points=cost,
            cost_declared=cost_present or cost is not None,
            coverage_declared=coverage_declared,
            coverage_partitions=partitions,
            qualification_cid=_optional_cid(
                data.get("qualification_cid") or data.get("result_cid"),
                "qualification_cid",
            ),
            disposition=str(disposition) if disposition is not None else None,
            qualified=qualified,
            max_cost_delta_bp=_nonneg_int(
                int(data.get("max_cost_delta_bp", MAX_COST_BP)),
                "max_cost_delta_bp",
                maximum=MAX_COST_BP,
            )
            if data.get("max_cost_delta_bp") is not None
            else MAX_COST_BP,
            metadata=_mapping(meta, "metadata"),
        )

    raise PromotionError(
        "evaluation must be RemediationEvaluationReport, "
        "RemediationQualificationResult, RemediationEvaluationRun, or mapping"
    )


def _normalize_campaign_receipt(
    value: AssuranceCampaignReceipt | Mapping[str, Any] | None,
) -> AssuranceCampaignReceipt | None:
    if value is None:
        return None
    if isinstance(value, AssuranceCampaignReceipt):
        return value
    if isinstance(value, Mapping):
        try:
            return AssuranceCampaignReceipt.from_dict(value)
        except ReceiptContractError as exc:
            raise PromotionError(f"invalid campaign receipt: {exc}") from exc
    raise PromotionError(
        "campaign_receipt must be AssuranceCampaignReceipt or mapping"
    )


def _normalize_signature(
    value: ReceiptSignatureBinding | Mapping[str, Any] | None,
    *,
    name: str = "promotion_signature",
) -> ReceiptSignatureBinding | None:
    if value is None:
        return None
    if isinstance(value, ReceiptSignatureBinding):
        return value
    if isinstance(value, Mapping):
        try:
            return ReceiptSignatureBinding.from_dict(value)
        except ReceiptContractError as exc:
            raise PromotionError(f"invalid {name}: {exc}") from exc
    raise PromotionError(
        f"{name} must be ReceiptSignatureBinding or mapping"
    )


def _cas_result_projection(result: Any | None) -> dict[str, Any] | None:
    if result is None:
        return None
    status = _status_value(result.status)
    before = result.before
    after = result.after
    payload: dict[str, Any] = {
        "status": status,
        "reason_code": str(result.reason_code),
        "operation_id": str(result.operation_id),
        "transition_cid": result.transition_cid,
        "local_durable": bool(result.local_durable),
        "replicated": bool(result.replicated),
        "before": {
            "policy_cid": getattr(before, "policy_cid", None),
            "promotion_cid": getattr(before, "promotion_cid", None),
            "generation": int(before.generation),
            "transition_cid": before.transition_cid,
            "namespace": str(before.namespace),
        },
        "after": {
            "policy_cid": getattr(after, "policy_cid", None),
            "promotion_cid": getattr(after, "promotion_cid", None),
            "generation": int(after.generation),
            "transition_cid": after.transition_cid,
            "namespace": str(after.namespace),
        },
    }
    for attr in (
        "candidate_cid",
        "evaluation_cid",
        "authorization_cid",
        "expected_old_policy_generation",
        "expected_old_policy_cid",
    ):
        if hasattr(result, attr):
            payload[attr] = getattr(result, attr)
    return payload


def _stable_receipt_id(*parts: str) -> str:
    digest = cid_for_structured(
        {
            "interface_id": PROMOTE_ASSURANCE_POLICY_INTERFACE,
            "parts": list(parts),
        }
    )
    suffix = re.sub(r"[^a-z0-9]", "", digest.lower())[-24:] or "0"
    return f"promo_{suffix}"


def _build_receipt_header(
    *,
    candidate_cid: str,
    evaluation_report_cid: str,
    authorization_cid: str,
    previous_policy_cid: str,
    promoted_policy_cid: str,
    campaign_receipt_cid: str,
    repository_state_cid: str | None,
    repository_id: str,
    terminal_status: AssuranceTerminalStatus | str,
) -> AssuranceArtifactHeader:
    repo_state = repository_state_cid or previous_policy_cid
    return AssuranceArtifactHeader(
        artifact_kind="assurance_policy_promotion_receipt",
        repository_id=repository_id,
        repository_state_cid=repo_state,
        target_symbol_ids=("assurance.policy.promotion",),
        target_artifact_cids=(promoted_policy_cid,),
        capsule_cids=(candidate_cid,),
        proof_unit_cids=(evaluation_report_cid,),
        environment_cid=repo_state,
        dependency_lock_cid=previous_policy_cid,
        versions=VersionBinding(
            operator_id="assurance_policy_promoter",
            operator_version=GENERATOR_VERSION,
            campaign_policy_id="assurance_policy",
            campaign_policy_version=GENERATOR_VERSION,
            generator=GeneratorIdentity(
                generator_id=GENERATOR_ID,
                generator_version=GENERATOR_VERSION,
                interface_id=PROMOTE_ASSURANCE_POLICY_INTERFACE,
            ),
        ),
        provenance=ArtifactProvenance(
            producer_id=PRODUCER_ID,
            producer_version=PRODUCER_VERSION,
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.RECEIPT,
            input_cids=tuple(
                sorted(
                    {
                        candidate_cid,
                        evaluation_report_cid,
                        authorization_cid,
                        previous_policy_cid,
                        promoted_policy_cid,
                        campaign_receipt_cid,
                    }
                )
            ),
            tool_ids=(TOOL_ID,),
            policy_cid=previous_policy_cid,
            notes=None,
        ),
        terminal_status=terminal_status,
        receipt_cids=(campaign_receipt_cid,),
        proof_cids=(),
        metadata={"evidence": AAE_PROMOTION_EVIDENCE},
    )


def _collect_self_promotion_forbidden_cids(
    *,
    candidate: NormalizedCandidate,
    evaluation: NormalizedEvaluation,
    campaign_receipt: AssuranceCampaignReceipt | None,
    seal_evidence_cid: str | None,
    promoted_policy_cid: str,
) -> set[str]:
    forbidden: set[str] = {
        candidate.candidate_cid,
        candidate.proposed_policy_cid,
        candidate.base_policy_cid,
        evaluation.evaluation_report_cid,
        promoted_policy_cid,
    }
    if candidate.plan_cid is not None:
        forbidden.add(candidate.plan_cid)
    if evaluation.qualification_cid is not None:
        forbidden.add(evaluation.qualification_cid)
    if seal_evidence_cid is not None:
        forbidden.add(seal_evidence_cid)
    if campaign_receipt is not None:
        forbidden.add(campaign_receipt.receipt_cid)
        forbidden.add(campaign_receipt.authorization_cid)
        if campaign_receipt.held_out_evaluation_cid is not None:
            forbidden.add(campaign_receipt.held_out_evaluation_cid)
        if campaign_receipt.seal_evidence_cid is not None:
            forbidden.add(campaign_receipt.seal_evidence_cid)
    return {cid for cid in forbidden if cid}


def verify_receipt_signature_bindings(
    signature: ReceiptSignatureBinding | Mapping[str, Any],
    *,
    expected_action: str | ReceiptAction | None = None,
    require_verified: bool = True,
    name: str = "signature",
) -> ReceiptSignatureBinding:
    """Verify signer/key/audience/action bindings on a receipt signature.

    Fail-closed when bindings are incomplete or verification status is not
    ``verified`` (when required). Does not perform host cryptography.
    """

    binding = _normalize_signature(signature, name=name)
    if binding is None:
        raise PromotionError(f"{name} is required")
    if not binding.signer_identity:
        raise PromotionError(f"{name}.signer_identity is required")
    if not binding.key_identity:
        raise PromotionError(f"{name}.key_identity is required")
    if not binding.audience:
        raise PromotionError(f"{name}.audience is required")
    if not binding.action:
        raise PromotionError(f"{name}.action is required")
    if require_verified:
        if (
            binding.signature_verification_status
            != SignatureVerificationStatus.VERIFIED.value
        ):
            raise PromotionError(
                f"{name} requires signature_verification_status=verified"
            )
        if not binding.signature:
            raise PromotionError(f"{name} requires nonempty signature bytes")
    if expected_action is not None:
        expected = (
            expected_action.value
            if isinstance(expected_action, ReceiptAction)
            else str(expected_action)
        )
        if binding.action != expected:
            raise PromotionError(
                f"{name}.action must be {expected!r}, got {binding.action!r}"
            )
    return binding


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AssurancePolicyPromotionResult:
    """Closed outcome of an authorized assurance-policy promotion attempt.

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
    campaign_receipt_cid: str | None
    seal_evidence_cid: str | None
    expected_generation: int | None
    expected_policy_cid: str | None
    promoted_policy_cid: str | None
    held_out_result: str | None
    receipt: AssurancePolicyPromotionReceipt | None
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
            "campaign_receipt_cid",
            "seal_evidence_cid",
            "expected_generation",
            "expected_policy_cid",
            "promoted_policy_cid",
            "held_out_result",
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
            self, "operation_id", _operation_id(self.operation_id)
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
            "campaign_receipt_cid",
            _optional_cid(self.campaign_receipt_cid, "campaign_receipt_cid"),
        )
        object.__setattr__(
            self,
            "seal_evidence_cid",
            _optional_cid(self.seal_evidence_cid, "seal_evidence_cid"),
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
        if self.held_out_result is not None:
            try:
                object.__setattr__(
                    self,
                    "held_out_result",
                    HeldOutResult(self.held_out_result).value,
                )
            except ValueError as exc:
                raise PromotionError(
                    f"held_out_result has unsupported value {self.held_out_result!r}"
                ) from exc
        if self.receipt is not None and not isinstance(
            self.receipt, AssurancePolicyPromotionReceipt
        ):
            raise PromotionError(
                "receipt must be AssurancePolicyPromotionReceipt"
            )
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

        if self.head_mutated and status != PromotionStatus.PROMOTED.value:
            raise PromotionError("head_mutated requires status=promoted")
        if status == PromotionStatus.PROMOTED.value and not self.head_mutated:
            raise PromotionError("status=promoted requires head_mutated=True")
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
            "evidence": AAE_PROMOTION_EVIDENCE,
            "status": self.status,
            "head_mutated": self.head_mutated,
            "blocking_reasons": list(self.blocking_reasons),
            "workspace": self.workspace,
            "operation_id": self.operation_id,
            "candidate_cid": self.candidate_cid,
            "evaluation_report_cid": self.evaluation_report_cid,
            "authorization_cid": self.authorization_cid,
            "campaign_receipt_cid": self.campaign_receipt_cid,
            "seal_evidence_cid": self.seal_evidence_cid,
            "expected_generation": self.expected_generation,
            "expected_policy_cid": self.expected_policy_cid,
            "promoted_policy_cid": self.promoted_policy_cid,
            "held_out_result": self.held_out_result,
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


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluate_promotion_gates(
    candidate: NormalizedCandidate | Mapping[str, Any],
    evaluation: NormalizedEvaluation | Mapping[str, Any],
    authorization_cid: str | None,
    *,
    campaign_receipt: AssuranceCampaignReceipt | Mapping[str, Any] | None,
    seal_status: str | SealAvailabilityStatus | None,
    seal_evidence_cid: str | None,
    current_policy_cid: str | None,
    current_generation: int,
    expected_generation: int | None = None,
    expected_policy_cid: str | None = None,
    promotion_signature: ReceiptSignatureBinding | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return stable ordered blocking reason codes (empty iff promotable)."""

    reasons: list[str] = []

    if isinstance(candidate, Mapping):
        candidate = _normalize_candidate(candidate)
    if isinstance(evaluation, Mapping):
        evaluation = _normalize_evaluation(evaluation)
    if not isinstance(candidate, NormalizedCandidate):
        raise PromotionError("candidate must be NormalizedCandidate or mapping")
    if not isinstance(evaluation, NormalizedEvaluation):
        raise PromotionError("evaluation must be NormalizedEvaluation or mapping")

    campaign = _normalize_campaign_receipt(campaign_receipt)
    seal_status_value: str | None
    if seal_status is None:
        seal_status_value = None
    elif isinstance(seal_status, SealAvailabilityStatus):
        seal_status_value = seal_status.value
    else:
        try:
            seal_status_value = SealAvailabilityStatus(str(seal_status)).value
        except ValueError:
            reasons.append(REASON_SEAL_UNAVAILABLE)
            seal_status_value = str(seal_status)

    # --- canonical candidate ---
    if not candidate.candidate_cid:
        reasons.append(REASON_ABSENT_CANDIDATE)
    if not candidate.proposed_policy_cid:
        reasons.append(REASON_MISSING_PROMOTED_POLICY)

    # --- held-out pass ---
    if evaluation.held_out_result != HeldOutResult.PASSED.value:
        reasons.append(REASON_HELD_OUT_NOT_PASS)
    if not evaluation.held_out_killed and evaluation.held_out_result != (
        HeldOutResult.PASSED.value
    ):
        if REASON_HELD_OUT_NOT_PASS not in reasons:
            reasons.append(REASON_HELD_OUT_NOT_PASS)
    if evaluation.verdict not in {
        EvaluationVerdict.QUALIFIED.value,
        "pass",
        "passed",
    } and not evaluation.qualified:
        reasons.append(REASON_EVALUATION_NOT_PASS)

    # --- regression / vacuity ---
    if evaluation.regression_detected:
        reasons.append(REASON_REGRESSION_DETECTED)
    if evaluation.vacuity_detected:
        reasons.append(REASON_VACUITY_DETECTED)

    # --- declared cost / coverage ---
    cost_declared = evaluation.cost_declared or (
        candidate.cost_delta_basis_points is not None
    )
    if not cost_declared:
        reasons.append(REASON_COST_NOT_DECLARED)
    else:
        cost_bp = (
            evaluation.cost_delta_basis_points
            if evaluation.cost_delta_basis_points is not None
            else candidate.cost_delta_basis_points
        )
        if cost_bp is not None and cost_bp > evaluation.max_cost_delta_bp:
            reasons.append(REASON_COST_EXCEEDED)

    coverage_declared = evaluation.coverage_declared or candidate.coverage_declared
    if not coverage_declared:
        reasons.append(REASON_COVERAGE_NOT_DECLARED)

    # --- authorization / self-promotion ---
    if authorization_cid is None:
        reasons.append(REASON_ABSENT_AUTHORIZATION)
    else:
        forbidden = _collect_self_promotion_forbidden_cids(
            candidate=candidate,
            evaluation=evaluation,
            campaign_receipt=campaign,
            seal_evidence_cid=seal_evidence_cid,
            promoted_policy_cid=candidate.proposed_policy_cid,
        )
        if authorization_cid in forbidden:
            reasons.append(REASON_SELF_PROMOTION)

    # --- campaign receipt + signature bindings ---
    if campaign is None:
        reasons.append(REASON_ABSENT_CAMPAIGN_RECEIPT)
    else:
        try:
            verify_campaign_receipt_identity(campaign)
            verify_receipt_signature_bindings(
                campaign.signature,
                expected_action=None,  # campaign may use complete_campaign
                require_verified=True,
                name="campaign_receipt.signature",
            )
            require_verified_signature_before_persistence(campaign)
        except (PromotionError, ReceiptContractError):
            reasons.append(REASON_UNVERIFIED_CAMPAIGN_RECEIPT)
            if REASON_INVALID_SIGNATURE_BINDINGS not in reasons:
                reasons.append(REASON_INVALID_SIGNATURE_BINDINGS)

    # --- promotion signature bindings (pre-receipt) ---
    if promotion_signature is not None:
        try:
            verify_receipt_signature_bindings(
                promotion_signature,
                expected_action=ReceiptAction.PROMOTE_POLICY,
                require_verified=True,
                name="promotion_signature",
            )
        except PromotionError:
            reasons.append(REASON_INVALID_SIGNATURE_BINDINGS)
            if REASON_UNVERIFIED_PROMOTION_RECEIPT not in reasons:
                reasons.append(REASON_UNVERIFIED_PROMOTION_RECEIPT)
    else:
        reasons.append(REASON_UNVERIFIED_PROMOTION_RECEIPT)

    # --- released incremental seal ---
    if seal_status_value != SealAvailabilityStatus.RELEASED.value:
        reasons.append(REASON_SEAL_UNAVAILABLE)
    if seal_evidence_cid is None:
        reasons.append(REASON_ABSENT_SEAL)
    if (
        seal_status_value == SealAvailabilityStatus.UNAVAILABLE.value
        and REASON_SEAL_UNAVAILABLE not in reasons
    ):
        reasons.append(REASON_SEAL_UNAVAILABLE)

    # --- stale candidate / expected-old head ---
    current_generation = _nonneg_int(current_generation, "current_generation")
    if current_policy_cid is not None:
        current_policy_cid = _cid(current_policy_cid, "current_policy_cid")

    if candidate.base_policy_cid != current_policy_cid:
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

    if (
        candidate.proposed_policy_cid == candidate.base_policy_cid
        and REASON_PROMOTED_POLICY_MISMATCH not in reasons
    ):
        reasons.append(REASON_PROMOTED_POLICY_MISMATCH)

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


def promote_assurance_policy(
    remediation: Any,
    evaluation: Any,
    authorization: str | Mapping[str, Any] | None,
    *,
    campaign_receipt: AssuranceCampaignReceipt | Mapping[str, Any],
    policy_repository: PolicyRepository | None,
    operation_id: str,
    promotion_signature: ReceiptSignatureBinding | Mapping[str, Any],
    seal_evidence_cid: str,
    seal_status: str | SealAvailabilityStatus = SealAvailabilityStatus.RELEASED,
    workspace: str = "default",
    expected_generation: int | None = None,
    expected_policy_cid: str | None = None,
    promoted_policy_cid: str | None = None,
    promoted_policy_version: str | None = None,
    base_policy_cid: str | None = None,
    base_policy_version: str | None = None,
    promotion_repository: PromotionStateRepository | None = None,
    repository_state_cid: str | None = None,
    repository_id: str = "repository:sha256:adversarial-assurance-promotion",
    seal_scope: Sequence[str] | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AssurancePolicyPromotionResult:
    """Authorize and CAS-publish an assurance-policy successor.

    Revalidates canonical candidate identity, held-out pass, regression/vacuity
    absence, declared cost/coverage, external authorization, verified
    signer/key/audience/action bindings on campaign and promotion receipts,
    released incremental seal evidence, and expected-old policy revision CAS
    at publication time. Stale, unauthorized, mismatched, or conflicting
    attempts leave the live head unchanged (``head_mutated=False``).

    Candidates cannot self-promote.
    """

    workspace = _token(workspace, "workspace")
    operation_id = _operation_id(operation_id)
    meta = _mapping(metadata, "metadata")
    notes = _optional_text(notes, "notes")

    if policy_repository is None:
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_MISSING_REPOSITORY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=None,
            evaluation_report_cid=None,
            authorization_cid=None,
            campaign_receipt_cid=None,
            seal_evidence_cid=_optional_cid(seal_evidence_cid, "seal_evidence_cid")
            if seal_evidence_cid
            else None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            promoted_policy_cid=None,
            held_out_result=None,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic="policy_repository is required",
            metadata=meta,
        )

    try:
        cand = _normalize_candidate(
            remediation,
            promoted_policy_cid=promoted_policy_cid,
            promoted_policy_version=promoted_policy_version,
            base_policy_cid=base_policy_cid,
            base_policy_version=base_policy_version,
        )
        evaluation_norm = _normalize_evaluation(evaluation)
        auth_cid = _extract_authorization_cid(authorization)
        campaign = _normalize_campaign_receipt(campaign_receipt)
        if campaign is None:
            raise PromotionError("campaign_receipt is required")
        promo_sig = _normalize_signature(
            promotion_signature, name="promotion_signature"
        )
        if promo_sig is None:
            raise PromotionError("promotion_signature is required")
        seal_cid = _cid(seal_evidence_cid, "seal_evidence_cid")
        current = policy_repository.current_policy(workspace)
        current_cid = current.policy_cid
        current_gen = int(current.generation)
    except PromotionError as exc:
        return AssurancePolicyPromotionResult(
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
            campaign_receipt_cid=None,
            seal_evidence_cid=None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            promoted_policy_cid=None,
            held_out_result=None,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic=_clip(str(exc)),
            metadata=meta,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=None,
            evaluation_report_cid=None,
            authorization_cid=None,
            campaign_receipt_cid=None,
            seal_evidence_cid=None,
            expected_generation=expected_generation,
            expected_policy_cid=expected_policy_cid,
            promoted_policy_cid=None,
            held_out_result=None,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic=_clip(str(exc)),
            metadata=meta,
        )

    exp_gen = current_gen if expected_generation is None else expected_generation
    exp_cid = current_cid if expected_policy_cid is None else expected_policy_cid

    # Already at the proposed successor: idempotent no-op (no second mutation).
    if current_cid is not None and current_cid == cand.proposed_policy_cid:
        early_block: list[str] = []
        if auth_cid is None:
            early_block.append(REASON_ABSENT_AUTHORIZATION)
        else:
            forbidden = _collect_self_promotion_forbidden_cids(
                candidate=cand,
                evaluation=evaluation_norm,
                campaign_receipt=campaign,
                seal_evidence_cid=seal_cid,
                promoted_policy_cid=cand.proposed_policy_cid,
            )
            if auth_cid in forbidden:
                early_block.append(REASON_SELF_PROMOTION)
        if not early_block:
            return AssurancePolicyPromotionResult(
                status=PromotionStatus.UNCHANGED.value,
                head_mutated=False,
                blocking_reasons=(),
                workspace=workspace,
                operation_id=operation_id,
                candidate_cid=cand.candidate_cid,
                evaluation_report_cid=evaluation_norm.evaluation_report_cid,
                authorization_cid=auth_cid,
                campaign_receipt_cid=campaign.receipt_cid,
                seal_evidence_cid=seal_cid,
                expected_generation=exp_gen,
                expected_policy_cid=exp_cid,
                promoted_policy_cid=cand.proposed_policy_cid,
                held_out_result=evaluation_norm.held_out_result,
                receipt=None,
                policy_cas=None,
                promotion_cas=None,
                diagnostic="policy head already at proposed_policy_cid",
                metadata=meta,
            )

    blocking = evaluate_promotion_gates(
        cand,
        evaluation_norm,
        auth_cid,
        campaign_receipt=campaign,
        seal_status=seal_status,
        seal_evidence_cid=seal_cid,
        current_policy_cid=current_cid,
        current_generation=current_gen,
        expected_generation=exp_gen,
        expected_policy_cid=exp_cid,
        promotion_signature=promo_sig,
    )

    if blocking:
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=blocking,
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
            campaign_receipt_cid=campaign.receipt_cid,
            seal_evidence_cid=seal_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            held_out_result=evaluation_norm.held_out_result,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic="promotion gates failed; policy head not mutated",
            metadata=meta,
        )

    # Gates clear — auth_cid, campaign, promo_sig, seal_cid are non-None.
    assert auth_cid is not None
    assert campaign is not None
    assert promo_sig is not None

    scope = tuple(seal_scope) if seal_scope else REQUIRED_PROMOTION_SEAL_SCOPE

    try:
        # Re-verify campaign bindings immediately before receipt construction.
        verify_campaign_receipt_identity(campaign)
        verify_receipt_signature_bindings(
            campaign.signature,
            require_verified=True,
            name="campaign_receipt.signature",
        )
        require_verified_signature_before_persistence(campaign)
        verify_receipt_signature_bindings(
            promo_sig,
            expected_action=ReceiptAction.PROMOTE_POLICY,
            require_verified=True,
            name="promotion_signature",
        )

        receipt = AssurancePolicyPromotionReceipt(
            header=_build_receipt_header(
                candidate_cid=cand.candidate_cid,
                evaluation_report_cid=evaluation_norm.evaluation_report_cid,
                authorization_cid=auth_cid,
                previous_policy_cid=cand.base_policy_cid,
                promoted_policy_cid=cand.proposed_policy_cid,
                campaign_receipt_cid=campaign.receipt_cid,
                repository_state_cid=repository_state_cid,
                repository_id=repository_id,
                terminal_status=AssuranceTerminalStatus.COMPLETE,
            ),
            receipt_id=_stable_receipt_id(
                cand.candidate_cid,
                evaluation_norm.evaluation_report_cid,
                auth_cid,
                operation_id,
            ),
            campaign_receipt_cid=campaign.receipt_cid,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            held_out_evaluation_cid=(
                campaign.held_out_evaluation_cid
                or evaluation_norm.evaluation_report_cid
            ),
            held_out_result=evaluation_norm.held_out_result,
            authorization_cid=auth_cid,
            expected_old_policy_cid=cand.base_policy_cid,
            expected_old_policy_version=cand.base_policy_version,
            previous_policy_cid=cand.base_policy_cid,
            previous_policy_version=cand.base_policy_version,
            promoted_policy_cid=cand.proposed_policy_cid,
            promoted_policy_version=cand.proposed_policy_version,
            rollback_policy_cid=cand.base_policy_cid,
            cas_expected_version=cand.base_policy_version,
            seal_scope=scope,
            seal_status=SealAvailabilityStatus.RELEASED,
            seal_evidence_cid=seal_cid,
            signature=promo_sig,
            notes=notes,
            metadata={
                "evidence": AAE_PROMOTION_EVIDENCE,
                "operation_id": operation_id,
                "expected_generation": exp_gen,
                "qualification_cid": evaluation_norm.qualification_cid,
                "adapter_id": ADAPTER_ID,
                **dict(meta),
            },
        )
        verify_promotion_receipt_identity(receipt)
        require_verified_signature_before_persistence(receipt)
    except (PromotionError, ReceiptContractError) as exc:
        msg = str(exc)
        block: list[str] = [REASON_SCHEMA_INTEGRITY]
        if "self" in msg.lower() or "authoriz" in msg.lower():
            block.insert(0, REASON_SELF_PROMOTION)
        if "signature" in msg.lower() or "verified" in msg.lower():
            block.append(REASON_UNVERIFIED_PROMOTION_RECEIPT)
            block.append(REASON_INVALID_SIGNATURE_BINDINGS)
        seen_block: set[str] = set()
        unique_block_list: list[str] = []
        for item in block:
            if item not in seen_block:
                seen_block.add(item)
                unique_block_list.append(item)
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=tuple(unique_block_list),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
            campaign_receipt_cid=campaign.receipt_cid,
            seal_evidence_cid=seal_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            held_out_result=evaluation_norm.held_out_result,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic=_clip(msg),
            metadata=meta,
        )

    # Atomic policy CAS under promotion identity bindings — only mutation path.
    try:
        cas = policy_repository.promote_policy(
            workspace,
            expected_generation=int(exp_gen),
            expected_policy_cid=exp_cid,
            new_policy_cid=cand.proposed_policy_cid,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
        )
    except Exception as exc:
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
            campaign_receipt_cid=campaign.receipt_cid,
            seal_evidence_cid=seal_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            held_out_result=evaluation_norm.held_out_result,
            receipt=None,
            policy_cas=None,
            promotion_cas=None,
            diagnostic=_clip(str(exc)),
            metadata=meta,
        )

    cas_status = _status_value(cas.status)
    cas_proj = _cas_result_projection(cas)

    if cas_status == "conflict":
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.CONFLICT.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_CONFLICT,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
            campaign_receipt_cid=campaign.receipt_cid,
            seal_evidence_cid=seal_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            held_out_result=evaluation_norm.held_out_result,
            receipt=None,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=_clip(f"policy CAS conflict: {cas.reason_code}"),
            metadata=meta,
        )
    if cas_status == "unavailable":
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.UNAVAILABLE.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_UNAVAILABLE,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
            campaign_receipt_cid=campaign.receipt_cid,
            seal_evidence_cid=seal_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            held_out_result=evaluation_norm.held_out_result,
            receipt=None,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=_clip(f"policy CAS unavailable: {cas.reason_code}"),
            metadata=meta,
        )
    if cas_status == "corrupt":
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.CORRUPT.value,
            head_mutated=False,
            blocking_reasons=(REASON_CAS_CORRUPT,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
            campaign_receipt_cid=campaign.receipt_cid,
            seal_evidence_cid=seal_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            held_out_result=evaluation_norm.held_out_result,
            receipt=None,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=_clip(f"policy CAS corrupt: {cas.reason_code}"),
            metadata=meta,
        )
    if cas_status == "unchanged":
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.UNCHANGED.value,
            head_mutated=False,
            blocking_reasons=(),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
            campaign_receipt_cid=campaign.receipt_cid,
            seal_evidence_cid=seal_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            held_out_result=evaluation_norm.held_out_result,
            receipt=receipt,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic="idempotent CAS replay; policy head unchanged",
            metadata=meta,
        )
    if cas_status != "updated":
        return AssurancePolicyPromotionResult(
            status=PromotionStatus.REJECTED.value,
            head_mutated=False,
            blocking_reasons=(REASON_SCHEMA_INTEGRITY,),
            workspace=workspace,
            operation_id=operation_id,
            candidate_cid=cand.candidate_cid,
            evaluation_report_cid=evaluation_norm.evaluation_report_cid,
            authorization_cid=auth_cid,
            campaign_receipt_cid=campaign.receipt_cid,
            seal_evidence_cid=seal_cid,
            expected_generation=exp_gen,
            expected_policy_cid=exp_cid,
            promoted_policy_cid=cand.proposed_policy_cid,
            held_out_result=evaluation_norm.held_out_result,
            receipt=None,
            policy_cas=cas_proj,
            promotion_cas=None,
            diagnostic=_clip(f"unexpected CAS status: {cas_status}"),
            metadata=meta,
        )

    # Optional promotion-head CAS publishes the receipt identity.
    promo_cas_proj: dict[str, Any] | None = None
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
                evaluation_cid=evaluation_norm.evaluation_report_cid,
                authorization_cid=auth_cid,
                expected_old_policy_generation=int(exp_gen),
                expected_old_policy_cid=exp_cid,
            )
            promo_cas_proj = _cas_result_projection(promo_cas)
        except Exception as exc:
            # Policy head already advanced; surface promotion-head failure as
            # diagnostic only (policy CAS remains authoritative).
            promo_cas_proj = {
                "status": "unavailable",
                "reason_code": "promotion_head_cas_failed",
                "diagnostic": _clip(str(exc)),
            }

    return AssurancePolicyPromotionResult(
        status=PromotionStatus.PROMOTED.value,
        head_mutated=True,
        blocking_reasons=(),
        workspace=workspace,
        operation_id=operation_id,
        candidate_cid=cand.candidate_cid,
        evaluation_report_cid=evaluation_norm.evaluation_report_cid,
        authorization_cid=auth_cid,
        campaign_receipt_cid=campaign.receipt_cid,
        seal_evidence_cid=seal_cid,
        expected_generation=exp_gen,
        expected_policy_cid=exp_cid,
        promoted_policy_cid=cand.proposed_policy_cid,
        held_out_result=evaluation_norm.held_out_result,
        receipt=receipt,
        policy_cas=cas_proj,
        promotion_cas=promo_cas_proj,
        diagnostic=None,
        metadata=meta,
    )


def promote_assurance_policy_descriptor() -> Mapping[str, Any]:
    """Return a frozen public descriptor for ``promote_assurance_policy@1``."""

    return MappingProxyType(
        {
            "interface_id": PROMOTE_ASSURANCE_POLICY_INTERFACE,
            "result_interface_id": POLICY_PROMOTION_RESULT_INTERFACE,
            "result_schema": POLICY_PROMOTION_RESULT_SCHEMA,
            "evidence": AAE_PROMOTION_EVIDENCE,
            "adapter_id": ADAPTER_ID,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "mandatory_gates": (
                "canonical_candidate",
                "held_out_pass",
                "no_regression",
                "no_vacuity",
                "declared_cost",
                "declared_coverage",
                "authorization",
                "verified_campaign_receipt_signature_bindings",
                "verified_promotion_receipt_signature_bindings",
                "expected_old_cas",
                "released_incremental_seal",
                "no_self_promotion",
            ),
            "self_promotion_forbidden": True,
            "production_policy_change_during_fixture_campaign": False,
        }
    )


__all__ = [
    "AAE_PROMOTION_EVIDENCE",
    "ADAPTER_ID",
    "ASSURANCE_POLICY_PROMOTION_RECEIPT_INTERFACE",
    "AssurancePolicyPromotionResult",
    "DEFAULT_AUDIENCE",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "NormalizedCandidate",
    "NormalizedEvaluation",
    "POLICY_PROMOTION_RESULT_INTERFACE",
    "POLICY_PROMOTION_RESULT_SCHEMA",
    "PROMOTE_ASSURANCE_POLICY_INTERFACE",
    "PolicyRepository",
    "PromotionError",
    "PromotionStateRepository",
    "PromotionStatus",
    "REASON_ABSENT_AUTHORIZATION",
    "REASON_ABSENT_CAMPAIGN_RECEIPT",
    "REASON_ABSENT_CANDIDATE",
    "REASON_ABSENT_EVALUATION",
    "REASON_ABSENT_SEAL",
    "REASON_CAS_CONFLICT",
    "REASON_CAS_CORRUPT",
    "REASON_CAS_UNAVAILABLE",
    "REASON_COST_EXCEEDED",
    "REASON_COST_NOT_DECLARED",
    "REASON_COVERAGE_NOT_DECLARED",
    "REASON_EVALUATION_NOT_PASS",
    "REASON_HELD_OUT_NOT_PASS",
    "REASON_INVALID_SIGNATURE_BINDINGS",
    "REASON_MISSING_PROMOTED_POLICY",
    "REASON_MISSING_REPOSITORY",
    "REASON_POLICY_HEAD_MISMATCH",
    "REASON_PROMOTED_POLICY_MISMATCH",
    "REASON_REGRESSION_DETECTED",
    "REASON_SCHEMA_INTEGRITY",
    "REASON_SEAL_UNAVAILABLE",
    "REASON_SELF_PROMOTION",
    "REASON_STALE_CANDIDATE",
    "REASON_UNVERIFIED_CAMPAIGN_RECEIPT",
    "REASON_UNVERIFIED_PROMOTION_RECEIPT",
    "REASON_VACUITY_DETECTED",
    "REQUIRED_PROMOTION_SEAL_SCOPE",
    "evaluate_promotion_gates",
    "promote_assurance_policy",
    "promote_assurance_policy_descriptor",
    "verify_receipt_signature_bindings",
]
