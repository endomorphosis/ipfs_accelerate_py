"""Generate candidates and execute held-out remediation evaluation (AAE-046).

Interface surface:

* ``propose_gap_remediation@1`` — compose released requirement-grounded candidate
  generation (AAE-032) for a surviving mutant and assurance gap.
* ``evaluate_remediation@1`` — evaluate a remediation plan/proposal against a
  held-out campaign covering original (unmutated), diagnosis, development,
  held-out, unrelated, performance/cost, false-positive, overconstraint,
  regression, and safety partitions; reject one-mutant overfit and mock bypass.

Authority rules (normative):

* Pure and deterministic orchestration: no store, worktree, or production-policy
  mutation.
* Canonical identity comes only from ``software_contracts.content``.
* Reuses AAE-032 ``propose_gap_remediation`` and AAE-033
  ``qualify_remediation_evaluation`` / partition policy.
* Qualification fails closed when any required evaluation partition is missing,
  fails, exhibits one-mutant overfit, or records mock bypass.
* Model drafts cannot self-promote; held-out pass is mandatory for qualification.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.analysis_contracts import (
    AssuranceGap,
    SurvivingMutantReport,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    AssuranceArtifactHeader,
    AssuranceBaseError,
    GeneratorIdentity,
    VersionBinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.held_out import (
    DEFAULT_MAX_COST_DELTA_BP,
    HELD_OUT_POLICY_EVIDENCE,
    REQUIRED_EVALUATION_PARTITIONS,
    HeldOutPolicyError,
    MutantPartitionPlan,
    QualificationDisposition,
    RemediationQualificationResult,
    qualify_remediation_evaluation,
    required_evaluation_partitions,
    verify_remediation_qualification_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation import (
    PROPOSE_GAP_REMEDIATION_INTERFACE as DATASETS_PROPOSE_INTERFACE,
    GapRemediationProposal,
    RemediationError as SpecRemediationError,
    propose_gap_remediation as datasets_propose_gap_remediation,
    verify_gap_remediation_proposal_identity,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    CandidateDraftStatus,
    EvaluationPartition,
    EvaluationVerdict,
    GapRemediationPlan,
    PartitionEvaluationEvidence,
    RejectionReason,
    RemediationContractError,
    RemediationEvaluationReport,
    RemediationPlanStatus,
    verify_evaluation_report_identity,
    verify_plan_identity,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

PROPOSE_GAP_REMEDIATION_INTERFACE: Final[str] = "propose_gap_remediation@1"
EVALUATE_REMEDIATION_INTERFACE: Final[str] = "evaluate_remediation@1"

REMEDIATION_PROPOSAL_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "remediation-proposal-run@1"
)
REMEDIATION_PROPOSAL_RUN_INTERFACE: Final[str] = "RemediationProposalRun@1"
REMEDIATION_EVALUATION_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "remediation-evaluation-run@1"
)
REMEDIATION_EVALUATION_RUN_INTERFACE: Final[str] = "RemediationEvaluationRun@1"
HELD_OUT_CAMPAIGN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "held-out-campaign@1"
)
HELD_OUT_CAMPAIGN_INTERFACE: Final[str] = "HeldOutCampaign@1"
CAMPAIGN_PARTITION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "campaign-partition-result@1"
)

AAE_REMEDIATION_EVALUATION_EVIDENCE: Final[str] = "aae/remediation-evaluation@1"
ADAPTER_ID: Final[str] = "aae-remediation-evaluation"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
GENERATOR_ID: Final[str] = "remediation_evaluation"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_LIST: Final[int] = 1_024
MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_MUTANTS: Final[int] = 4_096
MAX_COST_BP: Final[int] = 1_000_000
MAX_REASON_CODES: Final[int] = 128

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

# Plan acceptance names "original" for the unmutated suite partition.
ORIGINAL_PARTITION: Final[str] = EvaluationPartition.UNMUTATED.value
# Plan acceptance names "performance" for performance/cost evidence.
PERFORMANCE_PARTITION: Final[str] = EvaluationPartition.PERFORMANCE_COST.value

# Partitions that AAE-046 acceptance requires (original + eight behavioral + safety).
# Regression is also encoded because the durable report contract requires it.
AAE046_EVALUATION_PARTITIONS: Final[tuple[str, ...]] = (
    EvaluationPartition.UNMUTATED.value,  # original
    EvaluationPartition.DIAGNOSIS.value,
    EvaluationPartition.DEVELOPMENT.value,
    EvaluationPartition.HELD_OUT.value,
    EvaluationPartition.UNRELATED.value,
    EvaluationPartition.PERFORMANCE_COST.value,  # performance
    EvaluationPartition.FALSE_POSITIVE.value,
    EvaluationPartition.OVERCONSTRAINT.value,
    EvaluationPartition.REGRESSION.value,
    EvaluationPartition.SAFETY.value,
)

# Partitions whose kill counts feed one-mutant overfit detection.
_RELATED_KILL_PARTITIONS: Final[frozenset[str]] = frozenset(
    {
        EvaluationPartition.DIAGNOSIS.value,
        EvaluationPartition.DEVELOPMENT.value,
        EvaluationPartition.HELD_OUT.value,
    }
)

REASON_CANDIDATES_PROPOSED: Final[str] = "candidates_proposed"
REASON_REQUIREMENT_GROUNDED: Final[str] = "requirement_grounded"
REASON_HEURISTIC_CANDIDATE: Final[str] = "heuristic_candidate_only"
REASON_HELD_OUT_REQUIRED: Final[str] = "held_out_evaluation_required"
REASON_EVALUATION_COVERED: Final[str] = "required_partitions_covered"
REASON_ORIGINAL_EVALUATED: Final[str] = "original_unmutated_evaluated"
REASON_DIAGNOSIS_EVALUATED: Final[str] = "diagnosis_evaluated"
REASON_DEVELOPMENT_EVALUATED: Final[str] = "development_evaluated"
REASON_HELD_OUT_EVALUATED: Final[str] = "held_out_evaluated"
REASON_UNRELATED_EVALUATED: Final[str] = "unrelated_evaluated"
REASON_PERFORMANCE_EVALUATED: Final[str] = "performance_evaluated"
REASON_FALSE_POSITIVE_EVALUATED: Final[str] = "false_positive_evaluated"
REASON_OVERCONSTRAINT_EVALUATED: Final[str] = "overconstraint_evaluated"
REASON_SAFETY_EVALUATED: Final[str] = "safety_evaluated"
REASON_QUALIFIED: Final[str] = "remediation_qualified"
REASON_REJECTED: Final[str] = "remediation_rejected"
REASON_ONE_MUTANT_OVERFIT: Final[str] = "one_mutant_overfit_rejected"
REASON_MOCK_BYPASS: Final[str] = "mock_bypass_rejected"
REASON_NO_PRODUCTION_POLICY_CHANGE: Final[str] = "production_policy_unchanged"
REASON_QUALIFICATION_APPLIED: Final[str] = "held_out_qualification_applied"


# ---------------------------------------------------------------------------
# Errors and phase vocabulary
# ---------------------------------------------------------------------------


class RemediationRuntimeError(AssuranceBaseError):
    """Raised when remediation evaluation inputs or orchestration fail closed."""

    def __init__(self, message: str, *, reason_code: str = "malformed_input") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class OneMutantOverfitError(RemediationRuntimeError):
    """Raised when a remediation only fits a single diagnosis mutant."""


class MockBypassError(RemediationRuntimeError):
    """Raised when evaluation evidence indicates mock/simulated bypass."""


class RemediationPhase(str, Enum):
    """Ordered phases recorded on a remediation evaluation run."""

    ADMIT = "admit"
    PROPOSE = "propose"
    PARTITION = "partition"
    EVALUATE = "evaluate"
    DETECT = "detect"
    REPORT = "report"
    QUALIFY = "qualify"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False, maximum: int = MAX_TEXT_CHARS) -> str:
    if type(value) is not str or (not empty and not value):
        raise RemediationRuntimeError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise RemediationRuntimeError(f"{name} must be trimmed NFC text")
    if len(value) > maximum or any(not char.isprintable() for char in value):
        raise RemediationRuntimeError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise RemediationRuntimeError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise RemediationRuntimeError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RemediationRuntimeError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise RemediationRuntimeError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise RemediationRuntimeError(f"{name} exceeds maximum {maximum}")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        if isinstance(value, enum_type):
            return value.value  # type: ignore[return-value]
        return enum_type(value).value  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise RemediationRuntimeError(
            f"{name} has unsupported value {value!r}"
        ) from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise RemediationRuntimeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _stable_unique(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _unique_sorted_tokens(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int = MAX_LIST,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise RemediationRuntimeError(f"{name} must be a list")
    ordered = tuple(sorted(_token(value, name) for value in values))
    if len(ordered) > maximum:
        raise RemediationRuntimeError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise RemediationRuntimeError(f"{name} must not contain duplicates")
    return ordered


def _unique_sorted_cids(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int = MAX_LIST,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise RemediationRuntimeError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(value, name) for value in values))
    if len(ordered) > maximum:
        raise RemediationRuntimeError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise RemediationRuntimeError(f"{name} must not contain duplicates")
    return ordered


def _normalize_partition_name(value: Any, name: str = "partition") -> str:
    """Normalize acceptance aliases onto closed EvaluationPartition tokens."""

    if isinstance(value, EvaluationPartition):
        return value.value
    text = _text(value, name)
    aliases = {
        "original": EvaluationPartition.UNMUTATED.value,
        "unmutated": EvaluationPartition.UNMUTATED.value,
        "performance": EvaluationPartition.PERFORMANCE_COST.value,
        "performance_cost": EvaluationPartition.PERFORMANCE_COST.value,
        "cost": EvaluationPartition.PERFORMANCE_COST.value,
        "false-positive": EvaluationPartition.FALSE_POSITIVE.value,
        "false_positive": EvaluationPartition.FALSE_POSITIVE.value,
        "held-out": EvaluationPartition.HELD_OUT.value,
        "held_out": EvaluationPartition.HELD_OUT.value,
    }
    normalized = aliases.get(text, text)
    return _enum(normalized, EvaluationPartition, name)


# ---------------------------------------------------------------------------
# Campaign partition results and held-out campaign
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CampaignPartitionResult:
    """Observed evaluation result for one required partition.

    Interface-bearing observation used by ``HeldOutCampaign@1``.
    """

    partition: EvaluationPartition | str
    passed: bool
    mutant_ids: Sequence[str] = ()
    killed_mutant_ids: Sequence[str] = ()
    killed_count: int | None = None
    survived_count: int | None = None
    mock_bypass: bool = False
    freezes_implementation: bool = False
    one_mutant_only: bool = False
    evidence_cids: Sequence[str] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "partition",
            _normalize_partition_name(self.partition, "partition"),
        )
        object.__setattr__(self, "passed", _bool(self.passed, "passed"))
        mutants = _unique_sorted_tokens(list(self.mutant_ids), "mutant_ids", maximum=MAX_MUTANTS)
        object.__setattr__(self, "mutant_ids", mutants)
        killed_ids = _unique_sorted_tokens(
            list(self.killed_mutant_ids),
            "killed_mutant_ids",
            maximum=MAX_MUTANTS,
        )
        if killed_ids and mutants and not set(killed_ids).issubset(set(mutants)):
            raise RemediationRuntimeError(
                "killed_mutant_ids must be a subset of mutant_ids",
                reason_code="killed_not_subset",
            )
        object.__setattr__(self, "killed_mutant_ids", killed_ids)

        if self.killed_count is None:
            resolved_killed = len(killed_ids)
        else:
            resolved_killed = _nonneg_int(self.killed_count, "killed_count")
            if killed_ids and resolved_killed != len(killed_ids):
                raise RemediationRuntimeError(
                    "killed_count must match killed_mutant_ids length when both set",
                    reason_code="killed_count_mismatch",
                )
        object.__setattr__(self, "killed_count", resolved_killed)

        if self.survived_count is None:
            resolved_survived = max(0, len(mutants) - resolved_killed) if mutants else (
                0 if self.passed else 1
            )
        else:
            resolved_survived = _nonneg_int(self.survived_count, "survived_count")
        object.__setattr__(self, "survived_count", resolved_survived)

        object.__setattr__(self, "mock_bypass", _bool(self.mock_bypass, "mock_bypass"))
        object.__setattr__(
            self,
            "freezes_implementation",
            _bool(self.freezes_implementation, "freezes_implementation"),
        )
        object.__setattr__(
            self,
            "one_mutant_only",
            _bool(self.one_mutant_only, "one_mutant_only"),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _unique_sorted_cids(list(self.evidence_cids), "evidence_cids"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))

        # Mock bypass is never a passing observation.
        if self.mock_bypass and self.passed:
            raise RemediationRuntimeError(
                "mock_bypass cannot co-exist with passed=true",
                reason_code="mock_bypass_pass_conflict",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_PARTITION_RESULT_SCHEMA,
            "partition": self.partition,
            "passed": self.passed,
            "mutant_ids": list(self.mutant_ids),
            "killed_mutant_ids": list(self.killed_mutant_ids),
            "killed_count": self.killed_count,
            "survived_count": self.survived_count,
            "mock_bypass": self.mock_bypass,
            "freezes_implementation": self.freezes_implementation,
            "one_mutant_only": self.one_mutant_only,
            "evidence_cids": list(self.evidence_cids),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CampaignPartitionResult":
        if not isinstance(data, Mapping):
            raise RemediationRuntimeError("partition result must be a mapping")
        payload = dict(data)
        payload.pop("schema", None)
        return cls(
            partition=payload.get("partition"),
            passed=payload.get("passed"),
            mutant_ids=payload.get("mutant_ids", ()),
            killed_mutant_ids=payload.get("killed_mutant_ids", ()),
            killed_count=payload.get("killed_count"),
            survived_count=payload.get("survived_count"),
            mock_bypass=payload.get("mock_bypass", False),
            freezes_implementation=payload.get("freezes_implementation", False),
            one_mutant_only=payload.get("one_mutant_only", False),
            evidence_cids=payload.get("evidence_cids", ()),
            notes=payload.get("notes"),
        )

    def to_partition_evidence(
        self,
        *,
        force_failed: bool = False,
        extra_notes: str | None = None,
    ) -> PartitionEvaluationEvidence:
        passed = False if force_failed else self.passed
        notes = self.notes
        if extra_notes:
            notes = f"{notes}; {extra_notes}" if notes else extra_notes
        evidence_cids = list(self.evidence_cids)
        if not evidence_cids:
            # Deterministic synthetic evidence identity from partition observation.
            evidence_cids = [
                cid_for_structured(
                    {
                        "schema": CAMPAIGN_PARTITION_RESULT_SCHEMA,
                        "partition": self.partition,
                        "passed": passed,
                        "mutant_ids": list(self.mutant_ids),
                        "killed_count": self.killed_count,
                    }
                )
            ]
        return PartitionEvaluationEvidence(
            partition=self.partition,
            passed=passed,
            evidence_cids=tuple(evidence_cids),
            mutant_ids=tuple(self.mutant_ids),
            killed_count=int(self.killed_count or 0),
            survived_count=int(self.survived_count or 0),
            notes=notes,
        )


@dataclass(frozen=True, slots=True)
class HeldOutCampaign:
    """Held-out evaluation campaign inputs for ``evaluate_remediation@1``.

    Interface: ``HeldOutCampaign@1``

    Carries partition observations for original/unmutated, diagnosis,
    development, held-out, unrelated, performance, false-positive,
    overconstraint, regression, and safety behavior.
    """

    campaign_id: str
    header: AssuranceArtifactHeader | Mapping[str, Any]
    partition_results: Sequence[CampaignPartitionResult | Mapping[str, Any]]
    partition_plan: MutantPartitionPlan | Mapping[str, Any] | None = None
    cost_delta_basis_points: int = 0
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", _token(self.campaign_id, "campaign_id")
        )
        header = self.header
        if isinstance(header, Mapping):
            header = AssuranceArtifactHeader.from_dict(header)
        if not isinstance(header, AssuranceArtifactHeader):
            raise RemediationRuntimeError(
                "header must be AssuranceArtifactHeader or mapping"
            )
        object.__setattr__(self, "header", header)

        if not isinstance(self.partition_results, (list, tuple)):
            raise RemediationRuntimeError("partition_results must be a list")
        if len(self.partition_results) > MAX_LIST:
            raise RemediationRuntimeError("partition_results exceeds maximum length")
        sealed: list[CampaignPartitionResult] = []
        for index, item in enumerate(self.partition_results):
            if isinstance(item, CampaignPartitionResult):
                sealed.append(item)
            elif isinstance(item, Mapping):
                sealed.append(CampaignPartitionResult.from_dict(item))
            else:
                raise RemediationRuntimeError(
                    f"partition_results[{index}] must be CampaignPartitionResult "
                    "or mapping"
                )
        partitions = [item.partition for item in sealed]
        if len(partitions) != len(set(partitions)):
            raise RemediationRuntimeError(
                "partition_results partitions must be unique",
                reason_code="duplicate_partition",
            )
        object.__setattr__(self, "partition_results", tuple(sealed))

        if self.partition_plan is not None:
            if isinstance(self.partition_plan, Mapping):
                plan = MutantPartitionPlan.from_dict(self.partition_plan)
            elif isinstance(self.partition_plan, MutantPartitionPlan):
                plan = self.partition_plan
            else:
                raise RemediationRuntimeError(
                    "partition_plan must be MutantPartitionPlan or mapping"
                )
            object.__setattr__(self, "partition_plan", plan)
        object.__setattr__(
            self,
            "cost_delta_basis_points",
            _nonneg_int(
                self.cost_delta_basis_points,
                "cost_delta_basis_points",
                maximum=MAX_COST_BP,
            ),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def by_partition(self) -> Mapping[str, CampaignPartitionResult]:
        return MappingProxyType(
            {item.partition: item for item in self.partition_results}
        )

    def missing_required_partitions(
        self,
        required: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        needed = tuple(required or AAE046_EVALUATION_PARTITIONS)
        present = set(self.by_partition())
        return tuple(part for part in needed if part not in present)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": HELD_OUT_CAMPAIGN_SCHEMA,
            "interface_id": HELD_OUT_CAMPAIGN_INTERFACE,
            "campaign_id": self.campaign_id,
            "header": self.header.identity_payload(),
            "partition_results": [item.to_dict() for item in self.partition_results],
            "partition_plan_cid": (
                None
                if self.partition_plan is None
                else self.partition_plan.plan_cid
            ),
            "cost_delta_basis_points": self.cost_delta_basis_points,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @property
    def campaign_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["campaign_cid"] = self.campaign_cid
        payload["header"] = self.header.to_dict()
        if self.partition_plan is not None:
            payload["partition_plan"] = self.partition_plan.to_dict()
        else:
            payload["partition_plan"] = None
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeldOutCampaign":
        if not isinstance(data, Mapping):
            raise RemediationRuntimeError("held_out_campaign must be a mapping")
        payload = dict(data)
        payload.pop("campaign_cid", None)
        payload.pop("schema", None)
        payload.pop("interface_id", None)
        payload.pop("partition_plan_cid", None)
        return cls(
            campaign_id=payload.get("campaign_id"),
            header=payload.get("header"),
            partition_results=payload.get("partition_results", ()),
            partition_plan=payload.get("partition_plan"),
            cost_delta_basis_points=payload.get("cost_delta_basis_points", 0),
            notes=payload.get("notes"),
            metadata=payload.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Sealed run results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RemediationProposalRun:
    """Sealed orchestration result for ``propose_gap_remediation@1``."""

    schema: str = REMEDIATION_PROPOSAL_RUN_SCHEMA
    interface_id: str = REMEDIATION_PROPOSAL_RUN_INTERFACE
    run_cid: str = ""
    proposal: GapRemediationProposal | None = None
    proposal_cid: str | None = None
    plan_cid: str | None = None
    gap_cid: str | None = None
    survivor_report_cid: str | None = None
    candidate_cids: tuple[str, ...] = ()
    candidate_kinds: tuple[str, ...] = ()
    all_heuristic: bool = True
    requires_held_out_evaluation: bool = True
    production_policy_changed: bool = False
    reason_codes: tuple[str, ...] = ()
    evidence_subset: str = AAE_REMEDIATION_EVALUATION_EVIDENCE
    diagnostic: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        if self.proposal is not None and not isinstance(
            self.proposal, GapRemediationProposal
        ):
            raise RemediationRuntimeError(
                "proposal must be GapRemediationProposal"
            )
        object.__setattr__(
            self, "proposal_cid", _optional_cid(self.proposal_cid, "proposal_cid")
        )
        object.__setattr__(
            self, "plan_cid", _optional_cid(self.plan_cid, "plan_cid")
        )
        object.__setattr__(self, "gap_cid", _optional_cid(self.gap_cid, "gap_cid"))
        object.__setattr__(
            self,
            "survivor_report_cid",
            _optional_cid(self.survivor_report_cid, "survivor_report_cid"),
        )
        object.__setattr__(
            self,
            "candidate_cids",
            _unique_sorted_cids(list(self.candidate_cids), "candidate_cids")
            if self.candidate_cids
            else (),
        )
        object.__setattr__(
            self,
            "candidate_kinds",
            _unique_sorted_tokens(list(self.candidate_kinds), "candidate_kinds")
            if self.candidate_kinds
            else (),
        )
        object.__setattr__(
            self, "all_heuristic", _bool(self.all_heuristic, "all_heuristic")
        )
        if not self.all_heuristic:
            raise RemediationRuntimeError(
                "all_heuristic must be true; model drafts remain heuristic"
            )
        object.__setattr__(
            self,
            "requires_held_out_evaluation",
            _bool(
                self.requires_held_out_evaluation,
                "requires_held_out_evaluation",
            ),
        )
        if not self.requires_held_out_evaluation:
            raise RemediationRuntimeError(
                "requires_held_out_evaluation must be true"
            )
        # Hard invariant: never claim production policy change.
        object.__setattr__(self, "production_policy_changed", False)
        object.__setattr__(
            self,
            "reason_codes",
            _stable_unique(
                [
                    code
                    for code in (self.reason_codes or ())
                    if isinstance(code, str) and code
                ]
            ),
        )
        object.__setattr__(
            self,
            "evidence_subset",
            _text(self.evidence_subset, "evidence_subset"),
        )
        object.__setattr__(
            self,
            "diagnostic",
            _clip(
                _text(self.diagnostic, "diagnostic", empty=True),
                limit=MAX_DIAGNOSTIC,
            )
            if self.diagnostic
            else "",
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if not self.run_cid:
            object.__setattr__(self, "run_cid", self.compute_run_cid())

    def compute_run_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "proposal_cid": self.proposal_cid,
            "plan_cid": self.plan_cid,
            "gap_cid": self.gap_cid,
            "survivor_report_cid": self.survivor_report_cid,
            "candidate_cids": list(self.candidate_cids),
            "candidate_kinds": list(self.candidate_kinds),
            "all_heuristic": True,
            "requires_held_out_evaluation": True,
            "production_policy_changed": False,
            "reason_codes": list(self.reason_codes),
            "evidence_subset": self.evidence_subset,
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["run_cid"] = self.run_cid
        payload["proposal"] = (
            None if self.proposal is None else self.proposal.to_dict()
        )
        return payload


@dataclass(frozen=True, slots=True)
class RemediationEvaluationRun:
    """Sealed orchestration result for ``evaluate_remediation@1``."""

    schema: str = REMEDIATION_EVALUATION_RUN_SCHEMA
    interface_id: str = REMEDIATION_EVALUATION_RUN_INTERFACE
    run_cid: str = ""
    campaign_id: str = ""
    campaign_cid: str | None = None
    plan_cid: str = ""
    candidate_cids: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    evaluation_report: RemediationEvaluationReport | None = None
    evaluation_report_cid: str | None = None
    qualification: RemediationQualificationResult | None = None
    qualification_cid: str | None = None
    disposition: str = QualificationDisposition.REJECTED.value
    verdict: str = EvaluationVerdict.REJECTED.value
    partitions_covered: tuple[str, ...] = ()
    missing_partitions: tuple[str, ...] = ()
    failed_partitions: tuple[str, ...] = ()
    one_mutant_overfit: bool = False
    mock_bypass: bool = False
    qualified: bool = False
    production_policy_changed: bool = False
    reason_codes: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    evidence_subset: str = AAE_REMEDIATION_EVALUATION_EVIDENCE
    diagnostic: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        object.__setattr__(
            self, "campaign_id", _token(self.campaign_id, "campaign_id")
        )
        object.__setattr__(
            self, "campaign_cid", _optional_cid(self.campaign_cid, "campaign_cid")
        )
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "candidate_cids",
            _unique_sorted_cids(list(self.candidate_cids), "candidate_cids")
            if self.candidate_cids
            else (),
        )
        phases = tuple(
            _enum(phase, RemediationPhase, "phases") for phase in (self.phases or ())
        )
        object.__setattr__(self, "phases", phases)
        if self.evaluation_report is not None and not isinstance(
            self.evaluation_report, RemediationEvaluationReport
        ):
            raise RemediationRuntimeError(
                "evaluation_report must be RemediationEvaluationReport"
            )
        object.__setattr__(
            self,
            "evaluation_report_cid",
            _optional_cid(self.evaluation_report_cid, "evaluation_report_cid"),
        )
        if self.qualification is not None and not isinstance(
            self.qualification, RemediationQualificationResult
        ):
            raise RemediationRuntimeError(
                "qualification must be RemediationQualificationResult"
            )
        object.__setattr__(
            self,
            "qualification_cid",
            _optional_cid(self.qualification_cid, "qualification_cid"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, QualificationDisposition, "disposition"),
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, EvaluationVerdict, "verdict")
        )
        object.__setattr__(
            self,
            "partitions_covered",
            _unique_sorted_tokens(
                list(self.partitions_covered), "partitions_covered"
            )
            if self.partitions_covered
            else (),
        )
        object.__setattr__(
            self,
            "missing_partitions",
            _unique_sorted_tokens(
                list(self.missing_partitions), "missing_partitions"
            )
            if self.missing_partitions
            else (),
        )
        object.__setattr__(
            self,
            "failed_partitions",
            _unique_sorted_tokens(
                list(self.failed_partitions), "failed_partitions"
            )
            if self.failed_partitions
            else (),
        )
        object.__setattr__(
            self,
            "one_mutant_overfit",
            _bool(self.one_mutant_overfit, "one_mutant_overfit"),
        )
        object.__setattr__(
            self, "mock_bypass", _bool(self.mock_bypass, "mock_bypass")
        )
        object.__setattr__(self, "qualified", _bool(self.qualified, "qualified"))
        if self.qualified and (
            self.disposition != QualificationDisposition.QUALIFIED.value
            or self.verdict != EvaluationVerdict.QUALIFIED.value
            or self.one_mutant_overfit
            or self.mock_bypass
        ):
            raise RemediationRuntimeError(
                "qualified=true requires clean disposition/verdict without "
                "overfit or mock bypass",
                reason_code="qualified_invariant",
            )
        object.__setattr__(self, "production_policy_changed", False)
        object.__setattr__(
            self,
            "reason_codes",
            _stable_unique(
                [
                    code
                    for code in (self.reason_codes or ())
                    if isinstance(code, str) and code
                ]
            ),
        )
        object.__setattr__(
            self,
            "rejection_reasons",
            _unique_sorted_tokens(
                list(self.rejection_reasons), "rejection_reasons"
            )
            if self.rejection_reasons
            else (),
        )
        object.__setattr__(
            self,
            "evidence_subset",
            _text(self.evidence_subset, "evidence_subset"),
        )
        object.__setattr__(
            self,
            "diagnostic",
            _clip(
                _text(self.diagnostic, "diagnostic", empty=True),
                limit=MAX_DIAGNOSTIC,
            )
            if self.diagnostic
            else "",
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if not self.run_cid:
            object.__setattr__(self, "run_cid", self.compute_run_cid())

    def compute_run_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "campaign_id": self.campaign_id,
            "campaign_cid": self.campaign_cid,
            "plan_cid": self.plan_cid,
            "candidate_cids": list(self.candidate_cids),
            "phases": list(self.phases),
            "evaluation_report_cid": self.evaluation_report_cid,
            "qualification_cid": self.qualification_cid,
            "disposition": self.disposition,
            "verdict": self.verdict,
            "partitions_covered": list(self.partitions_covered),
            "missing_partitions": list(self.missing_partitions),
            "failed_partitions": list(self.failed_partitions),
            "one_mutant_overfit": self.one_mutant_overfit,
            "mock_bypass": self.mock_bypass,
            "qualified": self.qualified,
            "production_policy_changed": False,
            "reason_codes": list(self.reason_codes),
            "rejection_reasons": list(self.rejection_reasons),
            "evidence_subset": self.evidence_subset,
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["run_cid"] = self.run_cid
        payload["evaluation_report"] = (
            None
            if self.evaluation_report is None
            else self.evaluation_report.to_dict()
        )
        payload["qualification"] = (
            None if self.qualification is None else self.qualification.to_dict()
        )
        return payload


# ---------------------------------------------------------------------------
# Header / remediation normalization
# ---------------------------------------------------------------------------


def _clone_header(
    base: AssuranceArtifactHeader,
    *,
    artifact_kind: str,
    interface_id: str,
) -> AssuranceArtifactHeader:
    versions = base.versions
    generator = GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=interface_id,
    )
    new_versions = VersionBinding(
        operator_id=versions.operator_id,
        operator_version=versions.operator_version,
        campaign_policy_id=versions.campaign_policy_id,
        campaign_policy_version=versions.campaign_policy_version,
        generator=generator,
    )
    return AssuranceArtifactHeader(
        artifact_kind=artifact_kind,
        repository_id=base.repository_id,
        repository_state_cid=base.repository_state_cid,
        target_symbol_ids=tuple(base.target_symbol_ids),
        target_artifact_cids=tuple(base.target_artifact_cids),
        capsule_cids=tuple(base.capsule_cids),
        proof_unit_cids=tuple(base.proof_unit_cids),
        environment_cid=base.environment_cid,
        dependency_lock_cid=base.dependency_lock_cid,
        versions=new_versions,
        provenance=base.provenance,
        terminal_status=base.terminal_status,
        receipt_cids=tuple(base.receipt_cids),
        proof_cids=tuple(base.proof_cids),
        metadata=dict(base.metadata),
    )


def _normalize_proposal(
    value: GapRemediationProposal | Mapping[str, Any],
) -> GapRemediationProposal:
    if isinstance(value, GapRemediationProposal):
        return value
    if isinstance(value, Mapping):
        try:
            return GapRemediationProposal.from_dict(value)
        except (RemediationContractError, SpecRemediationError, AssuranceBaseError) as exc:
            raise RemediationRuntimeError(
                f"invalid GapRemediationProposal: {exc}",
                reason_code="invalid_proposal",
            ) from exc
    raise RemediationRuntimeError(
        "remediation must be GapRemediationProposal, GapRemediationPlan, or mapping"
    )


def _normalize_plan(
    value: GapRemediationPlan | Mapping[str, Any],
) -> GapRemediationPlan:
    if isinstance(value, GapRemediationPlan):
        plan = value
    elif isinstance(value, Mapping):
        try:
            plan = GapRemediationPlan.from_dict(value)
        except (RemediationContractError, AssuranceBaseError) as exc:
            raise RemediationRuntimeError(
                f"invalid GapRemediationPlan: {exc}",
                reason_code="invalid_plan",
            ) from exc
    else:
        raise RemediationRuntimeError(
            "remediation must be GapRemediationProposal, GapRemediationPlan, or mapping"
        )
    try:
        verify_plan_identity(plan)
    except RemediationContractError as exc:
        raise RemediationRuntimeError(
            f"plan identity failed: {exc}",
            reason_code="plan_identity",
        ) from exc
    return plan


def _extract_remediation(
    remediation: (
        GapRemediationProposal
        | GapRemediationPlan
        | RemediationProposalRun
        | Mapping[str, Any]
    ),
) -> tuple[GapRemediationPlan, tuple[str, ...], GapRemediationProposal | None]:
    """Return ``(plan, candidate_cids, optional_proposal)`` from remediation input."""

    if isinstance(remediation, RemediationProposalRun):
        if remediation.proposal is None:
            raise RemediationRuntimeError(
                "RemediationProposalRun.proposal is required",
                reason_code="missing_proposal",
            )
        proposal = remediation.proposal
        plan = proposal.plan
        candidate_cids = tuple(remediation.candidate_cids) or _candidate_cids_from_plan(
            plan
        )
        return plan, candidate_cids, proposal

    if isinstance(remediation, GapRemediationProposal):
        proposal = remediation
        plan = proposal.plan
        return plan, _candidate_cids_from_plan(plan), proposal

    if isinstance(remediation, GapRemediationPlan):
        plan = _normalize_plan(remediation)
        return plan, _candidate_cids_from_plan(plan), None

    if isinstance(remediation, Mapping):
        # Prefer proposal envelope when present.
        if "candidate_tests" in remediation or (
            remediation.get("interface_id") == DATASETS_PROPOSE_INTERFACE
            or remediation.get("interface_id") == PROPOSE_GAP_REMEDIATION_INTERFACE
        ):
            proposal = _normalize_proposal(remediation)
            plan = proposal.plan
            return plan, _candidate_cids_from_plan(plan), proposal
        if "proposal" in remediation and isinstance(remediation["proposal"], Mapping):
            proposal = _normalize_proposal(remediation["proposal"])
            plan = proposal.plan
            return plan, _candidate_cids_from_plan(plan), proposal
        if "plan" in remediation and isinstance(remediation["plan"], Mapping):
            # Nested plan under a proposal-like mapping without full candidate bodies.
            plan = _normalize_plan(remediation["plan"])
            cids = remediation.get("candidate_cids")
            if cids is not None:
                candidate_cids = _unique_sorted_cids(list(cids), "candidate_cids")
            else:
                candidate_cids = _candidate_cids_from_plan(plan)
            return plan, candidate_cids, None
        # Treat as bare plan mapping.
        plan = _normalize_plan(remediation)
        return plan, _candidate_cids_from_plan(plan), None

    raise RemediationRuntimeError(
        "remediation must be GapRemediationProposal, GapRemediationPlan, "
        "RemediationProposalRun, or mapping",
        reason_code="invalid_remediation",
    )


def _candidate_cids_from_plan(plan: GapRemediationPlan) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(plan.candidate_test_cids)
            | set(plan.candidate_proof_cids)
            | set(plan.candidate_policy_cids)
            | set(plan.candidate_analyzer_cids)
        )
    )


def _normalize_campaign(
    value: HeldOutCampaign | Mapping[str, Any],
) -> HeldOutCampaign:
    if isinstance(value, HeldOutCampaign):
        return value
    if isinstance(value, Mapping):
        return HeldOutCampaign.from_dict(value)
    raise RemediationRuntimeError(
        "held_out_campaign must be HeldOutCampaign or mapping",
        reason_code="invalid_campaign",
    )


# ---------------------------------------------------------------------------
# Overfit / mock-bypass detection
# ---------------------------------------------------------------------------


def detect_mock_bypass(
    campaign: HeldOutCampaign,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether any partition observation records mock bypass."""

    hit: list[str] = []
    for result in campaign.partition_results:
        if result.mock_bypass:
            hit.append(result.partition)
    return bool(hit), tuple(sorted(set(hit)))


def detect_one_mutant_overfit(
    campaign: HeldOutCampaign,
    *,
    proposal: GapRemediationProposal | None = None,
) -> tuple[bool, str | None]:
    """Detect classic one-mutant / diagnosis-only overfit.

    Overfit signals (any one is sufficient):

    * Explicit ``one_mutant_only`` flag on a related partition result.
    * Diagnosis killed while held-out failed (implementation assertion).
    * Exactly one related mutant killed while held-out mutants remain unkilled.
    * Candidate tests freeze implementation (when proposal is available).
    """

    by_part = campaign.by_partition()
    diagnosis = by_part.get(EvaluationPartition.DIAGNOSIS.value)
    development = by_part.get(EvaluationPartition.DEVELOPMENT.value)
    held_out = by_part.get(EvaluationPartition.HELD_OUT.value)

    for result in campaign.partition_results:
        if result.one_mutant_only and result.partition in _RELATED_KILL_PARTITIONS:
            return True, (
                f"partition {result.partition} marked one_mutant_only"
            )
        if result.freezes_implementation:
            return True, (
                f"partition {result.partition} freezes_implementation"
            )

    if proposal is not None:
        for test in proposal.candidate_tests:
            if test.freezes_implementation:
                return True, "candidate test freezes implementation"
            if test.draft_status != CandidateDraftStatus.HEURISTIC_CANDIDATE.value:
                # Non-heuristic without evaluation is self-promotion risk; treat
                # as overfit authority abuse for evaluation purposes.
                return True, "candidate draft is not heuristic_candidate"

    if diagnosis is None or held_out is None:
        return False, None

    diagnosis_killed = bool(diagnosis.passed and (diagnosis.killed_count or 0) > 0)
    # Diagnosis partition with no mutants can still "pass" if suite-level signal.
    if diagnosis.passed and not diagnosis.mutant_ids:
        diagnosis_killed = True
    held_out_killed = bool(held_out.passed and (
        (held_out.killed_count or 0) > 0 or not held_out.mutant_ids
    ))
    if not held_out.passed:
        held_out_killed = False

    if diagnosis_killed and not held_out_killed:
        return True, "diagnosis killed without held-out kill (one-mutant overfit)"

    related_killed = 0
    related_mutants = 0
    for key in _RELATED_KILL_PARTITIONS:
        result = by_part.get(key)
        if result is None:
            continue
        related_killed += int(result.killed_count or 0)
        related_mutants += len(result.mutant_ids)

    if (
        related_mutants > 1
        and related_killed == 1
        and diagnosis_killed
        and not held_out_killed
    ):
        return True, "only one related mutant killed across corpus"

    if (
        development is not None
        and development.mutant_ids
        and diagnosis_killed
        and not development.passed
        and not held_out_killed
    ):
        return True, "diagnosis-only kill without development/held-out generalization"

    return False, None


def _partition_coverage_reasons(
    present: Mapping[str, CampaignPartitionResult],
) -> list[str]:
    reasons: list[str] = [REASON_EVALUATION_COVERED]
    mapping = {
        EvaluationPartition.UNMUTATED.value: REASON_ORIGINAL_EVALUATED,
        EvaluationPartition.DIAGNOSIS.value: REASON_DIAGNOSIS_EVALUATED,
        EvaluationPartition.DEVELOPMENT.value: REASON_DEVELOPMENT_EVALUATED,
        EvaluationPartition.HELD_OUT.value: REASON_HELD_OUT_EVALUATED,
        EvaluationPartition.UNRELATED.value: REASON_UNRELATED_EVALUATED,
        EvaluationPartition.PERFORMANCE_COST.value: REASON_PERFORMANCE_EVALUATED,
        EvaluationPartition.FALSE_POSITIVE.value: REASON_FALSE_POSITIVE_EVALUATED,
        EvaluationPartition.OVERCONSTRAINT.value: REASON_OVERCONSTRAINT_EVALUATED,
        EvaluationPartition.SAFETY.value: REASON_SAFETY_EVALUATED,
    }
    for partition, reason in mapping.items():
        if partition in present:
            reasons.append(reason)
    return reasons


def _derive_report_flags(
    by_part: Mapping[str, CampaignPartitionResult],
    *,
    one_mutant_overfit: bool,
    mock_bypass: bool,
    cost_delta_bp: int,
    max_cost_delta_bp: int,
) -> dict[str, Any]:
    def _passed(name: str) -> bool:
        item = by_part.get(name)
        return bool(item is not None and item.passed)

    unmutated = _passed(EvaluationPartition.UNMUTATED.value)
    diagnosis = _passed(EvaluationPartition.DIAGNOSIS.value)
    development = _passed(EvaluationPartition.DEVELOPMENT.value)
    held_out = _passed(EvaluationPartition.HELD_OUT.value)
    unrelated = _passed(EvaluationPartition.UNRELATED.value)
    safety = _passed(EvaluationPartition.SAFETY.value)
    regression = not _passed(EvaluationPartition.REGRESSION.value)
    overconstraint = not _passed(EvaluationPartition.OVERCONSTRAINT.value)
    false_positive = not _passed(EvaluationPartition.FALSE_POSITIVE.value)
    cost_ok = _passed(EvaluationPartition.PERFORMANCE_COST.value) and (
        cost_delta_bp <= max_cost_delta_bp
    )

    rejection_reasons: list[str] = []
    if mock_bypass:
        rejection_reasons.append(RejectionReason.MOCK_BYPASS.value)
    if one_mutant_overfit:
        rejection_reasons.append(
            RejectionReason.OVERFIT_IMPLEMENTATION_ASSERTION.value
        )
        if not held_out:
            rejection_reasons.append(RejectionReason.HELD_OUT_FAILURE.value)
    if not unmutated:
        rejection_reasons.append(RejectionReason.UNMUTATED_SUITE_FAILED.value)
    if not diagnosis:
        rejection_reasons.append(RejectionReason.DIAGNOSIS_NOT_KILLED.value)
    if not held_out:
        rejection_reasons.append(RejectionReason.HELD_OUT_FAILURE.value)
    if not unrelated:
        rejection_reasons.append(RejectionReason.UNRELATED_BEHAVIOR_BROKEN.value)
    if not safety:
        rejection_reasons.append(RejectionReason.SAFETY_WEAKENING.value)
    if regression:
        rejection_reasons.append(RejectionReason.REGRESSION.value)
    if overconstraint:
        rejection_reasons.append(RejectionReason.OVERCONSTRAINT.value)
    if false_positive:
        rejection_reasons.append(RejectionReason.FALSE_POSITIVE.value)
    if not cost_ok:
        rejection_reasons.append(RejectionReason.UNAPPROVED_COST_INCREASE.value)

    # Failed partition evidence also contributes.
    for name, item in by_part.items():
        if not item.passed and name not in {
            EvaluationPartition.REGRESSION.value,
            EvaluationPartition.OVERCONSTRAINT.value,
            EvaluationPartition.FALSE_POSITIVE.value,
        }:
            # already mapped above for required behavioral partitions
            pass
        if item.mock_bypass:
            rejection_reasons.append(RejectionReason.MOCK_BYPASS.value)

    rejection_reasons = sorted(set(rejection_reasons))

    qualifies_flags = (
        unmutated
        and diagnosis
        and development
        and held_out
        and unrelated
        and safety
        and not regression
        and not overconstraint
        and not false_positive
        and cost_ok
        and not one_mutant_overfit
        and not mock_bypass
        and not rejection_reasons
    )

    if qualifies_flags:
        verdict = EvaluationVerdict.QUALIFIED.value
        reasons: tuple[str, ...] = ()
    else:
        if RejectionReason.MOCK_BYPASS.value in rejection_reasons:
            verdict = EvaluationVerdict.REJECTED.value
        elif RejectionReason.OVERFIT_IMPLEMENTATION_ASSERTION.value in rejection_reasons:
            verdict = EvaluationVerdict.OVERFIT.value
        elif RejectionReason.REGRESSION.value in rejection_reasons:
            verdict = EvaluationVerdict.REGRESSION.value
        elif RejectionReason.OVERCONSTRAINT.value in rejection_reasons:
            verdict = EvaluationVerdict.OVERCONSTRAINT.value
        elif RejectionReason.SAFETY_WEAKENING.value in rejection_reasons:
            verdict = EvaluationVerdict.SAFETY_WEAKENED.value
        elif RejectionReason.UNAPPROVED_COST_INCREASE.value in rejection_reasons:
            verdict = EvaluationVerdict.COST_EXCEEDED.value
        else:
            verdict = EvaluationVerdict.REJECTED.value
        if not rejection_reasons:
            rejection_reasons = [RejectionReason.HELD_OUT_FAILURE.value]
        reasons = tuple(sorted(set(rejection_reasons)))

    return {
        "unmutated_suite_passed": unmutated,
        "diagnosis_killed": diagnosis,
        "development_killed": development,
        "held_out_killed": held_out,
        "unrelated_behavior_preserved": unrelated,
        "safety_preserved": safety,
        "regression_detected": regression,
        "overconstraint_detected": overconstraint,
        "false_positive_detected": false_positive,
        "cost_ok": cost_ok,
        "verdict": verdict,
        "rejection_reasons": reasons,
        "qualifies_flags": qualifies_flags,
    }


# ---------------------------------------------------------------------------
# Public: propose_gap_remediation
# ---------------------------------------------------------------------------


def propose_gap_remediation(
    surviving_mutant: SurvivingMutantReport | Mapping[str, Any],
    assurance_gap: AssuranceGap | Mapping[str, Any],
    *,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RemediationProposalRun:
    """Generate requirement-grounded candidate remediations for one gap.

    Interface: ``propose_gap_remediation@1``

    Plan signature: ``propose_gap_remediation(surviving_mutant, assurance_gap)``.

    Composes released AAE-032 authority. Model drafts remain
    ``heuristic_candidate`` and always require held-out evaluation. Never
    changes production policy.
    """

    try:
        proposal = datasets_propose_gap_remediation(
            surviving_mutant,
            assurance_gap,
            notes=notes,
            metadata=metadata,
        )
    except SpecRemediationError as exc:
        raise RemediationRuntimeError(
            f"propose_gap_remediation failed: {exc}",
            reason_code="propose_failed",
        ) from exc

    verify_gap_remediation_proposal_identity(proposal)
    plan = proposal.plan
    candidate_cids = _candidate_cids_from_plan(plan)
    reasons = (
        REASON_CANDIDATES_PROPOSED,
        REASON_REQUIREMENT_GROUNDED,
        REASON_HEURISTIC_CANDIDATE,
        REASON_HELD_OUT_REQUIRED,
        REASON_NO_PRODUCTION_POLICY_CHANGE,
    )
    meta: dict[str, Any] = {
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "datasets_interface": DATASETS_PROPOSE_INTERFACE,
        "gap_class": proposal.gap_class,
        "production_policy_changed": False,
    }
    if metadata:
        meta.update(dict(metadata))

    return RemediationProposalRun(
        proposal=proposal,
        proposal_cid=proposal.result_cid,
        plan_cid=plan.plan_cid,
        gap_cid=proposal.gap_cid,
        survivor_report_cid=proposal.survivor_report_cid,
        candidate_cids=candidate_cids,
        candidate_kinds=tuple(proposal.candidate_kinds),
        all_heuristic=True,
        requires_held_out_evaluation=True,
        production_policy_changed=False,
        reason_codes=reasons,
        diagnostic=(
            f"proposed {len(candidate_cids)} heuristic candidate(s) for "
            f"gap_class={proposal.gap_class}; held-out evaluation required"
        ),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Public: evaluate_remediation
# ---------------------------------------------------------------------------


def evaluate_remediation(
    remediation: (
        GapRemediationProposal
        | GapRemediationPlan
        | RemediationProposalRun
        | Mapping[str, Any]
    ),
    held_out_campaign: HeldOutCampaign | Mapping[str, Any],
    *,
    max_cost_delta_bp: int = DEFAULT_MAX_COST_DELTA_BP,
    report_id: str | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    raise_on_hard_reject: bool = False,
) -> RemediationEvaluationRun:
    """Evaluate a remediation against a held-out campaign.

    Interface: ``evaluate_remediation@1``

    Plan signature: ``evaluate_remediation(remediation, held_out_campaign)``.

    Covers original (unmutated), diagnosis, development, held-out, unrelated,
    performance, false-positive, overconstraint, and safety behavior. Rejects
    one-mutant overfit and mock bypass. Qualifies via AAE-033
    ``qualify_remediation_evaluation``. Never changes production policy.
    """

    phases: list[str] = [RemediationPhase.ADMIT.value]
    plan, candidate_cids, proposal = _extract_remediation(remediation)
    if not candidate_cids:
        raise RemediationRuntimeError(
            "remediation must bind at least one candidate_cid",
            reason_code="missing_candidates",
        )
    if not plan.requires_held_out_evaluation:
        raise RemediationRuntimeError(
            "plan.requires_held_out_evaluation must be true",
            reason_code="held_out_not_required",
        )
    if plan.plan_status == RemediationPlanStatus.REJECTED.value:
        raise RemediationRuntimeError(
            "rejected plans cannot be evaluated",
            reason_code="plan_rejected",
        )

    campaign = _normalize_campaign(held_out_campaign)
    max_cost = _nonneg_int(
        max_cost_delta_bp, "max_cost_delta_bp", maximum=MAX_COST_BP
    )
    phases.append(RemediationPhase.PARTITION.value)

    missing = campaign.missing_required_partitions(AAE046_EVALUATION_PARTITIONS)
    by_part = dict(campaign.by_partition())
    phases.append(RemediationPhase.EVALUATE.value)

    mock_bypass, mock_partitions = detect_mock_bypass(campaign)
    one_mutant_overfit, overfit_note = detect_one_mutant_overfit(
        campaign, proposal=proposal
    )
    phases.append(RemediationPhase.DETECT.value)

    if raise_on_hard_reject and mock_bypass:
        raise MockBypassError(
            f"mock bypass observed in partitions {list(mock_partitions)}",
            reason_code=RejectionReason.MOCK_BYPASS.value,
        )
    if raise_on_hard_reject and one_mutant_overfit:
        raise OneMutantOverfitError(
            overfit_note or "one-mutant overfit detected",
            reason_code=RejectionReason.OVERFIT_IMPLEMENTATION_ASSERTION.value,
        )

    # Build partition evidence; force-fail mock-bypass / overfit partitions.
    evidence: list[PartitionEvaluationEvidence] = []
    failed: list[str] = []
    adjusted: dict[str, CampaignPartitionResult] = {}
    for partition_name in AAE046_EVALUATION_PARTITIONS:
        result = by_part.get(partition_name)
        if result is None:
            continue
        force_failed = False
        extra_notes: str | None = None
        if result.mock_bypass:
            force_failed = True
            extra_notes = "mock_bypass"
            failed.append(partition_name)
        elif one_mutant_overfit and partition_name == EvaluationPartition.HELD_OUT.value:
            # Overfit invalidates held-out generalization even if raw pass is set.
            force_failed = True
            extra_notes = overfit_note or "one_mutant_overfit"
            failed.append(partition_name)
        elif not result.passed:
            failed.append(partition_name)
        evidence.append(
            result.to_partition_evidence(
                force_failed=force_failed,
                extra_notes=extra_notes,
            )
        )
        if force_failed and result.passed:
            adjusted[partition_name] = CampaignPartitionResult(
                partition=result.partition,
                passed=False,
                mutant_ids=result.mutant_ids,
                killed_mutant_ids=result.killed_mutant_ids,
                killed_count=result.killed_count,
                survived_count=result.survived_count,
                mock_bypass=result.mock_bypass,
                freezes_implementation=result.freezes_implementation,
                one_mutant_only=result.one_mutant_only,
                evidence_cids=result.evidence_cids,
                notes=extra_notes or result.notes,
            )
        else:
            adjusted[partition_name] = result

    # When partitions are missing, still build a partial report only if
    # regression+overconstraint are present (contract minimum); otherwise fail
    # closed without a report object when we cannot satisfy contract.
    can_build_report = (
        EvaluationPartition.REGRESSION.value in by_part
        and EvaluationPartition.OVERCONSTRAINT.value in by_part
        and bool(evidence)
        and not missing
    )

    flags = _derive_report_flags(
        adjusted,
        one_mutant_overfit=one_mutant_overfit,
        mock_bypass=mock_bypass,
        cost_delta_bp=campaign.cost_delta_basis_points,
        max_cost_delta_bp=max_cost,
    )

    # Missing required partitions always block qualification.
    if missing:
        flags["qualifies_flags"] = False
        reasons = list(flags["rejection_reasons"])
        reasons.append(RejectionReason.HELD_OUT_FAILURE.value)
        flags["rejection_reasons"] = tuple(sorted(set(reasons)))
        if flags["verdict"] == EvaluationVerdict.QUALIFIED.value:
            flags["verdict"] = EvaluationVerdict.REJECTED.value

    report: RemediationEvaluationReport | None = None
    report_cid: str | None = None
    qualification: RemediationQualificationResult | None = None
    qualification_cid: str | None = None

    report_header = _clone_header(
        campaign.header,
        artifact_kind="remediation_evaluation_report",
        interface_id=EVALUATE_REMEDIATION_INTERFACE,
    )
    resolved_report_id = (
        _token(report_id, "report_id")
        if report_id is not None
        else _token(f"eval_{campaign.campaign_id}", "report_id")
    )

    note_parts: list[str] = []
    if notes:
        note_parts.append(_text(notes, "notes"))
    if campaign.notes:
        note_parts.append(campaign.notes)
    if overfit_note:
        note_parts.append(overfit_note)
    if mock_bypass:
        note_parts.append(
            f"mock_bypass in partitions={list(mock_partitions)}"
        )
    if missing:
        note_parts.append(f"missing_partitions={list(missing)}")
    report_notes = "; ".join(note_parts) if note_parts else None

    report_metadata: dict[str, Any] = {
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "evidence": AAE_REMEDIATION_EVALUATION_EVIDENCE,
        "held_out_policy_evidence": HELD_OUT_POLICY_EVIDENCE,
        "campaign_id": campaign.campaign_id,
        "campaign_cid": campaign.campaign_cid,
        "one_mutant_overfit": one_mutant_overfit,
        "mock_bypass": mock_bypass,
        "production_policy_changed": False,
        "required_partitions": list(AAE046_EVALUATION_PARTITIONS),
    }
    if metadata:
        report_metadata.update(dict(metadata))

    phases.append(RemediationPhase.REPORT.value)
    if can_build_report and not missing:
        try:
            report = RemediationEvaluationReport(
                header=report_header,
                report_id=resolved_report_id,
                plan_cid=plan.plan_cid,
                candidate_cids=candidate_cids,
                verdict=flags["verdict"],
                partition_evidence=tuple(evidence),
                regression_detected=bool(flags["regression_detected"]),
                overconstraint_detected=bool(flags["overconstraint_detected"]),
                false_positive_detected=bool(flags["false_positive_detected"]),
                unmutated_suite_passed=bool(flags["unmutated_suite_passed"]),
                diagnosis_killed=bool(flags["diagnosis_killed"]),
                development_killed=bool(flags["development_killed"]),
                held_out_killed=bool(flags["held_out_killed"]),
                unrelated_behavior_preserved=bool(
                    flags["unrelated_behavior_preserved"]
                ),
                safety_preserved=bool(flags["safety_preserved"]),
                cost_delta_basis_points=campaign.cost_delta_basis_points,
                rejection_reasons=list(flags["rejection_reasons"]),
                notes=report_notes,
                metadata=report_metadata,
            )
            report_cid = verify_evaluation_report_identity(report)
        except RemediationContractError as exc:
            raise RemediationRuntimeError(
                f"evaluation report construction failed: {exc}",
                reason_code="report_construction",
            ) from exc

        phases.append(RemediationPhase.QUALIFY.value)
        try:
            qualification = qualify_remediation_evaluation(
                report,
                partition_plan=campaign.partition_plan,
                max_cost_delta_bp=max_cost,
                notes=report_notes,
                metadata={
                    "adapter_id": ADAPTER_ID,
                    "campaign_id": campaign.campaign_id,
                    "one_mutant_overfit": one_mutant_overfit,
                    "mock_bypass": mock_bypass,
                },
            )
            qualification_cid = verify_remediation_qualification_identity(
                qualification
            )
        except HeldOutPolicyError as exc:
            raise RemediationRuntimeError(
                f"qualification failed: {exc}",
                reason_code="qualification_failed",
            ) from exc
    else:
        # Incomplete campaign: fail closed without durable qualified report.
        phases.append(RemediationPhase.QUALIFY.value)

    # Resolve final disposition.
    if (
        qualification is not None
        and qualification.disposition == QualificationDisposition.QUALIFIED.value
        and not one_mutant_overfit
        and not mock_bypass
        and not missing
    ):
        disposition = QualificationDisposition.QUALIFIED.value
        verdict = EvaluationVerdict.QUALIFIED.value
        qualified = True
        rejection_reasons: tuple[str, ...] = ()
    else:
        disposition = QualificationDisposition.REJECTED.value
        if qualification is not None:
            verdict = qualification.verdict
            rejection_reasons = tuple(qualification.rejection_reasons)
        else:
            verdict = flags["verdict"]
            rejection_reasons = tuple(flags["rejection_reasons"])
        if one_mutant_overfit:
            rejection_reasons = tuple(
                sorted(
                    set(rejection_reasons)
                    | {
                        RejectionReason.OVERFIT_IMPLEMENTATION_ASSERTION.value,
                        RejectionReason.HELD_OUT_FAILURE.value,
                    }
                )
            )
            verdict = EvaluationVerdict.OVERFIT.value
        if mock_bypass:
            rejection_reasons = tuple(
                sorted(
                    set(rejection_reasons) | {RejectionReason.MOCK_BYPASS.value}
                )
            )
            verdict = EvaluationVerdict.REJECTED.value
        if missing:
            rejection_reasons = tuple(
                sorted(
                    set(rejection_reasons)
                    | {RejectionReason.HELD_OUT_FAILURE.value}
                )
            )
        qualified = False

    reason_codes: list[str] = [
        REASON_NO_PRODUCTION_POLICY_CHANGE,
        REASON_QUALIFICATION_APPLIED,
    ]
    reason_codes.extend(_partition_coverage_reasons(by_part))
    if qualified:
        reason_codes.append(REASON_QUALIFIED)
    else:
        reason_codes.append(REASON_REJECTED)
    if one_mutant_overfit:
        reason_codes.append(REASON_ONE_MUTANT_OVERFIT)
    if mock_bypass:
        reason_codes.append(REASON_MOCK_BYPASS)

    failed_out = tuple(sorted(set(failed) | set(missing)))
    if qualification is not None and qualification.failed_partitions:
        failed_out = tuple(
            sorted(set(failed_out) | set(qualification.failed_partitions))
        )

    diagnostic = (
        f"evaluate_remediation campaign={campaign.campaign_id} "
        f"disposition={disposition} verdict={verdict} "
        f"covered={len(by_part)} missing={len(missing)} "
        f"overfit={one_mutant_overfit} mock_bypass={mock_bypass}"
    )

    run_metadata: dict[str, Any] = {
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "evidence": AAE_REMEDIATION_EVALUATION_EVIDENCE,
        "held_out_policy_evidence": HELD_OUT_POLICY_EVIDENCE,
        "max_cost_delta_bp": max_cost,
        "production_policy_changed": False,
        "overfit_note": overfit_note,
        "mock_partitions": list(mock_partitions),
        "plan_status": plan.plan_status,
    }
    if metadata:
        run_metadata.update(dict(metadata))

    return RemediationEvaluationRun(
        campaign_id=campaign.campaign_id,
        campaign_cid=campaign.campaign_cid,
        plan_cid=plan.plan_cid,
        candidate_cids=candidate_cids,
        phases=tuple(phases),
        evaluation_report=report,
        evaluation_report_cid=report_cid,
        qualification=qualification,
        qualification_cid=qualification_cid,
        disposition=disposition,
        verdict=verdict,
        partitions_covered=tuple(sorted(by_part)),
        missing_partitions=tuple(missing),
        failed_partitions=failed_out,
        one_mutant_overfit=one_mutant_overfit,
        mock_bypass=mock_bypass,
        qualified=qualified,
        production_policy_changed=False,
        reason_codes=tuple(reason_codes),
        rejection_reasons=rejection_reasons,
        diagnostic=diagnostic,
        metadata=run_metadata,
    )


# ---------------------------------------------------------------------------
# Descriptors and vocabulary helpers
# ---------------------------------------------------------------------------


def evaluate_remediation_descriptor() -> Mapping[str, Any]:
    """Return a static descriptor for the runtime evaluation interface."""

    return MappingProxyType(
        {
            "interface_id": EVALUATE_REMEDIATION_INTERFACE,
            "run_interface": REMEDIATION_EVALUATION_RUN_INTERFACE,
            "evidence": AAE_REMEDIATION_EVALUATION_EVIDENCE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "depends_on": (
                DATASETS_PROPOSE_INTERFACE,
                "qualify_remediation_evaluation@1",
                "partition_mutants@1",
            ),
            "required_partitions": list(AAE046_EVALUATION_PARTITIONS),
            "original_partition": ORIGINAL_PARTITION,
            "performance_partition": PERFORMANCE_PARTITION,
            "rejects": (
                RejectionReason.OVERFIT_IMPLEMENTATION_ASSERTION.value,
                RejectionReason.MOCK_BYPASS.value,
            ),
            "production_policy_change": False,
        }
    )


def propose_gap_remediation_descriptor() -> Mapping[str, Any]:
    """Return a static descriptor for the runtime proposal interface."""

    return MappingProxyType(
        {
            "interface_id": PROPOSE_GAP_REMEDIATION_INTERFACE,
            "run_interface": REMEDIATION_PROPOSAL_RUN_INTERFACE,
            "evidence": AAE_REMEDIATION_EVALUATION_EVIDENCE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "depends_on": (DATASETS_PROPOSE_INTERFACE,),
            "production_policy_change": False,
            "requires_held_out_evaluation": True,
        }
    )


def aae046_evaluation_partitions() -> tuple[str, ...]:
    """Return AAE-046 acceptance partition vocabulary (including regression)."""

    return AAE046_EVALUATION_PARTITIONS


def evaluation_covers_acceptance_partitions(
    run: RemediationEvaluationRun | Mapping[str, Any],
) -> bool:
    """True when a run covered every AAE-046 acceptance partition."""

    if isinstance(run, Mapping):
        covered = set(run.get("partitions_covered") or ())
        missing = set(run.get("missing_partitions") or ())
    else:
        covered = set(run.partitions_covered)
        missing = set(run.missing_partitions)
    # Acceptance list uses original/performance naming; map to closed tokens.
    required = set(AAE046_EVALUATION_PARTITIONS)
    return required.issubset(covered) and not missing and required.isdisjoint(set())


__all__ = [
    "AAE046_EVALUATION_PARTITIONS",
    "AAE_REMEDIATION_EVALUATION_EVIDENCE",
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "CAMPAIGN_PARTITION_RESULT_SCHEMA",
    "CampaignPartitionResult",
    "EVALUATE_REMEDIATION_INTERFACE",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "HELD_OUT_CAMPAIGN_INTERFACE",
    "HELD_OUT_CAMPAIGN_SCHEMA",
    "HeldOutCampaign",
    "MockBypassError",
    "ORIGINAL_PARTITION",
    "OneMutantOverfitError",
    "PERFORMANCE_PARTITION",
    "PROPOSE_GAP_REMEDIATION_INTERFACE",
    "REASON_CANDIDATES_PROPOSED",
    "REASON_DIAGNOSIS_EVALUATED",
    "REASON_DEVELOPMENT_EVALUATED",
    "REASON_EVALUATION_COVERED",
    "REASON_FALSE_POSITIVE_EVALUATED",
    "REASON_HELD_OUT_EVALUATED",
    "REASON_HELD_OUT_REQUIRED",
    "REASON_HEURISTIC_CANDIDATE",
    "REASON_MOCK_BYPASS",
    "REASON_NO_PRODUCTION_POLICY_CHANGE",
    "REASON_ONE_MUTANT_OVERFIT",
    "REASON_ORIGINAL_EVALUATED",
    "REASON_OVERCONSTRAINT_EVALUATED",
    "REASON_PERFORMANCE_EVALUATED",
    "REASON_QUALIFICATION_APPLIED",
    "REASON_QUALIFIED",
    "REASON_REJECTED",
    "REASON_REQUIREMENT_GROUNDED",
    "REASON_SAFETY_EVALUATED",
    "REASON_UNRELATED_EVALUATED",
    "REMEDIATION_EVALUATION_RUN_INTERFACE",
    "REMEDIATION_EVALUATION_RUN_SCHEMA",
    "REMEDIATION_PROPOSAL_RUN_INTERFACE",
    "REMEDIATION_PROPOSAL_RUN_SCHEMA",
    "RemediationEvaluationRun",
    "RemediationPhase",
    "RemediationProposalRun",
    "RemediationRuntimeError",
    "aae046_evaluation_partitions",
    "detect_mock_bypass",
    "detect_one_mutant_overfit",
    "evaluate_remediation",
    "evaluate_remediation_descriptor",
    "evaluation_covers_acceptance_partitions",
    "propose_gap_remediation",
    "propose_gap_remediation_descriptor",
    "required_evaluation_partitions",
]
