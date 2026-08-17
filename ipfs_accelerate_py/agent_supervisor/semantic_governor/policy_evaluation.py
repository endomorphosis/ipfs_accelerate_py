"""Evaluate rule/policy candidates only on disjoint held-out evidence (SCG-033).

``evaluate_rule_candidate`` validates a :class:`CompressionPolicyCandidate`
against an immutable held-out benchmark partition. Evaluation is pure and
emit-only: it never mutates the candidate, benchmark, or baseline policy.

Normative fail-closed invariants:

* Missing held-out cases reject.
* Overlap between held-out case identities and calibration, development, or
  candidate-generating identities rejects (partitions are immutable and
  disjoint; candidate-generating cases cannot score promotion).
* Critical omission detection and stale rejection cannot regress versus the
  baseline rates bound into the benchmark.
* Any hidden accepted regression blocks promotion (accepted_regression_bp must
  not exceed the declared max, default zero).
* Schema/integrity, high-risk assurance, and declared cost/context thresholds
  are checked; the result is a reproducible :class:`RuleEvaluationReport`.

Conflict policy: Calibration/development/held-out identities are disjoint;
candidate-generating cases cannot score promotion. Full-suite fallback cannot
be disabled by evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence
import re
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
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
    BASIS_POINTS,
    EvidencePartition,
    ratio_to_basis_points,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    CompressionPolicy,
    CompressionPolicyCandidate,
    EvaluationVerdict,
    PolicyContractError,
    ProtectedThresholds,
    RuleEvaluationReport,
    protected_threshold_reductions,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_HELD_OUT_EVALUATION_EVIDENCE: Final[str] = "scg/held-out-evaluation@1"
EVALUATE_RULE_CANDIDATE_INTERFACE: Final[str] = "evaluate_rule_candidate@1"
HELD_OUT_BENCHMARK_INTERFACE: Final[str] = "HeldOutBenchmark@1"
HELD_OUT_BENCHMARK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "held-out-benchmark@1"
)
HELD_OUT_CASE_OUTCOME_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "held-out-case-outcome@1"
)
EVALUATION_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "rule-evaluation-metrics@1"
)

GENERATOR_ID: Final[str] = "policy_evaluator"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "policy_evaluation.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_CASES: Final[int] = 4_096
MAX_CID_LIST: Final[int] = 4_096
MAX_BLOCKING_REASONS: Final[int] = 256
MAX_METADATA_KEYS: Final[int] = 64

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

# Stable reason codes for blocking / rejection paths.
REASON_MISSING_HELD_OUT: Final[str] = "missing_held_out_data"
REASON_OVERLAP: Final[str] = "held_out_partition_overlap"
REASON_CANDIDATE_GENERATING_OVERLAP: Final[str] = (
    "candidate_generating_case_in_held_out"
)
REASON_PARTITION_NOT_HELD_OUT: Final[str] = "partition_not_held_out"
REASON_SCHEMA_INTEGRITY: Final[str] = "schema_or_integrity_failure"
REASON_OMISSION_REGRESSION: Final[str] = "critical_omission_detection_regressed"
REASON_STALE_REGRESSION: Final[str] = "stale_rejection_regressed"
REASON_HIDDEN_REGRESSION: Final[str] = "hidden_accepted_regression"
REASON_OMISSION_ACCEPTED: Final[str] = "critical_omission_accepted"
REASON_OMISSION_THRESHOLD: Final[str] = "critical_omission_detection_below_threshold"
REASON_REGRESSION_THRESHOLD: Final[str] = "accepted_regression_above_threshold"
REASON_CONTEXT_REDUCTION: Final[str] = "median_context_reduction_below_threshold"
REASON_HIGH_RISK_REDUCTION: Final[str] = "high_risk_assurance_reduced"
REASON_FULL_SUITE_DISABLED: Final[str] = "full_suite_fallback_disabled"
REASON_CANDIDATE_POLICY_MISMATCH: Final[str] = "candidate_baseline_policy_mismatch"
REASON_DUPLICATE_CASE: Final[str] = "duplicate_held_out_case_identity"
REASON_EMPTY_PROBE: Final[str] = "missing_required_held_out_probes"


class PolicyEvaluationError(SemanticGovernorBaseError):
    """Raised when held-out policy evaluation inputs are malformed or unsafe."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise PolicyEvaluationError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise PolicyEvaluationError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise PolicyEvaluationError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise PolicyEvaluationError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise PolicyEvaluationError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PolicyEvaluationError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise PolicyEvaluationError(f"{name} must be a nonnegative integer")
    return value


def _basis_points(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise PolicyEvaluationError(
            f"{name} must be an integer basis-point ratio in [0, {BASIS_POINTS}]"
        )
    if value < 0 or value > BASIS_POINTS:
        raise PolicyEvaluationError(
            f"{name} must be an integer basis-point ratio in [0, {BASIS_POINTS}]"
        )
    return value


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_structured(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_structured(item) for item in value)
    return value


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_structured(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_structured(item) for item in value]
    return value


def _require_structured(value: Any, name: str) -> Any:
    thawed = _thaw_structured(value)
    try:
        validate_structured_value(thawed, path=name)
    except Exception as exc:
        raise PolicyEvaluationError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    reject_private_and_model_authority(thawed, path=name)
    return thawed


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyEvaluationError(f"{name} must be a mapping")
    return _freeze_structured(_require_structured(dict(value), name))


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise PolicyEvaluationError(f"{name} must be a mapping")
    actual = set(data)
    if actual != fields:
        raise PolicyEvaluationError(
            f"{name} fields must be exactly {sorted(fields)}, got {sorted(actual)}"
        )
    return dict(data)


def _unique_sorted_cids(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise PolicyEvaluationError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(value, name) for value in values))
    if len(ordered) > MAX_CID_LIST:
        raise PolicyEvaluationError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise PolicyEvaluationError(f"{name} must not contain duplicates")
    return ordered


def _partition(value: Any, name: str) -> str:
    if isinstance(value, EvidencePartition):
        return value.value
    text = _token(value, name) if type(value) is str and value else _text(value, name)
    try:
        return EvidencePartition(text).value
    except ValueError as exc:
        raise PolicyEvaluationError(
            f"{name} has unsupported partition {value!r}"
        ) from exc


def _rate_bp(numerator: int, denominator: int, *, empty: int = 0) -> int:
    """Integer basis-point ratio; empty denominator uses ``empty`` (fail-closed)."""

    ratio = ratio_to_basis_points(numerator, denominator)
    if ratio is None:
        return empty
    return ratio


def _median_int(values: Sequence[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    # Strict DAG-JSON: no floats; floor of even-length mean.
    return (ordered[mid - 1] + ordered[mid]) // 2


# ---------------------------------------------------------------------------
# Held-out benchmark models (evaluation inputs; pure data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeldOutCaseOutcome:
    """Per-case held-out outcome used solely for candidate scoring.

    Cases that probe critical omission or stale artifacts contribute to the
    corresponding rates. ``accepted_regression`` records a hidden accepted
    regression (selected/pass path that still regressed).
    """

    case_id: str
    case_cid: str
    partition: EvidencePartition | str
    critical_omission_present: bool = False
    critical_omission_detected: bool = False
    critical_omission_accepted: bool = False
    stale_artifact_present: bool = False
    stale_artifact_rejected: bool = False
    accepted_regression: bool = False
    context_reduction_bp: int = 0
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "case_id",
            "case_cid",
            "partition",
            "critical_omission_present",
            "critical_omission_detected",
            "critical_omission_accepted",
            "stale_artifact_present",
            "stale_artifact_rejected",
            "accepted_regression",
            "context_reduction_bp",
            "notes",
            "metadata",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _token(self.case_id, "case_id"))
        object.__setattr__(self, "case_cid", _cid(self.case_cid, "case_cid"))
        partition = _partition(self.partition, "partition")
        if partition != EvidencePartition.HELD_OUT.value:
            raise PolicyEvaluationError(
                "held-out case partition must be held_out and disjoint from "
                "calibration/development"
            )
        object.__setattr__(self, "partition", partition)
        object.__setattr__(
            self,
            "critical_omission_present",
            _bool(self.critical_omission_present, "critical_omission_present"),
        )
        object.__setattr__(
            self,
            "critical_omission_detected",
            _bool(self.critical_omission_detected, "critical_omission_detected"),
        )
        object.__setattr__(
            self,
            "critical_omission_accepted",
            _bool(self.critical_omission_accepted, "critical_omission_accepted"),
        )
        object.__setattr__(
            self,
            "stale_artifact_present",
            _bool(self.stale_artifact_present, "stale_artifact_present"),
        )
        object.__setattr__(
            self,
            "stale_artifact_rejected",
            _bool(self.stale_artifact_rejected, "stale_artifact_rejected"),
        )
        object.__setattr__(
            self,
            "accepted_regression",
            _bool(self.accepted_regression, "accepted_regression"),
        )
        object.__setattr__(
            self,
            "context_reduction_bp",
            _basis_points(self.context_reduction_bp, "context_reduction_bp"),
        )
        if self.critical_omission_detected and not self.critical_omission_present:
            raise PolicyEvaluationError(
                "critical_omission_detected requires critical_omission_present"
            )
        if self.critical_omission_accepted and not self.critical_omission_present:
            raise PolicyEvaluationError(
                "critical_omission_accepted requires critical_omission_present"
            )
        if self.critical_omission_accepted and self.critical_omission_detected:
            raise PolicyEvaluationError(
                "critical omission cannot be both detected and accepted"
            )
        if self.stale_artifact_rejected and not self.stale_artifact_present:
            raise PolicyEvaluationError(
                "stale_artifact_rejected requires stale_artifact_present"
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": HELD_OUT_CASE_OUTCOME_SCHEMA,
            "case_id": self.case_id,
            "case_cid": self.case_cid,
            "partition": self.partition,
            "critical_omission_present": self.critical_omission_present,
            "critical_omission_detected": self.critical_omission_detected,
            "critical_omission_accepted": self.critical_omission_accepted,
            "stale_artifact_present": self.stale_artifact_present,
            "stale_artifact_rejected": self.stale_artifact_rejected,
            "accepted_regression": self.accepted_regression,
            "context_reduction_bp": self.context_reduction_bp,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeldOutCaseOutcome":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        if payload.pop("schema") != HELD_OUT_CASE_OUTCOME_SCHEMA:
            raise PolicyEvaluationError(
                "unsupported HeldOutCaseOutcome schema version"
            )
        return cls(
            case_id=payload["case_id"],
            case_cid=payload["case_cid"],
            partition=payload["partition"],
            critical_omission_present=payload["critical_omission_present"],
            critical_omission_detected=payload["critical_omission_detected"],
            critical_omission_accepted=payload["critical_omission_accepted"],
            stale_artifact_present=payload["stale_artifact_present"],
            stale_artifact_rejected=payload["stale_artifact_rejected"],
            accepted_regression=payload["accepted_regression"],
            context_reduction_bp=payload["context_reduction_bp"],
            notes=payload["notes"],
            metadata=payload["metadata"],
        )


@dataclass(frozen=True, slots=True)
class HeldOutBenchmark:
    """Immutable held-out benchmark manifest for policy candidate evaluation.

    ``calibration_case_cids``, ``development_case_cids``, and
    ``candidate_generating_case_cids`` are the disjoint partition identities
    that must not appear among held-out case CIDs. Baseline rates are bound
    into the manifest so non-regression is reproducible without mutation.
    """

    benchmark_id: str
    partition: EvidencePartition | str
    case_outcomes: Sequence[HeldOutCaseOutcome]
    calibration_case_cids: Sequence[str] = ()
    development_case_cids: Sequence[str] = ()
    candidate_generating_case_cids: Sequence[str] = ()
    baseline_critical_omission_detection_bp: int = 9_500
    baseline_stale_rejection_rate_bp: int = 10_000
    baseline_accepted_regression_bp: int = 0
    baseline_policy_cid: str | None = None
    repository_state_cid: str | None = None
    context_pack_cid: str | None = None
    verification_bundle_cid: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "benchmark_id",
            "partition",
            "case_outcomes",
            "calibration_case_cids",
            "development_case_cids",
            "candidate_generating_case_cids",
            "baseline_critical_omission_detection_bp",
            "baseline_stale_rejection_rate_bp",
            "baseline_accepted_regression_bp",
            "baseline_policy_cid",
            "repository_state_cid",
            "context_pack_cid",
            "verification_bundle_cid",
            "notes",
            "metadata",
            "benchmark_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "benchmark_id", _token(self.benchmark_id, "benchmark_id")
        )
        partition = _partition(self.partition, "partition")
        if partition != EvidencePartition.HELD_OUT.value:
            raise PolicyEvaluationError(
                "held-out benchmark partition must be held_out and disjoint "
                "from calibration/development"
            )
        object.__setattr__(self, "partition", partition)
        if not isinstance(self.case_outcomes, (list, tuple)):
            raise PolicyEvaluationError("case_outcomes must be a list")
        if len(self.case_outcomes) > MAX_CASES:
            raise PolicyEvaluationError("case_outcomes exceeds maximum length")
        normalized_cases: list[HeldOutCaseOutcome] = []
        for item in self.case_outcomes:
            if isinstance(item, HeldOutCaseOutcome):
                normalized_cases.append(item)
            elif isinstance(item, Mapping):
                if "schema" in item:
                    normalized_cases.append(HeldOutCaseOutcome.from_dict(item))
                else:
                    normalized_cases.append(
                        HeldOutCaseOutcome(
                            case_id=item["case_id"],
                            case_cid=item["case_cid"],
                            partition=item.get("partition", EvidencePartition.HELD_OUT),
                            critical_omission_present=item.get(
                                "critical_omission_present", False
                            ),
                            critical_omission_detected=item.get(
                                "critical_omission_detected", False
                            ),
                            critical_omission_accepted=item.get(
                                "critical_omission_accepted", False
                            ),
                            stale_artifact_present=item.get(
                                "stale_artifact_present", False
                            ),
                            stale_artifact_rejected=item.get(
                                "stale_artifact_rejected", False
                            ),
                            accepted_regression=item.get("accepted_regression", False),
                            context_reduction_bp=item.get("context_reduction_bp", 0),
                            notes=item.get("notes"),
                            metadata=item.get("metadata", {}),
                        )
                    )
            else:
                raise PolicyEvaluationError(
                    "case_outcomes entries must be HeldOutCaseOutcome or mapping"
                )
        # Deterministic order by case_id then case_cid.
        cases = tuple(
            sorted(normalized_cases, key=lambda c: (c.case_id, c.case_cid))
        )
        object.__setattr__(self, "case_outcomes", cases)
        object.__setattr__(
            self,
            "calibration_case_cids",
            _unique_sorted_cids(
                list(self.calibration_case_cids), "calibration_case_cids"
            ),
        )
        object.__setattr__(
            self,
            "development_case_cids",
            _unique_sorted_cids(
                list(self.development_case_cids), "development_case_cids"
            ),
        )
        object.__setattr__(
            self,
            "candidate_generating_case_cids",
            _unique_sorted_cids(
                list(self.candidate_generating_case_cids),
                "candidate_generating_case_cids",
            ),
        )
        object.__setattr__(
            self,
            "baseline_critical_omission_detection_bp",
            _basis_points(
                self.baseline_critical_omission_detection_bp,
                "baseline_critical_omission_detection_bp",
            ),
        )
        object.__setattr__(
            self,
            "baseline_stale_rejection_rate_bp",
            _basis_points(
                self.baseline_stale_rejection_rate_bp,
                "baseline_stale_rejection_rate_bp",
            ),
        )
        object.__setattr__(
            self,
            "baseline_accepted_regression_bp",
            _basis_points(
                self.baseline_accepted_regression_bp,
                "baseline_accepted_regression_bp",
            ),
        )
        object.__setattr__(
            self,
            "baseline_policy_cid",
            _optional_cid(self.baseline_policy_cid, "baseline_policy_cid"),
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _optional_cid(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(
            self,
            "context_pack_cid",
            _optional_cid(self.context_pack_cid, "context_pack_cid"),
        )
        object.__setattr__(
            self,
            "verification_bundle_cid",
            _optional_cid(self.verification_bundle_cid, "verification_bundle_cid"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        # Immutable partition disjointness among declared identity sets.
        cal = set(self.calibration_case_cids)
        dev = set(self.development_case_cids)
        gen = set(self.candidate_generating_case_cids)
        if cal & dev:
            raise PolicyEvaluationError(
                "calibration and development case identities must be disjoint"
            )
        if cal & gen or dev & gen:
            # Generating cases may be a subset of cal/dev; allow subset but
            # still treat them as forbidden for held-out scoring below.
            pass

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": HELD_OUT_BENCHMARK_SCHEMA,
            "interface_id": HELD_OUT_BENCHMARK_INTERFACE,
            "benchmark_id": self.benchmark_id,
            "partition": self.partition,
            "case_outcomes": [case.identity_payload() for case in self.case_outcomes],
            "calibration_case_cids": list(self.calibration_case_cids),
            "development_case_cids": list(self.development_case_cids),
            "candidate_generating_case_cids": list(
                self.candidate_generating_case_cids
            ),
            "baseline_critical_omission_detection_bp": (
                self.baseline_critical_omission_detection_bp
            ),
            "baseline_stale_rejection_rate_bp": self.baseline_stale_rejection_rate_bp,
            "baseline_accepted_regression_bp": self.baseline_accepted_regression_bp,
            "baseline_policy_cid": self.baseline_policy_cid,
            "repository_state_cid": self.repository_state_cid,
            "context_pack_cid": self.context_pack_cid,
            "verification_bundle_cid": self.verification_bundle_cid,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def benchmark_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["benchmark_cid"] = self.benchmark_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeldOutBenchmark":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("benchmark_cid")
        if payload.pop("schema") != HELD_OUT_BENCHMARK_SCHEMA:
            raise PolicyEvaluationError(
                "unsupported HeldOutBenchmark schema version"
            )
        if payload.pop("interface_id") != HELD_OUT_BENCHMARK_INTERFACE:
            raise PolicyEvaluationError(
                "unsupported HeldOutBenchmark interface_id"
            )
        result = cls(
            benchmark_id=payload["benchmark_id"],
            partition=payload["partition"],
            case_outcomes=payload["case_outcomes"],
            calibration_case_cids=payload["calibration_case_cids"],
            development_case_cids=payload["development_case_cids"],
            candidate_generating_case_cids=payload[
                "candidate_generating_case_cids"
            ],
            baseline_critical_omission_detection_bp=payload[
                "baseline_critical_omission_detection_bp"
            ],
            baseline_stale_rejection_rate_bp=payload[
                "baseline_stale_rejection_rate_bp"
            ],
            baseline_accepted_regression_bp=payload[
                "baseline_accepted_regression_bp"
            ],
            baseline_policy_cid=payload["baseline_policy_cid"],
            repository_state_cid=payload["repository_state_cid"],
            context_pack_cid=payload["context_pack_cid"],
            verification_bundle_cid=payload["verification_bundle_cid"],
            notes=payload["notes"],
            metadata=payload["metadata"],
        )
        if claimed != result.benchmark_cid:
            raise PolicyEvaluationError(
                "HeldOutBenchmark benchmark_cid does not verify"
            )
        return result


@dataclass(frozen=True, slots=True)
class RuleEvaluationMetrics:
    """Aggregate metrics computed solely from held-out case outcomes."""

    case_count: int
    critical_omission_present_count: int
    critical_omission_detected_count: int
    critical_omission_accepted_count: int
    critical_omission_detection_bp: int
    stale_artifact_present_count: int
    stale_artifact_rejected_count: int
    stale_rejection_rate_bp: int
    accepted_regression_count: int
    accepted_regression_bp: int
    median_context_reduction_bp: int

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_METRICS_SCHEMA,
            "case_count": self.case_count,
            "critical_omission_present_count": self.critical_omission_present_count,
            "critical_omission_detected_count": self.critical_omission_detected_count,
            "critical_omission_accepted_count": self.critical_omission_accepted_count,
            "critical_omission_detection_bp": self.critical_omission_detection_bp,
            "stale_artifact_present_count": self.stale_artifact_present_count,
            "stale_artifact_rejected_count": self.stale_artifact_rejected_count,
            "stale_rejection_rate_bp": self.stale_rejection_rate_bp,
            "accepted_regression_count": self.accepted_regression_count,
            "accepted_regression_bp": self.accepted_regression_bp,
            "median_context_reduction_bp": self.median_context_reduction_bp,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_candidate(
    value: CompressionPolicyCandidate | Mapping[str, Any],
) -> CompressionPolicyCandidate:
    if isinstance(value, CompressionPolicyCandidate):
        return value
    if isinstance(value, Mapping):
        try:
            if "candidate_cid" in value and "schema" in value:
                return CompressionPolicyCandidate.from_dict(value)
            return CompressionPolicyCandidate(
                header=value["header"],
                candidate_id=value["candidate_id"],
                base_policy_cid=value["base_policy_cid"],
                base_policy_version=value["base_policy_version"],
                proposal_cid=value["proposal_cid"],
                proposed_policy_cid=value["proposed_policy_cid"],
                proposed_protected_thresholds=value["proposed_protected_thresholds"],
                baseline_protected_thresholds=value["baseline_protected_thresholds"],
                evaluation_partition=value.get(
                    "evaluation_partition", EvidencePartition.HELD_OUT
                ),
                external_authorization_cid=value.get("external_authorization_cid"),
                notes=value.get("notes"),
                metadata=value.get("metadata", {}),
            )
        except (PolicyContractError, SemanticGovernorBaseError, KeyError, TypeError) as exc:
            raise PolicyEvaluationError(
                f"candidate schema/integrity failure: {exc}"
            ) from exc
    raise PolicyEvaluationError(
        "candidate must be CompressionPolicyCandidate or mapping"
    )


def _normalize_benchmark(
    value: HeldOutBenchmark | Mapping[str, Any],
) -> HeldOutBenchmark:
    if isinstance(value, HeldOutBenchmark):
        return value
    if isinstance(value, Mapping):
        try:
            if "benchmark_cid" in value and "schema" in value:
                return HeldOutBenchmark.from_dict(value)
            return HeldOutBenchmark(
                benchmark_id=value["benchmark_id"],
                partition=value.get("partition", EvidencePartition.HELD_OUT),
                case_outcomes=value.get("case_outcomes", ()),
                calibration_case_cids=value.get("calibration_case_cids", ()),
                development_case_cids=value.get("development_case_cids", ()),
                candidate_generating_case_cids=value.get(
                    "candidate_generating_case_cids", ()
                ),
                baseline_critical_omission_detection_bp=value.get(
                    "baseline_critical_omission_detection_bp", 9_500
                ),
                baseline_stale_rejection_rate_bp=value.get(
                    "baseline_stale_rejection_rate_bp", 10_000
                ),
                baseline_accepted_regression_bp=value.get(
                    "baseline_accepted_regression_bp", 0
                ),
                baseline_policy_cid=value.get("baseline_policy_cid"),
                repository_state_cid=value.get("repository_state_cid"),
                context_pack_cid=value.get("context_pack_cid"),
                verification_bundle_cid=value.get("verification_bundle_cid"),
                notes=value.get("notes"),
                metadata=value.get("metadata", {}),
            )
        except (PolicyEvaluationError, KeyError, TypeError) as exc:
            raise PolicyEvaluationError(
                f"held_out_benchmark schema/integrity failure: {exc}"
            ) from exc
    raise PolicyEvaluationError(
        "held_out_benchmark must be HeldOutBenchmark or mapping"
    )


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
        except (PolicyContractError, SemanticGovernorBaseError) as exc:
            raise PolicyEvaluationError(
                f"baseline_policy schema/integrity failure: {exc}"
            ) from exc
    raise PolicyEvaluationError(
        "baseline_policy must be CompressionPolicy or mapping"
    )


def _normalize_thresholds(
    value: ProtectedThresholds | Mapping[str, Any],
) -> ProtectedThresholds:
    if isinstance(value, ProtectedThresholds):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value:
                return ProtectedThresholds.from_dict(value)
            defaults = ProtectedThresholds.default_production().to_dict()
            defaults.pop("schema")
            defaults.update(dict(value))
            return ProtectedThresholds(**defaults)  # type: ignore[arg-type]
        except (PolicyContractError, TypeError, ValueError) as exc:
            raise PolicyEvaluationError(str(exc)) from exc
    raise PolicyEvaluationError(
        "protected_thresholds must be ProtectedThresholds or mapping"
    )


# ---------------------------------------------------------------------------
# Metrics and integrity gates
# ---------------------------------------------------------------------------


def compute_held_out_metrics(
    cases: Sequence[HeldOutCaseOutcome],
) -> RuleEvaluationMetrics:
    """Aggregate held-out case outcomes into integer basis-point metrics."""

    if not isinstance(cases, (list, tuple)):
        raise PolicyEvaluationError("cases must be a list")
    omission_present = 0
    omission_detected = 0
    omission_accepted = 0
    stale_present = 0
    stale_rejected = 0
    regression_count = 0
    reductions: list[int] = []
    for case in cases:
        if not isinstance(case, HeldOutCaseOutcome):
            raise PolicyEvaluationError(
                "cases entries must be HeldOutCaseOutcome"
            )
        if case.critical_omission_present:
            omission_present += 1
            if case.critical_omission_detected:
                omission_detected += 1
            if case.critical_omission_accepted:
                omission_accepted += 1
        if case.stale_artifact_present:
            stale_present += 1
            if case.stale_artifact_rejected:
                stale_rejected += 1
        if case.accepted_regression:
            regression_count += 1
        reductions.append(case.context_reduction_bp)

    case_count = len(cases)
    return RuleEvaluationMetrics(
        case_count=case_count,
        critical_omission_present_count=omission_present,
        critical_omission_detected_count=omission_detected,
        critical_omission_accepted_count=omission_accepted,
        # Empty probe set is reported as 0 (fail-closed vs thresholds).
        critical_omission_detection_bp=_rate_bp(
            omission_detected, omission_present, empty=0
        ),
        stale_artifact_present_count=stale_present,
        stale_artifact_rejected_count=stale_rejected,
        stale_rejection_rate_bp=_rate_bp(
            stale_rejected, stale_present, empty=0
        ),
        accepted_regression_count=regression_count,
        accepted_regression_bp=_rate_bp(
            regression_count, case_count, empty=0
        ),
        median_context_reduction_bp=_median_int(reductions),
    )


def _partition_integrity_reasons(
    benchmark: HeldOutBenchmark,
) -> list[str]:
    """Return blocking reasons for missing/overlapping held-out data."""

    reasons: list[str] = []
    cases = benchmark.case_outcomes
    if not cases:
        reasons.append(REASON_MISSING_HELD_OUT)
        return reasons

    case_cids = [case.case_cid for case in cases]
    case_ids = [case.case_id for case in cases]
    if len(case_cids) != len(set(case_cids)) or len(case_ids) != len(set(case_ids)):
        reasons.append(REASON_DUPLICATE_CASE)

    for case in cases:
        if case.partition != EvidencePartition.HELD_OUT.value:
            reasons.append(REASON_PARTITION_NOT_HELD_OUT)
            break

    held = set(case_cids)
    calibration = set(benchmark.calibration_case_cids)
    development = set(benchmark.development_case_cids)
    generating = set(benchmark.candidate_generating_case_cids)

    if held & calibration or held & development:
        reasons.append(REASON_OVERLAP)
    if held & generating:
        reasons.append(REASON_CANDIDATE_GENERATING_OVERLAP)

    return reasons


def _threshold_blocking_reasons(
    metrics: RuleEvaluationMetrics,
    thresholds: ProtectedThresholds,
    benchmark: HeldOutBenchmark,
    *,
    high_risk_reduced: bool,
) -> list[str]:
    """Safety/non-regression gates over held-out metrics."""

    reasons: list[str] = []

    # Required probes: empty critical-omission or stale probe sets fail closed
    # when thresholds demand positive assurance (non-zero min rates).
    if (
        metrics.critical_omission_present_count == 0
        and thresholds.min_critical_omission_detection_bp > 0
    ):
        reasons.append(REASON_EMPTY_PROBE)
    if (
        metrics.stale_artifact_present_count == 0
        and thresholds.min_critical_omission_detection_bp > 0
    ):
        # Stale rejection is a separate safety surface; missing probes also
        # cannot prove non-regression of stale rejection.
        if REASON_EMPTY_PROBE not in reasons:
            reasons.append(REASON_EMPTY_PROBE)

    # Absolute threshold floors/ceilings from proposed protected thresholds.
    if (
        metrics.critical_omission_detection_bp
        < thresholds.min_critical_omission_detection_bp
    ):
        reasons.append(REASON_OMISSION_THRESHOLD)
    if (
        metrics.critical_omission_accepted_count
        > thresholds.max_critical_omission_accepted
    ):
        reasons.append(REASON_OMISSION_ACCEPTED)
    if metrics.accepted_regression_bp > thresholds.max_accepted_regression_bp:
        reasons.append(REASON_REGRESSION_THRESHOLD)
    if (
        metrics.median_context_reduction_bp
        < thresholds.min_median_context_reduction_bp
    ):
        reasons.append(REASON_CONTEXT_REDUCTION)
    if not thresholds.require_full_suite_fallback:
        reasons.append(REASON_FULL_SUITE_DISABLED)

    # Non-regression versus bound baseline rates (cannot get worse).
    if (
        metrics.critical_omission_detection_bp
        < benchmark.baseline_critical_omission_detection_bp
    ):
        reasons.append(REASON_OMISSION_REGRESSION)
    if metrics.stale_rejection_rate_bp < benchmark.baseline_stale_rejection_rate_bp:
        reasons.append(REASON_STALE_REGRESSION)

    # Hidden accepted regressions always block when any case accepted a
    # regression, even if the rate somehow rounded to zero with huge N.
    if metrics.accepted_regression_count > 0:
        reasons.append(REASON_HIDDEN_REGRESSION)
    if metrics.accepted_regression_bp > benchmark.baseline_accepted_regression_bp:
        if REASON_HIDDEN_REGRESSION not in reasons:
            reasons.append(REASON_HIDDEN_REGRESSION)

    if high_risk_reduced:
        reasons.append(REASON_HIGH_RISK_REDUCTION)

    # Stable unique order.
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return ordered


def _build_report_header(
    *,
    candidate: CompressionPolicyCandidate,
    benchmark: HeldOutBenchmark,
    baseline_policy_cid: str,
    terminal_status: GovernorTerminalStatus,
    assumptions: Sequence[GovernorAssumption],
) -> GovernorArtifactHeader:
    repo = (
        benchmark.repository_state_cid
        or candidate.header.repository_state_cid
    )
    context = (
        benchmark.context_pack_cid or candidate.header.context_pack_cid
    )
    bundle = (
        benchmark.verification_bundle_cid
        or candidate.header.verification_bundle_cid
    )
    return GovernorArtifactHeader(
        artifact_kind="rule_evaluation_report",
        repository_state_cid=repo,
        context_pack_cid=context,
        verification_bundle_cid=bundle,
        generator=GeneratorIdentity(
            generator_id=GENERATOR_ID,
            generator_version=GENERATOR_VERSION,
            interface_id=EVALUATE_RULE_CANDIDATE_INTERFACE,
        ),
        provenance=ArtifactProvenance(
            producer_id=PRODUCER_ID,
            producer_version=PRODUCER_VERSION,
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=tuple(
                sorted(
                    {
                        candidate.candidate_cid,
                        benchmark.benchmark_cid,
                        baseline_policy_cid,
                        candidate.proposal_cid,
                    }
                )
            ),
            tool_ids=(TOOL_ID,),
            policy_cid=baseline_policy_cid,
            notes=None,
        ),
        terminal_status=terminal_status,
        assumptions=tuple(assumptions),
        metadata={
            "evidence": SCG_HELD_OUT_EVALUATION_EVIDENCE,
            "benchmark_id": benchmark.benchmark_id,
            "candidate_id": candidate.candidate_id,
        },
    )


def _stable_report_id(candidate_cid: str, benchmark_cid: str) -> str:
    # Token-safe deterministic id fragment from content digests.
    digest = cid_for_structured(
        {
            "candidate_cid": candidate_cid,
            "benchmark_cid": benchmark_cid,
            "interface_id": EVALUATE_RULE_CANDIDATE_INTERFACE,
        }
    )
    # CIDs are base32; keep a short lowercase token suffix.
    suffix = re.sub(r"[^a-z0-9]", "", digest.lower())[-24:] or "0"
    return f"eval_{suffix}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_rule_candidate(
    candidate: CompressionPolicyCandidate | Mapping[str, Any],
    held_out_benchmark: HeldOutBenchmark | Mapping[str, Any],
    *,
    baseline_policy: CompressionPolicy | Mapping[str, Any] | None = None,
) -> RuleEvaluationReport:
    """Evaluate a policy/rule candidate solely on disjoint held-out evidence.

    Pure function: validates inputs, computes held-out metrics, applies
    safety/non-regression thresholds, and returns a content-addressed
    :class:`RuleEvaluationReport` without mutating any input.

    Parameters
    ----------
    candidate:
        :class:`CompressionPolicyCandidate` (or mapping) bound to held-out
        evaluation partition.
    held_out_benchmark:
        Immutable held-out benchmark manifest with case outcomes and the
        calibration/development/generating identities that must stay disjoint.
    baseline_policy:
        Optional full :class:`CompressionPolicy` for integrity cross-checks
        against ``candidate.base_policy_cid``.
    """

    try:
        cand = _normalize_candidate(candidate)
        bench = _normalize_benchmark(held_out_benchmark)
        policy = _normalize_policy(baseline_policy)
    except PolicyEvaluationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise PolicyEvaluationError(
            f"schema_or_integrity_failure: {exc}"
        ) from exc

    baseline_policy_cid = cand.base_policy_cid
    if policy is not None:
        if policy.policy_cid != cand.base_policy_cid:
            # Integrity failure: report rejected rather than raise, so callers
            # always receive a reproducible evaluation artifact when inputs
            # parse but fail gates. Still raise when identity is unusable.
            raise PolicyEvaluationError(
                "baseline_policy.policy_cid must match candidate.base_policy_cid"
            )
        baseline_policy_cid = policy.policy_cid
    if (
        bench.baseline_policy_cid is not None
        and bench.baseline_policy_cid != cand.base_policy_cid
    ):
        raise PolicyEvaluationError(
            "held_out_benchmark.baseline_policy_cid must match "
            "candidate.base_policy_cid"
        )

    if cand.evaluation_partition != EvidencePartition.HELD_OUT.value:
        raise PolicyEvaluationError(
            "candidate evaluation_partition must be held_out"
        )

    proposed = _normalize_thresholds(cand.proposed_protected_thresholds)
    baseline_thresholds = _normalize_thresholds(cand.baseline_protected_thresholds)

    reductions = protected_threshold_reductions(baseline_thresholds, proposed)
    # Candidate construction already requires auth for reductions; re-check
    # for the evaluation report flag. Authorized reductions still block a
    # pure "pass" because pass cannot claim high_risk_assurance_reduced.
    high_risk_reduced = bool(reductions)

    integrity_reasons = _partition_integrity_reasons(bench)
    if REASON_MISSING_HELD_OUT in integrity_reasons:
        metrics = RuleEvaluationMetrics(
            case_count=0,
            critical_omission_present_count=0,
            critical_omission_detected_count=0,
            critical_omission_accepted_count=0,
            critical_omission_detection_bp=0,
            stale_artifact_present_count=0,
            stale_artifact_rejected_count=0,
            stale_rejection_rate_bp=0,
            accepted_regression_count=0,
            accepted_regression_bp=0,
            median_context_reduction_bp=0,
        )
    else:
        metrics = compute_held_out_metrics(bench.case_outcomes)

    threshold_reasons = _threshold_blocking_reasons(
        metrics,
        proposed,
        bench,
        high_risk_reduced=high_risk_reduced,
    )

    blocking: list[str] = []
    for reason in integrity_reasons + threshold_reasons:
        if reason not in blocking:
            blocking.append(reason)
    if len(blocking) > MAX_BLOCKING_REASONS:
        blocking = blocking[:MAX_BLOCKING_REASONS]

    if not blocking:
        verdict = EvaluationVerdict.PASS
        terminal = GovernorTerminalStatus.COMPLETE
    else:
        # Integrity / overlap / missing data → rejected; other safety fails → fail.
        hard = {
            REASON_MISSING_HELD_OUT,
            REASON_OVERLAP,
            REASON_CANDIDATE_GENERATING_OVERLAP,
            REASON_PARTITION_NOT_HELD_OUT,
            REASON_SCHEMA_INTEGRITY,
            REASON_DUPLICATE_CASE,
            REASON_CANDIDATE_POLICY_MISMATCH,
        }
        if any(reason in hard for reason in blocking):
            verdict = EvaluationVerdict.REJECTED
            terminal = GovernorTerminalStatus.REJECTED
        else:
            verdict = EvaluationVerdict.FAIL
            terminal = GovernorTerminalStatus.COMPLETE

    assumptions = (
        GovernorAssumption(
            assumption_id="held_out_disjoint",
            kind=AssumptionKind.VERIFICATION,
            statement=(
                "Held-out case identities are disjoint from calibration, "
                "development, and candidate-generating identities"
            ),
            supporting_cids=(bench.benchmark_cid,),
        ),
        GovernorAssumption(
            assumption_id="evaluation_no_mutation",
            kind=AssumptionKind.OTHER,
            statement=(
                "evaluate_rule_candidate emits a report only; it does not "
                "mutate candidate, benchmark, or policy state"
            ),
            supporting_cids=(cand.candidate_cid,),
        ),
    )

    header = _build_report_header(
        candidate=cand,
        benchmark=bench,
        baseline_policy_cid=baseline_policy_cid,
        terminal_status=terminal,
        assumptions=assumptions,
    )
    report_id = _stable_report_id(cand.candidate_cid, bench.benchmark_cid)

    # declared_thresholds_applied is true whenever we applied the candidate's
    # proposed protected thresholds to the held-out metrics (always, by design).
    declared_thresholds_applied = True

    # Pass verdict contract: cannot claim high_risk_assurance_reduced.
    report_high_risk = high_risk_reduced and verdict != EvaluationVerdict.PASS.value
    if verdict == EvaluationVerdict.PASS:
        report_high_risk = False

    try:
        report = RuleEvaluationReport(
            header=header,
            report_id=report_id,
            candidate_cid=cand.candidate_cid,
            held_out_benchmark_cid=bench.benchmark_cid,
            baseline_policy_cid=baseline_policy_cid,
            partition=EvidencePartition.HELD_OUT,
            verdict=verdict,
            critical_omission_detection_bp=metrics.critical_omission_detection_bp,
            stale_rejection_rate_bp=metrics.stale_rejection_rate_bp,
            accepted_regression_bp=metrics.accepted_regression_bp,
            high_risk_assurance_reduced=report_high_risk,
            declared_thresholds_applied=declared_thresholds_applied,
            blocking_reasons=tuple(blocking),
            notes=None,
            metadata={
                "evidence": SCG_HELD_OUT_EVALUATION_EVIDENCE,
                "metrics": metrics.to_dict(),
                "protected_threshold_reductions": list(reductions),
                "benchmark_id": bench.benchmark_id,
                "candidate_id": cand.candidate_id,
                "case_count": metrics.case_count,
            },
        )
    except (PolicyContractError, SemanticGovernorBaseError) as exc:
        raise PolicyEvaluationError(
            f"failed to construct RuleEvaluationReport: {exc}"
        ) from exc

    return report


def verify_evaluation_report_identity(report: RuleEvaluationReport) -> str:
    """Recompute and return the report CID; raises if the claim does not match."""

    if not isinstance(report, RuleEvaluationReport):
        raise PolicyEvaluationError(
            "report must be a RuleEvaluationReport"
        )
    recomputed = report.report_cid
    # from_dict already verifies; expose a pure recompute helper for callers.
    restored = RuleEvaluationReport.from_dict(report.to_dict())
    if restored.report_cid != recomputed:
        raise PolicyEvaluationError("RuleEvaluationReport report_cid does not verify")
    return recomputed


__all__ = [
    "EVALUATE_RULE_CANDIDATE_INTERFACE",
    "EVALUATION_METRICS_SCHEMA",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "HELD_OUT_BENCHMARK_INTERFACE",
    "HELD_OUT_BENCHMARK_SCHEMA",
    "HELD_OUT_CASE_OUTCOME_SCHEMA",
    "HeldOutBenchmark",
    "HeldOutCaseOutcome",
    "PolicyEvaluationError",
    "REASON_CANDIDATE_GENERATING_OVERLAP",
    "REASON_CONTEXT_REDUCTION",
    "REASON_DUPLICATE_CASE",
    "REASON_EMPTY_PROBE",
    "REASON_FULL_SUITE_DISABLED",
    "REASON_HIDDEN_REGRESSION",
    "REASON_HIGH_RISK_REDUCTION",
    "REASON_MISSING_HELD_OUT",
    "REASON_OMISSION_ACCEPTED",
    "REASON_OMISSION_REGRESSION",
    "REASON_OMISSION_THRESHOLD",
    "REASON_OVERLAP",
    "REASON_PARTITION_NOT_HELD_OUT",
    "REASON_REGRESSION_THRESHOLD",
    "REASON_SCHEMA_INTEGRITY",
    "REASON_STALE_REGRESSION",
    "RuleEvaluationMetrics",
    "SCG_HELD_OUT_EVALUATION_EVIDENCE",
    "compute_held_out_metrics",
    "evaluate_rule_candidate",
    "verify_evaluation_report_identity",
]
