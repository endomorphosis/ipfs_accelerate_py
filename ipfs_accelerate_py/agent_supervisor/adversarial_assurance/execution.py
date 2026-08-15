"""Individual mutant execution and closed outcome classification (AAE-044).

Interface surface:

* ``execute_mutation@1`` — gate on an unmutated green (or explicitly blocked)
  baseline, run predicted detectors first, broaden only under explicit policy,
  and seal one honest ``MutationExecutionReceipt@1`` plus
  ``MutationOutcome@1``.
* ``classify_mutation_outcome@1`` — map observed detector evidence into exactly
  one closed terminal outcome status without false kill credit.

Normative properties (acceptance):

* Unmutated baseline must be green, or execution is explicitly blocked.
* Predicted checks always run before broader or full-suite fallback.
* Broader / full-suite fallback is policy-bound (never silent).
* Observed detectors and one closed terminal outcome are persisted honestly
  (observed ⊆ executed ⊆ selected; kill requires an observed detector).
* Invalid, uncompilable, infrastructure, timeout, inconclusive, equivalent,
  survival, and human-review outcomes never count as killed.
* No production policy change; cold import is side-effect free.

This module composes released authorities:

* AAE-040 baseline / detection-set planning concepts
* AAE-043 survivor broadening policy
* AAE-009 execution contracts (receipt + outcome)

It does not create worktrees, open stores, mutate production trees, or change
assurance policy.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.incremental import (
    BroadeningMode,
    MutantOutcomeClass,
    SurvivorBroadeningPolicy,
    resolve_broadening_mode,
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
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    CostMeasurement,
    DetectorClassification,
    DetectorKind,
    DetectorPrediction,
    DetectorStrength,
    ExecutionContractError,
    ExpectedDetectionSet,
    MutationExecutionPlan,
    MutationExecutionReceipt,
    MutationOutcome,
    MutationOutcomeStatus,
    assert_outcome_never_false_kill,
    counts_as_killed,
    killed_outcome_statuses,
    mutation_outcome_statuses,
    never_counted_as_killed_statuses,
    verify_detection_set_identity,
    verify_outcome_identity,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

EXECUTE_MUTATION_INTERFACE: Final[str] = "execute_mutation@1"
CLASSIFY_MUTATION_OUTCOME_INTERFACE: Final[str] = "classify_mutation_outcome@1"

MUTATION_EXECUTION_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "mutation-execution-report@1"
)
MUTATION_EXECUTION_REPORT_INTERFACE: Final[str] = "MutationExecutionReport@1"
EXECUTION_BASELINE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "execution-baseline@1"
)
DETECTOR_RUN_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "detector-run-observation@1"
)
OUTCOME_CLASSIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "outcome-classification@1"
)
FALLBACK_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "execution-fallback-policy@1"
)

AAE_OUTCOME_EVIDENCE: Final[str] = "aae/outcome@1"
ADAPTER_ID: Final[str] = "aae-mutation-execution"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
GENERATOR_ID: Final[str] = "mutation_execution"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_DETECTORS: Final[int] = 1_024
MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_COST_UNITS: Final[int] = 2**63 - 1
MAX_EXECUTION_SECONDS: Final[int] = 7 * 24 * 3_600
DEFAULT_TIMEOUT_SECONDS: Final[int] = 3_600
DEFAULT_DETECTOR_COST_UNITS: Final[int] = 1

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_REPOSITORY_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}$"
)

# Kill status ↔ detector kind (mirrors AAE-009 contracts).
_KIND_TO_KILL_STATUS: Final[Mapping[str, str]] = MappingProxyType(
    {
        DetectorKind.STATIC_RULE.value: (
            MutationOutcomeStatus.KILLED_BY_STATIC_ANALYSIS.value
        ),
        DetectorKind.TYPE_CHECK.value: (
            MutationOutcomeStatus.KILLED_BY_TYPE_CHECK.value
        ),
        DetectorKind.UNIT_TEST.value: MutationOutcomeStatus.KILLED_BY_TEST.value,
        DetectorKind.INTEGRATION_TEST.value: (
            MutationOutcomeStatus.KILLED_BY_TEST.value
        ),
        DetectorKind.PROPERTY_TEST.value: MutationOutcomeStatus.KILLED_BY_TEST.value,
        DetectorKind.FORMAL_OBLIGATION.value: (
            MutationOutcomeStatus.KILLED_BY_FORMAL_PROOF.value
        ),
        DetectorKind.INCREMENTAL_SEAL.value: (
            MutationOutcomeStatus.KILLED_BY_FORMAL_PROOF.value
        ),
        DetectorKind.POLICY_RULE.value: MutationOutcomeStatus.KILLED_BY_POLICY.value,
        DetectorKind.RUNTIME_INVARIANT.value: (
            MutationOutcomeStatus.KILLED_BY_RUNTIME_INVARIANT.value
        ),
        DetectorKind.FULL_SUITE.value: (
            MutationOutcomeStatus.KILLED_BY_FULL_SUITE.value
        ),
    }
)

FULL_SUITE_DETECTOR_ID: Final[str] = "full_suite"
FULL_SUITE_DETECTOR_PREFIX: Final[str] = "full_suite."

REASON_BASELINE_GREEN: Final[str] = "baseline_green"
REASON_BASELINE_BLOCKED: Final[str] = "baseline_blocked"
REASON_BASELINE_MISSING: Final[str] = "baseline_missing"
REASON_PREDICTED_FIRST: Final[str] = "predicted_checks_first"
REASON_FALLBACK_POLICY_BOUND: Final[str] = "fallback_policy_bound"
REASON_FALLBACK_DISABLED: Final[str] = "fallback_disabled"
REASON_FALLBACK_SKIPPED_KILL: Final[str] = "fallback_skipped_after_kill"
REASON_BROADER_APPLIED: Final[str] = "broader_fallback_applied"
REASON_FULL_SUITE_APPLIED: Final[str] = "full_suite_fallback_applied"
REASON_ONE_TERMINAL_OUTCOME: Final[str] = "one_closed_terminal_outcome"
REASON_OBSERVED_HONEST: Final[str] = "observed_detectors_honest"
REASON_NO_PRODUCTION_POLICY_CHANGE: Final[str] = "production_policy_unchanged"
REASON_DISPOSABLE_WORKTREE_REQUIRED: Final[str] = "disposable_worktree_required"
REASON_NETWORK_DISABLED_REQUIRED: Final[str] = "network_disabled_required"


# ---------------------------------------------------------------------------
# Errors and closed enums
# ---------------------------------------------------------------------------


class MutationExecutionError(ValueError):
    """Raised when mutant execution inputs are malformed or unsafe."""

    def __init__(self, message: str, *, reason_code: str = "malformed_input") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class BaselineGateError(MutationExecutionError):
    """Raised when baseline is neither green nor explicitly blocked."""


class FallbackPolicyError(MutationExecutionError):
    """Raised when fallback would expand outside explicit policy."""


class BaselineGateStatus(str, Enum):
    """Closed baseline gate vocabulary for individual mutant execution."""

    GREEN = "green"
    BLOCKED = "blocked"


class DetectorRunStatus(str, Enum):
    """Closed observation status for one detector execution attempt."""

    PASSED = "passed"  # ran clean; mutant not detected
    DETECTED = "detected"  # mutant observed / killed by this detector
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ExecutionDisposition(str, Enum):
    """High-level disposition of one execute_mutation call."""

    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class ExecutionPhase(str, Enum):
    """Ordered execution phases recorded on the report."""

    BASELINE = "baseline"
    PREDICTED = "predicted"
    BROADER = "broader"
    FULL_SUITE = "full_suite"
    CLASSIFIED = "classified"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise MutationExecutionError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise MutationExecutionError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise MutationExecutionError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, empty=False)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise MutationExecutionError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise MutationExecutionError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise MutationExecutionError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int = MAX_COST_UNITS) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise MutationExecutionError(f"{name} must be a nonnegative integer")
    if value > maximum:
        raise MutationExecutionError(f"{name} exceeds maximum")
    return value


def _pos_int(value: Any, name: str, *, maximum: int = MAX_COST_UNITS) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise MutationExecutionError(f"{name} must be a positive integer")
    if value > maximum:
        raise MutationExecutionError(f"{name} exceeds maximum")
    return value


def _repository_id(value: Any, name: str = "repository_id") -> str:
    text = _text(value, name)
    if _REPOSITORY_ID_RE.fullmatch(text) is None:
        raise MutationExecutionError(
            f"{name} must match repository identity pattern"
        )
    return text


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise MutationExecutionError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _unique_tokens(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int = MAX_DETECTORS,
    sort: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise MutationExecutionError(f"{name} must be a list or tuple")
    tokens = [_token(item, name) for item in values]
    if len(tokens) > maximum:
        raise MutationExecutionError(f"{name} exceeds maximum length")
    if len(tokens) != len(set(tokens)):
        raise MutationExecutionError(f"{name} must not contain duplicates")
    if sort:
        return tuple(sorted(tokens))
    return tuple(tokens)


def _stable_unique(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        if isinstance(value, enum_type):
            return value.value  # type: ignore[return-value]
        return enum_type(value).value  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise MutationExecutionError(
            f"{name} has unsupported value {value!r}"
        ) from exc


def _is_full_suite_detector_id(detector_id: str) -> bool:
    return detector_id == FULL_SUITE_DETECTOR_ID or detector_id.startswith(
        FULL_SUITE_DETECTOR_PREFIX
    )


def _kind_value(kind: DetectorKind | str) -> str:
    return _enum(kind, DetectorKind, "detector_kind")


# ---------------------------------------------------------------------------
# Execution baseline gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionBaseline:
    """Unmutated baseline observation required before mutant execution.

    Green baselines must be unmutated with complete green verification.
    Blocked baselines must carry an explicit block reason and never pretend to
    be green. Missing or incomplete baselines fail closed.
    """

    baseline_receipt_cid: str
    repository_id: str
    repository_state_cid: str
    status: BaselineGateStatus | str
    unmutated: bool = True
    verification_green: bool = True
    observation_complete: bool = True
    block_reason: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EXECUTION_BASELINE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_receipt_cid",
            _cid(self.baseline_receipt_cid, "baseline_receipt_cid"),
        )
        object.__setattr__(
            self, "repository_id", _repository_id(self.repository_id)
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _cid(self.repository_state_cid, "repository_state_cid"),
        )
        status = _enum(self.status, BaselineGateStatus, "status")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "unmutated", _bool(self.unmutated, "unmutated"))
        object.__setattr__(
            self,
            "verification_green",
            _bool(self.verification_green, "verification_green"),
        )
        object.__setattr__(
            self,
            "observation_complete",
            _bool(self.observation_complete, "observation_complete"),
        )
        object.__setattr__(
            self, "block_reason", _optional_text(self.block_reason, "block_reason")
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        object.__setattr__(
            self, "schema", _text(self.schema, "schema")
        )

        if not self.observation_complete:
            raise BaselineGateError(
                "baseline observation must be complete",
                reason_code=REASON_BASELINE_MISSING,
            )
        if status == BaselineGateStatus.GREEN.value:
            if not self.unmutated:
                raise BaselineGateError(
                    "green baseline must be unmutated",
                    reason_code="baseline_mutated",
                )
            if not self.verification_green:
                raise BaselineGateError(
                    "green baseline verification must be green",
                    reason_code="baseline_not_green",
                )
            if self.block_reason is not None:
                raise BaselineGateError(
                    "green baseline must not set block_reason",
                    reason_code="baseline_green_with_block_reason",
                )
        elif status == BaselineGateStatus.BLOCKED.value:
            if self.block_reason is None:
                raise BaselineGateError(
                    "blocked baseline requires an explicit block_reason",
                    reason_code=REASON_BASELINE_BLOCKED,
                )
            # Blocked baselines are honest: they are not claimed green.
            if self.verification_green:
                raise BaselineGateError(
                    "blocked baseline must not claim verification_green",
                    reason_code=REASON_BASELINE_BLOCKED,
                )
        else:  # pragma: no cover - enum already closed
            raise BaselineGateError(
                f"unsupported baseline status {status!r}",
                reason_code=REASON_BASELINE_MISSING,
            )

    @property
    def is_green(self) -> bool:
        return self.status == BaselineGateStatus.GREEN.value

    @property
    def is_blocked(self) -> bool:
        return self.status == BaselineGateStatus.BLOCKED.value

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "baseline_receipt_cid": self.baseline_receipt_cid,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "status": self.status,
            "unmutated": self.unmutated,
            "verification_green": self.verification_green,
            "observation_complete": self.observation_complete,
            "block_reason": self.block_reason,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @property
    def baseline_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["baseline_cid"] = self.baseline_cid
        return payload

    @classmethod
    def from_value(cls, value: Any) -> "ExecutionBaseline":
        if isinstance(value, ExecutionBaseline):
            return value
        if not isinstance(value, Mapping):
            raise MutationExecutionError(
                "baseline must be ExecutionBaseline or mapping",
                reason_code=REASON_BASELINE_MISSING,
            )
        return cls(
            baseline_receipt_cid=value["baseline_receipt_cid"],
            repository_id=value["repository_id"],
            repository_state_cid=value["repository_state_cid"],
            status=value["status"],
            unmutated=value.get("unmutated", True),
            verification_green=value.get("verification_green", True),
            observation_complete=value.get("observation_complete", True),
            block_reason=value.get("block_reason"),
            notes=value.get("notes"),
            metadata=value.get("metadata") or {},
        )


def evaluate_baseline_gate(
    baseline: ExecutionBaseline | Mapping[str, Any],
) -> tuple[ExecutionBaseline, tuple[str, ...]]:
    """Seal baseline and return reason codes for green or explicit block.

    Raises :class:`BaselineGateError` when the baseline is neither green nor
    explicitly blocked, or when observation is incomplete.
    """

    sealed = ExecutionBaseline.from_value(baseline)
    if sealed.is_green:
        return sealed, (REASON_BASELINE_GREEN,)
    if sealed.is_blocked:
        return sealed, (REASON_BASELINE_BLOCKED,)
    raise BaselineGateError(
        "baseline must be green or explicitly blocked",
        reason_code=REASON_BASELINE_MISSING,
    )


# ---------------------------------------------------------------------------
# Detector observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectorRunObservation:
    """Honest observation from executing one detector against a mutant."""

    detector_id: str
    detector_kind: DetectorKind | str
    status: DetectorRunStatus | str
    phase: ExecutionPhase | str = ExecutionPhase.PREDICTED
    cost_units: int = DEFAULT_DETECTOR_COST_UNITS
    execution_seconds: int = 0
    diagnostic: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = DETECTOR_RUN_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "detector_id", _token(self.detector_id, "detector_id")
        )
        object.__setattr__(
            self,
            "detector_kind",
            _kind_value(self.detector_kind),
        )
        object.__setattr__(
            self, "status", _enum(self.status, DetectorRunStatus, "status")
        )
        object.__setattr__(
            self, "phase", _enum(self.phase, ExecutionPhase, "phase")
        )
        object.__setattr__(
            self,
            "cost_units",
            _nonneg_int(self.cost_units, "cost_units", maximum=MAX_COST_UNITS),
        )
        object.__setattr__(
            self,
            "execution_seconds",
            _nonneg_int(
                self.execution_seconds,
                "execution_seconds",
                maximum=MAX_EXECUTION_SECONDS,
            ),
        )
        object.__setattr__(
            self, "diagnostic", _optional_text(self.diagnostic, "diagnostic")
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))

    @property
    def detected(self) -> bool:
        return self.status == DetectorRunStatus.DETECTED.value

    @property
    def executed(self) -> bool:
        return self.status in {
            DetectorRunStatus.PASSED.value,
            DetectorRunStatus.DETECTED.value,
            DetectorRunStatus.ERROR.value,
            DetectorRunStatus.TIMEOUT.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "detector_id": self.detector_id,
            "detector_kind": self.detector_kind,
            "status": self.status,
            "phase": self.phase,
            "cost_units": self.cost_units,
            "execution_seconds": self.execution_seconds,
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_value(cls, value: Any) -> "DetectorRunObservation":
        if isinstance(value, DetectorRunObservation):
            return value
        if not isinstance(value, Mapping):
            raise MutationExecutionError(
                "observation must be DetectorRunObservation or mapping"
            )
        return cls(
            detector_id=value["detector_id"],
            detector_kind=value["detector_kind"],
            status=value["status"],
            phase=value.get("phase", ExecutionPhase.PREDICTED),
            cost_units=value.get("cost_units", DEFAULT_DETECTOR_COST_UNITS),
            execution_seconds=value.get("execution_seconds", 0),
            diagnostic=value.get("diagnostic"),
            metadata=value.get("metadata") or {},
        )


DetectorRunner = Callable[[str, DetectorKind | str, ExecutionPhase], DetectorRunObservation]


# ---------------------------------------------------------------------------
# Fallback policy (policy-bound broader execution)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionFallbackPolicy:
    """Policy controlling broader / full-suite fallback after predicted checks.

    Fallback never runs silently: every expansion is attributed to an explicit
    policy flag or risk/uncertainty trigger, and predicted checks always run
    first.
    """

    schema: str = FALLBACK_POLICY_SCHEMA
    enable_broader_fallback: bool = True
    enable_full_suite_fallback: bool = False
    full_suite_on_high_risk: bool = True
    full_suite_on_uncertainty: bool = True
    high_risk_classes: tuple[str, ...] = (
        "critical_security",
        "authorization",
        "proof_receipt_trust",
    )
    always_full_suite: bool = False
    max_broader_detectors: int = MAX_DETECTORS
    require_disposable_worktree: bool = True
    require_network_disabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self,
            "enable_broader_fallback",
            _bool(self.enable_broader_fallback, "enable_broader_fallback"),
        )
        object.__setattr__(
            self,
            "enable_full_suite_fallback",
            _bool(self.enable_full_suite_fallback, "enable_full_suite_fallback"),
        )
        object.__setattr__(
            self,
            "full_suite_on_high_risk",
            _bool(self.full_suite_on_high_risk, "full_suite_on_high_risk"),
        )
        object.__setattr__(
            self,
            "full_suite_on_uncertainty",
            _bool(self.full_suite_on_uncertainty, "full_suite_on_uncertainty"),
        )
        object.__setattr__(
            self,
            "high_risk_classes",
            _unique_tokens(
                list(self.high_risk_classes),
                "high_risk_classes",
                maximum=256,
                sort=True,
            ),
        )
        object.__setattr__(
            self,
            "always_full_suite",
            _bool(self.always_full_suite, "always_full_suite"),
        )
        object.__setattr__(
            self,
            "max_broader_detectors",
            _nonneg_int(
                self.max_broader_detectors,
                "max_broader_detectors",
                maximum=MAX_DETECTORS,
            ),
        )
        for flag_name in (
            "require_disposable_worktree",
            "require_network_disabled",
        ):
            flag = _bool(getattr(self, flag_name), flag_name)
            if not flag:
                raise MutationExecutionError(
                    f"{flag_name} must be true",
                    reason_code=flag_name,
                )
            object.__setattr__(self, flag_name, flag)

    def to_survivor_policy(self) -> SurvivorBroadeningPolicy:
        return SurvivorBroadeningPolicy(
            broaden_survivors=self.enable_broader_fallback,
            full_suite_on_high_risk=self.full_suite_on_high_risk,
            full_suite_on_uncertainty=self.full_suite_on_uncertainty,
            high_risk_classes=self.high_risk_classes,
            always_full_suite=self.always_full_suite,
            max_broader_units=self.max_broader_detectors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "enable_broader_fallback": self.enable_broader_fallback,
            "enable_full_suite_fallback": self.enable_full_suite_fallback,
            "full_suite_on_high_risk": self.full_suite_on_high_risk,
            "full_suite_on_uncertainty": self.full_suite_on_uncertainty,
            "high_risk_classes": list(self.high_risk_classes),
            "always_full_suite": self.always_full_suite,
            "max_broader_detectors": self.max_broader_detectors,
            "require_disposable_worktree": self.require_disposable_worktree,
            "require_network_disabled": self.require_network_disabled,
        }

    @classmethod
    def from_value(cls, value: Any) -> "ExecutionFallbackPolicy":
        if value is None:
            return cls()
        if isinstance(value, ExecutionFallbackPolicy):
            return value
        if isinstance(value, SurvivorBroadeningPolicy):
            return cls(
                enable_broader_fallback=value.broaden_survivors,
                enable_full_suite_fallback=value.always_full_suite,
                full_suite_on_high_risk=value.full_suite_on_high_risk,
                full_suite_on_uncertainty=value.full_suite_on_uncertainty,
                high_risk_classes=value.high_risk_classes,
                always_full_suite=value.always_full_suite,
                max_broader_detectors=value.max_broader_units,
            )
        if not isinstance(value, Mapping):
            raise MutationExecutionError(
                "fallback_policy must be ExecutionFallbackPolicy or mapping"
            )
        return cls(
            enable_broader_fallback=value.get("enable_broader_fallback", True),
            enable_full_suite_fallback=value.get(
                "enable_full_suite_fallback", False
            ),
            full_suite_on_high_risk=value.get("full_suite_on_high_risk", True),
            full_suite_on_uncertainty=value.get(
                "full_suite_on_uncertainty", True
            ),
            high_risk_classes=tuple(value.get("high_risk_classes") or ()),
            always_full_suite=value.get("always_full_suite", False),
            max_broader_detectors=value.get(
                "max_broader_detectors", MAX_DETECTORS
            ),
            require_disposable_worktree=value.get(
                "require_disposable_worktree", True
            ),
            require_network_disabled=value.get(
                "require_network_disabled", True
            ),
        )


def resolve_execution_fallback(
    *,
    survived_predicted: bool,
    risk_class: str = "",
    uncertainty: bool = False,
    policy: ExecutionFallbackPolicy | Mapping[str, Any] | None = None,
) -> tuple[BroadeningMode, tuple[str, ...]]:
    """Resolve broader / full-suite mode under explicit fallback policy.

    Predicted checks are assumed already complete. When the mutant was already
    killed, fallback is skipped. When policy disables broadening, mode is
    ``none`` with an explicit reason.
    """

    sealed = ExecutionFallbackPolicy.from_value(policy)
    reasons: list[str] = [REASON_FALLBACK_POLICY_BOUND]

    if not survived_predicted:
        return BroadeningMode.NONE, (
            REASON_FALLBACK_POLICY_BOUND,
            REASON_FALLBACK_SKIPPED_KILL,
        )

    # Direct full-suite enablement from execution policy.
    if sealed.always_full_suite or sealed.enable_full_suite_fallback:
        if sealed.always_full_suite:
            return BroadeningMode.FULL_SUITE, (
                REASON_FALLBACK_POLICY_BOUND,
                REASON_FULL_SUITE_APPLIED,
            )
        # enable_full_suite_fallback alone still requires a trigger via the
        # survivor broadening path or risk/uncertainty — unless always.
        # Fall through to composed AAE-043 resolution with full-suite allowed.

    survivor_policy = SurvivorBroadeningPolicy(
        broaden_survivors=sealed.enable_broader_fallback,
        full_suite_on_high_risk=sealed.full_suite_on_high_risk
        and sealed.enable_full_suite_fallback,
        full_suite_on_uncertainty=sealed.full_suite_on_uncertainty
        and sealed.enable_full_suite_fallback,
        high_risk_classes=sealed.high_risk_classes,
        always_full_suite=sealed.always_full_suite,
        max_broader_units=sealed.max_broader_detectors,
    )
    mode, broaden_reasons = resolve_broadening_mode(
        mutant_outcome=MutantOutcomeClass.SURVIVOR,
        risk_class=risk_class,
        uncertainty=uncertainty,
        policy=survivor_policy,
    )
    reasons.extend(broaden_reasons)

    if mode is BroadeningMode.FULL_SUITE:
        if not (
            sealed.enable_full_suite_fallback
            or sealed.always_full_suite
            or sealed.full_suite_on_high_risk
            or sealed.full_suite_on_uncertainty
        ):
            raise FallbackPolicyError(
                "full-suite fallback resolved without policy authorization",
                reason_code="full_suite_unauthorized",
            )
        reasons.append(REASON_FULL_SUITE_APPLIED)
        return BroadeningMode.FULL_SUITE, _stable_unique(reasons)

    if mode is BroadeningMode.BROADER:
        if not sealed.enable_broader_fallback:
            return BroadeningMode.NONE, _stable_unique(
                [REASON_FALLBACK_POLICY_BOUND, REASON_FALLBACK_DISABLED]
            )
        reasons.append(REASON_BROADER_APPLIED)
        return BroadeningMode.BROADER, _stable_unique(reasons)

    if not sealed.enable_broader_fallback and not sealed.enable_full_suite_fallback:
        reasons.append(REASON_FALLBACK_DISABLED)
    return BroadeningMode.NONE, _stable_unique(reasons)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomeClassification:
    """Pure classification result: one closed terminal outcome + detector roles."""

    schema: str = OUTCOME_CLASSIFICATION_SCHEMA
    interface_id: str = CLASSIFY_MUTATION_OUTCOME_INTERFACE
    outcome_status: str = MutationOutcomeStatus.INCONCLUSIVE.value
    detector_classification: DetectorClassification | None = None
    killing_detector_id: str | None = None
    killing_detector_kind: str | None = None
    counts_as_killed: bool = False
    full_suite_executed: bool = False
    reason_codes: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        status = _enum(
            self.outcome_status, MutationOutcomeStatus, "outcome_status"
        )
        object.__setattr__(self, "outcome_status", status)
        if self.detector_classification is None:
            classification = DetectorClassification(
                predicted_detector_ids=(),
                selected_detector_ids=(),
                executed_detector_ids=(),
                observed_detector_ids=(),
            )
        elif isinstance(self.detector_classification, DetectorClassification):
            classification = self.detector_classification
        else:
            raise MutationExecutionError(
                "detector_classification must be DetectorClassification"
            )
        object.__setattr__(self, "detector_classification", classification)

        killing_id = self.killing_detector_id
        if killing_id is not None:
            killing_id = _token(killing_id, "killing_detector_id")
        object.__setattr__(self, "killing_detector_id", killing_id)

        killing_kind = self.killing_detector_kind
        if killing_kind is not None:
            killing_kind = _kind_value(killing_kind)
        object.__setattr__(self, "killing_detector_kind", killing_kind)

        killed = counts_as_killed(status)
        object.__setattr__(self, "counts_as_killed", killed)
        object.__setattr__(
            self,
            "full_suite_executed",
            _bool(self.full_suite_executed, "full_suite_executed"),
        )
        codes: list[str] = []
        for raw in self.reason_codes or (REASON_ONE_TERMINAL_OUTCOME,):
            if not isinstance(raw, str) or not raw:
                raise MutationExecutionError("reason_codes must be nonempty strings")
            codes.append(raw)
        object.__setattr__(self, "reason_codes", _stable_unique(codes))
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))

        if killed:
            if killing_id is None or killing_kind is None:
                raise MutationExecutionError(
                    "killed outcomes require killing detector id and kind"
                )
            if killing_id not in classification.observed_detector_ids:
                raise MutationExecutionError(
                    "killing_detector_id must be among observed_detector_ids"
                )
        else:
            if killing_id is not None or killing_kind is not None:
                raise MutationExecutionError(
                    "non-killed outcomes must not set killing detector fields"
                )
        if status in never_counted_as_killed_statuses() and killed:
            raise MutationExecutionError(
                f"outcome_status {status!r} must never count as killed"
            )

    def to_dict(self) -> dict[str, Any]:
        assert self.detector_classification is not None
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "outcome_status": self.outcome_status,
            "detector_classification": self.detector_classification.to_dict(),
            "killing_detector_id": self.killing_detector_id,
            "killing_detector_kind": self.killing_detector_kind,
            "counts_as_killed": self.counts_as_killed,
            "full_suite_executed": self.full_suite_executed,
            "reason_codes": list(self.reason_codes),
            "notes": self.notes,
        }


def _build_detector_classification(
    *,
    predicted_detector_ids: Sequence[str],
    selected_detector_ids: Sequence[str],
    executed_detector_ids: Sequence[str],
    observed_detector_ids: Sequence[str],
) -> DetectorClassification:
    """Build a sealed detector classification with honest nesting invariants."""

    predicted = _unique_tokens(
        list(predicted_detector_ids), "predicted_detector_ids", sort=True
    )
    # Preserve selection/execution honesty: subsets enforced by contract.
    selected = _unique_tokens(
        list(selected_detector_ids), "selected_detector_ids", sort=True
    )
    executed = _unique_tokens(
        list(executed_detector_ids), "executed_detector_ids", sort=True
    )
    observed = _unique_tokens(
        list(observed_detector_ids), "observed_detector_ids", sort=True
    )
    try:
        return DetectorClassification(
            predicted_detector_ids=predicted,
            selected_detector_ids=selected,
            executed_detector_ids=executed,
            observed_detector_ids=observed,
        )
    except ExecutionContractError as exc:
        raise MutationExecutionError(
            f"dishonest detector classification: {exc}",
            reason_code="dishonest_detectors",
        ) from exc


def classify_mutation_outcome(
    *,
    predicted_detector_ids: Sequence[str],
    selected_detector_ids: Sequence[str],
    observations: Sequence[DetectorRunObservation | Mapping[str, Any]] = (),
    executed_detector_ids: Sequence[str] | None = None,
    observed_detector_ids: Sequence[str] | None = None,
    detector_kinds: Mapping[str, DetectorKind | str] | None = None,
    full_suite_executed: bool = False,
    invalid_mutant: bool = False,
    uncompilable: bool = False,
    infrastructure_ok: bool = True,
    timed_out: bool = False,
    equivalence_status: str | None = None,
    human_review_required: bool = False,
    baseline_blocked: bool = False,
    notes: str | None = None,
) -> OutcomeClassification:
    """Classify one closed terminal mutation outcome (``classify_mutation_outcome@1``).

    Priority (fail-closed specials first, then honest kill/survival):

    1. baseline blocked → ``inconclusive``
    2. invalid mutant / uncompilable / infrastructure / timeout
    3. equivalence (equivalent / probably_equivalent)
    4. human review required
    5. first observed detection in execution order → kill by detector kind
    6. full suite executed with no detection → survived_full_verification
    7. selected execution with no detection → survived_selected_verification
    8. otherwise → inconclusive

    Observed detectors are those with status ``detected``. Kill credit is never
    granted without an observed detector.
    """

    predicted = _unique_tokens(
        list(predicted_detector_ids), "predicted_detector_ids", sort=True
    )
    selected = _unique_tokens(
        list(selected_detector_ids), "selected_detector_ids", sort=True
    )

    sealed_obs = [DetectorRunObservation.from_value(item) for item in observations]
    kinds: dict[str, str] = {}
    if detector_kinds:
        for key, value in detector_kinds.items():
            kinds[_token(key, "detector_kinds")] = _kind_value(value)
    for obs in sealed_obs:
        kinds.setdefault(obs.detector_id, str(obs.detector_kind))

    # Execution order from observations (predicted-first is caller's duty).
    executed_ordered: list[str] = []
    observed_ordered: list[str] = []
    for obs in sealed_obs:
        if obs.executed and obs.detector_id not in executed_ordered:
            executed_ordered.append(obs.detector_id)
        if obs.detected and obs.detector_id not in observed_ordered:
            observed_ordered.append(obs.detector_id)

    if executed_detector_ids is not None:
        executed_ordered = list(
            _unique_tokens(
                list(executed_detector_ids),
                "executed_detector_ids",
                sort=False,
            )
        )
    if observed_detector_ids is not None:
        observed_ordered = list(
            _unique_tokens(
                list(observed_detector_ids),
                "observed_detector_ids",
                sort=False,
            )
        )

    # Ensure selected covers executed (honesty).
    selected_set = set(selected)
    for detector_id in executed_ordered:
        if detector_id not in selected_set:
            selected = tuple(sorted(set(selected) | {detector_id}))
            selected_set.add(detector_id)

    classification = _build_detector_classification(
        predicted_detector_ids=predicted,
        selected_detector_ids=selected,
        executed_detector_ids=executed_ordered,
        observed_detector_ids=observed_ordered,
    )

    reasons: list[str] = [
        REASON_ONE_TERMINAL_OUTCOME,
        REASON_OBSERVED_HONEST,
    ]
    full_suite = _bool(full_suite_executed, "full_suite_executed")

    # --- fail-closed special statuses ---
    if _bool(baseline_blocked, "baseline_blocked"):
        return OutcomeClassification(
            outcome_status=MutationOutcomeStatus.INCONCLUSIVE.value,
            detector_classification=classification,
            full_suite_executed=full_suite,
            reason_codes=tuple(
                _stable_unique([*reasons, REASON_BASELINE_BLOCKED])
            ),
            notes=notes or "baseline explicitly blocked; mutant not executed",
        )

    if _bool(invalid_mutant, "invalid_mutant"):
        return OutcomeClassification(
            outcome_status=MutationOutcomeStatus.INVALID_MUTANT.value,
            detector_classification=classification,
            full_suite_executed=full_suite,
            reason_codes=tuple(_stable_unique([*reasons, "invalid_mutant"])),
            notes=notes,
        )

    if _bool(uncompilable, "uncompilable"):
        return OutcomeClassification(
            outcome_status=MutationOutcomeStatus.UNCOMPILABLE.value,
            detector_classification=classification,
            full_suite_executed=full_suite,
            reason_codes=tuple(_stable_unique([*reasons, "uncompilable"])),
            notes=notes,
        )

    if not _bool(infrastructure_ok, "infrastructure_ok"):
        return OutcomeClassification(
            outcome_status=MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value,
            detector_classification=classification,
            full_suite_executed=full_suite,
            reason_codes=tuple(
                _stable_unique([*reasons, "infrastructure_failure"])
            ),
            notes=notes,
        )

    if _bool(timed_out, "timed_out"):
        return OutcomeClassification(
            outcome_status=MutationOutcomeStatus.TIMEOUT.value,
            detector_classification=classification,
            full_suite_executed=full_suite,
            reason_codes=tuple(_stable_unique([*reasons, "timeout"])),
            notes=notes,
        )

    if equivalence_status is not None:
        eq = _text(equivalence_status, "equivalence_status")
        if eq == MutationOutcomeStatus.EQUIVALENT.value:
            return OutcomeClassification(
                outcome_status=MutationOutcomeStatus.EQUIVALENT.value,
                detector_classification=classification,
                full_suite_executed=full_suite,
                reason_codes=tuple(_stable_unique([*reasons, "equivalent"])),
                notes=notes,
            )
        if eq == MutationOutcomeStatus.PROBABLY_EQUIVALENT.value:
            return OutcomeClassification(
                outcome_status=MutationOutcomeStatus.PROBABLY_EQUIVALENT.value,
                detector_classification=classification,
                full_suite_executed=full_suite,
                reason_codes=tuple(
                    _stable_unique([*reasons, "probably_equivalent"])
                ),
                notes=notes,
            )
        # not_equivalent / unknown fall through to observation-based classification.
        if eq not in {"not_equivalent", "unknown"}:
            raise MutationExecutionError(
                f"unsupported equivalence_status {eq!r}"
            )

    if _bool(human_review_required, "human_review_required"):
        return OutcomeClassification(
            outcome_status=MutationOutcomeStatus.HUMAN_REVIEW_REQUIRED.value,
            detector_classification=classification,
            full_suite_executed=full_suite,
            reason_codes=tuple(
                _stable_unique([*reasons, "human_review_required"])
            ),
            notes=notes,
        )

    # --- kill from first honest observation in execution order ---
    if observed_ordered:
        killer_id = observed_ordered[0]
        kind = kinds.get(killer_id)
        if kind is None:
            raise MutationExecutionError(
                f"missing detector_kind for observed detector {killer_id!r}"
            )
        if kind == DetectorKind.HUMAN_REVIEW.value:
            return OutcomeClassification(
                outcome_status=MutationOutcomeStatus.HUMAN_REVIEW_REQUIRED.value,
                detector_classification=classification,
                full_suite_executed=full_suite,
                reason_codes=tuple(
                    _stable_unique([*reasons, "human_review_observed"])
                ),
                notes=notes,
            )
        kill_status = _KIND_TO_KILL_STATUS.get(kind)
        if kill_status is None:
            raise MutationExecutionError(
                f"detector_kind {kind!r} cannot produce a kill status"
            )
        # Full-suite detector ids force full-suite kill status when observed
        # during full-suite phase even if kind is a test kind? Keep kind map.
        if full_suite and _is_full_suite_detector_id(killer_id):
            kill_status = MutationOutcomeStatus.KILLED_BY_FULL_SUITE.value
            kind = DetectorKind.FULL_SUITE.value
        return OutcomeClassification(
            outcome_status=kill_status,
            detector_classification=classification,
            killing_detector_id=killer_id,
            killing_detector_kind=kind,
            full_suite_executed=full_suite,
            reason_codes=tuple(_stable_unique([*reasons, "killed"])),
            notes=notes,
        )

    # --- survival ---
    if full_suite and executed_ordered:
        return OutcomeClassification(
            outcome_status=(
                MutationOutcomeStatus.SURVIVED_FULL_VERIFICATION.value
            ),
            detector_classification=classification,
            full_suite_executed=True,
            reason_codes=tuple(
                _stable_unique([*reasons, "survived_full_verification"])
            ),
            notes=notes,
        )

    if executed_ordered:
        return OutcomeClassification(
            outcome_status=(
                MutationOutcomeStatus.SURVIVED_SELECTED_VERIFICATION.value
            ),
            detector_classification=classification,
            full_suite_executed=full_suite,
            reason_codes=tuple(
                _stable_unique([*reasons, "survived_selected_verification"])
            ),
            notes=notes,
        )

    return OutcomeClassification(
        outcome_status=MutationOutcomeStatus.INCONCLUSIVE.value,
        detector_classification=classification,
        full_suite_executed=full_suite,
        reason_codes=tuple(_stable_unique([*reasons, "inconclusive"])),
        notes=notes or "no detectors executed",
    )


# ---------------------------------------------------------------------------
# Sealed execution report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MutationExecutionReport:
    """Sealed report from ``execute_mutation@1`` for one admitted mutant."""

    schema: str = MUTATION_EXECUTION_REPORT_SCHEMA
    interface_id: str = MUTATION_EXECUTION_REPORT_INTERFACE
    report_cid: str = ""
    candidate_id: str = ""
    candidate_cid: str = ""
    mutant_identity_cid: str = ""
    disposition: ExecutionDisposition | str = ExecutionDisposition.COMPLETED
    baseline: ExecutionBaseline | None = None
    detection_set_cid: str = ""
    execution_plan_cid: str = ""
    phases: tuple[str, ...] = ()
    observations: tuple[DetectorRunObservation, ...] = ()
    classification: OutcomeClassification | None = None
    receipt: MutationExecutionReceipt | None = None
    outcome: MutationOutcome | None = None
    broadening_mode: BroadeningMode | str = BroadeningMode.NONE
    fallback_policy: ExecutionFallbackPolicy | None = None
    reason_codes: tuple[str, ...] = ()
    evidence_subset: str = AAE_OUTCOME_EVIDENCE
    production_policy_changed: bool = False
    diagnostic: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        object.__setattr__(
            self, "candidate_id", _token(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self,
            "mutant_identity_cid",
            _cid(self.mutant_identity_cid, "mutant_identity_cid"),
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ExecutionDisposition, "disposition"),
        )
        if self.baseline is not None and not isinstance(
            self.baseline, ExecutionBaseline
        ):
            raise MutationExecutionError("baseline must be ExecutionBaseline")
        object.__setattr__(
            self,
            "detection_set_cid",
            _cid(self.detection_set_cid, "detection_set_cid")
            if self.detection_set_cid
            else "",
        )
        object.__setattr__(
            self,
            "execution_plan_cid",
            _cid(self.execution_plan_cid, "execution_plan_cid")
            if self.execution_plan_cid
            else "",
        )
        phases = tuple(
            _enum(phase, ExecutionPhase, "phases") for phase in (self.phases or ())
        )
        object.__setattr__(self, "phases", phases)
        observations = tuple(
            DetectorRunObservation.from_value(item)
            for item in (self.observations or ())
        )
        object.__setattr__(self, "observations", observations)
        if self.classification is not None and not isinstance(
            self.classification, OutcomeClassification
        ):
            raise MutationExecutionError(
                "classification must be OutcomeClassification"
            )
        if self.receipt is not None and not isinstance(
            self.receipt, MutationExecutionReceipt
        ):
            raise MutationExecutionError(
                "receipt must be MutationExecutionReceipt"
            )
        if self.outcome is not None and not isinstance(
            self.outcome, MutationOutcome
        ):
            raise MutationExecutionError("outcome must be MutationOutcome")
        if not isinstance(self.broadening_mode, BroadeningMode):
            object.__setattr__(
                self,
                "broadening_mode",
                BroadeningMode(
                    _enum(self.broadening_mode, BroadeningMode, "broadening_mode")
                ),
            )
        if self.fallback_policy is not None and not isinstance(
            self.fallback_policy, ExecutionFallbackPolicy
        ):
            raise MutationExecutionError(
                "fallback_policy must be ExecutionFallbackPolicy"
            )
        codes = _stable_unique(
            [
                code
                for code in (self.reason_codes or ())
                if isinstance(code, str) and code
            ]
        )
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(
            self,
            "evidence_subset",
            _text(self.evidence_subset, "evidence_subset"),
        )
        # Hard invariant: never claim production policy change.
        object.__setattr__(self, "production_policy_changed", False)
        object.__setattr__(
            self,
            "diagnostic",
            _clip(_text(self.diagnostic, "diagnostic", empty=True), limit=MAX_DIAGNOSTIC)
            if self.diagnostic
            else "",
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if not self.report_cid:
            object.__setattr__(self, "report_cid", self.compute_report_cid())

    def compute_report_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "candidate_id": self.candidate_id,
            "candidate_cid": self.candidate_cid,
            "mutant_identity_cid": self.mutant_identity_cid,
            "disposition": self.disposition
            if isinstance(self.disposition, str)
            else self.disposition.value,
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "detection_set_cid": self.detection_set_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "phases": list(self.phases),
            "observations": [item.to_dict() for item in self.observations],
            "classification": (
                None
                if self.classification is None
                else self.classification.to_dict()
            ),
            "receipt_cid": (
                None if self.receipt is None else self.receipt.receipt_cid
            ),
            "outcome_cid": (
                None if self.outcome is None else self.outcome.outcome_cid
            ),
            "outcome_status": (
                None
                if self.outcome is None
                else self.outcome.outcome_status
            ),
            "broadening_mode": (
                self.broadening_mode.value
                if isinstance(self.broadening_mode, BroadeningMode)
                else self.broadening_mode
            ),
            "fallback_policy": (
                None
                if self.fallback_policy is None
                else self.fallback_policy.to_dict()
            ),
            "reason_codes": list(self.reason_codes),
            "evidence_subset": self.evidence_subset,
            "production_policy_changed": False,
            "diagnostic": self.diagnostic,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["report_cid"] = self.report_cid
        if self.receipt is not None:
            payload["receipt"] = self.receipt.to_dict()
        if self.outcome is not None:
            payload["outcome"] = self.outcome.to_dict()
        return payload


# ---------------------------------------------------------------------------
# Header / plan / cost helpers
# ---------------------------------------------------------------------------


def _default_generator() -> GeneratorIdentity:
    return GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=EXECUTE_MUTATION_INTERFACE,
    )


def _default_versions(
    *,
    operator_id: str = "mutation_execution",
    operator_version: str = "1",
    campaign_policy_id: str = "default_campaign",
    campaign_policy_version: str = "1.0.0",
) -> VersionBinding:
    return VersionBinding(
        operator_id=operator_id,
        operator_version=operator_version,
        campaign_policy_id=campaign_policy_id,
        campaign_policy_version=campaign_policy_version,
        generator=_default_generator(),
    )


def _default_provenance(
    *,
    policy_cid: str,
    input_cids: Sequence[str] = (),
) -> ArtifactProvenance:
    return ArtifactProvenance(
        producer_id="adversarial_assurance",
        producer_version="1",
        execution_mode=ExecutionMode.LIVE,
        authority_source=AuthoritySource.OBSERVED,
        input_cids=tuple(input_cids) or (policy_cid,),
        tool_ids=("mutation_executor.v1",),
        policy_cid=policy_cid,
        notes=None,
    )


def build_execution_header(
    artifact_kind: str,
    *,
    repository_id: str,
    repository_state_cid: str,
    environment_cid: str,
    dependency_lock_cid: str,
    policy_cid: str,
    target_symbol_ids: Sequence[str] = (),
    target_artifact_cids: Sequence[str] = (),
    capsule_cids: Sequence[str] = (),
    proof_unit_cids: Sequence[str] = (),
    receipt_cids: Sequence[str] = (),
    proof_cids: Sequence[str] = (),
    terminal_status: AssuranceTerminalStatus | str = AssuranceTerminalStatus.COMPLETE,
    metadata: Mapping[str, Any] | None = None,
    versions: VersionBinding | None = None,
    provenance: ArtifactProvenance | None = None,
) -> AssuranceArtifactHeader:
    """Build a sealed assurance header for execution artifacts."""

    return AssuranceArtifactHeader(
        artifact_kind=artifact_kind,
        repository_id=repository_id,
        repository_state_cid=repository_state_cid,
        target_symbol_ids=tuple(target_symbol_ids) or ("mutant.target",),
        target_artifact_cids=tuple(target_artifact_cids)
        or (cid_for_structured({"kind": "target", "id": "default"}),),
        capsule_cids=tuple(capsule_cids)
        or (cid_for_structured({"kind": "capsule", "id": "default"}),),
        proof_unit_cids=tuple(proof_unit_cids)
        or (cid_for_structured({"kind": "proof_unit", "id": "default"}),),
        environment_cid=environment_cid,
        dependency_lock_cid=dependency_lock_cid,
        versions=versions or _default_versions(),
        provenance=provenance
        or _default_provenance(policy_cid=policy_cid),
        terminal_status=terminal_status,
        receipt_cids=tuple(receipt_cids),
        proof_cids=tuple(proof_cids),
        metadata=dict(metadata or {}),
    )


def _normalize_detection_set(
    value: ExpectedDetectionSet | Mapping[str, Any],
) -> ExpectedDetectionSet:
    if isinstance(value, ExpectedDetectionSet):
        sealed = value
    elif isinstance(value, Mapping):
        try:
            sealed = ExpectedDetectionSet.from_dict(value)
        except (ExecutionContractError, AssuranceBaseError, KeyError, TypeError) as exc:
            raise MutationExecutionError(
                f"invalid expected detection set: {exc}"
            ) from exc
    else:
        raise MutationExecutionError(
            "expected_detection_set must be ExpectedDetectionSet or mapping"
        )
    try:
        verify_detection_set_identity(sealed)
    except ExecutionContractError as exc:
        raise MutationExecutionError(str(exc)) from exc
    return sealed


def _normalize_execution_plan(
    value: MutationExecutionPlan | Mapping[str, Any] | None,
    *,
    detection_set: ExpectedDetectionSet,
    header: AssuranceArtifactHeader | None,
    full_suite_fallback_enabled: bool,
    timeout_seconds: int,
) -> MutationExecutionPlan:
    if isinstance(value, MutationExecutionPlan):
        return value
    if isinstance(value, Mapping):
        try:
            return MutationExecutionPlan.from_dict(value)
        except (ExecutionContractError, AssuranceBaseError, KeyError, TypeError) as exc:
            raise MutationExecutionError(
                f"invalid mutation execution plan: {exc}"
            ) from exc

    plan_header = header
    if plan_header is None:
        plan_header = detection_set.header
    # Rebuild header with correct artifact_kind when needed.
    if plan_header.artifact_kind != "mutation_execution_plan":
        plan_header = AssuranceArtifactHeader(
            artifact_kind="mutation_execution_plan",
            repository_id=plan_header.repository_id,
            repository_state_cid=plan_header.repository_state_cid,
            target_symbol_ids=plan_header.target_symbol_ids,
            target_artifact_cids=plan_header.target_artifact_cids,
            capsule_cids=plan_header.capsule_cids,
            proof_unit_cids=plan_header.proof_unit_cids,
            environment_cid=plan_header.environment_cid,
            dependency_lock_cid=plan_header.dependency_lock_cid,
            versions=plan_header.versions,
            provenance=plan_header.provenance,
            terminal_status=plan_header.terminal_status,
            receipt_cids=plan_header.receipt_cids,
            proof_cids=plan_header.proof_cids,
            metadata=dict(plan_header.metadata),
        )
    predicted = list(detection_set.predicted_detector_ids)
    plan_id_candidate = f"exec_plan_{detection_set.candidate_id}"[:120]
    plan_id = (
        plan_id_candidate
        if _TOKEN_RE.fullmatch(plan_id_candidate)
        else "exec_plan_default"
    )
    return MutationExecutionPlan(
        header=plan_header,
        execution_plan_id=plan_id,
        candidate_id=detection_set.candidate_id,
        candidate_cid=detection_set.candidate_cid,
        expected_detection_set_cid=detection_set.detection_set_cid,
        selected_detector_ids=predicted,
        predicted_detector_ids=predicted,
        require_disposable_worktree=True,
        require_network_disabled=True,
        full_suite_fallback_enabled=full_suite_fallback_enabled,
        timeout_seconds=timeout_seconds,
    )


def _measure_cost(
    observations: Sequence[DetectorRunObservation],
    *,
    full_suite_counterfactual_units: int | None = None,
) -> CostMeasurement:
    incremental = sum(obs.cost_units for obs in observations if obs.executed)
    seconds = sum(obs.execution_seconds for obs in observations if obs.executed)
    counterfactual = (
        full_suite_counterfactual_units
        if full_suite_counterfactual_units is not None
        else max(incremental, incremental * 10 if incremental else 0)
    )
    if counterfactual < incremental:
        counterfactual = incremental
    return CostMeasurement(
        incremental_cost_units=int(incremental),
        full_suite_counterfactual_cost_units=int(counterfactual),
        execution_seconds=int(seconds),
    )


def _header_for_kind(
    template: AssuranceArtifactHeader,
    artifact_kind: str,
    *,
    terminal_status: AssuranceTerminalStatus | str | None = None,
    receipt_cids: Sequence[str] | None = None,
) -> AssuranceArtifactHeader:
    return AssuranceArtifactHeader(
        artifact_kind=artifact_kind,
        repository_id=template.repository_id,
        repository_state_cid=template.repository_state_cid,
        target_symbol_ids=template.target_symbol_ids,
        target_artifact_cids=template.target_artifact_cids,
        capsule_cids=template.capsule_cids,
        proof_unit_cids=template.proof_unit_cids,
        environment_cid=template.environment_cid,
        dependency_lock_cid=template.dependency_lock_cid,
        versions=template.versions,
        provenance=template.provenance,
        terminal_status=terminal_status or template.terminal_status,
        receipt_cids=tuple(receipt_cids)
        if receipt_cids is not None
        else template.receipt_cids,
        proof_cids=template.proof_cids,
        metadata=dict(template.metadata),
    )


def _prediction_index(
    detection_set: ExpectedDetectionSet,
) -> dict[str, DetectorPrediction]:
    return {
        detector.detector_id: detector
        for detector in detection_set.predicted_detectors
    }


def _run_detector_list(
    detector_ids: Sequence[str],
    *,
    predictions: Mapping[str, DetectorPrediction],
    kinds: Mapping[str, str],
    phase: ExecutionPhase,
    runner: DetectorRunner | None,
    supplied: Mapping[str, DetectorRunObservation],
    already_executed: set[str],
) -> list[DetectorRunObservation]:
    """Run detectors in given order, skipping already-executed ids."""

    out: list[DetectorRunObservation] = []
    for detector_id in detector_ids:
        if detector_id in already_executed:
            continue
        kind = kinds.get(detector_id)
        if kind is None and detector_id in predictions:
            kind = str(predictions[detector_id].detector_kind)
        if kind is None:
            if _is_full_suite_detector_id(detector_id):
                kind = DetectorKind.FULL_SUITE.value
            else:
                raise MutationExecutionError(
                    f"unknown detector_kind for {detector_id!r}"
                )
        if detector_id in supplied:
            obs = supplied[detector_id]
            # Force phase label for honesty of phase ordering.
            obs = DetectorRunObservation(
                detector_id=obs.detector_id,
                detector_kind=obs.detector_kind,
                status=obs.status,
                phase=phase,
                cost_units=obs.cost_units,
                execution_seconds=obs.execution_seconds,
                diagnostic=obs.diagnostic,
                metadata=dict(obs.metadata),
            )
        elif runner is not None:
            obs = DetectorRunObservation.from_value(
                runner(detector_id, kind, phase)
            )
            if obs.detector_id != detector_id:
                raise MutationExecutionError(
                    "detector runner returned mismatched detector_id"
                )
            obs = DetectorRunObservation(
                detector_id=obs.detector_id,
                detector_kind=obs.detector_kind or kind,
                status=obs.status,
                phase=phase,
                cost_units=obs.cost_units,
                execution_seconds=obs.execution_seconds,
                diagnostic=obs.diagnostic,
                metadata=dict(obs.metadata),
            )
        else:
            raise MutationExecutionError(
                f"no observation or runner for detector {detector_id!r}",
                reason_code="missing_observation",
            )
        out.append(obs)
        already_executed.add(detector_id)
        # Stop early on detection within the current phase batch; caller
        # decides whether to broaden. We still return the killing observation.
        if obs.detected:
            break
    return out


# ---------------------------------------------------------------------------
# Public: execute_mutation
# ---------------------------------------------------------------------------


def execute_mutation(
    *,
    candidate_id: str,
    candidate_cid: str,
    mutant_identity_cid: str,
    baseline: ExecutionBaseline | Mapping[str, Any],
    expected_detection_set: ExpectedDetectionSet | Mapping[str, Any],
    execution_plan: MutationExecutionPlan | Mapping[str, Any] | None = None,
    fallback_policy: ExecutionFallbackPolicy | Mapping[str, Any] | None = None,
    detector_runner: DetectorRunner | None = None,
    observations: Sequence[DetectorRunObservation | Mapping[str, Any]] = (),
    broader_detector_ids: Sequence[str] = (),
    full_suite_detector_ids: Sequence[str] = (),
    risk_class: str = "",
    uncertainty: bool = False,
    invalid_mutant: bool = False,
    uncompilable: bool = False,
    infrastructure_ok: bool = True,
    timed_out: bool = False,
    equivalence_status: str | None = None,
    equivalence_assessment_cid: str | None = None,
    human_review_required: bool = False,
    full_suite_counterfactual_cost_units: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    header_template: AssuranceArtifactHeader | None = None,
    receipt_id: str | None = None,
    outcome_id: str | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MutationExecutionReport:
    """Execute one admitted mutant and classify a closed terminal outcome.

    Interface: ``execute_mutation@1``

    Pipeline:

    1. Gate on unmutated green baseline or explicit block.
    2. Run *predicted* detectors first (stable order from detection set).
    3. On survival, broaden only when :class:`ExecutionFallbackPolicy` authorizes
       broader or full-suite expansion (policy-bound; never silent).
    4. Classify exactly one closed terminal outcome from honest observations.
    5. Seal ``MutationExecutionReceipt@1`` and ``MutationOutcome@1``.

    Production policy is never changed. Missing observations without a runner
    fail closed.
    """

    cand_id = _token(candidate_id, "candidate_id")
    cand_cid = _cid(candidate_cid, "candidate_cid")
    mutant_cid = _cid(mutant_identity_cid, "mutant_identity_cid")
    meta = dict(metadata or {})
    note_text = _optional_text(notes, "notes")
    policy = ExecutionFallbackPolicy.from_value(fallback_policy)
    phases: list[str] = [ExecutionPhase.BASELINE.value]
    reasons: list[str] = [
        REASON_NO_PRODUCTION_POLICY_CHANGE,
        REASON_DISPOSABLE_WORKTREE_REQUIRED,
        REASON_NETWORK_DISABLED_REQUIRED,
    ]

    sealed_baseline, baseline_reasons = evaluate_baseline_gate(baseline)
    reasons.extend(baseline_reasons)

    detection_set = _normalize_detection_set(expected_detection_set)
    if detection_set.candidate_id != cand_id:
        raise MutationExecutionError(
            "expected_detection_set.candidate_id must match candidate_id"
        )
    if detection_set.candidate_cid != cand_cid:
        raise MutationExecutionError(
            "expected_detection_set.candidate_cid must match candidate_cid"
        )

    plan = _normalize_execution_plan(
        execution_plan,
        detection_set=detection_set,
        header=header_template,
        full_suite_fallback_enabled=(
            policy.enable_full_suite_fallback or policy.always_full_suite
        ),
        timeout_seconds=_pos_int(
            timeout_seconds, "timeout_seconds", maximum=MAX_EXECUTION_SECONDS
        ),
    )
    if plan.candidate_id != cand_id or plan.candidate_cid != cand_cid:
        raise MutationExecutionError(
            "execution_plan candidate identity must match execute_mutation inputs"
        )
    if plan.expected_detection_set_cid != detection_set.detection_set_cid:
        raise MutationExecutionError(
            "execution_plan.expected_detection_set_cid must match detection set"
        )

    predictions = _prediction_index(detection_set)
    predicted_ids = list(detection_set.predicted_detector_ids)
    # Stable predicted order from detection set declaration order.
    predicted_ids = list(dict.fromkeys(predicted_ids))

    kinds: dict[str, str] = {
        detector_id: str(pred.detector_kind)
        for detector_id, pred in predictions.items()
    }

    # Index supplied observations by detector_id (last write wins).
    supplied: dict[str, DetectorRunObservation] = {}
    for raw in observations:
        obs = DetectorRunObservation.from_value(raw)
        supplied[obs.detector_id] = obs
        kinds.setdefault(obs.detector_id, str(obs.detector_kind))

    broader_ids = list(
        _unique_tokens(
            list(broader_detector_ids),
            "broader_detector_ids",
            sort=False,
        )
    ) if broader_detector_ids else []
    full_suite_ids = list(
        _unique_tokens(
            list(full_suite_detector_ids),
            "full_suite_detector_ids",
            sort=False,
        )
    ) if full_suite_detector_ids else []
    # Default full-suite marker when policy may expand.
    if not full_suite_ids and (
        policy.enable_full_suite_fallback or policy.always_full_suite
    ):
        full_suite_ids = [FULL_SUITE_DETECTOR_ID]
        kinds.setdefault(FULL_SUITE_DETECTOR_ID, DetectorKind.FULL_SUITE.value)

    for detector_id in broader_ids:
        kinds.setdefault(detector_id, DetectorKind.UNIT_TEST.value)
    for detector_id in full_suite_ids:
        kinds.setdefault(
            detector_id,
            DetectorKind.FULL_SUITE.value
            if _is_full_suite_detector_id(detector_id)
            else DetectorKind.UNIT_TEST.value,
        )

    template = header_template or detection_set.header

    # --- blocked baseline short-circuit ---
    if sealed_baseline.is_blocked:
        phases.append(ExecutionPhase.CLASSIFIED.value)
        classification = classify_mutation_outcome(
            predicted_detector_ids=predicted_ids,
            selected_detector_ids=predicted_ids,
            observations=(),
            full_suite_executed=False,
            baseline_blocked=True,
            notes=note_text
            or f"baseline blocked: {sealed_baseline.block_reason}",
        )
        reasons.extend(classification.reason_codes)
        reasons.append(REASON_BASELINE_BLOCKED)
        # Still seal receipt + outcome for honest persistence.
        receipt, outcome = _seal_receipt_and_outcome(
            template=template,
            candidate_id=cand_id,
            candidate_cid=cand_cid,
            mutant_identity_cid=mutant_cid,
            detection_set=detection_set,
            plan=plan,
            classification=classification,
            observations=(),
            infrastructure_ok=True,
            timed_out=False,
            equivalence_assessment_cid=None,
            full_suite_counterfactual_cost_units=full_suite_counterfactual_cost_units,
            receipt_id=receipt_id,
            outcome_id=outcome_id,
            notes=classification.notes,
            metadata=meta,
        )
        return MutationExecutionReport(
            candidate_id=cand_id,
            candidate_cid=cand_cid,
            mutant_identity_cid=mutant_cid,
            disposition=ExecutionDisposition.BLOCKED,
            baseline=sealed_baseline,
            detection_set_cid=detection_set.detection_set_cid,
            execution_plan_cid=plan.execution_plan_cid,
            phases=tuple(phases),
            observations=(),
            classification=classification,
            receipt=receipt,
            outcome=outcome,
            broadening_mode=BroadeningMode.NONE,
            fallback_policy=policy,
            reason_codes=_stable_unique(reasons),
            diagnostic=_clip(
                sealed_baseline.block_reason or "baseline blocked"
            ),
            metadata=meta,
        )

    # --- special pre-check outcomes (invalid / uncompilable / infra) ---
    if invalid_mutant or uncompilable or not infrastructure_ok:
        phases.append(ExecutionPhase.CLASSIFIED.value)
        classification = classify_mutation_outcome(
            predicted_detector_ids=predicted_ids,
            selected_detector_ids=predicted_ids,
            observations=(),
            invalid_mutant=invalid_mutant,
            uncompilable=uncompilable,
            infrastructure_ok=infrastructure_ok,
            timed_out=False,
            notes=note_text,
        )
        reasons.extend(classification.reason_codes)
        receipt, outcome = _seal_receipt_and_outcome(
            template=template,
            candidate_id=cand_id,
            candidate_cid=cand_cid,
            mutant_identity_cid=mutant_cid,
            detection_set=detection_set,
            plan=plan,
            classification=classification,
            observations=(),
            infrastructure_ok=infrastructure_ok,
            timed_out=False,
            equivalence_assessment_cid=equivalence_assessment_cid,
            full_suite_counterfactual_cost_units=full_suite_counterfactual_cost_units,
            receipt_id=receipt_id,
            outcome_id=outcome_id,
            notes=note_text,
            metadata=meta,
        )
        return MutationExecutionReport(
            candidate_id=cand_id,
            candidate_cid=cand_cid,
            mutant_identity_cid=mutant_cid,
            disposition=ExecutionDisposition.COMPLETED,
            baseline=sealed_baseline,
            detection_set_cid=detection_set.detection_set_cid,
            execution_plan_cid=plan.execution_plan_cid,
            phases=tuple(phases),
            observations=(),
            classification=classification,
            receipt=receipt,
            outcome=outcome,
            broadening_mode=BroadeningMode.NONE,
            fallback_policy=policy,
            reason_codes=_stable_unique(reasons),
            metadata=meta,
        )

    # --- phase: predicted checks first ---
    phases.append(ExecutionPhase.PREDICTED.value)
    reasons.append(REASON_PREDICTED_FIRST)
    already: set[str] = set()
    all_observations: list[DetectorRunObservation] = []
    selected_ids: list[str] = list(predicted_ids)

    predicted_obs = _run_detector_list(
        predicted_ids,
        predictions=predictions,
        kinds=kinds,
        phase=ExecutionPhase.PREDICTED,
        runner=detector_runner,
        supplied=supplied,
        already_executed=already,
    )
    all_observations.extend(predicted_obs)

    killed_after_predicted = any(obs.detected for obs in predicted_obs)
    infra_failed = any(
        obs.status == DetectorRunStatus.ERROR.value for obs in predicted_obs
    )
    timed = timed_out or any(
        obs.status == DetectorRunStatus.TIMEOUT.value for obs in predicted_obs
    )

    broadening_mode = BroadeningMode.NONE
    full_suite_executed = False

    if not killed_after_predicted and not infra_failed and not timed:
        broadening_mode, fallback_reasons = resolve_execution_fallback(
            survived_predicted=True,
            risk_class=risk_class,
            uncertainty=uncertainty,
            policy=policy,
        )
        reasons.extend(fallback_reasons)

        if broadening_mode is BroadeningMode.BROADER:
            phases.append(ExecutionPhase.BROADER.value)
            # Broader set: explicit broader_detector_ids, else unused predicted
            # (no-op) — typically caller supplies catalog extras.
            expand = [
                detector_id
                for detector_id in broader_ids
                if detector_id not in already
            ][: policy.max_broader_detectors]
            for detector_id in expand:
                if detector_id not in selected_ids:
                    selected_ids.append(detector_id)
            broader_obs = _run_detector_list(
                expand,
                predictions=predictions,
                kinds=kinds,
                phase=ExecutionPhase.BROADER,
                runner=detector_runner,
                supplied=supplied,
                already_executed=already,
            )
            all_observations.extend(broader_obs)
            if any(obs.detected for obs in broader_obs):
                killed_after_predicted = True
            if any(
                obs.status == DetectorRunStatus.ERROR.value for obs in broader_obs
            ):
                infra_failed = True
            if any(
                obs.status == DetectorRunStatus.TIMEOUT.value
                for obs in broader_obs
            ):
                timed = True

        elif broadening_mode is BroadeningMode.FULL_SUITE:
            phases.append(ExecutionPhase.FULL_SUITE.value)
            full_suite_executed = True
            expand = [
                detector_id
                for detector_id in full_suite_ids
                if detector_id not in already
            ]
            # Also include broader extras under full suite.
            for detector_id in broader_ids:
                if detector_id not in already and detector_id not in expand:
                    expand.append(detector_id)
            for detector_id in expand:
                if detector_id not in selected_ids:
                    selected_ids.append(detector_id)
            # Plan may need full_suite_fallback_enabled for selected expansion
            # beyond predicted; re-seal plan if needed.
            if set(selected_ids) - set(predicted_ids):
                plan = MutationExecutionPlan(
                    header=_header_for_kind(template, "mutation_execution_plan"),
                    execution_plan_id=plan.execution_plan_id,
                    candidate_id=plan.candidate_id,
                    candidate_cid=plan.candidate_cid,
                    expected_detection_set_cid=plan.expected_detection_set_cid,
                    selected_detector_ids=selected_ids,
                    predicted_detector_ids=predicted_ids,
                    require_disposable_worktree=True,
                    require_network_disabled=True,
                    full_suite_fallback_enabled=True,
                    timeout_seconds=plan.timeout_seconds,
                    notes=plan.notes,
                    metadata=dict(plan.metadata),
                )
            suite_obs = _run_detector_list(
                expand,
                predictions=predictions,
                kinds=kinds,
                phase=ExecutionPhase.FULL_SUITE,
                runner=detector_runner,
                supplied=supplied,
                already_executed=already,
            )
            all_observations.extend(suite_obs)
            if any(
                obs.status == DetectorRunStatus.ERROR.value for obs in suite_obs
            ):
                infra_failed = True
            if any(
                obs.status == DetectorRunStatus.TIMEOUT.value for obs in suite_obs
            ):
                timed = True
    else:
        reasons.append(REASON_FALLBACK_SKIPPED_KILL)

    # --- classify ---
    phases.append(ExecutionPhase.CLASSIFIED.value)
    # Detect full-suite phase presence.
    if ExecutionPhase.FULL_SUITE.value in phases:
        full_suite_executed = True

    classification = classify_mutation_outcome(
        predicted_detector_ids=predicted_ids,
        selected_detector_ids=selected_ids,
        observations=all_observations,
        detector_kinds=kinds,
        full_suite_executed=full_suite_executed,
        invalid_mutant=False,
        uncompilable=False,
        infrastructure_ok=infrastructure_ok and not infra_failed,
        timed_out=timed,
        equivalence_status=equivalence_status,
        human_review_required=human_review_required,
        notes=note_text,
    )
    reasons.extend(classification.reason_codes)

    # Equivalence outcomes require assessment CID when sealed into MutationOutcome.
    eq_cid = equivalence_assessment_cid
    if classification.outcome_status in {
        MutationOutcomeStatus.EQUIVALENT.value,
        MutationOutcomeStatus.PROBABLY_EQUIVALENT.value,
    }:
        if eq_cid is None:
            raise MutationExecutionError(
                "equivalent outcomes require equivalence_assessment_cid"
            )

    receipt, outcome = _seal_receipt_and_outcome(
        template=template,
        candidate_id=cand_id,
        candidate_cid=cand_cid,
        mutant_identity_cid=mutant_cid,
        detection_set=detection_set,
        plan=plan,
        classification=classification,
        observations=all_observations,
        infrastructure_ok=infrastructure_ok and not infra_failed,
        timed_out=timed,
        equivalence_assessment_cid=eq_cid,
        full_suite_counterfactual_cost_units=full_suite_counterfactual_cost_units,
        receipt_id=receipt_id,
        outcome_id=outcome_id,
        notes=note_text,
        metadata=meta,
    )

    # Honesty: exactly one terminal outcome, verified identity.
    try:
        verify_outcome_identity(outcome)
        assert_outcome_never_false_kill(outcome)
    except ExecutionContractError as exc:
        raise MutationExecutionError(
            f"outcome identity/honesty failure: {exc}",
            reason_code="dishonest_outcome",
        ) from exc

    return MutationExecutionReport(
        candidate_id=cand_id,
        candidate_cid=cand_cid,
        mutant_identity_cid=mutant_cid,
        disposition=ExecutionDisposition.COMPLETED,
        baseline=sealed_baseline,
        detection_set_cid=detection_set.detection_set_cid,
        execution_plan_cid=plan.execution_plan_cid,
        phases=tuple(phases),
        observations=tuple(all_observations),
        classification=classification,
        receipt=receipt,
        outcome=outcome,
        broadening_mode=broadening_mode,
        fallback_policy=policy,
        reason_codes=_stable_unique(reasons),
        metadata=meta,
    )


def _seal_receipt_and_outcome(
    *,
    template: AssuranceArtifactHeader,
    candidate_id: str,
    candidate_cid: str,
    mutant_identity_cid: str,
    detection_set: ExpectedDetectionSet,
    plan: MutationExecutionPlan,
    classification: OutcomeClassification,
    observations: Sequence[DetectorRunObservation],
    infrastructure_ok: bool,
    timed_out: bool,
    equivalence_assessment_cid: str | None,
    full_suite_counterfactual_cost_units: int | None,
    receipt_id: str | None,
    outcome_id: str | None,
    notes: str | None,
    metadata: Mapping[str, Any],
) -> tuple[MutationExecutionReceipt, MutationOutcome]:
    assert classification.detector_classification is not None
    cost = _measure_cost(
        observations,
        full_suite_counterfactual_units=full_suite_counterfactual_cost_units,
    )
    rid = receipt_id or f"receipt_{candidate_id}"
    if not _TOKEN_RE.fullmatch(rid):
        rid = "receipt_mutation_execution"
    oid = outcome_id or f"outcome_{candidate_id}"
    if not _TOKEN_RE.fullmatch(oid):
        oid = "outcome_mutation_execution"

    receipt_header = _header_for_kind(template, "mutation_execution_receipt")
    receipt = MutationExecutionReceipt(
        header=receipt_header,
        receipt_id=rid,
        candidate_id=candidate_id,
        candidate_cid=candidate_cid,
        execution_plan_cid=plan.execution_plan_cid,
        expected_detection_set_cid=detection_set.detection_set_cid,
        detector_classification=classification.detector_classification,
        cost=cost,
        mutant_identity_cid=mutant_identity_cid,
        infrastructure_ok=infrastructure_ok,
        timed_out=timed_out,
        notes=notes,
        metadata=dict(metadata),
    )

    outcome_terminal = AssuranceTerminalStatus.COMPLETE
    if classification.outcome_status in {
        MutationOutcomeStatus.INCONCLUSIVE.value,
        MutationOutcomeStatus.TIMEOUT.value,
        MutationOutcomeStatus.INFRASTRUCTURE_FAILURE.value,
    }:
        outcome_terminal = AssuranceTerminalStatus.INCONCLUSIVE
    elif classification.outcome_status in {
        MutationOutcomeStatus.INVALID_MUTANT.value,
        MutationOutcomeStatus.UNCOMPILABLE.value,
    }:
        outcome_terminal = AssuranceTerminalStatus.INVALID
    elif (
        classification.outcome_status
        == MutationOutcomeStatus.HUMAN_REVIEW_REQUIRED.value
    ):
        outcome_terminal = AssuranceTerminalStatus.HUMAN_REVIEW_REQUIRED

    outcome_header = _header_for_kind(
        template,
        "mutation_outcome",
        terminal_status=outcome_terminal,
        receipt_cids=(receipt.receipt_cid,),
    )
    outcome = MutationOutcome(
        header=outcome_header,
        outcome_id=oid,
        candidate_id=candidate_id,
        candidate_cid=candidate_cid,
        receipt_cid=receipt.receipt_cid,
        expected_detection_set_cid=detection_set.detection_set_cid,
        outcome_status=classification.outcome_status,
        detector_classification=classification.detector_classification,
        killing_detector_id=classification.killing_detector_id,
        killing_detector_kind=classification.killing_detector_kind,
        equivalence_assessment_cid=equivalence_assessment_cid,
        notes=notes,
        metadata=dict(metadata),
    )
    return receipt, outcome


# ---------------------------------------------------------------------------
# Descriptors / vocabulary exports
# ---------------------------------------------------------------------------


def execute_mutation_descriptor() -> Mapping[str, Any]:
    """Public descriptor for ``execute_mutation@1``."""

    return MappingProxyType(
        {
            "interface_id": EXECUTE_MUTATION_INTERFACE,
            "classify_interface_id": CLASSIFY_MUTATION_OUTCOME_INTERFACE,
            "report_schema": MUTATION_EXECUTION_REPORT_SCHEMA,
            "report_interface": MUTATION_EXECUTION_REPORT_INTERFACE,
            "evidence_subset": AAE_OUTCOME_EVIDENCE,
            "adapter_id": ADAPTER_ID,
            "board_namespace": BOARD_NAMESPACE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "closed_outcome_statuses": list(mutation_outcome_statuses()),
            "killed_outcome_statuses": list(killed_outcome_statuses()),
            "never_counted_as_killed_statuses": list(
                never_counted_as_killed_statuses()
            ),
            "acceptance": (
                "unmutated_baseline_green_or_explicitly_blocked",
                "predicted_checks_first",
                "broader_fallback_policy_bound",
                "observed_detectors_and_one_closed_terminal_outcome",
            ),
            "production_policy_changed": False,
        }
    )


def closed_mutation_outcome_statuses() -> tuple[str, ...]:
    """Return the closed mutation outcome vocabulary."""

    return mutation_outcome_statuses()


__all__ = [
    "AAE_OUTCOME_EVIDENCE",
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "CLASSIFY_MUTATION_OUTCOME_INTERFACE",
    "DETECTOR_RUN_OBSERVATION_SCHEMA",
    "EXECUTE_MUTATION_INTERFACE",
    "EXECUTION_BASELINE_SCHEMA",
    "FALLBACK_POLICY_SCHEMA",
    "FULL_SUITE_DETECTOR_ID",
    "MUTATION_EXECUTION_REPORT_INTERFACE",
    "MUTATION_EXECUTION_REPORT_SCHEMA",
    "OUTCOME_CLASSIFICATION_SCHEMA",
    "REASON_BASELINE_BLOCKED",
    "REASON_BASELINE_GREEN",
    "REASON_FALLBACK_DISABLED",
    "REASON_FALLBACK_POLICY_BOUND",
    "REASON_OBSERVED_HONEST",
    "REASON_ONE_TERMINAL_OUTCOME",
    "REASON_PREDICTED_FIRST",
    "BaselineGateError",
    "BaselineGateStatus",
    "DetectorRunObservation",
    "DetectorRunStatus",
    "DetectorRunner",
    "ExecutionBaseline",
    "ExecutionDisposition",
    "ExecutionFallbackPolicy",
    "ExecutionPhase",
    "FallbackPolicyError",
    "MutationExecutionError",
    "MutationExecutionReport",
    "OutcomeClassification",
    "build_execution_header",
    "classify_mutation_outcome",
    "closed_mutation_outcome_statuses",
    "evaluate_baseline_gate",
    "execute_mutation",
    "execute_mutation_descriptor",
    "resolve_execution_fallback",
]
