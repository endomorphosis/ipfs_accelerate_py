"""Execute bounded counterexample-guided context expansion before route escalation (SCG-029).

``execute_expansion_loop`` runs hypothesis/add/retry/verify cycles under hard
token, step, retry, escalation, wall-time, and spend caps taken from a
:class:`~ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts.ContextExpansionPlan`.

Normative fail-closed invariants:

* Hard limits are enforced on every step and **across restart** via a durable
  checkpoint ledger (spent counters are restored; remaining budget cannot be
  exceeded after resume).
* Where omission expansion is supported, context repair steps execute and may
  repair **before** frontier / model-route escalation.
* Both-context failure may request route escalation, but **never** blames
  compression (no omission-blame reason codes; ``compression_blamed`` is false).
* Same model route is retried after supported context expansion when
  appropriate; model escalation only after expansion is insufficient or
  evidence says reasoning failure.
* Expanded / repaired results remain evaluation candidates — never production
  acceptance upgrades.

Conflict policy: reuses datasets :class:`ContextExpansionPlan` /
:class:`ContextExpansionStep` contracts and closed comparative/route
vocabularies. Does not mint a second receipt hierarchy or treat text equality
as semantic success.

Importing this module performs no I/O, opens no sockets, and never invokes a
provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Final, Iterable, Mapping, Protocol, Sequence
import json
import os
import re
import threading
import time
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
    MAX_EXPANSION_STEPS,
    MAX_REASON_CODES,
    AuditContractError,
    ContextExpansionPlan,
    ContextExpansionStep,
    DecisionAction,
    ExpansionAction,
    ExpansionStepStatus,
    RouteTier,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    AssumptionKind,
    ArtifactProvenance,
    AuthoritySource,
    ContextSufficiencyState,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.expansion import (
    context_expansion_actions,
    route_escalation_actions,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    ComparativeOutcome,
    SemanticGovernorExecutionError,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_EXPANSION_LOOP_EVIDENCE: Final[str] = "scg/expansion-loop@1"
EXECUTE_EXPANSION_LOOP_INTERFACE: Final[str] = "execute_expansion_loop@1"
EXPANSION_LOOP_RESULT_INTERFACE: Final[str] = "ExpansionLoopResult@1"
EXPANSION_LOOP_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "expansion-loop-result@1"
)
EXPANSION_LOOP_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "expansion-loop-checkpoint@1"
)
EXPANSION_MODEL_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "expansion-model-policy@1"
)
EXPANSION_VERIFICATION_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "expansion-verification-policy@1"
)
EXPANSION_ATTEMPT_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "expansion-attempt-result@1"
)
EXPANSION_STEP_EXECUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "expansion-step-execution@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_expansion_loop"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "expansion_loop.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_METADATA_KEYS: Final[int] = 64
MAX_IDS: Final[int] = 256
MAX_ARTIFACT_IDS: Final[int] = 4_096

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

_CONTEXT_EXPANSION_ACTIONS: Final[frozenset[str]] = frozenset(
    context_expansion_actions()
)
_ROUTE_ESCALATION_ACTIONS: Final[frozenset[str]] = frozenset(
    route_escalation_actions()
)
_HUMAN_REVIEW_ACTIONS: Final[frozenset[str]] = frozenset(
    {ExpansionAction.REQUEST_HUMAN_REVIEW.value}
)

# Comparative outcomes that mean both paired contexts failed (plan §5 / SCG-014).
_BOTH_CONTEXT_FAILURE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        ComparativeOutcome.BOTH_FAILED_SAME_REASON.value,
        ComparativeOutcome.BOTH_FAILED_DIFFERENT_REASON.value,
    }
)

# Outcomes where expanded success after compressed failure supports omission repair.
_OMISSION_SUPPORTING_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value,
        ComparativeOutcome.EXPANDED_BETTER.value,
    }
)

# Reason codes that would blame compression / omission — forbidden on both-fail.
_COMPRESSION_BLAME_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "compression_omission",
        "omission_blame",
        "blame_compression",
        "compressed_context_omission",
        "compression_failure",
        "omission_caused_failure",
        "critical_omission_accepted",
    }
)

_DEFAULT_CLOCK: Callable[[], float] = time.monotonic


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExpansionLoopError(SemanticGovernorExecutionError):
    """Raised when expansion-loop input is malformed or fail-closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "expansion_loop_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class ExpansionLimitExceededError(ExpansionLoopError):
    """Hard expansion limit exhausted (step/token/retry/escalation/time/spend)."""

    def __init__(
        self,
        message: str,
        *,
        limit_kind: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            reason_code=f"limit_exceeded:{limit_kind}",
            details=details,
        )
        self.limit_kind = str(limit_kind)


class ExpansionCheckpointError(ExpansionLoopError):
    """Durable checkpoint is corrupt, mismatched, or fail-closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "checkpoint_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


# ---------------------------------------------------------------------------
# Closed enums
# ---------------------------------------------------------------------------


class ExpansionLoopDisposition(str, Enum):
    """Closed terminal disposition for one expansion-loop execution."""

    REPAIRED = "repaired"
    ROUTE_ESCALATION_REQUESTED = "route_escalation_requested"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    LIMITS_EXHAUSTED = "limits_exhausted"
    INCONCLUSIVE = "inconclusive"
    NO_ACTION = "no_action"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExpansionLimitKind(str, Enum):
    """Closed hard-limit dimensions enforced by the loop ledger."""

    STEPS = "steps"
    TOKENS = "tokens"
    RETRIES = "retries"
    ESCALATIONS = "escalations"
    WALL_TIME_MS = "wall_time_ms"
    SPEND_MICROS = "spend_micros"


class ExpansionAttemptStatus(str, Enum):
    """Closed status for one apply/retry/verify attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


class ExpansionLoopPhase(str, Enum):
    """Closed phases of the expansion loop (checkpointed)."""

    ADMITTED = "admitted"
    CONTEXT_EXPANSION = "context_expansion"
    SAME_ROUTE_RETRY = "same_route_retry"
    VERIFY = "verify"
    ROUTE_ESCALATION = "route_escalation"
    HUMAN_REVIEW = "human_review"
    COMPLETE = "complete"
    LIMITS_EXHAUSTED = "limits_exhausted"
    CANCELLED = "cancelled"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise ExpansionLoopError(f"{name} must be a nonempty string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ExpansionLoopError(f"{name} must be trimmed NFC text")
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise ExpansionLoopError(f"{name} contains invalid text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise ExpansionLoopError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise ExpansionLoopError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ExpansionLoopError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ExpansionLoopError(f"{name} must be a nonnegative integer")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise ExpansionLoopError(f"{name} has unsupported value {value!r}") from exc


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
        raise ExpansionLoopError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    try:
        reject_private_and_model_authority(thawed, path=name)
    except SemanticGovernorBaseError as exc:
        raise ExpansionLoopError(str(exc)) from exc
    return thawed


def _mapping(value: Any, name: str, *, max_keys: int = MAX_METADATA_KEYS) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpansionLoopError(f"{name} must be a mapping")
    if len(value) > max_keys:
        raise ExpansionLoopError(f"{name} exceeds metadata key bound")
    return _freeze_structured(_require_structured(dict(value), name))


def _unique_sorted_tokens(
    values: Iterable[Any],
    name: str,
    *,
    max_items: int = MAX_IDS,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ExpansionLoopError(f"{name} must be a list or tuple")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        token = _token(item, name)
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    ordered.sort()
    if len(ordered) > max_items:
        raise ExpansionLoopError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _unique_sorted_reason_codes(values: Iterable[Any], name: str) -> tuple[str, ...]:
    return _unique_sorted_tokens(values, name, max_items=MAX_REASON_CODES)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpansionModelPolicy:
    """Model-route policy for same-route retry vs frontier escalation.

    Same route is preferred after supported context expansion. Frontier
    escalation is allowed only after context expansion is insufficient or the
    comparative evidence indicates reasoning / both-context failure.
    """

    current_route_tier: RouteTier | str = RouteTier.MEDIUM
    allow_same_route_retry: bool = True
    allow_frontier_escalation: bool = True
    max_route_escalations: int = 1
    frontier_route_tier: RouteTier | str = RouteTier.FRONTIER
    policy_id: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_route_tier",
            _enum(self.current_route_tier, RouteTier, "current_route_tier"),
        )
        object.__setattr__(
            self,
            "allow_same_route_retry",
            _bool(self.allow_same_route_retry, "allow_same_route_retry"),
        )
        object.__setattr__(
            self,
            "allow_frontier_escalation",
            _bool(self.allow_frontier_escalation, "allow_frontier_escalation"),
        )
        object.__setattr__(
            self,
            "max_route_escalations",
            _nonneg_int(self.max_route_escalations, "max_route_escalations"),
        )
        object.__setattr__(
            self,
            "frontier_route_tier",
            _enum(self.frontier_route_tier, RouteTier, "frontier_route_tier"),
        )
        object.__setattr__(
            self, "policy_id", _optional_text(self.policy_id, "policy_id")
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXPANSION_MODEL_POLICY_SCHEMA,
            "current_route_tier": self.current_route_tier,
            "allow_same_route_retry": self.allow_same_route_retry,
            "allow_frontier_escalation": self.allow_frontier_escalation,
            "max_route_escalations": self.max_route_escalations,
            "frontier_route_tier": self.frontier_route_tier,
            "policy_id": self.policy_id,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def policy_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "policy_cid": self.policy_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpansionModelPolicy":
        if not isinstance(data, Mapping):
            raise ExpansionLoopError("ExpansionModelPolicy must be a mapping")
        payload = dict(data)
        payload.pop("policy_cid", None)
        schema = payload.pop("schema", None)
        if schema is not None and schema != EXPANSION_MODEL_POLICY_SCHEMA:
            raise ExpansionLoopError("unsupported ExpansionModelPolicy schema")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ExpansionVerificationPolicy:
    """Verification gates applied after each context expansion / retry.

    A verification pass never upgrades production acceptance; it only marks
    the expansion attempt as repaired for the evaluation candidate.
    """

    require_selected_tests: bool = True
    require_full_suite: bool = False
    require_proofs: bool = False
    require_static_checks: bool = False
    require_no_counterexample: bool = True
    accept_on_verification_pass: bool = True
    policy_id: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "require_selected_tests",
            _bool(self.require_selected_tests, "require_selected_tests"),
        )
        object.__setattr__(
            self,
            "require_full_suite",
            _bool(self.require_full_suite, "require_full_suite"),
        )
        object.__setattr__(
            self, "require_proofs", _bool(self.require_proofs, "require_proofs")
        )
        object.__setattr__(
            self,
            "require_static_checks",
            _bool(self.require_static_checks, "require_static_checks"),
        )
        object.__setattr__(
            self,
            "require_no_counterexample",
            _bool(self.require_no_counterexample, "require_no_counterexample"),
        )
        object.__setattr__(
            self,
            "accept_on_verification_pass",
            _bool(self.accept_on_verification_pass, "accept_on_verification_pass"),
        )
        object.__setattr__(
            self, "policy_id", _optional_text(self.policy_id, "policy_id")
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXPANSION_VERIFICATION_POLICY_SCHEMA,
            "require_selected_tests": self.require_selected_tests,
            "require_full_suite": self.require_full_suite,
            "require_proofs": self.require_proofs,
            "require_static_checks": self.require_static_checks,
            "require_no_counterexample": self.require_no_counterexample,
            "accept_on_verification_pass": self.accept_on_verification_pass,
            "policy_id": self.policy_id,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def policy_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "policy_cid": self.policy_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpansionVerificationPolicy":
        if not isinstance(data, Mapping):
            raise ExpansionLoopError(
                "ExpansionVerificationPolicy must be a mapping"
            )
        payload = dict(data)
        payload.pop("policy_cid", None)
        schema = payload.pop("schema", None)
        if schema is not None and schema != EXPANSION_VERIFICATION_POLICY_SCHEMA:
            raise ExpansionLoopError(
                "unsupported ExpansionVerificationPolicy schema"
            )
        return cls(**payload)

    def evaluate(self, attempt: "ExpansionAttemptResult") -> bool:
        """Return True when the attempt satisfies this verification policy."""

        if not self.accept_on_verification_pass:
            return False
        if attempt.status != ExpansionAttemptStatus.SUCCEEDED.value:
            return False
        if self.require_selected_tests and not attempt.selected_tests_passed:
            return False
        if self.require_full_suite and not attempt.full_suite_passed:
            return False
        if self.require_proofs and not attempt.proofs_passed:
            return False
        if self.require_static_checks and not attempt.static_checks_passed:
            return False
        if self.require_no_counterexample and attempt.counterexample_present:
            return False
        return True


def default_model_policy(**overrides: Any) -> ExpansionModelPolicy:
    fields: dict[str, Any] = {
        "current_route_tier": RouteTier.MEDIUM.value,
        "allow_same_route_retry": True,
        "allow_frontier_escalation": True,
        "max_route_escalations": 1,
        "frontier_route_tier": RouteTier.FRONTIER.value,
        "policy_id": "default_model_policy",
    }
    fields.update(overrides)
    return ExpansionModelPolicy(**fields)


def default_verification_policy(**overrides: Any) -> ExpansionVerificationPolicy:
    fields: dict[str, Any] = {
        "require_selected_tests": True,
        "require_full_suite": False,
        "require_proofs": False,
        "require_static_checks": False,
        "require_no_counterexample": True,
        "accept_on_verification_pass": True,
        "policy_id": "default_verification_policy",
    }
    fields.update(overrides)
    return ExpansionVerificationPolicy(**fields)


# ---------------------------------------------------------------------------
# Attempt / step execution records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpansionAttemptResult:
    """Result of one apply + same-route retry + verify cycle for a step."""

    status: ExpansionAttemptStatus | str
    route_tier: RouteTier | str
    step_id: str
    step_index: int
    attempt_index: int
    selected_tests_passed: bool = False
    full_suite_passed: bool = False
    proofs_passed: bool = False
    static_checks_passed: bool = False
    counterexample_present: bool = False
    wall_time_ms: int = 0
    spend_micros: int = 0
    token_cost: int = 0
    result_cid: str | None = None
    counterexample_cid: str | None = None
    reason_codes: Sequence[str] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, ExpansionAttemptStatus, "status")
        )
        object.__setattr__(
            self, "route_tier", _enum(self.route_tier, RouteTier, "route_tier")
        )
        object.__setattr__(self, "step_id", _token(self.step_id, "step_id"))
        object.__setattr__(
            self, "step_index", _nonneg_int(self.step_index, "step_index")
        )
        object.__setattr__(
            self, "attempt_index", _nonneg_int(self.attempt_index, "attempt_index")
        )
        object.__setattr__(
            self,
            "selected_tests_passed",
            _bool(self.selected_tests_passed, "selected_tests_passed"),
        )
        object.__setattr__(
            self,
            "full_suite_passed",
            _bool(self.full_suite_passed, "full_suite_passed"),
        )
        object.__setattr__(
            self, "proofs_passed", _bool(self.proofs_passed, "proofs_passed")
        )
        object.__setattr__(
            self,
            "static_checks_passed",
            _bool(self.static_checks_passed, "static_checks_passed"),
        )
        object.__setattr__(
            self,
            "counterexample_present",
            _bool(self.counterexample_present, "counterexample_present"),
        )
        object.__setattr__(
            self, "wall_time_ms", _nonneg_int(self.wall_time_ms, "wall_time_ms")
        )
        object.__setattr__(
            self, "spend_micros", _nonneg_int(self.spend_micros, "spend_micros")
        )
        object.__setattr__(
            self, "token_cost", _nonneg_int(self.token_cost, "token_cost")
        )
        object.__setattr__(
            self, "result_cid", _optional_cid(self.result_cid, "result_cid")
        )
        object.__setattr__(
            self,
            "counterexample_cid",
            _optional_cid(self.counterexample_cid, "counterexample_cid"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_reason_codes(list(self.reason_codes), "reason_codes"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXPANSION_ATTEMPT_RESULT_SCHEMA,
            "status": self.status,
            "route_tier": self.route_tier,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "attempt_index": self.attempt_index,
            "selected_tests_passed": self.selected_tests_passed,
            "full_suite_passed": self.full_suite_passed,
            "proofs_passed": self.proofs_passed,
            "static_checks_passed": self.static_checks_passed,
            "counterexample_present": self.counterexample_present,
            "wall_time_ms": self.wall_time_ms,
            "spend_micros": self.spend_micros,
            "token_cost": self.token_cost,
            "result_cid": self.result_cid,
            "counterexample_cid": self.counterexample_cid,
            "reason_codes": list(self.reason_codes),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def attempt_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "attempt_cid": self.attempt_cid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpansionAttemptResult":
        if not isinstance(data, Mapping):
            raise ExpansionLoopError("ExpansionAttemptResult must be a mapping")
        payload = dict(data)
        payload.pop("attempt_cid", None)
        schema = payload.pop("schema", None)
        if schema is not None and schema != EXPANSION_ATTEMPT_RESULT_SCHEMA:
            raise ExpansionLoopError("unsupported ExpansionAttemptResult schema")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ExpansionStepExecution:
    """Durable record of how one planned step was executed by the loop."""

    step_id: str
    step_index: int
    action: ExpansionAction | str
    planned_status: ExpansionStepStatus | str
    executed_status: ExpansionStepStatus | str
    token_increase: int
    artifact_ids_added: Sequence[str]
    hypothesis_cid: str | None = None
    hypothesis_supported: bool | None = None
    attempts: Sequence[ExpansionAttemptResult] = ()
    prior_result_cid: str | None = None
    new_result_cid: str | None = None
    reason_codes: Sequence[str] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _token(self.step_id, "step_id"))
        object.__setattr__(
            self, "step_index", _nonneg_int(self.step_index, "step_index")
        )
        object.__setattr__(
            self, "action", _enum(self.action, ExpansionAction, "action")
        )
        object.__setattr__(
            self,
            "planned_status",
            _enum(self.planned_status, ExpansionStepStatus, "planned_status"),
        )
        object.__setattr__(
            self,
            "executed_status",
            _enum(self.executed_status, ExpansionStepStatus, "executed_status"),
        )
        object.__setattr__(
            self, "token_increase", _nonneg_int(self.token_increase, "token_increase")
        )
        object.__setattr__(
            self,
            "artifact_ids_added",
            _unique_sorted_tokens(
                list(self.artifact_ids_added),
                "artifact_ids_added",
                max_items=MAX_ARTIFACT_IDS,
            ),
        )
        object.__setattr__(
            self, "hypothesis_cid", _optional_cid(self.hypothesis_cid, "hypothesis_cid")
        )
        if self.hypothesis_supported is not None:
            object.__setattr__(
                self,
                "hypothesis_supported",
                _bool(self.hypothesis_supported, "hypothesis_supported"),
            )
        attempts = tuple(self.attempts)
        normalized: list[ExpansionAttemptResult] = []
        for item in attempts:
            if isinstance(item, ExpansionAttemptResult):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(ExpansionAttemptResult.from_dict(item))
            else:
                raise ExpansionLoopError(
                    "attempts entries must be ExpansionAttemptResult or mapping"
                )
        object.__setattr__(self, "attempts", tuple(normalized))
        object.__setattr__(
            self,
            "prior_result_cid",
            _optional_cid(self.prior_result_cid, "prior_result_cid"),
        )
        object.__setattr__(
            self, "new_result_cid", _optional_cid(self.new_result_cid, "new_result_cid")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_reason_codes(list(self.reason_codes), "reason_codes"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXPANSION_STEP_EXECUTION_SCHEMA,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "action": self.action,
            "planned_status": self.planned_status,
            "executed_status": self.executed_status,
            "token_increase": self.token_increase,
            "artifact_ids_added": list(self.artifact_ids_added),
            "hypothesis_cid": self.hypothesis_cid,
            "hypothesis_supported": self.hypothesis_supported,
            "attempts": [item.identity_payload() for item in self.attempts],
            "prior_result_cid": self.prior_result_cid,
            "new_result_cid": self.new_result_cid,
            "reason_codes": list(self.reason_codes),
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def execution_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "attempts": [item.to_dict() for item in self.attempts],
            "execution_cid": self.execution_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpansionStepExecution":
        if not isinstance(data, Mapping):
            raise ExpansionLoopError("ExpansionStepExecution must be a mapping")
        payload = dict(data)
        payload.pop("execution_cid", None)
        schema = payload.pop("schema", None)
        if schema is not None and schema != EXPANSION_STEP_EXECUTION_SCHEMA:
            raise ExpansionLoopError("unsupported ExpansionStepExecution schema")
        return cls(**payload)


# ---------------------------------------------------------------------------
# Budget ledger (restart-safe)
# ---------------------------------------------------------------------------


@dataclass
class ExpansionBudgetLedger:
    """Mutable spent counters for hard expansion limits.

    Restored from a durable checkpoint so limits remain enforced across process
    restart. Zero remaining on a required dimension fails closed before the
    next apply/retry/verify cycle.
    """

    max_steps: int
    max_token_growth: int
    max_retries: int
    max_escalations: int
    max_wall_time_ms: int
    max_spend_micros: int
    spent_steps: int = 0
    spent_tokens: int = 0
    spent_retries: int = 0
    spent_escalations: int = 0
    spent_wall_time_ms: int = 0
    spent_spend_micros: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        self.max_steps = _nonneg_int(self.max_steps, "max_steps")
        if self.max_steps > MAX_EXPANSION_STEPS:
            raise ExpansionLoopError(
                f"max_steps must be <= {MAX_EXPANSION_STEPS}"
            )
        self.max_token_growth = _nonneg_int(self.max_token_growth, "max_token_growth")
        self.max_retries = _nonneg_int(self.max_retries, "max_retries")
        self.max_escalations = _nonneg_int(self.max_escalations, "max_escalations")
        self.max_wall_time_ms = _nonneg_int(self.max_wall_time_ms, "max_wall_time_ms")
        self.max_spend_micros = _nonneg_int(self.max_spend_micros, "max_spend_micros")
        self.spent_steps = _nonneg_int(self.spent_steps, "spent_steps")
        self.spent_tokens = _nonneg_int(self.spent_tokens, "spent_tokens")
        self.spent_retries = _nonneg_int(self.spent_retries, "spent_retries")
        self.spent_escalations = _nonneg_int(
            self.spent_escalations, "spent_escalations"
        )
        self.spent_wall_time_ms = _nonneg_int(
            self.spent_wall_time_ms, "spent_wall_time_ms"
        )
        self.spent_spend_micros = _nonneg_int(
            self.spent_spend_micros, "spent_spend_micros"
        )

    @classmethod
    def from_plan(cls, plan: ContextExpansionPlan) -> "ExpansionBudgetLedger":
        return cls(
            max_steps=int(plan.max_steps),
            max_token_growth=int(plan.max_token_growth),
            max_retries=int(plan.max_retries),
            max_escalations=int(plan.max_escalations),
            max_wall_time_ms=int(plan.max_wall_time_ms),
            max_spend_micros=int(plan.max_spend_micros),
        )

    def remaining(self, kind: ExpansionLimitKind | str) -> int:
        kind_value = _enum(kind, ExpansionLimitKind, "kind")
        with self._lock:
            if kind_value == ExpansionLimitKind.STEPS.value:
                return max(0, self.max_steps - self.spent_steps)
            if kind_value == ExpansionLimitKind.TOKENS.value:
                return max(0, self.max_token_growth - self.spent_tokens)
            if kind_value == ExpansionLimitKind.RETRIES.value:
                return max(0, self.max_retries - self.spent_retries)
            if kind_value == ExpansionLimitKind.ESCALATIONS.value:
                return max(0, self.max_escalations - self.spent_escalations)
            if kind_value == ExpansionLimitKind.WALL_TIME_MS.value:
                return max(0, self.max_wall_time_ms - self.spent_wall_time_ms)
            if kind_value == ExpansionLimitKind.SPEND_MICROS.value:
                return max(0, self.max_spend_micros - self.spent_spend_micros)
            raise ExpansionLoopError(f"unknown limit kind {kind_value!r}")

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_steps": self.max_steps,
                "max_token_growth": self.max_token_growth,
                "max_retries": self.max_retries,
                "max_escalations": self.max_escalations,
                "max_wall_time_ms": self.max_wall_time_ms,
                "max_spend_micros": self.max_spend_micros,
                "spent_steps": self.spent_steps,
                "spent_tokens": self.spent_tokens,
                "spent_retries": self.spent_retries,
                "spent_escalations": self.spent_escalations,
                "spent_wall_time_ms": self.spent_wall_time_ms,
                "spent_spend_micros": self.spent_spend_micros,
                "remaining_steps": max(0, self.max_steps - self.spent_steps),
                "remaining_tokens": max(0, self.max_token_growth - self.spent_tokens),
                "remaining_retries": max(0, self.max_retries - self.spent_retries),
                "remaining_escalations": max(
                    0, self.max_escalations - self.spent_escalations
                ),
                "remaining_wall_time_ms": max(
                    0, self.max_wall_time_ms - self.spent_wall_time_ms
                ),
                "remaining_spend_micros": max(
                    0, self.max_spend_micros - self.spent_spend_micros
                ),
            }

    def recheck(
        self,
        *,
        steps: int = 0,
        tokens: int = 0,
        retries: int = 0,
        escalations: int = 0,
        wall_time_ms: int = 0,
        spend_micros: int = 0,
    ) -> None:
        """Fail closed when remaining budgets cannot cover the next cycle."""

        need_steps = _nonneg_int(steps, "steps")
        need_tokens = _nonneg_int(tokens, "tokens")
        need_retries = _nonneg_int(retries, "retries")
        need_escalations = _nonneg_int(escalations, "escalations")
        need_wall = _nonneg_int(wall_time_ms, "wall_time_ms")
        need_spend = _nonneg_int(spend_micros, "spend_micros")

        with self._lock:
            if need_steps and self.spent_steps + need_steps > self.max_steps:
                raise ExpansionLimitExceededError(
                    f"steps limit exhausted "
                    f"(spent={self.spent_steps}, max={self.max_steps})",
                    limit_kind=ExpansionLimitKind.STEPS.value,
                    details=self.snapshot(),
                )
            if need_tokens and self.spent_tokens + need_tokens > self.max_token_growth:
                raise ExpansionLimitExceededError(
                    f"token growth limit exhausted "
                    f"(spent={self.spent_tokens}, max={self.max_token_growth})",
                    limit_kind=ExpansionLimitKind.TOKENS.value,
                    details=self.snapshot(),
                )
            if need_retries and self.spent_retries + need_retries > self.max_retries:
                raise ExpansionLimitExceededError(
                    f"retries limit exhausted "
                    f"(spent={self.spent_retries}, max={self.max_retries})",
                    limit_kind=ExpansionLimitKind.RETRIES.value,
                    details=self.snapshot(),
                )
            if (
                need_escalations
                and self.spent_escalations + need_escalations > self.max_escalations
            ):
                raise ExpansionLimitExceededError(
                    f"escalations limit exhausted "
                    f"(spent={self.spent_escalations}, max={self.max_escalations})",
                    limit_kind=ExpansionLimitKind.ESCALATIONS.value,
                    details=self.snapshot(),
                )
            # Wall time: max==0 means unlimited only when need is also 0.
            if self.max_wall_time_ms > 0 and need_wall:
                if self.spent_wall_time_ms + need_wall > self.max_wall_time_ms:
                    raise ExpansionLimitExceededError(
                        f"wall_time_ms limit exhausted "
                        f"(spent={self.spent_wall_time_ms}, "
                        f"max={self.max_wall_time_ms})",
                        limit_kind=ExpansionLimitKind.WALL_TIME_MS.value,
                        details=self.snapshot(),
                    )
            if self.max_spend_micros > 0 and need_spend:
                if self.spent_spend_micros + need_spend > self.max_spend_micros:
                    raise ExpansionLimitExceededError(
                        f"spend_micros limit exhausted "
                        f"(spent={self.spent_spend_micros}, "
                        f"max={self.max_spend_micros})",
                        limit_kind=ExpansionLimitKind.SPEND_MICROS.value,
                        details=self.snapshot(),
                    )

    def record(
        self,
        *,
        steps: int = 0,
        tokens: int = 0,
        retries: int = 0,
        escalations: int = 0,
        wall_time_ms: int = 0,
        spend_micros: int = 0,
    ) -> None:
        self.recheck(
            steps=steps,
            tokens=tokens,
            retries=retries,
            escalations=escalations,
            wall_time_ms=wall_time_ms,
            spend_micros=spend_micros,
        )
        with self._lock:
            self.spent_steps += int(steps)
            self.spent_tokens += int(tokens)
            self.spent_retries += int(retries)
            self.spent_escalations += int(escalations)
            self.spent_wall_time_ms += int(wall_time_ms)
            self.spent_spend_micros += int(spend_micros)

    def apply_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore spent counters from a checkpoint snapshot (restart path)."""

        if not isinstance(snapshot, Mapping):
            raise ExpansionCheckpointError("budget snapshot must be a mapping")
        with self._lock:
            for key in (
                "spent_steps",
                "spent_tokens",
                "spent_retries",
                "spent_escalations",
                "spent_wall_time_ms",
                "spent_spend_micros",
            ):
                if key in snapshot:
                    setattr(self, key, _nonneg_int(snapshot[key], key))
            # Maxima may be re-bound only when they match the plan; never raise.
            for key in (
                "max_steps",
                "max_token_growth",
                "max_retries",
                "max_escalations",
                "max_wall_time_ms",
                "max_spend_micros",
            ):
                if key in snapshot:
                    declared = _nonneg_int(snapshot[key], key)
                    current = getattr(self, key)
                    if declared != current:
                        raise ExpansionCheckpointError(
                            f"checkpoint {key}={declared} does not match plan "
                            f"{key}={current}",
                            reason_code="checkpoint_limit_mismatch",
                            details={"checkpoint": declared, "plan": current},
                        )


# ---------------------------------------------------------------------------
# Checkpoint (durable across restart)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpansionLoopCheckpoint:
    """Restart-safe durable state for an in-progress expansion loop.

    Restoring this checkpoint re-applies spent counters so hard limits remain
    enforced after process restart.
    """

    plan_cid: str
    model_policy_cid: str
    verification_policy_cid: str
    phase: ExpansionLoopPhase | str
    next_step_index: int
    budget: Mapping[str, int]
    executed_steps: Sequence[ExpansionStepExecution] = ()
    artifacts_included: Sequence[str] = ()
    last_result_cid: str | None = None
    comparative_outcome: str | None = None
    reason_codes: Sequence[str] = ()
    compression_blamed: bool = False
    frontier_escalation_requested: bool = False
    repaired: bool = False
    generation: int = 0
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "plan_cid",
            "model_policy_cid",
            "verification_policy_cid",
            "phase",
            "next_step_index",
            "budget",
            "executed_steps",
            "artifacts_included",
            "last_result_cid",
            "comparative_outcome",
            "reason_codes",
            "compression_blamed",
            "frontier_escalation_requested",
            "repaired",
            "generation",
            "notes",
            "metadata",
            "checkpoint_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self, "model_policy_cid", _cid(self.model_policy_cid, "model_policy_cid")
        )
        object.__setattr__(
            self,
            "verification_policy_cid",
            _cid(self.verification_policy_cid, "verification_policy_cid"),
        )
        object.__setattr__(
            self, "phase", _enum(self.phase, ExpansionLoopPhase, "phase")
        )
        object.__setattr__(
            self, "next_step_index", _nonneg_int(self.next_step_index, "next_step_index")
        )
        if not isinstance(self.budget, Mapping):
            raise ExpansionCheckpointError("budget must be a mapping")
        budget = {
            key: _nonneg_int(value, f"budget.{key}")
            for key, value in dict(self.budget).items()
            if type(key) is str
        }
        object.__setattr__(self, "budget", MappingProxyType(budget))
        steps: list[ExpansionStepExecution] = []
        if not isinstance(self.executed_steps, (list, tuple)):
            raise ExpansionCheckpointError("executed_steps must be a list")
        for item in self.executed_steps:
            if isinstance(item, ExpansionStepExecution):
                steps.append(item)
            elif isinstance(item, Mapping):
                steps.append(ExpansionStepExecution.from_dict(item))
            else:
                raise ExpansionCheckpointError(
                    "executed_steps entries must be ExpansionStepExecution or mapping"
                )
        object.__setattr__(self, "executed_steps", tuple(steps))
        object.__setattr__(
            self,
            "artifacts_included",
            _unique_sorted_tokens(
                list(self.artifacts_included),
                "artifacts_included",
                max_items=MAX_ARTIFACT_IDS,
            ),
        )
        object.__setattr__(
            self,
            "last_result_cid",
            _optional_cid(self.last_result_cid, "last_result_cid"),
        )
        object.__setattr__(
            self,
            "comparative_outcome",
            _optional_text(self.comparative_outcome, "comparative_outcome"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_reason_codes(list(self.reason_codes), "reason_codes"),
        )
        object.__setattr__(
            self, "compression_blamed", _bool(self.compression_blamed, "compression_blamed")
        )
        object.__setattr__(
            self,
            "frontier_escalation_requested",
            _bool(
                self.frontier_escalation_requested, "frontier_escalation_requested"
            ),
        )
        object.__setattr__(self, "repaired", _bool(self.repaired, "repaired"))
        object.__setattr__(
            self, "generation", _nonneg_int(self.generation, "generation")
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        _assert_no_compression_blame(
            self.reason_codes,
            compression_blamed=self.compression_blamed,
            comparative_outcome=self.comparative_outcome,
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXPANSION_LOOP_CHECKPOINT_SCHEMA,
            "plan_cid": self.plan_cid,
            "model_policy_cid": self.model_policy_cid,
            "verification_policy_cid": self.verification_policy_cid,
            "phase": self.phase,
            "next_step_index": self.next_step_index,
            "budget": dict(self.budget),
            "executed_steps": [step.identity_payload() for step in self.executed_steps],
            "artifacts_included": list(self.artifacts_included),
            "last_result_cid": self.last_result_cid,
            "comparative_outcome": self.comparative_outcome,
            "reason_codes": list(self.reason_codes),
            "compression_blamed": self.compression_blamed,
            "frontier_escalation_requested": self.frontier_escalation_requested,
            "repaired": self.repaired,
            "generation": self.generation,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def checkpoint_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "executed_steps": [step.to_dict() for step in self.executed_steps],
            "checkpoint_cid": self.checkpoint_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpansionLoopCheckpoint":
        if not isinstance(data, Mapping):
            raise ExpansionCheckpointError("checkpoint must be a mapping")
        payload = dict(data)
        claimed = payload.pop("checkpoint_cid", None)
        schema = payload.pop("schema", None)
        if schema is not None and schema != EXPANSION_LOOP_CHECKPOINT_SCHEMA:
            raise ExpansionCheckpointError(
                "unsupported ExpansionLoopCheckpoint schema"
            )
        result = cls(**payload)
        if claimed is not None and claimed != result.checkpoint_cid:
            raise ExpansionCheckpointError(
                "ExpansionLoopCheckpoint checkpoint_cid does not verify",
                reason_code="checkpoint_cid_mismatch",
            )
        return result


class ExpansionCheckpointStore(Protocol):
    """Durable store for expansion-loop checkpoints (restart path)."""

    def load(self, plan_cid: str) -> ExpansionLoopCheckpoint | None: ...

    def save(self, checkpoint: ExpansionLoopCheckpoint) -> None: ...


class InMemoryExpansionCheckpointStore:
    """Process-local checkpoint store (tests and single-process runs)."""

    def __init__(self) -> None:
        self._by_plan: dict[str, ExpansionLoopCheckpoint] = {}
        self._lock = threading.Lock()

    def load(self, plan_cid: str) -> ExpansionLoopCheckpoint | None:
        key = _cid(plan_cid, "plan_cid")
        with self._lock:
            return self._by_plan.get(key)

    def save(self, checkpoint: ExpansionLoopCheckpoint) -> None:
        if not isinstance(checkpoint, ExpansionLoopCheckpoint):
            raise ExpansionCheckpointError(
                "checkpoint must be ExpansionLoopCheckpoint"
            )
        with self._lock:
            self._by_plan[checkpoint.plan_cid] = checkpoint


class FilesystemExpansionCheckpointStore:
    """Atomic filesystem checkpoint store for restart-safe recovery."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, plan_cid: str) -> Path:
        # Content digests are filesystem-safe; avoid path traversal.
        safe = plan_cid.replace("/", "_").replace("\\", "_")
        return self._directory / f"{safe}.checkpoint.json"

    def load(self, plan_cid: str) -> ExpansionLoopCheckpoint | None:
        key = _cid(plan_cid, "plan_cid")
        path = self._path_for(key)
        with self._lock:
            if not path.is_file():
                return None
            try:
                raw = path.read_text(encoding="utf-8")
                payload = json.loads(raw)
            except (OSError, json.JSONDecodeError) as exc:
                raise ExpansionCheckpointError(
                    f"failed to load checkpoint: {exc}",
                    reason_code="checkpoint_load_failed",
                ) from exc
        return ExpansionLoopCheckpoint.from_dict(payload)

    def save(self, checkpoint: ExpansionLoopCheckpoint) -> None:
        if not isinstance(checkpoint, ExpansionLoopCheckpoint):
            raise ExpansionCheckpointError(
                "checkpoint must be ExpansionLoopCheckpoint"
            )
        path = self._path_for(checkpoint.plan_cid)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        payload = checkpoint.to_dict()
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with self._lock:
            try:
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, path)
            except OSError as exc:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
                raise ExpansionCheckpointError(
                    f"failed to save checkpoint: {exc}",
                    reason_code="checkpoint_save_failed",
                ) from exc


# ---------------------------------------------------------------------------
# Step runner protocol + default simulated runner
# ---------------------------------------------------------------------------


class ExpansionStepRunner(Protocol):
    """Apply one expansion step and return a same-route verify result."""

    def apply_and_verify(
        self,
        step: ContextExpansionStep,
        *,
        artifacts_included: Sequence[str],
        route_tier: str,
        attempt_index: int,
        model_policy: ExpansionModelPolicy,
        verification_policy: ExpansionVerificationPolicy,
    ) -> ExpansionAttemptResult: ...


@dataclass
class ScriptedExpansionStepRunner:
    """Deterministic runner driven by a script of outcomes (tests / offline).

    ``script`` maps ``step_id`` (or ``*`` default) to a sequence of attempt
    outcomes consumed in order. When the sequence is exhausted, the last
    outcome is reused.
    """

    script: Mapping[str, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    default_status: str = ExpansionAttemptStatus.FAILED.value
    wall_time_ms: int = 10
    spend_micros: int = 0
    _cursors: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def apply_and_verify(
        self,
        step: ContextExpansionStep,
        *,
        artifacts_included: Sequence[str],
        route_tier: str,
        attempt_index: int,
        model_policy: ExpansionModelPolicy,
        verification_policy: ExpansionVerificationPolicy,
    ) -> ExpansionAttemptResult:
        with self._lock:
            key = step.step_id if step.step_id in self.script else "*"
            sequence = list(self.script.get(key, ()))
            cursor = self._cursors.get(key, 0)
            if sequence:
                entry = dict(sequence[min(cursor, len(sequence) - 1)])
                self._cursors[key] = cursor + 1
            else:
                entry = {"status": self.default_status}

        status = entry.get("status", self.default_status)
        selected = bool(entry.get("selected_tests_passed", status == "succeeded"))
        full = bool(entry.get("full_suite_passed", False))
        proofs = bool(entry.get("proofs_passed", False))
        static = bool(entry.get("static_checks_passed", True))
        cex = bool(entry.get("counterexample_present", status != "succeeded"))
        reason_codes = list(entry.get("reason_codes", ()))
        if status == "succeeded" and not reason_codes:
            reason_codes = ["verification_pass"]
        if status != "succeeded" and not reason_codes:
            reason_codes = ["verification_failed"]

        return ExpansionAttemptResult(
            status=status,
            route_tier=route_tier,
            step_id=step.step_id,
            step_index=step.step_index,
            attempt_index=attempt_index,
            selected_tests_passed=selected,
            full_suite_passed=full,
            proofs_passed=proofs,
            static_checks_passed=static,
            counterexample_present=cex,
            wall_time_ms=int(entry.get("wall_time_ms", self.wall_time_ms)),
            spend_micros=int(entry.get("spend_micros", self.spend_micros)),
            token_cost=int(entry.get("token_cost", 0)),
            result_cid=entry.get("result_cid"),
            counterexample_cid=entry.get("counterexample_cid"),
            reason_codes=reason_codes,
            notes=entry.get("notes"),
            metadata={
                "artifacts_included": list(artifacts_included),
                "script_key": key,
            },
        )


@dataclass
class RepairingOnArtifactRunner:
    """Succeeds only after a required artifact has been included (omission repair)."""

    required_artifact_id: str
    wall_time_ms: int = 10
    spend_micros: int = 0

    def apply_and_verify(
        self,
        step: ContextExpansionStep,
        *,
        artifacts_included: Sequence[str],
        route_tier: str,
        attempt_index: int,
        model_policy: ExpansionModelPolicy,
        verification_policy: ExpansionVerificationPolicy,
    ) -> ExpansionAttemptResult:
        included = set(artifacts_included) | set(step.artifact_ids_added)
        repaired = self.required_artifact_id in included
        if repaired:
            return ExpansionAttemptResult(
                status=ExpansionAttemptStatus.SUCCEEDED.value,
                route_tier=route_tier,
                step_id=step.step_id,
                step_index=step.step_index,
                attempt_index=attempt_index,
                selected_tests_passed=True,
                full_suite_passed=False,
                proofs_passed=False,
                static_checks_passed=True,
                counterexample_present=False,
                wall_time_ms=self.wall_time_ms,
                spend_micros=self.spend_micros,
                token_cost=0,
                reason_codes=("omission_repair_verified",),
                notes="Supported omission expansion repaired the failure",
                metadata={"required_artifact_id": self.required_artifact_id},
            )
        return ExpansionAttemptResult(
            status=ExpansionAttemptStatus.FAILED.value,
            route_tier=route_tier,
            step_id=step.step_id,
            step_index=step.step_index,
            attempt_index=attempt_index,
            selected_tests_passed=False,
            full_suite_passed=False,
            proofs_passed=False,
            static_checks_passed=True,
            counterexample_present=True,
            wall_time_ms=self.wall_time_ms,
            spend_micros=self.spend_micros,
            token_cost=0,
            reason_codes=("counterexample_still_open",),
            notes="Required artifact still omitted",
            metadata={"required_artifact_id": self.required_artifact_id},
        )


@dataclass
class AlwaysFailRunner:
    """Always fails verification (both-context / escalation path tests)."""

    wall_time_ms: int = 5
    spend_micros: int = 0
    reason_codes: Sequence[str] = ("both_context_model_failure",)

    def apply_and_verify(
        self,
        step: ContextExpansionStep,
        *,
        artifacts_included: Sequence[str],
        route_tier: str,
        attempt_index: int,
        model_policy: ExpansionModelPolicy,
        verification_policy: ExpansionVerificationPolicy,
    ) -> ExpansionAttemptResult:
        return ExpansionAttemptResult(
            status=ExpansionAttemptStatus.FAILED.value,
            route_tier=route_tier,
            step_id=step.step_id,
            step_index=step.step_index,
            attempt_index=attempt_index,
            selected_tests_passed=False,
            full_suite_passed=False,
            proofs_passed=False,
            static_checks_passed=True,
            counterexample_present=True,
            wall_time_ms=self.wall_time_ms,
            spend_micros=self.spend_micros,
            token_cost=0,
            reason_codes=tuple(self.reason_codes),
            notes="Attempt failed under expanded context",
            metadata={"artifacts_included": list(artifacts_included)},
        )


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpansionLoopResult:
    """Closed result of :func:`execute_expansion_loop`."""

    plan_cid: str
    disposition: ExpansionLoopDisposition | str
    decision_action: DecisionAction | str
    sufficiency_state: ContextSufficiencyState | str
    route_tier: RouteTier | str
    repaired: bool
    frontier_escalation_requested: bool
    compression_blamed: bool
    context_before_model_escalation: bool
    reason_codes: Sequence[str]
    executed_steps: Sequence[ExpansionStepExecution]
    budget: Mapping[str, int]
    model_policy_cid: str
    verification_policy_cid: str
    comparative_outcome: str | None = None
    checkpoint_cid: str | None = None
    last_result_cid: str | None = None
    artifacts_included: Sequence[str] = ()
    requires_human_review: bool = False
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "plan_cid",
            "disposition",
            "decision_action",
            "sufficiency_state",
            "route_tier",
            "repaired",
            "frontier_escalation_requested",
            "compression_blamed",
            "context_before_model_escalation",
            "reason_codes",
            "executed_steps",
            "budget",
            "model_policy_cid",
            "verification_policy_cid",
            "comparative_outcome",
            "checkpoint_cid",
            "last_result_cid",
            "artifacts_included",
            "requires_human_review",
            "notes",
            "metadata",
            "result_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ExpansionLoopDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "decision_action",
            _enum(self.decision_action, DecisionAction, "decision_action"),
        )
        object.__setattr__(
            self,
            "sufficiency_state",
            _enum(self.sufficiency_state, ContextSufficiencyState, "sufficiency_state"),
        )
        object.__setattr__(
            self, "route_tier", _enum(self.route_tier, RouteTier, "route_tier")
        )
        object.__setattr__(self, "repaired", _bool(self.repaired, "repaired"))
        object.__setattr__(
            self,
            "frontier_escalation_requested",
            _bool(
                self.frontier_escalation_requested, "frontier_escalation_requested"
            ),
        )
        object.__setattr__(
            self, "compression_blamed", _bool(self.compression_blamed, "compression_blamed")
        )
        object.__setattr__(
            self,
            "context_before_model_escalation",
            _bool(
                self.context_before_model_escalation,
                "context_before_model_escalation",
            ),
        )
        reasons = _unique_sorted_reason_codes(list(self.reason_codes), "reason_codes")
        if not reasons:
            raise ExpansionLoopError("reason_codes must not be empty")
        object.__setattr__(self, "reason_codes", reasons)
        steps: list[ExpansionStepExecution] = []
        if not isinstance(self.executed_steps, (list, tuple)):
            raise ExpansionLoopError("executed_steps must be a list")
        for item in self.executed_steps:
            if isinstance(item, ExpansionStepExecution):
                steps.append(item)
            elif isinstance(item, Mapping):
                steps.append(ExpansionStepExecution.from_dict(item))
            else:
                raise ExpansionLoopError(
                    "executed_steps entries must be ExpansionStepExecution or mapping"
                )
        object.__setattr__(self, "executed_steps", tuple(steps))
        if not isinstance(self.budget, Mapping):
            raise ExpansionLoopError("budget must be a mapping")
        budget = {
            key: _nonneg_int(value, f"budget.{key}")
            for key, value in dict(self.budget).items()
            if type(key) is str
        }
        object.__setattr__(self, "budget", MappingProxyType(budget))
        object.__setattr__(
            self, "model_policy_cid", _cid(self.model_policy_cid, "model_policy_cid")
        )
        object.__setattr__(
            self,
            "verification_policy_cid",
            _cid(self.verification_policy_cid, "verification_policy_cid"),
        )
        object.__setattr__(
            self,
            "comparative_outcome",
            _optional_text(self.comparative_outcome, "comparative_outcome"),
        )
        object.__setattr__(
            self, "checkpoint_cid", _optional_cid(self.checkpoint_cid, "checkpoint_cid")
        )
        object.__setattr__(
            self, "last_result_cid", _optional_cid(self.last_result_cid, "last_result_cid")
        )
        object.__setattr__(
            self,
            "artifacts_included",
            _unique_sorted_tokens(
                list(self.artifacts_included),
                "artifacts_included",
                max_items=MAX_ARTIFACT_IDS,
            ),
        )
        object.__setattr__(
            self,
            "requires_human_review",
            _bool(self.requires_human_review, "requires_human_review"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        _assert_no_compression_blame(
            self.reason_codes,
            compression_blamed=self.compression_blamed,
            comparative_outcome=self.comparative_outcome,
        )
        if self.repaired and self.disposition != ExpansionLoopDisposition.REPAIRED.value:
            raise ExpansionLoopError(
                "repaired=true requires disposition=repaired"
            )
        if (
            self.frontier_escalation_requested
            and self.disposition
            != ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value
        ):
            raise ExpansionLoopError(
                "frontier_escalation_requested requires "
                "disposition=route_escalation_requested"
            )
        if self.compression_blamed:
            raise ExpansionLoopError(
                "expansion loop must never set compression_blamed=true; "
                "both-context failure escalates without blaming compression"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXPANSION_LOOP_RESULT_SCHEMA,
            "interface_id": EXECUTE_EXPANSION_LOOP_INTERFACE,
            "plan_cid": self.plan_cid,
            "disposition": self.disposition,
            "decision_action": self.decision_action,
            "sufficiency_state": self.sufficiency_state,
            "route_tier": self.route_tier,
            "repaired": self.repaired,
            "frontier_escalation_requested": self.frontier_escalation_requested,
            "compression_blamed": self.compression_blamed,
            "context_before_model_escalation": self.context_before_model_escalation,
            "reason_codes": list(self.reason_codes),
            "executed_steps": [step.identity_payload() for step in self.executed_steps],
            "budget": dict(self.budget),
            "model_policy_cid": self.model_policy_cid,
            "verification_policy_cid": self.verification_policy_cid,
            "comparative_outcome": self.comparative_outcome,
            "checkpoint_cid": self.checkpoint_cid,
            "last_result_cid": self.last_result_cid,
            "artifacts_included": list(self.artifacts_included),
            "requires_human_review": self.requires_human_review,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "executed_steps": [step.to_dict() for step in self.executed_steps],
            "result_cid": self.result_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpansionLoopResult":
        if not isinstance(data, Mapping):
            raise ExpansionLoopError("ExpansionLoopResult must be a mapping")
        unknown = set(data) - cls._FIELDS
        if unknown:
            raise ExpansionLoopError(
                f"ExpansionLoopResult has unknown fields: {sorted(unknown)}"
            )
        payload = dict(data)
        claimed = payload.pop("result_cid", None)
        schema = payload.pop("schema", None)
        interface = payload.pop("interface_id", None)
        if schema is not None and schema != EXPANSION_LOOP_RESULT_SCHEMA:
            raise ExpansionLoopError("unsupported ExpansionLoopResult schema")
        if (
            interface is not None
            and interface != EXECUTE_EXPANSION_LOOP_INTERFACE
        ):
            raise ExpansionLoopError("unsupported ExpansionLoopResult interface_id")
        result = cls(**payload)
        if claimed is not None and claimed != result.result_cid:
            raise ExpansionLoopError(
                "ExpansionLoopResult result_cid does not verify"
            )
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_no_compression_blame(
    reason_codes: Sequence[str],
    *,
    compression_blamed: bool,
    comparative_outcome: str | None,
) -> None:
    if compression_blamed:
        raise ExpansionLoopError(
            "compression_blamed must be false; expansion loop never blames "
            "compression for both-context failure"
        )
    blamed = set(reason_codes) & _COMPRESSION_BLAME_REASON_CODES
    if blamed and comparative_outcome in _BOTH_CONTEXT_FAILURE_OUTCOMES:
        raise ExpansionLoopError(
            "both-context failure must not include compression-blame reason "
            f"codes: {sorted(blamed)}"
        )
    if blamed and comparative_outcome is None:
        # Still forbid explicit blame codes on the expansion-loop surface.
        raise ExpansionLoopError(
            f"compression-blame reason codes are forbidden: {sorted(blamed)}"
        )


def _normalize_plan(
    value: ContextExpansionPlan | Mapping[str, Any],
) -> ContextExpansionPlan:
    if isinstance(value, ContextExpansionPlan):
        return value
    if isinstance(value, Mapping):
        try:
            return ContextExpansionPlan.from_dict(value)
        except AuditContractError as exc:
            raise ExpansionLoopError(str(exc)) from exc
    raise ExpansionLoopError("plan must be ContextExpansionPlan or mapping")


def _normalize_model_policy(
    value: ExpansionModelPolicy | Mapping[str, Any] | None,
) -> ExpansionModelPolicy:
    if value is None:
        return default_model_policy()
    if isinstance(value, ExpansionModelPolicy):
        return value
    if isinstance(value, Mapping):
        return ExpansionModelPolicy.from_dict(value)
    raise ExpansionLoopError("model_policy must be ExpansionModelPolicy or mapping")


def _normalize_verification_policy(
    value: ExpansionVerificationPolicy | Mapping[str, Any] | None,
) -> ExpansionVerificationPolicy:
    if value is None:
        return default_verification_policy()
    if isinstance(value, ExpansionVerificationPolicy):
        return value
    if isinstance(value, Mapping):
        return ExpansionVerificationPolicy.from_dict(value)
    raise ExpansionLoopError(
        "verification_policy must be ExpansionVerificationPolicy or mapping"
    )


def _normalize_comparative_outcome(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return ComparativeOutcome(value).value
    except (TypeError, ValueError) as exc:
        raise ExpansionLoopError(
            f"comparative_outcome has unsupported value {value!r}"
        ) from exc


def _is_context_action(action: str) -> bool:
    return action in _CONTEXT_EXPANSION_ACTIONS


def _is_escalate_action(action: str) -> bool:
    return action in _ROUTE_ESCALATION_ACTIONS


def _is_review_action(action: str) -> bool:
    return action in _HUMAN_REVIEW_ACTIONS


def _partition_plan_steps(
    plan: ContextExpansionPlan,
) -> tuple[
    list[ContextExpansionStep],
    list[ContextExpansionStep],
    list[ContextExpansionStep],
    list[ContextExpansionStep],
]:
    context: list[ContextExpansionStep] = []
    escalate: list[ContextExpansionStep] = []
    review: list[ContextExpansionStep] = []
    other: list[ContextExpansionStep] = []
    for step in plan.steps:
        if _is_context_action(step.action):
            context.append(step)
        elif _is_escalate_action(step.action):
            escalate.append(step)
        elif _is_review_action(step.action):
            review.append(step)
        else:
            other.append(step)
    # Preserve plan order within partitions.
    return context, escalate, review, other


def _assert_context_before_escalation(plan: ContextExpansionPlan) -> None:
    """Fail closed when a planned escalate step precedes a context step."""

    saw_escalate = False
    for step in plan.steps:
        if _is_escalate_action(step.action):
            saw_escalate = True
        elif _is_context_action(step.action) and saw_escalate:
            raise ExpansionLoopError(
                "plan violates context_before_model_escalation: context step "
                f"{step.step_id!r} follows an escalate_route step",
                reason_code="context_after_escalation",
            )


def _build_checkpoint(
    *,
    plan: ContextExpansionPlan,
    model_policy: ExpansionModelPolicy,
    verification_policy: ExpansionVerificationPolicy,
    phase: str,
    next_step_index: int,
    ledger: ExpansionBudgetLedger,
    executed_steps: Sequence[ExpansionStepExecution],
    artifacts_included: Sequence[str],
    last_result_cid: str | None,
    comparative_outcome: str | None,
    reason_codes: Sequence[str],
    compression_blamed: bool,
    frontier_escalation_requested: bool,
    repaired: bool,
    generation: int,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExpansionLoopCheckpoint:
    return ExpansionLoopCheckpoint(
        plan_cid=plan.plan_cid,
        model_policy_cid=model_policy.policy_cid,
        verification_policy_cid=verification_policy.policy_cid,
        phase=phase,
        next_step_index=next_step_index,
        budget=ledger.snapshot(),
        executed_steps=tuple(executed_steps),
        artifacts_included=tuple(artifacts_included),
        last_result_cid=last_result_cid,
        comparative_outcome=comparative_outcome,
        reason_codes=tuple(reason_codes),
        compression_blamed=compression_blamed,
        frontier_escalation_requested=frontier_escalation_requested,
        repaired=repaired,
        generation=generation,
        notes=notes,
        metadata=dict(metadata or {"evidence": SCG_EXPANSION_LOOP_EVIDENCE}),
    )


def _decision_for_disposition(
    disposition: str,
    *,
    repaired: bool,
    frontier: bool,
    human_review: bool,
) -> tuple[str, str, str]:
    """Return (decision_action, sufficiency_state, route_tier_hint)."""

    if human_review or disposition == ExpansionLoopDisposition.HUMAN_REVIEW_REQUIRED.value:
        return (
            DecisionAction.REQUIRE_HUMAN_REVIEW.value,
            ContextSufficiencyState.HUMAN_REVIEW_REQUIRED.value,
            RouteTier.HUMAN.value,
        )
    if repaired or disposition == ExpansionLoopDisposition.REPAIRED.value:
        return (
            DecisionAction.RETRY_SAME_ROUTE.value,
            ContextSufficiencyState.SUFFICIENT_WITH_CAVEATS.value,
            RouteTier.MEDIUM.value,
        )
    if frontier or disposition == ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value:
        return (
            DecisionAction.ESCALATE_FRONTIER.value,
            ContextSufficiencyState.FRONTIER_ESCALATION_REQUIRED.value,
            RouteTier.FRONTIER.value,
        )
    if disposition == ExpansionLoopDisposition.LIMITS_EXHAUSTED.value:
        return (
            DecisionAction.MARK_INCONCLUSIVE.value,
            ContextSufficiencyState.INCONCLUSIVE.value,
            RouteTier.MEDIUM.value,
        )
    if disposition == ExpansionLoopDisposition.NO_ACTION.value:
        return (
            DecisionAction.MARK_INCONCLUSIVE.value,
            ContextSufficiencyState.INCONCLUSIVE.value,
            RouteTier.MEDIUM.value,
        )
    if disposition == ExpansionLoopDisposition.CANCELLED.value:
        return (
            DecisionAction.MARK_INCONCLUSIVE.value,
            ContextSufficiencyState.INCONCLUSIVE.value,
            RouteTier.MEDIUM.value,
        )
    return (
        DecisionAction.REJECT.value,
        ContextSufficiencyState.EVALUATION_FAILED.value,
        RouteTier.MEDIUM.value,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def both_context_failure_outcomes() -> tuple[str, ...]:
    """Return comparative outcomes that never blame compression via omission."""

    return tuple(sorted(_BOTH_CONTEXT_FAILURE_OUTCOMES))


def omission_supporting_outcomes() -> tuple[str, ...]:
    """Return comparative outcomes that support ranked omission repair."""

    return tuple(sorted(_OMISSION_SUPPORTING_OUTCOMES))


def compression_blame_reason_codes() -> tuple[str, ...]:
    """Return reason codes forbidden when both contexts fail."""

    return tuple(sorted(_COMPRESSION_BLAME_REASON_CODES))


def execute_expansion_loop_interface_id() -> str:
    """Return the versioned public interface pin for this runtime."""

    return EXECUTE_EXPANSION_LOOP_INTERFACE


def execute_expansion_loop(
    plan: ContextExpansionPlan | Mapping[str, Any],
    model_policy: ExpansionModelPolicy | Mapping[str, Any] | None = None,
    verification_policy: ExpansionVerificationPolicy | Mapping[str, Any] | None = None,
    *,
    runner: ExpansionStepRunner | None = None,
    checkpoint_store: ExpansionCheckpointStore | None = None,
    checkpoint: ExpansionLoopCheckpoint | Mapping[str, Any] | None = None,
    comparative_outcome: ComparativeOutcome | str | None = None,
    counterexample_cids: Sequence[str] = (),
    cancel_requested: Callable[[], bool] | None = None,
    clock: Callable[[], float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExpansionLoopResult:
    """Execute bounded counterexample-guided context expansion before escalation.

    Parameters
    ----------
    plan:
        Durable :class:`ContextExpansionPlan` (or closed mapping) with hard
        step/token/retry/escalation/time/spend limits.
    model_policy:
        Same-route retry and frontier-escalation policy. Defaults admit same
        route retry and one frontier escalation.
    verification_policy:
        Gates applied after each expansion apply/retry cycle.
    runner:
        Injectable step runner. Defaults to :class:`AlwaysFailRunner` so live
        callers must supply a real runner; offline tests inject scripts.
    checkpoint_store:
        Optional durable store. When provided, a checkpoint is written after
        each completed step so limits survive process restart.
    checkpoint:
        Optional prior checkpoint to resume. When omitted, ``checkpoint_store``
        is consulted by ``plan_cid``.
    comparative_outcome:
        Optional prior differential outcome (e.g. both-context failure). When
        both contexts failed, the loop may request route escalation **without**
        blaming compression.
    counterexample_cids:
        Optional minimized counterexample CIDs bound into result metadata.
    cancel_requested:
        Optional cooperative cancel probe.
    clock:
        Optional monotonic clock for wall-time accounting (tests inject).
    metadata:
        Optional closed metadata merged into the result.

    Returns
    -------
    ExpansionLoopResult
        Closed disposition, decision action, executed steps, and budget ledger
        snapshot. Expanded/repaired outcomes never auto-accept production.
    """

    resolved_plan = _normalize_plan(plan)
    resolved_model = _normalize_model_policy(model_policy)
    resolved_verify = _normalize_verification_policy(verification_policy)
    outcome = _normalize_comparative_outcome(comparative_outcome)
    cex_cids = tuple(
        _cid(item, "counterexample_cids") for item in (counterexample_cids or ())
    )
    if len(cex_cids) > MAX_IDS:
        raise ExpansionLoopError("counterexample_cids exceeds maximum length")

    _assert_context_before_escalation(resolved_plan)

    step_runner: ExpansionStepRunner = (
        runner if runner is not None else AlwaysFailRunner()
    )
    mono = clock if clock is not None else _DEFAULT_CLOCK
    started = float(mono())

    ledger = ExpansionBudgetLedger.from_plan(resolved_plan)
    executed: list[ExpansionStepExecution] = []
    artifacts: list[str] = []
    reason_codes: list[str] = []
    last_result_cid: str | None = None
    next_index = 0
    generation = 0
    repaired = False
    frontier_requested = False
    compression_blamed = False
    phase = ExpansionLoopPhase.ADMITTED.value
    route_tier = resolved_model.current_route_tier

    # --- Restore durable checkpoint (restart path) --------------------------
    prior: ExpansionLoopCheckpoint | None = None
    if checkpoint is not None:
        if isinstance(checkpoint, ExpansionLoopCheckpoint):
            prior = checkpoint
        elif isinstance(checkpoint, Mapping):
            prior = ExpansionLoopCheckpoint.from_dict(checkpoint)
        else:
            raise ExpansionCheckpointError(
                "checkpoint must be ExpansionLoopCheckpoint or mapping"
            )
    elif checkpoint_store is not None:
        prior = checkpoint_store.load(resolved_plan.plan_cid)

    if prior is not None:
        if prior.plan_cid != resolved_plan.plan_cid:
            raise ExpansionCheckpointError(
                "checkpoint plan_cid does not match plan",
                reason_code="checkpoint_plan_mismatch",
            )
        if prior.model_policy_cid != resolved_model.policy_cid:
            raise ExpansionCheckpointError(
                "checkpoint model_policy_cid does not match model_policy",
                reason_code="checkpoint_policy_mismatch",
            )
        if prior.verification_policy_cid != resolved_verify.policy_cid:
            raise ExpansionCheckpointError(
                "checkpoint verification_policy_cid does not match "
                "verification_policy",
                reason_code="checkpoint_policy_mismatch",
            )
        ledger.apply_snapshot(prior.budget)
        executed = list(prior.executed_steps)
        artifacts = list(prior.artifacts_included)
        reason_codes = list(prior.reason_codes)
        last_result_cid = prior.last_result_cid
        next_index = int(prior.next_step_index)
        generation = int(prior.generation) + 1
        repaired = bool(prior.repaired)
        frontier_requested = bool(prior.frontier_escalation_requested)
        compression_blamed = bool(prior.compression_blamed)
        if prior.comparative_outcome is not None:
            outcome = prior.comparative_outcome
        phase = prior.phase
        if repaired:
            # Already repaired before restart — return closed repaired result.
            return _finalize_result(
                plan=resolved_plan,
                model_policy=resolved_model,
                verification_policy=resolved_verify,
                disposition=ExpansionLoopDisposition.REPAIRED.value,
                repaired=True,
                frontier_requested=False,
                compression_blamed=False,
                reason_codes=reason_codes or ("omission_repair_verified",),
                executed=executed,
                ledger=ledger,
                outcome=outcome,
                last_result_cid=last_result_cid,
                artifacts=artifacts,
                checkpoint_cid=prior.checkpoint_cid,
                route_tier=route_tier,
                notes="Resumed checkpoint already repaired; no further expansion",
                metadata=metadata,
                counterexample_cids=cex_cids,
                context_before=True,
            )

    context_steps, escalate_steps, review_steps, other_steps = _partition_plan_steps(
        resolved_plan
    )
    all_ordered = list(resolved_plan.steps)

    def _persist(current_phase: str, index: int) -> ExpansionLoopCheckpoint:
        ckpt = _build_checkpoint(
            plan=resolved_plan,
            model_policy=resolved_model,
            verification_policy=resolved_verify,
            phase=current_phase,
            next_step_index=index,
            ledger=ledger,
            executed_steps=executed,
            artifacts_included=artifacts,
            last_result_cid=last_result_cid,
            comparative_outcome=outcome,
            reason_codes=reason_codes,
            compression_blamed=compression_blamed,
            frontier_escalation_requested=frontier_requested,
            repaired=repaired,
            generation=generation,
        )
        if checkpoint_store is not None:
            checkpoint_store.save(ckpt)
        return ckpt

    last_checkpoint = _persist(phase, next_index)

    # Empty plan → no-action (bounded).
    if not all_ordered and not repaired:
        reason_codes = sorted(set(reason_codes) | {"no_action"})
        # Both-context failure with no expansion path still may escalate.
        if outcome in _BOTH_CONTEXT_FAILURE_OUTCOMES and resolved_model.allow_frontier_escalation:
            frontier_requested = True
            reason_codes = sorted(
                set(reason_codes)
                | {
                    "both_context_failure",
                    "route_escalation_without_omission_blame",
                    "no_supported_context_expansion",
                }
            )
            last_checkpoint = _persist(
                ExpansionLoopPhase.ROUTE_ESCALATION.value, next_index
            )
            return _finalize_result(
                plan=resolved_plan,
                model_policy=resolved_model,
                verification_policy=resolved_verify,
                disposition=ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value,
                repaired=False,
                frontier_requested=True,
                compression_blamed=False,
                reason_codes=reason_codes,
                executed=executed,
                ledger=ledger,
                outcome=outcome,
                last_result_cid=last_result_cid,
                artifacts=artifacts,
                checkpoint_cid=last_checkpoint.checkpoint_cid,
                route_tier=resolved_model.frontier_route_tier,
                notes=(
                    "Both-context failure with empty expansion plan; route "
                    "escalation requested without blaming compression"
                ),
                metadata=metadata,
                counterexample_cids=cex_cids,
                context_before=True,
            )
        last_checkpoint = _persist(ExpansionLoopPhase.COMPLETE.value, next_index)
        return _finalize_result(
            plan=resolved_plan,
            model_policy=resolved_model,
            verification_policy=resolved_verify,
            disposition=ExpansionLoopDisposition.NO_ACTION.value,
            repaired=False,
            frontier_requested=False,
            compression_blamed=False,
            reason_codes=reason_codes,
            executed=executed,
            ledger=ledger,
            outcome=outcome,
            last_result_cid=last_result_cid,
            artifacts=artifacts,
            checkpoint_cid=last_checkpoint.checkpoint_cid,
            route_tier=route_tier,
            notes="Empty expansion plan; no action",
            metadata=metadata,
            counterexample_cids=cex_cids,
            context_before=True,
        )

    # --- Main loop over planned steps in order ------------------------------
    limits_exhausted_kind: str | None = None

    while next_index < len(all_ordered) and not repaired:
        if cancel_requested is not None and cancel_requested():
            reason_codes = sorted(set(reason_codes) | {"cancelled"})
            last_checkpoint = _persist(
                ExpansionLoopPhase.CANCELLED.value, next_index
            )
            return _finalize_result(
                plan=resolved_plan,
                model_policy=resolved_model,
                verification_policy=resolved_verify,
                disposition=ExpansionLoopDisposition.CANCELLED.value,
                repaired=False,
                frontier_requested=False,
                compression_blamed=False,
                reason_codes=reason_codes,
                executed=executed,
                ledger=ledger,
                outcome=outcome,
                last_result_cid=last_result_cid,
                artifacts=artifacts,
                checkpoint_cid=last_checkpoint.checkpoint_cid,
                route_tier=route_tier,
                notes="Expansion loop cancelled",
                metadata=metadata,
                counterexample_cids=cex_cids,
                context_before=True,
            )

        step = all_ordered[next_index]
        elapsed_ms = max(0, int((float(mono()) - started) * 1000))
        # Account wall time growth against the ledger remaining.
        if resolved_plan.max_wall_time_ms > 0:
            already = ledger.snapshot()["spent_wall_time_ms"]
            delta = max(0, elapsed_ms - already)
            try:
                if delta:
                    ledger.recheck(wall_time_ms=delta)
            except ExpansionLimitExceededError as exc:
                limits_exhausted_kind = exc.limit_kind
                reason_codes = sorted(
                    set(reason_codes) | {"limit_exceeded", exc.limit_kind}
                )
                break

        # Human review steps short-circuit.
        if _is_review_action(step.action):
            try:
                ledger.record(steps=1)
            except ExpansionLimitExceededError as exc:
                limits_exhausted_kind = exc.limit_kind
                reason_codes = sorted(
                    set(reason_codes) | {"limit_exceeded", exc.limit_kind}
                )
                break
            exec_rec = ExpansionStepExecution(
                step_id=step.step_id,
                step_index=step.step_index,
                action=step.action,
                planned_status=step.status,
                executed_status=ExpansionStepStatus.APPLIED.value,
                token_increase=0,
                artifact_ids_added=(),
                hypothesis_cid=step.hypothesis_cid,
                hypothesis_supported=None,
                attempts=(),
                reason_codes=("human_review_required",),
                notes=step.notes or "Human review required by expansion plan",
            )
            executed.append(exec_rec)
            reason_codes = sorted(set(reason_codes) | {"human_review_required"})
            next_index += 1
            last_checkpoint = _persist(
                ExpansionLoopPhase.HUMAN_REVIEW.value, next_index
            )
            return _finalize_result(
                plan=resolved_plan,
                model_policy=resolved_model,
                verification_policy=resolved_verify,
                disposition=ExpansionLoopDisposition.HUMAN_REVIEW_REQUIRED.value,
                repaired=False,
                frontier_requested=False,
                compression_blamed=False,
                reason_codes=reason_codes,
                executed=executed,
                ledger=ledger,
                outcome=outcome,
                last_result_cid=last_result_cid,
                artifacts=artifacts,
                checkpoint_cid=last_checkpoint.checkpoint_cid,
                route_tier=RouteTier.HUMAN.value,
                notes="Plan requested human review; expansion halted",
                metadata=metadata,
                counterexample_cids=cex_cids,
                context_before=True,
                requires_human_review=True,
            )

        # Context expansion: apply, same-route retry, verify.
        if _is_context_action(step.action):
            phase = ExpansionLoopPhase.CONTEXT_EXPANSION.value
            token_cost = int(step.token_increase)
            try:
                ledger.recheck(steps=1, tokens=token_cost)
            except ExpansionLimitExceededError as exc:
                limits_exhausted_kind = exc.limit_kind
                reason_codes = sorted(
                    set(reason_codes) | {"limit_exceeded", exc.limit_kind}
                )
                # Mark step budget_exceeded without applying.
                executed.append(
                    ExpansionStepExecution(
                        step_id=step.step_id,
                        step_index=step.step_index,
                        action=step.action,
                        planned_status=step.status,
                        executed_status=ExpansionStepStatus.BUDGET_EXCEEDED.value,
                        token_increase=0,
                        artifact_ids_added=(),
                        hypothesis_cid=step.hypothesis_cid,
                        reason_codes=(exc.limit_kind, "budget_exceeded"),
                        notes=str(exc),
                    )
                )
                next_index += 1
                last_checkpoint = _persist(
                    ExpansionLoopPhase.LIMITS_EXHAUSTED.value, next_index
                )
                break

            # Charge the planned expansion tokens + step slot.
            ledger.record(steps=1, tokens=token_cost)
            for artifact_id in step.artifact_ids_added:
                if artifact_id not in artifacts:
                    artifacts.append(artifact_id)

            attempts: list[ExpansionAttemptResult] = []
            step_repaired = False
            prior_cid = last_result_cid
            new_cid = last_result_cid
            # First attempt after expansion (same route).
            max_attempts = 1 + max(0, ledger.remaining(ExpansionLimitKind.RETRIES.value))
            # Cap per-step attempts to remaining retries + 1 initial try.
            attempt_i = 0
            while attempt_i < max_attempts and not step_repaired:
                if attempt_i > 0:
                    # Same-route retry counts against retry budget.
                    if not resolved_model.allow_same_route_retry:
                        reason_codes = sorted(
                            set(reason_codes) | {"same_route_retry_disabled"}
                        )
                        break
                    try:
                        ledger.record(retries=1)
                    except ExpansionLimitExceededError as exc:
                        limits_exhausted_kind = exc.limit_kind
                        reason_codes = sorted(
                            set(reason_codes) | {"limit_exceeded", exc.limit_kind}
                        )
                        break

                phase = ExpansionLoopPhase.SAME_ROUTE_RETRY.value
                try:
                    attempt = step_runner.apply_and_verify(
                        step,
                        artifacts_included=tuple(artifacts),
                        route_tier=route_tier,
                        attempt_index=attempt_i,
                        model_policy=resolved_model,
                        verification_policy=resolved_verify,
                    )
                except ExpansionLimitExceededError as exc:
                    limits_exhausted_kind = exc.limit_kind
                    reason_codes = sorted(
                        set(reason_codes) | {"limit_exceeded", exc.limit_kind}
                    )
                    break
                except ExpansionLoopError:
                    raise
                except Exception as exc:  # noqa: BLE001 — fail closed
                    raise ExpansionLoopError(
                        f"step runner failed: {exc}",
                        reason_code="runner_failed",
                    ) from exc

                # Charge attempt wall/spend/token_cost.
                try:
                    ledger.record(
                        wall_time_ms=int(attempt.wall_time_ms),
                        spend_micros=int(attempt.spend_micros),
                        tokens=int(attempt.token_cost),
                    )
                except ExpansionLimitExceededError as exc:
                    # Attempt ran but over budget — record and stop.
                    limits_exhausted_kind = exc.limit_kind
                    reason_codes = sorted(
                        set(reason_codes) | {"limit_exceeded", exc.limit_kind}
                    )
                    attempts.append(attempt)
                    break

                attempts.append(attempt)
                if attempt.result_cid is not None:
                    new_cid = attempt.result_cid
                    last_result_cid = attempt.result_cid

                phase = ExpansionLoopPhase.VERIFY.value
                if resolved_verify.evaluate(attempt):
                    step_repaired = True
                    repaired = True
                    reason_codes = sorted(
                        set(reason_codes)
                        | {
                            "omission_repair_verified",
                            "supported_omission_repaired_before_frontier",
                        }
                        | set(attempt.reason_codes)
                    )
                    break
                attempt_i += 1

            executed_status = (
                ExpansionStepStatus.SUPPORTED.value
                if step_repaired
                else ExpansionStepStatus.APPLIED.value
            )
            if limits_exhausted_kind and not step_repaired and not attempts:
                executed_status = ExpansionStepStatus.BUDGET_EXCEEDED.value
            elif not step_repaired and attempts:
                # Expansion applied but hypothesis not yet supported.
                executed_status = ExpansionStepStatus.APPLIED.value

            step_reason_codes: list[str] = [
                "omission_repair" if step_repaired else "context_expansion_applied"
            ]
            if limits_exhausted_kind:
                step_reason_codes.append(limits_exhausted_kind)
            exec_rec = ExpansionStepExecution(
                step_id=step.step_id,
                step_index=step.step_index,
                action=step.action,
                planned_status=step.status,
                executed_status=executed_status,
                token_increase=(
                    token_cost
                    if executed_status != ExpansionStepStatus.BUDGET_EXCEEDED.value
                    else 0
                ),
                artifact_ids_added=tuple(step.artifact_ids_added),
                hypothesis_cid=step.hypothesis_cid,
                hypothesis_supported=True if step_repaired else False,
                attempts=tuple(attempts),
                prior_result_cid=prior_cid,
                new_result_cid=new_cid,
                reason_codes=tuple(sorted(set(step_reason_codes))),
                notes=(
                    "Supported omission expansion repaired failure before "
                    "frontier escalation"
                    if step_repaired
                    else "Context expansion applied; verification still failing"
                ),
            )
            executed.append(exec_rec)
            next_index += 1
            last_checkpoint = _persist(
                ExpansionLoopPhase.COMPLETE.value
                if repaired
                else (
                    ExpansionLoopPhase.LIMITS_EXHAUSTED.value
                    if limits_exhausted_kind
                    else ExpansionLoopPhase.CONTEXT_EXPANSION.value
                ),
                next_index,
            )
            if repaired:
                break
            if limits_exhausted_kind:
                break
            continue

        # Route escalation steps — only after context steps (ordering already checked).
        if _is_escalate_action(step.action):
            # If any earlier context step exists and was not executed, refuse.
            earlier_context = [
                s
                for s in all_ordered
                if s.step_index < step.step_index and _is_context_action(s.action)
            ]
            executed_ids = {e.step_id for e in executed}
            for earlier in earlier_context:
                if earlier.step_id not in executed_ids:
                    raise ExpansionLoopError(
                        "refusing route escalation before supported context "
                        f"expansion step {earlier.step_id!r}",
                        reason_code="escalation_before_context",
                    )

            if not resolved_model.allow_frontier_escalation:
                executed.append(
                    ExpansionStepExecution(
                        step_id=step.step_id,
                        step_index=step.step_index,
                        action=step.action,
                        planned_status=step.status,
                        executed_status=ExpansionStepStatus.SKIPPED.value,
                        token_increase=0,
                        artifact_ids_added=(),
                        hypothesis_cid=step.hypothesis_cid,
                        reason_codes=("frontier_escalation_disabled",),
                        notes="Frontier escalation disabled by model policy",
                    )
                )
                next_index += 1
                last_checkpoint = _persist(phase, next_index)
                continue

            # Effective escalations: plan max and model policy max.
            effective_max_esc = min(
                resolved_plan.max_escalations,
                resolved_model.max_route_escalations,
            )
            if ledger.spent_escalations >= effective_max_esc:
                limits_exhausted_kind = ExpansionLimitKind.ESCALATIONS.value
                executed.append(
                    ExpansionStepExecution(
                        step_id=step.step_id,
                        step_index=step.step_index,
                        action=step.action,
                        planned_status=step.status,
                        executed_status=ExpansionStepStatus.BUDGET_EXCEEDED.value,
                        token_increase=0,
                        artifact_ids_added=(),
                        hypothesis_cid=step.hypothesis_cid,
                        reason_codes=("escalations", "budget_exceeded"),
                        notes="Escalation budget exhausted",
                    )
                )
                next_index += 1
                reason_codes = sorted(
                    set(reason_codes) | {"limit_exceeded", "escalations"}
                )
                last_checkpoint = _persist(
                    ExpansionLoopPhase.LIMITS_EXHAUSTED.value, next_index
                )
                break

            try:
                ledger.record(steps=1, escalations=1)
            except ExpansionLimitExceededError as exc:
                limits_exhausted_kind = exc.limit_kind
                reason_codes = sorted(
                    set(reason_codes) | {"limit_exceeded", exc.limit_kind}
                )
                executed.append(
                    ExpansionStepExecution(
                        step_id=step.step_id,
                        step_index=step.step_index,
                        action=step.action,
                        planned_status=step.status,
                        executed_status=ExpansionStepStatus.BUDGET_EXCEEDED.value,
                        token_increase=0,
                        artifact_ids_added=(),
                        hypothesis_cid=step.hypothesis_cid,
                        reason_codes=(exc.limit_kind, "budget_exceeded"),
                        notes=str(exc),
                    )
                )
                next_index += 1
                last_checkpoint = _persist(
                    ExpansionLoopPhase.LIMITS_EXHAUSTED.value, next_index
                )
                break

            frontier_requested = True
            route_tier = resolved_model.frontier_route_tier
            esc_reasons = {
                "model_route_after_context",
                "route_escalation_requested",
            }
            if outcome in _BOTH_CONTEXT_FAILURE_OUTCOMES:
                esc_reasons |= {
                    "both_context_failure",
                    "route_escalation_without_omission_blame",
                }
            # Explicitly never blame compression.
            compression_blamed = False
            reason_codes = sorted(set(reason_codes) | esc_reasons)
            executed.append(
                ExpansionStepExecution(
                    step_id=step.step_id,
                    step_index=step.step_index,
                    action=step.action,
                    planned_status=step.status,
                    executed_status=ExpansionStepStatus.APPLIED.value,
                    token_increase=0,
                    artifact_ids_added=(),
                    hypothesis_cid=step.hypothesis_cid,
                    hypothesis_supported=None,
                    reason_codes=tuple(sorted(esc_reasons)),
                    notes=(
                        "Route escalation after context expansion; compression "
                        "is not blamed"
                        if outcome in _BOTH_CONTEXT_FAILURE_OUTCOMES
                        or any(_is_context_action(e.action) for e in executed[:-1])
                        else "Route escalation requested"
                    ),
                    metadata={
                        "compression_blamed": False,
                        "comparative_outcome": outcome,
                    },
                )
            )
            next_index += 1
            last_checkpoint = _persist(
                ExpansionLoopPhase.ROUTE_ESCALATION.value, next_index
            )
            # One successful escalation request is terminal for this loop.
            break

        # no_action / other steps — record and continue.
        try:
            ledger.record(steps=1)
        except ExpansionLimitExceededError as exc:
            limits_exhausted_kind = exc.limit_kind
            reason_codes = sorted(
                set(reason_codes) | {"limit_exceeded", exc.limit_kind}
            )
            break
        executed.append(
            ExpansionStepExecution(
                step_id=step.step_id,
                step_index=step.step_index,
                action=step.action,
                planned_status=step.status,
                executed_status=ExpansionStepStatus.SKIPPED.value,
                token_increase=0,
                artifact_ids_added=(),
                hypothesis_cid=step.hypothesis_cid,
                reason_codes=("no_action",),
                notes="Non-expansion step skipped",
            )
        )
        next_index += 1
        last_checkpoint = _persist(phase, next_index)

    # --- Terminal disposition -----------------------------------------------
    if repaired:
        last_checkpoint = _persist(ExpansionLoopPhase.COMPLETE.value, next_index)
        return _finalize_result(
            plan=resolved_plan,
            model_policy=resolved_model,
            verification_policy=resolved_verify,
            disposition=ExpansionLoopDisposition.REPAIRED.value,
            repaired=True,
            frontier_requested=False,
            compression_blamed=False,
            reason_codes=reason_codes
            or ("omission_repair_verified", "supported_omission_repaired_before_frontier"),
            executed=executed,
            ledger=ledger,
            outcome=outcome,
            last_result_cid=last_result_cid,
            artifacts=artifacts,
            checkpoint_cid=last_checkpoint.checkpoint_cid,
            route_tier=resolved_model.current_route_tier,
            notes=(
                "Supported omission expansion repaired the failure before "
                "frontier escalation"
            ),
            metadata=metadata,
            counterexample_cids=cex_cids,
            context_before=True,
        )

    if frontier_requested:
        last_checkpoint = _persist(
            ExpansionLoopPhase.ROUTE_ESCALATION.value, next_index
        )
        if outcome in _BOTH_CONTEXT_FAILURE_OUTCOMES:
            reason_codes = sorted(
                set(reason_codes)
                | {
                    "both_context_failure",
                    "route_escalation_without_omission_blame",
                }
            )
        return _finalize_result(
            plan=resolved_plan,
            model_policy=resolved_model,
            verification_policy=resolved_verify,
            disposition=ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value,
            repaired=False,
            frontier_requested=True,
            compression_blamed=False,
            reason_codes=reason_codes or ("route_escalation_requested",),
            executed=executed,
            ledger=ledger,
            outcome=outcome,
            last_result_cid=last_result_cid,
            artifacts=artifacts,
            checkpoint_cid=last_checkpoint.checkpoint_cid,
            route_tier=resolved_model.frontier_route_tier,
            notes=(
                "Both-context failure requests route escalation without "
                "blaming compression"
                if outcome in _BOTH_CONTEXT_FAILURE_OUTCOMES
                else "Context expansion insufficient; route escalation requested"
            ),
            metadata=metadata,
            counterexample_cids=cex_cids,
            context_before=True,
        )

    # Context exhausted without repair — may still escalate for both-fail.
    if (
        outcome in _BOTH_CONTEXT_FAILURE_OUTCOMES
        and resolved_model.allow_frontier_escalation
        and not frontier_requested
    ):
        # Request escalation even if the plan had no escalate_route step.
        if ledger.remaining(ExpansionLimitKind.ESCALATIONS.value) > 0 and (
            ledger.spent_escalations < resolved_model.max_route_escalations
        ):
            try:
                ledger.record(escalations=1)
                frontier_requested = True
                reason_codes = sorted(
                    set(reason_codes)
                    | {
                        "both_context_failure",
                        "route_escalation_without_omission_blame",
                        "route_escalation_requested",
                    }
                )
                last_checkpoint = _persist(
                    ExpansionLoopPhase.ROUTE_ESCALATION.value, next_index
                )
                return _finalize_result(
                    plan=resolved_plan,
                    model_policy=resolved_model,
                    verification_policy=resolved_verify,
                    disposition=ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value,
                    repaired=False,
                    frontier_requested=True,
                    compression_blamed=False,
                    reason_codes=reason_codes,
                    executed=executed,
                    ledger=ledger,
                    outcome=outcome,
                    last_result_cid=last_result_cid,
                    artifacts=artifacts,
                    checkpoint_cid=last_checkpoint.checkpoint_cid,
                    route_tier=resolved_model.frontier_route_tier,
                    notes=(
                        "Both-context failure after context expansion; route "
                        "escalation requested without blaming compression"
                    ),
                    metadata=metadata,
                    counterexample_cids=cex_cids,
                    context_before=True,
                )
            except ExpansionLimitExceededError as exc:
                limits_exhausted_kind = exc.limit_kind
                reason_codes = sorted(
                    set(reason_codes) | {"limit_exceeded", exc.limit_kind}
                )

    if limits_exhausted_kind:
        last_checkpoint = _persist(
            ExpansionLoopPhase.LIMITS_EXHAUSTED.value, next_index
        )
        return _finalize_result(
            plan=resolved_plan,
            model_policy=resolved_model,
            verification_policy=resolved_verify,
            disposition=ExpansionLoopDisposition.LIMITS_EXHAUSTED.value,
            repaired=False,
            frontier_requested=False,
            compression_blamed=False,
            reason_codes=reason_codes
            or ("limit_exceeded", limits_exhausted_kind),
            executed=executed,
            ledger=ledger,
            outcome=outcome,
            last_result_cid=last_result_cid,
            artifacts=artifacts,
            checkpoint_cid=last_checkpoint.checkpoint_cid,
            route_tier=route_tier,
            notes=f"Hard expansion limit exhausted: {limits_exhausted_kind}",
            metadata=metadata,
            counterexample_cids=cex_cids,
            context_before=True,
        )

    # Context steps applied but verification still failing; no escalate path.
    if executed and any(_is_context_action(e.action) for e in executed):
        last_checkpoint = _persist(ExpansionLoopPhase.COMPLETE.value, next_index)
        return _finalize_result(
            plan=resolved_plan,
            model_policy=resolved_model,
            verification_policy=resolved_verify,
            disposition=ExpansionLoopDisposition.FAILED.value,
            repaired=False,
            frontier_requested=False,
            compression_blamed=False,
            reason_codes=reason_codes
            or ("context_expansion_insufficient",),
            executed=executed,
            ledger=ledger,
            outcome=outcome,
            last_result_cid=last_result_cid,
            artifacts=artifacts,
            checkpoint_cid=last_checkpoint.checkpoint_cid,
            route_tier=route_tier,
            notes="Context expansion completed without repair",
            metadata=metadata,
            counterexample_cids=cex_cids,
            context_before=True,
        )

    last_checkpoint = _persist(ExpansionLoopPhase.COMPLETE.value, next_index)
    return _finalize_result(
        plan=resolved_plan,
        model_policy=resolved_model,
        verification_policy=resolved_verify,
        disposition=ExpansionLoopDisposition.INCONCLUSIVE.value,
        repaired=False,
        frontier_requested=False,
        compression_blamed=False,
        reason_codes=reason_codes or ("inconclusive",),
        executed=executed,
        ledger=ledger,
        outcome=outcome,
        last_result_cid=last_result_cid,
        artifacts=artifacts,
        checkpoint_cid=last_checkpoint.checkpoint_cid,
        route_tier=route_tier,
        notes="Expansion loop finished without decisive repair or escalation",
        metadata=metadata,
        counterexample_cids=cex_cids,
        context_before=True,
    )


def _finalize_result(
    *,
    plan: ContextExpansionPlan,
    model_policy: ExpansionModelPolicy,
    verification_policy: ExpansionVerificationPolicy,
    disposition: str,
    repaired: bool,
    frontier_requested: bool,
    compression_blamed: bool,
    reason_codes: Sequence[str],
    executed: Sequence[ExpansionStepExecution],
    ledger: ExpansionBudgetLedger,
    outcome: str | None,
    last_result_cid: str | None,
    artifacts: Sequence[str],
    checkpoint_cid: str | None,
    route_tier: str,
    notes: str | None,
    metadata: Mapping[str, Any] | None,
    counterexample_cids: Sequence[str],
    context_before: bool,
    requires_human_review: bool = False,
) -> ExpansionLoopResult:
    # Hard invariant: never blame compression.
    if compression_blamed:
        compression_blamed = False
    codes = list(reason_codes)
    # Strip any accidental blame codes.
    codes = [c for c in codes if c not in _COMPRESSION_BLAME_REASON_CODES]
    if not codes:
        codes = ["inconclusive"]

    decision_action, sufficiency, _hint = _decision_for_disposition(
        disposition,
        repaired=repaired,
        frontier=frontier_requested,
        human_review=requires_human_review,
    )
    # Prefer explicit route_tier when frontier/human requested.
    if frontier_requested:
        route_tier = model_policy.frontier_route_tier
        decision_action = DecisionAction.ESCALATE_FRONTIER.value
        sufficiency = ContextSufficiencyState.FRONTIER_ESCALATION_REQUIRED.value
    if requires_human_review:
        route_tier = RouteTier.HUMAN.value
        decision_action = DecisionAction.REQUIRE_HUMAN_REVIEW.value
        sufficiency = ContextSufficiencyState.HUMAN_REVIEW_REQUIRED.value
    if repaired:
        route_tier = model_policy.current_route_tier
        decision_action = DecisionAction.RETRY_SAME_ROUTE.value
        sufficiency = ContextSufficiencyState.SUFFICIENT_WITH_CAVEATS.value

    meta: dict[str, Any] = {
        "evidence": SCG_EXPANSION_LOOP_EVIDENCE,
        "interface_id": EXECUTE_EXPANSION_LOOP_INTERFACE,
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "audit_case_cid": plan.audit_case_cid,
        "omission_evidence_cid": plan.omission_evidence_cid,
        "counterexample_cids": list(counterexample_cids),
        "context_step_count": sum(
            1 for e in executed if _is_context_action(e.action)
        ),
        "escalation_step_count": sum(
            1 for e in executed if _is_escalate_action(e.action)
        ),
    }
    if metadata:
        thawed = _mapping(metadata, "metadata")
        meta.update(_thaw_structured(thawed))

    return ExpansionLoopResult(
        plan_cid=plan.plan_cid,
        disposition=disposition,
        decision_action=decision_action,
        sufficiency_state=sufficiency,
        route_tier=route_tier,
        repaired=repaired,
        frontier_escalation_requested=frontier_requested,
        compression_blamed=False,
        context_before_model_escalation=context_before,
        reason_codes=tuple(sorted(set(codes))),
        executed_steps=tuple(executed),
        budget=ledger.snapshot(),
        model_policy_cid=model_policy.policy_cid,
        verification_policy_cid=verification_policy.policy_cid,
        comparative_outcome=outcome,
        checkpoint_cid=checkpoint_cid,
        last_result_cid=last_result_cid,
        artifacts_included=tuple(artifacts),
        requires_human_review=requires_human_review,
        notes=notes,
        metadata=meta,
    )


__all__ = [
    "EXECUTE_EXPANSION_LOOP_INTERFACE",
    "EXPANSION_ATTEMPT_RESULT_SCHEMA",
    "EXPANSION_LOOP_CHECKPOINT_SCHEMA",
    "EXPANSION_LOOP_RESULT_INTERFACE",
    "EXPANSION_LOOP_RESULT_SCHEMA",
    "EXPANSION_MODEL_POLICY_SCHEMA",
    "EXPANSION_STEP_EXECUTION_SCHEMA",
    "EXPANSION_VERIFICATION_POLICY_SCHEMA",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "SCG_EXPANSION_LOOP_EVIDENCE",
    "AlwaysFailRunner",
    "ExpansionAttemptResult",
    "ExpansionAttemptStatus",
    "ExpansionBudgetLedger",
    "ExpansionCheckpointError",
    "ExpansionCheckpointStore",
    "ExpansionLimitExceededError",
    "ExpansionLimitKind",
    "ExpansionLoopCheckpoint",
    "ExpansionLoopDisposition",
    "ExpansionLoopError",
    "ExpansionLoopPhase",
    "ExpansionLoopResult",
    "ExpansionModelPolicy",
    "ExpansionStepExecution",
    "ExpansionStepRunner",
    "ExpansionVerificationPolicy",
    "FilesystemExpansionCheckpointStore",
    "InMemoryExpansionCheckpointStore",
    "RepairingOnArtifactRunner",
    "ScriptedExpansionStepRunner",
    "both_context_failure_outcomes",
    "compression_blame_reason_codes",
    "default_model_policy",
    "default_verification_policy",
    "execute_expansion_loop",
    "execute_expansion_loop_interface_id",
    "omission_supporting_outcomes",
]
