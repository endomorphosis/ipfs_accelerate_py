"""Execute sealed selection projections and compare selected-versus-full results.

Interface: ``SemanticVerification@1``

This module runs static checks, pytest commands, and optional proofs projected
by ``selection_execution`` through the existing ``ValidationScheduler`` /
``ProofScheduler`` surfaces.  It normalizes run facts for the datasets
``compare_test_selection_oracle`` consumer and defines controlled-fixture false
negatives exactly as:

    full-suite failure attributable to the mutation that was absent from the
    selected run (authored-oracle miss or new regression not covered by
    effective selection membership).

Timeouts and cancellation are typed.  Unavailable provers are never reported as
passed proofs.  Simulation never becomes production acceptance evidence.
Cold import is side-effect free.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HarnessError,
    TestSelectionRef,
    UnavailableResult,
    VerificationReceipt,
    validate_opaque_cid,
    _bool,
    _closed,
    _nonneg_int,
    _optional_cid,
    _text,
    _unique_sorted_cids,
    _unique_sorted_texts,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.selection_execution import (
    ADAPTER_ID as SELECTION_ADAPTER_ID,
    CommandBinding,
    CommandKind,
    FALLBACK_BOTH,
    FALLBACK_FULL_PYTEST,
    HarnessAssurancePolicy,
    MaterializedCommand,
    MaterializedSelectionPlan,
    SelectionCancelled,
    SelectionExecutionAdapter,
    SelectionTimeout,
    TypedTimeout,
    effective_fallback_for,
    producer_fallback_of,
    selected_pytest_node_ids_of,
)

# ---------------------------------------------------------------------------
# Interface pins
# ---------------------------------------------------------------------------

SEMANTIC_VERIFICATION_INTERFACE: Final[str] = "SemanticVerification@1"
VERIFICATION_SCHEMA: Final[str] = "semantic-state-verification@1"
ADAPTER_ID: Final[str] = "semantic-verification-runner"
_BASIS_POINTS: Final[int] = 10_000
_MAX_DIAGNOSTIC: Final[int] = 512
_REGRESSION_STATUSES: Final[frozenset[str]] = frozenset(
    {"failed", "error", "timeout"}
)
_FULL_PYTEST_FALLBACKS: Final[frozenset[str]] = frozenset(
    {FALLBACK_FULL_PYTEST, FALLBACK_BOTH}
)


class VerificationError(HarnessError):
    """Verification projection or comparison failed closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "verification_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "verification_error")


class VerificationTimeout(VerificationError):
    """Typed timeout for a verification stage or command."""

    def __init__(
        self,
        message: str,
        *,
        timeout: TypedTimeout,
        command_identity: str = "",
    ) -> None:
        super().__init__(message, reason_code="verification_timeout")
        self.timeout = timeout
        self.command_identity = str(command_identity or "")


class VerificationCancelled(VerificationError):
    """Typed cancellation for a verification stage or command."""

    def __init__(
        self,
        message: str,
        *,
        cancellation_id: str,
        reason: str = "cancelled",
        command_identity: str = "",
    ) -> None:
        super().__init__(message, reason_code="verification_cancelled")
        self.cancellation_id = str(cancellation_id or "")
        self.cancel_reason = str(reason or "cancelled")
        self.command_identity = str(command_identity or "")


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class NormalizedTestStatus(str, Enum):
    """Mirror of datasets normalized statuses for hermetic fixtures."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    XPASSED = "xpassed"
    ERROR = "error"
    TIMEOUT = "timeout"


class OracleApplicability(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(text: str) -> str:
    value = str(text or "").strip() or "unspecified"
    if len(value) > _MAX_DIAGNOSTIC:
        return value[: _MAX_DIAGNOSTIC - 3] + "..."
    return value


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _ratio_bp(numerator: int, denominator: int) -> int | None:
    if denominator <= 0:
        return None
    if numerator < 0:
        raise VerificationError("ratio numerator must be nonnegative")
    value = (numerator * _BASIS_POINTS) // denominator
    if value > _BASIS_POINTS:
        return _BASIS_POINTS
    return value


def _status_value(status: Any) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    return str(status)


def _raise_if_cancelled(
    cancellation: CancellationToken | None,
    *,
    command_identity: str = "",
) -> None:
    if cancellation is None:
        return
    if cancellation.is_cancelled():
        raise VerificationCancelled(
            f"verification cancelled: {cancellation.reason or 'cancelled'}",
            cancellation_id=cancellation.cancellation_id,
            reason=cancellation.reason or "cancelled",
            command_identity=command_identity,
        )


def _optional_timeout(value: Any, name: str) -> TypedTimeout | None:
    if value is None:
        return None
    if isinstance(value, TypedTimeout):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return TypedTimeout(seconds=float(value), stage=name)
    if isinstance(value, Mapping):
        return TypedTimeout.from_dict(value)
    raise VerificationError(f"{name} must be TypedTimeout, number, or object")


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticCheckResult:
    """Outcome of one deterministic static/lint/typecheck command."""

    command_identity: str
    shell_command: str
    status: str
    exit_code: int
    timed_out: bool
    cancelled: bool
    binding: CommandBinding
    selection_cid: str
    output_artifact_cids: tuple[str, ...] = ()
    diagnostic: str = ""
    timeout: TypedTimeout | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_identity",
            _text(self.command_identity, "command_identity"),
        )
        object.__setattr__(
            self, "shell_command", _text(self.shell_command, "shell_command")
        )
        try:
            status = VerificationStatus(str(self.status))
        except ValueError as exc:
            raise VerificationError(f"unsupported status {self.status!r}") from exc
        object.__setattr__(self, "status", status.value)
        if type(self.exit_code) is not int or isinstance(self.exit_code, bool):
            raise VerificationError("exit_code must be an integer")
        for name in ("timed_out", "cancelled"):
            if type(getattr(self, name)) is not bool:
                raise VerificationError(f"{name} must be a bool")
        if not isinstance(self.binding, CommandBinding):
            raise VerificationError("binding must be CommandBinding")
        object.__setattr__(
            self,
            "selection_cid",
            validate_opaque_cid(self.selection_cid, "selection_cid"),
        )
        object.__setattr__(
            self,
            "output_artifact_cids",
            _unique_sorted_cids(
                list(self.output_artifact_cids), "output_artifact_cids"
            ),
        )
        object.__setattr__(self, "diagnostic", _clip(self.diagnostic or ""))

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED.value and not (
            self.timed_out or self.cancelled
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_identity": self.command_identity,
            "shell_command": self.shell_command,
            "status": self.status,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "passed": self.passed,
            "binding": self.binding.to_dict(),
            "selection_cid": self.selection_cid,
            "output_artifact_cids": list(self.output_artifact_cids),
            "diagnostic": self.diagnostic,
            "timeout": None if self.timeout is None else self.timeout.to_dict(),
        }


@dataclass(frozen=True)
class PytestResult:
    """Outcome of one selected-node or full-suite pytest command."""

    command_identity: str
    shell_command: str
    status: str
    exit_code: int
    timed_out: bool
    cancelled: bool
    node_ids: tuple[str, ...]
    binding: CommandBinding
    selection_cid: str
    kind: str = CommandKind.PYTEST_NODE.value
    output_artifact_cids: tuple[str, ...] = ()
    diagnostic: str = ""
    timeout: TypedTimeout | None = None
    outcomes: tuple["NormalizedOutcome", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_identity",
            _text(self.command_identity, "command_identity"),
        )
        object.__setattr__(
            self, "shell_command", _text(self.shell_command, "shell_command")
        )
        try:
            status = VerificationStatus(str(self.status))
        except ValueError as exc:
            raise VerificationError(f"unsupported status {self.status!r}") from exc
        object.__setattr__(self, "status", status.value)
        if type(self.exit_code) is not int or isinstance(self.exit_code, bool):
            raise VerificationError("exit_code must be an integer")
        for name in ("timed_out", "cancelled"):
            if type(getattr(self, name)) is not bool:
                raise VerificationError(f"{name} must be a bool")
        object.__setattr__(
            self, "node_ids", _unique_sorted_texts(list(self.node_ids), "node_ids")
        )
        if not isinstance(self.binding, CommandBinding):
            raise VerificationError("binding must be CommandBinding")
        object.__setattr__(
            self,
            "selection_cid",
            validate_opaque_cid(self.selection_cid, "selection_cid"),
        )
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(
            self,
            "output_artifact_cids",
            _unique_sorted_cids(
                list(self.output_artifact_cids), "output_artifact_cids"
            ),
        )
        object.__setattr__(self, "diagnostic", _clip(self.diagnostic or ""))
        if any(not isinstance(item, NormalizedOutcome) for item in self.outcomes):
            raise VerificationError("outcomes must be NormalizedOutcome values")

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED.value and not (
            self.timed_out or self.cancelled
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_identity": self.command_identity,
            "shell_command": self.shell_command,
            "status": self.status,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "passed": self.passed,
            "node_ids": list(self.node_ids),
            "binding": self.binding.to_dict(),
            "selection_cid": self.selection_cid,
            "kind": self.kind,
            "output_artifact_cids": list(self.output_artifact_cids),
            "diagnostic": self.diagnostic,
            "timeout": None if self.timeout is None else self.timeout.to_dict(),
            "outcomes": [item.to_dict() for item in self.outcomes],
        }


@dataclass(frozen=True)
class ProverResult:
    """Outcome of one proof obligation.

    ``unavailable`` is a first-class status and is never coerced to passed.
    """

    command_identity: str
    proof_id: str
    status: str
    binding: CommandBinding
    selection_cid: str
    timed_out: bool = False
    cancelled: bool = False
    output_artifact_cids: tuple[str, ...] = ()
    diagnostic: str = ""
    timeout: TypedTimeout | None = None
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_identity",
            _text(self.command_identity, "command_identity"),
        )
        object.__setattr__(self, "proof_id", _text(self.proof_id, "proof_id"))
        try:
            status = VerificationStatus(str(self.status))
        except ValueError as exc:
            raise VerificationError(f"unsupported status {self.status!r}") from exc
        object.__setattr__(self, "status", status.value)
        for name in ("timed_out", "cancelled"):
            if type(getattr(self, name)) is not bool:
                raise VerificationError(f"{name} must be a bool")
        if not isinstance(self.binding, CommandBinding):
            raise VerificationError("binding must be CommandBinding")
        object.__setattr__(
            self,
            "selection_cid",
            validate_opaque_cid(self.selection_cid, "selection_cid"),
        )
        object.__setattr__(
            self,
            "output_artifact_cids",
            _unique_sorted_cids(
                list(self.output_artifact_cids), "output_artifact_cids"
            ),
        )
        object.__setattr__(self, "diagnostic", _clip(self.diagnostic or ""))
        object.__setattr__(self, "reason_code", str(self.reason_code or ""))
        # Hard invariant: unavailable never reports as passed.
        if self.status == VerificationStatus.UNAVAILABLE.value:
            pass

    @property
    def passed(self) -> bool:
        if self.status == VerificationStatus.UNAVAILABLE.value:
            return False
        return self.status == VerificationStatus.PASSED.value and not (
            self.timed_out or self.cancelled
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_identity": self.command_identity,
            "proof_id": self.proof_id,
            "status": self.status,
            "passed": self.passed,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "binding": self.binding.to_dict(),
            "selection_cid": self.selection_cid,
            "output_artifact_cids": list(self.output_artifact_cids),
            "diagnostic": self.diagnostic,
            "timeout": None if self.timeout is None else self.timeout.to_dict(),
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class NormalizedOutcome:
    """Node-ID-keyed normalized test outcome (no wall-clock timestamps)."""

    node_id: str
    status: str
    failure_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        try:
            status = NormalizedTestStatus(str(self.status))
        except ValueError as exc:
            raise VerificationError(f"unsupported test status {self.status!r}") from exc
        object.__setattr__(self, "status", status.value)
        if self.failure_fingerprint is not None:
            object.__setattr__(
                self,
                "failure_fingerprint",
                _text(self.failure_fingerprint, "failure_fingerprint"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "failure_fingerprint": self.failure_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizedOutcome":
        payload = _closed(
            data,
            frozenset({"node_id", "status", "failure_fingerprint"}),
            "NormalizedOutcome",
        )
        return cls(
            node_id=payload["node_id"],
            status=payload["status"],
            failure_fingerprint=payload["failure_fingerprint"],
        )


@dataclass(frozen=True)
class NormalizedRunFacts:
    """Normalized run facts supplied to the producer oracle comparison."""

    run_id: str
    outcomes: tuple[NormalizedOutcome, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        if any(not isinstance(item, NormalizedOutcome) for item in self.outcomes):
            raise VerificationError("outcomes must be NormalizedOutcome values")
        # Unique node IDs, sorted for determinism.
        by_id = {item.node_id: item for item in self.outcomes}
        if len(by_id) != len(self.outcomes):
            raise VerificationError("outcomes must not contain duplicate node_ids")
        object.__setattr__(
            self,
            "outcomes",
            tuple(by_id[key] for key in sorted(by_id)),
        )

    @property
    def facts_cid(self) -> str:
        return _digest(
            "sch-facts",
            {
                "schema": VERIFICATION_SCHEMA,
                "run_id": self.run_id,
                "outcomes": [item.to_dict() for item in self.outcomes],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "facts_cid": self.facts_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizedRunFacts":
        payload = _closed(
            data,
            frozenset({"run_id", "outcomes", "facts_cid"}),
            "NormalizedRunFacts",
        )
        outcomes = tuple(
            NormalizedOutcome.from_dict(item) for item in payload["outcomes"]
        )
        result = cls(run_id=payload["run_id"], outcomes=outcomes)
        claimed = payload.get("facts_cid")
        if claimed is not None and claimed != result.facts_cid:
            raise VerificationError("facts_cid does not verify")
        return result

    def node_ids(self) -> frozenset[str]:
        return frozenset(item.node_id for item in self.outcomes)

    def outcome_map(self) -> dict[str, NormalizedOutcome]:
        return {item.node_id: item for item in self.outcomes}


def normalize_run_facts(
    run_id: str,
    outcomes: Iterable[Mapping[str, Any] | NormalizedOutcome],
) -> NormalizedRunFacts:
    """Build closed ``NormalizedRunFacts`` from mappings or outcome records."""

    items: list[NormalizedOutcome] = []
    for item in outcomes:
        if isinstance(item, NormalizedOutcome):
            items.append(item)
        elif isinstance(item, Mapping):
            items.append(NormalizedOutcome.from_dict(item))
        else:
            raise VerificationError("outcomes entries must be mappings or NormalizedOutcome")
    return NormalizedRunFacts(run_id=run_id, outcomes=tuple(items))


# ---------------------------------------------------------------------------
# Full-suite comparison / controlled false-negative definition
# ---------------------------------------------------------------------------


def _is_regression_status(status: Any) -> bool:
    return _status_value(status) in _REGRESSION_STATUSES


def _failure_identity(outcome: NormalizedOutcome) -> tuple[str, str | None]:
    return (_status_value(outcome.status), outcome.failure_fingerprint)


def compute_new_regressions(
    baseline_full: NormalizedRunFacts,
    candidate_full: NormalizedRunFacts,
) -> tuple[str, ...]:
    """Node IDs with candidate fail/error/timeout not identical in baseline.

    A baseline fail/error/timeout with the same status and failure fingerprint
    is a known failure and is not attributed to the candidate mutation.
    """

    baseline = baseline_full.outcome_map()
    found: list[str] = []
    for outcome in candidate_full.outcomes:
        if not _is_regression_status(outcome.status):
            continue
        previous = baseline.get(outcome.node_id)
        if previous is not None and _is_regression_status(previous.status):
            if _failure_identity(previous) == _failure_identity(outcome):
                continue
        found.append(outcome.node_id)
    return tuple(sorted(found))


def compute_changed_outcome_node_ids(
    baseline_full: NormalizedRunFacts,
    candidate_full: NormalizedRunFacts,
) -> tuple[str, ...]:
    baseline = baseline_full.outcome_map()
    candidate = candidate_full.outcome_map()
    changed: list[str] = []
    for node_id in sorted(set(baseline) | set(candidate)):
        left = baseline.get(node_id)
        right = candidate.get(node_id)
        if left is None or right is None:
            changed.append(node_id)
            continue
        if _status_value(left.status) != _status_value(right.status):
            changed.append(node_id)
            continue
        if left.failure_fingerprint != right.failure_fingerprint:
            changed.append(node_id)
    return tuple(changed)


def _fallback_value(selection: Any) -> str:
    return producer_fallback_of(selection)


def _effective_selected_node_ids(
    selection: Any,
    candidate_full: NormalizedRunFacts,
) -> frozenset[str]:
    """Node IDs treated as selected for membership metrics.

    Domain-wide pytest fallback signals that accelerate runs the full suite.
    Membership then covers every candidate-full node so full fallback is not
    misreported as total miss.  Unaffected passing tests are never redefined
    as true positives against an authored oracle.
    """

    fallback = _fallback_value(selection)
    if fallback in _FULL_PYTEST_FALLBACKS:
        return frozenset(outcome.node_id for outcome in candidate_full.outcomes)
    return frozenset(selected_pytest_node_ids_of(selection))


def _normalize_authored_oracle(
    authored_oracle: Sequence[str] | None,
) -> tuple[str, ...]:
    if authored_oracle is None:
        return ()
    if isinstance(authored_oracle, (str, bytes)):
        raise VerificationError("authored_oracle must be a sequence of node IDs")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in authored_oracle:
        if type(item) is not str or not item or item != item.strip():
            raise VerificationError(
                "authored_oracle entries must be nonempty trimmed node ID strings"
            )
        if item in seen:
            raise VerificationError(
                f"authored_oracle must not contain duplicate node IDs: {item!r}"
            )
        seen.add(item)
        ordered.append(item)
    return tuple(sorted(ordered))


@dataclass(frozen=True)
class FullSuiteComparison:
    """Controlled selected-versus-full comparison metrics.

    False negatives are authored-oracle nodes absent from effective selection
    membership.  Missed regressions are new full-suite failures absent from
    that membership.  Empty authored oracles are ``not_applicable`` and never
    fabricate 100 percent recall.
    """

    selection_cid: str
    baseline_facts_cid: str
    selected_facts_cid: str
    candidate_full_facts_cid: str
    applicability: str
    new_regressions: tuple[str, ...]
    missed_regressions: tuple[str, ...]
    true_positives: tuple[str, ...]
    false_negatives: tuple[str, ...]
    false_positives: tuple[str, ...]
    fixture_recall_bp: int | None
    fixture_precision_bp: int | None
    selected_count: int
    full_count: int
    selection_ratio_bp: int | None
    execution_reduction_bp: int | None
    fallback_rate_bp: int | None
    changed_outcome_node_ids: tuple[str, ...]
    regression_recall_bp: int | None
    producer_fallback: str
    effective_fallback: str

    def __post_init__(self) -> None:
        for name in (
            "selection_cid",
            "baseline_facts_cid",
            "selected_facts_cid",
            "candidate_full_facts_cid",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        try:
            applicability = OracleApplicability(str(self.applicability))
        except ValueError as exc:
            raise VerificationError(
                f"unsupported applicability {self.applicability!r}"
            ) from exc
        object.__setattr__(self, "applicability", applicability.value)
        for name in (
            "new_regressions",
            "missed_regressions",
            "true_positives",
            "false_negatives",
            "false_positives",
            "changed_outcome_node_ids",
        ):
            object.__setattr__(
                self,
                name,
                _unique_sorted_texts(list(getattr(self, name)), name),
            )
        for name in ("selected_count", "full_count"):
            object.__setattr__(
                self, name, _nonneg_int(getattr(self, name), name)
            )
        object.__setattr__(
            self, "producer_fallback", _text(self.producer_fallback, "producer_fallback")
        )
        object.__setattr__(
            self,
            "effective_fallback",
            _text(self.effective_fallback, "effective_fallback"),
        )

    @property
    def zero_false_negatives(self) -> bool:
        return len(self.false_negatives) == 0 and len(self.missed_regressions) == 0

    @property
    def supports_100_percent_recall(self) -> bool:
        """True when applicability admits a recall score and recall is 100%."""

        if self.applicability != OracleApplicability.APPLICABLE.value:
            return False
        if self.fixture_recall_bp is None:
            return False
        return self.fixture_recall_bp == _BASIS_POINTS and self.zero_false_negatives

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_cid": self.selection_cid,
            "baseline_facts_cid": self.baseline_facts_cid,
            "selected_facts_cid": self.selected_facts_cid,
            "candidate_full_facts_cid": self.candidate_full_facts_cid,
            "applicability": self.applicability,
            "new_regressions": list(self.new_regressions),
            "missed_regressions": list(self.missed_regressions),
            "true_positives": list(self.true_positives),
            "false_negatives": list(self.false_negatives),
            "false_positives": list(self.false_positives),
            "fixture_recall_bp": self.fixture_recall_bp,
            "fixture_precision_bp": self.fixture_precision_bp,
            "selected_count": self.selected_count,
            "full_count": self.full_count,
            "selection_ratio_bp": self.selection_ratio_bp,
            "execution_reduction_bp": self.execution_reduction_bp,
            "fallback_rate_bp": self.fallback_rate_bp,
            "changed_outcome_node_ids": list(self.changed_outcome_node_ids),
            "regression_recall_bp": self.regression_recall_bp,
            "producer_fallback": self.producer_fallback,
            "effective_fallback": self.effective_fallback,
            "zero_false_negatives": self.zero_false_negatives,
            "supports_100_percent_recall": self.supports_100_percent_recall,
            "schema": VERIFICATION_SCHEMA,
            "interface": SEMANTIC_VERIFICATION_INTERFACE,
        }


def compare_full_suite(
    selection: Any,
    *,
    baseline_full: NormalizedRunFacts,
    selected_run: NormalizedRunFacts,
    candidate_full: NormalizedRunFacts,
    authored_oracle: Sequence[str] | None = None,
    oracle_fn: Callable[..., Any] | None = None,
) -> FullSuiteComparison:
    """Compare selected-versus-full results with controlled FN definition.

    When ``oracle_fn`` is supplied (typically datasets
    ``compare_test_selection_oracle`` via the adapter), its metrics are
    preferred for membership/TP/FN/FP.  Local computation always remains
    available for hermetic fixtures and matches the sealed producer semantics.
    """

    if not isinstance(baseline_full, NormalizedRunFacts):
        raise VerificationError("baseline_full must be NormalizedRunFacts")
    if not isinstance(selected_run, NormalizedRunFacts):
        raise VerificationError("selected_run must be NormalizedRunFacts")
    if not isinstance(candidate_full, NormalizedRunFacts):
        raise VerificationError("candidate_full must be NormalizedRunFacts")

    selection_cid = str(getattr(selection, "selection_cid", "") or "")
    if not selection_cid:
        raise VerificationError("selection missing selection_cid")
    producer_fb = _fallback_value(selection)
    effective_fb = effective_fallback_for(selection)

    # Optional datasets oracle path (pure; no execution).
    if oracle_fn is not None:
        try:
            remote = oracle_fn(
                selection,
                baseline_full=baseline_full,
                selected_run=selected_run,
                candidate_full=candidate_full,
                authored_oracle=authored_oracle,
            )
        except TypeError:
            # Datasets models expect their own TestRunFacts types; fall back.
            remote = None
        except Exception:
            remote = None
        if remote is not None:
            return _comparison_from_remote(
                remote,
                selection_cid=selection_cid,
                producer_fallback=producer_fb,
                effective_fallback=effective_fb,
                baseline_full=baseline_full,
                selected_run=selected_run,
                candidate_full=candidate_full,
            )

    oracle_nodes = _normalize_authored_oracle(authored_oracle)
    effective_selected = _effective_selected_node_ids(selection, candidate_full)
    declared_selected = frozenset(selected_pytest_node_ids_of(selection))

    new_regressions = compute_new_regressions(baseline_full, candidate_full)
    new_regression_set = frozenset(new_regressions)
    missed_regressions = tuple(
        sorted(
            node_id
            for node_id in new_regressions
            if node_id not in effective_selected
        )
    )

    if oracle_nodes:
        applicability = OracleApplicability.APPLICABLE
        oracle_set = frozenset(oracle_nodes)
        true_positives = tuple(sorted(effective_selected & oracle_set))
        false_negatives = tuple(sorted(oracle_set - effective_selected))
        false_positives = tuple(sorted(effective_selected - oracle_set))
        fixture_recall_bp = _ratio_bp(len(true_positives), len(oracle_set))
        fixture_precision_bp = _ratio_bp(len(true_positives), len(effective_selected))
    else:
        applicability = OracleApplicability.NOT_APPLICABLE
        true_positives = ()
        false_negatives = ()
        false_positives = ()
        fixture_recall_bp = None
        fixture_precision_bp = None

    full_count = len(candidate_full.outcomes)
    full_pytest_fallback = producer_fb in _FULL_PYTEST_FALLBACKS or (
        effective_fb in _FULL_PYTEST_FALLBACKS
    )

    if full_pytest_fallback:
        selected_count = full_count
        selection_ratio_bp = _ratio_bp(full_count, full_count) if full_count else None
        execution_reduction_bp = _ratio_bp(0, full_count) if full_count else None
        fallback_rate_bp = _BASIS_POINTS
    else:
        selected_count = len(declared_selected)
        selection_ratio_bp = _ratio_bp(selected_count, full_count)
        if full_count:
            reduction = full_count - selected_count
            if reduction < 0:
                reduction = 0
            execution_reduction_bp = _ratio_bp(reduction, full_count)
        else:
            execution_reduction_bp = None
        fallback_rate_bp = 0

    changed_outcome_node_ids = compute_changed_outcome_node_ids(
        baseline_full, candidate_full
    )
    caught = len(new_regression_set & effective_selected)
    regression_recall_bp = _ratio_bp(caught, len(new_regressions))

    return FullSuiteComparison(
        selection_cid=selection_cid,
        baseline_facts_cid=baseline_full.facts_cid,
        selected_facts_cid=selected_run.facts_cid,
        candidate_full_facts_cid=candidate_full.facts_cid,
        applicability=applicability.value,
        new_regressions=new_regressions,
        missed_regressions=missed_regressions,
        true_positives=true_positives,
        false_negatives=false_negatives,
        false_positives=false_positives,
        fixture_recall_bp=fixture_recall_bp,
        fixture_precision_bp=fixture_precision_bp,
        selected_count=selected_count,
        full_count=full_count,
        selection_ratio_bp=selection_ratio_bp,
        execution_reduction_bp=execution_reduction_bp,
        fallback_rate_bp=fallback_rate_bp,
        changed_outcome_node_ids=changed_outcome_node_ids,
        regression_recall_bp=regression_recall_bp,
        producer_fallback=producer_fb,
        effective_fallback=effective_fb,
    )


def _comparison_from_remote(
    remote: Any,
    *,
    selection_cid: str,
    producer_fallback: str,
    effective_fallback: str,
    baseline_full: NormalizedRunFacts,
    selected_run: NormalizedRunFacts,
    candidate_full: NormalizedRunFacts,
) -> FullSuiteComparison:
    """Project a datasets ``TestOracleComparison`` into ``FullSuiteComparison``."""

    def _seq(name: str) -> tuple[str, ...]:
        value = getattr(remote, name, None)
        if value is None and isinstance(remote, Mapping):
            value = remote.get(name)
        if value is None:
            return ()
        return tuple(str(item) for item in value)

    def _opt_int(name: str) -> int | None:
        value = getattr(remote, name, None)
        if value is None and isinstance(remote, Mapping):
            value = remote.get(name)
        if value is None:
            return None
        return int(value)

    def _int(name: str, default: int = 0) -> int:
        value = _opt_int(name)
        return default if value is None else value

    applicability = getattr(remote, "applicability", None)
    if applicability is None and isinstance(remote, Mapping):
        applicability = remote.get("applicability")
    applicability_value = _status_value(applicability or OracleApplicability.NOT_APPLICABLE)

    return FullSuiteComparison(
        selection_cid=str(
            getattr(remote, "selection_cid", None)
            or (remote.get("selection_cid") if isinstance(remote, Mapping) else None)
            or selection_cid
        ),
        baseline_facts_cid=str(
            getattr(remote, "baseline_facts_cid", None)
            or (
                remote.get("baseline_facts_cid")
                if isinstance(remote, Mapping)
                else None
            )
            or baseline_full.facts_cid
        ),
        selected_facts_cid=str(
            getattr(remote, "selected_facts_cid", None)
            or (
                remote.get("selected_facts_cid")
                if isinstance(remote, Mapping)
                else None
            )
            or selected_run.facts_cid
        ),
        candidate_full_facts_cid=str(
            getattr(remote, "candidate_full_facts_cid", None)
            or (
                remote.get("candidate_full_facts_cid")
                if isinstance(remote, Mapping)
                else None
            )
            or candidate_full.facts_cid
        ),
        applicability=applicability_value,
        new_regressions=_seq("new_regressions"),
        missed_regressions=_seq("missed_regressions"),
        true_positives=_seq("true_positives"),
        false_negatives=_seq("false_negatives"),
        false_positives=_seq("false_positives"),
        fixture_recall_bp=_opt_int("fixture_recall_bp"),
        fixture_precision_bp=_opt_int("fixture_precision_bp"),
        selected_count=_int("selected_count"),
        full_count=_int("full_count"),
        selection_ratio_bp=_opt_int("selection_ratio_bp"),
        execution_reduction_bp=_opt_int("execution_reduction_bp"),
        fallback_rate_bp=_opt_int("fallback_rate_bp"),
        changed_outcome_node_ids=_seq("changed_outcome_node_ids"),
        regression_recall_bp=_opt_int("regression_recall_bp"),
        producer_fallback=producer_fallback,
        effective_fallback=effective_fallback,
    )


# ---------------------------------------------------------------------------
# Verification runner
# ---------------------------------------------------------------------------


def _status_from_report(report: Mapping[str, Any]) -> tuple[str, int, bool, bool]:
    timed_out = bool(report.get("timed_out") or report.get("timeout"))
    cancelled = bool(report.get("cancelled"))
    exit_code = int(report.get("returncode", report.get("exit_code", 1)) or 0)
    if cancelled:
        return VerificationStatus.CANCELLED.value, exit_code or 130, timed_out, True
    if timed_out:
        return VerificationStatus.TIMEOUT.value, exit_code or 124, True, False
    if report.get("unavailable"):
        return VerificationStatus.UNAVAILABLE.value, exit_code or 75, False, False
    if bool(report.get("passed")) or exit_code == 0:
        return VerificationStatus.PASSED.value, exit_code, False, False
    return VerificationStatus.FAILED.value, exit_code or 1, False, False


@dataclass
class VerificationRunner:
    """Run static/pytest/proof stages for a sealed selection projection.

    Reuses ``SelectionExecutionAdapter`` for command materialization and
    scheduler projection.  Does not reselect tests or traverse graphs.
    """

    selection_adapter: SelectionExecutionAdapter = field(
        default_factory=SelectionExecutionAdapter
    )
    assurance: HarnessAssurancePolicy = field(default_factory=HarnessAssurancePolicy)
    oracle_fn: Callable[..., Any] | None = None
    adapter_id: str = ADAPTER_ID

    def materialize(
        self,
        selection: Any,
        *,
        binding: CommandBinding,
        selection_ref: TestSelectionRef | None = None,
        assurance: HarnessAssurancePolicy | None = None,
    ) -> MaterializedSelectionPlan:
        return self.selection_adapter.materialize(
            selection,
            binding=binding,
            selection_ref=selection_ref,
            assurance=assurance or self.assurance,
        )

    def run_static_checks(
        self,
        plan: MaterializedSelectionPlan,
        *,
        workspace_path: Path | str,
        cancellation: CancellationToken | None = None,
        runner: Callable[[MaterializedCommand], Mapping[str, Any]] | None = None,
    ) -> tuple[StaticCheckResult, ...]:
        _raise_if_cancelled(cancellation)
        results: list[StaticCheckResult] = []
        for command in plan.commands:
            if command.kind != CommandKind.STATIC_CHECK.value:
                continue
            _raise_if_cancelled(
                cancellation, command_identity=command.command_identity
            )
            report = self._run_one(command, workspace_path=workspace_path, runner=runner)
            status, exit_code, timed_out, cancelled = _status_from_report(report)
            if timed_out:
                # Typed timeout surface for callers that prefer exceptions.
                pass
            results.append(
                StaticCheckResult(
                    command_identity=command.command_identity,
                    shell_command=command.shell_command or "",
                    status=status,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    cancelled=cancelled,
                    binding=plan.binding,
                    selection_cid=plan.selection_ref.selection_cid,
                    output_artifact_cids=tuple(
                        str(item)
                        for item in report.get("output_artifact_cids", ()) or ()
                    ),
                    diagnostic=_clip(str(report.get("error") or report.get("diagnostic") or "")),
                    timeout=command.timeout,
                )
            )
        return tuple(results)

    def run_pytest(
        self,
        plan: MaterializedSelectionPlan,
        *,
        workspace_path: Path | str,
        cancellation: CancellationToken | None = None,
        runner: Callable[[MaterializedCommand], Mapping[str, Any]] | None = None,
        outcomes_by_command: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> tuple[PytestResult, ...]:
        _raise_if_cancelled(cancellation)
        results: list[PytestResult] = []
        for command in plan.commands:
            if command.kind not in {
                CommandKind.PYTEST_NODE.value,
                CommandKind.FULL_PYTEST.value,
            }:
                continue
            _raise_if_cancelled(
                cancellation, command_identity=command.command_identity
            )
            report = self._run_one(command, workspace_path=workspace_path, runner=runner)
            status, exit_code, timed_out, cancelled = _status_from_report(report)
            raw_outcomes = ()
            if outcomes_by_command is not None:
                raw_outcomes = outcomes_by_command.get(command.command_identity, ())
            elif "outcomes" in report:
                raw_outcomes = report.get("outcomes") or ()
            outcomes = tuple(
                item
                if isinstance(item, NormalizedOutcome)
                else NormalizedOutcome.from_dict(item)
                for item in raw_outcomes
            )
            results.append(
                PytestResult(
                    command_identity=command.command_identity,
                    shell_command=command.shell_command or "",
                    status=status,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    cancelled=cancelled,
                    node_ids=command.target_ids,
                    binding=plan.binding,
                    selection_cid=plan.selection_ref.selection_cid,
                    kind=command.kind,
                    output_artifact_cids=tuple(
                        str(item)
                        for item in report.get("output_artifact_cids", ()) or ()
                    ),
                    diagnostic=_clip(
                        str(report.get("error") or report.get("diagnostic") or "")
                    ),
                    timeout=command.timeout,
                    outcomes=outcomes,
                )
            )
        return tuple(results)

    def run_proofs(
        self,
        plan: MaterializedSelectionPlan,
        *,
        cancellation: CancellationToken | None = None,
        prover_available: bool | Callable[[str], bool] | None = None,
        proof_executor: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> tuple[ProverResult, ...]:
        raw = self.selection_adapter.run_proofs(
            plan,
            cancellation=cancellation,
            prover_available=prover_available,
            proof_executor=proof_executor,
        )
        results: list[ProverResult] = []
        for item in raw:
            status = str(item.get("status") or VerificationStatus.FAILED.value)
            if item.get("unavailable") or status == "unavailable":
                status = VerificationStatus.UNAVAILABLE.value
            if item.get("timed_out") or status == "timeout":
                status = VerificationStatus.TIMEOUT.value
            if item.get("cancelled") or status == "cancelled":
                status = VerificationStatus.CANCELLED.value
            # Never allow unavailable to flip to passed.
            if status == VerificationStatus.UNAVAILABLE.value:
                passed_flag = False
            else:
                passed_flag = bool(item.get("passed")) and status == (
                    VerificationStatus.PASSED.value
                )
                if passed_flag:
                    status = VerificationStatus.PASSED.value
                elif status == VerificationStatus.PASSED.value:
                    status = VerificationStatus.FAILED.value
            results.append(
                ProverResult(
                    command_identity=str(item.get("command_identity") or "proof"),
                    proof_id=str(item.get("proof_id") or ""),
                    status=status,
                    binding=plan.binding,
                    selection_cid=plan.selection_ref.selection_cid,
                    timed_out=bool(item.get("timed_out")),
                    cancelled=bool(item.get("cancelled")),
                    output_artifact_cids=tuple(
                        str(cid)
                        for cid in item.get("output_artifact_cids", ()) or ()
                    ),
                    diagnostic=_clip(
                        str(
                            item.get("diagnostic")
                            or item.get("error")
                            or item.get("reason_code")
                            or ""
                        )
                    ),
                    timeout=(
                        TypedTimeout.from_dict(item["timeout"])
                        if isinstance(item.get("timeout"), Mapping)
                        else None
                    ),
                    reason_code=str(item.get("reason_code") or ""),
                )
            )
            # Keep passed property consistent with status.
            if status == VerificationStatus.UNAVAILABLE.value and results[-1].passed:
                raise VerificationError(
                    "unavailable prover reported as passed",
                    reason_code="unavailable_as_passed",
                )
        return tuple(results)

    def _run_one(
        self,
        command: MaterializedCommand,
        *,
        workspace_path: Path | str,
        runner: Callable[[MaterializedCommand], Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        if runner is not None:
            started = time.monotonic()
            try:
                report = runner(command)
            except SelectionTimeout as exc:
                raise VerificationTimeout(
                    str(exc),
                    timeout=exc.timeout,
                    command_identity=exc.command_identity or command.command_identity,
                ) from exc
            except SelectionCancelled as exc:
                raise VerificationCancelled(
                    str(exc),
                    cancellation_id=exc.cancellation_id,
                    reason=exc.cancel_reason,
                    command_identity=exc.command_identity or command.command_identity,
                ) from exc
            elapsed = time.monotonic() - started
            if not isinstance(report, Mapping):
                raise VerificationError("command runner must return a mapping")
            result = dict(report)
            if elapsed > command.timeout.seconds and not result.get("timed_out"):
                result["timed_out"] = True
                result["returncode"] = int(result.get("returncode") or 124)
            return result

        # Scheduler path: only validation commands.
        if command.validation_command is None:
            return {
                "passed": False,
                "returncode": 75,
                "unavailable": True,
                "reason": "no_validation_command",
            }
        if self.selection_adapter.validation_scheduler is None:
            return {
                "passed": False,
                "returncode": 75,
                "unavailable": True,
                "reason": "validation_scheduler_unavailable",
            }
        run = getattr(self.selection_adapter.validation_scheduler, "run", None)
        run_staged = getattr(
            self.selection_adapter.validation_scheduler, "run_staged", None
        )
        if callable(run):
            report = run(
                [command.validation_command],
                workspace_path=workspace_path,
                changed_files=(),
            )
        elif callable(run_staged):
            report = run_staged(
                [command.validation_command],
                workspace_path=workspace_path,
                changed_files=(),
            )
        else:
            raise VerificationError(
                "validation_scheduler must provide run or run_staged",
                reason_code="scheduler_contract",
            )
        if not isinstance(report, Mapping):
            raise VerificationError("scheduler report must be a mapping")
        return dict(report)

    def run(
        self,
        selection: Any,
        *,
        binding: CommandBinding,
        workspace_path: Path | str,
        selection_ref: TestSelectionRef | None = None,
        assurance: HarnessAssurancePolicy | None = None,
        cancellation: CancellationToken | None = None,
        runner: Callable[[MaterializedCommand], Mapping[str, Any]] | None = None,
        prover_available: bool | Callable[[str], bool] | None = None,
        proof_executor: Callable[[str], Mapping[str, Any]] | None = None,
        outcomes_by_command: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        simulated: bool = False,
    ) -> dict[str, Any]:
        """Materialize and run static, pytest, and proof stages."""

        plan = self.materialize(
            selection,
            binding=binding,
            selection_ref=selection_ref,
            assurance=assurance,
        )
        static_results = self.run_static_checks(
            plan,
            workspace_path=workspace_path,
            cancellation=cancellation,
            runner=runner,
        )
        pytest_results = self.run_pytest(
            plan,
            workspace_path=workspace_path,
            cancellation=cancellation,
            runner=runner,
            outcomes_by_command=outcomes_by_command,
        )
        proof_results = self.run_proofs(
            plan,
            cancellation=cancellation,
            prover_available=prover_available,
            proof_executor=proof_executor,
        )

        all_passed = (
            all(item.passed for item in static_results)
            and all(item.passed for item in pytest_results)
            and all(
                item.passed or item.status == VerificationStatus.UNAVAILABLE.value
                for item in proof_results
            )
            # Unavailable proofs never pass, so acceptance requires no proofs
            # in unavailable status when acceptance_eligible is requested.
        )
        proofs_all_available_and_passed = all(
            item.passed for item in proof_results
        ) if proof_results else True
        acceptance_eligible = (
            all_passed
            and proofs_all_available_and_passed
            and not simulated
            and not any(item.timed_out or item.cancelled for item in static_results)
            and not any(item.timed_out or item.cancelled for item in pytest_results)
            and not any(item.timed_out or item.cancelled for item in proof_results)
        )

        return {
            "plan": plan.to_dict(),
            "static_checks": [item.to_dict() for item in static_results],
            "pytest": [item.to_dict() for item in pytest_results],
            "proofs": [item.to_dict() for item in proof_results],
            "passed": all_passed and proofs_all_available_and_passed,
            "acceptance_eligible": acceptance_eligible,
            "simulated": bool(simulated),
            "selection_cid": plan.selection_ref.selection_cid,
            "binding": plan.binding.to_dict(),
            "producer_fallback": plan.producer_fallback,
            "effective_fallback": plan.effective_fallback,
            "fallback_reasons": list(plan.fallback_reasons),
            "reason_path_cids": list(plan.reason_path_cids),
            "interface": SEMANTIC_VERIFICATION_INTERFACE,
            "adapter_id": self.adapter_id,
        }

    def compare_full_suite(
        self,
        selection: Any,
        *,
        baseline_full: NormalizedRunFacts,
        selected_run: NormalizedRunFacts,
        candidate_full: NormalizedRunFacts,
        authored_oracle: Sequence[str] | None = None,
    ) -> FullSuiteComparison:
        return compare_full_suite(
            selection,
            baseline_full=baseline_full,
            selected_run=selected_run,
            candidate_full=candidate_full,
            authored_oracle=authored_oracle,
            oracle_fn=self.oracle_fn,
        )

    def build_verification_receipt(
        self,
        *,
        binding: CommandBinding,
        selection_ref: TestSelectionRef,
        command_identity: str,
        exit_code: int,
        output_artifact_cids: Sequence[str] = (),
        simulated: bool = False,
        fresh: bool = True,
        acceptance_eligible: bool = False,
        root_cid: str | None = None,
    ) -> VerificationReceipt:
        """Project bindings into the closed harness ``VerificationReceipt``."""

        if simulated and acceptance_eligible:
            raise VerificationError(
                "simulated verification cannot be acceptance_eligible",
                reason_code="simulated_acceptance",
            )
        return VerificationReceipt.from_dict(
            {
                "tree_cid": binding.tree_cid,
                "config_cid": binding.config_cid,
                "dependency_lock_cid": binding.dependency_lock_cid,
                "policy_cid": binding.policy_cid,
                "interface_cid": binding.interface_cid,
                "root_cid": root_cid
                or selection_ref.current_semantic_state_root_cid,
                "command_identity": command_identity,
                "selection_ref": selection_ref.to_dict(),
                "exit_code": exit_code,
                "output_artifact_cids": list(output_artifact_cids),
                "simulated": simulated,
                "fresh": fresh,
                "acceptance_eligible": acceptance_eligible and not simulated,
            }
        )


def verification_descriptor() -> dict[str, Any]:
    return {
        "interface": SEMANTIC_VERIFICATION_INTERFACE,
        "schema": VERIFICATION_SCHEMA,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "selection_adapter_id": SELECTION_ADAPTER_ID,
        "operations": (
            "run_static_checks",
            "run_pytest",
            "run_proofs",
            "compare_full_suite",
            "build_verification_receipt",
        ),
        "forbids": (
            "run_impact_selected",
            "graph_traversal",
            "reselection",
            "unavailable_prover_as_passed",
            "simulated_acceptance",
            "fabricated_100_percent_empty_oracle",
        ),
    }


__all__ = [
    "ADAPTER_ID",
    "FullSuiteComparison",
    "NormalizedOutcome",
    "NormalizedRunFacts",
    "NormalizedTestStatus",
    "OracleApplicability",
    "ProverResult",
    "PytestResult",
    "SEMANTIC_VERIFICATION_INTERFACE",
    "StaticCheckResult",
    "VERIFICATION_SCHEMA",
    "VerificationCancelled",
    "VerificationError",
    "VerificationRunner",
    "VerificationStatus",
    "VerificationTimeout",
    "compare_full_suite",
    "compute_changed_outcome_node_ids",
    "compute_new_regressions",
    "normalize_run_facts",
    "verification_descriptor",
]
