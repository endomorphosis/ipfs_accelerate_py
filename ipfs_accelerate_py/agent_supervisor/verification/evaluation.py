"""Selected-versus-full-suite semantic fixture evaluation (IVP-015).

``compare_selected_with_full_suite`` / :class:`TestSelectionEvaluation` compare
planner-selected tests against reviewed ground-truth affected sets and a
separate full-suite oracle observation.  Both observations must bind the same
fresh tree/environment/lock/fixture/policy snapshot identities.

Normative measurement rules
---------------------------
* A **false negative** is either a ground-truth affected test omitted from
  selection, or a mutation-caused full-suite failure not observed by selected
  execution.  These two oracles are recorded separately and then unioned for
  the aggregate false-negative set.
* A **false positive** is a selected test outside the ground-truth affected
  set.  A selected test that merely *passes* is not, by itself, a false
  positive.
* Flaky, order-dependent, or selected/full outcome discrepancies that are not
  clean false negatives are classified as **inconclusive**.
* A full-suite timeout/unavailable result, a missing canonical
  semantic-capsule corpus, or zero evaluated fixtures is
  ``not_measured`` — never reported as zero false negatives.
* Uncertain/uncovered selectors or a missing validation-ID-to-pytest-node-ID
  mapping require broader/full suite before acceptance.
* Equivalent controlled fixtures keep distinct labels.
* Evidence binds corpus, evaluated count, repository, policy, environment,
  and selector identities plus measured timing metadata and never asserts
  target success (``authoritative`` is always false).

This module never installs packages, widens network policy, or upgrades
timeout/unavailable/inconclusive observations into measured zero-error wins.
Importing it performs no I/O.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

from .selection import (
    AffectedVerificationSelection,
    FallbackMode,
    SelectionError,
    SelectionPolicy,
    VerificationCatalog,
    select_affected_verification,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

TEST_SELECTION_EVALUATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/test-selection-evaluation@1"
)
TEST_SELECTION_EVALUATION_INTERFACE: Final[str] = "TestSelectionEvaluation@1"
SELECTION_EVALUATION_EVIDENCE: Final[str] = "ivp/test-selection-evaluation@1"

CORPUS_EVALUATION_SUMMARY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/selection-corpus-evaluation@1"
)
CORPUS_EVALUATION_SUMMARY_INTERFACE: Final[str] = (
    "SelectionCorpusEvaluationSummary@1"
)

EVALUATION_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/evaluation-snapshot-identity@1"
)
SUITE_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/suite-observation@1"
)
CONTROLLED_FIXTURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/controlled-semantic-fixture@1"
)

DEFAULT_FIXTURE_RELPATH: Final[str] = "test/fixtures/incremental_verification"
CANONICAL_CORPUS_ID: Final[str] = "ivp-semantic-capsule-controlled-v1"
CORPUS_MANIFEST_NAME: Final[str] = "corpus_manifest.json"

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 50_000
MAX_REASON_CODES: Final[int] = 128

# Reason codes (stable tokens).
REASON_GROUND_TRUTH_OMISSION: Final[str] = "ground_truth_affected_omitted"
REASON_ORACLE_FAILURE_NOT_OBSERVED: Final[str] = (
    "full_suite_failure_not_observed_by_selected"
)
REASON_SELECTED_OUTSIDE_GROUND_TRUTH: Final[str] = (
    "selected_outside_ground_truth"
)
REASON_FLAKY_OUTCOME: Final[str] = "flaky_outcome"
REASON_ORDER_DEPENDENT: Final[str] = "order_dependent_outcome"
REASON_OUTCOME_DISCREPANCY: Final[str] = "selected_full_outcome_discrepancy"
REASON_SNAPSHOT_MISMATCH: Final[str] = "selected_full_snapshot_mismatch"
REASON_FULL_SUITE_TIMEOUT: Final[str] = "full_suite_timeout"
REASON_FULL_SUITE_UNAVAILABLE: Final[str] = "full_suite_unavailable"
REASON_CORPUS_ABSENT: Final[str] = "canonical_semantic_capsule_corpus_absent"
REASON_ZERO_EVALUATED: Final[str] = "zero_evaluated_fixtures"
REASON_UNCERTAIN_SELECTOR: Final[str] = "uncertain_or_uncovered_selector"
REASON_VALIDATION_MAPPING_MISSING: Final[str] = (
    "validation_id_to_node_id_mapping_missing"
)
REASON_BROADER_REQUIRED: Final[str] = "broader_or_full_suite_required"
REASON_PASSING_NOT_FALSE_POSITIVE: Final[str] = (
    "passing_selected_is_not_false_positive"
)
REASON_IDENTITIES_BOUND: Final[str] = "identities_and_timing_bound"
REASON_TARGET_SUCCESS_NOT_ASSERTED: Final[str] = "target_success_not_asserted"


class EvaluationError(ValueError):
    """Malformed evaluation input or contract violation."""


class EvaluationBoundsError(EvaluationError):
    """An evaluation input exceeded deterministic compactness bounds."""


class MeasurementStatus(str, Enum):
    """Whether false-negative/positive rates are measured for a case."""

    MEASURED = "measured"
    NOT_MEASURED = "not_measured"
    INCONCLUSIVE = "inconclusive"


class ObservedTestOutcome(str, Enum):
    """Closed per-test observation vocabulary for evaluation."""

    PASSED = "passed"
    FAILED = "failed"
    FLAKY = "flaky"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    SKIPPED = "skipped"
    NOT_OBSERVED = "not_observed"
    ORDER_DEPENDENT = "order_dependent"


class SuiteRunStatus(str, Enum):
    """Closed suite-level run status for selected or full observations."""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    CANCELLED = "cancelled"
    NOT_RUN = "not_run"


class SuiteMode(str, Enum):
    SELECTED = "selected"
    FULL_SUITE = "full_suite"


class FixtureChangeKind(str, Enum):
    """Controlled fixture change families required by the plan."""

    DIRECT_SYMBOL = "direct_symbol"
    TRANSITIVE = "transitive"
    FIXTURE_EDGE = "fixture_edge"
    CONFIG_EDGE = "config_edge"
    ENVIRONMENT = "environment"
    LOCK = "lock"
    UNRELATED = "unrelated"
    OPAQUE = "opaque"
    DYNAMIC = "dynamic"
    DELIBERATELY_FAILING = "deliberately_failing"
    EQUIVALENT_CONTROLLED = "equivalent_controlled"
    VALIDATION_MAPPING = "validation_mapping"
    FULL_SUITE_TIMEOUT = "full_suite_timeout"
    FULL_SUITE_UNAVAILABLE = "full_suite_unavailable"
    FLAKY = "flaky"
    ORDER_DEPENDENT = "order_dependent"
    FALSE_NEGATIVE_SEED = "false_negative_seed"
    FALSE_POSITIVE_SEED = "false_positive_seed"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        if required:
            raise EvaluationError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise EvaluationError(f"{field_name} must be a string")
    text = value.strip()
    if required and not text:
        raise EvaluationError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise EvaluationError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise EvaluationBoundsError(
            f"{field_name} exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
        )
    return text


def _optional_text(value: Any, *, field_name: str) -> str:
    if value is None or value == "":
        return ""
    return _text(value, field_name=field_name, required=True)


def _boolean(value: Any, *, field_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise EvaluationError(f"{field_name} must be a boolean")
    return value


def _non_negative_int(
    value: Any,
    *,
    field_name: str,
    default: int = 0,
    maximum: int = 7 * 24 * 60 * 60 * 1_000,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{field_name} must be an integer")
    if value < 0 or value > maximum:
        raise EvaluationBoundsError(
            f"{field_name} must be in [0, {maximum}]"
        )
    return value


def _string_tuple(
    value: Any,
    *,
    field_name: str,
    sort: bool = True,
    maximum: int = MAX_COLLECTION_ITEMS,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        items = value
    else:
        raise EvaluationError(f"{field_name} must be a sequence of strings")
    if len(items) > maximum:
        raise EvaluationBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _text(item, field_name=f"{field_name}[{index}]")
        if text not in seen:
            seen.add(text)
            result.append(text)
    if sort:
        result.sort()
    return tuple(result)


def _unique_sorted(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(item) for item in items if str(item)}))


def _stable_unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return tuple(ordered)


def _outcome(value: Any, *, field_name: str) -> ObservedTestOutcome:
    if isinstance(value, ObservedTestOutcome):
        return value
    if value is None:
        return ObservedTestOutcome.NOT_OBSERVED
    text = _text(str(value), field_name=field_name).lower()
    aliases = {
        "pass": ObservedTestOutcome.PASSED,
        "passed": ObservedTestOutcome.PASSED,
        "ok": ObservedTestOutcome.PASSED,
        "fail": ObservedTestOutcome.FAILED,
        "failed": ObservedTestOutcome.FAILED,
        "failure": ObservedTestOutcome.FAILED,
        "flaky": ObservedTestOutcome.FLAKY,
        "timeout": ObservedTestOutcome.TIMEOUT,
        "unavailable": ObservedTestOutcome.UNAVAILABLE,
        "error": ObservedTestOutcome.ERROR,
        "skip": ObservedTestOutcome.SKIPPED,
        "skipped": ObservedTestOutcome.SKIPPED,
        "not_observed": ObservedTestOutcome.NOT_OBSERVED,
        "not_run": ObservedTestOutcome.NOT_OBSERVED,
        "order_dependent": ObservedTestOutcome.ORDER_DEPENDENT,
    }
    if text not in aliases:
        raise EvaluationError(
            f"{field_name} has unknown outcome {value!r}"
        )
    return aliases[text]


def _suite_status(value: Any, *, field_name: str) -> SuiteRunStatus:
    if isinstance(value, SuiteRunStatus):
        return value
    text = _text(str(value), field_name=field_name).lower()
    try:
        return SuiteRunStatus(text)
    except ValueError as exc:
        raise EvaluationError(
            f"{field_name} has unknown suite status {value!r}"
        ) from exc


def _suite_mode(value: Any, *, field_name: str) -> SuiteMode:
    if isinstance(value, SuiteMode):
        return value
    text = _text(str(value), field_name=field_name).lower()
    try:
        return SuiteMode(text)
    except ValueError as exc:
        raise EvaluationError(
            f"{field_name} has unknown suite mode {value!r}"
        ) from exc


def _measurement_status(value: Any, *, field_name: str) -> MeasurementStatus:
    if isinstance(value, MeasurementStatus):
        return value
    text = _text(str(value), field_name=field_name).lower()
    try:
        return MeasurementStatus(text)
    except ValueError as exc:
        raise EvaluationError(
            f"{field_name} has unknown measurement status {value!r}"
        ) from exc


def _outcome_map(
    value: Any, *, field_name: str
) -> Mapping[str, ObservedTestOutcome]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{field_name} must be a mapping")
    if len(value) > MAX_COLLECTION_ITEMS:
        raise EvaluationBoundsError(
            f"{field_name} exceeds {MAX_COLLECTION_ITEMS} items"
        )
    result: dict[str, ObservedTestOutcome] = {}
    for raw_key, raw_outcome in value.items():
        key = _text(str(raw_key), field_name=f"{field_name}.key")
        result[key] = _outcome(
            raw_outcome, field_name=f"{field_name}[{key}]"
        )
    return MappingProxyType(dict(sorted(result.items())))


def _identity_token(*parts: str) -> str:
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/evaluation-identity@1",
        "parts": list(parts),
    }
    return content_identity(payload)


# ---------------------------------------------------------------------------
# Snapshot / observation records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationSnapshotIdentity:
    """Fresh shared identity for selected and full-suite observations."""

    SCHEMA: ClassVar[str] = EVALUATION_SNAPSHOT_SCHEMA

    tree_id: str
    environment_id: str
    lock_id: str
    fixture_id: str
    policy_id: str
    repository_id: str = ""
    schema: str = EVALUATION_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, field_name="tree_id")
        )
        object.__setattr__(
            self,
            "environment_id",
            _text(self.environment_id, field_name="environment_id"),
        )
        object.__setattr__(
            self, "lock_id", _text(self.lock_id, field_name="lock_id")
        )
        object.__setattr__(
            self,
            "fixture_id",
            _text(self.fixture_id, field_name="fixture_id"),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id")
        )
        object.__setattr__(
            self,
            "repository_id",
            _optional_text(self.repository_id, field_name="repository_id"),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or EVALUATION_SNAPSHOT_SCHEMA, field_name="schema"),
        )

    @property
    def identity_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tree_id": self.tree_id,
            "environment_id": self.environment_id,
            "lock_id": self.lock_id,
            "fixture_id": self.fixture_id,
            "policy_id": self.policy_id,
            "repository_id": self.repository_id,
            "identity_cid": content_identity(
                {
                    "schema": self.schema,
                    "tree_id": self.tree_id,
                    "environment_id": self.environment_id,
                    "lock_id": self.lock_id,
                    "fixture_id": self.fixture_id,
                    "policy_id": self.policy_id,
                    "repository_id": self.repository_id,
                }
            ),
        }

    @classmethod
    def from_value(
        cls, value: "EvaluationSnapshotIdentity | Mapping[str, Any]"
    ) -> "EvaluationSnapshotIdentity":
        if isinstance(value, EvaluationSnapshotIdentity):
            return value
        if not isinstance(value, Mapping):
            raise EvaluationError("snapshot identity must be a mapping")
        return cls(
            tree_id=str(value.get("tree_id") or ""),
            environment_id=str(value.get("environment_id") or ""),
            lock_id=str(value.get("lock_id") or ""),
            fixture_id=str(value.get("fixture_id") or ""),
            policy_id=str(value.get("policy_id") or ""),
            repository_id=str(value.get("repository_id") or ""),
            schema=str(value.get("schema") or EVALUATION_SNAPSHOT_SCHEMA),
        )

    def matches(self, other: "EvaluationSnapshotIdentity") -> bool:
        return (
            self.tree_id == other.tree_id
            and self.environment_id == other.environment_id
            and self.lock_id == other.lock_id
            and self.fixture_id == other.fixture_id
            and self.policy_id == other.policy_id
            and self.repository_id == other.repository_id
        )


@dataclass(frozen=True, slots=True)
class SuiteObservation:
    """One selected or full-suite observation under a fixed snapshot."""

    SCHEMA: ClassVar[str] = SUITE_OBSERVATION_SCHEMA

    mode: SuiteMode
    snapshot: EvaluationSnapshotIdentity
    suite_status: SuiteRunStatus
    test_outcomes: Mapping[str, ObservedTestOutcome] = field(
        default_factory=dict
    )
    test_order: tuple[str, ...] = ()
    selector_identity: str = ""
    duration_ms: int = 0
    wall_time_ms: int = 0
    reason_codes: tuple[str, ...] = ()
    schema: str = SUITE_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _suite_mode(self.mode, field_name="mode")
        )
        snap = EvaluationSnapshotIdentity.from_value(self.snapshot)
        object.__setattr__(self, "snapshot", snap)
        object.__setattr__(
            self,
            "suite_status",
            _suite_status(self.suite_status, field_name="suite_status"),
        )
        object.__setattr__(
            self,
            "test_outcomes",
            _outcome_map(self.test_outcomes, field_name="test_outcomes"),
        )
        order = _string_tuple(
            self.test_order, field_name="test_order", sort=False
        )
        # If order omitted, derive a stable order from outcomes.
        if not order:
            order = tuple(sorted(self.test_outcomes.keys()))
        object.__setattr__(self, "test_order", order)
        object.__setattr__(
            self,
            "selector_identity",
            _optional_text(
                self.selector_identity, field_name="selector_identity"
            ),
        )
        object.__setattr__(
            self,
            "duration_ms",
            _non_negative_int(self.duration_ms, field_name="duration_ms"),
        )
        object.__setattr__(
            self,
            "wall_time_ms",
            _non_negative_int(self.wall_time_ms, field_name="wall_time_ms"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _stable_unique(self.reason_codes)[:MAX_REASON_CODES],
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or SUITE_OBSERVATION_SCHEMA, field_name="schema"),
        )

    @property
    def failed_tests(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                node
                for node, outcome in self.test_outcomes.items()
                if outcome is ObservedTestOutcome.FAILED
            )
        )

    @property
    def observed_tests(self) -> tuple[str, ...]:
        return tuple(sorted(self.test_outcomes.keys()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode.value,
            "snapshot": self.snapshot.to_dict(),
            "suite_status": self.suite_status.value,
            "test_outcomes": {
                key: value.value for key, value in self.test_outcomes.items()
            },
            "test_order": list(self.test_order),
            "selector_identity": self.selector_identity,
            "duration_ms": self.duration_ms,
            "wall_time_ms": self.wall_time_ms,
            "reason_codes": list(self.reason_codes),
            "failed_tests": list(self.failed_tests),
        }

    @classmethod
    def from_value(
        cls, value: "SuiteObservation | Mapping[str, Any]"
    ) -> "SuiteObservation":
        if isinstance(value, SuiteObservation):
            return value
        if not isinstance(value, Mapping):
            raise EvaluationError("suite observation must be a mapping")
        return cls(
            mode=str(value.get("mode") or SuiteMode.SELECTED.value),
            snapshot=EvaluationSnapshotIdentity.from_value(
                value.get("snapshot") or {}
            ),
            suite_status=str(
                value.get("suite_status") or SuiteRunStatus.NOT_RUN.value
            ),
            test_outcomes=value.get("test_outcomes") or {},
            test_order=tuple(value.get("test_order") or ()),
            selector_identity=str(value.get("selector_identity") or ""),
            duration_ms=int(value.get("duration_ms") or 0),
            wall_time_ms=int(value.get("wall_time_ms") or 0),
            reason_codes=tuple(value.get("reason_codes") or ()),
            schema=str(value.get("schema") or SUITE_OBSERVATION_SCHEMA),
        )


def make_suite_observation(
    *,
    mode: SuiteMode | str,
    snapshot: EvaluationSnapshotIdentity | Mapping[str, Any],
    suite_status: SuiteRunStatus | str = SuiteRunStatus.COMPLETED,
    test_outcomes: Mapping[str, Any] | None = None,
    test_order: Sequence[str] | None = None,
    selector_identity: str = "",
    duration_ms: int = 0,
    wall_time_ms: int = 0,
    reason_codes: Sequence[str] | None = None,
) -> SuiteObservation:
    """Build a normalized suite observation (selected or full-suite)."""

    return SuiteObservation(
        mode=mode,
        snapshot=snapshot,
        suite_status=suite_status,
        test_outcomes=dict(test_outcomes or {}),
        test_order=tuple(test_order or ()),
        selector_identity=selector_identity,
        duration_ms=duration_ms,
        wall_time_ms=wall_time_ms,
        reason_codes=tuple(reason_codes or ()),
    )


def fresh_identical_observations(
    *,
    snapshot: EvaluationSnapshotIdentity | Mapping[str, Any],
    selected_outcomes: Mapping[str, Any],
    full_outcomes: Mapping[str, Any],
    selected_order: Sequence[str] | None = None,
    full_order: Sequence[str] | None = None,
    selected_status: SuiteRunStatus | str = SuiteRunStatus.COMPLETED,
    full_status: SuiteRunStatus | str = SuiteRunStatus.COMPLETED,
    selected_selector_identity: str = "",
    full_selector_identity: str = "",
    selected_duration_ms: int = 0,
    full_duration_ms: int = 0,
    selected_reasons: Sequence[str] | None = None,
    full_reasons: Sequence[str] | None = None,
) -> tuple[SuiteObservation, SuiteObservation]:
    """Return selected and full observations bound to the *same* snapshot."""

    snap = EvaluationSnapshotIdentity.from_value(snapshot)
    # Re-materialize snapshot objects so each observation is "fresh" while
    # remaining identity-equal.
    selected_snap = EvaluationSnapshotIdentity.from_value(snap.to_dict())
    full_snap = EvaluationSnapshotIdentity.from_value(snap.to_dict())
    selected = make_suite_observation(
        mode=SuiteMode.SELECTED,
        snapshot=selected_snap,
        suite_status=selected_status,
        test_outcomes=selected_outcomes,
        test_order=selected_order,
        selector_identity=selected_selector_identity,
        duration_ms=selected_duration_ms,
        wall_time_ms=selected_duration_ms,
        reason_codes=selected_reasons,
    )
    full = make_suite_observation(
        mode=SuiteMode.FULL_SUITE,
        snapshot=full_snap,
        suite_status=full_status,
        test_outcomes=full_outcomes,
        test_order=full_order,
        selector_identity=full_selector_identity or selected_selector_identity,
        duration_ms=full_duration_ms,
        wall_time_ms=full_duration_ms,
        reason_codes=full_reasons,
    )
    return selected, full


# ---------------------------------------------------------------------------
# Controlled fixture recipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlledSemanticFixture:
    """Reviewed controlled semantic-capsule fixture recipe.

    Ground-truth affected tests are the authority for selection false
    positives/negatives.  Full-suite failure outcomes form a separate oracle.
    """

    SCHEMA: ClassVar[str] = CONTROLLED_FIXTURE_SCHEMA

    fixture_id: str
    change_kind: str
    snapshot: EvaluationSnapshotIdentity
    ground_truth_affected_tests: tuple[str, ...]
    all_tests: tuple[str, ...]
    changed_symbols: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    edges: tuple[Mapping[str, Any], ...] = ()
    catalog: Mapping[str, Any] = field(default_factory=dict)
    policy: Mapping[str, Any] = field(default_factory=dict)
    validation_selection: Mapping[str, Any] | None = None
    uncovered_symbols: tuple[str, ...] = ()
    uncovered_paths: tuple[str, ...] = ()
    truncated: bool = False
    requires_broader_selection: bool = False
    equivalence_label: str = ""
    corpus_id: str = CANONICAL_CORPUS_ID
    corpus_present: bool = True
    description: str = ""
    selected_observation: SuiteObservation | None = None
    full_suite_observation: SuiteObservation | None = None
    # When set, override the selector result's selected test set (seeded FN/FP).
    forced_selected_tests: tuple[str, ...] | None = None
    schema: str = CONTROLLED_FIXTURE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixture_id",
            _text(self.fixture_id, field_name="fixture_id"),
        )
        object.__setattr__(
            self,
            "change_kind",
            _text(self.change_kind, field_name="change_kind"),
        )
        object.__setattr__(
            self,
            "snapshot",
            EvaluationSnapshotIdentity.from_value(self.snapshot),
        )
        object.__setattr__(
            self,
            "ground_truth_affected_tests",
            _string_tuple(
                self.ground_truth_affected_tests,
                field_name="ground_truth_affected_tests",
            ),
        )
        object.__setattr__(
            self,
            "all_tests",
            _string_tuple(self.all_tests, field_name="all_tests"),
        )
        object.__setattr__(
            self,
            "changed_symbols",
            _string_tuple(self.changed_symbols, field_name="changed_symbols"),
        )
        object.__setattr__(
            self,
            "changed_paths",
            _string_tuple(self.changed_paths, field_name="changed_paths"),
        )
        edges: list[Mapping[str, Any]] = []
        for index, edge in enumerate(self.edges or ()):
            if not isinstance(edge, Mapping):
                raise EvaluationError(f"edges[{index}] must be a mapping")
            edges.append(MappingProxyType(dict(edge)))
        object.__setattr__(self, "edges", tuple(edges))
        catalog = self.catalog if isinstance(self.catalog, Mapping) else {}
        object.__setattr__(self, "catalog", MappingProxyType(dict(catalog)))
        policy = self.policy if isinstance(self.policy, Mapping) else {}
        object.__setattr__(self, "policy", MappingProxyType(dict(policy)))
        if self.validation_selection is not None:
            if not isinstance(self.validation_selection, Mapping):
                raise EvaluationError(
                    "validation_selection must be a mapping or None"
                )
            object.__setattr__(
                self,
                "validation_selection",
                MappingProxyType(dict(self.validation_selection)),
            )
        object.__setattr__(
            self,
            "uncovered_symbols",
            _string_tuple(
                self.uncovered_symbols, field_name="uncovered_symbols"
            ),
        )
        object.__setattr__(
            self,
            "uncovered_paths",
            _string_tuple(self.uncovered_paths, field_name="uncovered_paths"),
        )
        object.__setattr__(
            self,
            "truncated",
            _boolean(self.truncated, field_name="truncated"),
        )
        object.__setattr__(
            self,
            "requires_broader_selection",
            _boolean(
                self.requires_broader_selection,
                field_name="requires_broader_selection",
            ),
        )
        object.__setattr__(
            self,
            "equivalence_label",
            _optional_text(
                self.equivalence_label, field_name="equivalence_label"
            ),
        )
        object.__setattr__(
            self,
            "corpus_id",
            _text(
                self.corpus_id or CANONICAL_CORPUS_ID, field_name="corpus_id"
            ),
        )
        object.__setattr__(
            self,
            "corpus_present",
            _boolean(self.corpus_present, field_name="corpus_present", default=True),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, field_name="description"),
        )
        if self.selected_observation is not None:
            object.__setattr__(
                self,
                "selected_observation",
                SuiteObservation.from_value(self.selected_observation),
            )
        if self.full_suite_observation is not None:
            object.__setattr__(
                self,
                "full_suite_observation",
                SuiteObservation.from_value(self.full_suite_observation),
            )
        if self.forced_selected_tests is not None:
            object.__setattr__(
                self,
                "forced_selected_tests",
                _string_tuple(
                    self.forced_selected_tests,
                    field_name="forced_selected_tests",
                ),
            )
        object.__setattr__(
            self,
            "schema",
            _text(
                self.schema or CONTROLLED_FIXTURE_SCHEMA, field_name="schema"
            ),
        )
        # Snapshot fixture_id should align with recipe fixture_id when blank.
        if self.snapshot.fixture_id != self.fixture_id:
            # Keep both; mismatch is recorded at comparison time if observations
            # disagree, but recipe fixture_id is authoritative for labeling.
            pass

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "fixture_id": self.fixture_id,
            "change_kind": self.change_kind,
            "snapshot": self.snapshot.to_dict(),
            "ground_truth_affected_tests": list(
                self.ground_truth_affected_tests
            ),
            "all_tests": list(self.all_tests),
            "changed_symbols": list(self.changed_symbols),
            "changed_paths": list(self.changed_paths),
            "edges": [dict(edge) for edge in self.edges],
            "catalog": dict(self.catalog),
            "policy": dict(self.policy),
            "validation_selection": (
                dict(self.validation_selection)
                if self.validation_selection is not None
                else None
            ),
            "uncovered_symbols": list(self.uncovered_symbols),
            "uncovered_paths": list(self.uncovered_paths),
            "truncated": self.truncated,
            "requires_broader_selection": self.requires_broader_selection,
            "equivalence_label": self.equivalence_label,
            "corpus_id": self.corpus_id,
            "corpus_present": self.corpus_present,
            "description": self.description,
            "forced_selected_tests": (
                list(self.forced_selected_tests)
                if self.forced_selected_tests is not None
                else None
            ),
        }
        if self.selected_observation is not None:
            payload["selected_observation"] = (
                self.selected_observation.to_dict()
            )
        if self.full_suite_observation is not None:
            payload["full_suite_observation"] = (
                self.full_suite_observation.to_dict()
            )
        return payload

    @classmethod
    def from_value(
        cls, value: "ControlledSemanticFixture | Mapping[str, Any]"
    ) -> "ControlledSemanticFixture":
        if isinstance(value, ControlledSemanticFixture):
            return value
        if not isinstance(value, Mapping):
            raise EvaluationError("controlled fixture must be a mapping")
        forced = value.get("forced_selected_tests", None)
        return cls(
            fixture_id=str(value.get("fixture_id") or ""),
            change_kind=str(value.get("change_kind") or ""),
            snapshot=EvaluationSnapshotIdentity.from_value(
                value.get("snapshot") or {}
            ),
            ground_truth_affected_tests=tuple(
                value.get("ground_truth_affected_tests") or ()
            ),
            all_tests=tuple(value.get("all_tests") or ()),
            changed_symbols=tuple(value.get("changed_symbols") or ()),
            changed_paths=tuple(value.get("changed_paths") or ()),
            edges=tuple(value.get("edges") or ()),
            catalog=dict(value.get("catalog") or {}),
            policy=dict(value.get("policy") or {}),
            validation_selection=value.get("validation_selection"),
            uncovered_symbols=tuple(value.get("uncovered_symbols") or ()),
            uncovered_paths=tuple(value.get("uncovered_paths") or ()),
            truncated=bool(value.get("truncated", False)),
            requires_broader_selection=bool(
                value.get("requires_broader_selection", False)
            ),
            equivalence_label=str(value.get("equivalence_label") or ""),
            corpus_id=str(value.get("corpus_id") or CANONICAL_CORPUS_ID),
            corpus_present=bool(value.get("corpus_present", True)),
            description=str(value.get("description") or ""),
            selected_observation=value.get("selected_observation"),
            full_suite_observation=value.get("full_suite_observation"),
            forced_selected_tests=(
                None if forced is None else tuple(forced)
            ),
            schema=str(value.get("schema") or CONTROLLED_FIXTURE_SCHEMA),
        )


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestSelectionEvaluation:
    """Differential evaluation of selected tests versus full-suite oracle.

    Interface: ``TestSelectionEvaluation@1``.
    Evidence: ``ivp/test-selection-evaluation@1``.
    Never authoritative; never asserts target success.
    """

    __test__ = False

    SCHEMA: ClassVar[str] = TEST_SELECTION_EVALUATION_SCHEMA
    INTERFACE: ClassVar[str] = TEST_SELECTION_EVALUATION_INTERFACE

    schema: str = TEST_SELECTION_EVALUATION_SCHEMA
    interface: str = TEST_SELECTION_EVALUATION_INTERFACE
    evidence: str = SELECTION_EVALUATION_EVIDENCE
    fixture_id: str = ""
    change_kind: str = ""
    equivalence_label: str = ""
    corpus_id: str = ""
    measurement_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    snapshot: EvaluationSnapshotIdentity | None = None
    ground_truth_affected_tests: tuple[str, ...] = ()
    selected_tests: tuple[str, ...] = ()
    full_suite_tests: tuple[str, ...] = ()
    # Ground-truth set oracle.
    ground_truth_false_negatives: tuple[str, ...] = ()
    ground_truth_false_positives: tuple[str, ...] = ()
    # Full-suite failure oracle (separate).
    full_suite_oracle_false_negatives: tuple[str, ...] = ()
    full_suite_failures: tuple[str, ...] = ()
    selected_failures: tuple[str, ...] = ()
    # Aggregate (only meaningful when measurement_status is measured).
    false_negative_tests: tuple[str, ...] = ()
    false_positive_tests: tuple[str, ...] = ()
    inconclusive_tests: tuple[str, ...] = ()
    # Counts are None when not_measured so callers cannot treat absence as zero.
    false_negative_count: int | None = None
    false_positive_count: int | None = None
    inconclusive_count: int | None = None
    snapshots_identical: bool = False
    broader_suite_required_before_acceptance: bool = False
    acceptance_blocked_reasons: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    selection_fallback_mode: str = ""
    selection_broader_required: bool = False
    selection_full_suite_required: bool = False
    # Bound identities / timing (evidence only).
    repository_id: str = ""
    policy_id: str = ""
    environment_id: str = ""
    selector_identity: str = ""
    selected_duration_ms: int = 0
    full_suite_duration_ms: int = 0
    evaluation_duration_ms: int = 0
    selected_suite_status: str = ""
    full_suite_status: str = ""
    not_measured_reasons: tuple[str, ...] = ()
    inconclusive_reasons: tuple[str, ...] = ()
    authoritative: bool = False
    target_success_asserted: bool = False
    evaluated: bool = False

    def __post_init__(self) -> None:
        # Never authoritative; never assert target success.
        object.__setattr__(self, "authoritative", False)
        object.__setattr__(self, "target_success_asserted", False)
        object.__setattr__(
            self,
            "schema",
            _text(
                self.schema or TEST_SELECTION_EVALUATION_SCHEMA,
                field_name="schema",
            ),
        )
        object.__setattr__(
            self,
            "interface",
            _text(
                self.interface or TEST_SELECTION_EVALUATION_INTERFACE,
                field_name="interface",
            ),
        )
        object.__setattr__(
            self,
            "evidence",
            _text(
                self.evidence or SELECTION_EVALUATION_EVIDENCE,
                field_name="evidence",
            ),
        )
        object.__setattr__(
            self,
            "fixture_id",
            _optional_text(self.fixture_id, field_name="fixture_id"),
        )
        object.__setattr__(
            self,
            "change_kind",
            _optional_text(self.change_kind, field_name="change_kind"),
        )
        object.__setattr__(
            self,
            "equivalence_label",
            _optional_text(
                self.equivalence_label, field_name="equivalence_label"
            ),
        )
        object.__setattr__(
            self,
            "corpus_id",
            _optional_text(self.corpus_id, field_name="corpus_id"),
        )
        object.__setattr__(
            self,
            "measurement_status",
            _measurement_status(
                self.measurement_status, field_name="measurement_status"
            ),
        )
        if self.snapshot is not None:
            object.__setattr__(
                self,
                "snapshot",
                EvaluationSnapshotIdentity.from_value(self.snapshot),
            )
        for attr in (
            "ground_truth_affected_tests",
            "selected_tests",
            "full_suite_tests",
            "ground_truth_false_negatives",
            "ground_truth_false_positives",
            "full_suite_oracle_false_negatives",
            "full_suite_failures",
            "selected_failures",
            "false_negative_tests",
            "false_positive_tests",
            "inconclusive_tests",
        ):
            object.__setattr__(
                self,
                attr,
                _string_tuple(getattr(self, attr), field_name=attr),
            )
        object.__setattr__(
            self,
            "acceptance_blocked_reasons",
            _stable_unique(self.acceptance_blocked_reasons)[:MAX_REASON_CODES],
        )
        object.__setattr__(
            self,
            "reason_codes",
            _stable_unique(self.reason_codes)[:MAX_REASON_CODES],
        )
        object.__setattr__(
            self,
            "not_measured_reasons",
            _stable_unique(self.not_measured_reasons)[:MAX_REASON_CODES],
        )
        object.__setattr__(
            self,
            "inconclusive_reasons",
            _stable_unique(self.inconclusive_reasons)[:MAX_REASON_CODES],
        )
        # Enforce not_measured never reports zero FN/FP counts.
        if self.measurement_status is MeasurementStatus.NOT_MEASURED:
            object.__setattr__(self, "false_negative_count", None)
            object.__setattr__(self, "false_positive_count", None)
            object.__setattr__(self, "inconclusive_count", None)
        elif self.measurement_status is MeasurementStatus.INCONCLUSIVE:
            # Inconclusive may still report partial counts for transparent
            # bookkeeping, but measured FN/FP floors are not claimed.
            if self.false_negative_count is None:
                object.__setattr__(
                    self,
                    "false_negative_count",
                    len(self.false_negative_tests),
                )
            if self.false_positive_count is None:
                object.__setattr__(
                    self,
                    "false_positive_count",
                    len(self.false_positive_tests),
                )
            if self.inconclusive_count is None:
                object.__setattr__(
                    self,
                    "inconclusive_count",
                    len(self.inconclusive_tests),
                )
        else:
            object.__setattr__(
                self,
                "false_negative_count",
                len(self.false_negative_tests),
            )
            object.__setattr__(
                self,
                "false_positive_count",
                len(self.false_positive_tests),
            )
            object.__setattr__(
                self,
                "inconclusive_count",
                len(self.inconclusive_tests),
            )
        object.__setattr__(
            self,
            "snapshots_identical",
            _boolean(
                self.snapshots_identical, field_name="snapshots_identical"
            ),
        )
        object.__setattr__(
            self,
            "broader_suite_required_before_acceptance",
            _boolean(
                self.broader_suite_required_before_acceptance,
                field_name="broader_suite_required_before_acceptance",
            ),
        )
        object.__setattr__(
            self,
            "selection_fallback_mode",
            _optional_text(
                self.selection_fallback_mode,
                field_name="selection_fallback_mode",
            ),
        )
        object.__setattr__(
            self,
            "repository_id",
            _optional_text(self.repository_id, field_name="repository_id"),
        )
        object.__setattr__(
            self,
            "policy_id",
            _optional_text(self.policy_id, field_name="policy_id"),
        )
        object.__setattr__(
            self,
            "environment_id",
            _optional_text(self.environment_id, field_name="environment_id"),
        )
        object.__setattr__(
            self,
            "selector_identity",
            _optional_text(
                self.selector_identity, field_name="selector_identity"
            ),
        )
        for attr in (
            "selected_duration_ms",
            "full_suite_duration_ms",
            "evaluation_duration_ms",
        ):
            object.__setattr__(
                self,
                attr,
                _non_negative_int(getattr(self, attr), field_name=attr),
            )
        object.__setattr__(
            self,
            "selected_suite_status",
            _optional_text(
                self.selected_suite_status, field_name="selected_suite_status"
            ),
        )
        object.__setattr__(
            self,
            "full_suite_status",
            _optional_text(
                self.full_suite_status, field_name="full_suite_status"
            ),
        )
        object.__setattr__(
            self, "evaluated", _boolean(self.evaluated, field_name="evaluated")
        )

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence": self.evidence,
            "fixture_id": self.fixture_id,
            "change_kind": self.change_kind,
            "equivalence_label": self.equivalence_label,
            "corpus_id": self.corpus_id,
            "measurement_status": self.measurement_status.value,
            "snapshot": (
                self.snapshot.to_dict() if self.snapshot is not None else None
            ),
            "ground_truth_affected_tests": list(
                self.ground_truth_affected_tests
            ),
            "selected_tests": list(self.selected_tests),
            "full_suite_tests": list(self.full_suite_tests),
            "ground_truth_false_negatives": list(
                self.ground_truth_false_negatives
            ),
            "ground_truth_false_positives": list(
                self.ground_truth_false_positives
            ),
            "full_suite_oracle_false_negatives": list(
                self.full_suite_oracle_false_negatives
            ),
            "full_suite_failures": list(self.full_suite_failures),
            "selected_failures": list(self.selected_failures),
            "false_negative_tests": list(self.false_negative_tests),
            "false_positive_tests": list(self.false_positive_tests),
            "inconclusive_tests": list(self.inconclusive_tests),
            "false_negative_count": self.false_negative_count,
            "false_positive_count": self.false_positive_count,
            "inconclusive_count": self.inconclusive_count,
            "snapshots_identical": self.snapshots_identical,
            "broader_suite_required_before_acceptance": (
                self.broader_suite_required_before_acceptance
            ),
            "acceptance_blocked_reasons": list(self.acceptance_blocked_reasons),
            "reason_codes": list(self.reason_codes),
            "selection_fallback_mode": self.selection_fallback_mode,
            "selection_broader_required": self.selection_broader_required,
            "selection_full_suite_required": self.selection_full_suite_required,
            "repository_id": self.repository_id,
            "policy_id": self.policy_id,
            "environment_id": self.environment_id,
            "selector_identity": self.selector_identity,
            "selected_duration_ms": self.selected_duration_ms,
            "full_suite_duration_ms": self.full_suite_duration_ms,
            "evaluation_duration_ms": self.evaluation_duration_ms,
            "selected_suite_status": self.selected_suite_status,
            "full_suite_status": self.full_suite_status,
            "not_measured_reasons": list(self.not_measured_reasons),
            "inconclusive_reasons": list(self.inconclusive_reasons),
            "authoritative": False,
            "target_success_asserted": False,
            "evaluated": self.evaluated,
            "content_id": content_identity(
                {
                    "schema": self.schema,
                    "interface": self.interface,
                    "evidence": self.evidence,
                    "fixture_id": self.fixture_id,
                    "measurement_status": self.measurement_status.value,
                    "false_negative_tests": list(self.false_negative_tests),
                    "false_positive_tests": list(self.false_positive_tests),
                    "corpus_id": self.corpus_id,
                }
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TestSelectionEvaluation":
        if not isinstance(payload, Mapping):
            raise EvaluationError("evaluation payload must be a mapping")
        snapshot_raw = payload.get("snapshot")
        return cls(
            schema=str(
                payload.get("schema") or TEST_SELECTION_EVALUATION_SCHEMA
            ),
            interface=str(
                payload.get("interface")
                or TEST_SELECTION_EVALUATION_INTERFACE
            ),
            evidence=str(
                payload.get("evidence") or SELECTION_EVALUATION_EVIDENCE
            ),
            fixture_id=str(payload.get("fixture_id") or ""),
            change_kind=str(payload.get("change_kind") or ""),
            equivalence_label=str(payload.get("equivalence_label") or ""),
            corpus_id=str(payload.get("corpus_id") or ""),
            measurement_status=str(
                payload.get("measurement_status")
                or MeasurementStatus.NOT_MEASURED.value
            ),
            snapshot=(
                EvaluationSnapshotIdentity.from_value(snapshot_raw)
                if snapshot_raw
                else None
            ),
            ground_truth_affected_tests=tuple(
                payload.get("ground_truth_affected_tests") or ()
            ),
            selected_tests=tuple(payload.get("selected_tests") or ()),
            full_suite_tests=tuple(payload.get("full_suite_tests") or ()),
            ground_truth_false_negatives=tuple(
                payload.get("ground_truth_false_negatives") or ()
            ),
            ground_truth_false_positives=tuple(
                payload.get("ground_truth_false_positives") or ()
            ),
            full_suite_oracle_false_negatives=tuple(
                payload.get("full_suite_oracle_false_negatives") or ()
            ),
            full_suite_failures=tuple(
                payload.get("full_suite_failures") or ()
            ),
            selected_failures=tuple(payload.get("selected_failures") or ()),
            false_negative_tests=tuple(
                payload.get("false_negative_tests") or ()
            ),
            false_positive_tests=tuple(
                payload.get("false_positive_tests") or ()
            ),
            inconclusive_tests=tuple(
                payload.get("inconclusive_tests") or ()
            ),
            false_negative_count=payload.get("false_negative_count"),
            false_positive_count=payload.get("false_positive_count"),
            inconclusive_count=payload.get("inconclusive_count"),
            snapshots_identical=bool(
                payload.get("snapshots_identical", False)
            ),
            broader_suite_required_before_acceptance=bool(
                payload.get("broader_suite_required_before_acceptance", False)
            ),
            acceptance_blocked_reasons=tuple(
                payload.get("acceptance_blocked_reasons") or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            selection_fallback_mode=str(
                payload.get("selection_fallback_mode") or ""
            ),
            selection_broader_required=bool(
                payload.get("selection_broader_required", False)
            ),
            selection_full_suite_required=bool(
                payload.get("selection_full_suite_required", False)
            ),
            repository_id=str(payload.get("repository_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            environment_id=str(payload.get("environment_id") or ""),
            selector_identity=str(payload.get("selector_identity") or ""),
            selected_duration_ms=int(
                payload.get("selected_duration_ms") or 0
            ),
            full_suite_duration_ms=int(
                payload.get("full_suite_duration_ms") or 0
            ),
            evaluation_duration_ms=int(
                payload.get("evaluation_duration_ms") or 0
            ),
            selected_suite_status=str(
                payload.get("selected_suite_status") or ""
            ),
            full_suite_status=str(payload.get("full_suite_status") or ""),
            not_measured_reasons=tuple(
                payload.get("not_measured_reasons") or ()
            ),
            inconclusive_reasons=tuple(
                payload.get("inconclusive_reasons") or ()
            ),
            evaluated=bool(payload.get("evaluated", False)),
        )


@dataclass(frozen=True, slots=True)
class SelectionCorpusEvaluationSummary:
    """Aggregate evaluation over a controlled fixture corpus."""

    SCHEMA: ClassVar[str] = CORPUS_EVALUATION_SUMMARY_SCHEMA
    INTERFACE: ClassVar[str] = CORPUS_EVALUATION_SUMMARY_INTERFACE

    schema: str = CORPUS_EVALUATION_SUMMARY_SCHEMA
    interface: str = CORPUS_EVALUATION_SUMMARY_INTERFACE
    evidence: str = SELECTION_EVALUATION_EVIDENCE
    corpus_id: str = ""
    corpus_present: bool = True
    measurement_status: MeasurementStatus = MeasurementStatus.NOT_MEASURED
    evaluated_count: int = 0
    measured_count: int = 0
    not_measured_count: int = 0
    inconclusive_count: int = 0
    total_false_negatives: int | None = None
    total_false_positives: int | None = None
    fixture_ids: tuple[str, ...] = ()
    equivalence_labels: tuple[str, ...] = ()
    evaluations: tuple[TestSelectionEvaluation, ...] = ()
    reason_codes: tuple[str, ...] = ()
    repository_id: str = ""
    policy_id: str = ""
    environment_id: str = ""
    evaluation_duration_ms: int = 0
    authoritative: bool = False
    target_success_asserted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        object.__setattr__(self, "target_success_asserted", False)
        object.__setattr__(
            self,
            "measurement_status",
            _measurement_status(
                self.measurement_status, field_name="measurement_status"
            ),
        )
        object.__setattr__(
            self,
            "corpus_id",
            _optional_text(self.corpus_id, field_name="corpus_id"),
        )
        object.__setattr__(
            self,
            "fixture_ids",
            _string_tuple(self.fixture_ids, field_name="fixture_ids"),
        )
        object.__setattr__(
            self,
            "equivalence_labels",
            _string_tuple(
                self.equivalence_labels, field_name="equivalence_labels"
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _stable_unique(self.reason_codes)[:MAX_REASON_CODES],
        )
        evals = tuple(self.evaluations or ())
        for item in evals:
            if not isinstance(item, TestSelectionEvaluation):
                raise EvaluationError(
                    "evaluations must contain TestSelectionEvaluation"
                )
        object.__setattr__(self, "evaluations", evals)
        if self.measurement_status is MeasurementStatus.NOT_MEASURED:
            object.__setattr__(self, "total_false_negatives", None)
            object.__setattr__(self, "total_false_positives", None)
        object.__setattr__(
            self,
            "evaluated_count",
            _non_negative_int(
                self.evaluated_count, field_name="evaluated_count"
            ),
        )
        object.__setattr__(
            self,
            "evaluation_duration_ms",
            _non_negative_int(
                self.evaluation_duration_ms,
                field_name="evaluation_duration_ms",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence": self.evidence,
            "corpus_id": self.corpus_id,
            "corpus_present": self.corpus_present,
            "measurement_status": self.measurement_status.value,
            "evaluated_count": self.evaluated_count,
            "measured_count": self.measured_count,
            "not_measured_count": self.not_measured_count,
            "inconclusive_count": self.inconclusive_count,
            "total_false_negatives": self.total_false_negatives,
            "total_false_positives": self.total_false_positives,
            "fixture_ids": list(self.fixture_ids),
            "equivalence_labels": list(self.equivalence_labels),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "reason_codes": list(self.reason_codes),
            "repository_id": self.repository_id,
            "policy_id": self.policy_id,
            "environment_id": self.environment_id,
            "evaluation_duration_ms": self.evaluation_duration_ms,
            "authoritative": False,
            "target_success_asserted": False,
        }


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def _run_selection(
    fixture: ControlledSemanticFixture,
) -> AffectedVerificationSelection:
    catalog_payload = dict(fixture.catalog)
    if "tests" not in catalog_payload and fixture.all_tests:
        catalog_payload["tests"] = list(fixture.all_tests)
    policy_payload = dict(fixture.policy)
    try:
        return select_affected_verification(
            changed_symbols=fixture.changed_symbols,
            changed_paths=fixture.changed_paths,
            edges=list(fixture.edges),
            uncovered_symbols=fixture.uncovered_symbols,
            uncovered_paths=fixture.uncovered_paths,
            truncated=fixture.truncated,
            requires_broader_selection=fixture.requires_broader_selection,
            validation_selection=fixture.validation_selection,
            catalog=VerificationCatalog.from_value(catalog_payload)
            if catalog_payload
            else None,
            policy=SelectionPolicy.from_value(policy_payload)
            if policy_payload
            else None,
        )
    except SelectionError as exc:
        raise EvaluationError(f"selection failed: {exc}") from exc


def _selector_identity_for(
    fixture: ControlledSemanticFixture,
    selection: AffectedVerificationSelection,
    selected_tests: Sequence[str],
) -> str:
    return _identity_token(
        "selector",
        fixture.fixture_id,
        fixture.snapshot.policy_id,
        ",".join(selected_tests),
        selection.fallback_mode.value,
    )


def _validation_mapping_incomplete(
    fixture: ControlledSemanticFixture,
    selection: AffectedVerificationSelection,
) -> bool:
    if fixture.validation_selection is None:
        return False
    unmapped = fixture.validation_selection.get("unmapped_validation_ids") or ()
    requires = bool(
        fixture.validation_selection.get("requires_broader_selection", False)
    )
    if unmapped or requires:
        return True
    # Selection reason codes also encode incomplete mapping.
    codes = set(selection.fallback_reason_codes) | set(
        selection.full_suite_reason_codes
    )
    return "validation_id_mapping_incomplete" in codes


def _uncertain_selector(
    fixture: ControlledSemanticFixture,
    selection: AffectedVerificationSelection,
) -> bool:
    if fixture.uncovered_symbols or fixture.uncovered_paths or fixture.truncated:
        return True
    if fixture.requires_broader_selection:
        return True
    if selection.broader_selection_required or selection.full_suite_required:
        return True
    if selection.critical_uncertain_edges:
        return True
    if selection.fallback_mode is not FallbackMode.EXACT:
        return True
    return False


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def compare_selected_with_full_suite(
    *,
    fixture: ControlledSemanticFixture | Mapping[str, Any],
    selected: SuiteObservation | Mapping[str, Any] | None = None,
    full_suite: SuiteObservation | Mapping[str, Any] | None = None,
    selection: AffectedVerificationSelection | Mapping[str, Any] | None = None,
    corpus_available: bool | None = None,
    selected_tests_override: Sequence[str] | None = None,
) -> TestSelectionEvaluation:
    """Compare selected tests with full-suite oracle for one controlled fixture.

    Parameters
    ----------
    fixture:
        Controlled semantic-capsule fixture with reviewed ground-truth
        affected tests and shared snapshot identities.
    selected / full_suite:
        Fresh observations.  When omitted, the fixture's embedded observations
        are used.  Both must bind identical snapshot identities when measured.
    selection:
        Optional precomputed :class:`AffectedVerificationSelection`.  When
        omitted, selection is computed from the fixture recipe.
    corpus_available:
        Override corpus presence.  Defaults to ``fixture.corpus_present``.
    selected_tests_override:
        Optional forced selected set (seeded FN/FP fixtures).
    """

    started = time.perf_counter()
    fixture_obj = ControlledSemanticFixture.from_value(fixture)
    corpus_ok = (
        fixture_obj.corpus_present
        if corpus_available is None
        else bool(corpus_available)
    )

    # --- Resolve selection ---
    selection_obj: AffectedVerificationSelection | None = None
    if selection is not None:
        if isinstance(selection, AffectedVerificationSelection):
            selection_obj = selection
        elif isinstance(selection, Mapping):
            selection_obj = AffectedVerificationSelection.from_dict(selection)
        else:
            raise EvaluationError("selection must be a mapping or object")
    else:
        try:
            selection_obj = _run_selection(fixture_obj)
        except EvaluationError:
            selection_obj = None

    if selected_tests_override is not None:
        selected_tests = _string_tuple(
            selected_tests_override, field_name="selected_tests_override"
        )
    elif fixture_obj.forced_selected_tests is not None:
        selected_tests = fixture_obj.forced_selected_tests
    elif selection_obj is not None:
        selected_tests = selection_obj.selected_tests
    else:
        selected_tests = ()

    # --- Resolve observations ---
    selected_obs: SuiteObservation | None = None
    full_obs: SuiteObservation | None = None
    if selected is not None:
        selected_obs = SuiteObservation.from_value(selected)
    elif fixture_obj.selected_observation is not None:
        selected_obs = fixture_obj.selected_observation
    if full_suite is not None:
        full_obs = SuiteObservation.from_value(full_suite)
    elif fixture_obj.full_suite_observation is not None:
        full_obs = fixture_obj.full_suite_observation

    reasons: list[str] = [REASON_TARGET_SUCCESS_NOT_ASSERTED]
    not_measured_reasons: list[str] = []
    inconclusive_reasons: list[str] = []
    acceptance_blocked: list[str] = []

    broader_required = False
    selection_broader = False
    selection_full = False
    fallback_mode = ""
    if selection_obj is not None:
        selection_broader = selection_obj.broader_selection_required
        selection_full = selection_obj.full_suite_required
        fallback_mode = selection_obj.fallback_mode.value
        if _uncertain_selector(fixture_obj, selection_obj):
            broader_required = True
            reasons.append(REASON_UNCERTAIN_SELECTOR)
            reasons.append(REASON_BROADER_REQUIRED)
        if _validation_mapping_incomplete(fixture_obj, selection_obj):
            broader_required = True
            reasons.append(REASON_VALIDATION_MAPPING_MISSING)
            reasons.append(REASON_BROADER_REQUIRED)
            acceptance_blocked.append(REASON_VALIDATION_MAPPING_MISSING)
        if broader_required:
            acceptance_blocked.append(REASON_BROADER_REQUIRED)

    snapshot = fixture_obj.snapshot
    repository_id = snapshot.repository_id
    policy_id = snapshot.policy_id
    environment_id = snapshot.environment_id
    selector_identity = ""
    if selection_obj is not None:
        selector_identity = _selector_identity_for(
            fixture_obj, selection_obj, selected_tests
        )
    if selected_obs is not None and selected_obs.selector_identity:
        selector_identity = selected_obs.selector_identity or selector_identity

    selected_duration = selected_obs.duration_ms if selected_obs else 0
    full_duration = full_obs.duration_ms if full_obs else 0
    selected_status = (
        selected_obs.suite_status.value if selected_obs else ""
    )
    full_status = full_obs.suite_status.value if full_obs else ""

    # --- not_measured gates ---
    if not corpus_ok:
        not_measured_reasons.append(REASON_CORPUS_ABSENT)
        reasons.append(REASON_CORPUS_ABSENT)
        elapsed = int((time.perf_counter() - started) * 1000)
        return TestSelectionEvaluation(
            fixture_id=fixture_obj.fixture_id,
            change_kind=fixture_obj.change_kind,
            equivalence_label=fixture_obj.equivalence_label,
            corpus_id=fixture_obj.corpus_id,
            measurement_status=MeasurementStatus.NOT_MEASURED,
            snapshot=snapshot,
            ground_truth_affected_tests=fixture_obj.ground_truth_affected_tests,
            selected_tests=selected_tests,
            full_suite_tests=(),
            snapshots_identical=False,
            broader_suite_required_before_acceptance=broader_required
            or selection_broader
            or selection_full,
            acceptance_blocked_reasons=tuple(acceptance_blocked),
            reason_codes=tuple(reasons),
            selection_fallback_mode=fallback_mode,
            selection_broader_required=selection_broader,
            selection_full_suite_required=selection_full,
            repository_id=repository_id,
            policy_id=policy_id,
            environment_id=environment_id,
            selector_identity=selector_identity,
            selected_duration_ms=selected_duration,
            full_suite_duration_ms=full_duration,
            evaluation_duration_ms=elapsed,
            selected_suite_status=selected_status,
            full_suite_status=full_status,
            not_measured_reasons=tuple(not_measured_reasons),
            evaluated=False,
        )

    if full_obs is None:
        not_measured_reasons.append(REASON_FULL_SUITE_UNAVAILABLE)
        reasons.append(REASON_FULL_SUITE_UNAVAILABLE)
        elapsed = int((time.perf_counter() - started) * 1000)
        return TestSelectionEvaluation(
            fixture_id=fixture_obj.fixture_id,
            change_kind=fixture_obj.change_kind,
            equivalence_label=fixture_obj.equivalence_label,
            corpus_id=fixture_obj.corpus_id,
            measurement_status=MeasurementStatus.NOT_MEASURED,
            snapshot=snapshot,
            ground_truth_affected_tests=fixture_obj.ground_truth_affected_tests,
            selected_tests=selected_tests,
            full_suite_tests=(),
            snapshots_identical=False,
            broader_suite_required_before_acceptance=True,
            acceptance_blocked_reasons=tuple(
                _stable_unique(
                    (*acceptance_blocked, REASON_FULL_SUITE_UNAVAILABLE)
                )
            ),
            reason_codes=tuple(reasons),
            selection_fallback_mode=fallback_mode,
            selection_broader_required=selection_broader,
            selection_full_suite_required=selection_full,
            repository_id=repository_id,
            policy_id=policy_id,
            environment_id=environment_id,
            selector_identity=selector_identity,
            selected_duration_ms=selected_duration,
            full_suite_duration_ms=full_duration,
            evaluation_duration_ms=elapsed,
            selected_suite_status=selected_status,
            full_suite_status=full_status,
            not_measured_reasons=tuple(not_measured_reasons),
            evaluated=False,
        )

    if full_obs.suite_status is SuiteRunStatus.TIMEOUT:
        not_measured_reasons.append(REASON_FULL_SUITE_TIMEOUT)
        reasons.append(REASON_FULL_SUITE_TIMEOUT)
        elapsed = int((time.perf_counter() - started) * 1000)
        return TestSelectionEvaluation(
            fixture_id=fixture_obj.fixture_id,
            change_kind=fixture_obj.change_kind,
            equivalence_label=fixture_obj.equivalence_label,
            corpus_id=fixture_obj.corpus_id,
            measurement_status=MeasurementStatus.NOT_MEASURED,
            snapshot=snapshot,
            ground_truth_affected_tests=fixture_obj.ground_truth_affected_tests,
            selected_tests=selected_tests,
            full_suite_tests=full_obs.observed_tests,
            full_suite_failures=full_obs.failed_tests,
            snapshots_identical=(
                selected_obs is not None
                and selected_obs.snapshot.matches(full_obs.snapshot)
                and full_obs.snapshot.matches(snapshot)
            ),
            broader_suite_required_before_acceptance=True,
            acceptance_blocked_reasons=tuple(
                _stable_unique(
                    (*acceptance_blocked, REASON_FULL_SUITE_TIMEOUT)
                )
            ),
            reason_codes=tuple(reasons),
            selection_fallback_mode=fallback_mode,
            selection_broader_required=selection_broader,
            selection_full_suite_required=selection_full,
            repository_id=repository_id,
            policy_id=policy_id,
            environment_id=environment_id,
            selector_identity=selector_identity,
            selected_duration_ms=selected_duration,
            full_suite_duration_ms=full_obs.duration_ms,
            evaluation_duration_ms=elapsed,
            selected_suite_status=selected_status,
            full_suite_status=full_obs.suite_status.value,
            not_measured_reasons=tuple(not_measured_reasons),
            evaluated=False,
        )

    if full_obs.suite_status is SuiteRunStatus.UNAVAILABLE:
        not_measured_reasons.append(REASON_FULL_SUITE_UNAVAILABLE)
        reasons.append(REASON_FULL_SUITE_UNAVAILABLE)
        elapsed = int((time.perf_counter() - started) * 1000)
        return TestSelectionEvaluation(
            fixture_id=fixture_obj.fixture_id,
            change_kind=fixture_obj.change_kind,
            equivalence_label=fixture_obj.equivalence_label,
            corpus_id=fixture_obj.corpus_id,
            measurement_status=MeasurementStatus.NOT_MEASURED,
            snapshot=snapshot,
            ground_truth_affected_tests=fixture_obj.ground_truth_affected_tests,
            selected_tests=selected_tests,
            full_suite_tests=full_obs.observed_tests,
            snapshots_identical=False,
            broader_suite_required_before_acceptance=True,
            acceptance_blocked_reasons=tuple(
                _stable_unique(
                    (*acceptance_blocked, REASON_FULL_SUITE_UNAVAILABLE)
                )
            ),
            reason_codes=tuple(reasons),
            selection_fallback_mode=fallback_mode,
            selection_broader_required=selection_broader,
            selection_full_suite_required=selection_full,
            repository_id=repository_id,
            policy_id=policy_id,
            environment_id=environment_id,
            selector_identity=selector_identity,
            selected_duration_ms=selected_duration,
            full_suite_duration_ms=full_obs.duration_ms,
            evaluation_duration_ms=elapsed,
            selected_suite_status=selected_status,
            full_suite_status=full_obs.suite_status.value,
            not_measured_reasons=tuple(not_measured_reasons),
            evaluated=False,
        )

    # --- Snapshot identity equality ---
    snapshots_identical = full_obs.snapshot.matches(snapshot)
    if selected_obs is not None:
        snapshots_identical = (
            snapshots_identical
            and selected_obs.snapshot.matches(full_obs.snapshot)
            and selected_obs.snapshot.matches(snapshot)
        )
    else:
        # Without a selected observation we can still measure set membership
        # against ground truth using the forced/selected test list, but outcome
        # comparison requires selected observation for oracle FN.
        snapshots_identical = snapshots_identical

    if not snapshots_identical:
        reasons.append(REASON_SNAPSHOT_MISMATCH)
        inconclusive_reasons.append(REASON_SNAPSHOT_MISMATCH)
        acceptance_blocked.append(REASON_SNAPSHOT_MISMATCH)

    # --- Ground-truth set oracle ---
    gt = set(fixture_obj.ground_truth_affected_tests)
    selected_set = set(selected_tests)
    gt_fn = _unique_sorted(gt - selected_set)
    gt_fp = _unique_sorted(selected_set - gt)
    if gt_fn:
        reasons.append(REASON_GROUND_TRUTH_OMISSION)
    if gt_fp:
        reasons.append(REASON_SELECTED_OUTSIDE_GROUND_TRUTH)

    full_failures = set(full_obs.failed_tests)
    selected_failures: set[str] = set()
    if selected_obs is not None:
        selected_failures = set(selected_obs.failed_tests)

    # --- Full-suite failure oracle (separate) ---
    oracle_fn: list[str] = []
    inconclusive_tests: set[str] = set()

    for node in sorted(full_failures):
        if node not in selected_set:
            oracle_fn.append(node)
            reasons.append(REASON_ORACLE_FAILURE_NOT_OBSERVED)
            continue
        if selected_obs is None:
            # Cannot observe selected outcome — inconclusive for this node.
            inconclusive_tests.add(node)
            inconclusive_reasons.append(REASON_OUTCOME_DISCREPANCY)
            continue
        selected_outcome = selected_obs.test_outcomes.get(
            node, ObservedTestOutcome.NOT_OBSERVED
        )
        full_outcome = full_obs.test_outcomes.get(
            node, ObservedTestOutcome.NOT_OBSERVED
        )
        if selected_outcome is ObservedTestOutcome.FAILED:
            continue  # observed failure — not FN
        if selected_outcome in (
            ObservedTestOutcome.FLAKY,
            ObservedTestOutcome.ORDER_DEPENDENT,
        ) or full_outcome in (
            ObservedTestOutcome.FLAKY,
            ObservedTestOutcome.ORDER_DEPENDENT,
        ):
            inconclusive_tests.add(node)
            if selected_outcome is ObservedTestOutcome.FLAKY or (
                full_outcome is ObservedTestOutcome.FLAKY
            ):
                inconclusive_reasons.append(REASON_FLAKY_OUTCOME)
                reasons.append(REASON_FLAKY_OUTCOME)
            else:
                inconclusive_reasons.append(REASON_ORDER_DEPENDENT)
                reasons.append(REASON_ORDER_DEPENDENT)
            continue
        if selected_outcome is ObservedTestOutcome.PASSED:
            # Full failed, selected passed — outcome discrepancy / flaky class.
            inconclusive_tests.add(node)
            inconclusive_reasons.append(REASON_OUTCOME_DISCREPANCY)
            reasons.append(REASON_OUTCOME_DISCREPANCY)
            continue
        if selected_outcome is ObservedTestOutcome.NOT_OBSERVED:
            oracle_fn.append(node)
            reasons.append(REASON_ORACLE_FAILURE_NOT_OBSERVED)
            continue
        # Other non-fail selected outcomes with full fail → inconclusive.
        inconclusive_tests.add(node)
        inconclusive_reasons.append(REASON_OUTCOME_DISCREPANCY)
        reasons.append(REASON_OUTCOME_DISCREPANCY)

    # Scan selected/full outcome pairs for flaky/order discrepancies outside FN.
    if selected_obs is not None:
        shared = set(selected_obs.test_outcomes) | set(full_obs.test_outcomes)
        for node in sorted(shared):
            s_out = selected_obs.test_outcomes.get(
                node, ObservedTestOutcome.NOT_OBSERVED
            )
            f_out = full_obs.test_outcomes.get(
                node, ObservedTestOutcome.NOT_OBSERVED
            )
            if s_out is ObservedTestOutcome.FLAKY or f_out is ObservedTestOutcome.FLAKY:
                inconclusive_tests.add(node)
                inconclusive_reasons.append(REASON_FLAKY_OUTCOME)
                reasons.append(REASON_FLAKY_OUTCOME)
            elif (
                s_out is ObservedTestOutcome.ORDER_DEPENDENT
                or f_out is ObservedTestOutcome.ORDER_DEPENDENT
            ):
                inconclusive_tests.add(node)
                inconclusive_reasons.append(REASON_ORDER_DEPENDENT)
                reasons.append(REASON_ORDER_DEPENDENT)
            elif (
                s_out is ObservedTestOutcome.PASSED
                and f_out is ObservedTestOutcome.PASSED
            ):
                # Passing selected tests are never auto-classified as FP.
                reasons.append(REASON_PASSING_NOT_FALSE_POSITIVE)
            elif (
                node in selected_set
                and s_out != f_out
                and s_out
                not in (
                    ObservedTestOutcome.NOT_OBSERVED,
                    ObservedTestOutcome.SKIPPED,
                )
                and f_out
                not in (
                    ObservedTestOutcome.NOT_OBSERVED,
                    ObservedTestOutcome.SKIPPED,
                )
                and node not in full_failures
            ):
                inconclusive_tests.add(node)
                inconclusive_reasons.append(REASON_OUTCOME_DISCREPANCY)
                reasons.append(REASON_OUTCOME_DISCREPANCY)

        # Order-dependent: different observed orders for shared nodes with
        # differing outcomes already handled; also flag pure order swaps that
        # the recipe marks via suite reason codes.
        if (
            REASON_ORDER_DEPENDENT in selected_obs.reason_codes
            or REASON_ORDER_DEPENDENT in full_obs.reason_codes
            or "order_dependent" in selected_obs.reason_codes
            or "order_dependent" in full_obs.reason_codes
        ):
            reasons.append(REASON_ORDER_DEPENDENT)
            inconclusive_reasons.append(REASON_ORDER_DEPENDENT)

    oracle_fn_t = _unique_sorted(oracle_fn)
    # Aggregate FN: ground-truth omission ∪ oracle FN, excluding pure
    # inconclusive-only nodes that are not also GT omissions.
    aggregate_fn = _unique_sorted(
        set(gt_fn) | (set(oracle_fn_t) - inconclusive_tests)
    )
    # False positives remain ground-truth set membership only.
    aggregate_fp = gt_fp
    inconclusive_t = _unique_sorted(inconclusive_tests)

    # Measurement status.
    if not snapshots_identical:
        measurement = MeasurementStatus.INCONCLUSIVE
    elif inconclusive_t and not aggregate_fn and not aggregate_fp:
        # Pure inconclusive with no hard FP/FN still measured as inconclusive.
        measurement = MeasurementStatus.INCONCLUSIVE
    elif inconclusive_t and (aggregate_fn or aggregate_fp):
        # Mixed: still report measured with inconclusive subset.
        measurement = MeasurementStatus.MEASURED
    else:
        measurement = MeasurementStatus.MEASURED

    reasons.append(REASON_IDENTITIES_BOUND)
    reasons = list(_stable_unique(reasons))

    full_suite_tests = full_obs.observed_tests
    if not full_suite_tests and fixture_obj.all_tests:
        full_suite_tests = fixture_obj.all_tests

    elapsed = int((time.perf_counter() - started) * 1000)
    return TestSelectionEvaluation(
        fixture_id=fixture_obj.fixture_id,
        change_kind=fixture_obj.change_kind,
        equivalence_label=fixture_obj.equivalence_label,
        corpus_id=fixture_obj.corpus_id,
        measurement_status=measurement,
        snapshot=snapshot,
        ground_truth_affected_tests=fixture_obj.ground_truth_affected_tests,
        selected_tests=selected_tests,
        full_suite_tests=full_suite_tests,
        ground_truth_false_negatives=gt_fn,
        ground_truth_false_positives=gt_fp,
        full_suite_oracle_false_negatives=oracle_fn_t,
        full_suite_failures=_unique_sorted(full_failures),
        selected_failures=_unique_sorted(selected_failures),
        false_negative_tests=aggregate_fn,
        false_positive_tests=aggregate_fp,
        inconclusive_tests=inconclusive_t,
        snapshots_identical=snapshots_identical,
        broader_suite_required_before_acceptance=broader_required
        or selection_broader
        or selection_full,
        acceptance_blocked_reasons=tuple(_stable_unique(acceptance_blocked)),
        reason_codes=tuple(reasons),
        selection_fallback_mode=fallback_mode,
        selection_broader_required=selection_broader,
        selection_full_suite_required=selection_full,
        repository_id=repository_id,
        policy_id=policy_id,
        environment_id=environment_id,
        selector_identity=selector_identity,
        selected_duration_ms=(
            selected_obs.duration_ms if selected_obs is not None else 0
        ),
        full_suite_duration_ms=full_obs.duration_ms,
        evaluation_duration_ms=elapsed,
        selected_suite_status=(
            selected_obs.suite_status.value if selected_obs is not None else ""
        ),
        full_suite_status=full_obs.suite_status.value,
        not_measured_reasons=(),
        inconclusive_reasons=tuple(_stable_unique(inconclusive_reasons)),
        evaluated=True,
    )


def evaluate_controlled_fixture_corpus(
    fixtures: Sequence[ControlledSemanticFixture | Mapping[str, Any]] | None,
    *,
    corpus_id: str = CANONICAL_CORPUS_ID,
    corpus_present: bool = True,
    repository_id: str = "",
    policy_id: str = "",
    environment_id: str = "",
) -> SelectionCorpusEvaluationSummary:
    """Evaluate a sequence of controlled fixtures; empty/absent is not_measured."""

    started = time.perf_counter()
    reasons: list[str] = [REASON_TARGET_SUCCESS_NOT_ASSERTED]

    if not corpus_present:
        reasons.append(REASON_CORPUS_ABSENT)
        elapsed = int((time.perf_counter() - started) * 1000)
        return SelectionCorpusEvaluationSummary(
            corpus_id=corpus_id,
            corpus_present=False,
            measurement_status=MeasurementStatus.NOT_MEASURED,
            evaluated_count=0,
            measured_count=0,
            not_measured_count=0,
            inconclusive_count=0,
            total_false_negatives=None,
            total_false_positives=None,
            reason_codes=tuple(_stable_unique(reasons)),
            repository_id=repository_id,
            policy_id=policy_id,
            environment_id=environment_id,
            evaluation_duration_ms=elapsed,
        )

    items = list(fixtures or ())
    if not items:
        reasons.append(REASON_ZERO_EVALUATED)
        elapsed = int((time.perf_counter() - started) * 1000)
        return SelectionCorpusEvaluationSummary(
            corpus_id=corpus_id,
            corpus_present=True,
            measurement_status=MeasurementStatus.NOT_MEASURED,
            evaluated_count=0,
            measured_count=0,
            not_measured_count=0,
            inconclusive_count=0,
            total_false_negatives=None,
            total_false_positives=None,
            reason_codes=tuple(_stable_unique(reasons)),
            repository_id=repository_id,
            policy_id=policy_id,
            environment_id=environment_id,
            evaluation_duration_ms=elapsed,
        )

    evaluations: list[TestSelectionEvaluation] = []
    for raw in items:
        evaluations.append(
            compare_selected_with_full_suite(
                fixture=raw,
                corpus_available=corpus_present,
            )
        )

    measured = [
        item
        for item in evaluations
        if item.measurement_status is MeasurementStatus.MEASURED
    ]
    not_measured = [
        item
        for item in evaluations
        if item.measurement_status is MeasurementStatus.NOT_MEASURED
    ]
    inconclusive = [
        item
        for item in evaluations
        if item.measurement_status is MeasurementStatus.INCONCLUSIVE
    ]

    if not measured and not inconclusive:
        status = MeasurementStatus.NOT_MEASURED
        total_fn: int | None = None
        total_fp: int | None = None
        reasons.append(REASON_ZERO_EVALUATED)
    elif not measured and inconclusive:
        status = MeasurementStatus.INCONCLUSIVE
        total_fn = sum(len(item.false_negative_tests) for item in inconclusive)
        total_fp = sum(len(item.false_positive_tests) for item in inconclusive)
    else:
        status = MeasurementStatus.MEASURED
        total_fn = sum(
            len(item.false_negative_tests)
            for item in measured
            if item.false_negative_count is not None
        )
        total_fp = sum(
            len(item.false_positive_tests)
            for item in measured
            if item.false_positive_count is not None
        )

    fixture_ids = _unique_sorted(item.fixture_id for item in evaluations)
    equiv = _unique_sorted(
        item.equivalence_label
        for item in evaluations
        if item.equivalence_label
    )
    for item in evaluations:
        reasons.extend(item.reason_codes)

    # Bind first non-empty identities when caller did not supply them.
    if not repository_id:
        for item in evaluations:
            if item.repository_id:
                repository_id = item.repository_id
                break
    if not policy_id:
        for item in evaluations:
            if item.policy_id:
                policy_id = item.policy_id
                break
    if not environment_id:
        for item in evaluations:
            if item.environment_id:
                environment_id = item.environment_id
                break

    reasons.append(REASON_IDENTITIES_BOUND)
    elapsed = int((time.perf_counter() - started) * 1000)
    return SelectionCorpusEvaluationSummary(
        corpus_id=corpus_id,
        corpus_present=True,
        measurement_status=status,
        evaluated_count=len(evaluations),
        measured_count=len(measured),
        not_measured_count=len(not_measured),
        inconclusive_count=len(inconclusive),
        total_false_negatives=total_fn,
        total_false_positives=total_fp,
        fixture_ids=fixture_ids,
        equivalence_labels=equiv,
        evaluations=tuple(evaluations),
        reason_codes=tuple(_stable_unique(reasons)),
        repository_id=repository_id,
        policy_id=policy_id,
        environment_id=environment_id,
        evaluation_duration_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def default_fixture_root(
    repo_root: Path | str | None = None,
) -> Path:
    """Resolve ``test/fixtures/incremental_verification`` under the repo root."""

    if repo_root is not None:
        root = Path(repo_root)
        return (root / DEFAULT_FIXTURE_RELPATH).resolve()
    # Walk parents from this file and from cwd.
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        probe = candidate / DEFAULT_FIXTURE_RELPATH
        if probe.is_dir():
            return probe.resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        probe = candidate / DEFAULT_FIXTURE_RELPATH
        if probe.is_dir():
            return probe.resolve()
        if (candidate / "ipfs_accelerate_py").is_dir() and (
            candidate / "test"
        ).is_dir():
            return (candidate / DEFAULT_FIXTURE_RELPATH).resolve()
    return (cwd / DEFAULT_FIXTURE_RELPATH).resolve()


def load_controlled_fixtures(
    root: Path | str | None = None,
    *,
    require_present: bool = False,
) -> tuple[ControlledSemanticFixture, ...]:
    """Load controlled fixtures from a corpus manifest.

    Returns an empty tuple when the corpus is absent (unless
    ``require_present`` is true, which raises).
    """

    fixture_root = (
        default_fixture_root() if root is None else Path(root).resolve()
    )
    manifest_path = fixture_root / CORPUS_MANIFEST_NAME
    if not manifest_path.is_file():
        if require_present:
            raise EvaluationError(
                f"canonical semantic-capsule corpus absent at {manifest_path}"
            )
        return ()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if require_present:
            raise EvaluationError(
                f"failed to read corpus manifest: {exc}"
            ) from exc
        return ()
    if not isinstance(payload, Mapping):
        raise EvaluationError("corpus manifest must be a mapping")
    cases = payload.get("cases") or payload.get("fixtures") or ()
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise EvaluationError("corpus manifest cases must be a sequence")
    fixtures: list[ControlledSemanticFixture] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise EvaluationError(f"cases[{index}] must be a mapping")
        fixtures.append(ControlledSemanticFixture.from_value(case))
    fixtures.sort(key=lambda item: item.fixture_id)
    return tuple(fixtures)


def evaluate_default_corpus(
    root: Path | str | None = None,
    *,
    corpus_id: str = CANONICAL_CORPUS_ID,
) -> SelectionCorpusEvaluationSummary:
    """Load the default fixture corpus and evaluate it.

    Absent corpus / zero fixtures produce ``not_measured`` with null FN/FP
    totals (never zero).
    """

    fixture_root = (
        default_fixture_root() if root is None else Path(root).resolve()
    )
    manifest_path = fixture_root / CORPUS_MANIFEST_NAME
    corpus_present = manifest_path.is_file()
    fixtures = load_controlled_fixtures(fixture_root, require_present=False)
    return evaluate_controlled_fixture_corpus(
        fixtures,
        corpus_id=corpus_id,
        corpus_present=corpus_present,
    )


def suite_observation_from_outcomes(
    *,
    mode: SuiteMode | str,
    snapshot: EvaluationSnapshotIdentity | Mapping[str, Any],
    outcomes: Mapping[str, Any],
    suite_status: SuiteRunStatus | str = SuiteRunStatus.COMPLETED,
    duration_ms: int = 0,
    selector_identity: str = "",
    reason_codes: Sequence[str] | None = None,
    test_order: Sequence[str] | None = None,
) -> SuiteObservation:
    """Convenience constructor used by fixtures and tests."""

    return make_suite_observation(
        mode=mode,
        snapshot=snapshot,
        suite_status=suite_status,
        test_outcomes=outcomes,
        test_order=test_order,
        selector_identity=selector_identity,
        duration_ms=duration_ms,
        wall_time_ms=duration_ms,
        reason_codes=reason_codes,
    )


__all__ = [
    "CANONICAL_CORPUS_ID",
    "CONTROLLED_FIXTURE_SCHEMA",
    "CORPUS_EVALUATION_SUMMARY_INTERFACE",
    "CORPUS_EVALUATION_SUMMARY_SCHEMA",
    "CORPUS_MANIFEST_NAME",
    "DEFAULT_FIXTURE_RELPATH",
    "EVALUATION_SNAPSHOT_SCHEMA",
    "REASON_BROADER_REQUIRED",
    "REASON_CORPUS_ABSENT",
    "REASON_FLAKY_OUTCOME",
    "REASON_FULL_SUITE_TIMEOUT",
    "REASON_FULL_SUITE_UNAVAILABLE",
    "REASON_GROUND_TRUTH_OMISSION",
    "REASON_ORACLE_FAILURE_NOT_OBSERVED",
    "REASON_ORDER_DEPENDENT",
    "REASON_OUTCOME_DISCREPANCY",
    "REASON_PASSING_NOT_FALSE_POSITIVE",
    "REASON_SELECTED_OUTSIDE_GROUND_TRUTH",
    "REASON_SNAPSHOT_MISMATCH",
    "REASON_TARGET_SUCCESS_NOT_ASSERTED",
    "REASON_UNCERTAIN_SELECTOR",
    "REASON_VALIDATION_MAPPING_MISSING",
    "REASON_ZERO_EVALUATED",
    "SELECTION_EVALUATION_EVIDENCE",
    "SUITE_OBSERVATION_SCHEMA",
    "TEST_SELECTION_EVALUATION_INTERFACE",
    "TEST_SELECTION_EVALUATION_SCHEMA",
    "ControlledSemanticFixture",
    "EvaluationError",
    "EvaluationBoundsError",
    "EvaluationSnapshotIdentity",
    "FixtureChangeKind",
    "MeasurementStatus",
    "ObservedTestOutcome",
    "SelectionCorpusEvaluationSummary",
    "SuiteMode",
    "SuiteObservation",
    "SuiteRunStatus",
    "TestSelectionEvaluation",
    "compare_selected_with_full_suite",
    "default_fixture_root",
    "evaluate_controlled_fixture_corpus",
    "evaluate_default_corpus",
    "fresh_identical_observations",
    "load_controlled_fixtures",
    "make_suite_observation",
    "suite_observation_from_outcomes",
]
