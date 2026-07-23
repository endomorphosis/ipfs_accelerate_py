"""Semantic conformance suites and quarantine gates for prover routes.

The executable matrix establishes that a tool can run.  This module adds the
next trust boundary: it establishes whether a particular translator/profile
preserves the reviewed obligation.  Conformance receipts are deterministic,
bounded, property specific, and cannot upgrade a failed or incomplete route.

Legacy CEC/DCEC entry points and proof caches are registered as degraded by
default.  Discovery, smoke tests, or unrelated unit tests cannot release those
rules; only a complete semantic fixture receipt for the affected form can.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol

from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
)
from .logic_translation_validation import (
    LogicForm,
    SemanticDimension,
    SemanticInventory,
    TranslationArtifact,
    TranslationContract,
    TranslationValidationResult,
    validate_translation,
)


PROVER_CONFORMANCE_VERSION = 1
CONFORMANCE_FIXTURE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prover-conformance-fixture@1"
)
CONFORMANCE_FIXTURE_SET_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prover-conformance-fixture-set@1"
)
CONFORMANCE_CASE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prover-conformance-case@1"
)
CONFORMANCE_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prover-conformance-report@1"
)
CONFORMANCE_GATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prover-conformance-gate@1"
)
DEFAULT_MAX_CONFORMANCE_CASES = 256
DEFAULT_CONFORMANCE_TIMEOUT_SECONDS = 30.0


class ConformanceTestKind(str, Enum):
    ROUND_TRIP = "round_trip"
    DIFFERENTIAL = "differential"
    METAMORPHIC = "metamorphic"
    MUTATION = "mutation"
    NEGATIVE = "negative"


# Friendly aliases used by callers that call these "methods".
ConformanceMethod = ConformanceTestKind


class ConformanceStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMED_OUT = "timed_out"
    NOT_RUN = "not_run"


class RouteHealth(str, Enum):
    UNASSESSED = "unassessed"
    CONFORMANT = "conformant"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"


class QuarantineReason(str, Enum):
    API_DRIFT = "api_drift"
    TIMING_SENSITIVE_CACHE = "timing_sensitive_cache"
    TRANSLATION_NONCONFORMANCE = "translation_nonconformance"
    INCOMPLETE_FIXTURE_COVERAGE = "incomplete_fixture_coverage"
    STALE_FIXTURE_SET = "stale_fixture_set"
    TIMEOUT = "timeout"
    MALFORMED_RESULT = "malformed_result"


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return enum_type(str(getattr(value, "value", value)))
    except ValueError as exc:
        raise ContractValidationError(f"{name} is unsupported") from exc


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ContractValidationError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise ContractValidationError(f"{name} must not be empty")
    return value


def _strings(
    values: Iterable[Any] | None, name: str, *, required: bool = False
) -> tuple[str, ...]:
    if values is None:
        result: tuple[str, ...] = ()
    elif isinstance(values, (str, bytes, bytearray)):
        raise ContractValidationError(f"{name} must be a sequence")
    else:
        result = tuple(sorted({_text(item, name) for item in values}))
    if required and not result:
        raise ContractValidationError(f"{name} must not be empty")
    return result


def _strict_mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ContractValidationError(f"{name} must be an object with string keys")
    result = _canonical_value(dict(value))
    if not isinstance(result, dict):  # pragma: no cover
        raise ContractValidationError(f"{name} must be an object")
    return result


def _timestamp(clock: Callable[[], float]) -> str:
    return (
        datetime.fromtimestamp(clock(), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ContractValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


def _claimed_identity(
    payload: Mapping[str, Any], actual: str, noun: str
) -> None:
    claimed = payload.get("content_id") or payload.get("identity")
    if claimed and claimed != actual:
        raise ContractValidationError(f"{noun} content identity does not match payload")


@dataclass(frozen=True)
class ConformanceFixture(CanonicalContract):
    """One reviewed semantic test vector.

    ``source_text`` is inert fixture data.  Authoritative implementations must
    build ``source_inventory`` from their parsed AST rather than token-search
    this string.
    """

    SCHEMA = CONFORMANCE_FIXTURE_SCHEMA

    fixture_id: str
    kind: ConformanceTestKind
    source_form: LogicForm
    source_text: str
    source_inventory: SemanticInventory
    semantic_profile_id: str
    required_dimensions: tuple[SemanticDimension, ...]
    expected_outcome: str
    mutation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("fixture_id", "source_text", "semantic_profile_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "kind", _enum(self.kind, ConformanceTestKind, "kind")
        )
        object.__setattr__(
            self, "source_form", _enum(self.source_form, LogicForm, "source_form")
        )
        inventory = self.source_inventory
        if isinstance(inventory, Mapping):
            inventory = SemanticInventory.from_dict(inventory)
        if not isinstance(inventory, SemanticInventory):
            raise ContractValidationError(
                "source_inventory must be a SemanticInventory"
            )
        object.__setattr__(self, "source_inventory", inventory)
        dimensions = tuple(
            sorted(
                {
                    _enum(item, SemanticDimension, "required_dimensions")
                    for item in self.required_dimensions
                },
                key=lambda item: item.value,
            )
        )
        if not dimensions:
            raise ContractValidationError("required_dimensions must not be empty")
        object.__setattr__(self, "required_dimensions", dimensions)
        object.__setattr__(
            self, "expected_outcome", _text(self.expected_outcome, "expected_outcome")
        )
        object.__setattr__(self, "mutation", _strict_mapping(self.mutation, "mutation"))
        object.__setattr__(self, "metadata", _strict_mapping(self.metadata, "metadata"))
        if self.kind is ConformanceTestKind.MUTATION and not self.mutation:
            raise ContractValidationError("mutation fixtures require a mutation")

    def _payload(self) -> dict[str, Any]:
        return {
            "conformance_version": PROVER_CONFORMANCE_VERSION,
            "fixture_id": self.fixture_id,
            "kind": self.kind,
            "source_form": self.source_form,
            "source_text": self.source_text,
            "source_inventory": self.source_inventory,
            "semantic_profile_id": self.semantic_profile_id,
            "required_dimensions": tuple(
                value.value for value in self.required_dimensions
            ),
            "expected_outcome": self.expected_outcome,
            "mutation": self.mutation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformanceFixture":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("conformance fixture must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            fixture_id=payload.get("fixture_id", ""),
            kind=payload.get("kind", ""),
            source_form=payload.get("source_form", ""),
            source_text=payload.get("source_text", ""),
            source_inventory=payload.get("source_inventory") or {},
            semantic_profile_id=payload.get("semantic_profile_id", ""),
            required_dimensions=tuple(payload.get("required_dimensions") or ()),
            expected_outcome=payload.get("expected_outcome", ""),
            mutation=payload.get("mutation") or {},
            metadata=payload.get("metadata") or {},
        )
        _claimed_identity(payload, result.content_id, "conformance fixture")
        return result


@dataclass(frozen=True)
class ConformanceFixtureSet(CanonicalContract):
    """Versioned suite with coverage introspection."""

    SCHEMA = CONFORMANCE_FIXTURE_SET_SCHEMA

    name: str
    version: str
    fixtures: tuple[ConformanceFixture, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        fixtures = tuple(self.fixtures)
        if not fixtures or any(not isinstance(item, ConformanceFixture) for item in fixtures):
            raise ContractValidationError(
                "fixture set must contain ConformanceFixture values"
            )
        ids = [item.fixture_id for item in fixtures]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("fixture ids must be unique")
        object.__setattr__(
            self, "fixtures", tuple(sorted(fixtures, key=lambda item: item.fixture_id))
        )

    @property
    def fixture_set_id(self) -> str:
        return self.content_id

    @property
    def forms(self) -> frozenset[LogicForm]:
        return frozenset(item.source_form for item in self.fixtures)

    @property
    def kinds(self) -> frozenset[ConformanceTestKind]:
        return frozenset(item.kind for item in self.fixtures)

    def covers(
        self,
        forms: Iterable[LogicForm | str],
        kinds: Iterable[ConformanceTestKind | str],
        *,
        cross_product: bool = True,
    ) -> bool:
        expected_forms = {_enum(item, LogicForm, "form") for item in forms}
        expected_kinds = {
            _enum(item, ConformanceTestKind, "kind") for item in kinds
        }
        observed = {(item.source_form, item.kind) for item in self.fixtures}
        if cross_product:
            return all(
                (form, kind) in observed
                for form in expected_forms
                for kind in expected_kinds
            )
        return expected_forms <= self.forms and expected_kinds <= self.kinds

    def for_form(self, form: LogicForm | str) -> tuple[ConformanceFixture, ...]:
        selected = _enum(form, LogicForm, "form")
        return tuple(item for item in self.fixtures if item.source_form is selected)

    def _payload(self) -> dict[str, Any]:
        return {
            "conformance_version": PROVER_CONFORMANCE_VERSION,
            "name": self.name,
            "version": self.version,
            "fixtures": self.fixtures,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformanceFixtureSet":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("fixture set must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            name=payload.get("name", ""),
            version=payload.get("version", ""),
            fixtures=tuple(
                item
                if isinstance(item, ConformanceFixture)
                else ConformanceFixture.from_dict(item)
                for item in (payload.get("fixtures") or ())
            ),
        )
        _claimed_identity(payload, result.content_id, "fixture set")
        return result


@dataclass(frozen=True)
class ConformanceObservation:
    """Normalized output from an injected translator/solver fixture runner."""

    artifact: Optional[TranslationArtifact] = None
    round_trip_inventory: Optional[SemanticInventory] = None
    candidate_outcome: Optional[str] = None
    oracle_outcome: Optional[str] = None
    relation_preserved: Optional[bool] = None
    mutation_detected: Optional[bool] = None
    rejected: Optional[bool] = None
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact is not None and not isinstance(
            self.artifact, TranslationArtifact
        ):
            raise ContractValidationError(
                "observation artifact must be a TranslationArtifact"
            )
        inventory = self.round_trip_inventory
        if isinstance(inventory, Mapping):
            inventory = SemanticInventory.from_dict(inventory)
            object.__setattr__(self, "round_trip_inventory", inventory)
        if inventory is not None and not isinstance(inventory, SemanticInventory):
            raise ContractValidationError(
                "round_trip_inventory must be a SemanticInventory"
            )
        object.__setattr__(
            self, "candidate_outcome", _text(self.candidate_outcome, "candidate_outcome", required=False)
        )
        object.__setattr__(
            self, "oracle_outcome", _text(self.oracle_outcome, "oracle_outcome", required=False)
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason", required=False))
        object.__setattr__(self, "metadata", _strict_mapping(self.metadata, "metadata"))
        for name in ("relation_preserved", "mutation_detected", "rejected"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ContractValidationError(f"{name} must be boolean or null")

    @classmethod
    def from_value(cls, value: Any) -> "ConformanceObservation":
        if isinstance(value, cls):
            return value
        if isinstance(value, TranslationArtifact):
            return cls(artifact=value)
        if not isinstance(value, Mapping):
            raise ContractValidationError(
                "fixture runner must return an observation, artifact, or object"
            )
        artifact = value.get("artifact")
        if isinstance(artifact, Mapping):
            artifact = TranslationArtifact.from_dict(artifact)
        return cls(
            artifact=artifact,
            round_trip_inventory=value.get("round_trip_inventory"),
            candidate_outcome=value.get("candidate_outcome"),
            oracle_outcome=value.get("oracle_outcome"),
            relation_preserved=value.get("relation_preserved"),
            mutation_detected=value.get("mutation_detected"),
            rejected=value.get("rejected"),
            reason=value.get("reason", ""),
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class ConformanceCaseResult(CanonicalContract):
    SCHEMA = CONFORMANCE_CASE_SCHEMA

    fixture_id: str
    kind: ConformanceTestKind
    source_form: LogicForm
    status: ConformanceStatus
    reason: str
    duration_ms: int
    validation: Optional[TranslationValidationResult] = None
    observation_identity: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_id", _text(self.fixture_id, "fixture_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ConformanceTestKind, "kind")
        )
        object.__setattr__(
            self, "source_form", _enum(self.source_form, LogicForm, "source_form")
        )
        object.__setattr__(
            self, "status", _enum(self.status, ConformanceStatus, "status")
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise ContractValidationError("duration_ms must be non-negative")

    @property
    def passed(self) -> bool:
        return self.status is ConformanceStatus.PASSED

    def _payload(self) -> dict[str, Any]:
        return {
            "conformance_version": PROVER_CONFORMANCE_VERSION,
            "fixture_id": self.fixture_id,
            "kind": self.kind,
            "source_form": self.source_form,
            "status": self.status,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
            "validation": self.validation,
            "observation_identity": self.observation_identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformanceCaseResult":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("conformance case must be an object")
        _schema(payload, cls.SCHEMA)
        validation = payload.get("validation")
        if isinstance(validation, Mapping):
            validation = TranslationValidationResult.from_dict(validation)
        result = cls(
            fixture_id=payload.get("fixture_id", ""),
            kind=payload.get("kind", ""),
            source_form=payload.get("source_form", ""),
            status=payload.get("status", ""),
            reason=payload.get("reason", ""),
            duration_ms=payload.get("duration_ms", -1),
            validation=validation,
            observation_identity=payload.get("observation_identity"),
        )
        _claimed_identity(payload, result.content_id, "conformance case")
        return result


@dataclass(frozen=True)
class ConformanceReport(CanonicalContract):
    """Bounded receipt for a translator/profile fixture run."""

    SCHEMA = CONFORMANCE_REPORT_SCHEMA

    prover_id: str
    path_id: str
    contract_identity: str
    fixture_set_id: str
    started_at: str
    duration_ms: int
    timeout_seconds: float
    max_cases: int
    cases: tuple[ConformanceCaseResult, ...]
    complete: bool
    required_forms: tuple[LogicForm, ...]
    required_kinds: tuple[ConformanceTestKind, ...]
    permitted_assurance: AssuranceLevel

    def __post_init__(self) -> None:
        for name in (
            "prover_id",
            "path_id",
            "contract_identity",
            "fixture_set_id",
            "started_at",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
            or isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
            or isinstance(self.max_cases, bool)
            or not isinstance(self.max_cases, int)
            or self.max_cases < 1
        ):
            raise ContractValidationError("report resource bounds are invalid")
        if not isinstance(self.complete, bool):
            raise ContractValidationError("complete must be boolean")
        object.__setattr__(
            self,
            "required_forms",
            tuple(
                sorted(
                    {_enum(item, LogicForm, "required_forms") for item in self.required_forms},
                    key=lambda item: item.value,
                )
            ),
        )
        object.__setattr__(
            self,
            "required_kinds",
            tuple(
                sorted(
                    {
                        _enum(item, ConformanceTestKind, "required_kinds")
                        for item in self.required_kinds
                    },
                    key=lambda item: item.value,
                )
            ),
        )
        object.__setattr__(
            self,
            "permitted_assurance",
            _enum(self.permitted_assurance, AssuranceLevel, "permitted_assurance"),
        )
        if not self.complete and self.permitted_assurance is not AssuranceLevel.UNVERIFIED:
            raise ContractValidationError(
                "incomplete conformance cannot permit assurance"
            )
        if any(not case.passed for case in self.cases) and (
            self.permitted_assurance is not AssuranceLevel.UNVERIFIED
        ):
            raise ContractValidationError(
                "failed conformance cases cannot permit assurance"
            )

    @property
    def passed(self) -> bool:
        return (
            self.complete
            and bool(self.cases)
            and all(item.status is ConformanceStatus.PASSED for item in self.cases)
        )

    @property
    def report_id(self) -> str:
        return self.content_id

    @property
    def statuses(self) -> Mapping[str, int]:
        return {
            status.value: sum(item.status is status for item in self.cases)
            for status in ConformanceStatus
        }

    def covers(
        self,
        forms: Iterable[LogicForm | str],
        kinds: Iterable[ConformanceTestKind | str],
    ) -> bool:
        expected = {
            (_enum(form, LogicForm, "form"), _enum(kind, ConformanceTestKind, "kind"))
            for form in forms
            for kind in kinds
        }
        passed = {
            (case.source_form, case.kind) for case in self.cases if case.passed
        }
        return expected <= passed

    def _payload(self) -> dict[str, Any]:
        return {
            "conformance_version": PROVER_CONFORMANCE_VERSION,
            "prover_id": self.prover_id,
            "path_id": self.path_id,
            "contract_identity": self.contract_identity,
            "fixture_set_id": self.fixture_set_id,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "timeout_milliseconds": int(round(self.timeout_seconds * 1000)),
            "max_cases": self.max_cases,
            "cases": self.cases,
            "complete": self.complete,
            "required_forms": self.required_forms,
            "required_kinds": self.required_kinds,
            "permitted_assurance": self.permitted_assurance,
            "passed": self.passed,
            "statuses": self.statuses,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformanceReport":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("conformance report must be an object")
        _schema(payload, cls.SCHEMA)
        timeout_ms = payload.get("timeout_milliseconds")
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms < 1
        ):
            raise ContractValidationError(
                "timeout_milliseconds must be a positive integer"
            )
        result = cls(
            prover_id=payload.get("prover_id", ""),
            path_id=payload.get("path_id", ""),
            contract_identity=payload.get("contract_identity", ""),
            fixture_set_id=payload.get("fixture_set_id", ""),
            started_at=payload.get("started_at", ""),
            duration_ms=payload.get("duration_ms", -1),
            timeout_seconds=timeout_ms / 1000,
            max_cases=payload.get("max_cases", 0),
            cases=tuple(
                item
                if isinstance(item, ConformanceCaseResult)
                else ConformanceCaseResult.from_dict(item)
                for item in (payload.get("cases") or ())
            ),
            complete=payload.get("complete", False),
            required_forms=tuple(payload.get("required_forms") or ()),
            required_kinds=tuple(payload.get("required_kinds") or ()),
            permitted_assurance=payload.get(
                "permitted_assurance", AssuranceLevel.UNVERIFIED
            ),
        )
        _claimed_identity(payload, result.content_id, "conformance report")
        return result


class FixtureRunner(Protocol):
    def __call__(
        self, fixture: ConformanceFixture, contract: TranslationContract
    ) -> ConformanceObservation | TranslationArtifact | Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class ConformanceRunConfig:
    timeout_seconds: float = DEFAULT_CONFORMANCE_TIMEOUT_SECONDS
    max_cases: int = DEFAULT_MAX_CONFORMANCE_CASES
    require_cross_product: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ContractValidationError("timeout_seconds must be positive")
        if isinstance(self.max_cases, bool) or self.max_cases < 1:
            raise ContractValidationError("max_cases must be positive")


class ProverConformanceRunner:
    """Execute deterministic fixture callbacks under suite-level bounds."""

    def __init__(
        self,
        fixture_set: ConformanceFixtureSet,
        *,
        config: ConformanceRunConfig | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(fixture_set, ConformanceFixtureSet):
            raise ContractValidationError(
                "fixture_set must be a ConformanceFixtureSet"
            )
        self.fixture_set = fixture_set
        self.config = config or ConformanceRunConfig()
        self._monotonic = monotonic
        self._wall_clock = wall_clock

    @staticmethod
    def _inventory_matches(
        expected: SemanticInventory,
        actual: SemanticInventory,
        dimensions: Iterable[SemanticDimension],
    ) -> bool:
        return all(
            expected.values(dimension) == actual.values(dimension)
            for dimension in dimensions
        )

    def _evaluate(
        self,
        fixture: ConformanceFixture,
        contract: TranslationContract,
        observation: ConformanceObservation,
    ) -> tuple[bool, str, Optional[TranslationValidationResult]]:
        if observation.artifact is not None:
            if observation.artifact.source_identity != fixture.content_id:
                return False, "artifact is bound to a different source fixture", None
            if not self._inventory_matches(
                fixture.source_inventory,
                observation.artifact.source_inventory,
                fixture.required_dimensions,
            ):
                return False, "artifact source inventory differs from fixture", None
        validation = (
            validate_translation(contract, observation.artifact)
            if observation.artifact is not None
            else None
        )
        if validation is not None and not validation.conformant:
            codes = ",".join(issue.code.value for issue in validation.issues)
            return False, f"translation validation failed: {codes}", validation

        if fixture.kind is ConformanceTestKind.ROUND_TRIP:
            if validation is None or observation.round_trip_inventory is None:
                return False, "round-trip inventory is missing", validation
            passed = self._inventory_matches(
                fixture.source_inventory,
                observation.round_trip_inventory,
                fixture.required_dimensions,
            )
            return (
                passed,
                "round trip preserved semantic inventory"
                if passed
                else "round trip changed semantic inventory",
                validation,
            )
        if fixture.kind is ConformanceTestKind.DIFFERENTIAL:
            passed = (
                bool(observation.candidate_outcome)
                and observation.candidate_outcome == observation.oracle_outcome
                and observation.candidate_outcome == fixture.expected_outcome
            )
            return (
                passed,
                "candidate agreed with reference evaluator"
                if passed
                else "candidate disagreed with reference evaluator",
                validation,
            )
        if fixture.kind is ConformanceTestKind.METAMORPHIC:
            passed = observation.relation_preserved is True
            return (
                passed,
                "metamorphic relation preserved"
                if passed
                else "metamorphic relation was not preserved",
                validation,
            )
        if fixture.kind is ConformanceTestKind.MUTATION:
            passed = observation.mutation_detected is True
            return (
                passed,
                "semantic mutation was detected"
                if passed
                else "semantic mutation escaped detection",
                validation,
            )
        passed = observation.rejected is True
        return (
            passed,
            "negative fixture was rejected"
            if passed
            else "negative fixture was accepted",
            validation,
        )

    @staticmethod
    def _bounded_call(
        fixture_runner: FixtureRunner,
        fixture: ConformanceFixture,
        contract: TranslationContract,
        timeout_seconds: float,
    ) -> tuple[bool, Any]:
        """Run an in-process adapter without allowing it to block the suite.

        Production provider adapters should still use the repository's bounded
        subprocess protocol.  The daemon thread is a final containment boundary
        for injected in-process test/reference adapters; it is intentionally
        not reusable after a timeout.
        """

        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                result_queue.put((True, fixture_runner(fixture, contract)))
            except BaseException as exc:
                result_queue.put((False, exc))

        worker = threading.Thread(
            target=invoke,
            name=f"prover-conformance-{fixture.fixture_id}",
            daemon=True,
        )
        worker.start()
        worker.join(max(0.0, timeout_seconds))
        if worker.is_alive():
            return False, TimeoutError("fixture exceeded its remaining time budget")
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return False, RuntimeError("fixture runner returned no observation")

    def run(
        self,
        *,
        prover_id: str,
        path_id: str,
        contract: TranslationContract,
        fixture_runner: FixtureRunner,
        forms: Iterable[LogicForm | str] | None = None,
        kinds: Iterable[ConformanceTestKind | str] | None = None,
    ) -> ConformanceReport:
        if not isinstance(contract, TranslationContract):
            raise ContractValidationError("contract must be a TranslationContract")
        if not callable(fixture_runner):
            raise ContractValidationError("fixture_runner must be callable")
        selected_forms = tuple(
            sorted(
                {
                    _enum(item, LogicForm, "forms")
                    for item in (forms or (contract.source_form,))
                },
                key=lambda item: item.value,
            )
        )
        selected_kinds = tuple(
            sorted(
                {
                    _enum(item, ConformanceTestKind, "kinds")
                    for item in (kinds or tuple(ConformanceTestKind))
                },
                key=lambda item: item.value,
            )
        )
        if any(item is not contract.source_form for item in selected_forms):
            raise ContractValidationError(
                "all selected fixture forms must match the contract source_form"
            )
        fixtures = tuple(
            item
            for item in self.fixture_set.fixtures
            if item.source_form in selected_forms and item.kind in selected_kinds
        )
        suite_covered = self.fixture_set.covers(
            selected_forms,
            selected_kinds,
            cross_product=self.config.require_cross_product,
        )
        started_at = _timestamp(self._wall_clock)
        start = self._monotonic()
        deadline = start + self.config.timeout_seconds
        results: list[ConformanceCaseResult] = []
        for fixture in fixtures[: self.config.max_cases]:
            case_start = self._monotonic()
            if case_start >= deadline:
                results.append(
                    ConformanceCaseResult(
                        fixture_id=fixture.fixture_id,
                        kind=fixture.kind,
                        source_form=fixture.source_form,
                        status=ConformanceStatus.TIMED_OUT,
                        reason="suite time budget exhausted",
                        duration_ms=0,
                    )
                )
                continue
            try:
                fixture_contract = replace(
                    contract,
                    contract_id=f"{contract.contract_id}/{fixture.fixture_id}",
                    source_identity=fixture.content_id,
                )
                call_succeeded, raw = self._bounded_call(
                    fixture_runner,
                    fixture,
                    fixture_contract,
                    max(0.0, deadline - self._monotonic()),
                )
                if not call_succeeded:
                    raise raw
                observation = ConformanceObservation.from_value(raw)
                passed, reason, validation = self._evaluate(
                    fixture, fixture_contract, observation
                )
                status = (
                    ConformanceStatus.PASSED
                    if passed
                    else ConformanceStatus.FAILED
                )
                observation_identity = _digest(
                    {
                        "artifact": (
                            observation.artifact.to_dict()
                            if observation.artifact is not None
                            else None
                        ),
                        "round_trip_inventory": (
                            observation.round_trip_inventory.to_dict()
                            if observation.round_trip_inventory is not None
                            else None
                        ),
                        "candidate_outcome": observation.candidate_outcome,
                        "oracle_outcome": observation.oracle_outcome,
                        "relation_preserved": observation.relation_preserved,
                        "mutation_detected": observation.mutation_detected,
                        "rejected": observation.rejected,
                        "reason": observation.reason,
                        "metadata": observation.metadata,
                    }
                )
            except BaseException as exc:
                status = (
                    ConformanceStatus.TIMED_OUT
                    if isinstance(exc, TimeoutError)
                    else ConformanceStatus.ERROR
                )
                reason = f"fixture runner {type(exc).__name__}: {exc}"
                validation = None
                observation_identity = None
            duration_ms = max(0, int((self._monotonic() - case_start) * 1000))
            if self._monotonic() > deadline and status is ConformanceStatus.PASSED:
                status = ConformanceStatus.TIMED_OUT
                reason = "fixture completed after suite deadline"
                validation = None
            results.append(
                ConformanceCaseResult(
                    fixture_id=fixture.fixture_id,
                    kind=fixture.kind,
                    source_form=fixture.source_form,
                    status=status,
                    reason=reason,
                    duration_ms=duration_ms,
                    validation=validation,
                    observation_identity=observation_identity,
                )
            )

        expected_count = len(fixtures)
        complete = (
            suite_covered
            and expected_count <= self.config.max_cases
            and len(results) == expected_count
            and all(
                item.status
                in (ConformanceStatus.PASSED, ConformanceStatus.FAILED)
                for item in results
            )
            and self._monotonic() <= deadline
            and contract.fixture_set_id == self.fixture_set.fixture_set_id
        )
        return ConformanceReport(
            prover_id=prover_id,
            path_id=path_id,
            contract_identity=contract.content_id,
            fixture_set_id=self.fixture_set.fixture_set_id,
            started_at=started_at,
            duration_ms=max(0, int((self._monotonic() - start) * 1000)),
            timeout_seconds=self.config.timeout_seconds,
            max_cases=self.config.max_cases,
            cases=tuple(results),
            complete=complete,
            required_forms=selected_forms,
            required_kinds=selected_kinds,
            permitted_assurance=(
                contract.maximum_assurance
                if complete and all(item.passed for item in results)
                else AssuranceLevel.UNVERIFIED
            ),
        )


@dataclass(frozen=True)
class QuarantineRule:
    path_id: str
    reason: QuarantineReason
    detail: str
    required_forms: tuple[LogicForm, ...]
    required_kinds: tuple[ConformanceTestKind, ...] = tuple(ConformanceTestKind)
    degraded: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", _text(self.path_id, "path_id"))
        object.__setattr__(
            self, "reason", _enum(self.reason, QuarantineReason, "reason")
        )
        object.__setattr__(self, "detail", _text(self.detail, "detail"))
        object.__setattr__(
            self,
            "required_forms",
            tuple(
                sorted(
                    {_enum(item, LogicForm, "required_forms") for item in self.required_forms},
                    key=lambda item: item.value,
                )
            ),
        )
        object.__setattr__(
            self,
            "required_kinds",
            tuple(
                sorted(
                    {
                        _enum(item, ConformanceTestKind, "required_kinds")
                        for item in self.required_kinds
                    },
                    key=lambda item: item.value,
                )
            ),
        )
        if not self.required_forms or not self.required_kinds:
            raise ContractValidationError(
                "quarantine release requires forms and test kinds"
            )
        if not isinstance(self.degraded, bool):
            raise ContractValidationError("degraded must be boolean")


@dataclass(frozen=True)
class ConformanceGateDecision(CanonicalContract):
    SCHEMA = CONFORMANCE_GATE_SCHEMA

    path_id: str
    health: RouteHealth
    promotion_allowed: bool
    maximum_assurance: AssuranceLevel
    reasons: tuple[QuarantineReason, ...]
    report_id: Optional[str] = None
    retained_authorities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", _text(self.path_id, "path_id"))
        object.__setattr__(
            self, "health", _enum(self.health, RouteHealth, "health")
        )
        object.__setattr__(
            self,
            "maximum_assurance",
            _enum(self.maximum_assurance, AssuranceLevel, "maximum_assurance"),
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(
                sorted(
                    {_enum(item, QuarantineReason, "reasons") for item in self.reasons},
                    key=lambda item: item.value,
                )
            ),
        )
        object.__setattr__(
            self,
            "retained_authorities",
            _strings(self.retained_authorities, "retained_authorities"),
        )
        if self.report_id is not None:
            object.__setattr__(
                self, "report_id", _text(self.report_id, "report_id")
            )
        if not isinstance(self.promotion_allowed, bool):
            raise ContractValidationError("promotion_allowed must be boolean")
        if not self.promotion_allowed and (
            self.maximum_assurance is not AssuranceLevel.UNVERIFIED
            or self.retained_authorities
        ):
            raise ContractValidationError(
                "blocked paths cannot retain assurance or authorities"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "conformance_version": PROVER_CONFORMANCE_VERSION,
            "path_id": self.path_id,
            "health": self.health,
            "promotion_allowed": self.promotion_allowed,
            "maximum_assurance": self.maximum_assurance,
            "reasons": self.reasons,
            "report_id": self.report_id,
            "retained_authorities": self.retained_authorities,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformanceGateDecision":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("conformance gate must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            path_id=payload.get("path_id", ""),
            health=payload.get("health", ""),
            promotion_allowed=payload.get("promotion_allowed", False),
            maximum_assurance=payload.get(
                "maximum_assurance", AssuranceLevel.UNVERIFIED
            ),
            reasons=tuple(payload.get("reasons") or ()),
            report_id=payload.get("report_id"),
            retained_authorities=tuple(
                payload.get("retained_authorities") or ()
            ),
        )
        _claimed_identity(payload, result.content_id, "conformance gate")
        return result


LEGACY_CEC_DCEC_WRAPPER = "ipfs_datasets_py.logic.CEC.dcec_wrapper"
LEGACY_CEC_DEONTIC_API = "ipfs_datasets_py.logic.CEC.native.deontic"
LEGACY_CEC_PROOF_CACHE = "ipfs_datasets_py.logic.CEC.native.cec_proof_cache"
LEGACY_TDFOL_PROOF_CACHE = "ipfs_datasets_py.logic.TDFOL.tdfol_proof_cache"
LEGACY_DCEC_TO_TDFOL_TRANSLATOR = (
    "ipfs_datasets_py.logic.TDFOL.tdfol_converter.DCECToTDFOLConverter"
)
LEGACY_TDFOL_TO_FOL_TRANSLATOR = (
    "ipfs_datasets_py.logic.TDFOL.tdfol_converter.TDFOLToFOLConverter"
)

DEFAULT_QUARANTINE_RULES: tuple[QuarantineRule, ...] = (
    QuarantineRule(
        path_id=LEGACY_CEC_DCEC_WRAPPER,
        reason=QuarantineReason.API_DRIFT,
        detail=(
            "legacy wrapper mixes addStatement/printStatement with the native "
            "add_statement API and has no typed semantic receipt"
        ),
        required_forms=(LogicForm.DCEC,),
    ),
    QuarantineRule(
        path_id=LEGACY_CEC_DEONTIC_API,
        reason=QuarantineReason.API_DRIFT,
        detail=(
            "CEC deontic operator and constructor variants require exact "
            "actor, time, scope, and modality fixtures"
        ),
        required_forms=(LogicForm.DCEC,),
    ),
    QuarantineRule(
        path_id=LEGACY_CEC_PROOF_CACHE,
        reason=QuarantineReason.TIMING_SENSITIVE_CACHE,
        detail=(
            "legacy cache reconstructs timing-bearing proof attempts and must "
            "prove premise/bound/toolchain key separation deterministically"
        ),
        required_forms=(LogicForm.DCEC,),
    ),
    QuarantineRule(
        path_id=LEGACY_TDFOL_PROOF_CACHE,
        reason=QuarantineReason.TIMING_SENSITIVE_CACHE,
        detail=(
            "deprecated global TDFOL cache remains degraded until semantic "
            "keys and timing-independent repeated outcomes pass fixtures"
        ),
        required_forms=(LogicForm.TDFOL,),
    ),
    QuarantineRule(
        path_id=LEGACY_DCEC_TO_TDFOL_TRANSLATOR,
        reason=QuarantineReason.API_DRIFT,
        detail=(
            "legacy attribute-based DCEC conversion accepts multiple formula "
            "APIs and must demonstrate exact actor, quantifier, and modality mapping"
        ),
        required_forms=(LogicForm.DCEC,),
    ),
    QuarantineRule(
        path_id=LEGACY_TDFOL_TO_FOL_TRANSLATOR,
        reason=QuarantineReason.TRANSLATION_NONCONFORMANCE,
        detail=(
            "legacy FOL projection strips deontic and temporal operators and "
            "approximates binary temporal formulas; it cannot be an exact route"
        ),
        required_forms=(LogicForm.TDFOL,),
    ),
)


class ProverQuarantineRegistry:
    """Evaluate immutable built-in and caller-supplied quarantine rules."""

    def __init__(
        self, rules: Iterable[QuarantineRule] = DEFAULT_QUARANTINE_RULES
    ) -> None:
        values = tuple(rules)
        if any(not isinstance(item, QuarantineRule) for item in values):
            raise ContractValidationError("rules must be QuarantineRule values")
        if len({item.path_id for item in values}) != len(values):
            raise ContractValidationError("quarantine path ids must be unique")
        self._rules = {item.path_id: item for item in values}

    @property
    def rules(self) -> tuple[QuarantineRule, ...]:
        return tuple(self._rules[key] for key in sorted(self._rules))

    def rule(self, path_id: str) -> Optional[QuarantineRule]:
        return self._rules.get(path_id)

    def assess(
        self,
        path_id: str,
        report: Optional[ConformanceReport] = None,
        *,
        authoritative_for: Iterable[str] = (),
    ) -> ConformanceGateDecision:
        path_id = _text(path_id, "path_id")
        rule = self.rule(path_id)
        if report is not None and report.path_id != path_id:
            return ConformanceGateDecision(
                path_id=path_id,
                health=RouteHealth.QUARANTINED,
                promotion_allowed=False,
                maximum_assurance=AssuranceLevel.UNVERIFIED,
                reasons=(QuarantineReason.MALFORMED_RESULT,),
            )

        released = bool(
            report
            and report.passed
            and report.covers(
                rule.required_forms if rule else report.required_forms,
                rule.required_kinds if rule else report.required_kinds,
            )
        )
        if released:
            return ConformanceGateDecision(
                path_id=path_id,
                health=RouteHealth.CONFORMANT,
                promotion_allowed=report.permitted_assurance.rank > 0,
                maximum_assurance=report.permitted_assurance,
                reasons=(),
                report_id=report.report_id,
                retained_authorities=tuple(authoritative_for),
            )

        reasons: list[QuarantineReason] = []
        if rule is not None:
            reasons.append(rule.reason)
        if report is not None:
            if any(
                item.status is ConformanceStatus.TIMED_OUT for item in report.cases
            ):
                reasons.append(QuarantineReason.TIMEOUT)
            elif not report.complete:
                reasons.append(QuarantineReason.INCOMPLETE_FIXTURE_COVERAGE)
            else:
                reasons.append(QuarantineReason.TRANSLATION_NONCONFORMANCE)
        health = (
            RouteHealth.DEGRADED
            if rule is not None and rule.degraded and report is None
            else RouteHealth.QUARANTINED
            if report is not None
            else RouteHealth.UNASSESSED
        )
        return ConformanceGateDecision(
            path_id=path_id,
            health=health,
            promotion_allowed=False,
            maximum_assurance=AssuranceLevel.UNVERIFIED,
            reasons=tuple(reasons),
            report_id=report.report_id if report else None,
        )


def gate_prover_path(
    path_id: str,
    report: Optional[ConformanceReport] = None,
    *,
    authoritative_for: Iterable[str] = (),
    registry: Optional[ProverQuarantineRegistry] = None,
) -> ConformanceGateDecision:
    """Convenience fail-closed gate used by the subsequent portfolio router."""

    return (registry or ProverQuarantineRegistry()).assess(
        path_id, report, authoritative_for=authoritative_for
    )


def _fixture_inventory(form: LogicForm, kind: ConformanceTestKind) -> SemanticInventory:
    prefix = f"{form.value}:{kind.value}"
    return SemanticInventory(
        actors=(f"{prefix}:actor:agent-a",),
        times=(f"{prefix}:time:7",),
        quantifiers=(f"{prefix}:forall:task",),
        modal_operators=(f"{prefix}:obligation",),
        bounds=(f"{prefix}:upper=9",),
        premises=(f"{prefix}:premise:authorized",),
        predicates=(f"{prefix}:predicate:completed",),
        polarities=(f"{prefix}:positive",),
        variables=(f"{prefix}:variable:task",),
        metadata={"fixture_semantics": "reviewed-symbolic-inventory"},
    )


def _default_fixture(
    form: LogicForm, kind: ConformanceTestKind
) -> ConformanceFixture:
    slug = form.value.replace("+", "plus").replace("-", "_")
    source = {
        LogicForm.AST: '{"operator":"obligation","actor":"agent-a","time":7}',
        LogicForm.DCEC: "O(agent-a,7,forall task.completed(task))",
        LogicForm.TDFOL: "forall task. O_agent-a F[0,9] completed(task)",
        LogicForm.FOL: "forall task. authorized(agent-a,task) -> completed(task,7)",
        LogicForm.TPTP: "fof(f,axiom,![T]:(authorized(agent_a,T)=>completed(T,t7))).",
        LogicForm.SMT_LIB: "(assert (forall ((t Task)) (=> (authorized agent_a t) (completed t 7))))",
        LogicForm.TLA_PLUS: "Inv == \\A t \\in Tasks : Authorized[a,t] => Completed[t]",
        LogicForm.PROTOCOL: "rule Execute: [Authorized(a,t)] --[Completed(a,t,#i)]-> []",
        LogicForm.HYPERPROPERTY: "forall pi. forall pi'. lowEq(pi,pi') -> G lowEqOut(pi,pi')",
    }[form]
    expected = "rejected" if kind is ConformanceTestKind.NEGATIVE else "satisfied"
    return ConformanceFixture(
        fixture_id=f"{slug}-{kind.value}@1",
        kind=kind,
        source_form=form,
        source_text=source,
        source_inventory=_fixture_inventory(form, kind),
        semantic_profile_id=f"supervisor-{slug}@1",
        required_dimensions=tuple(SemanticDimension),
        expected_outcome=expected,
        mutation=(
            {"dimension": "premises", "operation": "drop"}
            if kind is ConformanceTestKind.MUTATION
            else {}
        ),
        metadata={
            "negative": kind is ConformanceTestKind.NEGATIVE,
            "inert_text": True,
        },
    )


DEFAULT_CONFORMANCE_FIXTURE_SET = ConformanceFixtureSet(
    name="supervisor-logic-translation-conformance",
    version="1",
    fixtures=tuple(
        _default_fixture(form, kind)
        for form in LogicForm
        for kind in ConformanceTestKind
    ),
)
DEFAULT_CONFORMANCE_FIXTURES = DEFAULT_CONFORMANCE_FIXTURE_SET.fixtures
DEFAULT_CONFORMANCE_FIXTURE_SET_ID = (
    DEFAULT_CONFORMANCE_FIXTURE_SET.fixture_set_id
)
REQUIRED_CONFORMANCE_FORMS = frozenset(LogicForm)
REQUIRED_CONFORMANCE_KINDS = frozenset(ConformanceTestKind)


__all__ = [
    "CONFORMANCE_CASE_SCHEMA",
    "CONFORMANCE_FIXTURE_SCHEMA",
    "CONFORMANCE_FIXTURE_SET_SCHEMA",
    "CONFORMANCE_GATE_SCHEMA",
    "CONFORMANCE_REPORT_SCHEMA",
    "DEFAULT_CONFORMANCE_FIXTURES",
    "DEFAULT_CONFORMANCE_FIXTURE_SET",
    "DEFAULT_CONFORMANCE_FIXTURE_SET_ID",
    "DEFAULT_CONFORMANCE_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CONFORMANCE_CASES",
    "DEFAULT_QUARANTINE_RULES",
    "LEGACY_CEC_DCEC_WRAPPER",
    "LEGACY_CEC_DEONTIC_API",
    "LEGACY_CEC_PROOF_CACHE",
    "LEGACY_DCEC_TO_TDFOL_TRANSLATOR",
    "LEGACY_TDFOL_PROOF_CACHE",
    "LEGACY_TDFOL_TO_FOL_TRANSLATOR",
    "PROVER_CONFORMANCE_VERSION",
    "REQUIRED_CONFORMANCE_FORMS",
    "REQUIRED_CONFORMANCE_KINDS",
    "ConformanceCaseResult",
    "ConformanceFixture",
    "ConformanceFixtureSet",
    "ConformanceGateDecision",
    "ConformanceMethod",
    "ConformanceObservation",
    "ConformanceReport",
    "ConformanceRunConfig",
    "ConformanceStatus",
    "ConformanceTestKind",
    "FixtureRunner",
    "ProverConformanceRunner",
    "ProverQuarantineRegistry",
    "QuarantineReason",
    "QuarantineRule",
    "RouteHealth",
    "gate_prover_path",
]
