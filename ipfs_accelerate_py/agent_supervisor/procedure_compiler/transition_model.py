"""Bounded transition prediction and integer-only calibration.

Transition models are cost/planning aids, not evidence producers.  Even an
``exact`` prediction cannot grant authority, establish a proof, establish a
postcondition, establish completion, or suppress validation.  It may discharge
only a deterministic *planning* obligation.  A ``conservative`` prediction may
do that only when the caller supplies the identity of a separately admitted
conservative-evidence receipt.  Empirical and heuristic predictions are
limited to cost and priority decisions.

Calibration compares integer quantities for files, symbols, effects, tests,
proofs, duration, tokens, provider cost, merge conflicts, and terminal status.
No floating-point score can hide an unsafe structural mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract
from .contracts import (
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ProcedureContractError,
    _bounded,
    _decode_fields,
    _enum,
    _identifier,
    _nested,
    _nonnegative_int,
    _schema_name,
    _strings,
    _verify_identity,
)
from .world_model import TransitionClass

MAX_TRANSITION_REFERENCES: Final[int] = 256
MAX_TRANSITION_QUANTITY: Final[int] = 2**63 - 1
CALIBRATION_DIMENSION_COUNT: Final[int] = 10


class TransitionModelError(ProcedureContractError):
    """A transition prediction, observation, or calibration is malformed."""


class PredictionUseError(TransitionModelError):
    """A prediction was offered for a use it can never discharge."""


class CalibrationAdmissionError(TransitionModelError):
    """Unadmitted or stale calibration evidence was offered to a model."""


class ConfidenceClass(str, Enum):
    EXACT = "exact"
    CONSERVATIVE = "conservative"
    EMPIRICAL = "empirical"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


class PredictionUse(str, Enum):
    COST = "cost"
    PRIORITY = "priority"
    DETERMINISTIC_PLANNING = "deterministic_planning"
    AUTHORITY = "authority"
    POSTCONDITION = "postcondition"
    PROOF = "proof"
    COMPLETION = "completion"
    VALIDATION_SUPPRESSION = "validation_suppression"
    HUMAN_REVIEW_SUPPRESSION = "human_review_suppression"


class TransitionTerminalStatus(str, Enum):
    UNKNOWN = "unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    ROLLED_BACK = "rolled_back"
    ESCALATED = "escalated"
    QUARANTINED = "quarantined"
    CANCELLED = "cancelled"
    REFUSED = "refused"


_TERMINAL_CODES: Final[dict[TransitionTerminalStatus, int]] = {
    status: index for index, status in enumerate(TransitionTerminalStatus)
}


class ObservationClass(str, Enum):
    ADMITTED_EXTERNAL = "admitted_external"
    CANDIDATE = "candidate"
    SIMULATED = "simulated"
    STALE = "stale"
    REJECTED = "rejected"


class CalibrationDimension(str, Enum):
    FILES = "files"
    SYMBOLS = "symbols"
    EFFECTS = "effects"
    TESTS = "tests"
    PROOFS = "proofs"
    DURATION = "duration"
    TOKENS = "tokens"
    COST = "cost"
    CONFLICTS = "conflicts"
    TERMINAL = "terminal"


class CalibrationDisposition(str, Enum):
    MATCHED = "matched"
    DRIFTED = "drifted"
    CRITICAL_DRIFT = "critical_drift"
    UNADMITTED = "unadmitted"


class TransitionModelState(str, Enum):
    CURRENT = "current"
    DEMOTED = "demoted"
    INVALIDATED = "invalidated"
    STALE = "stale"


_MEASUREMENT_FIELDS: Final[tuple[tuple[CalibrationDimension, str], ...]] = (
    (CalibrationDimension.FILES, "changed_files"),
    (CalibrationDimension.SYMBOLS, "changed_symbols"),
    (CalibrationDimension.EFFECTS, "effects"),
    (CalibrationDimension.TESTS, "tests"),
    (CalibrationDimension.PROOFS, "proofs"),
    (CalibrationDimension.DURATION, "duration_ms"),
    (CalibrationDimension.TOKENS, "tokens"),
    (CalibrationDimension.COST, "provider_cost_micros"),
    (CalibrationDimension.CONFLICTS, "merge_conflicts"),
)


def _counter(value: Any, field_name: str) -> int:
    return _nonnegative_int(value, field_name, maximum=MAX_TRANSITION_QUANTITY)


def _ids(values: Any, field_name: str) -> tuple[str, ...]:
    return _strings(
        values,
        field_name,
        limit=MAX_TRANSITION_REFERENCES,
        identifiers=True,
        preserve_order=False,
    )


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _measurements(value: Any) -> "TransitionMeasurements":
    return _nested(value, TransitionMeasurements, "measurements")


@dataclass(frozen=True)
class TransitionMeasurements(CanonicalContract):
    """Integer-only predicted or observed transition quantities."""

    SCHEMA: ClassVar[str] = _schema_name("TransitionMeasurements")

    changed_files: int = 0
    changed_symbols: int = 0
    effects: int = 0
    tests: int = 0
    proofs: int = 0
    duration_ms: int = 0
    tokens: int = 0
    provider_cost_micros: int = 0
    merge_conflicts: int = 0
    terminal_status: TransitionTerminalStatus = TransitionTerminalStatus.UNKNOWN

    def __post_init__(self) -> None:
        for _, name in _MEASUREMENT_FIELDS:
            object.__setattr__(self, name, _counter(getattr(self, name), name))
        object.__setattr__(
            self,
            "terminal_status",
            _enum(self.terminal_status, TransitionTerminalStatus, "terminal_status"),
        )
        _bounded(self, "TransitionMeasurements")

    @property
    def terminal_code(self) -> int:
        return _TERMINAL_CODES[self.terminal_status]

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            **{name: getattr(self, name) for _, name in _MEASUREMENT_FIELDS},
            "terminal_status": self.terminal_status.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionMeasurements":
        fields = tuple(name for _, name in _MEASUREMENT_FIELDS) + ("terminal_status",)
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TransitionPrediction(CanonicalContract):
    """One bounded prediction emitted by a proposal-only transition model."""

    SCHEMA: ClassVar[str] = _schema_name("TransitionPrediction")

    bindings: ArtifactBindings
    model_id: str
    model_revision: int
    transition_class: TransitionClass
    confidence_class: ConfidenceClass
    source_state_id: str
    predicted_state_id: str
    predicted_delta_id: str
    measurements: TransitionMeasurements
    changed_file_ids: tuple[str, ...] = ()
    changed_symbol_ids: tuple[str, ...] = ()
    effect_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    proof_ids: tuple[str, ...] = ()
    merge_conflict_ids: tuple[str, ...] = ()
    conservative_evidence_id: str = ""
    validation_dependency_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "model_id",
            "source_state_id",
            "predicted_state_id",
            "predicted_delta_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "model_revision", _counter(self.model_revision, "model_revision"))
        object.__setattr__(
            self,
            "transition_class",
            _enum(self.transition_class, TransitionClass, "transition_class"),
        )
        object.__setattr__(
            self,
            "confidence_class",
            _enum(self.confidence_class, ConfidenceClass, "confidence_class"),
        )
        object.__setattr__(self, "measurements", _measurements(self.measurements))
        for name in (
            "changed_file_ids",
            "changed_symbol_ids",
            "effect_ids",
            "test_ids",
            "proof_ids",
            "merge_conflict_ids",
            "validation_dependency_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "conservative_evidence_id",
            _identifier(
                self.conservative_evidence_id,
                "conservative_evidence_id",
                required=False,
            ),
        )
        _bounded(self, "TransitionPrediction")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_establish_postcondition(self) -> bool:
        return False

    @property
    def can_establish_proof(self) -> bool:
        return False

    @property
    def can_establish_completion(self) -> bool:
        return False

    def allows_use(
        self,
        use: PredictionUse | str,
        *,
        admitted_conservative_evidence_ids: Sequence[str] = (),
    ) -> bool:
        normalized = _enum(use, PredictionUse, "use")
        if normalized in {
            PredictionUse.AUTHORITY,
            PredictionUse.POSTCONDITION,
            PredictionUse.PROOF,
            PredictionUse.COMPLETION,
            PredictionUse.VALIDATION_SUPPRESSION,
            PredictionUse.HUMAN_REVIEW_SUPPRESSION,
        }:
            return False
        if normalized in {PredictionUse.COST, PredictionUse.PRIORITY}:
            return self.confidence_class is not ConfidenceClass.UNKNOWN
        if self.confidence_class is ConfidenceClass.EXACT:
            return True
        if self.confidence_class is ConfidenceClass.CONSERVATIVE:
            admitted = set(_ids(admitted_conservative_evidence_ids, "admitted_evidence_ids"))
            return bool(self.conservative_evidence_id and self.conservative_evidence_id in admitted)
        return False

    def require_use(
        self,
        use: PredictionUse | str,
        *,
        admitted_conservative_evidence_ids: Sequence[str] = (),
    ) -> None:
        if not self.allows_use(
            use,
            admitted_conservative_evidence_ids=admitted_conservative_evidence_ids,
        ):
            raise PredictionUseError("transition prediction cannot discharge the requested use")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "transition_class": self.transition_class.value,
            "confidence_class": self.confidence_class.value,
            "source_state_id": self.source_state_id,
            "predicted_state_id": self.predicted_state_id,
            "predicted_delta_id": self.predicted_delta_id,
            "measurements": self.measurements,
            "changed_file_ids": self.changed_file_ids,
            "changed_symbol_ids": self.changed_symbol_ids,
            "effect_ids": self.effect_ids,
            "test_ids": self.test_ids,
            "proof_ids": self.proof_ids,
            "merge_conflict_ids": self.merge_conflict_ids,
            "conservative_evidence_id": self.conservative_evidence_id,
            "validation_dependency_ids": self.validation_dependency_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionPrediction":
        fields = (
            "bindings",
            "model_id",
            "model_revision",
            "transition_class",
            "confidence_class",
            "source_state_id",
            "predicted_state_id",
            "predicted_delta_id",
            "measurements",
            "changed_file_ids",
            "changed_symbol_ids",
            "effect_ids",
            "test_ids",
            "proof_ids",
            "merge_conflict_ids",
            "conservative_evidence_id",
            "validation_dependency_ids",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "measurements" in values:
            values["measurements"] = _measurements(values["measurements"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TransitionObservation(CanonicalContract):
    """Observed transition linked to independently admitted evidence."""

    SCHEMA: ClassVar[str] = _schema_name("TransitionObservation")

    bindings: ArtifactBindings
    transition_class: TransitionClass
    source_state_id: str
    observed_state_id: str
    world_state_delta_id: str
    measurements: TransitionMeasurements
    producer_id: str
    observation_class: ObservationClass = ObservationClass.CANDIDATE
    admission_receipt_id: str = ""
    changed_file_ids: tuple[str, ...] = ()
    changed_symbol_ids: tuple[str, ...] = ()
    effect_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    proof_ids: tuple[str, ...] = ()
    merge_conflict_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self,
            "transition_class",
            _enum(self.transition_class, TransitionClass, "transition_class"),
        )
        for name in (
            "source_state_id",
            "observed_state_id",
            "world_state_delta_id",
            "producer_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "measurements", _measurements(self.measurements))
        object.__setattr__(
            self,
            "observation_class",
            _enum(self.observation_class, ObservationClass, "observation_class"),
        )
        object.__setattr__(
            self,
            "admission_receipt_id",
            _identifier(self.admission_receipt_id, "admission_receipt_id", required=False),
        )
        if (
            self.observation_class is ObservationClass.ADMITTED_EXTERNAL
            and not self.admission_receipt_id
        ):
            raise TransitionModelError(
                "an admitted observation must name its external admission receipt"
            )
        if self.observation_class is ObservationClass.SIMULATED and self.admission_receipt_id:
            raise TransitionModelError("simulated evidence cannot be represented as admitted")
        for name in (
            "changed_file_ids",
            "changed_symbol_ids",
            "effect_ids",
            "test_ids",
            "proof_ids",
            "merge_conflict_ids",
            "evidence_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        _bounded(self, "TransitionObservation")

    def externally_admitted(self, admitted_receipt_ids: Sequence[str]) -> bool:
        admitted = set(_ids(admitted_receipt_ids, "admitted_receipt_ids"))
        return bool(
            self.observation_class is ObservationClass.ADMITTED_EXTERNAL
            and self.admission_receipt_id
            and self.admission_receipt_id in admitted
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "transition_class": self.transition_class.value,
            "source_state_id": self.source_state_id,
            "observed_state_id": self.observed_state_id,
            "world_state_delta_id": self.world_state_delta_id,
            "measurements": self.measurements,
            "producer_id": self.producer_id,
            "observation_class": self.observation_class.value,
            "admission_receipt_id": self.admission_receipt_id,
            "changed_file_ids": self.changed_file_ids,
            "changed_symbol_ids": self.changed_symbol_ids,
            "effect_ids": self.effect_ids,
            "test_ids": self.test_ids,
            "proof_ids": self.proof_ids,
            "merge_conflict_ids": self.merge_conflict_ids,
            "evidence_ids": self.evidence_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionObservation":
        fields = (
            "bindings",
            "transition_class",
            "source_state_id",
            "observed_state_id",
            "world_state_delta_id",
            "measurements",
            "producer_id",
            "observation_class",
            "admission_receipt_id",
            "changed_file_ids",
            "changed_symbol_ids",
            "effect_ids",
            "test_ids",
            "proof_ids",
            "merge_conflict_ids",
            "evidence_ids",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "measurements" in values:
            values["measurements"] = _measurements(values["measurements"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class CalibrationComparison(CanonicalContract):
    """One integer comparison; derived error fields cannot be forged."""

    SCHEMA: ClassVar[str] = _schema_name("CalibrationComparison")

    dimension: CalibrationDimension
    predicted: int
    observed: int
    tolerance: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dimension", _enum(self.dimension, CalibrationDimension, "dimension")
        )
        for name in ("predicted", "observed", "tolerance"):
            object.__setattr__(self, name, _counter(getattr(self, name), name))

    @property
    def absolute_error(self) -> int:
        return abs(self.predicted - self.observed)

    @property
    def within_tolerance(self) -> bool:
        return self.absolute_error <= self.tolerance

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "dimension": self.dimension.value,
            "predicted": self.predicted,
            "observed": self.observed,
            "tolerance": self.tolerance,
            "absolute_error": self.absolute_error,
            "within_tolerance": self.within_tolerance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationComparison":
        fields = (
            "dimension",
            "predicted",
            "observed",
            "tolerance",
            "absolute_error",
            "within_tolerance",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        claimed_error = values.pop("absolute_error", None)
        claimed_within = values.pop("within_tolerance", None)
        record = cls(**values)
        if claimed_error is not None and claimed_error != record.absolute_error:
            raise TransitionModelError("calibration absolute error does not verify")
        if claimed_within is not None and claimed_within is not record.within_tolerance:
            raise TransitionModelError("calibration tolerance result does not verify")
        _verify_identity(payload, record)
        return record


def _comparisons(values: Any) -> tuple[CalibrationComparison, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TransitionModelError("comparisons must be a sequence")
    if len(values) != CALIBRATION_DIMENSION_COUNT:
        raise TransitionModelError("calibration must compare every required dimension")
    items: list[CalibrationComparison] = []
    for value in values:
        if isinstance(value, CalibrationComparison):
            item = value
        elif isinstance(value, Mapping) and "schema" in value:
            item = CalibrationComparison.from_dict(value)
        elif isinstance(value, Mapping):
            item = CalibrationComparison(**value)
        else:
            raise TransitionModelError("comparisons contains a malformed record")
        items.append(item)
    dimensions = {item.dimension for item in items}
    if dimensions != set(CalibrationDimension):
        raise TransitionModelError("calibration dimensions must be complete and unique")
    return tuple(sorted(items, key=lambda item: item.dimension.value))


@dataclass(frozen=True)
class CalibrationPolicy(CanonicalContract):
    """Reviewed integer tolerances and automatic drift dispositions."""

    SCHEMA: ClassVar[str] = _schema_name("CalibrationPolicy")

    files_tolerance: int = 0
    symbols_tolerance: int = 0
    effects_tolerance: int = 0
    tests_tolerance: int = 0
    proofs_tolerance: int = 0
    duration_tolerance_ms: int = 1_000
    tokens_tolerance: int = 128
    cost_tolerance_micros: int = 1_000
    conflicts_tolerance: int = 0
    terminal_tolerance: int = 0
    critical_dimensions: tuple[CalibrationDimension, ...] = (
        CalibrationDimension.FILES,
        CalibrationDimension.SYMBOLS,
        CalibrationDimension.EFFECTS,
        CalibrationDimension.TESTS,
        CalibrationDimension.PROOFS,
        CalibrationDimension.CONFLICTS,
        CalibrationDimension.TERMINAL,
    )
    invalidate_after_consecutive_drift: int = 2
    invalidate_after_total_drift: int = 3

    def __post_init__(self) -> None:
        for name in (
            "files_tolerance",
            "symbols_tolerance",
            "effects_tolerance",
            "tests_tolerance",
            "proofs_tolerance",
            "duration_tolerance_ms",
            "tokens_tolerance",
            "cost_tolerance_micros",
            "conflicts_tolerance",
            "terminal_tolerance",
            "invalidate_after_consecutive_drift",
            "invalidate_after_total_drift",
        ):
            object.__setattr__(self, name, _counter(getattr(self, name), name))
        if not self.invalidate_after_consecutive_drift or not self.invalidate_after_total_drift:
            raise TransitionModelError("drift invalidation thresholds must be positive")
        if self.terminal_tolerance != 0:
            raise TransitionModelError("terminal status calibration always requires exact equality")
        critical: set[CalibrationDimension] = set()
        for value in self.critical_dimensions:
            critical.add(_enum(value, CalibrationDimension, "critical_dimensions"))
        object.__setattr__(
            self, "critical_dimensions", tuple(sorted(critical, key=lambda item: item.value))
        )
        non_compensable = {
            CalibrationDimension.FILES,
            CalibrationDimension.EFFECTS,
            CalibrationDimension.TESTS,
            CalibrationDimension.PROOFS,
            CalibrationDimension.CONFLICTS,
            CalibrationDimension.TERMINAL,
        }
        if not non_compensable.issubset(critical):
            raise TransitionModelError(
                "critical_dimensions cannot omit scope, effect, validation, proof, "
                "conflict, or terminal drift"
            )
        _bounded(self, "CalibrationPolicy")

    def tolerance_for(self, dimension: CalibrationDimension) -> int:
        return {
            CalibrationDimension.FILES: self.files_tolerance,
            CalibrationDimension.SYMBOLS: self.symbols_tolerance,
            CalibrationDimension.EFFECTS: self.effects_tolerance,
            CalibrationDimension.TESTS: self.tests_tolerance,
            CalibrationDimension.PROOFS: self.proofs_tolerance,
            CalibrationDimension.DURATION: self.duration_tolerance_ms,
            CalibrationDimension.TOKENS: self.tokens_tolerance,
            CalibrationDimension.COST: self.cost_tolerance_micros,
            CalibrationDimension.CONFLICTS: self.conflicts_tolerance,
            CalibrationDimension.TERMINAL: self.terminal_tolerance,
        }[dimension]

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "files_tolerance": self.files_tolerance,
            "symbols_tolerance": self.symbols_tolerance,
            "effects_tolerance": self.effects_tolerance,
            "tests_tolerance": self.tests_tolerance,
            "proofs_tolerance": self.proofs_tolerance,
            "duration_tolerance_ms": self.duration_tolerance_ms,
            "tokens_tolerance": self.tokens_tolerance,
            "cost_tolerance_micros": self.cost_tolerance_micros,
            "conflicts_tolerance": self.conflicts_tolerance,
            "terminal_tolerance": self.terminal_tolerance,
            "critical_dimensions": tuple(item.value for item in self.critical_dimensions),
            "invalidate_after_consecutive_drift": self.invalidate_after_consecutive_drift,
            "invalidate_after_total_drift": self.invalidate_after_total_drift,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationPolicy":
        fields = (
            "files_tolerance",
            "symbols_tolerance",
            "effects_tolerance",
            "tests_tolerance",
            "proofs_tolerance",
            "duration_tolerance_ms",
            "tokens_tolerance",
            "cost_tolerance_micros",
            "conflicts_tolerance",
            "terminal_tolerance",
            "critical_dimensions",
            "invalidate_after_consecutive_drift",
            "invalidate_after_total_drift",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class PredictionCalibration(CanonicalContract):
    """Content-addressed integer comparison of one prediction and observation."""

    SCHEMA: ClassVar[str] = _schema_name("PredictionCalibration")

    bindings: ArtifactBindings
    prediction_id: str
    observation_id: str
    model_id: str
    model_revision: int
    transition_class: TransitionClass
    comparisons: tuple[CalibrationComparison, ...]
    identity_mismatch_dimensions: tuple[CalibrationDimension, ...] = ()
    transition_class_match: bool = True
    observation_admitted: bool = False
    admission_receipt_id: str = ""
    critical_dimensions: tuple[CalibrationDimension, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("prediction_id", "observation_id", "model_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "model_revision", _counter(self.model_revision, "model_revision"))
        object.__setattr__(
            self,
            "transition_class",
            _enum(self.transition_class, TransitionClass, "transition_class"),
        )
        object.__setattr__(self, "comparisons", _comparisons(self.comparisons))
        for name in ("identity_mismatch_dimensions", "critical_dimensions"):
            normalized = {_enum(value, CalibrationDimension, name) for value in getattr(self, name)}
            object.__setattr__(self, name, tuple(sorted(normalized, key=lambda item: item.value)))
        if type(self.transition_class_match) is not bool:
            raise TransitionModelError("transition_class_match must be a boolean")
        if type(self.observation_admitted) is not bool:
            raise TransitionModelError("observation_admitted must be a boolean")
        object.__setattr__(
            self,
            "admission_receipt_id",
            _identifier(self.admission_receipt_id, "admission_receipt_id", required=False),
        )
        if self.observation_admitted and not self.admission_receipt_id:
            raise TransitionModelError("admitted calibration requires an admission receipt")
        if not self.observation_admitted and self.admission_receipt_id:
            raise TransitionModelError("unadmitted calibration cannot retain an admitted receipt")
        drift = set(self.drift_dimensions)
        if set(self.critical_dimensions).difference(drift):
            raise TransitionModelError("critical calibration dimensions must have drifted")
        _bounded(self, "PredictionCalibration")

    @property
    def drift_dimensions(self) -> tuple[CalibrationDimension, ...]:
        dimensions = {item.dimension for item in self.comparisons if not item.within_tolerance}
        dimensions.update(self.identity_mismatch_dimensions)
        return tuple(sorted(dimensions, key=lambda item: item.value))

    @property
    def drift_detected(self) -> bool:
        return bool(self.drift_dimensions or not self.transition_class_match)

    @property
    def critical_drift(self) -> bool:
        return bool(self.critical_dimensions or not self.transition_class_match)

    @property
    def disposition(self) -> CalibrationDisposition:
        if not self.observation_admitted:
            return CalibrationDisposition.UNADMITTED
        if self.critical_drift:
            return CalibrationDisposition.CRITICAL_DRIFT
        if self.drift_detected:
            return CalibrationDisposition.DRIFTED
        return CalibrationDisposition.MATCHED

    @property
    def total_absolute_error(self) -> int:
        return sum(item.absolute_error for item in self.comparisons)

    def comparison_for(self, dimension: CalibrationDimension | str) -> CalibrationComparison:
        normalized = _enum(dimension, CalibrationDimension, "dimension")
        return next(item for item in self.comparisons if item.dimension is normalized)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "prediction_id": self.prediction_id,
            "observation_id": self.observation_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "transition_class": self.transition_class.value,
            "comparisons": self.comparisons,
            "identity_mismatch_dimensions": tuple(
                item.value for item in self.identity_mismatch_dimensions
            ),
            "transition_class_match": self.transition_class_match,
            "observation_admitted": self.observation_admitted,
            "admission_receipt_id": self.admission_receipt_id,
            "critical_dimensions": tuple(item.value for item in self.critical_dimensions),
            "drift_dimensions": tuple(item.value for item in self.drift_dimensions),
            "drift_detected": self.drift_detected,
            "critical_drift": self.critical_drift,
            "disposition": self.disposition.value,
            "total_absolute_error": self.total_absolute_error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionCalibration":
        fields = (
            "bindings",
            "prediction_id",
            "observation_id",
            "model_id",
            "model_revision",
            "transition_class",
            "comparisons",
            "identity_mismatch_dimensions",
            "transition_class_match",
            "observation_admitted",
            "admission_receipt_id",
            "critical_dimensions",
            "drift_dimensions",
            "drift_detected",
            "critical_drift",
            "disposition",
            "total_absolute_error",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        derived = {
            name: values.pop(name, None)
            for name in (
                "drift_dimensions",
                "drift_detected",
                "critical_drift",
                "disposition",
                "total_absolute_error",
            )
        }
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "comparisons" in values:
            values["comparisons"] = _comparisons(values["comparisons"])
        record = cls(**values)
        expected = {
            "drift_dimensions": [item.value for item in record.drift_dimensions],
            "drift_detected": record.drift_detected,
            "critical_drift": record.critical_drift,
            "disposition": record.disposition.value,
            "total_absolute_error": record.total_absolute_error,
        }
        for name, supplied in derived.items():
            if supplied is not None and supplied != expected[name]:
                raise TransitionModelError(f"derived calibration field {name} does not verify")
        _verify_identity(payload, record)
        return record


def _calibration_ids(values: Any) -> tuple[str, ...]:
    return _ids(values, "calibration_ids")


@dataclass(frozen=True)
class TransitionModel(CanonicalContract):
    """Versioned proposal model with fail-closed drift state."""

    SCHEMA: ClassVar[str] = _schema_name("TransitionModel")

    bindings: ArtifactBindings
    model_id: str
    revision: int
    transition_class: TransitionClass
    confidence_class: ConfidenceClass
    source_episode_ids: tuple[str, ...]
    operation_catalog_revision: str
    effect_policy_revision: str
    verification_policy_revision: str
    state: TransitionModelState = TransitionModelState.CURRENT
    calibration_ids: tuple[str, ...] = ()
    calibration_count: int = 0
    drift_count: int = 0
    consecutive_drift_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "model_id",
            "operation_catalog_revision",
            "effect_policy_revision",
            "verification_policy_revision",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "revision", _counter(self.revision, "revision"))
        object.__setattr__(
            self,
            "transition_class",
            _enum(self.transition_class, TransitionClass, "transition_class"),
        )
        object.__setattr__(
            self,
            "confidence_class",
            _enum(self.confidence_class, ConfidenceClass, "confidence_class"),
        )
        object.__setattr__(
            self, "source_episode_ids", _ids(self.source_episode_ids, "source_episode_ids")
        )
        if not self.source_episode_ids:
            raise TransitionModelError("transition model requires source episodes")
        object.__setattr__(self, "state", _enum(self.state, TransitionModelState, "state"))
        object.__setattr__(self, "calibration_ids", _calibration_ids(self.calibration_ids))
        for name in ("calibration_count", "drift_count", "consecutive_drift_count"):
            object.__setattr__(self, name, _counter(getattr(self, name), name))
        if self.calibration_count < len(self.calibration_ids):
            raise TransitionModelError("calibration_count cannot be below retained identities")
        if self.drift_count > self.calibration_count:
            raise TransitionModelError("drift_count cannot exceed calibration_count")
        if self.consecutive_drift_count > self.drift_count:
            raise TransitionModelError("consecutive drift cannot exceed total drift")
        if (
            self.state is TransitionModelState.INVALIDATED
            and self.confidence_class is not ConfidenceClass.UNKNOWN
        ):
            raise TransitionModelError("an invalidated model must have unknown confidence")
        _bounded(self, "TransitionModel")

    @property
    def usable(self) -> bool:
        return self.state is TransitionModelState.CURRENT

    def apply_calibration(
        self,
        calibration: PredictionCalibration,
        *,
        policy: CalibrationPolicy | Mapping[str, Any] | None = None,
        admitted_calibration_ids: Sequence[str] = (),
    ) -> "TransitionModel":
        """Apply only a separately admitted calibration artifact."""

        if not isinstance(calibration, PredictionCalibration):
            raise TransitionModelError("calibration must be PredictionCalibration")
        admitted = set(_ids(admitted_calibration_ids, "admitted_calibration_ids"))
        if calibration.content_id not in admitted:
            raise CalibrationAdmissionError(
                "transition-model update requires separately admitted calibration evidence"
            )
        return _apply_calibration(self, calibration, _policy(policy))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "model_id": self.model_id,
            "revision": self.revision,
            "transition_class": self.transition_class.value,
            "confidence_class": self.confidence_class.value,
            "source_episode_ids": self.source_episode_ids,
            "operation_catalog_revision": self.operation_catalog_revision,
            "effect_policy_revision": self.effect_policy_revision,
            "verification_policy_revision": self.verification_policy_revision,
            "state": self.state.value,
            "calibration_ids": self.calibration_ids,
            "calibration_count": self.calibration_count,
            "drift_count": self.drift_count,
            "consecutive_drift_count": self.consecutive_drift_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionModel":
        fields = (
            "bindings",
            "model_id",
            "revision",
            "transition_class",
            "confidence_class",
            "source_episode_ids",
            "operation_catalog_revision",
            "effect_policy_revision",
            "verification_policy_revision",
            "state",
            "calibration_ids",
            "calibration_count",
            "drift_count",
            "consecutive_drift_count",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


def _policy(value: CalibrationPolicy | Mapping[str, Any] | None) -> CalibrationPolicy:
    if value is None:
        return CalibrationPolicy()
    return _nested(value, CalibrationPolicy, "policy")


def _identity_mismatches(
    prediction: TransitionPrediction,
    observation: TransitionObservation,
) -> tuple[CalibrationDimension, ...]:
    fields = (
        (CalibrationDimension.FILES, "changed_file_ids"),
        (CalibrationDimension.SYMBOLS, "changed_symbol_ids"),
        (CalibrationDimension.EFFECTS, "effect_ids"),
        (CalibrationDimension.TESTS, "test_ids"),
        (CalibrationDimension.PROOFS, "proof_ids"),
        (CalibrationDimension.CONFLICTS, "merge_conflict_ids"),
    )
    result = []
    for dimension, field_name in fields:
        predicted = getattr(prediction, field_name)
        observed = getattr(observation, field_name)
        # Empty on both sides means the calibration has quantities only.  If
        # either side names concrete identities, exact set equality is required.
        if (predicted or observed) and set(predicted) != set(observed):
            result.append(dimension)
    return tuple(result)


def calibrate_prediction(
    prediction: TransitionPrediction,
    observation: TransitionObservation,
    *,
    policy: CalibrationPolicy | Mapping[str, Any] | None = None,
    admitted_observation_receipt_ids: Sequence[str] = (),
) -> PredictionCalibration:
    """Compare predicted and observed values without floating-point scoring."""

    if not isinstance(prediction, TransitionPrediction):
        raise TransitionModelError("prediction must be TransitionPrediction")
    if not isinstance(observation, TransitionObservation):
        raise TransitionModelError("observation must be TransitionObservation")
    if prediction.bindings.content_id != observation.bindings.content_id:
        raise TransitionModelError("prediction and observation authority bindings differ")
    if prediction.source_state_id != observation.source_state_id:
        raise TransitionModelError("prediction and observation source states differ")
    selected_policy = _policy(policy)
    comparisons: list[CalibrationComparison] = []
    for dimension, field_name in _MEASUREMENT_FIELDS:
        tolerance = (
            0
            if prediction.confidence_class is ConfidenceClass.EXACT
            else selected_policy.tolerance_for(dimension)
        )
        comparisons.append(
            CalibrationComparison(
                dimension=dimension,
                predicted=getattr(prediction.measurements, field_name),
                observed=getattr(observation.measurements, field_name),
                tolerance=tolerance,
            )
        )
    terminal_tolerance = (
        0
        if prediction.confidence_class is ConfidenceClass.EXACT
        else selected_policy.terminal_tolerance
    )
    # Terminal status is categorical.  Encode equality as integer error 0/1;
    # retain the closed statuses in the source records themselves.
    comparisons.append(
        CalibrationComparison(
            dimension=CalibrationDimension.TERMINAL,
            predicted=0,
            observed=(
                0
                if prediction.measurements.terminal_status
                is observation.measurements.terminal_status
                else 1
            ),
            tolerance=terminal_tolerance,
        )
    )
    identity_mismatches = _identity_mismatches(prediction, observation)
    drift_dimensions = {item.dimension for item in comparisons if not item.within_tolerance}
    drift_dimensions.update(identity_mismatches)
    critical = tuple(
        sorted(
            drift_dimensions.intersection(selected_policy.critical_dimensions),
            key=lambda item: item.value,
        )
    )
    admitted = observation.externally_admitted(admitted_observation_receipt_ids)
    return PredictionCalibration(
        bindings=prediction.bindings,
        prediction_id=prediction.content_id,
        observation_id=observation.content_id,
        model_id=prediction.model_id,
        model_revision=prediction.model_revision,
        transition_class=prediction.transition_class,
        comparisons=tuple(comparisons),
        identity_mismatch_dimensions=identity_mismatches,
        transition_class_match=(prediction.transition_class is observation.transition_class),
        observation_admitted=admitted,
        admission_receipt_id=(observation.admission_receipt_id if admitted else ""),
        critical_dimensions=critical,
    )


def _demoted_confidence(confidence: ConfidenceClass) -> ConfidenceClass:
    return {
        ConfidenceClass.EXACT: ConfidenceClass.UNKNOWN,
        ConfidenceClass.CONSERVATIVE: ConfidenceClass.EMPIRICAL,
        ConfidenceClass.EMPIRICAL: ConfidenceClass.HEURISTIC,
        ConfidenceClass.HEURISTIC: ConfidenceClass.UNKNOWN,
        ConfidenceClass.UNKNOWN: ConfidenceClass.UNKNOWN,
    }[confidence]


def _apply_calibration(
    model: TransitionModel,
    calibration: PredictionCalibration,
    policy: CalibrationPolicy,
) -> TransitionModel:
    if model.state in {TransitionModelState.INVALIDATED, TransitionModelState.STALE}:
        raise TransitionModelError("stale or invalidated transition model cannot be updated")
    if calibration.disposition is CalibrationDisposition.UNADMITTED:
        raise CalibrationAdmissionError("unadmitted observation cannot update transition model")
    if calibration.content_id in model.calibration_ids:
        return model
    if calibration.model_id != model.model_id or calibration.model_revision != model.revision:
        raise TransitionModelError("calibration does not bind the current model revision")
    if calibration.bindings.content_id != model.bindings.content_id:
        raise TransitionModelError("calibration does not bind the model authority roots")
    if calibration.transition_class is not model.transition_class:
        raise TransitionModelError("calibration transition class differs from the model")
    calibration_count = model.calibration_count + 1
    drift_count = model.drift_count + (1 if calibration.drift_detected else 0)
    consecutive = model.consecutive_drift_count + 1 if calibration.drift_detected else 0
    state = model.state
    confidence = model.confidence_class
    if calibration.drift_detected:
        invalidate = (
            calibration.critical_drift
            or model.confidence_class is ConfidenceClass.EXACT
            or consecutive >= policy.invalidate_after_consecutive_drift
            or drift_count >= policy.invalidate_after_total_drift
        )
        if invalidate:
            state = TransitionModelState.INVALIDATED
            confidence = ConfidenceClass.UNKNOWN
        else:
            state = TransitionModelState.DEMOTED
            confidence = _demoted_confidence(confidence)
    # A later match never silently re-promotes a demoted model.
    return TransitionModel(
        bindings=model.bindings,
        model_id=model.model_id,
        revision=model.revision + 1,
        transition_class=model.transition_class,
        confidence_class=confidence,
        source_episode_ids=model.source_episode_ids,
        operation_catalog_revision=model.operation_catalog_revision,
        effect_policy_revision=model.effect_policy_revision,
        verification_policy_revision=model.verification_policy_revision,
        state=state,
        calibration_ids=model.calibration_ids + (calibration.content_id,),
        calibration_count=calibration_count,
        drift_count=drift_count,
        consecutive_drift_count=consecutive,
    )


def calibrate_transition_model(
    model: TransitionModel,
    prediction: TransitionPrediction,
    observation: TransitionObservation,
    *,
    policy: CalibrationPolicy | Mapping[str, Any] | None = None,
    admitted_observation_receipt_ids: Sequence[str] = (),
) -> tuple[TransitionModel, PredictionCalibration]:
    """Calibrate and automatically demote/invalidate a drifted model.

    An unadmitted observation still yields an auditable comparison, but the
    model is returned unchanged.  This prevents receipt-shaped JSON from
    silently altering route or planning state.
    """

    if not isinstance(model, TransitionModel):
        raise TransitionModelError("model must be TransitionModel")
    if prediction.model_id != model.model_id or prediction.model_revision != model.revision:
        raise TransitionModelError("prediction does not bind the current model revision")
    if prediction.bindings.content_id != model.bindings.content_id:
        raise TransitionModelError("prediction does not bind the model authority roots")
    if prediction.transition_class is not model.transition_class:
        raise TransitionModelError("prediction transition class differs from model")
    selected_policy = _policy(policy)
    calibration = calibrate_prediction(
        prediction,
        observation,
        policy=selected_policy,
        admitted_observation_receipt_ids=admitted_observation_receipt_ids,
    )
    if calibration.disposition is CalibrationDisposition.UNADMITTED:
        return model, calibration
    return _apply_calibration(model, calibration, selected_policy), calibration


def prediction_may_discharge(
    prediction: TransitionPrediction,
    use: PredictionUse | str,
    *,
    admitted_conservative_evidence_ids: Sequence[str] = (),
) -> bool:
    """Public fail-closed decision rule for every prediction consumer."""

    if not isinstance(prediction, TransitionPrediction):
        raise TransitionModelError("prediction must be TransitionPrediction")
    return prediction.allows_use(
        use,
        admitted_conservative_evidence_ids=admitted_conservative_evidence_ids,
    )


__all__ = [
    "CalibrationAdmissionError",
    "CalibrationComparison",
    "CalibrationDimension",
    "CalibrationDisposition",
    "CalibrationPolicy",
    "ConfidenceClass",
    "ObservationClass",
    "PredictionCalibration",
    "PredictionUse",
    "PredictionUseError",
    "TransitionClass",
    "TransitionMeasurements",
    "TransitionModel",
    "TransitionModelError",
    "TransitionModelState",
    "TransitionObservation",
    "TransitionPrediction",
    "TransitionTerminalStatus",
    "calibrate_prediction",
    "calibrate_transition_model",
    "prediction_may_discharge",
]
