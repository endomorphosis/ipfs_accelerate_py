"""Bounded advisory OOD signals and independent conservative boundary checks.

OOD detections are advisory unless a boundary contract admits them as a hard
gate. Family, schema, effect, authority, repository, calibration, capability,
and context checks run independently and conservatively. Missing OOD detection
never establishes safety. High-risk unknown or missing group or context
independently abstains. Known in-boundary fixtures remain eligible.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
    bounded_int,
    bounded_json_mapping,
    canonical_id,
    optional_text,
    reject_candidate_authority,
    reject_secret_material,
    required_text,
    strict_fields,
    text_tuple,
)
from .residual_ir import MAX_SCORE_PPM, ResidualTaskInput
from .rights import TrainingCorpusAdmission

OOD_SIGNAL_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-ood-signal@1"
OOD_ASSESSMENT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-ood-assessment@1"
BOUNDARY_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-boundary-contract@1"
)
BOUNDARY_VIOLATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-boundary-violation@1"
)
DEFAULT_FAMILY_DISTANCE_THRESHOLD_PPM: Final = 250_000
MAX_FEATURE_RANGES: Final = 256
MAX_IDENTITY_BYTES: Final = 256
HIGH_RISK_DEFAULT: Final[tuple[str, ...]] = (RiskClass.R4.value, RiskClass.R5.value)

_FORBIDDEN_STATISTIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "example_body",
        "example_text",
        "examples",
        "hidden_test_body",
        "private_chain_of_thought",
        "private_source",
        "prose",
        "raw_source",
        "recoverable_source",
        "source_body",
        "source_code",
        "source_text",
    }
)
_OBSERVATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "operation",
        "repository_family",
        "effects",
        "authorities",
        "calibration_group",
        "available_capabilities",
        "context_reference_ids",
        "context_complete",
        "disagreement",
        "disagreement_sources",
        "family_distance_ppm",
        "ood_detection_ran",
    }
)
_REFERENCE_DISTRIBUTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "feature_ranges",
        "family_distance_threshold_ppm",
        "statistic_identity",
        "example_count",
        "held_out",
    }
)


class OODSignalKind(str, Enum):
    FAMILY_DISTANCE = "family_distance"
    FEATURE_RANGE = "feature_range"
    UNKNOWN_SCHEMA = "unknown_schema"
    UNKNOWN_OPERATION = "unknown_operation"
    UNKNOWN_REPOSITORY = "unknown_repository"
    UNKNOWN_FAMILY = "unknown_family"
    UNSEEN_EFFECT = "unseen_effect"
    UNSEEN_AUTHORITY = "unseen_authority"
    DISAGREEMENT = "disagreement"
    CALIBRATION_ABSENCE = "calibration_absence"
    CONTEXT_INCOMPLETE = "context_incomplete"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"


class BoundaryCheck(str, Enum):
    FAMILY = "family"
    SCHEMA = "schema"
    EFFECT = "effect"
    AUTHORITY = "authority"
    REPOSITORY = "repository"
    CALIBRATION = "calibration"
    CAPABILITY = "capability"
    CONTEXT = "context"


_CONSERVATIVE_CHECKS: Final[frozenset[BoundaryCheck]] = frozenset(
    {
        BoundaryCheck.FAMILY,
        BoundaryCheck.SCHEMA,
        BoundaryCheck.EFFECT,
        BoundaryCheck.AUTHORITY,
        BoundaryCheck.REPOSITORY,
        BoundaryCheck.CALIBRATION,
        BoundaryCheck.CONTEXT,
    }
)


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    return _require_bool(value, name)


def _known(value: str, allowed: Sequence[str]) -> bool:
    return bool(value) and value in allowed


def _integer_features(features: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in features.items()
        if type(value) is int
    }


def _feature_ranges_from_payload(value: Any, name: str) -> dict[str, tuple[int, int]]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError(f"{name} must be an object")
    if len(value) > MAX_FEATURE_RANGES:
        raise ResidualIntelligenceError(f"{name} exceeds {MAX_FEATURE_RANGES} items")
    ranges: dict[str, tuple[int, int]] = {}
    for raw_key, raw_bounds in value.items():
        key = required_text(raw_key, f"{name} key", max_bytes=MAX_IDENTITY_BYTES)
        if isinstance(raw_bounds, Sequence) and not isinstance(raw_bounds, (str, bytes, bytearray)):
            if len(raw_bounds) != 2:
                raise ResidualIntelligenceError(f"{name}.{key} must be a minimum/maximum pair")
            minimum = bounded_int(
                raw_bounds[0],
                f"{name}.{key}.minimum",
                minimum=-(10**12),
                maximum=10**12,
            )
            maximum = bounded_int(
                raw_bounds[1],
                f"{name}.{key}.maximum",
                minimum=-(10**12),
                maximum=10**12,
            )
        elif isinstance(raw_bounds, Mapping):
            strict_fields(
                raw_bounds,
                allowed={"minimum", "maximum"},
                required={"minimum", "maximum"},
                noun=f"{name}.{key}",
            )
            minimum = bounded_int(
                raw_bounds["minimum"],
                f"{name}.{key}.minimum",
                minimum=-(10**12),
                maximum=10**12,
            )
            maximum = bounded_int(
                raw_bounds["maximum"],
                f"{name}.{key}.maximum",
                minimum=-(10**12),
                maximum=10**12,
            )
        else:
            raise ResidualIntelligenceError(f"{name}.{key} must be an object")
        if minimum > maximum:
            raise ResidualIntelligenceError(f"{name}.{key} bounds are inverted")
        ranges[key] = (minimum, maximum)
    return ranges


def _feature_ranges_to_payload(ranges: Mapping[str, tuple[int, int]]) -> dict[str, dict[str, int]]:
    return {
        name: {"minimum": bounds[0], "maximum": bounds[1]}
        for name, bounds in sorted(ranges.items())
    }


def _reject_recoverable_private_source(value: Mapping[str, Any], *, noun: str) -> None:
    reject_secret_material(value, noun=noun)
    reject_candidate_authority(value)

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                token = str(key).strip().casefold().replace("-", "_")
                if token in _FORBIDDEN_STATISTIC_KEYS:
                    raise ResidualIntelligenceError(
                        f"{noun} contains recoverable private source field {path}{key}"
                    )
                visit(child, f"{path}{key}.")
        elif isinstance(item, str):
            if len(item.encode("utf-8")) > MAX_IDENTITY_BYTES:
                raise ResidualIntelligenceError(
                    f"{noun} compact statistics cannot contain recoverable private source at {path}"
                )
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}{index}.")

    visit(value, "")


def _require_admitted_reference(
    reference_distribution: Mapping[str, Any] | None,
    corpus_admission: TrainingCorpusAdmission | None,
) -> None:
    if reference_distribution is None:
        return
    if corpus_admission is None or not corpus_admission.can_train:
        raise ResidualIntelligenceError(
            "reference distributions require an admitted TrainingCorpusAdmission"
        )


def _parse_reference_distribution(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError("reference_distribution must be an object")
    payload = bounded_json_mapping(value, "reference_distribution")
    _reject_recoverable_private_source(payload, noun="reference_distribution")
    strict_fields(
        payload,
        allowed=_REFERENCE_DISTRIBUTION_FIELDS,
        noun="reference_distribution",
    )
    parsed: dict[str, Any] = {
        "feature_ranges": _feature_ranges_from_payload(
            payload.get("feature_ranges"),
            "reference_distribution.feature_ranges",
        ),
    }
    if "family_distance_threshold_ppm" in payload:
        parsed["family_distance_threshold_ppm"] = bounded_int(
            payload["family_distance_threshold_ppm"],
            "family_distance_threshold_ppm",
            minimum=0,
            maximum=MAX_SCORE_PPM,
        )
    if "statistic_identity" in payload:
        parsed["statistic_identity"] = required_text(
            payload["statistic_identity"],
            "statistic_identity",
            max_bytes=MAX_IDENTITY_BYTES,
        )
    if "example_count" in payload:
        parsed["example_count"] = bounded_int(
            payload["example_count"],
            "example_count",
            minimum=0,
            maximum=20_000,
        )
    if "held_out" in payload:
        parsed["held_out"] = _require_bool(payload["held_out"], "held_out")
    return parsed


@dataclass(frozen=True)
class _Observation:
    schema: str
    operation: str
    repository_family: str
    effects: tuple[str, ...]
    authorities: tuple[str, ...]
    calibration_group: str
    available_capabilities: tuple[str, ...]
    context_reference_ids: tuple[str, ...]
    context_complete: bool | None
    disagreement: bool
    disagreement_sources: tuple[str, ...]
    family_distance_ppm: int | None
    ood_detection_ran: bool


def _parse_observation(value: Mapping[str, Any] | None) -> _Observation:
    payload: Mapping[str, Any] = {} if value is None else value
    if not isinstance(payload, Mapping):
        raise ResidualIntelligenceError("observation must be an object")
    unknown = sorted(str(key) for key in payload if key not in _OBSERVATION_FIELDS)
    if unknown:
        raise UnknownFieldError(f"observation contains unknown fields: {', '.join(unknown)}")
    family_distance = payload.get("family_distance_ppm")
    return _Observation(
        schema=optional_text(payload.get("schema"), "schema", max_bytes=MAX_IDENTITY_BYTES),
        operation=optional_text(
            payload.get("operation"),
            "operation",
            max_bytes=MAX_IDENTITY_BYTES,
        ),
        repository_family=optional_text(
            payload.get("repository_family"),
            "repository_family",
            max_bytes=MAX_IDENTITY_BYTES,
        ),
        effects=text_tuple(payload.get("effects") or (), "effects", max_items=256),
        authorities=text_tuple(payload.get("authorities") or (), "authorities", max_items=256),
        calibration_group=optional_text(
            payload.get("calibration_group"),
            "calibration_group",
            max_bytes=MAX_IDENTITY_BYTES,
        ),
        available_capabilities=text_tuple(
            payload.get("available_capabilities") or (),
            "available_capabilities",
            max_items=256,
        ),
        context_reference_ids=text_tuple(
            payload.get("context_reference_ids") or (),
            "context_reference_ids",
            max_items=256,
        ),
        context_complete=_optional_bool(payload.get("context_complete"), "context_complete"),
        disagreement=_require_bool(payload.get("disagreement", False), "disagreement"),
        disagreement_sources=text_tuple(
            payload.get("disagreement_sources") or (),
            "disagreement_sources",
            max_items=64,
        ),
        family_distance_ppm=(
            None
            if family_distance is None
            else bounded_int(
                family_distance,
                "family_distance_ppm",
                minimum=0,
                maximum=MAX_SCORE_PPM,
            )
        ),
        ood_detection_ran=_require_bool(
            payload.get("ood_detection_ran", True),
            "ood_detection_ran",
        ),
    )


@dataclass(frozen=True)
class OODSignal:
    """One bounded advisory out-of-distribution observation."""

    kind: OODSignalKind
    subject: str
    reason_code: str
    score_ppm: int
    advisory: bool
    evidence_references: tuple[str, ...] = ()
    schema: str = OOD_SIGNAL_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "signal_id",
            "kind",
            "subject",
            "reason_code",
            "score_ppm",
            "advisory",
            "evidence_references",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != OOD_SIGNAL_SCHEMA:
            raise ResidualIntelligenceError("unsupported OOD signal schema")
        object.__setattr__(self, "kind", OODSignalKind(self.kind))
        object.__setattr__(
            self,
            "subject",
            required_text(self.subject, "subject", max_bytes=MAX_IDENTITY_BYTES),
        )
        object.__setattr__(
            self,
            "reason_code",
            required_text(self.reason_code, "reason_code", max_bytes=MAX_IDENTITY_BYTES),
        )
        object.__setattr__(
            self,
            "score_ppm",
            bounded_int(self.score_ppm, "score_ppm", minimum=0, maximum=MAX_SCORE_PPM),
        )
        object.__setattr__(self, "advisory", _require_bool(self.advisory, "advisory"))
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", max_items=64),
        )

    @property
    def signal_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind.value,
            "subject": self.subject,
            "reason_code": self.reason_code,
            "score_ppm": self.score_ppm,
            "advisory": self.advisory,
            "evidence_references": list(self.evidence_references),
        }
        if include_id:
            result["signal_id"] = self.signal_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OODSignal:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"signal_id"},
            noun="OOD signal",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            kind=OODSignalKind(str(payload.get("kind") or "")),
            subject=str(payload.get("subject") or ""),
            reason_code=str(payload.get("reason_code") or ""),
            score_ppm=payload.get("score_ppm"),
            advisory=payload.get("advisory"),
            evidence_references=tuple(payload.get("evidence_references") or ()),
        )
        claimed = str(payload.get("signal_id") or "")
        if claimed and claimed != result.signal_id:
            raise ResidualIntelligenceError("OOD signal identity mismatch")
        return result


@dataclass(frozen=True)
class BoundaryViolation:
    """One independent conservative boundary-contract failure."""

    check: BoundaryCheck
    reason_code: str
    subject: str
    conservative: bool
    schema: str = BOUNDARY_VIOLATION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "check",
            "reason_code",
            "subject",
            "conservative",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != BOUNDARY_VIOLATION_SCHEMA:
            raise ResidualIntelligenceError("unsupported boundary violation schema")
        object.__setattr__(self, "check", BoundaryCheck(self.check))
        object.__setattr__(
            self,
            "reason_code",
            required_text(self.reason_code, "reason_code", max_bytes=MAX_IDENTITY_BYTES),
        )
        object.__setattr__(
            self,
            "subject",
            required_text(self.subject, "subject", max_bytes=MAX_IDENTITY_BYTES),
        )
        object.__setattr__(self, "conservative", _require_bool(self.conservative, "conservative"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "check": self.check.value,
            "reason_code": self.reason_code,
            "subject": self.subject,
            "conservative": self.conservative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BoundaryViolation:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS,
            noun="boundary violation",
        )
        return cls(
            schema=str(payload.get("schema") or ""),
            check=BoundaryCheck(str(payload.get("check") or "")),
            reason_code=str(payload.get("reason_code") or ""),
            subject=str(payload.get("subject") or ""),
            conservative=payload.get("conservative"),
        )


@dataclass(frozen=True)
class BoundaryContract:
    """Independent conservative family, schema, effect, authority, repository,
    calibration, capability, and context gates.

    OOD signals remain advisory unless ``admit_ood_policy`` is true.
    """

    allowed_task_families: tuple[str, ...]
    allowed_schemas: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    allowed_repository_families: tuple[str, ...]
    allowed_effects: tuple[str, ...]
    allowed_authorities: tuple[str, ...]
    known_calibration_groups: tuple[str, ...]
    required_capabilities: tuple[str, ...] = ()
    required_context_references: tuple[str, ...] = ()
    feature_ranges: Mapping[str, tuple[int, int]] | None = None
    high_risk_classes: tuple[str, ...] = HIGH_RISK_DEFAULT
    family_distance_threshold_ppm: int = DEFAULT_FAMILY_DISTANCE_THRESHOLD_PPM
    admit_ood_policy: bool = False
    conservative_high_risk: bool = True
    schema: str = BOUNDARY_CONTRACT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "contract_id",
            "allowed_task_families",
            "allowed_schemas",
            "allowed_operations",
            "allowed_repository_families",
            "allowed_effects",
            "allowed_authorities",
            "known_calibration_groups",
            "required_capabilities",
            "required_context_references",
            "feature_ranges",
            "high_risk_classes",
            "family_distance_threshold_ppm",
            "admit_ood_policy",
            "conservative_high_risk",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != BOUNDARY_CONTRACT_SCHEMA:
            raise ResidualIntelligenceError("unsupported boundary contract schema")
        families = text_tuple(
            self.allowed_task_families,
            "allowed_task_families",
            allow_empty=False,
            max_items=256,
        )
        for family in families:
            ResidualTaskFamily(family)
        object.__setattr__(self, "allowed_task_families", families)
        object.__setattr__(
            self,
            "allowed_schemas",
            text_tuple(self.allowed_schemas, "allowed_schemas", allow_empty=False, max_items=256),
        )
        object.__setattr__(
            self,
            "allowed_operations",
            text_tuple(
                self.allowed_operations,
                "allowed_operations",
                allow_empty=False,
                max_items=256,
            ),
        )
        object.__setattr__(
            self,
            "allowed_repository_families",
            text_tuple(
                self.allowed_repository_families,
                "allowed_repository_families",
                allow_empty=False,
                max_items=256,
            ),
        )
        object.__setattr__(
            self,
            "allowed_effects",
            text_tuple(self.allowed_effects, "allowed_effects", allow_empty=False, max_items=256),
        )
        object.__setattr__(
            self,
            "allowed_authorities",
            text_tuple(
                self.allowed_authorities,
                "allowed_authorities",
                allow_empty=False,
                max_items=256,
            ),
        )
        object.__setattr__(
            self,
            "known_calibration_groups",
            text_tuple(
                self.known_calibration_groups,
                "known_calibration_groups",
                allow_empty=False,
                max_items=256,
            ),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            text_tuple(self.required_capabilities, "required_capabilities", max_items=256),
        )
        object.__setattr__(
            self,
            "required_context_references",
            text_tuple(
                self.required_context_references,
                "required_context_references",
                max_items=256,
            ),
        )
        ranges = _feature_ranges_from_payload(self.feature_ranges or {}, "feature_ranges")
        object.__setattr__(self, "feature_ranges", ranges)
        risks = text_tuple(
            self.high_risk_classes,
            "high_risk_classes",
            allow_empty=False,
            max_items=16,
        )
        for risk in risks:
            RiskClass(risk)
        object.__setattr__(self, "high_risk_classes", risks)
        object.__setattr__(
            self,
            "family_distance_threshold_ppm",
            bounded_int(
                self.family_distance_threshold_ppm,
                "family_distance_threshold_ppm",
                minimum=0,
                maximum=MAX_SCORE_PPM,
            ),
        )
        object.__setattr__(
            self,
            "admit_ood_policy",
            _require_bool(self.admit_ood_policy, "admit_ood_policy"),
        )
        object.__setattr__(
            self,
            "conservative_high_risk",
            _require_bool(self.conservative_high_risk, "conservative_high_risk"),
        )

    @property
    def contract_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def is_high_risk(self, risk_class: RiskClass | str) -> bool:
        return RiskClass(risk_class).value in self.high_risk_classes

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "allowed_task_families": list(self.allowed_task_families),
            "allowed_schemas": list(self.allowed_schemas),
            "allowed_operations": list(self.allowed_operations),
            "allowed_repository_families": list(self.allowed_repository_families),
            "allowed_effects": list(self.allowed_effects),
            "allowed_authorities": list(self.allowed_authorities),
            "known_calibration_groups": list(self.known_calibration_groups),
            "required_capabilities": list(self.required_capabilities),
            "required_context_references": list(self.required_context_references),
            "feature_ranges": _feature_ranges_to_payload(self.feature_ranges or {}),
            "high_risk_classes": list(self.high_risk_classes),
            "family_distance_threshold_ppm": self.family_distance_threshold_ppm,
            "admit_ood_policy": self.admit_ood_policy,
            "conservative_high_risk": self.conservative_high_risk,
        }
        if include_id:
            result["contract_id"] = self.contract_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BoundaryContract:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"contract_id"},
            noun="boundary contract",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            allowed_task_families=tuple(payload.get("allowed_task_families") or ()),
            allowed_schemas=tuple(payload.get("allowed_schemas") or ()),
            allowed_operations=tuple(payload.get("allowed_operations") or ()),
            allowed_repository_families=tuple(payload.get("allowed_repository_families") or ()),
            allowed_effects=tuple(payload.get("allowed_effects") or ()),
            allowed_authorities=tuple(payload.get("allowed_authorities") or ()),
            known_calibration_groups=tuple(payload.get("known_calibration_groups") or ()),
            required_capabilities=tuple(payload.get("required_capabilities") or ()),
            required_context_references=tuple(payload.get("required_context_references") or ()),
            feature_ranges=payload.get("feature_ranges") or {},
            high_risk_classes=tuple(payload.get("high_risk_classes") or ()),
            family_distance_threshold_ppm=payload.get("family_distance_threshold_ppm"),
            admit_ood_policy=payload.get("admit_ood_policy"),
            conservative_high_risk=payload.get("conservative_high_risk"),
        )
        claimed = str(payload.get("contract_id") or "")
        if claimed and claimed != result.contract_id:
            raise ResidualIntelligenceError("boundary contract identity mismatch")
        return result


def _violation(
    check: BoundaryCheck,
    reason_code: str,
    subject: str,
) -> BoundaryViolation:
    return BoundaryViolation(
        check=check,
        reason_code=reason_code,
        subject=subject or "missing",
        conservative=check in _CONSERVATIVE_CHECKS,
    )


def _signal(
    kind: OODSignalKind,
    subject: str,
    *,
    advisory: bool,
    score_ppm: int = MAX_SCORE_PPM,
    evidence: tuple[str, ...] = (),
) -> OODSignal:
    return OODSignal(
        kind=kind,
        subject=subject or "missing",
        reason_code=kind.value,
        score_ppm=score_ppm,
        advisory=advisory,
        evidence_references=evidence,
    )


def _independent_boundary_checks(
    task_input: ResidualTaskInput,
    contract: BoundaryContract,
    observation: _Observation,
) -> tuple[tuple[BoundaryViolation, ...], dict[BoundaryCheck, bool]]:
    family_ok = task_input.task_family.value in contract.allowed_task_families
    schema_ok = _known(observation.schema, contract.allowed_schemas)
    operation_ok = _known(observation.operation, contract.allowed_operations)
    repository_ok = _known(observation.repository_family, contract.allowed_repository_families)
    effects_ok = bool(observation.effects) and all(
        item in contract.allowed_effects for item in observation.effects
    )
    authorities_ok = bool(observation.authorities) and all(
        item in contract.allowed_authorities for item in observation.authorities
    )
    calibration_ok = _known(observation.calibration_group, contract.known_calibration_groups)
    missing_capabilities = tuple(
        item
        for item in contract.required_capabilities
        if item not in observation.available_capabilities
    )
    capability_ok = not missing_capabilities
    missing_context = tuple(
        item
        for item in contract.required_context_references
        if item not in observation.context_reference_ids
    )
    context_ok = observation.context_complete is True and not missing_context

    violations: list[BoundaryViolation] = []
    if not family_ok:
        violations.append(
            _violation(
                BoundaryCheck.FAMILY,
                OODSignalKind.UNKNOWN_FAMILY.value,
                task_input.task_family.value,
            )
        )
    if not schema_ok:
        violations.append(
            _violation(
                BoundaryCheck.SCHEMA,
                OODSignalKind.UNKNOWN_SCHEMA.value,
                observation.schema,
            )
        )
    if not operation_ok:
        violations.append(
            _violation(
                BoundaryCheck.SCHEMA,
                OODSignalKind.UNKNOWN_OPERATION.value,
                observation.operation,
            )
        )
    if not repository_ok:
        violations.append(
            _violation(
                BoundaryCheck.REPOSITORY,
                OODSignalKind.UNKNOWN_REPOSITORY.value,
                observation.repository_family,
            )
        )
    if not effects_ok:
        unseen = tuple(
            item for item in observation.effects if item not in contract.allowed_effects
        ) or ("missing",)
        for subject in unseen:
            violations.append(
                _violation(BoundaryCheck.EFFECT, OODSignalKind.UNSEEN_EFFECT.value, subject)
            )
    if not authorities_ok:
        unseen_auth = tuple(
            item for item in observation.authorities if item not in contract.allowed_authorities
        ) or ("missing",)
        for subject in unseen_auth:
            violations.append(
                _violation(
                    BoundaryCheck.AUTHORITY,
                    OODSignalKind.UNSEEN_AUTHORITY.value,
                    subject,
                )
            )
    if not calibration_ok:
        violations.append(
            _violation(
                BoundaryCheck.CALIBRATION,
                OODSignalKind.CALIBRATION_ABSENCE.value,
                observation.calibration_group,
            )
        )
    if not capability_ok:
        for subject in missing_capabilities:
            violations.append(
                _violation(
                    BoundaryCheck.CAPABILITY,
                    OODSignalKind.CAPABILITY_UNAVAILABLE.value,
                    subject,
                )
            )
    if not context_ok:
        subjects = missing_context or ("missing",)
        for subject in subjects:
            violations.append(
                _violation(
                    BoundaryCheck.CONTEXT,
                    OODSignalKind.CONTEXT_INCOMPLETE.value,
                    subject,
                )
            )

    flags = {
        BoundaryCheck.FAMILY: family_ok,
        BoundaryCheck.SCHEMA: schema_ok and operation_ok,
        BoundaryCheck.EFFECT: effects_ok,
        BoundaryCheck.AUTHORITY: authorities_ok,
        BoundaryCheck.REPOSITORY: repository_ok,
        BoundaryCheck.CALIBRATION: calibration_ok,
        BoundaryCheck.CAPABILITY: capability_ok,
        BoundaryCheck.CONTEXT: context_ok,
    }
    return tuple(violations), flags


def _feature_range_signals(
    task_input: ResidualTaskInput,
    ranges: Mapping[str, tuple[int, int]],
    *,
    advisory: bool,
) -> tuple[tuple[OODSignal, ...], bool]:
    integers = _integer_features(task_input.compact_features)
    if not ranges:
        return (), not integers
    signals: list[OODSignal] = []
    for name, (minimum, maximum) in sorted(ranges.items()):
        if name not in integers:
            signals.append(_signal(OODSignalKind.FEATURE_RANGE, name, advisory=advisory))
            continue
        value = integers[name]
        if value < minimum or value > maximum:
            signals.append(_signal(OODSignalKind.FEATURE_RANGE, name, advisory=advisory))
    if ranges:
        for name in sorted(set(integers) - set(ranges)):
            signals.append(_signal(OODSignalKind.FEATURE_RANGE, name, advisory=advisory))
    return tuple(signals), True


def _advisory_ood_signals(
    task_input: ResidualTaskInput,
    contract: BoundaryContract,
    observation: _Observation,
    *,
    ranges: Mapping[str, tuple[int, int]],
    distance_threshold_ppm: int,
) -> tuple[tuple[OODSignal, ...], bool]:
    advisory = not contract.admit_ood_policy
    signals: list[OODSignal] = []
    assessed_ranges, range_complete = _feature_range_signals(
        task_input,
        ranges,
        advisory=advisory,
    )
    signals.extend(assessed_ranges)
    if (
        observation.family_distance_ppm is not None
        and observation.family_distance_ppm > distance_threshold_ppm
    ):
        signals.append(
            _signal(
                OODSignalKind.FAMILY_DISTANCE,
                task_input.task_family.value,
                advisory=advisory,
                score_ppm=observation.family_distance_ppm,
            )
        )
    if task_input.task_family.value not in contract.allowed_task_families:
        signals.append(
            _signal(
                OODSignalKind.UNKNOWN_FAMILY,
                task_input.task_family.value,
                advisory=advisory,
            )
        )
    if not _known(observation.schema, contract.allowed_schemas):
        signals.append(
            _signal(OODSignalKind.UNKNOWN_SCHEMA, observation.schema, advisory=advisory)
        )
    if not _known(observation.operation, contract.allowed_operations):
        signals.append(
            _signal(OODSignalKind.UNKNOWN_OPERATION, observation.operation, advisory=advisory)
        )
    if not _known(observation.repository_family, contract.allowed_repository_families):
        signals.append(
            _signal(
                OODSignalKind.UNKNOWN_REPOSITORY,
                observation.repository_family,
                advisory=advisory,
            )
        )
    unseen_effects = tuple(
        item for item in observation.effects if item not in contract.allowed_effects
    )
    if not observation.effects:
        signals.append(_signal(OODSignalKind.UNSEEN_EFFECT, "missing", advisory=advisory))
    else:
        signals.extend(
            _signal(OODSignalKind.UNSEEN_EFFECT, item, advisory=advisory) for item in unseen_effects
        )
    unseen_authorities = tuple(
        item for item in observation.authorities if item not in contract.allowed_authorities
    )
    if not observation.authorities:
        signals.append(_signal(OODSignalKind.UNSEEN_AUTHORITY, "missing", advisory=advisory))
    else:
        signals.extend(
            _signal(OODSignalKind.UNSEEN_AUTHORITY, item, advisory=advisory)
            for item in unseen_authorities
        )
    if observation.disagreement:
        evidence = observation.disagreement_sources or ("disagreement",)
        signals.append(
            _signal(
                OODSignalKind.DISAGREEMENT,
                "disagreement",
                advisory=advisory,
                evidence=evidence,
            )
        )
    if not _known(observation.calibration_group, contract.known_calibration_groups):
        signals.append(
            _signal(
                OODSignalKind.CALIBRATION_ABSENCE,
                observation.calibration_group,
                advisory=advisory,
            )
        )
    if observation.context_complete is not True or any(
        item not in observation.context_reference_ids
        for item in contract.required_context_references
    ):
        missing = tuple(
            item
            for item in contract.required_context_references
            if item not in observation.context_reference_ids
        ) or ("missing",)
        signals.extend(
            _signal(OODSignalKind.CONTEXT_INCOMPLETE, item, advisory=advisory) for item in missing
        )
    missing_capabilities = tuple(
        item
        for item in contract.required_capabilities
        if item not in observation.available_capabilities
    )
    signals.extend(
        _signal(OODSignalKind.CAPABILITY_UNAVAILABLE, item, advisory=advisory)
        for item in missing_capabilities
    )
    return tuple(signals), range_complete


def _select_disposition(
    *,
    capability_violation: bool,
    conservative_abstain: bool,
    hard_ood: bool,
    hard_boundary: bool,
    high_risk: bool,
) -> ExpertDisposition:
    if capability_violation:
        return ExpertDisposition.CAPABILITY_UNAVAILABLE
    if conservative_abstain:
        return ExpertDisposition.ABSTAIN
    if hard_ood:
        return ExpertDisposition.OUT_OF_DISTRIBUTION
    if hard_boundary:
        return ExpertDisposition.REJECT_INPUT
    if high_risk:
        return ExpertDisposition.VALIDATION_REQUIRED
    return ExpertDisposition.ACCEPT


@dataclass(frozen=True)
class OODAssessment:
    """Deterministic OOD and independent boundary-check receipt.

    ``safety_established`` is never true when OOD detection did not run.
    The record remains ``candidate_only`` and creates no authority.
    """

    input_id: str
    contract_id: str
    signals: tuple[OODSignal, ...]
    boundary_violations: tuple[BoundaryViolation, ...]
    disposition: ExpertDisposition
    eligible: bool
    in_boundary: bool
    family_in_boundary: bool
    schema_in_boundary: bool
    effect_in_boundary: bool
    authority_in_boundary: bool
    repository_in_boundary: bool
    calibration_in_boundary: bool
    capability_in_boundary: bool
    context_in_boundary: bool
    conservative_abstain: bool
    ood_detected: bool
    ood_detection_ran: bool
    safety_established: bool
    reason_codes: tuple[str, ...]
    evidence_references: tuple[str, ...]
    candidate_only: bool = True
    schema: str = OOD_ASSESSMENT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "assessment_id",
            "input_id",
            "contract_id",
            "signals",
            "boundary_violations",
            "disposition",
            "eligible",
            "in_boundary",
            "family_in_boundary",
            "schema_in_boundary",
            "effect_in_boundary",
            "authority_in_boundary",
            "repository_in_boundary",
            "calibration_in_boundary",
            "capability_in_boundary",
            "context_in_boundary",
            "conservative_abstain",
            "ood_detected",
            "ood_detection_ran",
            "safety_established",
            "reason_codes",
            "evidence_references",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != OOD_ASSESSMENT_SCHEMA:
            raise ResidualIntelligenceError("unsupported OOD assessment schema")
        object.__setattr__(self, "input_id", required_text(self.input_id, "input_id"))
        object.__setattr__(self, "contract_id", required_text(self.contract_id, "contract_id"))
        if any(not isinstance(item, OODSignal) for item in self.signals):
            raise ResidualIntelligenceError("signals must contain OODSignal values")
        if any(not isinstance(item, BoundaryViolation) for item in self.boundary_violations):
            raise ResidualIntelligenceError(
                "boundary_violations must contain BoundaryViolation values"
            )
        object.__setattr__(self, "disposition", ExpertDisposition(self.disposition))
        for field in (
            "eligible",
            "in_boundary",
            "family_in_boundary",
            "schema_in_boundary",
            "effect_in_boundary",
            "authority_in_boundary",
            "repository_in_boundary",
            "calibration_in_boundary",
            "capability_in_boundary",
            "context_in_boundary",
            "conservative_abstain",
            "ood_detected",
            "ood_detection_ran",
            "safety_established",
            "candidate_only",
        ):
            object.__setattr__(self, field, _require_bool(getattr(self, field), field))
        if self.candidate_only is not True:
            raise ResidualIntelligenceError("OOD assessments must remain candidate_only=true")
        if not self.ood_detection_ran and self.safety_established:
            raise ResidualIntelligenceError("missing OOD detection never establishes safety")
        object.__setattr__(
            self,
            "reason_codes",
            text_tuple(self.reason_codes, "reason_codes", max_items=128),
        )
        object.__setattr__(
            self,
            "evidence_references",
            text_tuple(self.evidence_references, "evidence_references", max_items=256),
        )

    @property
    def assessment_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "input_id": self.input_id,
            "contract_id": self.contract_id,
            "signals": [item.to_dict() for item in self.signals],
            "boundary_violations": [item.to_dict() for item in self.boundary_violations],
            "disposition": self.disposition.value,
            "eligible": self.eligible,
            "in_boundary": self.in_boundary,
            "family_in_boundary": self.family_in_boundary,
            "schema_in_boundary": self.schema_in_boundary,
            "effect_in_boundary": self.effect_in_boundary,
            "authority_in_boundary": self.authority_in_boundary,
            "repository_in_boundary": self.repository_in_boundary,
            "calibration_in_boundary": self.calibration_in_boundary,
            "capability_in_boundary": self.capability_in_boundary,
            "context_in_boundary": self.context_in_boundary,
            "conservative_abstain": self.conservative_abstain,
            "ood_detected": self.ood_detected,
            "ood_detection_ran": self.ood_detection_ran,
            "safety_established": self.safety_established,
            "reason_codes": list(self.reason_codes),
            "evidence_references": list(self.evidence_references),
            "candidate_only": True,
        }
        if include_id:
            result["assessment_id"] = self.assessment_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OODAssessment:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"assessment_id"},
            noun="OOD assessment",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            input_id=str(payload.get("input_id") or ""),
            contract_id=str(payload.get("contract_id") or ""),
            signals=tuple(OODSignal.from_dict(item) for item in payload.get("signals") or ()),
            boundary_violations=tuple(
                BoundaryViolation.from_dict(item)
                for item in payload.get("boundary_violations") or ()
            ),
            disposition=ExpertDisposition(str(payload.get("disposition") or "")),
            eligible=payload.get("eligible"),
            in_boundary=payload.get("in_boundary"),
            family_in_boundary=payload.get("family_in_boundary"),
            schema_in_boundary=payload.get("schema_in_boundary"),
            effect_in_boundary=payload.get("effect_in_boundary"),
            authority_in_boundary=payload.get("authority_in_boundary"),
            repository_in_boundary=payload.get("repository_in_boundary"),
            calibration_in_boundary=payload.get("calibration_in_boundary"),
            capability_in_boundary=payload.get("capability_in_boundary"),
            context_in_boundary=payload.get("context_in_boundary"),
            conservative_abstain=payload.get("conservative_abstain"),
            ood_detected=payload.get("ood_detected"),
            ood_detection_ran=payload.get("ood_detection_ran"),
            safety_established=payload.get("safety_established"),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evidence_references=tuple(payload.get("evidence_references") or ()),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("assessment_id") or "")
        if claimed and claimed != result.assessment_id:
            raise ResidualIntelligenceError("OOD assessment identity mismatch")
        return result


def assess_out_of_distribution(
    task_input: ResidualTaskInput,
    contract: BoundaryContract,
    *,
    observation: Mapping[str, Any] | None = None,
    reference_distribution: Mapping[str, Any] | None = None,
    corpus_admission: TrainingCorpusAdmission | None = None,
) -> OODAssessment:
    """Assess advisory OOD signals and independent conservative boundaries.

    Reference distributions require an admitted ``TrainingCorpusAdmission``.
    Compact statistics cannot contain recoverable private source. OOD remains
    advisory unless ``contract.admit_ood_policy`` is true.
    """

    if not isinstance(task_input, ResidualTaskInput):
        raise ResidualIntelligenceError("task_input must be ResidualTaskInput")
    if not isinstance(contract, BoundaryContract):
        raise ResidualIntelligenceError("contract must be BoundaryContract")
    parsed_observation = _parse_observation(observation)
    _require_admitted_reference(reference_distribution, corpus_admission)
    parsed_reference = _parse_reference_distribution(reference_distribution)

    violations, flags = _independent_boundary_checks(task_input, contract, parsed_observation)
    ranges = parsed_reference.get("feature_ranges") or dict(contract.feature_ranges or {})
    distance_threshold = int(
        parsed_reference.get(
            "family_distance_threshold_ppm",
            contract.family_distance_threshold_ppm,
        )
    )
    range_complete = False
    if parsed_observation.ood_detection_ran:
        signals, range_complete = _advisory_ood_signals(
            task_input,
            contract,
            parsed_observation,
            ranges=ranges,
            distance_threshold_ppm=distance_threshold,
        )
    else:
        signals = ()

    high_risk = contract.is_high_risk(task_input.risk_class)
    conservative_abstain = bool(
        contract.conservative_high_risk
        and high_risk
        and any(item.conservative for item in violations)
    )
    capability_violation = not flags[BoundaryCheck.CAPABILITY]
    in_boundary = all(flags.values())
    ood_detected = bool(signals)
    hard_ood = ood_detected and contract.admit_ood_policy
    eligible = in_boundary and not conservative_abstain and not hard_ood
    safety_established = bool(
        parsed_observation.ood_detection_ran
        and range_complete
        and in_boundary
        and not ood_detected
        and not conservative_abstain
        and not capability_violation
    )
    disposition = _select_disposition(
        capability_violation=capability_violation,
        conservative_abstain=conservative_abstain,
        hard_ood=hard_ood,
        hard_boundary=bool(violations),
        high_risk=high_risk,
    )
    reason_codes = []
    if not parsed_observation.ood_detection_ran:
        reason_codes.append("ood_detection_missing")
    if conservative_abstain:
        reason_codes.append("conservative_high_risk")
    if hard_ood:
        reason_codes.append("ood_policy_admitted")
    reason_codes.extend(item.reason_code for item in signals)
    reason_codes.extend(item.reason_code for item in violations)
    evidence = []
    for item in signals:
        evidence.extend(item.evidence_references)
    evidence.extend(parsed_observation.disagreement_sources)
    return OODAssessment(
        input_id=task_input.input_id,
        contract_id=contract.contract_id,
        signals=signals,
        boundary_violations=violations,
        disposition=disposition,
        eligible=eligible,
        in_boundary=in_boundary,
        family_in_boundary=flags[BoundaryCheck.FAMILY],
        schema_in_boundary=flags[BoundaryCheck.SCHEMA],
        effect_in_boundary=flags[BoundaryCheck.EFFECT],
        authority_in_boundary=flags[BoundaryCheck.AUTHORITY],
        repository_in_boundary=flags[BoundaryCheck.REPOSITORY],
        calibration_in_boundary=flags[BoundaryCheck.CALIBRATION],
        capability_in_boundary=flags[BoundaryCheck.CAPABILITY],
        context_in_boundary=flags[BoundaryCheck.CONTEXT],
        conservative_abstain=conservative_abstain,
        ood_detected=ood_detected,
        ood_detection_ran=parsed_observation.ood_detection_ran,
        safety_established=safety_established,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        evidence_references=tuple(dict.fromkeys(evidence)),
    )


__all__ = (
    "BOUNDARY_CONTRACT_SCHEMA",
    "OOD_ASSESSMENT_SCHEMA",
    "OOD_SIGNAL_SCHEMA",
    "BoundaryCheck",
    "BoundaryContract",
    "BoundaryViolation",
    "OODAssessment",
    "OODSignal",
    "OODSignalKind",
    "assess_out_of_distribution",
)
