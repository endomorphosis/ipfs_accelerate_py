"""Bounded advisory OOD signals and independent conservative boundary checks.

OOD signals are advisory unless a caller-supplied policy admits them.  Family,
schema, effect, authority, repository, calibration, capability, and context
checks run independently and never short-circuit.  High-risk unknown or
missing group/context conservatively abstains even when OOD stays advisory.
Missing OOD detection never establishes safety.  Known in-boundary fixtures
remain eligible for later grouped calibration; this module does not ACCEPT.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .calibration import CalibrationGroup, MAX_CALIBRATION_EXAMPLES, MAX_SCORE_PPM
from .contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TrainingAvailability,
    bounded_int,
    bounded_json_mapping,
    canonical_id,
    reject_secret_material,
    required_text,
    strict_fields,
    text_tuple,
)
from .inventory import ResidualFamilyBoundary
from .residual_ir import ResidualTaskInput
from .rights import TrainingCorpusAdmission

OOD_SIGNAL_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-ood-signal@1"
OOD_ASSESSMENT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-ood-assessment@1"
OOD_OBSERVATION_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-ood-observation@1"
BOUNDARY_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-boundary-contract@1"
)
BOUNDARY_FINDING_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-boundary-finding@1"
FEATURE_RANGE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-feature-range@1"
REFERENCE_DISTRIBUTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-ood-reference-distribution@1"
)
MAX_OOD_SIGNALS: Final = 64
MAX_FEATURE_RANGES: Final = 256
MAX_FEATURE_ABS: Final = 1_000_000_000
MAX_REFERENCE_EXAMPLES: Final = MAX_CALIBRATION_EXAMPLES
CANDIDATE_ONLY_AUTHORITY: Final = "candidate_only"
HIGH_RISK_CLASSES: Final[frozenset[RiskClass]] = frozenset(
    {RiskClass.R3, RiskClass.R4, RiskClass.R5}
)
_FORBIDDEN_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "private_body",
        "raw_body",
        "hidden_test_body",
        "source_text",
        "prompt_text",
        "completion_text",
        "chain_of_thought",
    }
)

REASON_FEATURE_RANGE: Final = "feature_range"
REASON_FAMILY_DISTANCE: Final = "family_distance"
REASON_UNKNOWN_SCHEMA: Final = "unknown_schema"
REASON_UNKNOWN_OPERATION: Final = "unknown_operation"
REASON_UNKNOWN_REPOSITORY: Final = "unknown_repository"
REASON_UNSEEN_EFFECT: Final = "unseen_effect"
REASON_UNSEEN_AUTHORITY: Final = "unseen_authority"
REASON_DISAGREEMENT: Final = "disagreement"
REASON_CALIBRATION_ABSENCE: Final = "calibration_absence"
REASON_CONTEXT_INCOMPLETE: Final = "context_incomplete"
REASON_CAPABILITY_UNAVAILABLE: Final = "capability_unavailable"
REASON_MISSING_DETECTION: Final = "missing_ood_detection"
REASON_FAMILY_BOUNDARY: Final = "family_boundary"
REASON_HIGH_RISK_UNKNOWN: Final = "high_risk_unknown"
REASON_HIGH_RISK_MISSING_GROUP: Final = "high_risk_missing_group"
REASON_HIGH_RISK_INCOMPLETE_CONTEXT: Final = "high_risk_incomplete_context"
REASON_AUTHORITY_CONTRACT: Final = "authority_must_remain_candidate_only"
REASON_IN_BOUNDARY: Final = "in_boundary_eligible"
REASON_ADVISORY_ONLY: Final = "ood_advisory_unless_policy_admits"
REASON_SAFETY_NOT_ESTABLISHED: Final = "missing_ood_detection_never_establishes_safety"
REASON_BOUNDARY_INCONSISTENT: Final = "boundary_inconsistent_with_reference"


class OODSignalKind(str, Enum):
    FEATURE_RANGE = "feature_range"
    FAMILY_DISTANCE = "family_distance"
    UNKNOWN_SCHEMA = "unknown_schema"
    UNKNOWN_OPERATION = "unknown_operation"
    UNKNOWN_REPOSITORY = "unknown_repository"
    UNSEEN_EFFECT = "unseen_effect"
    UNSEEN_AUTHORITY = "unseen_authority"
    DISAGREEMENT = "disagreement"
    CALIBRATION_ABSENCE = "calibration_absence"
    CONTEXT_INCOMPLETE = "context_incomplete"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    MISSING_DETECTION = "missing_detection"
    FAMILY_BOUNDARY = "family_boundary"


class BoundaryAxis(str, Enum):
    FAMILY = "family"
    SCHEMA = "schema"
    EFFECT = "effect"
    AUTHORITY = "authority"
    REPOSITORY = "repository"
    CALIBRATION = "calibration"
    CAPABILITY = "capability"
    CONTEXT = "context"


BOUNDARY_AXES: Final[tuple[BoundaryAxis, ...]] = (
    BoundaryAxis.FAMILY,
    BoundaryAxis.SCHEMA,
    BoundaryAxis.EFFECT,
    BoundaryAxis.AUTHORITY,
    BoundaryAxis.REPOSITORY,
    BoundaryAxis.CALIBRATION,
    BoundaryAxis.CAPABILITY,
    BoundaryAxis.CONTEXT,
)


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _looks_like_private_body(value: str) -> bool:
    lowered = value.casefold()
    if lowered in _FORBIDDEN_BODY_MARKERS:
        return True
    head = lowered.split(":", 1)[0].split("/", 1)[0]
    if head in _FORBIDDEN_BODY_MARKERS:
        return True
    return any(
        lowered.startswith(marker + ":") or lowered.startswith(marker + "/")
        for marker in _FORBIDDEN_BODY_MARKERS
    )


def _reject_private_identities(values: Sequence[str], *, noun: str) -> tuple[str, ...]:
    result = text_tuple(values, noun)
    if any(_looks_like_private_body(item) for item in result):
        raise ResidualIntelligenceError(
            f"{noun} exposes a private body rather than a content identity"
        )
    return result


def _axis_text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value in (None, ""):
        return ""
    text = required_text(value, name, max_bytes=256)
    if _looks_like_private_body(text):
        raise ResidualIntelligenceError(
            f"{name} exposes a private body rather than a content identity"
        )
    return text


def _identity_tuple(values: Any, name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    return _reject_private_identities(
        text_tuple(values, name, allow_empty=allow_empty),
        noun=name,
    )


def _ppm(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=0, maximum=MAX_SCORE_PPM)


def _feature_int(value: Any, name: str) -> int:
    return bounded_int(value, name, minimum=-MAX_FEATURE_ABS, maximum=MAX_FEATURE_ABS)


def _families(values: Any, name: str, *, allow_empty: bool = False) -> tuple[ResidualTaskFamily, ...]:
    labels = text_tuple(values, name, allow_empty=allow_empty)
    result = tuple(ResidualTaskFamily(item) for item in labels)
    if len(set(result)) != len(result):
        raise ResidualIntelligenceError(f"{name} contains duplicate values")
    return result


def _subset_unknown(observed: Sequence[str], allowed: frozenset[str] | None) -> tuple[str, ...]:
    if allowed is None:
        return tuple(observed) if observed else ("missing",)
    return tuple(item for item in observed if item not in allowed)


@dataclass(frozen=True)
class FeatureRange:
    """Compact integer or token range for one named feature.

    Ranges are admitted statistics.  They never store recoverable source.
    """

    name: str
    minimum: int = 0
    maximum: int = 0
    allowed_values: tuple[str, ...] = ()
    observed_count: int = 0
    schema: str = FEATURE_RANGE_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "range_id",
            "name",
            "minimum",
            "maximum",
            "allowed_values",
            "observed_count",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != FEATURE_RANGE_SCHEMA:
            raise ResidualIntelligenceError("unsupported feature range schema")
        object.__setattr__(self, "name", _axis_text(self.name, "name"))
        object.__setattr__(self, "minimum", _feature_int(self.minimum, "minimum"))
        object.__setattr__(self, "maximum", _feature_int(self.maximum, "maximum"))
        if self.minimum > self.maximum:
            raise ResidualIntelligenceError("feature range bounds are inverted")
        object.__setattr__(
            self,
            "allowed_values",
            _identity_tuple(self.allowed_values, "allowed_values"),
        )
        object.__setattr__(
            self,
            "observed_count",
            bounded_int(
                self.observed_count,
                "observed_count",
                minimum=0,
                maximum=MAX_REFERENCE_EXAMPLES,
            ),
        )

    @property
    def categorical(self) -> bool:
        return bool(self.allowed_values)

    @property
    def range_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def covers(self, value: Any) -> bool:
        if self.categorical:
            return isinstance(value, str) and value in self.allowed_values
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return self.minimum <= value <= self.maximum

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "allowed_values": list(self.allowed_values),
            "observed_count": self.observed_count,
        }
        if include_id:
            result["range_id"] = self.range_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureRange:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"range_id", "allowed_values", "observed_count"},
            noun="feature range",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            name=str(payload.get("name") or ""),
            minimum=payload.get("minimum", 0),
            maximum=payload.get("maximum", 0),
            allowed_values=tuple(payload.get("allowed_values") or ()),
            observed_count=payload.get("observed_count", 0),
        )
        claimed = str(payload.get("range_id") or "")
        if claimed and claimed != result.range_id:
            raise ResidualIntelligenceError("feature range identity mismatch")
        return result


@dataclass(frozen=True)
class BoundaryContract:
    """Independent conservative envelope for the eight residual boundary axes."""

    family: ResidualTaskFamily
    schema: str
    effects: tuple[str, ...]
    authority_class: str
    repository: str
    calibration_group_key: str
    capabilities: tuple[str, ...]
    required_context_fields: tuple[str, ...]
    risk_ceiling: RiskClass
    candidate_only: bool = True
    schema_name: str = BOUNDARY_CONTRACT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "contract_id",
            "family",
            "task_schema",
            "effects",
            "authority_class",
            "repository",
            "calibration_group_key",
            "capabilities",
            "required_context_fields",
            "risk_ceiling",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_name != BOUNDARY_CONTRACT_SCHEMA:
            raise ResidualIntelligenceError("unsupported boundary contract schema")
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        object.__setattr__(self, "schema", _axis_text(self.schema, "schema"))
        object.__setattr__(self, "effects", _identity_tuple(self.effects, "effects"))
        object.__setattr__(
            self,
            "authority_class",
            _axis_text(self.authority_class, "authority_class"),
        )
        if self.authority_class.casefold() != CANDIDATE_ONLY_AUTHORITY:
            raise ResidualIntelligenceError(
                "boundary contract authority_class must remain candidate_only"
            )
        object.__setattr__(self, "repository", _axis_text(self.repository, "repository"))
        object.__setattr__(
            self,
            "calibration_group_key",
            required_text(self.calibration_group_key, "calibration_group_key"),
        )
        object.__setattr__(
            self,
            "capabilities",
            _identity_tuple(self.capabilities, "capabilities"),
        )
        object.__setattr__(
            self,
            "required_context_fields",
            _identity_tuple(self.required_context_fields, "required_context_fields"),
        )
        object.__setattr__(self, "risk_ceiling", RiskClass(self.risk_ceiling))
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("boundary contracts must remain candidate_only=true")

    @property
    def contract_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def conservative_high_risk(self) -> bool:
        return self.risk_ceiling in HIGH_RISK_CLASSES

    @classmethod
    def from_family_boundary(
        cls,
        family_boundary: ResidualFamilyBoundary,
        *,
        schema: str,
        effects: Sequence[str] = (),
        repository: str,
        calibration_group_key: str,
        capabilities: Sequence[str] = (),
        required_context_fields: Sequence[str] = (),
    ) -> BoundaryContract:
        if not isinstance(family_boundary, ResidualFamilyBoundary):
            raise ResidualIntelligenceError(
                "from_family_boundary requires a typed ResidualFamilyBoundary"
            )
        return cls(
            family=family_boundary.task_family,
            schema=schema,
            effects=tuple(effects),
            authority_class=family_boundary.authority_class,
            repository=repository,
            calibration_group_key=calibration_group_key,
            capabilities=tuple(capabilities),
            required_context_fields=tuple(required_context_fields),
            risk_ceiling=family_boundary.risk_class,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema_name,
            "family": self.family.value,
            "task_schema": self.schema,
            "effects": list(self.effects),
            "authority_class": self.authority_class,
            "repository": self.repository,
            "calibration_group_key": self.calibration_group_key,
            "capabilities": list(self.capabilities),
            "required_context_fields": list(self.required_context_fields),
            "risk_ceiling": self.risk_ceiling.value,
            "candidate_only": True,
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
            schema_name=str(payload.get("schema") or ""),
            family=ResidualTaskFamily(str(payload.get("family") or "")),
            schema=str(payload.get("task_schema") or ""),
            effects=tuple(payload.get("effects") or ()),
            authority_class=str(payload.get("authority_class") or ""),
            repository=str(payload.get("repository") or ""),
            calibration_group_key=str(payload.get("calibration_group_key") or ""),
            capabilities=tuple(payload.get("capabilities") or ()),
            required_context_fields=tuple(payload.get("required_context_fields") or ()),
            risk_ceiling=RiskClass(str(payload.get("risk_ceiling") or "")),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("contract_id") or "")
        if claimed and claimed != result.contract_id:
            raise ResidualIntelligenceError("boundary contract identity mismatch")
        return result


@dataclass(frozen=True)
class ReferenceDistribution:
    """Admitted compact reference for advisory OOD statistics.

    Construction requires an admitted TrainingCorpusAdmission.  Compact
    statistics are counts and identities only; private bodies are rejected.
    """

    group: CalibrationGroup
    admission_id: str
    admission_decision: TrainingAvailability
    allowed_families: tuple[ResidualTaskFamily, ...]
    allowed_schemas: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    allowed_repositories: tuple[str, ...]
    allowed_effects: tuple[str, ...]
    allowed_authorities: tuple[str, ...]
    allowed_capabilities: tuple[str, ...]
    required_context_fields: tuple[str, ...]
    feature_ranges: tuple[FeatureRange, ...]
    example_identities: tuple[str, ...]
    statistic_identities: tuple[str, ...] = ()
    compact_statistics: Mapping[str, int] = None  # type: ignore[assignment]
    family_distance_threshold_ppm: int = 0
    schema: str = REFERENCE_DISTRIBUTION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "distribution_id",
            "group",
            "admission_id",
            "admission_decision",
            "allowed_families",
            "allowed_schemas",
            "allowed_operations",
            "allowed_repositories",
            "allowed_effects",
            "allowed_authorities",
            "allowed_capabilities",
            "required_context_fields",
            "feature_ranges",
            "example_identities",
            "statistic_identities",
            "compact_statistics",
            "family_distance_threshold_ppm",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_DISTRIBUTION_SCHEMA:
            raise ResidualIntelligenceError("unsupported reference distribution schema")
        if not isinstance(self.group, CalibrationGroup):
            raise ResidualIntelligenceError("reference distribution requires a typed calibration group")
        object.__setattr__(self, "admission_id", required_text(self.admission_id, "admission_id"))
        object.__setattr__(
            self,
            "admission_decision",
            TrainingAvailability(self.admission_decision),
        )
        if self.admission_decision is not TrainingAvailability.ADMITTED:
            raise ResidualIntelligenceError(
                "reference distributions require an admitted TrainingCorpusAdmission"
            )
        families = _families(tuple(item.value for item in self.allowed_families), "allowed_families")
        object.__setattr__(self, "allowed_families", families)
        object.__setattr__(
            self,
            "allowed_schemas",
            _identity_tuple(self.allowed_schemas, "allowed_schemas", allow_empty=False),
        )
        object.__setattr__(
            self,
            "allowed_operations",
            _identity_tuple(self.allowed_operations, "allowed_operations"),
        )
        object.__setattr__(
            self,
            "allowed_repositories",
            _identity_tuple(self.allowed_repositories, "allowed_repositories", allow_empty=False),
        )
        object.__setattr__(
            self,
            "allowed_effects",
            _identity_tuple(self.allowed_effects, "allowed_effects"),
        )
        authorities = _identity_tuple(
            self.allowed_authorities,
            "allowed_authorities",
            allow_empty=False,
        )
        if any(item.casefold() != CANDIDATE_ONLY_AUTHORITY for item in authorities):
            raise ResidualIntelligenceError(
                "reference distribution authority must remain candidate_only"
            )
        object.__setattr__(self, "allowed_authorities", authorities)
        object.__setattr__(
            self,
            "allowed_capabilities",
            _identity_tuple(self.allowed_capabilities, "allowed_capabilities"),
        )
        object.__setattr__(
            self,
            "required_context_fields",
            _identity_tuple(self.required_context_fields, "required_context_fields"),
        )
        ranges = tuple(self.feature_ranges)
        if len(ranges) > MAX_FEATURE_RANGES:
            raise ResidualIntelligenceError("feature_ranges exceeds bound")
        if any(not isinstance(item, FeatureRange) for item in ranges):
            raise ResidualIntelligenceError("feature_ranges must be typed FeatureRange")
        names = [item.name for item in ranges]
        if len(set(names)) != len(names):
            raise ResidualIntelligenceError("feature_ranges contains duplicate names")
        object.__setattr__(self, "feature_ranges", ranges)
        identities = _identity_tuple(
            self.example_identities,
            "example_identities",
            allow_empty=False,
        )
        if len(identities) > MAX_REFERENCE_EXAMPLES:
            raise ResidualIntelligenceError("reference distribution exceeds example bound")
        object.__setattr__(self, "example_identities", identities)
        object.__setattr__(
            self,
            "statistic_identities",
            _identity_tuple(self.statistic_identities, "statistic_identities"),
        )
        stats = self.compact_statistics if self.compact_statistics is not None else {}
        normalized = bounded_json_mapping(stats, "compact_statistics")
        reject_secret_material(normalized, noun="compact_statistics")
        compact: dict[str, int] = {}
        for key, value in normalized.items():
            if _looks_like_private_body(key):
                raise ResidualIntelligenceError(
                    "compact_statistics exposes a private body rather than a content identity"
                )
            compact[key] = bounded_int(
                value,
                f"compact_statistics.{key}",
                minimum=0,
                maximum=MAX_FEATURE_ABS,
            )
        object.__setattr__(self, "compact_statistics", compact)
        object.__setattr__(
            self,
            "family_distance_threshold_ppm",
            _ppm(self.family_distance_threshold_ppm, "family_distance_threshold_ppm"),
        )

    @property
    def distribution_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def range_by_name(self) -> dict[str, FeatureRange]:
        return {item.name: item for item in self.feature_ranges}

    def validate_against_admission(self, admission: TrainingCorpusAdmission) -> None:
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("admission must be TrainingCorpusAdmission")
        if not admission.can_train:
            raise ResidualIntelligenceError(
                "reference distributions require an admitted TrainingCorpusAdmission"
            )
        if self.admission_id != admission.admission_id:
            raise ResidualIntelligenceError("reference distribution admission_id mismatch")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "group": self.group.to_dict(),
            "admission_id": self.admission_id,
            "admission_decision": self.admission_decision.value,
            "allowed_families": [item.value for item in self.allowed_families],
            "allowed_schemas": list(self.allowed_schemas),
            "allowed_operations": list(self.allowed_operations),
            "allowed_repositories": list(self.allowed_repositories),
            "allowed_effects": list(self.allowed_effects),
            "allowed_authorities": list(self.allowed_authorities),
            "allowed_capabilities": list(self.allowed_capabilities),
            "required_context_fields": list(self.required_context_fields),
            "feature_ranges": [item.to_dict() for item in self.feature_ranges],
            "example_identities": list(self.example_identities),
            "statistic_identities": list(self.statistic_identities),
            "compact_statistics": dict(self.compact_statistics),
            "family_distance_threshold_ppm": self.family_distance_threshold_ppm,
        }
        if include_id:
            result["distribution_id"] = self.distribution_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReferenceDistribution:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {
                "distribution_id",
                "statistic_identities",
                "compact_statistics",
                "family_distance_threshold_ppm",
                "feature_ranges",
                "allowed_operations",
                "allowed_effects",
                "allowed_capabilities",
                "required_context_fields",
            },
            noun="reference distribution",
        )
        group_payload = payload.get("group")
        if not isinstance(group_payload, Mapping):
            raise ResidualIntelligenceError("reference distribution group must be an object")
        range_payload = payload.get("feature_ranges") or ()
        if isinstance(range_payload, (str, bytes, bytearray)) or not isinstance(
            range_payload, Sequence
        ):
            raise ResidualIntelligenceError("feature_ranges must be a sequence")
        result = cls(
            schema=str(payload.get("schema") or ""),
            group=CalibrationGroup.from_dict(group_payload),
            admission_id=str(payload.get("admission_id") or ""),
            admission_decision=TrainingAvailability(str(payload.get("admission_decision") or "")),
            allowed_families=tuple(
                ResidualTaskFamily(item) for item in (payload.get("allowed_families") or ())
            ),
            allowed_schemas=tuple(payload.get("allowed_schemas") or ()),
            allowed_operations=tuple(payload.get("allowed_operations") or ()),
            allowed_repositories=tuple(payload.get("allowed_repositories") or ()),
            allowed_effects=tuple(payload.get("allowed_effects") or ()),
            allowed_authorities=tuple(payload.get("allowed_authorities") or ()),
            allowed_capabilities=tuple(payload.get("allowed_capabilities") or ()),
            required_context_fields=tuple(payload.get("required_context_fields") or ()),
            feature_ranges=tuple(FeatureRange.from_dict(item) for item in range_payload),
            example_identities=tuple(payload.get("example_identities") or ()),
            statistic_identities=tuple(payload.get("statistic_identities") or ()),
            compact_statistics=payload.get("compact_statistics") or {},
            family_distance_threshold_ppm=payload.get("family_distance_threshold_ppm", 0),
        )
        claimed = str(payload.get("distribution_id") or "")
        if claimed and claimed != result.distribution_id:
            raise ResidualIntelligenceError("reference distribution identity mismatch")
        return result


@dataclass(frozen=True)
class OODObservation:
    """One bounded probe presented to advisory OOD and hard boundary checks."""

    risk_class: RiskClass
    family: ResidualTaskFamily | None = None
    schema: str = ""
    operation: str = ""
    repository: str = ""
    effects: tuple[str, ...] = ()
    authority_class: str = CANDIDATE_ONLY_AUTHORITY
    features: Mapping[str, Any] = None  # type: ignore[assignment]
    calibration_group_key: str = ""
    context_fields: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    capability_available: bool = True
    disagreement: bool = False
    disagreement_identities: tuple[str, ...] = ()
    family_distance_ppm: int = 0
    detection_available: bool = True
    context_complete: bool = True
    schema_name: str = OOD_OBSERVATION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "observation_id",
            "risk_class",
            "family",
            "task_schema",
            "operation",
            "repository",
            "effects",
            "authority_class",
            "features",
            "calibration_group_key",
            "context_fields",
            "capabilities",
            "capability_available",
            "disagreement",
            "disagreement_identities",
            "family_distance_ppm",
            "detection_available",
            "context_complete",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_name != OOD_OBSERVATION_SCHEMA:
            raise ResidualIntelligenceError("unsupported OOD observation schema")
        object.__setattr__(self, "risk_class", RiskClass(self.risk_class))
        if self.family is not None:
            object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        object.__setattr__(self, "schema", _axis_text(self.schema, "schema", allow_empty=True))
        object.__setattr__(
            self,
            "operation",
            _axis_text(self.operation, "operation", allow_empty=True),
        )
        object.__setattr__(
            self,
            "repository",
            _axis_text(self.repository, "repository", allow_empty=True),
        )
        object.__setattr__(self, "effects", _identity_tuple(self.effects, "effects"))
        object.__setattr__(
            self,
            "authority_class",
            _axis_text(self.authority_class, "authority_class", allow_empty=True)
            or CANDIDATE_ONLY_AUTHORITY,
        )
        features = bounded_json_mapping(
            self.features if self.features is not None else {},
            "features",
        )
        reject_secret_material(features, noun="features")
        for key in features:
            if _looks_like_private_body(key):
                raise ResidualIntelligenceError(
                    "features expose a private body rather than a content identity"
                )
        object.__setattr__(self, "features", features)
        object.__setattr__(
            self,
            "calibration_group_key",
            ""
            if self.calibration_group_key in (None, "")
            else required_text(self.calibration_group_key, "calibration_group_key"),
        )
        object.__setattr__(
            self,
            "context_fields",
            _identity_tuple(self.context_fields, "context_fields"),
        )
        object.__setattr__(
            self,
            "capabilities",
            _identity_tuple(self.capabilities, "capabilities"),
        )
        for field in (
            "capability_available",
            "disagreement",
            "detection_available",
            "context_complete",
        ):
            object.__setattr__(self, field, _require_bool(getattr(self, field), field))
        object.__setattr__(
            self,
            "disagreement_identities",
            _identity_tuple(self.disagreement_identities, "disagreement_identities"),
        )
        object.__setattr__(
            self,
            "family_distance_ppm",
            _ppm(self.family_distance_ppm, "family_distance_ppm"),
        )

    @property
    def observation_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def high_risk(self) -> bool:
        return self.risk_class in HIGH_RISK_CLASSES

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema_name,
            "risk_class": self.risk_class.value,
            "family": None if self.family is None else self.family.value,
            "task_schema": self.schema,
            "operation": self.operation,
            "repository": self.repository,
            "effects": list(self.effects),
            "authority_class": self.authority_class,
            "features": dict(self.features),
            "calibration_group_key": self.calibration_group_key,
            "context_fields": list(self.context_fields),
            "capabilities": list(self.capabilities),
            "capability_available": self.capability_available,
            "disagreement": self.disagreement,
            "disagreement_identities": list(self.disagreement_identities),
            "family_distance_ppm": self.family_distance_ppm,
            "detection_available": self.detection_available,
            "context_complete": self.context_complete,
        }
        if include_id:
            result["observation_id"] = self.observation_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OODObservation:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS
            - {
                "observation_id",
                "family",
                "task_schema",
                "operation",
                "repository",
                "effects",
                "authority_class",
                "features",
                "calibration_group_key",
                "context_fields",
                "capabilities",
                "disagreement_identities",
            },
            noun="OOD observation",
        )
        family_raw = payload.get("family")
        result = cls(
            schema_name=str(payload.get("schema") or ""),
            risk_class=RiskClass(str(payload.get("risk_class") or "")),
            family=None if family_raw in (None, "") else ResidualTaskFamily(str(family_raw)),
            schema=str(payload.get("task_schema") or ""),
            operation=str(payload.get("operation") or ""),
            repository=str(payload.get("repository") or ""),
            effects=tuple(payload.get("effects") or ()),
            authority_class=str(payload.get("authority_class") or CANDIDATE_ONLY_AUTHORITY),
            features=payload.get("features") or {},
            calibration_group_key=str(payload.get("calibration_group_key") or ""),
            context_fields=tuple(payload.get("context_fields") or ()),
            capabilities=tuple(payload.get("capabilities") or ()),
            capability_available=payload.get("capability_available"),
            disagreement=payload.get("disagreement"),
            disagreement_identities=tuple(payload.get("disagreement_identities") or ()),
            family_distance_ppm=payload.get("family_distance_ppm", 0),
            detection_available=payload.get("detection_available"),
            context_complete=payload.get("context_complete"),
        )
        claimed = str(payload.get("observation_id") or "")
        if claimed and claimed != result.observation_id:
            raise ResidualIntelligenceError("OOD observation identity mismatch")
        return result


def observation_from_task_input(
    task_input: ResidualTaskInput,
    **overrides: Any,
) -> OODObservation:
    """Project a compact residual task input into an OOD observation."""

    if not isinstance(task_input, ResidualTaskInput):
        raise ResidualIntelligenceError("observation_from_task_input requires ResidualTaskInput")
    payload: dict[str, Any] = {
        "risk_class": task_input.risk_class,
        "family": task_input.task_family,
        "schema": task_input.schema,
        "repository": task_input.repository_state_cid,
        "features": dict(task_input.compact_features),
        "context_fields": (task_input.context_capsule_cid,) if task_input.context_capsule_cid else (),
        "context_complete": bool(task_input.context_capsule_cid),
    }
    payload.update(overrides)
    return OODObservation(**payload)


@dataclass(frozen=True)
class OODSignal:
    """One bounded advisory out-of-distribution signal.

    Signals never grant safety, completion, or routing authority.  Binding
    them as ``OUT_OF_DISTRIBUTION`` is a policy decision outside this record.
    """

    kind: OODSignalKind
    reason_code: str
    score_ppm: int = MAX_SCORE_PPM
    evidence_identities: tuple[str, ...] = ()
    advisory: bool = True
    candidate_only: bool = True
    schema: str = OOD_SIGNAL_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "signal_id",
            "kind",
            "reason_code",
            "score_ppm",
            "evidence_identities",
            "advisory",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != OOD_SIGNAL_SCHEMA:
            raise ResidualIntelligenceError("unsupported OOD signal schema")
        object.__setattr__(self, "kind", OODSignalKind(self.kind))
        object.__setattr__(self, "reason_code", required_text(self.reason_code, "reason_code"))
        object.__setattr__(self, "score_ppm", _ppm(self.score_ppm, "score_ppm"))
        object.__setattr__(
            self,
            "evidence_identities",
            _identity_tuple(self.evidence_identities, "evidence_identities"),
        )
        if type(self.advisory) is not bool or self.advisory is not True:
            raise ResidualIntelligenceError(
                "OOD signals remain advisory unless policy admits them"
            )
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("OOD signals must remain candidate_only=true")

    @property
    def signal_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind.value,
            "reason_code": self.reason_code,
            "score_ppm": self.score_ppm,
            "evidence_identities": list(self.evidence_identities),
            "advisory": True,
            "candidate_only": True,
        }
        if include_id:
            result["signal_id"] = self.signal_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OODSignal:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"signal_id", "evidence_identities"},
            noun="OOD signal",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            kind=OODSignalKind(str(payload.get("kind") or "")),
            reason_code=str(payload.get("reason_code") or ""),
            score_ppm=payload.get("score_ppm", MAX_SCORE_PPM),
            evidence_identities=tuple(payload.get("evidence_identities") or ()),
            advisory=payload.get("advisory"),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("signal_id") or "")
        if claimed and claimed != result.signal_id:
            raise ResidualIntelligenceError("OOD signal identity mismatch")
        return result


@dataclass(frozen=True)
class BoundaryFinding:
    """Result of one independent conservative boundary axis check."""

    axis: BoundaryAxis
    passed: bool
    unknown_or_missing: bool
    conservative_abstain: bool
    reason_codes: tuple[str, ...]
    schema: str = BOUNDARY_FINDING_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "finding_id",
            "axis",
            "passed",
            "unknown_or_missing",
            "conservative_abstain",
            "reason_codes",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != BOUNDARY_FINDING_SCHEMA:
            raise ResidualIntelligenceError("unsupported boundary finding schema")
        object.__setattr__(self, "axis", BoundaryAxis(self.axis))
        for field in ("passed", "unknown_or_missing", "conservative_abstain"):
            object.__setattr__(self, field, _require_bool(getattr(self, field), field))
        object.__setattr__(
            self,
            "reason_codes",
            text_tuple(self.reason_codes, "reason_codes", max_items=16),
        )
        if self.passed and self.conservative_abstain:
            raise ResidualIntelligenceError("a passing boundary cannot conservatively abstain")
        if self.passed and self.reason_codes:
            raise ResidualIntelligenceError("a passing boundary cannot carry failure reason codes")
        if not self.passed and not self.reason_codes:
            raise ResidualIntelligenceError("a failed boundary requires at least one reason code")

    @property
    def finding_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "axis": self.axis.value,
            "passed": self.passed,
            "unknown_or_missing": self.unknown_or_missing,
            "conservative_abstain": self.conservative_abstain,
            "reason_codes": list(self.reason_codes),
        }
        if include_id:
            result["finding_id"] = self.finding_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BoundaryFinding:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"finding_id"},
            noun="boundary finding",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            axis=BoundaryAxis(str(payload.get("axis") or "")),
            passed=payload.get("passed"),
            unknown_or_missing=payload.get("unknown_or_missing"),
            conservative_abstain=payload.get("conservative_abstain"),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        claimed = str(payload.get("finding_id") or "")
        if claimed and claimed != result.finding_id:
            raise ResidualIntelligenceError("boundary finding identity mismatch")
        return result


@dataclass(frozen=True)
class OODAssessment:
    """Combined advisory OOD receipt and independent boundary-gate receipt.

    ``safety_established`` is always false: OOD never proves safety, and
    missing detection cannot be treated as an in-distribution result.
    """

    signals: tuple[OODSignal, ...]
    boundary_findings: tuple[BoundaryFinding, ...]
    risk_class: RiskClass
    advisory_ood: bool
    policy_admits_ood: bool
    bound_ood: bool
    conservative_abstain: bool
    in_boundary_eligible: bool
    detection_available: bool
    reason_codes: tuple[str, ...]
    forced_disposition: ExpertDisposition | None = None
    safety_established: bool = False
    candidate_only: bool = True
    schema: str = OOD_ASSESSMENT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "assessment_id",
            "signals",
            "boundary_findings",
            "risk_class",
            "advisory_ood",
            "policy_admits_ood",
            "bound_ood",
            "conservative_abstain",
            "in_boundary_eligible",
            "detection_available",
            "reason_codes",
            "forced_disposition",
            "safety_established",
            "candidate_only",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != OOD_ASSESSMENT_SCHEMA:
            raise ResidualIntelligenceError("unsupported OOD assessment schema")
        signals = tuple(self.signals)
        if len(signals) > MAX_OOD_SIGNALS:
            raise ResidualIntelligenceError("OOD assessment exceeds signal bound")
        if any(not isinstance(item, OODSignal) for item in signals):
            raise ResidualIntelligenceError("assessment signals must be typed OODSignal")
        object.__setattr__(self, "signals", signals)
        findings = tuple(self.boundary_findings)
        if len(findings) != len(BOUNDARY_AXES):
            raise ResidualIntelligenceError("assessment must record every independent boundary axis")
        observed_axes = tuple(item.axis for item in findings)
        if observed_axes != BOUNDARY_AXES:
            raise ResidualIntelligenceError("boundary findings must follow the closed axis order")
        if any(not isinstance(item, BoundaryFinding) for item in findings):
            raise ResidualIntelligenceError("boundary findings must be typed BoundaryFinding")
        object.__setattr__(self, "boundary_findings", findings)
        object.__setattr__(self, "risk_class", RiskClass(self.risk_class))
        for field in (
            "advisory_ood",
            "policy_admits_ood",
            "bound_ood",
            "conservative_abstain",
            "in_boundary_eligible",
            "detection_available",
        ):
            object.__setattr__(self, field, _require_bool(getattr(self, field), field))
        if type(self.safety_established) is not bool or self.safety_established is not False:
            raise ResidualIntelligenceError("OOD assessment cannot establish safety")
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("OOD assessments must remain candidate_only=true")
        if self.bound_ood and not (self.advisory_ood and self.policy_admits_ood):
            raise ResidualIntelligenceError("bound OOD requires advisory signals and policy admission")
        if self.in_boundary_eligible and (
            self.advisory_ood or self.conservative_abstain or not self.detection_available
        ):
            raise ResidualIntelligenceError(
                "in-boundary eligibility requires detection and a clean envelope"
            )
        if self.forced_disposition is not None:
            object.__setattr__(
                self,
                "forced_disposition",
                ExpertDisposition(self.forced_disposition),
            )
            if self.forced_disposition is ExpertDisposition.ACCEPT:
                raise ResidualIntelligenceError("OOD assessment cannot ACCEPT")
        object.__setattr__(
            self,
            "reason_codes",
            text_tuple(self.reason_codes, "reason_codes", allow_empty=False, max_items=32),
        )

    @property
    def assessment_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def finding(self, axis: BoundaryAxis) -> BoundaryFinding:
        return self.boundary_findings[BOUNDARY_AXES.index(axis)]

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "signals": [item.to_dict() for item in self.signals],
            "boundary_findings": [item.to_dict() for item in self.boundary_findings],
            "risk_class": self.risk_class.value,
            "advisory_ood": self.advisory_ood,
            "policy_admits_ood": self.policy_admits_ood,
            "bound_ood": self.bound_ood,
            "conservative_abstain": self.conservative_abstain,
            "in_boundary_eligible": self.in_boundary_eligible,
            "detection_available": self.detection_available,
            "reason_codes": list(self.reason_codes),
            "forced_disposition": (
                None if self.forced_disposition is None else self.forced_disposition.value
            ),
            "safety_established": False,
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
            required=cls._FIELDS - {"assessment_id", "forced_disposition"},
            noun="OOD assessment",
        )
        signal_payload = payload.get("signals")
        finding_payload = payload.get("boundary_findings")
        if isinstance(signal_payload, (str, bytes, bytearray)) or not isinstance(
            signal_payload, Sequence
        ):
            raise ResidualIntelligenceError("OOD assessment signals must be a sequence")
        if isinstance(finding_payload, (str, bytes, bytearray)) or not isinstance(
            finding_payload, Sequence
        ):
            raise ResidualIntelligenceError("OOD assessment boundary_findings must be a sequence")
        forced = payload.get("forced_disposition")
        result = cls(
            schema=str(payload.get("schema") or ""),
            signals=tuple(OODSignal.from_dict(item) for item in signal_payload),
            boundary_findings=tuple(BoundaryFinding.from_dict(item) for item in finding_payload),
            risk_class=RiskClass(str(payload.get("risk_class") or "")),
            advisory_ood=payload.get("advisory_ood"),
            policy_admits_ood=payload.get("policy_admits_ood"),
            bound_ood=payload.get("bound_ood"),
            conservative_abstain=payload.get("conservative_abstain"),
            in_boundary_eligible=payload.get("in_boundary_eligible"),
            detection_available=payload.get("detection_available"),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            forced_disposition=None if forced in (None, "") else ExpertDisposition(str(forced)),
            safety_established=payload.get("safety_established", False),
            candidate_only=payload.get("candidate_only"),
        )
        claimed = str(payload.get("assessment_id") or "")
        if claimed and claimed != result.assessment_id:
            raise ResidualIntelligenceError("OOD assessment identity mismatch")
        return result


def _signal(
    kind: OODSignalKind,
    reason: str,
    *,
    score_ppm: int = MAX_SCORE_PPM,
    evidence: Sequence[str] = (),
) -> OODSignal:
    return OODSignal(
        kind=kind,
        reason_code=reason,
        score_ppm=score_ppm,
        evidence_identities=tuple(evidence),
    )


def _finding(
    axis: BoundaryAxis,
    *,
    passed: bool,
    unknown_or_missing: bool,
    risk: RiskClass,
    reasons: Sequence[str] = (),
    authority_contract_broken: bool = False,
) -> BoundaryFinding:
    conservative = False
    extra: list[str] = []
    if not passed:
        if authority_contract_broken:
            conservative = True
            extra.append(REASON_AUTHORITY_CONTRACT)
        if risk in HIGH_RISK_CLASSES:
            conservative = True
            extra.append(REASON_HIGH_RISK_UNKNOWN)
            if axis is BoundaryAxis.CALIBRATION and unknown_or_missing:
                extra.append(REASON_HIGH_RISK_MISSING_GROUP)
            if axis is BoundaryAxis.CONTEXT and unknown_or_missing:
                extra.append(REASON_HIGH_RISK_INCOMPLETE_CONTEXT)
            if axis is BoundaryAxis.FAMILY and unknown_or_missing:
                extra.append(REASON_HIGH_RISK_MISSING_GROUP)
    codes = tuple(dict.fromkeys((*reasons, *extra)))
    return BoundaryFinding(
        axis=axis,
        passed=passed,
        unknown_or_missing=unknown_or_missing,
        conservative_abstain=conservative,
        reason_codes=codes,
    )


def _validate_boundary_against_reference(
    boundary: BoundaryContract,
    reference: ReferenceDistribution,
) -> None:
    if boundary.family not in reference.allowed_families:
        raise ResidualIntelligenceError(REASON_BOUNDARY_INCONSISTENT)
    if boundary.schema not in reference.allowed_schemas:
        raise ResidualIntelligenceError(REASON_BOUNDARY_INCONSISTENT)
    if boundary.repository not in reference.allowed_repositories:
        raise ResidualIntelligenceError(REASON_BOUNDARY_INCONSISTENT)
    if boundary.calibration_group_key != reference.group.group_key:
        raise ResidualIntelligenceError(REASON_BOUNDARY_INCONSISTENT)
    unseen_effects = [item for item in boundary.effects if item not in reference.allowed_effects]
    if boundary.effects and unseen_effects:
        raise ResidualIntelligenceError(REASON_BOUNDARY_INCONSISTENT)
    if boundary.authority_class not in reference.allowed_authorities:
        raise ResidualIntelligenceError(REASON_BOUNDARY_INCONSISTENT)


def _collect_feature_range_evidence(
    observation: OODObservation,
    reference: ReferenceDistribution,
) -> tuple[str, ...]:
    ranges = reference.range_by_name()
    offenders: list[str] = []
    for name, value in observation.features.items():
        item = ranges.get(name)
        if item is None or not item.covers(value):
            offenders.append(name)
    return tuple(offenders)


def assess_out_of_distribution(
    observation: OODObservation,
    *,
    reference: ReferenceDistribution | None = None,
    boundary: BoundaryContract | None = None,
    admission: TrainingCorpusAdmission | None = None,
    policy_admits_ood: bool = False,
) -> OODAssessment:
    """Emit advisory OOD signals and independent conservative boundary findings.

    Statistical detection runs only when ``observation.detection_available``
    and an admitted reference distribution are both present.  Boundary axes
    always evaluate independently.  This function never returns ACCEPT and
    never sets ``safety_established``.
    """

    if not isinstance(observation, OODObservation):
        raise ResidualIntelligenceError("assess_out_of_distribution requires a typed OODObservation")
    policy_admits_ood = _require_bool(policy_admits_ood, "policy_admits_ood")
    if reference is not None and not isinstance(reference, ReferenceDistribution):
        raise ResidualIntelligenceError("reference must be a typed ReferenceDistribution")
    if boundary is not None and not isinstance(boundary, BoundaryContract):
        raise ResidualIntelligenceError("boundary must be a typed BoundaryContract")
    if admission is not None:
        if not isinstance(admission, TrainingCorpusAdmission):
            raise ResidualIntelligenceError("admission must be TrainingCorpusAdmission")
        if reference is not None:
            reference.validate_against_admission(admission)
    if boundary is not None and reference is not None:
        _validate_boundary_against_reference(boundary, reference)

    risk = observation.risk_class
    statistical_detection = observation.detection_available and reference is not None
    signals: list[OODSignal] = []

    allowed_families = (
        frozenset(reference.allowed_families)
        if reference is not None
        else (frozenset({boundary.family}) if boundary is not None else None)
    )
    allowed_schemas = (
        frozenset(reference.allowed_schemas)
        if reference is not None
        else (frozenset({boundary.schema}) if boundary is not None else None)
    )
    allowed_operations = (
        frozenset(reference.allowed_operations) if reference is not None else None
    )
    allowed_repositories = (
        frozenset(reference.allowed_repositories)
        if reference is not None
        else (frozenset({boundary.repository}) if boundary is not None else None)
    )
    allowed_effects = (
        frozenset(reference.allowed_effects)
        if reference is not None
        else (frozenset(boundary.effects) if boundary is not None else None)
    )
    allowed_authorities = (
        frozenset(reference.allowed_authorities)
        if reference is not None
        else (
            frozenset({boundary.authority_class})
            if boundary is not None
            else frozenset({CANDIDATE_ONLY_AUTHORITY})
        )
    )
    allowed_capabilities = (
        frozenset(reference.allowed_capabilities)
        if reference is not None
        else (frozenset(boundary.capabilities) if boundary is not None else None)
    )
    required_context = tuple(
        dict.fromkeys(
            (
                *(reference.required_context_fields if reference is not None else ()),
                *(boundary.required_context_fields if boundary is not None else ()),
            )
        )
    )
    expected_group = ""
    if boundary is not None:
        expected_group = boundary.calibration_group_key
    elif reference is not None:
        expected_group = reference.group.group_key

    family_unknown = observation.family is None
    family_unseen = (not family_unknown) and (
        allowed_families is None or observation.family not in allowed_families
    )
    if boundary is not None and observation.family is not None:
        family_unseen = family_unseen or observation.family != boundary.family
    family_passed = not family_unknown and not family_unseen
    family_finding = _finding(
        BoundaryAxis.FAMILY,
        passed=family_passed,
        unknown_or_missing=family_unknown or family_unseen or allowed_families is None,
        risk=risk,
        reasons=() if family_passed else (REASON_FAMILY_BOUNDARY,),
    )

    schema_missing = not observation.schema
    schema_unknown = allowed_schemas is None or observation.schema not in allowed_schemas
    if boundary is not None and observation.schema:
        schema_unknown = schema_unknown or observation.schema != boundary.schema
    schema_passed = not schema_missing and not schema_unknown
    schema_finding = _finding(
        BoundaryAxis.SCHEMA,
        passed=schema_passed,
        unknown_or_missing=schema_missing or schema_unknown,
        risk=risk,
        reasons=() if schema_passed else (REASON_UNKNOWN_SCHEMA,),
    )

    unseen_effects = _subset_unknown(observation.effects, allowed_effects)
    effect_unknown = allowed_effects is None
    effect_passed = not effect_unknown and not unseen_effects
    if boundary is not None:
        extra_effects = [item for item in observation.effects if item not in boundary.effects]
        if extra_effects:
            effect_passed = False
            unseen_effects = tuple(dict.fromkeys((*unseen_effects, *extra_effects)))
    effect_finding = _finding(
        BoundaryAxis.EFFECT,
        passed=effect_passed,
        unknown_or_missing=effect_unknown or bool(unseen_effects),
        risk=risk,
        reasons=() if effect_passed else (REASON_UNSEEN_EFFECT,),
    )

    authority_missing = not observation.authority_class
    authority_unseen = observation.authority_class not in allowed_authorities
    authority_broken = observation.authority_class.casefold() != CANDIDATE_ONLY_AUTHORITY
    authority_passed = not authority_missing and not authority_unseen and not authority_broken
    authority_finding = _finding(
        BoundaryAxis.AUTHORITY,
        passed=authority_passed,
        unknown_or_missing=authority_missing or authority_unseen,
        risk=risk,
        reasons=() if authority_passed else (REASON_UNSEEN_AUTHORITY,),
        authority_contract_broken=authority_broken,
    )

    repository_missing = not observation.repository
    repository_unknown = (
        allowed_repositories is None or observation.repository not in allowed_repositories
    )
    if boundary is not None and observation.repository:
        repository_unknown = repository_unknown or observation.repository != boundary.repository
    repository_passed = not repository_missing and not repository_unknown
    repository_finding = _finding(
        BoundaryAxis.REPOSITORY,
        passed=repository_passed,
        unknown_or_missing=repository_missing or repository_unknown,
        risk=risk,
        reasons=() if repository_passed else (REASON_UNKNOWN_REPOSITORY,),
    )

    calibration_missing = not observation.calibration_group_key or not expected_group
    calibration_mismatch = (
        not calibration_missing and observation.calibration_group_key != expected_group
    )
    calibration_passed = not calibration_missing and not calibration_mismatch
    calibration_finding = _finding(
        BoundaryAxis.CALIBRATION,
        passed=calibration_passed,
        unknown_or_missing=calibration_missing or calibration_mismatch,
        risk=risk,
        reasons=() if calibration_passed else (REASON_CALIBRATION_ABSENCE,),
    )

    capability_unseen = _subset_unknown(observation.capabilities, allowed_capabilities)
    capability_unknown = allowed_capabilities is None
    capability_passed = (
        observation.capability_available and not capability_unknown and not capability_unseen
    )
    if boundary is not None:
        extra_caps = [item for item in observation.capabilities if item not in boundary.capabilities]
        if extra_caps:
            capability_passed = False
            capability_unseen = tuple(dict.fromkeys((*capability_unseen, *extra_caps)))
    capability_finding = _finding(
        BoundaryAxis.CAPABILITY,
        passed=capability_passed,
        unknown_or_missing=capability_unknown or bool(capability_unseen),
        risk=risk,
        reasons=() if capability_passed else (REASON_CAPABILITY_UNAVAILABLE,),
    )

    missing_context = tuple(
        field for field in required_context if field not in observation.context_fields
    )
    context_unknown = not observation.context_complete or bool(missing_context)
    if not required_context and observation.context_complete:
        context_unknown = False
    if boundary is None and reference is None:
        context_unknown = True
    context_passed = not context_unknown
    context_finding = _finding(
        BoundaryAxis.CONTEXT,
        passed=context_passed,
        unknown_or_missing=context_unknown,
        risk=risk,
        reasons=() if context_passed else (REASON_CONTEXT_INCOMPLETE,),
    )

    findings = (
        family_finding,
        schema_finding,
        effect_finding,
        authority_finding,
        repository_finding,
        calibration_finding,
        capability_finding,
        context_finding,
    )

    if not statistical_detection:
        signals.append(
            _signal(
                OODSignalKind.MISSING_DETECTION,
                REASON_MISSING_DETECTION,
                evidence=("detection_unavailable",),
            )
        )
    else:
        assert reference is not None
        offenders = _collect_feature_range_evidence(observation, reference)
        if offenders:
            signals.append(
                _signal(
                    OODSignalKind.FEATURE_RANGE,
                    REASON_FEATURE_RANGE,
                    evidence=offenders,
                )
            )
        if observation.family_distance_ppm > reference.family_distance_threshold_ppm:
            signals.append(
                _signal(
                    OODSignalKind.FAMILY_DISTANCE,
                    REASON_FAMILY_DISTANCE,
                    score_ppm=observation.family_distance_ppm,
                    evidence=(observation.family.value if observation.family else "unknown_family",),
                )
            )
        if allowed_operations is not None:
            if observation.operation:
                operation_unknown = observation.operation not in allowed_operations
            else:
                operation_unknown = bool(allowed_operations)
            if operation_unknown:
                signals.append(
                    _signal(
                        OODSignalKind.UNKNOWN_OPERATION,
                        REASON_UNKNOWN_OPERATION,
                        evidence=(observation.operation or "missing_operation",),
                    )
                )

    if not family_passed:
        signals.append(
            _signal(
                OODSignalKind.FAMILY_BOUNDARY,
                REASON_FAMILY_BOUNDARY,
                evidence=(observation.family.value if observation.family else "unknown_family",),
            )
        )
    if not schema_passed:
        signals.append(
            _signal(
                OODSignalKind.UNKNOWN_SCHEMA,
                REASON_UNKNOWN_SCHEMA,
                evidence=(observation.schema or "missing_schema",),
            )
        )
    if not repository_passed:
        signals.append(
            _signal(
                OODSignalKind.UNKNOWN_REPOSITORY,
                REASON_UNKNOWN_REPOSITORY,
                evidence=(observation.repository or "missing_repository",),
            )
        )
    if not effect_passed:
        signals.append(
            _signal(
                OODSignalKind.UNSEEN_EFFECT,
                REASON_UNSEEN_EFFECT,
                evidence=unseen_effects or ("unseen_effect",),
            )
        )
    if not authority_passed:
        signals.append(
            _signal(
                OODSignalKind.UNSEEN_AUTHORITY,
                REASON_UNSEEN_AUTHORITY,
                evidence=(observation.authority_class or "missing_authority",),
            )
        )
    if observation.disagreement:
        signals.append(
            _signal(
                OODSignalKind.DISAGREEMENT,
                REASON_DISAGREEMENT,
                evidence=observation.disagreement_identities or ("disagreement",),
            )
        )
    if not calibration_passed:
        signals.append(
            _signal(
                OODSignalKind.CALIBRATION_ABSENCE,
                REASON_CALIBRATION_ABSENCE,
                evidence=(observation.calibration_group_key or "missing_calibration_group",),
            )
        )
    if not context_passed:
        signals.append(
            _signal(
                OODSignalKind.CONTEXT_INCOMPLETE,
                REASON_CONTEXT_INCOMPLETE,
                evidence=missing_context or ("context_incomplete",),
            )
        )
    if not capability_passed:
        signals.append(
            _signal(
                OODSignalKind.CAPABILITY_UNAVAILABLE,
                REASON_CAPABILITY_UNAVAILABLE,
                evidence=capability_unseen or ("capability_unavailable",),
            )
        )

    unique_signals: list[OODSignal] = []
    seen_kinds: set[OODSignalKind] = set()
    for item in signals:
        if item.kind in seen_kinds:
            continue
        seen_kinds.add(item.kind)
        unique_signals.append(item)
    if len(unique_signals) > MAX_OOD_SIGNALS:
        raise ResidualIntelligenceError("OOD assessment exceeds signal bound")

    advisory_ood = bool(unique_signals)
    bound_ood = advisory_ood and policy_admits_ood
    conservative_abstain = any(item.conservative_abstain for item in findings)
    if not statistical_detection and risk in HIGH_RISK_CLASSES:
        conservative_abstain = True
    in_boundary_eligible = (
        statistical_detection
        and all(item.passed for item in findings)
        and not advisory_ood
        and not conservative_abstain
    )

    forced: ExpertDisposition | None
    if not observation.capability_available:
        forced = ExpertDisposition.CAPABILITY_UNAVAILABLE
    elif conservative_abstain:
        forced = ExpertDisposition.ABSTAIN
    elif bound_ood:
        forced = ExpertDisposition.OUT_OF_DISTRIBUTION
    else:
        forced = None

    reasons: list[str] = []
    if in_boundary_eligible:
        reasons.append(REASON_IN_BOUNDARY)
    else:
        if not statistical_detection:
            reasons.append(REASON_MISSING_DETECTION)
            reasons.append(REASON_SAFETY_NOT_ESTABLISHED)
        if advisory_ood and not policy_admits_ood:
            reasons.append(REASON_ADVISORY_ONLY)
        for item in unique_signals:
            if item.reason_code not in reasons:
                reasons.append(item.reason_code)
        for item in findings:
            for code in item.reason_codes:
                if code not in reasons:
                    reasons.append(code)
        if conservative_abstain and REASON_HIGH_RISK_UNKNOWN not in reasons:
            if risk in HIGH_RISK_CLASSES:
                reasons.append(REASON_HIGH_RISK_UNKNOWN)
    if not reasons:
        reasons.append(REASON_ADVISORY_ONLY)

    return OODAssessment(
        signals=tuple(unique_signals),
        boundary_findings=findings,
        risk_class=risk,
        advisory_ood=advisory_ood,
        policy_admits_ood=policy_admits_ood,
        bound_ood=bound_ood,
        conservative_abstain=conservative_abstain,
        in_boundary_eligible=in_boundary_eligible,
        detection_available=statistical_detection,
        reason_codes=tuple(reasons),
        forced_disposition=forced,
        safety_established=False,
        candidate_only=True,
    )


__all__ = (
    "BOUNDARY_AXES",
    "BOUNDARY_CONTRACT_SCHEMA",
    "CANDIDATE_ONLY_AUTHORITY",
    "HIGH_RISK_CLASSES",
    "OOD_ASSESSMENT_SCHEMA",
    "OOD_SIGNAL_SCHEMA",
    "REASON_ADVISORY_ONLY",
    "REASON_CALIBRATION_ABSENCE",
    "REASON_CONTEXT_INCOMPLETE",
    "REASON_DISAGREEMENT",
    "REASON_FEATURE_RANGE",
    "REASON_IN_BOUNDARY",
    "REASON_MISSING_DETECTION",
    "REASON_SAFETY_NOT_ESTABLISHED",
    "REASON_UNKNOWN_OPERATION",
    "REASON_UNKNOWN_REPOSITORY",
    "REASON_UNKNOWN_SCHEMA",
    "REASON_UNSEEN_AUTHORITY",
    "REASON_UNSEEN_EFFECT",
    "BoundaryAxis",
    "BoundaryContract",
    "BoundaryFinding",
    "FeatureRange",
    "OODAssessment",
    "OODObservation",
    "OODSignal",
    "OODSignalKind",
    "ReferenceDistribution",
    "assess_out_of_distribution",
    "observation_from_task_input",
)
