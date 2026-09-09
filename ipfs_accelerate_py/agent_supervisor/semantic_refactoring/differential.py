"""SPAR-028 differential execution and import/runtime trace comparison.

This module extends current supervisor validation orchestration with
``DifferentialExecutionReceipt@1``.  It consumes SPAR-025 extraction-wave
receipts and SPAR-026 validation-selection receipts, then runs hermetic
paired old/new workflows and compares outputs, exceptions, effects, import
events, registrations, state transitions, resources, and declared trace
projections.

The comparator is nomination-only.  It cannot authorize a transition,
completion, or competing authority.  General Python equivalence is not
claimed.  Evidence is bounded by the declared observation profile.
Unsupported or unobserved required dimensions are typed terminals, never
success.  Network is denied.  Vector, model, and heuristic evidence cannot
admit agreement.  Observational metadata is excluded from identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)


TASK_ID: Final[str] = "SPAR-028"
GOAL_ID: Final[str] = "SPAR-G052"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "validation orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.differential@1"
)

DIFFERENTIAL_EXECUTION_INTERFACE: Final[str] = "DifferentialExecution@1"
OBSERVATION_PROFILE_INTERFACE: Final[str] = "ObservationProfile@1"
WORKFLOW_OBSERVATION_INTERFACE: Final[str] = "WorkflowObservation@1"
DIMENSION_COMPARISON_INTERFACE: Final[str] = "DimensionComparison@1"
DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE: Final[str] = (
    "DifferentialExecutionReceipt@1"
)

DIFFERENTIAL_EXECUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/differential-execution@1"
)
OBSERVATION_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observation-profile@1"
)
WORKFLOW_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/workflow-observation@1"
)
DIMENSION_COMPARISON_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dimension-comparison@1"
)
DIFFERENTIAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/differential-execution-receipt@1"
)

DIFFERENTIAL_CONTRACT_VERSION: Final[str] = "1"

DIFFERENTIAL_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
DIFFERENTIAL_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
DIFFERENTIAL_CAN_CREATE_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
EXECUTOR_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
NETWORK_DENIED: Final[bool] = True
HERMETIC_PAIRED_EXECUTION: Final[bool] = True
GENERAL_PYTHON_EQUIVALENCE_CLAIMED: Final[bool] = False
TRACES_PROVE_ONLY_OBSERVATIONS: Final[bool] = True
UNRESOLVED_DYNAMICS_LOWER_AUTONOMY: Final[bool] = True
IMPLICIT_INSTALL_FORBIDDEN: Final[bool] = True
IMPLICIT_NETWORK_FORBIDDEN: Final[bool] = True

NETWORK_DENY: Final[str] = "deny"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_DIMENSIONS: Final[int] = 32

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

_FORBIDDEN_CAPSULE_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FunctionSemanticCapsule",
        "MethodSemanticCapsule",
        "ClassSemanticCapsule",
        "TopLevelBlockCapsule",
        "ModuleSemanticCapsule",
        "PackageSemanticCapsule",
        "CallsiteSemanticCapsule",
        "StateOwnerCapsule",
        "RegistrationCapsule",
        "ResourceLifecycleCapsule",
    }
)

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "projection_is_authority",
)

_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "body",
        "code",
        "contents",
        "content",
        "file_content",
        "file_contents",
        "file_text",
        "full_ast",
        "full_graph",
        "full_proof",
        "full_source",
        "full_trace",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "transcript",
        "witness",
        "snippet",
        "solver_trace",
        "raw_output",
        "prover_output",
    }
)

_FORBIDDEN_WORKFLOW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "download",
        "install",
        "model_download",
        "pip",
        "pip_install",
        "implicit_network",
        "allow_network",
    }
)

FORBIDDEN_EXECUTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "claim_general_equivalence",
        "admit_vector_agreement",
        "open_network",
        "implicit_install",
        "suppress_raw_source",
    }
)


class DifferentialExecutionError(ValueError):
    """Fail-closed violation of a SPAR-028 differential-execution contract."""

    def __init__(
        self,
        message: str,
        *,
        negative_evidence_cids: Sequence[str] = (),
        comparisons: Sequence["DimensionComparison"] = (),
    ) -> None:
        super().__init__(message)
        self.negative_evidence_cids = tuple(negative_evidence_cids)
        self.comparisons = tuple(comparisons)


class TraceDimension(str, Enum):
    OUTPUTS = "outputs"
    EXCEPTIONS = "exceptions"
    EFFECTS = "effects"
    IMPORT_EVENTS = "import_events"
    REGISTRATIONS = "registrations"
    STATE_TRANSITIONS = "state_transitions"
    RESOURCES = "resources"
    DECLARED_TRACE_PROJECTIONS = "declared_trace_projections"


class ComparisonOutcome(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    UNOBSERVED = "unobserved"
    UNSUPPORTED = "unsupported"


class WorkflowRole(str, Enum):
    OLD = "old"
    NEW = "new"


class DifferentialStatus(str, Enum):
    EQUIVALENT = "equivalent"
    DIVERGENT = "divergent"
    BLOCKED = "blocked"
    REJECTED = "rejected"


DECLARED_DIMENSIONS: Final[tuple[str, ...]] = tuple(
    item.value for item in TraceDimension
)
DECLARED_COMPARISON_OUTCOMES: Final[frozenset[str]] = frozenset(
    item.value for item in ComparisonOutcome
)
DECLARED_WORKFLOW_ROLES: Final[frozenset[str]] = frozenset(
    item.value for item in WorkflowRole
)
DECLARED_DIFFERENTIAL_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in DifferentialStatus
)
REQUIRED_TRACE_DIMENSIONS: Final[tuple[str, ...]] = DECLARED_DIMENSIONS


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise DifferentialExecutionError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise DifferentialExecutionError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise DifferentialExecutionError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise DifferentialExecutionError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise DifferentialExecutionError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise DifferentialExecutionError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise DifferentialExecutionError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise DifferentialExecutionError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise DifferentialExecutionError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise DifferentialExecutionError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise DifferentialExecutionError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise DifferentialExecutionError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise DifferentialExecutionError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _reject_forbidden_bodies(payload: Any, name: str) -> None:
    if isinstance(payload, Mapping) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        present = _FORBIDDEN_BODY_KEYS & set(payload)
        if present:
            raise DifferentialExecutionError(
                f"{name} must remain body-free; forbidden keys: {sorted(present)}"
            )
        for key, item in payload.items():
            _reject_forbidden_bodies(item, f"{name}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        for index, item in enumerate(payload):
            _reject_forbidden_bodies(item, f"{name}[{index}]")


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise DifferentialExecutionError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise DifferentialExecutionError(f"{name} does not verify")


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise DifferentialExecutionError(f"{name} exceeds path bound")
    normalized = raw.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or normalized in {".", ""}
        or normalized.startswith("./")
        or any(char in normalized for char in "*?[]{}")
        or "//" in normalized
        or normalized.endswith("/")
    ):
        raise DifferentialExecutionError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise DifferentialExecutionError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DifferentialExecutionError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise DifferentialExecutionError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise DifferentialExecutionError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DifferentialExecutionError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise DifferentialExecutionError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise DifferentialExecutionError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise DifferentialExecutionError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise DifferentialExecutionError(f"missing {name}")
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        _reject_excluded(value, name)
        _reject_forbidden_bodies(value, name)
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping) and not isinstance(
            payload, (str, bytes, bytearray)
        ):
            _reject_excluded(payload, name)
            _reject_forbidden_bodies(payload, name)
            return dict(payload)
    raise DifferentialExecutionError(f"{name} must be a mapping")


def _nested_mapping(value: Any, name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        _reject_forbidden_bodies(value, name)
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            _reject_forbidden_bodies(payload, name)
            return dict(payload)
    raise DifferentialExecutionError(f"{name} must be an object")


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise DifferentialExecutionError(f"unknown {name}: {text}") from exc


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise DifferentialExecutionError(f"{name} cannot claim {flag}")


def differential_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def differential_descriptor() -> dict[str, Any]:
    return {
        "interface": DIFFERENTIAL_EXECUTION_INTERFACE,
        "receipt_interface": DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "network": NETWORK_DENY,
        "raw_source_required": True,
        "nomination_only": True,
        "claims_general_equivalence": False,
        "traces_prove_only_observations": True,
        "required_dimensions": list(REQUIRED_TRACE_DIMENSIONS),
        "forbids": (
            "claim_general_equivalence",
            "open_network",
            "implicit_install",
            "admit_vector_agreement",
            "suppress_raw_source",
            "unrestricted_scope",
        ),
    }


def _network_value(value: Any, name: str = "network") -> str:
    if value in (None, ""):
        return NETWORK_DENY
    text = _text(value, name)
    if text != NETWORK_DENY:
        raise DifferentialExecutionError("network is denied for hermetic differential execution")
    return NETWORK_DENY


def _dimension_value(value: Any, name: str = "dimension") -> str:
    if isinstance(value, TraceDimension):
        return value.value
    text = _text(value, name)
    if text not in DECLARED_DIMENSIONS:
        raise DifferentialExecutionError(f"unknown {name}: {text}")
    return text


def _dimensions(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, ()):
        if required:
            raise DifferentialExecutionError(f"{name} must not be empty")
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DifferentialExecutionError(f"{name} must be a list")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        dimension = _dimension_value(item, name)
        if dimension in seen:
            raise DifferentialExecutionError(f"{name} must not contain duplicates")
        seen.add(dimension)
        ordered.append(dimension)
    if required and not ordered:
        raise DifferentialExecutionError(f"{name} must not be empty")
    if len(ordered) > MAX_DIMENSIONS:
        raise DifferentialExecutionError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _ordered_declared(dimensions: Sequence[str]) -> tuple[str, ...]:
    rank = {name: index for index, name in enumerate(DECLARED_DIMENSIONS)}
    unknown = [item for item in dimensions if item not in rank]
    if unknown:
        raise DifferentialExecutionError(f"unknown dimension: {unknown}")
    return tuple(sorted(dimensions, key=lambda item: rank[item]))


def _observation_cid(value: Any, name: str) -> str:
    if value in (None, "", ()):
        return ""
    if type(value) is str:
        return _cid(value, name)
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        claimed = value.get("observation_cid")
        payload = {
            key: item for key, item in dict(value).items() if key != "observation_cid"
        }
        _reject_excluded(payload, name)
        _reject_forbidden_bodies(payload, name)
        computed = cid_for_dag_json(payload)
        if claimed not in (None, ""):
            _verify_cid(claimed, computed, name)
        return computed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[Any] = []
        for index, item in enumerate(value):
            if type(item) is str:
                items.append(_cid(item, f"{name}[{index}]"))
                continue
            if isinstance(item, Mapping) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                _reject_forbidden_bodies(item, f"{name}[{index}]")
                items.append(dict(item))
                continue
            raise DifferentialExecutionError(f"{name} items must be CIDs or objects")
        return cid_for_dag_json(items)
    raise DifferentialExecutionError(f"{name} must be a CID or DAG-JSON object")


def _order_dimension_pairs(
    pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    mapping = _pairs_to_map(tuple(pairs))
    return tuple((dimension, mapping[dimension]) for dimension in _ordered_declared(tuple(mapping)))


def _dimension_cid_pairs(
    value: Any,
    name: str = "dimension_cids",
) -> tuple[tuple[str, str], ...]:
    if value in (None, (), {}):
        return ()
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        pairs: list[tuple[str, str]] = []
        for key, item in value.items():
            dimension = _dimension_value(key, name)
            cid = _observation_cid(item, f"{name}.{dimension}")
            if not cid:
                continue
            pairs.append((dimension, cid))
        return _order_dimension_pairs(pairs)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        pairs = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray)
            ) and len(item) == 2:
                dimension = _dimension_value(item[0], name)
                cid = _observation_cid(item[1], f"{name}.{dimension}")
                if cid:
                    pairs.append((dimension, cid))
                continue
            if isinstance(item, Mapping) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                dimension = _dimension_value(
                    item.get("dimension") or item.get("name"), name
                )
                cid = _observation_cid(
                    item.get("observation_cid") or item.get("cid") or item.get("value"),
                    f"{name}.{dimension}",
                )
                if cid:
                    pairs.append((dimension, cid))
                continue
            raise DifferentialExecutionError(f"{name} items must be dimension/CID pairs")
        return _order_dimension_pairs(pairs)
    raise DifferentialExecutionError(f"{name} must be an object")


def _pairs_to_map(pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    payload = {dimension: cid for dimension, cid in pairs}
    if len(payload) != len(pairs):
        raise DifferentialExecutionError("dimension_cids must not contain duplicates")
    return payload


def _cids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, ()):
        if required:
            raise DifferentialExecutionError(f"{name} are required")
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DifferentialExecutionError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise DifferentialExecutionError(f"{name} are required")
    if len(ordered) != len(set(ordered)):
        raise DifferentialExecutionError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise DifferentialExecutionError(f"{name} exceed maximum length")
    return ordered


@dataclass(frozen=True, slots=True)
class ObservationProfile:
    """Declared hermetic observation/proof profile. Body-free."""

    required_dimensions: Sequence[str] = REQUIRED_TRACE_DIMENSIONS
    optional_dimensions: Sequence[str] = ()
    network: str = NETWORK_DENY
    claims_general_equivalence: bool = False
    traces_prove_only_observations: bool = True
    unresolved_dynamics_lower_autonomy: bool = True

    interface: ClassVar[str] = OBSERVATION_PROFILE_INTERFACE
    schema: ClassVar[str] = OBSERVATION_PROFILE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "required_dimensions",
            "optional_dimensions",
            "network",
            "claims_general_equivalence",
            "traces_prove_only_observations",
            "unresolved_dynamics_lower_autonomy",
            "profile_cid",
        }
    )

    def __post_init__(self) -> None:
        required = _ordered_declared(
            _dimensions(self.required_dimensions, "required_dimensions", required=True)
        )
        optional = _ordered_declared(
            _dimensions(self.optional_dimensions, "optional_dimensions")
        )
        overlap = set(required) & set(optional)
        if overlap:
            raise DifferentialExecutionError(
                f"optional dimensions cannot overlap required dimensions: {sorted(overlap)}"
            )
        if _bool(self.claims_general_equivalence, "claims_general_equivalence"):
            raise DifferentialExecutionError("general Python equivalence is not claimed")
        if not _bool(
            self.traces_prove_only_observations, "traces_prove_only_observations"
        ):
            raise DifferentialExecutionError("traces prove only observations")
        if not _bool(
            self.unresolved_dynamics_lower_autonomy,
            "unresolved_dynamics_lower_autonomy",
        ):
            raise DifferentialExecutionError(
                "unresolved dynamics must lower autonomy"
            )
        object.__setattr__(self, "required_dimensions", required)
        object.__setattr__(self, "optional_dimensions", optional)
        object.__setattr__(self, "network", _network_value(self.network))
        object.__setattr__(self, "claims_general_equivalence", False)
        object.__setattr__(self, "traces_prove_only_observations", True)
        object.__setattr__(self, "unresolved_dynamics_lower_autonomy", True)

    @property
    def declared_dimensions(self) -> tuple[str, ...]:
        return _ordered_declared(
            tuple(self.required_dimensions) + tuple(self.optional_dimensions)
        )

    def is_required(self, dimension: str) -> bool:
        return _dimension_value(dimension) in set(self.required_dimensions)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": OBSERVATION_PROFILE_SCHEMA,
            "interface": OBSERVATION_PROFILE_INTERFACE,
            "required_dimensions": list(self.required_dimensions),
            "optional_dimensions": list(self.optional_dimensions),
            "network": NETWORK_DENY,
            "claims_general_equivalence": False,
            "traces_prove_only_observations": True,
            "unresolved_dynamics_lower_autonomy": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def profile_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["profile_cid"] = self.profile_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationProfile":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("profile_cid")
        if payload.pop("schema") != OBSERVATION_PROFILE_SCHEMA:
            raise DifferentialExecutionError("unsupported ObservationProfile schema")
        if payload.pop("interface") != OBSERVATION_PROFILE_INTERFACE:
            raise DifferentialExecutionError("unsupported ObservationProfile interface")
        result = cls(**payload)
        _verify_cid(claimed, result.profile_cid, "profile_cid")
        return result


def compile_observation_profile(
    profile: ObservationProfile | Mapping[str, Any] | None = None,
) -> ObservationProfile:
    if profile is None:
        return ObservationProfile()
    if isinstance(profile, ObservationProfile):
        return profile
    payload = _as_mapping(profile, "observation_profile")
    if "profile_cid" in payload:
        return ObservationProfile.from_dict(payload)
    return ObservationProfile(
        required_dimensions=payload.get("required_dimensions", REQUIRED_TRACE_DIMENSIONS),
        optional_dimensions=payload.get("optional_dimensions", ()),
        network=payload.get("network", NETWORK_DENY),
        claims_general_equivalence=payload.get("claims_general_equivalence", False),
        traces_prove_only_observations=payload.get(
            "traces_prove_only_observations", True
        ),
        unresolved_dynamics_lower_autonomy=payload.get(
            "unresolved_dynamics_lower_autonomy", True
        ),
    )


@dataclass(frozen=True, slots=True)
class WorkflowObservation:
    """Hermetic one-sided workflow observation. Body-free CID projections."""

    role: WorkflowRole | str
    workflow_id: str
    dimension_cids: Sequence[tuple[str, str]] = ()
    unsupported_dimensions: Sequence[str] = ()
    unobserved_dimensions: Sequence[str] = ()
    raw_source_cids: Sequence[str] = ()
    network: str = NETWORK_DENY
    hermetic: bool = True

    interface: ClassVar[str] = WORKFLOW_OBSERVATION_INTERFACE
    schema: ClassVar[str] = WORKFLOW_OBSERVATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "role",
            "workflow_id",
            "dimension_cids",
            "unsupported_dimensions",
            "unobserved_dimensions",
            "raw_source_cids",
            "network",
            "hermetic",
            "observation_cid",
        }
    )

    def __post_init__(self) -> None:
        role = _enum(self.role, WorkflowRole, "role")
        pairs = _dimension_cid_pairs(self.dimension_cids)
        mapping = _pairs_to_map(pairs)
        unsupported = _ordered_declared(
            _dimensions(self.unsupported_dimensions, "unsupported_dimensions")
        )
        unobserved = _ordered_declared(
            _dimensions(self.unobserved_dimensions, "unobserved_dimensions")
        )
        if set(unsupported) & set(mapping):
            raise DifferentialExecutionError(
                "unsupported dimensions cannot also carry observation CIDs"
            )
        if set(unobserved) & set(mapping):
            raise DifferentialExecutionError(
                "unobserved dimensions cannot also carry observation CIDs"
            )
        if set(unsupported) & set(unobserved):
            raise DifferentialExecutionError(
                "unsupported and unobserved dimensions must be disjoint"
            )
        if not _bool(self.hermetic, "hermetic"):
            raise DifferentialExecutionError("workflow observation must remain hermetic")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "workflow_id", _text(self.workflow_id, "workflow_id"))
        object.__setattr__(self, "dimension_cids", pairs)
        object.__setattr__(self, "unsupported_dimensions", unsupported)
        object.__setattr__(self, "unobserved_dimensions", unobserved)
        object.__setattr__(
            self,
            "raw_source_cids",
            _cids(self.raw_source_cids, "raw_source_cids", required=True),
        )
        object.__setattr__(self, "network", _network_value(self.network))
        object.__setattr__(self, "hermetic", True)

    def cid_for(self, dimension: str) -> str:
        mapping = _pairs_to_map(self.dimension_cids)
        return mapping.get(_dimension_value(dimension), "")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": WORKFLOW_OBSERVATION_SCHEMA,
            "interface": WORKFLOW_OBSERVATION_INTERFACE,
            "role": self.role,
            "workflow_id": self.workflow_id,
            "dimension_cids": _pairs_to_map(self.dimension_cids),
            "unsupported_dimensions": list(self.unsupported_dimensions),
            "unobserved_dimensions": list(self.unobserved_dimensions),
            "raw_source_cids": list(self.raw_source_cids),
            "network": NETWORK_DENY,
            "hermetic": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def observation_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["observation_cid"] = self.observation_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowObservation":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("observation_cid")
        if payload.pop("schema") != WORKFLOW_OBSERVATION_SCHEMA:
            raise DifferentialExecutionError("unsupported WorkflowObservation schema")
        if payload.pop("interface") != WORKFLOW_OBSERVATION_INTERFACE:
            raise DifferentialExecutionError(
                "unsupported WorkflowObservation interface"
            )
        result = cls(**payload)
        _verify_cid(claimed, result.observation_cid, "observation_cid")
        return result


@dataclass(frozen=True, slots=True)
class DimensionComparison:
    """One declared-dimension old/new comparison. Nomination-only."""

    dimension: str
    old_observation_cid: str = ""
    new_observation_cid: str = ""
    outcome: ComparisonOutcome | str = ComparisonOutcome.UNOBSERVED
    required: bool = True

    interface: ClassVar[str] = DIMENSION_COMPARISON_INTERFACE
    schema: ClassVar[str] = DIMENSION_COMPARISON_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "dimension",
            "old_observation_cid",
            "new_observation_cid",
            "outcome",
            "required",
            "comparison_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", _dimension_value(self.dimension))
        object.__setattr__(
            self,
            "old_observation_cid",
            _optional_cid(self.old_observation_cid, "old_observation_cid"),
        )
        object.__setattr__(
            self,
            "new_observation_cid",
            _optional_cid(self.new_observation_cid, "new_observation_cid"),
        )
        object.__setattr__(
            self, "outcome", _enum(self.outcome, ComparisonOutcome, "outcome")
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": DIMENSION_COMPARISON_SCHEMA,
            "interface": DIMENSION_COMPARISON_INTERFACE,
            "dimension": self.dimension,
            "old_observation_cid": self.old_observation_cid,
            "new_observation_cid": self.new_observation_cid,
            "outcome": self.outcome,
            "required": self.required,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def comparison_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["comparison_cid"] = self.comparison_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DimensionComparison":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("comparison_cid")
        if payload.pop("schema") != DIMENSION_COMPARISON_SCHEMA:
            raise DifferentialExecutionError("unsupported DimensionComparison schema")
        if payload.pop("interface") != DIMENSION_COMPARISON_INTERFACE:
            raise DifferentialExecutionError(
                "unsupported DimensionComparison interface"
            )
        result = cls(**payload)
        _verify_cid(claimed, result.comparison_cid, "comparison_cid")
        return result


def _compare_dimension(
    dimension: str,
    *,
    old: WorkflowObservation,
    new: WorkflowObservation,
    required: bool,
) -> DimensionComparison:
    name = _dimension_value(dimension)
    if name in set(old.unsupported_dimensions) or name in set(new.unsupported_dimensions):
        return DimensionComparison(
            dimension=name,
            old_observation_cid=old.cid_for(name),
            new_observation_cid=new.cid_for(name),
            outcome=ComparisonOutcome.UNSUPPORTED,
            required=required,
        )
    old_cid = old.cid_for(name)
    new_cid = new.cid_for(name)
    if (
        name in set(old.unobserved_dimensions)
        or name in set(new.unobserved_dimensions)
        or not old_cid
        or not new_cid
    ):
        return DimensionComparison(
            dimension=name,
            old_observation_cid=old_cid,
            new_observation_cid=new_cid,
            outcome=ComparisonOutcome.UNOBSERVED,
            required=required,
        )
    outcome = (
        ComparisonOutcome.AGREE
        if old_cid == new_cid
        else ComparisonOutcome.DISAGREE
    )
    return DimensionComparison(
        dimension=name,
        old_observation_cid=old_cid,
        new_observation_cid=new_cid,
        outcome=outcome,
        required=required,
    )


def compare_declared_dimensions(
    old: WorkflowObservation,
    new: WorkflowObservation,
    *,
    profile: ObservationProfile | None = None,
) -> tuple[DimensionComparison, ...]:
    """Compare declared dimensions. Does not admit equivalence or completion."""

    resolved = profile or ObservationProfile()
    comparisons: list[DimensionComparison] = []
    for dimension in resolved.required_dimensions:
        comparisons.append(
            _compare_dimension(dimension, old=old, new=new, required=True)
        )
    for dimension in resolved.optional_dimensions:
        comparisons.append(
            _compare_dimension(dimension, old=old, new=new, required=False)
        )
    return tuple(comparisons)


def _wave_packet_cids(wave: Mapping[str, Any]) -> tuple[str, ...]:
    packets = wave.get("packet_cids")
    if packets in (None, ()):
        packet = wave.get("packet_cid")
        if packet in (None, ""):
            raise DifferentialExecutionError("SPAR-025 packet_cids are required")
        return (_cid(packet, "packet_cid"),)
    return _cids(packets, "packet_cids", required=True)


def _wave_cid(wave: Mapping[str, Any]) -> str:
    claimed = wave.get("receipt_cid") or wave.get("wave_cid")
    if claimed in (None, ""):
        raise DifferentialExecutionError("SPAR-025 receipt_cid is required")
    return _cid(claimed, "wave_cid")


def _selection_cid(selection: Mapping[str, Any]) -> str:
    claimed = (
        selection.get("validation_selection_cid")
        or selection.get("receipt_cid")
        or selection.get("selection_cid")
    )
    if claimed in (None, ""):
        raise DifferentialExecutionError("SPAR-026 validation_selection_cid is required")
    return _cid(claimed, "validation_selection_cid")


def _selection_packet_cid(selection: Mapping[str, Any]) -> str:
    claimed = selection.get("packet_cid")
    if claimed in (None, ""):
        raise DifferentialExecutionError("SPAR-026 packet_cid is required")
    return _cid(claimed, "packet_cid")


def _selection_sources(selection: Mapping[str, Any]) -> tuple[str, ...]:
    sources = selection.get("raw_source_cids") or selection.get("source_cids")
    if sources in (None, ()):
        preimage = _nested_mapping(selection.get("preimage"), "preimage")
        sources = preimage.get("source_cids")
    if sources in (None, ()):
        raise DifferentialExecutionError("raw source required")
    ordered = _cids(sources, "raw_source_cids", required=True)
    if not ordered:
        raise DifferentialExecutionError("raw source required")
    return ordered


def _selection_write_paths(selection: Mapping[str, Any]) -> tuple[str, ...]:
    if "write_paths" in selection:
        return _exact_paths(selection["write_paths"], "write_paths")
    scope = _nested_mapping(selection.get("effect_scope"), "effect_scope")
    if "write_paths" in scope:
        return _exact_paths(scope["write_paths"], "write_paths")
    raise DifferentialExecutionError("SPAR-026 write_paths are required")


def _wave_write_paths(wave: Mapping[str, Any]) -> tuple[str, ...]:
    if "write_paths" in wave:
        return _exact_paths(wave["write_paths"], "write_paths")
    raise DifferentialExecutionError("SPAR-025 write_paths are required")


def _selection_commands(selection: Mapping[str, Any]) -> tuple[str, ...]:
    commands = selection.get("validation_commands")
    if commands in (None, ()):
        raise DifferentialExecutionError("SPAR-026 validation_commands are required")
    return _commands(commands)


def _reject_vector_admission(vector_evidence: Any) -> None:
    if vector_evidence in (None, (), {}):
        return
    payload = (
        _as_mapping(vector_evidence, "vector_evidence")
        if not isinstance(vector_evidence, Sequence)
        else None
    )
    items: tuple[Mapping[str, Any], ...]
    if payload is not None:
        items = (payload,)
    else:
        if isinstance(vector_evidence, (str, bytes, bytearray)) or not isinstance(
            vector_evidence, Sequence
        ):
            raise DifferentialExecutionError("vector_evidence must be an object")
        mapped: list[Mapping[str, Any]] = []
        for item in vector_evidence:
            mapped.append(_as_mapping(item, "vector_evidence"))
        items = tuple(mapped)
    for item in items:
        evidence = str(item.get("evidence_class") or item.get("kind") or "")
        if item.get("suppress_raw_source") is True or item.get("skip_raw_source") is True:
            raise DifferentialExecutionError(
                "vectors/projections cannot suppress raw-source fallback"
            )
        if (
            item.get("admits_agreement") is True
            or item.get("admits_equivalence") is True
            or item.get("skip_differential") is True
        ):
            raise DifferentialExecutionError(
                "vector, model, or heuristic evidence cannot admit agreement"
            )
        if evidence in _NON_ADMITTING_EVIDENCE and item.get("admits_selection") is True:
            raise DifferentialExecutionError(
                "vector, model, or heuristic evidence cannot admit agreement"
            )


def _reject_workflow_surface(workflow: Mapping[str, Any], name: str) -> None:
    present = _FORBIDDEN_WORKFLOW_KEYS & set(workflow)
    if present:
        raise DifferentialExecutionError(
            f"{name} forbids implicit install/network: {sorted(present)}"
        )
    network = workflow.get("network", NETWORK_DENY)
    _network_value(network, f"{name}.network")
    if workflow.get("command") not in (None, "") and workflow.get("observations") in (
        None,
        {},
        (),
    ) and workflow.get("runner") is None:
        raise DifferentialExecutionError(
            f"{name} implicit subprocess execution is not admitted without hermetic observation capture"
        )


def _workflow_observations_map(workflow: Mapping[str, Any]) -> dict[str, Any]:
    observations = workflow.get("observations") or workflow.get("dimension_cids")
    if observations in (None, {}, ()):
        return {}
    if isinstance(observations, Mapping) and not isinstance(
        observations, (str, bytes, bytearray)
    ):
        return dict(observations)
    raise DifferentialExecutionError("observations must be an object")


def run_hermetic_workflow(
    workflow: Mapping[str, Any] | WorkflowObservation | Any,
    *,
    role: WorkflowRole | str | None = None,
    raw_source_cids: Sequence[str] | None = None,
) -> WorkflowObservation:
    """Compile or invoke one hermetic workflow observation. Network denied."""

    if isinstance(workflow, WorkflowObservation):
        if role is not None and workflow.role != _enum(role, WorkflowRole, "role"):
            raise DifferentialExecutionError("workflow role does not match")
        if workflow.network != NETWORK_DENY:
            raise DifferentialExecutionError("network is denied")
        return workflow

    runner = None
    if callable(workflow):
        runner = workflow
        payload: dict[str, Any] = {}
    else:
        payload = _as_mapping(workflow, "workflow")
        _reject_workflow_surface(payload, "workflow")
        runner = payload.get("runner")

    if callable(runner):
        produced = runner(role=_enum(role or payload.get("role") or WorkflowRole.OLD, WorkflowRole, "role"), network=NETWORK_DENY)
        if not isinstance(produced, Mapping):
            if isinstance(produced, WorkflowObservation):
                return run_hermetic_workflow(produced, role=role, raw_source_cids=raw_source_cids)
            raise DifferentialExecutionError("hermetic runner must return an observation mapping")
        payload = {**payload, **dict(produced)}
        payload.pop("runner", None)
        _reject_workflow_surface(payload, "workflow")

    resolved_role = _enum(
        role or payload.get("role") or WorkflowRole.OLD, WorkflowRole, "role"
    )
    workflow_id = payload.get("workflow_id") or payload.get("id")
    if workflow_id in (None, ""):
        raise DifferentialExecutionError("workflow_id is required")
    sources = payload.get("raw_source_cids") or raw_source_cids
    observations = _workflow_observations_map(payload)
    dimension_cids = payload.get("dimension_cids") or observations
    return WorkflowObservation(
        role=resolved_role,
        workflow_id=_text(workflow_id, "workflow_id"),
        dimension_cids=dimension_cids,
        unsupported_dimensions=payload.get("unsupported_dimensions") or (),
        unobserved_dimensions=payload.get("unobserved_dimensions") or (),
        raw_source_cids=sources or (),
        network=payload.get("network", NETWORK_DENY),
        hermetic=payload.get("hermetic", True),
    )


def _fail_required(
    comparisons: Sequence[DimensionComparison],
) -> None:
    required = [item for item in comparisons if item.required]
    unsupported = [
        item.dimension
        for item in required
        if item.outcome == ComparisonOutcome.UNSUPPORTED.value
    ]
    unobserved = [
        item.dimension
        for item in required
        if item.outcome == ComparisonOutcome.UNOBSERVED.value
    ]
    diverge = [
        item.dimension
        for item in required
        if item.outcome == ComparisonOutcome.DISAGREE.value
    ]
    evidence = tuple(item.comparison_cid for item in required if item.outcome != ComparisonOutcome.AGREE.value)
    if unsupported:
        raise DifferentialExecutionError(
            f"unsupported required dimensions remain blockers: {unsupported}",
            negative_evidence_cids=evidence,
            comparisons=comparisons,
        )
    if unobserved:
        raise DifferentialExecutionError(
            f"unobserved required dimensions remain blockers: {unobserved}",
            negative_evidence_cids=evidence,
            comparisons=comparisons,
        )
    if diverge:
        raise DifferentialExecutionError(
            f"required dimensions disagree: {diverge}",
            negative_evidence_cids=evidence,
            comparisons=comparisons,
        )


@dataclass(frozen=True, slots=True)
class DifferentialExecutionReceipt:
    """Content-addressed SPAR-028 paired-execution receipt. Nomination-only."""

    tree_id: str
    wave_cid: str
    selection_cid: str
    packet_cid: str
    profile_cid: str
    old_observation_cid: str
    new_observation_cid: str
    comparisons: Sequence[DimensionComparison]
    write_paths: Sequence[str]
    raw_source_cids: Sequence[str]
    validation_commands: Sequence[str]
    worktree_id: str = ""
    status: DifferentialStatus | str = DifferentialStatus.EQUIVALENT
    network: str = NETWORK_DENY
    analyzer_id: str = ANALYZER_ID
    executor_is_nomination_only: bool = True
    claims_general_equivalence: bool = False
    traces_prove_only_observations: bool = True
    unresolved_dynamics_lower_autonomy: bool = True

    interface: ClassVar[str] = DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE
    schema: ClassVar[str] = DIFFERENTIAL_EXECUTION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "wave_cid",
            "selection_cid",
            "packet_cid",
            "profile_cid",
            "old_observation_cid",
            "new_observation_cid",
            "comparisons",
            "write_paths",
            "raw_source_cids",
            "validation_commands",
            "worktree_id",
            "status",
            "network",
            "analyzer_id",
            "executor_is_nomination_only",
            "claims_general_equivalence",
            "traces_prove_only_observations",
            "unresolved_dynamics_lower_autonomy",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise DifferentialExecutionError("analyzer_id must remain the SPAR-028 analyzer")
        status = _enum(self.status, DifferentialStatus, "status")
        if status != DifferentialStatus.EQUIVALENT.value:
            raise DifferentialExecutionError(
                "receipt cannot admit a non-equivalent differential status"
            )
        if not _bool(self.executor_is_nomination_only, "executor_is_nomination_only"):
            raise DifferentialExecutionError("executor must remain nomination_only")
        if _bool(self.claims_general_equivalence, "claims_general_equivalence"):
            raise DifferentialExecutionError("general Python equivalence is not claimed")
        if not _bool(
            self.traces_prove_only_observations, "traces_prove_only_observations"
        ):
            raise DifferentialExecutionError("traces prove only observations")
        if not _bool(
            self.unresolved_dynamics_lower_autonomy,
            "unresolved_dynamics_lower_autonomy",
        ):
            raise DifferentialExecutionError("unresolved dynamics must lower autonomy")
        resolved: list[DimensionComparison] = []
        for item in self.comparisons:
            if isinstance(item, DimensionComparison):
                resolved.append(item)
            elif isinstance(item, Mapping):
                resolved.append(DimensionComparison.from_dict(item))
            else:
                raise DifferentialExecutionError("comparisons items must be objects")
        if len(resolved) > MAX_DIMENSIONS:
            raise DifferentialExecutionError("comparisons exceed maximum length")
        _fail_required(resolved)
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "wave_cid", _cid(self.wave_cid, "wave_cid"))
        object.__setattr__(
            self, "selection_cid", _cid(self.selection_cid, "selection_cid")
        )
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "profile_cid", _cid(self.profile_cid, "profile_cid"))
        object.__setattr__(
            self,
            "old_observation_cid",
            _cid(self.old_observation_cid, "old_observation_cid"),
        )
        object.__setattr__(
            self,
            "new_observation_cid",
            _cid(self.new_observation_cid, "new_observation_cid"),
        )
        object.__setattr__(self, "comparisons", tuple(resolved))
        object.__setattr__(
            self, "write_paths", _exact_paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self,
            "raw_source_cids",
            _cids(self.raw_source_cids, "raw_source_cids", required=True),
        )
        object.__setattr__(
            self, "validation_commands", _commands(self.validation_commands)
        )
        object.__setattr__(
            self, "worktree_id", _optional_cid(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "network", _network_value(self.network))
        object.__setattr__(self, "analyzer_id", analyzer)
        object.__setattr__(self, "executor_is_nomination_only", True)
        object.__setattr__(self, "claims_general_equivalence", False)
        object.__setattr__(self, "traces_prove_only_observations", True)
        object.__setattr__(self, "unresolved_dynamics_lower_autonomy", True)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def required_dimensions_agree(self) -> bool:
        return all(
            item.outcome == ComparisonOutcome.AGREE.value
            for item in self.comparisons
            if item.required
        )

    @property
    def agreeing_required_dimensions(self) -> tuple[str, ...]:
        return tuple(
            item.dimension
            for item in self.comparisons
            if item.required and item.outcome == ComparisonOutcome.AGREE.value
        )

    @property
    def comparison_cids(self) -> tuple[str, ...]:
        return tuple(item.comparison_cid for item in self.comparisons)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": DIFFERENTIAL_EXECUTION_RECEIPT_SCHEMA,
            "interface": DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "wave_cid": self.wave_cid,
            "selection_cid": self.selection_cid,
            "packet_cid": self.packet_cid,
            "profile_cid": self.profile_cid,
            "old_observation_cid": self.old_observation_cid,
            "new_observation_cid": self.new_observation_cid,
            "comparisons": [item.identity_payload() for item in self.comparisons],
            "write_paths": list(self.write_paths),
            "raw_source_cids": list(self.raw_source_cids),
            "validation_commands": list(self.validation_commands),
            "worktree_id": self.worktree_id,
            "status": self.status,
            "network": NETWORK_DENY,
            "analyzer_id": self.analyzer_id,
            "executor_is_nomination_only": True,
            "claims_general_equivalence": False,
            "traces_prove_only_observations": True,
            "unresolved_dynamics_lower_autonomy": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["comparisons"] = [item.to_dict() for item in self.comparisons]
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DifferentialExecutionReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != DIFFERENTIAL_EXECUTION_RECEIPT_SCHEMA:
            raise DifferentialExecutionError(
                "unsupported DifferentialExecutionReceipt schema"
            )
        if payload.pop("interface") != DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE:
            raise DifferentialExecutionError(
                "unsupported DifferentialExecutionReceipt interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("executor_is_nomination_only") is not True:
            raise DifferentialExecutionError("executor must remain nomination_only")
        if payload.pop("claims_general_equivalence") is not False:
            raise DifferentialExecutionError("general Python equivalence is not claimed")
        if payload.pop("traces_prove_only_observations") is not True:
            raise DifferentialExecutionError("traces prove only observations")
        if payload.pop("unresolved_dynamics_lower_autonomy") is not True:
            raise DifferentialExecutionError("unresolved dynamics must lower autonomy")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise DifferentialExecutionError("receipt analyzer_id must remain SPAR-028")
        comparisons = payload.pop("comparisons")
        resolved = []
        for item in comparisons:
            if isinstance(item, DimensionComparison):
                resolved.append(item)
            elif isinstance(item, Mapping):
                if "comparison_cid" in item:
                    resolved.append(DimensionComparison.from_dict(item))
                else:
                    resolved.append(
                        DimensionComparison(
                            dimension=item.get("dimension"),
                            old_observation_cid=item.get("old_observation_cid", ""),
                            new_observation_cid=item.get("new_observation_cid", ""),
                            outcome=item.get("outcome"),
                            required=item.get("required", True),
                        )
                    )
            else:
                raise DifferentialExecutionError("comparisons items must be objects")
        payload["comparisons"] = resolved
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


def execute_differential_pair(
    *,
    old_workflow: Mapping[str, Any] | WorkflowObservation | Any,
    new_workflow: Mapping[str, Any] | WorkflowObservation | Any,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
    observation_profile: ObservationProfile | Mapping[str, Any] | None = None,
    vector_evidence: Any = None,
) -> DifferentialExecutionReceipt:
    """Run hermetic paired old/new workflows and compare declared dimensions."""

    wave_map = _as_mapping(wave, "SPAR-025 wave")
    selection_map = _as_mapping(selection, "SPAR-026 selection")
    _reject_vector_admission(vector_evidence)

    wave_tree = _tree_id(wave_map.get("tree_id"))
    selection_tree = _tree_id(selection_map.get("tree_id"))
    if wave_tree != selection_tree:
        raise DifferentialExecutionError(
            "SPAR-025 tree_id does not match SPAR-026 tree_id"
        )
    tree_id = wave_tree

    status = _text(wave_map.get("status") or "applied", "SPAR-025 status")
    if status != "applied":
        raise DifferentialExecutionError("SPAR-025 wave must be applied")
    if wave_map.get("writes_repository") is True:
        raise DifferentialExecutionError("SPAR-025 wave cannot write the repository")
    if wave_map.get("executor_is_nomination_only") is False:
        raise DifferentialExecutionError("SPAR-025 executor must remain nomination_only")
    if selection_map.get("adapter_is_nomination_only") is False:
        raise DifferentialExecutionError("SPAR-026 adapter must remain nomination_only")
    if selection_map.get("raw_source_required") is False:
        raise DifferentialExecutionError("raw source required")
    if selection_map.get("datasets_owns_selection") is False:
        raise DifferentialExecutionError("datasets remains the selection authority")

    packet_cid = _selection_packet_cid(selection_map)
    packet_cids = _wave_packet_cids(wave_map)
    if packet_cid not in packet_cids:
        raise DifferentialExecutionError("SPAR-026 packet_cid is not in SPAR-025 packet_cids")

    write_paths = _selection_write_paths(selection_map)
    wave_paths = _wave_write_paths(wave_map)
    if write_paths != wave_paths:
        raise DifferentialExecutionError("SPAR-025/SPAR-026 write_paths do not match")

    sources = _selection_sources(selection_map)
    commands = _selection_commands(selection_map)
    profile = compile_observation_profile(observation_profile)

    old = run_hermetic_workflow(
        old_workflow, role=WorkflowRole.OLD, raw_source_cids=sources
    )
    new = run_hermetic_workflow(
        new_workflow, role=WorkflowRole.NEW, raw_source_cids=sources
    )
    if old.role != WorkflowRole.OLD.value:
        raise DifferentialExecutionError("old workflow role must be old")
    if new.role != WorkflowRole.NEW.value:
        raise DifferentialExecutionError("new workflow role must be new")
    if tuple(old.raw_source_cids) != sources or tuple(new.raw_source_cids) != sources:
        raise DifferentialExecutionError("workflow raw_source_cids must match SPAR-026 sources")

    comparisons = compare_declared_dimensions(old, new, profile=profile)
    _fail_required(comparisons)

    return DifferentialExecutionReceipt(
        tree_id=tree_id,
        wave_cid=_wave_cid(wave_map),
        selection_cid=_selection_cid(selection_map),
        packet_cid=packet_cid,
        profile_cid=profile.profile_cid,
        old_observation_cid=old.observation_cid,
        new_observation_cid=new.observation_cid,
        comparisons=comparisons,
        write_paths=write_paths,
        raw_source_cids=sources,
        validation_commands=commands,
        worktree_id=_optional_cid(wave_map.get("worktree_id"), "worktree_id"),
    )


def compile_differential_receipt(
    receipt: DifferentialExecutionReceipt | Mapping[str, Any],
) -> DifferentialExecutionReceipt:
    if isinstance(receipt, DifferentialExecutionReceipt):
        return receipt
    return DifferentialExecutionReceipt.from_dict(receipt)


def encode_canonical_receipt(receipt: DifferentialExecutionReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> DifferentialExecutionReceipt:
    return DifferentialExecutionReceipt.from_dict(payload)


def encode_canonical_observation(observation: WorkflowObservation) -> dict[str, Any]:
    return observation.to_dict()


def decode_canonical_observation(payload: Mapping[str, Any]) -> WorkflowObservation:
    return WorkflowObservation.from_dict(payload)


def encode_canonical_profile(profile: ObservationProfile) -> dict[str, Any]:
    return profile.to_dict()


def decode_canonical_profile(payload: Mapping[str, Any]) -> ObservationProfile:
    return ObservationProfile.from_dict(payload)


class DifferentialExecutor:
    """Hermetic paired old/new differential executor. Nomination-only."""

    interface: ClassVar[str] = DIFFERENTIAL_EXECUTION_INTERFACE
    schema: ClassVar[str] = DIFFERENTIAL_EXECUTION_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID
    receipt_interface: ClassVar[str] = DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE

    def execute(
        self,
        *,
        old_workflow: Mapping[str, Any] | WorkflowObservation | Any,
        new_workflow: Mapping[str, Any] | WorkflowObservation | Any,
        wave: Mapping[str, Any] | Any,
        selection: Mapping[str, Any] | Any,
        observation_profile: ObservationProfile | Mapping[str, Any] | None = None,
        vector_evidence: Any = None,
    ) -> DifferentialExecutionReceipt:
        return execute_differential_pair(
            old_workflow=old_workflow,
            new_workflow=new_workflow,
            wave=wave,
            selection=selection,
            observation_profile=observation_profile,
            vector_evidence=vector_evidence,
        )

    def compare(
        self,
        old: WorkflowObservation,
        new: WorkflowObservation,
        *,
        profile: ObservationProfile | None = None,
    ) -> tuple[DimensionComparison, ...]:
        return compare_declared_dimensions(old, new, profile=profile)

    def run_hermetic(
        self,
        workflow: Mapping[str, Any] | WorkflowObservation | Any,
        *,
        role: WorkflowRole | str | None = None,
        raw_source_cids: Sequence[str] | None = None,
    ) -> WorkflowObservation:
        return run_hermetic_workflow(
            workflow, role=role, raw_source_cids=raw_source_cids
        )


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise DifferentialExecutionError(
            f"differential executor must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "DECLARED_COMPARISON_OUTCOMES",
    "DECLARED_DIFFERENTIAL_STATUSES",
    "DECLARED_DIMENSIONS",
    "DECLARED_WORKFLOW_ROLES",
    "DIFFERENTIAL_CAN_AUTHORIZE_COMPLETION",
    "DIFFERENTIAL_CAN_AUTHORIZE_TRANSITION",
    "DIFFERENTIAL_CAN_CREATE_AUTHORITY",
    "DIFFERENTIAL_CONTRACT_VERSION",
    "DIFFERENTIAL_EXECUTION_INTERFACE",
    "DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE",
    "DIMENSION_COMPARISON_INTERFACE",
    "DUCKLAKE_IS_AUTHORITY",
    "EXECUTOR_IS_NOMINATION_ONLY",
    "FORBIDDEN_EXECUTION_NAMES",
    "GENERAL_PYTHON_EQUIVALENCE_CLAIMED",
    "GOAL_ID",
    "HERMETIC_PAIRED_EXECUTION",
    "IDENTITY_EXCLUDED_FIELDS",
    "IMPLICIT_INSTALL_FORBIDDEN",
    "IMPLICIT_NETWORK_FORBIDDEN",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "OBSERVATION_PROFILE_INTERFACE",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REQUIRED_TRACE_DIMENSIONS",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TRACES_PROVE_ONLY_OBSERVATIONS",
    "UNRESOLVED_DYNAMICS_LOWER_AUTONOMY",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "WORKFLOW_OBSERVATION_INTERFACE",
    "ComparisonOutcome",
    "DifferentialExecutionError",
    "DifferentialExecutionReceipt",
    "DifferentialExecutor",
    "DifferentialStatus",
    "DimensionComparison",
    "ObservationProfile",
    "TraceDimension",
    "WorkflowObservation",
    "WorkflowRole",
    "assert_not_competing_capsule_family",
    "compare_declared_dimensions",
    "compile_differential_receipt",
    "compile_observation_profile",
    "decode_canonical_observation",
    "decode_canonical_profile",
    "decode_canonical_receipt",
    "differential_cid_profile",
    "differential_descriptor",
    "encode_canonical_observation",
    "encode_canonical_profile",
    "encode_canonical_receipt",
    "execute_differential_pair",
    "provider_free_exports",
    "run_hermetic_workflow",
]
