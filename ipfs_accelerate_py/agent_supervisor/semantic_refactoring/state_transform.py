"""SPAR-022 explicit state-object and boundary-adapter executor.

This module extends current supervisor partition orchestration with
``ExplicitStateObjectPlan@1``.  It consumes SPAR-019
``RefactorTransformationPacket@1`` state/boundary/protocol/wrapper adapter
edits and SPAR-009 state-ownership mappings, then nominates deterministic
explicit state objects, protocols, boundary adapters, or injection only
when ownership, lifecycle, and synchronization obligations are complete.

SPAR-009 payloads are ingested as mappings only.  This module does not
replace datasets semantic authority, does not apply CST transforms, and
cannot authorize a transition, completion, or competing authority.
Vector, model, and heuristic evidence cannot admit a transform.
Missing release is recorded as absent and never guessed.  Duplicated
unique mutable owners fail closed.  Observational metadata is excluded
from identity.  Dry-run is deterministic and never mutates.
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
from .transformation_packet import (
    ANALYZER_ID as SPAR019_ANALYZER_ID,
    AdapterKind,
    EditKind,
    RefactorTransformationPacket,
)


TASK_ID: Final[str] = "SPAR-022"
GOAL_ID: Final[str] = "SPAR-G042"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.state_transform@1"
)

EXPLICIT_STATE_OBJECT_INTERFACE: Final[str] = "ExplicitStateObject@1"
EXPLICIT_STATE_OBJECT_PLAN_INTERFACE: Final[str] = "ExplicitStateObjectPlan@1"
EXPLICIT_STATE_OBJECT_RECEIPT_INTERFACE: Final[str] = (
    "ExplicitStateObjectReceipt@1"
)

EXPLICIT_STATE_OBJECT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/explicit-state-object@1"
)
EXPLICIT_STATE_OBJECT_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/explicit-state-object-plan@1"
)
EXPLICIT_STATE_OBJECT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/explicit-state-object-receipt@1"
)

TRANSFORM_CONTRACT_VERSION: Final[str] = "1"

TRANSFORM_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
TRANSFORM_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
TRANSFORM_CAN_CREATE_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
EXECUTOR_IS_NOMINATION_ONLY: Final[bool] = True
PLAN_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT: Final[bool] = True
DUPLICATED_MUTABLE_STATE_REJECTED: Final[bool] = True
MISSING_RELEASE_IS_NOT_GUESSED: Final[bool] = True
UNKNOWN_UNIQUENESS_WIDENS_FRONTIER: Final[bool] = True
INCOMPLETE_OBLIGATIONS_ARE_TYPED_TERMINAL: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_TRANSFORMS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_PATH_CHARS: Final[int] = 1_024
MAX_OWNERS: Final[int] = 4_096
MAX_RELATIONS: Final[int] = 8_192

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

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "projection_is_authority",
)

HANDLED_ADAPTER_KINDS: Final[frozenset[str]] = frozenset(
    {
        AdapterKind.STATE.value,
        AdapterKind.BOUNDARY.value,
        AdapterKind.PROTOCOL.value,
        AdapterKind.WRAPPER.value,
    }
)

UNIQUE_OWNERSHIP: Final[str] = "unique"
SHARED_OWNERSHIP: Final[str] = "shared"
UNKNOWN_OWNERSHIP: Final[str] = "unknown"
DECLARED_UNIQUENESS: Final[frozenset[str]] = frozenset(
    {UNIQUE_OWNERSHIP, SHARED_OWNERSHIP, UNKNOWN_OWNERSHIP}
)

LIFECYCLE_INITIALIZE: Final[str] = "initialize"
LIFECYCLE_ACQUIRE: Final[str] = "acquire"
LIFECYCLE_RELEASE: Final[str] = "release"
LIFECYCLE_FINALIZE: Final[str] = "finalize"
DECLARED_LIFECYCLE_KINDS: Final[frozenset[str]] = frozenset(
    {
        LIFECYCLE_INITIALIZE,
        LIFECYCLE_ACQUIRE,
        LIFECYCLE_RELEASE,
        LIFECYCLE_FINALIZE,
    }
)
REQUIRED_LIFECYCLE_START: Final[frozenset[str]] = frozenset(
    {LIFECYCLE_INITIALIZE, LIFECYCLE_ACQUIRE}
)

DECLARED_SYNCHRONIZATION_KINDS: Final[frozenset[str]] = frozenset(
    {"lock", "transaction", "condition"}
)

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

_ADAPTER_TO_TRANSFORM: Final[Mapping[str, str]] = {
    AdapterKind.STATE.value: "state_object",
    AdapterKind.PROTOCOL.value: "protocol",
    AdapterKind.BOUNDARY.value: "boundary_adapter",
    AdapterKind.WRAPPER.value: "injection",
}


class StateTransformError(ValueError):
    """Fail-closed violation of a SPAR-022 state-transform contract."""


class ExplicitStateObjectKind(str, Enum):
    STATE_OBJECT = "state_object"
    PROTOCOL = "protocol"
    BOUNDARY_ADAPTER = "boundary_adapter"
    INJECTION = "injection"


DECLARED_EXPLICIT_STATE_OBJECT_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in ExplicitStateObjectKind
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise StateTransformError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise StateTransformError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise StateTransformError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise StateTransformError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise StateTransformError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise StateTransformError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise StateTransformError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise StateTransformError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise StateTransformError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise StateTransformError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise StateTransformError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise StateTransformError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise StateTransformError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise StateTransformError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise StateTransformError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise StateTransformError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise StateTransformError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise StateTransformError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise StateTransformError(f"unknown {name}: {text}") from exc


def _project(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _project(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _project(to_dict())
    raise StateTransformError(
        f"unsupported projected type {type(value).__name__}"
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise StateTransformError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def state_transform_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if payload.pop(flag) is not False:
            raise StateTransformError(f"{name} cannot claim {flag}")


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise StateTransformError(f"{name} exceeds path bound")
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
        raise StateTransformError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise StateTransformError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise StateTransformError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise StateTransformError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise StateTransformError(f"{name} exceeds path bound")
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class ExplicitStateObject:
    """One explicit state object, protocol, boundary adapter, or injection.

    Nomination only.
    """

    transform_kind: ExplicitStateObjectKind | str
    owner_id: str
    uniqueness: str
    source_module: str
    destination_module: str
    write_paths: Sequence[str]
    preimage_cid: str
    packet_cid: str
    tree_id: str
    obligation_id: str = ""
    adapter_kind: str = ""
    alias_set_id: str = ""
    lifecycle_ids: Sequence[str] = ()
    synchronization_ids: Sequence[str] = ()
    complete_obligations: bool = True
    admitted: bool = True

    interface: ClassVar[str] = EXPLICIT_STATE_OBJECT_INTERFACE
    schema: ClassVar[str] = EXPLICIT_STATE_OBJECT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "transform_kind",
            "owner_id",
            "uniqueness",
            "source_module",
            "destination_module",
            "write_paths",
            "preimage_cid",
            "packet_cid",
            "tree_id",
            "obligation_id",
            "adapter_kind",
            "alias_set_id",
            "lifecycle_ids",
            "synchronization_ids",
            "complete_obligations",
            "admitted",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "transform_is_nomination_only",
            "transform_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _enum(self.transform_kind, ExplicitStateObjectKind, "transform_kind")
        uniqueness = _text(self.uniqueness, "uniqueness")
        if uniqueness not in DECLARED_UNIQUENESS:
            raise StateTransformError("unknown uniqueness")
        if uniqueness != UNIQUE_OWNERSHIP:
            raise StateTransformError(
                "explicit state transform requires unique ownership"
            )
        complete = _bool(self.complete_obligations, "complete_obligations")
        admitted = _bool(self.admitted, "admitted")
        if not complete:
            raise StateTransformError(
                "incomplete ownership/lifecycle/synchronization obligations "
                "are a typed terminal"
            )
        if not admitted:
            raise StateTransformError(
                "explicit state transform requires an admitted unique owner"
            )
        adapter = _text(self.adapter_kind, "adapter_kind", empty=True)
        if adapter and adapter not in HANDLED_ADAPTER_KINDS:
            raise StateTransformError("unknown adapter_kind")
        expected = _ADAPTER_TO_TRANSFORM.get(adapter)
        if adapter and expected != kind:
            raise StateTransformError("adapter_kind does not match transform_kind")
        object.__setattr__(self, "transform_kind", kind)
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "uniqueness", uniqueness)
        object.__setattr__(self, "source_module", _text(self.source_module, "source_module"))
        object.__setattr__(
            self,
            "destination_module",
            _text(self.destination_module, "destination_module"),
        )
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id", empty=True)
        )
        object.__setattr__(self, "adapter_kind", adapter)
        object.__setattr__(
            self, "alias_set_id", _text(self.alias_set_id, "alias_set_id", empty=True)
        )
        object.__setattr__(
            self,
            "lifecycle_ids",
            _unique_sorted_text(
                list(self.lifecycle_ids), "lifecycle_ids", limit=MAX_RELATIONS
            ),
        )
        object.__setattr__(
            self,
            "synchronization_ids",
            _unique_sorted_text(
                list(self.synchronization_ids),
                "synchronization_ids",
                limit=MAX_RELATIONS,
            ),
        )
        object.__setattr__(self, "complete_obligations", True)
        object.__setattr__(self, "admitted", True)

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
    def transform_is_nomination_only(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXPLICIT_STATE_OBJECT_SCHEMA,
            "interface": EXPLICIT_STATE_OBJECT_INTERFACE,
            "transform_kind": self.transform_kind,
            "owner_id": self.owner_id,
            "uniqueness": self.uniqueness,
            "source_module": self.source_module,
            "destination_module": self.destination_module,
            "write_paths": list(self.write_paths),
            "preimage_cid": self.preimage_cid,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "obligation_id": self.obligation_id,
            "adapter_kind": self.adapter_kind,
            "alias_set_id": self.alias_set_id,
            "lifecycle_ids": list(self.lifecycle_ids),
            "synchronization_ids": list(self.synchronization_ids),
            "complete_obligations": True,
            "admitted": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "transform_is_nomination_only": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def transform_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["transform_cid"] = self.transform_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExplicitStateObject":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("transform_cid")
        if payload.pop("schema") != EXPLICIT_STATE_OBJECT_SCHEMA:
            raise StateTransformError("unsupported ExplicitStateObject schema")
        if payload.pop("interface") != EXPLICIT_STATE_OBJECT_INTERFACE:
            raise StateTransformError("unsupported ExplicitStateObject interface")
        _pop_authority_flags(payload, "ExplicitStateObject")
        if payload.pop("transform_is_nomination_only") is not True:
            raise StateTransformError("transform must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.transform_cid, "ExplicitStateObject transform_cid")
        return result


@dataclass(frozen=True, slots=True)
class ExplicitStateObjectPlan:
    """Deterministic SPAR-022 explicit state-object plan. Nomination only."""

    transform_cids: Sequence[str]
    write_paths: Sequence[str]
    preimage_cid: str
    packet_cid: str
    tree_id: str
    obligation_ids: Sequence[str] = ()
    owner_ids: Sequence[str] = ()
    complete_obligations: bool = True
    unique_owners: bool = True
    no_duplicated_mutable_state: bool = True
    missing_release_guessed: bool = False

    interface: ClassVar[str] = EXPLICIT_STATE_OBJECT_PLAN_INTERFACE
    schema: ClassVar[str] = EXPLICIT_STATE_OBJECT_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "transform_cids",
            "write_paths",
            "preimage_cid",
            "packet_cid",
            "tree_id",
            "obligation_ids",
            "owner_ids",
            "complete_obligations",
            "unique_owners",
            "no_duplicated_mutable_state",
            "missing_release_guessed",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "plan_is_nomination_only",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        if not _bool(self.complete_obligations, "complete_obligations"):
            raise StateTransformError(
                "incomplete ownership/lifecycle/synchronization obligations "
                "are a typed terminal"
            )
        if not _bool(self.unique_owners, "unique_owners"):
            raise StateTransformError(
                "explicit state transform requires unique ownership"
            )
        if not _bool(self.no_duplicated_mutable_state, "no_duplicated_mutable_state"):
            raise StateTransformError("duplicated mutable state")
        if _bool(self.missing_release_guessed, "missing_release_guessed"):
            raise StateTransformError("missing release is not guessed")
        transforms = tuple(
            sorted(_cid(item, "transform_cids") for item in self.transform_cids)
        )
        if not transforms:
            raise StateTransformError("plan requires transform_cids")
        if len(transforms) != len(set(transforms)):
            raise StateTransformError("transform_cids must not contain duplicates")
        if len(transforms) > MAX_TRANSFORMS:
            raise StateTransformError("transform_cids exceed maximum length")
        owners = _unique_sorted_text(list(self.owner_ids), "owner_ids", limit=MAX_OWNERS)
        if not owners:
            raise StateTransformError("plan requires owner_ids")
        object.__setattr__(self, "transform_cids", transforms)
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "obligation_ids",
            _unique_sorted_text(
                list(self.obligation_ids), "obligation_ids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(self, "owner_ids", owners)
        object.__setattr__(self, "complete_obligations", True)
        object.__setattr__(self, "unique_owners", True)
        object.__setattr__(self, "no_duplicated_mutable_state", True)
        object.__setattr__(self, "missing_release_guessed", False)

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
    def plan_is_nomination_only(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXPLICIT_STATE_OBJECT_PLAN_SCHEMA,
            "interface": EXPLICIT_STATE_OBJECT_PLAN_INTERFACE,
            "transform_cids": list(self.transform_cids),
            "write_paths": list(self.write_paths),
            "preimage_cid": self.preimage_cid,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "obligation_ids": list(self.obligation_ids),
            "owner_ids": list(self.owner_ids),
            "complete_obligations": True,
            "unique_owners": True,
            "no_duplicated_mutable_state": True,
            "missing_release_guessed": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "plan_is_nomination_only": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def plan_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["plan_cid"] = self.plan_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExplicitStateObjectPlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != EXPLICIT_STATE_OBJECT_PLAN_SCHEMA:
            raise StateTransformError("unsupported ExplicitStateObjectPlan schema")
        if payload.pop("interface") != EXPLICIT_STATE_OBJECT_PLAN_INTERFACE:
            raise StateTransformError(
                "unsupported ExplicitStateObjectPlan interface"
            )
        _pop_authority_flags(payload, "ExplicitStateObjectPlan")
        if payload.pop("plan_is_nomination_only") is not True:
            raise StateTransformError("plan must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "ExplicitStateObjectPlan plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class ExplicitStateObjectReceipt:
    """Body-free SPAR-022 execution receipt. Independent validation remains separate."""

    tree_id: str
    packet_cid: str
    preimage_cid: str
    transform_cids: Sequence[str]
    plan_cid: str
    write_paths: Sequence[str]
    owner_ids: Sequence[str] = ()
    analyzer_id: str = ANALYZER_ID
    complete_obligations: bool = True
    unique_owners: bool = True
    no_duplicated_mutable_state: bool = True
    missing_release_guessed: bool = False
    preimage_verified: bool = True
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = EXPLICIT_STATE_OBJECT_RECEIPT_INTERFACE
    schema: ClassVar[str] = EXPLICIT_STATE_OBJECT_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "preimage_cid",
            "transform_cids",
            "plan_cid",
            "write_paths",
            "owner_ids",
            "analyzer_id",
            "complete_obligations",
            "unique_owners",
            "no_duplicated_mutable_state",
            "missing_release_guessed",
            "preimage_verified",
            "mutated",
            "deterministic",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "executor_is_nomination_only",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise StateTransformError("analyzer_id must remain the SPAR-022 analyzer")
        if not _bool(self.complete_obligations, "complete_obligations"):
            raise StateTransformError(
                "incomplete ownership/lifecycle/synchronization obligations "
                "are a typed terminal"
            )
        if not _bool(self.unique_owners, "unique_owners"):
            raise StateTransformError(
                "explicit state transform requires unique ownership"
            )
        if not _bool(self.no_duplicated_mutable_state, "no_duplicated_mutable_state"):
            raise StateTransformError("duplicated mutable state")
        if _bool(self.missing_release_guessed, "missing_release_guessed"):
            raise StateTransformError("missing release is not guessed")
        if not _bool(self.preimage_verified, "preimage_verified"):
            raise StateTransformError("receipt cannot skip preimage verification")
        if _bool(self.mutated, "mutated"):
            raise StateTransformError("executor cannot mutate")
        if not _bool(self.deterministic, "deterministic"):
            raise StateTransformError("executor must remain deterministic")
        transforms = tuple(
            sorted(_cid(item, "transform_cids") for item in self.transform_cids)
        )
        if not transforms:
            raise StateTransformError("receipt requires transform_cids")
        if len(transforms) != len(set(transforms)):
            raise StateTransformError("transform_cids must not contain duplicates")
        if len(transforms) > MAX_TRANSFORMS:
            raise StateTransformError("transform_cids exceed maximum length")
        owners = _unique_sorted_text(list(self.owner_ids), "owner_ids", limit=MAX_OWNERS)
        if not owners:
            raise StateTransformError("receipt requires owner_ids")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "transform_cids", transforms)
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "owner_ids", owners)
        object.__setattr__(self, "analyzer_id", analyzer)
        object.__setattr__(self, "complete_obligations", True)
        object.__setattr__(self, "unique_owners", True)
        object.__setattr__(self, "no_duplicated_mutable_state", True)
        object.__setattr__(self, "missing_release_guessed", False)
        object.__setattr__(self, "preimage_verified", True)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)

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
    def executor_is_nomination_only(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXPLICIT_STATE_OBJECT_RECEIPT_SCHEMA,
            "interface": EXPLICIT_STATE_OBJECT_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "preimage_cid": self.preimage_cid,
            "transform_cids": list(self.transform_cids),
            "plan_cid": self.plan_cid,
            "write_paths": list(self.write_paths),
            "owner_ids": list(self.owner_ids),
            "analyzer_id": self.analyzer_id,
            "complete_obligations": True,
            "unique_owners": True,
            "no_duplicated_mutable_state": True,
            "missing_release_guessed": False,
            "preimage_verified": True,
            "mutated": False,
            "deterministic": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "executor_is_nomination_only": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExplicitStateObjectReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != EXPLICIT_STATE_OBJECT_RECEIPT_SCHEMA:
            raise StateTransformError(
                "unsupported ExplicitStateObjectReceipt schema"
            )
        if payload.pop("interface") != EXPLICIT_STATE_OBJECT_RECEIPT_INTERFACE:
            raise StateTransformError(
                "unsupported ExplicitStateObjectReceipt interface"
            )
        _pop_authority_flags(payload, "ExplicitStateObjectReceipt")
        if payload.pop("executor_is_nomination_only") is not True:
            raise StateTransformError("executor must remain nomination_only")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "ExplicitStateObjectReceipt receipt_cid"
        )
        return result


def _coerce_packet(
    value: RefactorTransformationPacket | Mapping[str, Any],
) -> RefactorTransformationPacket:
    if isinstance(value, RefactorTransformationPacket):
        packet = value
    elif isinstance(value, Mapping):
        packet = RefactorTransformationPacket.from_dict(value)
    else:
        raise StateTransformError(
            "packet must be a SPAR-019 RefactorTransformationPacket"
        )
    if packet.analyzer_id != SPAR019_ANALYZER_ID:
        raise StateTransformError("packet must remain the SPAR-019 analyzer")
    return packet


def _as_mapping_list(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise StateTransformError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(_mapping(item, name))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            items.append(_mapping(to_dict(), name))
            continue
        raise StateTransformError(f"{name} entries must be objects")
    return tuple(items)


def _optional_bool(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    return _bool(value, name)


def _optional_text_list(value: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if value in (None, (), ""):
        return ()
    if isinstance(value, str):
        raise StateTransformError(f"{name} must be a list")
    return _unique_sorted_text(list(value), name, limit=limit)


def _reject_non_admitting_evidence(item: Mapping[str, Any], name: str) -> None:
    evidence = item.get("evidence_class")
    if evidence in (None, ""):
        return
    evidence_text = _text(evidence, "evidence_class")
    if evidence_text in _NON_ADMITTING_EVIDENCE:
        raise StateTransformError(
            f"vector, model, or heuristic evidence cannot admit {name}"
        )


def _coerce_ownership_graph(value: Any) -> dict[str, Any]:
    if value in (None, (), {}):
        return {
            "owners": (),
            "alias_sets": (),
            "lifecycle_relations": (),
            "synchronization_relations": (),
            "extraction_candidates": (),
            "unresolved": (),
        }
    payload = _mapping(value, "ownership_graph")
    owners = _as_mapping_list(payload.get("owners") or (), "owners")
    aliases = _as_mapping_list(
        payload.get("alias_sets") or payload.get("aliases") or (), "alias_sets"
    )
    lifecycle = _as_mapping_list(
        payload.get("lifecycle_relations") or payload.get("lifecycle") or (),
        "lifecycle_relations",
    )
    synchronization = _as_mapping_list(
        payload.get("synchronization_relations")
        or payload.get("synchronization")
        or (),
        "synchronization_relations",
    )
    candidates = _as_mapping_list(
        payload.get("extraction_candidates") or payload.get("candidates") or (),
        "extraction_candidates",
    )
    unresolved = _as_mapping_list(payload.get("unresolved") or (), "unresolved")
    if len(owners) > MAX_OWNERS:
        raise StateTransformError("owners exceed maximum length")
    if len(lifecycle) > MAX_RELATIONS or len(synchronization) > MAX_RELATIONS:
        raise StateTransformError("ownership_graph exceeds maximum length")
    for item in (*owners, *candidates):
        _reject_non_admitting_evidence(item, "a state owner")
    return {
        "owners": owners,
        "alias_sets": aliases,
        "lifecycle_relations": lifecycle,
        "synchronization_relations": synchronization,
        "extraction_candidates": candidates,
        "unresolved": unresolved,
    }


def _owner_id_of(item: Mapping[str, Any]) -> str:
    return _text(
        item.get("owner_id") or item.get("id") or "",
        "owner_id",
        empty=True,
    )


def _alias_id_of(item: Mapping[str, Any]) -> str:
    return _text(
        item.get("alias_set_id") or item.get("id") or "",
        "alias_set_id",
        empty=True,
    )


def _relation_id_of(item: Mapping[str, Any]) -> str:
    return _text(
        item.get("relation_id") or item.get("id") or "",
        "relation_id",
        empty=True,
    )


def _uniqueness_of(item: Mapping[str, Any]) -> str:
    uniqueness = _text(item.get("uniqueness") or UNIQUE_OWNERSHIP, "uniqueness")
    if uniqueness not in DECLARED_UNIQUENESS:
        raise StateTransformError("unknown uniqueness")
    return uniqueness


def _reject_unknown_uniqueness(item: Mapping[str, Any]) -> None:
    uniqueness = _uniqueness_of(item)
    if uniqueness == UNKNOWN_OWNERSHIP:
        raise StateTransformError(
            "unknown uniqueness widens the frontier and cannot admit a transform"
        )


def _reject_duplicated_unique_owners(graph: Mapping[str, Any]) -> None:
    alias_members: dict[str, tuple[str, ...]] = {}
    for alias in graph["alias_sets"]:
        alias_id = _alias_id_of(alias)
        if not alias_id:
            continue
        members = _optional_text_list(
            alias.get("member_ids") or alias.get("members") or (),
            "member_ids",
            limit=MAX_MEMBERS,
        )
        representative = _text(
            alias.get("representative_id") or "", "representative_id", empty=True
        )
        if not members and representative:
            members = (representative,)
        alias_members[alias_id] = members
    unique_members: dict[str, str] = {}
    unique_alias_ids: list[str] = []
    for owner in graph["owners"]:
        if _uniqueness_of(owner) != UNIQUE_OWNERSHIP:
            continue
        if not _optional_bool(owner.get("mutable"), "mutable", True):
            continue
        owner_id = _owner_id_of(owner)
        alias_id = _alias_id_of(owner)
        unique_alias_ids.append(alias_id)
        members = alias_members.get(alias_id, ())
        if not members:
            symbol = _text(
                owner.get("owning_symbol_id") or owner_id, "owning_symbol_id", empty=True
            )
            members = (symbol,) if symbol else (owner_id,)
        for member in members:
            prior = unique_members.get(member)
            if prior is not None and prior != owner_id:
                raise StateTransformError("duplicated mutable state")
            unique_members[member] = owner_id
    nonempty_alias_ids = [item for item in unique_alias_ids if item]
    if len(nonempty_alias_ids) != len(set(nonempty_alias_ids)):
        raise StateTransformError("duplicated mutable state")


def _relations_for_owner(
    relations: Sequence[Mapping[str, Any]], owner_id: str
) -> tuple[dict[str, Any], ...]:
    matched: list[dict[str, Any]] = []
    for item in relations:
        if _owner_id_of(item) == owner_id:
            matched.append(item)
    return tuple(matched)


def _lifecycle_kind(item: Mapping[str, Any]) -> str:
    kind = _text(item.get("kind") or "", "lifecycle_kind")
    if kind not in DECLARED_LIFECYCLE_KINDS:
        raise StateTransformError("unknown lifecycle kind")
    return kind


def _synchronization_kind(item: Mapping[str, Any]) -> str:
    kind = _text(item.get("kind") or "", "synchronization_kind")
    if kind not in DECLARED_SYNCHRONIZATION_KINDS:
        raise StateTransformError("unknown synchronization kind")
    return kind


def _ids_exist(claimed: Sequence[str], available: set[str], name: str) -> None:
    missing = [item for item in claimed if item not in available]
    if missing:
        raise StateTransformError(
            f"incomplete ownership/lifecycle/synchronization obligations "
            f"are a typed terminal"
        )


def verify_complete_obligations(
    ownership_graph: Mapping[str, Any] | None,
    owner_id: str,
) -> dict[str, Any]:
    """Verify SPAR-009 complete ownership/lifecycle/synchronization obligations.

    Does not authorize a transition. Missing release is never guessed.
    """

    graph = _coerce_ownership_graph(ownership_graph)
    owner_text = _text(owner_id, "owner_id")
    _reject_duplicated_unique_owners(graph)
    owners = [item for item in graph["owners"] if _owner_id_of(item) == owner_text]
    if not owners:
        raise StateTransformError(
            "incomplete ownership/lifecycle/synchronization obligations "
            "are a typed terminal"
        )
    owner = owners[0]
    _reject_unknown_uniqueness(owner)
    uniqueness = _uniqueness_of(owner)
    if uniqueness != UNIQUE_OWNERSHIP:
        raise StateTransformError(
            "explicit state transform requires unique ownership"
        )
    _reject_non_admitting_evidence(owner, "a state owner")
    unresolved_subjects = {
        _text(item.get("subject_id") or "", "subject_id", empty=True)
        for item in graph["unresolved"]
    }
    if owner_text in unresolved_subjects:
        raise StateTransformError(
            "unresolved state ownership cannot admit a transform"
        )
    lifecycle = _relations_for_owner(graph["lifecycle_relations"], owner_text)
    synchronization = _relations_for_owner(
        graph["synchronization_relations"], owner_text
    )
    for item in lifecycle:
        _lifecycle_kind(item)
        if _optional_bool(item.get("missing_counterpart"), "missing_counterpart", False):
            raise StateTransformError("missing release is not guessed")
    for item in synchronization:
        _synchronization_kind(item)
    mutable = _optional_bool(owner.get("mutable"), "mutable", True)
    kinds = {_lifecycle_kind(item) for item in lifecycle}
    if mutable:
        if not (kinds & REQUIRED_LIFECYCLE_START):
            raise StateTransformError(
                "incomplete ownership/lifecycle/synchronization obligations "
                "are a typed terminal"
            )
        if LIFECYCLE_RELEASE not in kinds:
            raise StateTransformError("missing release is not guessed")
    candidates = [
        item
        for item in graph["extraction_candidates"]
        if _owner_id_of(item) == owner_text
    ]
    lifecycle_ids = tuple(
        sorted(_relation_id_of(item) for item in lifecycle if _relation_id_of(item))
    )
    synchronization_ids = tuple(
        sorted(
            _relation_id_of(item) for item in synchronization if _relation_id_of(item)
        )
    )
    alias_set_id = _alias_id_of(owner)
    admitted = True
    complete = True
    if candidates:
        candidate = candidates[0]
        _reject_unknown_uniqueness(candidate)
        _reject_non_admitting_evidence(candidate, "an extraction candidate")
        if _uniqueness_of(candidate) != UNIQUE_OWNERSHIP:
            raise StateTransformError(
                "explicit state transform requires unique ownership"
            )
        complete = _optional_bool(
            candidate.get("complete_obligations"), "complete_obligations", False
        )
        admitted = _optional_bool(candidate.get("admitted"), "admitted", False)
        if not complete or not admitted:
            raise StateTransformError(
                "incomplete ownership/lifecycle/synchronization obligations "
                "are a typed terminal"
            )
        claimed_lifecycle = _optional_text_list(
            candidate.get("lifecycle_ids") or (),
            "lifecycle_ids",
            limit=MAX_RELATIONS,
        )
        claimed_sync = _optional_text_list(
            candidate.get("synchronization_ids") or (),
            "synchronization_ids",
            limit=MAX_RELATIONS,
        )
        available_lifecycle = {item for item in lifecycle_ids}
        available_sync = {item for item in synchronization_ids}
        _ids_exist(claimed_lifecycle, available_lifecycle, "lifecycle_ids")
        _ids_exist(claimed_sync, available_sync, "synchronization_ids")
        if claimed_lifecycle:
            lifecycle_ids = claimed_lifecycle
        if claimed_sync:
            synchronization_ids = claimed_sync
        candidate_alias = _alias_id_of(candidate)
        if candidate_alias:
            alias_set_id = candidate_alias
    return {
        "owner_id": owner_text,
        "uniqueness": uniqueness,
        "alias_set_id": alias_set_id,
        "lifecycle_ids": lifecycle_ids,
        "synchronization_ids": synchronization_ids,
        "complete_obligations": True,
        "admitted": True,
        "mutable": mutable,
    }


def verify_preimages(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    claimed_preimage_cid: str = "",
    source_cids: Sequence[str] | None = None,
) -> str:
    """Verify SPAR-019 packet preimages. Does not authorize a transition."""

    resolved = _coerce_packet(packet)
    preimage_cid = resolved.preimage.preimage_cid
    if claimed_preimage_cid:
        claimed = _cid(claimed_preimage_cid, "claimed_preimage_cid")
        if claimed != preimage_cid:
            raise StateTransformError("preimage does not verify")
    if source_cids is not None:
        claimed_sources = _unique_sorted_text(
            list(source_cids), "source_cids", limit=MAX_MEMBERS
        )
        if set(claimed_sources) != set(resolved.preimage.source_cids):
            raise StateTransformError("preimage does not verify")
    return preimage_cid


def _adapter_edits(packet: RefactorTransformationPacket) -> tuple[Any, ...]:
    return tuple(
        item
        for item in packet.edits
        if item.kind == EditKind.ADAPTER.value
        and item.adapter_kind in HANDLED_ADAPTER_KINDS
    )


def _sort_transforms(
    items: Sequence[ExplicitStateObject],
) -> tuple[ExplicitStateObject, ...]:
    unique: dict[str, ExplicitStateObject] = {}
    for item in items:
        unique[item.transform_cid] = item
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.transform_kind,
                item.owner_id,
                item.source_module,
                item.destination_module,
                item.obligation_id,
                item.adapter_kind,
                item.transform_cid,
            ),
        )
    )
    if len(ordered) > MAX_TRANSFORMS:
        raise StateTransformError("explicit state transforms exceed maximum length")
    return ordered


def _owner_tokens(owner: Mapping[str, Any]) -> set[str]:
    tokens = {
        _owner_id_of(owner),
        _alias_id_of(owner),
        _text(owner.get("owning_symbol_id") or "", "owning_symbol_id", empty=True),
    }
    return {item for item in tokens if item}


def _match_owner(
    graph: Mapping[str, Any],
    *,
    member_ids: Sequence[str],
    obligation_id: str,
) -> dict[str, Any]:
    tokens = {item for item in (*member_ids, obligation_id) if item}
    matched: list[dict[str, Any]] = []
    for owner in graph["owners"]:
        owner_tokens = _owner_tokens(owner)
        if tokens & owner_tokens:
            matched.append(owner)
            continue
        if obligation_id and obligation_id == _text(
            owner.get("obligation_id") or "", "obligation_id", empty=True
        ):
            matched.append(owner)
    if len(matched) == 1:
        return matched[0]
    candidates = graph["extraction_candidates"]
    candidate_matches: list[dict[str, Any]] = []
    for candidate in candidates:
        owner_id = _owner_id_of(candidate)
        if owner_id in tokens or obligation_id == owner_id:
            candidate_matches.append(candidate)
    if len(candidate_matches) == 1:
        owner_id = _owner_id_of(candidate_matches[0])
        owners = [item for item in graph["owners"] if _owner_id_of(item) == owner_id]
        if len(owners) == 1:
            return owners[0]
        if owners:
            return owners[0]
        return candidate_matches[0]
    if not matched and not candidate_matches:
        raise StateTransformError(
            "incomplete ownership/lifecycle/synchronization obligations "
            "are a typed terminal"
        )
    raise StateTransformError("duplicated mutable state")


def _collect_adapter_transforms(
    packet: RefactorTransformationPacket,
    graph: Mapping[str, Any],
) -> tuple[ExplicitStateObject, ...]:
    nominated: list[ExplicitStateObject] = []
    for edit in _adapter_edits(packet):
        kind = _ADAPTER_TO_TRANSFORM[edit.adapter_kind]
        members = edit.member_ids or (edit.obligation_id or edit.source_id,)
        owner = _match_owner(
            graph, member_ids=members, obligation_id=edit.obligation_id
        )
        owner_id = _owner_id_of(owner) or _text(members[0], "owner_id")
        obligations = verify_complete_obligations(graph, owner_id)
        nominated.append(
            ExplicitStateObject(
                transform_kind=kind,
                owner_id=obligations["owner_id"],
                uniqueness=obligations["uniqueness"],
                source_module=edit.source_id,
                destination_module=edit.destination_id,
                write_paths=edit.write_paths,
                preimage_cid=packet.preimage.preimage_cid,
                packet_cid=packet.packet_cid,
                tree_id=packet.tree_id,
                obligation_id=edit.obligation_id,
                adapter_kind=edit.adapter_kind,
                alias_set_id=obligations["alias_set_id"],
                lifecycle_ids=obligations["lifecycle_ids"],
                synchronization_ids=obligations["synchronization_ids"],
                complete_obligations=True,
                admitted=True,
            )
        )
    return _sort_transforms(nominated)


def _collect_extraction_transforms(
    packet: RefactorTransformationPacket,
    graph: Mapping[str, Any],
    adapters: Sequence[ExplicitStateObject],
) -> tuple[ExplicitStateObject, ...]:
    covered = {
        (item.owner_id, ExplicitStateObjectKind.STATE_OBJECT.value)
        for item in adapters
        if item.transform_kind == ExplicitStateObjectKind.STATE_OBJECT.value
    }
    destination = packet.expected_delta.destination_module_ids[0]
    source = packet.edits[0].source_id if packet.edits else destination
    nominated: list[ExplicitStateObject] = []
    for candidate in graph["extraction_candidates"]:
        owner_id = _owner_id_of(candidate)
        if not owner_id:
            continue
        if (owner_id, ExplicitStateObjectKind.STATE_OBJECT.value) in covered:
            continue
        if not _optional_bool(candidate.get("admitted"), "admitted", False):
            continue
        obligations = verify_complete_obligations(graph, owner_id)
        nominated.append(
            ExplicitStateObject(
                transform_kind=ExplicitStateObjectKind.STATE_OBJECT,
                owner_id=obligations["owner_id"],
                uniqueness=obligations["uniqueness"],
                source_module=source,
                destination_module=destination,
                write_paths=packet.effect_scope.write_paths,
                preimage_cid=packet.preimage.preimage_cid,
                packet_cid=packet.packet_cid,
                tree_id=packet.tree_id,
                obligation_id=_text(
                    candidate.get("obligation_id") or "", "obligation_id", empty=True
                ),
                adapter_kind=AdapterKind.STATE.value,
                alias_set_id=obligations["alias_set_id"],
                lifecycle_ids=obligations["lifecycle_ids"],
                synchronization_ids=obligations["synchronization_ids"],
                complete_obligations=True,
                admitted=True,
            )
        )
    return _sort_transforms(nominated)


def _require_handled(transforms: Sequence[ExplicitStateObject]) -> None:
    if transforms:
        return
    raise StateTransformError(
        "packet requires explicit state-object, protocol, boundary-adapter, "
        "or injection edits"
    )


def compile_explicit_state_objects(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    ownership_graph: Mapping[str, Any] | None = None,
    claimed_preimage_cid: str = "",
    source_cids: Sequence[str] | None = None,
) -> tuple[ExplicitStateObject, ...]:
    """Nominate exact explicit state-object transforms from a SPAR-019 packet."""

    resolved = _coerce_packet(packet)
    verify_preimages(
        resolved,
        claimed_preimage_cid=claimed_preimage_cid,
        source_cids=source_cids,
    )
    graph = _coerce_ownership_graph(ownership_graph)
    _reject_duplicated_unique_owners(graph)
    adapters = _collect_adapter_transforms(resolved, graph)
    extractions = _collect_extraction_transforms(resolved, graph, adapters)
    transforms = _sort_transforms((*adapters, *extractions))
    _require_handled(transforms)
    return transforms


def compile_explicit_state_object_plan(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ExplicitStateObjectPlan:
    """Compile the unique SPAR-022 ExplicitStateObjectPlan for one packet."""

    resolved = _coerce_packet(packet)
    transforms = compile_explicit_state_objects(resolved, **kwargs)
    obligations = tuple(
        sorted({item.obligation_id for item in transforms if item.obligation_id})
    )
    owners = tuple(sorted({item.owner_id for item in transforms}))
    return ExplicitStateObjectPlan(
        transform_cids=tuple(item.transform_cid for item in transforms),
        write_paths=resolved.effect_scope.write_paths,
        preimage_cid=resolved.preimage.preimage_cid,
        packet_cid=resolved.packet_cid,
        tree_id=resolved.tree_id,
        obligation_ids=obligations,
        owner_ids=owners,
        complete_obligations=True,
        unique_owners=True,
        no_duplicated_mutable_state=True,
        missing_release_guessed=False,
    )


def compile_explicit_state_object_receipt(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ExplicitStateObjectReceipt:
    """Compile a SPAR-022 receipt over nominated explicit state transforms."""

    resolved = _coerce_packet(packet)
    preimage_cid = verify_preimages(
        resolved, claimed_preimage_cid=kwargs.get("claimed_preimage_cid") or ""
    )
    plan = compile_explicit_state_object_plan(resolved, **kwargs)
    return ExplicitStateObjectReceipt(
        tree_id=resolved.tree_id,
        packet_cid=resolved.packet_cid,
        preimage_cid=preimage_cid,
        transform_cids=plan.transform_cids,
        plan_cid=plan.plan_cid,
        write_paths=resolved.effect_scope.write_paths,
        owner_ids=plan.owner_ids,
    )


def execute_explicit_state_objects(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ExplicitStateObjectReceipt:
    """Execute SPAR-022 as a deterministic no-mutation dry-run."""

    if kwargs.pop("mutate", False):
        raise StateTransformError("executor cannot mutate")
    return compile_explicit_state_object_receipt(packet, **kwargs)


def execute_explicit_state_object_plan(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ExplicitStateObjectPlan:
    """Execute the unique SPAR-022 plan without mutation."""

    if kwargs.pop("mutate", False):
        raise StateTransformError("executor cannot mutate")
    return compile_explicit_state_object_plan(packet, **kwargs)


def dry_run_explicit_state_objects(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ExplicitStateObjectReceipt:
    """Return a deterministic no-mutation dry-run of SPAR-022 transforms."""

    receipt = execute_explicit_state_objects(packet, **kwargs)
    if receipt.mutated:
        raise StateTransformError("dry-run cannot mutate")
    return receipt


def encode_canonical_transform(transform: ExplicitStateObject) -> dict[str, Any]:
    return transform.to_dict()


def decode_canonical_transform(payload: Mapping[str, Any]) -> ExplicitStateObject:
    return ExplicitStateObject.from_dict(payload)


def encode_canonical_plan(plan: ExplicitStateObjectPlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_plan(payload: Mapping[str, Any]) -> ExplicitStateObjectPlan:
    return ExplicitStateObjectPlan.from_dict(payload)


def encode_canonical_receipt(receipt: ExplicitStateObjectReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(
    payload: Mapping[str, Any],
) -> ExplicitStateObjectReceipt:
    return ExplicitStateObjectReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise StateTransformError(
            f"state transform must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "DECLARED_EXPLICIT_STATE_OBJECT_KINDS",
    "DECLARED_LIFECYCLE_KINDS",
    "DECLARED_SYNCHRONIZATION_KINDS",
    "DECLARED_UNIQUENESS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "DUPLICATED_MUTABLE_STATE_REJECTED",
    "EXECUTOR_IS_NOMINATION_ONLY",
    "EXPLICIT_STATE_OBJECT_INTERFACE",
    "EXPLICIT_STATE_OBJECT_PLAN_INTERFACE",
    "EXPLICIT_STATE_OBJECT_RECEIPT_INTERFACE",
    "GOAL_ID",
    "HANDLED_ADAPTER_KINDS",
    "IDENTITY_EXCLUDED_FIELDS",
    "INCOMPLETE_OBLIGATIONS_ARE_TYPED_TERMINAL",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MISSING_RELEASE_IS_NOT_GUESSED",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PLAN_IS_NOMINATION_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TRANSFORM_CAN_AUTHORIZE_COMPLETION",
    "TRANSFORM_CAN_AUTHORIZE_TRANSITION",
    "TRANSFORM_CAN_CREATE_AUTHORITY",
    "TRANSFORM_CONTRACT_VERSION",
    "UNKNOWN_UNIQUENESS_WIDENS_FRONTIER",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "ExplicitStateObject",
    "ExplicitStateObjectKind",
    "ExplicitStateObjectPlan",
    "ExplicitStateObjectReceipt",
    "StateTransformError",
    "assert_not_competing_capsule_family",
    "compile_explicit_state_object_plan",
    "compile_explicit_state_object_receipt",
    "compile_explicit_state_objects",
    "decode_canonical_plan",
    "decode_canonical_receipt",
    "decode_canonical_transform",
    "dry_run_explicit_state_objects",
    "encode_canonical_plan",
    "encode_canonical_receipt",
    "encode_canonical_transform",
    "execute_explicit_state_object_plan",
    "execute_explicit_state_objects",
    "provider_free_exports",
    "state_transform_cid_profile",
    "verify_complete_obligations",
    "verify_preimages",
]


assert TASK_ID == "SPAR-022"
assert EXPLICIT_STATE_OBJECT_PLAN_INTERFACE == "ExplicitStateObjectPlan@1"
assert EXPLICIT_STATE_OBJECT_INTERFACE == "ExplicitStateObject@1"
assert EXECUTOR_IS_NOMINATION_ONLY is True
assert PLAN_IS_NOMINATION_ONLY is True
assert TRANSFORM_CAN_AUTHORIZE_COMPLETION is False
assert TRANSFORM_CAN_AUTHORIZE_TRANSITION is False
assert MISSING_RELEASE_IS_NOT_GUESSED is True
assert DUPLICATED_MUTABLE_STATE_REJECTED is True
