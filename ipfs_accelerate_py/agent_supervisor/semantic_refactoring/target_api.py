"""SPAR-017 target module APIs and cycle-free dependency direction.

This module extends current supervisor partition orchestration with
``TargetModuleAPIPlan@1``.  It consumes SPAR-014 ranked partition candidates
and nominates public/private exports, protocols, adapters, state-owner
interfaces, responsibility statements, and a cycle-free dependency DAG.

The plan is nomination-only.  It cannot authorize a transition, completion,
or competing authority.  Vector, model, and heuristic evidence cannot admit a
module.  Observational metadata is excluded from identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
    PartitionGenerationReceipt,
    ProgramPartitionCandidate,
)
from .partition_policy import (
    ANALYZER_ID as SPAR014_ANALYZER_ID,
    PartitionComparisonReceipt,
    compare_partition_candidates,
)


TASK_ID: Final[str] = "SPAR-017"
GOAL_ID: Final[str] = "SPAR-G033"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.target_api@1"
)

TARGET_MODULE_API_PLAN_INTERFACE: Final[str] = "TargetModuleAPIPlan@1"
TARGET_MODULE_API_INTERFACE: Final[str] = "TargetModuleAPI@1"
TARGET_EXPORT_INTERFACE: Final[str] = "TargetExport@1"
TARGET_PROTOCOL_INTERFACE: Final[str] = "TargetProtocol@1"
TARGET_ADAPTER_INTERFACE: Final[str] = "TargetAdapter@1"
STATE_OWNER_INTERFACE_INTERFACE: Final[str] = "StateOwnerInterface@1"
RESPONSIBILITY_STATEMENT_INTERFACE: Final[str] = "ResponsibilityStatement@1"
MODULE_DEPENDENCY_EDGE_INTERFACE: Final[str] = "ModuleDependencyEdge@1"
TARGET_API_SYNTHESIS_RECEIPT_INTERFACE: Final[str] = (
    "TargetAPISynthesisReceipt@1"
)

TARGET_MODULE_API_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/target-module-api-plan@1"
)
TARGET_MODULE_API_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/target-module-api@1"
)
TARGET_EXPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/target-export@1"
)
TARGET_PROTOCOL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/target-protocol@1"
)
TARGET_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/target-adapter@1"
)
STATE_OWNER_INTERFACE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-owner-interface@1"
)
RESPONSIBILITY_STATEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/responsibility-statement@1"
)
MODULE_DEPENDENCY_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/module-dependency-edge@1"
)
TARGET_API_SYNTHESIS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/target-api-synthesis-receipt@1"
)

TARGET_API_CONTRACT_VERSION: Final[str] = "1"

TARGET_API_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
TARGET_API_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
TARGET_API_CAN_CREATE_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
PLAN_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_MODULES: Final[int] = 16_384
MAX_EDGES: Final[int] = 65_536
MAX_VIOLATIONS: Final[int] = 4_096
MAX_EVIDENCE_CIDS: Final[int] = 1_024

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

EXISTING_ADAPTER_AUTHORITIES: Final[tuple[str, ...]] = (
    "datasets_semantic",
    "kit_storage",
    "kit_vfs",
    "accelerator_supervisor",
    "accelerator_runtime",
    "spar_narrow",
)

PUBLIC_SURFACE_PROTOCOL: Final[str] = "public_surface"
ADAPTER_KIND: Final[str] = "adapter"
STATE_OWNER_KIND: Final[str] = "state_owner_interface"
SYMBOL_KIND: Final[str] = "symbol"

_ADAPTER_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("datasets:", "datasets_semantic"),
    ("semantic:", "datasets_semantic"),
    ("kit:", "kit_storage"),
    ("vfs:", "kit_vfs"),
    ("supervisor:", "accelerator_supervisor"),
    ("runtime:", "accelerator_runtime"),
    ("spar:", "spar_narrow"),
)


class TargetAPIError(ValueError):
    """Fail-closed violation of a SPAR-017 target-API contract."""


class ExportVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"


class AdapterAuthority(str, Enum):
    DATASETS_SEMANTIC = "datasets_semantic"
    KIT_STORAGE = "kit_storage"
    KIT_VFS = "kit_vfs"
    ACCELERATOR_SUPERVISOR = "accelerator_supervisor"
    ACCELERATOR_RUNTIME = "accelerator_runtime"
    SPAR_NARROW = "spar_narrow"


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise TargetAPIError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise TargetAPIError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise TargetAPIError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise TargetAPIError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise TargetAPIError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise TargetAPIError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TargetAPIError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise TargetAPIError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise TargetAPIError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise TargetAPIError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise TargetAPIError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise TargetAPIError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise TargetAPIError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise TargetAPIError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise TargetAPIError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TargetAPIError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise TargetAPIError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise TargetAPIError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise TargetAPIError(f"unknown {name}: {text}") from exc


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
    raise TargetAPIError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise TargetAPIError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def target_api_cid_profile() -> dict[str, str]:
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
            raise TargetAPIError(f"{name} cannot claim {flag}")


def _adapter_authority_for(target_id: str) -> str:
    for prefix, authority in _ADAPTER_PREFIXES:
        if target_id.startswith(prefix):
            return authority
    return AdapterAuthority.SPAR_NARROW.value


def _responsibility_text(
    module_id: str,
    member_ids: Sequence[str],
    state_owner_ids: Sequence[str],
) -> str:
    owners = ",".join(state_owner_ids) if state_owner_ids else "none"
    members = ",".join(member_ids)
    return f"module {module_id} owns members {members}; state owners {owners}"


def _has_cycle(module_ids: Sequence[str], edges: Sequence[tuple[str, str]]) -> bool:
    adjacency: dict[str, list[str]] = {module_id: [] for module_id in module_ids}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].append(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in adjacency[node]:
            if visit(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in module_ids)


@dataclass(frozen=True, slots=True)
class TargetExport:
    """One public or private member export. Nomination only."""

    member_id: str
    visibility: ExportVisibility | str
    kind: str = SYMBOL_KIND
    consumer_ids: Sequence[str] = ()

    interface: ClassVar[str] = TARGET_EXPORT_INTERFACE
    schema: ClassVar[str] = TARGET_EXPORT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "member_id",
            "visibility",
            "kind",
            "consumer_ids",
            "export_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_id", _text(self.member_id, "member_id"))
        object.__setattr__(
            self,
            "visibility",
            _enum(self.visibility, ExportVisibility, "visibility"),
        )
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(
            self,
            "consumer_ids",
            _unique_sorted_text(
                list(self.consumer_ids), "consumer_ids", limit=MAX_MEMBERS
            ),
        )
        if self.visibility == ExportVisibility.PRIVATE.value and self.consumer_ids:
            raise TargetAPIError("private export cannot list consumers")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TARGET_EXPORT_SCHEMA,
            "interface": TARGET_EXPORT_INTERFACE,
            "member_id": self.member_id,
            "visibility": self.visibility,
            "kind": self.kind,
            "consumer_ids": list(self.consumer_ids),
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def export_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["export_cid"] = self.export_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetExport":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("export_cid")
        if payload.pop("schema") != TARGET_EXPORT_SCHEMA:
            raise TargetAPIError("unsupported TargetExport schema")
        if payload.pop("interface") != TARGET_EXPORT_INTERFACE:
            raise TargetAPIError("unsupported TargetExport interface")
        result = cls(**payload)
        _verify_cid(claimed, result.export_cid, "TargetExport export_cid")
        return result


def _coerce_export(value: TargetExport | Mapping[str, Any]) -> TargetExport:
    if isinstance(value, TargetExport):
        return value
    if isinstance(value, Mapping):
        if "export_cid" in value:
            return TargetExport.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "export_cid"}
        }
        return TargetExport(**payload)
    raise TargetAPIError("export must be a TargetExport")


@dataclass(frozen=True, slots=True)
class TargetProtocol:
    """Nominated protocol covering a module's public surface."""

    module_id: str
    member_ids: Sequence[str]
    kind: str = PUBLIC_SURFACE_PROTOCOL

    interface: ClassVar[str] = TARGET_PROTOCOL_INTERFACE
    schema: ClassVar[str] = TARGET_PROTOCOL_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "module_id",
            "member_ids",
            "kind",
            "protocol_cid",
        }
    )

    def __post_init__(self) -> None:
        members = _unique_sorted_text(
            list(self.member_ids), "member_ids", limit=MAX_MEMBERS
        )
        if not members:
            raise TargetAPIError("protocol requires members")
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "kind", _text(self.kind, "kind"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TARGET_PROTOCOL_SCHEMA,
            "interface": TARGET_PROTOCOL_INTERFACE,
            "module_id": self.module_id,
            "member_ids": list(self.member_ids),
            "kind": self.kind,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def protocol_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["protocol_cid"] = self.protocol_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetProtocol":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("protocol_cid")
        if payload.pop("schema") != TARGET_PROTOCOL_SCHEMA:
            raise TargetAPIError("unsupported TargetProtocol schema")
        if payload.pop("interface") != TARGET_PROTOCOL_INTERFACE:
            raise TargetAPIError("unsupported TargetProtocol interface")
        result = cls(**payload)
        _verify_cid(claimed, result.protocol_cid, "TargetProtocol protocol_cid")
        return result


def _coerce_protocol(value: TargetProtocol | Mapping[str, Any]) -> TargetProtocol:
    if isinstance(value, TargetProtocol):
        return value
    if isinstance(value, Mapping):
        if "protocol_cid" in value:
            return TargetProtocol.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "protocol_cid"}
        }
        return TargetProtocol(**payload)
    raise TargetAPIError("protocol must be a TargetProtocol")


@dataclass(frozen=True, slots=True)
class TargetAdapter:
    """Narrow adapter onto an existing authority. Never competing."""

    module_id: str
    authority: AdapterAuthority | str
    external_ids: Sequence[str]
    kind: str = ADAPTER_KIND

    interface: ClassVar[str] = TARGET_ADAPTER_INTERFACE
    schema: ClassVar[str] = TARGET_ADAPTER_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "module_id",
            "authority",
            "external_ids",
            "kind",
            "can_create_authority",
            "adapter_cid",
        }
    )

    def __post_init__(self) -> None:
        authority = _enum(self.authority, AdapterAuthority, "authority")
        externals = _unique_sorted_text(
            list(self.external_ids), "external_ids", limit=MAX_MEMBERS
        )
        if not externals:
            raise TargetAPIError("adapter requires external_ids")
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "external_ids", externals)
        object.__setattr__(self, "kind", _text(self.kind, "kind"))

    @property
    def can_create_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TARGET_ADAPTER_SCHEMA,
            "interface": TARGET_ADAPTER_INTERFACE,
            "module_id": self.module_id,
            "authority": self.authority,
            "external_ids": list(self.external_ids),
            "kind": self.kind,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def adapter_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["adapter_cid"] = self.adapter_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetAdapter":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("adapter_cid")
        if payload.pop("schema") != TARGET_ADAPTER_SCHEMA:
            raise TargetAPIError("unsupported TargetAdapter schema")
        if payload.pop("interface") != TARGET_ADAPTER_INTERFACE:
            raise TargetAPIError("unsupported TargetAdapter interface")
        if payload.pop("can_create_authority") is not False:
            raise TargetAPIError("adapter cannot claim can_create_authority")
        result = cls(**payload)
        _verify_cid(claimed, result.adapter_cid, "TargetAdapter adapter_cid")
        return result


def _coerce_adapter(value: TargetAdapter | Mapping[str, Any]) -> TargetAdapter:
    if isinstance(value, TargetAdapter):
        return value
    if isinstance(value, Mapping):
        if "adapter_cid" in value:
            return TargetAdapter.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "adapter_cid",
                "can_create_authority",
            }
        }
        return TargetAdapter(**payload)
    raise TargetAPIError("adapter must be a TargetAdapter")


@dataclass(frozen=True, slots=True)
class StateOwnerInterface:
    """Unique state-owner interface bound to exactly one module."""

    owner_id: str
    module_id: str
    member_ids: Sequence[str]
    kind: str = STATE_OWNER_KIND

    interface: ClassVar[str] = STATE_OWNER_INTERFACE_INTERFACE
    schema: ClassVar[str] = STATE_OWNER_INTERFACE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "owner_id",
            "module_id",
            "member_ids",
            "kind",
            "interface_cid",
        }
    )

    def __post_init__(self) -> None:
        members = _unique_sorted_text(
            list(self.member_ids), "member_ids", limit=MAX_MEMBERS
        )
        if not members:
            raise TargetAPIError("state-owner interface requires members")
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "module_id", _text(self.module_id, "module_id"))
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "kind", _text(self.kind, "kind"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": STATE_OWNER_INTERFACE_SCHEMA,
            "interface": STATE_OWNER_INTERFACE_INTERFACE,
            "owner_id": self.owner_id,
            "module_id": self.module_id,
            "member_ids": list(self.member_ids),
            "kind": self.kind,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def interface_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["interface_cid"] = self.interface_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateOwnerInterface":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("interface_cid")
        if payload.pop("schema") != STATE_OWNER_INTERFACE_SCHEMA:
            raise TargetAPIError("unsupported StateOwnerInterface schema")
        if payload.pop("interface") != STATE_OWNER_INTERFACE_INTERFACE:
            raise TargetAPIError("unsupported StateOwnerInterface interface")
        result = cls(**payload)
        _verify_cid(
            claimed, result.interface_cid, "StateOwnerInterface interface_cid"
        )
        return result


def _coerce_state_owner_interface(
    value: StateOwnerInterface | Mapping[str, Any],
) -> StateOwnerInterface:
    if isinstance(value, StateOwnerInterface):
        return value
    if isinstance(value, Mapping):
        if "interface_cid" in value:
            return StateOwnerInterface.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "interface_cid"}
        }
        return StateOwnerInterface(**payload)
    raise TargetAPIError("state-owner interface must be a StateOwnerInterface")


@dataclass(frozen=True, slots=True)
class ResponsibilityStatement:
    """Deterministic module responsibility. Not completion prose."""

    module_id: str
    owned_member_ids: Sequence[str]
    owned_state_owner_ids: Sequence[str] = ()
    statement: str = ""

    interface: ClassVar[str] = RESPONSIBILITY_STATEMENT_INTERFACE
    schema: ClassVar[str] = RESPONSIBILITY_STATEMENT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "module_id",
            "owned_member_ids",
            "owned_state_owner_ids",
            "statement",
            "statement_cid",
        }
    )

    def __post_init__(self) -> None:
        module_id = _text(self.module_id, "module_id")
        members = _unique_sorted_text(
            list(self.owned_member_ids), "owned_member_ids", limit=MAX_MEMBERS
        )
        if not members:
            raise TargetAPIError("responsibility requires owned members")
        owners = _unique_sorted_text(
            list(self.owned_state_owner_ids),
            "owned_state_owner_ids",
            limit=MAX_MEMBERS,
        )
        expected = _responsibility_text(module_id, members, owners)
        statement = self.statement
        if statement in (None, ""):
            statement = expected
        else:
            statement = _text(statement, "statement")
        if statement != expected:
            raise TargetAPIError("responsibility statement must remain canonical")
        object.__setattr__(self, "module_id", module_id)
        object.__setattr__(self, "owned_member_ids", members)
        object.__setattr__(self, "owned_state_owner_ids", owners)
        object.__setattr__(self, "statement", statement)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": RESPONSIBILITY_STATEMENT_SCHEMA,
            "interface": RESPONSIBILITY_STATEMENT_INTERFACE,
            "module_id": self.module_id,
            "owned_member_ids": list(self.owned_member_ids),
            "owned_state_owner_ids": list(self.owned_state_owner_ids),
            "statement": self.statement,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def statement_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["statement_cid"] = self.statement_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResponsibilityStatement":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("statement_cid")
        if payload.pop("schema") != RESPONSIBILITY_STATEMENT_SCHEMA:
            raise TargetAPIError("unsupported ResponsibilityStatement schema")
        if payload.pop("interface") != RESPONSIBILITY_STATEMENT_INTERFACE:
            raise TargetAPIError("unsupported ResponsibilityStatement interface")
        result = cls(**payload)
        _verify_cid(
            claimed,
            result.statement_cid,
            "ResponsibilityStatement statement_cid",
        )
        return result


def _coerce_responsibility(
    value: ResponsibilityStatement | Mapping[str, Any],
) -> ResponsibilityStatement:
    if isinstance(value, ResponsibilityStatement):
        return value
    if isinstance(value, Mapping):
        if "statement_cid" in value:
            return ResponsibilityStatement.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "statement_cid"}
        }
        return ResponsibilityStatement(**payload)
    raise TargetAPIError("responsibility must be a ResponsibilityStatement")


@dataclass(frozen=True, slots=True)
class ModuleDependencyEdge:
    """Directed module dependency. Cycle-free plans only."""

    source_module_id: str
    target_module_id: str
    kind: str
    constraint_class: str = "soft"
    witness_member_ids: Sequence[str] = ()

    interface: ClassVar[str] = MODULE_DEPENDENCY_EDGE_INTERFACE
    schema: ClassVar[str] = MODULE_DEPENDENCY_EDGE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "source_module_id",
            "target_module_id",
            "kind",
            "constraint_class",
            "witness_member_ids",
            "edge_cid",
        }
    )

    def __post_init__(self) -> None:
        source = _text(self.source_module_id, "source_module_id")
        target = _text(self.target_module_id, "target_module_id")
        if source == target:
            raise TargetAPIError("dependency edge cannot be reflexive")
        object.__setattr__(self, "source_module_id", source)
        object.__setattr__(self, "target_module_id", target)
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(
            self,
            "constraint_class",
            _text(self.constraint_class, "constraint_class"),
        )
        object.__setattr__(
            self,
            "witness_member_ids",
            _unique_sorted_text(
                list(self.witness_member_ids),
                "witness_member_ids",
                limit=MAX_MEMBERS,
            ),
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MODULE_DEPENDENCY_EDGE_SCHEMA,
            "interface": MODULE_DEPENDENCY_EDGE_INTERFACE,
            "source_module_id": self.source_module_id,
            "target_module_id": self.target_module_id,
            "kind": self.kind,
            "constraint_class": self.constraint_class,
            "witness_member_ids": list(self.witness_member_ids),
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def edge_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["edge_cid"] = self.edge_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModuleDependencyEdge":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("edge_cid")
        if payload.pop("schema") != MODULE_DEPENDENCY_EDGE_SCHEMA:
            raise TargetAPIError("unsupported ModuleDependencyEdge schema")
        if payload.pop("interface") != MODULE_DEPENDENCY_EDGE_INTERFACE:
            raise TargetAPIError("unsupported ModuleDependencyEdge interface")
        result = cls(**payload)
        _verify_cid(claimed, result.edge_cid, "ModuleDependencyEdge edge_cid")
        return result


def _coerce_dependency_edge(
    value: ModuleDependencyEdge | Mapping[str, Any],
) -> ModuleDependencyEdge:
    if isinstance(value, ModuleDependencyEdge):
        return value
    if isinstance(value, Mapping):
        if "edge_cid" in value:
            return ModuleDependencyEdge.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "edge_cid"}
        }
        return ModuleDependencyEdge(**payload)
    raise TargetAPIError("dependency edge must be a ModuleDependencyEdge")


def _sorted_exports(
    values: Sequence[TargetExport | Mapping[str, Any]],
    name: str,
) -> tuple[TargetExport, ...]:
    items = tuple(_coerce_export(item) for item in values)
    if len(items) > MAX_MEMBERS:
        raise TargetAPIError(f"{name} exceed maximum length")
    return tuple(sorted(items, key=lambda item: (item.visibility, item.member_id)))


@dataclass(frozen=True, slots=True)
class TargetModuleAPI:
    """One nominated module API derived from a SPAR-014 ranked candidate."""

    tree_id: str
    candidate_cid: str
    member_ids: Sequence[str]
    public_exports: Sequence[TargetExport | Mapping[str, Any]] = ()
    private_exports: Sequence[TargetExport | Mapping[str, Any]] = ()
    protocols: Sequence[TargetProtocol | Mapping[str, Any]] = ()
    adapters: Sequence[TargetAdapter | Mapping[str, Any]] = ()
    state_owner_interfaces: Sequence[StateOwnerInterface | Mapping[str, Any]] = ()
    responsibility: ResponsibilityStatement | Mapping[str, Any] | None = None
    depends_on_module_ids: Sequence[str] = ()
    external_dependency_ids: Sequence[str] = ()
    state_owner_ids: Sequence[str] = ()
    module_id: str = ""

    interface: ClassVar[str] = TARGET_MODULE_API_INTERFACE
    schema: ClassVar[str] = TARGET_MODULE_API_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "candidate_cid",
            "member_ids",
            "public_exports",
            "private_exports",
            "protocols",
            "adapters",
            "state_owner_interfaces",
            "responsibility",
            "depends_on_module_ids",
            "external_dependency_ids",
            "state_owner_ids",
            "module_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "api_cid",
        }
    )

    def __post_init__(self) -> None:
        tree_id = _tree_id(self.tree_id)
        candidate_cid = _cid(self.candidate_cid, "candidate_cid")
        members = _unique_sorted_text(
            list(self.member_ids), "member_ids", limit=MAX_MEMBERS
        )
        if not members:
            raise TargetAPIError("target module API requires members")
        module_id = self.module_id
        if module_id in (None, ""):
            module_id = candidate_cid
        else:
            module_id = _text(module_id, "module_id")
        if module_id != candidate_cid:
            raise TargetAPIError("module_id must equal candidate_cid")
        public = _sorted_exports(self.public_exports, "public_exports")
        private = _sorted_exports(self.private_exports, "private_exports")
        if any(item.visibility != ExportVisibility.PUBLIC.value for item in public):
            raise TargetAPIError("public_exports must have public visibility")
        if any(item.visibility != ExportVisibility.PRIVATE.value for item in private):
            raise TargetAPIError("private_exports must have private visibility")
        exported = tuple(item.member_id for item in (*public, *private))
        if len(exported) != len(set(exported)):
            raise TargetAPIError("public and private exports must be disjoint")
        if tuple(sorted(exported)) != members:
            raise TargetAPIError("exports must cover every member exactly once")
        protocols = tuple(
            sorted(
                (_coerce_protocol(item) for item in self.protocols),
                key=lambda item: item.protocol_cid,
            )
        )
        for protocol in protocols:
            if protocol.module_id != module_id:
                raise TargetAPIError("protocol module_id must match module")
            unknown = [item for item in protocol.member_ids if item not in members]
            if unknown:
                raise TargetAPIError("protocol members must belong to the module")
        adapters = tuple(
            sorted(
                (_coerce_adapter(item) for item in self.adapters),
                key=lambda item: item.adapter_cid,
            )
        )
        for adapter in adapters:
            if adapter.module_id != module_id:
                raise TargetAPIError("adapter module_id must match module")
        owners = _unique_sorted_text(
            list(self.state_owner_ids), "state_owner_ids", limit=MAX_MEMBERS
        )
        owner_ifaces = tuple(
            sorted(
                (
                    _coerce_state_owner_interface(item)
                    for item in self.state_owner_interfaces
                ),
                key=lambda item: item.interface_cid,
            )
        )
        iface_owners = tuple(item.owner_id for item in owner_ifaces)
        if tuple(sorted(iface_owners)) != owners:
            raise TargetAPIError(
                "state-owner interfaces must cover every owner exactly once"
            )
        for iface in owner_ifaces:
            if iface.module_id != module_id:
                raise TargetAPIError("state-owner interface module_id must match")
            unknown = [item for item in iface.member_ids if item not in members]
            if unknown:
                raise TargetAPIError(
                    "state-owner interface members must belong to the module"
                )
        responsibility = self.responsibility
        if responsibility is None:
            responsibility = ResponsibilityStatement(
                module_id=module_id,
                owned_member_ids=members,
                owned_state_owner_ids=owners,
            )
        else:
            responsibility = _coerce_responsibility(responsibility)
        if responsibility.module_id != module_id:
            raise TargetAPIError("responsibility module_id must match module")
        if responsibility.owned_member_ids != members:
            raise TargetAPIError("responsibility members must match module members")
        if responsibility.owned_state_owner_ids != owners:
            raise TargetAPIError("responsibility owners must match state owners")
        depends = _unique_sorted_text(
            list(self.depends_on_module_ids),
            "depends_on_module_ids",
            limit=MAX_MODULES,
        )
        if module_id in depends:
            raise TargetAPIError("module cannot depend on itself")
        externals = _unique_sorted_text(
            list(self.external_dependency_ids),
            "external_dependency_ids",
            limit=MAX_MEMBERS,
        )
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "candidate_cid", candidate_cid)
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "public_exports", public)
        object.__setattr__(self, "private_exports", private)
        object.__setattr__(self, "protocols", protocols)
        object.__setattr__(self, "adapters", adapters)
        object.__setattr__(self, "state_owner_interfaces", owner_ifaces)
        object.__setattr__(self, "responsibility", responsibility)
        object.__setattr__(self, "depends_on_module_ids", depends)
        object.__setattr__(self, "external_dependency_ids", externals)
        object.__setattr__(self, "state_owner_ids", owners)
        object.__setattr__(self, "module_id", module_id)

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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TARGET_MODULE_API_SCHEMA,
            "interface": TARGET_MODULE_API_INTERFACE,
            "tree_id": self.tree_id,
            "candidate_cid": self.candidate_cid,
            "member_ids": list(self.member_ids),
            "public_exports": [item.to_dict() for item in self.public_exports],
            "private_exports": [item.to_dict() for item in self.private_exports],
            "protocols": [item.to_dict() for item in self.protocols],
            "adapters": [item.to_dict() for item in self.adapters],
            "state_owner_interfaces": [
                item.to_dict() for item in self.state_owner_interfaces
            ],
            "responsibility": self.responsibility.to_dict(),
            "depends_on_module_ids": list(self.depends_on_module_ids),
            "external_dependency_ids": list(self.external_dependency_ids),
            "state_owner_ids": list(self.state_owner_ids),
            "module_id": self.module_id,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def api_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["api_cid"] = self.api_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetModuleAPI":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("api_cid")
        if payload.pop("schema") != TARGET_MODULE_API_SCHEMA:
            raise TargetAPIError("unsupported TargetModuleAPI schema")
        if payload.pop("interface") != TARGET_MODULE_API_INTERFACE:
            raise TargetAPIError("unsupported TargetModuleAPI interface")
        _pop_authority_flags(payload, "TargetModuleAPI")
        result = cls(**payload)
        _verify_cid(claimed, result.api_cid, "TargetModuleAPI api_cid")
        return result


def _coerce_module_api(value: TargetModuleAPI | Mapping[str, Any]) -> TargetModuleAPI:
    if isinstance(value, TargetModuleAPI):
        return value
    if isinstance(value, Mapping):
        if "api_cid" in value:
            return TargetModuleAPI.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "api_cid",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return TargetModuleAPI(**payload)
    raise TargetAPIError("module API must be a TargetModuleAPI")


@dataclass(frozen=True, slots=True)
class TargetModuleAPIPlan:
    """Nominated target APIs and cycle-free dependency direction."""

    tree_id: str
    modules: Sequence[TargetModuleAPI | Mapping[str, Any]]
    dependency_edges: Sequence[ModuleDependencyEdge | Mapping[str, Any]] = ()
    selected_candidate_cids: Sequence[str] = ()
    rejected_candidate_cids: Sequence[str] = ()
    advisory_candidate_cids: Sequence[str] = ()
    evidence_cids: Sequence[str] = ()
    comparison_receipt_cid: str = ""
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = TARGET_MODULE_API_PLAN_INTERFACE
    schema: ClassVar[str] = TARGET_MODULE_API_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "modules",
            "dependency_edges",
            "selected_candidate_cids",
            "rejected_candidate_cids",
            "advisory_candidate_cids",
            "negative_evidence_cids",
            "evidence_cids",
            "comparison_receipt_cid",
            "analyzer_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "plan_is_nomination_only",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise TargetAPIError("analyzer_id must remain the SPAR-017 analyzer")
        tree_id = _tree_id(self.tree_id)
        modules = tuple(_coerce_module_api(item) for item in self.modules)
        if not modules:
            raise TargetAPIError("target API plan requires modules")
        if len(modules) > MAX_MODULES:
            raise TargetAPIError("modules exceed maximum length")
        mismatched = [item.module_id for item in modules if item.tree_id != tree_id]
        if mismatched:
            raise TargetAPIError("module tree_id does not match plan")
        module_ids = [item.module_id for item in modules]
        if len(module_ids) != len(set(module_ids)):
            raise TargetAPIError("duplicate target module identity")
        seen_members: dict[str, str] = {}
        seen_owners: dict[str, str] = {}
        for module in modules:
            for member in module.member_ids:
                previous = seen_members.get(member)
                if previous is not None:
                    raise TargetAPIError("overlapping modules cannot form a simultaneous API plan")
                seen_members[member] = module.module_id
            for owner in module.state_owner_ids:
                previous = seen_owners.get(owner)
                if previous is not None:
                    raise TargetAPIError("state owner split across modules")
                seen_owners[owner] = module.module_id
        modules = tuple(sorted(modules, key=lambda item: item.module_id))
        edges = tuple(
            sorted(
                (_coerce_dependency_edge(item) for item in self.dependency_edges),
                key=lambda item: item.edge_cid,
            )
        )
        if len(edges) > MAX_EDGES:
            raise TargetAPIError("dependency_edges exceed maximum length")
        module_id_set = {item.module_id for item in modules}
        depends: dict[str, set[str]] = {item.module_id: set() for item in modules}
        pair_kinds: list[tuple[str, str]] = []
        for edge in edges:
            if edge.source_module_id not in module_id_set:
                raise TargetAPIError("dependency source is not a selected module")
            if edge.target_module_id not in module_id_set:
                raise TargetAPIError("dependency target is not a selected module")
            depends[edge.source_module_id].add(edge.target_module_id)
            pair_kinds.append((edge.source_module_id, edge.target_module_id))
        for module in modules:
            expected = tuple(sorted(depends[module.module_id]))
            if module.depends_on_module_ids != expected:
                raise TargetAPIError("module depends_on_module_ids must match dependency edges")
        if _has_cycle(tuple(sorted(module_id_set)), pair_kinds):
            raise TargetAPIError("dependency direction must remain cycle-free")
        selected = _unique_sorted_text(
            list(self.selected_candidate_cids or module_ids),
            "selected_candidate_cids",
            limit=MAX_MODULES,
        )
        if tuple(sorted(module_ids)) != selected:
            raise TargetAPIError("selected_candidate_cids must match module identities")
        rejected = _unique_sorted_text(
            list(self.rejected_candidate_cids),
            "rejected_candidate_cids",
            limit=MAX_MODULES,
        )
        advisory = _unique_sorted_text(
            list(self.advisory_candidate_cids),
            "advisory_candidate_cids",
            limit=MAX_MODULES,
        )
        overlap = set(selected) & set(rejected)
        if overlap:
            raise TargetAPIError("rejected candidates cannot become modules")
        overlap = set(selected) & set(advisory)
        if overlap:
            raise TargetAPIError("advisory candidates cannot become modules")
        evidence = tuple(
            sorted(_cid(item, "evidence_cids") for item in self.evidence_cids)
        )
        if len(evidence) != len(set(evidence)):
            raise TargetAPIError("evidence_cids must not contain duplicates")
        if len(evidence) > MAX_EVIDENCE_CIDS:
            raise TargetAPIError("evidence_cids exceed maximum length")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "modules", modules)
        object.__setattr__(self, "dependency_edges", edges)
        object.__setattr__(self, "selected_candidate_cids", selected)
        object.__setattr__(self, "rejected_candidate_cids", rejected)
        object.__setattr__(self, "advisory_candidate_cids", advisory)
        object.__setattr__(self, "evidence_cids", evidence)
        object.__setattr__(
            self,
            "comparison_receipt_cid",
            _optional_cid(self.comparison_receipt_cid, "comparison_receipt_cid"),
        )
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def negative_evidence_cids(self) -> tuple[str, ...]:
        return self.rejected_candidate_cids

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
            "schema": TARGET_MODULE_API_PLAN_SCHEMA,
            "interface": TARGET_MODULE_API_PLAN_INTERFACE,
            "tree_id": self.tree_id,
            "modules": [item.to_dict() for item in self.modules],
            "dependency_edges": [item.to_dict() for item in self.dependency_edges],
            "selected_candidate_cids": list(self.selected_candidate_cids),
            "rejected_candidate_cids": list(self.rejected_candidate_cids),
            "advisory_candidate_cids": list(self.advisory_candidate_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "evidence_cids": list(self.evidence_cids),
            "comparison_receipt_cid": self.comparison_receipt_cid,
            "analyzer_id": self.analyzer_id,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetModuleAPIPlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != TARGET_MODULE_API_PLAN_SCHEMA:
            raise TargetAPIError("unsupported TargetModuleAPIPlan schema")
        if payload.pop("interface") != TARGET_MODULE_API_PLAN_INTERFACE:
            raise TargetAPIError("unsupported TargetModuleAPIPlan interface")
        _pop_authority_flags(payload, "TargetModuleAPIPlan")
        if payload.pop("plan_is_nomination_only") is not True:
            raise TargetAPIError("plan must remain nomination_only")
        payload.pop("negative_evidence_cids")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "TargetModuleAPIPlan plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class TargetAPISynthesisReceipt:
    """Body-free synthesis receipt. Independent validation remains separate."""

    tree_id: str
    plan: TargetModuleAPIPlan | Mapping[str, Any]
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = TARGET_API_SYNTHESIS_RECEIPT_INTERFACE
    schema: ClassVar[str] = TARGET_API_SYNTHESIS_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "plan",
            "analyzer_id",
            "plan_cid",
            "selected_candidate_cids",
            "rejected_candidate_cids",
            "negative_evidence_cids",
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
            raise TargetAPIError("analyzer_id must remain the SPAR-017 analyzer")
        plan = (
            self.plan
            if isinstance(self.plan, TargetModuleAPIPlan)
            else TargetModuleAPIPlan.from_dict(_mapping(self.plan, "plan"))
        )
        tree_id = _tree_id(self.tree_id)
        if plan.tree_id != tree_id:
            raise TargetAPIError("receipt tree_id does not match plan")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def plan_cid(self) -> str:
        return self.plan.plan_cid

    @property
    def selected_candidate_cids(self) -> tuple[str, ...]:
        return self.plan.selected_candidate_cids

    @property
    def rejected_candidate_cids(self) -> tuple[str, ...]:
        return self.plan.rejected_candidate_cids

    @property
    def negative_evidence_cids(self) -> tuple[str, ...]:
        return self.plan.negative_evidence_cids

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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TARGET_API_SYNTHESIS_RECEIPT_SCHEMA,
            "interface": TARGET_API_SYNTHESIS_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "plan": self.plan.to_dict(),
            "analyzer_id": self.analyzer_id,
            "plan_cid": self.plan_cid,
            "selected_candidate_cids": list(self.selected_candidate_cids),
            "rejected_candidate_cids": list(self.rejected_candidate_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
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
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetAPISynthesisReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != TARGET_API_SYNTHESIS_RECEIPT_SCHEMA:
            raise TargetAPIError("unsupported TargetAPISynthesisReceipt schema")
        if payload.pop("interface") != TARGET_API_SYNTHESIS_RECEIPT_INTERFACE:
            raise TargetAPIError("unsupported TargetAPISynthesisReceipt interface")
        _pop_authority_flags(payload, "TargetAPISynthesisReceipt")
        payload.pop("plan_cid")
        payload.pop("selected_candidate_cids")
        payload.pop("rejected_candidate_cids")
        payload.pop("negative_evidence_cids")
        result = cls(**payload)
        _verify_cid(
            claimed,
            result.receipt_cid,
            "TargetAPISynthesisReceipt receipt_cid",
        )
        return result


def _coerce_candidate(
    value: ProgramPartitionCandidate | Mapping[str, Any],
) -> ProgramPartitionCandidate:
    if isinstance(value, ProgramPartitionCandidate):
        return value
    if isinstance(value, Mapping):
        if "candidate_cid" in value:
            return ProgramPartitionCandidate.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "candidate_cid",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return ProgramPartitionCandidate(**payload)
    raise TargetAPIError("candidate must be a ProgramPartitionCandidate")


def _coerce_candidates(
    value: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    | Mapping[str, Any],
) -> tuple[ProgramPartitionCandidate, ...]:
    if isinstance(value, PartitionGenerationReceipt):
        return value.candidates
    if isinstance(value, Mapping) and "candidates" in value:
        if "receipt_cid" in value:
            return PartitionGenerationReceipt.from_dict(value).candidates
        items = value.get("candidates") or ()
        return tuple(_coerce_candidate(item) for item in items)
    if isinstance(value, (list, tuple)):
        candidates = tuple(_coerce_candidate(item) for item in value)
        if not candidates:
            raise TargetAPIError("target API synthesis requires candidates")
        return candidates
    raise TargetAPIError("candidates must be a SPAR-013 receipt or candidate list")


def _coerce_comparison(
    value: PartitionComparisonReceipt | Mapping[str, Any] | None,
    candidates: Sequence[ProgramPartitionCandidate],
) -> PartitionComparisonReceipt:
    if value is None:
        return compare_partition_candidates(candidates)
    if isinstance(value, PartitionComparisonReceipt):
        comparison = value
    elif isinstance(value, Mapping):
        comparison = PartitionComparisonReceipt.from_dict(value)
    else:
        raise TargetAPIError("comparison must be a SPAR-014 receipt")
    if comparison.analyzer_id != SPAR014_ANALYZER_ID:
        raise TargetAPIError("comparison must remain the SPAR-014 analyzer")
    return comparison


def _pairwise_disjoint(candidates: Sequence[ProgramPartitionCandidate]) -> bool:
    seen: set[str] = set()
    for candidate in candidates:
        overlap = seen.intersection(candidate.member_ids)
        if overlap:
            return False
        seen.update(candidate.member_ids)
    return True


def _public_member_ids(candidate: ProgramPartitionCandidate) -> set[str]:
    inside = set(candidate.member_ids)
    public: set[str] = set()
    for edge in candidate.cut_edges:
        if edge.target_id in inside and edge.source_id not in inside:
            public.add(edge.target_id)
    if candidate.consumer_ids and not public:
        public = set(inside)
    return public


def _build_module_api(
    candidate: ProgramPartitionCandidate,
    *,
    member_to_module: Mapping[str, str],
    module_id: str,
) -> TargetModuleAPI:
    inside = set(candidate.member_ids)
    public_ids = _public_member_ids(candidate)
    unknown_public = public_ids - inside
    if unknown_public:
        raise TargetAPIError("public export must belong to the module")
    consumers = candidate.consumer_ids if public_ids else ()
    public_exports = tuple(
        TargetExport(
            member_id=member_id,
            visibility=ExportVisibility.PUBLIC,
            kind=SYMBOL_KIND,
            consumer_ids=consumers,
        )
        for member_id in sorted(public_ids)
    )
    private_exports = tuple(
        TargetExport(
            member_id=member_id,
            visibility=ExportVisibility.PRIVATE,
            kind=SYMBOL_KIND,
        )
        for member_id in candidate.member_ids
        if member_id not in public_ids
    )
    protocols = ()
    if public_ids:
        protocols = (
            TargetProtocol(
                module_id=module_id,
                member_ids=tuple(sorted(public_ids)),
                kind=PUBLIC_SURFACE_PROTOCOL,
            ),
        )
    depends: set[str] = set()
    externals: set[str] = set()
    adapter_targets: dict[str, set[str]] = {}
    for edge in candidate.cut_edges:
        if edge.source_id in inside and edge.target_id not in inside:
            other = member_to_module.get(edge.target_id)
            if other and other != module_id:
                depends.add(other)
            elif other is None:
                externals.add(edge.target_id)
                authority = _adapter_authority_for(edge.target_id)
                adapter_targets.setdefault(authority, set()).add(edge.target_id)
    adapters = tuple(
        TargetAdapter(
            module_id=module_id,
            authority=authority,
            external_ids=tuple(sorted(ids)),
        )
        for authority, ids in sorted(adapter_targets.items())
    )
    owner_ifaces = tuple(
        StateOwnerInterface(
            owner_id=owner_id,
            module_id=module_id,
            member_ids=candidate.member_ids,
        )
        for owner_id in candidate.state_owner_ids
    )
    return TargetModuleAPI(
        tree_id=candidate.tree_id,
        candidate_cid=candidate.candidate_cid,
        member_ids=candidate.member_ids,
        public_exports=public_exports,
        private_exports=private_exports,
        protocols=protocols,
        adapters=adapters,
        state_owner_interfaces=owner_ifaces,
        state_owner_ids=candidate.state_owner_ids,
        depends_on_module_ids=tuple(sorted(depends)),
        external_dependency_ids=tuple(sorted(externals)),
        module_id=module_id,
    )


def _collect_dependency_edges(
    selected: Sequence[ProgramPartitionCandidate],
    member_to_module: Mapping[str, str],
) -> tuple[ModuleDependencyEdge, ...]:
    buckets: dict[tuple[str, str, str, str], set[str]] = {}
    for candidate in selected:
        source_module = candidate.candidate_cid
        inside = set(candidate.member_ids)
        for edge in candidate.cut_edges:
            if edge.source_id not in inside:
                continue
            target_module = member_to_module.get(edge.target_id)
            if target_module is None or target_module == source_module:
                continue
            key = (
                source_module,
                target_module,
                edge.kind,
                edge.constraint_class,
            )
            buckets.setdefault(key, set()).update((edge.source_id, edge.target_id))
    return tuple(
        ModuleDependencyEdge(
            source_module_id=source,
            target_module_id=target,
            kind=kind,
            constraint_class=constraint,
            witness_member_ids=tuple(sorted(witnesses)),
        )
        for (source, target, kind, constraint), witnesses in sorted(buckets.items())
    )


def synthesize_target_module_apis(
    candidates: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    | Mapping[str, Any],
    *,
    comparison: PartitionComparisonReceipt | Mapping[str, Any] | None = None,
    selected_candidate_cids: Sequence[str] | None = None,
) -> TargetModuleAPIPlan:
    """Nominate module APIs and a cycle-free dependency DAG from SPAR-014.

    Only SPAR-014 ranked, non-advisory candidates may become modules. Overlapping
    ranked alternatives require an explicit disjoint ``selected_candidate_cids``.
    Rejected comparisons remain negative evidence. The result cannot authorize a
    transition or completion.
    """

    resolved_candidates = _coerce_candidates(candidates)
    tree_ids = {item.tree_id for item in resolved_candidates}
    if len(tree_ids) != 1:
        raise TargetAPIError("candidates must share one tree_id")
    tree_id = next(iter(tree_ids))
    resolved_comparison = _coerce_comparison(comparison, resolved_candidates)
    if resolved_comparison.tree_id != tree_id:
        raise TargetAPIError("comparison tree_id does not match candidates")
    by_cid = {item.candidate_cid: item for item in resolved_candidates}
    ranked_cids = set(resolved_comparison.ranked_candidate_cids)
    rejected_cids = tuple(resolved_comparison.rejected_candidate_cids)
    advisory_cids = tuple(resolved_comparison.advisory_candidate_cids)
    if selected_candidate_cids is None:
        ranked = tuple(
            by_cid[item]
            for item in resolved_comparison.ranked_candidate_cids
            if item in by_cid
        )
        if not ranked:
            raise TargetAPIError("no ranked SPAR-014 candidates")
        if not _pairwise_disjoint(ranked):
            raise TargetAPIError(
                "overlapping ranked candidates require selected_candidate_cids"
            )
        selected = ranked
    else:
        selected_ids = _unique_sorted_text(
            list(selected_candidate_cids),
            "selected_candidate_cids",
            limit=MAX_MODULES,
        )
        selected_list: list[ProgramPartitionCandidate] = []
        for cid in selected_ids:
            if cid not in by_cid:
                raise TargetAPIError("selected candidate is not present")
            if cid not in ranked_cids:
                raise TargetAPIError("selected candidate is not ranked by SPAR-014")
            if cid in rejected_cids:
                raise TargetAPIError("rejected candidates cannot become modules")
            if cid in advisory_cids:
                raise TargetAPIError("advisory candidates cannot become modules")
            selected_list.append(by_cid[cid])
        selected = tuple(selected_list)
        if not selected:
            raise TargetAPIError("selected_candidate_cids must not be empty")
        if not _pairwise_disjoint(selected):
            raise TargetAPIError("overlapping modules cannot form a simultaneous API plan")
    for candidate in selected:
        if candidate.advisory:
            raise TargetAPIError("advisory candidates cannot become modules")
        if candidate.hard_constraint_violations:
            raise TargetAPIError("violating candidates cannot become modules")
        if candidate.evidence_class in _NON_ADMITTING_EVIDENCE:
            raise TargetAPIError("vector or model evidence cannot admit a module API")
        if candidate.tree_id != tree_id:
            raise TargetAPIError("selected candidate tree_id does not match")
    member_to_module: dict[str, str] = {}
    owner_to_module: dict[str, str] = {}
    for candidate in selected:
        for member in candidate.member_ids:
            previous = member_to_module.get(member)
            if previous is not None:
                raise TargetAPIError("overlapping modules cannot form a simultaneous API plan")
            member_to_module[member] = candidate.candidate_cid
        for owner in candidate.state_owner_ids:
            previous = owner_to_module.get(owner)
            if previous is not None:
                raise TargetAPIError("state owner split across modules")
            owner_to_module[owner] = candidate.candidate_cid
    modules = tuple(
        _build_module_api(
            candidate,
            member_to_module=member_to_module,
            module_id=candidate.candidate_cid,
        )
        for candidate in selected
    )
    edges = _collect_dependency_edges(selected, member_to_module)
    evidence = tuple(
        sorted(
            {
                *resolved_comparison.evidence_cids,
                *(cid for item in selected for cid in item.evidence_cids),
            }
        )
    )
    return TargetModuleAPIPlan(
        tree_id=tree_id,
        modules=modules,
        dependency_edges=edges,
        selected_candidate_cids=tuple(item.candidate_cid for item in selected),
        rejected_candidate_cids=rejected_cids,
        advisory_candidate_cids=advisory_cids,
        evidence_cids=evidence,
        comparison_receipt_cid=resolved_comparison.receipt_cid,
        analyzer_id=ANALYZER_ID,
    )


def compile_target_api_receipt(
    plan: TargetModuleAPIPlan | Mapping[str, Any],
) -> TargetAPISynthesisReceipt:
    resolved = (
        plan if isinstance(plan, TargetModuleAPIPlan) else TargetModuleAPIPlan.from_dict(plan)
    )
    return TargetAPISynthesisReceipt(tree_id=resolved.tree_id, plan=resolved)


def encode_canonical_plan(plan: TargetModuleAPIPlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_plan(payload: Mapping[str, Any]) -> TargetModuleAPIPlan:
    return TargetModuleAPIPlan.from_dict(payload)


def encode_canonical_receipt(receipt: TargetAPISynthesisReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> TargetAPISynthesisReceipt:
    return TargetAPISynthesisReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise TargetAPIError(
            f"target API must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_KIND",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "AdapterAuthority",
    "DUCKLAKE_IS_AUTHORITY",
    "EXISTING_ADAPTER_AUTHORITIES",
    "ExportVisibility",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "MODULE_DEPENDENCY_EDGE_INTERFACE",
    "PLAN_IS_NOMINATION_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "PUBLIC_SURFACE_PROTOCOL",
    "RAW_SOURCE_REQUIRED",
    "RESPONSIBILITY_STATEMENT_INTERFACE",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "STATE_OWNER_INTERFACE_INTERFACE",
    "STATE_OWNER_KIND",
    "SYMBOL_KIND",
    "TASK_ID",
    "TARGET_ADAPTER_INTERFACE",
    "TARGET_API_CAN_AUTHORIZE_COMPLETION",
    "TARGET_API_CAN_AUTHORIZE_TRANSITION",
    "TARGET_API_CAN_CREATE_AUTHORITY",
    "TARGET_API_CONTRACT_VERSION",
    "TARGET_API_SYNTHESIS_RECEIPT_INTERFACE",
    "TARGET_EXPORT_INTERFACE",
    "TARGET_MODULE_API_INTERFACE",
    "TARGET_MODULE_API_PLAN_INTERFACE",
    "TARGET_PROTOCOL_INTERFACE",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "ModuleDependencyEdge",
    "ResponsibilityStatement",
    "StateOwnerInterface",
    "TargetAPIError",
    "TargetAPISynthesisReceipt",
    "TargetAdapter",
    "TargetExport",
    "TargetModuleAPI",
    "TargetModuleAPIPlan",
    "TargetProtocol",
    "assert_not_competing_capsule_family",
    "compile_target_api_receipt",
    "decode_canonical_plan",
    "decode_canonical_receipt",
    "encode_canonical_plan",
    "encode_canonical_receipt",
    "provider_free_exports",
    "synthesize_target_module_apis",
    "target_api_cid_profile",
]
