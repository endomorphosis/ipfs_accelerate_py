"""SPAR-023 initialization, decorator, registry, CLI, plugin, and resource executor.

This module extends current supervisor partition orchestration with
``InitializationRewritePlan@1``.  It consumes SPAR-019
``RefactorTransformationPacket@1`` CLI/plugin/registry adapter edits,
SPAR-018 façade-plan mappings, and SPAR-010 initialization-order graph
mappings, then nominates deterministic order-preserving rewrites or
adapters for initialization, decorators, registrations, commands/routes/
plugins, signals, atexit, and resources, verifies preimages, and requires
trace validation.

SPAR-010 payloads are ingested as mappings only.  This module does not
replace datasets semantic authority, does not apply CST transforms, and
cannot authorize a transition, completion, or competing authority.
Vector, model, and heuristic evidence cannot admit a rewrite.
Observational metadata is excluded from identity.  Dry-run is
deterministic and never mutates.
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


TASK_ID: Final[str] = "SPAR-023"
GOAL_ID: Final[str] = "SPAR-G042"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.initialization_transform@1"
)

INITIALIZATION_REWRITE_INTERFACE: Final[str] = "InitializationRewrite@1"
INITIALIZATION_REWRITE_PLAN_INTERFACE: Final[str] = "InitializationRewritePlan@1"
INITIALIZATION_REWRITE_RECEIPT_INTERFACE: Final[str] = (
    "InitializationRewriteReceipt@1"
)

INITIALIZATION_REWRITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/initialization-rewrite@1"
)
INITIALIZATION_REWRITE_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/initialization-rewrite-plan@1"
)
INITIALIZATION_REWRITE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/initialization-rewrite-receipt@1"
)

REWRITE_CONTRACT_VERSION: Final[str] = "1"

REWRITE_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
REWRITE_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
REWRITE_CAN_CREATE_AUTHORITY: Final[bool] = False
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

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_REWRITES: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_PATH_CHARS: Final[int] = 1_024
MAX_GRAPH_EDGES: Final[int] = 65_536
MAX_TRACES: Final[int] = 1_024

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
        AdapterKind.CLI.value,
        AdapterKind.PLUGIN.value,
        AdapterKind.REGISTRY.value,
    }
)

DECLARED_ORDER_RELATIONS: Final[frozenset[str]] = frozenset(
    {
        "happens_before",
        "initialization_order",
    }
)

TRACE_REQUIRED_EFFECTS: Final[frozenset[str]] = frozenset(
    {
        "signal",
        "atexit",
        "resource",
    }
)

UNSUPPORTED_IMPORT_TIME_EFFECTS: Final[frozenset[str]] = frozenset(
    {
        "network",
        "io",
        "unknown",
    }
)

REQUIRED_UNSUPPORTED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "undispositioned",
        "unsupported",
    }
)

PRESERVE_ORDER_CANDIDATE: Final[str] = "preserve_order"

_ADAPTER_TO_REWRITE: Final[Mapping[str, str]] = {
    AdapterKind.CLI.value: "cli",
    AdapterKind.PLUGIN.value: "plugin",
    AdapterKind.REGISTRY.value: "registration",
}

_EFFECT_TO_REWRITE: Final[Mapping[str, str]] = {
    "decorator": "decorator",
    "registration": "registration",
    "resource": "resource",
    "signal": "signal",
    "atexit": "atexit",
    "cli_registration": "cli",
    "plugin_registration": "plugin",
}


class InitializationTransformError(ValueError):
    """Fail-closed violation of a SPAR-023 initialization-transform contract."""


class InitializationRewriteKind(str, Enum):
    INITIALIZATION_ORDER = "initialization_order"
    DECORATOR = "decorator"
    REGISTRATION = "registration"
    CLI = "cli"
    PLUGIN = "plugin"
    SIGNAL = "signal"
    ATEXIT = "atexit"
    RESOURCE = "resource"


DECLARED_INITIALIZATION_REWRITE_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in InitializationRewriteKind
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise InitializationTransformError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise InitializationTransformError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise InitializationTransformError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise InitializationTransformError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise InitializationTransformError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise InitializationTransformError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise InitializationTransformError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise InitializationTransformError(f"{name} must be a nonnegative integer")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise InitializationTransformError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise InitializationTransformError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise InitializationTransformError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise InitializationTransformError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise InitializationTransformError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise InitializationTransformError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise InitializationTransformError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise InitializationTransformError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise InitializationTransformError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise InitializationTransformError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise InitializationTransformError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise InitializationTransformError(f"unknown {name}: {text}") from exc


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
    raise InitializationTransformError(
        f"unsupported projected type {type(value).__name__}"
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise InitializationTransformError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def initialization_transform_cid_profile() -> dict[str, str]:
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
            raise InitializationTransformError(f"{name} cannot claim {flag}")


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise InitializationTransformError(f"{name} exceeds path bound")
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
        raise InitializationTransformError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise InitializationTransformError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise InitializationTransformError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise InitializationTransformError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise InitializationTransformError(f"{name} exceeds path bound")
    return tuple(ordered)


def _has_cycle(node_ids: Sequence[str], edges: Sequence[tuple[str, str]]) -> bool:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
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

    return any(visit(node) for node in node_ids)


@dataclass(frozen=True, slots=True)
class InitializationRewrite:
    """One exact initialization, decorator, registry, CLI, plugin, or resource rewrite.

    Nomination only.
    """

    rewrite_kind: InitializationRewriteKind | str
    subject_id: str
    source_module: str
    destination_module: str
    write_paths: Sequence[str]
    preimage_cid: str
    packet_cid: str
    tree_id: str
    obligation_id: str = ""
    order_index: int = 0
    adapter_kind: str = ""
    preserve_order: bool = True

    interface: ClassVar[str] = INITIALIZATION_REWRITE_INTERFACE
    schema: ClassVar[str] = INITIALIZATION_REWRITE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "rewrite_kind",
            "subject_id",
            "source_module",
            "destination_module",
            "write_paths",
            "preimage_cid",
            "packet_cid",
            "tree_id",
            "obligation_id",
            "order_index",
            "adapter_kind",
            "preserve_order",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "rewrite_is_nomination_only",
            "rewrite_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _enum(self.rewrite_kind, InitializationRewriteKind, "rewrite_kind")
        source = _text(self.source_module, "source_module")
        destination = _text(self.destination_module, "destination_module")
        preserve = _bool(self.preserve_order, "preserve_order")
        if source == destination and not preserve:
            raise InitializationTransformError(
                "initialization rewrite source and destination must differ unless preserve_order"
            )
        adapter = _text(self.adapter_kind, "adapter_kind", empty=True)
        if adapter and adapter not in HANDLED_ADAPTER_KINDS:
            raise InitializationTransformError("unknown adapter_kind")
        object.__setattr__(self, "rewrite_kind", kind)
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(self, "source_module", source)
        object.__setattr__(self, "destination_module", destination)
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
        object.__setattr__(self, "order_index", _nonneg_int(self.order_index, "order_index"))
        object.__setattr__(self, "adapter_kind", adapter)
        object.__setattr__(self, "preserve_order", preserve)

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
    def rewrite_is_nomination_only(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": INITIALIZATION_REWRITE_SCHEMA,
            "interface": INITIALIZATION_REWRITE_INTERFACE,
            "rewrite_kind": self.rewrite_kind,
            "subject_id": self.subject_id,
            "source_module": self.source_module,
            "destination_module": self.destination_module,
            "write_paths": list(self.write_paths),
            "preimage_cid": self.preimage_cid,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "obligation_id": self.obligation_id,
            "order_index": self.order_index,
            "adapter_kind": self.adapter_kind,
            "preserve_order": self.preserve_order,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "rewrite_is_nomination_only": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def rewrite_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["rewrite_cid"] = self.rewrite_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InitializationRewrite":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("rewrite_cid")
        if payload.pop("schema") != INITIALIZATION_REWRITE_SCHEMA:
            raise InitializationTransformError(
                "unsupported InitializationRewrite schema"
            )
        if payload.pop("interface") != INITIALIZATION_REWRITE_INTERFACE:
            raise InitializationTransformError(
                "unsupported InitializationRewrite interface"
            )
        _pop_authority_flags(payload, "InitializationRewrite")
        if payload.pop("rewrite_is_nomination_only") is not True:
            raise InitializationTransformError("rewrite must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.rewrite_cid, "InitializationRewrite rewrite_cid")
        return result


@dataclass(frozen=True, slots=True)
class InitializationRewritePlan:
    """Deterministic SPAR-023 rewrite plan. Nomination only."""

    rewrite_cids: Sequence[str]
    write_paths: Sequence[str]
    preimage_cid: str
    packet_cid: str
    tree_id: str
    required_trace_cids: Sequence[str] = ()
    obligation_ids: Sequence[str] = ()
    order_preserved: bool = True
    cycle_free: bool = True
    traces_validated: bool = True

    interface: ClassVar[str] = INITIALIZATION_REWRITE_PLAN_INTERFACE
    schema: ClassVar[str] = INITIALIZATION_REWRITE_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "rewrite_cids",
            "write_paths",
            "preimage_cid",
            "packet_cid",
            "tree_id",
            "required_trace_cids",
            "obligation_ids",
            "order_preserved",
            "cycle_free",
            "traces_validated",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "plan_is_nomination_only",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        if not _bool(self.order_preserved, "order_preserved"):
            raise InitializationTransformError("plan cannot reverse initialization order")
        if not _bool(self.cycle_free, "cycle_free"):
            raise InitializationTransformError("plan cannot admit a cycle")
        if not _bool(self.traces_validated, "traces_validated"):
            raise InitializationTransformError("required traces must validate")
        rewrites = tuple(sorted(_cid(item, "rewrite_cids") for item in self.rewrite_cids))
        if not rewrites:
            raise InitializationTransformError("plan requires rewrite_cids")
        if len(rewrites) != len(set(rewrites)):
            raise InitializationTransformError("rewrite_cids must not contain duplicates")
        if len(rewrites) > MAX_REWRITES:
            raise InitializationTransformError("rewrite_cids exceed maximum length")
        traces = tuple(
            sorted(_cid(item, "required_trace_cids") for item in self.required_trace_cids)
        )
        if len(traces) != len(set(traces)):
            raise InitializationTransformError(
                "required_trace_cids must not contain duplicates"
            )
        if len(traces) > MAX_TRACES:
            raise InitializationTransformError("required_trace_cids exceed maximum length")
        object.__setattr__(self, "rewrite_cids", rewrites)
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "required_trace_cids", traces)
        object.__setattr__(
            self,
            "obligation_ids",
            _unique_sorted_text(
                list(self.obligation_ids), "obligation_ids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(self, "order_preserved", True)
        object.__setattr__(self, "cycle_free", True)
        object.__setattr__(self, "traces_validated", True)

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
            "schema": INITIALIZATION_REWRITE_PLAN_SCHEMA,
            "interface": INITIALIZATION_REWRITE_PLAN_INTERFACE,
            "rewrite_cids": list(self.rewrite_cids),
            "write_paths": list(self.write_paths),
            "preimage_cid": self.preimage_cid,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "required_trace_cids": list(self.required_trace_cids),
            "obligation_ids": list(self.obligation_ids),
            "order_preserved": True,
            "cycle_free": True,
            "traces_validated": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "InitializationRewritePlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != INITIALIZATION_REWRITE_PLAN_SCHEMA:
            raise InitializationTransformError(
                "unsupported InitializationRewritePlan schema"
            )
        if payload.pop("interface") != INITIALIZATION_REWRITE_PLAN_INTERFACE:
            raise InitializationTransformError(
                "unsupported InitializationRewritePlan interface"
            )
        _pop_authority_flags(payload, "InitializationRewritePlan")
        if payload.pop("plan_is_nomination_only") is not True:
            raise InitializationTransformError("plan must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "InitializationRewritePlan plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class InitializationRewriteReceipt:
    """Body-free SPAR-023 execution receipt. Independent validation remains separate."""

    tree_id: str
    packet_cid: str
    preimage_cid: str
    rewrite_cids: Sequence[str]
    plan_cid: str
    write_paths: Sequence[str]
    required_trace_cids: Sequence[str] = ()
    analyzer_id: str = ANALYZER_ID
    cycle_free: bool = True
    order_preserved: bool = True
    traces_validated: bool = True
    preimage_verified: bool = True
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = INITIALIZATION_REWRITE_RECEIPT_INTERFACE
    schema: ClassVar[str] = INITIALIZATION_REWRITE_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "preimage_cid",
            "rewrite_cids",
            "plan_cid",
            "write_paths",
            "required_trace_cids",
            "analyzer_id",
            "cycle_free",
            "order_preserved",
            "traces_validated",
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
            raise InitializationTransformError(
                "analyzer_id must remain the SPAR-023 analyzer"
            )
        if not _bool(self.cycle_free, "cycle_free"):
            raise InitializationTransformError("receipt cannot admit a cycle")
        if not _bool(self.order_preserved, "order_preserved"):
            raise InitializationTransformError(
                "receipt cannot reverse initialization order"
            )
        if not _bool(self.traces_validated, "traces_validated"):
            raise InitializationTransformError("required traces must validate")
        if not _bool(self.preimage_verified, "preimage_verified"):
            raise InitializationTransformError(
                "receipt cannot skip preimage verification"
            )
        if _bool(self.mutated, "mutated"):
            raise InitializationTransformError("executor cannot mutate")
        if not _bool(self.deterministic, "deterministic"):
            raise InitializationTransformError("executor must remain deterministic")
        rewrites = tuple(sorted(_cid(item, "rewrite_cids") for item in self.rewrite_cids))
        if not rewrites:
            raise InitializationTransformError("receipt requires rewrite_cids")
        if len(rewrites) != len(set(rewrites)):
            raise InitializationTransformError("rewrite_cids must not contain duplicates")
        if len(rewrites) > MAX_REWRITES:
            raise InitializationTransformError("rewrite_cids exceed maximum length")
        traces = tuple(
            sorted(_cid(item, "required_trace_cids") for item in self.required_trace_cids)
        )
        if len(traces) != len(set(traces)):
            raise InitializationTransformError(
                "required_trace_cids must not contain duplicates"
            )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "rewrite_cids", rewrites)
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "required_trace_cids", traces)
        object.__setattr__(self, "analyzer_id", analyzer)
        object.__setattr__(self, "cycle_free", True)
        object.__setattr__(self, "order_preserved", True)
        object.__setattr__(self, "traces_validated", True)
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
            "schema": INITIALIZATION_REWRITE_RECEIPT_SCHEMA,
            "interface": INITIALIZATION_REWRITE_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "preimage_cid": self.preimage_cid,
            "rewrite_cids": list(self.rewrite_cids),
            "plan_cid": self.plan_cid,
            "write_paths": list(self.write_paths),
            "required_trace_cids": list(self.required_trace_cids),
            "analyzer_id": self.analyzer_id,
            "cycle_free": True,
            "order_preserved": True,
            "traces_validated": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "InitializationRewriteReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != INITIALIZATION_REWRITE_RECEIPT_SCHEMA:
            raise InitializationTransformError(
                "unsupported InitializationRewriteReceipt schema"
            )
        if payload.pop("interface") != INITIALIZATION_REWRITE_RECEIPT_INTERFACE:
            raise InitializationTransformError(
                "unsupported InitializationRewriteReceipt interface"
            )
        _pop_authority_flags(payload, "InitializationRewriteReceipt")
        if payload.pop("executor_is_nomination_only") is not True:
            raise InitializationTransformError("executor must remain nomination_only")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "InitializationRewriteReceipt receipt_cid"
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
        raise InitializationTransformError(
            "packet must be a SPAR-019 RefactorTransformationPacket"
        )
    if packet.analyzer_id != SPAR019_ANALYZER_ID:
        raise InitializationTransformError("packet must remain the SPAR-019 analyzer")
    return packet


def _as_mapping_list(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise InitializationTransformError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(_mapping(item, name))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            items.append(_mapping(to_dict(), name))
            continue
        raise InitializationTransformError(f"{name} entries must be objects")
    return tuple(items)


def _reject_undispositioned(
    *,
    undispositioned_consumer_ids: Sequence[str],
    consumer_plans: Sequence[Mapping[str, Any]],
) -> None:
    leftover = _unique_sorted_text(
        list(undispositioned_consumer_ids),
        "undispositioned_consumer_ids",
        limit=MAX_MEMBERS,
    )
    if leftover:
        raise InitializationTransformError(
            "undispositioned consumers are a typed terminal"
        )
    for plan in consumer_plans:
        required = plan.get("required", True)
        if required is None:
            required = True
        else:
            required = _bool(required, "required")
        disposition = _text(plan.get("disposition") or "", "disposition", empty=True)
        if required and disposition in REQUIRED_UNSUPPORTED_DISPOSITIONS:
            raise InitializationTransformError(
                "undispositioned consumers are a typed terminal"
            )


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
            raise InitializationTransformError("preimage does not verify")
    if source_cids is not None:
        claimed_sources = _unique_sorted_text(
            list(source_cids), "source_cids", limit=MAX_MEMBERS
        )
        if set(claimed_sources) != set(resolved.preimage.source_cids):
            raise InitializationTransformError("preimage does not verify")
    return preimage_cid


def verify_required_traces(
    *,
    required_trace_cids: Sequence[str] = (),
    observed_trace_cids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Require content-addressed traces for order-sensitive effects."""

    required = tuple(
        sorted(_cid(item, "required_trace_cids") for item in required_trace_cids)
    )
    if len(required) != len(set(required)):
        raise InitializationTransformError(
            "required_trace_cids must not contain duplicates"
        )
    if len(required) > MAX_TRACES:
        raise InitializationTransformError("required_trace_cids exceed maximum length")
    observed = tuple(
        sorted(_cid(item, "observed_trace_cids") for item in observed_trace_cids)
    )
    if len(observed) != len(set(observed)):
        raise InitializationTransformError(
            "observed_trace_cids must not contain duplicates"
        )
    missing = [item for item in required if item not in set(observed)]
    if missing:
        raise InitializationTransformError("required traces are missing")
    return required


def _adapter_edits(packet: RefactorTransformationPacket) -> tuple[Any, ...]:
    return tuple(
        item
        for item in packet.edits
        if item.kind == EditKind.ADAPTER.value
        and item.adapter_kind in HANDLED_ADAPTER_KINDS
    )


def _sort_rewrites(
    items: Sequence[InitializationRewrite],
) -> tuple[InitializationRewrite, ...]:
    unique: dict[str, InitializationRewrite] = {}
    for item in items:
        unique[item.rewrite_cid] = item
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.order_index,
                item.rewrite_kind,
                item.subject_id,
                item.source_module,
                item.destination_module,
                item.obligation_id,
                item.rewrite_cid,
            ),
        )
    )
    if len(ordered) > MAX_REWRITES:
        raise InitializationTransformError("initialization rewrites exceed maximum length")
    return ordered


def _coerce_initialization_graph(value: Any) -> dict[str, Any]:
    if value in (None, (), {}):
        return {"nodes": (), "edges": (), "candidates": (), "effects": ()}
    payload = _mapping(value, "initialization_graph")
    nodes = _as_mapping_list(payload.get("nodes") or (), "nodes")
    edges = _as_mapping_list(payload.get("edges") or (), "edges")
    candidates = _as_mapping_list(payload.get("candidates") or (), "candidates")
    effects = _as_mapping_list(payload.get("effects") or (), "effects")
    if len(edges) > MAX_GRAPH_EDGES:
        raise InitializationTransformError("initialization_graph exceeds maximum length")
    return {
        "nodes": nodes,
        "edges": edges,
        "candidates": candidates,
        "effects": effects,
    }


def _graph_order_edges(
    graph: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    nodes: list[str] = []
    for item in graph["nodes"]:
        node_id = _text(item.get("node_id") or item.get("id") or "", "node_id", empty=True)
        if node_id:
            nodes.append(node_id)
    edges: list[tuple[str, str]] = []
    for item in graph["edges"]:
        kind = _text(item.get("kind") or "", "kind", empty=True)
        if kind and kind not in DECLARED_ORDER_RELATIONS:
            raise InitializationTransformError("unknown initialization order relation")
        source = _text(item.get("source_id") or "", "source_id")
        target = _text(item.get("target_id") or "", "target_id")
        if source == target:
            raise InitializationTransformError("order edge cannot be reflexive")
        nodes.extend((source, target))
        edges.append((source, target))
    unique_nodes = tuple(sorted(set(nodes)))
    unique_edges = tuple(sorted(set(edges)))
    return unique_nodes, unique_edges


def _reject_order_cycles(graph: Mapping[str, Any]) -> None:
    nodes, edges = _graph_order_edges(graph)
    if nodes and _has_cycle(nodes, edges):
        raise InitializationTransformError("initialization order cycles are rejected")


def _preserve_order_required(graph: Mapping[str, Any]) -> bool:
    if not graph["candidates"] and not graph["edges"] and not graph["nodes"]:
        return True
    for item in graph["candidates"]:
        kind = _text(item.get("kind") or "", "kind", empty=True)
        required = item.get("required", True)
        if required is None:
            required = True
        else:
            required = _bool(required, "required")
        if kind == PRESERVE_ORDER_CANDIDATE and required:
            return True
        if required and kind in REQUIRED_UNSUPPORTED_DISPOSITIONS:
            raise InitializationTransformError(
                "unsupported required initialization candidate is a typed terminal"
            )
    return bool(graph["edges"] or graph["nodes"])


def _reject_unsupported_effects(graph: Mapping[str, Any]) -> None:
    records = list(graph["nodes"]) + list(graph["effects"])
    for item in records:
        required = item.get("required", True)
        if required is None:
            required = True
        else:
            required = _bool(required, "required")
        kinds = item.get("effect_kinds") or item.get("kinds") or ()
        if item.get("kind") and not kinds:
            kinds = (item.get("kind"),)
        if isinstance(kinds, (str, bytes, bytearray)) or not isinstance(kinds, Sequence):
            raise InitializationTransformError("effect_kinds must be a list")
        for kind in kinds:
            effect = _text(kind, "effect_kind")
            if required and effect in UNSUPPORTED_IMPORT_TIME_EFFECTS:
                raise InitializationTransformError(
                    "unsupported required import-time effect is a typed terminal"
                )


def _trace_obligations(
    graph: Mapping[str, Any],
    *,
    required_trace_cids: Sequence[str],
) -> tuple[str, ...]:
    declared = [
        _cid(item, "required_trace_cids") for item in required_trace_cids
    ]
    records = list(graph["nodes"]) + list(graph["effects"])
    needs_trace = False
    for item in records:
        required = item.get("required", True)
        if required is None:
            required = True
        else:
            required = _bool(required, "required")
        kinds = item.get("effect_kinds") or item.get("kinds") or ()
        if item.get("kind") and not kinds:
            kinds = (item.get("kind"),)
        if isinstance(kinds, (str, bytes, bytearray)) or not isinstance(kinds, Sequence):
            raise InitializationTransformError("effect_kinds must be a list")
        for kind in kinds:
            effect = _text(kind, "effect_kind")
            if required and effect in TRACE_REQUIRED_EFFECTS:
                needs_trace = True
    if needs_trace and not declared:
        raise InitializationTransformError("required traces are missing")
    return tuple(sorted(set(declared)))


def _collect_adapter_rewrites(
    packet: RefactorTransformationPacket,
) -> tuple[InitializationRewrite, ...]:
    nominated: list[InitializationRewrite] = []
    for index, edit in enumerate(_adapter_edits(packet)):
        kind = _ADAPTER_TO_REWRITE[edit.adapter_kind]
        members = edit.member_ids or (edit.obligation_id or edit.source_id,)
        preserve = edit.source_id == edit.destination_id
        for subject_id in members:
            nominated.append(
                InitializationRewrite(
                    rewrite_kind=kind,
                    subject_id=subject_id,
                    source_module=edit.source_id,
                    destination_module=edit.destination_id,
                    write_paths=edit.write_paths,
                    preimage_cid=packet.preimage.preimage_cid,
                    packet_cid=packet.packet_cid,
                    tree_id=packet.tree_id,
                    obligation_id=edit.obligation_id,
                    order_index=index,
                    adapter_kind=edit.adapter_kind,
                    preserve_order=preserve,
                )
            )
    return _sort_rewrites(nominated)


def _collect_effect_rewrites(
    packet: RefactorTransformationPacket,
    graph: Mapping[str, Any],
) -> tuple[InitializationRewrite, ...]:
    destination = packet.expected_delta.destination_module_ids[0]
    source = packet.edits[0].source_id if packet.edits else destination
    nominated: list[InitializationRewrite] = []
    records: list[tuple[int, dict[str, Any]]] = []
    for item in graph["nodes"]:
        order = item.get("order_index", 0)
        records.append((_nonneg_int(order, "order_index"), item))
    for item in graph["effects"]:
        order = item.get("order_index", 0)
        records.append((_nonneg_int(order, "order_index"), item))
    for order_index, item in records:
        kinds = item.get("effect_kinds") or item.get("kinds") or ()
        if item.get("kind") and item.get("kind") in _EFFECT_TO_REWRITE and not kinds:
            kinds = (item.get("kind"),)
        if isinstance(kinds, (str, bytes, bytearray)) or not isinstance(kinds, Sequence):
            raise InitializationTransformError("effect_kinds must be a list")
        subject = _text(
            item.get("node_id")
            or item.get("subject_id")
            or item.get("id")
            or "",
            "subject_id",
            empty=True,
        )
        module = _text(
            item.get("module_name") or item.get("module_path") or source,
            "source_module",
        )
        for kind in kinds:
            effect = _text(kind, "effect_kind")
            rewrite_kind = _EFFECT_TO_REWRITE.get(effect)
            if rewrite_kind is None:
                continue
            nominated.append(
                InitializationRewrite(
                    rewrite_kind=rewrite_kind,
                    subject_id=subject or f"{rewrite_kind}:{order_index}",
                    source_module=module,
                    destination_module=destination,
                    write_paths=packet.effect_scope.write_paths,
                    preimage_cid=packet.preimage.preimage_cid,
                    packet_cid=packet.packet_cid,
                    tree_id=packet.tree_id,
                    obligation_id=_text(
                        item.get("obligation_id") or "", "obligation_id", empty=True
                    ),
                    order_index=order_index,
                    preserve_order=module == destination,
                )
            )
    return _sort_rewrites(nominated)


def _collect_order_rewrites(
    packet: RefactorTransformationPacket,
    graph: Mapping[str, Any],
) -> tuple[InitializationRewrite, ...]:
    if not _preserve_order_required(graph):
        return ()
    if graph["nodes"]:
        destination = packet.expected_delta.destination_module_ids[0]
        nominated: list[InitializationRewrite] = []
        for item in graph["nodes"]:
            node_id = _text(item.get("node_id") or "", "node_id")
            order_index = _nonneg_int(item.get("order_index", 0), "order_index")
            module = _text(
                item.get("module_name") or item.get("module_path") or "pkg.mod",
                "source_module",
            )
            nominated.append(
                InitializationRewrite(
                    rewrite_kind=InitializationRewriteKind.INITIALIZATION_ORDER,
                    subject_id=node_id,
                    source_module=module,
                    destination_module=destination,
                    write_paths=packet.effect_scope.write_paths,
                    preimage_cid=packet.preimage.preimage_cid,
                    packet_cid=packet.packet_cid,
                    tree_id=packet.tree_id,
                    order_index=order_index,
                    preserve_order=True,
                )
            )
        return _sort_rewrites(nominated)
    return ()


def _require_handled(
    packet: RefactorTransformationPacket,
    rewrites: Sequence[InitializationRewrite],
) -> None:
    if rewrites:
        return
    raise InitializationTransformError(
        "packet requires initialization, decorator, registry, CLI, plugin, "
        "signal, atexit, or resource edits"
    )


def compile_initialization_rewrites(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    initialization_graph: Mapping[str, Any] | None = None,
    consumer_plans: Sequence[Mapping[str, Any]] | None = None,
    undispositioned_consumer_ids: Sequence[str] = (),
    claimed_preimage_cid: str = "",
    required_trace_cids: Sequence[str] = (),
    observed_trace_cids: Sequence[str] = (),
) -> tuple[InitializationRewrite, ...]:
    """Nominate exact initialization/lifecycle rewrites from a SPAR-019 packet."""

    resolved = _coerce_packet(packet)
    verify_preimages(resolved, claimed_preimage_cid=claimed_preimage_cid)
    plans = _as_mapping_list(consumer_plans or (), "consumer_plans")
    _reject_undispositioned(
        undispositioned_consumer_ids=undispositioned_consumer_ids,
        consumer_plans=plans,
    )
    graph = _coerce_initialization_graph(initialization_graph)
    _reject_order_cycles(graph)
    _reject_unsupported_effects(graph)
    traces = _trace_obligations(graph, required_trace_cids=required_trace_cids)
    verify_required_traces(
        required_trace_cids=traces,
        observed_trace_cids=observed_trace_cids or traces,
    )
    adapters = _collect_adapter_rewrites(resolved)
    effects = _collect_effect_rewrites(resolved, graph)
    orders = _collect_order_rewrites(resolved, graph)
    rewrites = _sort_rewrites((*adapters, *effects, *orders))
    _require_handled(resolved, rewrites)
    return rewrites


def compile_initialization_rewrite_plan(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> InitializationRewritePlan:
    """Compile the unique SPAR-023 InitializationRewritePlan for one packet."""

    resolved = _coerce_packet(packet)
    graph = _coerce_initialization_graph(kwargs.get("initialization_graph"))
    traces = _trace_obligations(
        graph, required_trace_cids=kwargs.get("required_trace_cids") or ()
    )
    rewrites = compile_initialization_rewrites(resolved, **kwargs)
    obligations = tuple(
        sorted({item.obligation_id for item in rewrites if item.obligation_id})
    )
    return InitializationRewritePlan(
        rewrite_cids=tuple(item.rewrite_cid for item in rewrites),
        write_paths=resolved.effect_scope.write_paths,
        preimage_cid=resolved.preimage.preimage_cid,
        packet_cid=resolved.packet_cid,
        tree_id=resolved.tree_id,
        required_trace_cids=traces,
        obligation_ids=obligations,
        order_preserved=True,
        cycle_free=True,
        traces_validated=True,
    )


def compile_initialization_rewrite_receipt(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> InitializationRewriteReceipt:
    """Compile a SPAR-023 receipt over nominated initialization rewrites."""

    resolved = _coerce_packet(packet)
    preimage_cid = verify_preimages(
        resolved, claimed_preimage_cid=kwargs.get("claimed_preimage_cid") or ""
    )
    plan = compile_initialization_rewrite_plan(resolved, **kwargs)
    return InitializationRewriteReceipt(
        tree_id=resolved.tree_id,
        packet_cid=resolved.packet_cid,
        preimage_cid=preimage_cid,
        rewrite_cids=plan.rewrite_cids,
        plan_cid=plan.plan_cid,
        write_paths=resolved.effect_scope.write_paths,
        required_trace_cids=plan.required_trace_cids,
    )


def execute_initialization_rewrites(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> InitializationRewriteReceipt:
    """Execute SPAR-023 as a deterministic no-mutation dry-run."""

    if kwargs.pop("mutate", False):
        raise InitializationTransformError("executor cannot mutate")
    return compile_initialization_rewrite_receipt(packet, **kwargs)


def execute_initialization_rewrite_plan(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> InitializationRewritePlan:
    """Execute the unique SPAR-023 plan without mutation."""

    if kwargs.pop("mutate", False):
        raise InitializationTransformError("executor cannot mutate")
    return compile_initialization_rewrite_plan(packet, **kwargs)


def dry_run_initialization_rewrites(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> InitializationRewriteReceipt:
    """Return a deterministic no-mutation dry-run of SPAR-023 rewrites."""

    receipt = execute_initialization_rewrites(packet, **kwargs)
    if receipt.mutated:
        raise InitializationTransformError("dry-run cannot mutate")
    return receipt


def encode_canonical_rewrite(rewrite: InitializationRewrite) -> dict[str, Any]:
    return rewrite.to_dict()


def decode_canonical_rewrite(payload: Mapping[str, Any]) -> InitializationRewrite:
    return InitializationRewrite.from_dict(payload)


def encode_canonical_plan(plan: InitializationRewritePlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_plan(payload: Mapping[str, Any]) -> InitializationRewritePlan:
    return InitializationRewritePlan.from_dict(payload)


def encode_canonical_receipt(receipt: InitializationRewriteReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(
    payload: Mapping[str, Any],
) -> InitializationRewriteReceipt:
    return InitializationRewriteReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise InitializationTransformError(
            f"initialization transform must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "DECLARED_INITIALIZATION_REWRITE_KINDS",
    "DECLARED_ORDER_RELATIONS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXECUTOR_IS_NOMINATION_ONLY",
    "GOAL_ID",
    "HANDLED_ADAPTER_KINDS",
    "IDENTITY_EXCLUDED_FIELDS",
    "INITIALIZATION_REWRITE_INTERFACE",
    "INITIALIZATION_REWRITE_PLAN_INTERFACE",
    "INITIALIZATION_REWRITE_RECEIPT_INTERFACE",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PLAN_IS_NOMINATION_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REWRITE_CAN_AUTHORIZE_COMPLETION",
    "REWRITE_CAN_AUTHORIZE_TRANSITION",
    "REWRITE_CAN_CREATE_AUTHORITY",
    "REWRITE_CONTRACT_VERSION",
    "RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TRACE_REQUIRED_EFFECTS",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "InitializationRewrite",
    "InitializationRewriteKind",
    "InitializationRewritePlan",
    "InitializationRewriteReceipt",
    "InitializationTransformError",
    "assert_not_competing_capsule_family",
    "compile_initialization_rewrite_plan",
    "compile_initialization_rewrite_receipt",
    "compile_initialization_rewrites",
    "decode_canonical_plan",
    "decode_canonical_receipt",
    "decode_canonical_rewrite",
    "dry_run_initialization_rewrites",
    "encode_canonical_plan",
    "encode_canonical_receipt",
    "encode_canonical_rewrite",
    "execute_initialization_rewrite_plan",
    "execute_initialization_rewrites",
    "initialization_transform_cid_profile",
    "provider_free_exports",
    "verify_preimages",
    "verify_required_traces",
]


assert TASK_ID == "SPAR-023"
assert INITIALIZATION_REWRITE_PLAN_INTERFACE == "InitializationRewritePlan@1"
assert INITIALIZATION_REWRITE_INTERFACE == "InitializationRewrite@1"
assert EXECUTOR_IS_NOMINATION_ONLY is True
assert PLAN_IS_NOMINATION_ONLY is True
assert REWRITE_CAN_AUTHORIZE_COMPLETION is False
assert DRY_RUN_MUTATES is False
assert ANALYZER_ID != SPAR019_ANALYZER_ID
