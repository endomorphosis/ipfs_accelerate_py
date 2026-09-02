"""SPAR-021 import, re-export, and callsite rewrite executor.

This module extends current supervisor partition orchestration with
``ImportRewrite@1`` and ``ReexportPlan@1``.  It consumes SPAR-019
``RefactorTransformationPacket@1`` rewrite edits, then nominates exact
symbol import/callsite rewrites and authorized re-exports, verifies
preimages, and rejects new cycles or undispositioned consumers.

The executor is nomination-only.  It does not apply CST transforms, does
not mutate, and cannot authorize a transition, completion, or competing
authority.  Vector, model, and heuristic evidence cannot admit a rewrite.
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
    EditKind,
    RefactorTransformationPacket,
    RewriteKind,
)


TASK_ID: Final[str] = "SPAR-021"
GOAL_ID: Final[str] = "SPAR-G041"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.import_rewriter@1"
)

IMPORT_REWRITE_INTERFACE: Final[str] = "ImportRewrite@1"
REEXPORT_PLAN_INTERFACE: Final[str] = "ReexportPlan@1"
IMPORT_REWRITE_RECEIPT_INTERFACE: Final[str] = "ImportRewriteReceipt@1"

IMPORT_REWRITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/import-rewrite@1"
)
REEXPORT_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reexport-plan@1"
)
IMPORT_REWRITE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/import-rewrite-receipt@1"
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
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_REWRITES: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_PATH_CHARS: Final[int] = 1_024
MAX_GRAPH_EDGES: Final[int] = 65_536

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

HANDLED_REWRITE_KINDS: Final[frozenset[str]] = frozenset(
    {
        RewriteKind.IMPORT.value,
        RewriteKind.CALLSITE.value,
        RewriteKind.REEXPORT.value,
    }
)
CONSUMER_REWRITE_KINDS: Final[frozenset[str]] = frozenset(
    {
        RewriteKind.IMPORT.value,
        RewriteKind.CALLSITE.value,
    }
)
REQUIRED_UNSUPPORTED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "undispositioned",
        "unsupported",
    }
)


class ImportRewriterError(ValueError):
    """Fail-closed violation of a SPAR-021 import-rewrite contract."""


class ImportRewriteKind(str, Enum):
    IMPORT = "import"
    CALLSITE = "callsite"


DECLARED_IMPORT_REWRITE_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in ImportRewriteKind
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise ImportRewriterError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ImportRewriterError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise ImportRewriterError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise ImportRewriterError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise ImportRewriterError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise ImportRewriterError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ImportRewriterError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ImportRewriterError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise ImportRewriterError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise ImportRewriterError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise ImportRewriterError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise ImportRewriterError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise ImportRewriterError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise ImportRewriterError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise ImportRewriterError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ImportRewriterError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise ImportRewriterError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise ImportRewriterError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise ImportRewriterError(f"unknown {name}: {text}") from exc


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
    raise ImportRewriterError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise ImportRewriterError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def import_rewriter_cid_profile() -> dict[str, str]:
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
            raise ImportRewriterError(f"{name} cannot claim {flag}")


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise ImportRewriterError(f"{name} exceeds path bound")
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
        raise ImportRewriterError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise ImportRewriterError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ImportRewriterError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ImportRewriterError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise ImportRewriterError(f"{name} exceeds path bound")
    return tuple(ordered)


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
class ImportRewrite:
    """One exact import or callsite symbol rewrite. Nomination only."""

    rewrite_kind: ImportRewriteKind | str
    consumer_id: str
    symbol_id: str
    source_module: str
    destination_module: str
    write_paths: Sequence[str]
    preimage_cid: str
    packet_cid: str
    tree_id: str
    obligation_id: str = ""

    interface: ClassVar[str] = IMPORT_REWRITE_INTERFACE
    schema: ClassVar[str] = IMPORT_REWRITE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "rewrite_kind",
            "consumer_id",
            "symbol_id",
            "source_module",
            "destination_module",
            "write_paths",
            "preimage_cid",
            "packet_cid",
            "tree_id",
            "obligation_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "rewrite_is_nomination_only",
            "rewrite_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _enum(self.rewrite_kind, ImportRewriteKind, "rewrite_kind")
        source = _text(self.source_module, "source_module")
        destination = _text(self.destination_module, "destination_module")
        if source == destination:
            raise ImportRewriterError(
                "import rewrite source and destination must differ"
            )
        object.__setattr__(self, "rewrite_kind", kind)
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        object.__setattr__(self, "symbol_id", _text(self.symbol_id, "symbol_id"))
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
            "schema": IMPORT_REWRITE_SCHEMA,
            "interface": IMPORT_REWRITE_INTERFACE,
            "rewrite_kind": self.rewrite_kind,
            "consumer_id": self.consumer_id,
            "symbol_id": self.symbol_id,
            "source_module": self.source_module,
            "destination_module": self.destination_module,
            "write_paths": list(self.write_paths),
            "preimage_cid": self.preimage_cid,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "obligation_id": self.obligation_id,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "ImportRewrite":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("rewrite_cid")
        if payload.pop("schema") != IMPORT_REWRITE_SCHEMA:
            raise ImportRewriterError("unsupported ImportRewrite schema")
        if payload.pop("interface") != IMPORT_REWRITE_INTERFACE:
            raise ImportRewriterError("unsupported ImportRewrite interface")
        _pop_authority_flags(payload, "ImportRewrite")
        if payload.pop("rewrite_is_nomination_only") is not True:
            raise ImportRewriterError("rewrite must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.rewrite_cid, "ImportRewrite rewrite_cid")
        return result


@dataclass(frozen=True, slots=True)
class ReexportPlan:
    """Authorized re-export from an original module. Nomination only."""

    source_module: str
    destination_module: str
    symbol_ids: Sequence[str]
    write_paths: Sequence[str]
    preimage_cid: str
    packet_cid: str
    tree_id: str
    consumer_ids: Sequence[str] = ()
    obligation_ids: Sequence[str] = ()
    authorized: bool = True

    interface: ClassVar[str] = REEXPORT_PLAN_INTERFACE
    schema: ClassVar[str] = REEXPORT_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "source_module",
            "destination_module",
            "symbol_ids",
            "write_paths",
            "preimage_cid",
            "packet_cid",
            "tree_id",
            "consumer_ids",
            "obligation_ids",
            "authorized",
            "rewrite_kind",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "plan_is_nomination_only",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        if not _bool(self.authorized, "authorized"):
            raise ImportRewriterError("reexport must be authorized")
        source = _text(self.source_module, "source_module")
        destination = _text(self.destination_module, "destination_module")
        if source == destination:
            raise ImportRewriterError(
                "reexport source and destination must differ"
            )
        symbols = _unique_sorted_text(
            list(self.symbol_ids), "symbol_ids", limit=MAX_MEMBERS
        )
        if not symbols:
            raise ImportRewriterError("reexport requires symbol_ids")
        object.__setattr__(self, "source_module", source)
        object.__setattr__(self, "destination_module", destination)
        object.__setattr__(self, "symbol_ids", symbols)
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
            "consumer_ids",
            _unique_sorted_text(list(self.consumer_ids), "consumer_ids", limit=MAX_MEMBERS),
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _unique_sorted_text(
                list(self.obligation_ids), "obligation_ids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(self, "authorized", True)

    @property
    def rewrite_kind(self) -> str:
        return RewriteKind.REEXPORT.value

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
            "schema": REEXPORT_PLAN_SCHEMA,
            "interface": REEXPORT_PLAN_INTERFACE,
            "source_module": self.source_module,
            "destination_module": self.destination_module,
            "symbol_ids": list(self.symbol_ids),
            "write_paths": list(self.write_paths),
            "preimage_cid": self.preimage_cid,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "consumer_ids": list(self.consumer_ids),
            "obligation_ids": list(self.obligation_ids),
            "authorized": True,
            "rewrite_kind": RewriteKind.REEXPORT.value,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "ReexportPlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != REEXPORT_PLAN_SCHEMA:
            raise ImportRewriterError("unsupported ReexportPlan schema")
        if payload.pop("interface") != REEXPORT_PLAN_INTERFACE:
            raise ImportRewriterError("unsupported ReexportPlan interface")
        _pop_authority_flags(payload, "ReexportPlan")
        if payload.pop("plan_is_nomination_only") is not True:
            raise ImportRewriterError("reexport plan must remain nomination_only")
        if payload.pop("rewrite_kind") != RewriteKind.REEXPORT.value:
            raise ImportRewriterError("reexport plan rewrite_kind must remain reexport")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "ReexportPlan plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class ImportRewriteReceipt:
    """Body-free SPAR-021 execution receipt. Independent validation remains separate."""

    tree_id: str
    packet_cid: str
    preimage_cid: str
    rewrite_cids: Sequence[str]
    reexport_cids: Sequence[str]
    write_paths: Sequence[str]
    analyzer_id: str = ANALYZER_ID
    cycle_free: bool = True
    preimage_verified: bool = True
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = IMPORT_REWRITE_RECEIPT_INTERFACE
    schema: ClassVar[str] = IMPORT_REWRITE_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "preimage_cid",
            "rewrite_cids",
            "reexport_cids",
            "write_paths",
            "analyzer_id",
            "cycle_free",
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
            raise ImportRewriterError("analyzer_id must remain the SPAR-021 analyzer")
        if not _bool(self.cycle_free, "cycle_free"):
            raise ImportRewriterError("receipt cannot admit a cycle")
        if not _bool(self.preimage_verified, "preimage_verified"):
            raise ImportRewriterError("receipt cannot skip preimage verification")
        if _bool(self.mutated, "mutated"):
            raise ImportRewriterError("executor cannot mutate")
        if not _bool(self.deterministic, "deterministic"):
            raise ImportRewriterError("executor must remain deterministic")
        rewrites = tuple(sorted(_cid(item, "rewrite_cids") for item in self.rewrite_cids))
        if len(rewrites) != len(set(rewrites)):
            raise ImportRewriterError("rewrite_cids must not contain duplicates")
        if len(rewrites) > MAX_REWRITES:
            raise ImportRewriterError("rewrite_cids exceed maximum length")
        reexports = tuple(
            sorted(_cid(item, "reexport_cids") for item in self.reexport_cids)
        )
        if len(reexports) != len(set(reexports)):
            raise ImportRewriterError("reexport_cids must not contain duplicates")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "rewrite_cids", rewrites)
        object.__setattr__(self, "reexport_cids", reexports)
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "analyzer_id", analyzer)
        object.__setattr__(self, "cycle_free", True)
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
            "schema": IMPORT_REWRITE_RECEIPT_SCHEMA,
            "interface": IMPORT_REWRITE_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "preimage_cid": self.preimage_cid,
            "rewrite_cids": list(self.rewrite_cids),
            "reexport_cids": list(self.reexport_cids),
            "write_paths": list(self.write_paths),
            "analyzer_id": self.analyzer_id,
            "cycle_free": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "ImportRewriteReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != IMPORT_REWRITE_RECEIPT_SCHEMA:
            raise ImportRewriterError("unsupported ImportRewriteReceipt schema")
        if payload.pop("interface") != IMPORT_REWRITE_RECEIPT_INTERFACE:
            raise ImportRewriterError("unsupported ImportRewriteReceipt interface")
        _pop_authority_flags(payload, "ImportRewriteReceipt")
        if payload.pop("executor_is_nomination_only") is not True:
            raise ImportRewriterError("executor must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "ImportRewriteReceipt receipt_cid")
        return result


def _coerce_packet(
    value: RefactorTransformationPacket | Mapping[str, Any],
) -> RefactorTransformationPacket:
    if isinstance(value, RefactorTransformationPacket):
        packet = value
    elif isinstance(value, Mapping):
        packet = RefactorTransformationPacket.from_dict(value)
    else:
        raise ImportRewriterError("packet must be a SPAR-019 RefactorTransformationPacket")
    if packet.analyzer_id != SPAR019_ANALYZER_ID:
        raise ImportRewriterError("packet must remain the SPAR-019 analyzer")
    return packet


def _as_mapping_list(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ImportRewriterError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(_mapping(item, name))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            items.append(_mapping(to_dict(), name))
            continue
        raise ImportRewriterError(f"{name} entries must be objects")
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
        raise ImportRewriterError(
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
            raise ImportRewriterError(
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
            raise ImportRewriterError("preimage does not verify")
    if source_cids is not None:
        claimed_sources = _unique_sorted_text(
            list(source_cids), "source_cids", limit=MAX_MEMBERS
        )
        if set(claimed_sources) != set(resolved.preimage.source_cids):
            raise ImportRewriterError("preimage does not verify")
    return preimage_cid


def _consumer_id_for(
    *,
    obligation_id: str,
    consumer_plans: Sequence[Mapping[str, Any]],
    fallback: str,
) -> str:
    for plan in consumer_plans:
        plan_obligation = _text(
            plan.get("obligation_id") or "", "obligation_id", empty=True
        )
        if plan_obligation and plan_obligation == obligation_id:
            return _text(plan.get("consumer_id") or fallback, "consumer_id")
    return _text(fallback, "consumer_id")


def _rewrite_edits(
    packet: RefactorTransformationPacket,
    kinds: frozenset[str],
) -> tuple[Any, ...]:
    return tuple(
        item
        for item in packet.edits
        if item.kind == EditKind.REWRITE.value and item.rewrite_kind in kinds
    )


def _require_handled_edits(packet: RefactorTransformationPacket) -> tuple[Any, ...]:
    handled = _rewrite_edits(packet, HANDLED_REWRITE_KINDS)
    if not handled:
        raise ImportRewriterError(
            "packet requires import, callsite, or reexport rewrite edits"
        )
    return handled


def _sort_rewrites(items: Sequence[ImportRewrite]) -> tuple[ImportRewrite, ...]:
    unique: dict[str, ImportRewrite] = {}
    for item in items:
        unique[item.rewrite_cid] = item
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.rewrite_kind,
                item.consumer_id,
                item.symbol_id,
                item.source_module,
                item.destination_module,
                item.obligation_id,
                item.rewrite_cid,
            ),
        )
    )
    if len(ordered) > MAX_REWRITES:
        raise ImportRewriterError("import rewrites exceed maximum length")
    return ordered


def _sort_plans(items: Sequence[ReexportPlan]) -> tuple[ReexportPlan, ...]:
    unique: dict[str, ReexportPlan] = {}
    for item in items:
        unique[item.plan_cid] = item
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.source_module,
                item.destination_module,
                item.plan_cid,
            ),
        )
    )


def _projected_import_graph(
    *,
    import_graph: Mapping[str, Sequence[str]] | None,
    rewrites: Sequence[ImportRewrite],
    plans: Sequence[ReexportPlan],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    adjacency: dict[str, set[str]] = {}
    if import_graph is not None:
        if not isinstance(import_graph, Mapping) or isinstance(
            import_graph, (str, bytes, bytearray)
        ):
            raise ImportRewriterError("import_graph must be an object")
        for source, destinations in import_graph.items():
            src = _text(source, "import_graph")
            if isinstance(destinations, (str, bytes, bytearray)) or not isinstance(
                destinations, Sequence
            ):
                raise ImportRewriterError("import_graph destinations must be a list")
            adjacency.setdefault(src, set())
            for destination in destinations:
                adjacency[src].add(_text(destination, "import_graph"))
    for rewrite in rewrites:
        targets = adjacency.setdefault(rewrite.consumer_id, set())
        targets.discard(rewrite.source_module)
        targets.add(rewrite.destination_module)
    for plan in plans:
        adjacency.setdefault(plan.source_module, set()).add(plan.destination_module)
    edge_count = sum(len(targets) for targets in adjacency.values())
    if edge_count > MAX_GRAPH_EDGES:
        raise ImportRewriterError("import_graph exceeds maximum length")
    edges = tuple(
        sorted(
            (source, target)
            for source, targets in adjacency.items()
            for target in targets
            if source != target
        )
    )
    nodes = tuple(
        sorted(
            set(adjacency)
            | {target for targets in adjacency.values() for target in targets}
        )
    )
    return nodes, edges


def _reject_cycles(
    *,
    import_graph: Mapping[str, Sequence[str]] | None,
    rewrites: Sequence[ImportRewrite],
    plans: Sequence[ReexportPlan],
) -> None:
    nodes, edges = _projected_import_graph(
        import_graph=import_graph,
        rewrites=rewrites,
        plans=plans,
    )
    if nodes and _has_cycle(nodes, edges):
        raise ImportRewriterError("new import cycles are rejected")


def _collect_import_rewrites(
    packet: RefactorTransformationPacket,
    *,
    consumer_plans: Sequence[Mapping[str, Any]],
) -> tuple[ImportRewrite, ...]:
    handled = _rewrite_edits(packet, HANDLED_REWRITE_KINDS)
    changed = set(packet.expected_delta.changed_binding_ids)
    nominated: list[ImportRewrite] = []
    for edit in handled:
        if edit.rewrite_kind not in CONSUMER_REWRITE_KINDS and not (
            edit.rewrite_kind == RewriteKind.REEXPORT.value
            and changed.intersection(edit.member_ids)
        ):
            continue
        kind = (
            edit.rewrite_kind
            if edit.rewrite_kind in CONSUMER_REWRITE_KINDS
            else ImportRewriteKind.IMPORT.value
        )
        consumer_id = _consumer_id_for(
            obligation_id=edit.obligation_id,
            consumer_plans=consumer_plans,
            fallback=edit.obligation_id or edit.source_id,
        )
        members = edit.member_ids
        if edit.rewrite_kind == RewriteKind.REEXPORT.value:
            members = tuple(item for item in members if item in changed)
        for symbol_id in members:
            nominated.append(
                ImportRewrite(
                    rewrite_kind=kind,
                    consumer_id=consumer_id,
                    symbol_id=symbol_id,
                    source_module=edit.source_id,
                    destination_module=edit.destination_id,
                    write_paths=edit.write_paths,
                    preimage_cid=packet.preimage.preimage_cid,
                    packet_cid=packet.packet_cid,
                    tree_id=packet.tree_id,
                    obligation_id=edit.obligation_id,
                )
            )
    return _sort_rewrites(nominated)


def _collect_reexport_plans(
    packet: RefactorTransformationPacket,
    *,
    consumer_plans: Sequence[Mapping[str, Any]],
) -> tuple[ReexportPlan, ...]:
    handled = _rewrite_edits(packet, frozenset({RewriteKind.REEXPORT.value}))
    grouped: dict[tuple[str, str, tuple[str, ...]], list[Any]] = {}
    for edit in handled:
        if edit.rewrite_kind != RewriteKind.REEXPORT.value:
            continue
        key = (edit.source_id, edit.destination_id, tuple(edit.write_paths))
        grouped.setdefault(key, []).append(edit)
    nominated: list[ReexportPlan] = []
    for (source, destination, paths), edits in grouped.items():
        symbols: list[str] = []
        consumers: list[str] = []
        obligations: list[str] = []
        for edit in edits:
            symbols.extend(edit.member_ids)
            if edit.obligation_id:
                obligations.append(edit.obligation_id)
                consumers.append(
                    _consumer_id_for(
                        obligation_id=edit.obligation_id,
                        consumer_plans=consumer_plans,
                        fallback=edit.obligation_id,
                    )
                )
        nominated.append(
            ReexportPlan(
                source_module=source,
                destination_module=destination,
                symbol_ids=tuple(sorted(set(symbols))),
                write_paths=paths,
                preimage_cid=packet.preimage.preimage_cid,
                packet_cid=packet.packet_cid,
                tree_id=packet.tree_id,
                consumer_ids=tuple(sorted(set(consumers))),
                obligation_ids=tuple(sorted(set(obligations))),
                authorized=True,
            )
        )
    return _sort_plans(nominated)


def compile_import_rewrites(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    consumer_plans: Sequence[Mapping[str, Any]] | None = None,
    undispositioned_consumer_ids: Sequence[str] = (),
    claimed_preimage_cid: str = "",
    import_graph: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ImportRewrite, ...]:
    """Nominate exact import/callsite rewrites from a SPAR-019 packet."""

    resolved = _coerce_packet(packet)
    verify_preimages(resolved, claimed_preimage_cid=claimed_preimage_cid)
    plans = _as_mapping_list(consumer_plans or (), "consumer_plans")
    _reject_undispositioned(
        undispositioned_consumer_ids=undispositioned_consumer_ids,
        consumer_plans=plans,
    )
    _require_handled_edits(resolved)
    rewrites = _collect_import_rewrites(resolved, consumer_plans=plans)
    reexport_plans = _collect_reexport_plans(resolved, consumer_plans=plans)
    _reject_cycles(import_graph=import_graph, rewrites=rewrites, plans=reexport_plans)
    return rewrites


def compile_reexport_plans(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    consumer_plans: Sequence[Mapping[str, Any]] | None = None,
    undispositioned_consumer_ids: Sequence[str] = (),
    claimed_preimage_cid: str = "",
    import_graph: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ReexportPlan, ...]:
    """Nominate authorized re-export plans from SPAR-019 reexport edits."""

    resolved = _coerce_packet(packet)
    verify_preimages(resolved, claimed_preimage_cid=claimed_preimage_cid)
    plans = _as_mapping_list(consumer_plans or (), "consumer_plans")
    _reject_undispositioned(
        undispositioned_consumer_ids=undispositioned_consumer_ids,
        consumer_plans=plans,
    )
    resolved_plans = _collect_reexport_plans(resolved, consumer_plans=plans)
    if not resolved_plans:
        raise ImportRewriterError("packet requires an authorized reexport")
    rewrites = _collect_import_rewrites(resolved, consumer_plans=plans)
    _reject_cycles(
        import_graph=import_graph,
        rewrites=rewrites,
        plans=resolved_plans,
    )
    return resolved_plans


def compile_reexport_plan(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ReexportPlan:
    """Return the unique authorized re-export plan for one packet."""

    plans = compile_reexport_plans(packet, **kwargs)
    if len(plans) != 1:
        raise ImportRewriterError("packet must nominate exactly one reexport plan")
    return plans[0]


def compile_import_rewrite_receipt(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    consumer_plans: Sequence[Mapping[str, Any]] | None = None,
    undispositioned_consumer_ids: Sequence[str] = (),
    claimed_preimage_cid: str = "",
    import_graph: Mapping[str, Sequence[str]] | None = None,
) -> ImportRewriteReceipt:
    """Compile a SPAR-021 receipt over nominated rewrites and re-exports."""

    resolved = _coerce_packet(packet)
    preimage_cid = verify_preimages(
        resolved, claimed_preimage_cid=claimed_preimage_cid
    )
    plans = _as_mapping_list(consumer_plans or (), "consumer_plans")
    rewrites = compile_import_rewrites(
        resolved,
        consumer_plans=plans,
        undispositioned_consumer_ids=undispositioned_consumer_ids,
        claimed_preimage_cid=preimage_cid,
        import_graph=import_graph,
    )
    reexport_plans = _collect_reexport_plans(resolved, consumer_plans=plans)
    _reject_cycles(
        import_graph=import_graph,
        rewrites=rewrites,
        plans=reexport_plans,
    )
    return ImportRewriteReceipt(
        tree_id=resolved.tree_id,
        packet_cid=resolved.packet_cid,
        preimage_cid=preimage_cid,
        rewrite_cids=tuple(item.rewrite_cid for item in rewrites),
        reexport_cids=tuple(item.plan_cid for item in reexport_plans),
        write_paths=resolved.effect_scope.write_paths,
    )


def execute_import_rewrites(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ImportRewriteReceipt:
    """Execute SPAR-021 as a deterministic no-mutation dry-run."""

    if kwargs.pop("mutate", False):
        raise ImportRewriterError("executor cannot mutate")
    return compile_import_rewrite_receipt(packet, **kwargs)


def execute_reexport_plan(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ReexportPlan:
    """Execute the unique authorized re-export plan without mutation."""

    if kwargs.pop("mutate", False):
        raise ImportRewriterError("executor cannot mutate")
    return compile_reexport_plan(packet, **kwargs)


def dry_run_import_rewrites(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> ImportRewriteReceipt:
    """Return a deterministic no-mutation dry-run of SPAR-021 rewrites."""

    receipt = execute_import_rewrites(packet, **kwargs)
    if receipt.mutated:
        raise ImportRewriterError("dry-run cannot mutate")
    return receipt


def encode_canonical_rewrite(rewrite: ImportRewrite) -> dict[str, Any]:
    return rewrite.to_dict()


def decode_canonical_rewrite(payload: Mapping[str, Any]) -> ImportRewrite:
    return ImportRewrite.from_dict(payload)


def encode_canonical_plan(plan: ReexportPlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_plan(payload: Mapping[str, Any]) -> ReexportPlan:
    return ReexportPlan.from_dict(payload)


def encode_canonical_receipt(receipt: ImportRewriteReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> ImportRewriteReceipt:
    return ImportRewriteReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise ImportRewriterError(
            f"import rewriter must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "CONSUMER_REWRITE_KINDS",
    "DECLARED_IMPORT_REWRITE_KINDS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXECUTOR_IS_NOMINATION_ONLY",
    "GOAL_ID",
    "HANDLED_REWRITE_KINDS",
    "IDENTITY_EXCLUDED_FIELDS",
    "IMPORT_REWRITE_INTERFACE",
    "IMPORT_REWRITE_RECEIPT_INTERFACE",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REEXPORT_PLAN_INTERFACE",
    "REWRITE_CAN_AUTHORIZE_COMPLETION",
    "REWRITE_CAN_AUTHORIZE_TRANSITION",
    "REWRITE_CAN_CREATE_AUTHORITY",
    "REWRITE_CONTRACT_VERSION",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "ImportRewrite",
    "ImportRewriteKind",
    "ImportRewriteReceipt",
    "ImportRewriterError",
    "ReexportPlan",
    "assert_not_competing_capsule_family",
    "compile_import_rewrite_receipt",
    "compile_import_rewrites",
    "compile_reexport_plan",
    "compile_reexport_plans",
    "decode_canonical_plan",
    "decode_canonical_receipt",
    "decode_canonical_rewrite",
    "dry_run_import_rewrites",
    "encode_canonical_plan",
    "encode_canonical_receipt",
    "encode_canonical_rewrite",
    "execute_import_rewrites",
    "execute_reexport_plan",
    "import_rewriter_cid_profile",
    "provider_free_exports",
    "verify_preimages",
]


assert TASK_ID == "SPAR-021"
assert IMPORT_REWRITE_INTERFACE == "ImportRewrite@1"
assert REEXPORT_PLAN_INTERFACE == "ReexportPlan@1"
assert EXECUTOR_IS_NOMINATION_ONLY is True
assert REWRITE_CAN_AUTHORIZE_COMPLETION is False
assert DRY_RUN_MUTATES is False
assert ANALYZER_ID != SPAR019_ANALYZER_ID
