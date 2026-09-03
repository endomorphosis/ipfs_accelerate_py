"""SPAR-039 semantic world-root, VFS outbox, recovery, and gitlink adapter.

This module extends current kit storage/VFS/WAL/CAS authorities with
``SemanticRefactorWorldRootAdapter@1``.  It nominates persistence of SPAR-025
packets, kit projections, receipts, and SPAR-033 transitions through those
authorities, then binds generation CAS, WAL recovery, stale-writer
rejection, gitlinks, and explicit cross-repository ownership.

``ipfs_kit_py`` remains the VFS, outbox, WAL/recovery, and generation-bearing
root-CAS authority.  This adapter never writes the repository, never mutates
VFS, never overwrites a conflicting or recovered root, and never creates a
competing task, graph, identity, VFS, proof, context, scheduler, vector,
merge, or state authority.

The adapter is nomination-only.  Crashes, stale writers, root conflicts, and
cross-repository integration recover without overwrite.  Nested repository
writes are nominated only when ``ipfs_accelerate_py`` owns the task and
gitlink integration is explicit.  Vector, model, and heuristic evidence
cannot admit a root.  Observational metadata is excluded from identity.
Dry-run is deterministic and never mutates.  Network is denied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)


TASK_ID: Final[str] = "SPAR-039"
GOAL_ID: Final[str] = "SPAR-G072"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "storage and retrieval authority"
AUTHORITY_OWNER: Final[str] = "ipfs_kit_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.world_root_adapter@1"
)
PREDECESSOR_TASK_IDS: Final[tuple[str, ...]] = ("SPAR-025", "SPAR-033")
VFS_AUTHORITY_OWNER: Final[str] = "ipfs_kit_py"
CAS_AUTHORITY_OWNER: Final[str] = "ipfs_kit_py"
OUTBOX_AUTHORITY_OWNER: Final[str] = "ipfs_kit_py"
WAL_AUTHORITY_OWNER: Final[str] = "ipfs_kit_py"
ACCELERATOR_TASK_OWNER: Final[str] = "ipfs_accelerate_py"

SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_INTERFACE: Final[str] = (
    "SemanticRefactorWorldRootAdapter@1"
)
SEMANTIC_WORLD_ROOT_INTERFACE: Final[str] = "SemanticWorldRoot@1"
GENERATION_CAS_BINDING_INTERFACE: Final[str] = "GenerationCasBinding@1"
VFS_OUTBOX_NOMINATION_INTERFACE: Final[str] = "VfsOutboxNomination@1"
WAL_RECOVERY_PLAN_INTERFACE: Final[str] = "WalRecoveryPlan@1"
STALE_WRITER_REJECTION_INTERFACE: Final[str] = "StaleWriterRejection@1"
GITLINK_BINDING_INTERFACE: Final[str] = "GitlinkBinding@1"
CROSS_REPOSITORY_OWNERSHIP_INTERFACE: Final[str] = "CrossRepositoryOwnership@1"
WORLD_ROOT_RECEIPT_INTERFACE: Final[str] = "WorldRootReceipt@1"
TYPED_TERMINAL_INTERFACE: Final[str] = "TypedTerminal@1"

SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-refactor-world-root-adapter@1"
)
SEMANTIC_WORLD_ROOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-world-root@1"
)
GENERATION_CAS_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/generation-cas-binding@1"
)
VFS_OUTBOX_NOMINATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-outbox-nomination@1"
)
WAL_RECOVERY_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/wal-recovery-plan@1"
)
STALE_WRITER_REJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/stale-writer-rejection@1"
)
GITLINK_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gitlink-binding@1"
)
CROSS_REPOSITORY_OWNERSHIP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cross-repository-ownership@1"
)
WORLD_ROOT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/world-root-receipt@1"
)
TYPED_TERMINAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/world-root-typed-terminal@1"
)
NOMINATED_PERSIST_ROOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/nominated-persist-root@1"
)

WORLD_ROOT_CONTRACT_VERSION: Final[str] = "1"

ADAPTER_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
ADAPTER_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
ADAPTER_CAN_CREATE_AUTHORITY: Final[bool] = False
ADAPTER_OWNS_VFS: Final[bool] = False
ADAPTER_OWNS_CAS: Final[bool] = False
ADAPTER_OWNS_OUTBOX: Final[bool] = False
ADAPTER_OWNS_WAL: Final[bool] = False
ADAPTER_WRITES_REPOSITORY: Final[bool] = False
KIT_OWNS_VFS: Final[bool] = True
KIT_OWNS_CAS: Final[bool] = True
KIT_OWNS_OUTBOX: Final[bool] = True
KIT_OWNS_WAL_RECOVERY: Final[bool] = True
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
ADAPTER_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
NETWORK_DENIED: Final[bool] = True
NETWORK_DENY: Final[str] = "deny"
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
NEGATIVE_EVIDENCE_RETAINED: Final[bool] = True
STALE_WRITER_REJECTED: Final[bool] = True
ROOT_CONFLICT_OVERWRITE_FORBIDDEN: Final[bool] = True
GENERATION_CAS_REQUIRED: Final[bool] = True
CRASH_RECOVERY_WITHOUT_OVERWRITE: Final[bool] = True
NESTED_WRITES_REQUIRE_EXPLICIT_GITLINK: Final[bool] = True
NESTED_WRITES_REQUIRE_ACCELERATOR_TASK_OWNER: Final[bool] = True
USES_CURRENT_LEASE_FENCE: Final[bool] = True
USES_CURRENT_WORKTREE: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_GITLINKS: Final[int] = 64
MAX_ROOT_GENERATION: Final[int] = 1_000_000_000

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

PERSIST_KINDS: Final[tuple[str, ...]] = (
    "packet",
    "projection",
    "receipt",
    "transition",
)
DECLARED_PERSIST_KINDS: Final[frozenset[str]] = frozenset(PERSIST_KINDS)
PERSIST_KIND_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "packet": "packet_cids",
        "projection": "projection_cids",
        "receipt": "receipt_cids",
        "transition": "transition_cids",
    }
)
DECLARED_REPOSITORY_OWNERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ipfs_accelerate_py": "operational",
        "ipfs_datasets_py": "semantic",
        "ipfs_kit_py": "storage",
    }
)
EXISTING_ADAPTER_AUTHORITIES: Final[tuple[str, ...]] = (
    "datasets_semantic",
    "kit_storage",
    "kit_vfs",
    "kit_outbox",
    "kit_wal_recovery",
    "kit_generation_cas",
    "accelerator_supervisor",
    "accelerator_runtime",
    "spar_narrow",
)

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
        "WorldRootStore",
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
    "owns_vfs",
    "owns_cas",
    "writes_repository",
    "worker_self_approval",
)

FORBIDDEN_WORLD_ROOT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "WorldRootStore",
        "admit_by_overwrite",
        "authorize_completion",
        "authorize_transition",
        "overwrite_root_conflict",
        "stale_writer_commit",
        "implicit_gitlink_write",
        "undeclared_repository_write",
    }
)


class WorldRootAdapterError(ValueError):
    """Fail-closed violation of a SPAR-039 world-root contract."""


class AdapterStatus(str, Enum):
    NOMINATED_PERSIST = "nominated_persist"
    REJECTED_STALE_WRITER = "rejected_stale_writer"
    REJECTED_ROOT_CONFLICT = "rejected_root_conflict"
    RECOVERED = "recovered"
    TYPED_TERMINAL = "typed_terminal"


class TerminalKind(str, Enum):
    UNSUPPORTED = "unsupported"
    HUMAN_REVIEW = "human_review"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    UNDECLARED_REPOSITORY = "undeclared_repository"


class PersistKind(str, Enum):
    PACKET = "packet"
    PROJECTION = "projection"
    RECEIPT = "receipt"
    TRANSITION = "transition"


DECLARED_ADAPTER_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in AdapterStatus
)
DECLARED_TERMINAL_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in TerminalKind
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise WorldRootAdapterError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise WorldRootAdapterError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise WorldRootAdapterError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise WorldRootAdapterError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise WorldRootAdapterError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise WorldRootAdapterError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise WorldRootAdapterError(f"{name} must be a boolean")
    return value


def _int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is bool or type(value) is not int:
        raise WorldRootAdapterError(f"{name} must be an integer")
    if value < minimum:
        raise WorldRootAdapterError(f"{name} is out of range")
    if maximum is not None and value > maximum:
        raise WorldRootAdapterError(f"{name} is out of range")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise WorldRootAdapterError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise WorldRootAdapterError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise WorldRootAdapterError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise WorldRootAdapterError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise WorldRootAdapterError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise WorldRootAdapterError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise WorldRootAdapterError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise WorldRootAdapterError(f"{name} does not verify")


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise WorldRootAdapterError(f"unknown {name}: {text}") from exc


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
    raise WorldRootAdapterError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise WorldRootAdapterError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def _cids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, (), []):
        ordered: tuple[str, ...] = ()
    elif not isinstance(values, (list, tuple)):
        raise WorldRootAdapterError(f"{name} must be a list")
    else:
        ordered = tuple(sorted(_cid(item, name) for item in values))
    if len(ordered) != len(set(ordered)):
        raise WorldRootAdapterError(f"{name} must not contain duplicates")
    if required and not ordered:
        raise WorldRootAdapterError(f"{name} must not be empty")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise WorldRootAdapterError(f"{name} exceeds maximum length")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise WorldRootAdapterError(f"{name} exceeds path bound")
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
        raise WorldRootAdapterError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise WorldRootAdapterError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, (), []):
        ordered: tuple[str, ...] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise WorldRootAdapterError(f"{name} must be a list of exact paths")
    else:
        seen: set[str] = set()
        collected: list[str] = []
        for item in values:
            path = _exact_path(item, name)
            if path not in seen:
                seen.add(path)
                collected.append(path)
        ordered = tuple(collected)
    if required and not ordered:
        raise WorldRootAdapterError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise WorldRootAdapterError(f"{name} exceeds path bound")
    return ordered


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise WorldRootAdapterError(f"{name} cannot claim {flag}")


def _reject_non_admitting(payload: Mapping[str, Any], name: str) -> None:
    present = _NON_ADMITTING_EVIDENCE & set(payload)
    if present:
        raise WorldRootAdapterError(
            f"{name} cannot admit a world root from {sorted(present)}"
        )


def world_root_adapter_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def world_root_adapter_descriptor() -> dict[str, Any]:
    return {
        "schema": SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_SCHEMA,
        "interface": SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "analyzer_id": ANALYZER_ID,
        "predecessor_task_ids": list(PREDECESSOR_TASK_IDS),
        "authority_owner": AUTHORITY_OWNER,
        "vfs_authority_owner": VFS_AUTHORITY_OWNER,
        "nomination_only": True,
        "kit_owns_vfs": True,
        "kit_owns_cas": True,
        "writes_repository": False,
        "network": NETWORK_DENY,
        "stale_writer_rejected": True,
        "root_conflict_overwrite_forbidden": True,
        "crash_recovery_without_overwrite": True,
    }


def _path_under_gitlink(path: str, gitlink_path: str) -> bool:
    if path == gitlink_path:
        return True
    return path.startswith(gitlink_path + "/")


def _parse_terminal(value: Any) -> TypedTerminal | None:
    if value in (None, "", {}):
        return None
    if isinstance(value, TypedTerminal):
        return value
    return TypedTerminal.from_mapping(value)


@dataclass(frozen=True, slots=True)
class TypedTerminal:
    """Unsupported required behavior is a typed terminal, never success."""

    kind: str
    reason: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "kind",
            "reason",
            "terminal_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, TerminalKind, "kind"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TYPED_TERMINAL_SCHEMA,
            "interface": TYPED_TERMINAL_INTERFACE,
            "kind": self.kind,
            "reason": self.reason,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def terminal_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["terminal_cid"] = self.terminal_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypedTerminal":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("terminal_cid")
        if payload.pop("schema") != TYPED_TERMINAL_SCHEMA:
            raise WorldRootAdapterError("unsupported TypedTerminal schema")
        if payload.pop("interface") != TYPED_TERMINAL_INTERFACE:
            raise WorldRootAdapterError("unsupported TypedTerminal interface")
        result = cls(**payload)
        _verify_cid(claimed, result.terminal_cid, "TypedTerminal terminal_cid")
        return result

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "TypedTerminal") -> "TypedTerminal":
        if isinstance(data, TypedTerminal):
            return data
        if "terminal_cid" in data:
            return cls.from_dict(data)
        payload = _mapping(data, "terminal")
        return cls(kind=payload.get("kind"), reason=payload.get("reason"))


@dataclass(frozen=True, slots=True)
class GitlinkBinding:
    """Explicit nested-repository gitlink. Implicit gitlinks cannot write."""

    gitlink_path: str
    nested_repository_id: str
    nested_tree_id: str
    owner_repository: str
    explicit: bool

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "gitlink_path",
            "nested_repository_id",
            "nested_tree_id",
            "owner_repository",
            "explicit",
            "binding_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "gitlink_path", _exact_path(self.gitlink_path, "gitlink_path")
        )
        object.__setattr__(
            self,
            "nested_repository_id",
            _text(self.nested_repository_id, "nested_repository_id"),
        )
        object.__setattr__(self, "nested_tree_id", _tree_id(self.nested_tree_id))
        object.__setattr__(
            self, "owner_repository", _text(self.owner_repository, "owner_repository")
        )
        object.__setattr__(self, "explicit", _bool(self.explicit, "explicit"))
        if self.nested_repository_id not in DECLARED_REPOSITORY_OWNERS:
            raise WorldRootAdapterError(
                "gitlink nested_repository_id is not a declared repository"
            )
        expected_owner = self.nested_repository_id
        if self.owner_repository != expected_owner:
            raise WorldRootAdapterError(
                "gitlink owner_repository does not match declared nested ownership"
            )

    def covers(self, path: str) -> bool:
        return _path_under_gitlink(path, self.gitlink_path)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": GITLINK_BINDING_SCHEMA,
            "interface": GITLINK_BINDING_INTERFACE,
            "gitlink_path": self.gitlink_path,
            "nested_repository_id": self.nested_repository_id,
            "nested_tree_id": self.nested_tree_id,
            "owner_repository": self.owner_repository,
            "explicit": self.explicit,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def binding_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["binding_cid"] = self.binding_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GitlinkBinding":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("binding_cid")
        if payload.pop("schema") != GITLINK_BINDING_SCHEMA:
            raise WorldRootAdapterError("unsupported GitlinkBinding schema")
        if payload.pop("interface") != GITLINK_BINDING_INTERFACE:
            raise WorldRootAdapterError("unsupported GitlinkBinding interface")
        result = cls(**payload)
        _verify_cid(claimed, result.binding_cid, "GitlinkBinding binding_cid")
        return result

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "GitlinkBinding") -> "GitlinkBinding":
        if isinstance(data, GitlinkBinding):
            return data
        if "binding_cid" in data:
            return cls.from_dict(data)
        payload = _mapping(data, "gitlink")
        return cls(
            gitlink_path=payload.get("gitlink_path"),
            nested_repository_id=payload.get("nested_repository_id"),
            nested_tree_id=payload.get("nested_tree_id"),
            owner_repository=payload.get("owner_repository"),
            explicit=payload.get("explicit"),
        )


def _parse_gitlinks(values: Any) -> tuple[GitlinkBinding, ...]:
    if values in (None, (), []):
        return ()
    if not isinstance(values, (list, tuple)):
        raise WorldRootAdapterError("gitlinks must be a list")
    bindings = tuple(GitlinkBinding.from_mapping(item) for item in values)
    if len(bindings) > MAX_GITLINKS:
        raise WorldRootAdapterError("gitlinks exceeds maximum length")
    paths = [item.gitlink_path for item in bindings]
    if len(paths) != len(set(paths)):
        raise WorldRootAdapterError("gitlinks must not contain duplicate paths")
    return bindings


@dataclass(frozen=True, slots=True)
class CrossRepositoryOwnership:
    """Explicit ownership for nested writes. Undeclared repositories fail."""

    task_owner: str
    gitlinks: Sequence[GitlinkBinding]
    nested_write_paths: Sequence[str]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "task_owner",
            "gitlinks",
            "nested_write_paths",
            "ownership_cid",
            "nested_writes_require_explicit_gitlink",
            "nested_writes_require_accelerator_task_owner",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_owner", _text(self.task_owner, "task_owner"))
        gitlinks = tuple(
            item if isinstance(item, GitlinkBinding) else GitlinkBinding.from_mapping(item)
            for item in self.gitlinks
        )
        object.__setattr__(self, "gitlinks", gitlinks)
        object.__setattr__(
            self,
            "nested_write_paths",
            _exact_paths(list(self.nested_write_paths), "nested_write_paths"),
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": CROSS_REPOSITORY_OWNERSHIP_SCHEMA,
            "interface": CROSS_REPOSITORY_OWNERSHIP_INTERFACE,
            "task_owner": self.task_owner,
            "gitlinks": [item.to_dict() for item in self.gitlinks],
            "nested_write_paths": list(self.nested_write_paths),
            "nested_writes_require_explicit_gitlink": True,
            "nested_writes_require_accelerator_task_owner": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def ownership_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["ownership_cid"] = self.ownership_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CrossRepositoryOwnership":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("ownership_cid")
        if payload.pop("schema") != CROSS_REPOSITORY_OWNERSHIP_SCHEMA:
            raise WorldRootAdapterError("unsupported CrossRepositoryOwnership schema")
        if payload.pop("interface") != CROSS_REPOSITORY_OWNERSHIP_INTERFACE:
            raise WorldRootAdapterError(
                "unsupported CrossRepositoryOwnership interface"
            )
        if payload.pop("nested_writes_require_explicit_gitlink") is not True:
            raise WorldRootAdapterError("nested writes require an explicit gitlink")
        if payload.pop("nested_writes_require_accelerator_task_owner") is not True:
            raise WorldRootAdapterError(
                "nested writes require ipfs_accelerate_py task ownership"
            )
        payload["gitlinks"] = _parse_gitlinks(payload["gitlinks"])
        result = cls(**payload)
        _verify_cid(
            claimed, result.ownership_cid, "CrossRepositoryOwnership ownership_cid"
        )
        return result


def _undeclared_repository_terminal(values: Any) -> TypedTerminal | None:
    if values in (None, (), []):
        return None
    if not isinstance(values, (list, tuple)):
        raise WorldRootAdapterError("gitlinks must be a list")
    for item in values:
        if isinstance(item, GitlinkBinding):
            repository_id = item.nested_repository_id
        else:
            payload = _mapping(item, "gitlink")
            repository_id = _text(
                payload.get("nested_repository_id"), "nested_repository_id"
            )
        if repository_id not in DECLARED_REPOSITORY_OWNERS:
            return TypedTerminal(
                kind=TerminalKind.UNDECLARED_REPOSITORY.value,
                reason="cross-repository integration requires a declared repository owner",
            )
    return None


def _nested_write_terminal(
    ownership: CrossRepositoryOwnership,
) -> TypedTerminal | None:
    if not ownership.nested_write_paths:
        return None
    if ownership.task_owner != ACCELERATOR_TASK_OWNER:
        return TypedTerminal(
            kind=TerminalKind.UNSUPPORTED.value,
            reason=(
                "nested repository writes require ipfs_accelerate_py task ownership "
                "and explicit gitlink integration"
            ),
        )
    for path in ownership.nested_write_paths:
        covering = [item for item in ownership.gitlinks if item.covers(path)]
        if not covering:
            return TypedTerminal(
                kind=TerminalKind.UNSUPPORTED.value,
                reason="nested write path is not covered by an explicit gitlink",
            )
        if any(not item.explicit for item in covering):
            return TypedTerminal(
                kind=TerminalKind.UNSUPPORTED.value,
                reason="nested repository writes require explicit gitlink integration",
            )
    return None


@dataclass(frozen=True, slots=True)
class GenerationCasBinding:
    """One expected-generation CAS check against the current kit root."""

    expected_root_generation: int
    current_root_generation: int
    pre_world_root_cid: str
    current_world_root_cid: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "expected_root_generation",
            "current_root_generation",
            "pre_world_root_cid",
            "current_world_root_cid",
            "cas_matches",
            "binding_cid",
            "kit_owns_cas",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_root_generation",
            _int(
                self.expected_root_generation,
                "expected_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "current_root_generation",
            _int(
                self.current_root_generation,
                "current_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self, "pre_world_root_cid", _cid(self.pre_world_root_cid, "pre_world_root_cid")
        )
        object.__setattr__(
            self,
            "current_world_root_cid",
            _cid(self.current_world_root_cid, "current_world_root_cid"),
        )

    @property
    def generation_matches(self) -> bool:
        return self.expected_root_generation == self.current_root_generation

    @property
    def root_matches(self) -> bool:
        return self.pre_world_root_cid == self.current_world_root_cid

    @property
    def cas_matches(self) -> bool:
        return self.generation_matches and self.root_matches

    @property
    def stale_writer(self) -> bool:
        return not self.generation_matches

    @property
    def root_conflict(self) -> bool:
        return self.generation_matches and not self.root_matches

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": GENERATION_CAS_BINDING_SCHEMA,
            "interface": GENERATION_CAS_BINDING_INTERFACE,
            "expected_root_generation": self.expected_root_generation,
            "current_root_generation": self.current_root_generation,
            "pre_world_root_cid": self.pre_world_root_cid,
            "current_world_root_cid": self.current_world_root_cid,
            "cas_matches": self.cas_matches,
            "kit_owns_cas": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def binding_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["binding_cid"] = self.binding_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationCasBinding":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("binding_cid")
        if payload.pop("schema") != GENERATION_CAS_BINDING_SCHEMA:
            raise WorldRootAdapterError("unsupported GenerationCasBinding schema")
        if payload.pop("interface") != GENERATION_CAS_BINDING_INTERFACE:
            raise WorldRootAdapterError("unsupported GenerationCasBinding interface")
        if payload.pop("kit_owns_cas") is not True:
            raise WorldRootAdapterError("kit remains the generation CAS authority")
        cas_matches = payload.pop("cas_matches")
        result = cls(**payload)
        if cas_matches is not result.cas_matches:
            raise WorldRootAdapterError("cas_matches does not verify")
        _verify_cid(claimed, result.binding_cid, "GenerationCasBinding binding_cid")
        return result


@dataclass(frozen=True, slots=True)
class VfsOutboxNomination:
    """Nominated kit-owned outbox persist. SPAR-039 never mutates VFS."""

    artifact_kind: str
    artifact_cid: str
    tree_id: str
    kit_owns_vfs: bool = True
    kit_owns_outbox: bool = True
    mutated: bool = False
    writes_repository: bool = False

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "artifact_kind",
            "artifact_cid",
            "tree_id",
            "kit_owns_vfs",
            "kit_owns_outbox",
            "mutated",
            "writes_repository",
            "nomination_cid",
            "can_authorize_transition",
            "can_create_authority",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_kind", _enum(self.artifact_kind, PersistKind, "artifact_kind")
        )
        object.__setattr__(self, "artifact_cid", _cid(self.artifact_cid, "artifact_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        if not _bool(self.kit_owns_vfs, "kit_owns_vfs"):
            raise WorldRootAdapterError("kit remains the VFS authority")
        if not _bool(self.kit_owns_outbox, "kit_owns_outbox"):
            raise WorldRootAdapterError("kit remains the outbox authority")
        if _bool(self.mutated, "mutated"):
            raise WorldRootAdapterError("adapter cannot mutate VFS")
        if _bool(self.writes_repository, "writes_repository"):
            raise WorldRootAdapterError("adapter cannot write the repository")
        object.__setattr__(self, "kit_owns_vfs", True)
        object.__setattr__(self, "kit_owns_outbox", True)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "writes_repository", False)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": VFS_OUTBOX_NOMINATION_SCHEMA,
            "interface": VFS_OUTBOX_NOMINATION_INTERFACE,
            "artifact_kind": self.artifact_kind,
            "artifact_cid": self.artifact_cid,
            "tree_id": self.tree_id,
            "kit_owns_vfs": True,
            "kit_owns_outbox": True,
            "mutated": False,
            "writes_repository": False,
            "can_authorize_transition": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def nomination_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["nomination_cid"] = self.nomination_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VfsOutboxNomination":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("nomination_cid")
        if payload.pop("schema") != VFS_OUTBOX_NOMINATION_SCHEMA:
            raise WorldRootAdapterError("unsupported VfsOutboxNomination schema")
        if payload.pop("interface") != VFS_OUTBOX_NOMINATION_INTERFACE:
            raise WorldRootAdapterError("unsupported VfsOutboxNomination interface")
        if payload.pop("can_authorize_transition") is not False:
            raise WorldRootAdapterError("outbox nomination cannot authorize a transition")
        if payload.pop("can_create_authority") is not False:
            raise WorldRootAdapterError("outbox nomination cannot create authority")
        result = cls(**payload)
        _verify_cid(claimed, result.nomination_cid, "VfsOutboxNomination nomination_cid")
        return result


def _outbox_nominations(
    *,
    tree_id: str,
    packet_cids: Sequence[str],
    projection_cids: Sequence[str],
    receipt_cids: Sequence[str],
    transition_cids: Sequence[str],
) -> tuple[VfsOutboxNomination, ...]:
    grouped = (
        (PersistKind.PACKET.value, packet_cids),
        (PersistKind.PROJECTION.value, projection_cids),
        (PersistKind.RECEIPT.value, receipt_cids),
        (PersistKind.TRANSITION.value, transition_cids),
    )
    nominations: list[VfsOutboxNomination] = []
    for kind, cids in grouped:
        for artifact_cid in cids:
            nominations.append(
                VfsOutboxNomination(
                    artifact_kind=kind,
                    artifact_cid=artifact_cid,
                    tree_id=tree_id,
                )
            )
    nominations.sort(key=lambda item: (item.artifact_kind, item.artifact_cid))
    if len(nominations) > MAX_MEMBERS:
        raise WorldRootAdapterError("outbox nominations exceed maximum length")
    return tuple(nominations)


@dataclass(frozen=True, slots=True)
class WalRecoveryPlan:
    """Restore the last committed kit root. Pending outbox is not applied."""

    last_committed_world_root_cid: str
    last_committed_root_generation: int
    recovered_world_root_cid: str
    recovered_root_generation: int
    pending_outbox_cids: Sequence[str]
    overwrite_prevented: bool = True
    kit_owns_wal: bool = True

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "last_committed_world_root_cid",
            "last_committed_root_generation",
            "recovered_world_root_cid",
            "recovered_root_generation",
            "pending_outbox_cids",
            "overwrite_prevented",
            "kit_owns_wal",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "last_committed_world_root_cid",
            _cid(self.last_committed_world_root_cid, "last_committed_world_root_cid"),
        )
        object.__setattr__(
            self,
            "last_committed_root_generation",
            _int(
                self.last_committed_root_generation,
                "last_committed_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "recovered_world_root_cid",
            _cid(self.recovered_world_root_cid, "recovered_world_root_cid"),
        )
        object.__setattr__(
            self,
            "recovered_root_generation",
            _int(
                self.recovered_root_generation,
                "recovered_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "pending_outbox_cids",
            _cids(list(self.pending_outbox_cids), "pending_outbox_cids"),
        )
        if not _bool(self.overwrite_prevented, "overwrite_prevented"):
            raise WorldRootAdapterError("recovery cannot overwrite committed roots")
        if not _bool(self.kit_owns_wal, "kit_owns_wal"):
            raise WorldRootAdapterError("kit remains the WAL/recovery authority")
        if self.recovered_world_root_cid != self.last_committed_world_root_cid:
            raise WorldRootAdapterError("recovery must restore the last committed root")
        if self.recovered_root_generation != self.last_committed_root_generation:
            raise WorldRootAdapterError(
                "recovery cannot advance or rewind past the last committed generation"
            )
        object.__setattr__(self, "overwrite_prevented", True)
        object.__setattr__(self, "kit_owns_wal", True)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": WAL_RECOVERY_PLAN_SCHEMA,
            "interface": WAL_RECOVERY_PLAN_INTERFACE,
            "last_committed_world_root_cid": self.last_committed_world_root_cid,
            "last_committed_root_generation": self.last_committed_root_generation,
            "recovered_world_root_cid": self.recovered_world_root_cid,
            "recovered_root_generation": self.recovered_root_generation,
            "pending_outbox_cids": list(self.pending_outbox_cids),
            "overwrite_prevented": True,
            "kit_owns_wal": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "WalRecoveryPlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != WAL_RECOVERY_PLAN_SCHEMA:
            raise WorldRootAdapterError("unsupported WalRecoveryPlan schema")
        if payload.pop("interface") != WAL_RECOVERY_PLAN_INTERFACE:
            raise WorldRootAdapterError("unsupported WalRecoveryPlan interface")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "WalRecoveryPlan plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class StaleWriterRejection:
    """Reject a writer whose expected generation is not current."""

    attempted_expected_root_generation: int
    current_root_generation: int
    attempted_pre_world_root_cid: str
    current_world_root_cid: str
    overwrite_prevented: bool = True

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "attempted_expected_root_generation",
            "current_root_generation",
            "attempted_pre_world_root_cid",
            "current_world_root_cid",
            "overwrite_prevented",
            "rejection_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempted_expected_root_generation",
            _int(
                self.attempted_expected_root_generation,
                "attempted_expected_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "current_root_generation",
            _int(
                self.current_root_generation,
                "current_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "attempted_pre_world_root_cid",
            _cid(self.attempted_pre_world_root_cid, "attempted_pre_world_root_cid"),
        )
        object.__setattr__(
            self,
            "current_world_root_cid",
            _cid(self.current_world_root_cid, "current_world_root_cid"),
        )
        if not _bool(self.overwrite_prevented, "overwrite_prevented"):
            raise WorldRootAdapterError("stale writer must not overwrite")
        if self.attempted_expected_root_generation == self.current_root_generation:
            raise WorldRootAdapterError(
                "stale-writer rejection requires a generation mismatch"
            )
        object.__setattr__(self, "overwrite_prevented", True)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": STALE_WRITER_REJECTION_SCHEMA,
            "interface": STALE_WRITER_REJECTION_INTERFACE,
            "attempted_expected_root_generation": self.attempted_expected_root_generation,
            "current_root_generation": self.current_root_generation,
            "attempted_pre_world_root_cid": self.attempted_pre_world_root_cid,
            "current_world_root_cid": self.current_world_root_cid,
            "overwrite_prevented": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def rejection_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["rejection_cid"] = self.rejection_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StaleWriterRejection":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("rejection_cid")
        if payload.pop("schema") != STALE_WRITER_REJECTION_SCHEMA:
            raise WorldRootAdapterError("unsupported StaleWriterRejection schema")
        if payload.pop("interface") != STALE_WRITER_REJECTION_INTERFACE:
            raise WorldRootAdapterError("unsupported StaleWriterRejection interface")
        result = cls(**payload)
        _verify_cid(claimed, result.rejection_cid, "StaleWriterRejection rejection_cid")
        return result


def _nominate_post_world_root_cid(
    *,
    tree_id: str,
    pre_world_root_cid: str,
    expected_root_generation: int,
    packet_cids: Sequence[str],
    projection_cids: Sequence[str],
    receipt_cids: Sequence[str],
    transition_cids: Sequence[str],
) -> str:
    payload = {
        "schema": NOMINATED_PERSIST_ROOT_SCHEMA,
        "tree_id": tree_id,
        "pre_world_root_cid": pre_world_root_cid,
        "expected_root_generation": expected_root_generation,
        "resulting_root_generation": expected_root_generation + 1,
        "packet_cids": list(packet_cids),
        "projection_cids": list(projection_cids),
        "receipt_cids": list(receipt_cids),
        "transition_cids": list(transition_cids),
        "kit_owns_vfs": True,
        "kit_owns_cas": True,
        "writes_repository": False,
    }
    _require_dag_json(payload, "nominated persist root")
    return cid_for_dag_json(payload)


@dataclass(frozen=True, slots=True)
class SemanticWorldRoot:
    """Nominated pre/post world-root pair bound to generation CAS."""

    tree_id: str
    pre_world_root_cid: str
    post_world_root_cid: str
    expected_root_generation: int
    resulting_root_generation: int
    packet_cids: Sequence[str] = ()
    projection_cids: Sequence[str] = ()
    receipt_cids: Sequence[str] = ()
    transition_cids: Sequence[str] = ()

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "pre_world_root_cid",
            "post_world_root_cid",
            "expected_root_generation",
            "resulting_root_generation",
            "packet_cids",
            "projection_cids",
            "receipt_cids",
            "transition_cids",
            "world_root_cid",
            "kit_owns_vfs",
            "kit_owns_cas",
            "writes_repository",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self, "pre_world_root_cid", _cid(self.pre_world_root_cid, "pre_world_root_cid")
        )
        object.__setattr__(
            self,
            "post_world_root_cid",
            _cid(self.post_world_root_cid, "post_world_root_cid"),
        )
        object.__setattr__(
            self,
            "expected_root_generation",
            _int(
                self.expected_root_generation,
                "expected_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "resulting_root_generation",
            _int(
                self.resulting_root_generation,
                "resulting_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(self, "packet_cids", _cids(list(self.packet_cids), "packet_cids"))
        object.__setattr__(
            self, "projection_cids", _cids(list(self.projection_cids), "projection_cids")
        )
        object.__setattr__(
            self, "receipt_cids", _cids(list(self.receipt_cids), "receipt_cids")
        )
        object.__setattr__(
            self, "transition_cids", _cids(list(self.transition_cids), "transition_cids")
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SEMANTIC_WORLD_ROOT_SCHEMA,
            "interface": SEMANTIC_WORLD_ROOT_INTERFACE,
            "tree_id": self.tree_id,
            "pre_world_root_cid": self.pre_world_root_cid,
            "post_world_root_cid": self.post_world_root_cid,
            "expected_root_generation": self.expected_root_generation,
            "resulting_root_generation": self.resulting_root_generation,
            "packet_cids": list(self.packet_cids),
            "projection_cids": list(self.projection_cids),
            "receipt_cids": list(self.receipt_cids),
            "transition_cids": list(self.transition_cids),
            "kit_owns_vfs": True,
            "kit_owns_cas": True,
            "writes_repository": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def world_root_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["world_root_cid"] = self.world_root_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticWorldRoot":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("world_root_cid")
        if payload.pop("schema") != SEMANTIC_WORLD_ROOT_SCHEMA:
            raise WorldRootAdapterError("unsupported SemanticWorldRoot schema")
        if payload.pop("interface") != SEMANTIC_WORLD_ROOT_INTERFACE:
            raise WorldRootAdapterError("unsupported SemanticWorldRoot interface")
        if payload.pop("kit_owns_vfs") is not True:
            raise WorldRootAdapterError("kit remains the VFS authority")
        if payload.pop("kit_owns_cas") is not True:
            raise WorldRootAdapterError("kit remains the generation CAS authority")
        if payload.pop("writes_repository") is not False:
            raise WorldRootAdapterError("world root cannot write the repository")
        result = cls(**payload)
        _verify_cid(claimed, result.world_root_cid, "SemanticWorldRoot world_root_cid")
        return result


@dataclass(frozen=True, slots=True)
class WorldRootReceipt:
    """Nomination-only SPAR world-root integration receipt."""

    tree_id: str
    status: str
    world_root: SemanticWorldRoot
    cas_binding: GenerationCasBinding
    outbox: Sequence[VfsOutboxNomination] = ()
    ownership: CrossRepositoryOwnership | None = None
    recovery: WalRecoveryPlan | None = None
    stale_writer: StaleWriterRejection | None = None
    terminal: TypedTerminal | None = None
    analyzer_id: str = ANALYZER_ID
    worktree_id: str = ""
    lease_id: str = ""
    fence_id: str = ""
    negative_evidence_cids: Sequence[str] = ()

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "status",
            "world_root",
            "cas_binding",
            "outbox",
            "ownership",
            "recovery",
            "stale_writer",
            "terminal",
            "analyzer_id",
            "worktree_id",
            "lease_id",
            "fence_id",
            "negative_evidence_cids",
            "receipt_cid",
            "nominated",
            "accepted",
            "adapter_is_nomination_only",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "owns_vfs",
            "owns_cas",
            "writes_repository",
            "worker_self_approval",
            "overwrite_prevented",
            "kit_owns_vfs",
            "kit_owns_cas",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        world_root = self.world_root
        if not isinstance(world_root, SemanticWorldRoot):
            world_root = SemanticWorldRoot.from_dict(world_root)
        object.__setattr__(self, "world_root", world_root)
        if world_root.tree_id != self.tree_id:
            raise WorldRootAdapterError("receipt tree_id does not match world root")
        cas_binding = self.cas_binding
        if not isinstance(cas_binding, GenerationCasBinding):
            cas_binding = GenerationCasBinding.from_dict(cas_binding)
        object.__setattr__(self, "cas_binding", cas_binding)
        object.__setattr__(self, "status", _enum(self.status, AdapterStatus, "status"))
        object.__setattr__(self, "analyzer_id", _text(self.analyzer_id, "analyzer_id"))
        if self.analyzer_id != ANALYZER_ID:
            raise WorldRootAdapterError("analyzer_id must remain SPAR-039")
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", empty=True)
        )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id", empty=True))
        object.__setattr__(self, "fence_id", _text(self.fence_id, "fence_id", empty=True))
        outbox = tuple(
            item
            if isinstance(item, VfsOutboxNomination)
            else VfsOutboxNomination.from_dict(item)
            for item in self.outbox
        )
        for item in outbox:
            if item.tree_id != self.tree_id:
                raise WorldRootAdapterError("outbox tree_id does not match receipt")
        object.__setattr__(self, "outbox", outbox)
        ownership = self.ownership
        if ownership not in (None,):
            if not isinstance(ownership, CrossRepositoryOwnership):
                ownership = CrossRepositoryOwnership.from_dict(ownership)
        else:
            ownership = None
        object.__setattr__(self, "ownership", ownership)
        recovery = self.recovery
        if recovery not in (None,):
            if not isinstance(recovery, WalRecoveryPlan):
                recovery = WalRecoveryPlan.from_dict(recovery)
        else:
            recovery = None
        object.__setattr__(self, "recovery", recovery)
        stale_writer = self.stale_writer
        if stale_writer not in (None,):
            if not isinstance(stale_writer, StaleWriterRejection):
                stale_writer = StaleWriterRejection.from_dict(stale_writer)
        else:
            stale_writer = None
        object.__setattr__(self, "stale_writer", stale_writer)
        terminal = self.terminal
        if terminal not in (None,):
            if not isinstance(terminal, TypedTerminal):
                terminal = TypedTerminal.from_mapping(terminal)
        else:
            terminal = None
        object.__setattr__(self, "terminal", terminal)
        object.__setattr__(
            self,
            "negative_evidence_cids",
            _cids(list(self.negative_evidence_cids), "negative_evidence_cids"),
        )
        self._assert_status_invariants()

    def _assert_status_invariants(self) -> None:
        if self.status == AdapterStatus.NOMINATED_PERSIST.value:
            if self.terminal is not None:
                raise WorldRootAdapterError("nominated persist cannot carry a typed terminal")
            if self.recovery is not None:
                raise WorldRootAdapterError("nominated persist cannot carry recovery")
            if self.stale_writer is not None:
                raise WorldRootAdapterError("nominated persist cannot carry a stale writer")
            if not self.cas_binding.cas_matches:
                raise WorldRootAdapterError("nominated persist requires matching generation CAS")
            if (
                self.world_root.resulting_root_generation
                != self.world_root.expected_root_generation + 1
            ):
                raise WorldRootAdapterError(
                    "nominated persist must advance resulting_root_generation by one"
                )
            if not (
                self.world_root.packet_cids
                or self.world_root.projection_cids
                or self.world_root.receipt_cids
                or self.world_root.transition_cids
            ):
                raise WorldRootAdapterError(
                    "nominated persist requires packet, projection, receipt, or transition artifacts"
                )
        if self.status == AdapterStatus.REJECTED_STALE_WRITER.value:
            if self.stale_writer is None:
                raise WorldRootAdapterError("stale-writer status requires a rejection")
            if self.world_root.post_world_root_cid != self.cas_binding.current_world_root_cid:
                raise WorldRootAdapterError("stale writer must not overwrite the current root")
            if (
                self.world_root.resulting_root_generation
                != self.cas_binding.current_root_generation
            ):
                raise WorldRootAdapterError("stale writer must not advance the root generation")
        if self.status == AdapterStatus.REJECTED_ROOT_CONFLICT.value:
            if self.cas_binding.root_conflict is not True:
                raise WorldRootAdapterError("root-conflict status requires a CID mismatch")
            if self.world_root.post_world_root_cid != self.cas_binding.current_world_root_cid:
                raise WorldRootAdapterError("root conflict must not overwrite the current root")
            if (
                self.world_root.resulting_root_generation
                != self.cas_binding.current_root_generation
            ):
                raise WorldRootAdapterError("root conflict must not advance the root generation")
        if self.status == AdapterStatus.RECOVERED.value:
            if self.recovery is None:
                raise WorldRootAdapterError("recovered status requires a WAL recovery plan")
            if self.world_root.post_world_root_cid != self.recovery.recovered_world_root_cid:
                raise WorldRootAdapterError("recovery must restore the last committed root")
            if (
                self.world_root.resulting_root_generation
                != self.recovery.recovered_root_generation
            ):
                raise WorldRootAdapterError("recovery must not overwrite committed generation")
        if self.status == AdapterStatus.TYPED_TERMINAL.value:
            if self.terminal is None:
                raise WorldRootAdapterError("typed terminal status requires a typed terminal")
        if (
            self.status != AdapterStatus.TYPED_TERMINAL.value
            and self.terminal is not None
        ):
            raise WorldRootAdapterError("non-terminal status cannot carry a typed terminal")

    @property
    def nominated(self) -> bool:
        return self.status == AdapterStatus.NOMINATED_PERSIST.value

    @property
    def accepted(self) -> bool:
        return False

    @property
    def overwrite_prevented(self) -> bool:
        return self.status != AdapterStatus.NOMINATED_PERSIST.value

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        overwrite_prevented = self.status != AdapterStatus.NOMINATED_PERSIST.value
        payload = {
            "schema": WORLD_ROOT_RECEIPT_SCHEMA,
            "interface": WORLD_ROOT_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "status": self.status,
            "world_root": self.world_root.to_dict(),
            "cas_binding": self.cas_binding.to_dict(),
            "outbox": [item.to_dict() for item in self.outbox],
            "ownership": None if self.ownership is None else self.ownership.to_dict(),
            "recovery": None if self.recovery is None else self.recovery.to_dict(),
            "stale_writer": (
                None if self.stale_writer is None else self.stale_writer.to_dict()
            ),
            "terminal": None if self.terminal is None else self.terminal.to_dict(),
            "analyzer_id": ANALYZER_ID,
            "worktree_id": self.worktree_id,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "nominated": self.nominated,
            "accepted": False,
            "adapter_is_nomination_only": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "owns_vfs": False,
            "owns_cas": False,
            "writes_repository": False,
            "worker_self_approval": False,
            "overwrite_prevented": overwrite_prevented,
            "kit_owns_vfs": True,
            "kit_owns_cas": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldRootReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != WORLD_ROOT_RECEIPT_SCHEMA:
            raise WorldRootAdapterError("unsupported WorldRootReceipt schema")
        if payload.pop("interface") != WORLD_ROOT_RECEIPT_INTERFACE:
            raise WorldRootAdapterError("unsupported WorldRootReceipt interface")
        if payload.pop("accepted") is not False:
            raise WorldRootAdapterError("workers cannot self-approve a world root")
        if payload.pop("adapter_is_nomination_only") is not True:
            raise WorldRootAdapterError("adapter must remain nomination_only")
        if payload.pop("kit_owns_vfs") is not True:
            raise WorldRootAdapterError("kit remains the VFS authority")
        if payload.pop("kit_owns_cas") is not True:
            raise WorldRootAdapterError("kit remains the generation CAS authority")
        nominated = payload.pop("nominated")
        payload.pop("overwrite_prevented")
        _pop_authority_flags(payload, "WorldRootReceipt")
        payload["world_root"] = SemanticWorldRoot.from_dict(payload["world_root"])
        payload["cas_binding"] = GenerationCasBinding.from_dict(payload["cas_binding"])
        payload["outbox"] = tuple(
            VfsOutboxNomination.from_dict(item) for item in payload["outbox"]
        )
        if payload["ownership"] is not None:
            payload["ownership"] = CrossRepositoryOwnership.from_dict(
                payload["ownership"]
            )
        if payload["recovery"] is not None:
            payload["recovery"] = WalRecoveryPlan.from_dict(payload["recovery"])
        if payload["stale_writer"] is not None:
            payload["stale_writer"] = StaleWriterRejection.from_dict(
                payload["stale_writer"]
            )
        if payload["terminal"] is not None:
            payload["terminal"] = TypedTerminal.from_dict(payload["terminal"])
        result = cls(**payload)
        if nominated is not result.nominated:
            raise WorldRootAdapterError("nominated flag does not match status")
        _verify_cid(claimed, result.receipt_cid, "WorldRootReceipt receipt_cid")
        return result


def _artifact_cids(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        "packet_cids": _cids(list(payload.get("packet_cids") or ()), "packet_cids"),
        "projection_cids": _cids(
            list(payload.get("projection_cids") or ()), "projection_cids"
        ),
        "receipt_cids": _cids(list(payload.get("receipt_cids") or ()), "receipt_cids"),
        "transition_cids": _cids(
            list(payload.get("transition_cids") or ()), "transition_cids"
        ),
    }


def _unchanged_world_root(
    *,
    tree_id: str,
    pre_world_root_cid: str,
    post_world_root_cid: str,
    expected_root_generation: int,
    resulting_root_generation: int,
    artifacts: Mapping[str, Sequence[str]],
) -> SemanticWorldRoot:
    return SemanticWorldRoot(
        tree_id=tree_id,
        pre_world_root_cid=pre_world_root_cid,
        post_world_root_cid=post_world_root_cid,
        expected_root_generation=expected_root_generation,
        resulting_root_generation=resulting_root_generation,
        packet_cids=artifacts["packet_cids"],
        projection_cids=artifacts["projection_cids"],
        receipt_cids=artifacts["receipt_cids"],
        transition_cids=artifacts["transition_cids"],
    )


def integrate_world_root(
    evidence: Mapping[str, Any],
    *,
    mutate: bool = False,
) -> WorldRootReceipt:
    """Nominate kit persist/recovery for one world-root CAS step."""

    if mutate is True:
        raise WorldRootAdapterError("adapter cannot mutate; dry-run never mutates")
    payload = _mapping(evidence, "world-root evidence")
    _reject_non_admitting(payload, "world-root evidence")
    if payload.get("mutate") is True:
        raise WorldRootAdapterError("adapter cannot mutate; dry-run never mutates")
    network = payload.get("network", NETWORK_DENY)
    if network != NETWORK_DENY:
        raise WorldRootAdapterError("network is denied for world-root integration")
    tree_id = _tree_id(payload.get("tree_id"))
    artifacts = _artifact_cids(payload)
    pre_world_root_cid = _cid(payload.get("pre_world_root_cid"), "pre_world_root_cid")
    current_world_root_cid = _cid(
        payload.get("current_world_root_cid") or payload.get("pre_world_root_cid"),
        "current_world_root_cid",
    )
    expected_root_generation = _int(
        payload.get("expected_root_generation", 0),
        "expected_root_generation",
        maximum=MAX_ROOT_GENERATION,
    )
    current_root_generation = _int(
        payload.get("current_root_generation", expected_root_generation),
        "current_root_generation",
        maximum=MAX_ROOT_GENERATION,
    )
    cas_binding = GenerationCasBinding(
        expected_root_generation=expected_root_generation,
        current_root_generation=current_root_generation,
        pre_world_root_cid=pre_world_root_cid,
        current_world_root_cid=current_world_root_cid,
    )
    undeclared = _undeclared_repository_terminal(payload.get("gitlinks"))
    ownership = CrossRepositoryOwnership(
        task_owner=_text(
            payload.get("task_owner", ACCELERATOR_TASK_OWNER), "task_owner"
        ),
        gitlinks=() if undeclared is not None else _parse_gitlinks(payload.get("gitlinks")),
        nested_write_paths=list(payload.get("nested_write_paths") or ()),
    )
    outbox = _outbox_nominations(tree_id=tree_id, **artifacts)
    worktree_id = str(payload.get("worktree_id") or "")
    lease_id = str(payload.get("lease_id") or "")
    fence_id = str(payload.get("fence_id") or "")
    terminal = _parse_terminal(payload.get("terminal"))
    kit_vfs_available = payload.get("kit_vfs_available", True)
    if kit_vfs_available is not True:
        if type(kit_vfs_available) is not bool:
            raise WorldRootAdapterError("kit_vfs_available must be a boolean")
        terminal = TypedTerminal(
            kind=TerminalKind.CAPABILITY_UNAVAILABLE.value,
            reason="kit VFS/outbox/CAS capability is unavailable",
        )

    nested_terminal = _nested_write_terminal(ownership)
    if undeclared is not None and terminal is None:
        terminal = undeclared
    if nested_terminal is not None and terminal is None:
        terminal = nested_terminal

    crash = payload.get("crash", False)
    if crash not in (True, False):
        raise WorldRootAdapterError("crash must be a boolean")

    if terminal is not None:
        world_root = _unchanged_world_root(
            tree_id=tree_id,
            pre_world_root_cid=pre_world_root_cid,
            post_world_root_cid=current_world_root_cid,
            expected_root_generation=expected_root_generation,
            resulting_root_generation=current_root_generation,
            artifacts=artifacts,
        )
        return WorldRootReceipt(
            tree_id=tree_id,
            status=AdapterStatus.TYPED_TERMINAL.value,
            world_root=world_root,
            cas_binding=cas_binding,
            outbox=outbox,
            ownership=ownership,
            terminal=terminal,
            worktree_id=worktree_id,
            lease_id=lease_id,
            fence_id=fence_id,
            negative_evidence_cids=list(payload.get("negative_evidence_cids") or ()),
        )

    if crash is True:
        last_committed_world_root_cid = _cid(
            payload.get("last_committed_world_root_cid") or current_world_root_cid,
            "last_committed_world_root_cid",
        )
        last_committed_root_generation = _int(
            payload.get("last_committed_root_generation", current_root_generation),
            "last_committed_root_generation",
            maximum=MAX_ROOT_GENERATION,
        )
        pending_outbox_cids = _cids(
            list(payload.get("pending_outbox_cids") or ()), "pending_outbox_cids"
        )
        recovery = WalRecoveryPlan(
            last_committed_world_root_cid=last_committed_world_root_cid,
            last_committed_root_generation=last_committed_root_generation,
            recovered_world_root_cid=last_committed_world_root_cid,
            recovered_root_generation=last_committed_root_generation,
            pending_outbox_cids=pending_outbox_cids,
        )
        world_root = _unchanged_world_root(
            tree_id=tree_id,
            pre_world_root_cid=pre_world_root_cid,
            post_world_root_cid=last_committed_world_root_cid,
            expected_root_generation=expected_root_generation,
            resulting_root_generation=last_committed_root_generation,
            artifacts=artifacts,
        )
        negative = list(payload.get("negative_evidence_cids") or ())
        negative.extend(pending_outbox_cids)
        return WorldRootReceipt(
            tree_id=tree_id,
            status=AdapterStatus.RECOVERED.value,
            world_root=world_root,
            cas_binding=cas_binding,
            outbox=outbox,
            ownership=ownership,
            recovery=recovery,
            worktree_id=worktree_id,
            lease_id=lease_id,
            fence_id=fence_id,
            negative_evidence_cids=negative,
        )

    if cas_binding.stale_writer:
        rejection = StaleWriterRejection(
            attempted_expected_root_generation=expected_root_generation,
            current_root_generation=current_root_generation,
            attempted_pre_world_root_cid=pre_world_root_cid,
            current_world_root_cid=current_world_root_cid,
        )
        world_root = _unchanged_world_root(
            tree_id=tree_id,
            pre_world_root_cid=pre_world_root_cid,
            post_world_root_cid=current_world_root_cid,
            expected_root_generation=expected_root_generation,
            resulting_root_generation=current_root_generation,
            artifacts=artifacts,
        )
        return WorldRootReceipt(
            tree_id=tree_id,
            status=AdapterStatus.REJECTED_STALE_WRITER.value,
            world_root=world_root,
            cas_binding=cas_binding,
            outbox=outbox,
            ownership=ownership,
            stale_writer=rejection,
            worktree_id=worktree_id,
            lease_id=lease_id,
            fence_id=fence_id,
            negative_evidence_cids=[rejection.rejection_cid],
        )

    if cas_binding.root_conflict:
        world_root = _unchanged_world_root(
            tree_id=tree_id,
            pre_world_root_cid=pre_world_root_cid,
            post_world_root_cid=current_world_root_cid,
            expected_root_generation=expected_root_generation,
            resulting_root_generation=current_root_generation,
            artifacts=artifacts,
        )
        return WorldRootReceipt(
            tree_id=tree_id,
            status=AdapterStatus.REJECTED_ROOT_CONFLICT.value,
            world_root=world_root,
            cas_binding=cas_binding,
            outbox=outbox,
            ownership=ownership,
            worktree_id=worktree_id,
            lease_id=lease_id,
            fence_id=fence_id,
            negative_evidence_cids=[cas_binding.binding_cid],
        )

    if not (
        artifacts["packet_cids"]
        or artifacts["projection_cids"]
        or artifacts["receipt_cids"]
        or artifacts["transition_cids"]
    ):
        raise WorldRootAdapterError(
            "persist nomination requires packet, projection, receipt, or transition artifacts"
        )

    post_world_root_cid = _nominate_post_world_root_cid(
        tree_id=tree_id,
        pre_world_root_cid=pre_world_root_cid,
        expected_root_generation=expected_root_generation,
        **artifacts,
    )
    world_root = SemanticWorldRoot(
        tree_id=tree_id,
        pre_world_root_cid=pre_world_root_cid,
        post_world_root_cid=post_world_root_cid,
        expected_root_generation=expected_root_generation,
        resulting_root_generation=expected_root_generation + 1,
        **artifacts,
    )
    return WorldRootReceipt(
        tree_id=tree_id,
        status=AdapterStatus.NOMINATED_PERSIST.value,
        world_root=world_root,
        cas_binding=cas_binding,
        outbox=outbox,
        ownership=ownership,
        worktree_id=worktree_id,
        lease_id=lease_id,
        fence_id=fence_id,
    )


def dry_run_world_root(evidence: Mapping[str, Any]) -> WorldRootReceipt:
    """Deterministic dry-run. Never mutates and never accepts a world root."""

    return integrate_world_root(evidence, mutate=False)


def persist_through_kit_authorities(evidence: Mapping[str, Any]) -> WorldRootReceipt:
    """Nominate persist of packets/projections/receipts/transitions through kit."""

    return integrate_world_root(evidence)


def recover_world_root(evidence: Mapping[str, Any]) -> WorldRootReceipt:
    """Nominate WAL recovery of the last committed root without overwrite."""

    payload = _mapping(evidence, "world-root evidence")
    payload["crash"] = True
    return integrate_world_root(payload)


class SemanticRefactorWorldRootAdapter:
    """SPAR-039 world-root adapter. Nomination-only; kit remains authority."""

    interface: ClassVar[str] = SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_INTERFACE
    schema: ClassVar[str] = SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID

    def integrate(
        self,
        evidence: Mapping[str, Any],
        *,
        mutate: bool = False,
    ) -> WorldRootReceipt:
        return integrate_world_root(evidence, mutate=mutate)

    def persist(self, evidence: Mapping[str, Any]) -> WorldRootReceipt:
        return persist_through_kit_authorities(evidence)

    def recover(self, evidence: Mapping[str, Any]) -> WorldRootReceipt:
        return recover_world_root(evidence)

    def dry_run(self, evidence: Mapping[str, Any]) -> WorldRootReceipt:
        return dry_run_world_root(evidence)


def encode_canonical_receipt(receipt: WorldRootReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> WorldRootReceipt:
    return WorldRootReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise WorldRootAdapterError(
            f"world-root adapter must not define competing types: {sorted(overlap)}"
        )
    if "WorldRootStore" in names:
        raise WorldRootAdapterError("WorldRootStore is not a SPAR world-root contract")


__all__ = [
    "ACCELERATOR_TASK_OWNER",
    "ADAPTER_CAN_AUTHORIZE_COMPLETION",
    "ADAPTER_CAN_AUTHORIZE_TRANSITION",
    "ADAPTER_CAN_CREATE_AUTHORITY",
    "ADAPTER_IS_NOMINATION_ONLY",
    "ADAPTER_OWNS_CAS",
    "ADAPTER_OWNS_OUTBOX",
    "ADAPTER_OWNS_VFS",
    "ADAPTER_OWNS_WAL",
    "ADAPTER_WRITES_REPOSITORY",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "AdapterStatus",
    "CAS_AUTHORITY_OWNER",
    "CRASH_RECOVERY_WITHOUT_OVERWRITE",
    "CROSS_REPOSITORY_OWNERSHIP_INTERFACE",
    "CrossRepositoryOwnership",
    "DECLARED_ADAPTER_STATUSES",
    "DECLARED_PERSIST_KINDS",
    "DECLARED_REPOSITORY_OWNERS",
    "DECLARED_TERMINAL_KINDS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXISTING_ADAPTER_AUTHORITIES",
    "FORBIDDEN_WORLD_ROOT_NAMES",
    "GENERATION_CAS_BINDING_INTERFACE",
    "GENERATION_CAS_REQUIRED",
    "GITLINK_BINDING_INTERFACE",
    "GOAL_ID",
    "GenerationCasBinding",
    "GitlinkBinding",
    "IDENTITY_EXCLUDED_FIELDS",
    "KIT_OWNS_CAS",
    "KIT_OWNS_OUTBOX",
    "KIT_OWNS_VFS",
    "KIT_OWNS_WAL_RECOVERY",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "NEGATIVE_EVIDENCE_RETAINED",
    "NESTED_WRITES_REQUIRE_ACCELERATOR_TASK_OWNER",
    "NESTED_WRITES_REQUIRE_EXPLICIT_GITLINK",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "OUTBOX_AUTHORITY_OWNER",
    "PERSIST_KINDS",
    "PREDECESSOR_TASK_IDS",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "PersistKind",
    "RAW_SOURCE_REQUIRED",
    "ROOT_CONFLICT_OVERWRITE_FORBIDDEN",
    "SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_INTERFACE",
    "SEMANTIC_WORLD_ROOT_INTERFACE",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "STALE_WRITER_REJECTED",
    "STALE_WRITER_REJECTION_INTERFACE",
    "SemanticRefactorWorldRootAdapter",
    "SemanticWorldRoot",
    "StaleWriterRejection",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TYPED_TERMINAL_INTERFACE",
    "TerminalKind",
    "TypedTerminal",
    "USES_CURRENT_LEASE_FENCE",
    "USES_CURRENT_WORKTREE",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "VFS_AUTHORITY_OWNER",
    "VFS_OUTBOX_NOMINATION_INTERFACE",
    "VfsOutboxNomination",
    "WAL_AUTHORITY_OWNER",
    "WAL_RECOVERY_PLAN_INTERFACE",
    "WORKER_SELF_APPROVAL",
    "WORLD_ROOT_CONTRACT_VERSION",
    "WORLD_ROOT_RECEIPT_INTERFACE",
    "WalRecoveryPlan",
    "WorldRootAdapterError",
    "WorldRootReceipt",
    "assert_not_competing_capsule_family",
    "decode_canonical_receipt",
    "dry_run_world_root",
    "encode_canonical_receipt",
    "integrate_world_root",
    "persist_through_kit_authorities",
    "provider_free_exports",
    "recover_world_root",
    "world_root_adapter_cid_profile",
    "world_root_adapter_descriptor",
]
