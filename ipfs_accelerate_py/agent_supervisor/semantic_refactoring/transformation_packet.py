"""SPAR-019 exact refactor transformation packets.

This module extends current supervisor partition orchestration with
``RefactorTransformationPacket@1``.  It consumes SPAR-016 boundary-contract
mappings, SPAR-017 target-API plan mappings, SPAR-018 façade-plan mappings,
and SPAR-014 ranked partition candidates, then binds exact preimages, bounded
moves/rewrites/adapters/façade edits, expected deltas, allowed paths/effects,
lease/fence identifiers, validation, and rollback.

SPAR-016/017/018 payloads are ingested as mappings only.  This module does
not replace datasets semantic authority, does not apply CST transforms, and
cannot authorize a transition, completion, or competing authority.  Vector,
model, and heuristic evidence cannot admit a packet.  Observational metadata
is excluded from identity.  Dry-run is deterministic and never mutates.
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
    PartitionGenerationReceipt,
    ProgramPartitionCandidate,
)
from .partition_policy import (
    ANALYZER_ID as SPAR014_ANALYZER_ID,
    PartitionComparisonReceipt,
    compare_partition_candidates,
)


TASK_ID: Final[str] = "SPAR-019"
GOAL_ID: Final[str] = "SPAR-G041"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.transformation_packet@1"
)

REFACTOR_TRANSFORMATION_PACKET_INTERFACE: Final[str] = (
    "RefactorTransformationPacket@1"
)
TRANSFORMATION_PREIMAGE_INTERFACE: Final[str] = "TransformationPreimage@1"
BOUNDED_EDIT_INTERFACE: Final[str] = "BoundedEdit@1"
EXPECTED_DELTA_INTERFACE: Final[str] = "ExpectedDelta@1"
EFFECT_SCOPE_INTERFACE: Final[str] = "EffectScope@1"
LEASE_FENCE_BINDING_INTERFACE: Final[str] = "LeaseFenceBinding@1"
ROLLBACK_PLAN_INTERFACE: Final[str] = "RollbackPlan@1"
TRANSFORMATION_PACKET_RECEIPT_INTERFACE: Final[str] = (
    "TransformationPacketReceipt@1"
)
DRY_RUN_RECEIPT_INTERFACE: Final[str] = "DryRunReceipt@1"

REFACTOR_TRANSFORMATION_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-transformation-packet@1"
)
TRANSFORMATION_PREIMAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transformation-preimage@1"
)
BOUNDED_EDIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/bounded-edit@1"
)
EXPECTED_DELTA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/expected-delta@1"
)
EFFECT_SCOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/effect-scope@1"
)
LEASE_FENCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/lease-fence-binding@1"
)
ROLLBACK_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rollback-plan@1"
)
TRANSFORMATION_PACKET_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transformation-packet-receipt@1"
)
DRY_RUN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dry-run-receipt@1"
)

PACKET_CONTRACT_VERSION: Final[str] = "1"

PACKET_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
PACKET_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
PACKET_CAN_CREATE_AUTHORITY: Final[bool] = False
PACKET_CAN_RETIRE_FACADE: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
PACKET_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_MODULES: Final[int] = 16_384
MAX_EDITS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024

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

DEFAULT_ALLOWED_EFFECTS: Final[tuple[str, ...]] = (
    "bounded_source_edit",
    "bounded_test_edit",
    "isolated_validation",
)
DECLARED_ALLOWED_EFFECTS: Final[frozenset[str]] = frozenset(DEFAULT_ALLOWED_EFFECTS)
DECLARED_FORBIDDEN_EFFECTS: Final[frozenset[str]] = frozenset(
    {
        "network",
        "install",
        "model_download",
        "protected_branch",
        "credential",
        "production",
        "unrestricted_diff",
        "hidden_dynamic_frontier",
        "direct_multiprocess_duckdb",
    }
)
INCOMPLETE_CONTRACT_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "retrieval",
        "proof",
        "abstention",
        "review",
    }
)
REQUIRED_UNSUPPORTED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "undispositioned",
        "unsupported",
    }
)
ROLLBACK_MODE: Final[str] = "restore_preimages_or_discard_worktree"


class TransformationPacketError(ValueError):
    """Fail-closed violation of a SPAR-019 transformation-packet contract."""


class EditKind(str, Enum):
    MOVE = "move"
    REWRITE = "rewrite"
    ADAPTER = "adapter"
    FACADE = "facade"


class RewriteKind(str, Enum):
    IMPORT = "import"
    CALLSITE = "callsite"
    REEXPORT = "reexport"
    DEPRECATION = "deprecation"
    SERIALIZATION = "serialization"
    INTROSPECTION = "introspection"
    TRACEBACK = "traceback"
    PATCH_TARGET = "patch_target"


class AdapterKind(str, Enum):
    STATE = "state"
    BOUNDARY = "boundary"
    PROTOCOL = "protocol"
    WRAPPER = "wrapper"
    CLI = "cli"
    PLUGIN = "plugin"
    REGISTRY = "registry"


class SyntaxSupport(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


DECLARED_EDIT_KINDS: Final[frozenset[str]] = frozenset(kind.value for kind in EditKind)
DECLARED_REWRITE_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in RewriteKind
)
DECLARED_ADAPTER_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in AdapterKind
)

_MIGRATION_TO_EDIT: Final[Mapping[str, tuple[str, str, str]]] = {
    "reexport": (EditKind.REWRITE.value, RewriteKind.REEXPORT.value, ""),
    "wrapper": (EditKind.ADAPTER.value, "", AdapterKind.WRAPPER.value),
    "facade": (EditKind.FACADE.value, "", ""),
    "deprecation": (EditKind.REWRITE.value, RewriteKind.DEPRECATION.value, ""),
    "cli": (EditKind.ADAPTER.value, "", AdapterKind.CLI.value),
    "plugin": (EditKind.ADAPTER.value, "", AdapterKind.PLUGIN.value),
    "registry": (EditKind.ADAPTER.value, "", AdapterKind.REGISTRY.value),
    "serialization": (EditKind.REWRITE.value, RewriteKind.SERIALIZATION.value, ""),
    "introspection": (EditKind.REWRITE.value, RewriteKind.INTROSPECTION.value, ""),
    "traceback": (EditKind.REWRITE.value, RewriteKind.TRACEBACK.value, ""),
    "patch_target": (EditKind.REWRITE.value, RewriteKind.PATCH_TARGET.value, ""),
}


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise TransformationPacketError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise TransformationPacketError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise TransformationPacketError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise TransformationPacketError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise TransformationPacketError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise TransformationPacketError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TransformationPacketError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise TransformationPacketError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise TransformationPacketError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise TransformationPacketError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise TransformationPacketError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise TransformationPacketError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise TransformationPacketError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise TransformationPacketError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise TransformationPacketError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TransformationPacketError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise TransformationPacketError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise TransformationPacketError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise TransformationPacketError(f"unknown {name}: {text}") from exc


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
    raise TransformationPacketError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise TransformationPacketError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def transformation_packet_cid_profile() -> dict[str, str]:
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
            raise TransformationPacketError(f"{name} cannot claim {flag}")


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise TransformationPacketError(f"{name} exceeds path bound")
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
        raise TransformationPacketError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise TransformationPacketError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TransformationPacketError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise TransformationPacketError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise TransformationPacketError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TransformationPacketError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise TransformationPacketError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise TransformationPacketError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise TransformationPacketError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _effects(values: Any, name: str, *, declared: frozenset[str]) -> tuple[str, ...]:
    ordered = _unique_sorted_text(list(values), name, limit=MAX_MEMBERS)
    unknown = [item for item in ordered if item not in declared]
    if unknown:
        raise TransformationPacketError(f"unknown {name}: {unknown}")
    return ordered


@dataclass(frozen=True, slots=True)
class TransformationPreimage:
    """Exact repository/tree/environment/graph/partition/source bindings."""

    tree_id: str
    repository_id: str
    environment_cid: str
    graph_cid: str
    partition_cid: str
    source_cids: Sequence[str]

    interface: ClassVar[str] = TRANSFORMATION_PREIMAGE_INTERFACE
    schema: ClassVar[str] = TRANSFORMATION_PREIMAGE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "repository_id",
            "environment_cid",
            "graph_cid",
            "partition_cid",
            "source_cids",
            "preimage_cid",
        }
    )

    def __post_init__(self) -> None:
        sources = tuple(sorted(_cid(item, "source_cids") for item in self.source_cids))
        if not sources:
            raise TransformationPacketError("preimage requires source_cids")
        if len(sources) != len(set(sources)):
            raise TransformationPacketError("source_cids must not contain duplicates")
        if len(sources) > MAX_EVIDENCE_CIDS:
            raise TransformationPacketError("source_cids exceed maximum length")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self, "environment_cid", _cid(self.environment_cid, "environment_cid")
        )
        object.__setattr__(self, "graph_cid", _cid(self.graph_cid, "graph_cid"))
        object.__setattr__(
            self, "partition_cid", _cid(self.partition_cid, "partition_cid")
        )
        object.__setattr__(self, "source_cids", sources)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TRANSFORMATION_PREIMAGE_SCHEMA,
            "interface": TRANSFORMATION_PREIMAGE_INTERFACE,
            "tree_id": self.tree_id,
            "repository_id": self.repository_id,
            "environment_cid": self.environment_cid,
            "graph_cid": self.graph_cid,
            "partition_cid": self.partition_cid,
            "source_cids": list(self.source_cids),
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def preimage_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["preimage_cid"] = self.preimage_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransformationPreimage":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("preimage_cid")
        if payload.pop("schema") != TRANSFORMATION_PREIMAGE_SCHEMA:
            raise TransformationPacketError("unsupported TransformationPreimage schema")
        if payload.pop("interface") != TRANSFORMATION_PREIMAGE_INTERFACE:
            raise TransformationPacketError(
                "unsupported TransformationPreimage interface"
            )
        result = cls(**payload)
        _verify_cid(claimed, result.preimage_cid, "TransformationPreimage preimage_cid")
        return result


def _coerce_preimage(
    value: TransformationPreimage | Mapping[str, Any],
) -> TransformationPreimage:
    if isinstance(value, TransformationPreimage):
        return value
    if isinstance(value, Mapping):
        if "preimage_cid" in value:
            return TransformationPreimage.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "preimage_cid"}
        }
        return TransformationPreimage(**payload)
    raise TransformationPacketError("preimage must be a TransformationPreimage")


@dataclass(frozen=True, slots=True)
class BoundedEdit:
    """One bounded move, rewrite, adapter, or façade edit. Nomination only."""

    kind: EditKind | str
    source_id: str
    destination_id: str
    member_ids: Sequence[str] = ()
    rewrite_kind: str = ""
    adapter_kind: str = ""
    syntax_support: SyntaxSupport | str = SyntaxSupport.SUPPORTED
    write_paths: Sequence[str] = ()
    obligation_id: str = ""

    interface: ClassVar[str] = BOUNDED_EDIT_INTERFACE
    schema: ClassVar[str] = BOUNDED_EDIT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "kind",
            "source_id",
            "destination_id",
            "member_ids",
            "rewrite_kind",
            "adapter_kind",
            "syntax_support",
            "write_paths",
            "obligation_id",
            "edit_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _enum(self.kind, EditKind, "kind")
        syntax = _enum(self.syntax_support, SyntaxSupport, "syntax_support")
        if syntax != SyntaxSupport.SUPPORTED.value:
            raise TransformationPacketError(
                "unsupported syntax is a typed terminal"
            )
        rewrite = _text(self.rewrite_kind, "rewrite_kind", empty=True)
        adapter = _text(self.adapter_kind, "adapter_kind", empty=True)
        if kind == EditKind.REWRITE.value:
            if rewrite not in DECLARED_REWRITE_KINDS:
                raise TransformationPacketError("rewrite requires a declared rewrite_kind")
            if adapter:
                raise TransformationPacketError("rewrite cannot declare adapter_kind")
        elif kind == EditKind.ADAPTER.value:
            if adapter not in DECLARED_ADAPTER_KINDS:
                raise TransformationPacketError("adapter requires a declared adapter_kind")
            if rewrite:
                raise TransformationPacketError("adapter cannot declare rewrite_kind")
        elif rewrite or adapter:
            raise TransformationPacketError(
                f"{kind} cannot declare rewrite_kind or adapter_kind"
            )
        members = _unique_sorted_text(
            list(self.member_ids), "member_ids", limit=MAX_MEMBERS
        )
        paths = _exact_paths(list(self.write_paths), "write_paths", required=True)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(
            self, "destination_id", _text(self.destination_id, "destination_id")
        )
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "rewrite_kind", rewrite)
        object.__setattr__(self, "adapter_kind", adapter)
        object.__setattr__(self, "syntax_support", syntax)
        object.__setattr__(self, "write_paths", paths)
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id", empty=True)
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": BOUNDED_EDIT_SCHEMA,
            "interface": BOUNDED_EDIT_INTERFACE,
            "kind": self.kind,
            "source_id": self.source_id,
            "destination_id": self.destination_id,
            "member_ids": list(self.member_ids),
            "rewrite_kind": self.rewrite_kind,
            "adapter_kind": self.adapter_kind,
            "syntax_support": self.syntax_support,
            "write_paths": list(self.write_paths),
            "obligation_id": self.obligation_id,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def edit_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["edit_cid"] = self.edit_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundedEdit":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("edit_cid")
        if payload.pop("schema") != BOUNDED_EDIT_SCHEMA:
            raise TransformationPacketError("unsupported BoundedEdit schema")
        if payload.pop("interface") != BOUNDED_EDIT_INTERFACE:
            raise TransformationPacketError("unsupported BoundedEdit interface")
        result = cls(**payload)
        _verify_cid(claimed, result.edit_cid, "BoundedEdit edit_cid")
        return result


def _coerce_edit(value: BoundedEdit | Mapping[str, Any]) -> BoundedEdit:
    if isinstance(value, BoundedEdit):
        return value
    if isinstance(value, Mapping):
        if "edit_cid" in value:
            return BoundedEdit.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "edit_cid"}
        }
        return BoundedEdit(**payload)
    raise TransformationPacketError("edit must be a BoundedEdit")


@dataclass(frozen=True, slots=True)
class ExpectedDelta:
    """Nominated graph/semantic delta. Not an accepted transition."""

    moved_member_ids: Sequence[str] = ()
    destination_module_ids: Sequence[str] = ()
    preserved_identity_ids: Sequence[str] = ()
    changed_binding_ids: Sequence[str] = ()
    facade_subject_ids: Sequence[str] = ()

    interface: ClassVar[str] = EXPECTED_DELTA_INTERFACE
    schema: ClassVar[str] = EXPECTED_DELTA_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "moved_member_ids",
            "destination_module_ids",
            "preserved_identity_ids",
            "changed_binding_ids",
            "facade_subject_ids",
            "delta_cid",
        }
    )

    def __post_init__(self) -> None:
        moved = _unique_sorted_text(
            list(self.moved_member_ids), "moved_member_ids", limit=MAX_MEMBERS
        )
        destinations = _unique_sorted_text(
            list(self.destination_module_ids),
            "destination_module_ids",
            limit=MAX_MODULES,
        )
        preserved = _unique_sorted_text(
            list(self.preserved_identity_ids),
            "preserved_identity_ids",
            limit=MAX_MEMBERS,
        )
        changed = _unique_sorted_text(
            list(self.changed_binding_ids), "changed_binding_ids", limit=MAX_MEMBERS
        )
        facades = _unique_sorted_text(
            list(self.facade_subject_ids), "facade_subject_ids", limit=MAX_MEMBERS
        )
        if set(preserved) - set(moved):
            raise TransformationPacketError(
                "preserved identities must be a subset of moved members"
            )
        object.__setattr__(self, "moved_member_ids", moved)
        object.__setattr__(self, "destination_module_ids", destinations)
        object.__setattr__(self, "preserved_identity_ids", preserved)
        object.__setattr__(self, "changed_binding_ids", changed)
        object.__setattr__(self, "facade_subject_ids", facades)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXPECTED_DELTA_SCHEMA,
            "interface": EXPECTED_DELTA_INTERFACE,
            "moved_member_ids": list(self.moved_member_ids),
            "destination_module_ids": list(self.destination_module_ids),
            "preserved_identity_ids": list(self.preserved_identity_ids),
            "changed_binding_ids": list(self.changed_binding_ids),
            "facade_subject_ids": list(self.facade_subject_ids),
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def delta_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["delta_cid"] = self.delta_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExpectedDelta":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("delta_cid")
        if payload.pop("schema") != EXPECTED_DELTA_SCHEMA:
            raise TransformationPacketError("unsupported ExpectedDelta schema")
        if payload.pop("interface") != EXPECTED_DELTA_INTERFACE:
            raise TransformationPacketError("unsupported ExpectedDelta interface")
        result = cls(**payload)
        _verify_cid(claimed, result.delta_cid, "ExpectedDelta delta_cid")
        return result


def _coerce_delta(value: ExpectedDelta | Mapping[str, Any]) -> ExpectedDelta:
    if isinstance(value, ExpectedDelta):
        return value
    if isinstance(value, Mapping):
        if "delta_cid" in value:
            return ExpectedDelta.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "delta_cid"}
        }
        return ExpectedDelta(**payload)
    raise TransformationPacketError("expected_delta must be an ExpectedDelta")


@dataclass(frozen=True, slots=True)
class EffectScope:
    """Exact allowed paths and effects. Unrestricted scope is rejected."""

    write_paths: Sequence[str]
    allowed_effects: Sequence[str] = DEFAULT_ALLOWED_EFFECTS
    forbidden_effects: Sequence[str] = tuple(sorted(DECLARED_FORBIDDEN_EFFECTS))

    interface: ClassVar[str] = EFFECT_SCOPE_INTERFACE
    schema: ClassVar[str] = EFFECT_SCOPE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "write_paths",
            "allowed_effects",
            "forbidden_effects",
            "scope_cid",
        }
    )

    def __post_init__(self) -> None:
        paths = _exact_paths(list(self.write_paths), "write_paths", required=True)
        allowed = _effects(
            list(self.allowed_effects or DEFAULT_ALLOWED_EFFECTS),
            "allowed_effects",
            declared=DECLARED_ALLOWED_EFFECTS,
        )
        forbidden = _effects(
            list(self.forbidden_effects or sorted(DECLARED_FORBIDDEN_EFFECTS)),
            "forbidden_effects",
            declared=DECLARED_FORBIDDEN_EFFECTS,
        )
        overlap = set(allowed) & set(forbidden)
        if overlap:
            raise TransformationPacketError(
                f"allowed effects cannot include forbidden effects: {sorted(overlap)}"
            )
        if set(allowed) & DECLARED_FORBIDDEN_EFFECTS:
            raise TransformationPacketError("forbidden packet effects cannot be allowed")
        if not allowed:
            raise TransformationPacketError("effect scope requires allowed_effects")
        object.__setattr__(self, "write_paths", paths)
        object.__setattr__(self, "allowed_effects", allowed)
        object.__setattr__(self, "forbidden_effects", forbidden)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EFFECT_SCOPE_SCHEMA,
            "interface": EFFECT_SCOPE_INTERFACE,
            "write_paths": list(self.write_paths),
            "allowed_effects": list(self.allowed_effects),
            "forbidden_effects": list(self.forbidden_effects),
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def scope_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["scope_cid"] = self.scope_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EffectScope":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("scope_cid")
        if payload.pop("schema") != EFFECT_SCOPE_SCHEMA:
            raise TransformationPacketError("unsupported EffectScope schema")
        if payload.pop("interface") != EFFECT_SCOPE_INTERFACE:
            raise TransformationPacketError("unsupported EffectScope interface")
        result = cls(**payload)
        _verify_cid(claimed, result.scope_cid, "EffectScope scope_cid")
        return result


def _coerce_scope(value: EffectScope | Mapping[str, Any]) -> EffectScope:
    if isinstance(value, EffectScope):
        return value
    if isinstance(value, Mapping):
        if "scope_cid" in value:
            return EffectScope.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "scope_cid"}
        }
        return EffectScope(**payload)
    raise TransformationPacketError("effect_scope must be an EffectScope")


@dataclass(frozen=True, slots=True)
class LeaseFenceBinding:
    """Content-addressed lease/fence identifiers. Not live observational state."""

    lease_id: str
    fence_id: str
    epoch_id: str = ""

    interface: ClassVar[str] = LEASE_FENCE_BINDING_INTERFACE
    schema: ClassVar[str] = LEASE_FENCE_BINDING_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "lease_id",
            "fence_id",
            "epoch_id",
            "binding_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _cid(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence_id", _cid(self.fence_id, "fence_id"))
        object.__setattr__(self, "epoch_id", _optional_cid(self.epoch_id, "epoch_id"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": LEASE_FENCE_BINDING_SCHEMA,
            "interface": LEASE_FENCE_BINDING_INTERFACE,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "epoch_id": self.epoch_id,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "LeaseFenceBinding":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("binding_cid")
        if payload.pop("schema") != LEASE_FENCE_BINDING_SCHEMA:
            raise TransformationPacketError("unsupported LeaseFenceBinding schema")
        if payload.pop("interface") != LEASE_FENCE_BINDING_INTERFACE:
            raise TransformationPacketError("unsupported LeaseFenceBinding interface")
        result = cls(**payload)
        _verify_cid(claimed, result.binding_cid, "LeaseFenceBinding binding_cid")
        return result


def _coerce_lease_fence(
    value: LeaseFenceBinding | Mapping[str, Any],
) -> LeaseFenceBinding:
    if isinstance(value, LeaseFenceBinding):
        return value
    if isinstance(value, Mapping):
        if "binding_cid" in value:
            return LeaseFenceBinding.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "binding_cid"}
        }
        return LeaseFenceBinding(**payload)
    raise TransformationPacketError("lease_fence must be a LeaseFenceBinding")


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """Restore exact preimages or discard the isolated worktree."""

    mode: str = ROLLBACK_MODE
    restore_source_cids: Sequence[str] = ()
    retain_negative_evidence: bool = True
    release_lease_fence: bool = True
    advance_accepted_roots: bool = False

    interface: ClassVar[str] = ROLLBACK_PLAN_INTERFACE
    schema: ClassVar[str] = ROLLBACK_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "mode",
            "restore_source_cids",
            "retain_negative_evidence",
            "release_lease_fence",
            "advance_accepted_roots",
            "rollback_cid",
        }
    )

    def __post_init__(self) -> None:
        mode = _text(self.mode, "mode")
        if mode != ROLLBACK_MODE:
            raise TransformationPacketError("rollback mode must remain canonical")
        sources = tuple(
            sorted(_cid(item, "restore_source_cids") for item in self.restore_source_cids)
        )
        if not sources:
            raise TransformationPacketError("rollback requires restore_source_cids")
        if len(sources) != len(set(sources)):
            raise TransformationPacketError(
                "restore_source_cids must not contain duplicates"
            )
        if _bool(self.advance_accepted_roots, "advance_accepted_roots"):
            raise TransformationPacketError("rollback cannot advance accepted roots")
        if not _bool(self.retain_negative_evidence, "retain_negative_evidence"):
            raise TransformationPacketError("rollback must retain negative evidence")
        if not _bool(self.release_lease_fence, "release_lease_fence"):
            raise TransformationPacketError("rollback must release lease/fence")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "restore_source_cids", sources)
        object.__setattr__(self, "retain_negative_evidence", True)
        object.__setattr__(self, "release_lease_fence", True)
        object.__setattr__(self, "advance_accepted_roots", False)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ROLLBACK_PLAN_SCHEMA,
            "interface": ROLLBACK_PLAN_INTERFACE,
            "mode": self.mode,
            "restore_source_cids": list(self.restore_source_cids),
            "retain_negative_evidence": True,
            "release_lease_fence": True,
            "advance_accepted_roots": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def rollback_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["rollback_cid"] = self.rollback_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RollbackPlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("rollback_cid")
        if payload.pop("schema") != ROLLBACK_PLAN_SCHEMA:
            raise TransformationPacketError("unsupported RollbackPlan schema")
        if payload.pop("interface") != ROLLBACK_PLAN_INTERFACE:
            raise TransformationPacketError("unsupported RollbackPlan interface")
        result = cls(**payload)
        _verify_cid(claimed, result.rollback_cid, "RollbackPlan rollback_cid")
        return result


def _coerce_rollback(value: RollbackPlan | Mapping[str, Any]) -> RollbackPlan:
    if isinstance(value, RollbackPlan):
        return value
    if isinstance(value, Mapping):
        if "rollback_cid" in value:
            return RollbackPlan.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "rollback_cid"}
        }
        return RollbackPlan(**payload)
    raise TransformationPacketError("rollback must be a RollbackPlan")


@dataclass(frozen=True, slots=True)
class RefactorTransformationPacket:
    """Exact bounded transformation packet. Nomination only."""

    tree_id: str
    preimage: TransformationPreimage | Mapping[str, Any]
    edits: Sequence[BoundedEdit | Mapping[str, Any]]
    expected_delta: ExpectedDelta | Mapping[str, Any]
    effect_scope: EffectScope | Mapping[str, Any]
    lease_fence: LeaseFenceBinding | Mapping[str, Any]
    rollback: RollbackPlan | Mapping[str, Any]
    validation_commands: Sequence[str]
    selected_candidate_cids: Sequence[str] = ()
    rejected_candidate_cids: Sequence[str] = ()
    advisory_candidate_cids: Sequence[str] = ()
    evidence_cids: Sequence[str] = ()
    boundary_contract_set_cid: str = ""
    target_api_plan_cid: str = ""
    facade_plan_cid: str = ""
    comparison_receipt_cid: str = ""
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = REFACTOR_TRANSFORMATION_PACKET_INTERFACE
    schema: ClassVar[str] = REFACTOR_TRANSFORMATION_PACKET_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "preimage",
            "edits",
            "expected_delta",
            "effect_scope",
            "lease_fence",
            "rollback",
            "validation_commands",
            "selected_candidate_cids",
            "rejected_candidate_cids",
            "advisory_candidate_cids",
            "negative_evidence_cids",
            "evidence_cids",
            "boundary_contract_set_cid",
            "target_api_plan_cid",
            "facade_plan_cid",
            "comparison_receipt_cid",
            "analyzer_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "can_retire_facade",
            "packet_is_nomination_only",
            "dry_run_is_deterministic",
            "unrestricted_scope",
            "unsupported_syntax",
            "packet_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise TransformationPacketError(
                "analyzer_id must remain the SPAR-019 analyzer"
            )
        tree_id = _tree_id(self.tree_id)
        preimage = _coerce_preimage(self.preimage)
        if preimage.tree_id != tree_id:
            raise TransformationPacketError("preimage tree_id does not match packet")
        edits = tuple(_coerce_edit(item) for item in self.edits)
        if not edits:
            raise TransformationPacketError("packet requires bounded edits")
        if len(edits) > MAX_EDITS:
            raise TransformationPacketError("edits exceed maximum length")
        edit_ids = [item.edit_cid for item in edits]
        if len(edit_ids) != len(set(edit_ids)):
            raise TransformationPacketError("edits must be unique")
        edits = tuple(
            sorted(
                edits,
                key=lambda item: (
                    item.kind,
                    item.source_id,
                    item.destination_id,
                    item.obligation_id,
                    item.edit_cid,
                ),
            )
        )
        delta = _coerce_delta(self.expected_delta)
        scope = _coerce_scope(self.effect_scope)
        lease_fence = _coerce_lease_fence(self.lease_fence)
        rollback = _coerce_rollback(self.rollback)
        if set(rollback.restore_source_cids) != set(preimage.source_cids):
            raise TransformationPacketError(
                "rollback restore_source_cids must match preimage source_cids"
            )
        commands = _commands(list(self.validation_commands))
        selected = _unique_sorted_text(
            list(self.selected_candidate_cids),
            "selected_candidate_cids",
            limit=MAX_MODULES,
        )
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
        if set(selected) & set(rejected):
            raise TransformationPacketError("rejected candidates cannot become packets")
        if set(selected) & set(advisory):
            raise TransformationPacketError("advisory candidates cannot become packets")
        evidence = tuple(sorted(_cid(item, "evidence_cids") for item in self.evidence_cids))
        if len(evidence) != len(set(evidence)):
            raise TransformationPacketError("evidence_cids must not contain duplicates")
        if len(evidence) > MAX_EVIDENCE_CIDS:
            raise TransformationPacketError("evidence_cids exceed maximum length")
        scope_paths = set(scope.write_paths)
        for edit in edits:
            extra = set(edit.write_paths) - scope_paths
            if extra:
                raise TransformationPacketError(
                    "edit write_paths must stay inside effect_scope"
                )
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "preimage", preimage)
        object.__setattr__(self, "edits", edits)
        object.__setattr__(self, "expected_delta", delta)
        object.__setattr__(self, "effect_scope", scope)
        object.__setattr__(self, "lease_fence", lease_fence)
        object.__setattr__(self, "rollback", rollback)
        object.__setattr__(self, "validation_commands", commands)
        object.__setattr__(self, "selected_candidate_cids", selected)
        object.__setattr__(self, "rejected_candidate_cids", rejected)
        object.__setattr__(self, "advisory_candidate_cids", advisory)
        object.__setattr__(self, "evidence_cids", evidence)
        object.__setattr__(
            self,
            "boundary_contract_set_cid",
            _optional_cid(self.boundary_contract_set_cid, "boundary_contract_set_cid"),
        )
        object.__setattr__(
            self,
            "target_api_plan_cid",
            _cid(self.target_api_plan_cid, "target_api_plan_cid")
            if self.target_api_plan_cid
            else "",
        )
        if not self.target_api_plan_cid:
            raise TransformationPacketError("packet requires target_api_plan_cid")
        object.__setattr__(
            self,
            "facade_plan_cid",
            _cid(self.facade_plan_cid, "facade_plan_cid") if self.facade_plan_cid else "",
        )
        if not self.facade_plan_cid:
            raise TransformationPacketError("packet requires facade_plan_cid")
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
    def can_retire_facade(self) -> bool:
        return False

    @property
    def packet_is_nomination_only(self) -> bool:
        return True

    @property
    def dry_run_is_deterministic(self) -> bool:
        return True

    @property
    def unrestricted_scope(self) -> bool:
        return False

    @property
    def unsupported_syntax(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFACTOR_TRANSFORMATION_PACKET_SCHEMA,
            "interface": REFACTOR_TRANSFORMATION_PACKET_INTERFACE,
            "tree_id": self.tree_id,
            "preimage": self.preimage.to_dict(),
            "edits": [item.to_dict() for item in self.edits],
            "expected_delta": self.expected_delta.to_dict(),
            "effect_scope": self.effect_scope.to_dict(),
            "lease_fence": self.lease_fence.to_dict(),
            "rollback": self.rollback.to_dict(),
            "validation_commands": list(self.validation_commands),
            "selected_candidate_cids": list(self.selected_candidate_cids),
            "rejected_candidate_cids": list(self.rejected_candidate_cids),
            "advisory_candidate_cids": list(self.advisory_candidate_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "evidence_cids": list(self.evidence_cids),
            "boundary_contract_set_cid": self.boundary_contract_set_cid,
            "target_api_plan_cid": self.target_api_plan_cid,
            "facade_plan_cid": self.facade_plan_cid,
            "comparison_receipt_cid": self.comparison_receipt_cid,
            "analyzer_id": self.analyzer_id,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "can_retire_facade": False,
            "packet_is_nomination_only": True,
            "dry_run_is_deterministic": True,
            "unrestricted_scope": False,
            "unsupported_syntax": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def packet_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["packet_cid"] = self.packet_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorTransformationPacket":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("packet_cid")
        if payload.pop("schema") != REFACTOR_TRANSFORMATION_PACKET_SCHEMA:
            raise TransformationPacketError(
                "unsupported RefactorTransformationPacket schema"
            )
        if payload.pop("interface") != REFACTOR_TRANSFORMATION_PACKET_INTERFACE:
            raise TransformationPacketError(
                "unsupported RefactorTransformationPacket interface"
            )
        _pop_authority_flags(payload, "RefactorTransformationPacket")
        if payload.pop("packet_is_nomination_only") is not True:
            raise TransformationPacketError("packet must remain nomination_only")
        if payload.pop("can_retire_facade") is not False:
            raise TransformationPacketError("packet cannot claim can_retire_facade")
        if payload.pop("dry_run_is_deterministic") is not True:
            raise TransformationPacketError("dry-run must remain deterministic")
        if payload.pop("unrestricted_scope") is not False:
            raise TransformationPacketError("packet cannot claim unrestricted_scope")
        if payload.pop("unsupported_syntax") is not False:
            raise TransformationPacketError("packet cannot claim unsupported_syntax")
        payload.pop("negative_evidence_cids")
        result = cls(**payload)
        _verify_cid(
            claimed, result.packet_cid, "RefactorTransformationPacket packet_cid"
        )
        return result


@dataclass(frozen=True, slots=True)
class TransformationPacketReceipt:
    """Body-free compilation receipt. Independent validation remains separate."""

    tree_id: str
    packet: RefactorTransformationPacket | Mapping[str, Any]
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = TRANSFORMATION_PACKET_RECEIPT_INTERFACE
    schema: ClassVar[str] = TRANSFORMATION_PACKET_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet",
            "analyzer_id",
            "packet_cid",
            "selected_candidate_cids",
            "rejected_candidate_cids",
            "negative_evidence_cids",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "can_retire_facade",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise TransformationPacketError(
                "analyzer_id must remain the SPAR-019 analyzer"
            )
        packet = (
            self.packet
            if isinstance(self.packet, RefactorTransformationPacket)
            else RefactorTransformationPacket.from_dict(_mapping(self.packet, "packet"))
        )
        tree_id = _tree_id(self.tree_id)
        if packet.tree_id != tree_id:
            raise TransformationPacketError("receipt tree_id does not match packet")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "packet", packet)
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def packet_cid(self) -> str:
        return self.packet.packet_cid

    @property
    def selected_candidate_cids(self) -> tuple[str, ...]:
        return self.packet.selected_candidate_cids

    @property
    def rejected_candidate_cids(self) -> tuple[str, ...]:
        return self.packet.rejected_candidate_cids

    @property
    def negative_evidence_cids(self) -> tuple[str, ...]:
        return self.packet.negative_evidence_cids

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
    def can_retire_facade(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TRANSFORMATION_PACKET_RECEIPT_SCHEMA,
            "interface": TRANSFORMATION_PACKET_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet": self.packet.to_dict(),
            "analyzer_id": self.analyzer_id,
            "packet_cid": self.packet_cid,
            "selected_candidate_cids": list(self.selected_candidate_cids),
            "rejected_candidate_cids": list(self.rejected_candidate_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "can_retire_facade": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "TransformationPacketReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != TRANSFORMATION_PACKET_RECEIPT_SCHEMA:
            raise TransformationPacketError(
                "unsupported TransformationPacketReceipt schema"
            )
        if payload.pop("interface") != TRANSFORMATION_PACKET_RECEIPT_INTERFACE:
            raise TransformationPacketError(
                "unsupported TransformationPacketReceipt interface"
            )
        _pop_authority_flags(payload, "TransformationPacketReceipt")
        if payload.pop("can_retire_facade") is not False:
            raise TransformationPacketError("receipt cannot claim can_retire_facade")
        payload.pop("packet_cid")
        payload.pop("selected_candidate_cids")
        payload.pop("rejected_candidate_cids")
        payload.pop("negative_evidence_cids")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "TransformationPacketReceipt receipt_cid"
        )
        return result


@dataclass(frozen=True, slots=True)
class DryRunReceipt:
    """Deterministic dry-run. Never mutates and never authorizes a transition."""

    packet_cid: str
    tree_id: str
    edit_cids: Sequence[str]
    write_paths: Sequence[str]
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = DRY_RUN_RECEIPT_INTERFACE
    schema: ClassVar[str] = DRY_RUN_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "packet_cid",
            "tree_id",
            "edit_cids",
            "write_paths",
            "mutated",
            "deterministic",
            "can_authorize_transition",
            "dry_run_cid",
        }
    )

    def __post_init__(self) -> None:
        if _bool(self.mutated, "mutated"):
            raise TransformationPacketError("dry-run cannot mutate")
        if not _bool(self.deterministic, "deterministic"):
            raise TransformationPacketError("dry-run must remain deterministic")
        edits = tuple(sorted(_cid(item, "edit_cids") for item in self.edit_cids))
        if len(edits) != len(set(edits)):
            raise TransformationPacketError("edit_cids must not contain duplicates")
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "edit_cids", edits)
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": DRY_RUN_RECEIPT_SCHEMA,
            "interface": DRY_RUN_RECEIPT_INTERFACE,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "edit_cids": list(self.edit_cids),
            "write_paths": list(self.write_paths),
            "mutated": False,
            "deterministic": True,
            "can_authorize_transition": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def dry_run_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["dry_run_cid"] = self.dry_run_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DryRunReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("dry_run_cid")
        if payload.pop("schema") != DRY_RUN_RECEIPT_SCHEMA:
            raise TransformationPacketError("unsupported DryRunReceipt schema")
        if payload.pop("interface") != DRY_RUN_RECEIPT_INTERFACE:
            raise TransformationPacketError("unsupported DryRunReceipt interface")
        if payload.pop("can_authorize_transition") is not False:
            raise TransformationPacketError("dry-run cannot claim can_authorize_transition")
        result = cls(**payload)
        _verify_cid(claimed, result.dry_run_cid, "DryRunReceipt dry_run_cid")
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
    raise TransformationPacketError("candidate must be a ProgramPartitionCandidate")


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
            raise TransformationPacketError("packet compilation requires SPAR-014 candidates")
        return candidates
    raise TransformationPacketError(
        "candidates must be a SPAR-013 receipt or candidate list"
    )


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
        raise TransformationPacketError("comparison must be a SPAR-014 receipt")
    if comparison.analyzer_id != SPAR014_ANALYZER_ID:
        raise TransformationPacketError("comparison must remain the SPAR-014 analyzer")
    return comparison


def _pairwise_disjoint(candidates: Sequence[ProgramPartitionCandidate]) -> bool:
    seen: set[str] = set()
    for candidate in candidates:
        overlap = seen.intersection(candidate.member_ids)
        if overlap:
            return False
        seen.update(candidate.member_ids)
    return True


def _select_candidates(
    candidates: Sequence[ProgramPartitionCandidate],
    comparison: PartitionComparisonReceipt,
    selected_candidate_cids: Sequence[str] | None,
) -> tuple[ProgramPartitionCandidate, ...]:
    by_cid = {item.candidate_cid: item for item in candidates}
    ranked_cids = set(comparison.ranked_candidate_cids)
    rejected_cids = set(comparison.rejected_candidate_cids)
    advisory_cids = set(comparison.advisory_candidate_cids)
    if selected_candidate_cids is None:
        ranked = tuple(
            by_cid[item] for item in comparison.ranked_candidate_cids if item in by_cid
        )
        if not ranked:
            raise TransformationPacketError("no ranked SPAR-014 candidates")
        if not _pairwise_disjoint(ranked):
            raise TransformationPacketError(
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
                raise TransformationPacketError("selected candidate is not present")
            if cid not in ranked_cids:
                raise TransformationPacketError(
                    "selected candidate is not ranked by SPAR-014"
                )
            if cid in rejected_cids:
                raise TransformationPacketError("rejected candidates cannot become packets")
            if cid in advisory_cids:
                raise TransformationPacketError("advisory candidates cannot become packets")
            selected_list.append(by_cid[cid])
        selected = tuple(selected_list)
        if not selected:
            raise TransformationPacketError("selected_candidate_cids must not be empty")
        if not _pairwise_disjoint(selected):
            raise TransformationPacketError(
                "overlapping modules cannot form a simultaneous packet"
            )
    tree_id = selected[0].tree_id
    for candidate in selected:
        if candidate.advisory:
            raise TransformationPacketError("advisory candidates cannot become packets")
        if candidate.hard_constraint_violations:
            raise TransformationPacketError("violating candidates cannot become packets")
        if candidate.evidence_class in _NON_ADMITTING_EVIDENCE:
            raise TransformationPacketError(
                "vector or model evidence cannot admit a packet"
            )
        if candidate.tree_id != tree_id:
            raise TransformationPacketError("selected candidate tree_id does not match")
    return selected


def _as_mapping_list(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise TransformationPacketError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(_mapping(item, name))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            items.append(_mapping(to_dict(), name))
            continue
        raise TransformationPacketError(f"{name} entries must be objects")
    return tuple(items)


def _coerce_boundary_contracts(value: Any, tree_id: str) -> dict[str, Any]:
    payload = _mapping(value, "boundary_contracts")
    contracts_tree = _tree_id(payload.get("tree_id") or tree_id)
    if contracts_tree != tree_id:
        raise TransformationPacketError("SPAR-016 tree_id does not match candidates")
    source_cid = _cid(payload.get("source_cid") or "", "source_cid")
    partition_cid = _optional_cid(
        payload.get("partition_cid") or payload.get("candidate_cid") or "",
        "partition_cid",
    )
    contracts = _as_mapping_list(payload.get("contracts") or (), "contracts")
    if not contracts:
        raise TransformationPacketError("packet requires SPAR-016 contracts")
    for contract in contracts:
        required = _bool(contract.get("required", True), "required")
        complete = contract.get("complete")
        if complete is None:
            complete = True
        else:
            complete = _bool(complete, "complete")
        disposition = _text(
            contract.get("disposition") or "admitted", "disposition", empty=False
        )
        if required and (not complete or disposition in INCOMPLETE_CONTRACT_DISPOSITIONS):
            raise TransformationPacketError(
                "incomplete SPAR-016 contract is a typed terminal"
            )
        if required and disposition == "unsupported":
            raise TransformationPacketError(
                "unsupported required SPAR-016 contract is a typed terminal"
            )
        allowed = contract.get("allowed_effects") or ()
        if allowed:
            _effects(list(allowed), "allowed_effects", declared=DECLARED_ALLOWED_EFFECTS)
        forbidden = contract.get("forbidden_effects") or ()
        if forbidden:
            _effects(
                list(forbidden),
                "forbidden_effects",
                declared=DECLARED_FORBIDDEN_EFFECTS,
            )
    set_cid = _optional_cid(
        payload.get("contract_set_cid") or payload.get("set_cid") or "",
        "contract_set_cid",
    )
    if not set_cid:
        set_cid = cid_for_dag_json(
            {
                "tree_id": contracts_tree,
                "source_cid": source_cid,
                "partition_cid": partition_cid,
                "edge_ids": [
                    _text(item.get("edge_id") or item.get("obligation_id") or "", "edge_id")
                    for item in contracts
                ],
            }
        )
    allowed: list[str] = []
    forbidden: list[str] = []
    for contract in contracts:
        allowed.extend(contract.get("allowed_effects") or ())
        forbidden.extend(contract.get("forbidden_effects") or ())
    if not allowed:
        allowed = list(DEFAULT_ALLOWED_EFFECTS)
    forbidden = sorted(set(forbidden) | DECLARED_FORBIDDEN_EFFECTS)
    return {
        "tree_id": contracts_tree,
        "source_cid": source_cid,
        "partition_cid": partition_cid,
        "contract_set_cid": set_cid,
        "contracts": contracts,
        "allowed_effects": tuple(sorted(set(allowed))),
        "forbidden_effects": tuple(sorted(set(forbidden))),
    }


def _coerce_target_api_plan(value: Any, tree_id: str) -> dict[str, Any]:
    payload = _mapping(value, "target_api_plan")
    plan_tree = payload.get("tree_id")
    if plan_tree not in {None, ""}:
        if _tree_id(plan_tree) != tree_id:
            raise TransformationPacketError("SPAR-017 tree_id does not match candidates")
    plan_cid = _cid(payload.get("plan_cid") or payload.get("api_cid") or "", "plan_cid")
    modules = _as_mapping_list(payload.get("modules") or (), "modules")
    if not modules:
        raise TransformationPacketError("packet requires SPAR-017 modules")
    return {"tree_id": tree_id, "plan_cid": plan_cid, "modules": modules}


def _coerce_facade_plan(value: Any, tree_id: str) -> dict[str, Any]:
    payload = _mapping(value, "facade_plan")
    plan_tree = payload.get("tree_id")
    if plan_tree not in {None, ""}:
        if _tree_id(plan_tree) != tree_id:
            raise TransformationPacketError("SPAR-018 tree_id does not match candidates")
    plan_cid = _cid(payload.get("plan_cid") or "", "plan_cid")
    consumer_plans = _as_mapping_list(
        payload.get("consumer_plans") or (), "consumer_plans"
    )
    if not consumer_plans:
        raise TransformationPacketError("packet requires SPAR-018 consumer_plans")
    subject_facades = _as_mapping_list(
        payload.get("subject_facades") or (), "subject_facades"
    )
    if not subject_facades:
        raise TransformationPacketError("packet requires SPAR-018 subject_facades")
    for item in consumer_plans:
        required = _bool(item.get("required", True), "required")
        disposition = _text(item.get("disposition") or "", "disposition")
        if required and disposition in REQUIRED_UNSUPPORTED_DISPOSITIONS:
            raise TransformationPacketError(
                "required undispositioned or unsupported SPAR-018 obligation is a typed terminal"
            )
    return {
        "tree_id": tree_id,
        "plan_cid": plan_cid,
        "consumer_plans": consumer_plans,
        "subject_facades": subject_facades,
    }


def _subject_module(facade_plan: Mapping[str, Any]) -> str:
    modules = {
        _text(item.get("subject_module") or "", "subject_module")
        for item in facade_plan["subject_facades"]
    }
    if len(modules) == 1:
        return next(iter(modules))
    return sorted(modules)[0]


def _edit_for_migration(
    *,
    migration_kind: str,
    source_id: str,
    destination_id: str,
    member_ids: Sequence[str],
    write_paths: Sequence[str],
    obligation_id: str,
) -> BoundedEdit:
    mapping = _MIGRATION_TO_EDIT.get(migration_kind)
    if mapping is None:
        raise TransformationPacketError("unsupported syntax is a typed terminal")
    kind, rewrite_kind, adapter_kind = mapping
    return BoundedEdit(
        kind=kind,
        source_id=source_id,
        destination_id=destination_id,
        member_ids=member_ids,
        rewrite_kind=rewrite_kind,
        adapter_kind=adapter_kind,
        syntax_support=SyntaxSupport.SUPPORTED,
        write_paths=write_paths,
        obligation_id=obligation_id,
    )


def compile_refactor_transformation_packet(
    *,
    boundary_contracts: Mapping[str, Any],
    target_api_plan: Mapping[str, Any],
    facade_plan: Mapping[str, Any],
    candidates: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    | Mapping[str, Any],
    comparison: PartitionComparisonReceipt | Mapping[str, Any] | None = None,
    selected_candidate_cids: Sequence[str] | None = None,
    preimage: Mapping[str, Any],
    write_paths: Sequence[str],
    lease_id: str,
    fence_id: str,
    epoch_id: str = "",
    validation_commands: Sequence[str],
    repository_id: str = PROGRAM,
) -> RefactorTransformationPacket:
    """Compile one exact bounded packet from SPAR-016/017/018 mappings.

    SPAR-014 ranked, non-advisory candidates bind destinations. Incomplete
    required SPAR-016 contracts and required undispositioned or unsupported
    SPAR-018 obligations fail closed. Unrestricted write scope and unsupported
    syntax are typed terminals. The result cannot authorize a transition or
    completion.
    """

    resolved_candidates = _coerce_candidates(candidates)
    tree_ids = {item.tree_id for item in resolved_candidates}
    if len(tree_ids) != 1:
        raise TransformationPacketError("candidates must share one tree_id")
    tree_id = next(iter(tree_ids))
    resolved_comparison = _coerce_comparison(comparison, resolved_candidates)
    if resolved_comparison.tree_id != tree_id:
        raise TransformationPacketError("comparison tree_id does not match candidates")
    selected = _select_candidates(
        resolved_candidates, resolved_comparison, selected_candidate_cids
    )
    contracts = _coerce_boundary_contracts(boundary_contracts, tree_id)
    api_plan = _coerce_target_api_plan(target_api_plan, tree_id)
    facades = _coerce_facade_plan(facade_plan, tree_id)
    paths = _exact_paths(list(write_paths), "write_paths", required=True)
    source_module = _subject_module(facades)
    partition_cid = contracts["partition_cid"] or selected[0].candidate_cid
    source_cids = list(preimage.get("source_cids") or ())
    if contracts["source_cid"] not in source_cids:
        source_cids.append(contracts["source_cid"])
    resolved_preimage = TransformationPreimage(
        tree_id=tree_id,
        repository_id=_text(
            preimage.get("repository_id") or repository_id, "repository_id"
        ),
        environment_cid=_cid(preimage.get("environment_cid") or "", "environment_cid"),
        graph_cid=_cid(preimage.get("graph_cid") or "", "graph_cid"),
        partition_cid=_cid(
            preimage.get("partition_cid") or partition_cid, "partition_cid"
        ),
        source_cids=source_cids,
    )
    edits: list[BoundedEdit] = []
    moved: list[str] = []
    destinations: list[str] = []
    for module in api_plan["modules"]:
        module_id = _text(module.get("module_id") or "", "module_id")
        members = tuple(
            sorted(_text(item, "member_ids") for item in module.get("member_ids") or ())
        )
        if not members:
            raise TransformationPacketError("SPAR-017 module requires member_ids")
        moved.extend(members)
        destinations.append(module_id)
        edits.append(
            BoundedEdit(
                kind=EditKind.MOVE,
                source_id=source_module,
                destination_id=module_id,
                member_ids=members,
                syntax_support=SyntaxSupport.SUPPORTED,
                write_paths=paths,
            )
        )
    changed_bindings: list[str] = []
    facade_subjects: list[str] = []
    for plan in facades["consumer_plans"]:
        migration_kind = _text(plan.get("migration_kind") or "", "migration_kind")
        subject_id = _text(plan.get("subject_id") or "", "subject_id")
        subject_module = _text(plan.get("subject_module") or source_module, "subject_module")
        destination = _text(
            plan.get("target_module_id") or destinations[0], "target_module_id"
        )
        obligation_id = _text(plan.get("obligation_id") or "", "obligation_id")
        disposition = _text(plan.get("disposition") or "", "disposition")
        edits.append(
            _edit_for_migration(
                migration_kind=migration_kind,
                source_id=subject_module,
                destination_id=destination,
                member_ids=(subject_id,),
                write_paths=paths,
                obligation_id=obligation_id,
            )
        )
        if disposition in {"migrate", "facade", "explicit_incompatibility"}:
            changed_bindings.append(subject_id)
        if migration_kind == EditKind.FACADE.value:
            facade_subjects.append(subject_id)
    for facade in facades["subject_facades"]:
        subject_id = _text(facade.get("subject_id") or "", "subject_id")
        required = _bool(facade.get("facade_required", False), "facade_required")
        if _bool(facade.get("can_retire_facade", False), "can_retire_facade"):
            raise TransformationPacketError("packet cannot retire a façade")
        if required:
            facade_subjects.append(subject_id)
            already = any(
                item.kind == EditKind.FACADE.value and subject_id in item.member_ids
                for item in edits
            )
            if not already:
                edits.append(
                    BoundedEdit(
                        kind=EditKind.FACADE,
                        source_id=_text(
                            facade.get("subject_module") or source_module, "subject_module"
                        ),
                        destination_id=_text(
                            facade.get("target_module_id") or destinations[0],
                            "target_module_id",
                        ),
                        member_ids=(subject_id,),
                        syntax_support=SyntaxSupport.SUPPORTED,
                        write_paths=paths,
                    )
                )
    evidence = tuple(
        sorted(
            {
                *resolved_comparison.evidence_cids,
                *(cid for item in selected for cid in item.evidence_cids),
                resolved_preimage.environment_cid,
                resolved_preimage.graph_cid,
                *resolved_preimage.source_cids,
            }
        )
    )
    return RefactorTransformationPacket(
        tree_id=tree_id,
        preimage=resolved_preimage,
        edits=edits,
        expected_delta=ExpectedDelta(
            moved_member_ids=moved,
            destination_module_ids=destinations,
            preserved_identity_ids=moved,
            changed_binding_ids=tuple(sorted(set(changed_bindings))),
            facade_subject_ids=tuple(sorted(set(facade_subjects))),
        ),
        effect_scope=EffectScope(
            write_paths=paths,
            allowed_effects=contracts["allowed_effects"],
            forbidden_effects=contracts["forbidden_effects"],
        ),
        lease_fence=LeaseFenceBinding(
            lease_id=lease_id,
            fence_id=fence_id,
            epoch_id=epoch_id,
        ),
        rollback=RollbackPlan(
            restore_source_cids=resolved_preimage.source_cids,
        ),
        validation_commands=validation_commands,
        selected_candidate_cids=tuple(item.candidate_cid for item in selected),
        rejected_candidate_cids=tuple(resolved_comparison.rejected_candidate_cids),
        advisory_candidate_cids=tuple(resolved_comparison.advisory_candidate_cids),
        evidence_cids=evidence,
        boundary_contract_set_cid=contracts["contract_set_cid"],
        target_api_plan_cid=api_plan["plan_cid"],
        facade_plan_cid=facades["plan_cid"],
        comparison_receipt_cid=resolved_comparison.receipt_cid,
        analyzer_id=ANALYZER_ID,
    )


def compile_transformation_packet_receipt(
    packet: RefactorTransformationPacket | Mapping[str, Any],
) -> TransformationPacketReceipt:
    resolved = (
        packet
        if isinstance(packet, RefactorTransformationPacket)
        else RefactorTransformationPacket.from_dict(packet)
    )
    return TransformationPacketReceipt(tree_id=resolved.tree_id, packet=resolved)


def dry_run_transformation_packet(
    packet: RefactorTransformationPacket | Mapping[str, Any],
) -> DryRunReceipt:
    """Return a deterministic no-mutation dry-run of one sealed packet."""

    resolved = (
        packet
        if isinstance(packet, RefactorTransformationPacket)
        else RefactorTransformationPacket.from_dict(packet)
    )
    return DryRunReceipt(
        packet_cid=resolved.packet_cid,
        tree_id=resolved.tree_id,
        edit_cids=tuple(item.edit_cid for item in resolved.edits),
        write_paths=resolved.effect_scope.write_paths,
        mutated=False,
        deterministic=True,
    )


def encode_canonical_packet(packet: RefactorTransformationPacket) -> dict[str, Any]:
    return packet.to_dict()


def decode_canonical_packet(payload: Mapping[str, Any]) -> RefactorTransformationPacket:
    return RefactorTransformationPacket.from_dict(payload)


def encode_canonical_receipt(receipt: TransformationPacketReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> TransformationPacketReceipt:
    return TransformationPacketReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise TransformationPacketError(
            f"transformation packet must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "BOUNDED_EDIT_INTERFACE",
    "DECLARED_ADAPTER_KINDS",
    "DECLARED_ALLOWED_EFFECTS",
    "DECLARED_EDIT_KINDS",
    "DECLARED_FORBIDDEN_EFFECTS",
    "DECLARED_REWRITE_KINDS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DRY_RUN_RECEIPT_INTERFACE",
    "DUCKLAKE_IS_AUTHORITY",
    "EFFECT_SCOPE_INTERFACE",
    "EXPECTED_DELTA_INTERFACE",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "LEASE_FENCE_BINDING_INTERFACE",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PACKET_CAN_AUTHORIZE_COMPLETION",
    "PACKET_CAN_AUTHORIZE_TRANSITION",
    "PACKET_CAN_CREATE_AUTHORITY",
    "PACKET_CAN_RETIRE_FACADE",
    "PACKET_CONTRACT_VERSION",
    "PACKET_IS_NOMINATION_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REFACTOR_TRANSFORMATION_PACKET_INTERFACE",
    "ROLLBACK_MODE",
    "ROLLBACK_PLAN_INTERFACE",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TRANSFORMATION_PACKET_RECEIPT_INTERFACE",
    "TRANSFORMATION_PREIMAGE_INTERFACE",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "AdapterKind",
    "BoundedEdit",
    "DryRunReceipt",
    "EditKind",
    "EffectScope",
    "ExpectedDelta",
    "LeaseFenceBinding",
    "RefactorTransformationPacket",
    "RollbackPlan",
    "RewriteKind",
    "SyntaxSupport",
    "TransformationPacketError",
    "TransformationPacketReceipt",
    "TransformationPreimage",
    "assert_not_competing_capsule_family",
    "compile_refactor_transformation_packet",
    "compile_transformation_packet_receipt",
    "decode_canonical_packet",
    "decode_canonical_receipt",
    "dry_run_transformation_packet",
    "encode_canonical_packet",
    "encode_canonical_receipt",
    "provider_free_exports",
    "transformation_packet_cid_profile",
]


assert TASK_ID == "SPAR-019"
assert REFACTOR_TRANSFORMATION_PACKET_INTERFACE == "RefactorTransformationPacket@1"
assert PACKET_IS_NOMINATION_ONLY is True
assert PACKET_CAN_AUTHORIZE_COMPLETION is False
assert DRY_RUN_MUTATES is False
