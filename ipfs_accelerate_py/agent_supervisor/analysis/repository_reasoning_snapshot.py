"""Canonical repository reasoning snapshot for Planner and Doctor (PDR-010).

Interface: ``RepositoryReasoningSnapshot@1``

One content-addressed snapshot joins the repository forest (superproject tree,
dirty overlay, recursive gitlinks), task-source state, and every tool/policy
root that Planner and Doctor share.  Source bodies, secrets, host locators,
and provider claims never enter the durable identity.

Checked bridges reuse existing SCA dispositions and program-behavior entries;
they do not silently alias incompatible schemas or invent a second repository
root.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interface, schemas, bounds
# ---------------------------------------------------------------------------

REPOSITORY_REASONING_SNAPSHOT_INTERFACE: Final[str] = "RepositoryReasoningSnapshot@1"
REPOSITORY_REASONING_SNAPSHOT_VERSION: Final[int] = 1

REPOSITORY_REASONING_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-reasoning-snapshot@1"
)
REASONING_PATH_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-reasoning-path-entry@1"
)
REASONING_GITLINK_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-reasoning-gitlink-entry@1"
)
REASONING_TOOL_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-reasoning-tool-roots@1"
)
TASK_SOURCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-reasoning-task-source-binding@1"
)
REASONING_STABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-reasoning-stability@1"
)
REASONING_TRUNCATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-reasoning-truncation@1"
)

MAX_PATH_ENTRIES: Final[int] = 8_192
MAX_GITLINK_ENTRIES: Final[int] = 1_024
MAX_EXCLUSIONS: Final[int] = 4_096
MAX_REASON_CODES: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_RECORD_BYTES: Final[int] = 524_288
MAX_GITLINK_DEPTH: Final[int] = 16

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "source_bytes",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "private_key",
        "credential",
        "authorization_header",
        "cookie",
        "session",
    }
)
_PRIVATE_FIELD_MARKERS: Final[tuple[str, ...]] = (
    "secret",
    "password",
    "token",
    "api_key",
    "private_key",
    "credential",
    "authorization",
    "cookie",
    "session",
)

# Path statuses required by acceptance (tracked overlay of HEAD/index/worktree).
_PATH_STATUS_VALUES: Final[frozenset[str]] = frozenset(
    {
        "tracked",
        "staged",
        "modified",
        "deleted",
        "renamed",
        "admitted_untracked",
        "staged_and_modified",
        "staged_deletion",
        "mode_changed",
        "clean",
        "excluded",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RepositoryReasoningSnapshotError(ContractValidationError):
    """Fail-closed rejection for an unsafe or incomplete reasoning snapshot."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "repository_reasoning_snapshot_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "repository_reasoning_snapshot_error")


class RepositoryReasoningBoundsError(RepositoryReasoningSnapshotError):
    """A snapshot component exceeded its declared bound."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="bounds_exceeded")


class RepositoryReasoningAuthorityError(RepositoryReasoningSnapshotError):
    """Repository root, forest, or tool-root authority failed closed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="authority_mismatch")


class RepositoryReasoningTamperError(RepositoryReasoningSnapshotError):
    """Stored content identity does not match the canonical preimage."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="tampered_identity")


class RepositoryReasoningInstabilityError(RepositoryReasoningSnapshotError):
    """Admitted bytes or roots changed while the snapshot was constructed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="instability")


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ReasoningPathStatus(str, Enum):
    """Working-tree / index / admission status for one path entry."""

    TRACKED = "tracked"
    STAGED = "staged"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    ADMITTED_UNTRACKED = "admitted_untracked"
    STAGED_AND_MODIFIED = "staged_and_modified"
    STAGED_DELETION = "staged_deletion"
    MODE_CHANGED = "mode_changed"
    CLEAN = "clean"
    EXCLUDED = "excluded"


class ReasoningEntryKind(str, Enum):
    REGULAR = "regular"
    SYMLINK = "symlink"
    GITLINK = "gitlink"
    DIRECTORY = "directory"


class ReasoningCoverageKind(str, Enum):
    """Coverage disposition relative to analysis admission."""

    ADMITTED = "admitted"
    EXCLUDED = "excluded"
    UNSUPPORTED = "unsupported"
    PARSE_FAILURE = "parse_failure"
    GITLINK = "gitlink"
    DEPENDENCY = "dependency"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise RepositoryReasoningSnapshotError(f"{field_name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise RepositoryReasoningSnapshotError(f"{field_name} is required")
    if "\0" in text:
        raise RepositoryReasoningSnapshotError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise RepositoryReasoningBoundsError(f"{field_name} exceeds its byte bound")
    return text


def _identifier(value: Any, field_name: str) -> str:
    text = _text(value, field_name, required=True, limit=512)
    if any(char.isspace() for char in text):
        raise RepositoryReasoningSnapshotError(
            f"{field_name} must be an opaque compact identifier"
        )
    return text


def _optional_identifier(value: Any, field_name: str) -> str:
    if value in (None, ""):
        return ""
    return _identifier(value, field_name)


def _repo_path(value: Any, field_name: str = "path", *, allow_root: bool = False) -> str:
    raw = str(value if value is not None else "").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if "\0" in raw:
        raise RepositoryReasoningAuthorityError(
            f"{field_name} contains NUL: {value!r}"
        )
    candidate = PurePosixPath(raw or ".")
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise RepositoryReasoningAuthorityError(
            f"{field_name} escapes its repository root: {value!r}"
        )
    normalized = candidate.as_posix()
    if normalized == ".":
        if allow_root:
            return "."
        raise RepositoryReasoningAuthorityError(f"{field_name} is required")
    return normalized


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RepositoryReasoningSnapshotError(f"{field_name} must be a boolean")
    return value


def _nonneg_int(value: Any, field_name: str, *, maximum: int = 2**31 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepositoryReasoningSnapshotError(
            f"{field_name} must be a non-negative integer"
        )
    if value < 0 or value > maximum:
        raise RepositoryReasoningBoundsError(f"{field_name} is outside its hard bound")
    return value


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted(item.value for item in enum))
        raise RepositoryReasoningSnapshotError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _string_tuple(
    values: Any,
    field_name: str,
    *,
    limit: int = MAX_REASON_CODES,
    required: bool = False,
    sort: bool = True,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RepositoryReasoningSnapshotError(
            f"{field_name} must be a sequence of strings"
        )
    else:
        raw = values
    if len(raw) > limit:
        raise RepositoryReasoningBoundsError(f"{field_name} exceeds its item bound")
    items = [_text(item, field_name, required=True, limit=512) for item in raw]
    if sort:
        out = tuple(sorted(set(items)))
    else:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        out = tuple(ordered)
    if required and not out:
        raise RepositoryReasoningSnapshotError(f"{field_name} must not be empty")
    return out


def _is_forbidden_payload_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_").strip()
    if normalized in _BODY_MARKERS:
        return True
    for marker in _PRIVATE_FIELD_MARKERS:
        if normalized == marker or normalized.endswith("_" + marker):
            return True
    return False


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    if isinstance(value, float):
        raise RepositoryReasoningSnapshotError(
            f"{field_name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RepositoryReasoningSnapshotError(
                    f"{field_name} has a non-string key"
                )
            if _is_forbidden_payload_key(key):
                raise RepositoryReasoningSnapshotError(
                    f"{field_name} may not contain source bodies or secrets"
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise RepositoryReasoningSnapshotError(
            f"{field_name} may not contain binary bodies"
        )


def _bounded(record: CanonicalContract, name: str) -> None:
    payload = record.to_dict()
    _assert_body_free(payload, name)
    if len(canonical_json_bytes(payload)) > MAX_RECORD_BYTES:
        raise RepositoryReasoningBoundsError(
            f"{name} exceeds its serialized byte bound"
        )


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", payload.get("snapshot_cid")))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise RepositoryReasoningTamperError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any],
    schema: str,
    fields: Sequence[str],
    name: str,
    *,
    extra_allowed: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise RepositoryReasoningSnapshotError(f"{name} has an unsupported schema")
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, REPOSITORY_REASONING_SNAPSHOT_VERSION, str(REPOSITORY_REASONING_SNAPSHOT_VERSION)):
        raise RepositoryReasoningSnapshotError(
            f"{name} has an unsupported contract version"
        )
    _assert_body_free(payload, name)
    allowed = set(fields) | {
        "schema",
        "contract_version",
        "schema_version",
        "content_id",
        "cid",
        "snapshot_cid",
        "snapshot_id",
    }
    if extra_allowed:
        allowed |= set(extra_allowed)
    unknown = set(payload).difference(allowed)
    if unknown:
        raise RepositoryReasoningSnapshotError(f"{name} contains unsupported fields")
    return {field_name: payload[field_name] for field_name in fields if field_name in payload}


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise RepositoryReasoningBoundsError("nested structure exceeds depth bound")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise RepositoryReasoningSnapshotError("floating-point values are not allowed")
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item, depth=depth + 1) for item in value]
    raise RepositoryReasoningSnapshotError("unsupported structured value type")


def _git_object_id(value: Any, field_name: str, *, allow_empty: bool = True) -> str:
    text = _text(value, field_name, required=not allow_empty, limit=128)
    if not text:
        return ""
    lowered = text.lower()
    if len(lowered) not in (40, 64) or any(
        char not in "0123456789abcdef" for char in lowered
    ):
        raise RepositoryReasoningSnapshotError(
            f"{field_name} must be a lowercase Git object id"
        )
    return lowered


# ---------------------------------------------------------------------------
# Component records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReasoningToolRoots(CanonicalContract):
    """Parser, index, toolchain, capability, policy, and IR roots for one snapshot.

    Every field is a compact content identity or empty.  Drift of any bound
    root invalidates the reasoning snapshot for Planner and Doctor joins.
    """

    SCHEMA: ClassVar[str] = REASONING_TOOL_ROOTS_SCHEMA

    repository_id: str
    forest_id: str
    tree_id: str
    overlay_id: str = ""
    head_commit_id: str = ""
    head_tree_id: str = ""
    index_tree_id: str = ""
    parser_root: str = ""
    index_root: str = ""
    toolchain_root: str = ""
    capability_root: str = ""
    policy_root: str = ""
    ir_root: str = ""
    intent_ir_root: str = ""
    legal_ir_root: str = ""
    security_ir_root: str = ""
    program_behavior_root: str = ""
    ast_root: str = ""
    evidence_graph_root: str = ""
    vector_root: str = ""
    cache_root: str = ""
    scope_policy_id: str = ""
    scanner_root: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "forest_id", _identifier(self.forest_id, "forest_id"))
        object.__setattr__(self, "tree_id", _identifier(self.tree_id, "tree_id"))
        for name in (
            "overlay_id",
            "parser_root",
            "index_root",
            "toolchain_root",
            "capability_root",
            "policy_root",
            "ir_root",
            "intent_ir_root",
            "legal_ir_root",
            "security_ir_root",
            "program_behavior_root",
            "ast_root",
            "evidence_graph_root",
            "vector_root",
            "cache_root",
            "scope_policy_id",
            "scanner_root",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        for name in ("head_commit_id", "head_tree_id", "index_tree_id"):
            object.__setattr__(
                self, name, _git_object_id(getattr(self, name), name, allow_empty=True)
            )
        _bounded(self, "reasoning tool roots")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPOSITORY_REASONING_SNAPSHOT_VERSION,
            "repository_id": self.repository_id,
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "overlay_id": self.overlay_id,
            "head_commit_id": self.head_commit_id,
            "head_tree_id": self.head_tree_id,
            "index_tree_id": self.index_tree_id,
            "parser_root": self.parser_root,
            "index_root": self.index_root,
            "toolchain_root": self.toolchain_root,
            "capability_root": self.capability_root,
            "policy_root": self.policy_root,
            "ir_root": self.ir_root,
            "intent_ir_root": self.intent_ir_root,
            "legal_ir_root": self.legal_ir_root,
            "security_ir_root": self.security_ir_root,
            "program_behavior_root": self.program_behavior_root,
            "ast_root": self.ast_root,
            "evidence_graph_root": self.evidence_graph_root,
            "vector_root": self.vector_root,
            "cache_root": self.cache_root,
            "scope_policy_id": self.scope_policy_id,
            "scanner_root": self.scanner_root,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningToolRoots":
        fields = (
            "repository_id",
            "forest_id",
            "tree_id",
            "overlay_id",
            "head_commit_id",
            "head_tree_id",
            "index_tree_id",
            "parser_root",
            "index_root",
            "toolchain_root",
            "capability_root",
            "policy_root",
            "ir_root",
            "intent_ir_root",
            "legal_ir_root",
            "security_ir_root",
            "program_behavior_root",
            "ast_root",
            "evidence_graph_root",
            "vector_root",
            "cache_root",
            "scope_policy_id",
            "scanner_root",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "reasoning tool roots")
        value = cls(**values)
        _verify_identity(payload, value)
        return value

    def require_same_repository(self, other: "ReasoningToolRoots") -> None:
        if self.repository_id != other.repository_id:
            raise RepositoryReasoningAuthorityError(
                "cross-repository replay is rejected"
            )
        if self.forest_id != other.forest_id or self.tree_id != other.tree_id:
            raise RepositoryReasoningAuthorityError(
                "forest/tree roots disagree for the same repository"
            )


@dataclass(frozen=True)
class ReasoningPathEntry(CanonicalContract):
    """One path projected across HEAD, index, worktree, and admission policy."""

    SCHEMA: ClassVar[str] = REASONING_PATH_ENTRY_SCHEMA

    path: str
    status: ReasoningPathStatus
    coverage: ReasoningCoverageKind = ReasoningCoverageKind.ADMITTED
    entry_kind: ReasoningEntryKind = ReasoningEntryKind.REGULAR
    tracked: bool = True
    overlay: bool = False
    head_digest: str = ""
    index_digest: str = ""
    worktree_digest: str = ""
    rename_from: str = ""
    reason_code: str = ""
    policy_rule: str = ""
    git_mode: str = ""
    git_object_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path, "path"))
        object.__setattr__(
            self, "status", _enum(self.status, ReasoningPathStatus, "status")
        )
        object.__setattr__(
            self, "coverage", _enum(self.coverage, ReasoningCoverageKind, "coverage")
        )
        object.__setattr__(
            self, "entry_kind", _enum(self.entry_kind, ReasoningEntryKind, "entry_kind")
        )
        object.__setattr__(self, "tracked", _bool(self.tracked, "tracked"))
        object.__setattr__(self, "overlay", _bool(self.overlay, "overlay"))
        for name in ("head_digest", "index_digest", "worktree_digest"):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        if self.rename_from:
            object.__setattr__(
                self, "rename_from", _repo_path(self.rename_from, "rename_from")
            )
        else:
            object.__setattr__(self, "rename_from", "")
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", limit=256)
        )
        object.__setattr__(
            self, "policy_rule", _text(self.policy_rule, "policy_rule", limit=256)
        )
        object.__setattr__(self, "git_mode", _text(self.git_mode, "git_mode", limit=16))
        object.__setattr__(
            self,
            "git_object_id",
            _git_object_id(self.git_object_id, "git_object_id", allow_empty=True),
        )
        if self.status is ReasoningPathStatus.RENAMED and not self.rename_from:
            raise RepositoryReasoningSnapshotError(
                f"renamed path {self.path!r} requires rename_from"
            )
        if self.status is ReasoningPathStatus.ADMITTED_UNTRACKED and self.tracked:
            raise RepositoryReasoningSnapshotError(
                f"admitted-untracked path {self.path!r} cannot be marked tracked"
            )
        if self.coverage is ReasoningCoverageKind.EXCLUDED:
            object.__setattr__(self, "status", ReasoningPathStatus.EXCLUDED)
        _bounded(self, "reasoning path entry")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPOSITORY_REASONING_SNAPSHOT_VERSION,
            "path": self.path,
            "status": self.status.value,
            "coverage": self.coverage.value,
            "entry_kind": self.entry_kind.value,
            "tracked": self.tracked,
            "overlay": self.overlay,
            "head_digest": self.head_digest,
            "index_digest": self.index_digest,
            "worktree_digest": self.worktree_digest,
            "rename_from": self.rename_from,
            "reason_code": self.reason_code,
            "policy_rule": self.policy_rule,
            "git_mode": self.git_mode,
            "git_object_id": self.git_object_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningPathEntry":
        fields = (
            "path",
            "status",
            "coverage",
            "entry_kind",
            "tracked",
            "overlay",
            "head_digest",
            "index_digest",
            "worktree_digest",
            "rename_from",
            "reason_code",
            "policy_rule",
            "git_mode",
            "git_object_id",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "reasoning path entry")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class ReasoningGitlinkEntry(CanonicalContract):
    """Recursive submodule / gitlink identity (not expanded as source)."""

    SCHEMA: ClassVar[str] = REASONING_GITLINK_ENTRY_SCHEMA

    path: str
    commit_id: str
    depth: int = 0
    mode: str = "160000"
    head_object_id: str = ""
    index_object_id: str = ""
    parent_path: str = ""
    nested: tuple["ReasoningGitlinkEntry", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path, "path"))
        object.__setattr__(
            self, "commit_id", _git_object_id(self.commit_id, "commit_id", allow_empty=False)
        )
        object.__setattr__(
            self, "depth", _nonneg_int(self.depth, "depth", maximum=MAX_GITLINK_DEPTH)
        )
        if self.depth > MAX_GITLINK_DEPTH:
            raise RepositoryReasoningBoundsError("gitlink depth exceeds hard bound")
        object.__setattr__(self, "mode", _text(self.mode or "160000", "mode", limit=16))
        object.__setattr__(
            self,
            "head_object_id",
            _git_object_id(self.head_object_id, "head_object_id", allow_empty=True),
        )
        object.__setattr__(
            self,
            "index_object_id",
            _git_object_id(self.index_object_id, "index_object_id", allow_empty=True),
        )
        if self.parent_path:
            object.__setattr__(
                self, "parent_path", _repo_path(self.parent_path, "parent_path")
            )
        else:
            object.__setattr__(self, "parent_path", "")
        nested = tuple(self.nested or ())
        normalized_nested: list[ReasoningGitlinkEntry] = []
        for item in nested:
            if isinstance(item, ReasoningGitlinkEntry):
                child = item
            elif isinstance(item, Mapping):
                child = ReasoningGitlinkEntry.from_dict(item)
            else:
                raise RepositoryReasoningSnapshotError(
                    "nested gitlinks must be ReasoningGitlinkEntry or mappings"
                )
            if child.depth != self.depth + 1:
                # Re-bind depth to enforce recursive depth chain.
                child = ReasoningGitlinkEntry(
                    path=child.path,
                    commit_id=child.commit_id,
                    depth=self.depth + 1,
                    mode=child.mode,
                    head_object_id=child.head_object_id,
                    index_object_id=child.index_object_id,
                    parent_path=self.path,
                    nested=child.nested,
                )
            normalized_nested.append(child)
        object.__setattr__(
            self,
            "nested",
            tuple(sorted(normalized_nested, key=lambda item: item.path)),
        )
        _bounded(self, "reasoning gitlink entry")

    def flatten(self) -> tuple["ReasoningGitlinkEntry", ...]:
        out: list[ReasoningGitlinkEntry] = [self]
        for child in self.nested:
            out.extend(child.flatten())
        return tuple(out)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPOSITORY_REASONING_SNAPSHOT_VERSION,
            "path": self.path,
            "commit_id": self.commit_id,
            "depth": self.depth,
            "mode": self.mode,
            "head_object_id": self.head_object_id,
            "index_object_id": self.index_object_id,
            "parent_path": self.parent_path,
            "nested": [item.to_dict() for item in self.nested],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningGitlinkEntry":
        fields = (
            "path",
            "commit_id",
            "depth",
            "mode",
            "head_object_id",
            "index_object_id",
            "parent_path",
            "nested",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "reasoning gitlink entry")
        nested_raw = values.pop("nested", ())
        nested: list[ReasoningGitlinkEntry] = []
        if nested_raw:
            if not isinstance(nested_raw, Sequence) or isinstance(
                nested_raw, (str, bytes, bytearray)
            ):
                raise RepositoryReasoningSnapshotError("nested must be a sequence")
            for item in nested_raw:
                if isinstance(item, Mapping):
                    nested.append(cls.from_dict(item))
                else:
                    raise RepositoryReasoningSnapshotError(
                        "nested gitlink items must be mappings"
                    )
        value = cls(nested=tuple(nested), **values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class TaskSourceBinding(CanonicalContract):
    """Task-source revision, status, evidence, and event cursor binding."""

    SCHEMA: ClassVar[str] = TASK_SOURCE_BINDING_SCHEMA

    revision: int
    status: str
    evidence_id: str = ""
    event_cursor: str = ""
    plan_root: str = ""
    board_namespace: str = ""
    source_kind: str = ""
    task_population_id: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "revision", _nonneg_int(self.revision, "revision", maximum=2**63 - 1)
        )
        if self.revision < 1:
            raise RepositoryReasoningSnapshotError("task-source revision must be >= 1")
        object.__setattr__(
            self, "status", _text(self.status, "status", required=True, limit=128)
        )
        object.__setattr__(
            self, "evidence_id", _optional_identifier(self.evidence_id, "evidence_id")
        )
        object.__setattr__(
            self,
            "event_cursor",
            _optional_identifier(self.event_cursor, "event_cursor"),
        )
        object.__setattr__(
            self, "plan_root", _optional_identifier(self.plan_root, "plan_root")
        )
        object.__setattr__(
            self,
            "board_namespace",
            _text(self.board_namespace, "board_namespace", limit=256),
        )
        object.__setattr__(
            self, "source_kind", _text(self.source_kind, "source_kind", limit=64)
        )
        object.__setattr__(
            self,
            "task_population_id",
            _optional_identifier(self.task_population_id, "task_population_id"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _string_tuple(self.evidence_refs, "evidence_refs", limit=MAX_REASON_CODES),
        )
        _bounded(self, "task-source binding")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPOSITORY_REASONING_SNAPSHOT_VERSION,
            "revision": self.revision,
            "status": self.status,
            "evidence_id": self.evidence_id,
            "event_cursor": self.event_cursor,
            "plan_root": self.plan_root,
            "board_namespace": self.board_namespace,
            "source_kind": self.source_kind,
            "task_population_id": self.task_population_id,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSourceBinding":
        fields = (
            "revision",
            "status",
            "evidence_id",
            "event_cursor",
            "plan_root",
            "board_namespace",
            "source_kind",
            "task_population_id",
            "evidence_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "task-source binding")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class ReasoningStability(CanonicalContract):
    """Stability / instability witnesses for one snapshot construction."""

    SCHEMA: ClassVar[str] = REASONING_STABILITY_SCHEMA

    stable: bool = True
    instability_codes: tuple[str, ...] = ()
    preflight_digest: str = ""
    postflight_digest: str = ""
    witnesses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stable", _bool(self.stable, "stable"))
        object.__setattr__(
            self,
            "instability_codes",
            _string_tuple(self.instability_codes, "instability_codes"),
        )
        object.__setattr__(
            self,
            "preflight_digest",
            _optional_identifier(self.preflight_digest, "preflight_digest"),
        )
        object.__setattr__(
            self,
            "postflight_digest",
            _optional_identifier(self.postflight_digest, "postflight_digest"),
        )
        object.__setattr__(
            self, "witnesses", _string_tuple(self.witnesses, "witnesses")
        )
        if self.stable and self.instability_codes:
            raise RepositoryReasoningSnapshotError(
                "stable snapshots cannot list instability codes"
            )
        if not self.stable and not self.instability_codes:
            raise RepositoryReasoningSnapshotError(
                "unstable snapshots require at least one instability code"
            )
        if (
            self.preflight_digest
            and self.postflight_digest
            and self.preflight_digest != self.postflight_digest
            and self.stable
        ):
            raise RepositoryReasoningInstabilityError(
                "preflight and postflight digests disagree on a claimed-stable snapshot"
            )
        _bounded(self, "reasoning stability")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPOSITORY_REASONING_SNAPSHOT_VERSION,
            "stable": self.stable,
            "instability_codes": list(self.instability_codes),
            "preflight_digest": self.preflight_digest,
            "postflight_digest": self.postflight_digest,
            "witnesses": list(self.witnesses),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningStability":
        fields = (
            "stable",
            "instability_codes",
            "preflight_digest",
            "postflight_digest",
            "witnesses",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "reasoning stability")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class ReasoningTruncation(CanonicalContract):
    """Declared truncation bounds and reasons for one snapshot."""

    SCHEMA: ClassVar[str] = REASONING_TRUNCATION_SCHEMA

    truncated: bool = False
    reasons: tuple[str, ...] = ()
    max_paths: int = MAX_PATH_ENTRIES
    max_gitlinks: int = MAX_GITLINK_ENTRIES
    max_bytes: int = MAX_RECORD_BYTES
    omitted_path_count: int = 0
    omitted_gitlink_count: int = 0
    omitted_symbol_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "truncated", _bool(self.truncated, "truncated"))
        object.__setattr__(
            self, "reasons", _string_tuple(self.reasons, "reasons")
        )
        for name in ("max_paths", "max_gitlinks", "max_bytes"):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), name, maximum=2**31 - 1),
            )
            if getattr(self, name) < 1:
                raise RepositoryReasoningBoundsError(f"{name} must be positive")
        for name in (
            "omitted_path_count",
            "omitted_gitlink_count",
            "omitted_symbol_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), name, maximum=2**31 - 1),
            )
        if self.truncated and not self.reasons:
            raise RepositoryReasoningSnapshotError(
                "truncated snapshots require at least one reason"
            )
        if not self.truncated and (
            self.omitted_path_count
            or self.omitted_gitlink_count
            or self.omitted_symbol_count
        ):
            raise RepositoryReasoningSnapshotError(
                "non-truncated snapshots cannot record omitted counts"
            )
        if not self.truncated and self.reasons:
            raise RepositoryReasoningSnapshotError(
                "non-truncated snapshots cannot list truncation reasons"
            )
        _bounded(self, "reasoning truncation")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPOSITORY_REASONING_SNAPSHOT_VERSION,
            "truncated": self.truncated,
            "reasons": list(self.reasons),
            "max_paths": self.max_paths,
            "max_gitlinks": self.max_gitlinks,
            "max_bytes": self.max_bytes,
            "omitted_path_count": self.omitted_path_count,
            "omitted_gitlink_count": self.omitted_gitlink_count,
            "omitted_symbol_count": self.omitted_symbol_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningTruncation":
        fields = (
            "truncated",
            "reasons",
            "max_paths",
            "max_gitlinks",
            "max_bytes",
            "omitted_path_count",
            "omitted_gitlink_count",
            "omitted_symbol_count",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "reasoning truncation")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Top-level snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepositoryReasoningSnapshot(CanonicalContract):
    """Canonical content-addressed repository reasoning snapshot.

    Serves Planner and Doctor without collapsing their evidence tiers.  Binds:

    * superproject tree and dirty overlay path ledger
      (tracked / staged / modified / deleted / renamed / admitted-untracked);
    * recursive gitlink / submodule closure;
    * explicit exclusions;
    * instability and truncation witnesses;
    * task-source revision / status / evidence / event cursor; and
    * parser, index, toolchain, capability, policy, and IR roots.
    """

    SCHEMA: ClassVar[str] = REPOSITORY_REASONING_SNAPSHOT_SCHEMA

    roots: ReasoningToolRoots
    paths: tuple[ReasoningPathEntry, ...]
    gitlinks: tuple[ReasoningGitlinkEntry, ...] = ()
    exclusions: tuple[str, ...] = ()
    task_source: TaskSourceBinding | None = None
    stability: ReasoningStability = field(default_factory=ReasoningStability)
    truncation: ReasoningTruncation = field(default_factory=ReasoningTruncation)
    primary_root: str = "."
    scope_id: str = ""
    dirty_overlay_id: str = ""
    completeness: str = "complete"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _coerce_roots(self.roots))
        paths = _coerce_path_entries(self.paths)
        if len(paths) > MAX_PATH_ENTRIES:
            raise RepositoryReasoningBoundsError("path entry count exceeds max_paths")
        path_keys = [item.path for item in paths]
        if len(path_keys) != len(set(path_keys)):
            raise RepositoryReasoningSnapshotError("path entries must be unique by path")
        object.__setattr__(self, "paths", paths)

        gitlinks = _coerce_gitlink_entries(self.gitlinks)
        flat = []
        for item in gitlinks:
            flat.extend(item.flatten())
        if len(flat) > MAX_GITLINK_ENTRIES:
            raise RepositoryReasoningBoundsError(
                "gitlink entry count exceeds max_gitlinks"
            )
        gitlink_paths = [item.path for item in flat]
        if len(gitlink_paths) != len(set(gitlink_paths)):
            raise RepositoryReasoningSnapshotError(
                "gitlink entries must be unique by path across the recursive closure"
            )
        object.__setattr__(self, "gitlinks", gitlinks)

        object.__setattr__(
            self,
            "exclusions",
            _string_tuple(
                self.exclusions,
                "exclusions",
                limit=MAX_EXCLUSIONS,
                sort=True,
            ),
        )
        # Normalize exclusion paths
        object.__setattr__(
            self,
            "exclusions",
            tuple(_repo_path(path, "exclusions") for path in self.exclusions),
        )

        if self.task_source is None:
            object.__setattr__(self, "task_source", None)
        elif isinstance(self.task_source, TaskSourceBinding):
            object.__setattr__(self, "task_source", self.task_source)
        elif isinstance(self.task_source, Mapping):
            object.__setattr__(
                self, "task_source", TaskSourceBinding.from_dict(self.task_source)
            )
        else:
            raise RepositoryReasoningSnapshotError(
                "task_source must be TaskSourceBinding, mapping, or None"
            )

        object.__setattr__(self, "stability", _coerce_stability(self.stability))
        object.__setattr__(self, "truncation", _coerce_truncation(self.truncation))
        object.__setattr__(
            self,
            "primary_root",
            _repo_path(self.primary_root, "primary_root", allow_root=True),
        )
        object.__setattr__(
            self, "scope_id", _optional_identifier(self.scope_id, "scope_id")
        )
        object.__setattr__(
            self,
            "dirty_overlay_id",
            _optional_identifier(self.dirty_overlay_id, "dirty_overlay_id"),
        )
        completeness = _text(
            self.completeness, "completeness", required=True, limit=64
        )
        if completeness not in {"complete", "partial_with_frontier", "abstained"}:
            raise RepositoryReasoningSnapshotError(
                "completeness must be complete, partial_with_frontier, or abstained"
            )
        object.__setattr__(self, "completeness", completeness)
        object.__setattr__(
            self, "notes", _string_tuple(self.notes, "notes", limit=MAX_REASON_CODES)
        )

        if self.truncation.truncated and self.completeness == "complete":
            raise RepositoryReasoningSnapshotError(
                "truncated snapshots cannot claim completeness=complete"
            )
        if not self.stability.stable and self.completeness == "complete":
            raise RepositoryReasoningSnapshotError(
                "unstable snapshots cannot claim completeness=complete"
            )

        # Excluded path ledger must cover every EXCLUDED status entry.
        excluded_paths = {
            item.path
            for item in self.paths
            if item.coverage is ReasoningCoverageKind.EXCLUDED
            or item.status is ReasoningPathStatus.EXCLUDED
        }
        for path in excluded_paths:
            if path not in self.exclusions and not any(
                path == item or path.startswith(item.rstrip("/") + "/")
                for item in self.exclusions
            ):
                # Path-level exclusion reason is sufficient when reason_code set.
                entry = next(item for item in self.paths if item.path == path)
                if not entry.reason_code:
                    raise RepositoryReasoningSnapshotError(
                        f"excluded path {path!r} lacks exclusion reason or list entry"
                    )

        _bounded(self, "repository reasoning snapshot")

    # -- projections --------------------------------------------------------

    def paths_with_status(
        self, *statuses: ReasoningPathStatus | str
    ) -> tuple[ReasoningPathEntry, ...]:
        wanted = {
            _enum(item, ReasoningPathStatus, "status") for item in statuses
        }
        return tuple(item for item in self.paths if item.status in wanted)

    def tracked_paths(self) -> tuple[ReasoningPathEntry, ...]:
        return tuple(item for item in self.paths if item.tracked)

    def admitted_untracked_paths(self) -> tuple[ReasoningPathEntry, ...]:
        return self.paths_with_status(ReasoningPathStatus.ADMITTED_UNTRACKED)

    def renamed_paths(self) -> tuple[ReasoningPathEntry, ...]:
        return self.paths_with_status(ReasoningPathStatus.RENAMED)

    def deleted_paths(self) -> tuple[ReasoningPathEntry, ...]:
        return self.paths_with_status(
            ReasoningPathStatus.DELETED, ReasoningPathStatus.STAGED_DELETION
        )

    def staged_paths(self) -> tuple[ReasoningPathEntry, ...]:
        return self.paths_with_status(
            ReasoningPathStatus.STAGED,
            ReasoningPathStatus.STAGED_AND_MODIFIED,
            ReasoningPathStatus.STAGED_DELETION,
        )

    def modified_paths(self) -> tuple[ReasoningPathEntry, ...]:
        return self.paths_with_status(
            ReasoningPathStatus.MODIFIED, ReasoningPathStatus.STAGED_AND_MODIFIED
        )

    def recursive_gitlinks(self) -> tuple[ReasoningGitlinkEntry, ...]:
        out: list[ReasoningGitlinkEntry] = []
        for item in self.gitlinks:
            out.extend(item.flatten())
        return tuple(sorted(out, key=lambda item: (item.depth, item.path)))

    def assert_stable(self) -> None:
        if not self.stability.stable:
            raise RepositoryReasoningInstabilityError(
                "repository reasoning snapshot is unstable: "
                + ",".join(self.stability.instability_codes)
            )

    def require_repository(self, repository_id: str) -> None:
        expected = _identifier(repository_id, "repository_id")
        if self.roots.repository_id != expected:
            raise RepositoryReasoningAuthorityError(
                "cross-repository replay is rejected"
            )

    def inventory(self) -> dict[str, Any]:
        """Compact inventory suitable for coverage and health classification."""

        return {
            "path_count": len(self.paths),
            "tracked_count": len(self.tracked_paths()),
            "staged_count": len(self.staged_paths()),
            "modified_count": len(self.modified_paths()),
            "deleted_count": len(self.deleted_paths()),
            "renamed_count": len(self.renamed_paths()),
            "admitted_untracked_count": len(self.admitted_untracked_paths()),
            "excluded_count": len(self.exclusions),
            "gitlink_count": len(self.recursive_gitlinks()),
            "stable": self.stability.stable,
            "truncated": self.truncation.truncated,
            "completeness": self.completeness,
            "task_source_revision": (
                self.task_source.revision if self.task_source is not None else 0
            ),
            "snapshot_id": self.snapshot_id,
            "content_id": self.content_id,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPOSITORY_REASONING_SNAPSHOT_VERSION,
            "roots": self.roots.to_dict(),
            "paths": [item.to_dict() for item in self.paths],
            "gitlinks": [item.to_dict() for item in self.gitlinks],
            "exclusions": list(self.exclusions),
            "task_source": (
                self.task_source.to_dict() if self.task_source is not None else None
            ),
            "stability": self.stability.to_dict(),
            "truncation": self.truncation.to_dict(),
            "primary_root": self.primary_root,
            "scope_id": self.scope_id,
            "dirty_overlay_id": self.dirty_overlay_id,
            "completeness": self.completeness,
            "notes": list(self.notes),
        }

    @property
    def snapshot_id(self) -> str:
        return f"repository-reasoning-snapshot:{self.content_id}"

    def to_dict(self) -> dict[str, Any]:
        # Identity-bearing payload only (no content_id / inventory recursion).
        return {"schema": self.SCHEMA, **self._payload()}

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "snapshot_id": self.snapshot_id,
            "content_id": self.content_id,
            "inventory": self.inventory(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryReasoningSnapshot":
        fields = (
            "roots",
            "paths",
            "gitlinks",
            "exclusions",
            "task_source",
            "stability",
            "truncation",
            "primary_root",
            "scope_id",
            "dirty_overlay_id",
            "completeness",
            "notes",
        )
        values = _decode_fields(
            payload,
            cls.SCHEMA,
            fields,
            "repository reasoning snapshot",
            extra_allowed=frozenset({"inventory"}),
        )
        values["roots"] = _coerce_roots(values["roots"])
        values["paths"] = _coerce_path_entries(values.get("paths") or ())
        values["gitlinks"] = _coerce_gitlink_entries(values.get("gitlinks") or ())
        if values.get("task_source") is not None:
            values["task_source"] = _coerce_task_source(values["task_source"])
        if "stability" in values:
            values["stability"] = _coerce_stability(values["stability"])
        if "truncation" in values:
            values["truncation"] = _coerce_truncation(values["truncation"])
        value = cls(**values)
        _verify_identity(payload, value)
        supplied_snapshot_id = payload.get("snapshot_id")
        if supplied_snapshot_id not in (None, "") and supplied_snapshot_id != value.snapshot_id:
            raise RepositoryReasoningTamperError(
                "stored snapshot_id does not match the canonical record"
            )
        return value


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_roots(value: Any) -> ReasoningToolRoots:
    if isinstance(value, ReasoningToolRoots):
        return value
    if isinstance(value, Mapping):
        if value.get("schema") == REASONING_TOOL_ROOTS_SCHEMA:
            return ReasoningToolRoots.from_dict(value)
        return ReasoningToolRoots(**{
            key: value[key]
            for key in (
                "repository_id",
                "forest_id",
                "tree_id",
                "overlay_id",
                "head_commit_id",
                "head_tree_id",
                "index_tree_id",
                "parser_root",
                "index_root",
                "toolchain_root",
                "capability_root",
                "policy_root",
                "ir_root",
                "intent_ir_root",
                "legal_ir_root",
                "security_ir_root",
                "program_behavior_root",
                "ast_root",
                "evidence_graph_root",
                "vector_root",
                "cache_root",
                "scope_policy_id",
                "scanner_root",
            )
            if key in value
        })
    raise RepositoryReasoningSnapshotError(
        "roots must be ReasoningToolRoots or a mapping"
    )


def _coerce_path_entries(value: Any) -> tuple[ReasoningPathEntry, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RepositoryReasoningSnapshotError("paths must be a sequence")
    out: list[ReasoningPathEntry] = []
    for item in value:
        if isinstance(item, ReasoningPathEntry):
            out.append(item)
        elif isinstance(item, Mapping):
            if item.get("schema") == REASONING_PATH_ENTRY_SCHEMA:
                out.append(ReasoningPathEntry.from_dict(item))
            else:
                payload = dict(item)
                payload.pop("content_id", None)
                payload.pop("cid", None)
                payload.pop("schema", None)
                payload.pop("contract_version", None)
                out.append(ReasoningPathEntry(**payload))
        else:
            raise RepositoryReasoningSnapshotError(
                "path entries must be ReasoningPathEntry or mappings"
            )
    return tuple(sorted(out, key=lambda item: item.path))


def _coerce_gitlink_entries(value: Any) -> tuple[ReasoningGitlinkEntry, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RepositoryReasoningSnapshotError("gitlinks must be a sequence")
    out: list[ReasoningGitlinkEntry] = []
    for item in value:
        if isinstance(item, ReasoningGitlinkEntry):
            out.append(item)
        elif isinstance(item, Mapping):
            if item.get("schema") == REASONING_GITLINK_ENTRY_SCHEMA:
                out.append(ReasoningGitlinkEntry.from_dict(item))
            else:
                payload = dict(item)
                payload.pop("content_id", None)
                payload.pop("cid", None)
                payload.pop("schema", None)
                payload.pop("contract_version", None)
                payload.pop("gitlink_id", None)
                nested = payload.pop("nested", ())
                out.append(
                    ReasoningGitlinkEntry(
                        nested=_coerce_gitlink_entries(nested),
                        **payload,
                    )
                )
        else:
            raise RepositoryReasoningSnapshotError(
                "gitlink entries must be ReasoningGitlinkEntry or mappings"
            )
    return tuple(sorted(out, key=lambda item: item.path))


def _coerce_task_source(value: Any) -> TaskSourceBinding | None:
    if value is None:
        return None
    if isinstance(value, TaskSourceBinding):
        return value
    if isinstance(value, Mapping):
        if value.get("schema") == TASK_SOURCE_BINDING_SCHEMA:
            return TaskSourceBinding.from_dict(value)
        payload = dict(value)
        payload.pop("content_id", None)
        payload.pop("cid", None)
        payload.pop("schema", None)
        payload.pop("contract_version", None)
        return TaskSourceBinding(**payload)
    raise RepositoryReasoningSnapshotError(
        "task_source must be TaskSourceBinding, mapping, or None"
    )


def _coerce_stability(value: Any) -> ReasoningStability:
    if isinstance(value, ReasoningStability):
        return value
    if value is None:
        return ReasoningStability()
    if isinstance(value, Mapping):
        if value.get("schema") == REASONING_STABILITY_SCHEMA:
            return ReasoningStability.from_dict(value)
        payload = dict(value)
        payload.pop("content_id", None)
        payload.pop("cid", None)
        payload.pop("schema", None)
        payload.pop("contract_version", None)
        return ReasoningStability(**payload)
    raise RepositoryReasoningSnapshotError(
        "stability must be ReasoningStability or a mapping"
    )


def _coerce_truncation(value: Any) -> ReasoningTruncation:
    if isinstance(value, ReasoningTruncation):
        return value
    if value is None:
        return ReasoningTruncation()
    if isinstance(value, Mapping):
        if value.get("schema") == REASONING_TRUNCATION_SCHEMA:
            return ReasoningTruncation.from_dict(value)
        payload = dict(value)
        payload.pop("content_id", None)
        payload.pop("cid", None)
        payload.pop("schema", None)
        payload.pop("contract_version", None)
        return ReasoningTruncation(**payload)
    raise RepositoryReasoningSnapshotError(
        "truncation must be ReasoningTruncation or a mapping"
    )


# ---------------------------------------------------------------------------
# Status mapping from existing SCA / program-behavior enumerations
# ---------------------------------------------------------------------------


_GIT_STATUS_TO_REASONING: Final[Mapping[str, ReasoningPathStatus]] = {
    "clean": ReasoningPathStatus.CLEAN,
    "tracked": ReasoningPathStatus.TRACKED,
    "modified": ReasoningPathStatus.MODIFIED,
    "staged": ReasoningPathStatus.STAGED,
    "staged_and_modified": ReasoningPathStatus.STAGED_AND_MODIFIED,
    "deleted": ReasoningPathStatus.DELETED,
    "staged_deletion": ReasoningPathStatus.STAGED_DELETION,
    "untracked": ReasoningPathStatus.ADMITTED_UNTRACKED,
    "admitted_untracked": ReasoningPathStatus.ADMITTED_UNTRACKED,
    "renamed": ReasoningPathStatus.RENAMED,
    "mode_changed": ReasoningPathStatus.MODE_CHANGED,
    "excluded": ReasoningPathStatus.EXCLUDED,
}


def map_git_status(value: Any) -> ReasoningPathStatus:
    """Map SCA / program-behavior git status strings onto reasoning statuses."""

    if isinstance(value, ReasoningPathStatus):
        return value
    text = str(getattr(value, "value", value) or "").strip().lower()
    if text in _GIT_STATUS_TO_REASONING:
        return _GIT_STATUS_TO_REASONING[text]
    if text in _PATH_STATUS_VALUES:
        return ReasoningPathStatus(text)
    raise RepositoryReasoningSnapshotError(f"unsupported git status: {text!r}")


def path_entry_from_sca_disposition(disposition: Any) -> ReasoningPathEntry:
    """Bridge an SCA ``CoverageDisposition`` into a reasoning path entry.

    Does not alias schemas: produces a new ``ReasoningPathEntry`` identity.
    """

    path = getattr(disposition, "path", None)
    if path is None and isinstance(disposition, Mapping):
        path = disposition.get("path")
    git_status = getattr(disposition, "git_status", None)
    if git_status is None and isinstance(disposition, Mapping):
        git_status = disposition.get("git_status")
    entry_kind_raw = getattr(disposition, "entry_kind", "regular")
    if isinstance(disposition, Mapping):
        entry_kind_raw = disposition.get("entry_kind", entry_kind_raw)
    kind_raw = getattr(disposition, "kind", "admitted")
    if isinstance(disposition, Mapping):
        kind_raw = disposition.get("kind", kind_raw)
    kind_text = str(getattr(kind_raw, "value", kind_raw) or "admitted").lower()
    if kind_text in {"excluded"}:
        coverage = ReasoningCoverageKind.EXCLUDED
    elif kind_text in {"gitlink"}:
        coverage = ReasoningCoverageKind.GITLINK
    elif kind_text in {"parse_failure"}:
        coverage = ReasoningCoverageKind.PARSE_FAILURE
    elif kind_text in {"unsupported", "unsupported_language"}:
        coverage = ReasoningCoverageKind.UNSUPPORTED
    elif kind_text in {"dependency", "lockfile", "manifest"}:
        coverage = ReasoningCoverageKind.DEPENDENCY
    else:
        coverage = ReasoningCoverageKind.ADMITTED

    entry_kind_text = str(getattr(entry_kind_raw, "value", entry_kind_raw) or "regular")
    try:
        entry_kind = ReasoningEntryKind(entry_kind_text)
    except ValueError:
        entry_kind = ReasoningEntryKind.REGULAR

    status = map_git_status(git_status)
    if coverage is ReasoningCoverageKind.EXCLUDED:
        status = ReasoningPathStatus.EXCLUDED

    tracked = bool(
        getattr(disposition, "tracked", True)
        if not isinstance(disposition, Mapping)
        else disposition.get("tracked", True)
    )
    if status is ReasoningPathStatus.ADMITTED_UNTRACKED:
        tracked = False

    def _attr(name: str, default: str = "") -> str:
        if isinstance(disposition, Mapping):
            return str(disposition.get(name, default) or default)
        return str(getattr(disposition, name, default) or default)

    return ReasoningPathEntry(
        path=str(path),
        status=status,
        coverage=coverage,
        entry_kind=entry_kind,
        tracked=tracked,
        overlay=bool(
            getattr(disposition, "overlay", False)
            if not isinstance(disposition, Mapping)
            else disposition.get("overlay", False)
        ),
        worktree_digest=_attr("content_digest"),
        rename_from=_attr("rename_from"),
        reason_code=_attr("reason_code"),
        policy_rule=_attr("policy_rule"),
        git_mode=_attr("git_mode"),
        git_object_id=_attr("git_object_id"),
    )


def gitlink_from_sca_record(record: Any, *, depth: int = 0) -> ReasoningGitlinkEntry:
    """Bridge an SCA ``GitlinkRecord`` into a reasoning gitlink entry."""

    def _attr(name: str, default: str = "") -> str:
        if isinstance(record, Mapping):
            return str(record.get(name, default) or default)
        return str(getattr(record, name, default) or default)

    return ReasoningGitlinkEntry(
        path=_attr("path"),
        commit_id=_attr("commit_id") or _attr("head_object_id"),
        depth=depth,
        mode=_attr("mode", "160000") or "160000",
        head_object_id=_attr("head_object_id"),
        index_object_id=_attr("index_object_id"),
    )


def build_repository_reasoning_snapshot(
    *,
    roots: ReasoningToolRoots | Mapping[str, Any],
    paths: Sequence[ReasoningPathEntry | Mapping[str, Any]] = (),
    gitlinks: Sequence[ReasoningGitlinkEntry | Mapping[str, Any]] = (),
    exclusions: Sequence[str] = (),
    task_source: TaskSourceBinding | Mapping[str, Any] | None = None,
    stability: ReasoningStability | Mapping[str, Any] | None = None,
    truncation: ReasoningTruncation | Mapping[str, Any] | None = None,
    primary_root: str = ".",
    scope_id: str = "",
    dirty_overlay_id: str = "",
    completeness: str = "complete",
    notes: Sequence[str] = (),
) -> RepositoryReasoningSnapshot:
    """Construct a validated repository reasoning snapshot."""

    return RepositoryReasoningSnapshot(
        roots=_coerce_roots(roots),
        paths=_coerce_path_entries(paths),
        gitlinks=_coerce_gitlink_entries(gitlinks),
        exclusions=tuple(exclusions),
        task_source=_coerce_task_source(task_source),
        stability=_coerce_stability(stability),
        truncation=_coerce_truncation(truncation),
        primary_root=primary_root,
        scope_id=scope_id,
        dirty_overlay_id=dirty_overlay_id,
        completeness=completeness,
        notes=tuple(notes),
    )


def reasoning_snapshot_from_sca_snapshot(
    sca_snapshot: Any,
    *,
    roots: ReasoningToolRoots | Mapping[str, Any],
    task_source: TaskSourceBinding | Mapping[str, Any] | None = None,
    stability: ReasoningStability | Mapping[str, Any] | None = None,
    truncation: ReasoningTruncation | Mapping[str, Any] | None = None,
    recursive_gitlinks: Sequence[ReasoningGitlinkEntry | Mapping[str, Any]] | None = None,
    notes: Sequence[str] = (),
) -> RepositoryReasoningSnapshot:
    """Project an SCA ``RepositorySnapshot`` into the reasoning snapshot.

    Preserves the existing SCA schema identity on the source object; this
    function only builds a checked bridge projection.
    """

    dispositions = getattr(sca_snapshot, "dispositions", ()) or ()
    path_entries = [path_entry_from_sca_disposition(item) for item in dispositions]
    if recursive_gitlinks is not None:
        gitlinks = list(recursive_gitlinks)
    else:
        sca_gitlinks = getattr(sca_snapshot, "gitlinks", ()) or ()
        gitlinks = [gitlink_from_sca_record(item) for item in sca_gitlinks]

    exclusions = [
        item.path
        for item in path_entries
        if item.coverage is ReasoningCoverageKind.EXCLUDED
        or item.status is ReasoningPathStatus.EXCLUDED
    ]
    for item in getattr(sca_snapshot, "excluded_paths", ()) or ():
        exclusions.append(str(item))

    root_obj = _coerce_roots(roots)
    # Bind HEAD/index tree ids from the SCA snapshot when not already set.
    updates: dict[str, str] = {}
    for attr, field_name in (
        ("head_commit_id", "head_commit_id"),
        ("head_tree_id", "head_tree_id"),
        ("index_tree_id", "index_tree_id"),
    ):
        current = getattr(root_obj, field_name)
        source = str(getattr(sca_snapshot, attr, "") or "")
        if not current and source:
            updates[field_name] = source
    if updates:
        payload = root_obj._payload()
        payload.update(updates)
        payload.pop("contract_version", None)
        root_obj = ReasoningToolRoots(**payload)

    completeness = "complete"
    trunc = _coerce_truncation(truncation)
    if trunc.truncated:
        completeness = "partial_with_frontier"
    stab = _coerce_stability(stability)
    if not stab.stable:
        completeness = "partial_with_frontier"

    scope_id = str(getattr(sca_snapshot, "scope_id", "") or "")
    primary_root = str(getattr(sca_snapshot, "primary_root", ".") or ".")
    overlay = str(getattr(sca_snapshot, "snapshot_id", "") or "")

    return build_repository_reasoning_snapshot(
        roots=root_obj,
        paths=path_entries,
        gitlinks=gitlinks,
        exclusions=exclusions,
        task_source=task_source,
        stability=stab,
        truncation=trunc,
        primary_root=primary_root,
        scope_id=scope_id,
        dirty_overlay_id=overlay,
        completeness=completeness,
        notes=notes,
    )


__all__ = [
    "MAX_EXCLUSIONS",
    "MAX_GITLINK_DEPTH",
    "MAX_GITLINK_ENTRIES",
    "MAX_PATH_ENTRIES",
    "REASONING_GITLINK_ENTRY_SCHEMA",
    "REASONING_PATH_ENTRY_SCHEMA",
    "REASONING_STABILITY_SCHEMA",
    "REASONING_TOOL_ROOTS_SCHEMA",
    "REASONING_TRUNCATION_SCHEMA",
    "REPOSITORY_REASONING_SNAPSHOT_INTERFACE",
    "REPOSITORY_REASONING_SNAPSHOT_SCHEMA",
    "REPOSITORY_REASONING_SNAPSHOT_VERSION",
    "TASK_SOURCE_BINDING_SCHEMA",
    "ReasoningCoverageKind",
    "ReasoningEntryKind",
    "ReasoningGitlinkEntry",
    "ReasoningPathEntry",
    "ReasoningPathStatus",
    "ReasoningStability",
    "ReasoningToolRoots",
    "ReasoningTruncation",
    "RepositoryReasoningAuthorityError",
    "RepositoryReasoningBoundsError",
    "RepositoryReasoningInstabilityError",
    "RepositoryReasoningSnapshot",
    "RepositoryReasoningSnapshotError",
    "RepositoryReasoningTamperError",
    "TaskSourceBinding",
    "build_repository_reasoning_snapshot",
    "gitlink_from_sca_record",
    "map_git_status",
    "path_entry_from_sca_disposition",
    "reasoning_snapshot_from_sca_snapshot",
]
