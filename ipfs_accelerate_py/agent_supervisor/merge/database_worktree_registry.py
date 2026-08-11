"""Persist repositories, branches, worktrees, snapshots, and dirty overlays.

DQP-016 / RepositoryForest@1, WorktreeIdentity@1, WorktreeSnapshot@1, DirtyOverlay@1
===================================================================================

:class:`DatabaseWorktreeRegistry` is the durable **semantic** authority for
repository forest membership, worktree identity, leases, lifecycle transitions,
setup cache, base/head/tree/index snapshots, path inventories, and dirty
overlays shared across parallel lanes.

Authority rules (fail-closed)
-----------------------------
* Git remains the **byte** authority: object IDs, index digests, and current
  worktree observations are never invented by the registry.
* DB history is the **semantic** authority: ownership, lease, fence, lifecycle
  state, and path policy live only in registry rows.
* Worktree reuse and cleanup require a matching lease **and** current Git
  observations that reconcile with the registered snapshot.
* Stale/dead owner recovery uses compare-and-swap against the fencing token.
* A worktree-local JSON index may mirror rows for diagnostics but cannot create,
  extend, or override registry state.

Cold import of this module performs no filesystem, database, network, provider,
or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    current_process_birth,
    owner_liveness,
    read_process_birth,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_WORKTREE_REGISTRY_INTERFACE: Final[str] = "DatabaseWorktreeRegistry@1"
REPOSITORY_FOREST_INTERFACE: Final[str] = "RepositoryForest@1"
WORKTREE_IDENTITY_INTERFACE: Final[str] = "WorktreeIdentity@1"
WORKTREE_SNAPSHOT_INTERFACE: Final[str] = "WorktreeSnapshot@1"
DIRTY_OVERLAY_INTERFACE: Final[str] = "DirtyOverlay@1"

DATABASE_WORKTREE_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-worktree-registry@1"
)
REPOSITORY_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repository-record@1"
)
BRANCH_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/branch-record@1"
)
GIT_REF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/git-ref@1"
)
SUBMODULE_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/submodule-edge@1"
)
WORKTREE_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worktree-identity@1"
)
WORKTREE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worktree-snapshot@1"
)
DIRTY_OVERLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dirty-overlay@1"
)
WORKTREE_PATH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worktree-path@1"
)
SETUP_CACHE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worktree-setup-cache@1"
)
LEASE_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worktree-lease-event@1"
)
LIFECYCLE_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worktree-lifecycle-transition@1"
)
GIT_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/git-observation@1"
)
LOCAL_JSON_MIRROR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worktree-local-json-mirror@1"
)

DEFAULT_LEASE_TTL_MS: Final[int] = 21_600_000
MAX_PAYLOAD_BYTES: Final[int] = 262_144
MAX_PATH_BYTES: Final[int] = 4_096
MAX_OVERLAY_ENTRIES: Final[int] = 65_536

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS registry_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS repositories (
    repository_id VARCHAR PRIMARY KEY,
    git_common_dir VARCHAR NOT NULL,
    canonical_root VARCHAR NOT NULL,
    remote_url VARCHAR NOT NULL DEFAULT '',
    head_commit VARCHAR NOT NULL DEFAULT '',
    head_tree VARCHAR NOT NULL DEFAULT '',
    server_generation BIGINT NOT NULL DEFAULT 1,
    registered_at_ms BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS repositories_common_dir_idx
    ON repositories(git_common_dir);

CREATE TABLE IF NOT EXISTS repository_revisions (
    revision_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    head_commit VARCHAR NOT NULL DEFAULT '',
    head_tree VARCHAR NOT NULL DEFAULT '',
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS repository_revisions_repo_idx
    ON repository_revisions(repository_id, revision);

CREATE TABLE IF NOT EXISTS branches (
    branch_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    tip_commit VARCHAR NOT NULL DEFAULT '',
    is_detached BOOLEAN NOT NULL DEFAULT FALSE,
    upstream VARCHAR NOT NULL DEFAULT '',
    registered_at_ms BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS branches_repo_name_uidx
    ON branches(repository_id, branch_name);
CREATE INDEX IF NOT EXISTS branches_repo_idx
    ON branches(repository_id);

CREATE TABLE IF NOT EXISTS git_refs (
    ref_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    ref_name VARCHAR NOT NULL,
    object_id VARCHAR NOT NULL,
    ref_kind VARCHAR NOT NULL DEFAULT 'branch',
    registered_at_ms BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS git_refs_repo_name_uidx
    ON git_refs(repository_id, ref_name);

CREATE TABLE IF NOT EXISTS submodule_edges (
    edge_id VARCHAR PRIMARY KEY,
    parent_repository_id VARCHAR NOT NULL,
    child_repository_id VARCHAR NOT NULL,
    gitlink_path VARCHAR NOT NULL,
    gitlink_commit VARCHAR NOT NULL DEFAULT '',
    registered_at_ms BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS submodule_edges_parent_path_uidx
    ON submodule_edges(parent_repository_id, gitlink_path);

CREATE TABLE IF NOT EXISTS worktrees (
    worktree_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    workspace_path VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL DEFAULT '',
    lane_id VARCHAR NOT NULL DEFAULT '',
    task_id VARCHAR NOT NULL DEFAULT '',
    attempt BIGINT NOT NULL DEFAULT 0,
    session_id VARCHAR NOT NULL DEFAULT '',
    lease_id VARCHAR NOT NULL DEFAULT '',
    fencing_token BIGINT NOT NULL DEFAULT 0,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    owner_process_birth_id VARCHAR NOT NULL DEFAULT '',
    owner_process_birth_json VARCHAR NOT NULL DEFAULT '{}',
    lifecycle_state VARCHAR NOT NULL,
    lease_expires_at_ms BIGINT NOT NULL DEFAULT 0,
    head_commit VARCHAR NOT NULL DEFAULT '',
    head_tree VARCHAR NOT NULL DEFAULT '',
    index_digest VARCHAR NOT NULL DEFAULT '',
    dirty_overlay_digest VARCHAR NOT NULL DEFAULT '',
    current_snapshot_id VARCHAR NOT NULL DEFAULT '',
    is_detached BOOLEAN NOT NULL DEFAULT FALSE,
    registered_at_ms BIGINT NOT NULL,
    updated_at_ms BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS worktrees_path_uidx
    ON worktrees(workspace_path);
CREATE INDEX IF NOT EXISTS worktrees_repo_idx
    ON worktrees(repository_id, status);
CREATE INDEX IF NOT EXISTS worktrees_lease_idx
    ON worktrees(lease_id, fencing_token);
CREATE INDEX IF NOT EXISTS worktrees_lifecycle_idx
    ON worktrees(lifecycle_state, status);

CREATE TABLE IF NOT EXISTS worktree_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    base_commit VARCHAR NOT NULL DEFAULT '',
    head_commit VARCHAR NOT NULL DEFAULT '',
    head_tree VARCHAR NOT NULL DEFAULT '',
    index_digest VARCHAR NOT NULL DEFAULT '',
    dirty_overlay_digest VARCHAR NOT NULL DEFAULT '',
    branch_name VARCHAR NOT NULL DEFAULT '',
    is_detached BOOLEAN NOT NULL DEFAULT FALSE,
    scanner_version VARCHAR NOT NULL DEFAULT '',
    observed_at_ms BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS worktree_snapshots_worktree_idx
    ON worktree_snapshots(worktree_id, revision);

CREATE TABLE IF NOT EXISTS worktree_paths (
    path_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    relative_path VARCHAR NOT NULL,
    path_kind VARCHAR NOT NULL,
    blob_id VARCHAR NOT NULL DEFAULT '',
    symlink_target VARCHAR NOT NULL DEFAULT '',
    is_symlink BOOLEAN NOT NULL DEFAULT FALSE,
    is_gitlink BOOLEAN NOT NULL DEFAULT FALSE,
    policy_disposition VARCHAR NOT NULL DEFAULT 'tracked',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS worktree_paths_snapshot_idx
    ON worktree_paths(snapshot_id, relative_path);
CREATE INDEX IF NOT EXISTS worktree_paths_worktree_idx
    ON worktree_paths(worktree_id);

CREATE TABLE IF NOT EXISTS dirty_overlays (
    overlay_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    entry_count BIGINT NOT NULL DEFAULT 0,
    rename_policy VARCHAR NOT NULL DEFAULT 'track',
    delete_policy VARCHAR NOT NULL DEFAULT 'track',
    untracked_policy VARCHAR NOT NULL DEFAULT 'include',
    entries_json VARCHAR NOT NULL DEFAULT '[]',
    recorded_at_ms BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS dirty_overlays_snapshot_idx
    ON dirty_overlays(snapshot_id);
CREATE INDEX IF NOT EXISTS dirty_overlays_worktree_idx
    ON dirty_overlays(worktree_id, revision);

CREATE TABLE IF NOT EXISTS setup_cache (
    cache_key VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL DEFAULT '',
    cache_digest VARCHAR NOT NULL,
    payload_json VARCHAR NOT NULL DEFAULT '{}',
    recorded_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS setup_cache_repo_idx
    ON setup_cache(repository_id);

CREATE TABLE IF NOT EXISTS lease_events (
    event_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    event_kind VARCHAR NOT NULL,
    actor_session_id VARCHAR NOT NULL DEFAULT '',
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS lease_events_worktree_idx
    ON lease_events(worktree_id, recorded_at_ms);

CREATE TABLE IF NOT EXISTS lifecycle_transitions (
    transition_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    from_state VARCHAR NOT NULL,
    to_state VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS lifecycle_transitions_worktree_idx
    ON lifecycle_transitions(worktree_id, recorded_at_ms);

CREATE TABLE IF NOT EXISTS local_json_mirrors (
    mirror_path VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    written_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorktreeRegistryError(RuntimeError):
    """Base error for database worktree registry failures."""


class WorktreeRegistryConflictError(WorktreeRegistryError):
    """Lease/fence mismatch, duplicate ownership, or CAS failure."""


class WorktreeRegistryIdentityError(WorktreeRegistryError):
    """Process-birth identity mismatch, dead, reused, or unknown."""


class WorktreeRegistryObservationError(WorktreeRegistryError):
    """Git observation missing, stale, or mismatched with registry state."""


class WorktreeRegistryAuthorityError(WorktreeRegistryError):
    """Attempt to grant authority from a non-registry source (local JSON)."""


class WorktreeRegistryBoundsError(WorktreeRegistryError, ValueError):
    """Payload or path bound exceeded."""


class WorktreeRegistryNotOpenError(WorktreeRegistryError):
    """Operation requires an open registry."""


class WorktreeRegistryContainmentError(WorktreeRegistryError):
    """Path escapes canonical root or violates containment policy."""


class DuckDBUnavailableError(WorktreeRegistryError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RepositoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    TOMBSTONED = "tombstoned"


class WorktreeStatus(str, Enum):
    PREPARING = "preparing"
    ACTIVE = "active"
    SETTLING = "settling"
    TERMINAL = "terminal"
    RECLAIMED = "reclaimed"
    SUPERSEDED = "superseded"


class WorktreeLifecycleState(str, Enum):
    PREPARING = "preparing"
    ACTIVE = "active"
    SETTLING = "settling"
    TERMINAL = "terminal"

    @property
    def is_terminal(self) -> bool:
        return self is WorktreeLifecycleState.TERMINAL

    @property
    def is_nonterminal(self) -> bool:
        return not self.is_terminal


class PathKind(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    GITLINK = "gitlink"
    MISSING = "missing"


class PathPolicyDisposition(str, Enum):
    TRACKED = "tracked"
    UNTRACKED = "untracked"
    RENAMED = "renamed"
    DELETED = "deleted"
    IGNORED = "ignored"
    PRIVATE = "private"
    ESCAPED = "escaped"


class OverlayEntryKind(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    UNTRACKED = "untracked"
    TYPECHANGE = "typechange"


class LeaseEventKind(str, Enum):
    ACQUIRE = "acquire"
    RENEW = "renew"
    RELEASE = "release"
    RECLAIM = "reclaim"
    DENY = "deny"


class ReuseDisposition(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    RECLAIM_THEN_ALLOW = "reclaim_then_allow"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LivenessProbe = Callable[[ProcessBirthIdentity], OwnerLiveness]
BirthReader = Callable[[int], ProcessBirthIdentity | None]
ClockMs = Callable[[], int]
GitObserver = Callable[[str], "GitObservation | None"]


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso_from_ms(epoch_ms: int) -> str:
    return (
        datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _default_clock_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise WorktreeRegistryError(f"{name} contains NUL")
    if required and not text:
        raise WorktreeRegistryError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorktreeRegistryBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorktreeRegistryBoundsError(f"{name} must be a positive integer")
    return value


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    name: str,
    max_bytes: int,
) -> dict[str, Any]:
    raw = dict(body or {})
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > max_bytes:
        raise WorktreeRegistryBoundsError(
            f"{name} exceeds the {max_bytes}-byte bound"
        )
    return raw


def _row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    # DuckDBRow and sqlite3.Row support key iteration.
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        keys = []
    if keys:
        return {str(key): row[key] for key in keys}
    try:
        return {str(index): row[index] for index in range(len(row))}  # type: ignore[arg-type]
    except Exception:
        return {}


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value or "").strip()
    if not text or text == "None":
        return {}
    try:
        loaded = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    text = str(value or "").strip()
    if not text or text == "None":
        return []
    try:
        loaded = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return list(loaded) if isinstance(loaded, list) else []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"", "0", "false", "f", "no", "n", "none", "null"}:
        return False
    return bool(value)


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement or statement.startswith("--"):
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def process_birth_id(birth: ProcessBirthIdentity) -> str:
    """Return a stable content id for multi-factor process birth.

    PID alone is never sufficient; start-time ticks, boot id, and parent pid
    participate so PID reuse yields a different identity.
    """

    if not isinstance(birth, ProcessBirthIdentity):
        raise TypeError("birth must be ProcessBirthIdentity")
    material = (
        f"{int(birth.pid)}:{int(birth.start_time_ticks)}:"
        f"{str(birth.boot_id or '')}:{int(birth.parent_pid or 0)}"
    )
    return f"birth:{_sha256_hex(material.encode('utf-8'))[7:39]}"


def process_births_match(
    expected: ProcessBirthIdentity,
    observed: ProcessBirthIdentity | None,
) -> bool:
    """Return whether observed birth is an exact multi-factor match."""

    if observed is None:
        return False
    if int(expected.pid) != int(observed.pid):
        return False
    if int(expected.start_time_ticks) and int(observed.start_time_ticks):
        if int(expected.start_time_ticks) != int(observed.start_time_ticks):
            return False
    elif int(expected.start_time_ticks) or int(observed.start_time_ticks):
        return False
    if expected.boot_id and observed.boot_id and expected.boot_id != observed.boot_id:
        return False
    return True


def _parse_enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    text = _text(value, name)
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise WorktreeRegistryError(f"unknown {name}: {text}") from exc


def normalize_workspace_path(path: str | Path) -> str:
    """Return a normalized absolute-style workspace path string."""

    text = _text(path, "workspace_path")
    # Preserve absolute form; collapse redundant separators and trailing slash.
    normalized = str(Path(text))
    if text.startswith("/") and not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized.rstrip("/") or normalized


def normalize_relative_path(path: str) -> str:
    """Normalize a repository-relative path; reject escapes."""

    text = str(path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = PurePosixPath(text).as_posix() if text else ""
    if not text or text == ".":
        raise WorktreeRegistryContainmentError("relative_path is required")
    if text.startswith("/") or text.startswith("../") or "/../" in f"/{text}/":
        raise WorktreeRegistryContainmentError(
            f"path escapes canonical containment: {path}"
        )
    parts = PurePosixPath(text).parts
    if ".." in parts or parts[0] == "/":
        raise WorktreeRegistryContainmentError(
            f"path escapes canonical containment: {path}"
        )
    if len(text.encode("utf-8")) > MAX_PATH_BYTES:
        raise WorktreeRegistryBoundsError("relative_path exceeds bound")
    return text


def path_contained_in_root(relative_path: str, *, canonical_root: str = "") -> str:
    """Validate relative path containment; return normalized path."""

    normalized = normalize_relative_path(relative_path)
    # Symlink targets are recorded separately; the relative path itself must
    # stay inside the worktree root regardless of target.
    if canonical_root:
        # Host root is recorded for identity; relative paths never leave it.
        _ = _text(canonical_root, "canonical_root", required=False)
    return normalized


def digest_overlay_entries(entries: Sequence[Mapping[str, Any]]) -> str:
    """Compute a stable digest over dirty-overlay entries (Git byte binding)."""

    material = [
        {
            "kind": str(item.get("kind") or ""),
            "path": str(item.get("path") or ""),
            "from_path": str(item.get("from_path") or ""),
            "blob_id": str(item.get("blob_id") or ""),
            "mode": str(item.get("mode") or ""),
        }
        for item in entries
    ]
    material.sort(key=lambda row: (row["path"], row["kind"], row["from_path"]))
    return _sha256_hex(_canonical_json(material).encode("utf-8"))


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitObservation:
    """Current Git byte observations for one worktree path.

    Git is byte authority: these fields are supplied by a Git probe (or a
    hermetic test double), never invented by the registry for ownership.
    """

    SCHEMA: ClassVar[str] = GIT_OBSERVATION_SCHEMA

    workspace_path: str
    head_commit: str = ""
    head_tree: str = ""
    index_digest: str = ""
    dirty_overlay_digest: str = ""
    branch_name: str = ""
    is_detached: bool = False
    git_common_dir: str = ""
    path_exists: bool = True
    is_symlink_root: bool = False
    observed_at_ms: int = 0
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_path",
            normalize_workspace_path(self.workspace_path),
        )
        object.__setattr__(
            self, "head_commit", _text(self.head_commit, "head_commit", required=False)
        )
        object.__setattr__(
            self, "head_tree", _text(self.head_tree, "head_tree", required=False)
        )
        object.__setattr__(
            self,
            "index_digest",
            _text(self.index_digest, "index_digest", required=False),
        )
        object.__setattr__(
            self,
            "dirty_overlay_digest",
            _text(self.dirty_overlay_digest, "dirty_overlay_digest", required=False),
        )
        object.__setattr__(
            self,
            "branch_name",
            _text(self.branch_name, "branch_name", required=False),
        )
        object.__setattr__(
            self,
            "git_common_dir",
            _text(self.git_common_dir, "git_common_dir", required=False),
        )
        object.__setattr__(
            self,
            "observed_at_ms",
            _nonneg_int(int(self.observed_at_ms), "observed_at_ms"),
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "workspace_path": self.workspace_path,
            "head_commit": self.head_commit,
            "head_tree": self.head_tree,
            "index_digest": self.index_digest,
            "dirty_overlay_digest": self.dirty_overlay_digest,
            "branch_name": self.branch_name,
            "is_detached": bool(self.is_detached),
            "git_common_dir": self.git_common_dir,
            "path_exists": bool(self.path_exists),
            "is_symlink_root": bool(self.is_symlink_root),
            "observed_at_ms": int(self.observed_at_ms),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class RepositoryRecord:
    """Registered Git repository (common-dir identity)."""

    INTERFACE: ClassVar[str] = REPOSITORY_FOREST_INTERFACE
    SCHEMA: ClassVar[str] = REPOSITORY_RECORD_SCHEMA

    repository_id: str
    git_common_dir: str
    canonical_root: str
    remote_url: str = ""
    head_commit: str = ""
    head_tree: str = ""
    server_generation: int = 1
    registered_at_ms: int = 0
    revision: int = 1
    status: RepositoryStatus = RepositoryStatus.ACTIVE
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self,
            "git_common_dir",
            normalize_workspace_path(self.git_common_dir),
        )
        object.__setattr__(
            self,
            "canonical_root",
            normalize_workspace_path(self.canonical_root),
        )
        object.__setattr__(
            self, "remote_url", _text(self.remote_url, "remote_url", required=False)
        )
        object.__setattr__(
            self, "head_commit", _text(self.head_commit, "head_commit", required=False)
        )
        object.__setattr__(
            self, "head_tree", _text(self.head_tree, "head_tree", required=False)
        )
        object.__setattr__(
            self,
            "server_generation",
            _positive_int(int(self.server_generation), "server_generation"),
        )
        object.__setattr__(
            self,
            "registered_at_ms",
            _nonneg_int(int(self.registered_at_ms), "registered_at_ms"),
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        status = self.status
        if not isinstance(status, RepositoryStatus):
            status = _parse_enum(status, RepositoryStatus, "status")
            object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "repository_id": self.repository_id,
            "git_common_dir": self.git_common_dir,
            "canonical_root": self.canonical_root,
            "remote_url": self.remote_url,
            "head_commit": self.head_commit,
            "head_tree": self.head_tree,
            "server_generation": int(self.server_generation),
            "registered_at_ms": int(self.registered_at_ms),
            "registered_at": (
                _utc_iso_from_ms(self.registered_at_ms) if self.registered_at_ms else ""
            ),
            "revision": int(self.revision),
            "status": self.status.value,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class BranchRecord:
    """Named branch tip bound to one repository."""

    SCHEMA: ClassVar[str] = BRANCH_RECORD_SCHEMA

    branch_id: str
    repository_id: str
    branch_name: str
    tip_commit: str = ""
    is_detached: bool = False
    upstream: str = ""
    registered_at_ms: int = 0
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", _text(self.branch_id, "branch_id"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self, "branch_name", _text(self.branch_name, "branch_name")
        )
        object.__setattr__(
            self, "tip_commit", _text(self.tip_commit, "tip_commit", required=False)
        )
        object.__setattr__(
            self, "upstream", _text(self.upstream, "upstream", required=False)
        )
        object.__setattr__(
            self,
            "registered_at_ms",
            _nonneg_int(int(self.registered_at_ms), "registered_at_ms"),
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "branch_id": self.branch_id,
            "repository_id": self.repository_id,
            "branch_name": self.branch_name,
            "tip_commit": self.tip_commit,
            "is_detached": bool(self.is_detached),
            "upstream": self.upstream,
            "registered_at_ms": int(self.registered_at_ms),
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class GitRefRecord:
    """Git ref (branch/tag/other) bound to an object id (byte authority)."""

    SCHEMA: ClassVar[str] = GIT_REF_SCHEMA

    ref_id: str
    repository_id: str
    ref_name: str
    object_id: str
    ref_kind: str = "branch"
    registered_at_ms: int = 0
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref_id", _text(self.ref_id, "ref_id"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "ref_name", _text(self.ref_name, "ref_name"))
        object.__setattr__(self, "object_id", _text(self.object_id, "object_id"))
        object.__setattr__(
            self, "ref_kind", _text(self.ref_kind, "ref_kind", required=False) or "branch"
        )
        object.__setattr__(
            self,
            "registered_at_ms",
            _nonneg_int(int(self.registered_at_ms), "registered_at_ms"),
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "ref_id": self.ref_id,
            "repository_id": self.repository_id,
            "ref_name": self.ref_name,
            "object_id": self.object_id,
            "ref_kind": self.ref_kind,
            "registered_at_ms": int(self.registered_at_ms),
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class SubmoduleEdge:
    """Nested gitlink edge between parent and child repositories."""

    SCHEMA: ClassVar[str] = SUBMODULE_EDGE_SCHEMA

    edge_id: str
    parent_repository_id: str
    child_repository_id: str
    gitlink_path: str
    gitlink_commit: str = ""
    registered_at_ms: int = 0
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_id", _text(self.edge_id, "edge_id"))
        object.__setattr__(
            self,
            "parent_repository_id",
            _text(self.parent_repository_id, "parent_repository_id"),
        )
        object.__setattr__(
            self,
            "child_repository_id",
            _text(self.child_repository_id, "child_repository_id"),
        )
        object.__setattr__(
            self,
            "gitlink_path",
            path_contained_in_root(self.gitlink_path),
        )
        object.__setattr__(
            self,
            "gitlink_commit",
            _text(self.gitlink_commit, "gitlink_commit", required=False),
        )
        object.__setattr__(
            self,
            "registered_at_ms",
            _nonneg_int(int(self.registered_at_ms), "registered_at_ms"),
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "edge_id": self.edge_id,
            "parent_repository_id": self.parent_repository_id,
            "child_repository_id": self.child_repository_id,
            "gitlink_path": self.gitlink_path,
            "gitlink_commit": self.gitlink_commit,
            "registered_at_ms": int(self.registered_at_ms),
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class WorktreeIdentity:
    """Fenced worktree identity shared across lanes."""

    INTERFACE: ClassVar[str] = WORKTREE_IDENTITY_INTERFACE
    SCHEMA: ClassVar[str] = WORKTREE_IDENTITY_SCHEMA

    worktree_id: str
    repository_id: str
    workspace_path: str
    branch_name: str = ""
    lane_id: str = ""
    task_id: str = ""
    attempt: int = 0
    session_id: str = ""
    lease_id: str = ""
    fencing_token: int = 0
    fence_epoch: int = 0
    owner_process_birth: ProcessBirthIdentity | None = None
    owner_process_birth_id: str = ""
    lifecycle_state: WorktreeLifecycleState = WorktreeLifecycleState.PREPARING
    lease_expires_at_ms: int = 0
    head_commit: str = ""
    head_tree: str = ""
    index_digest: str = ""
    dirty_overlay_digest: str = ""
    current_snapshot_id: str = ""
    is_detached: bool = False
    registered_at_ms: int = 0
    updated_at_ms: int = 0
    revision: int = 1
    status: WorktreeStatus = WorktreeStatus.PREPARING
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self,
            "workspace_path",
            normalize_workspace_path(self.workspace_path),
        )
        object.__setattr__(
            self, "branch_name", _text(self.branch_name, "branch_name", required=False)
        )
        object.__setattr__(
            self, "lane_id", _text(self.lane_id, "lane_id", required=False)
        )
        object.__setattr__(
            self, "task_id", _text(self.task_id, "task_id", required=False)
        )
        object.__setattr__(
            self, "attempt", _nonneg_int(int(self.attempt), "attempt")
        )
        object.__setattr__(
            self, "session_id", _text(self.session_id, "session_id", required=False)
        )
        object.__setattr__(
            self, "lease_id", _text(self.lease_id, "lease_id", required=False)
        )
        object.__setattr__(
            self,
            "fencing_token",
            _nonneg_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _nonneg_int(int(self.fence_epoch), "fence_epoch")
        )
        if self.owner_process_birth is not None:
            if not isinstance(self.owner_process_birth, ProcessBirthIdentity):
                raise TypeError("owner_process_birth must be ProcessBirthIdentity")
            birth_id = self.owner_process_birth_id or process_birth_id(
                self.owner_process_birth
            )
            object.__setattr__(self, "owner_process_birth_id", birth_id)
        else:
            object.__setattr__(
                self,
                "owner_process_birth_id",
                _text(
                    self.owner_process_birth_id,
                    "owner_process_birth_id",
                    required=False,
                ),
            )
        lifecycle = self.lifecycle_state
        if not isinstance(lifecycle, WorktreeLifecycleState):
            lifecycle = _parse_enum(lifecycle, WorktreeLifecycleState, "lifecycle_state")
            object.__setattr__(self, "lifecycle_state", lifecycle)
        status = self.status
        if not isinstance(status, WorktreeStatus):
            status = _parse_enum(status, WorktreeStatus, "status")
            object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "lease_expires_at_ms",
            _nonneg_int(int(self.lease_expires_at_ms), "lease_expires_at_ms"),
        )
        for field_name in (
            "head_commit",
            "head_tree",
            "index_digest",
            "dirty_overlay_digest",
            "current_snapshot_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name, required=False),
            )
        object.__setattr__(
            self,
            "registered_at_ms",
            _nonneg_int(int(self.registered_at_ms), "registered_at_ms"),
        )
        object.__setattr__(
            self, "updated_at_ms", _nonneg_int(int(self.updated_at_ms), "updated_at_ms")
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    @property
    def is_leased(self) -> bool:
        return bool(self.lease_id) and int(self.fencing_token) >= 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "worktree_id": self.worktree_id,
            "repository_id": self.repository_id,
            "workspace_path": self.workspace_path,
            "branch_name": self.branch_name,
            "lane_id": self.lane_id,
            "task_id": self.task_id,
            "attempt": int(self.attempt),
            "session_id": self.session_id,
            "lease_id": self.lease_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "owner_process_birth": (
                None
                if self.owner_process_birth is None
                else self.owner_process_birth.to_dict()
            ),
            "owner_process_birth_id": self.owner_process_birth_id,
            "lifecycle_state": self.lifecycle_state.value,
            "lease_expires_at_ms": int(self.lease_expires_at_ms),
            "head_commit": self.head_commit,
            "head_tree": self.head_tree,
            "index_digest": self.index_digest,
            "dirty_overlay_digest": self.dirty_overlay_digest,
            "current_snapshot_id": self.current_snapshot_id,
            "is_detached": bool(self.is_detached),
            "registered_at_ms": int(self.registered_at_ms),
            "updated_at_ms": int(self.updated_at_ms),
            "revision": int(self.revision),
            "status": self.status.value,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class WorktreeSnapshot:
    """Base/head/tree/index snapshot bound to Git object IDs."""

    INTERFACE: ClassVar[str] = WORKTREE_SNAPSHOT_INTERFACE
    SCHEMA: ClassVar[str] = WORKTREE_SNAPSHOT_SCHEMA

    snapshot_id: str
    worktree_id: str
    repository_id: str
    base_commit: str = ""
    head_commit: str = ""
    head_tree: str = ""
    index_digest: str = ""
    dirty_overlay_digest: str = ""
    branch_name: str = ""
    is_detached: bool = False
    scanner_version: str = ""
    observed_at_ms: int = 0
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        for field_name in (
            "base_commit",
            "head_commit",
            "head_tree",
            "index_digest",
            "dirty_overlay_digest",
            "branch_name",
            "scanner_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name, required=False),
            )
        object.__setattr__(
            self,
            "observed_at_ms",
            _nonneg_int(int(self.observed_at_ms), "observed_at_ms"),
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "snapshot_id": self.snapshot_id,
            "worktree_id": self.worktree_id,
            "repository_id": self.repository_id,
            "base_commit": self.base_commit,
            "head_commit": self.head_commit,
            "head_tree": self.head_tree,
            "index_digest": self.index_digest,
            "dirty_overlay_digest": self.dirty_overlay_digest,
            "branch_name": self.branch_name,
            "is_detached": bool(self.is_detached),
            "scanner_version": self.scanner_version,
            "observed_at_ms": int(self.observed_at_ms),
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class WorktreePathRecord:
    """One path row under a worktree snapshot."""

    SCHEMA: ClassVar[str] = WORKTREE_PATH_SCHEMA

    path_id: str
    snapshot_id: str
    worktree_id: str
    relative_path: str
    path_kind: PathKind = PathKind.FILE
    blob_id: str = ""
    symlink_target: str = ""
    is_symlink: bool = False
    is_gitlink: bool = False
    policy_disposition: PathPolicyDisposition = PathPolicyDisposition.TRACKED
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", _text(self.path_id, "path_id"))
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "relative_path", path_contained_in_root(self.relative_path)
        )
        kind = self.path_kind
        if not isinstance(kind, PathKind):
            kind = _parse_enum(kind, PathKind, "path_kind")
            object.__setattr__(self, "path_kind", kind)
        disposition = self.policy_disposition
        if not isinstance(disposition, PathPolicyDisposition):
            disposition = _parse_enum(
                disposition, PathPolicyDisposition, "policy_disposition"
            )
            object.__setattr__(self, "policy_disposition", disposition)
        object.__setattr__(
            self, "blob_id", _text(self.blob_id, "blob_id", required=False)
        )
        object.__setattr__(
            self,
            "symlink_target",
            _text(self.symlink_target, "symlink_target", required=False),
        )
        if self.is_symlink or kind is PathKind.SYMLINK:
            object.__setattr__(self, "is_symlink", True)
        if self.is_gitlink or kind is PathKind.GITLINK:
            object.__setattr__(self, "is_gitlink", True)
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "path_id": self.path_id,
            "snapshot_id": self.snapshot_id,
            "worktree_id": self.worktree_id,
            "relative_path": self.relative_path,
            "path_kind": self.path_kind.value,
            "blob_id": self.blob_id,
            "symlink_target": self.symlink_target,
            "is_symlink": bool(self.is_symlink),
            "is_gitlink": bool(self.is_gitlink),
            "policy_disposition": self.policy_disposition.value,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class DirtyOverlay:
    """Dirty worktree overlay with rename/delete/untracked policy."""

    INTERFACE: ClassVar[str] = DIRTY_OVERLAY_INTERFACE
    SCHEMA: ClassVar[str] = DIRTY_OVERLAY_SCHEMA

    overlay_id: str
    snapshot_id: str
    worktree_id: str
    overlay_digest: str
    entry_count: int = 0
    rename_policy: str = "track"
    delete_policy: str = "track"
    untracked_policy: str = "include"
    entries: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    recorded_at_ms: int = 0
    revision: int = 1
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overlay_id", _text(self.overlay_id, "overlay_id"))
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "overlay_digest", _text(self.overlay_digest, "overlay_digest")
        )
        entries = tuple(dict(item) for item in (self.entries or ()))
        if len(entries) > MAX_OVERLAY_ENTRIES:
            raise WorktreeRegistryBoundsError(
                f"dirty overlay exceeds {MAX_OVERLAY_ENTRIES} entries"
            )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "entry_count",
            _nonneg_int(
                int(self.entry_count if self.entry_count else len(entries)),
                "entry_count",
            ),
        )
        object.__setattr__(
            self,
            "rename_policy",
            _text(self.rename_policy, "rename_policy", required=False) or "track",
        )
        object.__setattr__(
            self,
            "delete_policy",
            _text(self.delete_policy, "delete_policy", required=False) or "track",
        )
        object.__setattr__(
            self,
            "untracked_policy",
            _text(self.untracked_policy, "untracked_policy", required=False)
            or "include",
        )
        object.__setattr__(
            self,
            "recorded_at_ms",
            _nonneg_int(int(self.recorded_at_ms), "recorded_at_ms"),
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.body or {}), name="body", max_bytes=MAX_PAYLOAD_BYTES
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "overlay_id": self.overlay_id,
            "snapshot_id": self.snapshot_id,
            "worktree_id": self.worktree_id,
            "overlay_digest": self.overlay_digest,
            "entry_count": int(self.entry_count),
            "rename_policy": self.rename_policy,
            "delete_policy": self.delete_policy,
            "untracked_policy": self.untracked_policy,
            "entries": [dict(item) for item in self.entries],
            "recorded_at_ms": int(self.recorded_at_ms),
            "revision": int(self.revision),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class SetupCacheEntry:
    """Shared setup cache entry keyed by repository/worktree digest."""

    SCHEMA: ClassVar[str] = SETUP_CACHE_SCHEMA

    cache_key: str
    repository_id: str
    worktree_id: str = ""
    cache_digest: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    recorded_at_ms: int = 0
    expires_at_ms: int = 0
    revision: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_key", _text(self.cache_key, "cache_key"))
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self,
            "cache_digest",
            _text(self.cache_digest, "cache_digest", required=False),
        )
        object.__setattr__(
            self,
            "payload",
            MappingProxyType(
                _bounded_mapping(
                    dict(self.payload or {}),
                    name="payload",
                    max_bytes=MAX_PAYLOAD_BYTES,
                )
            ),
        )
        object.__setattr__(
            self,
            "recorded_at_ms",
            _nonneg_int(int(self.recorded_at_ms), "recorded_at_ms"),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _nonneg_int(int(self.expires_at_ms), "expires_at_ms"),
        )
        object.__setattr__(
            self, "revision", _positive_int(int(self.revision), "revision")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "cache_key": self.cache_key,
            "repository_id": self.repository_id,
            "worktree_id": self.worktree_id,
            "cache_digest": self.cache_digest,
            "payload": dict(self.payload),
            "recorded_at_ms": int(self.recorded_at_ms),
            "expires_at_ms": int(self.expires_at_ms),
            "revision": int(self.revision),
        }


@dataclass(frozen=True)
class ReuseDecision:
    """Whether a worktree may be reused or cleaned up."""

    disposition: ReuseDisposition
    reason: str
    worktree: WorktreeIdentity | None = None
    requires_git_observation: bool = True
    requires_matching_lease: bool = True
    observation_matched: bool = False
    lease_matched: bool = False

    @property
    def allowed(self) -> bool:
        return self.disposition in {
            ReuseDisposition.ALLOW,
            ReuseDisposition.RECLAIM_THEN_ALLOW,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "requires_git_observation": bool(self.requires_git_observation),
            "requires_matching_lease": bool(self.requires_matching_lease),
            "observation_matched": bool(self.observation_matched),
            "lease_matched": bool(self.lease_matched),
            "worktree": None if self.worktree is None else self.worktree.to_dict(),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DatabaseWorktreeRegistry:
    """DuckDB-backed worktree/repository forest registry (semantic authority)."""

    INTERFACE: ClassVar[str] = DATABASE_WORKTREE_REGISTRY_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_WORKTREE_REGISTRY_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        liveness: LivenessProbe | None = None,
        birth_reader: BirthReader | None = None,
        git_observer: GitObserver | None = None,
        default_lease_ttl_ms: int = DEFAULT_LEASE_TTL_MS,
        server_generation: int | None = None,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseWorktreeRegistry; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._liveness = liveness or (lambda birth: owner_liveness(birth))
        if birth_reader is not None:
            self._birth_reader = birth_reader
        elif liveness is not None:
            self._birth_reader = lambda _pid: None
        else:
            self._birth_reader = lambda pid: read_process_birth(int(pid))
        self._git_observer = git_observer
        self._default_lease_ttl_ms = _positive_int(
            int(default_lease_ttl_ms), "default_lease_ttl_ms"
        )
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True
        self._server_generation = (
            _positive_int(int(server_generation), "server_generation")
            if server_generation is not None
            else 1
        )

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    @property
    def server_generation(self) -> int:
        return int(self._server_generation)

    def open(self) -> "DatabaseWorktreeRegistry":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            try:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                for key, value in (
                    ("interface", DATABASE_WORKTREE_REGISTRY_INTERFACE),
                    ("schema", DATABASE_WORKTREE_REGISTRY_SCHEMA),
                    ("byte_authority", "git"),
                    ("semantic_authority", "database"),
                ):
                    connection.execute(
                        """
                        INSERT INTO registry_metadata(key, value)
                        VALUES (?, ?)
                        ON CONFLICT (key) DO UPDATE SET
                            value = excluded.value
                        """,
                        [key, value],
                    )
                row = connection.execute(
                    "SELECT value FROM registry_metadata WHERE key = ?",
                    ["server_generation"],
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO registry_metadata(key, value) VALUES (?, ?)
                        """,
                        ["server_generation", str(self._server_generation)],
                    )
                else:
                    mapping = _row_mapping(row)
                    raw = (
                        mapping.get("value")
                        or mapping.get("VALUE")
                        or mapping.get("0")
                    )
                    self._server_generation = max(1, int(raw or 1))
                self._connection = connection
                self._closed = False
                self._commit_if_idle(connection)
                return self
            except Exception:
                try:
                    connection.close()
                except Exception:
                    pass
                raise

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseWorktreeRegistry":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise WorktreeRegistryNotOpenError("DatabaseWorktreeRegistry is not open")
        return self._connection

    def _begin(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        # Prefer SQL BEGIN so DuckDBConnection tracks in_transaction the same
        # way as DatabaseEventLog and other durable supervisor stores.
        connection.execute("BEGIN TRANSACTION")

    def _rollback_if_open(self, connection: Any) -> None:
        try:
            if getattr(connection, "in_transaction", False):
                connection.execute("ROLLBACK")
                return
            rollback = getattr(connection, "rollback", None)
            if callable(rollback):
                rollback()
        except Exception:
            pass

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            connection.execute("COMMIT")
            return
        # Autocommit path: best-effort flush for adapters that buffer writes.
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    def _now_ms(self) -> int:
        return int(self._clock_ms())

    def authority_policy(self) -> dict[str, str]:
        """Return the fixed authority split for this registry."""

        return {
            "semantic_authority": "database",
            "byte_authority": "git",
            "local_json_authority": "none",
            "interface": DATABASE_WORKTREE_REGISTRY_INTERFACE,
        }

    # -- repositories --------------------------------------------------------

    def register_repository(
        self,
        *,
        git_common_dir: str | Path,
        canonical_root: str | Path,
        repository_id: str | None = None,
        remote_url: str = "",
        head_commit: str = "",
        head_tree: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> RepositoryRecord:
        """Register or upsert one Git repository by common-dir identity."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        common = normalize_workspace_path(git_common_dir)
        root = normalize_workspace_path(canonical_root)
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    """
                    SELECT repository_id, revision, status FROM repositories
                    WHERE git_common_dir = ?
                    """,
                    [common],
                ).fetchone()
                if existing is not None:
                    mapping = _row_mapping(existing)
                    repo_id = str(mapping.get("repository_id") or mapping.get("0"))
                    revision = int(mapping.get("revision") or mapping.get("1") or 1) + 1
                else:
                    repo_id = _text(
                        repository_id or _new_id("repository"), "repository_id"
                    )
                    prior = connection.execute(
                        "SELECT repository_id FROM repositories WHERE repository_id = ?",
                        [repo_id],
                    ).fetchone()
                    if prior is not None:
                        raise WorktreeRegistryConflictError(
                            f"repository_id already registered: {repo_id}"
                        )
                    revision = 1
                record = RepositoryRecord(
                    repository_id=repo_id,
                    git_common_dir=common,
                    canonical_root=root,
                    remote_url=remote_url,
                    head_commit=head_commit,
                    head_tree=head_tree,
                    server_generation=self._server_generation,
                    registered_at_ms=now,
                    revision=revision,
                    status=RepositoryStatus.ACTIVE,
                    body=dict(body or {}),
                )
                connection.execute(
                    """
                    INSERT INTO repositories(
                        repository_id, git_common_dir, canonical_root, remote_url,
                        head_commit, head_tree, server_generation, registered_at_ms,
                        revision, status, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (repository_id) DO UPDATE SET
                        git_common_dir = excluded.git_common_dir,
                        canonical_root = excluded.canonical_root,
                        remote_url = excluded.remote_url,
                        head_commit = excluded.head_commit,
                        head_tree = excluded.head_tree,
                        server_generation = excluded.server_generation,
                        registered_at_ms = excluded.registered_at_ms,
                        revision = excluded.revision,
                        status = excluded.status,
                        body_json = excluded.body_json
                    """,
                    [
                        record.repository_id,
                        record.git_common_dir,
                        record.canonical_root,
                        record.remote_url,
                        record.head_commit,
                        record.head_tree,
                        int(record.server_generation),
                        int(record.registered_at_ms),
                        int(record.revision),
                        record.status.value,
                        _canonical_json(dict(record.body)),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO repository_revisions(
                        revision_id, repository_id, revision, head_commit,
                        head_tree, recorded_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        _new_id("repo-rev"),
                        record.repository_id,
                        int(record.revision),
                        record.head_commit,
                        record.head_tree,
                        now,
                        _canonical_json(dict(record.body)),
                    ],
                )
                self._commit_if_idle(connection)
                return record
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_repository(self, repository_id: str) -> RepositoryRecord | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT repository_id, git_common_dir, canonical_root, remote_url,
                       head_commit, head_tree, server_generation, registered_at_ms,
                       revision, status, body_json
                FROM repositories WHERE repository_id = ?
                """,
                [_text(repository_id, "repository_id")],
            ).fetchone()
            if row is None:
                return None
            return self._row_to_repository(_row_mapping(row))

    def list_repositories(self) -> list[RepositoryRecord]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT repository_id, git_common_dir, canonical_root, remote_url,
                       head_commit, head_tree, server_generation, registered_at_ms,
                       revision, status, body_json
                FROM repositories ORDER BY repository_id
                """
            ).fetchall()
            return [self._row_to_repository(_row_mapping(row)) for row in rows]

    # -- branches / refs / submodules ----------------------------------------

    def register_branch(
        self,
        *,
        repository_id: str,
        branch_name: str,
        tip_commit: str = "",
        is_detached: bool = False,
        upstream: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> BranchRecord:
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        repo_id = _text(repository_id, "repository_id")
        name = _text(branch_name, "branch_name")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._require_repository_locked(connection, repo_id)
                existing = connection.execute(
                    """
                    SELECT branch_id, revision FROM branches
                    WHERE repository_id = ? AND branch_name = ?
                    """,
                    [repo_id, name],
                ).fetchone()
                if existing is not None:
                    mapping = _row_mapping(existing)
                    branch_id = str(mapping.get("branch_id") or mapping.get("0"))
                    revision = int(mapping.get("revision") or mapping.get("1") or 1) + 1
                else:
                    branch_id = _new_id("branch")
                    revision = 1
                record = BranchRecord(
                    branch_id=branch_id,
                    repository_id=repo_id,
                    branch_name=name,
                    tip_commit=tip_commit,
                    is_detached=bool(is_detached),
                    upstream=upstream,
                    registered_at_ms=now,
                    revision=revision,
                    body=dict(body or {}),
                )
                connection.execute(
                    """
                    INSERT INTO branches(
                        branch_id, repository_id, branch_name, tip_commit,
                        is_detached, upstream, registered_at_ms, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (branch_id) DO UPDATE SET
                        repository_id = excluded.repository_id,
                        branch_name = excluded.branch_name,
                        tip_commit = excluded.tip_commit,
                        is_detached = excluded.is_detached,
                        upstream = excluded.upstream,
                        registered_at_ms = excluded.registered_at_ms,
                        revision = excluded.revision,
                        body_json = excluded.body_json
                    """,
                    [
                        record.branch_id,
                        record.repository_id,
                        record.branch_name,
                        record.tip_commit,
                        bool(record.is_detached),
                        record.upstream,
                        int(record.registered_at_ms),
                        int(record.revision),
                        _canonical_json(dict(record.body)),
                    ],
                )
                # Also publish a git_ref row so branch tips stay byte-bound.
                self._upsert_ref_locked(
                    connection,
                    repository_id=repo_id,
                    ref_name=f"refs/heads/{name}",
                    object_id=tip_commit or "unknown",
                    ref_kind="detached" if is_detached else "branch",
                    now_ms=now,
                )
                self._commit_if_idle(connection)
                return record
            except Exception:
                self._rollback_if_open(connection)
                raise

    def register_ref(
        self,
        *,
        repository_id: str,
        ref_name: str,
        object_id: str,
        ref_kind: str = "branch",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> GitRefRecord:
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._require_repository_locked(connection, repository_id)
                record = self._upsert_ref_locked(
                    connection,
                    repository_id=repository_id,
                    ref_name=ref_name,
                    object_id=object_id,
                    ref_kind=ref_kind,
                    body=body,
                    now_ms=now,
                )
                self._commit_if_idle(connection)
                return record
            except Exception:
                self._rollback_if_open(connection)
                raise

    def register_submodule_edge(
        self,
        *,
        parent_repository_id: str,
        child_repository_id: str,
        gitlink_path: str,
        gitlink_commit: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> SubmoduleEdge:
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        parent = _text(parent_repository_id, "parent_repository_id")
        child = _text(child_repository_id, "child_repository_id")
        path = path_contained_in_root(gitlink_path)
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._require_repository_locked(connection, parent)
                self._require_repository_locked(connection, child)
                existing = connection.execute(
                    """
                    SELECT edge_id, revision FROM submodule_edges
                    WHERE parent_repository_id = ? AND gitlink_path = ?
                    """,
                    [parent, path],
                ).fetchone()
                if existing is not None:
                    mapping = _row_mapping(existing)
                    edge_id = str(mapping.get("edge_id") or mapping.get("0"))
                    revision = int(mapping.get("revision") or mapping.get("1") or 1) + 1
                else:
                    edge_id = _new_id("gitlink")
                    revision = 1
                edge = SubmoduleEdge(
                    edge_id=edge_id,
                    parent_repository_id=parent,
                    child_repository_id=child,
                    gitlink_path=path,
                    gitlink_commit=gitlink_commit,
                    registered_at_ms=now,
                    revision=revision,
                    body=dict(body or {}),
                )
                connection.execute(
                    """
                    INSERT INTO submodule_edges(
                        edge_id, parent_repository_id, child_repository_id,
                        gitlink_path, gitlink_commit, registered_at_ms, revision,
                        body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (edge_id) DO UPDATE SET
                        parent_repository_id = excluded.parent_repository_id,
                        child_repository_id = excluded.child_repository_id,
                        gitlink_path = excluded.gitlink_path,
                        gitlink_commit = excluded.gitlink_commit,
                        registered_at_ms = excluded.registered_at_ms,
                        revision = excluded.revision,
                        body_json = excluded.body_json
                    """,
                    [
                        edge.edge_id,
                        edge.parent_repository_id,
                        edge.child_repository_id,
                        edge.gitlink_path,
                        edge.gitlink_commit,
                        int(edge.registered_at_ms),
                        int(edge.revision),
                        _canonical_json(dict(edge.body)),
                    ],
                )
                self._commit_if_idle(connection)
                return edge
            except Exception:
                self._rollback_if_open(connection)
                raise

    def list_branches(self, repository_id: str) -> list[BranchRecord]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT branch_id, repository_id, branch_name, tip_commit,
                       is_detached, upstream, registered_at_ms, revision, body_json
                FROM branches WHERE repository_id = ? ORDER BY branch_name
                """,
                [_text(repository_id, "repository_id")],
            ).fetchall()
            return [self._row_to_branch(_row_mapping(row)) for row in rows]

    def list_submodule_edges(self, parent_repository_id: str) -> list[SubmoduleEdge]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT edge_id, parent_repository_id, child_repository_id,
                       gitlink_path, gitlink_commit, registered_at_ms, revision,
                       body_json
                FROM submodule_edges
                WHERE parent_repository_id = ?
                ORDER BY gitlink_path
                """,
                [_text(parent_repository_id, "parent_repository_id")],
            ).fetchall()
            return [self._row_to_edge(_row_mapping(row)) for row in rows]

    # -- worktrees / leases --------------------------------------------------

    def register_worktree(
        self,
        *,
        repository_id: str,
        workspace_path: str | Path,
        branch_name: str = "",
        lane_id: str = "",
        task_id: str = "",
        attempt: int = 0,
        session_id: str = "",
        is_detached: bool = False,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> WorktreeIdentity:
        """Register a worktree identity without yet acquiring a lease."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        repo_id = _text(repository_id, "repository_id")
        path = normalize_workspace_path(workspace_path)
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._require_repository_locked(connection, repo_id)
                existing = connection.execute(
                    """
                    SELECT worktree_id, lifecycle_state, status, lease_id,
                           fencing_token, revision
                    FROM worktrees WHERE workspace_path = ?
                    """,
                    [path],
                ).fetchone()
                if existing is not None:
                    mapping = _row_mapping(existing)
                    status = str(mapping.get("status") or "")
                    lifecycle = str(mapping.get("lifecycle_state") or "")
                    if status not in {
                        WorktreeStatus.TERMINAL.value,
                        WorktreeStatus.RECLAIMED.value,
                        WorktreeStatus.SUPERSEDED.value,
                    } and lifecycle not in {
                        WorktreeLifecycleState.TERMINAL.value,
                    }:
                        raise WorktreeRegistryConflictError(
                            f"workspace path already registered as nonterminal: {path}"
                        )
                    worktree_id = str(mapping.get("worktree_id") or mapping.get("0"))
                    revision = int(mapping.get("revision") or mapping.get("5") or 1) + 1
                else:
                    worktree_id = _new_id("worktree")
                    revision = 1
                identity = WorktreeIdentity(
                    worktree_id=worktree_id,
                    repository_id=repo_id,
                    workspace_path=path,
                    branch_name=branch_name,
                    lane_id=lane_id,
                    task_id=task_id,
                    attempt=int(attempt),
                    session_id=session_id,
                    lifecycle_state=WorktreeLifecycleState.PREPARING,
                    registered_at_ms=now,
                    updated_at_ms=now,
                    revision=revision,
                    status=WorktreeStatus.PREPARING,
                    is_detached=bool(is_detached),
                    body=dict(body or {}),
                )
                self._upsert_worktree_locked(connection, identity)
                self._record_lifecycle_locked(
                    connection,
                    worktree_id=identity.worktree_id,
                    from_state="",
                    to_state=WorktreeLifecycleState.PREPARING.value,
                    fencing_token=0,
                    now_ms=now,
                    reason="register",
                )
                self._commit_if_idle(connection)
                return identity
            except Exception:
                self._rollback_if_open(connection)
                raise

    def acquire_lease(
        self,
        worktree_id: str,
        *,
        process_birth: ProcessBirthIdentity | None = None,
        session_id: str = "",
        ttl_ms: int | None = None,
        expected_fencing_token: int | None = None,
        now_ms: int | None = None,
    ) -> WorktreeIdentity:
        """Acquire or reclaim a worktree lease with CAS fencing."""

        birth = process_birth or current_process_birth()
        if not isinstance(birth, ProcessBirthIdentity):
            raise TypeError("process_birth must be ProcessBirthIdentity")
        if int(birth.pid) <= 0 or int(birth.start_time_ticks) <= 0:
            raise WorktreeRegistryIdentityError(
                "process birth requires pid and start_time_ticks; "
                "raw PID never proves identity"
            )
        liveness = self._liveness(birth)
        if liveness is not OwnerLiveness.ALIVE:
            raise WorktreeRegistryIdentityError(
                f"cannot acquire lease with {liveness.value} process birth"
            )
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        ttl = (
            self._default_lease_ttl_ms
            if ttl_ms is None
            else _positive_int(int(ttl_ms), "ttl_ms")
        )
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_worktree_locked(connection, worktree_id)
                if current is None:
                    raise WorktreeRegistryError(f"unknown worktree: {worktree_id}")
                if expected_fencing_token is not None and int(
                    expected_fencing_token
                ) != int(current.fencing_token):
                    raise WorktreeRegistryConflictError(
                        f"fence CAS mismatch: expected {expected_fencing_token}, "
                        f"observed {current.fencing_token}"
                    )
                if current.is_leased and current.lifecycle_state.is_nonterminal:
                    owner = current.owner_process_birth
                    owner_live = OwnerLiveness.DEAD
                    if owner is not None:
                        owner_live = self._liveness(owner)
                    if owner_live is OwnerLiveness.ALIVE:
                        if process_births_match(owner, birth) and current.lease_id:
                            # Same owner renews in place.
                            renewed = self._with_worktree(
                                current,
                                lease_expires_at_ms=now + ttl,
                                updated_at_ms=now,
                                session_id=session_id or current.session_id,
                                revision=current.revision + 1,
                            )
                            self._upsert_worktree_locked(connection, renewed)
                            self._record_lease_event_locked(
                                connection,
                                worktree_id=renewed.worktree_id,
                                lease_id=renewed.lease_id,
                                fencing_token=renewed.fencing_token,
                                event_kind=LeaseEventKind.RENEW,
                                actor_session_id=session_id,
                                now_ms=now,
                            )
                            self._commit_if_idle(connection)
                            return renewed
                        raise WorktreeRegistryConflictError(
                            "worktree lease is fenced by a live owner"
                        )
                    if owner_live is OwnerLiveness.UNKNOWN:
                        raise WorktreeRegistryIdentityError(
                            "incumbent owner liveness is unknown; fail closed"
                        )
                    # Dead owner: reclaim via fence bump (CAS).
                    reclaimed_from = current.fencing_token
                else:
                    reclaimed_from = None

                new_token = max(1, int(current.fencing_token) + 1)
                new_epoch = max(1, int(current.fence_epoch) + (1 if reclaimed_from else 0))
                if current.fence_epoch == 0:
                    new_epoch = 1
                lease_id = _new_id("lease")
                birth_id = process_birth_id(birth)
                updated = self._with_worktree(
                    current,
                    lease_id=lease_id,
                    fencing_token=new_token,
                    fence_epoch=new_epoch,
                    owner_process_birth=birth,
                    owner_process_birth_id=birth_id,
                    lease_expires_at_ms=now + ttl,
                    session_id=session_id or current.session_id,
                    lifecycle_state=WorktreeLifecycleState.ACTIVE,
                    status=WorktreeStatus.ACTIVE,
                    updated_at_ms=now,
                    revision=current.revision + 1,
                )
                # Fence CAS: advance only when the observed token is still current.
                # Never blind-upsert after a failed compare; concurrent reclaim
                # must lose rather than overwrite a newer fence.
                connection.execute(
                    """
                    UPDATE worktrees SET
                        lease_id = ?, fencing_token = ?, fence_epoch = ?,
                        owner_process_birth_id = ?, owner_process_birth_json = ?,
                        lease_expires_at_ms = ?, session_id = ?,
                        lifecycle_state = ?, status = ?, updated_at_ms = ?,
                        revision = ?, branch_name = ?, head_commit = ?,
                        head_tree = ?, index_digest = ?, dirty_overlay_digest = ?,
                        current_snapshot_id = ?, is_detached = ?, body_json = ?
                    WHERE worktree_id = ? AND fencing_token = ?
                    """,
                    [
                        updated.lease_id,
                        int(updated.fencing_token),
                        int(updated.fence_epoch),
                        updated.owner_process_birth_id,
                        _canonical_json(birth.to_dict()),
                        int(updated.lease_expires_at_ms),
                        updated.session_id,
                        updated.lifecycle_state.value,
                        updated.status.value,
                        int(updated.updated_at_ms),
                        int(updated.revision),
                        updated.branch_name,
                        updated.head_commit,
                        updated.head_tree,
                        updated.index_digest,
                        updated.dirty_overlay_digest,
                        updated.current_snapshot_id,
                        bool(updated.is_detached),
                        _canonical_json(dict(updated.body)),
                        current.worktree_id,
                        int(current.fencing_token),
                    ],
                )
                # Re-read is the durable CAS authority (rowcount is unreliable).
                verify = self._load_worktree_locked(connection, worktree_id)
                if verify is None:
                    raise WorktreeRegistryConflictError(
                        f"fence CAS failed: worktree disappeared during acquire "
                        f"({worktree_id})"
                    )
                cas_ok = (
                    int(verify.fencing_token) == int(updated.fencing_token)
                    and verify.lease_id == updated.lease_id
                    and verify.owner_process_birth_id == updated.owner_process_birth_id
                )
                if not cas_ok:
                    raise WorktreeRegistryConflictError(
                        f"fence CAS mismatch on acquire: expected "
                        f"{current.fencing_token}, observed {verify.fencing_token}"
                    )
                event_kind = (
                    LeaseEventKind.RECLAIM
                    if reclaimed_from is not None
                    else LeaseEventKind.ACQUIRE
                )
                self._record_lease_event_locked(
                    connection,
                    worktree_id=updated.worktree_id,
                    lease_id=updated.lease_id,
                    fencing_token=updated.fencing_token,
                    event_kind=event_kind,
                    actor_session_id=session_id,
                    now_ms=now,
                    body={
                        "prior_fencing_token": int(current.fencing_token),
                        "reclaimed": reclaimed_from is not None,
                    },
                )
                self._record_lifecycle_locked(
                    connection,
                    worktree_id=updated.worktree_id,
                    from_state=current.lifecycle_state.value,
                    to_state=updated.lifecycle_state.value,
                    fencing_token=updated.fencing_token,
                    now_ms=now,
                    reason=event_kind.value,
                )
                self._commit_if_idle(connection)
                return updated
            except Exception:
                self._rollback_if_open(connection)
                raise

    def release_lease(
        self,
        worktree_id: str,
        *,
        lease_id: str,
        fencing_token: int,
        process_birth: ProcessBirthIdentity | None = None,
        terminal_reason: str = "released",
        now_ms: int | None = None,
    ) -> WorktreeIdentity:
        """Release a lease only when lease id and fence match (CAS)."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_worktree_locked(connection, worktree_id)
                if current is None:
                    raise WorktreeRegistryError(f"unknown worktree: {worktree_id}")
                if current.lease_id != _text(lease_id, "lease_id"):
                    raise WorktreeRegistryConflictError("lease_id mismatch on release")
                if int(current.fencing_token) != int(fencing_token):
                    raise WorktreeRegistryConflictError(
                        f"fence CAS mismatch on release: expected {fencing_token}, "
                        f"observed {current.fencing_token}"
                    )
                if process_birth is not None and current.owner_process_birth is not None:
                    if not process_births_match(current.owner_process_birth, process_birth):
                        raise WorktreeRegistryIdentityError(
                            "release process birth does not match lease owner"
                        )
                updated = self._with_worktree(
                    current,
                    lifecycle_state=WorktreeLifecycleState.TERMINAL,
                    status=WorktreeStatus.TERMINAL,
                    lease_expires_at_ms=now,
                    updated_at_ms=now,
                    revision=current.revision + 1,
                    body={**dict(current.body), "terminal_reason": terminal_reason},
                )
                self._upsert_worktree_locked(connection, updated)
                self._record_lease_event_locked(
                    connection,
                    worktree_id=updated.worktree_id,
                    lease_id=lease_id,
                    fencing_token=int(fencing_token),
                    event_kind=LeaseEventKind.RELEASE,
                    now_ms=now,
                    body={"reason": terminal_reason},
                )
                self._record_lifecycle_locked(
                    connection,
                    worktree_id=updated.worktree_id,
                    from_state=current.lifecycle_state.value,
                    to_state=WorktreeLifecycleState.TERMINAL.value,
                    fencing_token=int(fencing_token),
                    now_ms=now,
                    reason=terminal_reason,
                )
                self._commit_if_idle(connection)
                return updated
            except Exception:
                self._rollback_if_open(connection)
                raise

    def reclaim_dead_owner(
        self,
        worktree_id: str,
        *,
        expected_fencing_token: int,
        process_birth: ProcessBirthIdentity | None = None,
        session_id: str = "",
        ttl_ms: int | None = None,
        now_ms: int | None = None,
    ) -> WorktreeIdentity:
        """Reclaim a worktree from a dead owner using fence CAS.

        Validates dead-owner evidence and the caller's expected fence under the
        registry lock, then acquires via the same CAS path as
        :meth:`acquire_lease` so concurrent reclaimers cannot both succeed.
        """

        birth = process_birth or current_process_birth()
        # Preflight under the lock so CAS failures surface as reclaim errors
        # (match="CAS") before the acquire path mutates state.
        with self._lock:
            connection = self._require()
            current = self._load_worktree_locked(connection, worktree_id)
            if current is None:
                raise WorktreeRegistryError(f"unknown worktree: {worktree_id}")
            if int(current.fencing_token) != int(expected_fencing_token):
                raise WorktreeRegistryConflictError(
                    f"fence CAS mismatch on reclaim: expected "
                    f"{expected_fencing_token}, observed {current.fencing_token}"
                )
            owner = current.owner_process_birth
            if owner is None:
                raise WorktreeRegistryIdentityError(
                    "no owner process birth recorded for reclaim"
                )
            live = self._liveness(owner)
            if live is OwnerLiveness.ALIVE:
                raise WorktreeRegistryConflictError(
                    "cannot reclaim: owner process birth is still alive"
                )
            if live is OwnerLiveness.UNKNOWN:
                raise WorktreeRegistryIdentityError(
                    "cannot reclaim: owner liveness is unknown"
                )
        try:
            return self.acquire_lease(
                worktree_id,
                process_birth=birth,
                session_id=session_id,
                ttl_ms=ttl_ms,
                expected_fencing_token=expected_fencing_token,
                now_ms=now_ms,
            )
        except WorktreeRegistryConflictError as exc:
            # Normalize acquire CAS races into reclaim CAS vocabulary.
            message = str(exc)
            if "CAS" in message or "fenced" in message:
                raise WorktreeRegistryConflictError(
                    f"fence CAS mismatch on reclaim: {message}"
                ) from exc
            raise

    def get_worktree(self, worktree_id: str) -> WorktreeIdentity | None:
        with self._lock:
            connection = self._require()
            return self._load_worktree_locked(connection, worktree_id)

    def get_worktree_by_path(
        self, workspace_path: str | Path
    ) -> WorktreeIdentity | None:
        path = normalize_workspace_path(workspace_path)
        with self._lock:
            connection = self._require()
            return self._load_worktree_by_path_locked(connection, path)

    def list_worktrees(
        self, *, repository_id: str | None = None
    ) -> list[WorktreeIdentity]:
        with self._lock:
            connection = self._require()
            if repository_id:
                rows = connection.execute(
                    """
                    SELECT worktree_id FROM worktrees
                    WHERE repository_id = ?
                    ORDER BY workspace_path
                    """,
                    [_text(repository_id, "repository_id")],
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT worktree_id FROM worktrees ORDER BY workspace_path"
                ).fetchall()
            result: list[WorktreeIdentity] = []
            for row in rows:
                mapping = _row_mapping(row)
                worktree_id = str(mapping.get("worktree_id") or mapping.get("0"))
                loaded = self._load_worktree_locked(connection, worktree_id)
                if loaded is not None:
                    result.append(loaded)
            return result

    # -- snapshots / paths / overlays ----------------------------------------

    def record_snapshot(
        self,
        worktree_id: str,
        *,
        observation: GitObservation,
        base_commit: str = "",
        scanner_version: str = "",
        paths: Sequence[Mapping[str, Any]] | None = None,
        lease_id: str | None = None,
        fencing_token: int | None = None,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> WorktreeSnapshot:
        """Record a snapshot from current Git observations (byte authority)."""

        if not isinstance(observation, GitObservation):
            raise TypeError("observation must be GitObservation")
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_worktree_locked(connection, worktree_id)
                if current is None:
                    raise WorktreeRegistryError(f"unknown worktree: {worktree_id}")
                if normalize_workspace_path(observation.workspace_path) != (
                    current.workspace_path
                ):
                    raise WorktreeRegistryObservationError(
                        "observation workspace_path does not match worktree"
                    )
                if lease_id is not None and lease_id != current.lease_id:
                    raise WorktreeRegistryConflictError(
                        "lease_id mismatch when recording snapshot"
                    )
                if fencing_token is not None and int(fencing_token) != int(
                    current.fencing_token
                ):
                    raise WorktreeRegistryConflictError(
                        "fence CAS mismatch when recording snapshot"
                    )
                if not observation.path_exists:
                    raise WorktreeRegistryObservationError(
                        "cannot snapshot a missing worktree path"
                    )
                if observation.is_symlink_root:
                    # Symlink roots are recorded but never broaden containment.
                    pass
                snapshot = WorktreeSnapshot(
                    snapshot_id=_new_id("snapshot"),
                    worktree_id=current.worktree_id,
                    repository_id=current.repository_id,
                    base_commit=base_commit or current.head_commit,
                    head_commit=observation.head_commit,
                    head_tree=observation.head_tree,
                    index_digest=observation.index_digest,
                    dirty_overlay_digest=observation.dirty_overlay_digest,
                    branch_name=observation.branch_name or current.branch_name,
                    is_detached=bool(observation.is_detached),
                    scanner_version=scanner_version,
                    observed_at_ms=observation.observed_at_ms or now,
                    revision=1,
                    body=dict(body or {}),
                )
                connection.execute(
                    """
                    INSERT INTO worktree_snapshots(
                        snapshot_id, worktree_id, repository_id, base_commit,
                        head_commit, head_tree, index_digest, dirty_overlay_digest,
                        branch_name, is_detached, scanner_version, observed_at_ms,
                        revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        snapshot.snapshot_id,
                        snapshot.worktree_id,
                        snapshot.repository_id,
                        snapshot.base_commit,
                        snapshot.head_commit,
                        snapshot.head_tree,
                        snapshot.index_digest,
                        snapshot.dirty_overlay_digest,
                        snapshot.branch_name,
                        bool(snapshot.is_detached),
                        snapshot.scanner_version,
                        int(snapshot.observed_at_ms),
                        int(snapshot.revision),
                        _canonical_json(dict(snapshot.body)),
                    ],
                )
                for raw_path in paths or ():
                    self._insert_path_locked(
                        connection,
                        snapshot_id=snapshot.snapshot_id,
                        worktree_id=current.worktree_id,
                        raw=raw_path,
                    )
                updated = self._with_worktree(
                    current,
                    head_commit=snapshot.head_commit,
                    head_tree=snapshot.head_tree,
                    index_digest=snapshot.index_digest,
                    dirty_overlay_digest=snapshot.dirty_overlay_digest,
                    current_snapshot_id=snapshot.snapshot_id,
                    branch_name=snapshot.branch_name or current.branch_name,
                    is_detached=bool(snapshot.is_detached),
                    updated_at_ms=now,
                    revision=current.revision + 1,
                )
                self._upsert_worktree_locked(connection, updated)
                self._commit_if_idle(connection)
                return snapshot
            except Exception:
                self._rollback_if_open(connection)
                raise

    def record_dirty_overlay(
        self,
        worktree_id: str,
        *,
        snapshot_id: str,
        entries: Sequence[Mapping[str, Any]],
        rename_policy: str = "track",
        delete_policy: str = "track",
        untracked_policy: str = "include",
        lease_id: str | None = None,
        fencing_token: int | None = None,
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> DirtyOverlay:
        """Record a dirty overlay; digest is derived from entries (byte bind)."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        normalized_entries: list[dict[str, Any]] = []
        for raw in entries:
            path = path_contained_in_root(str(raw.get("path") or ""))
            kind = str(raw.get("kind") or OverlayEntryKind.MODIFIED.value)
            try:
                OverlayEntryKind(kind)
            except ValueError as exc:
                raise WorktreeRegistryError(f"unknown overlay entry kind: {kind}") from exc
            from_path = str(raw.get("from_path") or "")
            if from_path:
                from_path = path_contained_in_root(from_path)
            if kind == OverlayEntryKind.RENAMED.value and rename_policy == "reject":
                raise WorktreeRegistryError("rename_policy rejects renamed entries")
            if kind == OverlayEntryKind.DELETED.value and delete_policy == "reject":
                raise WorktreeRegistryError("delete_policy rejects deleted entries")
            if kind == OverlayEntryKind.UNTRACKED.value and untracked_policy == "reject":
                raise WorktreeRegistryError(
                    "untracked_policy rejects untracked entries"
                )
            if kind == OverlayEntryKind.UNTRACKED.value and untracked_policy == "ignore":
                continue
            normalized_entries.append(
                {
                    "kind": kind,
                    "path": path,
                    "from_path": from_path,
                    "blob_id": str(raw.get("blob_id") or ""),
                    "mode": str(raw.get("mode") or ""),
                }
            )
        overlay_digest = digest_overlay_entries(normalized_entries)
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_worktree_locked(connection, worktree_id)
                if current is None:
                    raise WorktreeRegistryError(f"unknown worktree: {worktree_id}")
                if lease_id is not None and lease_id != current.lease_id:
                    raise WorktreeRegistryConflictError(
                        "lease_id mismatch when recording dirty overlay"
                    )
                if fencing_token is not None and int(fencing_token) != int(
                    current.fencing_token
                ):
                    raise WorktreeRegistryConflictError(
                        "fence CAS mismatch when recording dirty overlay"
                    )
                snap_row = connection.execute(
                    """
                    SELECT snapshot_id FROM worktree_snapshots
                    WHERE snapshot_id = ? AND worktree_id = ?
                    """,
                    [_text(snapshot_id, "snapshot_id"), current.worktree_id],
                ).fetchone()
                if snap_row is None:
                    raise WorktreeRegistryError(
                        f"unknown snapshot for worktree: {snapshot_id}"
                    )
                overlay = DirtyOverlay(
                    overlay_id=_new_id("overlay"),
                    snapshot_id=snapshot_id,
                    worktree_id=current.worktree_id,
                    overlay_digest=overlay_digest,
                    entry_count=len(normalized_entries),
                    rename_policy=rename_policy,
                    delete_policy=delete_policy,
                    untracked_policy=untracked_policy,
                    entries=normalized_entries,
                    recorded_at_ms=now,
                    revision=1,
                    body=dict(body or {}),
                )
                connection.execute(
                    """
                    INSERT INTO dirty_overlays(
                        overlay_id, snapshot_id, worktree_id, overlay_digest,
                        entry_count, rename_policy, delete_policy, untracked_policy,
                        entries_json, recorded_at_ms, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        overlay.overlay_id,
                        overlay.snapshot_id,
                        overlay.worktree_id,
                        overlay.overlay_digest,
                        int(overlay.entry_count),
                        overlay.rename_policy,
                        overlay.delete_policy,
                        overlay.untracked_policy,
                        _canonical_json([dict(e) for e in overlay.entries]),
                        int(overlay.recorded_at_ms),
                        int(overlay.revision),
                        _canonical_json(dict(overlay.body)),
                    ],
                )
                connection.execute(
                    """
                    UPDATE worktree_snapshots
                    SET dirty_overlay_digest = ?
                    WHERE snapshot_id = ?
                    """,
                    [overlay.overlay_digest, snapshot_id],
                )
                updated = self._with_worktree(
                    current,
                    dirty_overlay_digest=overlay.overlay_digest,
                    updated_at_ms=now,
                    revision=current.revision + 1,
                )
                self._upsert_worktree_locked(connection, updated)
                self._commit_if_idle(connection)
                return overlay
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_snapshot(self, snapshot_id: str) -> WorktreeSnapshot | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT snapshot_id, worktree_id, repository_id, base_commit,
                       head_commit, head_tree, index_digest, dirty_overlay_digest,
                       branch_name, is_detached, scanner_version, observed_at_ms,
                       revision, body_json
                FROM worktree_snapshots WHERE snapshot_id = ?
                """,
                [_text(snapshot_id, "snapshot_id")],
            ).fetchone()
            if row is None:
                return None
            return self._row_to_snapshot(_row_mapping(row))

    def list_paths(self, snapshot_id: str) -> list[WorktreePathRecord]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT path_id, snapshot_id, worktree_id, relative_path, path_kind,
                       blob_id, symlink_target, is_symlink, is_gitlink,
                       policy_disposition, body_json
                FROM worktree_paths
                WHERE snapshot_id = ?
                ORDER BY relative_path
                """,
                [_text(snapshot_id, "snapshot_id")],
            ).fetchall()
            return [self._row_to_path(_row_mapping(row)) for row in rows]

    def get_dirty_overlay(self, overlay_id: str) -> DirtyOverlay | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT overlay_id, snapshot_id, worktree_id, overlay_digest,
                       entry_count, rename_policy, delete_policy, untracked_policy,
                       entries_json, recorded_at_ms, revision, body_json
                FROM dirty_overlays WHERE overlay_id = ?
                """,
                [_text(overlay_id, "overlay_id")],
            ).fetchone()
            if row is None:
                return None
            return self._row_to_overlay(_row_mapping(row))

    # -- setup cache ---------------------------------------------------------

    def put_setup_cache(
        self,
        *,
        repository_id: str,
        cache_key: str,
        cache_digest: str,
        payload: Mapping[str, Any] | None = None,
        worktree_id: str = "",
        ttl_ms: int = 0,
        now_ms: int | None = None,
    ) -> SetupCacheEntry:
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                self._require_repository_locked(connection, repository_id)
                existing = connection.execute(
                    "SELECT revision FROM setup_cache WHERE cache_key = ?",
                    [_text(cache_key, "cache_key")],
                ).fetchone()
                revision = 1
                if existing is not None:
                    mapping = _row_mapping(existing)
                    revision = int(mapping.get("revision") or mapping.get("0") or 1) + 1
                entry = SetupCacheEntry(
                    cache_key=cache_key,
                    repository_id=repository_id,
                    worktree_id=worktree_id,
                    cache_digest=cache_digest,
                    payload=dict(payload or {}),
                    recorded_at_ms=now,
                    expires_at_ms=(now + int(ttl_ms)) if ttl_ms else 0,
                    revision=revision,
                )
                connection.execute(
                    """
                    INSERT INTO setup_cache(
                        cache_key, repository_id, worktree_id, cache_digest,
                        payload_json, recorded_at_ms, expires_at_ms, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        repository_id = excluded.repository_id,
                        worktree_id = excluded.worktree_id,
                        cache_digest = excluded.cache_digest,
                        payload_json = excluded.payload_json,
                        recorded_at_ms = excluded.recorded_at_ms,
                        expires_at_ms = excluded.expires_at_ms,
                        revision = excluded.revision
                    """,
                    [
                        entry.cache_key,
                        entry.repository_id,
                        entry.worktree_id,
                        entry.cache_digest,
                        _canonical_json(dict(entry.payload)),
                        int(entry.recorded_at_ms),
                        int(entry.expires_at_ms),
                        int(entry.revision),
                    ],
                )
                self._commit_if_idle(connection)
                return entry
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_setup_cache(self, cache_key: str) -> SetupCacheEntry | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT cache_key, repository_id, worktree_id, cache_digest,
                       payload_json, recorded_at_ms, expires_at_ms, revision
                FROM setup_cache WHERE cache_key = ?
                """,
                [_text(cache_key, "cache_key")],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            payload = _parse_json_object(mapping.get("payload_json"))
            return SetupCacheEntry(
                cache_key=str(
                    mapping.get("cache_key") or mapping.get("0") or ""
                ),
                repository_id=str(
                    mapping.get("repository_id") or mapping.get("1") or ""
                ),
                worktree_id=str(mapping.get("worktree_id") or ""),
                cache_digest=str(mapping.get("cache_digest") or ""),
                payload=payload,
                recorded_at_ms=int(mapping.get("recorded_at_ms") or 0),
                expires_at_ms=int(mapping.get("expires_at_ms") or 0),
                revision=int(mapping.get("revision") or 1),
            )

    # -- reuse / cleanup / reconciliation ------------------------------------

    def evaluate_reuse(
        self,
        *,
        workspace_path: str | Path,
        lease_id: str,
        fencing_token: int,
        observation: GitObservation | None = None,
        now_ms: int | None = None,
    ) -> ReuseDecision:
        """Allow reuse only with matching lease and current Git observations."""

        path = normalize_workspace_path(workspace_path)
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            worktree = self._load_worktree_by_path_locked(connection, path)
            if worktree is None:
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="worktree_not_registered",
                    lease_matched=False,
                    observation_matched=False,
                )
            lease_matched = (
                bool(worktree.lease_id)
                and worktree.lease_id == _text(lease_id, "lease_id")
                and int(worktree.fencing_token) == int(fencing_token)
            )
            if not lease_matched:
                self._begin(connection)
                try:
                    self._record_lease_event_locked(
                        connection,
                        worktree_id=worktree.worktree_id,
                        lease_id=lease_id,
                        fencing_token=int(fencing_token),
                        event_kind=LeaseEventKind.DENY,
                        now_ms=now,
                        body={"reason": "lease_mismatch", "operation": "reuse"},
                    )
                    self._commit_if_idle(connection)
                except Exception:
                    self._rollback_if_open(connection)
                    raise
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="lease_mismatch",
                    worktree=worktree,
                    lease_matched=False,
                    observation_matched=False,
                )
            if worktree.lease_expires_at_ms and now > int(worktree.lease_expires_at_ms):
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="lease_expired",
                    worktree=worktree,
                    lease_matched=True,
                    observation_matched=False,
                )
            obs = observation
            if obs is None and self._git_observer is not None:
                obs = self._git_observer(path)
            if obs is None:
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="git_observation_required",
                    worktree=worktree,
                    lease_matched=True,
                    observation_matched=False,
                )
            matched, reason = self._observations_match(worktree, obs)
            if not matched:
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason=reason,
                    worktree=worktree,
                    lease_matched=True,
                    observation_matched=False,
                )
            if worktree.lifecycle_state is WorktreeLifecycleState.TERMINAL:
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="worktree_terminal",
                    worktree=worktree,
                    lease_matched=True,
                    observation_matched=True,
                )
            return ReuseDecision(
                disposition=ReuseDisposition.ALLOW,
                reason="lease_and_git_observation_match",
                worktree=worktree,
                lease_matched=True,
                observation_matched=True,
            )

    def evaluate_cleanup(
        self,
        *,
        workspace_path: str | Path,
        lease_id: str | None = None,
        fencing_token: int | None = None,
        observation: GitObservation | None = None,
        now_ms: int | None = None,
    ) -> ReuseDecision:
        """Allow cleanup only with matching lease (or dead reclaim) and Git obs."""

        path = normalize_workspace_path(workspace_path)
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            worktree = self._load_worktree_by_path_locked(connection, path)
            if worktree is None:
                # Unregistered path: cleanup of unknown dirt requires observation
                # that the path is gone or empty; still fail closed without Git.
                obs = observation
                if obs is None and self._git_observer is not None:
                    obs = self._git_observer(path)
                if obs is None:
                    return ReuseDecision(
                        disposition=ReuseDisposition.DENY,
                        reason="unregistered_requires_git_observation",
                    )
                if not obs.path_exists:
                    return ReuseDecision(
                        disposition=ReuseDisposition.ALLOW,
                        reason="unregistered_path_absent",
                        observation_matched=True,
                    )
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="unregistered_path_present",
                    observation_matched=True,
                )

            obs = observation
            if obs is None and self._git_observer is not None:
                obs = self._git_observer(path)
            if obs is None:
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="git_observation_required",
                    worktree=worktree,
                    lease_matched=False,
                    observation_matched=False,
                )
            if not obs.path_exists:
                # Stale path: registry may still hold a record, but Git says gone.
                # Cleanup of registry state still needs lease or dead-owner reclaim.
                pass

            lease_matched = False
            if lease_id is not None and fencing_token is not None:
                lease_matched = (
                    worktree.lease_id == _text(lease_id, "lease_id")
                    and int(worktree.fencing_token) == int(fencing_token)
                )
            if lease_matched:
                if worktree.lifecycle_state is WorktreeLifecycleState.TERMINAL or (
                    not obs.path_exists
                ):
                    return ReuseDecision(
                        disposition=ReuseDisposition.ALLOW,
                        reason="matching_lease_and_git_observation",
                        worktree=worktree,
                        lease_matched=True,
                        observation_matched=True,
                    )
                # Active lease holder may settle then clean.
                if worktree.lifecycle_state is WorktreeLifecycleState.SETTLING:
                    return ReuseDecision(
                        disposition=ReuseDisposition.ALLOW,
                        reason="matching_lease_settling",
                        worktree=worktree,
                        lease_matched=True,
                        observation_matched=True,
                    )
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="active_lease_not_terminal",
                    worktree=worktree,
                    lease_matched=True,
                    observation_matched=True,
                )

            # No matching lease: allow reclaim-then-clean only for dead owners.
            owner = worktree.owner_process_birth
            if owner is None:
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="lease_mismatch_no_owner",
                    worktree=worktree,
                    lease_matched=False,
                    observation_matched=True,
                )
            live = self._liveness(owner)
            if live is OwnerLiveness.DEAD:
                return ReuseDecision(
                    disposition=ReuseDisposition.RECLAIM_THEN_ALLOW,
                    reason="dead_owner_reclaim_then_cleanup",
                    worktree=worktree,
                    lease_matched=False,
                    observation_matched=True,
                )
            if live is OwnerLiveness.UNKNOWN:
                return ReuseDecision(
                    disposition=ReuseDisposition.DENY,
                    reason="owner_liveness_unknown",
                    worktree=worktree,
                    lease_matched=False,
                    observation_matched=True,
                )
            return ReuseDecision(
                disposition=ReuseDisposition.DENY,
                reason="live_owner_lease_mismatch",
                worktree=worktree,
                lease_matched=False,
                observation_matched=True,
            )

    def reconcile(
        self,
        worktree_id: str,
        *,
        observation: GitObservation,
        lease_id: str | None = None,
        fencing_token: int | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Reconcile registry semantic state with current Git byte observations."""

        if not isinstance(observation, GitObservation):
            raise TypeError("observation must be GitObservation")
        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_worktree_locked(connection, worktree_id)
                if current is None:
                    raise WorktreeRegistryError(f"unknown worktree: {worktree_id}")
                if lease_id is not None and lease_id != current.lease_id:
                    raise WorktreeRegistryConflictError(
                        "lease_id mismatch during reconciliation"
                    )
                if fencing_token is not None and int(fencing_token) != int(
                    current.fencing_token
                ):
                    raise WorktreeRegistryConflictError(
                        "fence CAS mismatch during reconciliation"
                    )
                matched, reason = self._observations_match(current, observation)
                drift: dict[str, Any] = {
                    "matched": matched,
                    "reason": reason,
                    "registry_head_commit": current.head_commit,
                    "observed_head_commit": observation.head_commit,
                    "registry_head_tree": current.head_tree,
                    "observed_head_tree": observation.head_tree,
                    "registry_index_digest": current.index_digest,
                    "observed_index_digest": observation.index_digest,
                    "registry_dirty_overlay_digest": current.dirty_overlay_digest,
                    "observed_dirty_overlay_digest": observation.dirty_overlay_digest,
                    "path_exists": bool(observation.path_exists),
                    "is_symlink_root": bool(observation.is_symlink_root),
                    "is_detached": bool(observation.is_detached),
                }
                if not observation.path_exists:
                    # Stale path: mark settling so cleanup can proceed with lease.
                    updated = self._with_worktree(
                        current,
                        lifecycle_state=WorktreeLifecycleState.SETTLING,
                        status=WorktreeStatus.SETTLING,
                        updated_at_ms=now,
                        revision=current.revision + 1,
                        body={
                            **dict(current.body),
                            "reconciliation": drift,
                            "stale_path": True,
                        },
                    )
                    self._upsert_worktree_locked(connection, updated)
                    self._record_lifecycle_locked(
                        connection,
                        worktree_id=updated.worktree_id,
                        from_state=current.lifecycle_state.value,
                        to_state=updated.lifecycle_state.value,
                        fencing_token=updated.fencing_token,
                        now_ms=now,
                        reason="stale_path_reconciliation",
                    )
                    self._commit_if_idle(connection)
                    return {
                        "status": "stale_path",
                        "worktree": updated.to_dict(),
                        "drift": drift,
                    }
                # Update semantic pointers from Git bytes without inventing bytes.
                updated = self._with_worktree(
                    current,
                    head_commit=observation.head_commit,
                    head_tree=observation.head_tree,
                    index_digest=observation.index_digest,
                    dirty_overlay_digest=observation.dirty_overlay_digest,
                    branch_name=observation.branch_name or current.branch_name,
                    is_detached=bool(observation.is_detached),
                    updated_at_ms=now,
                    revision=current.revision + 1,
                    body={**dict(current.body), "reconciliation": drift},
                )
                self._upsert_worktree_locked(connection, updated)
                self._commit_if_idle(connection)
                return {
                    "status": "reconciled" if matched else "drift_recorded",
                    "worktree": updated.to_dict(),
                    "drift": drift,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    # -- local JSON mirrors (non-authoritative) ------------------------------

    def mirror_local_json(
        self,
        *,
        worktree_id: str,
        mirror_path: str | Path,
        body: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Record a diagnostic local JSON mirror. Never grants authority."""

        now = self._now_ms() if now_ms is None else _nonneg_int(int(now_ms), "now_ms")
        path = normalize_workspace_path(mirror_path)
        payload = _bounded_mapping(body, name="mirror_body", max_bytes=MAX_PAYLOAD_BYTES)
        digest = _sha256_hex(_canonical_json(payload).encode("utf-8"))
        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                current = self._load_worktree_locked(connection, worktree_id)
                if current is None:
                    raise WorktreeRegistryError(f"unknown worktree: {worktree_id}")
                connection.execute(
                    """
                    INSERT INTO local_json_mirrors(
                        mirror_path, worktree_id, content_digest, written_at_ms,
                        body_json
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (mirror_path) DO UPDATE SET
                        worktree_id = excluded.worktree_id,
                        content_digest = excluded.content_digest,
                        written_at_ms = excluded.written_at_ms,
                        body_json = excluded.body_json
                    """,
                    [
                        path,
                        current.worktree_id,
                        digest,
                        now,
                        _canonical_json(payload),
                    ],
                )
                self._commit_if_idle(connection)
                return {
                    "schema": LOCAL_JSON_MIRROR_SCHEMA,
                    "mirror_path": path,
                    "worktree_id": current.worktree_id,
                    "content_digest": digest,
                    "written_at_ms": now,
                    "authoritative": False,
                }
            except Exception:
                self._rollback_if_open(connection)
                raise

    def apply_local_json_index(
        self,
        *,
        mirror_path: str | Path,
        claimed_worktree: Mapping[str, Any],
    ) -> None:
        """Reject any attempt to override registry state from local JSON."""

        path = normalize_workspace_path(mirror_path)
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT worktree_id, content_digest, body_json
                FROM local_json_mirrors WHERE mirror_path = ?
                """,
                [path],
            ).fetchone()
            # Whether or not a mirror exists, local JSON cannot mutate authority.
            raise WorktreeRegistryAuthorityError(
                "worktree-local JSON index cannot override registry state; "
                f"path={path} claimed_keys={sorted(str(k) for k in claimed_worktree)}"
            )

    # -- internal helpers ----------------------------------------------------

    def _require_repository_locked(self, connection: Any, repository_id: str) -> None:
        row = connection.execute(
            "SELECT repository_id FROM repositories WHERE repository_id = ?",
            [_text(repository_id, "repository_id")],
        ).fetchone()
        if row is None:
            raise WorktreeRegistryError(f"unknown repository: {repository_id}")

    def _upsert_ref_locked(
        self,
        connection: Any,
        *,
        repository_id: str,
        ref_name: str,
        object_id: str,
        ref_kind: str = "branch",
        body: Mapping[str, Any] | None = None,
        now_ms: int,
    ) -> GitRefRecord:
        name = _text(ref_name, "ref_name")
        existing = connection.execute(
            """
            SELECT ref_id, revision FROM git_refs
            WHERE repository_id = ? AND ref_name = ?
            """,
            [_text(repository_id, "repository_id"), name],
        ).fetchone()
        if existing is not None:
            mapping = _row_mapping(existing)
            ref_id = str(mapping.get("ref_id") or mapping.get("0"))
            revision = int(mapping.get("revision") or mapping.get("1") or 1) + 1
        else:
            ref_id = _new_id("ref")
            revision = 1
        record = GitRefRecord(
            ref_id=ref_id,
            repository_id=repository_id,
            ref_name=name,
            object_id=object_id,
            ref_kind=ref_kind,
            registered_at_ms=now_ms,
            revision=revision,
            body=dict(body or {}),
        )
        connection.execute(
            """
            INSERT INTO git_refs(
                ref_id, repository_id, ref_name, object_id, ref_kind,
                registered_at_ms, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ref_id) DO UPDATE SET
                repository_id = excluded.repository_id,
                ref_name = excluded.ref_name,
                object_id = excluded.object_id,
                ref_kind = excluded.ref_kind,
                registered_at_ms = excluded.registered_at_ms,
                revision = excluded.revision,
                body_json = excluded.body_json
            """,
            [
                record.ref_id,
                record.repository_id,
                record.ref_name,
                record.object_id,
                record.ref_kind,
                int(record.registered_at_ms),
                int(record.revision),
                _canonical_json(dict(record.body)),
            ],
        )
        return record

    def _upsert_worktree_locked(
        self, connection: Any, identity: WorktreeIdentity
    ) -> None:
        birth_json = "{}"
        if identity.owner_process_birth is not None:
            birth_json = _canonical_json(identity.owner_process_birth.to_dict())
        connection.execute(
            """
            INSERT INTO worktrees(
                worktree_id, repository_id, workspace_path, branch_name, lane_id,
                task_id, attempt, session_id, lease_id, fencing_token, fence_epoch,
                owner_process_birth_id, owner_process_birth_json, lifecycle_state,
                lease_expires_at_ms, head_commit, head_tree, index_digest,
                dirty_overlay_digest, current_snapshot_id, is_detached,
                registered_at_ms, updated_at_ms, revision, status, body_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            ON CONFLICT (worktree_id) DO UPDATE SET
                repository_id = excluded.repository_id,
                workspace_path = excluded.workspace_path,
                branch_name = excluded.branch_name,
                lane_id = excluded.lane_id,
                task_id = excluded.task_id,
                attempt = excluded.attempt,
                session_id = excluded.session_id,
                lease_id = excluded.lease_id,
                fencing_token = excluded.fencing_token,
                fence_epoch = excluded.fence_epoch,
                owner_process_birth_id = excluded.owner_process_birth_id,
                owner_process_birth_json = excluded.owner_process_birth_json,
                lifecycle_state = excluded.lifecycle_state,
                lease_expires_at_ms = excluded.lease_expires_at_ms,
                head_commit = excluded.head_commit,
                head_tree = excluded.head_tree,
                index_digest = excluded.index_digest,
                dirty_overlay_digest = excluded.dirty_overlay_digest,
                current_snapshot_id = excluded.current_snapshot_id,
                is_detached = excluded.is_detached,
                registered_at_ms = excluded.registered_at_ms,
                updated_at_ms = excluded.updated_at_ms,
                revision = excluded.revision,
                status = excluded.status,
                body_json = excluded.body_json
            """,
            [
                identity.worktree_id,
                identity.repository_id,
                identity.workspace_path,
                identity.branch_name,
                identity.lane_id,
                identity.task_id,
                int(identity.attempt),
                identity.session_id,
                identity.lease_id,
                int(identity.fencing_token),
                int(identity.fence_epoch),
                identity.owner_process_birth_id,
                birth_json,
                identity.lifecycle_state.value,
                int(identity.lease_expires_at_ms),
                identity.head_commit,
                identity.head_tree,
                identity.index_digest,
                identity.dirty_overlay_digest,
                identity.current_snapshot_id,
                bool(identity.is_detached),
                int(identity.registered_at_ms),
                int(identity.updated_at_ms),
                int(identity.revision),
                identity.status.value,
                _canonical_json(dict(identity.body)),
            ],
        )

    def _load_worktree_locked(
        self, connection: Any, worktree_id: str
    ) -> WorktreeIdentity | None:
        row = connection.execute(
            """
            SELECT worktree_id, repository_id, workspace_path, branch_name, lane_id,
                   task_id, attempt, session_id, lease_id, fencing_token, fence_epoch,
                   owner_process_birth_id, owner_process_birth_json, lifecycle_state,
                   lease_expires_at_ms, head_commit, head_tree, index_digest,
                   dirty_overlay_digest, current_snapshot_id, is_detached,
                   registered_at_ms, updated_at_ms, revision, status, body_json
            FROM worktrees WHERE worktree_id = ?
            """,
            [_text(worktree_id, "worktree_id")],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_worktree(_row_mapping(row))

    def _load_worktree_by_path_locked(
        self, connection: Any, workspace_path: str
    ) -> WorktreeIdentity | None:
        row = connection.execute(
            """
            SELECT worktree_id FROM worktrees WHERE workspace_path = ?
            """,
            [workspace_path],
        ).fetchone()
        if row is None:
            return None
        mapping = _row_mapping(row)
        worktree_id = str(mapping.get("worktree_id") or mapping.get("0"))
        if not worktree_id:
            return None
        return self._load_worktree_locked(connection, worktree_id)

    def _insert_path_locked(
        self,
        connection: Any,
        *,
        snapshot_id: str,
        worktree_id: str,
        raw: Mapping[str, Any],
    ) -> WorktreePathRecord:
        relative = path_contained_in_root(str(raw.get("relative_path") or raw.get("path") or ""))
        kind_raw = str(raw.get("path_kind") or raw.get("kind") or PathKind.FILE.value)
        kind = _parse_enum(kind_raw, PathKind, "path_kind")
        disposition_raw = str(
            raw.get("policy_disposition")
            or raw.get("disposition")
            or PathPolicyDisposition.TRACKED.value
        )
        disposition = _parse_enum(
            disposition_raw, PathPolicyDisposition, "policy_disposition"
        )
        record = WorktreePathRecord(
            path_id=_new_id("path"),
            snapshot_id=snapshot_id,
            worktree_id=worktree_id,
            relative_path=relative,
            path_kind=kind,  # type: ignore[arg-type]
            blob_id=str(raw.get("blob_id") or ""),
            symlink_target=str(raw.get("symlink_target") or ""),
            is_symlink=bool(raw.get("is_symlink") or kind is PathKind.SYMLINK),
            is_gitlink=bool(raw.get("is_gitlink") or kind is PathKind.GITLINK),
            policy_disposition=disposition,  # type: ignore[arg-type]
            body=dict(raw.get("body") or {}),
        )
        connection.execute(
            """
            INSERT INTO worktree_paths(
                path_id, snapshot_id, worktree_id, relative_path, path_kind,
                blob_id, symlink_target, is_symlink, is_gitlink,
                policy_disposition, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.path_id,
                record.snapshot_id,
                record.worktree_id,
                record.relative_path,
                record.path_kind.value,
                record.blob_id,
                record.symlink_target,
                bool(record.is_symlink),
                bool(record.is_gitlink),
                record.policy_disposition.value,
                _canonical_json(dict(record.body)),
            ],
        )
        return record

    def _record_lease_event_locked(
        self,
        connection: Any,
        *,
        worktree_id: str,
        lease_id: str,
        fencing_token: int,
        event_kind: LeaseEventKind,
        actor_session_id: str = "",
        now_ms: int,
        body: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO lease_events(
                event_id, worktree_id, lease_id, fencing_token, event_kind,
                actor_session_id, recorded_at_ms, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                _new_id("lease-event"),
                worktree_id,
                lease_id,
                int(fencing_token),
                event_kind.value,
                actor_session_id,
                int(now_ms),
                _canonical_json(dict(body or {})),
            ],
        )

    def _record_lifecycle_locked(
        self,
        connection: Any,
        *,
        worktree_id: str,
        from_state: str,
        to_state: str,
        fencing_token: int,
        now_ms: int,
        reason: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO lifecycle_transitions(
                transition_id, worktree_id, from_state, to_state, fencing_token,
                recorded_at_ms, reason, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                _new_id("lifecycle"),
                worktree_id,
                from_state,
                to_state,
                int(fencing_token),
                int(now_ms),
                reason,
                _canonical_json(dict(body or {})),
            ],
        )

    @staticmethod
    def _observations_match(
        worktree: WorktreeIdentity, observation: GitObservation
    ) -> tuple[bool, str]:
        if normalize_workspace_path(observation.workspace_path) != worktree.workspace_path:
            return False, "workspace_path_mismatch"
        if not observation.path_exists:
            return False, "path_missing"
        # When registry has not yet recorded Git digests, accept first observation.
        if worktree.head_commit and observation.head_commit:
            if worktree.head_commit != observation.head_commit:
                return False, "head_commit_mismatch"
        if worktree.head_tree and observation.head_tree:
            if worktree.head_tree != observation.head_tree:
                return False, "head_tree_mismatch"
        if worktree.index_digest and observation.index_digest:
            if worktree.index_digest != observation.index_digest:
                return False, "index_digest_mismatch"
        if worktree.dirty_overlay_digest and observation.dirty_overlay_digest:
            if worktree.dirty_overlay_digest != observation.dirty_overlay_digest:
                return False, "dirty_overlay_digest_mismatch"
        if worktree.branch_name and observation.branch_name and not observation.is_detached:
            if worktree.branch_name != observation.branch_name:
                return False, "branch_mismatch"
        if worktree.is_detached != observation.is_detached and worktree.head_commit:
            # Detached state drift is meaningful once a snapshot is bound.
            if worktree.is_detached or observation.is_detached:
                # Allow if commits still match (checked above).
                pass
        return True, "ok"

    @staticmethod
    def _with_worktree(identity: WorktreeIdentity, **changes: Any) -> WorktreeIdentity:
        payload = identity.to_dict()
        payload.pop("schema", None)
        payload.pop("interface", None)
        owner = payload.pop("owner_process_birth", None)
        payload.pop("owner_process_birth_id", None)
        for key, value in changes.items():
            if key == "owner_process_birth":
                owner = None if value is None else value.to_dict()
                continue
            if key == "owner_process_birth_id":
                payload["owner_process_birth_id"] = value
                continue
            if key == "lifecycle_state" and isinstance(value, WorktreeLifecycleState):
                payload["lifecycle_state"] = value.value
                continue
            if key == "status" and isinstance(value, WorktreeStatus):
                payload["status"] = value.value
                continue
            if key == "body" and isinstance(value, Mapping):
                payload["body"] = dict(value)
                continue
            payload[key] = value
        birth = None
        if isinstance(owner, Mapping):
            birth = ProcessBirthIdentity.from_dict(owner)
        elif "owner_process_birth" in changes and changes["owner_process_birth"] is not None:
            birth = changes["owner_process_birth"]
        return WorktreeIdentity(
            worktree_id=str(payload["worktree_id"]),
            repository_id=str(payload["repository_id"]),
            workspace_path=str(payload["workspace_path"]),
            branch_name=str(payload.get("branch_name") or ""),
            lane_id=str(payload.get("lane_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            attempt=int(payload.get("attempt") or 0),
            session_id=str(payload.get("session_id") or ""),
            lease_id=str(payload.get("lease_id") or ""),
            fencing_token=int(payload.get("fencing_token") or 0),
            fence_epoch=int(payload.get("fence_epoch") or 0),
            owner_process_birth=birth,
            owner_process_birth_id=str(payload.get("owner_process_birth_id") or ""),
            lifecycle_state=WorktreeLifecycleState(
                str(payload.get("lifecycle_state") or "preparing")
            ),
            lease_expires_at_ms=int(payload.get("lease_expires_at_ms") or 0),
            head_commit=str(payload.get("head_commit") or ""),
            head_tree=str(payload.get("head_tree") or ""),
            index_digest=str(payload.get("index_digest") or ""),
            dirty_overlay_digest=str(payload.get("dirty_overlay_digest") or ""),
            current_snapshot_id=str(payload.get("current_snapshot_id") or ""),
            is_detached=bool(payload.get("is_detached")),
            registered_at_ms=int(payload.get("registered_at_ms") or 0),
            updated_at_ms=int(payload.get("updated_at_ms") or 0),
            revision=int(payload.get("revision") or 1),
            status=WorktreeStatus(str(payload.get("status") or "preparing")),
            body=dict(payload.get("body") or {}),
        )

    @staticmethod
    def _row_to_repository(mapping: Mapping[str, Any]) -> RepositoryRecord:
        body = _parse_json_object(mapping.get("body_json"))
        return RepositoryRecord(
            repository_id=str(
                mapping.get("repository_id") or mapping.get("0") or ""
            ),
            git_common_dir=str(
                mapping.get("git_common_dir") or mapping.get("1") or ""
            ),
            canonical_root=str(
                mapping.get("canonical_root") or mapping.get("2") or ""
            ),
            remote_url=str(mapping.get("remote_url") or ""),
            head_commit=str(mapping.get("head_commit") or ""),
            head_tree=str(mapping.get("head_tree") or ""),
            server_generation=int(mapping.get("server_generation") or 1),
            registered_at_ms=int(mapping.get("registered_at_ms") or 0),
            revision=int(mapping.get("revision") or 1),
            status=RepositoryStatus(str(mapping.get("status") or "active")),
            body=body,
        )

    @staticmethod
    def _row_to_branch(mapping: Mapping[str, Any]) -> BranchRecord:
        body = _parse_json_object(mapping.get("body_json"))
        return BranchRecord(
            branch_id=str(mapping.get("branch_id") or mapping.get("0") or ""),
            repository_id=str(
                mapping.get("repository_id") or mapping.get("1") or ""
            ),
            branch_name=str(
                mapping.get("branch_name") or mapping.get("2") or ""
            ),
            tip_commit=str(mapping.get("tip_commit") or ""),
            is_detached=_as_bool(mapping.get("is_detached")),
            upstream=str(mapping.get("upstream") or ""),
            registered_at_ms=int(mapping.get("registered_at_ms") or 0),
            revision=int(mapping.get("revision") or 1),
            body=body,
        )

    @staticmethod
    def _row_to_edge(mapping: Mapping[str, Any]) -> SubmoduleEdge:
        body = _parse_json_object(mapping.get("body_json"))
        return SubmoduleEdge(
            edge_id=str(mapping.get("edge_id") or mapping.get("0") or ""),
            parent_repository_id=str(
                mapping.get("parent_repository_id") or mapping.get("1") or ""
            ),
            child_repository_id=str(
                mapping.get("child_repository_id") or mapping.get("2") or ""
            ),
            gitlink_path=str(
                mapping.get("gitlink_path") or mapping.get("3") or ""
            ),
            gitlink_commit=str(mapping.get("gitlink_commit") or ""),
            registered_at_ms=int(mapping.get("registered_at_ms") or 0),
            revision=int(mapping.get("revision") or 1),
            body=body,
        )

    @staticmethod
    def _row_to_worktree(mapping: Mapping[str, Any]) -> WorktreeIdentity:
        body = _parse_json_object(mapping.get("body_json"))
        birth_payload = _parse_json_object(
            mapping.get("owner_process_birth_json")
        )
        birth = None
        if birth_payload:
            birth = ProcessBirthIdentity.from_dict(birth_payload)
        return WorktreeIdentity(
            worktree_id=str(
                mapping.get("worktree_id") or mapping.get("0") or ""
            ),
            repository_id=str(
                mapping.get("repository_id") or mapping.get("1") or ""
            ),
            workspace_path=str(
                mapping.get("workspace_path") or mapping.get("2") or ""
            ),
            branch_name=str(mapping.get("branch_name") or ""),
            lane_id=str(mapping.get("lane_id") or ""),
            task_id=str(mapping.get("task_id") or ""),
            attempt=int(mapping.get("attempt") or 0),
            session_id=str(mapping.get("session_id") or ""),
            lease_id=str(mapping.get("lease_id") or ""),
            fencing_token=int(mapping.get("fencing_token") or 0),
            fence_epoch=int(mapping.get("fence_epoch") or 0),
            owner_process_birth=birth,
            owner_process_birth_id=str(
                mapping.get("owner_process_birth_id") or ""
            ),
            lifecycle_state=WorktreeLifecycleState(
                str(mapping.get("lifecycle_state") or "preparing")
            ),
            lease_expires_at_ms=int(mapping.get("lease_expires_at_ms") or 0),
            head_commit=str(mapping.get("head_commit") or ""),
            head_tree=str(mapping.get("head_tree") or ""),
            index_digest=str(mapping.get("index_digest") or ""),
            dirty_overlay_digest=str(mapping.get("dirty_overlay_digest") or ""),
            current_snapshot_id=str(mapping.get("current_snapshot_id") or ""),
            is_detached=_as_bool(mapping.get("is_detached")),
            registered_at_ms=int(mapping.get("registered_at_ms") or 0),
            updated_at_ms=int(mapping.get("updated_at_ms") or 0),
            revision=int(mapping.get("revision") or 1),
            status=WorktreeStatus(str(mapping.get("status") or "preparing")),
            body=body,
        )

    @staticmethod
    def _row_to_snapshot(mapping: Mapping[str, Any]) -> WorktreeSnapshot:
        body = _parse_json_object(mapping.get("body_json"))
        return WorktreeSnapshot(
            snapshot_id=str(
                mapping.get("snapshot_id") or mapping.get("0") or ""
            ),
            worktree_id=str(
                mapping.get("worktree_id") or mapping.get("1") or ""
            ),
            repository_id=str(
                mapping.get("repository_id") or mapping.get("2") or ""
            ),
            base_commit=str(mapping.get("base_commit") or ""),
            head_commit=str(mapping.get("head_commit") or ""),
            head_tree=str(mapping.get("head_tree") or ""),
            index_digest=str(mapping.get("index_digest") or ""),
            dirty_overlay_digest=str(mapping.get("dirty_overlay_digest") or ""),
            branch_name=str(mapping.get("branch_name") or ""),
            is_detached=_as_bool(mapping.get("is_detached")),
            scanner_version=str(mapping.get("scanner_version") or ""),
            observed_at_ms=int(mapping.get("observed_at_ms") or 0),
            revision=int(mapping.get("revision") or 1),
            body=body,
        )

    @staticmethod
    def _row_to_path(mapping: Mapping[str, Any]) -> WorktreePathRecord:
        body = _parse_json_object(mapping.get("body_json"))
        return WorktreePathRecord(
            path_id=str(mapping.get("path_id") or mapping.get("0") or ""),
            snapshot_id=str(
                mapping.get("snapshot_id") or mapping.get("1") or ""
            ),
            worktree_id=str(
                mapping.get("worktree_id") or mapping.get("2") or ""
            ),
            relative_path=str(
                mapping.get("relative_path") or mapping.get("3") or ""
            ),
            path_kind=PathKind(str(mapping.get("path_kind") or "file")),
            blob_id=str(mapping.get("blob_id") or ""),
            symlink_target=str(mapping.get("symlink_target") or ""),
            is_symlink=_as_bool(mapping.get("is_symlink")),
            is_gitlink=_as_bool(mapping.get("is_gitlink")),
            policy_disposition=PathPolicyDisposition(
                str(mapping.get("policy_disposition") or "tracked")
            ),
            body=body,
        )

    @staticmethod
    def _row_to_overlay(mapping: Mapping[str, Any]) -> DirtyOverlay:
        body = _parse_json_object(mapping.get("body_json"))
        entries = _parse_json_list(mapping.get("entries_json"))
        return DirtyOverlay(
            overlay_id=str(
                mapping.get("overlay_id") or mapping.get("0") or ""
            ),
            snapshot_id=str(
                mapping.get("snapshot_id") or mapping.get("1") or ""
            ),
            worktree_id=str(
                mapping.get("worktree_id") or mapping.get("2") or ""
            ),
            overlay_digest=str(
                mapping.get("overlay_digest") or mapping.get("3") or ""
            ),
            entry_count=int(mapping.get("entry_count") or 0),
            rename_policy=str(mapping.get("rename_policy") or "track"),
            delete_policy=str(mapping.get("delete_policy") or "track"),
            untracked_policy=str(mapping.get("untracked_policy") or "include"),
            entries=entries,
            recorded_at_ms=int(mapping.get("recorded_at_ms") or 0),
            revision=int(mapping.get("revision") or 1),
            body=body,
        )


def open_worktree_registry(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    liveness: LivenessProbe | None = None,
    birth_reader: BirthReader | None = None,
    git_observer: GitObserver | None = None,
    default_lease_ttl_ms: int = DEFAULT_LEASE_TTL_MS,
    server_generation: int | None = None,
) -> DatabaseWorktreeRegistry:
    """Open and return an initialized :class:`DatabaseWorktreeRegistry`."""

    return DatabaseWorktreeRegistry(
        database_path,
        clock_ms=clock_ms,
        liveness=liveness,
        birth_reader=birth_reader,
        git_observer=git_observer,
        default_lease_ttl_ms=default_lease_ttl_ms,
        server_generation=server_generation,
    ).open()


__all__ = (
    "DATABASE_WORKTREE_REGISTRY_INTERFACE",
    "DATABASE_WORKTREE_REGISTRY_SCHEMA",
    "DEFAULT_LEASE_TTL_MS",
    "DIRTY_OVERLAY_INTERFACE",
    "DIRTY_OVERLAY_SCHEMA",
    "REPOSITORY_FOREST_INTERFACE",
    "WORKTREE_IDENTITY_INTERFACE",
    "WORKTREE_SNAPSHOT_INTERFACE",
    "BranchRecord",
    "DatabaseWorktreeRegistry",
    "DirtyOverlay",
    "DuckDBUnavailableError",
    "GitObservation",
    "GitRefRecord",
    "LeaseEventKind",
    "OverlayEntryKind",
    "OwnerLiveness",
    "PathKind",
    "PathPolicyDisposition",
    "ProcessBirthIdentity",
    "RepositoryRecord",
    "RepositoryStatus",
    "ReuseDecision",
    "ReuseDisposition",
    "SetupCacheEntry",
    "SubmoduleEdge",
    "WorktreeIdentity",
    "WorktreeLifecycleState",
    "WorktreePathRecord",
    "WorktreeRegistryAuthorityError",
    "WorktreeRegistryBoundsError",
    "WorktreeRegistryConflictError",
    "WorktreeRegistryContainmentError",
    "WorktreeRegistryError",
    "WorktreeRegistryIdentityError",
    "WorktreeRegistryNotOpenError",
    "WorktreeRegistryObservationError",
    "WorktreeSnapshot",
    "WorktreeStatus",
    "digest_overlay_entries",
    "duckdb_available",
    "normalize_relative_path",
    "normalize_workspace_path",
    "open_worktree_registry",
    "path_contained_in_root",
    "process_birth_id",
    "process_births_match",
)
