"""Canonical program-behavior identities for clean and dirty Git worktrees.

The decision runtime must authorize the program that will actually execute,
not merely the commit named by ``HEAD``.  This module builds a bounded,
location-independent snapshot over three distinct views:

* committed entries from ``HEAD``;
* staged entries from the Git index; and
* regular-file and symlink bytes currently visible in the worktree, including
  ignored, dot-prefixed, and otherwise untracked files inside the declared
  scope.

Source bodies are never embedded in a snapshot.  They are represented by
``BlobReference`` values from :mod:`artifact_store`; callers may supply a
``BoundedArtifactStore`` to persist those referenced bytes.  Python source is
parsed transiently into the existing path-independent ``ASTBlobRecord`` form
and fed through the incremental ``AnalysisASTIndex``.

Snapshot construction is deliberately fail-closed.  It rejects path and
symlink escapes, special files, unmerged index entries, unreadable or
oversized inputs, scan races, unsupported effects, and any Git or filesystem
change observed by the mandatory post-build verification pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .analysis_ast_index import AnalysisASTIndex, build_analysis_ast_index
from .artifact_store import BlobReference, BoundedArtifactStore
from .conflict_graph import ASTBlobRecord, build_python_ast_blob_record


PROGRAM_BEHAVIOR_SCHEMA_VERSION = 1
PROGRAM_BEHAVIOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-behavior@1"
)
REPOSITORY_SNAPSHOT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/repository-snapshot@1"
)
PROGRAM_OBSERVATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-observations@1"
)
TOOL_CATALOG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/tool-catalog@1"
)
ENVIRONMENT_FACTS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/environment-facts@1"
)
PROPOSED_EFFECT_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proposed-effect-manifest@1"
)

DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_FILES = 8_192
DEFAULT_MAX_OBSERVATIONS = 65_536
DEFAULT_MAX_EFFECTS = 256
DEFAULT_MAX_TOOL_OUTPUT_BYTES = 16_384
DEFAULT_MAX_MANIFEST_BYTES = 16 * 1024 * 1024

_SOURCE_BODY_KEYS = frozenset(
    {
        "body",
        "bytes",
        "content",
        "contents",
        "data",
        "payload",
        "secret",
        "source",
        "source_body",
        "source_text",
        "text",
        "token",
        "value",
    }
)


class ProgramBehaviorError(RuntimeError):
    """Base exception for an unsafe or malformed behavior snapshot."""


class RepositoryPathEscapeError(ProgramBehaviorError, ValueError):
    """A declared repository path is outside its repository root."""


class SymlinkEscapeError(ProgramBehaviorError, ValueError):
    """A symlink resolves outside the declared repository root."""


class RepositoryRaceError(ProgramBehaviorError):
    """A required input changed during or after hashing."""


class RequiredInputUnreadableError(ProgramBehaviorError):
    """A required repository or Git object could not be read safely."""


class RequiredInputTooLargeError(ProgramBehaviorError, ValueError):
    """A required input exceeded a hard byte or item bound."""


class UnsupportedEffectError(ProgramBehaviorError, ValueError):
    """A proposed effect is not in the closed effect vocabulary."""


class RepositoryStateError(ProgramBehaviorError):
    """Git metadata is missing, corrupt, or not representable exactly."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProgramBehaviorError(
            "program behavior values must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    return (
        f"{prefix}:sha256:"
        + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    )


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _frozen_json(value: Any, *, name: str, depth: int = 0) -> Any:
    """Validate, deeply freeze, and prohibit embedded bodies/secrets."""

    if depth > 24:
        raise ProgramBehaviorError(f"{name} exceeds its nesting bound")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        raise ProgramBehaviorError(f"{name} cannot contain floating values")
    if isinstance(value, str):
        if "\x00" in value:
            raise ProgramBehaviorError(f"{name} cannot contain NUL")
        if len(value.encode("utf-8")) > 16_384:
            raise RequiredInputTooLargeError(f"{name} text is oversized")
        return value
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise RequiredInputTooLargeError(f"{name} has too many fields")
        result: dict[str, Any] = {}
        for raw_key in sorted(value):
            key = str(raw_key)
            if key.casefold() in _SOURCE_BODY_KEYS:
                raise ProgramBehaviorError(
                    f"{name} cannot embed body or credential field {key!r}"
                )
            result[key] = _frozen_json(
                value[raw_key], name=name, depth=depth + 1
            )
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 1_024:
            raise RequiredInputTooLargeError(f"{name} has too many items")
        return tuple(
            _frozen_json(item, name=name, depth=depth + 1)
            for item in value
        )
    raise ProgramBehaviorError(f"{name} contains a non-JSON value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _repo_path(value: Any, *, allow_root: bool = False) -> str:
    raw = str(value if value is not None else "").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    candidate = PurePosixPath(raw or ".")
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise RepositoryPathEscapeError(
            f"repository path escapes its root: {value!r}"
        )
    normalized = candidate.as_posix()
    if normalized == ".":
        if allow_root:
            return "."
        raise RepositoryPathEscapeError("repository entry path is required")
    if normalized != raw.rstrip("/"):
        raise RepositoryPathEscapeError(
            f"repository path is not canonical: {value!r}"
        )
    return normalized


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _path_in_scope(path: str, scopes: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    for scope in scopes:
        if scope == ".":
            return True
        root = PurePosixPath(scope)
        if candidate == root or root in candidate.parents:
            return True
    return False


def _path_excluded(path: str, exclusions: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    return any(
        candidate == PurePosixPath(item)
        or PurePosixPath(item) in candidate.parents
        for item in exclusions
    )


def _run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    allow_failure: bool = False,
) -> bytes:
    command = (
        "git",
        "-c",
        "core.quotepath=false",
        "-C",
        str(root),
        *arguments,
    )
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RequiredInputUnreadableError("git executable is unavailable") from exc
    if result.returncode and not allow_failure:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RepositoryStateError(
            f"git {' '.join(arguments)} failed: {detail or result.returncode}"
        )
    return result.stdout if not result.returncode else b""


@dataclass(frozen=True)
class SnapshotBounds:
    """Hard resource bounds for a repository snapshot."""

    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_files: int = DEFAULT_MAX_FILES
    max_observations: int = DEFAULT_MAX_OBSERVATIONS

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("max_file_bytes cannot exceed max_total_bytes")

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


class RepositoryEntryKind(str, Enum):
    REGULAR = "regular"
    SYMLINK = "symlink"


class RepositoryEntryStatus(str, Enum):
    CLEAN = "clean"
    MODIFIED = "modified"
    STAGED = "staged"
    STAGED_AND_MODIFIED = "staged_and_modified"
    DELETED = "deleted"
    STAGED_DELETION = "staged_deletion"
    UNTRACKED = "untracked"
    RENAMED = "renamed"
    MODE_CHANGED = "mode_changed"


@dataclass(frozen=True)
class RepositoryEntry:
    """One path projected across HEAD, index, and current worktree."""

    path: str
    kind: RepositoryEntryKind
    status: RepositoryEntryStatus
    head_mode: str = ""
    head_object_id: str = ""
    head_blob: BlobReference | None = None
    index_mode: str = ""
    index_object_id: str = ""
    index_blob: BlobReference | None = None
    worktree_mode: str = ""
    worktree_blob: BlobReference | None = None
    rename_from: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "kind", RepositoryEntryKind(self.kind))
        object.__setattr__(self, "status", RepositoryEntryStatus(self.status))
        if self.rename_from:
            object.__setattr__(
                self, "rename_from", _repo_path(self.rename_from)
            )
        for name in ("head_mode", "index_mode", "worktree_mode"):
            mode = str(getattr(self, name) or "")
            if mode and mode not in {"100644", "100755", "120000"}:
                raise RepositoryStateError(
                    f"unsupported Git mode {mode!r} for {self.path}"
                )
            object.__setattr__(self, name, mode)
        for name in ("head_object_id", "index_object_id"):
            value = str(getattr(self, name) or "")
            if value and (
                len(value) not in {40, 64}
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise RepositoryStateError(
                    f"invalid Git object identity for {self.path}"
                )
            object.__setattr__(self, name, value)

    @property
    def executed_blob(self) -> BlobReference | None:
        return self.worktree_blob

    def _content_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "status": self.status.value,
            "head_mode": self.head_mode,
            "head_object_id": self.head_object_id,
            "head_blob": self.head_blob.to_dict() if self.head_blob else None,
            "index_mode": self.index_mode,
            "index_object_id": self.index_object_id,
            "index_blob": self.index_blob.to_dict() if self.index_blob else None,
            "worktree_mode": self.worktree_mode,
            "worktree_blob": (
                self.worktree_blob.to_dict() if self.worktree_blob else None
            ),
            "rename_from": self.rename_from,
        }

    @property
    def entry_id(self) -> str:
        return _identity("repository-entry", self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"entry_id": self.entry_id, **self._content_dict()}


@dataclass(frozen=True)
class RepositorySnapshotStats:
    entry_count: int
    executed_file_count: int
    unique_blob_count: int
    hashed_bytes: int
    reused_blob_count: int
    changed_entry_count: int
    untracked_entry_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class RepositorySnapshot:
    """Immutable, bounded identity of all decision-relevant repository bytes."""

    repository_root: str
    git_directory: str
    head_commit_id: str
    head_tree_id: str
    index_tree_id: str
    scopes: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    entries: tuple[RepositoryEntry, ...]
    stats: RepositorySnapshotStats
    bounds: SnapshotBounds
    schema_version: int = PROGRAM_BEHAVIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        root = Path(self.repository_root)
        if not root.is_absolute() or root == Path("/"):
            raise RepositoryPathEscapeError(
                "repository_root must be a non-root absolute path"
            )
        object.__setattr__(self, "repository_root", str(root))
        object.__setattr__(self, "git_directory", str(self.git_directory))
        object.__setattr__(
            self, "scopes", tuple(sorted({_repo_path(x, allow_root=True) for x in self.scopes}))
        )
        object.__setattr__(
            self,
            "excluded_paths",
            tuple(sorted({_repo_path(x) for x in self.excluded_paths})),
        )
        entries = tuple(sorted(self.entries, key=lambda item: item.path))
        if len(entries) != len({item.path for item in entries}):
            raise RepositoryStateError("repository snapshot paths must be unique")
        object.__setattr__(self, "entries", entries)
        if int(self.schema_version) != PROGRAM_BEHAVIOR_SCHEMA_VERSION:
            raise RepositoryStateError("unsupported repository snapshot version")

    @property
    def is_clean(self) -> bool:
        # A native Git tree is equivalent only to an exhaustive worktree
        # snapshot.  A path-limited snapshot remains a distinct behavior root
        # even if every path it happened to inspect was clean.
        return (
            bool(self.head_tree_id)
            and self.head_tree_id == self.index_tree_id
            and self.scopes == (".",)
            and not self.excluded_paths
            and all(
                item.status is RepositoryEntryStatus.CLEAN
                for item in self.entries
            )
        )

    @property
    def dirty(self) -> bool:
        return not self.is_clean

    def _content_dict(self) -> dict[str, Any]:
        # Filesystem locations are verification metadata, not behavior.  This
        # keeps identities identical across clones of the same exact snapshot.
        return {
            "schema": REPOSITORY_SNAPSHOT_SCHEMA,
            "schema_version": self.schema_version,
            "head_commit_id": self.head_commit_id,
            "head_tree_id": self.head_tree_id,
            "index_tree_id": self.index_tree_id,
            "scopes": list(self.scopes),
            "excluded_paths": list(self.excluded_paths),
            "entries": [item.to_dict() for item in self.entries],
            "bounds": self.bounds.to_dict(),
        }

    @property
    def snapshot_id(self) -> str:
        return _identity("repository-snapshot", self._content_dict())

    @property
    def repository_behavior_root(self) -> str:
        return self.snapshot_id

    @property
    def execution_tree_root(self) -> str:
        # Exact clean equivalence is deliberately explicit: when the scoped
        # index and filesystem agree with HEAD, the execution root is the
        # native Git tree identity rather than a parallel digest.
        return self.head_tree_id if self.is_clean else self.snapshot_id

    @property
    def dirty_worktree_root(self) -> str:
        return self.execution_tree_root

    def entry_for_path(self, path: str) -> RepositoryEntry | None:
        normalized = _repo_path(path)
        return next(
            (item for item in self.entries if item.path == normalized), None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "snapshot_id": self.snapshot_id,
            "execution_tree_root": self.execution_tree_root,
            "is_clean": self.is_clean,
            "stats": self.stats.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("utf-8")

    def verify_unchanged(self) -> None:
        """Re-hash the declared repository and reject any changed input."""

        current = build_repository_snapshot(
            self.repository_root,
            scopes=self.scopes,
            excluded_paths=self.excluded_paths,
            bounds=self.bounds,
            verify_after_hash=False,
        )
        if current.snapshot_id != self.snapshot_id:
            raise RepositoryRaceError(
                "repository changed after its behavior snapshot was hashed"
            )


@dataclass(frozen=True)
class _GitEntry:
    mode: str
    object_id: str


@dataclass(frozen=True)
class _WorktreeFile:
    path: str
    mode: str
    kind: RepositoryEntryKind
    data: bytes


def _parse_head_entries(root: Path) -> dict[str, _GitEntry]:
    output = _run_git(
        root,
        ("ls-tree", "-rz", "--full-tree", "HEAD"),
        allow_failure=True,
    )
    result: dict[str, _GitEntry] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RepositoryStateError("HEAD contains an undecodable entry") from exc
        if kind != "blob":
            # Submodules are executable external state and cannot be safely
            # represented as in-repository bytes by this contract.
            raise RepositoryStateError(
                f"unsupported HEAD entry kind {kind!r} at {path!r}"
            )
        result[_repo_path(path)] = _GitEntry(mode, object_id)
    return result


def _parse_index_entries(root: Path) -> dict[str, _GitEntry]:
    output = _run_git(root, ("ls-files", "--stage", "-z"))
    result: dict[str, _GitEntry] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RepositoryStateError("index contains an undecodable entry") from exc
        if stage != "0":
            raise RepositoryStateError(
                f"unmerged index entry is unsupported at {path!r}"
            )
        result[_repo_path(path)] = _GitEntry(mode, object_id)
    return result


def _safe_scope_path(root: Path, relative: str) -> Path:
    candidate = root if relative == "." else root.joinpath(*PurePosixPath(relative).parts)
    # The repository root's parent is necessarily outside the repository; for
    # the "." scope the already canonicalized root itself is the containment
    # anchor.
    resolved_parent = (
        root if candidate == root else candidate.parent.resolve(strict=True)
    )
    if not _is_within(resolved_parent, root):
        raise SymlinkEscapeError(
            f"scope or parent symlink escapes repository root: {relative!r}"
        )
    if candidate.exists() and not candidate.is_symlink():
        resolved = candidate.resolve(strict=True)
        if not _is_within(resolved, root):
            raise SymlinkEscapeError(
                f"scope escapes repository root: {relative!r}"
            )
    return candidate


def _stable_read(root: Path, path: Path, relative: str, bound: int) -> _WorktreeFile:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RequiredInputUnreadableError(
            f"required input is unreadable: {relative}"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        try:
            target = os.readlink(path)
            after = path.lstat()
        except OSError as exc:
            raise RequiredInputUnreadableError(
                f"required symlink is unreadable: {relative}"
            ) from exc
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RepositoryRaceError(f"symlink changed while hashing: {relative}")
        resolved = (path.parent / target).resolve(strict=False)
        if not _is_within(resolved, root):
            raise SymlinkEscapeError(
                f"symlink escapes repository root: {relative!r} -> {target!r}"
            )
        data = os.fsencode(target)
        if len(data) > bound:
            raise RequiredInputTooLargeError(
                f"required input exceeds {bound} bytes: {relative}"
            )
        return _WorktreeFile(
            relative, "120000", RepositoryEntryKind.SYMLINK, data
        )
    if not stat.S_ISREG(before.st_mode):
        raise RequiredInputUnreadableError(
            f"required input is not a regular file or symlink: {relative}"
        )
    if before.st_size > bound:
        raise RequiredInputTooLargeError(
            f"required input exceeds {bound} bytes: {relative}"
        )
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_size,
                opened.st_mtime_ns,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
            ):
                raise RepositoryRaceError(
                    f"file changed before hashing: {relative}"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, bound + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > bound:
                    raise RequiredInputTooLargeError(
                        f"required input exceeds {bound} bytes: {relative}"
                    )
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final_path = path.lstat()
    except ProgramBehaviorError:
        raise
    except OSError as exc:
        raise RequiredInputUnreadableError(
            f"required input is unreadable: {relative}"
        ) from exc
    signatures = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    if signatures != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ) or signatures != (
        final_path.st_dev,
        final_path.st_ino,
        final_path.st_mode,
        final_path.st_size,
        final_path.st_mtime_ns,
    ):
        raise RepositoryRaceError(f"file changed while hashing: {relative}")
    mode = "100755" if before.st_mode & 0o111 else "100644"
    return _WorktreeFile(
        relative, mode, RepositoryEntryKind.REGULAR, b"".join(chunks)
    )


def _scan_worktree(
    root: Path,
    scopes: Sequence[str],
    exclusions: Sequence[str],
    bounds: SnapshotBounds,
) -> dict[str, _WorktreeFile]:
    result: dict[str, _WorktreeFile] = {}

    def traversal_error(error: OSError) -> None:
        raise RequiredInputUnreadableError(
            f"required repository directory is unreadable: {error.filename}"
        ) from error

    def add(path: Path, relative: str) -> None:
        normalized = _repo_path(relative)
        if normalized == ".git" or normalized.startswith(".git/"):
            return
        if _path_excluded(normalized, exclusions):
            return
        if len(result) >= bounds.max_files:
            raise RequiredInputTooLargeError(
                f"repository scope exceeds {bounds.max_files} files"
            )
        result[normalized] = _stable_read(
            root, path, normalized, bounds.max_file_bytes
        )

    for scope in scopes:
        start = _safe_scope_path(root, scope)
        if not start.exists() and not start.is_symlink():
            continue
        if start.is_file() or start.is_symlink():
            add(start, scope)
            continue
        for directory, names, files in os.walk(
            start,
            topdown=True,
            onerror=traversal_error,
            followlinks=False,
        ):
            directory_path = Path(directory)
            relative_directory = directory_path.relative_to(root).as_posix()
            if relative_directory == ".git" or relative_directory.startswith(".git/"):
                names[:] = ()
                continue
            kept_names: list[str] = []
            for name in sorted(names):
                child = directory_path / name
                relative = child.relative_to(root).as_posix()
                if relative == ".git" or _path_excluded(relative, exclusions):
                    continue
                if child.is_symlink():
                    add(child, relative)
                else:
                    kept_names.append(name)
            names[:] = kept_names
            for name in sorted(files):
                child = directory_path / name
                add(child, child.relative_to(root).as_posix())
    # A symlink inside the byte scope must not smuggle in behavior from an
    # excluded or out-of-scope target.  Full-root scans naturally admit
    # internal targets, while path-limited scans must declare them too.
    for relative, item in result.items():
        if item.kind is not RepositoryEntryKind.SYMLINK:
            continue
        target = os.fsdecode(item.data)
        resolved = (
            root.joinpath(*PurePosixPath(relative).parts).parent / target
        ).resolve(strict=False)
        if not _is_within(resolved, root):
            # _stable_read already catches this; retain the assertion here so
            # future alternate scanners cannot bypass the invariant.
            raise SymlinkEscapeError(
                f"symlink escapes repository root: {relative!r}"
            )
        target_relative = resolved.relative_to(root).as_posix()
        if (
            not _path_in_scope(target_relative, scopes)
            or _path_excluded(target_relative, exclusions)
        ):
            raise SymlinkEscapeError(
                f"symlink target is outside the declared byte scope: {relative!r}"
            )
    return result


def _blob_from_bytes(
    data: bytes,
    *,
    kind: str,
    store: BoundedArtifactStore | None,
    cache: dict[str, BlobReference],
) -> BlobReference:
    digest = _sha256(data)
    existing = cache.get(digest)
    if existing is not None:
        if existing.size_bytes != len(data):
            raise RepositoryStateError("conflicting blob size for one digest")
        if store is None or store.verify_blob(existing):
            return existing
    if store is not None:
        reference = store.put_blob(
            data,
            kind=kind,
            media_type="application/octet-stream",
        )
    else:
        reference = BlobReference(
            artifact_id=f"blob:{digest}",
            digest=digest,
            size_bytes=len(data),
            kind=kind,
        )
    cache[digest] = reference
    return reference


def _git_blob(
    root: Path,
    object_id: str,
    *,
    bounds: SnapshotBounds,
    store: BoundedArtifactStore | None,
    cache: dict[str, BlobReference],
    counted: set[str],
    byte_counter: list[int],
) -> BlobReference:
    data = _run_git(root, ("cat-file", "blob", object_id))
    if len(data) > bounds.max_file_bytes:
        raise RequiredInputTooLargeError(
            f"required Git blob {object_id} exceeds {bounds.max_file_bytes} bytes"
        )
    digest = _sha256(data)
    if digest not in counted:
        byte_counter[0] += len(data)
        counted.add(digest)
    if byte_counter[0] > bounds.max_total_bytes:
        raise RequiredInputTooLargeError(
            f"repository inputs exceed {bounds.max_total_bytes} bytes"
        )
    return _blob_from_bytes(
        data, kind="repository-source", store=store, cache=cache
    )


def _entry_status(
    head: _GitEntry | None,
    index: _GitEntry | None,
    worktree: _WorktreeFile | None,
    head_blob: BlobReference | None,
    index_blob: BlobReference | None,
    worktree_blob: BlobReference | None,
    rename_from: str,
) -> RepositoryEntryStatus:
    if head is None and index is None:
        return RepositoryEntryStatus.UNTRACKED
    if worktree is None:
        if index is None:
            return RepositoryEntryStatus.STAGED_DELETION
        return RepositoryEntryStatus.DELETED
    head_equal_index = bool(
        head
        and index
        and head.mode == index.mode
        and head_blob == index_blob
    )
    index_equal_worktree = bool(
        index
        and index.mode == worktree.mode
        and index_blob == worktree_blob
    )
    if rename_from:
        return RepositoryEntryStatus.RENAMED
    if head_equal_index and index_equal_worktree:
        return RepositoryEntryStatus.CLEAN
    if (
        head_blob == index_blob == worktree_blob
        and head
        and index
        and (head.mode != index.mode or index.mode != worktree.mode)
    ):
        return RepositoryEntryStatus.MODE_CHANGED
    if head_equal_index:
        return RepositoryEntryStatus.MODIFIED
    if index_equal_worktree:
        return RepositoryEntryStatus.STAGED
    return RepositoryEntryStatus.STAGED_AND_MODIFIED


def build_repository_snapshot(
    repository_root: str | os.PathLike[str],
    *,
    scopes: Sequence[str] = (".",),
    excluded_paths: Sequence[str] = (),
    bounds: SnapshotBounds | None = None,
    artifact_store: BoundedArtifactStore | None = None,
    previous: RepositorySnapshot | None = None,
    verify_after_hash: bool = True,
) -> RepositorySnapshot:
    """Build and post-verify an exact, bounded Git/worktree snapshot."""

    selected_bounds = bounds or SnapshotBounds()
    requested = Path(repository_root)
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise RequiredInputUnreadableError(
            "repository root is unreadable"
        ) from exc
    discovered_raw = _run_git(root, ("rev-parse", "--show-toplevel"))
    try:
        discovered = Path(discovered_raw.decode("utf-8").strip()).resolve(
            strict=True
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise RepositoryStateError("could not resolve Git repository root") from exc
    if discovered != root:
        raise RepositoryPathEscapeError(
            "repository_root must name the exact Git worktree root"
        )
    normalized_scopes = tuple(
        sorted({_repo_path(item, allow_root=True) for item in scopes})
    )
    if not normalized_scopes:
        raise RepositoryPathEscapeError("at least one repository scope is required")
    normalized_exclusions = tuple(
        sorted({_repo_path(item) for item in excluded_paths})
    )
    for scope in normalized_scopes:
        _safe_scope_path(root, scope)

    git_directory = _run_git(
        root, ("rev-parse", "--absolute-git-dir")
    ).decode("utf-8", "strict").strip()
    head_commit = _run_git(
        root, ("rev-parse", "--verify", "HEAD"), allow_failure=True
    ).decode("ascii", "strict").strip()
    head_tree = _run_git(
        root, ("rev-parse", "--verify", "HEAD^{tree}"), allow_failure=True
    ).decode("ascii", "strict").strip()
    # write-tree is the canonical Git index tree operation.  It does not alter
    # the index or worktree, and fails for an unmerged index.
    index_tree = _run_git(root, ("write-tree",)).decode("ascii", "strict").strip()

    head_all = _parse_head_entries(root)
    index_all = _parse_index_entries(root)
    in_scope = lambda path: (
        _path_in_scope(path, normalized_scopes)
        and not _path_excluded(path, normalized_exclusions)
    )
    head = {path: item for path, item in head_all.items() if in_scope(path)}
    index = {path: item for path, item in index_all.items() if in_scope(path)}
    worktree = _scan_worktree(
        root, normalized_scopes, normalized_exclusions, selected_bounds
    )

    paths = sorted(set(head) | set(index) | set(worktree))
    if len(paths) > selected_bounds.max_files:
        raise RequiredInputTooLargeError(
            f"repository scope exceeds {selected_bounds.max_files} entries"
        )
    cache: dict[str, BlobReference] = {}
    if previous is not None:
        for entry in previous.entries:
            for reference in (
                entry.head_blob,
                entry.index_blob,
                entry.worktree_blob,
            ):
                if reference is not None:
                    cache[reference.digest] = reference
    initially_cached = set(cache)
    counted: set[str] = set()
    byte_counter = [0]
    git_cache: dict[str, BlobReference] = {}

    def git_reference(entry: _GitEntry | None) -> BlobReference | None:
        if entry is None:
            return None
        result = git_cache.get(entry.object_id)
        if result is None:
            result = _git_blob(
                root,
                entry.object_id,
                bounds=selected_bounds,
                store=artifact_store,
                cache=cache,
                counted=counted,
                byte_counter=byte_counter,
            )
            git_cache[entry.object_id] = result
        return result

    worktree_refs: dict[str, BlobReference] = {}
    for path, item in sorted(worktree.items()):
        digest = _sha256(item.data)
        if digest not in counted:
            byte_counter[0] += len(item.data)
            counted.add(digest)
        if byte_counter[0] > selected_bounds.max_total_bytes:
            raise RequiredInputTooLargeError(
                f"repository inputs exceed {selected_bounds.max_total_bytes} bytes"
            )
        worktree_refs[path] = _blob_from_bytes(
            item.data,
            kind="repository-source",
            store=artifact_store,
            cache=cache,
        )

    # Infer exact staged or unstaged renames only when a newly introduced
    # worktree/index path has one unique vanished HEAD source with the same
    # bytes.  Ambiguous copies stay ordinary additions/deletions.
    deleted_by_digest: dict[str, list[str]] = {}
    for path, item in head.items():
        if path not in worktree:
            reference = git_reference(item)
            assert reference is not None
            deleted_by_digest.setdefault(reference.digest, []).append(path)
    rename_from: dict[str, str] = {}
    for path in sorted(set(index) | set(worktree)):
        if path in head:
            continue
        reference = worktree_refs.get(path) or git_reference(index.get(path))
        if reference is None:
            continue
        candidates = deleted_by_digest.get(reference.digest, ())
        if len(candidates) == 1:
            rename_from[path] = candidates[0]

    entries: list[RepositoryEntry] = []
    for path in paths:
        head_item = head.get(path)
        index_item = index.get(path)
        worktree_item = worktree.get(path)
        head_blob = git_reference(head_item)
        index_blob = git_reference(index_item)
        worktree_blob = worktree_refs.get(path)
        kind = (
            worktree_item.kind
            if worktree_item is not None
            else RepositoryEntryKind.SYMLINK
            if (index_item or head_item)
            and (index_item or head_item).mode == "120000"
            else RepositoryEntryKind.REGULAR
        )
        source = rename_from.get(path, "")
        entries.append(
            RepositoryEntry(
                path=path,
                kind=kind,
                status=_entry_status(
                    head_item,
                    index_item,
                    worktree_item,
                    head_blob,
                    index_blob,
                    worktree_blob,
                    source,
                ),
                head_mode=head_item.mode if head_item else "",
                head_object_id=head_item.object_id if head_item else "",
                head_blob=head_blob,
                index_mode=index_item.mode if index_item else "",
                index_object_id=index_item.object_id if index_item else "",
                index_blob=index_blob,
                worktree_mode=worktree_item.mode if worktree_item else "",
                worktree_blob=worktree_blob,
                rename_from=source,
            )
        )

    referenced_digests = {
        reference.digest
        for entry in entries
        for reference in (
            entry.head_blob,
            entry.index_blob,
            entry.worktree_blob,
        )
        if reference is not None
    }
    snapshot = RepositorySnapshot(
        repository_root=str(root),
        git_directory=git_directory,
        head_commit_id=head_commit,
        head_tree_id=head_tree,
        index_tree_id=index_tree,
        scopes=normalized_scopes,
        excluded_paths=normalized_exclusions,
        entries=tuple(entries),
        stats=RepositorySnapshotStats(
            entry_count=len(entries),
            executed_file_count=len(worktree),
            unique_blob_count=len(counted),
            hashed_bytes=byte_counter[0],
            reused_blob_count=len(
                referenced_digests.intersection(initially_cached)
            ),
            changed_entry_count=sum(
                item.status is not RepositoryEntryStatus.CLEAN
                for item in entries
            ),
            untracked_entry_count=sum(
                item.status is RepositoryEntryStatus.UNTRACKED
                for item in entries
            ),
        ),
        bounds=selected_bounds,
    )
    if verify_after_hash:
        verification = build_repository_snapshot(
            root,
            scopes=normalized_scopes,
            excluded_paths=normalized_exclusions,
            bounds=selected_bounds,
            artifact_store=None,
            previous=snapshot,
            verify_after_hash=False,
        )
        if verification.snapshot_id != snapshot.snapshot_id:
            raise RepositoryRaceError(
                "repository changed after its behavior snapshot was hashed"
            )
    return snapshot


class ProgramObservationKind(str, Enum):
    AST = "ast"
    SYMBOL = "symbol"
    INTERFACE = "interface"
    CALL = "call"
    DATA_FLOW = "data_flow"


@dataclass(frozen=True)
class ProgramObservation:
    """Compact AST fact bound to an exact source reference."""

    kind: ProgramObservationKind
    path: str
    source_blob: BlobReference
    ast_record_id: str
    subject: str
    relationship: str
    target: str
    symbol_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProgramObservationKind(self.kind))
        object.__setattr__(self, "path", _repo_path(self.path))
        for name in (
            "ast_record_id",
            "subject",
            "relationship",
            "target",
            "symbol_hash",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value and name not in {"subject", "symbol_hash"}:
                raise ProgramBehaviorError(
                    f"program observation {name} is required"
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "source_blob": self.source_blob.to_dict(),
            "ast_record_id": self.ast_record_id,
            "subject": self.subject,
            "relationship": self.relationship,
            "target": self.target,
            "symbol_hash": self.symbol_hash,
        }


@dataclass(frozen=True)
class ProgramAnalysis:
    """Incremental AST index plus compact behavior observations."""

    ast_index: AnalysisASTIndex
    observations: tuple[ProgramObservation, ...]
    ast_index_blob: BlobReference
    observations_blob: BlobReference

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                self.observations,
                key=lambda item: (
                    item.path,
                    item.kind.value,
                    item.subject,
                    item.relationship,
                    item.target,
                ),
            )
        )
        object.__setattr__(self, "observations", ordered)

    @property
    def program_root(self) -> str:
        return _identity(
            "program-analysis",
            {
                "schema": PROGRAM_OBSERVATION_SCHEMA,
                "ast_index_id": self.ast_index.index_id,
                "ast_index_blob": self.ast_index_blob.to_dict(),
                "observations_blob": self.observations_blob.to_dict(),
            },
        )

    @property
    def ast_root(self) -> str:
        return self.ast_index.index_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_OBSERVATION_SCHEMA,
            "program_root": self.program_root,
            "ast_root": self.ast_root,
            "ast_index_blob": self.ast_index_blob.to_dict(),
            "observations_blob": self.observations_blob.to_dict(),
            "observation_count": len(self.observations),
            "ast_stats": self.ast_index.stats.to_dict(),
        }


def _observations_for(
    path: str,
    source: BlobReference,
    record: ASTBlobRecord,
) -> Iterable[ProgramObservation]:
    yield ProgramObservation(
        ProgramObservationKind.AST,
        path,
        source,
        record.record_id,
        "",
        "parses_as",
        record.language if not record.parse_error else f"error:{record.parse_error}",
    )
    for symbol in record.qualified_symbols:
        yield ProgramObservation(
            ProgramObservationKind.SYMBOL,
            path,
            source,
            record.record_id,
            symbol,
            "defines",
            symbol,
            record.symbol_hashes.get(symbol, ""),
        )
    for interface in record.interfaces:
        yield ProgramObservation(
            ProgramObservationKind.INTERFACE,
            path,
            source,
            record.record_id,
            interface.split(":", 1)[0],
            "provides",
            interface,
        )
    for call in record.calls:
        owner, separator, target = call.partition("->")
        yield ProgramObservation(
            ProgramObservationKind.CALL,
            path,
            source,
            record.record_id,
            owner if separator else "",
            "calls",
            target if separator else call,
        )
        yield ProgramObservation(
            ProgramObservationKind.DATA_FLOW,
            path,
            source,
            record.record_id,
            owner if separator else "",
            "value_flows_to_call",
            target if separator else call,
        )
    for imported in record.imports:
        yield ProgramObservation(
            ProgramObservationKind.DATA_FLOW,
            path,
            source,
            record.record_id,
            "",
            "imports",
            imported,
        )
    for transition in record.state_transitions:
        owner = transition.split(":", 1)[0]
        yield ProgramObservation(
            ProgramObservationKind.DATA_FLOW,
            path,
            source,
            record.record_id,
            owner,
            "state_transition",
            transition,
        )


def build_program_analysis(
    snapshot: RepositorySnapshot,
    *,
    previous: ProgramAnalysis | None = None,
    artifact_store: BoundedArtifactStore | None = None,
) -> ProgramAnalysis:
    """Parse current Python bytes and incrementally reuse unchanged AST facts."""

    prior_records = (
        {
            item.path: item.ast_record
            for item in previous.ast_index.path_records
        }
        if previous is not None
        else {}
    )
    path_records: list[tuple[str, ASTBlobRecord]] = []
    observations: list[ProgramObservation] = []
    for entry in snapshot.entries:
        reference = entry.worktree_blob
        if reference is None or not entry.path.endswith(".py"):
            continue
        prior = prior_records.get(entry.path)
        if (
            prior is not None
            and prior.source_sha256 == reference.digest
            and prior.blob_identity == reference.artifact_id
        ):
            record = prior
        else:
            absolute = Path(snapshot.repository_root).joinpath(
                *PurePosixPath(entry.path).parts
            )
            worktree = _stable_read(
                Path(snapshot.repository_root),
                absolute,
                entry.path,
                snapshot.bounds.max_file_bytes,
            )
            if _sha256(worktree.data) != reference.digest:
                raise RepositoryRaceError(
                    f"Python source changed after snapshot: {entry.path}"
                )
            try:
                source = worktree.data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RequiredInputUnreadableError(
                    f"Python source is not UTF-8: {entry.path}"
                ) from exc
            record = build_python_ast_blob_record(
                source,
                blob_identity=reference.artifact_id,
                source_sha256=reference.digest,
            )
        path_records.append((entry.path, record))
        observations.extend(_observations_for(entry.path, reference, record))
        if len(observations) > snapshot.bounds.max_observations:
            raise RequiredInputTooLargeError(
                "program observations exceed the declared bound"
            )
    index = build_analysis_ast_index(
        path_records,
        previous=previous.ast_index if previous is not None else None,
    )
    observation_dicts = [item.to_dict() for item in sorted(
        observations,
        key=lambda item: (
            item.path,
            item.kind.value,
            item.subject,
            item.relationship,
            item.target,
        ),
    )]
    cache: dict[str, BlobReference] = {}
    # Cache statistics and invalidation history describe how this snapshot was
    # reached, not the current program.  Keep the referenced AST projection
    # cold/warm equivalent just as AnalysisASTIndex.index_id is.
    ast_bytes = _canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/analysis-ast-index@1",
            "schema_version": index.schema_version,
            "index_id": index.index_id,
            "path_records": [
                item.to_dict() for item in index.path_records
            ],
        }
    )
    observation_bytes = _canonical_json_bytes(
        {"schema": PROGRAM_OBSERVATION_SCHEMA, "observations": observation_dicts}
    )
    for label, encoded in (
        ("AST index", ast_bytes),
        ("program observations", observation_bytes),
    ):
        if len(encoded) > min(
            snapshot.bounds.max_total_bytes, DEFAULT_MAX_MANIFEST_BYTES
        ):
            raise RequiredInputTooLargeError(
                f"{label} exceeds its serialized byte bound"
            )
    ast_ref = _blob_from_bytes(
        ast_bytes,
        kind="analysis-ast-index",
        store=artifact_store,
        cache=cache,
    )
    observation_ref = _blob_from_bytes(
        observation_bytes,
        kind="program-observations",
        store=artifact_store,
        cache=cache,
    )
    return ProgramAnalysis(
        ast_index=index,
        observations=tuple(observations),
        ast_index_blob=ast_ref,
        observations_blob=observation_ref,
    )


@dataclass(frozen=True)
class ToolDescriptor:
    """Versioned tool fact without mutable executable output bodies."""

    tool_id: str
    version: str
    executable: str
    version_digest: str

    def __post_init__(self) -> None:
        for name in ("tool_id", "version", "executable"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ProgramBehaviorError(f"tool {name} is required")
            object.__setattr__(self, name, value)
        digest = str(self.version_digest)
        if (
            not digest.startswith("sha256:")
            or len(digest) != 71
            or any(char not in "0123456789abcdef" for char in digest[7:])
        ):
            raise ProgramBehaviorError("tool version_digest is not canonical")

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_id": self.tool_id,
            "version": self.version,
            "executable": self.executable,
            "version_digest": self.version_digest,
        }


@dataclass(frozen=True)
class ToolCatalog:
    tools: tuple[ToolDescriptor, ...]

    def __post_init__(self) -> None:
        tools = tuple(sorted(self.tools, key=lambda item: item.tool_id))
        if len(tools) != len({item.tool_id for item in tools}):
            raise ProgramBehaviorError("tool catalog IDs must be unique")
        object.__setattr__(self, "tools", tools)

    @property
    def catalog_root(self) -> str:
        return _identity(
            "tool-catalog",
            {
                "schema": TOOL_CATALOG_SCHEMA,
                "tools": [item.to_dict() for item in self.tools],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TOOL_CATALOG_SCHEMA,
            "catalog_root": self.catalog_root,
            "tools": [item.to_dict() for item in self.tools],
        }


def capture_tool_catalog(
    tools: Mapping[str, Sequence[str]] | None = None,
) -> ToolCatalog:
    """Capture exact version output for behavior-affecting executables."""

    commands: Mapping[str, Sequence[str]] = (
        {
            "git": ("git", "--version"),
            "python": (sys.executable, "--version"),
        }
        if tools is None
        else tools
    )
    if len(commands) > 256:
        raise RequiredInputTooLargeError("tool catalog exceeds 256 tools")
    descriptors: list[ToolDescriptor] = []
    for tool_id, command in sorted(commands.items()):
        if isinstance(command, str) or not command:
            raise ProgramBehaviorError(
                f"tool command for {tool_id!r} must be a non-empty sequence"
            )
        try:
            result = subprocess.run(
                tuple(str(item) for item in command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as exc:
            raise RequiredInputUnreadableError(
                f"required tool is unavailable: {tool_id}"
            ) from exc
        output = result.stdout
        if len(output) > DEFAULT_MAX_TOOL_OUTPUT_BYTES:
            raise RequiredInputTooLargeError(
                f"tool version output is oversized: {tool_id}"
            )
        if result.returncode:
            raise RequiredInputUnreadableError(
                f"tool version command failed: {tool_id}"
            )
        version = output.decode("utf-8", "replace").strip()
        if not version:
            raise RequiredInputUnreadableError(
                f"tool version command returned no version: {tool_id}"
            )
        requested_executable = str(command[0])
        resolved_executable = shutil.which(requested_executable)
        if resolved_executable is None:
            raise RequiredInputUnreadableError(
                f"required tool is unavailable: {tool_id}"
            )
        executable = str(Path(resolved_executable).resolve())
        descriptors.append(
            ToolDescriptor(
                tool_id=str(tool_id),
                version=version,
                executable=executable,
                version_digest=_sha256(output),
            )
        )
    return ToolCatalog(tuple(descriptors))


@dataclass(frozen=True)
class EnvironmentFacts:
    """Explicit platform/toolchain and selected environment inputs."""

    operating_system: str
    operating_system_release: str
    platform_id: str
    machine: str
    byte_order: str
    filesystem_encoding: str
    python_implementation: str
    python_version: str
    python_cache_tag: str
    variables: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        variables: dict[str, str] = {}
        for key, value in sorted(dict(self.variables).items()):
            name = str(key)
            if not name or "=" in name or "\x00" in name:
                raise ProgramBehaviorError("environment variable name is invalid")
            rendered = str(value)
            if (
                not rendered.startswith("sha256:")
                or len(rendered) != 71
                or any(
                    char not in "0123456789abcdef" for char in rendered[7:]
                )
            ):
                raise ProgramBehaviorError(
                    "environment variables must contain only value digests"
                )
            variables[name] = rendered
        object.__setattr__(self, "variables", MappingProxyType(variables))

    @property
    def environment_root(self) -> str:
        return _identity("environment-facts", self._content_dict())

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": ENVIRONMENT_FACTS_SCHEMA,
            "operating_system": self.operating_system,
            "operating_system_release": self.operating_system_release,
            "platform_id": self.platform_id,
            "machine": self.machine,
            "byte_order": self.byte_order,
            "filesystem_encoding": self.filesystem_encoding,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "python_cache_tag": self.python_cache_tag,
            "variables": dict(self.variables),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "environment_root": self.environment_root,
        }


def capture_environment_facts(
    variable_names: Sequence[str] = (),
) -> EnvironmentFacts:
    """Capture only explicitly selected environment variables."""

    names = tuple(sorted({str(item) for item in variable_names}))
    if len(names) > 256:
        raise RequiredInputTooLargeError(
            "environment fact selection exceeds 256 variables"
        )
    # Bind presence and exact bytes without serializing credential-like ambient
    # values into decision receipts.
    variables = {
        name: _sha256(
            b"\x01" + os.fsencode(os.environ[name])
            if name in os.environ
            else b"\x00"
        )
        for name in names
    }
    implementation = platform.python_implementation()
    return EnvironmentFacts(
        operating_system=platform.system().casefold(),
        operating_system_release=platform.release(),
        platform_id=sys.platform,
        machine=platform.machine().casefold(),
        byte_order=sys.byteorder,
        filesystem_encoding=sys.getfilesystemencoding(),
        python_implementation=implementation,
        python_version=platform.python_version(),
        python_cache_tag=str(getattr(sys.implementation, "cache_tag", "")),
        variables=variables,
    )


class ProposedEffectKind(str, Enum):
    FILE = "file"
    PROCESS = "process"
    NETWORK = "network"
    CREDENTIAL = "credential"
    DATASET = "dataset"
    TASK_BOARD = "task_board"
    COMMIT = "commit"
    MERGE = "merge"


_EFFECT_OPERATIONS = {
    ProposedEffectKind.FILE: frozenset(
        {"read", "create", "write", "append", "delete", "rename", "chmod", "symlink"}
    ),
    ProposedEffectKind.PROCESS: frozenset(
        {"execute", "start", "stop", "signal"}
    ),
    ProposedEffectKind.NETWORK: frozenset(
        {"connect", "request", "listen", "download", "upload", "publish"}
    ),
    ProposedEffectKind.CREDENTIAL: frozenset(
        {"read", "use", "create", "update", "delete", "rotate"}
    ),
    ProposedEffectKind.DATASET: frozenset(
        {"read", "create", "write", "delete", "publish"}
    ),
    ProposedEffectKind.TASK_BOARD: frozenset(
        {"read", "create", "update", "delete", "complete", "reopen"}
    ),
    ProposedEffectKind.COMMIT: frozenset({"create", "amend", "sign"}),
    ProposedEffectKind.MERGE: frozenset(
        {"merge", "rebase", "cherry_pick", "fast_forward"}
    ),
}


@dataclass(frozen=True)
class ProposedEffect:
    """One closed-vocabulary, exact proposed external effect."""

    effect_id: str
    kind: ProposedEffectKind
    operation: str
    target: str
    repository_paths: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    credential_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        effect_id = str(self.effect_id or "").strip()
        target = str(self.target or "").strip()
        if not effect_id or not target:
            raise UnsupportedEffectError("effect_id and target are required")
        object.__setattr__(self, "effect_id", effect_id)
        object.__setattr__(self, "target", target)
        try:
            kind = ProposedEffectKind(self.kind)
        except ValueError as exc:
            raise UnsupportedEffectError(
                f"unsupported effect kind {self.kind!r}"
            ) from exc
        object.__setattr__(self, "kind", kind)
        operation = str(self.operation or "").strip()
        if operation not in _EFFECT_OPERATIONS[kind]:
            raise UnsupportedEffectError(
                f"unsupported {kind.value} effect operation {operation!r}"
            )
        object.__setattr__(self, "operation", operation)
        paths = tuple(sorted({_repo_path(item) for item in self.repository_paths}))
        if kind is ProposedEffectKind.FILE and not paths:
            raise UnsupportedEffectError(
                "file effects require exact repository_paths"
            )
        object.__setattr__(self, "repository_paths", paths)
        credentials = tuple(sorted({str(item).strip() for item in self.credential_ids}))
        if any(not item for item in credentials):
            raise UnsupportedEffectError("credential IDs must not be empty")
        if kind is ProposedEffectKind.CREDENTIAL and not credentials:
            raise UnsupportedEffectError(
                "credential effects require opaque credential_ids"
            )
        object.__setattr__(self, "credential_ids", credentials)
        object.__setattr__(
            self,
            "parameters",
            _frozen_json(self.parameters, name="effect parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "kind": self.kind.value,
            "operation": self.operation,
            "target": self.target,
            "repository_paths": list(self.repository_paths),
            "parameters": _thaw(self.parameters),
            "credential_ids": list(self.credential_ids),
        }


@dataclass(frozen=True)
class ProposedEffectManifest:
    effects: tuple[ProposedEffect, ...]

    def __post_init__(self) -> None:
        if len(self.effects) > DEFAULT_MAX_EFFECTS:
            raise RequiredInputTooLargeError(
                f"effect manifest exceeds {DEFAULT_MAX_EFFECTS} effects"
            )
        effects = tuple(sorted(self.effects, key=lambda item: item.effect_id))
        if len(effects) != len({item.effect_id for item in effects}):
            raise UnsupportedEffectError("effect IDs must be unique")
        object.__setattr__(self, "effects", effects)
        encoded = _canonical_json_bytes(
            {
                "schema": PROPOSED_EFFECT_MANIFEST_SCHEMA,
                "effects": [item.to_dict() for item in effects],
            }
        )
        if len(encoded) > 1_048_576:
            raise RequiredInputTooLargeError(
                "effect manifest exceeds 1048576 serialized bytes"
            )

    @property
    def manifest_root(self) -> str:
        return _identity(
            "proposed-effect-manifest",
            {
                "schema": PROPOSED_EFFECT_MANIFEST_SCHEMA,
                "effects": [item.to_dict() for item in self.effects],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROPOSED_EFFECT_MANIFEST_SCHEMA,
            "manifest_root": self.manifest_root,
            "effects": [item.to_dict() for item in self.effects],
        }


@dataclass(frozen=True)
class ProgramBehavior:
    """The complete decision identity for program bytes and proposed effects."""

    repository: RepositorySnapshot
    analysis: ProgramAnalysis
    tools: ToolCatalog
    environment: EnvironmentFacts
    effects: ProposedEffectManifest
    component_manifest_blob: BlobReference
    schema_version: int = PROGRAM_BEHAVIOR_SCHEMA_VERSION

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_BEHAVIOR_SCHEMA,
            "schema_version": self.schema_version,
            "repository_snapshot_id": self.repository.snapshot_id,
            "execution_tree_root": self.repository.execution_tree_root,
            "program_root": self.analysis.program_root,
            "ast_root": self.analysis.ast_root,
            "tool_catalog_root": self.tools.catalog_root,
            "environment_root": self.environment.environment_root,
            "effect_manifest_root": self.effects.manifest_root,
            "component_manifest_blob": self.component_manifest_blob.to_dict(),
        }

    @property
    def behavior_root(self) -> str:
        return _identity("program-behavior", self._content_dict())

    @property
    def root(self) -> str:
        return self.behavior_root

    @property
    def dirty_worktree_root(self) -> str:
        return self.repository.dirty_worktree_root

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "behavior_root": self.behavior_root,
            "repository": self.repository.to_dict(),
            "analysis": self.analysis.to_dict(),
            "tools": self.tools.to_dict(),
            "environment": self.environment.to_dict(),
            "effects": self.effects.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("utf-8")

    def verify_unchanged(self) -> None:
        self.repository.verify_unchanged()


def build_program_behavior(
    repository_root: str | os.PathLike[str],
    *,
    effects: Iterable[ProposedEffect] = (),
    scopes: Sequence[str] = (".",),
    excluded_paths: Sequence[str] = (),
    bounds: SnapshotBounds | None = None,
    artifact_store: BoundedArtifactStore | None = None,
    previous: ProgramBehavior | None = None,
    tools: Mapping[str, Sequence[str]] | ToolCatalog | None = None,
    environment_variable_names: Sequence[str] = (),
) -> ProgramBehavior:
    """Build and bind repository, AST, tool, environment, and effect roots."""

    repository = build_repository_snapshot(
        repository_root,
        scopes=scopes,
        excluded_paths=excluded_paths,
        bounds=bounds,
        artifact_store=artifact_store,
        previous=previous.repository if previous is not None else None,
    )
    analysis = build_program_analysis(
        repository,
        previous=previous.analysis if previous is not None else None,
        artifact_store=artifact_store,
    )
    catalog = tools if isinstance(tools, ToolCatalog) else capture_tool_catalog(tools)
    environment = capture_environment_facts(environment_variable_names)
    manifest = ProposedEffectManifest(tuple(effects))
    component_value = {
        "schema": PROGRAM_BEHAVIOR_SCHEMA,
        "schema_version": PROGRAM_BEHAVIOR_SCHEMA_VERSION,
        "repository_snapshot_id": repository.snapshot_id,
        "execution_tree_root": repository.execution_tree_root,
        "program_root": analysis.program_root,
        "ast_root": analysis.ast_root,
        "tool_catalog_root": catalog.catalog_root,
        "environment_root": environment.environment_root,
        "effect_manifest_root": manifest.manifest_root,
    }
    component_bytes = _canonical_json_bytes(component_value)
    component_ref = _blob_from_bytes(
        component_bytes,
        kind="program-behavior-components",
        store=artifact_store,
        cache={},
    )
    behavior = ProgramBehavior(
        repository=repository,
        analysis=analysis,
        tools=catalog,
        environment=environment,
        effects=manifest,
        component_manifest_blob=component_ref,
    )
    # AST construction and component persistence happen after the repository
    # snapshot.  Re-check all exact inputs before returning an authorization
    # identity so post-hash edits cannot inherit the old root.
    behavior.verify_unchanged()
    return behavior


# Explicit compatibility names make the contract discoverable to downstream
# proof-graph and permit work without adding it to the package-wide lazy API.
BehaviorRoot = ProgramBehavior
ProgramBehaviorRoot = ProgramBehavior
RepositoryBehaviorSnapshot = RepositorySnapshot
WorktreeSnapshot = RepositorySnapshot
EffectKind = ProposedEffectKind
EffectType = ProposedEffectKind
EffectManifest = ProposedEffectManifest
build_behavior_root = build_program_behavior
build_worktree_snapshot = build_repository_snapshot


__all__ = [
    "BehaviorRoot",
    "EffectKind",
    "EffectManifest",
    "EffectType",
    "EnvironmentFacts",
    "ProgramAnalysis",
    "ProgramBehavior",
    "ProgramBehaviorRoot",
    "ProgramBehaviorError",
    "ProgramObservation",
    "ProgramObservationKind",
    "ProposedEffect",
    "ProposedEffectKind",
    "ProposedEffectManifest",
    "RepositoryBehaviorSnapshot",
    "RepositoryEntry",
    "RepositoryEntryKind",
    "RepositoryEntryStatus",
    "RepositoryPathEscapeError",
    "RepositoryRaceError",
    "RepositorySnapshot",
    "RepositorySnapshotStats",
    "RepositoryStateError",
    "RequiredInputTooLargeError",
    "RequiredInputUnreadableError",
    "SnapshotBounds",
    "SymlinkEscapeError",
    "ToolCatalog",
    "ToolDescriptor",
    "UnsupportedEffectError",
    "WorktreeSnapshot",
    "build_behavior_root",
    "build_program_analysis",
    "build_program_behavior",
    "build_repository_snapshot",
    "build_worktree_snapshot",
    "capture_environment_facts",
    "capture_tool_catalog",
]
