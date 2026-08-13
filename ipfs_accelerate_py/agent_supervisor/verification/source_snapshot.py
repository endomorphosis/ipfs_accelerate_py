"""Canonical fixed-point source identity for IVP release evidence.

The release benchmark and report are tracked files, so neither can truthfully
contain the commit which first introduces its own bytes.  This module binds
those evidence documents to the effective source checkout instead.  The
identity deliberately omits Git history and exactly the two self-referential
evidence outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

SOURCE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ivp-source-snapshot@1"
)
SOURCE_SNAPSHOT_DOMAIN: Final[str] = "ivp-source-snapshot@1"

# Closed by contract.  Callers cannot widen this set.
SOURCE_SNAPSHOT_EXCLUDED_PATHS: Final[frozenset[str]] = frozenset(
    {
        "artifacts/agent_supervisor/incremental_verification/benchmark.json",
        "docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md",
    }
)
TODO_BOARD_PATH: Final[str] = (
    "docs/architecture/incremental_verification_planner.todo.md"
)

_REVIEWED_GITLINKS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ipfs_kit_py": "5a7a2df8181cfdc33bc19be09989df7ff83f2d4e",
        "ipfs_datasets_py": "6cd037c7738f44904add46391537588e67f6f238",
    }
)
_TASKBOARD_FRAME_DOMAIN: Final[bytes] = b"ivp-taskboard-status-normalization@1\0"


class SourceSnapshotError(RuntimeError):
    """Raised when a trustworthy source snapshot cannot be constructed."""


@dataclass(frozen=True, slots=True)
class SourceSnapshotEntry:
    """One effective present path in the canonical source manifest."""

    path: str
    mode: str
    sha256: str | None = None
    object_id: str | None = None

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            not self.path
            or self.path.startswith("/")
            or path.as_posix() != self.path
            or self.path == "."
            or ".." in path.parts
            or "\0" in self.path
        ):
            raise SourceSnapshotError("snapshot entry path is unsafe")
        if self.mode == "160000":
            if self.sha256 is not None or not _is_hex(self.object_id, lengths=(40, 64)):
                raise SourceSnapshotError("gitlink entry requires one valid object_id")
            return
        if self.mode not in {"100644", "100755", "120000"}:
            raise SourceSnapshotError("snapshot entry mode is not canonical")
        if self.object_id is not None or not _is_hex(self.sha256, lengths=(64,)):
            raise SourceSnapshotError("source entry requires one SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        value = {"path": self.path, "mode": self.mode}
        if self.sha256 is not None:
            value["sha256"] = self.sha256
        if self.object_id is not None:
            value["object_id"] = self.object_id
        return value


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Canonical source manifest plus non-authoritative observation metadata."""

    entries: tuple[SourceSnapshotEntry, ...]
    source_snapshot_id: str
    observed_head: str | None
    schema: str = SOURCE_SNAPSHOT_SCHEMA
    domain: str = SOURCE_SNAPSHOT_DOMAIN

    def __post_init__(self) -> None:
        if self.schema != SOURCE_SNAPSHOT_SCHEMA or self.domain != SOURCE_SNAPSHOT_DOMAIN:
            raise SourceSnapshotError("source snapshot schema or domain is invalid")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, SourceSnapshotEntry) for entry in self.entries
        ):
            raise SourceSnapshotError("source snapshot entries must be typed and immutable")
        paths = [entry.path for entry in self.entries]
        if paths != sorted(paths, key=lambda value: value.encode("utf-8")):
            raise SourceSnapshotError("source snapshot entries are not canonically sorted")
        if len(paths) != len(set(paths)):
            raise SourceSnapshotError("source snapshot contains duplicate paths")
        if SOURCE_SNAPSHOT_EXCLUDED_PATHS.intersection(paths):
            raise SourceSnapshotError("source snapshot contains an excluded path")
        expected = _snapshot_id(self.identity_manifest())
        if self.source_snapshot_id != expected:
            raise SourceSnapshotError("source_snapshot_id does not match its manifest")
        if self.observed_head is not None and not _is_hex(
            self.observed_head, lengths=(40, 64)
        ):
            raise SourceSnapshotError("observed_head must be a Git object ID or null")

    def identity_manifest(self) -> dict[str, object]:
        """Return the complete, history-free value covered by the identity."""

        return {
            "schema": self.schema,
            "domain": self.domain,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the evidence envelope; ``observed_head`` is diagnostic only."""

        return {
            **self.identity_manifest(),
            "source_snapshot_id": self.source_snapshot_id,
            "observed_head": self.observed_head,
        }


def _git(root: Path, *args: str, check: bool = True) -> bytes:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(root), *args],
            check=False,
            capture_output=True,
            env=environment,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceSnapshotError(f"git invocation failed: {exc}") from exc
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise SourceSnapshotError(
            f"git {' '.join(args)} failed with {completed.returncode}: {detail}"
        )
    return completed.stdout


def _is_hex(value: object, *, lengths: tuple[int, ...]) -> bool:
    if not isinstance(value, str) or len(value) not in lengths:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _decode_path(raw: bytes) -> str:
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceSnapshotError("repository path is not valid UTF-8") from exc
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or path.as_posix() != value
        or value == "."
        or ".." in path.parts
        or "\0" in value
    ):
        raise SourceSnapshotError(f"unsafe repository path: {value!r}")
    return value


def _index_entries(root: Path) -> dict[str, tuple[str, str]]:
    """Return stage-zero index paths as ``path -> (mode, object id)``."""

    fields = _git(
        root,
        "-c",
        "core.fsmonitor=false",
        "ls-files",
        "--stage",
        "-z",
    ).split(b"\0")
    entries: dict[str, tuple[str, str]] = {}
    for field in fields:
        if not field:
            continue
        try:
            metadata, raw_path = field.split(b"\t", 1)
            mode_raw, oid_raw, stage_raw = metadata.split(b" ", 2)
        except ValueError as exc:
            raise SourceSnapshotError("malformed git index entry") from exc
        if stage_raw != b"0":
            raise SourceSnapshotError("unmerged index cannot form a source snapshot")
        path = _decode_path(raw_path)
        entries[path] = (
            mode_raw.decode("ascii"),
            oid_raw.decode("ascii"),
        )
    return entries


def _head_entries(root: Path) -> dict[str, tuple[str, str]]:
    """Return the committed HEAD tree without consulting worktree filters."""

    fields = _git(root, "ls-tree", "-r", "--full-tree", "-z", "HEAD").split(b"\0")
    entries: dict[str, tuple[str, str]] = {}
    for field in fields:
        if not field:
            continue
        try:
            metadata, raw_path = field.split(b"\t", 1)
            mode_raw, _kind_raw, oid_raw = metadata.split(b" ", 2)
        except ValueError as exc:
            raise SourceSnapshotError("malformed Git HEAD tree entry") from exc
        entries[_decode_path(raw_path)] = (
            mode_raw.decode("ascii"),
            oid_raw.decode("ascii"),
        )
    return entries


def _untracked_paths(root: Path) -> set[str]:
    raw = _git(
        root,
        "-c",
        "core.fsmonitor=false",
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    return {_decode_path(item) for item in raw.split(b"\0") if item}


def _normalize_todo_statuses(data: bytes) -> bytes:
    """Injectively frame raw rows while normalizing exact lifecycle values."""

    lines = data.splitlines(keepends=True)
    in_task = False
    normalized: list[bytes] = [_TASKBOARD_FRAME_DOMAIN]

    def frame(tag: bytes, payload: bytes) -> bytes:
        return tag + len(payload).to_bytes(8, "big") + payload

    for line in lines:
        body = line.rstrip(b"\r\n")
        ending = line[len(body) :]
        if body.startswith(b"## "):
            in_task = False
        if body.startswith(b"## IVP-"):
            suffix = body[len(b"## IVP-") :]
            task_id, separator, _title = suffix.partition(b" ")
            in_task = bool(
                separator
                and len(task_id) == 3
                and task_id.isdigit()
            )
        if in_task and body in {b"- Status: todo", b"- Status: completed"}:
            normalized.append(frame(b"L", ending))
        else:
            normalized.append(frame(b"R", line))
    return b"".join(normalized)


def _regular_mode(mode: int) -> str:
    # Git canonicalizes regular-file executability from the owner execute bit.
    return "100755" if mode & stat.S_IXUSR else "100644"


def _stable_regular_bytes(path: Path) -> tuple[bytes, int]:
    """Read one regular file without following links or accepting mutation."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceSnapshotError(f"cannot open source path {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceSnapshotError(f"source path is not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    observed_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if observed_before != observed_after:
        raise SourceSnapshotError(f"source path changed while read: {path}")
    return b"".join(chunks), after.st_mode


def _stable_symlink_target(path: Path) -> bytes:
    try:
        before = path.lstat()
        target = os.readlink(path)
        after = path.lstat()
    except OSError as exc:
        raise SourceSnapshotError(f"cannot read symbolic link {path}: {exc}") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stat.S_ISLNK(before.st_mode) or before_identity != after_identity:
        raise SourceSnapshotError(f"symbolic link changed while read: {path}")
    return os.fsencode(target)


def _git_blob_object_id(data: bytes, *, algorithm: str) -> str:
    if algorithm not in {"sha1", "sha256"}:
        raise SourceSnapshotError(f"unsupported Git object format: {algorithm!r}")
    digest = hashlib.new(algorithm)
    digest.update(b"blob " + str(len(data)).encode("ascii") + b"\0")
    digest.update(data)
    return digest.hexdigest()


def _validate_physical_index(
    repository: Path,
    *,
    visited: set[Path] | None = None,
) -> None:
    """Compare physical tracked bytes to index objects without Git filters."""

    seen = set() if visited is None else visited
    resolved = repository.resolve()
    if resolved in seen:
        raise SourceSnapshotError("nested gitlink cycle is not admissible")
    seen.add(resolved)
    try:
        index = _index_entries(repository)
        if index != _head_entries(repository):
            raise SourceSnapshotError("reviewed gitlink index differs from HEAD")
        if _untracked_paths(repository):
            raise SourceSnapshotError("reviewed gitlink contains untracked paths")
        index_flags = _git(
            repository,
            "-c",
            "core.fsmonitor=false",
            "ls-files",
            "-v",
            "-z",
        )
        for flagged in index_flags.split(b"\0"):
            if flagged and not flagged.startswith(b"H "):
                raise SourceSnapshotError(
                    "reviewed gitlink uses a non-normal index flag"
                )
        object_format = _git(
            repository, "rev-parse", "--show-object-format"
        ).decode("ascii").strip()
        for path, (index_mode, object_id) in index.items():
            absolute = repository / path
            if index_mode == "160000":
                try:
                    nested_mode = absolute.lstat().st_mode
                except FileNotFoundError:
                    continue
                if not stat.S_ISDIR(nested_mode):
                    raise SourceSnapshotError(
                        f"nested gitlink path has the wrong type: {path}"
                    )
                if not (absolute / ".git").exists():
                    if next(absolute.iterdir(), None) is not None:
                        raise SourceSnapshotError(
                            f"nested gitlink is populated but not initialized: {path}"
                        )
                    continue
                head = _git(absolute, "rev-parse", "--verify", "HEAD").decode(
                    "ascii"
                ).strip()
                if head != object_id:
                    raise SourceSnapshotError(
                        f"nested gitlink HEAD does not match its index: {path}"
                    )
                _validate_physical_index(absolute, visited=seen)
                continue
            try:
                observed_mode = absolute.lstat().st_mode
            except FileNotFoundError as exc:
                raise SourceSnapshotError(
                    f"reviewed gitlink tracked path is missing: {path}"
                ) from exc
            if index_mode == "120000":
                if not stat.S_ISLNK(observed_mode):
                    raise SourceSnapshotError(
                        f"reviewed gitlink symlink has the wrong type: {path}"
                    )
                data = _stable_symlink_target(absolute)
                observed_git_mode = "120000"
            elif index_mode in {"100644", "100755"}:
                if not stat.S_ISREG(observed_mode):
                    raise SourceSnapshotError(
                        f"reviewed gitlink file has the wrong type: {path}"
                    )
                data, stable_mode = _stable_regular_bytes(absolute)
                observed_git_mode = _regular_mode(stable_mode)
            else:
                raise SourceSnapshotError(
                    f"reviewed gitlink index mode is unsupported: {index_mode}"
                )
            if observed_git_mode != index_mode:
                raise SourceSnapshotError(
                    f"reviewed gitlink mode differs from its index: {path}"
                )
            if _git_blob_object_id(data, algorithm=object_format) != object_id:
                raise SourceSnapshotError(
                    f"reviewed gitlink physical bytes differ from its index: {path}"
                )
    finally:
        seen.remove(resolved)


def _entry_for_path(
    root: Path,
    path: str,
    *,
    indexed: tuple[str, str] | None,
) -> SourceSnapshotEntry | None:
    absolute = root / path
    try:
        mode_bits = absolute.lstat().st_mode
    except FileNotFoundError:
        # A deleted tracked path is absent from the effective manifest.
        return None

    if indexed is not None and indexed[0] == "160000":
        if not stat.S_ISDIR(mode_bits):
            raise SourceSnapshotError(f"gitlink worktree is missing: {path}")
        return SourceSnapshotEntry(path=path, mode="160000", object_id=indexed[1])
    if stat.S_ISLNK(mode_bits):
        content = _stable_symlink_target(absolute)
        return SourceSnapshotEntry(
            path=path,
            mode="120000",
            sha256=hashlib.sha256(content).hexdigest(),
        )
    if not stat.S_ISREG(mode_bits):
        raise SourceSnapshotError(f"unsupported source path type: {path}")
    content, observed_mode = _stable_regular_bytes(absolute)
    if path == TODO_BOARD_PATH:
        content = _normalize_todo_statuses(content)
    return SourceSnapshotEntry(
        path=path,
        mode=_regular_mode(observed_mode),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _validate_gitlinks(
    root: Path,
    index: dict[str, tuple[str, str]],
    *,
    validate_physical: bool,
) -> None:
    for path, reviewed_oid in _REVIEWED_GITLINKS.items():
        if index.get(path) != ("160000", reviewed_oid):
            raise SourceSnapshotError(
                f"{path} gitlink must equal reviewed object {reviewed_oid}"
            )

    for path, (mode, object_id) in index.items():
        if mode != "160000":
            continue
        if path not in _REVIEWED_GITLINKS:
            # Generic gitlinks bind only their exact indexed object ID. Their
            # nested worktrees are deliberately outside this manifest.
            continue
        nested = root / path
        try:
            nested_mode = nested.lstat().st_mode
        except FileNotFoundError:
            raise SourceSnapshotError(f"{path} gitlink is not initialized")
        if not stat.S_ISDIR(nested_mode):
            raise SourceSnapshotError(f"{path} gitlink worktree is not a directory")
        if not (nested / ".git").exists():
            try:
                empty = next(nested.iterdir(), None) is None
            except OSError as exc:
                raise SourceSnapshotError(f"cannot inspect gitlink {path}: {exc}") from exc
            if not empty:
                raise SourceSnapshotError(f"{path} gitlink is not initialized")
            raise SourceSnapshotError(f"{path} gitlink is not initialized")
        head = _git(nested, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
        if head != object_id:
            raise SourceSnapshotError(
                f"{path} HEAD {head!r} does not equal gitlink {object_id!r}"
            )
        if validate_physical:
            _validate_physical_index(nested)
            _validate_physical_index(nested)


def _observed_head(root: Path) -> str | None:
    raw = _git(root, "rev-parse", "--verify", "HEAD", check=False)
    value = raw.decode("ascii", "replace").strip()
    return value or None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _snapshot_id(manifest: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()}"


def build_source_snapshot(repo_root: Path | str) -> SourceSnapshot:
    """Build the canonical fixed-point identity of an effective checkout.

    The path set is the union of present index paths and nonignored untracked
    paths.  Tracking provenance and Git history are intentionally absent.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise SourceSnapshotError(f"repository root does not exist: {root}")
    top = _git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(top).resolve() != root:
        raise SourceSnapshotError("repo_root must be the Git worktree root")

    index = _index_entries(root)
    _validate_gitlinks(root, index, validate_physical=False)
    untracked = _untracked_paths(root)
    paths = (set(index) | untracked) - SOURCE_SNAPSHOT_EXCLUDED_PATHS
    entries = tuple(
        entry
        for path in sorted(paths, key=lambda value: value.encode("utf-8"))
        if (entry := _entry_for_path(root, path, indexed=index.get(path))) is not None
    )
    manifest: dict[str, object] = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "domain": SOURCE_SNAPSHOT_DOMAIN,
        "entries": [entry.to_dict() for entry in entries],
    }
    if index != _index_entries(root) or untracked != _untracked_paths(root):
        raise SourceSnapshotError("repository path set changed during source snapshot")
    confirmed_entries = tuple(
        entry
        for path in sorted(paths, key=lambda value: value.encode("utf-8"))
        if (entry := _entry_for_path(root, path, indexed=index.get(path))) is not None
    )
    if confirmed_entries != entries:
        raise SourceSnapshotError("repository content changed during source snapshot")
    if index != _index_entries(root) or untracked != _untracked_paths(root):
        raise SourceSnapshotError("repository path set changed during source snapshot")
    _validate_gitlinks(root, index, validate_physical=True)
    return SourceSnapshot(
        entries=entries,
        source_snapshot_id=_snapshot_id(manifest),
        observed_head=_observed_head(root),
    )


def source_snapshot_id(repo_root: Path | str) -> str:
    """Return the canonical IVP source identity for ``repo_root``."""

    return build_source_snapshot(repo_root).source_snapshot_id


__all__ = [
    "SOURCE_SNAPSHOT_DOMAIN",
    "SOURCE_SNAPSHOT_EXCLUDED_PATHS",
    "SOURCE_SNAPSHOT_SCHEMA",
    "SourceSnapshot",
    "SourceSnapshotEntry",
    "SourceSnapshotError",
    "build_source_snapshot",
    "source_snapshot_id",
]
