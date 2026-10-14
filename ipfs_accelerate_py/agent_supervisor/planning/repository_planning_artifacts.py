"""Publish bounded planning receipts outside the source and session owners."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import uuid


MAX_ARTIFACT_BYTES = 262_144


class RepositoryPlanningArtifactError(ValueError):
    """An artifact destination cannot preserve the repository boundary."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise RepositoryPlanningArtifactError(code)


def _disjoint(directory: Path, excluded_roots: tuple[Path, ...]) -> None:
    for root in excluded_roots:
        _require(not directory.is_relative_to(root) and not root.is_relative_to(directory),
                 "planning_artifacts_overlap_source_or_state")


def _directory_fd(directory: Path) -> int:
    """Walk from the filesystem root without following any directory link."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(directory.anchor, flags)
    try:
        for component in directory.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _destination(descriptor: int, name: str) -> None:
    try:
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    _require(stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1,
             "planning_artifact_destination_not_regular_unlinked")


def write_repository_planning_artifact(*, directory: Path, name: str, text: str,
                                      excluded_roots: tuple[Path, ...]) -> int:
    """Atomically replace one regular receipt using an anchored directory FD.

    Existing symlinks, hardlinks and nonregular entries are refused. Replacing
    the destination name never opens its inode for writing, so a concurrent
    destination-link replacement cannot modify the link target. This does not
    lock the filesystem against a concurrent rename of the artifact directory.
    """
    _require(type(name) is str and name not in {"", ".", ".."}
             and not any(character in "/\\" or ord(character) < 32 or ord(character) == 127
                         for character in name)
             and len(name.encode("utf-8")) <= 255, "invalid_planning_artifact_name")
    _require(type(text) is str, "invalid_planning_artifact_text")
    _require(len(text) <= MAX_ARTIFACT_BYTES, "planning_artifact_byte_limit")
    raw = text.encode("utf-8")
    _require(len(raw) <= MAX_ARTIFACT_BYTES, "planning_artifact_byte_limit")
    _require(type(excluded_roots) is tuple and bool(excluded_roots),
             "invalid_planning_artifact_roots")
    path = Path(directory).absolute()
    _require(".." not in path.parts
             and not any(ord(character) < 32 or ord(character) == 127 for character in str(path)),
             "invalid_planning_artifact_directory")
    roots = tuple(Path(root).resolve(strict=True) for root in excluded_roots)
    _disjoint(path, roots)
    descriptor = _directory_fd(path)
    temporary = ".repository-planning-" + uuid.uuid4().hex + ".tmp"
    temporary_created = False
    try:
        _destination(descriptor, name)
        output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                         | os.O_CLOEXEC, 0o600, dir_fd=descriptor)
        temporary_created = True
        try:
            remaining = memoryview(raw)
            while remaining:
                written = os.write(output, remaining)
                _require(written > 0, "planning_artifact_write_incomplete")
                remaining = remaining[written:]
            os.fsync(output)
        finally:
            os.close(output)
        # Recheck both the visible directory and its identity before publication.
        # Every write itself is relative to the original no-follow descriptor.
        resolved = path.resolve(strict=True)
        _disjoint(resolved, roots)
        _require(resolved == path, "planning_artifact_directory_changed")
        current, opened = os.stat(path, follow_symlinks=False), os.fstat(descriptor)
        _require((current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino),
                 "planning_artifact_directory_changed")
        _destination(descriptor, name)
        os.replace(temporary, name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        temporary_created = False
        return len(raw)
    finally:
        try:
            if temporary_created:
                os.unlink(temporary, dir_fd=descriptor)
        finally:
            os.close(descriptor)


__all__ = ["RepositoryPlanningArtifactError", "write_repository_planning_artifact"]
