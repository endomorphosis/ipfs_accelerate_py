"""Candidate-only, replayable removal of exact native lifecycle inodes.

The retained original inodes are positive removal evidence. Canonical absence,
an intent alone, or a caller-supplied receipt never establishes completion.
This component does not authorize provider effects or settle task callbacks.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..task_sources import quack_owner_mutation as _native_publication
from ..task_sources.quack_owner_mutation import _publish_without_replace_at
from .worktree_lifecycle import WorkspaceLifecycleRecord, WorktreeLifecycleError

PREPARED_SCHEMA = "worktree-lifecycle-candidate-delete-prepared@1"
COMMITTED_SCHEMA = "worktree-lifecycle-candidate-delete-committed@1"
JOURNAL_DIR = ".candidate-observed-deletions"
JOURNAL_SCOPE_NAME = "native-scope.json"
MAX_BYTES = 256 * 1024
_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTITY_FIELDS = (
    "dev",
    "ino",
    "mode",
    "uid",
    "size",
    "mtime_ns",
    "ctime_ns",
    "nlink",
)


class CandidateDeletionUnverified(WorktreeLifecycleError):
    """Retain custody; exact native deletion evidence is unavailable."""


@dataclass(frozen=True)
class CandidateObservedDeletion:
    """Frozen native observation, not a reusable effect or callback grant."""

    evidence_json: str

    def to_dict(self):
        return json.loads(self.evidence_json)


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _seal(body):
    return {**body, "receipt_id": _digest(_bytes(body))}


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise CandidateDeletionUnverified("duplicate journal field")
        value[key] = item
    return value


def _identity(metadata):
    return dict(
        zip(
            _IDENTITY_FIELDS,
            (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_uid,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
                metadata.st_nlink,
            ),
            strict=True,
        )
    )


def _directory_identity(metadata):
    return {k: _identity(metadata)[k] for k in ("dev", "ino", "mode", "uid")}


def _open_directory(path, *, private=False):
    """Open an existing absolute path without following any symlink component."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or len(path.parts) > 128:
        raise CandidateDeletionUnverified("noncanonical lifecycle directory")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.geteuid() or (
            private and stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise CandidateDeletionUnverified(
                "lifecycle directory is not private-owned"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class _File:
    def __init__(self, directory, name, *, payload=None, limit=MAX_BYTES):
        self.fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory,
        )
        try:
            metadata = os.fstat(self.fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o022
                or metadata.st_nlink != 1
                or not 0 < metadata.st_size <= limit
            ):
                raise CandidateDeletionUnverified(
                    "journal entry is not owned bounded regular data"
                )
            self.identity = _identity(metadata)
            raw = bytearray()
            while len(raw) <= limit:
                chunk = os.read(self.fd, min(65536, limit + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
            if len(raw) != metadata.st_size:
                raise CandidateDeletionUnverified(
                    "journal read changed or exceeded budget"
                )
            self.raw = bytes(raw)
            self.value = json.loads(self.raw, object_pairs_hook=_unique)
            if not isinstance(self.value, dict) or (
                payload is not None and _bytes(self.value) != _bytes(payload)
            ):
                raise CandidateDeletionUnverified(
                    "journal payload does not match exact native record"
                )
            self.digest = _digest(self.raw)
            self.check(directory, name)
        except BaseException:
            os.close(self.fd)
            raise

    def check(self, directory, name):
        if (
            _identity(os.fstat(self.fd)) != self.identity
            or _identity(os.stat(name, dir_fd=directory, follow_symlinks=False))
            != self.identity
        ):
            raise CandidateDeletionUnverified("held journal inode changed")

    def close(self):
        os.close(self.fd)


def _exists(directory, name):
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _publish(directory, name, value):
    """Publish a complete immutable file; never replace a collision."""
    raw = _bytes(value) + b"\n"
    if len(raw) > MAX_BYTES:
        raise CandidateDeletionUnverified("journal publication exceeds budget")
    temporary = f".candidate-delete-{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory,
    )
    created = os.fstat(descriptor)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise CandidateDeletionUnverified(
                    "journal publication made no progress"
                )
            offset += written
        os.fsync(descriptor)
        completed = os.fstat(descriptor)
        current = os.stat(temporary, dir_fd=directory, follow_symlinks=False)
        if (
            _identity(current) != _identity(completed)
            or (completed.st_dev, completed.st_ino) != (created.st_dev, created.st_ino)
            or completed.st_size != len(raw)
            or not stat.S_ISREG(completed.st_mode)
            or completed.st_nlink != 1
        ):
            raise CandidateDeletionUnverified(
                "journal publication temporary was replaced"
            )
        _publish_without_replace_at(directory, temporary, name)
        os.fsync(directory)
        published = _identity(os.fstat(descriptor))
        if (
            _identity(os.stat(name, dir_fd=directory, follow_symlinks=False))
            != published
        ):
            raise CandidateDeletionUnverified("published journal inode changed")
        return published
    finally:
        os.close(descriptor)
        try:
            current = os.stat(temporary, dir_fd=directory, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


@contextmanager
def _guards(directory, names, *, observe):
    """Use native index->workspace guards without ever creating a guard file."""
    with ExitStack() as stack:
        for name in names:
            guard = f".{name}.update.lock"
            descriptor = os.open(
                guard,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            stack.callback(os.close, descriptor)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o022
                or metadata.st_nlink != 1
            ):
                raise CandidateDeletionUnverified("native update guard unavailable")
            fcntl.flock(
                descriptor,
                (fcntl.LOCK_SH if observe else fcntl.LOCK_EX) | fcntl.LOCK_NB,
            )
            stack.callback(fcntl.flock, descriptor, fcntl.LOCK_UN)
            if _identity(
                os.stat(guard, dir_fd=directory, follow_symlinks=False)
            ) != _identity(metadata):
                raise CandidateDeletionUnverified("native update guard replaced")
        yield


def _check_store(directory, path, expected, *, private=False):
    with ExitStack() as stack:
        fresh = _open_directory(path, private=private)
        stack.callback(os.close, fresh)
        if (
            _directory_identity(os.fstat(directory)) != expected
            or _directory_identity(os.fstat(fresh)) != expected
        ):
            raise CandidateDeletionUnverified("native lifecycle store binding changed")


def _validate_prepared(value, binding):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "binding", "sources", "receipt_id"}
        or value["schema"] != PREPARED_SCHEMA
        or _bytes(value["binding"]) != _bytes(binding)
        or value != _seal({k: v for k, v in value.items() if k != "receipt_id"})
        or not isinstance(value["sources"], dict)
        or set(value["sources"]) != {"record", "index"}
    ):
        raise CandidateDeletionUnverified("prepared deletion binding is invalid")
    for source in value["sources"].values():
        if (
            not isinstance(source, dict)
            or set(source) != {"identity", "bytes_id"}
            or not isinstance(source["identity"], dict)
            or set(source["identity"]) != set(_IDENTITY_FIELDS)
            or any(type(v) is not int for v in source["identity"].values())
            or not isinstance(source["bytes_id"], str)
            or not _ID.fullmatch(source["bytes_id"])
        ):
            raise CandidateDeletionUnverified("prepared source inode is invalid")


def _scope(repo, repo_identity, path, store_identity, journal_identity):
    return _seal(
        {
            "schema": "worktree-lifecycle-candidate-journal@1",
            "repo_root": str(repo),
            "repo_identity": repo_identity,
            "store_path": str(path),
            "store_identity": store_identity,
            "journal_name": JOURNAL_DIR,
            "journal_identity": journal_identity,
            "completion_authority": False,
        }
    )


def _provision_journal(directory, repo, repo_identity, path, store_identity):
    """Atomically publish a complete private child, never adopt a foreign child.

    An interrupted unpublished temporary is retained, never discovered/adopted
    as authority. It contains no original lifecycle inodes.
    """
    temporary = ".candidate-journal-" + uuid.uuid4().hex + ".tmp"
    os.mkdir(temporary, mode=0o700, dir_fd=directory)
    descriptor = os.open(
        temporary,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory,
    )
    try:
        identity = _directory_identity(os.fstat(descriptor))
        if identity["uid"] != os.geteuid() or stat.S_IMODE(identity["mode"]) != 0o700:
            raise CandidateDeletionUnverified("new private journal admission failed")
        _publish(
            descriptor,
            JOURNAL_SCOPE_NAME,
            _scope(repo, repo_identity, path, store_identity, identity),
        )
        if (
            _directory_identity(
                os.stat(temporary, dir_fd=directory, follow_symlinks=False)
            )
            != identity
        ):
            raise CandidateDeletionUnverified("private journal temporary replaced")
        _check_store(directory, path, store_identity)
        _publish_without_replace_at(directory, temporary, JOURNAL_DIR)
        if (
            _directory_identity(
                os.stat(JOURNAL_DIR, dir_fd=directory, follow_symlinks=False)
            )
            != identity
        ):
            raise CandidateDeletionUnverified("private journal publication replaced")
        os.fsync(directory)
    finally:
        os.close(descriptor)


def _move_without_replace(source_directory, source, target_directory, target):
    """Use the existing native no-replace syscall with two pinned directories."""
    rename = _native_publication._RENAMEAT2
    if rename is None:
        raise CandidateDeletionUnverified("atomic native move is unavailable")
    ctypes.set_errno(0)
    if (
        rename(
            source_directory,
            os.fsencode(source),
            target_directory,
            os.fsencode(target),
            _native_publication._RENAME_NOREPLACE,
        )
        != 0
    ):
        number = ctypes.get_errno() or errno.EIO
        raise OSError(number, os.strerror(number), target)


def _operate(store, expected, handoff_receipt_id, *, mode):
    if (
        type(expected) is not WorkspaceLifecycleRecord
        or not expected.is_terminal
        or expected.terminal_reason != "worktree_cleaned"
        or expected.record_id != expected.compute_record_id()
        or expected.fence < 2
        or expected.attempt < 1
        or not expected.lease_id
        or not expected.canonical_task_cid
        or expected.owner.pid <= 0
        or expected.owner.start_time_ticks <= 0
        or not isinstance(handoff_receipt_id, str)
        or not _ID.fullmatch(handoff_receipt_id)
    ):
        raise CandidateDeletionUnverified("candidate terminal binding is invalid")
    repo = Path(store.repo_root)
    if expected.repo_root != str(repo):
        raise CandidateDeletionUnverified("candidate repository binding differs")
    path = Path(store.store_dir)
    with ExitStack() as stack:
        repo_descriptor = _open_directory(repo)
        stack.callback(os.close, repo_descriptor)
        repo_identity = _directory_identity(os.fstat(repo_descriptor))
        directory = _open_directory(path)
        stack.callback(os.close, directory)
        store_identity = _directory_identity(os.fstat(directory))
        record_name = store.workspace_path_for(expected.workspace_path).name
        index_name = store.task_index_path_for(
            canonical_task_cid=expected.canonical_task_cid,
            task_id=expected.task_id,
            attempt=expected.attempt,
        ).name
        roles = (
            ("record", record_name, expected.to_dict()),
            ("index", index_name, store._task_index_payload(expected)),
        )
        journal_path = path / JOURNAL_DIR
        with _guards(directory, (index_name, record_name), observe=mode == "observe"):
            _check_store(directory, path, store_identity)
            _check_store(repo_descriptor, repo, repo_identity)
            if not _exists(directory, JOURNAL_DIR):
                if mode == "observe":
                    return None
                if mode == "resume":
                    raise CandidateDeletionUnverified(
                        "no prepared native deletion to resume"
                    )
                # Provision only after both actual originals qualify. Never
                # create a journal to migrate an absent/legacy deletion.
                for _, name, payload in roles:
                    source = _File(directory, name, payload=payload, limit=65536)
                    stack.callback(source.close)
                _provision_journal(directory, repo, repo_identity, path, store_identity)
            private = _open_directory(journal_path, private=True)
            stack.callback(os.close, private)
            journal_identity = _directory_identity(os.fstat(private))
            if journal_identity["dev"] != store_identity["dev"]:
                raise CandidateDeletionUnverified(
                    "journal is not on the native store filesystem"
                )

            scope_file = _File(
                private,
                JOURNAL_SCOPE_NAME,
                payload=_scope(
                    repo, repo_identity, path, store_identity, journal_identity
                ),
            )
            stack.callback(scope_file.close)

            def check_binding():
                _check_store(directory, path, store_identity)
                _check_store(repo_descriptor, repo, repo_identity)
                _check_store(private, journal_path, journal_identity, private=True)
                scope_file.check(private, JOURNAL_SCOPE_NAME)

            check_binding()
            binding = {
                "repo_root": str(repo),
                "repo_identity": repo_identity,
                "store_path": str(path),
                "store_identity": store_identity,
                "journal_identity": journal_identity,
                "journal_name": JOURNAL_DIR,
                "terminal": expected.to_dict(),
                "task_index": store._task_index_payload(expected),
                "handoff_receipt_id": handoff_receipt_id,
                "record_name": record_name,
                "index_name": index_name,
            }
            operation = _digest(_bytes(binding))
            prefix = "candidate-delete-" + operation.removeprefix("sha256:")
            prepared_name, committed_name = (
                prefix + ".prepared.json",
                prefix + ".committed.json",
            )
            prepared_file = None
            try:
                prepared_file = _File(private, prepared_name)
                stack.callback(prepared_file.close)
            except FileNotFoundError:
                if mode == "observe":
                    return None
                if mode == "resume":
                    raise CandidateDeletionUnverified(
                        "no prepared native deletion to resume"
                    ) from None
            if prepared_file is None:
                sources = {}
                for role, name, payload in roles:
                    if _exists(private, prefix + "." + role):
                        raise CandidateDeletionUnverified(
                            "foreign retained inode before preparation"
                        )
                    source = _File(directory, name, payload=payload, limit=65536)
                    stack.callback(source.close)
                    sources[role] = {
                        "identity": source.identity,
                        "bytes_id": source.digest,
                    }
                prepared = _seal(
                    {"schema": PREPARED_SCHEMA, "binding": binding, "sources": sources}
                )
                check_binding()
                _publish(private, prepared_name, prepared)
                prepared_file = _File(private, prepared_name, payload=prepared)
                stack.callback(prepared_file.close)
            prepared = prepared_file.value
            _validate_prepared(prepared, binding)
            committed_file = None
            try:
                committed_file = _File(private, committed_name)
                stack.callback(committed_file.close)
            except FileNotFoundError:
                if mode == "observe":
                    return None
            if mode != "observe" and committed_file is None:
                # A prior call may have failed to fsync publication. Reassert
                # intent and both directory dependencies BEFORE moving anything.
                os.fsync(prepared_file.fd)
                os.fsync(private)
                os.fsync(directory)
            moved = {}
            held = {}
            for role, name, payload in roles:
                retained = prefix + "." + role
                original = prepared["sources"][role]
                if _exists(private, retained):
                    if _exists(directory, name):
                        raise CandidateDeletionUnverified(
                            "canonical lifecycle replacement exists"
                        )
                    source = _File(private, retained, payload=payload, limit=65536)
                    stack.callback(source.close)
                    # rename changes ctime. Require exact inode/content and all
                    # stable metadata first; commit captures post-rename ctime.
                    stable = {
                        k: v for k, v in source.identity.items() if k != "ctime_ns"
                    }
                    if (
                        stable
                        != {
                            k: v
                            for k, v in original["identity"].items()
                            if k != "ctime_ns"
                        }
                        or source.digest != original["bytes_id"]
                    ):
                        raise CandidateDeletionUnverified(
                            "retained native inode changed"
                        )
                else:
                    if mode == "observe" or committed_file is not None:
                        raise CandidateDeletionUnverified(
                            "committed retained inode is absent"
                        )
                    source = _File(directory, name, payload=payload, limit=65536)
                    stack.callback(source.close)
                    if (
                        source.identity != original["identity"]
                        or source.digest != original["bytes_id"]
                    ):
                        raise CandidateDeletionUnverified(
                            "remaining original inode changed"
                        )
                    source.check(directory, name)
                    prepared_file.check(private, prepared_name)
                    check_binding()
                    _move_without_replace(directory, name, private, retained)
                    after = _identity(os.fstat(source.fd))
                    if (
                        {k: v for k, v in after.items() if k != "ctime_ns"}
                        != {
                            k: v
                            for k, v in original["identity"].items()
                            if k != "ctime_ns"
                        }
                        or _identity(
                            os.stat(retained, dir_fd=private, follow_symlinks=False)
                        )
                        != after
                        or _exists(directory, name)
                    ):
                        raise CandidateDeletionUnverified(
                            "native move did not retain exact inode"
                        )
                    source.identity = after
                    os.fsync(private)
                    os.fsync(directory)
                held[role] = source
                moved[role] = {"identity": source.identity, "bytes_id": source.digest}
            receipt = _seal(
                {
                    "schema": COMMITTED_SCHEMA,
                    "operation_id": operation,
                    "prepared_receipt_id": prepared["receipt_id"],
                    "terminal_record_id": expected.record_id,
                    "handoff_receipt_id": handoff_receipt_id,
                    "retained": moved,
                    "canonical_record_removed": True,
                    "canonical_index_removed": True,
                    "completion_authority": False,
                }
            )
            prepared_file.check(private, prepared_name)
            check_binding()
            for role, name, _ in roles:
                held[role].check(private, prefix + "." + role)
                if _exists(directory, name):
                    raise CandidateDeletionUnverified(
                        "canonical lifecycle replacement exists"
                    )
            if committed_file is not None:
                if _bytes(committed_file.value) != _bytes(receipt):
                    raise CandidateDeletionUnverified(
                        "committed deletion evidence changed"
                    )
                committed_file.check(private, committed_name)
            else:
                # A crash may have interrupted fsync after a successful move.
                os.fsync(private)
                os.fsync(directory)
                published_identity = _publish(private, committed_name, receipt)
                committed_file = _File(private, committed_name, payload=receipt)
                stack.callback(committed_file.close)
                if committed_file.identity != published_identity:
                    raise CandidateDeletionUnverified(
                        "new committed receipt inode changed"
                    )
            if mode != "observe":
                # A visible receipt may precede a failed publication fsync.
                # Reassert all durability dependencies even on committed replay.
                os.fsync(committed_file.fd)
                os.fsync(private)
                os.fsync(directory)
            committed_file.check(private, committed_name)
            prepared_file.check(private, prepared_name)
            check_binding()
            for role, name, _ in roles:
                held[role].check(private, prefix + "." + role)
                if _exists(directory, name):
                    raise CandidateDeletionUnverified(
                        "canonical lifecycle replacement exists"
                    )
            return CandidateObservedDeletion(
                _bytes({"prepared": prepared, "committed": receipt}).decode()
            )


def operate(store, expected, handoff_receipt_id, *, mode):
    if mode not in {"delete", "resume", "observe"}:
        raise ValueError("invalid candidate deletion mode")
    try:
        return _operate(store, expected, handoff_receipt_id, mode=mode)
    except CandidateDeletionUnverified:
        raise
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise CandidateDeletionUnverified(
            "native candidate deletion unavailable"
        ) from exc
