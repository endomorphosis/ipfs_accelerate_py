"""Create missing protected worktree parents without normalizing old paths.

This initial-owner filesystem helper changes no existing permissions. It grants
neither native allocation authority nor worker traversal. Checks are sequential,
not a continuous writer lease. Failure leaves any verified new 0700 prefix in
place; an attempted mkdir that raises may have effects outside that prefix.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import threading
import time

from .native_writable_allocation import _acl_state, _mount_identity

SCHEMA = "supervisor-owner-worktree-parent@1"
MAX_DEPTH = 64
MAX_PATH_BYTES = 4096
_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_LOCK = threading.RLock()
_ACTIVE = {}


class OwnerWorktreeParentError(ValueError):
    """Refusal retaining verified creations and any uncertain mkdir attempt."""

    def __init__(self, message):
        super().__init__(message)
        self.mutation_attempted = False
        self.created_prefix = ()
        self.attempted_path = None
        self.creation_effects_complete = True
        self.phase = "admission"


def _need(value, reason):
    if not value:
        raise OwnerWorktreeParentError(reason)


def _deadline(value):
    _need(value is None or (type(value) is int and value > 0),
        "owner parent deadline must be positive integer nanoseconds")
    _need(value is None or time.monotonic_ns() < value,
        "original owner parent deadline exhausted")


def _directory(value):
    path = Path(value)
    _need(path.is_absolute() and path.anchor == "/" and ".." not in path.parts
        and path != Path(path.anchor) and len(path.parts) - 1 <= MAX_DEPTH
        and len(os.fsencode(path)) <= MAX_PATH_BYTES,
        "bounded absolute lexical worktree parent required")
    return path


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _observe(descriptor, *, path, owner):
    before = os.fstat(descriptor)
    _need(stat.S_ISDIR(before.st_mode) and before.st_uid in (0, owner)
        and (before.st_uid != owner or before.st_gid == owner)
        and not before.st_mode & 0o022,
        "worktree parent ancestry is foreign or writable by another principal")
    mount = _mount_identity(descriptor)
    acl = _acl_state(path, descriptor=descriptor)
    named = path.lstat()
    _need(_identity(before) == _identity(os.fstat(descriptor)) == _identity(named)
        and path.resolve(strict=True) == path
        and _mount_identity(descriptor) == mount
        and _acl_state(path, descriptor=descriptor) == acl,
        "worktree parent ancestry changed during observation")
    return str(path), _identity(before), mount, acl


def _existing_prefix(directory, *, owner, deadline):
    """Read and validate every existing component before returning a missing tail."""
    directory = _directory(directory)
    _deadline(deadline)
    descriptor = os.open(directory.anchor, _FLAGS)
    current = Path(directory.anchor)
    rows = []
    try:
        rows.append(_observe(descriptor, path=current, owner=owner))
        components = directory.parts[1:]
        for index, name in enumerate(components):
            _deadline(deadline)
            try:
                child = os.open(name, _FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                _need(not rows[-1][1][2] & stat.S_ISGID,
                    "setgid inheritance cannot create an exact 0700 worktree parent")
                return descriptor, tuple(rows), components[index:]
            try:
                child_path = current / name
                row = _observe(child, path=child_path, owner=owner)
                _need(_identity(os.stat(name, dir_fd=descriptor, follow_symlinks=False)) == row[1]
                    and _observe(descriptor, path=current, owner=owner) == rows[-1],
                    "worktree parent named component changed")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor, current = child, child_path
            rows.append(row)
        return descriptor, tuple(rows), ()
    except BaseException:
        os.close(descriptor)
        raise


def _verify_prefix(rows, *, directory, owner, deadline):
    descriptor, current, missing = _existing_prefix(directory, owner=owner, deadline=deadline)
    try:
        _need(not missing and current == rows,
            "worktree parent prefix changed before its closing observation")
    finally:
        os.close(descriptor)


def _check_active(state):
    _need(not state["reentered"], "owner worktree parent preparation attempted reentry")


def prepare_owner_worktree_parent(*, directory, absolute_deadline_ns=None):
    """Create only missing owner 0700 directories under protected ancestors.

    Existing protected owner directories, including 0750, remain unchanged.
    An EEXIST race is refused rather than adopted. The error's created_prefix
    lists only new directories whose identity and policy were verified; it is
    not a complete effects claim after a failing mkdir or later path race.
    """
    state = dict(reentered=False, mutation_attempted=False, created_prefix=[],
        attempted_path=None, phase="admission")
    descriptor = None
    active = False
    key = None
    try:
        _deadline(absolute_deadline_ns)
        directory = _directory(directory)
        key = str(directory)
        owner = os.getuid()
        _need(type(owner) is int and owner > 0
            and owner == os.geteuid() == os.getgid() == os.getegid(),
            "worktree parent requires one positive owner UID and group")
        with _LOCK:
            existing = _ACTIVE.get(key)
            if existing is not None:
                existing["reentered"] = True
                raise OwnerWorktreeParentError("owner worktree parent preparation is already active")
            _ACTIVE[key] = state
            active = True
        state["phase"] = "existing_prefix"
        descriptor, rows, missing = _existing_prefix(directory,
            owner=owner, deadline=absolute_deadline_ns)
        current = Path(rows[-1][0])
        _need(rows[-1][1][3] == rows[-1][1][4] == owner,
            "managed worktree parent must belong to its current owner group")
        if missing:
            _need(rows[-1][1][2] & stat.S_IWUSR and rows[-1][1][2] & stat.S_IXUSR,
                "existing owner prefix cannot create worktree parent directories")
        else:
            _need(len(rows) > 1 and rows[-1][2] == rows[-2][2],
                "worktree parent target crosses its protected parent mount")
        # Close any preflight observer before the first mkdir. The actual
        # descriptor still names this same immutable prefix afterward.
        _verify_prefix(rows, directory=current, owner=owner, deadline=absolute_deadline_ns)
        _need(_observe(descriptor, path=current, owner=owner) == rows[-1],
            "retained worktree parent descriptor changed after preflight")
        _check_active(state)
        for name in missing:
            state["phase"] = "creation"
            _deadline(absolute_deadline_ns)
            _verify_prefix(rows, directory=current, owner=owner, deadline=absolute_deadline_ns)
            _need(_observe(descriptor, path=current, owner=owner) == rows[-1],
                "worktree parent changed before directory creation")
            _check_active(state)
            child_path = current / name
            state["attempted_path"] = str(child_path)
            state["mutation_attempted"] = True  # mkdir can take effect and then raise.
            os.mkdir(name, mode=0o700, dir_fd=descriptor)
            child = os.open(name, _FLAGS, dir_fd=descriptor)
            try:
                row = _observe(child, path=child_path, owner=owner)
                _need(row[1][3] == row[1][4] == owner and stat.S_IMODE(row[1][2]) == 0o700
                    and row[2] == rows[-1][2]
                    and _identity(os.stat(name, dir_fd=descriptor, follow_symlinks=False)) == row[1]
                    and _observe(descriptor, path=current, owner=owner) == rows[-1],
                    "new worktree parent differs from its exact owner 0700 policy")
                _check_active(state)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor, current = child, child_path
            rows = (*rows, row)
            state["created_prefix"].append(str(current))
        state["phase"] = "closing_prefix"
        _verify_prefix(rows, directory=directory, owner=owner, deadline=absolute_deadline_ns)
        _need(_observe(descriptor, path=directory, owner=owner) == rows[-1],
            "owner worktree parent changed after its closing walk")
        _check_active(state)
        _deadline(absolute_deadline_ns)
        return dict(schema=SCHEMA, directory=str(directory), owner_uid=owner, shared_group=owner,
            absolute_deadline_ns=absolute_deadline_ns, created_prefix=list(state["created_prefix"]),
            created_count=len(state["created_prefix"]), mode=oct(stat.S_IMODE(rows[-1][1][2])),
            named_parent_policy_verified=True, access_and_default_acl_absent=True,
            existing_permissions_changed=False, filesystem_writer_lease_verified=False,
            worker_traversal_qualified=False, native_allocation_verified=False,
            serialized_observation_grants_authority=False, execution_authority=False,
            completion_authority=False, proof_authority=False)
    except BaseException as original:
        error = original if isinstance(original, OwnerWorktreeParentError) else OwnerWorktreeParentError(
            "owner worktree parent preparation refused: " + type(original).__name__)
        error.mutation_attempted = state["mutation_attempted"]
        error.created_prefix = tuple(state["created_prefix"])
        error.attempted_path = state["attempted_path"]
        error.creation_effects_complete = not state["mutation_attempted"]
        error.phase = state["phase"]
        if not isinstance(original, Exception):
            for name in ("mutation_attempted", "created_prefix", "attempted_path", "creation_effects_complete", "phase"):
                setattr(original, name, getattr(error, name))
            raise
        if error is original:
            raise
        raise error from original
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if active:
            with _LOCK:
                if _ACTIVE.get(key) is state:
                    del _ACTIVE[key]


__all__ = ["SCHEMA", "OwnerWorktreeParentError", "prepare_owner_worktree_parent"]
