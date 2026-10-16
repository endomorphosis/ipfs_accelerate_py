"""Provision one initial owner-only worktree through verified descriptors.

The caller must still be in its initial no-live-worker phase. These sequential
checks are not a filesystem writer lease, and mode changes cannot revoke old
descriptors. This helper grants no native task, launch, or worker UID authority.

Full preflight precedes every permission change. A later failure does not roll
back partially changed descendants: it attempts to gate the same named root to
0700, reports whether that quarantine was verified, and returns no receipt.
The caller must not prepare a worker census or launch a child after a refusal.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time

from .native_writable_allocation import _acl_state, _boundary_file, _mount_identity

SCHEMA = "supervisor-owner-writable-worktree-provisioning@1"
MAX_ENTRIES = 65_536
MAX_DEPTH = 64
MAX_RELATIVE_PATH_BYTES = 8192
MAX_TOTAL_PATH_BYTES = 16 * 1024 * 1024
_LOCK = threading.RLock()
_ACTIVE = {}
_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


class OwnerWritableWorktreeProvisioningError(ValueError):
    """Refusal with explicit permission mutation and quarantine observations."""

    def __init__(self, message):
        super().__init__(message)
        self.mutation_started = False
        self.quarantine_verified = False
        self.quarantine_error_type = None
        self.changed_entries = 0
        self.phase = "admission"


def _need(value, reason):
    if not value:
        raise OwnerWritableWorktreeProvisioningError(reason)


def _deadline(value):
    _need(value is None or (type(value) is int and value > 0),
        "original absolute provisioning deadline must be positive integer nanoseconds")
    if value is not None and time.monotonic_ns() >= value:
        raise OwnerWritableWorktreeProvisioningError("original owner deadline exhausted during provisioning")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
        info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _stable_directory(info):
    return info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)


def _path(workspace, relative):
    return workspace.joinpath(*(os.fsdecode(part) for part in relative))


def _check_active(state):
    _need(not state["reentered"], "owner writable provisioning attempted reentry")


def _names(descriptor):
    names = []
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.scandir(descriptor) as entries:
        for entry in entries:
            _need(len(names) < MAX_ENTRIES, "worktree directory names exceed inventory bound")
            names.append(os.fsencode(entry.name))
    return sorted(names)


def _ancestry(path, *, owner, deadline):
    """Return the named parent FD and a protected, no-follow ancestry witness."""
    _deadline(deadline)
    _need(path.is_absolute() and path.resolve(strict=True) == path
        and len(os.fsencode(path)) <= 4096 and len(path.parts) <= MAX_DEPTH + 1,
        "canonical bounded provisioning ancestry required")
    descriptor = os.open(path.anchor, _FLAGS | os.O_DIRECTORY)
    rows = []
    current = Path(path.anchor)
    try:
        for component in (None, *path.parts[1:]):
            if component is not None:
                next_descriptor = os.open(component, _FLAGS | os.O_DIRECTORY, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
                current = current / component
            _deadline(deadline)
            before = os.fstat(descriptor)
            _need(stat.S_ISDIR(before.st_mode) and before.st_uid in (0, owner)
                and not before.st_mode & 0o022,
                "provisioning ancestry must remain protected")
            mount = _mount_identity(descriptor)
            acl = _acl_state(current, descriptor=descriptor)
            named = current.lstat()
            _need(_stable_directory(before) == _stable_directory(os.fstat(descriptor))
                == _stable_directory(named) and current.resolve(strict=True) == current,
                "provisioning ancestry changed during observation")
            rows.append((str(current), _stable_directory(before), mount, acl))
        return descriptor, tuple(rows)
    except BaseException:
        os.close(descriptor)
        raise


def _observe(descriptor, *, path, owner, mount, protected):
    before = os.fstat(descriptor)
    directory = stat.S_ISDIR(before.st_mode)
    _need((directory or stat.S_ISREG(before.st_mode)) and before.st_uid == owner
        and before.st_gid == owner and (directory or before.st_nlink == 1),
        "initial worktree requires only owner regular files and directories in its shared group")
    _need(not before.st_mode & 0o002
        and (not protected or not before.st_mode & 0o022)
        and (directory or not before.st_mode & 0o7000),
        "initial worktree permissions or special file bits are unsafe")
    _need(_mount_identity(descriptor) == mount,
        "worktree object crosses the retained mount or namespace")
    acl = _acl_state(path, descriptor=descriptor)
    after = os.fstat(descriptor)
    _need(_identity(before) == _identity(after)
        and _mount_identity(descriptor) == mount
        and _acl_state(path, descriptor=descriptor) == acl,
        "worktree object changed during descriptor observation")
    return dict(kind="directory" if directory else "file", current=_identity(after),
        original=_identity(after), acl=acl)


def _walk(root_fd, *, workspace, owner, mount, deadline, protected):
    manifest, total_paths = {}, 0

    def walk(descriptor, relative):
        nonlocal total_paths
        _deadline(deadline)
        _need(len(relative) <= MAX_DEPTH and len(manifest) < MAX_ENTRIES,
            "worktree provisioning inventory exceeds entry or depth bounds")
        path_bytes = len(b"/".join(relative))
        total_paths += path_bytes
        _need(path_bytes <= MAX_RELATIVE_PATH_BYTES and total_paths <= MAX_TOTAL_PATH_BYTES,
            "worktree provisioning path inventory exceeds bounds")
        record = _observe(descriptor, path=_path(workspace, relative),
            owner=owner, mount=mount, protected=protected)
        manifest[relative] = record
        if record["kind"] == "file":
            if relative == (b".git",):
                _need(record["current"][6] <= 4096, "linked worktree marker exceeds bound")
                raw = os.pread(descriptor, 4097, 0)
                _need(len(raw) == record["current"][6]
                    and _identity(os.fstat(descriptor)) == record["current"],
                    "linked worktree marker changed during read")
                record["marker_sha256"] = hashlib.sha256(raw).hexdigest()
            return
        names = _names(descriptor)
        _need(len(names) == len(set(names)), "worktree directory names repeat")
        for name in names:
            _need(name not in (b"", b".", b"..") and b"/" not in name and b"\0" not in name
                and (name != b".git" or not relative),
                "worktree contains an unsafe or nested Git control path")
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            _need(stat.S_ISDIR(named.st_mode) or stat.S_ISREG(named.st_mode),
                "worktree contains a symlink or special file")
            flags = _FLAGS | (os.O_DIRECTORY if stat.S_ISDIR(named.st_mode) else 0)
            child = os.open(name, flags, dir_fd=descriptor)
            try:
                _need(_identity(os.fstat(child)) == _identity(named),
                    "worktree named entry changed before descriptor open")
                walk(child, (*relative, name))
                _need(_identity(os.stat(name, dir_fd=descriptor, follow_symlinks=False))
                    == manifest[(*relative, name)]["current"],
                    "worktree named entry changed during inventory")
            finally:
                os.close(child)
        _need(_identity(os.fstat(descriptor)) == record["current"]
            and _names(descriptor) == names,
            "worktree directory inventory changed during observation")

    walk(root_fd, ())
    _need((b".git",) in manifest and manifest[(b".git",)]["kind"] == "file",
        "initial linked worktree requires its owner regular Git marker")
    return manifest


def _preflight(root_fd, *, workspace, owner, mount, deadline):
    return _walk(root_fd, workspace=workspace, owner=owner, mount=mount,
        deadline=deadline, protected=True)


def _verify_inventory(root_fd, manifest, *, workspace, owner, mount, deadline, completed=False):
    current = _walk(root_fd, workspace=workspace, owner=owner, mount=mount,
        deadline=deadline, protected=False)
    _need(set(current) == set(manifest) and all(
        current[key]["kind"] == retained["kind"]
        and current[key]["current"] == retained["current"]
        and current[key]["acl"] == retained["acl"]
        and current[key].get("marker_sha256") == retained.get("marker_sha256")
        for key, retained in manifest.items()),
        "worktree inventory differs from its retained descriptor journal")
    if completed:
        for relative, record in manifest.items():
            _need(stat.S_IMODE(record["current"][2]) == _target_mode(relative, record),
                "worktree completed permissions differ")


def _target_mode(relative, record):
    if not relative:
        return 0o3770
    if record["kind"] == "directory":
        return 0o2770
    return 0o440 if relative == (b".git",) else (0o770 if record["original"][2] & 0o111 else 0o660)


def _open_entry(root_fd, manifest, relative, *, workspace, owner, mount, deadline):
    descriptor = os.dup(root_fd)
    prefix = ()
    try:
        for name in (None, *relative):
            if name is not None:
                prefix = (*prefix, name)
                record = manifest[prefix]
                flags = _FLAGS | (os.O_DIRECTORY if record["kind"] == "directory" else 0)
                child = os.open(name, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            _deadline(deadline)
            observed = _observe(descriptor, path=_path(workspace, prefix), owner=owner,
                mount=mount, protected=False)
            _need(observed["current"] == manifest[prefix]["current"]
                and observed["kind"] == manifest[prefix]["kind"],
                "worktree journal ancestry or opened object changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _named_root(root_fd, parent_fd, *, workspace, ancestry, owner, mount, deadline):
    current_parent, current_ancestry = _ancestry(workspace.parent, owner=owner, deadline=deadline)
    try:
        _need(current_ancestry == ancestry
            and _stable_directory(os.fstat(current_parent)) == _stable_directory(os.fstat(parent_fd)),
            "worktree named parent ancestry changed")
        opened, named = os.fstat(root_fd), os.stat(workspace.name, dir_fd=parent_fd, follow_symlinks=False)
        _need(stat.S_ISDIR(opened.st_mode) and _identity(opened) == _identity(named)
            and opened.st_uid == opened.st_gid == owner and _mount_identity(root_fd) == mount,
            "worktree root no longer has its original named binding")
        _acl_state(workspace, descriptor=root_fd)
    finally:
        os.close(current_parent)


def _apply(root_fd, manifest, relative, mode, *, workspace, owner, mount, deadline, state):
    _check_active(state)
    descriptor = _open_entry(root_fd, manifest, relative,
        workspace=workspace, owner=owner, mount=mount, deadline=deadline)
    try:
        record = manifest[relative]
        before = record["current"]
        state["mutation_started"] = True  # fchmod may change mode and then raise.
        os.fchmod(descriptor, mode)
        state["changed_entries"] += 1
        _check_active(state)
        after = _identity(os.fstat(descriptor))
        _need(after[:2] == before[:2] and after[3:8] == before[3:8]
            and stat.S_IFMT(after[2]) == stat.S_IFMT(before[2])
            and stat.S_IMODE(after[2]) == mode,
            "worktree object changed while applying its declared mode")
        record["current"] = after
        _need(_mount_identity(descriptor) == mount,
            "worktree mount changed during permission mutation")
        _acl_state(_path(workspace, relative), descriptor=descriptor)
    finally:
        os.close(descriptor)


def _inventory_digest(manifest, field):
    rows = [(b"/".join(path).hex(), record["kind"], record[field],
        record.get("marker_sha256")) for path, record in sorted(manifest.items())]
    raw = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def provision_owner_writable_worktree(*, boundary, workspace, absolute_deadline_ns=None):
    """Provision an initial protected tree; return an inert scoped observation.

    The initial tree must contain only this owner's non-writable-group objects.
    Already delegated or worker-owned trees require another protocol. Failure
    after mutation attempts to quarantine, never automatically restores modes.
    """
    from . import source_formalization_context as guard

    state = dict(reentered=False, mutation_started=False, changed_entries=0, phase="admission")
    parent_fd = root_fd = None
    active = False
    workspace = Path(workspace)
    try:
        _deadline(absolute_deadline_ns)
        _need(workspace.is_absolute() and workspace.resolve(strict=True) == workspace,
            "canonical existing initial worktree required")
        with _LOCK:
            existing = _ACTIVE.get(str(workspace))
            if existing is not None:
                existing["reentered"] = True
                raise OwnerWritableWorktreeProvisioningError("owner writable provisioning is already active")
            _ACTIVE[str(workspace)] = state
            active = True
        boundary = Path(boundary)
        _need(boundary.is_absolute() and boundary.resolve(strict=True) == boundary
            and not boundary.is_relative_to(workspace),
            "protected owner boundary must remain outside its editable worktree")
        value, boundary_sha = guard._owner_boundary(boundary)
        value_bytes = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        owner, worker = value["owner_uid"], value["worker_uid"]
        _need(type(owner) is int and owner > 0
            and owner == os.getuid() == os.geteuid() == os.getgid() == os.getegid()
            and type(worker) is int and worker > 0 and worker != owner,
            "initial worktree requires distinct nonroot owner and selected worker")
        _need(type(value.get("allowed_worktree_roots")) is list
            and 1 <= len(value["allowed_worktree_roots"]) <= 64,
            "authenticated worktree roots required")
        allowed = [Path(name) for name in value["allowed_worktree_roots"] if type(name) is str]
        _need(len(allowed) == len(value["allowed_worktree_roots"])
            and all(path.is_absolute() and path.resolve(strict=True) == path for path in allowed)
            and any(workspace != path and workspace.is_relative_to(path) for path in allowed),
            "initial worktree is outside its authenticated boundary")
        _need(type(value.get("owner_private_paths")) is list
            and len(value["owner_private_paths"]) <= 64,
            "explicit authenticated owner private paths required")
        private = [Path(name) for name in value["owner_private_paths"] if type(name) is str]
        _need(len(private) == len(value["owner_private_paths"])
            and all(path.is_absolute() and path.resolve(strict=True) == path
                and not path.is_relative_to(workspace) and not workspace.is_relative_to(path)
                for path in private),
            "initial worktree overlaps or differs from its protected private path policy")
        boundary_witness = _boundary_file(boundary)
        _need(boundary_witness[1] == boundary_sha, "initial worktree boundary digest differs")
        boundary_parent_fd, boundary_ancestry = _ancestry(boundary.parent,
            owner=owner, deadline=absolute_deadline_ns)
        os.close(boundary_parent_fd)

        def boundary_current():
            current_value, current_sha = guard._owner_boundary(boundary)
            _need(json.dumps(current_value, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode() == value_bytes and current_sha == boundary_sha
                and _boundary_file(boundary) == boundary_witness,
                "initial worktree boundary changed during provisioning")
            current_parent, current_ancestry = _ancestry(boundary.parent,
                owner=owner, deadline=absolute_deadline_ns)
            try:
                _need(current_ancestry == boundary_ancestry,
                    "initial worktree boundary ancestry changed")
            finally:
                os.close(current_parent)

        parent_fd, ancestry = _ancestry(workspace.parent, owner=owner, deadline=absolute_deadline_ns)
        named = os.stat(workspace.name, dir_fd=parent_fd, follow_symlinks=False)
        _need(stat.S_ISDIR(named.st_mode), "initial worktree root must be a directory")
        root_fd = os.open(workspace.name, _FLAGS | os.O_DIRECTORY, dir_fd=parent_fd)
        _need(_identity(os.fstat(root_fd)) == _identity(named), "initial worktree root changed before open")
        mount = _mount_identity(root_fd)
        _need(mount == _mount_identity(parent_fd),
            "initial worktree root crosses its protected parent mount")
        state["phase"] = "preflight"
        manifest = _preflight(root_fd, workspace=workspace, owner=owner,
            mount=mount, deadline=absolute_deadline_ns)
        boundary_current()
        _verify_inventory(root_fd, manifest, workspace=workspace, owner=owner,
            mount=mount, deadline=absolute_deadline_ns)
        _named_root(root_fd, parent_fd, workspace=workspace, ancestry=ancestry,
            owner=owner, mount=mount, deadline=absolute_deadline_ns)
        _check_active(state)
        _deadline(absolute_deadline_ns)
        state["phase"] = "root_gate"
        _apply(root_fd, manifest, (), 0o700, workspace=workspace,
            owner=owner, mount=mount, deadline=absolute_deadline_ns, state=state)
        state["phase"] = "descendants"
        for relative in sorted((path for path in manifest if path), key=lambda path: (-len(path), path)):
            _apply(root_fd, manifest, relative, _target_mode(relative, manifest[relative]),
                workspace=workspace, owner=owner, mount=mount, deadline=absolute_deadline_ns, state=state)
        state["phase"] = "closing_inventory"
        boundary_current()
        _verify_inventory(root_fd, manifest, workspace=workspace, owner=owner,
            mount=mount, deadline=absolute_deadline_ns)
        _named_root(root_fd, parent_fd, workspace=workspace, ancestry=ancestry,
            owner=owner, mount=mount, deadline=absolute_deadline_ns)
        _check_active(state)
        state["phase"] = "root_publication"
        _apply(root_fd, manifest, (), 0o3770, workspace=workspace,
            owner=owner, mount=mount, deadline=absolute_deadline_ns, state=state)
        _verify_inventory(root_fd, manifest, workspace=workspace, owner=owner,
            mount=mount, deadline=absolute_deadline_ns, completed=True)
        _named_root(root_fd, parent_fd, workspace=workspace, ancestry=ancestry,
            owner=owner, mount=mount, deadline=absolute_deadline_ns)
        _check_active(state)
        _deadline(absolute_deadline_ns)
        return dict(schema=SCHEMA, workspace=str(workspace), owner_uid=owner, worker_uid=worker,
            shared_group=owner, boundary_manifest_sha256=boundary_sha,
            absolute_deadline_ns=absolute_deadline_ns, entries=len(manifest),
            original_inventory_sha256=_inventory_digest(manifest, "original"),
            provisioned_inventory_sha256=_inventory_digest(manifest, "current"),
            initial_owner_only_tree_verified=True, descriptor_permission_transition_verified=True,
            named_root_policy_verified=True, access_and_default_acl_absent=True,
            same_mount_verified=True, root_mode="03770", git_marker_mode="0440",
            worker_isolation_qualified=False, native_allocation_verified=False,
            source_contents_verified=False, filesystem_writer_lease_verified=False,
            serialized_observation_grants_authority=False, execution_authority=False,
            completion_authority=False, proof_authority=False)
    except BaseException as original:
        error = original if isinstance(original, OwnerWritableWorktreeProvisioningError) else (
            OwnerWritableWorktreeProvisioningError("owner worktree provisioning refused: " + type(original).__name__))
        error.mutation_started = state["mutation_started"]
        error.changed_entries = state["changed_entries"]
        error.phase = state["phase"]
        if state["mutation_started"] and root_fd is not None:
            try:
                os.fchmod(root_fd, 0o700)
            except BaseException as quarantine_error:
                error.quarantine_error_type = type(quarantine_error).__name__
            try:
                _need(stat.S_IMODE(os.fstat(root_fd).st_mode) == 0o700,
                    "failed provisioning root is not gated")
                _named_root(root_fd, parent_fd, workspace=workspace, ancestry=ancestry,
                    owner=owner, mount=mount, deadline=None)
                error.quarantine_verified = True
            except BaseException as quarantine_error:
                error.quarantine_error_type = type(quarantine_error).__name__
        if not isinstance(original, Exception):
            for name in ("mutation_started", "changed_entries", "phase", "quarantine_verified", "quarantine_error_type"):
                setattr(original, name, getattr(error, name))
            raise
        if error is original:
            raise
        raise error from original
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        if active:
            with _LOCK:
                if _ACTIVE.get(str(workspace)) is state:
                    del _ACTIVE[str(workspace)]


__all__ = ["SCHEMA", "OwnerWritableWorktreeProvisioningError", "provision_owner_writable_worktree"]
