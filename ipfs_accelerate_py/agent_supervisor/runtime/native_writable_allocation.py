"""Private custody policy for an already provisioned writable worktree root.

This module changes no permissions, issues no worker UID, and certifies no
permission transition or nested workspace tree. The native allocation kernel
retains task, Portal, lifecycle and Git controls separately.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

SCHEMA = "supervisor-native-writable-worktree-policy@1"
_SEAL = object()


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_gid,
        stat.S_IMODE(info.st_mode), info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns)


def _mount_identity(descriptor):
    """Observe the opened object rather than inferring its mount from a path."""
    with open("/proc/self/fdinfo/" + str(descriptor), "rb") as stream:
        raw = stream.read(8193)
    _need(len(raw) <= 8192, "writable allocation mount observation exceeds bound")
    rows = [row.split(b":", 1)[1].strip() for row in raw.splitlines()
        if row.startswith(b"mnt_id:")]
    _need(len(rows) == 1 and re.fullmatch(rb"[0-9]{1,20}", rows[0]) is not None,
        "writable allocation requires one native mount identity")
    namespace = os.readlink("/proc/self/ns/mnt")
    _need(re.fullmatch(r"mnt:\[[0-9]+\]", namespace) is not None,
        "writable allocation mount namespace is unavailable")
    return int(rows[0]), namespace


def _acl_state(path, *, descriptor):
    """Fail closed if an access or default ACL can widen the declared modes."""
    _need(isinstance(path, Path), "exact writable allocation path required")
    names = os.listxattr(descriptor)
    _need(all(type(name) is str for name in names)
        and not any(name in ("system.posix_acl_access", "system.posix_acl_default")
            for name in names), "writable allocation ACL delegation is not supported")
    return "access_and_default_acl_absent"


def _opened(path, *, directory, owner, group, mode, read=False):
    _need(isinstance(path, Path) and path.is_absolute()
        and path.resolve(strict=True) == path,
        "writable allocation path must be canonical")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    if directory:
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _need((stat.S_ISDIR(before.st_mode) if directory else stat.S_ISREG(before.st_mode))
            and before.st_uid == owner and before.st_gid == group
            and stat.S_IMODE(before.st_mode) == mode
            and (directory or before.st_nlink == 1),
            "writable allocation root or marker policy differs")
        mount = _mount_identity(descriptor)
        acl = _acl_state(path, descriptor=descriptor)
        raw = None
        if read:
            _need(0 <= before.st_size <= 4096,
                "writable allocation Git marker exceeds bound")
            raw = os.read(descriptor, 4097)
            _need(len(raw) == before.st_size,
                "writable allocation Git marker read differs")
        after, named = os.fstat(descriptor), path.lstat()
        # Directory contents may change under the worker. Root inode, mode,
        # group, mount and ACL custody remain fixed; marker bytes do not.
        identity = lambda info: ((info.st_dev, info.st_ino, info.st_uid,
            info.st_gid, stat.S_IMODE(info.st_mode)) if directory else _identity(info))
        _need(identity(before) == identity(after) == identity(named)
            and path.resolve(strict=True) == path
            and _mount_identity(descriptor) == mount
            and _acl_state(path, descriptor=descriptor) == acl,
            "writable allocation object changed during observation")
        return identity(after), mount, acl, None if raw is None else hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


def _boundary_file(path):
    """Pin an already authenticated attempt boundary; do not issue authority."""
    _need(path.is_absolute() and path.resolve(strict=True) == path,
        "writable allocation boundary must be canonical")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and not before.st_mode & 0o022 and 0 <= before.st_size <= 32768,
            "writable allocation requires a bounded protected boundary")
        raw = os.read(descriptor, 32769)
        after, named = os.fstat(descriptor), path.lstat()
        _need(len(raw) == before.st_size and _identity(before) == _identity(after) == _identity(named)
            and path.resolve(strict=True) == path,
            "writable allocation boundary changed during observation")
        return _identity(after), hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


class _WritableAllocationPolicy:
    """Exact private policy retaining the first owner attempt and native facts."""

    def __init__(self, seal, *, first_worker_attempt, workspace, worktree_root):
        from . import owner_worker_attempt as attempts
        from . import source_formalization_context as guard

        _need(seal is _SEAL and type(self) is _WritableAllocationPolicy
            and type(first_worker_attempt) is attempts.OwnerWorkerAttempt,
            "exact prepared first owner worker attempt required")
        common = attempts.OwnerWorkerAttempt._current(first_worker_attempt, deadline=True)
        boundary = first_worker_attempt._boundary
        value, digest = guard._owner_boundary(boundary)
        owner, worker = value["owner_uid"], value["worker_uid"]
        _need(type(owner) is int and owner > 0
            and owner == os.getuid() == os.geteuid() == os.getgid() == os.getegid()
            and type(worker) is int and worker > 0 and worker != owner,
            "writable allocation requires distinct nonroot owner and selected worker")
        _need(workspace == first_worker_attempt._workspace
            and type(value.get("allowed_worktree_roots")) is list
            and value["allowed_worktree_roots"],
            "writable allocation differs from first worker or boundary scope")
        allowed = []
        for name in value["allowed_worktree_roots"]:
            _need(type(name) is str, "writable allocation boundary root is invalid")
            root = Path(name)
            _need(root.is_absolute() and root.resolve(strict=True) == root,
                "writable allocation boundary root must be canonical")
            allowed.append(root)
        _need(worktree_root in allowed and workspace != worktree_root
            and workspace.is_relative_to(worktree_root),
            "writable allocation root is outside the authenticated boundary")
        self._seal, self._issued_self = seal, self
        self._attempt, self._state = first_worker_attempt, first_worker_attempt._state
        self._common, self._boundary = first_worker_attempt._common, boundary
        self._attempt_inputs = (first_worker_attempt._token, first_worker_attempt._parent,
            first_worker_attempt._parent_identity, first_worker_attempt._workspace_identity,
            self._state.lock)
        self._workspace, self._worktree_root = workspace, worktree_root
        self._owner, self._worker, self._digest = owner, worker, digest
        self._boundary_identity = _boundary_file(boundary)
        _need(self._boundary_identity[1] == digest == common["boundary_manifest_sha256"],
            "writable allocation boundary bytes differ from first worker")
        self._root = _opened(workspace, directory=True, owner=owner, group=owner, mode=0o3770)
        self._marker = _opened(workspace / ".git", directory=False,
            owner=owner, group=owner, mode=0o440, read=True)
        _need(self._root[1] == self._marker[1],
            "writable allocation Git marker is on a different mount")
        self._prepared = self._state.prepared
        _need(type(self._state) is attempts._State and type(self._prepared) is attempts._Event,
            "writable allocation requires its actual prepared census event")
        self._prepared_inputs = self._prepared.name, self._prepared.identity, self._prepared.raw
        self._original = (self, self._attempt, self._state, self._common, self._boundary,
            self._workspace, self._worktree_root, self._owner, self._worker, self._digest,
            self._boundary_identity, self._root, self._marker, self._prepared,
            self._attempt_inputs, self._prepared_inputs)
        self.require_prepared()
        material = dict(schema=SCHEMA, first_worker_attempt_id=common["attempt_id"],
            workspace=str(workspace), worktree_root=str(worktree_root),
            boundary_manifest_sha256=digest, owner_uid=owner, worker_uid=worker,
            shared_group=owner, absolute_deadline_ns=common["absolute_deadline_ns"],
            workspace_mode="03770", git_marker_mode="0440",
            mount_identity=self._root[1][0], mount_namespace=self._root[1][1],
            access_and_default_acl_absent=True, permission_transition_verified=False,
            current_workspace_policy_verified=True, nested_workspace_policy_verified=False,
            worker_isolation_qualified=False, serialized_observation_grants_authority=False,
            proof_authority=False, execution_authority=False, completion_authority=False)
        material["delegation_cid"] = cid_for_structured(material)
        self._material = _wire(material)
        self._original_material = self._material

    def _retained(self):
        from . import owner_worker_attempt as attempts
        current = (self, self._attempt, self._state, self._common, self._boundary,
            self._workspace, self._worktree_root, self._owner, self._worker, self._digest,
            self._boundary_identity, self._root, self._marker, self._prepared,
            self._attempt_inputs, self._prepared_inputs)
        _need(type(self) is _WritableAllocationPolicy and self._seal is _SEAL
            and self._issued_self is self
            and len(current) == len(self._original)
            and all(a is b if index < 4 or index in (13, 14, 15) else a == b
                for index, (a, b) in enumerate(zip(current, self._original)))
            and type(self._state) is attempts._State
            and type(self._prepared) is attempts._Event
            and (self._prepared.name, self._prepared.identity, self._prepared.raw) == self._prepared_inputs
            and self._attempt._common == self._common
            and self._attempt._state is self._state
            and self._attempt._boundary == self._boundary
            and self._attempt._workspace == self._workspace
            and self._state.prepared is self._prepared
            and self._attempt._token is self._attempt_inputs[0]
            and (self._attempt._parent, self._attempt._parent_identity,
                self._attempt._workspace_identity) == self._attempt_inputs[1:4]
            and self._state.lock is self._attempt_inputs[4]
            and (not hasattr(self, "_original_material")
                or self._material == self._original_material),
            "writable allocation original owner custody changed")
        common = attempts.OwnerWorkerAttempt._current(self._attempt, deadline=True)
        _need(_boundary_file(self._boundary) == self._boundary_identity,
            "writable allocation original boundary changed")
        descriptor = os.open(self._attempt._parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            _need(attempts._directory_identity(os.fstat(descriptor)) == self._attempt._parent_identity,
                "writable allocation first census directory changed")
            attempts.OwnerWorkerAttempt._check_event(self._attempt, descriptor, self._prepared)
            _need(self._prepared.raw == self._common,
                "writable allocation original prepared census changed")
        finally:
            os.close(descriptor)
        _need(_opened(self._workspace, directory=True, owner=self._owner,
                group=self._owner, mode=0o3770) == self._root
            and _opened(self._workspace / ".git", directory=False, owner=self._owner,
                group=self._owner, mode=0o440, read=True) == self._marker,
            "writable allocation root or marker custody changed")
        return common

    def require_prepared(self):
        common = self._retained()
        state = self._state
        with state.lock:
            _need(not state.spent and not state.retired and state.spawned is None
                and state.retired_event is None and state.process is None and state.child is None,
                "writable allocation must capture the prepared unspent first worker")
        return common

    def directory_identity(self, path):
        _need(path == self._workspace, "writable allocation only delegates its exact workspace root")
        self._retained()
        return self._root

    def observation(self):
        self._retained()
        return json.loads(self._material)

    def require_candidate_binding(self, *, prior_worker_attempt, source_broker, absolute_deadline_ns):
        from .worker_source_broker import WorkerSourceBroker
        common = self._retained()
        _need(prior_worker_attempt is self._attempt
            and type(source_broker) is WorkerSourceBroker
            and source_broker.workspace == self._workspace
            and source_broker.boundary == self._boundary
            and source_broker.owner_uid == self._owner
            and source_broker.worker_uid == self._worker
            and source_broker.descriptor["boundary_manifest_sha256"] == self._digest
            and source_broker._process is self._state.process
            and type(absolute_deadline_ns) is int
            and absolute_deadline_ns == common["absolute_deadline_ns"],
            "candidate differs from the original writable allocation delegation")
        return json.loads(self._material)


def _capture_policy(*, first_worker_attempt, workspace, worktree_root):
    return _WritableAllocationPolicy(_SEAL, first_worker_attempt=first_worker_attempt,
        workspace=workspace, worktree_root=worktree_root)
