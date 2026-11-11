"""Explicit initial group-search provisioning while the workspace stays private.

The trusted deployment boundary selects a dedicated anchor and exact parent
directories. This helper changes only selected 0700 directories to 0710. It
does not prove actual group membership, enable Git access, launch a worker, or
create a continuous writer lease. The caller must have no live worker yet.
Failure leaves partial parent changes in place and grants no launch authority.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import threading

from .native_writable_allocation import _acl_state, _boundary_file, _mount_identity
from .owner_worktree_parent import (
    MAX_DEPTH, MAX_PATH_BYTES, _deadline, _directory, _existing_prefix, _verify_prefix,
)

SCHEMA = "supervisor-owner-worktree-traversal@1"
MAX_DIRECTORIES = 64
_LOCK = threading.RLock()
_ACTIVE = {}


class OwnerWorktreeTraversalError(ValueError):
    """Refusal preserving acknowledged changes and an uncertain chmod attempt."""

    def __init__(self, message):
        super().__init__(message)
        self.mutation_attempted = False
        self.changed_directories = ()
        self.attempted_path = None
        self.effects_complete = True
        self.phase = "admission"


def _need(value, reason):
    if not value:
        raise OwnerWorktreeTraversalError(reason)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()


def _policy_path(value, *, allow_root=False):
    _need(type(value) is str, "exact traversal policy path required")
    path = Path(value)
    _need(path.is_absolute() and str(path) == value and ".." not in path.parts
        and (allow_root or path != Path(path.anchor))
        and len(path.parts) - 1 <= MAX_DEPTH and len(os.fsencode(path)) <= MAX_PATH_BYTES
        and path.resolve(strict=True) == path,
        "canonical bounded existing traversal policy path required")
    return path


def _overlap(first, second):
    return first.is_relative_to(second) or second.is_relative_to(first)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _check_active(state):
    _need(not state["reentered"], "owner worktree traversal attempted reentry")


def _open_directory(plan, index, *, deadline):
    path = Path(plan["rows"][index][0])
    descriptor, rows, missing = _existing_prefix(path, owner=plan["owner"], deadline=deadline)
    try:
        _need(not missing and rows == plan["rows"][:index + 1],
            "traversal directory differs from its original named ancestry")
        _need(_mount_identity(descriptor) == rows[-1][2]
            and _acl_state(path, descriptor=descriptor) == rows[-1][3],
            "traversal directory mount or ACL changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _single_child(plan, index, *, deadline):
    """A newly opened search gate may expose only the next admitted chain node."""
    descriptor = _open_directory(plan, index, deadline=deadline)
    try:
        names = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                names.append(entry.name)
                _need(len(names) <= 1, "private traversal gate contains an unrelated sibling")
        child = Path(plan["rows"][index + 1][0])
        _need(names == [child.name]
            and _identity(os.stat(child.name, dir_fd=descriptor, follow_symlinks=False))
                == plan["rows"][index + 1][1],
            "private traversal gate does not contain its exact next workspace directory")
        _need(_identity(os.fstat(descriptor)) == plan["rows"][index][1],
            "private traversal gate changed during its inventory check")
    finally:
        os.close(descriptor)


def _boundary_current(plan, *, deadline):
    from . import source_formalization_context as guard
    _deadline(deadline)
    value, digest = guard._owner_boundary(plan["boundary"])
    _need(_wire(value) == plan["boundary_bytes"] and digest == plan["boundary_digest"]
        and _boundary_file(plan["boundary"]) == plan["boundary_witness"],
        "trusted traversal boundary changed")
    _verify_prefix(plan["boundary_rows"], directory=plan["boundary"].parent,
        owner=plan["owner"], deadline=deadline)


def _closing(plan, *, deadline):
    _boundary_current(plan, deadline=deadline)
    for index in plan["changes"]:
        _single_child(plan, index, deadline=deadline)
    _verify_prefix(plan["rows"], directory=plan["workspace"], owner=plan["owner"], deadline=deadline)
    workspace_row = plan["rows"][-1]
    _need(_identity(os.fstat(plan["workspace_fd"])) == workspace_row[1]
        and stat.S_IMODE(workspace_row[1][2]) == 0o700
        and _mount_identity(plan["workspace_fd"]) == workspace_row[2]
        and _acl_state(plan["workspace"], descriptor=plan["workspace_fd"]) == workspace_row[3],
        "initial workspace lost its original private policy")


def _preflight(*, boundary, workspace, deadline):
    from . import source_formalization_context as guard
    workspace_fd = None
    try:
        _deadline(deadline)
        owner = os.getuid()
        _need(type(owner) is int and owner > 0
            and owner == os.geteuid() == os.getgid() == os.getegid(),
            "traversal provisioning requires one positive owner UID and group")
        boundary = _policy_path(os.fspath(boundary))
        value, digest = guard._owner_boundary(boundary)
        _need(type(value.get("owner_uid")) is int and value["owner_uid"] == owner
            and type(value.get("worker_uid")) is int
            and value["worker_uid"] > 0 and value["worker_uid"] != owner
            and type(value.get("worker_shared_group")) is int
            and value["worker_shared_group"] == owner,
            "explicit distinct worker and exact shared group policy required")
        anchor = _policy_path(value.get("worktree_traversal_anchor"))
        _need(workspace != anchor and workspace.is_relative_to(anchor),
            "traversal anchor must strictly contain the workspace")
        selected = value.get("worktree_traversal_directories")
        _need(type(selected) is list and len(selected) <= MAX_DIRECTORIES,
            "explicit bounded traversal directory list required")
        selected = tuple(_policy_path(name) for name in selected)
        _need(len(set(selected)) == len(selected), "traversal policy directories repeat")
        allowed = value.get("allowed_worktree_roots")
        _need(type(allowed) is list and 1 <= len(allowed) <= MAX_DIRECTORIES,
            "authenticated traversal worktree roots required")
        allowed = tuple(_policy_path(name) for name in allowed)
        admitted = tuple(path for path in allowed if path.is_relative_to(anchor)
            and path != workspace and workspace.is_relative_to(path))
        _need(admitted, "traversal anchor does not contain an admitted worktree root")
        private = value.get("owner_private_paths")
        _need(type(private) is list and len(private) <= MAX_DIRECTORIES,
            "explicit authenticated private path policy required")
        private = tuple(_policy_path(name, allow_root=True) for name in private)
        _need(all(not _overlap(path, protected) for path in (anchor, *selected)
            for protected in (boundary, *private)),
            "traversal policy overlaps a protected owner boundary or private path")
        _need(all(path != workspace and workspace.is_relative_to(path)
            and path.is_relative_to(anchor)
            and any(_overlap(path, root) for root in admitted) for path in selected),
            "selected traversal directory is outside its dedicated admitted parent chain")
        workspace_fd, rows, missing = _existing_prefix(workspace, owner=owner, deadline=deadline)
        _need(not missing and len(rows) >= 2, "existing initial workspace required")
        by_path = {Path(row[0]): index for index, row in enumerate(rows)}
        _need(anchor in by_path and rows[by_path[anchor]][1][3:5] == (owner, owner),
            "traversal anchor must be the owner's existing shared-group directory")
        workspace_row = rows[-1]
        _need(workspace_row[1][3:5] == (owner, owner)
            and stat.S_IMODE(workspace_row[1][2]) == 0o700
            and workspace_row[2] == rows[-2][2],
            "initial traversal workspace must remain owner 0700 on its parent mount")
        selected_indices = tuple(by_path[path] for path in selected)
        changes = []
        for index, row in enumerate(rows[:-1]):
            mode = stat.S_IMODE(row[1][2])
            if index in selected_indices:
                _need(row[1][3:5] == (owner, owner) and mode in (0o700, 0o710, 0o750)
                    and index > 0 and row[2] == rows[index - 1][2],
                    "selected traversal gate must be an exact protected owner directory on its parent mount")
                if mode == 0o700:
                    changes.append(index)
            else:
                search = 0o010 if row[1][4] == owner else 0o001
                _need(mode & search, "unselected ancestor lacks declared-worker group search")
        boundary_fd, boundary_rows, missing = _existing_prefix(boundary.parent, owner=owner, deadline=deadline)
        os.close(boundary_fd)
        _need(not missing, "trusted traversal boundary ancestry is missing")
        witness = _boundary_file(boundary)
        _need(witness[1] == digest, "traversal boundary digest differs")
        plan = dict(owner=owner, worker=value["worker_uid"], workspace=workspace,
            workspace_fd=workspace_fd, rows=rows, boundary=boundary, boundary_rows=boundary_rows,
            boundary_bytes=_wire(value), boundary_digest=digest, boundary_witness=witness,
            anchor=anchor, selected=selected_indices, changes=tuple(sorted(changes, reverse=True)))
        _closing(plan, deadline=deadline)
        return plan
    except BaseException:
        if workspace_fd is not None:
            os.close(workspace_fd)
        raise


def provision_owner_worktree_traversal(*, boundary, workspace, absolute_deadline_ns=None):
    """Open only explicitly selected initial parent search gates, without rollback."""
    state = dict(reentered=False, mutation_attempted=False, changed_directories=[],
        attempted_path=None, phase="admission")
    active = False
    plan = None
    key = None
    try:
        _deadline(absolute_deadline_ns)
        workspace = _directory(workspace)
        key = str(workspace)
        with _LOCK:
            existing = _ACTIVE.get(key)
            if existing is not None:
                existing["reentered"] = True
                raise OwnerWorktreeTraversalError("owner worktree traversal is already active")
            _ACTIVE[key] = state
            active = True
        state["phase"] = "preflight"
        plan = _preflight(boundary=boundary, workspace=workspace, deadline=absolute_deadline_ns)
        _closing(plan, deadline=absolute_deadline_ns)
        _check_active(state)
        for index in plan["changes"]:
            state["phase"] = "parent_search"
            _closing(plan, deadline=absolute_deadline_ns)
            _check_active(state)
            descriptor = _open_directory(plan, index, deadline=absolute_deadline_ns)
            try:
                before = plan["rows"][index]
                _deadline(absolute_deadline_ns)
                _check_active(state)
                state["attempted_path"] = before[0]
                state["mutation_attempted"] = True  # A failed fchmod may have taken effect.
                os.fchmod(descriptor, 0o710)
                actual = _identity(os.fstat(descriptor))
                _need(actual[:2] == before[1][:2] and actual[3:] == before[1][3:]
                    and stat.S_ISDIR(actual[2]) and stat.S_IMODE(actual[2]) == 0o710
                    and _mount_identity(descriptor) == before[2]
                    and _acl_state(Path(before[0]), descriptor=descriptor) == before[3],
                    "traversal gate changed while publishing group search")
                updated = before[0], actual, before[2], before[3]
                plan["rows"] = (*plan["rows"][:index], updated, *plan["rows"][index + 1:])
                _verify_prefix(plan["rows"][:index + 1], directory=Path(before[0]),
                    owner=plan["owner"], deadline=absolute_deadline_ns)
                state["changed_directories"].append(before[0])
                _check_active(state)
            finally:
                os.close(descriptor)
            _closing(plan, deadline=absolute_deadline_ns)
        state["phase"] = "closing_search"
        _deadline(absolute_deadline_ns)
        _closing(plan, deadline=absolute_deadline_ns)
        _check_active(state)
        _deadline(absolute_deadline_ns)
        _check_active(state)
        return dict(schema=SCHEMA, workspace=str(workspace), anchor=str(plan["anchor"]),
            owner_uid=plan["owner"], worker_uid=plan["worker"], worker_shared_group=plan["owner"],
            boundary_manifest_sha256=plan["boundary_digest"], absolute_deadline_ns=absolute_deadline_ns,
            selected_directories=[plan["rows"][index][0] for index in plan["selected"]],
            changed_directories=list(state["changed_directories"]), workspace_mode="0700",
            declared_group_search_bits_verified=True, changed_private_gate_inventory_verified=True,
            workspace_policy_unchanged=True, access_and_default_acl_absent=True,
            worker_group_membership_verified=False, worker_traversal_qualified=False,
            worker_isolation_qualified=False, git_control_access_verified=False,
            native_allocation_verified=False, filesystem_writer_lease_verified=False,
            serialized_observation_grants_authority=False, execution_authority=False,
            proof_authority=False, completion_authority=False)
    except BaseException as original:
        error = original if isinstance(original, OwnerWorktreeTraversalError) else OwnerWorktreeTraversalError(
            "owner worktree traversal refused: " + type(original).__name__)
        error.mutation_attempted = state["mutation_attempted"]
        error.changed_directories = tuple(state["changed_directories"])
        error.attempted_path = state["attempted_path"]
        error.effects_complete = not state["mutation_attempted"]
        error.phase = state["phase"]
        if not isinstance(original, Exception):
            for name in ("mutation_attempted", "changed_directories", "attempted_path", "effects_complete", "phase"):
                setattr(original, name, getattr(error, name))
            raise
        if error is original:
            raise
        raise error from original
    finally:
        if plan is not None:
            os.close(plan["workspace_fd"])
        if active:
            with _LOCK:
                if _ACTIVE.get(key) is state:
                    del _ACTIVE[key]


__all__ = ["SCHEMA", "OwnerWorktreeTraversalError", "provision_owner_worktree_traversal"]
