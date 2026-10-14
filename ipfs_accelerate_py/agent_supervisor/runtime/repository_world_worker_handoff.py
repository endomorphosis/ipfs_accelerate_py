"""A process-local native assignment survives one read-only root transition.

Only the allocated root's permission bits become 0750. Original source files,
canonical/control roots, native owners and immutable allocation witnesses are
retained. This object neither launches nor authenticates a worker; detached
observations cannot be hydrated into a live handoff.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
import weakref

from . import current_source_worktree_allocation as allocation_owner
from . import repository_world_prelaunch as prelaunch
from .current_source_lookup import CurrentSourceLookup
from .current_source_task_admission import NativeSourceTaskAdmission
from .repository_source_scope import RepositorySourceScope
from ..merge import worktree_lifecycle as lifecycle
from ..task_sources import typed_state_owner as typed
from ..task_sources.control_plane_contracts import canonical_json_bytes, content_identity

SCHEMA = "repository-world-readonly-worker-handoff@1"
PROFILE = "selected_source_observation@1"
MAX_OBSERVATION_BYTES = 16384
MAX_TIMEOUT_SECONDS = 180
DEFAULT_TIMEOUT_SECONDS = 90.0
FALSE_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "omission_authority", "provider_launch", "world_dispatch_ready", "compiled_context_verified",
    "worker_process_verified", "worker_edit_authority", "whole_repository_observed"})
_SEAL = object()
_ISSUED = weakref.WeakKeyDictionary()
_ISSUANCE_LOCK = threading.RLock()
_ALLOCATION_FIELDS = ("_seal", "_issued_self", "_task_admission", "_server", "_canonical_repository",
    "_workspace", "_worktree_root", "_attempt_root", "_merge_target", "_board_namespace", "_git",
    "_store_path", "_attempt_directory", "_directories", "_store", "_source", "_bridge", "_intent",
    "_original_policy", "_task_bytes", "_fixed_files", "_lifecycle_bytes", "_material",
    "_original_custody", "_lookup", "_lookup_refs", "_lookup_bytes", "_nominated_source_scope",
    "_original_nominated_source_scope", "_original")


class RepositoryWorldWorkerHandoffError(ValueError):
    """Closed refusal codes without private paths, prompt or owner credentials."""


def _need(value, code):
    if not value:
        raise RepositoryWorldWorkerHandoffError(code)


def _wire(value):
    raw = canonical_json_bytes(value)
    _need(len(raw) <= MAX_OBSERVATION_BYTES, "handoff_observation_byte_limit")
    return raw


def _directory_identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode,
        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory(path):
    fd = prelaunch._root_fd(path)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISDIR(info.st_mode), "handoff_directory_kind_changed")
        return _directory_identity(info)
    finally:
        os.close(fd)


def _selected_metadata(root, paths):
    """Bind UID/GID/modes and every selected ancestor without byte copies."""
    result = []
    for raw_path in paths:
        parts = bytes.fromhex(raw_path).split(b"/")
        _need(0 < len(parts) <= 64 and all(part and part not in {b".", b"..", b".git"}
              and b"\0" not in part for part in parts), "handoff_source_path_profile")
        held, fresh, leaf = [prelaunch._root_fd(root)], [], None
        try:
            for part in parts[:-1]:
                held.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=held[-1]))
                info = os.fstat(held[-1])
                _need(info.st_uid == os.geteuid() and not info.st_mode & 0o022,
                      "handoff_selected_ancestor_not_owner_controlled")
            leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=held[-1])
            info = os.fstat(leaf)
            _need(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                  and info.st_nlink == 1 and not info.st_mode & 0o022,
                  "handoff_selected_file_custody_changed")
            directories = tuple(_directory_identity(os.fstat(fd)) for fd in held)
            fresh.append(prelaunch._root_fd(root))
            for index in range(len(held)):
                if index:
                    fresh.append(os.open(parts[index - 1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fresh[-1]))
                _need(_directory_identity(os.fstat(fresh[-1])) == directories[index],
                      "handoff_selected_ancestor_namespace_changed")
            _need(_file_identity(info) == _file_identity(os.fstat(leaf))
                  == _file_identity(os.stat(parts[-1], dir_fd=fresh[-1], follow_symlinks=False)),
                  "handoff_selected_file_namespace_changed")
            result.append((raw_path, directories, _file_identity(info)))
        finally:
            if leaf is not None:
                os.close(leaf)
            for fd in reversed(fresh):
                os.close(fd)
            for fd in reversed(held):
                os.close(fd)
    return tuple(result)


class NativeRepositoryWorldWorkerHandoff:
    """Original native owner object; no worker or provider authority is issued."""

    __slots__ = ("_seal", "_assignment", "_task", "_allocation", "_server", "_source", "_canonical",
        "_workspace", "_prompt", "_pid", "_uid", "_start", "_boot", "_worker_uid", "_worker_gid",
        "_timeout", "_deadline", "_root_fd", "_root_identity", "_allocation_refs", "_directory_refs",
        "_control_gids", "_source_rows", "_source_witnesses", "_selected_refs", "_original_observation",
        "_renewal", "_state", "_lock", "_checking", "_material", "__weakref__")

    def __init__(self, seal, *, assignment, worker_uid, worker_gid, timeout_seconds):
        _need(seal is _SEAL and type(self) is NativeRepositoryWorldWorkerHandoff,
              "handoff_original_capture_required")
        _need(type(assignment) is prelaunch.NativeRepositoryWorldPrelaunchAssignment,
              "handoff_exact_prelaunch_assignment_required")
        _need(type(worker_uid) is int and 0 < worker_uid < (1 << 31) and worker_uid != os.geteuid()
              and type(worker_gid) is int and 0 <= worker_gid < (1 << 31), "handoff_worker_principal_invalid")
        _need(type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
              and 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS, "handoff_timeout_invalid")
        original = prelaunch.NativeRepositoryWorldPrelaunchAssignment.require_current(assignment)
        self._seal, self._assignment = seal, assignment
        self._task, self._allocation = assignment._task, assignment._allocation
        self._server, self._source = assignment._server, assignment._source
        self._canonical, self._workspace = assignment._canonical, assignment._workspace
        self._prompt = assignment._prompt
        self._pid, self._uid = os.getpid(), os.geteuid()
        self._start, _parent, self._boot = typed._process_runtime_facts(self._pid)
        self._worker_uid, self._worker_gid, self._timeout = worker_uid, worker_gid, float(timeout_seconds)
        timeout_nanoseconds = round(self._timeout * 1_000_000_000)
        _need(timeout_nanoseconds / 1_000_000_000 == self._timeout,
              "handoff_timeout_resolution_unsupported")
        self._deadline = time.monotonic() + self._timeout
        self._root_fd = prelaunch._root_fd(self._workspace)
        self._state, self._lock, self._checking = "CAPTURED", threading.RLock(), False
        try:
            self._root_identity = _directory_identity(os.fstat(self._root_fd))
            _need(worker_gid == self._root_identity[3] and self._root_identity[4] != 0o750
                  and not self._root_identity[4] & 0o077, "handoff_original_root_profile_required")
            self._allocation_refs = tuple((key, getattr(self._allocation, key)) for key in _ALLOCATION_FIELDS)
            directories = list(path for path, _identity in self._allocation._directories)
            directories.append(self._allocation._fixed_files[1])
            self._directory_refs = tuple((path, _directory(path)) for path in dict.fromkeys(directories))
            self._control_gids = tuple((path, path.lstat().st_gid) for path, _identity in self._allocation._fixed_files[0])
            self._source_rows, self._source_witnesses = assignment._source_rows, assignment._source_witnesses
            self._selected_refs = (_selected_metadata(self._canonical, self._source._paths),
                                   _selected_metadata(self._workspace, self._source._paths))
            self._original_observation = _wire(original)
            self._renewal = self._allocation._renewal
            material = dict(schema=SCHEMA, profile=PROFILE,
                original_assignment_cid=original["assignment_cid"], owner_pid=self._pid, owner_uid=self._uid,
                owner_start_time_ticks=self._start, boot_id=self._boot, worker_uid=worker_uid, worker_gid=worker_gid,
                prompt_sha256=self._prompt[0], prompt_bytes=self._prompt[1], timeout_nanoseconds=timeout_nanoseconds,
                task_cid=original["task_cid"], task_revision=original["task_revision"],
                expected_task_body_cid=original["expected_task_body_cid"], claim_identity=original["claim_identity"],
                allocation_cid=original["allocation_cid"], source_binding_cid=original["source_binding_cid"],
                nominated_scope=original["nominated_scope"],
                selected_source_count=original["selected_source_count"], selected_source_bytes=original["selected_source_bytes"],
                original_root_mode=self._root_identity[4], transitioned_root_mode=0o750,
                permission_scope="allocated_root_read_search_only_selected_files_unchanged", **dict.fromkeys(FALSE_FIELDS, False))
            material["handoff_id"] = content_identity(material)
            self._material = _wire(material)
        except BaseException:
            os.close(self._root_fd)
            self._root_fd = None
            raise

    def _seal_material(self):
        return (self._seal, self._assignment, self._task, self._allocation, self._server, self._source,
            self._canonical, self._workspace, self._prompt, self._pid, self._uid, self._start, self._boot,
            self._worker_uid, self._worker_gid, self._timeout, self._deadline, self._root_fd, self._root_identity,
            self._allocation_refs, self._directory_refs, self._control_gids, self._source_rows, self._source_witnesses,
            self._selected_refs, self._original_observation, self._lock, self._material)

    def _publish_state(self, state):
        self._state = state
        with _ISSUANCE_LOCK:
            _ISSUED[self] = (NativeRepositoryWorldWorkerHandoff._seal_material(self), state, self._renewal)

    def _retained(self, *, permit_closed=False):
        _need(type(self) is NativeRepositoryWorldWorkerHandoff and getattr(self, "_seal", None) is _SEAL,
              "handoff_original_live_object_required")
        with _ISSUANCE_LOCK:
            retained = _ISSUED.get(self)
            _need(retained is not None, "handoff_live_object_not_issued")
            current = NativeRepositoryWorldWorkerHandoff._seal_material(self)
            _need(retained[0] == current and retained[1] == self._state and retained[2] is self._renewal
                  and all(retained[0][index] is current[index] for index in tuple(range(1, 9)) + tuple(range(18, 28))),
                  "handoff_original_custody_replaced")
        _need(os.getpid() == self._pid and os.getuid() == os.geteuid() == self._uid
              and typed._process_runtime_facts(self._pid)[::2] == (self._start, self._boot),
              "handoff_original_owner_process_required")
        _need(permit_closed or self._state in {"CAPTURED", "TRANSITIONED"}, "handoff_not_current")
        _need(permit_closed or time.monotonic() < self._deadline, "handoff_expired")
        prelaunch.NativeRepositoryWorldPrelaunchAssignment._retained(self._assignment, issued=True)
        _need(self._task is self._assignment._task and self._allocation is self._assignment._allocation
              and self._server is self._assignment._server and self._source is self._assignment._source,
              "handoff_original_assignment_replaced")
        _need(all(getattr(self._allocation, key, None) is value for key, value in self._allocation_refs),
              "handoff_original_allocation_replaced")
        allocation = self._allocation
        _need(allocation._server is self._task._server
              and allocation._source.intent is allocation._intent
              and allocation._intent._bound_connection is self._task._connection
              and allocation._directories is allocation._original_policy[0]
              and (allocation._git, allocation._store_path, allocation._attempt_directory)
                  == allocation._original_policy[1:4]
              and allocation._intent is allocation._original_policy[4]
              and allocation._store.repo_root == self._canonical
              and allocation._store.store_dir == allocation._store_path
              and allocation._bridge.task_source is allocation._source
              and allocation._bridge.attempt_root == allocation._attempt_root
              and allocation._bridge.repo_root == self._canonical
              and allocation._bridge.board_namespace == allocation._board_namespace
              and allocation._bridge.merge_target_branch == allocation._merge_target
              and allocation._bridge.task_header_prefix == allocation._original_policy[5]
              and allocation._bridge.portal_factory is allocation._original_policy[6]
              and self._source._native_source_task_admission is self._task
              and self._source._native_source_worktree_allocation is allocation,
              "handoff_original_allocation_deep_policy_changed")
        root_mode = self._root_identity[4] if self._state == "CAPTURED" else 0o750
        expected = (*self._root_identity[:4], root_mode)
        if not permit_closed:
            _need(_directory_identity(os.fstat(self._root_fd)) == _directory(self._workspace) == expected,
                  "handoff_allocated_root_custody_changed")

    def _controls(self):
        for path, original in self._directory_refs:
            expected = (*original[:4], 0o750) if path == self._workspace and self._state == "TRANSITIONED" else original
            _need(_directory(path) == expected, "handoff_control_directory_changed")
        for (path, expected), (_same_path, gid) in zip(self._allocation._fixed_files[0], self._control_gids):
            _need(allocation_owner._read(path)[0] == expected and path.lstat().st_gid == gid,
                  "handoff_immutable_control_changed")
        store = self._allocation._store
        lifecycle_path = store.workspace_path_for(self._workspace)
        record_witness, record_raw = allocation_owner._read(lifecycle_path)
        value = allocation_owner._json(record_raw)
        record = lifecycle.WorkspaceLifecycleRecord.from_dict(value)
        _need(allocation_owner._wire(value) == allocation_owner._wire(record.to_dict())
              and record.record_id == record.compute_record_id(), "handoff_lifecycle_schema_changed")
        stable = record.to_dict()
        renewal = tuple(stable.pop(key) for key in ("fence", "updated_at", "expires_at"))
        _need(allocation_owner._wire(stable) == self._allocation._lifecycle_bytes
              and all(a >= b for a, b in zip(renewal, self._renewal))
              and record.created_at <= record.updated_at < record.expires_at
              and record.expires_at > time.time(), "handoff_lifecycle_identity_or_lease_changed")
        birth = lifecycle.read_process_birth(self._task._peer[0])
        _need(birth is not None and birth == record.owner and birth.start_time_ticks == self._task._peer[2],
              "handoff_lifecycle_owner_birth_changed")
        index_path = store.task_index_path_for(canonical_task_cid=record.canonical_task_cid,
            task_id=record.task_id, attempt=record.attempt)
        index_witness, index_raw = allocation_owner._read(index_path)
        _need(allocation_owner._wire(allocation_owner._json(index_raw))
              == allocation_owner._wire(store._task_index_payload(record)), "handoff_lifecycle_index_changed")
        _need(allocation_owner._read(lifecycle_path)[0] == record_witness
              and allocation_owner._read(index_path)[0] == index_witness,
              "handoff_lifecycle_changed_during_read")
        self._renewal = renewal
        NativeRepositoryWorldWorkerHandoff._publish_state(self, self._state)

    def _detached(self):
        NativeSourceTaskAdmission._native(self._task)
        NativeRepositoryWorldWorkerHandoff._controls(self)
        CurrentSourceLookup.check_native(self._source, self._task._connection, self._task._gateway.identity)
        CurrentSourceLookup._check_source(self._source)
        if self._assignment._nominated is not None:
            scope, expected, task_join, source, binding_bytes, cas = self._assignment._nominated
            _need(type(scope) is RepositorySourceScope and task_join is self._task._nominated_source_scope
                  and task_join is self._task._original_nominated_source_scope
                  and task_join[0] is scope and task_join[2] == expected and task_join[4] is source
                  and task_join[6] is cas and source is self._source and source._store.cas is cas
                  and binding_bytes == task_join[5] == self._allocation._lookup_bytes == self._task._source_bytes,
                  "handoff_original_nomination_join_changed")
            value = RepositorySourceScope.observation(scope)
            _need(canonical_json_bytes(value) == task_join[1] and value["scope_cid"] == expected
                  and source._paths == task_join[3] == tuple(row["raw_path_hex"] for row in value["selected"]),
                  "handoff_original_nomination_changed")
        witnesses = tuple((path, prelaunch._source_file(self._canonical, path, size, sha),
            prelaunch._source_file(self._workspace, path, size, sha)) for path, size, sha, _cid in self._source_rows)
        _need(witnesses == self._source_witnesses, "handoff_selected_source_bytes_or_custody_changed")
        selected = (_selected_metadata(self._canonical, self._source._paths),
                    _selected_metadata(self._workspace, self._source._paths))
        expected = self._selected_refs
        if self._state == "TRANSITIONED":
            expected = (expected[0], tuple((path, ((*directories[0][:4], 0o750), *directories[1:]), file)
                                           for path, directories, file in expected[1]))
        _need(selected == expected, "handoff_selected_metadata_changed")
        NativeSourceTaskAdmission._native(self._task)
        NativeRepositoryWorldWorkerHandoff._controls(self)
        CurrentSourceLookup.check_native(self._source, self._task._connection, self._task._gateway.identity)
        CurrentSourceLookup._check_source(self._source)
        NativeRepositoryWorldWorkerHandoff._retained(self)

    def _checked(self):
        NativeRepositoryWorldWorkerHandoff._retained(self)
        if self._state == "CAPTURED":
            observed = prelaunch.NativeRepositoryWorldPrelaunchAssignment.require_current(self._assignment)
            _need(_wire(observed) == self._original_observation, "handoff_original_assignment_changed")
        # The original CAS is replayed before the same native claim fence.
        if self._assignment._nominated is not None:
            NativeSourceTaskAdmission.require_nominated_source_scope(self._task)
        rows = prelaunch.NativeRepositoryWorldPrelaunchAssignment._rows(self._assignment)
        _need(rows == self._source_rows, "handoff_source_population_changed")
        NativeRepositoryWorldWorkerHandoff._retained(self)
        task = NativeSourceTaskAdmission.require_current(self._task)
        original = json.loads(self._original_observation)
        _need(all(task[key] == original[key] for key in
                  ("task_cid", "task_revision", "expected_task_body_cid", "claim_identity")),
              "handoff_native_assignment_changed")
        with self._server._lock:
            NativeRepositoryWorldWorkerHandoff._detached(self)
        return NativeRepositoryWorldWorkerHandoff._observation(self, current=True)

    def _observation(self, *, current):
        material = json.loads(self._material)
        material.update(state=self._state, source_continuity_verified=current,
            native_assignment_continuity_verified=current,
            read_only_transition_verified=current and self._state == "TRANSITIONED")
        _wire(material)
        return material

    def require_current(self):
        NativeRepositoryWorldWorkerHandoff._retained(self)
        with self._lock:
            _need(not self._checking, "handoff_reentry_refused")
            self._checking = True
            try:
                return NativeRepositoryWorldWorkerHandoff._checked(self)
            finally:
                self._checking = False

    def apply_readonly_transition(self):
        NativeRepositoryWorldWorkerHandoff._retained(self)
        with self._lock:
            _need(self._state == "CAPTURED" and not self._checking, "handoff_transition_already_consumed")
            self._checking = True
            try:
                NativeRepositoryWorldWorkerHandoff._checked(self)
                def guarded_transition(_lease):
                    with self._server._lock:
                        NativeRepositoryWorldWorkerHandoff._detached(self)
                        os.fchmod(self._root_fd, 0o750)
                        NativeRepositoryWorldWorkerHandoff._publish_state(self, "TRANSITIONED")
                        NativeRepositoryWorldWorkerHandoff._detached(self)
                self._task._coordinator.execute_with_task_claim_fence(self._task._claim, guarded_transition)
                return NativeRepositoryWorldWorkerHandoff._checked(self)
            except BaseException:
                NativeRepositoryWorldWorkerHandoff._publish_state(self, "RECOVERY_REQUIRED")
                raise
            finally:
                self._checking = False

    def require_observer_assignment(self, *, prompt, worker_uid, worker_gid):
        NativeRepositoryWorldWorkerHandoff._retained(self)
        _need(prelaunch._prompt_binding(prompt) == self._prompt
              and type(worker_uid) is int and type(worker_gid) is int
              and (worker_uid, worker_gid) == (self._worker_uid, self._worker_gid),
              "handoff_observer_assignment_mismatch")
        return NativeRepositoryWorldWorkerHandoff.require_current(self)

    def require_peer_assignment(self, *, server, canonical_repository, workspace, prompt,
                                task_admission, worktree_admission, worker_uid, worker_gid):
        NativeRepositoryWorldWorkerHandoff._retained(self)
        _need(server is self._server and task_admission is self._task and worktree_admission is self._allocation
              and isinstance(canonical_repository, Path) and canonical_repository == self._canonical
              and isinstance(workspace, Path) and workspace == self._workspace,
              "handoff_peer_assignment_mismatch")
        return NativeRepositoryWorldWorkerHandoff.require_observer_assignment(self,
            prompt=prompt, worker_uid=worker_uid, worker_gid=worker_gid)

    def close(self):
        NativeRepositoryWorldWorkerHandoff._retained(self, permit_closed=True)
        with self._lock:
            _need(not self._checking, "handoff_reentry_refused")
            if self._root_fd is not None:
                os.close(self._root_fd)
                self._root_fd = None
            state = "RECOVERY_REQUIRED" if self._state == "RECOVERY_REQUIRED" else "CLOSED"
            NativeRepositoryWorldWorkerHandoff._publish_state(self, state)
            return NativeRepositoryWorldWorkerHandoff._observation(self, current=False)


def capture_repository_world_worker_handoff(*, assignment, worker_uid, worker_gid,
                                           timeout_seconds=DEFAULT_TIMEOUT_SECONDS):
    """Capture original owner objects; no process, grant or worker UID is created."""
    handoff = NativeRepositoryWorldWorkerHandoff(_SEAL, assignment=assignment,
        worker_uid=worker_uid, worker_gid=worker_gid, timeout_seconds=timeout_seconds)
    NativeRepositoryWorldWorkerHandoff._publish_state(handoff, "CAPTURED")
    try:
        NativeRepositoryWorldWorkerHandoff.require_current(handoff)
        return handoff
    except BaseException:
        NativeRepositoryWorldWorkerHandoff.close(handoff)
        raise
