"""Process-local owner assignment checks before any worker permission handoff.

This gate reuses original native task, allocation and source-reader custody.
It issues no worker grant, packet or process; its observation cannot be hydrated
into authority. Literal prompt binding is not authenticated context compilation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import weakref

from .current_source_task_admission import NativeSourceTaskAdmission
from .current_source_worktree_allocation import NativeSourceWorktreeAllocation
from .current_source_lookup import CurrentSourceLookup
from ..task_sources.control_plane_contracts import canonical_json_bytes, content_identity


SCHEMA = "repository-world-owner-prelaunch-assignment@1"
MAX_PROMPT_BYTES = 2 * 1024 * 1024
MAX_SOURCE_PATHS = 64
MAX_SELECTED_SOURCE_BYTES = 1024 * 1024
MAX_OBSERVATION_BYTES = 16384
FALSE_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "omission_authority", "provider_launch", "world_dispatch_ready", "compiled_context_verified",
    "worker_process_verified", "worker_permission_transition_authorized"})
_SEAL = object()
_ISSUED = weakref.WeakKeyDictionary()
_ISSUANCE_LOCK = threading.RLock()


class RepositoryWorldPrelaunchError(ValueError):
    """Closed prelaunch refusal codes without paths, task text or credentials."""


def _need(value, code):
    if not value:
        raise RepositoryWorldPrelaunchError(code)


def _prompt_binding(prompt):
    _need(type(prompt) in {str, bytes}, "prelaunch_exact_prompt_required")
    if type(prompt) is str:
        # Reject oversized strings before allocating their UTF-8 representation.
        _need(0 < len(prompt) <= MAX_PROMPT_BYTES, "prelaunch_prompt_byte_limit")
        raw = prompt.encode("utf-8", "strict")
    else:
        raw = prompt
    _need(0 < len(raw) <= MAX_PROMPT_BYTES, "prelaunch_prompt_byte_limit")
    raw.decode("utf-8", "strict")
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _wire(value):
    raw = canonical_json_bytes(value)
    _need(len(raw) <= MAX_OBSERVATION_BYTES, "prelaunch_observation_byte_limit")
    return raw


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _root_fd(path):
    _need(isinstance(path, Path) and path.is_absolute() and ".." not in path.parts,
          "prelaunch_source_root_invalid")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in path.parts[1:]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        _need(info.st_uid == os.geteuid() and not info.st_mode & 0o022,
              "prelaunch_source_root_not_owner_controlled")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _source_file(root, raw_path, size, digest):
    _need(type(raw_path) is str and re.fullmatch(r"(?:[0-9a-f]{2})+", raw_path) is not None
          and len(raw_path) <= 8192, "prelaunch_source_path_invalid")
    parts = bytes.fromhex(raw_path).split(b"/")
    _need(all(part and part not in {b".", b".."} and b"\0" not in part for part in parts),
          "prelaunch_source_path_invalid")
    held = [_root_fd(root)]
    fresh = []
    fd = None
    try:
        for name in parts[:-1]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=held[-1])
            info = os.fstat(child)
            if info.st_uid != os.geteuid() or info.st_mode & 0o022:
                os.close(child)
                raise RepositoryWorldPrelaunchError("prelaunch_source_directory_not_owner_controlled")
            held.append(child)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=held[-1])
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
              and not before.st_mode & 0o022 and before.st_nlink == 1 and before.st_size == size,
              "prelaunch_source_file_custody_changed")
        sha, remaining = hashlib.sha256(), size
        while remaining:
            raw = os.read(fd, min(65536, remaining))
            _need(bool(raw), "prelaunch_source_short_read")
            remaining -= len(raw)
            sha.update(raw)
        _need(not os.read(fd, 1) and sha.hexdigest() == digest, "prelaunch_source_bytes_changed")
        _need(_file_identity(before) == _file_identity(os.fstat(fd))
              == _file_identity(os.stat(parts[-1], dir_fd=held[-1], follow_symlinks=False)),
              "prelaunch_source_file_custody_changed")
        # Reopen the namespace, rather than accepting a stable detached parent.
        # Every selected ancestor must still name the same owner-held directory.
        fresh.append(_root_fd(root))
        for index in range(len(held)):
            if index:
                fresh.append(os.open(parts[index - 1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                     dir_fd=fresh[-1]))
            current, original = os.fstat(fresh[-1]), os.fstat(held[index])
            _need((current.st_dev, current.st_ino, current.st_uid, current.st_mode)
                  == (original.st_dev, original.st_ino, original.st_uid, original.st_mode),
                  "prelaunch_source_directory_namespace_changed")
        _need(_file_identity(before) == _file_identity(os.stat(parts[-1], dir_fd=fresh[-1],
                                                               follow_symlinks=False)),
              "prelaunch_source_file_namespace_changed")
        return _file_identity(before)
    finally:
        if fd is not None:
            os.close(fd)
        for directory in reversed(fresh):
            os.close(directory)
        for directory in reversed(held):
            os.close(directory)


class NativeRepositoryWorldPrelaunchAssignment:
    """Issued only in its owner process; no dispatch or permission authority."""

    __slots__ = ("_seal", "_task", "_allocation", "_source", "_server", "_canonical",
        "_workspace", "_pid", "_uid", "_prompt", "_lock", "_checking", "_source_rows",
        "_source_witnesses", "_observation", "_nominated", "_original", "__weakref__")

    def __init__(self, seal, *, task_admission, worktree_admission, prompt):
        _need(seal is _SEAL and type(self) is NativeRepositoryWorldPrelaunchAssignment,
              "prelaunch_host_capture_required")
        _need(type(task_admission) is NativeSourceTaskAdmission
              and type(worktree_admission) is NativeSourceWorktreeAllocation,
              "prelaunch_exact_native_handles_required")
        _need(worktree_admission._task_admission is task_admission
              and worktree_admission._server is task_admission._server,
              "prelaunch_assignment_handles_disagree")
        source = task_admission._source
        _need(type(source) is CurrentSourceLookup and worktree_admission._lookup is source
              and source._native_source_task_admission is task_admission
              and source._native_source_worktree_allocation is worktree_admission
              and source._root == worktree_admission._workspace,
              "prelaunch_original_bound_source_required")
        self._seal = seal
        self._task, self._allocation, self._source = task_admission, worktree_admission, source
        self._server = task_admission._server
        self._canonical, self._workspace = worktree_admission._canonical_repository, worktree_admission._workspace
        self._pid, self._uid = os.getpid(), os.geteuid()
        self._prompt = _prompt_binding(prompt)
        self._lock, self._checking = threading.RLock(), False
        self._source_rows = self._source_witnesses = self._observation = None
        self._nominated = worktree_admission._nominated_source_scope
        self._original = (self._task, self._allocation, self._source, self._server,
            self._canonical, self._workspace, self._source._store, self._source._store.cas,
            self._source._request, self._source._limits, self._source._paths, self._nominated)

    def _seal_material(self):
        return (self._seal, self._original, self._prompt, self._pid, self._uid,
                self._source_rows, self._source_witnesses, self._observation)

    def _retained(self, *, issued=True):
        _need(type(self) is NativeRepositoryWorldPrelaunchAssignment and getattr(self, "_seal", None) is _SEAL
              and os.getpid() == getattr(self, "_pid", None)
              and os.getuid() == os.geteuid() == getattr(self, "_uid", None),
              "prelaunch_original_owner_process_required")
        if issued:
            with _ISSUANCE_LOCK:
                retained = _ISSUED.get(self)
                _need(retained is not None, "prelaunch_live_gate_not_issued")
                current = NativeRepositoryWorldPrelaunchAssignment._seal_material(self)
                _need(retained == current
                      and all(retained[index] is current[index] for index in (1, 2, 5, 6, 7)),
                      "prelaunch_live_gate_not_issued")
        _need(all(a is b for a, b in zip((self._task, self._allocation, self._source,
                  self._server, self._canonical, self._workspace), self._original[:6]))
              and self._nominated is self._original[-1], "prelaunch_original_assignment_replaced")
        current = (self._task, self._allocation, self._source, self._server,
            self._canonical, self._workspace, self._source._store, self._source._store.cas,
            self._source._request, self._source._limits, self._source._paths, self._nominated)
        _need(all(a is b for a, b in zip(current, self._original))
              and self._allocation._task_admission is self._task
              and self._task._server is self._allocation._server is self._server
              and self._task._source is self._allocation._lookup is self._source
              and self._source._root == self._workspace
              and self._allocation._canonical_repository is self._canonical
              and self._allocation._workspace is self._workspace
              and self._allocation._nominated_source_scope is self._nominated,
              "prelaunch_original_assignment_replaced")

    def _rows(self):
        paths = self._source._paths
        _need(type(paths) is tuple and 0 < len(paths) <= MAX_SOURCE_PATHS,
              "prelaunch_source_population_limit")
        restored = self._source._load()
        by_path = {row.entry.raw_path_hex: row for row in restored.entries}
        rows, total = [], 0
        for path in paths:
            row = by_path.get(path)
            _need(row is not None and row.entry.git_mode in {"100644", "100755"}
                  and type(row.data) is bytes, "prelaunch_selected_source_unavailable")
            total += len(row.data)
            _need(total <= MAX_SELECTED_SOURCE_BYTES, "prelaunch_selected_source_byte_limit")
            rows.append((path, len(row.data), hashlib.sha256(row.data).hexdigest(), row.source_cid))
        return tuple(rows)

    def _observe(self, *, issued):
        NativeRepositoryWorldPrelaunchAssignment._retained(self, issued=issued)
        task = NativeSourceTaskAdmission.require_current(self._task)
        allocation = NativeSourceWorktreeAllocation.require_current(self._allocation)
        nominated = (None if self._nominated is None else
            NativeSourceWorktreeAllocation.require_nominated_source_scope(self._allocation))
        rows = NativeRepositoryWorldPrelaunchAssignment._rows(self)
        NativeRepositoryWorldPrelaunchAssignment._retained(self, issued=issued)
        # CAS callbacks may release a claim while leaving its SQL body intact.
        # Close through the real native claim/allocation fence after the last
        # CAS replay, before SERVER is acquired for detached filesystem checks.
        closing_task = NativeSourceTaskAdmission.require_current(self._task)
        closing_allocation = NativeSourceWorktreeAllocation.require_current(self._allocation)
        _need(task == closing_task and allocation == closing_allocation,
              "prelaunch_assignment_changed_after_source_replay")
        # All coordinator clocks and CAS callbacks above finish before the
        # detached closing source checks below. No COOR acquisition under SERVER.
        with self._server._lock:
            NativeSourceTaskAdmission._native(self._task)
            fixed, record, _identity, _binding = NativeSourceWorktreeAllocation._observe(self._allocation)
            stable = record.to_dict()
            renewal = tuple(stable.pop(key) for key in ("fence", "updated_at", "expires_at"))
            _need(fixed == self._allocation._fixed_files
                  and self._allocation._lifecycle_bytes == NativeRepositoryWorldPrelaunchAssignment._allocation_module_wire(stable)
                  and all(a >= b for a, b in zip(renewal, self._allocation._renewal)),
                  "prelaunch_allocation_changed_after_native_fence")
            NativeSourceTaskAdmission.require_source_binding(self._task, server=self._server, service=self._source)
            NativeSourceWorktreeAllocation.require_source_binding(self._allocation, server=self._server, service=self._source)
            self._source.check_native(self._task._connection, self._task._gateway.identity)
            binding = self._source.binding(self._task._gateway.identity)
            _need(all(binding[key] == task[key] == allocation[key] for key in
                      ("task_cid", "task_revision", "expected_task_body_cid")),
                  "prelaunch_source_assignment_disagrees")
            self._source._check_source()
            witnesses = tuple((path, _source_file(self._canonical, path, size, sha),
                _source_file(self._workspace, path, size, sha)) for path, size, sha, _cid in rows)
            # Close filesystem and SQL custody after all reads. These detached
            # checks perform no CAS callback or coordinator acquisition.
            NativeSourceTaskAdmission._native(self._task)
            final_fixed, final_record, _identity, _binding = NativeSourceWorktreeAllocation._observe(self._allocation)
            final_stable = final_record.to_dict()
            final_renewal = tuple(final_stable.pop(key) for key in ("fence", "updated_at", "expires_at"))
            _need(final_fixed == fixed and NativeRepositoryWorldPrelaunchAssignment._allocation_module_wire(final_stable)
                  == self._allocation._lifecycle_bytes and all(a >= b for a, b in zip(final_renewal, renewal)),
                  "prelaunch_allocation_changed_after_source_read")
            self._source.check_native(self._task._connection, self._task._gateway.identity)
            self._source._check_source()
            NativeRepositoryWorldPrelaunchAssignment._retained(self, issued=issued)
        if self._source_rows is not None:
            _need(rows == self._source_rows and witnesses == self._source_witnesses,
                  "prelaunch_selected_source_custody_changed")
        nomination = (None if nominated is None else {key: nominated[key] for key in
            ("source_scope_cid", "frontier_cid", "population_cid", "native_snapshot_cid",
             "selected_count", "selected_bytes", "frontier_count", "allowed_paths_cid")})
        material = dict(schema=SCHEMA, readiness_scope="owner_prelaunch_assignment_only",
            task_cid=task["task_cid"], task_revision=task["task_revision"],
            expected_task_body_cid=task["expected_task_body_cid"], claim_identity=task["claim_identity"],
            allocation_cid=allocation["allocation_cid"], source_binding_cid=binding["binding_cid"],
            source_inventory_cid=binding["source_inventory_cid"], context_cid=binding["context_cid"],
            allowed_paths_cid=binding["allowed_paths_cid"], selected_source_count=len(rows),
            selected_source_bytes=sum(row[1] for row in rows),
            selected_source_witness_sha256=hashlib.sha256(_wire([list(row) for row in rows])).hexdigest(),
            prompt_sha256=self._prompt[0], prompt_bytes=self._prompt[1],
            prompt_binding_scope="caller_supplied_literal_only", nominated_scope=nomination,
            owner_pid=self._pid, owner_uid=self._uid,
            source_check_scope="complete_selected_native_path_population_in_both_original_roots",
            **dict.fromkeys(FALSE_FIELDS, False))
        material["assignment_cid"] = content_identity(material)
        encoded = _wire(material)
        _need(self._observation is None or encoded == self._observation,
              "prelaunch_assignment_material_changed")
        return encoded, rows, witnesses

    @staticmethod
    def _allocation_module_wire(value):
        # Lifecycle timestamps retain their original owner's JSON spelling.
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")

    def require_current(self):
        NativeRepositoryWorldPrelaunchAssignment._retained(self, issued=True)
        with self._lock:
            _need(not self._checking, "prelaunch_gate_reentry_refused")
            self._checking = True
            try:
                encoded, _rows, _witnesses = NativeRepositoryWorldPrelaunchAssignment._observe(self, issued=True)
                return json.loads(encoded)
            finally:
                self._checking = False

    def require_for_broker(self, *, server, canonical_repository, workspace, prompt,
                           task_admission, worktree_admission):
        NativeRepositoryWorldPrelaunchAssignment._retained(self, issued=True)
        _need(server is self._server and task_admission is self._task
              and worktree_admission is self._allocation
              and isinstance(canonical_repository, Path) and canonical_repository == self._canonical
              and isinstance(workspace, Path) and workspace == self._workspace
              and _prompt_binding(prompt) == self._prompt,
              "prelaunch_broker_assignment_mismatch")
        return NativeRepositoryWorldPrelaunchAssignment.require_current(self)


def capture_repository_world_prelaunch_assignment(*, task_admission, worktree_admission, prompt):
    """Capture original live assignment objects without dispatching or mutating source."""
    gate = NativeRepositoryWorldPrelaunchAssignment(_SEAL, task_admission=task_admission,
        worktree_admission=worktree_admission, prompt=prompt)
    encoded, rows, witnesses = NativeRepositoryWorldPrelaunchAssignment._observe(gate, issued=False)
    gate._observation, gate._source_rows, gate._source_witnesses = encoded, rows, witnesses
    with _ISSUANCE_LOCK:
        _ISSUED[gate] = NativeRepositoryWorldPrelaunchAssignment._seal_material(gate)
    NativeRepositoryWorldPrelaunchAssignment.require_current(gate)
    return gate
