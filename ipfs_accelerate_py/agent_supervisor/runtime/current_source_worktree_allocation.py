"""Native linked-worktree custody for one existing source task admission.

Host policy supplies the repository, roots and Portal board identity. This
component reads the genuine sealed projection and current lifecycle/index; it
does not bootstrap a daemon, allocate anything, compile context or admit a
coding process. Native claim fencing retains its normal bookkeeping writes.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ..merge import worktree_lifecycle as lifecycle
from ..task_sources.control_plane_contracts import canonical_json_bytes
from .current_source_lookup import CurrentSourceLookup, _SourceBindingAttempt
from .current_source_task_admission import NativeSourceTaskAdmission

SCHEMA = "supervisor-native-source-worktree-allocation@1"
NOMINATED_ALLOCATION_SCHEMA = "supervisor-native-nominated-source-worktree-allocation@1"
_SEAL = object()
_BOUND = 262_144


class CurrentSourceWorktreeAllocationError(ValueError):
    """The retained native allocation no longer matches its admitted task."""


def _need(value, reason):
    if not value:
        raise CurrentSourceWorktreeAllocationError(reason)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "allocation metadata contains duplicate keys")
        result[key] = value
    return result


def _json(raw):
    value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_pairs,
                       parse_constant=lambda _value: (_ for _ in ()).throw(
                           CurrentSourceWorktreeAllocationError("nonfinite allocation metadata")))
    _need(type(value) is dict, "allocation metadata must be an object")
    _wire(value)
    return value


def _directory(path):
    _need(isinstance(path, Path) and path.is_absolute() and path.resolve(strict=True) == path,
          "allocation directory must be a canonical existing path")
    info = path.lstat()
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
          and not info.st_mode & 0o022, "allocation directory must remain owner controlled")
    return (info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode))


def _read(path, bound=_BOUND):
    _need(path.is_absolute() and path.resolve(strict=True) == path,
          "allocation control path is not canonical")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
              and before.st_nlink == 1 and not before.st_mode & 0o022
              and 0 <= before.st_size <= bound, "allocation requires a bounded owner regular file")
        raw = stream.read(bound + 1)
        after, named = os.fstat(stream.fileno()), path.lstat()
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
                info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    _need(len(raw) == before.st_size <= bound
          and identity(before) == identity(after) == identity(named)
          and path.resolve(strict=True) == path, "allocation control changed during its read")
    return (identity(named), hashlib.sha256(raw).hexdigest()), raw


def _line(raw):
    text = raw.decode("utf-8", "strict")
    _need(text.endswith("\n") and "\n" not in text[:-1]
          and not any(c in text for c in "\0\r"), "allocation Git control must be one line")
    return text[:-1]


def _branch(value):
    _need(type(value) is str and 0 < len(value.encode()) <= 4096
          and not value.startswith(("/", ".", "-")) and not value.endswith(("/", "."))
          and not any(c in value for c in " ~^:?*[\\\0\r\n")
          and not any(c in value for c in ("..", "//", "@{"))
          and all(not p.startswith(".") and not p.endswith(".lock") for p in value.split("/")),
          "allocation branch name is invalid")
    return value


class NativeSourceWorktreeAllocation:
    """Exact owner-issued source allocation handle; observations are inert."""

    def __init__(self, seal, *, task_admission, canonical_repository, workspace,
                 worktree_root, attempt_root, merge_target_branch, board_namespace):
        _need(seal is _SEAL and type(self) is NativeSourceWorktreeAllocation
              and type(task_admission) is NativeSourceTaskAdmission,
              "exact native source task admission required")
        task = NativeSourceTaskAdmission.require_current(task_admission)
        self._seal, self._issued_self = seal, self
        self._task_admission, self._server = task_admission, task_admission._server
        self._canonical_repository, self._workspace = Path(canonical_repository), Path(workspace)
        self._worktree_root, self._attempt_root = Path(worktree_root), Path(attempt_root)
        self._merge_target, self._board_namespace = _branch(merge_target_branch), board_namespace
        _need(type(board_namespace) is str and 0 < len(board_namespace.encode()) <= 4096
              and board_namespace == board_namespace.strip()
              and not any(c in board_namespace for c in "\0\r\n"), "host-admitted board namespace required")
        _need(self._workspace != self._worktree_root
              and self._workspace.is_relative_to(self._worktree_root)
              and self._workspace != self._canonical_repository,
              "allocation must be a distinct workspace within its host-admitted root")
        self._git = self._canonical_repository / ".git"
        self._store_path = self._git / lifecycle.WORKTREE_LIFECYCLE_DIRNAME
        self._attempt_directory = self._attempt_root / hashlib.sha256(
            task_admission._claim["attempt_id"].encode()).hexdigest()[:24]
        self._directories = tuple((path, _directory(path)) for path in dict.fromkeys((
            self._canonical_repository, self._workspace, self._worktree_root,
            self._attempt_root, self._attempt_directory, self._git, self._git / "worktrees",
            self._store_path)))
        # Explicit existing store_dir avoids Git discovery and creates no path.
        self._store = lifecycle.WorktreeLifecycleStore(
            self._canonical_repository, store_dir=self._store_path)
        from ..task_sources.intent_repository import IntentRepository
        from ..task_sources.database_task_source import DatabaseTaskSource
        from ..todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
        self._source = DatabaseTaskSource(intent=IntentRepository(
            bound_connection=task_admission._connection, install_schema=False), install_schema=False)
        def no_provider(*_args, **_kwargs):
            raise CurrentSourceWorktreeAllocationError("allocation custody cannot invoke a provider")
        self._bridge = DatabasePortalExecutionBridge(task_source=self._source,
            attempt_root=self._attempt_root, portal_factory=no_provider,
            repo_root=self._canonical_repository, board_namespace=board_namespace,
            merge_target_branch=self._merge_target, task_header_prefix="## " + task["task_alias"])
        self._intent = self._source.intent
        self._original_policy = (self._directories, self._git, self._store_path, self._attempt_directory,
                                 self._intent, self._bridge.task_header_prefix, self._bridge.portal_factory)
        self._task_bytes = canonical_json_bytes(task)
        self._fixed_files = self._lifecycle_bytes = self._renewal = self._material = None
        self._original_custody = None
        self._lookup = self._lookup_refs = self._lookup_bytes = None
        self._nominated_source_scope = self._original_nominated_source_scope = None
        self._nominated_scope_checking = False
        self._binding = self._checking = False
        self._owner_callback_active = self._owner_callback_reentry_detected = False
        self._lock = threading.RLock()
        self._original = (self, task_admission, self._server, self._source, self._bridge, self._store,
            self._canonical_repository, self._workspace, self._worktree_root, self._attempt_root,
            self._merge_target, self._board_namespace, self._task_bytes)
        self.require_current()

    def _retained(self):
        current = (
                  self, self._task_admission, self._server, self._source, self._bridge, self._store,
                  self._canonical_repository, self._workspace, self._worktree_root, self._attempt_root,
                  self._merge_target, self._board_namespace, self._task_bytes)
        _need(type(self) is NativeSourceWorktreeAllocation and self._seal is _SEAL
              and self._issued_self is self and len(current) == len(self._original)
              and all(a is b if n < 6 else a == b for n, (a, b) in enumerate(zip(current, self._original)))
              and self._server is self._task_admission._server
              and self._source.intent is self._intent
              and self._intent._bound_connection is self._task_admission._connection
              and self._directories is self._original_policy[0]
              and (self._git, self._store_path, self._attempt_directory) == self._original_policy[1:4]
              and self._intent is self._original_policy[4]
              and self._store.repo_root == self._canonical_repository
              and self._store.store_dir == self._store_path
              and self._bridge.task_source is self._source
              and self._bridge.attempt_root == self._attempt_root
              and self._bridge.repo_root == self._canonical_repository
              and self._bridge.board_namespace == self._board_namespace
              and self._bridge.merge_target_branch == self._merge_target
              and self._bridge.task_header_prefix == self._original_policy[5]
              and self._bridge.portal_factory is self._original_policy[6]
              and (self._original_custody is None or all(a is b for a, b in zip(
                  self._original_custody, (self._fixed_files, self._lifecycle_bytes, self._material))))
              and all(_directory(path) == identity for path, identity in self._directories),
              "native source allocation original objects or policy changed")
        _need(self._nominated_source_scope is self._original_nominated_source_scope,
              "native allocation original nomination join changed")
        self._task_admission._retained()

    def _observe(self, *, _checked_at=None):
        from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, DatabaseTaskAttempt
        from ..todo_daemon.database_portal_bridge import DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
        self._retained()
        admission = self._task_admission
        task = self._source.get_task(admission._claim["task_cid"])
        _need(task is not None and task.status == "in_progress"
              and task.revision == json.loads(self._task_bytes)["task_revision"], "allocation native task changed")
        receipt = task.body["completion_receipt"]
        attempt = DatabaseTaskAttempt(task_cid=task.task_cid, task_alias=task.task_alias,
            **{key: admission._claim[key] for key in admission._claim if key != "task_cid"},
            committed_phase=receipt["attempt_execution_phase"], status="running", started_at_ms=0,
            revision=receipt["attempt_execution_revision"], body={})
        attempt = replace(attempt, body={"control_binding":
            DatabaseImplementationDaemon._control_claim_binding(attempt, task)})
        paths = self._bridge._paths(attempt)
        _need(paths.root == self._attempt_directory, "allocation Portal path changed")
        self._bridge._seal_attempt_directory(paths, attempt_id=attempt.attempt_id, create=False)
        binding_witness, binding_raw = _read(paths.binding)
        projection_witness, projection_raw = _read(paths.task_projection)
        binding = self._bridge._strict_binding(paths.binding)
        _need(binding == _json(binding_raw) and binding["schema"] == DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
              and binding == self._bridge._binding(attempt, task, self._bridge._render_projection(attempt, task),
                                                   schema=DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA),
              "allocation Portal binding differs from its current native claim")
        identity = self._bridge._prior_projection_identity(paths, binding)
        _need(identity["task_id"] == task.task_alias, "allocation Portal task alias differs")
        lifecycle_path = self._store.workspace_path_for(self._workspace)
        lifecycle_witness, lifecycle_raw = _read(lifecycle_path)
        payload = _json(lifecycle_raw)
        record = lifecycle.WorkspaceLifecycleRecord.from_dict(payload)
        _need(_wire(payload) == _wire(record.to_dict()) and record.record_id == record.compute_record_id(),
              "allocation lifecycle record is not the exact native schema")
        prefix = (1 << 52) | (attempt.attempt_number << 16)
        local_attempt = record.attempt - prefix
        lane = re.fullmatch(re.escape(str(paths.root)) + r":(\d+)/(\d+):" + str(admission._peer[0]), record.lane_id)
        _need(record.state is lifecycle.WorkspaceLifecycleState.ACTIVE
              and record.task_id == task.task_alias and record.canonical_task_cid == identity["canonical_task_cid"]
              and 1 <= local_attempt < (1 << 16) and record.attempt == prefix | local_attempt
              and record.workspace_path == str(self._workspace) and record.repo_root == str(self._canonical_repository)
              and record.state_dir == str(paths.root) and record.merge_target == self._merge_target
              and lane is not None and 0 <= int(lane[1]) < int(lane[2])
              and not record.terminal_reason, "allocation lifecycle task, attempt or workspace differs")
        birth = lifecycle.read_process_birth(admission._peer[0])
        _need(birth is not None and birth == record.owner and birth.start_time_ticks == admission._peer[2]
              and os.stat("/proc/" + str(birth.pid)).st_uid == admission._peer[1]
              and record.created_at <= record.updated_at < record.expires_at
              and record.expires_at > (time.time() if _checked_at is None else _checked_at),
              "allocation lifecycle owner or live lease differs")
        index_path = self._store.task_index_path_for(canonical_task_cid=record.canonical_task_cid,
            task_id=record.task_id, attempt=record.attempt)
        index_witness, index_raw = _read(index_path)
        _need(_wire(_json(index_raw)) == _wire(self._store._task_index_payload(record)),
              "allocation lifecycle task index differs")
        marker_witness, marker_raw = _read(self._workspace / ".git", 4096)
        marker = _line(marker_raw)
        _need(marker.startswith("gitdir: "), "allocation is not a linked Git worktree")
        registered = Path(marker.removeprefix("gitdir: "))
        _need(registered.parent == self._git / "worktrees", "allocation Git marker names another repository")
        registered_identity = _directory(registered)
        fixed = [(paths.binding, binding_witness), (paths.task_projection, projection_witness),
                 (self._workspace / ".git", marker_witness)]
        expected_lines = {"gitdir": str(self._workspace / ".git"), "commondir": "../..",
                          "HEAD": "ref: refs/heads/" + _branch(record.branch)}
        for name, expected in expected_lines.items():
            path = registered / name
            witness, raw = _read(path, 4096)
            _need(_line(raw) == expected, "allocation registered Git control differs: " + name)
            fixed.append((path, witness))
        fixed = (tuple(fixed), registered, registered_identity, canonical_json_bytes(identity),
                 canonical_json_bytes(binding))
        # Native helpers above perform their own reads. Recheck every exact
        # bounded control afterward, without invoking a callback after them.
        for path, expected in fixed[0]:
            _need(_read(path)[0] == expected, "allocation immutable control changed during observation")
        _need(_read(lifecycle_path)[0] == lifecycle_witness and _read(index_path)[0] == index_witness,
              "allocation lifecycle/index changed during observation")
        self._retained()
        return fixed, record, identity, binding

    def require_current(self):
        """Fence COOR before SERVER, then close with detached allocation reads."""
        with self._lock:
            self._refuse_owner_callback_reentry()
            _need(not self._checking, "native source allocation reentry refused")
            self._checking = True
            try:
                self._retained()
                admission = self._task_admission
                _need(canonical_json_bytes(NativeSourceTaskAdmission.require_current(admission)) == self._task_bytes,
                      "allocation task snapshot changed")
                def checked(_lease):
                    with self._server._lock:
                        before, _grant = admission._native()
                        observed = self._observe()
                        after, _current = admission._native()
                        _need(before == after, "allocation native claim changed during observation")
                        return observed
                observed = admission._coordinator.execute_with_task_claim_fence(admission._claim, checked)
                with self._server._lock:
                    admission._native()
                    final = self._observe()
                    _need(final == observed, "allocation changed after its native claim fence")
                return self._retain_observation(final)
            finally:
                self._checking = False

    def _retain_observation(self, observed):
        """Share the original immutable custody and monotonic renewal checks."""
        fixed, record, identity, binding = observed
        stable = record.to_dict()
        renewal = tuple(stable.pop(key) for key in ("fence", "updated_at", "expires_at"))
        _need(self._fixed_files is None or fixed == self._fixed_files,
              "allocation immutable custody changed")
        _need(self._lifecycle_bytes is None or _wire(stable) == self._lifecycle_bytes,
              "allocation lifecycle ownership changed")
        _need(self._renewal is None or all(a >= b for a, b in zip(renewal, self._renewal)),
              "allocation lifecycle renewal rolled back")
        if self._material is None:
            task = json.loads(self._task_bytes)
            material = dict(schema=SCHEMA, **{key: task[key] for key in (
                "task_cid", "task_alias", "task_revision", "expected_task_body_cid")},
                native_attempt_number=self._task_admission._claim["attempt_number"],
                portal_canonical_task_cid=identity["canonical_task_cid"], portal_binding_id=binding["binding_id"],
                lifecycle_record_id=record.record_id, lifecycle_attempt=record.attempt,
                lifecycle_initial_fence=record.fence,
                lifecycle_lease_identity_cid=cid_for_structured({"lease_id": record.lease_id}),
                lifecycle_owner_birth=record.owner.to_dict(), canonical_repository=str(self._canonical_repository),
                workspace=str(self._workspace), branch=record.branch, merge_target_branch=self._merge_target,
                allocation_ownership_verified=True, proof_authority=False, execution_authority=False,
                completion_authority=False, bootstrap_verified=False, compiled_context_verified=False,
                coding_process_verified=False)
            material["allocation_cid"] = cid_for_structured(material)
            self._material = canonical_json_bytes(material)
            self._fixed_files, self._lifecycle_bytes = fixed, _wire(stable)
            self._original_custody = (self._fixed_files, self._lifecycle_bytes, self._material)
        self._renewal = renewal
        return json.loads(self._material)

    def _refuse_owner_callback_reentry(self):
        if self._owner_callback_active:
            self._owner_callback_reentry_detected = True
            raise CurrentSourceWorktreeAllocationError("native allocation owner callback reentry refused")

    def _observe_callback_boundary(self):
        """Under COOR then SERVER, read task/grant and allocation controls only."""
        self._retained()
        admission = self._task_admission
        admission._require_inactive_owner_transaction()
        before, grant = admission._native()
        _need(before[0] == self._task_bytes, "allocation task snapshot changed")
        observed = self._observe()
        after, current = admission._native()
        _need(before == after and after[0] == self._task_bytes
              and grant.grant_id == current.grant_id,
              "allocation native claim or grant changed during owner callback boundary")
        self._retain_observation(observed)
        _need(not self._owner_callback_reentry_detected
              and not admission._owner_callback_reentry_detected,
              "native allocation owner callback attempted reentry")
        admission._require_inactive_owner_transaction()
        self._retained()

    def execute_with_current_claim(self, callback, *, minimum_remaining_ms=0):
        """Run a trusted owner callback under the existing live task fence.

        The callback receives the genuine immutable protected lease. Allocation
        and admission locks, then COOR and SERVER, span its execution. Native
        task/grant and allocation controls are checked before and after; source
        contents are not read. This method grants no launch, retry or completion
        authority. A failure after callback entry cannot undo external effects:
        the caller retains its child cleanup and possible-effects obligations.
        """
        _need(callable(callback), "trusted owner callback required")
        with self._lock:
            self._refuse_owner_callback_reentry()
            _need(not self._checking, "native source allocation reentry refused")
            self._retained()
            admission = self._task_admission
            with admission._lock:
                admission._refuse_owner_callback_reentry()
                _need(not admission._checking, "native source task admission reentry refused")
                # Check before acquiring COOR and again under SERVER, so an
                # unrelated borrowed SQL transaction never reaches the callback.
                admission._require_inactive_owner_transaction()
                self._checking = admission._checking = True
                self._owner_callback_active = admission._owner_callback_active = True
                self._owner_callback_reentry_detected = admission._owner_callback_reentry_detected = False
                try:
                    def checked(lease):
                        with self._server._lock:
                            self._observe_callback_boundary()
                            callback_error = None
                            result = None
                            try:
                                result = callback(lease)
                            except BaseException as error:
                                callback_error = error
                            try:
                                self._observe_callback_boundary()
                            except BaseException as closing_error:
                                if callback_error is not None:
                                    raise callback_error from closing_error
                                raise
                            if callback_error is not None:
                                raise callback_error
                            return result
                    result = admission._coordinator.execute_with_task_claim_fence(
                        admission._claim, checked, minimum_remaining_ms=minimum_remaining_ms)
                    # The coordinator's own closing clock/commit can invoke
                    # owner code. Close again after that fence without entering
                    # another claim transaction or reading candidate contents.
                    with self._server._lock:
                        self._observe_callback_boundary()
                    return result
                finally:
                    self._checking = admission._checking = False
                    self._owner_callback_active = admission._owner_callback_active = False
                    self._owner_callback_reentry_detected = admission._owner_callback_reentry_detected = False

    def bind_current_source_lookup(self, *, cas, expected_inventory_cid, request, limits,
                                   context_cid, allowed_paths, max_response_bytes=16384, max_slice_bytes=4096):
        return self._bind_source_lookup(cas=cas, arguments=dict(
            expected_inventory_cid=expected_inventory_cid, request=request, limits=limits,
            context_cid=context_cid, allowed_paths=allowed_paths,
            max_response_bytes=max_response_bytes, max_slice_bytes=max_slice_bytes))

    def bind_nominated_current_source_lookup(self, *, scope, expected_scope_cid,
                                             cas, expected_inventory_cid, request, limits,
                                             context_cid, max_response_bytes=16384, max_slice_bytes=4096):
        """Join the pinned historical nomination to this actual native allocation.

        The task binder independently replays the complete retained inventory.
        The allocation supplies its retained workspace; no caller path list,
        serialized allocation observation, or inferred worker authority is used.
        """
        return self._bind_source_lookup(cas=cas, arguments=dict(
            scope=scope, expected_scope_cid=expected_scope_cid,
            expected_inventory_cid=expected_inventory_cid, request=request, limits=limits,
            context_cid=context_cid, max_response_bytes=max_response_bytes,
            max_slice_bytes=max_slice_bytes), nomination=(scope, expected_scope_cid))

    def _bind_source_lookup(self, *, cas, arguments, nomination=None):
        """Serialize allocation custody and retain only our literal publication."""
        with self._lock:
            _need(not self._binding and self._lookup is None, "allocation source binding is already consumed")
            self._binding = True
            attempt = _SourceBindingAttempt()
            try:
                with self._server._lock:
                    self._retained()
                    self._task_admission._require_source_binding_transaction()
                self.require_current()

                def unlocked():
                    self.require_current()

                def locked(checked_at_ms):
                    self._retained()
                    fixed, record, _identity, _binding = self._observe(_checked_at=checked_at_ms / 1000)
                    stable = record.to_dict()
                    renewal = tuple(stable.pop(key) for key in ("fence", "updated_at", "expires_at"))
                    _need(fixed == self._fixed_files and _wire(stable) == self._lifecycle_bytes
                          and all(a >= b for a, b in zip(renewal, self._renewal)),
                          "allocation custody changed after source binding fences")

                attempt._add_checks(unlocked, locked)
                with self._task_admission._lock:
                    with self._server._lock:
                        _need(not self._task_admission._binding_in_progress
                              and self._task_admission._source is None
                              and self._task_admission._gateway._current_source_lookup is None,
                              "allocation requires an unbound native source reader")
                    try:
                        binder = (self._task_admission.bind_current_source_lookup if nomination is None
                                  else self._task_admission.bind_nominated_current_source_lookup)
                        result = binder(repository=self._workspace, cas=cas,
                            _binding_attempt=attempt, **arguments)
                    finally:
                        publication = attempt._published
                        if publication is not None:
                            _need(publication[0] is attempt
                                  and publication[1] is self._task_admission._gateway
                                  and publication[1]._current_source_lookup_publication is publication,
                                  "original allocation source publication witness changed")
                            service, body, original_cas, root = publication[4:]
                            self._lookup, self._lookup_refs = service, (service, original_cas, root)
                            self._lookup_bytes = body
                            service._native_source_worktree_allocation = self
                            if nomination is not None:
                                captured = (*nomination, self._task_admission._nominated_source_scope,
                                            service, body, original_cas)
                                self._nominated_source_scope = self._original_nominated_source_scope = captured
                self.require_current()
                with self._server._lock:
                    self.require_source_binding(server=self._server, service=self._lookup)
                return result
            finally:
                self._binding = False

    def _require_nominated_join(self):
        """Check original objects only; do not enter a native claim fence."""
        self._retained()
        retained = self._nominated_source_scope
        _need(type(retained) is tuple and len(retained) == 6
              and retained is self._original_nominated_source_scope,
              "native allocation reader has no retained nomination join")
        scope, expected, task_join, source, binding_bytes, cas = retained
        admission = self._task_admission
        _need(type(task_join) is tuple and len(task_join) == 7
              and task_join is admission._nominated_source_scope
              and task_join is admission._original_nominated_source_scope
              and task_join[0] is scope and task_join[2] == expected
              and task_join[4] is source and task_join[5] == binding_bytes
              and task_join[6] is cas
              and source is self._lookup and source is admission._source
              and type(source) is CurrentSourceLookup and source._store.cas is cas
              and binding_bytes == self._lookup_bytes == admission._source_bytes,
              "native allocation original nominated reader changed")
        value = scope.observation()
        _need(canonical_json_bytes(value) == task_join[1] and value["scope_cid"] == expected
              and task_join[3] == tuple(row["raw_path_hex"] for row in value["selected"])
              and source._paths == task_join[3],
              "native allocation nomination or selected path scope changed")
        return value

    def require_nominated_source_scope(self):
        """Observe the live nomination/allocation join; return no usable grant."""
        with self._lock:
            _need(not self._binding, "allocation nomination binding is still in progress")
            _need(not self._nominated_scope_checking, "allocation nomination observation reentry refused")
            self._nominated_scope_checking = True
            try:
                retained = self._nominated_source_scope
                self._require_nominated_join()
                allocation = self.require_current()
                nominated = self._task_admission.require_nominated_source_scope()
                closing = self.require_current()
                _need(allocation == closing and self._nominated_source_scope is retained,
                      "nominated allocation changed during native observation")
                # Both observers above acquire COOR before SERVER. Everything
                # below is a detached closing check under SERVER only, after
                # their coordinator clocks and retained-CAS callbacks finish.
                with self._server._lock:
                    self._task_admission._native()
                    self.require_source_binding(server=self._server, service=self._lookup)
                    fixed, record, _identity, _binding = self._observe()
                    stable = record.to_dict()
                    renewal = tuple(stable.pop(key) for key in ("fence", "updated_at", "expires_at"))
                    _need(fixed == self._fixed_files and _wire(stable) == self._lifecycle_bytes
                          and all(a >= b for a, b in zip(renewal, self._renewal)),
                          "allocation custody changed after nominated native fences")
                    value = self._require_nominated_join()
                    binding = self._lookup.binding(self._task_admission._gateway.identity)
                    _need(all(nominated[key] == binding[key] == allocation[key] for key in
                              ("task_cid", "task_revision", "expected_task_body_cid"))
                          and nominated["source_scope_cid"] == value["scope_cid"]
                          and nominated["frontier_cid"] == value["frontier_cid"]
                          and nominated["source_inventory_cid"] == binding["source_inventory_cid"]
                          and nominated["source_binding_cid"] == binding["binding_cid"]
                          and nominated["context_cid"] == binding["context_cid"],
                          "nomination, source binding and allocation identities differ")
                    # Check the complete working population after the native
                    # callbacks, including originals outside the selected set.
                    self._lookup._check_source()
                    self.require_source_binding(server=self._server, service=self._lookup)
                    self._require_nominated_join()
                    self._renewal = renewal
                return {**nominated, "schema": NOMINATED_ALLOCATION_SCHEMA,
                        "allocation_cid": allocation["allocation_cid"],
                        "allocation_ownership_verified": True, "bootstrap_verified": False}
            finally:
                self._nominated_scope_checking = False

    def require_source_binding(self, *, server, service):
        """Source join only; never acquire COOR while the caller holds SERVER."""
        self._retained()
        self._task_admission.require_source_binding(server=server, service=service)
        _need(server is self._server and service is self._lookup
              and self._lookup_refs is not None and service is self._lookup_refs[0]
              and service._store.cas is self._lookup_refs[1] and service._root == self._lookup_refs[2] == self._workspace
              and getattr(service, "_native_source_worktree_allocation", None) is self
              and canonical_json_bytes(service.binding(self._task_admission._gateway.identity)) == self._lookup_bytes,
              "native allocation source binding changed")


def capture_native_source_worktree_allocation(*, task_admission, canonical_repository, workspace,
                                             worktree_root, attempt_root, merge_target_branch, board_namespace):
    return NativeSourceWorktreeAllocation(_SEAL, task_admission=task_admission,
        canonical_repository=canonical_repository, workspace=workspace, worktree_root=worktree_root,
        attempt_root=attempt_root, merge_target_branch=merge_target_branch, board_namespace=board_namespace)
