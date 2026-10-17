"""Actual original cap2 host refusal after separately qualified bootstrap.

Source-only component, with no CLI or native admission. The enclosing reviewed
same-PID owner must create the actual daemon under its genuine bound admin
executor, then stop/drain/release admin before calling this case. No methods,
classes, modules, connection or native lease are replaced. SELECT-only borrow
witnesses and native renewals are permitted; SQL writes are forbidden.
"""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys

DAEMON_REL = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
TODO = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.'
NATIVE = 'ipfs_datasets_py.optimizers.logic_theorem_optimizer.'
_RETAINED_CASES = {}


class FixtureCustodyPending(BaseException):
    """Original caller must keep its native frames entered; never serialize."""
    def __init__(self, connection, host, phase, owner, *, raw_connection, lock_context):
        super().__init__('original_fixture_connection_and_native_frames_retained')
        self.connection, self.host, self.phase, self.owner = connection, host, phase, owner
        self.raw_connection, self.lock_context = raw_connection, lock_context
        self.current_raw_connection = getattr(connection, '_connection', None)
        _RETAINED_CASES[id(self)] = self

    def __reduce__(self):
        raise TypeError('actual_fixture_custody_not_serializable')


def need(value, reason):
    if not value:
        raise AssertionError(reason)


def checked(path, digest):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical_source_required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_uid in {0, os.getuid()} and before.st_mode & 2 == 0
             and 0 < before.st_size <= 256 * 1024**2, 'bounded_owned_source_required')
        hasher, count = hashlib.sha256(), 0
        while chunk := stream.read(1024**2):
            count += len(chunk)
            need(count <= before.st_size, 'source_grew')
            hasher.update(chunk)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid,
                         m.st_gid, m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(count == before.st_size and witness(before) == witness(after) == witness(path.lstat())
         and hasher.hexdigest() == digest, 'source_changed')


def actual_module(name, pins):
    module = sys.modules.get(name)
    need(module is not None and module.__name__ == name, 'actual_full_loaded_module_required')
    path = Path(module.__file__).absolute()
    need(str(path) in pins and path.as_posix().endswith('/' + name.replace('.', '/') + '.py'),
         'exact_loaded_module_origin_required')
    checked(path, pins[str(path)])
    return module


def run_cap_two_case(*, daemon, original_phase, fixture_owner, fixture_root,
                     protected_source_sha256, custody_source_sha256,
                     reviewed_daemon_sha256):
    """Use genuine original producers; caller retains all actual native frames.

    This intentionally has no CLI, scheduler/root constructor, process launch,
    repository creation, capsule installation or publication implementation.
    Its enclosing reviewed qualifier must supply and retain the actual original
    resource objects, charge all storage before creation, independently qualify
    constructor custody and hold admission through caller-owned connection close.
    """
    pins = dict(protected_source_sha256)
    need(0 < len(pins) <= 128 and sys.dont_write_bytecode, 'bounded_pinned_no_bytecode_case_required')
    for path, digest in pins.items():
        checked(path, digest)
    daemon_module = actual_module(TODO + 'implementation_daemon', pins)
    host_module = actual_module(TODO + 'report_only_native_host', pins)
    control = actual_module(TODO + 'portal_task_state_control_plane', pins)
    state_module = actual_module('ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state', pins)
    native = actual_module(NATIVE + 'resource_scheduler', pins)
    parented = actual_module(NATIVE + 'parented_phase_resources', pins)
    daemon_source = str(Path(daemon_module.__file__).absolute())
    need(type(reviewed_daemon_sha256) is str and len(reviewed_daemon_sha256) == 64
         and all(c in '0123456789abcdef' for c in reviewed_daemon_sha256)
         and pins[daemon_source] == reviewed_daemon_sha256 == host_module.DAEMON_SHA,
         'exact_reviewed_composed_original_daemon_required')
    need(type(original_phase) is native.ResourceLease and original_phase.owner_pid == os.getpid()
         and original_phase.parent_lease_id is not None and original_phase.cpu_slots == 3
         and original_phase.memory_mb == 4096 and original_phase.child_process_slots == 16
         and original_phase.renew() and not original_phase.cancelled and not original_phase.released,
         'actual_same_PID_original_phase_required')
    need(type(fixture_owner) is parented.ParentedPhaseResourceReservation,
         'actual_retained_fixture_owner_required')
    fixture_lease = fixture_owner.native_lease
    need(type(fixture_lease) is native.ResourceLease
         and fixture_lease.owner_pid == os.getpid()
         and fixture_lease.parent_lease_id == original_phase.lease_id
         and fixture_lease.cpu_slots == 1 and fixture_lease.memory_mb == 3072
         and fixture_lease.child_process_slots == 8 and fixture_lease.renew(),
         'actual_finite_fixture_data_allocation_required')
    root = Path(fixture_root)
    owner_root = Path(fixture_owner.directory)
    need(root.is_absolute() and owner_root in root.parents and root.is_dir()
         and not any(p.is_symlink() for p in (root, *root.parents)), 'charged_private_fixture_root_required')
    need(root.stat().st_uid == os.getuid() and root.stat().st_mode & 0o077 == 0,
         'private_fixture_permissions_required')
    # The enclosing preparation must create this local repository through its
    # separately admitted real Git mechanism, without alternates or hardlinks.
    repo = root / 'repository'
    need((repo / '.git').is_dir() and not (repo / '.git').is_symlink()
         and not (repo / '.git/objects/info/alternates').exists(), 'real_private_self_contained_repository_required')
    need(os.environ.get('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR') == str(root / 'orchestration')
         and os.environ.get('IPFS_ACCELERATE_AGENT_RECLAIM_DEAD_WORKTREE_LEASES_ON_STARTUP') == 'false',
         'explicit_private_orchestration_and_no_restart_reclaim_required')
    need(not any(k.startswith('IPFS_ACCELERATE_AGENT_QUACK_') or k in {
        'IPFS_ACCELERATE_AGENT_CONTROL_DUCKDB', 'IPFS_ACCELERATE_AGENT_CONTROL_DUCKLAKE'} for k in os.environ),
        'no_ambient_database_route_required')
    lane = root / 'lane'
    need(lane.is_dir() and not lane.is_symlink(), 'original_constructor_lane_required')
    state_path, target = lane / 'state.json', lane / 'state.duckdb'
    need(daemon.state_path == state_path and daemon.repo_root == repo, 'actual_private_constructor_paths_required')
    # The enclosing source-pinned worker supplied this exact original object
    # from its completed admitted import/constructor scope. This component
    # does not reconstruct or replace that object, and does not attest bootstrap.
    need(type(daemon) is daemon_module.PortalImplementationDaemon and daemon.max_task_attempts == 2,
         'actual_full_original_constructor_cap_two_required')
    connection = state_module.open_duckdb_connection(target, prefer_quack=False, memory_limit='64MB', threads=1)
    need(type(connection) is state_module.DuckDBConnection, 'actual_original_connection_producer_required')
    caller_raw, caller_lock = connection._connection, connection._lock_context
    host = None
    try:
        connection.execute(control.DDL)
        connection.execute(control.HANDOFF_DDL)
        initial = daemon_module.PortalTaskState(
            active_attempt=2, implementation_attempts={'fixture-only': 2}, completed_task_ids=[])
        # Setup is a private engineering fixture, outside the measured refusal.
        connection.execute('INSERT INTO portal_task_state VALUES (?, ?, ?, FALSE, ?)',
            [state_path.stem, daemon_module.json.dumps(asdict(initial), sort_keys=True), 1, 'fixture-baseline'])
        control.bind_task_state_control_plane(state_path, str(target))
        before = connection.execute('SELECT * FROM portal_task_state ORDER BY lane_key').fetchall()
        raw, epoch = connection._connection, connection._reconnect_epoch
        host = host_module.ReportOnlyNativeHost(daemon, parent_phase=original_phase,
            control_plane_connection=connection, control_plane_target=target,
            attempt_root=root / 'never-started-host', protected_source_sha256=pins,
            custody_source_sha256=custody_source_sha256)
        need(sys.getprofile() is None, 'unoccupied_same_thread_profiler_required')
        forbidden = {daemon_module.PortalImplementationDaemon._run_once.__code__,
            daemon_module.PortalImplementationDaemon._retained_provider_capacity_deferral.__code__,
            daemon_module.PortalImplementationDaemon._mark_implementation_started.__code__,
            control._connect.__code__, control._project_ducklake.__code__,
            state_module.open_duckdb_connection.__code__, state_module.DuckDBConnection.close.__code__,
            state_module.DuckDBConnection.reconnect_exclusive_owner.__code__,
            subprocess.Popen.__init__.__code__}
        identities, readonly_statements, guard_calls = [], [], []

        def observe(frame, event, value):
            if event == 'call' and frame.f_code in forbidden:
                raise AssertionError('cap_two_entered_forbidden_original_effect')
            if event == 'call' and frame.f_code is host_module.ReportOnlyNativeHost.guard_pass_entry.__code__:
                need(frame.f_locals['daemon'] is daemon and daemon.max_task_attempts == 2,
                     'original_daemon_guard_identity_changed')
                guard_calls.append(True)
            if event == 'return' and frame.f_code is control.require_borrowed_task_state_connection.__code__:
                need(value is connection, 'borrowed_original_connection_identity_changed')
                identities.append(True)
            if event == 'call' and frame.f_code is state_module.DuckDBConnection.execute.__code__:
                sql = ' '.join(str(frame.f_locals['sql']).strip().upper().split())
                need(sql.startswith('SELECT '), 'cap_two_executed_productive_SQL')
                readonly_statements.append(sql)

        observed = None
        sys.setprofile(observe)
        try:
            host.execute_daemon_once()
        except host_module.NativeHostRefusal as error:
            observed = error
        finally:
            sys.setprofile(None)
        need(type(observed) is host_module.NativeHostRefusal
             and str(observed) == 'explicit_unchanged_report_only_profile_required_before_original_pass',
             'exact_original_cap_two_refusal_required')
        need(len(guard_calls) == 1 and identities and readonly_statements,
             'actual_guard_and_same_borrow_identity_observations_required')
        need(host.finished and not host.bound and host.pending is None and daemon._report_only_native_host is None
             and not host.owners and not host.consumers and host.hold is None,
             'no_resource_host_refusal_must_finish_original_borrow')
        need(not control.task_state_connection_is_borrowed(state_path), 'refused_host_left_borrow_registered')
        need(connection._connection is raw and connection._reconnect_epoch == epoch and not connection._closed,
             'caller_connection_replaced_or_closed')
        need(connection.execute('SELECT * FROM portal_task_state ORDER BY lane_key').fetchall() == before
             and daemon.max_task_attempts == 2 and not (root / 'never-started-host').exists(),
             'cap_two_changed_original_counter_row_cap_or_host_storage')
        need(connection.execute('SELECT 59').fetchone()[0] == 59, 'caller_handle_not_live_after_refusal')
        return {'schema': 'legal-ir-full-original-host-cap-two-case@1', 'passed': True,
            'reviewed_composed_daemon_sha256': reviewed_daemon_sha256,
            'full_original_daemon_class_constructed': True, 'actual_private_DuckDB_connection': True,
            'original_cap_two': True, 'original_counter_row_unchanged': True,
            'configured_profile_refusal_qualified': True,
            'canonical_task_selection_or_cap_exhaustion_proved': False,
            'same_borrow_identity_observations': len(identities), 'original_guard_calls': len(guard_calls),
            'provider_calls_performed': False, 'Git_calls_inside_host_refusal': False,
            'constructor_Git_custody_attested_by_this_component': False,
            'actual_same_PID_three_CPU_phase': host.phase is original_phase,
            'host_finished': host.finished, 'host_pending': host.pending is not None,
            'hard_host_or_constructor_quota_qualified': False,
            'SQL_writes_during_measured_refusal': 0,
            'SELECT_witnesses_during_measured_refusal': len(readonly_statements), 'canonical_tasks_accepted': 0,
            'production_hook_deployed': False, 'acceptance_authority': False, 'completion_authority': False}
    finally:
        if host is not None and (host.bound or host.pending is not None):
            # This is a failed/incomplete fixture, never a successful refusal.
            # Do not let test cleanup close the actual host's borrowed handle.
            # The enclosing qualifier must catch inside its retained native
            # root/phase/owner frames and arrange original stopped settlement.
            raise FixtureCustodyPending(connection, host, original_phase, fixture_owner,
                raw_connection=caller_raw, lock_context=caller_lock)
        try:
            need(connection._connection is caller_raw and connection._poisoned is False,
                 'actual_caller_raw_identity_required_before_close')
            connection.close()  # Fixture caller owns this handle; host never does.
            need(connection._closed is True and connection._connection is None
                 and connection._poisoned is False and connection._lock_context is None
                 and connection._closing_owner == 0
                 and caller_lock is not None and caller_lock.gen.gi_frame is None,
                 'actual_caller_raw_and_original_lock_close_required')
        except BaseException as error:
            # A failed caller close must not be mistaken for a diagnostic
            # assertion whose known-stopped fixture may release its owner.
            raise FixtureCustodyPending(connection, host, original_phase, fixture_owner,
                raw_connection=caller_raw, lock_context=caller_lock) from error
