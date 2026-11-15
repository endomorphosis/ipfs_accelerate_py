"""Explicit native-owned, finite Git bootstrap execution in the original host.

Importing this module uses only the standard library and starts no process or
lease. Callers retain already admitted original SDK objects. This is cooperative
accounting in a trusted host, not a sandbox or transferable execution token.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
import weakref

_NATIVE = 'ipfs_datasets_py.optimizers.logic_theorem_optimizer.'
_CANONICAL = 'ipfs_accelerate_py.agent_supervisor.runtime.native_admin_execution'
_STATE = Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
_POLICY = 'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
_DATASETS = Path('/home/barberb/lift_coding/external/ipfs_datasets')
_LEDGER = _DATASETS/'workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
_ROOTS = (_DATASETS/'workspace/test-logs', _DATASETS/'workspace/todo-queues',
          _DATASETS/'docs/implementation/reports/evidence', Path('/tmp/pytest-of-barberb'))
_PYTHON = Path('/usr/bin/python3.12')
_GIT = Path('/usr/bin/git')
_BINARY_SHA = {
    str(_PYTHON): '1a301bb1763139d48ae638d97b11edf56de6cd185e1b054eae6dc28c271c0c5f',
    str(_GIT): 'aa6540695d076182256dd6e96c8b302e4d56381e3000bbfd5c71bbdfe94a4942',
}
_SDK_SHA = {
    'parented_phase_resources': 'd23e7f8a55d2563897b558502f05b11f89daa10f0ee6ff7dd2f142966ef7bce9',
    'autoencoder_daemon_resources': 'b9c8ca9f889366df93fd82887b7d6fcbfda9edee12a58a5667b485fe2c8a54ed',
    'resource_scheduler': '7c889bc858ff092e18bd2e97ff5a6407973e226eae33cddc9e5eef1d21f35986',
    'existing_resource_scheduler': 'e19dbd913e30c6960d4a848bc2b2f15236944e55bc7718598e047fbdf664f18d',
    'proof_resource_safety': '3655923785cb99409171caaba8c523354f03dbe8d82b6deb45a7b46db8394633',
}
_BOUND = ContextVar('original_native_admin_execution', default=None)
_ACTIVE = {}
_RETAINED = {}
_CREATED = weakref.WeakValueDictionary()
_ROUTE_PREFIX = ['-c', 'core.quotepath=false', '-c', 'core.fsmonitor=false',
                 '-c', 'core.hooksPath=/dev/null']
_ANCHOR = '''import os,resource,select,sys
for key,value in ((resource.RLIMIT_CPU,10),(resource.RLIMIT_AS,268435456),(resource.RLIMIT_NOFILE,64)):
 resource.setrlimit(key,(value,value))
if not select.select([0],[],[],float(sys.argv[1]))[0]: raise SystemExit(125)
if os.read(0,1)!=b'': raise SystemExit(125)
'''
_BARRIER = '''import json,os,resource,sys
spec=json.loads(sys.argv[1])
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
if os.read(0,1)!=b'G': raise SystemExit(125)
descriptor=os.open('/dev/null',os.O_RDONLY) if spec['stdin_fd'] is None else spec['stdin_fd']
os.dup2(descriptor,0)
if descriptor!=0: os.close(descriptor)
os.execve('/usr/bin/git',spec['argv'],dict(os.environ))
'''


class NativeAdminRefusal(BaseException):
    """Resource/source/custody refusal that must escape legacy Exception catches."""


def _need(value, reason):
    if not value:
        raise NativeAdminRefusal(reason)


def _canonical():
    module = sys.modules.get(_CANONICAL)
    _need(__name__ == _CANONICAL and module is not None and module.__dict__ is globals(),
          'original_canonical_runtime_module_required')


class _Nonportable:
    def __reduce__(self):
        raise NativeAdminRefusal('native_admin_objects_are_not_portable')
    def __reduce_ex__(self, protocol):
        raise NativeAdminRefusal('native_admin_objects_are_not_portable')
    def __copy__(self):
        raise NativeAdminRefusal('native_admin_objects_cannot_be_copied')
    def __deepcopy__(self, memo):
        raise NativeAdminRefusal('native_admin_objects_cannot_be_copied')


class NativeAdminCustodyPending(NativeAdminRefusal, _Nonportable):
    """Actual retained same-process references; no serialized cleanup authority."""
    def __init__(self, execution, cause):
        super().__init__('original_native_admin_custody_pending')
        self.actual_executor = execution
        self.actual_owner = execution.owner
        self.actual_phase = execution.phase_lease
        self.actual_anchor = execution.anchor
        self.actual_consumer = execution.consumer
        self.actual_processes = tuple(execution.processes)
        self.actual_cause = cause
    __reduce__ = _Nonportable.__reduce__
    __reduce_ex__ = _Nonportable.__reduce_ex__
    __copy__ = _Nonportable.__copy__
    __deepcopy__ = _Nonportable.__deepcopy__


def _directory(path):
    path = Path(path)
    _need(path.is_absolute() and str(path) == os.path.abspath(path), 'canonical_absolute_directory_required')
    for parent in (*reversed(path.parents), path):
        _need(stat.S_ISDIR(parent.lstat().st_mode), 'unaliased_directory_ancestry_required')
    return path


def _file_witness(value):
    return tuple(getattr(value, 'st_'+name) for name in
        ('dev','ino','mode','uid','gid','nlink','size','mtime_ns','ctime_ns'))


def _verify_pin(path, wanted):
    path = Path(path)
    _directory(path.parent)
    _need(type(wanted) is str and re.fullmatch('[0-9a-f]{64}', wanted), 'exact_SHA256_pin_required')
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and before.st_uid in {0, os.getuid()} and not before.st_mode & 0o002
              and 0 < before.st_size <= 32*1024*1024, 'bounded_unaliased_source_required')
        value = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = stream.read(min(65536,remaining))
            _need(chunk, 'source_short_read')
            value.update(chunk)
            remaining -= len(chunk)
        _need(stream.read(1) == b'', 'source_grew_during_bounded_hash')
        value = value.hexdigest()
        after = os.fstat(stream.fileno())
    _need(value == wanted and _file_witness(before) == _file_witness(after)
          == _file_witness(path.lstat()), 'original_native_admin_source_changed')


def _process(pid):
    try:
        descriptor = os.open(Path('/proc')/str(pid)/'stat', os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except (FileNotFoundError, ProcessLookupError):
        return None
    try:
        raw = os.read(descriptor, 8193)
    finally:
        os.close(descriptor)
    _need(len(raw) <= 8192 and b')' in raw, 'bounded_original_process_identity_required')
    fields = raw.rsplit(b')', 1)[1].split()
    _need(len(fields) >= 22, 'original_process_identity_unavailable')
    return dict(pid=pid, state=fields[0].decode('ascii'), parent_pid=int(fields[1]),
                group_pid=int(fields[2]), session_pid=int(fields[3]), birth=fields[19].decode('ascii'))


def _members(group):
    entries = list(Path('/proc').iterdir())
    _need(len(entries) <= 100000, 'bounded_process_group_census_required')
    result = []
    for entry in entries:
        if entry.name.isdecimal():
            row = _process(int(entry.name))
            if row is not None and row['group_pid'] == group and row['state'] != 'Z':
                result.append(row)
    return result


def _valid_ref(value):
    return bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/^-]{0,255}', value)
                and '..' not in value and '@{' not in value and '//' not in value
                and not value.endswith(('/', '.')))


def _git_arguments(argv):
    _need(type(argv) in (list, tuple) and 1 < len(argv) <= 32
          and all(type(item) is str and 0 < len(item) <= 4096 and '\0' not in item for item in argv),
          'finite_original_string_Git_argv_required')
    _need(argv[0] in ('git', str(_GIT)), 'only_exact_pinned_Git_supported')
    arguments = list(argv[1:])
    routed = arguments[:len(_ROUTE_PREFIX)] == _ROUTE_PREFIX
    if routed:
        arguments = arguments[len(_ROUTE_PREFIX):]
    allowed = arguments in ([ 'rev-parse', '--git-common-dir'],
        ['rev-parse','--show-toplevel'], ['rev-parse','--verify','HEAD^{commit}'],
        ['rev-parse','--verify','HEAD^{tree}'])
    allowed = allowed or (len(arguments) == 4 and arguments[:3] == ['rev-parse','--verify','--quiet']
                           and _valid_ref(arguments[3]))
    allowed = allowed or (routed and arguments == ['status','--porcelain=v1','-z','--untracked-files=all'])
    allowed = allowed or (routed and len(arguments) == 8 and arguments[:3] == ['ls-tree','-r','-z']
        and re.fullmatch('[0-9a-f]{40}', arguments[3]) and arguments[4:] == ['--','ipfs_accelerate_py',
        'scripts/ops/agent_supervisor/configured_board_scheduler.py',
        'scripts/ops/agent_supervisor/implementation_supervisor_entry.py'])
    allowed = allowed or (routed and len(arguments) == 3 and arguments[:2] in
        (['cat-file','-s'], ['cat-file','blob']) and re.fullmatch('[0-9a-f]{40}', arguments[2]))
    _need(allowed, 'Git_command_outside_original_bootstrap_readonly_profile')
    return [str(_GIT), *argv[1:]]


def _decode_output(raw, encoding, errors):
    # TextIOWrapper preserves subprocess's encoding default and universal newlines.
    with io.TextIOWrapper(io.BytesIO(raw), encoding=encoding, errors=errors, newline=None) as stream:
        return stream.read()


class NativeAdminExecution(_Nonportable):
    """One already admitted original owner, one anchor and finite joined Gits.

    Caller owns admission and retains owner/phase. Binding drains only this
    executor's children/consumer. Explicit release uses the owner's public
    stopped-release API; ancestors are never released here.
    """
    def __init__(self, owner, *, phase_lease, source_sha256, allowed_roots,
                 absolute_deadline, maximum_commands=128, maximum_stdout_bytes=1048576,
                 maximum_stderr_bytes=262144, maximum_total_output_bytes=16*1024*1024,
                 **unsupported):
        try:
            _canonical()
            _need(not unsupported, 'unsupported_native_admin_constructor_arguments')
            self._initialize(owner,phase_lease=phase_lease,source_sha256=source_sha256,
                allowed_roots=allowed_roots,absolute_deadline=absolute_deadline,
                maximum_commands=maximum_commands,maximum_stdout_bytes=maximum_stdout_bytes,
                maximum_stderr_bytes=maximum_stderr_bytes,
                maximum_total_output_bytes=maximum_total_output_bytes)
            _CREATED[id(self)] = self
        except NativeAdminRefusal:
            raise
        except BaseException as error:
            raise NativeAdminRefusal('original_native_admin_initialization_refused') from error

    def _initialize(self, owner, *, phase_lease, source_sha256, allowed_roots,
                    absolute_deadline, maximum_commands, maximum_stdout_bytes,
                    maximum_stderr_bytes, maximum_total_output_bytes):
        self.owner, self.phase_lease = owner, phase_lease
        self.anchor, self.anchor_birth, self.consumer = None, None, None
        self.processes = []
        self.command_count, self.total_output_bytes = 0, 0
        self.closed = self.released = False
        self._failed, self._pending, self._bound = False, None, False
        self._anchor_start_attempted = self._consumer_start_attempted = self._command_start_unknown = False
        self._internal_fds = set()
        self._host_pid, self._host_session = os.getpid(), os.getsid(0)
        self._host_thread = threading.get_ident()
        self._host_birth = _process(self._host_pid)['birth']
        _need(threading.current_thread() is threading.main_thread(), 'original_main_thread_required')
        _need(type(absolute_deadline) in (int,float) and math.isfinite(absolute_deadline)
              and 0 < absolute_deadline-time.monotonic() <= 600, 'finite_absolute_admin_deadline_required')
        _need(type(maximum_commands) is int and 1 <= maximum_commands <= 128, 'finite_command_limit_required')
        for value, limit in ((maximum_stdout_bytes,16*1024*1024), (maximum_stderr_bytes,1024*1024),
                             (maximum_total_output_bytes,64*1024*1024)):
            _need(type(value) is int and 0 < value <= limit, 'finite_output_bound_required')
        self._deadline, self._maximum_commands = absolute_deadline, maximum_commands
        self._stdout_limit, self._stderr_limit = maximum_stdout_bytes, maximum_stderr_bytes
        self._total_limit = maximum_total_output_bytes
        _need(type(source_sha256) is dict and 1 <= len(source_sha256) <= 512,
              'finite_explicit_source_closure_required')
        self._pins = dict(source_sha256)
        _need(str(Path(__file__).absolute()) in self._pins, 'actual_runtime_source_pin_required')
        _need(all(self._pins.get(path) == value for path,value in _BINARY_SHA.items()),
              'exact_original_binary_pins_required')
        _need(type(allowed_roots) in (tuple,list) and 1 <= len(allowed_roots) <= 16,
              'finite_actual_cwd_roots_required')
        self._roots = tuple(_directory(path) for path in allowed_roots)
        _need(len(set(self._roots)) == len(self._roots), 'unique_cwd_roots_required')
        self._root_ids = tuple((p.stat().st_dev,p.stat().st_ino) for p in self._roots)
        self._source_guard()
        modules = {}
        for name,wanted in _SDK_SHA.items():
            module = importlib.import_module(_NATIVE+name)
            path = str(Path(module.__file__).absolute())
            _need(self._pins.get(path) == wanted
                  and module.__spec__.origin == module.__file__, 'exact_original_SDK_module_required')
            _verify_pin(path, wanted)
            modules[name] = module
        self._owner_type = modules['parented_phase_resources'].ParentedPhaseResourceReservation
        self._lease_type = modules['resource_scheduler'].ResourceLease
        _need(type(owner) is self._owner_type and type(phase_lease) is self._lease_type,
              'actual_original_native_owner_and_phase_required')
        # A public handle only: no root/phase acquisition or key reconstruction.
        self._admission = modules['existing_resource_scheduler'].open_existing_scheduler(_STATE,_POLICY)
        self._native = None
        self._authority(full=True)

    def _identity(self):
        _canonical()
        _need(os.getpid() == self._host_pid and os.getsid(0) == self._host_session
              and threading.get_ident() == self._host_thread
              and threading.current_thread() is threading.main_thread(), 'original_same_PID_main_thread_required')
        row = _process(self._host_pid)
        _need(row is not None and row['birth'] == self._host_birth, 'original_host_birth_changed')

    def _source_guard(self):
        total = 0
        for path,wanted in self._pins.items():
            _need(type(path) is str and path == os.path.abspath(path), 'canonical_source_path_required')
            _need(time.monotonic() < self._deadline, 'source_guard_absolute_deadline_expired')
            total += Path(path).lstat().st_size
            _need(total <= 128*1024*1024, 'finite_aggregate_pinned_source_bound_exceeded')
            _verify_pin(path,wanted)

    def _authority(self, *, full=False):
        try:
            self._identity()
            _need(time.monotonic() < self._deadline, 'absolute_native_admin_deadline_expired')
            _need(type(self.owner) is self._owner_type and type(self.phase_lease) is self._lease_type,
                  'original_native_objects_changed')
            phase = self.phase_lease.to_dict()
            _need(phase['owner_pid'] == self._host_pid and phase['parent_lease_id'] is not None
                  and phase['lane'] == 'orchestration' and phase['cpu_slots'] == 3
                  and phase['memory_mb'] == 4096 and phase['child_process_slots'] == 16
                  and not phase['released'] and not phase['cancelled'] and not phase['requires_gpu']
                  and phase['gpu_memory_mb'] == phase['unified_memory_mb'] == 0
                  and Path(self.phase_lease.token.state_path) == _STATE
                  and self.phase_lease.renew(), 'actual_three_CPU_sixteen_slot_phase_required')
            if full:
                roots = [row for row in self._admission.active_leases()
                         if row.get('lease_id') == phase['parent_lease_id']]
                _need(len(roots) == 1, 'original_root_must_exist')
                root = roots[0]
                _need(root.get('parent_lease_id') is None and root.get('owner_pid') == self._host_pid
                      and root.get('lane') == 'orchestration' and root.get('cpu_slots') == 3
                      and root.get('memory_mb') == 4096 and root.get('child_process_slots') == 16
                      and root.get('cancelled') is False and root.get('requires_gpu') is False
                      and root.get('gpu_memory_mb') == root.get('unified_memory_mb') == 0,
                      'actual_original_same_PID_three_CPU_sixteen_slot_root_required')
            public = self.owner.to_dict()
            native = self.owner.native_lease
            value = native.to_dict()
            record = public.get('record') or {}
            _need(public.get('ledger_path') == str(_LEDGER)
                  and public.get('storage_limit_bytes') == 160000000000
                  and [row.get('path') for row in public.get('roots',())] == [str(root) for root in _ROOTS]
                  and record.get('parent_lease_id') == phase['lease_id']
                  and record.get('native_lease_id') == native.lease_id
                  and record.get('scheduler_state_path') == str(_STATE)
                  and record.get('scheduler_policy_sha256') == _POLICY,
                  'actual_original_global_storage_and_native_join_required')
            _need(type(native) is self._lease_type and public['status'] == 'active'
                  and value['owner_pid'] == self._host_pid and value['parent_lease_id'] == phase['lease_id']
                  and value['lane'] == 'orchestration' and value['cpu_slots'] == 1
                  and value['memory_mb'] == 512 and value['child_process_slots'] == 4
                  and not value['released'] and not value['cancelled'] and not value['requires_gpu']
                  and value['gpu_memory_mb'] == value['unified_memory_mb'] == 0
                  and Path(native.token.state_path) == _STATE and self.owner.heartbeat(),
                  'actual_one_CPU_512MiB_four_slot_admin_owner_required')
            _need(self._native is None or self._native is native, 'original_native_leaf_object_changed')
            self._native = native
            if self.consumer is not None:
                item = self.consumer.to_dict()
                _need(type(self.consumer) is self._lease_type and item['owner_pid'] == self._host_pid
                      and item['parent_lease_id'] == native.lease_id and item['lane'] == 'orchestration'
                      and item['cpu_slots'] == item['memory_mb'] == 1 and item['child_process_slots'] == 0
                      and not item['requires_gpu'] and item['gpu_memory_mb'] == item['unified_memory_mb'] == 0
                      and not item['released'] and not item['cancelled'] and self.consumer.renew(),
                      'actual_original_prestart_consumer_required')
            self.owner.check_usage(self.owner.directory, full=full)
            for path,identity in zip(self._roots,self._root_ids):
                _directory(path)
                _need((path.stat().st_dev,path.stat().st_ino) == identity, 'original_cwd_root_changed')
        except NativeAdminRefusal:
            raise
        except BaseException as error:
            raise NativeAdminRefusal('actual_original_SDK_authority_refused') from error

    def _anchor_identity(self, *, input_open=True):
        self._identity()
        _need(type(self.anchor) is subprocess.Popen and self.anchor.poll() is None
              and self.anchor_birth is not None and self.anchor.stdin is not None
              and (not input_open or not self.anchor.stdin.closed), 'original_live_anchor_object_required')
        row = _process(self.anchor.pid)
        _need(row is not None and row['birth'] == self.anchor_birth and row['state'] != 'Z'
              and row['parent_pid'] == self._host_pid and row['group_pid'] == row['pid'] == self.anchor.pid
              and row['session_pid'] == self._host_session and self.anchor.pid != os.getpgrp(),
              'original_direct_child_anchor_identity_required')
        return row

    def _anchor_only(self):
        self._authority()
        self._anchor_identity()
        rows = _members(self.anchor.pid)
        _need(len(rows) == 1 and rows[0]['pid'] == self.anchor.pid
              and rows[0]['birth'] == self.anchor_birth
              and all(type(p.poll()) is int for p in self.processes), 'exact_anchor_only_group_required')
        usage = self.owner.check_usage(self.owner.directory)
        _need(usage['group_rss']['live_processes'] == 1, 'native_anchor_only_group_required')

    def _start(self):
        _need(not self.closed and not self._failed and self._pending is None,
              'native_admin_scope_cannot_restart')
        if self.anchor is not None:
            self._anchor_only()
            return
        self._source_guard()
        self._authority(full=True)
        public = self.owner.to_dict()['record']
        _need(public.get('child') is None and public.get('prior_children') == [], 'fresh_native_admin_owner_required')
        _need(len(_ACTIVE) < 16 and len(_RETAINED) < 16,
              'bounded_actual_admin_scope_inventory_required')
        _ACTIVE[id(self)] = self  # Keep original refs before native acquisition/Popen can fail.
        try:
            self._consumer_start_attempted = True
            self.consumer = self._native.acquire_child(lane='orchestration', cpu_slots=1, memory_mb=1,
                child_process_slots=0, timeout=min(30,self._deadline-time.monotonic()),
                request_id='native-admin-bootstrap-group-hold')
            self._authority()
            self._anchor_start_attempted = True
            self.anchor = subprocess.Popen([str(_PYTHON),'-I','-S','-B','-c',_ANCHOR,
                str(max(.001,self._deadline-time.monotonic()))], cwd=self.owner.directory,
                env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','LANG':'C.UTF-8','LC_ALL':'C.UTF-8'},
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                process_group=0, close_fds=True)
            row = _process(self.anchor.pid)
            _need(row is not None, 'original_anchor_birth_unavailable')
            self.anchor_birth = row['birth']
            self._anchor_identity()
            attached = self.owner.check_usage(self.owner.directory, child_pid=self.anchor.pid, full=True)
            _need(attached['group_rss']['live_processes'] == 1
                  and self.owner.to_dict()['record'].get('child') ==
                  {'pid':self.anchor.pid,'birth':self.anchor_birth}, 'actual_native_anchor_attachment_required')
            self._anchor_only()
        except BaseException as error:
            self._stop_failed(error)
            raise NativeAdminRefusal('original_native_admin_start_refused') from error

    def _known_stopped(self):
        self._identity()
        if self._command_start_unknown:
            return False
        if self.anchor is None:
            return not self._anchor_start_attempted and not self.processes
        _need(type(self.anchor) is subprocess.Popen, 'actual_retained_anchor_required')
        if self.anchor.poll() is None or any(p.poll() is None for p in self.processes):
            return False
        row = _process(self.anchor.pid)
        _need(row is None or row['birth'] == self.anchor_birth, 'original_anchor_PID_reused')
        return not _members(self.anchor.pid)

    def _close_streams(self):
        for process in [*self.processes,*([self.anchor] if self.anchor is not None else [])]:
            for stream in (process.stdin,process.stdout,process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        for descriptor in tuple(self._internal_fds):
            os.close(descriptor)
            self._internal_fds.remove(descriptor)

    def _drain_consumer(self):
        _need(self._known_stopped(), 'original_native_admin_group_not_stopped')
        _need(not self._consumer_start_attempted or self.consumer is not None,
              'native_consumer_acquisition_outcome_unknown')
        if self.consumer is not None:
            _need(type(self.consumer) is self._lease_type, 'original_consumer_object_required')
            self.consumer.release()
            _need(self.consumer.released, 'original_consumer_release_not_observed')

    def _retain(self, cause):
        self._failed = True
        if self._pending is None:
            self._pending = NativeAdminCustodyPending(self,cause)
            _RETAINED[id(self)] = self
        raise self._pending from cause

    def _stop_failed(self, cause):
        if self._pending is not None:
            raise self._pending
        self._failed = True
        try:
            self._identity()
            if self.anchor is None:
                _need(not self._anchor_start_attempted, 'anchor_creation_outcome_unknown')
            elif self.anchor.poll() is None:
                self._anchor_identity(input_open=False)
                os.killpg(self.anchor.pid, signal.SIGKILL)
            deadline = time.monotonic()+10
            for process in [*self.processes,*([self.anchor] if self.anchor is not None else [])]:
                process.wait(timeout=max(.001,min(5,deadline-time.monotonic())))
            _need(self._known_stopped(), 'actual_original_admin_cleanup_unknown')
            self._close_streams()
            self._drain_consumer()
            self.closed = True
            _ACTIVE.pop(id(self),None)
        except BaseException as cleanup:
            self._retain(cleanup)

    def close(self):
        """Drain this original anchor/consumer; retain caller's owner/phase."""
        _need(_CREATED.get(id(self)) is self, 'original_constructed_executor_required')
        self._identity()
        if self._pending is not None:
            raise self._pending
        if self.closed:
            return
        try:
            if self.anchor is not None:
                self._anchor_only()
                self.anchor.stdin.close()
                _need(self.anchor.wait(timeout=5) == 0, 'original_anchor_close_failed')
            _need(self._known_stopped(), 'original_admin_group_not_quiescent')
            self._close_streams()
            self._drain_consumer()
            self.closed = True
            _ACTIVE.pop(id(self),None)
        except BaseException as error:
            self._stop_failed(error)
            raise NativeAdminRefusal('original_native_admin_close_refused') from error

    def run(self, argv, *, cwd=None, env=None, stdin=None, stdout=None, stderr=None,
            capture_output=False, text=None, encoding=None, errors=None, timeout=None,
            check=False, pass_fds=(), **unsupported):
        _need(_CREATED.get(id(self)) is self, 'original_constructed_executor_required')
        process, selector = None, None
        buffers = {'stdout':bytearray(),'stderr':bytearray()}
        original_argv = argv
        try:
            self._identity()
            _need(not unsupported, 'unsupported_bound_native_admin_run_arguments')
            _need(self._bound and _BOUND.get() is self, 'actual_bound_admin_scope_required')
            _need(type(capture_output) is bool and type(check) is bool and pass_fds == ()
                  and stdin in (None,subprocess.DEVNULL), 'original_bootstrap_stdin_and_flags_required')
            _need(text is None or type(text) is bool, 'original_text_flag_required')
            _need(encoding is None or type(encoding) is str, 'original_encoding_required')
            _need(errors is None or type(errors) is str, 'original_errors_required')
            _need(not capture_output or (stdout is None and stderr is None), 'capture_output_conflicts_with_streams')
            if capture_output:
                stdout = stderr = subprocess.PIPE
            _need(stdout in (subprocess.PIPE,subprocess.DEVNULL)
                  and stderr in (subprocess.PIPE,subprocess.DEVNULL), 'finite_PIPE_or_DEVNULL_outputs_required')
            _need(timeout is None or (type(timeout) in (int,float) and math.isfinite(timeout) and 0 < timeout <= 45),
                  'finite_supported_original_command_timeout_required')
            actual_argv = _git_arguments(argv)
            work = _directory(Path.cwd() if cwd is None else cwd)
            _need(any(work == root or root in work.parents for root in self._roots), 'Git_cwd_outside_actual_allowed_roots')
            environment = dict(os.environ) if env is None else env
            _need(type(environment) is dict and len(environment) <= 2048
                  and all(type(k) is type(v) is str and '\0' not in k+v and '=' not in k for k,v in environment.items())
                  and sum(len(k)+len(v) for k,v in environment.items()) <= 1024*1024,
                  'finite_original_string_environment_required')
            self._source_guard()
            self._start()
            self._anchor_only()
            self.command_count += 1
            _need(self.command_count <= self._maximum_commands, 'finite_admin_command_limit_exceeded')
            inherited = None
            if stdin is None:
                inherited = os.dup(0)
                self._internal_fds.add(inherited)
            spec = json.dumps({'argv':actual_argv,'stdin_fd':inherited},separators=(',',':'))
            self._command_start_unknown = True
            process = subprocess.Popen([str(_PYTHON),'-I','-S','-B','-c',_BARRIER,spec],
                cwd=work,env=environment,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,
                process_group=self.anchor.pid,pass_fds=(() if inherited is None else (inherited,)),close_fds=True)
            self.processes.append(process)
            self._command_start_unknown = False
            if inherited is not None:
                os.close(inherited)
                self._internal_fds.remove(inherited)
            row = _process(process.pid)
            _need(row is not None and row['parent_pid'] == self._host_pid
                  and row['group_pid'] == self.anchor.pid and row['session_pid'] == self._host_session
                  and row['state'] != 'Z', 'actual_gated_Git_group_identity_required')
            self._anchor_identity()
            usage = self.owner.check_usage(self.owner.directory)
            _need(usage['group_rss']['live_processes'] == 2
                  and {item['pid'] for item in _members(self.anchor.pid)} == {self.anchor.pid,process.pid},
                  'actual_anchor_plus_gated_Git_required_before_release')
            self._authority()
            process.stdin.write(b'G')
            process.stdin.close()
            selector = selectors.DefaultSelector()
            for name in buffers:
                stream = getattr(process,name)
                if stream is not None:
                    os.set_blocking(stream.fileno(),False)
                    selector.register(stream,selectors.EVENT_READ,name)
            command_timeout = 45 if timeout is None else timeout
            command_deadline, sample = time.monotonic()+command_timeout, 0.0
            while selector.get_map() or process.poll() is None:
                if time.monotonic() >= command_deadline:
                    raise subprocess.TimeoutExpired(original_argv,command_timeout,
                        output=bytes(buffers['stdout']),stderr=bytes(buffers['stderr']))
                _need(time.monotonic() < self._deadline, 'absolute_native_admin_deadline_expired')
                if time.monotonic() >= sample:
                    self._authority()
                    self._anchor_identity()
                    usage = self.owner.check_usage(self.owner.directory)
                    _need(1 <= usage['group_rss']['live_processes'] <= 4, 'native_admin_group_slot_bound_exceeded')
                    sample = time.monotonic()+.5
                for key,_ in selector.select(.1):
                    chunk = os.read(key.fileobj.fileno(),65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    self.total_output_bytes += len(chunk)
                    limit = self._stdout_limit if key.data == 'stdout' else self._stderr_limit
                    _need(len(buffers[key.data])+len(chunk) <= limit
                          and self.total_output_bytes <= self._total_limit, 'native_admin_output_bound_exceeded')
                    buffers[key.data].extend(chunk)
            returncode = process.wait(timeout=1)
            self._anchor_only()
        except BaseException as error:
            self._stop_failed(error)
            if isinstance(error, NativeAdminRefusal):
                raise
            raise NativeAdminRefusal('original_native_admin_execution_refused') from error
        finally:
            if selector is not None:
                selector.close()
            if process is not None and process.poll() is not None:
                for stream in (process.stdin,process.stdout,process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
        # These are known command-outcome semantics, separate from native refusal.
        outputs = {}
        for name,flag in (('stdout',stdout),('stderr',stderr)):
            raw = bytes(buffers[name])
            outputs[name] = (None if flag == subprocess.DEVNULL else
                _decode_output(raw,encoding,errors) if text or encoding or errors else raw)
        completed = subprocess.CompletedProcess(original_argv,returncode,**outputs)
        if check:
            completed.check_returncode()
        return completed

    def observation(self):
        """Safe observations only; no keys, grant, cancellation or portable object."""
        _need(_CREATED.get(id(self)) is self, 'original_constructed_executor_required')
        self._identity()
        public = self.owner.to_dict()
        record = public.get('record') or {}
        return dict(command_count=self.command_count,total_output_bytes=self.total_output_bytes,
            anchor_pid=None if self.anchor is None else self.anchor.pid,anchor_birth=self.anchor_birth,
            session_pid=self._host_session,host_pid=self._host_pid,closed=self.closed,released=self.released,
            pending=self._pending is not None,native_child=record.get('child'),
            native_prior_children=record.get('prior_children'),
            consumer=None if self.consumer is None else self.consumer.to_dict(),
            native_owner_status=public['status'],portable_projection_grants_authority=False)

    def release(self, *, artifacts_durable=False):
        """Use original public stopped release only after original group drain."""
        _need(_CREATED.get(id(self)) is self, 'original_constructed_executor_required')
        self._identity()
        _need(artifacts_durable is True, 'explicit_original_artifact_durability_required')
        self.close()
        _need(self._known_stopped(), 'original_native_admin_group_not_stopped')
        try:
            result = self.owner.release(artifacts_durable=True)
            _need(result['status'] == 'released', 'original_owner_release_not_observed')
            self.released = True
            return result
        except BaseException as error:
            self._retain(error)


def native_admin_execution_is_bound():
    """Pure evaluation discriminator for unchanged unbound original producers."""
    return _BOUND.get() is not None


def run_bound_native_admin_if_present(argv, *, cwd=None, env=None, stdin=None,
        stdout=None, stderr=None, capture_output=False, text=None, encoding=None,
        errors=None, timeout=None, check=False, pass_fds=(), **unsupported):
    execution = _BOUND.get()
    if execution is None:
        return None
    _canonical()
    _need(type(execution) is NativeAdminExecution, 'exact_original_bound_executor_required')
    return execution.run(argv,cwd=cwd,env=env,stdin=stdin,stdout=stdout,stderr=stderr,
        capture_output=capture_output,text=text,encoding=encoding,errors=errors,timeout=timeout,
        check=check,pass_fds=pass_fds,**unsupported)


@contextmanager
def bind_native_admin_execution(execution):
    _canonical()
    _need(type(execution) is NativeAdminExecution and _CREATED.get(id(execution)) is execution
          and _BOUND.get() is None
          and not execution._bound, 'one_exact_original_native_admin_binding_required')
    execution._start()
    token = _BOUND.set(execution)
    execution._bound = True
    try:
        yield execution
    except BaseException as error:
        execution._stop_failed(error)
        raise
    else:
        execution.close()
    finally:
        execution._bound = False
        _BOUND.reset(token)


def acknowledge_stopped_native_admin_custody(pending):
    """Observe original stopped objects and drain consumer; no signal/new grant."""
    _need(type(pending) is NativeAdminCustodyPending, 'exact_original_pending_admin_required')
    execution = pending.actual_executor
    _need(type(execution) is NativeAdminExecution and _RETAINED.get(id(execution)) is execution
          and execution._pending is pending and pending.actual_owner is execution.owner
          and pending.actual_phase is execution.phase_lease and pending.actual_anchor is execution.anchor
          and pending.actual_consumer is execution.consumer
          and pending.actual_processes == tuple(execution.processes), 'original_retained_admin_identity_required')
    execution._identity()
    if not execution._known_stopped():
        return False
    try:
        execution._close_streams()
        execution._drain_consumer()
    except BaseException:
        return False
    execution.closed = True
    execution._pending = None
    _RETAINED.pop(id(execution))
    _ACTIVE.pop(id(execution),None)
    return True


__all__ = ['NativeAdminExecution','NativeAdminRefusal','NativeAdminCustodyPending',
    'native_admin_execution_is_bound','run_bound_native_admin_if_present',
    'bind_native_admin_execution','acknowledge_stopped_native_admin_custody']
