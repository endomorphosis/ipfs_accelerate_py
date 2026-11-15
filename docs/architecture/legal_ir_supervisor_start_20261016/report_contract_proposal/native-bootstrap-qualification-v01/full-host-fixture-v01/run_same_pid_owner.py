"""Original same-PID 3CPU/4096MiB/16-slot full-host refusal qualifier.

Source proposal only. This is a direct original owner, never a worker nested
under the generic1CPU/8-slot launcher. Productive work is finite. Unknown
custody retains actual ancestors and objects instead of releasing on unwind.
"""
import importlib
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

from fixture_io import canonical, fsync_dir, need, read, sha, source_guard, strict, wire, write_new
from prepare_capsule_and_seed import population, relative

HERE = Path(__file__).absolute().parent
DATASETS = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets')
NATIVE = 'ipfs_datasets_py.optimizers.logic_theorem_optimizer.'
TODO = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.'
ADMIN = 'ipfs_accelerate_py.agent_supervisor.runtime.native_admin_execution'
STATE = Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
POLICY = 'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
ORIGINAL = Path('/home/barberb/lift_coding/external/ipfs_datasets')
ROOTS = (ORIGINAL / 'workspace/test-logs', ORIGINAL / 'workspace/todo-queues',
         ORIGINAL / 'docs/implementation/reports/evidence', Path('/tmp/pytest-of-barberb'))
LEDGER = ROOTS[0] / 'federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
_RETAINED = {}


class FullHostFixtureCustodyPending(BaseException):
    def __init__(self, actual_objects, cause):
        super().__init__('original_same_PID_full_host_fixture_custody_pending')
        self.actual_objects, self.actual_cause = actual_objects, cause
        _RETAINED[id(self)] = self

    def __reduce__(self):
        raise TypeError('original_fixture_custody_is_not_portable')


class FullHostFixtureFailure(Exception):
    def __init__(self, result):
        super().__init__('original_fixture_failed_after_known_stopped_closeout')
        self.result = result


def no_children(frame, event, _value):
    if event == 'call' and frame.f_code is subprocess.Popen.__init__.__code__:
        raise RuntimeError('child_launch_before_original_admin_binding_refused')


def fsync_tree(root):
    files, directories, _ = population(root, maximum_files=8192,
        maximum_bytes=128 * 1024**2, readonly=False)
    for name in files:
        fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        fsync_dir(path)


def preparation(request):
    observed = strict(read(request['preparation_root_observation_path'],
        request['preparation_root_observation_sha256'], maximum=1024**2))
    need(observed.get('schema') == 'legal-ir-native-bootstrap-preparation-root-observation@1'
         and observed.get('passed') is True and observed.get('native_finalized') is True
         and observed.get('actual_parent_finalized') is True
         and observed.get('actual_root_phase_owner_rows_drained') is True
         and observed.get('capsule_root') == request['capsule_root']
         and observed.get('capsule_manifest_sha256') == request['capsule_manifest_sha256']
         and observed.get('seed_root') == request['seed_root']
         and observed.get('seed_manifest_sha256') == request['seed_manifest_sha256'],
         'actual_independently_finalized_preparation_join_required')
    capsule = canonical(request['capsule_root'])
    manifest = strict(read(capsule / 'SOURCE_MANIFEST.json',
        request['capsule_manifest_sha256'], maximum=4 * 1024**2))
    need(manifest.get('schema') == 'legal-ir-native-bootstrap-full-host-capsule@1'
         and manifest.get('capsule_root') == str(capsule)
         and manifest.get('payload_file_count') == 2813
         and manifest.get('runtime_enabled') is False,
         'complete_original_readonly_bootstrap_capsule_required')
    expected, total = {}, 0
    for row in manifest['files']:
        name = relative(row['path']).as_posix()
        need(name.endswith('.py') and name not in expected, 'unique_capsule_python_source_required')
        raw = read(capsule / name, row['sha256']); total += len(raw)
        need(len(raw) == row['bytes'], 'capsule_size_changed')
        expected[name] = {'bytes': row['bytes'], 'sha256': row['sha256']}
    actual, _, _ = population(capsule, maximum_files=2814,
        maximum_bytes=120 * 1024**2, readonly=True)
    need(len(expected) == 2813 and total == manifest['payload_bytes']
         and set(actual) == set(expected) | {'SOURCE_MANIFEST.json'}
         and all(actual[name] == row for name, row in expected.items()),
         'exact_capsule_population_required')
    seed = canonical(request['seed_root'])
    seed_manifest = strict(read(request['seed_manifest_path'],
        request['seed_manifest_sha256'], maximum=1024**2))
    need(seed_manifest.get('schema') == 'legal-ir-native-bootstrap-private-Git-seed@1'
         and seed_manifest.get('seed_root') == str(seed)
         and seed_manifest.get('branch') == 'fixture-main'
         and seed_manifest.get('alternates') is False
         and seed_manifest.get('hardlinks') is False and seed_manifest.get('symlinks') is False
         and seed_manifest.get('submodules') is False,
         'self_contained_reviewed_private_seed_required')
    actual, _, seed_total = population(seed, maximum_files=128,
        maximum_bytes=1024**2, readonly=True)
    expected = {}
    for row in seed_manifest['files']:
        name = relative(row['path']).as_posix()
        need(name not in expected, 'unique_seed_file_required')
        expected[name] = {'bytes': row['bytes'], 'sha256': row['sha256']}
    need(actual == expected and len(actual) == seed_manifest['payload_file_count']
         and seed_total == seed_manifest['payload_bytes']
         and '.git/HEAD' in actual and not (seed / '.git/objects/info/alternates').exists(),
         'exact_immutable_seed_population_required')
    return capsule, manifest, seed, seed_manifest


def own_proc_file(name):
    need(name in {'stat', 'status'}, 'fixed_current_PID_proc_observation_required')
    # procfs reports zero st_size and /proc/self is a symlink, so the ordinary
    # immutable-source reader deliberately cannot be used for this observation.
    path = Path('/proc') / str(os.getpid()) / name
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(fd)
        need(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid(),
             'actual_current_PID_status_required')
        raw = os.read(fd, 65537)
        need(len(raw) <= 65536, 'bounded_current_PID_status_required')
    finally:
        os.close(fd)
    return raw


def own_birth():
    raw = own_proc_file('stat').decode()
    closing = raw.rfind(')')
    need(closing >= 0 and raw[:raw.find('(')].strip() == str(os.getpid()),
         'actual_current_PID_stat_required')
    fields = raw[closing + 2:].split()
    need(len(fields) > 19 and fields[19].isdigit(), 'actual_process_birth_ticks_required')
    return fields[19]


def sample_host(stage, expected_birth):
    before = own_birth(); fields = {}
    for line in own_proc_file('status').decode().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, amount, unit = line.split()
            need(unit == 'kB', 'original_host_RSS_units_required')
            fields[key[:-1]] = int(amount) * 1024
    need(before == own_birth() == expected_birth
         and set(fields) == {'VmRSS', 'VmHWM'} and fields['VmRSS'] <= 4096 * 1024**2,
         'observed_original_phase_host_RSS_bound_exceeded')
    return dict(stage=stage, owner_pid=os.getpid(), **fields,
        birth_ticks=before, observation_only_not_kernel_quota=True, host_self_attachment=False)


def cleanup_known_stopped_failure(objects, cause):
    """Use actual public cleanup; return false on any uncertain boundary.

    A cap-case pending exception explicitly retains unknown borrow/close state.
    An ordinary case assertion has already run its source-defined caller close.
    No JSON flag or absence observation substitutes for these original objects.
    """
    case = objects.get('case_module')
    if case is not None and type(cause) is case.FixtureCustodyPending:
        return False
    daemon = objects.get('daemon')
    if daemon is not None and getattr(daemon, '_report_only_native_host', None) is not None:
        return False
    try:
        admin, executor = objects.get('admin_module'), objects.get('executor')
        if executor is not None:
            if type(cause) is admin.NativeAdminCustodyPending:
                if objects.get('admin_pending_acknowledged') is not cause:
                    if admin.acknowledge_stopped_native_admin_custody(cause) is not True:
                        return False
                    objects['admin_pending_acknowledged'] = cause
            else:
                # Normal diagnostic failure may leave a known actual anchor
                # open. This public method closes only its original objects.
                executor.close()
            fsync_tree(Path(executor.owner.directory))
            executor.release(artifacts_durable=True)
        else:
            owner = objects.get('admin_owner')
            if owner is not None:
                fsync_tree(Path(owner.directory))
                owner.release(artifacts_durable=True)
        owner = objects.get('data_owner')
        if owner is not None:
            data = Path(owner.directory)
            # A late diagnostic can fail after successful data finalization.
            # Preserve its failure on stdout/root evidence, without writing a
            # new artifact after the original disk owner already released.
            if owner.to_dict()['status'] != 'released':
                failure_path = data / 'failed-full-host-qualification.json'
                failure_raw = wire({
                    'schema': 'legal-ir-full-host-known-stopped-failure@1', 'passed': False,
                    'failure_type': type(cause).__name__, 'provider_or_model_calls': False,
                    'canonical_counter_or_cap_changed': False, 'runtime_enabled': False,
                    'acceptance_authority': False, 'native_closeout_pending': True})
                if failure_path.exists():
                    need(read(failure_path) == failure_raw, 'existing_failure_artifact_conflict')
                else:
                    write_new(failure_path, failure_raw)
                fsync_tree(data)
            owner.release(artifacts_durable=True)
        for name in ('phase', 'root'):
            lease = objects.get(name)
            if lease is not None and not lease.released:
                lease.release()
        return True
    except BaseException as cleanup:
        # Retain both causes, including a public cleanup failure with new actual
        # pending references. Do not unwind or retry a productive operation.
        objects['cleanup_cause'] = cleanup
        return False


def run(request):
    objects = {}; old_umask = None
    try:
        need(sys.dont_write_bytecode and threading.current_thread() is threading.main_thread()
             and sys.getprofile() is None, 'fresh_main_thread_no_bytecode_owner_required')
        source_guard(HERE, request['fixture_source_manifest_sha256'])
        capsule, manifest, seed, seed_manifest = preparation(request)
        pins, custody, admin_pins = (dict(request[name]) for name in
            ('protected_source_sha256', 'custody_source_sha256', 'admin_source_sha256'))
        need(0 < len(pins) <= 128 and 0 < len(custody) <= 32 and 0 < len(admin_pins) <= 512
             and all(pins.get(name) == digest for name, digest in custody.items()),
             'bounded_closed_actual_source_sets_required')
        for source_set in (pins, custody, admin_pins):
            for name, digest in source_set.items():
                read(name, digest, maximum=64 * 1024**2)
        need(not any(name == 'ipfs_accelerate_py' or name.startswith('ipfs_accelerate_py.')
                     or name == 'ipfs_datasets_py' or name.startswith('ipfs_datasets_py.')
                     for name in sys.modules), 'fresh_product_and_native_imports_required')
        for name in tuple(os.environ):
            if name.startswith(('IPFS_ACCELERATE_AGENT_', 'GIT_')):
                del os.environ[name]
        os.environ.update(PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1',
            IPFS_ACCEL_SKIP_CORE='1', IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',
            IPFS_AUTO_INSTALL='false', IPFS_DATASETS_AUTO_INSTALL='false',
            IPFS_ACCELERATE_DUCKDB_ONLY='1', GIT_CONFIG_NOSYSTEM='1',
            GIT_CONFIG_GLOBAL='/dev/null', GIT_TERMINAL_PROMPT='0',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        sys.path.insert(0, str(DATASETS))
        sys.setprofile(no_children)
        try:
            existing = importlib.import_module(NATIVE + 'existing_resource_scheduler')
            native = importlib.import_module(NATIVE + 'resource_scheduler')
            parented = importlib.import_module(NATIVE + 'parented_phase_resources')
        finally:
            sys.setprofile(None)
        for module in (existing, native, parented):
            name = str(Path(module.__file__).absolute())
            need(name in pins, 'exact_original_native_module_origin_required'); read(name, pins[name])
        scheduler = existing.open_existing_scheduler(STATE, POLICY)
        cancelled = threading.Event()
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, lambda _number, _frame: cancelled.set())
        birth = own_birth()
        root = scheduler.acquire(native.ResourceLane.ORCHESTRATION, cpu_slots=3, memory_mb=4096,
            child_process_slots=16, timeout=900, cancel_event=cancelled, request_id=request['name'])
        objects['root'] = root
        phase = root.acquire_child(lane=native.ResourceLane.ORCHESTRATION,
            cpu_slots=3, memory_mb=4096, child_process_slots=16, timeout=900,
            cancel_event=cancelled, request_id=request['name'] + ':phase')
        objects['phase'] = phase
        attempt = canonical(request['attempt_directory'])
        need(any(root_path in attempt.parents for root_path in ROOTS)
             and not os.path.lexists(attempt), 'fresh_named_original_owner_attempt_required')
        old_umask = os.umask(0o077)
        attempt.mkdir(mode=0o700, parents=True)
        data, administrative = attempt / 'data', attempt / 'admin'
        data.mkdir(mode=0o700); administrative.mkdir(mode=0o700)
        for role, directory, memory, slots, storage in (
                ('data_owner', data, 3072, 8, 128 * 1024**2),
                ('admin_owner', administrative, 512, 4, 32 * 1024**2)):
            owner = parented.ParentedPhaseResourceReservation(LEDGER, roots=ROOTS,
                attempt_directory=directory, parent_lease=phase, storage_bytes=storage,
                memory_mb=memory, cpu_slots=1, child_process_slots=slots,
                scheduler_state_path=STATE, expected_scheduler_policy_sha256=POLICY,
                timeout_seconds=request['duration_seconds'] + 300, ledger_lock_timeout_seconds=None)
            objects[role] = owner; owner.__enter__()
        data_owner, admin_owner = objects['data_owner'], objects['admin_owner']
        fixture = data / 'case'; fixture.mkdir(mode=0o700)
        (fixture / 'lane').mkdir(mode=0o700); (data / 'tmp').mkdir(mode=0o700)
        os.environ.update(TMPDIR=str(data / 'tmp'), TMP=str(data / 'tmp'), TEMP=str(data / 'tmp'),
            IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(fixture / 'orchestration'),
            IPFS_ACCELERATE_AGENT_RECLAIM_DEAD_WORKTREE_LEASES_ON_STARTUP='false')
        need(canonical(tempfile.gettempdir()) == data / 'tmp',
             'original_tempfile_cache_must_select_actually_charged_data_directory')
        repo = fixture / 'repository'; repo.mkdir(mode=0o700)
        for row in seed_manifest['files']:
            target = repo / relative(row['path'])
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            write_new(target, read(seed / row['path'], row['sha256']), mode=0o600)
        need(not (repo / '.git/objects/info/alternates').exists(), 'copied_seed_alternates_refused')
        sys.path.insert(0, str(capsule))
        sys.setprofile(no_children)
        try:
            admin = importlib.import_module(ADMIN)
        finally:
            sys.setprofile(None)
        objects['admin_module'] = admin
        actual = str(Path(admin.__file__).absolute())
        need(actual in admin_pins and admin_pins[actual] == manifest['reviewed_native_admin_sha256'],
             'exact_source_owned_canonical_admin_required'); read(actual, admin_pins[actual])
        executor = admin.NativeAdminExecution(admin_owner, phase_lease=phase,
            source_sha256=admin_pins, allowed_roots=(fixture, capsule),
            absolute_deadline=time.monotonic() + 600, maximum_commands=128,
            maximum_stdout_bytes=4 * 1024**2, maximum_stderr_bytes=256 * 1024,
            maximum_total_output_bytes=16 * 1024**2)
        objects['executor'] = executor
        popen_observations, samples = [], [sample_host('before_bound_admin', birth)]
        allowed = {admin.NativeAdminExecution._start.__code__, admin.NativeAdminExecution.run.__code__}
        def owned_children(frame, event, _value):
            if frame.f_code is not subprocess.Popen.__init__.__code__:
                return
            if event == 'call':
                caller = frame.f_back
                need(caller is not None and caller.f_code in allowed
                     and caller.f_globals is vars(admin) and caller.f_locals.get('self') is executor,
                     'uncovered_bootstrap_child_before_launch_refused')
            elif event == 'return':
                child = frame.f_locals['self']
                popen_observations.append({'pid': child.pid, 'owner_pid': os.getpid(),
                    'pgid': os.getpgid(child.pid), 'session': os.getsid(child.pid)})
        sys.setprofile(owned_children)
        try:
            with admin.bind_native_admin_execution(executor):
                first = executor.observation(); semantics = []
                for ordinal in range(20):
                    need(not cancelled.is_set(), 'original_fixture_cancelled_before_command')
                    argv = ['/usr/bin/git', 'rev-parse', '--verify', 'HEAD^{commit}']
                    text = ordinal % 2 == 0
                    completed = admin.run_bound_native_admin_if_present(argv, cwd=repo,
                        stdin=subprocess.DEVNULL, capture_output=True, text=text, timeout=20, check=True)
                    expected = seed_manifest['head'] + '\n'
                    need(type(completed) is subprocess.CompletedProcess and completed.args is argv
                         and completed.returncode == 0
                         and completed.stdout == (expected if text else expected.encode())
                         and completed.stderr == ('' if text else b''), 'actual_Git_byte_text_semantics_failed')
                    observed = executor.observation()
                    need(observed['anchor_pid'] == first['anchor_pid']
                         and observed['native_child'] == first['native_child']
                         and observed['native_prior_children'] in (None, []),
                         'sequential_commands_replaced_original_anchor_history')
                    semantics.append({'ordinal': ordinal + 1, 'text': text, 'passed': True})
                missing = ['/usr/bin/git', 'rev-parse', '--verify', '--quiet', 'refs/heads/fixture-missing']
                nonzero = admin.run_bound_native_admin_if_present(missing, cwd=repo,
                    stdin=subprocess.DEVNULL, capture_output=True, text=False, timeout=20, check=False)
                need(type(nonzero) is subprocess.CompletedProcess and nonzero.args is missing
                     and nonzero.returncode != 0 and type(nonzero.stdout) is bytes
                     and type(nonzero.stderr) is bytes, 'original_nonzero_CompletedProcess_semantics_failed')
                caught = None
                try:
                    admin.run_bound_native_admin_if_present(missing, cwd=repo,
                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20, check=True)
                except subprocess.CalledProcessError as error:
                    caught = error
                need(type(caught) is subprocess.CalledProcessError and caught.cmd is missing
                     and caught.returncode != 0 and type(caught.output) is str,
                     'original_check_true_CalledProcessError_semantics_failed')
                samples.append(sample_host('before_full_original_import', birth))
                daemon_module = importlib.import_module(TODO + 'implementation_daemon')
                host_module = importlib.import_module(TODO + 'report_only_native_host')
                importlib.import_module(TODO + 'portal_task_state_control_plane')
                importlib.import_module('ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state')
                for module in (daemon_module, host_module):
                    actual = str(Path(module.__file__).absolute())
                    need(actual in pins, 'actual_full_module_source_pin_required'); read(actual, pins[actual])
                lane = fixture / 'lane'
                daemon = daemon_module.PortalImplementationDaemon(
                    todo_path=repo / 'tasks.todo.md', state_path=lane / 'state.json',
                    strategy_path=lane / 'strategy.json', events_path=lane / 'events.jsonl',
                    repo_root=repo, implement=True, implementation_command='/usr/bin/false',
                    production_provider_policy='', max_task_attempts=2, preserve_task_attempts=True,
                    use_ephemeral_worktree=True, worktree_root=fixture / 'worktrees',
                    retain_worktree_artifacts=True, worktree_pool_enabled=False,
                    merge_target_branch='fixture-main', merge_queue_dir=fixture / 'merge-queue',
                    validation_cache_dir=fixture / 'validation-cache', validation_max_workers=1,
                    validation_resource_budget=1, proof_cache_dir=fixture / 'proof-cache',
                    maintenance_interval_seconds=0, merge_reconciliation_max_merges=0,
                    merged_worktree_cleanup_max=0, require_active_plan_revision=False,
                    worktree_submodule_paths=(), report_only_candidate_freezing=True)
                objects['daemon'] = daemon
                samples.append(sample_host('after_original_constructor', birth))
                bound_observation = executor.observation()
        finally:
            sys.setprofile(None)
        closed = executor.observation()
        need(closed['closed'] and not closed['pending'] and closed['command_count'] >= 22
             and closed['native_prior_children'] in (None, [])
             and closed['consumer'] is not None and closed['consumer']['released'],
             'original_admin_anchor_consumer_drain_required_before_host')
        fsync_tree(administrative)
        admin_release = executor.release(artifacts_durable=True)
        need(admin_release['status'] == 'released', 'original_admin_release_before_host_required')
        import cap_two_case
        objects['case_module'] = cap_two_case
        need(not cancelled.is_set(), 'original_fixture_cancelled_before_host')
        case_result = cap_two_case.run_cap_two_case(daemon=daemon, original_phase=phase,
            fixture_owner=data_owner, fixture_root=fixture, protected_source_sha256=pins,
            custody_source_sha256=custody,
            reviewed_daemon_sha256=manifest['reviewed_composed_daemon_sha256'])
        samples.append(sample_host('after_cap_two_refusal', birth))
        data_owner.check_usage(data, full=True)
        result = dict(schema='legal-ir-same-PID-full-host-qualification-result@1', passed=True,
            same_PID=os.getpid(), actual_three_CPU_root_lease_id=root.lease_id,
            actual_three_CPU_phase_lease_id=phase.lease_id,
            bootstrap_admin_component_semantics=semantics,
            bootstrap_admin_nonzero_check_false_passed=True, bootstrap_admin_check_true_passed=True,
            bootstrap_admin_bound_observation=bound_observation,
            bootstrap_admin_closed_observation=closed, bootstrap_admin_release=admin_release,
            actual_Popen_observations=popen_observations, host_RSS_observations=samples,
            actual_cap_two_case=case_result, actual_consumed_ordinal_qualified=False,
            provider_or_model_calls=False, Docker_calls=False, runtime_enabled=False,
            canonical_tasks_accepted=0, canonical_configured_attempt_cap=2,
            canonical_counter_or_cap_changed=False, acceptance_authority=False,
            completion_authority=False, original_native_closeout_pending=True)
        write_new(data / 'full-host-qualification.json', wire(result))
        fsync_tree(data); fsync_dir(administrative); fsync_dir(attempt)
        data_release = data_owner.finalize(data, artifacts_durable=True)
        need(data_release['status'] == 'released', 'original_data_finalization_required')
        need(phase.release() and root.release(), 'actual_original_ancestor_release_required')
        ids = {root.lease_id, phase.lease_id}
        need(not any(row['lease_id'] in ids for row in scheduler.active_leases()),
             'original_ancestor_rows_still_present')
        result['original_native_closeout_pending'] = False
        result['observed_public_data_release'] = data_release
        result['observed_original_ancestor_rows_absent'] = True
        return result
    except BaseException as error:
        # Deliberately no blanket finally-release: keep actual capability and
        # borrow frames for public stopped acknowledgement/root-owned recovery.
        # This source author has not qualified these failure paths.
        if not objects:
            raise
        if cleanup_known_stopped_failure(objects, error):
            raise FullHostFixtureFailure({'schema': 'legal-ir-full-host-known-stopped-failure@1',
                'passed': False, 'failure_type': type(error).__name__,
                'actual_public_stopped_cleanup_completed': True,
                'original_native_closeout_requires_root_verification': True,
                'runtime_enabled': False, 'acceptance_authority': False}) from None
        pending = FullHostFixtureCustodyPending(objects, error)
        raise pending from error
    finally:
        sys.setprofile(None)
        if old_umask is not None:
            os.umask(old_umask)


def main():
    need(len(sys.argv) == 3, 'exact_original_same_PID_request_and_SHA_required')
    request = strict(read(sys.argv[1], sys.argv[2], maximum=4 * 1024**2))
    need(type(request) is dict and set(request) == {'schema', 'name', 'duration_seconds',
        'attempt_directory', 'fixture_source_manifest_sha256', 'preparation_root_observation_path',
        'preparation_root_observation_sha256', 'capsule_root', 'capsule_manifest_sha256',
        'seed_root', 'seed_manifest_path', 'seed_manifest_sha256',
        'protected_source_sha256', 'custody_source_sha256', 'admin_source_sha256'}
        and request['schema'] == 'legal-ir-full-host-same-PID-qualification-request@1'
        and type(request['duration_seconds']) is int and 10 <= request['duration_seconds'] <= 1800,
        'closed_finite_original_same_PID_owner_request_required')
    try:
        result = run(request)
    except FullHostFixtureFailure as failure:
        print(wire(failure.result).decode(), flush=True)
        raise SystemExit(1)
    except FullHostFixtureCustodyPending as pending:
        # No signals, new children/grants, retries or automatic host/borrow
        # settlement. Keep this genuine PID and objects available to root.
        print(wire({'schema': 'legal-ir-full-host-retained-custody@1', 'passed': False,
            'status': 'pending_original_stopped_proof', 'owner_pid': os.getpid(),
            'failure_type': type(pending.actual_cause).__name__, 'acceptance_authority': False}).decode(), flush=True)
        acknowledged_pending = None
        while True:
            # Public component acknowledgement observes only retained original
            # objects. It cannot authorize host borrowing/disposition release.
            admin = pending.actual_objects.get('admin_module')
            cause = pending.actual_objects.get('cleanup_cause', pending.actual_cause)
            cleanup_completed = False
            if cause is not acknowledged_pending and admin is not None and type(cause) is admin.NativeAdminCustodyPending:
                try:
                    if admin.acknowledge_stopped_native_admin_custody(cause) is True:
                        acknowledged_pending = cause
                        pending.actual_objects['admin_pending_acknowledged'] = cause
                        cleanup_completed = cleanup_known_stopped_failure(
                            pending.actual_objects, pending.actual_cause)
                except BaseException:
                    pass  # Unknown observation cannot unwind original custody.
            if cleanup_completed:
                print(wire({'schema': 'legal-ir-full-host-known-stopped-failure@1',
                    'passed': False, 'public_stopped_acknowledgement_completed': True,
                    'actual_public_stopped_cleanup_completed': True,
                    'original_native_closeout_requires_root_verification': True,
                    'runtime_enabled': False, 'acceptance_authority': False}).decode(), flush=True)
                raise SystemExit(1)
            time.sleep(1)
    else:
        print(wire(result).decode(), flush=True)


if __name__ == '__main__':
    main()
