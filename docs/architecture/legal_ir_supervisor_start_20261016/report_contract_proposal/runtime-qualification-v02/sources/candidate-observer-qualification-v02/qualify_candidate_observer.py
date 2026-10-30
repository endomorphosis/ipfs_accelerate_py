"""Genuine finite native/Git qualification using an explicit metadata fixture.

This entry is source-only until root executes its frozen request. Its candidate
is visibly synthetic; no provider lifecycle, native validation or task acceptance
can be established by this fixture or its actual read-only observer result.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time

HERE = Path(__file__).absolute().parent
PHASE = HERE.parent
ACCELERATOR = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
DATASETS = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets')
OBSERVER = PHASE / 'real-candidate-binding-v03/real_candidate_binding.py'
BOARD_SOURCE = Path('/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/acceptance-adapter/adapter-v02/tests/fixtures/bootstrap/tasks.todo.md')
DATASETS_ROOT = Path('/home/barberb/lift_coding/external/ipfs_datasets')
ROOTS = (DATASETS_ROOT / 'workspace/test-logs', DATASETS_ROOT / 'workspace/todo-queues',
         DATASETS_ROOT / 'docs/implementation/reports/evidence', Path('/tmp/pytest-of-barberb'))
STATE = Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
POLICY = 'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
LEDGER = DATASETS_ROOT / 'workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
NATIVE_NAMES = ('autoencoder_daemon_resources', 'resource_scheduler', 'existing_resource_scheduler', 'parented_phase_resources')
NATIVE_FILES = tuple(DATASETS / 'ipfs_datasets_py/optimizers/logic_theorem_optimizer' / (name + '.py') for name in NATIVE_NAMES)
PYTHON, GIT = Path('/usr/bin/python3.12'), Path('/usr/bin/git')
DAEMON = ACCELERATOR / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
IDENTITY = ACCELERATOR / 'ipfs_accelerate_py/agent_supervisor/task_sources/task_identity.py'
REPORT = 'data/legal_ir_span_graph_lean_20261016/LIR-9001/handoff-reconciliation.json'
BOARD = 'docs/architecture/legal_ir_supervisor_start_20261016/tasks.todo.md'
FIELDS = {'schema', 'name', 'scope', 'run_root', 'scheduler_state_path', 'expected_scheduler_policy_sha256',
          'ledger_path', 'roots', 'source_sha256', 'board_source_sha256', 'observer_source_sha256'}
FALSE = dict(provider_calls_performed=False, actual_provider_outcome_observed=False,
             native_validation_qualified=False, real_LIR_task_accepted=False,
             acceptance_authority=False, completion_authority=False, dispatch_authority=False,
             canonical_tasks_accepted=0, canonical_attempt_counters_changed=False)
FIXTURE_REPORT = {'schema': 'legal-ir-explicit-candidate-observer-metadata-fixture@1',
                  'scope': 'synthetic_metadata_correspondence_only',
                  'text': 'This JSON is not a provider-produced Legal IR reconciliation or accepted legal result.',
                  **FALSE}
_BARRIER = """import json,os,resource,select,sys
if not select.select([0],[],[],180)[0] or os.read(0,2)!=b'G': raise SystemExit(125)
argv=json.loads(sys.argv[1])
if type(argv) is not list or not argv or argv[0]!='/usr/bin/git': raise SystemExit(125)
for k,v in ((resource.RLIMIT_CPU,15),(resource.RLIMIT_AS,268435456),(resource.RLIMIT_FSIZE,4194304),(resource.RLIMIT_NOFILE,64)):
 resource.setrlimit(k,(v,v))
os.execv(argv[0],argv)
"""


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode() + b'\n'


def canonical_path(spelling):
    path = Path(spelling)
    need(type(spelling) is str and path.is_absolute() and os.path.abspath(path) == spelling
         and '..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
         'canonical unaliased absolute path required')
    return path


def stable_source(path, maximum=32 * 1024 * 1024):
    path = canonical_path(str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 < before.st_size <= maximum,
             'bounded owned pinned source required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
                          s.st_mode, s.st_uid, s.st_gid, s.st_nlink)
    need(len(raw) == before.st_size and identity(before) == identity(after) == identity(path.lstat()),
         'source changed while reading')
    return raw


def read_request(path, expected_sha):
    raw = stable_source(path, 1024 * 1024)
    need(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha) and sha(raw) == expected_sha,
         'frozen qualification request SHA differs')
    def pairs(items):
        output = {}
        for key, value in items:
            need(key not in output, 'duplicate request key')
            output[key] = value
        return output
    request = json.loads(raw, object_pairs_hook=pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite request')))
    need(type(request) is dict and set(request) == FIELDS
         and request['schema'] == 'legal-ir-native-candidate-observer-qualification-request@1'
         and request['scope'] == 'synthetic_metadata_fixture_actual_native_git_observation_only',
         'exact explicit fixture request required')
    need(type(request['name']) is str and re.fullmatch('legal-ir-candidate-observer-[a-z0-9][a-z0-9-]{0,80}', request['name']),
         'finite unique fixture name required')
    run_root = canonical_path(request['run_root'])
    need(run_root.name == request['name'] and ROOTS[0] in run_root.parents,
         'fresh fixture below actual owned test logs required')
    need(request['scheduler_state_path'] == str(STATE) and request['expected_scheduler_policy_sha256'] == POLICY
         and request['ledger_path'] == str(LEDGER) and request['roots'] == [str(p) for p in ROOTS],
         'original native scheduler and storage authority required')
    pins = request['source_sha256']
    required = {str(Path(__file__).absolute()), str(OBSERVER), str(BOARD_SOURCE), str(DAEMON), str(IDENTITY),
                str(PYTHON), str(GIT), *(str(p) for p in NATIVE_FILES)}
    need(type(pins) is dict and required <= set(pins) and len(pins) <= 256, 'complete finite source closure required')
    for spelling, digest in pins.items():
        need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest)
             and sha(stable_source(canonical_path(spelling))) == digest, 'frozen implementation source pin differs')
    need(request['board_source_sha256'] == pins[str(BOARD_SOURCE)]
         and request['observer_source_sha256'] == pins[str(OBSERVER)], 'explicit board/observer pins differ')
    return request, raw


def write_new(path, raw):
    need(type(raw) is bytes and 0 < len(raw) <= 4 * 1024 * 1024, 'bounded fixture output required')
    path = canonical_path(str(path))
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o644)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root):
    for directory, names, files in os.walk(root, followlinks=False):
        path = Path(directory)
        need(not any((path / name).is_symlink() for name in (*names, *files)), 'fixture Git tree must not contain symlinks')
        for name in files:
            descriptor = os.open(path / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                need(stat.S_ISREG(os.fstat(descriptor).st_mode), 'regular fixture Git file required')
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(path)


def guard_owner(owner, *, full=False):
    need(owner.heartbeat(), 'actual fixture owner heartbeat failed')
    usage = owner.check_usage(owner.directory, full=full)
    need(usage['group_rss']['live_processes'] == 0, 'prior actual fixture Git child is not quiescent')


def setup_git(owner, repository, arguments, source_guard):
    source_guard()
    guard_owner(owner)
    argv = [str(GIT), '--no-optional-locks', '--no-replace-objects', '-c', 'core.hooksPath=/dev/null',
            '-c', 'core.fsmonitor=false', '-c', 'core.untrackedCache=false', '-c', 'commit.gpgsign=false',
            '-c', 'tag.gpgsign=false', '-c', 'user.name=Legal IR Metadata Fixture',
            '-c', 'user.email=legal-ir-metadata-fixture@localhost', *arguments]
    environment = {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
        'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_TERMINAL_PROMPT': '0',
        'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_NO_LAZY_FETCH': '1',
        'GIT_LITERAL_PATHSPECS': '1', 'GIT_LFS_SKIP_SMUDGE': '1'}
    process = subprocess.Popen([str(PYTHON), '-I', '-S', '-B', '-c', _BARRIER, json.dumps(argv)],
        cwd=repository, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    selector, buffers = selectors.DefaultSelector(), {'stdout': bytearray(), 'stderr': bytearray()}
    try:
        owner.check_usage(owner.directory, child_pid=process.pid)
        need(owner.heartbeat(), 'actual Git attachment heartbeat failed')
        process.stdin.write(b'G')
        process.stdin.close()
        for name in buffers:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        deadline, next_sample = time.monotonic() + 20, 0
        while selector.get_map() or process.poll() is None:
            need(time.monotonic() < deadline, 'actual fixture Git productive deadline exceeded')
            if time.monotonic() >= next_sample:
                owner.check_usage(owner.directory)
                need(owner.heartbeat(), 'actual fixture Git heartbeat failed')
                next_sample = time.monotonic() + 0.5
            for key, _ in selector.select(0.1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffers[key.data].extend(chunk)
                need(len(buffers[key.data]) <= 65536, 'bounded fixture Git output exceeded')
        need(process.wait(timeout=1) == 0, 'actual fixture Git command failed: ' + bytes(buffers['stderr']).decode(errors='replace'))
        source_guard()
        guard_owner(owner)
        return bytes(buffers['stdout'])
    finally:
        try:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        finally:
            selector.close()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
            guard_owner(owner)


def main():
    need(len(sys.argv) == 3, 'usage: qualify_candidate_observer.py ABSOLUTE_REQUEST EXPECTED_REQUEST_SHA256')
    sys.dont_write_bytecode = True
    request_path = canonical_path(sys.argv[1])
    request, request_raw = read_request(request_path, sys.argv[2])
    run_root = Path(request['run_root'])
    need(not run_root.exists(), 'fresh qualification directory required; never replay')
    def source_guard():
        checked, raw = read_request(request_path, sys.argv[2])
        need(raw == request_raw and checked == request, 'qualification request changed')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', IPFS_DATASETS_PY_MINIMAL_IMPORTS='1', IPFS_ACCEL_SKIP_CORE='1',
        IPFS_AUTO_INSTALL='false', IPFS_DATASETS_AUTO_INSTALL='false', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path[:0] = [str(DATASETS), str(ACCELERATOR)]
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import existing_resource_scheduler, parented_phase_resources
    for module in (existing_resource_scheduler, parented_phase_resources):
        actual = Path(module.__file__).absolute()
        need(request['source_sha256'].get(str(actual)) == sha(stable_source(actual)), 'actual imported native implementation differs')
    source_guard()
    scheduler = existing_resource_scheduler.open_existing_scheduler(STATE, POLICY)
    def interrupted(_number, _frame):
        raise KeyboardInterrupt('finite owned candidate observer fixture interrupted')
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupted)
    # Real owner APIs require existing empty attempt directories. No productive
    # file bytes are written during this empty bootstrap.
    run_root.mkdir(mode=0o700)
    preparation, observation_attempt = run_root / 'preparation', run_root / 'observation'
    preparation.mkdir(mode=0o700)
    observation_attempt.mkdir(mode=0o700)
    repository = preparation / 'repository'
    baseline = candidate = ''
    with scheduler.acquire('orchestration', cpu_slots=1, memory_mb=1024, child_process_slots=1,
            requires_gpu=False, timeout=60, request_id=request['name'] + ':root') as root:
        with root.acquire_child(cpu_slots=1, memory_mb=1024, child_process_slots=1, requires_gpu=False,
                timeout=60, request_id=request['name'] + ':phase') as phase:
            arguments = dict(roots=ROOTS, parent_lease=phase, memory_mb=1024, cpu_slots=1,
                child_process_slots=1, scheduler_state_path=STATE, expected_scheduler_policy_sha256=POLICY,
                timeout_seconds=2400, ledger_lock_timeout_seconds=None)
            with parented_phase_resources.ParentedPhaseResourceReservation(LEDGER, attempt_directory=preparation,
                    storage_bytes=32 * 1024 * 1024, **arguments) as preparer:
                source_guard()
                guard_owner(preparer, full=True)
                repository.mkdir(mode=0o700)
                board_raw = stable_source(BOARD_SOURCE, 4 * 1024 * 1024)
                need(sha(board_raw) == request['board_source_sha256'], 'frozen64-task board changed')
                write_new(repository / BOARD, board_raw)
                git = lambda argv: setup_git(preparer, repository, argv, source_guard)
                git(['init', '--initial-branch=metadata-fixture'])
                git(['add', '--force', '--', BOARD])
                git(['commit', '--no-gpg-sign', '-m', 'Explicit metadata fixture baseline board'])
                baseline = git(['rev-parse', '--verify', 'HEAD^{commit}']).decode().strip()
                report_raw = wire(FIXTURE_REPORT)
                write_new(repository / REPORT, report_raw)
                git(['add', '--force', '--', REPORT])
                git(['commit', '--no-gpg-sign', '-m', 'Explicit report-only metadata fixture candidate'])
                candidate = git(['rev-parse', '--verify', 'HEAD^{commit}']).decode().strip()
                need(re.fullmatch('[0-9a-f]{40}', baseline) and re.fullmatch('[0-9a-f]{40}', candidate)
                     and baseline != candidate, 'actual distinct Git commits required')
                fsync_tree(repository)
                write_new(preparation / 'fixture-before-closeout.json', wire({
                    'schema': 'legal-ir-candidate-observer-fixture-preparation@1', 'scope': request['scope'],
                    'request_sha256': sha(request_raw), 'board_source_sha256': sha(board_raw),
                    'report_sha256': sha(report_raw), 'repository': str(repository),
                    'baseline_commit': baseline, 'candidate_commit': candidate,
                    'actual_setup_git_children': 7, 'actual_preparation_closeout_pending': True, **FALSE}))
                source_guard()
                guard_owner(preparer, full=True)
                prep_final = preparer.finalize(preparation, artifacts_durable=True)
                need(prep_final['status'] == 'released' and prep_final['resource_lease']['released'] is True,
                     'actual preparation owner did not finalize')
            # The independent read-only observer requires a new owner with no
            # previous child history. The original prep owner is already closed.
            with parented_phase_resources.ParentedPhaseResourceReservation(LEDGER, attempt_directory=observation_attempt,
                    storage_bytes=16 * 1024 * 1024, **arguments) as observer_owner:
                source_guard()
                guard_owner(observer_owner, full=True)
                temporary = observation_attempt / 'tmp'
                temporary.mkdir(mode=0o700)
                os.environ.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
                                  XDG_CACHE_HOME=str(temporary))
                specification = importlib.util.spec_from_file_location('legal_ir_candidate_observer_qualification_v03', OBSERVER)
                module = importlib.util.module_from_spec(specification)
                specification.loader.exec_module(module)
                need(Path(module.__file__).absolute() == OBSERVER and module.SCHEMA == 'legal-ir-real-candidate-binding-observation@3',
                     'reviewed actual observer successor required')
                observer_pins = {str(path): request['source_sha256'][str(path)] for path in module.required_source_paths()}
                observation = module.observe_real_candidate_binding(preparation_owner=observer_owner,
                    repository_root=repository, board_path=repository / BOARD,
                    expected_baseline_commit=baseline, expected_candidate_commit=candidate,
                    expected_board_sha256=sha(board_raw), expected_report_sha256=sha(report_raw),
                    protected_source_sha256=observer_pins)
                need(observation['report_only_diff_verified'] is True and observation['worktree_clean_observed'] is True
                     and observation['git_child_count'] == 10 and observation['observation_only'] is True
                     and all(observation[key] is False for key in module.NO_AUTHORITY),
                     'actual observation returned unsupported authority or incomplete binding')
                need(prep_final['resource_lease']['released'] is True,
                     'preparation closure must precede independent observation')
                write_new(observation_attempt / 'actual-fixture-observation.json', wire({
                    'schema': 'legal-ir-actual-candidate-observer-fixture-observation@1',
                    'scope': request['scope'], 'request_sha256': sha(request_raw),
                    'synthetic_metadata_fixture': True, 'actual_preparation_finalized_before_observation': True,
                    'observation': observation, 'actual_observer_closeout_pending': True, **FALSE}))
                source_guard()
                guard_owner(observer_owner, full=True)
                observer_final = observer_owner.finalize(observation_attempt, artifacts_durable=True)
                need(observer_final['status'] == 'released' and observer_final['resource_lease']['released'] is True,
                     'actual observer owner did not finalize')
            source_guard()
            need(root.renew() and phase.renew(), 'actual native root/phase final heartbeat failed')
            ids = {'root_lease_id': root.lease_id, 'phase_lease_id': phase.lease_id}
    print(json.dumps({'schema': 'legal-ir-native-candidate-observer-qualification-terminal@1',
        'passed': True, 'scope': request['scope'], 'request_sha256': sha(request_raw),
        'synthetic_metadata_fixture': True, 'actual_public_observer_and_git_qualified': True,
        'actual_preparation_finalized_before_observation': True, 'actual_preparation_finalized': True,
        'actual_observation_finalized': True, 'actual_root_released': root.released,
        'actual_phase_released': phase.released, 'actual_setup_git_children': 7,
        'actual_observer_git_children': 10, 'baseline_commit': baseline, 'candidate_commit': candidate,
        'board_sha256': sha(board_raw), 'report_sha256': sha(report_raw), **ids, **FALSE}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({'schema': 'legal-ir-native-candidate-observer-qualification-terminal@1',
            'passed': False, 'error_type': type(error).__name__, 'synthetic_metadata_fixture': True,
            'actual_public_observer_and_git_qualified': False, **FALSE}, sort_keys=True), flush=True)
        raise
