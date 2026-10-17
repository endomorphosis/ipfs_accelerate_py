"""Cancel one exact owned waiting nonmodel fixture; never issue native authority.

The genuine V05 parent owns admission/storage. This child only observes redacted
state and signals one freshly captured PIDFD after all guards. No provider,
container, task, lease mutation, attempt reset, escalation, or process-tree kill.
"""
from pathlib import Path
import hashlib
import importlib
import json
import os
import re
import select
import signal
import stat
import sys
import time

NEW = Path('/home/barberb/lift_coding/artifacts/legal-ir-callable-handoff-20261017-01')
BUNDLE = NEW / 'pending-fixture-cancellation-v01'
REQUEST = BUNDLE / 'request-v01.json'
RUNS = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs')
ATTEMPT = RUNS / 'legal-ir-pending-wrapper-cancellation-20261017-r01'
TARGET_RUN = RUNS / 'legal-ir-callable-handoff-20261017-r02/public-wrapper-prestart'
TARGET_REQUEST = NEW / 'public-wrapper-prestart-v01/request-r02.json'
TARGET_REQUEST_SHA = 'fb54a1a1fd537b3b3bf5d09970a2f9eb52c2cb72a2b4275351819c1578c0e6b5'
TARGET_PID = 1812898
TARGET_BIRTH = 125964024
TARGET_CMD_SHA = 'bc29d987f18e1c6c620b5d36149681f6f02e94282a7d51e1ee11e1bc2402784e'
TARGET_NAME = 'legal-ir-callable-public-wrapper-prestart-r02'
NAME = 'legal-ir-pending-wrapper-cancellation-r01'
DATASETS = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets')
STATE = Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
POLICY = 'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
PYTHON = Path('/usr/bin/python3.12')


def need(value, message):
    if not value:
        raise RuntimeError(message)


def pairs(rows):
    result = {}
    for key, value in rows:
        need(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def regular(path, limit=8 * 1024 * 1024):
    path = Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'source_alias_refused')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
             and before.st_nlink == 1 and not before.st_mode & 0o002
             and 0 < before.st_size <= limit, 'bounded_owned_regular_input_required')
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(1024 * 1024, limit + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            need(size <= limit, 'input_limit_exceeded')
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(identity(before) == identity(after) == identity(path.lstat()) and size == before.st_size,
             'input_changed_while_reading')
        return b''.join(chunks)
    finally:
        os.close(fd)


def source_guard(request, manifest_sha):
    need(hashlib.sha256(regular(BUNDLE / 'SOURCE_MANIFEST.json')).hexdigest() == manifest_sha,
         'source_manifest_changed')
    pins = request['source_sha256']
    need(type(pins) is dict and 16 <= len(pins) <= 128, 'bounded_complete_source_closure_required')
    for name, digest in pins.items():
        need(re.fullmatch('[0-9a-f]{64}', digest) is not None, 'source_digest_required')
        # Stream the inert Grok binary rather than retaining its bytes in RAM.
        path = Path(name)
        need(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'source_alias_refused')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.getuid())
                 and before.st_nlink == 1 and not before.st_mode & 0o002
                 and 0 < before.st_size <= 256 * 1024 * 1024, 'source_mode_or_size_refused')
            actual, size = hashlib.sha256(), 0
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                actual.update(block)
                size += len(block)
            after = os.fstat(fd)
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            need(identity(before) == identity(after) == identity(path.lstat())
                 and size == before.st_size and actual.hexdigest() == digest, 'protected_source_changed')
        finally:
            os.close(fd)


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    text = (proc / 'stat').read_text()
    fields = text.rsplit(')', 1)[1].split()
    need(fields[0] != 'Z' and proc.stat().st_uid == os.getuid(), 'live_same_uid_process_required')
    return dict(pid=pid, birth=int(fields[19]), parent_pid=int(fields[1]),
        cmd_sha256=hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest(),
        exe=str((proc / 'exe').resolve(strict=True)))


def target_identity():
    row = process_identity(TARGET_PID)
    need(row['birth'] == TARGET_BIRTH and row['cmd_sha256'] == TARGET_CMD_SHA
         and row['exe'] == str(PYTHON), 'exact_target_identity_changed')
    need((Path('/proc') / str(TARGET_PID) / 'task' / str(TARGET_PID) / 'children').read_text().strip() == '',
         'target_has_actual_children')
    need(not os.path.lexists(TARGET_RUN), 'target_already_entered_native_run')
    return row


def waiter_projection():
    # Native writes use atomic rename. Retry replaced snapshots; never export the
    # raw state or authority keys. No private disk ledger or native handle read.
    for _ in range(4):
        try:
            state = decode(regular(STATE))
            need(type(state) is dict and type(state.get('waiters')) is dict, 'scheduler_waiter_shape_changed')
            fields = ('request_id', 'owner_pid', 'owner_birth_marker', 'lane', 'child_process_slots', 'parent_lease_id')
            return [{key: row.get(key) for key in fields} for row in state['waiters'].values()
                    if row.get('owner_pid') == TARGET_PID or row.get('request_id') == TARGET_NAME]
        except (OSError, RuntimeError):
            time.sleep(.025)
    raise RuntimeError('stable_native_waiter_projection_unavailable')


def parent_and_target(scheduler, request_sha, *, waiting):
    parent = process_identity(os.getppid())
    receipt = decode(regular(ATTEMPT / 'receipt.json'))
    need(receipt.get('schema') == 'legal-ir-native-orchestration-receipt@1'
         and receipt.get('owner_pid') == parent['pid'] and receipt.get('child_pid') == os.getpid()
         and receipt.get('child_pgid') == os.getpid() and receipt.get('status') == 'running'
         and receipt.get('request_sha256') == request_sha, 'actual_generic_parent_receipt_join_required')
    snapshot = scheduler.snapshot()
    need(snapshot['capacity']['child_process_slots'] == 16, 'unchanged_global_16_slots_required')
    rows = scheduler.active_leases()
    by_id = {row['lease_id']: row for row in rows}
    root = by_id.get(receipt['root_lease_id'], {})
    phase = by_id.get(receipt['phase_lease_id'], {})
    leaves = [row for row in rows if row.get('request_id') == 'daemon:' + receipt['disk_reservation_id']]
    need(len(leaves) == 1, 'actual_parent_disk_native_leaf_required')
    leaf = leaves[0]
    need(root.get('request_id') == NAME and not root.get('parent_lease_id')
         and phase.get('request_id') == NAME + ':phase' and phase.get('parent_lease_id') == root.get('lease_id')
         and leaf.get('parent_lease_id') == phase.get('lease_id'), 'actual_generic_parent_ancestry_required')
    for row in (root, phase, leaf):
        need(row.get('owner_pid') == parent['pid'] and row.get('owner_birth_marker') == str(parent['birth'])
             and row.get('cpu_slots') == 1 and row.get('memory_mb') == 4096
             and row.get('child_process_slots') == 8 and row.get('lane') == 'orchestration'
             and row.get('cancelled') is False and row.get('expires_at', 0) > time.time(),
             'live_actual_parent_eight_slot_reservation_required')
    need(not any(row.get('owner_pid') == TARGET_PID or row.get('request_id') == TARGET_NAME for row in rows),
         'target_actual_native_lease_present')
    waiters = waiter_projection()
    if waiting:
        need(len(waiters) == 1 and waiters[0] == dict(request_id=TARGET_NAME, owner_pid=TARGET_PID,
             owner_birth_marker=str(TARGET_BIRTH), lane='orchestration', child_process_slots=16, parent_lease_id=None),
             'exact_unadmitted_target_waiter_required')
    need(not os.path.lexists(TARGET_RUN), 'target_run_must_remain_absent')
    return dict(parent_pid=parent['pid'], root_lease_id=root['lease_id'], phase_lease_id=phase['lease_id'],
        native_leaf_id=leaf['lease_id'], child_process_slots=8, global_child_process_slots=16,
        target_active_native_leases=0, target_waiter_count=len(waiters), target_run_absent=True)


def save(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) <= 128 * 1024, 'result_bound_exceeded')
    fd = os.open(ATTEMPT / 'cancellation-result.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(ATTEMPT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    need(len(sys.argv) == 2 and re.fullmatch('[0-9a-f]{64}', sys.argv[1]), 'expected_manifest_sha_required')
    request_raw = regular(REQUEST)
    request_sha = hashlib.sha256(request_raw).hexdigest()
    request = decode(request_raw)
    need(request['schema'] == 'legal-ir-native-orchestration-request@1' and request['name'] == NAME
         and request['purpose'] == 'controller_preparation' and request['duration_seconds'] == 600
         and request['attempt_directory'] == str(ATTEMPT) and request['storage_bytes'] == 512 * 1024 * 1024
         and request['external_charges'] == [] and request['command'] == [str(PYTHON), '-B', str(Path(__file__).absolute()), sys.argv[1]],
         'exact_one_target_controller_request_required')
    need(os.environ.get('TMPDIR') == str(ATTEMPT) and ATTEMPT.stat().st_uid == os.getuid()
         and stat.S_IMODE(ATTEMPT.stat().st_mode) == 0o700, 'genuine_private_parent_attempt_required')
    source_guard(request, sys.argv[1])
    target_request = decode(regular(TARGET_REQUEST))
    need(hashlib.sha256(regular(TARGET_REQUEST)).hexdigest() == TARGET_REQUEST_SHA
         and target_request['name'] == TARGET_NAME and target_request['root_and_phase_envelope']['child_process_slots'] == 16
         and target_request['run_directory'] == str(TARGET_RUN)
         and target_request['provider_calls_performed'] is False and target_request['Docker_operations_performed'] is False,
         'exact_original_waiting_nonmodel_request_required')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', IPFS_DATASETS_PY_MINIMAL_IMPORTS='1', IPFS_AUTO_INSTALL='false',
        IPFS_DATASETS_AUTO_INSTALL='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(DATASETS))
    module = importlib.import_module('ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler')
    need(Path(module.__file__).absolute() == DATASETS / 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/existing_resource_scheduler.py',
         'actual_pinned_existing_scheduler_required')
    scheduler = module.open_existing_scheduler(STATE, POLICY)
    result = dict(schema='legal-ir-pending-wrapper-cancellation-result@1', passed=False, target_count=1,
        target_pid=TARGET_PID, target_birth=TARGET_BIRTH, target_cmdline_sha256=TARGET_CMD_SHA,
        target_request_sha256=TARGET_REQUEST_SHA, source_request_sha256=request_sha, source_manifest_sha256=sys.argv[1],
        signal_number=int(signal.SIGINT), signals_sent=0, provider_calls_performed=False,
        Docker_operations_performed=False, canonical_tasks_accepted=0, acceptance_authority=False,
        native_finalization_observed_by_child=False, scope='owned_waiting_nonmodel_engineering_cancellation')
    pidfd = None
    try:
        target_identity()
        pidfd = os.pidfd_open(TARGET_PID, 0)
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        need(not poller.poll(0), 'target_already_terminated')
        # V05 attaches this actual child before publishing its running receipt.
        until = time.monotonic() + 10
        while decode(regular(ATTEMPT / 'receipt.json')).get('status') != 'running':
            need(time.monotonic() < until, 'parent_attach_receipt_not_ready')
            time.sleep(.05)
        parent_and_target(scheduler, request_sha, waiting=True)
        source_guard(request, sys.argv[1])
        need(regular(REQUEST) == request_raw, 'reviewed_controller_request_changed')
        result['pre_signal_observation'] = parent_and_target(scheduler, request_sha, waiting=True)
        result['pre_signal_process'] = target_identity()
        need(not poller.poll(0), 'target_terminated_before_signal')
        signal.pidfd_send_signal(pidfd, signal.SIGINT, None, 0)
        result['signals_sent'] = 1
        until = time.monotonic() + 60
        while True:
            final = parent_and_target(scheduler, request_sha, waiting=False)
            gone = bool(poller.poll(0))
            if gone and final['target_waiter_count'] == 0:
                result.update(passed=True, pidfd_terminal_readable=True, post_signal_observation=final)
                break
            need(time.monotonic() < until, 'single_signal_observation_timeout_no_escalation')
            time.sleep(.1)
    except Exception as error:
        result['failure_type'] = type(error).__name__
        result['failure_reason'] = str(error)[:256]
    finally:
        if pidfd is not None:
            os.close(pidfd)
    save(result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
