"""Finite, explicitly selected failed-fixture closeout after original V05 admission.

Only captured pytest aliases are removed. Regular evidence is preserved. The
unchanged public recovery API alone updates the selected retained disk claim.
"""
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import time

HERE = Path(__file__).absolute().parent
REQUEST = HERE / 'request-r01.json'
NAME = 'legal-ir-failed-fixture-recovery-r01'
ATTEMPT = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-failed-fixture-recovery-20261017-r01')
OLD = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-attempt-disposition-controls-20261017-r02')
DATASETS = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets')
BASE = Path('/home/barberb/lift_coding/external/ipfs_datasets')
ROOTS = (BASE/'workspace/test-logs', BASE/'workspace/todo-queues', BASE/'docs/implementation/reports/evidence', Path('/tmp/pytest-of-barberb'))
LEDGER = BASE/'workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
GATE = Path('/home/barberb/lift_coding/artifacts/legal-ir-daemon-settlement-20261017-01/repository-fixture-v02/parent_attachment_gate.py')
GATE_SHA = 'dba5662443619358e2225d0cf7ccaa2e2b6c5bf51b0a60d2d1809004d1667fb3'
RESERVATION = '0c6581ac42d74eb3a181ab5094680658'
RECORD_SHA = '3672394aedfcb7ffdc026c19acff92a7c5c0626b2d2017deaf4237a6c3ee05f1'
FAILURE_OBSERVATION = HERE.parent/'attempt-controls-failed-finalization-root-observation-r02.json'
STORAGE = 32*1024*1024
MAXIMUM_ENTRIES = 4096
MAXIMUM_FILE = 16*1024*1024


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'duplicate_JSON_key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_JSON')))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def witness(info):
    return {key: getattr(info, 'st_'+key) for key in
        ('dev', 'ino', 'mode', 'uid', 'gid', 'nlink', 'size', 'mtime_ns', 'ctime_ns')}


def directories(path):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path), 'canonical_absolute_path_required')
    for current in (*reversed(path.parents), path):
        need(stat.S_ISDIR(current.lstat().st_mode), 'real_directory_ancestry_required')
    return path


def stable(path, maximum=MAXIMUM_FILE, *, source=False):
    path = Path(path)
    directories(path.parent)
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_uid in {0, os.getuid()} and 0 <= before.st_size <= maximum,
             'bounded_owned_unaliased_regular_file_required')
        if source:
            need(not before.st_mode & 0o002, 'world_writable_source_refused')
        raw = stream.read(maximum+1)
        after = os.fstat(stream.fileno())
    need(len(raw) == before.st_size
         and witness(before) == witness(after) == witness(path.lstat()), 'stable_regular_file_required')
    return raw, witness(before)


def root_identity(path):
    path = directories(path)
    value = path.lstat()
    return {'path':str(path), 'device':value.st_dev, 'inode':value.st_ino}


def stopped():
    # PID reuse also refuses. Never signal or reap another owner's processes.
    for pid in (740900, 750376):
        try:
            (Path('/proc')/str(pid)).lstat()
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError('original_owner_or_child_PID_present')
    entries = list(Path('/proc').iterdir())
    need(len(entries) <= 100000, 'bounded_proc_inventory_required')
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            descriptor = os.open(entry/'stat', os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            raw = os.read(descriptor, 8193)
        finally:
            os.close(descriptor)
        need(len(raw) <= 8192 and b')' in raw, 'bounded_proc_stat_required')
        fields = raw.rsplit(b')', 1)[1].split()
        need(len(fields) >= 22 and int(fields[2]) != 750376,
             'original_child_group_member_present')


def selected_record():
    for retry in range(8):
        try:
            raw, _ = stable(LEDGER, 8*1024*1024)
            ledger = strict(raw)
            need(ledger.get('limit_bytes') == 160000000000
                 and [row['path'] for row in ledger['roots']] == [str(x) for x in ROOTS],
                 'original_disk_ledger_scope_required')
            row = ledger['reservations'][RESERVATION]
            need(type(row) is dict and row.get('reservation_id') == RESERVATION,
                 'exact_selected_reservation_required')
            return row
        except (OSError, ValueError, RuntimeError):
            if retry == 7:
                raise
            time.sleep(.05)


def selected_retained(capture):
    row = selected_record()
    need(row.get('status') == 'retained' and digest(row) == RECORD_SHA
         and row.get('owner_pid') == 740900
         and row.get('child') == {'pid':750376, 'birth':'127615740'}
         and row.get('prior_children') == [] and row.get('external_charges') == {}
         and row.get('attempt_directory') == capture['old_attempt_identity']
         and row.get('storage_bytes') == 128*1024*1024,
         'exact_original_retained_record_changed')


def inventory(capture, *, aliases_removed):
    need(root_identity(OLD) == capture['old_attempt_identity'], 'old_attempt_identity_changed')
    actual_files, actual_dirs, actual_links = [], [], []
    pending = [OLD]
    entries = 0
    total = 0
    while pending:
        directory = directories(pending.pop())
        actual_dirs.append({'path':directory.relative_to(OLD).as_posix(), 'metadata':witness(directory.lstat())})
        with os.scandir(directory) as stream:
            for entry in stream:
                entries += 1
                need(entries <= MAXIMUM_ENTRIES, 'bounded_old_attempt_inventory_required')
                path = Path(entry.path)
                relative = path.relative_to(OLD).as_posix()
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    need(not aliases_removed and path.parent == OLD/'pytest', 'unexpected_old_alias')
                    target = os.readlink(path)
                    actual_links.append({'path':relative, 'target':target, 'metadata':witness(info),
                        'target_directory':root_identity(Path(target))})
                elif stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    raw, metadata = stable(path)
                    total += len(raw)
                    need(total <= 128*1024*1024, 'old_inventory_byte_bound_exceeded')
                    actual_files.append({'path':relative, 'bytes':len(raw),
                        'sha256':hashlib.sha256(raw).hexdigest(), 'metadata':metadata})
                else:
                    raise RuntimeError('unexpected_old_special_entry')
    key = lambda row:row['path']
    need(sorted(actual_files, key=key) == capture['regular_files'], 'old_regular_evidence_changed')
    need(sorted(actual_links, key=key) == ([] if aliases_removed else capture['exact_framework_aliases']),
         'exact_captured_framework_aliases_changed')
    expected_dirs = capture['directories']
    need(len(actual_dirs) == len(expected_dirs), 'old_directory_population_changed')
    for actual, expected in zip(sorted(actual_dirs, key=key), expected_dirs):
        need(actual['path'] == expected['path'], 'old_directory_name_changed')
        if aliases_removed and actual['path'] == 'pytest':
            fields = ('dev','ino','mode','uid','gid','nlink')
            need(all(actual['metadata'][x] == expected['metadata'][x] for x in fields),
                 'retained_pytest_directory_identity_changed')
        else:
            need(actual == expected, 'old_directory_metadata_changed')
    return [{'path':row['path'], 'bytes':row['bytes'], 'sha256':row['sha256']}
        for row in sorted(actual_files, key=key)]


def fsync_directory(path):
    directories(path)
    descriptor = os.open(path, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(name, value, maximum=1024*1024):
    need(re.fullmatch(r'[a-z0-9-]+\.json', name), 'fixed_output_name_required')
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    need(len(raw) <= maximum, 'bounded_new_attempt_output_required')
    target, temporary = ATTEMPT/name, ATTEMPT/(name+'.partial')
    directories(ATTEMPT)
    descriptor = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        current = target.lstat()
    except FileNotFoundError:
        current = None
    if current is not None:
        need(stat.S_ISREG(current.st_mode) and current.st_nlink == 1
             and current.st_uid == os.getuid(), 'owned_output_identity_required')
    os.replace(temporary, target)
    fsync_directory(ATTEMPT)


def main():
    need(len(sys.argv) == 2 and re.fullmatch('[0-9a-f]{64}', sys.argv[1]),
         'exact_source_manifest_argument_required')
    expected = sys.argv[1]
    sys.dont_write_bytecode = True
    raw, _ = stable(HERE/'SOURCE_MANIFEST.json', 1024*1024, source=True)
    need(hashlib.sha256(raw).hexdigest() == expected, 'source_manifest_changed')
    manifest = strict(raw)
    need(manifest.get('schema') == 'legal-ir-failed-fixture-recovery-source@1', 'exact_source_generation_required')
    guards = {str(HERE/'SOURCE_MANIFEST.json'):expected}
    names = set()
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        need(not relative.is_absolute() and '..' not in relative.parts
             and relative.as_posix() == row['path'] and row['path'] not in names,
             'unique_relative_source_required')
        names.add(row['path'])
        data, _ = stable(HERE/row['path'], source=True)
        need(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
             'exact_source_row_changed')
        guards[str(HERE/row['path'])] = row['sha256']
    guards.update(manifest['input_pins'])
    def guard():
        for spelling, wanted in guards.items():
            data, _ = stable(spelling, 64*1024*1024, source=True)
            need(hashlib.sha256(data).hexdigest() == wanted, 'complete_source_closure_changed')
    guard()
    need(hashlib.sha256(stable(GATE, source=True)[0]).hexdigest() == GATE_SHA,
         'original_parent_gate_changed')
    gate_spec = importlib.util.spec_from_file_location('actual_original_parent_attachment_gate', GATE)
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    def joined():
        return gate.wait_for_original_generic_parent(REQUEST, attempt_directory=ATTEMPT,
            expected_request_name=NAME, storage_bytes=STORAGE,
            worker_command=['/usr/bin/python3.12','-B',str(HERE/'recover_fixture.py'),expected],
            timeout_seconds=60)
    attachment = joined()
    # No old-tree effects, product import or recovery operation occurs before this join.
    need(Path(os.environ['TMPDIR']) == ATTEMPT, 'fresh_original_charged_attempt_required')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',
        IPFS_AUTO_INSTALL='false', IPFS_DATASETS_AUTO_INSTALL='false')
    capture = strict(stable(HERE/'OLD_ATTEMPT_CAPTURE.json', source=True)[0])
    need(capture.get('reservation_id') == RESERVATION
         and capture.get('expected_record_sha256') == RECORD_SHA
         and capture.get('old_attempt') == str(OLD) and capture.get('framework_alias_count') == 28
         and capture.get('regular_file_count') == 283, 'exact_old_capture_required')
    removed = []
    stage = 'verify_old_capture'
    try:
        stopped()
        selected_retained(capture)
        inventory(capture, aliases_removed=False)
        guard()
        joined()
        save('old-attempt-capture.json', capture)
        failure_raw, _ = stable(FAILURE_OBSERVATION, source=True)
        failure = strict(failure_raw)
        need(failure.get('actual_parent_finalized') is False
             and failure.get('qualification_passed') is False
             and failure.get('worker_pytest_passed') is True
             and failure.get('worker_testcases') == 33
             and failure.get('selected_record_sha256') == RECORD_SHA,
             'exact_original_failed_native_observation_required')
        save('original-native-failure.json', failure)
        pytest_dir = OLD/'pytest'
        descriptor = os.open(pytest_dir, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            identity = os.fstat(descriptor)
            captured_parent = next(row['metadata'] for row in capture['directories'] if row['path'] == 'pytest')
            need(stat.S_ISDIR(identity.st_mode) and stat.S_IMODE(identity.st_mode) == 0o700
                 and identity.st_uid == os.getuid()
                 and all(witness(identity)[key] == captured_parent[key]
                         for key in ('dev','ino','mode','uid','gid','nlink')),
                 'captured_private_pytest_parent_required')
            need(root_identity(pytest_dir) == {'path':str(pytest_dir), 'device':identity.st_dev,
                 'inode':identity.st_ino}, 'exact_pytest_directory_fd_required')
            for row in capture['exact_framework_aliases']:
                stage = 'remove_exact_framework_alias'
                stopped()
                need(root_identity(pytest_dir) == {'path':str(pytest_dir), 'device':identity.st_dev,
                     'inode':identity.st_ino}
                     and all(witness(os.fstat(descriptor))[key] == captured_parent[key]
                             for key in ('dev','ino','mode','uid','gid','nlink')),
                     'captured_pytest_parent_changed_before_unlink')
                relative = PurePosixPath(row['path'])
                target = Path(row['target'])
                need(relative.parts[0] == 'pytest' and len(relative.parts) == 2
                     and re.fullmatch(r'test_[A-Za-z0-9_]+current', relative.name)
                     and target.is_absolute() and target.parent == pytest_dir
                     and target.name.startswith('test_') and not target.name.endswith('current'),
                     'captured_direct_framework_alias_shape_required')
                current = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
                need(stat.S_ISLNK(current.st_mode) and witness(current) == row['metadata']
                     and os.readlink(relative.name, dir_fd=descriptor) == row['target']
                     and root_identity(target) == row['target_directory'], 'captured_alias_identity_changed')
                # Cooperative dead-owner quiescence, not inode-conditional unlink/CAS.
                # Exact names only; never unlink a target or enumerate a cleanup pattern.
                os.unlink(relative.name, dir_fd=descriptor)
                os.fsync(descriptor)
                removed.append(row)
                save('removed-framework-aliases.json', {'schema':'exact-framework-alias-removals@1',
                    'old_attempt':str(OLD), 'removed':removed,
                    'regular_artifact_preservation':'pending_complete_post_removal_census',
                    'unlink_is_not_inode_CAS':True,
                    'unlink_to_journal_crash_window_exists':True})
        finally:
            os.close(descriptor)
        fsync_directory(OLD)
        stage = 'verify_preserved_regular_inventory'
        expected_artifacts = inventory(capture, aliases_removed=True)
        save('preserved-regular-inventory.json', expected_artifacts)
        stopped()
        selected_retained(capture)
        guard()
        joined()
        stage = 'public_default_capacity_reconciliation'
        sys.path.insert(0, str(DATASETS))
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_resource_recovery import reconcile_retained_reservation
        # Default require_capacity intentionally remains implicit and unchanged.
        recovery = reconcile_retained_reservation(LEDGER, roots=ROOTS,
            reservation_id=RESERVATION, expected_record_sha256=RECORD_SHA,
            attempt_directory=OLD, expected_artifacts=expected_artifacts, artifacts_durable=True)
        save('public-recovery-projection.json', recovery, 2*1024*1024)
        stage = 'verify_actual_selected_postwrite_record'
        row = selected_record()
        need(row.get('status') == 'released' and row.get('artifacts_durable_asserted') is True
             and row.get('explicit_reconciliation') == recovery
             and row.get('final_attempt_bytes') == capture['regular_file_bytes']
             and recovery.get('previous_record_sha256') == RECORD_SHA
             and recovery.get('reservation_id') == RESERVATION
             and recovery.get('artifacts') == expected_artifacts
             and recovery.get('outputs_deleted') is False
             and recovery.get('admitted') is False, 'actual_public_selected_release_required')
        stopped()
        inventory(capture, aliases_removed=True)
        guard()
        result = {'schema':'legal-ir-failed-fixture-recovery-result@1', 'public_reconciliation_returned':True,
            'actual_selected_postwrite_status_observed':'released', 'selected_previous_record_sha256':RECORD_SHA,
            'reservation_id':RESERVATION, 'framework_aliases_removed':len(removed),
            'regular_files_preserved':len(expected_artifacts), 'regular_bytes_preserved':capture['regular_file_bytes'],
            'old_control_evidence':capture['old_control_evidence'],
            'original_native_failure_observation_sha256':hashlib.sha256(failure_raw).hexdigest(),
            'original_native_finalization_succeeded':False, 'original_results_rewritten':False,
            'original_parent_attachment_order_observed':attachment['original_parent_attachment_order_observed'],
            'original_owner_and_child_group_observed_absent':True,
            'capacity_policy':'require_capacity', 'fresh_actual_postwrite_global_census_required':True,
            'returned_final_accounting':'API arithmetic prewrite projection; independent current census still required',
            'new_native_finalized':False, 'runtime_enabled':False, 'canonical_tasks_accepted':0,
            'completion_authority':False, 'github_qualification':'skipped_by_user'}
        save('failed-fixture-recovery.json', result, 256*1024)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        # Retain original regular evidence and exact partial-removal history on failure.
        save('failed-fixture-recovery.json', {'schema':'legal-ir-failed-fixture-recovery-result@1',
            'public_reconciliation_confirmed':False, 'stage':stage, 'error_type':type(error).__name__,
            'removed_framework_aliases':removed, 'original_results_rewritten':False,
            'original_native_finalization_succeeded':False, 'completion_authority':False,
            'canonical_tasks_accepted':0, 'new_native_finalized':False}, 256*1024)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
