"""Read-only ordering witness for the actual original generic native parent.

No project import, lease reconstruction/acquisition/renewal, signal or recovery.
Only the original parent retains actual SDK capabilities. Returned projections
are observations and cannot authorize later work or substitute for that parent.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

OWNER = Path('/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/run_image_owner_v05.py')
OWNER_SHA = 'ab244fa456eb1809be12235c9911353adb8b33400e7968218c40a41790a21696'
STATE = Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
POLICY = 'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
LEDGER = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json')


def need(value, reason):
    if not value: raise RuntimeError(reason)


def regular(path, maximum=8 * 1024 * 1024):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), 'gate_source_alias_refused')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and not before.st_mode & 0o002 and 0 < before.st_size <= maximum,
             'gate_bounded_owned_regular_file_required')
        raw = stream.read(maximum + 1); after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid,
                         m.st_gid, m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()),
         'gate_snapshot_replaced_during_read')
    return raw


def decode(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'gate_duplicate_json_key'); result[key] = value
        return result
    def finite(text):
        result = float(text); need(math.isfinite(result), 'gate_nonfinite_json'); return result
    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('gate_nonfinite_json')))


def process(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    need(fields[0] != 'Z' and proc.stat().st_uid == os.getuid(), 'gate_live_same_uid_process_required')
    return {'pid': pid, 'birth': int(fields[19]), 'ppid': int(fields[1]), 'pgid': int(fields[2])}


def ancestry(state, receipt, parent, name):
    policy = state.get('config')
    need(type(policy) is dict and hashlib.sha256(json.dumps(policy, sort_keys=True,
         separators=(',', ':'), allow_nan=False).encode()).hexdigest() == POLICY
         and policy.get('total_child_process_slots') == 16, 'gate_original_native_policy_required')
    leases = state.get('leases')
    need(type(leases) is dict, 'gate_native_lease_rows_required')
    root, phase = leases.get(receipt['root_lease_id'], {}), leases.get(receipt['phase_lease_id'], {})
    leaves = [row for row in leases.values() if row.get('request_id') == 'daemon:' + receipt['disk_reservation_id']]
    need(len(leaves) == 1, 'gate_exact_original_disk_native_leaf_required')
    leaf = leaves[0]
    need(root.get('request_id') == name and root.get('parent_lease_id') is None
         and phase.get('request_id') == name + ':phase'
         and phase.get('parent_lease_id') == root.get('lease_id')
         and leaf.get('parent_lease_id') == phase.get('lease_id'), 'gate_original_native_ancestry_required')
    for row in (root, phase, leaf):
        need(row.get('owner_pid') == parent['pid'] and row.get('owner_birth_marker') == str(parent['birth'])
             and row.get('cpu_slots') == 1 and row.get('memory_mb') == 4096
             and row.get('child_process_slots') == 8 and row.get('lane') == 'orchestration'
             and row.get('cancelled') is False and row.get('expires_at', 0) > time.time(),
             'gate_live_original_eight_slot_native_rows_required')
    # Never return raw state or private authority keys.
    return {'root_lease_id': root['lease_id'], 'phase_lease_id': phase['lease_id'],
            'native_leaf_id': leaf['lease_id'], 'parent_pid': parent['pid'], 'parent_birth': parent['birth']}


def wait_for_original_generic_parent(request_path, *, attempt_directory,
        expected_request_name, storage_bytes, worker_command, timeout_seconds=60):
    """Wait for original attachment, then join read-only actual parent evidence."""
    need(type(timeout_seconds) in {int, float} and 0 < timeout_seconds <= 120, 'gate_finite_wait_required')
    request_path, attempt = Path(request_path), Path(attempt_directory)
    request_raw = regular(request_path); request = decode(request_raw)
    need(request.get('schema') == 'legal-ir-native-orchestration-request@1'
         and request.get('name') == expected_request_name and request.get('purpose') == 'controller_preparation'
         and request.get('attempt_directory') == str(attempt) and request.get('storage_bytes') == storage_bytes
         and request.get('external_charges') == [] and request.get('command') == worker_command,
         'gate_exact_original_request_required')
    need(hashlib.sha256(regular(OWNER)).hexdigest() == OWNER_SHA
         and request.get('source_sha256', {}).get(str(OWNER)) == OWNER_SHA,
         'gate_original_parent_source_required')
    self_row, parent = process(os.getpid()), process(os.getppid())
    need(self_row['ppid'] == parent['pid'] and self_row['pgid'] == self_row['pid'] == os.getpgrp(),
         'gate_actual_owned_worker_group_leader_required')
    cmdline = (Path('/proc') / str(parent['pid']) / 'cmdline').read_bytes().split(b'\0')
    need(cmdline[:-1] == [b'/usr/bin/python3.12', b'-B', os.fsencode(OWNER), os.fsencode(request_path)],
         'gate_actual_original_parent_command_required')
    deadline = time.monotonic() + timeout_seconds
    while True:
        need(process(os.getppid()) == parent and process(os.getpid()) == self_row,
             'gate_actual_original_process_identity_changed')
        try:
            receipt = decode(regular(attempt / 'receipt.json'))
            if receipt.get('status') == 'running': break
        except (OSError, RuntimeError, ValueError): pass
        need(time.monotonic() < deadline, 'gate_original_parent_attachment_not_ready')
        time.sleep(.05)
    need(receipt.get('schema') == 'legal-ir-native-orchestration-receipt@1'
         and receipt.get('owner_pid') == parent['pid'] and receipt.get('child_pid') == self_row['pid']
         and receipt.get('child_pgid') == self_row['pgid']
         and receipt.get('request_sha256') == hashlib.sha256(request_raw).hexdigest(),
         'gate_actual_original_parent_receipt_join_required')
    # Atomic native writes can replace advisory snapshots. Retry only reads;
    # never call SDK snapshot/active_leases (those can recover stale rows).
    for retry in range(8):
        try:
            first = ancestry(decode(regular(STATE)), receipt, parent, expected_request_name)
            ledger = decode(regular(LEDGER))
            row = ledger.get('reservations', {}).get(receipt['disk_reservation_id'], {})
            need(ledger.get('limit_bytes') == 160000000000 and row.get('status') == 'active'
                 and row.get('owner_pid') == parent['pid'] and row.get('parent_lease_id') == first['phase_lease_id']
                 and row.get('native_lease_id') == first['native_leaf_id']
                 and row.get('storage_bytes') == storage_bytes
                 and row.get('cpu_slots') == 1 and row.get('memory_mb') == 4096
                 and row.get('child_process_slots') == 8
                 and row.get('child') == {'pid': self_row['pid'], 'birth': str(self_row['birth'])},
                 'gate_original_disk_group_attachment_required')
            metadata = attempt.lstat(); directory = row.get('attempt_directory', {})
            need(stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.getuid()
                 and stat.S_IMODE(metadata.st_mode) == 0o700
                 and directory == {'path': str(attempt), 'device': metadata.st_dev, 'inode': metadata.st_ino},
                 'gate_actual_original_attempt_identity_required')
            need(ancestry(decode(regular(STATE)), receipt, parent, expected_request_name) == first,
                 'gate_original_native_ancestry_changed_during_join')
            need(regular(request_path) == request_raw, 'gate_original_request_changed')
            return dict(first, child_pid=self_row['pid'], child_pgid=self_row['pgid'],
                child_birth=self_row['birth'], original_parent_attachment_order_observed=True,
                portable_projection_grants_authority=False)
        except (OSError, RuntimeError, ValueError):
            if retry == 7: raise
            time.sleep(.05)
