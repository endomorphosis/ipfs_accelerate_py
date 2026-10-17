#!/usr/bin/env python3
"""Bridge ordinary native validation to independent factual evidence.

The ordinary protected validation command waits for ONE root-owned gate file.
The independent native owner first authenticates the actual database claim and
Portal projection, runs factual_acceptance_v02 in its separate admitted job,
records the real special-kind evidence using DatabaseTaskSource.record_evidence,
then publishes the gate. This wrapper cannot record evidence or change status.

``publish_gate_after_checked_result`` is the callable root-watcher writer. It
receives an already-open real public source, actual source-pinned native Portal
readback and a separately executed validation descriptor. The native caller owns
admission, the independence of that job and provenance of projection observations.
No new signer, orchestrator, validation authority or database is created here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time

VALIDATOR_SHA256 = '18cdee9eb90ec390cd3b26e4371673dbf40ffb3a0f5a65b5b30e67b62ff53984'
GATE_SCHEMA = 'legal-ir-native-factual-gate/v2'
OBSERVATION_FIELDS = {
    'database_task_cid', 'portal_task_cid', 'native_attempt_id',
    'database_attempt_number', 'portal_attempt_number', 'claim_id', 'lease_id',
    'fencing_token', 'fence_epoch', 'execution_packet_revision',
    'projection_pin', 'projection_binding_pin', 'portal_state_pin',
}


def load_validator(path, expected_sha256=VALIDATOR_SHA256):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('exact protected validator path required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_mode & 0o022 or before.st_size > 1048576):
            raise ValueError('protected validator custody differs')
        raw = bytearray()
        while len(raw) <= 1048576:
            block = os.read(fd, 65536)
            if not block:
                break
            raw.extend(block)
        after = os.fstat(fd)
        if (len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha256
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or path.lstat() != after):
            raise ValueError('protected validator bytes changed')
    finally:
        os.close(fd)
    spec = importlib.util.spec_from_file_location('protected_legal_ir_factual_validator_v02', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(bytes(raw), str(path), 'exec'), module.__dict__)
    return module


def plain(value):
    if hasattr(value, 'items'):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def _write_gate_new(path, raw):
    path = Path(path)
    parent = path.parent
    info = parent.lstat()
    if (not path.is_absolute() or parent.resolve(strict=True) != parent
            or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_mode & 0o022 or path.exists()):
        raise ValueError('exclusive native-owned gate path required')
    temp = parent / (path.name + '.prepared-' + str(os.getpid()))
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError('short gate write')
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    # Atomic no-replace publication: both links refer to the same inode only
    # during this owner-controlled publication; remove the prepared name before
    # any consumer can accept the final singly linked gate.
    os.link(temp, path, follow_symlinks=False)
    temp.unlink()
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _checked_evidence_receipt(v, receipt, *, task_cid, digest, descriptor):
    v.need(type(receipt) is dict and receipt.get('event_type') == 'intent.evidence_recorded'
           and type(receipt.get('event_id')) is str and receipt['event_id']
           and type(receipt.get('global_sequence')) is int and receipt['global_sequence'] > 0,
           'actual_native_evidence_receipt_required')
    details = receipt.get('details')
    v.need(type(details) is dict and details.get('task_cid') == task_cid
           and details.get('evidence_kind') == v.EVIDENCE_KIND
           and details.get('digest') == digest and details.get('body') == descriptor,
           'native_evidence_receipt_binding_differs')


def publish_gate_after_checked_result(source, *, validator_path, controller_prefix,
                                     checked_result_pin, report_path, receipt_root,
                                     actual_claim_and_portal_observation, gate_path,
                                     source_guard, record_operation):
    """Root-watcher publication AFTER a genuine independent native job.

    ``actual_claim_and_portal_observation`` comes from original native bridge
    binding/projection owners under the caller's source custody, never from the
    candidate's asserted task CID. The exact observed projection and binding
    files remain hash-pinned here. Caller must retain the independent native job
    request/result/closeout and its original executed verifier source pins.
    """
    if not callable(source_guard) or not callable(record_operation):
        raise ValueError('native custody and durable recorder required')
    source_guard()
    v = load_validator(validator_path)
    capsule = v.load_capsule(controller_prefix)
    observation = plain(actual_claim_and_portal_observation)
    v.need(type(observation) is dict and set(observation) == OBSERVATION_FIELDS,
           'complete actual native claim/projection observation required')
    for key in ('projection_pin', 'projection_binding_pin', 'portal_state_pin'):
        pin = observation[key]
        v.need(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'},
               'native projection pin shape differs')
        v.read_checked(Path(pin['path']), pin=pin)
    raw, descriptor_pin = v.read_checked(Path(checked_result_pin['path']), pin=checked_result_pin)
    descriptor = v.strict_json(raw)
    v.need(descriptor.get('schema') == 'legal-ir-source-factual-check-result/v2'
           and descriptor.get('valid') is True and descriptor.get('native_evidence_written') is False,
           'actual independent checked descriptor required')
    row = source.get('LIR-9001')
    v.need(row is not None and row.status == 'in_progress'
           and row.task_cid == observation['database_task_cid']
           and row.task_cid == capsule['contract']['selected_task_cid'],
           'actual live database bootstrap claim differs')
    claim = row.body.get('completion_receipt')
    v.need(isinstance(claim, dict) or hasattr(claim, 'items'), 'actual native claim receipt missing')
    claim = plain(claim)
    v.need(claim.get('operation') in {'database_claim', 'database_attempt_admitted'},
           'actual native claim phase differs')
    for field, observed in (
        ('attempt_id', 'native_attempt_id'), ('attempt_number', 'database_attempt_number'),
        ('claim_id', 'claim_id'), ('lease_id', 'lease_id'),
        ('fencing_token', 'fencing_token'), ('fence_epoch', 'fence_epoch')):
        v.need(claim.get(field) == observation[observed], 'actual claim/projection crosswalk differs')
    v.need(observation['execution_packet_revision'] == capsule['contract']['execution_packet_revision']
           and claim.get('claimed_from_revision') == observation['execution_packet_revision'],
           'actual claim specification revision differs')
    native = {
        'task_cid': observation['database_task_cid'], 'portal_task_cid': observation['portal_task_cid'],
        'task_revision': observation['execution_packet_revision'],
        'native_attempt_id': observation['native_attempt_id'],
        'database_attempt_number': observation['database_attempt_number'],
        'attempt': observation['portal_attempt_number'],
        'report_sha256': descriptor['report']['sha256'],
        'invocation_sha256': descriptor['invocation_receipt']['sha256'],
        'native_prompt_sha256': descriptor['native_prompt_sha256'],
    }
    # Independently rederive the exact checked descriptor again before acting.
    rechecked = v.check_report(Path(report_path), capsule=capsule, native=native,
                               receipt_root=Path(receipt_root))
    v.need(rechecked == descriptor, 'separate independent result differs from current immutable inputs')
    digest = 'sha256:' + descriptor_pin['sha256']
    source_guard()
    record_operation('independent_factual_evidence_prepared', {
        'task_cid': native['task_cid'], 'native_attempt_id': native['native_attempt_id'],
        'descriptor': {'path': str(checked_result_pin['path']), **descriptor_pin},
        'kind': v.EVIDENCE_KIND, 'digest': digest})
    # This is the genuine public native mutation. No manually constructed
    # native receipt or completion transition is substituted for its result.
    receipt = source.record_evidence(task_cid=native['task_cid'], evidence_kind=v.EVIDENCE_KIND,
                                     digest=digest, body=descriptor)
    receipt = plain(receipt.to_dict())
    _checked_evidence_receipt(v, receipt, task_cid=native['task_cid'], digest=digest,
                              descriptor=descriptor)
    record_operation('independent_factual_evidence_recorded', receipt)
    evidence = source.current_evidence_for_task(native['task_cid'])
    v.need(any(item.get('evidence_kind') == v.EVIDENCE_KIND and item.get('digest') == digest
               and plain(item.get('body')) == descriptor for item in evidence),
           'actual native factual evidence readback differs')
    gate = {'schema': GATE_SCHEMA, 'task_id': v.TASK, 'status': 'factual_evidence_recorded',
            'native_bindings': native, 'checked_descriptor': {
                'path': str(checked_result_pin['path']), **descriptor_pin},
            'native_evidence_receipt': receipt, 'actual_projection_observation': observation,
            'completion_authority': False, 'task_status_changed_by_gate_writer': False}
    source_guard()
    _write_gate_new(Path(gate_path), v.canonical(gate))
    readback_raw, gate_pin = v.read_checked(Path(gate_path))
    v.need(v.strict_json(readback_raw) == gate, 'durable owner gate readback differs')
    record_operation('independent_factual_gate_published', {'path': str(gate_path), **gate_pin})
    return {'gate': {'path': str(gate_path), **gate_pin}, 'native_evidence_receipt': receipt,
            'task_completed_by_gate_writer': False}


def check_gate(v, *, gate_path, report_path, controller_prefix, receipt_root):
    # Ordinary native validation supplies the declared logical output relative
    # to its actual allocated cwd. Preserve lexical path components here; the
    # original stable read refuses aliases rather than resolving them away.
    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    raw, gate_pin = v.read_checked(Path(gate_path))
    gate = v.strict_json(raw)
    v.need(type(gate) is dict and set(gate) == {'schema', 'task_id', 'status',
        'native_bindings', 'checked_descriptor', 'native_evidence_receipt',
        'actual_projection_observation', 'completion_authority', 'task_status_changed_by_gate_writer'},
        'owner_gate_shape_differs')
    v.need(gate['schema'] == GATE_SCHEMA and gate['task_id'] == v.TASK
           and gate['status'] == 'factual_evidence_recorded'
           and gate['completion_authority'] is False
           and gate['task_status_changed_by_gate_writer'] is False, 'owner_gate_scope_differs')
    capsule = v.load_capsule(controller_prefix)
    native = gate['native_bindings']
    v.need(type(native) is dict and set(native) == set(v.NATIVE_FIELDS), 'owner_native_binding_shape_differs')
    environment = {name: str(native[key]) for key, name in v.NATIVE_FIELDS.items()}
    native = v.native_bindings(environment)
    pin = gate['checked_descriptor']
    v.need(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}, 'checked_descriptor_pin_differs')
    descriptor_raw, descriptor_pin = v.read_checked(Path(pin['path']), pin=pin)
    descriptor = v.strict_json(descriptor_raw)
    expected = v.check_report(Path(report_path), capsule=capsule, native=native,
                              receipt_root=Path(receipt_root))
    v.need(expected == descriptor, 'independent_descriptor_recheck_differs')
    _checked_evidence_receipt(v, gate['native_evidence_receipt'], task_cid=native['task_cid'],
                              digest='sha256:' + descriptor_pin['sha256'], descriptor=descriptor)
    observation = gate['actual_projection_observation']
    v.need(type(observation) is dict and set(observation) == OBSERVATION_FIELDS,
           'owner_native_projection_observation_differs')
    for key in ('projection_pin', 'projection_binding_pin', 'portal_state_pin'):
        v.read_checked(Path(observation[key]['path']), pin=observation[key])
    v.read_checked(Path(gate_path), pin=gate_pin)
    return {'schema': 'legal-ir-native-factual-gate-check/v2', 'valid': True,
            'task_id': v.TASK, 'task_cid': native['task_cid'],
            'portal_task_cid': native['portal_task_cid'], 'native_attempt_id': native['native_attempt_id'],
            'checked_descriptor': descriptor_pin, 'owner_gate': gate_pin,
            'native_evidence_already_recorded_by_independent_owner': True,
            'native_evidence_written_by_this_wrapper': False,
            'database_opened': False, 'product_imported': False, 'completion_authority': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-all', action='store_true')
    mode.add_argument('--check-report', type=Path)
    parser.add_argument('--controller-prefix', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--validator', type=Path)
    parser.add_argument('--receipt-root', type=Path)
    parser.add_argument('--owner-gate', type=Path)
    parser.add_argument('--wait-seconds', type=int, default=120)
    args = parser.parse_args(argv)
    try:
        v = load_validator(args.validator or args.controller_prefix / 'factual_acceptance_v02.py')
        if args.check_all:
            return v.main(['--check-all', '--controller-prefix', str(args.controller_prefix)])
        v.need(args.receipt_root is not None and args.owner_gate is not None,
               'native_receipt_root_and_owner_gate_required')
        v.need(1 <= args.wait_seconds <= 600, 'bounded_owner_gate_wait_required')
        deadline = time.monotonic() + args.wait_seconds
        while not args.owner_gate.exists():
            v.need(time.monotonic() < deadline, 'independent_native_factual_gate_pending')
            time.sleep(0.25)
        result = check_gate(v, gate_path=args.owner_gate, report_path=args.check_report,
                            controller_prefix=args.controller_prefix, receipt_root=args.receipt_root)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        reason = str(error) if type(error).__name__ == 'FactualRefusal' else type(error).__name__
        print(json.dumps({'schema': 'legal-ir-native-factual-gate-refusal/v2', 'valid': False,
                          'reason': reason, 'native_evidence_written': False,
                          'completion_authority': False}, sort_keys=True))
        return 1


if __name__ == '__main__':
    sys.exit(main())
