"""Explicit private offline ACK worker; no router, model, database or SDK call.

The sealed request and context are metadata. Only the real parent peer's ACK
can satisfy the actual owner client; this module grants no native authority.
The separate admitted M1 entry/pass-fd/claim wiring is not implemented here.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import select
import stat
import sys
import time


SCHEMA = 'private-M1a-sealed-offline-ACK-request/v1'
RECEIPT_SCHEMA = 'private-M1a-offline-ACK-observation/v1'
MAX_REQUEST = 65536
MAX_STDIN = 128000
MAX_RECEIPT = 65536
SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
STAGES = ('request_prepared', 'remaining_deadline', 'worker_v4_response')
WORKER_RELATIVE = Path('ipfs_accelerate_py/cli_runtime/grok_private_m1_offline_worker.py')
CLIENT_RELATIVE = Path('ipfs_accelerate_py/cli_runtime/grok_production_owner.py')
REQUIRED_CLIENT_SHA256 = '99e3409c26d0f31b730c5d535fdfe07a4dbc1bcc248dda16f6b44c496427386d'
FORBIDDEN_IMPORTS = ('ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner',
                     'ipfs_accelerate_py.cli_runtime.grok_named_session')


class OfflineRefusal(RuntimeError):
    pass


def need(value, reason):
    if not value:
        raise OfflineRefusal(reason)


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


def decode(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, 'duplicate_metadata_key')
            result[key] = value
        return result

    def nonfinite(_):
        raise OfflineRefusal('nonfinite_metadata')

    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)
    need(type(value) is dict and canonical(value) == raw, 'canonical_object_required')
    return value


def read_sealed_request(descriptor, expected_sha256):
    """Read one actual fully sealed inherited memfd; grant no capability."""
    need(type(descriptor) is int and descriptor >= 3 and type(expected_sha256) is str and
         re.fullmatch('[0-9a-f]{64}', expected_sha256), 'explicit_sealed_request_required')
    before = os.fstat(descriptor)
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 0 and before.st_uid == os.geteuid() and
         0 < before.st_size <= MAX_REQUEST and
         fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & SEALS == SEALS,
         'actual_fully_sealed_private_memfd_required')
    raw = os.pread(descriptor, before.st_size + 1, 0)
    after = os.fstat(descriptor)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_nlink, row.st_size)
    need(identity(before) == identity(after) and len(raw) == before.st_size and
         hashlib.sha256(raw).hexdigest() == expected_sha256, 'sealed_request_changed')
    return validate_request(decode(raw))


def validate_request(value):
    fields = {'schema', 'mode', 'one_shot', 'attempt_cap', 'provider_dispatch_allowed',
              'private_task_alias', 'database_task_cid', 'allocation_session_id',
              'accelerator_root', 'workspace', 'receipt_directory', 'source_generation',
              'work_deadline_monotonic', 'native_deadline_monotonic',
              'worker_pin', 'owner_client_pin', 'source_profile_pin', 'capsule_pin'}
    need(type(value) is dict and set(value) == fields and value['schema'] == SCHEMA,
         'closed_private_request_schema_required')
    need(value['mode'] == 'offline-ACK-only' and value['one_shot'] is True and
         type(value['attempt_cap']) is int and value['attempt_cap'] == 1 and
         value['provider_dispatch_allowed'] is False, 'offline_one_shot_cap1_only')
    need(type(value['private_task_alias']) is str and
         re.fullmatch('private-M1a-[0-9a-f]{32}', value['private_task_alias']),
         'fresh_private_task_alias_required')
    for field in ('database_task_cid', 'allocation_session_id'):
        item = value[field]
        need(type(item) is str and 0 < len(item) <= 512 and not any(ord(c) < 32 for c in item),
             'bounded_private_identity_required')
    for field in ('accelerator_root', 'workspace', 'receipt_directory'):
        item = value[field]
        need(type(item) is str and Path(item).is_absolute() and '..' not in Path(item).parts,
             'absolute_unaliased_paths_required')
    generation = value['source_generation']
    need(type(generation) is str and generation == 'grok-automatic-named-session/' + value['private_task_alias'],
         'exact_private_source_generation_required')
    for field in ('work_deadline_monotonic', 'native_deadline_monotonic'):
        item = value[field]
        need(type(item) in (int, float) and math.isfinite(item), 'finite_original_clocks_required')
    need(value['work_deadline_monotonic'] <= value['native_deadline_monotonic'],
         'original_native_clock_cannot_be_extended')
    for field in ('worker_pin', 'owner_client_pin', 'source_profile_pin', 'capsule_pin'):
        item = value[field]
        need(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'} and
             type(item['path']) is str and Path(item['path']).is_absolute() and
             type(item['bytes']) is int and 0 < item['bytes'] <= 2 * 1024**2 and
             type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']),
             'exact_bounded_metadata_source_pin_required')
    root = Path(value['accelerator_root'])
    need(value['worker_pin']['path'] == str(root / WORKER_RELATIVE) and
         value['owner_client_pin']['path'] == str(root / CLIENT_RELATIVE) and
         value['owner_client_pin']['sha256'] == REQUIRED_CLIENT_SHA256,
         'normal_worker_and_client_sources_required')
    return value


def read_pin(pin, *, maximum=2 * 1024**2, private=False):
    path = Path(pin['path'])
    need(path.resolve(strict=True) == path, 'source_or_metadata_alias')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
             0 < before.st_size <= maximum and before.st_size == pin['bytes'], 'regular_bounded_pin_required')
        if private:
            need(before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) == 0o600,
                 'private_metadata_custody_required')
        raw = bytearray()
        while len(raw) <= maximum:
            part = os.read(descriptor, min(65536, maximum + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        final = os.fstat(descriptor)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                                row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        need(len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256'] and
             identity(before) == identity(final) == identity(path.lstat()), 'pinned_bytes_or_identity_changed')
        return bytes(raw)
    finally:
        os.close(descriptor)


def read_complete_stdin(descriptor, work_deadline_monotonic):
    """Complete original stdin under its unchanged absolute work clock."""
    raw = bytearray()
    while True:
        remaining = work_deadline_monotonic - 3.0 - time.monotonic()
        need(remaining > 0, 'original_stdin_work_deadline_expired')
        readable, _, _ = select.select([descriptor], [], [], remaining)
        need(bool(readable), 'original_stdin_work_deadline_expired')
        part = os.read(descriptor, min(65536, MAX_STDIN + 1 - len(raw)))
        need(time.monotonic() < work_deadline_monotonic - 3.0, 'original_stdin_work_deadline_expired')
        if not part:
            break
        raw.extend(part)
        need(len(raw) <= MAX_STDIN, 'complete_original_stdin_exceeds_bound')
    need(bool(raw.strip()), 'complete_original_stdin_required')
    raw.decode('utf-8')
    return bytes(raw)


def read_private_context(path):
    """Bound the uncommitted context read before calculating its file digest."""
    path = Path(path)
    need(path.resolve(strict=True) == path, 'owner_context_path_alias')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid == os.geteuid() and
             stat.S_IMODE(before.st_mode) == 0o600 and 0 < before.st_size <= 8192,
             'bounded_actual_private_context_required')
        raw = os.read(descriptor, 8193)
        after = os.fstat(descriptor)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                                row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        need(len(raw) == before.st_size and identity(before) == identity(after) == identity(path.lstat()),
             'owner_context_changed_during_read')
        return decode(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    finally:
        os.close(descriptor)


def durable_observation(directory, value):
    directory = Path(directory)
    need(directory.resolve(strict=True) == directory and directory.is_dir() and
         directory.lstat().st_uid == os.geteuid() and stat.S_IMODE(directory.lstat().st_mode) == 0o700,
         'actual_private_receipt_directory_required')
    raw = canonical(value)
    need(len(raw) <= MAX_RECEIPT, 'offline_receipt_bound')
    path = directory / 'offline-ACK.json'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    result = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    need(read_pin(result, maximum=MAX_RECEIPT, private=True) == raw, 'offline_receipt_readback_required')
    return result


def run(args):
    need(args.private_m1_offline_handshake is True, 'explicit_private_M1_offline_opt_in_required')
    request = read_sealed_request(args.private_m1_request_fd, args.private_m1_request_sha256)
    root = Path(request['accelerator_root'])
    need(root.resolve(strict=True) == root and
         Path(__file__).resolve(strict=True) == root / WORKER_RELATIVE,
         'actual_selected_normal_worker_source_required')
    for name in ('worker_pin', 'owner_client_pin', 'source_profile_pin', 'capsule_pin'):
        read_pin(request[name])
    need(args.transport_capsule == request['capsule_pin']['path'] and
         args.transport_capsule_sha256 == request['capsule_pin']['sha256'], 'actual_handoff_capsule_argv_required')
    need(Path.cwd().resolve(strict=True) == Path(request['workspace']), 'actual_final_workspace_required')
    original_stdin = read_complete_stdin(sys.stdin.fileno(), request['work_deadline_monotonic'])
    # The real parent fsyncs this actual birth-bound context before providing
    # stdin; waiting for complete stdin avoids racing its on_started callback.
    context_path = Path(args.owner_context)
    directory = Path(request['receipt_directory'])
    need(context_path.parent == directory and context_path.resolve(strict=True) == context_path,
         'context_must_be_in_same_actual_charged_private_directory')
    context, context_pin = read_private_context(context_path)
    sys.path.insert(0, str(root))
    core = importlib.import_module('ipfs_accelerate_py.cli_runtime.grok_production_owner')
    need(core.__file__ == core.__spec__.origin == request['owner_client_pin']['path'] and
         sys.modules.get(core.__name__) is core, 'actual_normal_owner_client_origin_required')
    need(not any(name in sys.modules for name in FORBIDDEN_IMPORTS), 'offline_worker_must_not_import_provider_runtime')
    for field, expected in dict(portal_task_id=request['private_task_alias'],
            database_task_cid=request['database_task_cid'], database_attempt_number=1, portal_attempt=1,
            allocation_session_id=request['allocation_session_id'], workspace=request['workspace'],
            source_generation=request['source_generation'], work_deadline_monotonic=request['work_deadline_monotonic'],
            native_deadline_monotonic=request['native_deadline_monotonic'],
            worker_source_sha256=request['worker_pin']['sha256'], owner_source_sha256=request['owner_client_pin']['sha256'],
            source_bindings_sha256=request['source_profile_pin']['sha256'], capsule_sha256=request['capsule_pin']['sha256']).items():
        need(type(context.get(field)) is type(expected) and context[field] == expected,
             'actual_private_owner_context_join_required')
    parent = core.process(os.getppid())
    client = core.ProductionOwnerGuardClient(context, expected_owner_pid=parent['pid'],
        expected_owner_birth=parent['birth'], expected_request_sha256=args.owner_request_sha256,
        expected_source_generation=request['source_generation'], expected_worker_source_sha256=request['worker_pin']['sha256'],
        expected_allocation_session_id=args.owner_allocation_session_id)
    calls = []
    for stage in STAGES:
        started = time.monotonic()
        client.assert_current(stage)
        calls.append(dict(stage=stage, started_monotonic=started, returned_monotonic=time.monotonic()))
    for name in ('worker_pin', 'owner_client_pin', 'source_profile_pin', 'capsule_pin'):
        read_pin(request[name])
    need(decode(read_pin(context_pin, maximum=8192, private=True)) == context, 'owner_context_changed')
    observed = time.monotonic()
    need(observed < request['work_deadline_monotonic'] - 3.0, 'offline_response_original_work_expired')
    receipt = dict(schema=RECEIPT_SCHEMA, private_task_alias=request['private_task_alias'],
        database_task_cid=request['database_task_cid'], sealed_request_sha256=args.private_m1_request_sha256,
        request_sha256=context['request_sha256'], context_sha256=context['context_sha256'],
        allocation_session_id=request['allocation_session_id'], source_generation=request['source_generation'],
        worker_source_sha256=request['worker_pin']['sha256'], owner_source_sha256=request['owner_client_pin']['sha256'],
        source_bindings_sha256=request['source_profile_pin']['sha256'], capsule_sha256=request['capsule_pin']['sha256'],
        original_stdin_bytes=len(original_stdin), original_stdin_sha256=hashlib.sha256(original_stdin).hexdigest(),
        owner_rpc_receipts=client.receipts(), call_timings=calls, response_monotonic=observed,
        offline_only=True, provider_calls=0, allocation_capture_written=False,
        metadata_only=True, completion_authority=False, native_qualified=False, Lean_admitted=False)
    result = durable_observation(directory, receipt)
    need(time.monotonic() < request['work_deadline_monotonic'] - 3.0, 'offline_receipt_late')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-m1-offline-handshake', action='store_true')
    parser.add_argument('--private-m1-request-fd', type=int, default=-1)
    parser.add_argument('--private-m1-request-sha256', default='')
    for option in ('owner-context', 'owner-request-sha256', 'owner-allocation-session-id',
                   'transport-capsule', 'transport-capsule-sha256'):
        parser.add_argument('--' + option, default='')
    args = parser.parse_args(argv)
    try:
        pin = run(args)
    except BaseException as error:
        # No exception message/context/local/environment dump or success receipt.
        print(json.dumps(dict(schema='private-M1a-offline-worker-failure/v1',
            error_type=type(error).__name__, provider_calls=0, native_qualified=False)), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(dict(schema='private-M1a-offline-worker-result/v1', observation=pin,
                         provider_calls=0, native_qualified=False)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
