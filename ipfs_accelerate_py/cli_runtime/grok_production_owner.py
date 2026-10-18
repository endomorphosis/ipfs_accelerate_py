"""Source-only production transport core; observations never grant authority.

No project imports, resource acquisition, native calls or actions on import.
The production owner must separately execute genuine claim/native/source checks
before ACKing. This client cannot adopt the historical separate-owner launch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import struct
import time

CONTEXT_SCHEMA = 'grok-production-owner-context/v1'
RPC_SCHEMA = 'grok-production-owner-currentness/v1'
MAX_FRAME = 8192
MAX_REQUESTS = 512
PRELAUNCH_FIELDS = (
    'database_task_cid', 'database_attempt_id', 'database_claim_id', 'database_attempt_number',
    'coordination_lease_id', 'owner_session_id', 'fencing_token', 'fence_epoch',
    'portal_task_id', 'portal_task_revision_cid', 'portal_attempt', 'native_attempt_id',
    'allocation_session_id', 'credential_slot', 'credential_fingerprint', 'allocation_store_path',
    'workspace', 'source_generation', 'source_bindings_sha256', 'capsule_sha256',
    'worker_source_sha256', 'owner_source_sha256', 'work_deadline_monotonic',
    'native_deadline_monotonic', 'root_lease_id', 'phase_lease_id', 'disk_native_lease_id',
    'disk_reservation_id',
)
POSTBIRTH_FIELDS = ('request_sha256', 'owner_pid', 'owner_birth', 'worker_pid', 'worker_birth',
    'worker_process_group', 'endpoint', 'socket', 'nonce')
STAGES = frozenset({'request_prepared', 'allocation_binding', 'remaining_deadline',
    'bootstrap_start', 'bootstrap_protocol', 'headless_start', 'headless_close',
    'response_acceptance', 'worker_v4_dispatch', 'worker_v4_response'})


class ProductionOwnerRefusal(RuntimeError):
    """Closed reason codes; no raw bodies, secrets or authority fallback."""


def need(value, reason):
    if not value:
        raise ProductionOwnerRefusal(reason)


def canonical(value):
    try:
        return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                           separators=(',', ':'))+'\n').encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ProductionOwnerRefusal('owner_noncanonical_value') from None


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def current_source_sha256():
    path = Path(__file__).absolute()
    need(path.resolve(strict=True) == path, 'owner_client_callback_source_alias')
    descriptor = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= 2*1024**2,
             'owner_client_callback_source_bound')
        identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink, row.st_size,
                                row.st_mtime_ns, row.st_ctime_ns)
        result, count = hashlib.sha256(), 0
        while True:
            block = os.read(descriptor, 65536)
            if not block:
                break
            count += len(block)
            need(count <= 2*1024**2, 'owner_client_callback_source_bound')
            result.update(block)
        need(count == before.st_size and identity(before) == identity(os.fstat(descriptor)) == identity(path.lstat()),
             'owner_client_callback_source_changed_during_read')
        return result.hexdigest()
    finally:
        os.close(descriptor)


def decode(raw):
    need(type(raw) is bytes and len(raw) <= MAX_FRAME, 'owner_frame_bound')
    def unique(rows):
        value = {}
        for key, item in rows:
            need(key not in value, 'owner_duplicate_JSON_key')
            value[key] = item
        return value
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ProductionOwnerRefusal('owner_nonfinite_JSON')))
    except (ValueError, UnicodeError, RecursionError):
        raise ProductionOwnerRefusal('owner_invalid_JSON') from None
    need(type(value) is dict, 'owner_JSON_object_required')
    return value


def process(pid):
    need(type(pid) is int and pid > 1, 'owner_process_id_invalid')
    raw = Path(f'/proc/{pid}/stat').read_bytes()
    need(len(raw) <= 65536, 'owner_process_record_bound')
    end = raw.rfind(b')')
    fields = raw[end+2:].split()
    need(end > 0 and len(fields) >= 20, 'owner_process_record_shape')
    return {'pid': pid, 'parent_pid': int(fields[1]), 'group_pid': int(fields[2]), 'birth': int(fields[19])}


def socket_identity(path):
    need(type(path) is type(Path()) and path.is_absolute() and path.parent.resolve(strict=True) == path.parent,
         'owner_socket_path_invalid')
    for parent in (path.parent, *path.parent.parents):
        need(stat.S_ISDIR(parent.lstat().st_mode), 'owner_socket_parent_alias')
    metadata = path.lstat()
    parent = path.parent.lstat()
    need(parent.st_uid == os.geteuid() and stat.S_IMODE(parent.st_mode) == 0o700,
         'owner_socket_parent_not_private')
    need(stat.S_ISSOCK(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o600 and
         metadata.st_nlink == 1 and metadata.st_uid == os.geteuid(), 'owner_socket_identity_invalid')
    return {'device': metadata.st_dev, 'inode': metadata.st_ino, 'mode': metadata.st_mode,
            'uid': metadata.st_uid, 'gid': metadata.st_gid, 'nlink': metadata.st_nlink}


def peer(channel):
    pid, uid, gid = struct.unpack('3i', channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    need(pid > 1 and uid == os.geteuid() and gid == os.getegid(), 'owner_kernel_peer_credentials')
    return process(pid)


def _remaining_rpc_timeout(channel, deadline_monotonic):
    """One caller-owned absolute clock; byte arrivals cannot renew it."""
    need(type(deadline_monotonic) in (int, float) and math.isfinite(deadline_monotonic),
         'owner_rpc_absolute_deadline_invalid')
    remaining = deadline_monotonic-time.monotonic()
    need(remaining > 0, 'owner_rpc_absolute_deadline_expired')
    channel.settimeout(min(1.0, remaining))


def _connect_with_deadline(channel, endpoint, deadline_monotonic):
    _remaining_rpc_timeout(channel, deadline_monotonic)
    channel.connect(endpoint)
    _remaining_rpc_timeout(channel, deadline_monotonic)


def send(channel, value, *, deadline_monotonic=None):
    raw = canonical(value)
    need(len(raw) <= MAX_FRAME, 'owner_frame_bound')
    if deadline_monotonic is not None:
        _remaining_rpc_timeout(channel, deadline_monotonic)
    channel.sendall(raw)
    if deadline_monotonic is not None:
        _remaining_rpc_timeout(channel, deadline_monotonic)


def receive(channel, *, deadline_monotonic=None):
    raw = bytearray()
    while b'\n' not in raw:
        if deadline_monotonic is not None:
            _remaining_rpc_timeout(channel, deadline_monotonic)
        part = channel.recv(min(1024, MAX_FRAME+1-len(raw)))
        if deadline_monotonic is not None:
            _remaining_rpc_timeout(channel, deadline_monotonic)
        need(bool(part), 'owner_unexpected_EOF')
        raw.extend(part)
        need(len(raw) <= MAX_FRAME, 'owner_frame_bound')
    body, _, tail = raw.partition(b'\n')
    need(not tail, 'owner_multiple_frames_forbidden')
    value = decode(bytes(body))
    if deadline_monotonic is not None:
        _remaining_rpc_timeout(channel, deadline_monotonic)
    return value


def validate_context(context):
    need(type(context) is dict and set(context) == {'schema', 'context_sha256', *PRELAUNCH_FIELDS, *POSTBIRTH_FIELDS}
         and context.get('schema') == CONTEXT_SCHEMA, 'owner_context_fields')
    raw = canonical(context)
    need(len(raw) <= MAX_FRAME, 'owner_context_bound')
    for key in ('source_bindings_sha256', 'capsule_sha256', 'worker_source_sha256',
                'owner_source_sha256', 'request_sha256', 'context_sha256'):
        need(type(context[key]) is str and re.fullmatch(r'[0-9a-f]{64}', context[key]), 'owner_context_digest')
    for key in ('database_attempt_number', 'portal_attempt', 'owner_pid', 'owner_birth',
                'worker_pid', 'worker_birth', 'worker_process_group'):
        need(type(context[key]) is int and context[key] > 0, 'owner_context_positive_integer')
    for key in ('fencing_token', 'fence_epoch'):
        need(type(context[key]) is int and context[key] >= 0, 'owner_context_fence_integer')
    for key in ('work_deadline_monotonic', 'native_deadline_monotonic'):
        need(type(context[key]) in (int, float) and math.isfinite(context[key]), 'owner_context_deadline')
    need(context['work_deadline_monotonic'] <= context['native_deadline_monotonic'], 'owner_context_deadline_extension')
    need(context['database_attempt_id'] == context['native_attempt_id'], 'owner_context_claim_attempt_alias_mismatch')
    need(type(context['allocation_session_id']) is str and
         re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', context['allocation_session_id']), 'owner_context_allocation_id')
    need(type(context['credential_slot']) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', context['credential_slot']) and
         type(context['credential_fingerprint']) is str and re.fullmatch(r'[0-9a-f]{16}', context['credential_fingerprint']),
         'owner_context_credential_metadata')
    need(type(context['source_generation']) is str and context['source_generation'].startswith('grok-automatic-named-session/')
         and len(context['source_generation']) <= 128 and not any(ord(c) < 32 for c in context['source_generation']),
         'owner_context_source_generation')
    for key in ('database_task_cid', 'database_attempt_id', 'database_claim_id', 'coordination_lease_id',
                'owner_session_id', 'portal_task_id', 'portal_task_revision_cid', 'native_attempt_id',
                'root_lease_id', 'phase_lease_id', 'disk_native_lease_id', 'disk_reservation_id'):
        need(type(context[key]) is str and 1 <= len(context[key]) <= 256 and
             not any(ord(character) < 32 for character in context[key]), 'owner_context_identity_string')
    for key in ('allocation_store_path', 'workspace', 'endpoint'):
        value = context[key]
        need(type(value) is str and 1 <= len(value) <= 4096 and not any(ord(c) < 32 for c in value) and
             Path(value).is_absolute() and '..' not in Path(value).parts, 'owner_context_absolute_path')
    need(len(os.fsencode(context['endpoint'])) < 108, 'owner_context_socket_length')
    need(type(context['socket']) is dict and set(context['socket']) == {'device', 'inode', 'mode', 'uid', 'gid', 'nlink'} and
         all(type(value) is int for value in context['socket'].values()) and
         stat.S_ISSOCK(context['socket']['mode']) and stat.S_IMODE(context['socket']['mode']) == 0o600 and
         context['socket']['nlink'] == 1, 'owner_context_socket_witness')
    need(type(context['nonce']) is str and re.fullmatch(r'[0-9a-f]{48}', context['nonce']), 'owner_context_nonce')
    need(context['owner_pid'] != context['worker_pid'] and context['worker_process_group'] == context['worker_pid'],
         'owner_context_distinct_owned_worker')
    prelaunch = {'schema': CONTEXT_SCHEMA, **{key: context[key] for key in PRELAUNCH_FIELDS}}
    need(digest(prelaunch) == context['request_sha256'] and
         digest({key: value for key, value in context.items() if key != 'context_sha256'}) == context['context_sha256'],
         'owner_context_commitment_mismatch')
    return decode(raw)


class ProductionOwnerGuardClient:
    """No lease acquisition; every boundary requires a fresh real peer ACK."""
    def __init__(self, context, *, expected_owner_pid, expected_owner_birth, expected_request_sha256,
                 expected_source_generation, expected_worker_source_sha256, expected_allocation_session_id):
        context = validate_context(context)
        expected = dict(owner_pid=expected_owner_pid, owner_birth=expected_owner_birth,
            request_sha256=expected_request_sha256, source_generation=expected_source_generation,
            worker_source_sha256=expected_worker_source_sha256, allocation_session_id=expected_allocation_session_id)
        need(all(type(value) is type(context[key]) and value == context[key] for key, value in expected.items()),
             'owner_context_expected_binding_mismatch')
        need(current_source_sha256() == context['owner_source_sha256'],
             'owner_client_callback_source_changed')
        own = process(os.getpid())
        need(own['pid'] == context['worker_pid'] and own['birth'] == context['worker_birth'] and
             own['group_pid'] == context['worker_process_group'] and own['parent_pid'] == context['owner_pid'],
             'owner_client_actual_worker_identity')
        self._context_bytes = canonical(context)
        self._own = own
        self._sequence = 0
        self._receipts = []
        self._owner_peer = None
        self._failed = False

    def context(self):
        return decode(self._context_bytes)

    def receipts(self):
        return json.loads(json.dumps(self._receipts, allow_nan=False))

    def assert_current(self, stage):
        need(not self._failed, 'owner_client_failure_latched')
        try:
            context = self.context()
            need(type(stage) is str and stage in STAGES and self._sequence < MAX_REQUESTS,
                 'owner_client_stage_or_sequence')
            need(process(os.getpid()) == self._own, 'owner_client_process_changed')
            need(current_source_sha256() == context['owner_source_sha256'],
                 'owner_client_callback_source_changed')
            endpoint = Path(context['endpoint'])
            actual = socket_identity(endpoint)
            need(actual == context['socket'], 'owner_client_socket_changed')
            started = time.monotonic()
            remaining = context['work_deadline_monotonic']-started
            need(math.isfinite(remaining) and remaining > 3, 'owner_client_original_deadline')
            rpc_deadline = min(started+1.0, context['work_deadline_monotonic']-3.0)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                _connect_with_deadline(channel, str(endpoint), rpc_deadline)
                observed = peer(channel)
                need(observed['pid'] == context['owner_pid'] and observed['birth'] == context['owner_birth'] and
                     self._owner_peer in (None, observed), 'owner_client_actual_owner_peer')
                sequence = self._sequence+1
                request = {'schema': RPC_SCHEMA, 'sequence': sequence, 'stage': stage,
                    'request_sha256': context['request_sha256'], 'context_sha256': context['context_sha256'],
                    'nonce': context['nonce'], 'pid': self._own['pid'], 'birth': self._own['birth'],
                    'process_group': self._own['group_pid']}
                send(channel, request, deadline_monotonic=rpc_deadline)
                response = receive(channel, deadline_monotonic=rpc_deadline)
            finished = time.monotonic()
            need(finished < rpc_deadline, 'owner_rpc_absolute_deadline_expired')
            expected = {'schema': RPC_SCHEMA, 'status': 'current', 'sequence': sequence, 'stage': stage,
                **{key: context[key] for key in ('request_sha256', 'context_sha256', 'nonce', 'owner_pid',
                    'owner_birth', 'worker_pid', 'worker_birth', 'worker_process_group',
                    'work_deadline_monotonic', 'native_deadline_monotonic')}}
            need(set(response) == set(expected)|{'observed_monotonic'} and
                 all(type(response[key]) is type(value) and response[key] == value for key, value in expected.items()),
                 'owner_client_ACK_binding_mismatch')
            moment = response['observed_monotonic']
            need(type(moment) in (int, float) and math.isfinite(moment) and started <= moment <= finished and
                 finished < context['work_deadline_monotonic']-3, 'owner_client_ACK_not_fresh_or_deadline')
            need(socket_identity(endpoint) == actual and process(os.getpid()) == self._own,
                 'owner_client_identity_changed_during_RPC')
            receipt = {'sequence': sequence, 'stage': stage, 'request_sha256': context['request_sha256'],
                'context_sha256': context['context_sha256'], 'owner_pid': observed['pid'], 'owner_birth': observed['birth'],
                'owner_process_group': observed['group_pid'], 'client_pid': self._own['pid'], 'client_birth': self._own['birth'],
                'client_process_group': self._own['group_pid'], 'observed_monotonic': moment,
                'received_monotonic': finished, 'response_sha256': digest(response),
                'actual_kernel_peer_owner_verified': True, 'observation_is_not_native_authority': True}
            # Final identity reads and response hashing can consume the original
            # clock. Commit no client state based on the earlier receive time.
            accepted = time.monotonic()
            need(accepted < rpc_deadline and accepted < context['work_deadline_monotonic']-3,
                 'owner_rpc_absolute_deadline_expired')
            self._owner_peer = observed
            self._sequence = sequence
            self._receipts.append(receipt)
        except BaseException:
            self._failed = True
            raise


def owner_guard_for(client):
    need(type(client) is ProductionOwnerGuardClient, 'owner_guard_actual_client_required')
    def owner_guard(stage):
        client.assert_current(stage)
        return None
    return owner_guard
