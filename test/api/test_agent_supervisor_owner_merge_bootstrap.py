"""Native paired credentials over real local owner sockets and disposable stores."""
from dataclasses import replace
import copy
import os
import socket
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_merge_bootstrap_broker as broker_module
from ipfs_accelerate_py.agent_supervisor.task_sources import owner_merge_bootstrap as bundle_module
from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
    STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA, StateOwnerBootstrapError, _send_frame,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerConnection, TypedStateOwnerError,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import process_birth_id
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth
from ipfs_accelerate_py.agent_supervisor.merge.owner_recovery_runtime import OwnerRecoveryRuntimeClient
from test.api.semantic_refactoring.test_spar_merge_owner_bootstrap import (
    preserved as preserved, start, native,
)
from test.api.test_agent_supervisor_lgcvf_quack_successor import _lgcvf_test_execution_route_policy


@pytest.fixture
def paired(preserved, tmp_path):
    source, manifest, *_ = preserved
    policy = _lgcvf_test_execution_route_policy(SimpleNamespace(LGCVF_TASK_ALIASES=('DOEP-001',)))
    manifest['scope_bindings'][0]['plan_cid'] = policy.plan_root_cid
    prepared = native.prepare_offline_clone(
        offline_root=source, destination=tmp_path / 'queue', manifest=manifest)
    other = native.prepare_offline_clone(
        offline_root=source, destination=tmp_path / 'task',
        manifest={**manifest, 'store_id': 'native-task-store'})
    queue = start(prepared, tmp_path / 'queue-owner')
    task = start(other, tmp_path / 'task-owner')
    issued, revoked = [], []
    def issue(request, **_):
        token, grant = task.issue_typed_client_grant_record(
            client_id=request['client_id'], process_birth_id=request['process_birth_id'],
            allowed_operations=('whoami_metadata',), peer_pid=request['pid'], ttl_seconds=60)
        issued.append(grant.grant_id)
        return {
            'schema': STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA, 'ok': True,
            'endpoint': task.identity.listen_uri,
            'socket_path': str(task.typed_command_socket_path()),
            'store_id': task.identity.store_id, 'server_id': task.identity.server_id,
            'client_id': request['client_id'], 'process_birth_id': request['process_birth_id'],
            'token': token, 'execution_route_policy': policy.to_dict(),
        }, grant.grant_id
    def revoke(client, grant_id):
        revoked.append((client, grant_id))
        task.revoke_typed_client_grant(grant_id)
    broker = broker_module.NativeOwnerMergeBroker(
        task_server=task, queue_server=queue,
        repository_id=manifest['repository_id'], target_branch=manifest['target_branch'],
        scope_bindings=manifest['scope_bindings'], issue_task=issue, revoke_task=revoke,
        validate_peer=lambda **_: '0', ttl_seconds=60)
    birth = current_process_birth()
    scope = manifest['scope_bindings'][0]
    request = dict(schema=bundle_module.REQUEST_SCHEMA, request_id='a' * 32,
        pid=birth.pid, process_birth=birth.to_dict(), process_birth_id=process_birth_id(birth),
        client_id='database-implementation-daemon:native-lane-0',
        store_id=task.identity.store_id, config_cid=scope['config_cid'], plan_cid=scope['plan_cid'])
    result = SimpleNamespace(broker=broker, task=task, queue=queue, request=request,
        manifest=manifest, scope=scope, issued=issued, revoked=revoked, tmp_path=tmp_path)
    try:
        yield result
    finally:
        broker.close()
        task.stop()
        queue.stop()


def admit(own, request=None):
    return own.broker.admit(request or own.request, peer_pid=os.getpid(), peer_uid=os.geteuid())


def bundle(own, response):
    return bundle_module.OwnerMergeBootstrapBundle.from_response(
        response, request=own.request, peer_pid=os.getpid(), peer_uid=os.geteuid())


def connect(credential):
    return TypedStateOwnerConnection(socket_path=credential.socket_path,
        token=credential.token, client_id=credential.client_id,
        process_birth_id=credential.process_birth_id, store_id=credential.store_id,
        timeout_seconds=2)


def api(own, connection):
    response = own.broker._issued[own.request['client_id']]['response']
    return OwnerRecoveryRuntimeClient(connection, repository_id=own.manifest['repository_id'],
        target_branch=own.manifest['target_branch'], consumer_id=own.request['client_id'],
        recovery_scope_cid=response['recovery_scope_cid'])


def test_exact_request_replay_preserves_three_roles_and_live_connections(paired):
    own = paired
    response = admit(own)
    replay = admit(own)
    assert replay == response
    replay['queue']['token'] = 'corrupted-outside-cache'
    assert admit(own) == response
    assert len(own.issued) == 1
    credentials = bundle(own, response)
    assert len({credentials.task.token, credentials.queue.token, credentials.recovery.token}) == 3
    channels = [connect(role) for role in (credentials.task, credentials.queue, credentials.recovery)]
    try:
        assert api(own, channels[2]).describe_scope() == own.scope
        for channel in channels[:2]:
            with pytest.raises(TypedStateOwnerError):
                api(own, channel).describe_scope()
        entry = own.broker._issued[own.request['client_id']]
        before = own.queue._command_gateway._grants[credentials.recovery.token]
        entry['renew_at'] = 0
        own.broker.maintain()
        after = own.queue._command_gateway._grants[credentials.recovery.token]
        assert after.grant_id == before.grant_id
        assert after.entity_scopes == before.entity_scopes
        assert after.expires_at >= before.expires_at
        assert api(own, channels[2]).describe_scope() == own.scope
        assert api(own, channels[2]).load_cursors()['revision'] == 0
        with own.queue._owner_transaction_lock:
            assert own.queue._connection.execute(
                'SELECT COUNT(*) FROM legacy_merge_recovery_leases').fetchone()[0] == 0
    finally:
        for channel in reversed(channels):
            channel.close()


@pytest.mark.parametrize('field,value', [
    ('request_id', 'bad'), ('pid', True), ('process_birth_id', 'stale'),
    ('client_id', []), ('client_id', ''), ('config_cid', 'foreign'),
    ('plan_cid', 'foreign'), ('store_id', 'foreign'), ('extra', 1),
])
def test_request_refused_before_any_grant(paired, field, value):
    request = {**paired.request, field: value}
    with pytest.raises(StateOwnerBootstrapError):
        admit(paired, request)
    assert paired.issued == []
    assert paired.queue._command_gateway._grants == {}


def test_live_client_cannot_rotate_request_or_bypass_revocation(paired):
    response = admit(paired)
    with pytest.raises(StateOwnerBootstrapError, match='replace'):
        admit(paired, {**paired.request, 'request_id': 'b' * 32})
    grant = paired.queue._command_gateway._grants[response['recovery']['token']]
    paired.queue.revoke_typed_client_grant(grant.grant_id)
    with pytest.raises(StateOwnerBootstrapError, match='no longer current'):
        admit(paired)
    assert len(paired.issued) == 1


@pytest.mark.parametrize('failure', ['second_role', 'oversized'])
def test_partial_issuance_revokes_all_authority(paired, monkeypatch, failure):
    if failure == 'second_role':
        original = paired.queue.issue_typed_client_grant_record
        count = 0
        def issue(**kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('recovery role failed')
            return original(**kwargs)
        monkeypatch.setattr(paired.queue, 'issue_typed_client_grant_record', issue)
    else:
        monkeypatch.setattr(broker_module, 'MAX_STATE_OWNER_BOOTSTRAP_BYTES', 128)
    with pytest.raises((RuntimeError, StateOwnerBootstrapError)):
        admit(paired)
    assert len(paired.revoked) == 1
    assert paired.task._command_gateway._grants == {}
    assert paired.queue._command_gateway._grants == {}
    assert paired.broker._issued == {}


def test_expiry_is_not_renewed_or_replayed(paired):
    response = admit(paired)
    gateway = paired.queue._command_gateway
    token = response['recovery']['token']
    with gateway._grants_lock:
        gateway._grants[token] = replace(gateway._grants[token], issued_at=1, expires_at=2)
    paired.broker._issued[paired.request['client_id']]['renew_at'] = 0
    with pytest.raises(TypedStateOwnerError, match='expired'):
        paired.broker.maintain()
    assert token not in gateway._grants
    with pytest.raises(StateOwnerBootstrapError):
        admit(paired)


def test_actual_birth_is_used_for_gateway_revalidation(paired):
    response = admit(paired)
    credential = response['queue']
    grant = paired.queue._command_gateway._grants[credential['token']]
    wrong_birth = replace(current_process_birth(), start_time_ticks=grant.peer_start_time_ticks + 1)
    with pytest.raises(TypedStateOwnerError, match='kernel peer'):
        paired.broker._live_grant(paired.queue, grant.grant_id, wrong_birth)


def test_lost_response_retries_same_request_on_inherited_listener(paired):
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(paired.tmp_path / 'broker.sock'))
    listener.listen(2)
    requests, failures = [], []
    def serve():
        try:
            for index in range(2):
                accepted, _ = listener.accept()
                with accepted:
                    accepted.settimeout(3)
                    pid, uid, _ = struct.unpack('3i', accepted.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i')))
                    request = bundle_module.receive_bundle_frame(accepted)
                    requests.append(request)
                    response = paired.broker.admit(request, peer_pid=pid, peer_uid=uid)
                    if index:
                        _send_frame(accepted, response)
        except BaseException as error:
            failures.append(error)
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    descriptor = os.dup(listener.fileno())
    try:
        credentials = bundle_module.request_owner_merge_bootstrap(descriptor,
            client_id=paired.request['client_id'], store_id=paired.request['store_id'],
            config_cid=paired.scope['config_cid'], plan_cid=paired.scope['plan_cid'],
            timeout_seconds=3)
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert not failures
        assert len(requests) == 2 and requests[0] == requests[1]
        assert len(paired.issued) == 1
        assert credentials.scope_binding == paired.scope
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert listener.fileno() >= 3
    finally:
        listener.close()


def test_bundle_requires_same_task_plan_and_scope(paired):
    response = admit(paired)
    original = copy.deepcopy(response)
    response['scope_binding']['plan_cid'] = 'plan:foreign'
    request = {**paired.request, 'plan_cid': 'plan:foreign'}
    with pytest.raises(StateOwnerBootstrapError, match='admitted launch source'):
        bundle_module.OwnerMergeBootstrapBundle.from_response(response,
            request=request, peer_pid=os.getpid(), peer_uid=os.geteuid())
    assert admit(paired) == original


def test_native_attachment_has_independent_channels_and_rejects_scope_drift(paired, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.merge import checkout_lock
    own = paired
    credentials = bundle(own, admit(own))
    args = dict(repository_root=own.tmp_path,
        attempt_root=own.tmp_path / 'attempts', board_namespace=own.scope['board_namespace'],
        lane_id='0', admitted_config_cid=own.scope['config_cid'], admitted_plan_cid=own.scope['plan_cid'])
    with pytest.raises(StateOwnerBootstrapError, match='current factory scope'):
        credentials.attach_merge_runtime(**{**args, 'lane_id': '1'})
    # This fixture uses a deliberately synthetic repository ID. The native
    # attachment still checks it against both queue and recovery clients.
    monkeypatch.setattr(checkout_lock, 'checkout_repository_id', lambda root: own.manifest['repository_id'])
    attached = credentials.attach_merge_runtime(**args)
    try:
        assert len(attached.connections) == 2
        assert attached.connections[0] is not attached.connections[1]
        assert attached.runtime.client.load_cursors()['revision'] == 0
        attached.runtime.validate_factory_binding(**{
            'repository_root': args['repository_root'], 'attempt_root': args['attempt_root'],
            'board_namespace': args['board_namespace'], 'lane_id': args['lane_id'],
            'target_branch': own.manifest['target_branch'],
            'admitted_config_cid': args['admitted_config_cid'], 'admitted_plan_cid': args['admitted_plan_cid']})
    finally:
        attached.close()
    # Closing channels does not consume/release any consumer fence.
    again = credentials.attach_merge_runtime(**args)
    again.close()


def test_frame_reader_accepts_full_policy_population_and_rejects_ambiguous_json():
    import json
    left, right = socket.socketpair()
    try:
        payload = {'entries': [{'task': str(i), 'revision': 1} for i in range(1000)]}
        _send_frame(left, payload)
        assert bundle_module.receive_bundle_frame(right) == payload
        for body in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":1.5}'):
            left.sendall(len(body).to_bytes(4, 'big') + body)
            with pytest.raises(StateOwnerBootstrapError, match='malformed'):
                bundle_module.receive_bundle_frame(right)
    finally:
        left.close()
        right.close()
