"""Unexecuted pure/private-socket controls, not native owner qualification.

The private peer actor checks framing/identity/freshness only. It never claims
to hold a task claim, SDK phase or disk reservation; no SDK is imported here.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
from pathlib import Path
import socket
import time

import pytest

import ipfs_accelerate_py.cli_runtime.grok_production_owner as core


def finish_context(value):
    value = dict(value)
    value['request_sha256'] = core.digest({'schema': core.CONTEXT_SCHEMA,
        **{key: value[key] for key in core.PRELAUNCH_FIELDS}})
    value['context_sha256'] = core.digest({key: row for key, row in value.items() if key != 'context_sha256'})
    return value


def context_template(root, endpoint, witness, owner, worker):
    deadline = time.monotonic()+20
    value = {key: 'test-'+key for key in core.PRELAUNCH_FIELDS}
    value.update(schema=core.CONTEXT_SCHEMA, database_attempt_id='test-attempt', native_attempt_id='test-attempt',
        database_attempt_number=1, portal_attempt=1, fencing_token=2, fence_epoch=3,
        allocation_session_id='test-owner-session', credential_slot='test-slot', credential_fingerprint='1'*16,
        allocation_store_path=str(root/'allocation.duckdb'), workspace=str(root),
        source_generation='grok-automatic-named-session/test-client-only',
        source_bindings_sha256='2'*64, capsule_sha256='3'*64, worker_source_sha256='4'*64,
        owner_source_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        work_deadline_monotonic=deadline, native_deadline_monotonic=deadline,
        owner_pid=owner['pid'], owner_birth=owner['birth'], worker_pid=worker['pid'],
        worker_birth=worker['birth'], worker_process_group=worker['group_pid'],
        endpoint=str(endpoint), socket=witness, nonce='5'*48)
    return finish_context(value)


def _private_client(connection, stages):
    os.setsid()
    connection.send(core.process(os.getpid()))
    context = connection.recv()
    try:
        client = core.ProductionOwnerGuardClient(context,
            expected_owner_pid=os.getppid(), expected_owner_birth=core.process(os.getppid())['birth'],
            expected_request_sha256=context['request_sha256'], expected_source_generation=context['source_generation'],
            expected_worker_source_sha256=context['worker_source_sha256'],
            expected_allocation_session_id=context['allocation_session_id'])
        guard = core.owner_guard_for(client)
        for stage in stages:
            guard(stage)
        receipts = client.receipts()
        receipts[0]['stage'] = 'caller-mutated-copy'
        connection.send({'successful': True, 'receipts': client.receipts(),
            'callback_source': guard.__code__.co_filename, 'native_authority_claimed': False})
    except BaseException as error:
        first = str(error) if isinstance(error, core.ProductionOwnerRefusal) else type(error).__name__
        latched = None
        if 'client' in locals():
            try:
                client.assert_current('worker_v4_response')
            except core.ProductionOwnerRefusal as retry:
                latched = str(retry)
        connection.send({'successful': False, 'error': first, 'latched': latched,
            'receipts': client.receipts() if 'client' in locals() else [], 'native_authority_claimed': False})
    finally:
        connection.close()


def socket_case(root, *, mode='normal', stages=('worker_v4_dispatch', 'worker_v4_response')):
    endpoint = root/'s'
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint)); endpoint.chmod(0o600); listener.listen(1); listener.settimeout(5)
    parent, child = multiprocessing.get_context('fork').Pipe()
    worker = multiprocessing.get_context('fork').Process(target=_private_client, args=(child, stages))
    worker.start(); child.close()
    try:
        assert parent.poll(5)
        birth = parent.recv()
        assert birth['pid'] == worker.pid and birth['group_pid'] == worker.pid
        context = context_template(root, endpoint, core.socket_identity(endpoint), core.process(os.getpid()), birth)
        if mode == 'expired':
            context['work_deadline_monotonic'] = time.monotonic()+2
            context = finish_context(context)
        elif mode == 'wrong_source':
            context['owner_source_sha256'] = '0'*64
            context = finish_context(context)
        elif mode == 'wrong_worker':
            context['worker_birth'] += 1
            context = finish_context(context)
        if mode == 'wrong_socket_gid':
            context['socket']['gid'] += 1
            context = finish_context(context)
        parent.send(context)
        requests = []
        for index, stage in enumerate(() if mode in {'expired', 'wrong_source', 'wrong_worker', 'wrong_socket_gid'} else stages, 1):
            channel, _ = listener.accept()
            with channel:
                channel.settimeout(2)
                observed = core.peer(channel)
                assert observed['pid'] == birth['pid'] and observed['birth'] == birth['birth']
                assert observed['group_pid'] == birth['group_pid'] and observed['parent_pid'] == os.getpid()
                request = core.receive(channel)
                assert request['sequence'] == index and request['stage'] == stage
                assert request['request_sha256'] == context['request_sha256']
                assert request['context_sha256'] == context['context_sha256']
                assert request['pid'] == observed['pid'] and request['birth'] == observed['birth']
                assert request['process_group'] == observed['group_pid']
                requests.append(request)
                response = {'schema': core.RPC_SCHEMA, 'status': 'current', 'sequence': index, 'stage': stage,
                    **{key: context[key] for key in ('request_sha256', 'context_sha256', 'nonce', 'owner_pid',
                        'owner_birth', 'worker_pid', 'worker_birth', 'worker_process_group',
                        'work_deadline_monotonic', 'native_deadline_monotonic')},
                    'observed_monotonic': time.monotonic()}
                if mode == 'wrong_context':
                    response['context_sha256'] = '0'*64
                elif mode == 'wrong_stage':
                    response['stage'] = 'headless_start'
                elif mode == 'stale':
                    response['observed_monotonic'] -= 100
                elif mode == 'extra':
                    response['native_grant'] = True
                elif mode == 'replay' and index == 2:
                    response['sequence'] = 1
                core.send(channel, response)
            if mode not in {'normal', 'replay'} or mode == 'replay' and index == 2:
                break
        assert parent.poll(5)
        result = parent.recv()
        worker.join(5)
        assert worker.exitcode == 0
        return context, requests, result
    finally:
        if worker.is_alive():
            worker.kill(); worker.join(5)
        parent.close(); listener.close(); endpoint.unlink()


def test_real_kernel_peer_ack_sequence_callback_and_copied_receipts(tmp_path):
    context, requests, result = socket_case(tmp_path)
    assert result['successful'] is True and result['native_authority_claimed'] is False
    assert len(requests) == len(result['receipts']) == 2
    assert [row['sequence'] for row in result['receipts']] == [1, 2]
    assert [row['stage'] for row in result['receipts']] == ['worker_v4_dispatch', 'worker_v4_response']
    assert Path(result['callback_source']).resolve() == Path(core.__file__).resolve()
    assert all(row['context_sha256'] == context['context_sha256'] and
        row['actual_kernel_peer_owner_verified'] is True and row['observation_is_not_native_authority'] is True
        for row in result['receipts'])


@pytest.mark.parametrize('mode', ['wrong_context', 'wrong_stage', 'stale', 'extra', 'replay', 'expired', 'wrong_socket_gid'])
def test_bound_ack_failures_latch_without_retry_or_authority(tmp_path, mode):
    _, _, result = socket_case(tmp_path, mode=mode)
    assert result['successful'] is False and result['native_authority_claimed'] is False
    assert result['latched'] == 'owner_client_failure_latched'
    assert len(result['receipts']) == (1 if mode == 'replay' else 0)


@pytest.mark.parametrize('mode', ['wrong_source', 'wrong_worker'])
def test_source_and_actual_worker_birth_refuse_before_socket_exchange(tmp_path, mode):
    _, requests, result = socket_case(tmp_path, mode=mode)
    assert result['successful'] is False and result['native_authority_claimed'] is False
    assert requests == [] and result['receipts'] == []


@pytest.mark.parametrize('change', ['claim_alias', 'allocation', 'source', 'deadline', 'unknown_field', 'nonce'])
def test_context_rejects_changed_commitments_before_rpc(tmp_path, change):
    owner = {'pid': 100, 'birth': 1}; worker = {'pid': 101, 'birth': 2, 'group_pid': 101}
    witness = {'device': 1, 'inode': 2, 'mode': 0o140600, 'uid': os.geteuid(), 'gid': os.getegid(), 'nlink': 1}
    value = context_template(tmp_path, tmp_path/'s', witness, owner, worker)
    assert core.validate_context(value) == value
    if change == 'claim_alias':
        value['native_attempt_id'] = 'other-claim-attempt'; value = finish_context(value)
    elif change == 'allocation':
        value['allocation_session_id'] = 'different-session'
    elif change == 'source':
        value['source_bindings_sha256'] = '0'*64
    elif change == 'deadline':
        value['work_deadline_monotonic'] += 1; value = finish_context(value)
    elif change == 'unknown_field':
        value['lease_key'] = 'must-never-be-in-context'
    else:
        value['nonce'] = 'bad'
    with pytest.raises(core.ProductionOwnerRefusal):
        core.validate_context(value)


def test_duplicate_json_keys_and_oversize_frame_are_closed():
    with pytest.raises(core.ProductionOwnerRefusal):
        core.decode(b'{"a":1,"a":2}')
    with pytest.raises(core.ProductionOwnerRefusal):
        core.decode(b'x'*(core.MAX_FRAME+1))


def test_callback_factory_rejects_arbitrary_noop():
    with pytest.raises(core.ProductionOwnerRefusal):
        core.owner_guard_for(lambda _stage: None)
