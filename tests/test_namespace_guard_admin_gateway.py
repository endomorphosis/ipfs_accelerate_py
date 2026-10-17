"""Real owner transport/transaction controls; proof policy is a test boundary.

The historical source verifier has separate genuine-source controls. These
portable tests replace only that proof policy, never the gateway, peer grants,
DuckDB connection, evidence transaction or native transport.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource, NAMESPACE_GUARD_ADJUDICATION_EVIDENCE_KIND, TaskSourceIntegrityError,
    execute_quack_owner_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.todo_daemon import namespace_guard_adjudication as proof
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServer, QuackStateServerConfig,
)


def plain(value):
    if hasattr(value, 'items'):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def observe(source, connection):
    task=source.get_task('fixture-task')
    cursor=connection.execute('SELECT * FROM domain_events WHERE task_cid = ? ORDER BY global_sequence',
                              ['fixture-task'])
    rows=cursor.fetchall()
    if rows and isinstance(rows[0],Mapping):
        events=[dict(row) for row in rows]
    else:
        columns=[v[0] for v in cursor.description]
        events=[dict(zip(columns,row)) for row in rows]
    return plain({'task':task.to_dict(),
        'task_revision_history':source.task_revision_history_projection('fixture-task'),
        'plan_projection':source.plan_projection(task_cids=['fixture-task']),
        'domain_events':events})


@pytest.fixture
def native(tmp_path_factory,monkeypatch):
    root=tmp_path_factory.mktemp('g')
    path=root/'fixture.duckdb'
    repository=IntentRepository(path)
    repository.upsert_goal(goal_cid='fixture-goal',goal_alias='FIXTURE-G',title='Test only')
    repository.upsert_task(task_cid='fixture-task',task_alias='FIXTURE-T',goal_cid='fixture-goal',
                           status='in_progress',body={'objective':'Test only'})
    repository.close()
    connection=duckdb.connect(str(path))
    repository=IntentRepository(bound_connection=connection,install_schema=False)
    source=DatabaseTaskSource(intent=repository)
    before=observe(source,connection)
    context={'proof_id':'sha256:'+'7'*64,'fixture':'explicit test-only policy'}
    revision=before['task']['revision']
    scope={'task_cid':'fixture-task','proof_id':context['proof_id'],
           'expected_task_revision':revision,'recovery_scope_cid':'fixture-recovery-scope',
           'canonical_store_id':str(path),'logical_store_id':'fixture-logical-store'}
    expected_context=copy.deepcopy(context)
    calls=[]

    def scope_policy(value):
        if value!=expected_context:
            raise proof.NamespaceGuardAdjudicationError('fixture registered context differs')
        return dict(scope)

    def verify_policy(payload,*,authority_context,current_observation):
        calls.append(plain(current_observation))
        if (authority_context!=expected_context or type(payload) is not dict
                or set(payload)!={'operation','proof_id','expected_task_revision'}
                or payload!={'operation':proof.OPERATION,'proof_id':scope['proof_id'],
                             'expected_task_revision':revision}
                or current_observation!=before):
            raise proof.NamespaceGuardAdjudicationError('fixture proof CAS differs')
        return {'schema':'test-only-namespace-adjudication@1','task_cid':'fixture-task',
                'proof_id':scope['proof_id'],'unknown_preserved':True,
                'completion_authority':False,'automatic_retry_admitted':False}

    monkeypatch.setattr(proof,'namespace_guard_adjudication_scope',scope_policy)
    monkeypatch.setattr(proof,'verify_namespace_guard_adjudication',verify_policy)
    identity={'server_id':'fixture-owner','generation':1,'store_id':'fixture-logical-store'}
    gateway=typed.TypedStateOwnerGateway(connection=connection,socket_path=root/'owner.sock',
                                         store_id=identity['store_id'],identity=identity)
    clients=[]
    gateway.start()
    # The genuine gateway can install additive native event metadata. Freeze
    # the test preimage after that initialization, preserving every column.
    before=observe(source,connection)
    gateway.bind_namespace_guard_adjudication_service(expected_identity=identity,authority_context=context)
    payload={'operation':proof.OPERATION,'proof_id':scope['proof_id'],'expected_task_revision':revision}

    def client(token,grant):
        result=typed.TypedStateOwnerConnection(socket_path=gateway.socket_path,token=token,
            client_id=grant.client_id,process_birth_id=grant.process_birth_id,store_id=identity['store_id'])
        clients.append(result)
        return result

    def admin():
        return client(*gateway.issue_namespace_guard_adjudication_grant())

    def evidence_count():
        return connection.execute('SELECT COUNT(*) FROM evidence_nodes').fetchone()[0]

    yield SimpleNamespace(root=root,path=path,connection=connection,source=source,
        gateway=gateway,identity=identity,context=context,scope=scope,payload=payload,before=before,
        calls=calls,client=client,admin=admin,evidence_count=evidence_count)
    for client_instance in clients:
        client_instance.close()
    gateway.stop()
    source.close()
    connection.close()
    # Only this test's disposable fixture directory is removed. Pytest's own
    # current aliases remain in the separately precharged external temp root.
    shutil.rmtree(root)


def test_admin_audit_real_transport_is_idempotent_without_task_mutation(native):
    client=native.admin()
    try:
        first=client.adjudicate_namespace_guard_nonreachability(native.payload)
    except typed.TypedStateOwnerRemoteError:
        assert native.calls and native.calls[-1]==native.before
        raise
    second=client.adjudicate_namespace_guard_nonreachability(native.payload)
    assert first['replayed'] is False and second['replayed'] is True
    assert first['adjudication']==second['adjudication']
    assert first['audit_receipt']==second['audit_receipt']
    assert first['audit_receipt']['details']['evidence_kind']==proof.EVIDENCE_KIND
    assert native.evidence_count()==1
    assert native.source.get_task('fixture-task').to_dict()==native.before['task']
    assert len(native.calls)==2 and native.calls[0]==native.calls[1]==native.before
    assert proof.OPERATION not in typed.DATABASE_TASK_COMMANDS


def test_worker_grant_cannot_invoke_admin_service(native):
    token,grant=native.gateway.issue_grant(client_id='ordinary-worker',
        process_birth_id=typed.kernel_process_birth_id(os.getpid()),
        allowed_database_task_commands=('record_evidence',))
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        native.client(token,grant).adjudicate_namespace_guard_nonreachability(native.payload)
    assert native.calls==[] and native.evidence_count()==0


@pytest.mark.parametrize('change', ['client','scope','operations','commands','database_commands','birth'])
def test_admin_service_refuses_wider_or_foreign_grants(native,change):
    kwargs={'client_id':typed.NAMESPACE_GUARD_ADJUDICATION_ADMIN_CLIENT_ID,
        'process_birth_id':typed.kernel_process_birth_id(os.getpid()),
        'allowed_operations':(proof.OPERATION,),
        'entity_scopes':{'task_cid':'fixture-task','recovery_scope_cid':'fixture-recovery-scope'}}
    if change=='client':kwargs['client_id']='ordinary-worker'
    elif change=='scope':kwargs['entity_scopes']['task_cid']='another-task'
    elif change=='operations':kwargs['allowed_operations']+=(typed.HASH_OBSERVATION_SERVICE_OPERATION,)
    elif change=='commands':kwargs['allowed_command_operations']=('task.status.cas',)
    elif change=='database_commands':kwargs['allowed_database_task_commands']=('record_evidence',)
    elif change=='birth':kwargs['process_birth_id']='foreign-birth'
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        native.gateway.issue_grant(**kwargs)
    assert native.evidence_count()==0


def test_registered_admin_context_is_frozen_and_cannot_be_replaced(native):
    native.context['proof_id']='sha256:'+'8'*64
    result=native.admin().adjudicate_namespace_guard_nonreachability(native.payload)
    assert result['adjudication']['proof_id']==native.scope['proof_id']
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        native.gateway.bind_namespace_guard_adjudication_service(
            expected_identity=native.identity,authority_context=native.context)


@pytest.mark.parametrize('field',['pid','birth'])
def test_admin_grant_requires_original_owner_creator(native,field):
    if field=='pid':
        native.gateway._namespace_guard_adjudication_creator_pid=os.getpid()+1
    else:
        native.gateway._namespace_guard_adjudication_creator_birth='different-birth'
    with pytest.raises(typed.TypedStateOwnerAuthorizationError):
        native.gateway.issue_namespace_guard_adjudication_grant()
    assert native.evidence_count()==0


@pytest.mark.parametrize('mismatch',['logical_alias','physical_path'])
def test_historical_context_cannot_bind_to_another_canonical_store(native,mismatch):
    identity=dict(native.identity)
    if mismatch=='logical_alias':
        identity['store_id']='another-logical-store'
        connection=native.connection
    else:
        connection=duckdb.connect(str(native.root/'another.duckdb'))
    gateway=typed.TypedStateOwnerGateway(connection=connection,
        socket_path=native.root/'other.sock',store_id=identity['store_id'],identity=identity)
    gateway.start()
    try:
        with pytest.raises(typed.TypedStateOwnerAuthorizationError,match='canonical store differs'):
            gateway.bind_namespace_guard_adjudication_service(
                expected_identity=identity,authority_context=native.context)
    finally:
        gateway.stop()
        if mismatch=='physical_path':
            connection.close()
    assert native.evidence_count()==0


def test_stale_revision_and_extra_payload_fields_do_not_write_audit(native):
    client=native.admin()
    for changed in [dict(native.payload,expected_task_revision=99),
                    dict(native.payload,path='/untrusted/path')]:
        with pytest.raises(typed.TypedStateOwnerRemoteError):
            client.adjudicate_namespace_guard_nonreachability(changed)
    assert native.evidence_count()==0


def test_replay_after_native_task_movement_refuses_fresh_authority(native):
    client=native.admin()
    first=client.adjudicate_namespace_guard_nonreachability(native.payload)
    native.source.intent.upsert_task(task_cid='fixture-task',task_alias='FIXTURE-T',
        goal_cid='fixture-goal',status='failed',body={'objective':'moved fixture'},expected_revision=1)
    watermark=native.source.intent.event_watermark()
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        client.adjudicate_namespace_guard_nonreachability(native.payload)
    assert native.evidence_count()==1
    assert native.source.intent.event_watermark()==watermark
    assert native.source.get_task('fixture-task').status=='failed'
    assert first['audit_receipt']['event_id']


def test_audit_insert_and_native_event_rollback_together(native,monkeypatch):
    original=IntentRepository.record_evidence
    def interrupted(self,**kwargs):
        original(self,**kwargs)
        raise RuntimeError('fixture transaction interruption after both inserts')
    watermark=native.source.intent.event_watermark()
    with monkeypatch.context() as scoped:
        scoped.setattr(IntentRepository,'record_evidence',interrupted)
        with pytest.raises(typed.TypedStateOwnerRemoteError):
            native.admin().adjudicate_namespace_guard_nonreachability(native.payload)
    assert native.evidence_count()==0
    assert native.source.intent.event_watermark()==watermark
    assert native.admin().adjudicate_namespace_guard_nonreachability(native.payload)['replayed'] is False


def test_reserved_audit_kind_is_refused_by_actual_worker_command(native):
    def handle(command,payload,request_id,grant):
        return execute_quack_owner_command(native.source.intent,command,payload,
            request_id=request_id,store_id=str(native.path),store_generation='1')
    native.gateway.bind_database_task_command_handler(handle)
    token,grant=native.gateway.issue_grant(client_id='ordinary-worker',
        process_birth_id=typed.kernel_process_birth_id(os.getpid()),
        allowed_database_task_commands=('record_evidence',))
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        native.client(token,grant).execute_database_task_command('record_evidence',{
            'task_cid':'fixture-task','evidence_kind':NAMESPACE_GUARD_ADJUDICATION_EVIDENCE_KIND,
            'digest':'pretend-proof','body':{}},command_request_id=uuid.uuid4().hex)
    assert native.evidence_count()==0 and native.calls==[]
    with pytest.raises(TaskSourceIntegrityError):
        native.source.record_evidence(task_cid='fixture-task',
            evidence_kind=NAMESPACE_GUARD_ADJUDICATION_EVIDENCE_KIND,digest='pretend-proof')


def test_response_loss_preserves_exact_proof_and_safe_lookup_replay(native,monkeypatch):
    client=native.admin()
    original=client._request
    def lost_response(*args,**kwargs):
        original(*args,**kwargs)
        raise OSError('fixture response loss after actual native commit')
    with monkeypatch.context() as scoped:
        scoped.setattr(client,'_request',lost_response)
        with pytest.raises(typed.TypedStateOwnerNamespaceGuardOutcomeUnknownError) as caught:
            client.adjudicate_namespace_guard_nonreachability(native.payload)
    assert caught.value.proof_id==native.payload['proof_id']
    assert native.evidence_count()==1
    assert client.adjudicate_namespace_guard_nonreachability(native.payload)['replayed'] is True


def test_replay_refuses_tampered_evidence_row(native):
    client=native.admin()
    client.adjudicate_namespace_guard_nonreachability(native.payload)
    native.connection.execute('UPDATE evidence_nodes SET digest = ?', ['tampered-fixture'])
    watermark=native.source.intent.event_watermark()
    with pytest.raises(typed.TypedStateOwnerRemoteError):
        client.adjudicate_namespace_guard_nonreachability(native.payload)
    assert native.source.intent.event_watermark()==watermark


def test_public_quack_server_original_owner_binder_grant_and_transport(tmp_path_factory,monkeypatch):
    root=tmp_path_factory.mktemp('q')
    path=root/'canonical.duckdb'
    repository=IntentRepository(path)
    repository.upsert_goal(goal_cid='fixture-goal',goal_alias='FIXTURE-G',title='Test only')
    repository.upsert_task(task_cid='fixture-task',task_alias='FIXTURE-T',goal_cid='fixture-goal',
                           status='in_progress',body={'objective':'Test only'})
    repository.close()
    before=None
    context={'proof_id':'sha256:'+'9'*64,'fixture':'public original Quack owner test policy'}
    scope={'task_cid':'fixture-task','proof_id':context['proof_id'],
           'expected_task_revision':1,'recovery_scope_cid':'public-fixture-scope',
           'canonical_store_id':str(path),'logical_store_id':'fixture-quack-store'}
    payload={'operation':proof.OPERATION,'proof_id':context['proof_id'],'expected_task_revision':1}
    def scope_policy(value):
        assert value==context
        return dict(scope)
    def verify_policy(selected,*,authority_context,current_observation):
        assert selected==payload and authority_context==context and current_observation==before
        return {'task_cid':'fixture-task','proof_id':context['proof_id'],
                'unknown_preserved':True,'completion_authority':False,'automatic_retry_admitted':False}
    monkeypatch.setattr(proof,'namespace_guard_adjudication_scope',scope_policy)
    monkeypatch.setattr(proof,'verify_namespace_guard_adjudication',verify_policy)
    server=QuackStateServer(QuackStateServerConfig(database_path=path,state_dir=root/'owner',
        host='127.0.0.1',port=0,repository_id='fixture-repository',store_id='fixture-quack-store',
        secret_handle='env://IPFS_NAMESPACE_GUARD_TEST_ONLY_QUACK_TOKEN',allow_experimental=False))
    client=None
    started=False
    try:
        identity=server.start();started=True
        assert server.ready()['ready'] is True
        # Observe through the SAME genuine owner handle and original lock;
        # no second file connection is opened while the owner is live.
        with server._lock:
            source=DatabaseTaskSource(intent=IntentRepository(
                bound_connection=server._connection,install_schema=False))
            try:before=observe(source,server._connection)
            finally:source.close()
        assert dict(server.bind_namespace_guard_adjudication_service(authority_context=context))==scope
        birth=typed.kernel_process_birth_id(os.getpid())
        token,grant=server.issue_typed_client_grant_record(
            client_id=typed.NAMESPACE_GUARD_ADJUDICATION_ADMIN_CLIENT_ID,
            process_birth_id=birth,allowed_operations=(proof.OPERATION,),
            entity_scopes={'task_cid':'fixture-task','recovery_scope_cid':'public-fixture-scope'},
            ttl_seconds=60.0)
        assert not grant.allowed_database_task_commands and not grant.allowed_command_operations
        client=typed.TypedStateOwnerConnection(socket_path=server.typed_command_socket_path(),
            token=token,client_id=grant.client_id,process_birth_id=birth,store_id=identity.store_id)
        result=client.adjudicate_namespace_guard_nonreachability(payload)
        assert result['replayed'] is False and result['adjudication']['unknown_preserved'] is True
    finally:
        if client is not None:client.close()
        if started:server.stop()
        shutil.rmtree(root)
