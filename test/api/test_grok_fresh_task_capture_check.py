"""Inert metadata integrity + real private file/memfd refusal controls.

No positive runtime, native session, provider, allocation store or capability
is simulated. In-memory capture rows/UUID below test only a metadata reader;
no grok_cli row is inserted, no FIRST callback is replayed, no DB is opened.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

import ipfs_accelerate_py.cli_runtime.grok_fresh_task_capture_check as check


@pytest.fixture
def metadata():
    # All identities and times are INERT reader input, never actual authority.
    request=dict(schema='fresh-task-grok-capture-request/v1',mode='fresh-task-capture-diagnostic',
        one_shot=True,attempt_cap=1,provider_dispatch_allowed=True,execution_environment='isolated-loopback',
        accelerator_root='/inert/source',workspace='/inert/work',receipt_directory='/inert/receipts',
        task_alias='fresh-inert-task',task_cid='inert-task-cid',database_attempt_id='inert-attempt',
        database_attempt_number=1,allocation_session_id='inert-allocation',source_generation='grok-automatic-named-session/inert',
        work_deadline_monotonic=100.0,native_deadline_monotonic=160.0,owner_pid=10,owner_birth=20,
        owner_request_sha256='a'*64,context_path='/inert/receipts/owner-context.json',
        capsule_pin=dict(path='/inert/capsule',bytes=1,sha256='b'*64),
        source_profile_pin=dict(path='/inert/profile',bytes=1,sha256='c'*64),
        max_output_tokens=64,maximum_timeout_seconds=30)
    context=dict(schema='grok-production-owner-context/v1',portal_task_id=request['task_alias'],
        database_task_cid=request['task_cid'],database_attempt_id=request['database_attempt_id'],database_attempt_number=1,
        allocation_session_id=request['allocation_session_id'],source_generation=request['source_generation'],
        workspace=request['workspace'],owner_pid=10,owner_birth=20,worker_pid=30,worker_birth=40,worker_process_group=30,
        request_sha256=request['owner_request_sha256'],capsule_sha256='b'*64,source_bindings_sha256='c'*64,
        work_deadline_monotonic=100.0,native_deadline_monotonic=160.0,allocation_store_path='/inert/allocation.duckdb',
        nonce='d'*48,database_claim_id='inert-claim',coordination_lease_id='inert-lease',
        owner_session_id='inert-owner',fencing_token=1,fence_epoch=1,root_lease_id='inert-root',
        phase_lease_id='inert-phase',disk_native_lease_id='inert-disk-native',disk_reservation_id='inert-disk')
    context['context_sha256']=check.digest(context)
    session='12345678-1234-4234-8234-123456789abc'
    results=[dict(protocolVersion=1),dict(sessionId=session),dict(success=True),
             dict(error=None,result=dict(success=True,outcome='closed'))]
    frames=[dict(jsonrpc='2.0',id=i,result=x)for i,x in enumerate(results,1)]
    named=dict(schema='grok-owned-acp-named-session@1',report_scope='bootstrap_close_only',
        source_generation=request['source_generation'],allocation_session_id=request['allocation_session_id'],
        session_id=session,bootstrap_successful=True,native_process_reaped=True,process_group_baseline_restored=True,
        lifetime_closed=True,response_accepted=True,no_session_prompt_submitted=True,native_process_returncode=0,
        forced_cleanup=False,native_session_state_written_by_client=False,cleanup_error_types=[],failure_code=None,
        request_methods=list(check.METHODS),replies=[dict(id=i,method=check.METHODS[i-1],
            response_sha256=hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
            for i,x in enumerate(results,1)])
    row=dict(allocation_session_id=request['allocation_session_id'],provider='grok_cli',native_session_id=session,
        workspace=request['workspace'],resume_style='resume')
    stdin=dict(bytes=12,sha256='e'*64);argv=['/usr/bin/python3','-B','/inert/source/fresh-worker.py']
    transport=dict(schema='fresh-task-grok-capture-transport/v1',metadata_only=True,capture_accepted_by_parent=False,
        task_completion_accepted=False,production_qualified=False,owner_request_sha256=context['request_sha256'],
        sealed_request_sha256=check.digest(request),stdin=dict(stdin),worker_argv=list(argv),
        worker_identity=dict(pid=30,birth=40,group_pid=30,parent_pid=10),
        original_work_deadline_monotonic=100.0,original_native_deadline_monotonic=160.0,
        store_capture_before=dict(allocation_store_path=context['allocation_store_path'],row={},observed_monotonic=1.0),
        dispatch_monotonic=2.0,response_monotonic=3.0,
        store_capture_observation=dict(allocation_store_path=context['allocation_store_path'],row=dict(row),observed_monotonic=4.0),
        observed_monotonic=5.0,timeout_seconds=20)
    for key in('task_alias','task_cid','database_attempt_id','database_attempt_number','allocation_session_id',
               'source_generation','execution_environment','capsule_pin','source_profile_pin'):transport[key]=copy.deepcopy(request[key])
    owner_rows=[];client=[]
    for i,stage in enumerate(check.STAGES,1):
        response={'schema':'grok-production-owner-currentness/v1','status':'current','sequence':i,'stage':stage,
            **{k:context[k]for k in('request_sha256','context_sha256','nonce','owner_pid','owner_birth','worker_pid',
                'worker_birth','worker_process_group','work_deadline_monotonic','native_deadline_monotonic')},
            'observed_monotonic':float(i)}
        common=dict(sequence=i,stage=stage,request_sha256=context['request_sha256'],context_sha256=context['context_sha256'],
            response_sha256=check.digest(response),client_pid=30,client_birth=40,owner_pid=10,owner_birth=20,
            observed_monotonic=float(i))
        owner_rows.append(dict(common,claim_binding={k:context[k]for k in('database_task_cid','database_attempt_id',
            'database_claim_id','database_attempt_number','coordination_lease_id','owner_session_id','fencing_token','fence_epoch')},
            resource_binding={k:context[k]for k in('root_lease_id','phase_lease_id','disk_native_lease_id','disk_reservation_id')},
            source_bindings_sha256=context['source_bindings_sha256']))
        client.append(dict(common,actual_kernel_peer_owner_verified=True,received_monotonic=float(i)+0.1))
    transport['owner_rpc_receipts']=client
    router=dict(schema='router-implementation-invocation@1',status='provider_returned',provider='grok_cli',
        model='grok-4.7',purpose='planning',router_calls=1,workspace=request['workspace'],completion_authority=False,
        requested_output_tokens=64,native_prompt_bytes=12,native_prompt_sha256='e'*64,timeout_seconds=20,provider_invocation_policy=dict(owned_named_session_requested=True,
        owned_named_session=copy.deepcopy(named),allocation_session_id=request['allocation_session_id'],
        native_resume_session_id=session,native_resume_identity_verified=True))
    return dict(request=request,context=context,transport=transport,router_receipt=router,named_receipt=named,
        bootstrap_frames=frames,capture_row=row,owner_rows=owner_rows,expected_stdin=stdin,expected_argv=argv)


def test_inert_metadata_join_never_qualifies_native_or_task(metadata):
    result=check.validate_capture_metadata(**metadata)
    assert result['metadata_only']is True and result['metadata_joins']is True
    assert result['provider_quality']=='private-fake-provider'
    assert result['durable_pre_dispatch_absence_proved']is False
    assert result['task_completion_accepted']is result['native_qualified']is result['production_qualified']is False


@pytest.mark.parametrize('mutation',[
    'old_request','old_transport','original_task','attempt_bool','context_task','stdin','argv','birth','deadline',
    'preseeded_row','late_response','unaccepted_session','bootstrap_UUID','bootstrap_commitment','duplicate_reply',
    'router_UUID','capture_UUID','capture_workspace','RPC_digest','RPC_claim','RPC_sequence','RPC_deadline'])
def test_unjoined_or_replayed_capture_metadata_refuses(metadata,mutation):
    q=metadata['request'];c=metadata['context'];t=metadata['transport'];n=metadata['named_receipt'];r=metadata['router_receipt']
    if mutation=='old_request':q['schema']='private-M1a-sealed-offline-ACK-request/v1'
    elif mutation=='old_transport':t['schema']='private-M1a-offline-ACK-observation/v1'
    elif mutation=='original_task':q['original_task_cid']='historical-original-task'
    elif mutation=='attempt_bool':q['database_attempt_number']=True
    elif mutation=='context_task':c['database_task_cid']='other'
    elif mutation=='stdin':t['stdin']['bytes']=13
    elif mutation=='argv':t['worker_argv'].append('--unaccepted')
    elif mutation=='birth':t['worker_identity']['birth']=41
    elif mutation=='deadline':t['original_work_deadline_monotonic']=101.0
    elif mutation=='preseeded_row':t['store_capture_before']['row']=dict(metadata['capture_row'])
    elif mutation=='late_response':t['response_monotonic']=101.0
    elif mutation=='unaccepted_session':n['response_accepted']=False
    elif mutation=='bootstrap_UUID':metadata['bootstrap_frames'][1]['result']['sessionId']='00000000-0000-4000-8000-000000000000'
    elif mutation=='bootstrap_commitment':n['replies'][1]['response_sha256']='0'*64
    elif mutation=='duplicate_reply':metadata['bootstrap_frames'].append(copy.deepcopy(metadata['bootstrap_frames'][1]))
    elif mutation=='router_UUID':r['provider_invocation_policy']['native_resume_session_id']='00000000-0000-4000-8000-000000000000'
    elif mutation=='capture_UUID':metadata['capture_row']['native_session_id']='00000000-0000-4000-8000-000000000000'
    elif mutation=='capture_workspace':metadata['capture_row']['workspace']='/other'
    elif mutation=='RPC_digest':metadata['owner_rows'][0]['response_sha256']='0'*64;t['owner_rpc_receipts'][0]['response_sha256']='0'*64
    elif mutation=='RPC_claim':metadata['owner_rows'][0]['claim_binding']['database_claim_id']='other'
    elif mutation=='RPC_sequence':t['owner_rpc_receipts'][1]['sequence']=1
    elif mutation=='RPC_deadline':t['owner_rpc_receipts'][0]['received_monotonic']=101.0
    with pytest.raises(check.CaptureCheckRefusal):check.validate_capture_metadata(**metadata)


def test_genuine_FIRST_entry_rejects_metadata_without_real_handoff(metadata):
    with pytest.raises(check.CaptureCheckRefusal,match='actual_retained_normal_handoff_required'):
        check.first_after_stream_capture(handoff=None,completed=None,request=metadata['request'],
            observation_pin={},expected_stdin=metadata['expected_stdin'])


def test_original_real_sealed_memfd_read_preserves_issuer_FD_and_offset(metadata):
    q=metadata['request'];raw=check.canonical(q)
    fd=os.memfd_create('fresh-capture-unit-metadata',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    try:
        os.write(fd,raw);os.lseek(fd,2,os.SEEK_SET)
        seals=fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE
        fcntl.fcntl(fd,fcntl.F_ADD_SEALS,seals)
        assert check.read_borrowed_sealed_request(fd,q)==hashlib.sha256(raw).hexdigest()
        assert os.lseek(fd,0,os.SEEK_CUR)==2 and os.fstat(fd).st_size==len(raw)
        with pytest.raises(check.CaptureCheckRefusal,match='sealed_fresh_request_differs'):
            check.read_borrowed_sealed_request(fd,dict(q,task_alias='changed'))
    finally:os.close(fd)


def test_unsealed_real_memfd_is_not_accepted_metadata(metadata):
    fd=os.memfd_create('fresh-capture-unit-unsealed',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    try:
        os.write(fd,check.canonical(metadata['request']))
        with pytest.raises(check.CaptureCheckRefusal,match='fully_sealed_original_metadata_FD_required'):
            check.read_borrowed_sealed_request(fd,metadata['request'])
    finally:os.close(fd)


def test_real_private_receipt_file_tamper_is_refused(tmp_path):
    p=tmp_path/'receipt.json';raw=b'{"metadata_only":true}\n';p.write_bytes(raw);p.chmod(0o600)
    pin=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert check.read_pin(pin,private=True)==raw
    p.write_bytes(b'{"metadata_only":false}')
    with pytest.raises(check.CaptureCheckRefusal):check.read_pin(pin,private=True)


def test_duplicate_JSON_receipt_keys_are_not_reinterpreted():
    with pytest.raises(check.CaptureCheckRefusal,match='duplicate_JSON_key'):
        check.decode(b'{"schema":"fresh","schema":"old"}')
