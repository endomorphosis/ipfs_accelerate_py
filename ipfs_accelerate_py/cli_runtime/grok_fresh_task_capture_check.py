"""FIRST post-return factual capture checker for a separately sealed fresh task.

Import performs no project imports, provider/DB/native/resource operations.
Metadata alone is never authority. Only first_after_stream_capture consumes
an actual retained normal handoff, Popen, original sealed FD and public store.
It observes native capture, never settles/accepts a task or proves model quality.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import types
import uuid

SCHEMA = 'fresh-task-grok-capture-check/v1'
MODULE = 'ipfs_accelerate_py.cli_runtime.grok_fresh_task_capture_check'
WORKER = 'ipfs_accelerate_py.cli_runtime.grok_fresh_task_capture_worker'
MAX_BYTES = 2 * 1024 * 1024
CAPTURE_FIELDS = ('allocation_session_id','provider','native_session_id','workspace','resume_style')
METHODS = ('initialize','session/new','_x.ai/session/rename','_x.ai/session/close')
STAGES = ('request_prepared','remaining_deadline','worker_v4_response')


class CaptureCheckRefusal(RuntimeError):
    """Closed metadata-only reason; no exception messages/credentials copied."""


def need(condition, reason):
    if not condition:
        raise CaptureCheckRefusal(reason)


def canonical(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,
                       separators=(',',':'))+'\n').encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def decode(raw):
    def unique(pairs):
        out={}
        for key,value in pairs:
            need(key not in out,'duplicate_JSON_key')
            out[key]=value
        return out
    def bad(_):
        raise CaptureCheckRefusal('nonfinite_JSON')
    try:
        value=json.loads(raw.decode('utf-8'),object_pairs_hook=unique,parse_constant=bad)
    except (ValueError,UnicodeError,RecursionError):
        raise CaptureCheckRefusal('invalid_JSON') from None
    need(type(value) is dict,'JSON_object_required')
    return value


def read_pin(pin, *, maximum=MAX_BYTES, private=False):
    need(type(pin) is dict and set(pin)=={'path','bytes','sha256'} and
         type(pin['path']) is str and type(pin['bytes']) is int and
         0<pin['bytes']<=maximum and type(pin['sha256']) is str and
         re.fullmatch('[0-9a-f]{64}',pin['sha256']),'exact_bounded_pin_required')
    path=Path(pin['path'])
    need(path.is_absolute() and path.resolve(strict=True)==path,'unaliased_pin_path_required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and
             before.st_size==pin['bytes'],'regular_pin_file_required')
        if private:
            need(before.st_uid==os.geteuid() and stat.S_IMODE(before.st_mode)==0o600,
                 'private_owned_pin_file_required')
        body=bytearray()
        while len(body)<=pin['bytes']:
            piece=os.read(fd,min(65536,pin['bytes']+1-len(body)))
            if not piece:break
            body.extend(piece)
        signature=lambda row:(row.st_dev,row.st_ino,row.st_mode,row.st_nlink,
                                  row.st_size,row.st_mtime_ns,row.st_ctime_ns)
        need(len(body)==pin['bytes'] and signature(before)==signature(os.fstat(fd))==
             signature(path.lstat()) and hashlib.sha256(body).hexdigest()==pin['sha256'],
             'pin_changed_or_digest_mismatch')
        return bytes(body)
    finally:
        os.close(fd)


def read_borrowed_sealed_request(fd, request):
    """Read without seeking, mutating or closing the issuer's original memfd."""
    need(type(fd) is int and fd>=0,'original_borrowed_metadata_FD_required')
    row=os.fstat(fd)
    seals=fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE
    need(stat.S_ISREG(row.st_mode) and 0<row.st_size<=65536 and
         fcntl.fcntl(fd,fcntl.F_GET_SEALS)==seals and
         fcntl.fcntl(fd,fcntl.F_GETFD)&fcntl.FD_CLOEXEC,'fully_sealed_original_metadata_FD_required')
    raw=os.pread(fd,row.st_size+1,0)
    need(len(raw)==row.st_size and raw==canonical(request),'sealed_fresh_request_differs')
    return hashlib.sha256(raw).hexdigest()


def validate_capture_metadata(*,request,context,transport,router_receipt,named_receipt,
                              bootstrap_frames,capture_row,owner_rows,
                              expected_stdin,expected_argv):
    """Pure factual joins; inert unit metadata never qualifies native authority."""
    need(type(request) is dict and request.get('schema')=='fresh-task-grok-capture-request/v1' and
         request.get('mode')=='fresh-task-capture-diagnostic' and request.get('one_shot') is True and
         type(request.get('attempt_cap')) is int and request['attempt_cap']==1 and
         request.get('provider_dispatch_allowed') is True and
         not any(key.startswith('original_task') or key.startswith('original_catalog') for key in request),
         'distinct_sealed_fresh_task_required')
    need(request.get('execution_environment') in ('isolated-loopback','provider'),
         'explicit_execution_environment_required')
    need(type(transport) is dict and transport.get('schema')=='fresh-task-grok-capture-transport/v1' and
         transport.get('metadata_only') is True and transport.get('capture_accepted_by_parent') is False and
         transport.get('task_completion_accepted') is False and transport.get('production_qualified') is False,
         'fresh_transport_observation_required')
    for key in ('task_alias','task_cid','database_attempt_id','database_attempt_number',
                'allocation_session_id','source_generation','execution_environment'):
        need(type(transport.get(key)) is type(request.get(key)) and transport.get(key)==request.get(key),
             'fresh_task_transport_binding_differs')
    need(type(request.get('database_attempt_number')) is int and request['database_attempt_number']==1,
         'single_actual_attempt_required')
    pairs={'task_alias':'portal_task_id','task_cid':'database_task_cid',
        'database_attempt_id':'database_attempt_id','database_attempt_number':'database_attempt_number',
        'allocation_session_id':'allocation_session_id','source_generation':'source_generation',
        'workspace':'workspace','owner_pid':'owner_pid','owner_birth':'owner_birth',
        'owner_request_sha256':'request_sha256'}
    for key,target in pairs.items():
        need(type(request.get(key)) is type(context.get(target)) and request.get(key)==context.get(target),
             'fresh_request_context_differs')
    need(context.get('schema')=='grok-production-owner-context/v1' and
         digest({k:v for k,v in context.items() if k!='context_sha256'})==context.get('context_sha256'),
         'actual_context_commitment_required')
    for pin_key,context_key in (('capsule_pin','capsule_sha256'),('source_profile_pin','source_bindings_sha256')):
        need(transport.get(pin_key)==request.get(pin_key) and
             request[pin_key]['sha256']==context.get(context_key),'capsule_or_source_profile_differs')
    need(transport.get('owner_request_sha256')==context['request_sha256'] and
         transport.get('sealed_request_sha256')==digest(request),'original_sealed_request_commitment_required')
    need(type(expected_stdin) is dict and set(expected_stdin)=={'bytes','sha256'} and
         type(expected_stdin['bytes']) is int and 0<expected_stdin['bytes']<=128000 and
         re.fullmatch('[0-9a-f]{64}',expected_stdin['sha256']) and
         transport.get('stdin')==expected_stdin,'actual_original_stdin_commitment_required')
    need(type(expected_argv) in (tuple,list) and all(type(x) is str for x in expected_argv) and
         transport.get('worker_argv')==list(expected_argv),'actual_accepted_worker_argv_required')
    worker=transport.get('worker_identity')
    need(type(worker) is dict and worker.get('pid')==context.get('worker_pid') and
         worker.get('birth')==context.get('worker_birth') and
         worker.get('group_pid')==context.get('worker_process_group')==context.get('worker_pid') and
         worker.get('parent_pid')==context.get('owner_pid'),'actual_worker_birth_group_required')
    work=request.get('work_deadline_monotonic');native=request.get('native_deadline_monotonic')
    need(all(type(x) in (int,float) and math.isfinite(x) for x in (work,native)) and work<native and
         context.get('work_deadline_monotonic')==work and context.get('native_deadline_monotonic')==native and
         transport.get('original_work_deadline_monotonic')==work and
         transport.get('original_native_deadline_monotonic')==native,'original_absolute_clocks_required')
    before=transport.get('store_capture_before');after=transport.get('store_capture_observation')
    need(type(before) is dict and type(before.get('row')) is dict and not before['row'] and
         type(after) is dict and before.get('allocation_store_path')==after.get('allocation_store_path')==
         context.get('allocation_store_path'),'fresh_public_pre_dispatch_capture_observation_required')
    clocks=[before.get('observed_monotonic'),transport.get('dispatch_monotonic'),
            transport.get('response_monotonic'),after.get('observed_monotonic'),transport.get('observed_monotonic')]
    need(all(type(x) in (int,float) and math.isfinite(x) for x in clocks) and
         clocks==sorted(clocks) and clocks[2]<work and clocks[-1]<native,'actual_dispatch_response_clock_joins_required')
    need(type(transport.get('timeout_seconds')) in (int,float) and 0<transport['timeout_seconds']<=
         request['maximum_timeout_seconds']<=90,'original_timeout_bound_required')
    need(type(named_receipt) is dict and named_receipt.get('schema')=='grok-owned-acp-named-session@1' and
         named_receipt.get('report_scope')=='bootstrap_close_only' and
         named_receipt.get('source_generation')==request['source_generation'] and
         named_receipt.get('allocation_session_id')==request['allocation_session_id'],
         'actual_owned_named_session_receipt_required')
    for key in ('bootstrap_successful','native_process_reaped','process_group_baseline_restored',
                'lifetime_closed','response_accepted','no_session_prompt_submitted'):
        need(named_receipt.get(key) is True,'native_named_session_not_accepted_and_closed')
    need(type(named_receipt.get('native_process_returncode')) is int and named_receipt['native_process_returncode']==0 and
         named_receipt.get('forced_cleanup') is False and named_receipt.get('native_session_state_written_by_client') is False and
         named_receipt.get('cleanup_error_types')==[] and named_receipt.get('failure_code') is None and
         named_receipt.get('request_methods')==list(METHODS),'native_bootstrap_cleanup_or_protocol_differs')
    session=named_receipt.get('session_id')
    try:parsed=uuid.UUID(session)
    except (ValueError,TypeError,AttributeError):raise CaptureCheckRefusal('accepted_native_UUID_required') from None
    need(str(parsed)==session and parsed.version==4,'accepted_native_UUID_required')
    frames={}
    for frame in bootstrap_frames:
        need(type(frame) is dict,'native_bootstrap_frame_required')
        if 'id' not in frame:continue
        ident=frame['id'];need(type(ident) is int and ident in (1,2,3,4) and ident not in frames and
            frame.get('jsonrpc')=='2.0' and 'method' not in frame and 'error' not in frame and
            type(frame.get('result')) is dict,'native_bootstrap_reply_population_required')
        frames[ident]=frame['result']
    need(set(frames)=={1,2,3,4} and type(frames[1].get('protocolVersion')) is int and
         frames[1]['protocolVersion']==1 and frames[2].get('sessionId')==session and
         frames[3].get('success') is True and frames[4].get('error') is None and
         type(frames[4].get('result')) is dict and frames[4]['result'].get('success') is True and
         frames[4]['result'].get('outcome')=='closed','native_accepted_UUID_reply_required')
    replies=named_receipt.get('replies');need(type(replies) is list and len(replies)==4,'native_reply_commitments_required')
    for index,row in enumerate(replies,1):
        hashed=hashlib.sha256(json.dumps(frames[index],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        need(row==dict(id=index,method=METHODS[index-1],response_sha256=hashed),'native_reply_commitment_mismatch')
    need(type(request.get('max_output_tokens')) is int and 1<=request['max_output_tokens']<=4096 and
         type(request.get('maximum_timeout_seconds')) is int and 1<=request['maximum_timeout_seconds']<=90,
         'fixed_output_and_timeout_bounds_required')
    policy=router_receipt.get('provider_invocation_policy',{})
    need(router_receipt.get('schema')=='router-implementation-invocation@1' and
         router_receipt.get('status')=='provider_returned' and router_receipt.get('provider')=='grok_cli' and
         router_receipt.get('model')=='grok-4.7' and router_receipt.get('purpose')=='planning' and
         type(router_receipt.get('router_calls')) is int and router_receipt['router_calls']==1 and
         router_receipt.get('workspace')==request['workspace'] and router_receipt.get('completion_authority') is False and
         router_receipt.get('requested_output_tokens')==request['max_output_tokens'] and
         router_receipt.get('native_prompt_sha256')==expected_stdin['sha256'] and
         router_receipt.get('native_prompt_bytes')==expected_stdin['bytes'] and
         type(router_receipt.get('timeout_seconds')) in (int,float) and
         0<router_receipt['timeout_seconds']<=transport['timeout_seconds'] and
         policy.get('owned_named_session_requested') is True and policy.get('owned_named_session')==named_receipt and
         policy.get('allocation_session_id')==request['allocation_session_id'] and
         policy.get('native_resume_session_id')==session and policy.get('native_resume_identity_verified') is True,
         'actual_router_native_session_join_required')
    expected=dict(allocation_session_id=request['allocation_session_id'],provider='grok_cli',
                  native_session_id=session,workspace=request['workspace'],resume_style='resume')
    need(type(capture_row) is dict and {k:capture_row.get(k) for k in CAPTURE_FIELDS}==expected and
         after.get('row')==expected,'postdrain_actual_allocation_capture_required')
    client=transport.get('owner_rpc_receipts');need(type(client) is list and len(client)==len(owner_rows) and client,
        'actual_owner_RPC_population_required')
    need(client[0].get('stage')=='request_prepared' and client[-1].get('stage')=='worker_v4_response' and
         any(x.get('stage')=='remaining_deadline' for x in client),'actual_owner_RPC_stages_required')
    for index,(left,right) in enumerate(zip(client,owner_rows),1):
        need(left.get('sequence')==right.get('sequence')==index and left.get('stage')==right.get('stage') and
             left.get('request_sha256')==right.get('request_sha256')==context['request_sha256'] and
             left.get('context_sha256')==right.get('context_sha256')==context['context_sha256'] and
             left.get('response_sha256')==right.get('response_sha256') and
             left.get('client_pid')==right.get('client_pid')==context['worker_pid'] and
             left.get('client_birth')==right.get('client_birth')==context['worker_birth'] and
             left.get('owner_pid')==right.get('owner_pid')==context['owner_pid'] and
             left.get('owner_birth')==right.get('owner_birth')==context['owner_birth'] and
             left.get('actual_kernel_peer_owner_verified') is True,
             'actual_owner_client_RPC_join_required')
        need(right.get('claim_binding')=={key:context[key] for key in (
            'database_task_cid','database_attempt_id','database_claim_id','database_attempt_number',
            'coordination_lease_id','owner_session_id','fencing_token','fence_epoch')} and
             right.get('resource_binding')=={key:context[key] for key in (
            'root_lease_id','phase_lease_id','disk_native_lease_id','disk_reservation_id')} and
             right.get('source_bindings_sha256')==context['source_bindings_sha256'],
             'actual_owner_claim_resource_source_join_required')
        observed=right.get('observed_monotonic');received=left.get('received_monotonic')
        need(all(type(x) in (int,float) and math.isfinite(x) for x in (observed,received)) and
             observed<=received<work and left.get('observed_monotonic')==observed,
             'actual_RPC_original_clock_required')
        response={'schema':'grok-production-owner-currentness/v1','status':'current',
            'sequence':index,'stage':right['stage'],**{key:context[key] for key in (
            'request_sha256','context_sha256','nonce','owner_pid','owner_birth','worker_pid',
            'worker_birth','worker_process_group','work_deadline_monotonic','native_deadline_monotonic')},
            'observed_monotonic':observed}
        need(digest(response)==right['response_sha256'],'actual_RPC_response_commitment_required')
    return dict(schema=SCHEMA,metadata_only=True,metadata_joins=True,accepted_native_session_id=session,
        observed_capture=expected,actual_ACK_count=len(owner_rows),execution_environment=request['execution_environment'],
        provider_quality='private-fake-provider' if request['execution_environment']=='isolated-loopback' else 'unassessed',
        public_pre_dispatch_empty_observed=True,durable_pre_dispatch_absence_proved=False,
        task_completion_accepted=False,native_qualified=False,production_qualified=False,Lean_admitted=False)


def _module(root,name,pins):
    module=sys.modules.get(name);path=root/Path(*name.split('.')).with_suffix('.py')
    need(type(module) is types.ModuleType and module.__file__==module.__spec__.origin==str(path) and
         path.resolve(strict=True)==path and pins.get(str(path))==hashlib.sha256(path.read_bytes()).hexdigest(),
         'actual_normal_source_pinned_module_required')
    return module


def _original_group_absent(group):
    need(type(group) is int and group>1 and group!=os.getpgrp(),'actual_original_worker_group_required')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:row=(path/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):continue
        need(int(row[2])!=group,'actual_worker_group_not_drained')


def first_after_stream_capture(*,handoff,completed,request,observation_pin,expected_stdin):
    """Own FIRST ordinary handoff callback and independently read capture.

    Requires the separately reviewed M2 accepted-FD/issuer/first-callback seam.
    Existing M1/offline state cannot be reinterpreted. No callback is replayed.
    Caller must still refuse diagnostic task completion after this observation.
    """
    need(type(request) is dict and request.get('schema')=='fresh-task-grok-capture-request/v1',
         'distinct_sealed_fresh_task_required')
    root=Path(request['accelerator_root']);module=sys.modules.get('ipfs_accelerate_py.cli_runtime.grok_production_handoff')
    need(type(module) is types.ModuleType and type(handoff) is module.ProductionProviderHandoff,
         'actual_retained_normal_handoff_required')
    pins=handoff._pins;_module(root,module.__name__,pins);_module(root,MODULE,pins)
    worker=_module(root,WORKER,pins);worker.validate_request(request)
    handoff._lifetime()
    need(handoff._private_offline is None and handoff._private_capture_checker_active is True and
         handoff._prepared is True and handoff._closed is False and handoff._completed is False and
         handoff._capture is None,'actual_M2_FIRST_callback_required')
    need(type(completed) is subprocess.CompletedProcess and type(handoff._process) is subprocess.Popen and
         type(completed.returncode) is int and completed.returncode==0 and
         tuple(completed.args)==handoff.command==tuple(handoff._process.args),
         'actual_retained_successful_Popen_and_argv_required')
    native=request['native_deadline_monotonic'];need(time.monotonic()<native,'original_native_callback_clock_expired')
    remaining=native-time.monotonic()
    need(remaining>0,'original_native_wait_clock_expired')
    need(handoff._process.wait(timeout=remaining)==0 and
         handoff._process.returncode==completed.returncode,'actual_retained_waitstatus_required')
    identity=handoff._child_identity;need(type(identity) is dict and identity['pid']==handoff._process.pid and
        identity['group_pid']==handoff._process.pid,'actual_retained_birth_required')
    _original_group_absent(identity['group_pid'])
    fd=handoff.private_capture_request_descriptor()
    sealed=read_borrowed_sealed_request(fd,request)
    need(observation_pin['path']==str(Path(request['receipt_directory'])/'capture-transport.json'),
         'exact_fresh_transport_receipt_path_required')
    directory=Path(request['receipt_directory'])
    need(directory.resolve(strict=True)==directory and
         handoff.owner._disk.directory in directory.parents and
         stat.S_IMODE(directory.lstat().st_mode)==0o700 and
         handoff.server.context_path.parent==directory,
         'actual_charged_capture_receipt_directory_required')
    transport=decode(read_pin(observation_pin,private=True))
    need(transport['context_pin']['path']==request['context_path']==str(handoff.server.context_path),
         'exact_actual_owner_context_path_required')
    need(sealed==transport['sealed_request_sha256'],'original_sealed_FD_receipt_mismatch')
    need(decode(read_pin(transport['request_pin'],maximum=65536,private=True))==request,
         'durable_original_request_snapshot_required')
    need(transport['worker_identity']==identity,'retained_Popen_birth_transport_differs')
    context=handoff.core.validate_context(handoff.core.decode(handoff.server._context_bytes))
    need(decode(read_pin(transport['context_pin'],maximum=8192,private=True))==context and
         handoff.capsule_pin==request['capsule_pin'] and handoff.source_bindings_pin==request['source_profile_pin'],
         'actual_handoff_context_capsule_profile_required')
    capsule=decode(read_pin(request['capsule_pin'],maximum=65536))
    read_pin(request['source_profile_pin'],maximum=65536)
    need(capsule.get('schema')=='legal-ir-grok-planning-transport-capsule/v4' and
         capsule.get('source_generation')==request['source_generation'],
         'unchanged_owner_capsule_and_fresh_generation_required')
    router=decode(read_pin(transport['router_receipt_pin'],private=True))
    need(router.get('provider_invocation_policy',{}).get('named_session_binary_sha256')==
         capsule['named_session']['binary']['sha256'],'actual_named_native_binary_commitment_required')
    named=decode(read_pin(transport['named_receipt_pin'],private=True))
    frames=[decode(line) for line in read_pin(transport['bootstrap_stdout_pin'],private=True).splitlines() if line.strip()]
    for path,sha in pins.items():
        need(handoff.factory.source_digest(path)==sha,'actual_complete_producer_source_changed')
    # Validate genuine post-drain observations BEFORE consuming FIRST. These
    # are public reads only; no row is seeded or changed to satisfy a check.
    before_row=handoff.allocation_store.get_cli_session(request['allocation_session_id'],'grok_cli')
    before_rows=handoff.server.observations()
    preliminary=validate_capture_metadata(request=request,context=context,transport=transport,
        router_receipt=router,named_receipt=named,bootstrap_frames=frames,capture_row=before_row,
        owner_rows=before_rows,expected_stdin=expected_stdin,expected_argv=list(handoff._process.args))
    need(time.monotonic()<native,'original_native_clock_expired_before_FIRST')
    # Sole FIRST call. The genuine consumer closes the exact listener after
    # actual group drain, demands real store capture, and sets _completed.
    handoff.after_stream_return(completed)
    need(handoff._completed is True and handoff._closed is True and handoff.server.closed is True and
         handoff._capture is not None,'actual_ordinary_capture_callback_not_accepted')
    row=handoff.allocation_store.get_cli_session(request['allocation_session_id'],'grok_cli')
    readback_monotonic=time.monotonic()
    rows=handoff.server.observations()
    result=validate_capture_metadata(request=request,context=context,transport=transport,
        router_receipt=router,named_receipt=named,bootstrap_frames=frames,capture_row=row,
        owner_rows=rows,expected_stdin=expected_stdin,expected_argv=list(handoff._process.args))
    need(result==preliminary and rows==before_rows and
         handoff._capture['row']==result['observed_capture'],
         'independent_public_capture_differs_from_FIRST')
    for path,sha in pins.items():
        need(handoff.factory.source_digest(path)==sha,'producer_source_changed_after_capture_readback')
    read_pin(observation_pin,private=True)
    need(time.monotonic()<native,'original_native_clock_expired_after_public_readback')
    _original_group_absent(identity['group_pid'])
    for path,sha in pins.items():
        need(handoff.factory.source_digest(path)==sha,'producer_source_changed_after_final_group_read')
    need(time.monotonic()<native,'original_native_final_acceptance_clock_expired')
    result.update(metadata_only=False,actual_retained_Popen_waitstatus_joined=True,
        actual_first_ordinary_capture_callback=True,actual_postdrain_public_capture_readback=True,
        actual_capture_accepted=True,actual_listener_closed=True,callback_replayed=False,
        observation_pin=dict(observation_pin),actual_waitstatus=handoff._process.returncode,
        actual_worker_identity=dict(identity),actual_worker_argv=list(handoff._process.args),
        actual_original_stdin_commitment=dict(expected_stdin),actual_context_pin=dict(transport['context_pin']),
        actual_owner_RPC_observations=json.loads(json.dumps(rows,allow_nan=False)),
        actual_capture_observation=dict(allocation_store_path=str(handoff.allocation_store.path),
            row=dict(result['observed_capture']),observed_monotonic=readback_monotonic),
        task_completion_accepted=False)
    return result
