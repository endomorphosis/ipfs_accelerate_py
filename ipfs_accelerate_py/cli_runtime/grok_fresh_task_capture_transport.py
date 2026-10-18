"""Fresh-task transport over existing genuine Grok named-session APIs.

Top level is stdlib plus the inert sibling worker. No model, SDK, database,
resource acquisition or dispatch occurs on import. JSON supplies no authority.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import types
import uuid
from . import grok_fresh_task_capture_worker as worker

CAPSULE_SCHEMA='legal-ir-grok-planning-transport-capsule/v4'
SOURCE_SCHEMA='fresh-task-grok-capture-source-profile/v1'
FILES=frozenset({'worker','runtime','helper','owner_source','adapter'})
NAMED_FIELDS=frozenset({'native_home','audit_directory','preparation_home','binary','configuration',
    'owner_sampling_policy','system_requirements','cleanup_reserve_seconds','bootstrap_max_seconds'})
CAPTURE_FIELDS=('allocation_session_id','provider','native_session_id','workspace','resume_style')
METHODS=['initialize','session/new','_x.ai/session/rename','_x.ai/session/close']
RELATIVE={
 'worker':'ipfs_accelerate_py/cli_runtime/grok_fresh_task_capture_worker.py',
 'adapter':'ipfs_accelerate_py/cli_runtime/grok_fresh_task_capture_transport.py',
 'runtime':'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
 'helper':'ipfs_accelerate_py/cli_runtime/grok_named_session.py',
 'owner_source':'ipfs_accelerate_py/cli_runtime/grok_production_owner.py'}
MODULES={
 'ipfs_accelerate_py':'ipfs_accelerate_py/__init__.py',
 'ipfs_accelerate_py.cli_runtime.grok_fresh_task_capture_worker':RELATIVE['worker'],
 'ipfs_accelerate_py.cli_runtime.grok_fresh_task_capture_transport':RELATIVE['adapter'],
 'ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner':RELATIVE['runtime'],
 'ipfs_accelerate_py.cli_runtime.grok_named_session':RELATIVE['helper'],
 'ipfs_accelerate_py.cli_runtime.grok_production_owner':RELATIVE['owner_source'],
 'ipfs_accelerate_py.llm_router':'ipfs_accelerate_py/llm_router.py',
 'ipfs_accelerate_py.llm_allocation':'ipfs_accelerate_py/llm_allocation/__init__.py',
 'ipfs_accelerate_py.llm_allocation.duckdb_store':'ipfs_accelerate_py/llm_allocation/duckdb_store.py'}
_DISPATCH_ATTEMPTED=False
need=worker.need
canonical=worker.canonical

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def source_pin(path,expected):
    path=Path(path)
    pin=dict(path=str(path),bytes=path.lstat().st_size,sha256=expected)
    worker.read_pin(pin)
    return pin

def validate_capsule_shape(capsule,request):
    need(type(capsule)is dict and set(capsule)=={'schema','source_generation','files','named_session'} and
         capsule['schema']==CAPSULE_SCHEMA and capsule['source_generation']==request['source_generation'],
         'generic_V4_capsule_and_fresh_generation_required')
    files=capsule['files'];need(type(files)is dict and set(files)==FILES,'closed_five_source_roles_required')
    root=Path(request['accelerator_root'])
    for role,pin in files.items():
        worker.pin_shape(pin)
        need(pin['path']==str(root/RELATIVE[role]),'fixed_normal_source_role_path_required')
    named=capsule['named_session'];need(type(named)is dict and set(named)==NAMED_FIELDS,'closed_named_session_inputs_required')
    for name in ('binary','configuration','owner_sampling_policy'):
        worker.pin_shape(named[name],256*1024**2 if name=='binary'else 65536)
    if named['system_requirements']is not None:worker.pin_shape(named['system_requirements'],65536)
    for name in ('native_home','audit_directory','preparation_home'):
        p=named[name];need(type(p)is str and Path(p).is_absolute() and '..'not in Path(p).parts,'owned_named_path_shape_required')
    for name,upper in (('cleanup_reserve_seconds',10),('bootstrap_max_seconds',30)):
        n=named[name];need(type(n)in (int,float) and math.isfinite(n) and 1<=n<=upper,'finite_named_lifetime_input_required')
    return capsule

def load_source_profile(request):
    profile=worker.decode(worker.read_pin(request['source_profile_pin'],worker.MAX_REQUEST,private=True))
    need(set(profile)=={'schema','source_generation','sources'} and profile['schema']==SOURCE_SCHEMA and
         profile['source_generation']==request['source_generation'] and type(profile['sources'])is dict and
         1<=len(profile['sources'])<=512,'closed_normal_source_profile_required')
    root=Path(request['accelerator_root'])
    for name,d in profile['sources'].items():
        need(type(name)is str and Path(name).is_absolute() and Path(name).is_relative_to(root) and
             type(d)is str and re.fullmatch('[0-9a-f]{64}',d),'source_profile_member_shape_required')
    for relative in MODULES.values():need(str(root/relative)in profile['sources'],'explicit_normal_dependency_pin_required')
    return profile

def normal_module(name,request,profile):
    path=Path(request['accelerator_root'])/MODULES[name]
    source_pin(path,profile['sources'][str(path)])
    module=importlib.import_module(name)
    if name=='ipfs_accelerate_py':
        # The genuine pinned lazy root deliberately installs this exact subclass.
        # This exception applies to the root only, never arbitrary module proxies.
        need(isinstance(module,types.ModuleType),'genuine_normal_root_module_required')
        namespace=types.ModuleType.__getattribute__(module,'__dict__');cls=type(module)
        need(cls is namespace.get('_IPFSAccelerateModule') and cls.__module__==name and
             cls.__qualname__=='_IPFSAccelerateModule' and cls.__bases__==(types.ModuleType,),
             'genuine_pinned_normal_root_class_required')
        for method in ('__getattribute__','__setattr__','__delattr__','__dir__'):
            function=vars(cls).get(method)
            need(type(function)is types.FunctionType and function.__globals__ is namespace and
                 function.__module__==name and function.__qualname__=='_IPFSAccelerateModule.'+method and
                 function.__code__.co_filename==str(path),
                 'genuine_pinned_normal_root_method_required')
    else:
        need(type(module)is types.ModuleType,'genuine_exact_normal_submodule_required')
    need(sys.modules.get(name)is module and
         module.__file__==module.__spec__.origin==str(path) and path.resolve(strict=True)==path,
         'genuine_normal_module_object_spec_origin_required')
    source_pin(path,profile['sources'][str(path)])
    return module

def _private_pin(path,maximum):
    path=Path(path)
    need(path.resolve(strict=True)==path,'private_snapshot_path_alias')
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        before=os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_uid==os.geteuid() and
             stat.S_IMODE(before.st_mode)==0o600 and 0<before.st_size<=maximum,'bounded_private_snapshot_required')
        parts=[];count=0
        while count<=maximum:
            block=os.read(descriptor,min(65536,maximum+1-count))
            if not block:break
            count+=len(block);parts.append(block)
        raw=b''.join(parts)
        need(count==before.st_size and worker._identity(before)==worker._identity(os.fstat(descriptor))==worker._identity(path.lstat()),
             'private_snapshot_changed')
        pin=dict(path=str(path),bytes=count,sha256=hashlib.sha256(raw).hexdigest())
    finally:os.close(descriptor)
    worker.read_pin(pin,maximum,private=True)
    return pin


def _directory(path):
    path=Path(path);info=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISDIR(info.st_mode) and info.st_uid==os.geteuid() and
         stat.S_IMODE(info.st_mode)==0o700,'existing_private_unaliased_directory_required')
    return path

def fresh_record(directory,name,raw):
    directory=_directory(directory)
    need(type(name)is str and re.fullmatch('[A-Za-z0-9_.-]{1,96}',name) and
         type(raw)is bytes and 0<len(raw)<=worker.MAX_RECEIPT,'bounded_new_metadata_record_required')
    path=directory/name
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        view=memoryview(raw)
        while view:
            count=os.write(descriptor,view);need(count>0,'metadata_short_write');view=view[count:]
        os.fsync(descriptor)
    finally:os.close(descriptor)
    parent=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:os.fsync(parent)
    finally:os.close(parent)
    pin=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    need(worker.read_pin(pin,private=True)==raw,'metadata_durable_readback_differs')
    return pin

def prepare_parent_context(*,request,capsule):
    worker.validate_request(request);validate_capsule_shape(capsule,request)
    need(digest(capsule)==request['capsule_pin']['sha256'],'exact_generic_capsule_commitment_required')
    profile=load_source_profile(request)
    for role,pin in capsule['files'].items():
        worker.read_pin(pin);need(profile['sources'].get(pin['path'])==pin['sha256'],'capsule_and_source_profile_must_join')
    normal_module('ipfs_accelerate_py',request,profile)
    core=normal_module('ipfs_accelerate_py.cli_runtime.grok_production_owner',request,profile)
    context_pin=_private_pin(request['context_path'],core.MAX_FRAME)
    context=core.validate_context(core.decode(worker.read_pin(context_pin,core.MAX_FRAME,private=True)))
    expected=dict(database_task_cid=request['task_cid'],database_attempt_id=request['database_attempt_id'],
        database_attempt_number=request['database_attempt_number'],portal_task_id=request['task_alias'],
        allocation_session_id=request['allocation_session_id'],source_generation=request['source_generation'],
        request_sha256=request['owner_request_sha256'],workspace=request['workspace'],
        work_deadline_monotonic=request['work_deadline_monotonic'],native_deadline_monotonic=request['native_deadline_monotonic'],
        owner_pid=request['owner_pid'],owner_birth=request['owner_birth'],
        source_bindings_sha256=request['source_profile_pin']['sha256'],capsule_sha256=request['capsule_pin']['sha256'],
        worker_source_sha256=capsule['files']['worker']['sha256'],owner_source_sha256=capsule['files']['owner_source']['sha256'])
    need(all(type(v)is type(context[k]) and context[k]==v for k,v in expected.items()),'actual_parent_context_and_fresh_task_join_required')
    own=core.process(os.getpid());parent=core.process(os.getppid())
    need(parent['pid']==request['owner_pid'] and parent['birth']==request['owner_birth'] and
         own['pid']==context['worker_pid'] and own['birth']==context['worker_birth'] and
         own['group_pid']==context['worker_process_group']==os.getpid() and own['parent_pid']==parent['pid'],
         'actual_direct_worker_parent_birth_and_group_required')
    client=core.ProductionOwnerGuardClient(context,expected_owner_pid=parent['pid'],expected_owner_birth=parent['birth'],
        expected_request_sha256=request['owner_request_sha256'],expected_source_generation=request['source_generation'],
        expected_worker_source_sha256=capsule['files']['worker']['sha256'],expected_allocation_session_id=request['allocation_session_id'])
    client.assert_current('request_prepared')
    need(os.environ.get('IPFS_ACCELERATE_AGENT_TASK_ID')==request['task_alias'] and
         os.environ.get('IPFS_ACCELERATE_AGENT_TASK_CID')==request['task_cid'] and
         os.environ.get('IPFS_ACCELERATE_AGENT_TASK_ATTEMPT')==str(request['database_attempt_number']),
         'native_task_environment_join_required')
    _directory(request['receipt_directory'])
    return dict(core=core,client=client,context=context,context_pin=context_pin,profile=profile,worker_identity=own)

def bounded_timeout(remaining,maximum):
    need(type(remaining)in (int,float) and math.isfinite(remaining) and
         type(maximum)is int and 1<=maximum<=90,'bounded_original_timeout_input_required')
    value=min(maximum,math.floor(remaining));need(value>=1,'original_work_budget_exhausted')
    return value

def _worker_argv():
    raw=Path('/proc/self/cmdline').read_bytes()
    need(0<len(raw)<=65536 and raw.endswith(b'\0'),'actual_worker_argv_bound_required')
    return [part.decode('utf-8')for part in raw[:-1].split(b'\0')]

def _store_row(store,request,context,*,empty):
    need(str(store.path)==context['allocation_store_path'],'actual_bound_allocation_store_required')
    # A genuine first public query performs AllocationStore's lazy initialization.
    # Empty alone is insufficient: a missing backend also returns an empty dict.
    row=store.get_cli_session(request['allocation_session_id'],'grok_cli')
    need(store._ready is True and type(store._duckdb)is types.ModuleType and
         store._duckdb is sys.modules.get('duckdb'),'actual_bound_ready_allocation_store_required')
    if empty:need(type(row)is dict and row=={},'fresh_allocation_capture_must_be_empty')
    else:
        need(type(row)is dict and all(type(row.get(k))is str and row[k]for k in CAPTURE_FIELDS) and
             row['allocation_session_id']==request['allocation_session_id'] and row['provider']=='grok_cli' and
             row['workspace']==request['workspace'] and row['resume_style']=='resume','actual_allocation_capture_readback_required')
    return dict(allocation_store_path=str(store.path),row={}if empty else {k:row[k]for k in CAPTURE_FIELDS},observed_monotonic=time.monotonic())

def run_capture(*,request,capsule,prepared,request_bytes,sealed_request_sha256,original_stdin):
    global _DISPATCH_ATTEMPTED
    need(not _DISPATCH_ATTEMPTED,'one_fresh_capture_dispatch_only');_DISPATCH_ATTEMPTED=True
    need(canonical(request)==request_bytes and hashlib.sha256(request_bytes).hexdigest()==sealed_request_sha256,
         'actual_sealed_request_bytes_required')
    core,client,context,profile=(prepared[k]for k in ('core','client','context','profile'))
    need(prepared['worker_identity']==core.process(os.getpid()),'original_worker_identity_changed')
    need(Path.cwd().resolve(strict=True)==Path(request['workspace']),'actual_worker_workspace_required')
    need(type(original_stdin)is bytes and 0<len(original_stdin)<=worker.MAX_STDIN,'bounded_original_stdin_required')
    text=original_stdin.decode('utf-8');named=capsule['named_session']
    helper=normal_module('ipfs_accelerate_py.cli_runtime.grok_named_session',request,profile)
    runtime=normal_module('ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner',request,profile)
    normal_module('ipfs_accelerate_py.llm_router',request,profile)
    allocation=normal_module('ipfs_accelerate_py.llm_allocation',request,profile)
    allocation_module=normal_module('ipfs_accelerate_py.llm_allocation.duckdb_store',request,profile)
    need(os.environ.get('IPFS_ACCELERATE_LLM_ALLOCATION_DB')==context['allocation_store_path'],'actual_allocation_environment_required')
    store=allocation.get_allocation_store()
    need(type(store)is allocation_module.AllocationStore,'actual_normal_allocation_store_required')
    before=_store_row(store,request,context,empty=True)
    def source(pin):return helper.SourcePin(Path(pin['path']),pin['sha256'],maximum_bytes=pin['bytes'])
    owner_request=helper.NamedSessionRequest(workspace=Path(request['workspace']),native_home=Path(named['native_home']),
        audit_directory=Path(named['audit_directory']),binary=source(named['binary']),configuration=source(named['configuration']),
        owner_sampling_policy=source(named['owner_sampling_policy']),
        preparation=helper.NativeHomePreparation(Path(named['preparation_home']),source(named['configuration'])),
        helper_source=source(capsule['files']['helper']),owner_source=source(capsule['files']['owner_source']),
        owner_guard=core.owner_guard_for(client),source_generation=request['source_generation'],
        deadline_monotonic=request['work_deadline_monotonic'],allocation_session_id=request['allocation_session_id'],
        credential_slot=context['credential_slot'],credential_fingerprint=context['credential_fingerprint'],
        system_requirements=source(named['system_requirements'])if named['system_requirements']is not None else None,
        cleanup_reserve_seconds=named['cleanup_reserve_seconds'],bootstrap_max_seconds=named['bootstrap_max_seconds'])
    owner_request.assert_current('worker_v4_dispatch')
    timeout=bounded_timeout(owner_request.remaining_seconds(request['maximum_timeout_seconds']),request['maximum_timeout_seconds'])
    observed=[]
    dispatched=time.monotonic()
    output,router=runtime.run(prompt=text,provider='grok_cli',model='grok-4.7',purpose='planning',timeout=timeout,
        max_output_tokens=request['max_output_tokens'],reasoning_effort='high',grok_named_session_request=owner_request,
        receipt_observer=lambda row:observed.append(canonical(row)))
    response_time=time.monotonic()
    need(type(router)is dict and len(observed)==1 and observed[0]==canonical(router) and
         router.get('schema')=='router-implementation-invocation@1' and router.get('provider')=='grok_cli' and
         router.get('model')=='grok-4.7' and router.get('purpose')=='planning' and router.get('router_calls')==1 and
         router.get('status')=='provider_returned' and router.get('workspace')==request['workspace'] and
         router.get('native_prompt_sha256')==hashlib.sha256(original_stdin).hexdigest(),
         'actual_returned_router_receipt_required')
    need(type(router.get('invocation_id'))is str and re.fullmatch('[0-9a-f]{32}',router['invocation_id']) and
         router['invocation_id']!=request['allocation_session_id'],'distinct_actual_runtime_invocation_required')
    session=owner_request.completed_session_id;named_receipt=owner_request.receipt();policy=router.get('provider_invocation_policy',{})
    need(type(session)is str and str(uuid.UUID(session))==session and policy.get('native_resume_session_id')==session and
         policy.get('native_resume_identity_verified')is True and policy.get('owned_named_session')==named_receipt and
         named_receipt.get('response_accepted')is True and named_receipt.get('lifetime_closed')is True and
         named_receipt.get('native_process_reaped')is True and type(named_receipt.get('native_process_returncode'))is int and
         named_receipt['native_process_returncode']==0 and named_receipt.get('bootstrap_successful')is True and
         named_receipt.get('request_methods')==METHODS and named_receipt.get('forced_cleanup')is False and
         named_receipt.get('cleanup_error_types')==[],'actual_accepted_closed_native_session_required')
    after=_store_row(store,request,context,empty=False)
    need(after['row']['native_session_id']==session,'accepted_native_UUID_capture_join_required')
    owner_request.assert_current('worker_v4_response')
    need(time.monotonic()<request['work_deadline_monotonic'] and type(output)is str and
         0<len(output.encode('utf-8'))<=65536,'bounded_actual_provider_response_required')
    # Recheck every frozen producer before publishing any success metadata.
    for name,d in profile['sources'].items():source_pin(Path(name),d)
    worker.read_pin(request['capsule_pin'],worker.MAX_REQUEST,private=True)
    worker.read_pin(request['source_profile_pin'],worker.MAX_REQUEST,private=True)
    worker.read_pin(prepared['context_pin'],core.MAX_FRAME,private=True)
    directory=Path(request['receipt_directory'])
    request_pin=fresh_record(directory,'capture-request.json',request_bytes)
    router_pin=fresh_record(directory,'router-receipt.json',canonical(router))
    named_pin=fresh_record(directory,'named-session-receipt.json',canonical(named_receipt))
    bootstrap_pin=_private_pin(Path(named['audit_directory'])/'bootstrap-stdout.jsonl',worker.MAX_RECEIPT)
    record=dict(schema=worker.RECEIPT_SCHEMA,request_pin=request_pin,sealed_request_sha256=sealed_request_sha256,
        owner_request_sha256=request['owner_request_sha256'],context_pin=prepared['context_pin'],capsule_pin=request['capsule_pin'],
        source_profile_pin=request['source_profile_pin'],source_generation=request['source_generation'],task_alias=request['task_alias'],
        task_cid=request['task_cid'],database_attempt_id=request['database_attempt_id'],database_attempt_number=request['database_attempt_number'],
        allocation_session_id=request['allocation_session_id'],worker_identity=dict(prepared['worker_identity']),worker_argv=_worker_argv(),
        stdin=dict(bytes=len(original_stdin),sha256=hashlib.sha256(original_stdin).hexdigest()),
        original_work_deadline_monotonic=request['work_deadline_monotonic'],original_native_deadline_monotonic=request['native_deadline_monotonic'],
        dispatch_monotonic=dispatched,response_monotonic=response_time,observed_monotonic=time.monotonic(),timeout_seconds=timeout,
        router_receipt_pin=router_pin,named_receipt_pin=named_pin,bootstrap_stdout_pin=bootstrap_pin,
        store_capture_before=before,store_capture_observation=after,owner_rpc_receipts=client.receipts(),
        provider_response=dict(bytes=len(output.encode('utf-8')),sha256=hashlib.sha256(output.encode('utf-8')).hexdigest()),
        execution_environment=request['execution_environment'],metadata_only=True,capture_accepted_by_parent=False,
        task_completion_accepted=False,production_qualified=False)
    need(record['observed_monotonic']<request['work_deadline_monotonic'],'capture_metadata_after_original_work_deadline')
    return fresh_record(directory,'capture-transport.json',canonical(record))
