"""Explicit fresh-task Grok capture worker; never task-completion authority.

Only main can dispatch, after a fully sealed request, source-pinned generic
V4 capsule, actual parent context and live kernel-peer ACK. Import is inert.
"""
from __future__ import annotations
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
import types

SCHEMA='fresh-task-grok-capture-request/v1'
RECEIPT_SCHEMA='fresh-task-grok-capture-transport/v1'
MAX_REQUEST=65536
MAX_RECEIPT=2097152
MAX_STDIN=128000
SEALS=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL
WORKER_RELATIVE=Path('ipfs_accelerate_py/cli_runtime/grok_fresh_task_capture_worker.py')
TRANSPORT_NAME='ipfs_accelerate_py.cli_runtime.grok_fresh_task_capture_transport'
TRANSPORT_RELATIVE=Path('ipfs_accelerate_py/cli_runtime/grok_fresh_task_capture_transport.py')

class CaptureRefusal(RuntimeError):
    """Closed codes only; never interpolate credentials, prompt or model text."""

def need(value,reason):
    if not value:raise CaptureRefusal(reason)

def canonical(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')

def decode(raw):
    def pairs(rows):
        result={}
        for key,value in rows:
            need(key not in result,'duplicate_metadata_key');result[key]=value
        return result
    def nonfinite(_):raise CaptureRefusal('nonfinite_metadata')
    value=json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_constant=nonfinite)
    need(type(value)is dict and canonical(value)==raw,'canonical_metadata_required')
    return value

def pin_shape(pin,maximum=MAX_RECEIPT):
    need(type(pin)is dict and set(pin)=={'path','bytes','sha256'} and
         type(pin['path'])is str and Path(pin['path']).is_absolute() and
         '..'not in Path(pin['path']).parts and not any(ord(c)<32 for c in pin['path']) and
         type(pin['bytes'])is int and 0<pin['bytes']<=maximum and
         type(pin['sha256'])is str and re.fullmatch('[0-9a-f]{64}',pin['sha256']), 'bounded_exact_pin_required')

def _identity(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_nlink,info.st_size,info.st_mtime_ns,info.st_ctime_ns)

def read_pin(pin,maximum=MAX_RECEIPT,private=False):
    pin_shape(pin,maximum);path=Path(pin['path'])
    need(path.resolve(strict=True)==path,'input_path_alias')
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK)
    try:
        before=os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and
             before.st_uid in {0,os.geteuid()} and not before.st_mode&0o002 and before.st_size==pin['bytes'], 'input_custody_required')
        if private:need(before.st_uid==os.geteuid() and stat.S_IMODE(before.st_mode)==0o600,'private_input_mode_required')
        parts=[];count=0
        while count<=maximum:
            block=os.read(descriptor,min(65536,maximum+1-count))
            if not block:break
            parts.append(block);count+=len(block)
        raw=b''.join(parts)
        need(count==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'] and
             _identity(before)==_identity(os.fstat(descriptor))==_identity(path.lstat()) and
             path.resolve(strict=True)==path,'input_changed_or_pin_mismatch')
        return raw
    finally:os.close(descriptor)

def validate_request(value):
    fields={'schema','mode','one_shot','attempt_cap','provider_dispatch_allowed','execution_environment',
        'accelerator_root','workspace','receipt_directory','task_alias','task_cid','database_attempt_id',
        'database_attempt_number','allocation_session_id','source_generation','work_deadline_monotonic',
        'native_deadline_monotonic','owner_pid','owner_birth','owner_request_sha256','context_path',
        'capsule_pin','source_profile_pin','max_output_tokens','maximum_timeout_seconds'}
    need(type(value)is dict and set(value)==fields and value['schema']==SCHEMA,'closed_fresh_request_required')
    need(value['mode']=='fresh-task-capture-diagnostic' and value['one_shot']is True and
         type(value['attempt_cap'])is int and value['attempt_cap']==1 and
         value['provider_dispatch_allowed']is True,'explicit_one_shot_capture_request_required')
    need(value['execution_environment']in ('isolated-loopback','provider'),'explicit_execution_environment_required')
    for name in ('task_alias','task_cid','database_attempt_id','allocation_session_id','source_generation'):
        item=value[name];need(type(item)is str and 0<len(item)<=512 and not any(ord(c)<32 for c in item),'bounded_fresh_identity_required')
    need(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}',value['allocation_session_id']) and
         value['source_generation'].startswith('grok-automatic-named-session/') and len(value['source_generation'])<=128,
         'allocation_and_source_generation_shape_required')
    for name in ('database_attempt_number','owner_pid','owner_birth'):
        need(type(value[name])is int and value[name]>0,'positive_exact_integer_required')
    need(type(value['owner_request_sha256'])is str and re.fullmatch('[0-9a-f]{64}',value['owner_request_sha256']), 'owner_request_digest_required')
    for name in ('accelerator_root','workspace','receipt_directory','context_path'):
        item=value[name];need(type(item)is str and Path(item).is_absolute() and '..'not in Path(item).parts and
            not any(ord(c)<32 for c in item),'exact_absolute_path_required')
    need(Path(value['context_path']).parent==Path(value['receipt_directory']),'owner_context_in_receipt_directory_required')
    need(not Path(value['receipt_directory']).is_relative_to(Path(value['workspace'])) and
         not Path(value['workspace']).is_relative_to(Path(value['receipt_directory'])),'workspace_and_receipt_directory_must_not_overlap')
    for name in ('capsule_pin','source_profile_pin'):pin_shape(value[name],MAX_REQUEST)
    need(value['capsule_pin']['path']!=value['source_profile_pin']['path'],'distinct_input_pins_required')
    for name in ('work_deadline_monotonic','native_deadline_monotonic'):
        need(type(value[name])in (int,float) and math.isfinite(value[name]),'finite_original_deadline_required')
    need(0<value['work_deadline_monotonic']<=value['native_deadline_monotonic'] and
         value['native_deadline_monotonic']-value['work_deadline_monotonic']<=60,'original_native_cleanup_bound_required')
    for name,upper in (('max_output_tokens',4096),('maximum_timeout_seconds',90)):
        need(type(value[name])is int and 1<=value[name]<=upper,'bounded_exact_dispatch_limit_required')
    return value

def _read_sealed_bytes(descriptor,expected_sha256):
    need(type(descriptor)is int and descriptor>=3 and type(expected_sha256)is str and
         re.fullmatch('[0-9a-f]{64}',expected_sha256),'explicit_sealed_request_required')
    before=os.fstat(descriptor)
    need(stat.S_ISREG(before.st_mode) and before.st_nlink==0 and before.st_uid==os.geteuid() and
         0<before.st_size<=MAX_REQUEST and fcntl.fcntl(descriptor,fcntl.F_GET_SEALS)&SEALS==SEALS,
         'fully_sealed_private_memfd_required')
    raw=os.pread(descriptor,before.st_size+1,0)
    need(_identity(before)==_identity(os.fstat(descriptor)) and len(raw)==before.st_size and
         hashlib.sha256(raw).hexdigest()==expected_sha256,'sealed_request_changed')
    return raw

def read_sealed_request(descriptor,expected_sha256):
    return validate_request(decode(_read_sealed_bytes(descriptor,expected_sha256)))

def read_original_stdin(stream,*,deadline):
    need(type(deadline)in (int,float) and math.isfinite(deadline),'absolute_stdin_deadline_required')
    descriptor=stream.fileno();parts=[];count=0
    while True:
        remaining=deadline-time.monotonic();need(remaining>0,'original_stdin_deadline_expired')
        ready,_,_=select.select([descriptor],[],[],remaining)
        need(bool(ready),'original_stdin_deadline_expired')
        raw=os.read(descriptor,min(65536,MAX_STDIN+1-count))
        need(time.monotonic()<deadline,'original_stdin_deadline_expired')
        if not raw:break
        count+=len(raw);need(count<=MAX_STDIN,'original_stdin_byte_bound');parts.append(raw)
    result=b''.join(parts);need(bool(result.strip()),'nonempty_original_stdin_required');result.decode('utf-8')
    return result

def _load_transport(request):
    capsule=decode(read_pin(request['capsule_pin'],MAX_REQUEST,private=True))
    need(type(capsule.get('files'))is dict and 'adapter'in capsule['files'],'capsule_transport_pin_required')
    source=capsule['files']['adapter'];read_pin(source)
    root=Path(request['accelerator_root']);need(root.resolve(strict=True)==root,'actual_accelerator_root_required')
    need(source['path']==str(root/TRANSPORT_RELATIVE) and
         capsule['files'].get('worker',{}).get('path')==str(root/WORKER_RELATIVE) and
         Path(__file__).resolve(strict=True)==root/WORKER_RELATIVE,'fixed_normal_worker_transport_paths_required')
    read_pin(capsule['files']['worker'])
    # Authenticate the real package initializer BEFORE its import executes.
    profile=decode(read_pin(request['source_profile_pin'],MAX_REQUEST,private=True))
    need(profile.get('schema')=='fresh-task-grok-capture-source-profile/v1' and
         profile.get('source_generation')==request['source_generation'] and type(profile.get('sources'))is dict,
         'explicit_normal_source_profile_required')
    initializer=root/'ipfs_accelerate_py/__init__.py'
    expected=profile['sources'].get(str(initializer))
    need(type(expected)is str and re.fullmatch('[0-9a-f]{64}',expected),'normal_package_initializer_pin_required')
    read_pin(dict(path=str(initializer),bytes=initializer.lstat().st_size,sha256=expected))
    sys.path.insert(0,str(root))
    module=importlib.import_module(TRANSPORT_NAME)
    need(type(module)is types.ModuleType and sys.modules.get(TRANSPORT_NAME)is module and
         module.__file__==module.__spec__.origin==source['path'],'normal_transport_origin_required')
    read_pin(source)
    return module,capsule

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-m2-capture-handshake',action='store_true')
    parser.add_argument('--private-m2-request-fd',type=int,required=True)
    parser.add_argument('--private-m2-request-sha256',required=True)
    parser.add_argument('--transport-capsule',type=Path,required=True)
    parser.add_argument('--transport-capsule-sha256',required=True)
    parser.add_argument('--owner-context',type=Path,required=True)
    parser.add_argument('--owner-request-sha256',required=True)
    parser.add_argument('--owner-allocation-session-id',required=True)
    args=parser.parse_args(argv);stage='explicit_fresh_request'
    try:
        need(args.private_m2_capture_handshake,'explicit_capture_selector_required')
        raw=_read_sealed_bytes(args.private_m2_request_fd,args.private_m2_request_sha256)
        request=validate_request(decode(raw));os.set_inheritable(args.private_m2_request_fd,False)
        need(str(args.transport_capsule)==request['capsule_pin']['path'] and
             args.transport_capsule_sha256==request['capsule_pin']['sha256'] and
             str(args.owner_context)==request['context_path'] and args.owner_request_sha256==request['owner_request_sha256'] and
             args.owner_allocation_session_id==request['allocation_session_id'],'sealed_request_CLI_join_required')
        stage='normal_source_and_parent_context'
        transport,capsule=_load_transport(request)
        prepared=transport.prepare_parent_context(request=request,capsule=capsule)
        # No original stdin is read until actual context + live peer ACK.
        stage='original_stdin'
        original=read_original_stdin(sys.stdin.buffer,deadline=request['work_deadline_monotonic']-capsule['named_session']['cleanup_reserve_seconds'])
        stage='one_genuine_named_session_dispatch'
        record=transport.run_capture(request=request,capsule=capsule,prepared=prepared,
            request_bytes=raw,sealed_request_sha256=args.private_m2_request_sha256,original_stdin=original)
        print(json.dumps(dict(schema=RECEIPT_SCHEMA,transport_receipt_pin=record,
            task_completion_accepted=False,production_qualified=False),sort_keys=True),flush=True)
        return 0
    except BaseException as error:
        print(json.dumps(dict(schema='fresh-task-grok-capture-failure/v1',stage=stage,
            error_type=type(error).__name__,task_completion_accepted=False,production_qualified=False),sort_keys=True),file=sys.stderr,flush=True)
        return 1

if __name__=='__main__':raise SystemExit(main())
