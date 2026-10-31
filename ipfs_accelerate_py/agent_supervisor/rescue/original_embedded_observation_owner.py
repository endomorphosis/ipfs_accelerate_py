# Genuine embedded observation successor of accepted original0102 owner v11.
"""PRIVATE execution profile: finite original deterministic metadata CLI resource ownership.

This is finite observation plumbing; original scheduler, disk and source owners
retain all custody. The actor uses the original IntentRepository read_session.
External charges are conservative declared byte caps, with polled size checks;
ancestry/group observation and Python audit denials are cooperative, not a sandbox.
"""

from __future__ import annotations
import argparse
import fcntl
import math
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import signal
import stat
import subprocess
import sys
import threading
import time
import types

R = Path('/home/barberb/lift_coding/external/ipfs_datasets')
A = Path('/home/barberb/lift_coding/external/ipfs_accelerate')
WORKSPACE = R
POLICY = '4662bd8fe6caf6f2c048e3d7cff7abb4d80e5c875b21e02e204b1b89371b8286'
LEASE = R/'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py'
LEASE_SHA = '42e9f07d0d458177079269d9f495f0b70512180c1adc2f7fe0bfdb05518a12f9'
STATE = Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
LEDGER = R/'workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
ROOTS = (R/'workspace/test-logs', R/'workspace/todo-queues', R/'docs/implementation/reports/evidence', Path('/tmp/pytest-of-barberb'))
DATABASE = R/'workspace/test-logs/federal-corpus-audits/legal-ir-supervisor-plan-20261016/catalog-02/control.duckdb'
PUBLICATION_CAPS = {}
EXTERNAL_FILES = [
    {'max_bytes':128*1024**2,'path':str(DATABASE)},
    {'max_bytes':32*1024**2,'path':str(DATABASE)+'.wal'},
    {'max_bytes':64*1024,'path':str(DATABASE.with_name('.'+DATABASE.name+'.intent.lock'))},
    {'max_bytes':64*1024,'path':str(DATABASE.with_name('.'+DATABASE.name+'.lock'))},
]
EXECUTION_TEMPLATE = None  # Idle successor has a sealed per-cycle plan, no0102 mutation template.
EXECUTION_TEMPLATE_SHA = None
READONLY_PROFILE_GIT = [[
    '/usr/bin/git','--no-replace-objects','-c','core.hooksPath=/dev/null',
    '-c','core.fsmonitor=false','-C',str(WORKSPACE),'rev-parse',v
] for v in ('--show-toplevel','HEAD','HEAD^{tree}')]
PROCESS_ADMISSION_FIELDS = frozenset({'kind','argv','cwd','environment_sha256',
    'start_new_session','original_frame','source','sealed_launcher_fds','status_fd','declared_command_argv'})
PROCESS_ADMISSION_KINDS = frozenset({'readonly_profile_git','sealed_runner_probe','native_cli_subreaper','immutable_source_capsule_read'})
FD_ADMISSION_FIELDS = frozenset({'kind','event','fd','mode','source','original_frame','payload_sha256','memfd_name'})
ACTOR_PATH = Path(__file__).absolute().with_name('native_embedded_observation_driver.py')
SCHEMA = 'legal-ir-original-native-control-request/v1'
FIELDS = {'schema','purpose','request_id','argv','cwd','attempt_directory','source_sha256',
          'cpu_slots','memory_mb','child_process_slots','storage_bytes','timeout_seconds',
          'scheduler_state_path','expected_scheduler_policy_sha256','resource_ledger',
          'source_roots','external_files','database_preimage','allowed_subprocess_argv'}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def regular(path, maximum=4*1024**2):
    path = Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path,*path.parents)), 'unaliased absolute path required')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_uid==os.geteuid()
             and 0 <= before.st_size <= maximum, 'bounded owner file required')
        raw=f.read(maximum+1);after=os.fstat(f.fileno())
    witness=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_uid,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    need(len(raw)==before.st_size and witness(before)==witness(after)==witness(path.lstat()),'file changed during read')
    return raw


def decode(raw):
    def pairs(items):
        value={}
        for key,item in items:
            need(key not in value,'duplicate JSON key');value[key]=item
        return value
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def pin(path, maximum=256*1024**2):
    raw=regular(path,maximum)
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def write_json(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    need(len(raw)<=4*1024**2,'bounded result required')
    with path.open('xb') as f:
        os.fchmod(f.fileno(),0o600);f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def validate_import_git_binding(c):
    """Validate the retained original immutable-HEAD metadata contract only."""
    capsule=c['import_head_capsule_Git']
    fields={'blobs','ls_tree_tail','payload_bytes','required_relative_files','route_source',
        'scope','selected_file_count','source_head','source_root','unique_blob_count'}
    need(type(capsule)is dict and set(capsule)==fields,'closed original import HEAD contract required')
    need(capsule['source_root']==str(ACTOR_PATH.parents[3]) and type(capsule['source_head'])is str
         and re.fullmatch(r'[0-9a-f]{40}',capsule['source_head']), 'exact import source root/HEAD required')
    route=capsule['route_source'];path=ACTOR_PATH.parents[3]/'ipfs_accelerate_py/agent_implementation_route.py'
    need(type(route)is dict and set(route)=={'path','bytes','sha256'} and route['path']==str(path)
         and type(route['bytes'])is int and 0<route['bytes']<=4*1024**2
         and type(route['sha256'])is str and re.fullmatch(r'[0-9a-f]{64}',route['sha256'])
         and c['source_sha256'].get(str(path))==route['sha256'], 'original source route pin required')
    expected_tree=['ls-tree','-r','-z',capsule['source_head'],'--','ipfs_accelerate_py',
        'scripts/ops/agent_supervisor/configured_board_scheduler.py',
        'scripts/ops/agent_supervisor/implementation_supervisor_entry.py']
    need(capsule['ls_tree_tail']==expected_tree and capsule['scope']==
         'immutable HEAD bytes archived/hashed by original import capsule; no import or execution of omitted live source bodies',
         'original immutable HEAD read scope required')
    blobs=capsule['blobs'];need(type(blobs)is list and 1<=len(blobs)<=2048,'finite original HEAD blob population required')
    required_sql={'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql', 'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0003_state_server_restart_identity.sql', 'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0002_causal_event_federation_core.sql', 'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0004_hash_observations.sql', 'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0005_tasks_index_repair.sql'}
    paths=set();objects={};total=0
    for row in blobs:
        need(type(row)is dict and set(row)=={'bytes','object_id','relative_path'}
             and type(row['bytes'])is int and 0<=row['bytes']<=4*1024**2
             and type(row['object_id'])is str and re.fullmatch(r'[0-9a-f]{40}',row['object_id'])
             and type(row['relative_path'])is str and row['relative_path'] and '\\'not in row['relative_path']
             and not Path(row['relative_path']).is_absolute() and '..'not in Path(row['relative_path']).parts,
             'closed original HEAD blob descriptor required')
        name=row['relative_path'];need(name not in paths,'duplicate original HEAD relative source')
        need(name.startswith('ipfs_accelerate_py/') and name.endswith('.py') or name in required_sql
             or name=='ipfs_accelerate_py/agent_supervisor/task_sources/sql/0002_tasks_index_repair.sql'
             or name in expected_tree[-2:],
             'unselected original HEAD source family')
        paths.add(name);oid=row['object_id']
        need(oid not in objects or objects[oid]==row['bytes'],'inconsistent original Git object length')
        objects[oid]=row['bytes'];total+=row['bytes']
        need(total<=120*1024**2,'original import payload bound exceeded')
    required=capsule['required_relative_files']
    need(type(required)is list and required and all(type(x)is str and x in paths for x in required)
         and len(required)==len(set(required)) and required_sql<=paths and required_sql<=set(required),
         'original required source/SQL population differs')
    need(type(capsule['selected_file_count'])is int and capsule['selected_file_count']==len(blobs)
         and type(capsule['unique_blob_count'])is int and capsule['unique_blob_count']==len(objects)
         and type(capsule['payload_bytes'])is int and capsule['payload_bytes']==total,
         'original HEAD count/byte metadata differs')
    return capsule


def request(path,digest):
    raw=regular(path)
    need(hashlib.sha256(raw).hexdigest()==digest,'request digest changed')
    value=decode(raw)
    need(type(value) is dict and set(value)==FIELDS and value['schema']==SCHEMA
         and value['purpose']=='native_control_metadata','closed control request required')
    need(type(value['request_id']) is str and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}',value['request_id']), 'request identity required')
    need(all(type(value[k]) is int for k in ('cpu_slots','memory_mb','child_process_slots','storage_bytes','timeout_seconds')),
         'exact integer resource bounds required')
    need({k:value[k] for k in ('cpu_slots','memory_mb','child_process_slots','storage_bytes','timeout_seconds')}==
         {'cpu_slots':1,'memory_mb':1536,'child_process_slots':8,'storage_bytes':384*1024**2,'timeout_seconds':300},'reviewed finite control envelope required')
    need(value['scheduler_state_path']==str(STATE) and value['expected_scheduler_policy_sha256']==POLICY
         and value['resource_ledger']==str(LEDGER),'original scheduler/ledger identity required')
    need(value['source_roots']==[str(R),str(A)],'original R and A source custody required for observation lifetime')
    argv=value['argv']
    need(type(argv) is list and len(argv)>=3 and all(type(s) is str and '\0' not in s for s in argv)
         and argv[0]==sys.executable and argv[1]=='-B' and Path(argv[2]).is_absolute()
         and argv[2].endswith('.py'),'one source-bound Python control actor required')
    cwd=Path(value['cwd']);attempt=Path(value['attempt_directory'])
    need(cwd.is_absolute() and cwd.is_dir() and not any(p.is_symlink() for p in (cwd,*cwd.parents)), 'actual absolute cwd required')
    need(attempt.is_absolute() and any(root in attempt.parents for root in ROOTS)
         and not any(p.is_symlink() for p in (attempt,*attempt.parents)), 'attempt must be in original named roots')
    pins=value['source_sha256']
    need(type(pins) is dict and 1 <= len(pins) <= 1024 and str(Path(__file__).resolve()) in pins
         and argv[2] in pins,'launcher and actor source pins required')
    for name,d in pins.items():
        need(type(name) is str and Path(name).is_absolute() and type(d) is str and re.fullmatch(r'[0-9a-f]{64}',d),'invalid source pin')
    allowed=value['allowed_subprocess_argv']
    need(type(allowed) is list and allowed==READONLY_PROFILE_GIT,'only the three original readonly Git primitives are declared statically; checked native driver owns exact sealed runner families')
    publication=set(PUBLICATION_CAPS)
    expected={row['path'] for row in EXTERNAL_FILES}
    external=value['external_files']
    need(external==EXTERNAL_FILES,'exact reviewed four-file embedded observation precharge required')
    need(type(external) is list and len(external)==len(expected)
         and all(type(x) is dict and set(x)=={'path','max_bytes'} for x in external)
         and {x['path'] for x in external}==expected,'exact original catalog and mode-specific output declaration required')
    for row in external:
        need(any(root in Path(row['path']).parents for root in ROOTS),'external output outside original ledger roots')
        if row['path'] in {str(DATABASE.with_name('.'+DATABASE.name+'.intent.lock')),
                           str(DATABASE.with_name('.'+DATABASE.name+'.lock'))}:
            need(type(row['max_bytes']) is int and row['max_bytes']==64*1024,'exact native OS-lock cap required')
        if row['path'] in publication:
            need(type(row['max_bytes']) is int and 0<row['max_bytes']<=PUBLICATION_CAPS[row['path']], 'publication byte cap differs')
            # The claim journal is the durable record of an in-progress attempt.
            # Producer and verifier outputs must still be absent.
            if '.claim-intent' not in Path(row['path']).name:
                need(not os.path.lexists(row['path']),'versioned output exists; reconcile before control mutation')
    need(all(type(x['max_bytes']) is int and 0 < x['max_bytes'] <= value['storage_bytes'] for x in external)
         and sum(x['max_bytes'] for x in external) <= value['storage_bytes']-8*1024**2,'external precharges exceed envelope')
    need(not expected.intersection(pins),'mutable catalog/sidecars are not immutable source pins')
    prior=value['database_preimage']
    need(type(prior) is dict and set(prior)=={'path','bytes','sha256'} and prior['path']==str(DATABASE)
         and type(prior['bytes']) is int and 0 < prior['bytes'] <= value['storage_bytes']
         and type(prior['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}',prior['sha256']), 'exact pre-mutation database descriptor required')
    allowed=value['allowed_subprocess_argv']
    need(type(allowed) is list and allowed==READONLY_PROFILE_GIT,'only the three original readonly Git primitives are declared statically; checked native driver owns exact sealed runner families')
    need(len(argv)==7 and argv[2]==str(ACTOR_PATH) and argv[3]=='--plan' and argv[5]=='--plan-sha256'
         and argv[4] in pins and pins[argv[4]]==argv[6], 'only the exact pinned embedded observation actor/plan is permitted')
    need(cwd==WORKSPACE,'actual execution cwd must be the exact selected workspace')

    # The successor admits one sealed, finite read-only observation, not the old
    # one-task0102 mutation. The original kernel below remains its sole owner.
    plan=decode(regular(argv[4]))
    need(type(plan) is dict and set(plan)=={'schema','operation','accelerate_root','task_scope','native_metadata_cli'}
         and plan['schema']=='native-embedded-scoped-observation-plan/v1'
         and plan['operation']=='native_embedded_scoped_observation_v1'
         and plan['accelerate_root']==str(ACTOR_PATH.parents[3]),'closed selected normal-package observation plan required')
    scope=plan['task_scope']
    need(type(scope) is list and 1<=len(scope)<=256 and all(type(x) is str
         and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:-]{0,255}',x) for x in scope)
         and len(set(scope))==len(scope),'finite unique task CID scope required')
    c=plan['native_metadata_cli']
    need(type(c) is dict and set(c)=={'schema','runtime_root','cwd','attempt_directory','temporary_directory',
         'project_import_plan','pre_body_guard','result_path','source_sha256','observation_binding','import_head_capsule_Git'}
         and c['schema']=='native-embedded-scoped-observation-cli/v2'
         and c['runtime_root']==plan['accelerate_root'] and c['cwd']==str(WORKSPACE)
         and c['attempt_directory']==str(attempt) and c['temporary_directory']==str(attempt/'temporary')
         and c['result_path']==str(attempt/'exports/native-cli-driver-result.json'),
         'exact original observation paths/source-root binding required')
    need(type(c['source_sha256']) is dict and 1<=len(c['source_sha256'])<=1024
         and all(name in pins and pins[name]==digest for name,digest in c['source_sha256'].items()),
         'driver immutable source graph differs from original owner graph')
    validate_import_git_binding(c)
    binding=c['observation_binding']
    need(type(binding) is dict and set(binding)=={'schema','database_preimage','task_scope',
         'plan_root_cid','repository_tree_id','observer_source','source_schema'}
         and binding['schema']=='native-embedded-daemon-observation-binding@1'
         and binding['database_preimage']==prior and binding['task_scope']==scope,
         'exact original database/scope observation binding required')
    observer=binding['observer_source']
    helper=ACTOR_PATH.parents[3]/'ipfs_accelerate_py/agent_supervisor/todo_daemon/native_owner_bootstrap.py'
    need(type(observer) is dict and set(observer)=={'path','bytes','sha256'}
         and observer['path']==str(helper) and type(observer['bytes']) is int
         and 0<observer['bytes']<=4*1024**2 and c['source_sha256'].get(str(helper))==observer['sha256']
         and pin(helper)==observer,'exact canonical observer helper source required')
    need(all(type(binding[k]) is str and 0<len(binding[k])<=4096
         for k in ('plan_root_cid','repository_tree_id','source_schema')),'sealed catalogue fields required')
    import_plan=c['project_import_plan']
    need(type(import_plan) is dict and set(import_plan)=={'accelerate_root','datasets_root','publication_root','source_pins'}
         and import_plan['accelerate_root']==plan['accelerate_root']
         and import_plan['datasets_root']==str(R) and import_plan['publication_root']==str(R),
         'original before-body import source roots differ')
    need(type(import_plan['source_pins']) is list and 1<=len(import_plan['source_pins'])<=1024
         and all(type(ref) is dict and set(ref)=={'path','bytes','sha256'}
             and type(ref['path']) is str and ref['path'] in c['source_sha256']
             and c['source_sha256'][ref['path']]==ref['sha256']
             and type(ref['bytes']) is int and 0<=ref['bytes']<=4*1024**2
             for ref in import_plan['source_pins']), 'finite pre-body import graph must join original owner source pins')
    guard_ref=c['pre_body_guard']
    need(type(guard_ref) is dict and set(guard_ref)=={'path','bytes','sha256'}
         and guard_ref['path']==str(ACTOR_PATH.with_name('project_import_source_guard.py'))
         and guard_ref['sha256']=='1926462c717afdb265e28c90c7723e8ae04eabb27bc6e54ca73e8169cd27c45f'
         and c['source_sha256'].get(guard_ref['path'])==guard_ref['sha256']
         and pin(guard_ref['path'])==guard_ref,'exact original before-body guard required')
    need(attempt.is_dir() and attempt.lstat().st_uid==os.geteuid()
         and attempt.lstat().st_mode&0o777==0o700, 'caller-prepared private attempt required')
    return value,raw


def source_guard(value,path,raw):
    need(regular(path)==raw,'request bytes changed')
    for name,digest in value['source_sha256'].items():
        need(hashlib.sha256(regular(name)).hexdigest()==digest,'source changed: '+name)


def external_usage(value):
    rows=[]
    for row in value['external_files']:
        path=Path(row['path'])
        need(not any(p.is_symlink() for p in (path,*path.parents)),'external path became aliased')
        try:info=path.lstat()
        except FileNotFoundError:
            need(path!=DATABASE,'catalog disappeared');size=0
        else:
            need(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_uid==os.geteuid(), 'external file custody changed')
            size=info.st_size
        need(size <= row['max_bytes'],'external byte cap exceeded')
        rows.append({**row,'observed_bytes':size})
    return rows


def load_source_lease():
    raw=regular(LEASE)
    need(hashlib.sha256(raw).hexdigest()==LEASE_SHA,'original source lease changed')
    module=types.ModuleType('_original_control_source_lease_20261028')
    module.__file__=str(LEASE);sys.modules[module.__name__]=module
    exec(compile(raw,str(LEASE),'exec'),module.__dict__)
    return module



PARENT_CAPACITY_FD_ENV = "IPFS_ACCELERATE_NATIVE_METADATA_PARENT_FD"
PARENT_CAPACITY_SCHEMA = "native-metadata-sealed-parent-capacity/v1"
PARENT_CAPACITY_AUTHORITY = {"resource_capacity_only": True, "execution_authority": False,
    "provider_authority": False, "proof_authority": False, "completion_authority": False}


def _new_parent_capacity_fd():
    descriptor = os.memfd_create("native-metadata-parent-capacity-v1",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.fchmod(descriptor, 0o400)
    return descriptor


def _seal_parent_capacity_fd(descriptor, *, owner, value, request_path,
                             execution_request, work_deadline, native_deadline, child_pid, child_birth):
    # The original stdin barrier keeps the direct child outside its worker
    # until original bind/check_usage and this exact seal have both completed.
    from dataclasses import asdict
    native = owner.native_lease
    need(native.owner_pid == os.getpid() and not native.released
         and not native.cancelled, 'live original parent native lease required')
    token = asdict(native.token)
    need(set(token) == {'lease_id', 'lease_key', 'state_path'}
         and all(type(part) is str and part for part in token.values())
         and token['state_path'] == str(STATE), 'original parent token required')
    snapshot = owner.to_dict(); record = snapshot['record']
    need(snapshot['status'] == 'active' and record['status'] == 'active'
         and record['owner_pid'] == os.getpid()
         and record['native_lease_id'] == token['lease_id']
         and record['child'] == {'pid': child_pid, 'birth': child_birth},
         'original active disk/native direct-child binding differs')
    need(type(work_deadline) in (int, float) and math.isfinite(work_deadline)
         and work_deadline > time.monotonic()
         and type(native_deadline) in (int, float) and math.isfinite(native_deadline)
         and work_deadline <= native_deadline, 'original work/native deadline invalid')
    owner_state = _parent_capacity_process_identity(os.getpid())
    payload = {'schema': PARENT_CAPACITY_SCHEMA, 'authority': PARENT_CAPACITY_AUTHORITY,
        'owner_identity': owner_state, 'child_identity': {'pid': child_pid, 'birth': child_birth},
        'owner_uid': os.geteuid(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'native_lease_token': token, 'scheduler_policy_sha256': POLICY,
        'disk_ledger_path': str(LEDGER), 'disk_reservation_id': snapshot['reservation_id'],
        'expected_phase_lease_id': record['parent_lease_id'], 'storage_roots': [str(root) for root in ROOTS],
        'request_pin': pin(request_path), 'expected_request_sha256': pin(request_path)['sha256'],
        'execution_request_pin': execution_request, 'owner_source': pin(Path(__file__).resolve()),
        'attempt_identity': record['attempt_directory'], 'sealed_work_deadline_monotonic': work_deadline,
        'sealed_native_deadline_monotonic': native_deadline,
        'allocated_envelope': {key: value[key] for key in
            ('cpu_slots', 'memory_mb', 'child_process_slots', 'storage_bytes')}}
    data = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    need(len(data) <= 8192, 'private native resource capability exceeds bound')
    need(os.lseek(descriptor, 0, os.SEEK_END) == 0, 'parent capability descriptor reused')
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view); need(written > 0, 'parent capability write failed')
        view = view[written:]
    os.lseek(descriptor, 0, os.SEEK_SET)
    seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
    need(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
         'parent capability seals incomplete')
    # No token bytes, key, descriptor number, or payload enter any receipt.
    return {'schema': 'native-metadata-parent-capacity-handoff-public/v1',
        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
        'sealed': True, 'original_child_bound': True, 'authority': PARENT_CAPACITY_AUTHORITY}


def _parent_capacity_process_identity(pid):
    raw = Path('/proc/' + str(pid) + '/stat').read_text()
    return {'pid': pid, 'birth': raw.rsplit(')', 1)[1].split()[19]}


def _check_inherited_parent_capacity_fd():
    text = os.environ.get(PARENT_CAPACITY_FD_ENV, '')
    need(type(text) is str and text.isdecimal() and len(text) <= 10,
         'explicit original parent capacity descriptor missing')
    descriptor = int(text); need(descriptor > 2, 'parent descriptor must not be stdio')
    info = os.fstat(descriptor)
    seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 0 and info.st_uid == os.geteuid()
         and stat.S_IMODE(info.st_mode) == 0o400 and 0 < info.st_size <= 8192
         and os.readlink('/proc/self/fd/' + str(descriptor)).startswith('/memfd:native-metadata-parent-capacity-v1')
         and fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
         'original sealed private parent descriptor differs')
    # Producer, verifier, probes and subreaper never inherit this capability.
    os.set_inheritable(descriptor, False)


def _admit_parent_capacity_io(event, arguments, *, accelerate_root, source_map, records):
    if event not in {'open', 'os.rename', 'os.remove', 'os.unlink', 'os.mkdir', 'os.chmod'}:
        return False
    prefix = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.'
    receiver = sys.modules.get(prefix + 'native_metadata_parent_resources')
    module = sys.modules.get(prefix + 'native_metadata_parent_io')
    if receiver is None or module is None:
        return False
    context = getattr(receiver._OPERATION, 'value', None)
    if context is None:
        return False
    expected = Path(accelerate_root) / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/native_metadata_parent_io.py'
    function = module.authorize_parent_capacity_io
    need(type(module) is types.ModuleType and module.__dict__.get('__file__') == str(expected)
         and getattr(module.__dict__.get('__spec__'), 'origin', None) == str(expected)
         and str(expected) in source_map
         and function.__globals__ is module.__dict__
         and function.__code__.co_filename == str(expected),
         'normal source-bound parent IO admission helper required')
    record = function(event, arguments, context)
    if record is None:
        return False
    fields = frozenset({'schema', 'event', 'operation', 'paths', 'source', 'original_frame',
                        'disk_reservation_id', 'native_parent_lease_id'})
    need(type(record) is dict and set(record) == fields
         and module.PARENT_CAPACITY_IO_ADMISSION_FIELDS == fields
         and record['event'] == event and len(records) < 8192
         and len(json.dumps(record, sort_keys=True, allow_nan=False).encode()) <= 8192,
         'bounded closed parent IO admission required')
    records.append(record)
    return True

def child_main(args):
    value,raw=request(args.request,args.expected_request_sha256)
    # No numerical/native import or productive write before original attachment.
    need(sys.stdin.buffer.read(1)==b'1','original startup barrier not released')
    _check_inherited_parent_capacity_fd()
    source_guard(value,args.request,raw)
    need(str(Path.cwd())==value['cwd'],'actor cwd changed')
    need(os.environ.get('IPFS_ACCELERATE_AGENT_QUACK_PREFER')=='0','explicit original embedded file mode changed')
    denied=[];allowed_events=[];write_events=[];mkdir_events=[];fd_events=[];parent_io_events=[]
    actual_popen=subprocess.Popen
    spawn_lock=threading.RLock();pending_spawn={};spawn_index=0
    driver_raw=regular(ACTOR_PATH)
    need(hashlib.sha256(driver_raw).hexdigest()==value['source_sha256'][str(ACTOR_PATH)],'guarded driver source changed')
    driver=types.ModuleType('_exact_native_metadata_execution_driver')
    driver.__file__=str(ACTOR_PATH);sys.modules[driver.__name__]=driver
    exec(compile(driver_raw,str(ACTOR_PATH),'exec',dont_inherit=True),driver.__dict__)
    authorize=driver.authorize_native_control_popen
    authorize_fd=driver.authorize_native_control_fd_event
    need(authorize.__globals__ is driver.__dict__ and authorize.__code__.co_filename==str(ACTOR_PATH)
         and authorize_fd.__globals__ is driver.__dict__ and authorize_fd.__code__.co_filename==str(ACTOR_PATH),
         'exact source-bound driver admission functions required')
    plan_path=Path(value['argv'][4]);plan=decode(regular(plan_path))
    need(hashlib.sha256(regular(plan_path)).hexdigest()==value['argv'][6],'driver plan changed')
    sys.path[:0]=[str(R),plan['accelerate_root']]
    # The source guard inside driver.main refuses ipfs_datasets_py if it is already
    # loaded. Telemetry is only needed when a checked child exists, which is after
    # that guard is installed, so the import stays on the guarded path.
    process_host_path=R/'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py'
    need(hashlib.sha256(regular(process_host_path)).hexdigest()==value['source_sha256'][str(process_host_path)],
         'original process telemetry source identity differs')
    process_host_cache={}
    def process_host_module():
        host=process_host_cache.get('module')
        if host is None:
            need(any(type(finder).__name__=='GuardedProjectImports' for finder in sys.meta_path),
                 'process telemetry import requires the installed source guard')
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_daemon_resources as host
            need(Path(host.__file__).resolve()==process_host_path,'original process telemetry source identity differs')
            process_host_cache['module']=host
        return process_host_cache['module']
    def checked_record(record,fields):
        need(type(record) is dict and set(record)==fields,'closed driver admission record required')
        need(len(json.dumps(record,sort_keys=True,allow_nan=False).encode())<=64*1024,'bounded process admission record required')
        return record
    class CheckedNativePopen(actual_popen):
        def __init__(self,*popen_args,**kwargs):
            nonlocal spawn_index
            argv=kwargs.get('args',popen_args[0] if popen_args else None)
            host=process_host_module()
            with spawn_lock:
                admission=checked_record(authorize(argv,popen_args[1:] if popen_args else (),kwargs,plan),PROCESS_ADMISSION_FIELDS)
                need(admission['kind'] in PROCESS_ADMISSION_KINDS and admission['argv']==argv,'closed native process family required')
                need(not pending_spawn,'nested or concurrent unowned process constructor')
                spawn_index+=1;ordinal=spawn_index
                cwd=str(Path.cwd()) if kwargs.get('cwd') is None else str(kwargs['cwd'])
                expected=dict(argv=list(argv),cwd=cwd,env=kwargs.get('env'),admission=admission)
                intent=Path(value['attempt_directory'])/('subprocess-intent-%06d.json'%ordinal)
                effective=dict(kwargs);stdio=[]
                for stream in ('stdin','stdout','stderr'):
                    if effective.get(stream) is subprocess.DEVNULL or effective.get(stream)==subprocess.DEVNULL:
                        sink=Path(value['attempt_directory'])/('stdio-%06d-%s.bin'%(ordinal,stream))
                        flags=os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|(os.O_RDONLY if stream=='stdin' else os.O_WRONLY)
                        fd=os.open(sink,flags,0o600);info=os.fstat(fd)
                        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_uid==os.geteuid()
                             and info.st_size==0 and info.st_mode&0o077==0,'held ordinary private stdio sink required')
                        stdio.append((fd,sink,(info.st_dev,info.st_ino,info.st_uid,info.st_mode,info.st_nlink)))
                        effective[stream]=fd
                write_json(intent,dict(schema='native-metadata-process-intent/v1',request_sha256=args.expected_request_sha256,
                    ordinal=ordinal,admission=admission,successful_spawn=False,custody_outcome='constructor_not_returned'))
                pending_spawn.update(expected)
                try:
                    super().__init__(*popen_args,**effective)
                    identity=host._process(self.pid)
                    need(identity is not None,'spawn returned without observable original PID/birth')
                    spawn={'schema':'native-metadata-process-spawn/v1','request_sha256':args.expected_request_sha256,
                        'ordinal':ordinal,'admission':admission,'pid':self.pid,'birth':identity['birth'],
                        'group_pid':identity['group_pid'],'successful_spawn':True,
                        'held_stdio_sinks':[{'path':str(path),'device':witness[0],'inode':witness[1],
                            'uid':witness[2],'mode':witness[3],'nlink':witness[4]}for _,path,witness in stdio]}
                    write_json(Path(value['attempt_directory'])/('subprocess-spawn-%06d.json'%ordinal),spawn)
                    allowed_events.append(spawn)
                except BaseException as error:
                    write_json(Path(value['attempt_directory'])/('subprocess-uncertain-%06d.json'%ordinal),dict(
                        schema='native-metadata-process-constructor-uncertainty/v1',request_sha256=args.expected_request_sha256,
                        ordinal=ordinal,admission=admission,error_type=type(error).__name__,successful_spawn_not_proved=True))
                    raise
                finally:
                    pending_spawn.clear()
                    for fd,path,witness in stdio:
                        try:
                            info=os.fstat(fd);current=path.lstat()
                            need((info.st_dev,info.st_ino,info.st_uid,info.st_mode,info.st_nlink)==witness
                                 and (current.st_dev,current.st_ino,current.st_uid,current.st_mode,current.st_nlink)==witness,
                                 'held stdio identity changed')
                        finally:
                            os.close(fd)
    subprocess.Popen=CheckedNativePopen
    allowed=value['allowed_subprocess_argv']
    attempt=Path(value['attempt_directory'])
    external={row['path'] for row in value['external_files']}
    publication=external & set(PUBLICATION_CAPS)
    missing_parents=set();existing_external_parents={}
    catalog_parent=DATABASE.parent.lstat()
    catalog_parent_identity=(catalog_parent.st_dev,catalog_parent.st_ino,catalog_parent.st_uid,catalog_parent.st_mode)
    for name in publication:
        for directory in Path(name).parents:
            if directory.exists():
                need(directory.is_dir() and not directory.is_symlink(),'publication ancestor custody differs')
                info=directory.lstat();existing_external_parents[str(directory)]=(info.st_dev,info.st_ino,info.st_uid,info.st_mode)
                break
            need(any(root in directory.parents for root in ROOTS),'publication parent outside original roots')
            missing_parents.add(str(directory))
    def target_path(path,dir_fd=None):
        need(dir_fd in (None,-1) or type(dir_fd) is int and dir_fd>=0,'directory descriptor required')
        if type(path) is int:
            if path in (0,1,2):
                return None
            path=os.readlink('/proc/self/fd/'+str(path))
        path=Path(os.fsdecode(path))
        if path.is_absolute() or dir_fd in (None,-1):
            if not path.is_absolute():path=Path.cwd()/path
        else:
            base=Path(os.readlink('/proc/self/fd/'+str(dir_fd)))
            need(base.is_absolute() and not str(base).endswith(' (deleted)'),'directory descriptor path required')
            path=base/path
        path=Path(os.path.abspath(path))
        need(not any(p.is_symlink() for p in (path,*path.parents)),'write traverses an aliased path')
        return path
    def private(path):
        return path is not None and (path==attempt or attempt in path.parents)
    def writable(path):
        return private(path)
    def refuse(event, detail=''):
        denied.append(event)
        raise RuntimeError('control actor attempted an undeclared filesystem, process or network effect'
                           + ('' if not detail else ': '+detail))
    def check_audit(event,event_args):
        if _admit_parent_capacity_io(event,event_args,accelerate_root=Path(plan['accelerate_root']),
                source_map=value['source_sha256'],records=parent_io_events):return
        if event=='os.chmod' and len(event_args)==3 and type(event_args[0]) is int and event_args[0]>2:
            admission=authorize_fd(event,event_args,plan)
            if admission is not None:
                checked_record(admission,FD_ADMISSION_FIELDS)
                need(admission['kind']=='sealed_launcher_fd' and admission['event']==event
                     and admission['fd']==event_args[0] and admission['mode']==0o500,
                     'only the original sealed executable memfd construction is admitted')
                fd_events.append(admission);return
        if event=='open':
            path,mode,flags=event_args
            if (flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
                    or type(mode) is str and any(x in mode for x in 'wax+')):
                # Text-mode Popen audits io.open of the stdin pipe as a write.
                # The descriptor is anonymous; joining it onto the cwd is not a file write.
                anonymous=None
                if type(path) is int and path>2:
                    link=os.readlink('/proc/self/fd/'+str(path))
                    if link.startswith('pipe:['):anonymous=link
                if anonymous is not None:
                    write_events.append('anonymous-stdio:'+anonymous)
                else:
                    resolved=target_path(path)
                    if resolved is not None and not writable(resolved):refuse(event, str(resolved))
                    write_events.append(str(resolved) if resolved is not None else 'inherited-standard-descriptor')
        elif event=='os.mkdir':
            path,_mode,dir_fd=event_args;resolved=target_path(path,dir_fd)
            existing_owner_noop=False
            if resolved==DATABASE.parent or str(resolved) in existing_external_parents:
                current=resolved.lstat();expected_identity=(catalog_parent_identity if resolved==DATABASE.parent else existing_external_parents[str(resolved)])
                existing_owner_noop=(stat.S_ISDIR(current.st_mode) and
                    (current.st_dev,current.st_ino,current.st_uid,current.st_mode)==expected_identity)
            if not private(resolved) and str(resolved) not in missing_parents and not existing_owner_noop:refuse(event)
            mkdir_events.append({'path':str(resolved),'existing_native_owner_noop':existing_owner_noop})
        elif event=='os.rename':
            src,dst,src_fd,dst_fd=event_args
            source=target_path(src,src_fd);destination=target_path(dst,dst_fd)
            if not private(source) or not writable(destination):refuse(event)
        elif event in {'os.remove','os.rmdir'}:
            path,dir_fd=event_args
            resolved=target_path(path,dir_fd)
            if not private(resolved):refuse(event, str(resolved))
        elif event in {'os.chmod','os.chown','os.utime'}:
            path=event_args[0]
            dir_fd=event_args[2] if event=='os.chmod' else event_args[3]
            resolved=target_path(path,dir_fd)
            if not writable(resolved):refuse(event, str(resolved))
        elif event in {'os.symlink','os.link'}:
            refuse(event)
        if event=='subprocess.Popen':
            executable,call,cwd,environment=event_args
            actual=list(call) if isinstance(call,(list,tuple)) else None
            actual_cwd=str(Path.cwd()) if cwd is None else str(cwd)
            need(bool(pending_spawn),'process audit event has no held original driver admission')
            if (actual==pending_spawn['argv'] and executable==actual[0]
                    and actual_cwd==pending_spawn['cwd'] and environment==pending_spawn['env']):
                return
        if event in {'subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec',
                     'socket.connect','socket.bind','socket.getaddrinfo'}:
            denied.append(event)
            raise RuntimeError('control actor attempted an undeclared process or provider/network operation')
    def audit(event,event_args):
        before=len(denied)
        try:
            check_audit(event,event_args)
        except BaseException:
            if len(denied)==before:denied.append(event)
            raise
    sys.addaudithook(audit)
    sys.path[:0]=[str(R),plan['accelerate_root']]
    sys.argv=value['argv'][2:]
    try:
        code=driver.main();need(type(code)is int and code==0,'original guarded native CLI returned nonzero')
    finally:
        try:
            source_guard(value,args.request,raw)
        finally:
            write_json(Path(value['attempt_directory'])/'allowed-subprocess-events.json',{
                'schema':'legal-ir-control-allowed-subprocess-events/v1',
                'request_sha256':args.expected_request_sha256,'events':allowed_events,'denied_events':denied,
                'declared_subprocesses':len(allowed),'write_events':write_events,'mkdir_events':mkdir_events,
                'fd_admissions':fd_events,'parent_capacity_io_admissions':parent_io_events,'process_admission_driver':pin(ACTOR_PATH),
                'scope':'normal SDK import Git reads only; original owner retains actual PID/birth custody; no task/provider subprocesses',
                'writable_publication_outputs':sorted(publication),
                'filesystem_audit_scope':'Python audit denies external writes; original embedded read-session performs SELECT-only reads; native DB adapter is not a kernel sandbox',
                'subprocess_success_requires_the_actor_result_and_frozen_original_CLI_outputs':True})
        subprocess.Popen=actual_popen
        need(not denied,'control actor attempted a denied process/network effect')



def sampled_process_custody(tracker,child,birth,host):
    tracker.refresh()
    need(not tracker.error,'original sampled descendant tracker failed')
    live=tracker.live_descendants();identities=[]
    root=host._process(child.pid)
    need(root is None or root['birth']==birth,'original child PID reused')
    for pid in live:
        identity=host._process(pid);known=tracker.records.get(pid)
        need(identity is not None and known is not None and identity['birth']==str(known[0]),'sampled descendant PID/birth differs')
        identities.append(identity)
    leaders={child.pid,*live}
    need(all(row['group_pid'] in leaders for row in identities),'sampled descendant has an unowned process group leader')
    group=host._group_usage(dict(pid=child.pid,birth=birth))
    need(group.get('available') is True,'original group telemetry unavailable')
    extra=[row for row in identities if row['group_pid']!=child.pid]
    count=group['live_processes']+len(extra);rss=group['rss_bytes']+sum(row['rss_bytes']for row in extra)
    need(count<=8 and rss<=1536*1024**2,'sampled whole descendant tree exceeded original execution envelope')
    return dict(scope='sampled_birth_bound_descendants',live_processes=count,rss_bytes=rss,
        live_identities=[{k:row[k]for k in ('pid','birth','group_pid','parent_pid')}for row in identities],
        observed_identities=[identity.to_dict()for _,identity in sorted(tracker.history.items())],
        tracking_error=tracker.error,cooperative_not_kernel_containment=True)


def process_spawn_receipts(attempt,digest,host):
    intents=sorted(attempt.glob('subprocess-intent-*.json'));spawns=[]
    need(not list(attempt.glob('subprocess-uncertain-*.json')),'an actual process constructor has unresolved custody')
    for ordinal,path in enumerate(intents,1):
        intent=decode(regular(path));spawn_path=attempt/('subprocess-spawn-%06d.json'%ordinal)
        need(path.name=='subprocess-intent-%06d.json'%ordinal and intent['request_sha256']==digest
             and intent['ordinal']==ordinal,'durable process construction order differs')
        spawn=decode(regular(spawn_path));need(spawn['request_sha256']==digest and spawn['ordinal']==ordinal
             and spawn['admission']==intent['admission'] and spawn['successful_spawn'] is True,
             'durable process construction has no matching actual PID/birth receipt')
        current=host._process(spawn['pid'])
        need(current is None or current['birth']!=spawn['birth'] or current['state']=='Z','an admitted process identity is still alive')
        spawns.append(spawn)
    need(len(list(attempt.glob('subprocess-spawn-*.json')))==len(intents),'orphan process spawn receipt')
    return dict(schema='native-metadata-stopped-process-receipts/v1',intents=len(intents),spawns=spawns,
        all_recorded_original_births_stopped=True,resource_or_completion_authority=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--expected-request-sha256',required=True)
    parser.add_argument('--_child',action='store_true')
    args=parser.parse_args()
    if args._child:
        child_main(args);return 0
    value,raw=request(args.request,args.expected_request_sha256)
    os.environ.pop(PARENT_CAPACITY_FD_ENV, None)
    source_guard(value,args.request,raw)
    lease_module=load_source_lease()
    contexts=[];root=phase=owner=child=tracker=None
    owner_entered=False;birth=None;child_started=False;child_reaped=False;child_launch_attempted=False
    resource_scope=None;cleanup_scope=None;resource_scope_entered=False;cleanup_scope_entered=False
    compute_closed=False;storage_status='not_requested';failure=None;cleanup_errors=[]
    cancelled=threading.Event();signals=[];started=time.monotonic();deadline=started+value['timeout_seconds']
    work_deadline=deadline-60.0  # Reserve60s of the unchanged300s lifetime for stopped cleanup.
    result={'schema':'legal-ir-original-native-control-terminal/v1','request_sha256':args.expected_request_sha256,
        'owner_pid':os.getpid(),'completed':False,'child_returncode':None,'source_READ_closed':False,
        'control_effects':'not_started','task_completion_authority_granted_by_resource_owner':False,
        'resource_profile':'original_orchestration_parented_disk_sampled_birth_bound_descendants',
        'external_file_growth_enforcement':'polled_size_caps_and_original_conservative_precharge_not_sandbox'}
    previous={}
    parent_capacity_fd=None
    def remaining():
        need(not cancelled.is_set(),'control request cancelled')
        left=work_deadline-time.monotonic();need(left>0,'control request deadline expired');return left
    def receive(number,_frame):signals.append(number);cancelled.set()
    def guard():
        remaining();source_guard(value,args.request,raw)
        if root is not None:need(root.renew() and not root.cancelled,'original root revoked')
        if phase is not None:need(phase.renew() and not phase.cancelled,'original phase revoked')
        if owner_entered:need(owner.heartbeat() and not owner.cancelled,'original disk child revoked')
    try:
        for source_root in value['source_roots']:
            ctx=lease_module.source_lease(source_root,mode='read',timeout_seconds=min(60,remaining()))
            ctx.__enter__();contexts.append(ctx)
        os.environ.update(IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',IPFS_ACCEL_SKIP_CORE='1',
            IPFS_AUTO_INSTALL='false',IPFS_DATASETS_AUTO_INSTALL='0',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TRANSFORMERS_OFFLINE='1',
            IPFS_ACCELERATE_AGENT_QUACK_PREFER='0')
        # Optional mirrors must not write unrelated metadata stores.
        os.environ.pop('IPFS_ACCELERATE_META_INDEX_DUCKDB',None)
        os.environ.pop('IPFS_ACCELERATE_META_INDEX_DUCKLAKE',None)
        sys.path.insert(0,str(R));sys.dont_write_bytecode=True
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane, resource_admission_deadline, resource_cleanup_deadline
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_daemon_resources as host
        from ipfs_datasets_py.logic.backends.process import _LinuxDescendantTracker
        need(Path(host.__file__).resolve()==R/'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py','original resource module escaped')
        spec=importlib.util.spec_from_file_location('_original_group_cleanup_control',R/'scripts/ops/legal_ir/run_owned_validation.py')
        cleanup_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cleanup_module)
        result['owner_birth']=host._process(os.getpid())['birth']
        previous={n:signal.signal(n,receive) for n in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        # Refuse a stale DB before any original native/storage grant; repeat below.
        result['database_preimage_before_grants']=pin(DATABASE)
        need(result['database_preimage_before_grants']==value['database_preimage'],'closed catalog preimage differs before grants')
        resource_scope=resource_admission_deadline(work_deadline_monotonic=work_deadline,native_deadline_monotonic=deadline)
        resource_scope.__enter__();resource_scope_entered=True
        result['resource_deadline_profile']=dict(work_seconds=240,stopped_cleanup_reserve_seconds=60,native_lifetime_seconds=300,absolute_clock_shared_by_all_resource_operations=True)
        scheduler=open_existing_scheduler(STATE,POLICY)
        root=scheduler.acquire(ResourceLane.ORCHESTRATION,cpu_slots=1,memory_mb=1536,child_process_slots=8,
            requires_gpu=False,timeout=remaining(),cancel_event=cancelled,request_id=value['request_id']+':root')
        phase=root.acquire_child(lane=ResourceLane.ORCHESTRATION,cpu_slots=1,memory_mb=1536,child_process_slots=8,
            timeout=remaining(),cancel_event=cancelled,request_id=value['request_id']+':phase')
        attempt=Path(value['attempt_directory'])
        need(attempt.is_dir() and not any(attempt.iterdir()) and attempt.lstat().st_uid==os.geteuid()
             and attempt.lstat().st_mode&0o777==0o700,'caller-prepared private attempt changed before original admission')
        owner=ParentedPhaseResourceReservation(LEDGER,roots=ROOTS,attempt_directory=attempt,parent_lease=phase,
            storage_bytes=value['storage_bytes'],memory_mb=1536,cpu_slots=1,child_process_slots=8,
            scheduler_state_path=STATE,expected_scheduler_policy_sha256=POLICY,
            timeout_seconds=remaining(),ledger_lock_timeout_seconds=None)
        owner.__enter__();owner_entered=True;storage_status='active'
        need(owner.to_dict()['workload']=='canonical_parented_phase'
             and owner.to_dict()['scheduler_lane']=='orchestration','original control-purpose profile required')
        # One immutable logical catalog bundle covers DB, WAL and writer lock.
        # This avoids repeating the original full global census per sidecar.
        bundle=json.dumps(value['external_files'],sort_keys=True,separators=(',',':')).encode()
        result['external_precharge']=owner.account_external_bytes(
            'control-'+hashlib.sha256(bundle).hexdigest(),sum(row['max_bytes'] for row in value['external_files']))
        # Genuine producer receipt, emitted after original admission/precharge
        # and before any child. The distributed bundle is declared honestly;
        # it is NOT the one-tree external_precharge schema of legacy899f.
        execution_path=attempt/'execution-request.json'
        write_json(execution_path,dict(schema='native-metadata-resource-owner-execution-request/v1',
            disk_reservation_id=owner.reservation_id,request=pin(args.request),
            owner_source=pin(Path(__file__).resolve()),attempt_directory=owner.to_dict()['record']['attempt_directory'],
            external_precharge={'key':result['external_precharge']['key'],
                'bytes':result['external_precharge']['bytes'],'files':value['external_files']},
            original_charge_receipt=result['external_precharge'],source_roots=value['source_roots'],
            existing_single_tree_recovery_compatible=False,provider_authority=False,completion_authority=False))
        result['execution_request']=pin(execution_path)
        result['external_before']=external_usage(value)
        result['database_preimage_observed']=pin(DATABASE)
        need(result['database_preimage_observed']==value['database_preimage'],'closed catalog preimage differs')
        (attempt/'exports').mkdir(mode=0o700)
        (attempt/'temporary').mkdir(mode=0o700)
        # These exact declared metadata-output/journal parents are established
        # only after original admission and external byte precharge. No source,
        # profile, task input or existing file is created or replaced here.
        result['external_output_parent_identities']=[]
        for directory in sorted({Path(name).parent for name in PUBLICATION_CAPS},key=str):
            need(not any(parent.is_symlink() for parent in (directory,*directory.parents)),
                 'external output parent aliases are forbidden')
            directory.mkdir(mode=0o700,parents=True,exist_ok=True)
            info=directory.lstat();need(stat.S_ISDIR(info.st_mode)and info.st_uid==os.geteuid(),
                 'external output parent ownership differs')
            result['external_output_parent_identities'].append(dict(path=str(directory),device=info.st_dev,
                inode=info.st_ino,uid=info.st_uid,mode=info.st_mode))
        stdout=attempt/'stdout.log';stderr=attempt/'stderr.log'
        with stdout.open('xb') as out,stderr.open('xb') as err:
            os.fchmod(out.fileno(),0o600);os.fchmod(err.fileno(),0o600)
            guard()
            parent_capacity_fd=_new_parent_capacity_fd()
            child_launch_attempted=True  # A raised Popen is never proof that no child existed.
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--request',str(args.request),
                '--expected-request-sha256',args.expected_request_sha256,'--_child'],cwd=value['cwd'],
                env={**os.environ, PARENT_CAPACITY_FD_ENV: str(parent_capacity_fd)},
                pass_fds=(parent_capacity_fd,),stdin=subprocess.PIPE,stdout=out,stderr=err,start_new_session=True,umask=0o077)
            child_started=True
            current=host._process(child.pid);need(current is not None,'child exited before attachment')
            birth=current['birth'];owner.check_usage(attempt,child_pid=child.pid)
            tracker=_LinuxDescendantTracker(child.pid,retain_history=True);tracker.refresh()
            need(tracker.enabled and not tracker.error,'original descendant tracker unavailable')
            guard()
            result['parent_capacity_handoff']=_seal_parent_capacity_fd(parent_capacity_fd,
                owner=owner,value=value,request_path=args.request,execution_request=result['execution_request'],
                work_deadline=work_deadline,native_deadline=deadline,child_pid=child.pid,child_birth=birth)
            child.stdin.write(b'1');child.stdin.flush();child.stdin.close()
            os.close(parent_capacity_fd);parent_capacity_fd=None
            result['control_effects']='started_reconcile_on_failure'
            while child.poll() is None:
                guard();owner.check_usage(attempt);external_usage(value);tracker.refresh()
                result['sampled_process_custody']=sampled_process_custody(tracker,child,birth,host)
                time.sleep(.1)
            result['child_returncode']=child.returncode
            cleanup_module._stop_original_group(child,birth)
            _,live=tracker.drain(3);need(not live and not tracker.error,'original sampled descendant cleanup unresolved')
            result['sampled_process_custody_final']=sampled_process_custody(tracker,child,birth,host)
            need(result['sampled_process_custody_final']['live_processes']==0,'sampled process group/tree remains alive')
            result['process_spawn_receipts']=process_spawn_receipts(attempt,args.expected_request_sha256,host)
            child_reaped=True;out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
        guard();result['external_after']=external_usage(value)
        result['database_postimage']=pin(DATABASE)
        need(child.returncode==0,'control actor failed; reconcile exact partial operation before retry')
        event_path=attempt/'allowed-subprocess-events.json'
        observed=decode(regular(event_path))
        need(observed.get('request_sha256')==args.expected_request_sha256
             and observed.get('denied_events')==[]
              and observed.get('process_admission_driver')==pin(ACTOR_PATH)
             and len(observed.get('events',[]))==result['process_spawn_receipts']['intents'],
             'declared original CLI invocation receipt differs')
        result['allowed_subprocess_events']=pin(event_path)
        result['control_effects']='actor_exit_zero_independent_task_receipt_still_required'
        write_json(attempt/'control-execution.json',{**result,'completed':True})
        owner.finalize(attempt,artifacts_durable=True);storage_status='released'
        result['completed']=True
    except BaseException as error:
        failure=error;result.update(error_type=type(error).__name__,error=str(error))
        if isinstance(getattr(error,'_capacity_observation',None),dict):
            result['resource_capacity_observation']=getattr(error,'_capacity_observation')
    finally:
        if parent_capacity_fd is not None:
            os.close(parent_capacity_fd);parent_capacity_fd=None
        # Write truthful owned failure evidence before any deadline-bound close.
        # Never call this branch after even an attempted process construction.
        prechild_receipt=None
        if failure is not None and owner is not None and not child_launch_attempted and child is None and birth is None and tracker is None:
            try:
                observed=owner.to_dict();record=observed.get('record')
                if record is not None and record['status']!='released':
                    need(record['attempt_directory']==host._root_identity(attempt)
                         and record['child'] is None and record.get('prior_children',[])==[],
                         'prechild stopped owner identity differs')
                    failure_path=attempt/'control-failure-before-child.json'
                    write_json(failure_path,dict(schema='legal-ir-original-prechild-control-failure/v1',
                        request_sha256=args.expected_request_sha256,owner_pid=os.getpid(),owner_birth=result['owner_birth'],
                        reservation_id=record['reservation_id'],attempt_directory=record['attempt_directory'],
                        completed=False,control_effects='not_started',child_launch_attempted=False,
                        error_type=type(failure).__name__,error=str(failure),
                        resource_capacity_observation=result.get('resource_capacity_observation'),
                        original_control_failure_returncode=1,task_completion_authority=False))
                    prechild_receipt=pin(failure_path);result['prechild_failure_receipt']=prechild_receipt
            except BaseException as evidence_error:
                result['prechild_failure_evidence_error']=type(evidence_error).__name__+': '+str(evidence_error)
        if resource_scope_entered:
            try:
                cleanup_scope=resource_cleanup_deadline(native_deadline_monotonic=deadline)
                cleanup_scope.__enter__();cleanup_scope_entered=True
            except BaseException as deadline_error:
                cleanup_errors.append(type(deadline_error).__name__+': '+str(deadline_error))
        if prechild_receipt is not None and cleanup_scope_entered:
            try:
                if owner_entered and owner.to_dict()['status']=='active' and time.monotonic()<work_deadline:
                    owner.finalize(attempt,artifacts_durable=True)
                    result['prechild_storage_close_method']='original_finalize'
                else:
                    # Original documented stopped-cleanup wrapper, including
                    # a partial __enter__ that already released native custody.
                    owner.release(artifacts_durable=True)
                    result['prechild_storage_close_method']='original_stopped_release_same_fresh_finalizer'
                need(owner.to_dict()['status']=='released','prechild storage did not close')
                storage_status='released';result['prechild_storage_closed']=True
            except BaseException as close_error:
                result['prechild_storage_closed']=False
                result['prechild_storage_close_error']=type(close_error).__name__+': '+str(close_error)
                # Keep the real failure file; normal original retention follows.
        # No original handle or source lease is released across unknown cleanup.
        if child is not None and birth is not None:
            try:
                if tracker is not None:tracker.refresh()
                cleanup_module._stop_original_group(child,birth)
                if tracker is not None:
                    _,live=tracker.drain(3);need(not live and not tracker.error,'descendant cleanup unresolved')
                    result['sampled_process_custody_final']=sampled_process_custody(tracker,child,birth,host)
                    need(result['sampled_process_custody_final']['live_processes']==0,'sampled process custody remains live')
                    result['process_spawn_receipts']=process_spawn_receipts(attempt,args.expected_request_sha256,host)
                child_reaped=True
            except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
        elif child_launch_attempted:
            cleanup_errors.append('child construction attempted without proven birth/group reap; original process custody unresolved')
        if tracker is not None and not cleanup_errors:tracker.close()
        if owner is not None:
            if not cleanup_errors and (not child_launch_attempted or child_reaped):
                try:
                    if owner.to_dict()['status']!='released':
                        owner.__exit__(type(failure) if failure else RuntimeError,failure,None)
                    storage_status=owner.to_dict()['status'];result['reservation']=owner.to_dict()
                except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
            else:
                # Popen can raise after an attempted construction. A missing
                # returned child object is not a stopped-child witness.
                result['owner_handle_held_for_unknown_or_unclosed_custody']=True
                try:storage_status=owner.to_dict()['status'];result['reservation']=owner.to_dict()
                except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
        if not cleanup_errors and (not child_launch_attempted or child_reaped):
            try:
                for native in (phase,root):
                    if native is not None:native.release()
                ids={x.lease_id for x in (root,phase) if x is not None}
                if owner is not None and owner.to_dict()['resource_lease'] is not None:
                    ids.add(owner.to_dict()['resource_lease']['lease_id'])
                if ids:need(not any(x['lease_id'] in ids for x in scheduler.active_leases()),'own native ancestry remains live')
                source_guard(value,args.request,raw);compute_closed=True
            except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
        result.update(storage_status=storage_status,child_reaped=child_reaped,child_launch_attempted=child_launch_attempted,
            original_native_compute_closed=compute_closed,cleanup_errors=cleanup_errors,
            elapsed_seconds=time.monotonic()-started,received_signals=signals,
            partial_or_failed_control_must_be_reconciled=failure is not None)
        if compute_closed and not cleanup_errors:
            try:
                for ctx in reversed(contexts):ctx.__exit__(None,None,None)
                result['source_READ_closed']=True
            except BaseException as error:
                cleanup_errors.append(type(error).__name__+': '+str(error))
        try:
            if cleanup_scope_entered:cleanup_scope.__exit__(None,None,None);cleanup_scope_entered=False
            if resource_scope_entered:resource_scope.__exit__(None,None,None);resource_scope_entered=False
            result['absolute_resource_operation_deadline_closed']=True
        except BaseException as clock_error:
            cleanup_errors.append(type(clock_error).__name__+': '+str(clock_error))
        if result['source_READ_closed'] and not cleanup_errors:
            for number,handler in previous.items():signal.signal(number,handler)
            print(json.dumps(result,sort_keys=True),flush=True)
        else:
            result['source_READ_held_for_original_owner_cleanup']=True
            try:print(json.dumps(result,sort_keys=True),flush=True)
            except BaseException:pass
            # Exact original handles stay reachable. Never translate a logging,
            # resource or custody uncertainty into permission to drop READ.
            while True:time.sleep(10)
    return 0 if result['completed'] and result['source_READ_closed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
