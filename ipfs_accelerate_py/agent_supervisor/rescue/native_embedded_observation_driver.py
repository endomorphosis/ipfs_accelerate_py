"""Finite cold embedded observation through original source and resource owners.

This driver never claims a task or invokes an implementation/validation provider.
The enclosing original owner creates and closes every native/storage grant.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import runpy
import stat
import subprocess
import time
import sys
import tempfile
import types

DATASETS = Path('/home/barberb/lift_coding/external/ipfs_datasets')
WORKSPACE = DATASETS
RUNTIME = Path(__file__).absolute().parents[3]
DATABASE = DATASETS/'workspace/test-logs/federal-corpus-audits/legal-ir-supervisor-plan-20261016/catalog-02/control.duckdb'
CLI_MODULE = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon'
RUNNER_REL = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
HELPER_MODULE = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap'
HELPER_REL = 'ipfs_accelerate_py/agent_supervisor/todo_daemon/native_owner_bootstrap.py'
SOURCE_MODULE = 'ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source'
SOURCE_REL = 'ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py'
GUARD_SHA = '1926462c717afdb265e28c90c7723e8ae04eabb27bc6e54ca73e8169cd27c45f'
PLAN_FIELDS = frozenset({'schema','operation','accelerate_root','task_scope','native_metadata_cli'})
CLI_FIELDS = frozenset({'schema','runtime_root','cwd','attempt_directory','temporary_directory',
    'project_import_plan','pre_body_guard','result_path','source_sha256','observation_binding','import_head_capsule_Git'})
BINDING_FIELDS = frozenset({'schema','database_preimage','task_scope','plan_root_cid',
    'repository_tree_id','observer_source','source_schema'})


def need(value,message):
    if not value:
        raise RuntimeError(message)


def encode(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def regular(path,maximum=4*1024**2):
    path=Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path,*path.parents)),
         'unaliased absolute source required')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_uid==os.geteuid()
             and before.st_nlink==1 and 0<=before.st_size<=maximum,'bounded ordinary owner source required')
        raw=stream.read(maximum+1);after=os.fstat(stream.fileno())
    identity=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_uid,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    need(len(raw)==before.st_size and identity(before)==identity(after)==identity(path.lstat()),
         'source changed during read')
    return raw


def descriptor(path,maximum=4*1024**2):
    raw=regular(path,maximum)
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def decode(raw):
    def pairs(items):
        value={}
        for key,item in items:
            need(key not in value,'duplicate JSON key');value[key]=item
        return value
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def configuration(plan):
    need(type(plan) is dict and set(plan)==PLAN_FIELDS
         and plan['schema']=='native-embedded-scoped-observation-plan/v1'
         and plan['operation']=='native_embedded_scoped_observation_v1'
         and plan['accelerate_root']==str(RUNTIME),'closed source-bound observation plan required')
    scope=plan['task_scope']
    need(type(scope) is list and 1<=len(scope)<=256
         and all(type(x) is str and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:-]{0,255}',x) for x in scope)
         and len(set(scope))==len(scope),
         'finite unique task CID scope required')
    c=plan['native_metadata_cli']
    need(type(c) is dict and set(c)==CLI_FIELDS
         and c['schema']=='native-embedded-scoped-observation-cli/v2'
         and c['runtime_root']==str(RUNTIME) and c['cwd']==str(WORKSPACE),
         'closed original observation CLI binding required')
    attempt=Path(c['attempt_directory']);temporary=Path(c['temporary_directory'])
    need(attempt.is_absolute() and DATASETS/'workspace/test-logs' in attempt.parents
         and not any(p.is_symlink() for p in (attempt,*attempt.parents))
         and temporary==attempt/'temporary'
         and Path(c['result_path'])==attempt/'exports/native-cli-driver-result.json',
         'original owned observation paths required')
    pins=c['source_sha256']
    need(type(pins) is dict and 1<=len(pins)<=2048
         and all(type(k) is str and Path(k).is_absolute() and type(v) is str
             and re.fullmatch(r'[0-9a-f]{64}',v) for k,v in pins.items()),
         'explicit finite source closure required')
    guard=c['pre_body_guard']
    need(type(guard) is dict and set(guard)=={'path','bytes','sha256'}
         and guard['path']==str(RUNTIME/'ipfs_accelerate_py/agent_supervisor/rescue/project_import_source_guard.py')
         and guard['sha256']==GUARD_SHA,'canonical original before-body guard path required')
    binding=c['observation_binding']
    need(type(binding) is dict and set(binding)==BINDING_FIELDS
         and binding['schema']=='native-embedded-daemon-observation-binding@1'
         and binding['task_scope']==scope,'closed scoped observation binding required')
    prior=binding['database_preimage'];observer=binding['observer_source']
    for ref,maximum in ((prior,128*1024**2),(observer,4*1024**2)):
        need(type(ref) is dict and set(ref)=={'path','bytes','sha256'}
             and type(ref['path']) is str and type(ref['bytes']) is int
             and 0<ref['bytes']<=maximum and type(ref['sha256']) is str
             and re.fullmatch(r'[0-9a-f]{64}',ref['sha256']), 'closed physical descriptor required')
    need(prior['path']==str(DATABASE) and observer['path']==str(RUNTIME/HELPER_REL)
         and pins.get(observer['path'])==observer['sha256']
         and all(type(binding[k]) is str and 0<len(binding[k])<=4096
             for k in ('plan_root_cid','repository_tree_id','source_schema')),
         'exact actual database/helper/catalogue binding required')
    validate_import_git_binding(c)
    return c


def source_pin(path,c):
    observed=descriptor(path)
    need(c['source_sha256'].get(str(path))==observed['sha256'],'source pin differs: '+str(path))
    return observed


def module_owner(name,relative,c):
    module=sys.modules.get(name);path=RUNTIME/relative
    need(type(module)is types.ModuleType and Path(module.__file__)==path
         and module.__spec__ is not None and module.__spec__.origin==str(path),
         'original normal-import module required: '+name)
    return module,source_pin(path,c)


def frame_for(module,fn):
    need(type(fn)is types.FunctionType and fn.__globals__ is module.__dict__,
         'original function globals required')
    frame=sys._getframe(1)
    for _ in range(96):
        if frame is None:break
        if frame.f_code is fn.__code__ and frame.f_globals is module.__dict__:return frame
        frame=frame.f_back
    raise RuntimeError('original source-bound execution frame missing')


def environment_digest(environment):
    need(type(environment)is dict and all(type(k)is str and type(v)is str for k,v in environment.items()),
         'exact string environment required')
    return hashlib.sha256(encode(environment)).hexdigest()


def validate_import_git_binding(c):
    """Validate the retained original immutable-HEAD metadata contract only."""
    capsule=c['import_head_capsule_Git']
    fields={'blobs','ls_tree_tail','payload_bytes','required_relative_files','route_source',
        'scope','selected_file_count','source_head','source_root','unique_blob_count'}
    need(type(capsule)is dict and set(capsule)==fields,'closed original import HEAD contract required')
    need(capsule['source_root']==str(RUNTIME) and type(capsule['source_head'])is str
         and re.fullmatch(r'[0-9a-f]{40}',capsule['source_head']), 'exact import source root/HEAD required')
    route=capsule['route_source'];path=RUNTIME/'ipfs_accelerate_py/agent_implementation_route.py'
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


def authorize_native_control_popen(argv,args,kwargs,plan):
    """Admit only the normal SDK's source-bound read-only import Git calls."""
    c=configuration(plan);capsule=validate_import_git_binding(c)
    need(not args and type(argv)in (list,tuple) and 0<len(argv)<=256
         and all(type(x)is str and '\0'not in x and len(x)<=8192 for x in argv), 'bounded literal argv required')
    need(argv[0]=='git','embedded observation does not dispatch any subprocess except original import Git reads')
    route,source=module_owner('ipfs_accelerate_py.agent_implementation_route',
        'ipfs_accelerate_py/agent_implementation_route.py',c)
    daemon,_=module_owner(CLI_MODULE,RUNNER_REL,c)
    capture=frame_for(daemon,daemon._capture_imported_control_plane_generation)
    need(capture.f_locals.get('source_root')==RUNTIME and capsule['route_source']==source,
         'only current normal import source generation is admitted')
    need(type(kwargs)is dict and kwargs.get('cwd')==RUNTIME,
         'original import Git cwd required')
    prefix=['git','-c','core.quotepath=false','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null']
    need(list(argv[:7])==prefix,'exact original import Git prefix required')
    tail=list(argv[7:]);batch=tail==['cat-file','--batch']
    environment={name:value for name,value in os.environ.items() if not name.startswith('GIT_')}
    environment.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',GIT_TERMINAL_PROMPT='0',
        GIT_NO_REPLACE_OBJECTS='1',GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1',LC_ALL='C',LANG='C')
    need(environment.get('PATH')=='/usr/bin:/bin' and kwargs.get('env')==environment,
         'exact original noninteractive import Git environment required')
    need(getattr(route,'_AGENT_CONTROL_PLANE_GIT_TIMEOUT_SECONDS',None)==30,
         'original thirty-second Git deadline required')
    if batch:
        selected=frame_for(route,route._agent_control_plane_head_payloads)
        values=selected.f_locals
        need(values.get('root')==RUNTIME and values.get('source_head')==capsule['source_head']
             and values.get('blobs')=={row['relative_path']:row['object_id'] for row in capsule['blobs']}
             and set(capsule['required_relative_files'])==set(route._AGENT_CONTROL_PLANE_RELATIVE_FILES)
             and set(route._AGENT_CONTROL_PLANE_REQUIRED_SQL_FILES)=={'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql', 'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0003_state_server_restart_identity.sql', 'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0002_causal_event_federation_core.sql', 'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0004_hash_observations.sql', 'ipfs_accelerate_py/agent_supervisor/task_sources/sql/0005_tasks_index_repair.sql'},
             'actual immutable HEAD batch selection differs')
        expected={'cwd':RUNTIME,'env':environment,'stdin':subprocess.PIPE,'stdout':subprocess.PIPE,
            'stderr':subprocess.DEVNULL,'close_fds':True,'bufsize':0}
        need(set(kwargs)==set(expected) and kwargs==expected and type(kwargs['close_fds'])is bool
             and type(kwargs['bufsize'])is int,'exact original batch pipe/custody constructor required')
        need(type(values.get('deadline'))in (int,float) and
             0<values['deadline']-time.monotonic()<=30,'original batch deadline is not live')
        original_frame='_agent_control_plane_head_payloads'
    else:
        query=frame_for(route,route._agent_git_output)
        need(query.f_locals.get('root')==RUNTIME and list(query.f_locals.get('arguments',()))==tail,
             'original bounded Git query frame required')
        generation=tail in [['rev-parse','--verify','HEAD^{commit}'],['rev-parse','--verify','HEAD^{tree}'],
            ['rev-parse','--show-toplevel'],['status','--porcelain=v1','-z','--untracked-files=all']]
        need(generation or tail==capsule['ls_tree_tail'],'unprovided original source Git read denied')
        if tail==capsule['ls_tree_tail']:
            selected=frame_for(route,route._agent_control_plane_head_payloads)
            need(selected.f_locals.get('root')==RUNTIME and selected.f_locals.get('source_head')==capsule['source_head'],
                 'original immutable HEAD metadata selection frame required')
        run=frame_for(subprocess,subprocess.run)
        need(type(run.f_locals.get('timeout'))is int and run.f_locals['timeout']==30
             and run.f_locals.get('check')is False and run.f_locals.get('input')is None
             and run.f_locals.get('capture_output')is False,'original bounded subprocess.run semantics required')
        expected={'cwd':RUNTIME,'env':environment,'stdin':subprocess.DEVNULL,
            'stdout':subprocess.PIPE,'stderr':subprocess.PIPE}
        need(set(kwargs)==set(expected) and kwargs==expected,'exact original metadata pipe/custody constructor required')
        original_frame='_agent_git_output'
    return {'kind':'immutable_source_capsule_read','argv':list(argv),'cwd':str(RUNTIME),
        'environment_sha256':environment_digest(environment),'start_new_session':False,
        'original_frame':original_frame,'source':source,'sealed_launcher_fds':[],
        'status_fd':None,'declared_command_argv':[]}


def require_original_import_capability(daemon,c):
    """Refuse swallowed eager-import failures before emitting any idle result."""
    route,_=module_owner('ipfs_accelerate_py.agent_implementation_route',
        'ipfs_accelerate_py/agent_implementation_route.py',c)
    binding=validate_import_git_binding(c)
    capsule=getattr(daemon,'_IMPORTED_CONTROL_PLANE_CAPSULE',None)
    launch=getattr(daemon,'_IMPORTED_CONTROL_PLANE_LAUNCH',None)
    need(type(capsule)is route.AgentImplementationControlPlanePin
         and type(launch)is route.AgentImplementationSealedControlPlane
         and capsule.source_head==binding['source_head']
         and type(capsule.source_tree)is str and re.fullmatch(r'[0-9a-f]{40}',capsule.source_tree)
         and launch.capsule_id==capsule.capsule_id and launch.archive_sha256==capsule.archive_sha256,
         'genuine current eager source capsule and sealed launch required')
    verify=route.verify_agent_implementation_sealed_control_plane
    need(type(verify)is types.FunctionType and verify.__globals__ is route.__dict__
         and verify.__code__.co_filename==str(RUNTIME/'ipfs_accelerate_py/agent_implementation_route.py')
         and verify(capsule,launch.descriptor)==launch.executable_path,
         'original sealed control-plane verifier did not confirm current capability')


def authorize_native_control_fd_event(event,args,plan):
    """No executable-memfd or provider descriptor grant in observation mode."""
    configuration(plan)
    return None


def save(path,value):
    raw=encode(value);need(len(raw)<=4*1024**2,'bounded original observation receipt required')
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(),0o600);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    return descriptor(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--plan-sha256',required=True);args=parser.parse_args()
    raw=regular(args.plan);need(hashlib.sha256(raw).hexdigest()==args.plan_sha256,'plan changed')
    plan=decode(raw);c=configuration(plan);attempt=Path(c['attempt_directory'])
    need(sys.flags.dont_write_bytecode and not sys.flags.optimize,'normal no-bytecode source mode required')
    need(not any(n=='ipfs_accelerate_py' or n.startswith('ipfs_accelerate_py.') for n in sys.modules),
         'cold original SDK import required')
    for path,digest in c['source_sha256'].items():
        need(descriptor(path)['sha256']==digest,'source changed: '+path)
    guard_ref=c['pre_body_guard']
    need(descriptor(guard_ref['path'])==guard_ref and guard_ref['sha256']==GUARD_SHA,
         'original before-body source guard required')
    import_plan=c['project_import_plan']
    need(import_plan['accelerate_root']==str(RUNTIME) and import_plan['datasets_root']==str(DATASETS)
         and import_plan['publication_root']==str(DATASETS),'sealed original source roots required')
    guarded=runpy.run_path(guard_ref['path'],run_name='_original_embedded_observation_before_body_guard')
    importer=guarded['GuardedProjectImports'](import_plan,attempt/'exports').install()
    temporary=Path(c['temporary_directory']);info=temporary.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid==os.geteuid() and info.st_mode&0o077==0,
         'private original-owned temporary directory required')
    os.environ.update(TMPDIR=str(temporary),TMP=str(temporary),TEMP=str(temporary),
         PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',IPFS_ACCEL_SKIP_CORE='1',IPFS_ACCELERATE_AGENT_QUACK_PREFER='0')
    os.environ.pop('IPFS_ACCELERATE_META_INDEX_DUCKDB',None)
    os.environ.pop('IPFS_ACCELERATE_META_INDEX_DUCKLAKE',None)
    tempfile.tempdir=str(temporary);sys.path.insert(0,str(RUNTIME));sys.dont_write_bytecode=True
    source=daemon=None
    try:
        daemon_module=importlib.import_module(CLI_MODULE);module_owner(CLI_MODULE,RUNNER_REL,c)
        require_original_import_capability(daemon_module,c)
        helper=importlib.import_module(HELPER_MODULE);module_owner(HELPER_MODULE,HELPER_REL,c)
        source_module=importlib.import_module(SOURCE_MODULE);module_owner(SOURCE_MODULE,SOURCE_REL,c)
        binding=c['observation_binding']
        need(source_module.DATABASE_TASK_SOURCE_SCHEMA==binding['source_schema'],
             'sealed source schema differs from actual original source')
        observer=helper.observe_embedded_database_without_dispatch
        need(type(observer) is types.FunctionType and observer.__globals__ is helper.__dict__
             and observer.__code__.co_filename==str(RUNTIME/HELPER_REL),
             'original canonical observer callable required')
        source=source_module.DatabaseTaskSource(database_path=DATABASE,
            owner_id='native-embedded-scoped-observation',repository_tree_id=binding['repository_tree_id'],
            plan_root_cid=binding['plan_root_cid'],install_schema=False)
        daemon=daemon_module.DatabaseImplementationDaemon(database_path=DATABASE,
            coordination_path=attempt/'db/coordination.duckdb',execution_path=attempt/'db/execution.duckdb',
            state_dir=attempt/'db/daemon-state',state_prefix='embedded-observation',
            owner_session_id='native-embedded-scoped-observation',process_instance_id='observation:'+str(os.getpid()),
            authority_mode='embedded_exclusive',task_source_kind='duckdb',max_task_attempts=0,
            task_source=source,close_task_source=False,install_schema=False,
            execution_slice_task_cids=plan['task_scope'])
        result=observer(daemon,binding=binding)
        need(result['schema']=='native-embedded-daemon-observation@1'
             and result['task_scope']==plan['task_scope'] and result['provider_dispatched'] is False
             and result['attempt_consumed'] is False and result['completion_authority'] is False,
             'genuine observation result binding differs')
        save(Path(c['result_path']),result)
    finally:
        # Closing an unactivated original daemon does not create its stores.
        if daemon is not None:daemon.close()
        if source is not None:source.close()
        importer.checkpoint('closing-native-embedded-observation')
        need(regular(args.plan)==raw,'plan changed during original observation')
        for path,digest in c['source_sha256'].items():
            need(descriptor(path)['sha256']==digest,'source changed: '+path)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
