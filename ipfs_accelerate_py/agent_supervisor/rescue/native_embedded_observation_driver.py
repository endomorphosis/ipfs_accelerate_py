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
    'project_import_plan','pre_body_guard','result_path','source_sha256','observation_binding'})
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
         and c['schema']=='native-embedded-scoped-observation-cli/v1'
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
    return c


def module_owner(name,relative,c):
    module=sys.modules.get(name);path=RUNTIME/relative
    need(type(module) is types.ModuleType and Path(module.__file__)==path
         and module.__spec__ is not None and module.__spec__.origin==str(path),
         'original normal-import module required: '+name)
    need(c['source_sha256'].get(str(path))==descriptor(path)['sha256'],
         'normal-import module source pin differs: '+name)
    return module


def authorize_native_control_popen(argv,args,kwargs,plan):
    """An observation has no nested process constructor admission."""
    configuration(plan)
    raise RuntimeError('embedded observation does not dispatch any subprocess')


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
