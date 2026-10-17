"""Genuine disposable crash-WAL producer; never an original-store recovery API.

The future ordinary source owner may allow only the exact source-hashed Python
child argv below. Child stays in that owner's process group, never dispatches a
provider, and exits via os._exit after a real committed DuckDB transaction.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import resource
import re
import selectors
import stat
import subprocess
import sys
import time
import uuid

BASENAME='crash.execution.shard-0002-of-0003.duckdb'
MARKER='fixture-owner.json'
RESULT='committed-fixture.json'
INSTALLATION='duckdb-installation.json'
FAILURE='fixture-failure.json'
MAX_NATIVE_BYTES=128*1024*1024
MAX_SOURCE_BYTES=8*1024*1024
MAX_STREAM_BYTES=65536
CHILD_SECONDS=20.0
SCENARIOS=frozenset({'baseline','extra_relation','oversized_scalar'})
PRODUCER_PHASE='initializing'
PHASE_DIRECTORY=None
PHASE_SEQUENCE=0
PHASE_RECORD='producer-phase.json'


def _phase(value):
    global PRODUCER_PHASE,PHASE_SEQUENCE
    PRODUCER_PHASE=value
    if PHASE_DIRECTORY is None:
        return
    memory={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key=line.partition(':')[0]
        if key in {'VmSize','VmRSS'}:
            pieces=line.split()
            if len(pieces)==3 and pieces[1].isdigit() and pieces[2]=='kB':
                memory[key+'_bytes']=int(pieces[1])*1024
    PHASE_SEQUENCE+=1
    value={'schema':'disposable-crash-wal-producer-phase@1','fixture_only':True,
        'original_store_authority':False,'phase_announced_before_operation':value,
        'sequence':PHASE_SEQUENCE,'child_pid':os.getpid(),'memory':memory,
        'virtual_memory_limit_bytes':resource.getrlimit(resource.RLIMIT_AS)[0]}
    pending=PHASE_DIRECTORY/'producer-phase.pending'
    target=PHASE_DIRECTORY/PHASE_RECORD
    if target.exists():
        info=target.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError('fixture phase record identity')
    _new_file(pending,value)
    os.replace(pending,target)
    directory=os.open(PHASE_DIRECTORY,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
    try:os.fsync(directory)
    finally:os.close(directory)


def _regular_bytes(path, maximum=16*1024*1024):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path:raise ValueError('fixture file path alias')
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0<=before.st_size<=maximum:
            raise ValueError('fixture regular file bound')
        chunks=[];total=0
        while block:=os.read(descriptor,min(65536,maximum-total+1)):
            total+=len(block)
            if total>maximum:raise ValueError('fixture file bytes exceeded')
            chunks.append(block)
        named=path.stat();after=os.fstat(descriptor)
        def selected(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
        if selected(before)!=selected(after) or selected(before)!=selected(named) or total!=before.st_size:
            raise ValueError('fixture file changed during read')
        return b''.join(chunks)
    finally:os.close(descriptor)


def digest(path):return hashlib.sha256(_regular_bytes(path)).hexdigest()
def identity(path):
    info=path.stat()
    return dict(device=info.st_dev,inode=info.st_ino,size=info.st_size,mtime_ns=info.st_mtime_ns)


def _new_file(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        written=0
        while written<len(raw):
            count=os.write(descriptor,raw[written:])
            if count<=0:raise OSError('fixture result short write')
            written+=count
        os.fsync(descriptor)
    finally:os.close(descriptor)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
    try:os.fsync(directory)
    finally:os.close(directory)


def _regular_digest(path, maximum):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path:raise ValueError('fixture digest file alias')
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=maximum:
            raise ValueError('fixture digest file bound')
        value=hashlib.sha256();total=0
        while block:=os.read(descriptor,65536):
            total+=len(block)
            if total>maximum:raise ValueError('fixture digest bytes exceeded')
            value.update(block)
        after=os.fstat(descriptor);named=path.stat()
        def selected(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
        if selected(before)!=selected(after) or selected(before)!=selected(named) or total!=before.st_size:
            raise ValueError('fixture digest changed during read')
        return value.hexdigest()
    finally:os.close(descriptor)


def installed_duckdb_binding():
    """Describe the parent's already loaded genuine installed backend only."""
    modules={name:sys.modules.get(name)for name in ('duckdb','_duckdb')}
    if any(module is None for module in modules.values()):raise ValueError('fixture actual parent DuckDB must be loaded')
    package=Path(importlib.util.find_spec('duckdb').origin).absolute()
    native=Path(importlib.util.find_spec('_duckdb').origin).absolute()
    if any(Path(modules[name].__file__).absolute()!=path for name,path in (('duckdb',package),('_duckdb',native))):
        raise ValueError('fixture parent DuckDB origin mismatch')
    site_root=package.parent.parent
    if package.name!='__init__.py' or package.parent.name!='duckdb' or native.parent!=site_root or not native.name.startswith('_duckdb.') or native.suffix!='.so':
        raise ValueError('fixture installed DuckDB package scope')
    files=sorted(package.parent.rglob('*.py'))
    if not 1<=len(files)<=128:raise ValueError('fixture DuckDB source population bound')
    sources={str(path):_regular_digest(path,MAX_SOURCE_BYTES)for path in files}
    if sum(path.stat().st_size for path in files)>MAX_SOURCE_BYTES:raise ValueError('fixture DuckDB source bytes bound')
    version=getattr(modules['duckdb'],'__version__',None)
    if type(version) is not str or not re.fullmatch(r'[0-9A-Za-z.+_-]{1,64}',version):raise ValueError('fixture installed DuckDB version')
    return {'schema':'disposable-installed-duckdb-binding@1','site_root':str(site_root),
        'package_origin':str(package),'native_origin':str(native),'package_sources':sources,
        'native_sha256':_regular_digest(native,MAX_NATIVE_BYTES),'version':version}


def _load_exact_installed_duckdb(root,binding_sha):
    raw=_regular_bytes(root/INSTALLATION,65536)
    if hashlib.sha256(raw).hexdigest()!=binding_sha:raise ValueError('fixture installed DuckDB binding pin')
    value=json.loads(raw)
    if set(value)!={'schema','site_root','package_origin','native_origin','package_sources','native_sha256','version'} or value['schema']!='disposable-installed-duckdb-binding@1':
        raise ValueError('fixture installed DuckDB binding shape')
    site_root=Path(value['site_root']);package=Path(value['package_origin']);native=Path(value['native_origin'])
    if not site_root.is_absolute() or site_root.resolve()!=site_root or package!=site_root/'duckdb/__init__.py' or native.parent!=site_root or not native.name.startswith('_duckdb.') or native.suffix!='.so':
        raise ValueError('fixture installed DuckDB binding paths')
    sources=value['package_sources']
    if type(sources) is not dict or not 1<=len(sources)<=128 or str(package) not in sources:
        raise ValueError('fixture installed DuckDB source population')
    actual={str(path)for path in package.parent.rglob('*.py')}
    if set(sources)!=actual:raise ValueError('fixture installed DuckDB source population drift')
    total=0
    for name,wanted in sources.items():
        path=Path(name)
        if not path.is_relative_to(package.parent) or path.suffix!='.py' or re.fullmatch('[0-9a-f]{64}',wanted) is None:
            raise ValueError('fixture installed DuckDB source pin shape')
        if _regular_digest(path,MAX_SOURCE_BYTES)!=wanted:raise ValueError('fixture installed DuckDB source changed')
        total+=path.stat().st_size
    if total>MAX_SOURCE_BYTES or _regular_digest(native,MAX_NATIVE_BYTES)!=value['native_sha256']:
        raise ValueError('fixture installed DuckDB binary or aggregate bound')
    # Isolated Python remains enabled. Inserting this verified root does NOT
    # execute user-site .pth files or select any alternate DuckDB installation.
    # A fresh cache prefix plus -B avoids reading existing package bytecode.
    sys.pycache_prefix=str(root/'unwritten-isolated-pycache')
    sys.path.insert(0,str(site_root))
    import duckdb
    import _duckdb
    if Path(duckdb.__file__).absolute()!=package or Path(_duckdb.__file__).absolute()!=native or duckdb.__version__!=value['version']:
        raise ValueError('fixture loaded DuckDB origin or version mismatch')
    for name,module in tuple(sys.modules.items()):
        if name=='duckdb' or name.startswith('duckdb.'):
            origin=getattr(module,'__file__',None)
            if origin and str(Path(origin).absolute()) not in sources:raise ValueError('fixture loaded DuckDB source not bound')
    if _regular_digest(native,MAX_NATIVE_BYTES)!=value['native_sha256']:
        raise ValueError('fixture loaded DuckDB binary changed')
    return duckdb,value


def fixture_argv(root,producer_source,producer_sha,marker_sha,scenario='baseline',*,duckdb_binding_sha):
    return (sys.executable,'-I','-B',str(Path(__file__).absolute()),'--produce',
        '--fixture-root',str(root),'--producer-source',str(producer_source),
        '--producer-source-sha256',producer_sha,'--marker-sha256',marker_sha,'--scenario',scenario,'--duckdb-binding-sha256',duckdb_binding_sha)


def create_crash_fixture(root,producer_source,scenario='baseline'):
    """Run one exact actual child; caller's source owner retains all custody."""
    root=Path(root);producer_source=Path(producer_source)
    if type(scenario) is not str or scenario not in SCENARIOS:raise ValueError('fixture scenario closed set')
    if not root.is_absolute() or root.parent.resolve()!=root.parent or root.name in {'.','..'}:
        raise ValueError('fixture directory must be new absolute nonaliased child')
    if not producer_source.is_absolute() or producer_source.name!='implementation_daemon.py':
        raise ValueError('fixture producer source scope')
    root.mkdir(mode=0o700)
    installed=installed_duckdb_binding()
    _new_file(root/INSTALLATION,installed)
    installation_sha=digest(root/INSTALLATION)
    _new_file(root/MARKER,{'schema':'disposable-crash-wal-fixture@1','fixture_only':True,
        'original_store_authority':False,'nonce':uuid.uuid4().hex,'uid':os.geteuid(),'scenario':scenario,'duckdb_binding_sha256':installation_sha})
    source_sha=digest(producer_source);marker_sha=digest(root/MARKER)
    argv=fixture_argv(root,producer_source,source_sha,marker_sha,scenario,duckdb_binding_sha=installation_sha)
    environment=dict(os.environ)
    environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',IPFS_DATASETS_AUTO_INSTALL='0',IPFS_AUTO_INSTALL='0')
    # No new session/group: a failed outer unit owner drains this child too.
    process=subprocess.Popen(argv,cwd=root,env=environment,stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,start_new_session=False)
    streams=[bytearray(),bytearray()];started=time.monotonic();reaped=False
    try:
        if os.getpgid(process.pid)!=os.getpgrp():raise AssertionError('fixture child escaped owner process group')
        with selectors.DefaultSelector() as selector:
            for pipe,destination in zip((process.stdout,process.stderr),streams):
                os.set_blocking(pipe.fileno(),False);selector.register(pipe,selectors.EVENT_READ,destination)
            while selector.get_map():
                remaining=CHILD_SECONDS-(time.monotonic()-started)
                if remaining<=0:raise TimeoutError('disposable crash fixture deadline')
                for key,_events in selector.select(min(0.2,remaining)):
                    block=os.read(key.fd,65536)
                    if not block:selector.unregister(key.fileobj);key.fileobj.close();continue
                    key.data.extend(block)
                    if len(key.data)>MAX_STREAM_BYTES:raise AssertionError('disposable fixture output bound')
        code=process.wait(timeout=max(0.01,CHILD_SECONDS-(time.monotonic()-started)));reaped=True
        if code!=0:
            try:closed_type=json.loads(bytes(streams[1])).get('error_type')
            except (ValueError,AttributeError):closed_type=None
            if type(closed_type) is not str or re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,127}',closed_type) is None:closed_type='ChildExitFailure'
            raise AssertionError('disposable fixture producer failed: '+closed_type)
        path=root/BASENAME;wal=Path(str(path)+'.wal')
        result=json.loads(_regular_bytes(root/RESULT,16384))
        assert result['fixture_only'] is True and result['committed_before_os_exit'] is True
        assert result['producer_source_sha256']==source_sha and result['child_pid']==process.pid
        assert result['scenario']==scenario and result['duckdb_binding_sha256']==installation_sha
        assert result['duckdb_package_origin']==installed['package_origin'] and result['duckdb_native_origin']==installed['native_origin']
        assert result['duckdb_version']==installed['version']
        assert wal.is_file() and 0<wal.stat().st_size<=16*1024*1024
        assert path.is_file() and path.stat().st_size<=16*1024*1024
        return {'root':root,'path':path,'wal':wal,'result':result,'producer_source':producer_source,
            'producer_source_sha256':source_sha,'duckdb_installation_binding':installed,'argv':argv,'child_reaped':True,
            'child_pid':process.pid,'owned_process_group':os.getpgrp(),'stdout_bytes':len(streams[0]),'stderr_bytes':len(streams[1])}
    except BaseException as error:
        # Retain bounded private diagnostics; never print raw child output.
        _new_file(root/FAILURE,{'schema':'disposable-crash-wal-fixture-failure@1',
            'fixture_only':True,'original_store_authority':False,'error_type':type(error).__name__,
            'child_returncode':process.poll(),'child_reaped_at_capture':reaped,
            'stderr_text':bytes(streams[1]).decode('utf-8','replace'),
            'stdout_text':bytes(streams[0]).decode('utf-8','replace'),
            'stderr_bytes':len(streams[1]),'stdout_bytes':len(streams[0]),
            'producer_source_sha256':source_sha,'duckdb_binding_sha256':installation_sha})
        raise
    finally:
        if not reaped:
            process.kill();process.wait(timeout=5)
        for pipe in (process.stdout,process.stderr):
            if pipe is not None and not pipe.closed:pipe.close()


def _producer_ddl(path,wanted):
    if not path.is_absolute() or path.name!='implementation_daemon.py':raise ValueError('fixture producer source scope')
    if re.fullmatch(r'[0-9a-f]{64}',wanted) is None:raise ValueError('fixture producer pin shape')
    raw=_regular_bytes(path,MAX_SOURCE_BYTES)
    if hashlib.sha256(raw).hexdigest()!=wanted:raise ValueError('fixture producer source pin')
    tree=ast.parse(raw.decode())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_DAEMON_EXECUTION_SQL'for target in node.targets):
            ddl=ast.literal_eval(node.value)
            if type(ddl) is not str or not ddl:raise ValueError('fixture DDL not constant text')
            return ddl
    raise ValueError('actual producer DDL absent')


def produce(args):
    # Fixed source-fixture limits, before importing DuckDB or parsing producer.
    resource.setrlimit(resource.RLIMIT_CPU,(15,15))
    resource.setrlimit(resource.RLIMIT_AS,(4*1024*1024*1024,4*1024*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024*1024,16*1024*1024))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    _phase('validate_new_fixture_root')
    root=args.fixture_root
    if not root.is_absolute() or root.resolve()!=root or stat.S_IMODE(root.stat().st_mode)!=0o700 or root.stat().st_uid!=os.geteuid():
        raise ValueError('fixture root is not newly owned private directory')
    if sorted(path.name for path in root.iterdir())!=sorted([INSTALLATION,MARKER]) or digest(root/MARKER)!=args.marker_sha256:
        raise ValueError('fixture root is not empty marked source fixture')
    marker=json.loads(_regular_bytes(root/MARKER,4096))
    if (set(marker)!={'schema','fixture_only','original_store_authority','nonce','uid','scenario','duckdb_binding_sha256'}
            or marker.get('schema')!='disposable-crash-wal-fixture@1' or marker.get('fixture_only') is not True
            or marker.get('original_store_authority') is not False or type(marker.get('uid')) is not int
            or marker['uid']!=os.geteuid() or marker.get('duckdb_binding_sha256')!=args.duckdb_binding_sha256 or marker.get('scenario')!=args.scenario or args.scenario not in SCENARIOS
            or re.fullmatch(r'[0-9a-f]{32}',str(marker.get('nonce'))) is None):
        raise ValueError('fixture marker scope')
    global PHASE_DIRECTORY
    PHASE_DIRECTORY=root
    _phase('fixture_root_and_markers_validated')
    _phase('read_actual_producer_ddl')
    ddl=_producer_ddl(args.producer_source,args.producer_source_sha256)
    path=root/BASENAME
    import fcntl
    _phase('create_new_fixture_locks')
    descriptors=[]
    for suffix in ('.writer.lock','.lock'):
        descriptor=os.open(root/('.'+BASENAME+suffix),os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB);descriptors.append(descriptor)
    _phase('verify_and_load_installed_duckdb')
    duckdb,installed=_load_exact_installed_duckdb(root,args.duckdb_binding_sha256)
    _phase('connect_new_fixture_database')
    connection=duckdb.connect(str(path),config={'threads':'1','memory_limit':'256MB',
        'autoinstall_known_extensions':'false','autoload_known_extensions':'false',
        'enable_external_access':'false','allow_unsigned_extensions':'false','lock_configuration':'true'})
    _phase('execute_actual_execution_ddl')
    connection.execute(ddl)
    metadata={'interface':'DatabaseImplementationDaemon@1',
        'schema':'ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1',
        'authority_mode':'quack','logical_owner_session_id':'embedded-store:crash-fixture',
        'process_instance_id':'process:fixture-before-wal','state_schema_revision':'',
        'control_schema_profile_id':'fixture-profile','control_schema_fingerprint':'',
        'execution_state_dir':str(root),'execution_state_prefix':'crash','task_shard_count':'3',
        'task_shard_index':'2','strict_task_sharding':'true'}
    _phase('insert_baseline_metadata')
    connection.executemany('INSERT INTO daemon_execution_metadata VALUES (?,?)',list(metadata.items()))
    _phase('checkpoint_baseline')
    connection.execute('CHECKPOINT')
    _phase('begin_wal_transaction')
    connection.execute('BEGIN TRANSACTION')
    metadata['process_instance_id']='process:fixture-wal-committed'
    _phase('commit_new_metadata_in_wal')
    connection.execute("UPDATE daemon_execution_metadata SET value=? WHERE key='process_instance_id'",[metadata['process_instance_id']])
    _phase('insert_running_attempt')
    connection.execute('INSERT INTO database_task_attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',[
        'attempt:crash-fixture2','claim:crash-fixture2','task:crash-fixture','FIXTURE-WAL',2,
        'embedded-store:crash-fixture',2,2,'lease:crash-fixture2','context','running',1000,None,2,'{"feature_score":0.75}'])
    _phase('insert_attempt_phases')
    connection.executemany('INSERT INTO attempt_phases VALUES (?,?,?,?,?,?,?)',[
        ['attempt:crash-fixture2','claimed',1000,2,2,1,'{}'],['attempt:crash-fixture2','context',1001,2,2,2,'{}']])
    unknown={'schema':'database-portal-callback-intent@1','callback_state':'started_outcome_unknown',
        'provider_effect_state':'unknown_may_have_started','fixture_only':True}
    _phase('insert_unknown_provider_outcome')
    connection.execute('INSERT INTO provider_invocations VALUES (?,?,?,?,?,?,?)',[
        'invocation:crash-fixture','attempt:crash-fixture2','task:crash-fixture','provider:attempt:crash-fixture2',
        'embedded-store:crash-fixture',1002,json.dumps(unknown,sort_keys=True)])
    _phase('insert_fixture_event')
    connection.execute('INSERT INTO daemon_execution_events VALUES (?,?,?,?,?,?)',[
        'event:crash-fixture','attempt:crash-fixture2','task:crash-fixture','fixture_committed_unknown',1003,'{}'])
    _phase('insert_closed_scenario_material')
    if args.scenario=='extra_relation':
        connection.execute('CREATE TABLE fixture_extra_relation (value INTEGER)')
        connection.execute('INSERT INTO fixture_extra_relation VALUES (1)')
    elif args.scenario=='oversized_scalar':
        connection.execute('UPDATE database_task_attempts SET body_json=?',[json.dumps({'padding':'x'*16385})])
    _phase('commit_real_wal_transaction')
    connection.execute('COMMIT')
    wal=Path(str(path)+'.wal')
    if not wal.is_file() or wal.stat().st_size<=0:raise ValueError('real committed fixture WAL absent')
    _phase('fsync_real_committed_wal')
    wal_descriptor=os.open(wal,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:os.fsync(wal_descriptor)
    finally:os.close(wal_descriptor)
    _phase('write_committed_fixture_result')
    _new_file(root/RESULT,{'schema':'native-crash-wal-fixture-result@1','fixture_only':True,
        'original_store_authority':False,'committed_before_os_exit':True,'metadata':metadata,
        'provider_callback':unknown,'producer_source_sha256':args.producer_source_sha256,
        'child_pid':os.getpid(),'wal_bytes':wal.stat().st_size,'scenario':args.scenario,
        'duckdb_binding_sha256':args.duckdb_binding_sha256,'duckdb_package_origin':installed['package_origin'],
        'duckdb_native_origin':installed['native_origin'],'duckdb_version':duckdb.__version__})
    # Intentionally bypass connection.close and all atexit handlers. The WAL is
    # real committed native state; parent must reap the child before maintenance.
    _phase('exit_without_native_connection_close')
    os._exit(0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--produce',action='store_true',required=True)
    parser.add_argument('--fixture-root',type=Path,required=True)
    parser.add_argument('--producer-source',type=Path,required=True)
    parser.add_argument('--producer-source-sha256',required=True)
    parser.add_argument('--marker-sha256',required=True)
    parser.add_argument('--scenario',choices=sorted(SCENARIOS),required=True)
    parser.add_argument('--duckdb-binding-sha256',required=True)
    args=parser.parse_args()
    try:produce(args)
    except Exception as exc:
        # Only synthetic fixture paths/SQL are in scope. Retain the exact bounded
        # native reason privately; never inspect an original/authentication store.
        message=str(exc).encode('utf-8','replace')[:4096].decode('utf-8','ignore')
        sys.stderr.write(json.dumps({'fixture_failed':True,'error_type':type(exc).__name__,
            'producer_phase':PRODUCER_PHASE,'error_message':message})+'\n')
        sys.stderr.flush();os._exit(2)


if __name__=='__main__':main()
