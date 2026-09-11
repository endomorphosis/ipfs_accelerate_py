"""Disposable actual processes, pidfds, queue/task locks and opaque originals.

Fast tests double only the unit/cgroup and native task/source observations.
The separate host and accepted-native qualification exercise those boundaries.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.ops.agent_supervisor import spar_stopped_capture as stopped
from scripts.ops.agent_supervisor import spar_stopped_origin as origin
from scripts.ops.agent_supervisor import spar_legacy_origin as legacy_origin
from scripts.ops.agent_supervisor import spar_merge_owner as role
from ipfs_accelerate_py.agent_supervisor.task_sources.closeout_snapshot import RELATIONS
from test.api.semantic_refactoring.test_spar_legacy_capture import armed as armed_fixture
from test.api.semantic_refactoring.test_spar_retained_capture_driver import fleet_fixture

armed = armed_fixture

CONTROLLER = r'''import fcntl,json,os,struct,subprocess,sys
paths=json.loads(sys.argv[1]);fds=[]
for path,ofd in paths:
 fd=os.open(path,os.O_RDWR if ofd else os.O_RDONLY);fds.append(fd)
 if ofd: fcntl.fcntl(fd,fcntl.F_OFD_SETLK,struct.pack('hhqqi',fcntl.F_WRLCK,os.SEEK_SET,0,0,0))
 else: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
helper=subprocess.Popen(['/usr/bin/python3','-I','-S','-c',sys.argv[2]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd='/',close_fds=True)
assert helper.stdout.readline()==b'SPAR-CAPTURE-SENTINEL-READY\n'
print(json.dumps({'helper':helper.pid}),flush=True)
sys.stdin.buffer.read(1)
helper.stdin.close();helper.wait(timeout=5)
'''


def observation(source, bootstrap, owner):
    identity = owner['identity']
    plan_body = {"plan_cid": bootstrap['plan_root_cid'], "source_head": bootstrap['source_head'],
                 "repository_tree_id": bootstrap['repository_tree_id']}
    plan = {"plan_cid": bootstrap['plan_root_cid'], "status": "active", "revision": 1, "body": plan_body}
    facts = {"all_relations_available": True, "truncated": False,
             "relations": {name: {"available": True, "truncated": False, "rows": []} for name in RELATIONS}}
    facts['relations']['tasks']['rows'] = [{"task_cid": cid} for cid in bootstrap['database_task_source_receipt']['task_cids']]
    facts['relations']['effect_claims']['rows'] = [{"effect_id": "unknown-preserved", "state": None}]
    return {"schema": stopped.TASK_SCHEMA, "source": source, "bootstrap": bootstrap,
        "store_generations": [{**{k: identity[k] for k in ('generation','schema_revision','fence_epoch','database_uuid')}, 'birth_id': identity['process_birth_id']}],
        "state_servers": [{**{k: identity[k] for k in ('server_id','store_id','database_uuid','process_birth_id','schema_revision','generation')}, 'status':'stopped','stopped_at':'2026-09-11T12:00:00Z'}],
        "server_epochs": [{'server_id':identity['server_id'],'epoch':identity['startup_epoch'],'fence_epoch':identity['fence_epoch'],'ended_at':'2026-09-11T12:00:00Z'}],
        "plan":plan,"plan_revisions":[plan],"task_cids":bootstrap['database_task_source_receipt']['task_cids'],
        "facts":facts,"completion_authority":False,"callback_settled":False,
        "launch_amendment_admitted":False,"canonical_database_opened":False}


@pytest.fixture
def stopped_fixture(armed, monkeypatch, request):
    host = bool(getattr(request, "param", False))
    if host and os.environ.get("SPAR_RUN_HOST_CAPTURE_TEST") != "1":
        pytest.skip("explicit disposable user-systemd qualification")
    monkeypatch.setattr(stopped, 'probe_task_runtime', lambda value: {
        'source':value.source,'canonical_database_opened':False,'runtime':{'explicit_fixture':True}})
    actual_unit_snapshot = stopped.workflow.unit_snapshot
    actual_population = stopped.native._cgroup_population
    host_unit = None
    drops = None
    armed.close_native(); armed.session.close()
    operator = stopped.native._native_operator(armed.tmp)
    database = armed.tmp / 'control.duckdb'
    with role.open_duckdb_connection(database, prefer_quack=False) as connection:
        connection.execute('CREATE TABLE preserved (value BLOB)')
        connection.execute('INSERT INTO preserved VALUES (?)', [b'unknown-task-content'])
    for suffix in ('.lock','.state-owner.lock','.intent.lock','.migration.lock'):
        database.with_name('.'+database.name+suffix).touch()
    old_paths = operator._runtime_paths(operator._load_config(None)[0])
    operator._runtime_paths = lambda board: {**old_paths,'database':database}
    bootstrap = json.loads(old_paths['bootstrap_receipt'].read_bytes())
    bootstrap.pop('bootstrap_receipt_id')
    bootstrap.update(source_head=armed.source['head'], repository_tree_id=armed.source['tree'],
                     database_task_source_receipt={'task_cids':['fixture-task']})
    bootstrap['bootstrap_receipt_id']=role._cid(bootstrap)
    old_paths['bootstrap_receipt'].write_text(json.dumps(bootstrap))
    owner = json.loads(armed.owner_path.read_bytes())
    owner['identity'].update(status='stopped',schema_revision=3,startup_epoch=1789017867)
    armed.owner_path.write_text(json.dumps(owner))
    if host:
        host_unit = f"spar-stopped-qualification-{os.getpid()}-{time.time_ns()}.service"
        monkeypatch.setattr(stopped.native, "UNIT", host_unit)
    fleet = fleet_fixture(armed.tmp, monkeypatch, armed.tmp, armed.tmp/'HOLD')
    paths = [(str(fleet.state/'repairs/spar/queue.lock'),False),(str(fleet.state/'spar/watchdog.lock'),False)]
    paths += [(str(armed.queue/name),ofd) for name,ofd in (
        ('.merge_queue.duckdb.rebuild.lock',False),('.merge_queue.duckdb.lock',False),
        ('merge_queue.duckdb',True),('train/consumer.lock',False))]
    argv=['/usr/bin/python3','-I','-S','-c',CONTROLLER,json.dumps(paths),stopped.native.SENTINEL_CODE]
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd='/')
    helper=json.loads(process.stdout.readline())['helper']
    cgroup=armed.tmp/'disposable-test';cgroup.mkdir()
    monkeypatch.setattr(stopped,'CGROUP_ROOT',armed.tmp)
    armed.unit.update(MainPID='0',ActiveState='active',SubState='running',Delegate='yes',ExitType='cgroup')
    monkeypatch.setattr(stopped.workflow,'unit_snapshot',lambda _:copy.deepcopy(armed.unit))
    monkeypatch.setattr(stopped,'_files',lambda _:{'fixture-unit':'a'*64})
    members=[helper]
    def population(_):
        alive=[]
        for pid in members:
            try:fd=os.pidfd_open(pid)
            except ProcessLookupError:continue
            try:
                if not stopped.native._exited(fd): alive.append(pid)
            finally: os.close(fd)
        return alive, '1' if alive else '0'
    monkeypatch.setattr(stopped.native,'_cgroup_population',population)
    original_run=subprocess.run
    def run(argv,**kwargs):
        if argv[0]=='busctl':
            members.append(int(argv[-1]));return SimpleNamespace(returncode=0)
        return original_run(argv,**kwargs)
    monkeypatch.setattr(subprocess,'run',run)
    def fixture_observation(succession,path,*,task_files):
        value=observation(succession.source,succession.bootstrap,succession.owner)
        value['input_binding']=[{'file':entry,'device':1,'inode':i+1} for i,entry in enumerate(task_files)]
        return value
    monkeypatch.setattr(stopped,'observe_task_copy',fixture_observation)
    if host:
        monkeypatch.setattr(subprocess, 'run', original_run)
        monkeypatch.setattr(stopped.workflow, 'unit_snapshot', actual_unit_snapshot)
        monkeypatch.setattr(stopped.native, '_cgroup_population', actual_population)
        # Restore the real unit-file observer while keeping native task/source
        # admission explicitly supplied by the disposable public fixture.
        def actual_files(unit):
            fragment=stopped.workflow._property(unit,'Unit','FragmentPath','s')
            dropins=stopped.workflow._property(unit,'Unit','DropInPaths','as')
            return {path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [fragment,*dropins]}
        monkeypatch.setattr(stopped, '_files', actual_files)
        monkeypatch.setattr(stopped, 'CGROUP_ROOT', Path('/sys/fs/cgroup'))
        def host_run(argv):return original_run(argv,capture_output=True,check=True,timeout=15)
        host_run(['systemd-run','--user','--unit='+host_unit[:-8], '--property=Type=exec',
            '--property=WorkingDirectory='+str(armed.tmp),'--property=Delegate=yes','--property=ExitType=cgroup',
            '--property=Restart=no','--property=SendSIGKILL=no','--property=TimeoutStopSec=infinity',
            '/usr/bin/python3','-I','-S','-c','import time;time.sleep(240)'])
        host_run(['busctl','--user','call','org.freedesktop.systemd1','/org/freedesktop/systemd1',
            'org.freedesktop.systemd1.Manager','AttachProcessesToUnit','ssau',host_unit,'','1',str(helper)])
        mainpid=int(actual_unit_snapshot(host_unit)['MainPID'])
        mainfd=os.pidfd_open(mainpid)
        try:
            import signal
            signal.pidfd_send_signal(mainfd,signal.SIGTERM)
            stopped._wait_pidfd(mainfd,10)
        finally:os.close(mainfd)
        drops=Path(f'/run/user/{os.geteuid()}/systemd/user')/(host_unit+'.d')
        drops.mkdir(parents=True,exist_ok=False)
        (drops/'99-stopped-qualification.conf').write_text('[Unit]\nRefuseManualStart=yes\nConditionPathExists=!'+str(armed.tmp/'HOLD')+'\n')
        host_run(['systemctl','--user','daemon-reload'])
        deadline=time.monotonic()+10
        while actual_unit_snapshot(host_unit)['MainPID']!='0' and time.monotonic()<deadline:time.sleep(.02)
    value=stopped.KeeperSuccession(repository_root=armed.tmp,config_path=armed.tmp/'config.json',
        controller_birth=stopped._birth(process.pid),helper_birth=stopped._birth(helper),controller_argv=argv,
        hold_sha256=hashlib.sha256((armed.tmp/'HOLD').read_bytes()).hexdigest(),
        source_head=armed.source['head'],source_tree=armed.source['tree'],
        journal_root=armed.tmp.parent/(armed.tmp.name+'-succession'),fleet_config=fleet.config,
        task_runtime={'manifest_path':'fixture-only','manifest_sha256':'a'*64,'helper_sha256':'b'*64},
        owner_identity={key:owner['identity'][key] for key in stopped.native.IDENTITY_FIELDS})
    current=SimpleNamespace(armed=armed,succession=value,process=process,helper=helper,fleet=fleet,database=database,
                            session=None,output=armed.tmp.parent/(armed.tmp.name+'-capture'))
    try:
        yield current
    finally:
        if current.session is not None:
            current.session.resources.close();current.session._closed=True
        if process.poll() is None: process.stdin.close()
        process.wait(timeout=10)
        if value.keeper is not None:
            if value.keeper.poll() is None: value.keeper.stdin.close()
            value.keeper.wait(timeout=10)
        value.resources.close()
        for stream in (process.stdin,process.stdout,process.stderr):
            if not stream.closed:stream.close()
        if host_unit:
            (drops/'99-stopped-qualification.conf').unlink()
            drops.rmdir()
            original_run(['systemctl','--user','daemon-reload'],capture_output=True,check=True,timeout=15)
            original_run(['systemctl','--user','stop',host_unit],capture_output=True,timeout=15)
            assert not drops.exists()


def retire(f):
    old_fd=os.pidfd_open(f.process.pid)
    try:
        f.succession.overlap()
        assert not stopped.native._exited(old_fd)
        result=f.succession.retire()
        f.process.wait(timeout=10)
        assert stopped.native._exited(old_fd)
        assert result['task_or_store_authority'] is False
    finally: os.close(old_fd)
    return result


def capture(f):
    retire(f)
    f.session=stopped.StoppedCaptureSession(f.succession,fleet_config=f.fleet.config)
    return f.session.capture(destination=f.output)


def test_real_pidfd_succession_fresh_fences_complete_preservation(stopped_fixture):
    f=stopped_fixture
    queue={entry['path']:entry['sha256'] for entry in f.succession.queue_entries}
    task=f.database.read_bytes()
    captured=capture(f)
    assert f.succession.closed_gate()['observed_cgroup_members']==[f.succession.keeper.pid]
    assert len(f.session.locks)==9
    assert {entry['path']:entry['sha256'] for entry in captured.receipt['manifest']['files']}==queue
    assert f.database.read_bytes()==(captured.task_copy/'control.duckdb').read_bytes()==task
    assert captured.receipt['task_admission']['observation']['facts']['relations']['effect_claims']['rows']==[
        {'effect_id':'unknown-preserved','state':None}]
    assert captured.require_current()['completion_authority'] is False
    for path in (f.database,f.armed.queue/'merge_queue.duckdb'):
        denied=subprocess.run([sys.executable,'-c','import duckdb,sys;duckdb.connect(sys.argv[1])',str(path)],capture_output=True,timeout=10)
        assert denied.returncode!=0 and b'lock' in denied.stderr.lower()


@pytest.mark.parametrize('kind',['task','queue','consumer'])
def test_post_handover_competing_lock_denies_admission(stopped_fixture,kind):
    f=stopped_fixture;retire(f)
    path=f.database if kind=='task' else f.armed.queue/('merge_queue.duckdb' if kind=='queue' else 'train/consumer.lock')
    code=("import duckdb,sys;c=duckdb.connect(sys.argv[1]);print('ready',flush=True);sys.stdin.read(1)"
        if kind!='consumer' else "import fcntl,sys;f=open(sys.argv[1]);fcntl.flock(f,fcntl.LOCK_EX);print('ready',flush=True);sys.stdin.read(1)")
    process=subprocess.Popen([sys.executable,'-c',code,str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        assert process.stdout.readline()==b'ready\n'
        with pytest.raises((role.SparMergeOwnerError,BlockingIOError)):
            stopped.StoppedCaptureSession(f.succession,fleet_config=f.fleet.config)
        assert process.poll() is None and f.succession.keeper.poll() is None and not f.output.exists()
    finally:
        process.stdin.close();process.wait(timeout=10);process.stdout.close()


@pytest.mark.parametrize('change',['cursor','opaque','source','hold','unit','task-wal'])
def test_handover_or_capture_drift_denies_without_install(stopped_fixture,change):
    f=stopped_fixture;retire(f)
    if change=='cursor': f.armed.receipt_path.write_bytes(b'cursor-replaced')
    elif change=='opaque': (f.armed.queue/'private/callback-signing-material').write_bytes(b'changed')
    elif change=='source': f.armed.source['head']='f'*40
    elif change=='hold': (f.armed.tmp/'HOLD').write_text('changed')
    elif change=='unit': f.armed.unit['Restart']='on-failure'
    else:
        f.session=stopped.StoppedCaptureSession(f.succession,fleet_config=f.fleet.config)
        captured=f.session.capture(destination=f.output)
        f.database.with_suffix('.duckdb.wal').write_bytes(b'unknown-new-wal')
        with pytest.raises(role.SparMergeOwnerError):captured.require_current()
        return
    with pytest.raises(role.SparMergeOwnerError):
        stopped.StoppedCaptureSession(f.succession,fleet_config=f.fleet.config)
    assert not (f.armed.queue/origin.REQUIRED_MARKER).exists()


def test_stopped_and_live_capabilities_are_not_interchangeable(stopped_fixture):
    f=stopped_fixture;captured=capture(f)
    prepared=role.prepare_offline_clone(offline_root=captured.path,destination=f.output/'prepared',manifest=captured.receipt['manifest'])
    with pytest.raises(role.SparMergeOwnerError):legacy_origin.install_captured_queue(captured,prepared)
    with pytest.raises(role.SparMergeOwnerError):origin.install_stopped_queue(captured.receipt,prepared)
    forged=stopped.StoppedQueueCapture(f.session,captured.path,captured.receipt,captured._identities,captured.task_copy,captured.task_files)
    with pytest.raises(role.SparMergeOwnerError):forged.require_current()
    result=origin.install_stopped_queue(captured,prepared)
    assert result['installed'] is True and result['completion_authority'] is False
    f.session.resources.close();f.session._closed=True
    loaded=origin.load_stopped_origin(f.armed.queue/'merge_queue.duckdb',repository_id=f.armed.context['repository_id'],
        target_branch=f.armed.context['target_branch'],store_id=f.armed.context['store_id'],scopes=captured.receipt['manifest']['scope_bindings'])
    assert loaded.database_uuid==prepared.database_uuid
    f.succession.close_after_install()


@pytest.mark.parametrize('change',['generation','epoch','server','leases','truncation','plan','bootstrap','task-population'])
def test_closed_native_facts_must_match_exact_store_and_full_lineage(stopped_fixture,change):
    f=stopped_fixture
    value=observation(f.succession.source,f.succession.bootstrap,f.succession.owner)
    if change=='generation':value['store_generations'][0]['generation']+=1
    elif change=='epoch':value['server_epochs'][0]['ended_at']=None
    elif change=='server':value['state_servers'][0]['status']='running'
    elif change=='leases':value['facts']['relations']['leases']['rows']=[{'task_cid':'fixture-task','state':None}]
    elif change=='truncation':value['facts']['relations']['tasks']['truncated']=True
    elif change=='plan':value['plan']['body']['source_head']='f'*40
    elif change=='bootstrap':value['bootstrap']=dict(value['bootstrap'],source_head='f'*40)
    else:value['task_cids']=[]
    value['input_binding']=[{'file':{'path':'control.duckdb','size_bytes':1,'sha256':'a'*64},'device':1,'inode':1}]
    with pytest.raises(role.SparMergeOwnerError):
        stopped.validate_task_observation(value,owner=f.succession.owner,bootstrap=f.succession.bootstrap,
            source=f.succession.source,task_files=[{'path':'control.duckdb','size_bytes':1,'sha256':'a'*64}])


@pytest.mark.parametrize('requested',[None,'','native-fresh-origin@1','native-legacy-capture@1'])
def test_stopped_requirement_rejects_profile_bypass(tmp_path,requested):
    (tmp_path/origin.REQUIRED_MARKER).touch()
    if requested in (None,''):assert origin.required_profile(tmp_path,requested)==origin.STOPPED_PROFILE
    else:
        with pytest.raises(role.SparMergeOwnerError):origin.required_profile(tmp_path,requested)
    (tmp_path/legacy_origin.REQUIRED_MARKER).touch()
    with pytest.raises(role.SparMergeOwnerError):origin.required_profile(tmp_path,requested)


@pytest.mark.parametrize('stopped_fixture',[True],indirect=True)
@pytest.mark.parametrize('scenario',['capture-install','keeper-death'])
def test_actual_host_stopped_keeper_succession(stopped_fixture,scenario):
    f=stopped_fixture
    before=f.succession.unit_files
    if scenario=='keeper-death':
        retire(f)
        f.succession.keeper.stdin.close();f.succession.keeper.wait(timeout=10)
        with pytest.raises((role.SparMergeOwnerError,stopped.workflow.CaptureDriverDenied)):
            stopped.StoppedCaptureSession(f.succession,fleet_config=f.fleet.config)
        assert not f.output.exists()
    else:
        test_stopped_and_live_capabilities_are_not_interchangeable(f)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in before.items() if Path(path).exists())
    print('STOPPED_HOST_QUALIFICATION='+json.dumps({'scenario':scenario,'unit':stopped.native.UNIT,
        'old_controller':f.succession.controller_birth,'old_helper':f.succession.helper_birth,
        'new_keeper':f.succession.keeper_birth,'old_controller_exit':True,'keeper_exit':True,
        'unit_files_unchanged':True,'native_task_source_observation':'explicit isolated fixture only'}))


ISOLATED_DRIVER = '''import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
from scripts.ops.agent_supervisor import spar_stopped_capture as s
from scripts.ops.agent_supervisor import spar_stopped_capture_driver as d
value=json.loads(Path(sys.argv[2]).read_text());root=Path(value['root'])
board=SimpleNamespace(**value['board']);board.path=lambda name:Path(name)
operator=SimpleNamespace(_load_config=lambda path:(board,{}),
 _runtime_paths=lambda board:{name:Path(path) for name,path in value['paths'].items()},
 _identity=s.role._cid,source_binding=lambda config:value['source'])
s.native.UNIT=value['unit'];s.native._native_operator=lambda root:operator
s.native._repository_id=lambda root:'repo:spar'
s.workflow._job_unit=lambda:value['repair_idle']
s.probe_task_runtime=lambda succession:{'source':value['source'],'runtime':{'explicit_fixture':True},'canonical_database_opened':False}
def observed(succession,path,*,task_files):
 result=dict(value['observation']);result['input_binding']=[{'file':entry,'device':1,'inode':i+1} for i,entry in enumerate(task_files)];return result
s.observe_task_copy=observed
if value['failure']:
 original=s.producer.produce_offline_import_plan;count=[]
 def injected(**kwargs):
  result=original(**kwargs);count.append(1)
  if len(count)==1:raise ModuleNotFoundError('fixture-only',name='duckdb')
  return result
 s.producer.produce_offline_import_plan=injected
assert sys.flags.isolated==1
sys.argv=['stopped-driver','--request',value['request_path'],'--request-sha256',value['request_sha256']]
raise SystemExit(d.main())
'''


@pytest.mark.parametrize('stopped_fixture',[True],indirect=True)
@pytest.mark.parametrize('retry',[False,True])
def test_exact_isolated_stopped_driver_retains_authority_on_retry(stopped_fixture,retry):
    from scripts.ops.agent_supervisor import spar_capture_runtime as dependencies
    from scripts.ops.agent_supervisor import spar_stopped_capture_driver as driver
    from test.api.semantic_refactoring.test_spar_capture_isolated_host import _reply
    f=stopped_fixture;s=f.succession
    manifest=f.output.with_name(f.output.name+'-runtime.json')
    manifest.write_text(json.dumps(dependencies.observed_runtime()))
    helper_sha=hashlib.sha256(Path(dependencies.__file__).read_bytes()).hexdigest()
    request={'schema':driver.REQUEST_SCHEMA,'repository_root':str(s.root),'config_path':str(s.config),
        'fleet_config':str(f.fleet.config),'controller_birth':s.controller_birth,'helper_birth':s.helper_birth,
        'controller_argv':s.controller_argv,'hold_sha256':hashlib.sha256(s.hold_bytes).hexdigest(),
        'source_head':s.source['head'],'source_tree':s.source['tree'],'operation_root':str(f.output),
        'runtime_manifest':str(manifest),'runtime_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'runtime_helper_sha256':helper_sha,'owner_identity':s.identity}
    request_path=f.output.with_name(f.output.name+'-request.json');request_path.write_text(json.dumps(request))
    fixture={'root':str(s.root),'source':s.source,'board':{
        'repo_root':str(s.root),'board_namespace':s.board.board_namespace,'max_lanes':s.board.max_lanes,
        'task_prefix':s.board.task_prefix,'runtime_paths':s.board.runtime_paths,'payload':s.board.payload},
        'paths':{key:str(value) for key,value in s.paths.items()},'unit':stopped.native.UNIT,
        'repair_idle':f.fleet.unit,'observation':observation(s.source,s.bootstrap,s.owner),
        'request_path':str(request_path),'request_sha256':hashlib.sha256(request_path.read_bytes()).hexdigest(),
        'failure':retry}
    fixture_path=f.output.with_name(f.output.name+'-fixture.json');fixture_path.write_text(json.dumps(fixture))
    root=Path(__file__).resolve().parents[3]
    process=subprocess.Popen(['/usr/bin/python3','-I','-c',ISOLATED_DRIVER,str(root),str(fixture_path)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd='/')
    def command(name):
        process.stdin.write(json.dumps({'command':name})+'\n');process.stdin.flush();return _reply(process)
    try:
        assert _reply(process)['stage']=='inspected'
        for name,state in [('overlap','overlapped'),('retire','retired'),('fence','fenced')]:
            result=command(name);assert result['stage']==state,result
        result=command('capture')
        if retry:
            assert result['error']=='ModuleNotFoundError' and result['stage']=='fenced'
            assert result['custody_retained'] is True
            result=command('capture')
        assert result['stage']=='captured',result
        assert command('prepare')['stage']=='prepared'
        installed=command('install');assert installed['stage']=='installed',installed
        finished=command('finish');assert finished['stage']=='finished',finished
        process.wait(timeout=10);assert process.returncode==0
        assert len(list(f.output.glob('capture-*')))==(2 if retry else 1)
        print('STOPPED_ISOLATED_DRIVER='+json.dumps({'retry':retry,'finished':finished,'argv_prefix':['/usr/bin/python3','-I'],
            'runtime_manifest_sha256':request['runtime_manifest_sha256'],'native_task_observation':'explicit isolated fixture only'}))
    finally:
        if process.poll() is None:
            # Exact disposable failed driver only; inherited test no-state
            # keeper closes on its stdin EOF when this process is retired.
            process.terminate();process.wait(timeout=10)
        process.stdin.close();process.stdout.close();process.stderr.close()


def test_aborted_overlap_preserves_old_actual_custody(stopped_fixture):
    f=stopped_fixture
    f.succession.overlap();new_pid=f.succession.keeper.pid
    new_fd=os.pidfd_open(new_pid)
    try:
        f.succession.abort_overlap()
        assert stopped.native._exited(new_fd)
        assert f.process.poll() is None
        assert f.succession.controller_locks()==f.succession.old_locks
        assert f.succession.stage=='inspected'
        captured=capture(f)
        assert captured.require_current()['completion_authority'] is False
    finally:os.close(new_fd)


def test_retry_does_not_reset_pre_handover_inventory(stopped_fixture,monkeypatch):
    f=stopped_fixture;retire(f)
    f.session=stopped.StoppedCaptureSession(f.succession,fleet_config=f.fleet.config)
    original=stopped.producer.produce_offline_import_plan
    def fail(**kwargs):
        original(**kwargs)
        raise ModuleNotFoundError('fixture-only',name='duckdb')
    monkeypatch.setattr(stopped.producer,'produce_offline_import_plan',fail)
    with pytest.raises(ModuleNotFoundError):f.session.capture(destination=f.output)
    monkeypatch.setattr(stopped.producer,'produce_offline_import_plan',original)
    (f.armed.queue/'private/callback-signing-material').write_bytes(b'changed-after-failure')
    with pytest.raises(role.SparMergeOwnerError):f.session.capture(destination=f.output.with_name(f.output.name+'-retry'))
    assert f.succession.keeper.poll() is None


def test_fresh_receipt_is_immutable_after_capture(stopped_fixture):
    captured=capture(stopped_fixture)
    (captured.path.parent/'stopped-capture-receipt.json').write_text('{}')
    with pytest.raises(role.SparMergeOwnerError):captured.require_current()


def test_changed_fleet_after_retirement_cannot_rebind_fresh_fences(stopped_fixture):
    f=stopped_fixture;retire(f)
    original=f.fleet.config.read_bytes()
    f.fleet.config.write_bytes(original+b'\n')
    with pytest.raises(role.SparMergeOwnerError,match='fleet configuration changed'):
        stopped.StoppedCaptureSession(f.succession,fleet_config=f.fleet.config)
    assert f.succession.keeper.poll() is None
    assert not f.output.exists()


@pytest.mark.parametrize('phase',['failed-controller-graceful-exit-prepared','failed-workflow-exit-observed'])
def test_retirement_audit_failure_preserves_actual_stage_and_single_signal(stopped_fixture,monkeypatch,phase):
    from scripts.ops.agent_supervisor.spar_stopped_capture_driver import StoppedCaptureDriver
    f=stopped_fixture;f.succession.overlap()
    value=object.__new__(StoppedCaptureDriver);value.succession=f.succession;value.stage='overlapped'
    value.require_runtime=lambda:None;value.record=lambda _:None;value.result=lambda:{'stage':value.stage}
    record=f.succession.record;failures=[];signals=[]
    original_signal=stopped.signal.pidfd_send_signal
    def signal_once(*args):signals.append(args);return original_signal(*args)
    monkeypatch.setattr(stopped.signal,'pidfd_send_signal',signal_once)
    def fail_once(current):
        if current==phase and not failures:
            failures.append(current);raise OSError('disposable audit write failed')
        return record(current)
    monkeypatch.setattr(f.succession,'record',fail_once)
    with pytest.raises(OSError):value.retire()
    if phase=='failed-controller-graceful-exit-prepared':
        assert value.stage=='overlapped' and f.process.poll() is None and not signals
        value.abort_overlap();assert value.stage=='inspected'
        f.succession.overlap();value.stage='overlapped'
    else:
        assert value.stage=='retiring' and f.succession.stage=='retired' and len(signals)==1
    assert value.retire()['stage']=='retired'
    assert len(signals)==1 and f.succession.retirement_recorded


def test_failed_helper_spawn_allows_pre_retirement_abort(stopped_fixture,monkeypatch):
    f=stopped_fixture
    def fail(*args,**kwargs):raise OSError('disposable Popen failure')
    monkeypatch.setattr(subprocess,'Popen',fail)
    with pytest.raises(OSError):f.succession.overlap()
    assert f.succession.stage=='overlapping' and f.succession.keeper is None
    f.succession.abort_overlap()
    assert f.succession.stage=='inspected' and f.process.poll() is None
    assert f.succession.controller_locks()==f.succession.old_locks


def test_worker_path_replacement_is_denied_before_execution(stopped_fixture,monkeypatch):
    f=stopped_fixture;s=f.succession
    s.worker_path=f.armed.tmp/'disposable-worker.py';s.worker_path.write_bytes(s.worker_bytes)
    s.worker_path.write_text("print('invented facts')")
    def denied(*args,**kwargs):raise AssertionError('changed worker was executed')
    monkeypatch.setattr(subprocess,'run',denied)
    with pytest.raises(role.SparMergeOwnerError,match='worker code changed'):
        stopped._task_worker(s,probe=True)


@pytest.mark.parametrize('phase',['keeper-normal-exit-observed','driver-finished'])
def test_finish_audit_can_resume_after_actual_helper_exit(stopped_fixture,monkeypatch,phase):
    from scripts.ops.agent_supervisor.spar_stopped_capture_driver import StoppedCaptureDriver
    f=stopped_fixture;captured=capture(f)
    prepared=role.prepare_offline_clone(offline_root=captured.path,destination=f.output/'prepared',manifest=captured.receipt['manifest'])
    installed=origin.install_stopped_queue(captured,prepared)
    # The replacement database has its own retained writer through installation.
    denied=subprocess.run([sys.executable,'-c','import duckdb,sys;duckdb.connect(sys.argv[1])',str(f.armed.queue/'merge_queue.duckdb')],capture_output=True,timeout=10)
    assert denied.returncode!=0 and b'lock' in denied.stderr.lower()
    value=object.__new__(StoppedCaptureDriver);value.stage='installed';value.finish_recorded=False
    value.require_runtime=lambda:None;value.session=f.session;value.succession=f.succession
    value.captured=captured;value.installed=installed;value.result=lambda:{'stage':value.stage}
    old_record=f.succession.record;failed=[]
    def succession_record(current):
        if phase==current and not failed:failed.append(1);raise OSError('disposable journal failure')
        old_record(current)
    def driver_record(current):
        if phase=='driver-finished' and current=='finished' and not failed:
            failed.append(1);raise OSError('disposable journal failure')
    monkeypatch.setattr(f.succession,'record',succession_record);value.record=driver_record
    with pytest.raises(OSError):value.finish()
    assert f.session._closed and f.succession.keeper.poll()==0
    assert value.finish()['stage']=='finished' and value.finish_recorded and f.succession.finish_recorded


def test_private_native_reader_uses_retained_bytes_and_checks_digest(tmp_path):
    from scripts.ops.agent_supervisor.spar_stopped_task_observation import _private_inspection
    root=tmp_path/'input';root.mkdir(mode=0o700)
    path=root/'control.duckdb';raw=b'original preserved input';path.write_bytes(raw)
    fd=os.open(path,os.O_RDONLY)
    try:
        info=os.fstat(fd)
        entry={'path':path.name,'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        manifest={'files':[{'file':entry,'descriptor':fd,'device':info.st_dev,'inode':info.st_ino}]}
        path.rename(root/'retained-original')
        path.write_bytes(b'substituted path content')
        private,binding,produced,resources=_private_inspection(path,manifest)
        try:
            assert private.read_bytes()==raw
            assert binding==[{'file':entry,'device':info.st_dev,'inode':info.st_ino}]
            from scripts.ops.agent_supervisor.spar_stopped_task_observation import _private_paths_current
            private.rename(private.with_name('preserved-produced'))
            private.write_bytes(b'changed-produced-inode')
            with pytest.raises(RuntimeError,match='private task input changed'):_private_paths_current(private,produced)
            assert os.pread(produced['control.duckdb'][0],len(raw),0)==raw
        finally:resources.close()
        manifest['files'][0]['file']=dict(entry,sha256='0'*64)
        with pytest.raises(RuntimeError,match='content differs'):_private_inspection(path,manifest)
    finally:os.close(fd)
