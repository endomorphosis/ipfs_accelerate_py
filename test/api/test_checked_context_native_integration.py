"""Native hook routing controls; authored child handles establish no real birth."""
import shlex
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import (
    AdmittedBenchmarkRuntime, _require_observer_diagnostic_command, _checked_budget_spawn_environment,
    CHECKED_CONTEXT_ENV_NAMES, CHECKED_CONTEXT_WRAPPER_ENV,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime as stream


def _coordination_fixture(monkeypatch, root):
    import json
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_context_dispatch as dispatch
    for name in (*CHECKED_CONTEXT_ENV_NAMES,
                 "IPFS_ACCELERATE_FINITE_PROOF_QUERY_DISPATCH_SOCKET",
                 "IPFS_ACCELERATE_FINITE_PROOF_QUERY_CONTEXT_CID"):
        monkeypatch.delenv(name, raising=False)
    endpoint = root / "owner.sock"
    monkeypatch.setenv(dispatch.SOCKET_ENV, str(endpoint))
    monkeypatch.setenv(dispatch.CONTEXT_ENV, "cid:authored-routing-only")
    monkeypatch.setenv(dispatch.OWNER_ENV, json.dumps(dispatch.owner_dispatch_identity(), sort_keys=True, separators=(",", ":")))
    calls = []
    daemon = SimpleNamespace(_typed_quack_authority_binding=object(), authority_mode="quack",
        task_shard_count=1, task_shard_index=0, _require_live_typed_owner=lambda: calls.append("live-typed-owner"))
    return endpoint, daemon, calls


@pytest.mark.parametrize("change", [None, "partial-socket", "partial-context", "partial-birth",
    "untyped", "multiple-lanes", "wrong-lane", "public-parent", "public-socket", "mixed", "bad-owner-birth"])
def test_checked_route_requires_shared_native_coordination_and_complete_owner_routing(monkeypatch, change):
    import socket
    import tempfile
    from pathlib import Path
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon, DatabaseImplementationAuthorityError,
    )
    # A real private socket under a short owned directory exercises native
    # endpoint checks without relying on a pytest path fitting AF_UNIX.
    with tempfile.TemporaryDirectory(prefix="aa-coor-") as directory:
        root = Path(directory)
        endpoint, daemon, calls = _coordination_fixture(monkeypatch, root)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.bind(str(endpoint))
            endpoint.chmod(0o600)
            if change == "partial-socket": monkeypatch.delenv(CHECKED_CONTEXT_ENV_NAMES[0])
            elif change == "partial-context": monkeypatch.delenv(CHECKED_CONTEXT_ENV_NAMES[1])
            elif change == "partial-birth": monkeypatch.delenv(CHECKED_CONTEXT_ENV_NAMES[2])
            elif change == "untyped": daemon._typed_quack_authority_binding = None
            elif change == "multiple-lanes": daemon.task_shard_count = 2
            elif change == "wrong-lane": daemon.task_shard_index = 1
            elif change == "public-parent": root.chmod(0o755)
            elif change == "public-socket": endpoint.chmod(0o666)
            elif change == "mixed": monkeypatch.setenv("IPFS_ACCELERATE_FINITE_PROOF_QUERY_CONTEXT_CID", "cid:another-route")
            elif change == "bad-owner-birth": monkeypatch.setenv(CHECKED_CONTEXT_ENV_NAMES[2], '{"pid":true}')
            if change is None:
                assert DatabaseImplementationDaemon._checked_context_serialized_coordinator(daemon) is True
                assert calls == ["live-typed-owner"]
            else:
                with pytest.raises(DatabaseImplementationAuthorityError):
                    DatabaseImplementationDaemon._checked_context_serialized_coordinator(daemon)
                assert calls == []


def test_absent_checked_route_keeps_existing_coordinator_selection(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    for name in CHECKED_CONTEXT_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert DatabaseImplementationDaemon._checked_context_serialized_coordinator(SimpleNamespace()) is False


def test_checked_routing_cannot_replace_live_typed_owner_authority(monkeypatch):
    import socket
    import tempfile
    from pathlib import Path
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon, DatabaseImplementationAuthorityError,
    )
    with tempfile.TemporaryDirectory(prefix="aa-coor-") as directory:
        endpoint, daemon, calls = _coordination_fixture(monkeypatch, Path(directory))
        def refuse():
            calls.append("refused")
            raise DatabaseImplementationAuthorityError("authored unavailable native grant")
        daemon._require_live_typed_owner = refuse
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.bind(str(endpoint)); endpoint.chmod(0o600)
            with pytest.raises(DatabaseImplementationAuthorityError, match="unavailable native grant"):
                DatabaseImplementationDaemon._checked_context_serialized_coordinator(daemon)
        assert calls == ["refused"]


def command():
    return ["/opt/ipfs-supervisor/bin/router-worker", "--model", "authored-offline-observer",
        "--observe-model-input-only", "--purpose", "coding", "--semantic-repository", "/app",
        "--semantic-transport", "native", "--public-instruction-artifact", "/app/.runtime/instruction.json",
        "--public-instruction-sha256", "a"*64, "--public-instruction-task-cid", "cid:task",
        "--checked-context-owner-scope-cid", "cid:owner-scope"]


@pytest.mark.parametrize("selected", [False, True])
def test_only_signed_profile_selects_checked_owner_routing(selected):
    ambient={name:"ambient" for name in (*CHECKED_CONTEXT_ENV_NAMES,CHECKED_CONTEXT_WRAPPER_ENV)}
    ambient["PUBLIC_SETTING"]="preserved"
    signed={name:"signed-"+str(index) for index,name in enumerate(CHECKED_CONTEXT_ENV_NAMES)} if selected else None
    result=_checked_budget_spawn_environment(ambient,signed)
    assert result == {"PUBLIC_SETTING":"preserved",**(signed or {})}
    assert ambient[CHECKED_CONTEXT_WRAPPER_ENV]=="ambient"


@pytest.mark.parametrize("change", ["missing", "extra", "empty", "untyped"])
def test_partial_signed_routing_refuses_before_native_spawn(change):
    signed={name:"signed" for name in CHECKED_CONTEXT_ENV_NAMES}
    if change=="missing":signed.pop(CHECKED_CONTEXT_ENV_NAMES[0])
    elif change=="extra":signed[CHECKED_CONTEXT_WRAPPER_ENV]="ambient-ticket"
    elif change=="empty":signed[CHECKED_CONTEXT_ENV_NAMES[0]]=""
    else:signed[CHECKED_CONTEXT_ENV_NAMES[0]]=[]
    with pytest.raises(ValueError,match="signed checked budget routing"):
        _checked_budget_spawn_environment({},signed)


@pytest.mark.parametrize("change", [None, "partial", "duplicate", "equals", "mixed", "identifiers", "planning", "residual"])
def test_signed_observer_recipe_selects_only_closed_owner_context_route(change):
    args=command()
    if change == "partial": args.pop()
    elif change == "duplicate": args += args[-2:]
    elif change == "equals": args[-2:] = ["--checked-context-owner-scope-cid=cid:owner-scope"]
    elif change == "mixed": args += ["--checked-context-artifact", "/checked", "--checked-context-sha256", "b"*64, "--checked-context-task-cid", "cid:task"]
    elif change == "identifiers": args[args.index("--semantic-transport")+1]="identifiers"
    elif change == "planning": args[args.index("--purpose")+1]="planning"
    elif change == "residual": args += ["--doctor-residual-artifact", "/residual"]
    kwargs=dict(implement=True,context_bundle={"authored":True},candidate_runner={"authored":True})
    if change is None:
        selected=_require_observer_diagnostic_command(shlex.join(args),**kwargs)
        assert selected["--checked-context-owner-scope-cid"] == "cid:owner-scope"
        assert "--checked-context-artifact" not in selected
    else:
        with pytest.raises(ValueError): _require_observer_diagnostic_command(shlex.join(args),**kwargs)


@pytest.mark.parametrize("flag", ["--checked-context-owner-scope-cid", "--checked-context-owner-scope-cid=cid:foreign"])
def test_budget_marker_without_original_scope_refuses_before_state_creation(tmp_path,flag):
    args=command()[:-2]+[flag]
    if "=" not in flag: args.append("cid:owner-scope")
    with pytest.raises(ValueError,match="original owner scope"):
        AdmittedBenchmarkRuntime.create(tmp_path/"launch",admission=None,server=None,source=None,
            implement=True,implementation_command=shlex.join(args),native_dispatch_diagnostics=True)
    assert not (tmp_path/"launch").exists()


def test_native_prelaunch_uses_owner_derived_command_and_actual_input(monkeypatch,tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_context_dispatch as dispatch
    order=[];args=command();derived=args[:-2]+["--checked-context-artifact","/checked",
        "--checked-context-sha256","b"*64,"--checked-context-task-cid","cid:task"]
    child=SimpleNamespace(pid=12345)
    class Guard:
        command=derived
        def child_environment(self): return {dispatch.WRAPPER_ENV:"authored-routing-ticket"}
        def note_spawned(self,value): assert value is child;order.append("birth-ack")
        def abort(self): order.append("abort")
    def require(**kwargs):
        assert kwargs["input_text"] == "the actual daemon stdin\n"
        assert kwargs["dispatch_binding"] == {"authored":"native-attempt"}
        assert kwargs["command"] == args and kwargs["worktree"] == tmp_path
        order.append("owner-reserved");return Guard()
    def popen(actual,**kwargs):
        assert order == ["owner-reserved"] and actual == derived
        assert kwargs["env"][dispatch.WRAPPER_ENV] == "authored-routing-ticket"
        order.append("popen");return child
    monkeypatch.setattr(dispatch,"require_checked_context_worker_dispatch",require)
    monkeypatch.setattr(stream.subprocess,"Popen",popen)
    actual=stream.launch_process_child(args,cwd=tmp_path,inherit_environment=False,
        checked_context_input="the actual daemon stdin\n",dispatch_binding={"authored":"native-attempt"})
    assert actual is child and order == ["owner-reserved","popen","birth-ack"]


@pytest.mark.parametrize("failure", ["prepare", "popen"])
def test_prelaunch_refusal_never_launches_or_refunds_owner_attempt(monkeypatch,tmp_path,failure):
    from ipfs_accelerate_py.agent_supervisor.runtime import checked_context_dispatch as dispatch
    calls=[]
    class Guard:
        command=command()[:-2]
        def child_environment(self):return {}
        def abort(self):calls.append("charged-abort")
    def require(**kwargs):
        calls.append("prepare")
        if failure == "prepare":raise ValueError("owner refused")
        return Guard()
    def popen(*args,**kwargs):
        calls.append("popen");raise OSError("authored Popen failure")
    monkeypatch.setattr(dispatch,"require_checked_context_worker_dispatch",require)
    monkeypatch.setattr(stream.subprocess,"Popen",popen)
    with pytest.raises((ValueError,OSError)):
        stream.launch_process_child(command(),cwd=tmp_path,inherit_environment=False,
            checked_context_input="actual stdin")
    assert calls == (["prepare"] if failure == "prepare" else ["prepare","popen","charged-abort"])


def test_stream_forwards_same_stdin_before_literal_launch(monkeypatch,tmp_path):
    import errno
    seen=[]
    class Child:
        pid=31415926;returncode=75
        def communicate(self,*,input,timeout):seen.append(("communicate",input,timeout));return None,None
    def launch(args,**kwargs):seen.append(("launch",kwargs));return Child()
    def absent_group(pid,signum):
        assert pid==31415926 and signum==0
        raise OSError(errno.ESRCH,"authored group absent")
    monkeypatch.setattr(stream,"launch_process_child",launch)
    monkeypatch.setattr(stream.os,"killpg",absent_group)
    result=stream.run_process_group_stream(command(),cwd=tmp_path,stdout=None,input_text="original stdin",
        inherit_environment=False,timeout_seconds=2,dispatch_binding={"authored":"attempt"})
    assert seen[0][1]["checked_context_input"] == seen[1][1] == "original stdin"
    assert seen[0][1]["dispatch_binding"] == {"authored":"attempt"} and result.returncode==75


def test_legacy_child_has_unchanged_popen_arguments(monkeypatch,tmp_path):
    seen=[];child=object()
    monkeypatch.setattr(stream.subprocess,"Popen",lambda args,**kwargs:(seen.append((args,kwargs)) or child))
    assert stream.launch_process_child(["/usr/bin/true"],cwd=tmp_path,inherit_environment=False) is child
    assert seen[0][0] == ["/usr/bin/true"] and seen[0][1]["env"] == {}


def test_owner_monitor_ack_failure_fences_and_cleans_isolated_child(monkeypatch):
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    import signal,subprocess
    order=[]
    class Child:
        def terminate(self):order.append("terminate")
        def wait(self,**kwargs):order.append("wait");return 75
    class Guard:
        def note_spawned(self,process):order.append("ack");raise ValueError("authored lost ACK")
        def abort(self):order.append("charged-abort")
    monkeypatch.setattr(signal,"signal",lambda *args:None)
    monkeypatch.setattr(subprocess,"Popen",lambda args:(order.append("popen") or Child()))
    monkeypatch.setattr(subprocess,"run",lambda *args,**kwargs:(order.append("cleanup") or SimpleNamespace(returncode=0)))
    with pytest.raises(ValueError,match="lost ACK"):
        exec(compile(deployment.OWNER_MONITOR,"<native-owner-monitor>","exec"),
            {"command":["authored-worker"],"checked_context_guard":Guard()})
    assert order == ["popen","ack","terminate","wait","cleanup","charged-abort"]


def _worktree_owner_fixture(monkeypatch, tmp_path):
    """Authored same-UID response and custody mocks; no actual kernel authority."""
    import hashlib,json,os
    from pathlib import PurePosixPath
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    workspace="/opt/ipfs-supervisor/worktrees/authored"
    marker="/opt/ipfs-supervisor/bin/worker-entry"
    identity={"pid":13,"uid":1001,"parent_pid":12,"start_time_ticks":101,
        "process_birth_id":"birth:"+"a"*32,"command":["/opt/ipfs-supervisor/venv/bin/python","-I",marker,"--observe-model-input-only"]}
    response={"schema":"checked-worker-kernel-worktree-observation@1",
        **{k:identity[k] for k in ("pid","uid","parent_pid","start_time_ticks","process_birth_id")},
        "boot_id":"00000000-0000-0000-0000-000000000000","worktree":workspace,
        "command_sha256":hashlib.sha256(json.dumps(identity['command'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
        "namespaces":{"pid":"pid:[1]","mnt":"mnt:[2]","net":"net:[3]"},
        "metadata_only":True,"execution_authority":False,"proof_authority":False,
        "completion_authority":False,"source_body_executed":False,"provider_calls":0}
    state={"calls":[],"symlink":False,"helper_changed":False}
    class MetadataPath(PurePosixPath):
        def resolve(self,strict=False):return MetadataPath('/different') if state['symlink'] else self
        def is_dir(self):return True
        def is_symlink(self):return state['symlink']
    monkeypatch.setattr(deployment,'Path',MetadataPath)
    held=tmp_path/'held-helper-fd';held.write_bytes(b'authored mock helper')
    def custody():
        fd=os.open(held,os.O_RDONLY)
        def guard():
            state['calls'].append('helper-custody')
            if state['helper_changed']:raise deployment.WorkerWorktreeObservationError('authored helper changed')
        return fd,guard
    monkeypatch.setattr(deployment,'_worker_entry_custody',custody)
    def run(argv,**kwargs):
        state['calls'].append(('sudo',argv,kwargs))
        return SimpleNamespace(returncode=0,stdout=(json.dumps(response,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode())
    monkeypatch.setattr(deployment.subprocess,'run',run)
    return deployment,identity,workspace,response,state


def test_fixed_same_uid_metadata_route_is_bounded_and_joins_captured_identity(monkeypatch,tmp_path):
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    assert deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace)==response
    calls=[row for row in state['calls'] if isinstance(row,tuple)]
    assert len(calls)==1
    _,argv,kwargs=calls[0]
    assert argv==['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry',
        '--observe-worker-worktree','13','101','birth:'+'a'*32,workspace]
    assert kwargs=={'cwd':'/','env':{'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'},
        'stdin':deployment.subprocess.DEVNULL,'stdout':deployment.subprocess.PIPE,
        'stderr':deployment.subprocess.DEVNULL,'timeout':3.0,'check':False}
    assert state['calls'][0]==state['calls'][-1]=='helper-custody'


@pytest.mark.parametrize('change',['pid','uid','parent','ticks','birth','cwd','command','schema','extra',
    'authority','metadata','source','provider-bool','provider','namespace','boot','bool-pid','no-response','oversize','nonzero'])
def test_same_uid_metadata_refusals_cannot_replace_captured_worker(monkeypatch,tmp_path,change):
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    if change in ('pid','uid'):response[change]+=1
    elif change=='parent':response['parent_pid']+=1
    elif change=='ticks':response['start_time_ticks']+=1
    elif change=='birth':response['process_birth_id']='birth:'+'b'*32
    elif change=='cwd':response['worktree']+='/other'
    elif change=='command':response['command_sha256']='f'*64
    elif change=='schema':response['schema']='other'
    elif change=='extra':response['extra']='untrusted'
    elif change=='authority':response['execution_authority']=True
    elif change=='metadata':response['metadata_only']=False
    elif change=='source':response['source_body_executed']=True
    elif change=='provider-bool':response['provider_calls']=False
    elif change=='provider':response['provider_calls']=1
    elif change=='namespace':response['namespaces']['extra']='pid:[4]'
    elif change=='boot':response['boot_id']='invalid'
    elif change=='bool-pid':response['pid']=True
    else:
        def run(*args,**kwargs):
            state['calls'].append('authored-run')
            return SimpleNamespace(returncode=1 if change=='nonzero' else 0,
                stdout=b'' if change=='no-response' else b'x'*16385)
        monkeypatch.setattr(deployment.subprocess,'run',run)
    with pytest.raises(deployment.WorkerWorktreeObservationError):
        deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace)
    assert state['calls'][-1]=='helper-custody'


@pytest.mark.parametrize('change',['uid','pid-bool','ticks-bool','missing','extra','marker','metadata-worker','relative','outside','dotdot','symlink'])
def test_invalid_metadata_request_refuses_before_existing_sudo_route(monkeypatch,tmp_path,change):
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    if change=='uid':identity['uid']=1000
    elif change=='pid-bool':identity['pid']=True
    elif change=='ticks-bool':identity['start_time_ticks']=False
    elif change=='missing':identity.pop('parent_pid')
    elif change=='extra':identity['executable']='/untrusted'
    elif change=='marker':identity['command']=['/untrusted']
    elif change=='metadata-worker':identity['command'].append('--observe-worker-worktree')
    elif change=='relative':workspace='relative'
    elif change=='outside':workspace='/app'
    elif change=='dotdot':workspace='/opt/ipfs-supervisor/worktrees/../state'
    elif change=='symlink':state['symlink']=True
    with pytest.raises(deployment.WorkerWorktreeObservationError):
        deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace)
    assert state['calls']==[]


@pytest.mark.parametrize('keyword',['proc_root','executable','environment','timeout','command'])
def test_metadata_observer_accepts_no_helper_or_proc_substitution(monkeypatch,tmp_path,keyword):
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    with pytest.raises(TypeError):deployment.observe_worker_kernel_worktree(
        worker_identity=identity,expected_worktree=workspace,**{keyword:'/untrusted'})
    assert state['calls']==[]


def test_installed_helper_drift_after_subprocess_is_refused(monkeypatch,tmp_path):
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    def run(*args,**kwargs):
        state['helper_changed']=True
        return SimpleNamespace(returncode=0,stdout=b'{}\n')
    monkeypatch.setattr(deployment.subprocess,'run',run)
    with pytest.raises(deployment.WorkerWorktreeObservationError,match='helper changed'):
        deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace)


def _embedded_metadata_fixture(change=None):
    """Execute only the embedded stdlib function over wholly authored proc data."""
    import hashlib,json,os,pathlib,stat
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    root='/opt/ipfs-supervisor';workspace=root+'/worktrees/authored';boot='00000000-0000-0000-0000-000000000000'
    namespaces={'pid':'pid:[1]','mnt':'mnt:[2]','net':'net:[3]'}
    boundary={'schema':'supervisor-container-worker-boundary@1','container_id':'a'*64,'image_id':'sha256:'+'b'*64,
        'owner_uid':1000,'worker_uid':1001,'single_worker':True,'namespaces':namespaces,
        'allowed_worktree_roots':[root+'/worktrees'],'owner_private_paths':[root+'/state',root+'/home',root+'/codex-auth'],
        'validation_repository_roots':['/app']}
    state={'stat_reads':0,'cwd_reads':0,'status_reads':0,'clock':0.0,'sleeps':[]}
    if change=='not-single':boundary['single_worker']=False
    command=['/opt/ipfs-supervisor/venv/bin/python','-I',root+'/bin/worker-entry','--observe-model-input-only']
    if change=='marker':command[2]='/different/worker-entry'
    if change=='metadata-worker':command+=['--observe-worker-worktree']
    def status(uid=1001,readiness='1',caps='0'):
        return ('Uid:\t'+('\t'.join([str(uid)]*4))+'\nCapEff:\t'+caps+'\nCapPrm:\t0\nCapAmb:\t0\nNoNewPrivs:\t'+readiness+'\n').encode()
    class KernelPath(pathlib.PurePosixPath):
        def read_bytes(self):
            name=str(self)
            if name=='/proc/13/stat':
                state['stat_reads']+=1
                start=102 if change=='reuse' and state['stat_reads']>1 else 101
                facts=['Z' if change=='zombie' else 'S','14' if change=='await-parent' and state['stat_reads']>1 else '12']+['0']*17+[str(start)]
                return ('13 (authored worker) '+' '.join(facts)).encode()
            if name=='/proc/13/status':
                state['status_reads']+=1
                waiting=change in ('delayed-ready','never-ready','await-parent','await-cwd','await-command','await-caps')
                ready='0' if waiting and (change!='delayed-ready' or state['status_reads']<=3) else '1'
                if change=='regress-ready' and state['status_reads']>1:ready='0'
                if change=='invalid-ready':ready='2'
                return status(1000 if change=='uid' else 1001,ready,
                    '1' if change=='await-caps' and state['status_reads']>1 else '0')
            if name=='/proc/13/cmdline':
                current=command+(['changed'] if change=='await-command' and state['stat_reads']>1 else [])
                return b'\0'.join(part.encode() for part in current)+b'\0'
            if name=='/proc/self/status':return status()
            if name=='/proc/sys/kernel/random/boot_id':return boot.encode()
            raise FileNotFoundError(name)
        def read_text(self):return self.read_bytes().decode()
        def resolve(self,strict=False):return KernelPath('/different') if change=='symlink' and str(self)==workspace else self
        def is_symlink(self):return change=='symlink' and str(self)==workspace
        def is_dir(self):return True
        def lstat(self):
            private=str(self) in boundary['owner_private_paths']
            return SimpleNamespace(st_mode=stat.S_IFDIR|(0o700 if private else 0o755),st_uid=1000 if private else 0)
    def readlink(path):
        path=str(path)
        if path=='/proc/13/cwd':
            state['cwd_reads']+=1
            return workspace+'/other' if change=='cwd' or (change in ('late-cwd','await-cwd') and state['cwd_reads']>1) else workspace
        if '/ns/' in path:
            key=path.rsplit('/',1)[1]
            return 'pid:[9]' if change=='namespace' and path=='/proc/13/ns/pid' else namespaces[key]
        raise FileNotFoundError(path)
    def sleep(seconds):
        state['sleeps'].append(seconds);state['clock']+=seconds
    ns={'root':KernelPath(root),'pathlib':SimpleNamespace(Path=KernelPath),'json':json,'stat':stat,'hashlib':hashlib,
        'time':SimpleNamespace(monotonic=lambda:state['clock'],sleep=sleep),
        'os':SimpleNamespace(readlink=readlink,access=lambda *a:False,W_OK=os.W_OK,R_OK=os.R_OK,X_OK=os.X_OK),
        '_profile_read':lambda path:json.dumps(boundary).encode()}
    exec(compile(deployment.WORKER_WORKTREE_OBSERVER,'<authored-kernel-metadata-function>','exec'),ns)
    birth='birth:'+hashlib.sha256(('13:101:'+boot+':12').encode()).hexdigest()[:32]
    return ns['_observe_worker_worktree'],(13,101,birth,workspace),state


def test_embedded_metadata_observer_rechecks_same_uid_birth_command_and_actual_cwd():
    import json
    observe,args,state=_embedded_metadata_fixture()
    value=json.loads(observe(args))
    assert state['stat_reads']==state['cwd_reads']==state['status_reads']==2 and state['sleeps']==[]
    assert value['pid']==13 and value['uid']==1001 and value['worktree']==args[-1]
    assert value['metadata_only']is True and value['provider_calls']==0
    assert all(value[name]is False for name in ('execution_authority','proof_authority','completion_authority','source_body_executed'))


@pytest.mark.parametrize('change',['reuse','uid','cwd','late-cwd','symlink','namespace','marker','metadata-worker','not-single','zombie'])
def test_embedded_metadata_observer_refuses_authored_kernel_drift(change):
    observe,args,state=_embedded_metadata_fixture(change)
    with pytest.raises(SystemExit):observe(args)


def test_metadata_readiness_waits_only_for_stable_zero_privilege_transition():
    import json
    observe,args,state=_embedded_metadata_fixture('delayed-ready')
    value=json.loads(observe(args))
    assert state['stat_reads']==state['cwd_reads']==state['status_reads']==5
    assert state['sleeps']==[.01]*3 and state['clock']==.03
    assert value['pid']==13 and value['worktree']==args[-1] and value['provider_calls']==0


@pytest.mark.parametrize('change',['never-ready','regress-ready','invalid-ready','await-parent','await-cwd','await-command','await-caps'])
def test_metadata_readiness_cannot_relax_security_or_identity(change):
    observe,args,state=_embedded_metadata_fixture(change)
    with pytest.raises(SystemExit):observe(args)
    assert state['clock']<=2.01
    if change!='never-ready':assert state['status_reads']<=2


@pytest.mark.parametrize('timeout',[True,False,0,-1,3.01,float('nan'),float('inf'),'1',None])
def test_metadata_timeout_refuses_invalid_or_extended_original_window(monkeypatch,tmp_path,timeout):
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    with pytest.raises(deployment.WorkerWorktreeObservationError):
        deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace,timeout_seconds=timeout)
    assert state['calls']==[]


def test_metadata_timeout_accepts_only_smaller_original_window(monkeypatch,tmp_path):
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    assert deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace,timeout_seconds=.125)==response
    assert next(row for row in state['calls'] if isinstance(row,tuple))[2]['timeout']==.125


def _authored_helper_refusal():
    import hashlib
    return {'schema':'checked-worker-kernel-worktree-refusal@1','stage':'readiness','error_type':'SystemExit',
        'message_sha256':hashlib.sha256(b'authored readiness refusal').hexdigest(),
        'failure_frames':[{'file':'worker-entry','function':'_observe_worker_worktree','line':100}],
        'metadata_only':True,'execution_authority':False,'proof_authority':False,
        'completion_authority':False,'source_body_executed':False,'provider_calls':0}


def test_source_owned_helper_refusal_is_closed_hashed_data_and_remains_refusal(monkeypatch,tmp_path):
    import hashlib,json
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    refusal=_authored_helper_refusal();raw=(json.dumps(refusal,sort_keys=True,separators=(',',':'))+'\n').encode()
    monkeypatch.setattr(deployment.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=raw))
    with pytest.raises(deployment.WorkerWorktreeObservationError) as caught:
        deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace)
    value=caught.value.worker_worktree_failure
    assert value['helper_refusal']==refusal and value['returncode']==1 and value['phase']=='subprocess_response'
    assert value['stdout_bytes']==len(raw) and value['stdout_sha256']==hashlib.sha256(raw).hexdigest()
    assert value['helper_sha256']==hashlib.sha256(deployment.WORKER_ENTRY.encode()).hexdigest()
    assert value['helper_bytes']==len(deployment.WORKER_ENTRY.encode()) and value['timeout_seconds']==3.0
    assert state['calls'][0]==state['calls'][-1]=='helper-custody'
    assert value['metadata_only']is True and value['provider_calls']==0
    assert all(value[key]is False for key in ('execution_authority','proof_authority','completion_authority','source_body_executed'))


@pytest.mark.parametrize('change',['extra','authority','stage','frames','path','line-bool','provider-bool','noncanonical','duplicate','oversize'])
def test_invalid_helper_refusal_never_exports_unvalidated_child_metadata(monkeypatch,tmp_path,change):
    import json
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    refusal=_authored_helper_refusal()
    if change=='extra':refusal['raw_stderr']='private body'
    elif change=='authority':refusal['proof_authority']=True
    elif change=='stage':refusal['stage']='unknown'
    elif change=='frames':refusal['failure_frames']*=17
    elif change=='path':refusal['failure_frames'][0]['file']='/private/source.py'
    elif change=='line-bool':refusal['failure_frames'][0]['line']=True
    elif change=='provider-bool':refusal['provider_calls']=False
    raw=(json.dumps(refusal,sort_keys=True,separators=(',',':'))+'\n').encode()
    if change=='noncanonical':raw=b' '+raw
    elif change=='duplicate':raw=b'{"stage":"readiness",'+raw[1:]
    elif change=='oversize':raw=b'private body'*2000
    monkeypatch.setattr(deployment.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=raw))
    with pytest.raises(deployment.WorkerWorktreeObservationError) as caught:
        deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace)
    assert caught.value.worker_worktree_failure['helper_refusal']is None
    assert 'private body' not in json.dumps(caught.value.worker_worktree_failure)


def test_helper_timeout_preserves_original_custody_and_only_hashed_partial_output(monkeypatch,tmp_path):
    import hashlib
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    def run(*a,**kw):raise deployment.subprocess.TimeoutExpired(a[0],kw['timeout'],output=b'private partial bytes')
    monkeypatch.setattr(deployment.subprocess,'run',run)
    with pytest.raises(deployment.WorkerWorktreeObservationError) as caught:
        deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace,timeout_seconds=.125)
    value=caught.value.worker_worktree_failure
    assert value['phase']=='subprocess_timeout' and value['returncode']is None and value['helper_refusal']is None
    assert value['stdout_bytes']==21 and value['stdout_sha256']==hashlib.sha256(b'private partial bytes').hexdigest()
    assert value['timeout_seconds']==.125 and state['calls'][-1]=='helper-custody'


def test_exact_embedded_refusal_emitter_reaches_owner_decoder_without_plaintext(monkeypatch,tmp_path,capsys):
    import hashlib,json,pathlib
    deployment,identity,workspace,response,state=_worktree_owner_fixture(monkeypatch,tmp_path)
    namespace={'pathlib':pathlib,'hashlib':hashlib,'json':json}
    exec(compile(deployment.WORKER_WORKTREE_FAILURE,'<authored-metadata-emitter>','exec'),namespace)
    namespace['_worktree_observation_stage']='target_status'
    try:raise SystemExit('authored secret refusal text')
    except SystemExit as error:namespace['_worktree_failure'](error)
    raw=capsys.readouterr().out.encode()
    assert b'authored secret refusal text' not in raw
    monkeypatch.setattr(deployment.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=raw))
    with pytest.raises(deployment.WorkerWorktreeObservationError) as caught:
        deployment.observe_worker_kernel_worktree(worker_identity=identity,expected_worktree=workspace)
    decoded=caught.value.worker_worktree_failure['helper_refusal']
    assert decoded['stage']=='target_status' and decoded['error_type']=='SystemExit'
    assert decoded['message_sha256']==hashlib.sha256(b'authored secret refusal text').hexdigest()


@pytest.mark.parametrize('observer_only',[False,True])
def test_metadata_mode_is_closed_and_allowed_without_provider_execution_in_both_profiles(observer_only):
    import ast,hashlib,json,pathlib
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    tree=ast.parse(deployment.PROFILE_GUARD)
    chosen=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in
        ('_worktree_observation_arguments','_worker_profile')]
    boundary=b'authored protected boundary bytes'
    profile={'schema':deployment.PROFILE_SCHEMA,'observer_only':observer_only,'boundary_sha256':hashlib.sha256(boundary).hexdigest()}
    ns={'pathlib':pathlib,'hashlib':hashlib,'_profile_read':lambda path:boundary if path.name=='container-boundary.json'else b'profile',
        '_profile_json':lambda raw:profile,'_observer_absence':lambda *args:pytest.fail('metadata mode touched provider paths')}
    exec(compile(ast.Module(body=chosen,type_ignores=[]),'<closed-profile-mode>','exec'),ns)
    args=['--observe-worker-worktree','13','101','birth:'+'a'*32,'/opt/ipfs-supervisor/worktrees/authored']
    assert ns['_worker_profile'](pathlib.Path('/opt/ipfs-supervisor'),'worker',args)==profile
    for malformed in (args[:-1],args+['--model','other'],[args[0]+'=13',*args[2:]],
                      [args[0],'true',*args[2:]],[args[0],'013',*args[2:]],
                      [*args[:-1],'/opt/ipfs-supervisor/worktrees/../state']):
        with pytest.raises(SystemExit):ns['_worker_profile'](pathlib.Path('/opt/ipfs-supervisor'),'worker',malformed)
    assert deployment.WORKER_ENTRY.index('if observation_arguments is not None:')<deployment.WORKER_ENTRY.index(
        'from ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary import')


def _installed_entry_fixture(monkeypatch,tmp_path,change=None):
    """Real entry bytes/inodes/FDs with explicitly authored root ownership stat."""
    import os,stat
    from pathlib import PurePosixPath
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    installed=tmp_path/'worker-entry'
    installed.write_bytes(deployment.WORKER_ENTRY.encode());installed.chmod(0o555)
    if change=='bytes':installed.chmod(0o755);installed.write_bytes(b'wrong helper');installed.chmod(0o555)
    elif change=='mode':installed.chmod(0o777)
    elif change=='symlink':
        target=tmp_path/'actual';installed.rename(target);installed.symlink_to(target)
    elif change=='hardlink':os.link(installed,tmp_path/'extra-link')
    fixed='/opt/ipfs-supervisor/bin/worker-entry'
    def root_stat(info):
        values={name:getattr(info,name) for name in ('st_dev','st_ino','st_mode','st_uid','st_gid',
            'st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
        values['st_uid']=1000 if change=='owner' else 0
        return SimpleNamespace(**values)
    class InstalledPath(PurePosixPath):
        def __fspath__(self):return str(installed) if str(self)==fixed else str(self)
        def resolve(self,strict=False):return InstalledPath('/different') if installed.is_symlink() and str(self)==fixed else self
        def lstat(self):
            if str(self)==fixed:return root_stat(installed.lstat())
            return SimpleNamespace(st_mode=stat.S_IFDIR|(0o777 if change=='parent-mode' else 0o755),
                st_uid=1000 if change=='parent-owner' else 0)
    opened=[]
    def open_file(path,flags):opened.append(flags);return os.open(path,flags)
    monkeypatch.setattr(deployment,'Path',InstalledPath)
    monkeypatch.setattr(deployment,'os',SimpleNamespace(open=open_file,fstat=lambda fd:root_stat(os.fstat(fd)),
        pread=os.pread,close=os.close,O_RDONLY=os.O_RDONLY,O_NONBLOCK=os.O_NONBLOCK,O_NOFOLLOW=os.O_NOFOLLOW))
    return deployment,installed,opened


def test_installed_worker_entry_custody_reopens_original_protected_bytes(monkeypatch,tmp_path):
    import os
    deployment,installed,opened=_installed_entry_fixture(monkeypatch,tmp_path)
    fd,guard=deployment._worker_entry_custody()
    try:guard()
    finally:os.close(fd)
    assert len(opened)>=3 and all(flags & os.O_NOFOLLOW for flags in opened)


@pytest.mark.parametrize('change',['bytes','mode','owner','parent-mode','parent-owner','symlink','hardlink'])
def test_unprotected_or_substituted_installed_helper_refuses(monkeypatch,tmp_path,change):
    deployment,installed,opened=_installed_entry_fixture(monkeypatch,tmp_path,change)
    with pytest.raises(deployment.WorkerWorktreeObservationError):deployment._worker_entry_custody()


@pytest.mark.parametrize('change',['bytes','inode'])
def test_retained_installed_helper_detects_change_even_when_replacement_bytes_match(monkeypatch,tmp_path,change):
    import os
    deployment,installed,opened=_installed_entry_fixture(monkeypatch,tmp_path)
    fd,guard=deployment._worker_entry_custody()
    try:
        if change=='inode':
            installed.rename(tmp_path/'old-entry');installed.write_bytes(deployment.WORKER_ENTRY.encode());installed.chmod(0o555)
        else:installed.chmod(0o755);installed.write_bytes(b'changed');installed.chmod(0o555)
        with pytest.raises(deployment.WorkerWorktreeObservationError):guard()
    finally:os.close(fd)


def _git_read_fixture(monkeypatch,tmp_path):
    """Real disposable Git bytes; native allocation/claim authority is not asserted."""
    import os,subprocess
    from pathlib import Path
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    repository=tmp_path/'repository';repository.mkdir(mode=0o700)
    worktrees=tmp_path/'allocated-worktrees';worktrees.mkdir(mode=0o700)
    worktree=worktrees/'owned';git_env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C',
        'GIT_CONFIG_NOSYSTEM':'1','GIT_NO_REPLACE_OBJECTS':'1'}
    def git(*args):
        return subprocess.check_output(['/usr/bin/git','-C',str(repository),'-c','core.hooksPath=/dev/null',
            '-c','core.fsmonitor=false',*args],env=git_env,stderr=subprocess.DEVNULL,text=True).strip()
    previous=os.umask(0o077)
    try:
        git('init','-q','--initial-branch=master')
        (repository/'caller.py').write_text('def caller(n):\n    return n + 2\n')
        (repository/'build').mkdir();(repository/'build/instruction.txt').write_text('Authored current Git read fixture.\n')
        git('add','caller.py','build/instruction.txt')
        git('-c','user.name=Authored Fixture','-c','user.email=fixture@example.test','commit','-q','-m','fixture')
        git('worktree','add','-q','-b','implementation/owned',str(worktree),'HEAD')
    finally:os.umask(previous)
    original_path=Path
    def path(value):
        return worktrees if str(value)=='/opt/ipfs-supervisor/worktrees' else original_path(value)
    monkeypatch.setattr(deployment,'Path',path)
    registration=original_path((worktree/'.git').read_text()[8:].strip())
    return deployment,repository,worktree,registration,git


def test_real_git_current_registration_and_loose_objects_gain_only_group_reads(monkeypatch,tmp_path):
    import os,stat
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    try:
        receipt=guard.receipt;guard.require_current()
        assert receipt['selected_branch']=='implementation/owned' and receipt['selected_commit']==git('rev-parse','HEAD')
        assert receipt['current_objects']>=5 and receipt['parent_history_traversed']is False
        assert receipt['packed_objects_supported']is False
        assert receipt['metadata_directories']<=8 and receipt['object_directories']<=129
        for row in receipt['files']:
            mode=row['granted_mode']
            assert mode & 0o022==0 and mode & 0o700==row['original_mode'] & 0o700
            assert stat.S_IMODE(os.stat(row['path']).st_mode)==mode
            assert mode & 0o040
        object_rows=[row for row in receipt['files'] if row['kind']=='objects']
        assert object_rows and all(row['original_mode']==0o400 and row['granted_mode']==0o440 for row in object_rows)
        # A detached receipt or copied guard cannot reconstruct original custody.
        detached=guard.receipt;detached['selected_commit']='0'*40
        assert guard.receipt['selected_commit']!=detached['selected_commit']
        from copy import copy
        with pytest.raises(deployment.WorkerWorktreeObservationError):copy(guard).require_current()
    finally:guard.close()
    guard.close()
    with pytest.raises(deployment.WorkerWorktreeObservationError):guard.require_current()


@pytest.mark.parametrize('change',['outside-pointer','pointer-symlink','registration-symlink','head-symlink',
    'head-unsafe','other-branch','backlink','commondir','hardlink','group-write','canonical-mismatch','object-bytes','object-symlink'])
def test_real_git_read_grant_refuses_registration_or_object_drift_before_chmod(monkeypatch,tmp_path,change):
    import os
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    marker=worktree/'.git';head=registration/'HEAD'
    if change=='outside-pointer':marker.write_text('gitdir: '+str(tmp_path)+'\n')
    elif change=='pointer-symlink':marker.rename(worktree/'saved-marker');marker.symlink_to(worktree/'saved-marker')
    elif change=='registration-symlink':
        moved=registration.with_name('saved');registration.rename(moved);registration.symlink_to(moved,target_is_directory=True)
    elif change=='head-symlink':head.rename(registration/'saved-head');head.symlink_to(registration/'saved-head')
    elif change=='head-unsafe':head.write_text('ref: refs/heads/implementation/../master\n')
    elif change=='other-branch':head.write_text('ref: refs/heads/master\n')
    elif change=='backlink':(registration/'gitdir').write_text(str(repository/'.git')+'\n')
    elif change=='commondir':(registration/'commondir').write_text('../../../outside\n')
    elif change=='hardlink':os.link(head,registration/'extra-hardlink')
    elif change=='group-write':head.chmod(0o660)
    elif change=='canonical-mismatch':(repository/'.git/refs/heads/master').write_text('0'*40+'\n')
    else:
        identifier=git('rev-parse','HEAD');obj=repository/'.git/objects'/identifier[:2]/identifier[2:]
        if change=='object-bytes':obj.chmod(0o600);obj.write_bytes(b'not a loose Git object');obj.chmod(0o400)
        else:obj.rename(obj.with_name('saved-object'));obj.symlink_to(obj.with_name('saved-object'))
    before=head.lstat().st_mode
    with pytest.raises((deployment.WorkerWorktreeObservationError,OSError,ValueError)):
        deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    assert head.lstat().st_mode==before


@pytest.mark.parametrize('change',['branch-bytes','head-bytes','object-bytes','inode','mode'])
def test_real_retained_git_read_grant_detects_post_callback_changes(monkeypatch,tmp_path,change):
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    try:
        target=repository/'.git/refs/heads/implementation/owned'
        if change=='head-bytes':target=registration/'HEAD'
        elif change=='object-bytes':target=next(__import__('pathlib').Path(row['path']) for row in guard.receipt['files'] if row['kind']=='objects')
        if change=='inode':
            original=target.read_bytes();target.rename(target.with_name('saved'));target.write_bytes(original);target.chmod(0o640)
        elif change=='mode':target.chmod(0o660)
        else:target.chmod(0o640);target.write_bytes(b'changed selected metadata\n')
        with pytest.raises(deployment.WorkerWorktreeObservationError):guard.require_current()
    finally:
        # Verified STOP may retire the held original FDs after path/data drift.
        guard.close()


def test_current_git_loose_closure_does_not_grant_parent_history(monkeypatch,tmp_path):
    import os,stat,subprocess
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    old_commit=git('rev-parse','HEAD');old_blob=git('rev-parse','HEAD:caller.py')
    previous=os.umask(0o077)
    try:
        (repository/'caller.py').write_text('def caller(n):\n    return n + 3\n')
        git('add','caller.py');git('-c','user.name=Authored Fixture','-c','user.email=fixture@example.test','commit','-q','-m','current only')
        subprocess.check_call(['/usr/bin/git','-C',str(worktree),'-c','core.hooksPath=/dev/null','reset','--hard','master'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    finally:os.umask(previous)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    try:
        guard.require_current();selected={row['path'] for row in guard.receipt['files']}
        for identifier in (old_commit,old_blob):
            path=repository/'.git/objects'/identifier[:2]/identifier[2:]
            assert str(path) not in selected and stat.S_IMODE(path.stat().st_mode)==0o400
    finally:guard.close()


def test_packed_current_objects_fail_closed_without_pack_or_history_grant(monkeypatch,tmp_path):
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    before=(registration/'HEAD').stat().st_mode
    git('repack','-a','-d')
    with pytest.raises((deployment.WorkerWorktreeObservationError,OSError)):
        deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    assert (registration/'HEAD').stat().st_mode==before


@pytest.mark.parametrize('change',['objects','depth','blob-bytes'])
def test_current_git_object_size_count_and_tree_depth_bounds_precede_group_grant(monkeypatch,tmp_path,change):
    import os,subprocess
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    previous=os.umask(0o077)
    try:
        if change=='objects':
            for number in range(130):(repository/('item-'+str(number))).write_text(str(number)+' unique authored data\n')
        elif change=='depth':
            parent=repository
            for _ in range(18):parent=parent/'deep';parent.mkdir()
            (parent/'last').write_text('bounded depth fixture\n')
        else:(repository/'large').write_bytes(b'x'*(8*1024*1024+1))
        git('add','--all');git('-c','user.name=Authored Fixture','-c','user.email=fixture@example.test','commit','-q','-m','bound fixture')
        subprocess.check_call(['/usr/bin/git','-C',str(worktree),'-c','core.hooksPath=/dev/null','reset','--hard','master'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    finally:os.umask(previous)
    before=(registration/'HEAD').stat().st_mode
    with pytest.raises(deployment.WorkerWorktreeObservationError):
        deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    assert (registration/'HEAD').stat().st_mode==before


def test_retained_git_descriptor_swap_refuses_and_does_not_close_foreign_fd(monkeypatch,tmp_path):
    import os
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    held=guard._rows[0][1];os.close(held)
    foreign=tmp_path/'foreign';foreign.write_bytes(b'foreign descriptor fixture')
    opened=os.open(foreign,os.O_RDONLY)
    try:
        assert opened==held
        with pytest.raises(deployment.WorkerWorktreeObservationError):guard.require_current()
        with pytest.raises(deployment.WorkerWorktreeObservationError,match='close'):
            guard.close()
        assert os.pread(opened,256,0)==b'foreign descriptor fixture'
    finally:os.close(opened)


def test_git_read_resource_refuses_fork_identity_and_reconstructed_receipt(monkeypatch,tmp_path):
    import os
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    original=os.getpid()
    try:
        with pytest.raises(deployment.WorkerWorktreeObservationError):deployment.WorkerGitMetadataReadGuard([],guard.receipt)
        with monkeypatch.context() as patch:
            patch.setattr(deployment.os,'getpid',lambda:original+1)
            with pytest.raises(deployment.WorkerWorktreeObservationError):guard.require_current()
    finally:guard.close()


def test_native_instruction_git_read_callbacks_keep_exact_retained_metadata(monkeypatch,tmp_path):
    """Actual read helpers on disposable Git, without a native claim or grant."""
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_candidate_runner as doctor,local_planning_admission as local
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    (worktree/'caller.py').chmod(0o660);(worktree/'build/instruction.txt').chmod(0o660);(worktree/'.git').chmod(0o440)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    baseline=git('rev-parse','HEAD')
    manifest={'schema':local.INVENTORY_MANIFEST_SCHEMA,'baseline_commit':baseline,
        'sources':local._sources(repository,['caller.py','build/instruction.txt']),
        'created_outputs':[],'tasks':[{'outputs':[]}]}
    try:
        for root in (repository,worktree):
            for args in (('rev-parse','--show-toplevel'),('rev-parse','--path-format=absolute','--git-common-dir'),
                         ('rev-parse','HEAD'),('rev-parse',baseline+'^{commit}')):
                doctor._git(root,*args);guard.require_current()
            local.observe_public_inventory_manifest_sources(root,manifest,initial=True);guard.require_current()
    finally:guard.close()


@pytest.mark.parametrize('change',['directory-entry','file-bytes','file-inode','file-mode'])
def test_git_custody_diagnostic_identifies_exact_selected_path_and_stat_drift(monkeypatch,tmp_path,change):
    import json
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    secret=b'private authored bytes must not be exported'
    try:
        if change=='directory-entry':
            target=repository/'.git';(target/'authored-callback.lock').write_bytes(secret)
        else:
            target=repository/'.git/refs/heads/implementation/owned'
            if change=='file-inode':
                original=target.read_bytes();target.rename(target.with_name('original'));target.write_bytes(original);target.chmod(0o640)
            elif change=='file-mode':target.chmod(0o660)
            else:target.write_bytes(secret)
        with pytest.raises(deployment.WorkerWorktreeObservationError) as caught:guard.require_current()
        value=caught.value.git_metadata_drift
        # Earlier selected parent-directory identity can correctly fail first
        # on inode replacement; the diagnostic must name the actual failed row.
        selected={row[0] for row in guard._rows}
        observed=repository/value['relative_path']
        assert str(observed) in selected and value['path_role']=='canonical_git_metadata'
        assert value['schema']=='checked-worker-git-metadata-drift@1' and value['metadata_only']is True
        assert value['phase']=='retained' and value['changed_fields']
        keys=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
        expected=value['expected_stat'];actuals=[value[name] for name in ('current_fd_stat','current_path_stat','current_fresh_stat') if value[name]is not None]
        recomputed=sorted({key for actual in actuals for key in keys if expected[key]!=actual[key]})
        assert value['changed_fields']==recomputed and value['canonical_path_changed']is False
        assert value['expected_sha256']is value['observed_sha256']is None
        raw=json.dumps(value,sort_keys=True).encode()
        assert len(raw)<=16384 and secret not in raw and str(repository).encode()not in raw
        assert all(value[key]is False for key in ('execution_authority','proof_authority','completion_authority','source_body_executed'))
        assert type(value['provider_calls'])is int and value['provider_calls']==0
    finally:guard.close()


def test_git_drift_diagnostic_refuses_foreign_fd_without_reading_its_body(monkeypatch,tmp_path):
    import json,os
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    held=guard._rows[0][1];os.close(held)
    foreign=tmp_path/'private-foreign';foreign.write_bytes(b'never read this private authored body')
    opened=os.open(foreign,os.O_RDONLY)
    try:
        assert opened==held
        with pytest.raises(deployment.WorkerWorktreeObservationError)as caught:guard.require_current()
        value=caught.value.git_metadata_drift
        assert value['relative_path']=='.git' and value['path_role']=='allocated_git_pointer'
        assert 'st_ino'in value['changed_fields'] and value['phase']=='retained'
        assert value['expected_sha256']is value['observed_sha256']is None
        assert b'never read this private authored body'not in json.dumps(value).encode()
        with pytest.raises(deployment.WorkerWorktreeObservationError):guard.close()
    finally:os.close(opened)


def test_actual_dataset_snapshot_optional_git_write_is_prevented_without_stat_waiver(monkeypatch,tmp_path):
    import os
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import snapshot_repository
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    monkeypatch.delenv('GIT_OPTIONAL_LOCKS',raising=False)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    try:
        first=snapshot_repository(repository)
        with pytest.raises(deployment.WorkerWorktreeObservationError)as caught:guard.require_current()
        diagnostic=caught.value.git_metadata_drift
        assert diagnostic['relative_path']=='.git' and diagnostic['kind']=='directory'
        assert diagnostic['changed_fields']==['st_ctime_ns','st_mtime_ns']
    finally:guard.close()
    # A new original grant is created only after the known callback completed;
    # no old witness is rebased or accepted through the changed timestamps.
    original=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    before=[tuple(row[3])for row in original._rows]
    try:
        with original.readonly_query_environment():
            assert os.environ['GIT_OPTIONAL_LOCKS']=='0'
            second=snapshot_repository(repository)
            original.require_current()
        assert second.snapshot_cid==first.snapshot_cid and first.mode==second.mode=='git-clean'
        assert 'GIT_OPTIONAL_LOCKS'not in os.environ
        assert [deployment._git_stat(os.fstat(row[1]))for row in original._rows]==before
        original.require_current()
    finally:original.close()


@pytest.mark.parametrize('previous',[None,'0','1'])
def test_original_git_readonly_context_restores_exact_prior_environment(monkeypatch,tmp_path,previous):
    import os
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    if previous is None:monkeypatch.delenv('GIT_OPTIONAL_LOCKS',raising=False)
    else:monkeypatch.setenv('GIT_OPTIONAL_LOCKS',previous)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    try:
        with guard.readonly_query_environment():assert os.environ['GIT_OPTIONAL_LOCKS']=='0'
        assert os.environ.get('GIT_OPTIONAL_LOCKS')==previous
        guard.require_current()
    finally:guard.close()


def test_original_git_readonly_context_preserves_primary_exception_and_restores_after_flag_drift(monkeypatch,tmp_path):
    import os
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    monkeypatch.setenv('GIT_OPTIONAL_LOCKS','1')
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    error=ValueError('authored primary callback failure')
    try:
        with pytest.raises(ValueError)as caught:
            with guard.readonly_query_environment():
                os.environ['GIT_OPTIONAL_LOCKS']='1'
                raise error
        assert caught.value is error and os.environ['GIT_OPTIONAL_LOCKS']=='1'
        with guard.readonly_query_environment():assert os.environ['GIT_OPTIONAL_LOCKS']=='0'
    finally:guard.close()


def test_git_readonly_context_refuses_environment_drift_on_successful_callback(monkeypatch,tmp_path):
    import os
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    monkeypatch.delenv('GIT_OPTIONAL_LOCKS',raising=False)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    try:
        with pytest.raises(deployment.WorkerWorktreeObservationError,match='environment changed'):
            with guard.readonly_query_environment():os.environ['GIT_OPTIONAL_LOCKS']='1'
        assert 'GIT_OPTIONAL_LOCKS'not in os.environ
        guard.require_current()
    finally:guard.close()


@pytest.mark.parametrize('change',['directory-timestamps','file-content','directory-mode'])
def test_readonly_git_flag_cannot_waive_unrelated_filesystem_mutation(monkeypatch,tmp_path,change):
    import os
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    monkeypatch.delenv('GIT_OPTIONAL_LOCKS',raising=False)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    try:
        with pytest.raises(deployment.WorkerWorktreeObservationError):
            with guard.readonly_query_environment():
                if change=='directory-timestamps':
                    info=(repository/'.git').stat()
                    os.utime(repository/'.git',ns=(info.st_atime_ns,info.st_mtime_ns+1_000_000))
                elif change=='directory-mode':(repository/'.git').chmod(0o770)
                else:(repository/'.git/refs/heads/implementation/owned').write_bytes(b'changed selected ref')
        assert 'GIT_OPTIONAL_LOCKS'not in os.environ
    finally:guard.close()


def test_original_git_query_context_refuses_nested_copied_closed_and_rebound_uses(monkeypatch,tmp_path):
    import os
    from copy import copy
    deployment,repository,worktree,registration,git=_git_read_fixture(monkeypatch,tmp_path)
    monkeypatch.delenv('GIT_OPTIONAL_LOCKS',raising=False)
    guard=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    second=deployment.grant_worker_git_metadata_readonly(worktree=worktree,canonical_repository=repository)
    try:
        with pytest.raises(deployment.WorkerWorktreeObservationError):copy(guard).readonly_query_environment()
        with guard.readonly_query_environment():
            with pytest.raises(deployment.WorkerWorktreeObservationError):guard.readonly_query_environment()
            with pytest.raises(deployment.WorkerWorktreeObservationError):second.readonly_query_environment()
            with pytest.raises(deployment.WorkerWorktreeObservationError):guard.close()
        context=guard.readonly_query_environment()
        with monkeypatch.context()as patch:
            patch.setattr(deployment,'_GIT_OPTIONAL_QUERY_LOCK',object())
            with pytest.raises(deployment.WorkerWorktreeObservationError):
                with context:pytest.fail('rebound context yielded')
        assert 'GIT_OPTIONAL_LOCKS'not in os.environ
    finally:guard.close();second.close()
    with pytest.raises(deployment.WorkerWorktreeObservationError):guard.readonly_query_environment()


def test_observer_profile_sets_optional_git_read_flag_only_after_environment_scrub():
    from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
    source=deployment.WORKER_ENTRY
    flag="if profile['observer_only']:os.environ['GIT_OPTIONAL_LOCKS']='0'"
    assert source.count(flag)==1 and source.index('os.environ.clear()')<source.index(flag)
    assert source.index(flag)<source.index('if observation_arguments is not None:')
