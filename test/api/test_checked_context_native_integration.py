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
