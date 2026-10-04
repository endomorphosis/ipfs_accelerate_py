"""Source-free failure snapshots; all scheduler observations are controlled."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
import threading
from types import ModuleType

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_resource_diagnostics as diagnostics
from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualification
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import proof_resource_safety as resources


def snapshot():
    return dict(capacity=dict(cpu_slots=5, memory_mb=12288, usable_memory_mb=9830,
        reserved_memory_mb=2458, gpu_memory_mb=None, usable_gpu_memory_mb=None,
        reserved_gpu_memory_mb=0, unified_memory_mb=None, child_process_slots=64),
        allocated=dict(cpu_slots=1, memory_mb=6144), active_lease_count=1,
        active_root_lease_count=1, active_child_lease_count=0, waiting_request_count=0,
        proof_backoff=dict(until=1800000000.25, reason="proof_memory_headroom"),
        proof_recovery=dict(phase="settling", healthy_samples=1, grants_remaining=4,
            next_sample_at=1800000000.5, next_grant_at=0.0))


@pytest.fixture
def facade(monkeypatch):
    class Owner:
        def __init__(self):
            self.calls=0
            self.value=snapshot()
        def snapshot(self):
            self.calls+=1
            return self.value
    owner=Owner();module=ModuleType(diagnostics.MODULE)
    module.GlobalResourceScheduler=Owner
    module._GLOBAL_SCHEDULERS={"PRIVATE_STATE_PATH":owner}
    module._GLOBAL_SCHEDULERS_LOCK=threading.Lock()
    def forbidden(*args,**kwargs):
        pytest.fail("diagnostics selected a new scheduler or configuration")
    module.get_global_resource_scheduler=forbidden
    module.configure_global_resource_scheduler=forbidden
    monkeypatch.setitem(sys.modules,diagnostics.MODULE,module)
    return owner,module


def test_exact_whitelist_excludes_labels_keys_paths_and_source(facade):
    owner,module=facade;expected=deepcopy(owner.value)
    secret={"source":"PRIVATE_SOURCE", "state_path":"PRIVATE_PATH", "lease_key":"PRIVATE_KEY",
            "labels":["PRIVATE_LABEL"]*10000}
    owner.value.update(secret)
    for field in ("capacity","allocated","proof_backoff","proof_recovery"):
        owner.value[field].update(secret)
    result=diagnostics.collect_failure_scheduler()
    assert result["status"]=="observed" and owner.calls==1
    assert result["observation_boundary"]=="after_unwind"
    assert result["exact_admission_decision"] is result["causal_proof"] is False
    assert result["native_snapshot_may_recover_stale_owners"] is True
    for key,value in expected.items():assert result[key]==value
    encoded=json.dumps(result,allow_nan=False)
    assert len(encoded.encode())<=diagnostics.MAX_BYTES and "PRIVATE" not in encoded
    assert not module._GLOBAL_SCHEDULERS_LOCK.locked()


def test_empty_recovery_backoff_does_not_invent_a_refusal(facade):
    owner,_=facade;owner.value.update(proof_backoff={},proof_recovery={})
    result=diagnostics.collect_failure_scheduler()
    assert result["proof_backoff"]==result["proof_recovery"]=={}


def test_unknown_reasons_and_phases_are_not_exported(facade):
    owner,_=facade
    owner.value["proof_backoff"]["reason"]="PRIVATE_UNKNOWN_REASON"*10000
    owner.value["proof_recovery"]["phase"]="PRIVATE_UNKNOWN_PHASE"*10000
    result=diagnostics.collect_failure_scheduler()
    assert result["proof_backoff"]["reason"]==result["proof_recovery"]["phase"]=="unrecognized"
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("bad",[True,-1,2**100,float("nan"),float("inf"),"PRIVATE",[],{}])
def test_malformed_primitive_fields_fail_open_without_export(facade,bad):
    owner,_=facade;owner.value["capacity"]["cpu_slots"]=bad
    result=diagnostics.collect_failure_scheduler()
    assert result["status"]=="unavailable" and result["reason"]=="snapshot_unavailable"
    assert "capacity" not in result and "PRIVATE" not in json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("problem",["unloaded","layout","missing","ambiguous","busy","custom_owner","missing_fields","snapshot_error","snapshot_timeout"])
def test_unavailable_existing_owner_never_creates_or_reconfigures(facade,monkeypatch,problem):
    owner,module=facade
    if problem=="unloaded":monkeypatch.delitem(sys.modules,diagnostics.MODULE)
    elif problem=="layout":module._GLOBAL_SCHEDULERS=[]
    elif problem=="missing":module._GLOBAL_SCHEDULERS={}
    elif problem=="ambiguous":module._GLOBAL_SCHEDULERS["OTHER_PRIVATE_PATH"]=module.GlobalResourceScheduler()
    elif problem=="busy":module._GLOBAL_SCHEDULERS_LOCK.acquire()
    elif problem=="custom_owner":module._GLOBAL_SCHEDULERS={"PRIVATE":object()}
    elif problem=="missing_fields":owner.value={}
    else:
        def failed():raise (TimeoutError if problem=="snapshot_timeout" else ValueError)("PRIVATE_DIAGNOSTIC")
        monkeypatch.setattr(owner,"snapshot",failed)
    try:
        result=diagnostics.collect_failure_scheduler()
        assert result["status"]=="unavailable" and "PRIVATE" not in json.dumps(result)
        if problem not in {"missing_fields","snapshot_error","snapshot_timeout"}:assert owner.calls==0
    finally:
        if problem=="busy":module._GLOBAL_SCHEDULERS_LOCK.release()


def test_driver_failure_keeps_primary_traceback_when_snapshot_or_sample_unavailable(facade,monkeypatch):
    owner,_=facade
    def unavailable():raise TimeoutError("PRIVATE_TELEMETRY")
    monkeypatch.setattr(owner,"snapshot",unavailable)
    monkeypatch.setattr(resources,"collect_proof_host_resources",unavailable)
    try:raise ValueError("primary task error")
    except ValueError as error:
        result=driver._failure_diagnostics(error,phase="initial_context")
    assert result["error_phase"]=="initial_context"
    assert result["error_traceback"]["frames"]
    assert result["failure_resource_error"]=="TimeoutError"
    assert result["failure_scheduler"]["reason"]=="snapshot_unavailable"
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("diagnostic_failure",[False,True])
def test_actual_context_probe_failure_preserves_result_and_disarms_timer(tmp_path,facade,monkeypatch,diagnostic_failure):
    from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source
    output=tmp_path/"result.json";signals=[]
    monkeypatch.setattr(signal,"signal",lambda *args:None)
    monkeypatch.setattr(signal,"setitimer",lambda *args:signals.append(args))
    monkeypatch.setattr(preparation,"prepare",lambda **kwargs:{})
    primary=ValueError("primary context error")
    def initial(**kwargs):raise primary
    monkeypatch.setattr(preparation,"initial_context",initial)
    monkeypatch.setattr(source,"_pins",lambda:{"authored_fixture":True})
    monkeypatch.setattr(resources,"collect_proof_host_resources",lambda:resources.ProofHostResources(5,12288,8192))
    if diagnostic_failure:
        def failed():raise TimeoutError("PRIVATE_HELPER_ERROR")
        monkeypatch.setattr(diagnostics,"collect_failure_scheduler",failed)
    native_open=Path.open;native_read=Path.read_text
    def opened(path,*args,**kwargs):
        return native_open(output if str(path)==qualification.RESULT_PATH else path,*args,**kwargs)
    def read(path,*args,**kwargs):
        if str(path)=="/opt/ipfs-supervisor/source384-public-instruction.md":return "authored public instruction"
        return native_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",opened);monkeypatch.setattr(Path,"read_text",read)
    with pytest.raises(SystemExit) as exited:exec(compile(qualification.CONTEXT_PROBE,"context-probe","exec"),{})
    assert exited.value.code==1
    result=json.loads(output.read_bytes())
    assert result["error_type"]=="ValueError" and result["error"]==str(primary)
    assert result["error_phase"]=="initial_context" and result["qualified"] is False
    assert result["provider_calls"]==0 and result["source_qualified_proof_claimed"] is False
    assert result["failure_resources"]["available_memory_mb"]==8192
    assert (signal.ITIMER_REAL,0) in signals and "PRIVATE" not in json.dumps(result)
    if diagnostic_failure:assert result["failure_scheduler_error"]=="collection_unavailable"
    else:assert result["failure_scheduler"]["proof_backoff"]["reason"]=="proof_memory_headroom"


def test_helper_has_no_factory_or_heavy_success_path_import():
    import ast
    tree=ast.parse(Path(diagnostics.__file__).read_text())
    assert not any(isinstance(n,ast.ImportFrom) for n in ast.walk(tree))
    called={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert not {"get_global_resource_scheduler","configure_global_resource_scheduler"}&called
    assert "terminal_container_supervisor" not in qualification.CONTEXT_PROBE
