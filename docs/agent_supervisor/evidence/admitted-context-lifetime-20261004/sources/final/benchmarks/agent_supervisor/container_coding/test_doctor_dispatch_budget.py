"""A provider-free candidate must not spend an unused model timeout reserve."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor


@pytest.mark.parametrize('route', ['doctor_candidate','doctor_contract_candidate','model_router'])
@pytest.mark.parametrize('used_work_seconds', [240,245])
def test_late_candidate_dispatch_respects_work_cutoff_and_keeps_model_reserve(
        tmp_path, monkeypatch, route, used_work_seconds):
    now=[1000.];alarms=[];entered=[];cleanup=[]
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver.os,'geteuid',lambda:1000)
    monkeypatch.setattr(driver.os,'umask',lambda *args:None)
    monkeypatch.setattr(driver.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(driver.signal,'signal',lambda *args:None)
    monkeypatch.setattr(driver.signal,'setitimer',lambda *args:alarms.append(args))
    monkeypatch.setattr(driver,'_failure_diagnostics',lambda *args,**kwargs:{})
    monkeypatch.setattr(driver,'_final_context_audit',lambda *args,**kwargs:None)
    monkeypatch.setattr(driver.subprocess,'run',lambda argv,**kwargs:
        cleanup.append(argv) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver.preparation,'prepare',lambda **kwargs:{'intent_preplanning':{}})
    monkeypatch.setattr(driver.preparation,'initial_context',lambda **kwargs:{})
    monkeypatch.setattr(driver.preparation,'plan',lambda **kwargs:{'qualified':True})
    monkeypatch.setattr(driver.preparation,'context',lambda **kwargs:{'context_bundle':{}})
    monkeypatch.setattr(driver,'verify_local_benchmark_admission',lambda *args,**kwargs:
        {'graph':SimpleNamespace(tasks=[SimpleNamespace(task_cid='fixture-task',task_key='fixture-key')]),
         'manifest':{'repository_cid':'fixture-repository'}})

    def candidate(**kwargs):
        now[0]+=used_work_seconds
        return dict(route=route,status='residual' if route=='model_router' else 'candidate_ready',
            provider_calls=0,artifact='fixture-candidate.json',sha256='a'*64,task_cid='fixture-task')
    monkeypatch.setattr(doctor,'prepare_terminal_doctor_dispatch',candidate)

    class NativeBoundaryReached(RuntimeError):pass
    def native(**kwargs):
        entered.append(kwargs)
        raise NativeBoundaryReached('controlled native boundary')
    for name,attrs in [
        ('benchmarks.agent_supervisor.container_coding.native_quack_qualification',{'open_existing_native_owner':native}),
        ('ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime',{'AdmittedBenchmarkRuntime':object}),
        ('ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy',{'GROK_CODEX_EXECUTION_MODE':'fixture-route'}),
    ]:
        module=ModuleType(name);vars(module).update(attrs);monkeypatch.setitem(sys.modules,name,module)
    state=tmp_path/'state/run';state.mkdir(parents=True);(state/'admission.json').write_text('{}')
    report=driver.run(instruction=tmp_path/'instruction.md',state=state,arm='full',source384_config=tmp_path/'config.json')
    admitted=route!='model_router' and used_work_seconds<245
    assert bool(entered)==admitted
    assert report['error']['type']==('NativeBoundaryReached' if admitted else 'TimeoutError')
    assert report['error_phase']==('native_execution' if admitted else 'implementation_setup')
    assert report['work_cutoff_seconds']==245 and report['reserved_cleanup_seconds']==40
    assert report['provider_invocations']==[] and report['task_completed'] is False
    assert report['worker_cleanup_returncode']==0 and len(cleanup)==1
    assert (driver.signal.ITIMER_REAL,245) in alarms and alarms[-1]==(driver.signal.ITIMER_REAL,0)
    assert json.loads((state.parent/'run-result.json').read_text())==report
