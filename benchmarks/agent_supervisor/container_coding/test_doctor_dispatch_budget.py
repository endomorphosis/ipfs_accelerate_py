"""A provider-free candidate must not spend an unused model timeout reserve."""
import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import (
    EXTENDED_SOURCE384_PROFILE, SOURCE384_PROFILE, admission_environment,
)


@pytest.mark.parametrize('route', ['doctor_candidate','doctor_contract_candidate','model_router'])
@pytest.mark.parametrize('remaining_work_seconds', [None,5,1,0])
@pytest.mark.parametrize(('budget_kwargs','total','cleanup_seconds','source_seconds'), [
    ({},285,40,90),
    ({'resource_profile':EXTENDED_SOURCE384_PROFILE},900,60,180),
])
def test_late_candidate_dispatch_respects_work_cutoff_and_keeps_model_reserve(
        tmp_path, monkeypatch, route, remaining_work_seconds, budget_kwargs,total,cleanup_seconds,source_seconds,
        native_failure=False):
    cutoff=total-cleanup_seconds
    used_work_seconds=0 if remaining_work_seconds is None else cutoff-remaining_work_seconds
    now=[1000.];alarms=[];entered=[];cleanup=[];native_options=[];context_options=[];native_cleanup=[]
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE',raising=False)
    for key,value in admission_environment(budget_kwargs.get('resource_profile')).items():
        monkeypatch.setenv(key,value)
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
    monkeypatch.setattr(driver.preparation,'initial_context',lambda **kwargs:context_options.append(kwargs) or {})
    monkeypatch.setattr(driver.preparation,'plan',lambda **kwargs:{'qualified':True})
    monkeypatch.setattr(driver.preparation,'context',lambda **kwargs:{'context_bundle':{}})
    monkeypatch.setattr(driver,'verify_local_benchmark_admission',lambda *args,**kwargs:
        {'graph':SimpleNamespace(tasks=[SimpleNamespace(task_cid='fixture-task',task_key='fixture-key')]),
         'manifest':{'repository_cid':'fixture-repository',
                     'sources':{driver.preparation.INSTRUCTION:{'sha256':'b'*64}}}})

    def candidate(**kwargs):
        now[0]+=used_work_seconds
        return dict(route=route,status='residual' if route=='model_router' else 'candidate_ready',
            provider_calls=0,artifact='fixture-candidate.json',sha256='a'*64,task_cid='fixture-task')
    monkeypatch.setattr(doctor,'prepare_terminal_doctor_dispatch',candidate)

    class NativeBoundaryReached(RuntimeError):pass
    def native(**kwargs):
        entered.append(kwargs)
        return nullcontext(SimpleNamespace(server=object(),source=object()))
    def create(*args,**kwargs):
        native_options.append(kwargs)
        if native_failure:
            def start():
                raise NativeBoundaryReached('controlled native boundary')
            def diagnostics():
                raise RuntimeError('PRIVATE_STARTUP_DIAGNOSTIC_FAILURE')
            return SimpleNamespace(start=start, startup_diagnostics=diagnostics,
                stop=lambda:native_cleanup.append('stop') or SimpleNamespace(to_dict=lambda:{'status':'succeeded'}),
                close=lambda:native_cleanup.append('close'), state=state, profile=object(),
                process=SimpleNamespace(snapshot=lambda profile:SimpleNamespace(members=[])))
        raise NativeBoundaryReached('controlled native boundary')
    for name,attrs in [
        ('benchmarks.agent_supervisor.container_coding.native_quack_qualification',{'open_existing_native_owner':native}),
        ('ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime',{'AdmittedBenchmarkRuntime':SimpleNamespace(create=create)}),
        ('ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy',{'GROK_CODEX_EXECUTION_MODE':'fixture-route'}),
        ('ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction',{'prepare_public_instruction_context':lambda **kwargs:
            {'artifact':'fixture-instruction.json','sha256':'c'*64}}),
    ]:
        module=ModuleType(name);vars(module).update(attrs);monkeypatch.setitem(sys.modules,name,module)
    state=tmp_path/'state/run';state.mkdir(parents=True);(state/'admission.json').write_text('{}')
    report=driver.run(instruction=tmp_path/'instruction.md',state=state,arm='full',source384_config=tmp_path/'config.json',**budget_kwargs)
    reached_native=(route!='model_router' or remaining_work_seconds is None) and used_work_seconds<cutoff
    extended = budget_kwargs.get('resource_profile') == EXTENDED_SOURCE384_PROFILE
    admitted=reached_native and not (extended and remaining_work_seconds == 1)
    assert bool(entered)==admitted
    assert report['error']['type']==('NativeBoundaryReached' if admitted else 'TimeoutError')
    assert report['error_phase']==('native_execution' if reached_native else 'implementation_setup')
    assert report['work_cutoff_seconds']==cutoff and report['reserved_cleanup_seconds']==cleanup_seconds
    assert report['max_total_agent_seconds']==total
    assert report['source384_timeout_seconds']==source_seconds
    assert report['native_start_timeout_seconds'] == (120 if extended else 20)
    assert context_options[0]['source384_timeout_seconds']==source_seconds
    if admitted:
        assert native_options[0]['refresh_source384_on_completion'] is extended
        assert native_options[0]['lifetime_seconds']==min(900,max(120,cutoff-used_work_seconds+60))
        assert native_options[0]['timeout_ms']==20_000
        if extended:
            assert native_options[0]['start_timeout_ms'] == min(120,cutoff-used_work_seconds)*1000
        else:
            assert 'start_timeout_ms' not in native_options[0]
        assert report['native_start_timeout_ms'] == native_options[0].get('start_timeout_ms',20_000)
        assert report['native_stop_timeout_ms'] == 20_000
        if route=='model_router':
            import shlex
            command=shlex.split(native_options[0]['implementation_command'])
            assert int(command[command.index('--timeout')+1])==min(600,cutoff-25)
    assert report['provider_invocations']==[] and report['task_completed'] is False
    assert report['worker_cleanup_returncode']==0 and len(cleanup)==1
    assert (driver.signal.ITIMER_REAL,cutoff) in alarms and alarms[-1]==(driver.signal.ITIMER_REAL,0)
    assert json.loads((state.parent/'run-result.json').read_text())==report
    if native_failure:
        assert native_cleanup == ['stop','close']
        assert report['native_startup_error'] == 'collection_unavailable'
        assert report['stop']['status'] == 'succeeded' and report['remaining_processes'] == 0
        assert 'PRIVATE' not in json.dumps(report)


def test_startup_diagnostic_failure_does_not_mask_start_failure_or_skip_stop(tmp_path,monkeypatch):
    test_late_candidate_dispatch_respects_work_cutoff_and_keeps_model_reserve(
        tmp_path,monkeypatch,'doctor_contract_candidate',None,
        {'resource_profile':EXTENDED_SOURCE384_PROFILE},900,60,180,native_failure=True)


def _startup_observation():
    return dict(schema='admitted-native-startup-observation@1',start_timeout_ms=120000,
        stop_timeout_ms=20000,bootstrap_wait_seconds=120,
        observations=[dict(phase='control_validation',status='completed',seconds=2.5)],
        observations_truncated=False,bootstrap_receipt_count=1,bootstrap_error_count=0)


def test_startup_metadata_projection_detaches_bounded_observations():
    raw=_startup_observation()
    result=driver._project_native_startup(raw)
    assert result==raw
    result['observations'][0]['seconds']=99
    assert raw['observations'][0]['seconds']==2.5


@pytest.mark.parametrize('change',['source_body','row_body','phase','status','seconds','too_many','count','start_budget'])
def test_startup_projection_refuses_raw_or_unbounded_metadata(change):
    value=_startup_observation()
    if change=='source_body':value['source']='PRIVATE_SOURCE'
    elif change=='row_body':value['observations'][0]['error']='PRIVATE_EXCEPTION'
    elif change=='phase':value['observations'][0]['phase']='PRIVATE_PATH'
    elif change=='status':value['observations'][0]['status']='PRIVATE_MODEL'
    elif change=='seconds':value['observations'][0]['seconds']=float('inf')
    elif change=='too_many':value['observations']*=17
    elif change=='count':value['bootstrap_error_count']=True
    else:value['start_timeout_ms']=120001
    with pytest.raises(ValueError) as error:
        driver._project_native_startup(value)
    assert 'PRIVATE' not in str(error.value)


@pytest.mark.parametrize(('profile','timeout'), [
    (None,True),(None,285.0),(None,89),(None,301),(SOURCE384_PROFILE,900),
    (EXTENDED_SOURCE384_PROFILE,901),(EXTENDED_SOURCE384_PROFILE,90.0),
])
def test_driver_rejects_unbounded_or_wrong_profile_budget_before_side_effects(tmp_path,monkeypatch,profile,timeout):
    monkeypatch.setattr(driver.os,'umask',lambda *args:pytest.fail('invalid budget reached setup'))
    with pytest.raises(ValueError,match='bounded arm'):
        driver.run(instruction=tmp_path/'instruction',state=tmp_path/'state',arm='full',
                   resource_profile=profile,timeout_seconds=timeout)


@pytest.mark.parametrize(('profile','timeout'),[(None,300),(SOURCE384_PROFILE,300),
    (EXTENDED_SOURCE384_PROFILE,900),(EXTENDED_SOURCE384_PROFILE,120)])
def test_explicit_bounded_override_preserves_owner_identity_gate(tmp_path,monkeypatch,profile,timeout):
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE',raising=False)
    for key,value in admission_environment(profile).items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(driver.os,'geteuid',lambda:0)
    with pytest.raises(ValueError,match='deployed private supervisor owner'):
        driver.run(instruction=tmp_path/'instruction',state=tmp_path/'state',arm='full',
                   resource_profile=profile,timeout_seconds=timeout)


@pytest.mark.parametrize('change',['missing-profile','wrong-profile','missing-ledger','different-ledger','undeclared-profile'])
def test_selected_budget_requires_matching_explicit_admission_environment(tmp_path,monkeypatch,change):
    selected=admission_environment(EXTENDED_SOURCE384_PROFILE)
    for key,value in selected.items():monkeypatch.setenv(key,value)
    profile=EXTENDED_SOURCE384_PROFILE
    if change=='missing-profile':monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE')
    elif change=='wrong-profile':monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE','other@1')
    elif change=='missing-ledger':monkeypatch.delenv('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH')
    elif change=='different-ledger':monkeypatch.setenv('IPFS_DATASETS_RESOURCE_SCHEDULER_PATH','/tmp/other.json')
    else:profile=None
    monkeypatch.setattr(driver.os,'umask',lambda *args:pytest.fail('profile mismatch reached setup'))
    with pytest.raises(ValueError,match='admission profile differs'):
        driver.run(instruction=tmp_path/'instruction',state=tmp_path/'state',arm='full',resource_profile=profile)
