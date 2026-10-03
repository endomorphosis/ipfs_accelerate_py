"""Failure retention and opt-in dispatch; these do not claim live admission."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from benchmarks.agent_supervisor.container_coding import native_repository_finite_supervision as driver


def test_managed_driver_keeps_primary_failure_and_cleanup_failure_artifact(tmp_path,monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_pipeline_resources as pipeline
    from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as finite
    monkeypatch.setattr(finite,'seal_finite_integer_tools',lambda **kw: {})
    class FailedAdmission:
        def __init__(self,*args,**kwargs):pass
        @contextmanager
        def reserve(self,**kwargs):
            assert kwargs['budget'].cpu_slots==3 and kwargs['budget'].memory_mb==3072
            assert kwargs['policy'].protected_memory_mb==1024
            assert kwargs['policy'].protected_disk_bytes==640*1024**2
            assert kwargs['budget'].disk_bytes==1536*1024**2
            raise RuntimeError('primary managed admission failure')
            yield
    class FailedCleanup:
        def enter_context(self,context):return context.__enter__()
        def close(self):raise RuntimeError('retained cleanup failure')
    monkeypatch.setattr(pipeline,'RepositoryPipelineResources',FailedAdmission)
    monkeypatch.setattr(driver,'ExitStack',FailedCleanup)
    output=tmp_path/'qualification'
    result=driver.qualify(output=output,python=Path(sys.executable),lean=Path('/unselected/lean'),pipeline_resources=True)
    saved=json.loads((output/'result.json').read_text())
    assert saved==result
    assert not result['qualified']
    assert result['error']['message']=='primary managed admission failure'
    assert result['cleanup_errors'][0]['message']=='retained cleanup failure'
    assert result['full_trial_resources_selected'] and result['pipeline_resources_selected']
    assert result['provider_calls']==result['training_steps']==0
    profile=result['managed_profile']
    ordinary=profile['expected_ordinary_high_level_phases']*profile['high_level_write_ceiling_bytes']
    ordinary+=profile['expected_ordinary_preparation_phases']*profile['preparation_write_ceiling_bytes']
    protected=profile['expected_protected_high_level_phases']*profile['high_level_write_ceiling_bytes']
    protected+=profile['expected_protected_preparation_phases']*profile['preparation_write_ceiling_bytes']
    assert ordinary==profile['maximum_declared_ordinary_disk_bytes']<profile['disk_bytes']-profile['protected_disk_bytes']
    assert protected==profile['maximum_declared_protected_disk_bytes']<profile['protected_disk_bytes']


def test_pipeline_cli_is_explicit_and_legacy_full_flag_is_distinct(tmp_path,monkeypatch,capsys):
    requests=[]
    def observe(**kwargs):
        requests.append(kwargs)
        return dict(qualified=False,seconds=0,provider_calls=0)
    monkeypatch.setattr(driver,'qualify',observe)
    base=['native-driver','--output',str(tmp_path/'fresh'),'--python',sys.executable,'--lean','/unselected/lean']
    monkeypatch.setattr(sys,'argv',base+['--full-trial-resources'])
    assert driver.main()==1 and requests[-1]['full_trial_resources'] and not requests[-1]['pipeline_resources']
    monkeypatch.setattr(sys,'argv',base+['--pipeline-resources'])
    assert driver.main()==1 and requests[-1]['pipeline_resources'] and not requests[-1]['full_trial_resources']


def test_managed_driver_retains_primary_error_when_both_receipt_observers_fail(tmp_path,monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_pipeline_resources as pipeline
    from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as finite
    monkeypatch.setattr(finite,'seal_finite_integer_tools',lambda **kw: {})
    class FailedParentObservation:
        def remaining(self):raise RuntimeError('primary remaining failure')
        def receipt(self):raise RuntimeError('host receipt unavailable')
    class FailedPipelineObservation:
        parent=FailedParentObservation()
        def receipt(self):raise RuntimeError('pipeline receipt unavailable')
    class ControlledOwner:
        def __init__(self,*args,**kwargs):pass
        @contextmanager
        def reserve(self,**kwargs):
            yield FailedPipelineObservation()
    monkeypatch.setattr(pipeline,'RepositoryPipelineResources',ControlledOwner)
    output=tmp_path/'failed-receipts'
    result=driver.qualify(output=output,python=Path(sys.executable),lean=Path('/unselected/lean'),pipeline_resources=True)
    assert json.loads((output/'result.json').read_text())==result
    assert result['error']['message']=='primary remaining failure' and not result['qualified']
    assert {item['message'] for item in result['cleanup_errors']}=={
        'host receipt unavailable','pipeline receipt unavailable'}
