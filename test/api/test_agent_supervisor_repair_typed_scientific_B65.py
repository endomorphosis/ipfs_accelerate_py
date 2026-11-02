"""B65 source-only observational seed transport; no health or authorization credit."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
from ipfs_accelerate_py.agent_supervisor.rescue import repair_authorization as auth
from scripts.ops.agent_supervisor import install_fleet_watchdog as installer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, sort_keys=True) + '\n').encode())
    return auth.descriptor(path)


@pytest.fixture
def current_contract(tmp_path, monkeypatch):
    state=tmp_path/'state';release=tmp_path/'release';release.mkdir()
    profile=write(tmp_path/'profile.json',{'source_only_fixture':True})
    git=write(tmp_path/'typed-Git.json',{'schema':installer.SCIENTIFIC_GIT_BINDING_SCHEMA,'profile':profile})
    science={'id':'legal-ir-scientific','hold_files':['OPERATOR_STOP'],'native_executor_binding':profile,'native_executor_git_binding':git}
    config={'state_dir':str(state),'runtime_release':str(release),
        'repair_worker':{'candidate_policy':{'base_commit':'a'*40}},
        'boards':[{'id':'legal-ir','hold_files':['HOLD'],'retained_attempts':3},science]}
    configpath=tmp_path/'fleet.json';config_ref=write(configpath,config)
    units={name:write(tmp_path/(name+'.service'),{'fixture':name})for name in('watchdog','repair')}
    initial=write(release/'repair-source-commitment.json',{'snapshot_commit':'a'*40,'source_commitment':'source-only',
        'sources':{},'selected_sources':{}})
    prior=write(state/'legacy-job.json',{'attempts':3,'history':'retained'})
    jobdir=state/'current';write(jobdir/'job.json',{'repair_source_successors':[]})
    base={'source_successor':None,'config':config_ref,'units':units,'initial_release':initial,'prior_job':prior}
    adoption=write(tmp_path/'adoption.json',base)
    observed={name:write(tmp_path/('observed-'+name+'.json'),{'owner':{'fixture':name}})for name in units}
    calls=[]
    def observe(seed,runner):calls.append(copy.deepcopy(seed));return {'loaded_runtime':observed}
    monkeypatch.setattr(auth,'repair_job_directory',lambda *a:jobdir)
    monkeypatch.setattr(owners,'require_current_authorization',lambda *a,**kw:None)
    monkeypatch.setattr(installer,'observe_repair_upgrade',observe)
    def reseal():
        current=write(configpath,config);base['config']=current
        return {'adoption':write(tmp_path/'adoption.json',base)}
    return dict(config=config,science=science,profile=profile,git=git,base=base,
        policy={'adoption':adoption},frozen={'binding':{'board_id':'legal-ir'}},calls=calls,reseal=reseal,prior=prior)


@pytest.mark.parametrize('reverse',[False,True])
def test_B65_reconstructs_exact_typed_science_binding_in_both_board_orders(current_contract,reverse):
    d=current_contract
    if reverse:d['config']['boards'].reverse()
    result=owners.current_profile_contract(d['reseal'](),d['frozen'])
    assert d['calls'][0]['scientific_executor_binding']==d['profile']
    assert d['calls'][0]['scientific_executor_git_binding']==d['git']
    assert result['prior_job']==d['prior'] and json.loads(Path(d['prior']['path']).read_bytes())['attempts']==3


def test_B65_orphan_typed_binding_refuses_before_readiness(current_contract):
    d=current_contract;d['science'].pop('native_executor_binding')
    with pytest.raises(ValueError,match='requires its exact profile'):
        owners.current_profile_contract(d['reseal'](),d['frozen'])
    assert d['calls']==[]


@pytest.mark.parametrize('keep_legacy_profile',[False,True])
def test_B65_legacy_observational_seed_shape_remains_unchanged(current_contract,keep_legacy_profile):
    d=current_contract;d['science'].pop('native_executor_git_binding')
    if not keep_legacy_profile:d['science'].pop('native_executor_binding')
    owners.current_profile_contract(d['reseal'](),d['frozen'])
    assert 'scientific_executor_binding'not in d['calls'][0]
    assert 'scientific_executor_git_binding'not in d['calls'][0]



def test_B65_later_authenticated_chain_uses_current_typed_stage_not_stale_anchor(current_contract,tmp_path,monkeypatch):
    d=current_contract;old_pin=d['base']['config'];old_profile=d['profile']
    profile=write(tmp_path/'next-profile.json',{'next_source_only_fixture':True})
    git=write(tmp_path/'next-Git.json',{'schema':installer.SCIENTIFIC_GIT_BINDING_SCHEMA,'profile':profile})
    d['science'].update(native_executor_binding=profile,native_executor_git_binding=git)
    d['config']['repair_worker']['candidate_policy']={'base_commit':'b'*40}
    current=write(Path(old_pin['path']),d['config'])
    previous={'schema':'agent-supervisor/repair-profile-upgrade-stage@1',
        'source_successor':{'successor_policy':d['config']['repair_worker']['candidate_policy']},
        'scientific_executor_binding':profile,'scientific_executor_git_binding':git}
    stage_ref=write(tmp_path/'previous-stage.json',previous)
    activation=write(tmp_path/'activation.json',{'stage':stage_ref});link=write(tmp_path/'source-link.json',{'activation':activation})
    state=Path(d['config']['state_dir']);write(state/'current/job.json',{'repair_source_successors':[link]})
    seen=[];monkeypatch.setattr(installer,'verify_repair_upgrade',lambda value,installed=False:seen.append((copy.deepcopy(value),installed)))
    result=owners.current_profile_contract(d['policy'],d['frozen'])
    assert seen==[(previous,True)] and d['calls']==[previous]
    assert d['calls'][0]['scientific_executor_binding']==profile!=old_profile
    assert result['config']==current!=old_pin and result['prior_job']==d['prior']
    assert json.loads(Path(d['prior']['path']).read_bytes())['attempts']==3
