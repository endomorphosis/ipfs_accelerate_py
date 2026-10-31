"""Explicit host-input metadata qualification of the actual current renderer.

No copied source body is imported or executed. Readiness/authorization remains
modeled by the preserved owning fixture; this proves the real renderer's policy
DATA self-reference shape, not a runtime enrollment or a completed validation.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
from ipfs_accelerate_py.agent_supervisor.rescue import repair_authorization as auth
from scripts.ops.agent_supervisor import install_fleet_watchdog as installer

def pin(path):
    raw=path.read_bytes()
    return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def write(path,value,readonly=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    raw=value if type(value)is bytes else (json.dumps(value,sort_keys=True)+"\n").encode()
    path.write_bytes(raw);path.chmod(0o400 if readonly else 0o600)
    return pin(path)


def bootstrap_scope(tmp_path,monkeypatch):
    home=tmp_path/"home";source=tmp_path/"Z";release=home/".local/lib/releases/frozen";state=home/".local/state/legal-ir"
    source.mkdir();state.mkdir(parents=True);release.mkdir(parents=True)
    subject=source/"subject.py";write(subject,b"value = True\n");released=release/"subject.py";write(released,subject.read_bytes())
    ownfile=release/"ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py";write(ownfile,b"# fixture owner source\n")
    cp={"schema":"agent-supervisor/isolated-repair-candidate-policy@1","repository_root":str(source),"state_root":str(state),"namespace":"ptr_lane_0","base_commit":"a"*40,"source_paths":["subject.py"],"allowed_paths":["subject.py"],"max_source_bytes":4096,"max_candidate_bytes":4096,"max_files":4,"git_timeout_seconds":10}
    manifest={"subject.py":{**{k:v for k,v in pin(subject).items() if k!="path"},"executable":False}}
    snapshot={"prior_policy":cp,"new_commit":"a"*40,"new_tree":"b"*40,"source_commitment":auth.digest({"base_commit":"a"*40,"manifest":manifest}),"manifest":manifest}
    sources={}
    for key in ("fleet_watchdog","fleet_repair"):
        p=release/"ipfs_accelerate_py/agent_supervisor/rescue"/(key+".py");write(p,b"# fixture runtime\n");sources[key]=pin(p)
    initial=write(release/"repair-source-commitment.json",{"schema":"agent-supervisor/repair-source-release@1","source_commitment":snapshot["source_commitment"],"sources":sources,"snapshot_commit":snapshot["new_commit"],"snapshot_tree":snapshot["new_tree"],"selected_sources":{"subject.py":pin(released)}},True)
    configpath=home/".config/ipfs-taskboard-watchdog/legal-ir/fleet.json"
    old={"state_dir":str(state),"runtime_release":"/old/release","repair_worker":{"cwd":"/old/Y","job_unit":"fixture-job.service","timeout_seconds":600},"boards":[{"id":"legal-ir","hold_files":["HOLD"],"probe":{"argv":["probe"]},"repair":{"argv":["repair"]}},{"id":"legal-ir-scientific","hold_files":["OPERATOR_STOP"],"probe":{"argv":["probe"]},"repair":{"argv":["repair"]}}]}
    before=write(configpath,old)
    unitposts={};unitbefore={};unitpaths={}
    for suffix in ("watchdog","repair"):
        p=home/".config/systemd/user"/("ipfs-taskboard-legal-ir-"+suffix+".service")
        unitbefore[suffix]=write(p,b"original unit\n")
        write(p,("new immutable "+suffix+" unit\n").encode())
        unitpaths[suffix]=p;unitposts[suffix]={**pin(p),"mode":stat.S_IMODE(p.stat().st_mode)}
    inventory=write(home/".config/ipfs-taskboard-watchdog/legal-ir/inventory.json",{"fixture":True},True)
    prior=write(state/"repairs/legal-ir/job.json",{"board_id":"legal-ir","attempts":3,"status":"verified_healthy","escalation_reason":"repair_budget_exhausted"},True)
    epoch={"schema":auth.SCHEMA,"epoch_id":"fixture-finite-epoch","objective_id":"new-explicit-objective","board_id":"legal-ir","board_namespace":"ptr_lane_0","candidate_policy_sha256":auth.digest(cp),"max_attempts":3,"prior_job":prior,"prior_incident_signature_sha256":"d"*64,"not_before_observed_at":1000,"expires_at":2000}
    policy_path=tmp_path/"owner-policy.json";token=owners.INITIAL_PROFILE_POLICY_HASH
    def command(action,adoption=False):
        argv=[sys.executable,"-B",str(ownfile),action,"--policy",str(policy_path),"--policy-sha256",token,"--freeze-path","{freeze_path}","--receipt-path","{receipt_path}"]
        if adoption:argv+=["--validation-path","{validation_path}"]
        return argv
    runtime={"schema":"agent-supervisor/repair-candidate-runtime@1","source_descriptors":[pin(subject)],"validation_owner":{"argv":command("validate"),"verify_argv":command("verify-validation"),"source_descriptors":[pin(ownfile)],"timeout_seconds":30},"adoption_owner":{"argv":command("adopt",True),"verify_argv":command("verify-adoption"),"source_descriptors":[pin(ownfile)],"timeout_seconds":30},"resource_bounds":{"memory_max_bytes":1024**3,"cpu_quota_percent":100,"tasks_max":16,"runtime_max_seconds":600},"checkout_timeout_seconds":10}
    projection=copy.deepcopy(old);projection["runtime_release"]=str(release);projection["repair_worker"].update(candidate_policy=cp,candidate_runtime=runtime,cwd=str(release))
    projection["boards"][0]["repair_authorization_epoch"]=epoch
    for b in projection["boards"]:
        for key in ("probe","repair"):b[key]["env"]={"PYTHONPATH":str(release)+":"+str(source)}
    base={"schema":"agent-supervisor/repair-profile-upgrade-contract@1","profile":"legal-ir","config":before,"inventory":inventory,"units":unitbefore,"prior_job":prior,"expected_loaded_before":{"watchdog":{"pid":10,"birth":"10"},"repair":{"pid":11,"birth":"11"}},"job_unit":"fixture-job.service","source_successor":None,"initial_release":initial}
    anchor={"schema":owners.INITIAL_PROFILE_BOOTSTRAP_SCHEMA,"owner_policy_path":str(policy_path),"original_contract":base,"source_snapshot":snapshot,"expected_config_projection":projection,"expected_unit_postimages":unitposts}
    anchorpath=tmp_path/"bootstrap-anchor.json"
    policy={"schema":owners.POLICY_SCHEMA,"datasets_root":str(tmp_path/"D"),"python":sys.executable,"source_descriptors":[],"validation":{},"adoption":{}}
    data={"anchor":anchor,"anchorpath":anchorpath,"policy":policy,"policy_path":policy_path,"configpath":configpath,"base":base,"projection":projection,"snapshot":snapshot,"source":source,"subject":subject,"release":release,"unitpaths":unitpaths,"seen":[],"frozen":{"binding":{"board_id":"legal-ir"}}}
    def reseal():
        policy["adoption"]=write(anchorpath,anchor,True)
        reference=write(policy_path,policy,True)
        current=copy.deepcopy(projection)
        for role in ("validation_owner","adoption_owner"):
            for key in ("argv","verify_argv"):
                argv=current["repair_worker"]["candidate_runtime"][role][key];argv[argv.index("--policy-sha256")+1]=reference["sha256"]
        write(configpath,current);data["config"]=current;data["checked"]=reference
    data["reseal"]=reseal;reseal()
    jobdir=state/"repair-epochs/fixture-finite-epoch/repairs/legal-ir";write(jobdir/"job.json",{"repair_source_successors":[]})
    data["jobdir"]=jobdir
    monkeypatch.setattr(auth,"repair_job_directory",lambda state,board:jobdir)
    monkeypatch.setattr(owners,"require_current_authorization",lambda *args,**kwargs:None)
    observations={}
    for suffix in ("watchdog","repair"):
        observations[suffix]=write(tmp_path/("observed-"+suffix+".json"),{"owner":{"pid":200 if suffix=="watchdog" else 201,"birth":"200"}},True)
    data["observations"]=observations
    def observe(stage,runner):
        data["seen"].append(copy.deepcopy(stage))
        installer.verify_repair_upgrade(stage,installed=True)
        return {"loaded_runtime":observations}
    monkeypatch.setattr(installer,"observe_repair_upgrade",observe)
    return data


def invoke(data):
    return owners.current_profile_contract(data["policy"],data["frozen"],checked_policy_reference=data["checked"])



def test_real_current_control_renderer_bootstraps_exact_nested_policy_DATA_refs(tmp_path,monkeypatch):
    d=bootstrap_scope(tmp_path,monkeypatch)
    inputs=json.loads((Path(__file__).with_name('current-control-renderer-inputs-v01.json')).read_bytes())
    source_root=Path(owners.__file__).parents[3]
    private=tmp_path/'immutable-control';rows=[]
    relatives=('ipfs_accelerate_py/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py',
        'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py',
        'test/api/test_agent_supervisor_repair_initial_profile_bootstrap.py')
    for relative in relatives:
        original=source_root/relative
        target=private/relative
        rows.append(write(target,original.read_bytes(),True))
        assert target.read_bytes()==original.read_bytes()
    manifest=write(private/'control-manifest.json',{'schema':owners.CONTROL_SCHEMA,
        'root':str(private),'files':sorted(rows,key=lambda row:row['path']),
        'namespace_profiles':[]},True)
    private_owner=private/relatives[4];private_worker=private/relatives[5];trusted=private/relatives[6]
    assert pin(private_owner)['sha256']==pin(Path(owners.__file__))['sha256']
    assert pin(private_worker)['sha256']==pin(source_root/relatives[5])['sha256']
    for reference in inputs['public_source_descriptors']:
        assert pin(Path(reference['path']))==reference
    pytest_source=Path(pytest.__file__)
    pytest_support=pytest_source.parent.parent/'_pytest/python.py'
    d['policy'].update(datasets_root=inputs['datasets_root'],
        source_descriptors=[pin(private_owner),pin(private_worker),*inputs['public_source_descriptors']],
        control_release={'root':str(private),'manifest':manifest},
        validation={'trusted_tests':[pin(trusted)],
            'selected_checks':[str(trusted)+'::test_bootstrap_breaks_hash_cycle_and_builds_complete_actual_observational_stage'],
            'pytest_module':pin(pytest_source),'import_sources':[pin(pytest_support)],
            'attempt_root':str(tmp_path/'original-attempts'),'cpu_slots':1,'memory_mb':1536,
            'storage_bytes':33554432,'timeout_seconds':300,'child_process_slots':2,
            'scheduler_state_path':str(tmp_path/'scheduler-metadata-not-opened.json'),
            'expected_scheduler_policy_sha256':'a'*64,'roots':[str(tmp_path)],
            'resource_ledger':str(tmp_path/'ledger-metadata-not-opened.json')})
    # Use the real current pure renderer, without replacing checked_policy or
    # checked_control_release. Its complete-file private manifest remains real,
    # sealed input DATA; no copied control module is imported or invoked.
    preliminary=write(d['policy_path'],d['policy'],True)
    first=installer.render_control_owner_commands(preliminary,
        validation_timeout_seconds=30,adoption_timeout_seconds=30)
    current=copy.deepcopy(d['config'])
    current['repair_worker']['candidate_runtime'].update(first)
    projection=owners._bootstrap_config_projection(current,d['policy'],preliminary,str(d['policy_path']))
    assert len(owners._bootstrap_marker_paths(projection))==8
    d['projection'].clear();d['projection'].update(projection)
    d['reseal']()
    rendered=installer.render_control_owner_commands(d['checked'],
        validation_timeout_seconds=30,adoption_timeout_seconds=30)
    d['config']['repair_worker']['candidate_runtime'].update(rendered)
    write(d['configpath'],d['config'])
    protected={name:pin(path)for name,path in {'config':d['configpath'],
        'policy':d['policy_path'],**d['unitpaths']}.items()}
    result=invoke(d)
    assert result['config']==pin(d['configpath']) and len(d['seen'])==1
    assert owners._bootstrap_config_projection(d['config'],d['policy'],d['checked'],
        str(d['policy_path']))==d['projection']
    for role in ('validation_owner','adoption_owner'):
        assert d['config']['repair_worker']['candidate_runtime'][role]==rendered[role]
        matches=[row for row in rendered[role]['source_descriptors']if row['path']==str(d['policy_path'])]
        assert matches==[d['checked']]
    assert json.loads(Path(d['base']['prior_job']['path']).read_bytes())['attempts']==3
    assert protected=={name:pin(path)for name,path in {'config':d['configpath'],
        'policy':d['policy_path'],**d['unitpaths']}.items()}
