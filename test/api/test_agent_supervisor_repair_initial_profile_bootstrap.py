"""Bootstrap projection predicates; readiness is modeled, not a service/native receipt."""
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

def test_bootstrap_breaks_hash_cycle_and_builds_complete_actual_observational_stage(tmp_path,monkeypatch):
    d=bootstrap_scope(tmp_path,monkeypatch);result=invoke(d);stage=d["seen"][0]
    assert d["base"]["config"]["sha256"]!=pin(d["configpath"])["sha256"]
    assert Path(d["base"]["prior_job"]["path"]).parent != d["jobdir"] and "repair-epochs" in d["jobdir"].parts
    assert result["config"]==pin(d["configpath"]) and result["units"]=={k:pin(p) for k,p in d["unitpaths"].items()}
    assert set(stage["initial_bindings"])=={"candidate_policy","candidate_runtime","authorization_epoch","source_snapshot"}
    assert stage["source_successor"] is None and stage["initial_bindings"]["source_snapshot"]==d["snapshot"]
    assert stage["contract"]["config"]==pin(d["configpath"])
    assert stage["files"][-1]["postimage_utf8"].encode()==d["configpath"].read_bytes()
    assert result["expected_loaded_before"]=={"watchdog":{"pid":200,"birth":"200"},"repair":{"pid":201,"birth":"200"}}
    for key in ("inventory","prior_job","job_unit","initial_release","profile","schema"):
        assert result[key]==d["base"][key]

@pytest.mark.parametrize("change",["hold","counterpart","epoch","route","bounds","extra","cwd","pythonpath"])
def test_full_bootstrap_projection_refuses_every_unrelated_config_change(tmp_path,monkeypatch,change):
    d=bootstrap_scope(tmp_path,monkeypatch);c=copy.deepcopy(d["config"])
    if change=="hold":c["boards"][0]["hold_files"]=[]
    elif change=="counterpart":c["boards"][1]["id"]="foreign"
    elif change=="epoch":c["boards"][0]["repair_authorization_epoch"]["max_attempts"]=4
    elif change=="route":c["repair_worker"]["implementation_route"]={"primary_model_id":"foreign"}
    elif change=="bounds":c["repair_worker"]["candidate_runtime"]["resource_bounds"]["tasks_max"]=17
    elif change=="extra":c["unanchored"]=True
    elif change=="cwd":c["repair_worker"]["cwd"]="/old/Y"
    else:c["boards"][0]["probe"]["env"]["PYTHONPATH"]="/old/Y"
    write(d["configpath"],c)
    with pytest.raises(ValueError):invoke(d)
    assert d["seen"]==[]

@pytest.mark.parametrize("slot",["validation_owner.argv","validation_owner.verify_argv","adoption_owner.argv","adoption_owner.verify_argv"])
@pytest.mark.parametrize("failure",["hash","path","duplicate","alias"])
def test_all_four_policy_slots_authenticate_actual_policy_before_normalization(tmp_path,monkeypatch,slot,failure):
    d=bootstrap_scope(tmp_path,monkeypatch);c=copy.deepcopy(d["config"]);role,key=slot.split(".");argv=c["repair_worker"]["candidate_runtime"][role][key]
    if failure=="hash":argv[argv.index("--policy-sha256")+1]="f"*64
    elif failure=="path":argv[argv.index("--policy")+1]=str(tmp_path/"other-policy.json")
    elif failure=="duplicate":argv+=["--policy-sha256",d["checked"]["sha256"]]
    else:argv+=["--policy="+str(d["policy_path"])]
    write(d["configpath"],c)
    with pytest.raises(ValueError):invoke(d)
    assert d["seen"]==[]

@pytest.mark.parametrize("failure",["checked_ref","checked_policy","marker_live","marker_extra","anchor_writable","anchor_extra"])
def test_bootstrap_rejects_unchecked_or_ambiguous_authority_data(tmp_path,monkeypatch,failure):
    d=bootstrap_scope(tmp_path,monkeypatch)
    if failure=="checked_ref":d["checked"]=None
    elif failure=="checked_policy":d["policy"]={**d["policy"],"python":"/foreign/python"}
    elif failure=="marker_live":
        c=copy.deepcopy(d["config"]);c["notes"]=owners.INITIAL_PROFILE_POLICY_HASH;write(d["configpath"],c)
    elif failure=="marker_extra":
        d["anchor"]["expected_config_projection"]["notes"]=owners.INITIAL_PROFILE_POLICY_HASH;d["reseal"]()
    elif failure=="anchor_writable":d["anchorpath"].chmod(0o600)
    else:d["anchor"]["extra"]=True;d["reseal"]()
    with pytest.raises(ValueError):invoke(d)
    assert d["seen"]==[]

@pytest.mark.parametrize("field",["new_commit","new_tree","source_commitment","manifest"])
def test_bootstrap_source_snapshot_is_bound_to_actual_initial_release(tmp_path,monkeypatch,field):
    d=bootstrap_scope(tmp_path,monkeypatch)
    d["anchor"]["source_snapshot"][field]={} if field=="manifest" else "c"*40
    d["reseal"]()
    with pytest.raises(ValueError):invoke(d)
    assert d["seen"]==[]

@pytest.mark.parametrize("failure",["content","mode","target"])
def test_bootstrap_unit_postimages_are_exact_before_observation(tmp_path,monkeypatch,failure):
    d=bootstrap_scope(tmp_path,monkeypatch)
    if failure=="content":d["unitpaths"]["watchdog"].write_text("different")
    elif failure=="mode":d["unitpaths"]["watchdog"].chmod(0o644)
    else:d["anchor"]["expected_unit_postimages"]["watchdog"]["path"]=str(tmp_path/"foreign.service");d["reseal"]()
    with pytest.raises(ValueError):invoke(d)
    assert d["seen"]==[]

def test_changed_Z_source_is_not_accepted_as_the_anchored_initial_snapshot(tmp_path,monkeypatch):
    d=bootstrap_scope(tmp_path,monkeypatch);d["subject"].write_text("changed = True\n")
    with pytest.raises(ValueError):invoke(d)
    assert d["seen"]==[]

def test_successor_chain_keeps_existing_authenticated_chain_path(tmp_path,monkeypatch):
    d=bootstrap_scope(tmp_path,monkeypatch)
    current=copy.deepcopy(d["config"]);current["repair_worker"]["candidate_policy"]["base_commit"]="c"*40;write(d["configpath"],current)
    previous=write(tmp_path/"previous-stage.json",{"source_successor":{"successor_policy":current["repair_worker"]["candidate_policy"]}},True)
    activation=write(tmp_path/"activation.json",{"stage":previous},True);chain=write(tmp_path/"chain.json",{"activation":activation},True);write(d["jobdir"]/"job.json",{"repair_source_successors":[chain]})
    calls=[]
    monkeypatch.setattr(installer,"verify_repair_upgrade",lambda stage,installed=False:calls.append((stage,installed)))
    monkeypatch.setattr(installer,"observe_repair_upgrade",lambda stage,runner:{"loaded_runtime":d["observations"]})
    monkeypatch.setattr(owners,"_bootstrap_current_profile_contract",lambda *args:pytest.fail("bootstrap branch replayed after a source successor"))
    result=invoke(d)
    assert len(calls)==1 and calls[0][1] is True and result["config"]==pin(d["configpath"])
    assert result["prior_job"]==d["base"]["prior_job"] and result["initial_release"]==d["base"]["initial_release"]

def test_legacy_contract_branch_keeps_its_existing_observation_shape(tmp_path,monkeypatch):
    d=bootstrap_scope(tmp_path,monkeypatch);base=copy.deepcopy(d["base"]);base["config"]=pin(d["configpath"]);base["units"]={k:pin(p) for k,p in d["unitpaths"].items()};ref=write(tmp_path/"legacy-contract.json",base,True)
    seen=[]
    monkeypatch.setattr(installer,"observe_repair_upgrade",lambda stage,runner:(seen.append(stage) or {"loaded_runtime":d["observations"]}))
    result=owners.current_profile_contract({"adoption":ref},d["frozen"])
    assert result["config"]==base["config"] and "initial_bindings" not in seen[0]
    assert result["prior_job"]==base["prior_job"]

@pytest.mark.parametrize("when",["descriptor","before_readiness","during_readiness"])
def test_configuration_replacement_never_returns_unprojected_bootstrap_contract(tmp_path,monkeypatch,when):
    d=bootstrap_scope(tmp_path,monkeypatch);changed=[False]
    def replace():
        current=json.loads(d["configpath"].read_bytes());current["unanchored_race"]=True
        write(d["configpath"],current);changed[0]=True
    if when=="descriptor":
        original=owners.descriptor
        def raced(path):
            if Path(path)==d["configpath"] and not changed[0]:replace()
            return original(path)
        monkeypatch.setattr(owners,"descriptor",raced)
    elif when=="before_readiness":
        original=owners.checked_descriptor
        def raced(reference):
            raw=original(reference)
            if reference["path"]==str(d["unitpaths"]["repair"]) and not changed[0]:replace()
            return raw
        monkeypatch.setattr(owners,"checked_descriptor",raced)
    else:
        def raced(stage,runner):
            replace()
            return {"loaded_runtime":d["observations"]}
        monkeypatch.setattr(installer,"observe_repair_upgrade",raced)
    with pytest.raises(ValueError):invoke(d)
    assert changed[0] and d["seen"]==[]


@pytest.mark.parametrize("reverse", [False, True])
def test_bootstrap_reconstructs_exact_typed_science_binding_in_both_board_orders(tmp_path, monkeypatch, reverse):
    d = bootstrap_scope(tmp_path, monkeypatch)
    science = next(b for b in d["projection"]["boards"] if b["id"] == "legal-ir-scientific")
    profile = write(tmp_path / "operator-profile.json", {"source_only_fixture": True}, True)
    git = write(tmp_path / "operator-Git-binding.json", {"schema": installer.SCIENTIFIC_GIT_BINDING_SCHEMA, "profile": profile}, True)
    science["native_executor_binding"] = profile; science["native_executor_git_binding"] = git
    if reverse: d["projection"]["boards"].reverse()
    d["reseal"]()
    # Transport through the actual bootstrap is tested here. Composition cases
    # separately exercise the complete source/Git validator; no health is faked.
    calls = []
    def compose(config, profile_ref, release, git_ref=None):
        assert profile_ref == profile and git_ref == git
        calls.append((copy.deepcopy(profile_ref), copy.deepcopy(git_ref)))
        return config
    monkeypatch.setattr(installer, "_repair_scientific_configuration", compose)
    result = invoke(d); stage = d["seen"][0]
    assert calls and stage["scientific_executor_binding"] == profile and stage["scientific_executor_git_binding"] == git
    assert result["config"] == pin(d["configpath"])
    assert json.loads(stage["files"][-1]["postimage_utf8"]) == d["config"]
    counterpart = next(b for b in d["config"]["boards"] if b["id"] == "legal-ir")
    assert counterpart["hold_files"] == ["HOLD"] and counterpart["repair_authorization_epoch"]["max_attempts"] == 3


def test_bootstrap_rejects_orphan_typed_binding_before_readiness(tmp_path, monkeypatch):
    d = bootstrap_scope(tmp_path, monkeypatch)
    science = next(b for b in d["projection"]["boards"] if b["id"] == "legal-ir-scientific")
    science["native_executor_git_binding"] = write(tmp_path / "orphan.json", {"source_only_fixture": True}, True)
    d["reseal"]()
    with pytest.raises(ValueError, match="requires its exact profile"): invoke(d)
    assert d["seen"] == []



def current_renderer_scope(tmp_path,monkeypatch):
    """Use the current immutable-control renderer's two policy source-DATA slots."""
    d=bootstrap_scope(tmp_path,monkeypatch)
    for role in ('validation_owner','adoption_owner'):
        d['projection']['repair_worker']['candidate_runtime'][role]['source_descriptors'].append({
            'path':str(d['policy_path']),'bytes':owners.INITIAL_PROFILE_POLICY_BYTES,
            'sha256':owners.INITIAL_PROFILE_POLICY_HASH})
    d['reseal']()
    for role in ('validation_owner','adoption_owner'):
        refs=d['config']['repair_worker']['candidate_runtime'][role]['source_descriptors']
        refs[-1]=dict(d['checked'])
    write(d['configpath'],d['config'])
    return d


def test_current_control_renderer_policy_DATA_refs_break_all_self_hash_cycles(tmp_path,monkeypatch):
    d=current_renderer_scope(tmp_path,monkeypatch)
    result=invoke(d)
    assert result['config']==pin(d['configpath']) and d['seen']
    projection=owners._bootstrap_config_projection(d['config'],d['policy'],d['checked'],str(d['policy_path']))
    assert projection==d['projection'] and len(owners._bootstrap_marker_paths(projection))==8
    for role in ('validation_owner','adoption_owner'):
        assert d['config']['repair_worker']['candidate_runtime'][role]['source_descriptors'][-1]==d['checked']


@pytest.mark.parametrize('failure',['sha','bytes','path','duplicate','mixed','unexpected_marker'])
def test_current_control_policy_source_slots_refuse_drift_or_unbound_normalization(tmp_path,monkeypatch,failure):
    d=current_renderer_scope(tmp_path,monkeypatch)
    runtime=d['config']['repair_worker']['candidate_runtime']
    refs=runtime['validation_owner']['source_descriptors']
    if failure=='sha':refs[-1]['sha256']='0'*64
    elif failure=='bytes':refs[-1]['bytes']+=1
    elif failure=='path':refs[-1]['path']=str(tmp_path/'other-policy.json')
    elif failure=='duplicate':refs.append(dict(d['checked']))
    elif failure=='mixed':runtime['adoption_owner']['source_descriptors'].pop()
    else:d['config']['unrelated_marker']=owners.INITIAL_PROFILE_POLICY_BYTES
    write(d['configpath'],d['config'])
    with pytest.raises(ValueError):invoke(d)
    assert d['seen']==[]
