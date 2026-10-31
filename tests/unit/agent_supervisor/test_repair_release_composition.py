"""Pure release/profile joins; fixture parser grants no native or runtime authority."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
from scripts.ops.agent_supervisor import install_fleet_watchdog as installer

def pin(path):
    raw=path.read_bytes()
    return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))
    return pin(path)

def fixture(tmp_path,reverse=False):
    release=tmp_path/"release";source=tmp_path/"source"
    directory=release/"ipfs_accelerate_py/agent_supervisor/rescue"
    directory.mkdir(parents=True);source.mkdir()
    # The fake interface exercises authenticated real-file loading only.
    # The genuine 14-field parser is independently qualified by its owner.
    parser=directory/"native_executor_health.py"
    parser.write_text("import json\nfrom pathlib import Path\ndef _profile(ref):\n    return json.loads(Path(ref['path']).read_bytes())\n")
    roles={"health_module":pin(parser)}
    for key,name in (("main_entrypoint","scientific_executor.py"),("owner_source","original_embedded_observation_owner.py"),("driver_source","native_embedded_observation_driver.py")):
        p=directory/name;p.write_text("fixture_only = True\n");roles[key]=pin(p)
    catalogue={"board_namespace":"science","plan_root_cid":"plan","repository_tree_id":"tree","population_sha256":"a"*64,"database_uuid":"db","store_generation":1}
    profile={"schema":"fixture-interface-only","catalogue":catalogue,**roles}
    reference=write(tmp_path/"profile.json",profile)
    binding={k:v for k,v in catalogue.items() if k not in ("database_uuid","store_generation")}
    binding["max_age_seconds"]=180
    args=["python","-P","-m","ipfs_accelerate_py.agent_supervisor.rescue.native_catalogue_probe","--board","legal-ir-scientific","--namespace","science","--plan-root-cid","plan","--repository-tree-id","tree","--population-sha256","a"*64]
    env={"PYTHONPATH":"/old/source:/old/release","KEEP":"unchanged"}
    boards=[{"id":"legal-ir","probe":{"argv":["probe"],"env":dict(env)},"repair":{"argv":["repair"],"env":dict(env)},"hold_files":["HOLD"],"history":[3]},
            {"id":"legal-ir-scientific","native_catalogue_binding":binding,"probe":{"argv":args,"env":dict(env),"timeout_seconds":30},"repair":{"argv":["repair"],"env":dict(env)},"hold_files":["OPERATOR_STOP"],"cwd":"/retained/cwd","history":[3]}]
    if reverse:boards.reverse()
    return {"boards":boards,"repair_worker":{"untouched":True}},reference,profile,release,source

@pytest.mark.parametrize("reverse",[False,True])
def test_declared_board_commands_use_release_first_and_preserve_every_other_field(tmp_path,reverse):
    config,reference,profile,release,source=fixture(tmp_path,reverse)
    before=copy.deepcopy(config)
    actual=installer._repair_release_command_environments(config,release,source)
    expected=copy.deepcopy(before)
    expected["repair_worker"]["cwd"]=str(release)
    for board in expected["boards"]:
        for name in ("probe","repair"):board[name]["env"]["PYTHONPATH"]=str(release)+":"+str(source)
    assert actual==expected and config==before
    actual["boards"][0]["history"].append(4)
    assert config==before

@pytest.mark.parametrize("reverse",[False,True])
def test_science_profile_joins_exact_release_roles_and_preserves_engineering_board(tmp_path,reverse):
    config,reference,profile,release,source=fixture(tmp_path,reverse)
    before=copy.deepcopy(config)
    actual=installer._repair_scientific_configuration(config,reference,release)
    science=next(b for b in actual["boards"] if b["id"]=="legal-ir-scientific")
    assert science["native_executor_binding"]==reference
    assert science["probe"]["argv"][-6:]==["--executor-profile",reference["path"],"--executor-profile-bytes",str(reference["bytes"]),"--executor-profile-sha256",reference["sha256"]]
    for old,new in zip(before["boards"],actual["boards"]):
        if old["id"]=="legal-ir":assert new==old
        else:
            restored=copy.deepcopy(new);restored.pop("native_executor_binding");restored["probe"]["argv"]=restored["probe"]["argv"][:-6]
            assert restored==old
    assert config==before
    assert installer._repair_scientific_configuration(actual,reference,release)==actual
    assert not any(name.startswith("_ipfs_repair_release_health_") for name in sys.modules)

@pytest.mark.parametrize("field",["board_namespace","plan_root_cid","repository_tree_id","population_sha256"])
def test_science_profile_refuses_a_foreign_catalogue_without_mutation(tmp_path,field):
    config,reference,profile,release,source=fixture(tmp_path);before=copy.deepcopy(config)
    profile["catalogue"][field]="foreign";reference=write(Path(reference["path"]),profile)
    with pytest.raises(ValueError):installer._repair_scientific_configuration(config,reference,release)
    assert config==before

@pytest.mark.parametrize("role",["main_entrypoint","health_module","owner_source","driver_source"])
def test_science_profile_refuses_mutable_checkout_role_paths(tmp_path,role):
    config,reference,profile,release,source=fixture(tmp_path)
    foreign=source/Path(profile[role]["path"]).name;foreign.write_bytes(Path(profile[role]["path"]).read_bytes())
    profile[role]=pin(foreign);reference=write(Path(reference["path"]),profile)
    with pytest.raises(ValueError):installer._repair_scientific_configuration(config,reference,release)

@pytest.mark.parametrize("flag",["--executor-profile","--executor-profile-bytes","--executor-profile-sha256"])
def test_unbound_science_probe_flags_are_refused(tmp_path,flag):
    config,reference,profile,release,source=fixture(tmp_path)
    config["boards"][1]["probe"]["argv"] += [flag,"foreign"]
    with pytest.raises(ValueError):installer._repair_scientific_configuration(config,reference,release)

@pytest.mark.parametrize("failure",["missing","duplicate","foreign"])
def test_science_binding_requires_both_existing_boards(tmp_path,failure):
    config,reference,profile,release,source=fixture(tmp_path)
    if failure=="missing":config["boards"].pop(0)
    elif failure=="duplicate":config["boards"][0]["id"]="legal-ir-scientific"
    else:config["boards"][0]["id"]="other"
    with pytest.raises(ValueError):installer._repair_scientific_configuration(config,reference,release)

def test_scientific_descriptor_is_reauthenticated_after_profile_drift(tmp_path):
    config,reference,profile,release,source=fixture(tmp_path)
    assert installer._repair_scientific_configuration(config,reference,release)
    Path(reference["path"]).write_text("{}")
    with pytest.raises(ValueError):installer._repair_scientific_configuration(config,reference,release)

def test_existing_science_binding_must_match_its_probe_before_replacement(tmp_path):
    config,reference,profile,release,source=fixture(tmp_path)
    bound=installer._repair_scientific_configuration(config,reference,release)
    bound["boards"][1]["probe"]["argv"][-1]="0"*64
    with pytest.raises(ValueError):installer._repair_scientific_configuration(bound,reference,release)

@pytest.mark.parametrize("environment",[None,[],{"KEEP":False}])
def test_release_rebind_refuses_nonordinary_command_environments(tmp_path,environment):
    config,reference,profile,release,source=fixture(tmp_path);config["boards"][0]["probe"]["env"]=environment
    with pytest.raises(ValueError):installer._repair_release_command_environments(config,release,source)

def test_stage_verifier_and_apply_refuse_profile_drift_before_any_external_action(tmp_path,monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    config,reference,profile,release,source=fixture(tmp_path)
    before_config=write(tmp_path/"config.json",config)
    post=installer._repair_scientific_configuration(installer._repair_release_command_environments(config,release,source),reference,release)
    files=[]
    for name in ("watchdog","repair"):
        p=tmp_path/(name+".service");p.write_text("old")
        files.append({"before":pin(p),"postimage_utf8":"new","mode":p.stat().st_mode & 0o777})
    files.append({"before":before_config,"postimage_utf8":json.dumps(post),"mode":Path(before_config["path"]).stat().st_mode & 0o777})
    fixed=write(tmp_path/"fixed.json",{"fixture_only":True})
    plan={"schema":"agent-supervisor/repair-profile-upgrade-stage@1","files":files,"contract":{"config":before_config,"prior_job":fixed,"inventory":fixed,"initial_release":fixed,"expected_loaded_before":{},"job_unit":"fixture-only.service"},
          "release_commitment":fixed,"release_sources":{},"selected_release_sources":{},"release":str(release),"source_successor":None,"initial_bindings":{"candidate_policy":{"repository_root":str(source)}},"scientific_executor_binding":reference}
    installer.verify_repair_upgrade(plan)
    Path(reference["path"]).write_text("{}")
    calls=[]
    monkeypatch.setattr(owners,"verify_original_job_closed",lambda *args,**kwargs:calls.append("external"))
    with pytest.raises(ValueError):installer.verify_repair_upgrade(plan)
    with pytest.raises(ValueError):installer.apply_repair_upgrade(plan,lambda *args,**kwargs:calls.append("runner"))
    assert calls==[]

def test_release_liveness_identity_uses_actual_watchdog_and_rejects_stale_stage(tmp_path):
    config,reference,profile,release,source=fixture(tmp_path)
    config["liveness_identity"]={"schema":"agent-supervisor/watchdog-liveness-profile@1","deployment_id":"old-release","source_sha256":"0"*64}
    before=copy.deepcopy(config)
    watchdog=release/"ipfs_accelerate_py/agent_supervisor/rescue/fleet_watchdog.py";watchdog.write_text("fixture_only = True\n")
    watchdog_pin=pin(watchdog)
    actual=installer._repair_release_liveness_identity(config,release,watchdog_pin)
    assert actual["liveness_identity"]=={"schema":"agent-supervisor/watchdog-liveness-profile@1","deployment_id":release.name,"source_sha256":watchdog_pin["sha256"]}
    restored=copy.deepcopy(actual);restored["liveness_identity"]=before["liveness_identity"]
    assert restored==before and config==before
    bad={**watchdog_pin,"sha256":"f"*64}
    with pytest.raises(ValueError):installer._repair_release_liveness_identity(config,release,bad)
    before_config=write(tmp_path/"config.json",config)
    post=installer._repair_scientific_configuration(installer._repair_release_command_environments(actual,release,source),reference,release)
    files=[]
    for name in ("watchdog","repair"):
        p=tmp_path/(name+".service");p.write_text("old")
        files.append({"before":pin(p),"postimage_utf8":"new","mode":p.stat().st_mode & 0o777})
    files.append({"before":before_config,"postimage_utf8":json.dumps(post),"mode":Path(before_config["path"]).stat().st_mode & 0o777})
    fixed=write(tmp_path/"fixed.json",{"fixture_only":True})
    plan={"schema":"agent-supervisor/repair-profile-upgrade-stage@1","files":files,"contract":{"config":before_config,"prior_job":fixed,"inventory":fixed,"initial_release":fixed,"expected_loaded_before":{},"job_unit":"fixture-only.service"},
          "release_commitment":fixed,"release_sources":{"fleet_watchdog":watchdog_pin},"selected_release_sources":{},"release":str(release),"source_successor":None,"initial_bindings":{"candidate_policy":{"repository_root":str(source)}},"scientific_executor_binding":reference}
    installer.verify_repair_upgrade(plan)
    post["liveness_identity"]=before["liveness_identity"];files[-1]["postimage_utf8"]=json.dumps(post)
    with pytest.raises(ValueError):installer.verify_repair_upgrade(plan)

def test_worker_engine_uses_release_while_candidate_repository_and_workspace_stay_separate(tmp_path):
    config,reference,profile,release,source=fixture(tmp_path)
    config["repair_worker"]={"cwd":"/old/Y","argv":["original-worker"],"implementation_route":{"primary_provider_id":"grok_cli","primary_model_id":"grok-4.7"},"candidate_policy":{"repository_root":str(source)}}
    before=copy.deepcopy(config)
    actual=installer._repair_release_command_environments(config,release,source)
    assert actual["repair_worker"]["cwd"]==str(release)
    restored=copy.deepcopy(actual["repair_worker"]);restored["cwd"]=before["repair_worker"]["cwd"]
    assert restored==before["repair_worker"]
    assert actual["repair_worker"]["candidate_policy"]["repository_root"]==str(source)
    assert installer._repair_release_command_environments(actual,release,source)==actual and config==before
    bad=copy.deepcopy(config);bad["repair_worker"]=[]
    with pytest.raises(ValueError):installer._repair_release_command_environments(bad,release,source)
    before_config=write(tmp_path/"config.json",config);files=[]
    for name in ("watchdog","repair"):
        p=tmp_path/(name+".service");p.write_text("old")
        files.append({"before":pin(p),"postimage_utf8":"new","mode":p.stat().st_mode & 0o777})
    files.append({"before":before_config,"postimage_utf8":json.dumps(actual),"mode":Path(before_config["path"]).stat().st_mode & 0o777})
    fixed=write(tmp_path/"fixed.json",{"fixture_only":True})
    plan={"schema":"agent-supervisor/repair-profile-upgrade-stage@1","files":files,"contract":{"config":before_config,"prior_job":fixed,"inventory":fixed,"initial_release":fixed,"expected_loaded_before":{},"job_unit":"fixture-only.service"},"release_commitment":fixed,"release_sources":{},"selected_release_sources":{},"release":str(release),"source_successor":None,"initial_bindings":{"candidate_policy":{"repository_root":str(source)}}}
    installer.verify_repair_upgrade(plan)
    actual["repair_worker"]["cwd"]="/old/Y";files[-1]["postimage_utf8"]=json.dumps(actual)
    with pytest.raises(ValueError):installer.verify_repair_upgrade(plan)
