"""Typed scientific Git/profile joins; no service, DB, provider or native authority.

The H174 owning tests are reused with their explicit source-only fixture parsers
and modeled Git observations. Actual SDK/CLI3 compatibility is a separate
qualification test with authenticated real source inputs, not a fake witness.
"""
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


def git_fixture(tmp_path, monkeypatch, reverse=False):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    config, _, oldprofile, release, source = fixture(tmp_path, reverse)
    root = tmp_path / "F"; datasets = tmp_path / "R"; root.mkdir(); datasets.mkdir()
    files = {}
    names = {"main_entrypoint": "agent_supervisor/rescue/scientific_executor.py",
             "health_module": "agent_supervisor/rescue/native_executor_health.py",
             "owner_source": "agent_supervisor/rescue/original_embedded_observation_owner.py",
             "driver_source": "agent_supervisor/rescue/native_embedded_observation_driver.py",
             "pre_body_guard": "agent_supervisor/rescue/project_import_source_guard.py",
             "observer_source": "agent_supervisor/todo_daemon/native_owner_bootstrap.py",
             "route_source": "agent_implementation_route.py"}
    roles = {}
    for key, name in names.items():
        p = root / "ipfs_accelerate_py" / name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(Path(oldprofile["health_module"]["path"]).read_bytes() if key == "health_module" else (
            b"from pathlib import Path\nRUNTIME = Path(__file__).parents[3]\ndef configuration(plan):\n    c=plan['native_metadata_cli']; g=c['import_head_capsule_Git']\n    if g['scope'] != 'immutable HEAD bytes archived/hashed by original import capsule; no import or execution of omitted live source bodies': raise RuntimeError('scope')\n    if g['ls_tree_tail'] != ['ls-tree','-r','-z',g['source_head'],'--','ipfs_accelerate_py','scripts/ops/agent_supervisor/configured_board_scheduler.py','scripts/ops/agent_supervisor/implementation_supervisor_entry.py']: raise RuntimeError('tree scope')\n    return c\n" if key == "driver_source" else ("fixture_" + key + " = True\n").encode()))
        if key == "driver_source":
            (release / "ipfs_accelerate_py/agent_supervisor/rescue/native_embedded_observation_driver.py").write_bytes(p.read_bytes())
        roles[key] = pin(p)
        rel = p.relative_to(root).as_posix()
        files[rel] = {**pin(p), "relative_path": rel, "mode": 0o644,
                      "git_blob": hashlib.sha1(b"blob " + str(p.stat().st_size).encode() + b"\0" + p.read_bytes()).hexdigest()}
    head, tree = "a" * 40, "b" * 40; scope = ["retained-task"]
    catalogue = copy.deepcopy(oldprofile["catalogue"])
    original = {"source_root": str(root), "source_head": head, "blobs": [
        {"relative_path": rel, "object_id": row["git_blob"], "bytes": row["bytes"]} for rel, row in files.items()],
        "ls_tree_tail": ["ls-tree", "-r", "-z", head, "--", "ipfs_accelerate_py", "scripts/ops/agent_supervisor/configured_board_scheduler.py", "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"], "required_relative_files": list(files), "route_source": roles["route_source"],
        "scope": "immutable HEAD bytes archived/hashed by original import capsule; no import or execution of omitted live source bodies", "selected_file_count": len(files),
        "payload_bytes": sum(r["bytes"] for r in files.values()), "unique_blob_count": len(files)}
    inventory = {"schema": "legal-ir-scientific-source-inventory/v1", "source_root": str(root), "source_commit": head,
                 "source_tree": tree, "file_count": len(files), "source_bytes": original["payload_bytes"], "files": list(files.values())}
    capsule = {"schema": "legal-ir-scientific-capsule-DATA-scope/v1", "source_root": str(root), "source_commit": head,
               "source_tree": tree, "source_count": len(files), "source_bytes": original["payload_bytes"],
               "DATA_not_import_or_execution_credit": True, "sources": copy.deepcopy(list(files.values()))}
    observation = {"schema": "fixture-observation", "source_schema": "fixture-source", "database_preimage": {"fixture": True},
                   "observer_source": roles["observer_source"], "task_scope": scope,
                   **{k: catalogue[k] for k in ("plan_root_cid", "repository_tree_id")}}
    cli = {"schema": "native-embedded-scoped-observation-cli/v2", "runtime_root": str(root), "cwd": str(datasets),
           "attempt_directory": str(tmp_path / "not-executed"), "temporary_directory": str(tmp_path / "not-executed/tmp"),
           "result_path": str(tmp_path / "not-executed/result.json"), "pre_body_guard": roles["pre_body_guard"],
           "project_import_plan": {"accelerate_root": str(root), "datasets_root": str(datasets), "publication_root": str(datasets), "source_pins": list(roles.values())},
           "source_sha256": {r["path"]: r["sha256"] for r in roles.values()}, "observation_binding": observation,
           "import_head_capsule_Git": original}
    plan = {"schema": "native-embedded-scoped-observation-plan/v1", "operation": "native_embedded_scoped_observation_v1",
            "accelerate_root": str(root), "task_scope": scope, "native_metadata_cli": cli}
    request = {"schema": "legal-ir-original-native-control-request/v1", "cwd": str(datasets), "source_roots": [str(datasets), "/original/A"],
               "source_sha256": copy.deepcopy(cli["source_sha256"]), "cpu_slots": 1, "memory_mb": 1536, "child_process_slots": 8}
    d = {"config": config, "release": release, "source": source, "root": root, "datasets": datasets, "head": head, "tree": tree,
         "roles": roles, "files": files, "inventory": inventory, "capsule": capsule, "original": original, "plan": plan, "request": request,
         "git_fault": None, "git_calls": [], "service": None, "typed": None}
    profilepath = tmp_path / "Git-profile.json"
    def seal():
        inventory_ref = write(tmp_path / "inventory.json", inventory)
        capsule_ref = write(tmp_path / "capsule.json", capsule)
        original_ref = write(tmp_path / "original-Git.json", original)
        generation = write(tmp_path / "generation.json", {"source_generation_only": True})
        commitment = {"schema": "legal-ir-scientific-source-commitment/v1", "source_root": str(root), "source_commit": d["head"], "source_tree": d["tree"],
                      "source_inventory": inventory_ref, "capsule_DATA": capsule_ref, "original_import_Git_binding": original_ref,
                      "source_operation_plan": generation, "source_qualification": {}, "source_qualification_successors": [], "runtime_activation_or_task_authority": False}
        d["commitment_ref"] = write(tmp_path / "commitment.json", commitment)
        plan_ref = write(tmp_path / "plan.json", plan); request_ref = write(tmp_path / "request.json", request)
        template = {"schema": "agent-supervisor/scientific-executor-cycle-template@1", "kind": "genuine_scoped_idle", "plan_ref": plan_ref, "request_ref": request_ref}
        template_ref = write(tmp_path / "template.json", template)
        service = d["service"] or {"schema": "agent-supervisor/scientific-executor-service@1", "profile_path": str(profilepath), "catalogue": {**catalogue, "board_id": "legal-ir-scientific", "database_path": "/retained/catalogue", "timeout_seconds": 30}}
        service = {**service, "idle_template_ref": template_ref}
        service_ref = write(tmp_path / "service.json", service)
        profile = {"schema": "fixture-interface-only", "catalogue": catalogue, "task_scope": scope,
                   **{k: roles[k] for k in ("main_entrypoint", "health_module", "owner_source", "driver_source")},
                   "unit": {"working_directory": str(datasets), "argv": ["python", roles["main_entrypoint"]["path"], "--config", service_ref["path"], "--config-sha256", service_ref["sha256"]]},
                   "source_descriptors": [*roles.values(), service_ref]}
        d["profile"] = profile; d["profile_ref"] = write(profilepath, profile)
        typed = d["typed"] or {"schema": installer.SCIENTIFIC_GIT_BINDING_SCHEMA, "profile": d["profile_ref"], "source_commitment": d["commitment_ref"], "observer_plan": plan_ref}
        d["binding_ref"] = write(tmp_path / "typed-binding.json", typed)
    def git(root_arg, argv, **kwargs):
        assert root_arg == root and kwargs["timeout"] == 3
        assert "GIT_CONFIG_GLOBAL" in kwargs["environment"]
        d["git_calls"].append(argv)
        args = argv[4:]
        if args == ["rev-parse", "--show-toplevel"]: return str(root) + "\n"
        if args == ["rev-parse", "--verify", "HEAD"]: return ("c" * 40 if d["git_fault"] == "head" else head) + "\n"
        if args == ["rev-parse", head + "^{tree}"]: return ("c" * 40 if d["git_fault"] == "tree" else tree) + "\n"
        if args == ["status", "--porcelain=v1", "-z", "--untracked-files=all"]: return "?? untracked\0" if d["git_fault"] == "dirty" else ""
        if args == ["ls-tree", "-r", "-z", "--full-tree", head]:
            return "".join("100644 blob " + row["git_blob"] + "\t" + rel + "\0" for rel, row in files.items() if d["git_fault"] != "inventory" or rel != next(iter(files))) + "100644 blob " + "f" * 40 + "\tREADME.md\0" + "160000 commit " + "e" * 40 + "\texternal/retained-submodule\0"
        raise AssertionError(args)
    d["seal"] = seal; seal(); monkeypatch.setattr(owners, "git_command", git)
    return d


@pytest.mark.parametrize("reverse", [False, True])
def test_typed_Git_profile_preserves_both_namespaces_and_cwd(tmp_path, monkeypatch, reverse):
    d = git_fixture(tmp_path, monkeypatch, reverse); before = copy.deepcopy(d["config"])
    actual = installer._repair_scientific_configuration(d["config"], d["profile_ref"], d["release"], d["binding_ref"])
    science = next(b for b in actual["boards"] if b["id"] == "legal-ir-scientific")
    assert science["native_executor_git_binding"] == d["binding_ref"] and science["native_executor_binding"] == d["profile_ref"]
    assert d["profile"]["unit"]["working_directory"] == str(d["datasets"])
    assert next(b for b in actual["boards"] if b["id"] == "legal-ir") == next(b for b in before["boards"] if b["id"] == "legal-ir")
    assert installer._repair_scientific_configuration(actual, d["profile_ref"], d["release"], d["binding_ref"]) == actual
    assert d["config"] == before and d["git_calls"]


@pytest.mark.parametrize("failure", ["extra", "profile", "head", "tree", "dirty", "inventory", "role", "scope", "observer", "capsule", "catalogue", "cwd", "request", "service", "verifier", "masked_body", "read_scope", "tree_scope", "untrusted_parser"])
def test_typed_Git_profile_rejects_unbound_generation_and_source_drift(tmp_path, monkeypatch, failure):
    d = git_fixture(tmp_path, monkeypatch); before = copy.deepcopy(d["config"])
    if failure == "extra": d["typed"] = {"schema": installer.SCIENTIFIC_GIT_BINDING_SCHEMA, "profile": d["profile_ref"], "source_commitment": d["commitment_ref"], "observer_plan": pin(tmp_path / "plan.json"), "allow_external_root": True}
    elif failure == "profile": d["typed"] = {"schema": installer.SCIENTIFIC_GIT_BINDING_SCHEMA, "profile": {**d["profile_ref"], "sha256": "0" * 64}, "source_commitment": d["commitment_ref"], "observer_plan": pin(tmp_path / "plan.json")}
    elif failure in ("head", "tree", "dirty", "inventory"): d["git_fault"] = failure
    elif failure == "role": d["roles"]["owner_source"] = d["roles"]["driver_source"]
    elif failure == "scope": d["plan"]["task_scope"] = ["foreign-task"]
    elif failure == "observer": d["plan"]["native_metadata_cli"]["observation_binding"]["observer_source"] = d["roles"]["driver_source"]
    elif failure == "capsule": d["capsule"]["sources"][0]["git_blob"] = "0" * 40
    elif failure == "catalogue": d["plan"]["native_metadata_cli"]["observation_binding"]["repository_tree_id"] = "foreign"
    elif failure == "cwd": d["plan"]["native_metadata_cli"]["cwd"] = str(d["root"])
    elif failure == "request": d["request"]["source_sha256"][d["roles"]["driver_source"]["path"]] = "0" * 64
    elif failure == "service": d["service"] = {"schema": "agent-supervisor/scientific-executor-service@1", "profile_path": "/foreign/profile", "catalogue": {**d["profile"]["catalogue"], "board_id": "legal-ir-scientific"}}
    elif failure == "masked_body":
        p = Path(d["roles"]["owner_source"]["path"]); p.write_text("masked_worktree_change = True\n")
        # Model assume-unchanged/skip-worktree: metadata SHA follows live body,
        # while immutable HEAD OID and the status observation stay unchanged.
        newpin = pin(p); d["roles"]["owner_source"] = newpin
        row = d["files"][p.relative_to(d["root"]).as_posix()]; row.update(newpin)
        archived = next(r for r in d["capsule"]["sources"] if r["path"] == str(p)); archived.update(newpin)
        d["inventory"]["source_bytes"] = sum(r["bytes"] for r in d["files"].values())
        d["capsule"]["source_bytes"] = d["inventory"]["source_bytes"]
        d["original"]["payload_bytes"] = d["inventory"]["source_bytes"]
        next(r for r in d["original"]["blobs"] if r["relative_path"] == row["relative_path"])["bytes"] = newpin["bytes"]
        cli = d["plan"]["native_metadata_cli"]; cli["source_sha256"][str(p)] = newpin["sha256"]
        cli["project_import_plan"]["source_pins"] = list(d["roles"].values())
        d["request"]["source_sha256"][str(p)] = newpin["sha256"]
    elif failure == "read_scope": d["original"]["scope"] = "unbounded import authority"
    elif failure == "tree_scope": d["original"]["ls_tree_tail"] = ["ls-tree", "-r", "-z", d["head"]]
    elif failure == "untrusted_parser": (d["release"] / "ipfs_accelerate_py/agent_supervisor/rescue/native_embedded_observation_driver.py").write_text("raise AssertionError('must not execute')\n")
    elif failure == "verifier": (d["release"] / "ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py").write_text("raise AssertionError('must not execute')\n")
    d["seal"]()
    with pytest.raises(ValueError): installer._repair_scientific_configuration(d["config"], d["profile_ref"], d["release"], d["binding_ref"])
    assert d["config"] == before


def test_typed_Git_binding_cannot_be_silently_removed_or_retargeted(tmp_path, monkeypatch):
    d = git_fixture(tmp_path, monkeypatch)
    actual = installer._repair_scientific_configuration(d["config"], d["profile_ref"], d["release"], d["binding_ref"])
    with pytest.raises(ValueError): installer._repair_scientific_configuration(actual, d["profile_ref"], d["release"])
    other = copy.deepcopy(actual); other["boards"][1]["native_executor_git_binding"]["sha256"] = "0" * 64
    with pytest.raises(ValueError): installer._repair_scientific_configuration(other, d["profile_ref"], d["release"], d["binding_ref"])


def test_typed_Git_stage_drift_refuses_before_CAS_or_external_actions(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = git_fixture(tmp_path, monkeypatch)
    config_ref = write(tmp_path / "fleet.json", d["config"])
    post = installer._repair_scientific_configuration(d["config"], d["profile_ref"], d["release"], d["binding_ref"])
    rows = []
    for suffix in ("watchdog", "repair"):
        p = tmp_path / (suffix + ".service"); p.write_text("old")
        rows.append({"before": pin(p), "postimage_utf8": "new", "mode": p.stat().st_mode & 0o777})
    rows.append({"before": config_ref, "postimage_utf8": json.dumps(post), "mode": Path(config_ref["path"]).stat().st_mode & 0o777})
    fixed = write(tmp_path / "fixed.json", {"fixture_only": True})
    plan = {"schema": "agent-supervisor/repair-profile-upgrade-stage@1", "files": rows,
        "contract": {"config": config_ref, "prior_job": fixed, "inventory": fixed, "initial_release": fixed, "expected_loaded_before": {}, "job_unit": "fixture-only.service"},
        "release_commitment": fixed, "release_sources": {}, "selected_release_sources": {}, "release": str(d["release"]), "source_successor": None,
        "initial_bindings": {"candidate_policy": {"repository_root": str(d["source"])}}, "scientific_executor_binding": d["profile_ref"], "scientific_executor_git_binding": d["binding_ref"]}
    installer.verify_repair_upgrade(plan)
    calls = []; monkeypatch.setattr(owners, "verify_original_job_closed", lambda *a, **kw: calls.append("external"))
    d["git_fault"] = "head"
    with pytest.raises(ValueError): installer.verify_repair_upgrade(plan)
    with pytest.raises(ValueError): installer.apply_repair_upgrade(plan, lambda *a, **kw: calls.append("runner"))
    assert calls == [] and Path(config_ref["path"]).read_bytes() == json.dumps(d["config"], sort_keys=True).encode()
    d["git_fault"] = None; plan.pop("scientific_executor_git_binding")
    with pytest.raises(ValueError): installer.verify_repair_upgrade(plan)


@pytest.mark.parametrize("reverse", [False, True])
def test_successor_renderer_preserves_explicit_Git_binding_without_replacing_science(tmp_path, monkeypatch, reverse):
    d = git_fixture(tmp_path, monkeypatch, reverse)
    home = tmp_path / "operator-home"; state = home / ".local/state/profile"
    path = home / ".config/ipfs-taskboard-watchdog/legal-ir/fleet.json"
    config = installer._repair_scientific_configuration(d["config"], d["profile_ref"], d["release"], d["binding_ref"])
    cp = {"repository_root": str(d["source"]), "source_paths": []}; successor_cp = {**cp, "fixture_successor": True}
    config.update(state_dir=str(state), runtime_release="/old/release")
    config["repair_worker"].update(job_unit="fixture-job.service", candidate_policy=cp, candidate_runtime={"source_descriptors": []})
    before_science = copy.deepcopy(next(b for b in config["boards"] if b["id"] == "legal-ir-scientific"))
    config_ref = write(path, config)
    units = {}
    for suffix in ("watchdog", "repair"):
        p = home / ".config/systemd/user" / ("ipfs-taskboard-legal-ir-" + suffix + ".service")
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text("old"); units[suffix] = pin(p)
    fixed = write(tmp_path / "source-only-fixed.json", {"fixture_only": True})
    prior = write(state / "repairs/legal-ir/job.json", {"attempts": 3})
    contract = {"schema": "agent-supervisor/repair-profile-upgrade-contract@1", "profile": "legal-ir", "config": config_ref,
        "inventory": fixed, "units": units, "prior_job": prior, "expected_loaded_before": {}, "job_unit": "fixture-job.service",
        "initial_release": fixed, "source_successor": {"source_commitment": "source-only", "prior_policy": cp,
            "successor_policy": successor_cp, "successor_source_descriptors": [], "new_commit": "c" * 40, "new_tree": "d" * 40}}
    # Original materialization is separately qualified; this fixture keeps all
    # writes in its owned temporary population and tests the actual renderer.
    watchdog_source = Path(installer.__file__).parents[3] / "ipfs_accelerate_py/agent_supervisor/rescue/fleet_watchdog.py"
    watchdog_raw = watchdog_source.read_bytes()
    watchdog_target = d["release"] / "ipfs_accelerate_py/agent_supervisor/rescue/fleet_watchdog.py"
    watchdog_target.write_bytes(watchdog_raw)
    monkeypatch.setattr(installer, "_finite_repair_release",
        lambda *a: (d["release"], {"fleet_watchdog": watchdog_raw}, {}))
    stage = installer.prepare_repair_upgrade(d["source"], contract, "source-only")
    assert stage["scientific_executor_binding"] == d["profile_ref"] and stage["scientific_executor_git_binding"] == d["binding_ref"]
    post = json.loads(stage["files"][-1]["postimage_utf8"])
    science = next(b for b in post["boards"] if b["id"] == "legal-ir-scientific")
    restored = copy.deepcopy(science); restored["probe"]["env"] = before_science["probe"]["env"]; restored["repair"]["env"] = before_science["repair"]["env"]
    assert restored == before_science and post["repair_worker"]["candidate_policy"] == successor_cp
    assert Path(prior["path"]).read_bytes() == json.dumps({"attempts": 3}, sort_keys=True).encode()



def test_typed_Git_binding_refuses_equal_payload_profile_at_another_path(tmp_path, monkeypatch):
    d = git_fixture(tmp_path, monkeypatch)
    alias = tmp_path / 'equal-payload-profile.json'
    alias.write_bytes(Path(d['profile_ref']['path']).read_bytes())
    alias_ref = pin(alias)
    assert alias_ref['sha256'] == d['profile_ref']['sha256'] and alias_ref['path'] != d['profile_ref']['path']
    with pytest.raises(ValueError, match='closed explicit scientific Git/profile binding required'):
        installer._repair_scientific_profile(alias_ref, d['release'], d['binding_ref'])


def test_typed_Git_binding_does_not_replace_another_observer_profile(tmp_path, monkeypatch):
    d = git_fixture(tmp_path, monkeypatch); before = copy.deepcopy(d['config'])
    science = next(b for b in d['config']['boards'] if b['id'] == 'legal-ir-scientific')
    science['native_executor_binding'] = write(tmp_path/'other-profile.json', {'retained': True})
    expected = copy.deepcopy(d['config'])
    with pytest.raises(ValueError, match='cannot replace another producer profile'):
        installer._repair_scientific_configuration(d['config'], d['profile_ref'], d['release'], d['binding_ref'])
    assert d['config'] == expected and d['config'] != before



def test_explicit_different_Git_descriptor_is_not_silently_replaced(tmp_path, monkeypatch):
    d=git_fixture(tmp_path,monkeypatch)
    config=installer._repair_scientific_configuration(d['config'],d['profile_ref'],d['release'],d['binding_ref'])
    home=tmp_path/'operator-home';state=home/'.local/state/profile'
    path=home/'.config/ipfs-taskboard-watchdog/legal-ir/fleet.json'
    cp={'repository_root':str(d['source']),'source_paths':[]};config.update(state_dir=str(state),runtime_release='/old/release')
    config['repair_worker'].update(job_unit='fixture-job.service',candidate_policy=cp,candidate_runtime={'source_descriptors':[]})
    config_ref=write(path,config);units={}
    for suffix in('watchdog','repair'):
        q=home/'.config/systemd/user'/('ipfs-taskboard-legal-ir-'+suffix+'.service');q.parent.mkdir(parents=True,exist_ok=True);q.write_text('old');units[suffix]=pin(q)
    fixed=write(tmp_path/'fixed.json',{'fixture_only':True});prior=write(state/'repairs/legal-ir/job.json',{'attempts':3})
    contract={'schema':'agent-supervisor/repair-profile-upgrade-contract@1','profile':'legal-ir','config':config_ref,
        'inventory':fixed,'units':units,'prior_job':prior,'expected_loaded_before':{},'job_unit':'fixture-job.service',
        'initial_release':fixed,'source_successor':{'source_commitment':'source-only','prior_policy':cp,
            'successor_policy':{**cp,'fixture_successor':True},'successor_source_descriptors':[],'new_commit':'c'*40,'new_tree':'d'*40}}
    wrong=write(tmp_path/'different-Git.json',{'not_the_existing_binding':True});calls=[]
    monkeypatch.setattr(installer,'_finite_repair_release',lambda *a:calls.append('materialized'))
    with pytest.raises(ValueError,match='explicit scientific Git binding differs'):
        installer.prepare_repair_upgrade(d['source'],contract,'source-only',scientific_executor_git_binding=wrong)
    assert calls==[] and Path(config_ref['path']).read_bytes()==json.dumps(config,sort_keys=True).encode()


def separate_producer_fixture(tmp_path, monkeypatch, mode=0o600):
    """Modeled two-repository metadata; no producer body is imported/executed."""
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = git_fixture(tmp_path, monkeypatch)
    native_git = owners.git_command
    root = tmp_path / "PW"; root.mkdir()
    relatives = ["ipfs_accelerate_py/__init__.py", "ipfs_accelerate_py/agent_supervisor/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py"]
    rows = {}
    for rel in relatives:
        path = root / rel; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Producer source fixture: metadata only.\n")
        path.chmod(mode)
        body = path.read_bytes()
        rows[rel] = {**pin(path), "relative_path": rel, "mode": mode,
            "git_blob": hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()}
    d.update(producer_root=root, producer_rows=rows, producer_head="c"*40, producer_tree="d"*40,
             producer_fault=None, producer_calls=[], producer_changes=None)
    old_seal = d["seal"]
    def seal():
        old_seal()
        inventory = {"schema":"legal-ir-scientific-source-inventory/v1", "source_root":str(root),
            "source_commit":d["producer_head"], "source_tree":d["producer_tree"],
            "file_count":len(rows), "source_bytes":sum(row["bytes"] for row in rows.values()), "files":list(rows.values())}
        if d["producer_changes"] is not None:
            d["producer_changes"](inventory)
        d["producer_inventory"] = inventory
        inventory_ref = write(tmp_path/"producer-inventory.json", inventory)
        generation_ref = write(tmp_path/"producer-generation.json", {"source_only":True})
        qualification_ref = write(tmp_path/"producer-source-qualification.json", {"modeled_source_fixture_only":True})
        commitment = {"schema":"agent-supervisor/scientific-producer-source-commitment@1",
            "source_root":str(root), "source_commit":d["producer_head"], "source_tree":d["producer_tree"],
            "source_inventory":inventory_ref, "source_operation_plan":generation_ref,
            "source_qualification":qualification_ref, "runtime_activation_or_task_authority":False}
        d["producer_commitment"] = commitment
        producer_ref = write(tmp_path/"producer-commitment.json", commitment)
        profile = copy.deepcopy(d["profile"])
        native_main = profile["main_entrypoint"]["path"]
        main = pin(root/relatives[-1])
        profile["main_entrypoint"] = main
        profile["unit"]["argv"] = [main["path"] if value==native_main else value for value in profile["unit"]["argv"]]
        profile["unit"]["environment"] = {"PYTHONPATH":str(d["root"])+":"+str(d["datasets"])}
        profile["source_descriptors"].extend(pin(root/rel) for rel in relatives)
        d["profile"] = profile
        d["profile_ref"] = write(tmp_path/"Git-profile.json", profile)
        d["binding_ref"] = write(tmp_path/"typed-binding.json", {
            "schema":installer.SCIENTIFIC_PRODUCER_GIT_BINDING_SCHEMA, "profile":d["profile_ref"],
            "source_commitment":d["commitment_ref"], "observer_plan":pin(tmp_path/"plan.json"),
            "producer_source_commitment":producer_ref})
    def git(root_arg, argv, **kwargs):
        if root_arg != root:
            return native_git(root_arg, argv, **kwargs)
        assert kwargs["timeout"]==3 and kwargs["environment"]["GIT_CONFIG_GLOBAL"]=="/dev/null"
        d["producer_calls"].append(argv)
        args=argv[4:]
        if args==["rev-parse","--show-toplevel"]:return (str(tmp_path/"foreign") if d["producer_fault"]=="root" else str(root))+"\n"
        if args==["rev-parse","--verify","HEAD"]:return ("e"*40 if d["producer_fault"]=="head" else d["producer_head"])+"\n"
        if args==["rev-parse",d["producer_head"]+"^{tree}"]:return ("e"*40 if d["producer_fault"]=="tree" else d["producer_tree"])+"\n"
        if args==["status","--porcelain=v1","-z","--untracked-files=all"]:return "?? foreign\0" if d["producer_fault"]=="dirty" else ""
        if args==["ls-tree","-r","-z","--full-tree",d["producer_head"]]:
            return "".join(("100755" if row["mode"]&0o111 else "100644")+" blob "+row["git_blob"]+"\t"+rel+"\0"
                for rel,row in rows.items() if d["producer_fault"]!="inventory" or rel!=relatives[-1])
        raise AssertionError(args)
    d["seal"] = seal;seal();monkeypatch.setattr(owners,"git_command",git)
    return d


@pytest.mark.parametrize("mode", [0o600,0o644,0o664])
def test_separate_producer_Git_binding_preserves_native_roles_and_physical_modes(tmp_path,monkeypatch,mode):
    d=separate_producer_fixture(tmp_path,monkeypatch,mode)
    native=copy.deepcopy(d["plan"]);original_request=copy.deepcopy(d["request"])
    parsed=installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])
    assert parsed==d["profile"] and d["plan"]==native and d["request"]==original_request
    assert len(d["producer_calls"])==len(d["git_calls"])==18
    assert parsed["main_entrypoint"]["path"].startswith(str(d["producer_root"])+"/")
    assert all(parsed[role]["path"].startswith(str(d["root"])+"/") for role in ("health_module","owner_source","driver_source"))
    assert all(row["mode"]==mode for row in d["producer_inventory"]["files"])


@pytest.mark.parametrize("failure", ["root","head","tree","dirty","inventory"])
def test_separate_producer_refuses_foreign_or_uncommitted_Git_generation(tmp_path,monkeypatch,failure):
    d=separate_producer_fixture(tmp_path,monkeypatch);d["producer_fault"]=failure
    reason="producer selected genuine Git inventory differs" if failure=="inventory" else "producer genuine Git root/HEAD/tree/clean state differs"
    with pytest.raises(ValueError,match=reason):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])


@pytest.mark.parametrize("failure", ["schema","profile","native_commitment","producer_commitment_extra","producer_inventory_head",
    "producer_duplicate","producer_mode","producer_hash","native_role","extra_producer","missing_parent","pythonpath","entrypoint"])
def test_separate_producer_refuses_mixed_profile_and_source_metadata(tmp_path,monkeypatch,failure):
    d=separate_producer_fixture(tmp_path,monkeypatch)
    typed=json.loads(Path(d["binding_ref"]["path"]).read_bytes());profile=d["profile"]
    if failure=="schema":typed["schema"]=installer.SCIENTIFIC_GIT_BINDING_SCHEMA;reason="closed explicit scientific Git/profile binding required"
    elif failure=="profile":typed["profile"]={**d["profile_ref"],"sha256":"0"*64};reason="closed explicit scientific Git/profile binding required"
    elif failure=="native_commitment":typed["source_commitment"]=typed["producer_source_commitment"];reason="scientific source-only Git commitment required"
    elif failure=="producer_commitment_extra":
        c={**d["producer_commitment"],"allow_external_root":True};typed["producer_source_commitment"]=write(tmp_path/"wrong-producer-commitment.json",c);reason="closed source-only producer Git commitment required"
    elif failure=="producer_inventory_head":
        inv={**d["producer_inventory"],"source_commit":"e"*40};c={**d["producer_commitment"],"source_inventory":write(tmp_path/"wrong-producer-inventory.json",inv)};typed["producer_source_commitment"]=write(tmp_path/"wrong-producer-commitment.json",c);reason="producer inventory generation differs"
    elif failure=="producer_duplicate":
        inv=copy.deepcopy(d["producer_inventory"]);inv["files"].append(copy.deepcopy(inv["files"][0]));inv["file_count"]+=1;inv["source_bytes"]+=inv["files"][0]["bytes"]
        c={**d["producer_commitment"],"source_inventory":write(tmp_path/"wrong-producer-inventory.json",inv)};typed["producer_source_commitment"]=write(tmp_path/"wrong-producer-commitment.json",c);reason="producer inventory path/body/mode identity differs"
    elif failure in("producer_mode","producer_hash"):
        target=next(row for rel,row in d["producer_rows"].items() if rel.endswith("scientific_executor.py"))
        if failure=="producer_mode":Path(target["path"]).chmod(0o640)
        else:Path(target["path"]).write_bytes(b"# changed producer body\n")
        reason="producer physical source/mode differs" if failure=="producer_mode" else "repair descriptor changed"
    elif failure=="native_role":profile["owner_source"]=profile["main_entrypoint"];reason="exact separate producer entrypoint/parents and retained native roles required"
    elif failure=="extra_producer":
        target=d["producer_root"]/"extra.py";target.write_text("# unrelated\n");profile["source_descriptors"].append(pin(target));reason="exact separate producer entrypoint/parents and retained native roles required"
    elif failure=="missing_parent":profile["source_descriptors"].remove(pin(d["producer_root"]/"ipfs_accelerate_py/__init__.py"));reason="exact separate producer entrypoint/parents and retained native roles required"
    elif failure=="pythonpath":profile["unit"]["environment"]["PYTHONPATH"]=str(d["producer_root"])+":"+str(d["datasets"]);reason="separate producer must preserve native-only Python search roots and exact entrypoint"
    elif failure=="entrypoint":profile["unit"]["argv"][1]=d["roles"]["main_entrypoint"]["path"];reason="separate producer must preserve native-only Python search roots and exact entrypoint"
    if failure in("native_role","extra_producer","missing_parent","pythonpath","entrypoint"):
        typed["profile"]=write(Path(d["profile_ref"]["path"]),profile);d["profile_ref"]=typed["profile"]
    d["binding_ref"]=write(Path(d["binding_ref"]["path"]),typed)
    with pytest.raises(ValueError,match=reason):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])


def test_separate_producer_masked_live_body_cannot_replace_committed_blob(tmp_path,monkeypatch):
    d=separate_producer_fixture(tmp_path,monkeypatch)
    rel="ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py"
    row=d["producer_rows"][rel];path=Path(row["path"]);path.write_text("# assume-unchanged cannot authorize this body\n")
    row.update(pin(path));d["seal"]()
    with pytest.raises(ValueError,match="producer physical source/mode differs from its committed Git blob"):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])


def test_separate_producer_configuration_preserves_native_scope_and_engineering_board(tmp_path,monkeypatch):
    d=separate_producer_fixture(tmp_path,monkeypatch);before=copy.deepcopy(d["config"])
    actual=installer._repair_scientific_configuration(d["config"],d["profile_ref"],d["release"],d["binding_ref"])
    engineering=next(board for board in actual["boards"] if board["id"]=="legal-ir")
    science=next(board for board in actual["boards"] if board["id"]=="legal-ir-scientific")
    assert engineering==next(board for board in before["boards"] if board["id"]=="legal-ir")
    assert science["native_executor_git_binding"]==d["binding_ref"] and science["native_executor_binding"]==d["profile_ref"]
    assert d["plan"]["accelerate_root"]==str(d["root"])
    assert d["profile"]["unit"]["environment"]["PYTHONPATH"]==str(d["root"])+":"+str(d["datasets"])
    assert d["config"]==before


def test_separate_producer_drift_during_native_read_is_refused_before_return(tmp_path,monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d=separate_producer_fixture(tmp_path,monkeypatch);original=owners.git_command;changed=[]
    def observed(root,argv,**kwargs):
        if root==d["root"] and argv[4:]==["ls-tree","-r","-z","--full-tree",d["head"]] and not changed:
            Path(d["profile"]["main_entrypoint"]["path"]).write_text("# producer drift after initial producer Git read\n")
            changed.append(True)
        return original(root,argv,**kwargs)
    monkeypatch.setattr(owners,"git_command",observed)
    with pytest.raises(ValueError,match="repair descriptor changed"):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])
    assert changed==[True]
