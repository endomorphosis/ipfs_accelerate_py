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


def catalogue_producer_fixture(tmp_path, monkeypatch, mode=0o600):
    """Modeled @3/@2 producer and @1 owner metadata; execute no body."""
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = separate_producer_fixture(tmp_path, monkeypatch, mode)
    owner_relative = "ipfs_accelerate_py/agent_supervisor/rescue/original_embedded_observation_owner.py"
    # This extra ordinary fixture source is an explicit qualification input.
    # It is read and hashed, never imported or executed by these owning tests.
    # Historical fixture DATA from local Git80d8e850d927e8bd989c6c4466e83fa1dfa1ec25; never imported or executed.
    old_owner_source = Path(__file__).resolve().parents[1] / "fixtures/agent_supervisor/repair_typed_scientific_git_binding/original_embedded_observation_owner_e243cdf7.py.txt"
    old_owner_raw = old_owner_source.read_bytes()
    assert hashlib.sha256(old_owner_raw).hexdigest() == "e243cdf7b22a63f55e3c55101a2a7aeeee31492a98c8dbfe9f05b4d43b7fdc06"
    old_owner_path = Path(d["roles"]["owner_source"]["path"]); old_owner_path.write_bytes(old_owner_raw)
    d["roles"]["owner_source"] = pin(old_owner_path)
    d["files"][owner_relative].update(pin(old_owner_path))
    d["files"][owner_relative]["git_blob"] = hashlib.sha1(b"blob " + str(len(old_owner_raw)).encode() + b"\0" + old_owner_raw).hexdigest()
    d["original"]["blobs"] = [{"relative_path":relative,"object_id":row["git_blob"],"bytes":row["bytes"]} for relative,row in d["files"].items()]
    d["original"]["payload_bytes"] = sum(row["bytes"] for row in d["files"].values())
    d["inventory"]["source_bytes"] = d["original"]["payload_bytes"]
    d["capsule"]["source_bytes"] = d["original"]["payload_bytes"]
    d["capsule"]["sources"] = copy.deepcopy(list(d["files"].values()))
    cli = d["plan"]["native_metadata_cli"]
    cli["project_import_plan"]["source_pins"] = list(d["roles"].values())
    cli["source_sha256"] = {row["path"]:row["sha256"] for row in d["roles"].values()}
    d["request"]["source_sha256"] = copy.deepcopy(cli["source_sha256"])
    d["request"].update(storage_bytes=384*1024**2,timeout_seconds=300,
        expected_scheduler_policy_sha256="f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661")
    relative = "ipfs_accelerate_py/agent_supervisor/rescue/native_catalogue_probe.py"
    path = d["producer_root"] / relative
    path.write_text("raise AssertionError('catalogue parser metadata body must never execute')\n")
    path.chmod(mode)
    raw = path.read_bytes()
    d["producer_rows"][relative] = {**pin(path), "relative_path":relative, "mode":mode,
        "git_blob":hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}
    native_owner_root = tmp_path / "OW"; native_owner_root.mkdir()
    owner_rows = {}
    for owner_rel in ("ipfs_accelerate_py/__init__.py","ipfs_accelerate_py/agent_supervisor/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/rescue/__init__.py",owner_relative):
        target = native_owner_root/owner_rel; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text("raise AssertionError('native owner metadata body must never execute')\n")
        target.chmod(mode); body=target.read_bytes()
        owner_rows[owner_rel] = {**pin(target),"relative_path":owner_rel,"mode":mode,
            "git_blob":hashlib.sha1(b"blob "+str(len(body)).encode()+b"\0"+body).hexdigest()}
    d.update(catalogue_parser_relative=relative,catalogue_parser_path=path,native_owner_root=native_owner_root,
        native_owner_rows=owner_rows,native_owner_relative=owner_relative,native_owner_head="e"*40,native_owner_tree="f"*40,
        native_owner_calls=[],native_owner_fault=None)
    original_git = owners.git_command
    def owner_git(root,argv,**kwargs):
        if root != native_owner_root:return original_git(root,argv,**kwargs)
        assert kwargs["timeout"] == 3 and kwargs["environment"]["GIT_CONFIG_GLOBAL"] == "/dev/null"
        d["native_owner_calls"].append(argv);args=argv[4:]
        if args == ["rev-parse","--show-toplevel"]:return (str(tmp_path/"foreign") if d["native_owner_fault"] == "root" else str(root))+"\n"
        if args == ["rev-parse","--verify","HEAD"]:return ("a"*40 if d["native_owner_fault"] == "head" else d["native_owner_head"])+"\n"
        if args == ["rev-parse",d["native_owner_head"]+"^{tree}"]:return ("a"*40 if d["native_owner_fault"] == "tree" else d["native_owner_tree"])+"\n"
        if args == ["status","--porcelain=v1","-z","--untracked-files=all"]:return "?? foreign\0" if d["native_owner_fault"] == "dirty" else ""
        if args == ["ls-tree","-r","-z","--full-tree",d["native_owner_head"]]:
            return "".join(("100755" if row["mode"]&0o111 else "100644")+" blob "+row["git_blob"]+"\t"+rel+"\0"
                for rel,row in owner_rows.items() if d["native_owner_fault"] != "inventory" or rel != owner_relative)
        raise AssertionError(args)
    monkeypatch.setattr(owners,"git_command",owner_git)
    old_seal = d["seal"]
    def seal():
        owner_source = pin(native_owner_root/owner_relative)
        d["request"]["source_sha256"][owner_source["path"]] = owner_source["sha256"]
        old_seal()
        parser = pin(path)
        commitment = {**d["producer_commitment"],"schema":"agent-supervisor/scientific-producer-source-commitment@2",
            "catalogue_parser":parser}
        commitment_ref = write(tmp_path/"catalogue-producer-commitment.json",commitment)
        owner_source = pin(native_owner_root/owner_relative)
        owner_inventory = {"schema":"legal-ir-scientific-source-inventory/v1","source_root":str(native_owner_root),
            "source_commit":d["native_owner_head"],"source_tree":d["native_owner_tree"],"file_count":4,
            "source_bytes":sum(row["bytes"] for row in owner_rows.values()),"files":list(owner_rows.values())}
        generation=write(tmp_path/"native-owner-generation.json",{"source_only":True})
        qualified=write(tmp_path/"native-owner-qualification.json",{"modeled_source_fixture_only":True})
        native_commitment = {"schema":installer.SCIENTIFIC_NATIVE_OWNER_POLICY_SOURCE_COMMITMENT_SCHEMA,
            "source_root":str(native_owner_root),"source_commit":d["native_owner_head"],"source_tree":d["native_owner_tree"],
            "source_inventory":write(tmp_path/"native-owner-inventory.json",owner_inventory),
            "source_operation_plan":generation,"source_qualification":qualified,"owner_source":owner_source,
            "prior_owner_source":d["roles"]["owner_source"],
            "native_backend_source":d["roles"]["driver_source"],
            "scheduler_policy_transition":{"prior_sha256":"4662bd8fe6caf6f2c048e3d7cff7abb4d80e5c875b21e02e204b1b89371b8286",
                "target_sha256":"f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661"},
            "runtime_activation_or_task_authority":False}
        native_ref = write(tmp_path/"native-owner-commitment.json",native_commitment)
        profile = copy.deepcopy(d["profile"]); profile["owner_source"] = owner_source
        profile["source_descriptors"].extend([parser,*[pin(native_owner_root/rel) for rel in owner_rows]])
        profile["unit"]["argv"].extend(["--catalogue-parser-source",parser["path"],"--catalogue-parser-sha256",parser["sha256"]])
        d["request"]["source_sha256"][owner_source["path"]] = owner_source["sha256"]
        d["profile"] = profile;d["producer_commitment"] = commitment;d["native_owner_commitment"] = native_commitment
        d["native_owner_inventory"] = owner_inventory
        d["profile_ref"] = write(tmp_path/"Git-profile.json",profile)
        d["binding_ref"] = write(tmp_path/"typed-binding.json",{
            "schema":installer.SCIENTIFIC_CATALOGUE_PRODUCER_GIT_BINDING_SCHEMA,"profile":d["profile_ref"],
            "source_commitment":d["commitment_ref"],"observer_plan":pin(tmp_path/"plan.json"),
            "producer_source_commitment":commitment_ref,"native_owner_source_commitment":native_ref})
    d["seal"]=seal;seal()
    return d


@pytest.mark.parametrize("mode", [0o400,0o600,0o664])
def test_catalogue_producer_Git3_binds_exact_five_files_and_preserves_native_generation(tmp_path,monkeypatch,mode):
    d = catalogue_producer_fixture(tmp_path,monkeypatch,mode)
    native = copy.deepcopy(d["plan"]); request = copy.deepcopy(d["request"])
    before = copy.deepcopy(d["config"])
    profile = installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])
    assert profile == d["profile"] and d["plan"] == native and d["request"] == request and d["config"] == before
    assert len(d["producer_inventory"]["files"]) == 5
    assert len(d["producer_commitment"]) == 9
    assert all(row["mode"] == mode for row in d["producer_inventory"]["files"])
    assert d["producer_head"] != d["head"] and d["producer_root"] != d["source"]
    assert len(d["producer_calls"]) == len(d["native_owner_calls"]) == len(d["git_calls"]) == 18
    assert all(profile[role]["path"].startswith(str(d["root"])+"/") for role in ("health_module","driver_source"))
    assert profile["owner_source"]["path"].startswith(str(d["native_owner_root"])+"/")
    assert len(d["native_owner_commitment"]) == 12 and len(d["native_owner_inventory"]["files"]) == 4
    assert profile["unit"]["environment"]["PYTHONPATH"] == str(d["root"])+":"+str(d["datasets"])


@pytest.mark.parametrize("failure", ["binding_schema_v1","binding_schema_v2","producer_schema_v1",
    "producer_extra_key","missing_parser_key","crossed_parser_descriptor","unbound_probe",
    "missing_probe_row","extra_producer_row","duplicate_probe_row","missing_cli_source","missing_cli_sha",
    "duplicate_cli_source","duplicate_cli_sha","crossed_cli_source","crossed_cli_sha","mode_drift",
    "native_role","pythonpath","binding_extra_key","probe_wrong_git_mode"])
def test_catalogue_producer_Git3_rejects_crossed_missing_extra_schema_CLI_and_mode_roles(tmp_path,monkeypatch,failure):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = catalogue_producer_fixture(tmp_path,monkeypatch)
    typed = json.loads(Path(d["binding_ref"]["path"]).read_bytes())
    commitment = copy.deepcopy(d["producer_commitment"]); profile = copy.deepcopy(d["profile"])
    parser = commitment["catalogue_parser"]; inventory = copy.deepcopy(d["producer_inventory"])
    if failure == "binding_schema_v1": typed["schema"] = installer.SCIENTIFIC_GIT_BINDING_SCHEMA
    elif failure == "binding_schema_v2": typed["schema"] = installer.SCIENTIFIC_PRODUCER_GIT_BINDING_SCHEMA
    elif failure == "producer_schema_v1": commitment["schema"] = "agent-supervisor/scientific-producer-source-commitment@1"
    elif failure == "producer_extra_key": commitment["inferred_parser_allowed"] = True
    elif failure == "missing_parser_key": del commitment["catalogue_parser"]
    elif failure == "crossed_parser_descriptor": commitment["catalogue_parser"] = profile["main_entrypoint"]
    elif failure == "unbound_probe": profile["source_descriptors"].remove(parser)
    elif failure == "missing_probe_row":
        inventory["files"] = [row for row in inventory["files"] if row["relative_path"] != d["catalogue_parser_relative"]]
    elif failure == "extra_producer_row":
        extra = d["producer_root"] / "unrelated.py"; extra.write_text("# not a selected producer role\n")
        raw = extra.read_bytes()
        inventory["files"].append({**pin(extra), "relative_path":"unrelated.py", "mode":0o644,
            "git_blob":hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()})
    elif failure == "duplicate_probe_row":
        inventory["files"].append(copy.deepcopy(d["producer_rows"][d["catalogue_parser_relative"]]))
    elif failure in ("missing_cli_source","missing_cli_sha"):
        flag = "--catalogue-parser-source" if failure == "missing_cli_source" else "--catalogue-parser-sha256"
        index = profile["unit"]["argv"].index(flag); del profile["unit"]["argv"][index:index+2]
    elif failure in ("duplicate_cli_source","duplicate_cli_sha"):
        flag, value = ("--catalogue-parser-source",parser["path"]) if failure == "duplicate_cli_source" else ("--catalogue-parser-sha256",parser["sha256"])
        profile["unit"]["argv"].extend([flag,value])
    elif failure in ("crossed_cli_source","crossed_cli_sha"):
        flag, value = ("--catalogue-parser-source",profile["owner_source"]["path"]) if failure == "crossed_cli_source" else ("--catalogue-parser-sha256","0"*64)
        profile["unit"]["argv"][profile["unit"]["argv"].index(flag)+1] = value
    elif failure == "mode_drift": d["catalogue_parser_path"].chmod(0o640)
    elif failure == "native_role": profile["health_module"] = parser
    elif failure == "pythonpath": profile["unit"]["environment"]["PYTHONPATH"] = str(d["producer_root"])+":"+str(d["datasets"])
    elif failure == "binding_extra_key": typed["infer_catalogue_parser"] = True
    elif failure == "probe_wrong_git_mode":
        original_git = owners.git_command
        def wrong_mode(root,argv,**kwargs):
            response = original_git(root,argv,**kwargs)
            if root == d["producer_root"] and argv[4:] == ["ls-tree","-r","-z","--full-tree",d["producer_head"]]:
                entry = "100644 blob "+d["producer_rows"][d["catalogue_parser_relative"]]["git_blob"]+"\t"+d["catalogue_parser_relative"]
                assert entry in response
                response = response.replace(entry,entry.replace("100644","100755",1),1)
            return response
        monkeypatch.setattr(owners,"git_command",wrong_mode)
    inventory["file_count"] = len(inventory["files"])
    inventory["source_bytes"] = sum(row["bytes"] for row in inventory["files"])
    commitment["source_inventory"] = write(tmp_path/"rejected-catalogue-producer-inventory.json",inventory)
    typed["producer_source_commitment"] = write(tmp_path/"rejected-catalogue-producer-commitment.json",commitment)
    d["profile_ref"] = write(Path(d["profile_ref"]["path"]),profile);typed["profile"] = d["profile_ref"]
    binding = write(Path(d["binding_ref"]["path"]),typed)
    before = copy.deepcopy(d["config"])
    with pytest.raises(ValueError):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],binding)
    assert d["config"] == before


def test_catalogue_producer_Git3_cannot_mask_changed_probe_body_with_clean_Git_metadata(tmp_path,monkeypatch):
    d = catalogue_producer_fixture(tmp_path,monkeypatch)
    row = d["producer_rows"][d["catalogue_parser_relative"]]
    d["catalogue_parser_path"].write_text("raise AssertionError('changed parser body is not its committed blob')\n")
    row.update(pin(d["catalogue_parser_path"]))
    d["seal"]()
    with pytest.raises(ValueError,match="producer physical source/mode differs from its committed Git blob"):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])


def test_catalogue_producer_Git3_probe_drift_during_native_metadata_read_is_refused(tmp_path,monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = catalogue_producer_fixture(tmp_path,monkeypatch); original_git = owners.git_command; changed = []
    def drift(root,argv,**kwargs):
        if root == d["root"] and argv[4:] == ["ls-tree","-r","-z","--full-tree",d["head"]] and not changed:
            d["catalogue_parser_path"].write_text("raise AssertionError('probe drift after initial producer read')\n")
            changed.append(True)
        return original_git(root,argv,**kwargs)
    monkeypatch.setattr(owners,"git_command",drift)
    with pytest.raises(ValueError,match="repair descriptor changed"):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])
    assert changed == [True]


@pytest.mark.parametrize("failure", ["missing_commitment","extra_commitment_key","crossed_schema","wrong_policy",
    "wrong_prior_policy","crossed_owner","crossed_prior","unbound_prior","missing_parent","extra_file",
    "mode_drift","crossed_root","request_policy","request_bounds","request_owner_pin",
    "crossed_backend","git_head","git_tree","transition_extra_key","git_inventory"])
def test_Git3_native_owner_refuses_unbound_crossed_policy_source_and_Git_roles(tmp_path,monkeypatch,failure):
    d = catalogue_producer_fixture(tmp_path,monkeypatch)
    if failure in ("request_policy","request_bounds"):
        if failure == "request_policy": d["request"]["expected_scheduler_policy_sha256"] = "0"*64
        else: d["request"]["memory_mb"] = 1537
        d["seal"]()
    typed = json.loads(Path(d["binding_ref"]["path"]).read_bytes())
    commitment = copy.deepcopy(d["native_owner_commitment"])
    inventory = copy.deepcopy(d["native_owner_inventory"]); profile = copy.deepcopy(d["profile"])
    if failure == "missing_commitment": del typed["native_owner_source_commitment"]
    elif failure == "extra_commitment_key": commitment["infer_native_root"] = True
    elif failure == "crossed_schema": commitment["schema"] = "agent-supervisor/scientific-producer-source-commitment@1"
    elif failure == "wrong_policy": commitment["scheduler_policy_transition"]["target_sha256"] = "a"*64
    elif failure == "wrong_prior_policy": commitment["scheduler_policy_transition"]["prior_sha256"] = "b"*64
    elif failure == "crossed_owner": commitment["owner_source"] = profile["main_entrypoint"]
    elif failure == "crossed_prior": commitment["prior_owner_source"] = profile["health_module"]
    elif failure == "unbound_prior": profile["source_descriptors"].remove(commitment["prior_owner_source"])
    elif failure == "crossed_backend": commitment["native_backend_source"] = profile["health_module"]
    elif failure == "transition_extra_key": commitment["scheduler_policy_transition"]["allow_other_policy"] = True
    elif failure == "missing_parent":
        inventory["files"] = [row for row in inventory["files"] if row["relative_path"] != "ipfs_accelerate_py/__init__.py"]
    elif failure == "extra_file":
        extra = d["native_owner_root"] / "unrelated.py"; extra.write_text("# no native owner role\n"); raw = extra.read_bytes()
        inventory["files"].append({**pin(extra),"relative_path":"unrelated.py","mode":0o644,
            "git_blob":hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()})
    elif failure == "mode_drift": Path(commitment["owner_source"]["path"]).chmod(0o640)
    elif failure == "crossed_root": commitment["source_root"] = str(d["producer_root"])
    elif failure == "request_owner_pin":
        request = copy.deepcopy(d["request"]); request["source_sha256"].pop(commitment["owner_source"]["path"])
        request_ref = write(tmp_path/"request.json",request)
        template = json.loads((tmp_path/"template.json").read_bytes()); template["request_ref"] = request_ref
        service = json.loads((tmp_path/"service.json").read_bytes()); service["idle_template_ref"] = write(tmp_path/"template.json",template)
        service_ref = write(tmp_path/"service.json",service)
        profile["source_descriptors"] = [row for row in profile["source_descriptors"] if row["path"] != service_ref["path"]] + [service_ref]
        profile["unit"]["argv"][profile["unit"]["argv"].index("--config-sha256")+1] = service_ref["sha256"]
    elif failure.startswith("git_"): d["native_owner_fault"] = failure[len("git_"):]
    inventory["file_count"] = len(inventory["files"])
    inventory["source_bytes"] = sum(row["bytes"] for row in inventory["files"])
    commitment["source_inventory"] = write(tmp_path/"rejected-native-owner-inventory.json",inventory)
    if failure != "missing_commitment":
        typed["native_owner_source_commitment"] = write(tmp_path/"rejected-native-owner-commitment.json",commitment)
    profile_ref = write(Path(d["profile_ref"]["path"]),profile); typed["profile"] = profile_ref
    binding = write(Path(d["binding_ref"]["path"]),typed); before = copy.deepcopy(d["config"])
    with pytest.raises(ValueError):
        installer._repair_scientific_profile(profile_ref,d["release"],binding)
    assert d["config"] == before


def test_Git3_native_owner_cannot_mask_changed_body_with_clean_Git_metadata(tmp_path,monkeypatch):
    d = catalogue_producer_fixture(tmp_path,monkeypatch)
    row = d["native_owner_rows"][d["native_owner_relative"]]; path = Path(row["path"])
    path.write_text("raise AssertionError('changed native owner is not its committed blob')\n")
    row.update(pin(path)); d["seal"]()
    with pytest.raises(ValueError,match="native owner physical source/mode differs from its committed Git blob"):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])


def test_Git3_native_owner_drift_during_native_metadata_read_is_refused(tmp_path,monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = catalogue_producer_fixture(tmp_path,monkeypatch); original_git = owners.git_command; changed = []
    def drift(root,argv,**kwargs):
        if root == d["root"] and argv[4:] == ["ls-tree","-r","-z","--full-tree",d["head"]] and not changed:
            Path(d["native_owner_commitment"]["owner_source"]["path"]).write_text("raise AssertionError('owner drift after initial source read')\n")
            changed.append(True)
        return original_git(root,argv,**kwargs)
    monkeypatch.setattr(owners,"git_command",drift)
    with pytest.raises(ValueError,match="repair descriptor changed"):
        installer._repair_scientific_profile(d["profile_ref"],d["release"],d["binding_ref"])
    assert changed == [True]


# Explicit installer-only retention scope. Git observations and the command
# renderer are UNIT models; actual health parsing, descriptor reads, local CAS
# and installed-readiness validation remain real. No scientific executor,
# Source/Native admission, providers or systemd operations run in these cases.
def _held_science_file_identity(path):
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_gid',
              'st_size', 'st_mtime_ns', 'st_ctime_ns')
    value = Path(path).lstat()
    return {key: getattr(value, key) for key in fields}


def _held_science_retention_fixture(tmp_path, monkeypatch, *, reverse=False, stale=True):
    from ipfs_accelerate_py.agent_supervisor.rescue import native_executor_health as health
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = git_fixture(tmp_path, monkeypatch, reverse)
    # Keep the genuine strict health module in both authenticated role paths.
    # The existing Git fixture models observations but hashes the actual bodies.
    raw = Path(health.__file__).read_bytes()
    hp = Path(d['roles']['health_module']['path'])
    hp.write_bytes(raw)
    d['roles']['health_module'] = pin(hp)
    (d['release'] / 'ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py').write_bytes(raw)
    rel = hp.relative_to(d['root']).as_posix()
    row = d['files'][rel]
    row.update(d['roles']['health_module'])
    row['git_blob'] = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    next(x for x in d['capsule']['sources'] if x['relative_path'] == rel).update(row)
    original_row = next(x for x in d['original']['blobs'] if x['relative_path'] == rel)
    original_row.update(bytes=len(raw), object_id=row['git_blob'])
    total = sum(x['bytes'] for x in d['files'].values())
    d['inventory']['source_bytes'] = d['capsule']['source_bytes'] = total
    d['original']['payload_bytes'] = total
    cli = d['plan']['native_metadata_cli']
    cli['project_import_plan']['source_pins'] = list(d['roles'].values())
    cli['source_sha256'][str(hp)] = row['sha256']
    d['request']['source_sha256'][str(hp)] = row['sha256']
    d['seal']()
    local_d = d['datasets'] / 'historical_D_source.py'
    local_d.write_text('# Historical scientific dependency.\n')
    historical_d = pin(local_d)
    interpreter = Path(sys.executable).resolve()
    profile = copy.deepcopy(d['profile'])
    profile.update(schema=health.PROFILE_SCHEMA,
        receipt_path=str(tmp_path / 'scientific-proof' / 'receipt.json'),
        proof_root=str(tmp_path / 'scientific-proof'), max_age_seconds=180,
        max_cycle_seconds=180,
        cycle_contract={'plan_schema': 'retained-plan/v1', 'plan_operation': 'retained-idle'})
    profile['unit'].update(name='fixture-scientific.service', interpreter=str(interpreter),
        environment={'PYTHONDONTWRITEBYTECODE': '1'})
    profile['unit']['argv'][0] = str(interpreter)
    profile['source_descriptors'].extend((pin(interpreter), historical_d))
    d['profile'] = profile
    d['profile_ref'] = write(Path(d['profile_ref']['path']), profile)
    typed = json.loads(Path(d['binding_ref']['path']).read_bytes())
    typed['profile'] = d['profile_ref']
    d['binding_ref'] = write(Path(d['binding_ref']['path']), typed)
    config = installer._repair_scientific_configuration(
        d['config'], d['profile_ref'], d['release'], d['binding_ref'])
    hold = tmp_path / 'scientific' / 'watchdog.hold'
    hold.parent.mkdir()
    hold.write_text('Retained scientific review and original authority.\n')
    hold.chmod(0o600)
    science = next(x for x in config['boards'] if x['id'] == 'legal-ir-scientific')
    science['hold_files'] = [str(hold)]
    science.update(max_repair_attempts=3, stall_seconds=600,
        recovery_policy={'profile': 'infrastructure-only/v1', 'max_actions_per_window': 3})
    engineering = next(x for x in config['boards'] if x['id'] == 'legal-ir')
    engineering['repair_authorization_epoch'] = {
        'schema': 'agent-supervisor/finite-repair-authorization@1',
        'epoch_id': 'already-expired-unchanged', 'max_attempts': 1,
        'not_before_observed_at': 1, 'expires_at': 2}
    config.update(state_dir=str(tmp_path / 'state'), runtime_release=str(d['release']))
    # Only the nested command renderer is modeled. Distinct policy-bound argv,
    # descriptor and timeout values are compared by the real retention owner.
    renderer_calls = []
    def render(policy_ref, *, validation_timeout_seconds, adoption_timeout_seconds):
        renderer_calls.append(copy.deepcopy(policy_ref))
        commands = {}
        for role, action, seconds in (
                ('validation_owner', 'validate', validation_timeout_seconds),
                ('adoption_owner', 'adopt', adoption_timeout_seconds)):
            prefix = [str(interpreter), '-I', '-B', str(Path(policy_ref['path']).parent / 'owner.py')]
            arguments = ['--policy', policy_ref['path'], '--policy-sha256', policy_ref['sha256']]
            commands[role] = {'argv': prefix + [action] + arguments,
                'verify_argv': prefix + ['verify-validation' if action == 'validate' else 'verify-adoption'] + arguments,
                'source_descriptors': [copy.deepcopy(policy_ref)], 'timeout_seconds': seconds}
        return commands
    monkeypatch.setattr(installer, 'render_control_owner_commands', render)
    policies = {}
    for generation in ('old', 'new'):
        root = tmp_path / (generation + '-control')
        source = root / 'source.py'; source.parent.mkdir()
        source.write_text('# ' + generation + ' normal source.\n')
        trusted = root / 'test_retained.py'; trusted.write_text('def test_retained():\n    assert True\n')
        policy = {'schema': 'agent-supervisor/repair-public-owner-policy@2',
            'control_release': {'root': str(root)},
            'validation': {'timeout_seconds': 300, 'storage_bytes': 67108864,
                'cpu_slots': 1, 'memory_mb': 1536, 'child_process_slots': 2,
                'attempt_root': str(tmp_path / 'original-attempts'),
                'import_sources': [pin(source)], 'trusted_tests': [pin(trusted)],
                'selected_checks': [str(trusted) + '::test_retained']},
            'source_descriptors': [pin(source)], 'adoption': {'retained': True}}
        policies[generation] = write(root / 'policy.json', policy)
    old_commands = render(policies['old'], validation_timeout_seconds=900, adoption_timeout_seconds=900)
    new_commands = render(policies['new'], validation_timeout_seconds=900, adoption_timeout_seconds=900)
    config['repair_worker'].update(job_unit='fixture-original-job.service',
        candidate_policy={'repository_root': str(d['source']), 'base_commit': 'a' * 40},
        candidate_runtime={'schema': 'agent-supervisor/repair-candidate-runtime@1',
            **old_commands, 'resource_bounds': {'runtime_max_seconds': 2400, 'cpu_quota_percent': 100},
            'checkout_timeout_seconds': 30})
    config_path = tmp_path / 'home/.config/ipfs-taskboard-watchdog/legal-ir/fleet.json'
    config_ref = write(config_path, config)
    snapshot = tmp_path / 'old-config-DATA.json'
    snapshot.write_bytes(config_path.read_bytes()); snapshot.chmod(0o400)
    post = copy.deepcopy(config)
    post['repair_worker']['candidate_runtime'].update(new_commands)
    rows = []
    units = {}
    for suffix in ('watchdog', 'repair'):
        path = tmp_path / 'home/.config/systemd/user' / ('ipfs-taskboard-legal-ir-' + suffix + '.service')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('[Service]\nExecStart=/retained/' + suffix + '\n')
        units[suffix] = pin(path)
        rows.append({'before': pin(path), 'postimage_utf8': path.read_text(), 'mode': path.stat().st_mode & 0o777})
    rows.append({'before': config_ref, 'postimage_utf8': json.dumps(post, sort_keys=True),
                 'mode': config_path.stat().st_mode & 0o777})
    fixed = write(tmp_path / 'fixed-DATA.json', {'historical_fixture_DATA': True})
    release_sources = {}
    for suffix in ('watchdog', 'repair'):
        path = d['release'] / ('fleet_' + suffix + '.py')
        path.write_text('# Real fixture source identity: ' + suffix + '\n')
        release_sources['fleet_' + suffix] = pin(path)
    release_commitment = write(d['release'] / 'repair-source-commitment.json', {
        'schema': 'agent-supervisor/repair-source-release@1',
        'source_commitment': 'f' * 64, 'sources': release_sources,
        'selected_sources': {}})
    plan = {'schema': 'agent-supervisor/repair-profile-upgrade-stage@1',
        'files': rows, 'contract': {
            'schema': 'agent-supervisor/repair-profile-upgrade-contract@1',
            'profile': 'legal-ir', 'source_successor': None,
            'config': config_ref, 'prior_job': fixed,
            'inventory': fixed, 'initial_release': release_commitment, 'expected_loaded_before': {},
            'job_unit': 'fixture-original-job.service', 'units': units},
        'release': str(d['release']), 'release_commitment': release_commitment,
        'release_sources': release_sources, 'selected_release_sources': {},
        'state_dir': config['state_dir'], 'source_commitment': 'f' * 64,
        'source_successor': None, 'initial_bindings': None,
        'scientific_executor_binding': d['profile_ref'],
        'scientific_executor_git_binding': d['binding_ref'],
        'held_scientific_profile_retention': {
            'schema': 'agent-supervisor/held-scientific-profile-source-only-retention@1',
            'config_preimage': pin(snapshot), 'profile': d['profile_ref'],
            'git_binding': d['binding_ref'],
            'hold': {'file': pin(hold), 'identity': _held_science_file_identity(hold)}}}
    # Adoption DATA must describe exactly the same source-only transition.
    # The real pure projection API normalizes only policy self-reference slots;
    # neither this fixture nor the installer invents a broader normalization.
    source_snapshot = {'prior_policy': config['repair_worker']['candidate_policy'],
        'new_commit': 'a' * 40, 'new_tree': 'b' * 40,
        'source_commitment': 'f' * 64, 'manifest': {}}
    unit_postimages = {suffix: {**pin(Path(reference['path'])),
        'mode': Path(reference['path']).stat().st_mode & 0o777}
        for suffix, reference in units.items()}
    old_policy = json.loads(Path(policies['old']['path']).read_bytes())
    historical_contract = copy.deepcopy(plan['contract'])
    # A historical descriptor keeps its original canonical target; its body
    # predates the presently selected policy and is not a live-body assertion.
    historical_body = tmp_path / 'pre-bootstrap-config-DATA.json'
    historical_body.write_text('{"historical_pre_bootstrap":true}\n')
    historical_contract['config'] = {**pin(historical_body), 'path': str(config_path)}
    old_anchor = {'schema': owners.INITIAL_PROFILE_BOOTSTRAP_SCHEMA,
        'owner_policy_path': policies['old']['path'],
        'original_contract': historical_contract,
        'source_snapshot': copy.deepcopy(source_snapshot),
        'expected_config_projection': owners._bootstrap_config_projection(
            config, old_policy, policies['old'], policies['old']['path']),
        'expected_unit_postimages': copy.deepcopy(unit_postimages)}
    old_anchor_path = Path(policies['old']['path']).parent / 'bootstrap-DATA.json'
    old_anchor_ref = write(old_anchor_path, old_anchor); old_anchor_path.chmod(0o400)
    old_policy['adoption'] = old_anchor_ref
    policies['old'] = write(Path(policies['old']['path']), old_policy)
    config['repair_worker']['candidate_runtime'].update(render(
        policies['old'], validation_timeout_seconds=900, adoption_timeout_seconds=900))
    config_ref = write(config_path, config)
    snapshot.chmod(0o600); snapshot.write_bytes(config_path.read_bytes()); snapshot.chmod(0o400)
    plan['contract']['config'] = config_ref
    plan['files'][-1]['before'] = config_ref
    plan['held_scientific_profile_retention']['config_preimage'] = pin(snapshot)
    new_policy = json.loads(Path(policies['new']['path']).read_bytes())
    new_anchor = {'schema': owners.INITIAL_PROFILE_BOOTSTRAP_SCHEMA,
        'owner_policy_path': policies['new']['path'],
        'original_contract': copy.deepcopy(plan['contract']),
        'source_snapshot': copy.deepcopy(source_snapshot),
        'expected_config_projection': owners._bootstrap_config_projection(
            post, new_policy, policies['new'], policies['new']['path']),
        'expected_unit_postimages': copy.deepcopy(unit_postimages)}
    new_anchor_path = Path(policies['new']['path']).parent / 'bootstrap-DATA.json'
    new_anchor_ref = write(new_anchor_path, new_anchor); new_anchor_path.chmod(0o400)
    new_policy['adoption'] = new_anchor_ref
    policies['new'] = write(Path(policies['new']['path']), new_policy)
    post['repair_worker']['candidate_runtime'].update(render(
        policies['new'], validation_timeout_seconds=900, adoption_timeout_seconds=900))
    plan['files'][-1]['postimage_utf8'] = json.dumps(post, sort_keys=True)
    if stale:
        local_d.write_text('# Current D owner migrated; historical scientific DATA stays exact.\n')
    d.update(stage=plan, old_config=config, post_config=post, hold=hold,
        historical_D=historical_d, current_D=local_d, real_health=health,
        policies=policies, render=render, renderer_calls=renderer_calls,
        bootstrap_paths={'old': old_anchor_path, 'new': new_anchor_path})
    return d


def _held_science_set_post(d, post):
    d['stage']['files'][-1]['postimage_utf8'] = json.dumps(post, sort_keys=True)


def _held_science_no_strict_profile(monkeypatch):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('Historical retention may not invoke current scientific health parsing')
    monkeypatch.setattr(installer, '_repair_scientific_profile', forbidden)
    return calls


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('stale', [False, True])
def test_held_scientific_retention_keeps_original_profile_and_all_authority(tmp_path, monkeypatch, reverse, stale):
    d = _held_science_retention_fixture(tmp_path, monkeypatch, reverse=reverse, stale=stale)
    before = copy.deepcopy(d['stage']); old = Path(d['stage']['contract']['config']['path']).read_bytes()
    if stale:
        with pytest.raises(d['real_health'].ExecutorHealthError, match='descriptor_changed'):
            d['real_health']._profile(d['profile_ref'])
    else:
        assert d['real_health']._profile(d['profile_ref']) == d['profile']
    calls = _held_science_no_strict_profile(monkeypatch)
    installer._repair_held_scientific_profile(d['stage'])
    installer.verify_repair_upgrade(d['stage'])
    assert d['stage'] == before and Path(d['stage']['contract']['config']['path']).read_bytes() == old
    assert d['old_config']['boards'] == d['post_config']['boards']
    assert d['old_config']['runtime_release'] == d['post_config']['runtime_release']
    assert calls == [] and d['git_calls'] and d['renderer_calls']


@pytest.mark.parametrize('failure', ['schema', 'extra', 'missing_preimage', 'wrong_preimage_hash',
    'mutable_preimage', 'cross_profile', 'cross_git', 'extra_hold_key', 'missing_hold_identity',
    'boolean_inode', 'incomplete_identity', 'wrong_hold_hash'])
def test_held_scientific_retention_rejects_malformed_or_crossed_bindings(tmp_path, monkeypatch, failure):
    d = _held_science_retention_fixture(tmp_path, monkeypatch); plan = d['stage']
    scope = plan['held_scientific_profile_retention']
    if failure == 'schema': scope['schema'] += '-other'
    elif failure == 'extra': scope['authorize_science'] = True
    elif failure == 'missing_preimage': scope.pop('config_preimage')
    elif failure == 'wrong_preimage_hash': scope['config_preimage']['sha256'] = '0' * 64
    elif failure == 'mutable_preimage': Path(scope['config_preimage']['path']).chmod(0o600)
    elif failure == 'cross_profile': scope['profile'] = scope['git_binding']
    elif failure == 'cross_git': scope['git_binding'] = scope['profile']
    elif failure == 'extra_hold_key': scope['hold']['authorize'] = True
    elif failure == 'missing_hold_identity': scope['hold'].pop('identity')
    elif failure == 'boolean_inode': scope['hold']['identity']['st_ino'] = True
    elif failure == 'incomplete_identity': scope['hold']['identity'].pop('st_ctime_ns')
    elif failure == 'wrong_hold_hash': scope['hold']['file']['sha256'] = '0' * 64
    old = Path(plan['contract']['config']['path']).read_bytes(); runner_calls = []
    with pytest.raises(ValueError): installer.apply_repair_upgrade(plan, lambda *a, **k: runner_calls.append(a))
    assert runner_calls == [] and Path(plan['contract']['config']['path']).read_bytes() == old


@pytest.mark.parametrize('failure', ['scientific_epoch', 'engineering_epoch', 'science_probe', 'catalogue',
    'science_budget', 'science_hold_list', 'drop_git', 'drop_profile', 'board_order', 'extra_board',
    'runtime_release', 'route', 'retry', 'resource_budget', 'worker_timeout', 'source_successor',
    'initial_bindings', 'unit_body', 'owner_timeout', 'unrendered_command', 'verify_command'])
def test_held_scientific_retention_refuses_scope_authority_and_unrendered_changes(tmp_path, monkeypatch, failure):
    d = _held_science_retention_fixture(tmp_path, monkeypatch); plan = d['stage']; post = copy.deepcopy(d['post_config'])
    science = next(b for b in post['boards'] if b['id'] == 'legal-ir-scientific')
    engineering = next(b for b in post['boards'] if b['id'] == 'legal-ir')
    runtime = post['repair_worker']['candidate_runtime']
    if failure == 'scientific_epoch': science['repair_authorization_epoch'] = {'max_attempts': 1}
    elif failure == 'engineering_epoch': engineering['repair_authorization_epoch']['expires_at'] += 100
    elif failure == 'science_probe': science['probe']['argv'].append('--bypass')
    elif failure == 'catalogue': science['native_catalogue_binding']['population_sha256'] = '0' * 64
    elif failure == 'science_budget': science['max_repair_attempts'] += 1
    elif failure == 'science_hold_list': science['hold_files'] = []
    elif failure == 'drop_git': science.pop('native_executor_git_binding')
    elif failure == 'drop_profile': science.pop('native_executor_binding')
    elif failure == 'board_order': post['boards'].reverse()
    elif failure == 'extra_board': post['boards'].append({'id': 'another'})
    elif failure == 'runtime_release': post['runtime_release'] = str(tmp_path / 'another-release')
    elif failure == 'route': post['repair_worker']['argv'] = ['another-provider']
    elif failure == 'retry': runtime['validation_admission_retry'] = {'max_attempts': 2}
    elif failure == 'resource_budget': runtime['resource_bounds']['runtime_max_seconds'] += 1
    elif failure == 'worker_timeout': post['repair_worker']['timeout_seconds'] = 901
    elif failure == 'source_successor': plan['source_successor'] = {'not_retention': True}
    elif failure == 'initial_bindings': plan['initial_bindings'] = {'not_retention': True}
    elif failure == 'unit_body': plan['files'][0]['postimage_utf8'] += '# changed\n'
    elif failure == 'owner_timeout': runtime['validation_owner']['timeout_seconds'] += 1
    elif failure == 'unrendered_command': runtime['adoption_owner']['argv'][3] = '/another/owner.py'
    elif failure == 'verify_command': runtime['validation_owner']['verify_argv'].append('--new-scope')
    _held_science_set_post(d, post)
    old = Path(plan['contract']['config']['path']).read_bytes(); actions = []
    with pytest.raises(ValueError): installer.apply_repair_upgrade(plan, lambda *a, **k: actions.append(a))
    assert actions == [] and Path(plan['contract']['config']['path']).read_bytes() == old


@pytest.mark.parametrize('failure', ['removed', 'replaced_equal', 'body', 'mode', 'identity', 'symlink', 'hardlink'])
def test_held_scientific_retention_requires_the_original_ordinary_full_stop(tmp_path, monkeypatch, failure):
    d = _held_science_retention_fixture(tmp_path, monkeypatch); hold = d['hold']; original = hold.read_bytes()
    if failure == 'removed': hold.unlink()
    elif failure == 'replaced_equal':
        replacement = hold.with_name('equal-replacement'); replacement.write_bytes(original); replacement.chmod(0o600); replacement.replace(hold)
    elif failure == 'body': hold.write_text('different review\n')
    elif failure == 'mode': hold.chmod(0o640)
    elif failure == 'identity': d['stage']['held_scientific_profile_retention']['hold']['identity']['st_mtime_ns'] += 1
    elif failure == 'symlink':
        target = hold.with_name('target'); target.write_bytes(original); hold.unlink(); hold.symlink_to(target)
    elif failure == 'hardlink':
        import os
        os.link(hold, hold.with_name('alias'))
    calls = _held_science_no_strict_profile(monkeypatch)
    with pytest.raises(ValueError): installer.verify_repair_upgrade(d['stage'])
    assert calls == []


@pytest.mark.parametrize('field,value', [('ordinary_scratch_profile', {'schema': 'V5'}),
    ('timeout_seconds', 301), ('storage_bytes', 67108865), ('cpu_slots', 2),
    ('memory_mb', 1537), ('child_process_slots', 3), ('validation_admission_retry', {'max_attempts': 2})])
def test_held_scientific_retention_does_not_enable_V5_or_expand_nested_validation(tmp_path, monkeypatch, field, value):
    d = _held_science_retention_fixture(tmp_path, monkeypatch)
    path = Path(d['policies']['new']['path']); policy = json.loads(path.read_bytes())
    policy['validation'][field] = value
    d['policies']['new'] = write(path, policy)
    post = copy.deepcopy(d['post_config'])
    post['repair_worker']['candidate_runtime'].update(d['render'](
        d['policies']['new'], validation_timeout_seconds=900, adoption_timeout_seconds=900))
    _held_science_set_post(d, post)
    with pytest.raises(ValueError): installer.verify_repair_upgrade(d['stage'])
    assert 'ordinary_scratch_profile' not in json.loads(Path(d['policies']['old']['path']).read_bytes())['validation']


@pytest.mark.parametrize('stale', [False, True])
def test_held_scientific_retention_is_explicit_and_default_parser_remains_strict(tmp_path, monkeypatch, stale):
    d = _held_science_retention_fixture(tmp_path, monkeypatch, stale=stale)
    plan = copy.deepcopy(d['stage']); plan.pop('held_scientific_profile_retention')
    if stale:
        with pytest.raises(ValueError, match='descriptor_changed'):
            installer.verify_repair_upgrade(plan)
    else:
        installer.verify_repair_upgrade(plan)
    assert pin(Path(d['profile_ref']['path'])) == d['profile_ref']
    assert (pin(d['current_D']) != d['historical_D']) is stale


@pytest.mark.parametrize('scope', ['unconfigured', 'launch_only', 'aliased_preimage'])
def test_held_scientific_retention_requires_original_full_stop_scope(tmp_path, monkeypatch, scope):
    d = _held_science_retention_fixture(tmp_path, monkeypatch); plan = d['stage']
    if scope == 'aliased_preimage':
        plan['held_scientific_profile_retention']['config_preimage'] = copy.deepcopy(plan['contract']['config'])
    else:
        old = copy.deepcopy(d['old_config']); post = copy.deepcopy(d['post_config'])
        for config in (old, post):
            board = next(b for b in config['boards'] if b['id'] == 'legal-ir-scientific')
            if scope == 'unconfigured': board['hold_files'] = []
            else: board['launch_only_hold_files'] = [str(d['hold'])]
        config_ref = write(Path(plan['contract']['config']['path']), old)
        snapshot = Path(plan['held_scientific_profile_retention']['config_preimage']['path'])
        snapshot.chmod(0o600); snapshot.write_bytes(Path(config_ref['path']).read_bytes()); snapshot.chmod(0o400)
        plan['contract']['config'] = config_ref; plan['files'][-1]['before'] = config_ref
        plan['held_scientific_profile_retention']['config_preimage'] = pin(snapshot)
        _held_science_set_post(d, post)
    with pytest.raises(ValueError): installer.verify_repair_upgrade(plan)


@pytest.mark.parametrize('change', ['test_body', 'test_selector', 'test_outside_control', 'policy_hash_argument'])
def test_held_scientific_retention_preserves_trusted_acceptance_and_four_policy_joins(tmp_path, monkeypatch, change):
    d = _held_science_retention_fixture(tmp_path, monkeypatch)
    path = Path(d['policies']['new']['path']); policy = json.loads(path.read_bytes())
    if change == 'test_body':
        target = Path(policy['validation']['trusted_tests'][0]['path'])
        target.write_text('def test_retained():\n    assert False\n')
        policy['validation']['trusted_tests'] = [pin(target)]
    elif change == 'test_selector': policy['validation']['selected_checks'][0] += '_different'
    elif change == 'test_outside_control':
        target = tmp_path / 'unowned-test.py'; target.write_text('def test_retained():\n    assert True\n')
        policy['validation']['trusted_tests'] = [pin(target)]
    d['policies']['new'] = write(path, policy)
    post = copy.deepcopy(d['post_config'])
    post['repair_worker']['candidate_runtime'].update(d['render'](
        d['policies']['new'], validation_timeout_seconds=900, adoption_timeout_seconds=900))
    if change == 'policy_hash_argument':
        argv = post['repair_worker']['candidate_runtime']['adoption_owner']['verify_argv']
        argv[argv.index('--policy-sha256') + 1] = '0' * 64
    _held_science_set_post(d, post)
    with pytest.raises(ValueError): installer.verify_repair_upgrade(d['stage'])


@pytest.mark.parametrize('fault', ['head', 'dirty', 'driver_body', 'release_verifier'])
def test_held_scientific_retention_still_authenticates_committed_scientific_roles(tmp_path, monkeypatch, fault):
    d = _held_science_retention_fixture(tmp_path, monkeypatch)
    if fault in ('head', 'dirty'): d['git_fault'] = fault
    elif fault == 'driver_body': Path(d['profile']['driver_source']['path']).write_text('# changed original driver\n')
    else: (d['release'] / 'ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py').write_text('# wrong verifier\n')
    calls = _held_science_no_strict_profile(monkeypatch)
    with pytest.raises(ValueError): installer.verify_repair_upgrade(d['stage'])
    assert calls == []


@pytest.mark.parametrize('fault', ['schema', 'profile', 'extra', 'missing'])
def test_held_scientific_retention_requires_exact_original_contract(tmp_path, monkeypatch, fault):
    d = _held_science_retention_fixture(tmp_path, monkeypatch); contract = d['stage']['contract']
    if fault == 'schema': contract['schema'] += '-different'
    elif fault == 'profile': contract['profile'] = 'legal-ir-scientific'
    elif fault == 'extra': contract['authorize_repair'] = True
    else: contract.pop('profile')
    actions = []; old = Path(contract['config']['path']).read_bytes()
    with pytest.raises(ValueError): installer.apply_repair_upgrade(d['stage'], lambda *a, **k: actions.append(a))
    assert actions == [] and Path(contract['config']['path']).read_bytes() == old


@pytest.mark.parametrize('fault', ['source_commitment', 'sources', 'selected_sources', 'contract_binding'])
def test_held_scientific_retention_binds_the_original_immutable_release_generation(tmp_path, monkeypatch, fault):
    d = _held_science_retention_fixture(tmp_path, monkeypatch); plan = d['stage']
    if fault == 'contract_binding':
        plan['contract']['initial_release'] = plan['contract']['inventory']
    else:
        path = Path(plan['release_commitment']['path']); value = json.loads(path.read_bytes())
        if fault == 'source_commitment': value['source_commitment'] = '0' * 64
        elif fault == 'sources': value['sources'] = {}
        else: value['selected_sources'] = {'extra': plan['release_sources']['fleet_watchdog']}
        commitment = write(path, value)
        plan['release_commitment'] = plan['contract']['initial_release'] = commitment
    with pytest.raises(ValueError): installer.verify_repair_upgrade(plan)


@pytest.mark.parametrize('fault', ['source_extra', 'import_duplicate', 'import_renamed', 'policy_schema'])
def test_held_scientific_retention_preserves_source_population_and_policy_identity(tmp_path, monkeypatch, fault):
    d = _held_science_retention_fixture(tmp_path, monkeypatch)
    path = Path(d['policies']['new']['path']); value = json.loads(path.read_bytes())
    if fault == 'source_extra':
        source = path.parent / 'extra.py'; source.write_text('# Unselected control role.\n')
        value['source_descriptors'].append(pin(source))
    elif fault == 'import_duplicate': value['validation']['import_sources'].append(copy.deepcopy(value['validation']['import_sources'][0]))
    elif fault == 'import_renamed': value['validation']['import_sources'][0]['path'] = str(path.parent / 'different_role.py')
    else: value['schema'] = 'agent-supervisor/repair-public-owner-policy@different'
    reference = write(path, value); post = copy.deepcopy(d['post_config'])
    post['repair_worker']['candidate_runtime'].update(d['render'](
        reference, validation_timeout_seconds=900, adoption_timeout_seconds=900))
    _held_science_set_post(d, post)
    with pytest.raises(ValueError): installer.verify_repair_upgrade(d['stage'])


@pytest.mark.parametrize('fault', ['schema', 'source_snapshot', 'future_engineering_epoch',
    'future_science_authority', 'future_route', 'future_state_target', 'future_budget',
    'config_target', 'unit_target', 'unit_body', 'prior_job_target', 'policy_target'])
def test_held_scientific_retention_rejects_resealed_bootstrap_authority_and_target_changes(tmp_path, monkeypatch, fault):
    d = _held_science_retention_fixture(tmp_path, monkeypatch)
    path = d['bootstrap_paths']['new']; bootstrap = json.loads(path.read_bytes())
    projection = bootstrap['expected_config_projection']
    if fault == 'schema': bootstrap['schema'] += '-another'
    elif fault == 'source_snapshot': bootstrap['source_snapshot']['new_tree'] = '0' * 40
    elif fault == 'future_engineering_epoch':
        next(b for b in projection['boards'] if b['id'] == 'legal-ir')['repair_authorization_epoch']['expires_at'] += 10000
    elif fault == 'future_science_authority':
        next(b for b in projection['boards'] if b['id'] == 'legal-ir-scientific')['max_repair_attempts'] += 1
    elif fault == 'future_route': projection['repair_worker']['argv'] = ['another-provider']
    elif fault == 'future_state_target': projection['state_dir'] = str(tmp_path / 'different-state')
    elif fault == 'future_budget': projection['repair_worker']['candidate_runtime']['resource_bounds']['runtime_max_seconds'] += 1
    elif fault == 'config_target': bootstrap['original_contract']['config']['path'] = str(tmp_path / 'another-fleet.json')
    elif fault == 'unit_target': bootstrap['expected_unit_postimages']['watchdog']['path'] = str(tmp_path / 'another.service')
    elif fault == 'unit_body': bootstrap['expected_unit_postimages']['repair']['sha256'] = '0' * 64
    elif fault == 'prior_job_target': bootstrap['original_contract']['prior_job']['path'] = str(tmp_path / 'another-job.json')
    else: bootstrap['owner_policy_path'] = str(tmp_path / 'another-policy.json')
    path.chmod(0o600); adoption = write(path, bootstrap); path.chmod(0o400)
    policy_path = Path(d['policies']['new']['path']); policy = json.loads(policy_path.read_bytes())
    policy['adoption'] = adoption
    policy_ref = write(policy_path, policy)
    post = copy.deepcopy(d['post_config'])
    post['repair_worker']['candidate_runtime'].update(d['render'](
        policy_ref, validation_timeout_seconds=900, adoption_timeout_seconds=900))
    _held_science_set_post(d, post)
    # Both descriptor hashes and all four argv joins are internally consistent.
    # The refusal must therefore come from the bound bootstrap semantics.
    assert pin(path) == adoption and pin(policy_path) == policy_ref
    assert json.loads(policy_path.read_bytes())['adoption'] == adoption
    old = Path(d['stage']['contract']['config']['path']).read_bytes(); actions = []
    with pytest.raises(ValueError):
        installer.apply_repair_upgrade(d['stage'], lambda *a, **k: actions.append(a))
    assert actions == [] and Path(d['stage']['contract']['config']['path']).read_bytes() == old


@pytest.mark.parametrize('change', ['remove', 'replace', 'body'])
def test_held_scientific_retention_rechecks_full_stop_between_CAS_rows(tmp_path, monkeypatch, change):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = _held_science_retention_fixture(tmp_path, monkeypatch); plan = d['stage']
    original = installer.atomic_write; writes = []; actions = []
    monkeypatch.setattr(owners, 'verify_original_job_closed', lambda *a, **k: None)
    def write_then_change(path, raw, **kwargs):
        original(path, raw, **kwargs); writes.append(str(path))
        if len(writes) == 1:
            if change == 'remove': d['hold'].unlink()
            elif change == 'replace':
                q = d['hold'].with_name('replacement'); q.write_bytes(d['hold'].read_bytes()); q.chmod(0o600); q.replace(d['hold'])
            else: d['hold'].write_text('changed after first CAS\n')
    monkeypatch.setattr(installer, 'atomic_write', write_then_change)
    old = Path(plan['contract']['config']['path']).read_bytes()
    with pytest.raises(ValueError): installer.apply_repair_upgrade(plan, lambda *a, **k: actions.append(a))
    assert writes == [plan['files'][0]['before']['path']] and actions == []
    assert Path(plan['contract']['config']['path']).read_bytes() == old


def test_held_scientific_retention_cannot_override_original_active_job(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = _held_science_retention_fixture(tmp_path, monkeypatch); calls = []; writes = []; actions = []
    def active_job(unit, runner, **kwargs):
        calls.append(unit)
        raise ValueError('original job is still active')
    monkeypatch.setattr(owners, 'verify_original_job_closed', active_job)
    monkeypatch.setattr(installer, 'atomic_write', lambda *a, **k: writes.append(a))
    with pytest.raises(ValueError, match='original job is still active'):
        installer.apply_repair_upgrade(d['stage'], lambda *a, **k: actions.append(a))
    assert calls == ['fixture-original-job.service'] and writes == actions == []


def test_held_scientific_retention_installs_and_observes_only_the_two_existing_units(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = _held_science_retention_fixture(tmp_path, monkeypatch); plan = d['stage']; commands = []; closed = []
    hold_before = _held_science_file_identity(d['hold']); old = copy.deepcopy(d['old_config'])
    monkeypatch.setattr(owners, 'verify_original_job_closed', lambda unit, *a, **k: closed.append(unit))
    strict = _held_science_no_strict_profile(monkeypatch)
    states = {}
    for index, suffix in enumerate(('watchdog', 'repair'), 1):
        unit = 'ipfs-taskboard-legal-ir-' + suffix + '.service'
        states[unit] = {'owner': {'pid': 700 + index, 'birth': str(1000 + index)},
                        'invocation': str(index) * 32, 'suffix': suffix}
    monkeypatch.setattr(owners, 'proc_identity', lambda pid: next(
        (v['owner'] for v in states.values() if v['owner']['pid'] == pid), None))
    def runner(spec, *, cwd, timeout):
        argv = spec['argv']; commands.append(list(argv))
        assert 0 < timeout <= 45 and cwd == '/'
        if 'show' in argv:
            state = states[argv[3]]
            return {'returncode': 0, 'stdout': 'MainPID=' + str(state['owner']['pid']) + '\nInvocationID=' +
                state['invocation'] + '\nActiveState=active\nResult=success\n'}
        return {'returncode': 0, 'stdout': ''}
    result = installer.apply_repair_upgrade(plan, runner)
    assert result['completed'] is True and closed == ['fixture-original-job.service']
    assert commands == [['systemctl', '--user', 'daemon-reload'],
        ['systemctl', '--user', 'restart', 'ipfs-taskboard-legal-ir-watchdog.service'],
        ['systemctl', '--user', 'restart', 'ipfs-taskboard-legal-ir-repair.service']]
    current = json.loads(Path(plan['contract']['config']['path']).read_bytes())
    assert current['boards'] == old['boards'] and current == d['post_config']
    for value in states.values():
        target = Path(plan['state_dir']) / 'runtime-loaded' / (value['invocation'] + '-' + value['suffix'] + '.json')
        write(target, {'owner': value['owner'], 'invocation_id': value['invocation'],
            'source_commitment': plan['source_commitment'], 'release': plan['release'],
            'loaded_module': plan['release_sources']['fleet_' + value['suffix']],
            'config': pin(Path(plan['contract']['config']['path']))})
        target.chmod(0o400)
    observed = installer.observe_repair_upgrade(plan, runner)
    assert set(observed['loaded_runtime']) == {'watchdog', 'repair'}
    assert observed['runtime_unit_config_CAS_closed'] is True
    assert _held_science_file_identity(d['hold']) == hold_before and strict == []
    assert all('scientific.service' not in ' '.join(command) for command in commands)


def _held_science_maintenance_fixture(tmp_path, monkeypatch):
    """UNIT maintenance DATA, not an actual eighteen-case execution witness."""
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = _held_science_retention_fixture(tmp_path, monkeypatch)
    plan = d['stage']; base = d['release']; current = base.with_name('maintxx'); current.mkdir()
    # Actual local ordinary files bind both maps. Only one selected code body
    # changes; both scientific parser bodies remain the original exact bytes.
    base_doc_path = Path(plan['contract']['initial_release']['path'])
    base_doc = json.loads(base_doc_path.read_bytes())
    relative = 'fleet_watchdog.py'
    base_doc.update(snapshot_commit='a' * 40, snapshot_tree='b' * 40,
                    selected_sources={relative: base_doc['sources']['fleet_watchdog']})
    base_ref = write(base_doc_path, base_doc)
    plan['contract']['initial_release'] = base_ref
    current_sources = {}
    for role, source in base_doc['sources'].items():
        source_path = Path(source['path']); target = current / source_path.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = source_path.read_bytes() + (b'# Qualified UNIT maintenance source change.\n' if role == 'fleet_watchdog' else b'')
        target.write_bytes(body); current_sources[role] = pin(target)
    for name in ('native_executor_health.py', 'native_embedded_observation_driver.py'):
        source = base / 'ipfs_accelerate_py/agent_supervisor/rescue' / name
        target = current / source.relative_to(base); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    current_selected = {relative: current_sources['fleet_watchdog']}
    plan.update(release=str(current), source_commitment='1' * 64,
                release_sources=current_sources, selected_release_sources=current_selected)
    old_units = {}; current_units = {}
    for suffix in ('watchdog', 'repair'):
        row = next(row for row in plan['files'][:-1] if Path(row['before']['path']).name.endswith('-' + suffix + '.service'))
        target = Path(row['before']['path'])
        body = ('[Service]\nExecStart=' + str(current / ('fleet_' + suffix + '.py')) + '\n').encode()
        target.write_bytes(body); reference = pin(target)
        row.update(before=reference, postimage_utf8=body.decode())
        plan['contract']['units'][suffix] = reference
        prior = body.replace(str(current).encode(), str(base).encode())
        old_units[suffix] = {'path': str(target), 'bytes': len(prior),
            'sha256': hashlib.sha256(prior).hexdigest(), 'mode': row['mode']}
        current_units[suffix] = {**reference, 'mode': row['mode']}
    old_config = copy.deepcopy(d['old_config']); old_config['runtime_release'] = str(current)
    post = copy.deepcopy(d['post_config']); post['runtime_release'] = str(current)
    old_policy_path = Path(d['policies']['old']['path']); old_policy = json.loads(old_policy_path.read_bytes())
    old_anchor_path = d['bootstrap_paths']['old']; old_anchor = json.loads(old_anchor_path.read_bytes())
    old_anchor['original_contract']['initial_release'] = base_ref
    old_anchor['original_contract']['units'] = {
        suffix: {key: value for key, value in row.items() if key != 'mode'} for suffix, row in old_units.items()}
    old_anchor['expected_unit_postimages'] = old_units
    old_projection_config = copy.deepcopy(old_config); old_projection_config['runtime_release'] = str(base)
    old_anchor['expected_config_projection'] = owners._bootstrap_config_projection(
        old_projection_config, old_policy, d['policies']['old'], str(old_policy_path))
    old_anchor_path.chmod(0o600); old_anchor_ref = write(old_anchor_path, old_anchor); old_anchor_path.chmod(0o400)
    old_policy['adoption'] = old_anchor_ref
    d['policies']['old'] = write(old_policy_path, old_policy)
    old_config['repair_worker']['candidate_runtime'].update(d['render'](
        d['policies']['old'], validation_timeout_seconds=900, adoption_timeout_seconds=900))
    config_ref = write(Path(plan['contract']['config']['path']), old_config)
    plan['contract']['config'] = config_ref; plan['files'][-1]['before'] = config_ref
    copy_path = Path(plan['held_scientific_profile_retention']['config_preimage']['path'])
    copy_path.chmod(0o600); copy_path.write_bytes(Path(config_ref['path']).read_bytes()); copy_path.chmod(0o400)
    plan['held_scientific_profile_retention']['config_preimage'] = pin(copy_path)
    new_policy_path = Path(d['policies']['new']['path']); new_policy = json.loads(new_policy_path.read_bytes())
    new_anchor_path = d['bootstrap_paths']['new']; new_anchor = json.loads(new_anchor_path.read_bytes())
    new_anchor['original_contract'] = copy.deepcopy(plan['contract'])
    new_anchor['expected_unit_postimages'] = current_units
    new_anchor['expected_config_projection'] = owners._bootstrap_config_projection(
        post, new_policy, d['policies']['new'], str(new_policy_path))
    new_anchor_path.chmod(0o600); new_anchor_ref = write(new_anchor_path, new_anchor); new_anchor_path.chmod(0o400)
    new_policy['adoption'] = new_anchor_ref
    d['policies']['new'] = write(new_policy_path, new_policy)
    post['repair_worker']['candidate_runtime'].update(d['render'](
        d['policies']['new'], validation_timeout_seconds=900, adoption_timeout_seconds=900))
    _held_science_set_post(d, post)
    original_raw = Path(config_ref['path']).read_bytes()
    reconstructed = original_raw.replace(str(current).encode(), str(base).encode())
    model_root = tmp_path / 'modeled-maintenance-history'; model_root.mkdir()
    root_closeout = {'schema': 'legal-ir-native-executor-health-original-capture/v1',
        'completed': True, 'returncode': 0, 'child_reaped': True, 'original_CLI_exit_observed': True,
        'original_native_compute_closed': True, 'source_READ_closed': True,
        'source_custody_closed': True, 'source_pins_unchanged': True,
        'cleanup_errors': [], 'received_signals': [], 'native_task_or_service_mutations': False,
        'automatic_retry_or_storage_cleanup': False}
    qualified = {'schema': 'independent-actual-normal82-scope18-source-qualification-acceptance/v1',
        'accepted_actual_source_qualification': True, 'selected_cases': 18, 'pytest_testcase_rows': 18,
        'actual18collection_matches_sealed_scope': True,
        'errors': 0, 'failures': 0, 'skipped': 0,
        'original_Native_compute_closed': True, 'original_R_A_READ_closed': True,
        'original_processes_reaped_and_absent': True, 'storage_released_and_artifacts_durable': True,
        'live_runtime_upgrade_or_SCI_recovery_or_legal_Lean_authority': False,
        'qualified_source_hashes': {relative: current_sources['fleet_watchdog']['sha256']}}
    publication = {'schema': 'Root-qualified18-existing-PR321-actual-readback-publication/v2',
        'completed': True, 'cases': 18, 'errors': 0, 'failures': 0, 'skipped': 0,
        'commit': 'c' * 40, 'no_local_checkout_or_live_service_or_epoch_or_hold_or_task_mutations': True,
        'forced_update': False}
    basis = {'schema': 'operator-runtime-maintenance-stage-basis/v1', 'source_only': True,
        'expected_release': str(current), 'expected_source_commitment': plan['source_commitment'],
        'old_release_commitment': copy.deepcopy(base_ref),
        'config': {'path': config_ref['path'], 'bytes': len(reconstructed), 'sha256': hashlib.sha256(reconstructed).hexdigest()},
        'exact_raw_config_postimage': copy.deepcopy(config_ref),
        'substitutions': {relative: copy.deepcopy(current_sources['fleet_watchdog'])}}
    peer = {'schema': 'independent-operator-code-only-maintenance-stage-SOURCE-peer/v1',
        'accepted_for_metadata_stage_preparation': True, 'blocking_findings': [],
        'unit_postimages': {suffix: {key: row[key] for key in ('bytes', 'sha256')}
                            for suffix, row in current_units.items()}}
    maintenance = {'base_release_commitment': copy.deepcopy(base_ref),
        'base_genuine_snapshot_commit': 'a' * 40, 'base_genuine_snapshot_tree': 'b' * 40,
        'published_patch_commit': 'c' * 40, 'new84_file_Git_snapshot_claimed': False,
        'productive_repair_enrollment_or_successor_or_new_grant': False}
    def sealed(name, value):
        path = model_root / name
        if path.exists(): path.chmod(0o600)
        reference = write(path, value); path.chmod(0o400)
        return reference
    def reseal():
        root_ref = sealed('root-closeout-UNIT-DATA.json', root_closeout)
        qualified['root_closeout'] = root_ref
        qualified_ref = sealed('qualification18-UNIT-DATA.json', qualified)
        publication_ref = sealed('publication18-UNIT-DATA.json', publication)
        basis.update(actual18_peer=qualified_ref, actual18_root_closeout=root_ref, publication=publication_ref)
        basis_ref = sealed('basis-UNIT-DATA.json', basis)
        peer['basis'] = basis_ref
        peer_ref = sealed('peer-UNIT-DATA.json', peer)
        maintenance.update(basis=basis_ref, source_peer=peer_ref, actual18_peer=qualified_ref,
            actual18_root_closeout=root_ref, published_patch=publication_ref)
        document = {'schema': 'agent-supervisor/operator-runtime-maintenance-release@1',
            'source_commitment': plan['source_commitment'], 'sources': current_sources,
            'selected_sources': current_selected, 'operator_maintenance': maintenance}
        path = current / 'repair-source-commitment.json'
        if path.exists(): path.chmod(0o600)
        plan['release_commitment'] = write(path, document); path.chmod(0o400)
    reseal()
    d.update(old_config=old_config, post_config=post, maintenance_root=current,
        maintenance_base=base, maintenance=maintenance, maintenance_basis=basis,
        maintenance_peer=peer, maintenance_qualified=qualified,
        maintenance_closeout=root_closeout, maintenance_publication=publication,
        reseal_maintenance=reseal)
    return d


def test_held_scientific_retention_preserves_a_verified_operator_maintenance_generation(tmp_path, monkeypatch):
    d = _held_science_maintenance_fixture(tmp_path, monkeypatch); plan = d['stage']
    before = copy.deepcopy(plan); calls = _held_science_no_strict_profile(monkeypatch)
    assert plan['contract']['initial_release'] != plan['release_commitment']
    assert plan['release'] == d['old_config']['runtime_release'] == d['post_config']['runtime_release']
    old_anchor = json.loads(d['bootstrap_paths']['old'].read_bytes())
    new_anchor = json.loads(d['bootstrap_paths']['new'].read_bytes())
    assert old_anchor['expected_config_projection']['runtime_release'] == str(d['maintenance_base'])
    assert new_anchor['expected_config_projection']['runtime_release'] == str(d['maintenance_root'])
    installer.verify_repair_upgrade(plan)
    assert plan == before and calls == []
    assert d['old_config']['boards'] == d['post_config']['boards']


@pytest.mark.parametrize('fault', ['new_grant', 'new_Git_snapshot', 'base_binding', 'raw_config',
    'unit_digest', 'source_qualification', 'root_source_closed', 'publication_commit',
    'substitution_digest', 'qualified_source_digest'])
def test_held_scientific_retention_rejects_resealed_maintenance_lineage_or_authority_changes(tmp_path, monkeypatch, fault):
    d = _held_science_maintenance_fixture(tmp_path, monkeypatch)
    if fault == 'new_grant': d['maintenance']['productive_repair_enrollment_or_successor_or_new_grant'] = True
    elif fault == 'new_Git_snapshot': d['maintenance']['new84_file_Git_snapshot_claimed'] = True
    elif fault == 'base_binding': d['maintenance']['base_release_commitment'] = d['stage']['contract']['inventory']
    elif fault == 'raw_config': d['maintenance_basis']['exact_raw_config_postimage']['sha256'] = '0' * 64
    elif fault == 'unit_digest': d['maintenance_peer']['unit_postimages']['watchdog']['sha256'] = '0' * 64
    elif fault == 'source_qualification': d['maintenance_qualified']['accepted_actual_source_qualification'] = False
    elif fault == 'root_source_closed': d['maintenance_closeout']['source_READ_closed'] = False
    elif fault == 'publication_commit': d['maintenance_publication']['commit'] = '0' * 40
    elif fault == 'substitution_digest': d['maintenance_basis']['substitutions']['fleet_watchdog.py']['sha256'] = '0' * 64
    else: d['maintenance_qualified']['qualified_source_hashes']['fleet_watchdog.py'] = '0' * 64
    d['reseal_maintenance']()
    assert pin(Path(d['stage']['release_commitment']['path'])) == d['stage']['release_commitment']
    actions = []; old = Path(d['stage']['contract']['config']['path']).read_bytes()
    with pytest.raises(ValueError):
        installer.apply_repair_upgrade(d['stage'], lambda *a, **k: actions.append(a))
    assert actions == [] and Path(d['stage']['contract']['config']['path']).read_bytes() == old


def test_typed_Git_binding_refuses_different_profile_with_fully_joined_nonGit_probe(tmp_path, monkeypatch):
    """A joined legacy probe cannot authorize replacing its other profile."""
    d = git_fixture(tmp_path, monkeypatch)
    science = next(board for board in d['config']['boards'] if board['id'] == 'legal-ir-scientific')
    assert science.get('native_executor_binding') is None
    assert science.get('native_executor_git_binding') is None
    # Establish the existing non-Git profile join, then replace all three
    # argument values together with the descriptor; no owner action is invoked.
    science['native_executor_binding'] = d['profile_ref']
    science['probe']['argv'].extend(['--executor-profile', d['profile_ref']['path'],
        '--executor-profile-bytes', str(d['profile_ref']['bytes']),
        '--executor-profile-sha256', d['profile_ref']['sha256']])
    other = write(tmp_path / 'other-original-scientific-profile.json', {'retained_other_profile': True})
    science['native_executor_binding'] = other
    for flag, value in (('--executor-profile', other['path']),
            ('--executor-profile-bytes', str(other['bytes'])),
            ('--executor-profile-sha256', other['sha256'])):
        assert science['probe']['argv'].count(flag) == 1
        science['probe']['argv'][science['probe']['argv'].index(flag) + 1] = value
    before = copy.deepcopy(d['config'])
    with pytest.raises(ValueError, match='scientific Git binding cannot replace another producer profile'):
        installer._repair_scientific_configuration(d['config'], d['profile_ref'], d['release'], d['binding_ref'])
    assert d['config'] == before
    assert science['native_executor_binding'] == other
    assert science.get('native_executor_git_binding') is None


@pytest.mark.parametrize('maintenance', [False, True])
def test_held_scientific_retention_accepts_reordered_identical_descriptor_maps(tmp_path, monkeypatch, maintenance):
    """JSON object insertion order cannot invalidate an identical typed join."""
    d = (_held_science_maintenance_fixture(tmp_path, monkeypatch) if maintenance else
         _held_science_retention_fixture(tmp_path, monkeypatch))
    plan = d['stage']
    current = json.loads(Path(plan['release_commitment']['path']).read_bytes())
    for field in ('sources', 'selected_sources'):
        reordered = {role: dict(reversed(list(pin.items())))
                     for role, pin in reversed(list(current[field].items()))}
        assert reordered == current[field]
        plan['release_sources' if field == 'sources' else 'selected_release_sources'] = reordered
    plan['contract']['units'] = dict(reversed(list(plan['contract']['units'].items())))
    assert list(plan['release_sources']) != list(current['sources'])
    strict = _held_science_no_strict_profile(monkeypatch)
    before = copy.deepcopy(plan)
    installer.verify_repair_upgrade(plan)
    assert plan == before and strict == []
    assert d['old_config']['boards'] == d['post_config']['boards']


def test_held_scientific_retention_sorted_equality_keeps_boolean_authority_distinct_from_one(tmp_path, monkeypatch):
    d = _held_science_retention_fixture(tmp_path, monkeypatch)
    post = copy.deepcopy(d['post_config'])
    board = next(board for board in post['boards'] if board['id'] == 'legal-ir')
    assert type(board['repair_authorization_epoch']['max_attempts']) is int
    assert board['repair_authorization_epoch']['max_attempts'] == 1
    board['repair_authorization_epoch']['max_attempts'] = True
    _held_science_set_post(d, post)
    before = Path(d['stage']['contract']['config']['path']).read_bytes()
    with pytest.raises(ValueError): installer.verify_repair_upgrade(d['stage'])
    assert Path(d['stage']['contract']['config']['path']).read_bytes() == before
    assert d['old_config']['boards'][0]['repair_authorization_epoch']['max_attempts'] == 1


@pytest.fixture(autouse=True)
def _held_scientific_negative_shape_inventory_cleanup(request):
    """Restore only this future test's deliberate aliases after its assertions.

    The negative test body and rejection assertions remain byte-for-byte exact.
    This fixture never visits previous attempts or removes ordinary source DATA.
    """
    selected = getattr(request.node, 'originalname', None) == (
        'test_held_scientific_retention_requires_the_original_ordinary_full_stop')
    shape = getattr(getattr(request.node, 'callspec', None), 'params', {}).get('failure')
    hold = (request.getfixturevalue('tmp_path') / 'scientific/watchdog.hold'
            if selected and shape in ('symlink', 'hardlink') else None)
    yield
    if hold is None:
        return
    import stat
    if shape == 'symlink':
        info = hold.lstat()
        assert stat.S_ISLNK(info.st_mode)
        assert hold.readlink() == hold.with_name('target')
        target = hold.with_name('target').lstat()
        assert stat.S_ISREG(target.st_mode) and target.st_nlink == 1
        hold.unlink()
        assert not hold.exists() and not hold.is_symlink()
    else:
        alias = hold.with_name('alias')
        original, linked = hold.lstat(), alias.lstat()
        assert stat.S_ISREG(original.st_mode) and stat.S_ISREG(linked.st_mode)
        assert (original.st_dev, original.st_ino) == (linked.st_dev, linked.st_ino)
        assert original.st_nlink == linked.st_nlink == 2
        alias.unlink()
        assert hold.lstat().st_nlink == 1 and not alias.exists()


def _retained_rebinding_history_case(tmp_path, monkeypatch, *, depth=1):
    """Authored closed-history UNIT DATA; no actual cutover/execution is claimed."""
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = _held_science_maintenance_fixture(tmp_path, monkeypatch)
    assert 1 <= depth <= 9
    model = tmp_path / 'retained-rebinding-closed-UNIT-DATA'; model.mkdir()

    def seal(path, value):
        if path.exists(): path.chmod(0o600)
        reference = write(path, value); path.chmod(0o400)
        return reference

    def render(policy_ref, *, validation_timeout_seconds, adoption_timeout_seconds):
        # Full existing V1 renderer algebra, authored as UNIT DATA. The inert
        # local control owner is never imported or invoked by this fixture.
        policy = json.loads(Path(policy_ref['path']).read_bytes())
        refs = {row['path']: copy.deepcopy(row) for row in
                [*policy['source_descriptors'], policy_ref, policy['control_release']['manifest']]}
        arguments = ['--policy', policy_ref['path'], '--policy-sha256', policy_ref['sha256'],
            '--candidate-root', '{candidate_root}', '--freeze-path', '{freeze_path}',
            '--receipt-path', '{receipt_path}', '--source-commitment', '{source_commitment}',
            '--invocation-root', '{invocation_root}']
        prefix = [policy['python'], '-I', '-B', str(Path(policy['control_release']['root']) /
            'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py')]
        result = {}
        for role, action, timeout in (('validation_owner', 'validate', validation_timeout_seconds),
                ('adoption_owner', 'adopt', adoption_timeout_seconds)):
            selected = [*arguments, *(['--validation-path', '{validation_path}'] if action == 'adopt' else [])]
            result[role] = dict(argv=[*prefix, action, *selected],
                verify_argv=[*prefix, 'verify-validation' if action == 'validate' else 'verify-adoption', *selected],
                source_descriptors=[refs[key] for key in sorted(refs)], timeout_seconds=timeout)
        return result

    # Upgrade this new fixture alone. The old first-maintenance helper and all
    # its public tests remain byte-for-byte unchanged above this append point.
    for generation in ('old', 'new'):
        policy_path = Path(d['policies'][generation]['path'])
        policy = json.loads(policy_path.read_bytes()); root = policy_path.parent
        control_owner = root / 'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py'
        control_owner.parent.mkdir(parents=True)
        control_owner.write_text('# Inert closed-history UNIT DATA; never imported or executed.\n')
        control_owner.chmod(0o400)
        source = root / 'source.py'; trusted = root / 'test_retained.py'
        source.chmod(0o400); trusted.chmod(0o400)
        rows = sorted((pin(source), pin(trusted), pin(control_owner)), key=lambda row: row['path'])
        manifest_ref = seal(root / 'control-manifest-UNIT-DATA.json',
            dict(schema='agent-supervisor/repair-control-source-release@1', root=str(root),
                 files=rows, namespace_profiles=[]))
        policy.update(python=str(Path(sys.executable).resolve()), datasets_root=str(d['datasets']),
            control_release=dict(root=str(root), manifest=manifest_ref),
            source_descriptors=sorted((pin(source), pin(control_owner)), key=lambda row: row['path']))
        d['policies'][generation] = seal(policy_path, policy)
    d['render'] = render
    monkeypatch.setattr(installer, 'render_control_owner_commands', render)
    plan = d['stage']; before = copy.deepcopy(d['old_config']); post = copy.deepcopy(d['post_config'])
    old_path = Path(d['policies']['old']['path']); old_policy = json.loads(old_path.read_bytes())
    before['repair_worker']['candidate_runtime'].update(render(d['policies']['old'],
        validation_timeout_seconds=900, adoption_timeout_seconds=900))
    old_anchor = json.loads(d['bootstrap_paths']['old'].read_bytes())
    historical = copy.deepcopy(before); historical['runtime_release'] = str(d['maintenance_base'])
    old_anchor['expected_config_projection'] = owners._bootstrap_config_projection(
        historical, old_policy, d['policies']['old'], str(old_path))
    old_policy['adoption'] = seal(d['bootstrap_paths']['old'], old_anchor)
    d['policies']['old'] = seal(old_path, old_policy)
    before['repair_worker']['candidate_runtime'].update(render(d['policies']['old'],
        validation_timeout_seconds=900, adoption_timeout_seconds=900))
    config_ref = write(Path(plan['contract']['config']['path']), before)
    plan['contract']['config'] = config_ref; plan['files'][-1]['before'] = config_ref
    copy_path = Path(plan['held_scientific_profile_retention']['config_preimage']['path'])
    copy_path.chmod(0o600); copy_path.write_bytes(Path(config_ref['path']).read_bytes()); copy_path.chmod(0o400)
    plan['held_scientific_profile_retention']['config_preimage'] = pin(copy_path)
    raw = Path(config_ref['path']).read_bytes()
    historical_raw = raw.replace(json.dumps(str(d['maintenance_root'])).encode(),
                                 json.dumps(str(d['maintenance_base'])).encode(), 1)
    d['maintenance_basis'].update(exact_raw_config_postimage=config_ref,
        config=dict(path=config_ref['path'], bytes=len(historical_raw), sha256=hashlib.sha256(historical_raw).hexdigest()))
    d['reseal_maintenance']()
    new_path = Path(d['policies']['new']['path']); new_policy = json.loads(new_path.read_bytes())
    post['repair_worker']['candidate_runtime'].update(render(d['policies']['new'],
        validation_timeout_seconds=900, adoption_timeout_seconds=900))
    new_anchor = json.loads(d['bootstrap_paths']['new'].read_bytes())
    new_anchor.update(original_contract=copy.deepcopy(plan['contract']),
        expected_config_projection=owners._bootstrap_config_projection(
            post, new_policy, d['policies']['new'], str(new_path)))
    new_policy['adoption'] = seal(d['bootstrap_paths']['new'], new_anchor)
    d['policies']['new'] = seal(new_path, new_policy)
    post['repair_worker']['candidate_runtime'].update(render(d['policies']['new'],
        validation_timeout_seconds=900, adoption_timeout_seconds=900))
    d.update(old_config=before, post_config=post)
    _held_science_set_post(d, post)
    # Independently verify the upgraded original transition before recording
    # its modeled closure or attempting the repeated maintenance rebindings.
    installer.verify_repair_upgrade(d['stage'])
    maintenance_copy = copy.deepcopy(d['stage']['held_scientific_profile_retention']['config_preimage'])
    hops = []
    consumer = pin(Path(owners.__file__))

    def history(previous):
        return dict(schema='agent-supervisor/retained-policy-rebinding-history@1',
            maintenance_config_copy=copy.deepcopy(maintenance_copy),
            prior_cutover=copy.deepcopy(previous['actual_ref']),
            prior_readback=copy.deepcopy(previous['readback_ref']))

    def seal_hop(hop, index):
        folder = hop['folder']
        stage_ref = seal(folder / 'prior-stage-UNIT-DATA.json', hop['stage'])
        hop['selection']['full_stage'] = stage_ref
        selection_ref = seal(folder / 'prior-selection-UNIT-DATA.json', hop['selection'])
        hop['actual']['selection'] = selection_ref
        ready = {}
        for role, value in hop['loaded'].items():
            reference = seal(hop.get('loaded_paths', {}).get(role, Path(hop['stage']['state_dir']) /
                'runtime-loaded' / (value['invocation_id'] + '-' + role + '.json')), value)
            hop['actual']['observed']['loaded_runtime'][role] = reference
            ready[role] = dict(receipt=reference, current_birth_matches=True,
                owner=copy.deepcopy(value['owner']), invocation_id=value['invocation_id'],
                exact_new_config=copy.deepcopy(value['config']), exact_live_top_module=copy.deepcopy(value['loaded_module']))
        actual_ref = seal(folder / 'prior-cutover-UNIT-DATA.json', hop['actual'])
        hop['manager']['actual_runtime'] = actual_ref
        manager_ref = seal(folder / 'prior-manager-UNIT-DATA.json', hop['manager'])
        hop['readback'].update(actual_runtime=actual_ref, exact_Root_manager_observation=manager_ref,
                               actual_two_new_loaded_readiness=ready)
        readback_ref = seal(folder / 'prior-readback-UNIT-DATA.json', hop['readback'])
        hop.update(actual_ref=actual_ref, readback_ref=readback_ref, stage_ref=stage_ref,
                   selection_ref=selection_ref, manager_ref=manager_ref)

    def archive(plan, index):
        prior = copy.deepcopy(plan)
        path = Path(plan['contract']['config']['path'])
        # Materialize only ordinary local UNIT fixture files. This does not
        # invoke apply, systemd, Source, a resource owner, or a real wait.
        path.write_text(plan['files'][-1]['postimage_utf8']); path.chmod(plan['files'][-1]['mode'])
        config = pin(path)
        invocation = f'{index + 1:032x}'
        owner = dict(pid=910001 + index, birth=str(810001 + index))
        unit = 'ipfs-taskboard-legal-ir-cutover-UNIT-' + str(index + 1) + '.service'
        group = '/UNIT-DATA/' + unit
        clock = dict(owner=owner, invocation_id=invocation, started_monotonic=1000.0 + 1000 * index,
            deadline_monotonic=1120.0 + 1000 * index, existing_kernel_runtime_seconds=120)
        envelope = dict(unit=unit, invocation_id=invocation, control_group=group,
            cgroup_root='/sys/fs/cgroup' + group, memory_max_bytes=256 * 1024 ** 2,
            tasks_max=32, cpu_max=[100000, 100000], runtime_max_seconds=120,
            active_enter_monotonic_usec=int(clock['started_monotonic'] * 1000000))
        installation = dict(schema='agent-supervisor/repair-profile-upgrade-installation@1', completed=True,
            commands=[dict(argv=argv, returncode=0) for argv in (
                ['systemctl', '--user', 'daemon-reload'],
                ['systemctl', '--user', 'restart', 'ipfs-taskboard-legal-ir-watchdog.service'],
                ['systemctl', '--user', 'restart', 'ipfs-taskboard-legal-ir-repair.service'])],
            source_successor=dict(completed=True, source_WRITE_entered=False,
                source_WRITE_closed=True, postwrite_source_READ_closed=True),
            source_WRITE_closed=True, postwrite_source_READ_closed=True, config_units_CAS_closed=True)
        actual = dict(schema='held-scientific-existing-owner-code-only-cutover/v1', completed=True,
            source_READ_closed=True, installer_apply_observe_completed=True, error=None,
            native_admissions=0, scientific_commands=0, productive_epoch_budget_attempt_route_expiry_changes=0,
            V5_selector_enabled=False, automatic_preparer_enabled=False, live_top_runtime_changed=False,
            nested_consumer_generation_sha256=consumer['sha256'], original_external_owner_exit_not_yet_observed=True,
            original_external_unit=unit, cutover_clock=clock, outer_envelope=envelope,
            completed_monotonic=clock['started_monotonic'] + 5.0, installation=installation,
            observed=dict(loaded_runtime={}, runtime_unit_config_CAS_closed=True,
                read_only_readiness_wait_closed=True, readiness_deadline_monotonic=clock['deadline_monotonic']))
        loaded = {}
        for ordinal, role in enumerate(('watchdog', 'repair')):
            loaded[role] = dict(schema='agent-supervisor/repair-actual-activation@1', service=role,
                config=copy.deepcopy(config), invocation_id=f'{100 + index * 2 + ordinal:032x}',
                owner=dict(pid=920001 + index * 2 + ordinal, birth=str(820001 + index * 2 + ordinal)),
                loaded_module=copy.deepcopy(plan['release_sources']['fleet_' + role]),
                source_commitment=plan['source_commitment'], release=plan['release'])
        props = dict(MainPID='0', Result='success', ExecMainStatus='0', ControlGroup='',
                     TasksCurrent='[not set]', KillMode='control-group', SubState='exited', InvocationID=invocation)
        manager = dict(schema='Root-original-runtime-manager-kernel-terminal-observation/v1',
            external_owner=dict(unit=unit, owner=owner, invocation_id=invocation), manager_returncode=0,
            manager_argv=['systemctl', '--user', 'show', unit,
                '--property=MainPID,InvocationID,Result,ExecMainStatus,SubState,ControlGroup,TasksCurrent,KillMode'],
            manager_properties_exact=props, manager_stdout=''.join(key + '=' + value + '\n' for key, value in props.items()),
            manager_stderr='', observed_process=None, original_pid_birth_absent=True,
            original_cgroup_path_exists=False, recorded_original_cgroup_root=envelope['cgroup_root'],
            TasksCurrent_coerced_to_zero=False, canonical_strict_TasksCurrent_zero_validator_claimed_pass=False,
            Source_Native_service_or_budget_actions=0)
        postimages = []
        for row in plan['files']:
            target = Path(row['before']['path'])
            postimages.append(dict(current=pin(target), exact_selected_stage_postimage=True,
                                   full9=_held_science_file_identity(target)))
        readback = dict(schema='independent-actual-runtime-cutover-original-cgroup-retirement-readback/v1',
            verified_apply_and_readiness=True, Source_READs_closed=True, same_invocation_manager_terminal_success=True,
            original_PID_proc_entry_independently_absent=True, original_cgroup_root_independently_absent=True,
            scientific_full_stop_full9_unchanged=True, all_boards_epochs_attempts_budgets_routes_and_other_config_unchanged=True,
            prior_job_inventory_initial_release_and_all_live_top_source_refs_unchanged=True,
            external_genuine_wait_or_returncode_invented=False, TasksCurrent_coerced_to_zero=False,
            canonical_strict_TasksCurrent_zero_validator_claimed_pass=False, productive_authorization_or_V5_selector_renewed=False,
            future_admin_hook_qualified=False, reviewer_Source_Product_API_service_Native_test_or_process_actions=False,
            original_immutable_preexit_owner_exit_not_yet_observed_flag_preserved=True,
            completed_monotonic=actual['completed_monotonic'],
            elapsed_seconds=actual['completed_monotonic'] - clock['started_monotonic'],
            original_clock=copy.deepcopy(clock), exact_original_envelope=copy.deepcopy(envelope),
            exact_manager_properties=copy.deepcopy(props), exact_three_current_postimages=postimages,
            native_admissions=0, scientific_commands=0)
        folder = model / ('hop-' + str(index)); folder.mkdir()
        hop = dict(folder=folder, stage=prior, actual=actual, manager=manager, readback=readback, loaded=loaded,
            selection=dict(schema='held-scientific-code-only-cutover-exact-selection/v1', Root_selected=True,
                original_external_unit=unit,
                future_V5_selector_enabled=False, automatic_preparer_enabled=False,
                productive_repair_renewal_or_dispatch_granted=False,
                normal_canonical_sources=dict(consumer=copy.deepcopy(consumer))))
        seal_hop(hop, index)
        return hop

    for index in range(depth):
        previous = archive(d['stage'], index); hops.append(previous)
        before = copy.deepcopy(d['post_config'])
        old_pin = copy.deepcopy(d['policies']['new'])
        old_policy = json.loads(Path(old_pin['path']).read_bytes())
        old_anchor_path = Path(old_policy['adoption']['path'])
        old_anchor = json.loads(old_anchor_path.read_bytes())
        plan = copy.deepcopy(d['stage'])
        current_ref = pin(Path(plan['contract']['config']['path']))
        plan['contract']['config'] = current_ref
        plan['files'][-1]['before'] = current_ref
        before_copy = model / ('current-before-' + str(index + 1) + '-UNIT-DATA.json')
        before_copy.write_bytes(Path(current_ref['path']).read_bytes()); before_copy.chmod(0o400)
        plan['held_scientific_profile_retention']['config_preimage'] = pin(before_copy)
        plan['retained_policy_rebinding'] = history(previous)
        new_path = model / ('owner-policy-' + str(index + 1) + '-UNIT-DATA.json')
        anchor_path = model / ('bootstrap-' + str(index + 1) + '-UNIT-DATA.json')
        policy = copy.deepcopy(old_policy)
        post = copy.deepcopy(before)
        anchor = copy.deepcopy(old_anchor)
        anchor.update(owner_policy_path=str(new_path), original_contract=copy.deepcopy(plan['contract']),
            expected_unit_postimages={role: {**pin_value, 'mode': row['mode']}
                for role, pin_value in plan['contract']['units'].items()
                for row in plan['files'][:-1] if row['before'] == pin_value})
        candidate_pin = seal(new_path, policy)
        post['repair_worker']['candidate_runtime'].update(render(
            candidate_pin, validation_timeout_seconds=900, adoption_timeout_seconds=900))
        anchor['expected_config_projection'] = owners._bootstrap_config_projection(post, policy, candidate_pin, str(new_path))
        policy['adoption'] = seal(anchor_path, anchor)
        new_pin = seal(new_path, policy)
        post['repair_worker']['candidate_runtime'].update(d['render'](
            new_pin, validation_timeout_seconds=900, adoption_timeout_seconds=900))
        d.update(stage=plan, old_config=before, post_config=post,
            policies=dict(old=old_pin, new=new_pin), bootstrap_paths=dict(old=old_anchor_path, new=anchor_path))
        _held_science_set_post(d, post)

    def reseal():
        for index, hop in enumerate(hops):
            if index:
                hop['stage']['retained_policy_rebinding'] = history(hops[index - 1])
            seal_hop(hop, index)
        d['stage']['retained_policy_rebinding'] = history(hops[-1])

    d.update(rebinding_hops=hops, reseal_rebinding=reseal, seal_rebinding_artifact=seal,
             rebinding_maintenance_copy=maintenance_copy)
    return d


def _assert_retained_rebinding_refused_without_actions(d, monkeypatch, *, match=None):
    before = Path(d['stage']['contract']['config']['path']).read_bytes()
    held = _held_science_file_identity(d['hold'])
    calls = _held_science_no_strict_profile(monkeypatch)
    actions = []
    def forbidden(*args, **kwargs):
        actions.append((args, kwargs))
        raise AssertionError('REFUSED_RETAINED_HISTORY_REACHED_ORIGINAL_ACTION')
    with pytest.raises(ValueError, match=match): installer.apply_repair_upgrade(d['stage'], forbidden)
    assert actions == [] and calls == []
    assert Path(d['stage']['contract']['config']['path']).read_bytes() == before
    assert _held_science_file_identity(d['hold']) == held


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_held_scientific_retained_policy_rebinding_accepts_closed_repeated_history(tmp_path, monkeypatch, depth):
    d = _retained_rebinding_history_case(tmp_path, monkeypatch, depth=depth)
    before = copy.deepcopy(d['stage']); raw = Path(d['stage']['contract']['config']['path']).read_bytes()
    calls = _held_science_no_strict_profile(monkeypatch)
    try:
        installer.verify_repair_upgrade(d['stage'])
    except ValueError as error:
        if str(error) == 'maintenance historical genuine snapshot identity differs':
            pytest.fail('RETAINED_POLICY_REBINDING_HISTORY_UNSUPPORTED: original first-maintenance passes but repeat is refused')
        raise
    assert d['stage'] == before and calls == []
    assert Path(d['stage']['contract']['config']['path']).read_bytes() == raw
    assert d['stage']['source_successor'] is d['stage']['initial_bindings'] is None
    assert d['old_config']['boards'] == d['post_config']['boards']


def test_held_scientific_retained_policy_rebinding_accepts_exact_eight_hop_bound(tmp_path, monkeypatch):
    d = _retained_rebinding_history_case(tmp_path, monkeypatch, depth=8)
    installer.verify_repair_upgrade(d['stage'])


@pytest.mark.parametrize('damage', ['missing', 'extra', 'schema', 'raw_cutover', 'raw_readback',
    'mixed_readback', 'mutable_copy', 'changed_copy', 'cyclic_object', 'descriptor_boolean_bytes', 'descriptor_oversize'])
def test_held_scientific_retained_policy_rebinding_refuses_bad_history_transport(tmp_path, monkeypatch, damage):
    d = _retained_rebinding_history_case(tmp_path, monkeypatch, depth=2)
    profile = d['stage']['retained_policy_rebinding']
    if damage == 'missing': d['stage'].pop('retained_policy_rebinding')
    elif damage == 'extra': profile['productive_authority'] = False
    elif damage == 'schema': profile['schema'] += '-other'
    elif damage == 'raw_cutover': profile['prior_cutover'] = copy.deepcopy(d['rebinding_hops'][-1]['actual'])
    elif damage == 'raw_readback': profile['prior_readback'] = copy.deepcopy(d['rebinding_hops'][-1]['readback'])
    elif damage == 'mixed_readback': profile['prior_readback'] = copy.deepcopy(d['rebinding_hops'][0]['readback_ref'])
    elif damage == 'mutable_copy': Path(profile['maintenance_config_copy']['path']).chmod(0o600)
    elif damage == 'changed_copy':
        profile['maintenance_config_copy'] = d['seal_rebinding_artifact'](tmp_path / 'wrong-maintenance-UNIT-DATA.json', {'other': True})
    elif damage == 'cyclic_object': profile['prior_cutover'] = profile
    elif damage == 'descriptor_boolean_bytes': profile['prior_cutover']['bytes'] = True
    else: profile['prior_cutover']['bytes'] = 8 * 1024 ** 2 + 1
    _assert_retained_rebinding_refused_without_actions(d, monkeypatch)


@pytest.mark.parametrize('damage', ['completed', 'Source', 'installation', 'observed', 'Native', 'V5',
    'preexit', 'consumer', 'selection', 'source_successor', 'clock', 'envelope_boolean', 'manager_stdout',
    'manager_PID', 'manager_command', 'manager_duplicate', 'manager_positive_tasks', 'readback_closed', 'readback_authority', 'readback_preexit', 'postimage', 'postimage_mode', 'loaded_config', 'selection_unit', 'cgroup', 'active_enter',
    'readiness_deadline', 'shortened_readiness', 'shortened_kernel_runtime', 'readback_completion', 'readback_elapsed', 'loaded_path', 'loaded_owner', 'loaded_invocation'])
def test_held_scientific_retained_policy_rebinding_refuses_resealed_closed_history_changes(tmp_path, monkeypatch, damage):
    d = _retained_rebinding_history_case(tmp_path, monkeypatch, depth=2)
    hop = d['rebinding_hops'][0]
    if damage == 'completed': hop['actual']['completed'] = False
    elif damage == 'Source': hop['actual']['source_READ_closed'] = False
    elif damage == 'installation': hop['actual']['installation']['commands'][0]['returncode'] = False
    elif damage == 'observed': hop['actual']['observed']['read_only_readiness_wait_closed'] = False
    elif damage == 'Native': hop['actual']['native_admissions'] = 1
    elif damage == 'V5': hop['actual']['V5_selector_enabled'] = True
    elif damage == 'preexit': hop['actual']['original_external_owner_exit_not_yet_observed'] = False
    elif damage == 'consumer': hop['actual']['nested_consumer_generation_sha256'] = '0' * 64
    elif damage == 'selection': hop['selection']['productive_repair_renewal_or_dispatch_granted'] = True
    elif damage == 'source_successor': hop['stage']['source_successor'] = {'not_code_only': True}
    elif damage == 'clock': hop['actual']['cutover_clock']['deadline_monotonic'] += 1
    elif damage == 'envelope_boolean': hop['actual']['outer_envelope']['tasks_max'] = True
    elif damage == 'manager_stdout': hop['manager']['manager_stdout'] += 'MainPID=9\n'
    elif damage == 'manager_PID':
        hop['manager']['manager_properties_exact']['MainPID'] = '9'
        hop['manager']['manager_stdout'] = ''.join(k + '=' + v + '\n' for k, v in hop['manager']['manager_properties_exact'].items())
    elif damage == 'manager_command': hop['manager']['manager_argv'][3] = 'different-UNIT.service'
    elif damage == 'manager_duplicate': hop['manager']['manager_stdout'] += 'MainPID=0\n'
    elif damage == 'manager_positive_tasks':
        hop['manager']['manager_properties_exact']['TasksCurrent'] = '1'
        hop['readback']['exact_manager_properties']['TasksCurrent'] = '1'
        hop['manager']['manager_stdout'] = ''.join(k + '=' + v + '\n' for k, v in hop['manager']['manager_properties_exact'].items())
    elif damage == 'readback_closed': hop['readback']['Source_READs_closed'] = False
    elif damage == 'readback_authority': hop['readback']['future_admin_hook_qualified'] = True
    elif damage == 'readback_preexit': hop['readback']['original_immutable_preexit_owner_exit_not_yet_observed_flag_preserved'] = False
    elif damage == 'postimage': hop['readback']['exact_three_current_postimages'][-1]['current']['sha256'] = '0' * 64
    elif damage == 'postimage_mode': hop['readback']['exact_three_current_postimages'][-1]['full9']['st_mode'] = 0
    elif damage == 'loaded_config': hop['loaded']['watchdog']['config']['sha256'] = '0' * 64
    elif damage == 'selection_unit': hop['selection']['original_external_unit'] = 'different-UNIT.service'
    elif damage == 'cgroup': hop['actual']['outer_envelope']['cgroup_root'] += '-different'
    elif damage == 'active_enter': hop['actual']['outer_envelope']['active_enter_monotonic_usec'] = 0
    elif damage == 'readiness_deadline': hop['actual']['observed']['readiness_deadline_monotonic'] += 1
    elif damage == 'shortened_readiness': hop['actual']['observed']['readiness_deadline_monotonic'] -= 1
    elif damage == 'shortened_kernel_runtime':
        hop['actual']['outer_envelope']['runtime_max_seconds'] = 1
        hop['readback']['exact_original_envelope']['runtime_max_seconds'] = 1
    elif damage == 'readback_completion': hop['readback']['completed_monotonic'] += 1
    elif damage == 'readback_elapsed': hop['readback']['elapsed_seconds'] += 1
    elif damage == 'loaded_path': hop['loaded_paths'] = {'watchdog': tmp_path / 'wrong-loaded-UNIT-DATA.json'}
    elif damage == 'loaded_owner': hop['loaded']['watchdog']['owner']['pid'] = True
    else: hop['loaded']['repair']['invocation_id'] = hop['loaded']['watchdog']['invocation_id']
    d['reseal_rebinding']()
    _assert_retained_rebinding_refused_without_actions(d, monkeypatch)


def test_held_scientific_retained_policy_rebinding_refuses_ninth_history_hop(tmp_path, monkeypatch):
    d = _retained_rebinding_history_case(tmp_path, monkeypatch, depth=9)
    _assert_retained_rebinding_refused_without_actions(d, monkeypatch)


def test_held_scientific_retained_policy_rebinding_refuses_aggregate_eight_MiB_history(tmp_path, monkeypatch):
    d = _retained_rebinding_history_case(tmp_path, monkeypatch, depth=2)
    # Two distinct, individually sub-bound immutable metadata artifacts exceed
    # the aggregate limit. This case needs its own original16MiB fixture scope.
    for hop in d['rebinding_hops']: hop['selection']['UNIT_DATA_padding'] = 'x' * (4 * 1024 ** 2)
    d['reseal_rebinding']()
    for hop in d['rebinding_hops']:
        reference = hop['selection_ref']; path = Path(reference['path'])
        assert 0 < reference['bytes'] <= 8 * 1024 ** 2 and pin(path) == reference
        assert json.loads(path.read_bytes()) == hop['selection']
        assert not path.stat().st_mode & 0o222
    _assert_retained_rebinding_refused_without_actions(d, monkeypatch,
        match='prior policy-rebinding history exceeds aggregate byte bound')


@pytest.mark.parametrize('damage', ['source_snapshot', 'validation_budget', 'board_authority'])
def test_held_scientific_retained_policy_rebinding_keeps_new_target_authority_fences(tmp_path, monkeypatch, damage):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
    d = _retained_rebinding_history_case(tmp_path, monkeypatch, depth=2)
    path = Path(d['policies']['new']['path']); policy = json.loads(path.read_bytes())
    anchor_path = d['bootstrap_paths']['new']; anchor = json.loads(anchor_path.read_bytes())
    post = copy.deepcopy(d['post_config'])
    if damage == 'source_snapshot': anchor['source_snapshot']['new_tree'] = '0' * 40
    elif damage == 'validation_budget': policy['validation']['timeout_seconds'] += 1
    else: next(board for board in post['boards'] if board['id'] == 'legal-ir-scientific')['max_repair_attempts'] += 1
    candidate_pin = d['seal_rebinding_artifact'](path, policy)
    post['repair_worker']['candidate_runtime'].update(d['render'](
        candidate_pin, validation_timeout_seconds=900, adoption_timeout_seconds=900))
    anchor['expected_config_projection'] = owners._bootstrap_config_projection(post, policy, candidate_pin, str(path))
    policy['adoption'] = d['seal_rebinding_artifact'](anchor_path, anchor)
    d['policies']['new'] = d['seal_rebinding_artifact'](path, policy)
    post['repair_worker']['candidate_runtime'].update(d['render'](
        d['policies']['new'], validation_timeout_seconds=900, adoption_timeout_seconds=900))
    _held_science_set_post(d, post)
    _assert_retained_rebinding_refused_without_actions(d, monkeypatch)


@pytest.mark.parametrize('damage', ['executable', 'extra_argument', 'missing_source_map'])
def test_held_scientific_retained_policy_rebinding_refuses_resealed_nonV1_history_commands(tmp_path, monkeypatch, damage):
    d = _retained_rebinding_history_case(tmp_path, monkeypatch, depth=1)
    before = copy.deepcopy(d['old_config'])
    row = before['repair_worker']['candidate_runtime']['validation_owner']
    if damage == 'executable': row['argv'][0] = '/UNIT-DATA/arbitrary-executable'
    elif damage == 'extra_argument': row['argv'].append('--UNIT-DATA-unapproved-option')
    else: row['source_descriptors'] = []
    # Reseal the real config preimage and its exact predecessor postimage,
    # rather than stopping at a hash mismatch before the command guard.
    target = Path(d['stage']['contract']['config']['path'])
    current = write(target, before)
    d['stage']['contract']['config'] = current
    d['stage']['files'][-1]['before'] = current
    copy_path = Path(d['stage']['held_scientific_profile_retention']['config_preimage']['path'])
    copy_path.chmod(0o600); copy_path.write_bytes(target.read_bytes()); copy_path.chmod(0o400)
    d['stage']['held_scientific_profile_retention']['config_preimage'] = pin(copy_path)
    d['rebinding_hops'][-1]['stage']['files'][-1]['postimage_utf8'] = target.read_text()
    d['reseal_rebinding']()
    _assert_retained_rebinding_refused_without_actions(d, monkeypatch,
        match='historical v1 executable/arguments/source-role map differs')
