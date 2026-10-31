"""Prospective pure metadata/rendering regressions; no original owner credit.

The private files and hashes are real. Modeled Git/authority/receipt values test
only the parser/CAS boundaries named below; they grant no native, Git, process,
source lease, recovery, activation, validation, task, or model acceptance.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import _pytest.python as pytest_python
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate as candidate
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_runtime as runtime
from ipfs_accelerate_py.agent_supervisor.rescue import repair_authorization as authorization
from scripts.ops.agent_supervisor import install_fleet_watchdog as installer


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if type(value) is bytes else owners.canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o400)
    return pin(path)


def candidate_policy(repository, state, sources, allowed):
    return {"schema": candidate.POLICY_SCHEMA, "repository_root": str(repository), "state_root": str(state),
        "namespace": "ptr_lane_0", "base_commit": "1" * 40, "source_paths": sources, "allowed_paths": allowed,
        "max_source_bytes": 1048576, "max_candidate_bytes": 2097152, "max_files": 32, "git_timeout_seconds": 10}


@pytest.fixture
def metadata_scope(tmp_path):
    """Six real control bodies; a metadata-only, deliberately unexecuted policy."""
    control = tmp_path / "control"; control.mkdir()
    origin = Path(owners.__file__).absolute().parents[3]
    relatives = ("ipfs_accelerate_py/__init__.py", "ipfs_accelerate_py/agent_supervisor/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/__init__.py", "ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py", "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py")
    files = [put(control / relative, (origin / relative).read_bytes()) for relative in relatives]
    owner = control / relatives[-2]; worker = control / relatives[-1]
    test = put(control / "test/api/test_private_metadata.py", b"def test_original_case():\n    assert True\n")
    files.append(test)
    manifest = put(control / "repair-control-manifest.json", {"schema": owners.CONTROL_SCHEMA,
        "root": str(control), "files": sorted(files, key=lambda row: row["path"]), "namespace_profiles": []})
    d = tmp_path / "D"; d.mkdir()
    required = [d / "scripts/ops/legal_ir/run_owned_validation.py", *[d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer" / name
        for name in ("autoencoder_source_lease.py", "autoencoder_resource_recovery.py", "validation_run_resources.py")]]
    sources = [pin(owner), pin(worker)]
    for path in required:
        raw = (b"def inspect_validation_reservation():\n    pass\ndef inspect_validation_disposition():\n    pass\n"
            if path.name == "autoencoder_resource_recovery.py" else b"# private metadata parser fixture only\n")
        sources.append(put(path, raw))
    state = tmp_path / "state"; state.mkdir()
    source = tmp_path / "active"; source.mkdir()
    selected_source = put(source / "source.py", b"value = True\n")
    selected = candidate_policy(source, state, ["source.py"], ["source.py"])
    # This is the REAL maintained parser and real fixed candidate namespace.
    parsed = candidate.parse_candidate_policy({"candidate_policy": selected})
    assert parsed is not None and parsed.namespace == "ptr_lane_0"
    policy = {"schema": owners.POLICY_SCHEMA, "control_release": {"root": str(control), "manifest": manifest},
        "datasets_root": str(d), "python": sys.executable, "source_descriptors": sources,
        "adoption": put(tmp_path / "adoption.json", {"private_metadata_only": True}),
        "validation": {"trusted_tests": [test], "selected_checks": [test["path"] + "::test_original_case"],
            "pytest_module": pin(Path(pytest.__file__).absolute()), "import_sources": [pin(Path(pytest_python.__file__).absolute())],
            "attempt_root": str(tmp_path / "attempts"), "cpu_slots": 1, "memory_mb": 1536, "storage_bytes": 67108864,
            "timeout_seconds": 300, "child_process_slots": 2, "scheduler_state_path": str(tmp_path / "native.json"),
            "expected_scheduler_policy_sha256": "a" * 64, "roots": [str(tmp_path)], "resource_ledger": str(tmp_path / "ledger.json")}}
    policy_pin = put(tmp_path / "policy.json", policy)
    return SimpleNamespace(control=control, policy=policy, policy_pin=policy_pin, source=source, state=state,
        selected=selected, selected_source=selected_source)


def test_metadata_renderer_uses_fixed_control_owner_while_cold_default_requires_actual_origin(metadata_scope):
    scope = metadata_scope
    checked = owners.checked_policy(Path(scope.policy_pin["path"]), scope.policy_pin["sha256"], current_owner=False)
    assert checked == scope.policy
    with pytest.raises(ValueError, match="actual owner is outside"):
        owners.checked_policy(Path(scope.policy_pin["path"]), scope.policy_pin["sha256"])
    commands = installer.render_control_owner_commands(scope.policy_pin, validation_timeout_seconds=360, adoption_timeout_seconds=300)
    for command in commands.values():
        assert command["argv"][1:3] == ["-I", "-B"]
        assert Path(command["argv"][3]) == scope.control / "ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py"
        assert scope.policy_pin in command["source_descriptors"]
        assert scope.policy["control_release"]["manifest"] in command["source_descriptors"]


@pytest.mark.parametrize("mutation", ["argv", "source_pin"])
def test_initial_binding_rejects_changed_nested_control_before_any_source_lease(metadata_scope, monkeypatch, mutation):
    scope = metadata_scope
    commands = installer.render_control_owner_commands(scope.policy_pin, validation_timeout_seconds=360, adoption_timeout_seconds=300)
    if mutation == "argv":
        commands["adoption_owner"]["argv"][3] = str(scope.source / "owner.py")
    else:
        commands["adoption_owner"]["source_descriptors"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(owners, "public_source_lease", lambda policy: pytest.fail("source lease reached despite invalid nested controls"))
    bindings = {"candidate_policy": scope.selected, "candidate_runtime": commands,
        "authorization_epoch": {"board_namespace": "ptr_lane_0"}, "source_snapshot": None}
    with pytest.raises(ValueError, match="immutable control renderer"):
        installer.prepare_existing_repair_configuration(scope.source, {}, bindings)


def test_initial_binding_refuses_mutable_source_overlapping_control_root(metadata_scope, monkeypatch):
    scope = metadata_scope
    commands = installer.render_control_owner_commands(scope.policy_pin, validation_timeout_seconds=360, adoption_timeout_seconds=300)
    selected = {**scope.selected, "repository_root": str(scope.control)}
    monkeypatch.setattr(owners, "public_source_lease", lambda policy: pytest.fail("source lease reached despite control collision"))
    with pytest.raises(ValueError, match="must be separate"):
        installer.prepare_existing_repair_configuration(scope.control, {}, {"candidate_policy": selected,
            "candidate_runtime": commands, "authorization_epoch": {"board_namespace": "ptr_lane_0"}, "source_snapshot": None})


@pytest.mark.parametrize("scientific_first", [True, False])
def test_actual_initial_renderer_preserves_both_boards_and_fixed_nested_controls(metadata_scope, tmp_path, monkeypatch, scientific_first):
    """Full initial CAS renderer; explicit modeled authority/Git, no commands."""
    scope = metadata_scope
    commands = installer.render_control_owner_commands(scope.policy_pin, validation_timeout_seconds=360, adoption_timeout_seconds=300)
    # Keep actual policy parsing; this narrow fixture models only unavailable
    # external authority and Git observations. No genuine grant is asserted.
    @contextmanager
    def read_source(*args, **kwargs):
        assert kwargs == {"mode": "read", "timeout_seconds": 10}
        yield None
    monkeypatch.setattr(owners, "public_source_lease", lambda policy: read_source)
    def git_command(root, argv, **kwargs):
        assert root == scope.source
        if argv == ["rev-parse", "--verify", "HEAD"]:
            return scope.selected["base_commit"]
        if argv == ["rev-parse", scope.selected["base_commit"] + "^{tree}"]:
            return "2" * 40
        assert argv == ["show", scope.selected["base_commit"] + ":source.py"]
        return (scope.source / "source.py").read_text()
    monkeypatch.setattr(owners, "git_command", git_command)
    monkeypatch.setattr(authorization, "require_current_authorization", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "parse_runtime_policy", lambda worker, parsed: SimpleNamespace())
    home = tmp_path / "home"; state = scope.state
    legal = {"id": "legal-ir", "keep": {"values": [False, "original"]}}
    scientific = {"id": "legal-ir-scientific", "activation": {"foreign_gate": [1, None, True]}, "max_workers": 3}
    boards = [scientific, legal] if scientific_first else [legal, scientific]
    old = {"state_dir": str(state), "boards": boards, "repair_worker": {"job_unit": "ipfs-taskboard-legal-ir-job.service"}}
    config = put(home / ".config/ipfs-taskboard-watchdog/legal-ir/fleet.json", old)
    units = {name: put(home / ("ipfs-taskboard-legal-ir-" + name + ".service"), b"# private original unit fixture\n") for name in ("watchdog", "repair")}
    release = tmp_path / "private-release"; release.mkdir()
    contents = {name: b"# private modeled release, never executed\n" for name in ("fleet_watchdog", "fleet_repair", "repair_candidate_owners")}
    for name, raw in contents.items():
        put(release / "ipfs_accelerate_py/agent_supervisor/rescue" / (name + ".py"), raw)
    relative = "ipfs_accelerate_py/agent_supervisor/rescue/fleet_watchdog.py"
    monkeypatch.setattr(installer, "_finite_repair_release", lambda *args: (release, contents, {relative: contents["fleet_watchdog"]}))
    manifest = {"source.py": {"bytes": scope.selected_source["bytes"], "sha256": scope.selected_source["sha256"], "executable": False}}
    commitment = owners.digest({"base_commit": scope.selected["base_commit"], "manifest": manifest})
    sources = {name: pin(release / "ipfs_accelerate_py/agent_supervisor/rescue" / (name + ".py")) for name in contents}
    initial = put(release / "repair-source-commitment.json", installer.encoded_json({"schema": "agent-supervisor/repair-source-release@1",
        "source_commitment": commitment, "sources": sources, "snapshot_commit": scope.selected["base_commit"], "snapshot_tree": "2" * 40,
        "selected_sources": {relative: pin(release / relative)}}))
    contract = {"schema": "agent-supervisor/repair-profile-upgrade-contract@1", "profile": "legal-ir", "config": config,
        "inventory": put(tmp_path / "inventory.json", {}), "units": units, "prior_job": put(state / "repairs/legal-ir/job.json", {}),
        "expected_loaded_before": {}, "job_unit": old["repair_worker"]["job_unit"], "source_successor": None, "initial_release": initial}
    epoch = {"board_namespace": "ptr_lane_0", "private_modeled_authority": True}
    plan = installer.prepare_existing_repair_configuration(scope.source, contract,
        {"candidate_policy": scope.selected, "candidate_runtime": commands, "authorization_epoch": epoch, "source_snapshot": None})
    post = json.loads(next(row["postimage_utf8"] for row in plan["files"] if row["before"] == config))
    assert len(post["boards"]) == 2 and [row["id"] for row in post["boards"]] == [row["id"] for row in old["boards"]]
    assert next(row for row in post["boards"] if row["id"] == "legal-ir-scientific") == scientific
    assert next(row for row in post["boards"] if row["id"] == "legal-ir") == {**legal, "repair_authorization_epoch": epoch}
    assert post["repair_worker"]["candidate_runtime"] == commands
    assert plan["service_actions"] == 0 and plan["staged_only"] is True
    assert pin(Path(config["path"])) == config


@pytest.mark.parametrize("point", ["opening", "closing"])
def test_control_inspection_rehashes_same_policy_before_success(tmp_path, monkeypatch, point):
    policy = {"private_inspection_binding_fixture": True}; policy_pin = put(tmp_path / "policy.json", policy)
    args = SimpleNamespace(policy=Path(policy_pin["path"]), policy_sha256=policy_pin["sha256"])
    if point == "opening":
        args.policy.chmod(0o600); args.policy.write_bytes(b"{}\n"); args.policy.chmod(0o400)
        monkeypatch.setattr(owners, "checked_control_release", lambda *args: pytest.fail("changed policy reached control parser"))
    else:
        root = tmp_path / "control"; root.mkdir(); source = put(root / "authorization.py", b"# private modeled inspection source\n")
        control = {"root": root, "namespace_profiles": [], "file_count": 1, "source_bytes": source["bytes"]}
        monkeypatch.setattr(owners, "checked_control_release", lambda *args: control)
        monkeypatch.setattr(owners, "_CONTROL_GUARD", SimpleNamespace(executed={source["path"]: source["sha256"]}, denials=[], verify_normal_namespaces=lambda: {}))
        def changed_import(name):
            args.policy.chmod(0o600); args.policy.write_bytes(b"{}\n"); args.policy.chmod(0o400)
            return SimpleNamespace(__file__=source["path"])
        monkeypatch.setattr(owners.importlib, "import_module", changed_import)
    with pytest.raises(ValueError, match="policy changed"):
        owners.inspect_control(args)


def worker_profile(selector, trusted):
    # Literal child/tool data are deliberately unexecuted protocol inputs.
    return {"schema": "agent-supervisor/bounded-external-fixtures@1", "alias_slots": [],
        "source_slots": [{"selector": selector, "owner_source": trusted, "function": "test_original_case", "line": 2,
            "path_suffix": "private/helper.py", "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}],
        "children": {"schema": "agent-supervisor/bounded-validation-children@1", "python_executable": sys.executable,
            "installed_tool_inputs": [], "installer_tool": {}, "installer_child_contract": {"private_stub_node": selector, "systemd_allowed_nodes": [selector]},
            "frozen_child_source": "pass\n", "landlock_launcher_source": "pass\n", "landlock_test_source": "pass\n",
            "git_node_selectors": [selector], "frozen_owner_node_selectors": [], "cold_control_slots": [], "python_file_slots": [], "inline_slots": []}}


@pytest.fixture
def plan_scope(tmp_path):
    control = tmp_path / "control"; control.mkdir()
    candidate_root = tmp_path / "candidate"; candidate_root.mkdir()
    repository = tmp_path / "active"; repository.mkdir()
    state = tmp_path / "state"; state.mkdir()
    relative = "test/api/test_original.py"
    raw = b"def test_original_case():\n    assert True\n"
    trusted = put(control / relative, raw); actual = put(candidate_root / relative, raw)
    edited = put(candidate_root / "source.py", b"value = 2\n")
    selected = candidate_policy(repository, state, [relative, "source.py"], ["source.py"])
    parsed = candidate.parse_candidate_policy({"candidate_policy": selected})
    binding = {"policy_sha256": owners.digest(parsed.to_dict()), "repository_root": str(repository), "namespace": parsed.namespace,
        "board_id": "legal-ir", "attempt_id": "a" * 32, "attempt_number": 1, "base_commit": parsed.base_commit,
        "candidate_root": str(candidate_root), "metadata_root": str(tmp_path / "metadata"), "active_root": str(repository)}
    baseline = {relative: {"bytes": trusted["bytes"], "sha256": trusted["sha256"], "executable": False},
        "source.py": {"bytes": 10, "sha256": "b" * 64, "executable": False}}
    frozen_manifest = {**baseline, "source.py": {"bytes": edited["bytes"], "sha256": edited["sha256"], "executable": False}}
    frozen = {"binding": binding, "baseline_sha256": owners.digest(baseline), "manifest": frozen_manifest,
        "changed_paths": ["source.py"], "source_commitment": "c" * 64}
    invocation = tmp_path / "metadata/original-validate-invocation"; invocation.mkdir(parents=True)
    prepared = put(invocation.parent / "prepared.json", {"binding": binding, "policy": parsed.to_dict(), "baseline": baseline})
    freeze_pin = put(invocation.parent / "candidate-ready.json", frozen)
    d = tmp_path / "D"; d.mkdir()
    package = d / "ipfs_datasets_py"; directory = package / "optimizers/logic_theorem_optimizer"
    required = [d / "scripts/ops/legal_ir/run_owned_validation.py", *(directory / name for name in
        ("validation_run_resources.py", "autoencoder_daemon_resources.py", "autoencoder_source_lease.py", "autoencoder_resource_recovery.py",
         "resource_scheduler.py", "existing_resource_scheduler.py", "proof_resource_safety.py", "runtime_telemetry.py", "__init__.py")),
        package / "optimizers/__init__.py", package / "__init__.py", package / "router_deps.py", package / "ipfs_backend_router.py", package / "logic/__init__.py", package / "logic/backends/process.py"]
    d_sources = [put(path, b"# private unexecuted protocol source fixture\n") for path in required]
    worker = put(control / "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py", b"# private unexecuted worker fixture\n")
    selector = trusted["path"]
    natural = selector + "::test_original_case"
    namespace = {"fullname": "ipfs_datasets_py.logic.backends", "path": str(package / "logic/backends"),
        "device": (package / "logic/backends").stat().st_dev, "inode": (package / "logic/backends").stat().st_ino, "initializer_absent": True}
    policy = {"schema": owners.POLICY_SCHEMA, "control_release": {"root": str(control)}, "datasets_root": str(d), "python": sys.executable,
        "source_descriptors": [worker, *d_sources], "validation": {"trusted_tests": [trusted], "selected_checks": [selector],
            "pytest_module": pin(Path(pytest.__file__).absolute()), "import_sources": [], "attempt_root": str(tmp_path / "attempts"),
            "cpu_slots": 1, "memory_mb": 1536, "storage_bytes": 67108864, "timeout_seconds": 300, "child_process_slots": 4,
            "scheduler_state_path": str(tmp_path / "native.json"), "expected_scheduler_policy_sha256": "d" * 64,
            "roots": [str(tmp_path)], "resource_ledger": str(tmp_path / "ledger.json"),
            "external_fixture_profile": worker_profile(selector, trusted), "normal_namespace_profiles": [namespace],
            "expected_nodeids": [natural], "selector_case_counts": {selector: 1}}}
    # Per-case fixture grants remain narrower than the file selector.
    policy["validation"]["external_fixture_profile"]["source_slots"][0]["selector"] = natural
    policy["validation"]["external_fixture_profile"]["children"]["git_node_selectors"] = [natural]
    policy["validation"]["external_fixture_profile"]["children"]["installer_child_contract"] = {
        "private_stub_node": natural, "systemd_allowed_nodes": [natural]}
    policy_pin = put(tmp_path / "policy.json", policy)
    return SimpleNamespace(policy=policy, policy_pin=policy_pin, frozen=frozen, freeze_pin=freeze_pin,
        root=candidate_root, invocation=invocation, relative=relative, trusted=trusted, actual=actual,
        prepared=prepared, baseline=baseline)


def test_v3_plan_binds_real_candidate_lane_preexisting_test_copy_and_distinct_registry(plan_scope):
    s = plan_scope
    plan = owners.validation_plan(s.policy, s.policy_pin, s.frozen, s.freeze_pin, s.root, s.invocation)
    selector = s.actual["path"]
    assert plan["selected_checks"] == [selector]
    assert plan["expected_nodeids"] == [selector + "::test_original_case"]
    assert plan["selector_case_counts"] == {selector: 1}
    assert plan["trusted_test_copies"] == [{"trusted": s.trusted, "actual": s.actual}]
    assert plan["trusted_test_preparation"] == s.prepared
    assert plan["external_fixture_profile"]["source_slots"][0]["owner_source"] == s.actual
    children = plan["external_fixture_profile"]["children"]
    assert children["git_node_selectors"] == [selector + "::test_original_case"]
    assert children["installer_child_contract"]["private_stub_node"] == selector + "::test_original_case"
    assert plan["attempt_directory"].endswith("/ptr_lane_0/legal-ir/" + "a" * 32)
    assert plan["public_request_path"].endswith("/control-requests/legal_ir_unattended_repair/legal-ir/" + "a" * 32 + "/execution-request.json")
    assert s.frozen["binding"]["namespace"] == "ptr_lane_0" and "request_sha256" not in plan
    plan_pin = put(s.invocation / "validation-plan.json", plan)
    request = owners.validation_execution_request(s.policy, plan, plan_pin)
    assert request["schema"] == "owned-validation-request/v3"
    assert request["external_fixture_profile"]["max_bytes"] == 33554432
    assert request["declared_external_writable_paths"] == [plan["external_fixture_directory"]]
    assert request["source_sha256"][plan_pin["path"]] == plan_pin["sha256"]
    assert request["selected_checks"] == [selector] and request["argv"][-1] == plan_pin["sha256"]
    assert pin(Path(s.actual["path"])) == s.actual


@pytest.mark.parametrize("problem", ["allowed_write", "not_in_baseline", "raw_changed", "wrong_preparation", "wrong_lane"])
def test_v3_plan_refuses_untrusted_test_copy_or_candidate_binding(plan_scope, problem):
    s = plan_scope
    frozen = copy.deepcopy(s.frozen)
    if problem == "allowed_write":
        prepared = json.loads(Path(s.prepared["path"]).read_bytes())
        prepared["policy"]["allowed_paths"].append(s.relative)
        prepared["binding"]["policy_sha256"] = owners.digest(prepared["policy"])
        frozen["binding"] = prepared["binding"]
        path = Path(s.prepared["path"]); path.chmod(0o600); path.write_bytes(owners.canonical(prepared) + b"\n"); path.chmod(0o400)
    elif problem == "not_in_baseline":
        frozen["manifest"].pop(s.relative)
    elif problem == "raw_changed":
        path = Path(s.actual["path"]); path.chmod(0o600); path.write_bytes(b"def test_other():\n    pass\n"); path.chmod(0o400)
    elif problem == "wrong_preparation":
        frozen["baseline_sha256"] = "0" * 64
    else:
        prepared = json.loads(Path(s.prepared["path"]).read_bytes())
        prepared["policy"]["namespace"] = "ptr_lane_1"
        prepared["binding"]["namespace"] = "ptr_lane_1"
        prepared["binding"]["policy_sha256"] = owners.digest(prepared["policy"])
        frozen["binding"] = prepared["binding"]
        path = Path(s.prepared["path"]); path.chmod(0o600); path.write_bytes(owners.canonical(prepared) + b"\n"); path.chmod(0o400)
    with pytest.raises((ValueError, KeyError)):
        owners.validation_plan(s.policy, s.policy_pin, frozen, s.freeze_pin, s.root, s.invocation)


def test_legacy_plan_keeps_original_outside_nodes_and_v1_request(plan_scope):
    s = plan_scope; policy = copy.deepcopy(s.policy)
    for key in ("external_fixture_profile", "normal_namespace_profiles", "expected_nodeids", "selector_case_counts"):
        policy["validation"].pop(key)
    policy["validation"]["selected_checks"] = [s.trusted["path"] + "::test_original_case"]
    policy["validation"]["child_process_slots"] = 2
    frozen = copy.deepcopy(s.frozen); frozen["binding"]["namespace"] = "legal_ir_unattended_repair"
    plan = owners.validation_plan(policy, s.policy_pin, frozen, s.freeze_pin, s.root, s.invocation)
    assert "trusted_test_copies" not in plan and "expected_nodeids" not in plan
    assert plan["selected_checks"] == policy["validation"]["selected_checks"]
    request = owners.validation_execution_request(policy, plan, put(s.invocation / "validation-plan.json", plan))
    assert request["schema"] == "owned-validation-request/v1" and request["declared_external_writable_paths"] == []
    assert "external_fixture_profile" not in request and request["child_process_slots"] == 2


@pytest.mark.parametrize("problem", ["duplicate", "count_bool", "ambiguous", "partial"])
def test_natural_case_expansion_requires_exact_unique_selector_counts(problem):
    selector = "/private/test_original.py"; node = selector + "::test_case"
    plan = {"selected_checks": [selector], "expected_nodeids": [node], "selector_case_counts": {selector: 1}, "expected_cases": 1}
    if problem == "duplicate":
        plan["expected_nodeids"] *= 2; plan["expected_cases"] = 2; plan["selector_case_counts"][selector] = 2
    elif problem == "count_bool":
        plan["selector_case_counts"][selector] = True
    elif problem == "ambiguous":
        plan["selected_checks"].append(node); plan["selector_case_counts"][node] = 1
        plan["expected_nodeids"].append(selector + "::test_other"); plan["expected_cases"] = 2
    else:
        plan.pop("expected_nodeids")
    with pytest.raises(ValueError):
        owners.expected_natural_nodes(plan)


@pytest.mark.parametrize("key,bad", [("cpu_slots", 2), ("memory_mb", 2048), ("storage_bytes", 134217728), ("timeout_seconds", 301), ("child_process_slots", 5)])
def test_v3_profile_cannot_expand_the_reviewed_resource_envelope(plan_scope, key, bad):
    policy = copy.deepcopy(plan_scope.policy); policy["validation"][key] = bad
    with pytest.raises(ValueError):
        owners.checked_validation_profile(policy, {"root": Path(policy["control_release"]["root"]), "namespace_profiles": policy["validation"]["normal_namespace_profiles"]})


@pytest.fixture
def external_scope(plan_scope):
    """Modeled original receipts, never a claimed real grant or inventory."""
    s = plan_scope
    plan = owners.validation_plan(s.policy, s.policy_pin, s.frozen, s.freeze_pin, s.root, s.invocation)
    plan_pin = put(s.invocation / "validation-plan.json", plan)
    request = owners.validation_execution_request(s.policy, plan, plan_pin)
    request_pin = put(Path(plan["public_request_path"]), request)
    local = put(s.invocation / "execution-request.json", request)
    attempt = Path(plan["attempt_directory"]); attempt.mkdir(parents=True)
    external = Path(plan["external_fixture_directory"]); external.mkdir()
    info = external.stat(); identity = {"path": str(external), "device": info.st_dev, "inode": info.st_ino}
    rid = "b" * 32; owner = {"pid": 123, "birth": "456"}; key = "external-fixtures-" + request_pin["sha256"][:32]
    flags = {"admitted": False, "Lean_admitted": False, "test_or_task_acceptance_asserted": False}
    path_receipt = put(attempt / "logs/external-fixture-path.json", {"schema": "owned-validation-external-original-path/v2",
        "request_sha256": request_pin["sha256"], "reservation_id": rid, "external_fixture_directory": str(external),
        "root_identity": identity, "outside_attempt": True, "owner": owner,
        "owner_started_monotonic": 100.0, "owner_deadline_monotonic": 400.0, **flags})
    charge_receipt = put(attempt / "logs/external-fixture-charge.json", {"schema": "owned-validation-external-original-charge/v1",
        "request_sha256": request_pin["sha256"], "reservation_id": rid, "path_receipt": path_receipt,
        "accounting_result": {"key": key, "bytes": 33554432, "already_charged": False, "usage": {}}, **flags})
    manifest = {"schema": "qualified-external-synthetic-filesystem-disposition/v1", "root_identity": identity,
        "root_metadata": {}, "entries": [], "typed_counts": {"regular": 0, "symlink": 0, "fifo": 0, "directory": 0},
        "entry_count": 0, "apparent_bytes": 0, "targets_followed": False, "special_payloads_opened": False,
        "regular_payloads_opened": False, "all_regular_hardlinks_internal": True, "ordinary_external_artifact_durability_asserted": False}
    # Match the original D producer wire, including ASCII escaping/newline.
    manifest_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    inventory_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest_pin = put(attempt / ("logs/external-synthetic-disposition-" + inventory_sha + ".json"), manifest_raw + b"\n")
    disposition = {"schema": "owned-validation-external-disposition-observation/v1", "request_sha256": request_pin["sha256"],
        "reservation_id": rid, "root_identity": identity, "manifest": manifest_pin, "inventory_sha256": inventory_sha,
        "apparent_bytes": 0, "entry_count": 0, "typed_counts": manifest["typed_counts"], "stable_repeated_inventory": True,
        "targets_followed": False, "special_payloads_opened": False, "regular_payloads_opened": False,
        "all_regular_hardlinks_internal": True, "ordinary_external_artifact_durability_asserted": False, **flags}
    bound = {"schema": "owned-validation-external-fixtures/v1", "external_fixture_directory": str(external),
        "request_sha256": request_pin["sha256"], "charge_key": key, "charged_bytes": 33554432, "root_identity": identity}
    record = {"status": "retained", "retention_reason": "context_failed", "reservation_id": rid,
        "owner_pid": owner["pid"], "owner_birth": owner["birth"], "storage_bytes": 67108864, "validation_timeout_seconds": 300,
        "native_lease_id": "c" * 32, "external_charges": {key: 33554432}, "external_fixture_profile": bound,
        "external_fixture_original_receipts": {"path": path_receipt, "charge": charge_receipt}, "external_fixture_disposition": disposition}
    return SimpleNamespace(base=s, plan=plan, plan_pin=plan_pin, request=request, request_pin=request_pin,
        local=local, record=record, owner=owner, key=key)


def test_v3_original_profile_reads_only_bound_receipts_preserving_charge_and_false_durability(external_scope):
    s = external_scope
    result = owners.original_external_validation_profile(s.base.policy, s.plan, s.request_pin, s.record)
    assert result == {"request_schema": "owned-validation-request/v3", "external_fixture_profile": s.record["external_fixture_profile"],
        "original_receipts": s.record["external_fixture_original_receipts"], "terminal_disposition": s.record["external_fixture_disposition"]}
    assert s.record["external_charges"] == {s.key: 33554432}
    assert result["terminal_disposition"]["ordinary_external_artifact_durability_asserted"] is False


@pytest.mark.parametrize("problem", ["charge_zero", "request_hash", "root", "receipt_owner", "durability", "manifest_changed", "request_location", "clock_bool", "deadline_reset"])
def test_v3_original_profile_refuses_changed_charge_path_receipts_or_typed_terminal(external_scope, problem):
    s = external_scope; record = copy.deepcopy(s.record); request_pin = copy.deepcopy(s.request_pin)
    if problem == "charge_zero":
        record["external_charges"][s.key] = 0
    elif problem == "request_hash":
        record["external_fixture_profile"]["request_sha256"] = "0" * 64
    elif problem == "root":
        record["external_fixture_profile"]["root_identity"]["inode"] += 1
    elif problem == "receipt_owner":
        receipt = record["external_fixture_original_receipts"]["path"]
        value = json.loads(Path(receipt["path"]).read_bytes()); value["owner"]["birth"] = "999"
        path = Path(receipt["path"]); path.chmod(0o600); path.write_bytes(owners.canonical(value) + b"\n"); path.chmod(0o400)
        record["external_fixture_original_receipts"]["path"] = pin(path)
    elif problem in {"clock_bool", "deadline_reset"}:
        receipt = record["external_fixture_original_receipts"]["path"]
        value = json.loads(Path(receipt["path"]).read_bytes())
        if problem == "clock_bool":
            value["owner_started_monotonic"] = True
        else:
            value["owner_deadline_monotonic"] += 1
        path = Path(receipt["path"]); path.chmod(0o600); path.write_bytes(owners.canonical(value) + b"\n"); path.chmod(0o400)
        record["external_fixture_original_receipts"]["path"] = pin(path)
    elif problem == "durability":
        record["external_fixture_disposition"]["ordinary_external_artifact_durability_asserted"] = True
    elif problem == "manifest_changed":
        path = Path(record["external_fixture_disposition"]["manifest"]["path"])
        path.chmod(0o600); path.write_bytes(b"{}\n"); path.chmod(0o400)
    else:
        copied = put(s.base.invocation / "different-request-location.json", s.request)
        request_pin = copied
    with pytest.raises(ValueError):
        owners.original_external_validation_profile(s.base.policy, s.plan, request_pin, record)
    assert s.record["external_charges"] == {s.key: 33554432}


def test_v3_closed_marker_preserves_actual_candidate_binding_without_semantic_acceptance(external_scope):
    """Modeled wait metadata tests joining/publication only, never actual wait."""
    s = external_scope
    terminal = {"reservation": {"record": s.record}, "failure": {"external_fixture_disposition": s.record["external_fixture_disposition"]}}
    terminal_pin = put(s.base.invocation / "original-failure-terminal.json", terminal)
    wait = {"actual_Popen_returned": True, "actual_process_waited": True, "runner_identity": s.owner,
        "request": s.request_pin, "stdout": json.dumps(terminal)}
    wait_pin = put(s.base.invocation / "runner-wait.json", wait)
    proof = {"schema": owners.NEGATIVE_SCHEMA, "semantic_negative": False, "negative_profile": None,
        "source_closed": True, "native_closed": True, "processes_closed": True, "storage_closed": False,
        "original_failure_terminal": terminal_pin, "original_runner_wait": wait_pin,
        "original_record_sha256": hashlib.sha256(json.dumps(s.record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "original_owner_identity": s.owner, "reservation_id": s.record["reservation_id"], "native_lease_id": s.record["native_lease_id"]}
    negative_pin = put(s.base.invocation / "negative-validation.json", proof)
    result = owners.publish_closed_validation_control(s.base.policy, s.plan, s.base.frozen, s.request_pin, s.base.invocation, negative_pin)
    marker = json.loads(Path(result["path"]).read_bytes())
    assert marker["schema"] == "agent-supervisor/public-validation-closed-owner@2"
    assert marker["binding"] == {key: s.base.frozen["binding"][key] for key in ("namespace", "board_id", "attempt_id")}
    assert marker["binding"]["namespace"] == "ptr_lane_0"
    assert "/control-requests/legal_ir_unattended_repair/legal-ir/" in result["path"]
    assert marker["original_validation_profile"] == owners.original_external_validation_profile(s.base.policy, s.plan, s.request_pin, s.record)
    assert marker["storage_claim_released"] is marker["admitted"] is marker["Lean_admitted"] is marker["retry_authorized"] is False
    assert "semantic_negative" not in marker and s.record["external_charges"] == {s.key: 33554432}
    assert Path(marker["failure_terminal"]["path"]).read_bytes() == Path(terminal_pin["path"]).read_bytes()
    assert Path(marker["original_wait"]["path"]).read_bytes() == Path(wait_pin["path"]).read_bytes()


def test_control_manifest_descriptor_refuses_even_pinned_bytes_outside_immutable_root(metadata_scope, tmp_path):
    s = metadata_scope; policy = copy.deepcopy(s.policy)
    original = Path(policy["control_release"]["manifest"]["path"])
    policy["control_release"]["manifest"] = put(tmp_path / "outside-control-manifest.json", original.read_bytes())
    with pytest.raises(ValueError, match="directly inside"):
        owners.checked_control_release(policy, current_owner=False)


@pytest.mark.parametrize("entry_name,target", [("définition", "plain"), ("plain", "café")])
def test_original_D_unicode_manifest_uses_original_wire_not_A_metadata_wire(external_scope, entry_name, target):
    """Inert original receipt metadata; no external tree/API is inspected."""
    s = external_scope
    record = copy.deepcopy(s.record)
    observation = record["external_fixture_disposition"]
    manifest = json.loads(Path(observation["manifest"]["path"]).read_bytes())
    target_raw = target.encode()
    manifest.update(entries=[{"path": entry_name, "kind": "symlink",
        "identity": {"device": 1, "inode": 2, "mode": 41471, "uid": 1000,
            "links": 1, "bytes": len(target_raw), "mtime_ns": 3, "ctime_ns": 4},
        "target": target, "target_sha256": hashlib.sha256(target_raw).hexdigest()}],
        typed_counts={"regular": 0, "directory": 0, "symlink": 1, "fifo": 0},
        entry_count=1, apparent_bytes=len(target_raw))
    # These are the exact two different stdlib serializer contracts. The D
    # producer uses ensure_ascii=True; owners.canonical intentionally stays UTF8.
    original_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()
    assert original_raw != owners.canonical(manifest)
    inventory = hashlib.sha256(original_raw).hexdigest()
    original_pin = put(Path(s.plan["attempt_directory"]) /
        ("logs/external-synthetic-disposition-" + inventory + ".json"), original_raw + b"\n")
    observation.update(manifest=original_pin, inventory_sha256=inventory,
        typed_counts=manifest["typed_counts"], entry_count=1, apparent_bytes=len(target_raw))
    before = Path(original_pin["path"]).read_bytes()
    result = owners.original_external_validation_profile(s.base.policy, s.plan, s.request_pin, record)
    assert result["terminal_disposition"] == observation
    assert Path(original_pin["path"]).read_bytes() == before
    assert record["external_charges"] == {s.key: 33554432}
    assert result["terminal_disposition"]["ordinary_external_artifact_durability_asserted"] is False


def pure_worker_interface(*, capability="agent-supervisor/repair-validation-v3@1",
        omit_function=None, omit_method=None):
    """Metadata shape fixture only; it is never loaded as executable code."""
    lines = []
    if capability is not None:
        lines.append("V3_VALIDATION_CAPABILITY = " + repr(capability))
    lines.extend(["class PinnedSources:",
        "    def __init__(self, pins, roots, normal_namespace_profiles=None): pass"])
    for name in ("expected_natural_nodes", "authenticated_fixture_mapping"):
        if name != omit_function:
            lines.append("def " + name + "(*args, **kwargs): pass")
    lines.append("class TrustedFixtures:")
    for name in ("__init__", "verify", "set_node", "alias", "compile_source",
            "exec_source", "setup_children", "checked_popen", "close_children",
            "deadline_enter", "deadline_exit"):
        if name != omit_method:
            lines.append("    def " + name + "(self, *args, **kwargs): pass")
    return ("\n".join(lines) + "\n").encode()


def test_complete_worker_interface_preflight_remains_metadata_only(tmp_path):
    worker = put(tmp_path / "worker.py", pure_worker_interface())
    control = {"worker": Path(worker["path"]), "pins": {worker["path"]: worker["sha256"]}}
    before = Path(worker["path"]).read_bytes()
    value = owners.checked_v3_validation_worker(control)
    assert value == {"capability": "agent-supervisor/repair-validation-v3@1",
        "worker": worker, "source_interface_only": True, "worker_executed": False}
    assert Path(worker["path"]).read_bytes() == before


@pytest.mark.parametrize("problem", ["namespace_only", "wrong_capability", "missing_function", "missing_method"])
def test_initial_v3_enrollment_refuses_pure_or_incomplete_worker_before_dispatch(metadata_scope, monkeypatch, problem):
    s = metadata_scope
    worker_path = s.control / "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py"
    if problem == "namespace_only":
        raw = b"class PinnedSources:\n    def __init__(self, pins, roots, normal_namespace_profiles=None): pass\n"
    else:
        raw = pure_worker_interface(capability="old-pure-worker" if problem == "wrong_capability" else "agent-supervisor/repair-validation-v3@1",
            omit_function="authenticated_fixture_mapping" if problem == "missing_function" else None,
            omit_method="checked_popen" if problem == "missing_method" else None)
    worker_path.chmod(0o600); worker_path.write_bytes(raw); worker_path.chmod(0o400)
    worker = pin(worker_path)
    policy = copy.deepcopy(s.policy)
    policy["source_descriptors"] = [worker if row["path"] == worker["path"] else row for row in policy["source_descriptors"]]
    manifest_path = Path(policy["control_release"]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"] = [worker if row["path"] == worker["path"] else row for row in manifest["files"]]
    manifest_path.chmod(0o600); manifest_path.write_bytes(owners.canonical(manifest) + b"\n"); manifest_path.chmod(0o400)
    policy["control_release"]["manifest"] = pin(manifest_path)
    selected = policy["validation"]["selected_checks"]
    policy["validation"].update(external_fixture_profile={}, normal_namespace_profiles=[],
        expected_nodeids=selected, selector_case_counts={node: 1 for node in selected}, child_process_slots=4)
    policy_pin = put(Path(s.policy_pin["path"]).with_name("v3-incompatible-worker-policy.json"), policy)
    monkeypatch.setattr(owners, "checked_worker_fixture_profile", lambda *args, **kwargs: pytest.fail("incompatible worker reached fixture planning"))
    monkeypatch.setattr(owners, "public_source_lease", lambda *args: pytest.fail("incompatible worker reached source admission"))
    with pytest.raises(ValueError, match="v3 validation worker"):
        installer.render_control_owner_commands(policy_pin, validation_timeout_seconds=360, adoption_timeout_seconds=300)


@pytest.mark.parametrize("missing", ["router", "pytest_main"])
def test_initial_v3_enrollment_requires_router_presence_pin_before_dispatch(metadata_scope, monkeypatch, missing):
    s = metadata_scope
    policy = copy.deepcopy(s.policy)
    package = Path(policy["datasets_root"]) / "ipfs_datasets_py"
    directory = package / "optimizers/logic_theorem_optimizer"
    required = [*(directory / name for name in ("validation_run_resources.py", "autoencoder_daemon_resources.py",
        "resource_scheduler.py", "existing_resource_scheduler.py", "proof_resource_safety.py", "runtime_telemetry.py", "__init__.py")),
        package / "optimizers/__init__.py", package / "__init__.py", package / "router_deps.py",
        package / "ipfs_backend_router.py", package / "logic/__init__.py", package / "logic/backends/process.py"]
    declared = {row["path"] for row in policy["source_descriptors"]}
    router = package / "ipfs_backend_router.py"
    for path in required:
        if not path.exists():
            put(path, b"# Private unexecuted normal-metadata source fixture.\n")
        if str(path) not in declared and (missing != "router" or path != router):
            policy["source_descriptors"].append(pin(path))
    assert router.is_file() and (str(router) in {row["path"] for row in policy["source_descriptors"]}) is (missing != "router")
    namespace = package / "logic/backends"; observed = namespace.stat()
    profiles = [{"fullname": "ipfs_datasets_py.logic.backends", "path": str(namespace),
        "device": observed.st_dev, "inode": observed.st_ino, "initializer_absent": True}]
    manifest_path = Path(policy["control_release"]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_bytes()); manifest["namespace_profiles"] = profiles
    manifest_path.chmod(0o600); manifest_path.write_bytes(owners.canonical(manifest) + b"\n"); manifest_path.chmod(0o400)
    policy["control_release"]["manifest"] = pin(manifest_path)
    selected = policy["validation"]["selected_checks"]
    policy["validation"].update(external_fixture_profile={}, normal_namespace_profiles=profiles,
        expected_nodeids=selected, selector_case_counts={node: 1 for node in selected}, child_process_slots=4)
    policy_pin = put(Path(s.policy_pin["path"]).with_name("v3-missing-router-policy.json"), policy)
    monkeypatch.setattr(owners, "checked_worker_fixture_profile", lambda *args, **kwargs: pytest.fail("missing router pin reached fixture planning"))
    monkeypatch.setattr(owners, "public_source_lease", lambda *args, **kwargs: pytest.fail("missing router pin reached source admission"))
    expected_error = ("complete original public v3 normal control source population" if missing == "router"
        else "original Pytest selected-module collection source must be pinned")
    with pytest.raises(ValueError, match=expected_error):
        installer.render_control_owner_commands(policy_pin, validation_timeout_seconds=360, adoption_timeout_seconds=300)
    assert router.read_bytes() == b"# Private unexecuted normal-metadata source fixture.\n"
