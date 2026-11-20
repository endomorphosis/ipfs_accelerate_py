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


# Opt-in V5 controls: real short engine/Popen waits, explicitly MODELED Source
# leases and CLI inspectors. These fixtures confer no native/Source qualification,
# task acceptance, storage release, retry, or installed-profile authority.
def _ordinary_v5_policy(policy):
    value = copy.deepcopy(policy)
    for name in ("external_fixture_profile", "normal_namespace_profiles", "expected_nodeids", "selector_case_counts"):
        value["validation"].pop(name, None)
    value["validation"].update(storage_bytes=16777216, child_process_slots=4,
        ordinary_scratch_profile={"schema": "owned-validation-ordinary-scratch-live/v1"})
    return value


def test_v5_ordinary_policy_plan_and_public_request_preserve_lane_and_registry(plan_scope):
    """Intended tests-first RED: old profile parser rejects this explicit V5."""
    s = plan_scope; policy = _ordinary_v5_policy(s.policy)
    policy["validation"]["selected_checks"] = [s.trusted["path"] + "::test_original_case"]
    policy["validation"]["import_sources"] = [pin(Path(pytest_python.__file__).absolute())]
    owners.checked_validation_profile(policy, {"root": Path(policy["control_release"]["root"]), "namespace_profiles": []})
    policy_pin = put(s.invocation / "v5-owner-policy.json", policy)
    plan = owners.validation_plan(policy, policy_pin, s.frozen, s.freeze_pin, s.root, s.invocation)
    assert plan["ordinary_scratch_profile"] == {"schema": "owned-validation-ordinary-scratch-live/v1"}
    assert "external_fixture_profile" not in plan and "normal_namespace_profiles" not in plan
    assert plan["attempt_directory"].endswith("/ptr_lane_0/legal-ir/" + "a" * 32)
    plan_pin = put(s.invocation / "validation-plan.json", plan)
    request = owners.validation_execution_request(policy, plan, plan_pin)
    assert request["schema"] == "owned-validation-request/v5"
    assert request["ordinary_scratch_profile"] == plan["ordinary_scratch_profile"]
    assert request["declared_external_writable_paths"] == [] and request["storage_bytes"] == 16777216
    assert request["child_process_slots"] == 4 and request["timeout_seconds"] == 300
    local = put(s.invocation / "execution-request.json", request)
    public = owners.publish_validation_request(policy, s.frozen, s.invocation, local)
    assert "/control-requests/legal_ir_unattended_repair/legal-ir/" in public["path"]
    assert owners.checked_descriptor(public) == owners.checked_descriptor(local)
    assert s.frozen["binding"]["namespace"] == "ptr_lane_0"


def test_v5_request_builder_refuses_omitted_lifecycle_source_pin(plan_scope):
    s = plan_scope; policy = _ordinary_v5_policy(s.policy)
    policy_pin = put(s.invocation / "v5-owner-policy.json", policy)
    plan = owners.validation_plan(policy, policy_pin, s.frozen, s.freeze_pin, s.root, s.invocation)
    source = Path(policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/proof_resource_safety.py"
    assert source.is_file()
    plan["original_public_control_pins"].pop(str(source))
    plan_pin = put(s.invocation / "validation-plan.json", plan)
    with pytest.raises(ValueError, match="explicit original lifecycle owner source pins required"):
        owners.validation_execution_request(policy, plan, plan_pin)


@pytest.mark.parametrize("change", ["storage", "aliases"])
def test_v5_ordinary_profile_refuses_budget_or_v3_permission_inference(plan_scope, change):
    policy = _ordinary_v5_policy(plan_scope.policy)
    if change == "storage":
        policy["validation"]["storage_bytes"] = 67108864
    else:
        policy["validation"]["external_fixture_profile"] = {}
    with pytest.raises(ValueError, match="(finite validation profile|explicit V5 ordinary)"):
        owners.checked_validation_profile(policy, {"root": Path(policy["control_release"]["root"]), "namespace_profiles": []})


@pytest.fixture
def modeled_v5_process_scope(tmp_path, monkeypatch):
    """Only engine/Popen is genuine; Source and native verdicts are modeled."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    d = tmp_path / "modeled-D"; cwd = tmp_path / "candidate"; cwd.mkdir()
    invocation = tmp_path / "invocation"; invocation.mkdir()
    marker = tmp_path / "actual-child-started"
    raw = '''import json, os, sys, time
from pathlib import Path

def classify_owned_validation_source_custody(terminal, request_raw, *, expected_request_sha256, owner_pid, owner_birth, returncode, expected_resource_ledger):
    return {"verdict": "unknown", "source_custody_closed": False, "storage_release_authorized": False}

def inspect_owned_validation_no_child_native_closed_v1(terminal, request_raw, *, expected_request_sha256, owner_pid, owner_birth, returncode, expected_resource_ledger, cleanup_timeout_seconds=5):
    raise AssertionError("modeled productive fixture unexpectedly used no-child route")

def inspect_owned_validation_no_child_native_closed_v5(terminal, request_raw, *, expected_request_sha256, owner_pid, owner_birth, returncode, expected_resource_ledger, cleanup_timeout_seconds=5):
    raise AssertionError("modeled productive fixture unexpectedly used V5 no-child route")

def inspect_owned_validation_productive_native_closed_v1(terminal, request_raw, *, expected_request_sha256, owner_pid, owner_birth, returncode, expected_resource_ledger, original_wait_deadline_monotonic):
    q = json.loads(request_raw)
    assert terminal["owner_pid"] == owner_pid and terminal["owner_birth"] == owner_birth
    assert terminal["code"] == returncode == 1 and terminal["request_sha256"] == expected_request_sha256
    assert expected_resource_ledger == str(Path(q["roots"][0]) / "private-ledger.json")
    assert original_wait_deadline_monotonic > time.monotonic()
    if q["selected_checks"] == ["unknown"]:
        return {"verdict": "unknown", "source_custody_closed": False, "storage_release_authorized": False}
    return {"verdict": "proven_original_productive_native_closed_storage_separate", "source_custody_closed": True, "storage_release_authorized": False}

if __name__ == "__main__":
    q = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_bytes())
    sha = sys.argv[sys.argv.index("--expected-request-sha256") + 1]
    Path(q["roots"][0], "actual-child-started").write_text("actual Popen reached")
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    time.sleep(.05)
    code = 0 if q["selected_checks"] == ["success"] else 1
    row = {"status": "active", "storage_bytes": 16777216, "child": {"modeled": True}}
    print(json.dumps({"owner_pid": os.getpid(), "owner_birth": fields[19], "code": code, "request_sha256": sha,
        "reservation": {"status": "active", "record": row, "cleanup_error": "DaemonResourceError"}}), flush=True)
    raise SystemExit(code)
'''
    cli = put(d / "scripts/ops/legal_ir/run_owned_validation.py", raw.encode())
    events = []; held = []
    @contextmanager
    def source_read(root, *, mode, timeout_seconds):
        assert mode == "read" and timeout_seconds > 0
        events.append(("enter", str(root)))
        try:
            yield SimpleNamespace(to_dict=lambda: {"modeled_Source_only": True, "root": str(root)})
        finally:
            events.append(("exit", str(root)))
    monkeypatch.setattr(owners, "public_source_lease", lambda policy: source_read)
    class ModeledUnknown(RuntimeError):
        pass
    def held_local(self, error):
        held.append({"error": error, "source_closed": self.closed, "process": self.process,
            "returncode": None if self.process is None else self.process.returncode,
            "open_contexts": sum(k == "enter" for k, _ in events) - sum(k == "exit" for k, _ in events),
            "identity_error": self.identity_error, "engine_returned": self.engine_returned})
        # Test-only cleanup of modeled contexts and this fixture's exact child.
        # Production hold retains both indefinitely; this is no custody proof.
        try:
            if self.process is not None and self.process.returncode is None:
                self.process.wait(timeout=2)
        finally:
            self.stack.close()
        raise ModeledUnknown(type(error).__name__) from error
    monkeypatch.setattr(owners._V5ValidationSourceCustody, "hold", held_local)
    directory = d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
    lifecycle_pins = [put(directory / name, b"# Modeled lifecycle source only.\n") for name in (
        "validation_run_resources.py", "autoencoder_daemon_resources.py", "resource_scheduler.py",
        "existing_resource_scheduler.py", "proof_resource_safety.py")]
    policy = {"datasets_root": str(d), "python": sys.executable,
        "source_descriptors": [cli, pin(Path(engine.__file__).absolute()), pin(Path(owners.__file__).absolute()), *lifecycle_pins],
        "validation": {"ordinary_scratch_profile": {"schema": "owned-validation-ordinary-scratch-live/v1"},
            "cpu_slots": 1, "memory_mb": 1536, "storage_bytes": 16777216, "timeout_seconds": 300,
            "child_process_slots": 4, "scheduler_state_path": str(tmp_path / "private-native.json"),
            "expected_scheduler_policy_sha256": "a" * 64, "roots": [str(tmp_path)],
            "resource_ledger": str(tmp_path / "private-ledger.json")}}
    def setup(mode, *, omit_lifecycle_pin=False):
        q = {"schema": "owned-validation-request/v5", "ordinary_scratch_profile": policy["validation"]["ordinary_scratch_profile"],
            "declared_external_writable_paths": [], "process_scope": "sampled_birth_bound_descendants", "cwd": str(cwd),
            "selected_checks": [mode], "attempt_directory": str(tmp_path / "unexecuted-attempt"),
            "source_sha256": {r["path"]: r["sha256"] for r in policy["source_descriptors"]},
            **{k: policy["validation"][k] for k in ("cpu_slots", "memory_mb", "storage_bytes", "timeout_seconds",
                "child_process_slots", "scheduler_state_path", "expected_scheduler_policy_sha256", "roots")}}
        if omit_lifecycle_pin:
            q["source_sha256"].pop(str(directory / "proof_resource_safety.py"))
        request_pin = put(invocation / "request.json", q)
        plan = {"selected_checks": [mode], "attempt_directory": q["attempt_directory"], "candidate_root": str(cwd)}
        context = {"plan": plan, "request_pin": request_pin, "invocation": invocation, "frozen": {}}
        argv = [sys.executable, "-I", "-B", cli["path"], "--request", request_pin["path"],
            "--expected-request-sha256", request_pin["sha256"], "--resource-ledger", policy["validation"]["resource_ledger"]]
        return context, argv
    return SimpleNamespace(policy=policy, setup=setup, cwd=cwd, invocation=invocation, marker=marker,
        events=events, held=held, Unknown=ModeledUnknown, engine=engine)


@pytest.mark.parametrize("mode", ["success", "active"])
def test_v5_real_engine_wait_precedes_modeled_verifier_and_source_closeout(modeled_v5_process_scope, monkeypatch, mode):
    s = modeled_v5_process_scope; context, argv = s.setup(mode)
    def verify(policy, plan, request_pin, terminal, invocation):
        wait = json.loads((invocation / "runner-wait.json").read_bytes())
        assert wait["request"] == request_pin and wait["actual_Popen_returned"] is True
        assert wait["actual_process_waited"] is True and wait["returncode"] == 0
        assert wait["runner_identity"]["pid"] == terminal["owner_pid"]
        assert [kind for kind, _ in s.events] == ["enter", "enter"]
        return {"modeled_checks_only": True}
    monkeypatch.setattr(owners, "verify_validation_evidence", verify)
    completed, witness = owners.original_runner_wait(s.policy, argv, s.cwd, 360, validation_context=context)
    assert completed.returncode == (0 if mode == "success" else 1) and s.marker.is_file()
    assert witness["actual_Popen_returned"] is True and witness["actual_process_waited"] is True
    parent = witness["parent_source_custody"]
    assert parent["source_closed"] is True and parent["native_closed"] is True
    assert parent["storage_release_authorized"] is False and parent["retry_authorized"] is False
    assert parent["closure_kind"] == ("existing_verified_success" if mode == "success" else "productive_native_closed_storage_separate")
    assert parent["storage_status"] == ("released" if mode == "success" else "active")
    assert [kind for kind, _ in s.events] == ["enter", "enter", "exit", "exit"] and s.held == []
    saved = json.loads(owners.checked_descriptor(witness["parent_source_closeout"]))
    assert saved["request"] == context["request_pin"] and saved["original_wait"] == parent["original_wait"]
    assert saved["parent_source_custody"] == parent
    if mode == "success":
        assert owners.verify_v5_parent_source_closeout(s.policy, context["request_pin"], s.invocation,
            witness["parent_source_closeout"], {"modeled_checks_only": True}) == saved


@pytest.mark.parametrize("fault", ["identity_io", "unknown_native"])
def test_v5_actual_engine_reaps_before_modeled_unknown_retains_source(modeled_v5_process_scope, monkeypatch, fault):
    s = modeled_v5_process_scope; context, argv = s.setup("unknown" if fault == "unknown_native" else "active")
    if fault == "identity_io":
        def identity_io(pid):
            raise OSError("modeled process observation unavailable")
        monkeypatch.setattr(owners, "proc_identity", identity_io)
    with pytest.raises(s.Unknown):
        owners.original_runner_wait(s.policy, argv, s.cwd, 360, validation_context=context)
    assert s.marker.is_file() and len(s.held) == 1
    held = s.held[0]
    assert held["process"] is not None and held["returncode"] == 1 and held["engine_returned"] is True
    assert held["source_closed"] is False and held["open_contexts"] == 2
    assert held["identity_error"] == ("OSError" if fault == "identity_io" else None)
    assert not (s.invocation / "parent-source-closeout.json").exists()


def test_v5_prespawn_deadline_refuses_actual_popen_without_renewal(modeled_v5_process_scope, monkeypatch):
    import time as real_time
    s = modeled_v5_process_scope; context, argv = s.setup("active")
    phase = {"engine_seen": False, "after_engine_pin": 0}
    actual_descriptor = owners.descriptor
    engine_path = str(Path(s.engine.__file__).absolute())
    def descriptor(path):
        value = actual_descriptor(path)
        if str(path) == engine_path:
            phase["engine_seen"] = True
        return value
    def monotonic():
        if phase["engine_seen"]:
            phase["after_engine_pin"] += 1
            if phase["after_engine_pin"] >= 3:
                return 461.0
        return 100.0
    monkeypatch.setattr(owners, "descriptor", descriptor)
    monkeypatch.setattr(owners, "time", SimpleNamespace(monotonic=monotonic, sleep=real_time.sleep))
    with pytest.raises(ValueError, match="original V5 source/native proof budget expired"):
        owners.original_runner_wait(s.policy, argv, s.cwd, 360, validation_context=context)
    assert not s.marker.exists() and s.held == []
    assert [kind for kind, _ in s.events] == ["enter", "enter", "exit", "exit"]



@pytest.mark.parametrize(
    "delays, allowances",
    [((7, 5), (30, 23)), ((31, 0), (30,)), ((4, 27), (30, 26))],
    ids=["shared-allowance", "first-read-overrun", "second-read-overrun"],
)
def test_v5_source_read_admission_preserves_the_full_launch_window(
    modeled_v5_process_scope, monkeypatch, delays, allowances,
):
    s = modeled_v5_process_scope
    context, argv = s.setup("active")
    now = 100.0
    waits = []
    monkeypatch.setattr(owners, "time", SimpleNamespace(monotonic=lambda: now))

    @contextmanager
    def source_read(root, *, mode, timeout_seconds):
        nonlocal now
        assert mode == "read"
        now += delays[len(waits)]
        waits.append((root, timeout_seconds))
        s.events.append(("enter", root))
        try:
            yield SimpleNamespace(to_dict=lambda: {"modeled_Source_only": True})
        finally:
            s.events.append(("exit", root))

    monkeypatch.setattr(owners, "public_source_lease", lambda policy: source_read)
    if sum(delays) > 30:
        # Git identity checks in the original Source owner have separate
        # bounds. Model a read returning after its lock-wait allowance.
        with pytest.raises(ValueError, match="full nominal request window"):
            owners._V5ValidationSourceCustody(s.policy, argv, s.cwd, 360, context)
    else:
        custody = owners._V5ValidationSourceCustody(s.policy, argv, s.cwd, 360, context)
        try:
            assert custody.launch_clock() == 318
            assert (custody.deadline, custody.work_deadline, custody.cleanup_deadline) == (460, 430, 440)
        finally:
            custody.stack.close()
    roots = [Path(s.policy["datasets_root"]), s.cwd][:len(allowances)]
    assert waits == list(zip(roots, allowances, strict=True))
    assert s.events == [
        *[("enter", root) for root in roots],
        *[("exit", root) for root in reversed(roots)],
    ]
    assert s.held == [] and not s.marker.exists()
    assert not (s.invocation / "runner-wait.json").exists()


def test_v5_direct_runner_refuses_omitted_lifecycle_pin_before_source_or_engine(modeled_v5_process_scope, monkeypatch):
    s = modeled_v5_process_scope; context, argv = s.setup("active", omit_lifecycle_pin=True)
    monkeypatch.setattr(s.engine, "run_command", lambda *args, **kwargs: pytest.fail("engine reached"))
    with pytest.raises(ValueError, match="explicit original lifecycle owner source pins required"):
        owners.original_runner_wait(s.policy, argv, s.cwd, 360, validation_context=context)
    assert s.events == [] and s.held == [] and not s.marker.exists()
    assert not (s.invocation / "runner-wait.json").exists()


def test_v5_crossed_candidate_context_refused_before_modeled_source(modeled_v5_process_scope):
    s = modeled_v5_process_scope; context, argv = s.setup("active")
    context["plan"]["candidate_root"] = str(s.cwd.parent / "crossed-candidate")
    with pytest.raises(ValueError, match="V5 original request/context differs"):
        owners.original_runner_wait(s.policy, argv, s.cwd, 10, validation_context=context)
    assert s.events == [] and not s.marker.exists()


@pytest.mark.parametrize("entry", ["verify", "adopt"])
def test_v5_existing_verifier_and_adopter_require_persisted_parent_closeout(modeled_v5_process_scope, monkeypatch, entry):
    """Real wait, modeled semantic acceptance; no adoption or Source authority."""
    s = modeled_v5_process_scope; context, argv = s.setup("success")
    summary = {"modeled_checks_only": True}
    monkeypatch.setattr(owners, "verify_validation_evidence", lambda policy, plan, request_pin, terminal, invocation: summary)
    completed, witness = owners.original_runner_wait(s.policy, argv, s.cwd, 360, validation_context=context)
    terminal = put(s.invocation / "original-terminal.json", json.loads(completed.stdout))
    put(s.invocation / "validation-plan.json", context["plan"])
    receipt = {"schema": owners.VALIDATION_SCHEMA, "passed": True, "freeze_sha256": "a" * 64,
        "source_commitment": "b" * 64, "original_receipt_reference": terminal["path"],
        "original_receipt_descriptor": terminal, "verified_checks": summary,
        "parent_source_closeout": witness["parent_source_closeout"]}
    receipt_pin = put(s.invocation / "receipt.json", receipt)
    monkeypatch.setattr(owners, "checked_policy", lambda *args, **kwargs: s.policy)
    monkeypatch.setattr(owners, "frozen_candidate", lambda *args: {"freeze_sha256": "a" * 64})
    monkeypatch.setattr(owners, "original_execution_request", lambda invocation: context["request_pin"])
    args = SimpleNamespace(policy=Path(receipt_pin["path"]), policy_sha256=receipt_pin["sha256"],
        receipt_path=Path(receipt_pin["path"]), invocation_root=s.invocation,
        freeze_path=s.invocation / "unused-model-freeze.json", source_commitment="b" * 64, candidate_root=s.cwd)
    if entry == "verify":
        owners.verify_validation(args)
        changed = dict(receipt); changed.pop("parent_source_closeout")
        args.receipt_path = Path(put(s.invocation / "missing-parent-receipt.json", changed)["path"])
        with pytest.raises(KeyError, match="parent_source_closeout"):
            owners.verify_validation(args)
    else:
        reached = []
        class StopBeforeAdoption(RuntimeError):
            pass
        def contract(*args, **kwargs):
            reached.append(True)
            raise StopBeforeAdoption("modeled barrier before any adoption work")
        monkeypatch.setattr(owners, "current_profile_contract", contract)
        pending = {"stage": "adoption_pending", "freeze_sha256": "a" * 64, "validation": receipt}
        args.validation_path = Path(put(s.invocation / "pending.json", pending)["path"])
        with pytest.raises(StopBeforeAdoption):
            owners.adopt(args)
        assert reached == [True]
        changed = copy.deepcopy(pending); changed["validation"].pop("parent_source_closeout")
        args.validation_path = Path(put(s.invocation / "missing-parent-pending.json", changed)["path"])
        with pytest.raises(KeyError, match="parent_source_closeout"):
            owners.adopt(args)
        assert reached == [True]

    # A self-consistent changed receipt + changed parent observation must still
    # disagree with the newly returned real-verifier summary at BOTH callsites.
    closeout_path = Path(witness["parent_source_closeout"]["path"])
    changed_parent = json.loads(closeout_path.read_bytes())
    changed_parent["parent_source_custody"]["verified_checks"] = {"changed_modeled_summary": True}
    closeout_path.chmod(0o600)
    closeout_path.write_bytes(owners.canonical(changed_parent) + b"\n")
    closeout_path.chmod(0o400)
    changed_receipt = dict(receipt, verified_checks={"changed_modeled_summary": True}, parent_source_closeout=pin(closeout_path))
    if entry == "verify":
        args.receipt_path = Path(put(s.invocation / "changed-summary-receipt.json", changed_receipt)["path"])
        with pytest.raises(ValueError, match="V5 verified summary differs"):
            owners.verify_validation(args)
    else:
        changed_pending = {**pending, "validation": changed_receipt}
        args.validation_path = Path(put(s.invocation / "changed-summary-pending.json", changed_pending)["path"])
        with pytest.raises(ValueError, match="V5 verified summary differs"):
            owners.adopt(args)
        assert reached == [True]


def test_v5_postclose_metadata_io_failure_propagates_without_unknown_hold(modeled_v5_process_scope, monkeypatch):
    s = modeled_v5_process_scope; context, argv = s.setup("active")
    actual_new_json = owners.new_json
    def fail_postclose(path, value):
        if path == s.invocation / "parent-source-closeout.json":
            assert [kind for kind, _ in s.events] == ["enter", "enter", "exit", "exit"]
            raise OSError("modeled postclose metadata write unavailable")
        return actual_new_json(path, value)
    monkeypatch.setattr(owners, "new_json", fail_postclose)
    with pytest.raises(OSError, match="postclose metadata write unavailable"):
        owners.original_runner_wait(s.policy, argv, s.cwd, 360, validation_context=context)
    assert s.marker.is_file() and s.held == []
    assert [kind for kind, _ in s.events] == ["enter", "enter", "exit", "exit"]
    assert not (s.invocation / "parent-source-closeout.json").exists()



# Typed wait and same-budget controls below model every process/Source/native
# boundary explicitly. No new OS process, signal, escaped-child or retry fixture.
@pytest.fixture
def typed_v5_modeled_boundaries(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    now = {"value": 1010.0}; events = []; native_calls = []; verified = []
    monkeypatch.setattr(owners, "time", SimpleNamespace(monotonic=lambda: now["value"]))
    class ModeledPopen:
        # Attributes only. Any test reliance on poll/wait would fail explicitly.
        def __init__(self, argv):
            self.args = list(argv); self.pid = 4101; self.returncode = 1
        def poll(self):
            pytest.fail("modeled typed wait cannot infer reap through poll")
        def wait(self, *args, **kwargs):
            pytest.fail("modeled typed wait cannot launch or reap a process")
    monkeypatch.setattr(owners, "_ORIGINAL_POPEN", ModeledPopen)
    class ModeledStack:
        def close(self):
            events.append("source_exit")
    c = object.__new__(owners._V5ValidationSourceCustody)
    c.argv = ["explicitly-modeled-original-cli"]
    c.process = ModeledPopen(c.argv); c.identity = {"pid": 4101, "birth": "12345"}
    c.started = 1000.0; c.deadline = 1360.0; c.work_deadline = 1330.0; c.cleanup_deadline = 1340.0
    c.engine_source = {"path": str(Path(engine.__file__).absolute()), "bytes": 1, "sha256": "e" * 64}
    producer = {"path": str(Path(owners.__file__).absolute()), "bytes": 1, "sha256": "a" * 64}
    c.policy = {"source_descriptors": [c.engine_source, producer],
        "validation": {"ordinary_scratch_profile": {"schema": "owned-validation-ordinary-scratch-live/v1"}}}
    c.engine = engine; c.engine_runner = engine.run_command; c.engine_returned = True
    c.engine_started_at = 1001.0; c.engine_finished_at = None
    c.spawn_returned_at = 1002.0; c.identity_observed_at = 1003.0
    c.spawn_attempted = True; c.execution_ref = None; c.execution_observation = None
    c.process_count = 0; c.identity_error = None; c.wait_ref = None; c.source_exit_finished_at = None
    c.stack = ModeledStack(); c.closed = False; c.receipts = [{"modeled": "D"}, {"modeled": "candidate"}]
    c.request = {"timeout_seconds": 300, "storage_bytes": 16777216}
    c.raw = owners.canonical(c.request)
    c.request_pin = put(tmp_path / "request.json", c.raw)
    c.context = {"invocation": tmp_path, "plan": {}, "frozen": {}}
    c.ledger = tmp_path / "unexecuted-ledger.json"
    c.guard = lambda: events.append("source_guard")  # Explicit modeled Source boundary.
    verdict = {"value": {"verdict": "proven_original_productive_native_closed_storage_separate",
        "source_custody_closed": True, "storage_release_authorized": False}}
    def no_grant(terminal, request_raw, **arguments):
        native_calls.append(("no_grant", terminal, arguments))
        return {"verdict": "unknown", "source_custody_closed": False, "storage_release_authorized": False}
    def no_child(terminal, request_raw, **arguments):
        native_calls.append(("no_child", terminal, arguments))
        return {"verdict": "proven_original_no_child_native_closed",
            "source_custody_closed": True, "storage_release_authorized": False}
    def inspector(terminal, request_raw, **arguments):
        native_calls.append(("inspector", terminal, arguments))
        assert arguments["original_wait_deadline_monotonic"] == 1360.0
        return verdict["value"]
    c.no_grant = no_grant; c.no_child = no_child; c.inspector = inspector
    def verify(policy, plan, request_pin, terminal, invocation):
        saved = json.loads((invocation / "runner-wait.json").read_bytes())
        assert saved["execution_observation"] == c.execution_ref
        assert (invocation / "runner-execution-observation.json").is_file()
        verified.append(terminal)
        return {"explicit_modeled_verifier": True}
    monkeypatch.setattr(owners, "verify_validation_evidence", verify)
    terminal = {"reservation": {"status": "active", "cleanup_error": "modeled-retention-failed",
        "record": {"status": "active", "storage_bytes": 16777216, "child": {"modeled": True}}}}
    def result(*, kind="completed", public=1, actual=1, output="complete", waited=True, edits=None, stdout=None):
        c.process.returncode = actual
        values = {"schema": "command-execution-observation/v1", "completion_kind": kind,
            "process_pid": c.process.pid, "process_returncode": actual, "leader_waited": waited,
            "output_capture": output, "work_deadline_monotonic": c.work_deadline,
            "cleanup_deadline_monotonic": c.cleanup_deadline, "observed_at_monotonic": 1005.0}
        values.update(edits or {})
        observation = engine.CommandExecutionObservation(**values)
        return engine.CommandResult(tuple(c.argv), public, owners.canonical(terminal).decode() if stdout is None else stdout,
            "arbitrary child diagnostic mentioning timeout124", execution_observation=observation)
    def witness(completed):
        observation = completed.execution_observation
        return {"runner_identity": c.identity, "actual_Popen_returned": True,
            "actual_process_waited": None if observation is None else observation.leader_waited,
            "actual_process_returncode": c.process.returncode, "engine_source": c.engine_source}
    def record(completed):
        c.record_execution(completed, engine, [(c.process, c.identity)])
        return witness(completed)
    return SimpleNamespace(c=c, engine=engine, now=now, events=events, native_calls=native_calls,
        verified=verified, verdict=verdict, result=result, record=record, witness=witness,
        terminal=terminal, producer=producer, invocation=tmp_path)


@pytest.mark.parametrize("kind,public,actual", [("completed", 124, 124), ("timeout", 124, -9), ("timeout", 124, -15)])
def test_v5_typed_wait_separates_normal124_and_timeout_actual_code(typed_v5_modeled_boundaries, kind, public, actual):
    s = typed_v5_modeled_boundaries; completed = s.result(kind=kind, public=public, actual=actual)
    witness = s.record(completed)
    result = s.c.finish(completed, witness)
    observed = json.loads(Path(s.c.execution_ref["path"]).read_bytes())
    assert observed["public_returncode"] == public and observed["actual_process_returncode"] == actual
    assert observed["execution_observation"]["completion_kind"] == kind
    assert observed["execution_observation"]["leader_waited"] is True
    assert [call[0] for call in s.native_calls] == ["no_grant", "inspector"]
    assert all(call[2]["returncode"] == actual for call in s.native_calls)
    assert result["source_closed"] is True and result["storage_status"] == "active"
    assert result["storage_release_authorized"] is False and result["retry_authorized"] is False
    assert result["authority_deadline_monotonic"] == 1360.0 and result["work_deadline_monotonic"] == 1330.0


def test_v5_typed_complete_zero_requires_fresh_verifier_before_source_exit(typed_v5_modeled_boundaries):
    s = typed_v5_modeled_boundaries; completed = s.result(public=0, actual=0)
    result = s.c.finish(completed, s.record(completed))
    assert s.verified == [s.terminal] and s.native_calls == []
    assert result["closure_kind"] == "existing_verified_success" and result["source_closed"] is True
    assert result["execution_observation"] == s.c.execution_ref and s.events[-1] == "source_exit"
    assert result["schema"] == "agent-supervisor/v5-validation-parent-source-custody@2"


@pytest.mark.parametrize("capture", ["incomplete", "unknown"])
@pytest.mark.parametrize("kind,public,actual", [("completed", 0, 0), ("completed", 124, 124), ("timeout", 124, -9)])
def test_v5_valid_json_prefix_with_unobserved_EOF_never_reaches_classifier(typed_v5_modeled_boundaries, capture, kind, public, actual):
    s = typed_v5_modeled_boundaries; completed = s.result(kind=kind, public=public, actual=actual, output=capture)
    witness = s.record(completed)
    with pytest.raises(ValueError, match="terminal EOF"):
        s.c.finish(completed, witness)
    saved = json.loads(Path(s.c.execution_ref["path"]).read_bytes())
    assert saved["execution_observation"]["output_capture"] == capture
    assert saved["actual_process_returncode"] == actual and saved["source_closed"] is False
    assert s.native_calls == [] and s.verified == [] and s.c.closed is False
    assert "source_exit" not in s.events and not (s.invocation / "runner-wait.json").exists()


@pytest.mark.parametrize("fault", ["missing_observation", "missing_birth", "unknown_wait", "false_wait",
    "wrong_pid", "wrong_returncode", "wrong_work_deadline", "wrong_cleanup_deadline", "future_observation",
    "early_observation", "unknown_kind", "foreign_schema", "boolean_returncode"])
def test_v5_typed_provenance_fault_persists_observation_and_retains_context(typed_v5_modeled_boundaries, fault):
    s = typed_v5_modeled_boundaries
    edits = {"wrong_pid": {"process_pid": 9999}, "wrong_returncode": {"process_returncode": 2},
        "wrong_work_deadline": {"work_deadline_monotonic": 1331.0},
        "wrong_cleanup_deadline": {"cleanup_deadline_monotonic": 1341.0},
        "future_observation": {"observed_at_monotonic": 1011.0},
        "early_observation": {"observed_at_monotonic": 1000.0},
        "unknown_kind": {"completion_kind": "not_spawned"}, "foreign_schema": {"schema": "caller-booleans/v1"},
        "boolean_returncode": {"process_returncode": True}}
    completed = s.result(waited=None if fault == "unknown_wait" else False if fault == "false_wait" else True,
        edits=edits.get(fault))
    if fault == "missing_observation":
        completed = s.engine.CommandResult(completed.command, completed.returncode, completed.stdout, completed.stderr)
    if fault == "missing_birth":
        s.c.identity = None; s.c.identity_error = "OSError"
    witness = s.record(completed)
    with pytest.raises(ValueError):
        s.c.finish(completed, witness)
    saved = json.loads(Path(s.c.execution_ref["path"]).read_bytes())
    assert saved["public_returncode"] == 1 and saved["actual_process_returncode"] == 1
    assert saved["actual_Popen_returned"] is True and saved["source_closed"] is False
    assert s.c.process.pid == 4101 and s.c.closed is False and "source_exit" not in s.events
    assert s.native_calls == [] and s.verified == []


def test_v5_expired_original_D_still_records_wait_without_fresh_proof(typed_v5_modeled_boundaries):
    s = typed_v5_modeled_boundaries; completed = s.result(kind="timeout", public=124, actual=-9)
    s.now["value"] = 1360.0
    witness = s.record(completed)
    with pytest.raises(ValueError, match="original V5 source/native proof budget expired"):
        s.c.finish(completed, witness)
    saved = json.loads(Path(s.c.execution_ref["path"]).read_bytes())
    assert saved["actual_process_returncode"] == -9 and saved["public_returncode"] == 124
    assert saved["authority_deadline_monotonic"] == s.c.deadline == 1360.0
    assert saved["execution_observation"]["leader_waited"] is True
    assert s.native_calls == [] and s.verified == [] and s.events == [] and s.c.closed is False


def test_v5_unknown_fresh_native_inspector_cannot_close_source(typed_v5_modeled_boundaries):
    s = typed_v5_modeled_boundaries; completed = s.result()
    s.verdict["value"] = {"verdict": "unknown", "source_custody_closed": False, "storage_release_authorized": False}
    with pytest.raises(ValueError, match="native/source closure remains unknown"):
        s.c.finish(completed, s.record(completed))
    assert [call[0] for call in s.native_calls] == ["no_grant", "inspector"]
    assert s.c.deadline == 1360.0 and s.c.closed is False and "source_exit" not in s.events
    assert s.c.process.returncode == 1 and s.c.wait_ref is not None and s.c.execution_ref is not None


@pytest.mark.parametrize("now,reason", [(1031.0, "full nominal request window"), (1330.0, "worker window expired"),
    (1360.0, "source/native proof budget expired")])
def test_v5_fixed_reserve_refuses_before_launch_without_deadline_renewal(typed_v5_modeled_boundaries, now, reason):
    s = typed_v5_modeled_boundaries; s.now["value"] = now; s.c.spawn_attempted = False; s.c.process = None
    with pytest.raises(ValueError, match=reason):
        s.c.launch_clock()
    assert s.c.spawn_attempted is False and s.c.process is None and s.native_calls == []
    assert (s.c.deadline, s.c.work_deadline, s.c.cleanup_deadline) == (1360.0, 1330.0, 1340.0)


def test_v5_exact_prelaunch_300_is_nominal_cli_window_only(typed_v5_modeled_boundaries):
    s = typed_v5_modeled_boundaries; s.now["value"] = 1030.0
    assert s.c.launch_clock() == 300.0
    completed = s.result(); s.record(completed)
    saved = json.loads(Path(s.c.execution_ref["path"]).read_bytes())
    assert saved["requested_nominal_window_seconds"] == 300
    assert "no inner SDK productive-time guarantee" in saved["window_scope"]
    assert s.c.deadline == 1360.0 and s.c.work_deadline == 1330.0


def test_v5_spawn_exception_records_partial_unknown_without_fabricated_child(typed_v5_modeled_boundaries):
    s = typed_v5_modeled_boundaries; s.c.process = None; s.c.identity = None
    s.c.engine_returned = False; s.c.spawn_returned_at = None; s.c.identity_observed_at = None
    s.c.record_execution(None, s.engine, [])
    saved = json.loads(Path(s.c.execution_ref["path"]).read_bytes())
    assert saved["spawn_attempted"] is True and saved["actual_Popen_returned"] is False
    assert saved["execution_observation"] is None and saved["actual_process_returncode"] is None
    assert saved["source_closed"] is False and saved["native_closed"] is False
    assert saved["storage_release_authorized"] is False and saved["retry_authorized"] is False
    assert s.c.closed is False and "source_exit" not in s.events


def test_v5_late_irreversible_source_exit_is_recorded_closed_without_acceptance(typed_v5_modeled_boundaries):
    s = typed_v5_modeled_boundaries; completed = s.result()
    def late_close():
        s.events.append("source_exit"); s.now["value"] = 1361.0
    s.c.stack.close = late_close
    with pytest.raises(ValueError, match="Source exit completed after"):
        s.c.finish(completed, s.record(completed))
    saved = json.loads((s.invocation / "source-custody-late-close.json").read_bytes())
    assert s.c.closed is True and saved["source_closed"] is True
    assert saved["validation_accepted"] is False and saved["storage_release_authorized"] is False
    assert not (s.invocation / "parent-source-closeout.json").exists()


def _typed_v5_modeled_closeout(s, monkeypatch):
    """Real local metadata; current source/engine pins are explicitly modeled."""
    completed = s.result(public=0, actual=0); witness = s.record(completed)
    custody = s.c.finish(completed, witness)
    actual_descriptor = owners.descriptor
    def descriptor(path):
        path = str(Path(path).absolute())
        if path == s.producer["path"]:
            return dict(s.producer)
        if path == s.c.engine_source["path"]:
            return dict(s.c.engine_source)
        return actual_descriptor(Path(path))
    monkeypatch.setattr(owners, "descriptor", descriptor)
    closeout = {"schema": "agent-supervisor/v5-validation-parent-source-closeout@2", "request": s.c.request_pin,
        "original_wait": s.c.wait_ref, "producer": s.producer, "parent_source_custody": custody}
    closeout_pin = put(s.invocation / "parent-source-closeout.json", closeout)
    return closeout, closeout_pin


@pytest.mark.parametrize("entry", ["verify", "adopt"])
@pytest.mark.parametrize("fault", ["missing", "changed", "rebound_incomplete", "historical_schema",
    "rebound_unknown_birth", "rebound_boolean_code", "rebound_authority", "rebound_extra_field"])
def test_v5_verify_and_direct_adopt_refuse_missing_changed_or_detached_typed_EOF(typed_v5_modeled_boundaries, monkeypatch, entry, fault):
    s = typed_v5_modeled_boundaries; closeout, closeout_pin = _typed_v5_modeled_closeout(s, monkeypatch)
    summary = {"explicit_modeled_verifier": True}
    assert owners.verify_v5_parent_source_closeout(s.c.policy, s.c.request_pin, s.invocation, closeout_pin, summary) == closeout
    def rewrite(path, value):
        path.chmod(0o600); path.write_bytes(owners.canonical(value) + b"\n"); path.chmod(0o400)
        return pin(path)
    execution_path = Path(s.c.execution_ref["path"])
    if fault == "missing":
        execution_path.unlink()
    elif fault == "changed":
        observed = json.loads(execution_path.read_bytes()); observed["public_returncode"] = 124
        rewrite(execution_path, observed)
    elif fault == "historical_schema":
        closeout["schema"] = "agent-supervisor/v5-validation-parent-source-closeout@1"
        closeout_pin = rewrite(Path(closeout_pin["path"]), closeout)
    else:
        observed = json.loads(execution_path.read_bytes())
        if fault == "rebound_incomplete":
            observed["execution_observation"]["output_capture"] = "incomplete"
        elif fault == "rebound_unknown_birth":
            observed["runner_identity"]["birth"] = ""
        elif fault == "rebound_boolean_code":
            observed["actual_process_returncode"] = False
            observed["execution_observation"]["process_returncode"] = False
        elif fault == "rebound_authority":
            observed["source_closed"] = True
        else:
            observed["unprovided_authority"] = True
        changed_execution = rewrite(execution_path, observed)
        wait_path = Path(s.c.wait_ref["path"]); wait = json.loads(wait_path.read_bytes())
        wait["execution_observation"] = changed_execution
        wait["runner_identity"] = observed["runner_identity"]
        wait["actual_process_returncode"] = observed["actual_process_returncode"]
        changed_wait = rewrite(wait_path, wait)
        closeout["original_wait"] = changed_wait
        closeout["parent_source_custody"]["original_wait"] = changed_wait
        closeout["parent_source_custody"]["execution_observation"] = changed_execution
        closeout_pin = rewrite(Path(closeout_pin["path"]), closeout)
    terminal_pin = put(s.invocation / "original-terminal.json", s.terminal)
    receipt = {"schema": owners.VALIDATION_SCHEMA, "passed": True, "freeze_sha256": "a" * 64,
        "source_commitment": "b" * 64, "original_receipt_reference": terminal_pin["path"],
        "original_receipt_descriptor": terminal_pin, "verified_checks": summary, "parent_source_closeout": closeout_pin}
    receipt_pin = put(s.invocation / "receipt.json", receipt)
    put(s.invocation / "validation-plan.json", {})
    monkeypatch.setattr(owners, "checked_policy", lambda *args, **kwargs: s.c.policy)
    monkeypatch.setattr(owners, "frozen_candidate", lambda *args: {"freeze_sha256": "a" * 64})
    monkeypatch.setattr(owners, "original_execution_request", lambda invocation: s.c.request_pin)
    monkeypatch.setattr(owners, "verify_validation_evidence", lambda *args, **kwargs: summary)
    rechecked = []; real_recheck = owners.verify_v5_parent_source_closeout
    def recheck(*args, **kwargs):
        rechecked.append(True)
        return real_recheck(*args, **kwargs)
    monkeypatch.setattr(owners, "verify_v5_parent_source_closeout", recheck)
    reached = []
    monkeypatch.setattr(owners, "current_profile_contract", lambda *args, **kwargs: reached.append(True))
    args = SimpleNamespace(policy=Path(receipt_pin["path"]), policy_sha256=receipt_pin["sha256"],
        receipt_path=Path(receipt_pin["path"]), invocation_root=s.invocation,
        freeze_path=s.invocation / "unexecuted-freeze.json", source_commitment="b" * 64, candidate_root=s.invocation)
    if entry == "adopt":
        args.validation_path = Path(put(s.invocation / "pending.json", {
            "stage": "adoption_pending", "freeze_sha256": "a" * 64, "validation": receipt})["path"])
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        (owners.verify_validation if entry == "verify" else owners.adopt)(args)
    assert rechecked == [True] and reached == []


@pytest.mark.parametrize("spawn_uncertain", [False, True])
def test_v5_runner_passes_original_W_C_and_persists_exception_before_custody_disposition(typed_v5_modeled_boundaries, monkeypatch, spawn_uncertain):
    s = typed_v5_modeled_boundaries; c = s.c; c.process = None; c.identity = None
    c.spawn_attempted = False; c.engine_returned = False; c.spawn_returned_at = None; c.identity_observed_at = None
    called = []; held = []
    class ModeledEngineBoundary(RuntimeError):
        pass
    class ModeledUnknown(RuntimeError):
        pass
    def run_command(command, **kwargs):
        called.append((command, kwargs))
        c.spawn_attempted = spawn_uncertain  # Explicit modeled constructor uncertainty.
        raise ModeledEngineBoundary("modeled engine boundary; no child exists")
    def hold(error):
        held.append(error)
        assert c.execution_ref is not None and c.closed is False
        raise ModeledUnknown("modeled retained actual Source custody")
    c.hold = hold
    monkeypatch.setattr(owners, "_V5ValidationSourceCustody", lambda *args: c)
    monkeypatch.setattr(s.engine, "run_command", run_command)
    monkeypatch.setattr(s.engine.subprocess, "Popen", owners._ORIGINAL_POPEN)
    actual_descriptor = owners.descriptor
    monkeypatch.setattr(owners, "descriptor", lambda path: dict(c.engine_source)
        if str(Path(path).absolute()) == c.engine_source["path"] else actual_descriptor(path))
    with pytest.raises(ModeledUnknown if spawn_uncertain else ModeledEngineBoundary):
        owners.original_runner_wait(c.policy, c.argv, s.invocation, 360, validation_context=c.context)
    assert len(called) == 1 and called[0][1]["work_deadline_monotonic"] == 1330.0
    assert called[0][1]["cleanup_deadline_monotonic"] == 1340.0 and called[0][1]["timeout_seconds"] == 320.0
    saved = json.loads(Path(c.execution_ref["path"]).read_bytes())
    assert saved["execution_observation"] is None and saved["spawn_attempted"] is spawn_uncertain
    assert saved["authority_deadline_monotonic"] == 1360.0 and saved["actual_Popen_returned"] is False
    assert bool(held) is spawn_uncertain and c.closed is (not spawn_uncertain)
    assert ("source_exit" in s.events) is (not spawn_uncertain)


def test_v5_direct_timeout10_cannot_bypass_fixed_reserve_or_launch(typed_v5_modeled_boundaries, monkeypatch):
    s = typed_v5_modeled_boundaries; c = s.c; s.now["value"] = 1001.0
    c.process = None; c.identity = None; c.spawn_attempted = False; c.engine_returned = False
    c.spawn_returned_at = None; c.identity_observed_at = None
    def custody(policy, argv, cwd, timeout, context):
        c.deadline = c.started + timeout; c.work_deadline = c.deadline - 30.0; c.cleanup_deadline = c.deadline - 20.0
        return c
    monkeypatch.setattr(owners, "_V5ValidationSourceCustody", custody)
    monkeypatch.setattr(s.engine, "run_command", lambda *args, **kwargs: pytest.fail("engine reached despite impossible fixed reserve"))
    monkeypatch.setattr(s.engine.subprocess, "Popen", owners._ORIGINAL_POPEN)
    actual_descriptor = owners.descriptor
    monkeypatch.setattr(owners, "descriptor", lambda path: dict(c.engine_source)
        if str(Path(path).absolute()) == c.engine_source["path"] else actual_descriptor(path))
    with pytest.raises(ValueError, match="worker window expired before launch"):
        owners.original_runner_wait(c.policy, c.argv, s.invocation, 10, validation_context=c.context)
    assert c.deadline == 1010.0 and c.work_deadline == 980.0 and c.cleanup_deadline == 990.0
    assert c.spawn_attempted is False and c.process is None and c.closed is True
    assert s.events == ["source_exit"] and s.native_calls == []
    observed = json.loads(Path(c.execution_ref["path"]).read_bytes())
    assert observed["execution_observation"] is None and observed["actual_Popen_returned"] is False
    assert observed["storage_release_authorized"] is False and observed["retry_authorized"] is False


def test_v5_older_four_field_result_without_observation_property_persists_unknown(typed_v5_modeled_boundaries, monkeypatch):
    import dataclasses
    s = typed_v5_modeled_boundaries
    @dataclasses.dataclass(frozen=True)
    class ModeledOlderResult:
        command: tuple
        returncode: int
        stdout: str
        stderr: str
    monkeypatch.setattr(s.engine, "CommandResult", ModeledOlderResult)  # Explicitly modeled old ABI shape.
    completed = ModeledOlderResult(tuple(s.c.argv), 1, owners.canonical(s.terminal).decode(), "child text")
    s.c.record_execution(completed, s.engine, [(s.c.process, s.c.identity)])
    witness = {"runner_identity": s.c.identity, "actual_Popen_returned": True,
        "actual_process_waited": None, "actual_process_returncode": 1, "engine_source": s.c.engine_source}
    with pytest.raises(ValueError, match="typed immutable original V5 execution observation"):
        s.c.finish(completed, witness)
    saved = json.loads(Path(s.c.execution_ref["path"]).read_bytes())
    assert saved["exact_command_result"] is True and saved["exact_execution_observation"] is False
    assert saved["execution_observation"] is None and saved["actual_process_returncode"] == 1
    assert s.c.closed is False and s.native_calls == [] and "source_exit" not in s.events


# Reviewed native admitted-parent controls. These authored boundaries do not
# assert real Source/SDK/Native admission. The one actual child route below is
# separately environment-gated and must run only under the original owner.
def reviewed_parent_unissued_load_v1(value):
    return owners.load_parent_verified_reviewed_preservation_v1(
        value, (), None, {}, {},
        expected_receipt_sha256="a" * 64, expected_packet_sha256="b" * 64,
        expected_binary_sha256={}, expected_preservation_sha256="c" * 64,
        expected_graph_sha256="d" * 64, expected_admission_sha256="e" * 64,
        toolchain="leanprover/lean4:v4.34.1")


@pytest.mark.parametrize("offered", ["none", "object", "dict", "json", "unissued"])
def test_reviewed_parent_success_rejects_unissued_before_native_owners(monkeypatch, offered):
    # The None selector is the intended missing-API RED. All other values also
    # fail before imports, preparation reads, receipt parsing or native work.
    if offered == "none": value = None
    elif offered == "object": value = object()
    elif offered == "dict": value = {"original_success_authenticated": True}
    elif offered == "json": value = json.loads('{"original_admitted_parent_custody_authenticated":true}')
    else: value = object.__new__(owners.VerifiedReviewedPreservationParentSuccess)
    def forbidden(*args, **kwargs):
        pytest.fail("native owner resolution preceded invalid parent handle refusal")
    monkeypatch.setattr(owners, "_parent_native_owner", forbidden, raising=False)
    with pytest.raises(ValueError, match="(?i)(issued|handle)"):
        reviewed_parent_unissued_load_v1(value)


def test_reviewed_parent_success_constructor_copy_and_json_do_not_mint():
    import pickle
    cls = owners.VerifiedReviewedPreservationParentSuccess
    with pytest.raises(TypeError): cls()
    with pytest.raises(TypeError): type("ForgedParentSuccess", (cls,), {})
    offered = object.__new__(cls)
    for copier in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError): copier(offered)
    with pytest.raises(ValueError, match="(?i)(issued|handle)"):
        reviewed_parent_unissued_load_v1(offered)


def test_reviewed_parent_profile_keeps_existing_typed_success_terminal_admitted_false(
        typed_v5_modeled_boundaries):
    # Existing typed fixture models Source and durability. It establishes the
    # admitted=False terminal vocabulary only, never parent capability custody.
    s = typed_v5_modeled_boundaries
    s.terminal["admitted"] = False
    completed = s.result(public=0, actual=0)
    result = s.c.finish(completed, s.record(completed))
    assert s.verified == [s.terminal]
    assert s.terminal["admitted"] is False
    assert result["closure_kind"] == "existing_verified_success"
    assert result["source_closed"] is result["native_closed"] is True
    assert result["storage_release_authorized"] is result["retry_authorized"] is False
    assert result.get("original_success_authenticated", False) is False
    assert s.native_calls == [] and s.events[-1] == "source_exit"


@pytest.fixture
def reviewed_parent_packet_join_scope(tmp_path):
    """77 tiny modeled byte rows; no Lean, original success or parent mint."""
    attempt = tmp_path / "modeled-attempt"; attempt.mkdir()
    packet_root = "exports/reviewed-preservation-native"
    relative = ["build-receipt.json", "stdout.txt", "stderr.txt", "runner-error.txt"]
    relative += ["project/corpus-manifest.json"]
    relative += ["project/Module%02d.lean" % i for i in range(33)]
    relative += ["executed-inputs/Module%02d.lean" % i for i in range(33)]
    relative += ["compiled/Module%02d.olean" % i for i in range(6)]
    assert len(relative) == len(set(relative)) == 77
    inventory = []; durable = []; packet = []
    for name in sorted(relative):
        raw = owners.canonical({"fixture_only": True, "path": name}) + b"\n"
        ref = put(attempt / packet_root / name, raw)
        info = Path(ref["path"]).stat()
        packet.append((name, raw))
        inventory.append({"path": name, "bytes": ref["bytes"], "sha256": ref["sha256"]})
        durable.append({"path": packet_root + "/" + name, "bytes": ref["bytes"],
                        "sha256": ref["sha256"], "device": info.st_dev, "inode": info.st_ino})
    data = [put(tmp_path / "modeled-native-data" / name, b"Fixture only; not a native executable.\n")
            for name in ("prlimit", "lake", "lean", "libLake_shared.so", "libInit_shared.so",
                         "libleanshared_1.so", "libleanshared_2.so", "libleanshared.so")]
    binaries = {row["path"]: row["sha256"] for row in data if Path(row["path"]).name in ("lake", "lean")}
    native_argv = [[data[0]["path"], "--core=0:0", "--fsize=16777216:16777216", "--", row["path"], "--version"]
                   for row in data[1:3]]
    native_argv += [[data[0]["path"], "--core=0:0", "--fsize=134217728:134217728", "--cpu=60:60", "--",
                     data[1]["path"], "--offline", "--no-cache", "build", "legal"]]
    source = put(tmp_path / "modeled-source.py", b"# Modeled metadata only; never imported.\n")
    operation = {"operation_id": "modeled-reviewed-preservation-join-A", "packet_root": packet_root,
        "nodeid": str(Path(__file__).absolute()) + "::modeled_not_a_native_operation",
        "expected_source_sha256": {source["path"]: source["sha256"]},
        "expected_graph_sha256": "a" * 64, "expected_admission_sha256": "b" * 64,
        "expected_preservation_sha256": "c" * 64, "toolchain": "leanprover/lean4:v4.34.1",
        "expected_binary_sha256": binaries, "native_argv": native_argv, "native_data": data,
        "native_inputs": put(tmp_path / "modeled-native-inputs.json", {"fixture_only": True, "not_an_original_native_input": True})}
    binding = {"schema": "reviewed-preservation-native-binding/v1", "operation_id": operation["operation_id"],
        "packet_root": packet_root, "receipt_sha256": next(r["sha256"] for r in inventory if r["path"] == "build-receipt.json"),
        "packet_sha256": owners.digest(inventory), "preparation_sha256": operation["expected_preservation_sha256"],
        "graph_sha256": operation["expected_graph_sha256"], "admission_sha256": operation["expected_admission_sha256"],
        "toolchain": operation["toolchain"], "binary_sha256": copy.deepcopy(binaries),
        "producer_pins": dict(operation["expected_source_sha256"]), "execution_projection_sha256": "d" * 64,
        "packet_inventory": inventory, "native_argv": copy.deepcopy(native_argv)}
    return SimpleNamespace(attempt=attempt, operation=operation, binding=binding, durable=durable,
                           packet=tuple(packet), source=source)


def test_reviewed_parent_packet_join_matches_all77_modeled_rows_without_native_authority(
        reviewed_parent_packet_join_scope):
    s = reviewed_parent_packet_join_scope
    value = owners._reviewed_parent_packet_binding_v1(s.operation, s.binding, s.durable,
                                                     attempt_directory=s.attempt)
    assert type(value) is tuple and len(value) == 77
    expected = tuple((row["path"], row["bytes"], row["sha256"], row["device"], row["inode"])
                     for row in sorted(s.durable, key=lambda row: row["path"]))
    assert value == expected
    # A pure metadata join has neither a handle nor a success/authentication bit.
    assert all(type(row) is tuple and len(row) == 5 for row in value)


@pytest.mark.parametrize("fault", ["operation", "root", "graph", "admission", "preparation", "toolchain",
    "binary", "producer", "argv", "receipt", "packet_pin", "missing_row", "extra_row", "duplicate_row",
    "missing_packet", "extra_packet", "device_bool", "inode_bool"])
def test_reviewed_parent_packet_join_refuses_crossed_or_incomplete_metadata(
        reviewed_parent_packet_join_scope, fault):
    s = reviewed_parent_packet_join_scope
    binding = copy.deepcopy(s.binding); durable = copy.deepcopy(s.durable)
    if fault == "operation": binding["operation_id"] += "-B"
    elif fault == "root": binding["packet_root"] = "exports/other-packet"
    elif fault in ("graph", "admission", "preparation"): binding[fault + "_sha256"] = "e" * 64
    elif fault == "toolchain": binding["toolchain"] = "leanprover/lean4:v4.35.0"
    elif fault == "binary": binding["binary_sha256"][next(iter(binding["binary_sha256"]))] = "e" * 64
    elif fault == "producer": binding["producer_pins"][s.source["path"]] = "e" * 64
    elif fault == "argv": binding["native_argv"][-1][-1] = "another-target"
    elif fault == "receipt": binding["receipt_sha256"] = "e" * 64
    elif fault == "packet_pin": binding["packet_sha256"] = "e" * 64
    elif fault == "missing_row": durable.pop()
    elif fault == "extra_row": durable.append(dict(durable[-1], path=s.operation["packet_root"] + "/unexpected.bin"))
    elif fault == "duplicate_row": durable.append(dict(durable[-1]))
    elif fault == "missing_packet": binding["packet_inventory"].pop()
    elif fault == "extra_packet": binding["packet_inventory"].append({"path": "unexpected.bin", "bytes": 1, "sha256": "e" * 64})
    elif fault == "device_bool": durable[0]["device"] = True
    elif fault == "inode_bool": durable[0]["inode"] = True
    with pytest.raises(ValueError):
        owners._reviewed_parent_packet_binding_v1(s.operation, binding, durable, attempt_directory=s.attempt)


@pytest.fixture
def reviewed_parent_modeled_durability_scope(tmp_path):
    """Real local artifact reads, deliberately modeled admission/Source facts."""
    attempt = tmp_path / "modeled-verified-attempt"; attempt.mkdir()
    invocation = tmp_path / "modeled-invocation"; invocation.mkdir()
    source = put(tmp_path / "modeled-body.py", b"# Never executed; modeled verifier fixture.\n")
    engine = put(tmp_path / "modeled-engine.py", b"# Never executed; not a real engine.\n")
    node = source["path"] + "::modeled_case"
    request = {"schema": "owned-validation-request/v5", "source_sha256": {source["path"]: source["sha256"]}}
    request_pin = put(invocation / "execution-request.json", request)
    plan = {"schema": "modeled-pure-verifier-plan/v1", "attempt_directory": str(attempt),
        "source_pins": request["source_sha256"], "required_candidate_bodies": [source["path"]],
        "selected_checks": [node], "expected_cases": 1}
    plan_pin = put(invocation / "validation-plan.json", plan)
    result = put(attempt / "logs/result.json", {"schema": "owned-validation-result/v1", "completed": True,
        "returncode": 0, "request_sha256": request_pin["sha256"]})
    row = {"status": "released", "artifacts_durable_asserted": True, "native_lease_id": "modeled-native",
        "validation_request_sha256": request_pin["sha256"], "owner_pid": 4101, "owner_birth": "12345",
        "reservation_id": "modeled-only-reservation"}
    terminal = {"admitted": False, "result": result["path"], "reservation": {"status": "released", "record": row,
        "resource_lease": {"released": True, "lease_id": "modeled-native", "owner_pid": 4101},
        "cleanup_error": None, "process_observation": {"live_processes": 0,
            "out_of_group_live_identities": [], "tracking_error": ""}}}
    put(invocation / "runner-wait.json", {"request": request_pin, "original_process_owner_returned": True,
        "returncode": 0, "stdout": owners.canonical(terminal).decode(), "runner_identity": {"pid": 4101, "birth": "12345"},
        "actual_Popen_returned": True, "actual_process_waited": True, "actual_process_returncode": 0,
        "engine_source": engine})
    policy = {"schema": "explicit-modeled-verifier-only", "python": sys.executable,
              "validation": {"pytest_module": pin(Path(pytest.__file__).absolute())}}
    put(attempt / "exports/trusted-validation.json", {"schema": owners.WORK_SCHEMA, "passed": True,
        "source_READ_closed": True, "source_snapshot_stable": True, "cleanup_errors": [], "error": None,
        "junit_error": None, "guard_denials": [], "plan_sha256": plan_pin["sha256"],
        "full_source_manifest_sha256": owners.digest(plan["source_pins"]), "full_source_file_count": 1,
        "pytest_module_origin": policy["validation"]["pytest_module"],
        "startup": {"python": sys.executable, "isolated": 1, "no_bytecode": True},
        "executed_source_buffers": plan["source_pins"],
        "phases": [{"nodeid": node, "when": phase, "outcome": "passed"} for phase in ("setup", "call", "teardown")]})
    put(attempt / "exports/test-results.xml", b'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0" /></testsuites>')
    payload = put(attempt / "exports/modeled-payload.bin", b"Original exact modeled artifact bytes.\n")
    info = Path(payload["path"]).stat()
    durability = {"schema": "owned-validation-durability/v1", "request_sha256": request_pin["sha256"],
        "reservation_id": row["reservation_id"], "result_sha256": result["sha256"],
        "files": [{"path": "exports/modeled-payload.bin", "bytes": payload["bytes"],
                   "sha256": payload["sha256"], "device": info.st_dev, "inode": info.st_ino}]}
    put(attempt / "logs/validation-durability.json", durability)
    return SimpleNamespace(policy=policy, plan=plan, request_pin=request_pin, terminal=terminal,
        invocation=invocation, attempt=attempt, payload=payload, durability=durability)


def test_reviewed_parent_original_verifier_accepts_false_admitted_without_minting(
        reviewed_parent_modeled_durability_scope):
    s = reviewed_parent_modeled_durability_scope
    value = owners.verify_validation_evidence(s.policy, s.plan, s.request_pin, s.terminal, s.invocation)
    assert s.terminal["admitted"] is False
    assert value["cases"] == 1 and value["natural_phase_counter_verified"] is True
    assert "original_success_authenticated" not in value


@pytest.mark.parametrize("fault", ["inode", "device", "content"])
def test_reviewed_parent_original_durability_join_rejects_changed_physical_artifact(
        reviewed_parent_modeled_durability_scope, fault):
    s = reviewed_parent_modeled_durability_scope
    if fault == "content":
        path = Path(s.payload["path"]); path.chmod(0o600); path.write_bytes(b"Changed exact modeled artifact bytes.\n"); path.chmod(0o400)
    else:
        changed = copy.deepcopy(s.durability)
        changed["files"][0][fault] += 1
        path = s.attempt / "logs/validation-durability.json"; path.chmod(0o600)
        path.write_bytes(owners.canonical(changed) + b"\n"); path.chmod(0o400)
    with pytest.raises(ValueError, match="durable artifact changed"):
        owners.verify_validation_evidence(s.policy, s.plan, s.request_pin, s.terminal, s.invocation)


@pytest.fixture
def reviewed_parent_worker_declaration_scope(reviewed_parent_packet_join_scope):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    s = reviewed_parent_packet_join_scope
    operation = copy.deepcopy(s.operation)
    binary_root = s.attempt.parent / "modeled-toolchain/bin"
    lake, lean = str(binary_root / "lake"), str(binary_root / "lean")
    binaries = {lake: "a" * 64, lean: "b" * 64}
    roster = ["/usr/bin/prlimit", lake, lean] + [str(binary_root.parent / "lib/lean" / name) for name in (
        "libLake_shared.so", "libInit_shared.so", "libleanshared_1.so", "libleanshared_2.so", "libleanshared.so")]
    operation["expected_binary_sha256"] = binaries
    operation["native_data"] = [{"path": path, "bytes": 1, "sha256": binaries.get(path, "c" * 64)} for path in roster]
    prefix = ["/usr/bin/prlimit", "--core=0:0"]
    operation["native_argv"] = [prefix + ["--fsize=16777216:16777216", "--", lake, "--version"],
        prefix + ["--fsize=16777216:16777216", "--", lean, "--version"],
        prefix + ["--fsize=134217728:134217728", "--cpu=60:60", "--", lake, "--offline", "--no-cache", "build", "legal"]]
    root = str(s.attempt.parent / "modeled-A")
    validation = {"ordinary_scratch_profile": {"schema": "owned-validation-ordinary-scratch-live/v1"},
                  "cpu_slots": 1, "memory_mb": 1536, "storage_bytes": 16777216,
                  "timeout_seconds": 300, "child_process_slots": 4}
    policy = {"schema": "reviewed-preservation-parent/v1", "source_root": root,
              "datasets_root": str(s.attempt.parent / "modeled-D"), "operation": operation,
              "validation": validation}
    plan = {"schema": "reviewed-preservation-native-worker/v1", "operation": copy.deepcopy(operation),
        "selected_checks": [operation["nodeid"]], "expected_cases": 1,
        "candidate_root": root, "collection_root": root,
        "ordinary_scratch_profile": copy.deepcopy(validation["ordinary_scratch_profile"]),
        "attempt_directory": str(s.attempt), "parent_source_observation_path": str(s.attempt.parent / "modeled-source-observation.json"),
        "execution_request_path": str(s.attempt.parent / "modeled-request.json"),
        "parent_nonce_sha256": "d" * 64, "parent_identity": {"pid": 4100, "birth": "12345"},
        "native_inputs": dict(operation["native_inputs"]),
        "original_public_control_pins": {operation["native_inputs"]["path"]: operation["native_inputs"]["sha256"]},
        "source_pins": dict(operation["expected_source_sha256"])}
    return SimpleNamespace(worker=worker, policy=policy, plan=plan)


def test_reviewed_parent_worker_declaration_is_detached_metadata_only(reviewed_parent_worker_declaration_scope):
    s = reviewed_parent_worker_declaration_scope
    before = copy.deepcopy((s.plan, s.policy))
    observed = s.worker.reviewed_preservation_native_operation_v1(s.plan, s.policy)
    assert observed == s.policy["operation"] and observed is not s.plan["operation"]
    observed["native_argv"].clear()
    assert (s.plan, s.policy) == before
    assert not Path(s.plan["attempt_directory"], "exports/reviewed-preservation-native-binding.json").exists()


@pytest.mark.parametrize("fault", ["crossed_operation", "escape_root", "foreign_collection", "multiple_nodes",
    "bad_nonce", "boolean_parent_pid", "bad_toolchain", "changed_command", "extra_command", "source_pin",
    "missing_library", "extra_library", "duplicate_library", "boolean_bytes", "overbound_data", "total_data",
    "child_budget", "time_budget", "legacy_scratch", "foreign_profile"])
def test_reviewed_parent_worker_rejects_unselected_operation_before_any_native_import(
        reviewed_parent_worker_declaration_scope, fault):
    s = reviewed_parent_worker_declaration_scope
    plan, policy = copy.deepcopy(s.plan), copy.deepcopy(s.policy)
    op = policy["operation"]
    if fault == "crossed_operation": plan["operation"]["operation_id"] += "-other"
    elif fault == "escape_root": op["packet_root"] = "exports/../foreign"
    elif fault == "foreign_collection": plan["collection_root"] = policy["datasets_root"]
    elif fault == "multiple_nodes": plan["selected_checks"].append(op["nodeid"] + "-other")
    elif fault == "bad_nonce": plan["parent_nonce_sha256"] = "not-a-nonce-digest"
    elif fault == "boolean_parent_pid": plan["parent_identity"]["pid"] = True
    elif fault == "bad_toolchain": op["toolchain"] = "leanprover/lean4:latest"
    elif fault == "changed_command": op["native_argv"][-1][-1] = "another-target"
    elif fault == "extra_command": op["native_argv"].append(["/bin/true"])
    elif fault == "source_pin": op["expected_source_sha256"][next(iter(op["expected_source_sha256"]))] = "e" * 64
    elif fault == "missing_library": op["native_data"].pop()
    elif fault == "extra_library": op["native_data"].append({"path": "/foreign/lib.so", "bytes": 1, "sha256": "e" * 64})
    elif fault == "duplicate_library": op["native_data"][-1] = copy.deepcopy(op["native_data"][-2])
    elif fault == "boolean_bytes": op["native_data"][-1]["bytes"] = True
    elif fault == "overbound_data": op["native_data"][-1]["bytes"] = 512 * 1024**2 + 1
    elif fault == "total_data":
        for row in op["native_data"]: row["bytes"] = 128 * 1024**2
    elif fault == "child_budget": policy["validation"]["child_process_slots"] = 5
    elif fault == "time_budget": policy["validation"]["timeout_seconds"] = 301
    elif fault == "legacy_scratch": policy["validation"]["ordinary_scratch_profile"] = {"schema": "not-the-ordinary-profile"}
    elif fault == "foreign_profile": policy["schema"] = owners.POLICY_SCHEMA
    if fault != "crossed_operation": plan["operation"] = copy.deepcopy(op)
    with pytest.raises(ValueError):
        s.worker.reviewed_preservation_native_operation_v1(plan, policy)


def reviewed_parent_forbidden_callable_v1(*args, **kwargs):
    raise AssertionError("changed parent custody source callback was invoked")


@pytest.mark.parametrize("target", ["original_runner_wait", "verify_validation_evidence",
    "_reviewed_parent_packet_binding_v1", "_reviewed_parent_proc_stat", "_reviewed_parent_fd_observation", "_parent_native_owner"])
@pytest.mark.parametrize("mutation", ["identity", "code"])
def test_reviewed_parent_source_critical_callable_changes_refuse_before_mint(monkeypatch, target, mutation):
    function = getattr(owners, target)
    if mutation == "identity": monkeypatch.setattr(owners, target, reviewed_parent_forbidden_callable_v1)
    else: monkeypatch.setattr(function, "__code__", reviewed_parent_forbidden_callable_v1.__code__)
    with pytest.raises(ValueError, match="(?i)(changed|binding|original|canonical)"):
        owners._reviewed_parent_guard()


@pytest.fixture
def reviewed_parent_source_delegation_scope(tmp_path, monkeypatch):
    """Real tiny lock paths with modeled /proc facts; no FLOCK or Source call."""
    import os
    roots = [tmp_path / "modeled-D", tmp_path / "modeled-A"]
    receipts = []; rows = []; fd_observations = {}
    parent = {"pid": 4100, "birth": "12345"}
    for index, root in enumerate(roots):
        lock = root / "workspace/control-plane/source-leases/source.lock"
        put(lock, b""); lock.chmod(0o600)
        info = lock.stat(); fd = 30 + index
        row = {"root": str(root), "lock_path": str(lock), "fd": fd,
               "device": info.st_dev, "inode": info.st_ino}
        rows.append(row)
        receipts.append({"canonical_root": str(root), "mode": "read", "lock_path": str(lock),
            "waited_seconds": 0.25, "schema_version": "autoencoder-source-lease/v1", "cooperative": True})
        fd_observations[fd] = {"device": info.st_dev, "inode": info.st_ino, "lock_path": str(lock),
            "fdinfo": "flags:\t0100002\nlock:\t1: FLOCK  ADVISORY  READ  4100 %x:%x:%d 0 EOF\n" %
                      (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)}
    request = put(tmp_path / "modeled-request.json", {"fixture_only": True})
    argv = [sys.executable, "-I", "-B", "/modeled/run_owned_validation.py", "--request", request["path"],
            "--expected-request-sha256", request["sha256"], "--resource-ledger", str(tmp_path / "never-opened.json")]
    nonce = "4f" * 32
    observation = {"schema": "reviewed-preservation-parent-source-observation/v1", "request": request,
        "parent_identity": parent, "nonce": nonce, "source_receipts": receipts, "source_fds": rows, "sdk_argv": argv}
    facts = {4100: {"pid": 4100, "ppid": 1, "birth": "12345"},
             4101: {"pid": 4101, "ppid": 4100, "birth": "12346"},
             os.getpid(): {"pid": os.getpid(), "ppid": 4101, "birth": "12347"}}
    state = {"facts": facts, "fd": fd_observations, "executable": Path(sys.executable).resolve(strict=True),
             "cmdline": b"\x00".join(os.fsencode(word) for word in argv) + b"\x00"}
    calls = []
    def proc(pid):
        calls.append(("stat", pid))
        if pid not in state["facts"]: raise OSError("modeled absent process")
        return dict(state["facts"][pid])
    def fd_observation(pid, fd):
        calls.append(("fd", pid, fd))
        assert pid == parent["pid"]
        if fd not in state["fd"]: raise OSError("modeled absent descriptor")
        return dict(state["fd"][fd])
    original_read = Path.read_bytes
    def read(path):
        if str(path) == "/proc/4101/cmdline": return state["cmdline"]
        return original_read(path)
    monkeypatch.setattr(owners, "_reviewed_parent_proc_stat", proc)
    monkeypatch.setattr(owners, "_reviewed_parent_fd_observation", fd_observation)
    monkeypatch.setattr(owners, "_reviewed_parent_proc_executable", lambda pid: state["executable"])
    monkeypatch.setattr(Path, "read_bytes", read)
    def forbidden(*args, **kwargs): pytest.fail("Source admission occurred inside delegated read-only proof")
    monkeypatch.setattr(owners, "public_source_lease", forbidden)
    arguments = {"expected_request_sha256": request["sha256"], "expected_parent_identity": dict(parent),
        "expected_nonce_sha256": hashlib.sha256(bytes.fromhex(nonce)).hexdigest(),
        "expected_sdk_argv": list(argv), "expected_roots": roots}
    return SimpleNamespace(value=observation, args=arguments, state=state, calls=calls, roots=roots)


def test_reviewed_parent_source_delegation_checks_modeled_fd_roster_without_acquiring(
        reviewed_parent_source_delegation_scope):
    s = reviewed_parent_source_delegation_scope
    raw = owners.canonical(s.value)
    result = owners.verify_reviewed_preservation_parent_source_observation_v1(raw, **s.args)
    assert result["observation_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["request_sha256"] == s.args["expected_request_sha256"]
    assert result["source_fds"] == s.value["source_fds"]
    assert {call[2] for call in s.calls if call[0] == "fd"} == {30, 31}
    assert "source_closed" not in result and "original_success_authenticated" not in result


@pytest.mark.parametrize("fault", ["fake_schema", "nonce", "request", "parent_pid", "parent_birth",
    "sdk_parent", "sdk_command", "sdk_executable", "missing_fd", "fd_inode", "row_inode", "fd_path", "write_lock",
    "unlocked", "wrong_lock_pid", "unrelated_lock", "write_receipt", "wrong_root",
    "boolean_wait", "string_wait", "negative_wait"])
def test_reviewed_parent_source_delegation_refuses_forged_or_unheld_handoff(
        reviewed_parent_source_delegation_scope, tmp_path, fault):
    import os
    s = reviewed_parent_source_delegation_scope; value = copy.deepcopy(s.value)
    if fault == "fake_schema": value["schema"] = "caller-source-closed-booleans/v1"
    elif fault == "nonce": value["nonce"] = "00" * 32
    elif fault == "request": value["request"]["sha256"] = "0" * 64
    elif fault == "parent_pid": value["parent_identity"]["pid"] += 1
    elif fault == "parent_birth": s.state["facts"][4100]["birth"] = "99999"
    elif fault == "sdk_parent": s.state["facts"][4101]["ppid"] = 4999
    elif fault == "sdk_command": s.state["cmdline"] = b"/modeled/foreign-cli\x00"
    elif fault == "sdk_executable": s.state["executable"] = tmp_path / "foreign-python"
    elif fault == "missing_fd": value["source_fds"][0]["fd"] = 999
    elif fault == "fd_inode": s.state["fd"][30]["inode"] += 1
    elif fault == "row_inode": value["source_fds"][0]["inode"] += 1
    elif fault == "fd_path": s.state["fd"][30]["lock_path"] += ".foreign"
    elif fault == "write_lock": s.state["fd"][30]["fdinfo"] = s.state["fd"][30]["fdinfo"].replace("READ", "WRITE")
    elif fault == "unlocked": s.state["fd"][30]["fdinfo"] = "flags:\t0100002\n"
    elif fault == "wrong_lock_pid": s.state["fd"][30]["fdinfo"] = s.state["fd"][30]["fdinfo"].replace("READ  4100", "READ  4999")
    elif fault == "unrelated_lock":
        lock = s.roots[0] / "workspace/unrelated.lock"; put(lock, b""); info = lock.stat()
        value["source_fds"][0].update(lock_path=str(lock), device=info.st_dev, inode=info.st_ino)
        value["source_receipts"][0]["lock_path"] = str(lock)
        s.state["fd"][30] = {"device": info.st_dev, "inode": info.st_ino, "lock_path": str(lock),
            "fdinfo": "lock: 1: FLOCK ADVISORY READ 4100 %x:%x:%d 0 EOF\n" %
                      (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)}
    elif fault == "write_receipt": value["source_receipts"][0]["mode"] = "write"
    elif fault == "wrong_root": value["source_receipts"][0]["canonical_root"] = str(tmp_path / "foreign-root")
    elif fault == "boolean_wait": value["source_receipts"][0]["waited_seconds"] = True
    elif fault == "string_wait": value["source_receipts"][0]["waited_seconds"] = "0.25"
    elif fault == "negative_wait": value["source_receipts"][0]["waited_seconds"] = -0.25
    with pytest.raises((ValueError, OSError)):
        owners.verify_reviewed_preservation_parent_source_observation_v1(owners.canonical(value), **s.args)


def reviewed_parent_original_R_fixture_module_v1(datasets_root):
    """Reuse the original R fixture at its honest physical source filename."""
    import importlib.util
    path = Path(datasets_root) / "tests/unit/logic/autoformal/test_legal_corpus_definition_use_native_v2.py"
    spec = importlib.util.spec_from_file_location("_reviewed_parent_original_R_native_fixture_v1", path)
    assert spec is not None and spec.origin == str(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__file__ == str(path)
    return module


def reviewed_parent_fixture_tree_encode_v1(value):
    """Closed ordinary DATA tree. No signer, private key or Python object codec."""
    import base64
    import math
    denied = {"private_key", "private_key_bytes", "secret_key", "signing_key", "private_bytes", "signers"}
    if type(value) is dict:
        assert all(type(key) is str and key not in denied for key in value)
        return {"kind": "dict", "items": [[key, reviewed_parent_fixture_tree_encode_v1(value[key])] for key in sorted(value)]}
    if type(value) in (list, tuple):
        return {"kind": "list" if type(value) is list else "tuple", "items": [reviewed_parent_fixture_tree_encode_v1(row) for row in value]}
    if type(value) is bytes:
        return {"kind": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    assert value is None or type(value) in (str, bool, int) or (type(value) is float and math.isfinite(value))
    return {"kind": "atom", "value": value}


def reviewed_parent_fixture_tree_decode_v1(value, *, _budget=None, _depth=0):
    import base64
    import math
    if _budget is None: _budget = [2_000_000]
    _budget[0] -= 1
    assert _budget[0] >= 0 and _depth <= 64 and type(value) is dict
    kind = value.get("kind")
    def child(row): return reviewed_parent_fixture_tree_decode_v1(row, _budget=_budget, _depth=_depth + 1)
    if kind == "dict":
        assert set(value) == {"kind", "items"} and type(value["items"]) is list
        result = {}
        for row in value["items"]:
            assert type(row) is list and len(row) == 2 and type(row[0]) is str and row[0] not in result
            assert row[0] not in {"private_key", "private_key_bytes", "secret_key", "signing_key", "private_bytes", "signers"}
            result[row[0]] = child(row[1])
        return result
    if kind in ("list", "tuple"):
        assert set(value) == {"kind", "items"} and type(value["items"]) is list
        rows = [child(row) for row in value["items"]]
        return rows if kind == "list" else tuple(rows)
    if kind == "bytes":
        assert set(value) == {"kind", "base64"} and type(value["base64"]) is str
        return base64.b64decode(value["base64"], validate=True)
    assert kind == "atom" and set(value) == {"kind", "value"}
    atom = value["value"]
    assert atom is None or type(atom) in (str, bool, int) or (type(atom) is float and math.isfinite(atom))
    return atom


def prepare_reviewed_parent_native_fixture_inputs_v1(path, *, datasets_root=None):
    """Prepare fixed signed fictional inputs once; never execute a native tool.

    Private signing handles remain only inside the original fixture creator.
    This DATA transport supplies no handle, original execution or legal proof.
    Root selects and pins it before the future original parent request.
    """
    if datasets_root is None: datasets_root = Path(__file__).resolve().parents[3] / "ipfs_datasets"
    helper = reviewed_parent_original_R_fixture_module_v1(Path(datasets_root))
    control = helper.reviewed_native_fixture_v2()
    prepared = control["preparation"]
    arguments = helper.reviewed_native_arguments_v2(control)
    value = {"schema": "reviewed-preservation-fixture-native-inputs/v1", "fixture_only": True,
        "provenance_kind": "fictional_fixture", "preparation": {"manifest_json": prepared.manifest_json,
            "files": [[name, body] for name, body in prepared.files]},
        "graph": reviewed_parent_fixture_tree_encode_v1(control["graph"]),
        "admission": reviewed_parent_fixture_tree_encode_v1(control["admission"]),
        "args": reviewed_parent_fixture_tree_encode_v1(control["args"]),
        "kwargs": reviewed_parent_fixture_tree_encode_v1(control["kwargs"]),
        **{name: arguments[name] for name in ("expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain")}}
    raw = owners.canonical(value) + b"\n"
    assert len(raw) <= 4 * 1024**2
    reference = put(Path(path), raw)
    return {"native_inputs": reference, **{name: value[name] for name in
        ("expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain")}}


def test_reviewed_preservation_parent_actual_native_child():
    """One future original-parent child body, separately gated from UNIT cases."""
    import os
    import importlib
    names = ("IPFS_REVIEWED_PRESERVATION_PLAN", "IPFS_REVIEWED_PRESERVATION_ATTEMPT", "IPFS_REVIEWED_PRESERVATION_PACKET_ROOT")
    if os.environ.get("IPFS_DATASETS_RUN_NATIVE_LEAN") != "1" or not all(os.environ.get(name) for name in names):
        pytest.skip("explicit original-parent reviewed native child enrollment required")
    plan_path, attempt, packet_root = (Path(os.environ[name]) for name in names)
    plan = owners.json_object(owners.ordinary_bytes(plan_path, 4 * 1024**2))
    assert plan["schema"] == "reviewed-preservation-native-worker/v1"
    policy = owners.json_object(owners.checked_descriptor(plan["policy"]))
    operation = plan["operation"]
    assert operation == policy["operation"] and plan["native_inputs"] == operation["native_inputs"]
    assert operation["nodeid"] == str(Path(__file__).absolute()) + "::test_reviewed_preservation_parent_actual_native_child"
    assert attempt == Path(plan["attempt_directory"]) and packet_root == attempt / operation["packet_root"]
    assert not packet_root.exists()
    helper = reviewed_parent_original_R_fixture_module_v1(Path(policy["datasets_root"]))
    reference = operation["native_inputs"]
    raw = owners.ordinary_bytes(Path(reference["path"]), 4 * 1024**2)
    assert pin(Path(reference["path"])) == reference
    value = owners.json_object(raw)
    assert set(value) == {"schema", "fixture_only", "provenance_kind", "preparation", "graph", "admission", "args", "kwargs",
                          "expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain"}
    assert value["schema"] == "reviewed-preservation-fixture-native-inputs/v1" and value["fixture_only"] is True
    assert value["provenance_kind"] == "fictional_fixture" and set(value["preparation"]) == {"manifest_json", "files"}
    shared = importlib.import_module("ipfs_datasets_py.logic.autoformal.legal_corpus_shared_lake")
    prepared = shared.ReviewedPreservationCorpusPreparation(value["preparation"]["manifest_json"],
        tuple((name, body) for name, body in value["preparation"]["files"]))
    control = {"shared": shared, "preparation": prepared,
        "graph": reviewed_parent_fixture_tree_decode_v1(value["graph"]),
        "admission": reviewed_parent_fixture_tree_decode_v1(value["admission"]),
        "args": reviewed_parent_fixture_tree_decode_v1(value["args"]),
        "kwargs": reviewed_parent_fixture_tree_decode_v1(value["kwargs"]),
        "graph_sha": value["expected_graph_sha256"], "admission_sha": value["expected_admission_sha256"]}
    arguments = helper.reviewed_native_arguments_v2(control)
    for name in ("expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain"):
        assert arguments[name] == value[name] == operation[name]
    binaries = operation["expected_binary_sha256"]
    executable = next(path for path in binaries if Path(path).name == "lake")
    # Existing original R functions own every validation, compiler and reader.
    handle = helper.reviewed_live_success_build_v2(control, executable, packet_root)
    packet = helper.reviewed_live_success_packet_v2(packet_root)
    assert len(packet) == 77
    result = helper.reviewed_live_success_load_v2(control, handle, packet)
    assert result["native_execution_authenticated"] is result["live_native_success_authenticated"] is True
    assert result["original_success_authenticated"] is result["original_admitted_parent_custody_authenticated"] is False
    assert result["retained_packet_sha256"] == helper.native.reviewed_norm_preservation_packet_sha256_v2(packet)
    assert result["training_mask"] == result["evaluation_gold_mask"] == 0
    assert all(flag is False for flag in result["authority"].values())
    assert result["release_ready"] is result["Lean_admitted"] is False
    assert json.loads(dict(packet)["build-receipt.json"])["binary_sha256"] == binaries


@pytest.mark.parametrize("mutation", ["identity", "code"])
def test_reviewed_parent_sdk_executable_reader_drift_refuses_before_mint(monkeypatch, mutation):
    function = owners._reviewed_parent_proc_executable
    if mutation == "identity": monkeypatch.setattr(owners, "_reviewed_parent_proc_executable", reviewed_parent_forbidden_callable_v1)
    else: monkeypatch.setattr(function, "__code__", reviewed_parent_forbidden_callable_v1.__code__)
    with pytest.raises(ValueError, match="(?i)(changed|binding|original|canonical)"):
        owners._reviewed_parent_guard()


@pytest.mark.parametrize("offered", [None, {"closed": True, "source_closed": True, "native_closed": True}])
def test_reviewed_parent_private_issuer_refuses_nonoriginal_wait_frame(offered):
    with pytest.raises(ValueError, match="only original parent wait"):
        owners._reviewed_parent_issue(offered, None, {})


@pytest.mark.parametrize("timeout", [0, True, 419, 421, float("nan")])
def test_reviewed_parent_capture_keeps_exact420_window_before_context_or_source(tmp_path, timeout):
    # Minimal explicitly selected policy intentionally lacks every live
    # admission field: a wrong clock must stop before context/Source access.
    with pytest.raises(ValueError, match="reviewed parent420 profile"):
        owners.original_runner_wait({"schema": "reviewed-preservation-parent/v1"}, [], tmp_path,
                                    timeout, validation_context=None, capture_parent_success=True)



def reviewed_parent_noop_need_v1(condition, message):
    """An untrusted replacement predicate; never use it to make an assertion."""
    return None


@pytest.mark.parametrize("mutation", ["identity", "code"])
def test_reviewed_parent_predicate_drift_refuses_independently(monkeypatch, mutation):
    # A refusal from an unrelated package/class defect cannot satisfy this test.
    owners._reviewed_parent_guard()
    function = owners.need
    if mutation == "identity": monkeypatch.setattr(owners, "need", reviewed_parent_noop_need_v1)
    else: monkeypatch.setattr(function, "__code__", reviewed_parent_noop_need_v1.__code__)
    # The guard must check the predicate with its own unchanged raising branch.
    with pytest.raises(ValueError, match="^reviewed parent callable bindings changed: need$"):
        owners._reviewed_parent_guard()


def test_reviewed_parent_public_fixture_transport_prepares_and_reopens(tmp_path, monkeypatch, request):
    """Retain one fictional public DATA fixture under the original pytest owner.

    Signatures are created only by the existing authored fixture helper in this
    admitted preparation cohort. Reopening uses only the fixed public bytes;
    it creates no signer, capability, native process or legal admission.
    """
    import importlib
    datasets_root = Path(__file__).resolve().parents[3] / "ipfs_datasets"
    helper = reviewed_parent_original_R_fixture_module_v1(datasets_root)
    def forbidden(*args, **kwargs):
        pytest.fail("fixture transport preparation or reopening attempted a native tool")
    monkeypatch.setattr(helper.native, "_installed_toolchain", forbidden)
    monkeypatch.setattr(helper.process.BoundedToolRunner, "run", forbidden)
    path = tmp_path / "public-native-inputs.json"
    prepared_record = prepare_reviewed_parent_native_fixture_inputs_v1(path, datasets_root=datasets_root)
    reference = prepared_record["native_inputs"]
    assert reference == pin(path) and reference["bytes"] <= 4 * 1024**2
    value = owners.json_object(owners.checked_descriptor(reference))
    assert set(value) == {"schema", "fixture_only", "provenance_kind", "preparation", "graph", "admission", "args", "kwargs",
                          "expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain"}
    assert value["schema"] == "reviewed-preservation-fixture-native-inputs/v1"
    assert value["fixture_only"] is True and value["provenance_kind"] == "fictional_fixture"
    assert set(value["preparation"]) == {"manifest_json", "files"}
    public = {name: reviewed_parent_fixture_tree_decode_v1(value[name]) for name in ("graph", "admission", "args", "kwargs")}
    pending = [public, value["preparation"]]
    denied = {"private_key", "private_key_bytes", "secret_key", "signing_key", "private_bytes", "signers"}
    while pending:
        item = pending.pop()
        if type(item) is dict:
            assert not denied.intersection(item)
            pending.extend(item.values())
        elif type(item) in (tuple, list): pending.extend(item)
        else: assert item is None or type(item) in (str, bool, int, float, bytes)
    shared = importlib.import_module("ipfs_datasets_py.logic.autoformal.legal_corpus_shared_lake")
    graph_owner = importlib.import_module("ipfs_datasets_py.logic.autoformal.corpus_norm_definition_adapter")
    admission_owner = importlib.import_module("ipfs_datasets_py.logic.autoformal.legal_norm_content_adjudication_v1")
    preparation = shared.ReviewedPreservationCorpusPreparation(value["preparation"]["manifest_json"],
        tuple((name, body) for name, body in value["preparation"]["files"]))
    assert shared.reviewed_preservation_corpus_sha256(preparation) == prepared_record["expected_preservation_sha256"]
    assert graph_owner.norm_definition_graph_sha256(public["graph"]) == prepared_record["expected_graph_sha256"]
    assert admission_owner.norm_content_adjudication_sha256(public["admission"]) == prepared_record["expected_admission_sha256"]
    control = {"shared": shared, "preparation": preparation, **public,
        "graph_sha": value["expected_graph_sha256"], "admission_sha": value["expected_admission_sha256"]}
    arguments = helper.reviewed_native_arguments_v2(control)
    for name in ("expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain"):
        assert arguments[name] == value[name] == prepared_record[name]
    assert helper.reviewed_native_validate_v2(control) == preparation
    assert public["admission"]["fixture_only"] is True
    assert public["admission"]["provenance_kind"] == "fictional_fixture"
    assert public["admission"]["training_mask"] == public["admission"]["evaluation_gold_mask"] == 0
    assert sorted(child.name for child in tmp_path.iterdir()) == ["public-native-inputs.json"]
    assert pin(path) == reference
    # This descriptor is public DATA only. The retained tmp_path body and JUnit
    # property let the parent select it for a later, separately admitted request.
    request.node.user_properties.append(("reviewed_preservation_native_inputs", owners.canonical(reference).decode("utf-8")))



def test_reviewed_parent_unmodified_canonical_guard_accepts_actual_package_profile():
    """Exercise the real canonical package/class path without Source or minting."""
    owners._reviewed_parent_guard()


# Explicit Native-only normal namespace declaration v2. These private scratch
# directories and inert source files are metadata fixtures, never Native builds,
# original parent evidence, actual candidate code or statutory meaning.
@pytest.fixture
def reviewed_parent_normal_namespace_v2_scope(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    a = Path(__file__).resolve().parents[2]
    d = tmp_path / 'namespace-datasets'
    definitions = (
        ('ipfs_datasets_py.logic.backends', 'ipfs_datasets_py/logic/backends', 'process.py'),
        ('ipfs_datasets_py.logic.families', 'ipfs_datasets_py/logic/families', 'namespaces.py'),
        ('ipfs_datasets_py.logic.intent_ir.formalize', 'ipfs_datasets_py/logic/intent_ir/formalize', 'modal_projections.py'),
    )
    profiles, representatives = [], []
    for fullname, relative, leaf in definitions:
        directory = d / relative
        representative = directory / leaf
        put(representative, b'# Inert metadata-only namespace fixture.\n')
        info = directory.stat()
        profiles.append(dict(fullname=fullname, path=str(directory), device=info.st_dev,
            inode=info.st_ino, initializer_absent=True))
        representatives.append(representative)
    lifecycle = [d / 'scripts/ops/legal_ir/run_owned_validation.py',
        *(d / 'ipfs_datasets_py/optimizers/logic_theorem_optimizer' / name for name in (
            'autoencoder_source_lease.py', 'validation_run_resources.py', 'autoencoder_daemon_resources.py',
            'resource_scheduler.py', 'existing_resource_scheduler.py', 'proof_resource_safety.py'))]
    for path in lifecycle: put(path, b'# Inert metadata-only lifecycle declaration.\n')
    original_a = [Path(owners.__file__).resolve(),
        a / 'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py',
        a / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py']
    descriptors = [pin(path) for path in (*original_a, *lifecycle, *representatives)]
    v2 = [dict(profile, profile='reviewed-preservation-normal-namespaces/v2') for profile in profiles]
    validation = dict(trusted_tests=[pin(Path(__file__).resolve())],
        selected_checks=[str(Path(__file__).resolve()) + '::test_reviewed_preservation_parent_actual_native_child'],
        pytest_module=pin(Path(pytest.__file__).resolve()), import_sources=[],
        attempt_root=str(tmp_path / 'attempts'), cpu_slots=1, memory_mb=1536, storage_bytes=16777216,
        timeout_seconds=300, child_process_slots=4, scheduler_state_path=str(tmp_path / 'scheduler.json'),
        expected_scheduler_policy_sha256='f' * 64, roots=[str(tmp_path)],
        resource_ledger=str(tmp_path / 'ledger.json'),
        ordinary_scratch_profile={'schema': 'owned-validation-ordinary-scratch-live/v1'},
        normal_namespace_profiles=v2)
    policy = dict(schema='reviewed-preservation-parent/v1', datasets_root=str(d), source_root=str(a),
        python=sys.executable, source_descriptors=descriptors, validation=validation, operation={})
    return SimpleNamespace(worker=worker, a=a, d=d, root=tmp_path, roots=[a, d, a],
        profiles=profiles, v2=v2, representatives=representatives,
        pins={row['path']: row['sha256'] for row in descriptors}, policy=policy)


def _namespace_v2_policy_before_invalid_operation(scope, profiles, label, *, remove_source=None):
    policy = copy.deepcopy(scope.policy)
    policy['validation']['normal_namespace_profiles'] = copy.deepcopy(profiles)
    if remove_source is not None:
        policy['source_descriptors'] = [row for row in policy['source_descriptors'] if row['path'] != str(remove_source)]
    reference = put(scope.root / ('namespace-policy-' + label + '.json'), policy)
    return Path(reference['path']), reference['sha256']


def _namespace_v2_mutation(scope, fault):
    profiles, pins = copy.deepcopy(scope.v2), dict(scope.pins)
    if fault == 'missing': profiles.pop()
    elif fault == 'extra': profiles.append(copy.deepcopy(profiles[-1]))
    elif fault == 'duplicate': profiles[1]['fullname'] = profiles[0]['fullname']
    elif fault == 'unknown': profiles[1]['fullname'] = 'ipfs_datasets_py.logic.unknown_namespace'
    elif fault == 'marker': profiles[1]['profile'] = 'reviewed-preservation-normal-namespaces/v3'
    elif fault == 'mixed_keys': profiles[1].pop('profile')
    elif fault == 'extra_key': profiles[1]['extra'] = True
    elif fault == 'bool_device': profiles[1]['device'] = True
    elif fault == 'zero_inode': profiles[1]['inode'] = 0
    elif fault == 'wrong_inode': profiles[1]['inode'] += 1
    elif fault == 'foreign_path': profiles[1]['path'] = str(scope.root / 'foreign')
    elif fault == 'missing_source': pins.pop(str(scope.representatives[1]))
    else: raise AssertionError('unknown namespace v2 test mutation')
    return profiles, pins


@pytest.mark.parametrize('native', [False, True])
def test_reviewed_parent_namespace_v2_preserves_legacy_backends_worker(reviewed_parent_normal_namespace_v2_scope, native):
    s = reviewed_parent_normal_namespace_v2_scope
    guard = s.worker.PinnedSources(s.pins, s.roots, s.profiles[:1], reviewed_native_profile=native)
    assert guard.normal_namespace_profiles == {s.profiles[0]['fullname']: s.profiles[0]}


@pytest.mark.parametrize('profiles', [None, []])
def test_reviewed_parent_namespace_v2_preserves_empty_worker_profile(reviewed_parent_normal_namespace_v2_scope, profiles):
    s = reviewed_parent_normal_namespace_v2_scope
    guard = s.worker.PinnedSources(s.pins, s.roots, profiles)
    assert guard.normal_namespace_profiles == {}


def test_reviewed_parent_namespace_v2_accepts_exact_native_worker_declaration(reviewed_parent_normal_namespace_v2_scope):
    s = reviewed_parent_normal_namespace_v2_scope
    guard = s.worker.PinnedSources(s.pins, s.roots, s.v2, reviewed_native_profile=True)
    assert guard.normal_namespace_profiles == {row['fullname']: row for row in s.v2}
    assert guard.verify_normal_namespaces() == {}


@pytest.mark.parametrize('profile', ['marked_v2', 'unmarked_three'])
def test_reviewed_parent_namespace_v2_does_not_expand_ordinary_worker(reviewed_parent_normal_namespace_v2_scope, profile):
    s = reviewed_parent_normal_namespace_v2_scope
    with pytest.raises(ValueError):
        s.worker.PinnedSources(s.pins, s.roots, s.v2 if profile == 'marked_v2' else s.profiles,
            reviewed_native_profile=False)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'duplicate', 'unknown', 'marker', 'mixed_keys',
    'extra_key', 'bool_device', 'zero_inode', 'wrong_inode', 'foreign_path', 'missing_source'])
def test_reviewed_parent_namespace_v2_worker_refuses_changed_declaration(reviewed_parent_normal_namespace_v2_scope, fault):
    s = reviewed_parent_normal_namespace_v2_scope
    profiles, pins = _namespace_v2_mutation(s, fault)
    with pytest.raises((ValueError, OSError)):
        s.worker.PinnedSources(pins, s.roots, profiles, reviewed_native_profile=True)


@pytest.mark.parametrize('profile', ['legacy', 'marked_v2'])
def test_reviewed_parent_namespace_v2_policy_reaches_only_inert_operation_gate(reviewed_parent_normal_namespace_v2_scope, profile):
    s = reviewed_parent_normal_namespace_v2_scope
    path, digest = _namespace_v2_policy_before_invalid_operation(s, s.profiles[:1] if profile == 'legacy' else s.v2, profile)
    with pytest.raises(ValueError, match='^closed reviewed parent operation required$'):
        owners.checked_reviewed_preservation_parent_policy_v1(path, digest)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'duplicate', 'unknown', 'marker', 'mixed_keys',
    'extra_key', 'bool_device', 'zero_inode', 'wrong_inode', 'foreign_path', 'missing_source'])
def test_reviewed_parent_namespace_v2_policy_refuses_changed_declaration(reviewed_parent_normal_namespace_v2_scope, fault):
    s = reviewed_parent_normal_namespace_v2_scope
    profiles, _ = _namespace_v2_mutation(s, fault)
    path, digest = _namespace_v2_policy_before_invalid_operation(s, profiles, fault,
        remove_source=s.representatives[1] if fault == 'missing_source' else None)
    with pytest.raises((ValueError, OSError)) as caught:
        owners.checked_reviewed_preservation_parent_policy_v1(path, digest)
    assert str(caught.value) != 'closed reviewed parent operation required'


def test_reviewed_parent_namespace_v2_unmarked_expansion_stays_refused(reviewed_parent_normal_namespace_v2_scope):
    s = reviewed_parent_normal_namespace_v2_scope
    with pytest.raises(ValueError, match='finite explicit normal namespace profiles required'):
        s.worker.PinnedSources(s.pins, s.roots, s.profiles, reviewed_native_profile=True)
    path, digest = _namespace_v2_policy_before_invalid_operation(s, s.profiles, 'unmarked')
    with pytest.raises(ValueError, match='single exact existing normal backends namespace profile required'):
        owners.checked_reviewed_preservation_parent_policy_v1(path, digest)


@pytest.mark.parametrize('where', ['worker_open', 'owner_open', 'worker_closing'])
def test_reviewed_parent_namespace_v2_refuses_real_initializer_presence(reviewed_parent_normal_namespace_v2_scope, where):
    s = reviewed_parent_normal_namespace_v2_scope
    guard = (s.worker.PinnedSources(s.pins, s.roots, s.v2, reviewed_native_profile=True)
        if where == 'worker_closing' else None)
    put(Path(s.v2[1]['path']) / '__init__.py', b'# New initializer invalidates namespace declaration.\n')
    if where == 'owner_open':
        path, digest = _namespace_v2_policy_before_invalid_operation(s, s.v2, 'initializer')
        with pytest.raises(ValueError, match='initializer absence changed'):
            owners.checked_reviewed_preservation_parent_policy_v1(path, digest)
    else:
        with pytest.raises(ValueError, match='initializer absence changed'):
            guard.verify_normal_namespaces() if guard is not None else s.worker.PinnedSources(s.pins, s.roots, s.v2, reviewed_native_profile=True)


def test_reviewed_parent_namespace_v2_worker_authenticates_leaf_before_code(reviewed_parent_normal_namespace_v2_scope):
    s = reviewed_parent_normal_namespace_v2_scope
    pins = dict(s.pins); pins[str(s.representatives[1])] = '0' * 64
    guard = s.worker.PinnedSources(pins, s.roots, s.v2, reviewed_native_profile=True)
    with pytest.raises(ImportError, match='raw first-party source changed'):
        guard.find_spec('ipfs_datasets_py.logic.families.namespaces', [str(s.representatives[1].parent)])
    assert guard.executed == {} and guard.compiled_codes == {}


def test_reviewed_parent_namespace_v2_parent_plan_preserves_exact_marker_rows(reviewed_parent_normal_namespace_v2_scope):
    import os
    s = reviewed_parent_normal_namespace_v2_scope
    policy = copy.deepcopy(s.policy)
    data = put(s.root / 'public-inert-native-input-data.json', {'fixture_only': True})
    operation = dict(operation_id='namespace-plan-only', native_inputs=data,
        nodeid=str(Path(__file__).resolve()) + '::test_reviewed_preservation_parent_actual_native_child')
    policy['operation'] = operation
    policy_pin = put(s.root / 'namespace-plan-policy.json', policy)
    sources, controls = owners._reviewed_parent_plan_sources(policy, policy_pin)
    invocation = s.root / 'namespace-plan-invocation'; invocation.mkdir()
    nonce = '5' * 64
    plan = dict(schema='reviewed-preservation-native-worker/v1', policy=policy_pin,
        operation=operation, native_inputs=data, candidate_root=str(s.a), collection_root=str(s.a),
        attempt_directory=str(Path(policy['validation']['attempt_root']) / operation['operation_id']),
        source_pins=sources, original_public_control_pins=controls, selected_checks=[operation['nodeid']],
        expected_cases=1, required_candidate_bodies=[str(s.d / 'ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py'), str(Path(__file__).resolve())],
        ordinary_scratch_profile={'schema': 'owned-validation-ordinary-scratch-live/v1'},
        normal_namespace_profiles=copy.deepcopy(s.v2),
        parent_source_observation_path=str(invocation / 'parent-source-observation.json'),
        parent_nonce_sha256=hashlib.sha256(bytes.fromhex(nonce)).hexdigest(),
        parent_identity=owners.proc_identity(os.getpid()), execution_request_path=str(invocation / 'execution-request.json'))
    plan_pin = put(invocation / 'validation-plan.json', plan)
    request = dict(source_sha256={**controls, plan_pin['path']: plan_pin['sha256']},
        argv=[policy['python'], '-I', '-B', str(s.a / 'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py'),
              '--plan', plan_pin['path'], '--plan-sha256', plan_pin['sha256']])
    request_pin = put(invocation / 'execution-request.json', request)
    context = dict(plan=plan, request_pin=request_pin, invocation=invocation,
        frozen={'schema': 'reviewed-preservation-original-parent-context/v1', 'parent_nonce': nonce})
    owners._reviewed_parent_check_plan_v1(policy, context)
    assert plan['normal_namespace_profiles'] == policy['validation']['normal_namespace_profiles'] == s.v2
    changed = copy.deepcopy(context); changed['plan']['normal_namespace_profiles'][1]['profile'] = 'changed'
    with pytest.raises(ValueError, match='closed predeclared reviewed parent plan differs'):
        owners._reviewed_parent_check_plan_v1(policy, changed)


# Prospective canonical-A consumer integration controls. The engine/Popen wait
# is real in the process cases. Source contexts and current D Native/ledger
# inspection verdicts are explicit modeled dependencies, never live custody.
# The real D V5 predicate has a separate original-owner qualification packet.
_V5_ACQUIRED_UNKNOWN = {'verdict': 'requires_original_native_cleanup',
    'source_custody_closed': False, 'storage_release_authorized': False}
_V5_ACQUIRED_CLOSED = {'verdict': 'proven_original_no_child_native_closed',
    'source_custody_closed': True, 'storage_release_authorized': False}


def _v5_acquired_consumer_process_scope(scope, *, api='current', terminal_schema='owned-validation-failure-terminal/v1', outcome=None, reason='declared-local-current-inspection-DATA'):
    """Extend the existing modeled-source fixture, preserving actual engine wait."""
    cli_path = Path(next(row['path'] for row in scope.policy['source_descriptors']
                        if row['path'].endswith('/scripts/ops/legal_ir/run_owned_validation.py')))
    source = '''import json, os, sys, time
from pathlib import Path

def classify_owned_validation_source_custody(*args, **kwargs):
    return {"verdict":"requires_original_native_cleanup","source_custody_closed":False,"storage_release_authorized":False}

def inspect_owned_validation_no_child_native_closed_v1(*args, **kwargs):
    raise AssertionError("V5 consumer must not dispatch to legacy V1")

def inspect_owned_validation_productive_native_closed_v1(*args, **kwargs):
    raise AssertionError("acquired pre-child failure must not enter productive path")

def inspect_owned_validation_no_child_native_closed_v5(terminal, request_raw, *, expected_request_sha256, owner_pid, owner_birth, returncode, expected_resource_ledger, cleanup_timeout_seconds=5):
    q=json.loads(request_raw)
    assert terminal["owner_pid"]==owner_pid and terminal["owner_birth"]==owner_birth
    assert terminal["request_sha256"]==expected_request_sha256 and returncode==1
    assert expected_resource_ledger==str(Path(q["roots"][0])/"private-ledger.json")
    assert 0 < cleanup_timeout_seconds <= 5
    options=json.loads(Path(q["roots"][0],"modeled-V5-current-inspection.json").read_text())
    Path(q["roots"][0],"modeled-V5-inspector-called.json").write_text(json.dumps({
        "owner_pid":owner_pid,"owner_birth":owner_birth,"returncode":returncode,
        "cleanup_timeout_seconds":cleanup_timeout_seconds,"request_sha256":expected_request_sha256,
        "terminal_schema":terminal["schema"],"modeled_dependency_reason":options["reason"]}))
    return options["verdict"]

if __name__=="__main__":
    q=json.loads(Path(sys.argv[sys.argv.index("--request")+1]).read_bytes())
    sha=sys.argv[sys.argv.index("--expected-request-sha256")+1]
    Path(q["roots"][0],"actual-child-started").write_text("actual original engine Popen reached")
    fields=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
    time.sleep(.05)
    options=json.loads(Path(q["roots"][0],"modeled-V5-current-inspection.json").read_text())
    zero={"scope":"sampled_birth_bound_descendants","group_rss":{"available":True,"live_processes":0,"rss_bytes":0},
        "live_processes":0,"rss_bytes":0,"observed_identities":[],"out_of_group_live_identities":[],"tracking_error":""}
    row={"status":"retained","storage_bytes":16777216,"child":None,"retention_reason":"context_failed",
        "artifacts_durable_asserted":False}
    terminal={"schema":options["terminal_schema"],"owner_pid":os.getpid(),"owner_birth":fields[19],"request_sha256":sha,
        "failure":{"error_type":"ResourceAdmissionDeadlineError","error":"original absolute resource operation deadline expired",
            "executed_checks":[],"elapsed_seconds":q["timeout_seconds"]+1,"process_observation":zero},
        "reservation":{"status":"retained","record":row,"cleanup_error":None,"process_observation":zero,
            "resource_lease":{"released":True,"cancelled":True},
            "ordinary_scratch_profile":{"schema":"owned-validation-ordinary-scratch-live/v1","request_sha256":sha,
                "ordinary_absences_observed":0,"live_exception_active":False,"unseen_leaf_ownership_attested":False,"whole_census_resampling":False}}}
    print(json.dumps(terminal),flush=True)
    raise SystemExit(1)
'''
    if api == 'missing':
        source = source.replace('def inspect_owned_validation_no_child_native_closed_v5(',
                                'def absent_V5_export(')
    elif api == 'alias':
        source = source.replace('if __name__=="__main__":',
            'inspect_owned_validation_no_child_native_closed_v5 = classify_owned_validation_source_custody\n\nif __name__=="__main__":')
    elif api == 'not_callable':
        source = source.replace('if __name__=="__main__":',
            'inspect_owned_validation_no_child_native_closed_v5 = 1\n\nif __name__=="__main__":')
    else:
        assert api == 'current'
    cli_path.chmod(0o600); cli_path.write_text(source); cli_path.chmod(0o400)
    current = pin(cli_path)
    scope.policy['source_descriptors'][:] = [current if row['path']==str(cli_path) else row
                                           for row in scope.policy['source_descriptors']]
    options = {'terminal_schema':terminal_schema,'reason':reason,
               'verdict':copy.deepcopy(_V5_ACQUIRED_CLOSED if outcome is None else outcome)}
    options_pin=put(scope.cwd.parent/'modeled-V5-current-inspection.json', options)
    return scope, options_pin


def test_v5_acquired_consumer_real_engine_wait_dispatches_new_API_before_productive_gate(modeled_v5_process_scope):
    s, options = _v5_acquired_consumer_process_scope(modeled_v5_process_scope)
    context, argv = s.setup('acquired-before-child')
    original_request = owners.checked_descriptor(context['request_pin'])
    completed, witness = owners.original_runner_wait(s.policy, argv, s.cwd, 360, validation_context=context)
    terminal=json.loads(completed.stdout)
    assert completed.returncode==1 and witness['actual_process_waited'] is True
    assert terminal['failure']['executed_checks']==[] and terminal['reservation']['record']['child'] is None
    assert terminal['reservation']['record']['retention_reason']=='context_failed'
    assert terminal['failure']['process_observation']['observed_identities']==[]
    parent=witness['parent_source_custody']
    assert parent['closure_kind']=='original_no_child_native_closed' and parent['storage_status']=='retained'
    assert parent['source_closed'] is True and parent['native_closed'] is True
    assert parent['storage_release_authorized'] is False and parent['retry_authorized'] is False
    assert [kind for kind,_ in s.events]==['enter','enter','exit','exit'] and s.held==[]
    call=json.loads((s.cwd.parent/'modeled-V5-inspector-called.json').read_bytes())
    assert call['owner_pid']==witness['runner_identity']['pid'] and call['returncode']==1
    assert call['request_sha256']==context['request_pin']['sha256']
    assert 0 < call['cleanup_timeout_seconds'] <= 5
    assert owners.checked_descriptor(context['request_pin'])==original_request
    assert owners.checked_descriptor(options) is not None
    saved=json.loads(owners.checked_descriptor(witness['parent_source_closeout']))
    assert saved['parent_source_custody']==parent


@pytest.mark.parametrize('api', ['missing','alias','not_callable'])
def test_v5_acquired_consumer_refuses_missing_or_wrong_API_before_actual_Popen(modeled_v5_process_scope, api):
    s,_=_v5_acquired_consumer_process_scope(modeled_v5_process_scope,api=api)
    context,argv=s.setup('acquired-before-child')
    with pytest.raises((AttributeError,ValueError)):
        owners.original_runner_wait(s.policy,argv,s.cwd,360,validation_context=context)
    assert not s.marker.exists() and s.held==[]
    assert [kind for kind,_ in s.events]==['enter','enter','exit','exit']
    assert not (s.invocation/'runner-wait.json').exists()
    assert not (s.invocation/'parent-source-closeout.json').exists()


@pytest.mark.parametrize('reason,verdict', [
    ('schema_refusal',_V5_ACQUIRED_UNKNOWN),('current_ledger_mismatch',_V5_ACQUIRED_UNKNOWN),
    ('live_Native',_V5_ACQUIRED_UNKNOWN),('ambiguous_native_waiter',_V5_ACQUIRED_UNKNOWN),
    ('wrong_profile',{'verdict':'proven_original_no_child_no_native_grant','source_custody_closed':True,'storage_release_authorized':False}),
    ('extra_authority',dict(_V5_ACQUIRED_CLOSED,Native_grant_created=True)),
    ('closed_flag_false',dict(_V5_ACQUIRED_CLOSED,source_custody_closed=False)),
])
def test_v5_acquired_consumer_UNKNOWN_or_wrong_typed_current_result_retains_Source(modeled_v5_process_scope, reason, verdict):
    # Native/ledger/schema decisions are explicit current-D dependency DATA.
    # This owning test verifies that A neither bypasses UNKNOWN nor falls back
    # to another inspector/source release after the trusted D rejection.
    s,options=_v5_acquired_consumer_process_scope(modeled_v5_process_scope,
        terminal_schema='wrong' if reason=='schema_refusal' else 'owned-validation-failure-terminal/v1',outcome=verdict,reason=reason)
    context,argv=s.setup('acquired-before-child')
    with pytest.raises(s.Unknown):
        owners.original_runner_wait(s.policy,argv,s.cwd,360,validation_context=context)
    assert s.marker.exists() and len(s.held)==1
    held=s.held[0]
    assert held['returncode']==1 and held['engine_returned'] is True
    assert held['source_closed'] is False and held['open_contexts']==2
    assert (s.invocation/'runner-wait.json').exists()
    assert not (s.invocation/'parent-source-closeout.json').exists()
    call=json.loads((s.cwd.parent/'modeled-V5-inspector-called.json').read_bytes())
    assert call['owner_pid']==held['process'].pid


@pytest.mark.parametrize('fault',['unknown_wait','false_wait','wrong_pid','wrong_cutoff'])
def test_v5_acquired_consumer_requires_existing_typed_original_wait_before_inspection(typed_v5_modeled_boundaries,fault):
    s=typed_v5_modeled_boundaries
    s.terminal['reservation'].update(status='retained',cleanup_error=None)
    s.terminal['reservation']['record'].update(status='retained',child=None)
    edits={'wrong_pid':{'process_pid':9999},'wrong_cutoff':{'cleanup_deadline_monotonic':1341.0}}
    completed=s.result(waited=None if fault=='unknown_wait' else False if fault=='false_wait' else True,
                       edits=edits.get(fault))
    with pytest.raises(ValueError):s.c.finish(completed,s.record(completed))
    assert s.native_calls==[] and s.c.closed is False and 'source_exit' not in s.events
    assert not (s.invocation/'runner-wait.json').exists()


def test_v5_acquired_consumer_expired_original_authority_cannot_begin_new_proof(typed_v5_modeled_boundaries):
    s=typed_v5_modeled_boundaries
    s.terminal['reservation'].update(status='retained',cleanup_error=None)
    s.terminal['reservation']['record'].update(status='retained',child=None)
    completed=s.result();s.now['value']=s.c.deadline
    witness=s.record(completed)
    with pytest.raises(ValueError,match='original V5 source/native proof budget expired'):
        s.c.finish(completed,witness)
    assert s.native_calls==[] and s.c.closed is False and s.events==[]
    assert not (s.invocation/'runner-wait.json').exists()


@pytest.mark.parametrize('new_API_present',[False,True])
def test_v5_acquired_consumer_checked_policy_requires_explicit_new_ABI_before_work(metadata_scope,monkeypatch,new_API_present):
    s=metadata_scope;policy=_ordinary_v5_policy(s.policy);d=Path(policy['datasets_root'])
    cli=d/'scripts/ops/legal_ir/run_owned_validation.py'
    body=b'def inspect_owned_validation_productive_native_closed_v1():\n    pass\n'
    if new_API_present:
        body+=b'def inspect_owned_validation_no_child_native_closed_v5():\n    pass\n'
    cli.chmod(0o600);cli.write_bytes(body);cli.chmod(0o400)
    rows={row['path']:row for row in policy['source_descriptors']};rows[str(cli)]=pin(cli)
    directory=d/'ipfs_datasets_py/optimizers/logic_theorem_optimizer'
    required=owners._v5_lifecycle_owner_paths(d)
    required.update(str(path) for path in (directory/'autoencoder_source_lease.py',
        directory/'autoencoder_resource_recovery.py',directory/'runtime_telemetry.py',directory/'validation_pytest.py',
        directory/'__init__.py',d/'ipfs_datasets_py/optimizers/__init__.py',d/'ipfs_datasets_py/__init__.py',
        d/'ipfs_datasets_py/router_deps.py',d/'ipfs_datasets_py/ipfs_backend_router.py',
        d/'ipfs_datasets_py/logic/__init__.py',d/'ipfs_datasets_py/logic/backends/process.py'))
    for spelling in sorted(required):
        path=Path(spelling)
        if not path.exists():put(path,b'# Explicit metadata-only dependency fixture.\n')
        rows[spelling]=pin(path)
    policy['source_descriptors']=list(rows.values())
    selected=put(s.state/'new-ABI-metadata-policy.json',policy)
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append((args,kwargs));raise AssertionError('metadata ABI gate cannot open Source or run work')
    monkeypatch.setattr(owners,'public_source_lease',forbidden)
    monkeypatch.setattr(owners,'original_runner_wait',forbidden)
    if new_API_present:
        assert owners.checked_policy(Path(selected['path']),selected['sha256'],current_owner=False)==policy
    else:
        with pytest.raises(ValueError,match='public V5 acquired-no-child source inspector is unavailable'):
            owners.checked_policy(Path(selected['path']),selected['sha256'],current_owner=False)
    assert calls==[]


def test_v5_acquired_consumer_preserves_legacy_metadata_policy_without_new_API(metadata_scope):
    # The ABI requirement is confined to the already explicit ordinary V5
    # policy. Legacy metadata/default V1 does not infer the opt-in.
    s=metadata_scope
    assert 'ordinary_scratch_profile' not in s.policy['validation']
    checked=owners.checked_policy(Path(s.policy_pin['path']),s.policy_pin['sha256'],current_owner=False)
    assert checked==s.policy


# UNIT-only diagnostics and bootstrap-identity regressions. These use the real
# maintained parser/profile methods but never a Native constructor, SDK, lease,
# build, verified result, or compiler packet. Modeled collector state cannot mint.
def _reviewed_bad_pytest_entrypoint_policy(scope, label):
    policy = copy.deepcopy(scope.policy)
    entrypoint = Path(pytest.__file__).resolve().with_name('__main__.py')
    assert entrypoint.is_file()
    policy['validation']['pytest_module'] = pin(entrypoint)
    reference = put(scope.root / ('wrong-pytest-entrypoint-' + label + '.json'), policy)
    return Path(reference['path']), reference['sha256']


def test_reviewed_parent_pytest_package_identity_reaches_inert_operation_without_import(
        reviewed_parent_normal_namespace_v2_scope, monkeypatch):
    import importlib
    import importlib.util
    s = reviewed_parent_normal_namespace_v2_scope
    path, digest = _namespace_v2_policy_before_invalid_operation(s, s.v2, 'normal-pytest-identity')
    def forbidden(*args, **kwargs):
        pytest.fail('metadata-only Pytest identity checker executed a module resolver/import')
    with monkeypatch.context() as patch:
        patch.setattr(importlib, 'import_module', forbidden)
        patch.setattr(importlib.util, 'find_spec', forbidden)
        with pytest.raises(ValueError, match='^closed reviewed parent operation required$'):
            owners.checked_reviewed_preservation_parent_policy_v1(path, digest)


def test_reviewed_parent_pytest_matching_hash_entrypoint_is_not_package_identity(
        reviewed_parent_normal_namespace_v2_scope):
    s = reviewed_parent_normal_namespace_v2_scope
    path, digest = _reviewed_bad_pytest_entrypoint_policy(s, 'checked-policy')
    with pytest.raises(ValueError, match='^reviewed parent Pytest must pin its normal package initializer$'):
        owners.checked_reviewed_preservation_parent_policy_v1(path, digest)
    assert not Path(s.policy['validation']['attempt_root']).exists()
    assert not Path(s.policy['validation']['resource_ledger']).exists()
    assert not Path(s.policy['validation']['scheduler_state_path']).exists()


def test_reviewed_parent_wrong_pytest_identity_refuses_public_entry_before_dispatch(
        reviewed_parent_normal_namespace_v2_scope):
    s = reviewed_parent_normal_namespace_v2_scope
    path, digest = _reviewed_bad_pytest_entrypoint_policy(s, 'public-entry')
    invocation = s.root / 'must-not-create-original-parent-invocation'
    observations = []
    active = [True]
    def observe(event, values):
        if active[0] and event == 'subprocess.Popen':
            observations.append(event)
    sys.addaudithook(observe)
    try:
        # Keep original guarded owner/Popen callables untouched. The operation
        # remains deliberately inert too; an old checker still cannot dispatch.
        with pytest.raises(ValueError, match='^reviewed parent Pytest must pin its normal package initializer$'):
            owners.run_reviewed_preservation_parent_v1(path, digest, invocation_directory=invocation)
    finally:
        active[0] = False
    assert observations == []
    assert not invocation.exists()
    assert not Path(s.policy['validation']['attempt_root']).exists()
    assert not Path(s.policy['validation']['resource_ledger']).exists()
    assert not Path(s.policy['validation']['scheduler_state_path']).exists()


def _reviewed_modeled_native_build_unwind(mode, sentinel):
    if mode == 'raise':
        raise sentinel
    if mode == 'return_none':
        return None
    return object()


def _reviewed_modeled_native_collector_for_unwind():
    import types
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    native = types.ModuleType('unit_only_modeled_native_profile_target')
    native.__dict__['__builtins__'] = __builtins__
    native.VerifiedReviewedPreservationNativeSuccess = type('UnitOnlyUnissuedSuccess', (), {})
    build = types.FunctionType(_reviewed_modeled_native_build_unwind.__code__, vars(native),
        '_reviewed_modeled_native_build_unwind')
    collector = object.__new__(worker.ReviewedPreservationNativeCollectorV1)
    collector.native = native
    collector.build = build
    collector.build_code = build.__code__
    collector.popen_code = _reviewed_modeled_native_collector_for_unwind.__code__
    collector.popen_globals = {}
    collector.pending_popen = None
    collector.operation = {'nodeid': 'unit-only-modeled-build'}
    collector.current_node = collector.operation['nodeid']
    collector.phase = 'call'
    collector.build_calls = 0
    collector.build_frame = None
    collector.native_events = []
    collector.processes = []
    collector.binding = collector.handle = collector.inputs = None
    collector.build_return_refusal = None
    verified = []
    collector.verify_parent = lambda: verified.append('modeled-parent-check')
    collector._read_binding = lambda: pytest.fail('failed UNIT build must never read or issue a Native binding')
    collector.profile_hook = collector.profile
    return collector, build, verified


@pytest.mark.parametrize('mode', ['raise', 'return_none', 'return_object'])
def test_reviewed_parent_profile_nonhandle_return_preserves_exception_and_cannot_qualify(mode):
    collector, build, verified = _reviewed_modeled_native_collector_for_unwind()
    sentinel = RuntimeError('original modeled native build failure sentinel')
    previous = sys.getprofile()
    assert previous is None
    audit_events = []
    active = [False]
    def strict_profile_audit(event, values):
        if active[0] and event == 'sys.setprofile':
            audit_events.append(event)
            raise PermissionError('modeled strict profile must not be automatically cleared')
    sys.addaudithook(strict_profile_audit)
    sys.setprofile(collector.profile_hook)
    try:
        active[0] = True
        if mode == 'raise':
            with pytest.raises(RuntimeError) as caught:
                build(mode, sentinel)
            assert caught.value is sentinel
        else:
            result = build(mode, sentinel)
            assert (result is None) is (mode == 'return_none')
        assert sys.getprofile() is collector.profile_hook
        assert audit_events == []
        assert verified == ['modeled-parent-check']
        assert collector.build_calls == 1 and collector.build_frame is None
        assert collector.binding is collector.handle is collector.inputs is None
        assert collector.build_return_refusal == {
            'reason': 'non_verified_native_success_return',
            'returned_none': mode != 'return_object'}
        with pytest.raises(ValueError, match='^selected original native observation is incomplete$'):
            collector.finish()
        assert collector.binding is collector.handle is collector.inputs is None
    finally:
        active[0] = False
        sys.setprofile(previous)


def test_reviewed_parent_profile_missing_binding_stays_refused_even_with_three_modeled_events():
    collector, _, _ = _reviewed_modeled_native_collector_for_unwind()
    previous = sys.getprofile()
    assert previous is None
    collector.build_calls = 1
    collector.native_events = [None, None, None]
    collector.processes = [None, None, None]
    sys.setprofile(collector.profile_hook)
    try:
        with pytest.raises(ValueError, match='^selected original native observation is incomplete$'):
            collector.finish()
        assert collector.binding is collector.handle is collector.inputs is None
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize('mode', [None, 'w', 'wb+'])
def test_reviewed_parent_guarded_write_refusal_keeps_reason_and_exact_target_metadata(tmp_path, mode):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    denied, details = [], []
    target = tmp_path / 'outside-original-admitted-outputs'
    flags = 577
    with pytest.raises(PermissionError, match='^validation write outside original admitted outputs$'):
        worker._refuse_guarded_write_v1(denied, details, target, mode, flags)
    assert denied == ['outside-owned-write']
    assert details == [{'kind': 'guarded-write-refusal/v1', 'path': str(target),
        'path_truncated': False, 'mode': mode, 'mode_truncated': False, 'flags': flags}]
    assert not target.exists()


def test_reviewed_parent_guarded_write_refusal_diagnostics_have_fixed_size_and_no_contents():
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    denied, details = [], []
    for index in range(10):
        with pytest.raises(PermissionError, match='^validation write outside original admitted outputs$'):
            worker._refuse_guarded_write_v1(denied, details, '/' + 'x' * 4096,
                'w' * 32, 2**63 if index == 0 else 577)
    assert denied == ['outside-owned-write'] * 10
    assert len(details) == 8
    for row in details:
        assert set(row) == {'kind', 'path', 'path_truncated', 'mode', 'mode_truncated', 'flags'}
        assert len(row['path']) == 2048 and row['path_truncated'] is True
        assert len(row['mode']) == 16 and row['mode_truncated'] is True
    assert details[0]['flags'] is None and all(row['flags'] == 577 for row in details[1:])
    assert len(json.dumps(details).encode()) < 18000


def _rewrite_reviewed_modeled_json(path, value):
    """Only the private modeled verifier fixture; not an admitted receipt."""
    path.chmod(0o600)
    path.write_bytes(owners.canonical(value) + b'\n')
    path.chmod(0o400)


def _reviewed_native_modeled_verifier_scope(durability_scope, packet_scope, report_value):
    s, packet = durability_scope, packet_scope
    # Join the existing modeled durability and 77-byte-row fixtures through the
    # real verifier. No original execution/Source/Native facts are asserted.
    operation = copy.deepcopy(packet.operation)
    operation['nodeid'] = s.plan['selected_checks'][0]
    s.policy['schema'] = owners.REVIEWED_PARENT_POLICY_SCHEMA
    s.policy['operation'] = copy.deepcopy(operation)
    s.plan.update(schema=owners.REVIEWED_PARENT_WORKER_SCHEMA,
        operation=copy.deepcopy(operation), parent_nonce_sha256='d' * 64)
    _rewrite_reviewed_modeled_json(s.invocation / 'validation-plan.json', s.plan)
    for relative, raw in packet.packet:
        reference = put(s.attempt / operation['packet_root'] / relative, raw)
        info = Path(reference['path']).stat()
        s.durability['files'].append({'path': operation['packet_root'] + '/' + relative,
            'bytes': reference['bytes'], 'sha256': reference['sha256'],
            'device': info.st_dev, 'inode': info.st_ino})
    binding = put(s.attempt / 'exports/reviewed-preservation-native-binding.json', packet.binding)
    _rewrite_reviewed_modeled_json(s.attempt / 'logs/validation-durability.json', s.durability)
    report_path = s.attempt / 'exports/trusted-validation.json'
    report = json.loads(report_path.read_bytes())
    report.update(schema=owners.REVIEWED_PARENT_WORKER_SCHEMA, source_READ_closed=False,
        source_READ_admission_complete=False, source_custody_mode='delegated-original-parent-read/v1',
        parent_nonce_sha256=s.plan['parent_nonce_sha256'], native_binding=binding,
        parent_source_observation={'fixture_only': True, 'not_original_Source_evidence': True},
        plan_sha256=pin(s.invocation / 'validation-plan.json')['sha256'])
    if report_value != 'missing':
        report['native_return_refusal'] = (None if report_value == 'none' else False
            if report_value == 'false' else {'reason': 'non_verified_native_success_return', 'returned_none': True})
    _rewrite_reviewed_modeled_json(report_path, report)
    return s


@pytest.mark.parametrize('report_value', ['missing', 'none', 'refusal', 'false'])
def test_reviewed_parent_native_refusal_metadata_is_consistent_with_positive_binding(
        reviewed_parent_modeled_durability_scope, reviewed_parent_packet_join_scope, report_value):
    s = _reviewed_native_modeled_verifier_scope(reviewed_parent_modeled_durability_scope,
        reviewed_parent_packet_join_scope, report_value)
    if report_value in ('missing', 'none'):
        summary = owners.verify_validation_evidence(s.policy, s.plan, s.request_pin, s.terminal, s.invocation)
        assert summary['cases'] == 1 and len(summary['packet_inventory']) == 77
        assert s.terminal['admitted'] is False
        assert 'original_success_authenticated' not in summary
    else:
        with pytest.raises(ValueError, match='^reviewed worker recorded a failed native build return$'):
            owners.verify_validation_evidence(s.policy, s.plan, s.request_pin, s.terminal, s.invocation)


def test_reviewed_parent_native_refusal_metadata_does_not_change_ordinary_profile(
        reviewed_parent_modeled_durability_scope):
    s = reviewed_parent_modeled_durability_scope
    path = s.attempt / 'exports/trusted-validation.json'
    report = json.loads(path.read_bytes())
    report['native_return_refusal'] = {'fixture_only': True, 'not_a_native_profile': True}
    _rewrite_reviewed_modeled_json(path, report)
    summary = owners.verify_validation_evidence(s.policy, s.plan, s.request_pin, s.terminal, s.invocation)
    assert summary['cases'] == 1 and 'native_binding' not in summary


@contextmanager
def _reviewed_descriptor_relative_write_unit_scope(tmp_path, monkeypatch):
    """Real canonical R writer/FDs; modeled guard/collector, no Native authority.

    Cwd remains inside the original UNIT attempt but outside the modeled inner
    boundary, so the unchanged outer ordinary profile needs no permission.
    """
    import os
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    from ipfs_datasets_py.logic.autoformal import legal_corpus_lake as corpus
    root = tmp_path / 'modeled-inner-attempt'; root.mkdir()
    output = root / 'output'; output.mkdir()
    cwd = tmp_path / 'unit-cwd'; cwd.mkdir()
    foreign = tmp_path / 'outside-inner-attempt'; foreign.mkdir()
    monkeypatch.chdir(cwd)
    source = Path(corpus.__file__)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    guard = SimpleNamespace(_reviewed_native_source=True, pins={str(source): source_sha},
        executed={str(source): source_sha}, compiled_codes={corpus._write_new_file.__code__: str(source)})
    collector = object.__new__(worker.ReviewedPreservationNativeCollectorV1)
    collector.guard = guard; collector.policy = {'datasets_root': str(source.parents[3])}
    collector.attempt = root; collector.operation = {'nodeid': 'UNIT-modeled-selected-call'}
    collector.current_node = collector.operation['nodeid']; collector.phase = 'call'
    collector.build_calls = 1; collector.build_frame = sys._getframe(2)
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    s = SimpleNamespace(worker=worker, corpus=corpus, writer=corpus._write_new_file,
        collector=collector, guard=guard, source=str(source), directory_fd=directory_fd,
        root=root, output=output, foreign=foreign, cwd=cwd, events=[], active=False,
        mutate_event=lambda values: values, alternate_frame=None)
    def audit(event, values):
        if not s.active or event != 'open' or type(values[0]) not in (str, bytes):
            return
        if not values[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return
        observed = s.mutate_event(values)
        frame = sys._getframe(1) if s.alternate_frame is None else s.alternate_frame
        try:
            target = worker._reviewed_native_directory_write_target_v1(observed, frame, collector)
            if target is None:
                target = worker._owned_validation_private_target_v1(observed[0], root)
        except (OSError, ValueError):
            raise PermissionError('UNIT inner owned-target refusal') from None
        s.events.append((tuple(values), target))
    sys.addaudithook(audit)
    try:
        yield s
    finally:
        s.active = False
        try:
            os.close(directory_fd)
        except OSError:
            pass


def test_reviewed_parent_real_descriptor_writer_rebases_exact_open_audit_target(tmp_path, monkeypatch):
    import os
    with _reviewed_descriptor_relative_write_unit_scope(tmp_path, monkeypatch) as s:
        raw = b'real descriptor anchored UNIT bytes\n'
        s.active = True
        try:
            s.writer(s.directory_fd, 'exact.lean', raw)
        finally:
            s.active = False
        assert (s.output / 'exact.lean').read_bytes() == raw
        assert not (s.cwd / 'exact.lean').exists()
        assert s.events == [(('exact.lean', None,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC),
            s.output / 'exact.lean')]
        assert s.collector.guard is s.guard


@pytest.mark.parametrize('fault', ['foreign', 'closed', 'deleted', 'regular'])
def test_reviewed_parent_descriptor_writer_refuses_wrong_kernel_directory_fd(tmp_path, monkeypatch, fault):
    import os
    with _reviewed_descriptor_relative_write_unit_scope(tmp_path, monkeypatch) as s:
        extra_fd = None
        if fault == 'foreign':
            extra_fd = os.open(s.foreign, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        elif fault == 'closed':
            os.close(s.directory_fd)
        elif fault == 'deleted':
            s.output.rmdir()
        else:
            marker = s.root / 'regular-marker'; marker.write_bytes(b'unchanged marker')
            extra_fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW)
        selected = s.directory_fd if extra_fd is None else extra_fd
        s.active = True
        try:
            with pytest.raises(PermissionError, match='^UNIT inner owned-target refusal$'):
                s.writer(selected, 'refused.lean', b'never persisted')
        finally:
            s.active = False
            if extra_fd is not None:
                os.close(extra_fd)
        assert not (s.foreign / 'refused.lean').exists()
        assert not (s.cwd / 'refused.lean').exists()
        assert not (s.output / 'refused.lean').exists()
        assert s.events == []


@pytest.mark.parametrize('fault', ['ordinary', 'wrong_node', 'setup', 'teardown', 'no_build',
    'build_count', 'wrong_ancestor', 'no_compiled', 'wrong_compiled', 'no_executed',
    'no_pin', 'wrong_pin', 'wrong_function', 'code_clone', 'wrong_globals', 'changed_open'])
def test_reviewed_parent_descriptor_writer_requires_actual_original_provenance(tmp_path, monkeypatch, fault):
    import types
    with _reviewed_descriptor_relative_write_unit_scope(tmp_path, monkeypatch) as s:
        if fault == 'ordinary': s.guard._reviewed_native_source = False
        elif fault == 'wrong_node': s.collector.current_node = 'other-call'
        elif fault in ('setup', 'teardown'): s.collector.phase = fault
        elif fault == 'no_build': s.collector.build_frame = None
        elif fault == 'build_count': s.collector.build_calls = 2
        elif fault == 'wrong_ancestor':
            def unrelated_frame(): return sys._getframe()
            s.collector.build_frame = unrelated_frame()
        elif fault == 'no_compiled': s.guard.compiled_codes.clear()
        elif fault == 'wrong_compiled': s.guard.compiled_codes[s.writer.__code__] = s.source + '.other'
        elif fault == 'no_executed': s.guard.executed.clear()
        elif fault == 'no_pin': s.guard.pins.clear()
        elif fault == 'wrong_pin': s.guard.pins[s.source] = '0' * 64
        elif fault == 'wrong_function': monkeypatch.setattr(s.corpus, '_write_new_file', lambda *a: None)
        elif fault in ('code_clone', 'wrong_globals'):
            code = s.writer.__code__.replace() if fault == 'code_clone' else s.writer.__code__
            namespace = s.writer.__globals__ if fault == 'code_clone' else dict(s.writer.__globals__)
            s.writer = types.FunctionType(code, namespace, '_write_new_file')
            monkeypatch.setattr(s.corpus, '_write_new_file', s.writer)
        elif fault == 'changed_open': monkeypatch.setattr(s.worker, '_ORIGINAL_OS_OPEN', object())
        s.active = True
        try:
            with pytest.raises(PermissionError, match='^UNIT inner owned-target refusal$'):
                s.writer(s.directory_fd, 'refused.lean', b'never persisted')
        finally:
            s.active = False
        assert not (s.output / 'refused.lean').exists()
        assert not (s.cwd / 'refused.lean').exists()
        assert s.events == []


@pytest.mark.parametrize('fault', ['mode', 'extra_flags', 'missing_cloexec', 'bytes_path',
    'different_basename', 'parent_traversal', 'boolean_flags'])
def test_reviewed_parent_descriptor_writer_refuses_mismatched_audit_event(tmp_path, monkeypatch, fault):
    import os
    with _reviewed_descriptor_relative_write_unit_scope(tmp_path, monkeypatch) as s:
        def changed(values):
            name, mode, flags = values
            if fault == 'mode': mode = 'w'
            elif fault == 'extra_flags': flags |= os.O_TRUNC
            elif fault == 'missing_cloexec': flags &= ~os.O_CLOEXEC
            elif fault == 'bytes_path': name = name.encode()
            elif fault == 'different_basename': name = 'other.lean'
            elif fault == 'parent_traversal': name = '../refused.lean'
            else: flags = True
            return name, mode, flags
        s.mutate_event = changed
        s.active = True
        try:
            with pytest.raises(PermissionError, match='^UNIT inner owned-target refusal$'):
                s.writer(s.directory_fd, 'refused.lean', b'never persisted')
        finally:
            s.active = False
        assert not (s.output / 'refused.lean').exists()
        assert not (s.cwd / 'refused.lean').exists()
        assert s.events == []


def test_reviewed_parent_unattributed_relative_open_keeps_original_target_boundary(tmp_path, monkeypatch):
    import os
    with _reviewed_descriptor_relative_write_unit_scope(tmp_path, monkeypatch) as s:
        s.active = True
        try:
            with pytest.raises(PermissionError, match='^UNIT inner owned-target refusal$'):
                os.open('unattributed.lean', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        finally:
            s.active = False
        assert not (s.cwd / 'unattributed.lean').exists()
        assert not (s.output / 'unattributed.lean').exists()
        assert s.events == []



def test_reviewed_parent_descriptor_writer_rejects_replaced_open_before_source_read(tmp_path, monkeypatch):
    import os
    with _reviewed_descriptor_relative_write_unit_scope(tmp_path, monkeypatch) as s:
        original = os.open
        replacement_calls = []
        def replacement(*args, **kwargs):
            replacement_calls.append((args, kwargs))
            raise AssertionError('replaced os.open must never be dispatched')
        def changed(values):
            # The genuine R os.open has already entered its actual audit event.
            # Change the global only now; helper attribution must refuse before
            # its source reauthentication can dispatch through this callable.
            os.open = replacement
            return values
        s.mutate_event = changed
        s.active = True
        try:
            with pytest.raises(PermissionError, match='^UNIT inner owned-target refusal$'):
                s.writer(s.directory_fd, 'refused.lean', b'never persisted')
        finally:
            s.active = False
            os.open = original
        assert replacement_calls == []
        assert not (s.output / 'refused.lean').exists()
        assert not (s.cwd / 'refused.lean').exists()
        assert s.events == []



def _reviewed_selected_test_call_unit_scope(request, frame):
    """Genuine current Pytest frame/item; explicitly modeled Native metadata."""
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    collector, _, _ = _reviewed_modeled_native_collector_for_unwind()
    source = str(Path(__file__).absolute())
    driver = pytest_python.pytest_pyfunc_call
    driver_source = driver.__code__.co_filename
    pins = {source: pin(Path(source))['sha256'], driver_source: pin(Path(driver_source))['sha256']}
    guard = SimpleNamespace(_reviewed_native_source=True, pins=dict(pins),
        executed={driver_source: pins[driver_source]},
        compiled_codes={frame.f_code: source, driver.__code__: driver_source})
    node = source + '::' + request.node.nodeid.partition('::')[2]
    collector.guard = guard
    collector.operation = {'nodeid': node}
    collector.current_node = node; collector.phase = 'call'
    collector.plan = {'source_pins': dict(pins), 'required_candidate_bodies': [source]}
    collector.policy = {'validation': {'trusted_tests': [pin(Path(source))]}}
    collector.raw_compiled_sources = {source: pins[source]}
    collector.selected_test_source_calls = {}; collector.selected_test_path = source
    collector.pytest_python = pytest_python; collector.pytest_call = driver
    collector.pytest_call_code = driver.__code__; collector.pytest_call_globals = driver.__globals__
    collector.pytest_function_type = pytest_python.Function
    assert frame.f_back.f_code is driver.__code__
    assert frame.f_back.f_locals['pyfuncitem'] is request.node
    assert frame.f_back.f_locals['testfunction'] is request.node.obj
    return SimpleNamespace(worker=worker, collector=collector, guard=guard,
        source=source, driver_source=driver_source, pins=pins)


def test_reviewed_parent_real_pytest_call_witness_is_distinct_from_loader_completion(request):
    frame = sys._getframe()
    s = _reviewed_selected_test_call_unit_scope(request, frame)
    before = dict(s.guard.executed)
    previous_profile = sys.getprofile()
    # The event adapter is modeled; the current frame and immediate Pytest
    # caller/item/function are genuine. This creates no Native authority.
    s.collector.profile(frame, 'call', None)
    assert sys.getprofile() is previous_profile
    assert s.guard.executed == before and s.source not in before
    witness = s.collector.selected_test_source_calls[s.source]
    assert witness['kind'] == 'selected-trusted-test-function-call/v1'
    assert witness['sha256'] == s.pins[s.source]
    assert witness['nodeid'] == s.collector.operation['nodeid']
    assert witness['module_loader_completion_claimed'] is False
    projected = s.worker._reviewed_native_source_execution_projection_v1(
        s.guard, s.collector, s.collector.plan, s.collector.raw_compiled_sources)
    assert projected['executed_source_buffers'][s.source] == s.pins[s.source]
    attribution = projected['source_execution_attribution']
    assert attribution['module_loader_executed_source_buffers'] == before
    assert attribution['selected_test_function_call_witnesses'] == [witness]
    assert attribution['missing_required_candidate_body_count'] == 0
    assert s.guard.executed == before


@pytest.mark.parametrize('fault', ['ordinary', 'wrong_node', 'wrong_phase', 'no_raw', 'wrong_raw',
    'no_pin', 'wrong_pin', 'no_code', 'equal_code_clone', 'wrong_driver', 'wrong_driver_code',
    'wrong_driver_globals', 'no_driver_execution', 'wrong_driver_pin', 'wrong_item_type',
    'changed_open', 'source_read_refusal'])
def test_reviewed_parent_pytest_call_attribution_refuses_without_profile_errors(request, monkeypatch, fault):
    frame = sys._getframe()
    s = _reviewed_selected_test_call_unit_scope(request, frame)
    if fault == 'ordinary': s.guard._reviewed_native_source = False
    elif fault == 'wrong_node': s.collector.current_node = 'other-node'
    elif fault == 'wrong_phase': s.collector.phase = 'setup'
    elif fault == 'no_raw': s.collector.raw_compiled_sources.clear()
    elif fault == 'wrong_raw': s.collector.raw_compiled_sources[s.source] = '0' * 64
    elif fault == 'no_pin': s.guard.pins.pop(s.source)
    elif fault == 'wrong_pin': s.guard.pins[s.source] = '0' * 64
    elif fault == 'no_code': s.guard.compiled_codes.pop(frame.f_code)
    elif fault == 'equal_code_clone':
        s.guard.compiled_codes.clear()
        s.guard.compiled_codes.update({frame.f_code.replace(): s.source,
            pytest_python.pytest_pyfunc_call.__code__: s.driver_source})
    elif fault == 'wrong_driver': s.collector.pytest_call = lambda: None
    elif fault == 'wrong_driver_code': s.collector.pytest_call_code = s.collector.pytest_call_code.replace()
    elif fault == 'wrong_driver_globals': s.collector.pytest_call_globals = dict(s.collector.pytest_call_globals)
    elif fault == 'no_driver_execution': s.guard.executed.pop(s.driver_source)
    elif fault == 'wrong_driver_pin': s.guard.pins[s.driver_source] = '0' * 64
    elif fault == 'wrong_item_type': s.collector.pytest_function_type = type(None)
    elif fault == 'changed_open': monkeypatch.setattr(s.worker, '_ORIGINAL_OS_OPEN', object())
    else:
        def refused(*args, **kwargs): raise OSError('modeled attribution read refusal')
        monkeypatch.setattr(s.worker, 'ordinary_bytes', refused)
    before = dict(s.guard.executed)
    previous_profile = sys.getprofile()
    s.collector.profile(frame, 'call', None)
    assert sys.getprofile() is previous_profile
    assert s.guard.executed == before
    assert s.collector.selected_test_source_calls == {}


def test_reviewed_parent_raw_compilation_and_noncall_events_do_not_credit_execution(request):
    frame = sys._getframe()
    s = _reviewed_selected_test_call_unit_scope(request, frame)
    s.collector.profile(frame, 'c_call', None)
    s.collector.profile(frame, 'return', None)
    assert s.collector.selected_test_source_calls == {} and s.source not in s.guard.executed
    projected = s.worker._reviewed_native_source_execution_projection_v1(
        s.guard, s.collector, s.collector.plan, s.collector.raw_compiled_sources)
    assert s.source not in projected['executed_source_buffers']
    missing = projected['source_execution_attribution']['missing_required_candidate_bodies']
    assert missing == [{'path': s.source, 'path_truncated': False,
        'expected_sha256': s.pins[s.source], 'module_loader_executed_sha256': None,
        'raw_compiled_sha256': s.pins[s.source]}]


def test_reviewed_parent_missing_source_attribution_details_are_bounded(request):
    frame = sys._getframe()
    s = _reviewed_selected_test_call_unit_scope(request, frame)
    paths = ['/' + 'x' * 3000 + str(index) for index in range(12)]
    plan = {'source_pins': {p: '0' * 64 for p in paths}, 'required_candidate_bodies': paths}
    projected = s.worker._reviewed_native_source_execution_projection_v1(s.guard, None, plan, {})
    d = projected['source_execution_attribution']
    assert d['missing_required_candidate_body_count'] == 12
    assert d['missing_details_truncated'] is True
    assert len(d['missing_required_candidate_bodies']) == 8
    assert all(len(row['path']) == 2048 and row['path_truncated'] is True
        for row in d['missing_required_candidate_bodies'])
    assert s.collector.selected_test_source_calls == {} and s.source not in s.guard.executed


def test_reviewed_parent_execution_projection_refuses_unrelated_call_witness(request):
    frame = sys._getframe()
    s = _reviewed_selected_test_call_unit_scope(request, frame)
    s.collector.profile(frame, 'call', None)
    witness = s.collector.selected_test_source_calls.pop(s.source)
    other = s.source + '.other'
    s.collector.selected_test_source_calls[other] = {**witness, 'path': other}
    projected = s.worker._reviewed_native_source_execution_projection_v1(
        s.guard, s.collector, s.collector.plan, s.collector.raw_compiled_sources)
    assert s.source not in projected['executed_source_buffers']
    assert other not in projected['executed_source_buffers']
    assert projected['source_execution_attribution']['selected_test_function_call_witnesses'] == []
    assert projected['source_execution_attribution']['missing_required_candidate_body_count'] == 1


# Incoming-SDK archival controls use actual local sources and the real current
# metadata readers. All execution/Source/native statements below are explicitly
# FICTIONAL unit DATA; no process, Source lease, native grant, service or old code
# is executed/loaded. The ordinary current API remains strict and unchanged.
@pytest.fixture
def archived_v5_sdk_scope(metadata_scope):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    s=metadata_scope;policy=_ordinary_v5_policy(s.policy);d=Path(policy['datasets_root'])
    cli=d/'scripts/ops/legal_ir/run_owned_validation.py'
    cli.chmod(0o600);cli.write_bytes(b'def inspect_owned_validation_productive_native_closed_v1(): pass\n'
        b'def inspect_owned_validation_no_child_native_closed_v5(): pass\n');cli.chmod(0o400)
    rows={row['path']:row for row in policy['source_descriptors']};rows[str(cli)]=pin(cli)
    directory=d/'ipfs_datasets_py/optimizers/logic_theorem_optimizer'
    required=owners._v5_lifecycle_owner_paths(d)
    required.update(str(path)for path in (directory/'autoencoder_source_lease.py',directory/'autoencoder_resource_recovery.py',
        directory/'runtime_telemetry.py',directory/'validation_pytest.py',directory/'__init__.py',
        d/'ipfs_datasets_py/optimizers/__init__.py',d/'ipfs_datasets_py/__init__.py',d/'ipfs_datasets_py/router_deps.py',
        d/'ipfs_datasets_py/ipfs_backend_router.py',d/'ipfs_datasets_py/logic/__init__.py',d/'ipfs_datasets_py/logic/backends/process.py'))
    for spelling in sorted(required):
        path=Path(spelling)
        if not path.exists():put(path,b'# Fictional current Source dependency DATA only.\n')
        rows[spelling]=pin(path)
    engine_ref=put(s.control/'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py',Path(engine.__file__).read_bytes())
    rows[engine_ref['path']]=engine_ref;policy['source_descriptors']=list(rows.values())
    old_manifest=policy['control_release']['manifest'];manifest=json.loads(Path(old_manifest['path']).read_bytes())
    manifest['files']=sorted([*manifest['files'],engine_ref],key=lambda row:row['path'])
    manifest_ref=put(s.control/'complete-incoming-control-manifest.json',manifest)
    policy['control_release']['manifest']=manifest_ref
    policy_ref=put(s.state/'archived-v5-current-body-policy.json',policy)
    s.archive_policy=policy;s.archive_policy_ref=policy_ref;s.archive_engine=engine_ref
    s.archive_manifest=manifest_ref;s.archive_handles=[]
    yield s
    for handle in s.archive_handles:
        if handle in getattr(owners,'_ARCHIVED_V5_SDK_SOURCES',{}):owners.close_archived_v5_sdk_source(handle)


def _archive_sdk_handle(scope):
    handle=owners.authenticate_archived_v5_sdk_source(scope.archive_policy_ref)
    scope.archive_handles.append(handle);return handle


def _archive_sdk_replace(path,value):
    path.chmod(0o600)
    path.write_bytes(value if type(value) is bytes else owners.canonical(value)+b'\n')
    path.chmod(0o400);return pin(path)


def test_archived_v5_sdk_public_capability_refuses_unissued_descriptor_before_any_lifecycle():
    """Minimal source-only RED1 on unchanged7ad: missing archival API."""
    with pytest.raises(ValueError):owners.authenticate_archived_v5_sdk_source({})


def test_archived_v5_sdk_same_NEW_body_at_distinct_roots_is_source_DATA_only(archived_v5_sdk_scope,monkeypatch):
    s=archived_v5_sdk_scope
    def forbidden(*args,**kwargs):pytest.fail('archived SDK metadata reader reached a lifecycle action')
    monkeypatch.setattr(owners,'public_source_lease',forbidden)
    monkeypatch.setattr(owners,'original_runner_wait',forbidden)
    with pytest.raises(ValueError,match='actual owner is outside'):
        owners.checked_policy(Path(s.archive_policy_ref['path']),s.archive_policy_ref['sha256'])
    handle=_archive_sdk_handle(s);data=owners.archived_v5_sdk_source_data(handle)
    assert data['producer_ref']['path']!=str(Path(owners.__file__).absolute())
    assert data['control_root']==str(s.control) and data['policy']==s.archive_policy
    assert data['same_source_generation_bytes_authenticated'] is True
    for key in ('current_producer_verified','current_engine_verified','Source_authority','native_authority',
        'dispatch_authority','retry_authority','storage_release_authority','runtime_activation_authority',
        'completion_authority','current_health_authority'):assert data[key] is False
    # A mutable DATA copy is never the registry's policy or a usable capability.
    data['policy']['python']='caller-rewritten'
    assert owners.archived_v5_sdk_source_data(handle)['policy']==s.archive_policy
    with pytest.raises(ValueError,match='not authenticated'):owners.archived_v5_sdk_source_data(data)


@pytest.mark.parametrize('which',['owner','engine','unused_initializer','manifest','policy'])
def test_archived_v5_sdk_rechecks_every_frozen_source_and_identity(archived_v5_sdk_scope,which):
    s=archived_v5_sdk_scope;handle=_archive_sdk_handle(s)
    paths={'owner':s.control/'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py',
        'engine':Path(s.archive_engine['path']),'unused_initializer':s.control/'ipfs_accelerate_py/__init__.py',
        'manifest':Path(s.archive_manifest['path']),'policy':Path(s.archive_policy_ref['path'])}
    path=paths[which]
    # Byte-identical content with a changed mode/ctime must still invalidate the
    # physical immutable-source capability; all hashes otherwise stay correct.
    path.chmod(0o444)
    with pytest.raises(ValueError,match='identity changed'):owners.archived_v5_sdk_source_data(handle)


@pytest.mark.parametrize('which',['owner','engine'])
def test_archived_v5_sdk_different_incoming_body_is_not_qualified_as_NEW_reader(archived_v5_sdk_scope,which):
    s=archived_v5_sdk_scope;policy=copy.deepcopy(s.archive_policy)
    path=(s.control/'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py'
          if which=='owner' else Path(s.archive_engine['path']))
    changed=_archive_sdk_replace(path,b'# Another OLD SDK/source body; never executed.\n')
    manifest=json.loads(Path(s.archive_manifest['path']).read_bytes())
    manifest['files']=[changed if row['path']==str(path)else row for row in manifest['files']]
    policy['control_release']['manifest']=put(s.control/('different-'+which+'-manifest.json'),manifest)
    policy['source_descriptors']=[changed if row['path']==str(path)else row for row in policy['source_descriptors']]
    selected=put(s.state/('different-'+which+'-policy.json'),policy)
    with pytest.raises(ValueError,match='not the same source generation'):
        owners.authenticate_archived_v5_sdk_source(selected)


@pytest.mark.parametrize('mutation',['missing_bytes','float_bytes','boolean_bytes','extra_key'])
def test_archived_v5_sdk_policy_descriptor_is_exact_typed_and_bound(archived_v5_sdk_scope,mutation):
    s=archived_v5_sdk_scope;offered=dict(s.archive_policy_ref)
    if mutation=='missing_bytes':offered.pop('bytes')
    elif mutation=='float_bytes':offered['bytes']=float(offered['bytes'])
    elif mutation=='boolean_bytes':offered['bytes']=True
    else:offered['not_descriptor']=True
    with pytest.raises(ValueError):owners.authenticate_archived_v5_sdk_source(offered)


@pytest.mark.parametrize('offered',[None,{},True])
def test_archived_v5_sdk_capability_cannot_be_supplied_as_general_origin_map(offered):
    with pytest.raises(ValueError,match='not authenticated'):owners.archived_v5_sdk_source_data(offered)


def _archive_sdk_typed_closeout_DATA(scope):
    """Genuine reader inputs; EVERY live process/Source fact is fictional DATA."""
    s=scope;invocation=s.state/'original-FICTIONAL-invocation';invocation.mkdir()
    request_ref=put(invocation/'execution-request.json',{'schema':'owned-validation-request/v5','fixture_only':True})
    pid=987654321;identity={'pid':pid,'birth':'12345'};producer=pin(s.control/'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py')
    engine_ref=s.archive_engine;summary={'fixture_only':'No original owner or native execution occurred'}
    observed={'schema':'command-execution-observation/v1','completion_kind':'completed','process_pid':pid,
        'process_returncode':0,'leader_waited':True,'output_capture':'complete','work_deadline_monotonic':970.,
        'cleanup_deadline_monotonic':980.,'observed_at_monotonic':710.}
    execution={'schema':'agent-supervisor/v5-runner-execution-observation@1','request':request_ref,'engine_source':engine_ref,
        'exact_command_result':True,'exact_execution_observation':True,'execution_observation':observed,
        'public_returncode':0,'actual_process_pid':pid,'actual_process_returncode':0,'actual_Popen_returned':True,
        'spawn_attempted':True,'process_count':1,'runner_identity':identity,'identity_observation_error':None,
        'engine_returned':True,'engine_started_at_monotonic':700.,'engine_finished_at_monotonic':720.,
        'spawn_returned_at_monotonic':702.,'identity_observed_at_monotonic':704.,'authority_deadline_monotonic':1000.,
        'work_deadline_monotonic':970.,'cleanup_deadline_monotonic':980.,'requested_nominal_window_seconds':300,
        'window_scope':'FICTIONAL DATA, no process executed','source_closed':False,'native_closed':False,
        'storage_release_authorized':False,'retry_authorized':False}
    execution_ref=put(invocation/'runner-execution-observation.json',execution)
    wait={'request':request_ref,'actual_Popen_returned':True,'actual_process_waited':True,'original_process_owner_returned':True,
        'returncode':0,'actual_process_returncode':0,'engine_source':engine_ref,'runner_identity':identity,
        'execution_observation':execution_ref}
    wait_ref=put(invocation/'runner-wait.json',wait)
    custody={'schema':'agent-supervisor/v5-validation-parent-source-custody@2','closure_kind':'existing_verified_success',
        'storage_status':'released','source_closed':True,'native_closed':True,'storage_release_authorized':False,
        'retry_authorized':False,'original_wait':wait_ref,'verified_checks':summary,'source_receipts':[{'fictional':'R'},{'fictional':'candidate'}],
        'authority_deadline_monotonic':1000.,'execution_observation':execution_ref,'source_exit_finished_at_monotonic':730.,
        'work_deadline_monotonic':970.,'cleanup_deadline_monotonic':980.}
    closeout={'schema':'agent-supervisor/v5-validation-parent-source-closeout@2','request':request_ref,
        'original_wait':wait_ref,'producer':producer,'parent_source_custody':custody}
    closeout_ref=put(invocation/'parent-source-closeout.json',closeout)
    return invocation,request_ref,closeout_ref,closeout,summary


def test_archived_v5_sdk_reuses_complete_typed_core_at_exact_original_paths(archived_v5_sdk_scope):
    s=archived_v5_sdk_scope;handle=_archive_sdk_handle(s)
    invocation,request_ref,closeout_ref,value,summary=_archive_sdk_typed_closeout_DATA(s)
    before=Path(closeout_ref['path']).read_bytes()
    with pytest.raises(ValueError,match='producer differs'):
        owners.verify_v5_parent_source_closeout(s.archive_policy,request_ref,invocation,closeout_ref,summary)
    result=owners.verify_archived_v5_parent_source_closeout(handle,request_ref,invocation,closeout_ref,summary)
    assert result['original_parent_source_closeout']==value and result['historical_evidence_verified'] is True
    for key in ('current_producer_verified','current_engine_verified','Source_authority','native_authority','dispatch_authority',
        'retry_authority','storage_release_authority','runtime_activation_authority','completion_authority',
        'current_health_authority','new_original_process_or_native_execution_performed'):assert result[key] is False
    assert Path(closeout_ref['path']).read_bytes()==before
    # Current public API cannot take an archival capability or hidden flags.
    with pytest.raises(TypeError):
        owners.verify_v5_parent_source_closeout(s.archive_policy,request_ref,invocation,closeout_ref,summary,archived_sdk_source=handle)


@pytest.mark.parametrize('mutation',['producer','engine','deadline','boolean_returncode','incomplete_EOF','authority'])
def test_archived_v5_sdk_does_not_relax_original_paths_typed_wait_or_clock(archived_v5_sdk_scope,mutation):
    s=archived_v5_sdk_scope;handle=_archive_sdk_handle(s)
    invocation,request_ref,closeout_ref,value,summary=_archive_sdk_typed_closeout_DATA(s)
    if mutation=='producer':value['producer']=pin(Path(owners.__file__).absolute())
    elif mutation=='deadline':value['parent_source_custody']['source_exit_finished_at_monotonic']=1000.
    elif mutation=='authority':value['parent_source_custody']['retry_authorized']=True
    else:
        path=invocation/'runner-execution-observation.json';execution=json.loads(path.read_bytes())
        if mutation=='engine':execution['engine_source']=pin(Path(__import__('ipfs_accelerate_py.agent_supervisor.todo_daemon.engine',fromlist=['engine']).__file__).absolute())
        elif mutation=='boolean_returncode':execution['actual_process_returncode']=False
        else:execution['execution_observation']['output_capture']='incomplete'
        changed=_archive_sdk_replace(path,execution)
        wait_path=invocation/'runner-wait.json';wait=json.loads(wait_path.read_bytes());wait['execution_observation']=changed
        wait_ref=_archive_sdk_replace(wait_path,wait)
        value['original_wait']=wait_ref;value['parent_source_custody']['original_wait']=wait_ref
        value['parent_source_custody']['execution_observation']=changed
    closeout_ref=_archive_sdk_replace(Path(closeout_ref['path']),value)
    with pytest.raises(ValueError):
        owners.verify_archived_v5_parent_source_closeout(handle,request_ref,invocation,closeout_ref,summary)


def test_archived_v5_sdk_disposed_capability_is_not_rearmed(archived_v5_sdk_scope):
    s=archived_v5_sdk_scope;handle=_archive_sdk_handle(s);owners.close_archived_v5_sdk_source(handle)
    with pytest.raises(ValueError,match='not authenticated'):owners.archived_v5_sdk_source_data(handle)



# Pure protocol controls: source/proof facts below are authored DATA, not a
# claim that a real CLI/Source/native operation took place in these fixtures.

def test_ledger_busy_profile_requires_distinct_explicit_fixed_authority():
    profile={'schema':'agent-supervisor/validation-ledger-busy-retry@1','max_additional_invocations':2,
        'cooldown_seconds':60,'authorization_epoch_sha256':'a'*64,'expires_at':2000.}
    assert owners.checked_validation_ledger_busy_retry(profile)==profile
    for key,value in [('max_additional_invocations',True),('max_additional_invocations',3),('cooldown_seconds',61),
                      ('expires_at',True),('expires_at',float('inf')),('authorization_epoch_sha256','bad')]:
        with pytest.raises(ValueError):owners.checked_validation_ledger_busy_retry(dict(profile,**{key:value}))
    frozen={'freeze_sha256':'b'*64}
    first=owners.ledger_busy_request_attempt_id(frozen,'a'*64,1)
    assert first==owners.ledger_busy_request_attempt_id(frozen,'a'*64,1)
    assert first!=owners.ledger_busy_request_attempt_id(frozen,'a'*64,2)
    assert first!=owners.ledger_busy_request_attempt_id({'freeze_sha256':'c'*64},'a'*64,1)


def test_original_repair_plan_keeps_unchanged_candidate_binding_and_no_generation(plan_scope):
    s=plan_scope;policy=_ordinary_v5_policy(s.policy);before=copy.deepcopy(s.frozen)
    plan=owners.validation_plan(policy,s.policy_pin,s.frozen,s.freeze_pin,s.root,s.invocation)
    assert s.frozen==before
    assert plan['attempt_directory'].endswith('/'+s.frozen['binding']['attempt_id'])
    assert plan['required_candidate_bodies']==[str(s.root/'source.py')]


def test_ledger_busy_generation_rechecks_original4096_source_cap_after_extra_pins(plan_scope,monkeypatch):
    s=plan_scope;policy=_ordinary_v5_policy(s.policy)
    policy['validation']['trusted_tests']=[];policy['source_descriptors']=[]
    policy['validation']['import_sources']=[{'path':'/modeled/import-%04d.py'%i,'sha256':'a'*64}for i in range(4093)]
    # 4093 modeled imports +pytest+one candidate=4095; generation+ticket=>4097.
    ticket=put(s.invocation.parent/'ledger-busy-slot-01.json',{'modeled':'source-pin cap only'})
    put(s.invocation/'ledger-busy-generation.json',{'spent_slot_ref':ticket})
    monkeypatch.setattr(owners,'validation_request_generation',lambda *a,**k:s.frozen['binding']['attempt_id'])
    with pytest.raises(ValueError,match='complete normal import manifest exceeds'):
        owners.validation_plan(policy,s.policy_pin,s.frozen,s.freeze_pin,s.root,s.invocation)


def test_ledger_busy_reader_refuses_preCLI_SourceLeaseTimeout_before_classifier(tmp_path,monkeypatch):
    invocation=tmp_path/'invocation';invocation.mkdir()
    closeout=put(invocation/'parent-source-closeout.json',{})
    request=put(invocation/'execution-request.json',{'cwd':str(tmp_path)})
    put(invocation/'runner-wait.json',{'stdout':json.dumps({'error_type':'SourceLeaseTimeout'}),
        'argv':['/different/preCLI'], 'runner_identity':{'pid':1,'birth':'1'}})
    policy={'python':sys.executable,'datasets_root':str(tmp_path), 'validation':{'resource_ledger':str(tmp_path/'ledger')}}
    monkeypatch.setattr(owners,'_current_v5_cli_classifier',lambda *a:pytest.fail('preCLI refusal reached classifier'))
    with pytest.raises(ValueError,match='original CLI argv differs'):
        owners.verify_v5_ledger_busy_closeout(policy,request,invocation,closeout)


def test_ledger_busy_fresh_request_generation_preserves_frozen_binding_and_immutable_old_publication(plan_scope,monkeypatch):
    s=plan_scope;policy=_ordinary_v5_policy(s.policy);frozen=copy.deepcopy(s.frozen);frozen['freeze_sha256']='f'*64
    profile={'schema':owners.LEDGER_BUSY_RETRY_SCHEMA,'max_additional_invocations':2,'cooldown_seconds':60,
        'authorization_epoch_sha256':'e'*64,'expires_at':owners.time.time()+1000}
    policy['validation']['validation_ledger_busy_retry']=profile
    prior=put(s.invocation/'operational-ledger-busy.json',{'schema':owners.LEDGER_BUSY_SCHEMA,
        'binding':frozen['binding'],'freeze_sha256':frozen['freeze_sha256'],'source_commitment':frozen['source_commitment'],
        'invocation_root':str(s.invocation),'request':put(s.invocation/'prior-request.json',{'fictional':'old request'}),
        'parent_source_closeout':put(s.invocation/'prior-closeout.json',{'fictional':'old Source DATA'}),
        'retry_authorized':False,'semantic_negative':False})
    # Only prior genuine custody authentication is modeled. Generation, immutable
    # ticket, plan, request builder and public publication readers run unchanged.
    checked=[];monkeypatch.setattr(owners,'verify_v5_ledger_busy_closeout',lambda *args:checked.append(args))
    fresh=s.invocation.parent/'original-validate-ledger-busy-01';fresh.mkdir()
    request_id=owners.ledger_busy_request_attempt_id(frozen,'e'*64,1)
    context={'schema':owners.LEDGER_BUSY_CONTEXT_SCHEMA,'index':1,'authorization_epoch_sha256':'e'*64,
        'expires_at':profile['expires_at'],'freeze_sha256':'f'*64,'source_commitment':frozen['source_commitment'],
        'request_attempt_id':request_id,'prior_operational_ref':prior}
    ticket=put(fresh.parent/'ledger-busy-slot-01.json',dict(context,schema='agent-supervisor/spent-validation-ledger-busy-slot@1',
        invocation=str(fresh),spent_at=owners.time.time(),attempt_number=frozen['binding']['attempt_number'],attempt_token=frozen['binding']['attempt_id']))
    context['spent_slot_ref']=ticket;context_path=fresh/'ledger-busy-generation.json';put(context_path,context)
    original=copy.deepcopy(frozen)
    assert owners.validation_request_generation(policy,frozen,fresh,live=True)==request_id
    policy_pin=put(fresh/'generation-policy.json',policy)
    fresh_freeze=put(fresh.parent/'generation-candidate-ready.json',frozen)
    plan=owners.validation_plan(policy,policy_pin,frozen,fresh_freeze,s.root,fresh)
    assert plan['attempt_directory'].endswith('/'+request_id)
    assert plan['required_candidate_bodies']==[str(s.root/'source.py')]
    plan_pin=put(fresh/'validation-plan.json',plan)
    request=owners.validation_execution_request(policy,plan,plan_pin)
    public=owners.publish_validation_request(policy,frozen,fresh,put(fresh/'execution-request.json',request))
    assert public['path']==str(owners.validation_control_directory(policy,frozen,fresh)/'execution-request.json')
    assert '/'+request_id+'/'in public['path'] and frozen==original
    assert owners.original_execution_request(fresh)==public and checked
    for changed in [dict(context,authorization_epoch_sha256='0'*64),dict(context,spent_slot_ref=prior)]:
        context_path.chmod(0o600);context_path.write_bytes(owners.canonical(changed)+b'\n');context_path.chmod(0o400)
        with pytest.raises((ValueError,KeyError)):owners.validation_request_generation(policy,frozen,fresh)




def test_ledger_busy_generation_refuses_partial_or_rebound_authoritative_ticket(plan_scope,monkeypatch):
    s=plan_scope;policy=_ordinary_v5_policy(s.policy);frozen=copy.deepcopy(s.frozen);frozen['freeze_sha256']='f'*64
    policy['validation']['validation_ledger_busy_retry']={'schema':owners.LEDGER_BUSY_RETRY_SCHEMA,
        'max_additional_invocations':2,'cooldown_seconds':60,'authorization_epoch_sha256':'e'*64,'expires_at':owners.time.time()+1000}
    prior=put(s.invocation/'prior-operational.json',{}) # Never reached before strict ticket rejection.
    fresh=s.invocation.parent/'original-validate-ledger-busy-01';fresh.mkdir()
    context={'schema':owners.LEDGER_BUSY_CONTEXT_SCHEMA,'index':1,'authorization_epoch_sha256':'e'*64,
        'expires_at':policy['validation']['validation_ledger_busy_retry']['expires_at'],'freeze_sha256':'f'*64,
        'source_commitment':frozen['source_commitment'],'request_attempt_id':owners.ledger_busy_request_attempt_id(frozen,'e'*64,1),
        'prior_operational_ref':prior}
    ticket=dict(context,schema='agent-supervisor/spent-validation-ledger-busy-slot@1',invocation=str(fresh),
        spent_at=owners.time.time(),attempt_number=1,attempt_token=frozen['binding']['attempt_id'])
    path=fresh.parent/'ledger-busy-slot-01.json';context_path=fresh/'ledger-busy-generation.json'
    for index,changed in enumerate([{k:v for k,v in ticket.items()if k!='attempt_number'},dict(ticket,extra=True),
            dict(ticket,attempt_number=True),dict(ticket,attempt_token='0'*32)]):
        raw=owners.canonical(changed)+b'\n'
        if path.exists():path.chmod(0o600)
        path.write_bytes(raw);path.chmod(0o400)
        current=dict(context,spent_slot_ref=pin(path))
        if context_path.exists():context_path.chmod(0o600)
        context_path.write_bytes(owners.canonical(current)+b'\n');context_path.chmod(0o400)
        with pytest.raises(ValueError,match='original immutable spent slot'):
            owners.validation_request_generation(policy,frozen,fresh)



def test_ledger_busy_REAL_SDK_reader_requires_complete_typed_Source2_wait_request_and_no_grant(plan_scope,monkeypatch):
    """Real SDK metadata readers; ONLY CLI classifier verdict is modeled.

    Every original process/Source/native/PID/clock statement is fictional DATA.
    No lifecycle/owner API, Source lease, native grant or process is invoked.
    """
    s=plan_scope;policy=_ordinary_v5_policy(s.policy)
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    producer=pin(Path(owners.__file__).absolute());engine_ref=pin(Path(engine.__file__).absolute())
    policy['source_descriptors']=[*policy['source_descriptors'],producer,engine_ref]
    policy_pin=put(s.invocation/'operational-policy.json',policy)
    plan=owners.validation_plan(policy,policy_pin,s.frozen,s.freeze_pin,s.root,s.invocation)
    plan_pin=put(s.invocation/'validation-plan.json',plan)
    request=owners.validation_execution_request(policy,plan,plan_pin)
    request_ref=owners.publish_validation_request(policy,s.frozen,s.invocation,put(s.invocation/'execution-request.json',request))
    Path(request['attempt_directory']).mkdir(parents=True)
    identity={'pid':987654321,'birth':'12345'}
    life={'schema':'owned-validation-pre-grant-ledger-lock-refusal/v1','stage':'owner_open_ledger_lock_refused_before_record'}
    terminal={'schema':'owned-validation-failure-terminal/v1','request_sha256':request_ref['sha256'],
        'completed':False,'admitted':False,'Lean_admitted':False,'failure':{'error_type':'DaemonResourceError','error':'disk ledger lock unavailable'},
        'reservation':{'status':'not_entered'},'process_cleanup_error':None,'owner_exit_error':None,'source_custody_lifecycle':life}
    observed={'schema':'command-execution-observation/v1','completion_kind':'completed','process_pid':identity['pid'],
        'process_returncode':1,'leader_waited':True,'output_capture':'complete','work_deadline_monotonic':970.,
        'cleanup_deadline_monotonic':980.,'observed_at_monotonic':710.}
    execution={'schema':'agent-supervisor/v5-runner-execution-observation@1','request':request_ref,'engine_source':engine_ref,
        'exact_command_result':True,'exact_execution_observation':True,'execution_observation':observed,
        'public_returncode':1,'actual_process_pid':identity['pid'],'actual_process_returncode':1,'actual_Popen_returned':True,
        'spawn_attempted':True,'process_count':1,'runner_identity':identity,'identity_observation_error':None,
        'engine_returned':True,'engine_started_at_monotonic':700.,'engine_finished_at_monotonic':720.,
        'spawn_returned_at_monotonic':702.,'identity_observed_at_monotonic':704.,'authority_deadline_monotonic':1000.,
        'work_deadline_monotonic':970.,'cleanup_deadline_monotonic':980.,'requested_nominal_window_seconds':300,
        'window_scope':'FICTIONAL UNIT DATA; no process executed','source_closed':False,'native_closed':False,
        'storage_release_authorized':False,'retry_authorized':False}
    execution_ref=put(s.invocation/'runner-execution-observation.json',execution)
    argv=[policy['python'],'-I','-B',str(Path(policy['datasets_root'])/'scripts/ops/legal_ir/run_owned_validation.py'),
        '--request',request_ref['path'],'--expected-request-sha256',request_ref['sha256'],'--resource-ledger',policy['validation']['resource_ledger']]
    wait={'request':request_ref,'actual_Popen_returned':True,'actual_process_waited':True,'original_process_owner_returned':True,
        'returncode':1,'actual_process_returncode':1,'engine_source':engine_ref,'runner_identity':identity,
        'execution_observation':execution_ref,'argv':argv,'stdout':json.dumps(terminal)}
    wait_ref=put(s.invocation/'runner-wait.json',wait)
    receipts=[{'schema_version':'autoencoder-source-lease/v1','canonical_root':str(root),'cooperative':True,
        'lock_path':str(root/'workspace/control-plane/source-leases/source.lock'),'mode':'read','waited_seconds':0.}
        for root in (Path(policy['datasets_root']),s.root)]
    custody={'schema':'agent-supervisor/v5-validation-parent-source-custody@2','closure_kind':'original_no_child_no_native_grant',
        'storage_status':'not_entered','source_closed':True,'native_closed':True,'storage_release_authorized':False,
        'retry_authorized':False,'original_wait':wait_ref,'source_receipts':receipts,
        'authority_deadline_monotonic':1000.,'execution_observation':execution_ref,'source_exit_finished_at_monotonic':730.,
        'work_deadline_monotonic':970.,'cleanup_deadline_monotonic':980.}
    closeout={'schema':'agent-supervisor/v5-validation-parent-source-closeout@2','request':request_ref,
        'original_wait':wait_ref,'producer':producer,'parent_source_custody':custody}
    closeout_ref=put(s.invocation/'parent-source-closeout.json',closeout)
    verdict={'verdict':'proven_original_no_child_no_native_grant','source_custody_closed':True,'storage_release_authorized':False}
    calls=[]
    def modeled_classifier(term,raw,**kw):
        calls.append((term,raw,kw));return dict(verdict)
    monkeypatch.setattr(owners,'_current_v5_cli_classifier',lambda policy:modeled_classifier)
    before={p:p.read_bytes()for p in s.invocation.iterdir()if p.is_file()}
    result=owners.verify_v5_ledger_busy_closeout(policy,request_ref,s.invocation,closeout_ref)
    assert result['storage_status']=='not_entered'and result['semantic_negative']is False and result['retry_authorized']is False
    assert calls and calls[-1][2]['returncode']==1
    assert {p:p.read_bytes()for p in before}==before and not tuple(Path(request['attempt_directory']).iterdir())
    verdict['source_custody_closed']=False
    with pytest.raises(ValueError,match='no-grant classifier refused'):
        owners.verify_v5_ledger_busy_closeout(policy,request_ref,s.invocation,closeout_ref)
    verdict['source_custody_closed']=True
    # Rehash EVERY dependent link so the original typed core sees the bool fault.
    execution['execution_observation']['process_returncode']=True
    execution_ref=_archive_sdk_replace(s.invocation/'runner-execution-observation.json',execution)
    wait['execution_observation']=execution_ref;wait_ref=_archive_sdk_replace(s.invocation/'runner-wait.json',wait)
    custody.update(original_wait=wait_ref,execution_observation=execution_ref)
    closeout.update(original_wait=wait_ref,parent_source_custody=custody)
    closeout_ref=_archive_sdk_replace(s.invocation/'parent-source-closeout.json',closeout)
    with pytest.raises(ValueError,match='complete typed terminal'):
        owners.verify_v5_ledger_busy_closeout(policy,request_ref,s.invocation,closeout_ref)


# V5 namespace projection controls. Real files/manifests and readers below are
# engineering DATA; no fixture produces an original Source/native receipt.
@pytest.fixture
def ordinary_v5_control_namespace_scope(metadata_scope):
    import importlib.util
    s=metadata_scope; policy=_ordinary_v5_policy(s.policy); d=Path(policy['datasets_root'])
    worker_path=s.control/'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py'
    specification=importlib.util.spec_from_file_location('_ordinary_v5_namespace_actual_worker_DATA',worker_path)
    worker=importlib.util.module_from_spec(specification);specification.loader.exec_module(worker)
    assert Path(worker.__file__)==worker_path and worker.validation_worker.__code__.co_filename==str(worker_path)
    directory=d/'ipfs_datasets_py/logic/backends'
    leaf=put(directory/'process.py',b'# Explicit inert namespace DATA leaf.\n')
    info=directory.stat(); profile=dict(fullname='ipfs_datasets_py.logic.backends',path=str(directory),
        device=info.st_dev,inode=info.st_ino,initializer_absent=True)
    manifest=json.loads(Path(policy['control_release']['manifest']['path']).read_bytes())
    manifest['namespace_profiles']=[profile]
    policy['control_release']['manifest']=put(s.control/'v5-namespace-control-manifest.json',manifest)
    rows={row['path']:row for row in policy['source_descriptors']};rows[leaf['path']]=leaf
    for spelling in sorted(owners._v5_lifecycle_owner_paths(d)):
        path=Path(spelling)
        if not path.exists():put(path,b'# Inert lifecycle-pin DATA.\n')
        rows[spelling]=pin(path)
    policy['source_descriptors']=list(rows.values())
    policy['validation']['selected_checks']=[policy['validation']['trusted_tests'][0]['path']+'::test_original_case']
    frozen=dict(binding=dict(namespace='ptr_lane_0',board_id='legal-ir',attempt_id='a'*32),
        manifest={'source.py':dict(bytes=s.selected_source['bytes'],sha256=s.selected_source['sha256'],executable=False)},
        changed_paths=['source.py'],source_commitment='c'*64)
    invocation=s.state/'v5-namespace-invocation';invocation.mkdir()
    policy_pin=put(s.state/'v5-namespace-policy.json',policy)
    freeze_pin=put(s.state/'v5-namespace-freeze.json',frozen)
    return SimpleNamespace(worker=worker,scope=s,policy=policy,policy_pin=policy_pin,frozen=frozen,
        freeze_pin=freeze_pin,invocation=invocation,profile=profile,manifest=manifest,leaf=leaf)


def _ordinary_v5_namespace_plan(scope):
    return owners.validation_plan(scope.policy,scope.policy_pin,scope.frozen,scope.freeze_pin,
        scope.scope.source,scope.invocation)


def test_v5_ordinary_plan_inherits_exact_existing_control_namespace(ordinary_v5_control_namespace_scope):
    s=ordinary_v5_control_namespace_scope; before=copy.deepcopy(s.policy)
    plan=_ordinary_v5_namespace_plan(s)
    assert plan.get('normal_namespace_profiles')==[s.profile]
    assert s.policy==before and 'normal_namespace_profiles' not in s.policy['validation']
    assert plan['required_candidate_bodies']==[str(s.scope.source/'source.py')]
    plan['normal_namespace_profiles'][0]['inode']+=1
    assert s.policy==before


def test_v5_namespace_request_keeps_original18_fields_and16MiB(ordinary_v5_control_namespace_scope):
    s=ordinary_v5_control_namespace_scope;plan=_ordinary_v5_namespace_plan(s)
    plan_ref=put(s.invocation/'validation-plan.json',plan)
    request=owners.validation_execution_request(s.policy,plan,plan_ref)
    assert request['schema']=='owned-validation-request/v5' and len(request)==18
    assert 'normal_namespace_profiles' not in request and request['declared_external_writable_paths']==[]
    assert (request['storage_bytes'],request['cpu_slots'],request['memory_mb'],request['child_process_slots'],
        request['timeout_seconds'])==(16777216,1,1536,4,300)


def test_v5_worker_reads_exact_control_namespace_before_normal_imports(ordinary_v5_control_namespace_scope):
    s=ordinary_v5_control_namespace_scope;plan=_ordinary_v5_namespace_plan(s)
    # The whole maintained worker was normally loaded from this actual sealed
    # control file; no __file__, receipt or function-source alias is assigned.
    profiles=s.worker.validation_namespace_profiles(plan,s.policy)
    guard=s.worker.PinnedSources(plan['source_pins'],[s.scope.source,Path(s.policy['datasets_root']),s.scope.control],profiles)
    assert guard.normal_namespace_profiles=={s.profile['fullname']:s.profile}
    assert guard.verify_normal_namespaces()=={}


@pytest.mark.parametrize('mutation',['missing','empty','extra','inode','bool_inode','foreign_name'])
def test_v5_worker_refuses_missing_or_changed_manifest_projection(ordinary_v5_control_namespace_scope,mutation):
    s=ordinary_v5_control_namespace_scope;plan=_ordinary_v5_namespace_plan(s)
    if mutation=='missing':plan.pop('normal_namespace_profiles')
    elif mutation=='empty':plan['normal_namespace_profiles']=[]
    elif mutation=='extra':plan['normal_namespace_profiles'].append(copy.deepcopy(s.profile))
    elif mutation=='inode':plan['normal_namespace_profiles'][0]['inode']+=1
    elif mutation=='bool_inode':plan['normal_namespace_profiles'][0]['inode']=True
    else:plan['normal_namespace_profiles'][0]['fullname']='ipfs_datasets_py.logic.unapproved'
    with pytest.raises(ValueError,match='namespace differs from the immutable control manifest'):
        s.worker.validation_namespace_profiles(plan,s.policy)


def test_v5_control_namespace_rechecks_actual_initializer_absence(ordinary_v5_control_namespace_scope):
    s=ordinary_v5_control_namespace_scope
    put(Path(s.profile['path'])/'__init__.py',b'# Unexpected real namespace initializer.\n')
    with pytest.raises(ValueError,match='namespace acquired an initializer'):
        _ordinary_v5_namespace_plan(s)


@pytest.mark.parametrize('mutation',['missing_report','changed_report','missing_plan','changed_plan'])
def test_v5_saved_namespace_report_must_match_same_control_declaration(ordinary_v5_control_namespace_scope,mutation):
    s=ordinary_v5_control_namespace_scope;plan=_ordinary_v5_namespace_plan(s)
    report={'declared_normal_namespace_profiles':copy.deepcopy(plan['normal_namespace_profiles'])}
    owners._v5_validation_namespace_evidence(s.policy,plan,report)
    if mutation=='missing_report':report.clear()
    elif mutation=='changed_report':report['declared_normal_namespace_profiles'][0]['inode']+=1
    elif mutation=='missing_plan':plan.pop('normal_namespace_profiles')
    else:plan['normal_namespace_profiles'][0]['inode']+=1
    with pytest.raises(ValueError,match='V5 immutable control namespace declaration differs'):
        owners._v5_validation_namespace_evidence(s.policy,plan,report)


@pytest.fixture
def isolated_namespace_plan_scope(tmp_path):
    root=tmp_path/'namespace-legacy-plan'
    root.mkdir()
    return plan_scope.__wrapped__(root)


def test_v5_empty_control_projection_and_legacy_policy_comparison_stay_empty(metadata_scope,isolated_namespace_plan_scope):
    plan_scope=isolated_namespace_plan_scope
    policy=_ordinary_v5_policy(metadata_scope.policy)
    assert owners._v5_control_namespace_profiles(policy)==[]
    owners._v5_validation_namespace_evidence(policy,{}, {})
    s=plan_scope;policy=_ordinary_v5_policy(s.policy)
    plan=owners.validation_plan(policy,s.policy_pin,s.frozen,s.freeze_pin,s.root,s.invocation)
    assert 'normal_namespace_profiles' not in plan
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    assert worker.validation_namespace_profiles({}, {'validation':{}})==[]
    with pytest.raises(ValueError,match='original declared normal namespace profiles differ'):
        worker.validation_namespace_profiles({'normal_namespace_profiles':[{'unexpected':'DATA'}]}, {'validation':{}})


# Explicit immutable-source qualification controls; inner lifecycle facts are DATA.
def _immutable_source_qualification_RED_scope(scope):
    policy=_ordinary_v5_policy(scope.policy)
    origin=Path(owners.__file__).absolute().parents[3]
    original_manifest_pin=copy.deepcopy(policy['control_release']['manifest'])
    original_manifest=Path(original_manifest_pin['path']).read_bytes()
    manifest=json.loads(original_manifest)
    roles=[]
    for relative in (
            'ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py',
            'ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py'):
        roles.append(put(scope.control/relative,(origin/relative).read_bytes()))
    manifest['files']=sorted(manifest['files']+roles,key=lambda row:row['path'])
    policy['control_release']['manifest']=put(
        scope.control/'immutable-source-qualification-control-manifest.json',manifest)
    assert len(manifest['files'])==9
    assert Path(original_manifest_pin['path']).read_bytes()==original_manifest
    d=Path(policy['datasets_root'])
    rows={row['path']:row for row in policy['source_descriptors']}
    for spelling in sorted(owners._v5_lifecycle_owner_paths(d)):
        path=Path(spelling)
        if not path.exists():
            put(path,b'# Explicit unexecuted lifecycle-pin DATA.\n')
        rows[spelling]=pin(path)
    policy['source_descriptors']=list(rows.values())
    binding=dict(namespace='ptr_lane_0',board_id='legal-ir',attempt_id='a'*32)
    population={str(Path(row['path']).relative_to(scope.control)):
        dict(bytes=row['bytes'],sha256=row['sha256'],
             executable=bool(Path(row['path']).stat().st_mode&0o111))
        for row in manifest['files']}
    snapshot=dict(schema='agent-supervisor/immutable-source-snapshot@1',
        binding=binding,source_root=str(scope.control),manifest=population,
        source_commitment=owners.digest(dict(source_root=str(scope.control),manifest=population)))
    snapshot_pin=put(scope.state/'immutable-source-snapshot.json',snapshot)
    required=sorted(roles,key=lambda row:row['path'])
    profile=dict(schema='agent-supervisor/immutable-source-qualification@1',
        snapshot=snapshot_pin,required_source_bodies=required)
    invocation=scope.state/'immutable-source-qualification-invocation'
    invocation.mkdir()
    return SimpleNamespace(scope=scope,policy=policy,snapshot=snapshot,
        snapshot_pin=snapshot_pin,profile=profile,required=required,invocation=invocation)


def test_immutable_source_qualification_keeps_default_empty_change_denial(metadata_scope):
    s=_immutable_source_qualification_RED_scope(metadata_scope)
    before=copy.deepcopy(s.policy)
    control=owners.checked_control_release(s.policy,current_owner=False)
    assert len(control['pins'])==9
    assert {str(s.scope.control/relative):row['sha256']
            for relative,row in s.snapshot['manifest'].items()}==control['pins']
    legacy_manifest={'source.py':dict(bytes=s.scope.selected_source['bytes'],
        sha256=s.scope.selected_source['sha256'],executable=False)}
    legacy=dict(binding=copy.deepcopy(s.snapshot['binding']),manifest=legacy_manifest,
        changed_paths=[],source_commitment=owners.digest(legacy_manifest))
    legacy_pin=put(s.scope.state/'legacy-empty-change-DATA.json',legacy)
    legacy_policy_pin=put(s.scope.state/'legacy-empty-change-policy.json',s.policy)
    with pytest.raises(ValueError,match='actually changed Python body'):
        owners.validation_plan(s.policy,legacy_policy_pin,legacy,legacy_pin,
            s.scope.source,s.invocation)
    assert s.policy==before
    policy=copy.deepcopy(s.policy)
    policy['validation']['source_qualification']=copy.deepcopy(s.profile)
    policy_pin=put(s.scope.state/'immutable-source-qualification-policy.json',policy)
    # Baseline RED must stop here: existing SDK rejects this closed new profile.
    owners.checked_validation_profile(policy,control)
    snapshot_before=Path(s.snapshot_pin['path']).read_bytes()
    plan=owners.validation_plan(policy,policy_pin,s.snapshot,s.snapshot_pin,
        s.scope.control,s.invocation)
    assert plan['candidate_root']==str(s.scope.control)
    assert plan['required_candidate_bodies']==[]
    assert plan['required_source_bodies']==[row['path']for row in s.required]
    assert plan['source_qualification']==s.profile
    assert plan['attempt_directory'].endswith('/ptr_lane_0/legal-ir/'+'a'*32)
    assert 'changed_paths'not in s.snapshot and 'stage'not in s.snapshot
    assert Path(s.snapshot_pin['path']).read_bytes()==snapshot_before
    plan_pin=put(s.invocation/'validation-plan.json',plan)
    request=owners.validation_execution_request(policy,plan,plan_pin)
    assert len(request)==18 and request['schema']=='owned-validation-request/v5'
    assert {key:request[key]for key in ('storage_bytes','cpu_slots','memory_mb',
        'child_process_slots','timeout_seconds')}==dict(storage_bytes=16777216,
            cpu_slots=1,memory_mb=1536,child_process_slots=4,timeout_seconds=300)
    assert request['ordinary_scratch_profile']=={'schema':'owned-validation-ordinary-scratch-live/v1'}
    assert request['declared_external_writable_paths']==[]
    assert request['cwd']==str(s.scope.control)
    assert request['selected_checks']==policy['validation']['selected_checks']
    assert s.policy==before and 'source_qualification'not in s.policy['validation']


def _immutable_qualification_ready_plan(metadata_scope):
    s=_immutable_source_qualification_RED_scope(metadata_scope)
    policy=copy.deepcopy(s.policy)
    policy['validation']['source_qualification']=copy.deepcopy(s.profile)
    policy_pin=put(s.scope.state/'qualified-owner-policy.json',policy)
    plan=owners.validation_plan(policy,policy_pin,s.snapshot,s.snapshot_pin,s.scope.control,s.invocation)
    return s,policy,plan


def _immutable_qualification_actual_worker(scope):
    import importlib.util
    path=scope.scope.control/'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py'
    spec=importlib.util.spec_from_file_location('_immutable_qualification_actual_worker_DATA',path)
    worker=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    assert Path(worker.__file__)==path
    assert worker.checked_source_qualification_plan.__code__.co_filename==str(path)
    return worker


@pytest.mark.parametrize('damage',['false_profile','mixed_retry','mixed_v3','missing_health',
    'duplicate_subject','unordered_subjects','snapshot_extra_field','binding_extra_field',
    'missing_source','extra_source','different_root','wrong_commitment','bool_bytes','nonpython_subject'])
def test_immutable_qualification_profile_rejects_changed_authority(metadata_scope,damage):
    s=_immutable_source_qualification_RED_scope(metadata_scope)
    policy=copy.deepcopy(s.policy);profile=copy.deepcopy(s.profile);snapshot=copy.deepcopy(s.snapshot)
    if damage=='false_profile':profile=False
    elif damage=='mixed_retry':policy['validation']['validation_ledger_busy_retry']={}
    elif damage=='mixed_v3':policy['validation']['external_fixture_profile']={}
    elif damage=='missing_health':profile['required_source_bodies']=[p for p in s.required if not p['path'].endswith('/native_executor_health.py')]
    elif damage=='duplicate_subject':profile['required_source_bodies'].append(profile['required_source_bodies'][-1])
    elif damage=='unordered_subjects':profile['required_source_bodies'].reverse()
    elif damage=='snapshot_extra_field':snapshot['changed_paths']=[]
    elif damage=='binding_extra_field':snapshot['binding']['attempt_number']=1
    elif damage=='missing_source':snapshot['manifest'].pop(next(iter(snapshot['manifest'])))
    elif damage in ('extra_source','nonpython_subject'):
        extra=put(s.scope.control/('unlisted.py'if damage=='extra_source'else'body-data.txt'),b'# DATA only\n')
        snapshot['manifest'][Path(extra['path']).name]=dict(bytes=extra['bytes'],sha256=extra['sha256'],executable=False)
        if damage=='nonpython_subject':
            manifest=json.loads(Path(policy['control_release']['manifest']['path']).read_bytes())
            manifest['files']=sorted(manifest['files']+[extra],key=lambda r:r['path'])
            policy['control_release']['manifest']=put(s.scope.control/'nonpython-body-control-manifest.json',manifest)
            profile['required_source_bodies']=sorted(profile['required_source_bodies']+[extra],key=lambda r:r['path'])
    elif damage=='different_root':snapshot['source_root']=str(s.scope.source)
    elif damage=='wrong_commitment':snapshot['source_commitment']='0'*64
    else:snapshot['manifest'][next(iter(snapshot['manifest']))]['bytes']=True
    if isinstance(profile,dict):
        if damage!='wrong_commitment':snapshot['source_commitment']=owners.digest(dict(source_root=snapshot['source_root'],manifest=snapshot['manifest']))
        profile['snapshot']=put(s.scope.state/'mutated-immutable-snapshot.json',snapshot)
    policy['validation']['source_qualification']=profile
    with pytest.raises(ValueError):owners.checked_source_qualification(policy)


@pytest.mark.parametrize('damage',['extra_source_pin','omitted_source_pin','extra_control_pin','wrong_attempt','different_selection'])
def test_immutable_qualification_reconstructs_complete_plan(metadata_scope,damage):
    s,policy,plan=_immutable_qualification_ready_plan(metadata_scope)
    context=owners._source_qualification_plan(policy,plan,s.invocation)
    assert context['snapshot_ref']==s.snapshot_pin
    altered=copy.deepcopy(plan)
    if damage=='extra_source_pin':altered['source_pins']['/tmp/unlisted-source.py']='0'*64
    elif damage=='omitted_source_pin':altered['source_pins'].pop(s.required[0]['path'])
    elif damage=='extra_control_pin':altered['original_public_control_pins']['/tmp/unlisted-control.py']='0'*64
    elif damage=='wrong_attempt':altered['attempt_directory']+='-unissued'
    else:altered['selected_checks']=[altered['selected_checks'][0]+'-unissued']
    with pytest.raises(ValueError,match='complete immutable source qualification plan differs'):
        owners._source_qualification_plan(policy,altered,s.invocation)


@pytest.mark.parametrize('damage',['profile','required','freeze','commitment','semantic_negative','raw_compile','execution'])
def test_immutable_qualification_report_requires_exact_raw_execution(metadata_scope,damage):
    s,policy,plan=_immutable_qualification_ready_plan(metadata_scope)
    context=owners._source_qualification_plan(policy,plan,s.invocation)
    report=dict(source_qualification=copy.deepcopy(s.profile),required_source_bodies=[p['path']for p in s.required],
        freeze=s.snapshot_pin,source_commitment=s.snapshot['source_commitment'],semantic_negative=False,negative_profile=None,
        raw_compiled_sources={p['path']:p['sha256']for p in s.required},
        executed_source_buffers={p['path']:p['sha256']for p in s.required})
    owners._source_qualification_report(context,plan,report)
    if damage=='profile':report['source_qualification']=False
    elif damage=='required':report['required_source_bodies']=[]
    elif damage=='freeze':report['freeze']={}
    elif damage=='commitment':report['source_commitment']='0'*64
    elif damage=='semantic_negative':report['semantic_negative']=True
    elif damage=='raw_compile':report['raw_compiled_sources'].pop(s.required[0]['path'])
    else:report['executed_source_buffers'].pop(s.required[0]['path'])
    with pytest.raises(ValueError):owners._source_qualification_report(context,plan,report)


@pytest.mark.parametrize('damage',['candidate_bodies','source_bodies','snapshot','missing_pin','expanded_budget'])
def test_immutable_qualification_worker_rechecks_independently(metadata_scope,damage):
    s,policy,plan=_immutable_qualification_ready_plan(metadata_scope)
    worker=_immutable_qualification_actual_worker(s)
    assert worker.checked_source_qualification_plan(plan,policy)==s.profile
    changed=copy.deepcopy(plan);changed_policy=copy.deepcopy(policy)
    if damage=='candidate_bodies':changed['required_candidate_bodies']=[s.required[0]['path']]
    elif damage=='source_bodies':changed['required_source_bodies']=[]
    elif damage=='snapshot':changed['freeze']={}
    elif damage=='missing_pin':changed['source_pins'].pop(s.required[0]['path'])
    else:changed_policy['validation']['storage_bytes']+=1
    with pytest.raises(ValueError):worker.checked_source_qualification_plan(changed,changed_policy)


def test_immutable_qualification_legacy_fields_and_adoption_stay_closed(metadata_scope,monkeypatch):
    s,policy,plan=_immutable_qualification_ready_plan(metadata_scope)
    with pytest.raises(ValueError,match='repair plan cannot carry'):
        owners._source_qualification_plan(s.policy,plan,s.invocation)
    with pytest.raises(ValueError,match='repair report cannot carry'):
        owners._source_qualification_report(None,{},dict(source_qualification=s.profile))
    worker=_immutable_qualification_actual_worker(s)
    with pytest.raises(ValueError,match='repair plan cannot carry'):
        worker.checked_source_qualification_plan(plan,s.policy)
    with pytest.raises(ValueError,match='no adoption profile'):
        owners.current_profile_contract(policy,s.snapshot)
    monkeypatch.setattr(owners,'checked_policy',lambda *a,**k:policy)
    args=SimpleNamespace(policy=Path('/unused-policy'),policy_sha256='a'*64)
    for entry in (owners.verify_validation,owners.verify_negative_validation,owners.verify_operational_validation,owners.adopt):
        with pytest.raises(ValueError,match='immutable qualification cannot enter repair'):entry(args)
    monkeypatch.setattr(candidate,'_frozen',lambda *a,**k:None)
    session=SimpleNamespace(stage='candidate_ready')
    with pytest.raises(ValueError,match='original candidate validation receipt required'):
        candidate.record_validation(session,dict(schema=owners.SOURCE_QUALIFICATION_SCHEMA,passed=True))
    assert session.stage=='candidate_ready'


def test_immutable_qualification_pregrant_failure_does_not_publish_repair_retry(metadata_scope,monkeypatch):
    s,policy,plan=_immutable_qualification_ready_plan(metadata_scope)
    policy_pin=pin(Path(plan['policy']['path']))
    args=SimpleNamespace(policy=Path(policy_pin['path']),policy_sha256=policy_pin['sha256'],
        freeze_path=Path(s.snapshot_pin['path']),source_commitment=s.snapshot['source_commitment'],
        candidate_root=s.scope.control,invocation_root=s.invocation,receipt_path=s.invocation/'source-only-receipt.json')
    monkeypatch.setattr(owners,'checked_policy',lambda *a,**k:policy)
    witness=dict(parent_source_custody=dict(source_closed=True,native_closed=True,
        closure_kind='original_no_child_no_native_grant'),parent_source_closeout={'fictional':'DATA'})
    monkeypatch.setattr(owners,'original_runner_wait',lambda *a,**k:(SimpleNamespace(returncode=1,stdout='{}',stderr=''),witness))
    for name in ('operational_ledger_busy_material','capture_negative_validation'):
        monkeypatch.setattr(owners,name,lambda *a,**k:pytest.fail('immutable qualification reached repair publisher'))
    assert owners.validate(args)==1
    failed=json.loads((s.invocation/'failed-source-qualification.json').read_bytes())
    assert failed['schema']=='agent-supervisor/v5-source-qualification-failed-source-closeout@1'
    assert failed['semantic_negative']is False and failed['retry_authorized']is False
    assert not (s.invocation/'operational-ledger-busy.json').exists()
    assert not args.receipt_path.exists()


def _immutable_qualification_bound_request_DATA(metadata_scope):
    """Real sealed metadata; all process/native statements below are fictional DATA."""
    s=_immutable_source_qualification_RED_scope(metadata_scope)
    policy=copy.deepcopy(s.policy);policy['validation']['source_qualification']=copy.deepcopy(s.profile)
    engine=put(s.scope.state/'fictional-original-engine-DATA.py',b'# Unexecuted modeled wait source.\n')
    policy['source_descriptors'].append(engine)
    policy_pin=put(s.scope.state/'saved-qualification-policy.json',policy)
    plan=owners.validation_plan(policy,policy_pin,s.snapshot,s.snapshot_pin,s.scope.control,s.invocation)
    plan_pin=put(s.invocation/'validation-plan.json',plan)
    request=owners.validation_execution_request(policy,plan,plan_pin)
    local=put(s.invocation/'execution-request.json',request)
    public=owners.publish_validation_request(policy,s.snapshot,s.invocation,local)
    args=SimpleNamespace(policy=Path(policy_pin['path']),policy_sha256=policy_pin['sha256'],
        freeze_path=Path(s.snapshot_pin['path']),source_commitment=s.snapshot['source_commitment'],
        candidate_root=s.scope.control,invocation_root=s.invocation,receipt_path=s.invocation/'source-qualification-receipt.json')
    return s,policy,plan,plan_pin,public,args,engine


def test_immutable_qualification_public_verifier_rechecks_saved_DATA_receipt(metadata_scope,monkeypatch):
    s,policy,plan,plan_pin,request,args,engine=_immutable_qualification_bound_request_DATA(metadata_scope)
    terminal=put(s.invocation/'original-terminal.json',dict(fictional_original_terminal_DATA=True))
    parent=put(s.invocation/'parent-source-closeout.json',dict(fictional_Source2_DATA=True))
    summary=dict(cases=1,phases=3,reservation_id='e'*32,native_lease_id='f'*32,natural_phase_counter_verified=True)
    receipt=dict(schema=owners.SOURCE_QUALIFICATION_SCHEMA,snapshot=s.snapshot_pin,source_qualification=s.profile,
        source_commitment=s.snapshot['source_commitment'],passed=True,source_closed=True,native_closed=True,
        processes_closed=True,storage_closed=True,cleanup_errors=[],original_receipt_reference=terminal['path'],
        original_receipt_descriptor=terminal,verified_checks=summary,parent_source_closeout=parent,
        repair_authority=False,adoption_authority=False,activation_authority=False,completion_authority=False)
    put(args.receipt_path,receipt)
    calls=[]
    # Narrow modeled original-evidence/parent seams; this is not a real Native success.
    def evidence(p,work,pin,original,invocation):
        assert p==policy and work==plan and pin==request and original=={'fictional_original_terminal_DATA':True}
        assert invocation==s.invocation;calls.append('evidence');return copy.deepcopy(summary)
    def closeout(p,pin,invocation,proof,actual):
        assert p==policy and pin==request and proof==parent and actual==summary
        calls.append('parent')
    monkeypatch.setattr(owners,'checked_policy',lambda *a,**k:policy)
    monkeypatch.setattr(owners,'install_control_guard',lambda *a,**k:None)
    monkeypatch.setattr(owners,'verify_validation_evidence',evidence)
    monkeypatch.setattr(owners,'verify_v5_parent_source_closeout',closeout)
    def command(path):
        return ['verify-source-qualification','--policy',str(args.policy),'--policy-sha256',args.policy_sha256,
            '--candidate-root',str(args.candidate_root),'--freeze-path',str(args.freeze_path),
            '--source-commitment',args.source_commitment,'--invocation-root',str(args.invocation_root),
            '--receipt-path',str(path)]
    assert owners.main(command(args.receipt_path))==0 and calls==['evidence','parent']
    for index,(field,value) in enumerate((('schema',candidate.VALIDATION_SCHEMA),('repair_authority',True),
            ('snapshot',{}),('source_closed',False),('source_commitment','0'*64),('activation_authority',True))):
        bad=copy.deepcopy(receipt);bad[field]=value
        path=s.invocation/('bad-source-receipt-%d.json'%index);put(path,bad)
        with pytest.raises(ValueError,match='positive source-only qualification required'):
            owners.main(command(path))
    assert calls==['evidence','parent']


def test_immutable_qualification_retained_failure_requires_original_provenance_DATA(metadata_scope):
    s,policy,plan,plan_pin,request,args,engine=_immutable_qualification_bound_request_DATA(metadata_scope)
    attempt=Path(plan['attempt_directory']);attempt.mkdir(parents=True)
    owner=dict(pid=123,birth='456');child=dict(pid=234,birth='567')
    record=dict(validation_request_sha256=request['sha256'],owner_pid=owner['pid'],owner_birth=owner['birth'],
        native_lease_id='f'*32,reservation_id='e'*32,status='retained',child=child)
    producer=next(pin for pin in policy['source_descriptors']if pin['path'].endswith('/scripts/ops/legal_ir/run_owned_validation.py'))
    terminal=dict(schema='owned-validation-failure-terminal/v1',completed=False,request_sha256=request['sha256'],
        process_cleanup_error=None,owner_exit_error=None,reservation=dict(record=record,
            resource_lease=dict(lease_id=record['native_lease_id'],owner_pid=owner['pid'],released=True),
            status='retained',cleanup_error=None,process_observation=dict(live_processes=0,out_of_group_live_identities=[],tracking_error=False)),
        source_custody_lifecycle=dict(request_sha256=request['sha256'],owner=owner,reservation_id=record['reservation_id'],
            producer=dict(path=producer['path'],sha256=producer['sha256']),events=dict(owner_exit_handling_completed=True),
            source_guards=dict(errors=[],request_unchanged=True,source_pins_unchanged=True,owner_identity_unchanged=True)))
    wait=dict(returncode=1,actual_process_returncode=1,request=request,original_process_owner_returned=True,
        actual_Popen_returned=True,actual_process_waited=True,stdout=owners.canonical(terminal).decode(),
        runner_identity=owner,engine_source=engine)
    report=dict(worker_identity=child,schema=owners.WORK_SCHEMA,passed=False,semantic_negative=False,negative_profile=None,
        startup=dict(python=policy['python'],isolated=1,no_bytecode=True),plan_sha256=plan_pin['sha256'],
        source_READ_closed=True,cleanup_errors=[],freeze=s.snapshot_pin,source_commitment=s.snapshot['source_commitment'],
        source_qualification=s.profile,required_source_bodies=[p['path']for p in s.required],
        full_source_manifest_sha256=owners.digest(plan['source_pins']),full_source_file_count=len(plan['source_pins']))
    report_path=attempt/'exports/trusted-validation.json';put(report_path,report)
    actual=owners._source_qualification_failed_custody(policy,plan,request,s.invocation,terminal,wait)
    assert actual==dict(source_closed=True,native_closed=True,processes_closed=True,
        storage_closed=False,semantic_negative=False,negative_profile=None)
    for bad in (dict(report,startup=dict(python=policy['python'],isolated=0,no_bytecode=True)),
                dict(report,semantic_negative=True)):
        report_path.chmod(0o600);report_path.write_bytes(owners.canonical(bad)+b'\n');report_path.chmod(0o400)
        with pytest.raises(ValueError,match='failure custody remains unknown'):
            owners._source_qualification_failed_custody(policy,plan,request,s.invocation,terminal,wait)
    report_path.chmod(0o600);report_path.write_bytes(owners.canonical(report)+b'\n')
    with pytest.raises(ValueError,match='report must be sealed'):
        owners._source_qualification_failed_custody(policy,plan,request,s.invocation,terminal,wait)


@pytest.fixture(scope="module")
def cold_public_source_order_bundle(tmp_path_factory):
    """Immutable real source copies; children import only, never enter leases."""
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_source_lease as actual_source_owner
    import stat
    origin=Path(owners.__file__).absolute().parents[3]
    actual_D=Path(actual_source_owner.__file__).absolute().parents[3]
    root=tmp_path_factory.mktemp('cold-public-source-order',numbered=False)
    control=root/'C';datasets=root/'D'
    actual_A_relatives=('ipfs_accelerate_py/__init__.py', 'ipfs_accelerate_py/agent_supervisor/__init__.py', 'ipfs_accelerate_py/agent_supervisor/control/__init__.py', 'ipfs_accelerate_py/agent_supervisor/control/control_contracts.py', 'ipfs_accelerate_py/agent_supervisor/control/control_plane.py', 'ipfs_accelerate_py/agent_supervisor/core/__init__.py', 'ipfs_accelerate_py/agent_supervisor/core/wrapper_utils.py', 'ipfs_accelerate_py/agent_supervisor/git_environment.py', 'ipfs_accelerate_py/agent_supervisor/merge/__init__.py', 'ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py', 'ipfs_accelerate_py/agent_supervisor/merge/quarantine_validation.py', 'ipfs_accelerate_py/agent_supervisor/merge/workspace_quarantine.py', 'ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py', 'ipfs_accelerate_py/agent_supervisor/objectives/__init__.py', 'ipfs_accelerate_py/agent_supervisor/objectives/goal_completion.py', 'ipfs_accelerate_py/agent_supervisor/proof/__init__.py', 'ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py', 'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py', 'ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py', 'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py', 'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py', 'ipfs_accelerate_py/agent_supervisor/runtime/__init__.py', 'ipfs_accelerate_py/agent_supervisor/runtime/event_log.py', 'ipfs_accelerate_py/agent_supervisor/self_improvement/__init__.py', 'ipfs_accelerate_py/agent_supervisor/self_improvement/self_improvement_completion.py', 'ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_v2_contracts.py', 'ipfs_accelerate_py/agent_supervisor/task_sources/__init__.py', 'ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/__init__.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/app.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/artifacts.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/auto_commit.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/cli.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/context.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/deterministic_fallback.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/diagnostics.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/file_replacement.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/git_utils.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/history.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/lifecycle_wrapper.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/llm_defaults.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/plans.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/runner.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/specs.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/status.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/task_board.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/typescript.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/worktrees.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/wrapper.py', 'ipfs_accelerate_py/agent_supervisor/validation/__init__.py', 'ipfs_accelerate_py/agent_supervisor/validation/validation_commands.py', 'ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py')
    actual_R_relatives=('ipfs_datasets_py/__init__.py', 'ipfs_datasets_py/ipfs_backend_router.py', 'ipfs_datasets_py/logic/__init__.py', 'ipfs_datasets_py/logic/backends/process.py', 'ipfs_datasets_py/optimizers/__init__.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/__init__.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_daemon_resources.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_external_resource_recovery.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/existing_resource_scheduler.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/parented_phase_resources.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/proof_resource_safety.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/runtime_telemetry.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/validation_pytest.py', 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/validation_run_resources.py', 'ipfs_datasets_py/router_deps.py', 'scripts/__init__.py', 'scripts/ops/__init__.py', 'scripts/ops/legal_ir/__init__.py', 'scripts/ops/legal_ir/run_owned_validation.py')
    files=[put(control/name,(origin/name).read_bytes())for name in actual_A_relatives]
    imports=[put(datasets/name,(actual_D/name).read_bytes())for name in actual_R_relatives]
    # Preserve the REAL vendored bootstrap shim that made D-first unsafe.
    shadow=put(datasets/'ipfs_accelerate_py/__init__.py',(actual_D/'ipfs_accelerate_py/__init__.py').read_bytes())
    namespace=datasets/'ipfs_datasets_py/logic/backends'
    profile={'fullname':'ipfs_datasets_py.logic.backends','path':str(namespace),'device':namespace.stat().st_dev,
        'inode':namespace.stat().st_ino,'initializer_absent':True}
    assert not(namespace/'__init__.py').exists()
    manifest=put(control/'repair-control-manifest.json',{'schema':owners.CONTROL_SCHEMA,'root':str(control),
        'files':sorted(files,key=lambda p:p['path']),'namespace_profiles':[profile]})
    owner=control/'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py'
    source=datasets/'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py'
    policy={'schema':owners.POLICY_SCHEMA,'control_release':{'root':str(control),'manifest':manifest},'datasets_root':str(datasets),
        'python':sys.executable,'source_descriptors':[pin(owner),pin(owner.with_name('repair_validation_worker.py')),pin(source)],
        'validation':{'import_sources':imports},'adoption':{}}
    frozen=[*files,*imports,shadow,manifest]
    for p in frozen:
        info=Path(p['path']).lstat();assert stat.S_ISREG(info.st_mode)and info.st_nlink==1 and stat.S_IMODE(info.st_mode)==0o400
    return SimpleNamespace(control=control,datasets=datasets,owner=owner,policy=policy,frozen=frozen)


@pytest.mark.parametrize('mode',['guarded','legacy'])
def test_public_source_lease_preserves_cold_control_order_without_entering_leases(cold_public_source_order_bundle,tmp_path,mode):
    """Real cold imports; modeled policy confers no Source/Native permission."""
    import os
    import subprocess
    s=cold_public_source_order_bundle
    before={p['path']:Path(p['path']).read_bytes()for p in s.frozen}
    policy=put(tmp_path/'owner-policy.json',copy.deepcopy(s.policy))
    code="import hashlib,importlib.util,json,sys\nfrom pathlib import Path\nowner_path,policy_path,policy_sha,mode=sys.argv[1:]\nassert not any(n=='ipfs_accelerate_py'or n.startswith('ipfs_accelerate_py.')for n in sys.modules)\nspec=importlib.util.spec_from_file_location('_cold_source_order_full_SDK',owner_path)\nowner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)\npolicy=json.loads(Path(policy_path).read_bytes());c=Path(policy['control_release']['root']);d=Path(policy['datasets_root'])\nif mode=='guarded':owner.install_control_guard(Path(policy_path),policy_sha)\nlease=owner.public_source_lease(policy)\nsource=sys.modules[lease.__module__];expected=d/'ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py'\nassert Path(source.__file__).absolute()==expected\nresult={'mode':mode,'source_owner':str(expected),'source_lease_context_entered':False,'Native_or_Git_operation_performed':False}\nif mode=='guarded':\n    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine\n    actual=Path(engine.__file__).absolute();assert actual==c/'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py'\n    assert sys.path[:2]==[str(c),str(d)] and not owner._CONTROL_GUARD.denials\n    assert owner._CONTROL_GUARD.executed[str(actual)]==hashlib.sha256(actual.read_bytes()).hexdigest()\n    assert str(d/'ipfs_accelerate_py/__init__.py')not in owner._CONTROL_GUARD.executed\n    result.update(engine=str(actual),guard_denials=owner._CONTROL_GUARD.denials,control_prefix=sys.path[:2])\nelse:\n    assert owner._CONTROL_GUARD is None and sys.path[0]==str(d)\n    result.update(legacy_D_prepend_preserved=True)\nprint(json.dumps(result,sort_keys=True))\n"
    environment=os.environ.copy();environment.pop('PYTEST_CURRENT_TEST',None)
    environment.update(IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',IPFS_AUTO_INSTALL='false',IPFS_DATASETS_AUTO_INSTALL='0',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    completed=subprocess.run([sys.executable,'-I','-B','-c',code,str(s.owner),policy['path'],policy['sha256'],mode],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30,env=environment)
    assert len(completed.stdout.encode())+len(completed.stderr.encode())<262144
    put(tmp_path/'cold-child.stdout',completed.stdout.encode());put(tmp_path/'cold-child.stderr',completed.stderr.encode())
    assert completed.returncode==0,completed.stderr
    value=json.loads(completed.stdout.splitlines()[-1]);assert value['mode']==mode
    assert value['source_lease_context_entered']is False and value['Native_or_Git_operation_performed']is False
    if mode=='guarded':
        assert value['engine']==str(s.control/'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py')
        assert value['control_prefix']==[str(s.control),str(s.datasets)] and value['guard_denials']==[]
    else:assert value['legacy_D_prepend_preserved']is True
    assert all(Path(path).read_bytes()==raw for path,raw in before.items())


# Cancelled retained owner controls: typed boundaries are explicitly modeled DATA.
@pytest.fixture
def cancelled_retained_source_custody_DATA(typed_v5_modeled_boundaries, monkeypatch):
    s=typed_v5_modeled_boundaries;c=s.c
    c.policy['validation']['source_qualification']={'explicit_modeled_source_mode':True}
    s.terminal['reservation'].update(status='retained',cleanup_error=None)
    s.terminal['reservation']['record']['status']='retained'
    checks=[]
    def source_plan(policy,plan,invocation):
        assert policy is c.policy and plan is c.context['plan'] and invocation==c.context['invocation']
        checks.append('plan');s.events.append('source_plan')
        return {'explicit_modeled_current_source_scope':True}
    monkeypatch.setattr(owners,'_source_qualification_plan',source_plan)
    def old_report(*args,**kwargs):
        raise ValueError('modeled legacy failed-report path remains closed')
    monkeypatch.setattr(owners,'_source_qualification_failed_custody',old_report)
    verdict={'value':{'verdict':'proven_original_cancelled_retained_native_closed',
        'source_custody_closed':True,'storage_release_authorized':False}}
    def inspector(terminal,raw,**arguments):
        assert terminal==s.terminal and raw==c.raw
        assert arguments['returncode']==c.process.returncode==1
        assert arguments['owner_pid']==c.identity['pid'] and arguments['owner_birth']==c.identity['birth']
        assert arguments['cleanup_profile']=='cancelled_retained_v5'
        assert arguments['original_wait_deadline_monotonic']==c.deadline==1360.0
        assert checks==['plan'] and not c.closed
        s.native_calls.append(('cancelled_retained',terminal,arguments));s.events.append('native_inspector')
        return verdict['value']
    c.inspector=inspector
    return SimpleNamespace(base=s,c=c,checks=checks,verdict=verdict,inspector=inspector)


def test_source_qualification_typed_timeout_closes_retained_custody_without_report_credit(
        cancelled_retained_source_custody_DATA):
    f=cancelled_retained_source_custody_DATA;s=f.base
    completed=s.result(kind='timeout',public=124,actual=1)
    witness=s.record(completed);parent=f.c.finish(completed,witness)
    assert parent['closure_kind']=='original_cancelled_retained_native_closed'
    assert parent['source_closed'] is parent['native_closed'] is True
    assert parent['storage_status']=='retained'
    assert parent['storage_release_authorized'] is parent['retry_authorized'] is False
    assert s.events.index('source_plan')<s.events.index('native_inspector')<s.events.index('source_exit')
    assert s.verified==[] and f.checks==['plan']
    saved=json.loads(Path(f.c.wait_ref['path']).read_bytes())
    assert saved['returncode']==124 and saved['actual_process_returncode']==1
    assert not (s.invocation/'source-qualification-receipt.json').exists()
    assert not (s.invocation/'exports/trusted-validation.json').exists()
    # No underlying test/call count or qualified worker claim is established.


@pytest.mark.parametrize('fault',['completed_kind','wrong_public','not_waited','incomplete_capture',
    'wrong_process_pid','wrong_birth','wrong_engine_origin','source_guard','source_plan','unknown_native',
    'expired_before','expired_after_inspection','unknown_profile_verdict'])
def test_source_qualification_cancelled_retained_binding_fault_never_closes_source(
        cancelled_retained_source_custody_DATA,monkeypatch,fault):
    f=cancelled_retained_source_custody_DATA;s=f.base;c=f.c
    options=dict(kind='timeout',public=124,actual=1)
    if fault=='completed_kind':options['kind']='completed'
    elif fault=='wrong_public':options['public']=1
    elif fault=='not_waited':options['waited']=False
    elif fault=='incomplete_capture':options['output']='incomplete'
    elif fault=='wrong_process_pid':options['edits']={'process_pid':c.process.pid+1}
    completed=s.result(**options);witness=s.record(completed)
    if fault=='wrong_birth':witness['runner_identity']={**c.identity,'birth':'99999'}
    elif fault=='wrong_engine_origin':c.engine_source={**c.engine_source,'path':'/untrusted/engine.py'}
    elif fault=='source_guard':
        def guard():raise ValueError('modeled current source changed')
        c.guard=guard
    elif fault=='source_plan':
        def plan(*args):raise ValueError('modeled source plan changed')
        monkeypatch.setattr(owners,'_source_qualification_plan',plan)
    elif fault=='unknown_native':
        f.verdict['value']={'verdict':'requires_original_native_cleanup',
            'source_custody_closed':False,'storage_release_authorized':False}
    elif fault=='unknown_profile_verdict':
        f.verdict['value']={'verdict':'proven_original_productive_native_closed_storage_separate',
            'source_custody_closed':True,'storage_release_authorized':False}
    elif fault=='expired_before':s.now['value']=c.deadline
    elif fault=='expired_after_inspection':
        def inspector(*args,**kwargs):
            result=f.inspector(*args,**kwargs);s.now['value']=c.deadline;return result
        c.inspector=inspector
    with pytest.raises(ValueError):c.finish(completed,witness)
    assert c.closed is False and 'source_exit'not in s.events and s.verified==[]
    assert not (s.invocation/'source-qualification-receipt.json').exists()


def test_source_qualification_cancelled_retained_late_actual_source_exit_is_not_acceptance(
        cancelled_retained_source_custody_DATA):
    f=cancelled_retained_source_custody_DATA;s=f.base;c=f.c
    old_close=c.stack.close
    def late_close():old_close();s.now['value']=c.deadline
    c.stack.close=late_close
    completed=s.result(kind='timeout',public=124,actual=1);witness=s.record(completed)
    with pytest.raises(ValueError,match='Source exit completed after'):c.finish(completed,witness)
    assert c.closed is True and 'source_exit'in s.events
    late=json.loads((s.invocation/'source-custody-late-close.json').read_bytes())
    assert late['source_closed'] is True and late['validation_accepted'] is False
    assert late['storage_release_authorized'] is late['retry_authorized'] is False
    assert not (s.invocation/'source-qualification-receipt.json').exists()


@pytest.mark.parametrize('mode',['repair','report_failure'])
def test_source_qualification_cancelled_profile_is_not_a_default_or_completed_failure_bypass(
        cancelled_retained_source_custody_DATA,mode):
    f=cancelled_retained_source_custody_DATA;s=f.base;c=f.c
    if mode=='repair':c.policy['validation'].pop('source_qualification')
    completed=s.result(kind=('completed' if mode=='report_failure' else 'timeout'),
        public=(1 if mode=='report_failure' else 124),actual=1);witness=s.record(completed)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):c.finish(completed,witness)
    assert c.closed is False and 'source_exit'not in s.events
    assert not any(call[0]=='cancelled_retained' for call in s.native_calls)


@pytest.fixture
def failed_retained_source_custody_DATA(typed_v5_modeled_boundaries, monkeypatch):
    s = typed_v5_modeled_boundaries
    c = s.c
    c.policy['validation']['source_qualification'] = {'explicit_modeled_source_mode': True}
    c.request['selected_checks'] = ['explicit reported command selector DATA']
    c.context['plan'] = {'attempt_directory': str(s.invocation / 'attempt-DATA')}
    s.terminal['reservation'].update(status='retained', cleanup_error=None)
    s.terminal['reservation']['record']['status'] = 'retained'
    s.terminal['failure'] = {'error_type': 'RuntimeError', 'returncode': 7,
        'error': 'validation command failed with returncode 7', 'received_signals': [],
        'executed_checks': list(c.request['selected_checks'])}
    checks = []
    def plan(policy, selected, invocation):
        assert policy is c.policy and selected is c.context['plan'] and invocation == c.context['invocation']
        checks.append('plan'); s.events.append('source_plan')
        return {'explicit_modeled_plan_DATA': True}
    monkeypatch.setattr(owners, '_source_qualification_plan', plan)
    def old_failure(*args, **kwargs):
        raise ValueError('modeled old failed-report qualification remains closed')
    monkeypatch.setattr(owners, '_source_qualification_failed_custody', old_failure)
    verdict = {'value': {'verdict': 'proven_original_failed_retained_native_closed',
        'source_custody_closed': True, 'storage_release_authorized': False}}
    def inspector(terminal, raw, **arguments):
        if arguments.get('cleanup_profile') != 'failed_retained_v5':
            return {'verdict': 'requires_original_native_cleanup', 'source_custody_closed': False,
                'storage_release_authorized': False}
        assert terminal == s.terminal and raw == c.raw
        assert arguments['returncode'] == c.process.returncode == 1
        assert arguments['owner_pid'] == c.identity['pid'] and arguments['owner_birth'] == c.identity['birth']
        assert arguments['original_wait_deadline_monotonic'] == c.deadline == 1360.0
        assert arguments['cleanup_profile'] == 'failed_retained_v5'
        assert checks == ['plan'] and not c.closed
        s.native_calls.append(('failed_retained', terminal, arguments)); s.events.append('native_inspector')
        return verdict['value']
    c.inspector = inspector
    return SimpleNamespace(base=s, c=c, checks=checks, verdict=verdict, inspector=inspector)


@pytest.mark.parametrize('worker_output', ['absent', 'partial-unsealed', 'unaccepted-sealed'])
def test_completed_failed_retained_closes_Source_without_qualifying_worker_report(failed_retained_source_custody_DATA, worker_output):
    f = failed_retained_source_custody_DATA; s = f.base
    path = Path(f.c.context['plan']['attempt_directory']) / 'exports/trusted-validation.json'
    if worker_output != 'absent':
        path.parent.mkdir(parents=True)
        path.write_bytes(b'{partial' if worker_output == 'partial-unsealed' else b'{"passed":true}\n')
        path.chmod(0o600 if worker_output == 'partial-unsealed' else 0o400)
    before = path.read_bytes() if path.exists() else None
    completed = s.result(kind='completed', public=1, actual=1)
    result = f.c.finish(completed, s.record(completed))
    assert result['closure_kind'] == 'original_failed_retained_native_closed'
    assert result['source_closed'] is result['native_closed'] is True and result['storage_status'] == 'retained'
    assert result['storage_release_authorized'] is result['retry_authorized'] is False
    assert 'verified_checks' not in result and s.verified == []
    assert s.events.index('source_plan') < s.events.index('native_inspector') < s.events.index('source_exit')
    assert (path.read_bytes() if path.exists() else None) == before
    wait = json.loads(Path(f.c.wait_ref['path']).read_bytes())
    assert wait['returncode'] == wait['actual_process_returncode'] == 1
    assert not (s.invocation / 'source-qualification-receipt.json').exists()
    # Neither reported selectors nor an absent report establish actual case counts.


@pytest.mark.parametrize('fault', ['timeout_kind', 'wrong_public', 'not_waited', 'incomplete_capture',
    'wrong_pid', 'wrong_birth', 'wrong_engine_origin', 'source_guard', 'source_plan', 'unknown_native',
    'expired_before', 'expired_after_inspection', 'wrong_profile_verdict', 'zero_worker', 'boolean_worker',
    'other_error', 'received_signal', 'repair_mode'])
def test_completed_failed_retained_unproven_provenance_never_closes_Source(failed_retained_source_custody_DATA, monkeypatch, fault):
    f = failed_retained_source_custody_DATA; s = f.base; c = f.c
    options = dict(kind='completed', public=1, actual=1)
    if fault == 'timeout_kind': options.update(kind='timeout', public=124)
    elif fault == 'wrong_public': options['public'] = 7
    elif fault == 'not_waited': options['waited'] = False
    elif fault == 'incomplete_capture': options['output'] = 'incomplete'
    elif fault == 'wrong_pid': options['edits'] = {'process_pid': c.process.pid+1}
    elif fault == 'wrong_birth': pass
    elif fault == 'wrong_engine_origin': c.engine_source = {**c.engine_source, 'path': '/untrusted/engine.py'}
    elif fault == 'source_guard':
        def guard(): raise ValueError('modeled original source changed')
        c.guard = guard
    elif fault == 'source_plan':
        def plan(*args): raise ValueError('modeled source plan changed')
        monkeypatch.setattr(owners, '_source_qualification_plan', plan)
    elif fault == 'unknown_native':
        f.verdict['value'] = {'verdict': 'requires_original_native_cleanup', 'source_custody_closed': False,
            'storage_release_authorized': False}
    elif fault == 'expired_before': s.now['value'] = c.deadline
    elif fault == 'expired_after_inspection':
        def inspector(*args, **kwargs):
            result = f.inspector(*args, **kwargs); s.now['value'] = c.deadline; return result
        c.inspector = inspector
    elif fault == 'wrong_profile_verdict':
        f.verdict['value'] = {'verdict': 'proven_original_cancelled_retained_native_closed',
            'source_custody_closed': True, 'storage_release_authorized': False}
    elif fault == 'zero_worker': s.terminal['failure']['returncode'] = 0
    elif fault == 'boolean_worker': s.terminal['failure']['returncode'] = True
    elif fault == 'other_error': s.terminal['failure']['error'] = 'different failure'
    elif fault == 'received_signal': s.terminal['failure']['received_signals'] = [15]
    else:
        c.policy['validation'].pop('source_qualification')
    completed = s.result(**options); witness = s.record(completed)
    if fault == 'wrong_birth': witness['runner_identity'] = {**c.identity, 'birth': '99999'}
    # Nonmatching frames keep the old classifier/report path, which is not a
    # source-qualification cleanup grant. Do not reinterpret their reports.
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        c.finish(completed, witness)
    assert c.closed is False and 'source_exit' not in s.events and s.verified == []
    assert not (s.invocation / 'source-qualification-receipt.json').exists()


def test_completed_failed_retained_late_Source_exit_is_closed_without_acceptance(failed_retained_source_custody_DATA):
    f = failed_retained_source_custody_DATA; s = f.base; c = f.c
    original = c.stack.close
    def late_close():
        original(); s.now['value'] = c.deadline
    c.stack.close = late_close
    completed = s.result(kind='completed', public=1, actual=1)
    with pytest.raises(ValueError, match='Source exit completed after'):
        c.finish(completed, s.record(completed))
    late = json.loads((s.invocation / 'source-custody-late-close.json').read_bytes())
    assert c.closed is True and late['source_closed'] is True and late['validation_accepted'] is False
    assert late['storage_release_authorized'] is late['retry_authorized'] is False
    assert not (s.invocation / 'source-qualification-receipt.json').exists()


def test_source_qualification_validate_persists_failed_cleanup_without_report_acceptance(metadata_scope, monkeypatch):
    """Current immutable@1 public metadata flow; original runner is modeled DATA."""
    s, policy, expected_plan = _immutable_qualification_ready_plan(metadata_scope)
    policy_pin = pin(s.scope.state / 'qualified-owner-policy.json')
    assert s.profile['schema'] == 'agent-supervisor/immutable-source-qualification@1'
    monkeypatch.setattr(owners, 'checked_policy', lambda *a, **k: policy)
    result_custody = {'source_closed': True, 'native_closed': True,
        'closure_kind': 'original_failed_retained_native_closed', 'storage_status': 'retained',
        'storage_release_authorized': False, 'retry_authorized': False}
    calls = []
    def original_runner(offered, argv, cwd, timeout, *, validation_context):
        calls.append('original_runner_DATA')
        plan = validation_context['plan']
        assert offered is policy and timeout == 360
        assert owners._source_qualification_plan(policy, plan, s.invocation)['profile'] == s.profile
        assert owners.canonical(plan) == owners.canonical(expected_plan)
        assert 'source_parent' not in validation_context
        assert owners.original_execution_request(s.invocation) == validation_context['request_pin']
        assert argv[argv.index('--request') + 1] == validation_context['request_pin']['path']
        return SimpleNamespace(returncode=1, stdout='{"explicit_completed_failed_cli_DATA":true}', stderr=''), {
            'parent_source_custody': copy.deepcopy(result_custody)}
    monkeypatch.setattr(owners, 'original_runner_wait', original_runner)
    def forbidden(*a, **k):
        pytest.fail('cleanup-only public failure entered qualification, negative diagnosis or hold')
    for name in ('verify_validation_evidence', 'capture_negative_validation', '_source_qualification_failed_custody'):
        monkeypatch.setattr(owners, name, forbidden)
    monkeypatch.setattr(owners._V5ValidationSourceCustody, 'hold', forbidden)
    args = SimpleNamespace(policy=Path(policy_pin['path']), policy_sha256=policy_pin['sha256'],
        freeze_path=Path(s.snapshot_pin['path']), source_commitment=s.snapshot['source_commitment'],
        candidate_root=s.scope.control, invocation_root=s.invocation,
        receipt_path=s.invocation / 'source-qualification-receipt.json')
    assert owners.validate(args) == 1
    saved = json.loads((s.invocation / 'failed-source-qualification.json').read_bytes())
    assert saved['parent_source_custody'] == result_custody and saved['passed'] is False
    assert saved['storage_closed'] is saved['semantic_negative'] is saved['retry_authorized'] is False
    assert calls == ['original_runner_DATA'] and not args.receipt_path.exists()


# Explicit mixed-V3 original-parent controls. Local byte rows below are modeled
# inventory, never a compiler result, issued handle, Source proof or Native grant.
@pytest.fixture
def reviewed_parent_packet_join_scope_v3(reviewed_parent_packet_join_scope):
    old = reviewed_parent_packet_join_scope
    binding, durable = copy.deepcopy(old.binding), copy.deepcopy(old.durable)
    packet = list(old.packet)
    # Extend the existing owned fixture by exactly the six mixed-sort files.
    for folder, suffix in (("project", "lean"), ("executed-inputs", "lean"), ("compiled", "olean")):
        for index in (33, 34) if folder != "compiled" else (6, 7):
            relative = folder + "/Mixed%02d." % index + suffix
            raw = owners.canonical({"fixture_only": True, "mixed_v3": relative}) + b"\n"
            reference = put(old.attempt / old.operation["packet_root"] / relative, raw)
            info = Path(reference["path"]).stat()
            packet.append((relative, raw))
            binding["packet_inventory"].append({"path": relative, "bytes": len(raw), "sha256": reference["sha256"]})
            durable.append({"path": old.operation["packet_root"] + "/" + relative, "bytes": len(raw),
                "sha256": reference["sha256"], "device": info.st_dev, "inode": info.st_ino})
    binding["schema"] = "reviewed-preservation-native-binding/v3"
    binding["packet_inventory"].sort(key=lambda row: row["path"])
    binding["packet_sha256"] = owners.digest(binding["packet_inventory"])
    assert len(packet) == len(durable) == len(binding["packet_inventory"]) == 83
    operation = copy.deepcopy(old.operation)
    operation["expected_packet_paths"] = [row["path"] for row in binding["packet_inventory"]]
    binding["expected_packet_paths"] = list(operation["expected_packet_paths"])
    return SimpleNamespace(attempt=old.attempt, operation=operation, binding=binding,
        durable=durable, packet=tuple(sorted(packet)), source=old.source)


def test_reviewed_parent_v3_joins_all83_modeled_rows_without_issuing_a_handle(reviewed_parent_packet_join_scope_v3):
    s = reviewed_parent_packet_join_scope_v3
    result = owners._reviewed_parent_packet_binding_v3(s.operation, s.binding, s.durable, attempt_directory=s.attempt)
    expected = tuple((row["path"], row["bytes"], row["sha256"], row["device"], row["inode"])
                     for row in sorted(s.durable, key=lambda row: row["path"]))
    assert result == expected and len(result) == 83
    assert all(type(row) is tuple and len(row) == 5 for row in result)


@pytest.mark.parametrize("fault", ["missing", "extra", "reorder", "duplicate", "partition", "receipt",
    "packet", "graph", "admission", "preparation", "durability", "inode_bool", "device_bool", "domain"])
def test_reviewed_parent_v3_refuses_incomplete_or_crossed83_metadata(reviewed_parent_packet_join_scope_v3, fault):
    s = reviewed_parent_packet_join_scope_v3
    binding, durable = copy.deepcopy(s.binding), copy.deepcopy(s.durable)
    if fault == "missing": binding["packet_inventory"].pop()
    elif fault == "extra": binding["packet_inventory"].append(dict(binding["packet_inventory"][-1], path="unexpected.bin"))
    elif fault == "reorder": binding["packet_inventory"].reverse()
    elif fault == "duplicate": binding["packet_inventory"][-1] = dict(binding["packet_inventory"][0])
    elif fault == "partition": binding["packet_inventory"][-1]["path"] = "wrong/Last.lean"
    elif fault == "receipt": binding["receipt_sha256"] = "f" * 64
    elif fault == "packet": binding["packet_sha256"] = "f" * 64
    elif fault in ("graph", "admission", "preparation"): binding[fault + "_sha256"] = "f" * 64
    elif fault == "durability": durable.pop()
    elif fault == "inode_bool": durable[0]["inode"] = True
    elif fault == "device_bool": durable[0]["device"] = True
    else: binding["schema"] = "reviewed-preservation-native-binding/v1"
    with pytest.raises(ValueError):
        owners._reviewed_parent_packet_binding_v3(s.operation, binding, durable, attempt_directory=s.attempt)


@pytest.mark.parametrize("direction", ["old_reader_new_packet", "new_reader_old_packet"])
def test_reviewed_parent_v3_and_v2_inventory_domains_cannot_be_retagged(
        reviewed_parent_packet_join_scope, reviewed_parent_v3_complete_packet_scope, direction):
    """Closed metadata refusals only; no original capability or Native claim."""
    legacy = reviewed_parent_packet_join_scope
    opening = reviewed_parent_v3_complete_packet_scope.packet
    # The complete V3 opening comes from pinned real preparation DATA, never
    # from the substituted packet. Its compiled bytes remain explicitly modeled.
    public = owners.json_object(owners.checked_descriptor(opening.operation["native_inputs"]))
    expected_paths = owners._reviewed_parent_packet_paths_v3(public)
    assert expected_paths == opening.operation["expected_packet_paths"]
    assert expected_paths == opening.binding["expected_packet_paths"]
    current = owners._reviewed_parent_packet_binding_v3(
        opening.operation, opening.binding, opening.durable, attempt_directory=opening.attempt)
    assert len(current) == len(expected_paths)
    legacy_current = owners._reviewed_parent_packet_binding_v1(
        legacy.operation, legacy.binding, legacy.durable, attempt_directory=legacy.attempt)
    assert len(legacy_current) == 77

    for translation in ("schema_only", "destination_closed_shape"):
        if direction == "old_reader_new_packet":
            operation = copy.deepcopy(opening.operation)
            binding = copy.deepcopy(opening.binding)
            binding["schema"] = "reviewed-preservation-native-binding/v1"
            function = owners._reviewed_parent_packet_binding_v1
            durable, attempt = opening.durable, opening.attempt
            if translation == "destination_closed_shape":
                operation.pop("expected_packet_paths")
                binding.pop("expected_packet_paths")
                assert set(binding) == set(legacy.binding)
                assert len(binding["packet_inventory"]) != 77
        else:
            operation = copy.deepcopy(opening.operation)
            function = owners._reviewed_parent_packet_binding_v3
            durable, attempt = legacy.durable, legacy.attempt
            if translation == "schema_only":
                binding = copy.deepcopy(legacy.binding)
                binding["schema"] = "reviewed-preservation-native-binding/v3"
            else:
                # Keep the real V3 opening and its expected paths, substituting
                # and resealing only the legacy packet inventory commitments.
                binding = copy.deepcopy(opening.binding)
                binding["packet_inventory"] = copy.deepcopy(legacy.binding["packet_inventory"])
                binding["packet_sha256"] = owners.digest(binding["packet_inventory"])
                binding["receipt_sha256"] = next(row["sha256"] for row in binding["packet_inventory"]
                    if row["path"] == "build-receipt.json")
                assert set(binding) == set(opening.binding)
                assert [row["path"] for row in binding["packet_inventory"]] != expected_paths

        if translation == "schema_only":
            expected_error = "closed original reviewed native packet binding required"
        elif direction == "old_reader_new_packet":
            expected_error = "complete original77-file native packet required"
        else:
            expected_error = "exact independently pinned V3 packet population required"
        with pytest.raises(ValueError) as caught:
            function(operation, binding, durable, attempt_directory=attempt)
        assert str(caught.value) == expected_error
        assert opening.operation["expected_packet_paths"] == expected_paths


def test_reviewed_parent_v3_worker_walks_the_same83_owned_files(reviewed_parent_packet_join_scope_v3):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    s = reviewed_parent_packet_join_scope_v3
    collector = object.__new__(worker.ReviewedPreservationNativeCollectorV3)
    collector.packet_root = s.attempt / s.operation["packet_root"]
    collector.reviewed_version = 3
    collector.operation = s.operation
    assert collector._packet() == s.packet
    # These objects exercise only ordinary inventory; no fake valid capability.
    old = object.__new__(worker.ReviewedPreservationNativeCollectorV1)
    old.packet_root = collector.packet_root
    with pytest.raises(ValueError, match="fixed profile"):
        old._packet()


@pytest.mark.parametrize("offered", ["none", "object", "dict", "json", "unissued"])
def test_reviewed_parent_v3_refuses_unissued_capabilities_before_native_resolution(monkeypatch, offered):
    value = {"none": None, "object": object(), "dict": {"original_success_authenticated": True},
        "json": json.loads('{"schema":"reviewed-preservation-admitted-parent-success/v3"}'),
        "unissued": object.__new__(owners.VerifiedReviewedPreservationParentSuccess)}[offered]
    def forbidden(*args, **kwargs):
        pytest.fail("unissued V3 parent handle reached native owner")
    monkeypatch.setattr(owners, "_parent_native_owner", forbidden)
    with pytest.raises(ValueError, match="(?i)(issued|handle)"):
        owners.load_parent_verified_reviewed_preservation_v3(value, (), None, {}, {},
            expected_receipt_sha256="a" * 64, expected_packet_sha256="b" * 64,
            expected_binary_sha256={}, expected_preservation_sha256="c" * 64,
            expected_graph_sha256="d" * 64, expected_admission_sha256="e" * 64,
            toolchain="leanprover/lean4:v4.34.1")


@pytest.mark.parametrize("version", ["legacy", "mixed"])
def test_reviewed_parent_public_policy_versions_refuse_each_others_schema_before_root_io(tmp_path, version):
    schema = "reviewed-preservation-parent/v3" if version == "legacy" else "reviewed-preservation-parent/v1"
    value = dict(schema=schema, datasets_root=None, source_root=None, python=None,
        source_descriptors=None, validation=None, operation=None)
    reference = put(tmp_path / "wrong-domain-policy.json", value)
    function = (owners.checked_reviewed_preservation_parent_policy_v1 if version == "legacy"
                else owners.checked_reviewed_preservation_parent_policy_v3)
    with pytest.raises(ValueError, match="closed reviewed parent policy required"):
        function(Path(reference["path"]), reference["sha256"])


@pytest.mark.parametrize("worker_schema,parent_schema", [
    ("reviewed-preservation-native-worker/v1", "reviewed-preservation-parent/v1"),
    ("reviewed-preservation-native-worker/v3", "reviewed-preservation-parent/v1"),
    ("reviewed-preservation-native-worker/v1", "reviewed-preservation-parent/v3"),
    ("reviewed-preservation-native-worker/v99", "reviewed-preservation-parent/v3"),
])
def test_reviewed_parent_v3_worker_rejects_crossed_domains_before_any_operation(worker_schema, parent_schema):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_validation_worker as worker
    with pytest.raises(ValueError, match="explicit reviewed native worker profile required"):
        worker.reviewed_preservation_native_operation_v3({"schema": worker_schema}, {"schema": parent_schema})


def prepare_reviewed_parent_native_fixture_inputs_v3(path, control=None, *, datasets_root=None):
    """Serialize the caller's exact V3 component; retain old fixture-call mode.

    Explicit control never regenerates a target or signing material. Omitting
    control keeps the existing ordinary CPU fixture helper for compatibility.
    Neither mode runs a native tool or issues a capability.
    """
    import tempfile
    if datasets_root is None:
        datasets_root = Path(__file__).resolve().parents[3] / "ipfs_datasets"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    helper = reviewed_parent_original_R_fixture_module_v1(Path(datasets_root))

    def encode(selected):
        assert selected["admission"]["fixture_only"] is True
        assert selected["admission"]["provenance_kind"] == "fictional_fixture"
        arguments = helper.reviewed_native_arguments_v3(selected)
        prepared = selected["preparation"]
        value = {"schema": "reviewed-preservation-fixture-native-inputs/v3",
            "fixture_only": True, "provenance_kind": "fictional_fixture",
            "preparation": {"manifest_json": prepared.manifest_json,
                "files": [[name, body] for name, body in prepared.files]},
            **{name: reviewed_parent_fixture_tree_encode_v1(selected[name])
                for name in ("graph", "admission", "args", "kwargs")},
            **{name: arguments[name] for name in ("expected_preservation_sha256",
                "expected_graph_sha256", "expected_admission_sha256", "toolchain")}}
        paths = owners._reviewed_parent_packet_paths_v3(value)
        raw = owners.canonical(value) + b"\n"
        assert len(raw) <= 4 * 1024**2
        reference = put(path, raw)
        return {"native_inputs": reference, "expected_packet_paths": paths,
            **{name: value[name] for name in ("expected_preservation_sha256",
                "expected_graph_sha256", "expected_admission_sha256", "toolchain")}}

    if control is not None:
        return encode(control)
    with tempfile.TemporaryDirectory(prefix="reviewed-v3-inputs-", dir=path.parent) as scratch:
        factory = SimpleNamespace(mktemp=lambda label, **_options: Path(
            tempfile.mkdtemp(prefix=label + "-", dir=scratch)))
        seed = helper.reviewed_native_seed_v3.__wrapped__(factory)
        selected = helper.reviewed_native_control_v3.__wrapped__(seed)
        assert helper.reviewed_native_validate_v3(selected) == selected["preparation"]
        return encode(selected)


def test_reviewed_preservation_parent_actual_native_child_v3():
    """One future original-parent child body, separately gated from UNIT cases."""
    import os
    import importlib
    names = ("IPFS_REVIEWED_PRESERVATION_PLAN", "IPFS_REVIEWED_PRESERVATION_ATTEMPT", "IPFS_REVIEWED_PRESERVATION_PACKET_ROOT")
    if os.environ.get("IPFS_DATASETS_RUN_NATIVE_LEAN") != "1" or not all(os.environ.get(name) for name in names):
        pytest.skip("explicit original-parent reviewed native child enrollment required")
    plan_path, attempt, packet_root = (Path(os.environ[name]) for name in names)
    plan = owners.json_object(owners.ordinary_bytes(plan_path, 4 * 1024**2))
    assert plan["schema"] == "reviewed-preservation-native-worker/v3"
    policy = owners.json_object(owners.checked_descriptor(plan["policy"]))
    operation = plan["operation"]
    assert operation == policy["operation"] and plan["native_inputs"] == operation["native_inputs"]
    assert operation["nodeid"] == str(Path(__file__).absolute()) + "::test_reviewed_preservation_parent_actual_native_child_v3"
    assert attempt == Path(plan["attempt_directory"]) and packet_root == attempt / operation["packet_root"]
    assert not packet_root.exists()
    helper = reviewed_parent_original_R_fixture_module_v1(Path(policy["datasets_root"]))
    reference = operation["native_inputs"]
    raw = owners.ordinary_bytes(Path(reference["path"]), 4 * 1024**2)
    assert pin(Path(reference["path"])) == reference
    value = owners.json_object(raw)
    assert set(value) == {"schema", "fixture_only", "provenance_kind", "preparation", "graph", "admission", "args", "kwargs",
                          "expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain"}
    assert value["schema"] == "reviewed-preservation-fixture-native-inputs/v3" and value["fixture_only"] is True
    assert value["provenance_kind"] == "fictional_fixture" and set(value["preparation"]) == {"manifest_json", "files"}
    shared = importlib.import_module("ipfs_datasets_py.logic.autoformal.legal_corpus_shared_lake")
    prepared = shared.ReviewedPreservationCorpusPreparation(value["preparation"]["manifest_json"],
        tuple((name, body) for name, body in value["preparation"]["files"]))
    control = {"shared": shared, "preparation": prepared,
        "graph": reviewed_parent_fixture_tree_decode_v1(value["graph"]),
        "admission": reviewed_parent_fixture_tree_decode_v1(value["admission"]),
        "args": reviewed_parent_fixture_tree_decode_v1(value["args"]),
        "kwargs": reviewed_parent_fixture_tree_decode_v1(value["kwargs"]),
        "graph_sha": value["expected_graph_sha256"], "admission_sha": value["expected_admission_sha256"]}
    assert control["admission"]["fixture_only"] is True and control["admission"]["provenance_kind"] == "fictional_fixture"
    arguments = helper.reviewed_native_arguments_v3(control)
    for name in ("expected_preservation_sha256", "expected_graph_sha256", "expected_admission_sha256", "toolchain"):
        assert arguments[name] == value[name] == operation[name]
    binaries = operation["expected_binary_sha256"]
    executable = next(path for path in binaries if Path(path).name == "lake")
    # Existing original R functions own every validation, compiler and reader.
    handle = helper.reviewed_live_build_v3_fixture(control, executable, packet_root)
    packet = helper.reviewed_live_success_packet_v2(packet_root)
    assert len(packet) == 83
    result = helper.reviewed_live_load_v3_fixture(control, handle, packet)
    assert result["native_execution_authenticated"] is result["live_native_success_authenticated"] is True
    assert result["original_success_authenticated"] is result["original_admitted_parent_custody_authenticated"] is False
    assert result["retained_packet_sha256"] == helper.native.reviewed_norm_preservation_packet_sha256_v3(packet)
    assert result["training_mask"] == result["evaluation_gold_mask"] == 0
    assert all(flag is False for flag in result["authority"].values())
    assert result["release_ready"] is result["Lean_admitted"] is False
    assert json.loads(dict(packet)["build-receipt.json"])["binary_sha256"] == binaries


# Public tagged-DATA depth controls. These stop at an inert operation-ID gate;
# they do not create an invocation, Source lease, process or parent capability.
def _reviewed_parent_public_depth_value(dict_edges, list_edges, *, version=3):
    logical = 0
    for kind in reversed(["dict"] * dict_edges + ["list"] * list_edges):
        logical = {"next": logical} if kind == "dict" else [logical]
    value = {"schema": "reviewed-preservation-fixture-native-inputs/v3" if version == 3
        else "reviewed-preservation-fixture-native-inputs/v1",
        "fixture_only": True, "provenance_kind": "fictional_fixture",
        "preparation": {"manifest_json": "{}", "files": []},
        "graph": reviewed_parent_fixture_tree_encode_v1({}),
        "admission": reviewed_parent_fixture_tree_encode_v1({}),
        "args": reviewed_parent_fixture_tree_encode_v1(()),
        "kwargs": reviewed_parent_fixture_tree_encode_v1(logical),
        "expected_preservation_sha256": "a" * 64, "expected_graph_sha256": "b" * 64,
        "expected_admission_sha256": "c" * 64, "toolchain": "leanprover/lean4:v4.34.1"}
    return value, logical


def _reviewed_parent_public_raw_depth(value):
    pending, count, maximum = [(value, 0)], 0, 0
    while pending:
        row, depth = pending.pop(); count += 1; maximum = max(maximum, depth)
        if type(row) is dict: pending.extend((child, depth + 1) for child in row.values())
        elif type(row) is list: pending.extend((child, depth + 1) for child in row)
    return count, maximum


def _reviewed_parent_public_depth_policy(scope, value, label, *, version=3):
    policy = copy.deepcopy(scope.policy)
    policy["schema"] = "reviewed-preservation-parent/v3" if version == 3 else "reviewed-preservation-parent/v1"
    native = put(scope.root / (label + "-public.json"), value)
    # A deliberately invalid operation ID is the FIRST post-tree gate. Reaching
    # this named refusal proves only the parser boundary, never Native success.
    policy["operation"] = {"operation_id": None, "packet_root": None, "nodeid": None,
        "expected_source_sha256": None, "expected_binary_sha256": None,
        "native_argv": None, "native_data": None, "native_inputs": native,
        **{key: value[key] for key in ("expected_graph_sha256", "expected_admission_sha256",
                                      "expected_preservation_sha256", "toolchain")}}
    if version == 3:
        # This parser-only policy must still stop at its deliberately invalid ID.
        # Empty population is never accepted as an executable V3 declaration.
        policy["operation"]["expected_packet_paths"] = []
    reference = put(scope.root / (label + "-policy.json"), policy)
    return Path(reference["path"]), reference["sha256"], native


@pytest.mark.parametrize("version,dict_edges,list_edges,raw_depth,accepted", [
    (2, 20, 1, 64, True), (2, 21, 0, 65, False),
    (3, 64, 0, 194, True), (3, 63, 2, 195, False),
])
def test_reviewed_parent_public_depth_bound_matches_closed_tagged_codec(
        reviewed_parent_normal_namespace_v2_scope, version, dict_edges, list_edges, raw_depth, accepted):
    scope = reviewed_parent_normal_namespace_v2_scope
    value, logical = _reviewed_parent_public_depth_value(dict_edges, list_edges, version=version)
    assert _reviewed_parent_public_raw_depth(value)[1] == raw_depth
    if dict_edges + list_edges <= 64:
        assert reviewed_parent_fixture_tree_decode_v1(value["kwargs"]) == logical
    else:
        # The enlarged serialized envelope cannot relax the old logical64 codec.
        with pytest.raises(AssertionError): reviewed_parent_fixture_tree_decode_v1(value["kwargs"])
    path, digest, _ = _reviewed_parent_public_depth_policy(scope, value, "depth-bound", version=version)
    function = (owners.checked_reviewed_preservation_parent_policy_v3 if version == 3
                else owners.checked_reviewed_preservation_parent_policy_v1)
    message = "^exact reviewed operation ID required$" if accepted else "^bounded public native input DATA tree required$"
    with pytest.raises(ValueError, match=message): function(path, digest)
    assert not (scope.root / "attempts").exists()


def test_reviewed_parent_v3_policy_accepts_encoded_public_fixture_above_legacy_depth(
        reviewed_parent_normal_namespace_v2_scope):
    scope = reviewed_parent_normal_namespace_v2_scope
    offered = owners.os.environ.get("IPFS_REVIEWED_PARENT_V3_POLICY_PUBLIC_INPUTS")
    if offered:
        # Optional selected original DATA is hash-authenticated and read-only.
        # It is not a callback, capability or serialized authenticity boolean.
        reference = json.loads(offered)
        value = owners.json_object(owners.checked_descriptor(reference))
        assert value["schema"] == "reviewed-preservation-fixture-native-inputs/v3"
        assert value["fixture_only"] is True and value["provenance_kind"] == "fictional_fixture"
    else:
        # Portable authored DATA reproduces raw67 without private workspace paths.
        value, _ = _reviewed_parent_public_depth_value(17, 7)
    count, depth = _reviewed_parent_public_raw_depth(value)
    assert count <= 2_000_000 and 64 < depth <= 194
    for key in ("graph", "admission", "args", "kwargs"):
        decoded = reviewed_parent_fixture_tree_decode_v1(value[key])
        assert owners.canonical(reviewed_parent_fixture_tree_encode_v1(decoded)) == owners.canonical(value[key])
    path, digest, _ = _reviewed_parent_public_depth_policy(scope, value, "selected-public")
    with pytest.raises(ValueError, match="^exact reviewed operation ID required$"):
        owners.checked_reviewed_preservation_parent_policy_v3(path, digest)
    assert not (scope.root / "attempts").exists()


@pytest.mark.parametrize("forbidden", ["private_key", "signers"])
def test_reviewed_parent_v3_expanded_envelope_still_refuses_private_DATA(
        reviewed_parent_normal_namespace_v2_scope, forbidden):
    scope = reviewed_parent_normal_namespace_v2_scope
    value, _ = _reviewed_parent_public_depth_value(17, 7)
    value["preparation"][forbidden] = []
    path, digest, _ = _reviewed_parent_public_depth_policy(scope, value, "private-DATA")
    with pytest.raises(ValueError, match="^private signer or signing key DATA is forbidden$"):
        owners.checked_reviewed_preservation_parent_policy_v3(path, digest)
    assert not (scope.root / "attempts").exists()


def test_reviewed_parent_v3_expanded_envelope_keeps_two_million_node_limit(
        reviewed_parent_normal_namespace_v2_scope):
    scope = reviewed_parent_normal_namespace_v2_scope
    value, _ = _reviewed_parent_public_depth_value(0, 0)
    # Intentionally malformed shallow DATA exceeds the raw node budget before
    # any tagged decoding. Zero values keep the serialized body below4MiB.
    value["kwargs"] = [0] * 2_000_000
    assert len(owners.canonical(value)) < 4 * 1024**2
    path, digest, _ = _reviewed_parent_public_depth_policy(scope, value, "node-bound")
    with pytest.raises(ValueError, match="^bounded public native input DATA tree required$"):
        owners.checked_reviewed_preservation_parent_policy_v3(path, digest)
    assert not (scope.root / "attempts").exists()


def test_reviewed_parent_v3_expanded_envelope_keeps_four_MiB_input_limit(
        reviewed_parent_normal_namespace_v2_scope):
    scope = reviewed_parent_normal_namespace_v2_scope
    value, _ = _reviewed_parent_public_depth_value(0, 0)
    value["preparation"]["manifest_json"] = "x" * (4 * 1024**2)
    path, digest, native = _reviewed_parent_public_depth_policy(scope, value, "byte-bound")
    assert native["bytes"] > 4 * 1024**2
    with pytest.raises(ValueError, match="^bounded original public native input DATA descriptor required$"):
        owners.checked_reviewed_preservation_parent_policy_v3(path, digest)
    assert not (scope.root / "attempts").exists()


# Canonical Source roots below are metadata only. Tiny Git worktrees exercise
# the maintained Source42 resolver; the existing /proc fixture remains modeled.
@pytest.fixture
def reviewed_parent_linked_source_roots(reviewed_parent_source_delegation_scope, tmp_path, monkeypatch):
    scope = reviewed_parent_source_delegation_scope
    # Test-owned Git configuration: no ambient templates, hooks, credentials or
    # signing program may run while constructing these ordinary metadata trees.
    templates = tmp_path / "git-templates-empty"; templates.mkdir(mode=0o700)
    hooks = tmp_path / "git-hooks-empty"; hooks.mkdir(mode=0o700)
    for key in tuple(owners.os.environ):
        if key.startswith("GIT_"): monkeypatch.delenv(key, raising=False)
    for key, value in {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_SYSTEM": owners.os.devnull,
            "GIT_CONFIG_GLOBAL": owners.os.devnull, "GIT_TERMINAL_PROMPT": "0",
            "GIT_TEMPLATE_DIR": str(templates), "GIT_CONFIG_COUNT": "3",
            "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": str(hooks),
            "GIT_CONFIG_KEY_1": "commit.gpgSign", "GIT_CONFIG_VALUE_1": "false",
            "GIT_CONFIG_KEY_2": "tag.gpgSign", "GIT_CONFIG_VALUE_2": "false",
            "PATH": owners.os.pathsep.join((str(Path(sys.executable).parent), "/usr/bin", "/bin"))}.items():
        monkeypatch.setenv(key, value)
    git_prefix = ["/usr/bin/git", "-c", "core.hooksPath=" + str(hooks),
        "-c", "commit.gpgSign=false", "-c", "tag.gpgSign=false", "-c", "init.templateDir=" + str(templates)]
    module = owners.importlib.import_module("ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease")
    source_pin = pin(Path(module.__file__))
    lanes = []
    for index, root in enumerate(scope.roots):
        for arguments in (("init", "--template=" + str(templates)), ("-c", "user.name=Source fixture", "-c", "user.email=source@example.invalid",
                "commit", "--no-verify", "--no-gpg-sign", "--allow-empty", "-m", "fixture")):
            result = owners.subprocess.run([*git_prefix, "-C", str(root), *arguments],
                capture_output=True, text=True, timeout=5)
            assert result.returncode == 0, result.stderr
        lane = tmp_path / ("linked-source-%d" % index)
        result = owners.subprocess.run([*git_prefix, "-C", str(root), "worktree", "add", "--detach", str(lane), "HEAD"],
            capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        lanes.append(lane)
    binding = owners._reviewed_parent_prepare_source_root_binding(source_pin, lanes)
    return SimpleNamespace(base=scope, module=module, source_pin=source_pin, lanes=tuple(lanes), binding=binding)


def test_reviewed_parent_linked_worktree_Source_roots_keep_strict_canonical_FD_handoff(reviewed_parent_linked_source_roots):
    s = reviewed_parent_linked_source_roots
    expected = owners._reviewed_parent_check_source_root_binding(s.binding, s.source_pin, s.lanes)
    assert expected == tuple(s.base.roots) and expected != s.lanes
    arguments = {**s.base.args, "expected_roots": expected}
    result = owners.verify_reviewed_preservation_parent_source_observation_v1(owners.canonical(s.base.value), **arguments)
    assert result["source_fds"] == s.base.value["source_fds"]
    assert "original_success_authenticated" not in result and "source_closed" not in result


@pytest.mark.parametrize("fault", ["physical_receipt_root", "physical_fd_root", "crossed_roots", "wrong_lock"])
def test_reviewed_parent_canonical_root_bridge_refuses_mismatched_handoff(reviewed_parent_linked_source_roots, fault):
    s = reviewed_parent_linked_source_roots
    value = copy.deepcopy(s.base.value)
    expected = owners._reviewed_parent_check_source_root_binding(s.binding, s.source_pin, s.lanes)
    if fault == "physical_receipt_root": value["source_receipts"][0]["canonical_root"] = str(s.lanes[0])
    elif fault == "physical_fd_root": value["source_fds"][0]["root"] = str(s.lanes[0])
    elif fault == "crossed_roots": expected = tuple(reversed(expected))
    else: value["source_fds"][0]["lock_path"] = str(s.lanes[0] / "workspace/control-plane/source-leases/source.lock")
    with pytest.raises(ValueError):
        owners.verify_reviewed_preservation_parent_source_observation_v1(owners.canonical(value),
            **{**s.base.args, "expected_roots": expected})


def test_reviewed_parent_canonical_binding_rechecks_without_Git_or_Source_after_prepare(
        reviewed_parent_linked_source_roots, monkeypatch):
    s = reviewed_parent_linked_source_roots
    def forbidden(*args, **kwargs): pytest.fail("cached Source binding spawned a process")
    monkeypatch.setattr(s.module.subprocess, "run", forbidden)
    for _ in range(3):
        assert owners._reviewed_parent_check_source_root_binding(s.binding, s.source_pin, s.lanes) == tuple(s.base.roots)


@pytest.mark.parametrize("target", ["canonical_source_root", "_git", "_absolute"])
@pytest.mark.parametrize("mutation", ["identity", "code"])
def test_reviewed_parent_canonical_binding_refuses_Source_callable_drift(
        reviewed_parent_linked_source_roots, monkeypatch, target, mutation):
    s = reviewed_parent_linked_source_roots
    def forbidden(*args, **kwargs): raise AssertionError("changed Source canonicalizer was invoked")
    function = getattr(s.module, target)
    if mutation == "identity": monkeypatch.setattr(s.module, target, forbidden)
    else: monkeypatch.setattr(function, "__code__", forbidden.__code__)
    with pytest.raises(ValueError, match="pinned canonical Source function changed"):
        owners._reviewed_parent_check_source_root_binding(s.binding, s.source_pin, s.lanes)


@pytest.mark.parametrize("fault", ["unissued", "source_pin", "physical_roots"])
def test_reviewed_parent_canonical_binding_refuses_crossed_or_unissued_metadata(reviewed_parent_linked_source_roots, fault):
    s = reviewed_parent_linked_source_roots; binding, source_pin, roots = s.binding, dict(s.source_pin), s.lanes
    if fault == "unissued": binding = object.__new__(owners._ReviewedParentSourceRoots)
    elif fault == "source_pin": source_pin["sha256"] = "f" * 64
    else: roots = tuple(reversed(roots))
    with pytest.raises(ValueError, match="canonical Source root binding"):
        owners._reviewed_parent_check_source_root_binding(binding, source_pin, roots)


def test_reviewed_parent_canonical_binding_preserves_duplicate_READ_roots(reviewed_parent_linked_source_roots):
    s = reviewed_parent_linked_source_roots
    raw = (s.lanes[0], s.lanes[0])
    binding = owners._reviewed_parent_prepare_source_root_binding(s.source_pin, raw)
    assert owners._reviewed_parent_check_source_root_binding(binding, s.source_pin, raw) == (s.base.roots[0], s.base.roots[0])


# Ordinary @2 delegation controls. /proc, Source42 and Native boundaries below
# are explicit DATA seams; no control establishes a genuine held lease/result.
def _immutable_parent_plan_DATA(metadata_scope):
    s=_immutable_source_qualification_RED_scope(metadata_scope)
    policy=copy.deepcopy(s.policy)
    profile=copy.deepcopy(s.profile)
    profile.update(schema='agent-supervisor/immutable-source-qualification@2',
        source_custody_mode='delegated-original-parent-read/v1')
    policy['validation']['source_qualification']=profile
    policy_pin=put(s.scope.state/'delegated-qualification-policy.json',policy)
    nonce='4f'*32
    context=dict(parent_identity={'pid':4100,'birth':'12345'},
        parent_nonce_sha256=hashlib.sha256(bytes.fromhex(nonce)).hexdigest(),
        parent_source_observation_path=str(s.invocation/'parent-source-observation.json'),
        execution_request_path=str(owners.validation_control_directory(policy,s.snapshot,s.invocation)/'execution-request.json'),
        source_root_bindings=[{'physical_root':root,'canonical_root':root}
            for root in (policy['datasets_root'],str(s.scope.control))])
    plan=owners.validation_plan(policy,policy_pin,s.snapshot,s.snapshot_pin,s.scope.control,s.invocation,
        source_parent_context=context)
    return SimpleNamespace(s=s,policy=policy,profile=profile,policy_pin=policy_pin,context=context,plan=plan)


def test_immutable_source_parent_plan_is_explicit_and_keeps_snapshot_and_v5_request(metadata_scope):
    f=_immutable_parent_plan_DATA(metadata_scope);s=f.s
    snapshot_raw=Path(s.snapshot_pin['path']).read_bytes()
    assert owners.checked_source_qualification(f.policy)['snapshot']==s.snapshot
    assert owners._source_qualification_plan(f.policy,f.plan,s.invocation)['profile']==f.profile
    assert {key:f.plan[key]for key in owners._IMMUTABLE_PARENT_FIELDS}==f.context
    assert 'parent_nonce'not in f.plan and 'parent_nonce'not in s.snapshot
    request=owners.validation_execution_request(f.policy,f.plan,put(s.invocation/'validation-plan.json',f.plan))
    assert len(request)==18 and request['schema']=='owned-validation-request/v5'
    assert {key:request[key]for key in ('cpu_slots','memory_mb','storage_bytes','child_process_slots','timeout_seconds')}==dict(
        cpu_slots=1,memory_mb=1536,storage_bytes=16777216,child_process_slots=4,timeout_seconds=300)
    assert f.plan['required_candidate_bodies']==[]
    assert Path(s.snapshot_pin['path']).read_bytes()==snapshot_raw
    assert owners._source_qualification_receipt_schema(owners.checked_source_qualification(f.policy))==\
        'agent-supervisor/immutable-source-qualification-receipt@2'


@pytest.mark.parametrize('fault',['mode-missing','mode-wrong','schema1-with-mode','extra-profile',
    'retry-mixed','missing-context','wrong-parent-type','wrong-observation-path','wrong-public-path','extra-plan','omitted-source'])
def test_immutable_source_parent_profile_and_complete_plan_fail_closed(metadata_scope,fault):
    f=_immutable_parent_plan_DATA(metadata_scope);policy=copy.deepcopy(f.policy);plan=copy.deepcopy(f.plan)
    if fault=='mode-missing':policy['validation']['source_qualification'].pop('source_custody_mode')
    elif fault=='mode-wrong':policy['validation']['source_qualification']['source_custody_mode']='caller-held'
    elif fault=='schema1-with-mode':policy['validation']['source_qualification']['schema']=f.s.profile['schema']
    elif fault=='extra-profile':policy['validation']['source_qualification']['source_closed']=True
    elif fault=='retry-mixed':policy['validation']['validation_ledger_busy_retry']={}
    elif fault=='missing-context':
        with pytest.raises(ValueError,match='trusted immutable parent plan context'):
            owners.validation_plan(policy,f.policy_pin,f.s.snapshot,f.s.snapshot_pin,f.s.scope.control,f.s.invocation)
        return
    elif fault=='wrong-parent-type':plan['parent_identity']['pid']=True
    elif fault=='wrong-observation-path':plan['parent_source_observation_path']+='.foreign'
    elif fault=='wrong-public-path':plan['execution_request_path']=str(f.s.invocation/'execution-request.json')
    elif fault=='extra-plan':plan['caller_claimed_Source_closed']=True
    else:plan['source_pins'].pop(next(iter(plan['source_pins'])))
    with pytest.raises(ValueError):
        owners._source_qualification_plan(policy,plan,f.s.invocation)


@pytest.fixture
def immutable_parent_observation_DATA(reviewed_parent_source_delegation_scope,tmp_path):
    s=reviewed_parent_source_delegation_scope
    physical=[tmp_path/'published-D',tmp_path/'immutable-C']
    for root in physical:root.mkdir()
    rows=[{'physical_root':str(path),'canonical_root':str(root)}for path,root in zip(physical,s.roots)]
    value=copy.deepcopy(s.value);value.update(schema='agent-supervisor/immutable-source-parent-observation@1',
        source_root_bindings=rows)
    arguments={key:item for key,item in s.args.items()if key!='expected_roots'}
    arguments['expected_source_root_bindings']=rows
    return SimpleNamespace(base=s,value=value,args=arguments,physical=physical,rows=rows)


def test_immutable_source_parent_live_verifier_reuses_kernel_and_chain_without_source_admission(
        immutable_parent_observation_DATA):
    f=immutable_parent_observation_DATA;raw=owners.canonical(f.value)
    value=owners.verify_immutable_source_parent_observation_v1(raw,**f.args)
    assert set(value)=={'observation_sha256','request_sha256','nonce_sha256','parent_identity','source_fds','source_root_bindings'}
    assert value['source_root_bindings']==f.rows and value['source_fds']==f.value['source_fds']
    assert {call[2]for call in f.base.calls if call[0]=='fd'}=={30,31}
    assert 'source_closed'not in value and 'native_closed'not in value and 'qualification_passed'not in value
    # The same held-fd core still accepts the unchanged reviewed schema/default.
    assert owners.verify_reviewed_preservation_parent_source_observation_v1(
        owners.canonical(f.base.value),**f.base.args)['source_fds']==f.base.value['source_fds']


@pytest.mark.parametrize('fault',['old-schema','missing-map','crossed-map','physical-root','canonical-root',
    'extra-map-field','nonce','parent-birth','worker-chain','foreign-command','missing-fd','write-lock',
    'unlocked','replaced-inode','write-receipt','boolean-fd','extra-observation','duplicate-json'])
def test_immutable_source_parent_live_verifier_rejects_unheld_or_crossed_delegation(
        immutable_parent_observation_DATA,fault):
    f=immutable_parent_observation_DATA;value=copy.deepcopy(f.value)
    if fault=='old-schema':value['schema']='reviewed-preservation-parent-source-observation/v1'
    elif fault=='missing-map':value.pop('source_root_bindings')
    elif fault=='crossed-map':value['source_root_bindings'].reverse()
    elif fault=='physical-root':value['source_root_bindings'][0]['physical_root']+='.foreign'
    elif fault=='canonical-root':value['source_root_bindings'][0]['canonical_root']+='.foreign'
    elif fault=='extra-map-field':value['source_root_bindings'][0]['source_closed']=True
    elif fault=='nonce':value['nonce']='00'*32
    elif fault=='parent-birth':f.base.state['facts'][4100]['birth']='9999'
    elif fault=='worker-chain':f.base.state['facts'][4101]['ppid']=9999
    elif fault=='foreign-command':f.base.state['cmdline']=b'/foreign/runner\x00'
    elif fault=='missing-fd':value['source_fds'][0]['fd']=999
    elif fault=='write-lock':f.base.state['fd'][30]['fdinfo']=f.base.state['fd'][30]['fdinfo'].replace('READ','WRITE')
    elif fault=='unlocked':f.base.state['fd'][30]['fdinfo']='flags: 0100002\n'
    elif fault=='replaced-inode':value['source_fds'][0]['inode']+=1
    elif fault=='write-receipt':value['source_receipts'][0]['mode']='write'
    elif fault=='boolean-fd':value['source_fds'][0]['fd']=True
    elif fault=='extra-observation':value['source_READ_closed']=True
    raw=owners.canonical(value)
    if fault=='duplicate-json':raw=raw[:-1]+b',"schema":"agent-supervisor/immutable-source-parent-observation@1"}'
    with pytest.raises((ValueError,OSError)):
        owners.verify_immutable_source_parent_observation_v1(raw,**f.args)


def test_immutable_source_parent_archive_join_is_metadata_only_and_checks_all_five_fields(
        immutable_parent_observation_DATA,tmp_path,monkeypatch):
    f=immutable_parent_observation_DATA;invocation=tmp_path/'saved-invocation';invocation.mkdir()
    policy={'python':sys.executable,'datasets_root':str(f.physical[0]),'validation':{'resource_ledger':str(tmp_path/'never-opened-ledger')}}
    plan={'candidate_root':str(f.physical[1]),'parent_identity':f.value['parent_identity'],
        'parent_nonce_sha256':f.args['expected_nonce_sha256'],
        'parent_source_observation_path':str(invocation/'parent-source-observation.json'),
        'execution_request_path':str(tmp_path/'published'/'execution-request.json'),'source_root_bindings':f.rows}
    request=put(Path(plan['execution_request_path']),Path(f.value['request']['path']).read_bytes());value=copy.deepcopy(f.value)
    value['request']=request
    value['sdk_argv']=owners._immutable_source_parent_argv(policy,request)
    observation=put(Path(plan['parent_source_observation_path']),value)
    def forbidden(*args,**kwargs):pytest.fail('archived DATA verification invoked process/Source/Git observation')
    for name in ('public_source_lease','_immutable_source_owner','_reviewed_parent_proc_stat',
            '_reviewed_parent_fd_observation','_parent_source_child_chain','git_command'):
        monkeypatch.setattr(owners,name,forbidden)
    monkeypatch.setattr(owners.subprocess,'Popen',forbidden)
    summary=owners._immutable_source_parent_saved_observation(policy,plan,request,invocation)
    assert summary==dict(source_custody_mode='delegated-original-parent-read/v1',parent_source_observation=observation,
        parent_nonce_sha256=plan['parent_nonce_sha256'],parent_identity=plan['parent_identity'],source_root_bindings=f.rows)
    report={'schema':'immutable-source-parent-delegated-validation@1','source_READ_closed':False,
        'source_READ_admission_complete':False,**summary}
    assert owners._immutable_source_parent_report_binding(policy,plan,request,report,invocation)==summary
    for key in summary:
        bad=copy.deepcopy(report);bad[key]=None
        with pytest.raises(ValueError,match='report parent/observation/root mapping'):
            owners._immutable_source_parent_report_binding(policy,plan,request,bad,invocation)
    owners._immutable_source_parent_receipts(value['source_receipts'],f.rows,value)
    changed=copy.deepcopy(value['source_receipts']);changed[0]['waited_seconds']+=1
    with pytest.raises(ValueError,match='observed Source receipts differ'):
        owners._immutable_source_parent_receipts(changed,f.rows,value)
    assert not (invocation/'source-qualification-receipt.json').exists()


def test_immutable_source_parent_runtime_mapping_comes_only_from_original_resolver(tmp_path,monkeypatch):
    import os
    physical=[tmp_path/'published-R',tmp_path/'sealed-C'];canonical=[tmp_path/'primary-R',tmp_path/'sealed-C']
    for root in set(physical+canonical):root.mkdir()
    calls=[]
    def resolver(root):calls.append(root);return canonical[physical.index(root)]
    monkeypatch.setattr(owners,'_immutable_source_owner',lambda policy:(None,None,None,resolver))
    monkeypatch.setattr(owners,'checked_source_qualification',lambda policy:{'root':physical[1],
        'snapshot':{'binding':dict(namespace='ptr_lane_0',board_id='legal-ir',attempt_id='a'*32)}})
    monkeypatch.setattr(owners,'proc_identity',lambda pid:{'pid':pid,'birth':'12345'})
    policy={'datasets_root':str(physical[0]),'validation':{'attempt_root':str(tmp_path/'attempts'),
        'ordinary_scratch_profile':{'schema':'owned-validation-ordinary-scratch-live/v1'},'source_qualification':{'fixture_only':True}}}
    value=owners._immutable_source_parent_runtime(policy,tmp_path/'invocation')
    assert calls==physical and value['source_root_bindings']==[{'physical_root':str(p),'canonical_root':str(c)}
        for p,c in zip(physical,canonical)]
    assert value['parent_identity']['pid']==os.getpid() and len(value['parent_nonce'])==64
    assert hashlib.sha256(bytes.fromhex(value['parent_nonce'])).hexdigest()==value['parent_nonce_sha256']


@pytest.mark.parametrize('slot',[0,1])
def test_immutable_source_parent_live_owner_rejects_correlated_foreign_canonical_mapping(tmp_path,monkeypatch,slot):
    import types
    physical=[tmp_path/'published-R',tmp_path/'sealed-C']
    canonical=[tmp_path/'primary-R',tmp_path/'primary-C']
    for root in physical+canonical:root.mkdir()
    foreign=tmp_path/'foreign-primary';foreign.mkdir()
    rows=[{'physical_root':str(p),'canonical_root':str(c)}for p,c in zip(physical,canonical)]
    # All untrusted future observation/receipt/plan copies can agree on the
    # wrong row; real trusted resolver output still determines admission.
    rows[slot]['canonical_root']=str(foreign)
    module=types.ModuleType('explicit_Source42_boundary_DATA')
    def original_generator_DATA():yield None
    module.source_lease=SimpleNamespace(__wrapped__=original_generator_DATA)
    calls=[]
    def resolver(root):calls.append(root);return canonical[physical.index(root)]
    module.canonical_source_root=resolver
    frame=SimpleNamespace(f_globals=vars(module),f_locals={})
    manager=SimpleNamespace(gen=SimpleNamespace(gi_code=original_generator_DATA.__code__,gi_frame=frame))
    custody=SimpleNamespace(immutable_source_binding=(module,{'explicit_unexecuted_pin_DATA':True},
        original_generator_DATA.__code__,resolver),context={'plan':{'source_root_bindings':rows}},
        immutable_source_fds=[{'previous_slot_DATA':True}]if slot else [])
    monkeypatch.setattr(owners,'checked_descriptor',lambda pin:b'explicit_Source42_DATA')
    held=SimpleNamespace(canonical_root=str(foreign))
    with pytest.raises(ValueError,match='Source42 mapping changed'):
        owners._immutable_source_parent_record_fd(custody,manager,held,physical[slot])
    assert calls==[physical[slot]]
    assert len(custody.immutable_source_fds)==slot


# PID1 parser/typed-join controls. All process, FD, parent Source and closeout
# observations remain explicitly modeled DATA through the existing fixtures.
def _immutable_parent_observation_PID1_DATA(scope):
    scope.base.value['parent_identity']['pid'] = 1  # Also updates the fixture FD observer's captured identity.
    scope.value['parent_identity']['pid'] = 1
    scope.args['expected_parent_identity']['pid'] = 1
    facts = scope.base.state['facts']
    parent = facts.pop(4100)
    parent.update(pid=1, ppid=0)
    facts[1] = parent
    facts[4101]['ppid'] = 1
    for row in scope.base.state['fd'].values():
        row['fdinfo'] = row['fdinfo'].replace('READ  4100 ', 'READ  1 ')
    return scope


@pytest.mark.parametrize('profile', ['ordinary', 'reviewed'])
def test_parent_source_PID1_metadata_keeps_live_reader_contract(immutable_parent_observation_DATA, profile):
    f = _immutable_parent_observation_PID1_DATA(immutable_parent_observation_DATA)
    if profile == 'ordinary':
        result = owners.verify_immutable_source_parent_observation_v1(owners.canonical(f.value), **f.args)
    else:
        result = owners.verify_reviewed_preservation_parent_source_observation_v1(
            owners.canonical(f.base.value), **f.base.args)
    assert type(result['parent_identity']['pid']) is int and result['parent_identity']['pid'] == 1
    assert {call[1] for call in f.base.calls if call[0] == 'fd'} == {1}
    assert 'source_closed' not in result and 'qualification_passed' not in result


@pytest.mark.parametrize('profile', ['ordinary', 'reviewed'])
@pytest.mark.parametrize('live', [True, False])
def test_parent_source_observed_boolean_PID_is_not_typed_PID1(immutable_parent_observation_DATA, profile, live):
    f = _immutable_parent_observation_PID1_DATA(immutable_parent_observation_DATA)
    value = copy.deepcopy(f.value if profile == 'ordinary' else f.base.value)
    value['parent_identity']['pid'] = True
    assert value['parent_identity'] == f.args['expected_parent_identity']  # The Python equality trap under test.
    raw = owners.canonical(value)
    with pytest.raises(ValueError, match='typed observed parent Source identity'):
        if live and profile == 'ordinary':
            owners.verify_immutable_source_parent_observation_v1(raw, **f.args)
        elif live:
            owners.verify_reviewed_preservation_parent_source_observation_v1(raw, **f.base.args)
        else:
            owners._parent_source_observation_core(raw,
                expected_request_sha256=f.args['expected_request_sha256'],
                expected_parent_identity=f.args['expected_parent_identity'],
                expected_nonce_sha256=f.args['expected_nonce_sha256'], expected_sdk_argv=f.args['expected_sdk_argv'],
                expected_roots=f.base.roots,
                expected_schema=('agent-supervisor/immutable-source-parent-observation@1' if profile == 'ordinary'
                    else 'reviewed-preservation-parent-source-observation/v1'),
                source_root_bindings=f.rows if profile == 'ordinary' else None, live=False)
    assert f.base.calls == []  # Refused as typed metadata, before any modeled process/FD observation.


@pytest.fixture
def immutable_parent_PID1_archive_DATA(immutable_parent_observation_DATA, tmp_path, monkeypatch):
    f = _immutable_parent_observation_PID1_DATA(immutable_parent_observation_DATA)
    invocation = tmp_path / 'PID1-archive-DATA'
    invocation.mkdir()
    policy = {'python': sys.executable, 'datasets_root': str(f.physical[0]),
        'validation': {'resource_ledger': str(tmp_path / 'never-opened-PID1-ledger')}}
    plan = {'candidate_root': str(f.physical[1]), 'parent_identity': f.value['parent_identity'],
        'parent_nonce_sha256': f.args['expected_nonce_sha256'],
        'parent_source_observation_path': str(invocation / 'parent-source-observation.json'),
        'execution_request_path': str(tmp_path / 'PID1-public' / 'execution-request.json'), 'source_root_bindings': f.rows}
    request = put(Path(plan['execution_request_path']), Path(f.value['request']['path']).read_bytes())
    value = copy.deepcopy(f.value)
    value.update(request=request, sdk_argv=owners._immutable_source_parent_argv(policy, request))
    observation = put(Path(plan['parent_source_observation_path']), value)
    def forbidden(*args, **kwargs):
        pytest.fail('PID1 archived DATA parser invoked live Source/Git/process/FD observation')
    for name in ('public_source_lease', '_immutable_source_owner', '_reviewed_parent_proc_stat',
            '_reviewed_parent_fd_observation', '_parent_source_child_chain', 'git_command'):
        monkeypatch.setattr(owners, name, forbidden)
    monkeypatch.setattr(owners.subprocess, 'Popen', forbidden)
    summary = owners._immutable_source_parent_saved_observation(policy, plan, request, invocation)
    return SimpleNamespace(base=f, policy=policy, plan=plan, request=request, observation=observation,
        invocation=invocation, summary=summary)


def test_immutable_parent_report_boolean_PID_cannot_equal_saved_PID1(immutable_parent_PID1_archive_DATA):
    s = immutable_parent_PID1_archive_DATA
    report = {'schema': 'immutable-source-parent-delegated-validation@1', 'source_READ_closed': False,
        'source_READ_admission_complete': False, **copy.deepcopy(s.summary)}
    assert owners._immutable_source_parent_report_binding(s.policy, s.plan, s.request, report, s.invocation) == s.summary
    report['parent_identity']['pid'] = True
    assert report['parent_identity'] == s.summary['parent_identity']
    with pytest.raises(ValueError, match='report parent/observation/root mapping'):
        owners._immutable_source_parent_report_binding(s.policy, s.plan, s.request, report, s.invocation)


def _immutable_parent_PID1_closeout_DATA(scope, observation_scope, monkeypatch):
    """Existing typed closeout reader, modeled engine/Source boundaries only."""
    s = scope
    f = _immutable_parent_observation_PID1_DATA(observation_scope)
    s.c.request_pin = put(s.invocation / 'PID1-closeout-public/execution-request.json', s.c.raw)
    closeout, old_ref = _typed_v5_modeled_closeout(s, monkeypatch)
    s.c.policy.update(python=sys.executable, datasets_root=str(f.physical[0]))
    s.c.policy['validation']['source_qualification'] = {'schema': 'agent-supervisor/immutable-source-qualification@2'}
    s.c.policy['validation']['resource_ledger'] = str(s.invocation / 'never-opened-PID1-ledger')
    plan = {'candidate_root': str(f.physical[1]), 'parent_identity': f.value['parent_identity'],
        'parent_nonce_sha256': f.args['expected_nonce_sha256'],
        'parent_source_observation_path': str(s.invocation / 'parent-source-observation.json'),
        'execution_request_path': s.c.request_pin['path'], 'source_root_bindings': f.rows}
    put(s.invocation / 'validation-plan.json', plan)
    value = copy.deepcopy(f.value)
    value.update(request=s.c.request_pin, sdk_argv=owners._immutable_source_parent_argv(s.c.policy, s.c.request_pin))
    put(Path(plan['parent_source_observation_path']), value)
    saved = owners._immutable_source_parent_saved_observation(s.c.policy, plan, s.c.request_pin, s.invocation)
    summary = {'explicit_modeled_verifier': True, **copy.deepcopy(saved)}
    custody = closeout['parent_source_custody']
    custody.update(copy.deepcopy(saved))
    custody['verified_checks'] = copy.deepcopy(summary)
    custody['source_receipts'] = copy.deepcopy(value['source_receipts'])
    path = Path(old_ref['path'])
    path.chmod(0o600)
    path.write_bytes(owners.canonical(closeout) + b'\n')
    path.chmod(0o400)
    return closeout, pin(path), summary


@pytest.mark.parametrize('fault', ['custody_PID', 'verified_PID', 'receipt_cooperative'])
def test_immutable_parent_archived_custody_preserves_exact_PID1_and_receipt_types(
        typed_v5_modeled_boundaries, immutable_parent_observation_DATA, monkeypatch, fault):
    s = typed_v5_modeled_boundaries
    closeout, reference, summary = _immutable_parent_PID1_closeout_DATA(s, immutable_parent_observation_DATA, monkeypatch)
    assert owners._verify_v5_parent_source_closeout(s.c.policy, s.c.request_pin, s.invocation, reference, summary) == closeout
    custody = closeout['parent_source_custody']
    if fault == 'custody_PID':
        custody['parent_identity']['pid'] = True
    elif fault == 'verified_PID':
        custody['verified_checks']['parent_identity']['pid'] = True
    else:
        custody['source_receipts'][0]['cooperative'] = 1
    path = Path(reference['path'])
    path.chmod(0o600)
    path.write_bytes(owners.canonical(closeout) + b'\n')
    path.chmod(0o400)
    with pytest.raises(ValueError, match='archived'):
        owners._verify_v5_parent_source_closeout(s.c.policy, s.c.request_pin, s.invocation, pin(path), summary)


def test_immutable_parent_receipt_boolean_PID_summary_refuses_before_parent_closeout(
        immutable_parent_PID1_archive_DATA, tmp_path, monkeypatch):
    """Actual receipt parser; upstream proof readers are explicitly modeled."""
    s = immutable_parent_PID1_archive_DATA
    invocation = tmp_path / 'PID1-receipt-DATA'
    invocation.mkdir()
    freeze = put(tmp_path / 'PID1-receipt-snapshot.json', {'explicit_snapshot_DATA': True})
    profile = {'schema': 'agent-supervisor/immutable-source-qualification@2'}
    commitment = 'a' * 64
    context = {'profile': profile, 'snapshot_ref': freeze, 'snapshot': {'source_commitment': commitment},
        'root': Path(s.plan['candidate_root'])}
    policy = {'validation': {'source_qualification': profile}}
    terminal = put(invocation / 'original-terminal.json', {'explicit_terminal_DATA': True})
    put(invocation / 'validation-plan.json', {'explicit_plan_DATA': True})
    bad_summary = copy.deepcopy(s.summary)
    bad_summary['parent_identity']['pid'] = True
    assert bad_summary == s.summary
    receipt = {'schema': 'agent-supervisor/immutable-source-qualification-receipt@2', 'snapshot': freeze,
        'source_qualification': profile, 'source_commitment': commitment, 'passed': True,
        'source_closed': True, 'native_closed': True, 'processes_closed': True, 'storage_closed': True,
        'cleanup_errors': [], 'original_receipt_reference': terminal['path'], 'original_receipt_descriptor': terminal,
        'verified_checks': bad_summary, 'parent_source_closeout': {'explicit_unreached_closeout_DATA': True},
        'repair_authority': False, 'adoption_authority': False, 'activation_authority': False, 'completion_authority': False}
    receipt_pin = put(invocation / 'source-qualification-receipt.json', receipt)
    args = SimpleNamespace(policy=tmp_path / 'unread-policy.json', policy_sha256='b' * 64,
        receipt_path=Path(receipt_pin['path']), invocation_root=invocation, freeze_path=Path(freeze['path']),
        candidate_root=context['root'], source_commitment=commitment)
    monkeypatch.setattr(owners, 'checked_policy', lambda *a, **k: policy)
    monkeypatch.setattr(owners, 'checked_source_qualification', lambda *a, **k: context)
    monkeypatch.setattr(owners, '_source_qualification_plan', lambda *a, **k: context)
    monkeypatch.setattr(owners, 'original_execution_request', lambda *a, **k: s.request)
    monkeypatch.setattr(owners, 'verify_validation_evidence', lambda *a, **k: copy.deepcopy(s.summary))
    def forbidden(*a, **k):
        pytest.fail('typed-invalid receipt reached parent Source closeout reader')
    monkeypatch.setattr(owners, 'verify_v5_parent_source_closeout', forbidden)
    with pytest.raises(ValueError, match='immutable qualification verified summary differs'):
        owners.verify_source_qualification(args)


def test_delegated_parent_source_qualification_validate_persists_failed_cleanup_without_report_acceptance(metadata_scope, monkeypatch):
    """Post-finish public metadata flow only; original runner is modeled DATA."""
    f = _immutable_parent_plan_DATA(metadata_scope)
    source_parent = {**copy.deepcopy(f.context), 'parent_nonce': '4f'*32}
    monkeypatch.setattr(owners, 'checked_policy', lambda *a, **k: f.policy)
    monkeypatch.setattr(owners, '_immutable_source_parent_runtime', lambda *a, **k: source_parent)
    result_custody = {'source_closed': True, 'native_closed': True,
        'closure_kind': 'original_failed_retained_native_closed', 'storage_status': 'retained',
        'storage_release_authorized': False, 'retry_authorized': False}
    calls = []
    def original_runner(policy, argv, cwd, timeout, *, validation_context):
        calls.append('original_runner_DATA')
        plan = validation_context['plan']
        assert owners._source_qualification_plan(policy, plan, f.s.invocation)['profile'] == f.profile
        assert validation_context['source_parent'] == source_parent and timeout == 360
        assert plan['execution_request_path'] == validation_context['request_pin']['path']
        return SimpleNamespace(returncode=1, stdout='{"explicit_completed_failed_cli_DATA":true}', stderr=''), {
            'parent_source_custody': copy.deepcopy(result_custody)}
    monkeypatch.setattr(owners, 'original_runner_wait', original_runner)
    def forbidden(*a, **k):
        pytest.fail('cleanup-only public failure entered qualification, negative diagnosis or hold')
    for name in ('verify_validation_evidence', 'capture_negative_validation', '_source_qualification_failed_custody'):
        monkeypatch.setattr(owners, name, forbidden)
    monkeypatch.setattr(owners._V5ValidationSourceCustody, 'hold', forbidden)
    args = SimpleNamespace(policy=Path(f.policy_pin['path']), policy_sha256=f.policy_pin['sha256'],
        freeze_path=Path(f.s.snapshot_pin['path']), source_commitment=f.s.snapshot['source_commitment'],
        candidate_root=f.s.scope.control, invocation_root=f.s.invocation,
        receipt_path=f.s.invocation/'source-qualification-receipt.json')
    assert owners.validate(args) == 1
    saved = json.loads((f.s.invocation/'failed-source-qualification.json').read_bytes())
    assert saved['parent_source_custody'] == result_custody and saved['passed'] is False
    assert saved['storage_closed'] is saved['semantic_negative'] is saved['retry_authorized'] is False
    assert calls == ['original_runner_DATA'] and not args.receipt_path.exists()


@pytest.mark.parametrize('delegated', [False, True])
def test_immutable_source_qualification_report_cannot_retag_other_profile_DATA(metadata_scope, delegated):
    """Modeled successful process metadata cannot cross @1/@2 report schemas."""
    if delegated:
        f = _immutable_parent_plan_DATA(metadata_scope)
        s, policy, plan = f.s, f.policy, f.plan
        plan_pin = put(s.invocation / 'validation-plan.json', plan)
        local = put(s.invocation / 'execution-request.json', owners.validation_execution_request(policy, plan, plan_pin))
        request = owners.publish_validation_request(policy, s.snapshot, s.invocation, local)
        offered_schema = owners.WORK_SCHEMA
    else:
        s, policy, plan, plan_pin, request, _, _ = _immutable_qualification_bound_request_DATA(metadata_scope)
        offered_schema = owners.IMMUTABLE_PARENT_WORKER_SCHEMA
    attempt = Path(plan['attempt_directory'])
    owner = {'pid': 4100, 'birth': '12345'}
    record = dict(status='released', artifacts_durable_asserted=True, native_lease_id='f'*32,
        validation_request_sha256=request['sha256'], owner_pid=owner['pid'], owner_birth=owner['birth'])
    result = put(attempt / 'logs/result.json', dict(schema='owned-validation-result/v1',
        completed=True, returncode=0, request_sha256=request['sha256']))
    terminal = dict(result=result['path'], reservation=dict(status='released', record=record,
        resource_lease=dict(released=True, lease_id=record['native_lease_id'], owner_pid=owner['pid']),
        cleanup_error=None, process_observation=dict(live_processes=0, out_of_group_live_identities=[], tracking_error=False)))
    engine = put(s.invocation / 'modeled-report-domain-engine.py', b'# Modeled process DATA only.\n')
    put(s.invocation / 'runner-wait.json', dict(request=request, original_process_owner_returned=True,
        returncode=0, stdout=owners.canonical(terminal).decode(), runner_identity=owner,
        actual_Popen_returned=True, actual_process_waited=True, actual_process_returncode=0, engine_source=engine))
    report = put(attempt / 'exports/trusted-validation.json', dict(schema=offered_schema, passed=True))
    with pytest.raises(ValueError, match='trusted source/semantic result missing'):
        owners.verify_validation_evidence(policy, plan, request, terminal, s.invocation)
    assert Path(report['path']).read_bytes() == owners.canonical(dict(schema=offered_schema, passed=True)) + b'\n'
    assert not (s.invocation / 'source-qualification-receipt.json').exists()


# The original reviewed Native parent has no repair control_release field.
# These controls cross the real success verifier's namespace gate, then stop
# at a named, deliberately modeled packet join. They mint no parent handle,
# Source proof, Native execution, admission, or final validation acceptance.
class _ReviewedParentNamespaceJoinReached(RuntimeError):
    pass


def _reviewed_parent_namespace_success_scope_v1(scope, parent_schema, monkeypatch):
    s = scope
    profile = {"profile": "reviewed-preservation-normal-namespaces/v2",
        "fullname": "ipfs_datasets_py.logic.backends",
        "path": str(s.attempt / "modeled-unimported-namespace"),
        "device": 41, "inode": 73, "initializer_absent": True}
    s.policy["schema"] = parent_schema
    s.policy["validation"].update(
        ordinary_scratch_profile={"schema": "owned-validation-ordinary-scratch-live/v1"},
        normal_namespace_profiles=[copy.deepcopy(profile)])
    s.policy["operation"] = {"fixture_only": True, "no_native_operation_claimed": True}
    s.plan.update(schema={"reviewed-preservation-parent/v1": "reviewed-preservation-native-worker/v1",
        "reviewed-preservation-parent/v3": "reviewed-preservation-native-worker/v3"}[parent_schema],
        operation=copy.deepcopy(s.policy["operation"]), normal_namespace_profiles=[copy.deepcopy(profile)],
        parent_nonce_sha256="d" * 64)
    report_path = s.attempt / "exports/trusted-validation.json"
    s.report = json.loads(report_path.read_bytes())
    s.report.update(schema=s.plan["schema"], source_READ_closed=False,
        source_READ_admission_complete=False, source_custody_mode="delegated-original-parent-read/v1",
        parent_nonce_sha256=s.plan["parent_nonce_sha256"],
        declared_normal_namespace_profiles=[copy.deepcopy(profile)])
    s.report["native_binding"] = put(s.attempt / "exports/reviewed-preservation-native-binding.json",
        {"fixture_only": True, "not_a_native_result": True})
    s.namespace_join_calls = []

    def reached_native_join(*args, **kwargs):
        s.namespace_join_calls.append(True)
        raise _ReviewedParentNamespaceJoinReached("modeled native join reached after namespace proof")

    monkeypatch.setattr(owners, "_reviewed_parent_binding_v3", reached_native_join)
    return s


def _persist_reviewed_parent_namespace_success_scope_v1(scope):
    # Only this test's authored ordinary DATA is changed. No original evidence
    # or resource owner is entered, and the real packet join remains a sentinel.
    plan_path = scope.invocation / "validation-plan.json"
    plan_path.chmod(0o600)
    plan_path.write_bytes(owners.canonical(scope.plan) + b"\n")
    plan_path.chmod(0o400)
    scope.report["plan_sha256"] = pin(plan_path)["sha256"]
    report_path = scope.attempt / "exports/trusted-validation.json"
    report_path.chmod(0o600)
    report_path.write_bytes(owners.canonical(scope.report) + b"\n")
    report_path.chmod(0o400)


@pytest.mark.parametrize("parent_schema", ["reviewed-preservation-parent/v1", "reviewed-preservation-parent/v3"])
def test_reviewed_parent_success_verifier_uses_own_namespace_without_repair_control_release(
        reviewed_parent_modeled_durability_scope, monkeypatch, parent_schema):
    s = _reviewed_parent_namespace_success_scope_v1(
        reviewed_parent_modeled_durability_scope, parent_schema, monkeypatch)
    assert "control_release" not in s.policy
    before = copy.deepcopy((s.policy, s.plan))
    _persist_reviewed_parent_namespace_success_scope_v1(s)
    with pytest.raises(_ReviewedParentNamespaceJoinReached, match="after namespace proof"):
        owners.verify_validation_evidence(s.policy, s.plan, s.request_pin, s.terminal, s.invocation)
    assert s.namespace_join_calls == [True]
    assert (s.policy, s.plan) == before
    assert not (s.invocation / "parent-source-closeout.json").exists()
    assert not (s.invocation / "source-custody-unknown.json").exists()


@pytest.mark.parametrize("parent_schema", ["reviewed-preservation-parent/v1", "reviewed-preservation-parent/v3"])
@pytest.mark.parametrize("mutation", ["missing_plan", "missing_report", "changed_plan", "changed_report",
    "crossed_both", "boolean_inode"])
def test_reviewed_parent_success_verifier_refuses_crossed_policy_plan_report_namespaces(
        reviewed_parent_modeled_durability_scope, monkeypatch, parent_schema, mutation):
    s = _reviewed_parent_namespace_success_scope_v1(
        reviewed_parent_modeled_durability_scope, parent_schema, monkeypatch)
    if mutation == "missing_plan":
        s.plan.pop("normal_namespace_profiles")
    elif mutation == "missing_report":
        s.report.pop("declared_normal_namespace_profiles")
    elif mutation == "changed_plan":
        s.plan["normal_namespace_profiles"][0]["inode"] += 1
    elif mutation == "changed_report":
        s.report["declared_normal_namespace_profiles"][0]["inode"] += 1
    elif mutation == "crossed_both":
        s.plan["normal_namespace_profiles"][0]["inode"] += 1
        s.report["declared_normal_namespace_profiles"][0]["inode"] += 1
    else:
        s.report["declared_normal_namespace_profiles"][0]["inode"] = True
    _persist_reviewed_parent_namespace_success_scope_v1(s)
    with pytest.raises(ValueError, match="V5 immutable control namespace declaration differs"):
        owners.verify_validation_evidence(s.policy, s.plan, s.request_pin, s.terminal, s.invocation)
    assert s.namespace_join_calls == []
    assert not (s.invocation / "parent-source-closeout.json").exists()


# New failure-close controls model process/Source boundaries explicitly and use
# the real tiny saved operational artifacts. They execute no SDK, Native tool,
# Source lease, release, retry, original capability or separate runner.
class _BoundedFailedCloseUnknown(RuntimeError):
    pass


def _bounded_failed_close_poison(*args, **kwargs):
    raise AssertionError("unauthenticated failed-close callable was invoked")


def _bounded_failed_close_rewrite(path, value):
    path.chmod(0o600)
    path.write_bytes(value if type(value) is bytes else owners.canonical(value) + b"\n")
    path.chmod(0o400)
    return pin(path)


@pytest.fixture
def bounded_reviewed_failed_close_scope(typed_v5_modeled_boundaries,
        reviewed_parent_modeled_durability_scope, monkeypatch):
    s = typed_v5_modeled_boundaries
    r = reviewed_parent_modeled_durability_scope
    c = s.c
    # Preserve the preparation fixture's preauthored wait. The real finish must
    # publish its own wait exactly once in a separate fresh run invocation.
    r.prepared_invocation = r.invocation
    r.invocation = r.invocation.parent / "fresh-failed-close-invocation"
    r.invocation.mkdir(mode=0o700)
    engine_ref = pin(Path(s.engine.__file__).absolute())
    c.engine_source = engine_ref
    c.policy = copy.deepcopy(r.policy)
    c.policy["schema"] = "reviewed-preservation-parent/v3"
    c.policy["source_descriptors"] = [engine_ref]
    c.policy["validation"]["ordinary_scratch_profile"] = {"schema": "owned-validation-ordinary-scratch-live/v1"}
    request = json.loads(Path(r.request_pin["path"]).read_bytes())
    request.update(timeout_seconds=300, storage_bytes=16777216)
    c.request_pin = put(r.invocation / "modeled-failure-close-request.json", request)
    c.raw = owners.checked_descriptor(c.request_pin)
    c.request = request
    r.terminal["reservation"]["record"]["validation_request_sha256"] = c.request_pin["sha256"]
    result_path = r.attempt / "logs/result.json"
    result = json.loads(result_path.read_bytes())
    result["request_sha256"] = c.request_pin["sha256"]
    _bounded_failed_close_rewrite(result_path, result)
    c.context = {"plan": r.plan, "invocation": r.invocation, "frozen": {}}
    f = SimpleNamespace(base=s, real=r, c=c, terminal=r.terminal, events=s.events,
        held=[], failure=KeyError("modeled metadata failure"), completed=None,
        metadata_calls=0, delegation_calls=0, failed_write_seen=False,
        output_capture="complete", missing_observation=False, metadata_hook=None)
    # This fixture's Source/runtime authenticity is modeled, as in the existing
    # typed boundary fixture. Separate unmocked controls below own real callable
    # guard refusal. No issued Native handle or original success is fabricated.
    def modeled_reviewed_guard():
        return None
    monkeypatch.setattr(owners, "_reviewed_parent_guard", modeled_reviewed_guard)
    monkeypatch.setattr(owners, "_REVIEWED_PARENT_GUARD_OBJECT", modeled_reviewed_guard)
    monkeypatch.setattr(owners, "_REVIEWED_PARENT_GUARD_CODE", modeled_reviewed_guard.__code__)
    class ModeledCompletedPopen:
        def __init__(self, command, **kwargs):
            self.args = list(command)
            self.pid = 4101
            self.returncode = 0
        def communicate(self, **kwargs):
            return owners.canonical(f.terminal).decode(), ""
        def poll(self):
            pytest.fail("failed-close fixture must use the original typed observation")
        def wait(self, *args, **kwargs):
            pytest.fail("failed-close fixture must never reap or launch a process")
    monkeypatch.setattr(owners, "_ORIGINAL_POPEN", ModeledCompletedPopen)
    monkeypatch.setattr(s.engine.subprocess, "Popen", ModeledCompletedPopen)
    monkeypatch.setattr(owners, "proc_identity", lambda pid: {"pid": pid, "birth": "12345"})
    # Model construction/dispatch without changing the production class's
    # CPython allocation slots. Unmocked guard negatives remain separate below.
    monkeypatch.setattr(owners, "_V5ValidationSourceCustody", lambda *args, **kwargs: c)
    def modeled_trusted_dispatch(selected):
        assert selected is c
        return type(c).close_failed_operational_verification.__get__(c, type(c))
    monkeypatch.setattr(owners, "_reviewed_parent_failed_close_method_v1", modeled_trusted_dispatch)
    monkeypatch.setattr(s.engine, "time", SimpleNamespace(monotonic=lambda: s.now["value"]))
    # Run the unchanged engine emitter against this explicit fake communicate
    # boundary. No synthetic code is compiled under a canonical source name.
    actual_record = c.record_execution
    def modeled_record(completed, engine, records):
        f.completed = completed
        if f.missing_observation:
            object.__setattr__(completed, "_execution_observation", None)
        elif f.output_capture != "complete":
            before = completed.execution_observation
            values = {name: getattr(before, name) for name in owners._V5_EXECUTION_FIELDS}
            values["output_capture"] = f.output_capture
            object.__setattr__(completed, "_execution_observation", engine.CommandExecutionObservation(**values))
        return actual_record(completed, engine, records)
    c.record_execution = modeled_record
    def metadata_failure(*args, **kwargs):
        f.metadata_calls += 1
        if f.metadata_hook is not None:
            f.metadata_hook(*args, **kwargs)
        raise f.failure
    monkeypatch.setattr(owners, "verify_validation_evidence", metadata_failure)
    def held(error):
        f.held.append(error)
        raise _BoundedFailedCloseUnknown("modeled existing UNKNOWN hold")
    c.hold = held
    return f


def _run_bounded_failed_close_scope(scope):
    c = scope.c
    return owners.original_runner_wait(c.policy, c.argv, scope.real.invocation, 360,
        validation_context=c.context)


@pytest.mark.parametrize("metadata_stage", ["early_qualification", "report", "delegation"])
def test_reviewed_failed_close_rethrows_metadata_error_after_same_original_Source_exit(
        bounded_reviewed_failed_close_scope, monkeypatch, metadata_stage):
    f = bounded_reviewed_failed_close_scope
    if metadata_stage == "early_qualification":
        actual_verifier = next(state[0] for name, state in owners._REVIEWED_PARENT_FUNCTIONS
            if name == "verify_validation_evidence")
        monkeypatch.setattr(owners, "verify_validation_evidence", actual_verifier)
        def early(*args, **kwargs):
            f.metadata_calls += 1
            assert type(f.c._released_operational_raw) is bytes
            raise f.failure
        monkeypatch.setattr(owners, "_source_qualification_plan", early)
    elif metadata_stage == "delegation":
        def verified(*args):
            f.metadata_calls += 1
            return {"modeled_semantic_summary": True}
        monkeypatch.setattr(owners, "verify_validation_evidence", verified)
        def later(*args, **kwargs):
            f.delegation_calls += 1
            raise f.failure
        monkeypatch.setattr(owners, "_reviewed_parent_check_worker_delegation", later)
    before = len(owners._REVIEWED_PARENT_CAPABILITIES)
    with pytest.raises(KeyError) as caught:
        _run_bounded_failed_close_scope(f)
    assert caught.value is f.failure
    assert f.metadata_calls == 1
    assert f.delegation_calls == (1 if metadata_stage == "delegation" else 0)
    assert f.c.closed is True and f.events.count("source_exit") == 1 and f.held == []
    record = json.loads((f.real.invocation / "failed-operational-source-closeout.json").read_bytes())
    assert record["source_closed"] is record["native_closed"] is True
    assert record["closure_kind"] == "released_original_metadata_verification_failed"
    assert record["SDK_returncode"] == 0 and record["storage_status"] == "released"
    for name in ("validation_accepted", "original_success_authenticated", "native_execution_authenticated",
            "parent_handle_issued", "storage_release_authorized", "retry_authorized", "further_productive_work"):
        assert record[name] is False
    assert record["authority_deadline_monotonic"] == f.c.deadline == 1360.0
    assert len(owners._REVIEWED_PARENT_CAPABILITIES) == before
    assert f.base.native_calls == [] and not (f.real.invocation / "parent-source-closeout.json").exists()
    assert (f.real.prepared_invocation / "runner-wait.json").is_file()


@pytest.mark.parametrize("fault", ["retained_row", "request_sha", "owner_birth", "native_lease",
    "live_descendant", "out_of_group", "tracking_error", "result_request",
    "incomplete_EOF", "unknown_EOF", "missing_observation", "changed_stdout", "changed_request",
    "changed_wait", "expired_original"])
def test_reviewed_failed_close_unproven_original_never_closes_Source(
        bounded_reviewed_failed_close_scope, fault):
    f = bounded_reviewed_failed_close_scope
    reservation = f.terminal["reservation"]
    if fault == "retained_row":
        reservation["record"]["status"] = "retained"
    elif fault == "request_sha":
        reservation["record"]["validation_request_sha256"] = "f" * 64
    elif fault == "owner_birth":
        reservation["record"]["owner_birth"] = "54321"
    elif fault == "native_lease":
        reservation["resource_lease"]["lease_id"] = "foreign-lease"
    elif fault == "live_descendant":
        reservation["process_observation"]["live_processes"] = 1
    elif fault == "out_of_group":
        reservation["process_observation"]["out_of_group_live_identities"] = [{"pid": 77}]
    elif fault == "tracking_error":
        reservation["process_observation"]["tracking_error"] = "modeled uncertainty"
    elif fault == "result_request":
        p = f.real.attempt / "logs/result.json"
        value = json.loads(p.read_bytes()); value["request_sha256"] = "f" * 64
        _bounded_failed_close_rewrite(p, value)
    elif fault in ("incomplete_EOF", "unknown_EOF"):
        f.output_capture = "incomplete" if fault == "incomplete_EOF" else "unknown"
    elif fault == "missing_observation":
        f.missing_observation = True
    else:
        def change(*args, **kwargs):
            assert type(f.c._released_operational_raw) is bytes
            if fault == "changed_stdout":
                object.__setattr__(f.completed, "stdout", f.completed.stdout + " ")
            elif fault == "changed_request":
                _bounded_failed_close_rewrite(Path(f.c.request_pin["path"]), {"changed": True})
            elif fault == "changed_wait":
                p = Path(f.c.wait_ref["path"])
                value = json.loads(p.read_bytes()); value["returncode"] = 9
                _bounded_failed_close_rewrite(p, value)
            else:
                f.base.now["value"] = f.c.deadline
        f.metadata_hook = change
    with pytest.raises(_BoundedFailedCloseUnknown):
        _run_bounded_failed_close_scope(f)
    assert f.c.closed is False and "source_exit" not in f.events
    assert len(f.held) == 1 and f.base.native_calls == []
    assert not isinstance(f.held[0], FileExistsError)
    if fault in ("changed_stdout", "changed_request", "changed_wait", "expired_original"):
        assert f.metadata_calls == 1 and f.held == [f.failure]
        assert type(f.c._released_operational_raw) is bytes
    else:
        assert f.metadata_calls == 0 and isinstance(f.held[0], ValueError)
        expected = ("terminal EOF" if fault in ("incomplete_EOF", "unknown_EOF") else
            "typed immutable original V5 execution observation" if fault == "missing_observation" else
            "original runner PID/birth/wait binding differs" if fault == "owner_birth" else
            "original descendant custody remains unknown" if fault in ("live_descendant", "out_of_group", "tracking_error") else
            "original result mismatch" if fault == "result_request" else
            "original native/storage closure is not proven")
        assert expected in str(f.held[0])
    assert f.c.deadline == 1360.0
    assert not (f.real.invocation / "failed-operational-source-closeout.json").exists()


def test_reviewed_failed_close_preserves_detached_SDK_attestation_when_metadata_mutates_its_DATA_copy(
        bounded_reviewed_failed_close_scope):
    f = bounded_reviewed_failed_close_scope
    def changed_copy(policy, plan, request_pin, terminal, invocation):
        terminal["reservation"]["record"]["status"] = "changed-by-semantic-callback"
    f.metadata_hook = changed_copy
    with pytest.raises(KeyError) as caught:
        _run_bounded_failed_close_scope(f)
    assert caught.value is f.failure and f.c.closed and f.held == []
    assert f.metadata_calls == 1
    assert f.events.count("source_exit") == 1
    record = json.loads((f.real.invocation / "failed-operational-source-closeout.json").read_bytes())
    assert record["storage_status"] == "released" and record["validation_accepted"] is False


def test_reviewed_failed_close_Source_exit_error_retains_existing_UNKNOWN(
        bounded_reviewed_failed_close_scope):
    f = bounded_reviewed_failed_close_scope
    def failing_close():
        f.events.append("source_exit_failed")
        raise OSError("modeled Source exit failed")
    f.c.stack.close = failing_close
    with pytest.raises(_BoundedFailedCloseUnknown):
        _run_bounded_failed_close_scope(f)
    assert f.c.closed is False and f.events.count("source_exit_failed") == 1
    assert f.metadata_calls == 1
    assert f.held == [f.failure] and f.base.native_calls == []
    assert not (f.real.invocation / "failed-operational-source-closeout.json").exists()


def test_reviewed_failed_close_late_irreversible_Source_exit_never_accepts_or_renews(
        bounded_reviewed_failed_close_scope):
    f = bounded_reviewed_failed_close_scope
    def late_close():
        f.events.append("source_exit")
        f.base.now["value"] = 1361.0
    f.c.stack.close = late_close
    with pytest.raises(KeyError) as caught:
        _run_bounded_failed_close_scope(f)
    assert caught.value is f.failure and f.c.closed and f.held == []
    assert f.metadata_calls == 1
    record = json.loads((f.real.invocation / "failed-operational-source-closeout.json").read_bytes())
    assert record["late_source_close"] is True and record["validation_accepted"] is False
    assert record["authority_deadline_monotonic"] == f.c.deadline == 1360.0


def test_reviewed_failed_close_evidence_write_error_cannot_reenter_UNKNOWN_after_Source_exit(
        bounded_reviewed_failed_close_scope, monkeypatch):
    f = bounded_reviewed_failed_close_scope
    original = owners.new_json
    def failing_evidence(path, value):
        if Path(path).name == "failed-operational-source-closeout.json":
            assert f.c.closed and f.events.count("source_exit") == 1
            f.failed_write_seen = True
            raise OSError("modeled failed-closeout metadata write")
        return original(path, value)
    monkeypatch.setattr(owners, "new_json", failing_evidence)
    with pytest.raises(KeyError) as caught:
        _run_bounded_failed_close_scope(f)
    assert caught.value is f.failure and f.c.closed and f.held == []
    assert f.metadata_calls == 1 and f.failed_write_seen is True
    assert any("OSError" in note for note in caught.value.__notes__)
    assert (f.real.invocation / "runner-wait.json").is_file()
    assert (f.real.invocation / "runner-execution-observation.json").is_file()
    assert not (f.real.invocation / "failed-operational-source-closeout.json").exists()


@pytest.mark.parametrize("mode", ["ordinary", "BaseException"])
def test_reviewed_failed_close_does_not_expand_ordinary_or_BaseException_disposition(
        bounded_reviewed_failed_close_scope, mode):
    f = bounded_reviewed_failed_close_scope
    if mode == "ordinary":
        f.c.policy["schema"] = "explicit-modeled-verifier-only"
    else:
        f.failure = KeyboardInterrupt("modeled interruption remains UNKNOWN")
    with pytest.raises(_BoundedFailedCloseUnknown):
        _run_bounded_failed_close_scope(f)
    assert f.c.closed is False and "source_exit" not in f.events and f.held == [f.failure]
    assert f.metadata_calls == 1
    assert not (f.real.invocation / "failed-operational-source-closeout.json").exists()


@pytest.mark.parametrize("helper", ["_validation_released_operational_evidence_v1",
    "_validation_released_operational_projection_v1", "_reviewed_parent_failed_close_method_v1"])
@pytest.mark.parametrize("mutation", ["identity", "code"])
def test_reviewed_failed_close_guard_refuses_helper_changes_before_dispatch(monkeypatch, helper, mutation):
    custody = object.__new__(owners._V5ValidationSourceCustody)
    function = getattr(owners, helper)
    if mutation == "identity":
        monkeypatch.setattr(owners, helper, _bounded_failed_close_poison)
    else:
        monkeypatch.setattr(function, "__code__", _bounded_failed_close_poison.__code__)
    with pytest.raises(ValueError, match="(?i)(changed|binding)"):
        # The actual dispatch authenticates this unchanged guard before calling
        # a selector whose function object or __code__ may have been replaced.
        owners._reviewed_parent_guard()
        owners._reviewed_parent_failed_close_method_v1(custody)


@pytest.mark.parametrize("mutation", ["instance", "class", "code"])
def test_reviewed_failed_close_guard_refuses_replaced_recovery_method_before_call(monkeypatch, mutation):
    cls = owners._V5ValidationSourceCustody
    custody = object.__new__(cls)
    custody.closed = False
    if mutation == "instance":
        custody.close_failed_operational_verification = _bounded_failed_close_poison.__get__(custody, cls)
    elif mutation == "class":
        monkeypatch.setattr(cls, "close_failed_operational_verification", _bounded_failed_close_poison)
    else:
        monkeypatch.setattr(cls.close_failed_operational_verification, "__code__", _bounded_failed_close_poison.__code__)
    with pytest.raises(ValueError, match="(?i)(changed|binding)"):
        owners._reviewed_parent_failed_close_method_v1(custody)
    assert custody.closed is False


@pytest.fixture
def reviewed_parent_persisted_reader_version_model(request, tmp_path, monkeypatch):
    """Model only capability/native boundaries; never issue or accept a handle.

    Real tiny77/83 files and current loader/fence/domain code are retained. The
    strict persisted reader stops before seal_record and requires the keyword
    whose omission failed the genuine V3 caller. No private registry is edited.
    """
    def prepare(packet_version):
        assert packet_version in (2, 3)
        fixture = ("reviewed_parent_packet_join_scope" if packet_version == 2
                   else "reviewed_parent_packet_join_scope_v3")
        s = request.getfixturevalue(fixture)
        binding = copy.deepcopy(s.binding)
        projection = {"fixture_only": True, "modeled_execution": "not-an-original"}
        binding["execution_projection_sha256"] = owners.digest(projection)
        evidence = {"binding": binding, "attempt_directory": str(s.attempt),
            "reviewed_version": packet_version,
            **({"operation": copy.deepcopy(s.operation),
                "expected_packet_paths": list(s.operation["expected_packet_paths"])}
               if packet_version == 3 else {}),
            "inventory": [[row["path"], row["bytes"], row["sha256"], row["device"], row["inode"]]
                          for row in s.durable]}
        for name in ("request", "plan", "wait", "closeout", "durability"):
            evidence[name] = put(tmp_path / "explicit-model-boundary" / (name + ".json"),
                                 {"fixture_only": True, "not_an_original": name})
        sentinel = object()  # Neither the opaque class nor an issued capability.
        entry = (None, None, None, None, owners.canonical(evidence))
        calls, persisted_versions, native_resolutions, sealed = [], [], [], []
        class PersistedReaderReached(RuntimeError):
            pass
        def modeled_guard():
            return None
        def modeled_entry(value):
            assert value is sentinel
            return entry
        # Explicit test-only trust boundaries. Real unissued/callable negatives
        # remain separately selected; these objects cannot mint runtime custody.
        monkeypatch.setattr(owners, "_reviewed_parent_guard", modeled_guard)
        monkeypatch.setattr(owners, "_REVIEWED_PARENT_GUARD_OBJECT", modeled_guard)
        monkeypatch.setattr(owners, "_REVIEWED_PARENT_GUARD_CODE", modeled_guard.__code__)
        monkeypatch.setattr(owners, "_reviewed_parent_capability_entry", modeled_entry)
        def forbid_seal(value):
            sealed.append(value)
            pytest.fail("modeled routing control reached acceptance/sealing")
        transport = SimpleNamespace(contracts=SimpleNamespace(seal_record=forbid_seal))
        checked_project = object()
        receipt = {"reviewed_execution_evidence": projection}
        def replay_for(version):
            def replay(*args, **kwargs):
                assert version == packet_version
                calls.append(("retained_replay", version))
                return {"fixture_only": True}
            return replay
        def validate_for(version):
            def validate(*args, **kwargs):
                assert version == packet_version
                calls.append(("current_validation", version))
                return checked_project
            return validate
        def parse(raw, maximum):
            assert raw == dict(s.packet)["build-receipt.json"]
            return receipt
        def persisted(output, checked, offered_receipt, offered_transport, *, reviewed_version):
            # No default and no **kwargs: the pre-fix missing keyword fails
            # before this body instead of passing through a permissive mock.
            assert type(reviewed_version) is int and reviewed_version == packet_version
            assert output == s.attempt / binding["packet_root"]
            assert checked is checked_project and offered_receipt is receipt and offered_transport is transport
            persisted_versions.append(reviewed_version)
            calls.append(("persisted_reader", reviewed_version))
            raise PersistedReaderReached("strict persisted reader reached with explicit domain")
        native = SimpleNamespace(
            _reviewed_retained_packet=lambda packet: (dict(packet), binding["packet_inventory"]),
            _wire=lambda value, maximum: owners.canonical(value),
            _sha=lambda raw: hashlib.sha256(raw).hexdigest(), MAX_RECORD_BYTES=1024**2, MAX_RECEIPT_BYTES=1024**2,
            _live_original_inputs=lambda *args, **kwargs: b"explicit unchanged modeled inputs",
            _parse=parse, _owners=lambda: (None, transport), _live_read_persisted_packet=persisted,
            load_reviewed_norm_preservation_native_v2=replay_for(2),
            load_reviewed_norm_preservation_native_v3=replay_for(3),
            validate_reviewed_norm_preservation_native_v2=validate_for(2),
            validate_reviewed_norm_preservation_native_v3=validate_for(3))
        def native_owner(value, *, reviewed_version):
            assert type(reviewed_version) is int and reviewed_version == packet_version
            assert value == evidence
            native_resolutions.append(True)
            return native
        monkeypatch.setattr(owners, "_parent_native_owner", native_owner)
        arguments = dict(expected_receipt_sha256=binding["receipt_sha256"],
            expected_packet_sha256=binding["packet_sha256"], expected_binary_sha256=binding["binary_sha256"],
            expected_preservation_sha256=binding["preparation_sha256"], expected_graph_sha256=binding["graph_sha256"],
            expected_admission_sha256=binding["admission_sha256"], toolchain=binding["toolchain"])
        return SimpleNamespace(packet=s.packet, handle=sentinel, args=arguments, calls=calls,
            persisted_versions=persisted_versions, native_resolutions=native_resolutions,
            sealed=sealed, reached=PersistedReaderReached)
    return prepare


@pytest.mark.parametrize("version", [2, 3])
def test_reviewed_parent_persisted_reader_receives_explicit_native_version(
        reviewed_parent_persisted_reader_version_model, version):
    s = reviewed_parent_persisted_reader_version_model(version)
    loader = (owners.load_parent_verified_reviewed_preservation_v1 if version == 2
              else owners.load_parent_verified_reviewed_preservation_v3)
    with pytest.raises(s.reached, match="strict persisted reader reached with explicit domain"):
        loader(s.handle, s.packet, object(), {}, {}, **s.args)
    assert s.calls == [("retained_replay", version), ("current_validation", version), ("persisted_reader", version)]
    assert s.persisted_versions == [version] and s.native_resolutions and s.sealed == []


@pytest.mark.parametrize("reader_version,packet_version", [(2, 3), (3, 2)])
def test_reviewed_parent_persisted_reader_crossed_domain_refuses_before_native_resolution(
        reviewed_parent_persisted_reader_version_model, reader_version, packet_version):
    s = reviewed_parent_persisted_reader_version_model(packet_version)
    loader = (owners.load_parent_verified_reviewed_preservation_v1 if reader_version == 2
              else owners.load_parent_verified_reviewed_preservation_v3)
    with pytest.raises(ValueError, match="capability belongs to another reviewed native domain"):
        loader(s.handle, s.packet, object(), {}, {}, **s.args)
    assert s.native_resolutions == s.calls == s.persisted_versions == s.sealed == []


@pytest.fixture(scope="module")
def reviewed_parent_v3_complete_packet_scope(tmp_path_factory):
    """Real reviewed V3 project; modeled compiled bytes and no Native authority."""
    directory = tmp_path_factory.mktemp("reviewed-parent-v3-pure")
    base = reviewed_parent_packet_join_scope.__wrapped__(directory)
    prepared = prepare_reviewed_parent_native_fixture_inputs_v3(directory / "public-v3-inputs.json")
    data = owners.json_object(owners.checked_descriptor(prepared["native_inputs"]))
    manifest = json.loads(data["preparation"]["manifest_json"])
    assert owners.digest(data["preparation"]) == data["expected_preservation_sha256"]
    assert [name for name, _ in data["preparation"]["files"]] == sorted(manifest["file_sha256"])
    attempt = directory / "modeled-v3-attempt"; attempt.mkdir()
    operation = copy.deepcopy(base.operation)
    operation.update(prepared)
    operation["operation_id"] += "-v3"
    operation["nodeid"] = str(Path(__file__).absolute()) + "::test_reviewed_parent_v3_complete_project_worker_declaration_and_pure_packet_join"
    bodies = {"build-receipt.json": b"Modeled receipt only; no original success.\n",
        "stdout.txt": b"", "stderr.txt": b"", "runner-error.txt": b"",
        "project/corpus-manifest.json": data["preparation"]["manifest_json"].encode("utf-8")}
    for name, body in data["preparation"]["files"]:
        bodies["project/" + name] = bodies["executed-inputs/" + name] = body.encode("utf-8")
    for module in manifest["expected_modules"]:
        bodies["compiled/" + module.replace(".", "/") + ".olean"] = (
            "Modeled compiled bytes only: " + module).encode("utf-8")
    assert sorted(bodies) == prepared["expected_packet_paths"]
    inventory, durable, packet = [], [], []
    for name in sorted(bodies):
        raw = bodies[name]
        ref = put(attempt / operation["packet_root"] / name, raw)
        info = Path(ref["path"]).stat()
        packet.append((name, raw))
        inventory.append({"path": name, "bytes": ref["bytes"], "sha256": ref["sha256"]})
        durable.append({"path": operation["packet_root"] + "/" + name,
            "bytes": ref["bytes"], "sha256": ref["sha256"], "device": info.st_dev, "inode": info.st_ino})
    binding = dict(copy.deepcopy(base.binding),
        schema="reviewed-preservation-native-binding/v3",
        operation_id=operation["operation_id"],
        preparation_sha256=operation["expected_preservation_sha256"],
        graph_sha256=operation["expected_graph_sha256"], admission_sha256=operation["expected_admission_sha256"],
        toolchain=operation["toolchain"], packet_inventory=inventory,
        expected_packet_paths=list(prepared["expected_packet_paths"]),
        packet_sha256=owners.digest(inventory),
        receipt_sha256=next(row["sha256"] for row in inventory if row["path"] == "build-receipt.json"))
    packet_scope = SimpleNamespace(attempt=attempt, operation=operation, binding=binding,
        durable=durable, packet=tuple(packet), source=base.source)
    declared = reviewed_parent_worker_declaration_scope.__wrapped__(packet_scope)
    declared.policy["schema"] = "reviewed-preservation-parent/v3"
    declared.plan["schema"] = "reviewed-preservation-native-worker/v3"
    return SimpleNamespace(packet=packet_scope, declaration=declared, data=data, metadata=prepared)

def test_reviewed_parent_v3_complete_project_worker_declaration_and_pure_packet_join(
        reviewed_parent_v3_complete_packet_scope):
    s = reviewed_parent_v3_complete_packet_scope
    before = copy.deepcopy((s.declaration.plan, s.declaration.policy))
    # This call is the focused missing-route RED after complete genuine V3
    # preparation and the full modeled durable packet have been constructed.
    operation = s.declaration.worker.reviewed_preservation_native_operation_v3(
        s.declaration.plan, s.declaration.policy)
    assert operation == s.declaration.policy["operation"]
    assert operation is not s.declaration.plan["operation"]
    assert (s.declaration.plan, s.declaration.policy) == before
    assert owners._reviewed_parent_packet_paths_v3(s.data) == s.metadata["expected_packet_paths"]
    result = owners._reviewed_parent_packet_binding_v3(
        s.packet.operation, s.packet.binding, s.packet.durable, attempt_directory=s.packet.attempt)
    expected = tuple((row["path"], row["bytes"], row["sha256"], row["device"], row["inode"])
        for row in sorted(s.packet.durable, key=lambda row: row["path"]))
    assert result == expected and len(result) == len(s.metadata["expected_packet_paths"])
    assert all(type(row) is tuple and len(row) == 5 for row in result)

@pytest.mark.parametrize("initial_mode, expected_chmods", [(0o600, []), (0o640, [0o600])])
def test_normal_api_evidence_seed_is_idempotent_and_corrects_permissions(
        tmp_path, monkeypatch, pytestconfig, initial_mode, expected_chmods):
    import stat
    expected = str(Path(__file__).with_name("conftest.py"))
    plugins = [plugin for plugin in pytestconfig.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None) == expected]
    assert len(plugins) == 1
    hook = plugins[0]
    assert hook._seed_r45_receipt_evidence.__globals__ is vars(hook)
    assert hook._seed_r45_receipt_evidence.__code__.co_filename == expected
    monkeypatch.setattr(hook, "repo_root", tmp_path)
    dest = tmp_path / "data/aseh/evidence/bootstrap" / hook._R45_EVIDENCE_NAME
    dest.parent.mkdir(parents=True)
    raw = b'{"fixture_only":true,"scope":"private hook permission regression"}\n'
    dest.write_bytes(raw)
    dest.chmod(initial_mode)
    before = dest.stat()
    chmods = []
    original_chmod = Path.chmod
    def record_chmod(path, mode, *args, **kwargs):
        if path == dest:
            chmods.append(mode)
        return original_chmod(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "chmod", record_chmod)
    hook._seed_r45_receipt_evidence()
    hook._seed_r45_receipt_evidence()
    after = dest.stat()
    assert chmods == expected_chmods
    assert stat.S_IMODE(after.st_mode) == 0o600
    assert dest.read_bytes() == raw
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if initial_mode == 0o600:
        assert after.st_ctime_ns == before.st_ctime_ns

def test_reviewed_parent_v3_worker_operation_is_detached(reviewed_parent_v3_complete_packet_scope):
    s = reviewed_parent_v3_complete_packet_scope.declaration
    before = copy.deepcopy((s.plan, s.policy))
    observed = s.worker.reviewed_preservation_native_operation_v3(s.plan, s.policy)
    observed["expected_packet_paths"].clear()
    observed["native_argv"].clear()
    assert (s.plan, s.policy) == before

@pytest.mark.parametrize("fault", ["plan_v1", "policy_v1", "unsupported_plan", "missing_population",
    "duplicate_population", "unsorted_population", "escaping_population", "overbound_population",
    "source_pin", "multiple_nodes", "changed_command", "extra_command",
    "cpu_budget", "memory_budget", "storage_budget", "child_budget", "time_budget"])
def test_reviewed_parent_v3_worker_rejects_downgrade_population_and_envelope_changes(
        reviewed_parent_v3_complete_packet_scope, fault):
    s = reviewed_parent_v3_complete_packet_scope.declaration
    plan, policy = copy.deepcopy(s.plan), copy.deepcopy(s.policy)
    op = policy["operation"]
    if fault == "plan_v1": plan["schema"] = "reviewed-preservation-native-worker/v1"
    elif fault == "policy_v1": policy["schema"] = "reviewed-preservation-parent/v1"
    elif fault == "unsupported_plan": plan["schema"] = "reviewed-preservation-native-worker/v4"
    elif fault == "missing_population": op.pop("expected_packet_paths")
    elif fault == "duplicate_population": op["expected_packet_paths"].append(op["expected_packet_paths"][-1])
    elif fault == "unsorted_population": op["expected_packet_paths"].reverse()
    elif fault == "escaping_population": op["expected_packet_paths"] = ["../foreign"]
    elif fault == "overbound_population": op["expected_packet_paths"] = [f"file-{index:03d}" for index in range(173)]
    elif fault == "source_pin": op["expected_source_sha256"][next(iter(op["expected_source_sha256"]))] = "e" * 64
    elif fault == "multiple_nodes": plan["selected_checks"].append(op["nodeid"] + "-other")
    elif fault == "changed_command": op["native_argv"][-1][-1] = "another-target"
    elif fault == "extra_command": op["native_argv"].append(["/bin/true"])
    else:
        key = {"cpu_budget": "cpu_slots", "memory_budget": "memory_mb", "storage_budget": "storage_bytes",
            "child_budget": "child_process_slots", "time_budget": "timeout_seconds"}[fault]
        policy["validation"][key] += 1
    plan["operation"] = copy.deepcopy(op)
    with pytest.raises(ValueError):
        s.worker.reviewed_preservation_native_operation_v3(plan, policy)

@pytest.mark.parametrize("fault", ["body_without_new_pin", "manifest_without_new_pin", "v2_schema",
    "foreign_profile", "two_units", "nine_units", "missing_file", "duplicate_file", "file_hash",
    "escaping_file", "duplicate_module", "escaping_module", "overbound_modules"])
def test_reviewed_parent_v3_opening_population_refuses_changed_or_incomplete_preparation(
        reviewed_parent_v3_complete_packet_scope, fault):
    data = copy.deepcopy(reviewed_parent_v3_complete_packet_scope.data)
    preparation = data["preparation"]
    manifest = json.loads(preparation["manifest_json"])
    if fault == "body_without_new_pin": preparation["files"][0][1] += "\n-- changed source\n"
    elif fault == "manifest_without_new_pin": preparation["manifest_json"] += " "
    elif fault == "v2_schema": manifest["schema"] = "legal-corpus-reviewed-preservation-preparation/v2"
    elif fault == "foreign_profile": manifest["profile"] = "reviewed-multi-unit-relational-preservation/v2"
    elif fault == "two_units": manifest["units"] = manifest["units"][:2]
    elif fault == "nine_units": manifest["units"] = [copy.deepcopy(manifest["units"][0]) for _ in range(9)]
    elif fault == "missing_file": preparation["files"].pop()
    elif fault == "duplicate_file": preparation["files"].append(copy.deepcopy(preparation["files"][-1]))
    elif fault == "file_hash": manifest["file_sha256"][preparation["files"][0][0]] = "e" * 64
    elif fault == "escaping_file":
        old = preparation["files"][0][0]
        preparation["files"][0][0] = "../foreign.lean"
        manifest["file_sha256"]["../foreign.lean"] = manifest["file_sha256"].pop(old)
    elif fault == "duplicate_module": manifest["expected_modules"].append(manifest["expected_modules"][-1])
    elif fault == "escaping_module": manifest["expected_modules"][0] = "../Foreign"
    elif fault == "overbound_modules": manifest["expected_modules"] = [f"Module{index}" for index in range(12)]
    if fault not in ("body_without_new_pin", "manifest_without_new_pin"):
        preparation["manifest_json"] = owners.canonical(manifest).decode("utf-8")
        data["expected_preservation_sha256"] = owners.digest(preparation)
    with pytest.raises(ValueError):
        owners._reviewed_parent_packet_paths_v3(data)

@pytest.mark.parametrize("fault", ["v1_binding", "missing_population", "crossed_population", "omitted_packet",
    "duplicate_packet", "missing_durable", "extra_durable", "changed_durable_bytes", "boolean_inode",
    "source_pin", "graph_pin", "admission_pin", "preparation_pin", "toolchain", "packet_pin",
    "resealed_missing_receipt", "resealed_missing_log"])
def test_reviewed_parent_v3_packet_join_refuses_crossed_and_resealed_incomplete_population(
        reviewed_parent_v3_complete_packet_scope, fault):
    s = reviewed_parent_v3_complete_packet_scope.packet
    operation, binding, durable = copy.deepcopy(s.operation), copy.deepcopy(s.binding), copy.deepcopy(s.durable)
    if fault == "v1_binding": binding["schema"] = "reviewed-preservation-native-binding/v1"
    elif fault == "missing_population": binding.pop("expected_packet_paths")
    elif fault == "crossed_population": binding["expected_packet_paths"] = binding["expected_packet_paths"][:-1]
    elif fault == "omitted_packet":
        binding["packet_inventory"].pop()
        binding["packet_sha256"] = owners.digest(binding["packet_inventory"])
    elif fault == "duplicate_packet":
        binding["packet_inventory"].append(copy.deepcopy(binding["packet_inventory"][-1]))
        binding["packet_sha256"] = owners.digest(binding["packet_inventory"])
    elif fault == "missing_durable": durable.pop()
    elif fault == "extra_durable": durable.append(dict(durable[-1], path=operation["packet_root"] + "/foreign.bin"))
    elif fault == "changed_durable_bytes": durable[0]["bytes"] += 1
    elif fault == "boolean_inode": durable[0]["inode"] = True
    elif fault == "source_pin": binding["producer_pins"][s.source["path"]] = "e" * 64
    elif fault in ("graph_pin", "admission_pin", "preparation_pin"):
        binding[fault.removesuffix("_pin") + "_sha256"] = "e" * 64
    elif fault == "toolchain": binding["toolchain"] = "leanprover/lean4:v4.35.0"
    elif fault == "packet_pin": binding["packet_sha256"] = "e" * 64
    else:
        omitted = "build-receipt.json" if fault == "resealed_missing_receipt" else "stdout.txt"
        operation["expected_packet_paths"].remove(omitted)
        binding["expected_packet_paths"].remove(omitted)
        binding["packet_inventory"] = [row for row in binding["packet_inventory"] if row["path"] != omitted]
        binding["packet_sha256"] = owners.digest(binding["packet_inventory"])
        durable = [row for row in durable if row["path"] != operation["packet_root"] + "/" + omitted]
    match = "top-level receipt and log" if fault.startswith("resealed_missing_") else None
    with pytest.raises(ValueError, match=match):
        owners._reviewed_parent_packet_binding_v3(operation, binding, durable, attempt_directory=s.attempt)

def test_reviewed_parent_v3_worker_packet_collection_matches_complete_opening_roster(
        reviewed_parent_v3_complete_packet_scope):
    s = reviewed_parent_v3_complete_packet_scope
    # Exercise the collector's real private-file walker without constructing its
    # live Source/Native observer or manufacturing an original-parent handle.
    collector = SimpleNamespace(packet_root=s.packet.attempt / s.packet.operation["packet_root"],
        reviewed_version=3, operation=s.packet.operation)
    observed = s.declaration.worker.ReviewedPreservationNativeCollectorV1._packet(collector)
    assert observed == s.packet.packet
    assert [name for name, _ in observed] == s.metadata["expected_packet_paths"]

@pytest.mark.parametrize("fault", ["omitted_opening_path", "extra_opening_path"])
def test_reviewed_parent_v3_worker_packet_collection_refuses_incomplete_roster(
        reviewed_parent_v3_complete_packet_scope, fault):
    s = reviewed_parent_v3_complete_packet_scope
    operation = copy.deepcopy(s.packet.operation)
    if fault == "omitted_opening_path": operation["expected_packet_paths"].pop()
    else: operation["expected_packet_paths"] = sorted(operation["expected_packet_paths"] + ["foreign.bin"])
    collector = SimpleNamespace(packet_root=s.packet.attempt / s.packet.operation["packet_root"],
        reviewed_version=3, operation=operation)
    with pytest.raises(ValueError):
        s.declaration.worker.ReviewedPreservationNativeCollectorV1._packet(collector)

@pytest.mark.parametrize("offered", ["none", "object", "dict", "unissued"])
def test_reviewed_parent_v3_success_rejects_unissued_before_native_owners(monkeypatch, offered):
    if offered == "none": value = None
    elif offered == "object": value = object()
    elif offered == "dict": value = {"original_success_authenticated": True, "reviewed_version": 3}
    else: value = object.__new__(owners.VerifiedReviewedPreservationParentSuccess)
    def forbidden(*args, **kwargs):
        pytest.fail("native owner resolution preceded invalid V3 parent handle refusal")
    monkeypatch.setattr(owners, "_parent_native_owner", forbidden)
    with pytest.raises(ValueError, match="(?i)(issued|handle)"):
        owners.load_parent_verified_reviewed_preservation_v3(
            value, (), None, {}, {},
            expected_receipt_sha256="a" * 64, expected_packet_sha256="b" * 64,
            expected_binary_sha256={}, expected_preservation_sha256="c" * 64,
            expected_graph_sha256="d" * 64, expected_admission_sha256="e" * 64,
            toolchain="leanprover/lean4:v4.34.1")

@pytest.mark.parametrize("target", ["_reviewed_parent_packet_paths_v3", "_reviewed_parent_packet_binding_v3",
    "_load_parent_verified_reviewed_preservation", "load_parent_verified_reviewed_preservation_v3",
    "_checked_reviewed_preservation_parent_policy", "run_reviewed_preservation_parent_v3"])
@pytest.mark.parametrize("mutation", ["identity", "code", "defaults"])
def test_reviewed_parent_v3_new_helpers_remain_in_original_callable_snapshot(monkeypatch, target, mutation):
    function = getattr(owners, target)
    if mutation == "identity": monkeypatch.setattr(owners, target, reviewed_parent_forbidden_callable_v1)
    elif mutation == "code": monkeypatch.setattr(function, "__code__", reviewed_parent_forbidden_callable_v1.__code__)
    else: monkeypatch.setattr(function, "__defaults__", (object(),))
    with pytest.raises(ValueError, match="(?i)(changed|binding|original|canonical)"):
        owners._reviewed_parent_guard()

def _legacy_native_guard_source_evidence_v1(monkeypatch):
    import importlib
    datasets_root = Path(__file__).resolve().parents[3] / "ipfs_datasets"
    monkeypatch.syspath_prepend(str(datasets_root))
    # Fixed approved R46 path population. Hash current Source directly before
    # any mutation; a Native callback never discovers or supplies this evidence.
    # The existing API requires the Native leaf pin and uses its genuine guard
    # for canonical imported-Source provenance. This is not a claim that every
    # transitive import is independently parent-custodied.
    relatives = (
        'ipfs_datasets_py/logic/autoformal/corpus_norm_definition_adapter.py',
        'ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py',
        'ipfs_datasets_py/logic/autoformal/legal_corpus_shared_lake.py',
        'ipfs_datasets_py/logic/autoformal/legal_release_coordinator_v1.py',
        'scripts/ops/legal_ir/validate_uscode_corpus_release.py',
        'tests/unit/logic/autoformal/test_corpus_release_cli.py',
        'tests/unit/logic/autoformal/test_corpus_norm_definition_adapter.py',
        'ipfs_datasets_py/logic/autoformal/legal_native_ast_preservation.py',
        'ipfs_datasets_py/logic/autoformal/legal_native_ast_preservation_native.py',
        'ipfs_datasets_py/logic/autoformal/corpus_graph.py',
        'ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_lake_v2.py',
        'ipfs_datasets_py/logic/autoformal/legal_corpus_lake.py',
        'ipfs_datasets_py/logic/autoformal/legal_norm_definition_evidence_v1.py',
        'ipfs_datasets_py/logic/autoformal/legal_pilot_lake.py',
        'ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_preservation_packet_v2.py',
        'ipfs_datasets_py/logic/autoformal/legal_reviewed_norm_admission_v3.py',
        'tests/unit/logic/autoformal/test_legal_reviewed_norm_admission_v3.py',
        'tests/unit/logic/autoformal/test_legal_corpus_definition_use_lake_v2.py',
        'tests/unit/logic/autoformal/test_legal_corpus_shared_lake.py',
        'tests/unit/logic/autoformal/test_legal_corpus_definition_use_native_v2.py',
        'pytest.ini',
        'conftest.py',
        'tests/conftest.py',
        'tests/unit/conftest.py',
        'tests/unit/logic/autoformal/test_corpus_definition_use_review_v2.py',
        'tests/unit/logic/autoformal/test_corpus_definition_use_v2.py',
        'tests/unit/logic/autoformal/test_corpus_definition_review_v2.py',
        'tests/unit/logic/autoformal/test_corpus_definition_adapter_v2.py',
        'ipfs_datasets_py/logic/backends/process.py',
        'ipfs_datasets_py/logic/autoformal/corpus_source_bindings.py',
        'ipfs_datasets_py/logic/autoformal/legal_norm_content_adjudication_v1.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_span_consumer_binding.py',
        'tests/unit/logic/autoformal/test_corpus_source_bindings.py',
        'tests/unit/logic/autoformal/test_legal_norm_content_adjudication_v1.py',
        'ipfs_datasets_py/logic/autoformal/legal_semantic_graph.py',
        'ipfs_datasets_py/logic/autoformal/legal_graph_review.py',
        'ipfs_datasets_py/logic/autoformal/corpus_contracts.py',
        'ipfs_datasets_py/logic/formalization/autoencoder/native_family_lean_emitters.py',
        'ipfs_datasets_py/logic/formalization/autoencoder/native_legal_qualified_lean.py',
        'ipfs_datasets_py/logic/formalization/autoencoder/native_qualified_lean.py',
        'ipfs_datasets_py/optimizers/logic_theorem_optimizer/uscode_dataset.py',
        'ipfs_datasets_py/logic/autoformal/uscode_ingest.py',
        'ipfs_datasets_py/logic/autoformal/legal_entity_index_intake.py',
        'ipfs_datasets_py/logic/autoformal/constitution_inventory.py',
        'ipfs_datasets_py/logic/autoformal/meta_ontology.py',
        'ipfs_datasets_py/logic/autoformal/legal_parent_text_views.py',
    )
    source_pins = {
        str(datasets_root / relative): hashlib.sha256((datasets_root / relative).read_bytes()).hexdigest()
        for relative in relatives
    }
    native = importlib.import_module(
        "ipfs_datasets_py.logic.autoformal.legal_corpus_definition_use_native_v2")
    assert str(Path(native.__file__).resolve()) in source_pins
    return native, {"datasets_root": str(datasets_root), "source_pins": source_pins}

@pytest.mark.parametrize("reviewed_version", (2, 3))
def test_parent_native_owner_existing_api_accepts_genuine_source(monkeypatch, reviewed_version):
    native, evidence = _legacy_native_guard_source_evidence_v1(monkeypatch)
    assert owners._parent_native_owner(evidence, reviewed_version=reviewed_version) is native

@pytest.mark.parametrize("reviewed_version", (2, 3))
@pytest.mark.parametrize("snapshot_kind", ("empty", "resealed"))
def test_parent_native_owner_refuses_empty_or_resealed_snapshots_before_callbacks(
        monkeypatch, reviewed_version, snapshot_kind):
    native, evidence = _legacy_native_guard_source_evidence_v1(monkeypatch)
    # The SAME complete existing-API evidence must pass the genuine route
    # before mutation. Missing Source pins or unrelated genuine-guard failures
    # cannot make this negative test pass.
    assert owners._parent_native_owner(evidence, reviewed_version=reviewed_version) is native
    callbacks = []
    original_states = native._LIVE_NATIVE_FUNCTIONS

    def forged_matcher(*args, **kwargs):
        callbacks.append("matcher")
        return True

    def forged_guard(*args, **kwargs):
        callbacks.append("guard")
        return native._live_callable_matches(None, None)

    with monkeypatch.context() as change:
        change.setattr(native, "_live_callable_matches", forged_matcher)
        change.setattr(native, "_live_success_guard", forged_guard)
        if snapshot_kind == "empty":
            replacement_states = ()
        else:
            # Reseal current mutable metadata to the substituted callbacks.
            # No product predicate is called to manufacture this snapshot.
            replacement_states = tuple(
                (name, (function, function.__code__, function.__defaults__,
                        tuple(sorted(({} if function.__kwdefaults__ is None
                                      else function.__kwdefaults__).items()))))
                for name, _old_state in original_states
                for function in (vars(native)[name],)
            )
        change.setattr(native, "_LIVE_NATIVE_FUNCTIONS", replacement_states)
        refused = False
        try:
            owners._parent_native_owner(evidence, reviewed_version=reviewed_version)
        except ValueError:
            refused = True
        assert callbacks == [], ("mutable Native callback ran before independent authentication", callbacks)
        assert refused is True, "empty/resealed Native metadata was accepted"

@pytest.mark.parametrize("offered", ["none", "object", "dict", "json", "unissued"])
def test_live_physical_join_refuses_unissued_before_any_R_resolution(monkeypatch, offered):
    value = {"none": None, "object": object(), "dict": {"admitted": True},
             "json": json.loads('{"original_success_authenticated":true}'),
             "unissued": object.__new__(owners.VerifiedReviewedPreservationParentSuccess)}[offered]
    def forbidden(*args, **kwargs):
        pytest.fail("R resolution preceded genuine parent handle refusal")
    monkeypatch.setattr(owners, "_parent_release_owner_binding_v3", forbidden)
    with pytest.raises(ValueError, match="(?i)(issued|handle)"):
        owners.load_parent_verified_uscode_physical_closure_v3(
            value, b"request", expected_request_sha256=hashlib.sha256(b"request").hexdigest())

@pytest.mark.parametrize("crossed", ["process", "native_thread", "thread_object"])
def test_live_physical_join_capability_entry_refuses_crossed_custody_without_parsing_evidence(
        monkeypatch, crossed):
    import os
    import threading
    import weakref
    handle = object.__new__(owners.VerifiedReviewedPreservationParentSuccess)
    # Deliberately modeled private entry: this tests only refusal ordering and
    # never constitutes an actual original-parent receipt or successful handle.
    values = [os.getpid(), threading.get_native_id(),
              weakref.ref(threading.current_thread()), weakref.ref(handle), b"not evidence"]
    if crossed == "process": values[0] += 1
    elif crossed == "native_thread": values[1] += 1
    else: values[2] = lambda: None
    monkeypatch.setitem(owners._REVIEWED_PARENT_CAPABILITIES, handle, tuple(values))
    with pytest.raises(ValueError, match="(?i)(issued|handle|process|thread)"):
        owners._reviewed_parent_capability_entry(handle)

@pytest.mark.parametrize("target", [
    "load_parent_verified_uscode_physical_closure_v3",
    "_parent_release_owner_binding_v3", "_joined_contracts_primitive_binding_v1",
    "_capture_parent_verified_reviewed_preservation", "_reviewed_parent_joined_opening_v3"])
@pytest.mark.parametrize("mutation", ["identity", "code", "defaults"])
def test_live_physical_join_helpers_are_in_original_A_callable_guard(monkeypatch, target, mutation):
    function = getattr(owners, target)
    if mutation == "identity":
        monkeypatch.setattr(owners, target, reviewed_parent_forbidden_callable_v1)
    elif mutation == "code":
        monkeypatch.setattr(function, "__code__", reviewed_parent_forbidden_callable_v1.__code__)
    else:
        monkeypatch.setattr(function, "__defaults__", (object(),))
    with pytest.raises(ValueError, match="(?i)(changed|binding|original|canonical)"):
        owners._reviewed_parent_guard()

@pytest.mark.parametrize("kind", ["metaclass", "tuple", "list", "dict", "set"])
def test_joined_identity_refuses_unsupported_types_without_user_callbacks(kind):
    called = []
    def trap(*args, **kwargs):
        called.append(True)
        raise AssertionError("identity refusal invoked a user callback")
    class TrappingMeta(type):
        __eq__ = trap
    class TrappingValue(metaclass=TrappingMeta):
        pass
    base = {"tuple": tuple, "list": list, "dict": dict, "set": set}.get(kind)
    if base is None:
        value = TrappingValue()
    else:
        value = type("Trapping" + kind, (base,), {
            "__len__": trap, "__iter__": trap, "items": trap, "__eq__": trap})()
    with pytest.raises(ValueError, match="ordinary native joined identity"):
        owners._joined_literal_signature_v1(value)
    assert called == []

def _current_joined_owner_software_evidence():
    """Current Source pins only; no parent capability or Native success evidence."""
    root = Path(owners.__file__).resolve().parents[4] / "ipfs_datasets"
    paths = (
        "ipfs_datasets_py/logic/autoformal/corpus_norm_definition_adapter.py",
        "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py",
        "ipfs_datasets_py/logic/autoformal/legal_corpus_shared_lake.py",
        "ipfs_datasets_py/logic/autoformal/legal_release_coordinator_v1.py",
        "scripts/ops/legal_ir/validate_uscode_corpus_release.py",
        "tests/unit/logic/autoformal/test_corpus_release_cli.py",
        "tests/unit/logic/autoformal/test_corpus_norm_definition_adapter.py",
        "ipfs_datasets_py/logic/autoformal/legal_native_ast_preservation.py",
        "ipfs_datasets_py/logic/autoformal/legal_native_ast_preservation_native.py",
        "ipfs_datasets_py/logic/autoformal/corpus_graph.py",
        "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_lake_v2.py",
        "ipfs_datasets_py/logic/autoformal/legal_corpus_lake.py",
        "ipfs_datasets_py/logic/autoformal/legal_norm_definition_evidence_v1.py",
        "ipfs_datasets_py/logic/autoformal/legal_pilot_lake.py",
        "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_preservation_packet_v2.py",
        "ipfs_datasets_py/logic/autoformal/legal_reviewed_norm_admission_v3.py",
        "tests/unit/logic/autoformal/test_legal_reviewed_norm_admission_v3.py",
        "tests/unit/logic/autoformal/test_legal_corpus_definition_use_lake_v2.py",
        "tests/unit/logic/autoformal/test_legal_corpus_shared_lake.py",
        "tests/unit/logic/autoformal/test_legal_corpus_definition_use_native_v2.py",
        "pytest.ini",
        "conftest.py",
        "tests/conftest.py",
        "tests/unit/conftest.py",
        "tests/unit/logic/autoformal/test_corpus_definition_use_review_v2.py",
        "tests/unit/logic/autoformal/test_corpus_definition_use_v2.py",
        "tests/unit/logic/autoformal/test_corpus_definition_review_v2.py",
        "tests/unit/logic/autoformal/test_corpus_definition_adapter_v2.py",
        "ipfs_datasets_py/logic/backends/process.py",
        "ipfs_datasets_py/logic/autoformal/corpus_source_bindings.py",
        "ipfs_datasets_py/logic/autoformal/legal_norm_content_adjudication_v1.py",
        "ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_span_consumer_binding.py",
        "tests/unit/logic/autoformal/test_corpus_source_bindings.py",
        "tests/unit/logic/autoformal/test_legal_norm_content_adjudication_v1.py",
        "ipfs_datasets_py/logic/autoformal/legal_semantic_graph.py",
        "ipfs_datasets_py/logic/autoformal/legal_graph_review.py",
        "ipfs_datasets_py/logic/autoformal/corpus_contracts.py",
        "ipfs_datasets_py/logic/formalization/autoencoder/native_family_lean_emitters.py",
        "ipfs_datasets_py/logic/formalization/autoencoder/native_legal_qualified_lean.py",
        "ipfs_datasets_py/logic/formalization/autoencoder/native_qualified_lean.py",
        "ipfs_datasets_py/optimizers/logic_theorem_optimizer/uscode_dataset.py",
        "ipfs_datasets_py/logic/autoformal/uscode_ingest.py",
        "ipfs_datasets_py/logic/autoformal/legal_entity_index_intake.py",
        "ipfs_datasets_py/logic/autoformal/constitution_inventory.py",
        "ipfs_datasets_py/logic/autoformal/meta_ontology.py",
        "ipfs_datasets_py/logic/autoformal/legal_parent_text_views.py",
    )
    # Fixed approved Source roster, with current bytes read under the ordinary
    # Source epoch. Native callbacks never discover this pin population.
    pins = {str(root / relative): hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in paths}
    return {"datasets_root": str(root), "operation": {"expected_source_sha256": dict(pins)},
            "source_pins": pins}

def test_joined_owner_binding_replays_real_current_loaded_owners_without_minting():
    evidence = _current_joined_owner_software_evidence()
    module, first = owners._parent_release_owner_binding_v3(evidence)
    reopened, second = owners._parent_release_owner_binding_v3(evidence)
    assert reopened is module
    assert owners._same_joined_owner_binding_v1(first, second)
    assert type(first) is tuple and len(first) == 2
    assert not isinstance(first, owners.VerifiedReviewedPreservationParentSuccess)

@pytest.mark.parametrize("omitted", ["legal_release_coordinator_v1", "corpus_contracts",
                                    "legal_entity_index_intake"])
def test_joined_owner_binding_requires_every_predeclared_source_before_dispatch(omitted):
    evidence = _current_joined_owner_software_evidence()
    path = next(path for path in evidence["source_pins"] if Path(path).stem == omitted)
    evidence["source_pins"].pop(path)
    with pytest.raises(ValueError, match="independently pinned"):
        owners._parent_release_owner_binding_v3(evidence)

@pytest.mark.parametrize("version", [2, 3])
def test_unjoined_legacy_and_v3_policy_do_not_resolve_R(monkeypatch, tmp_path, version):
    def forbidden(*args, **kwargs):
        pytest.fail("unjoined policy resolved a new owner")
    monkeypatch.setattr(owners, "_parent_release_owner_binding_v3", forbidden)
    policy = {"schema": owners._reviewed_parent_domain(version)[0],
              "datasets_root": str(tmp_path), "source_descriptors": [],
              "operation": {"expected_source_sha256": {}}}
    assert owners._reviewed_parent_joined_opening_v3(policy) is None

@pytest.mark.parametrize("position", ["defaults", "kwdefaults"])
def test_joined_primitive_authentication_refuses_bad_defaults_without_equality_callbacks(
        monkeypatch, position):
    from ipfs_datasets_py.logic.autoformal import corpus_contracts as contracts
    called = []
    class TrappingMeta(type):
        def __eq__(self, other):
            called.append("metaclass equality")
            raise AssertionError("default authentication invoked equality")
    class TrappingValue(metaclass=TrappingMeta):
        pass
    if position == "defaults":
        target = contracts._loaded_owner_shape_v1
        monkeypatch.setattr(target, "__defaults__", (TrappingValue(),))
    else:
        target = contracts._capture_loaded_owner_guard_v1
        changed = dict(target.__kwdefaults__)
        changed["value_names"] = TrappingValue()
        monkeypatch.setattr(target, "__kwdefaults__", changed)
    path = Path(contracts.__file__)
    with pytest.raises(ValueError, match="ordinary native joined identity"):
        owners._joined_contracts_primitive_binding_v1(contracts, path.read_bytes(), path)
    assert called == []

@pytest.mark.parametrize("has_primary", [False, True])
def test_joined_closing_checks_do_not_render_hostile_errors_or_skip_fences(has_primary):
    called = []
    class HostileError(ValueError):
        def __str__(self):
            raise AssertionError("closing custody rendered arbitrary exception")
    first = HostileError()
    second = HostileError()
    primary = HostileError() if has_primary else None
    def check(index):
        called.append(index)
        if index == 0: raise first
        if index == 1: raise second
    with pytest.raises(HostileError) as caught:
        owners._joined_complete_checks_v1(tuple(
            (str(index), lambda index=index: check(index)) for index in range(3)), primary=primary)
    assert called == [0, 1, 2] and caught.value is first
    if primary is not None:
        assert caught.value.__cause__ is primary

@pytest.fixture
def live_physical_join_software_route(monkeypatch):
    """Explicit routing doubles; no genuine parent or Native receipt is issued."""
    calls = []
    raw = b"software route request only"
    request_sha = hashlib.sha256(raw).hexdigest()
    binding = (("software-only binding",), ())
    evidence = {"reviewed_version": 3, "binding": {
        "graph_sha256": "a" * 64, "admission_sha256": "b" * 64,
        "preparation_sha256": "c" * 64, "packet_sha256": "d" * 64,
        "receipt_sha256": hashlib.sha256(b"not a Native receipt").hexdigest()}}
    handle = object()
    entry = (None, None, None, None, owners.canonical(evidence), binding)
    parent = {name: str(index) * 64 for index, name in enumerate(
        ("parent_request_sha256", "parent_closeout_sha256", "parent_durability_sha256"), 1)}
    context = {"reviewed_version": 3, "physical_request_sha256": request_sha,
        "native": {"packet_files": (("build-receipt.json", b"not a Native receipt"),),
            "receipt_raw": b"not a Native receipt", "expected_packet_sha256": "d" * 64,
            "expected_binary_sha256": {}},
        "preparation": object(), "graph": {}, "admission": {}, "args": (),
        "kwargs": {"current_source": "unchanged"},
        "pins": {"graph": "a" * 64, "admission": "b" * 64, "preservation": "c" * 64},
        "toolchain": "leanprover/lean4:v4.34.1"}
    false_fields = ("release_ready", "closure_complete", "admitted", "formalized",
        "proof_verified", "source_semantics_verified", "source_authenticity_verified",
        "source_metadata_arrow_bytes_verified", "original_native_parent_custody_verified",
        "remote_readback_verified", "model_or_compiler_execution_performed")
    physical = dict.fromkeys(false_fields, False)
    physical.update(schema="software-schema/v3", profile="software-profile/v3",
        request_sha256=request_sha, graph_sha256="a" * 64, admission_sha256="b" * 64,
        preservation_sha256="c" * 64, local_replay_complete=True,
        local_source_cache_entity_bindings_verified=True,
        independent_reviewed_crosswalk_verified=True, native_receipt_replayed=True,
        blockers=["original_admitted_native_parent_custody_not_established"])
    scope = SimpleNamespace(calls=calls, raw=raw, request_sha=request_sha, handle=handle,
        entry=entry, evidence=evidence, binding=binding, context=context, physical=physical,
        after_observer=None, replay_error=None, fence_error=None, current_binding=binding, captured=[])
    def guard():
        calls.append("A.guard")
    monkeypatch.setattr(owners, "_reviewed_parent_guard", guard)
    monkeypatch.setattr(owners, "_REVIEWED_PARENT_GUARD_OBJECT", guard)
    monkeypatch.setattr(owners, "_REVIEWED_PARENT_GUARD_CODE", guard.__code__)
    monkeypatch.setattr(owners, "_reviewed_parent_capability_entry", lambda value: scope.entry)
    def snapshot(value):
        return (value["pins"]["graph"], value["kwargs"]["current_source"])
    def postclose(value):
        calls.append("R.postclose")
    def replay(offered, *, expected_request_sha256, reviewed_version, _observe_checked_context,
            _native_transport):
        assert offered is raw and expected_request_sha256 == request_sha and reviewed_version == 3
        assert _native_transport == "direct-packet/v1"
        calls.append("R.open")
        try:
            _observe_checked_context(context)
            if scope.after_observer is not None:
                scope.after_observer()
            if scope.replay_error is not None:
                raise scope.replay_error
            return physical
        finally:
            calls.append("R.final")
    release = SimpleNamespace(_validate_uscode_release_physical_closure=replay,
        _physical_replay_context_snapshot=snapshot, _physical_postclose_fence_v1=postclose,
        CLOSURE_SCHEMA_V3=physical["schema"], CLOSURE_PROFILE_V3_DIRECT=physical["profile"],
        _canonical=owners.canonical)
    def resolve(offered):
        assert offered == json.loads(owners.canonical(evidence))
        calls.append("A.owners")
        return release, scope.current_binding
    monkeypatch.setattr(owners, "_parent_release_owner_binding_v3", resolve)
    def capture(offered, packet, preparation, graph, admission, *args, **kwargs):
        calls.append("A.capture")
        assert offered is handle and packet is context["native"]["packet_files"]
        assert preparation is context["preparation"] and graph is context["graph"]
        assert admission is context["admission"] and args == context["args"]
        scope.captured.append(kwargs)
        def fence():
            calls.append("A.current")
            if scope.fence_error is not None:
                raise scope.fence_error
        return parent, fence
    monkeypatch.setattr(owners, "_capture_parent_verified_reviewed_preservation", capture)
    def seal(value):
        calls.append("A.seal")
        return value
    native = SimpleNamespace(_owners=lambda: (None, SimpleNamespace(
        contracts=SimpleNamespace(seal_record=seal))), AUTHORITY={"software_double": False})
    monkeypatch.setattr(owners, "_parent_native_owner", lambda *args, **kwargs: native)
    return scope

def test_live_physical_join_fixed_route_is_once_finalized_and_retains_false_authority(
        live_physical_join_software_route):
    s = live_physical_join_software_route
    result = owners.load_parent_verified_uscode_physical_closure_v3(
        s.handle, s.raw, expected_request_sha256=s.request_sha)
    assert s.calls.count("R.open") == s.calls.count("R.final") == s.calls.count("A.capture") == 1
    assert s.calls.index("R.final") < s.calls.index("A.seal")
    assert s.calls[-1] == "A.current"
    assert s.captured[0]["expected_graph_sha256"] == s.context["pins"]["graph"]
    assert s.captured[0]["expected_admission_sha256"] == s.context["pins"]["admission"]
    assert s.captured[0]["expected_preservation_sha256"] == s.context["pins"]["preservation"]
    assert result["live_original_parent_and_current_physical_replay_joined"] is True
    assert result["wire_report_original_parent_flag_unchanged"] is True
    assert all(result[name] is False for name in (
        "release_ready", "closure_complete", "admitted", "formalized", "proof_verified",
        "source_semantics_verified", "source_authenticity_verified", "remote_readback_verified"))

@pytest.mark.parametrize("fault", ["wire_authority", "wire_parent_flag", "wire_target",
    "changed_loaded_binding", "changed_current_context", "missing_native"])
def test_live_physical_join_routing_doubles_refuse_crossed_or_authoritative_results(
        live_physical_join_software_route, fault):
    s = live_physical_join_software_route
    if fault == "wire_authority": s.physical["admitted"] = True
    elif fault == "wire_parent_flag": s.physical["original_native_parent_custody_verified"] = True
    elif fault == "wire_target": s.physical["graph_sha256"] = "e" * 64
    elif fault == "changed_loaded_binding":
        s.after_observer = lambda: setattr(s, "current_binding", (("changed binding",), ()))
    elif fault == "changed_current_context":
        s.after_observer = lambda: s.context["kwargs"].update(current_source="changed")
    else: s.context["native"] = None
    with pytest.raises(ValueError):
        owners.load_parent_verified_uscode_physical_closure_v3(
            s.handle, s.raw, expected_request_sha256=s.request_sha)
    assert s.calls.count("R.final") == 1 and "A.seal" not in s.calls
    if fault != "missing_native":
        assert "A.current" in s.calls

@pytest.mark.parametrize("fault", ["version", "unbound_entry", "request_pin"])
def test_live_physical_join_refuses_before_traversal_when_opening_binding_is_wrong(
        live_physical_join_software_route, fault):
    s = live_physical_join_software_route
    pin = s.request_sha
    if fault == "version":
        changed = dict(s.evidence, reviewed_version=2)
        s.entry = (*s.entry[:4], owners.canonical(changed), s.binding)
    elif fault == "unbound_entry": s.entry = s.entry[:5]
    else: pin = "0" * 64
    with pytest.raises(ValueError):
        owners.load_parent_verified_uscode_physical_closure_v3(
            s.handle, s.raw, expected_request_sha256=pin)
    assert "R.open" not in s.calls and "A.capture" not in s.calls and "A.seal" not in s.calls

def test_live_physical_join_keeps_unchanged_R_exception_and_runs_all_final_fences(
        live_physical_join_software_route):
    s = live_physical_join_software_route
    sentinel = RuntimeError("unchanged finalized R failure")
    s.replay_error = sentinel
    with pytest.raises(RuntimeError) as caught:
        owners.load_parent_verified_uscode_physical_closure_v3(
            s.handle, s.raw, expected_request_sha256=s.request_sha)
    assert caught.value is sentinel
    assert s.calls.count("R.final") == 1 and "A.seal" not in s.calls
    assert "R.postclose" in s.calls and s.calls[-1] == "A.current"

@pytest.mark.parametrize("kind", ["object_key", "str_subclass"])
def test_joined_primitive_refuses_nonexact_namespace_keys_before_string_methods(monkeypatch, kind):
    from ipfs_datasets_py.logic.autoformal import corpus_contracts as contracts
    called = []
    def trap(*args, **kwargs):
        called.append(True)
        raise AssertionError("namespace refusal invoked key callbacks")
    if kind == "object_key":
        class Key:
            startswith = trap
            __hash__ = object.__hash__
        key = Key()
    else:
        class Key(str):
            startswith = trap
        key = Key("private-nonexact-key")
    monkeypatch.setitem(vars(contracts), key, None)
    path = Path(contracts.__file__)
    with pytest.raises(ValueError, match="namespace|binding|keys"):
        owners._joined_contracts_primitive_binding_v1(contracts, path.read_bytes(), path)
    assert called == []

@pytest.mark.parametrize("field", ["module_name", "module_file", "spec_name", "spec_origin", "spec_dict"])
def test_joined_owner_metadata_refusal_does_not_invoke_caller_equality_or_mapping(
        monkeypatch, field):
    from ipfs_datasets_py.logic.autoformal import legal_release_coordinator_v1 as release
    evidence = _current_joined_owner_software_evidence()
    called = []
    class Trap:
        def __eq__(self, other):
            called.append("equality")
            raise AssertionError("metadata refusal invoked equality")
    class TrapDict(dict):
        def get(self, *args, **kwargs):
            called.append("get")
            raise AssertionError("metadata refusal invoked mapping")
    if field == "module_name": monkeypatch.setattr(release, "__name__", Trap())
    elif field == "module_file": monkeypatch.setattr(release, "__file__", Trap())
    elif field == "spec_name": monkeypatch.setattr(release.__spec__, "name", Trap())
    elif field == "spec_origin": monkeypatch.setattr(release.__spec__, "origin", Trap())
    else: monkeypatch.setattr(release.__spec__, "__dict__", TrapDict(vars(release.__spec__)))
    with pytest.raises(ValueError, match="canonical joined-owner|ordinary joined-owner"):
        owners._parent_release_owner_binding_v3(evidence)
    assert called == []

@pytest.mark.parametrize("position", ["defaults", "kwdefaults"])
def test_joined_primitive_refuses_nonexact_default_containers_before_truthiness(
        monkeypatch, position):
    from ipfs_datasets_py.logic.autoformal import corpus_contracts as contracts
    called = []
    def trap(*args, **kwargs):
        called.append(True)
        raise AssertionError("default-container refusal invoked a callback")
    if position == "defaults":
        class DefaultTuple(tuple):
            __len__ = trap
        monkeypatch.setattr(contracts._loaded_owner_shape_v1, "__defaults__", DefaultTuple((None,)))
    else:
        class DefaultDict(dict):
            __len__ = trap
        monkeypatch.setattr(contracts._capture_loaded_owner_guard_v1, "__kwdefaults__",
                            DefaultDict(contracts._capture_loaded_owner_guard_v1.__kwdefaults__))
    path = Path(contracts.__file__)
    with pytest.raises(ValueError, match="defaults differ"):
        owners._joined_contracts_primitive_binding_v1(contracts, path.read_bytes(), path)
    assert called == []

def test_live_physical_join_keeps_first_R_refusal_when_later_A_fence_also_fails(
        live_physical_join_software_route):
    s = live_physical_join_software_route
    class FirstRefusal(ValueError):
        def __str__(self):
            raise AssertionError("first refusal was rendered during finalization")
    first = FirstRefusal()
    s.replay_error = first
    s.fence_error = ValueError("later current-parent refusal")
    with pytest.raises(FirstRefusal) as caught:
        owners.load_parent_verified_uscode_physical_closure_v3(
            s.handle, s.raw, expected_request_sha256=s.request_sha)
    assert caught.value is first
    assert s.calls.count("R.final") == 1 and "R.postclose" in s.calls
    assert s.calls[-1] == "A.current" and "A.seal" not in s.calls
    assert any("original parent/current inputs" in note for note in first.__notes__)

def test_joined_opening_direct_descriptor_keeps_native_producer_pin_scope(monkeypatch, tmp_path):
    root = tmp_path / "datasets"
    marker = root / "ipfs_datasets_py/logic/autoformal/legal_release_coordinator_v1.py"
    native = root / "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py"
    operation = {"expected_source_sha256": {str(native): "a" * 64}}
    descriptor = {"path": str(marker), "bytes": 1, "sha256": "b" * 64}
    native_descriptor = {"path": str(native), "bytes": 1, "sha256": "a" * 64}
    policy = {"schema": owners._reviewed_parent_domain(3)[0],
        "datasets_root": str(root), "source_descriptors": [descriptor, native_descriptor],
        "operation": operation, "validation": {"import_sources": [], "trusted_tests": [],
                                               "pytest_module": native_descriptor}}
    binding = (("software-only implementation pin join",), ())
    observed = []
    def resolve(evidence):
        observed.append(evidence)
        return None, binding
    monkeypatch.setattr(owners, "_parent_release_owner_binding_v3", resolve)
    assert owners._reviewed_parent_joined_opening_v3(policy) is binding
    assert len(observed) == 1 and observed[0]["operation"] is operation
    assert observed[0]["source_pins"][str(marker)] == descriptor["sha256"]
    assert operation["expected_source_sha256"] == {str(native): "a" * 64}

def _current_native_bootstrap_fixture(reviewed_version):
    # Reuse the owning fixture's fixed, previously approved canonical Source
    # roster and current post-CAS pins; never discover authority via a callback.
    evidence = _current_joined_owner_software_evidence()
    native = owners._parent_native_owner(evidence, reviewed_version=reviewed_version)
    path = Path(native.__file__).resolve()
    raw = path.read_bytes()
    assert evidence["source_pins"][str(path)] == hashlib.sha256(raw).hexdigest()
    return native, evidence, raw, path

def _native_bootstrap_private_cell(native, name):
    integrity = native._live_integrity_checker
    cells = dict(zip(integrity.__code__.co_freevars, integrity.__closure__, strict=True))
    if name == "states":
        verifier = cells["verify_native_functions"].cell_contents
        cells = dict(zip(verifier.__code__.co_freevars, verifier.__closure__, strict=True))
    elif name == "integrity_state":
        wrapper = native.build_verified_reviewed_norm_preservation_native_v2
        cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__, strict=True))
    return cells[name]

@pytest.mark.parametrize("reviewed_version", (2, 3))
def test_parent_native_bootstrap_accepts_actual_source_chain_without_build(monkeypatch, reviewed_version):
    native, evidence, raw, path = _current_native_bootstrap_fixture(reviewed_version)
    with monkeypatch.context() as change:
        for key in ("_LIVE_MODULE_BINDINGS", "_LIVE_REVIEW_BINDINGS", "_LIVE_MIXED_REVIEW_BINDINGS"):
            change.setattr(native, key, None)
        assert owners._parent_native_bootstrap_v1(native, raw, path, evidence["source_pins"],
                                                   reviewed_version=reviewed_version) is None
        # Exercise the actual authenticated binder with no owner traversal or
        # packet/handle/build operation, then the unchanged versioned A route.
        assert native._live_entry_guard(reviewed_version) is None
        assert owners._parent_native_owner(evidence, reviewed_version=reviewed_version) is native

@pytest.mark.parametrize("reviewed_version", (2, 3))
@pytest.mark.parametrize("target", (
    "empty_native_states", "resealed_native_states", "replaced_guard", "replaced_matcher",
    "replaced_runtime_alias", "hostile_subprocess_alias", "injected_builtin", "injected_exception", "unexpected_global",
    "replaced_builtins", "empty_handle_table", "empty_popen_table", "malformed_handle_state",
    "hostile_handle_table", "hostile_popen_table", "hostile_import_table", "hostile_review_table",
    "hostile_kwdefaults", "hostile_class_kwdefaults", "hostile_spec_fields", "hostile_module_metadata", "hostile_package_metadata",
    "hostile_private_states", "hostile_private_integrity", "hostile_class_metaclass",
    "hostile_native_class_descriptor", "hostile_dependency_field_descriptor",
    "hostile_dependency_namespace_key",
))
def test_parent_native_bootstrap_refuses_before_mutable_native_callbacks(monkeypatch, reviewed_version, target):
    import types
    from collections.abc import Mapping
    native, evidence, raw, path = _current_native_bootstrap_fixture(reviewed_version)
    callbacks = []

    def called(label):
        callbacks.append(label)
        raise AssertionError("untrusted Native callback ran")

    class Trap:
        def __call__(self, *args, **kwargs):
            return called("call")
        def __iter__(self):
            return called("iter")
        def __len__(self):
            return called("len")
        def __bool__(self):
            return called("bool")
        def __eq__(self, other):
            return called("eq")
        def __getitem__(self, key):
            return called("getitem")
        def get(self, *args):
            return called("get")
        def items(self):
            return called("items")

    class TrapDict(dict):
        def __iter__(self):
            return called("dict.iter")
        def __len__(self):
            return called("dict.len")
        def __bool__(self):
            return called("dict.bool")
        def __getitem__(self, key):
            return called("dict.getitem")
        def get(self, *args):
            return called("dict.get")
        def items(self):
            return called("dict.items")
        def __eq__(self, other):
            return called("dict.eq")

    class TrapTuple(tuple):
        def __iter__(self):
            return called("tuple.iter")
        def __len__(self):
            return called("tuple.len")
        def __bool__(self):
            return called("tuple.bool")
        def __getitem__(self, key):
            return called("tuple.getitem")
        def __eq__(self, other):
            return called("tuple.eq")

    class TrapMapping(Mapping):
        def __getitem__(self, key):
            return called("mapping.getitem")
        def __iter__(self):
            return called("mapping.iter")
        def __len__(self):
            return called("mapping.len")

    def predicate(*args, **kwargs):
        return called("predicate")

    proxy = types.MappingProxyType(TrapMapping())
    with monkeypatch.context() as change:
        if target == "empty_native_states":
            change.setattr(native, "_LIVE_NATIVE_FUNCTIONS", ())
        elif target == "resealed_native_states":
            change.setattr(native, "_live_callable_matches", predicate)
            rows = []
            for name, _state in native._LIVE_NATIVE_FUNCTIONS:
                function = vars(native)[name]
                kwargs = {} if function.__kwdefaults__ is None else function.__kwdefaults__
                rows.append((name, (function, function.__code__, function.__defaults__,
                                    tuple(sorted(kwargs.items())))))
            resealed = tuple(rows)
            change.setattr(native, "_LIVE_NATIVE_FUNCTIONS", resealed)
            change.setattr(_native_bootstrap_private_cell(native, "states"), "cell_contents", resealed)
        elif target == "replaced_guard":
            change.setattr(native, "_live_success_guard", predicate)
        elif target == "replaced_matcher":
            change.setattr(native, "_live_callable_matches", predicate)
        elif target == "replaced_runtime_alias":
            change.setattr(native, "Path", Trap())
        elif target == "hostile_subprocess_alias":
            class SubprocessAlias:
                @property
                def Popen(self):
                    return called("native.subprocess.Popen")
            change.setattr(native, "subprocess", SubprocessAlias())
        elif target == "injected_builtin":
            change.setattr(native, "type", Trap(), raising=False)
        elif target == "injected_exception":
            change.setattr(native, "ValueError", Trap(), raising=False)
        elif target == "unexpected_global":
            change.setattr(native, "_caller_native_success", True, raising=False)
        elif target == "replaced_builtins":
            change.setattr(native, "__builtins__", proxy)
        elif target == "empty_handle_table":
            change.setattr(native, "_LIVE_HANDLE_FUNCTIONS", ())
        elif target == "empty_popen_table":
            change.setattr(native, "_LIVE_POPEN_FUNCTIONS", ())
        elif target == "malformed_handle_state":
            table = native._LIVE_HANDLE_FUNCTIONS
            change.setattr(native, "_LIVE_HANDLE_FUNCTIONS", ((table[0][0], (Trap(),)), *table[1:]))
        elif target == "hostile_handle_table":
            change.setattr(native, "_LIVE_HANDLE_FUNCTIONS", TrapTuple())
        elif target == "hostile_popen_table":
            change.setattr(native, "_LIVE_POPEN_FUNCTIONS", proxy)
        elif target == "hostile_import_table":
            change.setattr(native, "_LIVE_MODULE_BINDINGS", proxy)
        elif target == "hostile_review_table":
            change.setattr(native, "_LIVE_REVIEW_BINDINGS" if reviewed_version == 2
                           else "_LIVE_MIXED_REVIEW_BINDINGS", Trap())
        elif target == "hostile_kwdefaults":
            change.setattr(native._owners, "__kwdefaults__", TrapDict())
        elif target == "hostile_class_kwdefaults":
            function = vars(native.VerifiedReviewedPreservationNativeSuccess)["__new__"].__func__
            change.setattr(function, "__kwdefaults__", TrapDict())
        elif target == "hostile_spec_fields":
            change.setattr(native.__spec__, "__dict__", TrapDict(vars(native.__spec__)))
        elif target == "hostile_module_metadata":
            change.setattr(native, "__name__", Trap())
        elif target == "hostile_package_metadata":
            change.setattr(native, "__package__", Trap())
        elif target == "hostile_private_states":
            change.setattr(_native_bootstrap_private_cell(native, "states"), "cell_contents", TrapTuple())
        elif target == "hostile_private_integrity":
            change.setattr(_native_bootstrap_private_cell(native, "integrity_state"), "cell_contents", proxy)
        elif target == "hostile_native_class_descriptor":
            class EqualityMeta(type):
                def __eq__(self, other):
                    return called("descriptor.metaclass.eq")
                __hash__ = type.__hash__
            class Descriptor(metaclass=EqualityMeta):
                pass
            descriptor = Descriptor()
            callbacks.clear()
            change.setattr(native.VerifiedReviewedPreservationNativeSuccess, "__new__", descriptor)
        elif target == "hostile_dependency_field_descriptor":
            class EqualityMeta(type):
                def __eq__(self, other):
                    return called("dependency.field.metaclass.eq")
                __hash__ = type.__hash__
            class Descriptor(metaclass=EqualityMeta):
                pass
            dependency_class = next(row[1] for snapshot in native._LIVE_MODULE_BINDINGS
                                    for row in snapshot[3])
            descriptor = Descriptor()
            callbacks.clear()
            change.setattr(dependency_class, "_caller_descriptor", descriptor, raising=False)
            change.setattr(native, "_LIVE_MODULE_BINDINGS", None)
            change.setattr(native, "_LIVE_REVIEW_BINDINGS", None)
            change.setattr(native, "_LIVE_MIXED_REVIEW_BINDINGS", None)
        elif target == "hostile_dependency_namespace_key":
            dependency = native._LIVE_MODULE_BINDINGS[0][0]
            class Key:
                def __hash__(self):
                    return hash("__name__")
                def __eq__(self, other):
                    return called("dependency.namespace.key.eq")
            key = Key()
            namespace = vars(dependency)
            change.delitem(namespace, "__name__")
            change.setitem(namespace, key, None)
        elif target == "hostile_class_metaclass":
            class Meta(type):
                def __getattribute__(self, name):
                    return called("metaclass.getattribute")
            replacement = Meta("VerifiedReviewedPreservationNativeSuccess", (), {})
            callbacks.clear()
            change.setattr(native, "VerifiedReviewedPreservationNativeSuccess", replacement)
        else:
            raise AssertionError("unknown mutation")
        with pytest.raises(ValueError):
            owners._parent_native_owner(evidence, reviewed_version=reviewed_version)
        assert callbacks == []
    assert callbacks == []

@pytest.mark.parametrize("binding", ("type", "len", "ValueError", "__builtins__"))
def test_source_builtin_preflight_refuses_contracts_shadow_before_callback(monkeypatch, binding):
    import ast
    from ipfs_datasets_py.logic.autoformal import corpus_contracts as contracts
    raw = Path(contracts.__file__).read_bytes()
    tree = ast.parse(raw, filename=contracts.__file__)
    calls = []
    class Trap:
        def __call__(self, *args, **kwargs):
            calls.append("call")
            raise AssertionError("untrusted contracts builtin ran")
        def __bool__(self):
            calls.append("bool")
            raise AssertionError("untrusted contracts truthiness ran")
    assert owners._parent_source_builtin_bindings_v1(vars(contracts), tree) is None
    with monkeypatch.context() as change:
        change.setattr(contracts, binding, Trap(), raising=False)
        with pytest.raises(ValueError):
            owners._parent_source_builtin_bindings_v1(vars(contracts), tree)
        assert calls == []

def test_joined_coordinator_binding_lookup_rejects_custom_keys_before_equality(monkeypatch):
    from ipfs_datasets_py.logic.autoformal import legal_release_coordinator_v1 as release
    evidence = _current_joined_owner_software_evidence()
    namespace = vars(release)
    key_name = "_JOIN_RELEASE_OWNER_MODULES_V1"
    called = []
    class Key:
        def __hash__(self):
            return hash(key_name)
        def __eq__(self, other):
            called.append(True)
            raise AssertionError("coordinator lookup invoked caller equality")
    with monkeypatch.context() as change:
        change.delitem(namespace, key_name)
        change.setitem(namespace, Key(), None)
        with pytest.raises(ValueError, match="coordinator namespace keys"):
            owners._parent_release_owner_binding_v3(evidence)
    assert called == []

def test_live_physical_join_retains_available_parent_fence_after_postsnapshot_refusal(
        live_physical_join_software_route, monkeypatch):
    s = live_physical_join_software_route
    release, _binding = owners._parent_release_owner_binding_v3(s.evidence)
    original = release._physical_replay_context_snapshot
    failure = ValueError("injected immediate post-capture snapshot refusal")
    injected = []
    def snapshot(context):
        if "A.capture" in s.calls and not injected:
            injected.append(True)
            raise failure
        return original(context)
    monkeypatch.setattr(release, "_physical_replay_context_snapshot", snapshot)
    with pytest.raises(ValueError) as caught:
        owners.load_parent_verified_uscode_physical_closure_v3(
            s.handle, s.raw, expected_request_sha256=s.request_sha)
    assert caught.value is failure and injected == [True]
    assert s.calls.count("A.capture") == s.calls.count("R.final") == 1
    assert "R.postclose" in s.calls and s.calls[-1] == "A.current"
    assert "A.seal" not in s.calls


# Explicit A protocol4 / R physical-native5 controls.  Modeled byte inventories
# below exercise declarations only; they never authenticate Native execution.
def prepare_reviewed_parent_physical_realization_fixture_inputs_v1(path, control, *, datasets_root,
        physical_request_raw, expected_physical_request_sha256):
    assert type(control) is dict and type(physical_request_raw) is bytes
    assert hashlib.sha256(physical_request_raw).hexdigest() == expected_physical_request_sha256
    shared = control["shared"]
    assert control["admission"]["fixture_only"] is True
    assert control["admission"]["provenance_kind"] == "fictional_fixture"
    value = {"schema": "reviewed-definition-realization-fixture-native-inputs/v1",
        "fixture_only": True, "provenance_kind": "fictional_fixture",
        **{name: {"manifest_json": control[name].manifest_json,
            "files": [[key, body] for key, body in control[name].files]}
            for name in ("preparation", "preservation")},
        **{name: reviewed_parent_fixture_tree_encode_v1(control[name])
            for name in ("graph", "admission", "args", "kwargs")},
        "expected_realization_sha256": shared.reviewed_preservation_corpus_sha256(control["preparation"]),
        "expected_preservation_sha256": shared.reviewed_preservation_corpus_sha256(control["preservation"]),
        "expected_graph_sha256": control["graph_sha"], "expected_admission_sha256": control["admission_sha"],
        "toolchain": json.loads(control["preparation"].manifest_json)["toolchain"],
        "instance_identity_profile": "reviewed-source-use/v1",
        "definition_application_profile": "whole-norm-definition-expansion/v1",
        "physical_request_raw": reviewed_parent_fixture_tree_encode_v1(physical_request_raw),
        "expected_physical_request_sha256": expected_physical_request_sha256}
    # The original declaration is already inside kwargs.preparation_inputs;
    # no independent review, signer, target or consumer declaration is exported.
    paths = owners._reviewed_parent_realization_packet_paths_v1(value)
    raw = owners.canonical(value) + b"\n"
    assert len(raw) <= 4 * 1024**2
    reference = put(Path(path), raw)
    return {"native_inputs": reference, "expected_packet_paths": paths,
        **{name: value[name] for name in ("expected_realization_sha256", "expected_preservation_sha256",
            "expected_graph_sha256", "expected_admission_sha256", "toolchain",
            "expected_physical_request_sha256", "instance_identity_profile", "definition_application_profile")}}


@pytest.fixture
def reviewed_parent_physical_realization_declaration_v1(reviewed_parent_v3_complete_packet_scope):
    """Modeled A4 declaration only over retained V3 unit inventory; no receipt."""
    old = reviewed_parent_v3_complete_packet_scope
    plan, policy = copy.deepcopy((old.declaration.plan, old.declaration.policy))
    operation = plan["operation"]
    operation.update(expected_realization_sha256="a" * 64, expected_physical_request_sha256="b" * 64,
        instance_identity_profile="reviewed-source-use/v1", definition_application_profile="whole-norm-definition-expansion/v1")
    policy["operation"] = copy.deepcopy(operation)
    policy["validation"]["storage_bytes"] = 256 * 1024**2
    plan["schema"] = "reviewed-definition-realization-native-worker/v1"
    policy["schema"] = "reviewed-definition-realization-parent/v1"
    return old.declaration.worker, plan, policy


def test_reviewed_physical_realization_worker_declaration_is_detached_and_domain_specific(
        reviewed_parent_physical_realization_declaration_v1):
    worker, plan, policy = reviewed_parent_physical_realization_declaration_v1
    before = copy.deepcopy((plan, policy))
    declared = worker.reviewed_definition_realization_native_operation_v1(plan, policy)
    assert declared == policy["operation"] and declared is not plan["operation"]
    assert (plan, policy) == before
    assert owners._reviewed_parent_domain_v3(4)["native_version"] == 5
    assert owners._reviewed_parent_domain_v3(2)["native_version"] == 2
    assert owners._reviewed_parent_domain_v3(3)["native_version"] == 3
    assert owners._reviewed_parent_version_v3(policy) == worker._reviewed_native_plan_version_v3(plan) == 4
    with pytest.raises(ValueError, match="explicit reviewed native worker profile"):
        worker.reviewed_preservation_native_operation_v3(plan, policy)


@pytest.mark.parametrize("field,value", [
    ("expected_realization_sha256", None), ("expected_physical_request_sha256", "0"),
    ("instance_identity_profile", None), ("definition_application_profile", "constructed-model-only"),
])
def test_reviewed_physical_realization_declaration_refuses_crossed_or_missing_original_inputs(
        reviewed_parent_physical_realization_declaration_v1, field, value):
    worker, plan, policy = reviewed_parent_physical_realization_declaration_v1
    plan["operation"][field] = policy["operation"][field] = value
    with pytest.raises(ValueError, match="physical source-use whole-norm"):
        worker.reviewed_definition_realization_native_operation_v1(plan, policy)


@pytest.mark.parametrize("version", [None, True, 1, 5])
def test_reviewed_physical_realization_protocol_is_not_native_version_or_flag(version):
    with pytest.raises(ValueError, match="closed reviewed native version"):
        owners._reviewed_parent_domain_v3(version)


def test_reviewed_physical_realization_unissued_parent_refused_before_native_lookup(monkeypatch):
    calls = []
    monkeypatch.setattr(owners, "_parent_native_owner", lambda *_a, **_k: calls.append(True))
    with pytest.raises(ValueError, match="not issued by this process and native thread"):
        owners.load_parent_verified_uscode_physical_realization_closure_v1(
            object(), b"{}", expected_request_sha256=hashlib.sha256(b"{}").hexdigest())
    assert calls == []


def test_reviewed_definition_realization_parent_actual_native_child_v1():
    """One actual physicalV2 build when enrolled by the original A4 parent only."""
    import importlib
    import os
    names = ("IPFS_REVIEWED_PRESERVATION_PLAN", "IPFS_REVIEWED_PRESERVATION_ATTEMPT", "IPFS_REVIEWED_PRESERVATION_PACKET_ROOT")
    if os.environ.get("IPFS_DATASETS_RUN_NATIVE_LEAN") != "1" or not all(os.environ.get(name) for name in names):
        pytest.skip("explicit original-parent physical realization enrollment required")
    plan_path, attempt, packet_root = (Path(os.environ[name]) for name in names)
    plan = owners.json_object(owners.ordinary_bytes(plan_path, 4 * 1024**2))
    assert plan["schema"] == "reviewed-definition-realization-native-worker/v1"
    policy = owners.json_object(owners.checked_descriptor(plan["policy"])); operation = plan["operation"]
    assert operation == policy["operation"] and plan["native_inputs"] == operation["native_inputs"]
    assert operation["nodeid"] == str(Path(__file__).absolute()) + "::test_reviewed_definition_realization_parent_actual_native_child_v1"
    assert attempt == Path(plan["attempt_directory"]) and packet_root == attempt / operation["packet_root"]
    assert not packet_root.exists()
    helper = reviewed_parent_original_R_fixture_module_v1(Path(policy["datasets_root"]))
    value = owners.json_object(owners.checked_descriptor(operation["native_inputs"]))
    physical = owners._reviewed_parent_realization_inputs_v1(value, operation)
    assert owners._reviewed_parent_realization_packet_paths_v1(value) == operation["expected_packet_paths"]
    shared = importlib.import_module("ipfs_datasets_py.logic.autoformal.legal_corpus_shared_lake")
    c = {name: shared.ReviewedPreservationCorpusPreparation(value[name]["manifest_json"],
            tuple((key, body) for key, body in value[name]["files"])) for name in ("preparation", "preservation")}
    c.update({name: reviewed_parent_fixture_tree_decode_v1(value[name]) for name in ("graph", "admission", "args", "kwargs")})
    c.update(shared=shared, owner=shared._reviewed_mixed_preservation_owners()[1],
        graph_sha=value["expected_graph_sha256"], admission_sha=value["expected_admission_sha256"])
    c["case"] = {"declaration": c["kwargs"]["preparation_inputs"]["declaration"]}
    options = {name: value[name] for name in ("expected_realization_sha256", "expected_preservation_sha256",
        "expected_graph_sha256", "expected_admission_sha256", "toolchain", "instance_identity_profile",
        "definition_application_profile", "expected_physical_request_sha256")}
    options["physical_request_raw"] = physical
    manifest = c["preparation"].to_dict()
    assert len(manifest["units"]) == 3
    assert manifest["definition_application_obligations"] and manifest["definition_applications"]
    assert set(c["preservation"].to_dict()["preservation_obligations"]) < set(manifest["preservation_obligations"])
    assert len(manifest["expected_modules"]) <= 11 and len(c["preparation"].files) <= 78
    assert manifest["upstream_manifest"] == c["preservation"].to_dict()
    executable = next(path for path in operation["expected_binary_sha256"] if Path(path).name == "lake")
    native_handle = helper.native.build_verified_reviewed_physical_definition_realization_native_v2(
        c["preparation"], c["preservation"], c["graph"], c["admission"], *c["args"],
        **options, **c["kwargs"], lake_executable=executable, output_directory=packet_root, timeout_seconds=60)
    packet = helper.reviewed_live_success_packet_v2(packet_root)
    assert sorted(name for name, _ in packet) == operation["expected_packet_paths"]
    result = helper.native.load_verified_reviewed_physical_definition_realization_native_v2(
        native_handle, packet, c["preparation"], c["preservation"], c["graph"], c["admission"], *c["args"],
        expected_receipt_sha256=hashlib.sha256(dict(packet)["build-receipt.json"]).hexdigest(),
        expected_packet_sha256=helper.native.reviewed_physical_definition_realization_packet_sha256_v2(packet),
        expected_binary_sha256=operation["expected_binary_sha256"], **options, **c["kwargs"])
    assert result["live_native_success_authenticated"] is result["native_execution_authenticated"] is True
    assert result["original_success_authenticated"] is result["original_admitted_parent_custody_authenticated"] is False
    assert result["training_mask"] == result["evaluation_gold_mask"] == 0
    assert all(flag is False for flag in result["authority"].values())
    assert result["release_ready"] is result["Lean_admitted"] is False
    # This exact original signed replay drives the existing fresh wrong-target
    # refusal (zero callback), then one detached complete-three-unit inspection.
    helper.reviewed_whole_norm_consumer_checks_v3(c)


@pytest.mark.parametrize("version,storage", [(2, 16 * 1024**2), (3, 16 * 1024**2), (4, 256 * 1024**2)])
def test_reviewed_parent_physical_realization_uses_only_its_explicit_storage_envelope(
        reviewed_parent_physical_realization_declaration_v1, version, storage):
    worker, plan, policy = reviewed_parent_physical_realization_declaration_v1
    if version != 4:
        for key in owners._reviewed_realization_operation_fields_v1():
            plan["operation"].pop(key); policy["operation"].pop(key)
        if version == 2:
            plan["operation"].pop("expected_packet_paths"); policy["operation"].pop("expected_packet_paths")
        domain = owners._reviewed_parent_domain_v3(version)
        plan["schema"], policy["schema"] = domain["worker_schema"], domain["policy_schema"]
    policy["validation"]["storage_bytes"] = storage
    worker._reviewed_preservation_native_operation_core_v3(plan, policy, reviewed_version=version)
    policy["validation"]["storage_bytes"] = 16 * 1024**2 if version == 4 else 256 * 1024**2
    with pytest.raises(ValueError, match="validation envelope"):
        worker._reviewed_preservation_native_operation_core_v3(plan, policy, reviewed_version=version)


def test_reviewed_physical_realization_packet_budget_is48MiB_without_widening_v3(
        reviewed_parent_v3_complete_packet_scope):
    # Pure modeled inventory arithmetic only. No large file/compiler/receipt or
    # purported original success is manufactured by this boundary control.
    s = reviewed_parent_v3_complete_packet_scope
    operation, binding, durable = copy.deepcopy((s.packet.operation, s.packet.binding, s.packet.durable))
    compiled = [row for row in binding["packet_inventory"] if row["path"].startswith("compiled/")]
    assert len(compiled) >= 7
    for row in compiled[:6]:
        row.update(bytes=7 * 1024**2, sha256="d" * 64)
        physical = next(x for x in durable if x["path"] == operation["packet_root"] + "/" + row["path"])
        physical.update(bytes=row["bytes"], sha256=row["sha256"])
    binding["packet_sha256"] = owners.digest(binding["packet_inventory"])
    with pytest.raises(ValueError, match="sorted bounded complete original native packet"):
        owners._reviewed_parent_packet_binding(operation, binding, durable,
            attempt_directory=s.packet.attempt, reviewed_version=3)
    operation.update(expected_realization_sha256="a" * 64, expected_physical_request_sha256="b" * 64,
        instance_identity_profile="reviewed-source-use/v1", definition_application_profile="whole-norm-definition-expansion/v1")
    binding.update(schema="reviewed-definition-realization-native-binding/v1", preparation_sha256="a" * 64,
        upstream_preservation_sha256=operation["expected_preservation_sha256"], physical_request_sha256="b" * 64,
        instance_identity_profile=operation["instance_identity_profile"],
        definition_application_profile=operation["definition_application_profile"])
    assert owners._reviewed_parent_packet_binding(operation, binding, durable,
        attempt_directory=s.packet.attempt, reviewed_version=4)
    row = compiled[6]; row.update(bytes=7 * 1024**2, sha256="e" * 64)
    physical = next(x for x in durable if x["path"] == operation["packet_root"] + "/" + row["path"])
    physical.update(bytes=row["bytes"], sha256=row["sha256"])
    binding["packet_sha256"] = owners.digest(binding["packet_inventory"])
    with pytest.raises(ValueError, match="sorted bounded complete original native packet"):
        owners._reviewed_parent_packet_binding(operation, binding, durable,
            attempt_directory=s.packet.attempt, reviewed_version=4)


@pytest.fixture(scope="module")
def reviewed_parent_genuine_physical_realization_inputs_v1(tmp_path_factory):
    from ipfs_datasets_py.logic.autoformal import legal_corpus_definition_use_native_v2 as current_native
    root = Path(current_native.__file__).resolve().parents[3]
    helper = reviewed_parent_original_R_fixture_module_v1(root)
    c = helper.reviewed_physical_realization_native_control_v2.__wrapped__(tmp_path_factory)
    directory = tmp_path_factory.mktemp("a-physical-realization-public-data")
    prepared = prepare_reviewed_parent_physical_realization_fixture_inputs_v1(
        directory / "native-inputs.json", c, datasets_root=root,
        physical_request_raw=c["physical_request_raw"], expected_physical_request_sha256=c["physical_request_sha256"])
    return c, prepared


def test_reviewed_parent_physical_realization_opening_uses_genuine_original_signed_inputs(
        reviewed_parent_genuine_physical_realization_inputs_v1):
    c, prepared = reviewed_parent_genuine_physical_realization_inputs_v1
    data = owners.json_object(owners.checked_descriptor(prepared["native_inputs"]))
    assert owners._reviewed_parent_realization_inputs_v1(data, data) == c["physical_request_raw"]
    assert owners._reviewed_parent_realization_packet_paths_v1(data) == prepared["expected_packet_paths"]
    assert data["expected_realization_sha256"] == c["realization_sha"]
    assert data["expected_physical_request_sha256"] == c["physical_request_sha256"]
    assert reviewed_parent_fixture_tree_decode_v1(data["args"])[0] == c["args"][0]
    assert reviewed_parent_fixture_tree_decode_v1(data["kwargs"]) == c["kwargs"]
    assert owners.digest(data["preparation"]) != owners.digest(data["preservation"])
    assert c["admission"]["fixture_only"] is True
    assert c["admission"]["training_mask"] == c["admission"]["evaluation_gold_mask"] == 0


@pytest.mark.parametrize("mutation", ["upstream", "realization", "physical", "profile", "file_body"])
def test_reviewed_parent_physical_realization_opening_refuses_crossed_complete_inputs(
        reviewed_parent_genuine_physical_realization_inputs_v1, mutation):
    _c, prepared = reviewed_parent_genuine_physical_realization_inputs_v1
    data = owners.json_object(owners.checked_descriptor(prepared["native_inputs"]))
    if mutation == "upstream":
        data["preservation"] = copy.deepcopy(data["preparation"])
    elif mutation == "realization":
        data["preparation"] = copy.deepcopy(data["preservation"])
    elif mutation == "physical":
        data["physical_request_raw"] = reviewed_parent_fixture_tree_encode_v1(b"{}")
    elif mutation == "profile":
        data["instance_identity_profile"] = None
    else:
        data["preparation"]["files"][0][1] += "\n-- changed\n"
        data["expected_realization_sha256"] = owners.digest(data["preparation"])
    with pytest.raises(ValueError):
        owners._reviewed_parent_realization_packet_paths_v1(data)
