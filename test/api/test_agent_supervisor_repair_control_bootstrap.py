"""Prospective genuine cold control-source tests; no native/service credit.

Private control copies contain complete real maintained normal source files.
Each cold process uses actual Popen/wait under the qualification owner's finite
process envelope. No test or native admission is inferred from source readiness.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys

import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners
from scripts.ops.agent_supervisor import install_fleet_watchdog as installer

# Filled from the already reviewed SOURCE43 normal A closure. The qualifier must
# include these current exact maintained files, with the proposed owners/worker
# overlay, in its own frozen source manifest; no canonical fallback is used.
CONTROL_NORMAL_CLOSURE = ('ipfs_accelerate_py/__init__.py', 'ipfs_accelerate_py/agent_supervisor/__init__.py', 'ipfs_accelerate_py/agent_supervisor/control/__init__.py', 'ipfs_accelerate_py/agent_supervisor/control/control_contracts.py', 'ipfs_accelerate_py/agent_supervisor/control/control_plane.py', 'ipfs_accelerate_py/agent_supervisor/core/__init__.py', 'ipfs_accelerate_py/agent_supervisor/core/wrapper_utils.py', 'ipfs_accelerate_py/agent_supervisor/git_environment.py', 'ipfs_accelerate_py/agent_supervisor/merge/__init__.py', 'ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py', 'ipfs_accelerate_py/agent_supervisor/merge/quarantine_validation.py', 'ipfs_accelerate_py/agent_supervisor/merge/workspace_quarantine.py', 'ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py', 'ipfs_accelerate_py/agent_supervisor/objectives/__init__.py', 'ipfs_accelerate_py/agent_supervisor/objectives/goal_completion.py', 'ipfs_accelerate_py/agent_supervisor/proof/__init__.py', 'ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py', 'ipfs_accelerate_py/agent_supervisor/rescue/__init__.py', 'ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py', 'ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py', 'ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py', 'ipfs_accelerate_py/agent_supervisor/runtime/__init__.py', 'ipfs_accelerate_py/agent_supervisor/runtime/event_log.py', 'ipfs_accelerate_py/agent_supervisor/self_improvement/__init__.py', 'ipfs_accelerate_py/agent_supervisor/self_improvement/self_improvement_completion.py', 'ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_v2_contracts.py', 'ipfs_accelerate_py/agent_supervisor/task_sources/__init__.py', 'ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/__init__.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/app.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/artifacts.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/auto_commit.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/cli.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/context.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/deterministic_fallback.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/diagnostics.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/file_replacement.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/git_utils.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/history.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/lifecycle_wrapper.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/llm_defaults.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/plans.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/runner.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/specs.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/status.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/task_board.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/typescript.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/worktrees.py', 'ipfs_accelerate_py/agent_supervisor/todo_daemon/wrapper.py', 'ipfs_accelerate_py/agent_supervisor/validation/__init__.py', 'ipfs_accelerate_py/agent_supervisor/validation/validation_commands.py', 'ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def file_pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def sealed(path, raw):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw)
    return file_pin(path)


def cold_scope(tmp_path, *, complete=True, manifest_name="repair-control-manifest.json"):
    source_root = Path(os.environ["IPFS_ACCELERATE_VALIDATION_CONTROL_ROOT"])
    manifest_path = Path(os.environ["IPFS_ACCELERATE_VALIDATION_CONTROL_MANIFEST_PATH"])
    expected = os.environ["IPFS_ACCELERATE_VALIDATION_CONTROL_MANIFEST_SHA256"]
    manifest_raw = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == expected
    original = json.loads(manifest_raw)
    assert original["root"] == str(source_root)
    original_pins = {row["path"]: row for row in original["files"]}
    root = tmp_path / "immutable-control"; root.mkdir(mode=0o700)
    pins = []
    minimum = {"ipfs_accelerate_py/__init__.py", "ipfs_accelerate_py/agent_supervisor/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/__init__.py", "ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py", "ipfs_accelerate_py/agent_supervisor/rescue/repair_validation_worker.py"}
    for relative in CONTROL_NORMAL_CLOSURE:
        if not complete and relative not in minimum:
            continue
        source = source_root / relative
        raw = source.read_bytes(); expected_pin = original_pins[str(source)]
        assert len(raw) == expected_pin["bytes"] and hashlib.sha256(raw).hexdigest() == expected_pin["sha256"]
        pins.append(sealed(root / relative, raw))
    owner = root / "ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py"
    worker = owner.with_name("repair_validation_worker.py")
    manifest = {"schema": "agent-supervisor/repair-control-source-release@1", "root": str(root), "files": sorted(pins, key=lambda pin: pin["path"]), "namespace_profiles": []}
    manifest_pin = sealed(root / manifest_name, canonical(manifest) + b"\n")
    d = tmp_path / "canonical-D"; d.mkdir()
    active = tmp_path / "active-A"; active.mkdir()
    # This unrelated source must never run, even with cwd and PYTHONPATH set.
    sealed(active / "ipfs_accelerate_py/__init__.py", b"raise RuntimeError('untrusted_active_A_ran')\n")
    policy = {"schema": "agent-supervisor/repair-public-owner-policy@2", "datasets_root": str(d), "python": sys.executable,
        "source_descriptors": [file_pin(owner), file_pin(worker)],
        "validation": {"import_sources": []}, "adoption": {},
        "control_release": {"root": str(root), "manifest": manifest_pin}}
    policy_pin = sealed(tmp_path / "owner-policy.json", canonical(policy) + b"\n")
    return root, owner, active, d, policy, policy_pin, manifest


def cold_probe(owner, active, policy_pin):
    command = [sys.executable, "-I", "-B", str(owner), "inspect-control",
        "--policy", policy_pin["path"], "--policy-sha256", policy_pin["sha256"]]
    environment = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
    environment["PYTHONPATH"] = str(active)
    process = subprocess.Popen(command, cwd=active, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    # Root's original cohort still owns process-tree stop/reap if this test
    # encounters an infrastructure timeout. The bounded cold child is real.
    try:
        stdout, stderr = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        # This is our original still-owned child session, never a discovered
        # unrelated process. Preserve its real terminal while stopping it.
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        raise
    assert process.poll() == process.returncode and process.returncode is not None
    return process.returncode, stdout, stderr


def test_cold_control_survives_changed_active_source_and_poisoned_pythonpath(tmp_path):
    assert owners.POLICY_SCHEMA == "agent-supervisor/repair-public-owner-policy@2"
    assert owners.CONTROL_SCHEMA == "agent-supervisor/repair-control-source-release@1"
    root, owner, active, _, policy, pin, manifest = cold_scope(tmp_path, manifest_name="pinned-control-sources.json")
    rc, stdout, stderr = cold_probe(owner, active, pin)
    assert rc == 0, stderr.decode(errors="replace")
    value = json.loads(stdout)
    assert value["schema"] == "agent-supervisor/repair-control-source-inspection@1"
    assert value["control_release"] == policy["control_release"]
    assert value["actual_owner"] == file_pin(owner)
    assert value["file_count"] == len(manifest["files"])
    assert value["python_isolated"] == 1 and value["no_bytecode"] is True
    assert value["state_written"] is value["resource_admission_performed"] is value["task_acceptance_asserted"] is False
    assert value["source_guard_denials"] == []
    assert value["declared_normal_namespace_profiles"] == [] and value["observed_normal_namespace_origins"] == {}
    for relative in ("ipfs_accelerate_py/__init__.py", "ipfs_accelerate_py/agent_supervisor/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/rescue/__init__.py",
            "ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py"):
        path = root / relative
        assert value["executed_source_buffers"][str(path)] == file_pin(path)["sha256"]
    assert all(root in Path(path).parents for path in value["executed_source_buffers"])


@pytest.mark.parametrize("failure", ["manifest_tamper", "worker_tamper", "manifest_writable", "missing_source", "policy_tamper"])
def test_cold_control_refuses_changed_or_incomplete_population_without_active_fallback(tmp_path, failure):
    root, owner, active, _, _, pin, _ = cold_scope(tmp_path, complete=False)
    if failure == "manifest_tamper":
        target = root / "repair-control-manifest.json"; target.chmod(0o600); target.write_bytes(b"{}\n"); target.chmod(0o400)
    elif failure == "worker_tamper":
        target = owner.with_name("repair_validation_worker.py"); target.chmod(0o600)
        target.write_bytes(b"raise RuntimeError('tampered_worker_body_ran')\n"); target.chmod(0o400)
    elif failure == "manifest_writable":
        (root / "repair-control-manifest.json").chmod(0o600)
    elif failure == "policy_tamper":
        target = Path(pin["path"]); target.chmod(0o600); target.write_bytes(b"{}\n"); target.chmod(0o400)
    else:
        (root / "ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py").unlink()
    rc, stdout, stderr = cold_probe(owner, active, pin)
    assert rc != 0 and stdout == b""
    assert b"untrusted_active_A_ran" not in stderr and b"tampered_worker_body_ran" not in stderr
    assert b"control descriptor changed" in stderr or b"sealed before use" in stderr or b"FileNotFoundError" in stderr or b"owner policy changed before control bootstrap" in stderr


def test_cold_control_never_uses_cached_bytecode_over_changed_raw_source(tmp_path):
    root, owner, active, _, _, pin, _ = cold_scope(tmp_path, complete=False)
    source = root / "ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py"
    cached = Path(py_compile.compile(str(source), doraise=True))
    assert cached.is_file() and cached.read_bytes()
    source.chmod(0o600); source.write_bytes(b"def uncompiled_changed_source(:\n"); source.chmod(0o400)
    rc, stdout, stderr = cold_probe(owner, active, pin)
    assert rc != 0 and stdout == b"" and b"control descriptor changed" in stderr


def test_control_manifest_rejects_aliased_source_before_any_project_initialization(tmp_path):
    root, owner, active, _, _, pin, _ = cold_scope(tmp_path, complete=False)
    source = root / "ipfs_accelerate_py/agent_supervisor/rescue/repair_authorization.py"
    target = tmp_path / "same-bytes-foreign-authorization.py"
    sealed(target, source.read_bytes()); source.unlink(); source.symlink_to(target)
    rc, stdout, stderr = cold_probe(owner, active, pin)
    assert rc != 0 and stdout == b"" and b"unaliased absolute owner file required" in stderr


@pytest.mark.parametrize("scientific_first", [True, False])
def test_upgrade_renderer_selects_unique_legal_ir_and_preserves_other_board_values(tmp_path, monkeypatch, scientific_first):
    """Actual CAS/unit renderer, private modeled release population only.

    No Git/source lineage, manager, activation, native, or test acceptance is
    asserted. The original descriptor checks and final plan verifier run.
    """
    home = tmp_path / "home"; state = tmp_path / "state"; state.mkdir()
    source = tmp_path / "active-source"; source.mkdir()
    relative = "ipfs_accelerate_py/agent_supervisor/rescue/fleet_watchdog.py"
    source_pin = sealed(source / relative, b"# private renderer source fixture\n")
    prior = {"repository_root": str(source), "source_paths": [relative], "base_commit": "1" * 40}
    following = {**prior, "base_commit": "2" * 40}
    runtime = {"source_descriptors": [source_pin],
        "validation_owner": {"preserved_immutable_owner": "validate"},
        "adoption_owner": {"preserved_immutable_owner": "adopt"}}
    engineering = {"id": "legal-ir", "repair_authorization_epoch": {"unchanged_existing_epoch": True}, "arbitrary": [1, {"keep": True}]}
    scientific = {"id": "legal-ir-scientific", "activation": {"foreign_exact_gate": "retained"},
        "diagnostic_handoff": {"values": [False, None, "exact"]}, "max_workers": 3}
    boards = [scientific, engineering] if scientific_first else [engineering, scientific]
    old = {"state_dir": str(state), "boards": boards,
        "repair_worker": {"candidate_policy": prior, "candidate_runtime": runtime, "job_unit": "ipfs-taskboard-legal-ir-job.service"}}
    config_pin = sealed(home / ".config/ipfs-taskboard-watchdog/legal-ir/fleet.json", canonical(old) + b"\n")
    inventory_pin = sealed(tmp_path / "inventory.json", b"{}\n")
    job_pin = sealed(state / "repairs/legal-ir/job.json", b"{}\n")
    initial_pin = sealed(tmp_path / "initial-release.json", b"{}\n")
    units = {suffix: sealed(home / ("ipfs-taskboard-legal-ir-" + suffix + ".service"), b"# original private unit fixture\n")
        for suffix in ("watchdog", "repair")}
    release = tmp_path / "immutable-runtime"; release.mkdir()
    contents = {name: b"# private modeled renderer release; never executed\n" for name in ("fleet_watchdog", "fleet_repair", "repair_candidate_owners")}
    for name, raw in contents.items():
        sealed(release / "ipfs_accelerate_py/agent_supervisor/rescue" / (name + ".py"), raw)
    monkeypatch.setattr(installer, "_finite_repair_release", lambda selected_source, selected_home, selected_paths: (release, contents, {relative: contents["fleet_watchdog"]}))
    commitment = "a" * 64
    successor = {"source_commitment": commitment, "prior_policy": prior, "successor_policy": following,
        "successor_source_descriptors": [source_pin], "new_commit": "2" * 40, "new_tree": "3" * 40}
    contract = {"schema": "agent-supervisor/repair-profile-upgrade-contract@1", "profile": "legal-ir",
        "config": config_pin, "inventory": inventory_pin, "units": units, "prior_job": job_pin,
        "expected_loaded_before": {}, "job_unit": old["repair_worker"]["job_unit"], "source_successor": successor, "initial_release": initial_pin}
    plan = installer.prepare_repair_upgrade(source, contract, commitment)
    post = json.loads(next(row["postimage_utf8"] for row in plan["files"] if row["before"] == config_pin))
    assert post["boards"] == old["boards"]
    assert post["repair_worker"]["candidate_runtime"]["validation_owner"] == runtime["validation_owner"]
    assert post["repair_worker"]["candidate_runtime"]["adoption_owner"] == runtime["adoption_owner"]
    assert post["repair_worker"]["candidate_policy"] == following
    assert plan["service_actions"] == 0 and plan["staged_only"] is True
    assert Path(config_pin["path"]).read_bytes() == canonical(old) + b"\n"
    assert file_pin(Path(job_pin["path"])) == job_pin
