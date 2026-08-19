"""Focused fail-closed tests for the EAAEF bootstrap materializer."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
    install_eaaef_operational_schema,
)

ROOT = Path(__file__).resolve().parents[2]
MATERIALIZER_PATH = (
    ROOT / "scripts/materialize_external_agent_autonomous_execution_fabric_control_plane.py"
)
SPEC = importlib.util.spec_from_file_location("eaaef_materializer_test_subject", MATERIALIZER_PATH)
assert SPEC is not None and SPEC.loader is not None
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _config(prefix: str = "data/eaaef-test/run-v1") -> dict[str, object]:
    control = f"{prefix}/control.duckdb"
    runtime_binding = json.loads(
        materializer.CONFIG_PATH.read_text(encoding="utf-8")
    )["bootstrap_runtime_binding"]
    runtime_binding["launcher"]["allowed_commands"] = [
        "build",
        "runtime-check",
        "materialize",
        "verify",
        "launch-plan",
        "configured-board-launch",
    ]
    runtime_binding["launcher"]["sha256"] = (
        "sha256:"
        + hashlib.sha256(
            Path(runtime_binding["launcher"]["resolved_path"]).read_bytes()
        ).hexdigest()
    )
    return {
        "schema": materializer.SCHEDULER_CONFIG_SCHEMA,
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "taskboard_path": "docs/architecture/external_agent_autonomous_execution_fabric/TASK_BOARD.md",
        "task_prefix": "EAAEF-",
        "merge_target_branch": "integration/external-agent-autonomous-execution-fabric-v1",
        "protected_paths": [
            "docs/architecture/external_agent_autonomous_execution_fabric/TASK_BOARD.md"
        ],
        "worktree_submodule_paths": [
            "ipfs_datasets_py",
            "ipfs_kit_py",
            "ipfs_accelerate_py/mcplusplus",
        ],
        "bootstrap_runtime_binding": runtime_binding,
        "bootstrap_database_program": {
            "authority_mode": "embedded",
            "task_source_kind": "duckdb",
            "store_id": control,
            "coordination_store_id": f"{prefix}/control.coordination.duckdb",
            "execution_store_id": f"{prefix}/control.execution.duckdb",
            "store_generation": "eaaef-test-run-v1",
            "schema_revision": (
                "datasets-authoritative-eaaef-operational-control-plane@2"
            ),
            "event_store_path": f"{prefix}/events",
            "runtime_registry_path": f"{prefix}/registry",
            "worktree_root": f"{prefix}/worktrees",
            "merge_queue_dir": f"{prefix}/merge-queue",
            "state_dir": f"{prefix}/state",
            "export_profile": "eaaef-test-run-v1",
            "failover_policy": "fail_closed",
            "maximum_writer_processes": 1,
        },
        "database_program": {
            "authority_mode": "quack",
            "task_source_kind": "duckdb",
            "endpoint_secret_handle": "secret-handle:eaaef-test-quack",
            "quack_endpoint": "quack:127.0.0.1:19494",
            "store_id": "eaaef-test-control",
            "store_generation": "eaaef-test-run-v1",
            "schema_revision": (
                "datasets-authoritative-eaaef-operational-control-plane@2"
            ),
            "event_store_path": f"{prefix}/events",
            "runtime_registry_path": f"{prefix}/registry",
            "worktree_root": f"{prefix}/worktrees",
            "export_profile": "eaaef-test-run-v1",
            "failover_policy": "fail_closed",
            "explicit_legacy": False,
        },
        "operational_command_fabric": {
            "schema": materializer.SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA,
            "transport_kind": "signed_command_fabric",
            "board_namespace": "external-agent-autonomous-execution-fabric-v1",
            "shard_id": "control-shard-0",
            "ingress_endpoint": "quack:127.0.0.1:19494",
            "ingress_secret_handle": "secret-handle:eaaef-test-ingress",
            "projection_endpoint": "quack:127.0.0.1:19495",
            "projection_secret_handle": "secret-handle:eaaef-test-projection",
            "store_id": "eaaef-test-control",
            "store_generation": "eaaef-test-run-v1",
            "schema_revision": (
                "datasets-authoritative-eaaef-operational-control-plane@2"
            ),
            "owner_qualification_schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "eaaef-quack-owner-qualification@1"
            ),
            "command_envelope_schema": (
                "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
            ),
            "state_command_schema": (
                "ipfs_accelerate_py/agent-supervisor/state-command@1"
            ),
            "ingress_relation": "command_inbox",
            "ingress_append_only": True,
            "ingress_accepts_signed_envelopes_only": True,
            "operational_database_private": True,
            "operational_tables_remotely_exposed": False,
            "one_mutable_owner": True,
            "owner_verifies_signed_envelopes": True,
            "projection_read_only": True,
            "projection_append_allowed": False,
            "atomic_plan_r2_required": True,
            "direct_file_fallback": False,
            "failover_policy": "fail_closed",
            "child_adapter_status": "implemented_unqualified_fail_closed",
        },
        "container_policy": {
            "live_dispatch_allowed": False,
            "bootstrap_image_status": "not_admitted",
            "bootstrap_image_digest": "",
        },
        "launch_policy": {
            "live_single_supervisor_allowed": False,
            "blockers": ["test no-go"],
        },
    }


def test_runtime_binding_accepts_exact_isolated_interpreter(tmp_path: Path) -> None:
    config = _config()
    binding = config["bootstrap_runtime_binding"]
    assert isinstance(binding, dict)
    launcher = binding["launcher"]
    assert isinstance(launcher, dict)
    argv_prefix = launcher["argv_prefix"]
    assert isinstance(argv_prefix, list)
    result = subprocess.run(
        [*argv_prefix, "runtime-check"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["runtime_binding"] == binding
    assert payload["invocation"]["orig_argv"] == [*argv_prefix, "runtime-check"]

    launch_plan = subprocess.run(
        [*argv_prefix, "launch-plan"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    launch_payload = json.loads(launch_plan.stdout)
    assert "sys.orig_argv differs" not in str(launch_payload)
    if launch_plan.returncode == 0:
        assert launch_payload["execution_prohibited"] is True
        assert launch_payload["process_started"] is False
    else:
        assert launch_payload["valid"] is False

    interpreter = binding["interpreter"]
    assert isinstance(interpreter, dict)

    program = """
import importlib.util
import json
import sys
from pathlib import Path

materializer_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
sys.pycache_prefix = config["bootstrap_runtime_binding"]["interpreter"]["pycache_prefix"]
sys.path.insert(0, sys.argv[3])
if len(sys.argv) > 4:
    sys.path.insert(1, sys.argv[4])
spec = importlib.util.spec_from_file_location("eaaef_isolated_runtime_probe", materializer_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runtime = module._validated_runtime_binding(config)
print(json.dumps(module._validated_runtime_invocation(runtime, "runtime-check"), sort_keys=True))
"""
    bypass = subprocess.run(
        [
            str(interpreter["resolved_path"]),
            "-I",
            "-S",
            "-B",
            "-c",
            program,
            str(MATERIALIZER_PATH),
            str(materializer.CONFIG_PATH),
            str(binding["approved_import_root"]),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bypass.returncode != 0
    assert "sys.orig_argv differs" in bypass.stderr

    extra_site = tmp_path / "other-site-packages"
    extra_site.mkdir()
    rejected_extra_root = subprocess.run(
        [
            str(interpreter["resolved_path"]),
            "-I",
            "-S",
            "-B",
            "-c",
            program,
            str(MATERIALIZER_PATH),
            str(materializer.CONFIG_PATH),
            str(binding["approved_import_root"]),
            str(extra_site),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected_extra_root.returncode != 0
    assert "closed repository/stdlib/import-root projection" in rejected_extra_root.stderr


def test_board_validation_reopens_only_the_admitted_import_root(
    tmp_path: Path,
) -> None:
    config = _config()
    binding = config["bootstrap_runtime_binding"]
    assert isinstance(binding, dict)

    report = materializer._validate_board(binding)

    assert report["valid"] is True
    assert report["live_launch_allowed"] is False

    forged = json.loads(json.dumps(binding, sort_keys=True))
    forged["approved_import_root"] = str(tmp_path / "missing-import-root")
    with pytest.raises(
        materializer.MaterializationError,
        match="approved import root is unavailable",
    ):
        materializer._validate_board(forged)


def test_board_validation_fails_closed_on_a_wedged_sealed_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    binding = config["bootstrap_runtime_binding"]
    assert isinstance(binding, dict)

    def timeout(*args: object, **kwargs: object) -> object:
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(materializer.subprocess, "run", timeout)
    with pytest.raises(
        materializer.MaterializationError,
        match="30-second sealed child deadline",
    ):
        materializer._validate_board(binding)


def test_runtime_binding_rejects_tamper_and_noncanonical_paths() -> None:
    config = _config()
    binding = config["bootstrap_runtime_binding"]
    assert isinstance(binding, dict)
    duckdb_binding = binding["duckdb"]
    assert isinstance(duckdb_binding, dict)
    duckdb_binding["extension_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(materializer.MaterializationError, match="runtime file differs"):
        materializer._runtime_binding_contract(config)

    config = _config()
    binding = config["bootstrap_runtime_binding"]
    assert isinstance(binding, dict)
    binding["approved_import_root"] = str(binding["approved_import_root"]) + "/."
    with pytest.raises(materializer.MaterializationError, match="not a canonical resolved path"):
        materializer._runtime_binding_contract(config)


def test_duckdb_record_member_tamper_is_rejected(tmp_path: Path) -> None:
    member = tmp_path / "duckdb/module.py"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"trusted\n")
    digest = base64.urlsafe_b64encode(hashlib.sha256(member.read_bytes()).digest()).decode(
        "ascii"
    ).rstrip("=")
    record = tmp_path / "duckdb-1.5.2.dist-info/RECORD"
    record.parent.mkdir()
    record.write_text(
        f"duckdb/module.py,sha256={digest},{member.stat().st_size}\n"
        "duckdb-1.5.2.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    projection = materializer._verify_duckdb_record(record, tmp_path)
    assert projection["record_entry_count"] == 2
    assert projection["record_verified_file_count"] == 1

    member.write_bytes(b"tampered\n")
    with pytest.raises(materializer.MaterializationError, match="size/digest"):
        materializer._verify_duckdb_record(record, tmp_path)


def test_nonisolated_runtime_fails_before_namespace_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("state/nonisolated")
    sentinel_claim = tmp_path / "claim-must-not-exist.json"
    monkeypatch.setattr(materializer, "_claim_path", lambda _config: sentinel_claim)
    with pytest.raises(materializer.MaterializationError, match="exact -I -S -B flags"):
        materializer.materialize(config)
    assert not sentinel_claim.exists()


def test_paths_match_supported_database_daemon_sidecars() -> None:
    paths = materializer._paths(_config())
    assert paths["coordination"] == paths["control"].with_name(
        "control.coordination.duckdb"
    )
    assert paths["execution"] == paths["control"].with_name(
        "control.execution.duckdb"
    )


def test_paths_reject_an_invented_sidecar_contract() -> None:
    config = _config()
    config["bootstrap_database_program"]["coordination_store_id"] = (  # type: ignore[index]
        "data/eaaef-test/run-v1/coordination.duckdb"
    )
    with pytest.raises(materializer.MaterializationError, match="deterministic"):
        materializer._paths(config)


def test_immutable_json_publish_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(materializer, "ROOT", tmp_path)
    target = tmp_path / "registry/receipt.json"
    materializer._write_json_immutable(target, {"value": 1})
    first_bytes = target.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(materializer.MaterializationError, match="refusing to overwrite"):
        materializer._write_json_immutable(target, {"value": 2})
    assert target.read_bytes() == first_bytes
    assert json.loads(first_bytes) == {"value": 1}


def test_namespace_freshness_includes_every_runtime_subtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(materializer, "ROOT", tmp_path)
    config = _config("state/run-v1")
    config["bootstrap_database_program"]["state_dir"] = "legacy/state"  # type: ignore[index]
    stale_pid = tmp_path / "legacy/state/eaaef.pid"
    stale_pid.parent.mkdir(parents=True)
    stale_pid.write_text("123\n", encoding="utf-8")

    existing = [path for path in materializer._namespace_artifacts(config) if path.exists()]

    assert tmp_path / "legacy/state" in existing


def test_launch_plan_uses_database_program_cli_and_remains_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    invocation_commands: list[str] = []

    def verified(_config, *, invocation_command="verify"):
        invocation_commands.append(invocation_command)
        return {"receipt_cid": "sha256:" + "1" * 64}

    monkeypatch.setattr(
        materializer,
        "verify",
        verified,
    )
    result = materializer.launch_plan(config)
    assert invocation_commands == ["launch-plan"]
    assert result["allowed"] is False
    assert result["argv"] == []
    assert result["candidate_argv"] == []
    assert result["candidate_executable_withheld"] is True
    assert result["candidate_argv_length"] > 0
    assert result["candidate_argv_cid"].startswith("sha256:")
    assert result["execution_prohibited"] is True
    assert result["process_started"] is False
    assert "signed_command_fabric_child_adapter_unavailable" in result["blockers"]
    assert (
        "worker_network_authorization_propagation_unavailable"
        in result["blockers"]
    )


def test_configured_board_launcher_cannot_bypass_launch_plan() -> None:
    config = _config()
    binding = config["bootstrap_runtime_binding"]
    assert isinstance(binding, dict)
    launcher = binding["launcher"]
    assert isinstance(launcher, dict)
    argv_prefix = launcher["argv_prefix"]
    assert isinstance(argv_prefix, list)

    result = subprocess.run(
        [*argv_prefix, "configured-board-launch"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "configured-board launch is not admitted" in result.stderr


def test_bootstrap_and_operational_database_programs_are_non_conflated() -> None:
    config = _config()
    bindings = materializer._database_program_bindings(config)
    assert bindings["bootstrap"]["authority_mode"] == "embedded"
    assert bindings["operational"]["authority_mode"] == "quack"
    assert bindings["materializer_opens_operational_profile"] is False
    assert bindings["direct_file_fallback"] is False
    assert bindings["bootstrap_profile_cid"] != bindings["operational_profile_cid"]
    assert bindings["operational_command_fabric"]["transport_kind"] == (
        "signed_command_fabric"
    )
    assert bindings["operational_command_fabric"]["board_namespace"] == (
        "external-agent-autonomous-execution-fabric-v1"
    )
    assert bindings["operational_command_fabric"]["shard_id"] == (
        "control-shard-0"
    )
    assert bindings["operational_child_adapter_status"] == (
        "implemented_unqualified_fail_closed"
    )
    assert "/" not in bindings["operational"]["store_id"]


@pytest.mark.parametrize(
    "missing",
    [
        "bootstrap_database_program",
        "database_program",
        "operational_command_fabric",
    ],
)
def test_database_program_role_absence_fails_closed(missing: str) -> None:
    config = _config()
    config.pop(missing)
    with pytest.raises(materializer.MaterializationError, match="required|missing"):
        materializer._database_program_bindings(config)


def test_database_program_role_inversion_and_direct_file_fallback_fail_closed() -> None:
    config = _config()
    for name in (
        "bootstrap_database_program",
        "database_program",
        "operational_command_fabric",
    ):
        profile = config[name]
        assert isinstance(profile, dict)
        profile["schema_revision"] = "datasets-authoritative-operational-v1"
    with pytest.raises(materializer.MaterializationError, match="profile @2"):
        materializer._database_program_bindings(config)

    config = _config()
    config["bootstrap_database_program"], config["database_program"] = (  # type: ignore[misc]
        config["database_program"],
        config["bootstrap_database_program"],
    )
    with pytest.raises(materializer.MaterializationError, match="bootstrap"):
        materializer._database_program_bindings(config)

    config = _config()
    operational = config["database_program"]
    assert isinstance(operational, dict)
    operational["store_id"] = "data/eaaef-test/control.duckdb"
    with pytest.raises(materializer.MaterializationError, match="direct-file fallback"):
        materializer._database_program_bindings(config)


def test_launch_plan_attaches_unsigned_no_go_admission_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    statement = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-admission-statement@1",
        "decision": "no_go",
        "blockers": ["worker_network_runtime_principals_unavailable"],
    }

    monkeypatch.setattr(
        materializer,
        "verify",
        lambda _config, **_kwargs: {"receipt_cid": "sha256:" + "3" * 64},
    )
    monkeypatch.setattr(
        materializer,
        "_unsigned_bootstrap_admission_statement",
        lambda _config, **_kwargs: statement,
    )
    result = materializer.launch_plan(config)
    assert result["allowed"] is False
    assert result["process_started"] is False
    assert result["bootstrap_admission_published"] is False
    assert result["bootstrap_admission_statement"] == statement
    assert "worker_network_runtime_principals_unavailable" in result["blockers"]


def test_launch_plan_emits_typed_no_go_when_verify_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()

    def rejected(_config, *, invocation_command="verify"):
        assert invocation_command == "launch-plan"
        raise materializer.MaterializationError(
            "ipfs_accelerate_py nested checkout is dirty"
        )

    monkeypatch.setattr(materializer, "verify", rejected)
    result = materializer.launch_plan(config)
    assert result["schema"].endswith("eaaef-launch-plan@2")
    assert result["allowed"] is False
    assert result["argv"] == []
    assert result["candidate_executable_withheld"] is True
    assert result["execution_prohibited"] is True
    assert result["process_started"] is False
    assert result["materialization_receipt_cid"] == ""
    assert "ipfs_accelerate_py nested checkout is dirty" in result["blockers"]
    assert "signed_command_fabric_child_adapter_unavailable" in result["blockers"]
    assert (
        "worker_network_authorization_propagation_unavailable"
        in result["blockers"]
    )
    assert "board validation has not admitted live launch" in result["blockers"]
    assert "container_policy.live_dispatch_allowed is not true" in result["blockers"]


def test_launch_plan_cannot_be_enabled_while_container_is_unadmitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["launch_policy"] = {
        "live_single_supervisor_allowed": True,
        "blockers": [],
    }
    monkeypatch.setattr(
        materializer,
        "verify",
        lambda _config, **_kwargs: {"receipt_cid": "sha256:" + "2" * 64},
    )
    result = materializer.launch_plan(config)
    assert result["allowed"] is False
    assert any("container_policy" in blocker for blocker in result["blockers"])


def test_expected_population_resolves_native_dependency_aliases_to_cids() -> None:
    first_cid = "sha256:" + "a" * 64
    second_cid = "sha256:" + "b" * 64
    projection = materializer._expected_population_projection(
        {
            "repository_tree_id": "1" * 40,
            "plan_root_cid": "sha256:" + "c" * 64,
            "objectives": [],
            "plans": [],
            "tasks": [
                {
                    "task_cid": first_cid,
                    "task_id": "EAAEF-000",
                    "task_alias": "EAAEF-000",
                    "goal_cid": "goal:root",
                    "depends_on": [],
                },
                {
                    "task_cid": second_cid,
                    "task_id": "EAAEF-001",
                    "task_alias": "EAAEF-001",
                    "goal_cid": "goal:root",
                    "depends_on": ["EAAEF-000"],
                    "dependencies": [first_cid],
                },
            ],
        }
    )

    assert projection["dependencies"] == [
        {
            "task_cid": second_cid,
            "dependency_task_cid": first_cid,
            "kind": "depends_on",
        }
    ]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_isolated_materialization_is_sealed_idempotent_and_read_only_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("state/run-v1")
    config["initial_projection"] = {"ready_task_ids": ["EAAEF-001"]}
    plan_cid = "sha256:" + "a" * 64
    goal_cid = "sha256:" + "b" * 64
    task_cid = "sha256:" + "c" * 64
    source_generation = {
        "ipfs_accelerate_py": {
            "head": "1" * 40,
            "tree": "2" * 40,
            "required_integration_head": "1" * 40,
            "required_integration_tree": "2" * 40,
        },
        "planning_source_forest_root": "sha256:" + "3" * 64,
    }
    source_generation["source_generation_cid"] = materializer._cid(source_generation)
    population = {
        "schema": materializer.POPULATION_SCHEMA,
        "repository_tree_id": "2" * 40,
        "source_head": "1" * 40,
        "source_generation": source_generation,
        "plan_root_cid": plan_cid,
        "controls": {"board": "sha256:" + "4" * 64},
        "objectives": [
            {
                "goal_cid": goal_cid,
                "goal_id": "EAAEF-G000",
                "goal_alias": "EAAEF-G000",
                "title": "Root",
                "ordinal": 1,
                "status": "open",
                "objective_id": "objective:eaaef-root",
                "objective_alias": "EAAEF-G000",
                "parent_goal_cid": "",
                "priority": "P0",
                "body": {},
            }
        ],
        "goal_edges": [],
        "plans": [
            {
                "plan_cid": plan_cid,
                "plan_alias": "EAAEF-PLAN-R1",
                "goal_cid": goal_cid,
                "status": "active",
                "body": {},
            }
        ],
        "tasks": [
            {
                "task_cid": task_cid,
                "task_id": "EAAEF-001",
                "task_alias": "EAAEF-001",
                "goal_cid": goal_cid,
                "plan_cid": plan_cid,
                "ordinal": 1,
                "status": "todo",
                "priority": "P0",
                "title": "Bootstrap",
                "depends_on": [],
                "dependencies": [],
                "outputs": [
                    {
                        "path": "test/api/test_bootstrap.py",
                        "effect_id": "effect:eaaef-bootstrap-test",
                    }
                ],
                "acceptance": ["receipt"],
                "validations": [
                    {
                        "working_directory": ".",
                        "argv": [
                            "python3",
                            "-m",
                            "pytest",
                            "-q",
                            "test/api/test_bootstrap.py",
                        ],
                    }
                ],
                "execution_owned_files": ["test/api/test_bootstrap.py"],
                "execution_validation": [
                    {
                        "working_directory": ".",
                        "argv": [
                            "python3",
                            "-m",
                            "pytest",
                            "-q",
                            "test/api/test_bootstrap.py",
                        ],
                    }
                ],
            }
        ],
        "task_cids_by_alias": {"EAAEF-001": task_cid},
        "goal_cids_by_alias": {"EAAEF-G000": goal_cid},
        "initial_task_aliases": ["EAAEF-001"],
        "ready_task_aliases": ["EAAEF-001"],
        "initial_task_count": 1,
        "goal_count": 1,
        "future_task_count": 0,
    }
    population["population_cid"] = materializer._cid(population)
    monkeypatch.setattr(materializer, "ROOT", tmp_path)
    monkeypatch.setattr(materializer, "_assert_clean", lambda: None)
    monkeypatch.setattr(
        materializer,
        "_validate_board",
        lambda _runtime_binding: {"valid": True, "schema": "test-validation@1"},
    )
    monkeypatch.setattr(materializer, "build_population", lambda _config: population)
    runtime_binding = json.loads(
        json.dumps(config["bootstrap_runtime_binding"], sort_keys=True)
    )
    monkeypatch.setattr(
        materializer,
        "_validated_runtime_binding",
        lambda _config: json.loads(json.dumps(runtime_binding, sort_keys=True)),
    )

    def test_invocation(_runtime_binding, command):
        value = {
            "schema": materializer.RUNTIME_INVOCATION_SCHEMA,
            "command": command,
            "orig_argv": ["test-launcher", command],
            "materializer_argv": ["test-materializer", command],
            "launcher_path": "test-launcher",
            "launcher_sha256": "sha256:" + "d" * 64,
        }
        value["invocation_cid"] = materializer._cid(value)
        return value

    monkeypatch.setattr(materializer, "_validated_runtime_invocation", test_invocation)
    monkeypatch.setattr(materializer, "_runtime_invocation_projection", test_invocation)

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    native_materialize_population = DatabaseImplementationDaemon.materialize_population

    def forged_materialize_population(self, *args, **kwargs):
        forged = dict(native_materialize_population(self, *args, **kwargs))
        forged_task_source = dict(forged["task_source"])
        forged_task_source["task_count"] = 999
        forged_task_source["plan_root_cid"] = "sha256:" + "e" * 64
        forged["task_source"] = forged_task_source
        return forged

    monkeypatch.setattr(
        DatabaseImplementationDaemon,
        "materialize_population",
        forged_materialize_population,
    )
    forged_config = _config("state/forged-native-receipt")
    with pytest.raises(
        materializer.MaterializationError,
        match="database materialization receipt differs",
    ):
        materializer.materialize(forged_config)
    assert materializer._claim_path(forged_config).is_file()
    assert not materializer._receipt_path(forged_config).exists()
    monkeypatch.setattr(
        DatabaseImplementationDaemon,
        "materialize_population",
        native_materialize_population,
    )

    receipt = materializer.materialize(config)
    assert receipt["process_started"] is False
    assert receipt["schema_install"]["changed"] is True
    assert {
        item["tool_version"] for item in receipt["schema_install"]["receipts"]
    } == {runtime_binding["duckdb"]["module_version"]}
    assert receipt["operational_profile_verification"]["valid"] is True
    assert (
        receipt["operation_vocabulary_cid"]
        == receipt["operational_profile_verification"]["operation_vocabulary_cid"]
    )
    handler_evidence = receipt["borrowed_transaction_handler_source_evidence"]
    assert handler_evidence["operation_count"] == 31
    assert handler_evidence["production_admitted"] is False
    assert "command_principal_did" in handler_evidence["runtime_authority_fields"]
    assert receipt["control_schema_projection"]["connection_mode"] == "read_only"
    assert receipt["runtime_binding"] == runtime_binding
    assert receipt["runtime_binding_cid"] == materializer._cid(runtime_binding)
    claim = materializer._load_object(materializer._claim_path(config))
    assert claim["runtime_binding"] == runtime_binding
    assert claim["runtime_binding_cid"] == materializer._cid(runtime_binding)
    forged_initial_projection = json.loads(
        json.dumps(receipt["control_projection"], sort_keys=True)
    )
    forged_initial_projection["task_outputs"][0]["path"] = "evil-before-seal.py"
    with pytest.raises(
        materializer.MaterializationError,
        match="differs from the admitted board projection",
    ):
        materializer._assert_population_equivalent(
            population, forged_initial_projection
        )
    forged_initial_projection = json.loads(
        json.dumps(receipt["control_projection"], sort_keys=True)
    )
    forged_initial_projection["goals"][0]["title"] = "evil-before-seal"
    with pytest.raises(
        materializer.MaterializationError,
        match="differs from the admitted board projection",
    ):
        materializer._assert_population_equivalent(
            population, forged_initial_projection
        )
    forged_initial_projection = json.loads(
        json.dumps(receipt["control_projection"], sort_keys=True)
    )
    forged_initial_projection["task_revisions"][0]["revision"] = 9
    with pytest.raises(
        materializer.MaterializationError,
        match="differs from the admitted board projection",
    ):
        materializer._assert_population_equivalent(
            population, forged_initial_projection
        )

    def namespace_snapshot() -> dict[str, tuple[int, int, str]]:
        return {
            path.relative_to(tmp_path).as_posix(): (
                path.stat().st_mtime_ns,
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    before_verify = namespace_snapshot()
    verification = materializer.verify(config)
    assert verification["verification_mode"] == "read_only"
    assert (
        verification["operational_profile_verification_cid"]
        == receipt["operational_profile_verification"]["verification_cid"]
    )
    assert (
        verification["borrowed_transaction_handler_source_evidence_cid"]
        == handler_evidence["handler_source_evidence_cid"]
    )
    assert namespace_snapshot() == before_verify

    receipt_path = materializer._receipt_path(config)
    receipt_bytes = receipt_path.read_bytes()
    forged_receipt = json.loads(receipt_bytes)
    forged_receipt["runtime_binding"]["duckdb"]["module_version"] = "0.0.0"
    forged_receipt.pop("receipt_cid")
    forged_receipt["receipt_cid"] = materializer._cid(forged_receipt)
    receipt_path.write_text(
        json.dumps(forged_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(materializer.MaterializationError, match="runtime_binding"):
        materializer.verify(config)
    receipt_path.write_bytes(receipt_bytes)

    forged_receipt = json.loads(receipt_bytes)
    forged_receipt["operational_profile_verification"]["schema_fingerprint"] = (
        "sha256:" + "0" * 64
    )
    forged_receipt.pop("receipt_cid")
    forged_receipt["receipt_cid"] = materializer._cid(forged_receipt)
    receipt_path.write_text(
        json.dumps(forged_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="operational profile .* differs",
    ):
        materializer.verify(config)
    receipt_path.write_bytes(receipt_bytes)

    forged_receipt = json.loads(receipt_bytes)
    forged_receipt["borrowed_transaction_handler_source_evidence"][
        "handler_source_evidence_cid"
    ] = "sha256:" + "1" * 64
    forged_receipt.pop("receipt_cid")
    forged_receipt["receipt_cid"] = materializer._cid(forged_receipt)
    receipt_path.write_text(
        json.dumps(forged_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="handler source differs",
    ):
        materializer.verify(config)
    receipt_path.write_bytes(receipt_bytes)

    with pytest.raises(materializer.MaterializationError, match="refusing to overwrite"):
        materializer.materialize(config)

    import duckdb

    control = materializer._paths(config)["control"]
    connection = duckdb.connect(str(control))
    try:
        connection.execute(
            "UPDATE task_outputs SET path = 'evil.py' WHERE task_cid = ?",
            [task_cid],
        )
    finally:
        connection.close()
    with pytest.raises(materializer.MaterializationError, match="control authority differs"):
        materializer.verify(config)

    connection = duckdb.connect(str(control))
    try:
        connection.execute(
            "UPDATE task_outputs SET path = 'test/api/test_bootstrap.py' WHERE task_cid = ?",
            [task_cid],
        )
        connection.execute(
            "UPDATE goals SET title = 'forged goal' WHERE goal_cid = ?",
            [goal_cid],
        )
    finally:
        connection.close()
    with pytest.raises(materializer.MaterializationError, match="control authority differs"):
        materializer.verify(config)

    connection = duckdb.connect(str(control))
    try:
        connection.execute("UPDATE goals SET title = 'Root' WHERE goal_cid = ?", [goal_cid])
        connection.execute(
            "UPDATE task_acceptance SET criterion = 'forged acceptance' WHERE task_cid = ?",
            [task_cid],
        )
    finally:
        connection.close()
    with pytest.raises(materializer.MaterializationError, match="control authority differs"):
        materializer.verify(config)

    connection = duckdb.connect(str(control))
    try:
        connection.execute(
            "UPDATE task_acceptance SET criterion = 'receipt' WHERE task_cid = ?",
            [task_cid],
        )
        connection.execute(
            "UPDATE task_validations SET argv_json = '[\"false\"]' WHERE task_cid = ?",
            [task_cid],
        )
    finally:
        connection.close()
    with pytest.raises(materializer.MaterializationError, match="control authority differs"):
        materializer.verify(config)


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_control_schema_projection_is_byte_stable_and_read_only(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    install_eaaef_operational_schema(
        database,
        application_version="test",
        tool_version="test",
        owner_id="eaaef-materializer-test",
    )
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    projection = materializer._control_schema_projection(database)
    after = hashlib.sha256(database.read_bytes()).hexdigest()
    assert projection["valid"] is True
    assert projection["connection_mode"] == "read_only"
    assert after == before
    assert not Path(f"{database}.wal").exists()
