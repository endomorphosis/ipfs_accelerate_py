"""Focused fail-closed tests for the EAAEF bootstrap materializer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_datasets_authoritative_operational_schema,
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
    return {
        "schema": materializer.SCHEDULER_CONFIG_SCHEMA,
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
        "database_program": {
            "authority_mode": "embedded",
            "task_source_kind": "duckdb",
            "store_id": control,
            "coordination_store_id": f"{prefix}/control.coordination.duckdb",
            "execution_store_id": f"{prefix}/control.execution.duckdb",
            "store_generation": "eaaef-test-run-v1",
            "schema_revision": "datasets-authoritative-operational-v1",
            "event_store_path": f"{prefix}/events",
            "runtime_registry_path": f"{prefix}/registry",
            "worktree_root": f"{prefix}/worktrees",
            "merge_queue_dir": f"{prefix}/merge-queue",
            "state_dir": f"{prefix}/state",
            "export_profile": "eaaef-test-run-v1",
            "failover_policy": "fail_closed",
            "maximum_writer_processes": 1,
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
    config["database_program"]["coordination_store_id"] = (  # type: ignore[index]
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
    config["database_program"]["state_dir"] = "legacy/state"  # type: ignore[index]
    stale_pid = tmp_path / "legacy/state/eaaef.pid"
    stale_pid.parent.mkdir(parents=True)
    stale_pid.write_text("123\n", encoding="utf-8")

    existing = [path for path in materializer._namespace_artifacts(config) if path.exists()]

    assert tmp_path / "legacy/state" in existing


def test_launch_plan_uses_database_program_cli_and_remains_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setattr(
        materializer,
        "verify",
        lambda _config: {"receipt_cid": "sha256:" + "1" * 64},
    )
    result = materializer.launch_plan(config)
    assert result["allowed"] is False
    assert result["argv"] == []
    assert result["candidate_argv"] == []
    assert result["candidate_executable_withheld"] is True
    assert result["candidate_argv_length"] > 0
    assert result["candidate_argv_cid"].startswith("sha256:")
    assert result["execution_prohibited"] is True
    assert result["process_started"] is False


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
        lambda _config: {"receipt_cid": "sha256:" + "2" * 64},
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
        lambda: {"valid": True, "schema": "test-validation@1"},
    )
    monkeypatch.setattr(materializer, "build_population", lambda _config: population)

    receipt = materializer.materialize(config)
    assert receipt["process_started"] is False
    assert receipt["schema_install"]["changed"] is True
    assert receipt["control_schema_projection"]["connection_mode"] == "read_only"
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
    assert materializer.verify(config)["verification_mode"] == "read_only"
    assert namespace_snapshot() == before_verify
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
    install_datasets_authoritative_operational_schema(
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
