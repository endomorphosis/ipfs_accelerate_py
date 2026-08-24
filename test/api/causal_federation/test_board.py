"""Hermetic contracts for the sealed CASF planning/control board."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    materialize_task_dependency_dag,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    configured_board_common_args,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_text,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "scripts/validate_agent_supervisor_causal_event_federation_board.py"
CONFIG = REPO_ROOT / "config/agent_supervisor_causal_event_federation_scheduler.json"
TODO = REPO_ROOT / "docs/architecture/agent_supervisor_causal_event_federation.todo.md"
OBJECTIVES = REPO_ROOT / "docs/architecture/agent_supervisor_causal_event_federation.objectives.md"
EXPECTED_ROOT = "sha256:29be8bb6fbc6f37352ee7a312b2ecf87c15d575624ebcfd7bbc971968d688ecb"


def _validator_module():
    spec = importlib.util.spec_from_file_location("casf_board_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_reports_exact_sealed_program() -> None:
    report = _validator_module().validate_program()
    assert report["valid"] is True, report["errors"]
    assert report["program_identifier"] == "agent-supervisor-causal-event-federation-v1"
    assert report["root_objective_id"] == "CASF-G000"
    assert report["task_prefix"] == "CASF-"
    assert report["task_count"] == 44
    assert report["goal_count"] == 17
    assert report["task_dependency_count"] == 191
    assert report["task_dependency_root_cid"] == EXPECTED_ROOT
    assert report["completed_task_ids"] == []
    assert report["blocked_task_ids"] == []
    assert report["initial_ready_task_ids"] == ["CASF-000"]
    assert report["terminal_task_id"] == "CASF-043"
    assert report["task_waves"][0] == {"id": "W0", "task_ids": ["CASF-000"]}
    assert report["task_waves"][-1] == {"id": "W29", "task_ids": ["CASF-043"]}
    assert report["high_concurrency_gate_open"] is False


def test_validator_cli_emits_one_json_object() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True, payload["errors"]
    assert payload["source_checks_performed"] is False
    assert payload["database_presence_checked"] is False
    assert completed.stderr == ""


def test_inventory_only_validation_seals_baseline_artifacts() -> None:
    report = _validator_module().validate_program(inventory_only=True)
    assert report["valid"] is True, report["errors"]
    assert report["inventory_only"] is True
    assert set(report["inventory_artifact_identities"]) == {
        "docs/architecture/causal_event_federation_inventory/README.md",
        "docs/architecture/causal_event_federation_inventory/authorities.json",
        "docs/architecture/causal_event_federation_inventory/capability_snapshot.json",
        "docs/architecture/causal_event_federation_inventory/starting_tree.json",
    }


def test_production_parsers_preserve_goal_and_task_dags() -> None:
    tasks = parse_task_text(
        TODO.read_text(encoding="utf-8"),
        path=TODO,
        task_header_prefix="## CASF-",
    )
    goals = parse_goal_heap(OBJECTIVES.read_text(encoding="utf-8"))
    graph = materialize_task_dependency_dag(tasks)
    assert [task.task_id for task in tasks] == [f"CASF-{index:03d}" for index in range(44)]
    assert all(task.status == "todo" for task in tasks)
    assert [goal.goal_id for goal in goals] == [
        "CASF-G000",
        "CASF-G010",
        "CASF-G011",
        "CASF-G012",
        "CASF-G013",
        "CASF-G020",
        "CASF-G021",
        "CASF-G022",
        "CASF-G023",
        "CASF-G030",
        "CASF-G031",
        "CASF-G032",
        "CASF-G033",
        "CASF-G040",
        "CASF-G041",
        "CASF-G042",
        "CASF-G043",
    ]
    assert graph.invalid_task_cids == []
    assert len(graph.nodes) == 44
    assert len(graph.edges) == 191
    by_id = {task.task_id: task for task in tasks}
    assert all(
        by_id[f"CASF-{index:03d}"].metadata.get("no-change completion") == "allowed"
        for index in range(41)
    )
    assert all(
        by_id[f"CASF-{index:03d}"].metadata.get("no-change completion") is None
        for index in range(41, 44)
    )


def test_generic_scheduler_accepts_one_lane_quack_program() -> None:
    configured = load_configured_board(CONFIG, repo_root=REPO_ROOT)
    program = configured.resolved_database_program()
    args = configured_board_common_args(configured, implement=True)
    assert configured.board_namespace == "agent-supervisor-causal-event-federation-v1"
    assert configured.merge_target_branch == "codex/causal-event-supervisor-federation-v1"
    assert configured.max_lanes == 1
    assert configured.idle_lane_work_stealing == ""
    assert "--idle-lane-work-stealing" not in args
    assert program.authority_mode == "quack"
    assert program.task_source_kind == "duckdb"
    assert program.endpoint_secret_handle == "handle:casf-v1"
    assert program.quack_endpoint == "quack:127.0.0.1:41417"
    assert program.store_generation == "casf-v1"
    assert program.schema_revision == "3"
    assert args.count("--state-schema-revision") == 1
    revision_index = args.index("--state-schema-revision")
    assert args[revision_index + 1] == "3"
    assert program.failover_policy == "fail_closed"


def test_authority_roles_endpoint_and_capacity_are_fail_closed() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    database = payload["database_program"]
    control = payload["operational_control_plane"]
    projection = payload["ducklake_projection_program"]
    endpoint = payload["endpoint_allocation_policy"]
    gate = payload["high_concurrency_gate"]
    assert database["authority_mode"] == "quack"
    assert database["task_source_kind"] == "duckdb"
    assert control["direct_multi_process_duckdb_file_open_permitted"] is False
    assert control["automatic_file_fallback_permitted"] is False
    assert control["arbitrary_sql_from_agents_permitted"] is False
    assert projection["authority"] is False
    assert projection["scheduling_prerequisite"] is False
    assert projection["lease_prerequisite"] is False
    assert projection["policy_prerequisite"] is False
    assert projection["completion_prerequisite"] is False
    assert projection["may_grant_authority"] is False
    assert endpoint["host"] == "127.0.0.1"
    assert endpoint["port"] == 41417
    assert endpoint["mandatory_prelaunch_recheck"] is True
    assert endpoint["collision_disposition"].startswith("fail_closed")
    assert endpoint["agent_supplied_endpoint_permitted"] is False
    assert payload["max_lanes"] == 1
    assert payload["bootstrap_capacity"] == {
        "supervisors": 1,
        "registered_logical_subagents": 1,
        "maximum_active_subagents": 1,
        "lanes": 1,
        "provider_concurrency": 1,
    }
    assert gate["enabled_at_bootstrap"] is False
    assert gate["required_accepted_task_ids"] == [
        "CASF-005",
        "CASF-009",
        "CASF-010",
        "CASF-016",
        "CASF-024",
        "CASF-029",
    ]
    assert gate["target_profile"] == {
        "supervisors": 12,
        "registered_logical_subagents": 256,
        "maximum_active_subagents": 64,
    }


def test_bootstrap_qualifies_only_exact_typed_wait_without_federation_claim() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    wait = payload["event_wait_policy"]
    assert wait["bootstrap_qualification"] == "CASF-010_server_owned_typed_wait_qualified"
    assert wait["server_owned_typed_wait_qualified"] is True
    assert wait["federation_event_driven_execution_qualified"] is False
    assert wait["federation_event_driven_execution_gate"] == "unavailable_until_CASF-021"
    assert wait["compatibility_polling_for_qualified_mode"] is False
    assert wait["busy_loop_permitted"] is False
    assert wait["idle_full_board_scan_permitted"] is False
    assert wait["idle_model_call_permitted"] is False
    assert wait["idle_context_rebuild_permitted"] is False
    assert wait["idle_unchanged_write_permitted"] is False
    assert wait["event_driven_claim_permitted_at_bootstrap"] is False
    assert payload["promotion_claims"] == {
        "high_concurrency_qualified": False,
        "multi_supervisor_qualified": False,
        "parallel_execution_qualified": False,
        "causal_coordination_qualified": False,
        "production_ready": False,
        "ducklake_projection_promoted": False,
    }


def test_source_check_binds_branch_and_starting_tree() -> None:
    report = _validator_module().validate_program(check_source=True)
    assert report["valid"] is True, report["errors"]
    assert report["source_checks_performed"] is True


def test_validator_rejects_missing_required_task_declaration(tmp_path, monkeypatch) -> None:
    validator = _validator_module()
    corrupted = tmp_path / "board.todo.md"
    corrupted.write_text(
        TODO.read_text(encoding="utf-8").replace(
            "- Database migrations:", "- Database migrationz:", 1
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "BOARD", corrupted)
    report = validator.validate_program()
    assert report["valid"] is False
    assert any("missing fields" in error and "CASF-000" in error for error in report["errors"])


def test_validator_rejects_completed_task_with_pending_result_identity(
    tmp_path, monkeypatch
) -> None:
    validator = _validator_module()
    corrupted = tmp_path / "board.todo.md"
    corrupted.write_text(
        TODO.read_text(encoding="utf-8").replace(
            "- Status: todo", "- Status: completed", 1
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "BOARD", corrupted)
    report = validator.validate_program()
    assert report["valid"] is False
    assert report["completed_task_ids"] == ["CASF-000"]
    assert any(
        "pending final result identity cannot materialize as completed" in error
        for error in report["errors"]
    )


def test_validator_rejects_no_change_population_drift(tmp_path, monkeypatch) -> None:
    validator = _validator_module()
    original = TODO.read_text(encoding="utf-8")

    missing = tmp_path / "missing-no-change.todo.md"
    missing.write_text(
        original.replace("- No-change completion: allowed\n", "", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "BOARD", missing)
    report = validator.validate_program()
    assert report["valid"] is False
    assert any(
        "CASF-000: sealed landed task must allow exact validated no-change completion"
        in error
        for error in report["errors"]
    )

    broadened = tmp_path / "broadened-no-change.todo.md"
    heading = "## CASF-041 Build cross-supervisor token-efficiency benchmark"
    prefix, suffix = original.split(heading, 1)
    suffix = suffix.replace(
        "- Completion: auto\n",
        "- Completion: auto\n- No-change completion: allowed\n",
        1,
    )
    broadened.write_text(prefix + heading + suffix, encoding="utf-8")
    monkeypatch.setattr(validator, "BOARD", broadened)
    report = validator.validate_program()
    assert report["valid"] is False
    assert any(
        "CASF-041: unlanded task must remain outside no-change completion" in error
        for error in report["errors"]
    )


def test_validator_rejects_noncanonical_directory_shaped_task_paths(
    tmp_path, monkeypatch
) -> None:
    validator = _validator_module()
    canonical = "ipfs_accelerate_py/agent_supervisor/federation/formal"
    original = TODO.read_text(encoding="utf-8")
    assert original.count(canonical) == 3
    assert f"{canonical}/" not in original

    corrupted = tmp_path / "trailing-slash.todo.md"
    corrupted.write_text(original.replace(canonical, f"{canonical}/"), encoding="utf-8")
    monkeypatch.setattr(validator, "BOARD", corrupted)
    report = validator.validate_program()

    assert report["valid"] is False
    for field in ("Owned paths", "Predicted files", "Outputs"):
        assert any(
            error
            == f"CASF-036: unsafe {field} path '{canonical}/'"
            for error in report["errors"]
        )


def test_validator_rejects_ducklake_authority(tmp_path, monkeypatch) -> None:
    validator = _validator_module()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["ducklake_projection_program"]["authority"] = True
    corrupted = tmp_path / "scheduler.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(validator, "CONFIG", corrupted)
    report = validator.validate_program()
    assert report["valid"] is False
    assert any("DuckLake authority must be false" in error for error in report["errors"])


def test_validator_rejects_premature_concurrency(tmp_path, monkeypatch) -> None:
    validator = _validator_module()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["max_lanes"] = 12
    payload["high_concurrency_gate"]["enabled_at_bootstrap"] = True
    corrupted = tmp_path / "scheduler.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(validator, "CONFIG", corrupted)
    report = validator.validate_program()
    assert report["valid"] is False
    assert any("bootstrap must use one strict lane" in error for error in report["errors"])
    assert any("high-concurrency gate mismatch" in error for error in report["errors"])


def test_required_inventory_artifacts_fail_if_they_disappear(tmp_path, monkeypatch) -> None:
    validator = _validator_module()
    empty_inventory = tmp_path / "inventory"
    empty_inventory.mkdir()
    monkeypatch.setattr(validator, "INVENTORY", empty_inventory)
    report = validator.validate_program()
    assert report["valid"] is False
    assert report["completed_task_ids"] == []
    assert any("starting_tree.json" in error for error in report["errors"])


def test_inventory_only_rejects_baseline_drift_and_symlink_substitution(
    tmp_path, monkeypatch
) -> None:
    source = REPO_ROOT / "docs/architecture/causal_event_federation_inventory"

    drift_root = tmp_path / "drift-root"
    drift_inventory = drift_root / "inventory"
    shutil.copytree(source, drift_inventory)
    authorities = drift_inventory / "authorities.json"
    payload = json.loads(authorities.read_text(encoding="utf-8"))
    payload["starting_commit"] = "0" * 40
    authorities.write_text(json.dumps(payload), encoding="utf-8")

    validator = _validator_module()
    monkeypatch.setattr(validator, "ROOT", drift_root)
    monkeypatch.setattr(validator, "INVENTORY", drift_inventory)
    report = validator.validate_program(inventory_only=True)
    assert report["valid"] is False
    assert "authorities.json: sealed repository baseline mismatch" in report["errors"]

    symlink_root = tmp_path / "symlink-root"
    symlink_inventory = symlink_root / "inventory"
    shutil.copytree(source, symlink_inventory)
    capability = symlink_inventory / "capability_snapshot.json"
    capability.unlink()
    capability.symlink_to(source / "capability_snapshot.json")

    validator = _validator_module()
    monkeypatch.setattr(validator, "ROOT", symlink_root)
    monkeypatch.setattr(validator, "INVENTORY", symlink_inventory)
    report = validator.validate_program(inventory_only=True)
    assert report["valid"] is False
    assert (
        "capability_snapshot.json: inventory artifact must be a regular file"
        in report["errors"]
    )
