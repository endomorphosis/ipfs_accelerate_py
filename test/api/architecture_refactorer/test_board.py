"""Hermetic contracts for the PCAR goals, task DAG, and authority split."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    materialize_task_dependency_dag,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_text,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "scripts/validate_agent_supervisor_architecture_refactorer_board.py"
CONFIG = REPO_ROOT / "config/agent_supervisor_architecture_refactorer_scheduler.json"
TODO = REPO_ROOT / "docs/architecture/agent_supervisor_architecture_refactorer.todo.md"
OBJECTIVES = REPO_ROOT / "docs/architecture/agent_supervisor_architecture_refactorer.objectives.md"
BENCHMARK = REPO_ROOT / "benchmarks/agent_supervisor/architecture_refactorer/manifest.json"


def _validator_module():
    spec = importlib.util.spec_from_file_location("pcar_board_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_reports_exact_unblocked_program() -> None:
    report = _validator_module().validate_program()
    assert report["valid"] is True, report["errors"]
    assert report["program_identifier"] == (
        "agent-supervisor-proof-carrying-architecture-refactorer-v1"
    )
    assert report["task_count"] == 32
    assert report["goal_count"] == 14
    assert report["task_dependency_count"] == 119
    assert report["task_dependency_root_cid"] == (
        "sha256:9670f3b2ff880b197d2aea47ce341273778be6930f698a5fc9896901bb171ef5"
    )
    assert report["completed_task_ids"] == []
    assert report["blocked_task_ids"] == []
    assert report["initial_ready_task_ids"] == ["PCAR-000"]
    assert report["terminal_task_id"] == "PCAR-031"
    assert report["task_waves"][0] == {"id": "W0", "task_ids": ["PCAR-000"]}
    assert report["task_waves"][-1] == {"id": "W19", "task_ids": ["PCAR-031"]}


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


def test_generic_scheduler_accepts_quack_database_program() -> None:
    configured = load_configured_board(CONFIG, repo_root=REPO_ROOT)
    program = configured.resolved_database_program()
    assert configured.merge_target_branch == (
        "codex/proof-carrying-architecture-refactorer-v1"
    )
    assert configured.max_lanes == 3
    assert program.authority_mode == "quack"
    assert program.task_source_kind == "duckdb"
    assert program.endpoint_secret_handle == "handle:pcar-v1"
    assert program.quack_endpoint == "quack:127.0.0.1:41317"
    assert program.store_generation == "pcar-v1"
    assert program.schema_revision == "1"
    assert program.failover_policy == "fail_closed"


def test_production_parsers_preserve_goal_and_task_dags() -> None:
    tasks = parse_task_text(
        TODO.read_text(encoding="utf-8"),
        path=TODO,
        task_header_prefix="## PCAR-",
    )
    goals = parse_goal_heap(OBJECTIVES.read_text(encoding="utf-8"))
    graph = materialize_task_dependency_dag(tasks)
    assert [task.task_id for task in tasks] == [
        f"PCAR-{index:03d}" for index in range(32)
    ]
    assert [goal.goal_id for goal in goals] == [
        "PCAR-G000",
        "PCAR-G010",
        "PCAR-G011",
        "PCAR-G012",
        "PCAR-G013",
        "PCAR-G020",
        "PCAR-G021",
        "PCAR-G022",
        "PCAR-G030",
        "PCAR-G031",
        "PCAR-G032",
        "PCAR-G040",
        "PCAR-G041",
        "PCAR-G042",
    ]
    assert graph.invalid_task_cids == []
    assert len(graph.nodes) == 32
    assert len(graph.edges) == 119


def test_authority_roles_are_unequal_and_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    database = config["database_program"]
    projection = config["ducklake_projection_program"]
    policy = config["authority_policy"]
    assert database["authority_mode"] == "quack"
    assert database["task_source_kind"] == "duckdb"
    assert policy["duckdb_transactional_authority"] is True
    assert policy["quack_exclusive_state_owner_transport"] is True
    assert policy["automatic_quack_to_file_fallback"] is False
    assert projection["mode"] == "enabled_non_authoritative"
    assert projection["authority"] is False
    assert projection["scheduling_prerequisite"] is False
    assert projection["acceptance_prerequisite"] is False
    assert projection["completion_prerequisite"] is False
    assert projection["may_grant_authority"] is False


def test_validator_rejects_blocked_initial_task(tmp_path, monkeypatch) -> None:
    validator = _validator_module()
    corrupted = tmp_path / "board.todo.md"
    corrupted.write_text(
        TODO.read_text(encoding="utf-8").replace(
            "- Status: todo", "- Status: blocked", 1
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validator, "BOARD", corrupted)
    report = validator.validate_program()
    assert report["valid"] is False
    assert "PCAR-000" in report["blocked_task_ids"]
    assert any("initial status must be todo" in error for error in report["errors"])


def test_validator_rejects_ducklake_authority(tmp_path, monkeypatch) -> None:
    validator = _validator_module()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["ducklake_projection_program"]["authority"] = True
    corrupted = tmp_path / "scheduler.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(validator, "CONFIG", corrupted)
    report = validator.validate_program()
    assert report["valid"] is False
    assert any(
        "DuckLake projection authority must be false" in error
        for error in report["errors"]
    )


def test_frozen_benchmark_covers_required_domains_and_task_types() -> None:
    manifest = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    cases_path = REPO_ROOT / manifest["cases_path"]
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]

    assert manifest["frozen"] is True
    assert {case["domain"] for case in cases} == set(manifest["required_domains"])
    assert set(manifest["required_task_types"]) <= {
        case["task_type"] for case in cases
    }
    assert all(case["repository_tree"] == (
        "a698da9e4b54e2929adacb613bc61ba3e72eed58"
    ) for case in cases)
