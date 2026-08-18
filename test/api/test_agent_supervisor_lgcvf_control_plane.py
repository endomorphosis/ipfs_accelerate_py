"""Focused conformance tests for the LGCVF one-writer control plane."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    FormalWorkPlan,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)

from scripts import (
    materialize_logic_governed_compositional_verification_fabric_control_plane as materializer,
)

ROOT = Path(__file__).resolve().parents[2]


def _population() -> tuple[dict[str, Any], dict[str, Any]]:
    config = materializer.load_config()
    formal_path = ROOT / str(config["formal_plan_path"])
    todo_path = ROOT / str(config["taskboard_path"])
    formal = FormalWorkPlan.from_dict(json.loads(formal_path.read_text(encoding="utf-8")))
    source = {
        "accelerator_head": "1" * 40,
        "accelerator_tree": "2" * 40,
        "source_forest_root": "baguqeera-test-source-forest",
    }
    population = materializer.project_population(
        config,
        formal_plan=formal,
        todo_text=todo_path.read_text(encoding="utf-8"),
        source=source,
    )
    return config, population


def test_lgcvf_scheduler_is_single_writer_and_protects_control_evidence() -> None:
    config = materializer.load_config()
    board = load_configured_board(
        materializer.CONFIG_PATH,
        repo_root=ROOT,
    )

    assert board.board_namespace == materializer.EXPECTED_NAMESPACE
    assert board.merge_target_branch == (
        "agent/logic-governed-compositional-verification-fabric-v1"
    )
    assert board.max_lanes == 1
    assert board.worktree_submodule_paths == ("ipfs_datasets_py",)
    assert config["bootstrap_writer_policy"] == {
        "maximum_processes": 1,
        "quack_required": False,
        "direct_multi_process_duckdb_permitted": False,
        "automatic_installation_permitted": False,
    }
    program = board.resolved_database_program()
    assert program.authority_mode == "embedded"
    assert program.task_source_kind == "duckdb"
    assert program.quack_endpoint == ""
    assert program.failover_policy == "fail_closed"
    protected = set(board.protected_paths)
    assert {
        str(config["formal_plan_path"]),
        str(config["taskboard_path"]),
        str(config["validator_path"]),
        str(config["materializer_path"]),
        "docs/architecture/logic_governed_compositional_verification/paired_benchmark_result.json",
        "test/api/test_agent_supervisor_compositional_verification_vertical.py",
        "test/fixtures/agent_supervisor/compositional_verification/tests/test_selected.py",
        "test/fixtures/agent_supervisor/compositional_verification/tests/test_unselected.py",
    }.issubset(protected)


def test_explicit_objective_heap_keeps_optional_objective_daemon_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    optional_module = (
        "ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon"
    )
    monkeypatch.setitem(sys.modules, optional_module, None)
    holder = SimpleNamespace(
        config=SimpleNamespace(
            objective_path=(
                ROOT
                / "docs/architecture/"
                "logic_governed_compositional_verification_fabric.objectives.md"
            ),
            repo_root=ROOT,
        )
    )

    goals = PortalImplementationSupervisor._objective_goals_for_finding_mapping(holder)

    assert {goal.goal_id for goal in goals} >= {"LGCVF-G000", "LGCVF-G130"}


def test_population_preserves_formal_identities_dependencies_and_closed_gates() -> None:
    config, population = _population()
    formal = FormalWorkPlan.from_dict(
        json.loads((ROOT / str(config["formal_plan_path"])).read_text(encoding="utf-8"))
    )
    projected = {str(item["task_id"]): item for item in population["tasks"]}
    formal_tasks = {item.task_id: item for item in formal.tasks}

    assert population["plan_root_cid"] == formal.content_id
    assert len(population["objectives"]) == 14
    assert len(population["tasks"]) == 27
    for task_id, task in formal_tasks.items():
        record = projected[task_id]
        assert record["task_cid"] == task.content_id
        assert record["formal_task_content_id"] == task.content_id
        assert record["dependencies"] == [
            formal_tasks[dependency].content_id for dependency in task.depends_on
        ]

    assert projected["LGCVF-121"]["status"] == "blocked"
    assert projected["LGCVF-121"]["construction_status"] == ("blocked_external_authority")
    assert projected["LGCVF-123"]["status"] == "blocked"
    assert projected["LGCVF-123"]["construction_status"] == "blocked_manual"
    for task_id, completion in (
        ("LGCVF-121", "external-authority"),
        ("LGCVF-123", "manual"),
    ):
        assert projected[task_id]["completion"] == completion
        assert projected[task_id]["review_only"] is True
        assert projected[task_id]["is_schedulable"] is False


def test_typed_materialization_read_only_replay_and_overwrite_rejection(
    tmp_path: Path,
) -> None:
    assert DatabaseTaskSource.available(), "required DuckDB capability is unavailable"
    config, population = _population()
    temporary = copy.deepcopy(config)
    temporary["database_program"]["store_id"] = "run-v1/control.duckdb"
    temporary["runtime_paths"]["evidence"] = "run-v1/evidence"

    receipt = materializer.materialize(
        temporary,
        population,
        root=tmp_path,
        recheck_source=False,
    )
    assert receipt["maximum_writer_processes"] == 1
    assert receipt["quack_qualified"] is False
    assert receipt["plan_root_cid"] == population["plan_root_cid"]
    assert receipt["verification"]["valid"] is True
    assert receipt["verification"]["stores_unchanged"] is True

    live = materializer.verify_read_only(
        temporary,
        population,
        root=tmp_path,
        expected_stage="live",
    )
    assert live["valid"] is True
    assert live["verification_mode"] == "read_only"
    assert live["control"]["task_count"] == 27
    assert live["coordination"]["counts"]["registered_tasks"] == 27
    assert not any(live["execution"]["row_counts"].values())

    control_path = tmp_path / str(temporary["database_program"]["store_id"])
    source = DatabaseTaskSource(control_path, install_schema=False)
    try:
        ready = {item.task_alias for item in source.ready_tasks(limit=100).tasks}
        manual = source.get_task("LGCVF-123")
        external = source.get_task("LGCVF-121")
    finally:
        source.close()
    assert ready == {"LGCVF-051", "LGCVF-060", "LGCVF-070", "LGCVF-080"}
    assert manual is not None
    assert manual.status == "blocked"
    assert manual.body["construction_status"] == "blocked_manual"
    assert manual.body["completion"] == "manual"
    assert external is not None
    assert external.body["construction_status"] == "blocked_external_authority"

    with pytest.raises(materializer.MaterializationError, match="refusing to overwrite"):
        materializer.materialize(
            temporary,
            population,
            root=tmp_path,
            recheck_source=False,
        )
