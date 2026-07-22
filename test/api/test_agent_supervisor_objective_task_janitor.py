from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.objective_graph import ObjectiveGoal
from ipfs_accelerate_py.agent_supervisor.objective_task_janitor import (
    JANITOR_RECEIPT_SCHEMA,
    reconcile_objective_task_strategy,
    registered_goal_ids_from_bundle_index,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


def _dynamic_task(*, status: str = "todo") -> PortalTask:
    metadata = {
        "goal id": "codebase/runtime/src-runtime",
        "graph parents": "codebase/runtime",
        "candidate kind": "codebase_scan",
        "goal registration": "dynamic",
    }
    if status == "blocked":
        metadata["blocked reason"] = (
            "Retired by objective-task janitor during launch steering because "
            "orphaned_goal_reference."
        )
    return PortalTask(
        task_id="AUTO-002",
        title="Resolve code annotation in src/runtime.py:2",
        status=status,
        completion="manual",
        priority="P1",
        track="runtime",
        metadata=metadata,
    )


def test_bundle_index_registers_open_dynamic_goals_only():
    payload = {
        "bundles": {
            "codebase/runtime/src-runtime": {
                "tasks": [
                    {
                        "task_id": "AUTO-002",
                        "status": "blocked",
                        "candidate_kind": "codebase_scan",
                        "goal_registration": "dynamic",
                        "goal_id": "codebase/runtime/src-runtime",
                        "parent_goal_id": "codebase/runtime",
                        "parent_goal_ids": ["codebase"],
                    }
                ]
            },
            "codebase/runtime/src-complete": {
                "tasks": [
                    {
                        "task_id": "AUTO-003",
                        "status": "completed",
                        "goal_registration": "dynamic",
                        "goal_id": "codebase/runtime/src-complete",
                    }
                ]
            },
            "objective/static": {
                "tasks": [
                    {
                        "task_id": "AUTO-001",
                        "status": "todo",
                        "candidate_kind": "seed",
                        "goal_id": "G1.S1",
                    }
                ]
            },
        }
    }

    assert registered_goal_ids_from_bundle_index(payload) == [
        "codebase/runtime/src-runtime",
        "codebase/runtime",
        "codebase",
    ]


def test_registered_dynamic_goal_is_not_retired_as_orphan():
    result = reconcile_objective_task_strategy(
        goals=[
            ObjectiveGoal(
                goal_id="G1",
                title="Static objective",
                fields={"status": "active", "priority": "P0"},
            )
        ],
        tasks=[_dynamic_task()],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
        registered_goal_ids=["codebase/runtime", "codebase/runtime/src-runtime"],
    )

    assert result["blocked_task_ids"] == []
    assert result["deprioritized_task_ids"] == []
    assert result["registered_goal_ids"] == [
        "codebase/runtime",
        "codebase/runtime/src-runtime",
    ]


def test_registered_dynamic_goal_unblocks_prior_janitor_retirement():
    prior_receipt = {
        "schema": JANITOR_RECEIPT_SCHEMA,
        "action": "block",
        "task_id": "AUTO-002",
        "retired_task_reason": "orphaned_goal_reference",
    }
    result = reconcile_objective_task_strategy(
        goals=[],
        tasks=[_dynamic_task(status="blocked")],
        strategy={
            "blocked_tasks": ["AUTO-002"],
            "objective_task_janitor_receipts": [prior_receipt],
        },
        now="2026-07-22T00:00:00+00:00",
        registered_goal_ids=["codebase/runtime", "codebase/runtime/src-runtime"],
    )

    assert result["blocked_task_ids"] == []
    assert result["unblocked_task_ids"] == ["AUTO-002"]
    assert result["strategy"]["blocked_tasks"] == []
    assert result["receipts"][0]["action"] == "unblock"


def test_registered_dynamic_goal_recovers_materialized_block_without_receipt():
    result = reconcile_objective_task_strategy(
        goals=[],
        tasks=[_dynamic_task(status="blocked")],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
        registered_goal_ids=["codebase/runtime", "codebase/runtime/src-runtime"],
    )

    assert result["unblocked_task_ids"] == ["AUTO-002"]
    assert result["strategy"]["blocked_tasks"] == []


def test_unresolved_materialized_block_retains_janitor_ownership():
    result = reconcile_objective_task_strategy(
        goals=[],
        tasks=[_dynamic_task(status="blocked")],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
    )

    assert result["blocked_task_ids"] == ["AUTO-002"]
    assert result["strategy"]["blocked_tasks"] == ["AUTO-002"]
    assert result["receipts"][0]["retired_task_reason"] == "orphaned_goal_reference"


def test_supervisor_loads_bundle_registry_and_materializes_unblock(tmp_path):
    todo_path = tmp_path / "todo.md"
    objective_path = tmp_path / "objective.md"
    bundle_dir = tmp_path / "bundles"
    state_dir = tmp_path / "state"
    strategy_path = state_dir / "strategy.json"
    events_path = state_dir / "events.jsonl"
    state_path = state_dir / "task_state.json"
    bundle_dir.mkdir()
    state_dir.mkdir()
    objective_path.write_text(
        """# Objective heap

## G1 Static objective

- Status: active
- Priority: P0
""",
        encoding="utf-8",
    )
    todo_path.write_text(
        """# Todo

## AUTO-002 Resolve code annotation in src/runtime.py:2

- Status: blocked
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Goal id: codebase/runtime/src-runtime
- Graph parents: codebase/runtime
- Candidate kind: codebase_scan
- Goal registration: dynamic
- Blocked reason: Retired by objective-task janitor because orphaned_goal_reference.
""",
        encoding="utf-8",
    )
    (bundle_dir / "index.json").write_text(
        json.dumps(
            {
                "bundles": {
                    "codebase/runtime/src-runtime": {
                        "tasks": [
                            {
                                "task_id": "AUTO-002",
                                "status": "blocked",
                                "candidate_kind": "codebase_scan",
                                "goal_id": "codebase/runtime/src-runtime",
                                "parent_goal_id": "codebase/runtime",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    strategy_path.write_text("{}\n", encoding="utf-8")
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=todo_path,
            state_path=state_path,
            strategy_path=strategy_path,
            events_path=events_path,
            state_dir=state_dir,
            task_prefix="## AUTO-",
            objective_path=objective_path,
            objective_bundle_dir=bundle_dir,
        )
    )

    result = supervisor.reconcile_objective_task_janitor()

    assert result["unblocked_task_ids"] == ["AUTO-002"]
    assert result["materialized"]["unblocked_task_ids"] == ["AUTO-002"]
    assert result["materialized"]["removed_reason_task_ids"] == ["AUTO-002"]
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "- Status: todo" in todo_text
    assert "- Blocked reason:" not in todo_text


def _goal_task(goal_id: str = "G1") -> PortalTask:
    return PortalTask(
        task_id="AUTO-010",
        title="Implement reopened objective",
        status="todo",
        completion="manual",
        priority="P0",
        track="runtime",
        metadata={"goal id": goal_id},
    )


def test_reopened_goal_remains_schedulable_and_keeps_linked_work_open():
    result = reconcile_objective_task_strategy(
        goals=[ObjectiveGoal("G1", "Regressed objective", {"status": "reopened", "priority": "P0"})],
        tasks=[_goal_task()],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
    )

    assert result["blocked_task_ids"] == []
    assert result["active_goal_ids"] == ["G1"]
    assert result["scheduled_goal_ids"] == ["G1"]
    assert result["open_goal_ids"] == ["G1"]


@pytest.mark.parametrize("status", ["verified_complete", "completed"])
def test_verified_goal_retires_linked_work_as_completed(status: str):
    result = reconcile_objective_task_strategy(
        goals=[ObjectiveGoal("G1", "Done objective", {"status": status})],
        tasks=[_goal_task()],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
    )

    assert result["blocked_task_ids"] == ["AUTO-010"]
    assert result["receipts"][0]["retired_task_reason"] == "goal_completed"
