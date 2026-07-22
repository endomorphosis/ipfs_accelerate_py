from __future__ import annotations

import json

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
    return PortalTask(
        task_id="AUTO-002",
        title="Resolve code annotation in src/runtime.py:2",
        status=status,
        completion="manual",
        priority="P1",
        track="runtime",
        metadata={
            "goal id": "codebase/runtime/src-runtime",
            "graph parents": "codebase/runtime",
            "candidate kind": "codebase_scan",
            "goal registration": "dynamic",
        },
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
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["AUTO-002"],
                "objective_task_janitor_receipts": [
                    {
                        "schema": JANITOR_RECEIPT_SCHEMA,
                        "action": "block",
                        "task_id": "AUTO-002",
                        "retired_task_reason": "orphaned_goal_reference",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
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
