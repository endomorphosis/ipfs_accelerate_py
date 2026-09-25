from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomy.recovery_board_fence import (
    build_recovery_delta_items,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinator,
    duckdb_available,
    open_database_coordinator,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    LifecycleState,
)
from ipfs_accelerate_py.agent_supervisor.runtime.spar_runtime_settlement import (
    BOARD,
    CONFIG_SCHEMA,
    PROGRAM,
    observe_spar_runtime_settlement,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_execution_schema import (
    install_database_execution_schema,
)

_TARGET = "repository:ipfs_accelerate_py"
_BRANCH = "codex/semantic-preserving-autonomous-remodularization-v1"


def _spar_config() -> dict:
    return {
        "schema": CONFIG_SCHEMA,
        "program_identifier": PROGRAM,
        "board_namespace": BOARD,
        "max_lanes": 3,
        "merge_target_branch": _BRANCH,
        "database_program": {"store_id": "control.duckdb"},
        "runtime_paths": {
            "state": "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state",
            "merge_queue": "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/merge-queue",
        },
        "lanes": [
            {"index": 0, "name": "spar-lane-0", "strict_shard_remainder": 0},
            {"index": 1, "name": "spar-lane-1", "strict_shard_remainder": 1},
            {"index": 2, "name": "spar-lane-2", "strict_shard_remainder": 2},
        ],
    }


def _runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / "spar-runtime"
    config = (
        root
        / "config"
        / "agent_supervisor_semantic_preserving_remodularization_scheduler.json"
    )
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(_spar_config(), indent=2))
    state = root / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state"
    queue = root / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/merge-queue"
    state.mkdir(parents=True)
    queue.mkdir(parents=True)
    for index in range(3):
        lane = state / f"lane-{index}"
        lane.mkdir()
        prefix = f"spar_lane_{index}"
        coordination = lane / f"{prefix}_database_coordination.duckdb"
        execution = lane / f"{prefix}_database_execution.duckdb"
        DatabaseCoordinator(coordination).open().close()
        install_database_execution_schema(
            execution,
            metadata={
                "authority_mode": "quack",
                "logical_owner_session_id": f"owner-{index}",
                "process_instance_id": f"process:{index + 1:024x}",
                "state_schema_revision": "1",
                "control_schema_profile_id": "bootstrap",
                "control_schema_fingerprint": "bootstrap",
            },
        )
    MergeQueue(queue, target_repository_id=_TARGET, target_branch=_BRANCH, require_target_binding=True)
    return root


def test_outstanding_recovery_plan_delta_blocks_spar_settlement(tmp_path) -> None:
    if not duckdb_available():
        pytest.skip("DuckDB is required for recovery PlanDelta settlement")

    root = _runtime_root(tmp_path)
    owner = {"generation": 1, "store_id": "control.duckdb", "repository_id": _TARGET}
    idle = observe_spar_runtime_settlement(
        root, owner_identity=owner, target_repository_id=_TARGET
    )
    assert idle["admitted"] is True
    assert idle["recovery_plan_delta_outstanding"] is False
    assert idle["merge_queue_empty"] is True

    coordination = (
        root
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/state"
        / "lane-0"
        / "spar_lane_0_database_coordination.duckdb"
    )
    items = build_recovery_delta_items(
        {
            "action": "invalidate_stale_evidence",
            "admit": False,
            "stall_reason": "stale_evidence",
        },
        target_cid="task:spar-recovery",
        lifecycle=LifecycleState.CLAIMED,
    )
    coordinator = open_database_coordinator(coordination)
    try:
        recorded = coordinator.record_recovery_plan_delta(
            task_cid="task:spar-recovery",
            items=tuple(item.to_dict() for item in items),
            now_ms=5,
        )
        event_id = recorded["event_id"]
    finally:
        coordinator.close()
    blocked = observe_spar_runtime_settlement(
        root, owner_identity=owner, target_repository_id=_TARGET
    )
    assert blocked["admitted"] is False, blocked
    assert blocked["settled"] is False
    assert blocked["recovery_plan_delta_outstanding"] is True
    assert blocked["merge_queue_empty"] is False
    assert blocked["active_count"] >= 1
    coordinator = open_database_coordinator(coordination)
    try:
        coordinator.consume_recovery_plan_delta(
            delta_event_id=event_id,
            task_cid="task:spar-recovery",
            now_ms=6,
        )
    finally:
        coordinator.close()
    cleared = observe_spar_runtime_settlement(
        root, owner_identity=owner, target_repository_id=_TARGET
    )
    assert cleared["admitted"] is True
    assert cleared["recovery_plan_delta_outstanding"] is False
    assert cleared["merge_queue_empty"] is True
