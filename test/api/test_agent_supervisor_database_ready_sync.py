"""Cross-lane readiness reconciliation for database-authoritative daemons."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database readiness reconciliation tests",
)


def test_runtime_receipt_schemas_and_wire_bounds_are_defined() -> None:
    """Receipt consumers retain the exact closed schemas and byte ceilings."""

    assert implementation_daemon_module.PROOF_REUSE_STATE_ROOT_ENV == (
        "IPFS_PROOF_REUSE_STATE_ROOT"
    )
    assert implementation_daemon_module.PROVIDER_FILESYSTEM_BOUNDARY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/provider-filesystem-boundary@1"
    )
    assert implementation_daemon_module.PROVIDER_ROUTE_RECEIPT_SCHEMA == (
        "ipfs_accelerate_py/provider-route@1"
    )
    assert implementation_daemon_module.MAX_PROVIDER_ROUTE_RECEIPT_BYTES == (
        16 * 1_024
    )
    assert implementation_daemon_module.ACTIONABLE_RETRY_EVIDENCE_SCHEMA == (
        "ptr/actionable-retry-evidence@1"
    )
    assert implementation_daemon_module.MAX_ACTIONABLE_RETRY_EVIDENCE_BYTES == (
        16 * 1_024
    )
    assert implementation_daemon_module.MAX_ACTIONABLE_RETRY_TEXT_BYTES == 2_048
    assert (
        implementation_daemon_module.MAX_ACTIONABLE_RETRY_TEXT_BYTES
        < implementation_daemon_module.MAX_ACTIONABLE_RETRY_EVIDENCE_BYTES
    )
    assert implementation_daemon_module.RECONCILIATION_PROPOSAL_ADMISSION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/"
        "reconciliation-proposal-admission@1"
    )
    assert (
        implementation_daemon_module.RECONCILIATION_LIFECYCLE_AUTHORITY_SCHEMA
        == "ipfs_accelerate_py/agent-supervisor/"
        "reconciliation-lifecycle-authority@1"
    )


def _population(*, prerequisite_status: str = "ready") -> dict[str, object]:
    return {
        "repository_tree_id": "tree:cross-lane-ready-sync",
        "objectives": [
            {
                "objective_id": "objective:cross-lane-ready-sync",
                "objective_alias": "PCAR-G000",
                "title": "Cross-lane ready sync",
                "goal_cid": "goal:cid:root",
                "goal_alias": "PCAR-G000",
                "status": "open",
            }
        ],
        "tasks": [
            {
                "task_cid": "task:cid:pcar-000",
                "task_id": "PCAR-000",
                "goal_cid": "goal:cid:root",
                "status": prerequisite_status,
                "priority": "P0",
                "ordinal": 0,
                "title": "Seal baseline",
            },
            {
                "task_cid": "task:cid:pcar-001",
                "task_id": "PCAR-001",
                "goal_cid": "goal:cid:root",
                "status": "ready",
                "priority": "P0",
                "ordinal": 1,
                "title": "Inventory architecture",
                "dependencies": ["task:cid:pcar-000"],
            },
            {
                "task_cid": "task:cid:pcar-002",
                "task_id": "PCAR-002",
                "goal_cid": "goal:cid:root",
                "status": "ready",
                "priority": "P0",
                "ordinal": 2,
                "title": "Define architecture contracts",
                "dependencies": ["task:cid:pcar-000"],
            },
        ],
    }


def _open_lane(
    tmp_path: Path,
    *,
    lane: int,
    require_real_execution: bool = True,
) -> DatabaseImplementationDaemon:
    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        return {
            "status": "succeeded",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    def effect(
        attempt: DatabaseTaskAttempt,
        provider_result: dict[str, object],
    ) -> dict[str, object]:
        return {
            "status": "applied",
            "task_cid": attempt.task_cid,
            "provider_result": dict(provider_result),
        }

    def validation(
        attempt: DatabaseTaskAttempt,
        effect_result: dict[str, object],
    ) -> dict[str, object]:
        return {
            "outcome": "passed",
            "evidence_digest": "sha256:" + "a" * 64,
            "argv": ["cross-lane-ready-sync", attempt.task_cid],
            "effect_result": dict(effect_result),
        }

    return DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / f"lane-{lane}.coordination.duckdb",
        execution_path=tmp_path / f"lane-{lane}.execution.duckdb",
        owner_session_id=f"session:lane-{lane}",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_shard_count=3,
        task_shard_index=lane,
        strict_task_sharding=True,
        provider_fn=provider,
        effect_fn=effect,
        validation_fn=validation,
        require_real_execution=require_real_execution,
    )


def test_cross_lane_completion_is_projected_before_successor_claim(
    tmp_path: Path,
) -> None:
    """A completion owned by lane 2 unblocks its lane-0 successor sidecar."""

    completing_lane = _open_lane(tmp_path, lane=2)
    try:
        completing_lane.materialize_population(_population())
        assert completing_lane._task_home_shard_index("PCAR-000") == 2
        result = completing_lane.run_once()
        assert result["claimed_task_cid"] == "task:cid:pcar-000"
        completed = completing_lane.task_source.get("task:cid:pcar-000")
        assert completed is not None
        assert completed.status == "completed"
    finally:
        completing_lane.close()

    successor_lane = _open_lane(tmp_path, lane=0)
    try:
        assert successor_lane._task_home_shard_index("PCAR-002") == 0

        first = successor_lane.sync_ready_tasks_into_coordination()
        first_projection = (
            successor_lane.coordinator.coordination_registry_projection()
        )
        second = successor_lane.sync_ready_tasks_into_coordination()
        second_projection = (
            successor_lane.coordinator.coordination_registry_projection()
        )

        assert first == second == [
            "task:cid:pcar-001",
            "task:cid:pcar-002",
        ]
        assert first_projection["projection_root"] == second_projection[
            "projection_root"
        ]
        assert first_projection["counts"]["logical_completions"] == 1
        ready_frontier_revision = first_projection["logical_completions"][0][
            "body"
        ]["ready_frontier_revision"]
        assert ready_frontier_revision > 0
        assert first_projection["logical_completions"] == [
            {
                "task_cid": "task:cid:pcar-000",
                "status": "succeeded",
                "body": {
                    "authority": "DatabaseTaskSource@1",
                    "projection": "canonical_dependency_completion",
                    "ready_frontier_revision": ready_frontier_revision,
                },
            }
        ]
        readiness = successor_lane.coordinator.claimability(
            "task:cid:pcar-002"
        )
        assert readiness["claimable"] is True
        assert readiness["satisfied_dependency_task_cids"] == [
            "task:cid:pcar-000"
        ]

        attempt = successor_lane.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:pcar-002"
    finally:
        successor_lane.close()

    second_successor_lane = _open_lane(tmp_path, lane=1)
    try:
        assert second_successor_lane._task_home_shard_index("PCAR-001") == 1
        attempt = second_successor_lane.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:pcar-001"
        projection = (
            second_successor_lane.coordinator.coordination_registry_projection()
        )
        assert projection["counts"]["logical_completions"] == 1
    finally:
        second_successor_lane.close()


def test_skipped_canonical_prerequisite_satisfies_lane_local_dependency(
    tmp_path: Path,
) -> None:
    """Canonical skipped semantics are preserved in the local claim sidecar."""

    seed = _open_lane(tmp_path, lane=2, require_real_execution=False)
    try:
        seed.materialize_population(_population(prerequisite_status="skipped"))
    finally:
        seed.close()

    successor_lane = _open_lane(tmp_path, lane=0)
    try:
        assert successor_lane.sync_ready_tasks_into_coordination() == [
            "task:cid:pcar-001",
            "task:cid:pcar-002",
        ]
        projection = successor_lane.coordinator.coordination_registry_projection()
        assert projection["logical_completions"][0]["status"] == "succeeded"
        assert projection["logical_completions"][0]["body"]["authority"] == (
            "DatabaseTaskSource@1"
        )
        attempt = successor_lane.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:pcar-002"
    finally:
        successor_lane.close()


def test_ready_frontier_accepts_1001_dependencies_without_point_reads(
    tmp_path: Path,
) -> None:
    """The canonical 1,024/task bound is not narrowed to the 1,000 page bound."""

    dependencies = tuple(f"task:cid:dep-{index:04d}" for index in range(1_001))
    ready_task = SimpleNamespace(
        task_cid="task:cid:wide-successor",
        task_alias="PCAR-WIDE",
        status="ready",
        revision=7,
        dependencies=dependencies,
        body={},
    )

    class TaskSource:
        point_reads = 0

        def ready_tasks(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(tasks=(ready_task,), revision=17)

        def get(self, _task_cid: str) -> None:
            self.point_reads += 1
            pytest.fail("ready-snapshot reconciliation performed a point read")

    class Coordinator:
        def __init__(self) -> None:
            self.registered: list[str] = []
            self.completions: list[tuple[str, dict[str, object]]] = []

        def coordination_registry_projection(self) -> dict[str, object]:
            return {"tasks": [], "logical_completions": []}

        def register_task(self, *, task_cid: str, **_kwargs: object) -> None:
            self.registered.append(task_cid)

        def mark_task_complete(
            self,
            task_cid: str,
            *,
            body: dict[str, object],
            **_kwargs: object,
        ) -> None:
            self.completions.append((task_cid, dict(body)))

    task_source = TaskSource()
    coordinator = Coordinator()
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "wide-control.duckdb",
        coordination_path=tmp_path / "wide-coordination.duckdb",
        execution_path=tmp_path / "wide-execution.duckdb",
        owner_session_id="session:wide-frontier",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=task_source,
        coordinator=coordinator,
        install_schema=False,
        require_real_execution=True,
    )
    try:
        assert daemon.sync_ready_tasks_into_coordination() == [
            "task:cid:wide-successor"
        ]
    finally:
        daemon.close()

    assert task_source.point_reads == 0
    assert coordinator.registered[:-1] == sorted(dependencies)
    assert coordinator.registered[-1] == "task:cid:wide-successor"
    assert len(coordinator.completions) == 1_001
    assert all(
        body == {
            "authority": "DatabaseTaskSource@1",
            "projection": "canonical_dependency_completion",
            "ready_frontier_revision": 17,
        }
        for _task_cid, body in coordinator.completions
    )
