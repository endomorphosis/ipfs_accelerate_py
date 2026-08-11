"""TaskSource protocol extensions for plan-runtime readiness/CAS (PDR-033)."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.parallel_plan_compiler import (
    compile_parallel_execution_plan,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    PARALLEL_PLAN_RUNTIME_INTERFACE,
    TASK_SOURCE_PROTOCOL_SCHEMA,
    ActivePlanBinding,
    TaskSourceConflictError,
    TaskSourceTask,
    bind_active_plan_revision,
    load_active_plan_binding_from_store,
    recompute_readiness_statuses,
    recompute_status_cas,
)


NOW = 1_000_000


def _capacity() -> dict[str, object]:
    return {
        "snapshot_id": "capacity:current",
        "observed_at_ms": NOW,
        "fresh_until_ms": NOW + 60_000,
        "cpu_slots": 4,
        "process_slots": 4,
        "memory_bytes": 4_000,
        "disk_bytes": 10_000,
        "resource_class_slots": {"cpu-small": 4},
    }


def _provider() -> dict[str, object]:
    return {
        "snapshot_id": "provider:alpha:snapshot:current",
        "provider_id": "provider:alpha",
        "observed_at_ms": NOW,
        "fresh_until_ms": NOW + 60_000,
        "healthy": True,
        "available_slots": 2,
        "context_limit": 8_000,
        "available_tokens": 8_000,
        "available_quota": 10,
        "available_cost_micros": 5_000,
        "latency_ms": 50,
        "capabilities": ["llm:code"],
    }


def _task(task_id: str, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "task_id": task_id,
        "outputs": [f"src/{task_id}.py"],
        "produces": [f"leaf:{task_id}"],
        "duration_ms": 500,
        "resource_contract": {
            "resource_class": "cpu-small",
            "resource_stage": "implementation",
            "cpu_slots": 1,
            "process_slots": 1,
            "memory_bytes": 50,
            "disk_bytes": 50,
        },
        "lease_contract": {
            "lease_scope": "task",
            "lease_duration_ms": 10_000,
            "heartbeat_interval_ms": 1_000,
        },
        "worktree_contract": {"policy": "isolated", "isolation_required": True},
        "merge_strategy": {
            "merge_train_id": "merge-train:main",
            "post_merge_validation": ["validation:task"],
        },
    }
    value.update(overrides)
    return value


def _plan_and_binding() -> tuple[Any, ActivePlanBinding]:
    plan = compile_parallel_execution_plan(
        [_task("TS-A"), _task("TS-B", dependencies=["TS-A"], depends_on=["TS-A"])],
        requested_width=1,
        repository_snapshot={
            "tree_id": "tree:test",
            "snapshot_id": "repo:test",
            "fencing_epoch": 1,
        },
        capacity_snapshot=_capacity(),
        provider_snapshots=[_provider()],
        current_time_ms=NOW,
        required_leaf_ids=["leaf:TS-A", "leaf:TS-B"],
    )
    payload = plan.to_dict()
    binding = bind_active_plan_revision(
        active={
            "revision_cid": "revision:ts-1",
            "plan_root_cid": "plan-root:ts-1",
            "semantic_revision": 1,
            "event_cursor": "c1",
            "active_cid": "active:ts-1",
        },
        revision={
            "revision_cid": "revision:ts-1",
            "plan_root_cid": "plan-root:ts-1",
            "execution_plan_cid": payload["plan_id"],
            "semantic_revision": 1,
        },
        execution_plan=payload,
    )
    return plan, binding


def test_protocol_schema_and_runtime_interface_constants() -> None:
    assert TASK_SOURCE_PROTOCOL_SCHEMA.endswith("task-source-protocol@1")
    assert PARALLEL_PLAN_RUNTIME_INTERFACE == "ParallelPlanRuntime@1"


def test_recompute_readiness_with_task_source_task_records() -> None:
    _plan, binding = _plan_and_binding()
    tasks = [
        TaskSourceTask(
            task_id="TS-A",
            task_cid="cid:TS-A",
            goal_id="G1",
            goal_cid="cid:G1",
            title="A",
            status="ready",
            revision=1,
            ordinal=0,
            dependency_task_ids=(),
        ),
        TaskSourceTask(
            task_id="TS-B",
            task_cid="cid:TS-B",
            goal_id="G1",
            goal_cid="cid:G1",
            title="B",
            status="ready",
            revision=1,
            ordinal=1,
            dependency_task_ids=("TS-A",),
        ),
    ]
    statuses = recompute_readiness_statuses(tasks, binding=binding)
    assert statuses["TS-A"] == "ready"
    assert statuses["TS-B"] == "waiting"


class _FakeSource:
    def __init__(self, tasks: dict[str, TaskSourceTask]) -> None:
        self._tasks = tasks
        self.cas_calls: list[dict[str, Any]] = []

    def get(self, task_id: str) -> TaskSourceTask | None:
        return self._tasks.get(task_id)

    def compare_and_swap_status(
        self,
        task_id: str,
        *,
        expected_status: str | list[str],
        new_status: str,
        expected_revision: str | int,
        receipt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.cas_calls.append(
            {
                "task_id": task_id,
                "expected_status": expected_status,
                "new_status": new_status,
                "expected_revision": expected_revision,
                "receipt": receipt,
            }
        )
        task = self._tasks[task_id]
        updated = TaskSourceTask(
            task_id=task.task_id,
            task_cid=task.task_cid,
            goal_id=task.goal_id,
            goal_cid=task.goal_cid,
            title=task.title,
            status=new_status,
            revision=int(task.revision) + 1 if str(task.revision).isdigit() else task.revision,
            ordinal=task.ordinal,
            dependency_task_ids=task.dependency_task_ids,
        )
        self._tasks[task_id] = updated

        class _Result:
            changed = True
            task = updated
            previous_status = "ready"
            revision = updated.revision
            event_cursor = "evt:1"
            receipt_id = "receipt:1"
            identity = None

            def to_dict(self) -> dict[str, Any]:
                return {"changed": True, "task_id": task_id}

        return _Result()  # type: ignore[return-value]


def test_recompute_status_cas_rejects_non_ready_and_admits_ready() -> None:
    _plan, binding = _plan_and_binding()
    tasks = {
        "TS-A": TaskSourceTask(
            task_id="TS-A",
            task_cid="cid:TS-A",
            goal_id="G1",
            goal_cid="cid:G1",
            title="A",
            status="ready",
            revision=1,
            ordinal=0,
        ),
        "TS-B": TaskSourceTask(
            task_id="TS-B",
            task_cid="cid:TS-B",
            goal_id="G1",
            goal_cid="cid:G1",
            title="B",
            status="ready",
            revision=1,
            ordinal=1,
            dependency_task_ids=("TS-A",),
        ),
    }
    source = _FakeSource(tasks)
    # Child is not ready while parent incomplete.
    with pytest.raises(TaskSourceConflictError):
        recompute_status_cas(
            source,  # type: ignore[arg-type]
            "TS-B",
            expected_status="ready",
            new_status="in_progress",
            expected_revision=1,
            binding=binding,
        )
    assert source.cas_calls == []

    result = recompute_status_cas(
        source,  # type: ignore[arg-type]
        "TS-A",
        expected_status="ready",
        new_status="in_progress",
        expected_revision=1,
        binding=binding,
        receipt={"plan_revision_cid": binding.revision_cid},
    )
    assert result.changed is True
    assert len(source.cas_calls) == 1
    assert source.cas_calls[0]["new_status"] == "in_progress"


class _FakePlanStore:
    def __init__(self, active: dict[str, Any], revision: dict[str, Any], plan: dict[str, Any]) -> None:
        self._active = active
        self._revision = revision
        self._plan = plan
        self._quarantined = False

    def is_quarantined(self) -> bool:
        return self._quarantined

    def get_active(self) -> dict[str, Any]:
        return self._active

    def load_revision(self, revision_cid: str) -> dict[str, Any]:
        assert revision_cid == self._active["revision_cid"]
        return self._revision

    def get_cas(self, cid: str) -> dict[str, Any]:
        assert cid == self._revision["execution_plan_cid"]
        return self._plan


def test_load_active_plan_binding_from_store() -> None:
    plan, _binding = _plan_and_binding()
    payload = plan.to_dict()
    store = _FakePlanStore(
        active={
            "revision_cid": "revision:ts-1",
            "plan_root_cid": "plan-root:ts-1",
            "semantic_revision": 1,
            "event_cursor": "c1",
            "active_cid": "active:ts-1",
            "quarantined": False,
        },
        revision={
            "revision_cid": "revision:ts-1",
            "plan_root_cid": "plan-root:ts-1",
            "execution_plan_cid": payload["plan_id"],
            "semantic_revision": 1,
        },
        plan=payload,
    )
    loaded = load_active_plan_binding_from_store(store)
    assert loaded.revision_cid == "revision:ts-1"
    assert loaded.plan_id == payload["plan_id"]
    assert "TS-A" in loaded.plan_task_ids
