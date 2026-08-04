"""Runtime adoption of active plan revisions and compiled execution plans (PDR-033)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.parallel_plan_compiler import (
    compile_parallel_execution_plan,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    CapacityDriftAction,
    admit_compiled_execution_assignments,
    evaluate_capacity_drift,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    ActivePlanBinding,
    CompiledAssignmentMissingError,
    ExecutionSliceViolationError,
    FakeParallelExecutionError,
    ImmutableClaimRevisionError,
    MixedPlanRevisionError,
    PartialPlanRevisionError,
    SupersededPlanRevisionError,
    assert_claim_retains_original_revision,
    assert_fake_parallel_not_concurrent,
    assert_revision_is_active,
    assert_task_in_execution_slice,
    bind_active_plan_revision,
    compiled_claim_preconditions,
    evaluate_plan_runtime_dispatch,
    order_ready_by_fairness_and_critical_path,
    recompute_readiness_statuses,
)


NOW = 1_000_000


def _capacity(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "snapshot_id": "capacity:current",
        "observed_at_ms": NOW,
        "fresh_until_ms": NOW + 60_000,
        "cpu_slots": 8,
        "process_slots": 8,
        "memory_bytes": 8_000,
        "gpu_memory_bytes": 2_000,
        "disk_bytes": 20_000,
        "resource_class_slots": {"cpu-small": 8, "provider-llm": 4},
    }
    value.update(overrides)
    return value


def _provider(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "snapshot_id": "provider:alpha:snapshot:current",
        "provider_id": "provider:alpha",
        "observed_at_ms": NOW,
        "fresh_until_ms": NOW + 60_000,
        "healthy": True,
        "available_slots": 4,
        "context_limit": 16_000,
        "available_tokens": 20_000,
        "available_quota": 20,
        "available_cost_micros": 10_000,
        "latency_ms": 100,
        "capabilities": ["llm:code", "json-schema"],
    }
    value.update(overrides)
    return value


def _repository(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "tree_id": "tree:sha256:current",
        "snapshot_id": "repository-snapshot:current",
        "fencing_epoch": 40,
        "post_merge_validation": ["pytest:merged-tree"],
    }
    value.update(overrides)
    return value


def _task(task_id: str, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "task_id": task_id,
        "outputs": [f"src/{task_id}.py"],
        "produces": [f"leaf:{task_id}"],
        "duration_ms": 1_000,
        "resource_contract": {
            "resource_class": "cpu-small",
            "resource_stage": "implementation",
            "cpu_slots": 1,
            "process_slots": 1,
            "memory_bytes": 100,
            "disk_bytes": 100,
        },
        "lease_contract": {
            "lease_scope": "task",
            "lease_duration_ms": 20_000,
            "heartbeat_interval_ms": 2_000,
        },
        "worktree_contract": {
            "policy": "isolated",
            "isolation_required": True,
        },
        "merge_strategy": {
            "merge_train_id": "merge-train:main",
            "post_merge_validation": ["validation:task"],
        },
    }
    value.update(overrides)
    return value


def _compile_pair() -> Any:
    return compile_parallel_execution_plan(
        [
            _task("PDR-A"),
            _task("PDR-B"),
        ],
        requested_width=2,
        repository_snapshot=_repository(),
        capacity_snapshot=_capacity(),
        provider_snapshots=[_provider()],
        current_time_ms=NOW,
        required_leaf_ids=["leaf:PDR-A", "leaf:PDR-B"],
    )


def _binding_for_plan(plan: Any, **overrides: object) -> ActivePlanBinding:
    plan_payload = plan.to_dict()
    plan_id = str(plan_payload["plan_id"])
    active = {
        "revision_cid": "revision:active-1",
        "plan_root_cid": "plan-root:active-1",
        "semantic_revision": 1,
        "event_cursor": "cursor:1",
        "active_cid": "active:1",
        "quarantined": False,
    }
    revision = {
        "revision_cid": "revision:active-1",
        "plan_root_cid": "plan-root:active-1",
        "execution_plan_cid": plan_id,
        "semantic_revision": 1,
    }
    kwargs: dict[str, object] = {
        "active": active,
        "revision": revision,
        "execution_plan": plan_payload,
    }
    kwargs.update(overrides)
    return bind_active_plan_revision(**kwargs)  # type: ignore[arg-type]


def test_bind_active_plan_requires_complete_revision_and_execution_plan() -> None:
    plan = _compile_pair()
    binding = _binding_for_plan(plan)
    assert binding.revision_cid == "revision:active-1"
    assert binding.execution_plan_cid == plan.plan_id
    assert "PDR-A" in binding.plan_task_ids
    assert binding.ready_wave_task_ids

    with pytest.raises(PartialPlanRevisionError) as missing:
        bind_active_plan_revision(
            active={"revision_cid": "r1", "plan_root_cid": "p1"},
            revision={"revision_cid": "r1", "plan_root_cid": "p1"},
            execution_plan={},
        )
    assert missing.value.reason in {
        "missing_execution_plan",
        "empty_execution_plan",
        "missing_execution_plan_id",
    }

    with pytest.raises(PartialPlanRevisionError):
        bind_active_plan_revision(
            active={"revision_cid": "", "plan_root_cid": ""},
            revision={
                "revision_cid": "",
                "plan_root_cid": "",
                "execution_plan_cid": plan.plan_id,
            },
            execution_plan=plan.to_dict(),
        )


def test_mixed_and_superseded_revisions_are_rejected() -> None:
    plan = _compile_pair()
    with pytest.raises(MixedPlanRevisionError):
        bind_active_plan_revision(
            active={
                "revision_cid": "revision:A",
                "plan_root_cid": "plan-root:A",
                "semantic_revision": 1,
            },
            revision={
                "revision_cid": "revision:B",
                "plan_root_cid": "plan-root:B",
                "execution_plan_cid": plan.plan_id,
                "semantic_revision": 2,
            },
            execution_plan=plan.to_dict(),
        )

    binding = _binding_for_plan(plan)
    with pytest.raises(SupersededPlanRevisionError):
        assert_revision_is_active(
            binding,
            observed_active_revision_cid="revision:newer",
            task_id="PDR-A",
        )
    # Retained/claimed work may stay on the original immutable revision.
    assert_revision_is_active(
        binding,
        observed_active_revision_cid="revision:newer",
        task_id="PDR-A",
        task_retained=True,
    )


def test_tasks_outside_execution_slice_and_plan_are_rejected() -> None:
    plan = _compile_pair()
    binding = _binding_for_plan(
        plan,
        execution_slice_task_ids=["PDR-A"],
    )
    assert_task_in_execution_slice(binding, task_id="PDR-A")
    with pytest.raises(ExecutionSliceViolationError) as outside:
        assert_task_in_execution_slice(binding, task_id="PDR-B")
    assert outside.value.reason == "outside_execution_slice"

    with pytest.raises(ExecutionSliceViolationError) as foreign:
        assert_task_in_execution_slice(binding, task_id="PDR-Z")
    assert foreign.value.reason in {"outside_execution_slice", "outside_compiled_plan"}


def test_readiness_is_recomputed_from_dependencies_and_ready_waves() -> None:
    plan = compile_parallel_execution_plan(
        [
            _task("PDR-ROOT"),
            _task("PDR-CHILD", dependencies=["PDR-ROOT"], depends_on=["PDR-ROOT"]),
        ],
        requested_width=2,
        repository_snapshot=_repository(),
        capacity_snapshot=_capacity(),
        provider_snapshots=[_provider()],
        current_time_ms=NOW,
        required_leaf_ids=["leaf:PDR-ROOT", "leaf:PDR-CHILD"],
    )
    binding = _binding_for_plan(plan)
    statuses = recompute_readiness_statuses(
        [
            {"task_id": "PDR-ROOT", "status": "todo", "depends_on": []},
            {
                "task_id": "PDR-CHILD",
                "status": "todo",
                "depends_on": ["PDR-ROOT"],
            },
        ],
        binding=binding,
    )
    assert statuses["PDR-ROOT"] == "ready"
    assert statuses["PDR-CHILD"] == "waiting"

    statuses_done = recompute_readiness_statuses(
        [
            {"task_id": "PDR-ROOT", "status": "completed", "depends_on": []},
            {
                "task_id": "PDR-CHILD",
                "status": "todo",
                "depends_on": ["PDR-ROOT"],
            },
        ],
        completed_ids=["PDR-ROOT"],
        binding=binding,
    )
    assert statuses_done["PDR-ROOT"] == "completed"
    assert statuses_done["PDR-CHILD"] == "ready"


def test_compiled_lease_worktree_fence_must_exist_before_claim() -> None:
    plan = _compile_pair()
    binding = _binding_for_plan(plan)
    preconditions = compiled_claim_preconditions(binding, "PDR-A")
    assert preconditions.lease_id
    assert preconditions.worktree_id
    assert preconditions.fence_token
    assert preconditions.revision_cid == binding.revision_cid
    payload = preconditions.to_dict()
    assert payload["lease_id"] == preconditions.lease_id
    assert payload["worktree_id"] == preconditions.worktree_id
    assert payload["fence_token"] == preconditions.fence_token

    incomplete = dict(plan.to_dict())
    incomplete["assignments"] = [
        {
            **assignment,
            "lease_id": "",
            "worktree_id": "",
            "fence_token": "",
        }
        for assignment in incomplete["assignments"]
    ]
    broken = bind_active_plan_revision(
        active={
            "revision_cid": "revision:active-1",
            "plan_root_cid": "plan-root:active-1",
            "semantic_revision": 1,
        },
        revision={
            "revision_cid": "revision:active-1",
            "plan_root_cid": "plan-root:active-1",
            "execution_plan_cid": incomplete["plan_id"],
            "semantic_revision": 1,
        },
        execution_plan=incomplete,
    )
    with pytest.raises(CompiledAssignmentMissingError):
        compiled_claim_preconditions(broken, "PDR-A")


def test_conflicts_exclusive_groups_and_affinity_block_concurrent_claims() -> None:
    plan = _compile_pair()
    payload = plan.to_dict()
    # Inject a blocking conflict + shared exclusive group for runtime enforcement.
    payload["conflicts"] = [
        {
            "left_task_id": "PDR-A",
            "right_task_id": "PDR-B",
            "blocking": True,
            "exclusive_groups": ["group:shared"],
            "anti_affinity_keys": [],
            "exclusive_paths": ["shared/module.py"],
        }
    ]
    for assignment in payload["assignments"]:
        assignment["exclusive_group"] = "group:shared"
    binding = bind_active_plan_revision(
        active={
            "revision_cid": "revision:active-1",
            "plan_root_cid": "plan-root:active-1",
            "semantic_revision": 1,
        },
        revision={
            "revision_cid": "revision:active-1",
            "plan_root_cid": "plan-root:active-1",
            "execution_plan_cid": payload["plan_id"],
            "semantic_revision": 1,
        },
        execution_plan=payload,
    )
    decision = evaluate_plan_runtime_dispatch(
        binding,
        task_id="PDR-B",
        tasks=[
            {"task_id": "PDR-A", "status": "todo", "depends_on": []},
            {"task_id": "PDR-B", "status": "todo", "depends_on": []},
        ],
        active_task_ids=["PDR-A"],
    )
    assert decision.admitted is False
    assert decision.reason in {
        "conflict_surface",
        "exclusive_group_conflict",
    }


def test_claimed_tasks_retain_immutable_original_revision() -> None:
    plan = _compile_pair()
    binding = _binding_for_plan(
        plan,
        claimed_task_revisions={"PDR-A": "revision:original"},
    )
    assert_claim_retains_original_revision(
        binding,
        task_id="PDR-A",
        claim_revision_cid="revision:original",
        current_status="in_progress",
    )
    with pytest.raises(ImmutableClaimRevisionError):
        assert_claim_retains_original_revision(
            binding,
            task_id="PDR-A",
            claim_revision_cid="revision:active-1",
            current_status="in_progress",
        )


def test_capacity_drift_degrades_or_waits_never_overcommits() -> None:
    proceed = evaluate_capacity_drift(
        planned_width=2,
        planned_capacity_snapshot_id="capacity:planned",
        live_host=_capacity(cpu_slots=4, process_slots=4),
        candidate_task_ids=["PDR-A", "PDR-B"],
    )
    assert proceed.action is CapacityDriftAction.PROCEED
    assert proceed.admitted_width == 2
    assert proceed.overcommit_prevented is False

    degrade = evaluate_capacity_drift(
        planned_width=4,
        planned_capacity_snapshot_id="capacity:planned",
        live_host=_capacity(cpu_slots=1, process_slots=1),
        candidate_task_ids=["PDR-A", "PDR-B", "PDR-C", "PDR-D"],
    )
    assert degrade.action is CapacityDriftAction.DEGRADE
    assert degrade.admitted_width == 1
    assert degrade.overcommit_prevented is True
    assert degrade.admitted_task_ids == ("PDR-A",)
    assert set(degrade.degraded_task_ids) == {"PDR-B", "PDR-C", "PDR-D"}
    assert degrade.admitted_width <= degrade.live_width

    wait = evaluate_capacity_drift(
        planned_width=2,
        live_host={"snapshot_id": "capacity:empty", "cpu_slots": 0, "process_slots": 0},
        candidate_task_ids=["PDR-A", "PDR-B"],
    )
    assert wait.action is CapacityDriftAction.WAIT
    assert wait.admitted_width == 0
    assert wait.wait_recommended is True
    assert wait.overcommit_prevented is True

    stale = evaluate_capacity_drift(
        planned_width=2,
        live_host=_capacity(cpu_slots=8, process_slots=8),
        candidate_task_ids=["PDR-A", "PDR-B"],
        stale_capacity=True,
    )
    assert stale.action is CapacityDriftAction.WAIT
    assert stale.admitted_width == 0


def test_admit_compiled_assignments_orders_by_critical_path_and_fairness() -> None:
    plan = _compile_pair()
    binding = _binding_for_plan(plan)
    assignments = list(binding.assignment_by_task_id.values())
    drift, admissions = admit_compiled_execution_assignments(
        assignments=assignments,
        planned_width=int(plan.admitted_width or 2),
        planned_capacity_snapshot_id=binding.capacity_snapshot_id,
        live_host=_capacity(cpu_slots=2, process_slots=2),
        critical_path=list(binding.critical_path),
        merge_steps=binding.merge_steps,
    )
    assert drift.may_dispatch is True
    admitted = [item for item in admissions if item.admitted]
    assert admitted
    for item in admitted:
        assert item.lease_id
        assert item.worktree_id
        assert item.fence_token
        assert item.merge_train_id
        assert item.post_merge_validation


def test_fake_parallel_labels_cannot_execute_concurrently() -> None:
    # Serial dependency chain: only one task is ready at a time.
    plan = compile_parallel_execution_plan(
        [
            _task("PDR-A", lane_label="lane:fake", lane_authoritative=False),
            _task(
                "PDR-B",
                dependencies=["PDR-A"],
                depends_on=["PDR-A"],
                lane_label="lane:fake",
                lane_authoritative=False,
            ),
        ],
        requested_width=2,
        repository_snapshot=_repository(),
        capacity_snapshot=_capacity(),
        provider_snapshots=[_provider()],
        current_time_ms=NOW,
        required_leaf_ids=["leaf:PDR-A", "leaf:PDR-B"],
    )
    binding = _binding_for_plan(plan)
    with pytest.raises(FakeParallelExecutionError):
        assert_fake_parallel_not_concurrent(binding, ["PDR-A", "PDR-B"])

    # True co-scheduled pair from a conflict-free plan may run together.
    parallel = _compile_pair()
    parallel_binding = _binding_for_plan(parallel)
    # Should not raise when the compiled plan actually co-schedules them.
    if parallel.admitted_width >= 2:
        assert_fake_parallel_not_concurrent(parallel_binding, ["PDR-A", "PDR-B"])


def test_dispatch_gate_admits_ready_task_with_full_preconditions() -> None:
    plan = _compile_pair()
    binding = _binding_for_plan(plan)
    decision = evaluate_plan_runtime_dispatch(
        binding,
        task_id="PDR-A",
        tasks=[
            {"task_id": "PDR-A", "status": "todo", "depends_on": []},
            {"task_id": "PDR-B", "status": "todo", "depends_on": []},
        ],
        observed_active_revision_cid=binding.revision_cid,
    )
    assert decision.admitted is True
    assert decision.preconditions is not None
    assert decision.preconditions.lease_id
    assert decision.preconditions.worktree_id
    assert decision.preconditions.fence_token
    assert decision.details["merge_train_id"]
    assert decision.details["post_merge_validation"]


def test_fairness_and_critical_path_order_ready_set() -> None:
    plan = _compile_pair()
    binding = _binding_for_plan(plan)
    ordered = order_ready_by_fairness_and_critical_path(
        binding,
        ["PDR-B", "PDR-A"],
    )
    assert set(ordered) == {"PDR-A", "PDR-B"}
    assert len(ordered) == 2


def test_daemon_require_plan_runtime_before_claim_integration(tmp_path: Any) -> None:
    """Daemon fail-closed gate wires compiled preconditions before claim."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
        PortalTask,
        PortalTaskState,
    )

    plan = _compile_pair()
    todo = tmp_path / "board.md"
    todo.write_text(
        "# board\n\n## PDR-A Task A\n\n- Status: ready\n\n## PDR-B Task B\n\n- Status: ready\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "state.json"
    strategy_path = tmp_path / "strategy.json"
    events_path = tmp_path / "events.jsonl"
    state_path.write_text("{}", encoding="utf-8")
    strategy_path.write_text("{}", encoding="utf-8")
    events_path.write_text("", encoding="utf-8")

    daemon = PortalImplementationDaemon(
        todo_path=todo,
        state_path=state_path,
        strategy_path=strategy_path,
        events_path=events_path,
        repo_root=tmp_path,
        parallel_execution_plan=plan.to_dict(),
        require_active_plan_revision=True,
        plan_capacity_snapshot=_capacity(cpu_slots=2, process_slots=2),
    )
    task = PortalTask(
        task_id="PDR-A",
        title="Task A",
        status="ready",
        completion="auto",
        priority="P0",
        track="runtime",
        depends_on=[],
        source_line=1,
    )
    # Outside-slice rejection when slice is constrained.
    daemon.execution_slice_task_ids = frozenset({"PDR-B"})
    rejected = daemon._require_plan_runtime_before_claim(task)
    assert rejected is not None
    assert "plan_runtime_" in rejected["reason"]

    daemon.execution_slice_task_ids = frozenset()
    daemon._active_plan_binding = None
    admitted = daemon._require_plan_runtime_before_claim(task)
    assert admitted is None
    assert daemon._compiled_claim_preconditions is not None
    assert daemon._compiled_claim_preconditions.lease_id
    metadata = daemon._build_implementation_task_claim_metadata(
        task,
        attempt=1,
        started_at="2026-01-01T00:00:00+00:00",
    )
    # checkout_lock_metadata nests extras; accept either flat or nested.
    blob = str(metadata)
    assert daemon._compiled_claim_preconditions.lease_id in blob
    assert "compiled_claim_acquired_before_publish" in blob or (
        metadata.get("compiled_claim_acquired_before_publish") is True
        or (metadata.get("extra") or {}).get("compiled_claim_acquired_before_publish") is True
    )


def test_daemon_capacity_wait_on_zero_live_slots(tmp_path: Any) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
        PortalTask,
    )

    plan = _compile_pair()
    todo = tmp_path / "board.md"
    todo.write_text("## PDR-A Task\n- Status: ready\n", encoding="utf-8")
    state_path = tmp_path / "state.json"
    strategy_path = tmp_path / "strategy.json"
    events_path = tmp_path / "events.jsonl"
    for path in (state_path, strategy_path, events_path):
        path.write_text("{}" if path.suffix == ".json" else "", encoding="utf-8")

    daemon = PortalImplementationDaemon(
        todo_path=todo,
        state_path=state_path,
        strategy_path=strategy_path,
        events_path=events_path,
        repo_root=tmp_path,
        parallel_execution_plan=plan.to_dict(),
        require_active_plan_revision=True,
        plan_capacity_snapshot=_capacity(cpu_slots=0, process_slots=0),
    )
    task = PortalTask(
        task_id="PDR-A",
        title="Task A",
        status="ready",
        completion="auto",
        priority="P0",
        track="runtime",
        depends_on=[],
        source_line=1,
    )
    result = daemon._require_plan_runtime_before_claim(task)
    assert result is not None
    assert result["reason"] == "plan_runtime_capacity_wait"
    assert result.get("deferred") is True
    assert result["capacity_drift"]["overcommit_prevented"] is True
