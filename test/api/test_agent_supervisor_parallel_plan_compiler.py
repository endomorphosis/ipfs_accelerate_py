"""Contract tests for replayable parallel execution planning (PDR-026)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.parallel_plan_compiler import (
    PARALLEL_EXECUTION_PLAN_INTERFACE,
    ParallelPlanCompilationRequest,
    ParallelPlanCompiler,
    ParallelPlanError,
    ParallelPlanIssueCode,
    ParallelPlanOutcome,
    ParallelPlanRejectedError,
    compile_parallel_execution_plan,
    replay_parallel_execution_plan,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ProviderCapacity,
)

NOW = 1_000_000


def _repository(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "tree_id": "tree:sha256:current",
        "snapshot_id": "repository-snapshot:current",
        "fencing_epoch": 40,
        "post_merge_validation": ["pytest:merged-tree"],
    }
    value.update(overrides)
    return value


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
        "validation_commands": [f"pytest:{task_id}"],
    }
    value.update(overrides)
    return value


def _compile(
    tasks: list[dict[str, object]],
    **overrides: object,
):
    values: dict[str, object] = {
        "requested_width": 4,
        "repository_snapshot": _repository(),
        "capacity_snapshot": _capacity(),
        "current_time_ms": NOW,
        "deadline_ms": NOW + 30_000,
        "budget": {
            "max_ready_width": 4,
            "max_provider_tokens": 40_000,
            "max_cost_micros": 20_000,
        },
    }
    values.update(overrides)
    return compile_parallel_execution_plan(tasks, **values)


def _codes(plan: object) -> set[ParallelPlanIssueCode]:
    return {issue.code for issue in plan.issues}  # type: ignore[attr-defined]


def test_compiles_leaf_closed_ready_waves_critical_path_and_all_assignments() -> None:
    tasks = [
        _task("scan", duration_ms=2_000, shard_key="analysis"),
        _task("proof", duration_ms=3_000, affinity_key="proof-cache"),
        _task(
            "implement",
            depends_on=["scan", "proof"],
            duration_ms=4_000,
            exclusive_group="writer:implementation",
        ),
        _task("docs", depends_on=["scan"], duration_ms=1_000),
        _task(
            "validate",
            depends_on=["implement", "docs"],
            duration_ms=2_000,
            produces=["leaf:acceptance"],
        ),
    ]

    plan = _compile(tasks, required_leaf_ids=["leaf:acceptance"])

    assert plan.outcome is ParallelPlanOutcome.DEGRADED
    assert plan.admitted and plan.ready
    assert plan.leaf_producer_closure.closed
    assert plan.leaf_producer_closure.producer_by_leaf_id["leaf:acceptance"] == "validate"
    assert plan.widths.to_dict() == {
        "requested_width": 4,
        "graph_width": 2,
        "conflict_width": 2,
        "resource_width": 2,
        "admitted_width": 2,
    }
    assert [wave.graph_ready_task_ids for wave in plan.ready_waves] == [
        ("proof", "scan"),
        ("docs", "implement"),
        ("validate",),
    ]
    assert plan.critical_path == ("proof", "implement", "validate")
    assert plan.critical_path_duration_ms == 9_000
    assert plan.estimated_makespan_ms == 9_000
    assert len(plan.assignments) == len(tasks)
    assert len({item.worktree_id for item in plan.assignments}) == len(tasks)
    assert len({item.lease_id for item in plan.assignments}) == len(tasks)
    assert [item.fence_epoch for item in plan.assignments] == [41, 42, 43, 44, 45]
    assert all(item.fence_token.startswith("fence:sha256:") for item in plan.assignments)
    assert all(item.base_revision == "tree:sha256:current" for item in plan.assignments)
    assert all(item.merge_target == "tree:sha256:current" for item in plan.assignments)
    assert all(item.lease_duration_ms == 20_000 for item in plan.assignments)
    assert all(item.heartbeat_interval_ms == 2_000 for item in plan.assignments)
    assert all(item.lease_owner_rule == "lane-owner" for item in plan.assignments)
    assert len(plan.merge_order) == len(tasks)
    assert [step.task_id for step in plan.merge_order] == [
        "proof",
        "scan",
        "docs",
        "implement",
        "validate",
    ]
    assert all(step.rollback_boundary and step.checkpoint_id for step in plan.merge_order)
    assert all("pytest:merged-tree" in step.post_merge_validation for step in plan.merge_order)
    payload = plan.to_dict()
    assert payload["interface"] == PARALLEL_EXECUTION_PLAN_INTERFACE
    assert payload["requested_width"] == 4
    assert payload["graph_width"] == 2
    assert payload["conflict_width"] == 2
    assert payload["resource_width"] == 2
    assert payload["admitted_width"] == 2
    assert payload["dependency_graph"]["critical_path"] == list(plan.critical_path)
    assert payload["conflict_graph"]["task_ids"] == sorted(task["task_id"] for task in tasks)
    assert plan.resource_feasibility.feasible
    assert plan.resource_feasibility.freshness_proved


def test_conflict_surface_is_complete_and_serializes_only_the_conflicting_wave() -> None:
    tasks = [
        _task(
            "left",
            outputs=["out/left.json"],
            predicted_paths=["src/shared.py"],
            predicted_symbols=["Shared.run"],
            interfaces=["Shared@1"],
            generated_artifacts=["generated/schema.json"],
            exclusive_paths=["locks/catalog"],
            exclusive_group="catalog-writer",
            anti_affinity_key="host-a",
        ),
        _task(
            "right",
            outputs=["out/right.json"],
            predicted_paths=["src/shared.py", "locks/catalog/item"],
            predicted_symbols=["Shared.run"],
            interfaces=["Shared@1"],
            generated_artifacts=["generated/schema.json"],
            exclusive_group="catalog-writer",
            anti_affinity_key="host-a",
        ),
        _task("independent", outputs=["out/independent.json"]),
    ]

    plan = _compile(tasks, requested_width=3)

    assert plan.outcome is ParallelPlanOutcome.DEGRADED
    conflict = next(
        item
        for item in plan.conflicts
        if {item.left_task_id, item.right_task_id} == {"left", "right"}
    )
    assert set(conflict.kinds) >= {
        "paths",
        "symbols",
        "interfaces",
        "generated_artifacts",
        "exclusive_paths",
        "exclusive_groups",
        "anti_affinity_keys",
    }
    assert plan.graph_width == 3
    assert plan.conflict_width == 2
    assert all(
        not ({"left", "right"} <= set(wave.task_ids))
        for wave in plan.execution_waves
    )
    assert any(
        "independent" in wave.task_ids and len(wave.task_ids) == 2
        for wave in plan.execution_waves
    )


def test_provider_context_tokens_cost_quota_and_affinity_are_bound() -> None:
    provider_task = _task(
        "model",
        resource_contract={
            "resource_class": "provider-llm",
            "cpu_slots": 1,
            "process_slots": 1,
        },
        provider_contract={
            "provider_requirement": "provider:alpha",
            "context_tokens": 8_000,
            "output_token_budget": 2_000,
            "quota_units": 2,
            "cost_limit_micros": 500,
            "max_provider_latency_ms": 1_000,
        },
        required_capabilities=["llm:code"],
        affinity_key="provider:alpha/cache:code",
    )

    plan = _compile([provider_task], provider_snapshots=[_provider()], requested_width=1)

    assert plan.outcome is ParallelPlanOutcome.SERIAL
    assert plan.assignments[0].provider_id == "provider:alpha"
    assert plan.assignments[0].affinity_key == "provider:alpha/cache:code"
    usage = plan.execution_waves[0].provider_usage["provider:alpha"]
    assert usage == {
        "requests": 1,
        "context_tokens": 8_000,
        "output_tokens": 2_000,
        "quota_units": 2,
        "cost_micros": 500,
    }


def test_existing_runtime_host_and_provider_capacity_contracts_are_consumed() -> None:
    host = HostResourceSnapshot(
        observed_at_ms=NOW,
        memory_total_bytes=8_000,
        memory_available_bytes=4_000,
        disk_total_bytes=20_000,
        disk_available_bytes=10_000,
        worker_limit=3,
        active_workers=1,
        available_worker_capacity=2,
        capabilities=("cpu",),
        resource_classes=("cpu-small", "provider-llm"),
    )
    provider = ProviderCapacity(
        provider_id="provider:alpha",
        healthy=True,
        quota_remaining=5,
        latency_ms=50,
        context_window_tokens=12_000,
        token_budget_remaining=5_000,
        max_concurrency=3,
        active_requests=1,
        capabilities=("llm:code",),
        observed_at_ms=NOW,
    )
    task = _task(
        "runtime-contract",
        resource_contract={
            "resource_class": "provider-llm",
            "cpu_slots": 1,
            "process_slots": 1,
            "memory_bytes": 100,
            "disk_bytes": 100,
        },
        provider_contract={
            "provider_requirement": "provider:alpha",
            "context_tokens": 4_000,
            "output_token_budget": 1_000,
            "quota_units": 1,
        },
        required_capabilities=["llm:code"],
    )

    plan = _compile(
        [task],
        capacity_snapshot=host,
        provider_snapshots={"provider:alpha": provider},
        requested_width=1,
    )

    assert plan.outcome is ParallelPlanOutcome.SERIAL
    assert plan.assignments[0].provider_id == "provider:alpha"
    assert plan.resource_width == 1


def test_deterministic_replay_is_order_stable_and_tampering_is_detected() -> None:
    tasks = [_task("b"), _task("a")]
    request = ParallelPlanCompilationRequest(
        tasks=tuple(tasks),
        requested_width=2,
        repository_snapshot=_repository(),
        capacity_snapshot=_capacity(),
        current_time_ms=NOW,
        deadline_ms=NOW + 10_000,
    )
    compiler = ParallelPlanCompiler()

    first = compiler.compile(request)
    second = compiler.compile(request)
    replayed = replay_parallel_execution_plan(first)

    assert first.plan_id == second.plan_id == replayed.plan_id
    assert first.to_json() == second.to_json() == replayed.to_json()
    assert first.deterministic_replay["input_digest"] == first.input_digest
    assert first.deterministic_replay["plan_id"] == first.plan_id

    with pytest.raises(ParallelPlanError, match="identity"):
        replace(first, plan_id="parallel-execution-plan:sha256:" + "0" * 64)


def test_output_collisions_reject_without_execution_authority() -> None:
    plan = _compile(
        [
            _task("left", outputs=["generated/api"]),
            _task("right", outputs=["generated/api/client.py"]),
        ]
    )

    assert plan.outcome is ParallelPlanOutcome.REJECTED
    assert ParallelPlanIssueCode.OUTPUT_COLLISION in _codes(plan)
    assert plan.admitted_width == 0
    assert plan.execution_waves == ()
    assert plan.assignments == ()
    assert plan.merge_order == ()

    with pytest.raises(ParallelPlanRejectedError) as raised:
        ParallelPlanCompiler().compile(
            [_task("left", outputs=["same.py"]), _task("right", outputs=["same.py"])],
            requested_width=2,
            repository_snapshot=_repository(),
            capacity_snapshot=_capacity(),
            current_time_ms=NOW,
            raise_on_rejection=True,
        )
    assert ParallelPlanIssueCode.OUTPUT_COLLISION in _codes(raised.value.plan)


@pytest.mark.parametrize(
    ("tasks", "overrides", "code"),
    [
        (
            [
                _task("a", outputs=["a.py"], predicted_paths=["policy/seal.json"]),
                _task("b", outputs=["b.py"], predicted_paths=["policy/seal.json"]),
            ],
            {"repository_snapshot": _repository(protected_paths=["policy/seal.json"])},
            ParallelPlanIssueCode.PROTECTED_BOTTLENECK,
        ),
        (
            [
                _task("a", outputs=["a.py"], predicted_paths=["vendor/lib/a.py"]),
                _task("b", outputs=["b.py"], predicted_paths=["vendor/lib/b.py"]),
            ],
            {"repository_snapshot": _repository(submodule_paths=["vendor/lib"])},
            ParallelPlanIssueCode.OVERLAPPING_SUBMODULES,
        ),
        (
            [_task("a")],
            {"capacity_snapshot": _capacity(fresh_until_ms=NOW)},
            ParallelPlanIssueCode.STALE_CAPACITY,
        ),
        (
            [_task("a", duration_ms=10_000)],
            {"deadline_ms": NOW + 9_999},
            ParallelPlanIssueCode.IMPOSSIBLE_DEADLINE,
        ),
        (
            [
                _task(
                    "a",
                    outputs=["a.py"],
                    predicted_paths=["shared.py"],
                    parallel_lane="fake-a",
                    claimed_parallel_with=["b"],
                ),
                _task(
                    "b",
                    outputs=["b.py"],
                    predicted_paths=["shared.py"],
                    parallel_lane="fake-b",
                ),
            ],
            {},
            ParallelPlanIssueCode.FAKE_LANE_LABEL,
        ),
    ],
)
def test_hard_parallelism_rejections_are_typed(
    tasks: list[dict[str, object]],
    overrides: dict[str, object],
    code: ParallelPlanIssueCode,
) -> None:
    plan = _compile(tasks, **overrides)

    assert plan.outcome is ParallelPlanOutcome.REJECTED
    assert code in _codes(plan)


@pytest.mark.parametrize(
    ("tasks", "overrides", "code"),
    [
        (
            [_task("large", resource_contract={"cpu_slots": 1, "process_slots": 1, "memory_bytes": 500})],
            {"capacity_snapshot": _capacity(memory_bytes=100)},
            ParallelPlanIssueCode.RESOURCE_INFEASIBLE,
        ),
        (
            [
                _task(
                    "context",
                    provider_contract={"provider_requirement": "provider:alpha", "context_tokens": 10_000},
                )
            ],
            {"provider_snapshots": [_provider(context_limit=9_999)]},
            ParallelPlanIssueCode.CONTEXT_INFEASIBLE,
        ),
        (
            [
                _task(
                    "tokens",
                    provider_contract={"provider_requirement": "provider:alpha", "output_token_budget": 5_000},
                )
            ],
            {"provider_snapshots": [_provider(available_tokens=4_999)]},
            ParallelPlanIssueCode.TOKEN_INFEASIBLE,
        ),
        (
            [
                _task(
                    "cost",
                    provider_contract={"provider_requirement": "provider:alpha", "cost_limit_micros": 501},
                )
            ],
            {"provider_snapshots": [_provider(available_cost_micros=500)]},
            ParallelPlanIssueCode.COST_INFEASIBLE,
        ),
        (
            [
                _task(
                    "capability",
                    provider_contract={"provider_requirement": "provider:alpha", "context_tokens": 1},
                    required_capabilities=["llm:missing"],
                )
            ],
            {"provider_snapshots": [_provider()]},
            ParallelPlanIssueCode.PROVIDER_INFEASIBLE,
        ),
    ],
)
def test_resource_provider_token_cost_and_context_infeasibility_rejects(
    tasks: list[dict[str, object]],
    overrides: dict[str, object],
    code: ParallelPlanIssueCode,
) -> None:
    plan = _compile(tasks, **overrides)

    assert plan.outcome is ParallelPlanOutcome.REJECTED
    assert code in _codes(plan)


def test_dependency_and_leaf_producer_closure_fail_closed() -> None:
    unknown = _compile([_task("a", depends_on=["missing"])])
    cycle = _compile(
        [_task("a", depends_on=["b"]), _task("b", depends_on=["a"])]
    )
    missing_leaf = _compile([_task("a")], required_leaf_ids=["leaf:required"])

    assert ParallelPlanIssueCode.UNKNOWN_DEPENDENCY in _codes(unknown)
    assert ParallelPlanIssueCode.DEPENDENCY_CYCLE in _codes(cycle)
    assert ParallelPlanIssueCode.MISSING_LEAF_PRODUCER in _codes(missing_leaf)
    assert all(plan.outcome is ParallelPlanOutcome.REJECTED for plan in (unknown, cycle, missing_leaf))


def test_resource_serialization_is_included_in_deadline_feasibility() -> None:
    plan = _compile(
        [_task("a", duration_ms=10_000), _task("b", duration_ms=10_000)],
        requested_width=2,
        capacity_snapshot=_capacity(cpu_slots=1, process_slots=1),
        deadline_ms=NOW + 15_000,
    )

    assert plan.critical_path_duration_ms == 10_000
    assert plan.estimated_makespan_ms == 20_000
    assert plan.outcome is ParallelPlanOutcome.REJECTED
    assert ParallelPlanIssueCode.IMPOSSIBLE_DEADLINE in _codes(plan)


def test_serial_degraded_parallel_and_review_only_outcomes_are_distinct() -> None:
    serial = _compile([_task("one")], requested_width=1)
    degraded = _compile([_task("a"), _task("b")], requested_width=4)
    parallel = _compile([_task("a"), _task("b")], requested_width=2)
    request_limited = _compile([_task("a"), _task("b")], requested_width=1)
    review = compile_parallel_execution_plan(
        [{"task_id": "review", "review_only": True}],
        requested_width=4,
        review_only=True,
    )

    assert serial.outcome is ParallelPlanOutcome.SERIAL
    assert serial.admitted_width == 1
    assert degraded.outcome is ParallelPlanOutcome.DEGRADED
    assert degraded.admitted_width == 2
    assert parallel.outcome is ParallelPlanOutcome.PARALLEL
    assert parallel.admitted_width == 2
    assert request_limited.outcome is ParallelPlanOutcome.SERIAL
    assert request_limited.resource_width == 2
    assert request_limited.admitted_width == 1
    assert review.outcome is ParallelPlanOutcome.REVIEW_ONLY
    assert review.ready and review.admitted
    assert review.admitted_width == 0
    assert review.execution_waves == ()
    assert review.assignments == ()
    assert review.merge_order == ()


def test_stale_provider_capacity_and_fake_authoritative_lane_reject() -> None:
    provider_task = _task(
        "model",
        provider_contract={"provider_requirement": "provider:alpha", "context_tokens": 1},
    )
    stale = _compile(
        [provider_task],
        provider_snapshots=[_provider(fresh_until_ms=NOW)],
    )
    fake = _compile([_task("claimed", lane_authoritative=True, parallel_lane="pdr-fast")])

    assert ParallelPlanIssueCode.STALE_CAPACITY in _codes(stale)
    assert ParallelPlanIssueCode.FAKE_LANE_LABEL in _codes(fake)
    assert stale.outcome is fake.outcome is ParallelPlanOutcome.REJECTED
