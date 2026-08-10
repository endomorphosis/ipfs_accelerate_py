"""DCR-064: schedule plans within leases, lanes, and resource budgets.

Acceptance:
* Same plan/policy yields the same schedule identity and wave layout.
* Overlapping writes never execute concurrently.
* Starvation/deadlock tests terminate.
* Assignments cover lanes, leases, fencing tokens, timeouts, retry budgets,
  and validation resources deterministically.
* Runtime model calls remain 0; write authority is never granted.
* Strict sharding cannot override dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.repair_resource_scheduler import (
    DCR_RESOURCE_SCHEDULE_EVIDENCE,
    DEFAULT_RESOURCE_SCHEDULES_REL,
    PATH_LEASE_INTERFACE,
    REPAIR_RESOURCE_SCHEDULE_INTERFACE,
    ConflictKind,
    PathLease,
    PathLeasePlan,
    RepairResourceSchedule,
    RepairResourceScheduler,
    RepairResourceSchedulerError,
    ResourceSchedulePolicy,
    ScheduleDisposition,
    SchedulableNode,
    build_conflict_graph,
    materialize_resource_schedules,
    schedule_repair_resources,
    simulate_schedule_execution,
    topological_order,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _node(
    node_id: str,
    *,
    write_set: tuple[str, ...] = (),
    depends_on: tuple[str, ...] = (),
    owner_root: str = "ipfs-accelerate",
    resource_class: str = "cpu-medium",
    kind: str = "operator_apply",
    validation_ref: str = "",
    endpoints: tuple[str, ...] = (),
    duration_ms: int = 10_000,
    timeout_ms: int = 60_000,
    retry_budget: int = 2,
    shard_hint: str = "",
    exclusive_group: str = "",
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "kind": kind,
        "write_set": list(write_set),
        "depends_on": list(depends_on),
        "owner_root": owner_root,
        "resource_class": resource_class,
        "validation_ref": validation_ref or f"validation:{node_id}",
        "endpoints": list(endpoints),
        "duration_ms": duration_ms,
        "timeout_ms": timeout_ms,
        "retry_budget": retry_budget,
        "shard_hint": shard_hint,
        "exclusive_group": exclusive_group,
    }


def _plan(*nodes: dict[str, object], plan_id: str = "plan:dcr064") -> dict[str, object]:
    return {"plan_id": plan_id, "nodes": list(nodes)}


def _policy(**overrides: object) -> ResourceSchedulePolicy:
    base: dict[str, object] = {
        "policy_id": "policy:dcr064-test",
        "max_lanes": 2,
        "max_wave_width": 2,
        "max_solver_concurrency": 1,
        "max_validation_concurrency": 1,
        "max_root_writers": 1,
        "default_timeout_ms": 120_000,
        "default_retry_budget": 1,
        "lease_duration_ms": 300_000,
        "base_fence_epoch": 3,
    }
    base.update(overrides)
    return ResourceSchedulePolicy.from_dict(base)


def test_interfaces_and_evidence_are_stable() -> None:
    assert PATH_LEASE_INTERFACE == "PathLease@1"
    assert REPAIR_RESOURCE_SCHEDULE_INTERFACE == "RepairResourceSchedule@1"
    assert DCR_RESOURCE_SCHEDULE_EVIDENCE == "dcr/resource-schedule@1"
    assert RepairResourceScheduler is not None
    assert PathLeasePlan is not None


def test_same_plan_and_policy_yield_same_schedule() -> None:
    plan = _plan(
        _node(
            "node:a",
            write_set=("external/ipfs_accelerate/a.py",),
            resource_class="cpu-proof-solver",
        ),
        _node(
            "node:b",
            write_set=("external/ipfs_datasets/b.py",),
            owner_root="ipfs-datasets",
            resource_class="cpu-medium",
        ),
        _node(
            "node:c",
            write_set=("external/ipfs_accelerate/c.py",),
            depends_on=("node:a",),
            resource_class="cpu-proof-solver",
        ),
    )
    policy = _policy()
    first = schedule_repair_resources(plan, policy=policy)
    second = RepairResourceScheduler(policy=policy).schedule(plan)
    third = schedule_repair_resources(plan, policy=policy.to_dict())

    assert first.ok is True
    assert first.disposition is ScheduleDisposition.SCHEDULED
    assert first.schedule_cid == second.schedule_cid == third.schedule_cid
    assert first.to_dict() == second.to_dict() == third.to_dict()
    assert first.runtime_model_calls == 0
    assert first.grants_write_authority is False

    # Policy identity is content-addressed and stable.
    assert policy.policy_cid == ResourceSchedulePolicy.from_dict(policy.to_dict()).policy_cid


def test_overlapping_writes_never_execute_concurrently() -> None:
    plan = _plan(
        _node(
            "node:left",
            write_set=("external/ipfs_accelerate/pkg/shared.py",),
            resource_class="cpu-medium",
        ),
        _node(
            "node:right",
            write_set=(
                "external/ipfs_accelerate/pkg/shared.py",
                "external/ipfs_accelerate/pkg/other.py",
            ),
            resource_class="cpu-medium",
        ),
        _node(
            "node:child",
            write_set=("external/ipfs_accelerate/pkg/shared/child.py",),
            resource_class="cpu-medium",
        ),
    )
    schedule = schedule_repair_resources(plan, policy=_policy(max_lanes=4, max_wave_width=4))
    assert schedule.ok is True

    concurrent = set(schedule.concurrent_pairs())
    # Exact and prefix-overlapping writers must not share a wave.
    assert ("node:left", "node:right") not in concurrent
    assert ("node:child", "node:left") not in concurrent
    assert ("node:child", "node:right") not in concurrent

    path_edges = [
        edge
        for edge in schedule.conflict_graph
        if ConflictKind.PATH.value in edge.kinds
    ]
    assert path_edges
    simulation = simulate_schedule_execution(schedule)
    assert simulation["terminated"] is True
    assert simulation["ok"] is True
    assert simulation["concurrent_write_violations"] == []


def test_independent_roots_may_run_in_parallel() -> None:
    plan = _plan(
        _node(
            "node:accel",
            write_set=("external/ipfs_accelerate/a.py",),
            owner_root="ipfs-accelerate",
            resource_class="cpu-medium",
        ),
        _node(
            "node:datasets",
            write_set=("external/ipfs_datasets/b.py",),
            owner_root="ipfs-datasets",
            resource_class="cpu-medium",
        ),
    )
    schedule = schedule_repair_resources(
        plan,
        policy=_policy(
            max_lanes=2,
            max_wave_width=2,
            max_solver_concurrency=2,
            max_validation_concurrency=2,
        ),
    )
    assert schedule.ok is True
    assert ("node:accel", "node:datasets") in set(schedule.concurrent_pairs())
    wave0 = schedule.waves[0]
    assert set(wave0.node_ids) == {"node:accel", "node:datasets"}


def test_solver_and_root_budgets_serialize() -> None:
    plan = _plan(
        _node(
            "node:s1",
            write_set=("external/ipfs_accelerate/s1.py",),
            resource_class="cpu-proof-solver",
        ),
        _node(
            "node:s2",
            write_set=("external/ipfs_accelerate/s2.py",),
            resource_class="cpu-proof-solver",
        ),
    )
    schedule = schedule_repair_resources(
        plan,
        policy=_policy(
            max_lanes=4,
            max_wave_width=4,
            max_solver_concurrency=1,
            max_root_writers=1,
        ),
    )
    assert schedule.ok is True
    # Same root writers + exclusive solver slot => serialized.
    assert schedule.concurrent_pairs() == ()
    assert len(schedule.waves) == 2


def test_dependencies_dominate_shard_hints() -> None:
    plan = _plan(
        _node(
            "node:parent",
            write_set=("external/ipfs_accelerate/parent.py",),
            shard_hint="shard:forced-together",
        ),
        _node(
            "node:child",
            write_set=("external/ipfs_accelerate/child.py",),
            depends_on=("node:parent",),
            shard_hint="shard:forced-together",
        ),
    )
    schedule = schedule_repair_resources(
        plan,
        policy=_policy(max_lanes=4, max_wave_width=4, max_root_writers=2),
    )
    assert schedule.ok is True
    assignment = schedule.assignment_map()
    assert assignment["node:parent"].wave < assignment["node:child"].wave
    # Same shard hint is preserved on both, but waves still respect deps.
    assert assignment["node:parent"].shard_id == "shard:forced-together"
    assert assignment["node:child"].shard_id == "shard:forced-together"


def test_path_leases_and_fencing_tokens_are_assigned() -> None:
    plan = _plan(
        _node(
            "node:write",
            write_set=("external/ipfs_accelerate/write.py",),
            resource_class="cpu-medium",
        ),
        _node(
            "node:validate",
            write_set=(),
            kind="validation",
            resource_class="cpu-validation",
            depends_on=("node:write",),
        ),
    )
    schedule = schedule_repair_resources(plan, policy=_policy(base_fence_epoch=9))
    assert schedule.ok is True
    write = schedule.assignment_map()["node:write"]
    assert write.lease_id.startswith("lease:")
    assert write.fencing_token.startswith("fence:")
    assert write.fence_epoch == 9
    assert write.timeout_ms == 120_000
    assert write.retry_budget == 1
    assert write.validation_resource == "validation:node:write"

    lease_plan = schedule.path_lease_plan
    assert isinstance(lease_plan, PathLeasePlan)
    lease = lease_plan.lease_for_node("node:write")
    assert isinstance(lease, PathLease)
    assert lease.INTERFACE == PATH_LEASE_INTERFACE
    assert lease.contains("external/ipfs_accelerate/write.py")
    assert lease.fence_epoch == 9

    # Validation-only node has no write lease.
    validate = schedule.assignment_map()["node:validate"]
    assert validate.lease_id == ""
    assert validate.fencing_token == ""
    assert validate.validation_resource.startswith("validation:")


def test_evidence_subset_contains_required_fields() -> None:
    plan = _plan(
        _node("node:a", write_set=("external/ipfs_accelerate/a.py",)),
        _node(
            "node:b",
            write_set=("external/ipfs_datasets/b.py",),
            owner_root="ipfs-datasets",
            depends_on=("node:a",),
        ),
    )
    schedule = schedule_repair_resources(plan, policy=_policy())
    evidence = schedule.evidence_subset()
    assert evidence["evidence_id"] == DCR_RESOURCE_SCHEDULE_EVIDENCE
    assert evidence["schedule_cid"] == schedule.schedule_cid
    assert evidence["lane_shard"]
    assert evidence["conflict_graph"] is not None
    assert evidence["lease_fence"]
    assert evidence["budgets"]
    assert evidence["critical_path"]
    assert evidence["runtime_model_calls"] == 0
    assert evidence["grants_write_authority"] is False
    for item in evidence["lane_shard"]:
        assert "lane" in item and "shard" in item and "wave" in item


def test_critical_path_tracks_longest_dependency_chain() -> None:
    plan = _plan(
        _node("node:a", write_set=("external/ipfs_accelerate/a.py",), duration_ms=5_000),
        _node(
            "node:b",
            write_set=("external/ipfs_accelerate/b.py",),
            depends_on=("node:a",),
            duration_ms=7_000,
        ),
        _node(
            "node:c",
            write_set=("external/ipfs_datasets/c.py",),
            owner_root="ipfs-datasets",
            duration_ms=20_000,
        ),
    )
    schedule = schedule_repair_resources(
        plan,
        policy=_policy(max_lanes=2, max_wave_width=2, max_root_writers=2),
    )
    assert schedule.ok is True
    # a->b is 12_000; c alone is 20_000 — critical path is c when independent,
    # but if a and c start together then b follows, makespan is max(20k, 5k+7k)=20k
    # critical path by dependency duration is the longest chain: either a->b or c.
    assert schedule.critical_path_duration_ms == 20_000
    assert schedule.critical_path[-1] in {"node:b", "node:c"}
    if schedule.critical_path[-1] == "node:b":
        assert schedule.critical_path == ("node:a", "node:b")
    else:
        assert schedule.critical_path == ("node:c",)


def test_starvation_and_deadlock_simulation_terminates() -> None:
    # Contended writers + long dependency chain + exclusive solver slots.
    nodes = [
        _node(
            "node:seed",
            write_set=("external/ipfs_accelerate/seed.py",),
            resource_class="cpu-proof-solver",
            duration_ms=1_000,
        )
    ]
    for index in range(12):
        nodes.append(
            _node(
                f"node:w{index}",
                write_set=(
                    (
                        "external/ipfs_accelerate/shared/area.py"
                        if index % 2 == 0
                        else f"external/ipfs_accelerate/leaf_{index}.py"
                    ),
                ),
                depends_on=("node:seed",) if index < 4 else (f"node:w{index - 4}",),
                resource_class="cpu-proof-solver" if index % 3 == 0 else "cpu-medium",
                duration_ms=1_000 + index,
                exclusive_group="group-a" if index % 5 == 0 else "",
            )
        )
    plan = _plan(*nodes, plan_id="plan:dcr064-starvation")
    schedule = schedule_repair_resources(
        plan,
        policy=_policy(
            max_lanes=2,
            max_wave_width=2,
            max_solver_concurrency=1,
            max_root_writers=1,
        ),
    )
    assert schedule.ok is True
    assert len(schedule.assignments) == len(nodes)

    simulation = simulate_schedule_execution(schedule, max_steps=64)
    assert simulation["terminated"] is True
    assert simulation["ok"] is True
    assert len(simulation["completed"]) == len(nodes)
    assert simulation["concurrent_write_violations"] == []

    # Re-running the pure scheduler remains bounded and identical.
    again = schedule_repair_resources(plan, policy=_policy(
        max_lanes=2,
        max_wave_width=2,
        max_solver_concurrency=1,
        max_root_writers=1,
    ))
    assert again.schedule_cid == schedule.schedule_cid


def test_endpoint_conflicts_are_serialized() -> None:
    plan = _plan(
        _node(
            "node:e1",
            write_set=("external/ipfs_accelerate/e1.py",),
            endpoints=("endpoint:shared-api",),
            owner_root="ipfs-accelerate",
        ),
        _node(
            "node:e2",
            write_set=("external/ipfs_datasets/e2.py",),
            owner_root="ipfs-datasets",
            endpoints=("endpoint:shared-api",),
        ),
    )
    schedule = schedule_repair_resources(
        plan,
        policy=_policy(
            max_lanes=4,
            max_wave_width=4,
            max_root_writers=2,
            serialize_endpoints=True,
        ),
    )
    assert schedule.ok is True
    assert schedule.concurrent_pairs() == ()
    endpoint_edges = [
        edge
        for edge in schedule.conflict_graph
        if ConflictKind.ENDPOINT.value in edge.kinds
    ]
    assert endpoint_edges


def test_cycle_is_rejected_without_hanging() -> None:
    plan = _plan(
        _node("node:a", write_set=("external/ipfs_accelerate/a.py",), depends_on=("node:b",)),
        _node("node:b", write_set=("external/ipfs_accelerate/b.py",), depends_on=("node:a",)),
    )
    schedule = schedule_repair_resources(plan, policy=_policy())
    assert schedule.ok is False
    assert schedule.disposition is ScheduleDisposition.REJECTED
    assert schedule.reason_codes
    simulation = simulate_schedule_execution(schedule)
    assert simulation["terminated"] is True
    assert simulation["ok"] is False


def test_topological_order_and_conflict_helpers() -> None:
    nodes = (
        SchedulableNode.from_dict(
            _node("node:a", write_set=("external/ipfs_accelerate/a.py",))
        ),
        SchedulableNode.from_dict(
            _node(
                "node:b",
                write_set=("external/ipfs_accelerate/a.py",),
                depends_on=("node:a",),
            )
        ),
    )
    order = topological_order(nodes)
    assert order == ("node:a", "node:b")
    edges = build_conflict_graph(nodes, _policy())
    assert any(ConflictKind.PATH.value in edge.kinds for edge in edges)

    with pytest.raises(RepairResourceSchedulerError):
        topological_order(
            (
                SchedulableNode.from_dict(
                    _node("node:a", depends_on=("node:b",), write_set=("external/ipfs_accelerate/a.py",))
                ),
                SchedulableNode.from_dict(
                    _node("node:b", depends_on=("node:a",), write_set=("external/ipfs_accelerate/b.py",))
                ),
            )
        )


def test_schedule_round_trip_and_path_lease_identity() -> None:
    plan = _plan(
        _node("node:a", write_set=("external/ipfs_accelerate/a.py",)),
        _node(
            "node:b",
            write_set=("external/ipfs_datasets/b.py",),
            owner_root="ipfs-datasets",
        ),
    )
    schedule = schedule_repair_resources(plan, policy=_policy())
    rebuilt = RepairResourceSchedule.from_dict(schedule.to_dict())
    assert rebuilt.schedule_cid == schedule.schedule_cid
    assert rebuilt.evidence_subset()["schedule_cid"] == schedule.schedule_cid

    lease = schedule.path_lease_plan.leases[0]
    rebuilt_lease = PathLease.from_dict(lease.to_dict())
    assert rebuilt_lease.content_id == lease.content_id
    assert rebuilt_lease.overlaps(lease) is True


def test_materialize_resource_schedules_fixture(tmp_path: Path) -> None:
    destination = tmp_path / "resource-schedules.json"
    payload = materialize_resource_schedules(destination=destination)
    assert destination.is_file()
    assert payload["evidence_id"] == DCR_RESOURCE_SCHEDULE_EVIDENCE
    assert payload["interfaces"]["path_lease"] == PATH_LEASE_INTERFACE
    assert payload["interfaces"]["repair_resource_schedule"] == REPAIR_RESOURCE_SCHEDULE_INTERFACE
    assert payload["runtime_model_calls"] == 0
    assert payload["grants_write_authority"] is False
    assert payload["schedule"]["disposition"] == ScheduleDisposition.SCHEDULED.value
    assert payload["simulation"]["terminated"] is True
    assert payload["simulation"]["ok"] is True
    assert DEFAULT_RESOURCE_SCHEDULES_REL.endswith("resource-schedules.json")


def test_integrates_with_proof_carrying_repair_plan_when_available() -> None:
    try:
        from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
            compile_proof_carrying_repair_plan,
        )
    except Exception:  # pragma: no cover - optional integration surface
        pytest.skip("proof-carrying repair DAG unavailable")

    compilation = compile_proof_carrying_repair_plan(
        {
            "proposal_id": "t-accel-sched",
            "operator": {
                "operator_id": "doctor-operator:add_registration@1",
                "kind": "add_registration",
            },
            "write_paths": [
                "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/op.py"
            ],
            "before_hashes": {
                "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/op.py":
                    "sha256:before-0"
            },
            "applicability_proof_cid": "proof:applicability:t-accel-sched",
            "rollback_ref": "rollback:t-accel-sched",
            "expected_proof_transition": "proof:t-accel-sched->admitted",
            "resource_class": "cpu-proof-solver",
            "include_pin_update": True,
        },
        plan_id="plan:dcr064-from-dag",
    )
    assert compilation.ok is True
    assert compilation.plan is not None
    schedule = schedule_repair_resources(compilation.plan, policy=_policy(max_lanes=2))
    assert schedule.ok is True
    assert schedule.plan_id == "plan:dcr064-from-dag"
    assert len(schedule.assignments) == len(compilation.plan.nodes)
    # Dependency order from the DAG is preserved in wave start offsets.
    assignment = schedule.assignment_map()
    order = list(compilation.plan.topological_order)
    for node in compilation.plan.nodes:
        for dep in node.depends_on:
            # Dependencies always finish in an earlier wave; shard labels cannot
            # collapse a producer into the same concurrent batch as its consumer.
            assert assignment[dep].wave < assignment[node.node_id].wave
    assert order[0] in assignment
    assert content_identity({"tag": "stable"})  # exercise content identity import
