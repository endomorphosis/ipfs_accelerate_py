"""End-to-end regression coverage for adaptive, fenced parallel slices."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.execution_plan import (
    AdaptiveExecutionScheduler, CapacitySnapshot, ExecutionClaimConflictError,
    ExecutionSlice, ExecutionSliceViolationError, ExecutionTask, InvocationBudget,
)


NOW = 1_000_000


def _capacity(*, lanes: int = 3) -> CapacitySnapshot:
    return CapacitySnapshot("capacity:current", lanes, lanes, NOW, NOW + 60_000)


def _task(name: str, *, paths: tuple[str, ...], depends: tuple[str, ...] = (), scope: tuple[str, ...] = ()) -> ExecutionTask:
    return ExecutionTask(name, dependencies=depends, declared_paths=paths, scope_paths=scope)


def test_disjoint_tasks_overlap_in_wall_clock_time(tmp_path: Path) -> None:
    scheduler = AdaptiveExecutionScheduler(tmp_path / "ledger.sqlite")
    tasks = (_task("A", paths=("src/a.py",)), _task("B", paths=("src/b.py",)))
    plan = scheduler.compile(plan_revision="revision:1", tasks=tasks, budget=InvocationBudget(2), capacity=_capacity(lanes=2), now_ms=NOW)
    entered = threading.Barrier(2)
    def runner(task: ExecutionTask, _claim: object) -> tuple[str, ...]:
        entered.wait(timeout=2)
        time.sleep(0.04)
        return task.declared_paths
    attempts = scheduler.execute(plan, tasks, runner)
    assert plan.admitted_lanes == 2
    assert max(item.started_at_ms for item in attempts) < min(item.finished_at_ms for item in attempts)
    assert all(item.effect.accepted for item in attempts)


def test_conflicts_dependencies_and_capacity_loss_are_serialized(tmp_path: Path) -> None:
    scheduler = AdaptiveExecutionScheduler(tmp_path / "ledger.sqlite")
    tasks = (
        _task("A", paths=("src/shared.py",)),
        _task("B", paths=("src/shared.py",)),
        _task("C", paths=("src/c.py",), depends=("A",)),
    )
    plan = scheduler.compile(plan_revision="revision:1", tasks=tasks, budget=InvocationBudget(3), capacity=_capacity(lanes=1), now_ms=NOW)
    assert plan.ready_task_cids == ("A", "B")
    assert plan.admitted_lanes == 1
    assert plan.selected_task_cids == ("A",)
    assert ("A", "B") in plan.conflict_pairs


def test_empty_and_restarted_slices_cannot_claim_another_lane_task(tmp_path: Path) -> None:
    scheduler = AdaptiveExecutionScheduler(tmp_path / "ledger.sqlite")
    empty = ExecutionSlice("revision:1", "lane-0", (), "capacity:current")
    other = ExecutionSlice("revision:1", "lane-1", ("B",), "capacity:current")
    scheduler.ledger.register_slices((empty, other))
    with pytest.raises(ExecutionSliceViolationError):
        scheduler.claim(empty, "B", now_ms=NOW)
    with pytest.raises(ExecutionSliceViolationError):
        scheduler.claim(other, "A", now_ms=NOW)
    assert scheduler.claim(other, "B", now_ms=NOW).task_cid == "B"


def test_same_revision_work_steal_is_explicit_and_fenced(tmp_path: Path) -> None:
    scheduler = AdaptiveExecutionScheduler(tmp_path / "ledger.sqlite")
    donor = ExecutionSlice("revision:1", "lane-0", ("A",), "capacity:current")
    scheduler.ledger.register_slices((donor,))
    recipient = scheduler.steal(donor_slice=donor, recipient_lane_id="lane-1", task_cid="A")
    with pytest.raises(ExecutionSliceViolationError):
        scheduler.claim(donor, "A", now_ms=NOW)
    assert scheduler.claim(recipient, "A", now_ms=NOW).lane_id == "lane-1"


def test_duplicate_claims_effects_and_undeclared_overlaps_are_fenced_and_replanned(tmp_path: Path) -> None:
    scheduler = AdaptiveExecutionScheduler(tmp_path / "ledger.sqlite")
    task_a = _task("A", paths=("src/a.py",))
    task_b = _task("B", paths=("src/b.py",))
    slice_a = ExecutionSlice("revision:1", "lane-0", ("A",), "capacity:current")
    slice_b = ExecutionSlice("revision:1", "lane-1", ("B",), "capacity:current")
    scheduler.ledger.register_slices((slice_a, slice_b))
    claim_a = scheduler.claim(slice_a, "A", now_ms=NOW)
    with pytest.raises(ExecutionClaimConflictError):
        scheduler.claim(slice_a, "A", now_ms=NOW + 1)
    first = scheduler.record_effect(claim_a, task_a, ("src/a.py",))
    assert first.accepted
    assert scheduler.record_effect(claim_a, task_a, ("src/a.py",)).effect_id == first.effect_id
    claim_b = scheduler.claim(slice_b, "B", now_ms=NOW)
    fenced = scheduler.record_effect(claim_b, task_b, ("src/a.py",))
    assert not fenced.accepted and fenced.replan_required
    assert fenced.reason == "undeclared_scope"
    assert scheduler.ledger.replan_attempts("revision:1", "B") == 1

    # Two independent workers discovering the same undeclared scope are both
    # fenced; neither can convert that surprise into an accepted effect.
    task_c = _task("C", paths=("src/c.py",))
    task_d = _task("D", paths=("src/d.py",))
    slice_c = ExecutionSlice("revision:1", "lane-2", ("C",), "capacity:current")
    slice_d = ExecutionSlice("revision:1", "lane-3", ("D",), "capacity:current")
    scheduler.ledger.register_slices((slice_c, slice_d))
    assert scheduler.record_effect(scheduler.claim(slice_c, "C", now_ms=NOW), task_c, ("src/hidden.py",)).replan_required
    assert scheduler.record_effect(scheduler.claim(slice_d, "D", now_ms=NOW), task_d, ("src/hidden.py",)).replan_required
    assert scheduler.ledger.replan_attempts("revision:1", "C") == 1
    assert scheduler.ledger.replan_attempts("revision:1", "D") == 1
