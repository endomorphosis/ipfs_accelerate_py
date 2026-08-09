"""Production adapter coverage for canonical plan-bound parallel waves."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.entrypoints import execution_plan
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    InvocationBudget as CanonicalInvocationBudget,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.execution_plan import (
    ConfiguredBoardExecutionSlice,
    ConfiguredBoardExecutionSlices,
    ExecutionPlanError,
    PlanSliceReassignment,
    ProductionParallelPlanAdapter,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    PlanRevisionStore,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    ImplementationProposal,
    ProposalFindingCode,
    ProposalValidationPolicy,
    validate_implementation_proposal,
)

NOW = 1_000_000


def test_slice_canonicalization_preserves_exact_id_cid_pairs() -> None:
    execution_slice = ConfiguredBoardExecutionSlice(
        lane_index=0,
        lane_id="lane-0",
        task_ids=("TASK-B", "TASK-A"),
        task_cids=("cid-for-b", "cid-for-a"),
        plan_root_cid="plan-root",
        compiler_plan_id="compiler-plan",
        capacity_snapshot_id="capacity",
        repository_tree_id="tree",
    )
    assert execution_slice.task_pairs == (
        ("TASK-A", "cid-for-a"),
        ("TASK-B", "cid-for-b"),
    )

    manifest = ConfiguredBoardExecutionSlices(
        board_namespace="board",
        plan_root_cid="plan-root",
        compiler_plan_id="compiler-plan",
        capacity_snapshot_id="capacity",
        repository_tree_id="tree",
        source_head="head",
        task_source_revision="tasks",
        configuration_root="configuration",
        slices=(execution_slice,),
    )
    payload = manifest.to_dict()
    payload["slices"][0]["task_ids"] = ["TASK-B", "TASK-A"]
    payload["slices"][0]["task_cids"] = ["cid-for-b", "cid-for-a"]
    with pytest.raises(ExecutionPlanError, match="round trip"):
        ConfiguredBoardExecutionSlices.from_dict(payload)


def test_slice_and_reassignment_authority_decoders_are_closed_and_exact() -> None:
    execution_slice = ConfiguredBoardExecutionSlice(
        lane_index=0,
        lane_id="lane-0",
        task_ids=("TASK-A",),
        task_cids=("cid-a",),
        plan_root_cid="plan-root",
        compiler_plan_id="compiler-plan",
        capacity_snapshot_id="capacity",
        repository_tree_id="tree",
    )
    manifest = ConfiguredBoardExecutionSlices(
        board_namespace="board",
        plan_root_cid="plan-root",
        compiler_plan_id="compiler-plan",
        capacity_snapshot_id="capacity",
        repository_tree_id="tree",
        source_head="head",
        task_source_revision="tasks",
        configuration_root="configuration",
        slices=(execution_slice,),
    )
    base_manifest = manifest.to_dict()
    manifest_mutations = (
        lambda value: value.update({"extra": "smuggled"}),
        lambda value: value.__setitem__("wave_index", True),
        lambda value: value.__setitem__("wave_index", 0.0),
        lambda value: value.__setitem__("wave_index", "0"),
        lambda value: value.__setitem__("slices", ["foreign"]),
        lambda value: value["slices"][0].update({"extra": "smuggled"}),
        lambda value: value["slices"][0].__setitem__("lane_index", True),
        lambda value: value["slices"][0].__setitem__("lane_index", 0.0),
        lambda value: value["slices"][0].__setitem__("lane_index", "0"),
        lambda value: value["slices"][0].__setitem__(
            "plan_root_cid", "foreign-plan-root"
        ),
    )
    for mutate in manifest_mutations:
        payload = copy.deepcopy(base_manifest)
        mutate(payload)
        with pytest.raises(ExecutionPlanError):
            ConfiguredBoardExecutionSlices.from_dict(payload)

    reassignment = PlanSliceReassignment(
        revision_cid="revision",
        plan_root_cid="plan-root",
        slice_manifest_cid="manifest",
        slice_id=execution_slice.slice_id,
        donor_lane_id="lane-0",
        recipient_lane_id="lane-1",
        task_ids=execution_slice.task_ids,
        task_cids=execution_slice.task_cids,
        generation=1,
        prior_reassignment_cid="",
        donor_process_birth_cid="birth",
        attempt_absence_cid="attempt",
        claim_absence_cid="claim",
    )
    base_reassignment = reassignment.to_dict()
    reassignment_mutations = (
        lambda value: value.update({"extra": "smuggled"}),
        lambda value: value.__setitem__("generation", True),
        lambda value: value.__setitem__("generation", 1.0),
        lambda value: value.__setitem__("generation", "1"),
        lambda value: value.__setitem__("task_ids", "TASK-A"),
        lambda value: value.__setitem__("prior_reassignment_cid", None),
    )
    for mutate in reassignment_mutations:
        payload = copy.deepcopy(base_reassignment)
        mutate(payload)
        with pytest.raises(ExecutionPlanError):
            PlanSliceReassignment.from_dict(payload)


def _host(*, lanes: int) -> dict[str, object]:
    return {
        "observed_at_ms": NOW,
        "worker_limit": lanes,
        "available_worker_capacity": lanes,
        "active_workers": 0,
        "cpu_percent": 1,
        "memory_percent": 1,
        "disk_percent": 1,
        "memory_total_bytes": 1_000_000_000,
        "memory_available_bytes": 900_000_000,
        "disk_total_bytes": 1_000_000_000,
        "disk_available_bytes": 900_000_000,
        "capabilities": ["cpu"],
        "resource_classes": ["cpu-small", "coordinator"],
    }


def _providers(*, lanes: int) -> tuple[dict[str, object], ...]:
    return (
        {
            "provider_id": "grok_cli",
            "healthy": True,
            "max_concurrency": lanes,
            "active_requests": 0,
            "quota_remaining": 10,
            "token_budget_remaining": 100_000,
            "context_window_tokens": 100_000,
            "observed_at_ms": NOW,
            "max_age_ms": 60_000,
        },
    )


def _task(
    task_id: str,
    path: str,
    *,
    depends_on: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "canonical_task_cid": f"task-cid:{task_id}",
        "depends_on": list(depends_on),
        "outputs": [path],
        "predicted_files": [path],
        "resource_class": "cpu-small",
        "provider_id": "grok_cli",
        "validation_commands": ["python -m pytest -q"],
    }


def _compile(
    tmp_path: Path,
    tasks: tuple[dict[str, object], ...],
    *,
    requested_lanes: int = 4,
    live_lanes: int = 4,
    completed: tuple[str, ...] = (),
):
    adapter = ProductionParallelPlanAdapter(
        PlanRevisionStore(tmp_path / "plan-revision-store")
    )
    host = _host(lanes=live_lanes)
    providers = _providers(lanes=live_lanes)
    return adapter.compile_wave(
        board_namespace="test-board",
        plan_root_cid="plan-root:test",
        tasks=tasks,
        budget=CanonicalInvocationBudget(max_lanes=requested_lanes),
        repository_snapshot={
            "tree_id": "git-tree:test",
            "merge_target": "main",
            "protected_paths": [],
        },
        capacity_snapshot={**host, "host": host, "providers": list(providers)},
        provider_snapshots=providers,
        completed_task_ids=completed,
        post_merge_validation=("python -m pytest -q",),
        source_head="head:test",
        task_source_revision="task-source:test",
        configuration_root="configuration:test",
        current_time_ms=NOW,
    )


def test_adapter_uses_canonical_budget_and_owns_no_private_ledger() -> None:
    assert execution_plan.InvocationBudget is CanonicalInvocationBudget
    assert not hasattr(execution_plan, "ExecutionLedger")
    assert not hasattr(execution_plan, "AdaptiveExecutionScheduler")
    source = inspect.getsource(execution_plan).casefold()
    assert "sqlite" not in source


def test_disjoint_ready_tasks_materialize_exact_id_cid_slices(
    tmp_path: Path,
) -> None:
    plan, manifest = _compile(
        tmp_path,
        (_task("A", "src/a.py"), _task("B", "src/b.py")),
        requested_lanes=2,
        live_lanes=2,
    )
    assert plan.admitted
    assert tuple(item.task_ids for item in manifest.slices) == (("A",), ("B",))
    assert tuple(item.task_cids for item in manifest.slices) == (
        ("task-cid:A",),
        ("task-cid:B",),
    )
    assert len({item.slice_id for item in manifest.slices}) == 2


def test_declared_path_and_dependency_conflicts_never_share_first_wave(
    tmp_path: Path,
) -> None:
    plan, manifest = _compile(
        tmp_path,
        (
            _task("A", "src/shared"),
            _task("B", "src/shared/child.py"),
            _task("C", "src/c.py", depends_on=("A",)),
        ),
        requested_lanes=3,
        live_lanes=3,
    )
    first_wave = {task_id for item in manifest.slices for task_id in item.task_ids}
    assert not {"A", "B"}.issubset(first_wave)
    assert "C" not in first_wave
    assert any({item.left_task_id, item.right_task_id} == {"A", "B"} for item in plan.conflicts)


def test_live_capacity_shrink_narrows_the_materialized_wave(tmp_path: Path) -> None:
    tasks = (_task("A", "src/a.py"), _task("B", "src/b.py"))
    _wide_plan, wide = _compile(
        tmp_path / "wide", tasks, requested_lanes=2, live_lanes=2
    )
    _narrow_plan, narrow = _compile(
        tmp_path / "narrow", tasks, requested_lanes=2, live_lanes=1
    )
    assert len(wide.slices) == 2
    assert len(narrow.slices) == 1


def test_empty_population_and_noncanonical_budget_fail_closed(
    tmp_path: Path,
) -> None:
    adapter = ProductionParallelPlanAdapter(PlanRevisionStore(tmp_path / "store"))
    with pytest.raises(ExecutionPlanError, match="nonempty"):
        adapter.compile_wave(
            board_namespace="test-board",
            plan_root_cid="plan-root:test",
            tasks=(),
            budget=CanonicalInvocationBudget(max_lanes=1),
            repository_snapshot={"tree_id": "git-tree:test"},
            capacity_snapshot=_host(lanes=1),
            source_head="head:test",
            task_source_revision="task-source:test",
            configuration_root="configuration:test",
            current_time_ms=NOW,
        )
    with pytest.raises(TypeError, match="canonical InvocationBudget"):
        adapter.compile_wave(
            board_namespace="test-board",
            plan_root_cid="plan-root:test",
            tasks=(_task("A", "src/a.py"),),
            budget=object(),  # type: ignore[arg-type]
            repository_snapshot={"tree_id": "git-tree:test"},
            capacity_snapshot=_host(lanes=1),
            source_head="head:test",
            task_source_revision="task-source:test",
            configuration_root="configuration:test",
            current_time_ms=NOW,
        )


def _proposal_result(
    *,
    old_path: str,
    new_path: str,
    allowed_paths: tuple[str, ...],
    change_kind: DiffChangeKind = DiffChangeKind.MODIFY,
):
    entry = CandidateDiffEntry(
        old_path=old_path,
        new_path=new_path,
        change_kind=change_kind,
        before_source="VALUE = 1\n",
        after_source="VALUE = 2\n",
    )
    proposal = ImplementationProposal(
        task_id="TEST-TASK",
        accepted_plan_id="plan:test",
        repository_id="repo:test",
        repository_tree_id="tree:test",
        objective_id="goal:test",
        baseline_id="baseline:test",
        candidate_diff=(entry,),
        declared_paths=tuple(sorted({old_path, new_path})),
    )
    policy = ProposalValidationPolicy(
        allowed_paths=allowed_paths,
        task_owned_paths=allowed_paths,
        expected_task_id="TEST-TASK",
        expected_plan_id="plan:test",
        expected_repository_id="repo:test",
        expected_repository_tree_id="tree:test",
        expected_objective_id="goal:test",
    )
    return validate_implementation_proposal(proposal, policy=policy)


def test_production_proposal_gate_matches_exact_parent_and_rename_endpoints() -> None:
    assert _proposal_result(
        old_path="src/pkg/file.py",
        new_path="src/pkg/file.py",
        allowed_paths=("src/pkg/file.py",),
    ).accepted
    assert _proposal_result(
        old_path="src/pkg/file.py",
        new_path="src/pkg/file.py",
        allowed_paths=("src/pkg",),
    ).accepted
    rename_rejection = _proposal_result(
        old_path="src/old.py",
        new_path="src/new.py",
        allowed_paths=("src/new.py",),
        change_kind=DiffChangeKind.RENAME,
    )
    assert not rename_rejection.accepted
    assert ProposalFindingCode.PATH_OUTSIDE_SCOPE in {
        item.code for item in rename_rejection.receipt.findings
    }
    assert _proposal_result(
        old_path="src/old.py",
        new_path="src/new.py",
        allowed_paths=("src/old.py", "src/new.py"),
        change_kind=DiffChangeKind.RENAME,
    ).accepted


def test_planned_disjoint_tasks_cannot_both_enqueue_one_undeclared_actual_path(
    tmp_path: Path,
) -> None:
    _plan, manifest = _compile(
        tmp_path,
        (_task("A", "src/a.py"), _task("B", "src/b.py")),
        requested_lanes=2,
        live_lanes=2,
    )
    assert len(manifest.slices) == 2
    rejected = tuple(
        _proposal_result(
            old_path="src/hidden.py",
            new_path="src/hidden.py",
            allowed_paths=(declared,),
        )
        for declared in ("src/a.py", "src/b.py")
    )
    assert all(not result.accepted for result in rejected)
    assert all(
        ProposalFindingCode.PATH_OUTSIDE_SCOPE
        in {item.code for item in result.receipt.findings}
        for result in rejected
    )


def test_glob_like_mutation_envelope_fails_closed_before_parallel_admission(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExecutionPlanError, match="glob-like"):
        _compile(
            tmp_path,
            (_task("A", "src/**/*.py"), _task("B", "docs/b.md")),
            requested_lanes=2,
            live_lanes=2,
        )
