"""PDR-081: Compile benchmark/Doctor residuals into bounded derived goals/tasks.

Covers:
* at most 8 goals / 24 tasks per epoch and 48 open tasks
* proposals carry source roots, hierarchy, minimal files/context,
  acceptance/validation, resource/conflict/dependencies, stop policy
* duplicates and semantically unchanged failures back off
* exact population replay is a no-op
* candidates cannot edit anchors, authorize themselves, lower thresholds,
  or mark complete
* generated work enters the separate DuckDB source only after independent
  formal plan / admission / parallel compilation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.planner_doctor_refill import (
    DEFAULT_MAX_GOALS_PER_EPOCH,
    DEFAULT_MAX_OPEN_TASKS,
    DEFAULT_MAX_TASKS_PER_EPOCH,
    DEFAULT_PARENT_GOAL_ID,
    DEFAULT_PROTECTED_ANCHORS,
    DEFAULT_STOP_POLICY,
    DERIVED_RUNTIME_SOURCE_GATE,
    PLANNER_DOCTOR_REFILL_INTERFACE,
    PRODUCER_ID,
    REFILL_AUTHORIZES_COMPLETION,
    REFILL_AUTHORIZES_MUTATION,
    REFILL_AUTHORIZES_SEED_BOARD_EDIT,
    REFILL_AUTHORIZES_SELF_AUTHORIZATION,
    REFILL_AUTHORIZES_THRESHOLD_LOWER,
    DerivedResidual,
    PlannerDoctorRefill,
    PlannerDoctorRefillAuthorityError,
    PlannerDoctorRefillBoundsError,
    PlannerDoctorRefillDisposition,
    PlannerDoctorRefillError,
    PlannerDoctorRefillMemory,
    PlannerDoctorRefillPolicy,
    ResidualSourceKind,
    create_planner_doctor_refill,
    refill_planner_doctor_residuals,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    ObjectiveWorkKind,
)


duckdb = pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_task_source import (  # noqa: E402
    DERIVED_RUNTIME_SOURCE_ROLE,
    DuckDBTaskSource,
    TaskSourceIntegrityError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_residual(**overrides: object) -> DerivedResidual:
    base: dict[str, object] = {
        "issue_id": "issue:contract-mismatch-1",
        "source_kind": ResidualSourceKind.DOCTOR,
        "obligation_id": "obligation:proof-1",
        "root_id": "tree:sha256:fixture-root",
        "source_root": "tree:sha256:fixture-root",
        "attempt_id": "attempt:1",
        "predicted_files": (
            "ipfs_accelerate_py/agent_supervisor/objectives/planner_doctor_refill.py",
        ),
        "context_paths": (
            "ipfs_accelerate_py/agent_supervisor/objectives/planner_doctor_refill.py",
        ),
        "title": "Resolve residual issue:contract-mismatch-1",
        "rationale": "Open Doctor residual after fixed-point iteration.",
        "validation_commands": (
            "python -m pytest test/api/test_agent_supervisor_planner_doctor_refill.py -q",
        ),
        "acceptance_criteria": (
            "resolve residual with evidence",
            "no completion authority",
            "no seed board mutation",
        ),
    }
    base.update(overrides)
    return DerivedResidual(**base)  # type: ignore[arg-type]


def make_benchmark_residual(**overrides: object) -> DerivedResidual:
    base: dict[str, object] = {
        "issue_id": "benchmark:gap:quality-1",
        "source_kind": ResidualSourceKind.BENCHMARK,
        "root_id": "tree:sha256:benchmark-root",
        "source_root": "tree:sha256:benchmark-root",
        "attempt_id": "attempt:bench-1",
        "predicted_files": (
            "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py",
        ),
        "title": "Close benchmark quality residual",
        "validation_commands": (
            "python -m pytest test/api/test_agent_supervisor_planner_doctor_refill.py -q",
        ),
    }
    base.update(overrides)
    return DerivedResidual(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Interface / authority surface
# ---------------------------------------------------------------------------


def test_interfaces_and_authority_constants() -> None:
    assert PLANNER_DOCTOR_REFILL_INTERFACE == "PlannerDoctorRefill@1"
    assert PRODUCER_ID == "planner-doctor-refill@1"
    assert REFILL_AUTHORIZES_COMPLETION is False
    assert REFILL_AUTHORIZES_MUTATION is False
    assert REFILL_AUTHORIZES_SEED_BOARD_EDIT is False
    assert REFILL_AUTHORIZES_SELF_AUTHORIZATION is False
    assert REFILL_AUTHORIZES_THRESHOLD_LOWER is False
    assert DERIVED_RUNTIME_SOURCE_GATE == "PDR-081"
    assert DEFAULT_MAX_GOALS_PER_EPOCH == 8
    assert DEFAULT_MAX_TASKS_PER_EPOCH == 24
    assert DEFAULT_MAX_OPEN_TASKS == 48
    service = create_planner_doctor_refill()
    assert service.INTERFACE == PLANNER_DOCTOR_REFILL_INTERFACE
    assert service.producer_id == PRODUCER_ID


def test_residual_rejects_completion_authority_metadata() -> None:
    with pytest.raises(PlannerDoctorRefillAuthorityError):
        make_residual(metadata={"completion_authority": True})


def test_residual_rejects_threshold_lower_metadata() -> None:
    with pytest.raises(PlannerDoctorRefillAuthorityError):
        make_residual(metadata={"lower_threshold": True})


def test_policy_enforces_hard_ceilings() -> None:
    with pytest.raises(PlannerDoctorRefillBoundsError):
        PlannerDoctorRefillPolicy(max_goals_per_epoch=9)
    with pytest.raises(PlannerDoctorRefillBoundsError):
        PlannerDoctorRefillPolicy(max_tasks_per_epoch=25)
    with pytest.raises(PlannerDoctorRefillBoundsError):
        PlannerDoctorRefillPolicy(max_open_tasks=49)


def test_policy_rejects_authority_flags_in_payload() -> None:
    with pytest.raises(PlannerDoctorRefillAuthorityError):
        PlannerDoctorRefillPolicy.from_dict({"completion_authority": True})


# ---------------------------------------------------------------------------
# Fixed-point / empty / hierarchy
# ---------------------------------------------------------------------------


def test_successful_fixed_point_emits_no_work() -> None:
    receipt = refill_planner_doctor_residuals(
        fixed_point={
            "complete": True,
            "residual_free": True,
            "residual_finding_ids": [],
        }
    )
    assert receipt.disposition is PlannerDoctorRefillDisposition.FIXED_POINT_CLOSED
    assert receipt.emits_work is False
    assert receipt.goals == ()
    assert receipt.tasks == ()
    assert receipt.completion_authority is False
    assert receipt.mutation_authority is False


def test_empty_input_disposition() -> None:
    receipt = refill_planner_doctor_residuals([])
    assert receipt.disposition is PlannerDoctorRefillDisposition.EMPTY_INPUT
    assert receipt.emits_work is False


def test_compiles_doctor_and_benchmark_residuals_with_hierarchy() -> None:
    receipt = refill_planner_doctor_residuals(
        doctor_residuals=[make_residual()],
        benchmark_residuals=[make_benchmark_residual()],
        repository_tree_id="tree:sha256:fixture-root",
    )
    assert receipt.disposition in {
        PlannerDoctorRefillDisposition.ADMITTED,
        PlannerDoctorRefillDisposition.COMPILED,
    }
    assert receipt.emits_work is True
    assert 1 <= len(receipt.goals) <= 8
    assert 1 <= len(receipt.tasks) <= 24
    assert len(receipt.tasks) == 2
    # Hierarchy: root + subgoals
    assert any(not goal.is_subgoal for goal in receipt.goals)
    assert any(goal.is_subgoal for goal in receipt.goals)
    for task in receipt.tasks:
        assert task.source_root
        assert task.predicted_files
        assert task.acceptance_criteria
        assert task.validation_commands
        assert task.stop_policy
        assert task.resource_class
        assert task.goal_id
        assert task.goal_cid
    for proposal in receipt.work_proposals:
        assert proposal.kind is ObjectiveWorkKind.TASK
        assert proposal.predicted_files
        assert proposal.context_paths
        assert proposal.acceptance_subset
        assert proposal.validation_commands
    assert receipt.admission is not None
    assert receipt.admission.admitted is True
    assert receipt.compilation is not None
    assert receipt.compilation.formal_plan_id
    assert receipt.compilation.source_identity
    assert receipt.compilation.parallel_plan_digest
    assert receipt.derived_runtime_admitted is True
    assert receipt.to_dict()["derived_runtime_gate"] == "PDR-081"
    assert receipt.to_dict()["stop_policy"] == DEFAULT_STOP_POLICY
    assert receipt.policy.stop_policy == DEFAULT_STOP_POLICY


def test_proposals_carry_resource_conflict_dependencies_and_stop_policy() -> None:
    residual = make_residual(
        dependencies=("dep:task-a",),
        conflicts=("conflict:path-overlap",),
        stop_policy=DEFAULT_STOP_POLICY,
    )
    receipt = refill_planner_doctor_residuals(
        [residual], repository_tree_id="tree:sha256:fixture-root"
    )
    task = receipt.tasks[0]
    assert "dep:task-a" in task.dependencies
    assert "conflict:path-overlap" in task.conflicts
    assert task.stop_policy == DEFAULT_STOP_POLICY
    proposal = receipt.work_proposals[0]
    assert proposal.resource_class
    assert proposal.conflicts or task.conflicts


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_max_tasks_per_epoch_bound() -> None:
    residuals = [
        make_residual(
            issue_id=f"issue:bound-{index}",
            attempt_id=f"attempt:{index}",
            obligation_id=f"obligation:{index}",
        )
        for index in range(30)
    ]
    receipt = refill_planner_doctor_residuals(
        residuals, repository_tree_id="tree:sha256:fixture-root"
    )
    assert len(receipt.tasks) <= 24
    assert len(receipt.goals) <= 8


def test_open_work_ceiling_blocks_emission() -> None:
    memory = PlannerDoctorRefillMemory(open_task_count=48)
    receipt = refill_planner_doctor_residuals(
        [make_residual()],
        memory=memory,
        repository_tree_id="tree:sha256:fixture-root",
    )
    assert receipt.disposition is PlannerDoctorRefillDisposition.OPEN_WORK_CEILING
    assert receipt.tasks == ()


# ---------------------------------------------------------------------------
# Dedupe / backoff / replay
# ---------------------------------------------------------------------------


def test_duplicates_collapse() -> None:
    residual = make_residual()
    receipt = refill_planner_doctor_residuals(
        [residual, residual, residual],
        repository_tree_id="tree:sha256:fixture-root",
    )
    assert len(receipt.tasks) == 1
    assert residual.identity_key in receipt.duplicate_identity_keys


def test_unchanged_failure_backs_off() -> None:
    residual = make_residual(unchanged_failure=True)
    receipt = refill_planner_doctor_residuals(
        [residual], repository_tree_id="tree:sha256:fixture-root"
    )
    assert receipt.disposition is PlannerDoctorRefillDisposition.UNCHANGED_BACKOFF
    assert receipt.emits_work is False
    assert residual.identity_key in receipt.backoff_identity_keys


def test_identical_fingerprint_backs_off_on_second_pass() -> None:
    residual = make_residual()
    first = refill_planner_doctor_residuals(
        [residual], repository_tree_id="tree:sha256:fixture-root"
    )
    assert first.emits_work is True
    second = refill_planner_doctor_residuals(
        [residual],
        memory=first.next_memory,
        repository_tree_id="tree:sha256:fixture-root",
    )
    assert second.disposition is PlannerDoctorRefillDisposition.UNCHANGED_BACKOFF
    assert second.emits_work is False


def test_exact_source_identity_replay_is_noop() -> None:
    residual = make_residual()
    first = refill_planner_doctor_residuals(
        [residual], repository_tree_id="tree:sha256:fixture-root"
    )
    # Force memory to remember source identity without fingerprint backoff.
    memory = PlannerDoctorRefillMemory(
        entries=(),
        open_task_count=0,
        last_source_identity=first.compilation.source_identity if first.compilation else "",
        last_plan_root_cid=first.compilation.plan_root_cid if first.compilation else "",
    )
    second = refill_planner_doctor_residuals(
        [residual],
        memory=memory,
        repository_tree_id="tree:sha256:fixture-root",
    )
    assert second.disposition is PlannerDoctorRefillDisposition.REPLAY_NOOP
    assert "exact_source_identity_replay" in second.reason_codes


# ---------------------------------------------------------------------------
# Protected anchors / authority
# ---------------------------------------------------------------------------


def test_protected_anchor_paths_are_rejected() -> None:
    residual = make_residual(
        predicted_files=(DEFAULT_PROTECTED_ANCHORS[0],),
        context_paths=(DEFAULT_PROTECTED_ANCHORS[0],),
    )
    receipt = refill_planner_doctor_residuals(
        [residual], repository_tree_id="tree:sha256:fixture-root"
    )
    assert receipt.tasks == ()
    assert any(
        decision.disposition.value == "anchor_rejected"
        for decision in receipt.decisions
    )


def test_receipt_never_grants_completion_or_mutation() -> None:
    receipt = refill_planner_doctor_residuals(
        [make_residual()], repository_tree_id="tree:sha256:fixture-root"
    )
    payload = receipt.to_dict()
    assert payload["completion_authority"] is False
    assert payload["mutation_authority"] is False
    assert payload["seed_board_edit"] is False
    assert payload["threshold_lower_authority"] is False
    assert payload["self_authorization"] is False
    if receipt.admission is not None:
        assert receipt.admission.to_dict()["completion_authority"] is False


# ---------------------------------------------------------------------------
# DuckDB materialization (independent gates)
# ---------------------------------------------------------------------------


def test_materialize_requires_independent_gates(tmp_path: Path) -> None:
    source = DuckDBTaskSource(tmp_path / "derived.duckdb")
    residual = make_residual()
    receipt = refill_planner_doctor_residuals(
        [residual],
        repository_tree_id="tree:sha256:fixture-root",
        duckdb_source=source,
        materialize=True,
    )
    assert receipt.disposition in {
        PlannerDoctorRefillDisposition.MATERIALIZED,
        PlannerDoctorRefillDisposition.REPLAY_NOOP,
    }
    assert receipt.materialization
    assert receipt.materialization.get("source_role") == DERIVED_RUNTIME_SOURCE_ROLE
    assert receipt.materialization.get("mutates_seed_board") is False
    assert receipt.materialization.get("admission_receipt_cid")
    assert receipt.materialization.get("parallel_plan_digest")
    assert receipt.materialization.get("formal_plan_id")
    snapshot = source.snapshot()
    assert snapshot.task_count >= 1
    assert snapshot.goal_count >= 1


def test_materialize_replay_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "derived-replay.duckdb"
    residual = make_residual()
    first = refill_planner_doctor_residuals(
        [residual],
        repository_tree_id="tree:sha256:fixture-root",
        duckdb_source=DuckDBTaskSource(path),
        materialize=True,
    )
    assert first.disposition is PlannerDoctorRefillDisposition.MATERIALIZED
    # Fresh memory so fingerprint backoff does not fire; DuckDB itself is no-op.
    second = refill_planner_doctor_residuals(
        [residual],
        repository_tree_id="tree:sha256:fixture-root",
        duckdb_source=DuckDBTaskSource(path),
        materialize=True,
        memory=PlannerDoctorRefillMemory(),
    )
    # Either exact source-identity replay at the refill layer or DuckDB replay.
    assert second.disposition in {
        PlannerDoctorRefillDisposition.REPLAY_NOOP,
        PlannerDoctorRefillDisposition.MATERIALIZED,
    }
    if second.disposition is PlannerDoctorRefillDisposition.MATERIALIZED:
        assert second.materialization.get("replayed") is True
        assert second.materialization.get("changed") is False


def test_duckdb_materialize_derived_runtime_rejects_missing_gates(
    tmp_path: Path,
) -> None:
    source = DuckDBTaskSource(tmp_path / "missing-gates.duckdb")
    with pytest.raises(TaskSourceIntegrityError):
        source.materialize_derived_runtime(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/formal-plan-input@1",
                "repository_tree_id": "tree:x",
                "objectives": [],
                "taskboard": [],
            },
            formal_plan_id="",
            source_identity="",
            parallel_plan_digest="",
            admission_receipt_cid="",
        )


def test_duckdb_materialize_derived_runtime_rejects_identity_mismatch(
    tmp_path: Path,
) -> None:
    residual = make_residual()
    compiled = refill_planner_doctor_residuals(
        [residual], repository_tree_id="tree:sha256:fixture-root"
    )
    assert compiled.compilation is not None
    formal_input = dict(compiled.compilation.formal_input)
    source = DuckDBTaskSource(tmp_path / "mismatch.duckdb")
    with pytest.raises(TaskSourceIntegrityError):
        source.materialize_derived_runtime(
            formal_input,
            formal_plan_id="baguqeera000000000000000000000000000000000000000000000000000",
            source_identity=compiled.compilation.source_identity,
            parallel_plan_digest=compiled.compilation.parallel_plan_digest,
            admission_receipt_cid=(
                compiled.admission.admission_receipt_cid
                if compiled.admission
                else "admission:missing"
            ),
            repository_tree_id="tree:sha256:fixture-root",
        )


def test_duckdb_materialize_derived_runtime_rejects_protected_anchor(
    tmp_path: Path,
) -> None:
    residual = make_residual()
    compiled = refill_planner_doctor_residuals(
        [residual], repository_tree_id="tree:sha256:fixture-root"
    )
    assert compiled.compilation is not None
    formal_input = dict(compiled.compilation.formal_input)
    # Inject a protected-anchor effect into the formal input.
    for task in formal_input.get("taskboard") or []:
        task["effects"] = [
            {
                "effect_id": "effect:bad",
                "operation": "assign",
                "fluent_id": "output:anchor",
                "path": DEFAULT_PROTECTED_ANCHORS[0],
                "value": "modify",
            }
        ]
        task["predicted_files"] = [DEFAULT_PROTECTED_ANCHORS[0]]
    source = DuckDBTaskSource(tmp_path / "anchor.duckdb")
    with pytest.raises(TaskSourceIntegrityError):
        source.materialize_derived_runtime(
            formal_input,
            formal_plan_id=compiled.compilation.formal_plan_id,
            source_identity=compiled.compilation.source_identity,
            parallel_plan_digest=compiled.compilation.parallel_plan_digest,
            admission_receipt_cid=(
                compiled.admission.admission_receipt_cid
                if compiled.admission
                else "admission:missing"
            ),
            repository_tree_id="tree:sha256:fixture-root",
        )


def test_service_facade_compile_and_materialize(tmp_path: Path) -> None:
    service = PlannerDoctorRefill()
    residual = make_residual(issue_id="issue:service-1", attempt_id="attempt:svc")
    compiled = service.compile_and_admit(
        [residual], repository_tree_id="tree:sha256:fixture-root"
    )
    assert compiled.admission is not None and compiled.admission.admitted
    # New service instance so fingerprint memory does not block materialize.
    materializer = create_planner_doctor_refill()
    materialized = materializer.materialize(
        [residual],
        duckdb_source=DuckDBTaskSource(tmp_path / "service.duckdb"),
        repository_tree_id="tree:sha256:fixture-root",
    )
    assert materialized.disposition in {
        PlannerDoctorRefillDisposition.MATERIALIZED,
        PlannerDoctorRefillDisposition.REPLAY_NOOP,
    }


def test_parent_goal_default_is_pdr_g090() -> None:
    assert DEFAULT_PARENT_GOAL_ID == "PDR-G090"
    receipt = refill_planner_doctor_residuals(
        [make_residual()], repository_tree_id="tree:sha256:fixture-root"
    )
    assert any(goal.goal_id.startswith("PDR-G090") for goal in receipt.goals)
