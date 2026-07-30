from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.contract_findings import (
    AppendOutcome,
    AppendReceipt,
    CallSlice,
    CallSliceStep,
    ContractFindingLedger,
    EvidenceReferences,
    FindingAdmissionState,
    build_contract_finding,
)
from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import (
    ClaimLevel,
    EvidenceFreshness,
    FindingSeverity,
    FindingStatus,
)
from ipfs_accelerate_py.agent_supervisor.symbolic_finding_refill import (
    REFILL_AUTHORIZES_COMPLETION,
    REFILL_AUTHORIZES_EXECUTION,
    REFILL_IDEMPOTENCY_SCHEMA,
    SYMBOLIC_REFILL_EPOCH_SCHEMA,
    BacklogRefinery,
    FindingDisposition,
    HealthyExhaustionReceipt,
    RefillAncestryError,
    RefillBinding,
    RefillGoal,
    RefillReason,
    RefillState,
    SupervisorBacklogSnapshot,
    SymbolicFindingRefillPolicy,
    TaskKind,
    refill_symbolic_findings,
)


def _binding() -> RefillBinding:
    return RefillBinding(
        repository_id="repository:alpha",
        tree_id="tree:abc",
        policy_id="policy:implementation",
        policy_revision="policy:v1",
        objective_forest_id="forest:alpha",
        objective_forest_revision="forest:v1",
        refinement_goal_id="root",
    )


def _root() -> RefillGoal:
    return RefillGoal(goal_id="root", title="Symbolic assurance")


def _family_goal(family: str = "schema-drift", *, goal_id: str = "family") -> RefillGoal:
    return RefillGoal(
        goal_id=goal_id,
        title=f"Repair {family}",
        root_cause_family=family,
        semantic_key=f"goal:{family}:{goal_id}",
        parent_goal_id="root",
        ancestor_goal_ids=("root",),
        depth=1,
    )


def _finding(
    index: int,
    *,
    family: str = "schema-drift",
    status: FindingStatus = FindingStatus.CONTRACT_BROKEN,
    tree_id: str = "tree:abc",
    policy_revision: str = "policy:v1",
    path: str | None = None,
):
    symbol = f"pkg.api.call{index}"
    output_path = path or f"ipfs_accelerate_py/pkg/call{index}.py"
    return build_contract_finding(
        claim_level=ClaimLevel.MODEL_DISPROVED,
        status=status,
        severity=FindingSeverity.HIGH,
        confidence_millionths=950_000,
        freshness=EvidenceFreshness.CURRENT,
        repositories=("repository:alpha",),
        symbols=(symbol,),
        interfaces=(f"mcp://pkg/call{index}",),
        expected_contract_cid=f"expected:{index}",
        observed_contract_cid=f"observed:{index}",
        root_cause_family=family,
        merge_fate=symbol,
        summary=f"{family} at {symbol}",
        call_slice=CallSlice(
            steps=(
                CallSliceStep(
                    symbol=symbol,
                    interface=f"mcp://pkg/call{index}",
                    repository_id="repository:alpha",
                    path=output_path,
                ),
            )
        ),
        evidence=EvidenceReferences(
            counterexample_cids=(f"counterexample:{index}",),
            artifact_cids=(f"artifact:{index}",),
        ),
        assumptions=("hermetic analysis",),
        analyzer_versions={"symbolic-checker": "1"},
        remediation_scope=(output_path,),
        tree_id=tree_id,
        policy_revision=policy_revision,
        repository_observation_id=f"observation:{index}",
        verdict="violated",
    )


def _ledger(tmp_path, *findings):
    ledger = ContractFindingLedger(tmp_path / "findings.jsonl")
    return ledger, tuple(ledger.append(finding) for finding in findings)


def _refill(
    ledger,
    receipts,
    *,
    goals=None,
    tasks=(),
    state=None,
    now=100,
    dependencies=None,
    conclusive_healthy=False,
):
    binding = _binding()
    healthy_exhaustion = (
        HealthyExhaustionReceipt(
            repository_id=binding.repository_id,
            tree_id=binding.tree_id,
            policy_id=binding.policy_id,
            policy_revision=binding.policy_revision,
            objective_forest_id=binding.objective_forest_id,
            objective_forest_revision=binding.objective_forest_revision,
            conclusive=True,
            healthy=True,
            coverage_complete=True,
            evidence_cids=("evidence:complete",),
        )
        if conclusive_healthy
        else None
    )
    return refill_symbolic_findings(
        ledger=ledger,
        receipts=receipts,
        binding=binding,
        goals=goals if goals is not None else (_root(), _family_goal()),
        tasks=tasks,
        state=state or RefillState(),
        now_epoch=now,
        dependencies=dependencies or {},
        healthy_exhaustion=healthy_exhaustion,
    )


def test_materializes_exact_family_with_stable_identity_and_binding(tmp_path):
    finding = _finding(1)
    ledger, receipts = _ledger(tmp_path, finding)

    outcome = _refill(ledger, receipts)
    replayed_from_same_snapshot = _refill(ledger, receipts, now=101)

    assert outcome.reason is RefillReason.REFILLED
    assert not outcome.new_goals
    assert len(outcome.new_tasks) == 1
    task = outcome.new_tasks[0]
    assert task.goal_id == "family"
    assert task.ancestor_goal_ids == ("root",)
    assert task.output_paths == ("ipfs_accelerate_py/pkg/call1.py",)
    assert task.repository_id == _binding().repository_id
    assert task.tree_id == _binding().tree_id
    assert task.policy_id == _binding().policy_id
    assert task.policy_revision == _binding().policy_revision
    assert task.objective_forest_id == _binding().objective_forest_id
    assert task.objective_forest_revision == _binding().objective_forest_revision
    assert task.task_id == replayed_from_same_snapshot.new_tasks[0].task_id
    assert task.semantic_key == replayed_from_same_snapshot.new_tasks[0].semantic_key
    assert outcome.idempotency_id == replayed_from_same_snapshot.idempotency_id
    assert outcome.refill_epoch_id != replayed_from_same_snapshot.refill_epoch_id
    assert not task.write_authorized
    assert not REFILL_AUTHORIZES_EXECUTION
    assert not REFILL_AUTHORIZES_COMPLETION
    assert outcome.diagnostics[0].disposition is FindingDisposition.MATERIALIZED


def test_epoch_and_idempotency_evidence_prove_stateful_replay_noop(tmp_path):
    finding = _finding(1)
    ledger, receipts = _ledger(tmp_path, finding)

    admitted = _refill(ledger, receipts)
    replay = _refill(
        ledger,
        receipts,
        state=admitted.state,
        now=admitted.state.next_allowed_epoch,
    )

    assert admitted.evidence_methods == (
        SYMBOLIC_REFILL_EPOCH_SCHEMA,
        REFILL_IDEMPOTENCY_SCHEMA,
    )
    epoch_record, idempotency_record = admitted.evidence_records()
    assert epoch_record["schema"] == SYMBOLIC_REFILL_EPOCH_SCHEMA
    assert epoch_record["epoch_id"] == admitted.refill_epoch_id
    assert epoch_record["binding"] == _binding().to_record()
    assert epoch_record["fresh_receipt_ids"] == (receipts[0].receipt_id,)
    assert epoch_record["processed_receipt_ids"] == (receipts[0].receipt_id,)
    assert epoch_record["emitted_task_ids"] == (admitted.new_tasks[0].task_id,)
    assert epoch_record["changed"]
    assert idempotency_record["schema"] == REFILL_IDEMPOTENCY_SCHEMA
    assert idempotency_record["idempotency_id"] == admitted.idempotency_id
    assert not idempotency_record["replay_noop"]

    assert replay.reason is RefillReason.DIAGNOSTICS_ONLY
    assert not replay.new_goals
    assert not replay.new_tasks
    assert replay.idempotency_id == admitted.idempotency_id
    assert replay.refill_epoch_id != admitted.refill_epoch_id
    replay_epoch, replay_idempotency = replay.evidence_records()
    assert replay_epoch["prior_state_id"] != epoch_record["prior_state_id"]
    assert replay_epoch["emitted_task_ids"] == ()
    assert replay_idempotency["replay_receipt_ids"] == (
        receipts[0].receipt_id,
    )
    assert replay_idempotency["resolved_task_ids"] == (
        admitted.new_tasks[0].task_id,
    )
    assert replay_idempotency["replay_noop"]

    restored_taskboard = _refill(
        ledger,
        receipts,
        tasks=(admitted.new_tasks[0],),
        state=RefillState(),
        now=102,
    )
    assert not restored_taskboard.new_tasks
    assert restored_taskboard.state.semantic_task_ids == (
        (finding.semantic_key_id, admitted.new_tasks[0].task_id),
    )
    assert restored_taskboard.idempotency_evidence is not None
    assert restored_taskboard.idempotency_evidence.resolved_task_ids == (
        admitted.new_tasks[0].task_id,
    )
    assert restored_taskboard.idempotency_evidence.replay_noop


def test_supervisor_snapshot_keeps_refill_bound_to_objective_heap(tmp_path):
    finding = _finding(1, family="new-family")
    ledger, receipts = _ledger(tmp_path, finding)
    snapshot = SupervisorBacklogSnapshot(
        binding=_binding(),
        goals=(_root(),),
    )

    outcome = BacklogRefinery(ledger).refill(snapshot, receipts, now_epoch=100)

    assert snapshot.goals == (_root(),)
    assert snapshot.tasks == ()
    assert snapshot.state == RefillState()
    assert len(outcome.new_goals) == 1
    assert outcome.new_goals[0].parent_goal_id == _root().goal_id
    assert outcome.new_tasks[0].objective_forest_id == (
        snapshot.binding.objective_forest_id
    )
    assert outcome.new_tasks[0].objective_forest_revision == (
        snapshot.binding.objective_forest_revision
    )
    assert outcome.evidence_methods == snapshot.evidence_methods

    with pytest.raises(ValueError, match="exact refinement goal"):
        SupervisorBacklogSnapshot(
            binding=_binding(),
            goals=(_family_goal(),),
        )


def test_bounded_refinement_caps_children_and_preserves_ancestry(tmp_path):
    findings = tuple(_finding(index, family=f"family-{index}") for index in range(4))
    ledger, receipts = _ledger(tmp_path, *findings)

    outcome = _refill(ledger, receipts, goals=(_root(),))

    assert len(outcome.new_goals) == 3
    assert len(outcome.new_tasks) == 3
    assert all(goal.parent_goal_id == "root" for goal in outcome.new_goals)
    assert all(goal.ancestor_goal_ids == ("root",) for goal in outcome.new_goals)
    assert all(goal.depth == 1 for goal in outcome.new_goals)
    assert {
        diagnostic.disposition for diagnostic in outcome.diagnostics
    } >= {FindingDisposition.CHILD_LIMIT}


def test_ambiguous_family_and_invalid_imported_forest_fail_closed(tmp_path):
    ledger, receipts = _ledger(tmp_path, _finding(1))
    ambiguous_goals = (
        _root(),
        _family_goal(goal_id="family-a"),
        _family_goal(goal_id="family-b"),
    )

    outcome = _refill(ledger, receipts, goals=ambiguous_goals)

    assert not outcome.new_tasks
    assert not outcome.new_goals
    assert outcome.diagnostics[0].disposition is FindingDisposition.AMBIGUOUS

    invalid_goal = replace(
        _family_goal(),
        parent_goal_id="missing",
        ancestor_goal_ids=("missing",),
    )
    with pytest.raises(RefillAncestryError):
        _refill(ledger, receipts, goals=(_root(), invalid_goal))


def test_low_watermark_open_ceiling_and_cooldown_gate_refill(tmp_path):
    ledger, receipts = _ledger(tmp_path, _finding(1))
    open_tasks = tuple({"task_id": f"open-{index}", "status": "open"} for index in range(4))

    enough_work = _refill(ledger, receipts, tasks=open_tasks)
    assert enough_work.reason is RefillReason.THRESHOLD_SATISFIED
    assert not enough_work.new_tasks

    ceiling_tasks = tuple({"task_id": f"open-{index}", "status": "open"} for index in range(12))
    at_ceiling = _refill(ledger, receipts, tasks=ceiling_tasks)
    assert at_ceiling.reason is RefillReason.OPEN_WORK_CEILING

    cooling_down = _refill(
        ledger,
        receipts,
        state=RefillState(next_allowed_epoch=101),
        now=100,
    )
    assert cooling_down.reason is RefillReason.COOLDOWN


def test_fresh_receipts_outrank_replay_prefix_and_unchanged_replay_backs_off(tmp_path):
    findings = tuple(_finding(index) for index in range(9))
    ledger, receipts = _ledger(tmp_path, *findings)
    state = RefillState(
        seen_receipt_ids=tuple(receipt.receipt_id for receipt in receipts[:8]),
        last_sequence=receipts[7].sequence,
    )

    fresh = _refill(ledger, receipts, state=state)
    assert len(fresh.new_tasks) == 1
    assert fresh.new_tasks[0].finding_semantic_key == findings[8].semantic_key_id

    first_replay = _refill(
        ledger,
        receipts,
        state=fresh.state,
        now=fresh.state.next_allowed_epoch,
    )
    assert first_replay.reason is RefillReason.DIAGNOSTICS_ONLY
    assert not first_replay.new_tasks
    assert {diagnostic.disposition for diagnostic in first_replay.diagnostics} == {
        FindingDisposition.REPLAY
    }

    unchanged = _refill(
        ledger,
        receipts,
        state=first_replay.state,
        now=first_replay.state.next_allowed_epoch,
    )
    assert unchanged.reason is RefillReason.DIAGNOSTICS_ONLY
    assert {diagnostic.disposition for diagnostic in unchanged.diagnostics} == {
        FindingDisposition.UNCHANGED_BACKOFF
    }
    assert (
        unchanged.state.next_allowed_epoch - unchanged.state.last_refill_epoch
        > first_replay.state.next_allowed_epoch - first_replay.state.last_refill_epoch
    )


def test_stale_ambiguous_and_rejected_receipts_are_retained_without_work(tmp_path):
    valid = _finding(1)
    suspected = _finding(2, status=FindingStatus.SUSPECTED)
    ledger, stored_receipts = _ledger(tmp_path, valid, suspected)
    stale_receipt = AppendReceipt(
        outcome=AppendOutcome.REJECTED,
        finding_cid=valid.finding_cid,
        sequence=100,
        semantic_key_id=valid.semantic_key_id,
        admission=FindingAdmissionState.STALE,
        reasons=("evidence is stale",),
    )
    rejected_receipt = AppendReceipt(
        outcome=AppendOutcome.REJECTED,
        finding_cid=valid.finding_cid,
        sequence=101,
        semantic_key_id=valid.semantic_key_id,
        admission=FindingAdmissionState.REJECTED,
        reasons=("admission rejected",),
    )

    outcome = _refill(
        ledger,
        (stale_receipt, stored_receipts[1], rejected_receipt),
    )

    assert not outcome.new_tasks
    assert not outcome.new_goals
    assert {diagnostic.disposition for diagnostic in outcome.diagnostics} == {
        FindingDisposition.STALE,
        FindingDisposition.AMBIGUOUS,
        FindingDisposition.REJECTED,
    }
    assert set(outcome.state.seen_receipt_ids) >= {
        stale_receipt.receipt_id,
        stored_receipts[1].receipt_id,
        rejected_receipt.receipt_id,
    }


def test_exhausted_retries_create_exactly_one_bounded_review_task(tmp_path):
    finding = _finding(1)
    ledger, receipts = _ledger(tmp_path, finding)
    failed = {
        "task_id": "failed-repair",
        "finding_semantic_key": finding.semantic_key_id,
        "semantic_key": "repair:failed",
        "kind": TaskKind.REPAIR.value,
        "status": "failed",
        "attempts": 3,
    }
    state = RefillState(
        semantic_task_ids=((finding.semantic_key_id, "failed-repair"),)
    )

    outcome = _refill(ledger, receipts, tasks=(failed,), state=state)

    assert len(outcome.new_tasks) == 1
    review = outcome.new_tasks[0]
    assert review.kind is TaskKind.UNBLOCK_REVIEW
    assert review.goal_id == "family"
    assert review.output_paths == ("ipfs_accelerate_py/pkg/call1.py",)

    existing_review = replace(review, status="open")
    repeated = _refill(
        ledger,
        receipts,
        tasks=(failed, existing_review),
        state=state,
    )
    assert not repeated.new_tasks
    assert repeated.diagnostics[0].disposition is FindingDisposition.REPLAY

    # The normal lifecycle exhausts retries only after the original refill has
    # consumed its receipt.  Replayed evidence must still provenance one review
    # task, and that obligation takes precedence over healthy exhaustion.
    original = _refill(ledger, receipts)
    exhausted_after_refill = replace(
        original.new_tasks[0],
        status="retry_exhausted",
        attempts=3,
    )
    replay_review = _refill(
        ledger,
        receipts,
        tasks=(exhausted_after_refill,),
        state=original.state,
        now=original.state.next_allowed_epoch,
        conclusive_healthy=True,
    )
    assert replay_review.reason is RefillReason.REFILLED
    assert len(replay_review.new_tasks) == 1
    assert replay_review.new_tasks[0].kind is TaskKind.UNBLOCK_REVIEW
    assert replay_review.new_tasks[0].receipt_ids == (receipts[0].receipt_id,)

    no_second_review = _refill(
        ledger,
        receipts,
        tasks=(exhausted_after_refill, replay_review.new_tasks[0]),
        state=replay_review.state,
        now=replay_review.state.next_allowed_epoch,
    )
    assert not no_second_review.new_tasks


def test_dependency_dag_is_topological_and_cycles_create_no_work(tmp_path):
    findings = tuple(_finding(index) for index in range(3))
    ledger, receipts = _ledger(tmp_path, *findings)
    a, b, c = (finding.semantic_key_id for finding in findings)

    outcome = _refill(
        ledger,
        receipts,
        dependencies={b: (a,), c: (b,)},
    )

    assert len(outcome.new_tasks) == 3
    positions = {
        task.finding_semantic_key: index
        for index, task in enumerate(outcome.new_tasks)
    }
    assert positions[a] < positions[b] < positions[c]
    tasks_by_key = {
        task.finding_semantic_key: task for task in outcome.new_tasks
    }
    assert tasks_by_key[b].depends_on == (tasks_by_key[a].task_id,)
    assert tasks_by_key[c].depends_on == (tasks_by_key[b].task_id,)

    cyclic = _refill(
        ledger,
        receipts[:2],
        dependencies={a: (b,), b: (a,)},
    )
    assert not cyclic.new_tasks
    assert {diagnostic.disposition for diagnostic in cyclic.diagnostics} == {
        FindingDisposition.DEPENDENCY_CYCLE
    }


def test_imprecise_output_and_binding_mismatch_create_no_work(tmp_path):
    outside = _finding(1, path="../outside.py")
    wrong_tree = _finding(2, tree_id="tree:other")
    ledger, receipts = _ledger(tmp_path, outside, wrong_tree)

    outcome = _refill(ledger, receipts)

    assert not outcome.new_tasks
    assert {diagnostic.disposition for diagnostic in outcome.diagnostics} == {
        FindingDisposition.IMPRECISE_SCOPE,
        FindingDisposition.UNBOUND,
    }


def test_conclusive_healthy_exhaustion_does_not_create_busywork(tmp_path):
    ledger, _ = _ledger(tmp_path)

    healthy = _refill(ledger, (), conclusive_healthy=True)
    inconclusive = _refill(ledger, (), conclusive_healthy=False)
    replay = _refill(
        ledger,
        (),
        state=healthy.state,
        now=healthy.state.next_allowed_epoch,
        conclusive_healthy=True,
    )

    assert healthy.reason is RefillReason.HEALTHY_EXHAUSTED
    assert healthy.diagnostics[0].disposition is FindingDisposition.HEALTHY_EXHAUSTED
    assert not healthy.new_tasks
    assert not healthy.new_goals
    assert inconclusive.reason is RefillReason.NO_FRESH_RECEIPTS
    assert not inconclusive.diagnostics
    assert replay.reason is RefillReason.NO_FRESH_RECEIPTS
    assert not replay.diagnostics
    assert not replay.new_tasks


def test_hard_policy_limits_are_fail_closed():
    with pytest.raises(ValueError):
        SymbolicFindingRefillPolicy(max_children=4)
    with pytest.raises(ValueError):
        SymbolicFindingRefillPolicy(max_goal_depth=5)
    with pytest.raises(ValueError):
        SymbolicFindingRefillPolicy(max_findings_per_pass=9)
    with pytest.raises(ValueError):
        SymbolicFindingRefillPolicy(max_surplus_per_goal=3)
