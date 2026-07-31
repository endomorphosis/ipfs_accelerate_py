from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

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
    OBJECTIVE_GOAL_G160_ID,
    OBJECTIVE_GOAL_G161_ID,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_TASK_G160_ID,
    OBJECTIVE_TASK_G161_ID,
    OBJECTIVE_TASK_PACKET_ID,
    REFILL_AUTHORIZES_COMPLETION,
    REFILL_AUTHORIZES_EXECUTION,
    REFILL_IDEMPOTENCY_CLAIM_SCHEMA,
    REFILL_IDEMPOTENCY_EVIDENCE,
    REFILL_IDEMPOTENCY_SCHEMA,
    SYMBOLIC_REFILL_EPOCH_CLAIM_SCHEMA,
    SYMBOLIC_REFILL_EPOCH_EVIDENCE,
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
    all_covered_evidence_terms,
    covered_evidence_terms,
    packet_evidence_terms,
    prove_autonomous_refill_packet,
    prove_refill_idempotency,
    prove_symbolic_refill_epoch,
    refill_idempotency_acceptance_dimensions,
    refill_idempotency_evidence,
    refill_idempotency_evidence_terms,
    refill_symbolic_findings,
    symbolic_refill_epoch_acceptance_dimensions,
    symbolic_refill_epoch_evidence,
    symbolic_refill_epoch_evidence_terms,
    verify_refill_idempotency,
    verify_symbolic_refill_epoch,
)


def _load_adaptive_goal_refiner_bridge():
    """Load the declared bridge module by path.

    Package-root ``adaptive_goal_refiner`` is a landed alias of
    ``objectives.adaptive_goal_refiner``.  The task-declared bridge file at
    ``agent_supervisor/adaptive_goal_refiner.py`` still owns the
    autonomous-refill discovery surface for VFS-G160/G161.
    """

    path = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "adaptive_goal_refiner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ipfs_accelerate_py.agent_supervisor._adaptive_goal_refiner_bridge",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert idempotency_record["resolved_goal_ids"] == ("family",)
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
    assert replay_idempotency["resolved_goal_ids"] == ("family",)
    assert replay_idempotency["replay_noop"]

    restored_taskboard = _refill(
        ledger,
        receipts,
        tasks=(admitted.new_tasks[0],),
        state=RefillState(),
        now=102,
    )
    assert not restored_taskboard.new_tasks
    assert restored_taskboard.state.semantic_goal_ids == (
        (finding.semantic_key_id, admitted.new_tasks[0].goal_id),
    )
    assert restored_taskboard.state.semantic_task_ids == (
        (finding.semantic_key_id, admitted.new_tasks[0].task_id),
    )
    assert restored_taskboard.idempotency_evidence is not None
    assert restored_taskboard.idempotency_evidence.resolved_goal_ids == (
        admitted.new_tasks[0].goal_id,
    )
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
    assert outcome.idempotency_evidence is not None
    assert outcome.idempotency_evidence.emitted_goal_ids == (
        outcome.new_goals[0].goal_id,
    )
    assert outcome.idempotency_evidence.resolved_goal_ids == (
        outcome.new_goals[0].goal_id,
    )

    restored_snapshot = SupervisorBacklogSnapshot(
        binding=_binding(),
        goals=(*snapshot.goals, *outcome.new_goals),
        tasks=outcome.new_tasks,
        state=outcome.state,
    )
    replay = BacklogRefinery(ledger).refill(
        restored_snapshot,
        receipts,
        now_epoch=outcome.state.next_allowed_epoch,
    )
    assert not replay.new_goals
    assert not replay.new_tasks
    assert replay.idempotency_id == outcome.idempotency_id
    assert replay.idempotency_evidence is not None
    assert replay.idempotency_evidence.resolved_goal_ids == (
        outcome.new_goals[0].goal_id,
    )
    assert replay.idempotency_evidence.resolved_task_ids == (
        outcome.new_tasks[0].task_id,
    )
    assert replay.idempotency_evidence.replay_noop

    with pytest.raises(ValueError, match="exact refinement goal"):
        SupervisorBacklogSnapshot(
            binding=_binding(),
            goals=(_family_goal(),),
        )
    with pytest.raises(ValueError, match="goal absent from the objective heap"):
        SupervisorBacklogSnapshot(
            binding=_binding(),
            goals=(_root(),),
            tasks=outcome.new_tasks,
            state=outcome.state,
        )
    with pytest.raises(ValueError, match="lineage differs"):
        SupervisorBacklogSnapshot(
            binding=_binding(),
            goals=restored_snapshot.goals,
            tasks=(
                replace(
                    outcome.new_tasks[0],
                    ancestor_goal_ids=(),
                    parent_goal_id="",
                ),
            ),
            state=outcome.state,
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
    assert verify_symbolic_refill_epoch(outcome)
    assert verify_refill_idempotency(outcome)
    assert outcome.idempotency_evidence is not None
    assert set(outcome.idempotency_evidence.resolved_goal_ids) == {
        goal.goal_id for goal in outcome.new_goals
    }
    assert prove_autonomous_refill_packet(outcome)["satisfied"] is True
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


def test_vfs_g160_symbolic_refill_epoch_evidence_discoverable(tmp_path):
    """Prove vfs/symbolic-refill-epoch@1 for VFS-G160 / VFS-080.

    Parent VFS-G120 acceptance is demonstrated only through the epoch
    receipt: fresh admissions, family reuse, hard ceilings, content-addressed
    state transition, and non-authoritative proposals.
    """

    assert SYMBOLIC_REFILL_EPOCH_EVIDENCE == "vfs/symbolic-refill-epoch@1"
    assert SYMBOLIC_REFILL_EPOCH_SCHEMA == "vfs/symbolic-refill-epoch@1"
    assert symbolic_refill_epoch_evidence() == "vfs/symbolic-refill-epoch@1"
    assert symbolic_refill_epoch_evidence_terms() == (
        "vfs/symbolic-refill-epoch@1",
    )
    assert OBJECTIVE_GOAL_G160_ID == "VFS-G160"
    assert OBJECTIVE_TASK_G160_ID == "VFS-080"
    assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G120"
    assert "vfs/symbolic-refill-epoch@1" in covered_evidence_terms()
    assert "vfs/symbolic-refill-epoch@1" in packet_evidence_terms()
    assert "vfs/symbolic-refill-epoch@1" in all_covered_evidence_terms()
    bridge = _load_adaptive_goal_refiner_bridge()
    assert bridge.symbolic_refill_epoch_evidence() == "vfs/symbolic-refill-epoch@1"
    assert bridge.OBJECTIVE_TASK_G160_ID == "VFS-080"
    assert "vfs/symbolic-refill-epoch@1" in bridge.covered_evidence_terms()

    finding = _finding(1)
    ledger, receipts = _ledger(tmp_path, finding)
    admitted = _refill(ledger, receipts)
    later = _refill(ledger, receipts, now=101)

    assert verify_symbolic_refill_epoch(admitted)
    assert admitted.epoch_evidence is not None
    payload = admitted.epoch_evidence.to_record()
    assert payload["schema"] == "vfs/symbolic-refill-epoch@1"
    assert payload["epoch_id"] == admitted.refill_epoch_id
    assert payload["changed"]
    assert payload["fresh_receipt_ids"] == (receipts[0].receipt_id,)
    assert payload["emitted_task_ids"] == (admitted.new_tasks[0].task_id,)
    # Observation time changes the epoch without rewriting task identity.
    assert later.refill_epoch_id != admitted.refill_epoch_id
    assert later.new_tasks[0].task_id == admitted.new_tasks[0].task_id
    assert later.idempotency_id == admitted.idempotency_id

    dimensions = symbolic_refill_epoch_acceptance_dimensions(admitted)
    assert all(dimensions.values())
    assert dimensions["fresh_admitted_only"]
    assert dimensions["goal_family_reuse_or_bounded_child"]
    assert dimensions["breadth_depth_open_work_cooldown"]
    assert dimensions["prior_and_result_state_tracked"]
    assert dimensions["epoch_distinct_from_task_identity"]
    assert dimensions["binding_identity"]
    assert dimensions["non_authoritative"]

    claim = prove_symbolic_refill_epoch(admitted)
    assert claim["schema"] == SYMBOLIC_REFILL_EPOCH_CLAIM_SCHEMA
    assert claim["evidence"] == "vfs/symbolic-refill-epoch@1"
    assert claim["evidence_terms"] == ["vfs/symbolic-refill-epoch@1"]
    assert claim["goal_id"] == "VFS-G160"
    assert claim["parent_goal_id"] == "VFS-G120"
    assert claim["task_id"] == "VFS-080"
    assert claim["packet_task_id"] == OBJECTIVE_TASK_PACKET_ID
    assert claim["packet_goal_ids"] == ["VFS-G160", "VFS-G161"]
    assert claim["satisfied"] is True
    assert claim["verified"] is True
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False
    assert claim["semantic_authority"] is False
    assert claim["authorizes_execution"] is False
    assert claim["authorizes_completion"] is False
    assert all(claim["acceptance_dimensions"].values())
    assert claim["binding_id"] == _binding().binding_id
    assert claim["epoch_id"] == admitted.refill_epoch_id

    # Gate-only epochs still verify without creating work.
    cooling = _refill(
        ledger,
        receipts,
        state=RefillState(next_allowed_epoch=101),
        now=100,
    )
    assert cooling.reason is RefillReason.COOLDOWN
    assert verify_symbolic_refill_epoch(cooling)
    cool_claim = prove_symbolic_refill_epoch(cooling)
    assert cool_claim["verified"] is True
    assert cool_claim["changed"] is False
    assert cool_claim["acceptance_dimensions"]["non_authoritative"]
    assert cool_claim["acceptance_dimensions"]["breadth_depth_open_work_cooldown"]

    # Stale/rejected receipts create no work; epoch still evidences the pass.
    stale_receipt = AppendReceipt(
        outcome=AppendOutcome.REJECTED,
        finding_cid=finding.finding_cid,
        sequence=100,
        semantic_key_id=finding.semantic_key_id,
        admission=FindingAdmissionState.STALE,
        reasons=("evidence is stale",),
    )
    retained = _refill(ledger, (stale_receipt,))
    assert not retained.new_tasks
    assert verify_symbolic_refill_epoch(retained)
    retained_claim = prove_symbolic_refill_epoch(retained)
    assert retained_claim["verified"] is True
    assert retained_claim["acceptance_dimensions"]["fresh_admitted_only"]
    assert retained_claim["changed"] is False


def test_vfs_g161_refill_idempotency_evidence_discoverable(tmp_path):
    """Prove vfs/refill-idempotency@1 for VFS-G161 / VFS-083."""

    assert REFILL_IDEMPOTENCY_EVIDENCE == "vfs/refill-idempotency@1"
    assert REFILL_IDEMPOTENCY_SCHEMA == "vfs/refill-idempotency@1"
    assert refill_idempotency_evidence() == "vfs/refill-idempotency@1"
    assert refill_idempotency_evidence_terms() == ("vfs/refill-idempotency@1",)
    assert OBJECTIVE_GOAL_G161_ID == "VFS-G161"
    assert OBJECTIVE_TASK_G161_ID == "VFS-083"
    assert "vfs/refill-idempotency@1" in covered_evidence_terms()
    assert "vfs/refill-idempotency@1" in packet_evidence_terms()
    bridge = _load_adaptive_goal_refiner_bridge()
    assert bridge.refill_idempotency_evidence() == "vfs/refill-idempotency@1"
    assert bridge.OBJECTIVE_TASK_G161_ID == "VFS-083"
    assert "goal, subgoal, and task identities survive replay" in (
        bridge.REFILL_IDEMPOTENCY_INVARIANTS
    )

    finding = _finding(1, family="new-family")
    ledger, receipts = _ledger(tmp_path, finding)
    refinery = BacklogRefinery(ledger)
    initial_snapshot = SupervisorBacklogSnapshot(
        binding=_binding(),
        goals=(_root(),),
    )
    admitted = refinery.refill(initial_snapshot, receipts, now_epoch=100)
    restored_snapshot = SupervisorBacklogSnapshot(
        binding=_binding(),
        goals=(*initial_snapshot.goals, *admitted.new_goals),
        tasks=admitted.new_tasks,
        state=admitted.state,
    )
    replay = refinery.refill(
        restored_snapshot,
        receipts,
        now_epoch=admitted.state.next_allowed_epoch,
    )

    assert verify_refill_idempotency(admitted)
    assert verify_refill_idempotency(replay)
    assert admitted.idempotency_id == replay.idempotency_id
    assert replay.idempotency_evidence is not None
    assert admitted.idempotency_evidence is not None
    assert admitted.idempotency_evidence.emitted_goal_ids == (
        admitted.new_goals[0].goal_id,
    )
    assert admitted.idempotency_evidence.resolved_goal_ids == (
        admitted.new_goals[0].goal_id,
    )
    assert replay.idempotency_evidence.emitted_goal_ids == ()
    assert replay.idempotency_evidence.resolved_goal_ids == (
        admitted.new_goals[0].goal_id,
    )
    assert replay.idempotency_evidence.resolved_task_ids == (
        admitted.new_tasks[0].task_id,
    )
    assert replay.idempotency_evidence.replay_noop

    dimensions = refill_idempotency_acceptance_dimensions(replay)
    assert all(dimensions.values())
    assert dimensions["replay_noop_when_replayed"]
    assert dimensions["resolved_goals_cover_emitted"]
    assert dimensions["goal_task_identity_paired"]
    assert dimensions["wall_clock_excluded"]

    claim = prove_refill_idempotency(replay)
    assert claim["schema"] == REFILL_IDEMPOTENCY_CLAIM_SCHEMA
    assert claim["evidence"] == "vfs/refill-idempotency@1"
    assert claim["goal_id"] == "VFS-G161"
    assert claim["parent_goal_id"] == "VFS-G120"
    assert claim["task_id"] == "VFS-083"
    assert claim["packet_goal_ids"] == ["VFS-G160", "VFS-G161"]
    assert claim["satisfied"] is True
    assert claim["verified"] is True
    assert claim["replay_noop"] is True
    assert claim["idempotency_id"] == admitted.idempotency_id
    assert claim["emitted_goal_ids"] == []
    assert claim["resolved_goal_ids"] == [admitted.new_goals[0].goal_id]
    assert claim["resolved_task_ids"] == [admitted.new_tasks[0].task_id]
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False


def test_autonomous_refill_packet_covers_g160_and_g161_together(tmp_path):
    """Packet claim covers both autonomous-refill evidence terms cohesively."""

    finding = _finding(1)
    ledger, receipts = _ledger(tmp_path, finding)
    admitted = _refill(ledger, receipts)
    packet = prove_autonomous_refill_packet(admitted)

    assert packet["evidence_terms"] == [
        "vfs/symbolic-refill-epoch@1",
        "vfs/refill-idempotency@1",
    ]
    assert packet["all_evidence_terms"] == list(covered_evidence_terms())
    assert packet["goal_ids"] == ["VFS-G160", "VFS-G161"]
    assert packet["parent_goal_id"] == "VFS-G120"
    assert packet["task_ids"] == ["VFS-080", "VFS-083"]
    assert packet["packet_task_id"] == "VFS-079"
    assert packet["satisfied"] is True
    assert packet["authorizes_execution"] is False
    assert packet["authorizes_completion"] is False
    assert packet["symbolic_refill_epoch"]["evidence"] == (
        "vfs/symbolic-refill-epoch@1"
    )
    assert packet["refill_idempotency"]["evidence"] == "vfs/refill-idempotency@1"
    assert all(packet["symbolic_refill_epoch"]["acceptance_dimensions"].values())
    assert all(packet["refill_idempotency"]["acceptance_dimensions"].values())

    # Declared adaptive_goal_refiner bridge re-exports the same discovery surface.
    bridge = _load_adaptive_goal_refiner_bridge()
    assert bridge.packet_evidence_terms() == packet_evidence_terms()
    assert bridge.prove_autonomous_refill_packet(admitted)["satisfied"] is True
    assert bridge.OBJECTIVE_PACKET_GOAL_IDS == ("VFS-G160", "VFS-G161")
