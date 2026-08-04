"""Tests for revision-bound plan-steer preview (PDR-028 / PlanSteerService@1)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    CompletionAuthority,
    DeltaEffectClass,
    LifecycleState,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanRequestBudget,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanSteerRequest,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
    plan_revision_cid,
)
from ipfs_accelerate_py.agent_supervisor.prompt.plan_steer_service import (
    PLAN_STEER_SERVICE_INTERFACE,
    PlanSteerLiveState,
    PlanSteerPopulationPartition,
    PlanSteerPreviewMaterials,
    PlanSteerPreviewReceipt,
    PlanSteerRejectionCode,
    PlanSteerScanImpact,
    PlanSteerService,
    PlanSteerServiceError,
    PlanSteerTaskRecord,
    PlanSteerVerdict,
    partition_live_task_populations,
    preview_steer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cid(name: str) -> str:
    return plan_revision_cid({"fixture": name})


def _roots(**changes: object) -> PlanAuthorityRoots:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:test",
        "repository_root_cid": _cid("repo-root"),
        "dirty_worktree_root": _cid("dirty"),
        "task_source_id": "task-source:markdown:board",
        "task_source_revision": _cid("ts-rev-1"),
        "policy_root": _cid("policy"),
        "intent_ir_root": _cid("intent"),
        "legal_ir_root": _cid("legal"),
        "security_ir_root": _cid("security"),
        "program_root": _cid("program"),
        "capability_catalog_root": _cid("capability"),
        "provider_catalog_root": _cid("provider-catalog"),
        "usage_policy_root": _cid("usage"),
        "configuration_root": _cid("config"),
    }
    values.update(changes)
    return PlanAuthorityRoots(**values)


def _budget(**changes: int) -> PlanRequestBudget:
    values = {
        "max_goals": 16,
        "max_tasks": 64,
        "max_graph_depth": 8,
        "max_output_paths": 128,
        "max_ready_width": 1,
        "max_repair_rounds": 2,
        "max_scan_bytes": 8 * 1024 * 1024,
        "max_analysis_operations": 16,
        "max_evidence_items": 64,
        "max_logic_families": 4,
        "max_model_calls": 2,
        "max_latency_ms": 60_000,
        "max_provider_tokens": 8_192,
        "max_cost_micros": 0,
    }
    values.update(changes)
    return PlanRequestBudget(**values)


def _population(kind: PopulationKind, *members: str) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=members)


TASK_DONE = _cid("task-done")
TASK_RUNNING = _cid("task-running")
TASK_CLAIMED = _cid("task-claimed")
TASK_SETTLING = _cid("task-settling")
TASK_READY = _cid("task-ready")
TASK_BLOCKED = _cid("task-blocked")
TASK_SUPERSEDED = _cid("task-superseded")
TASK_FAILED = _cid("task-failed")
GOAL_ROOT = _cid("goal-root")


def _tasks_mixed() -> tuple[dict[str, object], ...]:
    return (
        {
            "task_cid": TASK_DONE,
            "lifecycle_state": LifecycleState.ACCEPTED.value,
            "spec_revision": _cid("spec-done"),
        },
        {
            "task_cid": TASK_RUNNING,
            "lifecycle_state": LifecycleState.RUNNING.value,
            "spec_revision": _cid("spec-running"),
            "attempt_id": "attempt:1",
            "lease_id": "lease:1",
        },
        {
            "task_cid": TASK_CLAIMED,
            "lifecycle_state": LifecycleState.CLAIMED.value,
            "spec_revision": _cid("spec-claimed"),
            "lease_id": "lease:1",
        },
        {
            "task_cid": TASK_SETTLING,
            "lifecycle_state": LifecycleState.SETTLING.value,
            "spec_revision": _cid("spec-settling"),
        },
        {
            "task_cid": TASK_READY,
            "lifecycle_state": LifecycleState.UNSTARTED.value,
            "spec_revision": _cid("spec-ready"),
        },
        {
            "task_cid": TASK_BLOCKED,
            "lifecycle_state": LifecycleState.BLOCKED.value,
            "spec_revision": _cid("spec-blocked"),
        },
        {
            "task_cid": TASK_SUPERSEDED,
            "lifecycle_state": LifecycleState.SUPERSEDED.value,
            "spec_revision": _cid("spec-superseded"),
        },
        {
            "task_cid": TASK_FAILED,
            "lifecycle_state": LifecycleState.FAILED.value,
            "spec_revision": _cid("spec-failed"),
        },
    )


def _claimed_digest_for(tasks: tuple[dict[str, object], ...]) -> PlanPopulationDigest:
    members = [
        str(task["task_cid"])
        for task in tasks
        if str(task["lifecycle_state"])
        in {
            LifecycleState.CLAIMED.value,
            LifecycleState.RUNNING.value,
            LifecycleState.SETTLING.value,
        }
    ]
    return _population(PopulationKind.CLAIMED, *members)


def _steer_request(
    *,
    tasks: tuple[dict[str, object], ...] | None = None,
    **changes: object,
) -> PlanSteerRequest:
    task_rows = tasks if tasks is not None else _tasks_mixed()
    claimed = _claimed_digest_for(task_rows)
    accepted = _population(PopulationKind.ACCEPTED, TASK_DONE)
    status = _population(
        PopulationKind.UNSTARTED,
        TASK_READY,
        TASK_RUNNING,
        TASK_CLAIMED,
        TASK_SETTLING,
    )
    values: dict[str, object] = {
        "directive_cid": _cid("directive"),
        "base_admitted_plan_root": _cid("admitted-plan"),
        "base_materialized_plan_root": _cid("materialized-plan"),
        "plan_revision": 2,
        "parent_revision": 1,
        "roots": _roots(),
        "event_cursor": _cid("cursor-1"),
        "status_population": status,
        "claimed_population": claimed,
        "accepted_population": accepted,
        "accepted_evidence_root": _cid("accepted-evidence"),
        "completion_revision": _cid("completion-1"),
        "allowed_delta_operations": (
            PlanDeltaOperation.ADD_TASK.value,
            PlanDeltaOperation.ATTACH_EVIDENCE.value,
            PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK.value,
            PlanDeltaOperation.BLOCK_UNSTARTED_TASK.value,
            PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION.value,
            PlanDeltaOperation.RECORD_UNCERTAINTY.value,
        ),
        "budget": _budget(),
        "may_request_lifecycle_action": True,
        "caller": "principal:test",
        "idempotency_key": "steer/attempt-1",
        "lease_id": "lease:1",
        "fencing_epoch": 3,
        "supervisor_run_id": "run:1",
        "supervisor_state_revision": _cid("supervisor-state"),
        "redacted_directive_metadata": {
            "affected_paths": [
                "ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py"
            ],
            "affected_symbols": ["PlanSteerService.preview_steer"],
        },
    }
    values.update(changes)
    return PlanSteerRequest(**values)


def _base_plan(request: PlanSteerRequest) -> PlanRevision:
    return PlanRevision(
        plan_root_cid=request.base_materialized_plan_root,
        semantic_revision=request.plan_revision,
        parent_plan_root=_cid("parent-create"),
        origin=PlanOrigin.STEER,
        roots=request.roots,
        request_cid=_cid("prior-request"),
        delta_cid=_cid("prior-delta"),
        scan_receipt_cid=_cid("prior-scan"),
        query_plan_cid=_cid("prior-query"),
        evidence_bundle_cid=_cid("prior-evidence"),
        admission_receipt_cid=_cid("prior-admission"),
        execution_plan_cid=_cid("prior-exec"),
        goal_population=_population(PopulationKind.RETAINED, GOAL_ROOT),
        task_population=_population(
            PopulationKind.RETAINED,
            TASK_DONE,
            TASK_RUNNING,
            TASK_CLAIMED,
            TASK_SETTLING,
            TASK_READY,
            TASK_BLOCKED,
            TASK_SUPERSEDED,
            TASK_FAILED,
        ),
        added_population=_population(PopulationKind.ADDED),
        superseded_population=_population(PopulationKind.SUPERSEDED, TASK_SUPERSEDED),
        retained_population=_population(
            PopulationKind.RETAINED,
            TASK_DONE,
            TASK_RUNNING,
            TASK_CLAIMED,
            TASK_SETTLING,
            TASK_READY,
            TASK_BLOCKED,
            TASK_FAILED,
        ),
        deferred_population=_population(PopulationKind.DEFERRED),
        claimed_population=request.claimed_population,
        completed_population=_population(PopulationKind.COMPLETED, TASK_DONE),
        blocked_population=_population(PopulationKind.BLOCKED, TASK_BLOCKED),
        resource_contract=PlanResourceContract(),
        provider_contract=PlanProviderContract(),
        lease_contract=PlanLeaseContract(fencing_epoch=request.fencing_epoch),
        retry_contract=PlanRetryContract(),
        worktree_contract=PlanWorktreeContract(
            policy="isolated",
            expected_base_revision=_cid("worktree-base"),
        ),
        merge_strategy=PlanMergeStrategy(kind=MergeStrategyKind.SERIAL),
        conflict_contract=PlanConflictContract(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py",
            ),
        ),
        completion_rule=PlanCompletionRule(
            authority=CompletionAuthority.VALIDATION_GATE,
        ),
        validation_dag=(
            PlanValidationNode(
                validation_key="validation:pytest",
                argv=("python", "-m", "pytest", "-q"),
            ),
        ),
        event_cursor=request.event_cursor,
    )


def _live_state(
    request: PlanSteerRequest,
    *,
    tasks: tuple[dict[str, object], ...] | None = None,
    **changes: object,
) -> PlanSteerLiveState:
    task_rows = tasks if tasks is not None else _tasks_mixed()
    values: dict[str, object] = {
        "current_roots": request.roots,
        "plan_revision": request.plan_revision,
        "event_cursor": request.event_cursor,
        "accepted_evidence_root": request.accepted_evidence_root,
        "base_plan_root": request.base_materialized_plan_root,
        "base_admitted_plan_root": request.base_admitted_plan_root,
        "completion_revision": request.completion_revision,
        "supervisor_run_id": request.supervisor_run_id,
        "supervisor_state_revision": request.supervisor_state_revision,
        "lease_id": request.lease_id,
        "fencing_epoch": request.fencing_epoch,
        "tasks": task_rows,
        "goals": (
            {
                "goal_cid": GOAL_ROOT,
                "lifecycle_state": LifecycleState.ADMITTED.value,
            },
        ),
        "lease_contract": PlanLeaseContract(fencing_epoch=request.fencing_epoch),
        "worktree_contract": PlanWorktreeContract(
            policy="isolated",
            expected_base_revision=_cid("worktree-base"),
        ),
        "merge_strategy": PlanMergeStrategy(kind=MergeStrategyKind.SERIAL),
        "base_plan": _base_plan(request),
        "scan": {
            "scan_cid": _cid("scan-current"),
            "repository_root_cid": request.roots.repository_root_cid,
            "dirty_worktree_root": request.roots.dirty_worktree_root,
        },
        "impact": {
            "impacted_paths": [
                "ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py"
            ],
            "modified_paths": [
                "ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py"
            ],
            "impacted_symbols": ["PlanSteerService.preview_steer"],
        },
        "merge_state": {"status": "idle", "queue_depth": 0},
        "run_state": {
            "run_id": request.supervisor_run_id,
            "state_revision": request.supervisor_state_revision,
        },
    }
    values.update(changes)
    return PlanSteerLiveState(**values)


def _successor_item(
    *,
    target: str = TASK_RUNNING,
    lifecycle: LifecycleState = LifecycleState.RUNNING,
    deferred: bool = True,
) -> PlanDeltaItem:
    after = _cid("successor-task")
    return PlanDeltaItem(
        item_key="delta:successor-running",
        operation=PlanDeltaOperation.ADD_TASK,
        target_cid=target,
        expected_target_lifecycle=lifecycle,
        expected_target_spec_revision=_cid("spec-running"),
        before_digest="",
        after_record_cid=after,
        effect_class=(
            DeltaEffectClass.DEFERRED if deferred else DeltaEffectClass.MATERIALIZABLE_NOW
        ),
        rationale="Add a successor that depends on the running task.",
        provenance={"source": "test"},
        expected_effects=("append-task",),
        affected_goal_cids=(GOAL_ROOT,),
        affected_task_cids=(after, target),
        affected_paths=(
            "ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py",
        ),
        preconditions=(f"target-terminal:{target}",) if deferred else (),
    )


def _supersede_unstarted_item() -> PlanDeltaItem:
    after = _cid("replacement-task")
    return PlanDeltaItem(
        item_key="delta:supersede-ready",
        operation=PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK,
        target_cid=TASK_READY,
        expected_target_lifecycle=LifecycleState.UNSTARTED,
        expected_target_spec_revision=_cid("spec-ready"),
        before_digest=_cid("spec-ready"),
        after_record_cid=after,
        effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
        rationale="Replace an unstarted task after impact scan.",
        provenance={"source": "test"},
        expected_effects=("supersede-task",),
        affected_task_cids=(TASK_READY, after),
        affected_paths=(
            "ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py",
        ),
    )


def _lifecycle_request_item() -> PlanDeltaItem:
    return PlanDeltaItem(
        item_key="delta:lifecycle-cancel-request",
        operation=PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
        target_cid=TASK_RUNNING,
        expected_target_lifecycle=LifecycleState.RUNNING,
        expected_target_spec_revision=_cid("spec-running"),
        before_digest="",
        after_record_cid=_cid("lifecycle-request"),
        effect_class=DeltaEffectClass.LIFECYCLE_REQUEST,
        rationale="Request cancel; do not perform it in steer apply.",
        provenance={"source": "test", "action": "cancel"},
        expected_effects=("lifecycle-request:cancel",),
        affected_task_cids=(TASK_RUNNING,),
    )


def _deferred_supersede_running_item() -> PlanDeltaItem:
    """Express deferred supersession as a successor ADD_TASK, not an in-place edit.

    ``SUPERSEDE_UNSTARTED_TASK`` is contract-forbidden against running history;
    the closed language defers replacement via a successor whose preconditions
    require the running attempt to become terminal.
    """

    after = _cid("deferred-replacement")
    return PlanDeltaItem(
        item_key="delta:deferred-supersede-running",
        operation=PlanDeltaOperation.ADD_TASK,
        target_cid=TASK_RUNNING,
        expected_target_lifecycle=LifecycleState.RUNNING,
        expected_target_spec_revision=_cid("spec-running"),
        before_digest="",
        after_record_cid=after,
        effect_class=DeltaEffectClass.DEFERRED,
        rationale=(
            "Deferred replacement successor; activates only after the running "
            "attempt is terminal (explicit deferred supersession)."
        ),
        provenance={
            "source": "test",
            "deferred_supersession_of": TASK_RUNNING,
        },
        expected_effects=("deferred-supersede",),
        affected_task_cids=(TASK_RUNNING, after),
        preconditions=(f"target-terminal:{TASK_RUNNING}",),
    )


# ---------------------------------------------------------------------------
# Population partitioning
# ---------------------------------------------------------------------------


def test_partition_populations_covers_all_lifecycle_slices() -> None:
    request = _steer_request()
    state = _live_state(request)
    service = PlanSteerService()
    partition = service.partition_populations(state)

    assert TASK_DONE in partition.accepted.member_cids
    assert TASK_RUNNING in partition.running.member_cids
    assert TASK_CLAIMED in partition.claimed.member_cids
    assert TASK_SETTLING in partition.settling.member_cids
    assert TASK_READY in partition.unstarted.member_cids
    assert TASK_BLOCKED in partition.blocked.member_cids
    assert TASK_SUPERSEDED in partition.superseded.member_cids
    assert TASK_FAILED in partition.failed.member_cids
    # Disjoint partition
    assert len(partition.all_member_cids()) == 8
    assert partition.claimed_family_digest == request.claimed_population.digest


def test_partition_helper_round_trips_without_full_state() -> None:
    partition = partition_live_task_populations(_tasks_mixed())
    assert isinstance(partition, PlanSteerPopulationPartition)
    encoded = partition.to_record()
    assert PlanSteerPopulationPartition.from_dict(encoded).content_id == (
        partition.content_id
    )


# ---------------------------------------------------------------------------
# Happy path: admitted preview
# ---------------------------------------------------------------------------


def test_preview_steer_admits_successor_without_editing_running_work() -> None:
    request = _steer_request()
    state = _live_state(request)
    materials = PlanSteerPreviewMaterials(
        request=request,
        live_state=state,
        proposed_delta_items=(_successor_item(), _supersede_unstarted_item()),
        expected_effects=("append-task", "supersede-task"),
    )
    receipt = PlanSteerService().preview_steer(materials)

    assert receipt.admitted is True
    assert receipt.verdict is PlanSteerVerdict.ADMITTED
    assert receipt.read_only is True
    assert receipt.wrote_task_source is False
    assert receipt.restart_serializable is True
    assert receipt.service_interface == PLAN_STEER_SERVICE_INTERFACE
    assert receipt.delta_cid
    assert receipt.candidate_plan_root
    assert receipt.candidate_plan_revision == request.plan_revision + 1
    assert receipt.base_plan_root == request.base_materialized_plan_root
    assert "delta:successor-running" in receipt.successor_item_keys
    assert "delta:successor-running" in receipt.deferred_item_keys
    assert "delta:supersede-ready" in receipt.materializable_item_keys
    assert receipt.scan_receipt_cid == _cid("scan-current")
    assert receipt.admission_receipt_cid
    # Body-free durable receipt
    encoded = receipt.to_json()
    assert "prompt_text" not in encoded
    assert "source_body" not in encoded
    assert "password" not in encoded
    # Restart-serializable
    restored = PlanSteerPreviewReceipt.from_dict(receipt.to_record())
    assert restored.content_id == receipt.content_id
    assert restored.admitted is True


def test_preview_steer_idempotent_for_same_request_cid() -> None:
    request = _steer_request()
    state = _live_state(request)
    service = PlanSteerService()
    materials = PlanSteerPreviewMaterials(
        request=request,
        live_state=state,
        proposed_delta_items=(_successor_item(),),
    )
    first = service.preview_steer(materials)
    second = service.preview_steer(materials)
    assert first.content_id == second.content_id
    assert first.receipt_cid == second.receipt_cid


def test_module_level_preview_steer_wrapper() -> None:
    request = _steer_request()
    state = _live_state(request)
    receipt = preview_steer(
        request,
        state,
        proposed_delta_items=(_successor_item(),),
    )
    assert receipt.admitted is True


def test_deterministic_fallback_delta_when_no_items_proposed() -> None:
    request = _steer_request()
    state = _live_state(request)
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(request=request, live_state=state)
    )
    assert receipt.admitted is True
    assert receipt.delta_cid
    # Successor against running work is deferred by default.
    assert receipt.deferred_item_keys or receipt.successor_item_keys


# ---------------------------------------------------------------------------
# Stale integrity failures
# ---------------------------------------------------------------------------


def test_stale_roots_fail_closed() -> None:
    request = _steer_request()
    state = _live_state(
        request,
        current_roots=replace(_roots(), dirty_worktree_root=_cid("dirty-2")),
    )
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_ROOT.value in receipt.reason_codes or (
        PlanSteerRejectionCode.STALE_SCAN.value in receipt.reason_codes
    )


def test_stale_revision_fails() -> None:
    request = _steer_request()
    state = _live_state(request, plan_revision=3)
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_REVISION.value in receipt.reason_codes


def test_stale_cursor_fails() -> None:
    request = _steer_request()
    state = _live_state(request, event_cursor=_cid("cursor-stale"))
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_CURSOR.value in receipt.reason_codes


def test_stale_claimed_population_fails() -> None:
    request = _steer_request()
    # Drop the running task so claimed population no longer matches.
    tasks = tuple(
        task
        for task in _tasks_mixed()
        if task["task_cid"] != TASK_RUNNING
    )
    state = _live_state(request, tasks=tasks)
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(target=TASK_CLAIMED),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_CLAIMED.value in receipt.reason_codes


def test_stale_lease_and_fence_fail() -> None:
    request = _steer_request()
    stale_lease = _live_state(request, lease_id="lease:other")
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=stale_lease,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_LEASE.value in receipt.reason_codes

    stale_fence = _live_state(request, fencing_epoch=99)
    receipt2 = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=stale_fence,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt2.admitted is False
    assert PlanSteerRejectionCode.STALE_FENCE.value in receipt2.reason_codes


def test_stale_policy_root_fails() -> None:
    request = _steer_request()
    state = _live_state(
        request,
        current_roots=replace(_roots(), policy_root=_cid("policy-other")),
    )
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    codes = set(receipt.reason_codes)
    assert codes.intersection(
        {
            PlanSteerRejectionCode.STALE_ROOT.value,
            PlanSteerRejectionCode.STALE_POLICY.value,
        }
    )


def test_stale_base_plan_root_fails() -> None:
    request = _steer_request()
    state = _live_state(request, base_plan_root=_cid("wrong-base"))
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_BASE.value in receipt.reason_codes


def test_stale_scan_dirty_root_fails() -> None:
    request = _steer_request()
    state = _live_state(
        request,
        scan={
            "scan_cid": _cid("scan-current"),
            "repository_root_cid": request.roots.repository_root_cid,
            "dirty_worktree_root": _cid("dirty-stale"),
        },
    )
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_SCAN.value in receipt.reason_codes


def test_stale_accepted_evidence_fails() -> None:
    request = _steer_request()
    state = _live_state(request, accepted_evidence_root=_cid("accepted-other"))
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_ACCEPTED.value in receipt.reason_codes


def test_stale_supervisor_run_fails() -> None:
    request = _steer_request()
    state = _live_state(request, supervisor_run_id="run:other")
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_RUN.value in receipt.reason_codes


# ---------------------------------------------------------------------------
# Lifecycle safety: running work never edited
# ---------------------------------------------------------------------------


def test_running_work_cannot_be_mutated_in_place() -> None:
    request = _steer_request()
    state = _live_state(request)
    # Contract layer already rejects non-deferred mutating ops on running.
    with pytest.raises(Exception):
        PlanDeltaItem(
            item_key="delta:mutate-running",
            operation=PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK,
            target_cid=TASK_RUNNING,
            expected_target_lifecycle=LifecycleState.RUNNING,
            expected_target_spec_revision=_cid("spec-running"),
            before_digest=_cid("spec-running"),
            after_record_cid=_cid("mutated"),
            effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
            rationale="Illegal in-place edit of running work.",
            provenance={"source": "test"},
        )


def test_deferred_supersession_of_running_work_is_explicit() -> None:
    request = _steer_request()
    state = _live_state(request)
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(
                _deferred_supersede_running_item(),
                _successor_item(),
            ),
        )
    )
    assert receipt.admitted is True
    assert "delta:deferred-supersede-running" in receipt.deferred_item_keys


def test_lifecycle_request_is_explicit_and_not_a_taskboard_write() -> None:
    request = _steer_request()
    state = _live_state(request)
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(
                _successor_item(),
                _lifecycle_request_item(),
            ),
        )
    )
    assert receipt.admitted is True
    assert "delta:lifecycle-cancel-request" in receipt.lifecycle_request_item_keys
    assert receipt.wrote_task_source is False


def test_lifecycle_request_rejected_when_not_permitted() -> None:
    request = _steer_request(may_request_lifecycle_action=False)
    state = _live_state(request)
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(
                _successor_item(),
                _lifecycle_request_item(),
            ),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.FORBIDDEN_OPERATION.value in receipt.reason_codes


def test_completed_history_can_only_receive_evidence_or_successors() -> None:
    request = _steer_request()
    state = _live_state(request)
    attach = PlanDeltaItem(
        item_key="delta:attach-completed",
        operation=PlanDeltaOperation.ATTACH_EVIDENCE,
        target_cid=TASK_DONE,
        expected_target_lifecycle=LifecycleState.ACCEPTED,
        expected_target_spec_revision=_cid("spec-done"),
        before_digest="",
        after_record_cid=_cid("evidence-ref"),
        effect_class=DeltaEffectClass.EVIDENCE_ONLY,
        rationale="Attach additional evidence to accepted history.",
        provenance={"source": "test"},
        expected_effects=("attach-evidence",),
        affected_task_cids=(TASK_DONE,),
    )
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(attach, _successor_item()),
        )
    )
    assert receipt.admitted is True


def test_forbidden_operation_outside_request_allowlist_fails() -> None:
    request = _steer_request(
        allowed_delta_operations=(PlanDeltaOperation.ADD_TASK.value,)
    )
    state = _live_state(request)
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(
                _successor_item(),
                _supersede_unstarted_item(),
            ),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.FORBIDDEN_OPERATION.value in receipt.reason_codes


# ---------------------------------------------------------------------------
# Validation of resulting plan / admission
# ---------------------------------------------------------------------------


def test_admission_callback_can_reject_candidate() -> None:
    request = _steer_request()
    state = _live_state(request)

    def deny(_candidate: object) -> dict[str, object]:
        return {
            "admitted": False,
            "reasons": ("resource infeasible",),
            "receipt_cid": _cid("admission-deny"),
        }

    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
            admit_candidate=deny,
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.ADMISSION_FAILED.value in receipt.reason_codes


def test_admission_callback_can_admit_candidate() -> None:
    request = _steer_request()
    state = _live_state(request)

    def allow(_candidate: object) -> tuple[bool, str, tuple[str, ...]]:
        return True, _cid("admission-ok"), ()

    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
            admit_candidate=allow,
        )
    )
    assert receipt.admitted is True
    assert receipt.admission_receipt_cid == _cid("admission-ok")


def test_affected_population_budget_enforced() -> None:
    request = _steer_request(max_affected_tasks=1)
    state = _live_state(request)
    # Successor item affects both after + target => 2 tasks.
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert (
        PlanSteerRejectionCode.AFFECTED_POPULATION_EXCEEDED.value
        in receipt.reason_codes
    )


def test_empty_delta_rejected() -> None:
    request = _steer_request(
        allowed_delta_operations=(PlanDeltaOperation.ATTACH_EVIDENCE.value,)
    )
    state = _live_state(request)
    # No proposed items and ADD_TASK not allowed -> empty delta path.
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(request=request, live_state=state)
    )
    assert receipt.admitted is False
    assert receipt.reason_codes


# ---------------------------------------------------------------------------
# Body-free / read-only / restart-serializable invariants
# ---------------------------------------------------------------------------


def test_live_state_rejects_body_bearing_fields() -> None:
    request = _steer_request()
    with pytest.raises(PlanSteerServiceError, match="secret|body"):
        _live_state(
            request,
            scan={
                "scan_cid": _cid("scan-current"),
                "repository_root_cid": request.roots.repository_root_cid,
                "dirty_worktree_root": request.roots.dirty_worktree_root,
                "source_body": "secret file contents",
            },
        )


def test_preview_receipt_is_body_free_and_read_only() -> None:
    request = _steer_request()
    state = _live_state(request)
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    payload = receipt.to_dict()
    assert payload["read_only"] is True
    assert payload["wrote_task_source"] is False
    assert payload["restart_serializable"] is True
    assert "schema" in payload
    # Round-trip preserves identity.
    again = PlanSteerPreviewReceipt.from_dict(receipt.to_record())
    assert again.receipt_cid == receipt.receipt_cid
    assert again.delta_cid == receipt.delta_cid
    assert again.population_partition.partition_cid == (
        receipt.population_partition.partition_cid
    )


def test_receipt_store_persists_canonical_record() -> None:
    store: dict[str, object] = {}
    request = _steer_request()
    state = _live_state(request)
    service = PlanSteerService(receipt_store=store)
    receipt = service.preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.content_id in store
    assert store[receipt.content_id]["content_id"] == receipt.content_id


def test_scan_impact_round_trip() -> None:
    impact = PlanSteerScanImpact(
        scan_receipt_cid=_cid("scan"),
        repository_root_cid=_cid("repo"),
        dirty_worktree_root=_cid("dirty"),
        base_plan_root=_cid("base"),
        impacted_paths=(
            "ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py",
        ),
        modified_paths=(
            "ipfs_accelerate_py/agent_supervisor/prompt/plan_steer_service.py",
        ),
    )
    restored = PlanSteerScanImpact.from_dict(impact.to_record())
    assert restored.content_id == impact.content_id


def test_task_record_from_value_normalizes_aliases() -> None:
    record = PlanSteerTaskRecord.from_value(
        {
            "task_cid": _cid("t1"),
            "status": "in_progress",
            "spec_revision": _cid("s1"),
        }
    )
    assert record.lifecycle_state is LifecycleState.RUNNING


def test_query_planner_injection_binds_query_plan_cid() -> None:
    class _Planner:
        def compile(self, request: object) -> object:
            class _Plan:
                plan_id = _cid("query-plan")

            return _Plan()

    request = _steer_request()
    state = _live_state(request)
    receipt = PlanSteerService(query_planner=_Planner()).preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is True
    assert receipt.query_plan_cid == _cid("query-plan")


def test_root_observer_can_fail_stale_observation() -> None:
    request = _steer_request()
    state = _live_state(request)

    def observe(_request: object, _state: object) -> dict[str, object]:
        return {
            "repository_root_cid": request.roots.repository_root_cid,
            "dirty_worktree_root": request.roots.dirty_worktree_root,
            "policy_root": _cid("policy-drift"),
            "task_source_revision": request.roots.task_source_revision,
            "plan_revision": request.plan_revision,
            "event_cursor": request.event_cursor,
        }

    receipt = PlanSteerService(root_observer=observe).preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.STALE_ROOT.value in receipt.reason_codes


def test_candidate_plan_preserves_claimed_and_accepted_history() -> None:
    request = _steer_request()
    state = _live_state(request)
    service = PlanSteerService()
    materials = PlanSteerPreviewMaterials(
        request=request,
        live_state=state,
        proposed_delta_items=(_successor_item(), _supersede_unstarted_item()),
    )
    receipt = service.preview_steer(materials)
    assert receipt.admitted is True
    # Re-run internal apply to inspect candidate populations via service path
    # already completed; ensure claimed family still in claimed digest.
    partition = service.partition_populations(state)
    assert TASK_RUNNING in (
        set(partition.running.member_cids)
        | set(partition.claimed.member_cids)
        | set(partition.settling.member_cids)
    )
    assert TASK_DONE in partition.accepted.member_cids
    # Supersede only touched unstarted ready task, not running.
    assert TASK_RUNNING not in set(
        receipt.population_partition.superseded.member_cids
    )


def test_missing_scan_receipt_fails() -> None:
    request = _steer_request()
    state = _live_state(request, scan={}, impact={})
    receipt = PlanSteerService().preview_steer(
        PlanSteerPreviewMaterials(
            request=request,
            live_state=state,
            proposed_delta_items=(_successor_item(),),
        )
    )
    assert receipt.admitted is False
    assert PlanSteerRejectionCode.MISSING_SCAN.value in receipt.reason_codes


def test_interface_constant_is_stable() -> None:
    assert PlanSteerService.INTERFACE == "PlanSteerService@1"
    assert PlanSteerService.VERSION == 1
