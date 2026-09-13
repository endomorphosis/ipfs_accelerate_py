"""Independent current-tree checks for DOEP-045 stale-plan-epoch handling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    CompletionAuthority,
    DeltaEffectClass,
    LifecycleState,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
    plan_revision_cid,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    ControlPlaneContractError,
    TaskState,
    TaskStateSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    REVISION_CAS_TRANSITION_BINDING,
    STATE_TRANSACTION_INTERFACE,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    MIN_PLAN_EPOCH,
    PLAN_REVISION_STORE_INTERFACE,
    STALE_PLAN_EPOCH_BINDING,
    PlanRevisionApplyRequest,
    PlanRevisionStore,
    PlanRevisionStoreStalePlanEpochError,
    assert_plan_epoch_current,
    assert_task_cas_completion_plan_epoch,
    assert_task_snapshot_plan_epoch_current,
    plan_epoch_is_current,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
STORE_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-045.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-045.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py",
    "test/api/doep/test_doep_045_add_stale_plan_epoch_handling.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-045.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-045.json",
)
TASK_CID = "sha256:d06323f1ccdd5b1cca2302ed1a069dd61c673a6e5a71903e4580d40541f42074"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _cid(name: str) -> str:
    return plan_revision_cid({"fixture": name})


def _roots() -> PlanAuthorityRoots:
    return PlanAuthorityRoots(
        repository_id="repository:sha256:doep-045",
        repository_root_cid=_cid("repo-root"),
        dirty_worktree_root=_cid("dirty"),
        task_source_id="task-source:markdown:doep-045",
        task_source_revision=_cid("ts-rev-1"),
        policy_root=_cid("policy"),
        intent_ir_root=_cid("intent"),
        legal_ir_root=_cid("legal"),
        security_ir_root=_cid("security"),
        program_root=_cid("program"),
        capability_catalog_root=_cid("capability"),
        provider_catalog_root=_cid("provider-catalog"),
        usage_policy_root=_cid("usage"),
        configuration_root=_cid("config"),
    )


def _population(kind: PopulationKind, *members: str) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=members)


def _revision(**changes: object) -> PlanRevision:
    values: dict[str, object] = {
        "plan_root_cid": _cid("plan-root-1"),
        "semantic_revision": 1,
        "parent_plan_root": "",
        "origin": PlanOrigin.CREATE,
        "roots": _roots(),
        "request_cid": _cid("create-request"),
        "delta_cid": "",
        "scan_receipt_cid": _cid("scan"),
        "query_plan_cid": _cid("query"),
        "evidence_bundle_cid": _cid("evidence"),
        "admission_receipt_cid": _cid("admission"),
        "execution_plan_cid": _cid("exec-plan"),
        "goal_population": _population(PopulationKind.RETAINED, _cid("goal-1")),
        "task_population": _population(PopulationKind.RETAINED, _cid("task-1")),
        "added_population": _population(
            PopulationKind.ADDED, _cid("goal-1"), _cid("task-1")
        ),
        "superseded_population": _population(PopulationKind.SUPERSEDED),
        "retained_population": _population(PopulationKind.RETAINED),
        "deferred_population": _population(PopulationKind.DEFERRED),
        "claimed_population": _population(PopulationKind.CLAIMED),
        "completed_population": _population(PopulationKind.COMPLETED),
        "blocked_population": _population(PopulationKind.BLOCKED),
        "resource_contract": PlanResourceContract(),
        "provider_contract": PlanProviderContract(),
        "lease_contract": PlanLeaseContract(),
        "retry_contract": PlanRetryContract(),
        "worktree_contract": PlanWorktreeContract(),
        "merge_strategy": PlanMergeStrategy(kind=MergeStrategyKind.SERIAL),
        "conflict_contract": PlanConflictContract(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py",
            ),
        ),
        "completion_rule": PlanCompletionRule(
            authority=CompletionAuthority.VALIDATION_GATE,
        ),
        "validation_dag": (
            PlanValidationNode(
                validation_key="validation:pytest",
                argv=("python", "-m", "pytest", "-q"),
            ),
        ),
        "event_cursor": _cid("cursor-0"),
    }
    values.update(changes)
    return PlanRevision(**values)


def _delta(base_plan_root: str) -> PlanDelta:
    return PlanDelta(
        base_plan_root=base_plan_root,
        base_plan_revision=1,
        request_cid=_cid("steer-request"),
        roots=_roots(),
        items=(
            PlanDeltaItem(
                item_key="delta:add-task",
                operation=PlanDeltaOperation.ADD_TASK,
                target_cid="",
                expected_target_lifecycle=LifecycleState.PROPOSED,
                expected_target_spec_revision="",
                before_digest="",
                after_record_cid=_cid("new-task"),
                effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
                rationale="Add a successor task.",
                expected_effects=("append-task",),
            ),
        ),
        expected_effects=("append-task",),
        claimed_population_digest=_cid("claimed-pop"),
        accepted_population_digest=_cid("accepted-pop"),
        scan_receipt_cid=_cid("scan"),
        evidence_bundle_cid=_cid("evidence"),
        admission_receipt_cid=_cid("admission"),
    )


def _snapshot(**updates: object) -> TaskStateSnapshot:
    values: dict[str, object] = {
        "task_cid": TASK_CID,
        "state": TaskState.IN_PROGRESS,
        "revision": 7,
        "lease_id": "lease:current",
        "fence_epoch": 3,
        "policy_cid": "policy:current",
        "repository_tree_id": "tree:current",
        "plan_cid": PLAN_CID,
        "plan_epoch": 1,
    }
    values.update(updates)
    return TaskStateSnapshot(**values)


def _steer_request(
    *,
    base: PlanRevision,
    plan_epoch: int,
    expected_plan_epoch: int = 0,
    worker_assertion: bool = False,
    root_name: str = "plan-root-2",
) -> PlanRevisionApplyRequest:
    delta = _delta(base.plan_root_cid)
    child = _revision(
        plan_root_cid=_cid(root_name),
        semantic_revision=2,
        parent_plan_root=base.plan_root_cid,
        origin=PlanOrigin.STEER,
        roots=base.roots,
        request_cid=_cid("steer-request"),
        delta_cid=delta.delta_cid,
        task_population=_population(
            PopulationKind.RETAINED, _cid("task-1"), _cid("new-task")
        ),
        added_population=_population(PopulationKind.ADDED, _cid("new-task")),
        retained_population=_population(PopulationKind.RETAINED, _cid("task-1")),
    )
    return PlanRevisionApplyRequest(
        revision=child,
        observed_roots=child.roots,
        idempotency_key=f"idem:{root_name}:{plan_epoch}",
        expected_effects=delta.expected_effects,
        delta=delta,
        expected_active_plan_root=base.plan_root_cid,
        plan_epoch=plan_epoch,
        expected_plan_epoch=expected_plan_epoch,
        worker_assertion=worker_assertion,
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_stale_plan_epoch_helpers_consume_canonical_bindings() -> None:
    assert PLAN_REVISION_STORE_INTERFACE == "PlanRevisionStore@1"
    assert STALE_PLAN_EPOCH_BINDING == "StalePlanEpochHandling@1"
    assert PlanRevisionStore.INTERFACE == PLAN_REVISION_STORE_INTERFACE
    assert PlanRevisionStore.STALE_PLAN_EPOCH_BINDING == STALE_PLAN_EPOCH_BINDING
    assert (
        PlanRevisionStore.CONSUMES_TASK_STATE_MACHINE
        == CANONICAL_TASK_STATE_MACHINE_INTERFACE
    )
    assert PlanRevisionStore.CONSUMES_REVISION_CAS == REVISION_CAS_TRANSITION_BINDING
    assert PlanRevisionStore.CONSUMES_STATE_TRANSACTION == STATE_TRANSACTION_INTERFACE
    assert MIN_PLAN_EPOCH == 1
    assert assert_plan_epoch_current(1, 1) == 1
    assert plan_epoch_is_current(1, 1)
    assert not plan_epoch_is_current(1, 2, worker_assertion=True)
    with pytest.raises(PlanRevisionStoreStalePlanEpochError) as stale:
        assert_plan_epoch_current(1, 2, worker_assertion=True)
    assert stale.value.expected_plan_epoch == 1
    assert stale.value.live_plan_epoch == 2
    current = _snapshot(plan_epoch=2)
    with pytest.raises(PlanRevisionStoreStalePlanEpochError):
        assert_task_snapshot_plan_epoch_current(
            _snapshot(plan_epoch=1), 2, current=current, worker_assertion=True
        )
    assert (
        assert_task_snapshot_plan_epoch_current(_snapshot(plan_epoch=2), 2) == 2
    )
    with pytest.raises(ControlPlaneContractError):
        assert_task_cas_completion_plan_epoch(
            _snapshot(state=TaskState.READY, revision=4, plan_epoch=1),
            _snapshot(state=TaskState.COMPLETED, revision=5, plan_epoch=1),
            1,
        )


def test_store_rejects_stale_epoch_apply_and_advances_monotonically(
    tmp_path: Path,
) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    assert store.current_plan_epoch() == 0
    base = _revision()
    created = store.apply(
        PlanRevisionApplyRequest(
            revision=base,
            observed_roots=base.roots,
            idempotency_key="idem:create",
            expected_effects=("create",),
            plan_epoch=1,
        )
    )
    assert created.committed
    assert created.plan_epoch == 1
    assert store.current_plan_epoch() == 1
    assert store.get_active() is not None
    assert store.get_active().plan_epoch == 1  # type: ignore[union-attr]

    same_epoch = store.apply(
        _steer_request(
            base=base,
            plan_epoch=1,
            expected_plan_epoch=1,
            root_name="plan-root-same-epoch",
        )
    )
    assert same_epoch.committed
    assert same_epoch.plan_epoch == 1
    assert store.current_plan_epoch() == 1

    with pytest.raises(PlanRevisionStoreStalePlanEpochError):
        store.journal_intent(
            _steer_request(
                base=store.load_revision(store.get_active().revision_cid),  # type: ignore[union-attr]
                plan_epoch=1,
                expected_plan_epoch=2,
                worker_assertion=True,
                root_name="plan-root-expected-mismatch",
            )
        )

    live = store.get_active()
    assert live is not None
    live_revision = store.load_revision(live.revision_cid)
    with pytest.raises(PlanRevisionStoreStalePlanEpochError):
        store.apply(
            _steer_request(
                base=live_revision,
                plan_epoch=3,
                expected_plan_epoch=1,
                root_name="plan-root-skip",
            )
        )

    advanced = store.apply(
        _steer_request(
            base=live_revision,
            plan_epoch=2,
            expected_plan_epoch=1,
            root_name="plan-root-epoch-2",
        )
    )
    assert advanced.committed
    assert advanced.plan_epoch == 2
    assert store.current_plan_epoch() == 2
    events = store.list_events()
    assert any(row.get("event_type") == "plan_epoch_advanced" for row in events)

    with pytest.raises(PlanRevisionStoreStalePlanEpochError):
        store.apply(
            _steer_request(
                base=store.load_revision(store.get_active().revision_cid),  # type: ignore[union-attr]
                plan_epoch=1,
                expected_plan_epoch=1,
                worker_assertion=True,
                root_name="plan-root-stale",
            )
        )


def test_stale_plan_epoch_cannot_complete_even_with_worker_assertion(
    tmp_path: Path,
) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    base = _revision()
    store.apply(
        PlanRevisionApplyRequest(
            revision=base,
            observed_roots=base.roots,
            idempotency_key="idem:complete-base",
            expected_effects=("create",),
            plan_epoch=1,
        )
    )
    live_revision = store.load_revision(store.get_active().revision_cid)  # type: ignore[union-attr]
    store.apply(
        _steer_request(
            base=live_revision,
            plan_epoch=2,
            expected_plan_epoch=1,
            root_name="plan-root-complete-epoch-2",
        )
    )
    assert store.current_plan_epoch() == 2
    stale = _snapshot(plan_epoch=1)
    current = _snapshot(plan_epoch=1)
    with pytest.raises(PlanRevisionStoreStalePlanEpochError):
        store.assert_task_may_complete(stale, current=current, worker_assertion=True)
    with pytest.raises(PlanRevisionStoreStalePlanEpochError):
        store.assert_task_cas_completion(
            stale,
            _snapshot(state=TaskState.COMPLETED, revision=8, plan_epoch=1),
            current=current,
            worker_assertion=True,
        )
    current_epoch = _snapshot(plan_epoch=2)
    assert store.assert_task_may_complete(current_epoch) == 2
    assert (
        store.assert_task_cas_completion(
            current_epoch,
            _snapshot(state=TaskState.COMPLETED, revision=8, plan_epoch=2),
        )
        == 8
    )
    with pytest.raises(ControlPlaneContractError):
        store.assert_task_cas_completion(
            _snapshot(state=TaskState.CLAIMED, revision=7, plan_epoch=2),
            _snapshot(state=TaskState.COMPLETED, revision=8, plan_epoch=2),
            worker_assertion=True,
        )


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-045"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["carrier"] == "PlanRevisionStore"
    assert manifest["canonical_extension"]["binding"] == PLAN_REVISION_STORE_INTERFACE
    assert (
        manifest["canonical_extension"]["stale_plan_epoch_binding"]
        == STALE_PLAN_EPOCH_BINDING
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(STORE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == "pending_independent_fenced_supervisor"
