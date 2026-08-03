"""LPR-038: post-apply doctor fixed-point validation.

Covers:
* Rebuild AST/graphs/KG/vector tombstones
* Cache/CAS invalidation
* Reparse/type/static/differential/proof/memory/effect/resource checks
* Redelta / reclose / replan / reprove until residual-free
* Bound exhaustion, oscillation, drift compensation
* Rollback failure → quarantine
* Neither incomplete nor quarantine claims completion or calls a model
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorPlan,
    DoctorAuthorityRoots,
    DoctorConsumerDisposition,
    DoctorEditSite,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction import (
    DeterministicDoctorTransaction,
    DoctorCheckoutLock,
    DoctorSandboxEnforcementLevel,
    DoctorSandboxPolicy,
    DoctorStepApplyRequest,
    DoctorStepApplyResult,
    DoctorStepDisposition,
    DoctorWriterLease,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    PathBeforeHash,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_fixed_point import (
    DEFAULT_FIXED_POINT_BOUND,
    DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE,
    PRODUCER_ID,
    CandidateDoctorFixedPointEvidence,
    DeterministicDoctorFixedPointError,
    DeterministicDoctorFixedPointValidator,
    DoctorCacheInvalidationEvidence,
    DoctorFixedPointDisposition,
    DoctorFixedPointIterationReceipt,
    DoctorFixedPointReason,
    DoctorRebuildEvidence,
    DoctorRecloseEvidence,
    DoctorRedeltaEvidence,
    DoctorReplanEvidence,
    DoctorReproveEvidence,
    DoctorStaticCheckEvidence,
    build_fixture_committed_transaction_report,
    daemon_require_doctor_fixed_point,
    validate_deterministic_doctor_fixed_point,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:lpr-038-fp",
        "forest_id": "forest:lpr-038",
        "tree_id": "tree:candidate",
        "overlay_id": "overlay:lpr-038",
        "file_root_id": "file-root:lpr-038",
        "ast_root_id": "ast:lpr-038",
        "graph_id": "graph:lpr-038",
        "corpus_id": "corpus:lpr-038",
        "index_id": "index:lpr-038",
        "model_id": "model:lpr-038",
        "cache_id": "cache:lpr-038",
        "operator_registry_id": "operators:lpr-038",
        "translator_id": "translator:lpr-038",
        "solver_id": "solver:lpr-038",
        "kernel_id": "kernel:lpr-038",
        "toolchain_id": "toolchain:lpr-038",
        "policy_id": "policy:lpr-038",
        "sandbox_id": "sandbox:lpr-038",
        "environment_id": "environment:lpr-038",
        "lease_id": "lease:writer-1",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _consumer(
    auth: DoctorAuthorityRoots,
    consumer_id: str = "consumer:one",
) -> DoctorConsumerDisposition:
    return DoctorConsumerDisposition(
        roots=auth,
        consumer_id=consumer_id,
        disposition=DoctorRepairDisposition.SUPPORTED,
        reason_codes=("ok",),
    )


def _admitted_plan(auth: DoctorAuthorityRoots | None = None) -> DeterministicDoctorPlan:
    auth = auth or roots()
    site = DoctorEditSite(
        path="pkg/caller.py",
        before_hash="sha256:pkg-caller.py",
        span_start=0,
        span_end=8,
        artifact_id="blob:caller",
    )
    step = DoctorPlanStep(
        step_id="step:migrate-one",
        kind="analytical",
        operator_id="operator:add-arg",
        consumer_ids=("consumer:one",),
        edit_site_refs=(site.content_id,),
        write_paths=("pkg/caller.py",),
    )
    return DeterministicDoctorPlan(
        roots=auth,
        plan_id="plan:lpr-038-fp",
        snapshot_id="snapshot:lpr-038",
        finding_ids=("finding:one",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=(_consumer(auth),),
        impact_closure_id="impact:lpr-038",
        steps=(step,),
        edit_sites=(site,),
        operator_ids=("operator:add-arg",),
        target_ref="symbol:target",
        value_source_ref="value:source",
        placement_ref="placement:site",
        selected_operator_id="operator:add-arg",
        permitted_read_paths=("pkg/caller.py",),
        permitted_write_paths=("pkg/caller.py",),
        lease_id="lease:writer-1",
        checkpoint_ref="checkpoint:content-addressed",
        rollback_ref="rollback:restore-checkpoint",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )


def _sandbox() -> DoctorSandboxPolicy:
    return DoctorSandboxPolicy(
        sandbox_id="sandbox:lpr-038",
        worktree_root_ref="worktree:candidate-1",
        permitted_paths=("pkg/caller.py",),
        enforcement_level=DoctorSandboxEnforcementLevel.ENFORCED,
    )


def _passing_applicator(request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
    return DoctorStepApplyResult(
        disposition=DoctorStepDisposition.PASSED,
        written_paths=request.step.write_paths,
        observed_before_hashes=tuple(
            PathBeforeHash(
                path=p,
                before_hash=request.checkpoint.hash_map().get(
                    p, f"sha256:{p.replace('/', '-')}"
                ),
            )
            for p in request.step.write_paths
        ),
    )


def _committed_report(plan: DeterministicDoctorPlan):
    # Post-PDR-052 execute requires independent effect verification; pure
    # fixed-point unit tests seal a committed provisional report with complete
    # effect receipts via the fixture helper.
    return build_fixture_committed_transaction_report(
        plan,
        transaction_id="txn:lpr-038-fp",
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        path_before_hashes=(
            PathBeforeHash(path="pkg/caller.py", before_hash="sha256:pkg-caller.py"),
        ),
        lease_id="lease:writer-1",
    )


def _iteration(
    *,
    iteration: int = 1,
    residual: bool = False,
    second_order: tuple[str, ...] = (),
    discharged_second: tuple[str, ...] = (),
    failed_checks: tuple[str, ...] = (),
    unplanned_deltas: tuple[str, ...] = (),
    residual_findings: tuple[str, ...] = (),
    requires_another: bool = False,
    plan_current: bool = True,
    all_promoted: bool = True,
    clean_rebuild: bool = True,
    cache_complete: bool = True,
    fingerprint: str = "",
    reclose_complete: bool | None = None,
    open_frontier: tuple[str, ...] = (),
    unresolved: tuple[str, ...] = (),
) -> DoctorFixedPointIterationReceipt:
    tree = "tree:candidate"
    residual_free_inputs = not residual and not requires_another and not residual_findings
    if reclose_complete is None:
        reclose_complete = residual_free_inputs and not unresolved and not open_frontier
        if second_order and set(second_order) - set(discharged_second):
            reclose_complete = False
    return DoctorFixedPointIterationReceipt(
        iteration=iteration,
        rebuild=DoctorRebuildEvidence(
            candidate_tree_id=tree,
            repository_index_id="repo-index:rebuilt",
            ast_index_id="ast-index:rebuilt",
            vector_row_ids=("vector:caller",),
            kg_node_ids=("kg:caller",),
            call_graph_id="call-graph:rebuilt",
            dependency_graph_id="dep-graph:rebuilt",
            schema_graph_id="schema-graph:rebuilt",
            value_graph_id="value-graph:rebuilt",
            tombstone_ids=("tombstone:vector:old", "tombstone:kg:old"),
            reparsed_paths=("pkg/caller.py",),
            clean_rebuild_equivalent=clean_rebuild,
        ),
        cache_invalidation=DoctorCacheInvalidationEvidence(
            candidate_tree_id=tree,
            invalidated_cache_ids=("cache:proof:old",),
            invalidated_cas_ids=("cas:index:old",),
            tombstone_ids=("tombstone:cache:old",),
            remaining_stale_ids=() if cache_complete else ("cache:stale",),
            complete=cache_complete,
        ),
        static_checks=DoctorStaticCheckEvidence(
            candidate_tree_id=tree,
            reparsed_paths=("pkg/caller.py",),
            type_check_receipt_ids=("typecheck:ok",),
            static_check_receipt_ids=("static:ok",),
            differential_check_receipt_ids=("diff:ok",),
            proof_check_receipt_ids=("proof:ok",),
            memory_effect_receipt_ids=("memory:ok",),
            resource_check_receipt_ids=("resource:ok",),
            failed_check_ids=failed_checks,
            all_passed=not failed_checks,
        ),
        redelta=DoctorRedeltaEvidence(
            candidate_tree_id=tree,
            original_delta_ids=("delta:one",),
            recomputed_delta_ids=("delta:one",),
            breaking_delta_ids=("delta:one",),
            unplanned_breaking_delta_ids=unplanned_deltas,
            matches_plan_delta=not unplanned_deltas,
        ),
        reclose=DoctorRecloseEvidence(
            candidate_tree_id=tree,
            original_finding_ids=("finding:one",),
            discharged_original_ids=("finding:one",)
            if reclose_complete and not unresolved
            else (),
            second_order_finding_ids=second_order,
            discharged_second_order_ids=discharged_second,
            unresolved_mandatory_ids=unresolved,
            open_required_frontier_ids=open_frontier,
            complete=reclose_complete,
        ),
        replan=DoctorReplanEvidence(
            candidate_tree_id=tree,
            diagnosis_root_id="diagnosis:current",
            tactician_plan_id="tactician:plan-1",
            goal_root_ids=("goal:caller-migrate",),
            residual_gap_ids=() if plan_current else ("gap:residual",),
            plan_current=plan_current,
        ),
        reprove=DoctorReproveEvidence(
            candidate_tree_id=tree,
            hammer_receipt_ids=("hammer:receipt-1",),
            native_goal_binding_ids=("native:goal-1",),
            prediction_receipt_ids=("prediction:admit-1",),
            stale_prediction_ids=() if all_promoted else ("prediction:stale",),
            failed_reconstruction_ids=(),
            all_promoted_clauses_current=all_promoted,
        ),
        residual_finding_ids=residual_findings,
        oscillation_fingerprint=fingerprint or f"fp:iter-{iteration}",
        requires_another_iteration=requires_another,
    )


def _evidence(
    auth: DoctorAuthorityRoots,
    *iterations: DoctorFixedPointIterationReceipt,
    replay: str = "replay:identity-1",
) -> CandidateDoctorFixedPointEvidence:
    if not iterations:
        iterations = (_iteration(),)
    return CandidateDoctorFixedPointEvidence(
        candidate_tree_id="tree:candidate",
        roots=auth,
        iterations=iterations,
        expected_tombstone_ids=("tombstone:vector:old",),
        identity_replay_receipt_id=replay,
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def test_interface_constant() -> None:
    assert DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE == (
        "DeterministicDoctorFixedPointValidator@1"
    )
    assert (
        DeterministicDoctorFixedPointValidator.INTERFACE
        == DETERMINISTIC_DOCTOR_FIXED_POINT_INTERFACE
    )
    assert PRODUCER_ID == "deterministic-doctor-fixed-point@1"
    assert DEFAULT_FIXED_POINT_BOUND == 8


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_residual_free_fixed_point_completes() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    assert report.committed

    outcome = validate_deterministic_doctor_fixed_point(
        plan,
        report,
        evidence=_evidence(auth, _iteration()),
    )
    assert outcome.complete
    assert outcome.fixed_point is not None
    assert outcome.fixed_point.disposition is DoctorFixedPointDisposition.COMPLETE
    assert outcome.fixed_point.model_invocation_count == 0
    assert outcome.fixed_point.provider_invocation_count == 0
    assert outcome.report.complete
    assert outcome.rolled_back is False
    assert outcome.quarantined is False
    payload = outcome.fixed_point.to_dict()
    assert payload["claims_completion"] is True
    assert payload["may_call_model"] is False
    assert payload["partial_merge_allowed"] is False


def test_daemon_require_complete() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    receipt = daemon_require_doctor_fixed_point(
        plan, report, evidence=_evidence(auth, _iteration())
    )
    assert receipt.complete
    assert receipt.identity_replay_receipt_id == "replay:identity-1"


def test_second_order_then_fixed_point() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    first = _iteration(
        iteration=1,
        second_order=("finding:second",),
        discharged_second=(),
        requires_another=True,
        residual_findings=("finding:second",),
        plan_current=False,
        all_promoted=False,
        reclose_complete=False,
        fingerprint="fp:open",
    )
    second = _iteration(
        iteration=2,
        second_order=("finding:second",),
        discharged_second=("finding:second",),
        fingerprint="fp:closed",
    )
    outcome = DeterministicDoctorFixedPointValidator().validate(
        plan,
        report,
        evidence=_evidence(auth, first, second),
    )
    assert outcome.complete
    assert outcome.report.iteration_count == 2


# ---------------------------------------------------------------------------
# Admission failures
# ---------------------------------------------------------------------------


def test_non_committed_transaction_incomplete() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    txn = DeterministicDoctorTransaction(
        step_applicator=lambda r: DoctorStepApplyResult(
            disposition=DoctorStepDisposition.FAILED,
            reason_codes=("step_failure",),
        )
    )
    report = txn.execute(
        plan,
        sandbox_policy=_sandbox(),
        checkout_lock=DoctorCheckoutLock(
            lock_id="lock:1",
            holder_id="holder:txn",
            worktree_root_ref="worktree:candidate-1",
            base_tree_cid="tree:base",
            fence_id="fence:lock",
        ),
        lease=DoctorWriterLease(
            lease_id="lease:writer-1",
            fence_id="fence:1",
            holder_id="holder:txn",
            permitted_write_paths=("pkg/caller.py",),
        ),
        path_before_hashes=(
            PathBeforeHash(path="pkg/caller.py", before_hash="sha256:pkg-caller.py"),
        ),
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
    )
    assert not report.committed
    outcome = validate_deterministic_doctor_fixed_point(
        plan, report, evidence=_evidence(auth, _iteration())
    )
    assert not outcome.complete
    assert DoctorFixedPointReason.TRANSACTION_NOT_PROVISIONAL.value in outcome.report.reason_codes


# ---------------------------------------------------------------------------
# Stage failures → rollback
# ---------------------------------------------------------------------------


def test_rebuild_incomplete_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = validate_deterministic_doctor_fixed_point(
        plan,
        report,
        evidence=_evidence(auth, _iteration(clean_rebuild=False)),
    )
    assert not outcome.complete
    assert outcome.rolled_back
    assert outcome.compensating_rollback is not None
    assert outcome.compensating_rollback.restored
    assert DoctorFixedPointReason.REBUILD_INCOMPLETE.value in outcome.report.reason_codes
    assert outcome.report.disposition is DoctorFixedPointDisposition.ROLLED_BACK
    assert outcome.report.disposition.claims_completion is False
    assert outcome.report.disposition.may_call_model is False


def test_missing_tombstone_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    # expected tombstone not present in rebuild
    evidence = CandidateDoctorFixedPointEvidence(
        candidate_tree_id="tree:candidate",
        roots=auth,
        iterations=(_iteration(),),
        expected_tombstone_ids=("tombstone:missing",),
        identity_replay_receipt_id="replay:identity-1",
    )
    # Override rebuild tombstones without the expected one.
    bad = _iteration()
    # reconstruct with empty tombstones
    bad = DoctorFixedPointIterationReceipt(
        iteration=1,
        rebuild=DoctorRebuildEvidence(
            candidate_tree_id="tree:candidate",
            repository_index_id="repo-index:rebuilt",
            ast_index_id="ast-index:rebuilt",
            vector_row_ids=("vector:caller",),
            kg_node_ids=("kg:caller",),
            call_graph_id="call-graph:rebuilt",
            dependency_graph_id="dep-graph:rebuilt",
            schema_graph_id="schema-graph:rebuilt",
            value_graph_id="value-graph:rebuilt",
            tombstone_ids=(),
            reparsed_paths=("pkg/caller.py",),
            clean_rebuild_equivalent=True,
        ),
        cache_invalidation=bad.cache_invalidation,
        static_checks=bad.static_checks,
        redelta=bad.redelta,
        reclose=bad.reclose,
        replan=bad.replan,
        reprove=bad.reprove,
    )
    evidence = CandidateDoctorFixedPointEvidence(
        candidate_tree_id="tree:candidate",
        roots=auth,
        iterations=(bad,),
        expected_tombstone_ids=("tombstone:missing",),
        identity_replay_receipt_id="replay:identity-1",
    )
    outcome = validate_deterministic_doctor_fixed_point(
        plan, report, evidence=evidence
    )
    assert not outcome.complete
    assert DoctorFixedPointReason.TOMBSTONE_MISSING.value in outcome.report.reason_codes


def test_cache_invalidation_incomplete_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = validate_deterministic_doctor_fixed_point(
        plan,
        report,
        evidence=_evidence(auth, _iteration(cache_complete=False)),
    )
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.CACHE_INVALIDATION_INCOMPLETE.value
        in outcome.report.reason_codes
    )


def test_static_check_failure_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = validate_deterministic_doctor_fixed_point(
        plan,
        report,
        evidence=_evidence(auth, _iteration(failed_checks=("check:type:fail",))),
    )
    assert not outcome.complete
    assert DoctorFixedPointReason.TYPE_CHECK_FAILED.value in outcome.report.reason_codes


def test_unplanned_breaking_delta_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = validate_deterministic_doctor_fixed_point(
        plan,
        report,
        evidence=_evidence(auth, _iteration(unplanned_deltas=("delta:surprise",))),
    )
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.UNPLANNED_BREAKING_DELTA.value
        in outcome.report.reason_codes
    )


def test_unresolved_mandatory_finding_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = validate_deterministic_doctor_fixed_point(
        plan,
        report,
        evidence=_evidence(
            auth,
            _iteration(
                unresolved=("finding:open",),
                reclose_complete=False,
                residual_findings=("finding:open",),
            ),
        ),
    )
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.UNRESOLVED_MANDATORY_FINDING.value
        in outcome.report.reason_codes
        or DoctorFixedPointReason.FIXED_POINT_NOT_REACHED.value
        in outcome.report.reason_codes
    )


# ---------------------------------------------------------------------------
# Bounds / oscillation / missing replay
# ---------------------------------------------------------------------------


def test_bound_exhaustion_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    iters = tuple(
        _iteration(
            iteration=i,
            requires_another=True,
            residual_findings=("finding:open",),
            reclose_complete=False,
            plan_current=False,
            fingerprint=f"fp:{i}",
        )
        for i in range(1, 4)
    )
    outcome = DeterministicDoctorFixedPointValidator(fixed_point_bound=2).validate(
        plan,
        report,
        evidence=_evidence(auth, *iters),
    )
    assert not outcome.complete
    assert DoctorFixedPointReason.BOUND_EXHAUSTED.value in outcome.report.reason_codes
    assert outcome.rolled_back


def test_oscillation_detected() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    # Alternating fingerprints within window while residual remains.
    iters = (
        _iteration(
            iteration=1,
            requires_another=True,
            residual_findings=("finding:open",),
            reclose_complete=False,
            plan_current=False,
            fingerprint="fp:a",
        ),
        _iteration(
            iteration=2,
            requires_another=True,
            residual_findings=("finding:open",),
            reclose_complete=False,
            plan_current=False,
            fingerprint="fp:b",
        ),
        _iteration(
            iteration=3,
            requires_another=True,
            residual_findings=("finding:open",),
            reclose_complete=False,
            plan_current=False,
            fingerprint="fp:a",
        ),
        _iteration(
            iteration=4,
            requires_another=True,
            residual_findings=("finding:open",),
            reclose_complete=False,
            plan_current=False,
            fingerprint="fp:b",
        ),
    )
    outcome = DeterministicDoctorFixedPointValidator(
        fixed_point_bound=8, oscillation_window=4
    ).validate(plan, report, evidence=_evidence(auth, *iters))
    assert not outcome.complete
    # Oscillation or bound / not-reached are all fail-closed compensations.
    assert outcome.rolled_back or outcome.quarantined
    assert outcome.report.disposition.claims_completion is False


def test_missing_identity_replay_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = validate_deterministic_doctor_fixed_point(
        plan,
        report,
        evidence=_evidence(auth, _iteration(), replay=""),
    )
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.IDENTITY_REPLAY_MISMATCH.value
        in outcome.report.reason_codes
    )


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


def test_rollback_failure_quarantines() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    validator = DeterministicDoctorFixedPointValidator(
        restore_adapter=lambda _ckpt: False
    )
    outcome = validator.validate(
        plan,
        report,
        evidence=_evidence(auth, _iteration(clean_rebuild=False)),
    )
    assert not outcome.complete
    assert outcome.quarantined
    assert not outcome.rolled_back
    assert outcome.compensating_rollback is not None
    assert outcome.compensating_rollback.quarantined
    assert DoctorFixedPointReason.QUARANTINE_REQUIRED.value in outcome.report.reason_codes
    assert outcome.report.disposition is DoctorFixedPointDisposition.QUARANTINED
    # Quarantine may never claim completion or call a model.
    assert outcome.report.disposition.claims_completion is False
    assert outcome.report.disposition.may_call_model is False
    assert outcome.to_dict()["claims_completion"] is False
    assert outcome.to_dict()["may_call_model"] is False


def test_require_complete_raises() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    with pytest.raises(DeterministicDoctorFixedPointError, match="rejected"):
        DeterministicDoctorFixedPointValidator().require_complete(
            plan,
            report,
            evidence=_evidence(auth, _iteration(clean_rebuild=False)),
        )


def test_malformed_input_incomplete() -> None:
    outcome = DeterministicDoctorFixedPointValidator().validate(
        "not-a-plan",  # type: ignore[arg-type]
        "not-a-report",  # type: ignore[arg-type]
        evidence="bad",  # type: ignore[arg-type]
    )
    assert not outcome.complete
    assert DoctorFixedPointReason.MALFORMED_INPUT.value in outcome.report.reason_codes


def test_deterministic_fixed_point_receipt_ids() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    evidence = _evidence(auth, _iteration())
    a = validate_deterministic_doctor_fixed_point(plan, report, evidence=evidence)
    b = validate_deterministic_doctor_fixed_point(plan, report, evidence=evidence)
    assert a.complete and b.complete
    assert a.fixed_point is not None and b.fixed_point is not None
    assert a.fixed_point.content_id == b.fixed_point.content_id
    assert a.report.report_id == b.report.report_id


def test_complete_receipt_forbids_residuals() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = validate_deterministic_doctor_fixed_point(
        plan, report, evidence=_evidence(auth, _iteration())
    )
    assert outcome.fixed_point is not None
    with pytest.raises(DeterministicDoctorFixedPointError):
        # Reconstruct with illegal residual on complete disposition.
        from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_fixed_point import (
            DoctorFixedPointReceipt,
        )

        DoctorFixedPointReceipt(
            roots=auth,
            receipt_id="receipt:bad",
            plan_id=plan.plan_id,
            transaction_id=report.transaction_id,
            candidate_tree_cid="tree:candidate",
            committed_tree_cid="tree:candidate",
            checkpoint_id=report.checkpoint.checkpoint_id,
            iteration_count=1,
            disposition=DoctorFixedPointDisposition.COMPLETE,
            iteration_receipt_ids=("iter:1",),
            residual_finding_ids=("finding:open",),
            identity_replay_receipt_id="replay:1",
        )
