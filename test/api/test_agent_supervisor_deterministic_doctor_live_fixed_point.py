"""PDR-053: live reparse/static/security/replan/reprove fixed-point stages.

Covers:
* Independent reparse/rebuild/re-index from file bytes
* Dependency-local cache invalidation
* Contract redelta and consumer/SCC reclose
* Code security facts + IntentIR/code forbidden logic + hyperproperties
* Second-order findings → bounded re-iteration
* Oscillation / unchanged residual / budget / capability loss → abort+rollback
* Prebuilt fixed-point mappings or booleans cannot complete
* Pure validator still accepts sealed live receipts
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
from ipfs_accelerate_py.agent_supervisor.security_contract_analysis import (
    FindingClassification,
    SecurityRuleFamily,
    check_intent_code_forbidden_logic,
    evaluate_fixed_point_security,
    extract_code_security_facts,
    make_evidence,
    make_flow_edge,
    make_flow_node,
    make_security_property,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_fixed_point import (
    DeterministicDoctorFixedPointError,
    DeterministicDoctorFixedPointValidator,
    DoctorFixedPointDisposition,
    DoctorFixedPointReason,
    build_fixture_committed_transaction_report,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_live_fixed_point import (
    DETERMINISTIC_DOCTOR_LIVE_FIXED_POINT_INTERFACE,
    LIVE_FIXED_POINT_PRODUCER_ID,
    DeterministicDoctorLiveFixedPoint,
    DeterministicDoctorLiveFixedPointError,
    LiveFixedPointAbortReason,
    LiveFixedPointRequest,
    daemon_require_live_doctor_fixed_point,
    reject_prebuilt_completion,
    run_live_doctor_fixed_point,
)


# ---------------------------------------------------------------------------
# Shared fixtures (mirrors pure fixed-point tests)
# ---------------------------------------------------------------------------


def roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:pdr-053-fp",
        "forest_id": "forest:pdr-053",
        "tree_id": "tree:candidate",
        "overlay_id": "overlay:pdr-053",
        "file_root_id": "file-root:pdr-053",
        "ast_root_id": "ast:pdr-053",
        "graph_id": "graph:pdr-053",
        "corpus_id": "corpus:pdr-053",
        "index_id": "index:pdr-053",
        "model_id": "model:pdr-053",
        "cache_id": "cache:pdr-053",
        "operator_registry_id": "operators:pdr-053",
        "translator_id": "translator:pdr-053",
        "solver_id": "solver:pdr-053",
        "kernel_id": "kernel:pdr-053",
        "toolchain_id": "toolchain:pdr-053",
        "policy_id": "policy:pdr-053",
        "sandbox_id": "sandbox:pdr-053",
        "environment_id": "environment:pdr-053",
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
        plan_id="plan:pdr-053-fp",
        snapshot_id="snapshot:pdr-053",
        finding_ids=("finding:one",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=(_consumer(auth),),
        impact_closure_id="impact:pdr-053",
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
        sandbox_id="sandbox:pdr-053",
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
    return build_fixture_committed_transaction_report(
        plan,
        transaction_id="txn:pdr-053-fp",
        base_tree_cid="tree:base",
        candidate_tree_cid="tree:candidate",
        path_before_hashes=(
            PathBeforeHash(path="pkg/caller.py", before_hash="sha256:pkg-caller.py"),
        ),
        lease_id="lease:writer-1",
    )


def _valid_source() -> bytes:
    return b"import os\n\ndef caller(x):\n    return x + 1\n"


def _happy_request(**overrides: object) -> LiveFixedPointRequest:
    base: dict[str, object] = {
        "changed_paths": ("pkg/caller.py",),
        "file_bytes": {"pkg/caller.py": _valid_source()},
        "original_finding_ids": ("finding:one",),
        "original_delta_ids": ("delta:plan",),
        "prior_cache_ids": ("cache:stale-1",),
        "expected_tombstone_ids": ("tombstone:pkg/caller.py",),
        "intent_effects": ("effect:pkg/caller.py",),
        "code_effects": ("effect:pkg/caller.py",),
        "effects_by_path": {"pkg/caller.py": ("effect:pkg/caller.py",)},
        "required_hyperproperty_ids": ("worktree_isolation",),
        "held_hyperproperty_receipt_ids": (
            "hyperproperty:worktree_isolation",
        ),
    }
    base.update(overrides)
    return LiveFixedPointRequest(**base)  # type: ignore[arg-type]


def _live_runner(
    *,
    restore: bool = True,
) -> DeterministicDoctorLiveFixedPoint:
    return DeterministicDoctorLiveFixedPoint(
        restore_adapter=lambda _ckpt: restore,
        require_independent_restore=True,
    )


# ---------------------------------------------------------------------------
# Interface / prebuilt rejection
# ---------------------------------------------------------------------------


def test_live_interface_constant() -> None:
    assert DETERMINISTIC_DOCTOR_LIVE_FIXED_POINT_INTERFACE == (
        "DeterministicDoctorLiveFixedPoint@1"
    )
    assert LIVE_FIXED_POINT_PRODUCER_ID.startswith("deterministic-doctor-live")
    assert DeterministicDoctorLiveFixedPoint.INTERFACE == (
        DETERMINISTIC_DOCTOR_LIVE_FIXED_POINT_INTERFACE
    )


def test_prebuilt_boolean_cannot_complete() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(prebuilt_complete=True)
    assert reject_prebuilt_completion(request)
    outcome = _live_runner().run(plan, report, request)
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.PREBUILT_EVIDENCE_REJECTED.value
        in outcome.report.reason_codes
    )
    assert LiveFixedPointAbortReason.PREBUILT_BOOLEAN.value in outcome.report.reason_codes
    assert outcome.fixed_point is None
    assert outcome.report.disposition.claims_completion is False


def test_prebuilt_mapping_cannot_complete() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(
        prebuilt_fixed_point={"complete": True, "residual_free": True}
    )
    outcome = run_live_doctor_fixed_point(
        plan,
        report,
        request,
        # use default runner path via module wrapper — restore fails closed
    )
    # Module wrapper uses require_independent_restore=True with no adapter →
    # quarantine after prebuilt reject. Either way completion is forbidden.
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.PREBUILT_EVIDENCE_REJECTED.value
        in outcome.report.reason_codes
    )
    assert outcome.report.disposition.claims_completion is False


def test_prebuilt_false_boolean_still_rejected() -> None:
    """Even a False boolean is caller-supplied and cannot authorize the path."""

    reasons = reject_prebuilt_completion(
        LiveFixedPointRequest(prebuilt_complete=False)
    )
    assert DoctorFixedPointReason.PREBUILT_EVIDENCE_REJECTED.value in reasons


# ---------------------------------------------------------------------------
# Happy path: independent stages reach residual-free fixed point
# ---------------------------------------------------------------------------


def test_live_residual_free_fixed_point_completes() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    assert report.committed

    outcome = _live_runner().run(plan, report, _happy_request())
    assert outcome.complete, outcome.report.reason_codes
    assert outcome.fixed_point is not None
    assert outcome.fixed_point.disposition is DoctorFixedPointDisposition.COMPLETE
    assert outcome.fixed_point.model_invocation_count == 0
    assert outcome.report.iteration_count >= 1
    # Live evidence always includes security.
    last = outcome.report.iteration_receipts[-1]
    assert last.security is not None
    assert last.security.all_passed
    assert last.rebuild.clean_rebuild_equivalent
    assert last.cache_invalidation.complete
    assert last.static_checks.all_passed
    assert last.reclose.complete
    assert last.replan.plan_current
    assert last.reprove.all_promoted_clauses_current
    assert last.residual_free


def test_live_reparses_and_rebuilds_from_bytes() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    source = b"from pkg import util\n\ndef caller():\n    return util.f()\n"
    outcome = _live_runner().run(
        plan,
        report,
        _happy_request(
            file_bytes={"pkg/caller.py": source},
            effects_by_path={"pkg/caller.py": ("effect:pkg/caller.py",)},
        ),
    )
    assert outcome.complete, outcome.report.reason_codes
    rebuild = outcome.report.iteration_receipts[-1].rebuild
    assert "pkg/caller.py" in rebuild.reparsed_paths
    assert rebuild.repository_index_id
    assert rebuild.ast_index_id
    assert rebuild.call_graph_id
    assert rebuild.dependency_graph_id


def test_live_invalidates_dependency_local_caches() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    outcome = _live_runner().run(
        plan,
        report,
        _happy_request(prior_cache_ids=("cache:old-1", "cache:old-2")),
    )
    assert outcome.complete
    cache = outcome.report.iteration_receipts[-1].cache_invalidation
    assert cache.complete
    assert not cache.remaining_stale_ids
    assert "cache:old-1" in cache.invalidated_cache_ids
    assert any(item.startswith("cas:") for item in cache.invalidated_cas_ids)


def test_daemon_require_live_complete() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    runner = _live_runner()
    receipt = runner.require_complete(plan, report, _happy_request())
    assert receipt.complete
    assert receipt.identity_replay_receipt_id


# ---------------------------------------------------------------------------
# Second-order iteration
# ---------------------------------------------------------------------------


def test_second_order_findings_trigger_another_iteration() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(
        second_order_schedule={1: ("finding:second",)},
        discharge_schedule={2: ("finding:second",)},
        fixed_point_bound=4,
    )
    outcome = _live_runner().run(plan, report, request)
    assert outcome.complete, outcome.report.reason_codes
    assert outcome.report.iteration_count == 2
    first = outcome.report.iteration_receipts[0]
    second = outcome.report.iteration_receipts[1]
    assert "finding:second" in first.reclose.second_order_finding_ids
    assert not first.residual_free
    assert second.residual_free
    assert "finding:second" in second.reclose.discharged_second_order_ids


# ---------------------------------------------------------------------------
# Security / forbidden logic / hyperproperties
# ---------------------------------------------------------------------------


def test_forbidden_logic_aborts_and_rolls_back() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(
        intent_effects=("effect:pkg/caller.py", "effect:forbidden-op"),
        code_effects=("effect:pkg/caller.py", "effect:forbidden-op"),
        forbidden_effect_ids=("effect:forbidden-op",),
        effects_by_path={
            "pkg/caller.py": ("effect:pkg/caller.py", "effect:forbidden-op")
        },
    )
    outcome = _live_runner(restore=True).run(plan, report, request)
    assert not outcome.complete
    codes = set(outcome.report.reason_codes)
    assert DoctorFixedPointReason.FORBIDDEN_LOGIC_VIOLATION.value in codes or (
        DoctorFixedPointReason.SECURITY_CHECK_FAILED.value in codes
    )
    assert outcome.rolled_back or outcome.quarantined
    assert outcome.report.disposition.claims_completion is False


def test_missing_required_hyperproperty_aborts() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(
        required_hyperproperty_ids=("prompt_isolation", "worktree_isolation"),
        held_hyperproperty_receipt_ids=("hyperproperty:worktree_isolation",),
        # prompt_isolation missing
    )
    outcome = _live_runner(restore=True).run(plan, report, request)
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.HYPERPROPERTY_FAILED.value in outcome.report.reason_codes
        or DoctorFixedPointReason.SECURITY_CHECK_FAILED.value
        in outcome.report.reason_codes
    )


def test_security_flow_vulnerability_blocks_completion() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    nodes = (
        make_flow_node(
            "n:src",
            "pkg.api.user_path",
            role="source",
            tags=("untrusted_path",),
            path="pkg/caller.py",
        ),
        make_flow_node(
            "n:sink",
            "pkg.vfs.open",
            role="sink",
            tags=("fs_open",),
            path="pkg/caller.py",
        ),
    )
    edges = (make_flow_edge("e1", "n:src", "n:sink"),)
    prop = make_security_property(
        "prop:path",
        SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
        resource="vfs.root",
        statement="Paths must stay under the declared root.",
    )
    request = _happy_request(
        flow_nodes=nodes,
        flow_edges=edges,
        security_properties=(prop,),
        security_evidence=make_evidence("artifact:cex:1"),
    )
    outcome = _live_runner(restore=True).run(plan, report, request)
    # Vulnerability findings must prevent residual-free completion.
    assert not outcome.complete
    assert outcome.report.disposition.claims_completion is False


def test_extract_code_security_facts_and_forbidden_helpers() -> None:
    facts = extract_code_security_facts(
        paths=("pkg/caller.py",),
        effects_by_path={"pkg/caller.py": ("effect:a",)},
        tree_id="tree:candidate",
    )
    assert len(facts) == 1
    assert facts[0].path == "pkg/caller.py"
    assert facts[0].effect_id == "effect:a"

    dual = check_intent_code_forbidden_logic(
        intent_effects=("effect:a",),
        code_effects=("effect:a",),
        forbidden_effect_ids=(),
        covered_effect_ids=("effect:a",),
    )
    assert dual.passed

    gap = check_intent_code_forbidden_logic(
        intent_effects=("effect:a",),
        code_effects=(),
        forbidden_effect_ids=(),
    )
    assert not gap.passed
    assert "intent_code_stream_gap" in gap.reason_codes

    receipt = evaluate_fixed_point_security(
        candidate_tree_id="tree:candidate",
        code_facts=facts,
        intent_effects=("effect:a",),
        code_effects=("effect:a",),
        required_hyperproperty_ids=("worktree_isolation",),
        held_hyperproperty_receipt_ids=("hyperproperty:worktree_isolation",),
        run_flow_analysis=False,
    )
    assert receipt.all_passed
    assert receipt.receipt_id


# ---------------------------------------------------------------------------
# Abort conditions
# ---------------------------------------------------------------------------


def test_parse_error_rebuild_fails() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(file_bytes={"pkg/caller.py": b"def broken(:\n"})
    outcome = _live_runner(restore=True).run(plan, report, request)
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.REBUILD_INCOMPLETE.value in outcome.report.reason_codes
        or DoctorFixedPointReason.REPARSE_FAILED.value in outcome.report.reason_codes
        or DoctorFixedPointReason.STATIC_CHECK_FAILED.value
        in outcome.report.reason_codes
    )
    assert outcome.rolled_back


def test_unchanged_residual_aborts() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    # Same second-order finding never discharged → identical residual fingerprints.
    request = _happy_request(
        second_order_schedule={
            1: ("finding:stuck",),
            2: ("finding:stuck",),
            3: ("finding:stuck",),
        },
        discharge_schedule={},
        fixed_point_bound=4,
    )
    outcome = _live_runner(restore=True).run(plan, report, request)
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.UNCHANGED_RESIDUAL.value in outcome.report.reason_codes
        or DoctorFixedPointReason.BOUND_EXHAUSTED.value in outcome.report.reason_codes
        or DoctorFixedPointReason.FIXED_POINT_NOT_REACHED.value
        in outcome.report.reason_codes
    )
    assert outcome.rolled_back or outcome.quarantined


def test_capability_loss_aborts() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    runner = DeterministicDoctorLiveFixedPoint(
        restore_adapter=lambda _ckpt: True,
        capability_probe=lambda: (False, ("hammer_backend_missing",)),
    )
    outcome = runner.run(plan, report, _happy_request())
    assert not outcome.complete
    assert DoctorFixedPointReason.CAPABILITY_LOST.value in outcome.report.reason_codes
    assert "hammer_backend_missing" in outcome.report.reason_codes


def test_budget_exhaustion_aborts() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request(max_stage_invocations=2, fixed_point_bound=4)
    outcome = _live_runner(restore=True).run(plan, report, request)
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.BUDGET_EXHAUSTED.value in outcome.report.reason_codes
        or LiveFixedPointAbortReason.BUDGET_EXHAUSTED.value
        in outcome.report.reason_codes
    )


def test_restore_failure_quarantines() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    # Force a stage failure with restore=False.
    request = _happy_request(file_bytes={"pkg/caller.py": b"def broken(:\n"})
    outcome = _live_runner(restore=False).run(plan, report, request)
    assert not outcome.complete
    assert outcome.quarantined
    assert not outcome.rolled_back
    assert outcome.report.disposition is DoctorFixedPointDisposition.QUARANTINED
    assert outcome.report.disposition.claims_completion is False
    assert outcome.report.disposition.may_call_model is False


def test_require_complete_raises_on_failure() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    with pytest.raises(DeterministicDoctorFixedPointError):
        _live_runner().require_complete(
            plan,
            report,
            _happy_request(file_bytes={"pkg/caller.py": b"def broken(:\n"}),
        )


# ---------------------------------------------------------------------------
# Pure validator compatibility with live security evidence
# ---------------------------------------------------------------------------


def test_pure_validator_accepts_live_produced_evidence() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    live = _live_runner().run(plan, report, _happy_request())
    assert live.complete
    # Re-validate the sealed evidence through the pure validator alone.
    evidence_iterations = live.report.iteration_receipts
    from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_fixed_point import (
        CandidateDoctorFixedPointEvidence,
        validate_deterministic_doctor_fixed_point,
    )

    evidence = CandidateDoctorFixedPointEvidence(
        candidate_tree_id="tree:candidate",
        roots=auth,
        iterations=evidence_iterations,
        expected_tombstone_ids=("tombstone:pkg/caller.py",),
        identity_replay_receipt_id=live.fixed_point.identity_replay_receipt_id
        if live.fixed_point
        else "replay:1",
    )
    pure = validate_deterministic_doctor_fixed_point(
        plan, report, evidence=evidence
    )
    assert pure.complete
    assert pure.fixed_point is not None


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
    outcome = _live_runner().run(plan, report, _happy_request())
    assert not outcome.complete
    assert (
        DoctorFixedPointReason.TRANSACTION_NOT_PROVISIONAL.value
        in outcome.report.reason_codes
    )


def test_deterministic_live_receipt_ids() -> None:
    auth = roots()
    plan = _admitted_plan(auth)
    report = _committed_report(plan)
    request = _happy_request()
    a = _live_runner().run(plan, report, request)
    b = _live_runner().run(plan, report, request)
    assert a.complete and b.complete
    assert a.fixed_point is not None and b.fixed_point is not None
    assert a.fixed_point.content_id == b.fixed_point.content_id
