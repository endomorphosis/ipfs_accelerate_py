"""Conformance tests for deterministic-doctor contracts and policy (LPR-029)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    ALLOWED_DOCTOR_MODES,
    ALL_APPROVAL_CLASSES,
    ALL_DOCTOR_OPERATIONS,
    ALL_REPAIR_DISPOSITIONS,
    AUTHORITY_ROOT_FIELDS,
    DEFAULT_DOCTOR_MODE,
    DETERMINISTIC_DOCTOR_POLICY_SCHEMA,
    DOCTOR_TCB_PATH_MARKERS,
    FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS,
    READ_ONLY_OPERATIONS,
    DeterministicDoctorAuthorityError,
    DeterministicDoctorBoundsError,
    DeterministicDoctorError,
    DeterministicDoctorFinding,
    DeterministicDoctorPlan,
    DeterministicDoctorRunReceipt,
    DeterministicDoctorSafetyError,
    DoctorApprovalClass,
    DoctorAuthorityRoots,
    DoctorCacheAuditDisposition,
    DoctorConsumerDisposition,
    DoctorEditSite,
    DoctorEvidenceRole,
    DoctorEvidenceSnapshot,
    DoctorMode,
    DoctorOperation,
    DoctorOperatorKind,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorProofCacheAuditReceipt,
    DoctorRejectionReason,
    DoctorRepairDisposition,
    DoctorRepairOperatorSpec,
    DoctorResourceBounds,
    ForgedDeterministicDoctorIdentityError,
    consumer_disposition_set_identity,
    default_doctor_mode,
    is_doctor_tcb_path,
    operation_is_read_only,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import (
    DEFAULT_LIMITS,
    DeterministicDoctorPolicy,
    DoctorPolicyDecision,
    PolicyVerdict,
    assert_run_receipt_policy,
    classify_change_approval_classes,
    classify_path_approval_classes,
    default_deterministic_doctor_policy,
    evaluate_doctor_operation,
    load_deterministic_doctor_policy,
)


def _roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "file_root_id": "file-root:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": "graph:fixture",
        "corpus_id": "corpus:fixture",
        "index_id": "index:fixture",
        "model_id": "model:fixture",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
        "lease_id": "lease:fixture",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _snapshot(roots: DoctorAuthorityRoots | None = None) -> DoctorEvidenceSnapshot:
    roots = roots or _roots()
    return DoctorEvidenceSnapshot(
        roots=roots,
        snapshot_id="snapshot:fixture",
        file_blob_cids=("blob:a", "blob:b"),
        completeness="complete",
        invalidation_refs=("tree:fixture",),
        clean_rebuild_equivalence_receipt_id="rebuild:eq:1",
    )


def _consumer(
    roots: DoctorAuthorityRoots,
    consumer_id: str = "consumer:one",
    disposition: DoctorRepairDisposition = DoctorRepairDisposition.SUPPORTED,
) -> DoctorConsumerDisposition:
    return DoctorConsumerDisposition(
        roots=roots,
        consumer_id=consumer_id,
        disposition=disposition,
        reason_codes=("ok",),
    )


def _admitted_plan(roots: DoctorAuthorityRoots | None = None) -> DeterministicDoctorPlan:
    roots = roots or _roots()
    site = DoctorEditSite(
        path="pkg/module.py",
        before_hash="sha256:before",
        span_start=0,
        span_end=10,
        artifact_id="blob:module",
    )
    step = DoctorPlanStep(
        step_id="step:1",
        kind="analytical",
        operator_id="operator:rename",
        consumer_ids=("consumer:one",),
        edit_site_refs=(site.content_id,),
        write_paths=("pkg/module.py",),
    )
    return DeterministicDoctorPlan(
        roots=roots,
        plan_id="plan:fixture",
        snapshot_id="snapshot:fixture",
        finding_ids=("finding:one",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=(_consumer(roots),),
        impact_closure_id="impact:fixture",
        steps=(step,),
        edit_sites=(site,),
        operator_ids=("operator:rename",),
        target_ref="symbol:target",
        value_source_ref="value:source",
        placement_ref="placement:site",
        selected_operator_id="operator:rename",
        permitted_read_paths=("pkg/module.py",),
        permitted_write_paths=("pkg/module.py",),
        lease_id="lease:fixture",
        checkpoint_ref="checkpoint:fixture",
        rollback_ref="rollback:fixture",
        proof_refs=("proof:fixture",),
        invalidation_refs=("tree:fixture",),
    )


# ---------------------------------------------------------------------------
# Operations, modes, dispositions, roots
# ---------------------------------------------------------------------------


def test_operations_modes_and_read_only_defaults() -> None:
    assert set(ALL_DOCTOR_OPERATIONS) == {
        DoctorOperation.INSPECT,
        DoctorOperation.EXPLAIN,
        DoctorOperation.PLAN,
        DoctorOperation.REPAIR,
        DoctorOperation.REPLAY,
        DoctorOperation.ROLLBACK,
    }
    for op in (
        DoctorOperation.INSPECT,
        DoctorOperation.EXPLAIN,
        DoctorOperation.PLAN,
        DoctorOperation.REPLAY,
    ):
        assert op.is_read_only
        assert operation_is_read_only(op)
        assert op in READ_ONLY_OPERATIONS
    assert DoctorOperation.REPAIR.may_write
    assert not DoctorOperation.REPAIR.is_read_only
    assert DEFAULT_DOCTOR_MODE is DoctorMode.REPORT_ONLY
    assert default_doctor_mode() is DoctorMode.REPORT_ONLY
    assert ALLOWED_DOCTOR_MODES == (
        DoctorMode.REPORT_ONLY,
        DoctorMode.PLAN,
        DoctorMode.SANDBOX_AUTO,
        DoctorMode.NARROW_AUTO,
    )
    assert not DoctorMode.REPORT_ONLY.allows_source_write
    assert DoctorMode.NARROW_AUTO.allows_source_write
    assert DoctorMode.SANDBOX_AUTO.allows_sandbox_write


def test_repair_dispositions_and_approval_classes_are_closed() -> None:
    assert set(ALL_REPAIR_DISPOSITIONS) == {
        DoctorRepairDisposition.SUPPORTED,
        DoctorRepairDisposition.ABSTAIN,
        DoctorRepairDisposition.APPROVAL_REQUIRED,
        DoctorRepairDisposition.ROLLED_BACK,
        DoctorRepairDisposition.QUARANTINED,
    }
    assert DoctorRepairDisposition.SUPPORTED.grants_write_authority
    assert not DoctorRepairDisposition.ABSTAIN.grants_write_authority
    assert set(ALL_APPROVAL_CLASSES) == {
        DoctorApprovalClass.DOCTOR_TRUSTED_COMPUTING_BASE,
        DoctorApprovalClass.STATEFUL_BEHAVIOR,
        DoctorApprovalClass.PUBLIC_API_OR_SCHEMA,
        DoctorApprovalClass.DYNAMIC_OR_GENERATED_CODE,
        DoctorApprovalClass.NATIVE_OR_FFI,
        DoctorApprovalClass.CROSS_REPOSITORY_EDIT,
        DoctorApprovalClass.NEW_EXTERNAL_DEPENDENCY,
        DoctorApprovalClass.UNSUPPORTED_MEMORY_OR_LIFETIME_CLAIM,
    }


def test_authority_roots_bind_full_root_surface() -> None:
    roots = _roots()
    assert set(AUTHORITY_ROOT_FIELDS) == {
        "repository_id",
        "forest_id",
        "tree_id",
        "overlay_id",
        "file_root_id",
        "ast_root_id",
        "graph_id",
        "corpus_id",
        "index_id",
        "model_id",
        "cache_id",
        "operator_registry_id",
        "translator_id",
        "solver_id",
        "kernel_id",
        "toolchain_id",
        "policy_id",
        "sandbox_id",
        "environment_id",
        "lease_id",
    }
    assert roots.content_id.startswith("b")
    assert DoctorAuthorityRoots.from_dict(roots.to_record()) == roots


def test_forged_cid_rejected_on_roots_and_snapshot() -> None:
    roots = _roots()
    payload = roots.to_record()
    payload["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ForgedDeterministicDoctorIdentityError):
        DoctorAuthorityRoots.from_dict(payload)

    snapshot = _snapshot(roots)
    snap_payload = snapshot.to_record()
    snap_payload["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ForgedDeterministicDoctorIdentityError):
        DoctorEvidenceSnapshot.from_dict(snap_payload)


def test_bodies_and_secrets_rejected() -> None:
    roots = _roots()
    with pytest.raises(DeterministicDoctorError, match="source bodies or secrets"):
        DeterministicDoctorFinding.from_dict(
            {
                "schema": DeterministicDoctorFinding.SCHEMA,
                "roots": roots.to_dict(),
                "finding_id": "finding:x",
                "snapshot_id": "snapshot:x",
                "disposition": "abstain",
                "observed_fact_refs": [],
                "expected_behavior_refs": [],
                "source_body": "def evil(): pass",
                "invalidation_refs": ["tree:x"],
            }
        )
    with pytest.raises(DeterministicDoctorError, match="source bodies or secrets"):
        DeterministicDoctorFinding.from_dict(
            {
                "schema": DeterministicDoctorFinding.SCHEMA,
                "roots": roots.to_dict(),
                "finding_id": "finding:x",
                "snapshot_id": "snapshot:x",
                "disposition": "abstain",
                "observed_fact_refs": [],
                "expected_behavior_refs": [],
                # Use an admitted placeholder value so proposal validation does
                # not treat this fixture as concrete secret material. Rejection
                # is keyed on the field name, not the value.
                "api_key": "redacted",
                "invalidation_refs": ["tree:x"],
            }
        )


def test_observed_facts_separated_from_expected_behavior() -> None:
    roots = _roots()
    with pytest.raises(DeterministicDoctorAuthorityError, match="separate"):
        DeterministicDoctorFinding(
            roots=roots,
            finding_id="finding:1",
            snapshot_id="snapshot:1",
            disposition=DoctorRepairDisposition.ABSTAIN,
            observed_fact_refs=("fact:shared",),
            expected_behavior_refs=("fact:shared",),
            invalidation_refs=("tree:1",),
        )

    finding = DeterministicDoctorFinding(
        roots=roots,
        finding_id="finding:1",
        snapshot_id="snapshot:1",
        disposition=DoctorRepairDisposition.SUPPORTED,
        observed_fact_refs=("fact:ast",),
        expected_behavior_refs=("behavior:reviewed-contract",),
        evidence_role=DoctorEvidenceRole.OBSERVED_FACT,
        invalidation_refs=("tree:1",),
    )
    assert finding.observed_fact_refs == ("fact:ast",)
    assert finding.expected_behavior_refs == ("behavior:reviewed-contract",)
    assert finding.semantic_authority is False


def test_nomination_cannot_grant_supported_or_write_authority() -> None:
    roots = _roots()
    with pytest.raises(DeterministicDoctorAuthorityError, match="nomination"):
        DeterministicDoctorFinding(
            roots=roots,
            finding_id="finding:1",
            snapshot_id="snapshot:1",
            disposition=DoctorRepairDisposition.SUPPORTED,
            observed_fact_refs=("fact:1",),
            expected_behavior_refs=("behavior:1",),
            evidence_role=DoctorEvidenceRole.NOMINATION,
            invalidation_refs=("tree:1",),
        )

    operator = DoctorRepairOperatorSpec(
        roots=roots,
        operator_id="operator:rename",
        kind=DoctorOperatorKind.EXACT_RENAME,
        supported_languages=("python",),
        precondition_refs=("pre:1",),
        postcondition_refs=("post:1",),
        write_paths=("pkg/a.py",),
    )
    assert operator.grants_write_authority is False
    assert operator.semantic_authority is False
    with pytest.raises(DeterministicDoctorAuthorityError, match="cannot grant write"):
        DoctorRepairOperatorSpec(
            roots=roots,
            operator_id="operator:bad",
            kind=DoctorOperatorKind.EXACT_RENAME,
            supported_languages=("python",),
            precondition_refs=("pre:1",),
            postcondition_refs=("post:1",),
            grants_write_authority=True,
        )


def test_plan_rejects_cycles_partial_open_frontier_and_missing_prereqs() -> None:
    roots = _roots()
    consumer = _consumer(roots)

    # Cycle in step dependencies.
    step_a = DoctorPlanStep(
        step_id="step:a",
        kind="analytical",
        dependency_step_ids=("step:b",),
    )
    step_b = DoctorPlanStep(
        step_id="step:b",
        kind="analytical",
        dependency_step_ids=("step:a",),
    )
    with pytest.raises(DeterministicDoctorError, match="acyclic"):
        DeterministicDoctorPlan(
            roots=roots,
            plan_id="plan:cycle",
            snapshot_id="snapshot:1",
            finding_ids=("finding:1",),
            disposition=DoctorPlanDisposition.ABSTAINED,
            consumer_dispositions=(consumer,),
            impact_closure_id="impact:1",
            steps=(step_a, step_b),
            invalidation_refs=("tree:1",),
        )

    # Open required frontier blocks admission.
    with pytest.raises(DeterministicDoctorAuthorityError, match="required frontiers"):
        DeterministicDoctorPlan(
            roots=roots,
            plan_id="plan:frontier",
            snapshot_id="snapshot:1",
            finding_ids=("finding:1",),
            disposition=DoctorPlanDisposition.ADMITTED,
            consumer_dispositions=(consumer,),
            impact_closure_id="impact:1",
            steps=(
                DoctorPlanStep(
                    step_id="step:1",
                    kind="analytical",
                    operator_id="operator:x",
                    write_paths=("pkg/a.py",),
                ),
            ),
            edit_sites=(
                DoctorEditSite(path="pkg/a.py", before_hash="h1"),
            ),
            target_ref="t",
            value_source_ref="v",
            placement_ref="p",
            selected_operator_id="operator:x",
            permitted_write_paths=("pkg/a.py",),
            lease_id="lease:1",
            checkpoint_ref="checkpoint:1",
            rollback_ref="rollback:1",
            proof_refs=("proof:1",),
            open_required_frontiers=("frontier:required:reflection",),
            invalidation_refs=("tree:1",),
        )

    # Admitted plan without lease/checkpoint/rollback.
    with pytest.raises(DeterministicDoctorAuthorityError, match="lease"):
        DeterministicDoctorPlan(
            roots=_roots(lease_id=""),
            plan_id="plan:nolease",
            snapshot_id="snapshot:1",
            finding_ids=("finding:1",),
            disposition=DoctorPlanDisposition.ADMITTED,
            consumer_dispositions=(_consumer(_roots(lease_id="")),),
            impact_closure_id="impact:1",
            steps=(
                DoctorPlanStep(
                    step_id="step:1",
                    kind="analytical",
                    operator_id="operator:x",
                    write_paths=("pkg/a.py",),
                ),
            ),
            edit_sites=(DoctorEditSite(path="pkg/a.py", before_hash="h1"),),
            target_ref="t",
            value_source_ref="v",
            placement_ref="p",
            selected_operator_id="operator:x",
            permitted_write_paths=("pkg/a.py",),
            checkpoint_ref="checkpoint:1",
            rollback_ref="rollback:1",
            proof_refs=("proof:1",),
            invalidation_refs=("tree:1",),
        )


def test_admitted_plan_round_trip_and_nonadmitted_cannot_write() -> None:
    plan = _admitted_plan()
    assert plan.is_admitted
    assert plan.model_invocation_count == 0
    assert plan.no_model_invariant is True
    assert DeterministicDoctorPlan.from_dict(plan.to_record()) == plan

    roots = _roots()
    with pytest.raises(DeterministicDoctorAuthorityError, match="write path"):
        DeterministicDoctorPlan(
            roots=roots,
            plan_id="plan:abstain",
            snapshot_id="snapshot:1",
            finding_ids=("finding:1",),
            disposition=DoctorPlanDisposition.ABSTAINED,
            consumer_dispositions=(_consumer(roots, disposition=DoctorRepairDisposition.ABSTAIN),),
            impact_closure_id="impact:1",
            permitted_write_paths=("pkg/a.py",),
            invalidation_refs=("tree:1",),
        )


def test_plan_rejects_llm_and_semantic_authority() -> None:
    roots = _roots()
    consumer = _consumer(roots, disposition=DoctorRepairDisposition.ABSTAIN)
    with pytest.raises(DeterministicDoctorSafetyError, match="no_model"):
        DeterministicDoctorPlan(
            roots=roots,
            plan_id="plan:model",
            snapshot_id="snapshot:1",
            finding_ids=("finding:1",),
            disposition=DoctorPlanDisposition.ABSTAINED,
            consumer_dispositions=(consumer,),
            impact_closure_id="impact:1",
            no_model_invariant=False,
            invalidation_refs=("tree:1",),
        )
    with pytest.raises(DeterministicDoctorSafetyError, match="llm_router"):
        DeterministicDoctorPlan(
            roots=roots,
            plan_id="plan:llm",
            snapshot_id="snapshot:1",
            finding_ids=("finding:1",),
            disposition=DoctorPlanDisposition.ABSTAINED,
            consumer_dispositions=(consumer,),
            impact_closure_id="impact:1",
            llm_router_enabled=True,
            invalidation_refs=("tree:1",),
        )
    with pytest.raises(DeterministicDoctorSafetyError, match="semantic authority"):
        DeterministicDoctorPlan(
            roots=roots,
            plan_id="plan:sem",
            snapshot_id="snapshot:1",
            finding_ids=("finding:1",),
            disposition=DoctorPlanDisposition.ABSTAINED,
            consumer_dispositions=(consumer,),
            impact_closure_id="impact:1",
            semantic_authority_flags={"vector_semantic_authority": True},
            invalidation_refs=("tree:1",),
        )


def test_tcb_paths_protected_on_edit_sites_and_plans() -> None:
    assert any("proof/" in marker for marker in DOCTOR_TCB_PATH_MARKERS)
    assert is_doctor_tcb_path(
        "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py"
    )
    assert is_doctor_tcb_path(
        "ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_contracts.py"
    )
    assert not is_doctor_tcb_path("pkg/module.py")

    with pytest.raises(DeterministicDoctorAuthorityError, match="trusted computing base"):
        DoctorEditSite(
            path="ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py",
            before_hash="h1",
        )


def test_cache_audit_metadata_is_not_semantic_authority() -> None:
    roots = _roots()
    audit = DoctorProofCacheAuditReceipt(
        roots=roots,
        audit_id="audit:1",
        cache_namespace="ns:proof",
        cache_key="key:1",
        disposition=DoctorCacheAuditDisposition.HIT,
        reconstruction_ref="recon:1",
        premise_refs=("premise:1",),
        authoritative=True,
        invalidation_refs=("tree:1",),
    )
    assert audit.semantic_authority is False
    with pytest.raises(DeterministicDoctorSafetyError, match="semantic_authority"):
        DoctorProofCacheAuditReceipt(
            roots=roots,
            audit_id="audit:2",
            cache_namespace="ns:proof",
            cache_key="key:2",
            disposition=DoctorCacheAuditDisposition.HIT,
            semantic_authority=True,
            invalidation_refs=("tree:1",),
        )
    with pytest.raises(
        DeterministicDoctorAuthorityError,
        match="hit/reconstructed|quarantined",
    ):
        DoctorProofCacheAuditReceipt(
            roots=roots,
            audit_id="audit:3",
            cache_namespace="ns:proof",
            cache_key="key:3",
            disposition=DoctorCacheAuditDisposition.QUARANTINED,
            authoritative=True,
            reconstruction_ref="recon:1",
            premise_refs=("premise:1",),
            invalidation_refs=("tree:1",),
        )


def test_run_receipt_requires_zero_model_invocations_and_repair_prereqs() -> None:
    roots = _roots()
    receipt = DeterministicDoctorRunReceipt(
        roots=roots,
        receipt_id="receipt:inspect",
        operation=DoctorOperation.INSPECT,
        mode=DoctorMode.REPORT_ONLY,
        disposition=DoctorRepairDisposition.ABSTAIN,
        snapshot_id="snapshot:1",
        invalidation_refs=("tree:1",),
    )
    assert receipt.model_invocation_count == 0
    assert DeterministicDoctorRunReceipt.from_dict(receipt.to_record()) == receipt

    with pytest.raises(DeterministicDoctorSafetyError, match="zero"):
        DeterministicDoctorRunReceipt(
            roots=roots,
            receipt_id="receipt:llm",
            operation=DoctorOperation.INSPECT,
            mode=DoctorMode.REPORT_ONLY,
            disposition=DoctorRepairDisposition.ABSTAIN,
            snapshot_id="snapshot:1",
            model_invocation_count=1,
            invalidation_refs=("tree:1",),
        )

    with pytest.raises(DeterministicDoctorSafetyError, match="LLM|model-provider"):
        DeterministicDoctorRunReceipt(
            roots=roots,
            receipt_id="receipt:remote",
            operation=DoctorOperation.INSPECT,
            mode=DoctorMode.REPORT_ONLY,
            disposition=DoctorRepairDisposition.ABSTAIN,
            snapshot_id="snapshot:1",
            remote_model_provider_invoked=True,
            invalidation_refs=("tree:1",),
        )

    with pytest.raises(DeterministicDoctorAuthorityError, match="plan"):
        DeterministicDoctorRunReceipt(
            roots=roots,
            receipt_id="receipt:repair",
            operation=DoctorOperation.REPAIR,
            mode=DoctorMode.SANDBOX_AUTO,
            disposition=DoctorRepairDisposition.SUPPORTED,
            snapshot_id="snapshot:1",
            lease_id="lease:1",
            checkpoint_ref="checkpoint:1",
            rollback_ref="rollback:1",
            invalidation_refs=("tree:1",),
        )

    repair = DeterministicDoctorRunReceipt(
        roots=roots,
        receipt_id="receipt:repair-ok",
        operation=DoctorOperation.REPAIR,
        mode=DoctorMode.SANDBOX_AUTO,
        disposition=DoctorRepairDisposition.SUPPORTED,
        snapshot_id="snapshot:1",
        plan_id="plan:1",
        lease_id="lease:1",
        checkpoint_ref="checkpoint:1",
        rollback_ref="rollback:1",
        candidate_tree_cid="tree:candidate",
        invalidation_refs=("tree:1",),
    )
    assert repair.committed_tree_cid == ""


def test_rollback_and_quarantined_states() -> None:
    roots = _roots()
    rolled = DeterministicDoctorRunReceipt(
        roots=roots,
        receipt_id="receipt:rb",
        operation=DoctorOperation.ROLLBACK,
        mode=DoctorMode.SANDBOX_AUTO,
        disposition=DoctorRepairDisposition.ROLLED_BACK,
        snapshot_id="snapshot:1",
        checkpoint_ref="checkpoint:1",
        rollback_ref="rollback:1",
        invalidation_refs=("tree:1",),
    )
    assert rolled.disposition is DoctorRepairDisposition.ROLLED_BACK

    quarantined = DeterministicDoctorFinding(
        roots=roots,
        finding_id="finding:q",
        snapshot_id="snapshot:1",
        disposition=DoctorRepairDisposition.QUARANTINED,
        observed_fact_refs=("fact:poisoned",),
        expected_behavior_refs=(),
        reason_codes=("poisoned_cache",),
        invalidation_refs=("tree:1",),
    )
    assert quarantined.disposition is DoctorRepairDisposition.QUARANTINED


def test_resource_bounds_reject_unbounded_values() -> None:
    with pytest.raises(DeterministicDoctorError):
        DoctorResourceBounds(max_findings=0)
    bounds = DoctorResourceBounds()
    assert bounds.max_plan_steps == 256
    assert DoctorResourceBounds.from_dict(bounds.to_record()) == bounds


def test_consumer_disposition_set_identity_is_stable() -> None:
    roots = _roots()
    a = _consumer(roots, "consumer:a")
    b = _consumer(roots, "consumer:b")
    assert consumer_disposition_set_identity((a, b)) == consumer_disposition_set_identity(
        (b, a)
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_default_policy_is_report_only_no_model() -> None:
    policy = default_deterministic_doctor_policy()
    assert policy.SCHEMA == DETERMINISTIC_DOCTOR_POLICY_SCHEMA
    assert policy.default_mode is DoctorMode.REPORT_ONLY
    assert policy.enabled is False
    assert policy.llm_invocations_allowed is False
    assert policy.llm_router_enabled is False
    assert policy.remote_model_provider_calls_allowed is False
    assert policy.knowledge_graph_semantic_authority is False
    assert policy.vector_semantic_authority is False
    assert policy.embedding_semantic_authority is False
    assert policy.tactician_semantic_authority is False
    assert policy.hammer_candidate_semantic_authority is False
    assert policy.proof_cache_metadata_semantic_authority is False
    assert policy.unknown_or_unsupported_disposition == "abstain"
    assert policy.ambiguous_disposition == "abstain"
    assert set(policy.approval_required_classes) == {
        item.value for item in ALL_APPROVAL_CLASSES
    }
    assert set(policy.limits) == set(DEFAULT_LIMITS)
    assert DeterministicDoctorPolicy.from_dict(policy.to_record()) == policy


def test_policy_rejects_safety_flag_elevation() -> None:
    with pytest.raises(DeterministicDoctorSafetyError, match="llm"):
        DeterministicDoctorPolicy(llm_invocations_allowed=True)
    with pytest.raises(DeterministicDoctorSafetyError, match="semantic"):
        DeterministicDoctorPolicy(vector_semantic_authority=True)
    with pytest.raises(DeterministicDoctorSafetyError, match="gate must remain"):
        DeterministicDoctorPolicy(compensating_rollback_required=False)


def test_policy_allows_inspect_explain_plan_as_read_only_default() -> None:
    policy = DeterministicDoctorPolicy.default()
    for op in (DoctorOperation.INSPECT, DoctorOperation.EXPLAIN, DoctorOperation.PLAN):
        decision = policy.evaluate(operation=op)
        assert decision.verdict is PolicyVerdict.ALLOW
        assert decision.read_only is True
        assert decision.mode is DoctorMode.REPORT_ONLY


def test_policy_rejects_repair_without_admitted_plan_lease_checkpoint_rollback() -> None:
    policy = DeterministicDoctorPolicy(enabled=True)
    decision = policy.evaluate(
        operation=DoctorOperation.REPAIR,
        mode=DoctorMode.SANDBOX_AUTO,
    )
    assert decision.verdict is PolicyVerdict.REJECT
    assert DoctorRejectionReason.REPAIR_WITHOUT_ADMITTED_PLAN.value in decision.reason_codes[0] or any(
        "plan" in code for code in decision.reason_codes
    )

    plan = _admitted_plan()
    ok = policy.evaluate(
        operation=DoctorOperation.REPAIR,
        mode=DoctorMode.SANDBOX_AUTO,
        plan=plan,
        write_paths=plan.permitted_write_paths,
    )
    assert ok.verdict is PolicyVerdict.ALLOW

    # Plan still carries lease/checkpoint/rollback so allow even if kwargs empty.
    decision_lease = policy.evaluate(
        operation=DoctorOperation.REPAIR,
        mode=DoctorMode.SANDBOX_AUTO,
        plan=plan,
        lease_id="",
        checkpoint_ref="",
        rollback_ref="",
    )
    assert decision_lease.verdict is PolicyVerdict.ALLOW

    # Explicitly reject open frontier repair.
    frontier = policy.evaluate(
        operation=DoctorOperation.REPAIR,
        mode=DoctorMode.SANDBOX_AUTO,
        plan=plan,
        open_required_frontiers=("frontier:required:native",),
    )
    assert frontier.verdict is PolicyVerdict.ABSTAIN


def test_policy_rejects_forged_bodies_cycles_unbounded_partial() -> None:
    policy = DeterministicDoctorPolicy.default()
    for kwargs, reason in (
        ({"forged_cid": True}, DoctorRejectionReason.FORGED_CID.value),
        ({"has_body_or_secret": True}, DoctorRejectionReason.BODY_OR_SECRET.value),
        ({"has_cycle": True}, DoctorRejectionReason.CYCLE.value),
        ({"unbounded": True}, DoctorRejectionReason.UNBOUNDED_DATA.value),
        ({"partial_plan": True}, DoctorRejectionReason.PARTIAL_PLAN.value),
    ):
        decision = policy.evaluate(operation=DoctorOperation.PLAN, **kwargs)
        assert decision.verdict is PolicyVerdict.REJECT
        assert reason in decision.reason_codes


def test_policy_rejects_llm_and_semantic_authority_claims() -> None:
    policy = DeterministicDoctorPolicy.default()
    llm = policy.evaluate(
        operation=DoctorOperation.INSPECT,
        llm_router_invoked=True,
    )
    assert llm.verdict is PolicyVerdict.REJECT

    remote = policy.evaluate(
        operation=DoctorOperation.INSPECT,
        remote_model_provider_invoked=True,
    )
    assert remote.verdict is PolicyVerdict.REJECT

    semantic = policy.evaluate(
        operation=DoctorOperation.PLAN,
        semantic_authority_flags={"embedding_semantic_authority": True},
    )
    assert semantic.verdict is PolicyVerdict.REJECT


def test_policy_protects_tcb_and_approval_classes() -> None:
    policy = DeterministicDoctorPolicy(enabled=True)
    plan = _admitted_plan()
    tcb = policy.evaluate(
        operation=DoctorOperation.REPAIR,
        mode=DoctorMode.SANDBOX_AUTO,
        plan=plan,
        write_paths=(
            "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py",
        ),
    )
    assert tcb.verdict is PolicyVerdict.REJECT
    assert DoctorRejectionReason.TCB_PATH.value in tcb.reason_codes

    public = classify_path_approval_classes("pkg/public/api/schema.py")
    assert DoctorApprovalClass.PUBLIC_API_OR_SCHEMA in public

    classes = classify_change_approval_classes(
        paths=("pkg/native/ffi_bridge.py",),
        new_external_dependency=True,
        cross_repository=True,
    )
    assert DoctorApprovalClass.NATIVE_OR_FFI in classes
    assert DoctorApprovalClass.NEW_EXTERNAL_DEPENDENCY in classes
    assert DoctorApprovalClass.CROSS_REPOSITORY_EDIT in classes

    approval = policy.evaluate(
        operation=DoctorOperation.REPAIR,
        mode=DoctorMode.SANDBOX_AUTO,
        plan=plan,
        write_paths=("pkg/public/api/handlers.py",),
    )
    assert approval.verdict is PolicyVerdict.APPROVAL_REQUIRED


def test_policy_mode_forbids_repair_in_report_only() -> None:
    policy = DeterministicDoctorPolicy(enabled=True)
    plan = _admitted_plan()
    decision = policy.evaluate(
        operation=DoctorOperation.REPAIR,
        mode=DoctorMode.REPORT_ONLY,
        plan=plan,
    )
    assert decision.verdict is PolicyVerdict.REJECT
    assert DoctorRejectionReason.MODE_FORBIDS_OPERATION.value in decision.reason_codes


def test_policy_scheduler_dict_and_load_round_trip() -> None:
    policy = DeterministicDoctorPolicy.default()
    scheduler = policy.to_scheduler_dict()
    assert scheduler["schema"] == DETERMINISTIC_DOCTOR_POLICY_SCHEMA
    assert scheduler["default_mode"] == "report_only"
    assert scheduler["allowed_modes"] == [
        "report_only",
        "plan",
        "sandbox_auto",
        "narrow_auto",
    ]
    loaded = load_deterministic_doctor_policy(scheduler)
    assert loaded.default_mode is DoctorMode.REPORT_ONLY
    assert loaded.limits["max_findings"] == 256


def test_evaluate_doctor_operation_helper_and_receipt_assert() -> None:
    decision = evaluate_doctor_operation(DoctorOperation.EXPLAIN)
    assert isinstance(decision, DoctorPolicyDecision)
    assert decision.allowed

    roots = _roots()
    receipt = DeterministicDoctorRunReceipt(
        roots=roots,
        receipt_id="receipt:1",
        operation=DoctorOperation.INSPECT,
        mode=DoctorMode.REPORT_ONLY,
        disposition=DoctorRepairDisposition.ABSTAIN,
        snapshot_id="snapshot:1",
        invalidation_refs=("tree:1",),
    )
    assert assert_run_receipt_policy(receipt).receipt_id == "receipt:1"


def test_approval_required_finding_requires_class() -> None:
    roots = _roots()
    with pytest.raises(DeterministicDoctorError, match="approval class"):
        DeterministicDoctorFinding(
            roots=roots,
            finding_id="finding:apr",
            snapshot_id="snapshot:1",
            disposition=DoctorRepairDisposition.APPROVAL_REQUIRED,
            observed_fact_refs=("fact:1",),
            expected_behavior_refs=("behavior:1",),
            invalidation_refs=("tree:1",),
        )
    finding = DeterministicDoctorFinding(
        roots=roots,
        finding_id="finding:apr",
        snapshot_id="snapshot:1",
        disposition=DoctorRepairDisposition.APPROVAL_REQUIRED,
        observed_fact_refs=("fact:1",),
        expected_behavior_refs=("behavior:1",),
        approval_classes=(DoctorApprovalClass.STATEFUL_BEHAVIOR.value,),
        invalidation_refs=("tree:1",),
    )
    assert finding.disposition is DoctorRepairDisposition.APPROVAL_REQUIRED


def test_forbidden_semantic_authority_flag_names_match_scheduler() -> None:
    assert set(FORBIDDEN_SEMANTIC_AUTHORITY_FLAGS) == {
        "knowledge_graph_semantic_authority",
        "vector_semantic_authority",
        "embedding_semantic_authority",
        "tactician_semantic_authority",
        "hammer_candidate_semantic_authority",
        "proof_cache_metadata_semantic_authority",
    }


def test_snapshot_has_open_required_frontier_helper() -> None:
    roots = _roots()
    snap = DoctorEvidenceSnapshot(
        roots=roots,
        snapshot_id="snapshot:open",
        file_blob_cids=(),
        completeness="partial_with_frontier",
        unsupported_frontiers=("frontier:required:reflection",),
        invalidation_refs=("tree:1",),
    )
    assert snap.has_open_required_frontier


def test_policy_forged_identity_on_load() -> None:
    policy = DeterministicDoctorPolicy.default()
    payload = policy.to_record()
    payload["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ForgedDeterministicDoctorIdentityError):
        DeterministicDoctorPolicy.from_dict(payload)


def test_unbounded_plan_steps_rejected() -> None:
    roots = _roots()
    consumer = _consumer(roots, disposition=DoctorRepairDisposition.ABSTAIN)
    steps = tuple(
        DoctorPlanStep(step_id=f"step:{i}", kind="analytical")
        for i in range(300)
    )
    with pytest.raises(
        DeterministicDoctorBoundsError,
        match="max_plan_steps|exceeds its item bound",
    ):
        DeterministicDoctorPlan(
            roots=roots,
            plan_id="plan:big",
            snapshot_id="snapshot:1",
            finding_ids=("finding:1",),
            disposition=DoctorPlanDisposition.ABSTAINED,
            consumer_dispositions=(consumer,),
            impact_closure_id="impact:1",
            steps=steps,
            invalidation_refs=("tree:1",),
        )
