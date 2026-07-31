"""Fail-closed coverage for deterministic doctor impact closure (LPR-037)."""

from __future__ import annotations

import hashlib

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    ImpactCompleteness,
    ProgramContractDelta,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorEditSite,
    DoctorPlanDisposition,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_impact import (
    DETERMINISTIC_DOCTOR_IMPACT_INTERFACE,
    PRODUCER_ID,
    DeterministicDoctorImpactAnalyzer,
    DoctorConsumerDisposition,
    DoctorGraphEdgeObservation,
    DoctorImpactClosureReceipt,
    DoctorImpactConsumerObservation,
    DoctorImpactError,
    DoctorImpactFrontierObservation,
    DoctorImpactPlanDisposition,
    DoctorImpactReason,
    DoctorImpactRequest,
    DoctorPlanCompilationRequest,
    all_consumer_dispositions,
    compile_deterministic_doctor_plan,
    create_deterministic_doctor_impact_analyzer,
    doctor_roots_to_propagation_roots,
    map_to_plan_repair_disposition,
    map_to_propagation_disposition,
    mutation_requires_complete_closure,
    path_is_forbidden,
    rebuild_candidate_program_contract_delta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def roots(**overrides: str) -> DoctorAuthorityRoots:
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


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_delta(auth: DoctorAuthorityRoots) -> ProgramContractDelta:
    prop = doctor_roots_to_propagation_roots(auth)
    return ProgramContractDelta(
        roots=prop,
        change_set_id="changeset:one",
        subject_symbol_id="symbol:target_fn",
        before_contract_ref="contract:before:target_fn",
        after_contract_ref="contract:after:target_fn",
        clauses=(
            ContractClauseDelta(
                clause_id="clause:target_fn:signature",
                kind=DeltaKind.PARAMETER_ADD,
                disposition=DeltaDisposition.BREAKING,
                subject_symbol_id="symbol:target_fn",
                consumer_domain="domain:python-callers",
                before_contract_ref="contract:before:target_fn",
                after_contract_ref="contract:after:target_fn",
                reason="add required argument",
            ),
        ),
        evidence_refs=("evidence:ast",),
        proof_refs=("proof:delta",),
    )


def _consumer(
    consumer_id: str,
    path: str,
    *,
    disposition: DoctorConsumerDisposition | None = DoctorConsumerDisposition.MIGRATED,
    depth: int = 1,
    mandatory: bool = True,
    second_order: bool = False,
    stale: bool = False,
    owner_id: str = "",
    symbol_id: str = "",
) -> DoctorImpactConsumerObservation:
    return DoctorImpactConsumerObservation(
        consumer_id=consumer_id,
        path=path,
        symbol_id=symbol_id or consumer_id.replace("consumer:", "symbol:"),
        depth=depth,
        mandatory=mandatory,
        disposition=disposition,
        second_order=second_order,
        stale=stale,
        owner_id=owner_id,
        edge_refs=(f"edge:{consumer_id}",),
        proof_refs=(f"proof:{consumer_id}",),
    )


def _edit_site(path: str, body: str = "x") -> DoctorEditSite:
    return DoctorEditSite(
        path=path,
        before_hash=_sha(body),
        span_start=0,
        span_end=len(body),
        artifact_id=f"blob:{path}",
    )


def _happy_request(
    auth: DoctorAuthorityRoots | None = None,
    *,
    with_second_order: bool = False,
) -> DoctorImpactRequest:
    auth = auth or roots()
    consumers = (
        _consumer("consumer:direct", "pkg/caller_a.py"),
        _consumer("consumer:wrapper", "pkg/caller_b.py", depth=2),
        _consumer(
            "consumer:test",
            "tests/test_caller.py",
            disposition=DoctorConsumerDisposition.PROVED_COMPATIBLE,
            depth=2,
        ),
    )
    second = ()
    if with_second_order:
        second = (
            _consumer(
                "consumer:second",
                "pkg/second_order.py",
                disposition=DoctorConsumerDisposition.MIGRATED,
                second_order=True,
                depth=3,
            ),
        )
    return DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        overlay_id="overlay:repair-1",
        overlay_path="pkg/target.py",
        overlay_patch_cid="cid:patch:1",
        overlay_before_hash=_sha("before"),
        overlay_after_hash=_sha("after"),
        subject_symbol_id="symbol:target_fn",
        consumers=consumers,
        second_order_consumers=second,
        edges=(
            DoctorGraphEdgeObservation(
                "consumer:wrapper", "consumer:direct", kind="calls"
            ),
            DoctorGraphEdgeObservation(
                "consumer:test", "consumer:direct", kind="tests"
            ),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
        proof_refs=("proof:overlay",),
        evidence_refs=("evidence:snapshot",),
    )


def _admit_plan_request(
    closure: DoctorImpactClosureReceipt,
    *,
    auth: DoctorAuthorityRoots | None = None,
) -> DoctorPlanCompilationRequest:
    auth = auth or closure.roots
    migrate_paths = sorted(
        {
            item.path
            for item in closure.consumers
            if item.disposition is DoctorConsumerDisposition.MIGRATED
        }
    )
    sites = tuple(_edit_site(path) for path in migrate_paths)
    return DoctorPlanCompilationRequest(
        roots=auth,
        closure=closure,
        snapshot_id="snapshot:fixture",
        finding_ids=("finding:arity",),
        selected_operator_id="operator:add_argument",
        target_ref="target:symbol:target_fn",
        value_source_ref="value:ctx",
        placement_ref="placement:arg2",
        proof_refs=("proof:hammer", "proof:overlay"),
        edit_sites=sites,
        permitted_read_paths=tuple(migrate_paths),
        permitted_write_paths=tuple(migrate_paths),
        lease_id=auth.lease_id,
        checkpoint_ref="checkpoint:content-addressed",
        rollback_ref="rollback:restore-checkpoint",
        operator_ids=("operator:add_argument",),
        validation_refs=("validation:pytest",),
        invalidation_refs=("invalidate:impact-closure",),
    )


# ---------------------------------------------------------------------------
# Vocabulary / helpers
# ---------------------------------------------------------------------------


def test_interface_and_factory() -> None:
    assert DETERMINISTIC_DOCTOR_IMPACT_INTERFACE == "DeterministicDoctorImpactAnalyzer@1"
    assert PRODUCER_ID == "deterministic-doctor-impact@1"
    analyzer = create_deterministic_doctor_impact_analyzer()
    assert isinstance(analyzer, DeterministicDoctorImpactAnalyzer)
    assert analyzer.INTERFACE == DETERMINISTIC_DOCTOR_IMPACT_INTERFACE


def test_closed_disposition_vocabulary() -> None:
    kinds = {item.value for item in all_consumer_dispositions()}
    assert kinds == {
        "migrated",
        "proved_compatible",
        "unaffected",
        "approval",
        "unsupported",
    }
    assert DoctorConsumerDisposition.MIGRATED.requires_write is True
    assert DoctorConsumerDisposition.PROVED_COMPATIBLE.requires_write is False
    assert DoctorConsumerDisposition.APPROVAL.blocks_autonomous_mutation is True
    assert DoctorConsumerDisposition.UNSUPPORTED.blocks_autonomous_mutation is True


def test_disposition_mapping() -> None:
    assert (
        map_to_plan_repair_disposition(DoctorConsumerDisposition.MIGRATED)
        is DoctorRepairDisposition.SUPPORTED
    )
    assert (
        map_to_plan_repair_disposition(DoctorConsumerDisposition.APPROVAL)
        is DoctorRepairDisposition.APPROVAL_REQUIRED
    )
    assert (
        map_to_plan_repair_disposition(DoctorConsumerDisposition.UNSUPPORTED)
        is DoctorRepairDisposition.ABSTAIN
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
        ConsumerDisposition,
    )

    assert (
        map_to_propagation_disposition(DoctorConsumerDisposition.MIGRATED)
        is ConsumerDisposition.MIGRATE
    )
    assert (
        map_to_propagation_disposition(DoctorConsumerDisposition.PROVED_COMPATIBLE)
        is ConsumerDisposition.COMPATIBLE
    )


def test_path_is_forbidden_tcb_and_vendor() -> None:
    assert path_is_forbidden(
        "ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_contracts.py"
    )
    assert path_is_forbidden("vendor/lib/foo.py")
    assert path_is_forbidden("pkg/ok.py", extra_forbidden=("pkg/ok.py",))
    assert not path_is_forbidden("pkg/caller.py")


# ---------------------------------------------------------------------------
# Candidate delta rebuild
# ---------------------------------------------------------------------------


def test_rebuild_candidate_delta_from_base() -> None:
    auth = roots()
    base = _base_delta(auth)
    rebuilt = rebuild_candidate_program_contract_delta(
        roots=auth,
        base_delta=base,
        overlay_id="overlay:1",
        overlay_patch_cid="cid:patch",
    )
    assert isinstance(rebuilt, ProgramContractDelta)
    assert rebuilt.subject_symbol_id == base.subject_symbol_id
    assert "overlay:overlay:1" in rebuilt.evidence_refs
    assert "patch:cid:patch" in rebuilt.evidence_refs
    # Roots rebinding: candidate tree is current doctor tree.
    assert rebuilt.roots.candidate_tree_id == auth.tree_id
    # Identity is stable on replay.
    again = rebuild_candidate_program_contract_delta(
        roots=auth,
        base_delta=base,
        overlay_id="overlay:1",
        overlay_patch_cid="cid:patch",
    )
    assert again.content_id == rebuilt.content_id


def test_analyzer_rebuild_matches_helper() -> None:
    auth = roots()
    req = _happy_request(auth)
    analyzer = DeterministicDoctorImpactAnalyzer()
    delta = analyzer.rebuild_candidate_delta(req)
    helper = rebuild_candidate_program_contract_delta(
        roots=auth,
        base_delta=req.base_delta,
        overlay_id=req.overlay_id,
        overlay_patch_cid=req.overlay_patch_cid,
        subject_symbol_id=req.subject_symbol_id,
        proof_refs=req.proof_refs,
        evidence_refs=req.evidence_refs,
    )
    assert delta.content_id == helper.content_id


# ---------------------------------------------------------------------------
# Impact closure — happy path
# ---------------------------------------------------------------------------


def test_complete_closure_one_disposition_per_consumer() -> None:
    analyzer = DeterministicDoctorImpactAnalyzer()
    receipt = analyzer.analyze(_happy_request())
    assert isinstance(receipt, DoctorImpactClosureReceipt)
    assert receipt.completeness is ImpactCompleteness.COMPLETE
    assert receipt.mutation_admissible is True
    assert receipt.no_model_invariant is True
    assert receipt.model_invocation_count == 0
    assert not receipt.open_required_frontiers
    ids = [item.consumer_id for item in receipt.consumers]
    assert len(ids) == len(set(ids))
    assert set(ids) == {
        "consumer:direct",
        "consumer:wrapper",
        "consumer:test",
    }
    by_id = {item.consumer_id: item for item in receipt.consumers}
    assert by_id["consumer:direct"].disposition is DoctorConsumerDisposition.MIGRATED
    assert (
        by_id["consumer:test"].disposition
        is DoctorConsumerDisposition.PROVED_COMPATIBLE
    )
    # Deterministic replay.
    again = analyzer.analyze(_happy_request())
    assert again.content_id == receipt.content_id
    assert again.impact_closure_id == receipt.impact_closure_id


def test_second_order_consumers_discovered_from_overlay() -> None:
    analyzer = DeterministicDoctorImpactAnalyzer()
    receipt = analyzer.analyze(_happy_request(with_second_order=True))
    assert "consumer:second" in receipt.second_order_consumer_ids
    by_id = {item.consumer_id: item for item in receipt.consumers}
    assert by_id["consumer:second"].second_order is True
    assert by_id["consumer:second"].disposition is DoctorConsumerDisposition.MIGRATED
    assert DoctorImpactReason.SECOND_ORDER_DISCOVERED.value in receipt.reason_codes


def test_scc_grouping_is_deterministic() -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        overlay_id="overlay:scc",
        consumers=(
            _consumer("consumer:a", "pkg/a.py"),
            _consumer("consumer:b", "pkg/b.py"),
            _consumer("consumer:c", "pkg/c.py"),
        ),
        edges=(
            DoctorGraphEdgeObservation("consumer:a", "consumer:b"),
            DoctorGraphEdgeObservation("consumer:b", "consumer:a"),
            DoctorGraphEdgeObservation("consumer:c", "consumer:a"),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert receipt.completeness is ImpactCompleteness.COMPLETE
    # a and b form an SCC; c is separate.
    members = {scc.scc_id: set(scc.member_consumer_ids) for scc in receipt.sccs}
    assert any(members[sid] == {"consumer:a", "consumer:b"} for sid in members)
    again = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert [scc.scc_id for scc in again.sccs] == [scc.scc_id for scc in receipt.sccs]


# ---------------------------------------------------------------------------
# Open frontiers block mutation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("reflection", DoctorImpactReason.REFLECTION_FRONTIER),
        ("unknown_dispatch", DoctorImpactReason.UNKNOWN_DISPATCH_FRONTIER),
        ("generated_code", DoctorImpactReason.GENERATED_CODE_FRONTIER),
        ("native_ffi", DoctorImpactReason.NATIVE_FFI_FRONTIER),
        ("unsupported_interprocedural", DoctorImpactReason.INTERPROCEDURAL_FRONTIER),
    ],
)
def test_required_open_frontiers_block_complete_mutation(
    kind: str, reason: DoctorImpactReason
) -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        overlay_id="overlay:frontier",
        consumers=(_consumer("consumer:direct", "pkg/caller.py"),),
        frontiers=(
            DoctorImpactFrontierObservation(
                kind=kind,
                route=f"route:{kind}",
                required=True,
                closed=False,
            ),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert receipt.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert receipt.mutation_admissible is False
    assert receipt.open_required_frontiers
    assert kind.replace("unknown_dispatch", "unknown_dispatch") in {
        k for k in receipt.frontier_kinds
    } or any(kind.split("_")[0] in k for k in receipt.frontier_kinds)
    assert (
        reason.value in receipt.reason_codes
        or DoctorImpactReason.OPEN_REQUIRED_FRONTIER.value in receipt.reason_codes
    )
    # Plan must abstain before any write.
    plan = compile_deterministic_doctor_plan(_admit_plan_request(receipt))
    assert plan.disposition is DoctorImpactPlanDisposition.ABSTAINED
    assert plan.may_mutate is False
    assert plan.plan is not None
    assert plan.plan.disposition is DoctorPlanDisposition.ABSTAINED
    assert not plan.plan.permitted_write_paths


# ---------------------------------------------------------------------------
# Missed / duplicate / stale / circular ownership / forbidden path
# ---------------------------------------------------------------------------


def test_missed_expected_consumer_abstains() -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        consumers=(_consumer("consumer:direct", "pkg/caller.py"),),
        expected_consumer_ids=("consumer:direct", "consumer:missing"),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert receipt.completeness is ImpactCompleteness.ABSTAINED
    assert "consumer:missing" in receipt.missed_consumer_ids
    assert receipt.mutation_admissible is False
    plan = compile_deterministic_doctor_plan(_admit_plan_request(receipt))
    assert plan.may_mutate is False
    assert DoctorImpactReason.MISSED_CONSUMER.value in plan.reason_codes


def test_duplicate_consumer_ids_abstain() -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        consumers=(
            _consumer("consumer:dup", "pkg/a.py"),
            _consumer("consumer:dup", "pkg/b.py"),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert receipt.completeness is ImpactCompleteness.ABSTAINED
    assert "consumer:dup" in receipt.duplicate_consumer_ids
    assert receipt.mutation_admissible is False


def test_stale_consumer_blocks_mutation() -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        consumers=(
            _consumer("consumer:stale", "pkg/old.py", stale=True),
            _consumer("consumer:ok", "pkg/ok.py"),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert receipt.completeness is ImpactCompleteness.ABSTAINED
    assert "consumer:stale" in receipt.stale_consumer_ids
    assert receipt.mutation_admissible is False


def test_circular_ownership_abstains() -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        consumers=(
            _consumer("consumer:a", "pkg/a.py", owner_id="consumer:b"),
            _consumer("consumer:b", "pkg/b.py", owner_id="consumer:a"),
        ),
        edges=(
            DoctorGraphEdgeObservation(
                "consumer:a", "consumer:b", kind="owns", ownership=True
            ),
            DoctorGraphEdgeObservation(
                "consumer:b", "consumer:a", kind="owns", ownership=True
            ),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert receipt.completeness is ImpactCompleteness.ABSTAINED
    assert receipt.circular_ownership_refs
    assert DoctorImpactReason.CIRCULAR_OWNERSHIP.value in receipt.reason_codes
    plan = compile_deterministic_doctor_plan(_admit_plan_request(receipt))
    assert plan.may_mutate is False


def test_forbidden_write_path_blocks_mutation() -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        consumers=(
            _consumer(
                "consumer:tcb",
                "ipfs_accelerate_py/agent_supervisor/proof/kernel_verification.py",
            ),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    # Forbidden path hits prevent complete mutation-admissible closure.
    assert receipt.mutation_admissible is False
    assert receipt.forbidden_path_hits or receipt.completeness is not ImpactCompleteness.COMPLETE


# ---------------------------------------------------------------------------
# Atomic plan compilation
# ---------------------------------------------------------------------------


def test_compile_atomic_plan_covers_all_scc_steps() -> None:
    analyzer = DeterministicDoctorImpactAnalyzer()
    closure = analyzer.analyze(_happy_request(with_second_order=True))
    assert closure.mutation_admissible is True
    plan_receipt = compile_deterministic_doctor_plan(_admit_plan_request(closure))
    assert plan_receipt.disposition is DoctorImpactPlanDisposition.ADMITTED
    assert plan_receipt.may_mutate is True
    assert plan_receipt.plan is not None
    plan = plan_receipt.plan
    assert plan.disposition is DoctorPlanDisposition.ADMITTED
    assert plan.impact_closure_id == closure.impact_closure_id
    assert plan.selected_operator_id == "operator:add_argument"
    assert plan.permitted_write_paths
    assert plan.lease_id
    assert plan.checkpoint_ref
    assert plan.rollback_ref
    assert plan.proof_refs
    assert plan.edit_sites
    assert plan.no_model_invariant is True
    assert plan.model_invocation_count == 0
    # Exactly one plan-level disposition per resolved consumer.
    plan_ids = [item.consumer_id for item in plan.consumer_dispositions]
    assert len(plan_ids) == len(set(plan_ids))
    assert set(plan_ids) == {item.consumer_id for item in closure.consumers}
    # Every migrated consumer covered by a write step.
    migrate_ids = {
        item.consumer_id
        for item in closure.consumers
        if item.disposition is DoctorConsumerDisposition.MIGRATED
    }
    covered: set[str] = set()
    for step in plan.steps:
        if step.write_paths:
            covered.update(step.consumer_ids)
    assert migrate_ids <= covered
    # SCC steps present and acyclic dependencies.
    step_ids = {step.step_id for step in plan.steps}
    for step in plan.steps:
        assert set(step.dependency_step_ids) <= step_ids
        assert step.step_id not in step.dependency_step_ids
    assert mutation_requires_complete_closure(closure, plan_receipt) is True


def test_plan_gaps_abstain_before_write() -> None:
    analyzer = DeterministicDoctorImpactAnalyzer()
    closure = analyzer.analyze(_happy_request())
    # Missing operator / proofs / edit sites → plan gap.
    bad = DoctorPlanCompilationRequest(
        roots=closure.roots,
        closure=closure,
        snapshot_id="snapshot:fixture",
        finding_ids=("finding:arity",),
        selected_operator_id="",
        target_ref="",
        value_source_ref="",
        placement_ref="",
        proof_refs=(),
        edit_sites=(),
        permitted_write_paths=(),
        lease_id="",
        checkpoint_ref="",
        rollback_ref="",
        invalidation_refs=("invalidate:impact-closure",),
    )
    plan = compile_deterministic_doctor_plan(bad)
    assert plan.disposition is DoctorImpactPlanDisposition.ABSTAINED
    assert plan.may_mutate is False
    assert DoctorImpactReason.PLAN_GAP.value in plan.reason_codes or any(
        code
        in {
            DoctorImpactReason.MISSING_OPERATOR.value,
            DoctorImpactReason.MISSING_LEASE.value,
            DoctorImpactReason.MISSING_PROOF.value,
        }
        for code in plan.reason_codes
    )
    if plan.plan is not None:
        assert not plan.plan.permitted_write_paths


def test_approval_disposition_requires_approval_not_autonomous_write() -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        consumers=(
            _consumer(
                "consumer:public",
                "pkg/public_api.py",
                disposition=DoctorConsumerDisposition.APPROVAL,
            ),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    closure = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert closure.mutation_admissible is False
    plan = compile_deterministic_doctor_plan(_admit_plan_request(closure))
    assert plan.disposition in {
        DoctorImpactPlanDisposition.APPROVAL_REQUIRED,
        DoctorImpactPlanDisposition.ABSTAINED,
    }
    assert plan.may_mutate is False


def test_close_and_plan_cohesive_path() -> None:
    analyzer = DeterministicDoctorImpactAnalyzer()
    auth = roots()
    req = _happy_request(auth)
    closure = analyzer.analyze(req)
    plan_req = _admit_plan_request(closure, auth=auth)
    c2, plan = analyzer.close_and_plan(req, plan_req)
    assert c2.impact_closure_id == closure.impact_closure_id
    assert plan.disposition is DoctorImpactPlanDisposition.ADMITTED
    assert plan.may_mutate is True


def test_mutation_requires_complete_closure_helper() -> None:
    analyzer = DeterministicDoctorImpactAnalyzer()
    closure = analyzer.analyze(_happy_request())
    assert mutation_requires_complete_closure(closure, None) is False
    plan = compile_deterministic_doctor_plan(_admit_plan_request(closure))
    assert mutation_requires_complete_closure(closure, plan) is True


def test_receipt_to_dict_round_trip_stable() -> None:
    receipt = DeterministicDoctorImpactAnalyzer().analyze(_happy_request())
    payload = receipt.to_dict()
    restored = DoctorImpactClosureReceipt.from_dict(payload)
    assert restored.content_id == receipt.content_id
    assert restored.mutation_admissible == receipt.mutation_admissible


def test_module_exports_ast_symbols() -> None:
    import ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_impact as mod

    for name in (
        "DeterministicDoctorImpactAnalyzer",
        "DoctorConsumerDisposition",
        "DoctorImpactClosureReceipt",
        "compile_deterministic_doctor_plan",
    ):
        assert hasattr(mod, name), name


def test_no_model_invariant_on_impact_and_plan() -> None:
    analyzer = DeterministicDoctorImpactAnalyzer()
    closure = analyzer.analyze(_happy_request())
    assert closure.model_invocation_count == 0
    plan = compile_deterministic_doctor_plan(_admit_plan_request(closure))
    assert plan.model_invocation_count == 0
    assert plan.no_model_invariant is True
    assert plan.plan is not None
    assert plan.plan.llm_router_enabled is False


def test_unaffected_and_unsupported_dispositions() -> None:
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        consumers=(
            _consumer(
                "consumer:skip",
                "pkg/skip.py",
                disposition=DoctorConsumerDisposition.UNAFFECTED,
                mandatory=False,
            ),
            _consumer(
                "consumer:bad",
                "pkg/bad.py",
                disposition=DoctorConsumerDisposition.UNSUPPORTED,
            ),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    by_id = {item.consumer_id: item for item in receipt.consumers}
    assert by_id["consumer:skip"].disposition is DoctorConsumerDisposition.UNAFFECTED
    assert by_id["consumer:bad"].disposition is DoctorConsumerDisposition.UNSUPPORTED
    assert receipt.mutation_admissible is False


def test_malformed_request_fails_closed() -> None:
    with pytest.raises(DoctorImpactError):
        DoctorImpactRequest(roots="not-roots")  # type: ignore[arg-type]
    with pytest.raises(DoctorImpactError):
        DeterministicDoctorImpactAnalyzer().analyze({"roots": "bad"})


def test_forged_mutation_admissible_is_coerced() -> None:
    """mutation_admissible is derived; open frontiers cannot claim true."""
    auth = roots()
    req = DoctorImpactRequest(
        roots=auth,
        base_delta=_base_delta(auth),
        consumers=(_consumer("consumer:direct", "pkg/caller.py"),),
        frontiers=(
            DoctorImpactFrontierObservation(
                kind="reflection", route="route:x", required=True
            ),
        ),
        current_graph_cid=auth.graph_id,
        current_index_cid=auth.index_id,
        current_ast_cid=auth.ast_root_id,
    )
    receipt = DeterministicDoctorImpactAnalyzer().analyze(req)
    assert receipt.mutation_admissible is False
    # Construction with forged flag is coerced by DoctorImpactClosureReceipt.
    forged = DoctorImpactClosureReceipt(
        roots=receipt.roots,
        impact_closure_id=receipt.impact_closure_id,
        delta_id=receipt.delta_id,
        candidate_delta_id=receipt.candidate_delta_id,
        completeness=receipt.completeness,
        consumers=receipt.consumers,
        sccs=receipt.sccs,
        open_required_frontiers=receipt.open_required_frontiers,
        frontier_kinds=receipt.frontier_kinds,
        reason_codes=receipt.reason_codes,
        mutation_admissible=True,  # forged
    )
    assert forged.mutation_admissible is False
