"""Doctor-to-Planner shared obligation-kernel tests for PDR-043."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.doctor_causal_localization import (
    CausalLocalizationDisposition,
    DoctorCausalLocalizationReceipt,
    MinimalMismatchSlice,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_contract_adapters import (
    DIAGNOSIS_OBLIGATION_BRIDGE_INTERFACE,
    AuthorityRootBridge,
    DiagnosisObligationBridge,
    FindingBridge,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics import (
    DoctorAuthorityRoots,
    DoctorDiagnosticFinding,
    ExpectationSourceKind,
    FindingDisposition,
    FindingKind,
)
from ipfs_accelerate_py.agent_supervisor.planning.diagnosis_obligation_adapter import (
    DIAGNOSIS_OBLIGATION_ADAPTER_INTERFACE,
    DIAGNOSIS_OBLIGATION_COMPILATION_SCHEMA,
    ContractMismatch,
    DiagnosisCompilationDisposition,
    DiagnosisObligationAdapter,
    DiagnosisObligationAuthorityError,
    DiagnosisObligationCompilation,
    DiagnosisObligationRequest,
    DiagnosisObligationTamperError,
    compile_contract_mismatch_obligations,
    compile_diagnosis_obligations,
    formal_obligation_signature_for_graph,
)
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority,
    ObligationGraphDecision,
    PredicatePolarity,
    RefinementKind,
    SemanticSupport,
)


def _roots(**updates: str) -> DoctorAuthorityRoots:
    values = {
        "repository_id": "repository:diagnosis-adapter",
        "forest_id": "forest:current",
        "tree_id": "tree:current",
        "overlay_id": "overlay:current",
        "file_root_id": "files:current",
        "policy_id": "policy:current",
        "toolchain_id": "toolchain:current",
        "ast_index_id": "ast:current",
        "dependency_graph_id": "graph:current",
        "contract_root_id": "contracts:current",
        "corpus_root_id": "corpus:current",
        "operator_registry_id": "operators:current",
        "translator_id": "translator:current",
        "solver_id": "solver:current",
        "kernel_id": "kernel:current",
        "sandbox_id": "sandbox:current",
        "environment_id": "environment:current",
    }
    values.update(updates)
    return DoctorAuthorityRoots(**values)


def _finding(
    *,
    message: str = "human diagnostic text must remain non-formal",
    disposition: FindingDisposition = FindingDisposition.SUPPORTED,
    kind: FindingKind = FindingKind.CONTRACT,
    open_frontiers: tuple[str, ...] = (),
) -> DoctorDiagnosticFinding:
    return DoctorDiagnosticFinding(
        kind=kind,
        disposition=disposition,
        path="src/service.py",
        symbol="dispatch",
        message=message,
        observation_refs=("fact:dispatch:observed-contract",),
        expectation_source=ExpectationSourceKind.REVIEWED_CONTRACT,
        expectation_ref="contract:dispatch:expected",
        expectation_precedence=100,
        open_frontier_refs=open_frontiers,
        evidence_refs=("evidence:contract-delta",),
        details={
            "human_note": "also not a theorem",
            "observed_argument_count": 1,
        },
    )


def _bridge(
    finding: DoctorDiagnosticFinding | None = None,
    *,
    roots: DoctorAuthorityRoots | None = None,
) -> tuple[DiagnosisObligationBridge, DoctorDiagnosticFinding]:
    diagnostic = finding or _finding()
    root_record = roots or _roots()
    finding_bridge = FindingBridge.bridge(
        diagnostic,
        roots=root_record,
        snapshot_id="snapshot:deterministic",
    )
    return (
        DiagnosisObligationBridge(
            repository_id=root_record.repository_id,
            finding_bridges=(finding_bridge,),
            root_bridge=AuthorityRootBridge.bridge(root_record),
            expected_contract_refs=finding_bridge.expected_refs,
            observed_contract_refs=finding_bridge.observed_refs,
            causal_slice_refs=("slice:bridge-declared",),
            open_frontier_refs=diagnostic.open_frontier_refs,
        ),
        diagnostic,
    )


def _localization(
    finding: DoctorDiagnosticFinding,
    *,
    disposition: CausalLocalizationDisposition = (
        CausalLocalizationDisposition.LOCALIZED
    ),
    repository_id: str = "repository:diagnosis-adapter",
    snapshot_cid: str = "snapshot:diagnostic",
    reasons: tuple[str, ...] = (),
    frontiers: tuple[str, ...] = (),
) -> DoctorCausalLocalizationReceipt:
    localized = disposition is CausalLocalizationDisposition.LOCALIZED
    mismatch = MinimalMismatchSlice(
        issue_cid="issue:semantic-contract-mismatch",
        cause_id="cause:dispatch-contract" if localized else "",
        evidence_ids=("evidence:contract-delta", "evidence:unsat-core"),
        contract_refs=("contract:dispatch:expected",),
        graph_refs=("graph:current",),
        dataflow_refs=("dataflow:dispatch",),
        runtime_refs=("trace:dispatch",),
        unsat_core_refs=("evidence:unsat-core",),
        mandatory_consumer_ids=("consumer:caller",),
        open_frontier_refs=frontiers,
    )
    return DoctorCausalLocalizationReceipt(
        repository_id=repository_id,
        snapshot_cid=snapshot_cid,
        diagnostic_finding_cid=finding.finding_cid,
        issue_cid=mismatch.issue_cid,
        disposition=disposition,
        selected_cause_id=mismatch.cause_id,
        candidate_cause_ids=(
            (mismatch.cause_id,) if mismatch.cause_id else ("cause:a", "cause:b")
        ),
        mismatch_slice=mismatch,
        exact_evidence_ids=(
            "evidence:contract-delta",
            "evidence:unsat-core",
        ),
        mandatory_consumer_ids=("consumer:caller",),
        open_frontier_refs=frontiers,
        reason_codes=reasons,
        complete_frontier_accounting=not frontiers,
    )


def _compiled(
    *,
    finding: DoctorDiagnosticFinding | None = None,
) -> DiagnosisObligationCompilation:
    bridge, diagnostic = _bridge(finding)
    return compile_diagnosis_obligations(
        bridge,
        localizations=(_localization(diagnostic),),
        proof_requirement_refs=("proof:external-contract-equivalence",),
        security_requirement_refs=("security:no-new-dataflow",),
        validation_requirement_refs=("validation:consumer-suite",),
    )


def test_supported_finding_populates_every_typed_obligation_family() -> None:
    result = _compiled()

    assert DIAGNOSIS_OBLIGATION_ADAPTER_INTERFACE == (
        DIAGNOSIS_OBLIGATION_BRIDGE_INTERFACE
    )
    assert DiagnosisObligationAdapter.INTERFACE == (
        "DiagnosisObligationBridge@1"
    )
    assert result.disposition is DiagnosisCompilationDisposition.COMPILED
    assert result.graph.decision is ObligationGraphDecision.READY
    assert result.graph.task_candidates
    assert result.desired_predicate_ids
    assert result.observed_fact_ids
    assert result.assumption_ids
    assert len(result.prohibition_obligation_ids) == 2
    assert result.impact_obligation_ids
    assert result.proof_obligation_ids
    assert result.security_obligation_ids
    assert result.validation_obligation_ids
    assert len(result.alternative_repair_subgoal_ids) == 2
    assert not result.review_obligation_ids
    assert not result.abstention_obligation_ids

    predicate_types = {item.predicate_type for item in result.graph.predicates}
    assert {
        "desired_contract_state",
        "observed_contract_state",
        "repair_prohibition",
        "impact_compatibility",
        "proof_requirement",
        "security_requirement",
        "validation_requirement",
    } <= predicate_types
    prohibitions = [
        item
        for item in result.graph.predicates
        if item.predicate_type == "repair_prohibition"
    ]
    assert prohibitions
    assert all(
        item.polarity is PredicatePolarity.NEGATIVE
        for item in prohibitions
    )
    assert all(
        fact.authority is FactAuthority.BOUNDED_OBSERVATION
        for fact in result.graph.facts
    )

    desired = next(
        item
        for item in result.graph.predicates
        if item.predicate_type == "desired_contract_state"
    )
    refinements = result.graph.refinements_for(
        next(
            node.obligation_id
            for node in result.graph.nodes
            if node.predicate_id == desired.predicate_id
            and node.producer_id == ""
        )
    )
    assert len(refinements) == 1
    assert refinements[0].kind is RefinementKind.OR
    assert len(refinements[0].child_obligation_ids) == 2

    payload = result.to_dict()
    assert payload["authority"] == {
        "proof_authority": False,
        "security_attestation_authority": False,
        "effect_authority": False,
        "mutation_authority": False,
        "completion_authority": False,
        "candidate_generation_only": True,
    }
    assert result.graph.to_dict()["authority"]["proof_authority"] is False


def test_root_schema_evidence_issue_and_causal_ids_round_trip() -> None:
    result = _compiled()
    restored = DiagnosisObligationCompilation.from_dict(result.to_record())

    assert restored.content_id == result.content_id
    assert restored.graph_id == result.graph_id
    assert restored.authority_root_ids == result.authority_root_ids
    assert restored.authority_root_ids["tree_id"] == "tree:current"
    assert restored.authority_root_ids["deterministic_root_cid"]
    assert restored.authority_root_ids["diagnostic_root_cid"]
    assert restored.schema_ids == result.schema_ids
    assert DIAGNOSIS_OBLIGATION_COMPILATION_SCHEMA in restored.schema_ids
    assert restored.evidence_ids == result.evidence_ids
    assert "evidence:contract-delta" in restored.evidence_ids
    assert restored.issue_ids == result.issue_ids
    assert restored.semantic_issue_ids == (
        "issue:semantic-contract-mismatch",
    )
    assert restored.causal_slice_ids == result.causal_slice_ids
    assert "slice:bridge-declared" in restored.causal_slice_ids
    assert restored.snapshot_ids == result.snapshot_ids

    tampered = result.to_record()
    tampered["evidence_ids"] = [*tampered["evidence_ids"], "evidence:forged"]
    with pytest.raises(DiagnosisObligationTamperError):
        DiagnosisObligationCompilation.from_dict(tampered)


def test_incomplete_diagnosis_yields_review_and_abstention_obligations() -> None:
    bridge, _ = _bridge()

    # No causal localization is an incomplete diagnosis, even when the source
    # finding optimistically says "supported".
    result = DiagnosisObligationAdapter().compile(bridge)

    assert result.disposition is (
        DiagnosisCompilationDisposition.REVIEW_REQUIRED
    )
    assert result.graph.decision is ObligationGraphDecision.REVIEW_REQUIRED
    assert result.review_obligation_ids
    assert result.abstention_obligation_ids
    assert len(result.alternative_repair_subgoal_ids) == 2
    assert not result.graph.task_candidates
    assert "diagnosis_incomplete" in result.reason_codes
    assert all(
        item.support is SemanticSupport.UNKNOWN
        for item in result.graph.predicates
        if item.predicate_id in result.desired_predicate_ids
    )


def test_contradictory_causal_diagnosis_abstains_without_repair_candidate() -> None:
    bridge, diagnostic = _bridge()
    contradictory = _localization(
        diagnostic,
        disposition=CausalLocalizationDisposition.ABSTAINED,
        reasons=("contradictory_decisive_evidence", "correct_abstention"),
    )

    result = compile_diagnosis_obligations(
        DiagnosisObligationRequest(
            bridge=bridge,
            localizations=(contradictory,),
        )
    )

    assert result.disposition is DiagnosisCompilationDisposition.ABSTAINED
    assert result.review_required
    assert result.abstained
    assert result.review_obligation_ids
    assert result.abstention_obligation_ids
    assert "contradictory_diagnosis" in result.reason_codes
    assert result.graph.task_candidates == ()


def test_free_form_finding_text_is_neither_theorem_nor_effect() -> None:
    first = _compiled(finding=_finding(message="first arbitrary explanation"))
    second = _compiled(finding=_finding(message="entirely different prose"))

    first_graph = first.graph.to_json()
    assert "first arbitrary explanation" not in first_graph
    assert "human_note" not in first_graph
    assert "also not a theorem" not in first_graph
    for predicate in first.graph.predicates:
        assert "first arbitrary explanation" not in predicate.subject_ref
        assert "first arbitrary explanation" not in predicate.object_ref
        assert "first arbitrary explanation" not in predicate.proof_requirement_refs
        assert (
            "first arbitrary explanation"
            not in predicate.validation_requirement_refs
        )
    for producer in first.graph.producers:
        assert "first arbitrary explanation" not in producer.provenance_refs

    # The bridge CID changes because the audit finding changed, but formal
    # obligations remain equal because free-form text is provenance, not logic.
    assert first.bridge_ids != second.bridge_ids
    assert first.formal_obligation_signature == (
        second.formal_obligation_signature
    )
    assert first.graph.root_obligation_ids == second.graph.root_obligation_ids


def test_planner_and_doctor_share_exact_formal_obligation_identity() -> None:
    doctor = _compiled()
    planner_mismatch = ContractMismatch(
        repository_id="repository:diagnosis-adapter",
        current_root_id="tree:current",
        finding_kind="contract",
        expected_refs=("contract:dispatch:expected",),
        observed_refs=("fact:dispatch:observed-contract",),
        subject_refs=("dispatch",),
        consumer_refs=("consumer:caller",),
        evidence_refs=(
            "evidence:contract-delta",
            "evidence:unsat-core",
            "fact:dispatch:observed-contract",
        ),
        causal_slice_refs=("planner:slice",),
        source_refs=("planner:intent",),
        issue_ids=("planner:issue",),
        diagnosis_complete=True,
    )
    planner = compile_contract_mismatch_obligations(
        planner_mismatch,
        proof_requirement_refs=("proof:external-contract-equivalence",),
        security_requirement_refs=("security:no-new-dataflow",),
        validation_requirement_refs=("validation:consumer-suite",),
    )

    assert planner.root_obligation_ids == doctor.graph.root_obligation_ids
    assert {item.producer_id for item in planner.producers} == {
        item.producer_id for item in doctor.graph.producers
    }
    assert formal_obligation_signature_for_graph(planner) == (
        doctor.formal_obligation_signature
    )


def test_open_frontier_and_approval_requirement_fail_closed_to_review() -> None:
    frontier_finding = _finding(
        open_frontiers=("frontier:dynamic_dispatch",)
    )
    bridge, diagnostic = _bridge(frontier_finding)
    result = compile_diagnosis_obligations(
        bridge,
        localizations=(
            _localization(
                diagnostic,
                frontiers=("frontier:dynamic_dispatch",),
            ),
        ),
    )
    assert result.review_required
    assert result.frontier_ids == (
        "frontier:dynamic_dispatch",
    )
    assert result.graph.task_candidates == ()

    approval_bridge, _ = _bridge(
        _finding(disposition=FindingDisposition.APPROVAL_REQUIRED)
    )
    approval = compile_diagnosis_obligations(approval_bridge)
    assert approval.review_required
    assert "approval_required" in approval.reason_codes
    assert approval.graph.task_candidates == ()


def test_localization_repository_and_snapshot_replay_are_rejected() -> None:
    bridge, diagnostic = _bridge()
    foreign = _localization(
        diagnostic, repository_id="repository:foreign"
    )
    with pytest.raises(DiagnosisObligationAuthorityError):
        compile_diagnosis_obligations(
            bridge, localizations=(foreign,)
        )

    # A snapshot bridge makes the diagnostic snapshot identity mandatory.
    # The checked top-level bridge used here has no snapshot projection, so
    # repository binding is the applicable replay check.
    unknown_finding = replace(
        _localization(diagnostic),
        diagnostic_finding_cid="finding:foreign",
    )
    with pytest.raises(DiagnosisObligationAuthorityError):
        compile_diagnosis_obligations(
            bridge, localizations=(unknown_finding,)
        )


def test_unsupported_kind_and_keyed_localization_inputs_remain_fail_closed() -> None:
    unsupported_finding = _finding(kind=FindingKind.UNSUPPORTED)
    unsupported_bridge, diagnostic = _bridge(unsupported_finding)
    localization = _localization(diagnostic)

    result = compile_diagnosis_obligations(
        unsupported_bridge,
        localizations={diagnostic.finding_cid: localization},
    )
    assert result.review_required
    assert result.graph.decision is ObligationGraphDecision.REVIEW_REQUIRED
    assert "unsupported_finding_kind" in result.reason_codes
    assert result.graph.task_candidates == ()

    # A single receipt is also accepted without weakening identity checks.
    supported_bridge, supported = _bridge()
    single = compile_diagnosis_obligations(
        supported_bridge,
        localizations=_localization(supported),
    )
    assert single.disposition is DiagnosisCompilationDisposition.COMPILED
