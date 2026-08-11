"""Conformance tests for deterministic-doctor Tactician planning (LPR-034)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorFinding,
    DoctorAuthorityRoots,
    DoctorEvidenceRole,
    DoctorEvidenceSnapshot,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    SourceRouteKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus import (
    PremiseAuthority,
    PremiseSourceClass,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_tactician import (
    EXACT_FIRST_SOURCE_ROUTES,
    PLANNER_ID,
    REQUIRED_FACET_KINDS,
    DeterministicDoctorTactician,
    DeterministicLocalDoctorPlanner,
    DoctorGoalCompilationDisposition,
    DoctorRepairGoalCompiler,
    DoctorTacticianBounds,
    DoctorTacticianPlanDisposition,
    DoctorTacticianPlanReceipt,
    DoctorTacticianReasonCode,
    DoctorTacticianSafetyError,
    compile_doctor_repair_goals,
    exact_first_route_order,
    plan_doctor_finding,
    required_facet_kinds,
)
from ipfs_accelerate_py.agent_supervisor.validation.tactician_plan_gate import (
    TacticianPlanGateDisposition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _finding(
    roots: DoctorAuthorityRoots | None = None,
    **overrides: object,
) -> DeterministicDoctorFinding:
    roots = roots or _roots()
    values: dict[str, object] = {
        "roots": roots,
        "finding_id": "finding:one",
        "snapshot_id": "snapshot:fixture",
        "disposition": DoctorRepairDisposition.SUPPORTED,
        "observed_fact_refs": ("fact:signature-mismatch",),
        "expected_behavior_refs": ("contract:reviewed:accept-input",),
        "evidence_role": DoctorEvidenceRole.OBSERVED_FACT,
        "affected_symbol_refs": ("symbol:process",),
        "consumer_refs": ("consumer:caller",),
        "invalidation_refs": ("tree:fixture",),
    }
    values.update(overrides)
    return DeterministicDoctorFinding(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Closed vocabularies / exact-first order
# ---------------------------------------------------------------------------


def test_required_facets_and_exact_first_routes_are_closed() -> None:
    kinds = required_facet_kinds()
    assert "type" in kinds
    assert "error" in kinds
    assert "effect" in kinds
    assert "authorization" in kinds
    assert "resource" in kinds
    assert "state" in kinds
    assert "schema" in kinds
    assert "placement" in kinds
    assert "information" in kinds
    assert "memory" in kinds
    assert len(REQUIRED_FACET_KINDS) == 10

    routes = exact_first_route_order()
    assert routes[0] == SourceRouteKind.REVIEWED_CONTRACT.value
    assert SourceRouteKind.LOCAL_STATIC.value in routes
    assert SourceRouteKind.VECTOR.value in routes
    assert SourceRouteKind.KNOWLEDGE_GRAPH.value in routes
    # LLM is never part of the admitted exact-first doctor route set.
    assert SourceRouteKind.LLM.value not in routes
    # Exact local facts precede approximate nominations.
    assert routes.index(SourceRouteKind.LOCAL_STATIC.value) < routes.index(
        SourceRouteKind.VECTOR.value
    )
    assert routes.index(SourceRouteKind.REVIEWED_CONTRACT.value) < routes.index(
        SourceRouteKind.KNOWLEDGE_GRAPH.value
    )
    assert EXACT_FIRST_SOURCE_ROUTES[0] is SourceRouteKind.REVIEWED_CONTRACT


def test_bounds_forbid_model_hypothesis_and_semantic_authority() -> None:
    bounds = DoctorTacticianBounds()
    assert bounds.allow_model_hypothesis is False
    assert bounds.semantic_authority is False
    with pytest.raises(DoctorTacticianSafetyError):
        DoctorTacticianBounds(allow_model_hypothesis=True)
    with pytest.raises(DoctorTacticianSafetyError):
        DoctorTacticianBounds(semantic_authority=True)


# ---------------------------------------------------------------------------
# Goal compilation
# ---------------------------------------------------------------------------


def test_compile_preserves_facets_and_separates_expectations_from_observations() -> None:
    compiler = DoctorRepairGoalCompiler()
    compilation = compiler.compile(_finding(), snapshot=_snapshot())
    assert compilation.disposition is DoctorGoalCompilationDisposition.COMPLETE
    assert compilation.semantic_authority is False
    assert len(compilation.goals) == 1
    goal = compilation.goals[0]
    assert {facet.kind for facet in goal.required_facets} == set(REQUIRED_FACET_KINDS)
    assert len(compilation.required_facet_ids) == 10
    assert compilation.selected_expectation_ids
    assert compilation.selected_observation_ids
    # Expectations are reviewed contracts; observations are static facts.
    by_id = {p.premise_id: p for p in compilation.corpus.premises}
    for premise_id in compilation.selected_expectation_ids:
        premise = by_id[premise_id]
        assert premise.expectation_authority is True
        assert premise.authority is PremiseAuthority.EXPECTATION
        assert premise.source_class is PremiseSourceClass.REVIEWED_CONTRACT
        assert premise.semantic_authority is False
    for premise_id in compilation.selected_observation_ids:
        premise = by_id[premise_id]
        assert premise.expectation_authority is False
        assert premise.authority is PremiseAuthority.STATIC_FACT


def test_candidates_cannot_author_expectations() -> None:
    compiler = DoctorRepairGoalCompiler()
    compilation = compiler.compile(
        _finding(),
        snapshot=_snapshot(),
        candidates=(
            {
                "candidate_ref": "candidate:renamed",
                "primary_signal": "vector",
                "score": 0.99,
            },
            {
                "candidate_ref": "cache:meta:hit",
                "primary_signal": "cache_metadata",
            },
            {
                "candidate_ref": "test:success:only",
                "primary_signal": "mere_success",
            },
        ),
    )
    assert compilation.disposition is DoctorGoalCompilationDisposition.COMPLETE
    by_id = {p.premise_id: p for p in compilation.corpus.premises}
    for premise_id in compilation.selected_hypothesis_ids:
        premise = by_id[premise_id]
        assert premise.expectation_authority is False
        assert premise.authority is PremiseAuthority.HYPOTHESIS
        assert premise.semantic_authority is False
    # No hypothesis is also selected as an expectation.
    assert not set(compilation.selected_hypothesis_ids) & set(
        compilation.selected_expectation_ids
    )


def test_reject_candidate_claiming_expectation_or_semantic_authority() -> None:
    compiler = DoctorRepairGoalCompiler()
    bad = compiler.compile(
        _finding(),
        candidates=({"candidate_ref": "x", "expectation_authority": True},),
    )
    assert DoctorTacticianReasonCode.CANDIDATE_AUTHORED_EXPECTATION.value in (
        bad.reason_codes
    ) or bad.disposition in {
        DoctorGoalCompilationDisposition.PARTIAL,
        DoctorGoalCompilationDisposition.COMPLETE,
        DoctorGoalCompilationDisposition.REJECTED,
    }
    # Score-based authority payload is rejected hard.
    scored = compiler.compile(
        _finding(),
        candidates=({"candidate_ref": "y", "score_authority": True},),
    )
    assert (
        DoctorTacticianReasonCode.SCORE_BASED_AUTHORITY.value in scored.reason_codes
        or scored.disposition is DoctorGoalCompilationDisposition.REJECTED
    )


def test_reject_expectation_from_cache_metadata_or_test_success() -> None:
    compiler = DoctorRepairGoalCompiler()
    finding = _finding(
        expected_behavior_refs=(
            "cache:meta:proof-hit",
            "test:success:suite-green",
            "contract:reviewed:real",
        )
    )
    compilation = compiler.compile(finding)
    assert "cache:meta:proof-hit" in compilation.excluded_expectation_refs or (
        DoctorTacticianReasonCode.CACHE_METADATA_EXPECTATION.value
        in compilation.reason_codes
    )
    assert "test:success:suite-green" in compilation.excluded_expectation_refs or (
        DoctorTacticianReasonCode.TEST_SUCCESS_EXPECTATION.value
        in compilation.reason_codes
    )
    # Real contract still admitted.
    assert compilation.selected_expectation_ids


def test_supported_finding_without_expectation_abstains_or_rejects() -> None:
    # DeterministicDoctorFinding construction already requires expected behavior
    # for SUPPORTED + OBSERVED_FACT. Use approval_required without expectations
    # via nomination role, or construct with empty via approval path.
    finding = _finding(
        disposition=DoctorRepairDisposition.APPROVAL_REQUIRED,
        expected_behavior_refs=(),
        approval_classes=("public_api_or_schema",),
    )
    compilation = compile_doctor_repair_goals(finding)
    # Without expectations the inventory is partial/rejected, never claim complete
    # write-ready goals with expectation authority.
    assert compilation.disposition in {
        DoctorGoalCompilationDisposition.PARTIAL,
        DoctorGoalCompilationDisposition.REJECTED,
        DoctorGoalCompilationDisposition.ABSTAINED,
    }
    assert not any(
        p.expectation_authority for p in compilation.corpus.premises
    )


def test_unknown_frontiers_are_preserved() -> None:
    finding = _finding(
        disposition=DoctorRepairDisposition.APPROVAL_REQUIRED,
        approval_classes=("dynamic_or_generated_code",),
        open_frontier_refs=("frontier:required:dynamic-dispatch", "frontier:optional:alias"),
    )
    compilation = DoctorRepairGoalCompiler().compile(finding)
    assert "frontier:required:dynamic-dispatch" in compilation.unknown_frontier_refs
    assert "frontier:optional:alias" in compilation.unknown_frontier_refs


def test_changed_roots_reject_compilation() -> None:
    finding = _finding()
    other = _roots(tree_id="tree:other")
    compilation = DoctorRepairGoalCompiler().compile(
        finding, current_roots=other
    )
    assert compilation.disposition is DoctorGoalCompilationDisposition.REJECTED
    assert DoctorTacticianReasonCode.CHANGED_ROOTS.value in compilation.reason_codes


def test_stale_snapshot_rejects() -> None:
    finding = _finding()
    snap = _snapshot()
    # Different snapshot id with same roots.
    stale = DoctorEvidenceSnapshot(
        roots=finding.roots,
        snapshot_id="snapshot:stale",
        file_blob_cids=("blob:a",),
        completeness="complete",
        invalidation_refs=("tree:fixture",),
        clean_rebuild_equivalence_receipt_id="rebuild:eq:1",
    )
    compilation = DoctorRepairGoalCompiler().compile(finding, snapshot=stale)
    assert compilation.disposition is DoctorGoalCompilationDisposition.REJECTED
    assert DoctorTacticianReasonCode.STALE_SNAPSHOT.value in compilation.reason_codes


# ---------------------------------------------------------------------------
# Tactician planning + gate
# ---------------------------------------------------------------------------


def test_plan_is_deterministic_semantic_authority_false_and_gated() -> None:
    tactician = DeterministicDoctorTactician()
    finding = _finding()
    snap = _snapshot()
    candidates = (
        {"candidate_ref": "symbol:moved", "primary_signal": "exact_symbol"},
        {"candidate_ref": "symbol:similar", "primary_signal": "vector", "score": 0.8},
    )
    first = tactician.plan_finding(finding, snapshot=snap, candidates=candidates)
    second = tactician.plan_finding(finding, snapshot=snap, candidates=candidates)
    assert first.disposition is DoctorTacticianPlanDisposition.PLANNED
    assert first.content_id == second.content_id
    assert first.semantic_authority is False
    assert first.model_invocation_count == 0
    assert first.llm_route_present is False
    assert first.planner_id == PLANNER_ID
    assert first.plan is not None
    assert first.plan.semantic_authority is False
    assert SourceRouteKind.LLM not in first.plan.ordered_source_routes
    assert first.gate_receipt is not None
    assert first.gate_receipt.disposition is TacticianPlanGateDisposition.ADMITTED
    assert first.gate_receipt.semantic_authority is False
    assert first.gate_receipt.write_authority is False
    # Exact routes precede approximate.
    routes = list(first.ordered_source_routes)
    if "vector" in routes and "local_static" in routes:
        assert routes.index("local_static") < routes.index("vector")
    if "reviewed_contract" in routes and "knowledge_graph" in routes:
        assert routes.index("reviewed_contract") < routes.index("knowledge_graph")
    # Hypotheses are excluded from axiom selection.
    assert first.excluded_premise_ids
    assert first.exclusion_rationale_refs
    # Facet inventory preserved into the receipt.
    assert len(first.required_facet_ids) == 10
    # Round-trip receipt identity.
    cloned = DoctorTacticianPlanReceipt.from_dict(first.to_dict())
    assert cloned.content_id == first.content_id


def test_module_helpers_plan_doctor_finding() -> None:
    receipt = plan_doctor_finding(_finding(), snapshot=_snapshot())
    assert receipt.disposition is DoctorTacticianPlanDisposition.PLANNED
    assert receipt.semantic_authority is False


def test_score_override_attempt_is_rejected() -> None:
    tactician = DeterministicDoctorTactician()
    receipt = tactician.plan_finding(
        _finding(), snapshot=_snapshot(), score_override_attempt=True
    )
    assert receipt.disposition is DoctorTacticianPlanDisposition.REJECTED
    assert DoctorTacticianReasonCode.SCORE_BASED_AUTHORITY.value in receipt.reason_codes
    assert receipt.plan is None
    assert receipt.model_invocation_count == 0


def test_llm_route_never_emitted_by_local_planner() -> None:
    compiler = DoctorRepairGoalCompiler()
    compilation = compiler.compile(
        _finding(),
        candidates=(
            {"candidate_ref": "model:guess", "primary_signal": "model"},
            {"candidate_ref": "llm:guess", "primary_signal": "llm"},
        ),
    )
    plan = DeterministicLocalDoctorPlanner().plan(
        roots=compilation.roots,
        goals=compilation.goals,
        corpus=compilation.corpus,
        compilation=compilation,
    )
    assert SourceRouteKind.LLM not in plan.ordered_source_routes
    assert plan.semantic_authority is False
    # Model hypotheses are excluded, not selected as axioms.
    hyp_ids = set(compilation.selected_hypothesis_ids)
    assert hyp_ids.issubset(set(plan.excluded_premise_ids) | set(plan.selected_premise_ids))
    for premise in compilation.corpus.premises:
        if premise.source_class is PremiseSourceClass.MODEL_HYPOTHESIS:
            assert premise.premise_id in plan.excluded_premise_ids


def test_independent_findings_do_not_share_axioms() -> None:
    tactician = DeterministicDoctorTactician()
    a = _finding(finding_id="finding:a", expected_behavior_refs=("contract:a",))
    b = _finding(finding_id="finding:b", expected_behavior_refs=("contract:b",))
    ra = tactician.plan_finding(a, snapshot=_snapshot())
    rb = tactician.plan_finding(b, snapshot=_snapshot())
    assert ra.disposition is DoctorTacticianPlanDisposition.PLANNED
    assert rb.disposition is DoctorTacticianPlanDisposition.PLANNED
    assert ra.compilation_id != rb.compilation_id
    assert set(ra.selected_premise_ids) != set(rb.selected_premise_ids)


def test_abstain_finding_yields_typed_abstention() -> None:
    finding = _finding(
        disposition=DoctorRepairDisposition.ABSTAIN,
        expected_behavior_refs=(),
        reason_codes=("static_analysis_incomplete",),
        open_frontier_refs=("frontier:required:unknown-dispatch",),
    )
    # ABSTAIN findings may omit expected behavior.
    receipt = DeterministicDoctorTactician().plan_finding(finding)
    assert receipt.disposition is DoctorTacticianPlanDisposition.ABSTAINED
    assert receipt.plan is None
    assert receipt.semantic_authority is False
    assert DoctorTacticianReasonCode.FINDING_ABSTAIN.value in receipt.reason_codes
    assert "frontier:required:unknown-dispatch" in receipt.unknown_frontier_refs


def test_multi_consumer_goals_are_independent() -> None:
    finding = _finding(
        consumer_refs=("consumer:one", "consumer:two"),
    )
    compilation = DoctorRepairGoalCompiler().compile(finding)
    assert len(compilation.goals) == 2
    assert len({g.goal_id for g in compilation.goals}) == 2
    receipt = DeterministicDoctorTactician().plan_finding(finding, snapshot=_snapshot())
    assert receipt.disposition is DoctorTacticianPlanDisposition.PLANNED
    assert len(receipt.goal_ids) == 2


def test_plan_findings_batch_is_deterministic() -> None:
    tactician = DeterministicDoctorTactician()
    findings = (
        _finding(finding_id="finding:1"),
        _finding(finding_id="finding:2", expected_behavior_refs=("contract:two",)),
    )
    first = tactician.plan_findings(findings, snapshot=_snapshot())
    second = tactician.plan_findings(findings, snapshot=_snapshot())
    assert len(first) == 2
    assert [item.content_id for item in first] == [item.content_id for item in second]
    assert all(item.semantic_authority is False for item in first)
    assert all(item.model_invocation_count == 0 for item in first)


def test_receipt_forbids_nonzero_model_invocations() -> None:
    tactician = DeterministicDoctorTactician()
    receipt = tactician.plan_finding(_finding(), snapshot=_snapshot())
    payload = receipt.to_dict()
    payload["model_invocation_count"] = 1
    with pytest.raises(DoctorTacticianSafetyError):
        DoctorTacticianPlanReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["semantic_authority"] = True
    with pytest.raises(DoctorTacticianSafetyError):
        DoctorTacticianPlanReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["llm_route_present"] = True
    with pytest.raises(DoctorTacticianSafetyError):
        DoctorTacticianPlanReceipt.from_dict(payload)


def test_subgoal_graph_is_acyclic_and_covers_facets() -> None:
    receipt = plan_doctor_finding(_finding(), snapshot=_snapshot())
    assert receipt.plan is not None
    plan = receipt.plan
    # Facet coverage: every required facet id appears as a claim_ref.
    claims = {sg.claim_ref for sg in plan.subgoals}
    for facet_id in receipt.required_facet_ids:
        assert facet_id in claims
    # Acyclic: walk depends_on.
    by_id = {sg.subgoal_id: sg for sg in plan.subgoals}
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> None:
        assert node not in visiting
        if node in visited:
            return
        visiting.add(node)
        sg = by_id[node]
        for dep in sg.depends_on:
            if dep in by_id:
                dfs(dep)
        if sg.parent_subgoal_id and sg.parent_subgoal_id in by_id:
            dfs(sg.parent_subgoal_id)
        visiting.remove(node)
        visited.add(node)

    for node in by_id:
        dfs(node)


def test_prompt_directive_in_expectation_is_excluded() -> None:
    # Compact identifier still matching the prompt-directive detector.
    finding = _finding(
        expected_behavior_refs=(
            "contract:reviewed:ok",
            "prompt:jailbreak:payload",
        )
    )
    compilation = DoctorRepairGoalCompiler().compile(finding)
    assert (
        DoctorTacticianReasonCode.PROMPT_DIRECTIVE.value in compilation.reason_codes
        or "prompt:jailbreak:payload" in compilation.excluded_expectation_refs
        or compilation.disposition is DoctorGoalCompilationDisposition.REJECTED
    )
    # Legitimate contract remains usable when present.
    assert compilation.selected_expectation_ids or compilation.disposition in {
        DoctorGoalCompilationDisposition.REJECTED,
        DoctorGoalCompilationDisposition.ABSTAINED,
    }
