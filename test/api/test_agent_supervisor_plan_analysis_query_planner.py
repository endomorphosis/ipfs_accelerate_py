"""Contract tests for deterministic reasoning-query planning (PDR-021)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_operation_registry import (
    AnalysisOperation,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_analysis_query_planner import (
    ClaimClass,
    CoverageDecision,
    EvidenceAuthority,
    PlanAnalysisQueryPlanner,
    QueryEvidenceSlot,
    QueryFailureDisposition,
    QueryInputKind,
    QueryPlanningBudgetError,
    QueryRequirement,
    UnsafeModelSuggestionError,
    compile_reasoning_query_plan,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    DirtyTreePolicy,
    FallbackPolicy,
    PlanAuthorityRoots,
    PlanCreateRequest,
    PlanDeltaOperation,
    PlanPopulationDigest,
    PlanRequestBudget,
    PlanSteerRequest,
    PopulationKind,
    TaskSourceKind,
    plan_revision_cid,
)

MANDATORY_SLOTS = {
    QueryEvidenceSlot.SYMBOL_IMPACT,
    QueryEvidenceSlot.GRAPHRAG_NOMINATION,
    QueryEvidenceSlot.PREMISES,
    QueryEvidenceSlot.CONTRADICTIONS,
    QueryEvidenceSlot.LOGIC_TRANSLATION,
    QueryEvidenceSlot.PROOF,
    QueryEvidenceSlot.COUNTEREXAMPLE,
    QueryEvidenceSlot.SECURITY,
}


def _cid(label: str) -> str:
    return plan_revision_cid({"fixture": label})


def _roots() -> PlanAuthorityRoots:
    return PlanAuthorityRoots(
        repository_id="repository:sha256:test",
        repository_root_cid=_cid("repository-root"),
        dirty_worktree_root=_cid("effective-tree"),
        task_source_id="task-source:markdown:test",
        task_source_revision=_cid("objective-revision"),
        policy_root=_cid("policy"),
        intent_ir_root=_cid("intent"),
        legal_ir_root=_cid("legal"),
        security_ir_root=_cid("security"),
        program_root=_cid("program"),
        capability_catalog_root=_cid("capability-catalog"),
        provider_catalog_root=_cid("provider-catalog"),
        usage_policy_root=_cid("usage"),
        configuration_root=_cid("configuration"),
    )


def _budget(**overrides: int) -> PlanRequestBudget:
    values = {
        "max_goals": 16,
        "max_tasks": 64,
        "max_graph_depth": 8,
        "max_output_paths": 128,
        "max_ready_width": 1,
        "max_repair_rounds": 2,
        "max_scan_bytes": 8 * 1024 * 1024,
        "max_analysis_operations": 16,
        "max_evidence_items": 128,
        "max_logic_families": 4,
        "max_model_calls": 2,
        "max_latency_ms": 80_000,
        "max_provider_tokens": 8_192,
        "max_cost_micros": 800,
    }
    values.update(overrides)
    return PlanRequestBudget(**values)


def _create(**overrides: object) -> PlanCreateRequest:
    values: dict[str, object] = {
        "prompt_source_cid": _cid("prompt"),
        "repository_id": "repository:sha256:test",
        "repository_root": "/workspace/test",
        "scope_paths": ("ipfs_accelerate_py/agent_supervisor",),
        "dirty_tree_policy": DirtyTreePolicy.OBSERVE_AND_BIND,
        "task_source_kind": TaskSourceKind.BOTH,
        "board_namespace": "planner-test",
        "alias_prefix": "PDR",
        "roots": _roots(),
        "budget": _budget(),
        "required_analysis_operations": ("symbol_impact",),
        "optional_analysis_operations": ("graphrag_retrieval",),
        "required_logic_families": ("tdfol",),
        "optional_logic_families": (),
        "fallback_policy": FallbackPolicy.FAIL_CLOSED,
        "redacted_source_metadata": {
            "concepts": ["authorization", "cache invalidation"],
            "changed_paths": ["ipfs_accelerate_py/agent_supervisor/planning/x.py"],
            "symbols": ["Planner.compile"],
            "contract_refs": ["contract:plan@1"],
        },
        "caller": "principal:test",
        "idempotency_key": "create:1",
    }
    values.update(overrides)
    return PlanCreateRequest(**values)


def _population(kind: PopulationKind, label: str) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=(_cid(label),))


def _steer(**overrides: object) -> PlanSteerRequest:
    values: dict[str, object] = {
        "directive_cid": _cid("directive"),
        "base_admitted_plan_root": _cid("admitted"),
        "base_materialized_plan_root": _cid("materialized"),
        "plan_revision": 2,
        "parent_revision": 1,
        "roots": _roots(),
        "event_cursor": _cid("event"),
        "status_population": _population(PopulationKind.UNSTARTED, "ready"),
        "claimed_population": _population(PopulationKind.CLAIMED, "running"),
        "accepted_population": _population(PopulationKind.ACCEPTED, "done"),
        "accepted_evidence_root": _cid("accepted-evidence"),
        "completion_revision": _cid("completion"),
        "allowed_delta_operations": (PlanDeltaOperation.ADD_TASK.value,),
        "budget": _budget(),
        "redacted_directive_metadata": {
            "affected_paths": ["ipfs_accelerate_py/agent_supervisor/prompt/service.py"],
            "affected_symbols": ["PlanService.steer"],
            "open_frontier_refs": ["frontier:dynamic-dispatch"],
        },
        "caller": "principal:test",
        "idempotency_key": "steer:2",
    }
    values.update(overrides)
    return PlanSteerRequest(**values)


def _diagnosis() -> dict[str, object]:
    roots = _roots()
    return {
        "finding_cid": _cid("finding"),
        "authority_roots": {
            "repository_id": roots.repository_id,
            "tree_id": roots.dirty_worktree_root,
            "task_source_revision": roots.task_source_revision,
            "policy_root": roots.policy_root,
            "capability_catalog_root": roots.capability_catalog_root,
            "provider_catalog_root": roots.provider_catalog_root,
            "security_ir_root": roots.security_ir_root,
            "intent_ir_root": roots.intent_ir_root,
        },
        "path": "ipfs_accelerate_py/agent_supervisor/analysis/broken.py",
        "symbol": "Broken.run",
        "observation_refs": ["observation:test-failure"],
        "open_frontier_refs": ["frontier:ffi"],
        "proof_obligations": ["obligation:postcondition"],
        "budget": _budget()._payload(),
    }


@pytest.mark.parametrize(
    ("input_record", "kind"),
    [
        (_create(), QueryInputKind.CREATE),
        (_steer(), QueryInputKind.STEER),
        (_diagnosis(), QueryInputKind.DIAGNOSIS),
    ],
)
def test_fixed_rules_compile_all_required_queries_for_every_input_kind(
    input_record: object, kind: QueryInputKind
) -> None:
    plan = PlanAnalysisQueryPlanner().compile(input_record)

    assert plan.input_kind is kind
    assert plan.ready
    assert {query.slot for query in plan.required_queries} == MANDATORY_SLOTS
    assert all(
        query.requirement is QueryRequirement.REQUIRED
        for query in plan.required_queries
    )
    assert all(query.why and query.question for query in plan.required_queries)
    assert all(query.provider_capabilities for query in plan.required_queries)
    assert all(query.operation_spec_id for query in plan.required_queries)
    assert all(query.strategy_ids for query in plan.required_queries)
    assert plan.query_for_slot(QueryEvidenceSlot.GRAPHRAG_NOMINATION).nomination_only
    assert (
        plan.query_for_slot(QueryEvidenceSlot.SECURITY).operation
        is AnalysisOperation.CONTRADICTION_SEARCH
    )


def test_queries_bind_exact_scope_bounds_cache_and_fail_closed_semantics() -> None:
    request = _create()
    plan = compile_reasoning_query_plan(request)

    assert plan.scope.repository_id == request.repository_id
    assert plan.scope.tree_id == request.roots.dirty_worktree_root
    assert plan.scope.objective_revision == request.roots.task_source_revision
    assert plan.scope.policy_id == request.roots.policy_root
    assert "Planner.compile" in plan.scope.symbols
    assert plan.scope.changed_paths == (
        "ipfs_accelerate_py/agent_supervisor/planning/x.py",
    )

    for query in plan.required_queries:
        assert query.bounds.executable
        assert query.bounds.max_input_bytes <= request.budget.max_scan_bytes
        assert query.bounds.max_output_bytes <= request.budget.max_scan_bytes
        assert query.bounds.max_items <= request.budget.max_evidence_items
        assert query.bounds.timeout_ms <= request.budget.max_latency_ms
        assert query.bounds.max_cost_micros <= request.budget.max_cost_micros
        assert query.cache.allow_stale is False
        assert query.cache.semantic_cache_key.startswith(
            "reasoning-query-cache-key:sha256:"
        )
        assert query.failure.unavailable is (
            QueryFailureDisposition.BLOCK_CANDIDATE_GENERATION
        )
        assert query.failure.timeout is (
            QueryFailureDisposition.BLOCK_CANDIDATE_GENERATION
        )

    assert sum(query.bounds.max_cost_micros for query in plan.queries) <= (
        request.budget.max_cost_micros
    )
    payload = plan.to_dict()
    assert payload["provider_selection_deferred_to_registry"] is True
    assert all(
        query["provider_selection"] == "registry_at_execution"
        and query["endpoint_selection"] is False
        and query["credential_selection"] is False
        for query in payload["queries"]
    )


def test_query_execution_envelopes_are_built_only_through_registry() -> None:
    planner = PlanAnalysisQueryPlanner()
    plan = planner.compile(_create())

    requests = [
        query.to_analysis_request(planner.operation_registry)
        for query in plan.required_queries
    ]

    assert {request.request_id for request in requests} == {
        query.query_id for query in plan.required_queries
    }
    assert all(
        request.metadata["registry_id"] == planner.operation_registry.registry_id
        for request in requests
    )
    assert all(
        request.metadata["tree_id"] == plan.scope.tree_id for request in requests
    )


def test_model_suggestions_cannot_suppress_fixed_rules() -> None:
    plan = PlanAnalysisQueryPlanner().compile(
        _create(),
        model_suggestions={
            "suppress_operations": [
                "symbol_impact",
                "contradiction_search",
                "proof_candidate_analysis",
            ],
            "required": [],
            "optional_operations": ["graphrag_retrieval"],
        },
    )

    assert {query.slot for query in plan.required_queries} == MANDATORY_SLOTS
    assert all(
        query.suggestion_source != "model_nomination" for query in plan.required_queries
    )


@pytest.mark.parametrize(
    "suggestion",
    [
        {"provider_id": "remote:chosen-by-model"},
        {"endpoint": "https://untrusted.invalid"},
        {"nested": {"api_key": "not-allowed"}},
    ],
)
def test_model_suggestions_cannot_select_providers_endpoints_or_credentials(
    suggestion: dict[str, object],
) -> None:
    with pytest.raises(UnsafeModelSuggestionError):
        PlanAnalysisQueryPlanner().compile(_create(), model_suggestions=suggestion)


def test_mandatory_query_budget_fails_closed() -> None:
    request = _create(budget=replace(_budget(), max_analysis_operations=7))
    with pytest.raises(QueryPlanningBudgetError):
        PlanAnalysisQueryPlanner().compile(request)


def _all_query_evidence(plan: object) -> dict[str, tuple[str, ...]]:
    return {
        query.query_id: (f"evidence:{query.slot.value}",)
        for query in plan.required_queries
    }


@pytest.mark.parametrize(
    "claim_class",
    [
        ClaimClass.CODE,
        ClaimClass.POLICY,
        ClaimClass.SECURITY,
        ClaimClass.AUTHORIZATION,
        ClaimClass.RESOURCE,
        ClaimClass.PROOF,
        ClaimClass.COMPLETION,
    ],
)
def test_post_proposal_coverage_rejects_prompt_only_authority(
    claim_class: ClaimClass,
) -> None:
    planner = PlanAnalysisQueryPlanner()
    plan = planner.compile(_create())
    receipt = planner.rerun_coverage_after_proposal(
        plan,
        {
            "proposal_id": "proposal:prompt-only",
            "claims": [
                {
                    "claim_id": f"claim:{claim_class.value}",
                    "claim_class": claim_class.value,
                    "evidence_authorities": [EvidenceAuthority.PROMPT.value],
                    "evidence_references": ["prompt-source:1"],
                }
            ],
        },
        query_evidence=_all_query_evidence(plan),
    )

    assert receipt.decision is CoverageDecision.BLOCKED
    assert receipt.phase == "post_proposal"
    assert any(
        "prompt_model_or_retrieval_evidence_not_authoritative" in slot.reason_codes
        for slot in receipt.claim_slots
    )


def test_post_proposal_coverage_requires_every_query_then_accepts_independent_claims() -> (
    None
):
    planner = PlanAnalysisQueryPlanner()
    plan = planner.compile(_diagnosis())
    evidence = _all_query_evidence(plan)
    evidence.pop(plan.query_for_slot(QueryEvidenceSlot.SECURITY).query_id)

    blocked = planner.rerun_coverage_after_proposal(
        plan,
        {"proposal_id": "proposal:1", "claims": []},
        query_evidence=evidence,
    )
    assert blocked.planning_blocked
    assert any("security" in blocker for blocker in blocked.blockers)

    ready = planner.rerun_coverage_after_proposal(
        plan,
        {
            "proposal_id": "proposal:2",
            "claims": [
                {
                    "claim_id": "claim:code",
                    "claim_class": "code",
                    "evidence_authorities": ["current_root_fact"],
                    "evidence_references": ["ast-fact:current"],
                },
                {
                    "claim_id": "claim:policy",
                    "claim_class": "policy",
                    "evidence_authorities": ["reviewed_policy"],
                    "evidence_references": ["policy-receipt:reviewed"],
                },
                {
                    "claim_id": "claim:security",
                    "claim_class": "security",
                    "evidence_authorities": ["security_analysis"],
                    "evidence_references": ["security-ir-check:1"],
                },
            ],
        },
        query_evidence=_all_query_evidence(plan),
    )
    assert ready.ready
    assert ready.query_plan_id == plan.plan_id
    assert ready.current_tree_id == plan.scope.tree_id


def test_query_coverage_rejects_prompt_and_cross_root_result_bindings() -> None:
    planner = PlanAnalysisQueryPlanner()
    plan = planner.compile(_create())
    evidence: dict[str, object] = dict(_all_query_evidence(plan))
    proof = plan.query_for_slot(QueryEvidenceSlot.PROOF)
    evidence[proof.query_id] = {
        "query_id": proof.query_id,
        "tree_id": plan.scope.tree_id,
        "authority": "prompt_nomination",
        "evidence_references": ["prompt:claimed-proof"],
    }
    security = plan.query_for_slot(QueryEvidenceSlot.SECURITY)
    evidence[security.query_id] = {
        "query_id": security.query_id,
        "tree_id": "tree:stale",
        "authority": "security_analysis",
        "evidence_references": ["security-check:stale"],
    }

    receipt = planner.rerun_coverage_after_proposal(
        plan,
        {"proposal_id": "proposal:untrusted-query-results", "claims": []},
        query_evidence=evidence,
    )

    assert receipt.planning_blocked
    reasons = {reason for slot in receipt.query_slots for reason in slot.reason_codes}
    assert "prompt_or_model_evidence_not_authoritative" in reasons
    assert "stale_or_cross_root_evidence" in reasons


def test_plan_and_query_identities_are_deterministic() -> None:
    planner = PlanAnalysisQueryPlanner()
    first = planner.compile(_steer())
    second = planner.compile(_steer())

    assert first.plan_id == second.plan_id
    assert [item.query_id for item in first.queries] == [
        item.query_id for item in second.queries
    ]
    assert first.to_dict() == second.to_dict()
