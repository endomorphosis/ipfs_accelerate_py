from __future__ import annotations

from dataclasses import replace
import hashlib
from statistics import median

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_retrieval import (
    retrieval_response_to_context_references,
    retrieve_analysis_evidence,
)
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    MIN_INPUT_TOKEN_REDUCTION_BPS,
    MIN_RETRY_INPUT_TOKEN_REDUCTION_BPS,
    VALUE_OF_INFORMATION_REQUIREMENT_ID,
    ContentAddressedContextStore,
    ContextCompilationError,
    ContextCompiler,
    ContextExpansionError,
    EvidenceExpansionRequest,
    EvidenceValuePairedFixture,
    EvidenceValuePolicy,
    ExclusionReason,
    InclusionReason,
    ValueOfInformationEvidence,
    expand_context,
    expand_context_for_question,
    expand_context_references,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
)


BINDING = {
    "repository_id": "repo:voi",
    "tree_id": "tree:current",
    "objective_id": "ASI-G210",
    "objective_revision": "sha256:objective",
    "policy_id": "policy:voi",
    "policy_revision": "sha256:policy",
    "caller": "supervisor:test",
    "stage": "planning",
}
CORE = {
    "goal": {"id": "ASI-G210", "summary": "Compile decision-changing context"},
    "authority": {"mode": "proposal", "allowed_paths": ["src"]},
    "scope": {"paths": ["src/context.py"]},
    "acceptance": {"criteria": ["required coverage and safety remain unchanged"]},
}


def _budget(*, max_items: int = 16) -> ContextBudget:
    return ContextBudget(
        max_input_tokens=1_200,
        reserved_output_tokens=100,
        reserved_tool_tokens=20,
        max_items=max_items,
        max_item_bytes=16_384,
        max_serialized_bytes=262_144,
    )


def _tokenizer(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 32)


def _reference(
    reference_id: str,
    *,
    required: bool = False,
    priority: int = 0,
    expected_change: int | None = None,
    uncertainty: int = 0,
    uncertainty_reduction: int = 0,
    latency: int = 0,
    invalidation: int = 0,
    expansion: int = 0,
    diversity: str = "",
    summary: str = "bounded evidence",
) -> ContextReference:
    metadata: dict[str, object] = {
        "required": required,
        "priority": priority,
        "coverage_ids": (
            "criterion:safety" if required else f"criterion:{reference_id}",
        ),
        "uncertainty_bps": uncertainty,
        "uncertainty_reduction_bps": uncertainty_reduction,
        "latency_cost": latency,
        "invalidation_cost": invalidation,
        "expansion_cost": expansion,
        "diversity_key": diversity,
    }
    if expected_change is not None:
        metadata["expected_decision_change_bps"] = expected_change
    return ContextReference(
        reference_id=reference_id,
        kind="test-evidence",
        tier=ContextTier.INVARIANT if required else ContextTier.EVIDENCE,
        referenced_content_id="sha256:" + reference_id,
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        summary=summary,
        metadata=metadata,
    )


def _compiler(*, max_optional_items: int | None = None) -> ContextCompiler:
    return ContextCompiler(
        _budget(),
        tokenizer=_tokenizer,
        provider_context_window=1_420,
        value_policy=EvidenceValuePolicy(
            max_optional_items=max_optional_items
        ),
    )


def test_value_estimate_uses_all_benefit_and_cost_terms() -> None:
    reference = _reference(
        "estimate",
        expected_change=5_000,
        uncertainty=6_000,
        uncertainty_reduction=3_000,
        latency=50,
        invalidation=25,
        expansion=25,
        diversity="parser",
    )
    policy = EvidenceValuePolicy()

    first = policy.estimate(reference, token_cost=100)
    redundant = policy.estimate(
        reference,
        token_cost=100,
        selected_diversity_count=1,
    )

    assert first.total_cost == 200
    assert first.raw_value_score == 40_000_000
    assert first.value_score == first.raw_value_score
    assert redundant.diversity_penalty_bps == 5_000
    assert redundant.value_score < first.value_score


def test_required_evidence_bypasses_auction_and_zero_value_is_explicit() -> None:
    required = _reference(
        "required",
        required=True,
        expected_change=0,
    )
    zero_value = _reference(
        "no-decision-value",
        expected_change=0,
        uncertainty=8_000,
        uncertainty_reduction=0,
    )
    useful = _reference(
        "useful",
        expected_change=5_000,
        uncertainty=7_000,
        uncertainty_reduction=4_000,
    )

    result = _compiler().compile(
        **BINDING,
        **CORE,
        evidence=(zero_value, useful, required),
    )
    decisions = {item.reference_id: item for item in result.decisions}

    assert decisions["required"].included
    assert decisions["required"].reason is InclusionReason.REQUIRED
    assert decisions["no-decision-value"].reason is ExclusionReason.LOW_VALUE
    assert decisions["no-decision-value"].uncertainty_bps == 8_000
    assert decisions["no-decision-value"].value_score == 0
    assert "no-decision-value:token_budget" in result.capsule.omissions
    deferred = next(
        item
        for item in result.capsule.expansion_references
        if item.reference_id == "no-decision-value"
    )
    assert deferred.metadata["selection_exclusion_reason"] == "low_value"
    forged_deferred = replace(
        deferred,
        metadata={
            **dict(deferred.metadata),
            "selection_exclusion_reason": "item_limit",
        },
    )
    forged_capsule = replace(
        result.capsule,
        expansion_references=tuple(
            forged_deferred
            if item.reference_id == deferred.reference_id
            else item
            for item in result.capsule.expansion_references
        ),
    )
    forged_witness = replace(
        result.receipt.evidence,
        capsule_id=forged_capsule.capsule_id,
        artifact_digest=(
            "sha256:"
            + hashlib.sha256(
                forged_capsule.canonical_bytes()
            ).hexdigest()
        ),
    )
    forged_receipt = replace(
        result.receipt,
        capsule_id=forged_capsule.capsule_id,
        evidence=forged_witness,
    )
    with pytest.raises(ContextCompilationError, match="exclusion reason"):
        replace(
            result,
            capsule=forged_capsule,
            receipt=forged_receipt,
            verifier=None,
        )
    assert result.required_context_preserved


def test_marginal_ranking_penalizes_redundancy_before_item_limit() -> None:
    required = _reference("required", required=True, expected_change=0)
    first_parser = _reference(
        "parser-a",
        expected_change=9_000,
        diversity="parser",
    )
    redundant_parser = _reference(
        "parser-b",
        expected_change=8_500,
        diversity="parser",
    )
    independent_proof = _reference(
        "proof",
        expected_change=7_000,
        diversity="proof",
    )

    result = _compiler(max_optional_items=2).compile(
        **BINDING,
        **CORE,
        evidence=(
            redundant_parser,
            independent_proof,
            first_parser,
            required,
        ),
    )
    selected = {item.reference_id for item in result.capsule.evidence}
    decisions = {item.reference_id: item for item in result.decisions}

    assert selected == {"required", "parser-a", "proof"}
    assert decisions["parser-b"].reason is ExclusionReason.ITEM_LIMIT
    assert decisions["parser-b"].diversity_penalty_bps == 5_000
    assert decisions["parser-b"].value_score < decisions["proof"].value_score


def test_retrieval_relevance_is_a_bounded_input_to_context_value() -> None:
    response = retrieve_analysis_evidence(
        "parser proof",
        records=[
            {"task_id": "TASK-1", "title": "parser proof", "goal_id": "GOAL-1"},
            {"task_id": "TASK-2", "title": "parser helper", "goal_id": "GOAL-1"},
        ],
    )
    references = retrieval_response_to_context_references(
        response,
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        latency_cost=12,
        invalidation_cost=8,
        expansion_cost=5,
    )

    assert references
    assert all(item.metadata["retrieval_relevance_bps"] > 0 for item in references)
    assert all(item.metadata["expected_decision_change_bps"] > 0 for item in references)
    assert all(item.metadata["latency_cost"] == 12 for item in references)
    assert all(item.metadata["diversity_key"] == "goal:GOAL-1" for item in references)
    compiled = _compiler().compile(
        **BINDING,
        **CORE,
        evidence=references,
    )
    assert all(item.value_score > 0 for item in compiled.decisions)


def test_question_bound_expansion_requires_named_question_and_hash_handle() -> None:
    compiler = _compiler()
    store = ContentAddressedContextStore()
    question = "Which parser invariant invalidates the current branch?"
    handle = store.make_reference(
        "The parser requires a non-empty authority scope.",
        reference_id="parser-detail",
        kind="analysis-detail",
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        unresolved_questions=(question,),
    )
    candidate = ContextReference(
        reference_id=handle.reference_id,
        kind=handle.kind,
        tier=ContextTier.EVIDENCE,
        referenced_content_id=handle.referenced_content_id,
        repository_id=handle.repository_id,
        tree_id=handle.tree_id,
        summary="deferred parser detail",
        metadata={
            **dict(handle.metadata),
            "expected_decision_change_bps": 0,
        },
    )
    required = _reference("required", required=True, expected_change=0)
    parent = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, candidate),
    ).capsule

    with pytest.raises(ContextExpansionError, match="named unresolved question"):
        expand_context_references(
            compiler,
            parent,
            ("parser-detail",),
            store,
        )
    with pytest.raises(ContextExpansionError, match="named unresolved question"):
        expand_context(compiler, parent, (candidate,))
    with pytest.raises(ContextExpansionError, match="not bound"):
        expand_context_for_question(
            compiler,
            parent,
            EvidenceExpansionRequest("A different question?", ("parser-detail",)),
            store,
        )

    result = expand_context_for_question(
        compiler,
        parent,
        EvidenceExpansionRequest(question, ("parser-detail",)),
        store,
    )
    expanded = next(
        item
        for item in result.delta_capsule.evidence
        if item.reference_id == "parser-detail"
    )
    decision = next(
        item
        for item in result.decisions
        if item.reference_id == "parser-detail"
    )
    assert handle.referenced_content_id.startswith("sha256:")
    assert expanded.referenced_content_id == handle.referenced_content_id
    assert expanded.metadata["expansion_question"] == question
    assert decision.unresolved_question == question


def _paired_fixture(
    fixture_id: str,
    *,
    baseline: int = 1_000,
    selected: int = 550,
    baseline_retry: int = 1_000,
    selected_retry: int = 350,
) -> EvidenceValuePairedFixture:
    return EvidenceValuePairedFixture(
        fixture_id=fixture_id,
        accepted_criterion_ids=(f"criterion:{fixture_id}",),
        baseline_input_tokens=baseline,
        selected_input_tokens=selected,
        baseline_retry_input_tokens=baseline_retry,
        selected_retry_input_tokens=selected_retry,
        baseline_required_coverage_ids=("required:safety",),
        selected_required_coverage_ids=("required:safety",),
    )


def test_paired_evidence_proves_40_60_gates_without_coverage_drift() -> None:
    evidence = ValueOfInformationEvidence(
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        policy_id=BINDING["policy_id"],
        policy_revision=BINDING["policy_revision"],
        provider_id="provider:test",
        model_id="model:test",
        fixtures=(_paired_fixture("b"), _paired_fixture("a")),
    )

    assert evidence.requirement_id == VALUE_OF_INFORMATION_REQUIREMENT_ID
    assert evidence.input_token_reduction_bps >= MIN_INPUT_TOKEN_REDUCTION_BPS
    assert (
        evidence.retry_input_token_reduction_bps
        >= MIN_RETRY_INPUT_TOKEN_REDUCTION_BPS
    )
    assert ValueOfInformationEvidence.from_json(evidence.to_json()) == evidence
    assert evidence.to_dict()["required_coverage_preserved"] is True
    assert evidence.to_dict()["safety_preserved"] is True

    with pytest.raises(ContextCompilationError, match="required coverage"):
        replace(
            _paired_fixture("coverage"),
            selected_required_coverage_ids=("required:different",),
        )
    with pytest.raises(ContextCompilationError, match="less than 40"):
        ValueOfInformationEvidence(
            repository_id=BINDING["repository_id"],
            tree_id=BINDING["tree_id"],
            policy_id=BINDING["policy_id"],
            policy_revision=BINDING["policy_revision"],
            provider_id="provider:test",
            model_id="model:test",
            fixtures=(_paired_fixture("weak", selected=700),),
        )


def test_measured_paired_compiler_outputs_clear_40_60_gates() -> None:
    fixtures: list[EvidenceValuePairedFixture] = []
    base_reductions: list[float] = []
    retry_reductions: list[float] = []

    for fixture_index, evidence_lines in enumerate((70, 80, 90), start=1):
        required = _reference(
            f"required-{fixture_index}",
            required=True,
            expected_change=0,
        )
        optional = tuple(
            _reference(
                f"f{fixture_index}-optional-{item_index}",
                expected_change=9_500 - item_index * 500,
                uncertainty=8_000,
                uncertainty_reduction=7_500 - item_index * 400,
                diversity=f"topic:{item_index}",
                summary=("decision evidence " * evidence_lines).strip(),
            )
            for item_index in range(8)
        )
        baseline_compiler = _compiler()
        selected_compiler = _compiler(max_optional_items=2)
        baseline = baseline_compiler.compile(
            **BINDING,
            **CORE,
            evidence=(required, *optional),
        )
        selected = selected_compiler.compile(
            **BINDING,
            **CORE,
            evidence=(required, *optional),
        )

        selected_optional = next(
            item for item in selected.capsule.evidence if not item.required
        )
        changed_selected = tuple(
            replace(
                item,
                summary=item.summary + " updated",
                referenced_content_id=item.referenced_content_id + ":v2",
            )
            if item.reference_id == selected_optional.reference_id
            else item
            for item in selected.capsule.evidence
        )
        compact_retry = selected_compiler.compile_delta(
            selected.capsule,
            evidence=changed_selected,
        )
        changed_baseline = tuple(
            replace(
                item,
                summary=item.summary + " updated",
                referenced_content_id=item.referenced_content_id + ":v2",
            )
            if item.reference_id == selected_optional.reference_id
            else item
            for item in (required, *optional)
        )
        full_retry = baseline_compiler.compile(
            **BINDING,
            **CORE,
            evidence=changed_baseline,
        )

        baseline_coverage = tuple(
            sorted(
                coverage
                for item in baseline.capsule.evidence
                if item.required
                for coverage in item.coverage_ids
            )
        )
        selected_coverage = tuple(
            sorted(
                coverage
                for item in selected.capsule.evidence
                if item.required
                for coverage in item.coverage_ids
            )
        )
        retry_coverage = tuple(
            sorted(
                coverage
                for item in compact_retry.reconstructed_capsule.evidence
                if item.required
                for coverage in item.coverage_ids
            )
        )
        assert baseline_coverage == selected_coverage == retry_coverage
        assert baseline.required_context_preserved
        assert selected.required_context_preserved
        assert compact_retry.invariant_core_preserved

        fixtures.append(
            EvidenceValuePairedFixture(
                fixture_id=f"measured-{fixture_index}",
                accepted_criterion_ids=(
                    f"criterion:measured-{fixture_index}",
                ),
                baseline_input_tokens=baseline.capsule.input_tokens,
                selected_input_tokens=selected.capsule.input_tokens,
                baseline_retry_input_tokens=full_retry.capsule.input_tokens,
                selected_retry_input_tokens=compact_retry.receipt.delta_tokens,
                baseline_required_coverage_ids=baseline_coverage,
                selected_required_coverage_ids=selected_coverage,
            )
        )
        base_reductions.append(
            1
            - selected.capsule.input_tokens
            / baseline.capsule.input_tokens
        )
        retry_reductions.append(
            1
            - compact_retry.receipt.delta_tokens
            / full_retry.capsule.input_tokens
        )

    evidence = ValueOfInformationEvidence(
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        policy_id=BINDING["policy_id"],
        policy_revision=BINDING["policy_revision"],
        provider_id="provider:test",
        model_id="model:test",
        fixtures=tuple(fixtures),
    )

    assert median(base_reductions) >= 0.40
    assert median(retry_reductions) >= 0.60
    assert evidence.input_token_reduction_bps >= MIN_INPUT_TOKEN_REDUCTION_BPS
    assert (
        evidence.retry_input_token_reduction_bps
        >= MIN_RETRY_INPUT_TOKEN_REDUCTION_BPS
    )
