from __future__ import annotations

import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    ProofRepairPolicy,
    ProofRepairWorkKind,
    generate_proof_repair_work,
)
from ipfs_accelerate_py.agent_supervisor.plan_evaluator import (
    PlanBranch,
    ProofAwarePlanCandidate,
    ProofAwarePlanPolicy,
    evaluate_proof_aware_plans,
)
from ipfs_accelerate_py.agent_supervisor.proof_context import (
    ContextEntry,
    ContextTrust,
    ProofContextError,
    ProofContextCapsule,
    ProofContextLimits,
    ProofContextQuery,
    ProofContextTarget,
    ProofPlanningContextCapsule,
    ProofPlanningContextLimits,
    ProofTranscriptExcerpt,
    SourceExcerpt,
    build_proof_planning_context_capsule,
    estimate_context_tokens,
)


def _branch(branch_id: str, *, branch_risk: float = 0.2) -> PlanBranch:
    return PlanBranch.from_dict(
        {
            "branch_id": branch_id,
            "summary": f"Implement the {branch_id} proof-aware plan.",
            "predicted_files": [f"src/{branch_id}.py", f"tests/test_{branch_id}.py"],
            "predicted_symbols": [f"{branch_id}.implement", f"test_{branch_id}"],
            "dependencies": ["REF-267"],
            "validation_commands": [f"python -m pytest tests/test_{branch_id}.py -q"],
            "validation_proof": [
                "The focused test exercises the obligation and checks its evidence."
            ],
            "estimated_cost": 1.0,
            "risk": branch_risk,
            "expected_objective_delta": 0.7,
            "source": "llm_router",
        }
    )


def _candidate(
    candidate_id: str,
    *,
    obligation_impact: tuple[str, ...] = ("obligation:lease-safety",),
    required_assurance: str = "kernel_verified",
    proof_cost: float = 1.0,
    cache_likelihood: float = 0.5,
    dependencies: tuple[str, ...] = ("REF-267",),
    expected_evidence_delta: tuple[str, ...] = ("kernel receipt for lease safety",),
    proof_critical_path: float = 0.5,
    downstream_unlock_value: float = 0.5,
    risk: float = 0.2,
    freshness: float = 1.0,
    resource_classes: tuple[str, ...] = ("solver", "kernel"),
) -> ProofAwarePlanCandidate:
    return ProofAwarePlanCandidate(
        branch=_branch(candidate_id, branch_risk=risk),
        obligation_impact=obligation_impact,
        required_assurance=required_assurance,
        proof_cost=proof_cost,
        cache_likelihood=cache_likelihood,
        dependencies=dependencies,
        expected_evidence_delta=expected_evidence_delta,
        proof_critical_path=proof_critical_path,
        downstream_unlock_value=downstream_unlock_value,
        risk=risk,
        freshness=freshness,
        resource_classes=resource_classes,
    )


def _proof_context() -> ProofContextCapsule:
    """A deliberately large graph capsule whose router projection must be small."""

    return ProofContextCapsule(
        target=ProofContextTarget.CODEX,
        query=ProofContextQuery(
            task_id="REF-270",
            obligation_ids=(
                "obligation:lease-safety",
                "obligation:router-bounds",
            ),
        ),
        limits=ProofContextLimits(),
        trusted_facts=(
            ContextEntry(
                trust=ContextTrust.TRUSTED_FACT,
                kind="proof_receipt",
                record_id="receipt:lease-safety",
                fields={
                    "obligation_id": "obligation:lease-safety",
                    "verdict": "proved",
                    "freshness": "current",
                    "authoritative_assurance": "kernel_verified",
                },
            ),
        ),
        unsupported_semantics=(
            ContextEntry(
                trust=ContextTrust.UNSUPPORTED_SEMANTICS,
                kind="proof_obligation",
                record_id="obligation:router-bounds",
                fields={
                    "obligation_id": "obligation:router-bounds",
                    "support_status": "unsupported",
                    "required_assurance": "kernel_verified",
                },
            ),
        ),
        required_fallback_checks=("pytest:test_router_bounds",),
        source_excerpts=(
            SourceExcerpt(
                symbol="repository.UnrelatedHugeSymbol",
                path="repository/unrelated.py",
                text="UNRELATED-SOURCE-MUST-NOT-REACH-ROUTER " * 300,
            ),
        ),
        proof_transcripts=(
            ProofTranscriptExcerpt(
                receipt_id="receipt:lease-safety",
                obligation_id="obligation:lease-safety",
                text="UNBOUNDED-KERNEL-TRANSCRIPT-MUST-NOT-REACH-ROUTER " * 300,
            ),
        ),
    )


def test_candidate_declares_complete_proof_planning_contract_and_round_trips() -> None:
    candidate = _candidate(
        "critical",
        obligation_impact=("obligation:a", "obligation:b"),
        proof_cost=2.5,
        cache_likelihood=0.75,
        dependencies=("REF-252", "REF-267"),
        expected_evidence_delta=("receipt:a", "receipt:b"),
        resource_classes=("solver", "kernel", "artifact_store"),
    )

    payload = candidate.to_dict()
    assert payload["candidate_id"] == "critical"
    assert payload["obligation_impact"] == ["obligation:a", "obligation:b"]
    assert payload["required_assurance"] == "kernel_verified"
    assert payload["proof_cost"] == 2.5
    assert payload["cache_likelihood"] == 0.75
    assert payload["dependencies"] == ["REF-252", "REF-267"]
    assert payload["expected_evidence_delta"] == ["receipt:a", "receipt:b"]
    assert payload["resource_classes"] == ["solver", "kernel", "artifact_store"]
    assert ProofAwarePlanCandidate.from_dict(payload) == candidate

    inherited_risk = ProofAwarePlanCandidate(
        branch=_branch("risk-inheritance", branch_risk=0.65),
        obligation_impact=("obligation:risk",),
        required_assurance="kernel_verified",
        proof_cost=1.0,
        cache_likelihood=0.0,
        dependencies=(),
        expected_evidence_delta=("receipt:risk",),
        resource_classes=("kernel",),
    )
    assert inherited_risk.risk == 0.65


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("obligation_impact", []),
        ("required_assurance", ""),
        ("proof_cost", -1),
        ("cache_likelihood", 1.01),
        ("expected_evidence_delta", []),
        ("risk", float("nan")),
        ("freshness", -0.01),
    ],
)
def test_candidate_rejects_missing_or_non_finite_proof_metadata(
    field: str,
    value: object,
) -> None:
    payload = _candidate("invalid").to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        ProofAwarePlanCandidate.from_dict(payload)


def test_priority_uses_proof_path_unlock_risk_freshness_cache_and_resources() -> None:
    critical = _candidate(
        "critical",
        proof_critical_path=1.0,
        downstream_unlock_value=1.0,
        risk=0.05,
        freshness=0.1,
        cache_likelihood=0.9,
        resource_classes=("solver", "kernel"),
    )
    stale_unavailable = _candidate(
        "stale-unavailable",
        proof_critical_path=0.2,
        downstream_unlock_value=0.2,
        risk=0.8,
        freshness=1.0,
        cache_likelihood=0.0,
        resource_classes=("gpu-proof-provider",),
    )
    result = evaluate_proof_aware_plans(
        [stale_unavailable, critical],
        policy=ProofAwarePlanPolicy(
            available_resource_classes=("solver", "kernel", "artifact_store")
        ),
    )

    assert result.selected.candidate.candidate_id == "critical"
    assert len(result.rejected) == 1
    rejected = result.rejected[0]
    assert rejected.candidate.candidate_id == "stale-unavailable"
    assert rejected.rationale
    assert result.selected.score_millionths > rejected.score_millionths
    rationale = " ".join(
        (*result.selected.rationale, *rejected.rationale)
    ).lower()
    for factor in (
        "critical path",
        "downstream",
        "risk",
        "freshness",
        "cache",
        "resource",
    ):
        assert factor in rationale

    # Provider order is never a priority input, and every rejected alternative
    # remains visible with an actionable rationale.
    reverse = evaluate_proof_aware_plans(
        [critical, stale_unavailable],
        policy=ProofAwarePlanPolicy(
            available_resource_classes=("solver", "kernel", "artifact_store")
        ),
    )
    assert reverse.to_dict() == result.to_dict()
    assert reverse.to_dict()["rejected"][0]["rationale"]


def test_router_projection_is_bounded_and_retains_rejected_rationale_only() -> None:
    candidates = [
        _candidate(
            f"candidate-{index}",
            obligation_impact=(f"obligation:{index}",),
            expected_evidence_delta=(f"receipt:{index}",),
        )
        for index in range(6)
    ]
    evaluation = evaluate_proof_aware_plans(
        candidates,
        policy=ProofAwarePlanPolicy(
            available_resource_classes=("solver", "kernel")
        ),
    )
    limits = ProofPlanningContextLimits(
        max_candidates=3,
        max_obligations=3,
        max_rejected_alternatives=2,
        max_rationale_bytes=96,
        max_dependencies=3,
        max_resource_classes=3,
        max_bytes=6_000,
        max_tokens=1_500,
    )

    capsule = build_proof_planning_context_capsule(
        _proof_context(),
        candidates=candidates,
        evaluation=evaluation,
        available_resource_classes=("solver", "kernel"),
        proof_critical_path=("obligation:0", "obligation:1"),
        limits=limits,
    )

    prompt = capsule.render_for_router()
    payload = json.loads(prompt)
    assert len(capsule.candidates) <= 3
    assert len(capsule.obligations) <= 3
    assert len(capsule.rejected_alternatives) <= 2
    lease_obligation = next(
        item
        for item in capsule.obligations
        if item.obligation_id == "obligation:lease-safety"
    )
    assert lease_obligation.reusable_receipt_ids == ("receipt:lease-safety",)
    assert lease_obligation.required_assurance == "kernel_verified"
    assert all(item.rationale for item in capsule.rejected_alternatives)
    assert all(
        item.rationale_bytes <= 96
        for item in capsule.rejected_alternatives
    )
    assert capsule.usage.bytes == len(prompt.encode("utf-8")) <= limits.max_bytes
    assert capsule.usage.tokens == estimate_context_tokens(prompt) <= limits.max_tokens
    assert capsule.truncated is True
    assert payload["source_capsule_id"] == _proof_context().capsule_id

    for prohibited in (
        "UNRELATED-SOURCE-MUST-NOT-REACH-ROUTER",
        "UNBOUNDED-KERNEL-TRANSCRIPT-MUST-NOT-REACH-ROUTER",
        "source_excerpts",
        "proof_transcripts",
        "untrusted_suggestions",
    ):
        assert prohibited not in prompt
    assert ProofPlanningContextCapsule.from_json(prompt) == capsule
    assert _proof_context().for_router(
        candidates=candidates,
        evaluation=evaluation,
        available_resource_classes=("solver", "kernel"),
        limits=limits,
    ).usage.bytes <= limits.max_bytes

    with pytest.raises(ProofContextError, match="rationale"):
        build_proof_planning_context_capsule(
            _proof_context(),
            candidates=candidates,
            evaluation=evaluation,
            limits=ProofPlanningContextLimits(max_rationale_bytes=1),
        )


def _obligation(
    obligation_id: str,
    status: str,
    *,
    statement: str,
    template_id: str = "",
    premise_ids: tuple[str, ...] = (),
    fallback_checks: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = ("REF-267",),
) -> dict[str, Any]:
    return {
        "obligation_id": obligation_id,
        "task_id": "REF-270",
        "statement": statement,
        "status": status,
        "required_assurance": "kernel_verified",
        "template_id": template_id,
        "premise_ids": list(premise_ids),
        "fallback_checks": list(fallback_checks),
        "dependencies": list(dependencies),
    }


def test_unsupported_and_failed_obligations_generate_finite_deduplicated_repairs() -> None:
    obligations = [
        _obligation(
            "obligation:template",
            "unsupported",
            statement="The router projection is bounded.",
        ),
        _obligation(
            "obligation:test",
            "unsupported",
            statement="The fallback rejects stale fencing tokens.",
            template_id="lease-fencing",
            fallback_checks=("pytest:test_stale_fencing_token",),
        ),
        _obligation(
            "obligation:premise",
            "failed",
            statement="The proof uses the current dependency snapshot.",
            template_id="dependency-snapshot",
            premise_ids=("premise:missing-current-tree",),
        ),
        _obligation(
            "obligation:manual",
            "contradicted",
            statement="The policy contradiction requires human disposition.",
            template_id="policy-consistency",
            fallback_checks=("manual:review-policy-contradiction",),
        ),
        # Same semantic obligation under another provider ID must not create an
        # unbounded second repair.
        _obligation(
            "provider-retry:template",
            "unsupported",
            statement="  the ROUTER projection is bounded  ",
        ),
    ]
    policy = ProofRepairPolicy(max_work_items=4, max_per_obligation=2)

    forward = generate_proof_repair_work(obligations, policy=policy)
    reverse = generate_proof_repair_work(reversed(obligations), policy=policy)

    assert forward.to_dict() == reverse.to_dict()
    assert 1 <= len(forward.work) <= 4
    assert len({item.semantic_key for item in forward.work}) == len(forward.work)
    assert {item.repair_kind for item in forward.work} == {
        ProofRepairWorkKind.TEMPLATE,
        ProofRepairWorkKind.TEST,
        ProofRepairWorkKind.PREMISE,
        ProofRepairWorkKind.MANUAL_REVIEW,
    }
    assert all(item.obligation_ids for item in forward.work)
    assert all(item.dependencies == ("REF-267",) for item in forward.work)
    assert all(item.expected_evidence_delta for item in forward.work)
    assert any(
        rejection.reason in {"semantic_duplicate", "duplicate_semantic_work"}
        for rejection in forward.rejected
    )

    repeated = generate_proof_repair_work(
        obligations,
        existing_work=forward.work,
        policy=policy,
    )
    assert repeated.work == ()
    assert repeated.rejected
    assert all(
        item.reason
        in {
            "semantic_duplicate",
            "duplicate_semantic_work",
            "existing_semantic_work",
        }
        for item in repeated.rejected
    )


def test_satisfied_obligations_do_not_generate_repair_work() -> None:
    result = generate_proof_repair_work(
        [
            _obligation(
                "obligation:done",
                "satisfied",
                statement="The current kernel receipt covers the obligation.",
                template_id="coverage",
            ),
            _obligation(
                "obligation:pending",
                "pending",
                statement="The leased proof attempt is still running.",
                template_id="lease",
            ),
        ]
    )

    assert result.work == ()
    assert result.truncated is False
