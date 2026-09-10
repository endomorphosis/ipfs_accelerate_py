"""ASEH-023: typed, budget-reserved smallest-adequate executor admission."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_budget import (
    CognitiveCost,
    ObjectiveCognitiveBudgetLedger,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    BudgetPurpose,
    CognitiveBudget,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.route_policy import (
    EscalationDisposition,
    admit_model_escalation,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.unresolved_question import (
    build_unresolved_question,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import ModelRoute
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    AnalysisKind,
    AvailableModelCapability,
    CapabilityLocality,
    CounterexampleQuality,
    ModelRouteFacts,
    ModelRoutePolicy,
    default_inventory,
    decide_model_route,
    policy_cid_for,
)


def _budget(**overrides: int) -> CognitiveBudget:
    values = {
        "max_total_model_calls": 3,
        "max_strong_model_calls": 1,
        "max_input_tokens": 10_000,
        "max_output_tokens": 2_000,
        "max_provider_spend_micros": 100_000,
        "max_proof_time_ms": 500,
        "max_validation_time_ms": 1_000,
        "max_human_questions": 2,
        "max_repair_rounds": 2,
        "max_plan_branches": 2,
        "max_context_expansions": 2,
        "max_wall_time_ms": 10_000,
        "validation_reserve_ms": 400,
        "proof_reserve_ms": 0,
    }
    values.update(overrides)
    return CognitiveBudget(**values)


def _facts(**overrides: object) -> ModelRouteFacts:
    values: dict[str, object] = {
        "context_token_estimate": 1_000,
        "analysis_kind": AnalysisKind.LOCALIZED_EXACT,
        "opaque_dependency_count": 0,
        "risk_level": "low",
        "dependency_cone_size": 2,
        "changed_file_count": 1,
        "counterexample_quality": CounterexampleQuality.MINIMIZED,
        "exact_contract_available": True,
        "environment_reproducible": True,
    }
    values.update(overrides)
    return ModelRouteFacts(**values)


def _decision(facts: ModelRouteFacts, inventory: tuple[AvailableModelCapability, ...] | None = None):
    return decide_model_route(
        facts,
        available_models=inventory if inventory is not None else default_inventory(),
        policy=ModelRoutePolicy(policy_cid=policy_cid_for("aseh-023")),
    )


def _question(
    *,
    route: ModelRoute = ModelRoute.SMALL_LOCAL_MODEL,
    context_budget: int = 2_000,
    cost_budget: int = 50_000,
):
    capability = {
        ModelRoute.SMALL_LOCAL_MODEL: "local_small_specialist_model",
        ModelRoute.MEDIUM_MODEL: "local_or_remote_medium_model",
        ModelRoute.FRONTIER_MODEL: "remote_strong_or_frontier_model",
        ModelRoute.HUMAN_REVIEW_REQUIRED: "human_decision",
    }[route]
    return build_unresolved_question(
        exact_question="Which bounded executor can resolve the remaining contract ambiguity?",
        why_prior_deterministic_stages_could_not_resolve=(
            "All deterministic stages completed but two valid contract interpretations remain."
        ),
        evidence_available=["current deterministic-stage receipt"],
        evidence_missing=["authoritative interpretation"],
        candidate_decisions_answer_could_change=sorted(
            {route.value, ModelRoute.HUMAN_REVIEW_REQUIRED.value}
            if route is not ModelRoute.HUMAN_REVIEW_REQUIRED
            else {route.value, ModelRoute.FRONTIER_MODEL.value}
        ),
        minimum_model_capability=capability,
        context_budget=context_budget,
        response_schema={"type": "string", "enum": ["interpretation_a", "interpretation_b"]},
        deadline="2027-01-02T03:04:05Z",
        cost_budget=cost_budget,
    )


def _admit(decision, question, ledger, **overrides: int):
    values = {
        "expected_output_tokens": 200,
        "expected_provider_spend_micros": 10_000,
        "expected_wall_time_ms": 500,
    }
    values.update(overrides)
    return admit_model_escalation(
        decision,
        unresolved_question=question,
        budget_ledger=ledger,
        idempotency_key="aseh-023-admission",
        action_id="aseh-023-route",
        **values,
    )


def test_small_local_capability_beats_an_available_remote_frontier() -> None:
    inventory = (
        AvailableModelCapability(
            capability_tier=ModelRoute.SMALL_LOCAL_MODEL,
            context_limit_tokens=4_000,
            locality=CapabilityLocality.LOCAL,
            available=True,
        ),
        AvailableModelCapability(
            capability_tier=ModelRoute.FRONTIER_MODEL,
            context_limit_tokens=100_000,
            locality=CapabilityLocality.REMOTE,
            available=True,
        ),
    )
    decision = _decision(_facts(), inventory)
    assert decision.route is ModelRoute.SMALL_LOCAL_MODEL


def test_remote_small_inventory_does_not_satisfy_the_local_small_route() -> None:
    inventory = (
        AvailableModelCapability(
            capability_tier=ModelRoute.SMALL_LOCAL_MODEL,
            context_limit_tokens=4_000,
            locality=CapabilityLocality.REMOTE,
            available=True,
        ),
        AvailableModelCapability(
            capability_tier=ModelRoute.FRONTIER_MODEL,
            context_limit_tokens=100_000,
            locality=CapabilityLocality.REMOTE,
            available=True,
        ),
    )
    decision = _decision(_facts(), inventory)
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert "required_tier_unavailable" in decision.decisive_reason_codes


def test_model_invocation_requires_matching_question_and_budget_reservation() -> None:
    decision = _decision(_facts())
    ledger = ObjectiveCognitiveBudgetLedger(_budget())
    admission = _admit(decision, _question(), ledger)

    assert admission.disposition is EscalationDisposition.ADMITTED
    assert admission.admitted is True
    assert admission.reservation is not None
    assert admission.reservation.question_id == admission.question_id
    assert admission.reservation.purpose is BudgetPurpose.MODEL
    assert admission.reservation.max_total_model_calls == 1
    assert admission.reservation.max_input_tokens == decision.context_token_estimate
    assert admission.reservation.max_provider_spend_micros == 10_000


def test_missing_or_mismatched_question_denies_without_reservation() -> None:
    decision = _decision(_facts())
    ledger = ObjectiveCognitiveBudgetLedger(_budget())

    missing = _admit(decision, None, ledger)
    assert missing.disposition is EscalationDisposition.DENIED
    assert "missing_typed_unresolved_question" in missing.reason_codes
    assert not ledger.snapshot().reservations

    wrong_capability = _admit(
        decision,
        _question(route=ModelRoute.MEDIUM_MODEL),
        ledger,
    )
    assert wrong_capability.disposition is EscalationDisposition.DENIED
    assert "question_capability_does_not_match_selected_route" in wrong_capability.reason_codes
    assert not ledger.snapshot().reservations


def test_question_context_and_cost_ceilings_fail_closed_before_reservation() -> None:
    decision = _decision(_facts())
    ledger = ObjectiveCognitiveBudgetLedger(_budget())

    context = _admit(decision, _question(context_budget=999), ledger)
    assert "question_context_budget_exceeded" in context.reason_codes

    cost = _admit(
        decision,
        _question(cost_budget=9_999),
        ledger,
    )
    assert "question_cost_budget_exceeded" in cost.reason_codes
    assert not ledger.snapshot().reservations

    time = _admit(
        decision,
        _question(),
        ObjectiveCognitiveBudgetLedger(_budget(max_wall_time_ms=499)),
    )
    assert time.disposition is EscalationDisposition.DENIED
    assert "budget_reservation_denied" in time.reason_codes


def test_exhausted_validation_reserve_blocks_model_admission() -> None:
    decision = _decision(_facts())
    ledger = ObjectiveCognitiveBudgetLedger(_budget())
    validation = ledger.reserve(
        idempotency_key="validation-work",
        question_id="validation-question",
        action_id="validation-action",
        purpose=BudgetPurpose.VALIDATION,
        requested=CognitiveCost(validation_time_ms=700),
    )
    assert validation.max_validation_time_ms == 700

    admission = _admit(decision, _question(), ledger)
    assert admission.disposition is EscalationDisposition.DENIED
    assert "validation_reserve_exhausted" in admission.reason_codes
    assert len(ledger.snapshot().reservations) == 1


def test_human_boundary_reserves_human_capacity_and_never_reserves_model_calls() -> None:
    decision = _decision(_facts(unresolved_authority=True))
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    ledger = ObjectiveCognitiveBudgetLedger(_budget())

    admission = _admit(
        decision,
        _question(route=ModelRoute.HUMAN_REVIEW_REQUIRED),
        ledger,
        expected_provider_spend_micros=0,
    )
    assert admission.admitted
    assert admission.requires_human_review
    assert admission.reservation is not None
    assert admission.reservation.purpose is BudgetPurpose.HUMAN
    assert admission.reservation.max_human_questions == 1
    assert admission.reservation.max_total_model_calls == 0
