"""IVP-012: provider-neutral next-repair model routing.

Acceptance coverage:

* Mechanical exact formatting/import/codemod/rename => deterministic
* Bounded localized exact work with a good counterexample => small
* Several-file synthesis without opaque critical edges => medium
* Ambiguous/broad/opaque/conflicting/overflow or failed-smaller => frontier
* available_models is provider-neutral inventory (never vendor preference)
* Unavailable required tier cannot downgrade
* Pending mandatory full suite => human review with verification-incomplete
* Unresolved authority, unmodeled high risk, scope crossing, proof/test
  conflict, unsafe context, non-reproducible environment => human review
  before any model route
* Output contains no provider identity
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    ModelRoute,
    ModelRouteDecision,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    MODEL_ROUTE_EVIDENCE,
    MODEL_ROUTE_PLANNER_INTERFACE,
    MODEL_ROUTE_PLANNER_SCHEMA,
    REASON_AMBIGUOUS_WORK,
    REASON_BROAD_DEPENDENCY_CONE,
    REASON_CONFLICTING_PROOF_REQUIREMENTS,
    REASON_CONTEXT_OVERFLOW,
    REASON_LOCALIZED_EXACT_COUNTEREXAMPLE,
    REASON_MANDATORY_FULL_SUITE_PENDING,
    REASON_MECHANICAL_EXACT_WORK,
    REASON_MULTI_FILE_SYNTHESIS,
    REASON_NONREPRODUCIBLE_ENVIRONMENT,
    REASON_OPAQUE_CRITICAL_DEPENDENCY,
    REASON_PROOF_TEST_CONFLICT,
    REASON_REQUIRED_TIER_UNAVAILABLE,
    REASON_SCOPE_CROSSING,
    REASON_SMALLER_ROUTE_FAILED,
    REASON_UNMODELED_HIGH_RISK,
    REASON_UNRESOLVED_AUTHORITY,
    REASON_UNSAFE_CONTEXT,
    REASON_VERIFICATION_INCOMPLETE,
    AnalysisKind,
    AvailableModelCapability,
    CapabilityLocality,
    CounterexampleQuality,
    ModelRouteError,
    ModelRouteFacts,
    ModelRoutePlanner,
    ModelRoutePolicy,
    ModelRoutePolicyError,
    PriorRepairAttempt,
    RiskLevel,
    choose_model_route,
    decide_model_route,
    default_inventory,
    policy_cid_for,
    select_required_route,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _policy(**kwargs: Any) -> ModelRoutePolicy:
    return ModelRoutePolicy(policy_cid=policy_cid_for("test-route-policy"), **kwargs)


def _facts(**kwargs: Any) -> ModelRouteFacts:
    base: dict[str, Any] = {
        "context_token_estimate": 2_048,
        "analysis_kind": AnalysisKind.LOCALIZED_EXACT,
        "opaque_dependency_count": 0,
        "risk_level": RiskLevel.LOW,
        "dependency_cone_size": 2,
        "changed_file_count": 1,
        "counterexample_quality": CounterexampleQuality.MINIMIZED,
        "exact_contract_available": True,
        "environment_reproducible": True,
    }
    base.update(kwargs)
    return ModelRouteFacts(**base)


def _inventory(**flags: bool) -> tuple[AvailableModelCapability, ...]:
    return default_inventory(**flags)


def _decide(
    facts: ModelRouteFacts,
    *,
    prior: list[Any] | None = None,
    inventory: tuple[AvailableModelCapability, ...] | None = None,
    policy: ModelRoutePolicy | None = None,
) -> ModelRouteDecision:
    return decide_model_route(
        facts,
        prior_attempts=prior or (),
        available_models=inventory if inventory is not None else _inventory(),
        policy=policy or _policy(),
    )


def _assert_no_provider_identity(decision: ModelRouteDecision) -> None:
    payload = decision.to_record()
    banned = {
        "provider",
        "provider_id",
        "vendor",
        "vendor_id",
        "model",
        "model_id",
        "model_name",
        "endpoint",
        "openai",
        "anthropic",
        "grok",
        "codex",
    }
    assert not banned & set(payload)
    blob = str(payload).lower()
    for token in (
        "openai",
        "anthropic",
        "grok",
        "gemini",
        "codex",
        "ollama",
        "huggingface",
        "vendor-specific",
    ):
        assert token not in blob
    parameters = inspect.signature(ModelRouteDecision).parameters
    assert not {"provider", "vendor", "model_id"} & set(parameters)


# ---------------------------------------------------------------------------
# Interface / inventory contracts
# ---------------------------------------------------------------------------


def test_planner_interface_and_evidence_constants() -> None:
    assert MODEL_ROUTE_PLANNER_INTERFACE == "ModelRoutePlanner@1"
    assert MODEL_ROUTE_PLANNER_SCHEMA.endswith("verification-model-route-planner@1")
    assert MODEL_ROUTE_EVIDENCE == "ivp/model-route@1"
    planner = ModelRoutePlanner(default_policy=_policy())
    assert planner.INTERFACE == MODEL_ROUTE_PLANNER_INTERFACE
    assert planner.EVIDENCE == MODEL_ROUTE_EVIDENCE


def test_available_models_rejects_provider_identity_fields() -> None:
    with pytest.raises(ModelRoutePolicyError, match="provider identity"):
        AvailableModelCapability.from_value(
            {
                "capability_tier": "small_local_model",
                "context_limit_tokens": 8_000,
                "locality": "local",
                "available": True,
                "provider": "vendor-x",
            }
        )
    with pytest.raises(ModelRoutePolicyError, match="provider identity"):
        AvailableModelCapability.from_value(
            {
                "capability_tier": "medium_model",
                "context_limit_tokens": 32_000,
                "locality": "any",
                "available": True,
                "model_id": "secret-model",
            }
        )


def test_available_models_is_capability_tier_context_locality_availability() -> None:
    item = AvailableModelCapability.from_value(
        {
            "capability_tier": "frontier_model",
            "context_limit_tokens": 128_000,
            "locality": "remote",
            "available": True,
        }
    )
    assert item.capability_tier is ModelRoute.FRONTIER_MODEL
    assert item.context_limit_tokens == 128_000
    assert item.locality is CapabilityLocality.REMOTE
    assert item.available is True
    assert set(item.to_dict()) == {
        "schema",
        "capability_tier",
        "context_limit_tokens",
        "locality",
        "available",
    }


def test_inventory_rejects_human_review_as_tier() -> None:
    with pytest.raises(ModelRoutePolicyError, match="human_review_required"):
        AvailableModelCapability(
            capability_tier=ModelRoute.HUMAN_REVIEW_REQUIRED,
            context_limit_tokens=1,
            locality=CapabilityLocality.ANY,
            available=True,
        )


# ---------------------------------------------------------------------------
# Mechanical => deterministic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        AnalysisKind.MECHANICAL_FORMATTING,
        AnalysisKind.MECHANICAL_IMPORT,
        AnalysisKind.MECHANICAL_CODEMOD,
        AnalysisKind.MECHANICAL_RENAME,
        "formatting",
        "import",
        "codemod",
        "rename",
    ],
)
def test_mechanical_exact_work_selects_deterministic(kind: Any) -> None:
    decision = _decide(
        _facts(
            analysis_kind=kind,
            counterexample_quality=CounterexampleQuality.NONE,
            changed_file_count=1,
        )
    )
    assert decision.route is ModelRoute.DETERMINISTIC_ONLY
    assert REASON_MECHANICAL_EXACT_WORK in decision.decisive_reason_codes
    assert not decision.requires_human_review
    _assert_no_provider_identity(decision)


# ---------------------------------------------------------------------------
# Localized + good counterexample => small
# ---------------------------------------------------------------------------


def test_bounded_localized_exact_with_good_counterexample_selects_small() -> None:
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.MINIMIZED,
            risk_level=RiskLevel.MODERATE,
            changed_file_count=1,
            dependency_cone_size=3,
        )
    )
    assert decision.route is ModelRoute.SMALL_LOCAL_MODEL
    assert REASON_LOCALIZED_EXACT_COUNTEREXAMPLE in decision.decisive_reason_codes
    assert "bounded_context" in decision.required_capabilities
    assert "local_execution" in decision.required_capabilities
    _assert_no_provider_identity(decision)


def test_localized_without_good_counterexample_does_not_select_small() -> None:
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.POOR,
            changed_file_count=1,
        )
    )
    assert decision.route is not ModelRoute.SMALL_LOCAL_MODEL
    assert decision.route in {
        ModelRoute.MEDIUM_MODEL,
        ModelRoute.FRONTIER_MODEL,
    }


# ---------------------------------------------------------------------------
# Multi-file synthesis => medium
# ---------------------------------------------------------------------------


def test_several_file_synthesis_without_opaque_critical_selects_medium() -> None:
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS,
            counterexample_quality=CounterexampleQuality.GOOD,
            changed_file_count=5,
            dependency_cone_size=20,
            opaque_dependency_count=0,
            risk_level=RiskLevel.MODERATE,
        )
    )
    assert decision.route is ModelRoute.MEDIUM_MODEL
    assert REASON_MULTI_FILE_SYNTHESIS in decision.decisive_reason_codes
    _assert_no_provider_identity(decision)


def test_multi_file_with_opaque_critical_escalates_to_frontier() -> None:
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS,
            changed_file_count=5,
            opaque_dependency_count=2,
        )
    )
    assert decision.route is ModelRoute.FRONTIER_MODEL
    assert REASON_OPAQUE_CRITICAL_DEPENDENCY in decision.decisive_reason_codes


# ---------------------------------------------------------------------------
# Frontier conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"analysis_kind": AnalysisKind.AMBIGUOUS}, REASON_AMBIGUOUS_WORK),
        (
            {
                "analysis_kind": AnalysisKind.MULTI_FILE_SYNTHESIS,
                "dependency_cone_size": 10_000,
            },
            REASON_BROAD_DEPENDENCY_CONE,
        ),
        (
            {
                "analysis_kind": AnalysisKind.OPAQUE,
                "opaque_dependency_count": 3,
            },
            REASON_OPAQUE_CRITICAL_DEPENDENCY,
        ),
        (
            {
                "analysis_kind": AnalysisKind.CONFLICTING,
                "failure_kind": "conflicting_proof",
            },
            REASON_CONFLICTING_PROOF_REQUIREMENTS,
        ),
        (
            {
                "analysis_kind": AnalysisKind.MULTI_FILE_SYNTHESIS,
                "context_token_estimate": 500_000,
            },
            REASON_CONTEXT_OVERFLOW,
        ),
    ],
)
def test_ambiguous_broad_opaque_conflicting_overflow_selects_frontier(
    kwargs: dict[str, Any], reason: str
) -> None:
    # Policy max_context_tokens triggers overflow classification; inventory
    # context limits remain large enough so availability does not rewrite the
    # required frontier tier into human review.
    policy = _policy(max_context_tokens=128_000)
    inventory = default_inventory(context_limit_tokens=1_000_000)
    decision = _decide(_facts(**kwargs), policy=policy, inventory=inventory)
    assert decision.route is ModelRoute.FRONTIER_MODEL
    assert reason in decision.decisive_reason_codes
    _assert_no_provider_identity(decision)


def test_failed_smaller_route_escalates() -> None:
    # Small failed on localized work without remaining small eligibility -> medium/frontier
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.MINIMIZED,
            changed_file_count=1,
        ),
        prior=[
            PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True),
        ],
    )
    assert decision.route in {ModelRoute.MEDIUM_MODEL, ModelRoute.FRONTIER_MODEL}
    assert REASON_SMALLER_ROUTE_FAILED in decision.decisive_reason_codes

    # Medium also failed => frontier
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS,
            changed_file_count=4,
            opaque_dependency_count=0,
        ),
        prior=[
            PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True),
            PriorRepairAttempt(route=ModelRoute.MEDIUM_MODEL, failed=True),
        ],
    )
    assert decision.route is ModelRoute.FRONTIER_MODEL
    assert REASON_SMALLER_ROUTE_FAILED in decision.decisive_reason_codes


def test_failed_frontier_selects_human_review() -> None:
    decision = _decide(
        _facts(analysis_kind=AnalysisKind.AMBIGUOUS),
        prior=[PriorRepairAttempt(route=ModelRoute.FRONTIER_MODEL, failed=True)],
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_review


# ---------------------------------------------------------------------------
# Human review precedence (before any model route)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"unresolved_authority": True}, REASON_UNRESOLVED_AUTHORITY),
        ({"unmodeled_high_risk": True}, REASON_UNMODELED_HIGH_RISK),
        ({"scope_crossing": True}, REASON_SCOPE_CROSSING),
        ({"proof_test_conflict": True}, REASON_PROOF_TEST_CONFLICT),
        ({"unsafe_context": True}, REASON_UNSAFE_CONTEXT),
        ({"environment_reproducible": False}, REASON_NONREPRODUCIBLE_ENVIRONMENT),
    ],
)
def test_human_review_gates_precede_model_routes(
    kwargs: dict[str, Any], reason: str
) -> None:
    # Even when mechanical work would otherwise be deterministic.
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.MECHANICAL_FORMATTING,
            **kwargs,
        )
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert decision.requires_human_review
    assert reason in decision.decisive_reason_codes
    # Would-be deterministic is not selected.
    assert decision.route is not ModelRoute.DETERMINISTIC_ONLY
    _assert_no_provider_identity(decision)


def test_pending_mandatory_full_suite_returns_verification_incomplete() -> None:
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            full_suite_required=True,
            full_suite_pending=True,
        )
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_VERIFICATION_INCOMPLETE in decision.decisive_reason_codes
    assert REASON_MANDATORY_FULL_SUITE_PENDING in decision.decisive_reason_codes
    _assert_no_provider_identity(decision)


def test_plan_human_review_required_is_honored() -> None:
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.MECHANICAL_RENAME,
            plan_human_review_required=True,
            plan_human_review_reason_codes=("scope_crossing",),
        )
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert "scope_crossing" in decision.decisive_reason_codes


# ---------------------------------------------------------------------------
# Availability: no downgrade
# ---------------------------------------------------------------------------


def test_unavailable_required_tier_does_not_downgrade() -> None:
    # Require frontier (ambiguous) but only small is available.
    inventory = default_inventory(
        include_deterministic=True, small=True, medium=False, frontier=False
    )
    decision = _decide(
        _facts(analysis_kind=AnalysisKind.AMBIGUOUS, context_token_estimate=1_000),
        inventory=inventory,
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_REQUIRED_TIER_UNAVAILABLE in decision.decisive_reason_codes
    # Must not silently fall back to small.
    assert decision.route is not ModelRoute.SMALL_LOCAL_MODEL


def test_unavailable_small_tier_does_not_downgrade_to_deterministic() -> None:
    inventory = default_inventory(
        include_deterministic=True, small=False, medium=True, frontier=True
    )
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.GOOD,
        ),
        inventory=inventory,
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_REQUIRED_TIER_UNAVAILABLE in decision.decisive_reason_codes


def test_tier_present_but_context_limit_too_small_is_unavailable() -> None:
    inventory = (
        AvailableModelCapability(
            capability_tier=ModelRoute.SMALL_LOCAL_MODEL,
            context_limit_tokens=100,
            locality=CapabilityLocality.LOCAL,
            available=True,
        ),
    )
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.MINIMIZED,
            context_token_estimate=2_048,
        ),
        inventory=inventory,
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_REQUIRED_TIER_UNAVAILABLE in decision.decisive_reason_codes


def test_available_required_tier_is_selected() -> None:
    inventory = (
        AvailableModelCapability(
            capability_tier=ModelRoute.SMALL_LOCAL_MODEL,
            context_limit_tokens=16_000,
            locality=CapabilityLocality.LOCAL,
            available=True,
        ),
        AvailableModelCapability(
            capability_tier=ModelRoute.FRONTIER_MODEL,
            context_limit_tokens=200_000,
            locality=CapabilityLocality.ANY,
            available=True,
        ),
    )
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.GOOD,
            context_token_estimate=4_000,
        ),
        inventory=inventory,
    )
    assert decision.route is ModelRoute.SMALL_LOCAL_MODEL


def test_unavailable_flag_on_inventory_row_blocks_tier() -> None:
    inventory = (
        AvailableModelCapability(
            capability_tier=ModelRoute.MEDIUM_MODEL,
            context_limit_tokens=64_000,
            locality=CapabilityLocality.ANY,
            available=False,
        ),
    )
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS,
            changed_file_count=4,
        ),
        inventory=inventory,
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_REQUIRED_TIER_UNAVAILABLE in decision.decisive_reason_codes


# ---------------------------------------------------------------------------
# choose_model_route public API / planner
# ---------------------------------------------------------------------------


def test_choose_model_route_from_mappings() -> None:
    policy = _policy()
    decision = choose_model_route(
        {
            "token_estimate": 1_500,
            "contracts": [{"id": "exact-1"}],
        },
        {
            "full_suite_required": False,
            "human_review_required": False,
        },
        prior_attempts=[],
        available_models=list(_inventory()),
        policy=policy,
        routing_hints={
            "analysis_kind": "mechanical_import",
            "risk_level": "low",
            "changed_file_count": 1,
            "opaque_dependency_count": 0,
        },
    )
    assert decision.route is ModelRoute.DETERMINISTIC_ONLY
    assert decision.policy_cid == policy.policy_cid
    assert decision.context_token_estimate == 1_500
    _assert_no_provider_identity(decision)


def test_choose_model_route_full_suite_from_plan_mapping() -> None:
    decision = choose_model_route(
        {"token_estimate": 800},
        {
            "full_suite_required": True,
            "full_suite_pending": True,
            "human_review_required": False,
        },
        prior_attempts=[],
        available_models=list(_inventory()),
        policy=_policy(),
        routing_hints={"analysis_kind": "localized_exact"},
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_VERIFICATION_INCOMPLETE in decision.decisive_reason_codes


def test_model_route_planner_choose_and_decide() -> None:
    planner = ModelRoutePlanner(default_policy=_policy())
    decision = planner.decide(
        _facts(analysis_kind=AnalysisKind.MECHANICAL_CODEMOD),
        available_models=_inventory(),
    )
    assert decision.route is ModelRoute.DETERMINISTIC_ONLY

    decision = planner.choose(
        {"token_estimate": 2_000},
        None,
        prior_attempts=[],
        available_models=_inventory(),
        routing_hints={
            "analysis_kind": "multi_file_synthesis",
            "changed_file_count": 6,
            "dependency_cone_size": 15,
            "opaque_dependency_count": 0,
            "risk_level": "moderate",
            "counterexample_quality": "good",
        },
    )
    assert decision.route is ModelRoute.MEDIUM_MODEL


def test_planner_requires_policy_when_default_missing() -> None:
    planner = ModelRoutePlanner()
    with pytest.raises(ModelRoutePolicyError, match="policy is required"):
        planner.decide(_facts())


# ---------------------------------------------------------------------------
# Decision envelope invariants
# ---------------------------------------------------------------------------


def test_decision_round_trip_and_considered_routes() -> None:
    decision = _decide(_facts(analysis_kind=AnalysisKind.LOCALIZED_EXACT))
    restored = ModelRouteDecision.from_dict(decision.to_record())
    assert restored == decision
    assert decision.route in decision.considered_routes
    assert len(decision.considered_routes) == len(set(decision.considered_routes))
    assert decision.decisive_reason_codes
    assert decision.context_token_estimate >= 0


def test_select_required_route_is_pure_and_table_driven() -> None:
    facts = _facts(analysis_kind=AnalysisKind.MECHANICAL_FORMATTING)
    route, reasons = select_required_route(facts, (), _policy())
    assert route is ModelRoute.DETERMINISTIC_ONLY
    assert REASON_MECHANICAL_EXACT_WORK in reasons

    facts = _facts(unresolved_authority=True)
    route, reasons = select_required_route(facts, (), _policy())
    assert route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_UNRESOLVED_AUTHORITY in reasons


def test_routing_hints_reject_provider_identity() -> None:
    with pytest.raises(ModelRoutePolicyError, match="provider identity"):
        choose_model_route(
            {"token_estimate": 10},
            None,
            [],
            _inventory(),
            _policy(),
            routing_hints={"provider": "nope", "analysis_kind": "localized_exact"},
        )


def test_high_risk_ambiguous_requires_human_review() -> None:
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.OPAQUE,
            risk_level=RiskLevel.HIGH,
            opaque_dependency_count=1,
        )
    )
    assert decision.route is ModelRoute.HUMAN_REVIEW_REQUIRED


def test_deterministic_does_not_require_model_inventory() -> None:
    # Empty inventory still allows deterministic_only.
    decision = _decide(
        _facts(analysis_kind=AnalysisKind.MECHANICAL_RENAME),
        inventory=(),
    )
    assert decision.route is ModelRoute.DETERMINISTIC_ONLY


def test_output_never_embeds_vendor_in_reason_codes_or_capabilities() -> None:
    decision = _decide(_facts(analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS, changed_file_count=4))
    for code in decision.decisive_reason_codes:
        assert "provider" not in code
        assert "vendor" not in code
        assert "openai" not in code
        assert "model_id" not in code
    for cap in decision.required_capabilities:
        assert "provider" not in cap
        assert "vendor" not in cap


def test_decide_rejects_non_object_facts() -> None:
    with pytest.raises(ModelRouteError):
        decide_model_route("not-a-mapping", policy=_policy())  # type: ignore[arg-type]


def test_prior_attempt_from_mapping() -> None:
    attempt = PriorRepairAttempt.from_value(
        {"route": "small_local_model", "failed": True, "reason_codes": ["repair_missed"]}
    )
    assert attempt.route is ModelRoute.SMALL_LOCAL_MODEL
    assert attempt.failed is True
    decision = _decide(
        _facts(
            analysis_kind=AnalysisKind.LOCALIZED_EXACT,
            counterexample_quality=CounterexampleQuality.MINIMIZED,
        ),
        prior=[attempt],
    )
    assert decision.route is not ModelRoute.SMALL_LOCAL_MODEL


def test_policy_from_cid_string() -> None:
    cid = policy_cid_for("string-policy")
    decision = decide_model_route(
        _facts(analysis_kind=AnalysisKind.MECHANICAL_IMPORT),
        available_models=_inventory(),
        policy=cid,
    )
    assert decision.policy_cid == cid
    assert decision.route is ModelRoute.DETERMINISTIC_ONLY
