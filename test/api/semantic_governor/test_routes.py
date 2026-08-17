"""Tests for SCG-030 model-route calibration (separate from context sufficiency).

Acceptance criteria enforced here:

* Context omission and reasoning failure are separate counters.
* Unavailable required tier never downgrades (escalates to human).
* Changes are proposals only (no production route mutation).
* Capability tier only — provider identity is rejected.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
    RouteTier,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.routes import (
    DEFAULT_MIN_USES_FOR_PROPOSAL,
    PROPOSE_ROUTE_THRESHOLD_CHANGE_INTERFACE,
    ROUTE_CALIBRATION_STATE_SCHEMA,
    SCG_ROUTE_CALIBRATION_EVIDENCE,
    UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE,
    ModelRouteCalibrationState,
    RouteAvailabilityDisposition,
    RouteCalibrationDisposition,
    RouteCalibrationError,
    RouteFailureKind,
    RouteRunObservation,
    RouteThresholdDisposition,
    RouteThresholdParameter,
    RouteThresholdPolicy,
    RouteThresholdProposal,
    RouteTierMetrics,
    default_route_threshold_policy,
    observation_from_receipt_fields,
    propose_route_threshold_change,
    propose_route_threshold_change_interface_id,
    resolve_route_availability,
    route_calibration_evidence_id,
    route_failure_kinds,
    route_threshold_parameters,
    route_tiers,
    update_model_route_calibration,
    update_model_route_calibration_interface_id,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/routes.py"
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _obs(
    observation_id: str = "obs_0001",
    *,
    route_tier: str = RouteTier.MEDIUM.value,
    accepted: bool = True,
    **overrides: Any,
) -> RouteRunObservation:
    fields: dict[str, Any] = {
        "observation_id": observation_id,
        "route_tier": route_tier,
        "accepted": accepted,
        "retried": False,
        "expansion_used": False,
        "verification_passed": True,
        "context_omission_failure": False,
        "reasoning_failure": False,
        "required_route_tier": route_tier,
        "required_tier_available": True,
        "cost_micros": 100,
        "latency_ms": 10,
        "receipt_cid": _cid(f"receipt-{observation_id}"),
        "decision_cid": _cid(f"decision-{observation_id}"),
        "failure_kind": RouteFailureKind.NONE.value,
        "reason_codes": (),
        "simulated": False,
        "metadata": {},
    }
    fields.update(overrides)
    return RouteRunObservation(**fields)


def _state(**overrides: Any) -> ModelRouteCalibrationState:
    base = ModelRouteCalibrationState.empty(state_id="route_calibration_test")
    if not overrides:
        return base
    payload = base.to_dict()
    payload.update(overrides)
    # Drop derived cid so from_dict recomputes.
    payload.pop("state_cid", None)
    return ModelRouteCalibrationState.from_dict(payload)


def _seed_tier(
    *,
    route_tier: str = RouteTier.MEDIUM.value,
    total_uses: int = 10,
    accepted_count: int = 4,
    context_omission_failure_count: int = 4,
    reasoning_failure_count: int = 3,
    retry_count: int = 5,
    unavailable_required_tier_count: int = 0,
) -> ModelRouteCalibrationState:
    """Build a state with enough uses to trigger proposals (compact recipe)."""

    state = ModelRouteCalibrationState.empty()
    metrics = RouteTierMetrics(
        route_tier=route_tier,
        total_uses=total_uses,
        accepted_count=accepted_count,
        retry_count=retry_count,
        expansion_count=2,
        verification_pass_count=accepted_count,
        verification_fail_count=total_uses - accepted_count,
        context_omission_failure_count=context_omission_failure_count,
        reasoning_failure_count=reasoning_failure_count,
        unavailable_required_tier_count=unavailable_required_tier_count,
        cost_micros_total=total_uses * 100,
        latency_ms_total=total_uses * 10,
        source_receipt_cids=tuple(_cid(f"seed-{route_tier}-{i}") for i in range(3)),
    )
    tier_map = {tier: state.tier_metrics[tier] for tier in route_tiers()}
    tier_map[route_tier] = metrics
    return ModelRouteCalibrationState(
        header=state.header,
        state_id=state.state_id,
        partition=state.partition,
        revision=1,
        tier_metrics=tier_map,
        applied_observation_cids=tuple(_cid(f"applied-{i}") for i in range(total_uses)),
        notes="seeded",
        metadata={"track": "route-calibration"},
    )


# ---------------------------------------------------------------------------
# Module surface / evidence / import safety
# ---------------------------------------------------------------------------


def test_evidence_and_interfaces_are_stable() -> None:
    assert SCG_ROUTE_CALIBRATION_EVIDENCE == "scg/route-calibration@1"
    assert route_calibration_evidence_id() == SCG_ROUTE_CALIBRATION_EVIDENCE
    assert (
        update_model_route_calibration_interface_id()
        == UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE
    )
    assert (
        propose_route_threshold_change_interface_id()
        == PROPOSE_ROUTE_THRESHOLD_CHANGE_INTERFACE
    )
    assert UPDATE_MODEL_ROUTE_CALIBRATION_INTERFACE == (
        "update_model_route_calibration@1"
    )
    assert PROPOSE_ROUTE_THRESHOLD_CHANGE_INTERFACE == (
        "propose_route_threshold_change@1"
    )


def test_closed_vocabularies() -> None:
    assert route_tiers() == (
        "deterministic",
        "small",
        "medium",
        "frontier",
        "human",
    )
    kinds = route_failure_kinds()
    assert "context_omission" in kinds
    assert "reasoning_failure" in kinds
    assert "context_omission" != "reasoning_failure"
    params = route_threshold_parameters()
    assert RouteThresholdParameter.MAX_CONTEXT_OMISSION_RATE_BP.value in params
    assert RouteThresholdParameter.MAX_REASONING_FAILURE_RATE_BP.value in params


def test_module_import_performs_no_io() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "open",
        "urlopen",
        "urlretrieve",
        "system",
        "Popen",
        "run",
        "check_output",
        "check_call",
        "connect",
        "create_connection",
        "socket",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in forbidden_calls:
                # Allow only inside function bodies that are not module-level.
                # Module-level calls are the real concern for import-time I/O.
                pass
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            # No bare call statements at module level beyond imports handled by AST
            # structure; enforce no open/socket at any Call with those names at
            # module body.
            pass

    # Strict: no Call to forbidden names appears in Module body statements.
    module_calls: list[str] = []
    for stmt in tree.body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            # Skip nested function/class defs.
            continue
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id in forbidden_calls:
                    module_calls.append(func.id)
                if isinstance(func, ast.Attribute) and func.attr in forbidden_calls:
                    module_calls.append(func.attr)
    assert module_calls == []


def test_docstring_declares_no_io_and_separation() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "Importing this module performs no I/O" in source
    assert "separately from context sufficiency" in source.lower() or (
        "separate counters" in source.lower()
    )
    assert "proposals only" in source.lower()


# ---------------------------------------------------------------------------
# Round-trip identities
# ---------------------------------------------------------------------------


def test_empty_state_round_trip() -> None:
    state = ModelRouteCalibrationState.empty()
    restored = ModelRouteCalibrationState.from_dict(state.to_dict())
    assert restored.state_cid == state.state_cid
    assert restored.revision == 0
    assert set(restored.tier_metrics) == set(route_tiers())
    for tier in route_tiers():
        assert restored.metrics_for(tier).total_uses == 0


def test_observation_round_trip_and_identity() -> None:
    obs = _obs("obs_rt", context_omission_failure=True, accepted=False)
    restored = RouteRunObservation.from_dict(obs.to_dict())
    assert restored.observation_cid == obs.observation_cid
    assert restored.context_omission_failure is True
    assert restored.reasoning_failure is False
    assert restored.failure_kind == RouteFailureKind.CONTEXT_OMISSION.value


def test_observation_from_receipt_fields_does_not_auto_blame_omission() -> None:
    """Omission evidence CID alone must not auto-increment omission counter."""

    obs = observation_from_receipt_fields(
        observation_id="obs_receipt",
        route_tier=RouteTier.SMALL.value,
        accepted=False,
        receipt_cid=_cid("run-receipt"),
        decision_cid=_cid("run-decision"),
        wall_time_ms=42,
        spend_micros=900,
        expansion_plan_cid=_cid("expansion-plan"),
        omission_evidence_cid=_cid("omission-evidence"),
        context_omission_failure=False,
        reasoning_failure=True,
    )
    assert obs.expansion_used is True
    assert obs.context_omission_failure is False
    assert obs.reasoning_failure is True
    assert obs.failure_kind == RouteFailureKind.REASONING_FAILURE.value
    assert obs.metadata["omission_evidence_cid"] == _cid("omission-evidence")


# ---------------------------------------------------------------------------
# Acceptance: separate omission and reasoning counters
# ---------------------------------------------------------------------------


def test_omission_and_reasoning_are_separate_counters() -> None:
    state = ModelRouteCalibrationState.empty()
    observations = [
        _obs(
            "obs_omit",
            accepted=False,
            context_omission_failure=True,
            reasoning_failure=False,
            verification_passed=False,
        ),
        _obs(
            "obs_reason",
            accepted=False,
            context_omission_failure=False,
            reasoning_failure=True,
            verification_passed=False,
        ),
        _obs(
            "obs_ok",
            accepted=True,
            context_omission_failure=False,
            reasoning_failure=False,
            verification_passed=True,
        ),
    ]
    result = update_model_route_calibration(state, observations)
    assert result.disposition == RouteCalibrationDisposition.APPLIED.value
    metrics = result.state.metrics_for(RouteTier.MEDIUM.value)
    assert metrics.total_uses == 3
    assert metrics.context_omission_failure_count == 1
    assert metrics.reasoning_failure_count == 1
    # Counters are independent fields — not a single merged failure total.
    assert metrics.context_omission_failure_count != metrics.total_uses
    assert metrics.reasoning_failure_count != metrics.total_uses
    assert (
        metrics.context_omission_failure_count + metrics.reasoning_failure_count
        == 2
    )
    assert metrics.context_omission_rate_bp != metrics.reasoning_failure_rate_bp or (
        metrics.context_omission_failure_count == metrics.reasoning_failure_count
    )


def test_both_flags_increment_both_counters_without_merge() -> None:
    """Even if both flags are set, counters stay separate fields."""

    state = ModelRouteCalibrationState.empty()
    obs = _obs(
        "obs_both",
        accepted=False,
        context_omission_failure=True,
        reasoning_failure=True,
        verification_passed=False,
    )
    result = update_model_route_calibration(state, [obs])
    metrics = result.state.metrics_for(RouteTier.MEDIUM.value)
    assert metrics.context_omission_failure_count == 1
    assert metrics.reasoning_failure_count == 1
    payload = metrics.identity_payload()
    assert "context_omission_failure_count" in payload
    assert "reasoning_failure_count" in payload
    assert payload["context_omission_failure_count"] == 1
    assert payload["reasoning_failure_count"] == 1


def test_tracks_accepted_retries_expansion_verification_cost_latency() -> None:
    state = ModelRouteCalibrationState.empty()
    observations = [
        _obs(
            "obs_a",
            accepted=True,
            retried=True,
            expansion_used=True,
            verification_passed=True,
            cost_micros=500,
            latency_ms=25,
        ),
        _obs(
            "obs_b",
            accepted=False,
            retried=False,
            expansion_used=False,
            verification_passed=False,
            cost_micros=200,
            latency_ms=15,
            route_tier=RouteTier.FRONTIER.value,
            required_route_tier=RouteTier.FRONTIER.value,
        ),
    ]
    result = update_model_route_calibration(state, observations)
    medium = result.state.metrics_for(RouteTier.MEDIUM.value)
    frontier = result.state.metrics_for(RouteTier.FRONTIER.value)
    assert medium.accepted_count == 1
    assert medium.retry_count == 1
    assert medium.expansion_count == 1
    assert medium.verification_pass_count == 1
    assert medium.cost_micros_total == 500
    assert medium.latency_ms_total == 25
    assert frontier.total_uses == 1
    assert frontier.accepted_count == 0
    assert frontier.verification_fail_count == 1
    assert frontier.cost_micros_total == 200


def test_all_route_tiers_are_tracked_independently() -> None:
    state = ModelRouteCalibrationState.empty()
    observations = [
        _obs(f"obs_{tier}", route_tier=tier, required_route_tier=tier, accepted=True)
        for tier in route_tiers()
    ]
    result = update_model_route_calibration(state, observations)
    for tier in route_tiers():
        assert result.state.metrics_for(tier).total_uses == 1
        assert result.state.metrics_for(tier).accepted_count == 1


# ---------------------------------------------------------------------------
# Acceptance: unavailable required tier never downgrades
# ---------------------------------------------------------------------------


def test_unavailable_required_tier_escalates_to_human_never_downgrades() -> None:
    decision = resolve_route_availability(
        RouteTier.FRONTIER.value,
        available_route_tiers=(RouteTier.SMALL.value, RouteTier.MEDIUM.value),
    )
    assert decision.downgraded is False
    assert decision.resolved_route_tier == RouteTier.HUMAN.value
    assert (
        decision.disposition
        == RouteAvailabilityDisposition.UNAVAILABLE_ESCALATED_TO_HUMAN.value
    )
    assert "required_tier_unavailable" in decision.reason_codes
    # Must not pick a weaker model tier.
    assert decision.resolved_route_tier not in {
        RouteTier.DETERMINISTIC.value,
        RouteTier.SMALL.value,
        RouteTier.MEDIUM.value,
    }


def test_available_required_tier_is_selected() -> None:
    decision = resolve_route_availability(
        RouteTier.MEDIUM.value,
        available_route_tiers=(
            RouteTier.SMALL.value,
            RouteTier.MEDIUM.value,
            RouteTier.FRONTIER.value,
        ),
    )
    assert decision.resolved_route_tier == RouteTier.MEDIUM.value
    assert decision.disposition == RouteAvailabilityDisposition.AVAILABLE.value
    assert decision.downgraded is False


def test_deterministic_always_available_without_inventory() -> None:
    decision = resolve_route_availability(
        RouteTier.DETERMINISTIC.value,
        available_route_tiers=(),
    )
    assert decision.resolved_route_tier == RouteTier.DETERMINISTIC.value
    assert (
        decision.disposition
        == RouteAvailabilityDisposition.DETERMINISTIC_ALWAYS_AVAILABLE.value
    )


def test_observation_recording_downgrade_is_rejected() -> None:
    state = ModelRouteCalibrationState.empty()
    # Required frontier unavailable, but observation claims small (downgrade).
    bad = _obs(
        "obs_downgrade",
        route_tier=RouteTier.SMALL.value,
        required_route_tier=RouteTier.FRONTIER.value,
        required_tier_available=False,
        accepted=False,
    )
    with pytest.raises(RouteCalibrationError, match="downgrade"):
        update_model_route_calibration(state, [bad])


def test_unavailable_observation_may_record_human_escalation() -> None:
    state = ModelRouteCalibrationState.empty()
    obs = _obs(
        "obs_human_escalation",
        route_tier=RouteTier.HUMAN.value,
        required_route_tier=RouteTier.FRONTIER.value,
        required_tier_available=False,
        accepted=False,
        failure_kind=RouteFailureKind.UNAVAILABLE_TIER.value,
    )
    result = update_model_route_calibration(state, [obs])
    human = result.state.metrics_for(RouteTier.HUMAN.value)
    assert human.total_uses == 1
    assert human.unavailable_required_tier_count == 1


def test_availability_decision_forbids_downgraded_flag() -> None:
    with pytest.raises(RouteCalibrationError, match="never downgrade|downgrade"):
        from ipfs_accelerate_py.agent_supervisor.semantic_governor.routes import (
            RouteAvailabilityDecision,
        )

        RouteAvailabilityDecision(
            required_route_tier=RouteTier.FRONTIER.value,
            resolved_route_tier=RouteTier.SMALL.value,
            disposition=RouteAvailabilityDisposition.AVAILABLE.value,
            available_route_tiers=(RouteTier.SMALL.value,),
            reason_codes=("bad",),
            downgraded=True,
        )


# ---------------------------------------------------------------------------
# Acceptance: changes are proposals only
# ---------------------------------------------------------------------------


def test_propose_route_threshold_change_is_proposal_only() -> None:
    state = _seed_tier()
    before_cid = state.state_cid
    result = propose_route_threshold_change(state)
    assert result.mutates_production is False
    assert result.disposition == RouteThresholdDisposition.PROPOSED.value
    assert len(result.proposals) >= 1
    # State identity unchanged — proposals do not apply thresholds.
    assert state.state_cid == before_cid
    for proposal in result.proposals:
        assert proposal.is_proposal_only is True
        assert proposal.mutates_production is False
        assert proposal.high_risk_assurance_reduced is False


def test_proposals_keep_omission_and_reasoning_separate() -> None:
    state = _seed_tier(
        context_omission_failure_count=5,
        reasoning_failure_count=4,
        accepted_count=3,
        total_uses=10,
    )
    result = propose_route_threshold_change(state)
    params = {p.parameter for p in result.proposals}
    assert RouteThresholdParameter.MAX_CONTEXT_OMISSION_RATE_BP.value in params
    assert RouteThresholdParameter.MAX_REASONING_FAILURE_RATE_BP.value in params
    omission_props = [
        p
        for p in result.proposals
        if p.parameter
        == RouteThresholdParameter.MAX_CONTEXT_OMISSION_RATE_BP.value
    ]
    reasoning_props = [
        p
        for p in result.proposals
        if p.parameter
        == RouteThresholdParameter.MAX_REASONING_FAILURE_RATE_BP.value
    ]
    assert omission_props and reasoning_props
    assert omission_props[0].proposal_id != reasoning_props[0].proposal_id
    assert "omission_counter_separate" in omission_props[0].reason_codes
    assert "reasoning_counter_separate" in reasoning_props[0].reason_codes


def test_proposal_rejects_production_mutation_flag() -> None:
    with pytest.raises(RouteCalibrationError, match="mutat"):
        RouteThresholdProposal(
            proposal_id="bad_mut",
            route_tier=RouteTier.MEDIUM.value,
            parameter=RouteThresholdParameter.MIN_ACCEPTED_RATE_BP.value,
            current_value="7000",
            proposed_value="8000",
            reason_codes=("test",),
            mutates_production=True,
            is_proposal_only=True,
        )


def test_proposal_requires_proposal_only_flag() -> None:
    with pytest.raises(RouteCalibrationError, match="proposals only"):
        RouteThresholdProposal(
            proposal_id="bad_flag",
            route_tier=RouteTier.MEDIUM.value,
            parameter=RouteThresholdParameter.MIN_ACCEPTED_RATE_BP.value,
            current_value="7000",
            proposed_value="8000",
            reason_codes=("test",),
            mutates_production=False,
            is_proposal_only=False,
        )


def test_high_risk_never_auto_lowers_min_route_tier() -> None:
    state = _seed_tier(
        route_tier=RouteTier.MEDIUM.value,
        context_omission_failure_count=8,
        reasoning_failure_count=8,
        accepted_count=1,
        total_uses=10,
    )
    policy = RouteThresholdPolicy(
        policy_id="high_risk_policy",
        high_risk_min_route_tier=RouteTier.FRONTIER.value,
        min_uses_for_proposal=1,
        max_context_omission_rate_bp=500,
        max_reasoning_failure_rate_bp=500,
    )
    result = propose_route_threshold_change(state, policy, high_risk=True)
    for proposal in result.proposals:
        if proposal.parameter == RouteThresholdParameter.MIN_ROUTE_TIER.value:
            # Proposed tier must not fall below high-risk floor.
            rank = {
                "deterministic": 0,
                "small": 1,
                "medium": 2,
                "frontier": 3,
                "human": 4,
            }
            assert rank[proposal.proposed_value] >= rank[RouteTier.FRONTIER.value]


def test_disabled_escalate_on_unavailable_proposes_restore() -> None:
    state = ModelRouteCalibrationState.empty()
    policy = RouteThresholdPolicy(
        policy_id="bad_policy",
        escalate_on_unavailable=False,
        min_uses_for_proposal=DEFAULT_MIN_USES_FOR_PROPOSAL,
    )
    result = propose_route_threshold_change(state, policy)
    assert any(
        p.parameter == RouteThresholdParameter.ESCALATE_ON_UNAVAILABLE.value
        and p.proposed_value == "true"
        for p in result.proposals
    )
    assert result.mutates_production is False


def test_no_change_when_metrics_healthy() -> None:
    state = _seed_tier(
        total_uses=10,
        accepted_count=10,
        context_omission_failure_count=0,
        reasoning_failure_count=0,
        retry_count=0,
        unavailable_required_tier_count=0,
    )
    policy = default_route_threshold_policy()
    result = propose_route_threshold_change(state, policy)
    assert result.disposition == RouteThresholdDisposition.NO_CHANGE.value
    assert result.proposals == ()
    assert result.mutates_production is False


# ---------------------------------------------------------------------------
# Idempotency / simulated exclusion / mapping inputs
# ---------------------------------------------------------------------------


def test_simulated_observations_excluded_from_live_quality() -> None:
    state = ModelRouteCalibrationState.empty()
    obs = _obs("obs_sim", simulated=True, accepted=True)
    result = update_model_route_calibration(state, [obs])
    assert result.disposition == RouteCalibrationDisposition.SKIPPED_SIMULATED.value
    # Live counters unchanged.
    assert result.state.metrics_for(RouteTier.MEDIUM.value).total_uses == 0
    assert obs.observation_cid in result.skipped_observation_cids
    assert "skipped_simulated" in result.reason_codes


def test_idempotent_observation_cids() -> None:
    state = ModelRouteCalibrationState.empty()
    obs = _obs("obs_idem")
    first = update_model_route_calibration(state, [obs])
    second = update_model_route_calibration(first.state, [obs])
    assert first.disposition == RouteCalibrationDisposition.APPLIED.value
    assert second.disposition == RouteCalibrationDisposition.SKIPPED_IDEMPOTENT.value
    assert second.state.metrics_for(RouteTier.MEDIUM.value).total_uses == 1


def test_update_accepts_mapping_inputs() -> None:
    state = ModelRouteCalibrationState.empty()
    obs = _obs("obs_map", accepted=True, expansion_used=True)
    result = update_model_route_calibration(state.to_dict(), [obs.to_dict()])
    assert result.disposition == RouteCalibrationDisposition.APPLIED.value
    assert result.state.metrics_for(RouteTier.MEDIUM.value).expansion_count == 1


def test_empty_observations_skipped() -> None:
    state = ModelRouteCalibrationState.empty()
    result = update_model_route_calibration(state, [])
    assert result.disposition == RouteCalibrationDisposition.SKIPPED_EMPTY.value
    assert result.state.state_cid == state.state_cid


# ---------------------------------------------------------------------------
# Provider identity / context-sufficiency separation fail-closed
# ---------------------------------------------------------------------------


def test_provider_identity_rejected_on_observation_metadata() -> None:
    with pytest.raises(RouteCalibrationError, match="provider identity"):
        _obs("obs_provider", metadata={"provider_id": "openai.gpt-test"})


def test_provider_token_rejected_in_reason_codes() -> None:
    with pytest.raises(RouteCalibrationError, match="provider identity"):
        _obs("obs_vendor", reason_codes=("use_openai_backend",))


def test_context_sufficiency_mutation_rejected() -> None:
    with pytest.raises(RouteCalibrationError, match="context sufficiency"):
        _obs(
            "obs_sufficiency",
            metadata={"sufficiency_state": "sufficient"},
        )


def test_route_calibration_does_not_export_context_sufficiency_api() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    # Must not implement evaluate_context_sufficiency — that is separate.
    assert "def evaluate_context_sufficiency" not in source
    assert "update_model_route_calibration" in source
    assert "propose_route_threshold_change" in source


# ---------------------------------------------------------------------------
# Schema pins
# ---------------------------------------------------------------------------


def test_state_schema_pin() -> None:
    state = ModelRouteCalibrationState.empty()
    assert state.to_dict()["schema"] == ROUTE_CALIBRATION_STATE_SCHEMA
    assert state.to_dict()["interface_id"] == "ModelRouteCalibrationState@1"


def test_proposal_result_identity_stable() -> None:
    state = _seed_tier()
    a = propose_route_threshold_change(state)
    b = propose_route_threshold_change(state)
    assert a.result_cid == b.result_cid
    assert a.mutates_production is False
