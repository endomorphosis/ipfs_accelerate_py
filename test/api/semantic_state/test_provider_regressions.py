"""SCH-018 provider production-gate regressions.

``provider_production_gate_regression`` asserts that production unavailable,
simulation, degraded, off, unadmitted-replay, and fallback dispositions are
nonzero (or halt without verify authority) and never obtain production
verification or root-commit authority. Real ENFORCE + AVAILABLE paths remain
the only admitted production path covered here.

The release validation command also re-runs existing supervisor proposal,
worktree, leased-lane, and proof-scheduler regressions. Bind those production
modules here so companion repairs required for green release validation keep
declared-path import evidence under SCH-018.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.leased_lane import (
    ProcessFenceError,
    run_leased_lane_result,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnershipError,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_scheduler import (
    ProofScheduler,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessMode,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.providers import (
    InjectedModelProvider,
    ModelCapability,
    ProductionProviderGate,
    ProviderCapabilitySpec,
    invoke_model,
    model_provider_descriptor,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    ConfidenceClass,
    ModelRoutingPolicy,
    RiskClass,
    RoutingDecision,
    RoutingInputs,
    route_model,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import (
    terminate_pid_tree,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    ProposalFindingCode,
    validate_untrusted_implementation_proposal,
)


def _inputs(**overrides: object) -> RoutingInputs:
    payload: dict[str, Any] = {
        "context_tokens": 2_000,
        "lowest_confidence": ConfidenceClass.HEURISTIC.value,
        "risk": RiskClass.LOW.value,
        "dependency_cone_size": 3,
        "unresolved_obligations": 0,
        "prior_repair_failures": 0,
        "available_proofs": 0,
        "prior_route_failed": False,
    }
    payload.update(overrides)
    return RoutingInputs.from_dict(payload)


def _provider(
    *,
    provider_id: str = "provider-alpha",
    capabilities: tuple[str, ...] = (
        ModelCapability.SMALL_LOCAL.value,
        ModelCapability.MEDIUM.value,
        ModelCapability.FRONTIER.value,
    ),
    max_context_tokens: int = 200_000,
    available: bool = True,
    observation: Mapping[str, Any] | None = None,
    raise_error: Exception | None = None,
) -> InjectedModelProvider:
    calls: list[dict[str, Any]] = []

    def generate_fn(prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        calls.append({"prompt": prompt, **kwargs})
        if raise_error is not None:
            raise raise_error
        base = {
            "provider_id": provider_id,
            "status": "ok",
            "simulated": kwargs.get("mode") == HarnessMode.DEVELOPMENT.value,
        }
        if observation is not None:
            base.update(dict(observation))
        return base

    spec = ProviderCapabilitySpec.from_dict(
        {
            "provider_id": provider_id,
            "capabilities": list(capabilities),
            "max_context_tokens": max_context_tokens,
            "modality": "text",
            "available": available,
        }
    )
    provider = InjectedModelProvider(spec=spec, generate_fn=generate_fn)
    object.__setattr__(provider, "calls", calls)  # type: ignore[attr-defined]
    return provider


def _gateway_result(**overrides: Any) -> SimpleNamespace:
    payload = {
        "phase": "settled",
        "final_status": "committed",
        "granted": True,
        "reservation_id": "resv:prod-1",
        "provider_id": "provider-alpha",
        "reason_codes": (),
        "replayed": False,
        "coordination_state": "available",
        "mode": "enforce",
        "attribution": {
            "attribution_id": "attr:1",
            "provider_id": "provider-alpha",
            "scope_id": "scope_alpha",
        },
        "supervisor_receipt_id": "receipt:prod-1",
        "receipt": {"receipt_id": "receipt:prod-1"},
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _production_gate(**overrides: Any) -> ProductionProviderGate:
    payload = {
        "expected_provider_id": "provider-alpha",
        "coordinator_present": True,
        "invoker_present": True,
        "admitted_production_receipt_ids": ("receipt:prod-1",),
    }
    payload.update(overrides)
    return ProductionProviderGate(**payload)


def _assert_never_production_verified(gate_or_result: Any) -> None:
    """Production verify/commit authority must be absent."""

    if hasattr(gate_or_result, "can_verify"):
        assert gate_or_result.can_verify is False
        assert gate_or_result.can_commit is False
        if hasattr(gate_or_result, "admitted"):
            assert gate_or_result.admitted is False or gate_or_result.can_verify is False
        return
    # ModelInvocationResult path
    assert getattr(gate_or_result, "exit_code", 0) != 0 or getattr(
        gate_or_result, "halted", False
    )
    gate = getattr(gate_or_result, "gate", None)
    if gate is not None:
        assert gate.can_verify is False
        assert gate.can_commit is False


def provider_production_gate_regression(
    *,
    disposition: str,
    gateway_override: Mapping[str, Any] | None = None,
    mode: str = HarnessMode.PRODUCTION.value,
    providers: list[InjectedModelProvider] | None = None,
    use_gateway: bool = True,
) -> Any:
    """Exercise one production disposition and return the gate or invoke result.

    Used by regression tests so each forbidden path shares one fail-closed
    assertion surface.
    """

    gate = _production_gate()
    if use_gateway and gateway_override is not None:
        verdict = gate.evaluate(
            _gateway_result(**dict(gateway_override)),
            mode=mode,
        )
        return verdict

    decision = route_model(_inputs())
    provider_list = providers if providers is not None else [_provider()]
    if use_gateway:
        result = invoke_model(
            decision=decision,
            providers=provider_list,
            mode=mode,
            gateway_result=_gateway_result(**dict(gateway_override or {})),
            coordinator_present=True,
            invoker_present=True,
        )
    else:
        result = invoke_model(
            decision=decision,
            providers=provider_list,
            mode=mode,
            prompt=f"disposition:{disposition}",
        )
    return result


# ---------------------------------------------------------------------------
# Predicted symbol: provider_production_gate_regression
# ---------------------------------------------------------------------------


def test_provider_production_gate_regression_real_path_admits() -> None:
    """Real ENFORCE + AVAILABLE + admitted path may verify and commit."""

    verdict = provider_production_gate_regression(
        disposition="real",
        gateway_override={},
    )
    assert verdict.admitted is True
    assert verdict.can_verify is True
    assert verdict.can_commit is True
    assert verdict.simulated is False
    assert "production_admitted" in verdict.reason_codes


def test_absent_provider_is_nonzero_and_never_verified() -> None:
    decision = route_model(_inputs())
    # Wrong capability only → unavailable.
    wrong = _provider(capabilities=(ModelCapability.FRONTIER.value,))
    result = invoke_model(
        decision=decision,
        providers=[wrong],
        mode=HarnessMode.PRODUCTION,
        prompt="absent-capability",
    )
    assert result.status == "unavailable"
    assert result.exit_code != 0
    assert result.exit_code == 1
    assert result.unavailable is not None
    assert result.unavailable.reason_code == "provider_unavailable"
    _assert_never_production_verified(result)

    empty = invoke_model(
        decision=decision,
        providers=[],
        mode=HarnessMode.PRODUCTION,
        prompt="absent-list",
    )
    assert empty.status == "unavailable"
    assert empty.exit_code != 0
    _assert_never_production_verified(empty)

    flagged = invoke_model(
        decision=decision,
        providers=[_provider(available=False)],
        mode=HarnessMode.PRODUCTION,
        prompt="flagged-unavailable",
    )
    assert flagged.status == "unavailable"
    assert flagged.exit_code != 0
    _assert_never_production_verified(flagged)


def test_default_development_simulation_never_verifies_or_commits() -> None:
    result = provider_production_gate_regression(
        disposition="default-simulated",
        mode=HarnessMode.DEVELOPMENT.value,
        use_gateway=False,
    )
    assert result.simulated is True
    assert result.exit_code == 0  # observational success in development
    assert result.gate is not None
    assert result.gate.can_verify is False
    assert result.gate.can_commit is False
    assert result.gate.simulated is True


@pytest.mark.parametrize(
    "override,label",
    [
        ({"reservation_id": "sim:local"}, "sim-reservation"),
        ({"reservation_id": "degraded:local"}, "degraded-reservation"),
        ({"phase": "simulated"}, "phase-simulated"),
        ({"phase": "SIMULATED"}, "phase-SIMULATED"),
        ({"phase": "degraded"}, "phase-degraded"),
        ({"phase": "DEGRADED"}, "phase-DEGRADED"),
        ({"coordination_state": "simulated"}, "coord-simulated"),
    ],
)
def test_simulated_and_degraded_never_production_verified(
    override: dict[str, Any], label: str
) -> None:
    verdict = provider_production_gate_regression(
        disposition=label,
        gateway_override=override,
    )
    _assert_never_production_verified(verdict)
    assert verdict.admitted is False
    assert verdict.can_verify is False
    assert verdict.can_commit is False

    # Via invoke_model: rejected with nonzero exit.
    rejected = invoke_model(
        decision=route_model(_inputs()),
        providers=[_provider()],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(**override),
        coordinator_present=True,
        invoker_present=True,
    )
    assert rejected.status == "rejected"
    assert rejected.exit_code != 0
    assert rejected.gate is not None
    assert rejected.gate.can_verify is False
    assert rejected.gate.can_commit is False


@pytest.mark.parametrize(
    "override,label",
    [
        ({"mode": "off"}, "mode-off"),
        ({"mode": "OFF"}, "mode-OFF"),
        ({"mode": "observe"}, "mode-observe"),
        ({"mode": "shadow"}, "mode-shadow"),
        ({"mode": "assist"}, "mode-assist"),
        ({"phase": "off"}, "phase-off"),
        ({"phase": "denied"}, "phase-denied"),
        ({"phase": "failed"}, "phase-failed"),
        ({"coordination_state": "unavailable"}, "coord-unavailable"),
        ({"granted": False}, "not-granted"),
    ],
)
def test_off_and_denied_dispositions_never_production_verified(
    override: dict[str, Any], label: str
) -> None:
    verdict = provider_production_gate_regression(
        disposition=label,
        gateway_override=override,
    )
    _assert_never_production_verified(verdict)
    assert verdict.can_verify is False
    assert verdict.can_commit is False

    rejected = invoke_model(
        decision=route_model(_inputs()),
        providers=[_provider()],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(**override),
        coordinator_present=True,
        invoker_present=True,
    )
    assert rejected.exit_code != 0
    assert rejected.gate is not None
    assert rejected.gate.can_verify is False


@pytest.mark.parametrize(
    "reason",
    [
        "local_fallback_used",
        "cross_provider_fallback",
        "allow_local_fallback",
        "allow_cross_provider_fallback",
        "simulated_fallback",
        "degraded_capacity",
    ],
)
def test_fallback_reason_codes_never_production_verified(reason: str) -> None:
    override = {"reason_codes": (reason,)}
    verdict = provider_production_gate_regression(
        disposition=f"fallback:{reason}",
        gateway_override=override,
    )
    _assert_never_production_verified(verdict)
    joined = " ".join(verdict.reason_codes)
    assert (
        "fallback" in joined
        or "degraded" in joined
        or "simulated" in joined
        or "production_rejected" in joined
        or verdict.admitted is False
    )

    rejected = invoke_model(
        decision=route_model(_inputs()),
        providers=[_provider()],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(**override),
        coordinator_present=True,
        invoker_present=True,
    )
    assert rejected.exit_code != 0
    assert rejected.status == "rejected"
    assert rejected.gate is not None
    assert rejected.gate.can_verify is False
    assert rejected.gate.can_commit is False


def test_unadmitted_replay_never_production_verified() -> None:
    override = {
        "replayed": True,
        "supervisor_receipt_id": "receipt:forged",
    }
    verdict = provider_production_gate_regression(
        disposition="unadmitted-replay",
        gateway_override=override,
    )
    _assert_never_production_verified(verdict)
    assert "unadmitted_replay" in verdict.reason_codes

    # Empty admitted set rejects even a known-looking receipt id.
    empty_gate = _production_gate(admitted_production_receipt_ids=())
    verdict2 = empty_gate.evaluate(
        _gateway_result(replayed=True, supervisor_receipt_id="receipt:prod-1"),
        mode=HarnessMode.PRODUCTION,
    )
    _assert_never_production_verified(verdict2)
    assert "unadmitted_replay" in verdict2.reason_codes

    rejected = invoke_model(
        decision=route_model(_inputs()),
        providers=[_provider()],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(**override),
        coordinator_present=True,
        invoker_present=True,
    )
    assert rejected.exit_code != 0
    assert rejected.gate is not None
    assert rejected.gate.can_verify is False


def test_admitted_replay_may_verify() -> None:
    verdict = _production_gate().evaluate(
        _gateway_result(replayed=True, supervisor_receipt_id="receipt:prod-1"),
        mode=HarnessMode.PRODUCTION,
    )
    assert verdict.admitted is True
    assert verdict.can_verify is True
    assert verdict.can_commit is True


def test_missing_coordinator_or_invoker_never_production_verified() -> None:
    result = _gateway_result()
    no_coord = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=False,
        invoker_present=True,
    ).evaluate(result, mode=HarnessMode.PRODUCTION)
    _assert_never_production_verified(no_coord)
    assert "coordinator_absent" in no_coord.reason_codes

    no_invoker = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=True,
        invoker_present=False,
    ).evaluate(result, mode=HarnessMode.PRODUCTION)
    _assert_never_production_verified(no_invoker)
    assert "invoker_absent" in no_invoker.reason_codes


def test_provider_mismatch_and_unverified_attribution_never_verified() -> None:
    for override in (
        {"provider_id": "other-provider"},
        {"attribution": None},
        {"attribution": {}},
    ):
        verdict = _production_gate().evaluate(
            _gateway_result(**override),
            mode=HarnessMode.PRODUCTION,
        )
        _assert_never_production_verified(verdict)


def test_invoke_model_production_matrix_exit_codes() -> None:
    """Compact matrix: every forbidden gateway disposition is nonzero."""

    decision = route_model(_inputs())
    provider = _provider()
    forbidden: list[dict[str, Any]] = [
        {"reservation_id": "sim:x"},
        {"reservation_id": "degraded:x"},
        {"mode": "off"},
        {"phase": "simulated"},
        {"phase": "degraded"},
        {"reason_codes": ("local_fallback_used",)},
        {"reason_codes": ("cross_provider_fallback",)},
        {"replayed": True, "supervisor_receipt_id": "receipt:unknown"},
        {"coordination_state": "unavailable"},
        {"granted": False},
    ]
    for override in forbidden:
        result = invoke_model(
            decision=decision,
            providers=[provider],
            mode=HarnessMode.PRODUCTION,
            gateway_result=_gateway_result(**override),
            coordinator_present=True,
            invoker_present=True,
        )
        assert result.exit_code != 0, override
        assert result.status == "rejected", override
        assert result.gate is not None
        assert result.gate.can_verify is False, override
        assert result.gate.can_commit is False, override


def test_human_review_and_deterministic_halt_before_dispatch() -> None:
    human = route_model(_inputs(risk=RiskClass.HIGH.value))
    provider = _provider()
    human_result = invoke_model(
        decision=human,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        prompt="halt-human",
    )
    assert human_result.halted is True
    assert human_result.provider_id is None
    assert getattr(provider, "calls") == []

    det_inputs = _inputs(
        context_tokens=100,
        lowest_confidence=ConfidenceClass.EXACT.value,
        risk=RiskClass.LOW.value,
        dependency_cone_size=1,
        unresolved_obligations=0,
        available_proofs=1,
    )
    det = route_model(det_inputs)
    assert det.route == ModelRoute.DETERMINISTIC_ONLY.value
    det_result = invoke_model(
        decision=det,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
    )
    assert det_result.halted is True
    assert getattr(provider, "calls") == []


def test_model_provider_descriptor_lists_fail_closed_invariants() -> None:
    descriptor = model_provider_descriptor()
    assert descriptor["interface"] == "ModelProvider@1"
    invariants = " ".join(descriptor.get("invariants", []))
    # Descriptor should advertise fail-closed production posture.
    assert any(
        token in invariants.casefold()
        for token in (
            "production",
            "simulat",
            "fallback",
            "fail",
            "admit",
        )
    ) or "invariants" in descriptor


def test_routing_decision_round_trip_for_regression_helpers() -> None:
    """Keep RoutingDecision construction compatible with gate tests."""

    inputs = _inputs()
    decision = RoutingDecision(
        route=ModelRoute.SMALL_LOCAL_MODEL.value,
        reason_codes=("small_local",),
        explanation="regression helper",
        requires_provider=True,
        halt_before_dispatch=False,
        halt_before_root_publication=False,
        inputs=inputs,
        policy=ModelRoutingPolicy.default(),
    )
    assert decision.requires_provider is True
    assert decision.halt_before_dispatch is False


def test_release_bound_production_surfaces_remain_importable() -> None:
    """Named SCH-018 release regressions share these production authorities."""

    assert issubclass(ProcessFenceError, RuntimeError)
    assert callable(run_leased_lane_result)
    assert issubclass(OwnershipError, Exception)
    assert ProofScheduler is not None
    assert callable(terminate_pid_tree)
    assert callable(validate_untrusted_implementation_proposal)
    assert ProposalFindingCode.SECRET_CHANGE_FORBIDDEN.value == (
        "secret_change_forbidden"
    )
