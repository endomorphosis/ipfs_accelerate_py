"""SCH-018 provider production-gate regressions.

Covers real, absent, default-simulated, degraded, off, replayed, and fallback
provider dispositions. Production unavailable / simulation / fallback paths must
be nonzero and never report verification or root-commit authority.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessMode,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.providers import (
    InjectedModelProvider,
    ModelCapability,
    ProductionProviderGate,
    ProviderCapabilitySpec,
    build_llm_router_invoker,
    invoke_model,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    ConfidenceClass,
    RiskClass,
    RoutingInputs,
    route_model,
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
    simulated: bool = False,
) -> InjectedModelProvider:
    calls: list[dict[str, Any]] = []

    def generate_fn(prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        calls.append({"prompt": prompt, **kwargs})
        if raise_error is not None:
            raise raise_error
        base = {
            "provider_id": provider_id,
            "status": "ok",
            "simulated": simulated
            or kwargs.get("mode") == HarnessMode.DEVELOPMENT.value,
        }
        if observation is not None:
            base.update(dict(observation))
        return base

    provider = InjectedModelProvider(
        spec=ProviderCapabilitySpec(
            provider_id=provider_id,
            capabilities=capabilities,
            max_context_tokens=max_context_tokens,
            available=available,
        ),
        generate_fn=generate_fn,
    )
    # Test-only call log on a frozen dataclass.
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


def _assert_never_verifies(result: Any) -> None:
    """Shared production non-authority assertions."""

    assert result.exit_code != 0
    if getattr(result, "gate", None) is not None:
        assert result.gate.can_verify is False
        assert result.gate.can_commit is False
        assert result.gate.admitted is False


# ---------------------------------------------------------------------------
# Real path
# ---------------------------------------------------------------------------


def test_real_production_gateway_path_admits_and_may_verify() -> None:
    """provider_production_gate_regression: real ENFORCE path admits."""

    decision = route_model(_inputs())
    provider = _provider()
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(),
        coordinator_present=True,
        invoker_present=True,
    )
    assert result.status == "admitted"
    assert result.exit_code == 0
    assert result.gate is not None
    assert result.gate.admitted is True
    assert result.gate.can_verify is True
    assert result.gate.can_commit is True
    assert result.gate.simulated is False
    assert result.simulated is False
    # Gating a gateway result must not open a second direct generate path.
    assert getattr(provider, "calls") == []


def test_production_gate_direct_evaluate_real_path() -> None:
    verdict = _production_gate().evaluate(
        _gateway_result(),
        mode=HarnessMode.PRODUCTION,
    )
    assert verdict.admitted is True
    assert verdict.can_verify is True
    assert verdict.can_commit is True
    assert verdict.simulated is False
    assert "production_admitted" in verdict.reason_codes


# ---------------------------------------------------------------------------
# Absent / unavailable
# ---------------------------------------------------------------------------


def test_absent_provider_is_typed_unavailable_nonzero_never_verifies() -> None:
    decision = route_model(_inputs())
    # Wrong capability set: no match for small_local / medium routes.
    provider = _provider(capabilities=(ModelCapability.FRONTIER.value,))
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
    )
    assert result.status == "unavailable"
    assert result.exit_code == 1
    assert result.exit_code != 0
    assert result.unavailable is not None
    assert result.unavailable.reason_code == "provider_unavailable"
    assert result.gate is None or result.gate.can_verify is False

    empty = invoke_model(
        decision=decision,
        providers=[],
        mode=HarnessMode.PRODUCTION,
    )
    assert empty.status == "unavailable"
    assert empty.exit_code != 0
    assert empty.unavailable is not None


def test_unavailable_flag_is_nonzero_and_non_verifying() -> None:
    decision = route_model(_inputs())
    provider = _provider(available=False)
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
    )
    assert result.status == "unavailable"
    assert result.exit_code != 0
    assert result.unavailable is not None


# ---------------------------------------------------------------------------
# Default simulated (development)
# ---------------------------------------------------------------------------


def test_default_development_simulation_never_verifies_or_commits() -> None:
    decision = route_model(_inputs())
    provider = _provider()
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.DEVELOPMENT,
        prompt="dev-sim",
    )
    assert result.simulated is True
    # Development observation may exit 0 but must never verify/commit.
    assert result.gate is not None
    assert result.gate.can_verify is False
    assert result.gate.can_commit is False
    assert result.gate.simulated is True
    assert result.gate.admitted is False or result.gate.can_verify is False


def test_development_sim_reservation_never_verifies() -> None:
    verdict = _production_gate().evaluate(
        _gateway_result(reservation_id="sim:dev"),
        mode=HarnessMode.DEVELOPMENT,
    )
    assert verdict.can_verify is False
    assert verdict.can_commit is False
    assert verdict.simulated is True


# ---------------------------------------------------------------------------
# Degraded / off / simulated phases and reservations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"reservation_id": "sim:local"}, "sim"),
        ({"reservation_id": "degraded:local"}, "degraded"),
        ({"phase": "degraded"}, "phase_degraded"),
        ({"phase": "simulated"}, "phase_simulated"),
        ({"phase": "off"}, "phase_off"),
        ({"mode": "off"}, "mode_off"),
        ({"mode": "assist"}, "non_enforce_mode"),
        ({"mode": "observe"}, "non_enforce_mode"),
        ({"coordination_state": "unavailable"}, "coordination"),
        ({"coordination_state": "simulated"}, "coordination"),
    ],
)
def test_production_rejects_degraded_off_simulated_dispositions(
    override: dict[str, Any], fragment: str
) -> None:
    decision = route_model(_inputs())
    provider = _provider()
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(**override),
        coordinator_present=True,
        invoker_present=True,
    )
    assert result.status == "rejected"
    _assert_never_verifies(result)
    joined = " ".join(result.reason_codes) + " " + (result.diagnostic or "")
    assert fragment in joined or fragment in result.gate.reason_codes  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Replayed
# ---------------------------------------------------------------------------


def test_unadmitted_replay_is_nonzero_and_never_verifies() -> None:
    decision = route_model(_inputs())
    provider = _provider()
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(
            replayed=True,
            supervisor_receipt_id="receipt:unknown",
        ),
        coordinator_present=True,
        invoker_present=True,
    )
    assert result.status == "rejected"
    _assert_never_verifies(result)
    assert "unadmitted_replay" in result.reason_codes


def test_admitted_replay_may_verify() -> None:
    decision = route_model(_inputs())
    provider = _provider()
    gate = _production_gate(
        admitted_production_receipt_ids=("receipt:prod-1",),
    )
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(
            replayed=True,
            supervisor_receipt_id="receipt:prod-1",
        ),
        gate=gate,
        coordinator_present=True,
        invoker_present=True,
    )
    # Admitted production receipt IDs allow replay through the gate.
    assert result.status == "admitted"
    assert result.exit_code == 0
    assert result.gate is not None
    assert result.gate.can_verify is True
    assert result.gate.can_commit is True


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason_codes",
    [
        ("local_fallback_used",),
        ("cross_provider_fallback",),
        ("allow_local_fallback",),
        ("fallback",),
        ("degraded_fallback",),
    ],
)
def test_production_fallback_reasons_are_nonzero_never_verified(
    reason_codes: tuple[str, ...],
) -> None:
    decision = route_model(_inputs())
    provider = _provider()
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(reason_codes=reason_codes),
        coordinator_present=True,
        invoker_present=True,
    )
    assert result.status == "rejected"
    _assert_never_verifies(result)
    assert "fallback_reason_present" in result.reason_codes


def test_production_simulated_direct_observation_never_verifies() -> None:
    """Direct generate that claims simulated under production is rejected."""

    decision = route_model(_inputs())
    provider = _provider(simulated=True)
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        prompt="should-reject",
    )
    assert result.status == "rejected"
    _assert_never_verifies(result)
    assert result.simulated is True
    assert "simulated_observation" in result.reason_codes


# ---------------------------------------------------------------------------
# llm_router invoker: no silent fallback
# ---------------------------------------------------------------------------


def test_llm_router_invoker_forces_fallback_flags_off() -> None:
    captured: dict[str, Any] = {}

    def fake_generate(prompt: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        captured["prompt"] = prompt
        return "ok-text"

    def fake_trace() -> Mapping[str, Any]:
        return {
            "effective_provider_name": "provider-alpha",
            "provider_name": "provider-alpha",
        }

    invoker = build_llm_router_invoker(
        provider_id="provider-alpha",
        generate_text=fake_generate,
        get_last_generation_trace=fake_trace,
    )
    observation = invoker(
        SimpleNamespace(
            provider_id="provider-alpha",
            operation="model_invocation",
            metadata={"prompt": "hello"},
        )
    )
    assert captured.get("allow_local_fallback") is False
    assert captured.get("allow_cross_provider_fallback") is False
    assert observation["allow_local_fallback"] is False
    assert observation["allow_cross_provider_fallback"] is False
    assert observation["provider_id"] == "provider-alpha"
    assert observation["effective_provider"] == "provider-alpha"


def test_llm_router_invoker_rejects_effective_provider_mismatch() -> None:
    def fake_generate(prompt: str, **kwargs: Any) -> str:
        return "ok"

    def fake_trace() -> Mapping[str, Any]:
        return {"effective_provider_name": "other-provider"}

    invoker = build_llm_router_invoker(
        provider_id="provider-alpha",
        generate_text=fake_generate,
        get_last_generation_trace=fake_trace,
    )
    with pytest.raises(Exception, match="effective provider"):
        invoker(
            SimpleNamespace(
                provider_id="provider-alpha",
                operation="model_invocation",
                metadata={"prompt": "x"},
            )
        )


# ---------------------------------------------------------------------------
# Halts before dispatch (still non-provider)
# ---------------------------------------------------------------------------


def test_human_review_and_deterministic_never_dispatch() -> None:
    high = route_model(_inputs(risk=RiskClass.HIGH.value))
    assert high.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    provider = _provider()
    halted = invoke_model(
        decision=high,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        prompt="nope",
    )
    assert halted.halted is True
    assert halted.provider_id is None
    assert getattr(provider, "calls") == []

    det = route_model(
        _inputs(
            context_tokens=100,
            lowest_confidence=ConfidenceClass.EXACT.value,
            risk=RiskClass.LOW.value,
            dependency_cone_size=1,
            unresolved_obligations=0,
            available_proofs=1,
        )
    )
    assert det.route == ModelRoute.DETERMINISTIC_ONLY.value
    provider2 = _provider()
    result = invoke_model(decision=det, providers=[provider2])
    assert result.halted is True
    assert getattr(provider2, "calls") == []


# ---------------------------------------------------------------------------
# Missing coordinator / invoker
# ---------------------------------------------------------------------------


def test_missing_coordinator_or_invoker_never_verifies() -> None:
    result = _gateway_result()
    no_coord = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=False,
        invoker_present=True,
    ).evaluate(result, mode=HarnessMode.PRODUCTION)
    assert no_coord.admitted is False
    assert no_coord.can_verify is False
    assert no_coord.can_commit is False
    assert "coordinator_absent" in no_coord.reason_codes

    no_invoker = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=True,
        invoker_present=False,
    ).evaluate(result, mode=HarnessMode.PRODUCTION)
    assert no_invoker.can_verify is False
    assert no_invoker.can_commit is False
    assert "invoker_absent" in no_invoker.reason_codes
