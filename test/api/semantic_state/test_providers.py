"""SCH-007 real-provider adapter and production gate tests."""

from __future__ import annotations

import importlib
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    HarnessMode,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.providers import (
    MODEL_PROVIDER_INTERFACE,
    InjectedModelProvider,
    ModelCapability,
    ModelInvocationResult,
    ProductionGateVerdict,
    ProductionProviderGate,
    ProviderCapabilitySpec,
    build_llm_router_invoker,
    capability_for_route,
    invoke_model,
    model_provider_descriptor,
    select_provider_for_route,
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
    # Expose call log for assertions.
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


def test_cold_import_starts_no_resources_threads_processes_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "ipfs_accelerate_py.agent_supervisor.semantic_state.providers"
    sys.modules.pop(module_name, None)

    before_threads = {t.ident for t in threading.enumerate()}
    real_thread_start = threading.Thread.start
    started_threads: list[str] = []

    def guarded_start(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        started_threads.append(self.name)
        return real_thread_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", guarded_start)

    real_popen = subprocess.Popen
    popen_calls: list[Any] = []

    def guarded_popen(*args: Any, **kwargs: Any):
        popen_calls.append((args, kwargs))
        raise AssertionError("providers import must not spawn processes")

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)

    # Ensure import does not eagerly pull llm_router network paths.
    import_calls: list[str] = []
    real_import = __import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name == "ipfs_accelerate_py.llm_router" or name.endswith(".llm_router"):
            import_calls.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    mod = importlib.import_module(module_name)
    after_threads = {t.ident for t in threading.enumerate()}

    assert mod.MODEL_PROVIDER_INTERFACE == MODEL_PROVIDER_INTERFACE
    assert not started_threads
    assert not popen_calls
    assert not import_calls
    assert after_threads == before_threads


def test_capability_for_route_mapping() -> None:
    assert (
        capability_for_route(ModelRoute.SMALL_LOCAL_MODEL)
        == ModelCapability.SMALL_LOCAL
    )
    assert capability_for_route("frontier_model") == ModelCapability.FRONTIER
    with pytest.raises(HarnessError):
        capability_for_route("not_a_route")


def test_human_review_required_halts_before_provider_dispatch() -> None:
    decision = route_model(_inputs(risk=RiskClass.HIGH.value))
    provider = _provider()
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        prompt="should-not-run",
    )
    assert result.halted is True
    assert result.status == "human_review_required"
    assert result.provider_id is None
    assert result.exit_code == 0
    assert "halt_before_dispatch" in result.reason_codes
    assert getattr(provider, "calls") == []


def test_deterministic_only_never_dispatches() -> None:
    decision = route_model(
        _inputs(
            context_tokens=100,
            lowest_confidence=ConfidenceClass.EXACT.value,
            risk=RiskClass.LOW.value,
            dependency_cone_size=1,
            unresolved_obligations=0,
            available_proofs=1,
        )
    )
    assert decision.route == ModelRoute.DETERMINISTIC_ONLY.value
    provider = _provider()
    result = invoke_model(decision=decision, providers=[provider])
    assert result.halted is True
    assert result.status == "deterministic_only"
    assert getattr(provider, "calls") == []


def test_missing_provider_is_typed_unavailable_and_nonzero() -> None:
    decision = route_model(
        _inputs(
            context_tokens=2_000,
            lowest_confidence=ConfidenceClass.HEURISTIC.value,
            risk=RiskClass.LOW.value,
        )
    )
    assert decision.route == ModelRoute.SMALL_LOCAL_MODEL.value
    # Wrong capability only.
    provider = _provider(capabilities=(ModelCapability.FRONTIER.value,))
    result = invoke_model(decision=decision, providers=[provider])
    assert result.status == "unavailable"
    assert result.exit_code == 1
    assert result.exit_code != 0
    assert result.unavailable is not None
    assert result.unavailable.reason_code == "provider_unavailable"
    assert "provider_unavailable" in result.reason_codes

    # No providers at all.
    empty = invoke_model(decision=decision, providers=[])
    assert empty.status == "unavailable"
    assert empty.exit_code != 0


def test_unavailable_provider_flag_yields_typed_unavailable() -> None:
    decision = route_model(_inputs())
    provider = _provider(available=False)
    result = invoke_model(decision=decision, providers=[provider])
    assert result.status == "unavailable"
    assert result.exit_code == 1
    assert result.unavailable is not None


def test_development_simulation_never_verifies_or_commits() -> None:
    decision = route_model(_inputs())
    provider = _provider()
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.DEVELOPMENT,
        prompt="dev-sim",
    )
    assert result.simulated is True
    assert result.exit_code == 0
    assert result.gate is not None
    assert result.gate.can_verify is False
    assert result.gate.can_commit is False
    assert result.gate.simulated is True


def test_production_gate_admits_enforce_available_real_path() -> None:
    gate = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=True,
        invoker_present=True,
        admitted_production_receipt_ids=("receipt:prod-1",),
    )
    verdict = gate.evaluate(
        _gateway_result(),
        mode=HarnessMode.PRODUCTION,
    )
    assert verdict.admitted is True
    assert verdict.can_verify is True
    assert verdict.can_commit is True
    assert verdict.simulated is False
    assert "production_admitted" in verdict.reason_codes


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"mode": "off"}, "mode_off"),
        ({"mode": "assist"}, "non_enforce_mode"),
        ({"coordination_state": "unavailable"}, "coordination_not_available"),
        ({"coordination_state": "simulated"}, "coordination_simulated"),
        ({"reservation_id": "sim:abc"}, "simulated"),
        ({"reservation_id": "degraded:abc"}, "simulated_or_degraded_reservation"),
        ({"phase": "degraded"}, "phase_degraded"),
        ({"phase": "simulated"}, "phase_simulated"),
        ({"reason_codes": ("local_fallback_used",)}, "fallback_reason_present"),
        ({"reason_codes": ("cross_provider_fallback",)}, "fallback_reason_present"),
        ({"provider_id": "other-provider"}, "provider_mismatch"),
        ({"attribution": None}, "attribution_unverified"),
        ({"attribution": {}}, "attribution_unverified"),
        ({"granted": False}, "not_granted"),
        (
            {"replayed": True, "supervisor_receipt_id": "receipt:unknown"},
            "unadmitted_replay",
        ),
    ],
)
def test_production_gate_rejects_forbidden_paths(
    override: dict[str, Any], fragment: str
) -> None:
    gate = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=True,
        invoker_present=True,
        admitted_production_receipt_ids=("receipt:prod-1",),
    )
    verdict = gate.evaluate(
        _gateway_result(**override),
        mode=HarnessMode.PRODUCTION,
    )
    assert verdict.admitted is False
    assert verdict.can_verify is False
    assert verdict.can_commit is False
    joined = " ".join(verdict.reason_codes)
    assert fragment in joined or fragment in verdict.diagnostic


def test_production_gate_rejects_missing_coordinator_or_invoker() -> None:
    result = _gateway_result()
    no_coord = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=False,
        invoker_present=True,
    ).evaluate(result, mode=HarnessMode.PRODUCTION)
    assert no_coord.admitted is False
    assert "coordinator_absent" in no_coord.reason_codes

    no_invoker = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=True,
        invoker_present=False,
    ).evaluate(result, mode=HarnessMode.PRODUCTION)
    assert no_invoker.admitted is False
    assert "invoker_absent" in no_invoker.reason_codes


def test_production_gate_allows_admitted_replay() -> None:
    gate = ProductionProviderGate(
        expected_provider_id="provider-alpha",
        coordinator_present=True,
        invoker_present=True,
        admitted_production_receipt_ids=("receipt:prod-1",),
    )
    verdict = gate.evaluate(
        _gateway_result(replayed=True, supervisor_receipt_id="receipt:prod-1"),
        mode=HarnessMode.PRODUCTION,
    )
    assert verdict.admitted is True
    assert verdict.can_commit is True


def test_invoke_model_applies_gate_to_gateway_result() -> None:
    decision = route_model(_inputs())
    provider = _provider()
    admitted = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(),
        coordinator_present=True,
        invoker_present=True,
    )
    assert admitted.status == "admitted"
    assert admitted.exit_code == 0
    assert admitted.gate is not None
    assert admitted.gate.can_verify is True
    # Direct generate must not have been called when gating a gateway result.
    assert getattr(provider, "calls") == []

    rejected = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        gateway_result=_gateway_result(reservation_id="sim:bad"),
        coordinator_present=True,
        invoker_present=True,
    )
    assert rejected.status == "rejected"
    assert rejected.exit_code == 1
    assert rejected.gate is not None
    assert rejected.gate.can_commit is False


def test_build_llm_router_invoker_disables_fallback_and_verifies_provider() -> None:
    seen: dict[str, Any] = {}

    def fake_generate(prompt: str, **kwargs: Any) -> str:
        seen.update(kwargs)
        seen["prompt"] = prompt
        return "ok-text"

    def fake_trace() -> Mapping[str, Any]:
        return {"effective_provider_name": "provider-alpha"}

    invoker = build_llm_router_invoker(
        provider_id="provider-alpha",
        generate_text=fake_generate,
        get_last_generation_trace=fake_trace,
        model_name="model-x",
    )
    observation = invoker(
        SimpleNamespace(
            provider_id="provider-alpha",
            metadata={"prompt": "hello"},
            operation="model_invocation",
        )
    )
    assert seen["allow_local_fallback"] is False
    assert seen["allow_cross_provider_fallback"] is False
    assert seen["provider"] == "provider-alpha"
    assert observation["effective_provider"] == "provider-alpha"
    assert observation["allow_local_fallback"] is False
    assert observation["allow_cross_provider_fallback"] is False


def test_build_llm_router_invoker_rejects_effective_provider_mismatch() -> None:
    invoker = build_llm_router_invoker(
        provider_id="provider-alpha",
        generate_text=lambda prompt, **kwargs: "x",
        get_last_generation_trace=lambda: {
            "effective_provider_name": "some-other-provider"
        },
    )
    with pytest.raises(HarnessError, match="effective provider"):
        invoker(
            SimpleNamespace(
                provider_id="provider-alpha",
                metadata={"prompt": "p"},
                operation="model_invocation",
            )
        )


def test_select_provider_matches_capability_and_context() -> None:
    small = _provider(
        provider_id="small-only",
        capabilities=(ModelCapability.SMALL_LOCAL.value,),
        max_context_tokens=4_000,
    )
    frontier = _provider(
        provider_id="frontier-only",
        capabilities=(ModelCapability.FRONTIER.value,),
        max_context_tokens=200_000,
    )
    selected = select_provider_for_route(
        [small, frontier],
        route=ModelRoute.SMALL_LOCAL_MODEL,
        context_tokens=2_000,
    )
    assert selected is not None
    assert selected.provider_id == "small-only"

    # Context exceeds small provider budget → no match among remaining.
    none_selected = select_provider_for_route(
        [small],
        route=ModelRoute.SMALL_LOCAL_MODEL,
        context_tokens=10_000,
    )
    assert none_selected is None


def test_provider_invoke_failure_is_typed() -> None:
    decision = route_model(_inputs())
    provider = _provider(raise_error=RuntimeError("boom"))
    result = invoke_model(decision=decision, providers=[provider])
    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.unavailable is not None
    assert result.unavailable.reason_code == "provider_invoke_failed"


def test_closed_records_round_trip() -> None:
    decision = route_model(_inputs())
    provider = _provider()
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.DEVELOPMENT,
    )
    restored = ModelInvocationResult.from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()

    verdict = ProductionGateVerdict.from_dict(
        {
            "admitted": False,
            "can_verify": False,
            "can_commit": False,
            "reason_codes": ["production_rejected", "mode_off"],
            "diagnostic": "rejected",
            "simulated": False,
            "mode": HarnessMode.PRODUCTION.value,
        }
    )
    assert verdict.admitted is False

    with pytest.raises(HarnessError, match="never verify"):
        ProductionGateVerdict.from_dict(
            {
                "admitted": True,
                "can_verify": True,
                "can_commit": True,
                "reason_codes": ["bad"],
                "diagnostic": "bad",
                "simulated": True,
                "mode": HarnessMode.DEVELOPMENT.value,
            }
        )


def test_model_provider_descriptor_lists_fail_closed_invariants() -> None:
    descriptor = model_provider_descriptor()
    assert descriptor["interface"] == MODEL_PROVIDER_INTERFACE
    invariants = set(descriptor["invariants"])
    assert "development_simulation_never_verifies_or_commits" in invariants
    assert "rejects_sim_and_degraded_reservations" in invariants
    assert "no_second_provider_execution_gateway" in invariants
    assert "llm_router_disables_local_and_cross_provider_fallback" in invariants


def test_production_rejects_simulated_direct_observation() -> None:
    decision = route_model(_inputs())
    provider = _provider(observation={"simulated": True})
    result = invoke_model(
        decision=decision,
        providers=[provider],
        mode=HarnessMode.PRODUCTION,
        prompt="prod",
    )
    assert result.status == "rejected"
    assert result.exit_code == 1
    assert result.simulated is True
    assert result.gate is not None
    assert result.gate.can_commit is False
