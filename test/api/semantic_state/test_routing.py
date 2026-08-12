"""SCH-007 deterministic model routing tests."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    ContextPack,
    HarnessError,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    MODEL_ROUTING_INTERFACE,
    ConfidenceClass,
    ModelRoutingPolicy,
    RiskClass,
    RoutingDecision,
    RoutingInputs,
    model_routing_descriptor,
    route_allows_provider_dispatch,
    route_model,
    route_requires_human_review,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _inputs(**overrides: object) -> RoutingInputs:
    payload: dict[str, Any] = {
        "context_tokens": 512,
        "lowest_confidence": ConfidenceClass.EXACT.value,
        "risk": RiskClass.LOW.value,
        "dependency_cone_size": 2,
        "unresolved_obligations": 0,
        "prior_repair_failures": 0,
        "available_proofs": 1,
        "prior_route_failed": False,
    }
    payload.update(overrides)
    return RoutingInputs.from_dict(payload)


def test_cold_import_starts_no_resources_threads_processes_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "ipfs_accelerate_py.agent_supervisor.semantic_state.routing"
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
        raise AssertionError("routing import must not spawn processes")

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)

    mod = importlib.import_module(module_name)
    after_threads = {t.ident for t in threading.enumerate()}

    assert mod.MODEL_ROUTING_INTERFACE == MODEL_ROUTING_INTERFACE
    assert not started_threads
    assert not popen_calls
    assert after_threads == before_threads


def test_route_decision_is_deterministic_and_explained() -> None:
    inputs = _inputs(
        context_tokens=8_000,
        lowest_confidence=ConfidenceClass.HEURISTIC.value,
        risk=RiskClass.MEDIUM.value,
        dependency_cone_size=12,
        unresolved_obligations=1,
        available_proofs=0,
    )
    first = route_model(inputs)
    second = route_model(inputs.to_dict())
    assert first.route == second.route
    assert first.reason_codes == second.reason_codes
    assert first.explanation == second.explanation
    assert first.explanation
    assert first.reason_codes
    assert first.route in {item.value for item in ModelRoute}
    # Round-trip closed record.
    restored = RoutingDecision.from_dict(first.to_dict())
    assert restored.to_dict() == first.to_dict()


def test_deterministic_only_for_exact_low_risk_proof_covered() -> None:
    decision = route_model(
        _inputs(
            context_tokens=100,
            lowest_confidence=ConfidenceClass.EXACT.value,
            risk=RiskClass.LOW.value,
            dependency_cone_size=1,
            unresolved_obligations=1,
            available_proofs=1,
        )
    )
    assert decision.route == ModelRoute.DETERMINISTIC_ONLY.value
    assert decision.requires_provider is False
    assert decision.halt_before_dispatch is True
    assert decision.halt_before_root_publication is False
    assert route_allows_provider_dispatch(decision) is False


def test_human_review_required_for_high_risk() -> None:
    decision = route_model(_inputs(risk=RiskClass.HIGH.value))
    assert decision.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert decision.halt_before_dispatch is True
    assert decision.halt_before_root_publication is True
    assert decision.requires_provider is False
    assert route_requires_human_review(decision) is True
    assert route_allows_provider_dispatch(decision) is False
    assert "risk_high" in decision.reason_codes


def test_human_review_required_for_opaque_confidence() -> None:
    decision = route_model(
        _inputs(lowest_confidence=ConfidenceClass.OPAQUE.value, risk=RiskClass.LOW.value)
    )
    assert decision.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert "confidence_opaque" in decision.reason_codes


def test_human_review_required_for_oversized_context() -> None:
    policy = ModelRoutingPolicy.default()
    decision = route_model(
        _inputs(context_tokens=policy.oversized_context_tokens + 1)
    )
    assert decision.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert "context_oversized" in decision.reason_codes


def test_human_review_required_for_prior_failures_and_failed_route() -> None:
    policy = ModelRoutingPolicy.default()
    failed = route_model(
        _inputs(prior_repair_failures=policy.max_prior_failures_before_human + 1)
    )
    assert failed.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert "prior_repair_failures_exceeded" in failed.reason_codes

    prior_failed = route_model(_inputs(prior_route_failed=True))
    assert prior_failed.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert "prior_route_failed" in prior_failed.reason_codes


def test_small_medium_frontier_score_bands() -> None:
    small = route_model(
        _inputs(
            context_tokens=2_000,
            lowest_confidence=ConfidenceClass.HEURISTIC.value,
            risk=RiskClass.LOW.value,
            dependency_cone_size=3,
            unresolved_obligations=0,
            available_proofs=0,
            prior_repair_failures=0,
        )
    )
    # Heuristic prevents deterministic_only; small band otherwise.
    assert small.route == ModelRoute.SMALL_LOCAL_MODEL.value
    assert small.requires_provider is True
    assert small.halt_before_dispatch is False

    medium = route_model(
        _inputs(
            context_tokens=10_000,
            lowest_confidence=ConfidenceClass.HEURISTIC.value,
            risk=RiskClass.MEDIUM.value,
            dependency_cone_size=5,
            unresolved_obligations=0,
            available_proofs=0,
        )
    )
    assert medium.route == ModelRoute.MEDIUM_MODEL.value

    frontier = route_model(
        _inputs(
            context_tokens=50_000,
            lowest_confidence=ConfidenceClass.HEURISTIC.value,
            risk=RiskClass.MEDIUM.value,
            dependency_cone_size=40,
            unresolved_obligations=2,
            available_proofs=0,
        )
    )
    assert frontier.route == ModelRoute.FRONTIER_MODEL.value


def test_policy_thresholds_are_validated() -> None:
    with pytest.raises(HarnessError, match="nondecreasing"):
        ModelRoutingPolicy(
            small_context_tokens=10_000,
            medium_context_tokens=1_000,
        )
    with pytest.raises(HarnessError, match="nondecreasing"):
        ModelRoutingPolicy(
            small_dependency_cone=50,
            medium_dependency_cone=10,
            large_dependency_cone=5,
        )


def test_routing_inputs_from_context_pack() -> None:
    pack = ContextPack.from_dict(
        {
            "objective": "fix-one-symbol",
            "target_source_cid": _cid("target"),
            "surrounding_source_cid": _cid("surrounding"),
            "test_source_cid": _cid("test"),
            "dependency_capsule_cids": [_cid("cap-a")],
            "obligation_cids": [_cid("obl-a"), _cid("obl-b")],
            "counterexample_cids": [],
            "delta_cid": _cid("delta"),
            "interface_cids": [],
            "assumptions": ["heuristic-only"],
            "exclusions": [],
            "token_totals": {"total": 1500, "target": 900, "tests": 600},
            "estimator_version": "est@1",
            "risk": RiskClass.LOW.value,
            "route": ModelRoute.SMALL_LOCAL_MODEL.value,
            "escalation_recommendation": "none",
        }
    )
    inputs = RoutingInputs.from_context_pack(
        pack,
        lowest_confidence=ConfidenceClass.CONSERVATIVE.value,
        dependency_cone_size=4,
        available_proofs=2,
    )
    assert inputs.context_tokens == 1500
    assert inputs.unresolved_obligations == 2
    assert inputs.available_proofs == 2
    decision = route_model(inputs)
    assert decision.route == ModelRoute.DETERMINISTIC_ONLY.value


def test_closed_records_reject_unknown_fields() -> None:
    with pytest.raises(HarnessError):
        RoutingInputs.from_dict(
            {
                **_inputs().to_dict(),
                "secret_prompt": "nope",
            }
        )
    with pytest.raises(HarnessError):
        RoutingDecision.from_dict(
            {
                **route_model(_inputs()).to_dict(),
                "extra": 1,
            }
        )


def test_human_review_decision_cannot_require_provider_on_round_trip() -> None:
    decision = route_model(_inputs(risk=RiskClass.CRITICAL.value))
    payload = decision.to_dict()
    payload["requires_provider"] = True
    with pytest.raises(HarnessError, match="must not require a provider"):
        RoutingDecision.from_dict(payload)


def test_model_routing_descriptor_is_closed() -> None:
    descriptor = model_routing_descriptor()
    assert descriptor["interface"] == MODEL_ROUTING_INTERFACE
    assert set(descriptor["routes"]) == {item.value for item in ModelRoute}
    assert "route_decision_is_deterministic_and_explained" in descriptor["invariants"]
    # Deterministic JSON encoding for interface metadata.
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    assert "ModelRoutingPolicy" in encoded


def test_identical_inputs_across_policy_default_instances() -> None:
    a = route_model(_inputs(context_tokens=9_000, risk=RiskClass.MEDIUM.value))
    b = route_model(
        _inputs(context_tokens=9_000, risk=RiskClass.MEDIUM.value),
        policy=ModelRoutingPolicy.default(),
    )
    assert a.to_dict() == b.to_dict()
